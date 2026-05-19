"""
LPW folder-1 전용 Skip Filter 평가 스크립트.
zeroshot_eval_LPW.py를 기반으로 Skip Filter + RITnet 전처리를 추가.
"""
import os
import cv2
import torch
import numpy as np
import csv
import re
import types
from pathlib import Path
import torchvision.transforms.functional as TF
import argparse

from networks.vit_seg_modeling import VisionTransformer
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg

WEIGHTS_PATH = "./models_transunet/best_model.pth"
RAW_BASE_DIR = Path("./LPW")
GT_BASE_DIR = Path("./Pupils_in_the_wild_improved")
TABLE_DIR = Path("./final_v2/LPW_tables")
OVERLAY_DIR = Path("./final_v2/LPW_overlays")

IMG_SIZE = 224
NUM_CLASSES = 4
PUPIL_CLASS_ID = 3

# ── NEW: Pure FFT Gaussian Low-Pass Filter ──────────────────────────────────
def ue_fft_pure_lowpass(img_gray, cutoff_radius=50):
    """
    이미지 전체에 대해 FFT를 수행하고 고주파(글린트)를 억제하는 가우시안 LPF 적용.
    공간 마스크 일절 없음.
    """
    H, W = img_gray.shape
    f = np.fft.fft2(img_gray.astype(np.float32))
    fshift = np.fft.fftshift(f)
    
    cy, cx = H//2, W//2
    yy, xx = np.ogrid[:H, :W]
    
    # 가우시안 저주파 필터 생성
    dist = (yy - cy)**2 + (xx - cx)**2
    # cutoff_radius가 작을수록 더 많이 뭉개짐
    kernel = np.exp(-dist / (2 * (cutoff_radius**2)))
    
    fshift_filtered = fshift * kernel
    img_back = np.abs(np.fft.ifft2(np.fft.ifftshift(fshift_filtered)))
    return np.clip(img_back, 0, 255).astype(np.uint8)

def ue_zoom_out(img_gray, scale=0.65):
    H, W = img_gray.shape
    nH, nW = int(H*scale), int(W*scale)
    resized = cv2.resize(img_gray, (nW, nH), interpolation=cv2.INTER_AREA)
    bg = int(np.median(img_gray))
    canvas = np.full((H, W), bg, dtype=np.uint8)
    y0=(H-nH)//2; x0=(W-nW)//2
    canvas[y0:y0+nH, x0:x0+nW] = resized
    return canvas, y0, x0, nH, nW


def ritnet_preprocess(img_gray):
    normalized = img_gray.astype(np.float32) / 255.0
    gamma_corrected = np.power(normalized, 0.8)
    gamma_corrected = (gamma_corrected * 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
    return clahe.apply(gamma_corrected)

def calc_metrics(pred_mask, gt_mask, class_id):
    pred_bin = (pred_mask == class_id)
    gt_bin = (gt_mask == 255)
    intersection = np.logical_and(pred_bin, gt_bin).sum()
    union = np.logical_or(pred_bin, gt_bin).sum()
    if union == 0:
        iou = 1.0 if np.sum(pred_bin) == 0 else 0.0
        dice = 1.0 if np.sum(pred_bin) == 0 else 0.0
    else:
        iou = intersection / union
        dice = 2 * intersection / (pred_bin.sum() + gt_bin.sum())
    return iou, dice

def _make_blur(sigma):
    k_size = int(4 * sigma + 0.5)
    if k_size % 2 == 0: k_size += 1
    if k_size < 3: k_size = 3
    return k_size

def patched_decoder_forward(self, hidden_states, features=None):
    sigma0 = getattr(self, 'filter_sigma0', 1.0)
    sigma1 = getattr(self, 'filter_sigma1', 0.5)
    if features is not None:
        features = list(features)
        if sigma0 > 0 and len(features) > 0:
            k0 = _make_blur(sigma0)
            features[0] = TF.gaussian_blur(features[0], kernel_size=[k0, k0], sigma=[sigma0, sigma0])
        if sigma1 > 0 and len(features) > 1:
            k1 = _make_blur(sigma1)
            features[1] = TF.gaussian_blur(features[1], kernel_size=[k1, k1], sigma=[sigma1, sigma1])

    B, n_patch, hidden = hidden_states.size()
    h, w = int(np.sqrt(n_patch)), int(np.sqrt(n_patch))
    x = hidden_states.permute(0, 2, 1)
    x = x.contiguous().view(B, hidden, h, w)
    x = self.conv_more(x)
    for i, decoder_block in enumerate(self.blocks):
        skip = features[i] if (features is not None and i < self.config.n_skip) else None
        x = decoder_block(x, skip=skip)
    return x

def get_model(device, sigma0=1.0, sigma1=0.5):
    config_vit = CONFIGS_ViT_seg['R50-ViT-B_16']
    config_vit.n_classes = NUM_CLASSES
    config_vit.n_skip = 3
    if config_vit.patches.get('grid') is not None:
        config_vit.patches.grid = (int(IMG_SIZE / 16), int(IMG_SIZE / 16))
    model = VisionTransformer(config_vit, img_size=IMG_SIZE, num_classes=NUM_CLASSES)
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
    model.decoder.filter_sigma0 = sigma0
    model.decoder.filter_sigma1 = sigma1
    model.decoder.forward = types.MethodType(patched_decoder_forward, model.decoder)
    model.to(device)
    model.eval()
    return model

def ellipse_postprocess(pred_mask, pupil_id=3, min_points=5):
    binary = (pred_mask == pupil_id).astype(np.uint8)
    
    # 1. 형태학적 닫기 (속눈썹으로 인한 얇은 분절 연결) - 최적 커널 크기 13 적용
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return pred_mask
        
    # 2. 가장 큰 조각 찾기
    largest = max(contours, key=cv2.contourArea)
    largest_area = cv2.contourArea(largest)
    if largest_area == 0 or len(largest) < min_points:
        return pred_mask
        
    M_large = cv2.moments(largest)
    cx_large = M_large["m10"] / M_large["m00"] if M_large["m00"] != 0 else 0
    cy_large = M_large["m01"] / M_large["m00"] if M_large["m00"] != 0 else 0
    
    valid_points = [largest]
    for cnt in contours:
        if cnt is largest:
            continue
        area = cv2.contourArea(cnt)
        
        # 조건 1: 가장 큰 조각 대비 면적이 10% 미만이면 노이즈 (Grid Search 최적화 결과 반영)
        if area < largest_area * 0.10:
            continue
            
        # 조건 2: 거리가 너무 멀면 노이즈 (224 이미지 기준 반경 50픽셀 이내만 허용)
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cx = M["m10"] / M["m00"]
            cy = M["m01"] / M["m00"]
            dist = np.sqrt((cx - cx_large)**2 + (cy - cy_large)**2)
            if dist > 50:
                continue
                
        valid_points.append(cnt)
        
    all_points = np.vstack(valid_points)
    if len(all_points) < min_points:
        return pred_mask
        
    ellipse = cv2.fitEllipse(all_points)
    
    # 타원의 크기가 비정상(음수나 0)인 경우 기하학적 오류 방지
    if ellipse[1][0] <= 0 or ellipse[1][1] <= 0:
        return pred_mask
        
    result = pred_mask.copy()
    result[result == pupil_id] = 0
    try:
        cv2.ellipse(result, ellipse, pupil_id, -1)
    except cv2.error as e:
        # 드문 확률로 collinear points 등에서 발생하는 내부 에러 방어
        return pred_mask
        
    return result

def build_gt_mapping(gt_dir, target_folder=None):
    mapping = {}
    for f in gt_dir.rglob("*_pupil.mp4"):
        match = re.search(r'folder-(\d+)_file-(\d+)', f.name)
        if match:
            folder_idx = int(match.group(1))
            file_idx = int(match.group(2))
            if target_folder is not None and folder_idx != target_folder:
                continue
            mapping[f"{folder_idx}_{file_idx}"] = f
    return mapping

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sigma0', type=float, default=1.0, help='Sigma for Skip 0')
    parser.add_argument('--sigma1', type=float, default=0.5, help='Sigma for Skip 1')
    parser.add_argument('--ellipse', action='store_true')
    parser.add_argument('--preprocess', action='store_true')
    parser.add_argument('--folder', type=int, default=1, help='LPW folder number to evaluate')
    parser.add_argument('--all_folders', action='store_true', help='Evaluate all folders (1-22)')
    parser.add_argument('--f_start', type=int, default=None)
    parser.add_argument('--f_end', type=int, default=None)
    parser.add_argument('--device', type=str, default='cuda:0', help='cuda device to use')
    parser.add_argument('--dry_run', action='store_true', help='Dry run to test 10 frames only')
    parser.add_argument('--ue_fft', action='store_true', help='Apply UE FFT pure LPF + zoom out')
    args = parser.parse_args()

    if args.f_start is not None and args.f_end is not None:
        folder_list = list(range(args.f_start, args.f_end + 1))
        folder_str = f"f{args.f_start}_to_f{args.f_end}"
    else:
        folder_list = list(range(1, 23)) if args.all_folders else [args.folder]
        folder_str = f"f{'ALL' if args.all_folders else args.folder}"
        
    suffix = f"{folder_str}_s0_{args.sigma0}_s1_{args.sigma1}_ell{'O' if args.ellipse else 'X'}_pre{'O' if args.preprocess else 'X'}_fft{'O' if args.ue_fft else 'X'}"
    
    frame_csv_path = TABLE_DIR / f"lpw_skipfilter_{suffix}_frames.csv"
    compact_csv_path = TABLE_DIR / f"lpw_skipfilter_{suffix}_compact.csv"
    overlay_dir = OVERLAY_DIR / suffix

    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    overlay_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"LPW 평가 시작 (Folders: {folder_list}, s0={args.sigma0}, s1={args.sigma1}, ellipse={args.ellipse})")

    model = get_model(device, args.sigma0, args.sigma1)
    
    # GT는 전체 폴더에 대해 매핑 (target_folder 지정 안함)
    gt_mapping = build_gt_mapping(GT_BASE_DIR, target_folder=None)
    print(f"로드된 GT 비디오 총 개수: {len(gt_mapping)}개")

    total_iou_all, total_dice_all = [], []
    video_summaries = []

    with open(frame_csv_path, mode='w', newline='') as f_csv, open(compact_csv_path, mode='w', newline='') as c_csv:
        frame_writer = csv.writer(f_csv)
        frame_writer.writerow(['Folder', 'Video', 'Frame', 'IoU', 'Dice'])
        
        compact_writer = csv.writer(c_csv)
        compact_writer.writerow(['Folder', 'Video', 'mIoU', 'mDice'])

        with torch.no_grad():
            for folder_idx in folder_list:
                folder_dir = RAW_BASE_DIR / str(folder_idx)
                raw_videos = list(folder_dir.glob("*.avi")) if folder_dir.exists() else []
                
                if not raw_videos:
                    print(f"[{folder_idx}] 원본 비디오를 찾을 수 없습니다. 건너뜁니다.")
                    continue
                    
                folder_iou, folder_dice = [], []
                
                for raw_path in sorted(raw_videos):
                    if raw_path.name.startswith("._"):
                        continue
                    try:
                        file_idx = int(raw_path.stem)
                    except ValueError:
                        print(f"  [Folder {folder_idx}] 비정상적인 파일 이름 건너뜀: {raw_path.name}")
                        continue
                        
                    key = f"{folder_idx}_{file_idx}"
                    
                    if key not in gt_mapping:
                        print(f"  [Folder {folder_idx}] GT 없음 건너뜀: {raw_path.name}")
                        continue
                        
                    gt_path = gt_mapping[key]
                    
                    vid_overlay_dir = overlay_dir / str(folder_idx)
                    vid_overlay_dir.mkdir(parents=True, exist_ok=True)

                    cap_raw = cv2.VideoCapture(str(raw_path))
                    cap_gt = cv2.VideoCapture(str(gt_path))
                    
                    fps = cap_raw.get(cv2.CAP_PROP_FPS)
                    if fps == 0 or np.isnan(fps): fps = 30.0
                    width  = int(cap_raw.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap_raw.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    
                    out_vid_path = vid_overlay_dir / f"{raw_path.stem}.avi"
                    fourcc = cv2.VideoWriter_fourcc(*'XVID')
                    out_vid = cv2.VideoWriter(str(out_vid_path), fourcc, fps, (width, height))

                    frame_idx = 0
                    vid_iou, vid_dice = [], []

                    while True:
                        ret_r, frame_r = cap_raw.read()
                        ret_g, frame_g = cap_gt.read()
                        if not (ret_r and ret_g):
                            break

                        h, w = frame_r.shape[:2]
                        gray = cv2.cvtColor(frame_r, cv2.COLOR_BGR2GRAY) if len(frame_r.shape) == 3 else frame_r

                        img_for_model = gray.copy()
                        if args.preprocess:
                            img_for_model = ritnet_preprocess(img_for_model)

                        is_ue = False
                        y0, x0, nH, nW = 0, 0, h, w
                        
                        if args.ue_fft and gray.mean() < 110:
                            is_ue = True
                            img_for_model = ue_fft_pure_lowpass(img_for_model, cutoff_radius=50)
                            img_for_model, y0, x0, nH, nW = ue_zoom_out(img_for_model, scale=0.65)

                        rgb = cv2.cvtColor(img_for_model, cv2.COLOR_GRAY2RGB)
                        resized = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE))
                        tensor = torch.from_numpy(resized).float().permute(2, 0, 1).unsqueeze(0) / 255.0
                        tensor = tensor.to(device)

                        logits = model(tensor)
                        pred = logits.argmax(1).squeeze(0).cpu().numpy().astype(np.uint8)

                        if args.ellipse:
                            pred = ellipse_postprocess(pred, PUPIL_CLASS_ID)

                        if is_ue and args.ue_fft:
                            pred_canvas = cv2.resize(pred, (w, h), interpolation=cv2.INTER_NEAREST)
                            pred_region = pred_canvas[y0:y0+nH, x0:x0+nW]
                            pred_full = cv2.resize(pred_region, (w, h), interpolation=cv2.INTER_NEAREST)
                        else:
                            pred_full = cv2.resize(pred, (w, h), interpolation=cv2.INTER_NEAREST)

                        gray_gt = cv2.cvtColor(frame_g, cv2.COLOR_BGR2GRAY) if len(frame_g.shape) == 3 else frame_g
                        gray_gt = cv2.resize(gray_gt, (w, h), interpolation=cv2.INTER_NEAREST)
                        _, gt_bin = cv2.threshold(gray_gt, 127, 255, cv2.THRESH_BINARY)

                        iou, dice = calc_metrics(pred_full, gt_bin, PUPIL_CLASS_ID)
                        vid_iou.append(iou)
                        vid_dice.append(dice)
                        frame_writer.writerow([folder_idx, raw_path.stem, frame_idx, f"{iou:.4f}", f"{dice:.4f}"])

                        img_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
                        ov = img_bgr.copy()
                        ov[pred_full == PUPIL_CLASS_ID] = [0, 0, 255]
                        cv2.addWeighted(ov, 0.5, img_bgr, 0.5, 0, img_bgr)
                        out_vid.write(img_bgr)
                        
                        # PNG 프레임 저장 (추후 논문 삽입용)
                        png_dir = vid_overlay_dir / f"{raw_path.stem}_frames"
                        png_dir.mkdir(parents=True, exist_ok=True)
                        cv2.imwrite(str(png_dir / f"frame_{frame_idx:04d}.png"), img_bgr)

                        frame_idx += 1
                        
                        if args.dry_run and frame_idx >= 10:
                            break

                    cap_raw.release()
                    cap_gt.release()
                    out_vid.release()
                    
                    if args.dry_run:
                        print("Dry run completed successfully.")
                        return

                    if vid_iou:
                        m_iou = np.mean(vid_iou)
                        m_dice = np.mean(vid_dice)
                        print(f"  [Folder {folder_idx} | {raw_path.stem}.avi] mIoU={m_iou:.4f} mDice={m_dice:.4f} ({len(vid_iou)} frames)")
                        folder_iou.extend(vid_iou)
                        folder_dice.extend(vid_dice)
                        total_iou_all.extend(vid_iou)
                        total_dice_all.extend(vid_dice)
                        compact_writer.writerow([folder_idx, raw_path.stem, f"{m_iou:.4f}", f"{m_dice:.4f}"])
                        
                if folder_iou:
                    fm_iou = np.mean(folder_iou)
                    fm_dice = np.mean(folder_dice)
                    print(f"[Folder {folder_idx} Total] mIoU={fm_iou:.4f} mDice={fm_dice:.4f}")
                    compact_writer.writerow([folder_idx, 'FOLDER_TOTAL', f"{fm_iou:.4f}", f"{fm_dice:.4f}"])
                    
        if total_iou_all:
            tm_iou = np.mean(total_iou_all)
            tm_dice = np.mean(total_dice_all)
            print(f"\n[All Folders Total] mIoU={tm_iou:.4f} mDice={tm_dice:.4f}")
            compact_writer.writerow(['ALL', 'TOTAL', f"{tm_iou:.4f}", f"{tm_dice:.4f}"])

if __name__ == '__main__':
    main()
