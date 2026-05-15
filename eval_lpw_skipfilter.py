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

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from networks.vit_seg_modeling import VisionTransformer
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg

WEIGHTS_PATH = "./models_transunet/best_model.pth"
RAW_BASE_DIR = Path("./LPW")
GT_BASE_DIR = Path("./Pupils_in_the_wild_improved")
TABLE_DIR = Path("./LPW_tables")
OVERLAY_DIR = Path("./LPW_overlays_skipfilter")

IMG_SIZE = 224
NUM_CLASSES = 4
PUPIL_CLASS_ID = 3

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
    sigma01 = getattr(self, 'filter_sigma', 0.0)
    sigma2 = getattr(self, 'filter_sigma2', 0.0)
    if features is not None:
        features = list(features)
        if sigma01 > 0:
            k = _make_blur(sigma01)
            if len(features) > 0:
                features[0] = TF.gaussian_blur(features[0], kernel_size=[k, k], sigma=[sigma01, sigma01])
            if len(features) > 1:
                features[1] = TF.gaussian_blur(features[1], kernel_size=[k, k], sigma=[sigma01, sigma01])
        if sigma2 > 0 and len(features) > 2:
            k2 = _make_blur(sigma2)
            features[2] = TF.gaussian_blur(features[2], kernel_size=[k2, k2], sigma=[sigma2, sigma2])

    B, n_patch, hidden = hidden_states.size()
    h, w = int(np.sqrt(n_patch)), int(np.sqrt(n_patch))
    x = hidden_states.permute(0, 2, 1)
    x = x.contiguous().view(B, hidden, h, w)
    x = self.conv_more(x)
    for i, decoder_block in enumerate(self.blocks):
        skip = features[i] if (features is not None and i < self.config.n_skip) else None
        x = decoder_block(x, skip=skip)
    return x

def get_model(device, sigma, sigma2=0.0):
    config_vit = CONFIGS_ViT_seg['R50-ViT-B_16']
    config_vit.n_classes = NUM_CLASSES
    config_vit.n_skip = 3
    if config_vit.patches.get('grid') is not None:
        config_vit.patches.grid = (int(IMG_SIZE / 16), int(IMG_SIZE / 16))
    model = VisionTransformer(config_vit, img_size=IMG_SIZE, num_classes=NUM_CLASSES)
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
    model.decoder.filter_sigma = sigma
    model.decoder.filter_sigma2 = sigma2
    model.decoder.forward = types.MethodType(patched_decoder_forward, model.decoder)
    model.to(device)
    model.eval()
    return model

def ellipse_postprocess(pred_mask, pupil_id=3, min_points=5):
    binary = (pred_mask == pupil_id).astype(np.uint8)
    
    # 1. 형태학적 닫기 (속눈썹으로 인한 얇은 분절 연결)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
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
        
        # 조건 1: 가장 큰 조각 대비 면적이 5% 미만이면 노이즈
        if area < largest_area * 0.05:
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
    result = pred_mask.copy()
    result[result == pupil_id] = 0
    cv2.ellipse(result, ellipse, pupil_id, -1)
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
    parser.add_argument('--sigma', type=float, default=0.0)
    parser.add_argument('--sigma2', type=float, default=0.0)
    parser.add_argument('--ellipse', action='store_true')
    parser.add_argument('--preprocess', action='store_true')
    parser.add_argument('--folder', type=int, default=1, help='LPW folder number to evaluate')
    args = parser.parse_args()

    suffix = f"f{args.folder}_sig{args.sigma}_s2{args.sigma2}_ell{'O' if args.ellipse else 'X'}_pre{'O' if args.preprocess else 'X'}"
    csv_path = TABLE_DIR / f"lpw_skipfilter_{suffix}.csv"
    overlay_dir = OVERLAY_DIR / suffix

    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    overlay_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"LPW Folder {args.folder} 평가 (sigma={args.sigma}, sigma2={args.sigma2}, ellipse={args.ellipse}, preprocess={args.preprocess})")

    model = get_model(device, args.sigma, args.sigma2)
    gt_mapping = build_gt_mapping(GT_BASE_DIR, target_folder=args.folder)
    print(f"GT 비디오: {len(gt_mapping)}개")

    folder_dir = RAW_BASE_DIR / str(args.folder)
    raw_videos = list(folder_dir.glob("1.avi")) if folder_dir.exists() else []
    print(f"원본 비디오: {len(raw_videos)}개")

    with open(csv_path, mode='w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['Video', 'Frame', 'IoU', 'Dice'])
        total_iou, total_dice = [], []

        with torch.no_grad():
            for raw_path in sorted(raw_videos):
                file_idx = int(raw_path.stem)
                key = f"{args.folder}_{file_idx}"
                if key not in gt_mapping:
                    print(f"  GT 없음: {raw_path.name}")
                    continue
                gt_path = gt_mapping[key]
                vid_overlay = overlay_dir / raw_path.stem
                vid_overlay.mkdir(parents=True, exist_ok=True)

                cap_raw = cv2.VideoCapture(str(raw_path))
                cap_gt = cv2.VideoCapture(str(gt_path))
                
                # 비디오 속성 가져오기
                fps = cap_raw.get(cv2.CAP_PROP_FPS)
                if fps == 0 or np.isnan(fps): fps = 30.0
                width  = int(cap_raw.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap_raw.get(cv2.CAP_PROP_FRAME_HEIGHT))
                
                # VideoWriter 초기화 (XVID 코덱 사용, .avi 포맷)
                out_vid_path = overlay_dir / f"{raw_path.stem}.avi"
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

                    img_for_model = gray
                    if args.preprocess:
                        img_for_model = ritnet_preprocess(gray)

                    rgb = cv2.cvtColor(img_for_model, cv2.COLOR_GRAY2RGB)
                    resized = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE))
                    tensor = torch.from_numpy(resized).float().permute(2, 0, 1).unsqueeze(0) / 255.0
                    tensor = tensor.to(device)

                    logits = model(tensor)
                    pred = logits.argmax(1).squeeze(0).cpu().numpy().astype(np.uint8)

                    if args.ellipse:
                        pred = ellipse_postprocess(pred, PUPIL_CLASS_ID)

                    pred_full = cv2.resize(pred, (w, h), interpolation=cv2.INTER_NEAREST)

                    gray_gt = cv2.cvtColor(frame_g, cv2.COLOR_BGR2GRAY) if len(frame_g.shape) == 3 else frame_g
                    gray_gt = cv2.resize(gray_gt, (w, h), interpolation=cv2.INTER_NEAREST)
                    _, gt_bin = cv2.threshold(gray_gt, 127, 255, cv2.THRESH_BINARY)

                    iou, dice = calc_metrics(pred_full, gt_bin, PUPIL_CLASS_ID)
                    vid_iou.append(iou)
                    vid_dice.append(dice)
                    csv_writer.writerow([raw_path.stem, frame_idx, f"{iou:.4f}", f"{dice:.4f}"])

                    # 결과 프레임을 비디오에 쓰기
                    img_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
                    ov = img_bgr.copy()
                    ov[pred_full == PUPIL_CLASS_ID] = [0, 0, 255]
                    cv2.addWeighted(ov, 0.5, img_bgr, 0.5, 0, img_bgr)
                    out_vid.write(img_bgr)

                    frame_idx += 1

                cap_raw.release()
                cap_gt.release()
                out_vid.release()

                if vid_iou:
                    m = np.mean(vid_iou)
                    print(f"  [{raw_path.stem}.avi] mIoU={m:.4f} mDice={np.mean(vid_dice):.4f} ({len(vid_iou)} frames)")
                    total_iou.extend(vid_iou)
                    total_dice.extend(vid_dice)

        if total_iou:
            print(f"\n[Folder {args.folder} Total] mIoU={np.mean(total_iou):.4f} mDice={np.mean(total_dice):.4f}")

if __name__ == '__main__':
    main()
