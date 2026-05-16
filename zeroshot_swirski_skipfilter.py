import os
import cv2
import torch
import numpy as np
import csv
import re
import math
import types
from pathlib import Path
import torchvision.transforms.functional as TF
import argparse

# 🚨 GPU 1 강제 할당
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from networks.vit_seg_modeling import VisionTransformer
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg

# =====================================================================
WEIGHTS_PATH = "./models_transunet/best_model.pth"
BASE_DIR = Path("./Swirski_Dataset")
TABLE_DIR = Path("./Swirski_tables")
OVERLAY_DIR = Path("./Swirski_overlays_skipfilter")

IMG_SIZE = 224
NUM_CLASSES = 4
PUPIL_CLASS_ID = 3
# =====================================================================

def ritnet_preprocess(img_gray):
    """RITnet 4.3: Gamma(0.8) + CLAHE(grid=8, clip=1.5)"""
    # 1. Fixed Gamma Correction (exponent 0.8)
    normalized = img_gray.astype(np.float32) / 255.0
    gamma_corrected = np.power(normalized, 0.8)
    gamma_corrected = (gamma_corrected * 255).astype(np.uint8)
    # 2. Local CLAHE
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
    enhanced = clahe.apply(gamma_corrected)
    return enhanced

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

def load_ellipse_gt(txt_path):
    gt_dict = {}
    if not txt_path.exists():
        return gt_dict
    with open(txt_path, 'r') as f:
        for line in f:
            if '|' not in line:
                continue
            parts = line.split('|')
            if len(parts) != 2:
                continue
            try:
                frame_idx = int(parts[0].strip())
                params = parts[1].strip().split()
                if len(params) >= 5:
                    gt_dict[frame_idx] = {
                        'x': float(params[0]), 'y': float(params[1]),
                        'a': float(params[2]), 'b': float(params[3]),
                        'angle_rad': float(params[4])
                    }
            except ValueError:
                continue
    return gt_dict

def draw_gt_ellipse(h, w, gt_param):
    mask = np.zeros((h, w), dtype=np.uint8)
    if gt_param['a'] <= 0 or gt_param['b'] <= 0:
        return mask
    angle_deg = math.degrees(gt_param['angle_rad'])
    center = (int(gt_param['x']), int(gt_param['y']))
    axes = (int(gt_param['a']), int(gt_param['b']))
    cv2.ellipse(mask, center, axes, angle_deg, 0, 360, 255, -1)
    return mask

def create_segmentation_overlay(img, pred_mask):
    img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    overlay = img_bgr.copy()
    overlay[pred_mask == PUPIL_CLASS_ID] = [0, 0, 255]
    alpha = 0.5
    cv2.addWeighted(overlay, alpha, img_bgr, 1 - alpha, 0, img_bgr)
    return img_bgr

def extract_frame_idx(filename):
    match = re.search(r'(\d+)-eye\.png', filename)
    return int(match.group(1)) if match else -1

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
        return pred_mask
        
    return result

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

def get_transunet_model(device, sigma, sigma2=0.0):
    config_vit = CONFIGS_ViT_seg['R50-ViT-B_16']
    config_vit.n_classes = NUM_CLASSES
    config_vit.n_skip = 3
    if config_vit.patches.get('grid') is not None:
        config_vit.patches.grid = (int(IMG_SIZE / 16), int(IMG_SIZE / 16))

    model = VisionTransformer(config_vit, img_size=IMG_SIZE, num_classes=NUM_CLASSES)
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
    
    # 런타임 몽키패치
    model.decoder.filter_sigma = sigma
    model.decoder.filter_sigma2 = sigma2
    model.decoder.forward = types.MethodType(patched_decoder_forward, model.decoder)
    
    model.to(device)
    model.eval()
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sigma', type=float, default=0.0, help='Sigma for Gaussian Blur on Skip 0 & 1')
    parser.add_argument('--sigma2', type=float, default=0.0, help='Sigma for Gaussian Blur on Skip 2 (112x112)')
    parser.add_argument('--ellipse', action='store_true', help='Enable Ellipse fitting post-processing')
    parser.add_argument('--preprocess', action='store_true', help='Apply RITnet preprocessing (Gamma 0.8 + CLAHE)')
    args = parser.parse_args()

    # 결과물 구분을 위한 네이밍 (최적화 파라미터 반영 태그 추가)
    suffix = f"sig{args.sigma}_s2{args.sigma2}_ell{'O' if args.ellipse else 'X'}_pre{'O' if args.preprocess else 'X'}_opt0.10"
    csv_path = TABLE_DIR / f"transunet_swirski_skipfilter_{suffix}.csv"
    cur_overlay_dir = OVERLAY_DIR / suffix
    
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    cur_overlay_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"TransUNet 가동 (sigma={args.sigma}, ellipse={args.ellipse})")

    model = get_transunet_model(device, args.sigma, args.sigma2)
    case_dirs = [d for d in BASE_DIR.iterdir() if d.is_dir() and 'p' in d.name]

    with open(csv_path, mode='w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['Case', 'Frame_Idx', 'IoU', 'Dice'])
        total_iou, total_dice = [], []

        with torch.no_grad():
            for case_dir in sorted(case_dirs):
                case_name = case_dir.name
                frames_dir = case_dir / "frames"
                gt_path = case_dir / "pupil-ellipses.txt"

                if not frames_dir.exists() or not gt_path.exists():
                    continue

                case_overlay_dir = cur_overlay_dir / case_name
                case_overlay_dir.mkdir(parents=True, exist_ok=True)

                gt_dict = load_ellipse_gt(gt_path)
                frame_files = [f for f in frames_dir.iterdir() if f.name.endswith('-eye.png')]
                frame_files.sort(key=lambda x: extract_frame_idx(x.name))

                case_iou, case_dice = [], []
                for frame_file in frame_files:
                    frame_idx = extract_frame_idx(frame_file.name)
                    if frame_idx not in gt_dict:
                        continue

                    img_raw = cv2.imread(str(frame_file), cv2.IMREAD_GRAYSCALE)
                    if img_raw is None:
                        continue
                    
                    h, w = img_raw.shape
                    img_for_model = img_raw
                    if args.preprocess:
                        img_for_model = ritnet_preprocess(img_raw)
                    rgb = cv2.cvtColor(img_for_model, cv2.COLOR_GRAY2RGB)
                    resized = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE))
                    
                    tensor = torch.from_numpy(resized).float().permute(2,0,1).unsqueeze(0) / 255.0
                    tensor = tensor.to(device)

                    logits = model(tensor)
                    pred = logits.argmax(1).squeeze(0).cpu().numpy().astype(np.uint8)
                    
                    if args.ellipse:
                        pred = ellipse_postprocess(pred, PUPIL_CLASS_ID)
                    
                    pred_resized = cv2.resize(pred, (w, h), interpolation=cv2.INTER_NEAREST)

                    gt_param = gt_dict[frame_idx]
                    gt_mask = draw_gt_ellipse(h, w, gt_param)

                    iou, dice = calc_metrics(pred_resized, gt_mask, PUPIL_CLASS_ID)
                    case_iou.append(iou)
                    case_dice.append(dice)
                    csv_writer.writerow([case_name, frame_idx, f"{iou:.4f}", f"{dice:.4f}"])

                    if frame_idx % 10 == 0:
                        overlay = create_segmentation_overlay(img_raw, pred_resized)
                        cv2.imwrite(str(case_overlay_dir / f"overlay_{frame_idx:04d}.png"), overlay)

                if case_iou:
                    mIoU = sum(case_iou) / len(case_iou)
                    mDice = sum(case_dice) / len(case_dice)
                    print(f"  [{case_name}] mIoU: {mIoU:.4f} | mDice: {mDice:.4f}")
                    total_iou.extend(case_iou)
                    total_dice.extend(case_dice)

        if total_iou:
            t_mIoU = sum(total_iou) / len(total_iou)
            t_mDice = sum(total_dice) / len(total_dice)
            print(f"\n[Total] mIoU: {t_mIoU:.4f} | mDice: {t_mDice:.4f}")

if __name__ == '__main__':
    main()
