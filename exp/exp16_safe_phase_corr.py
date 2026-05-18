import os
import cv2
import torch
import torch.nn.functional as F
import numpy as np
import math
import types
from pathlib import Path
import sys
import argparse

import torchvision.transforms.functional as TF
from networks.vit_seg_modeling import VisionTransformer
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg

WEIGHTS_PATH = "./models_transunet/best_model.pth"
BASE_DIR = Path("./Swirski_Dataset")
IMG_SIZE = 224
NUM_CLASSES = 4
PUPIL_CLASS_ID = 3

def calc_metrics(pred_mask, gt_mask, class_id):
    pred_bin = (pred_mask == class_id)
    gt_bin = (gt_mask == 255)
    intersection = np.logical_and(pred_bin, gt_bin).sum()
    union = np.logical_or(pred_bin, gt_bin).sum()
    if union == 0:
        return 1.0 if np.sum(pred_bin) == 0 else 0.0
    return intersection / union

def load_ellipse_gt(txt_path):
    gt_dict = {}
    if not txt_path.exists(): return gt_dict
    with open(txt_path, 'r') as f:
        for line in f:
            if '|' not in line: continue
            parts = line.split('|')
            if len(parts) != 2: continue
            try:
                frame_idx = int(parts[0].strip())
                params = parts[1].strip().split()
                if len(params) >= 5:
                    gt_dict[frame_idx] = {
                        'x': float(params[0]), 'y': float(params[1]),
                        'a': float(params[2]), 'b': float(params[3]),
                        'angle_rad': float(params[4])
                    }
            except ValueError: continue
    return gt_dict

def draw_gt_ellipse(h, w, gt_param):
    mask = np.zeros((h, w), dtype=np.uint8)
    if gt_param['a'] <= 0 or gt_param['b'] <= 0: return mask
    angle_deg = math.degrees(gt_param['angle_rad'])
    center = (int(gt_param['x']), int(gt_param['y']))
    axes = (int(gt_param['a']), int(gt_param['b']))
    cv2.ellipse(mask, center, axes, angle_deg, 0, 360, 255, -1)
    return mask

def ellipse_postprocess(pred_mask, pupil_id=3, min_points=5):
    binary = (pred_mask == pupil_id).astype(np.uint8)
    
    # 1. 형태학적 닫기 (속눈썹으로 인한 얇은 분절 연결) - 최적 커널 크기 13 적용 (224x224 해상도 기준)
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
        return pred_mask
        
    return result

def patched_vit_forward_safe_phase_corr(self, x):
    # Detect over-exposure: if more than 8% of pixels are > 0.94 (240/255)
    high_ratio = (x > 0.94).float().mean()
    self.decoder.is_overexposed = (high_ratio.item() > 0.08)
    
    if x.size()[1] == 1:
        x = x.repeat(1,3,1,1)
    x, attn_weights, features = self.transformer(x)
    x = self.decoder(x, features)
    logits = self.segmentation_head(x)
    return logits

def patched_decoder_forward_safe_phase_corr(self, hidden_states, features=None):
    sigma0, sigma1 = 1.0, 0.5
    is_overexposed = getattr(self, 'is_overexposed', False)
    alpha = getattr(self, 'alpha', 0.1)

    if features is not None:
        features = list(features)
        
        # 1. Apply standard Skip Filter (Contribution 1)
        if len(features) > 0:
            k0 = int(4*sigma0+0.5); k0 = k0+1 if k0%2==0 else k0
            features[0] = TF.gaussian_blur(features[0], [k0,k0], [sigma0,sigma0])
        if len(features) > 1:
            k1 = int(4*sigma1+0.5); k1 = k1+1 if k1%2==0 else k1
            features[1] = TF.gaussian_blur(features[1], [k1,k1], [sigma1,sigma1])

        # 2. Apply Safe Phase Correction ONLY if overexposed
        if is_overexposed and len(features) > 0 and alpha > 0:
            skip = features[0]
            skip_2d = skip.mean(dim=1)
            
            # Extract ViT 2D output
            B, n_patch, hidden = hidden_states.size()
            h = w = int(n_patch**0.5)
            x_vit = hidden_states.permute(0, 2, 1).contiguous().view(B, hidden, h, w)
            x_vit = self.conv_more(x_vit)
            vit_2d = x_vit.mean(dim=1)
            
            vit_up = F.interpolate(
                vit_2d.unsqueeze(1), size=skip_2d.shape[1:],
                mode='bilinear', align_corners=False
            ).squeeze(1)
            
            skip_fft = torch.fft.fft2(skip_2d)
            vit_fft = torch.fft.fft2(vit_up)
            
            skip_amp = torch.abs(skip_fft)
            skip_phase = torch.angle(skip_fft)
            vit_phase = torch.angle(vit_fft)
            
            corrected_phase = (1 - alpha) * skip_phase + alpha * vit_phase
            
            corrected_2d = torch.fft.ifft2(skip_amp * torch.exp(1j * corrected_phase)).real
            
            ratio = corrected_2d / (skip_2d + 1e-5)
            ratio = torch.clamp(ratio, min=0.9, max=1.1)  # Strict clamping to avoid explosion!
            
            features[0] = skip * ratio.unsqueeze(1)

    B, n_patch, hidden = hidden_states.size()
    h = w = int(n_patch**0.5)
    x = hidden_states.permute(0,2,1).contiguous().view(B, hidden, h, w)
    x = self.conv_more(x)
    for i, block in enumerate(self.blocks):
        skip = features[i] if (features is not None and i < self.config.n_skip) else None
        x = block(x, skip=skip)
    return x

def get_model(device, alpha):
    config_vit = CONFIGS_ViT_seg['R50-ViT-B_16']
    config_vit.n_classes = NUM_CLASSES
    config_vit.n_skip = 3
    if config_vit.patches.get('grid') is not None:
        config_vit.patches.grid = (int(IMG_SIZE / 16), int(IMG_SIZE / 16))

    model = VisionTransformer(config_vit, img_size=IMG_SIZE, num_classes=NUM_CLASSES, vis=False)
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
    
    model.decoder.alpha = alpha
    model.forward = types.MethodType(patched_vit_forward_safe_phase_corr, model)
    model.decoder.forward = types.MethodType(patched_decoder_forward_safe_phase_corr, model.decoder)
    
    model.to(device)
    model.eval()
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--alpha', type=float, default=0.1, help='Phase correction alpha')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"========================================")
    print(f"🚀 Safe Phase Correction (alpha={args.alpha}) on {device}")
    print(f"========================================")
    sys.stdout.flush()

    model = get_model(device, args.alpha)
    case_dirs = sorted([d for d in BASE_DIR.iterdir() if d.is_dir() and 'p' in d.name])

    total_iou = []
    
    for case_dir in case_dirs:
        case_name = case_dir.name
        frames_dir = case_dir / "frames"
        gt_path = case_dir / "pupil-ellipses.txt"
        
        if not frames_dir.exists() or not gt_path.exists(): continue
        gt_dict = load_ellipse_gt(gt_path)
        
        frame_files = sorted([f for f in frames_dir.iterdir() if f.name.endswith('-eye.png')], 
                             key=lambda x: int(x.name.split('-')[0]))
        case_iou = []
        
        print(f"\n📁 Processing Case: {case_name}")
        sys.stdout.flush()
        
        for i, frame_file in enumerate(frame_files):
            frame_idx = int(frame_file.name.split('-')[0])
            if frame_idx not in gt_dict: continue

            img_raw = cv2.imread(str(frame_file), cv2.IMREAD_GRAYSCALE)
            if img_raw is None: continue
            
            h, w = img_raw.shape
            rgb = cv2.cvtColor(img_raw, cv2.COLOR_GRAY2RGB)
            resized = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE))
            tensor = torch.from_numpy(resized).float().permute(2,0,1).unsqueeze(0) / 255.0
            tensor = tensor.to(device)

            with torch.no_grad():
                logits = model(tensor)
                
            pred = logits.argmax(1).squeeze(0).cpu().numpy().astype(np.uint8)
            
            # Apply ellipse fitting at 224x224 resolution (EXACT match to baseline!)
            pred_post = ellipse_postprocess(pred, PUPIL_CLASS_ID)
            
            pred_resized = cv2.resize(pred_post, (w, h), interpolation=cv2.INTER_NEAREST)

            gt_param = gt_dict[frame_idx]
            gt_mask = draw_gt_ellipse(h, w, gt_param)

            iou = calc_metrics(pred_resized, gt_mask, PUPIL_CLASS_ID)
            case_iou.append(iou)
            
            if i % 50 == 0:
                print(f"   [{case_name}] Frame {frame_idx:04d} IoU: {iou:.4f} (Overexposed: {model.decoder.is_overexposed})")
                sys.stdout.flush()

        if case_iou:
            mIoU = sum(case_iou) / len(case_iou)
            print(f"🎯 [{case_name}] Final mIoU: {mIoU:.4f}")
            total_iou.extend(case_iou)
            sys.stdout.flush()

    if total_iou:
        t_mIoU = sum(total_iou) / len(total_iou)
        print(f"\n🏆 [Total Safe Phase Correction alpha={args.alpha}] mIoU: {t_mIoU:.4f}")
        sys.stdout.flush()

if __name__ == '__main__':
    main()
