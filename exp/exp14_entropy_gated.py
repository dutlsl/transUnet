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

def hybrid_ellipse_postprocess(pred_mask, morph_kernel=13, area_ratio_thresh=0.10, distance_thresh=50):
    pred_bin = (pred_mask == PUPIL_CLASS_ID).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_kernel, morph_kernel))
    closed = cv2.morphologyEx(pred_bin, cv2.MORPH_CLOSE, kernel)
    
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours: return np.zeros_like(pred_mask)
        
    max_area = 0
    max_contour = None
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > max_area:
            max_area = area
            max_contour = cnt
            
    if max_area == 0: return np.zeros_like(pred_mask)
        
    M = cv2.moments(max_contour)
    if M["m00"] != 0:
        cx_max = int(M["m10"] / M["m00"])
        cy_max = int(M["m01"] / M["m00"])
    else:
        cx_max, cy_max = 0, 0
        
    valid_contours = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < max_area * area_ratio_thresh: continue
        M_cnt = cv2.moments(cnt)
        if M_cnt["m00"] != 0:
            cx = int(M_cnt["m10"] / M_cnt["m00"])
            cy = int(M_cnt["m01"] / M_cnt["m00"])
            dist = np.sqrt((cx - cx_max)**2 + (cy - cy_max)**2)
            if dist > distance_thresh: continue
        valid_contours.append(cnt)
        
    if not valid_contours: return np.zeros_like(pred_mask)
        
    all_points = np.vstack(valid_contours)
    if len(all_points) >= 5:
        ellipse = cv2.fitEllipse(all_points)
        result_mask = np.zeros_like(pred_mask)
        cv2.ellipse(result_mask, ellipse, PUPIL_CLASS_ID, -1)
        return result_mask
    return np.zeros_like(pred_mask)


def patched_vit_forward_entropy(self, x):
    if x.size()[1] == 1:
        x = x.repeat(1,3,1,1)
    
    x, attn_weights, features = self.transformer(x)
    
    if attn_weights:
        last_attn = attn_weights[-1] # [B, 12, 196, 196]
        p = last_attn.mean(dim=1) # [B, 196, 196]
        p = p / (p.sum(dim=-1, keepdim=True) + 1e-8)
        entropy = -(p * (p + 1e-8).log()).sum(dim=-1).mean()
        H_norm = (entropy / math.log(196)).clamp(0, 1)
        self.decoder.H_norm = H_norm.item()
    else:
        self.decoder.H_norm = 0.0
        
    x = self.decoder(x, features)
    logits = self.segmentation_head(x)
    return logits

def patched_decoder_forward_entropy(self, hidden_states, features=None):
    H_norm = getattr(self, 'H_norm', 0.0)
    gamma = getattr(self, 'gamma', 0.5)
    
    sigma0 = 1.0 + gamma * H_norm
    sigma1 = 0.5 + gamma * H_norm
    
    if features is not None:
        features = list(features)
        
        if len(features) > 0:
            k0 = int(4 * sigma0 + 0.5); k0 = k0 + 1 if k0 % 2 == 0 else k0
            features[0] = TF.gaussian_blur(features[0], kernel_size=[k0, k0], sigma=[sigma0, sigma0])
            
        if len(features) > 1:
            k1 = int(4 * sigma1 + 0.5); k1 = k1 + 1 if k1 % 2 == 0 else k1
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

def get_model(device, gamma):
    config_vit = CONFIGS_ViT_seg['R50-ViT-B_16']
    config_vit.n_classes = NUM_CLASSES
    config_vit.n_skip = 3
    if config_vit.patches.get('grid') is not None:
        config_vit.patches.grid = (int(IMG_SIZE / 16), int(IMG_SIZE / 16))

    model = VisionTransformer(config_vit, img_size=IMG_SIZE, num_classes=NUM_CLASSES, vis=True)
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
    
    model.decoder.gamma = gamma
    model.forward = types.MethodType(patched_vit_forward_entropy, model)
    model.decoder.forward = types.MethodType(patched_decoder_forward_entropy, model.decoder)
    
    model.to(device)
    model.eval()
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gamma', type=float, default=0.5, help='Gamma multiplier for entropy')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"========================================")
    print(f"🚀 Entropy-Gated Blur (gamma={args.gamma}) on {device}")
    print(f"========================================")
    sys.stdout.flush()

    model = get_model(device, args.gamma)
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
            pred_resized = cv2.resize(pred, (w, h), interpolation=cv2.INTER_NEAREST)
            
            result_mask = hybrid_ellipse_postprocess(pred_resized)

            gt_param = gt_dict[frame_idx]
            gt_mask = draw_gt_ellipse(h, w, gt_param)

            iou = calc_metrics(result_mask, gt_mask, PUPIL_CLASS_ID)
            case_iou.append(iou)
            
            if i % 50 == 0:
                print(f"   [{case_name}] Frame {frame_idx:04d} IoU: {iou:.4f}")
                sys.stdout.flush()

        if case_iou:
            mIoU = sum(case_iou) / len(case_iou)
            print(f"🎯 [{case_name}] Final mIoU: {mIoU:.4f}")
            total_iou.extend(case_iou)
            sys.stdout.flush()

    if total_iou:
        t_mIoU = sum(total_iou) / len(total_iou)
        print(f"\n🏆 [Total Entropy-Gated Blur] mIoU: {t_mIoU:.4f}")
        sys.stdout.flush()

if __name__ == '__main__':
    main()
