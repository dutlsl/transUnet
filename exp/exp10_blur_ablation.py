import os
import cv2
import torch
import torch.nn.functional as F
import numpy as np
import csv
import re
import math
import types
from pathlib import Path
import argparse

import torchvision.transforms.functional as TF

from networks.vit_seg_modeling import VisionTransformer
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from tta_fft_core import FFTChannelGate

# =====================================================================
WEIGHTS_PATH = "./models_transunet/best_model.pth"
BASE_DIR = Path("./Swirski_Dataset")
TABLE_DIR = Path("./Swirski_tables/Blur_Ablation")
IMG_SIZE = 224
NUM_CLASSES = 4
PUPIL_CLASS_ID = 3
# =====================================================================

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

def ritnet_preprocess(img_gray):
    normalized = img_gray.astype(np.float32) / 255.0
    gamma_corrected = np.power(normalized, 0.8)
    gamma_corrected = (gamma_corrected * 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
    return clahe.apply(gamma_corrected)

def extract_frame_idx(filename):
    match = re.search(r'(\d+)-eye\.png', filename)
    return int(match.group(1)) if match else -1

def patched_decoder_forward_tta(self, hidden_states, features=None):
    sigma0 = getattr(self, 'sigma0', 1.0)
    sigma1 = getattr(self, 'sigma1', 0.5)
    
    if features is not None:
        features = list(features)
        if len(features) > 0 and sigma0 > 0:
            k0 = int(4 * sigma0 + 0.5)
            if k0 % 2 == 0: k0 += 1
            features[0] = TF.gaussian_blur(features[0], kernel_size=[k0, k0], sigma=[sigma0, sigma0])
        if len(features) > 1 and sigma1 > 0:
            k1 = int(4 * sigma1 + 0.5)
            if k1 % 2 == 0: k1 += 1
            features[1] = TF.gaussian_blur(features[1], kernel_size=[k1, k1], sigma=[sigma1, sigma1])

    B, n_patch, hidden = hidden_states.size()
    h, w = int(np.sqrt(n_patch)), int(np.sqrt(n_patch))
    x = hidden_states.permute(0, 2, 1)
    x = x.contiguous().view(B, hidden, h, w)
    x = self.conv_more(x)
    
    vit_output = x 
    
    for i, decoder_block in enumerate(self.blocks):
        skip = features[i] if (features is not None and i < self.config.n_skip) else None
        
        if i == getattr(self, 'target_position', 1) and skip is not None and hasattr(self, 'fft_gate'):
            channel_weights = self.fft_gate(skip)
            skip = skip * channel_weights
            self.gated_skip = skip
            self.vit_output = vit_output 
            
        x = decoder_block(x, skip=skip)
    return x

def fourier_alignment_loss(gated_skip, vit_output):
    skip_sam = gated_skip.mean(dim=1)
    vit_sam = vit_output.mean(dim=1)
    
    skip_fft = torch.fft.fft2(skip_sam)
    skip_amp = torch.abs(torch.fft.fftshift(skip_fft))
    
    vit_fft = torch.fft.fft2(vit_sam)
    vit_amp = torch.abs(torch.fft.fftshift(vit_fft))
    
    vit_amp = vit_amp.unsqueeze(1)
    vit_amp_resized = F.interpolate(vit_amp, size=skip_amp.shape[1:], mode='bilinear', align_corners=False).squeeze(1)
    
    skip_amp_norm = (skip_amp - skip_amp.min()) / (skip_amp.max() - skip_amp.min() + 1e-8)
    vit_amp_norm = (vit_amp_resized - vit_amp_resized.min()) / (vit_amp_resized.max() - vit_amp_resized.min() + 1e-8)
    
    return F.mse_loss(skip_amp_norm, vit_amp_norm)

def get_transunet_model(device, position, sigma0, sigma1):
    config_vit = CONFIGS_ViT_seg['R50-ViT-B_16']
    config_vit.n_classes = NUM_CLASSES
    config_vit.n_skip = 3
    if config_vit.patches.get('grid') is not None:
        config_vit.patches.grid = (int(IMG_SIZE / 16), int(IMG_SIZE / 16))

    model = VisionTransformer(config_vit, img_size=IMG_SIZE, num_classes=NUM_CLASSES)
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
    
    for param in model.parameters():
        param.requires_grad = False
        
    model.decoder.target_position = position
    model.decoder.sigma0 = sigma0
    model.decoder.sigma1 = sigma1
    model.decoder.forward = types.MethodType(patched_decoder_forward_tta, model.decoder)
    
    model.to(device)
    model.eval()
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--position', type=int, default=1)
    parser.add_argument('--sigma0', type=float, default=1.0)
    parser.add_argument('--sigma1', type=float, default=0.5)
    parser.add_argument('--radius', type=int, default=8)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--iterations', type=int, default=3)
    args = parser.parse_args()

    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = TABLE_DIR / f"sig{args.sigma0}_{args.sigma1}_pos{args.position}.csv"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Starting Blur Ablation (Sig0={args.sigma0}, Sig1={args.sigma1}) on {device}")

    model = get_transunet_model(device, args.position, args.sigma0, args.sigma1)
    
    if args.position == 0: channels = 512
    elif args.position == 1: channels = 256
    elif args.position == 2: channels = 64
    else: raise ValueError("Invalid position")

    case_dirs = [d for d in BASE_DIR.iterdir() if d.is_dir() and 'p' in d.name]

    with open(csv_path, mode='w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['Case', 'Frame_Idx', 'IoU', 'Dice'])
        total_iou, total_dice = [], []

        for case_dir in sorted(case_dirs):
            case_name = case_dir.name
            frames_dir = case_dir / "frames"
            gt_path = case_dir / "pupil-ellipses.txt"

            if not frames_dir.exists() or not gt_path.exists(): continue
            gt_dict = load_ellipse_gt(gt_path)
            frame_files = [f for f in frames_dir.iterdir() if f.name.endswith('-eye.png')]
            frame_files.sort(key=lambda x: extract_frame_idx(x.name))

            case_iou, case_dice = [], []

            for frame_file in frame_files:
                frame_idx = extract_frame_idx(frame_file.name)
                if frame_idx not in gt_dict: continue

                img_raw = cv2.imread(str(frame_file), cv2.IMREAD_GRAYSCALE)
                if img_raw is None: continue
                
                h, w = img_raw.shape
                img_for_model = ritnet_preprocess(img_raw)
                rgb = cv2.cvtColor(img_for_model, cv2.COLOR_GRAY2RGB)
                resized = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE))
                
                tensor = torch.from_numpy(resized).float().permute(2,0,1).unsqueeze(0) / 255.0
                tensor = tensor.to(device)

                fft_gate = FFTChannelGate(channels=channels, radius=args.radius).to(device)
                model.decoder.fft_gate = fft_gate
                optimizer = torch.optim.Adam(fft_gate.parameters(), lr=args.lr)
                
                fft_gate.train()
                for _ in range(args.iterations):
                    optimizer.zero_grad()
                    _ = model(tensor)
                    loss = fourier_alignment_loss(model.decoder.gated_skip, model.decoder.vit_output)
                    loss.backward()
                    optimizer.step()
                
                fft_gate.eval()
                with torch.no_grad():
                    logits = model(tensor)
                    
                pred = logits.argmax(1).squeeze(0).cpu().numpy().astype(np.uint8)
                pred_resized = cv2.resize(pred, (w, h), interpolation=cv2.INTER_NEAREST)

                gt_param = gt_dict[frame_idx]
                gt_mask = draw_gt_ellipse(h, w, gt_param)

                iou, dice = calc_metrics(pred_resized, gt_mask, PUPIL_CLASS_ID)
                case_iou.append(iou)
                case_dice.append(dice)
                csv_writer.writerow([case_name, frame_idx, f"{iou:.4f}", f"{dice:.4f}"])

            if case_iou:
                mIoU = sum(case_iou) / len(case_iou)
                print(f"  [{case_name}] mIoU: {mIoU:.4f}")
                total_iou.extend(case_iou)

        if total_iou:
            t_mIoU = sum(total_iou) / len(total_iou)
            print(f"\n[Total Sig0={args.sigma0}, Sig1={args.sigma1}] mIoU: {t_mIoU:.4f}")

if __name__ == '__main__':
    main()
