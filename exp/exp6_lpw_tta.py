import os
import cv2
import torch
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
from tta_fft_core import FFTChannelGate, alignment_loss

# =====================================================================
WEIGHTS_PATH = "./models_transunet/best_model.pth"
RAW_BASE_DIR = Path("./LPW")
GT_BASE_DIR = Path("./Pupils_in_the_wild_improved")
TABLE_DIR = Path("./LPW_tables")
OVERLAY_DIR = Path("./LPW_overlays_tta")

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

def _make_blur(sigma):
    k_size = int(4 * sigma + 0.5)
    if k_size % 2 == 0: k_size += 1
    if k_size < 3: k_size = 3
    return k_size

def ritnet_preprocess(img_gray):
    normalized = img_gray.astype(np.float32) / 255.0
    gamma_corrected = np.power(normalized, 0.8)
    gamma_corrected = (gamma_corrected * 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
    return clahe.apply(gamma_corrected)

def ellipse_postprocess(pred_mask, pupil_id=3, min_points=5):
    binary = (pred_mask == pupil_id).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return pred_mask
    largest = max(contours, key=cv2.contourArea)
    largest_area = cv2.contourArea(largest)
    if largest_area == 0 or len(largest) < min_points:
        return pred_mask
    M_large = cv2.moments(largest)
    cx_large = M_large["m10"] / M_large["m00"] if M_large["m00"] != 0 else 0
    cy_large = M_large["m01"] / M_large["m00"] if M_large["m00"] != 0 else 0
    valid_points = [largest]
    for cnt in contours:
        if cnt is largest: continue
        area = cv2.contourArea(cnt)
        if area < largest_area * 0.10: continue
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cx, cy = M["m10"] / M["m00"], M["m01"] / M["m00"]
            if np.sqrt((cx - cx_large)**2 + (cy - cy_large)**2) > 50: continue
        valid_points.append(cnt)
    all_points = np.vstack(valid_points)
    if len(all_points) < min_points: return pred_mask
    ellipse = cv2.fitEllipse(all_points)
    if ellipse[1][0] <= 0 or ellipse[1][1] <= 0: return pred_mask
    result = pred_mask.copy()
    result[result == pupil_id] = 0
    try:
        cv2.ellipse(result, ellipse, pupil_id, -1)
    except cv2.error:
        pass
    return result

def build_gt_mapping(gt_dir):
    mapping = {}
    for f in gt_dir.rglob("*_pupil.mp4"):
        match = re.search(r'folder-(\d+)_file-(\d+)', f.name)
        if match:
            folder_idx = int(match.group(1))
            file_idx = int(match.group(2))
            mapping[f"{folder_idx}_{file_idx}"] = f
    return mapping

def patched_decoder_forward_tta(self, hidden_states, features=None):
    sigma0 = 1.0
    sigma1 = 0.5
    if features is not None:
        features = list(features)
        if len(features) > 0:
            k0 = _make_blur(sigma0)
            features[0] = TF.gaussian_blur(features[0], kernel_size=[k0, k0], sigma=[sigma0, sigma0])
        if len(features) > 1:
            k1 = _make_blur(sigma1)
            features[1] = TF.gaussian_blur(features[1], kernel_size=[k1, k1], sigma=[sigma1, sigma1])

    B, n_patch, hidden = hidden_states.size()
    h, w = int(np.sqrt(n_patch)), int(np.sqrt(n_patch))
    x = hidden_states.permute(0, 2, 1)
    x = x.contiguous().view(B, hidden, h, w)
    x = self.conv_more(x)
    
    vit_output = x 
    
    for i, decoder_block in enumerate(self.blocks):
        skip = features[i] if (features is not None and i < self.config.n_skip) else None
        
        if i == 1 and skip is not None and hasattr(self, 'fft_gate'):
            channel_weights = self.fft_gate(skip)
            skip = skip * channel_weights
            self.gated_skip1 = skip
            self.vit_output = vit_output
            
        x = decoder_block(x, skip=skip)
    return x

def get_transunet_model(device):
    config_vit = CONFIGS_ViT_seg['R50-ViT-B_16']
    config_vit.n_classes = NUM_CLASSES
    config_vit.n_skip = 3
    if config_vit.patches.get('grid') is not None:
        config_vit.patches.grid = (int(IMG_SIZE / 16), int(IMG_SIZE / 16))

    model = VisionTransformer(config_vit, img_size=IMG_SIZE, num_classes=NUM_CLASSES)
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
    
    for param in model.parameters():
        param.requires_grad = False
        
    model.decoder.forward = types.MethodType(patched_decoder_forward_tta, model.decoder)
    model.to(device)
    model.eval()
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--radius', type=int, default=8)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--iterations', type=int, default=3)
    parser.add_argument('--ellipse', action='store_true')
    parser.add_argument('--preprocess', action='store_true')
    parser.add_argument('--dry_run', action='store_true')
    parser.add_argument('--folder', type=int, default=1, help='Specific LPW folder to process')
    parser.add_argument('--all_folders', action='store_true', help='Process all folders 1-22')
    args = parser.parse_args()

    folder_list = list(range(1, 23)) if args.all_folders else [args.folder]
    suffix = f"tta_r{args.radius}_lr{args.lr}_it{args.iterations}_ell{'O' if args.ellipse else 'X'}_pre{'O' if args.preprocess else 'X'}_f{'ALL' if args.all_folders else args.folder}"
    frame_csv_path = TABLE_DIR / f"lpw_{suffix}_frames.csv"
    compact_csv_path = TABLE_DIR / f"lpw_{suffix}_compact.csv"

    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"LPW TTA 시작 (Radius={args.radius}, LR={args.lr}, Iterations={args.iterations})")

    model = get_transunet_model(device)
    gt_mapping = build_gt_mapping(GT_BASE_DIR)

    with open(frame_csv_path, mode='w', newline='') as f_csv, open(compact_csv_path, mode='w', newline='') as c_csv:
        frame_writer = csv.writer(f_csv)
        frame_writer.writerow(['Folder', 'Video', 'Frame', 'IoU', 'Dice'])
        
        compact_writer = csv.writer(c_csv)
        compact_writer.writerow(['Folder', 'Video', 'mIoU', 'mDice'])

        for folder_idx in folder_list:
            folder_dir = RAW_BASE_DIR / str(folder_idx)
            raw_videos = list(folder_dir.glob("*.avi")) if folder_dir.exists() else []
            
            if not raw_videos: continue
                
            for raw_path in sorted(raw_videos):
                if raw_path.name.startswith("._"): continue
                try:
                    file_idx = int(raw_path.stem)
                except ValueError: continue
                    
                key = f"{folder_idx}_{file_idx}"
                if key not in gt_mapping: continue
                    
                gt_path = gt_mapping[key]
                cap_raw = cv2.VideoCapture(str(raw_path))
                cap_gt = cv2.VideoCapture(str(gt_path))
                
                frame_idx = 0
                vid_iou, vid_dice = [], []

                while True:
                    ret_r, frame_r = cap_raw.read()
                    ret_g, frame_g = cap_gt.read()
                    if not (ret_r and ret_g): break

                    h, w = frame_r.shape[:2]
                    gray = cv2.cvtColor(frame_r, cv2.COLOR_BGR2GRAY) if len(frame_r.shape) == 3 else frame_r

                    img_for_model = gray
                    if args.preprocess:
                        img_for_model = ritnet_preprocess(gray)

                    rgb = cv2.cvtColor(img_for_model, cv2.COLOR_GRAY2RGB)
                    resized = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE))
                    tensor = torch.from_numpy(resized).float().permute(2, 0, 1).unsqueeze(0) / 255.0
                    tensor = tensor.to(device)

                    # --- TTA LOOP START ---
                    fft_gate = FFTChannelGate(channels=256, radius=args.radius).to(device)
                    model.decoder.fft_gate = fft_gate
                    optimizer = torch.optim.Adam(fft_gate.parameters(), lr=args.lr)
                    
                    fft_gate.train()
                    for _ in range(args.iterations):
                        optimizer.zero_grad()
                        _ = model(tensor)
                        loss = alignment_loss(model.decoder.gated_skip1, model.decoder.vit_output)
                        loss.backward()
                        optimizer.step()
                    
                    fft_gate.eval()
                    with torch.no_grad():
                        logits = model(tensor)
                    # --- TTA LOOP END ---

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
                    frame_writer.writerow([folder_idx, raw_path.stem, frame_idx, f"{iou:.4f}", f"{dice:.4f}"])

                    frame_idx += 1
                    if args.dry_run and frame_idx >= 10: break

                if vid_iou:
                    mIoU = sum(vid_iou) / len(vid_iou)
                    mDice = sum(vid_dice) / len(vid_dice)
                    compact_writer.writerow([folder_idx, raw_path.stem, f"{mIoU:.4f}", f"{mDice:.4f}"])
                    print(f"  [Folder {folder_idx}] {raw_path.stem} mIoU: {mIoU:.4f}")

if __name__ == '__main__':
    main()
