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
from collections import defaultdict

from networks.vit_seg_modeling import VisionTransformer
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg

WEIGHTS_PATH = "./models_transunet/best_model.pth"
RAW_BASE_DIR = Path("./LPW")
GT_BASE_DIR = Path("./Pupils_in_the_wild_improved")
TABLE_DIR = Path("./LPW_tables")

IMG_SIZE = 224
NUM_CLASSES = 4
PUPIL_CLASS_ID = 3

def _make_blur(sigma):
    k_size = int(4 * sigma + 0.5)
    if k_size % 2 == 0: k_size += 1
    if k_size < 3: k_size = 3
    return k_size

def patched_decoder_forward(self, hidden_states, features=None):
    sigma01 = getattr(self, 'filter_sigma', 0.0)
    if features is not None:
        features = list(features)
        if sigma01 > 0:
            k = _make_blur(sigma01)
            if len(features) > 0:
                features[0] = TF.gaussian_blur(features[0], kernel_size=[k, k], sigma=[sigma01, sigma01])
            if len(features) > 1:
                features[1] = TF.gaussian_blur(features[1], kernel_size=[k, k], sigma=[sigma01, sigma01])

    B, n_patch, hidden = hidden_states.size()
    h, w = int(np.sqrt(n_patch)), int(np.sqrt(n_patch))
    x = hidden_states.permute(0, 2, 1)
    x = x.contiguous().view(B, hidden, h, w)
    x = self.conv_more(x)
    for i, decoder_block in enumerate(self.blocks):
        skip = features[i] if (features is not None and i < self.config.n_skip) else None
        x = decoder_block(x, skip=skip)
    return x

def get_model(device, sigma=1.0):
    config_vit = CONFIGS_ViT_seg['R50-ViT-B_16']
    config_vit.n_classes = NUM_CLASSES
    config_vit.n_skip = 3
    if config_vit.patches.get('grid') is not None:
        config_vit.patches.grid = (int(IMG_SIZE / 16), int(IMG_SIZE / 16))
    model = VisionTransformer(config_vit, img_size=IMG_SIZE, num_classes=NUM_CLASSES)
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
    model.decoder.filter_sigma = sigma
    model.decoder.forward = types.MethodType(patched_decoder_forward, model.decoder)
    model.to(device)
    model.eval()
    return model

def build_gt_mapping(gt_dir):
    mapping = {}
    for f in gt_dir.rglob("*_pupil.mp4"):
        match = re.search(r'folder-(\d+)_file-(\d+)', f.name)
        if match:
            folder_idx = int(match.group(1))
            file_idx = int(match.group(2))
            mapping[f"{folder_idx}_{file_idx}"] = f
    return mapping

def calc_metrics(pred_bin, gt_bin):
    intersection = np.logical_and(pred_bin, gt_bin).sum()
    union = np.logical_or(pred_bin, gt_bin).sum()
    if union == 0:
        iou = 1.0 if np.sum(pred_bin) == 0 else 0.0
        dice = 1.0 if np.sum(pred_bin) == 0 else 0.0
    else:
        iou = intersection / union
        dice = 2 * intersection / (pred_bin.sum() + gt_bin.sum())
    return iou, dice

def apply_ellipse_params_with_kernel(pred_mask, kernel_size, area_ratio=0.10, dist_thresh=50.0, pupil_id=3):
    binary = (pred_mask == pupil_id).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return pred_mask
        
    largest = max(contours, key=cv2.contourArea)
    largest_area = cv2.contourArea(largest)
    if largest_area == 0 or len(largest) < 5: return pred_mask
        
    M_large = cv2.moments(largest)
    cx_large = M_large["m10"] / M_large["m00"] if M_large["m00"] != 0 else 0
    cy_large = M_large["m01"] / M_large["m00"] if M_large["m00"] != 0 else 0
    
    valid_points = [largest]
    for cnt in contours:
        if cnt is largest: continue
        area = cv2.contourArea(cnt)
        if area < largest_area * area_ratio: continue
            
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cx = M["m10"] / M["m00"]
            cy = M["m01"] / M["m00"]
            dist = np.sqrt((cx - cx_large)**2 + (cy - cy_large)**2)
            if dist > dist_thresh: continue
                
        valid_points.append(cnt)
        
    all_points = np.vstack(valid_points)
    if len(all_points) < 5: return pred_mask
        
    ellipse = cv2.fitEllipse(all_points)
    if ellipse[1][0] <= 0 or ellipse[1][1] <= 0: return pred_mask
        
    result = pred_mask.copy()
    result[result == pupil_id] = 0
    try: cv2.ellipse(result, ellipse, pupil_id, -1)
    except cv2.error: return pred_mask
    return result

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--f_start', type=int, required=True)
    parser.add_argument('--f_end', type=int, required=True)
    parser.add_argument('--dry_run', action='store_true')
    args = parser.parse_args()

    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = TABLE_DIR / f"exp3_kernel_search_f{args.f_start}_to_f{args.f_end}.csv"

    kernel_sizes = [5, 7, 9, 11, 13]
    print(f"Testing Kernel Sizes: {kernel_sizes}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = get_model(device, sigma=1.0)
    gt_mapping = build_gt_mapping(GT_BASE_DIR)
    
    with open(out_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Folder', 'Video', 'KernelSize', 'mIoU', 'mDice'])
        
        for folder_idx in range(args.f_start, args.f_end + 1):
            folder_dir = RAW_BASE_DIR / str(folder_idx)
            raw_videos = list(folder_dir.glob("*.avi")) if folder_dir.exists() else []
            if not raw_videos: continue
                
            for raw_path in sorted(raw_videos):
                if raw_path.name.startswith("._"): continue
                try: file_idx = int(raw_path.stem)
                except ValueError: continue
                
                key = f"{folder_idx}_{file_idx}"
                if key not in gt_mapping: continue
                
                gt_path = gt_mapping[key]
                cap_raw = cv2.VideoCapture(str(raw_path))
                cap_gt = cv2.VideoCapture(str(gt_path))
                
                vid_scores = {i: ([], []) for i in range(len(kernel_sizes))}
                
                frame_count = 0
                with torch.no_grad():
                    while True:
                        ret_r, frame_r = cap_raw.read()
                        ret_g, frame_g = cap_gt.read()
                        if not (ret_r and ret_g): break
                        
                        h, w = frame_r.shape[:2]
                        gray = cv2.cvtColor(frame_r, cv2.COLOR_BGR2GRAY) if len(frame_r.shape) == 3 else frame_r
                        rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
                        resized = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE))
                        tensor = torch.from_numpy(resized).float().permute(2, 0, 1).unsqueeze(0) / 255.0
                        tensor = tensor.to(device)
                        
                        logits = model(tensor)
                        pred_raw = logits.argmax(1).squeeze(0).cpu().numpy().astype(np.uint8)
                        
                        gray_gt = cv2.cvtColor(frame_g, cv2.COLOR_BGR2GRAY) if len(frame_g.shape) == 3 else frame_g
                        gray_gt = cv2.resize(gray_gt, (w, h), interpolation=cv2.INTER_NEAREST)
                        _, gt_bin = cv2.threshold(gray_gt, 127, 255, cv2.THRESH_BINARY)
                        gt_bin_bool = (gt_bin == 255)
                        
                        # Evaluate all kernel sizes
                        for c_idx, k_size in enumerate(kernel_sizes):
                            pred_post = apply_ellipse_params_with_kernel(pred_raw, k_size, 0.10, 50.0, PUPIL_CLASS_ID)
                            pred_full = cv2.resize(pred_post, (w, h), interpolation=cv2.INTER_NEAREST)
                            pred_bin = (pred_full == PUPIL_CLASS_ID)
                            
                            iou, dice = calc_metrics(pred_bin, gt_bin_bool)
                            vid_scores[c_idx][0].append(iou)
                            vid_scores[c_idx][1].append(dice)
                            
                        frame_count += 1
                        if args.dry_run and frame_count >= 10: break
                
                cap_raw.release()
                cap_gt.release()
                
                if frame_count > 0:
                    for c_idx, k_size in enumerate(kernel_sizes):
                        m_iou = np.mean(vid_scores[c_idx][0])
                        m_dice = np.mean(vid_scores[c_idx][1])
                        writer.writerow([folder_idx, raw_path.stem, k_size, f"{m_iou:.4f}", f"{m_dice:.4f}"])
                    f.flush()
                    print(f"  [Folder {folder_idx} | {raw_path.stem}.avi] completed 5 kernels ({frame_count} frames)")
                    if args.dry_run:
                        print("Dry run completed successfully.")
                        return

if __name__ == '__main__':
    main()
