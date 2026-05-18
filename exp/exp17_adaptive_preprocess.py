"""
exp17_adaptive_preprocess.py
----------------------------
Overexposed frame detection → Conditional preprocessing pipeline
When high_pixel_ratio > threshold:
  Apply: CLAHE + optional glare suppression (bright pixel masking + inpaint)
Otherwise: pass through unchanged (identical to baseline)
"""
import os
import cv2
import torch
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

# ─────────────────────────────────────────────
# Baseline utilities
# ─────────────────────────────────────────────
def calc_metrics(pred_mask, gt_mask, class_id):
    pred_bin = (pred_mask == class_id)
    gt_bin   = (gt_mask == 255)
    inter = np.logical_and(pred_bin, gt_bin).sum()
    union = np.logical_or(pred_bin, gt_bin).sum()
    if union == 0:
        return 1.0 if np.sum(pred_bin) == 0 else 0.0
    return inter / union

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
                p = parts[1].strip().split()
                if len(p) >= 5:
                    gt_dict[frame_idx] = {
                        'x': float(p[0]), 'y': float(p[1]),
                        'a': float(p[2]), 'b': float(p[3]),
                        'angle_rad': float(p[4])
                    }
            except ValueError: continue
    return gt_dict

def draw_gt_ellipse(h, w, gt_param):
    mask = np.zeros((h, w), dtype=np.uint8)
    if gt_param['a'] <= 0 or gt_param['b'] <= 0: return mask
    cv2.ellipse(mask,
                (int(gt_param['x']), int(gt_param['y'])),
                (int(gt_param['a']), int(gt_param['b'])),
                math.degrees(gt_param['angle_rad']), 0, 360, 255, -1)
    return mask

def ellipse_postprocess(pred_mask, pupil_id=3, min_points=5):
    binary = (pred_mask == pupil_id).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return pred_mask
    largest     = max(contours, key=cv2.contourArea)
    largest_area = cv2.contourArea(largest)
    if largest_area == 0 or len(largest) < min_points: return pred_mask
    M = cv2.moments(largest)
    cx_l = M["m10"] / M["m00"] if M["m00"] != 0 else 0
    cy_l = M["m01"] / M["m00"] if M["m00"] != 0 else 0
    valid = [largest]
    for cnt in contours:
        if cnt is largest: continue
        if cv2.contourArea(cnt) < largest_area * 0.10: continue
        Mc = cv2.moments(cnt)
        if Mc["m00"] != 0:
            cx = Mc["m10"] / Mc["m00"]; cy = Mc["m01"] / Mc["m00"]
            if np.hypot(cx - cx_l, cy - cy_l) > 50: continue
        valid.append(cnt)
    all_pts = np.vstack(valid)
    if len(all_pts) < min_points: return pred_mask
    ellipse = cv2.fitEllipse(all_pts)
    if ellipse[1][0] <= 0 or ellipse[1][1] <= 0: return pred_mask
    result = pred_mask.copy()
    result[result == pupil_id] = 0
    try:   cv2.ellipse(result, ellipse, pupil_id, -1)
    except cv2.error: return pred_mask
    return result

# ─────────────────────────────────────────────
# Overexposure detection
# ─────────────────────────────────────────────
def is_overexposed(img_gray, thresh=0.08, bright_val=240):
    return (img_gray > bright_val).mean() > thresh

# ─────────────────────────────────────────────
# Adaptive preprocessing strategies
# ─────────────────────────────────────────────
def preprocess_clahe(img_gray, clip=2.0, grid=8):
    """Baseline CLAHE only."""
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(grid, grid))
    return clahe.apply(img_gray)

def preprocess_gamma(img_gray, gamma=0.5):
    """Gamma compression to suppress highlights."""
    lut = np.array([(i / 255.0) ** gamma * 255 for i in range(256)], dtype=np.uint8)
    return cv2.LUT(img_gray, lut)

def preprocess_inpaint_glare(img_gray, bright_val=240, dilate_k=5):
    """Mask saturated glare regions, inpaint with Telea."""
    mask = ((img_gray > bright_val) * 255).astype(np.uint8)
    if dilate_k > 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_k, dilate_k))
        mask = cv2.dilate(mask, k)
    return cv2.inpaint(img_gray, mask, inpaintRadius=7, flags=cv2.INPAINT_TELEA)

def preprocess_combo(img_gray, clip=2.0, grid=8, gamma=0.6, bright_val=240, dilate_k=5, use_inpaint=True, use_gamma=True, use_clahe=True):
    """Full combo: glare inpaint → gamma → CLAHE."""
    out = img_gray.copy()
    if use_inpaint:
        out = preprocess_inpaint_glare(out, bright_val=bright_val, dilate_k=dilate_k)
    if use_gamma:
        out = preprocess_gamma(out, gamma=gamma)
    if use_clahe:
        out = preprocess_clahe(out, clip=clip, grid=grid)
    return out

# ─────────────────────────────────────────────
# Model  (baseline skip filter patched decoder)
# ─────────────────────────────────────────────
def _make_k(sigma):
    k = int(4 * sigma + 0.5)
    return k + 1 if k % 2 == 0 else k

def patched_decoder_baseline(self, hidden_states, features=None):
    sigma0, sigma1 = 1.0, 0.5
    if features is not None:
        features = list(features)
        if len(features) > 0:
            k0 = _make_k(sigma0)
            features[0] = TF.gaussian_blur(features[0], [k0, k0], [sigma0, sigma0])
        if len(features) > 1:
            k1 = _make_k(sigma1)
            features[1] = TF.gaussian_blur(features[1], [k1, k1], [sigma1, sigma1])
    B, n_patch, hidden = hidden_states.size()
    h = w = int(n_patch ** 0.5)
    x = hidden_states.permute(0, 2, 1).contiguous().view(B, hidden, h, w)
    x = self.conv_more(x)
    for i, block in enumerate(self.blocks):
        skip = features[i] if (features is not None and i < self.config.n_skip) else None
        x = block(x, skip=skip)
    return x

def get_model(device):
    config_vit = CONFIGS_ViT_seg['R50-ViT-B_16']
    config_vit.n_classes = NUM_CLASSES
    config_vit.n_skip = 3
    if config_vit.patches.get('grid'):
        config_vit.patches.grid = (IMG_SIZE // 16, IMG_SIZE // 16)
    model = VisionTransformer(config_vit, img_size=IMG_SIZE, num_classes=NUM_CLASSES, vis=False)
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
    model.decoder.forward = types.MethodType(patched_decoder_baseline, model.decoder)
    model.to(device).eval()
    return model

# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def run(device, model, strategy, oe_thresh):
    """
    strategy: one of
      'clahe'             – CLAHE only
      'gamma'             – gamma compression only
      'inpaint'           – glare inpaint only
      'inpaint_gamma'     – inpaint + gamma
      'inpaint_clahe'     – inpaint + CLAHE
      'inpaint_gamma_clahe' – full combo
    """
    case_dirs = sorted([d for d in BASE_DIR.iterdir() if d.is_dir() and 'p' in d.name])
    total_iou = []

    for case_dir in case_dirs:
        case_name = case_dir.name
        frames_dir = case_dir / "frames"
        gt_path    = case_dir / "pupil-ellipses.txt"
        if not frames_dir.exists() or not gt_path.exists(): continue

        gt_dict = load_ellipse_gt(gt_path)
        frame_files = sorted(
            [f for f in frames_dir.iterdir() if f.name.endswith('-eye.png')],
            key=lambda x: int(x.name.split('-')[0])
        )
        case_iou = []
        print(f"\n📁 {case_name}")
        sys.stdout.flush()

        for i, ff in enumerate(frame_files):
            frame_idx = int(ff.name.split('-')[0])
            if frame_idx not in gt_dict: continue

            img_raw = cv2.imread(str(ff), cv2.IMREAD_GRAYSCALE)
            if img_raw is None: continue
            h, w = img_raw.shape

            # ── Conditional preprocessing ──
            oe = is_overexposed(img_raw, thresh=oe_thresh)
            if oe:
                if strategy == 'clahe':
                    img_proc = preprocess_clahe(img_raw)
                elif strategy == 'gamma':
                    img_proc = preprocess_gamma(img_raw, gamma=0.6)
                elif strategy == 'inpaint':
                    img_proc = preprocess_inpaint_glare(img_raw)
                elif strategy == 'inpaint_gamma':
                    img_proc = preprocess_combo(img_raw, use_inpaint=True, use_gamma=True, use_clahe=False)
                elif strategy == 'inpaint_clahe':
                    img_proc = preprocess_combo(img_raw, use_inpaint=True, use_gamma=False, use_clahe=True)
                elif strategy == 'inpaint_gamma_clahe':
                    img_proc = preprocess_combo(img_raw, use_inpaint=True, use_gamma=True, use_clahe=True)
                else:
                    img_proc = img_raw
            else:
                img_proc = img_raw  # unchanged – identical to baseline

            rgb     = cv2.cvtColor(img_proc, cv2.COLOR_GRAY2RGB)
            resized = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE))
            tensor  = torch.from_numpy(resized).float().permute(2, 0, 1).unsqueeze(0) / 255.0
            tensor  = tensor.to(device)

            with torch.no_grad():
                logits = model(tensor)

            pred      = logits.argmax(1).squeeze(0).cpu().numpy().astype(np.uint8)
            pred_post = ellipse_postprocess(pred, PUPIL_CLASS_ID)
            pred_resz = cv2.resize(pred_post, (w, h), interpolation=cv2.INTER_NEAREST)

            gt_mask = draw_gt_ellipse(h, w, gt_dict[frame_idx])
            iou     = calc_metrics(pred_resz, gt_mask, PUPIL_CLASS_ID)
            case_iou.append(iou)

            if i % 50 == 0:
                print(f"   [{case_name}] Frame {frame_idx:04d} IoU: {iou:.4f} (OE={oe})")
                sys.stdout.flush()

        if case_iou:
            mIoU = sum(case_iou) / len(case_iou)
            print(f"🎯 [{case_name}] mIoU: {mIoU:.4f}")
            total_iou.extend(case_iou)
            sys.stdout.flush()

    if total_iou:
        t = sum(total_iou) / len(total_iou)
        print(f"\n🏆 [{strategy}] Total mIoU: {t:.4f}")
        sys.stdout.flush()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--strategy', type=str, default='inpaint_gamma_clahe',
                        choices=['clahe','gamma','inpaint','inpaint_gamma',
                                 'inpaint_clahe','inpaint_gamma_clahe'])
    parser.add_argument('--oe_thresh', type=float, default=0.08,
                        help='Overexposure detection threshold (ratio of pixels > 240)')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"==============================================")
    print(f"🚀 Adaptive Preprocess [{args.strategy}] | OE thresh={args.oe_thresh} | {device}")
    print(f"==============================================")
    sys.stdout.flush()

    model = get_model(device)
    run(device, model, args.strategy, args.oe_thresh)

if __name__ == '__main__':
    main()
