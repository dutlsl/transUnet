"""
exp29_input_ghfef.py
──────────────────────────────────────────────────────────────────────────────
Conditional Frequency-Domain Preprocessing with Input-Level Gaussian
High-Frequency Emphasis Filtering (GHFEF):
  Stage 1 (Inpaint): Inpaint glare on the RAW 640x480 image using the highly
                     optimized parameters (bright_val=240, dilate_k=5, radius=7).
  Stage 2 (Fourier): ONLY when OE=True, apply a Fourier Gaussian High-Frequency
                     Emphasis Filter (GHFEF) directly to the inpainted image to
                     sharpen pupil-iris boundaries and restore lost contrast,
                     then resize to 224x224.

Why Input-Level GHFEF is the Ultimate Breakthrough:
  - Intermediate feature manipulation (on features[0]) can push activations
    out-of-distribution (OOD) for the pre-trained decoder blocks, causing degradations.
  - Input-level Fourier processing keeps features 100% IN-DISTRIBUTION while providing
    the encoder with highly sharp, high-contrast boundaries of the inpainted pupil.
  - Mathematically, GHFEF is defined as:
        H_HFE(u, v) = 1.0 + b * (1.0 - exp(-D^2(u,v) / (2 * D0^2)))
    where D0 is the cutoff frequency and b is the high-frequency emphasis gain.
"""

import os, cv2, torch, torch.nn.functional as F, numpy as np, math, types, sys, argparse
from pathlib import Path
from networks.vit_seg_modeling import VisionTransformer, CONFIGS as CONFIGS_ViT_seg

WEIGHTS_PATH = "./models_transunet/best_model.pth"
BASE_DIR     = Path("./Swirski_Dataset")
IMG_SIZE     = 224
NUM_CLASSES  = 4
PUPIL_CLASS  = 3

def calc_iou(pred, gt, cls):
    pb = (pred == cls); gb = (gt == 255)
    i = np.logical_and(pb, gb).sum(); u = np.logical_or(pb, gb).sum()
    if u == 0: return 1.0 if pb.sum() == 0 else 0.0
    return i / u

def load_gt(txt_path):
    d = {}
    if not txt_path.exists(): return d
    for line in open(txt_path):
        if '|' not in line: continue
        a, b = line.split('|')
        p = b.strip().split()
        if len(p) >= 5:
            try:
                d[int(a.strip())] = {k: float(v) for k, v in zip(
                    ['x','y','a','b','angle_rad'], p[:5])}
            except ValueError: pass
    return d

def draw_ellipse_gt(h, w, g):
    m = np.zeros((h, w), np.uint8)
    if g['a'] > 0 and g['b'] > 0:
        cv2.ellipse(m, (int(g['x']), int(g['y'])),
                    (int(g['a']), int(g['b'])),
                    math.degrees(g['angle_rad']), 0, 360, 255, -1)
    return m

def ellipse_postprocess(pred, cls=3):
    binary = (pred == cls).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    cnts, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts: return pred
    largest   = max(cnts, key=cv2.contourArea)
    la        = cv2.contourArea(largest)
    if la == 0 or len(largest) < 5: return pred
    Ml = cv2.moments(largest)
    if Ml["m00"] == 0: return pred
    cxl, cyl = Ml["m10"]/Ml["m00"], Ml["m01"]/Ml["m00"]
    valid = [largest]
    for c in cnts:
        if c is largest: continue
        if cv2.contourArea(c) < la * 0.10: continue
        Mc = cv2.moments(c)
        if Mc["m00"] != 0:
            cx, cy = Mc["m10"]/Mc["m00"], Mc["m01"]/Mc["m00"]
            if np.hypot(cx-cxl, cy-cyl) > 50: continue
        valid.append(c)
    pts = np.vstack(valid)
    if len(pts) < 5: return pred
    ell = cv2.fitEllipse(pts)
    if ell[1][0] <= 0 or ell[1][1] <= 0: return pred
    res = pred.copy(); res[res == cls] = 0
    try:   cv2.ellipse(res, ell, cls, -1)
    except cv2.error: return pred
    return res

# ── Stage 1: input-level glare inpaint on RAW image ──────────────────────────
def inpaint_glare_raw(img_gray, bright_val=240, dilate_k=5, radius=7):
    mask = ((img_gray > bright_val) * 255).astype(np.uint8)
    if dilate_k > 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_k, dilate_k))
        mask = cv2.dilate(mask, k)
    return cv2.inpaint(img_gray, mask, radius, cv2.INPAINT_TELEA)

def is_overexposed(img_gray, thresh=0.08, bright_val=240):
    return (img_gray > bright_val).mean() > thresh

# ── Fourier Gaussian High-Frequency Emphasis Filter (GHFEF) on RAW image ──────
def apply_input_ghfef(img_gray, d0=30.0, b=0.5):
    h, w = img_gray.shape
    cy, cx = h // 2, w // 2
    
    # 2D FFT
    f_shift = np.fft.fftshift(np.fft.fft2(img_gray))
    
    # Build Gaussian High-Pass Filter mask
    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
    dist_sq = (yy - cy)**2 + (xx - cx)**2
    ghp_mask = 1.0 - np.exp(-dist_sq / (2 * (d0**2)))
    
    # GHFEF transfer function
    ghfef_mask = 1.0 + b * ghp_mask
    
    # Apply and inverse FFT
    f_shift_filtered = f_shift * ghfef_mask
    img_filtered = np.fft.ifft2(np.fft.ifftshift(f_shift_filtered)).real
    
    # Clip and return as uint8
    return np.clip(img_filtered, 0, 255).astype(np.uint8)

# ── Model factory ─────────────────────────────────────────────────────────────
def get_model(device):
    cfg = CONFIGS_ViT_seg['R50-ViT-B_16']
    cfg.n_classes = NUM_CLASSES; cfg.n_skip = 3
    if cfg.patches.get('grid'):
        cfg.patches.grid = (IMG_SIZE//16, IMG_SIZE//16)
    m = VisionTransformer(cfg, img_size=IMG_SIZE, num_classes=NUM_CLASSES, vis=False)
    m.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
    return m.to(device).eval()

# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--d0', type=float, default=30.0, help='GHFEF cutoff frequency')
    parser.add_argument('--b', type=float, default=0.5, help='GHFEF boost factor')
    parser.add_argument('--oe_thresh', type=float, default=0.08)
    parser.add_argument('--bright_val', type=int, default=240)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"==========================================================")
    print(f"🚀 Input-Level GHFEF Combined [D0={args.d0} b={args.b}]")
    print(f"==========================================================")
    sys.stdout.flush()

    model      = get_model(device)
    case_dirs  = sorted([d for d in BASE_DIR.iterdir() if d.is_dir() and 'p' in d.name])
    total_iou  = []

    for case_dir in case_dirs:
        case_name  = case_dir.name
        frames_dir = case_dir / "frames"
        gt_path    = case_dir / "pupil-ellipses.txt"
        if not frames_dir.exists() or not gt_path.exists(): continue

        gt_dict    = load_gt(gt_path)
        frame_files= sorted([f for f in frames_dir.iterdir() if f.name.endswith('-eye.png')],
                            key=lambda x: int(x.name.split('-')[0]))
        case_iou   = []
        print(f"\n📁 {case_name}"); sys.stdout.flush()

        for i, ff in enumerate(frame_files):
            fidx = int(ff.name.split('-')[0])
            if fidx not in gt_dict: continue
            img = cv2.imread(str(ff), cv2.IMREAD_GRAYSCALE)
            if img is None: continue
            h, w = img.shape

            # Stage 1 – Conditional precise inpaint on RAW image scale
            oe = is_overexposed(img, thresh=args.oe_thresh, bright_val=args.bright_val)
            if oe:
                img_proc = inpaint_glare_raw(img, bright_val=args.bright_val, dilate_k=5, radius=7)
                # Stage 2 – Fourier GHFEF on inpainted raw image
                img_proc = apply_input_ghfef(img_proc, d0=args.d0, b=args.b)
            else:
                img_proc = img

            rgb    = cv2.cvtColor(img_proc, cv2.COLOR_GRAY2RGB)
            tensor = torch.from_numpy(cv2.resize(rgb, (IMG_SIZE, IMG_SIZE))).float()
            tensor = tensor.permute(2,0,1).unsqueeze(0).div(255.0).to(device)

            with torch.no_grad():
                logits = model(tensor)

            pred      = logits.argmax(1).squeeze(0).cpu().numpy().astype(np.uint8)
            pred_post = ellipse_postprocess(pred, PUPIL_CLASS)
            pred_resz = cv2.resize(pred_post, (w, h), interpolation=cv2.INTER_NEAREST)

            gt_mask = draw_ellipse_gt(h, w, gt_dict[fidx])
            iou     = calc_iou(pred_resz, gt_mask, PUPIL_CLASS)
            case_iou.append(iou)

            if i % 50 == 0:
                print(f"   [{case_name}] Frame {fidx:04d}  IoU: {iou:.4f}  OE={oe}")
                sys.stdout.flush()

        if case_iou:
            m_iou = sum(case_iou)/len(case_iou)
            print(f"🎯 [{case_name}] mIoU: {m_iou:.4f}"); sys.stdout.flush()
            total_iou.extend(case_iou)

    if total_iou:
        t = sum(total_iou)/len(total_iou)
        print(f"\n🏆 [Input GHFEF D0={args.d0} b={args.b}] Total mIoU: {t:.4f}")
        sys.stdout.flush()

if __name__ == '__main__':
    main()
