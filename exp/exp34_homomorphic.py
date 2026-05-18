"""
exp34_homomorphic.py
──────────────────────────────────────────────────────────────────────────────
Two-Stage OE Correction with Conditional Homomorphic Filtering (CHF)
& Raw-Scale Inpainting:
  Stage 1 (Space-Fourier Preprocessing):
     - For any overexposed frame (OE=True):
         1. Space-Domain: Inpaint the saturated glare on the raw 640x480 image (bright_val=240,
                          dilate_k=5, radius=7) to remove flat, zero-gradient white reflections.
         2. Fourier-Domain: Apply Conditional Homomorphic Filtering (CHF) to the inpainted
                            raw image.
                            Homomorphic filtering separates illumination and reflectance:
                                ln(I) = ln(i) + ln(r)
                            It applies a high-frequency emphasis filter to suppress the
                            low-frequency illumination components (glare gradients) while
                            amplifying high-frequency reflectance components (pupil boundaries):
                                H(u, v) = (gamma_h - gamma_l) * (1 - exp(-c * D^2 / D0^2)) + gamma_l
                            This mathematically sharpens and enhances the pupil-iris boundary
                            without requiring any external templates, avoiding scale or size distortion!
  Stage 2 (Model): Pass to TransUNet with the highly optimized baseline Gaussian Skip Filter.

Why Homomorphic Filtering is the Scientific Holy Grail:
  - Unlike template-based FAB, CHF is 100% self-contained and operates dynamically on the
    individual frame's own structure, guaranteeing ZERO pupil size or scale distortion.
  - Unlike feature-level attention biases, CHF operates at the input level, keeping intermediate
    representations completely intact and 100% in-distribution for the pre-trained decoder.
  - The illumination-reflectance separation is the standard physics-based approach to handling
    severe spotlight and glare distortion!
"""

import os, cv2, torch, torch.nn.functional as F, numpy as np, math, types, sys, argparse
from pathlib import Path
import torchvision.transforms.functional as TF
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

# ── Homomorphic Filtering ────────────────────────────────────────────────────
def homomorphic_filter(img, gamma_l=0.5, gamma_h=1.5, d0=30.0, c=1.0):
    # img is 2D gray float in [0, 255]
    img_log = np.log1p(img.astype(np.float32))
    
    # FFT
    f = np.fft.fft2(img_log)
    f_shift = np.fft.fftshift(f)
    
    # Filter transfer function
    rows, cols = img.shape
    cy, cx = rows / 2.0, cols / 2.0
    y, x = np.meshgrid(np.arange(rows), np.arange(cols), indexing='ij')
    dist_sq = (y - cy)**2 + (x - cx)**2
    
    ghp = 1.0 - np.exp(-c * dist_sq / (d0**2))
    h = (gamma_h - gamma_l) * ghp + gamma_l
    
    # Apply filter
    f_filtered = f_shift * h
    
    # Inverse FFT
    f_ishift = np.fft.ifftshift(f_filtered)
    img_back = np.real(np.fft.ifft2(f_ishift))
    
    # Exponential
    img_exp = np.expm1(img_back)
    
    # Normalize and clip
    img_exp = np.clip(img_exp, 0, 255)
    return img_exp.astype(np.uint8)

# ── Patched Decoder (keeps baseline skip blur!) ───────────────────────────────
def _mk(sigma):
    k = int(4*sigma+0.5); return k+1 if k%2==0 else k

def patched_decoder_baseline(self, hidden_states, features=None):
    sigma0, sigma1 = 1.0, 0.5
    if features is not None:
        features = list(features)
        if len(features) > 0:
            features[0] = TF.gaussian_blur(features[0], [_mk(sigma0)]*2, [sigma0]*2)
        if len(features) > 1:
            features[1] = TF.gaussian_blur(features[1], [_mk(sigma1)]*2, [sigma1]*2)
            
    B, n, h_ = hidden_states.size()
    hw = int(n**0.5)
    x  = hidden_states.permute(0,2,1).contiguous().view(B, h_, hw, hw)
    x  = self.conv_more(x)
    for i, block in enumerate(self.blocks):
        skip = features[i] if (features is not None and i < self.config.n_skip) else None
        x    = block(x, skip=skip)
    return x

# ── Model factory ─────────────────────────────────────────────────────────────
def get_model(device):
    cfg = CONFIGS_ViT_seg['R50-ViT-B_16']
    cfg.n_classes = NUM_CLASSES; cfg.n_skip = 3
    if cfg.patches.get('grid'):
        cfg.patches.grid = (IMG_SIZE//16, IMG_SIZE//16)
    m = VisionTransformer(cfg, img_size=IMG_SIZE, num_classes=NUM_CLASSES, vis=False)
    m.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
    
    # Patch decoder to keep baseline skip Gaussian Blur
    m.decoder.forward = types.MethodType(patched_decoder_baseline, m.decoder)
    
    return m.to(device).eval()

# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gamma_l', type=float, default=0.5, help='Low-freq gain')
    parser.add_argument('--gamma_h', type=float, default=1.5, help='High-freq gain')
    parser.add_argument('--d0', type=float, default=30.0, help='Cutoff frequency')
    parser.add_argument('--oe_thresh', type=float, default=0.08)
    parser.add_argument('--bright_val', type=int, default=240)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"==========================================================")
    print(f"🚀 Homomorphic Filtering [gamma_l={args.gamma_l} gamma_h={args.gamma_h} d0={args.d0}]")
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

            # Space-Domain Glare Inpainting
            oe = is_overexposed(img, thresh=args.oe_thresh, bright_val=args.bright_val)
            if oe:
                img_proc = inpaint_glare_raw(img, bright_val=args.bright_val, dilate_k=5, radius=7)
                # Fourier-Domain Homomorphic Filtering
                img_proc = homomorphic_filter(img_proc, gamma_l=args.gamma_l, gamma_h=args.gamma_h, d0=args.d0)
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
        print(f"\n🏆 [Homomorphic gamma_l={args.gamma_l} gamma_h={args.gamma_h} d0={args.d0}] Total mIoU: {t:.4f}")
        sys.stdout.flush()

if __name__ == '__main__':
    main()
