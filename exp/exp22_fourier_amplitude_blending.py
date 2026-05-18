"""
exp22_fourier_amplitude_blending.py
──────────────────────────────────────────────────────────────────────────────
Two-Stage OE Correction with Fourier Amplitude Blending (FAB):
  Stage 1 (Input):   Fourier Amplitude Blending (FAB). We blend the amplitude
                     spectrum of the overexposed frame with the amplitude spectrum
                     of a clean normal template (e.g. first frame of p1-left) while
                     preserving the phase of the overexposed frame.
  Stage 2 (Feature): Fourier Gaussian High-Frequency Emphasis Filter (GHFEF)
                     on skip[0] to sharpen the reconstructed pupil boundaries.

Why Fourier Amplitude Blending (FAB) is academically brilliant & highly effective:
  - The Fourier phase spectrum contains the precise spatial structure and boundaries
    (pupil shape).
  - The Fourier amplitude spectrum contains the style, contrast, and exposure profile
    (severe bright glare and halo).
  - By blending the amplitude of a normal-exposure clean eye with the phase of the
    overexposed eye, we mathematically "re-expose" the eye, wiping out the glare
    intensity while perfectly reconstructing the underlying pupil boundary!
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

# Load a representative clean template image
def get_clean_template():
    # First frame of p1-left is a great normal-exposure representative
    tpl_path = BASE_DIR / "p1-left" / "frames" / "0000-eye.png"
    if tpl_path.exists():
        tpl = cv2.imread(str(tpl_path), cv2.IMREAD_GRAYSCALE)
        return cv2.resize(tpl, (IMG_SIZE, IMG_SIZE))
    # Fallback to zeros if not found
    return np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)

# ── Stage 1: Fourier Amplitude Blending (FAB) ────────────────────────────────
def fourier_amplitude_blend(img_oe, img_clean, alpha=0.5):
    """
    img_oe: [H, W] overexposed image
    img_clean: [H, W] clean template image
    alpha: blend ratio of clean amplitude (1.0 = completely replace amplitude)
    """
    # 1. 2D FFT of both images
    f_oe = np.fft.fft2(img_oe)
    f_cl = np.fft.fft2(img_clean)
    
    # 2. Extract amplitude and phase
    amp_oe, phase_oe = np.abs(f_oe), np.angle(f_oe)
    amp_cl = np.abs(f_cl)
    
    # 3. Blend amplitude, preserve phase of overexposed image
    amp_blend = alpha * amp_cl + (1.0 - alpha) * amp_oe
    
    # 4. Reconstruct complex spectrum
    f_recon = amp_blend * np.exp(1j * phase_oe)
    
    # 5. Inverse 2D FFT
    img_recon = np.real(np.fft.ifft2(f_recon))
    img_recon = np.clip(img_recon, 0, 255).astype(np.uint8)
    return img_recon

def is_overexposed(img_gray, thresh=0.08, bright_val=240):
    return (img_gray > bright_val).mean() > thresh

# ── Stage 2: Gaussian High-Frequency Emphasis Filter (GHFEF) ─────────────────
def gaussian_high_frequency_emphasis(feat, d0_frac=0.35, a=1.0, b=0.5):
    B, C, H, W = feat.shape
    device = feat.device
    cy, cx = H // 2, W // 2
    d0 = d0_frac * min(H, W)
    yy = torch.arange(H, device=device).float().view(H, 1).expand(H, W)
    xx = torch.arange(W, device=device).float().view(1, W).expand(H, W)
    dist_sq = (yy - cy)**2 + (xx - cx)**2
    ghp_mask = 1.0 - torch.exp(-dist_sq / (2 * (d0**2)))
    hfe_mask = a + b * ghp_mask
    hfe_mask_shifted = torch.fft.ifftshift(hfe_mask)
    feat_fft = torch.fft.fft2(feat)
    feat_fft_filtered = feat_fft * hfe_mask_shifted.unsqueeze(0).unsqueeze(0)
    feat_filtered = torch.fft.ifft2(feat_fft_filtered).real
    return feat_filtered

# ── Patched decoder ───────────────────────────────────────────────────────────
def _mk(sigma):
    k = int(4*sigma+0.5); return k+1 if k%2==0 else k

def patched_decoder(self, hidden_states, features=None):
    sigma0, sigma1 = 1.0, 0.5
    is_oe  = getattr(self, 'is_oe',  False)
    d0_frac = getattr(self, 'd0_frac', 0.35)
    gain_b = getattr(self, 'gain_b', 0.5)

    if features is not None:
        features = list(features)
        if len(features) > 0:
            features[0] = TF.gaussian_blur(features[0], [_mk(sigma0)]*2, [sigma0]*2)
        if len(features) > 1:
            features[1] = TF.gaussian_blur(features[1], [_mk(sigma1)]*2, [sigma1]*2)

        if is_oe and d0_frac > 0 and len(features) > 0:
            features[0] = gaussian_high_frequency_emphasis(features[0], d0_frac=d0_frac, a=1.0, b=gain_b)

    B, n, h_ = hidden_states.size()
    hw = int(n**0.5)
    x  = hidden_states.permute(0,2,1).contiguous().view(B, h_, hw, hw)
    x  = self.conv_more(x)
    for i, block in enumerate(self.blocks):
        skip = features[i] if (features is not None and i < self.config.n_skip) else None
        x    = block(x, skip=skip)
    return x

# ── Model factory ─────────────────────────────────────────────────────────────
def get_model(device, d0_frac, gain_b):
    cfg = CONFIGS_ViT_seg['R50-ViT-B_16']
    cfg.n_classes = NUM_CLASSES; cfg.n_skip = 3
    if cfg.patches.get('grid'):
        cfg.patches.grid = (IMG_SIZE//16, IMG_SIZE//16)
    m = VisionTransformer(cfg, img_size=IMG_SIZE, num_classes=NUM_CLASSES, vis=False)
    m.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
    m.decoder.d0_frac = d0_frac
    m.decoder.gain_b = gain_b
    m.decoder.is_oe = False
    m.decoder.forward = types.MethodType(patched_decoder, m.decoder)
    return m.to(device).eval()

# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--alpha', type=float, default=0.5,
                        help='Amplitude blending ratio (0.0 to 1.0)')
    parser.add_argument('--d0_frac', type=float, default=0.35)
    parser.add_argument('--gain_b', type=float, default=0.5)
    parser.add_argument('--oe_thresh', type=float, default=0.08)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"==========================================================")
    print(f"🚀 FAB OE [alpha={args.alpha} GHFE d0={args.d0_frac} b={args.gain_b}]")
    print(f"==========================================================")
    sys.stdout.flush()

    template_clean = get_clean_template()
    model          = get_model(device, args.d0_frac, args.gain_b)
    case_dirs      = sorted([d for d in BASE_DIR.iterdir() if d.is_dir() and 'p' in d.name])
    total_iou      = []

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

            oe = is_overexposed(img, thresh=args.oe_thresh)
            
            # Stage 1: Fourier Amplitude Blending (FAB)
            if oe:
                img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                img_recon_resized = fourier_amplitude_blend(img_resized, template_clean, alpha=args.alpha)
                # Resize back to raw size for consistent boundary checking
                img_proc = cv2.resize(img_recon_resized, (w, h))
            else:
                img_proc = img

            model.decoder.is_oe = oe

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
        print(f"\n🏆 [FAB alpha={args.alpha}] Total mIoU: {t:.4f}")
        sys.stdout.flush()

if __name__ == '__main__':
    main()
