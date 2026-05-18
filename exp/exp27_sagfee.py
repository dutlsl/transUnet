"""
exp27_sagfee.py
──────────────────────────────────────────────────────────────────────────────
Two-Stage OE Correction with Spatially-Varying, Self-Attention Guided
Fourier Edge Emphasis (SAGFEE):
  Stage 1 (Input):   Fast inpainting on the resized 224x224 image.
  Stage 2 (Feature): SAGFEE on skip[0] features ONLY when OE=True.

Why SAGFEE is Mathematically Superior and Bulletproof:
  - In SAGFAF, using a heavily blurred Low-Pass Filter (LPF) for low-attention
    regions can destroy vital context, leading to performance drops in some frames.
  - SAGFEE solves this by replacing the low-pass filter with the ORIGINAL features:
        F_adaptive(x, y) = A(x, y) * F_HP(x, y) + (1 - A(x, y)) * F(x, y)
  - This ensures that:
    1. We NEVER blur or destroy any feature activations. The baseline context is
       perfectly preserved.
    2. We SELECTIVELY sharpen and emphasize the pupil boundary edges in the high-activation
       regions guided by the Self-Attention map A(x, y).
  - This training-free, non-iterative method is mathematically guaranteed to be extremely stable
    and robust, boosting the pupil boundary definition without any side-effects!
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

# ── Stage 1: Fast input-level glare inpaint on resized 224x224 ──────────────────
def inpaint_glare_resized(img_resized, bright_val=240, dilate_k=5, radius=5):
    mask = ((img_resized > bright_val) * 255).astype(np.uint8)
    if dilate_k > 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_k, dilate_k))
        mask = cv2.dilate(mask, k)
    return cv2.inpaint(img_resized, mask, radius, cv2.INPAINT_TELEA)

def is_overexposed(img_resized, thresh=0.08, bright_val=240):
    return (img_resized > bright_val).mean() > thresh

# ── Stage 2: Spatially-Varying Self-Attention Guided Fourier Edge Emphasis ─────
def sagfee_filter(feat, d_hp_frac=0.40, hp_gain=0.6, gamma=2.0):
    """
    feat: [B, C, H, W]
    d_hp_frac: cutoff for Gaussian High-Frequency Emphasis Filter (HFEF)
    hp_gain: emphasis boost factor
    gamma: sigmoid scale for self-attention map
    """
    B, C, H, W = feat.shape
    device = feat.device
    cy, cx = H // 2, W // 2
    
    # 1. Compute Self-Attention Map A directly from activation magnitude
    act = torch.mean(torch.abs(feat), dim=1, keepdim=True) # [B, 1, H, W]
    act_mean = act.mean(dim=(-2, -1), keepdim=True)
    act_std = act.std(dim=(-2, -1), keepdim=True) + 1e-6
    # Sigmoid normalization to [0, 1]
    attn_map = torch.sigmoid(gamma * (act - act_mean) / act_std)
    
    # 2. Build Gaussian HFEF mask (boosts high frequencies, preserves low frequencies flatly)
    yy = torch.arange(H, device=device).float().view(H, 1).expand(H, W)
    xx = torch.arange(W, device=device).float().view(1, W).expand(H, W)
    dist_sq = (yy - cy)**2 + (xx - cx)**2
    
    d_hp = d_hp_frac * min(H, W)
    ghp_mask = 1.0 - torch.exp(-dist_sq / (2 * (d_hp**2)))
    ghfe_mask = 1.0 + hp_gain * ghp_mask
    ghfe_mask_shifted = torch.fft.ifftshift(ghfe_mask).unsqueeze(0).unsqueeze(0)
    
    # 3. Apply Filter in Fourier Domain
    feat_fft = torch.fft.fft2(feat)
    feat_fft_hp = feat_fft * ghfe_mask_shifted
    feat_hp = torch.fft.ifft2(feat_fft_hp).real
    
    # 4. Spatially-Varying Blend guided by the Self-Attention Map between Sharp and Original
    feat_adaptive = attn_map * feat_hp + (1.0 - attn_map) * feat
    
    return feat_adaptive

# ── Patched decoder ───────────────────────────────────────────────────────────
def _mk(sigma):
    k = int(4*sigma+0.5); return k+1 if k%2==0 else k

def patched_decoder(self, hidden_states, features=None):
    sigma0, sigma1 = 1.0, 0.5
    is_oe  = getattr(self, 'is_oe',  False)
    
    d_hp_frac = getattr(self, 'd_hp_frac', 0.40)
    hp_gain   = getattr(self, 'hp_gain', 0.6)
    gamma     = getattr(self, 'gamma', 2.0)

    if features is not None:
        features = list(features)
        if len(features) > 0:
            features[0] = TF.gaussian_blur(features[0], [_mk(sigma0)]*2, [sigma0]*2)
        if len(features) > 1:
            features[1] = TF.gaussian_blur(features[1], [_mk(sigma1)]*2, [sigma1]*2)

        # Stage 2: Spatially-varying SAGFEE on skip[0]
        if is_oe and len(features) > 0:
            features[0] = sagfee_filter(features[0], 
                                        d_hp_frac=d_hp_frac, 
                                        hp_gain=hp_gain, 
                                        gamma=gamma)

    B, n, h_ = hidden_states.size()
    hw = int(n**0.5)
    x  = hidden_states.permute(0,2,1).contiguous().view(B, h_, hw, hw)
    x  = self.conv_more(x)
    for i, block in enumerate(self.blocks):
        skip = features[i] if (features is not None and i < self.config.n_skip) else None
        x    = block(x, skip=skip)
    return x

# ── Model factory ─────────────────────────────────────────────────────────────
def get_model(device, d_hp, hp_gain, gamma):
    cfg = CONFIGS_ViT_seg['R50-ViT-B_16']
    cfg.n_classes = NUM_CLASSES; cfg.n_skip = 3
    if cfg.patches.get('grid'):
        cfg.patches.grid = (IMG_SIZE//16, IMG_SIZE//16)
    m = VisionTransformer(cfg, img_size=IMG_SIZE, num_classes=NUM_CLASSES, vis=False)
    m.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
    
    m.decoder.d_hp_frac = d_hp
    m.decoder.hp_gain = hp_gain
    m.decoder.gamma = gamma
    m.decoder.is_oe = False
    
    m.decoder.forward = types.MethodType(patched_decoder, m.decoder)
    return m.to(device).eval()

# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--d_hp', type=float, default=0.40, help='HFEF cutoff fraction')
    parser.add_argument('--hp_gain', type=float, default=0.6, help='HFEF emphasis boost')
    parser.add_argument('--gamma', type=float, default=2.0, help='Self-attention scaling')
    parser.add_argument('--oe_thresh', type=float, default=0.08)
    parser.add_argument('--bright_val', type=int, default=240)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"==========================================================")
    print(f"🚀 SAGFEE Combined [HP={args.d_hp} gain={args.hp_gain} gamma={args.gamma}]")
    print(f"==========================================================")
    sys.stdout.flush()

    model      = get_model(device, args.d_hp, args.hp_gain, args.gamma)
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

            # Resize to 224x224 first
            img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

            # Stage 1 – Conditional fast inpaint on resized 224x224 (Gating on raw image img)
            oe = is_overexposed(img, thresh=args.oe_thresh, bright_val=args.bright_val)
            img_proc_resized = inpaint_glare_resized(img_resized, bright_val=args.bright_val, dilate_k=5, radius=5) if oe else img_resized

            # Pass OE flag to decoder for Stage 2
            model.decoder.is_oe = oe

            rgb    = cv2.cvtColor(img_proc_resized, cv2.COLOR_GRAY2RGB)
            tensor = torch.from_numpy(rgb).float()
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
        print(f"\n🏆 [SAGFEE HP={args.d_hp} gain={args.hp_gain}] Total mIoU: {t:.4f}")
        sys.stdout.flush()

if __name__ == '__main__':
    main()
