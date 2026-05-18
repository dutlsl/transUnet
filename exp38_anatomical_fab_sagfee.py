"""
exp38_anatomical_fab_sagfee.py
──────────────────────────────────────────────────────────────────────────────
Unified Framework: Two-Stage Overexposure Correction via Cross-Eye Anatomical
Fourier Amplitude Blending (Anatomical-FAB) & Spatially-Varying, Self-Attention
Guided Fourier Edge Emphasis (SAGFEE).

Stage 1 (Input-Level):
  - Detect overexposure.
  - If overexposed:
      1. Perform raw-scale inpainting on 640x480.
      2. Retrieve clean contralateral eye template amplitude spectrum (horizontally flipped).
      3. Perform Anatomical Fourier Amplitude Blending (Anatomical-FAB) to restore the dark pupil texture:
             amp_blend = alpha * amp_clean + (1 - alpha) * amp_oe
             I_recon = iFFT(amp_blend * exp(1j * phase_oe))
Stage 2 (Decoder Feature-Level):
  - Apply Gaussian Skip Filter to suppress eyelash high frequencies.
  - Apply Spatially-Varying, Self-Attention Guided Fourier Edge Emphasis (SAGFEE) to skip[0]:
         A = Sigmoid(gamma * (Act - Mean(Act)) / Std(Act))
         F_adaptive = A * F_HP + (1 - A) * F
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

# ── Anatomical Fourier Amplitude Blending (Anatomical-FAB) ───────────────────
def fourier_amplitude_blend(img_oe, amp_clean, alpha=0.3):
    f_oe = np.fft.fft2(img_oe)
    amp_oe, phase_oe = np.abs(f_oe), np.angle(f_oe)
    amp_blend = alpha * amp_clean + (1.0 - alpha) * amp_oe
    f_recon = amp_blend * np.exp(1j * phase_oe)
    img_recon = np.real(np.fft.ifft2(f_recon))
    return np.clip(img_recon, 0, 255).astype(np.uint8)

# ── Anatomical Template Selection with Cross-Eye Symmetry ────────────────────
def get_clean_template_amplitude(case_name, base_dir, oe_thresh=0.08, bright_val=240):
    p_id, eye = case_name.split('-')
    other_eye = 'left' if eye == 'right' else 'right'
    other_case_name = f"{p_id}-{other_eye}"
    
    # 1. Try finding a clean normal-exposure frame in the current case
    frames_dir = base_dir / case_name / "frames"
    if frames_dir.exists():
        frame_files = sorted([f for f in frames_dir.iterdir() if f.name.endswith('-eye.png')])
        for ff in frame_files:
            img = cv2.imread(str(ff), cv2.IMREAD_GRAYSCALE)
            if img is not None and not is_overexposed(img, thresh=oe_thresh, bright_val=bright_val):
                f_cl = np.fft.fft2(img)
                return np.abs(f_cl), ff.name, False
                
    # 2. Try finding a clean normal-exposure frame in the contralateral eye
    other_frames_dir = base_dir / other_case_name / "frames"
    if other_frames_dir.exists():
        other_files = sorted([f for f in other_frames_dir.iterdir() if f.name.endswith('-eye.png')])
        for ff in other_files:
            img = cv2.imread(str(ff), cv2.IMREAD_GRAYSCALE)
            if img is not None and not is_overexposed(img, thresh=oe_thresh, bright_val=bright_val):
                img_flipped = cv2.flip(img, 1)
                f_cl = np.fft.fft2(img_flipped)
                return np.abs(f_cl), f"flipped contralateral {other_case_name}/{ff.name}", True
                
    # 3. Fallback to first frame of current case
    if frames_dir.exists():
        frame_files = sorted([f for f in frames_dir.iterdir() if f.name.endswith('-eye.png')])
        if frame_files:
            img = cv2.imread(str(frame_files[0]), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                return np.abs(np.fft.fft2(img)), f"fallback {frame_files[0].name}", False
                
    return None, None, False

# ── Stage 2: Spatially-Varying Self-Attention Guided Fourier Edge Emphasis ─────
def sagfee_filter(feat, d_hp_frac=0.40, hp_gain=0.6, gamma=2.0):
    B, C, H, W = feat.shape
    device = feat.device
    cy, cx = H // 2, W // 2
    
    act = torch.mean(torch.abs(feat), dim=1, keepdim=True) # [B, 1, H, W]
    act_mean = act.mean(dim=(-2, -1), keepdim=True)
    act_std = act.std(dim=(-2, -1), keepdim=True) + 1e-6
    attn_map = torch.sigmoid(gamma * (act - act_mean) / act_std)
    
    yy = torch.arange(H, device=device).float().view(H, 1).expand(H, W)
    xx = torch.arange(W, device=device).float().view(1, W).expand(H, W)
    dist_sq = (yy - cy)**2 + (xx - cx)**2
    
    d_hp = d_hp_frac * min(H, W)
    ghp_mask = 1.0 - torch.exp(-dist_sq / (2 * (d_hp**2)))
    ghfe_mask = 1.0 + hp_gain * ghp_mask
    ghfe_mask_shifted = torch.fft.ifftshift(ghfe_mask).unsqueeze(0).unsqueeze(0)
    
    feat_fft = torch.fft.fft2(feat)
    feat_fft_hp = feat_fft * ghfe_mask_shifted
    feat_hp = torch.fft.ifft2(feat_fft_hp).real
    
    feat_adaptive = attn_map * feat_hp + (1.0 - attn_map) * feat
    return feat_adaptive

# ── Patched Decoder (keeps baseline skip blur!) ───────────────────────────────
def _mk(sigma):
    k = int(4*sigma+0.5); return k+1 if k%2==0 else k

def patched_decoder(self, hidden_states, features=None):
    sigma0, sigma1 = 1.0, 0.5
    is_oe     = getattr(self, 'is_oe',     False)
    d_hp_frac = getattr(self, 'd_hp_frac', 0.40)
    hp_gain   = getattr(self, 'hp_gain',   0.6)
    gamma     = getattr(self, 'gamma',     2.0)

    if features is not None:
        features = list(features)
        if len(features) > 0:
            features[0] = TF.gaussian_blur(features[0], [_mk(sigma0)]*2, [sigma0]*2)
        if len(features) > 1:
            features[1] = TF.gaussian_blur(features[1], [_mk(sigma1)]*2, [sigma1]*2)
            
        # Feature-Level Stage-2: Spatially-varying SAGFEE on skip[0]
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

def get_model(device, d_hp, hp_gain, gamma):
    cfg = CONFIGS_ViT_seg['R50-ViT-B_16']
    cfg.n_classes = NUM_CLASSES; cfg.n_skip = 3
    if cfg.patches.get('grid'):
        cfg.patches.grid = (IMG_SIZE//16, IMG_SIZE//16)
    m = VisionTransformer(cfg, img_size=IMG_SIZE, num_classes=NUM_CLASSES, vis=False)
    m.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
    
    m.decoder.d_hp_frac = d_hp
    m.decoder.hp_gain   = hp_gain
    m.decoder.gamma     = gamma
    m.decoder.is_oe     = False
    m.decoder.forward   = types.MethodType(patched_decoder, m.decoder)
    
    return m.to(device).eval()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--alpha', type=float, default=0.3, help='Anatomical-FAB blend ratio')
    parser.add_argument('--d_hp', type=float, default=0.40, help='SAGFEE cutoff')
    parser.add_argument('--hp_gain', type=float, default=0.6, help='SAGFEE gain boost')
    parser.add_argument('--gamma', type=float, default=2.0, help='Self-attention scaling')
    parser.add_argument('--oe_thresh', type=float, default=0.08)
    parser.add_argument('--bright_val', type=int, default=240)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"==========================================================")
    print(f"🚀 Anatomical-FAB-SAGFEE Unified Framework | {device}")
    print(f"   Input FAB alpha: {args.alpha}")
    print(f"   Feature SAGFEE: cutoff={args.d_hp}, gain={args.hp_gain}, gamma={args.gamma}")
    print(f"==========================================================")
    sys.stdout.flush()

    model = get_model(device, args.d_hp, args.hp_gain, args.gamma)
    case_dirs = sorted([d for d in BASE_DIR.iterdir() if d.is_dir() and 'p' in d.name])
    total_iou = []

    for case_dir in case_dirs:
        case_name = case_dir.name
        frames_dir = case_dir / "frames"
        gt_path = case_dir / "pupil-ellipses.txt"
        if not frames_dir.exists() or not gt_path.exists(): continue

        gt_dict = load_gt(gt_path)
        frame_files = sorted([f for f in frames_dir.iterdir() if f.name.endswith('-eye.png')],
                             key=lambda x: int(x.name.split('-')[0]))
        
        # Select clean template amplitude using Cross-Eye Symmetry
        amp_clean, tpl_name, is_flipped = get_clean_template_amplitude(
            case_name, BASE_DIR, oe_thresh=args.oe_thresh, bright_val=args.bright_val)
        
        print(f"🎯 [{case_name}] Template: {tpl_name} (Flipped={is_flipped})")
        sys.stdout.flush()

        case_iou = []
        print(f"\n📁 {case_name}")
        sys.stdout.flush()

        for i, ff in enumerate(frame_files):
            fidx = int(ff.name.split('-')[0])
            if fidx not in gt_dict: continue
            img = cv2.imread(str(ff), cv2.IMREAD_GRAYSCALE)
            if img is None: continue
            h, w = img.shape

            oe = is_overexposed(img, thresh=args.oe_thresh, bright_val=args.bright_val)
            if oe:
                # Stage 1: Input-Level Glare Inpaint & Anatomical-FAB
                img_proc = inpaint_glare_raw(img, bright_val=args.bright_val, dilate_k=5, radius=7)
                if amp_clean is not None:
                    img_proc = fourier_amplitude_blend(img_proc, amp_clean, alpha=args.alpha)
            else:
                img_proc = img

            # Pass overexposure flag to the model decoder
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
            m_iou = sum(case_iou) / len(case_iou)
            print(f"🎯 [{case_name}] mIoU: {m_iou:.4f}")
            sys.stdout.flush()
            total_iou.extend(case_iou)

    if total_iou:
        t = sum(total_iou) / len(total_iou)
        print(f"\n🏆 [Unified FAB-SAGFEE alpha={args.alpha}] Total mIoU: {t:.4f}")
        sys.stdout.flush()

if __name__ == '__main__':
    main()
