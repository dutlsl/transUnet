"""
exp37_cgf.py
──────────────────────────────────────────────────────────────────────────────
Two-Stage OE Correction with Contour Glare Filtering (CGF) Ellipse Fitting
& Raw-Scale Inpainting:
  Stage 1 (Space Preprocessing):
     - For any overexposed frame (OE=True):
         Inpaint the saturated glare on the raw 640x480 image (bright_val=240,
         dilate_k=5, radius=7).
  Stage 2 (Contour Glare Filtering Post-Processing at 224x224):
     - Post-processing runs at the 224x224 scale, maintaining identical parameters
       to the highly optimized baseline (13x13 kernel, 50-pixel distance threshold).
     - For overexposed frames, CGF filters the boundary points of the predicted pupil mask P,
       discarding any point lying inside the 224x224 overexposure mask.
     - We then fit the ellipse purely to the unoccluded boundary points.
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

def ellipse_postprocess_cgf(pred, oe_mask_224=None, cls=3):
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
            if np.hypot(cx-cxl, cy-cyl) > 50: # Match baseline 50 pixels!
                continue
        valid.append(c)
    pts = np.vstack(valid) # [N, 1, 2]
    
    # Apply Contour Glare Filtering (CGF) if oe_mask is provided
    if oe_mask_224 is not None:
        clean_pts = []
        for pt in pts:
            x, y = int(pt[0][0]), int(pt[0][1])
            if 0 <= x < 224 and 0 <= y < 224:
                if oe_mask_224[y, x] == 0:
                    clean_pts.append(pt)
        if len(clean_pts) >= 5:
            pts = np.array(clean_pts)
            
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
    return cv2.inpaint(img_gray, mask, radius, cv2.INPAINT_TELEA), mask

def is_overexposed(img_gray, thresh=0.08, bright_val=240):
    return (img_gray > bright_val).mean() > thresh

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
    parser.add_argument('--oe_thresh', type=float, default=0.08)
    parser.add_argument('--bright_val', type=int, default=240)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"==========================================================")
    print(f"🚀 Contour Glare Filtering (CGF) [bright_val={args.bright_val}]")
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

            # Space-Domain Glare Inpainting & Mask Extraction
            oe = is_overexposed(img, thresh=args.oe_thresh, bright_val=args.bright_val)
            if oe:
                img_proc, oe_mask = inpaint_glare_raw(img, bright_val=args.bright_val, dilate_k=5, radius=7)
                oe_mask_224 = cv2.resize(oe_mask, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)
            else:
                img_proc, oe_mask_224 = img, None

            rgb    = cv2.cvtColor(img_proc, cv2.COLOR_GRAY2RGB)
            tensor = torch.from_numpy(cv2.resize(rgb, (IMG_SIZE, IMG_SIZE))).float()
            tensor = tensor.permute(2,0,1).unsqueeze(0).div(255.0).to(device)

            with torch.no_grad():
                logits = model(tensor)

            pred      = logits.argmax(1).squeeze(0).cpu().numpy().astype(np.uint8)
            
            # Postprocess with Contour Glare Filtering (CGF) at 224x224
            pred_post = ellipse_postprocess_cgf(pred, oe_mask_224=oe_mask_224, cls=PUPIL_CLASS)

            # Resize postprocessed mask to raw scale for evaluation
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
        print(f"\n🏆 [CGF] Total mIoU: {t:.4f}")
        sys.stdout.flush()

if __name__ == '__main__':
    main()
