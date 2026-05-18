"""
eval_lpw_comprehensive.py
──────────────────────────────────────────────────────────────────────────────
Comprehensive LPW evaluation script utilizing dual-gate adaptive preprocessing:
1. Overexposure (OE): Glare Inpaint + FAB (Same-Video Template) + SAGFEE (gain=0.3)
2. Underexposure (UE): Zoom-Out Spatial Scaling (scale=0.65, padded)
3. Normal: Pristine Skip-Filter + Ellipse Postprocessing.
Supports parallel GPU execution across folders.
"""

import os, cv2, torch, numpy as np, csv, re, types, argparse
from pathlib import Path
import torchvision.transforms.functional as TF
from networks.vit_seg_modeling import VisionTransformer, CONFIGS as CONFIGS_ViT_seg

WEIGHTS_PATH = "./models_transunet/best_model.pth"
RAW_BASE_DIR = Path("./LPW")
GT_BASE_DIR  = Path("./Pupils_in_the_wild_improved")
TABLE_DIR    = Path("./LPW_tables")

IMG_SIZE = 224
NUM_CLASSES = 4
PUPIL_CLASS_ID = 3

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

# ── OE Stage 1: input-level glare inpaint on RAW image ──────────────────────────
def inpaint_glare_raw(img_gray, bright_val=240, dilate_k=5, radius=7):
    mask = ((img_gray > bright_val) * 255).astype(np.uint8)
    if dilate_k > 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_k, dilate_k))
        mask = cv2.dilate(mask, k)
    return cv2.inpaint(img_gray, mask, radius, cv2.INPAINT_TELEA)

def is_overexposed(img_gray, thresh=0.08, bright_val=240):
    return (img_gray > bright_val).mean() > thresh

def is_underexposed(img_gray, thresh=110):
    return img_gray.mean() < thresh

def fourier_amplitude_blend(img_oe, amp_clean, alpha=0.3):
    f_oe = np.fft.fft2(img_oe)
    amp_oe, phase_oe = np.abs(f_oe), np.angle(f_oe)
    amp_blend = alpha * amp_clean + (1.0 - alpha) * amp_oe
    f_recon = amp_blend * np.exp(1j * phase_oe)
    img_recon = np.real(np.fft.ifft2(f_recon))
    return np.clip(img_recon, 0, 255).astype(np.uint8)

def get_clean_template_same_video(video_path, oe_thresh=0.08, bright_val=240):
    cap = cv2.VideoCapture(str(video_path))
    amp_clean = None
    while True:
        ret, frame = cap.read()
        if not ret: break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        if not is_overexposed(gray, thresh=oe_thresh, bright_val=bright_val):
            amp_clean = np.abs(np.fft.fft2(gray))
            break
    cap.release()
    return amp_clean

# ── OE Stage 2: Spatially-Varying Self-Attention Guided Fourier Edge Emphasis ─────
def sagfee_filter(feat, d_hp_frac=0.40, hp_gain=0.3, gamma=2.0):
    B, C, H, W = feat.shape
    device = feat.device
    cy, cx = H // 2, W // 2
    
    act = torch.mean(torch.abs(feat), dim=1, keepdim=True)
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

# ── Patched Decoder (keeps baseline skip blur + supports dynamic gates) ────────
def patched_decoder_forward(self, hidden_states, features=None):
    sigma0 = getattr(self, 'filter_sigma0', 1.0)
    sigma1 = getattr(self, 'filter_sigma1', 0.5)
    is_oe     = getattr(self, 'is_oe',     False)
    is_ue     = getattr(self, 'is_ue',     False)
    d_hp_frac = getattr(self, 'd_hp_frac', 0.40)
    hp_gain   = getattr(self, 'hp_gain',   0.3)
    gamma     = getattr(self, 'gamma',     2.0)

    if features is not None:
        features = list(features)
        if sigma0 > 0 and len(features) > 0:
            k0 = _make_blur(sigma0)
            features[0] = TF.gaussian_blur(features[0], kernel_size=[k0, k0], sigma=[sigma0, sigma0])
        if sigma1 > 0 and len(features) > 1:
            k1 = _make_blur(sigma1)
            features[1] = TF.gaussian_blur(features[1], kernel_size=[k1, k1], sigma=[sigma1, sigma1])
            
        # Spatially-varying SAGFEE on skip[0] under overexposure
        if is_oe and len(features) > 0:
            features[0] = sagfee_filter(features[0], d_hp_frac=d_hp_frac, hp_gain=hp_gain, gamma=gamma)

    B, n_patch, hidden = hidden_states.size()
    h, w = int(np.sqrt(n_patch)), int(np.sqrt(n_patch))
    x = hidden_states.permute(0, 2, 1)
    x = x.contiguous().view(B, hidden, h, w)
    x = self.conv_more(x)
    for i, decoder_block in enumerate(self.blocks):
        skip = features[i] if (features is not None and i < self.config.n_skip) else None
        x = decoder_block(x, skip=skip)
    return x

def get_model(device, sigma0=1.0, sigma1=0.5, d_hp=0.40, hp_gain=0.3, gamma=2.0):
    config_vit = CONFIGS_ViT_seg['R50-ViT-B_16']
    config_vit.n_classes = NUM_CLASSES
    config_vit.n_skip = 3
    if config_vit.patches.get('grid') is not None:
        config_vit.patches.grid = (int(IMG_SIZE / 16), int(IMG_SIZE / 16))
    model = VisionTransformer(config_vit, img_size=IMG_SIZE, num_classes=NUM_CLASSES)
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
    
    model.decoder.filter_sigma0 = sigma0
    model.decoder.filter_sigma1 = sigma1
    model.decoder.d_hp_frac     = d_hp
    model.decoder.hp_gain       = hp_gain
    model.decoder.gamma         = gamma
    model.decoder.is_oe         = False
    model.decoder.is_ue         = False
    model.decoder.forward       = types.MethodType(patched_decoder_forward, model.decoder)
    
    model.to(device)
    model.eval()
    return model

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
            cx = M["m10"] / M["m00"]
            cy = M["m01"] / M["m00"]
            dist = np.sqrt((cx - cx_large)**2 + (cy - cy_large)**2)
            if dist > 50: continue
        valid_points.append(cnt)
        
    all_points = np.vstack(valid_points)
    if len(all_points) < min_points:
        return pred_mask
        
    ellipse = cv2.fitEllipse(all_points)
    if ellipse[1][0] <= 0 or ellipse[1][1] <= 0:
        return pred_mask
        
    result = pred_mask.copy()
    result[result == pupil_id] = 0
    try:
        cv2.ellipse(result, ellipse, pupil_id, -1)
    except cv2.error:
        return pred_mask
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--start_folder', type=int, default=1)
    parser.add_argument('--end_folder', type=int, default=22)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--alpha', type=float, default=0.3, help='Anatomical-FAB alpha')
    parser.add_argument('--hp_gain', type=float, default=0.3, help='SAGFEE gain boost')
    parser.add_argument('--scale_factor', type=float, default=0.65, help='Zoom-out scale under underexposure')
    parser.add_argument('--ue_thresh', type=int, default=110, help='Underexposure threshold')
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"==========================================================")
    print(f"🚀 LPW Dual-Gate (OE & UE) Comprehensive Eval | {device}")
    print(f"   Folders: {args.start_folder} to {args.end_folder}")
    print(f"   FAB alpha: {args.alpha}, SAGFEE gain: {args.hp_gain}")
    print(f"   Zoom-out scale: {args.scale_factor}, UE thresh: {args.ue_thresh}")
    print(f"==========================================================")

    model = get_model(device, sigma0=1.0, sigma1=0.5, d_hp=0.40, hp_gain=args.hp_gain, gamma=2.0)
    gt_mapping = build_gt_mapping(GT_BASE_DIR)
    
    suffix = f"gpu{args.gpu}_f{args.start_folder}_to_f{args.end_folder}"
    frame_csv_path = TABLE_DIR / f"lpw_comprehensive_{suffix}_frames.csv"
    compact_csv_path = TABLE_DIR / f"lpw_comprehensive_{suffix}_compact.csv"
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    folder_list = list(range(args.start_folder, args.end_folder + 1))
    total_iou_all, total_dice_all = [], []

    with open(frame_csv_path, mode='w', newline='') as f_csv, open(compact_csv_path, mode='w', newline='') as c_csv:
        frame_writer = csv.writer(f_csv)
        frame_writer.writerow(['Folder', 'Video', 'Frame', 'IoU', 'Dice'])
        
        compact_writer = csv.writer(c_csv)
        compact_writer.writerow(['Folder', 'Video', 'mIoU', 'mDice'])

        with torch.no_grad():
            for folder_idx in folder_list:
                folder_dir = RAW_BASE_DIR / str(folder_idx)
                raw_videos = list(folder_dir.glob("*.avi")) if folder_dir.exists() else []
                
                if not raw_videos:
                    print(f"[{folder_idx}] No raw videos found.")
                    continue
                    
                folder_iou, folder_dice = [], []
                
                for raw_path in sorted(raw_videos):
                    if raw_path.name.startswith("._"): continue
                    try:
                        file_idx = int(raw_path.stem)
                    except ValueError: continue
                        
                    key = f"{folder_idx}_{file_idx}"
                    if key not in gt_mapping: continue
                    gt_path = gt_mapping[key]
                    
                    # Pre-extract overexposure template amplitude
                    amp_clean = get_clean_template_same_video(raw_path, oe_thresh=0.08, bright_val=240)
                    if amp_clean is not None:
                        print(f"  [Folder {folder_idx} | {raw_path.name}] Clean template spectrum successfully extracted.")
                    
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

                        # ── Detect Exposure State (OE, UE, Normal) ──
                        oe = is_overexposed(gray, thresh=0.08, bright_val=240)
                        ue = is_underexposed(gray, thresh=args.ue_thresh)
                        
                        img_proc = gray.copy()
                        
                        if oe:
                            # Stage 1: Overexposure Glare Inpainting & Fourier Amplitude Blending
                            img_proc = inpaint_glare_raw(img_proc, bright_val=240, dilate_k=5, radius=7)
                            if amp_clean is not None:
                                img_proc = fourier_amplitude_blend(img_proc, amp_clean, alpha=args.alpha)
                        
                        # Set dynamic gates in the decoder
                        model.decoder.is_oe = oe
                        model.decoder.is_ue = ue

                        rgb = cv2.cvtColor(img_proc, cv2.COLOR_GRAY2RGB)
                        
                        # Stage 2: Underexposure Spatial Zoom-Out Scaling
                        if ue and args.scale_factor < 1.0:
                            sw = int(IMG_SIZE * args.scale_factor)
                            sh = int(IMG_SIZE * args.scale_factor)
                            resized_small = cv2.resize(rgb, (sw, sh))
                            
                            pad_val = int(img_proc.mean())
                            padded = np.ones((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8) * pad_val
                            dy = (IMG_SIZE - sh) // 2
                            dx = (IMG_SIZE - sw) // 2
                            padded[dy:dy+sh, dx:dx+sw] = resized_small
                            
                            tensor = torch.from_numpy(padded).float().permute(2, 0, 1).unsqueeze(0) / 255.0
                        else:
                            resized = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE))
                            tensor = torch.from_numpy(resized).float().permute(2, 0, 1).unsqueeze(0) / 255.0

                        tensor = tensor.to(device)
                        logits = model(tensor)
                        pred = logits.argmax(1).squeeze(0).cpu().numpy().astype(np.uint8)

                        if ue and args.scale_factor < 1.0:
                            # Re-expand the mask if Zoom-Out was applied
                            dy = (IMG_SIZE - sh) // 2
                            dx = (IMG_SIZE - sw) // 2
                            cropped_pred = pred[dy:dy+sh, dx:dx+sw]
                            pred = cv2.resize(cropped_pred, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)

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

                    cap_raw.release()
                    cap_gt.release()
                    
                    if vid_iou:
                        m_iou = np.mean(vid_iou)
                        m_dice = np.mean(vid_dice)
                        print(f"  [Folder {folder_idx} | {raw_path.stem}.avi] mIoU={m_iou:.4f} mDice={m_dice:.4f} ({len(vid_iou)} frames)")
                        folder_iou.extend(vid_iou)
                        folder_dice.extend(vid_dice)
                        total_iou_all.extend(vid_iou)
                        total_dice_all.extend(vid_dice)
                        compact_writer.writerow([folder_idx, raw_path.stem, f"{m_iou:.4f}", f"{m_dice:.4f}"])
                        
                if folder_iou:
                    fm_iou = np.mean(folder_iou)
                    fm_dice = np.mean(folder_dice)
                    print(f"[Folder {folder_idx} Total] mIoU={fm_iou:.4f} mDice={fm_dice:.4f}")
                    compact_writer.writerow([folder_idx, 'FOLDER_TOTAL', f"{fm_iou:.4f}", f"{fm_dice:.4f}"])
                    
        if total_iou_all:
            tm_iou = np.mean(total_iou_all)
            tm_dice = np.mean(total_dice_all)
            print(f"\n[Folders {args.start_folder}-{args.end_folder} Total] mIoU={tm_iou:.4f} mDice={tm_dice:.4f}")
            compact_writer.writerow(['ALL', 'TOTAL', f"{tm_iou:.4f}", f"{tm_dice:.4f}"])

if __name__ == '__main__':
    main()
