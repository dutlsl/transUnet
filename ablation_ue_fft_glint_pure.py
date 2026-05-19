"""
ablation_ue_fft_glint_pure.py
----------------------------------------------------------
공간 마스크를 전혀 사용하지 않고, 이미지 전체에 주파수 영역의
Low-Pass Filter를 적용하여 고주파 글린트를 날려버리는 '순수 FFT' 실험.

논리: 큰 동공은 저주파 성분이므로, LPF를 걸어도 형태가 보존됨.
"""
import cv2, torch, numpy as np, types, re, csv
from pathlib import Path
import torchvision.transforms.functional as TF
from networks.vit_seg_modeling import VisionTransformer, CONFIGS as CONFIGS_ViT_seg

WEIGHTS_PATH = "./models_transunet/best_model.pth"
LPW_DIR      = Path("./LPW")
GT_DIR       = Path("./Pupils_in_the_wild_improved")
OUT_DIR      = Path("./ablation_ue_fft_pure")
IMG_SIZE     = 224
NUM_CLASSES  = 4
PUPIL_ID     = 3
DEVICE       = None  # set at runtime

UE_THRESH = 110
ZOOM_SCALE = 0.65

# ── helpers ─────────────────────────────────────────────────────────────────
def _make_blur(sigma):
    k = int(4*sigma+0.5); k += (k%2==0); return max(k, 3)

def patched_forward(self, hidden_states, features=None):
    if features is not None:
        features = list(features)
        for i, s in enumerate([1.0, 0.5]):
            if i < len(features):
                k = _make_blur(s)
                features[i] = TF.gaussian_blur(features[i], [k,k], [s,s])
    B,n,h_ = hidden_states.size(); hw=int(n**0.5)
    x = hidden_states.permute(0,2,1).contiguous().view(B,h_,hw,hw)
    x = self.conv_more(x)
    for i,blk in enumerate(self.blocks):
        skip = features[i] if (features and i<self.config.n_skip) else None
        x = blk(x, skip=skip)
    return x

def get_model():
    cfg = CONFIGS_ViT_seg['R50-ViT-B_16']
    cfg.n_classes=NUM_CLASSES; cfg.n_skip=3
    cfg.patches['grid']=(IMG_SIZE//16, IMG_SIZE//16)
    m = VisionTransformer(cfg, img_size=IMG_SIZE, num_classes=NUM_CLASSES)
    m.load_state_dict(torch.load(WEIGHTS_PATH, map_location=DEVICE))
    m.decoder.forward = types.MethodType(patched_forward, m.decoder)
    return m.to(DEVICE).eval()

def ellipse_post(pred, pid=PUPIL_ID, min_pts=5):
    binary=(pred==pid).astype(np.uint8)
    k=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(13,13))
    closed=cv2.morphologyEx(binary,cv2.MORPH_CLOSE,k)
    cnts,_=cv2.findContours(closed,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    if not cnts: return pred
    lg=max(cnts,key=cv2.contourArea); la=cv2.contourArea(lg)
    if la==0 or len(lg)<min_pts: return pred
    Ml=cv2.moments(lg); cx0=Ml['m10']/Ml['m00']; cy0=Ml['m01']/Ml['m00']
    pts=[lg]
    for c in cnts:
        if c is lg: continue
        if cv2.contourArea(c)<la*0.10: continue
        M=cv2.moments(c)
        if M['m00']!=0 and np.hypot(M['m10']/M['m00']-cx0,M['m01']/M['m00']-cy0)>50: continue
        pts.append(c)
    all_pts=np.vstack(pts)
    if len(all_pts)<min_pts: return pred
    ell=cv2.fitEllipse(all_pts)
    if ell[1][0]<=0 or ell[1][1]<=0: return pred
    res=pred.copy(); res[res==pid]=0
    try: cv2.ellipse(res,ell,pid,-1)
    except: pass
    return res

def calc_iou(pred, gt):
    pb=(pred==PUPIL_ID); gb=(gt==255)
    union=np.logical_or(pb,gb).sum()
    if union==0: return 1.0 if pb.sum()==0 else 0.0
    return np.logical_and(pb,gb).sum()/union

# ── NEW: Pure FFT Gaussian Low-Pass Filter ──────────────────────────────────
def ue_fft_pure_lowpass(img_gray, cutoff_radius=20):
    """
    이미지 전체에 대해 FFT를 수행하고 고주파(글린트)를 억제하는 가우시안 LPF 적용.
    공간 마스크 일절 없음.
    """
    H, W = img_gray.shape
    f = np.fft.fft2(img_gray.astype(np.float32))
    fshift = np.fft.fftshift(f)
    
    cy, cx = H//2, W//2
    yy, xx = np.ogrid[:H, :W]
    
    # 가우시안 저주파 필터 생성
    dist = (yy - cy)**2 + (xx - cx)**2
    # cutoff_radius가 작을수록 더 많이 뭉개짐
    kernel = np.exp(-dist / (2 * (cutoff_radius**2)))
    
    fshift_filtered = fshift * kernel
    img_back = np.abs(np.fft.ifft2(np.fft.ifftshift(fshift_filtered)))
    return np.clip(img_back, 0, 255).astype(np.uint8)

def ue_zoom_out(img_gray, scale=0.65):
    H, W = img_gray.shape
    nH, nW = int(H*scale), int(W*scale)
    resized = cv2.resize(img_gray, (nW, nH), interpolation=cv2.INTER_AREA)
    bg = int(np.median(img_gray))
    canvas = np.full((H, W), bg, dtype=np.uint8)
    y0=(H-nH)//2; x0=(W-nW)//2
    canvas[y0:y0+nH, x0:x0+nW] = resized
    return canvas, y0, x0, nH, nW

def infer(model, img_gray):
    rgb = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)
    t = torch.from_numpy(cv2.resize(rgb,(IMG_SIZE,IMG_SIZE))).float().permute(2,0,1).unsqueeze(0)/255.0
    with torch.no_grad():
        pred = model(t.to(DEVICE)).argmax(1).squeeze(0).cpu().numpy().astype(np.uint8)
    return ellipse_post(pred)

def find_gt(folder_idx, file_idx):
    for f in GT_DIR.rglob(f"folder-{folder_idx}_file-{file_idx}_pupil.mp4"):
        return f
    return None

# 파라미터 스윕: 컷오프 반경 (작을수록 고주파를 더 많이 날림)
CONFIGS = [
    {"name": "baseline_no_ue", "fft_pure": False, "ue_gate": False},
    {"name": "fft_only_r20",   "fft_pure": True,  "ue_gate": False, "cutoff": 20},
    {"name": "fft_only_r40",   "fft_pure": True,  "ue_gate": False, "cutoff": 40},
    {"name": "fft_only_r60",   "fft_pure": True,  "ue_gate": False, "cutoff": 60},
    {"name": "fft_only_r80",   "fft_pure": True,  "ue_gate": False, "cutoff": 80},
    {"name": "fft_only_r100",  "fft_pure": True,  "ue_gate": False, "cutoff": 100},
    {"name": "fft_only_r120",  "fft_pure": True,  "ue_gate": False, "cutoff": 120},
    {"name": "fft_only_r140",  "fft_pure": True,  "ue_gate": False, "cutoff": 140},
]

def run_ablation(folder_idx, file_idx, frame_start=None, frame_end=None):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model = get_model()
    results = []

    raw_path = LPW_DIR / str(folder_idx) / f"{file_idx}.avi"
    gt_path  = find_gt(folder_idx, file_idx)
    if not raw_path.exists() or gt_path is None:
        print(f"[SKIP] LPW/{folder_idx}/{file_idx}.avi or GT not found"); return

    print(f"\n{'='*60}\nTarget: LPW/{folder_idx}/{file_idx}.avi (Frames: {frame_start} to {frame_end})\n{'='*60}")

    for cfg in CONFIGS:
        cap_r = cv2.VideoCapture(str(raw_path))
        cap_g = cv2.VideoCapture(str(gt_path))
        
        if frame_start is not None:
            cap_r.set(cv2.CAP_PROP_POS_FRAMES, frame_start)
            cap_g.set(cv2.CAP_PROP_POS_FRAMES, frame_start)
            frame_idx = frame_start
        else:
            frame_idx = 0
            
        ious = []

        while True:
            ret_r, fr = cap_r.read()
            ret_g, fg = cap_g.read()
            if not (ret_r and ret_g): break

            H, W = fr.shape[:2]
            gray   = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY) if fr.ndim==3 else fr
            gt_bin = cv2.cvtColor(fg, cv2.COLOR_BGR2GRAY) if fg.ndim==3 else fg
            gt_bin = cv2.resize(gt_bin, (W,H), interpolation=cv2.INTER_NEAREST)
            _, gt_bin = cv2.threshold(gt_bin, 127, 255, cv2.THRESH_BINARY)

            # 오리지널 게이트 로직 그대로 유지
            is_ue = gray.mean() < UE_THRESH
            img   = gray.copy()

            # 1. FFT 처리는 프레임이 작아지기 이전에 수행 (순수 주파수 필터링)
            if is_ue and cfg.get("fft_pure", False):
                img = ue_fft_pure_lowpass(img, cutoff_radius=cfg["cutoff"])

            # 2. 그 이후에 Zoom-Out 수행
            if is_ue and cfg.get("ue_gate", False):
                img, y0, x0, nH, nW = ue_zoom_out(img, ZOOM_SCALE)

            pred = infer(model, img)
            
            # Zoom-out 마스크 역변환
            if is_ue and cfg.get("ue_gate", False):
                pred_canvas = cv2.resize(pred, (W, H), interpolation=cv2.INTER_NEAREST)
                pred_region = pred_canvas[y0:y0+nH, x0:x0+nW]
                pred = cv2.resize(pred_region, (W, H), interpolation=cv2.INTER_NEAREST)
            else:
                pred = cv2.resize(pred, (W, H), interpolation=cv2.INTER_NEAREST)

            ious.append(calc_iou(pred, gt_bin))
            frame_idx += 1
            
            if frame_end is not None and frame_idx >= frame_end:
                break

        cap_r.release(); cap_g.release()
        mIoU = float(np.mean(ious)) if ious else 0.0
        print(f"  [{cfg['name']:25s}]  mIoU={mIoU:.4f}  ({len(ious)} frames)")
        results.append({"config": cfg["name"], "folder": folder_idx, "video": file_idx, "mIoU": mIoU, "frames": len(ious)})

    out_csv = OUT_DIR / f"ablation_pure_f{folder_idx}_v{file_idx}_{frame_start}_{frame_end}.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["config","folder","video","mIoU","frames"])
        w.writeheader(); w.writerows(results)
    print(f"\n결과 저장 완료: {out_csv}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=int, default=11)
    parser.add_argument("--video",  type=int, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--frame_start", type=int, default=None)
    parser.add_argument("--frame_end", type=int, default=None)
    args = parser.parse_args()

    DEVICE = torch.device(args.device)
    run_ablation(folder_idx=args.folder, file_idx=args.video, 
                 frame_start=args.frame_start, frame_end=args.frame_end)

