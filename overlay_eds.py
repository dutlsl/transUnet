#!/usr/bin/env python3
import os, argparse, time
from pathlib import Path
import numpy as np
import cv2
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from torchmetrics import JaccardIndex
from torchmetrics.classification import Accuracy

# (TransUNet)
from networks.vit_seg_modeling import VisionTransformer as ViT_seg
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg


# ---------- Dataset & helpers ----------
class OpenEDSSeg(Dataset):
    def __init__(self, ids, img_dir, lab_dir, img_ext, mask_ext, img_size):
        self.ids = ids
        self.img_dir = Path(img_dir)
        self.lab_dir = Path(lab_dir)
        self.img_ext = img_ext
        self.mask_ext = mask_ext
        self.img_size = int(img_size)

    def __len__(self): return len(self.ids)

    def __getitem__(self, idx):
        sid = self.ids[idx]
        ip = str(self.img_dir / f'{sid}{self.img_ext}')
        lp = str(self.lab_dir / f'{sid}{self.mask_ext}')

        img_bgr = cv2.imread(ip, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise FileNotFoundError(ip)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        H0, W0 = img_rgb.shape[:2]

        img_res = cv2.resize(img_rgb, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
        img_chw = np.transpose(img_res, (2, 0, 1))  # 3xHxW

        # --- mask robust loader (.png or .npy) [ONLY THIS PART CHANGED] ---
        ext = Path(lp).suffix.lower()
        if ext == '.npy':
            mask = np.load(lp, allow_pickle=False).astype(np.int64)
        elif ext in ('.png', '.jpg', '.jpeg'):
            m = cv2.imread(lp, cv2.IMREAD_UNCHANGED)
            if m is None:
                raise FileNotFoundError(lp)
            if m.ndim == 3:
                m = cv2.cvtColor(m, cv2.COLOR_BGR2GRAY)
            mask = m.astype(np.int64)
        else:
            raise ValueError(f'Unsupported label ext: {ext} (path={lp})')

        meta = {'img_id': sid, 'img_path': ip, 'H0': H0, 'W0': W0}
        return torch.from_numpy(img_chw), torch.from_numpy(mask), meta


def list_matched_ids(img_dir, img_ext, label_dir, mask_ext):
    img_ids = {p.stem for p in Path(img_dir).glob(f'*{img_ext}')}
    lab_ids = {p.stem for p in Path(label_dir).glob(f'*{mask_ext}')}
    ids = sorted(list(img_ids & lab_ids))
    if not ids:
        raise FileNotFoundError(f'No matched pairs under {img_dir} & {label_dir}')
    return ids


def split_ids(ids, tr, va, te, seed):
    import random
    s = tr + va + te
    if abs(s - 1.0) > 1e-6:
        raise ValueError('split ratios must sum to 1.0')
    rng = random.Random(seed)
    ids = ids[:]
    rng.shuffle(ids)
    n = len(ids)
    n_tr = int(round(n * tr))
    n_va = int(round(n * va))
    return ids[:n_tr], ids[n_tr:n_tr + n_va], ids[n_tr + n_va:]


# ---------- Viz ----------
PALETTE = np.array([
    [0, 0, 0],      # bg
    [255, 0, 0],    # sclera (R)
    [0, 255, 0],    # iris   (G)
    [0, 0, 255],    # pupil  (B)
], dtype=np.uint8)  # RGB (UNet++와 동일)

def decode_mask(mask2d):
    h, w = mask2d.shape
    c = min(PALETTE.shape[0], int(mask2d.max()) + 1)
    out = np.zeros((h, w, 3), dtype=np.uint8)
    for cls in range(c):
        out[mask2d == cls] = PALETTE[cls]
    return out

def overlay_image(img_bgr, color_rgb, alpha):
    # UNet++와 동일한 채도/블렌딩: RGB 팔레트를 BGR로 변환 없이 바로 가중합 (cv2는 BGR 이미지에 RGB 배열도 uint8이면 섞임)
    # 다만 보기 혼동 없게 기존 함수명 유지
    return cv2.addWeighted(img_bgr, 1.0, color_rgb, float(alpha), 0)


# ---------- Main ----------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    # data / split  (DEFAULTS UNCHANGED)
    ap.add_argument('--raw_images', type=str, default='pupil-eye-and-iris-segmentation/val/image')
    ap.add_argument('--raw_labels', type=str, default='pupil-eye-and-iris-segmentation/val/segmentation')
    ap.add_argument('--img_ext', type=str, default='.png')
    ap.add_argument('--mask_ext', type=str, default='.png')
    ap.add_argument('--split_train', type=float, default=0.002)
    ap.add_argument('--split_val', type=float, default=0.002)
    ap.add_argument('--split_test', type=float, default=0.996)
    ap.add_argument('--split_seed', type=int, default=41)

    # model (DEFAULTS UNCHANGED)
    ap.add_argument('--vit_name', type=str, default='R50-ViT-B_16')
    ap.add_argument('--img_size', type=int, default=224)
    ap.add_argument('--n_skip', type=int, default=3)
    ap.add_argument('--num_classes', type=int, default=4)
    ap.add_argument('--ckpt', type=str, default='models_transunet/best_model.pth')
    ap.add_argument('--gpus', type=str, default='0')

    # output (DEFAULTS UNCHANGED)
    ap.add_argument('--out_dir', type=str, default='models_transunet')
    ap.add_argument('--overlay_dirname', type=str, default='overlay_unseen')
    ap.add_argument('--predmask_dirname', type=str, default='pred_masks_unseen')
    ap.add_argument('--alpha', type=float, default=0.5)

    # logging (optional) (DEFAULTS UNCHANGED)
    ap.add_argument('--wandb_project', type=str, default='')
    ap.add_argument('--wandb_mode', type=str, default='disabled', choices=['online','offline','disabled'])
    ap.add_argument('--run_name', type=str, default='')

    args = ap.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # optional wandb
    use_wandb = (args.wandb_mode != 'disabled' and len(args.wandb_project) > 0)
    if use_wandb:
        import wandb, time as _t
        os.environ['WANDB_MODE'] = args.wandb_mode
        wandb.init(project=args.wandb_project,
                   name=args.run_name if args.run_name else f"transunet-eval-{int(_t.time())}",
                   config=vars(args))

    # split
    ids_all = list_matched_ids(args.raw_images, args.img_ext, args.raw_labels, args.mask_ext)
    _, _, te_ids = split_ids(ids_all, args.split_train, args.split_val, args.split_test, args.split_seed)
    print(f"[Split] test={len(te_ids)}", flush=True)

    # dataset/loader (DEFAULT num_workers=4 그대로)
    ds_te = OpenEDSSeg(te_ids, args.raw_images, args.raw_labels, args.img_ext, args.mask_ext, args.img_size)
    ld_te = DataLoader(ds_te, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

    # model config
    cfg = CONFIGS_ViT_seg[args.vit_name]
    cfg.n_classes = args.num_classes
    cfg.n_skip = args.n_skip
    if 'R50' in args.vit_name:
        patch = cfg.patches.size
        if isinstance(patch, (tuple, list)):
            patch = int(patch[0])
        else:
            patch = int(patch)
        cfg.patches.grid = (args.img_size // patch, args.img_size // patch)

    # build & load
    net = ViT_seg(cfg, img_size=args.img_size, num_classes=cfg.n_classes).to(device)
    sd = torch.load(args.ckpt, map_location='cpu')
    state = sd if (isinstance(sd, dict) and 'state_dict' not in sd) else sd.get('model', sd)
    net.load_state_dict(state, strict=False)
    net.eval()

    # model info
    n_params = sum(p.numel() for p in net.parameters())
    try:
        ckpt_mb = os.path.getsize(args.ckpt) / 1024 / 1024
    except Exception:
        ckpt_mb = float('nan')

    # dirs
    out_root = Path(args.out_dir)
    pm_dir = out_root / args.predmask_dirname
    ov_dir = out_root / args.overlay_dirname
    pm_dir.mkdir(parents=True, exist_ok=True)
    ov_dir.mkdir(parents=True, exist_ok=True)

    # torchmetrics (mIoU + per-class IoU + pixel accuracy)
    miou_metric   = JaccardIndex(task='multiclass', num_classes=args.num_classes).to(device)
    percls_metric = JaccardIndex(task='multiclass', num_classes=args.num_classes, average=None).to(device)
    pixacc_metric = Accuracy(task='multiclass', num_classes=args.num_classes).to(device)

    ce = torch.nn.CrossEntropyLoss(reduction='mean').to(device)
    losses = []

    # inference timing (forward-only)
    latencies_ms = []
    def _sync():
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    # warmup (avoid first-call CUDA overhead)
    with torch.no_grad():
        for _ in range(min(3, len(ld_te))):
            x_w, y_w, _ = next(iter(ld_te))
            x_w = x_w.to(device, non_blocking=True)
            _ = net(x_w)

    with torch.no_grad():
        for x, y_gt, meta in tqdm(ld_te, total=len(ld_te), desc='[eval+overlay]'):
            sid = meta['img_id'][0]
            ip = meta['img_path'][0]
            H0 = int(meta['H0'][0]); W0 = int(meta['W0'][0])

            # resize GT to network input and cast Long
            y_gt_res = cv2.resize(y_gt.numpy().astype(np.int64)[0], (args.img_size, args.img_size),
                                  interpolation=cv2.INTER_NEAREST)
            y = torch.from_numpy(y_gt_res).long().unsqueeze(0).to(device)  # (1,H,W)

            x = x.to(device, non_blocking=True)  # (1,3,H,W)

            # timed forward
            _sync()
            t0 = time.perf_counter()
            logits = net(x)
            _sync()
            t1 = time.perf_counter()
            latencies_ms.append((t1 - t0) * 1000.0)

            if isinstance(logits, list):
                logits = logits[-1]

            losses.append(float(ce(logits, y).item()))

            pred = logits.argmax(1)  # (1,H,W) int64

            # torchmetrics update
            miou_metric.update(pred, y)
            percls_metric.update(pred, y)
            pixacc_metric.update(pred, y)

            # save (resize to original)
            pred_np = pred[0].detach().cpu().numpy().astype(np.uint8)
            pred_res = cv2.resize(pred_np, (W0, H0), interpolation=cv2.INTER_NEAREST)

            # UNet++ 동일 팔레트/채도 오버레이
            img0 = cv2.imread(ip, cv2.IMREAD_COLOR)        # BGR
            mask_rgb = PALETTE[pred_res]                   # RGB
            overlay = cv2.addWeighted(img0, 1.0, mask_rgb, float(args.alpha), 0.0)

            cv2.imwrite(str(pm_dir / f'{sid}.png'), cv2.cvtColor(mask_rgb, cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(ov_dir / f'{sid}.jpg'), overlay)

    # compute metrics
    miou = float(miou_metric.compute().item())
    per_class_iou = [float(v) for v in percls_metric.compute().detach().cpu().tolist()]
    pixacc = float(pixacc_metric.compute().item())
    test_loss = float(np.mean(losses)) if len(losses) else float('nan')

    # latency summary
    lat_ms = np.array(latencies_ms, dtype=np.float64)
    avg_ms = float(lat_ms.mean()) if lat_ms.size else float('nan')
    p50_ms  = float(np.percentile(lat_ms, 50)) if lat_ms.size else float('nan')
    p90_ms  = float(np.percentile(lat_ms, 90)) if lat_ms.size else float('nan')
    fps = (1000.0 / avg_ms) if avg_ms and avg_ms > 0 else float('nan')

    # console summary
    print(f"n_params,{n_params}")
    print(f"model_ckpt_mb,{ckpt_mb:.2f}")
    print(f"test_loss(mean),{test_loss:.6f}")
    print(f"mIoU,{miou:.6f}")
    print(f"PixelAcc,{pixacc:.6f}")
    print(f"latency_ms_avg,{avg_ms:.2f} | p50,{p50_ms:.2f} | p90,{p90_ms:.2f} | fps,{fps:.2f}")

    # CSVs
    with open(out_root / 'test_metrics.csv', 'w', newline='') as f:
        import csv as _csv
        w = _csv.writer(f)
        w.writerow(['metric', 'value'])
        w.writerow(['n_params', n_params])
        w.writerow(['model_ckpt_mb', f'{ckpt_mb:.2f}'])
        w.writerow(['test_loss(mean)', f'{test_loss:.6f}'])
        w.writerow(['mIoU', f'{miou:.6f}'])
        w.writerow(['PixelAcc', f'{pixacc:.6f}'])
        w.writerow(['latency_ms_avg', f'{avg_ms:.2f}'])
        w.writerow(['latency_ms_p50', f'{p50_ms:.2f}'])
        w.writerow(['latency_ms_p90', f'{p90_ms:.2f}'])
        w.writerow(['throughput_fps', f'{fps:.2f}'])

    with open(out_root / 'test_class_iou.csv', 'w', newline='') as f:
        import csv as _csv
        w = _csv.writer(f)
        w.writerow(['class', 'IoU'])
        for i, v in enumerate(per_class_iou):
            w.writerow([f'class_{i}', f'{v:.6f}'])

    # wandb logging (optional)
    if use_wandb:
        wandb.log({
            'n_params': n_params,
            'model_ckpt_mb': ckpt_mb,
            'test_loss_mean': test_loss,
            'mIoU': miou,
            'PixelAcc': pixacc,
            'latency_ms_avg': avg_ms,
            'latency_ms_p50': p50_ms,
            'latency_ms_p90': p90_ms,
            'throughput_fps': fps
        })
        wandb.log({f'IoU/class_{i}': v for i, v in enumerate(per_class_iou)})
        wandb.finish()

    print(f"WROTE {out_root/'test_metrics.csv'}")
    print(f"WROTE {out_root/'test_class_iou.csv'}")
    print(f"SAVED {pm_dir} and {ov_dir}")
