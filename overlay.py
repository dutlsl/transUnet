# overlay.py
import os, argparse
from pathlib import Path
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from torchmetrics import JaccardIndex

from networks.vit_seg_modeling import VisionTransformer as ViT_seg
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg

# ---------- Dataset (PNG labels) ----------
class SegPairs(Dataset):
    def __init__(self, img_dir, lab_dir, img_ext, lab_ext, input_h, input_w):
        self.img_dir  = Path(img_dir)
        self.lab_dir  = Path(lab_dir)
        self.img_ext  = img_ext
        self.lab_ext  = lab_ext
        self.H = int(input_h); self.W = int(input_w)
        img_ids = {p.stem for p in self.img_dir.glob(f'*{self.img_ext}')}
        lab_ids = {p.stem for p in self.lab_dir.glob(f'*{self.lab_ext}')}
        self.ids = sorted(list(img_ids & lab_ids))
        if not self.ids:
            raise FileNotFoundError(f'No matched pairs under {img_dir} & {lab_dir}')
    def __len__(self): return len(self.ids)
    def __getitem__(self, idx):
        sid = self.ids[idx]
        ip = str(self.img_dir / f'{sid}{self.img_ext}')
        lp = str(self.lab_dir / f'{sid}{self.lab_ext}')

        img_bgr = cv2.imread(ip, cv2.IMREAD_COLOR)
        if img_bgr is None: raise FileNotFoundError(ip)
        H0, W0 = img_bgr.shape[:2]

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)/255.0
        img_res = cv2.resize(img_rgb, (self.W, self.H), interpolation=cv2.INTER_LINEAR)
        img_chw = np.transpose(img_res, (2,0,1))  # 3xH xW

        lab = cv2.imread(lp, cv2.IMREAD_UNCHANGED)
        if lab is None: raise FileNotFoundError(lp)
        if lab.ndim == 3:
            raise ValueError(f'Label must be single-channel PNG with class indices. Got 3-chan: {lp}')
        lab_res = cv2.resize(lab, (self.W, self.H), interpolation=cv2.INTER_NEAREST).astype(np.int64)

        meta = {'img_id': sid, 'img_path': ip, 'H0': H0, 'W0': W0}
        return torch.from_numpy(img_chw), torch.from_numpy(lab_res), meta

# ---------- Utils ----------
PALETTE = np.array([
    [0, 0, 0],      # bg
    [255, 0, 0],    # sclera
    [0, 255, 0],    # iris
    [0, 0, 255],    # pupil
], dtype=np.uint8)  # [MOD] unet++와 동일 팔레트로 변경 (RGB)

def decode_mask(mask2d):
    c = min(PALETTE.shape[0], int(mask2d.max())+1)
    out = np.zeros((*mask2d.shape,3), dtype=np.uint8)
    for cls in range(c): out[mask2d==cls] = PALETTE[cls]
    return out

def overlay_image(img_bgr, color_rgb, alpha):
    color_bgr = cv2.cvtColor(color_rgb, cv2.COLOR_RGB2BGR)
    return cv2.addWeighted(img_bgr, 1.0, color_bgr, float(alpha), 0)

def _resize_pos_embed(pe_ckpt: torch.Tensor, new_hw: tuple[int,int]) -> torch.Tensor:
    # pe_ckpt: (1, N_old, C), new_hw: (H_tokens, W_tokens)
    assert pe_ckpt.ndim == 3 and pe_ckpt.shape[0] == 1
    N_old, C = pe_ckpt.shape[1], pe_ckpt.shape[2]
    H_old = int(round(np.sqrt(N_old)))
    W_old = N_old // H_old
    grid_old = pe_ckpt[0].transpose(0,1).reshape(C, H_old, W_old)      # (C,H,W)
    grid_new = F.interpolate(grid_old.unsqueeze(0), size=new_hw, mode='bilinear', align_corners=False)[0]
    pe_new = grid_new.reshape(C, -1).transpose(0,1).unsqueeze(0)       # (1, Hn*Wn, C)
    return pe_new

def load_ckpt_resize_pe(net, state_dict, new_hw):
    sd = state_dict if (isinstance(state_dict, dict) and 'state_dict' not in state_dict) else state_dict.get('model', state_dict)
    key = 'transformer.embeddings.position_embeddings'
    if key in sd:
        sd[key] = _resize_pos_embed(sd[key], new_hw)
    missing, unexpected = net.load_state_dict(sd, strict=False)
    if missing or unexpected:
        print('[state_dict] missing:', missing)
        print('[state_dict] unexpected:', unexpected)

# ---------- Main ----------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--raw_images', type=str, default='pupil-eye-and-iris-segmentation/val/image')
    ap.add_argument('--raw_labels', type=str, default='pupil-eye-and-iris-segmentation/val/segmentation')
    ap.add_argument('--img_ext',   type=str, default='.png')
    ap.add_argument('--lab_ext',   type=str, default='.png')
    ap.add_argument('--input_h',   type=int, default=224)
    ap.add_argument('--input_w',   type=int, default=224)

    ap.add_argument('--vit_name',  type=str, default='R50-ViT-B_16')
    ap.add_argument('--n_skip',    type=int, default=3)
    ap.add_argument('--num_classes', type=int, default=4)
    ap.add_argument('--ckpt',      type=str, default='models_transunet/best_model.pth')
    ap.add_argument('--gpus',      type=str, default='0')

    ap.add_argument('--out_dir',   type=str, default='runs_eval/transunet_pupil_val_224/overlay_fit')
    ap.add_argument('--overlay_dirname',  type=str, default='overlay')
    ap.add_argument('--predmask_dirname', type=str, default='pred_masks')
    ap.add_argument('--alpha',     type=float, default=0.5)
    args = ap.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # data
    ds = SegPairs(args.raw_images, args.raw_labels, args.img_ext, args.lab_ext, args.input_h, args.input_w)
    dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

    # model config
    cfg = CONFIGS_ViT_seg[args.vit_name]
    cfg.n_classes = args.num_classes
    cfg.n_skip    = args.n_skip

    patch = cfg.patches.size
    patch = int(patch[0] if isinstance(patch,(tuple,list)) else patch)
    grid_h, grid_w = args.input_h // patch, args.input_w // patch

    if 'R50' in args.vit_name:
        cfg.patches.grid = (grid_h, grid_w)       # hybrid model requires explicit grid
    else:
        if not hasattr(cfg, 'skip_channels'):
            cfg.skip_channels = [0,0,0,0]         # ensure DecoderCup doesn't expect ResNet skips

    # build & load (pos-embed resize)
    net = ViT_seg(cfg, img_size=(args.input_h, args.input_w), num_classes=cfg.n_classes).to(device)
    ckpt = torch.load(args.ckpt, map_location='cpu')
    load_ckpt_resize_pe(net, ckpt, (grid_h, grid_w))
    net.eval()

    # out dirs
    out_root = Path(args.out_dir)
    pm_dir = out_root / args.predmask_dirname
    ov_dir = out_root / args.overlay_dirname
    pm_dir.mkdir(parents=True, exist_ok=True)
    ov_dir.mkdir(parents=True, exist_ok=True)

    # metrics
    miou_metric   = JaccardIndex(task='multiclass', num_classes=args.num_classes).to(device)
    percls_metric = JaccardIndex(task='multiclass', num_classes=args.num_classes, average=None).to(device)
    ce = torch.nn.CrossEntropyLoss(reduction='mean').to(device)
    losses = []

    with torch.no_grad():
        for x, y, meta in tqdm(dl, total=len(dl), desc='[eval+overlay]'):
            sid = meta['img_id']; ip = meta['img_path']; H0 = int(meta['H0']); W0 = int(meta['W0'])
            if isinstance(sid,(list,tuple)): sid = sid[0]
            if isinstance(ip, (list,tuple)): ip  = ip[0]

            x = x.to(device, non_blocking=True)   # (1,3,H,W)

            # ---- target to (N,H,W) and Long dtype ----
            if y.dim() == 2:
                y = y.unsqueeze(0)                # (1,H,W)
            if y.dim() == 4 and y.size(1) == 1:
                y = y.squeeze(1)                  # (N,H,W)
            y = y.to(device).long()
            # ------------------------------------------

            logits = net(x)                       # (1,C,H,W)
            if isinstance(logits, list): logits = logits[-1]
            pred = logits.argmax(1)               # (1,H,W)

            # torchmetrics
            miou_metric.update(pred, y)
            percls_metric.update(pred, y)

            losses.append(float(ce(logits, y).item()))

            # save overlay & pred mask at original size
            pred_np = pred[0].detach().cpu().numpy().astype(np.uint8)
            pred_res = cv2.resize(pred_np, (W0, H0), interpolation=cv2.INTER_NEAREST)
            color = decode_mask(pred_res)         # RGB
            img0 = cv2.imread(ip, cv2.IMREAD_COLOR)
            overlay_img = overlay_image(img0, color, alpha=args.alpha)
            cv2.imwrite(str(pm_dir/f'{sid}.png'), cv2.cvtColor(color, cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(ov_dir/f'{sid}.jpg'), overlay_img)

    miou = float(miou_metric.compute().item())
    per_class_iou = [float(v) for v in percls_metric.compute().detach().cpu().tolist()]
    test_loss = float(np.mean(losses)) if losses else float('nan')

    with open(out_root/'test_metrics.csv', 'w') as f:
        f.write('metric,value\n')
        f.write(f'test_loss(mean),{test_loss:.6f}\n')
        f.write(f'mIoU,{miou:.6f}\n')
    with open(out_root/'test_class_iou.csv', 'w') as f:
        f.write('class,IoU\n')
        for i,v in enumerate(per_class_iou):
            f.write(f'class_{i},{v:.6f}\n')

    print(f'[DONE] mIoU={miou:.4f} | wrote: {out_root}')
