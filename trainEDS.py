# trainEDS.py
import argparse, os, time, random, csv
from pathlib import Path
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import cv2
import wandb

# TransUNet
from networks.vit_seg_modeling import VisionTransformer as ViT_seg
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg

# -----------------------------
# 데이터셋: train 이미지/라벨만 사용, 내부에서 split
# -----------------------------
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
        img = cv2.imread(ip, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(ip)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)

        mask = np.load(lp)                          # (H,W) int
        mask = cv2.resize(mask, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)
        mask = mask.astype(np.int64)                # resize 이후 int64로 고정

        img = np.transpose(img, (2, 0, 1))          # CHW
        return torch.from_numpy(img), torch.from_numpy(mask).long(), {'img_id': sid, 'img_path': ip}

# -----------------------------
# 유틸(스플릿/지표)
# -----------------------------
def list_matched_ids(img_dir, img_ext, label_dir, mask_ext):
    img_ids = {Path(p).stem for p in Path(img_dir).glob(f'*{img_ext}')}
    lab_ids = {Path(p).stem for p in Path(label_dir).glob(f'*{mask_ext}')}
    ids = sorted(list(img_ids & lab_ids))
    if not ids:
        raise FileNotFoundError(f'No matched pairs under {img_dir} & {label_dir}')
    return ids

def split_ids(ids, tr, va, te, seed):
    s = tr + va + te
    if abs(s - 1.0) > 1e-6:
        raise ValueError(f'split ratios must sum to 1.0 (got {s})')
    rng = random.Random(seed)
    ids = ids[:]
    rng.shuffle(ids)
    n = len(ids); n_tr = int(round(n*tr)); n_va = int(round(n*va))
    return ids[:n_tr], ids[n_tr:n_tr+n_va], ids[n_tr+n_va:]

def _fast_hist(gt, pr, n):
    k = (gt >= 0) & (gt < n)
    return np.bincount(n * gt[k].astype(np.int64) + pr[k].astype(np.int64),
                       minlength=n*n).reshape(n, n)

def _miou_acc_from_hist(hist):
    n = hist.shape[0]
    tp = np.diag(hist).astype(np.float64)
    fp = hist.sum(0) - tp
    fn = hist.sum(1) - tp
    denom = tp + fp + fn
    valid = denom > 0
    eps = 1e-7
    iou_c = np.zeros(n, dtype=np.float64)
    iou_c[valid] = tp[valid] / (denom[valid] + eps)
    miou = iou_c[valid].mean() if valid.any() else 0.0
    pixel_acc = tp.sum() / (hist.sum() + eps)
    return float(miou), iou_c, tp, fp, fn, float(pixel_acc)

# -----------------------------
# 루프
# -----------------------------
def train_epoch(model, loader, optimizer, device, criterion, num_classes):
    model.train()
    total_loss = 0.0
    hist = np.zeros((num_classes, num_classes), dtype=np.int64)
    pbar = tqdm(total=len(loader), desc='[train]')
    for x, y, _ in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True).long()
        optimizer.zero_grad(set_to_none=True)
        out = model(x)
        logits = out[-1] if isinstance(out, list) else out
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item())
        with torch.no_grad():
            pred = logits.argmax(1)
            gt = y.detach().cpu().numpy().astype(np.int64)
            pr = pred.detach().cpu().numpy().astype(np.int64)
            for i in range(gt.shape[0]):
                hist += _fast_hist(gt[i], pr[i], num_classes)
        miou, *_ = _miou_acc_from_hist(hist)
        pbar.set_postfix(loss=f'{total_loss/(pbar.n+1):.4f}', miou=f'{miou:.4f}')
        pbar.update(1)
    pbar.close()
    miou, *_ = _miou_acc_from_hist(hist)
    return total_loss / max(1, len(loader)), miou

@torch.no_grad()
def valid_epoch(model, loader, device, criterion, num_classes, phase='val'):
    model.eval()
    total_loss = 0.0
    hist = np.zeros((num_classes, num_classes), dtype=np.int64)
    pbar = tqdm(total=len(loader), desc=f'[{phase}]')
    for x, y, _ in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True).long()
        out = model(x)
        logits = out[-1] if isinstance(out, list) else out
        loss = criterion(logits, y)
        total_loss += float(loss.item())
        pred = logits.argmax(1)
        gt = y.detach().cpu().numpy().astype(np.int64)
        pr = pred.detach().cpu().numpy().astype(np.int64)
        for i in range(gt.shape[0]):
            hist += _fast_hist(gt[i], pr[i], num_classes)
        miou, *_ = _miou_acc_from_hist(hist)
        pbar.set_postfix(**{f'{phase}_loss': f'{total_loss/(pbar.n+1):.4f}', f'{phase}_miou': f'{miou:.4f}'})
        pbar.update(1)
    pbar.close()
    miou, iou_c, tp, fp, fn, pixel_acc = _miou_acc_from_hist(hist)
    return total_loss / max(1, len(loader)), miou, iou_c, tp, fp, fn, pixel_acc

# -----------------------------
# EarlyStopping (쿨다운/최소에폭 지원)
# -----------------------------
class EarlyStopping:
    def __init__(self, patience=6, min_delta=1e-4, min_epochs=8, cooldown=0, mode="max"):
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.min_epochs = int(min_epochs)
        self.cooldown = int(cooldown)
        self.mode = mode
        self.best = None
        self.bad = 0
        self.cool = 0
        self.stopped = False
        self.stop_epoch = None

    def update(self, value, epoch):
        # 최소 에폭 이전엔 기록만
        if (epoch + 1) < self.min_epochs:
            if self.best is None: self.best = value
            elif (value > self.best) if self.mode == "max" else (value < self.best): self.best = value
            return False
        if self.best is None:
            self.best = value
            return False
        improved = (value > self.best + self.min_delta) if self.mode == "max" else (value < self.best - self.min_delta)
        if improved:
            self.best = value
            self.bad = 0
            self.cool = self.cooldown
        else:
            if self.cool > 0:
                self.cool -= 1
            else:
                self.bad += 1
                if self.bad >= self.patience:
                    self.stopped = True
                    self.stop_epoch = epoch + 1
                    return True
        return False

def write_rows_csv(path, rows):
    if not rows: return
    keys = list(rows[0].keys())
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows: w.writerow(r)

def _save_test_reports(out_dir, miou, pixel_acc, test_loss, iou_c, tp, fp, fn):
    # metrics
    metrics_csv = Path(out_dir) / 'test_metrics.csv'
    with open(metrics_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['metric','value'])
        w.writerow(['mIoU', f'{float(miou):.6f}'])
        w.writerow(['PixelAcc', f'{float(pixel_acc):.6f}'])
        w.writerow(['test_loss(mean)', f'{float(test_loss):.6f}'])
    # class-wise
    class_csv = Path(out_dir) / 'test_class_iou.csv'
    with open(class_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['class','IoU','TP','FP','FN'])
        for i in range(len(iou_c)):
            w.writerow([f'class_{i}', f'{float(iou_c[i]):.6f}', int(tp[i]), int(fp[i]), int(fn[i])])
    print(f'WROTE {metrics_csv}')
    print(f'WROTE {class_csv}')

# -----------------------------
# 인자 (최소 추가)
# -----------------------------
parser = argparse.ArgumentParser()
# 데이터 (train만 제공되고 내부 스플릿)
parser.add_argument('--raw_images', type=str, default='openEDS/train/images')
parser.add_argument('--raw_labels', type=str, default='openEDS/train/labels')
parser.add_argument('--img_ext', type=str, default='.png')
parser.add_argument('--mask_ext', type=str, default='.npy')
parser.add_argument('--split_train', type=float, default=0.7)
parser.add_argument('--split_val',   type=float, default=0.1)
parser.add_argument('--split_test',  type=float, default=0.2)
parser.add_argument('--split_seed',  type=int,   default=41)

# 모델/학습(원본 느낌 유지)
parser.add_argument('--num_classes', type=int, default=4)
parser.add_argument('--max_epochs', type=int, default=175)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--base_lr', type=float, default=1e-4)
parser.add_argument('--img_size', type=int, default=224)
parser.add_argument('--seed', type=int, default=41)
parser.add_argument('--n_skip', type=int, default=3)
parser.add_argument('--vit_name', type=str, default='R50-ViT-B_16')
parser.add_argument('--vit_patches_size', type=int, default=16)

# Early Stop 파라미터
parser.add_argument('--early_stop', type=int, default=1)
parser.add_argument('--patience', type=int, default=6)
parser.add_argument('--min_delta', type=float, default=1e-4)
parser.add_argument('--min_epochs', type=int, default=8)
parser.add_argument('--cooldown', type=int, default=0)

# 실행/출력
parser.add_argument('--out_dir', type=str, default='models_transunet')
parser.add_argument('--wandb_project', type=str, default='unet-seg3')
parser.add_argument('--wandb_mode', type=str, choices=['online','offline','disabled'], default='disabled')  # 기본 disabled
parser.add_argument('--gpus', type=str, default='0')

# 테스트 전용/체크포인트
parser.add_argument('--test_only', type=int, default=0)
parser.add_argument('--resume', type=str, default='')

args = parser.parse_args()

if __name__ == "__main__":
    # GPU0만 사용 (이 프로세스 내부에만 영향)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # 시드/성능
    random.seed(args.seed); np.random.seed(args.seed)
    torch.manual_seed(args.seed); torch.cuda.manual_seed(args.seed)
    cudnn.benchmark = True; cudnn.deterministic = False

    # 매칭/스플릿
    ids_all = list_matched_ids(args.raw_images, args.img_ext, args.raw_labels, args.mask_ext)
    tr_ids, va_ids, te_ids = split_ids(ids_all, args.split_train, args.split_val, args.split_test, args.split_seed)
    print(f"[Split] total={len(ids_all)} | train={len(tr_ids)} val={len(va_ids)} test={len(te_ids)}", flush=True)

    # 데이터로더
    ds_tr = OpenEDSSeg(tr_ids, args.raw_images, args.raw_labels, args.img_ext, args.mask_ext, args.img_size)
    ds_va = OpenEDSSeg(va_ids, args.raw_images, args.raw_labels, args.img_ext, args.mask_ext, args.img_size)
    ds_te = OpenEDSSeg(te_ids, args.raw_images, args.raw_labels, args.img_ext, args.mask_ext, args.img_size)

    ld_tr = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True,  num_workers=8, pin_memory=True, drop_last=True)
    ld_va = DataLoader(ds_va, batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True)
    ld_te = DataLoader(ds_te, batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True)

    # 모델
    config_vit = CONFIGS_ViT_seg[args.vit_name]
    config_vit.n_classes = args.num_classes
    config_vit.n_skip = args.n_skip
    if 'R50' in args.vit_name:
        config_vit.patches.grid = (args.img_size // args.vit_patches_size, args.img_size // args.vit_patches_size)
    net = ViT_seg(config_vit, img_size=args.img_size, num_classes=config_vit.n_classes).to(device)
    # 사전학습 로드(가능한 경우)
    try:
        net.load_from(weights=np.load(config_vit.pretrained_path))
    except Exception as e:
        tqdm.write(f"[warn] pretrained load failed: {e}")

    # 옵티마이저/스케줄러/손실
    optimizer = optim.AdamW(net.parameters(), lr=args.base_lr, weight_decay=1e-4)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epochs, last_epoch=-1)
    class_weight = torch.tensor([1,1,2,4][:args.num_classes], dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weight).to(device)

    # W&B (기본 disabled)
    if args.wandb_mode != 'disabled':
        os.environ['WANDB_MODE'] = args.wandb_mode
        wandb.init(project=args.wandb_project, name=f"TransUNet-{int(time.time())}", config=vars(args))

    # 출력 경로
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    best_path = str(Path(args.out_dir)/'best_model.pth')
    last_ckpt = str(Path(args.out_dir)/'last.ckpt')

    # test-only
    if args.test_only:
        if args.resume and os.path.isfile(args.resume):
            state = torch.load(args.resume, map_location='cpu')
            net.load_state_dict(state['model'] if isinstance(state, dict) and 'model' in state else state)
        # 베스트 로드 가능한 경우 우선
        if os.path.isfile(best_path):
            try: net.load_state_dict(torch.load(best_path, map_location='cpu'))
            except Exception: pass
        te_loss, te_miou, iou_c, tp, fp, fn, pixel_acc = valid_epoch(net, ld_te, device, criterion, args.num_classes, phase='test')
        _save_test_reports(args.out_dir, te_miou, pixel_acc, te_loss, iou_c, tp, fp, fn)
        print(f">> TEST: loss={te_loss:.6f} mIoU={te_miou:.6f} pixelAcc={pixel_acc:.6f}", flush=True)
        if args.wandb_mode != 'disabled':
            wandb.summary['test/loss'] = te_loss; wandb.summary['test/miou'] = te_miou; wandb.finish()
        raise SystemExit(0)

    # 학습
    stopper = EarlyStopping(patience=args.patience, min_delta=args.min_delta,
                            min_epochs=args.min_epochs, cooldown=args.cooldown, mode="max") if args.early_stop else None
    best = -1.0
    log_rows = []
    for epoch in range(args.max_epochs):
        tqdm.write(f"--- Epoch {epoch+1}/{args.max_epochs} ---")
        tr_loss, tr_miou = train_epoch(net, ld_tr, optimizer, device, criterion, args.num_classes)
        va_loss, va_miou, *_ = valid_epoch(net, ld_va, device, criterion, args.num_classes, phase='val')
        scheduler.step()

        log_rows.append({'epoch': epoch, 'train_loss': f'{tr_loss:.6f}', 'train_miou': f'{tr_miou:.6f}',
                         'val_loss': f'{va_loss:.6f}', 'val_miou': f'{va_miou:.6f}',
                         'lr': f'{optimizer.param_groups[0]["lr"]:.8f}'})
        write_rows_csv(Path(args.out_dir)/'train_log.csv', log_rows)

        if args.wandb_mode != 'disabled':
            wandb.log({'epoch': epoch+1, 'train/loss': tr_loss, 'train/miou': tr_miou,
                       'val/loss': va_loss, 'val/miou': va_miou,
                       'lr': optimizer.param_groups[0]['lr']}, step=epoch+1)

        if va_miou > best:
            best = va_miou
            torch.save(net.state_dict(), best_path)
            tqdm.write(f">> saved best (val_mIoU={best:.4f})")

        torch.save({'model': net.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': int(epoch),
                    'best_metric': float(best)}, last_ckpt)

        if stopper and stopper.update(va_miou, epoch):
            tqdm.write(f">> early stop at epoch {epoch+1} (best={best:.4f})")
            # 기록
            try:
                with open(Path(args.out_dir)/'early_stop.txt', 'w') as f:
                    f.write(f'early_stop_epoch={epoch+1}\n')
                    f.write(f'best_val_miou={best:.6f}\n')
                    f.write(f'patience={args.patience} min_delta={args.min_delta} min_epochs={args.min_epochs} cooldown={args.cooldown}\n')
            except Exception:
                pass
            break

    # 테스트 전, 항상 베스트 로드
    if os.path.isfile(best_path):
        try:
            net.load_state_dict(torch.load(best_path, map_location='cpu'))
            tqdm.write(">> loaded best_model.pth for final test")
        except Exception as e:
            tqdm.write(f"[warn] best load failed before test: {e}")

    # 최종 테스트
    te_loss, te_miou, iou_c, tp, fp, fn, pixel_acc = valid_epoch(net, ld_te, device, criterion, args.num_classes, phase='test')
    _save_test_reports(args.out_dir, te_miou, pixel_acc, te_loss, iou_c, tp, fp, fn)
    print(f">> TEST: loss={te_loss:.6f} mIoU={te_miou:.6f} pixelAcc={pixel_acc:.6f}", flush=True)
    if args.wandb_mode != 'disabled':
        wandb.summary['test/loss'] = te_loss; wandb.summary['test/miou'] = te_miou; wandb.finish()
