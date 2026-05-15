"""
TransUNet 내부 feature map의 주파수 특성을 레이어별로 정밀 분석.
목적: 속눈썹/글레어로 인한 실패가 어느 레이어에서 발생하는지 추적.
"""
import os, sys, json
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import torch.nn as nn
import numpy as np
import cv2
from pathlib import Path
from collections import OrderedDict

from networks.vit_seg_modeling import VisionTransformer, CONFIGS as CONFIGS_ViT_seg

# ============ CONFIG ============
WEIGHTS = "./models_transunet/best_model.pth"
IMG_SIZE = 224
NUM_CLASSES = 4
PUPIL_ID = 3
# ================================

def load_model(device):
    cfg = CONFIGS_ViT_seg['R50-ViT-B_16']
    cfg.n_classes = NUM_CLASSES
    cfg.n_skip = 3
    cfg.patches.grid = (IMG_SIZE // 16, IMG_SIZE // 16)
    model = VisionTransformer(cfg, img_size=IMG_SIZE, num_classes=NUM_CLASSES)
    model.load_state_dict(torch.load(WEIGHTS, map_location=device))
    model.to(device).eval()
    return model

def prep_image(path, size=224):
    """grayscale frame → (1,3,H,W) tensor"""
    raw = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    rgb = cv2.cvtColor(raw, cv2.COLOR_GRAY2RGB)
    resized = cv2.resize(rgb, (size, size))
    t = torch.from_numpy(resized).float().permute(2,0,1) / 255.0
    return t.unsqueeze(0), raw

def freq_energy_ratio(feat_map):
    """
    feature map (C,H,W)에 대해 채널 평균 후
    2D FFT → 고주파 에너지 비율(outer 75% 영역 / 전체) 반환.
    또한 주파수 밴드별 에너지 분포도 반환.
    """
    # 채널 평균
    avg = feat_map.mean(dim=0).cpu().numpy()  # (H, W)
    H, W = avg.shape
    
    fft = np.fft.fftshift(np.fft.fft2(avg))
    mag = np.abs(fft) ** 2  # power spectrum
    total = mag.sum()
    if total == 0:
        return 0.0, []
    
    cH, cW = H // 2, W // 2
    
    # 주파수 밴드별 에너지: 5개 동심원 밴드
    max_r = min(cH, cW)
    bands = []
    n_bands = 5
    for b in range(n_bands):
        r_inner = max_r * b / n_bands
        r_outer = max_r * (b + 1) / n_bands
        mask = np.zeros((H, W), dtype=bool)
        for y in range(H):
            for x in range(W):
                r = np.sqrt((y - cH)**2 + (x - cW)**2)
                if r_inner <= r < r_outer:
                    mask[y, x] = True
        band_energy = mag[mask].sum() / total
        bands.append(float(band_energy))
    
    # 고주파 비율: 중심 25% 밖 = "고주파"
    low_r = max_r * 0.25
    low_mask = np.zeros((H, W), dtype=bool)
    for y in range(H):
        for x in range(W):
            if np.sqrt((y - cH)**2 + (x - cW)**2) <= low_r:
                low_mask[y, x] = True
    low_energy = mag[low_mask].sum() / total
    high_ratio = 1.0 - low_energy
    
    return float(high_ratio), bands

def channelwise_freq_std(feat_map):
    """
    채널별 고주파 에너지의 분산을 측정.
    분산이 크면 = 특정 채널에만 고주파가 집중 = 노이즈/아티팩트 가능성 높음.
    """
    C, H, W = feat_map.shape
    cH, cW = H // 2, W // 2
    max_r = min(cH, cW)
    low_r = max_r * 0.25
    
    # low-freq mask (사전 계산)
    low_mask = np.zeros((H, W), dtype=bool)
    for y in range(H):
        for x in range(W):
            if np.sqrt((y - cH)**2 + (x - cW)**2) <= low_r:
                low_mask[y, x] = True
    
    ch_highs = []
    for c in range(min(C, 64)):  # 채널 수가 많으면 샘플링
        ch = feat_map[c].cpu().numpy()
        fft = np.fft.fftshift(np.fft.fft2(ch))
        mag = np.abs(fft) ** 2
        total = mag.sum()
        if total == 0:
            ch_highs.append(0.0)
            continue
        low_e = mag[low_mask].sum() / total
        ch_highs.append(1.0 - low_e)
    
    return float(np.mean(ch_highs)), float(np.std(ch_highs))

def hook_features(model):
    """모델 내부의 핵심 feature를 캡처하는 hook 설치"""
    captured = {}
    
    def make_hook(name):
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                captured[name] = output[0].detach()
            else:
                captured[name] = output.detach()
        return hook_fn
    
    # ResNet stages
    resnet = model.transformer.embeddings.hybrid_model
    resnet.root.register_forward_hook(make_hook('resnet_root'))      # 64ch, 112x112
    resnet.body.block1.register_forward_hook(make_hook('resnet_b1'))  # 256ch, 56x56
    resnet.body.block2.register_forward_hook(make_hook('resnet_b2'))  # 512ch, 28x28
    resnet.body.block3.register_forward_hook(make_hook('resnet_b3'))  # 1024ch, 14x14
    
    # ViT encoder output
    model.transformer.encoder.register_forward_hook(make_hook('vit_encoded'))  # 196x768
    
    # Decoder conv_more (bottleneck → 512ch)
    model.decoder.conv_more.register_forward_hook(make_hook('dec_conv_more'))  # 512ch, 14x14
    
    # Decoder blocks outputs
    for i, block in enumerate(model.decoder.blocks):
        block.register_forward_hook(make_hook(f'dec_block{i}'))
    
    # Segmentation head
    model.segmentation_head.register_forward_hook(make_hook('seg_head'))
    
    return captured


def analyze_frame(model, img_path, device, captured):
    """한 프레임에 대해 전체 feature 주파수 분석"""
    tensor, raw = prep_image(img_path)
    tensor = tensor.to(device)
    
    with torch.no_grad():
        logits = model(tensor)
    
    pred = logits.argmax(1).squeeze(0).cpu().numpy()
    pupil_pixels = (pred == PUPIL_ID).sum()
    
    results = {'path': str(img_path), 'pupil_pixels': int(pupil_pixels)}
    
    # 각 캡처된 feature에 대해 분석
    feature_order = [
        'resnet_root', 'resnet_b1', 'resnet_b2', 'resnet_b3',
        'dec_conv_more',
        'dec_block0', 'dec_block1', 'dec_block2', 'dec_block3',
        'seg_head'
    ]
    
    for name in feature_order:
        if name not in captured:
            continue
        feat = captured[name]
        if feat.dim() == 3:  # (B, N, C) from ViT
            continue
        feat = feat.squeeze(0)  # (C, H, W)
        
        hi_ratio, bands = freq_energy_ratio(feat)
        ch_mean, ch_std = channelwise_freq_std(feat)
        
        results[name] = {
            'shape': list(feat.shape),
            'high_freq_ratio': round(hi_ratio, 4),
            'freq_bands': [round(b, 4) for b in bands],
            'ch_high_mean': round(ch_mean, 4),
            'ch_high_std': round(ch_std, 4),
        }
    
    return results


def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = load_model(device)
    captured = hook_features(model)
    
    # 대표 프레임 선정:
    # 실패: p1-right (과노출+소형동공), p2-right frame 670 (hollow)
    # 성공: p2-left frame 210 (큰 동공, 깨끗), p2-right frame 230
    test_frames = {
        'p1r_fail_overexp_f10': 'Swirski_Dataset/p1-right/frames/10-eye.png',
        'p1r_fail_eyelash_f400': 'Swirski_Dataset/p1-right/frames/400-eye.png',
        'p1r_fail_tiny_f720': 'Swirski_Dataset/p1-right/frames/720-eye.png',
        'p2l_fail_hollow_f90': 'Swirski_Dataset/p2-left/frames/90-eye.png',
        'p2l_success_f204': 'Swirski_Dataset/p2-left/frames/204-eye.png',
        'p2r_success_f230': 'Swirski_Dataset/p2-right/frames/230-eye.png',
        'p2r_fail_hollow_f670': 'Swirski_Dataset/p2-right/frames/670-eye.png',
        'p1l_fail_frag_f0': 'Swirski_Dataset/p1-left/frames/0-eye.png',
    }
    
    all_results = {}
    for label, path in test_frames.items():
        p = Path(path)
        if not p.exists():
            print(f"  SKIP {label}: {path} not found")
            continue
        print(f"Analyzing: {label} ...")
        r = analyze_frame(model, p, device, captured)
        all_results[label] = r
        
        # 요약 출력
        print(f"  pupil_pixels={r['pupil_pixels']}")
        for k, v in r.items():
            if isinstance(v, dict) and 'shape' in v:
                shape_str = 'x'.join(map(str, v['shape']))
                print(f"  {k:20s} [{shape_str:>15s}]  HF={v['high_freq_ratio']:.3f}  "
                      f"bands={v['freq_bands']}  ch_std={v['ch_high_std']:.4f}")
        print()
    
    # 결과 저장
    out = Path('probe_results.json')
    with open(out, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"Saved to {out}")

if __name__ == '__main__':
    main()
