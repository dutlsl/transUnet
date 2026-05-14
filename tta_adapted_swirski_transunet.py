import os
import cv2
import torch
import torch.nn as nn
import numpy as np
import csv
import re
import math
import copy
from pathlib import Path

# 분리해 둔 TENT 모듈 임포트
from tent_transunet import TentTransUNet

# TransUNet 모듈 임포트
from networks.vit_seg_modeling import VisionTransformer
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg

# =====================================================================
# 🚨 U-Mamba 간섭을 막기 위해 1번 GPU로 명시적 고정합니다.
DEVICE = torch.device("cuda:1")

WEIGHTS_PATH = "./models_transunet/best_model.pth"

BASE_DIR = Path("./Swirski_Dataset")
TABLE_DIR = Path("./Swirski_tables")
OVERLAY_DIR = Path("./Swirski_overlays_transunet_tta")

CSV_PATH = TABLE_DIR / "transunet_swirski_tta_scores.csv"

IMG_SIZE = 224
NUM_CLASSES = 4
PUPIL_CLASS_ID = 3

# TTA (Episodic Calibration) 세팅
CALIB_FRAMES = 10
TENT_EPOCHS = 25  # 파라미터 튜닝의 기준점이 될 에폭
TENT_LR = 1e-4
# =====================================================================

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

def load_ellipse_gt(txt_path):
    gt_dict = {}
    if not txt_path.exists():
        return gt_dict

    with open(txt_path, 'r') as f:
        for line in f:
            if '|' not in line:
                continue
            parts = line.split('|')
            if len(parts) != 2:
                continue
            try:
                frame_idx = int(parts[0].strip())
                params = parts[1].strip().split()
                if len(params) >= 5:
                    gt_dict[frame_idx] = {
                        'x': float(params[0]), 'y': float(params[1]),
                        'a': float(params[2]), 'b': float(params[3]),
                        'angle_rad': float(params[4])
                    }
            except ValueError:
                continue
    return gt_dict

def draw_gt_ellipse(h, w, gt_param):
    mask = np.zeros((h, w), dtype=np.uint8)
    if gt_param['a'] <= 0 or gt_param['b'] <= 0:
        return mask

    angle_deg = math.degrees(gt_param['angle_rad'])
    center = (int(gt_param['x']), int(gt_param['y']))
    axes = (int(gt_param['a']), int(gt_param['b']))

    cv2.ellipse(mask, center, axes, angle_deg, 0, 360, 255, -1)
    return mask

def create_segmentation_overlay(img, pred_mask):
    """GT 외곽선 없이 예측 마스크 영역만 붉은색 반투명으로 덧씌웁니다."""
    img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    overlay = img_bgr.copy()

    overlay[pred_mask == PUPIL_CLASS_ID] = [0, 0, 255]

    alpha = 0.5
    cv2.addWeighted(overlay, alpha, img_bgr, 1 - alpha, 0, img_bgr)
    return img_bgr

def extract_frame_idx(filename):
    match = re.search(r'(\d+)-eye\.png', filename)
    return int(match.group(1)) if match else -1

def get_transunet_model():
    config_vit = CONFIGS_ViT_seg['R50-ViT-B_16']
    config_vit.n_classes = NUM_CLASSES
    config_vit.n_skip = 3

    if config_vit.patches.get('grid') is not None:
        config_vit.patches.grid = (int(IMG_SIZE / 16), int(IMG_SIZE / 16))

    model = VisionTransformer(config_vit, img_size=IMG_SIZE, num_classes=NUM_CLASSES)
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=DEVICE))
    model.to(DEVICE)
    return model

def main():
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    OVERLAY_DIR.mkdir(parents=True, exist_ok=True)

    print(f"1. TransUNet 가동 중... (고정 Device: {DEVICE})")
    base_model = get_transunet_model()
    print("가중치 로드 성공!\n")

    print("2. Source 모델 원본 가중치 백업 중...")
    initial_weights = copy.deepcopy(base_model.state_dict())

    case_dirs = [d for d in BASE_DIR.iterdir() if d.is_dir() and 'p' in d.name]

    with open(CSV_PATH, mode='w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['Case', 'Frame_Idx', 'IoU', 'Dice'])

        case_metrics = {}
        total_iou, total_dice = [], []

        for case_dir in sorted(case_dirs):
            case_name = case_dir.name
            frames_dir = case_dir / "frames"
            gt_path = case_dir / "pupil-ellipses.txt"

            if not frames_dir.exists() or not gt_path.exists():
                continue

            case_overlay_dir = OVERLAY_DIR / case_name
            case_overlay_dir.mkdir(parents=True, exist_ok=True)

            gt_dict = load_ellipse_gt(gt_path)
            frame_files = [f for f in frames_dir.iterdir() if f.name.endswith('-eye.png')]
            frame_files.sort(key=lambda x: extract_frame_idx(x.name))

            if len(frame_files) == 0:
                continue

            print(f"\n[Case {case_name}] TransUNet TTA 진행 중... (총 {len(frame_files)} 프레임)")

            case_metrics[case_name] = {'iou': [], 'dice': []}
            video_iou = case_metrics[case_name]['iou']
            video_dice = case_metrics[case_name]['dice']

            # ==============================================================
            # STEP 1: 모델 리셋 및 TENT 결합
            # ==============================================================
            base_model.load_state_dict(initial_weights)
            optimizer = torch.optim.Adam(base_model.parameters(), lr=TENT_LR)
            tent_model = TentTransUNet(base_model, optimizer)

            # ==============================================================
            # STEP 2: 첫 10 프레임 오프라인 캘리브레이션 (TENT)
            # ==============================================================
            calib_tensors = []
            calib_frame_files = frame_files[:CALIB_FRAMES]

            for f_path in calib_frame_files:
                frame_raw = cv2.imread(str(f_path), cv2.IMREAD_GRAYSCALE)
                rgb_frame = cv2.cvtColor(frame_raw, cv2.COLOR_GRAY2RGB)
                resized_frame = cv2.resize(rgb_frame, (IMG_SIZE, IMG_SIZE))
                img_tensor = torch.from_numpy(resized_frame).float().permute(2, 0, 1) / 255.0
                calib_tensors.append(img_tensor)

            if len(calib_tensors) > 0:
                calib_batch = torch.stack(calib_tensors).to(DEVICE)
                for epoch in range(TENT_EPOCHS):
                    _ = tent_model(calib_batch)
                print(f"  -> {len(calib_tensors)}프레임 TENT 캘리브레이션 완료 ({TENT_EPOCHS} Epochs)")

            # ==============================================================
            # STEP 3: 전체 프레임 추론, 평가 및 오버레이 저장
            # ==============================================================
            base_model.eval()

            with torch.no_grad():
                for idx, frame_path in enumerate(frame_files):
                    frame_idx = extract_frame_idx(frame_path.name)

                    if frame_idx not in gt_dict:
                        continue

                    gt_param = gt_dict[frame_idx]
                    frame_raw = cv2.imread(str(frame_path), cv2.IMREAD_GRAYSCALE)
                    h, w = frame_raw.shape

                    rgb_frame = cv2.cvtColor(frame_raw, cv2.COLOR_GRAY2RGB)
                    resized_frame = cv2.resize(rgb_frame, (IMG_SIZE, IMG_SIZE))
                    img_tensor = torch.from_numpy(resized_frame).float().permute(2, 0, 1) / 255.0
                    img_tensor = img_tensor.unsqueeze(0).to(DEVICE)

                    logits = base_model(img_tensor)
                    pred_mask_224 = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy()

                    pred_mask = cv2.resize(
                        pred_mask_224.astype(np.uint8),
                        (w, h),
                        interpolation=cv2.INTER_NEAREST
                    )

                    gt_bin_mask = draw_gt_ellipse(h, w, gt_param)
                    iou, dice = calc_metrics(pred_mask, gt_bin_mask, PUPIL_CLASS_ID)

                    video_iou.append(iou)
                    video_dice.append(dice)
                    total_iou.append(iou)
                    total_dice.append(dice)

                    csv_writer.writerow([case_name, frame_idx, f"{iou:.4f}", f"{dice:.4f}"])

                    if idx % 10 == 0:
                        overlay_img = create_segmentation_overlay(frame_raw, pred_mask)
                        cv2.imwrite(str(case_overlay_dir / f"overlay_{frame_idx:04d}.png"), overlay_img)

            # 🚨 비디오(Case) 하나가 완료될 때마다 mIoU 출력
            if video_iou:
                print(f"  -> 평가 완료! 저장됨: {case_overlay_dir} | mIoU: {np.mean(video_iou):.4f}")
            else:
                print(f"  -> {case_name} 건너뜀 (유효한 평가 프레임 없음)")

        # ==============================================================
        # STEP 4: 요약 통계 작성
        # ==============================================================
        csv_writer.writerow([])
        csv_writer.writerow(['=== CASE SUMMARY ==='])
        csv_writer.writerow(['Case', 'mIoU', 'mDice'])
        for case, metrics in case_metrics.items():
            if metrics['iou']:
                csv_writer.writerow([case, f"{np.mean(metrics['iou']):.4f}", f"{np.mean(metrics['dice']):.4f}"])

        if total_iou:
            csv_writer.writerow([])
            csv_writer.writerow(['=== TOTAL SUMMARY ==='])
            csv_writer.writerow(['Total', f"{np.mean(total_iou):.4f}", f"{np.mean(total_dice):.4f}"])

    print(f"\n평가 종료! 최종 결과가 {CSV_PATH} 에 저장되었습니다.")

if __name__ == "__main__":
    main()