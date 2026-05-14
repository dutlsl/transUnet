import os
import cv2
import torch
import numpy as np
import csv
import re
import math
from pathlib import Path

# 🚨 구현부(FDA 로직) 임포트 (fda_core.py 파일이 같은 경로에 있어야 합니다)
from fda_core import apply_fda

# 🚨 GPU 1 강제 할당
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# TransUNet 모듈 임포트
from networks.vit_seg_modeling import VisionTransformer
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg

# =====================================================================
# 경로 및 FDA 세팅
# =====================================================================
WEIGHTS_PATH = "./models_transunet/best_model.pth"

BASE_DIR = Path("./Swirski_Dataset")
TABLE_DIR = Path("./Swirski_tables")
OVERLAY_DIR = Path("./Swirski_overlays_transunet_fda")

CSV_PATH = TABLE_DIR / "transunet_swirski_fda_scores.csv"

# [🔥 FDA 추가 세팅]
SOURCE_REF_PATH = "/mnt/ssd1/PycharmProjects/U-Mamba/data/nnUNet_raw/Dataset000_openEDS/imagesTr/openeds_000002_0000.png"
BETA = 0.05 # 저주파 교체 비율 (0.01 ~ 0.09 권장)

IMG_SIZE = 224
NUM_CLASSES = 4
PUPIL_CLASS_ID = 3
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

def get_transunet_model(device):
    config_vit = CONFIGS_ViT_seg['R50-ViT-B_16']
    config_vit.n_classes = NUM_CLASSES
    config_vit.n_skip = 3

    if config_vit.patches.get('grid') is not None:
        config_vit.patches.grid = (int(IMG_SIZE / 16), int(IMG_SIZE / 16))

    model = VisionTransformer(config_vit, img_size=IMG_SIZE, num_classes=NUM_CLASSES)
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
    model.to(device)
    model.eval()
    return model

def main():
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    OVERLAY_DIR.mkdir(parents=True, exist_ok=True)

    # 소스(OpenEDS) 레퍼런스 이미지 로드
    if not Path(SOURCE_REF_PATH).exists():
        print(f"🚨 에러: 소스 기준 이미지({SOURCE_REF_PATH})를 찾을 수 없습니다.")
        return
    source_img = cv2.imread(SOURCE_REF_PATH, cv2.IMREAD_GRAYSCALE)

    # 1번 GPU 명시적 지정
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"1. TransUNet 가동 중... (인식된 Device: {device}, 실제 물리 GPU: 1번)")

    model = get_transunet_model(device)
    print("가중치 로드 성공!\n")

    case_dirs = [d for d in BASE_DIR.iterdir() if d.is_dir() and 'p' in d.name]

    with open(CSV_PATH, mode='w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['Case', 'Frame_Idx', 'IoU', 'Dice'])

        case_metrics = {}
        total_iou, total_dice = [], []

        with torch.no_grad():
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

                print(f"\n평가 시작: {case_name} (총 {len(frame_files)} 프레임, 유효 GT: {len(gt_dict)}개)")

                case_metrics[case_name] = {'iou': [], 'dice': []}
                video_iou = case_metrics[case_name]['iou']
                video_dice = case_metrics[case_name]['dice']

                for idx, frame_path in enumerate(frame_files):
                    frame_idx = extract_frame_idx(frame_path.name)

                    if frame_idx not in gt_dict:
                        continue

                    gt_param = gt_dict[frame_idx]
                    frame_raw = cv2.imread(str(frame_path), cv2.IMREAD_GRAYSCALE)
                    h, w = frame_raw.shape

                    # ==============================================================
                    # 🚀 FDA 적용: 원본 흑백 이미지의 저주파를 소스 이미지 스타일로 변경
                    # ==============================================================
                    frame_fda = apply_fda(frame_raw, source_img, beta=BETA)

                    # --- 1. TransUNet 전처리 및 추론 ---
                    # FDA가 적용된 이미지를 3채널 RGB로 변환하고 리사이즈합니다.
                    rgb_frame = cv2.cvtColor(frame_fda, cv2.COLOR_GRAY2RGB)
                    resized_frame = cv2.resize(rgb_frame, (IMG_SIZE, IMG_SIZE))

                    img_tensor = torch.from_numpy(resized_frame).float().permute(2, 0, 1) / 255.0
                    img_tensor = img_tensor.unsqueeze(0).to(device)

                    logits = model(img_tensor)
                    pred_mask_224 = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy()

                    # 원본 크기로 복구
                    pred_mask = cv2.resize(
                        pred_mask_224.astype(np.uint8),
                        (w, h),
                        interpolation=cv2.INTER_NEAREST
                    )

                    # --- 2. 스코어 계산용 GT 렌더링 ---
                    gt_bin_mask = draw_gt_ellipse(h, w, gt_param)

                    # --- 3. 평가 ---
                    iou, dice = calc_metrics(pred_mask, gt_bin_mask, PUPIL_CLASS_ID)

                    video_iou.append(iou)
                    video_dice.append(dice)
                    total_iou.append(iou)
                    total_dice.append(dice)

                    csv_writer.writerow([case_name, frame_idx, f"{iou:.4f}", f"{dice:.4f}"])

                    # --- 4. 디스크 관리를 위해 10프레임당 1개씩 오버레이 저장 ---
                    if idx % 10 == 0:
                        # 시각화는 스타일이 변환된 FDA 이미지가 아닌 원본 이미지(frame_raw) 위에 그립니다.
                        overlay_img = create_segmentation_overlay(frame_raw, pred_mask)
                        cv2.imwrite(str(case_overlay_dir / f"overlay_{frame_idx:04d}.png"), overlay_img)

                if video_iou:
                    print(f"  -> {case_name} 처리 완료! mIoU: {np.mean(video_iou):.4f} | mDice: {np.mean(video_dice):.4f}")
                else:
                    print(f"  -> {case_name} 건너뜀 (유효한 평가 프레임 없음)")

        # --- 요약 통계 테이블 작성 ---
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