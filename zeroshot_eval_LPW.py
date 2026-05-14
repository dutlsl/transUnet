import os
import cv2
import torch
import numpy as np
import csv
import re
from pathlib import Path

# 🚨 1번 GPU에 강제 할당합니다.
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# TransUNet 모듈 임포트
from networks.vit_seg_modeling import VisionTransformer
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg

# =====================================================================
# 경로 및 파라미터 세팅
WEIGHTS_PATH = "./models_transunet/best_model.pth"
RAW_BASE_DIR = Path("./LPW")
GT_BASE_DIR = Path("./Pupils_in_the_wild_improved")

TABLE_DIR = Path("./LPW_tables")
CSV_PATH = TABLE_DIR / "transunet_lpw_video_gt_scores.csv"

IMG_SIZE = 224
NUM_CLASSES = 4
PUPIL_CLASS_ID = 3
# =====================================================================

def calc_metrics(pred_mask, gt_mask, class_id):
    """IoU와 Dice Score를 계산합니다."""
    pred_bin = (pred_mask == class_id)
    gt_bin = (gt_mask == 255) # 라벨 비디오의 흰색(255) 픽셀이 동공

    intersection = np.logical_and(pred_bin, gt_bin).sum()
    union = np.logical_or(pred_bin, gt_bin).sum()

    if union == 0:
        iou = 1.0 if np.sum(pred_bin) == 0 else 0.0
        dice = 1.0 if np.sum(pred_bin) == 0 else 0.0
    else:
        iou = intersection / union
        dice = 2 * intersection / (pred_bin.sum() + gt_bin.sum())

    return iou, dice

def build_gt_mapping(gt_dir):
    """GT 폴더를 스캔하여 '폴더번호_파일번호' 형태의 딕셔너리로 매핑합니다."""
    mapping = {}
    if not gt_dir.exists():
        print(f"🚨 GT 폴더를 찾을 수 없습니다: {gt_dir.absolute()}")
        return mapping

    for f in gt_dir.rglob("*_pupil.mp4"):
        match = re.search(r'folder-(\d+)_file-(\d+)', f.name)
        if match:
            folder_idx = int(match.group(1))
            file_idx = int(match.group(2))
            mapping[f"{folder_idx}_{file_idx}"] = f
    return mapping

def get_transunet_model(device):
    """TransUNet 세팅 (Grid ZeroDivision 방어 포함)"""
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

    # 환경 변수로 1번 GPU만 보이게 했으므로, 'cuda:0'은 물리적인 1번 GPU를 의미합니다.
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"1. TransUNet 가동 중... (인식된 Device: {device}, 실제 물리 GPU: 1번)")

    model = get_transunet_model(device)
    print("가중치 로드 성공!\n")

    # 정답지(GT) 자동 매핑
    gt_mapping = build_gt_mapping(GT_BASE_DIR)
    print(f"찾아낸 정답(GT) 비디오: {len(gt_mapping)}개 (목표: 66개)")

    raw_video_paths = list(RAW_BASE_DIR.rglob("*.avi"))
    print(f"발견된 로데이터 비디오: {len(raw_video_paths)}개\n")

    with open(CSV_PATH, mode='w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['Case', 'Video_Name', 'Frame_Idx', 'IoU', 'Dice'])

        case_metrics = {}
        total_iou, total_dice = [], []
        processed_videos = 0

        with torch.no_grad():
            for raw_path in raw_video_paths:
                rel_path = raw_path.relative_to(RAW_BASE_DIR)
                try:
                    folder_idx = int(rel_path.parts[0])
                    file_idx = int(raw_path.stem)
                except ValueError:
                    continue

                key = f"{folder_idx}_{file_idx}"

                if key not in gt_mapping:
                    print(f"⚠️ 매칭 GT 없음 (스킵): {raw_path}")
                    continue

                gt_path = gt_mapping[key]
                processed_videos += 1
                case_num = str(folder_idx)
                video_name = raw_path.name

                if case_num not in case_metrics:
                    case_metrics[case_num] = {'iou': [], 'dice': []}

                print(f"평가 중: {rel_path} <--> {gt_path.name}")

                cap_raw = cv2.VideoCapture(str(raw_path))
                cap_gt = cv2.VideoCapture(str(gt_path))

                frame_count = 1
                video_iou, video_dice = [], []

                while True:
                    ret_raw, frame_raw = cap_raw.read()
                    ret_gt, frame_gt = cap_gt.read()

                    if not (ret_raw and ret_gt):
                        break

                    h, w = frame_raw.shape[:2]

                    # --- 1. TransUNet 추론 (224 Resize) ---
                    rgb_frame = cv2.cvtColor(frame_raw, cv2.COLOR_BGR2RGB)
                    resized_frame = cv2.resize(rgb_frame, (IMG_SIZE, IMG_SIZE))

                    img_tensor = torch.from_numpy(resized_frame).float().permute(2, 0, 1) / 255.0
                    img_tensor = img_tensor.unsqueeze(0).to(device)

                    logits = model(img_tensor)
                    pred_mask_224 = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy()

                    # 원본 크기로 예측 마스크 복구
                    pred_mask = cv2.resize(
                        pred_mask_224.astype(np.uint8),
                        (w, h),
                        interpolation=cv2.INTER_NEAREST
                    )

                    # --- 2. 정답 마스크 처리 (Nearest Neighbor Resize) ---
                    gray_gt = cv2.cvtColor(frame_gt, cv2.COLOR_BGR2GRAY)
                    resized_gt = cv2.resize(gray_gt, (w, h), interpolation=cv2.INTER_NEAREST)
                    _, gt_bin_mask = cv2.threshold(resized_gt, 127, 255, cv2.THRESH_BINARY)

                    # --- 3. 평가 및 기록 ---
                    iou, dice = calc_metrics(pred_mask, gt_bin_mask, PUPIL_CLASS_ID)

                    video_iou.append(iou)
                    video_dice.append(dice)
                    total_iou.append(iou)
                    total_dice.append(dice)
                    case_metrics[case_num]['iou'].append(iou)
                    case_metrics[case_num]['dice'].append(dice)

                    csv_writer.writerow([case_num, video_name, frame_count, f"{iou:.4f}", f"{dice:.4f}"])
                    frame_count += 1

                cap_raw.release()
                cap_gt.release()

                if video_iou:
                    print(f"  -> 완료! mIoU: {np.mean(video_iou):.4f} | mDice: {np.mean(video_dice):.4f}")

        # --- 요약 통계 ---
        if processed_videos > 0 and len(total_iou) > 0:
            csv_writer.writerow([])
            csv_writer.writerow([f'=== CASE SUMMARY (Total Evaluated: {processed_videos}) ==='])
            csv_writer.writerow(['Case', 'mIoU', 'mDice'])
            for case, metrics in case_metrics.items():
                if metrics['iou']:
                    csv_writer.writerow([case, f"{np.mean(metrics['iou']):.4f}", f"{np.mean(metrics['dice']):.4f}"])

            csv_writer.writerow([])
            csv_writer.writerow(['=== TOTAL SUMMARY ==='])
            csv_writer.writerow(['Total', f"{np.mean(total_iou):.4f}", f"{np.mean(total_dice):.4f}"])
            print(f"\n평가 완료! 총 {processed_videos}개의 비디오가 정상적으로 분석되었습니다.")
        else:
            print("\n🚨 경고: 처리된 비디오가 없습니다.")

if __name__ == "__main__":
    main()