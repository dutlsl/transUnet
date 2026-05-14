import os
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import csv
import re
import copy
from pathlib import Path

# TransUNet 모듈 임포트
from networks.vit_seg_modeling import VisionTransformer as ViT_seg
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from tent_transunet import TentTransUNet

# =====================================================================
# 🚨 다른 로직은 그대로 유지합니다.
DEVICE = torch.device("cuda:1")

MODEL_WEIGHTS = "./models_transunet/best_model.pth"
RAW_BASE_DIR = Path("./LPW")
GT_BASE_DIR = Path("Pupils_in_the_wild_improved")
TABLE_DIR = Path("./LPW_tables")
CSV_PATH = TABLE_DIR / "transunet_lpw_TTA_scores-2.csv"

OUTPUT_BASE_DIR = Path("./LPW_results_transunet_tta-2")

PUPIL_CLASS_ID = 3
NUM_CLASSES = 4
IMG_SIZE = 224

CALIB_FRAMES = 10
TENT_EPOCHS = 25  # 사용자의 요청대로 높은 에폭 유지
TENT_LR = 1e-4
# =====================================================================

def calc_metrics(pred_mask, gt_mask, class_id):
    pred_bin = (pred_mask == class_id)
    gt_bin = (gt_mask == 255)
    intersection = np.logical_and(pred_bin, gt_bin).sum()
    union = np.logical_or(pred_bin, gt_bin).sum()
    if union == 0:
        return (1.0, 1.0) if np.sum(pred_bin) == 0 else (0.0, 0.0)
    return intersection / union, 2 * intersection / (pred_bin.sum() + gt_bin.sum())

def build_gt_mapping(gt_dir):
    mapping = {}
    if not gt_dir.exists(): return mapping
    for f in gt_dir.rglob("*_pupil.mp4"):
        match = re.search(r'folder-(\d+)_file-(\d+)', f.name)
        if match:
            mapping[f"{int(match.group(1))}_{int(match.group(2))}"] = f
    return mapping

def preprocess_frame(frame_bgr, target_size=(IMG_SIZE, IMG_SIZE)):
    rgb_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    resized_frame = cv2.resize(rgb_frame, target_size)
    img_tensor = torch.from_numpy(resized_frame).float().permute(2, 0, 1) / 255.0
    return img_tensor.unsqueeze(0).to(DEVICE)

def load_transunet():
    config_vit = CONFIGS_ViT_seg['R50-ViT-B_16']
    config_vit.n_classes = NUM_CLASSES
    config_vit.n_skip = 3
    if config_vit.patches.get('grid') is not None:
        config_vit.patches.grid = (int(IMG_SIZE / 16), int(IMG_SIZE / 16))

    model = ViT_seg(config_vit, img_size=IMG_SIZE, num_classes=NUM_CLASSES)
    model.load_state_dict(torch.load(MODEL_WEIGHTS, map_location=DEVICE))
    model.to(DEVICE)
    return model

def main():
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    base_model = load_transunet()
    initial_weights = copy.deepcopy(base_model.state_dict())

    gt_mapping = build_gt_mapping(GT_BASE_DIR)
    raw_video_paths = list(RAW_BASE_DIR.rglob("*.avi"))

    with open(CSV_PATH, mode='w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['Case', 'Video_Name', 'Frame_Idx', 'IoU', 'Dice'])

        case_metrics = {}
        total_iou, total_dice = [], []
        processed_videos = 0

        for raw_path in raw_video_paths:
            rel_path = raw_path.relative_to(RAW_BASE_DIR)
            try:
                folder_idx = int(rel_path.parts[0])
                file_idx = int(raw_path.stem)
            except ValueError:
                continue

            key = f"{folder_idx}_{file_idx}"
            if key not in gt_mapping: continue

            gt_path = gt_mapping[key]
            processed_videos += 1
            case_num = str(folder_idx)
            video_name = raw_path.name

            if case_num not in case_metrics:
                case_metrics[case_num] = {'iou': [], 'dice': []}

            print(f"\n[Video {processed_videos}] TransUNet TTA 진행 중: {rel_path}")

            out_path = OUTPUT_BASE_DIR / rel_path
            out_path.parent.mkdir(parents=True, exist_ok=True)

            cap_raw = cv2.VideoCapture(str(raw_path))
            cap_gt = cv2.VideoCapture(str(gt_path))

            fps = cap_raw.get(cv2.CAP_PROP_FPS)
            w = int(cap_raw.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap_raw.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))

            base_model.load_state_dict(initial_weights)
            optimizer = torch.optim.Adam(base_model.parameters(), lr=TENT_LR)
            tent_model = TentTransUNet(base_model, optimizer)

            calib_tensors = []
            for _ in range(CALIB_FRAMES):
                ret, frame = cap_raw.read()
                if not ret: break
                calib_tensors.append(preprocess_frame(frame).squeeze(0))

            if len(calib_tensors) > 0:
                calib_batch = torch.stack(calib_tensors).to(DEVICE)
                for epoch in range(TENT_EPOCHS):
                    _ = tent_model(calib_batch)

            cap_raw.set(cv2.CAP_PROP_POS_FRAMES, 0)
            base_model.eval()

            frame_count = 1
            video_iou, video_dice = [], []

            with torch.no_grad():
                while True:
                    ret_raw, frame_raw = cap_raw.read()
                    ret_gt, frame_gt = cap_gt.read()

                    if not (ret_raw and ret_gt): break

                    input_tensor = preprocess_frame(frame_raw)
                    logits = base_model(input_tensor)

                    pred_mask_224 = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy()
                    pred_mask_resized = cv2.resize(
                        pred_mask_224.astype(np.uint8),
                        (w, h),
                        interpolation=cv2.INTER_NEAREST
                    )

                    result_frame = frame_raw.copy()
                    result_frame[pred_mask_resized == PUPIL_CLASS_ID] = [0, 0, 255]
                    out.write(result_frame)

                    gray_gt = cv2.cvtColor(frame_gt, cv2.COLOR_BGR2GRAY)
                    resized_gt = cv2.resize(gray_gt, (w, h), interpolation=cv2.INTER_NEAREST)
                    _, gt_bin_mask = cv2.threshold(resized_gt, 127, 255, cv2.THRESH_BINARY)

                    iou, dice = calc_metrics(pred_mask_resized, gt_bin_mask, PUPIL_CLASS_ID)
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
            out.release()

            # 🚨 [수정된 부분] 비디오 완료 시 mIoU 출력 추가
            if video_iou:
                print(f"  -> 평가 완료 및 영상 저장됨: {out_path} | mIoU: {np.mean(video_iou):.4f}")

        if processed_videos > 0 and len(total_iou) > 0:
            csv_writer.writerow([])
            csv_writer.writerow(['=== TOTAL SUMMARY ==='])
            csv_writer.writerow(['Total', f"{np.mean(total_iou):.4f}", f"{np.mean(total_dice):.4f}"])
            print(f"\n최종 완료! 총 {processed_videos}개의 비디오 점수 산출 및 오버레이 영상이 저장되었습니다.")

if __name__ == "__main__":
    main()