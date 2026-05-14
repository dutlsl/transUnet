import cv2
import torch
import numpy as np
import os
from pathlib import Path

# TransUNet 모듈 임포트
from networks.vit_seg_modeling import VisionTransformer
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg

# =====================================================================
# 1. 경로 및 파라미터 세팅
WEIGHTS_PATH = "./models_transunet/best_model.pth"
INPUT_BASE_DIR = Path("./LPW")
OUTPUT_BASE_DIR = Path("./LPW_results_transunet")

# 🚨 주의: 학습 시 사용했던 이미지 해상도와 동일해야 합니다.
IMG_SIZE = 224
NUM_CLASSES = 4 # 0:배경, 1:공막, 2:홍채, 3:동공
# =====================================================================

def get_transunet_model(device):
    config_vit = CONFIGS_ViT_seg['R50-ViT-B_16']
    config_vit.n_classes = NUM_CLASSES
    config_vit.n_skip = 3

    # Grid ZeroDivisionError 방어 코드
    if config_vit.patches.get('grid') is not None:
        config_vit.patches.grid = (int(IMG_SIZE / 16), int(IMG_SIZE / 16))

    model = VisionTransformer(config_vit, img_size=IMG_SIZE, num_classes=NUM_CLASSES)
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
    model.to(device)
    model.eval()
    return model

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"1. TransUNet 가동 및 가중치 로드 중... (Device: {device})")

    # 모델은 루프 밖에서 딱 한 번만 메모리에 올립니다.
    model = get_transunet_model(device)
    print("가중치 로드 완벽 성공!\n")

    # LPW 폴더 내의 모든 .avi 파일 탐색
    video_paths = list(INPUT_BASE_DIR.rglob("*.avi"))
    total_videos = len(video_paths)
    print(f"총 {total_videos}개의 비디오를 발견했습니다. 전체 인퍼런스를 시작합니다.\n")

    with torch.no_grad(): # 파라미터 고정 (VRAM 누수 방지 및 속도 향상)
        for idx, video_path in enumerate(video_paths, 1):
            # 상대 경로 추출 및 결과 저장 디렉토리 생성
            rel_path = video_path.relative_to(INPUT_BASE_DIR)
            out_path = OUTPUT_BASE_DIR / rel_path
            out_path.parent.mkdir(parents=True, exist_ok=True)

            print(f"[{idx}/{total_videos}] 처리 중: {video_path}")

            cap = cv2.VideoCapture(str(video_path))
            fps = cap.get(cv2.CAP_PROP_FPS)
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))

            frame_count = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # 전처리 로직
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                resized_frame = cv2.resize(rgb_frame, (IMG_SIZE, IMG_SIZE))

                img_tensor = torch.from_numpy(resized_frame).float().permute(2, 0, 1) / 255.0
                img_tensor = img_tensor.unsqueeze(0).to(device)

                # 모델 추론
                logits = model(img_tensor)
                pred_mask = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy()

                # 마스크 후처리 및 원본 크기 복구
                mask_resized = cv2.resize(
                    pred_mask.astype(np.uint8),
                    (w, h),
                    interpolation=cv2.INTER_NEAREST
                )

                # 동공(3번 클래스) 붉은색 오버레이
                result_frame = frame.copy()
                result_frame[mask_resized == 3] = [0, 0, 255]

                out.write(result_frame)
                frame_count += 1

            cap.release()
            out.release()
            print(f"  -> 완료! ({frame_count} 프레임) 저장됨: {out_path}")

    print("\n모든 비디오 인퍼런스 처리가 완료되었습니다!")

if __name__ == "__main__":
    main()