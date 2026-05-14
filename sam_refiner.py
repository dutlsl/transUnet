import cv2
import torch
import numpy as np
from segment_anything import sam_model_registry, SamPredictor

class SamRefiner:
    def __init__(self, model_type="vit_h", checkpoint_path="sam_vit_h_4b8939.pth", device="cuda:0"):
        print(f"SAM ({model_type}) 로드 중... (경로: {checkpoint_path})")
        sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
        sam.to(device=device)
        self.predictor = SamPredictor(sam)
        self.device = device

    def extract_largest_bbox(self, mask, class_id=3, margin=10):
        """
        위양성(False Positive) 제거를 위해 가장 큰 덩어리(Largest Connected Component)의 Box만 추출합니다.
        """
        binary_mask = np.uint8(mask == class_id)

        # 예측된 마스크가 없으면 None 반환
        if np.sum(binary_mask) == 0:
            return None

        # 연결 요소 레이블링으로 가장 큰 덩어리 찾기
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)

        # 배경(0번 레이블)을 제외하고 가장 넓이가 큰 레이블 탐색
        if num_labels <= 1:
            return None

        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        x = stats[largest_label, cv2.CC_STAT_LEFT]
        y = stats[largest_label, cv2.CC_STAT_TOP]
        w = stats[largest_label, cv2.CC_STAT_WIDTH]
        h = stats[largest_label, cv2.CC_STAT_HEIGHT]

        # SAM이 윤곽선을 잘 잡도록 Box에 여유 공간(margin) 부여
        img_h, img_w = mask.shape
        x_min = max(0, x - margin)
        y_min = max(0, y - margin)
        x_max = min(img_w, x + w + margin)
        y_max = min(img_h, y + h + margin)

        return np.array([x_min, y_min, x_max, y_max])

    def refine_mask(self, raw_image, coarse_mask, class_id=3):
        """
        원본 이미지와 거친 마스크를 받아 SAM으로 정제된 마스크를 반환합니다.
        """
        bbox = self.extract_largest_bbox(coarse_mask, class_id)

        # 유효한 Bbox를 찾지 못했으면 원본 마스크 그대로 반환
        if bbox is None:
            return coarse_mask

        # SAM은 RGB 이미지를 요구하므로 변환
        if len(raw_image.shape) == 2:
            image_rgb = cv2.cvtColor(raw_image, cv2.COLOR_GRAY2RGB)
        else:
            image_rgb = raw_image

        self.predictor.set_image(image_rgb)

        # Bbox 프롬프트를 이용해 마스크 예측
        masks, scores, _ = self.predictor.predict(
            box=bbox[None, :],
            multimask_output=False # 단일 최적 마스크만 반환
        )

        refined_mask = np.zeros_like(coarse_mask)
        refined_mask[masks[0]] = class_id

        return refined_mask