# TransUNet Cross-Domain Pupil Segmentation — 기술 문서

## 1. 프로젝트 개요

### 1.1 목표
OpenEDS(근적외선) 데이터셋으로 학습된 TransUNet R50-ViT-B_16 모델의 가중치를 **재학습 없이** Swirski(가시광) 데이터셋의 **Pupil Segmentation** 성능을 개선합니다.

### 1.2 핵심 제약
- OpenEDS 외에 4-class eye segmentation 라벨이 공개된 데이터셋이 거의 없으므로, 현재 가중치를 최대한 활용해야 합니다.
- Swirski 데이터셋에는 pupil-ellipse GT만 존재합니다 (4-class 라벨 없음).
- 추론 시간 증가를 최소화해야 합니다.

---

## 2. 모델 아키텍처 분석

### 2.1 TransUNet R50-ViT-B_16 데이터 흐름

```
입력 (3, 224, 224)
  ↓
ResNet Root — Conv7×7, GN, ReLU → (64, 112, 112) → Skip #2 로 분기
  ↓ MaxPool
ResNet Block1 — 3 Bottleneck units → (256, 56, 56) → Skip #1 로 분기
  ↓
ResNet Block2 — 4 units, stride=2 → (512, 28, 28) → Skip #0 로 분기
  ↓
ResNet Block3 — 9 units, stride=2 → (1024, 14, 14)
  ↓ Patch Embedding (Conv1×1)
ViT Encoder — 12 Transformer Blocks, (196, 768)
  ↓ Reshape to (768, 14, 14)
conv_more — 768→512, 3×3+BN+ReLU → (512, 14, 14) [바틀넥]
  ↓
DecoderBlock 0 — Up 14→28, concat Skip #0 (512ch) → (256, 28, 28)
DecoderBlock 1 — Up 28→56, concat Skip #1 (256ch) → (128, 56, 56)
DecoderBlock 2 — Up 56→112, concat Skip #2 (64ch) → (64, 112, 112)
DecoderBlock 3 — Up 112→224, no skip → (16, 224, 224)
  ↓
SegHead — Conv3×3 → (4, 224, 224) → argmax → 예측 마스크
```

### 2.2 config (R50-ViT-B_16)
- `decoder_channels`: (256, 128, 64, 16)
- `skip_channels`: [512, 256, 64, 16]
- `n_skip`: 3 (Skip #0, #1, #2 사용)
- `n_classes`: 4 (Background=0, Sclera=1, Iris=2, Pupil=3)
- `hidden_size`: 768
- `transformer.num_layers`: 12, `num_heads`: 12

### 2.3 관련 소스 파일
- `networks/vit_seg_modeling.py` — VisionTransformer, Transformer, DecoderCup, DecoderBlock, SegmentationHead
- `networks/vit_seg_modeling_resnet_skip.py` — ResNetV2, PreActBottleneck
- `networks/vit_seg_configs.py` — 모델별 config 정의

---

## 3. Swirski 데이터셋 구조

### 3.1 디렉토리
```
Swirski_Dataset/
├── p1-left/
│   ├── frames/          # 0-eye.png, 1-eye.png, ... (파일명: zero-padding 없음)
│   └── pupil-ellipses.txt
├── p1-right/
├── p2-left/
└── p2-right/
```

### 3.2 GT 라벨 포맷 (`pupil-ellipses.txt`)
```
frame_idx | center_x center_y semi_major semi_minor angle_rad
```
예시: `90 | 299.530017 142.801535 56.004241 37.306758 -0.083261`

### 3.3 케이스별 특성

| Case | 총 프레임 | 유효 GT | 주요 특성 |
|------|----------|---------|----------|
| p1-left | ~920 | ~150 | 속눈썹 관통 빈번, 중간 난이도 |
| p1-right | ~750 | ~150 | **과노출 극심**, IR 글레어, 소형 동공(semi-minor ~10px), 최고 난이도 |
| p2-left | ~700 | ~150 | 큰 동공(semi-major ~56px), 비교적 깨끗하나 일부 프레임 무너짐 |
| p2-right | ~700 | ~150 | 가장 쉬움, 큰 동공, 글레어 적음 |

---

## 4. 성능 저하 원인 분석

### 4.1 사전 실험 성적표

| 방법 | Total mIoU | 비고 |
|------|-----------|------|
| **Baseline** (TransUNet zero-shot) | 0.5831 | 기준선 |
| FDA (주파수 진폭 스왑) | 0.4171 (▼28%) | 입력 훼손으로 역효과 |
| TTA (TENT, BN 적응) | 0.5537 (▼5%) | 디코더 BN만 적응, 부족 |
| SAM ViT-H Refine | 0.7278 (▲25%) | 추론 비용 비현실적 (ViT-H 2.5GB) |

### 4.2 시각적 실패 패턴 3종

**패턴 A — 과노출/글레어로 인한 완전 붕괴:**
- p1-right에서 지배적. IR 조명이 과도하여 동공 영역이 하얗게 washout됨.
- 동공 semi-minor가 10~12px 수준으로 극히 작아 모델이 탐지 자체를 실패.
- IoU: 0.00~0.22 범위

**패턴 B — 속눈썹 관통에 의한 분절(Fragmentation):**
- 전 케이스에 걸쳐 발생. OpenEDS(근적외선)에서는 속눈썹이 거의 안 보이므로 완전히 unseen 패턴.
- 동공 경계(edge)는 정확히 따라가지만, 속눈썹이 관통하는 부분에서 내부가 끊김.
- 가장 큰 blob만 남기면 절반이 날아감.
- IoU: 0.25~0.45 범위

**패턴 C — 경계만 잡고 내부 미채움(Hollow Ring):**
- 동공 경계의 호(arc)만 인식하고 내부를 칠하지 못함.
- Swirski 동공 내부 밝기 프로파일이 OpenEDS와 달라 "동공 아님"으로 판정.
- 경계만 잡혔으면 기하학적 타원 피팅으로 해결 가능 (cv2.fitEllipse).
- IoU: 0.05~0.35 범위

### 4.3 주파수 분석(Feature Probe) 결과

`probe_features.py`로 TransUNet 내부 feature map의 고주파 에너지 비율을 실패/성공 프레임 8개에 대해 레이어별로 측정한 결과:

**핵심 발견: 바틀넥(14×14)이 아닌 Skip Connection이 실패의 원인**

| Layer (해상도) | 실패 HF ratio (평균) | 성공 HF ratio (평균) | 격차 |
|---------------|--------------------|--------------------|------|
| resnet_b3 / dec_conv_more (14×14) | 0.130 / 0.176 | 0.117 / 0.201 | Δ0.013 (미미, **역전됨**) |
| **resnet_b1 (56×56, Skip #1)** | **0.185** | **0.114** | **Δ0.071 (62% 증가)** |
| resnet_b2 (28×28, Skip #0) | 0.046 | 0.024 | Δ0.022 (92% 증가) |
| resnet_root (112×112, Skip #2) | 0.253 | 0.207 | Δ0.046 |

- **바틀넥(14×14)**은 실패/성공 간 HF 차이가 미미하고 심지어 역전됨 → 주파수 필터 삽입에 부적합.
- **Skip #1 (56×56)**에서 실패 시 고주파 에너지가 62% 더 높음 → 속눈썹 에지가 이 레이어에 집중.
- ViT main path는 12 Transformer의 글로벌 self-attention으로 노이즈를 자연 감쇠시키지만, Skip은 ViT를 우회하여 디코더에 직통 전달.

**결론: Skip Connection이 unseen domain 아티팩트(속눈썹, 글레어)를 디코더에 날것으로 주입하는 구조적 취약점.**

---

## 5. 개선 전략

### 5.1 Skip LP Filter (Gaussian Blur)

**원리:** Skip feature에 Gaussian blur를 적용하여 속눈썹 등 고주파 아티팩트를 감쇠시킵니다. 기존 가중치를 일절 변경하지 않으며, 순전파 경로에 고정 Gaussian 커널을 끼워넣을 뿐입니다.

**적용 위치:** `DecoderCup.forward()` 내부, Skip #0 (28×28)과 Skip #1 (56×56)에 합류 직전.

**하이퍼파라미터:**
- `sigma` (σ): Gaussian blur의 표준편차. 높을수록 강한 low-pass.
- 최적 σ는 sweep으로 결정 (0.5~3.0 범위).

### 5.2 Ellipse Fitting 후처리

**원리:** 모델이 동공 경계(edge)는 잡지만 내부를 채우지 못하거나 분절된 경우, 가장 큰 blob의 contour에 타원을 피팅하여 내부를 채웁니다.

**구현:** `cv2.fitEllipse(largest_contour)` → `cv2.ellipse(result, ellipse, pupil_id, -1)`

**Fallback:** 예측 픽셀이 0이거나 contour 점이 5개 미만이면 원본 예측을 그대로 유지합니다.

---

## 6. 사용자 가이드

### 6.1 환경 설정
```bash
conda activate transUnet
# GPU: CUDA_VISIBLE_DEVICES="0" (스크립트 내부에서 고정)
```

### 6.2 추론 스크립트 인터페이스

**파일:** `zeroshot_swirski_skipfilter.py`

```bash
# 기본 사용법
python zeroshot_swirski_skipfilter.py --sigma <값> [--sigma2 <값>] [--ellipse] [--preprocess]
```

**인자:**
| 인자 | 타입 | 기본값 | 설명 |
|------|------|--------|------|
| `--sigma` | float | 0.0 | Skip #0 (28×28), #1 (56×56)에 적용할 Gaussian blur σ. 0.0이면 미적용 (Baseline). |
| `--sigma2` | float | 0.0 | Skip #2 (112×112)에 적용할 Gaussian blur σ. 독립 제어. |
| `--ellipse` | flag | False | cv2.fitEllipse 후처리 활성화 |
| `--preprocess` | flag | False | RITnet 전처리 (Gamma 0.8 + CLAHE grid=8 clip=1.5) 적용 |

**실험 조합 예시:**
```bash
# Exp A: Baseline
python zeroshot_swirski_skipfilter.py --sigma 0.0

# Exp C: Skip Filter (σ=1.0, 최적)
python zeroshot_swirski_skipfilter.py --sigma 1.0

# Exp D: Skip Filter + Ellipse
python zeroshot_swirski_skipfilter.py --sigma 1.0 --ellipse

# RITnet 전처리 + Skip Filter
python zeroshot_swirski_skipfilter.py --sigma 1.0 --preprocess

# Skip #2까지 독립 필터
python zeroshot_swirski_skipfilter.py --sigma 1.0 --sigma2 0.5

# σ sweep
for s in 0.5 1.0 1.5 2.0 2.5 3.0; do
  python zeroshot_swirski_skipfilter.py --sigma $s
done
```

**LPW 데이터셋 평가:**
```bash
# LPW folder-1 baseline
python eval_lpw_skipfilter.py --folder 1 --sigma 0.0

# LPW folder-1 + Skip Filter
python eval_lpw_skipfilter.py --folder 1 --sigma 1.0

# LPW folder-1 + Ellipse (합산 피팅 방식)
python eval_lpw_skipfilter.py --folder 1 --sigma 1.0 --ellipse
```

**배치 실행:**
```bash
chmod +x run_experiments.sh
./run_experiments.sh
```

### 6.3 산출물

| 산출물 | 위치 | 설명 |
|--------|------|------|
| Swirski 점수 CSV | `Swirski_tables/transunet_swirski_skipfilter_sig{σ}_s2{σ2}_ell{O/X}_pre{O/X}.csv` | 프레임별 IoU, Dice |
| Swirski 오버레이 | `Swirski_overlays_skipfilter/sig{σ}_s2{σ2}_ell{O/X}_pre{O/X}/{case}/` | 10 프레임 간격 |
| LPW 점수 CSV | `LPW_tables/lpw_skipfilter_f{N}_sig{σ}_s2{σ2}_ell{O/X}_pre{O/X}.csv` | 프레임별 IoU, Dice |
| LPW 오버레이 | `LPW_overlays_skipfilter/f{N}_sig{σ}_.../{video}/` | 50 프레임 간격 |

**CSV 포맷:**
```
Case,Frame_Idx,IoU,Dice
p1-left,0,0.6234,0.7682
...
```

### 6.4 코드 구조

```
transUnet/
├── zeroshot_swirski.py                # 원본 baseline 추론 스크립트
├── zeroshot_swirski_skipfilter.py     # Swirski 전용 Skip Filter + Ellipse (본 실험용)
├── eval_lpw_skipfilter.py             # LPW 전용 Skip Filter + Ellipse (폴더별 평가)
├── probe_features.py                 # 레이어별 주파수 에너지 분석 도구
├── run_experiments.sh                 # 배치 실행 스크립트
├── networks/
│   ├── vit_seg_modeling.py            # 모델 정의 (DecoderCup, VisionTransformer)
│   ├── vit_seg_modeling_resnet_skip.py # ResNetV2 인코더
│   └── vit_seg_configs.py            # 모델 config
├── models_transunet/
│   └── best_model.pth                # 학습된 가중치 (OpenEDS)
├── Swirski_Dataset/                  # Swirski 테스트 데이터
├── LPW/                              # LPW 테스트 데이터 (폴더별 .avi 비디오)
├── Pupils_in_the_wild_improved/      # LPW GT (폴더별 pupil mask 비디오)
├── Swirski_tables/                   # Swirski 점수 CSV
├── LPW_tables/                       # LPW 점수 CSV
├── Swirski_overlays_*/               # Swirski 오버레이
├── LPW_overlays_skipfilter/          # LPW 오버레이
└── docs/                             # 기술 문서
```

### 6.5 Skip LP Filter 작동 방식 (구현 상세)

기존 `DecoderCup.forward()`를 런타임에 몽키패치합니다. 기존 모델 파일(`vit_seg_modeling.py`)은 수정하지 않습니다.

```python
# 핵심 로직 (zeroshot_swirski_skipfilter.py 내 patched_decoder_forward)
if features is not None and self.filter_sigma > 0:
    k_size = int(4 * self.filter_sigma + 0.5)
    if k_size % 2 == 0: k_size += 1
    features = list(features)
    # Skip #0 (512ch, 28×28)에 Gaussian blur
    features[0] = TF.gaussian_blur(features[0], [k_size, k_size], [sigma, sigma])
    # Skip #1 (256ch, 56×56)에 Gaussian blur
    features[1] = TF.gaussian_blur(features[1], [k_size, k_size], [sigma, sigma])
```

- `torchvision.transforms.functional.gaussian_blur`를 사용하여 GPU 텐서에서 직접 동작.
- 커널 사이즈는 `4σ + 0.5`에서 홀수로 올림.
- 기존 가중치를 전혀 변경하지 않음 — 추론 시 feature를 통과시킬 뿐.
