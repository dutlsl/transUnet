# 🗄️ 실패 방법론 아카이브 (Failed Attempts Archive)

이 문서는 광량 과다(Overexposure, OE) 및 노이즈 극복 실험 과정에서 실패했던 시도들과 그 원인을 간략히 아카이빙하여, 향후 동일한 실패 방법론을 반복하지 않기 위한 지침서입니다.

---

## 1. Fourier Domain Phase Correction (exp11, exp16)
* **내용:** ViT의 attention 피처로부터 위상(Phase) 성분을 추출하여 고해상도 skip 피처의 위상을 교정하려는 시도.
* **실패 원인:**
  1. **수치 불안정성:** `ratio = corrected / (skip + 1e-8)` 연산 시, 스킵 피처 값이 `0`에 가까운 어두운 영역(속눈썹, 그림자)에서 ratio가 $10^7$배 이상 폭발(Feature Explosion)하여 디코더가 오작동함.
  2. **해상도 Mismatch:** ViT의 $14 \times 14$ 저해상도 위상 정보를 $112 \times 112$ skip 피처로 강제 매핑하면서 기존의 고주파 경계 신호가 모두 뭉개짐.

## 2. Attention-Guided Adaptive Sigma (exp12, exp14)
* **내용:** ViT의 spatial attention 가중치를 블러 커널의 가중치로 삼아 중요한 영역(동공)은 블러를 약하게, 중요도가 낮은 영역은 강하게 적용하려는 시도.
* **실패 원인:**
  * **해상도 한계 및 노이즈 보존:** $14 \times 14$ 패치 크기가 속눈썹 에지보다 너무 큼. 패치 내에 동공 경계와 속눈썹이 공존할 때 attention이 높게 나와 **속눈썹 노이즈까지 그대로 보존**해 버리는 역효과 발생.

## 3. Frequency-Selective Channel Suppression (FSCS) (exp13)
* **내용:** 2D FFT를 통해 고주파 노이즈 성분이 과도하게 몰려 있는 특정 피처 채널들을 식별하여 억제(Soft Suppression)하려는 시도.
* **실패 원인:**
  * **에지 훼손:** 동공 경계를 포착하는 중요한 고주파 신호 채널까지 함께 억제되면서, 경계선 세그멘테이션이 전체적으로 뭉개지고 IoU가 급락함.

## 4. Global Domain Normalization (CLAHE / Gamma) (exp17, exp36)
* **내용:** 입력 이미지에 CLAHE 보정 및 감마 조절을 강하게 적용하여 명암비를 끌어올리려는 전처리 시도.
* **실패 원인:**
  * **일반화 성능 훼손:** 과노출이 아닌 정상 노출 프레임들의 도메인 통계(Distribution)까지 변경되어 모델이 본래의 성능을 내지 못하고 전체 성능이 하락함. (도메인 처리는 반드시 **과노출 검출 기반 조건부로만 실행**해야 함)

## 5. Non-Anatomical Fourier Amplitude Blending (exp22, exp32)
* **내용:** 과노출 검출 시 다른 임의의 정상 프레임 진폭을 맹목적으로 가져와 Blending 하려던 시도.
* **실패 원인:**
  * **해부학적Mismatch:** 안구 크기, 렌즈 스케일, 카메라 거리 등이 다른 진폭 스펙트럼을 투영할 경우, 푸리에 복원 후 동공의 형태가 비정상적으로 찌그러지거나 크기가 맞지 않아 세그멘테이션이 완전히 무너짐. (반드시 동일 인물의 contralateral eye 혹은 동일 비디오의 clean template을 매칭해야 함)
