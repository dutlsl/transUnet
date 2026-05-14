import cv2
import numpy as np

def apply_fda(target_img: np.ndarray, source_img: np.ndarray, beta: float) -> np.ndarray:
    """
    [FDA 핵심 로직]
    타겟 이미지(ex. Swirski)의 진폭(Amplitude) 스펙트럼 중 저주파(중심부) 영역을
    소스 이미지(ex. OpenEDS)의 진폭 스펙트럼으로 교체합니다.
    위상(Phase, 구조적 정보)은 타겟을 그대로 유지합니다.
    """
    h, w = target_img.shape

    # 1. 소스 이미지를 타겟 이미지와 동일한 해상도로 맞춤
    src_resized = cv2.resize(source_img, (w, h), interpolation=cv2.INTER_CUBIC)

    # 2. 푸리에 변환 (FFT) 및 중심 이동 (저주파를 중앙으로)
    fft_trg = np.fft.fftshift(np.fft.fft2(target_img))
    fft_src = np.fft.fftshift(np.fft.fft2(src_resized))

    # 3. 진폭(Amplitude)과 위상(Phase) 분리
    amp_trg, pha_trg = np.abs(fft_trg), np.angle(fft_trg)
    amp_src = np.abs(fft_src)

    # 4. 저주파 영역 마스크(Mask) 생성
    b_h, b_w = int(h * beta), int(w * beta)
    c_h, c_w = h // 2, w // 2

    mask = np.zeros_like(amp_trg)
    mask[c_h - b_h : c_h + b_h, c_w - b_w : c_w + b_w] = 1

    # 5. 진폭 스왑: 타겟 고주파(1-mask) + 소스 저주파(mask)
    amp_mod = amp_trg * (1 - mask) + amp_src * mask

    # 6. 위상 복원 및 역 푸리에 변환 (iFFT)
    fft_mod = amp_mod * np.exp(1j * pha_trg)
    img_mod = np.real(np.fft.ifft2(np.fft.ifftshift(fft_mod)))

    # 안전하게 0~255 uint8 값으로 클리핑 후 반환
    return np.clip(img_mod, 0, 255).astype(np.uint8)