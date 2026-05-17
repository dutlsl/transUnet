import os
import cv2
import torch
import numpy as np
import types
from pathlib import Path
import torchvision.transforms.functional as TF

# TransUNet imports
from networks.vit_seg_modeling import VisionTransformer
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg

WEIGHTS_PATH = "./models_transunet/best_model.pth"
IMG_SIZE = 224
NUM_CLASSES = 4
PUPIL_CLASS_ID = 3

def _make_blur(sigma):
    k_size = int(4 * sigma + 0.5)
    if k_size % 2 == 0: k_size += 1
    if k_size < 3: k_size = 3
    return k_size

def patched_decoder_forward(self, hidden_states, features=None):
    sigma0 = getattr(self, 'filter_sigma0', 1.0)
    sigma1 = getattr(self, 'filter_sigma1', 0.5)
    if features is not None:
        features = list(features)
        if sigma0 > 0 and len(features) > 0:
            k0 = _make_blur(sigma0)
            features[0] = TF.gaussian_blur(features[0], kernel_size=[k0, k0], sigma=[sigma0, sigma0])
        if sigma1 > 0 and len(features) > 1:
            k1 = _make_blur(sigma1)
            features[1] = TF.gaussian_blur(features[1], kernel_size=[k1, k1], sigma=[sigma1, sigma1])

    B, n_patch, hidden = hidden_states.size()
    h, w = int(np.sqrt(n_patch)), int(np.sqrt(n_patch))
    x = hidden_states.permute(0, 2, 1)
    x = x.contiguous().view(B, hidden, h, w)
    x = self.conv_more(x)
    for i, decoder_block in enumerate(self.blocks):
        skip = features[i] if (features is not None and i < self.config.n_skip) else None
        x = decoder_block(x, skip=skip)
    return x

def get_model(device, sigma0=1.0, sigma1=0.5):
    config_vit = CONFIGS_ViT_seg['R50-ViT-B_16']
    config_vit.n_classes = NUM_CLASSES
    config_vit.n_skip = 3
    if config_vit.patches.get('grid') is not None:
        config_vit.patches.grid = (int(IMG_SIZE / 16), int(IMG_SIZE / 16))
    model = VisionTransformer(config_vit, img_size=IMG_SIZE, num_classes=NUM_CLASSES)
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
    model.decoder.filter_sigma0 = sigma0
    model.decoder.filter_sigma1 = sigma1
    model.decoder.forward = types.MethodType(patched_decoder_forward, model.decoder)
    model.to(device)
    model.eval()
    return model

def ellipse_postprocess(pred_mask, pupil_id=3, min_points=5):
    binary = (pred_mask == pupil_id).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return pred_mask
        
    largest = max(contours, key=cv2.contourArea)
    largest_area = cv2.contourArea(largest)
    if largest_area == 0 or len(largest) < min_points:
        return pred_mask
        
    M_large = cv2.moments(largest)
    cx_large = M_large["m10"] / M_large["m00"] if M_large["m00"] != 0 else 0
    cy_large = M_large["m01"] / M_large["m00"] if M_large["m00"] != 0 else 0
    
    valid_points = [largest]
    for cnt in contours:
        if cnt is largest:
            continue
        area = cv2.contourArea(cnt)
        if area < largest_area * 0.10:
            continue
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cx = M["m10"] / M["m00"]
            cy = M["m01"] / M["m00"]
            dist = np.sqrt((cx - cx_large)**2 + (cy - cy_large)**2)
            if dist > 50:
                continue
        valid_points.append(cnt)
        
    all_points = np.vstack(valid_points)
    if len(all_points) < min_points:
        return pred_mask
        
    ellipse = cv2.fitEllipse(all_points)
    if ellipse[1][0] <= 0 or ellipse[1][1] <= 0:
        return pred_mask
        
    result = pred_mask.copy()
    result[result == pupil_id] = 0
    try:
        cv2.ellipse(result, ellipse, pupil_id, -1)
    except cv2.error:
        return pred_mask
        
    return result

def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    img_path = Path("Swirski_Dataset/p1-left/frames/206-eye.png")
    
    if not img_path.exists():
        print(f"Error: Image not found at {img_path}")
        return
        
    img = cv2.imread(str(img_path))
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    resized = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE))
    tensor = torch.from_numpy(resized).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    tensor = tensor.to(device)

    # 1. Baseline
    model = get_model(device, sigma0=0.0, sigma1=0.0)
    with torch.no_grad():
        logits = model(tensor)
        pred = logits.argmax(1).squeeze(0).cpu().numpy().astype(np.uint8)
    save_overlay(gray, pred, "baseline.png")

    # 2. Ellipse Only
    pred_ell = ellipse_postprocess(pred, PUPIL_CLASS_ID)
    save_overlay(gray, pred_ell, "ellipse_only.png")

    # 3. SF Only
    model = get_model(device, sigma0=1.0, sigma1=0.5)
    with torch.no_grad():
        logits = model(tensor)
        pred_sf = logits.argmax(1).squeeze(0).cpu().numpy().astype(np.uint8)
    save_overlay(gray, pred_sf, "sf_only.png")

    # 4. Final (SF + Ellipse)
    pred_final = ellipse_postprocess(pred_sf, PUPIL_CLASS_ID)
    save_overlay(gray, pred_final, "final.png")
    
    print("All 4 images generated successfully.")

def save_overlay(gray, pred, filename):
    h, w = gray.shape[:2]
    pred_full = cv2.resize(pred, (w, h), interpolation=cv2.INTER_NEAREST)
    img_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    ov = img_bgr.copy()
    ov[pred_full == PUPIL_CLASS_ID] = [0, 0, 255]
    cv2.addWeighted(ov, 0.5, img_bgr, 0.5, 0, img_bgr)
    cv2.imwrite(filename, img_bgr)

if __name__ == "__main__":
    main()
