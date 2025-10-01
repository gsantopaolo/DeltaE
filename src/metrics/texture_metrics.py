from __future__ import annotations
import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim

def ssim_L(a_bgr: np.ndarray, b_bgr: np.ndarray, mask: np.ndarray) -> float:
    a_y = cv2.cvtColor(a_bgr, cv2.COLOR_BGR2GRAY)
    b_y = cv2.cvtColor(b_bgr, cv2.COLOR_BGR2GRAY)
    m = mask.astype(bool)
    if m.sum() < 10:
        return float("nan")
    # Crop to bbox for SSIM speed
    ys, xs = np.where(m)
    y0, y1 = ys.min(), ys.max()+1
    x0, x1 = xs.min(), xs.max()+1
    return float(ssim(a_y[y0:y1, x0:x1], b_y[y0:y1, x0:x1]))
