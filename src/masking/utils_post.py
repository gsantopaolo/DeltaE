from __future__ import annotations
import numpy as np, cv2
from typing import Tuple

def largest_component(binm: np.ndarray) -> np.ndarray:
    num, labels, stats, _ = cv2.connectedComponentsWithStats(binm, connectivity=8)
    if num <= 1:
        return binm
    largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    return np.where(labels == largest, 255, 0).astype(np.uint8)

def fill_holes(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    ff = mask.copy()
    cv2.floodFill(ff, np.zeros((h+2, w+2), np.uint8), (0, 0), 128)
    holes = (ff == 0).astype(np.uint8) * 255
    return cv2.bitwise_or(mask, holes)

def band_mask(h: int, top_pct: float, bot_pct: float) -> np.ndarray:
    y0 = int(max(0.0, min(1.0, top_pct)) * h)
    y1 = int(max(0.0, min(1.0, bot_pct)) * h)
    band = np.zeros((h, 1), np.uint8)
    band[y0:y1, 0] = 255
    return band

def erode(mask: np.ndarray, px: int) -> np.ndarray:
    if px <= 0: return mask
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (px*2+1, px*2+1))
    return cv2.erode(mask, k)

def open_close(mask: np.ndarray, ksize: int = 5, iters_open: int = 1, iters_close: int = 2) -> np.ndarray:
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    m = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=iters_open)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k, iterations=iters_close)
    return m

def crf_refine(image_bgr: np.ndarray, prob_fg: np.ndarray) -> np.ndarray:
    """Optional CRF refine (requires pydensecrf). prob_fg in [0,1]. Returns 0/255."""
    try:
        import pydensecrf.densecrf as dcrf
        import pydensecrf.utils as dutils
    except Exception:
        return (prob_fg >= 0.5).astype(np.uint8) * 255
    H, W = prob_fg.shape
    unary = np.stack([1.0 - prob_fg, prob_fg], axis=0).astype(np.float32)
    unary = -np.log(np.clip(unary, 1e-6, 1.0))
    unary = unary.reshape((2, -1))
    d = dcrf.DenseCRF2D(W, H, 2)
    d.setUnaryEnergy(unary)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    d.addPairwiseGaussian(sxy=3, compat=3)
    d.addPairwiseBilateral(sxy=20, srgb=5, rgbim=image_rgb, compat=10)
    Q = d.inference(5)
    prob1 = np.array(Q)[1].reshape(H, W)
    return (prob1 >= 0.5).astype(np.uint8) * 255
