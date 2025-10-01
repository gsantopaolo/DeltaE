from __future__ import annotations
import numpy as np
import cv2
from skimage.color import rgb2lab, deltaE_ciede2000

def _lab_image(bgr: np.ndarray) -> np.ndarray:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return rgb2lab(rgb)

def region_lab_median(bgr: np.ndarray, mask: np.ndarray) -> tuple[float, float, float]:
    lab = _lab_image(bgr)
    m = mask.astype(bool)
    if m.sum() == 0:
        return float("nan"), float("nan"), float("nan")
    return (float(np.median(lab[..., 0][m])),
            float(np.median(lab[..., 1][m])),
            float(np.median(lab[..., 2][m])))

def deltaE_between_medians(ref_bgr: np.ndarray, ref_mask: np.ndarray,
                           test_bgr: np.ndarray, test_mask: np.ndarray) -> float:
    Lr, ar, br = region_lab_median(ref_bgr, ref_mask)
    Lt, at, bt = region_lab_median(test_bgr, test_mask)
    if any(np.isnan(x) for x in (Lr, ar, br, Lt, at, bt)):
        return float("nan")
    ref = np.array([[[Lr, ar, br]]], dtype=np.float32)
    tst = np.array([[[Lt, at, bt]]], dtype=np.float32)
    return float(deltaE_ciede2000(ref, tst)[0, 0])

def deltaE_q_to_ref_median(test_bgr: np.ndarray, test_mask: np.ndarray,
                           ref_bgr: np.ndarray, ref_mask: np.ndarray,
                           q: float = 95.0) -> float:
    lab_t = _lab_image(test_bgr)
    Lr, ar, br = region_lab_median(ref_bgr, ref_mask)
    if any(np.isnan(x) for x in (Lr, ar, br)):
        return float("nan")
    ref = np.array([[[Lr, ar, br]]], dtype=np.float32)
    dE = deltaE_ciede2000(lab_t, np.tile(ref, (lab_t.shape[0], lab_t.shape[1], 1)))
    v = dE[test_mask.astype(bool)]
    if v.size == 0:
        return float("nan")
    return float(np.percentile(v, q))
