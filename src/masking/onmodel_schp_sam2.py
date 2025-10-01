from __future__ import annotations
import numpy as np, cv2
from .base import Masker
from rembg import remove
from ..schemas.config import OnModelColorPriorConfig

def _largest_component(binm: np.ndarray) -> np.ndarray:
    num, labels, stats, _ = cv2.connectedComponentsWithStats(binm, connectivity=8)
    if num <= 1: return binm
    largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    return np.where(labels == largest, 255, 0).astype(np.uint8)

def _skin_mask(image_bgr: np.ndarray) -> np.ndarray:
    # Intersection of classic YCrCb and HSV skin ranges (safer for white/grey garments)
    ycrcb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2YCrCb)
    skin_y = cv2.inRange(ycrcb, (0, 133, 77), (255, 180, 127))  # tighter
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    skin_h = cv2.inRange(hsv, (0, 40, 60), (25, 200, 255))
    skin = cv2.bitwise_and(skin_y, skin_h)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    skin = cv2.erode(skin, k, iterations=1)   # conservative removal
    return skin

def _person_fg(image_bgr: np.ndarray) -> np.ndarray:
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    rgba = remove(rgb)
    alpha = rgba[...,3] if rgba.shape[-1] == 4 else np.full(rgb.shape[:2], 255, np.uint8)
    _, binm = cv2.threshold(alpha, 0, 255, cv2.THRESH_BINARY)
    return _largest_component(binm)

def _ab_median_and_mad(ref_bgr: np.ndarray, ref_mask: np.ndarray) -> tuple[float,float,float]:
    lab = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    m = ref_mask.astype(bool)
    a = lab[...,1][m]; b = lab[...,2][m]
    if a.size == 0: return 0.0, 0.0, 25.0
    am = float(np.median(a)); bm = float(np.median(b))
    r = np.sqrt((a - am)**2 + (b - bm)**2)
    mad = float(np.median(np.abs(r - np.median(r)))) + 1e-6
    return am, bm, mad

def _kmeans_ab_within(masked_bgr: np.ndarray, fg_mask: np.ndarray, K: int = 3) -> tuple[np.ndarray, np.ndarray]:
    lab = cv2.cvtColor(masked_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    m = fg_mask.astype(bool)
    pts = np.stack([lab[...,1][m], lab[...,2][m]], axis=1)
    if pts.shape[0] < 100:
        return np.zeros_like(fg_mask), np.array([])
    # kmeans on a,b
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.5)
    compactness, labels, centers = cv2.kmeans(pts.astype(np.float32), K, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
    # rebuild mask for each cluster
    out = np.zeros_like(fg_mask)
    out_idx = 0  # placeholder, caller decides which center to pick
    return labels.reshape(-1), centers

class OnModelColorPriorMasker(Masker):
    def __init__(self, cfg: OnModelColorPriorConfig) -> None:
        self.cfg = cfg

    def get_mask_with_ref(self, image_bgr: np.ndarray,
                          ref_bgr: np.ndarray, ref_mask_core: np.ndarray,
                          logger=None) -> np.ndarray:
        fg = _person_fg(image_bgr)
        skin = _skin_mask(image_bgr)
        fg = cv2.bitwise_and(fg, cv2.bitwise_not(skin))

        a_ref, b_ref, mad = _ab_median_and_mad(ref_bgr, ref_mask_core)
        lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
        a = lab[...,1]; b = lab[...,2]
        dist = np.sqrt((a - a_ref)**2 + (b - b_ref)**2)

        tau = max(self.cfg.tau_base, self.cfg.tau_k * mad)
        color_like = (dist <= tau).astype(np.uint8) * 255
        cand = cv2.bitwise_and(fg, color_like)

        k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
        cand = cv2.morphologyEx(cand, cv2.MORPH_OPEN,  k5, iterations=1)
        cand = cv2.morphologyEx(cand, cv2.MORPH_CLOSE, k5, iterations=2)
        if self.cfg.dilate_px > 0:
            kd = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.cfg.dilate_px*2+1,)*2)
            cand = cv2.dilate(cand, kd, iterations=1)

        mask = _largest_component(cand)
        px = int(mask.sum() // 255)

        # --- K-means fallback if τ under-segments ---
        if px < self.cfg.min_mask_pixels:
            m = fg.astype(bool)
            lab_full = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
            pts = np.stack([lab_full[...,1][m], lab_full[...,2][m]], axis=1)
            if pts.shape[0] >= 100:
                K = 3
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.5)
                _, labels, centers = cv2.kmeans(pts.astype(np.float32), K, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
                # choose center closest to (a_ref, b_ref)
                d = np.sqrt((centers[:,0]-a_ref)**2 + (centers[:,1]-b_ref)**2)
                best = int(np.argmin(d))
                sel = (labels.ravel() == best)
                mask = np.zeros_like(fg)
                mask[m] = (sel.astype(np.uint8) * 255)
                mask = _largest_component(mask)
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k5, iterations=2)
                px = int(mask.sum() // 255)

        # fill holes (logos)
        h, w = mask.shape
        ff = mask.copy()
        cv2.floodFill(ff, np.zeros((h+2, w+2), np.uint8), (0, 0), 128)
        holes = (ff == 0).astype(np.uint8) * 255
        mask = cv2.bitwise_or(mask, holes)

        if logger is not None:
            logger.info(f"🎯 Color-prior τ={tau:.2f} | mask_px={px}")

        return mask

    def get_mask(self, image_bgr: np.ndarray) -> np.ndarray:
        fg = _person_fg(image_bgr)
        skin = _skin_mask(image_bgr)
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
        base = cv2.bitwise_and(fg, cv2.bitwise_not(skin))
        base = cv2.morphologyEx(base, cv2.MORPH_CLOSE, k, iterations=2)
        return _largest_component(base)
