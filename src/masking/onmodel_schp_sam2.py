# src/masking/onmodel_schp_sam2.py
from __future__ import annotations
from typing import Tuple
import numpy as np
import cv2
from rembg import remove

from .base import Masker
from ..schemas.config import OnModelColorPriorConfig

def _largest_component(binm: np.ndarray) -> np.ndarray:
    num, labels, stats, _ = cv2.connectedComponentsWithStats(binm, connectivity=8)
    if num <= 1:
        return binm
    largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    return np.where(labels == largest, 255, 0).astype(np.uint8)

def _skin_mask(image_bgr: np.ndarray) -> np.ndarray:
    # Intersection of classic YCrCb and HSV skin ranges (conservative)
    ycrcb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2YCrCb)
    skin_y = cv2.inRange(ycrcb, (0, 133, 77), (255, 180, 127))
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    skin_h = cv2.inRange(hsv, (0, 40, 60), (25, 200, 255))
    skin = cv2.bitwise_and(skin_y, skin_h)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    skin = cv2.erode(skin, k, iterations=1)
    return skin

def _person_fg(image_bgr: np.ndarray) -> np.ndarray:
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    rgba = remove(rgb)  # returns RGBA
    alpha = rgba[..., 3] if rgba.shape[-1] == 4 else np.full(rgb.shape[:2], 255, np.uint8)
    _, binm = cv2.threshold(alpha, 0, 255, cv2.THRESH_BINARY)
    return _largest_component(binm)

def _ab_median_and_mad(ref_bgr: np.ndarray, ref_mask: np.ndarray) -> Tuple[float, float, float]:
    lab = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    m = ref_mask.astype(bool)
    a = lab[..., 1][m]; b = lab[..., 2][m]
    if a.size == 0:
        return 0.0, 0.0, 25.0
    am = float(np.median(a)); bm = float(np.median(b))
    r = np.sqrt((a - am) ** 2 + (b - bm) ** 2)
    mad = float(np.median(np.abs(r - np.median(r)))) + 1e-6
    return am, bm, mad

class OnModelColorPriorMasker(Masker):
    def __init__(self, cfg: OnModelColorPriorConfig) -> None:
        self.cfg = cfg

    def get_mask_with_ref(self,
                          image_bgr: np.ndarray,
                          ref_bgr: np.ndarray,
                          ref_mask_core: np.ndarray,
                          logger=None) -> np.ndarray:
        # Person FG ∩ ¬Skin
        fg = _person_fg(image_bgr)
        skin = _skin_mask(image_bgr)
        fg_noskin = cv2.bitwise_and(fg, cv2.bitwise_not(skin))

        # Color prior in Lab (a*, b*) around still-life median with MAD-based τ
        a_ref, b_ref, mad = _ab_median_and_mad(ref_bgr, ref_mask_core)
        lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
        a = lab[..., 1]; b = lab[..., 2]
        dist = np.sqrt((a - a_ref) ** 2 + (b - b_ref) ** 2)
        tau = max(self.cfg.tau_base, self.cfg.tau_k * mad)

        color_like = (dist <= tau).astype(np.uint8) * 255
        cand = cv2.bitwise_and(fg_noskin, color_like)

        # Clean up
        k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        cand = cv2.morphologyEx(cand, cv2.MORPH_OPEN, k5, iterations=1)
        cand = cv2.morphologyEx(cand, cv2.MORPH_CLOSE, k5, iterations=2)
        if self.cfg.dilate_px > 0:
            kd = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.cfg.dilate_px * 2 + 1,) * 2)
            cand = cv2.dilate(cand, kd, iterations=1)

        mask = _largest_component(cand)
        px = int(mask.sum() // 255)

        # K-means fallback if τ under-segments
        if px < self.cfg.min_mask_pixels:
            m = fg_noskin.astype(bool)
            if m.any():
                pts = np.stack([a[m], b[m]], axis=1).astype(np.float32)
                if pts.shape[0] >= 100:
                    K = 3
                    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.5)
                    _, labels, centers = cv2.kmeans(pts, K, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
                    d = np.sqrt((centers[:, 0] - a_ref) ** 2 + (centers[:, 1] - b_ref) ** 2)
                    best = int(np.argmin(d))
                    sel = (labels.ravel() == best)
                    kmask = np.zeros_like(fg_noskin)
                    kmask[m] = (sel.astype(np.uint8) * 255)
                    kmask = _largest_component(kmask)
                    kmask = cv2.morphologyEx(kmask, cv2.MORPH_CLOSE, k5, iterations=2)
                    mask = kmask
                    px = int(mask.sum() // 255)

        # Fill small holes (logos)
        h, w = mask.shape
        ff = mask.copy()
        cv2.floodFill(ff, np.zeros((h + 2, w + 2), np.uint8), (0, 0), 128)
        holes = (ff == 0).astype(np.uint8) * 255
        mask = cv2.bitwise_or(mask, holes)

        if logger is not None:
            logger.info(f"🎯 Color-prior τ={tau:.2f} | mask_px={px}")

        return (mask > 0).astype(np.uint8) * 255

    def get_mask(self, image_bgr: np.ndarray) -> np.ndarray:
        fg = _person_fg(image_bgr)
        skin = _skin_mask(image_bgr)
        base = cv2.bitwise_and(fg, cv2.bitwise_not(skin))
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        base = cv2.morphologyEx(base, cv2.MORPH_CLOSE, k, iterations=2)
        return _largest_component(base)
