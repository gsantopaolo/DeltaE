from __future__ import annotations
import numpy as np, cv2
from typing import Tuple
from rembg import remove
from ..schemas.config import OnModelColorPriorConfig
from .utils_post import largest_component, open_close, fill_holes, band_mask

def _skin_mask(image_bgr: np.ndarray) -> np.ndarray:
    ycrcb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2YCrCb)
    skin_y = cv2.inRange(ycrcb, (0,133,77), (255,180,127))
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    skin_h = cv2.inRange(hsv, (0, 40, 60), (25, 200, 255))
    skin = cv2.bitwise_and(skin_y, skin_h)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    return cv2.erode(skin, k, iterations=1)

def _person_fg(image_bgr: np.ndarray) -> np.ndarray:
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    rgba = remove(rgb)
    alpha = rgba[...,3] if rgba.shape[-1] == 4 else np.full(rgb.shape[:2],255,np.uint8)
    _, binm = cv2.threshold(alpha, 0, 255, cv2.THRESH_BINARY)
    return largest_component(binm)

def _ref_lab_stats(ref_bgr: np.ndarray, ref_mask: np.ndarray) -> Tuple[float,float,float]:
    lab = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    m = ref_mask.astype(bool)
    L = float(np.median(lab[...,0][m])) if m.any() else 70.0
    a = float(np.median(lab[...,1][m])) if m.any() else 0.0
    b = float(np.median(lab[...,2][m])) if m.any() else 0.0
    return L, a, b

class OnModelColorPrior:
    def __init__(self, cfg: OnModelColorPriorConfig) -> None:
        self.cfg = cfg

    def get_mask_with_ref(self, image_bgr: np.ndarray, ref_bgr: np.ndarray, ref_mask_core: np.ndarray) -> np.ndarray:
        H, W = image_bgr.shape[:2]
        fg = _person_fg(image_bgr)
        skin = _skin_mask(image_bgr)
        core = cv2.bitwise_and(fg, cv2.bitwise_not(skin))

        Lr, ar, br = _ref_lab_stats(ref_bgr, ref_mask_core)
        lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
        a = lab[...,1]; b = lab[...,2]
        dist = np.sqrt((a - ar)**2 + (b - br)**2)
        mad = 10.0  # fallback MAD—we keep tau configurable
        tau = max(self.cfg.tau_base, self.cfg.tau_k * mad)

        # Neutral guard: if ref chroma is small, add L gate and shrink tau
        chroma = np.sqrt(ar*ar + br*br)
        color_like = (dist <= tau).astype(np.uint8) * 255
        if chroma < self.cfg.neutral_chroma_thresh:
            L = lab[...,0]
            lo = Lr + self.cfg.l_gate_lo
            hi = Lr + self.cfg.l_gate_hi
            l_gate = ((L >= lo) & (L <= hi)).astype(np.uint8) * 255
            color_like = cv2.bitwise_and(color_like, l_gate)
            # shrink tau slightly for neutrals
            color_like = cv2.bitwise_and(color_like, (dist <= min(tau, 8.0)).astype(np.uint8)*255)

        # Mid-body vertical band
        band = band_mask(H, self.cfg.band_top_pct, self.cfg.band_bot_pct)
        band = np.repeat(band, W, axis=1)

        cand = cv2.bitwise_and(core, color_like)
        cand = cv2.bitwise_and(cand, band)

        cand = open_close(cand, 5, 1, 2)
        if self.cfg.dilate_px > 0:
            kd = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.cfg.dilate_px*2+1,)*2)
            cand = cv2.dilate(cand, kd, iterations=1)

        mask = largest_component(cand)
        mask = fill_holes(mask)
        return mask

    def get_mask(self, image_bgr: np.ndarray) -> np.ndarray:
        # Fallback without reference: FG - skin + band + cleanup
        H, W = image_bgr.shape[:2]
        fg = _person_fg(image_bgr)
        skin = _skin_mask(image_bgr)
        band = band_mask(H, 0.25, 0.85); band = np.repeat(band, W, axis=1)
        base = cv2.bitwise_and(cv2.bitwise_and(fg, cv2.bitwise_not(skin)), band)
        base = open_close(base, 5, 1, 2)
        return largest_component(base)
