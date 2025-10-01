from __future__ import annotations
import numpy as np
import cv2
from skimage.color import rgb2lab, lab2rgb, deltaE_ciede2000

def _median_ab(img_bgr: np.ndarray, mask: np.ndarray) -> tuple[float, float]:
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    lab = rgb2lab(rgb)
    m = mask.astype(bool)
    return (float(np.median(lab[..., 1][m])),
            float(np.median(lab[..., 2][m])))

def _lab_to_lch(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    C = np.sqrt(a*a + b*b)
    h = np.arctan2(b, a)
    return C, h

def _lch_to_ab(C: np.ndarray, h: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return C * np.cos(h), C * np.sin(h)

def _clip_lab_inplace(Lab: np.ndarray) -> None:
    """
    Clamp Lab to a safe gamut before converting to sRGB.
    L in [0,100], a,b in roughly [-128,127] to avoid out-of-gamut artifacts.
    """
    Lab[..., 0] = np.clip(Lab[..., 0], 0.0, 100.0)
    Lab[..., 1] = np.clip(Lab[..., 1], -128.0, 127.0)
    Lab[..., 2] = np.clip(Lab[..., 2], -128.0, 127.0)

class ClassicalLabCorrector:
    """
    Estimate mapping on the *core* masks (robust), apply it to the *full* on-model mask.
    Preserve L; rotate hue & scale chroma; tiny ΔE feedback to hit target.
    """
    def __init__(self, deltaE_target: float = 2.0) -> None:
        self.deltaE_target = deltaE_target

    def correct(self,
                on_model_bgr: np.ndarray,
                on_model_mask_core: np.ndarray,
                on_model_mask_full: np.ndarray,
                ref_bgr: np.ndarray,
                ref_mask_core: np.ndarray) -> np.ndarray:

        # Prepare Lab images
        om_rgb = cv2.cvtColor(on_model_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        rf_rgb = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        om_lab = rgb2lab(om_rgb)  # float64 by skimage
        rf_lab = rgb2lab(rf_rgb)

        # Robust medians on cores
        a_ref, b_ref = _median_ab(ref_bgr, ref_mask_core)
        a_om,  b_om  = _median_ab(on_model_bgr, on_model_mask_core)

        C_ref, h_ref = _lab_to_lch(np.array(a_ref), np.array(b_ref))
        C_om,  h_om  = _lab_to_lch(np.array(a_om),  np.array(b_om))

        C_scale = float((C_ref + 1e-6) / (C_om + 1e-6))
        h_shift = float(h_ref - h_om)
        ab_offset = (0.0, 0.0)

        # Precompute reference medians in Lab for feedback (once)
        rm = ref_mask_core.astype(bool)
        rf_Lm = float(np.median(rf_lab[..., 0][rm])) if rm.any() else 70.0
        rf_am = float(np.median(rf_lab[..., 1][rm])) if rm.any() else 0.0
        rf_bm = float(np.median(rf_lab[..., 2][rm])) if rm.any() else 0.0
        ref_1x1 = np.array([[[rf_Lm, rf_am, rf_bm]]], dtype=np.float64)

        def apply_map(mask_apply: np.ndarray, ab_off=(0.0, 0.0)) -> np.ndarray:
            out = om_lab.copy()
            a = out[..., 1]; b = out[..., 2]
            C, h = _lab_to_lch(a, b)
            mm = mask_apply.astype(bool)
            if mm.any():
                C2 = C.copy(); h2 = h.copy()
                C2[mm] = C[mm] * C_scale
                h2[mm] = h[mm] + h_shift
                a2, b2 = _lch_to_ab(C2, h2)
                a2[mm] = a2[mm] + ab_off[0]
                b2[mm] = b2[mm] + ab_off[1]
                out[..., 1] = a2; out[..., 2] = b2
            return out

        # First pass on the FULL garment mask
        out_lab = apply_map(on_model_mask_full, ab_offset)

        # Feedback: ΔE between medians (core vs ref core) computed directly in Lab
        core_m = on_model_mask_core.astype(bool)

        def dE_med_against_ref(out_lab_arr: np.ndarray) -> float:
            if not core_m.any():
                return float("nan")
            Lm = float(np.median(out_lab_arr[..., 0][core_m]))
            am = float(np.median(out_lab_arr[..., 1][core_m]))
            bm = float(np.median(out_lab_arr[..., 2][core_m]))
            tst_1x1 = np.array([[[Lm, am, bm]]], dtype=np.float64)
            return float(deltaE_ciede2000(ref_1x1, tst_1x1)[0, 0])

        dE = dE_med_against_ref(out_lab)
        for _ in range(2):
            if not np.isfinite(dE) or dE <= self.deltaE_target:
                break
            scale = self.deltaE_target / max(dE, 1e-3)
            ab_offset = (ab_offset[0] * scale, ab_offset[1] * scale)
            out_lab = apply_map(on_model_mask_full, ab_offset)
            dE = dE_med_against_ref(out_lab)

        # Final convert with gamut clamp to avoid warnings / artifacts
        _clip_lab_inplace(out_lab)
        out_rgb = np.clip(lab2rgb(out_lab), 0, 1)
        return (out_rgb * 255.0).astype(np.uint8)
