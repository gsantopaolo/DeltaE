from __future__ import annotations
import numpy as np, cv2
from typing import Optional
from ..schemas.config import OnModelMaskingConfig
from ..utils.logging_utils import get_logger
from .schp_human_parsing import SCHPParser
from .sam2_refine import SAM2Refiner
from .onmodel_color_prior import OnModelColorPrior
from .utils_post import open_close, largest_component, crf_refine

class OnModelMaskerPipeline:
    def __init__(self, cfg: OnModelMaskingConfig):
        self.cfg = cfg
        self.logger = get_logger("OnModelMasker")
        # Initialize submodules (graceful if unavailable)
        self.schp = SCHPParser(cfg.schp.weights_path, device=cfg.schp.device, include_labels=cfg.schp.include_labels) if cfg.schp.enabled else None
        self.sam2 = SAM2Refiner(cfg.sam2.checkpoint_path, device=cfg.schp.device) if cfg.sam2.enabled else None
        self.color_prior = OnModelColorPrior(cfg.color_prior)

    def _refine_optional(self, image_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
        if self.sam2 and self.sam2.is_ready():
            mask = self.sam2.refine(image_bgr, mask, expand_box_px=self.cfg.sam2.expand_box_px)
        if self.cfg.use_crf:
            prob = (mask.astype(np.float32) / 255.0)
            mask = crf_refine(image_bgr, prob)
        mask = open_close(mask, 5, 1, 2)
        return largest_component(mask)

    def get_mask_with_ref(self, on_model_bgr: np.ndarray, ref_bgr: np.ndarray, ref_mask_core: np.ndarray) -> np.ndarray:
        """Try SCHP → SAM2 refine; fallback to color-prior; final fallback heuristic."""
        h, w = on_model_bgr.shape[:2]
        for method in self.cfg.method_order:
            try:
                if method == "schp" and self.schp and self.schp.is_ready():
                    self.logger.info("🧩 SCHP → mask")
                    m = self.schp.get_mask(on_model_bgr)
                    if m.sum() > 0:
                        m = self._refine_optional(on_model_bgr, m)
                        if int(m.sum() // 255) >= self.cfg.color_prior.min_mask_pixels:
                            return m
                elif method == "color_prior":
                    self.logger.info("🎯 Color-prior → mask")
                    m = self.color_prior.get_mask_with_ref(on_model_bgr, ref_bgr, ref_mask_core)
                    m = self._refine_optional(on_model_bgr, m)
                    if int(m.sum() // 255) >= self.cfg.color_prior.min_mask_pixels:
                        return m
                elif method == "heuristic":
                    self.logger.info("🪄 Heuristic fallback → mask")
                    m = self.color_prior.get_mask(on_model_bgr)
                    m = self._refine_optional(on_model_bgr, m)
                    if int(m.sum() // 255) >= self.cfg.color_prior.min_mask_pixels:
                        return m
            except Exception as e:
                self.logger.warning(f"⚠️ {method} failed: {e}")
                continue
        # last resort: return whatever we got (or zeros)
        self.logger.warning("⚠️ All masking methods failed; returning empty mask.")
        return np.zeros((h, w), np.uint8)
