# src/masking/onmodel_pipeline.py
from __future__ import annotations
from typing import Optional, List
import numpy as np
import os

from ..utils.logging_utils import get_logger
from ..schemas.config import OnModelMaskingConfig

# Color-prior coarse masker
from .onmodel_schp_sam2 import OnModelColorPriorMasker

# Optional Segformer human parser (safe import)
try:
    from .segformer_human_parsing import SegformerHumanParser
    _HAS_SEGFORMER = True
except Exception:
    SegformerHumanParser = None  # type: ignore
    _HAS_SEGFORMER = False

# SAM v1 refiner (safe import)
try:
    from .sam_v1_refiner import SAMV1Refiner
    _HAS_SAMV1 = True
except Exception:
    SAMV1Refiner = None  # type: ignore
    _HAS_SAMV1 = False


class OnModelMaskerPipeline:
    """
    On-model garment masking:
      method_order: ["segformer", "sam_v1", "color_prior", "heuristic"]

    Flow:
      - Try Segformer (CLI) if available & weights exist.
      - Else seed with color-prior.
      - Refine with SAM v1 if enabled & available.
      - Heuristic == color-prior as last resort.
    """

    def __init__(self, cfg: OnModelMaskingConfig):
        self.cfg = cfg
        self.logger = get_logger("onmodel_pipeline")

        # Normalize order (accept both 'schp' and 'segformer' for compatibility)
        allowed = {"schp", "segformer", "sam_v1", "color_prior", "heuristic"}
        self.method_order: List[str] = []
        for m in cfg.method_order:
            if m == "schp":
                # Backward compatibility: map 'schp' → 'segformer'
                self.method_order.append("segformer")
            elif m in allowed:
                self.method_order.append(m)
        
        if not self.method_order:
            self.method_order = ["color_prior", "heuristic"]
        self.logger.info(f"🪄 On-model masking order: {self.method_order}")

        # --- Segformer (HuggingFace Transformers) ---
        self.segformer = None
        if "segformer" in self.method_order and getattr(cfg.schp, "enabled", False) and _HAS_SEGFORMER:
            try:
                # Use HuggingFace model (no weights file needed - downloads from hub)
                model_name = getattr(cfg.schp, "model_name", "mattmdjaga/segformer_b2_clothes")
                self.segformer = SegformerHumanParser(
                    model_name=model_name,
                    include_labels=cfg.schp.include_labels,
                    device=cfg.schp.device,
                )
                if not self.segformer.is_ready():
                    self.logger.info("⏭️ Segformer not ready; skipping Segformer.")
                    self.segformer = None
                else:
                    self.logger.info("👚 Segformer parser ready.")
            except Exception as e:
                self.logger.info(f"⏭️ Segformer init failed ({e}); skipping Segformer.")
                self.segformer = None
        elif "segformer" in self.method_order:
            self.logger.info("⏭️ Segformer module not available; skipping Segformer.")

        # --- Color prior ---
        self.color_prior = OnModelColorPriorMasker(cfg.color_prior)

        # --- SAM v1 ---
        self.samv1 = None
        if (
            "sam_v1" in self.method_order
            and getattr(cfg, "sam_v1", None)
            and getattr(cfg.sam_v1, "enabled", False)
            and _HAS_SAMV1
        ):
            try:
                self.samv1 = SAMV1Refiner(
                    checkpoint_path=cfg.sam_v1.checkpoint_path,
                    model_type=cfg.sam_v1.model_type,
                )
                if not self.samv1.is_ready():
                    self.logger.info("⏭️ SAM v1 not ready; skipping SAM refinement.")
                    self.samv1 = None
                else:
                    self.logger.info("✨ SAM v1 refiner ready.")
            except Exception as e:
                self.logger.info(f"⏭️ SAM v1 init failed ({e}); skipping refinement.")
                self.samv1 = None
        elif "sam_v1" in self.method_order:
            self.logger.info("⏭️ SAM v1 module not available; skipping SAM refinement.")

        self.expand_box_px: int = getattr(getattr(cfg, "sam_v1", object()), "expand_box_px", 16)

    @staticmethod
    def _count_px(mask: np.ndarray) -> int:
        return int(mask.sum() // 255)

    def _refine_if_possible(self, image_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
        if self.samv1 is None:
            return mask
        refined = self.samv1.refine(image_bgr, mask, expand_box_px=self.expand_box_px)
        if refined is None or refined.dtype != np.uint8:
            return mask
        return refined

    def get_mask_with_ref(
        self,
        on_model_bgr: np.ndarray,
        still_life_bgr: np.ndarray,
        still_core_mask: np.ndarray,
        logger=None,
    ) -> np.ndarray:
        log = logger or self.logger
        current: Optional[np.ndarray] = None

        for step in self.method_order:
            if step == "segformer":
                if self.segformer is None:
                    log.info("⏭️ Segformer not available → skipping")
                else:
                    try:
                        m = self.segformer.get_mask(on_model_bgr)
                        px = self._count_px(m)
                        log.info(f"👚 Segformer → mask (px={px})")
                        if px >= self.cfg.color_prior.min_mask_pixels:
                            current = m
                        else:
                            log.info("ℹ️ Segformer mask too small → try next stage")
                    except Exception as e:
                        log.info(f"⏭️ Segformer failed ({e}) → try next stage")

            elif step == "color_prior":
                # Only use color-prior as fallback if we don't have a semantic mask
                if current is not None:
                    log.info("ℹ️ Already have semantic mask, skipping color-prior")
                    continue
                    
                try:
                    seed = self.color_prior.get_mask_with_ref(
                        on_model_bgr, still_life_bgr, still_core_mask, logger=log
                    )
                    px = self._count_px(seed)
                    log.info(f"🎯 Color-prior → seed mask (px={px})")
                    current = seed
                except Exception as e:
                    log.info(f"⏭️ Color-prior failed ({e}) → try next stage")

            elif step == "sam_v1":
                if current is None:
                    # auto-seed from color-prior
                    try:
                        seed = self.color_prior.get_mask_with_ref(
                            on_model_bgr, still_life_bgr, still_core_mask, logger=log
                        )
                        px = self._count_px(seed)
                        log.info(f"🎯 Color-prior (auto-seed) → px={px}")
                        current = seed
                    except Exception as e:
                        log.info(f"⏭️ Could not auto-seed for SAM v1 ({e}) → skipping refinement")
                        continue

                if self.samv1 is None:
                    log.info("⏭️ SAM v1 not available → skipping refinement")
                else:
                    try:
                        refined = self._refine_if_possible(on_model_bgr, current)
                        pxr = self._count_px(refined)
                        log.info(f"✨ SAM v1 refine → mask (px={pxr})")
                        current = refined
                    except Exception as e:
                        log.info(f"⏭️ SAM v1 refine failed ({e}) → keep current mask")

            elif step == "heuristic":
                if current is not None:
                    continue
                try:
                    seed = self.color_prior.get_mask_with_ref(
                        on_model_bgr, still_life_bgr, still_core_mask, logger=log
                    )
                    px = self._count_px(seed)
                    log.info(f"🪄 Heuristic (color-prior) → mask (px={px})")
                    current = seed
                except Exception as e:
                    log.info(f"⏭️ Heuristic/color-prior failed ({e})")

        if current is None:
            raise RuntimeError("OnModelMaskerPipeline: no mask produced by any stage.")

        return (current > 0).astype(np.uint8) * 255
