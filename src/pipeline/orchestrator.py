# src/pipeline/orchestrator.py
from __future__ import annotations

from pathlib import Path
from typing import Iterable
import numpy as np
import cv2

from ..schemas.config import AppConfig
from ..utils.logging_utils import get_logger
from ..pipeline.io import Pair
from ..masking.base import Masker
from ..masking.stilllife_rembg_sam2 import StillLifeRembgMasker
from ..color.classical_lab import ClassicalLabCorrector
from ..metrics.color_metrics import deltaE_between_medians, deltaE_q_to_ref_median
from ..metrics.texture_metrics import ssim_L
from ..qc.rules import evaluate

# Prefer the new pipeline (SCHP→SAM2→color-prior→heuristic). If missing, we’ll fallback.
try:
    from ..masking.onmodel_pipeline import OnModelMaskerPipeline
    _HAS_PIPELINE = True
except Exception:
    OnModelMaskerPipeline = None  # type: ignore
    _HAS_PIPELINE = False


def _feather_mask(bin_mask: np.ndarray, px: int) -> np.ndarray:
    """Return float32 alpha in [0,1], feathered but strictly confined to the original mask."""
    px = max(1, int(px))
    blurred = cv2.GaussianBlur(bin_mask, (0, 0), px).astype(np.float32) / 255.0
    # 🔒 hard gate: no alpha outside the binary mask
    inside = (bin_mask > 0).astype(np.float32)
    return blurred * inside



def _alpha_blend(src: np.ndarray, dst: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """Blend dst into src using scalar alpha (H×W, 0..1)."""
    out = src.astype(np.float32)
    a3 = np.dstack([alpha, alpha, alpha])
    out = out * (1.0 - a3) + dst.astype(np.float32) * a3
    return np.clip(out, 0, 255).astype(np.uint8)


def _make_on_model_masker(cfg: AppConfig, logger):
    """
    Build the on-model masker.
    Priority:
      1) OnModelMaskerPipeline (SCHP→SAM2→color-prior→heuristic) if cfg.masking.on_model exists
      2) Legacy OnModelColorPriorMasker (color-prior only) using cfg.masking.onmodel_color_prior
    """
    if _HAS_PIPELINE and hasattr(cfg.masking, "on_model"):
        logger.info("🧠 Using OnModelMaskerPipeline (SCHP→SAM2→color-prior→heuristic)")
        return OnModelMaskerPipeline(cfg.masking.on_model)  # type: ignore

    # Fallback: legacy color-prior masker
    from ..masking.onmodel_schp_sam2 import OnModelColorPriorMasker
    logger.info("🎯 Using legacy OnModelColorPriorMasker (color-prior fallback)")
    return OnModelColorPriorMasker(cfg.masking.onmodel_color_prior)  # type: ignore[attr-defined]

# Back-compat alias to avoid NameError if caller uses the other name
_make_on_model_mask = _make_on_model_masker


def _get_on_model_mask(masker: object,
                       om_bgr: np.ndarray,
                       st_bgr: np.ndarray,
                       st_core: np.ndarray) -> np.ndarray:
    """
    Call the appropriate API depending on masker type.
    - Pipeline exposes: get_mask_with_ref(om, st, st_core)
    - Legacy exposes:   get_mask_with_ref(om, st, st_core, logger=...)
    """
    if hasattr(masker, "get_mask_with_ref"):
        try:
            return masker.get_mask_with_ref(om_bgr, st_bgr, st_core)  # type: ignore[attr-defined]
        except TypeError:
            return masker.get_mask_with_ref(om_bgr, st_bgr, st_core, logger=None)  # type: ignore[attr-defined]
    if hasattr(masker, "get_mask"):
        return masker.get_mask(om_bgr)  # type: ignore[attr-defined]
    raise RuntimeError("On-model masker object has no callable mask method.")


def run_pairs(
    pairs: Iterable[Pair],
    masks_dir: Path,
    output_dir: Path,
    logs_dir: Path,
    cfg: AppConfig,
) -> None:
    """
    Process each (still-life, on-model) pair and write results.
    - Masks:
        • Still-life: rembg → largest component (+ cleanup in the masker)
        • On-model: SCHP→SAM2→color-prior→heuristic (pipeline) or legacy color-prior
    - Color:
        • Classical LCh mapping (hue rotation + chroma scale + ΔE feedback), preserves L
        • Blended back with feathered alpha (no spill)
    - Metrics (if writing corrected outputs):
        • Region-wise ΔE (median to median; 95th vs ref median), SSIM(L), spill ΔE in outer ring
    """
    logger = get_logger("orchestrator", logs_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    # Optional mask-only mode (safe even if RunConfig doesn’t define it)
    write_corrected: bool = getattr(getattr(cfg, "run", object()), "write_corrected", True)

    # Build modules
    on_masker = _make_on_model_masker(cfg, logger)
    st_masker = StillLifeRembgMasker()
    corrector = ClassicalLabCorrector(deltaE_target=cfg.color.deltaE_target)

    for i, p in enumerate(pairs, 1):
        try:
            logger.info(f"🧩 [{i}] ID={p.id} | Loading images")
            om = cv2.imread(str(p.on_model), cv2.IMREAD_COLOR)
            st = cv2.imread(str(p.still_life), cv2.IMREAD_COLOR)
            if om is None or st is None:
                raise RuntimeError("Failed to read one or both images.")

            logger.info("🧥 Getting masks (on-model, still-life)…")
            # Still-life mask + core
            st_mask = st_masker.get_mask(st)
            st_core = Masker.erode(st_mask, cfg.masking.erosion_px)

            # On-model mask via pipeline or legacy color-prior
            om_mask = _get_on_model_mask(on_masker, om, st, st_core)
            om_core = Masker.erode(om_mask, cfg.masking.erosion_px)

            # Feather for pretty edges
            alpha = _feather_mask(om_mask, cfg.masking.feather_px)

            # Guard: tiny/empty on-model mask → save masks and skip
            min_px = getattr(getattr(cfg.masking, "on_model", getattr(cfg.masking, "onmodel_color_prior", object())),
                             "min_mask_pixels",
                             2000)
            if int(om_core.sum() // 255) < int(min_px):
                logger.warning(f"⚠️ [{i}] ID={p.id} on-model garment mask too small; skipping.")
                cv2.imwrite(str(masks_dir / f"on-model-{p.id}.png"), om_mask)
                cv2.imwrite(str(masks_dir / f"still-life-{p.id}.png"), st_mask)
                continue

            # Always save masks for audit
            cv2.imwrite(str(masks_dir / f"on-model-{p.id}.png"), om_mask)
            cv2.imwrite(str(masks_dir / f"still-life-{p.id}.png"), st_mask)

            # If mask-only mode, skip color stage & metrics
            if not write_corrected:
                logger.info("🛑 Mask-only mode: skipping color correction & metrics.")
                continue

            logger.info("🎛️ Color correcting (classical LCh + ΔE feedback)…")
            corrected_inside = corrector.correct(
                on_model_bgr=om,
                on_model_mask_core=om_core,
                on_model_mask_full=om_mask,
                ref_bgr=st,
                ref_mask_core=st_core,
            )
            corrected = _alpha_blend(om, corrected_inside, alpha)

            # --- Region-wise metrics (no pixel alignment assumptions) ---
            dE_med = deltaE_between_medians(st, st_core, corrected, om_core)
            dE_p95 = deltaE_q_to_ref_median(corrected, om_core, st, st_core, q=95.0)
            ssim_val = ssim_L(corrected, om, om_core)

            # Spill: ΔE(corrected, original) in a ring just outside the mask
            ring = cv2.dilate(om_mask, np.ones((5, 5), np.uint8), iterations=1)
            ring = cv2.subtract(ring, om_mask)
            if ring.sum() == 0:
                spill_med = 0.0
            else:
                from skimage.color import rgb2lab, deltaE_ciede2000

                def _lab(x: np.ndarray) -> np.ndarray:
                    rgb = cv2.cvtColor(x, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
                    return rgb2lab(rgb)

                dE = deltaE_ciede2000(_lab(corrected), _lab(om))
                vals = dE[ring.astype(bool)]
                spill_med = float(np.median(vals)) if vals.size else 0.0

            qc = evaluate(
                dE_med if np.isfinite(dE_med) else 999.0,
                dE_p95 if np.isfinite(dE_p95) else 999.0,
                ssim_val if np.isfinite(ssim_val) else 0.0,
                spill_med,
                cfg.qc.max_deltaE_median,
                cfg.qc.max_deltaE_p95,
                cfg.qc.min_ssim_L,
                cfg.qc.max_spill_deltaE,
            )

            logger.info(
                f"📊 QC | ΔE_med={dE_med:.2f} ΔE_p95={dE_p95:.2f} "
                f"SSIM_L={ssim_val:.3f} spill={spill_med:.2f} → "
                f"{'✅ PASS' if qc.passed else '❌ FAIL'}"
            )

            # Write final corrected image
            cv2.imwrite(str(output_dir / f"corrected-on-model-{p.id}.jpg"), corrected)

        except Exception as e:
            logger.error(f"❌ [{i}] ID={p.id} failed: {e}")
