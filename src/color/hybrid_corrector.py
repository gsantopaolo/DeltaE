# src/color/hybrid_corrector.py
"""
Hybrid Color Corrector: Best of Classical + OT approaches.

Strategy:
1. Histogram matching (from OT) for distribution alignment
2. Global median shift for accurate color targeting
3. Iterative feedback for fine-tuning

This combines distribution-aware correction with precise color accuracy.
"""
from __future__ import annotations
import numpy as np
import cv2
from sklearn.cluster import KMeans
from skimage.color import rgb2lab, lab2rgb, deltaE_ciede2000

from ..utils.logging_utils import get_logger


def _clip_lab_inplace(Lab: np.ndarray) -> None:
    """Clamp Lab to safe gamut before RGB conversion."""
    Lab[..., 0] = np.clip(Lab[..., 0], 0.0, 100.0)
    Lab[..., 1] = np.clip(Lab[..., 1], -128.0, 127.0)
    Lab[..., 2] = np.clip(Lab[..., 2], -128.0, 127.0)


class HybridCorrector:
    """
    Hybrid color corrector combining histogram matching + global shift.
    
    Best of both worlds:
    - Distribution matching from OT approach
    - Precise median targeting from classical approach
    - Multi-cluster support for complex garments
    """

    def __init__(
        self,
        deltaE_target: float = 2.0,
        num_clusters: int = 3,
        use_clustering: bool = True,
        min_cluster_size: int = 500,
    ) -> None:
        self.deltaE_target = deltaE_target
        self.num_clusters = num_clusters
        self.use_clustering = use_clustering
        self.min_cluster_size = min_cluster_size
        self.logger = get_logger("Hybrid")

    def correct(
        self,
        on_model_bgr: np.ndarray,
        on_model_mask_core: np.ndarray,
        on_model_mask_full: np.ndarray,
        ref_bgr: np.ndarray,
        ref_mask_core: np.ndarray,
    ) -> np.ndarray:
        """Apply hybrid correction."""
        # Convert to Lab
        om_rgb = cv2.cvtColor(on_model_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        rf_rgb = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        om_lab = rgb2lab(om_rgb)
        rf_lab = rgb2lab(rf_rgb)

        # Extract masked pixels
        om_core_mask = on_model_mask_core.astype(bool)
        rf_core_mask = ref_mask_core.astype(bool)
        om_full_mask = on_model_mask_full.astype(bool)

        om_chroma = om_lab[om_core_mask, 1:3]
        rf_chroma = rf_lab[rf_core_mask, 1:3]

        if om_chroma.shape[0] < 100 or rf_chroma.shape[0] < 100:
            self.logger.warning("⚠️ Insufficient pixels, skipping correction")
            return on_model_bgr

        # Step 1: Histogram matching (distribution alignment)
        if self.use_clustering and om_chroma.shape[0] >= self.min_cluster_size * self.num_clusters:
            corrected_lab = self._histogram_match_clustered(
                om_lab, om_full_mask, om_chroma, rf_chroma
            )
        else:
            corrected_lab = self._histogram_match_simple(
                om_lab, om_full_mask, om_chroma, rf_chroma
            )

        # Step 2: Global median shift (precise targeting)
        corrected_lab = self._apply_global_shift(
            corrected_lab, om_full_mask, rf_lab, rf_core_mask
        )

        # Convert back
        _clip_lab_inplace(corrected_lab)
        out_rgb = np.clip(lab2rgb(corrected_lab), 0, 1)
        return cv2.cvtColor((out_rgb * 255.0).astype(np.uint8), cv2.COLOR_RGB2BGR)

    def _histogram_match_simple(
        self,
        om_lab: np.ndarray,
        om_mask: np.ndarray,
        om_chroma: np.ndarray,
        rf_chroma: np.ndarray,
    ) -> np.ndarray:
        """Simple histogram matching on entire garment."""
        self.logger.info(f"🎨 Hybrid: Histogram matching {om_chroma.shape[0]} pixels")
        
        out_lab = om_lab.copy()

        for ch_idx in range(2):  # a*, b*
            om_vals = om_chroma[:, ch_idx]
            rf_vals = rf_chroma[:, ch_idx]

            om_sorted = np.sort(om_vals)
            rf_sorted = np.sort(rf_vals)

            om_cdf = np.linspace(0, 1, len(om_sorted))
            rf_cdf = np.linspace(0, 1, len(rf_sorted))

            om_full_vals = out_lab[om_mask, ch_idx + 1]
            mapped_vals = np.interp(
                np.interp(om_full_vals, om_sorted, om_cdf),
                rf_cdf, rf_sorted
            )
            out_lab[om_mask, ch_idx + 1] = mapped_vals

        return out_lab

    def _histogram_match_clustered(
        self,
        om_lab: np.ndarray,
        om_mask: np.ndarray,
        om_chroma: np.ndarray,
        rf_chroma: np.ndarray,
    ) -> np.ndarray:
        """Multi-cluster histogram matching."""
        K = self.num_clusters
        self.logger.info(f"🎨 Hybrid: Multi-cluster ({K}) histogram matching")

        kmeans = KMeans(n_clusters=K, random_state=42, n_init=10)
        labels = kmeans.fit_predict(om_chroma)

        ys, xs = np.where(om_mask)
        n_chroma = om_chroma.shape[0]
        if len(ys) != n_chroma:
            ys, xs = ys[:n_chroma], xs[:n_chroma]
        
        pixel_coords = np.stack([xs, ys], axis=1)
        out_lab = om_lab.copy()

        for k in range(K):
            cluster_mask = (labels == k)
            n_cluster = cluster_mask.sum()

            if n_cluster < self.min_cluster_size:
                continue

            om_cluster_chroma = om_chroma[cluster_mask]
            cluster_pixels = pixel_coords[cluster_mask]
            
            for ch_idx in range(2):
                om_vals = om_cluster_chroma[:, ch_idx]
                rf_vals = rf_chroma[:, ch_idx]

                om_sorted = np.sort(om_vals)
                rf_sorted = np.sort(rf_vals)

                om_cdf = np.linspace(0, 1, len(om_sorted))
                rf_cdf = np.linspace(0, 1, len(rf_sorted))

                mapped_vals = np.interp(
                    np.interp(om_vals, om_sorted, om_cdf),
                    rf_cdf, rf_sorted
                )
                
                for i, (x, y) in enumerate(cluster_pixels):
                    out_lab[y, x, ch_idx + 1] = mapped_vals[i]

        return out_lab

    def _apply_global_shift(
        self,
        corrected_lab: np.ndarray,
        om_mask: np.ndarray,
        rf_lab: np.ndarray,
        rf_mask: np.ndarray,
    ) -> np.ndarray:
        """Apply global shift to hit median target precisely."""
        rf_m = rf_mask.astype(bool)
        om_m = om_mask.astype(bool)

        if not rf_m.any() or not om_m.any():
            return corrected_lab

        # Reference median
        rf_a = np.median(rf_lab[rf_m, 1])
        rf_b = np.median(rf_lab[rf_m, 2])

        # Current median after histogram matching
        cor_a = np.median(corrected_lab[om_m, 1])
        cor_b = np.median(corrected_lab[om_m, 2])

        # Compute shift needed
        shift_a = rf_a - cor_a
        shift_b = rf_b - cor_b

        # Apply global shift (preserves L channel)
        out = corrected_lab.copy()
        out[om_m, 1] += shift_a
        out[om_m, 2] += shift_b

        # Compute final ΔE
        ref_1x1 = np.array([[[np.median(rf_lab[rf_m, 0]), rf_a, rf_b]]], dtype=np.float64)
        cor_1x1 = np.array([[[np.median(out[om_m, 0]), np.median(out[om_m, 1]), np.median(out[om_m, 2])]]], dtype=np.float64)
        dE_med = float(deltaE_ciede2000(ref_1x1, cor_1x1)[0, 0])

        self.logger.info(f"  Global shift: Δa={shift_a:.2f}, Δb={shift_b:.2f} → ΔE={dE_med:.2f}")

        return out
