# src/metrics/spatial_coherence.py
"""
Spatial Coherence Index (SCI) - Measures local consistency of color correction.

Analyzes correction quality across spatial patches to detect:
- Stripe artifacts
- Shadow/highlight issues
- Cluster transition problems
- Regional correction failures
"""
from __future__ import annotations
import numpy as np
import cv2
from typing import Tuple, Dict, Optional
from skimage.color import rgb2lab, deltaE_ciede2000


def compute_spatial_coherence(
    corrected_bgr: np.ndarray,
    reference_lab: np.ndarray,
    reference_mask: np.ndarray,
    mask: np.ndarray,
    patch_size: int = 32,
    threshold_good: float = 3.0,
    threshold_poor: float = 5.0,
) -> Dict[str, float]:
    """
    Compute Spatial Coherence Index and patch-level statistics.
    
    Args:
        corrected_bgr: Corrected image (BGR)
        reference_lab: Reference image in LAB space
        reference_mask: Reference mask for computing target color
        mask: Garment mask for analysis
        patch_size: Size of square patches for analysis
        threshold_good: ΔE threshold for "good" patches
        threshold_poor: ΔE threshold for "poor" patches
        
    Returns:
        Dictionary with SCI metrics and patch statistics
    """
    # Convert corrected to LAB
    corr_rgb = cv2.cvtColor(corrected_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    corr_lab = rgb2lab(corr_rgb)
    
    # Get reference median color
    ref_mask = reference_mask.astype(bool)
    if not ref_mask.any():
        return {"sci": 0.0, "valid": False}
    
    ref_L = np.median(reference_lab[ref_mask, 0])
    ref_a = np.median(reference_lab[ref_mask, 1])
    ref_b = np.median(reference_lab[ref_mask, 2])
    
    # Compute ΔE map
    H, W = corrected_bgr.shape[:2]
    ref_1x1 = np.array([[[ref_L, ref_a, ref_b]]], dtype=np.float64)
    dE_map = deltaE_ciede2000(corr_lab, np.tile(ref_1x1, (H, W, 1)))
    
    # Extract patches
    m = mask.astype(bool)
    patch_means = []
    patch_stds = []
    patch_coords = []
    
    for y in range(0, H - patch_size, patch_size):
        for x in range(0, W - patch_size, patch_size):
            patch_mask = m[y:y+patch_size, x:x+patch_size]
            
            # Skip patches with insufficient mask coverage
            if patch_mask.sum() < (patch_size * patch_size * 0.5):
                continue
            
            patch_dE = dE_map[y:y+patch_size, x:x+patch_size][patch_mask]
            
            if len(patch_dE) > 0:
                patch_means.append(np.mean(patch_dE))
                patch_stds.append(np.std(patch_dE))
                patch_coords.append((x, y))
    
    if len(patch_means) == 0:
        return {"sci": 0.0, "valid": False}
    
    patch_means = np.array(patch_means)
    patch_stds = np.array(patch_stds)
    
    # Compute SCI (inverse of variance of patch means, normalized)
    global_variance = np.var(patch_means)
    sci = 1.0 / (1.0 + global_variance)  # Range: 0-1, higher is better
    
    # Compute patch quality stats
    good_patches = (patch_means <= threshold_good).sum()
    poor_patches = (patch_means > threshold_poor).sum()
    total_patches = len(patch_means)
    
    # Find worst patch
    worst_idx = np.argmax(patch_means)
    worst_patch_dE = patch_means[worst_idx]
    worst_patch_coord = patch_coords[worst_idx]
    
    # Spatial autocorrelation (simple: correlation of adjacent patches)
    autocorr = 0.0
    n_pairs = 0
    for i, (x, y) in enumerate(patch_coords):
        # Look for neighbors
        for j, (x2, y2) in enumerate(patch_coords):
            if i != j and abs(x - x2) <= patch_size and abs(y - y2) <= patch_size:
                autocorr += patch_means[i] * patch_means[j]
                n_pairs += 1
    
    if n_pairs > 0:
        autocorr /= n_pairs
        autocorr /= (np.mean(patch_means) ** 2 + 1e-6)  # Normalize
    
    return {
        "sci": float(sci),
        "valid": True,
        "global_variance": float(global_variance),
        "mean_patch_dE": float(np.mean(patch_means)),
        "std_patch_dE": float(np.std(patch_means)),
        "good_patches_pct": float(100 * good_patches / total_patches),
        "poor_patches_pct": float(100 * poor_patches / total_patches),
        "total_patches": int(total_patches),
        "worst_patch_dE": float(worst_patch_dE),
        "worst_patch_coord": worst_patch_coord,
        "autocorrelation": float(autocorr),
        # For visualization
        "patch_means": patch_means,
        "patch_coords": patch_coords,
        "patch_size": patch_size,
    }


def create_heatmap_visualization(
    corrected_bgr: np.ndarray,
    mask: np.ndarray,
    sci_results: Dict,
    output_path: Optional[str] = None,
) -> Optional[np.ndarray]:
    """
    Create a color-coded heatmap visualization of patch ΔE.
    
    Args:
        corrected_bgr: Corrected image
        mask: Garment mask
        sci_results: Results from compute_spatial_coherence
        output_path: If provided, save heatmap to this path
        
    Returns:
        Heatmap image (BGR) or None if no valid patches
    """
    if not sci_results.get("valid", False):
        return None
    
    patch_means = sci_results["patch_means"]
    patch_coords = sci_results["patch_coords"]
    patch_size = sci_results["patch_size"]
    
    # Create heatmap overlay
    H, W = corrected_bgr.shape[:2]
    heatmap = np.zeros((H, W, 3), dtype=np.uint8)
    
    # Color scale: green (good) → yellow → orange → red (poor)
    # ΔE: 0-2 (green), 2-5 (yellow), 5-10 (orange), 10+ (red)
    for dE, (x, y) in zip(patch_means, patch_coords):
        if dE <= 2.0:
            color = (0, 255, 0)  # Green
        elif dE <= 5.0:
            # Interpolate green → yellow
            t = (dE - 2.0) / 3.0
            color = (0, int(255 * (1 - t * 0.5)), int(255 * t))
        elif dE <= 10.0:
            # Interpolate yellow → red
            t = (dE - 5.0) / 5.0
            color = (0, int(255 * (1 - t)), 255)
        else:
            color = (0, 0, 255)  # Red
        
        cv2.rectangle(heatmap, (x, y), (x + patch_size, y + patch_size), color, -1)
    
    # Apply mask
    m = mask.astype(bool)
    heatmap[~m] = 0
    
    # Blend with original image (50/50)
    vis = cv2.addWeighted(corrected_bgr, 0.5, heatmap, 0.5, 0)
    
    # Mark worst patch with border
    worst_x, worst_y = sci_results["worst_patch_coord"]
    cv2.rectangle(vis, (worst_x, worst_y), (worst_x + patch_size, worst_y + patch_size), (255, 0, 255), 3)
    
    # Add legend text
    cv2.putText(vis, f"SCI: {sci_results['sci']:.3f}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(vis, f"Worst: dE={sci_results['worst_patch_dE']:.1f}", (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)
    
    if output_path:
        cv2.imwrite(output_path, vis)
    
    return vis


def create_ascii_heatmap(sci_results: Dict, width: int = 40) -> str:
    """
    Create ASCII art heatmap for console display.
    
    Args:
        sci_results: Results from compute_spatial_coherence
        width: Width of ASCII display in characters
        
    Returns:
        Multi-line string with ASCII heatmap
    """
    if not sci_results.get("valid", False):
        return "No valid patches"
    
    patch_means = sci_results["patch_means"]
    patch_coords = sci_results["patch_coords"]
    
    # Create grid
    xs = [x for x, y in patch_coords]
    ys = [y for x, y in patch_coords]
    
    if not xs or not ys:
        return "No patches"
    
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    
    # Normalize to ASCII grid
    grid_h = int(width * (y_max - y_min) / (x_max - x_min + 1)) + 1
    grid_w = width
    
    # Initialize grid with spaces
    grid = [[' ' for _ in range(grid_w)] for _ in range(grid_h)]
    
    # Map patches to grid
    for dE, (x, y) in zip(patch_means, patch_coords):
        gx = int((x - x_min) / (x_max - x_min + 1) * (grid_w - 1))
        gy = int((y - y_min) / (y_max - y_min + 1) * (grid_h - 1))
        
        # Choose character based on ΔE
        if dE <= 2.0:
            char = '█'  # Excellent
        elif dE <= 3.0:
            char = '▓'  # Good
        elif dE <= 5.0:
            char = '▒'  # Acceptable
        elif dE <= 10.0:
            char = '░'  # Poor
        else:
            char = '·'  # Very poor
        
        grid[gy][gx] = char
    
    # Convert to string
    lines = [''.join(row) for row in grid]
    
    # Add legend
    legend = "\n  Legend: █ Excellent (≤2) | ▓ Good (2-3) | ▒ OK (3-5) | ░ Poor (5-10) | · Very Poor (>10)"
    
    return '\n'.join(lines) + legend
