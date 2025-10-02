# src/metrics/triplet_analysis.py
"""
Triplet ΔE2000 Analysis - Quantitative proof of color correction effectiveness.

Compares three images:
1. Still-life (reference)
2. On-model (original, degraded)
3. Corrected-on-model (after correction)

Measures:
- ΔE (still → on-model): How much color shifted
- ΔE (still → corrected): How well correction recovered color
- Improvement: ΔE_before - ΔE_after
"""
from __future__ import annotations
import numpy as np
import cv2
from typing import Dict, Optional
from skimage.color import rgb2lab, deltaE_ciede2000


def compute_triplet_delta_e(
    still_life_bgr: np.ndarray,
    still_life_mask: np.ndarray,
    on_model_bgr: np.ndarray,
    on_model_mask: np.ndarray,
    corrected_bgr: np.ndarray,
) -> Dict[str, float]:
    """
    Compute ΔE2000 between triplet images.
    
    Args:
        still_life_bgr: Reference still-life image
        still_life_mask: Still-life mask (for median computation)
        on_model_bgr: Original on-model image (before correction)
        on_model_mask: On-model mask
        corrected_bgr: Corrected on-model image
        
    Returns:
        Dictionary with:
        - dE_still_vs_original: ΔE between reference and original
        - dE_still_vs_corrected: ΔE between reference and corrected
        - improvement: How much correction helped (positive = better)
        - improvement_pct: Improvement as percentage
    """
    # Convert to LAB
    still_rgb = cv2.cvtColor(still_life_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    om_rgb = cv2.cvtColor(on_model_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    corr_rgb = cv2.cvtColor(corrected_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    
    still_lab = rgb2lab(still_rgb)
    om_lab = rgb2lab(om_rgb)
    corr_lab = rgb2lab(corr_rgb)
    
    # Get reference median color from still-life
    st_mask = still_life_mask.astype(bool)
    if not st_mask.any():
        return {
            "dE_still_vs_original": 0.0,
            "dE_still_vs_corrected": 0.0,
            "improvement": 0.0,
            "improvement_pct": 0.0,
            "valid": False,
        }
    
    ref_L = np.median(still_lab[st_mask, 0])
    ref_a = np.median(still_lab[st_mask, 1])
    ref_b = np.median(still_lab[st_mask, 2])
    
    # Get median colors from on-model images
    om_mask_bool = on_model_mask.astype(bool)
    if not om_mask_bool.any():
        return {
            "dE_still_vs_original": 0.0,
            "dE_still_vs_corrected": 0.0,
            "improvement": 0.0,
            "improvement_pct": 0.0,
            "valid": False,
        }
    
    om_L = np.median(om_lab[om_mask_bool, 0])
    om_a = np.median(om_lab[om_mask_bool, 1])
    om_b = np.median(om_lab[om_mask_bool, 2])
    
    corr_L = np.median(corr_lab[om_mask_bool, 0])
    corr_a = np.median(corr_lab[om_mask_bool, 1])
    corr_b = np.median(corr_lab[om_mask_bool, 2])
    
    # Compute ΔE2000 between medians
    ref_color = np.array([[[ref_L, ref_a, ref_b]]], dtype=np.float64)
    om_color = np.array([[[om_L, om_a, om_b]]], dtype=np.float64)
    corr_color = np.array([[[corr_L, corr_a, corr_b]]], dtype=np.float64)
    
    dE_still_vs_original = float(deltaE_ciede2000(ref_color, om_color)[0, 0])
    dE_still_vs_corrected = float(deltaE_ciede2000(ref_color, corr_color)[0, 0])
    
    # Improvement metrics
    improvement = dE_still_vs_original - dE_still_vs_corrected
    improvement_pct = 100.0 * improvement / (dE_still_vs_original + 1e-6)
    
    return {
        "dE_still_vs_original": dE_still_vs_original,
        "dE_still_vs_corrected": dE_still_vs_corrected,
        "improvement": improvement,
        "improvement_pct": improvement_pct,
        "valid": True,
    }


def create_triplet_visualization(
    still_life_bgr: np.ndarray,
    on_model_bgr: np.ndarray,
    corrected_bgr: np.ndarray,
    output_path: Optional[str] = None,
) -> np.ndarray:
    """
    Create a triplet visualization image showing the three images side-by-side plus a difference map.
    
    Args:
        still_life_bgr: Reference still-life image
        on_model_bgr: Original on-model image (before correction)
        corrected_bgr: Corrected on-model image
        output_path: Optional path to save the visualization image
        
    Returns:
        Visualization image as a numpy array
    """
    # Convert to RGB
    still_rgb = cv2.cvtColor(still_life_bgr, cv2.COLOR_BGR2RGB)
    om_rgb = cv2.cvtColor(on_model_bgr, cv2.COLOR_BGR2RGB)
    corr_rgb = cv2.cvtColor(corrected_bgr, cv2.COLOR_BGR2RGB)
    
    # Create a difference map
    diff_map = np.abs(om_rgb - corr_rgb)
    diff_map = cv2.cvtColor(diff_map, cv2.COLOR_RGB2GRAY)
    diff_map = cv2.applyColorMap(diff_map, cv2.COLORMAP_JET)
    
    # Stack images horizontally
    visualization = np.hstack([still_rgb, om_rgb, corr_rgb, diff_map])
    
    if output_path:
        cv2.imwrite(output_path, visualization)
    
    return visualization


def format_triplet_table(results: list[dict], mode: str = "console") -> str:
    """
    Format triplet analysis results as a table.
    
    Args:
        results: List of dicts with keys: id, dE_still_vs_original, dE_still_vs_corrected, improvement
        mode: "console" for rich table or "markdown" for markdown table
        
    Returns:
        Formatted table string
    """
    if not results:
        return "No results to display"
    
    if mode == "markdown":
        # Markdown table for file saving
        lines = [
            "# Color Correction Triplet Analysis",
            "",
            "Quantitative proof of correction effectiveness using ΔE2000 metrics.",
            "",
            "| Image ID | ΔE (Still → Original) | ΔE (Still → Corrected) | Improvement (ΔE) | Improvement (%) | Status |",
            "|----------|------------------------|-------------------------|------------------|-----------------|--------|"
        ]
        
        for r in results:
            status = "✅ Better" if r["improvement"] > 0 else "❌ Worse" if r["improvement"] < 0 else "➖ Same"
            lines.append(
                f"| {r['id']} | {r['dE_still_vs_original']:.2f} | {r['dE_still_vs_corrected']:.2f} | "
                f"{r['improvement']:.2f} | {r['improvement_pct']:.1f}% | {status} |"
            )
        
        # Add summary stats
        avg_before = np.mean([r["dE_still_vs_original"] for r in results])
        avg_after = np.mean([r["dE_still_vs_corrected"] for r in results])
        avg_improvement = avg_before - avg_after
        avg_improvement_pct = 100.0 * avg_improvement / (avg_before + 1e-6)
        
        lines.extend([
            "",
            "## Summary",
            "",
            f"- **Average ΔE (before correction):** {avg_before:.2f}",
            f"- **Average ΔE (after correction):** {avg_after:.2f}",
            f"- **Average improvement:** {avg_improvement:.2f} ({avg_improvement_pct:.1f}%)",
            f"- **Total images:** {len(results)}",
            f"- **Images improved:** {sum(1 for r in results if r['improvement'] > 0)}",
            "",
            "---",
            "",
            "_Generated by DeltaE Color Correction Pipeline_"
        ])
        
        return "\n".join(lines)
    
    else:
        # Console table using tabulate with nice box drawing
        from tabulate import tabulate
        
        headers = ["Image ID", "ΔE Before", "ΔE After", "Improvement", "Improve %", "Status"]
        rows = []
        for r in results:
            status_icon = "✅" if r["improvement"] > 0 else "❌" if r["improvement"] < 0 else "➖"
            rows.append([
                r['id'],
                f"{r['dE_still_vs_original']:.2f}",
                f"{r['dE_still_vs_corrected']:.2f}",
                f"{r['improvement']:.2f}",
                f"{r['improvement_pct']:.1f}%",
                status_icon
            ])
        
        # Create main table with rounded grid
        table_str = tabulate(rows, headers=headers, tablefmt="rounded_grid")
        
        # Add summary stats
        avg_before = np.mean([r["dE_still_vs_original"] for r in results])
        avg_after = np.mean([r["dE_still_vs_corrected"] for r in results])
        avg_improvement = avg_before - avg_after
        avg_improvement_pct = 100.0 * avg_improvement / (avg_before + 1e-6)
        improved_count = sum(1 for r in results if r['improvement'] > 0)
        
        summary_lines = [
            "",
            "Summary Statistics:",
            f"  • Average ΔE (before): {avg_before:.2f}",
            f"  • Average ΔE (after):  {avg_after:.2f}",
            f"  • Average improvement: {avg_improvement:.2f} ({avg_improvement_pct:.1f}%)",
            f"  • Images improved:     {improved_count}/{len(results)}"
        ]
        
        return table_str + "\n" + "\n".join(summary_lines)


def create_triplet_visualization(
    still_life_bgr: np.ndarray,
    on_model_bgr: np.ndarray,
    corrected_bgr: np.ndarray,
    on_model_mask: np.ndarray,
    image_id: str,
    output_path: str,
) -> None:
    """
    Create a 4-panel visualization showing the triplet and difference maps.
    
    Layout:
    [Still-life Reference] [On-model Original] [Corrected] [ΔE Difference Maps]
    
    Args:
        still_life_bgr: Reference image
        on_model_bgr: Original on-model (before correction)
        corrected_bgr: Corrected on-model
        on_model_mask: Mask for on-model region
        image_id: Image identifier for labeling
        output_path: Where to save the visualization
    """
    # Convert to LAB for ΔE computation
    def bgr_to_lab(img):
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        return rgb2lab(rgb)
    
    om_lab = bgr_to_lab(on_model_bgr)
    corr_lab = bgr_to_lab(corrected_bgr)
    still_lab = bgr_to_lab(still_life_bgr)
    
    # Compute ΔE maps from reference
    dE_original = deltaE_ciede2000(still_lab, om_lab)  # Still vs Original (BEFORE)
    dE_corrected = deltaE_ciede2000(still_lab, corr_lab)  # Still vs Corrected (AFTER)
    
    # Apply mask
    mask_bool = on_model_mask.astype(bool)
    
    # Create side-by-side ΔE heatmap (before | after)
    H, W = on_model_bgr.shape[:2]
    combined_heatmap = np.zeros((H, W * 2, 3), dtype=np.uint8)  # Double width for two maps
    
    # Left half: ΔE original (before correction)
    for y in range(H):
        for x in range(W):
            if not mask_bool[y, x]:
                continue
            
            dE = dE_original[y, x]
            
            # Color scale: Green (ΔE≤2) → Yellow (2-5) → Orange (5-10) → Red (>10)
            if dE <= 2.0:
                color = (0, 255, 0)  # Green (good)
            elif dE <= 5.0:
                t = (dE - 2.0) / 3.0
                color = (0, int(255 * (1 - t * 0.5)), int(255 * (0.5 + t * 0.5)))  # Green→Yellow
            elif dE <= 10.0:
                t = (dE - 5.0) / 5.0
                color = (0, int(255 * (1 - t)), 255)  # Yellow→Red
            else:
                color = (0, 0, 255)  # Red (poor)
            
            combined_heatmap[y, x] = color
    
    # Right half: ΔE corrected (after correction)
    for y in range(H):
        for x in range(W):
            if not mask_bool[y, x]:
                continue
            
            dE = dE_corrected[y, x]
            
            if dE <= 2.0:
                color = (0, 255, 0)
            elif dE <= 5.0:
                t = (dE - 2.0) / 3.0
                color = (0, int(255 * (1 - t * 0.5)), int(255 * (0.5 + t * 0.5)))
            elif dE <= 10.0:
                t = (dE - 5.0) / 5.0
                color = (0, int(255 * (1 - t)), 255)
            else:
                color = (0, 0, 255)
            
            combined_heatmap[y, x + W] = color
    
    # Add divider line between the two heatmaps
    cv2.line(combined_heatmap, (W, 0), (W, H), (255, 255, 255), 2)
    
    # Resize all to same height
    target_h = 400
    target_w = int(W * target_h / H)
    
    still_resized = cv2.resize(still_life_bgr, (target_w, target_h))
    om_resized = cv2.resize(on_model_bgr, (target_w, target_h))
    corr_resized = cv2.resize(corrected_bgr, (target_w, target_h))
    heatmap_resized = cv2.resize(combined_heatmap, (target_w * 2, target_h))
    
    # Add labels to images
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    font_color = (255, 255, 255)
    font_thickness = 2
    bg_color = (0, 0, 0)
    
    # Helper to add text with background
    def add_label(img, text, pos):
        (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, font_thickness)
        cv2.rectangle(img, (pos[0]-5, pos[1]-text_h-5), (pos[0]+text_w+5, pos[1]+5), bg_color, -1)
        cv2.putText(img, text, pos, font, font_scale, font_color, font_thickness)
    
    add_label(still_resized, "Reference (Still)", (10, 30))
    add_label(om_resized, "Original (Before)", (10, 30))
    add_label(corr_resized, "Corrected (After)", (10, 30))
    
    # Add labels to heatmap
    add_label(heatmap_resized, "dE Before", (10, 30))
    add_label(heatmap_resized, "dE After", (target_w + 10, 30))
    
    # Add legend to heatmap
    legend_y = target_h - 80
    cv2.rectangle(heatmap_resized, (5, legend_y - 5), (150, target_h - 5), (0, 0, 0), -1)
    cv2.putText(heatmap_resized, "Green: dE<=2", (10, legend_y + 15), font, 0.4, (0, 255, 0), 1)
    cv2.putText(heatmap_resized, "Yellow: dE 2-5", (10, legend_y + 35), font, 0.4, (0, 255, 255), 1)
    cv2.putText(heatmap_resized, "Orange: dE 5-10", (10, legend_y + 55), font, 0.4, (0, 165, 255), 1)
    cv2.putText(heatmap_resized, "Red: dE>10", (10, legend_y + 75), font, 0.4, (0, 0, 255), 1)
    
    # Concatenate horizontally
    combined = np.hstack([still_resized, om_resized, corr_resized, heatmap_resized])
    
    # Add title bar
    title_bar = np.zeros((60, combined.shape[1], 3), dtype=np.uint8)
    cv2.putText(title_bar, f"Triplet Analysis: {image_id}", (20, 40), font, 1.0, (255, 255, 255), 2)
    
    final = np.vstack([title_bar, combined])
    
    # Save
    cv2.imwrite(output_path, final)
