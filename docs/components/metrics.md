# Metrics & Quality Control

Comprehensive quality measurement system for color correction validation.

---

## Overview

Five metrics measure different quality aspects:
- **ΔE2000**: Color accuracy (standard)
- **SSIM**: Texture preservation (standard)
- **Spill**: Edge quality (standard)
- **SCI**: Spatial coherence (bonus) ⭐
- **Triplet**: Before/after proof (bonus) ⭐

See [Evaluation Documentation](../evaluation.md) for detailed analysis.

---

## Standard Metrics

### 1. ΔE2000 (Color Accuracy)

**File**: `src/metrics/color_metrics.py`

**Functions**:
```python
deltaE_between_medians(ref, ref_mask, img, img_mask)
deltaE_q_to_ref_median(img, mask, ref, ref_mask, q=95)
```

**What it measures**: Perceptually-uniform color difference
**Target**: ΔE median ≤ 3.0, P95 ≤ 60.0
**Can capture**: Color accuracy, perceptual differences
**Cannot capture**: Spatial distribution, texture

### 2. SSIM (Texture Preservation)

**File**: `src/metrics/texture_metrics.py`

**Function**:
```python
ssim_L(img1, img2, mask)  # L-channel only
```

**What it measures**: Structural similarity (texture)
**Target**: SSIM ≥ 0.90
**Can capture**: Texture quality, material appearance
**Cannot capture**: Color accuracy (by design)

### 3. Spill Detection

**Function**: Computed in orchestrator
**What it measures**: Color leakage outside mask
**Target**: Spill ≤ 0.5
**Can capture**: Edge quality, mask precision
**Cannot capture**: Internal errors

---

## Bonus Metrics

### 4. Spatial Coherence Index (SCI)

**File**: `src/metrics/spatial_coherence.py`

**Key Functions**:
```python
compute_spatial_coherence(corrected, ref_lab, ref_mask, mask, patch_size=32)
create_heatmap_visualization(corrected, mask, sci_results, output_path)
```

**Algorithm**:
1. Divide image into 32×32 patches
2. Compute per-patch mean ΔE
3. Calculate variance (low = uniform correction)
4. Identify worst patches
5. Generate heatmap visualization

**Metrics Returned**:
- `sci`: Inverse variance (higher = more uniform)
- `good_patches_pct`: % patches with ΔE ≤ 5.0
- `poor_patches_pct`: % patches with ΔE > 10.0
- `worst_patch_coord`: Location of worst patch
- `spatial_autocorrelation`: Neighboring patch similarity

**Visualization**: Color-coded heatmap saved as `*-hm.jpg`
- 🟢 Green: ΔE ≤ 2.0
- 🟡 Yellow: ΔE 2.0-5.0
- 🟠 Orange: ΔE 5.0-10.0
- 🔴 Red: ΔE > 10.0

**Can capture**: WHERE failures occur, spatial consistency, regional performance
**Cannot capture**: WHY failures occur, natural variance

### 5. Triplet ΔE2000 Analysis

**File**: `src/metrics/triplet_analysis.py`

**Key Functions**:
```python
compute_triplet_delta_e(still, still_mask, original, om_mask, corrected)
create_triplet_visualization(still, original, corrected, mask, id, path)
format_triplet_table(results, mode="console")  # or "markdown"
```

**Algorithm**:
1. Compute median ΔE: still → original (before)
2. Compute median ΔE: still → corrected (after)
3. Calculate improvement: before - after
4. Generate 4-panel visualization
5. Create summary table

**Outputs**:
- Console table (rounded_grid format with tabulate)
- Markdown table (saved as `triplet_analysis_YYYYMMDD_HHMMSS.md`)
- 4-panel images (`*-triplet.jpg`): Reference | Original | Corrected | ΔE Maps

**Can capture**: Quantitative improvement proof, before/after comparison
**Cannot capture**: Spatial detail, texture quality

---

## Quality Control

**File**: `src/qc/rules.py`

```python
def evaluate(dE_med, dE_p95, ssim, spill, 
             threshold_dE_med, threshold_dE_p95, 
             threshold_ssim, threshold_spill):
    """
    Returns: QCResult(passed=True/False, reasons=[...])
    """
    passed = (
        dE_med <= threshold_dE_med and
        dE_p95 <= threshold_dE_p95 and
        ssim >= threshold_ssim and
        spill <= threshold_spill
    )
    
    return QCResult(passed=passed, reasons=failures)
```

**Default Thresholds** (configurable in YAML):
- ΔE median ≤ 3.0
- ΔE P95 ≤ 60.0
- SSIM ≥ 0.90
- Spill ≤ 0.5

---

## Configuration

```yaml
qc:
  # Standard thresholds
  max_deltaE_median: 3.0
  max_deltaE_p95: 60.0
  min_ssim_L: 0.90
  max_spill_deltaE: 0.5
  
  # SCI configuration
  enable_sci: true
  sci_patch_size: 32
  sci_save_heatmap: true
  
  # Triplet analysis
  enable_triplet_analysis: true
  save_triplet_table: true
  save_triplet_visualization: true
```

---

## Usage Example

```python
from src.metrics.color_metrics import deltaE_between_medians
from src.metrics.texture_metrics import ssim_L
from src.metrics.spatial_coherence import compute_spatial_coherence
from src.metrics.triplet_analysis import compute_triplet_delta_e

# Color accuracy
dE_med = deltaE_between_medians(ref_lab, ref_mask, corr_lab, corr_mask)

# Texture preservation
ssim_val = ssim_L(original_bgr, corrected_bgr, mask)

# Spatial coherence
sci_results = compute_spatial_coherence(corrected, ref_lab, ref_mask, mask)

# Triplet analysis
triplet = compute_triplet_delta_e(still, still_mask, original, om_mask, corrected)
```

---

## References

- [Evaluation Results](../evaluation.md) - Complete metrics analysis
- [Architecture](../architecture.md) - System overview
- [Methodology](../methodology.md) - Approach justification
