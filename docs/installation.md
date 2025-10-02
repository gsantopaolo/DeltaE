# Installation Guide

Complete setup instructions for DeltaE color correction pipeline.

---

## Table of Contents

- [Prerequisites](#prerequisites)
- [Installation Steps](#installation-steps)
- [Download Sample Data](#download-sample-data)
- [Running the Pipeline](#running-the-pipeline)
- [Configuration Reference](#configuration-reference)
- [Troubleshooting](#troubleshooting)

---

## Prerequisites

### System Requirements
- **Python**: 3.10 or higher
- **RAM**: 8GB minimum (16GB recommended)
- **Storage**: 5GB for dependencies + models
- **GPU**: Optional (MPS for Mac M2/M3, CUDA for Linux/Windows)

### Supported Platforms
- ✅ **macOS** (M2/M3 with MPS acceleration)
- ✅ **Linux** (CUDA 11.8+ recommended)
- ✅ **Windows** (CPU or CUDA)

---

## Installation Steps

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/DeltaE.git
cd DeltaE
```

### 2. Create Conda Environment

```bash
# Create environment
conda create -n deltae python=3.10
conda activate deltae
```

### 3. Install Dependencies

```bash
# Install all required packages
pip install -r requirements.txt
```



### 4. Verify Installation

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'MPS available: {torch.backends.mps.is_available()}')"
```

Expected output (Mac M2/M3):
```
PyTorch: 2.8.0
MPS available: True
```

---

## Download Sample Data

### Sample Images and Weights

**📦 Download Link:** [Google Drive - DeltaE Sample Data](https://drive.google.com/drive/folders/YOUR_LINK_HERE)

The package includes:
- `dataset/` - 10-20 sample image pairs (still-life + on-model)
- `weights/` - Pre-trained model checkpoints

### Manual Setup

After downloading, extract to your project directory:

```bash
# Your structure should look like:
DeltaE/
├── data/
│   └── dataset/
│       ├── 00000_still-life.jpg
│       ├── 00000_on-model.jpg
│       ├── 00001_still-life.jpg
│       ├── 00001_on-model.jpg
│       └── ...
└── weights/
    ├── sam2_hiera_base_plus.pt
    └── (Segformer downloads automatically from HuggingFace)
```

### Alternative: Download Weights Programmatically


---

## Running the Pipeline

### Quick Start

```bash
# Process all images in dataset
python -m src.main --config configs/default.yaml

# Process limited number (for testing)
python -m src.main --config configs/default.yaml --limit 5
```

### Output Files

Results are saved to `data/outputs/`:

```bash
data/outputs/
├── corrected-on-model-00000.jpg       # Final corrected image
├── corrected-on-model-00000-hm.jpg    # SCI spatial heatmap
├── corrected-on-model-00000-triplet.jpg  # 4-panel comparison
├── ...
└── triplet_analysis_20251002_165644.md  # Summary report
```

### Command Line Options

```bash
python -m src.main [OPTIONS]

Options:
  --config PATH      Path to YAML config file (default: configs/default.yaml)
  --limit N          Process only first N images (for testing)
  --help             Show help message
```

---

## Configuration Reference

### Complete Parameter Table

Edit `configs/default.yaml` to customize the pipeline:

| Section | Parameter | Type | Default | Description |
|---------|-----------|------|---------|-------------|
| **run** ||||
|| `input_dir` | str | `data/dataset` | Directory containing input image pairs |
|| `output_dir` | str | `data/outputs` | Where to save corrected images |
|| `masks_dir` | str | `data/masks` | Where to save generated masks |
|| `logs_dir` | str | `data/logs` | Log file destination |
|| `write_corrected` | bool | `true` | Whether to generate corrected images |
| **color** ||||
|| `mode` | str | `hybrid` | Color correction algorithm: `classical`, `ot`, or `hybrid` |
|| `deltaE_target` | float | `3.0` | Target median ΔE for QC pass |
|| `max_iter` | int | `3` | Maximum feedback iterations |
|| `feedback_strength` | float | `0.7` | Correction strength per iteration (0-1) |
| **color** (Hybrid mode) ||||
|| `hybrid_num_clusters` | int | `3` | Number of K-means clusters for patterns |
|| `hybrid_global_shift` | bool | `true` | Apply global median shift after histogram |
| **color** (OT mode) ||||
|| `ot_num_clusters` | int | `3` | Number of clusters for multi-color garments |
|| `ot_reg` | float | `0.1` | Entropic regularization (0.01-0.5, higher=stable) |
|| `ot_use_clustering` | bool | `true` | Enable multi-cluster vs single OT |
|| `ot_min_cluster_size` | int | `500` | Minimum pixels per cluster |
|| `ot_max_samples` | int | `5000` | Max pixels for OT (prevents OOM) |
| **masking** ||||
|| `erosion_px` | int | `5` | Mask erosion to avoid edges (pixels) |
|| `feather_px` | int | `15` | Feather radius for smooth blending |
| **masking.onmodel_color_prior** ||||
|| `ref_dilate_iters` | int | `10` | Reference mask dilation for color sampling |
|| `saturation_boost` | float | `1.2` | Boost saturation for better detection |
|| `morph_open_ksize` | int | `5` | Morphological opening kernel size |
|| `morph_close_ksize` | int | `7` | Morphological closing kernel size |
|| `min_mask_pixels` | int | `2000` | Minimum mask size to be valid |
| **qc** ||||
|| `max_deltaE_median` | float | `3.0` | Maximum median ΔE for PASS |
|| `max_deltaE_p95` | float | `60.0` | Maximum 95th percentile ΔE for PASS |
|| `min_ssim_L` | float | `0.90` | Minimum SSIM (L-channel) for PASS |
|| `max_spill_deltaE` | float | `0.5` | Maximum color spill outside mask |
| **qc** (SCI - Spatial Coherence) ||||
|| `enable_sci` | bool | `true` | Compute spatial coherence metrics |
|| `sci_patch_size` | int | `32` | Patch size for spatial analysis (16/32/64) |
|| `sci_save_heatmap` | bool | `true` | Save heatmap visualization images |
| **qc** (Triplet Analysis) ||||
|| `enable_triplet_analysis` | bool | `true` | Compare still vs original vs corrected |
|| `save_triplet_table` | bool | `true` | Save markdown summary table |
|| `save_triplet_visualization` | bool | `true` | Save 4-panel comparison images |

### Example Configurations

#### Fast Mode (Classical, No Visualizations)

```yaml
color:
  mode: "classical"
  max_iter: 1

qc:
  enable_sci: false
  sci_save_heatmap: false
  enable_triplet_analysis: false
```

#### Maximum Quality (Hybrid + All Metrics)

```yaml
color:
  mode: "hybrid"
  max_iter: 5
  feedback_strength: 0.8

qc:
  enable_sci: true
  sci_save_heatmap: true
  enable_triplet_analysis: true
  save_triplet_visualization: true
```

#### Debug Mode (OT with Aggressive Settings)

```yaml
color:
  mode: "ot"
  ot_num_clusters: 5
  ot_max_samples: 10000
  max_iter: 10
```

---

### Performance Optimization

**For faster processing:**
1. Use `mode: "classical"` (10x faster than hybrid)
2. Disable visualizations: `sci_save_heatmap: false`
3. Reduce `max_iter` to 1-2
4. Use GPU (MPS/CUDA) instead of CPU

**For better quality:**
1. Use `mode: "hybrid"` (best results)
2. Increase `max_iter` to 5-10
3. Enable all metrics for validation
4. Increase `feedback_strength` to 0.8-0.9

---

## Next Steps

After installation:
1. ✅ **Test run**: `python -m src.main --config configs/default.yaml --limit 2`
2. 📖 **Read architecture**: [docs/architecture.md](architecture.md)
3. 🎨 **Understand algorithms**: [docs/components/color.md](components/color.md)
4. 📊 **Review metrics**: [docs/evaluation.md](evaluation.md)

---

## Additional Resources

- **Architecture Overview**: [docs/architecture.md](architecture.md)
- **Methodology**: [docs/methodology.md](methodology.md)
- **Component Details**: [docs/components/](components/)
- **Main README**: [../README.md](../README.md)

---

