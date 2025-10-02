# System Architecture

Comprehensive architectural overview of the DeltaE color correction pipeline.

---

## Table of Contents

- [High-Level Overview](#high-level-overview)
- [System Components](#system-components)
- [Data Flow](#data-flow)
- [Processing Pipeline](#processing-pipeline)
- [Component Details](#component-details)

---

## High-Level Overview

DeltaE implements a modular pipeline architecture with three main stages:

```mermaid
graph LR
    A[Input Images] --> B[Masking]
    B --> C[Color Correction]
    C --> D[Quality Control]
    D --> E[Output + Metrics]
    
    style B fill:#e1f5ff
    style C fill:#fff4e1
    style D fill:#e8f5e9
```

### Design Principles

1. **Modularity**: Independent, swappable components
2. **Configurability**: YAML-driven parameters
3. **Observability**: Comprehensive logging and metrics
4. **Robustness**: Fallback strategies and error handling

---

## System Components

### Component Hierarchy

```mermaid
graph TD
    Main[main.py<br/>CLI Entry Point]
    Main --> Orch[Orchestrator<br/>Pipeline Controller]
    
    Orch --> Mask[Masking<br/>Components]
    Orch --> Color[Color<br/>Correction]
    Orch --> QC[Quality<br/>Control]
    
    Mask --> Seg[Segformer Parser]
    Mask --> Prior[Color Prior]
    Mask --> Heur[Heuristic Fallback]
    
    Color --> Class[Classical LCh]
    Color --> OT[OT Multi-Cluster]
    Color --> Hyb[Hybrid Corrector]
    
    QC --> Met[Metrics]
    QC --> Rules[Pass/Fail Rules]
    
    Met --> DeltaE[ΔE2000]
    Met --> SSIM[SSIM Texture]
    Met --> SCI[Spatial Coherence]
    Met --> Trip[Triplet Analysis]
    
    style Main fill:#ff6b6b
    style Orch fill:#4ecdc4
    style Mask fill:#e1f5ff
    style Color fill:#fff4e1
    style QC fill:#e8f5e9
```

### Component Responsibilities

| Component | Responsibility | Key Files |
|-----------|---------------|-----------|
| **Orchestrator** | Pipeline coordination, error handling | `pipeline/orchestrator.py` |
| **Masking** | Garment segmentation, mask generation | `masking/*.py` |
| **Color Correction** | Color transformation algorithms | `color/*.py` |
| **Metrics** | Quality measurement, validation | `metrics/*.py` |
| **QC** | Pass/fail evaluation | `qc/rules.py` |
| **I/O** | Data loading, image pairs | `pipeline/io.py` |

---

## Data Flow

### End-to-End Processing

```mermaid
sequenceDiagram
    participant User
    participant Main
    participant Orch as Orchestrator
    participant Mask as Masker
    participant Color as Corrector
    participant Metrics
    participant QC
    participant Output
    
    User->>Main: python -m src.main --config default.yaml
    Main->>Orch: Initialize pipeline
    
    loop For each image pair
        Orch->>Mask: Load still-life image
        Mask->>Orch: Still-life mask
        
        Orch->>Mask: Load on-model image
        Mask->>Orch: On-model mask (Segformer → color-prior → heuristic)
        
        Orch->>Color: Correct color (reference, degraded, masks)
        Color->>Color: Apply algorithm (classical/OT/hybrid)
        Color->>Orch: Corrected image
        
        Orch->>Metrics: Compute ΔE, SSIM, SCI, Triplet
        Metrics->>Orch: Metric values
        
        Orch->>QC: Evaluate quality
        QC->>Orch: PASS/FAIL
        
        Orch->>Output: Save corrected image
        Orch->>Output: Save heatmaps
        Orch->>Output: Save triplet visualization
    end
    
    Orch->>Output: Generate summary table
    Orch->>User: Complete
```

---

## Processing Pipeline

### 1. Image Loading

```mermaid
graph LR
    A[Dataset Directory] --> B{Parse Filenames}
    B --> C[Still-life Image]
    B --> D[On-model Image]
    C --> E[Image Pair]
    D --> E
    E --> F[Validation]
    F --> G[Ready for Processing]
    
    style A fill:#e3f2fd
    style G fill:#c8e6c9
```

**Implementation**: `pipeline/io.py`
- Scans `input_dir` for `{ID}_still-life.jpg` and `{ID}_on-model.jpg`
- Creates `Pair` objects with metadata
- Validates image dimensions and format

### 2. Masking Pipeline

```mermaid
graph TD
    A[Input Image] --> B{Image Type?}
    
    B -->|Still-life| C[Rembg Background Removal]
    C --> D[Largest Component]
    D --> Z[Final Mask]
    
    B -->|On-model| E[Strategy 1: Segformer]
    E --> F{Semantic Mask Valid?}
    F -->|Yes| Z
    F -->|No| G[Strategy 2: Color Prior]
    
    G --> H{Color Match Found?}
    H -->|Yes| Z
    H -->|No| I[Strategy 3: Heuristic]
    I --> Z
    
    Z --> K[Erosion + Feathering]
    K --> L[Output Mask]
    
    style E fill:#4caf50
    style G fill:#ff9800
    style I fill:#f44336
```

**Multi-Strategy Fallback**:
1. **Segformer** (primary): Semantic segmentation, most accurate
2. **Color Prior**: Match colors from reference still-life
3. **Heuristic**: Center-weighted largest component

**Details**: [Masking Components](components/masking.md)

### 3. Color Correction Algorithms

```mermaid
graph TD
    A[Input: Ref + Degraded + Masks] --> B{Correction Mode}
    
    B -->|Classical| C[Convert to LCh]
    C --> D[Compute Median Hue Shift]
    D --> E[Compute Chroma Scale]
    E --> F[Apply Transformation]
    F --> G[Feedback Loop]
    G -->|Iterate| D
    G -->|Done| Z
    
    B -->|OT| H[K-means Clustering K=3]
    H --> I[Per-Cluster Histogram Match]
    I --> J[Combine Clusters]
    J --> G
    
    B -->|Hybrid| K[Multi-Cluster Histogram]
    K --> L[Global Median Shift]
    L --> M[Light Feedback]
    M --> Z
    
    Z[Corrected Image] --> N[Alpha Blend with Feather]
    N --> O[Final Output]
    
    style C fill:#bbdefb
    style H fill:#fff9c4
    style K fill:#c8e6c9
```

**Algorithm Comparison**:

| Algorithm | Approach | Strength | Weakness |
|-----------|----------|----------|----------|
| **Classical** | Hue rotation + chroma scaling | Fast, simple | Poor on patterns |
| **OT** | Histogram matching per cluster | Handles patterns | Can be unstable |
| **Hybrid** | Histogram + global shift | Best accuracy | Slower |

**Details**: [Color Correction Components](components/color.md)

### 4. Metrics Computation

```mermaid
graph TD
    A[Corrected Image] --> B[Standard Metrics]
    A --> C[Bonus Metrics]
    
    B --> D[ΔE2000 Median]
    B --> E[ΔE2000 P95]
    B --> F[SSIM L-channel]
    B --> G[Spill Detection]
    
    C --> H[Spatial Coherence Index]
    C --> I[Triplet ΔE Analysis]
    
    H --> J[Patch-level Analysis]
    J --> K[Heatmap Visualization]
    
    I --> L[Before/After Comparison]
    L --> M[4-Panel Image]
    L --> N[Summary Table]
    
    D --> O[QC Evaluation]
    E --> O
    F --> O
    G --> O
    
    O --> P{Pass/Fail}
    
    style B fill:#e1f5fe
    style C fill:#f3e5f5
    style P fill:#fff3e0
```

**Details**: [Metrics & QC Components](components/metrics.md)

---

## Component Details

### Core Modules

#### 1. **Orchestrator** (`pipeline/orchestrator.py`)

**Purpose**: Main pipeline controller

**Responsibilities**:
- Initialize all components (maskers, correctors)
- Iterate over image pairs
- Coordinate masking → correction → metrics flow
- Handle errors and logging
- Generate summary reports

**Key Methods**:
```python
run_pairs(pairs, masks_dir, output_dir, logs_dir, cfg)
```

[→ Detailed Documentation](components/pipeline.md)

---

#### 2. **Masking Pipeline** (`masking/`)

**Purpose**: Generate precise garment masks

**Components**:

##### a. **Segformer Parser** (`segformer_parser.py`)
- Semantic segmentation using HuggingFace Transformers
- Targets: upper-clothes, coat, dress, jacket, etc.
- Device: MPS (Mac) / CUDA (Linux) / CPU
- **[Documentation](components/masking.md#segformer)**

##### b. **Color Prior** (`onmodel_color_prior.py`)
- Fallback: match colors from reference still-life
- HSV-based color distance
- Morphological cleanup
- **[Documentation](components/masking.md#color-prior)**

##### c. **Heuristic** (`base.py`)
- Last resort: center-weighted largest component
- Simple but robust
- **[Documentation](components/masking.md#heuristic)**

**Pipeline Class**: `OnModelMaskerPipeline` (`onmodel_pipeline.py`)
- Tries strategies in order
- Returns first valid mask
- Logs which strategy succeeded

---

#### 3. **Color Correction** (`color/`)

**Purpose**: Transform degraded colors to match reference

##### a. **Classical LCh** (`classical_lab.py`)
```python
ClassicalLabCorrector(deltaE_target, max_iter, feedback_strength)
```
- Converts to LCh color space
- Preserves luminance (L)
- Rotates hue, scales chroma
- Iterative feedback loop
- **[Documentation](components/color.md#classical)**

##### b. **Optimal Transport** (`ot_color_corrector.py`)
```python
OptimalTransportCorrector(num_clusters, ot_reg, max_samples, ...)
```
- K-means clustering (K=3)
- Per-cluster histogram matching
- Subsampling to prevent OOM
- **[Documentation](components/color.md#optimal-transport)**

##### c. **Hybrid** (`hybrid_corrector.py`) ⭐
```python
HybridCorrector(num_clusters, global_shift, feedback_strength, ...)
```
- Multi-cluster histogram matching
- Global median shift for precision
- Light feedback refinement
- **Best performance: 80% pass rate**
- **[Documentation](components/color.md#hybrid)**

---

#### 4. **Metrics** (`metrics/`)

**Purpose**: Measure correction quality

##### a. **Color Metrics** (`color_metrics.py`)
```python
deltaE_between_medians(ref, ref_mask, img, img_mask)
deltaE_q_to_ref_median(img, mask, ref, ref_mask, q=95)
```
- ΔE2000 (CIEDE2000) computation
- Median-to-median comparison
- Percentile metrics
- **[Documentation](components/metrics.md#color)**

##### b. **Texture Metrics** (`texture_metrics.py`)
```python
ssim_L(img1, img2, mask)
```
- SSIM on L-channel only
- Preserves texture evaluation
- **[Documentation](components/metrics.md#texture)**

##### c. **Spatial Coherence** (`spatial_coherence.py`)
```python
compute_spatial_coherence(corrected, ref_lab, ref_mask, mask, patch_size)
create_heatmap_visualization(...)
```
- Patch-level ΔE analysis
- Identifies regional failures
- Heatmap generation
- **[Documentation](components/metrics.md#spatial-coherence)**

##### d. **Triplet Analysis** (`triplet_analysis.py`)
```python
compute_triplet_delta_e(still, still_mask, original, original_mask, corrected)
create_triplet_visualization(...)
format_triplet_table(results, mode)
```
- Before/after comparison
- 4-panel visualizations
- Summary tables
- **[Documentation](components/metrics.md#triplet-analysis)**

---

#### 5. **Quality Control** (`qc/rules.py`)

**Purpose**: Pass/fail evaluation

```python
evaluate(dE_med, dE_p95, ssim, spill, 
         threshold_dE_med, threshold_dE_p95, 
         threshold_ssim, threshold_spill)
```

**Pass Criteria**:
- ΔE median ≤ 3.0
- ΔE P95 ≤ 60.0
- SSIM ≥ 0.90
- Spill ≤ 0.5

**[Documentation](components/metrics.md#quality-control)**

---

#### 6. **Configuration** (`schemas/config.py`)

**Purpose**: Type-safe configuration with Pydantic

**Schema Hierarchy**:
```python
AppConfig
├── RunConfig (paths, directories)
├── ColorConfig (correction mode, parameters)
├── MaskingConfig (erosion, feathering)
│   └── OnModelColorPriorConfig
└── QCConfig (thresholds, metrics)
```

**Benefits**:
- Type validation
- Default values
- IDE autocomplete
- Error messages

---

## Technology Stack

### Core Libraries

```mermaid
graph TD
    A[DeltaE Pipeline] --> B[Deep Learning]
    A --> C[Computer Vision]
    A --> D[Color Science]
    A --> E[Metrics]
    
    B --> B1[PyTorch 2.8.0]
    B --> B2[Transformers]
    B --> B3[Segformer]
    
    C --> C1[OpenCV]
    C --> C2[scikit-image]
    C --> C3[Rembg]
    
    D --> D1[POT Optimal Transport]
    D --> D2[scikit-learn K-means]
    
    E --> E1[CIEDE2000]
    E --> E2[SSIM]
    E --> E3[Custom SCI/Triplet]
    
    style A fill:#ff6b6b
    style B fill:#4ecdc4
    style C fill:#95e1d3
    style D fill:#f38181
    style E fill:#aa96da
```

### Platform Support

| Platform | Acceleration | Status |
|----------|--------------|--------|
| **Mac M2/M3** | MPS | ✅ Tested |
| **Linux** | CUDA 11.8+ | ✅ Compatible |
| **Windows** | CUDA / CPU | ✅ Compatible |

---

## Performance Characteristics

### Processing Time (per image)

| Component | Time | Bottleneck |
|-----------|------|------------|
| Segformer Masking | ~0.5s | GPU inference |
| Classical Correction | ~0.1s | Lightweight |
| OT Correction | ~2.0s | Histogram matching |
| Hybrid Correction | ~1.5s | Clustering |
| Metrics (all) | ~0.3s | ΔE computation |
| **Total (Hybrid)** | **~2.5s** | - |

### Memory Usage

| Operation | Peak RAM |
|-----------|----------|
| Segformer Model | ~1.5GB |
| Image Processing | ~500MB |
| OT Clustering | ~200MB |
| **Total** | **~2.5GB** |

---

## Error Handling & Robustness

### Fallback Strategies

```mermaid
graph TD
    A[Operation] --> B{Success?}
    B -->|Yes| Z[Continue]
    B -->|No| C{Fallback Available?}
    C -->|Yes| D[Try Fallback]
    D --> B
    C -->|No| E[Log Error]
    E --> F[Skip Image]
    F --> G[Continue to Next]
    
    style B fill:#4caf50
    style C fill:#ff9800
    style E fill:#f44336
```

**Examples**:
1. **Masking**: Segformer → Color Prior → Heuristic
2. **Color**: NaN check → fallback to histogram matching
3. **Metrics**: Invalid result → skip metric, continue pipeline

---

## Extensibility

### Adding New Components

#### 1. New Color Corrector

```python
# src/color/my_corrector.py
from .base import ColorCorrector

class MyCorrector(ColorCorrector):
    def correct(self, on_model_bgr, on_model_mask_core, 
                on_model_mask_full, ref_bgr, ref_mask_core):
        # Your algorithm here
        return corrected_bgr
```

Update `orchestrator.py`:
```python
elif cfg.color.mode == "my_algorithm":
    corrector = MyCorrector(...)
```

#### 2. New Metric

```python
# src/metrics/my_metric.py
def compute_my_metric(corrected, reference, mask):
    # Your metric here
    return metric_value
```

Add to orchestrator metrics section.

---

## Next Steps

- **[Installation Guide](installation.md)** - Setup and configuration
- **[Component Documentation](components/)** - Detailed module descriptions
- **[Methodology](methodology.md)** - Dataset and approach
- **[Evaluation](evaluation.md)** - Metrics and results

---

**For questions or contributions**, see the main [README](../README.md).
