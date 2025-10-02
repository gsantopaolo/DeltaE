from __future__ import annotations
from pydantic import BaseModel, Field
from typing import Literal, List

class OnModelColorPriorConfig(BaseModel):
    tau_base: float = 12.0
    tau_k: float = 3.0
    dilate_px: int = 1
    min_mask_pixels: int = 2000
    neutral_chroma_thresh: float = 10.0  # if ref chroma < this, enable L-gate
    l_gate_lo: float = -12.0             # L tolerance below ref L*
    l_gate_hi: float = +18.0             # L tolerance above ref L*
    band_top_pct: float = 0.25           # keep 25%..85% vertical band
    band_bot_pct: float = 0.85

class OnModelSCHPConfig(BaseModel):
    enabled: bool = True
    weights_path: str = "weights/checkpoints/exp-schp-201908261155-lip.pth"
    model_name: str = "mattmdjaga/segformer_b2_clothes"  # HuggingFace model for Segformer
    device: Literal["cpu", "mps", "cuda"] = "mps"
    include_labels: List[str] = Field(
        default_factory=lambda: [
            "upper-clothes", "coat", "dress", "jacket", "t-shirt",
            "vest", "hoodie", "cardigan", "blouse", "sweater"
        ]
    )

class OnModelSAM2Config(BaseModel):
    enabled: bool = False
    checkpoint_path: str = "weights/sam2_hiera_base_plus.pt"
    points_from_mask: bool = True
    expand_box_px: int = 16

# NEW: SAM v1 (Segment Anything) config
class OnModelSAMV1Config(BaseModel):
    enabled: bool = True
    checkpoint_path: str = "weights/sam/sam_vit_h_4b8939.pth"
    model_type: Literal["vit_b", "vit_l", "vit_h"] = "vit_h"
    points_from_mask: bool = True
    expand_box_px: int = 16

class OnModelMaskingConfig(BaseModel):
    method_order: List[Literal["schp", "sam_v1", "sam2", "color_prior", "heuristic"]] = \
        Field(default_factory=lambda: ["schp", "sam_v1", "color_prior", "heuristic"])
    schp: OnModelSCHPConfig = Field(default_factory=OnModelSCHPConfig)
    sam2: OnModelSAM2Config = Field(default_factory=OnModelSAM2Config)
    sam_v1: OnModelSAMV1Config = Field(default_factory=OnModelSAMV1Config)
    color_prior: OnModelColorPriorConfig = Field(default_factory=OnModelColorPriorConfig)
    use_crf: bool = False  # requires pydensecrf if True

class PathsConfig(BaseModel):
    input_dir: str
    masks_dir: str
    output_dir: str
    logs_dir: str

class RunConfig(BaseModel):
    limit: int = 300
    num_workers: int = 2
    save_debug: bool = False
    write_corrected: bool = True  # <-- so YAML flag is honored

class MaskingConfig(BaseModel):
    erosion_px: int = 2
    feather_px: int = 2
    # legacy/back-compat
    onmodel_color_prior: OnModelColorPriorConfig = Field(default_factory=OnModelColorPriorConfig)
    # new unified on-model block
    on_model: OnModelMaskingConfig = Field(default_factory=OnModelMaskingConfig)

class ColorConfig(BaseModel):
    mode: Literal["classical", "ot", "hybrid"] = "hybrid"
    deltaE_target: float = 2.0
    
    # OT (Optimal Transport) specific parameters
    ot_num_clusters: int = 3  # Number of color clusters for multi-color garments
    ot_reg: float = 0.01  # Entropic regularization for OT (higher = smoother)
    ot_use_clustering: bool = True  # Enable multi-cluster mode for complex patterns
    ot_min_cluster_size: int = 500  # Minimum pixels per cluster
    ot_max_samples: int = 5000  # Max pixels for OT computation (prevents OOM)

class QCConfig(BaseModel):
    max_deltaE_median: float = 3.0
    max_deltaE_p95: float = 60.0
    min_ssim_L: float = 0.90
    max_spill_deltaE: float = 0.5
    
    # Spatial Coherence Index (SCI) - Bonus metric
    enable_sci: bool = True  # Compute spatial coherence metrics
    sci_patch_size: int = 32  # Size of patches for spatial analysis
    sci_save_heatmap: bool = True  # Save heatmap visualization to disk
    
    # Triplet Analysis - Quantitative proof of correction effectiveness
    enable_triplet_analysis: bool = True  # Compare still-life vs original vs corrected
    save_triplet_table: bool = True  # Save summary table as markdown file with datetime
    save_triplet_visualization: bool = True  # Save 4-panel comparison images with difference maps

class AppConfig(BaseModel):
    paths: PathsConfig
    run: RunConfig = Field(default_factory=RunConfig)
    masking: MaskingConfig = Field(default_factory=MaskingConfig)
    color: ColorConfig = Field(default_factory=ColorConfig)
    qc: QCConfig = Field(default_factory=QCConfig)
