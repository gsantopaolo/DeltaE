# src/masking/segformer_human_parsing.py
from __future__ import annotations
import numpy as np
import cv2
import torch
import warnings
from PIL import Image
from typing import Iterable, List, Dict

from ..utils.logging_utils import get_logger

# ATR (Apparel Transfer) / LIP (Look Into Person) label mappings
# Using ATR-style labels (compatible with most fashion parsing models)
ATR_LABEL_MAP: Dict[str, int] = {
    "background": 0,
    "hat": 1,
    "hair": 2,
    "sunglasses": 3,
    "upper-clothes": 4,
    "skirt": 5,
    "pants": 6,
    "dress": 7,
    "belt": 8,
    "left-shoe": 9,
    "right-shoe": 10,
    "face": 11,
    "left-leg": 12,
    "right-leg": 13,
    "left-arm": 14,
    "right-arm": 15,
    "bag": 16,
    "scarf": 17,
}

# Extended labels for some models (adding common garment types)
EXTENDED_GARMENT_LABELS = {
    "coat": 4,      # map to upper-clothes
    "jacket": 4,    # map to upper-clothes
    "t-shirt": 4,   # map to upper-clothes
    "vest": 4,      # map to upper-clothes
    "hoodie": 4,    # map to upper-clothes
    "cardigan": 4,  # map to upper-clothes
    "blouse": 4,    # map to upper-clothes
    "sweater": 4,   # map to upper-clothes
}


def _largest_component(binm: np.ndarray) -> np.ndarray:
    """Extract largest connected component from binary mask."""
    num, labels, stats, _ = cv2.connectedComponentsWithStats(binm, connectivity=8)
    if num <= 1:
        return binm
    largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    return np.where(labels == largest, 255, 0).astype(np.uint8)


class SegformerHumanParser:
    """
    HuggingFace Segformer-based human parsing for garment segmentation.
    Drop-in replacement for SCHP CLI wrapper with no C++ compilation needed.
    
    Compatible with Mac M2 (MPS), CPU, and CUDA.
    """

    def __init__(
        self,
        model_name: str = "mattmdjaga/segformer_b2_clothes",
        include_labels: Iterable[str] = ("upper-clothes", "dress", "coat", "jacket"),
        device: str = "mps",
    ) -> None:
        self.log = get_logger("Segformer")
        self.model_name = model_name
        self.device_str = device
        self.include_labels = list(include_labels)
        self._ready = False
        
        # Map label names to IDs (with extended garment support)
        self.include_ids: List[int] = []
        label_map = {**ATR_LABEL_MAP, **EXTENDED_GARMENT_LABELS}
        for label in self.include_labels:
            if label in label_map:
                label_id = label_map[label]
                if label_id not in self.include_ids:
                    self.include_ids.append(label_id)
            else:
                self.log.warning(f"⚠️ Unknown label '{label}', skipping")
        
        if not self.include_ids:
            self.log.error("❌ No valid garment labels specified")
            return
        
        # Try to load model
        try:
            from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation
            
            self.log.info(f"🔄 Loading Segformer model: {model_name}")
            
            # Determine device
            if device == "mps" and torch.backends.mps.is_available():
                self.device = torch.device("mps")
            elif device == "cuda" and torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
                if device != "cpu":
                    self.log.info(f"⚠️ {device} not available, falling back to CPU")
            
            # Suppress the reduce_labels warning (it's a harmless model config mismatch)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=".*reduce_labels.*")
                # Load processor with use_fast=True to use faster Rust-based processor
                self.processor = AutoImageProcessor.from_pretrained(model_name, use_fast=True)
            
            self.model = AutoModelForSemanticSegmentation.from_pretrained(model_name)
            self.model.to(self.device)
            self.model.eval()
            
            self._ready = True
            self.log.info(f"✅ Segformer loaded on {self.device} | Target labels: {self.include_labels}")
            
        except ImportError:
            self.log.error("❌ 'transformers' not installed. Run: pip install transformers")
        except Exception as e:
            self.log.error(f"❌ Failed to load Segformer: {e}")

    def is_ready(self) -> bool:
        """Check if model is loaded and ready."""
        return self._ready

    def get_mask(self, image_bgr: np.ndarray) -> np.ndarray:
        """
        Generate binary garment mask from BGR image.
        
        Args:
            image_bgr: Input image in BGR format (OpenCV standard)
            
        Returns:
            Binary mask (uint8, 0/255) of target garment regions
        """
        if not self._ready:
            raise RuntimeError("SegformerHumanParser not ready (model failed to load)")
        
        H, W = image_bgr.shape[:2]
        
        # Convert BGR → RGB for HuggingFace models
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(image_rgb)
        
        # Preprocess
        inputs = self.processor(images=pil_img, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Inference
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits  # shape: [1, num_classes, H', W']
        
        # Upsample to original resolution
        logits = torch.nn.functional.interpolate(
            logits,
            size=(H, W),
            mode="bilinear",
            align_corners=False,
        )
        
        # Get predicted class per pixel
        pred_labels = logits.argmax(dim=1)[0].cpu().numpy()  # [H, W]
        
        # Create binary mask for target garment classes
        mask = np.isin(pred_labels, self.include_ids).astype(np.uint8) * 255
        
        # Post-processing: morphological cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = _largest_component(mask)
        
        return mask
