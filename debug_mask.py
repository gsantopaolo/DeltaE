#!/usr/bin/env python
"""Debug script to visualize masks and identify high ΔE pixels."""
import cv2
import numpy as np
from skimage.color import rgb2lab, deltaE_ciede2000

# Load images
om_orig = cv2.imread('data/dataset/on-model-00000.jpg')
om_corr = cv2.imread('data/outputs/corrected-on-model-00000.jpg')
still = cv2.imread('data/dataset/still-life-00000.jpg')
mask = cv2.imread('data/masks/on-model-00000.png', cv2.IMREAD_GRAYSCALE)

print(f"Mask shape: {mask.shape}")
print(f"Mask pixels: {np.count_nonzero(mask)}")

# Get still-life reference color (median LAB)
still_rgb = cv2.cvtColor(still, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
still_lab = rgb2lab(still_rgb)
still_mask = cv2.imread('data/masks/still-life-00000.png', cv2.IMREAD_GRAYSCALE)
sm = still_mask.astype(bool)
ref_L = np.median(still_lab[..., 0][sm])
ref_a = np.median(still_lab[..., 1][sm])
ref_b = np.median(still_lab[..., 2][sm])
print(f"\nReference LAB: L={ref_L:.1f}, a={ref_a:.1f}, b={ref_b:.1f}")

# Compute ΔE map for corrected image
corr_rgb = cv2.cvtColor(om_corr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
corr_lab = rgb2lab(corr_rgb)
ref_1x1 = np.array([[[ref_L, ref_a, ref_b]]], dtype=np.float64)
dE_map = deltaE_ciede2000(corr_lab, np.tile(ref_1x1, (corr_lab.shape[0], corr_lab.shape[1], 1)))

# Extract ΔE values within mask
m = mask.astype(bool)
dE_vals = dE_map[m]

print(f"\nΔE statistics within mask:")
print(f"  Median: {np.median(dE_vals):.2f}")
print(f"  Mean: {np.mean(dE_vals):.2f}")
print(f"  P95: {np.percentile(dE_vals, 95):.2f}")
print(f"  P99: {np.percentile(dE_vals, 99):.2f}")
print(f"  Max: {np.max(dE_vals):.2f}")

# Find high ΔE pixels (>20)
high_dE_mask = (dE_map > 20) & m
n_high = np.count_nonzero(high_dE_mask)
pct_high = 100 * n_high / np.count_nonzero(m)
print(f"\nPixels with ΔE > 20: {n_high} ({pct_high:.1f}%)")

# Create visualization
vis = om_corr.copy()
# Highlight high-error pixels in red
vis[high_dE_mask] = [0, 0, 255]

# Save visualization
cv2.imwrite('debug_high_dE_pixels.jpg', vis)
print(f"\n✅ Saved visualization to: debug_high_dE_pixels.jpg")
print(f"   (High ΔE pixels shown in red)")

# Erosion test: what if we erode the mask more aggressively?
for erosion in [5, 10, 15]:
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erosion*2+1, erosion*2+1))
    eroded = cv2.erode(mask, kernel)
    em = eroded.astype(bool)
    if em.sum() > 1000:
        dE_eroded = dE_map[em]
        print(f"\nWith erosion={erosion}px:")
        print(f"  Pixels: {np.count_nonzero(em)}")
        print(f"  ΔE median: {np.median(dE_eroded):.2f}")
        print(f"  ΔE P95: {np.percentile(dE_eroded, 95):.2f}")
