

### 3. **Console Output - Summary Table** (High Priority)
**Filename**: `docs/images/console_output.jpg`

**What to capture**: Terminal screenshot showing the triplet analysis summary table

**Shows**:
- Rounded grid table with tabulate formatting
- Multiple rows of results
- Summary statistics at bottom
- Clean, readable output

**How to capture**:
```bash
python -m src.main --config configs/default.yaml --limit 5
# Screenshot the summary table section at the end
```

**Usage**: README, installation.md

---

### 4. **Before/After Comparison** (Medium Priority)
**Filename**: `docs/images/before_after_comparison.jpg`

**What to create**: Side-by-side comparison showing:
- Left: Original degraded on-model image
- Right: Corrected on-model image
- Label showing ΔE improvement

**Usage**: README, methodology.md

---

### 5. **Architecture Diagram** (Optional - can use mermaid)
**Filename**: `docs/images/architecture_overview.png`

**What to show**: Pipeline flow diagram (already in mermaid, but a rendered version could be nice)

**Usage**: README

---

### 6. **Configuration Example** (Low Priority)
**Filename**: `docs/images/config_example.jpg`

**What to capture**: Screenshot of `configs/default.yaml` with syntax highlighting

**Shows**: YAML configuration with comments

**Usage**: installation.md

---

### 7. **Dataset Sample** (Low Priority)
**Filename**: `docs/images/dataset_sample.jpg`

**What to show**: Grid of sample input images showing variety (solid colors, patterns, multi-color)

**Usage**: methodology.md

---

## Screenshot Specifications

### General Guidelines
- **Format**: JPG or PNG
- **Resolution**: High enough to be readable (at least 1200px width)
- **Quality**: Clear, well-lit, professional
- **Annotations**: Add arrows/labels if helpful

### Specific Requirements

#### Terminal Screenshots
- Use dark theme or light theme (consistent)
- Full terminal width visible
- Clear, readable font
- No personal info in path/username

#### Image Screenshots
- Show full image or clear crop
- Include file name if relevant
- Maintain aspect ratio

---

## Quick Capture Commands

### Generate All Output Files
```bash
# Run pipeline to generate all visualizations
python -m src.main --config configs/default.yaml --limit 5

# Files will be in:
# - data/outputs/corrected-on-model-*-triplet.jpg
# - data/outputs/corrected-on-model-*-hm.jpg
# - data/outputs/triplet_analysis_*.md
```

### Take Terminal Screenshot
1. Run: `python -m src.main --config configs/default.yaml --limit 3`
2. Wait for summary table to appear
3. Screenshot the terminal window
4. Crop to show just the table section

---

## Checklist

After adding screenshots, update the README.md:

- [ ] Replace `[TODO: Add screenshot here - Triplet comparison...]` with actual image
- [ ] Replace `[TODO: Add screenshot here - SCI spatial heatmap...]` with actual image  
- [ ] Replace `[TODO: Add screenshot here - Console output...]` with actual image
- [ ] Verify all image paths are correct
- [ ] Test that images display in GitHub markdown preview
- [ ] Add alt text for accessibility

---

## Image Placeholders Currently in README

Search for `[TODO: Add screenshot here` in README.md to find all placeholders.

Current placeholders:
1. Line ~170: Triplet comparison
2. Line ~175: SCI heatmap  
3. Line ~180: Console output

---

**After screenshots are added, this file can be deleted.**
