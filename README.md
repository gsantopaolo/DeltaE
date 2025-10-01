dataset
https://github.com/shadow2496/VITON-HD
https://github.com/aimagelab/dress-code
https://github.com/switchablenorms/DeepFashion2
https://nlp.cs.unc.edu/data/jielei/hadi/street2shop/wheretobuyit/README.txt



Dataset part of VITON-HD repo https://github.com/shadow2496/VITON-HD?utm_source=chatgpt.com
download https://drive.google.com/file/d/1tLx8LRp-sxDp0EcYmYoV_vXdSc-jJ79w/view

python prepare-pairs.py \
  --cloth-dir excluded/datasets/train/cloth \
  --image-dir excluded/datasets/train/image \
  --dest-dir dataset \
  --limit 300

pip install -r requirements.txt
# (Install PyTorch suited to your GPU separately)
python -m src.main --config configs/default.yaml --input-dir dataset --limit 300


# DeltaE
Automated garment color correction with precise color fidelity (ΔE2000) and texture preservation.
