dataset
https://github.com/shadow2496/VITON-HD
https://github.com/aimagelab/dress-code
https://github.com/switchablenorms/DeepFashion2
https://nlp.cs.unc.edu/data/jielei/hadi/street2shop/wheretobuyit/README.txt



Dataset part of VITON-HD repo https://github.com/shadow2496/VITON-HD?utm_source=chatgpt.com
download https://drive.google.com/file/d/1tLx8LRp-sxDp0EcYmYoV_vXdSc-jJ79w/view

SCHP too long to implement on a mac, switching to Segformer human parsser for simplicity

python prepare-pairs.py \
  --cloth-dir excluded/datasets/train/cloth \
  --image-dir excluded/datasets/train/image \
  --dest-dir dataset \
  --limit 300

pip install -r requirements.txt

SCHP weights
gdown --folder https://drive.google.com/drive/folders/1uOaQCpNtosIjEL2phQKEdiYd0Td18jNo -O weights

python - <<'PY'
from huggingface_hub import hf_hub_download
hf_hub_download("levihsu/OOTDiffusion",
                "checkpoints/humanparsing/exp-schp-201908261155-lip.pth",
                local_dir="weights")
PY

SAM 2 checkpoint
python - <<'PY'
from huggingface_hub import hf_hub_download
hf_hub_download("facebook/sam2-hiera-base-plus", "sam2_hiera_base_plus.pt", local_dir="weights")
PY


git clone https://github.com/facebookresearch/sam2.git third_party/sam2
pip install -e third_party/sam2


# 2) clone SCHP to the temp dir (not into your repo)
git clone https://github.com/PeikeLi/Self-Correction-Human-Parsing third_party/schp
conda create -n temp_schp python=3.8
conda activate temp_schp
pip install opencv-python==4.10.0.84
conda deactivate


# (Install PyTorch suited to your GPU separately)
python -m src.main --config configs/default.yaml --input-dir dataset --limit 300


# DeltaE
Automated garment color correction with precise color fidelity (ΔE2000) and texture preservation.
