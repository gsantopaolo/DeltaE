from __future__ import annotations
import argparse, os, sys, torch

# locate the temp SCHP clone
HERE = os.path.abspath(os.path.dirname(__file__))
SCHP = os.path.join(HERE, "third_party/schp")
sys.path.insert(0, SCHP)

def build_schp_model(num_classes: int = 20):
    # SCHP exposes resnet101 via networks.AugmentCE2P (re-exported in networks/__init__.py)
    from networks import resnet101
    # SegmentationNetwork lives under networks/segnet.py
    from networks.segnet import SegmentationNetwork

    backbone = resnet101(pretrained=None)
    model = SegmentationNetwork(backbone=backbone, num_classes=num_classes, use_deconv=False)
    model.eval()
    return model

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cpu", choices=["cpu","mps","cuda"])
    args = ap.parse_args()

    device = torch.device(args.device)
    model = build_schp_model(num_classes=20).to(device)

    ckpt = torch.load(args.ckpt, map_location="cpu")
    state = ckpt.get("state_dict", ckpt)
    state = {k.replace("module.", ""): v for k,v in state.items()}
    model.load_state_dict(state, strict=False)
    model.eval()

    # trace with a dummy tensor (adjust size if your images differ a lot)
    example = torch.randn(1, 3, 512, 384, device=device)
    ts = torch.jit.trace(model, example)
    ts.save(args.out)
    print("Saved TorchScript:", args.out)

if __name__ == "__main__":
    main()
