"""Optional - Grad-CAM heat maps for the final proposed model on held-out test images.

Shows whether the model attends to lesions rather than background.

Example:
  python gradcam.py --split splits/bell_split.csv --results results/bell --n 8
"""
import argparse, os, json
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image

from common import build_transforms, load_or_compute_glcm
from models import build_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--results", required=True)
    ap.add_argument("--model", default="proposed")
    ap.add_argument("--n", type=int, default=8)
    a = ap.parse_args()
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    run = json.load(open(os.path.join(a.results, a.model, "run_config.json")))
    df = pd.read_csv(a.split); classes = df.sort_values("label").class_name.unique().tolist()
    model = build_model(a.model, len(classes), pretrained=False, img_size=run["img_size"], **run["hp"])
    model.load_state_dict(torch.load(os.path.join(a.results, a.model, "final_model.pt"), map_location="cpu"))
    model.eval()
    glcm = load_or_compute_glcm(df, os.path.splitext(a.split)[0] + "_glcm.npz")
    dev = df[~df.is_test]; G = np.stack([glcm[p] for p in dev.path]); mu, sd = G.mean(0), G.std(0) + 1e-6
    _, tf = build_transforms(run["img_size"])
    test = df[df.is_test].groupby("label", group_keys=False).apply(lambda d: d.sample(min(len(d), a.n // len(classes) or 1), random_state=0))

    feats = {}
    layer = model.backbone.layer4 if hasattr(model.backbone, "layer4") else list(model.backbone.children())[-1]
    layer.register_forward_hook(lambda m, i, o: feats.__setitem__("a", o))
    layer.register_full_backward_hook(lambda m, gi, go: feats.__setitem__("g", go[0]))

    fig, axes = plt.subplots(2, len(test), figsize=(2.4 * len(test), 5))
    axes = np.atleast_2d(axes).reshape(2, -1)
    for j, (_, r) in enumerate(test.iterrows()):
        img = Image.open(r.path).convert("RGB")
        x = tf(img).unsqueeze(0); g = torch.from_numpy(((glcm[r.path] - mu) / sd).astype(np.float32)).unsqueeze(0)
        out = model(x, g); c = out.argmax(1).item()
        model.zero_grad(); out[0, c].backward()
        w = feats["g"].mean((2, 3), keepdim=True)
        cam = F.relu((w * feats["a"]).sum(1, keepdim=True))
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0].detach().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        shown = np.asarray(img.resize(x.shape[-2:][::-1]))
        axes[0, j].imshow(shown); axes[0, j].set_title(f"true: {r.class_name}", fontsize=7)
        axes[1, j].imshow(shown); axes[1, j].imshow(cam, alpha=0.45, cmap="jet")
        axes[1, j].set_title(f"pred: {classes[c]}", fontsize=7)
        for ax in axes[:, j]:
            ax.axis("off")
    fig.tight_layout(); fig.savefig(os.path.join(a.results, "gradcam.png"), dpi=300)
    print("saved", os.path.join(a.results, "gradcam.png"))


if __name__ == "__main__":
    main()
