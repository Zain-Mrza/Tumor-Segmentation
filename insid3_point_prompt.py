"""Point-prompted segmentation with INSID3-debiased DINOv3 features.

Image + positive/negative point prompts -> similarity map -> binary mask.

    1. Extract DINOv3 patch features and remove the positional subspace (INSID3Debias).
    2. For each prompt point, take the feature of the patch it falls in.
    3. Similarity map = mean cosine sim to positive patches - mean cosine sim to negative
       patches (just the positive mean if no negatives are given). Range: [-2, 2].
    4. Bilinearly upsample the map to the original image size.
    5. Mask = map > thresh (default 0).

Points are (x, y) in ORIGINAL image pixel coordinates.

Example:
    python insid3_point_prompt.py --image frame.png \
        --repo_dir /path/to/dinov3 --weights /path/to/dinov3_vitl16_pretrain_lvd1689m.pth \
        --pos 512,300 540,320 --neg 100,100 --thresh 0 \
        --out_mask mask.png --out_vis vis.png
"""

from __future__ import annotations

import argparse

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# First file (positional debiasing). Rename the module here if you saved it under another name.
from insid3_debias import INSID3Debias, load_dinov3_local

Point = tuple[float, float]


# ──────── Core ────────

def points_to_patches(points: list[Point], img_hw: tuple[int, int],
                      feat_hw: tuple[int, int]) -> tuple[torch.Tensor, torch.Tensor]:
    """(x, y) in original pixels -> (iy, ix) patch indices on the feature grid."""
    H, W = img_hw
    h, w = feat_hw
    xs = np.asarray([p[0] for p in points], dtype=np.float64)
    ys = np.asarray([p[1] for p in points], dtype=np.float64)
    if (xs < 0).any() or (ys < 0).any() or (xs >= W).any() or (ys >= H).any():
        raise ValueError(f"Point outside image of size (W={W}, H={H}): {points}")
    iy = np.minimum((ys * h / H).astype(int), h - 1)
    ix = np.minimum((xs * w / W).astype(int), w - 1)
    return torch.as_tensor(iy), torch.as_tensor(ix)


@torch.no_grad()
def similarity_map(feats: torch.Tensor, pos_points: list[Point], neg_points: list[Point] | None,
                   img_hw: tuple[int, int]) -> np.ndarray:
    """feats: (C, h, w) L2-normalized. Returns (H, W) float map at original resolution."""
    if not pos_points:
        raise ValueError("Need at least one positive point.")
    h, w = feats.shape[-2:]

    def mean_sim(points):
        iy, ix = points_to_patches(points, img_hw, (h, w))
        q = feats[:, iy.to(feats.device), ix.to(feats.device)]          # (C, N)
        return torch.einsum("cn,chw->nhw", q, feats).mean(dim=0)        # (h, w)

    s = mean_sim(pos_points)
    if neg_points:
        s = s - mean_sim(neg_points)

    s = F.interpolate(s[None, None].float(), size=img_hw, mode="bilinear", align_corners=False)
    return s[0, 0].cpu().numpy()


@torch.no_grad()
def segment_from_points(model: INSID3Debias, image: str | Image.Image,
                        pos_points: list[Point], neg_points: list[Point] | None = None,
                        thresh: float = 0.0, use_debias: bool = True):
    """Returns (mask bool (H, W), similarity map float (H, W)) at original image size."""
    if isinstance(image, str):
        image = Image.open(image)
    image = image.convert("RGB")
    img_hw = (image.height, image.width)

    deb, raw = model(image, return_raw=True)                  # (1, C, h, w) each
    feats = (deb if use_debias else raw)[0]
    sim = similarity_map(feats, pos_points, neg_points, img_hw)
    return sim > thresh, sim


# ──────── Visualization ────────

def save_vis(path: str, image: Image.Image, sim: np.ndarray, mask: np.ndarray,
             pos_points: list[Point], neg_points: list[Point], thresh: float, alpha: float = 0.45):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    img = np.asarray(image.convert("RGB"))
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    axes[0].imshow(img)
    axes[0].set_title("Image + prompts")

    axes[1].imshow(img)
    im = axes[1].imshow(sim, cmap="viridis", alpha=alpha)
    fig.colorbar(im, ax=axes[1], fraction=0.046)
    axes[1].set_title("Similarity (mean pos - mean neg)" if neg_points else "Similarity (mean pos)")

    overlay = img.astype(np.float32).copy()
    overlay[mask] = (1 - alpha) * overlay[mask] + alpha * np.array([255, 140, 0], np.float32)
    axes[2].imshow(overlay.astype(np.uint8))
    axes[2].contour(mask, levels=[0.5], colors="orange", linewidths=1)
    axes[2].set_title(f"Mask (map > {thresh:g})  |  {mask.mean() * 100:.1f}% of pixels")

    for ax in axes:
        if pos_points:
            ax.scatter(*zip(*pos_points), c="lime", s=90, marker="x", linewidths=2.5)
        if neg_points:
            ax.scatter(*zip(*neg_points), c="red", s=90, marker="x", linewidths=2.5)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(path, dpi=100)
    plt.close(fig)


# ──────── CLI ────────

def _parse_point(s: str) -> Point:
    try:
        x, y = s.split(",")
        return float(x), float(y)
    except ValueError:
        raise argparse.ArgumentTypeError(f"Point must be 'x,y', got '{s}'")


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image", required=True)
    ap.add_argument("--pos", type=_parse_point, nargs="+", required=True, metavar="X,Y",
                    help="Positive points in original pixel coords")
    ap.add_argument("--neg", type=_parse_point, nargs="*", default=[], metavar="X,Y",
                    help="Negative points in original pixel coords")
    ap.add_argument("--thresh", type=float, default=0.0, help="Similarity-map threshold")
    # Model
    ap.add_argument("--repo_dir", required=True, help="Local clone of facebookresearch/dinov3")
    ap.add_argument("--model_name", default="dinov3_vitl16")
    ap.add_argument("--weights", default=None, help="Local .pth checkpoint")
    ap.add_argument("--image_size", type=int, nargs="+", default=[768],
                    help="Encoder input size: one value (square) or H W")
    ap.add_argument("--patch_grid", type=int, nargs="+", default=None,
                    help="Patches per side (one value or h w); overrides --image_size")
    ap.add_argument("--svd_components", type=int, default=500)
    ap.add_argument("--no_debias", action="store_true", help="Use raw (non-debiased) features")
    ap.add_argument("--no_bf16", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    # Outputs
    ap.add_argument("--out_mask", default="mask.png", help="Binary mask PNG (0/255)")
    ap.add_argument("--out_sim", default=None, help="Optional .npy of the raw similarity map")
    ap.add_argument("--out_vis", default=None, help="Optional visualization PNG")
    return ap.parse_args()


def _size_arg(v):
    if v is None:
        return None
    return v[0] if len(v) == 1 else tuple(v[:2])


def main():
    args = parse_args()

    encoder = load_dinov3_local(args.repo_dir, args.model_name, args.weights, device=args.device)
    model = INSID3Debias(encoder, image_size=_size_arg(args.image_size),
                         patch_grid=_size_arg(args.patch_grid),
                         svd_components=args.svd_components,
                         use_bf16=not args.no_bf16, device=args.device)

    image = Image.open(args.image).convert("RGB")
    mask, sim = segment_from_points(model, image, args.pos, args.neg,
                                    thresh=args.thresh, use_debias=not args.no_debias)

    Image.fromarray(mask.astype(np.uint8) * 255).save(args.out_mask)
    if args.out_sim:
        np.save(args.out_sim, sim)
    if args.out_vis:
        save_vis(args.out_vis, image, sim, mask, args.pos, args.neg, args.thresh)

    print(f"grid={model.patch_grid}  sim range=[{sim.min():.3f}, {sim.max():.3f}]  "
          f"mask coverage={mask.mean() * 100:.2f}%  -> {args.out_mask}")


if __name__ == "__main__":
    main()