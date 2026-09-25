"""INSID3 positional debiasing on a frozen, locally loaded DINOv3 encoder.

Stripped-down version of INSID3: only feature extraction + positional debiasing
(no candidate matching, clustering, CRF, or keypoint matching).

Method from: https://github.com/visinf/insid3

DINOv3 embeddings naturally carry positional information, making patchwise 
similarity matching impossible (a tumor patch on the right and left side will have
low similarity scores just because they are far away, even though they are 
semantically similar). 

This paper removes the positional information from DINOv3 embeddings
enabling robust patchwise similarity scoring.

1. Pure gaussian noise has no semantic information, so an image of gaussian noise 
   passed through DINOv3 will result in embeddings only encoding positional information.
2. Do SVD of these embeddings and take the top K vectors.
3. Use these vectors to construct a basis, call it the positional subspace.
4. Now, when we want to pass an actual image through DINOv3, we calculate its embeddings
   like usual.
5. Next, we will project the regular DINOv3 embeddings along the orthogonal complement
   of the positional basis we previously calculated.
6. The embeddings now have no (or significantly less) positional information, enabling
   us to take a patch containing tumor, compute its cosine similarity across all other 
   patches in an image.
7. We will ideally have other patches containing tumor "light up" in the similarity map,
   which we can then threshold to obtain binary masks.
"""

from __future__ import annotations

import warnings

import einops
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
from torch import nn

# LVD-1689M (web) weights use ImageNet stats; SAT-493M (satellite) weights use different ones.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def load_dinov3_local(
    repo_dir: str,
    model_name: str = "dinov3_vitl16",
    weights: str | None = None,
    device: str = "cuda",
) -> nn.Module:
    """Load DINOv3 from a local clone of facebookresearch/dinov3.

    Args:
        repo_dir: path to the local dinov3 repo (the folder containing hubconf.py).
        model_name: hub entry, e.g. dinov3_vits16, dinov3_vitb16, dinov3_vitl16, dinov3_vith16plus, dinov3_vit7b16.
        weights: path to a local .pth checkpoint (or URL). None -> hub default.
    """
    kwargs = {} if weights is None else {"weights": weights}
    encoder = torch.hub.load(repo_dir, model_name, source="local", **kwargs)
    return encoder.eval().requires_grad_(False).to(device)


def _pair(x: int | tuple[int, int]) -> tuple[int, int]:
    return (x, x) if isinstance(x, int) else tuple(x)


class INSID3Debias(nn.Module):
    """Frozen DINOv3 features with INSID3 positional debiasing.

    Resolution can be set either by input size in pixels (``image_size``) or directly by
    the patch-grid size (``patch_grid``, number of patches per side). With a 16px patch,
    patch_grid=64 <=> image_size=1024. Non-square sizes are allowed: (H, W).
    """

    def __init__(
        self,
        encoder: nn.Module,
        image_size: int | tuple[int, int] = 1024,
        patch_grid: int | tuple[int, int] | None = None,
        svd_components: int = 500,
        mean: tuple[float, ...] = IMAGENET_MEAN,
        std: tuple[float, ...] = IMAGENET_STD,
        use_bf16: bool = True,
        device: str = "cuda",
    ):
        super().__init__()
        self.encoder = encoder.eval().requires_grad_(False).to(device)
        self.device = device
        self.patch_size = int(getattr(encoder, "patch_size", 16))
        self.svd_components = svd_components
        self.mean, self.std = list(mean), list(std)
        self.use_bf16 = use_bf16
        self._basis_cache: dict[tuple[int, int], torch.Tensor] = {}
        self.set_resolution(image_size=image_size, patch_grid=patch_grid)

    # ──────── Resolution ────────

    def set_resolution(
        self,
        image_size: int | tuple[int, int] | None = None,
        patch_grid: int | tuple[int, int] | None = None,
    ) -> None:
        """Change the working resolution. ``patch_grid`` takes precedence over ``image_size``."""
        p = self.patch_size
        if patch_grid is not None:
            gh, gw = _pair(patch_grid)
            self.image_size = (gh * p, gw * p)
        elif image_size is not None:
            H, W = _pair(image_size)
            if H % p or W % p:
                H, W = round(H / p) * p, round(W / p) * p
                warnings.warn(f"image_size snapped to multiple of patch size {p}: {(H, W)}")
            self.image_size = (H, W)
        else:
            raise ValueError("Provide image_size or patch_grid.")
        # Build (and cache) the positional basis for this resolution up front.
        self.positional_basis = self.get_positional_basis(self.image_size)

    @property
    def patch_grid(self) -> tuple[int, int]:
        return self.image_size[0] // self.patch_size, self.image_size[1] // self.patch_size

    # ──────── Preprocessing ────────

    def preprocess(self, image: str | Image.Image | torch.Tensor) -> torch.Tensor:
        """Path / PIL -> resized, normalized (1, 3, H, W). Tensors are assumed already
        normalized and are only resized if needed."""
        if isinstance(image, str):
            image = Image.open(image)
        if isinstance(image, Image.Image):
            x = TF.to_tensor(image.convert("RGB"))
            x = TF.resize(x, list(self.image_size), antialias=True)
            x = TF.normalize(x, self.mean, self.std)
        else:
            x = image.float()
            if x.shape[-2:] != self.image_size:
                x = F.interpolate(x.view(-1, *x.shape[-3:]), size=self.image_size,
                                  mode="bilinear", align_corners=False)
        return x.view(-1, *x.shape[-3:]).to(self.device)

    # ──────── Encoder ────────

    @torch.no_grad()
    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        """(N, 3, H, W) -> last-layer patch features (N, C, h, w), float32."""
        if self.use_bf16 and x.is_cuda:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                f = self.encoder.get_intermediate_layers(x, n=1, reshape=True)[0]
        else:
            f = self.encoder.get_intermediate_layers(x, n=1, reshape=True)[0]
        return f.float()

    @torch.no_grad()
    def extract_features(self, images: str | Image.Image | torch.Tensor | list) -> torch.Tensor:
        """Returns L2-normalized raw features (N, C, h, w)."""
        if isinstance(images, (list, tuple)):
            x = torch.cat([self.preprocess(im) for im in images], dim=0)
        else:
            x = self.preprocess(images)
        return F.normalize(self._encode(x), p=2, dim=1)

    # ──────── Positional debiasing (INSID3) ────────

    @torch.no_grad()
    def get_positional_basis(self, image_size: tuple[int, int]) -> torch.Tensor:
        """Estimate the positional subspace (C, k) from a content-free image via SVD.

        The basis is resolution-dependent, so it is cached per input size.
        """
        image_size = tuple(image_size)
        if image_size in self._basis_cache:
            return self._basis_cache[image_size]

        blank = torch.zeros(1, 3, *image_size)
        blank = TF.normalize(blank, self.mean, self.std).to(self.device)
        f = F.normalize(self._encode(blank), p=2, dim=1)

        E = einops.rearrange(f, "b c h w -> c (b h w)")
        E = E - E.mean(dim=1, keepdim=True)
        U, _, _ = torch.linalg.svd(E, full_matrices=False)

        C, n_tokens = E.shape
        k_max = min(C, n_tokens - 1)  # centering removes one dof
        k = self.svd_components
        if k > k_max:
            warnings.warn(
                f"svd_components={k} exceeds rank {k_max} at grid {f.shape[-2:]}; clamping. "
                f"Consider lowering svd_components at small resolutions."
            )
            k = k_max
        basis = U[:, :k].contiguous()
        self._basis_cache[image_size] = basis
        return basis

    def debias(self, fmaps: torch.Tensor, basis: torch.Tensor | None = None) -> torch.Tensor:
        """Project features (..., C, h, w) onto the orthogonal complement of the
        positional subspace and re-normalize."""
        if basis is None:
            basis = self.positional_basis
        *lead, C, H, W = fmaps.shape
        X = F.normalize(fmaps.reshape(-1, C, H * W).float(), p=2, dim=1)
        B = basis.to(device=X.device, dtype=X.dtype)
        X_deb = X - B @ (B.T @ X)  # (I - BB^T) X without forming the CxC matrix
        return F.normalize(X_deb, p=2, dim=1).reshape(*lead, C, H, W)

    @torch.no_grad()
    def forward(self, images, return_raw: bool = False):
        """images -> debiased features (N, C, h, w) [, raw normalized features]."""
        raw = self.extract_features(images)
        deb = self.debias(raw)
        return (deb, raw) if return_raw else deb


if __name__ == "__main__":
    encoder = load_dinov3_local(
        repo_dir="/path/to/dinov3",
        model_name="dinov3_vitl16",
        weights="/path/to/dinov3_vitl16_pretrain_lvd1689m.pth",
    )
    model = INSID3Debias(encoder, patch_grid=48, svd_components=500)  # 768x768 input
    feats = model("frame.png")                      # (1, C, 48, 48) debiased
    model.set_resolution(patch_grid=(40, 72))       # e.g. 640x1152 for 16:9 video
    deb, raw = model(["a.png", "b.png"], return_raw=True)
    print(deb.shape, raw.shape, model.patch_grid)