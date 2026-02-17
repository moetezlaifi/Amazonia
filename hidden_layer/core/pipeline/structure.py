from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import rasterio
import torch
import torch.nn.functional as F
from torchvision import transforms

# Cache model within the process so repeated runs are fast
_MODEL = None


def _normalize01(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    lo, hi = np.percentile(x[np.isfinite(x)], 2), np.percentile(x[np.isfinite(x)], 98)
    if not np.isfinite(lo) or not np.isfinite(hi) or (hi - lo) < 1e-6:
        return np.zeros_like(x, dtype=np.float32)
    y = (x - lo) / (hi - lo + 1e-6)
    return np.clip(y, 0.0, 1.0)


def _load_model(model_name: str, device: str):
    global _MODEL
    if _MODEL is None:
        # First run may download weights – do once before demo.
        _MODEL = torch.hub.load("facebookresearch/dinov2", model_name, pretrained=True)
        _MODEL.eval().to(device)
    return _MODEL


def run_dino_structure_anomaly(
    rgb_tif_path: str | Path,
    out_tif_path: str | Path,
    *,
    model_name: str = "dinov2_vits14",
    max_side: int = 512,
    device: Optional[str] = None,
) -> dict:
    """
    Very simple 'structure anomaly' score:
    anomaly = 1 - cosine_similarity(patch_embedding, mean_embedding)

    Writes a 1-band float32 GeoTIFF aligned to the input RGB GeoTIFF.
    """
    rgb_tif_path = Path(rgb_tif_path)
    out_tif_path = Path(out_tif_path)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Optional CPU tuning
    if device == "cpu":
        try:
            torch.set_num_threads(max(1, (torch.get_num_threads() // 2)))
        except Exception:
            pass

    with rasterio.open(rgb_tif_path) as src:
        rgb = src.read([1, 2, 3]).astype(np.uint8)  # (3,H,W)
        profile = src.profile
    rgb_hwc = np.transpose(rgb, (1, 2, 0))  # (H,W,3)
    H, W, _ = rgb_hwc.shape

    # Downsample for speed
    scale = min(1.0, max_side / float(max(H, W)))
    if scale < 1.0:
        new_h = int(round(H * scale))
        new_w = int(round(W * scale))
        x = torch.from_numpy(rgb_hwc).permute(2, 0, 1).unsqueeze(0).float()  # 1,3,H,W
        x = F.interpolate(x, size=(new_h, new_w), mode="bilinear", align_corners=False)
        rgb_small = x[0].permute(1, 2, 0).byte().cpu().numpy()
    else:
        rgb_small = rgb_hwc
        new_h, new_w = H, W

    model = _load_model(model_name, device)

    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                             std=(0.229, 0.224, 0.225))
    ])

    x = tfm(rgb_small).unsqueeze(0).to(device)  # (1,3,new_h,new_w)

    patch = 14
    crop_h = (new_h // patch) * patch
    crop_w = (new_w // patch) * patch
    x = x[:, :, :crop_h, :crop_w]

    with torch.inference_mode():
        feats = model.forward_features(x)["x_norm_patchtokens"]  # (1,N,D)
        f = feats[0]  # (N,D)
        f = F.normalize(f, dim=1)
        mu = f.mean(dim=0, keepdim=True)

        cos = (f * mu).sum(dim=1)
        anomaly = (1.0 - cos).clamp(min=0.0)

    gh = crop_h // patch
    gw = crop_w // patch
    anomaly_map = anomaly.view(gh, gw)[None, None, :, :].float().cpu()  # (1,1,gh,gw)

    # Upsample to cropped small size
    an_small = F.interpolate(anomaly_map, size=(crop_h, crop_w),
                             mode="bilinear", align_corners=False)[0, 0].numpy()

    # Place into full downsampled size
    small_full = np.zeros((new_h, new_w), dtype=np.float32)
    small_full[:crop_h, :crop_w] = an_small
    small_full = _normalize01(small_full)

    # Upsample back to original H,W
    t = torch.from_numpy(small_full)[None, None, :, :].float()
    up = F.interpolate(t, size=(H, W), mode="bilinear", align_corners=False)[0, 0].numpy()
    up = _normalize01(up)

    # Write GeoTIFF aligned with input
    out_profile = profile.copy()
    out_profile.update(dtype="float32", count=1, compress="deflate")

    out_tif_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_tif_path, "w", **out_profile) as dst:
        dst.write(up.astype(np.float32), 1)

    return {
        "enabled": True,
        "device": device,
        "model": model_name,
        "max_side": max_side,
        "output": str(out_tif_path),
    }
