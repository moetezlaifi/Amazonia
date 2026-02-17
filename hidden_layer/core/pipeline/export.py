import os
import numpy as np
import rasterio
from rasterio.warp import transform_bounds
from PIL import Image


def tif_rgb_to_png(rgb_tif, out_png):
    with rasterio.open(rgb_tif) as src:
        rgb = src.read()  # (3,H,W)
    rgb = np.transpose(rgb, (1, 2, 0)).astype(np.uint8)  # HWC
    Image.fromarray(rgb).save(out_png)


def tif_heat_to_png(heat_tif, out_png):
    with rasterio.open(heat_tif) as src:
        heat = src.read(1).astype(np.float32)

    heat = np.nan_to_num(heat, nan=0.0)
    lo = np.percentile(heat, 2)
    hi = np.percentile(heat, 98)
    heat = np.clip((heat - lo) / (hi - lo + 1e-6), 0, 1)

    # simple “inferno-ish” colormap (same idea as your demo)
    r = (255 * heat).astype(np.uint8)
    g = (150 * (heat ** 0.7)).astype(np.uint8)
    b = (50 * (1 - heat)).astype(np.uint8)
    rgb = np.stack([r, g, b], axis=-1)
    Image.fromarray(rgb).save(out_png)


def compute_bbox_4326(any_tif_path):
    with rasterio.open(any_tif_path) as src:
        left, bottom, right, top = src.bounds
        crs = src.crs
    minlon, minlat, maxlon, maxlat = transform_bounds(crs, "EPSG:4326", left, bottom, right, top)
    return [float(minlon), float(minlat), float(maxlon), float(maxlat)]
