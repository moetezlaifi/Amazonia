import os
import numpy as np
import rasterio


def read_single(path):
    with rasterio.open(path) as src:
        arr = src.read(1).astype(np.float32)
        profile = src.profile
    return arr, profile


def write_single(path, arr, profile):
    profile = profile.copy()
    profile.update(dtype="float32", count=1, compress="deflate")
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(arr.astype(np.float32), 1)


def run_spectral(data_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    red, profile = read_single(os.path.join(data_dir, "red.tif"))
    nir, _ = read_single(os.path.join(data_dir, "nir.tif"))

    ndvi = (nir - red) / (nir + red + 1e-6)
    ndvi[(red == 0) | (nir == 0)] = np.nan

    lo, hi = np.nanpercentile(ndvi, [2, 98])
    ndvi_norm = (ndvi - lo) / (hi - lo + 1e-6)
    ndvi_norm = np.clip(ndvi_norm, 0, 1)

    spectral_score = 1.0 - ndvi_norm  # lower NDVI => higher anomaly proxy

    write_single(os.path.join(out_dir, "ndvi.tif"), ndvi, profile)
    write_single(os.path.join(out_dir, "spectral_score.tif"), spectral_score, profile)

    return {"ndvi_lo": float(lo), "ndvi_hi": float(hi)}
