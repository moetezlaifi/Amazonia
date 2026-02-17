from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import rasterio
from scipy.ndimage import label


@dataclass
class Hotspot:
    lon: float
    lat: float
    score: float
    priority: str
    reasons: list[str]


def _priority_from_score(score: float) -> str:
    # Tune as needed for demo stability
    if score >= 0.85:
        return "High"
    if score >= 0.70:
        return "Medium"
    return "Low"


def extract_hotspots(
    score_tif_path: str,
    *,
    max_hotspots: int = 25,
    min_pixels: int = 30,
    threshold_quantile: float = 0.985,
) -> list[dict[str, Any]]:
    """
    Reads a single-band anomaly/spectral score GeoTIFF and returns hotspots as dicts:
      {lon, lat, score, priority, reasons}
    """
    with rasterio.open(score_tif_path) as ds:
        arr = ds.read(1).astype("float32")
        nodata = ds.nodata
        if nodata is not None:
            arr = np.where(arr == nodata, np.nan, arr)

        # Robust threshold: quantile of non-nan pixels
        valid = arr[np.isfinite(arr)]
        if valid.size == 0:
            return []

        thr = float(np.quantile(valid, threshold_quantile))
        mask = np.isfinite(arr) & (arr >= thr)

        # Connected components (8-connectivity)
        structure = np.array([[1, 1, 1],
                              [1, 1, 1],
                              [1, 1, 1]], dtype=np.uint8)
        labeled, n = label(mask.astype(np.uint8), structure=structure)
        if n == 0:
            return []

        hotspots = []
        for comp_id in range(1, n + 1):
            ys, xs = np.where(labeled == comp_id)
            if ys.size < min_pixels:
                continue

            scores = arr[ys, xs]
            mean_score = float(np.nanmean(scores))

            # centroid in pixel coords
            cy = float(np.mean(ys))
            cx = float(np.mean(xs))

            # pixel -> geographic (lon/lat). rasterio uses (col, row)
            lon, lat = ds.transform * (cx, cy)

            priority = _priority_from_score(mean_score)
            hotspots.append(
                {
                    "lon": float(lon),
                    "lat": float(lat),
                    "score": mean_score,
                    "priority": priority,
                    "reasons": ["spectral anomaly"],  # later add "structure anomaly"
                }
            )

        # Sort by score desc, keep top N
        hotspots.sort(key=lambda h: h["score"], reverse=True)
        return hotspots[:max_hotspots]
