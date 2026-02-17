import os
import numpy as np
import rasterio
from scipy.ndimage import gaussian_filter, label
import geopandas as gpd
from shapely.geometry import MultiPoint
from pathlib import Path

def run_dinov2_hotspots(in_path: str | Path, out_geo: str | Path):
    with rasterio.open(in_path) as src:
        score = src.read(1).astype(np.float32)
        transform = src.transform
        crs = src.crs
    
    score = gaussian_filter(np.nan_to_num(score, nan=0.0), sigma=3)
    thr = np.percentile(score, 97)
    mask = score > thr
    labeled, n = label(mask)
    
    polys = []
    for rid in range(1, n + 1):
        ys, xs = np.where(labeled == rid)
        if len(xs) < 80:
            continue
        pts = [transform * (int(x), int(y)) for x, y in zip(xs, ys)]
        polys.append(MultiPoint(pts).convex_hull)
    
    if not polys:
        # Create an empty GeoDataFrame if no hotspots found
        gdf = gpd.GeoDataFrame(columns=["thr", "geometry"], crs=crs)
    else:
        gdf = gpd.GeoDataFrame({"thr": [float(thr)] * len(polys)}, geometry=polys, crs=crs)
    
    out_geo = Path(out_geo)
    out_geo.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(out_geo, driver="GeoJSON")
    return str(out_geo)

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 2:
        run_dinov2_hotspots(sys.argv[1], sys.argv[2])