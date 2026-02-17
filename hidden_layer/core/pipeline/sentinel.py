import os
import numpy as np
import rasterio

from pystac_client import Client
import planetary_computer as pc
import stackstac


def choose_best_item(items):
    # lowest cloud, then newest
    def key(it):
        cloud = it.properties.get("eo:cloud_cover", 999)
        dt = it.datetime
        return (cloud, -dt.timestamp())
    return sorted(items, key=key)[0]


def robust_to_uint8(x):
    x = x.astype(np.float64)
    x = np.where(x == 0.0, np.nan, x)  # treat 0 as nodata for display scaling
    if np.all(np.isnan(x)):
        return np.zeros_like(x, dtype=np.uint8)
    lo, hi = np.nanpercentile(x, [2, 98])
    if not np.isfinite(lo) or not np.isfinite(hi) or (hi - lo) < 1e-10:
        return np.zeros_like(x, dtype=np.uint8)
    y = (x - lo) / (hi - lo + 1e-10)
    y = np.clip(y, 0, 1)
    y = np.nan_to_num(y, nan=0.0)
    return (y * 255).astype(np.uint8)


def write_singleband_geotiff(path, arr_2d, transform, crs):
    h, w = arr_2d.shape
    profile = {
        "driver": "GTiff",
        "height": h,
        "width": w,
        "count": 1,
        "dtype": "float32",
        "crs": crs,
        "transform": transform,
        "compress": "deflate",
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(arr_2d.astype(np.float32), 1)


def write_rgb_geotiff(path, rgb_3hw_uint8, transform, crs):
    _, h, w = rgb_3hw_uint8.shape
    profile = {
        "driver": "GTiff",
        "height": h,
        "width": w,
        "count": 3,
        "dtype": "uint8",
        "crs": crs,
        "transform": transform,
        "compress": "deflate",
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(rgb_3hw_uint8.astype(np.uint8))


def fetch_sentinel_to_dir(
    bbox_latlon,                 # [minlon, minlat, maxlon, maxlat]
    date_range,                  # "YYYY-MM-DD/YYYY-MM-DD"
    max_cloud,                   # int
    out_dir,                     # run_dir / "data"
    epsg=3857,
    resolution=10,
):
    os.makedirs(out_dir, exist_ok=True)

    catalog = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")

    search = catalog.search(
        collections=["sentinel-2-l2a"],
        bbox=bbox_latlon,
        datetime=date_range,
        query={"eo:cloud_cover": {"lt": max_cloud}},
        max_items=50,
    )

    items = list(search.items())
    if not items:
        raise RuntimeError("No Sentinel-2 items found for this AOI/date/cloud filter.")

    best = choose_best_item(items)
    best = pc.sign(best)

    assets = ["B02", "B03", "B04", "B08"]  # Blue, Green, Red, NIR

    # stackstac needs dtype compatible with fill_value (your original fix)
    data = stackstac.stack(
        [best],
        assets=assets,
        resolution=resolution,
        bounds_latlon=bbox_latlon,
        epsg=epsg,
        chunksize=2048,
        dtype="float64",
        fill_value=0.0,
        rescale=False,
    ).compute()

    band_data = data[0].values  # (4, H, W)
    b02, b03, b04, b08 = band_data

    # sanity
    if float(np.mean(b04 == 0.0)) > 0.95:
        raise RuntimeError("Too much NoData returned (likely edge/tile mismatch). Try a slightly larger AOI.")

    crs = data.crs
    transform = data.transform

    # Save red + nir float32
    write_singleband_geotiff(os.path.join(out_dir, "red.tif"), b04, transform, crs)
    write_singleband_geotiff(os.path.join(out_dir, "nir.tif"), b08, transform, crs)

    # Save RGB uint8
    rgb_u8 = np.stack(
        [robust_to_uint8(b04), robust_to_uint8(b03), robust_to_uint8(b02)],
        axis=0
    )
    write_rgb_geotiff(os.path.join(out_dir, "rgb.tif"), rgb_u8, transform, crs)

    return {
        "item_id": best.id,
        "datetime": best.datetime.isoformat(),
        "cloud": best.properties.get("eo:cloud_cover"),
        "crs": str(crs),
    }
