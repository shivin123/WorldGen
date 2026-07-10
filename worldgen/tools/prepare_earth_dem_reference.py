"""Prepare an external Earth DEM NPZ for WorldGen Real Earth calibration.

Preferred source: NOAA ETOPO 2022 global relief. Download a global or tiled
GeoTIFF/NetCDF from NOAA/NCEI, then convert it into the compact NPZ expected by
``worldgen.generators.real_earth_terrain``.

Examples
--------
GeoTIFF input::

    python -m worldgen.tools.prepare_earth_dem_reference \
        --input ETOPO_2022_v1_60s_N90W180_surface.tif \
        --output worldgen/data/earth_dem_etopo2022_2160x1080.npz \
        --width 2160 --height 1080

NPZ input, already containing elevation_m and is_land::

    python -m worldgen.tools.prepare_earth_dem_reference \
        --input my_dem.npz --output worldgen/data/earth_dem_etopo2022_2160x1080.npz
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert an Earth DEM to WorldGen's Real Earth NPZ format.")
    parser.add_argument("--input", required=True, help="Input DEM path (.npz, .tif/.tiff, or .png grayscale heightmap).")
    parser.add_argument("--output", required=True, help="Output .npz path.")
    parser.add_argument("--width", type=int, default=2160, help="Output width. Default: 2160.")
    parser.add_argument("--height", type=int, default=1080, help="Output height. Default: 1080.")
    parser.add_argument("--source-label", default="NOAA ETOPO 2022 global relief reference", help="Stored source label.")
    parser.add_argument("--source-url", default="https://www.ncei.noaa.gov/products/etopo-global-relief-model", help="Stored source URL.")
    parser.add_argument("--land-threshold-m", type=float, default=0.0, help="Elevation >= threshold is land. Default: 0.")
    args = parser.parse_args()

    src = Path(args.input).expanduser().resolve()
    out = Path(args.output).expanduser().resolve()
    elevation = _load_elevation(src)
    if elevation.shape != (args.height, args.width):
        elevation = _resize_elevation(elevation, args.width, args.height)
    land = elevation >= args.land_threshold_m
    elevation = elevation.round().astype(np.int32)
    elevation = np.where(land & (elevation < 1), 1, elevation)
    elevation = np.where((~land) & (elevation > -1), -1, elevation)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        elevation_m=elevation,
        is_land=land.astype(np.bool_),
        source_label=np.array(args.source_label),
        source_url=np.array(args.source_url),
    )
    print(f"Wrote {out} ({args.width} x {args.height})")
    print(f"Land fraction: {float(land.mean()):.3f}")
    print(f"Elevation range: {int(elevation.min())} to {int(elevation.max())} m")


def _load_elevation(path: Path) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix == ".npz":
        data = np.load(path)
        if "elevation_m" not in data.files:
            raise ValueError("NPZ input must contain elevation_m")
        return data["elevation_m"].astype(np.float32)
    if suffix in {".tif", ".tiff"}:
        try:
            import rasterio
            with rasterio.open(path) as ds:
                arr = ds.read(1).astype(np.float32)
                nodata = ds.nodata
                if nodata is not None:
                    arr = np.where(arr == nodata, np.nan, arr)
                if np.isnan(arr).any():
                    fill = float(np.nanmedian(arr)) if np.isfinite(np.nanmedian(arr)) else 0.0
                    arr = np.nan_to_num(arr, nan=fill)
                return arr
        except ImportError as exc:
            raise RuntimeError("GeoTIFF conversion requires rasterio. Install rasterio or convert to NPZ first.") from exc
    if suffix in {".png", ".jpg", ".jpeg"}:
        with Image.open(path) as img:
            arr = np.asarray(img.convert("L"), dtype=np.float32)
        # Grayscale fallback is intended only for manually prepared heightmaps.
        return (arr / 255.0) * 9000.0 - 6000.0
    raise ValueError(f"Unsupported input format: {path.suffix}")


def _resize_elevation(elevation: np.ndarray, width: int, height: int) -> np.ndarray:
    img = Image.fromarray(elevation.astype(np.float32), mode="F")
    return np.asarray(img.resize((width, height), Image.Resampling.BILINEAR), dtype=np.float32)


if __name__ == "__main__":
    main()
