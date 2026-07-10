# External Earth DEM reference support

WorldGen can now use an external NOAA ETOPO 2022 DEM prepared as:

```text
earth_dem_etopo2022_2160x1080.npz
```

Place that file in `worldgen/data/`, or set the environment variable:

```text
WORLDGEN_EARTH_DEM_NPZ=/path/to/earth_dem_etopo2022_2160x1080.npz
```

Expected NPZ arrays:

```text
elevation_m  int/float array, shape (height, width)
is_land      boolean array, same shape
source_label optional scalar string
source_url   optional scalar string
```

Recommended source: NOAA/NCEI ETOPO 2022 global relief model. ETOPO 2022 is available in global 30/60 arc-second forms and higher-resolution tiles. The data are released by NOAA under CC0-1.0 according to NOAA metadata.

The package still includes a compact offline fallback grid for convenience. The fallback is useful for smoke tests, but serious Real Earth calibration should use a prepared ETOPO reference NPZ.
