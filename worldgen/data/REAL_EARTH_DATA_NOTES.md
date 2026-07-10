# Real Earth data notes

WorldGen Real Earth Terrain mode is for calibration, not for authoritative GIS output.

## DEM / terrain source

The preferred terrain source is an external NOAA ETOPO 2022 global relief grid converted to WorldGen NPZ format. WorldGen will load it from either:

```text
WORLDGEN_EARTH_DEM_NPZ=/path/to/earth_dem_etopo2022_2160x1080.npz
```

or:

```text
worldgen/data/earth_dem_etopo2022_2160x1080.npz
```

Prepare the file with:

```powershell
python -m worldgen.tools.prepare_earth_dem_reference --input <ETOPO GeoTIFF/NPZ> --output worldgen/data/earth_dem_etopo2022_2160x1080.npz
```

If no external/prepared ETOPO NPZ is present, WorldGen falls back to the compact bundled Earth relief grid. That fallback is artifact-repaired, but it is not a survey-grade DEM and should not be used as the final calibration standard.

## Köppen-Geiger reference

WorldGen bundles a small Köppen-Geiger reference raster and legend from the BSD-licensed `kgcpy` package for simulated-vs-reference climate classification diagnostics.

## Real Earth calibration policy

Real Earth Terrain mode now uses the same temperature and precipitation calculation as procedural generated worlds. Earth-specific geographic temperature/precipitation nudges were removed because they can create straight-line rectangular artifacts and hide model weaknesses.

Use the Earth diagnostics to locate remaining issues:

```text
earth_artifact_scores.csv
earth_artifact_terrain_gradient.png
earth_artifact_temperature_gradient.png
earth_artifact_precipitation_gradient.png
earth_artifact_row_column_score.png
earth_koppen_reference_agreement.csv
earth_koppen_match_map.png
```
