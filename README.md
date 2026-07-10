# WorldGen

WorldGen is a Python procedural world generator for building a physically informed single-star solar system and a detailed main-planet simulation. It currently includes staged generation for solar system setup, main planet selection, terrain/tectonics, hydrology, climate, biomes/Köppen classification, diagnostics, and a local web UI for map review.

## Requirements

Python 3.12 is recommended.

Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

## Start the local Web UI

```powershell
python -m worldgen.webui --open
```

## Example full staged run

```powershell
python -m worldgen.pipeline new --output-dir staged_web_run_1 --seed 143 --map-width 4096 --map-height 2048 --image-max-width 4096 --terrain-mode plate_history_v4 --tectonic-history-myr 2000 --tectonic-timestep-myr 2 --tectonic-grid-scale native --continental-shelf-strength 1.65 --erosion-deposition-strength 1.35 --shelf-width-factor 0.9 --v4-topology-strength 1.0 --v4-island-strength 1.0 --v4-rift-strength 1.0 --system-architecture random --main-planet-preference earthlike --moon-strength strong --run-to outputs --skip-json --yes --full-res-images --suppress-polar-land --koppen-detail local9 --climate-mode seasonal_v5
```

## Faster smoke test

```powershell
python -m worldgen.pipeline new --output-dir staged_smoke_test --seed 81 --map-width 1024 --map-height 512 --image-max-width 1024 --terrain-mode plate_history_v4 --run-to outputs --skip-json --yes --climate-mode seasonal_v5
```

## Climate modes

The older modes are intentionally preserved during review:

- `legacy`
- `seasonal_v1`
- `seasonal_v2`
- `seasonal_v3`
- `seasonal_v4`
- `seasonal_v5`

`seasonal_v5` is the current review mode, but older modes are useful for rollback and comparison until the project is consolidated.

## Terrain modes

The current review terrain mode is:

```text
plate_history_v4
```

Earlier terrain modes are kept for comparison and rollback until cleanup is complete.

## Project notes

- Generated output folders can be very large.
- Reference data under `worldgen/data/` is part of the source tree unless deliberately moved to external storage later.
