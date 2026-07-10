# UPDATE_CLIMATE_OVERHAUL05

This update is a **large climate/UI package** focused on the current feedback round.

## Main goals addressed

1. **Global map registry + per-run map index**
   - Added a run-independent registry of known map outputs in `worldgen/output/map_registry.py`.
   - Export now writes:
     - `diagnostics/map_registry.json`
     - `diagnostics/map_manifest.json`
     - `diagnostics/map_manifest.csv`
   - The manifest records whether each registered map/family was:
     - generated
     - missing
     - not applicable
     - partial
   - It also records **why** a map is missing/not applicable.
   - Added a new Web UI page:
     - `/map-index?output_dir=...`
   - Pipeline and stage pages now link to this new map index.

2. **Monthly map controls kept reachable**
   - Added a **floating in-viewer monthly control bar** so playback controls remain reachable while scrolling inside the map viewer.
   - This applies to the monthly temperature and monthly precipitation sequences.

3. **Monthly/seasonal precipitation readability**
   - Seasonal and monthly precipitation diagnostics now use a **log-scaled display transform**.
   - The precipitation palette was expanded so mid-range rainfall differences are easier to see.

4. **Seasonal circulation / ITCZ / pressure-belt rework**
   - ITCZ now has a stronger **warm-land / monsoon pull** and smoother longitudinal behavior.
   - Pressure belts now shift more coherently with the thermal equator instead of remaining too fixed.
   - Midlatitude storm-belt moisture recharge was broadened to reduce the sharp straight band artifact.

5. **Orographic / rain-shadow / coastal-desert climate tuning**
   - Stronger windward uplift.
   - Stronger rain-shadow drying plume.
   - Added a **cold-current coastal-desert suppression term** so west-coast deserts can emerge more naturally.

6. **Ocean gyre/current diagnostic rework**
   - Reworked the current model toward a more **loop-first** structure:
     - equatorial flow
     - equatorial countercurrent
     - westerly/subpolar return structure
     - coastal boundary intensification
   - The goal is a clearer gyre pattern and better warm/cold coastal-current behavior.

7. **Köppen / biome rework**
   - Expanded Köppen handling from the simplified set to a much broader rule set.
   - Added more full classes, including examples such as:
     - `Cwa`, `Cwb`, `Cwc`
     - `Cfc`
     - `Dwa`, `Dwb`, `Dwc`, `Dwd`
     - `Dsa`, `Dsb`, `Dsc`, `Dsd`
     - `Dfd`
   - Updated map colors/full labels.
   - Updated biome mapping to understand the new classes.

## Files changed

- `worldgen/output/map_registry.py` **(new)**
- `worldgen/output/export.py`
- `worldgen/webui.py`
- `worldgen/visualization/system_plot.py`
- `worldgen/generators/climate_seasonal.py`
- `worldgen/generators/biome_generator.py`

## Quick validation performed

Tested with:

```bash
python -m worldgen.main --preset synthetic-earth --output-dir /tmp/test_big_update --yes --skip-json --map-width 512 --map-height 256 --image-max-width 512 --koppen-detail cell
```

The run completed successfully and produced the new map manifest outputs.

## Recommended user test run

For the next real feedback pass, I recommend:

```bash
python -m worldgen.main --preset synthetic-earth --output-dir output_u05_big_climate --seed 3857 --map-width 4096 --map-height 2048 --image-max-width 4096 --skip-json --yes --full-res-images --koppen-detail cell
```

If you want to inspect the Web UI with a generated world instead, a good companion run is:

```bash
python -m worldgen.main --seed 7 --output-dir output_u05_big_generated --map-width 4096 --map-height 2048 --image-max-width 4096 --skip-json --yes --koppen-detail local4
```

## Notes

- Older runs that were created before this update can still be browsed, but the best results from the new map index come after rerunning export-producing stages.
- The ocean-current / gyre logic is improved, but I would still expect another refinement round after you inspect the new outputs.
- The Köppen expansion is intentionally substantial, so it is worth reviewing both the map appearance and the downstream biome behavior together.
