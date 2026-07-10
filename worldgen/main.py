"""Command-line entry point for WorldGen."""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

from worldgen.config import load_config, merge_cli_overrides
from worldgen.generators.earth_generator import generate_synthetic_earth_system, generate_real_earth_terrain_system
from worldgen.generators.system_generator import generate_star_system
from worldgen.output.export import system_to_text, write_outputs
from worldgen.random_utils import create_rng
from worldgen import performance


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate a physically informed starter star system.")
    parser.add_argument("--config", help="Path to a JSON config file.")
    parser.add_argument("--seed", type=int, help="Random seed. Same seed produces the same system.")
    parser.add_argument("--preset", choices=["generated", "synthetic-earth", "real-earth-terrain"], default="generated", help="Use generated worlds, Synthetic Earth calibration, or bundled Real Earth terrain calibration.")
    parser.add_argument("--planet-count", type=int, help="Override number of planets.")
    parser.add_argument("--output-dir", help="Directory where outputs will be written.")
    parser.add_argument("--map-width", type=int, help="Main Planet map width in grid cells/pixels.")
    parser.add_argument("--map-height", type=int, help="Main Planet map height in grid cells/pixels.")
    parser.add_argument("--no-images", action="store_true", help="Skip PNG visualization output.")
    parser.add_argument("--preview", action="store_true", help="Fast preview run: cap the map at 1024 x 512 and keep all major layers enabled.")
    parser.add_argument("--fast", action="store_true", help="Fast run: cap the map at 2048 x 1024, skip region analysis, and write fewer heavy outputs.")
    parser.add_argument("--skip-hydrology", action="store_true", help="Skip hydrology, drainage basins, rivers, and dependent river/lake effects.")
    parser.add_argument("--skip-biomes", action="store_true", help="Skip biome generation. Implies region analysis will use placeholder biome data.")
    parser.add_argument("--skip-regions", action="store_true", help="Skip 24 x 12 region analysis and region CSV output.")
    parser.add_argument("--yes", action="store_true", help="Confirm very large map runs without an interactive prompt.")
    parser.add_argument("--skip-json", action="store_true", help="Skip compact system.json output for faster large-map runs.")
    parser.add_argument("--save-rasters", action="store_true", help="Write full-resolution raster grids to main_planet_rasters.npz for fast binary storage/analysis.")
    parser.add_argument("--skip-diagnostics", action="store_true", help="Skip diagnostic CSV/TXT outputs for faster large-map runs.")
    parser.add_argument("--image-max-width", type=int, default=None, help="Downsample large raster PNG outputs to this width without changing simulation resolution.")
    parser.add_argument("--full-res-images", action="store_true", help="Do not auto-downsample large-map PNG outputs.")
    parser.add_argument("--no-accelerated-terrain", action="store_true", help="Compatibility flag; terrain is always generated directly at the requested resolution now.")
    parser.add_argument("--no-accelerated-climate", action="store_true", help="Compatibility flag; climate is generated at requested resolution except explicit Köppen smoothing.")
    parser.add_argument("--accelerated-terrain", action="store_true", help="Deprecated/no-op. Terrain acceleration has been disabled to avoid upscaled land/water artifacts.")
    parser.add_argument("--accelerated-climate", action="store_true", help="Deprecated/no-op. Climate acceleration has been disabled except preview/fast map-size caps.")
    parser.add_argument(
        "--koppen-detail",
        choices=("regional", "local9", "local4", "cell"),
        default=None,
        help="Controls Köppen classification smoothing. 'cell' uses raw cell climate; 'local4'/'local9' use small local groups; 'regional' is smoother.",
    )
    parser.add_argument(
        "--climate-mode",
        choices=("seasonal_v5", "seasonal_v4", "seasonal_v3", "seasonal_v2", "seasonal_v1", "legacy"),
        default=None,
        help="Climate backend. seasonal_v5 adds component-based moisture/rainfall coupling on top of seasonal_v4; seasonal_v4 is the refined structured atmosphere + basin-ocean mode; seasonal_v3 preserves the first basin-ocean mode; seasonal_v2 preserves the structured atmosphere-only review mode; seasonal_v1 is the stable seasonal overhaul; legacy preserves the previous heuristic model.",
    )
    parser.add_argument(
        "--terrain-mode",
        choices=("procedural_legacy", "plate_tectonic_v1", "plate_history_v1", "plate_history_v2", "plate_history_v3", "plate_history_v4"),
        default=None,
        help="Terrain backend for generated worlds. Earth presets always use real-world data through Stage 3.",
    )
    parser.add_argument(
        "--tectonic-history-myr",
        type=float,
        default=None,
        help="For plate_history_v1/v2/v3/v4, simulate this many millions of years of coarse plate history before deriving terrain. Default is randomized from the planet age/geology.",
    )
    parser.add_argument(
        "--tectonic-timestep-myr",
        type=float,
        default=None,
        help="For plate_history_v1/v2/v3/v4, requested geological timestep in Myr. The engine may internally sample fewer epochs for speed while preserving the total history length.",
    )
    parser.add_argument(
        "--tectonic-grid-scale",
        choices=("legacy", "preview", "normal", "high", "native", "custom"),
        default=None,
        help="For plate_history_v1/v2/v3/v4, choose requested tectonic detail label. WorldGen now uses the stable macro-history grid plus full-resolution detail unless developer-only raw-grid mode is explicitly unlocked.",
    )
    parser.add_argument(
        "--tectonic-grid-policy",
        choices=("stable", "raw"),
        default=None,
        help="Developer/debug only. raw is ignored unless --allow-experimental-tectonic-grid is also supplied. Normal runs use the stable macro-history grid plus full-resolution detail.",
    )
    parser.add_argument(
        "--allow-experimental-tectonic-grid",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--tectonic-grid-width",
        type=int,
        default=None,
        help="For plate_history_v1/v2/v3/v4 with --tectonic-grid-scale custom, explicit tectonic simulation grid width.",
    )
    parser.add_argument(
        "--tectonic-grid-height",
        type=int,
        default=None,
        help="For plate_history_v1/v2/v3/v4 with --tectonic-grid-scale custom, explicit tectonic simulation grid height. Defaults to requested aspect ratio.",
    )
    parser.add_argument(
        "--suppress-polar-land",
        action="store_true",
        help="Short-term plate-terrain option: convert most high-latitude land to ocean/ice-ocean to avoid distorted polar landmasses on equirectangular maps.",
    )
    return parser


def _cap_map_size(width: int | None, height: int | None, cap_w: int, cap_h: int) -> tuple[int | None, int | None, bool]:
    if width is None or height is None:
        return width, height, False
    if width <= cap_w and height <= cap_h:
        return width, height, False
    return min(width, cap_w), min(height, cap_h), True


def _confirm_large_map(width: int, height: int, args: argparse.Namespace) -> None:
    cells = width * height
    if cells <= 8_500_000:
        return
    level = "large" if cells <= 16_800_000 else "very large"
    print(f"[warning] Requested {width} x {height} = {cells:,} cells ({level}).", flush=True)
    print("[warning] WorldGen stores several full-size grids: terrain, climate, hydrology, basins, biomes, and images.", flush=True)
    if cells >= 30_000_000:
        print("[warning] This size can take a long time and use a lot of RAM. Use --preview or --fast for iteration.", flush=True)
    if args.yes:
        print("[progress] Large-map confirmation accepted with --yes.", flush=True)
        return
    if not sys.stdin.isatty():
        raise SystemExit("Large map requires --yes in non-interactive mode.")
    answer = input("Continue? Type YES to proceed: ").strip()
    if answer != "YES":
        raise SystemExit("Cancelled large-map run.")


def main() -> None:
    performance.reset()
    args = build_parser().parse_args()
    config = load_config(args.config)
    if args.seed is None and config.seed is None:
        generated_seed = random.randint(1, 999_999_999)
    else:
        generated_seed = args.seed if args.seed is not None else config.seed

    map_width = args.map_width
    map_height = args.map_height
    if args.preview:
        map_width, map_height, capped = _cap_map_size(map_width or 1024, map_height or 512, 1024, 512)
        if capped:
            print(f"[progress] Preview mode capped map to {map_width} x {map_height}.", flush=True)
    elif args.fast:
        map_width, map_height, capped = _cap_map_size(map_width or 2048, map_height or 1024, 2048, 1024)
        if capped:
            print(f"[progress] Fast mode capped map to {map_width} x {map_height}.", flush=True)

    generate_hydrology = not args.skip_hydrology
    generate_biomes = not args.skip_biomes
    generate_regions = not (args.skip_regions or args.fast)
    if not generate_hydrology:
        print("[progress] Hydrology skipped; rivers, basins, and lake candidates will be placeholders.", flush=True)
    if not generate_biomes:
        print("[progress] Biomes skipped; biome output will be placeholders.", flush=True)
    if not generate_regions:
        print("[progress] Region analysis skipped.", flush=True)

    config = merge_cli_overrides(
        config,
        {
            "seed": generated_seed,
            "planet_count": args.planet_count,
            "map_width": map_width,
            "map_height": map_height,
            "generate_hydrology": generate_hydrology,
            "generate_biomes": generate_biomes,
            "generate_regions": generate_regions,
            "fast_mode": args.fast or args.preview,
            # Terrain and climate are now generated at the requested grid.
            # --preview/--fast reduce the requested map size before generation; they do not upscale.
            "no_accelerated_terrain": True,
            "no_accelerated_climate": True,
            "koppen_detail": args.koppen_detail,
            "climate_generation_mode": args.climate_mode,
            "terrain_generation_mode": "real_world_stage3" if args.preset in {"synthetic-earth", "real-earth-terrain"} else args.terrain_mode,
            "suppress_polar_land": args.suppress_polar_land,
            "tectonic_history_myr": args.tectonic_history_myr,
            "tectonic_timestep_myr": args.tectonic_timestep_myr,
            "tectonic_grid_scale": args.tectonic_grid_scale,
            "tectonic_grid_policy": ("raw" if getattr(args, "allow_experimental_tectonic_grid", False) and args.tectonic_grid_policy == "raw" else ("stable" if args.tectonic_grid_policy == "raw" else args.tectonic_grid_policy)),
            "allow_experimental_tectonic_grid": bool(getattr(args, "allow_experimental_tectonic_grid", False)),
            "tectonic_grid_width": args.tectonic_grid_width,
            "tectonic_grid_height": args.tectonic_grid_height,
        },
    )

    width = config.planet_profile.map_width
    height = config.planet_profile.map_height
    _confirm_large_map(width, height, args)

    image_max_width = args.image_max_width
    if image_max_width is None and not args.full_res_images and width * height > 4_500_000:
        image_max_width = 2048
        print("[progress] Large map detected; raster PNGs will be downsampled to width 2048. Use --full-res-images to disable.", flush=True)
    elif args.full_res_images:
        print("[progress] Full-resolution image output requested.", flush=True)

    performance.add_metadata("preset", args.preset)
    performance.add_metadata("seed", config.seed)
    performance.add_metadata("map_width", width)
    performance.add_metadata("map_height", height)
    performance.add_metadata("cells", width * height)
    performance.add_metadata("images", not args.no_images)
    performance.add_metadata("accelerated_terrain", not config.planet_profile.no_accelerated_terrain)
    performance.add_metadata("accelerated_climate", not config.planet_profile.no_accelerated_climate)
    print("[progress] Terrain will be simulated at requested resolution (no terrain upscaling).", flush=True)
    print("[progress] Climate will be simulated at requested resolution (except Köppen smoothing).", flush=True)
    print(f"[progress] Climate generation mode: {config.planet_profile.climate_generation_mode}.", flush=True)
    if args.accelerated_terrain or args.accelerated_climate:
        print("[warning] --accelerated-terrain/--accelerated-climate are deprecated no-ops in this build.", flush=True)
    performance.add_metadata("image_max_width", image_max_width or "full/default")
    performance.add_metadata("skip_hydrology", args.skip_hydrology)
    performance.add_metadata("skip_biomes", args.skip_biomes)
    performance.add_metadata("skip_regions", args.skip_regions or args.fast)
    performance.add_metadata("no_accelerated_terrain", args.no_accelerated_terrain)
    performance.add_metadata("no_accelerated_climate", args.no_accelerated_climate)
    performance.add_metadata("koppen_detail", config.planet_profile.koppen_detail)
    performance.add_metadata("climate_generation_mode", config.planet_profile.climate_generation_mode)
    performance.add_metadata("terrain_generation_mode", config.planet_profile.terrain_generation_mode)
    performance.add_metadata("suppress_polar_land", config.planet_profile.suppress_polar_land)
    performance.add_metadata("tectonic_history_myr", config.planet_profile.tectonic_history_myr)
    performance.add_metadata("tectonic_timestep_myr", config.planet_profile.tectonic_timestep_myr)
    if args.preset == "generated":
        print(f"[progress] Terrain generation mode: {config.planet_profile.terrain_generation_mode}.", flush=True)
        if config.planet_profile.suppress_polar_land:
            print("[progress] Polar land suppression enabled for plate-terrain output.", flush=True)
    else:
        print("[progress] Earth presets use real-world terrain data through Stage 3; terrain-mode override is ignored for preset terrain.", flush=True)
    if args.no_accelerated_terrain:
        print("[progress] Full-resolution terrain generation requested; the terrain step may be much slower.", flush=True)
    total_start = time.perf_counter()
    performance.mark(f"Starting WorldGen preset={args.preset}, seed={config.seed}, map={width}x{height}.")
    print(f"[progress] Starting WorldGen preset={args.preset}, seed={config.seed}, map={width}x{height}.", flush=True)
    rng = create_rng(config.seed)
    if args.preset == "synthetic-earth":
        system = generate_synthetic_earth_system(rng, config)
    elif args.preset == "real-earth-terrain":
        system = generate_real_earth_terrain_system(rng, config)
    else:
        system = generate_star_system(rng, config)

    generation_elapsed = time.perf_counter() - total_start
    performance.record_stage("total generation before output", generation_elapsed)
    performance.mark(f"Generation complete in {generation_elapsed:.1f}s. Preparing text summary.")
    print(f"[progress] Generation complete in {generation_elapsed:.1f}s. Preparing text summary.", flush=True)
    print(system_to_text(system))

    if args.output_dir:
        print("[progress] Writing output files...", flush=True)
        write_start = time.perf_counter()
        write_outputs(
            system,
            args.output_dir,
            include_images=not args.no_images,
            write_json=not args.skip_json,
            save_rasters=args.save_rasters,
            image_max_width=image_max_width,
            write_diagnostics_outputs=not args.skip_diagnostics,
        )
        write_elapsed = time.perf_counter() - write_start
        performance.record_stage("total output writing", write_elapsed)
        total_elapsed = time.perf_counter() - total_start
        performance.record_stage("total run", total_elapsed)
        print(f"\nWrote outputs to: {args.output_dir}", flush=True)
        print(f"[progress] Output writing complete in {write_elapsed:.1f}s. Total elapsed: {total_elapsed:.1f}s.", flush=True)
        if args.preset in {"synthetic-earth", "real-earth-terrain"}:
            earth_files = [
                "earth_validation_report.txt",
                "earth_validation_checks.csv",
                "earth_latitude_bands.csv",
                "earth_biome_summary.csv",
                "earth_hydrology_summary.csv",
            ]
            if args.preset == "real-earth-terrain":
                earth_files.extend([
                    "earth_heightmap_quality.csv",
                    "earth_koppen_reference_agreement.csv",
                    "earth_koppen_reference_summary.csv",
                    "earth_koppen_reference_confusion.csv",
                    "earth_koppen_reference_map.png",
                    "earth_koppen_match_map.png",
                ])
            existing_earth_files = [name for name in earth_files if Path(args.output_dir, name).exists()]
            print("Wrote Earth calibration files to: " + ", ".join(f"{args.output_dir}/{name}" for name in existing_earth_files), flush=True)
        if not args.no_images:
            image_names = [
                "system_orbits.png",
                "system_sizes.png",
                "main_planet_moon.png",
                "main_planet_terrain.png",
                "main_planet_temperature.png",
                "main_planet_precipitation.png",
                "main_planet_koppen.png",
                "main_planet_hydrology.png",
                "main_planet_drainage_basins.png",
                "main_planet_biomes.png",
                "main_planet_regions.png",
                "main_planet_delta_mouths.png",
                "main_planet_tectonic_plates.png",
                "main_planet_plate_boundaries.png",
                "main_planet_crust_type.png",
                "main_planet_coastline_margin_types.png",
                "main_planet_inland_lakes.png",
                "main_planet_islands_archipelago.png",
                "main_planet_erosion_deposition.png",
                "main_planet_wind_currents.png",
                "main_planet_ocean_currents.png",
                "main_planet_moisture_transport.png",
                "main_planet_rain_shadow.png",
            ]
            existing_images = [str(Path(args.output_dir) / name) for name in image_names if (Path(args.output_dir) / name).exists()]
            if existing_images:
                print("Wrote visualizations to: " + ", ".join(existing_images), flush=True)
            else:
                print("No visualization PNGs were written.", flush=True)
            if image_max_width:
                print(f"Raster PNGs were downsampled to width <= {image_max_width}; simulation grids were not downsampled.", flush=True)
            print(f"Wrote diagnostics to: {args.output_dir}/world_quality_report.txt, {args.output_dir}/terrain_province_summary.csv, {args.output_dir}/climate_diagnostics.csv, {args.output_dir}/hydrology_diagnostics.csv, {args.output_dir}/landform_diagnostics.csv, {args.output_dir}/physical_scale_diagnostics.csv, {args.output_dir}/river_mouths.csv", flush=True)
            print(f"Wrote portable diagnostic bundle to: {args.output_dir}/worldgen_diagnostic_bundle.zip", flush=True)
            if args.save_rasters:
                print(f"Wrote binary rasters to: {args.output_dir}/main_planet_rasters.npz", flush=True)
        import os
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)


if __name__ == "__main__":
    main()
