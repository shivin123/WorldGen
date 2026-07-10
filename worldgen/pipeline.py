"""Staged WorldGen pipeline entry point.

This module is intentionally additive: the classic ``python -m worldgen.main``
command still performs a normal all-in-one run.  The pipeline command adds a
stateful workflow for long experiments:

    python -m worldgen.pipeline new --seed 143 --output-dir out --run-to solar-system
    # inspect/edit state/01_solar_system.json
    python -m worldgen.pipeline run-to terrain --output-dir out
    # inspect terrain maps/state, then continue
    python -m worldgen.pipeline run-from climate --output-dir out
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
import time
from pathlib import Path

from worldgen import performance
from worldgen.config import load_config, merge_cli_overrides
from worldgen.generators.planet_profile_generator import (
    assemble_main_planet_profile,
    build_planet_physics_review,
    generate_biome_stage,
    generate_climate_stage,
    generate_hydrology_stage,
    generate_main_planet_physical_states,
    generate_region_stage,
    generate_terrain_stage,
)
from worldgen.generators.system_generator import generate_star_system_shell
from worldgen.main import _cap_map_size, _confirm_large_map
from worldgen.pipeline_overrides import apply_stage_overrides
from worldgen.output.export import system_to_text, write_outputs
from worldgen.pipeline_state import (
    STAGE_ORDER,
    TERRAIN_SUBPHASES,
    assemble_available_system,
    ensure_layout,
    load_biome_state,
    load_climate_state,
    load_hydrology_state,
    load_physics_state,
    load_regions_state,
    load_system_state,
    load_terrain_state,
    mark_stage,
    normalize_stage,
    read_resolved_config,
    remove_stage_and_after,
    save_biome_state,
    save_climate_state,
    save_hydrology_state,
    save_physics_state,
    save_regions_state,
    save_system_state,
    save_terrain_state,
    save_all_terrain_subphase_checkpoints,
    stage_exists,
    stage_is_stale,
    stage_is_terrain_subphase,
    stage_index,
    stage_needs_rerun,
    stage_seed,
    status_rows,
    unapplied_override_reason,
    validate_staged_run,
    write_resolved_config,
    write_stage_graph,
    write_status_report,
)
from worldgen.random_utils import create_rng
from worldgen.terrain_review import build_terrain_review, write_terrain_review_outputs
from worldgen.visualization.system_plot import (
    save_main_planet_biome_view,
    save_main_planet_coastline_margin_types_view,
    save_main_planet_crust_type_view,
    save_main_planet_delta_mouths_view,
    save_main_planet_drainage_basins_view,
    save_main_planet_erosion_deposition_view,
    save_main_planet_hydrology_view,
    save_main_planet_inland_lakes_view,
    save_main_planet_islands_archipelago_view,
    save_main_planet_land_exactly_1m_view,
    save_main_planet_koppen_view,
    save_main_planet_moisture_transport_view,
    save_main_planet_moon_view,
    save_main_planet_ocean_currents_view,
    save_main_planet_plate_boundaries_view,
    save_main_planet_final_plate_boundaries_view,
    save_main_planet_boundary_history_density_view,
    save_main_planet_orogeny_history_view,
    save_main_planet_suture_history_view,
    save_main_planet_submerged_continental_crust_view,
    save_main_planet_continental_shelf_support_view,
    save_main_planet_shelf_depth_target_view,
    save_main_planet_shelf_zones_view,
    save_main_planet_lake_depth_limit_view,
    save_main_planet_final_plate_components_view,
    save_main_planet_ripple_artifact_risk_view,
    save_main_planet_v4_boundary_deformation_view,
    save_main_planet_v4_volcanic_island_support_view,
    save_main_planet_v4_rift_cut_support_view,
    save_main_planet_v4_mountain_branch_support_view,
    save_main_planet_v4_island_chain_view,
    save_main_planet_v4_boundary_network_view,
    save_main_planet_v4_orogen_network_view,
    save_main_planet_v4_control_response_view,
    save_main_planet_v4_elevation_delta_view,
    save_main_planet_v4_landform_change_view,
    save_main_planet_v4_plate_topology_view,
    save_main_planet_precipitation_view,
    save_main_planet_rain_shadow_view,
    save_main_planet_itcz_position_view,
    save_main_planet_itcz_nh_summer_view,
    save_main_planet_itcz_equinox_view,
    save_main_planet_itcz_nh_winter_view,
    save_main_planet_pressure_nh_summer_view,
    save_main_planet_pressure_equinox_view,
    save_main_planet_pressure_nh_winter_view,
    save_main_planet_thermal_equator_view,
    save_main_planet_thermal_equator_nh_summer_view,
    save_main_planet_thermal_equator_equinox_view,
    save_main_planet_thermal_equator_nh_winter_view,
    save_main_planet_pressure_belts_nh_summer_view,
    save_main_planet_pressure_belts_equinox_view,
    save_main_planet_pressure_belts_nh_winter_view,
    save_main_planet_moisture_wind_nh_summer_view,
    save_main_planet_moisture_wind_equinox_view,
    save_main_planet_moisture_wind_nh_winter_view,
    save_main_planet_storm_track_moisture_view,
    save_main_planet_storm_track_moisture_nh_summer_view,
    save_main_planet_storm_track_moisture_equinox_view,
    save_main_planet_storm_track_moisture_nh_winter_view,
    save_main_planet_circulation_zones_view,
    save_main_planet_circulation_zones_nh_summer_view,
    save_main_planet_circulation_zones_equinox_view,
    save_main_planet_circulation_zones_nh_winter_view,
    save_main_planet_ocean_gyres_view,
    save_main_planet_regions_view,
    save_main_planet_tectonic_plates_view,
    save_main_planet_temperature_view,
    save_main_planet_terrain_provinces_view,
    save_main_planet_terrain_region_maps,
    save_main_planet_terrain_view,
    save_main_planet_wind_currents_view,
    save_system_orbit_map,
    save_system_size_chart,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run WorldGen in inspectable/resumable stages.")
    sub = parser.add_subparsers(dest="command", required=True)

    new = sub.add_parser("new", help="Create a staged run directory and optionally run to a stage.")
    _add_generation_args(new)
    new.add_argument("--run-to", default="solar-system", help="Stage to run after creating the directory. Default: solar-system.")

    run_to = sub.add_parser("run-to", help="Run missing stages from the start through the requested stage.")
    run_to.add_argument("stage", help="Target stage, e.g. solar-system, terrain, climate, outputs.")
    _add_existing_run_args(run_to)

    run_from = sub.add_parser("run-from", help="Invalidate the chosen stage and rerun it plus downstream stages, or stop at --stop-at.")
    run_from.add_argument("stage", help="First stage to rerun, e.g. terrain or climate.")
    run_from.add_argument("--stop-at", default="outputs", help="Last stage to run after invalidating from the chosen stage. Default: outputs.")
    _add_existing_run_args(run_from)

    single = sub.add_parser("run-stage", help="Run exactly one stage after verifying prerequisites exist.")
    single.add_argument("stage", help="Stage to run.")
    _add_existing_run_args(single)

    maps = sub.add_parser("maps", help="Generate maps for all data currently available in the staged state.")
    _add_existing_run_args(maps)

    status = sub.add_parser("status", help="Show staged-run status and write diagnostics/pipeline_status.*.")
    status.add_argument("--output-dir", required=True)

    validate = sub.add_parser("validate", help="Validate editable staged state and write diagnostics/pipeline_validation_report.txt.")
    validate.add_argument("--output-dir", required=True)

    apply_overrides = sub.add_parser("apply-overrides", help="Apply config/stage_overrides.json to the staged config/state files.")
    apply_overrides.add_argument("--output-dir", required=True)
    apply_overrides.add_argument("--validate", action="store_true", help="Validate the run after applying overrides.")

    return parser


def _add_generation_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", help="Path to a JSON config file.")
    parser.add_argument("--seed", type=int, help="Random seed. Same seed produces the same staged system.")
    parser.add_argument("--planet-count", type=int, help="Override number of planets.")
    parser.add_argument("--star-class", choices=("G", "K", "G0V", "G2V", "G5V", "K0V", "K3V", "K5V", "K7V"), help="Broad generated star class/subclass hint for Stage 1.")
    parser.add_argument("--star-mass", type=float, help="Override generated star mass in solar masses.")
    parser.add_argument("--star-age", type=float, help="Override generated star age in Gyr.")
    parser.add_argument("--metallicity", type=float, help="Override generated star metallicity [Fe/H]-like value.")
    parser.add_argument(
        "--system-architecture",
        choices=("random", "compact_rocky_inner", "solar_like_mixed", "outer_giant_dominated", "low_mass_quiet", "volatile_rich", "sparse_old"),
        help="Stage 1 orbital/system architecture. Random leaves it to the generator.",
    )
    parser.add_argument(
        "--main-planet-preference",
        choices=("earthlike", "dry_terrestrial", "oceanic", "super_earth", "colder_world", "warmer_world"),
        help="Preference used when placing/scoring the habitable-zone candidate.",
    )
    parser.add_argument("--moon-strength", choices=("weak", "moderate", "strong"), help="Preferred broad major-moon tidal strength.")
    parser.add_argument("--no-major-moon", action="store_true", help="Do not require/generate a major moon for the Main Planet.")
    parser.add_argument("--output-dir", required=True, help="Staged run directory.")
    parser.add_argument("--map-width", type=int, help="Main Planet map width in grid cells/pixels.")
    parser.add_argument("--map-height", type=int, help="Main Planet map height in grid cells/pixels.")
    parser.add_argument("--preview", action="store_true", help="Cap map at 1024 x 512 for iteration.")
    parser.add_argument("--fast", action="store_true", help="Cap map at 2048 x 1024 and skip region analysis.")
    parser.add_argument("--skip-hydrology", action="store_true", help="Skip hydrology stage; downstream maps get placeholders.")
    parser.add_argument("--skip-biomes", action="store_true", help="Skip biome stage; downstream maps get placeholders.")
    parser.add_argument("--skip-regions", action="store_true", help="Skip region analysis.")
    parser.add_argument("--yes", action="store_true", help="Confirm very large terrain stages without an interactive prompt.")
    parser.add_argument("--skip-json", action="store_true", help="Skip compact system.json during final output stage.")
    parser.add_argument("--save-rasters", action="store_true", help="Write main_planet_rasters.npz during final output stage.")
    parser.add_argument("--skip-diagnostics", action="store_true", help="Skip diagnostic CSV/TXT outputs during final output stage.")
    parser.add_argument("--no-images", action="store_true", help="Skip stage map output.")
    parser.add_argument("--image-max-width", type=int, default=None, help="Downsample raster PNG outputs to this width.")
    parser.add_argument("--full-res-images", action="store_true", help="Do not auto-downsample large-map PNG outputs.")
    parser.add_argument(
        "--koppen-detail",
        choices=("regional", "local9", "local4", "cell"),
        default=None,
        help="Controls Köppen classification smoothing.",
    )
    parser.add_argument(
        "--climate-mode",
        choices=("seasonal_v5", "seasonal_v4", "seasonal_v3", "seasonal_v2", "seasonal_v1", "legacy"),
        default=None,
        help="Climate backend. seasonal_v5 adds component-based moisture/rainfall coupling on top of seasonal_v4; seasonal_v4 is the refined structured atmosphere + basin-ocean mode; seasonal_v3 preserves the first basin-ocean mode; seasonal_v2 preserves the structured atmosphere-only review mode; seasonal_v1 is the stable seasonal overhaul; legacy preserves the previous heuristic climate model for rollback/comparison.",
    )
    parser.add_argument(
        "--terrain-mode",
        choices=("procedural_legacy", "plate_tectonic_v1", "plate_history_v1", "plate_history_v2", "plate_history_v3", "plate_history_v4"),
        default=None,
        help="Terrain backend for generated worlds. procedural_legacy keeps the current generator; plate_tectonic_v1 preserves the Update 16 plate-owned terrain stack; plate_history_v1 runs a compact time-evolved plate-history model; plate_history_v2 is an experimental stronger structural reconstruction on top of v1; plate_history_v3 is the stable unified continuous-field model; plate_history_v4 is the recommended v4 terrain model built on stable v3 with conservative topology/island/rift improvements.",
    )
    parser.add_argument(
        "--tectonic-history-myr",
        type=float,
        default=None,
        help="For plate_history_v1/v2/v3/v4, simulate this many millions of years of plate history before deriving terrain.",
    )
    parser.add_argument(
        "--tectonic-timestep-myr",
        type=float,
        default=None,
        help="For plate_history_v1/v2/v3/v4, requested geological timestep in Myr. Internal sampling may be capped for speed.",
    )
    parser.add_argument(
        "--tectonic-grid-scale",
        choices=("legacy", "preview", "normal", "high", "native", "custom"),
        default=None,
        help="For plate_history_v1/v2/v3/v4, choose requested tectonic detail label. WorldGen now always uses the stable macro-history grid plus full-resolution tectonic detail unless developer-only raw-grid mode is explicitly unlocked.",
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
        help="Bias the Main Planet terrain run away from high-latitude/polar land and enforce polar-ocean cleanup in plate mode.",
    )
    parser.add_argument(
        "--erosion-deposition-strength",
        "--erosion-deposition-multiplier",
        dest="erosion_deposition_multiplier",
        type=float,
        default=None,
        help="For plate_history_v3/v4, multiplier for erosion/deposition conditioning. 1.0 is neutral, 1.35 is the current default, 0.0 is near-disabled, and 2.0+ is stress-test strong.",
    )
    parser.add_argument(
        "--continental-shelf-strength",
        type=float,
        default=None,
        help="For plate_history_v3/v4, multiplier for submerged continental shelf support. 1.65 is the current default; higher values widen/shallow passive continental shelves.",
    )
    parser.add_argument(
        "--shelf-width-factor",
        type=float,
        default=None,
        help="For plate_history_v3/v4, broadness factor for the submerged continental apron and shelf/slope transition. 0.9 is the current default; higher values make shelves extend farther offshore where continental support exists.",
    )
    parser.add_argument(
        "--v4-topology-strength",
        type=float,
        default=None,
        help="For plate_history_v4 only, strength of experimental boundary deformation, microplate/sliver, and mountain-branching behavior. 1.0 is the conservative default; 1.3+ is stronger topology.",
    )
    parser.add_argument(
        "--v4-island-strength",
        type=float,
        default=None,
        help="For plate_history_v4 only, strength of physically supported volcanic island-chain uplift. 1.0 is the conservative default; 1.3+ creates more volcanic archipelagos where support exists.",
    )
    parser.add_argument(
        "--v4-rift-strength",
        type=float,
        default=None,
        help="For plate_history_v4 only, strength of experimental rift-cut corridors, narrow seas, and lake-prone extensional basins. 1.0 is the conservative default; 1.2+ makes rifts/gulfs stronger.",
    )


def _add_existing_run_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--yes", action="store_true", help="Confirm very large terrain stages without an interactive prompt.")
    parser.add_argument("--no-images", action="store_true", help="Skip stage map output.")
    parser.add_argument("--image-max-width", type=int, default=None, help="Downsample raster PNG outputs to this width.")
    parser.add_argument("--full-res-images", action="store_true", help="Do not auto-downsample large-map PNG outputs.")
    parser.add_argument("--skip-json", action="store_true", help="Skip compact system.json during final output stage.")
    parser.add_argument("--save-rasters", action="store_true", help="Write main_planet_rasters.npz during final output stage.")
    parser.add_argument("--skip-diagnostics", action="store_true", help="Skip diagnostic CSV/TXT outputs during final output stage.")


def _resolved_config_from_new_args(args: argparse.Namespace):
    config = load_config(args.config)
    if args.seed is None and config.seed is None:
        generated_seed = random.randint(1, 999_999_999)
    else:
        generated_seed = args.seed if args.seed is not None else config.seed

    map_width = args.map_width
    map_height = args.map_height
    if args.preview:
        map_width, map_height, _ = _cap_map_size(map_width or 1024, map_height or 512, 1024, 512)
    elif args.fast:
        map_width, map_height, _ = _cap_map_size(map_width or 2048, map_height or 1024, 2048, 1024)

    star_config = config.star
    if any(getattr(args, key, None) is not None for key in ("star_class", "star_mass", "star_age", "metallicity")):
        from worldgen.config import StarConfig
        star_config = StarConfig(
            stellar_class=args.star_class if args.star_class is not None else config.star.stellar_class,
            mass_solar=args.star_mass if args.star_mass is not None else config.star.mass_solar,
            age_gyr=args.star_age if args.star_age is not None else config.star.age_gyr,
            metallicity=args.metallicity if args.metallicity is not None else config.star.metallicity,
        )
        config = type(config)(seed=config.seed, star=star_config, system=config.system, planet_profile=config.planet_profile)

    return merge_cli_overrides(
        config,
        {
            "seed": generated_seed,
            "planet_count": args.planet_count,
            "architecture_type": None if args.system_architecture == "random" else args.system_architecture,
            "main_planet_preference": args.main_planet_preference,
            "require_major_moon": False if args.no_major_moon else config.system.require_major_moon,
            "moon_strength_preference": args.moon_strength,
            "map_width": map_width,
            "map_height": map_height,
            "generate_hydrology": not args.skip_hydrology,
            "generate_biomes": not args.skip_biomes,
            "generate_regions": not (args.skip_regions or args.fast),
            "fast_mode": args.fast or args.preview,
            "no_accelerated_terrain": True,
            "no_accelerated_climate": True,
            "koppen_detail": args.koppen_detail,
            "climate_generation_mode": args.climate_mode,
            "terrain_generation_mode": args.terrain_mode,
            "suppress_polar_land": bool(args.suppress_polar_land),
            "tectonic_history_myr": args.tectonic_history_myr,
            "tectonic_timestep_myr": args.tectonic_timestep_myr,
            "tectonic_grid_scale": args.tectonic_grid_scale,
            "tectonic_grid_policy": ("raw" if getattr(args, "allow_experimental_tectonic_grid", False) and args.tectonic_grid_policy == "raw" else ("stable" if args.tectonic_grid_policy == "raw" else args.tectonic_grid_policy)),
            "allow_experimental_tectonic_grid": bool(getattr(args, "allow_experimental_tectonic_grid", False)),
            "tectonic_grid_width": args.tectonic_grid_width,
            "tectonic_grid_height": args.tectonic_grid_height,
        },
    )


def _write_stage2_report(output_dir: str | Path, star, main, rotation, atmosphere, hydrosphere, geology) -> None:
    path = ensure_layout(output_dir) / "state" / "02_planet_physics.json"
    if not path.exists():
        return
    data = json.loads(path.read_text(encoding="utf-8"))
    data["review"] = build_planet_physics_review(star, main, rotation, atmosphere, hydrosphere, geology)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _write_stage1_report(output_dir: str | Path, system) -> None:
    """Write a plain-text Stage 1 review report for the Web UI and file browser."""
    output_dir = Path(output_dir)
    diagnostics = system.diagnostics or {}
    lines: list[str] = ["Stage 1 Solar System Report", "===========================", ""]
    for line in diagnostics.get("system_report", []) or []:
        lines.append(str(line))
    explanation = diagnostics.get("habitability_explanation") or {}
    if isinstance(explanation, dict):
        lines.extend(["", "Habitability explanation", "------------------------"])
        if explanation.get("summary"):
            lines.append(str(explanation["summary"]))
        positives = explanation.get("positive_factors") or []
        concerns = explanation.get("concerns") or []
        if positives:
            lines.extend(["", "Positive factors:"])
            lines.extend(f"- {item}" for item in positives)
        if concerns:
            lines.extend(["", "Concerns:"])
            lines.extend(f"- {item}" for item in concerns)
    warnings = diagnostics.get("stage1_warnings") or []
    if warnings:
        lines.extend(["", "Review warnings / notices", "-------------------------"])
        for item in warnings:
            if isinstance(item, dict):
                lines.append(f"- {item.get('level', 'notice')}: {item.get('message', '')}")
    candidates = diagnostics.get("main_planet_candidates") or []
    if candidates:
        lines.extend(["", "Main Planet candidates", "----------------------"])
        for item in candidates:
            if isinstance(item, dict):
                lines.append(
                    f"- {item.get('name')}: {item.get('status')} | score {item.get('habitability_score')} | "
                    f"{item.get('selected_or_rejected_reason', '')}"
                )
    path = output_dir / "diagnostics" / "stage1_solar_system_report.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _attach_and_write_terrain_review(output_dir: str | Path, terrain, main, hydrosphere, geology, config) -> None:
    """Attach Stage 3 review metadata and write diagnostic folders/maps."""
    try:
        terrain.terrain_diagnostics = build_terrain_review(
            terrain,
            main,
            hydrosphere,
            geology,
            config.planet_profile,
            output_dir=output_dir,
        )
        write_terrain_review_outputs(output_dir, terrain)
    except Exception as exc:
        print(f"[warning] Could not write Stage 3 terrain review diagnostics: {exc}", flush=True)


def _configure_image_cap(output_dir: str | Path, width: int, height: int, args: argparse.Namespace) -> int | None:
    image_max_width = getattr(args, "image_max_width", None)
    if image_max_width is None and not getattr(args, "full_res_images", False) and width * height > 4_500_000:
        image_max_width = 2048
        print("[progress] Large map detected; raster PNGs will be downsampled to width 2048. Use --full-res-images to disable.", flush=True)
    return image_max_width


def _tectonic_grid_request(width: int, height: int, config) -> dict[str, object]:
    """Return a lightweight summary of how the plate-history grid will run."""
    profile = config.planet_profile
    mode = str(getattr(profile, "terrain_generation_mode", "") or "")
    scale = str(getattr(profile, "tectonic_grid_scale", "legacy") or "legacy").strip().lower()
    policy = str(getattr(profile, "tectonic_grid_policy", "stable") or "stable").strip().lower()
    allow_raw = bool(getattr(profile, "allow_experimental_tectonic_grid", False))
    if scale not in {"legacy", "preview", "normal", "high", "native", "custom"}:
        scale = "legacy"
    if policy not in {"stable", "raw"}:
        policy = "stable"
    if policy == "raw" and not allow_raw:
        policy = "stable"
    stable_w = int(max(192, min(512, round(width / 8))))
    if stable_w % 2:
        stable_w += 1
    stable_h = max(64, stable_w // 2)
    divisor = {"preview": 8.0, "normal": 4.0, "high": 2.0, "native": 1.0, "custom": 4.0}.get(scale, 8.0)
    if scale == "custom" and getattr(profile, "tectonic_grid_width", None):
        req_w = int(getattr(profile, "tectonic_grid_width"))
        req_h = int(getattr(profile, "tectonic_grid_height", 0) or round(req_w * height / max(width, 1)))
    elif scale == "legacy":
        req_w, req_h = stable_w, stable_h
    else:
        req_w = int(round(width / divisor))
        req_h = int(round(height / divisor))
    req_w, req_h = max(1, req_w), max(1, req_h)
    actual_w, actual_h = (req_w, req_h) if policy == "raw" and scale != "legacy" else (stable_w, stable_h)
    return {
        "terrain_mode": mode,
        "scale": scale,
        "policy": policy,
        "stable_w": stable_w,
        "stable_h": stable_h,
        "requested_w": req_w,
        "requested_h": req_h,
        "actual_w": actual_w,
        "actual_h": actual_h,
        "requested_cells": req_w * req_h,
        "actual_cells": actual_w * actual_h,
        "uses_stable_hybrid": bool(policy != "raw" and scale != "legacy"),
        "raw_allowed": bool(allow_raw),
    }


def _print_tectonic_grid_notice(config) -> None:
    profile = config.planet_profile
    mode = str(getattr(profile, "terrain_generation_mode", "") or "")
    if mode not in {"plate_history_v1", "plate_history_v2", "plate_history_v3", "plate_history_v4"}:
        return
    width = int(getattr(profile, "map_width", 0) or 0)
    height = int(getattr(profile, "map_height", 0) or 0)
    info = _tectonic_grid_request(width, height, config)
    scale = str(info["scale"])
    policy = str(info["policy"])
    if scale == "legacy":
        print(f"[progress] Tectonic history grid: legacy/stable macro grid {info['actual_w']} x {info['actual_h']}.", flush=True)
        return
    raw_requested = str(getattr(profile, "tectonic_grid_policy", "stable") or "stable").strip().lower() == "raw"
    if raw_requested and not bool(info.get("raw_allowed", False)):
        print(
            "[warning] Raw/high tectonic-history grids are disabled for normal runs because they currently produce poorer terrain. "
            "Using the stable macro-history grid plus full-resolution tectonic detail instead.",
            flush=True,
        )
    if policy == "raw":
        print(
            f"[warning] Developer raw tectonic-history grid enabled: {scale!r} will run {info['actual_w']} x {info['actual_h']} "
            f"({info['actual_cells']:,} cells). This is an experimental research path and may produce worse terrain.",
            flush=True,
        )
        return
    print(
        f"[warning] Tectonic grid scale {scale!r} requested for a {width} x {height} map. "
        f"For stability, WorldGen will use the macro history grid {info['actual_w']} x {info['actual_h']} "
        f"instead of the raw requested {info['requested_w']} x {info['requested_h']} ({info['requested_cells']:,} cells), then apply full-resolution tectonic detail.",
        flush=True,
    )
    print(
        "[progress] Higher-resolution raw tectonic-history grids are deferred terrain research; see docs/TERRAIN_MODEL_BACKLOG.md.",
        flush=True,
    )


def _stage_rng_seed(output_dir: str | Path, config, stage: str) -> int:
    base = stage_seed(config.seed, stage)
    offsets_path = Path(output_dir) / "config" / "stage_seed_offsets.json"
    if offsets_path.exists():
        try:
            offsets = json.loads(offsets_path.read_text(encoding="utf-8"))
            if isinstance(offsets, dict):
                base += int(offsets.get(normalize_stage(stage), 0) or 0)
        except Exception:
            pass
    return base


def _run_stage(output_dir: str | Path, stage: str, args: argparse.Namespace, force: bool = False) -> None:
    stage = normalize_stage(stage)
    output_dir = ensure_layout(output_dir)
    config = read_resolved_config(output_dir)
    if stage_exists(output_dir, stage) and not force and not stage_is_stale(output_dir, stage):
        print(f"[progress] Stage '{stage}' already complete; skipping.", flush=True)
        return
    if stage_exists(output_dir, stage) and not force and stage_is_stale(output_dir, stage):
        print(f"[progress] Stage '{stage}' is stale; rerunning.", flush=True)

    start = time.perf_counter()
    print(f"[progress] Running stage: {stage}", flush=True)

    if stage == "solar-system":
        rng = create_rng(stage_seed(config.seed, "solar-system"))
        system = generate_star_system_shell(rng, config, include_moon=True)
        save_system_state(output_dir, system)
        (output_dir / "system_summary.txt").write_text(system_to_text(system), encoding="utf-8")
        _write_stage1_report(output_dir, system)
        if not getattr(args, "no_images", False):
            _write_maps_available(output_dir, args, through_stage="solar-system")
        elapsed = time.perf_counter() - start
        mark_stage(output_dir, stage, "complete", elapsed, "Generated star, orbits, planet list, Main Planet selection, and moon.")
        return

    if stage == "planet-physics":
        _require(output_dir, "solar-system")
        system = load_system_state(output_dir)
        main = system.main_planet
        if main is None:
            raise RuntimeError("No Main Planet selected in solar-system state.")
        rng = create_rng(_stage_rng_seed(output_dir, config, "planet-physics"))
        rotation, atmosphere, hydrosphere, geology = generate_main_planet_physical_states(rng, system.star, main, config.planet_profile)
        save_physics_state(output_dir, rotation, atmosphere, hydrosphere, geology)
        # Apply any saved locks/visual-editor overrides after regeneration, then
        # rebuild the review metadata from the final editable state.
        apply_stage_overrides(output_dir)
        rotation, atmosphere, hydrosphere, geology = load_physics_state(output_dir)
        _write_stage2_report(output_dir, system.star, main, rotation, atmosphere, hydrosphere, geology)
        elapsed = time.perf_counter() - start
        mark_stage(output_dir, stage, "complete", elapsed, "Generated editable rotation/atmosphere/hydrosphere/geology state and Stage 2 review metadata.")
        return

    if stage_is_terrain_subphase(stage):
        _require(output_dir, "solar-system")
        _require(output_dir, "planet-physics")
        # Apply saved visual-editor/raw overrides before terrain reads Stage 2
        # physics. Without this, changing Ocean fraction target then rerunning
        # from Stage 3 could reuse the old hydrosphere state and produce the
        # same terrain despite the edited target.
        override_result = apply_stage_overrides(output_dir)
        if override_result.changed:
            print("[progress] Applied staged overrides before terrain synthesis:", flush=True)
            for change in override_result.changes[:12]:
                print(f"[progress]   - {change}", flush=True)
            if len(override_result.changes) > 12:
                print(f"[progress]   - ... {len(override_result.changes) - 12} more changes", flush=True)
            config = read_resolved_config(output_dir)
        width = config.planet_profile.map_width
        height = config.planet_profile.map_height
        _print_tectonic_grid_notice(config)
        _confirm_large_map(width, height, args)

        # Update65 exposes terrain sub-phases as real review/diagnostic targets. The
        # existing terrain synthesizer is still internally monolithic, so the
        # first requested terrain sub-stage generates the shared terrain raster
        # once, then writes marker checkpoints for every terrain sub-stage.
        # Future terrain-overhaul updates can replace this wrapper with true
        # incremental generation without changing command names.
        if stage_exists(output_dir, "terrain-finalization-recentering") and not force and not stage_needs_rerun(output_dir, "terrain-finalization-recentering"):
            terrain = load_terrain_state(output_dir)
            save_all_terrain_subphase_checkpoints(output_dir, terrain)
            try:
                write_terrain_review_outputs(output_dir, terrain)
            except Exception as exc:
                print(f"[warning] Could not refresh terrain diagnostics: {exc}", flush=True)
            if not getattr(args, "no_images", False):
                _write_maps_available(output_dir, args, through_stage=stage)
            elapsed = time.perf_counter() - start
            mark_stage(output_dir, stage, "complete", elapsed, "Refreshed terrain sub-stage marker from existing terrain checkpoint.")
            return

        system = load_system_state(output_dir)
        main = system.main_planet
        if main is None:
            raise RuntimeError("No Main Planet selected in solar-system state.")
        _rotation, _atmosphere, hydrosphere, geology = load_physics_state(output_dir)
        rng = create_rng(stage_seed(config.seed, "terrain"))
        mode = getattr(config.planet_profile, "terrain_generation_mode", "procedural_legacy")
        print(f"[progress] Running terrain/geology generator mode: {mode}.", flush=True)
        if mode == "plate_tectonic_v1":
            print("[progress] Plate Terrain 11/16 stack is active: native plate setup, topology repair, continent assemblies, motion/boundaries, ocean-floor fields, tectonic landform belts, margin-profile coasts/shelves/islands, drainage-ready valleys/basins, and final plate-mode QA are generated. Legacy terrain core is not run.", flush=True)
        elif mode == "plate_history_v1":
            print("[progress] Plate History v1 is active: compact kinematic plate history is simulated over geological time, then terrain is derived from accumulated crust fields. Legacy and plate_tectonic_v1 terrain cores are not run.", flush=True)
        elif mode == "plate_history_v2":
            print("[progress] Plate History v2 is active: stable v1 macro history plus stronger structural crust, visible ridge spines, bead volcanic arcs, coast migration, and erosion/deposition reconstruction. Legacy and plate_tectonic_v1 terrain cores are not run.", flush=True)
        elif mode == "plate_history_v3":
            print("[progress] Plate History v3 is active: isolated unified continuous-field tectonic reconstruction with deformable diagnostics, isostasy, bathymetry, lakes, age-aware erosion, and readable moving-stage diagnostics. Legacy, plate_tectonic_v1, and v2 final terrain cores are not used as final authority.", flush=True)
        else:
            print("[progress] Running monolithic legacy terrain/geology generator for requested terrain sub-stage.", flush=True)
        print("[progress] Terrain review checkpoints and diagnostics will be written for these sub-stages:", flush=True)
        for sub in TERRAIN_SUBPHASES:
            print(f"[progress]   - {sub}", flush=True)
        terrain = generate_terrain_stage(rng, main, hydrosphere, geology, config.planet_profile, output_dir=str(output_dir))
        terrain_elapsed = time.perf_counter() - start
        _attach_and_write_terrain_review(output_dir, terrain, main, hydrosphere, geology, config)
        # Save updated hydrosphere actual ocean fraction after terrain thresholding
        # before terrain checkpoints are written, so dependency timestamp checks
        # do not immediately mark the freshly generated terrain as stale.
        rotation, atmosphere, _old_hydro, geology = load_physics_state(output_dir)
        save_physics_state(output_dir, rotation, atmosphere, hydrosphere, geology)
        save_terrain_state(output_dir, terrain)
        save_all_terrain_subphase_checkpoints(output_dir, terrain, elapsed_s=terrain_elapsed)
        if not getattr(args, "no_images", False):
            _write_maps_available(output_dir, args, through_stage=stage)
        elapsed = time.perf_counter() - start
        for sub in TERRAIN_SUBPHASES:
            mark_stage(output_dir, sub, "complete", terrain_elapsed, "Shared monolithic terrain pass total; exact per-sub-stage timing awaits internal terrain refactor.")
        return

    if stage == "climate":
        _require(output_dir, "terrain")
        rotation, atmosphere, _hydro, _geology = load_physics_state(output_dir)
        terrain = load_terrain_state(output_dir)
        climate = generate_climate_stage(rotation, atmosphere, terrain, config.planet_profile)
        save_climate_state(output_dir, climate)
        if not getattr(args, "no_images", False):
            _write_maps_available(output_dir, args, through_stage="climate")
        elapsed = time.perf_counter() - start
        mark_stage(output_dir, stage, "complete", elapsed, "Generated climate checkpoint.")
        return

    if stage == "hydrology":
        _require(output_dir, "climate")
        terrain = load_terrain_state(output_dir)
        climate = load_climate_state(output_dir)
        hydrology = generate_hydrology_stage(terrain, climate, config.planet_profile)
        save_hydrology_state(output_dir, hydrology)
        if not getattr(args, "no_images", False):
            _write_maps_available(output_dir, args, through_stage="hydrology")
        elapsed = time.perf_counter() - start
        mark_stage(output_dir, stage, "complete", elapsed, "Generated hydrology checkpoint.")
        return

    if stage == "biomes":
        _require(output_dir, "hydrology")
        terrain = load_terrain_state(output_dir)
        climate = load_climate_state(output_dir)
        hydrology = load_hydrology_state(output_dir)
        biomes = generate_biome_stage(terrain, climate, hydrology, config.planet_profile)
        save_biome_state(output_dir, biomes)
        if not getattr(args, "no_images", False):
            _write_maps_available(output_dir, args, through_stage="biomes")
        elapsed = time.perf_counter() - start
        mark_stage(output_dir, stage, "complete", elapsed, "Generated biome checkpoint.")
        return

    if stage == "regions":
        _require(output_dir, "biomes")
        terrain = load_terrain_state(output_dir)
        climate = load_climate_state(output_dir)
        hydrology = load_hydrology_state(output_dir)
        biomes = load_biome_state(output_dir)
        regions = generate_region_stage(terrain, climate, hydrology, biomes, config.planet_profile)
        save_regions_state(output_dir, regions)
        if not getattr(args, "no_images", False):
            _write_maps_available(output_dir, args, through_stage="regions")
        elapsed = time.perf_counter() - start
        mark_stage(output_dir, stage, "complete", elapsed, "Generated regional summaries checkpoint.")
        return

    if stage == "outputs":
        _require(output_dir, "regions")
        system = assemble_available_system(output_dir)
        print(system_to_text(system))
        width = system.main_planet_profile.terrain.width if system.main_planet_profile else config.planet_profile.map_width
        height = system.main_planet_profile.terrain.height if system.main_planet_profile else config.planet_profile.map_height
        image_max_width = _configure_image_cap(output_dir, width, height, args)
        write_outputs(
            system,
            output_dir,
            include_images=not getattr(args, "no_images", False),
            write_json=not getattr(args, "skip_json", False),
            save_rasters=getattr(args, "save_rasters", False),
            image_max_width=image_max_width,
            write_diagnostics_outputs=not getattr(args, "skip_diagnostics", False),
        )
        elapsed = time.perf_counter() - start
        mark_stage(output_dir, stage, "complete", elapsed, "Wrote final summaries, maps, diagnostics, and diagnostic bundle.")
        return

    raise AssertionError(stage)


def _require(output_dir: str | Path, stage: str) -> None:
    if not stage_exists(output_dir, stage):
        raise RuntimeError(f"Required stage '{stage}' is missing. Run: python -m worldgen.pipeline run-to {stage} --output-dir {output_dir}")


def _run_to(output_dir: str | Path, target_stage: str, args: argparse.Namespace, force_from: str | None = None) -> None:
    target = normalize_stage(target_stage)
    start_i = 0
    if force_from is not None:
        start_i = stage_index(force_from)
        remove_stage_and_after(output_dir, force_from)
    for stage in STAGE_ORDER[: stage_index(target) + 1]:
        force = force_from is not None and stage_index(stage) >= start_i
        if force_from is None and stage_exists(output_dir, stage) and stage_is_stale(output_dir, stage):
            print(f"[progress] Stage '{stage}' is stale; clearing it and downstream stages before continuing.", flush=True)
            remove_stage_and_after(output_dir, stage)
        _run_stage(output_dir, stage, args, force=force)
    # Ensure the requested stopping point has its available maps when the stage
    # was already complete and no stage writer refreshed images.  Do not rewrite
    # the full map set after every run-to; that made logs noisy and repeatedly
    # regenerated the same PNGs during long runs.
    if not getattr(args, "no_images", False) and not _stage_visual_marker_exists(output_dir, target):
        _write_maps_available(output_dir, args, through_stage=target)



def _stage_visual_marker_exists(output_dir: str | Path, stage: str) -> bool:
    output_dir = Path(output_dir)
    stage = normalize_stage(stage)
    marker_by_stage = {
        "solar-system": "system_orbits.png",
        "planet-physics": "system_orbits.png",
        "terrain-foundation-mask": "main_planet_terrain.png",
        "terrain-tectonic-provinces": "main_planet_terrain_provinces.png",
        "terrain-crust-and-boundaries": "main_planet_crust_type.png",
        "terrain-mountains-basins-rifts": "main_planet_terrain.png",
        "terrain-coasts-shelves-islands": "main_planet_coastline_margin_types.png",
        "terrain-erosion-deposition": "main_planet_erosion_deposition.png",
        "terrain-finalization-recentering": "main_planet_terrain.png",
        "climate": "main_planet_koppen.png",
        "hydrology": "main_planet_hydrology.png",
        "biomes": "main_planet_biomes.png",
        "regions": "main_planet_regions.png",
        "outputs": "worldgen_diagnostic_bundle.zip",
    }
    name = marker_by_stage.get(stage, "system_orbits.png")
    path = output_dir / name
    if not path.exists():
        return False
    if path.suffix.lower() == ".png" and not path.with_suffix(".legend.json").exists() and name not in {"system_orbits.png", "system_sizes.png", "main_planet_moon.png"}:
        return False
    return True

def _write_maps_available(output_dir: str | Path, args: argparse.Namespace, through_stage: str | None = None) -> None:
    """Write maps for whichever staged data exists.

    Maps are written once to their canonical root output names with matching
    .legend.json sidecars.  Older builds also copied PNGs into maps/ without
    metadata, which created duplicate map cards.  The Web UI now groups maps by
    recursive discovery/metadata instead of duplicated files.
    """
    output_dir = Path(output_dir)
    system = assemble_available_system(output_dir)
    image_max_width = getattr(args, "image_max_width", None)
    if image_max_width is not None and image_max_width > 0:
        import os

        os.environ["WORLDGEN_IMAGE_MAX_WIDTH"] = str(image_max_width)

    def save(func, name: str) -> None:
        print(f"[progress] Writing available map {name}...", flush=True)
        func(system, output_dir / name)
        # Remove stale duplicate copies created by older builds.  The canonical
        # image plus its .legend.json sidecar stay at the root/diagnostic path.
        for stale in (output_dir / "maps" / name, (output_dir / "maps" / name).with_suffix(".legend.json")):
            try:
                if stale.exists():
                    stale.unlink()
            except Exception:
                pass

    save(save_system_orbit_map, "system_orbits.png")
    save(save_system_size_chart, "system_sizes.png")
    if system.main_planet is not None and system.main_planet.moon is not None:
        save(save_main_planet_moon_view, "main_planet_moon.png")

    if system.main_planet_profile is None:
        return

    stage_i = stage_index(through_stage or "outputs") if through_stage else stage_index("outputs")
    if stage_i >= stage_index(TERRAIN_SUBPHASES[0]) and stage_exists(output_dir, "terrain"):
        terrain_maps: list[tuple[object, str]] = []

        def add_terrain_maps(entries):
            for entry in entries:
                if entry not in terrain_maps:
                    terrain_maps.append(entry)

        add_terrain_maps([(save_main_planet_terrain_view, "main_planet_terrain.png")])
        if stage_i >= stage_index("terrain-tectonic-provinces"):
            add_terrain_maps([
                (save_main_planet_terrain_provinces_view, "main_planet_terrain_provinces.png"),
                (save_main_planet_tectonic_plates_view, "main_planet_tectonic_plates.png"),
                (save_main_planet_plate_boundaries_view, "main_planet_plate_boundaries.png"),
                (save_main_planet_final_plate_boundaries_view, "main_planet_final_plate_boundaries.png"),
            ])
        if stage_i >= stage_index("terrain-crust-and-boundaries"):
            add_terrain_maps([
                (save_main_planet_crust_type_view, "main_planet_crust_type.png"),
                (save_main_planet_plate_boundaries_view, "main_planet_plate_boundaries.png"),
                (save_main_planet_final_plate_boundaries_view, "main_planet_final_plate_boundaries.png"),
                (save_main_planet_boundary_history_density_view, "main_planet_boundary_history_density.png"),
                (save_main_planet_orogeny_history_view, "main_planet_orogeny_history.png"),
                (save_main_planet_suture_history_view, "main_planet_suture_history.png"),
                (save_main_planet_submerged_continental_crust_view, "main_planet_submerged_continental_crust.png"),
                (save_main_planet_continental_shelf_support_view, "main_planet_continental_shelf_support.png"),
                (save_main_planet_shelf_depth_target_view, "main_planet_shelf_depth_target.png"),
                (save_main_planet_shelf_zones_view, "main_planet_shelf_zones.png"),
                (save_main_planet_final_plate_components_view, "main_planet_final_plate_components.png"),
                (save_main_planet_ripple_artifact_risk_view, "main_planet_ripple_artifact_risk.png"),
                (save_main_planet_v4_boundary_deformation_view, "main_planet_v4_boundary_deformation.png"),
                (save_main_planet_v4_volcanic_island_support_view, "main_planet_v4_volcanic_island_support.png"),
                (save_main_planet_v4_rift_cut_support_view, "main_planet_v4_rift_cut_support.png"),
                (save_main_planet_v4_mountain_branch_support_view, "main_planet_v4_mountain_branch_support.png"),
                (save_main_planet_v4_island_chain_view, "main_planet_v4_island_chains.png"),
                (save_main_planet_v4_boundary_network_view, "main_planet_v4_boundary_network.png"),
                (save_main_planet_v4_orogen_network_view, "main_planet_v4_orogen_network.png"),
                (save_main_planet_v4_control_response_view, "main_planet_v4_control_response.png"),
                (save_main_planet_v4_elevation_delta_view, "main_planet_v4_elevation_delta.png"),
                (save_main_planet_v4_landform_change_view, "main_planet_v4_landform_change.png"),
                (save_main_planet_v4_plate_topology_view, "main_planet_v4_plate_topology.png"),
            ])
        if stage_i >= stage_index("terrain-mountains-basins-rifts"):
            add_terrain_maps([
                (save_main_planet_terrain_provinces_view, "main_planet_terrain_provinces.png"),
                (save_main_planet_terrain_view, "main_planet_terrain.png"),
            ])
        if stage_i >= stage_index("terrain-coasts-shelves-islands"):
            add_terrain_maps([
                (save_main_planet_coastline_margin_types_view, "main_planet_coastline_margin_types.png"),
                (save_main_planet_submerged_continental_crust_view, "main_planet_submerged_continental_crust.png"),
                (save_main_planet_continental_shelf_support_view, "main_planet_continental_shelf_support.png"),
                (save_main_planet_shelf_depth_target_view, "main_planet_shelf_depth_target.png"),
                (save_main_planet_shelf_zones_view, "main_planet_shelf_zones.png"),
                (save_main_planet_lake_depth_limit_view, "main_planet_lake_depth_limit.png"),
                (save_main_planet_inland_lakes_view, "main_planet_inland_lakes.png"),
                (save_main_planet_islands_archipelago_view, "main_planet_islands_archipelago.png"),
                (save_main_planet_land_exactly_1m_view, "main_planet_land_exactly_1m.png"),
            ])
        if stage_i >= stage_index("terrain-erosion-deposition"):
            add_terrain_maps([
                (save_main_planet_erosion_deposition_view, "main_planet_erosion_deposition.png"),
                (save_main_planet_ripple_artifact_risk_view, "main_planet_ripple_artifact_risk.png"),
                (save_main_planet_v4_boundary_deformation_view, "main_planet_v4_boundary_deformation.png"),
                (save_main_planet_v4_volcanic_island_support_view, "main_planet_v4_volcanic_island_support.png"),
                (save_main_planet_v4_rift_cut_support_view, "main_planet_v4_rift_cut_support.png"),
                (save_main_planet_v4_mountain_branch_support_view, "main_planet_v4_mountain_branch_support.png"),
                (save_main_planet_v4_island_chain_view, "main_planet_v4_island_chains.png"),
                (save_main_planet_v4_boundary_network_view, "main_planet_v4_boundary_network.png"),
                (save_main_planet_v4_orogen_network_view, "main_planet_v4_orogen_network.png"),
                (save_main_planet_v4_control_response_view, "main_planet_v4_control_response.png"),
                (save_main_planet_v4_elevation_delta_view, "main_planet_v4_elevation_delta.png"),
                (save_main_planet_v4_landform_change_view, "main_planet_v4_landform_change.png"),
                (save_main_planet_v4_plate_topology_view, "main_planet_v4_plate_topology.png"),
            ])

        for func, name in terrain_maps:
            try:
                save(func, name)
            except Exception as exc:
                print(f"[warning] Could not write {name}: {exc}", flush=True)
        if stage_i >= stage_index("terrain-finalization-recentering"):
            try:
                save_main_planet_terrain_region_maps(system, output_dir / "terrain_regions")
            except Exception as exc:
                print(f"[warning] Could not write terrain regional maps: {exc}", flush=True)

    if stage_i >= stage_index("climate") and stage_exists(output_dir, "climate"):
        for func, name in [
            (save_main_planet_temperature_view, "main_planet_temperature.png"),
            (save_main_planet_precipitation_view, "main_planet_precipitation.png"),
            (save_main_planet_koppen_view, "main_planet_koppen.png"),
            (save_main_planet_wind_currents_view, "main_planet_wind_currents.png"),
            (save_main_planet_ocean_currents_view, "main_planet_ocean_currents.png"),
            (save_main_planet_moisture_transport_view, "main_planet_moisture_transport.png"),
            (save_main_planet_rain_shadow_view, "main_planet_rain_shadow.png"),
            (save_main_planet_itcz_position_view, "main_planet_itcz_position.png"),
            (save_main_planet_itcz_nh_summer_view, "main_planet_itcz_nh_summer.png"),
            (save_main_planet_itcz_equinox_view, "main_planet_itcz_equinox.png"),
            (save_main_planet_itcz_nh_winter_view, "main_planet_itcz_nh_winter.png"),
            (save_main_planet_pressure_nh_summer_view, "main_planet_pressure_nh_summer.png"),
            (save_main_planet_pressure_equinox_view, "main_planet_pressure_equinox.png"),
            (save_main_planet_pressure_nh_winter_view, "main_planet_pressure_nh_winter.png"),
            (save_main_planet_thermal_equator_view, "main_planet_thermal_equator.png"),
            (save_main_planet_thermal_equator_nh_summer_view, "main_planet_thermal_equator_nh_summer.png"),
            (save_main_planet_thermal_equator_equinox_view, "main_planet_thermal_equator_equinox.png"),
            (save_main_planet_thermal_equator_nh_winter_view, "main_planet_thermal_equator_nh_winter.png"),
            (save_main_planet_pressure_belts_nh_summer_view, "main_planet_pressure_belts_nh_summer.png"),
            (save_main_planet_pressure_belts_equinox_view, "main_planet_pressure_belts_equinox.png"),
            (save_main_planet_pressure_belts_nh_winter_view, "main_planet_pressure_belts_nh_winter.png"),
            (save_main_planet_moisture_wind_nh_summer_view, "main_planet_moisture_wind_nh_summer.png"),
            (save_main_planet_moisture_wind_equinox_view, "main_planet_moisture_wind_equinox.png"),
            (save_main_planet_moisture_wind_nh_winter_view, "main_planet_moisture_wind_nh_winter.png"),
            (save_main_planet_storm_track_moisture_view, "main_planet_storm_track_moisture.png"),
            (save_main_planet_storm_track_moisture_nh_summer_view, "main_planet_storm_track_moisture_nh_summer.png"),
            (save_main_planet_storm_track_moisture_equinox_view, "main_planet_storm_track_moisture_equinox.png"),
            (save_main_planet_storm_track_moisture_nh_winter_view, "main_planet_storm_track_moisture_nh_winter.png"),
            (save_main_planet_circulation_zones_view, "main_planet_circulation_zones.png"),
            (save_main_planet_circulation_zones_nh_summer_view, "main_planet_circulation_zones_nh_summer.png"),
            (save_main_planet_circulation_zones_equinox_view, "main_planet_circulation_zones_equinox.png"),
            (save_main_planet_circulation_zones_nh_winter_view, "main_planet_circulation_zones_nh_winter.png"),
            (save_main_planet_ocean_gyres_view, "main_planet_ocean_gyres.png"),
        ]:
            try:
                save(func, name)
            except Exception as exc:
                print(f"[warning] Could not write {name}: {exc}", flush=True)

    if stage_i >= stage_index("hydrology") and stage_exists(output_dir, "hydrology"):
        for func, name in [
            (save_main_planet_hydrology_view, "main_planet_hydrology.png"),
            (save_main_planet_drainage_basins_view, "main_planet_drainage_basins.png"),
            (save_main_planet_delta_mouths_view, "main_planet_delta_mouths.png"),
        ]:
            try:
                save(func, name)
            except Exception as exc:
                print(f"[warning] Could not write {name}: {exc}", flush=True)

    if stage_i >= stage_index("biomes") and stage_exists(output_dir, "biomes"):
        try:
            save(save_main_planet_biome_view, "main_planet_biomes.png")
        except Exception as exc:
            print(f"[warning] Could not write main_planet_biomes.png: {exc}", flush=True)

    # Update 10: main_planet_regions.png is deprecated/hidden and no longer
    # generated by staged output refresh. Region CSV/state can remain for internal
    # summaries, but the image cluttered map review and confused page titles.


def _print_status(output_dir: str | Path) -> None:
    write_status_report(output_dir)
    print("WorldGen staged-run status")
    print("==========================")
    print(f"Output directory: {output_dir}")
    override_reason = unapplied_override_reason(output_dir)
    if override_reason:
        print(f"Override note: {override_reason}. Run apply-overrides before continuing if those edits should take effect.")
    print("")
    print(f"{'Stage':38s}  {'Status':14s}  {'Completed':24s}  Reason")
    print("-" * 96)
    for stage, status, completed, reason in status_rows(output_dir):
        print(f"{stage:38s}  {status:14s}  {completed:24s}  {reason}")
    print("")
    print("Terrain/geology sub-phases:")
    for sub in TERRAIN_SUBPHASES:
        print(f"  - {sub}")
    print("")
    print("Status reports written to diagnostics/pipeline_status.json and diagnostics/pipeline_status.csv")


def main() -> None:
    performance.reset()
    args = build_parser().parse_args()

    if args.command == "new":
        out = ensure_layout(args.output_dir)
        config = _resolved_config_from_new_args(args)
        width = config.planet_profile.map_width
        height = config.planet_profile.map_height
        write_resolved_config(out, config, vars(args))
        write_stage_graph(out)
        print(f"[progress] Created staged WorldGen run at {out}", flush=True)
        print(f"[progress] Resolved seed={config.seed}, map={width}x{height}", flush=True)
        _print_tectonic_grid_notice(config)
        _run_to(out, args.run_to, args)
        _print_status(out)
        return

    if args.command == "status":
        _print_status(args.output_dir)
        return

    if args.command == "validate":
        warnings = validate_staged_run(args.output_dir)
        if warnings:
            print("Validation warnings:")
            for warning in warnings:
                print(f"  - {warning}")
        else:
            print("No validation warnings found.")
        print("Validation report written to diagnostics/pipeline_validation_report.txt")
        return

    if args.command == "apply-overrides":
        result = apply_stage_overrides(args.output_dir)
        if result.changes:
            print("Applied overrides:")
            for change in result.changes:
                print(f"  - {change}")
        else:
            print("No override values changed staged state/config.")
        if result.warnings:
            print("Warnings:")
            for warning in result.warnings:
                print(f"  - {warning}")
        print("Override report written to diagnostics/override_application_report.json")
        if args.validate:
            warnings = validate_staged_run(args.output_dir)
            print(f"Validation completed with {len(warnings)} warning(s).")
        _print_status(args.output_dir)
        return

    if args.command == "run-to":
        _run_to(args.output_dir, args.stage, args)
        _print_status(args.output_dir)
        return

    if args.command == "run-from":
        stage = normalize_stage(args.stage)
        stop_at = normalize_stage(getattr(args, "stop_at", "outputs"))
        if stage_index(stop_at) < stage_index(stage):
            raise RuntimeError(f"--stop-at stage '{stop_at}' cannot be before rerun stage '{stage}'.")
        _run_to(args.output_dir, stop_at, args, force_from=stage)
        _print_status(args.output_dir)
        return

    if args.command == "run-stage":
        _run_stage(args.output_dir, args.stage, args, force=True)
        _print_status(args.output_dir)
        return

    if args.command == "maps":
        _write_maps_available(args.output_dir, args)
        return

    raise AssertionError(args.command)


if __name__ == "__main__":
    main()
