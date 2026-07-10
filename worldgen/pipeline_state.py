"""State persistence helpers for the staged WorldGen pipeline.

Update45 introduced an inspectable/resumable file layout without changing the
classic ``python -m worldgen.main`` workflow.  The stage runner stores small
human-editable JSON files for system/planet physics and fast NPZ files for large
rasters so a user can inspect or edit state before running downstream stages.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from worldgen.config import PlanetProfileConfig, StarConfig, SystemConfig, WorldGenConfig
from worldgen.models.bodies import Composition, Moon, MoonOrbit, Orbit, Planet, Star, ValueSource
from worldgen.models.planet_profile import (
    Atmosphere,
    BiomeMap,
    ClimateMap,
    GeologyState,
    HydrologyMap,
    Hydrosphere,
    MainPlanetProfile,
    RegionAnalysis,
    RegionSummary,
    RotationState,
    TerrainMap,
)
from worldgen.models.system import StarSystem

SCHEMA_VERSION = 1
PIPELINE_VERSION = "plate_terrain_6_final_integration_qa"

# Executable stages in dependency order.  Update65 keeps terrain sub-stages inspectable with
# user-addressable sub-stages so long runs can stop before climate/hydrology and
# emit diagnostic maps as soon as terrain data is available.  The current terrain
# synthesizer is still internally monolithic; these sub-stages are checkpointed
# and addressable wrappers around the generated terrain state, ready for deeper
# internal splitting in later updates.
TERRAIN_SUBPHASES = [
    "terrain-foundation-mask",
    "terrain-tectonic-provinces",
    "terrain-crust-and-boundaries",
    "terrain-mountains-basins-rifts",
    "terrain-coasts-shelves-islands",
    "terrain-erosion-deposition",
    "terrain-finalization-recentering",
]

FINAL_TERRAIN_STAGE = TERRAIN_SUBPHASES[-1]

STAGE_ORDER = [
    "solar-system",
    "planet-physics",
    *TERRAIN_SUBPHASES,
    "climate",
    "hydrology",
    "biomes",
    "regions",
    "outputs",
]

TERRAIN_SUBPHASE_FILES = {
    "terrain-foundation-mask": ["state/03a_terrain_foundation_mask.json"],
    "terrain-tectonic-provinces": ["state/03b_terrain_tectonic_provinces.json"],
    "terrain-crust-and-boundaries": ["state/03c_terrain_crust_and_boundaries.json"],
    "terrain-mountains-basins-rifts": ["state/03d_terrain_mountains_basins_rifts.json"],
    "terrain-coasts-shelves-islands": ["state/03e_terrain_coasts_shelves_islands.json"],
    "terrain-erosion-deposition": ["state/03f_terrain_erosion_deposition.json"],
    "terrain-finalization-recentering": ["state/03_terrain.npz", "state/03_terrain_metadata.json", "state/03g_terrain_finalization_recentering.json"],
}

STAGE_FILES = {
    "solar-system": ["state/01_solar_system.json"],
    "planet-physics": ["state/02_planet_physics.json"],
    **TERRAIN_SUBPHASE_FILES,
    "climate": ["state/04_climate.npz", "state/04_climate_metadata.json"],
    "hydrology": ["state/05_hydrology.npz", "state/05_hydrology_metadata.json"],
    "biomes": ["state/06_biomes.npz", "state/06_biomes_metadata.json"],
    "regions": ["state/07_regions.json"],
    "outputs": ["worldgen_diagnostic_bundle.zip"],
}


def now_stamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def ensure_layout(output_dir: str | Path) -> Path:
    path = Path(output_dir)
    for child in ("config", "state", "maps", "diagnostics"):
        (path / child).mkdir(parents=True, exist_ok=True)
    return path


def config_to_dict(config: WorldGenConfig) -> dict[str, Any]:
    return {
        "seed": config.seed,
        "star": asdict(config.star),
        "system": asdict(config.system),
        "planet_profile": asdict(config.planet_profile),
    }


def config_from_dict(data: dict[str, Any]) -> WorldGenConfig:
    return WorldGenConfig(
        seed=data.get("seed"),
        star=StarConfig(**data.get("star", {})),
        system=SystemConfig(**data.get("system", {})),
        planet_profile=PlanetProfileConfig(**data.get("planet_profile", {})),
    )


def write_resolved_config(output_dir: str | Path, config: WorldGenConfig, command_args: dict[str, Any] | None = None) -> None:
    path = ensure_layout(output_dir)
    data = config_to_dict(config)
    if command_args:
        data["command_args"] = command_args
    (path / "config" / "resolved_config.json").write_text(json.dumps(data, indent=2), encoding="utf-8")
    overrides = path / "config" / "stage_overrides.json"
    if not overrides.exists():
        overrides.write_text(
            json.dumps(
                {
                    "_comment": "Optional manual overrides. Edit values, then run: python -m worldgen.pipeline apply-overrides --output-dir <run>. Null values are ignored.",
                    "star": {
                        "mass_solar": None,
                        "luminosity_solar": None,
                        "age_gyr": None,
                        "metallicity": None
                    },
                    "system": {
                        "planet_count": None,
                        "min_planets": None,
                        "max_planets": None,
                        "architecture_type": None,
                        "main_planet_preference": None,
                        "require_major_moon": None,
                        "moon_strength_preference": None
                    },
                    "planet_profile": {
                        "map_width": None,
                        "map_height": None,
                        "koppen_detail": None,
                        "climate_generation_mode": None,
                        "terrain_generation_mode": None,
                        "suppress_polar_land": None,
                        "tectonic_grid_scale": None,
                        "tectonic_grid_policy": None,
                        "allow_experimental_tectonic_grid": None,
                        "tectonic_grid_width": None,
                        "tectonic_grid_height": None,
                        "generate_hydrology": None,
                        "generate_biomes": None,
                        "generate_regions": None
                    },
                    "main_planet": {
                        "mass_earth": None,
                        "radius_earth": None,
                        "surface_gravity_g": None,
                        "stellar_flux_earth": None,
                        "orbit": {
                            "semi_major_axis_au": None,
                            "eccentricity": None,
                            "orbital_period_days": None
                        }
                    },
                    "planet_physics": {
                        "rotation": {
                            "rotation_period_hours": None,
                            "axial_tilt_degrees": None
                        },
                        "atmosphere": {
                            "pressure_bar": None,
                            "carbon_dioxide_ppm": None,
                            "water_vapor_factor": None,
                            "greenhouse_warming_k": None
                        },
                        "hydrosphere": {
                            "ocean_fraction_target": None,
                            "volatile_fraction": None
                        },
                        "geology": {
                            "internal_heat": None,
                            "volcanism": None,
                            "erosion": None,
                            "mountain_factor": None,
                            "surface_roughness": None
                        }
                    },
                    "terrain": {
                        "_comment": "Terrain review controls. Derived defaults come from Stage 1/2; non-null values here are treated as manual terrain precondition overrides.",
                        "terrain_generation_mode": None,
                        "terrain_style": None,
                        "supercontinent_tendency": None,
                        "target_plate_count": None,
                        "fragmentation_tendency": None,
                        "coastline_complexity": None,
                        "island_density": None,
                        "shelf_width_factor": None,
                        "mountain_belt_strength": None,
                        "rift_strength": None,
                        "interior_relief": None,
                        "erosion_deposition_strength": None,
                        "erosion_deposition_multiplier": None,
                        "continental_shelf_strength": None,
                        "v4_topology_strength": None,
                        "v4_island_strength": None,
                        "v4_rift_strength": None,
                        "diagnostic_detail": None
                    }
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    # CLI terrain tuning values are persisted as terrain overrides so Stage 3
    # receives them through the same path as the Web UI/manual editor.
    if command_args:
        cli_terrain_keys = {
            "erosion_deposition_multiplier": "erosion_deposition_multiplier",
            "continental_shelf_strength": "continental_shelf_strength",
            "shelf_width_factor": "shelf_width_factor",
            "v4_topology_strength": "v4_topology_strength",
            "v4_island_strength": "v4_island_strength",
            "v4_rift_strength": "v4_rift_strength",
        }
        try:
            override_data = json.loads(overrides.read_text(encoding="utf-8")) if overrides.exists() else {}
            if not isinstance(override_data, dict):
                override_data = {}
            terrain = override_data.setdefault("terrain", {})
            changed = False
            for arg_key, terrain_key in cli_terrain_keys.items():
                value = command_args.get(arg_key)
                if value is not None and value != "":
                    terrain[terrain_key] = value
                    changed = True
            if changed:
                overrides.write_text(json.dumps(override_data, indent=2), encoding="utf-8")
        except Exception:
            pass


def read_resolved_config(output_dir: str | Path) -> WorldGenConfig:
    path = Path(output_dir) / "config" / "resolved_config.json"
    if not path.exists():
        raise FileNotFoundError(f"No resolved config found at {path}; run 'python -m worldgen.pipeline new ...' first.")
    return config_from_dict(json.loads(path.read_text(encoding="utf-8")))


def stage_index(stage: str) -> int:
    stage = normalize_stage(stage)
    return STAGE_ORDER.index(stage)


def normalize_stage(stage: str) -> str:
    key = stage.strip().lower().replace("_", "-")
    aliases = {
        "system": "solar-system",
        "solar": "solar-system",
        "orbits": "solar-system",
        "physics": "planet-physics",
        "planet": "planet-physics",
        "profile": "planet-physics",
        "geology": FINAL_TERRAIN_STAGE,
        "terrain": FINAL_TERRAIN_STAGE,
        "terrain-geology": FINAL_TERRAIN_STAGE,
        "terrain-final": FINAL_TERRAIN_STAGE,
        "terrain-finalization": FINAL_TERRAIN_STAGE,
        "koppen": "climate",
        "biome": "biomes",
        "region": "regions",
        "maps": "outputs",
        "diagnostics": "outputs",
        "full": "outputs",
        "all": "outputs",
    }
    key = aliases.get(key, key)
    if key not in STAGE_ORDER:
        valid = list(STAGE_ORDER) + ["terrain"]
        raise ValueError(f"Unknown stage '{stage}'. Valid stages: {', '.join(valid)}")
    return key


def stage_seed(base_seed: int | None, stage: str) -> int:
    normalized = normalize_stage(stage)
    # All terrain sub-stages currently share one deterministic terrain seed so
    # rerunning from any terrain checkpoint reproduces the same terrain until the
    # internal terrain generator is split into independent resumable passes.
    seed_stage = "terrain" if normalized in TERRAIN_SUBPHASES else normalized
    material = f"{base_seed if base_seed is not None else 0}:worldgen:{seed_stage}".encode("utf-8")
    digest = hashlib.sha256(material).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)


def stage_exists(output_dir: str | Path, stage: str) -> bool:
    path = Path(output_dir)
    return all((path / rel).exists() for rel in STAGE_FILES[normalize_stage(stage)])


def remove_stage_and_after(output_dir: str | Path, stage: str) -> None:
    path = Path(output_dir)
    normalized = normalize_stage(stage)
    # The current terrain generator is still a single pass. If any terrain
    # sub-phase is invalidated, clear all terrain sub-phase markers and the final
    # terrain raster so regenerated checkpoints cannot be mixed with stale ones.
    start = stage_index(TERRAIN_SUBPHASES[0]) if normalized in TERRAIN_SUBPHASES else stage_index(normalized)
    manifest = read_manifest(path)
    stages = manifest.setdefault("stages", {})
    changed_manifest = False
    for later in STAGE_ORDER[start:]:
        if later in stages:
            stages.pop(later, None)
            changed_manifest = True
        for rel in STAGE_FILES[later]:
            file_path = path / rel
            if file_path.exists():
                file_path.unlink()
    if changed_manifest:
        write_manifest(path, manifest)


def read_manifest(output_dir: str | Path) -> dict[str, Any]:
    path = Path(output_dir) / "worldgen_run_manifest.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {
        "schema_version": SCHEMA_VERSION,
        "pipeline_version": PIPELINE_VERSION,
        "created_at": now_stamp(),
        "updated_at": now_stamp(),
        "stages": {},
        "terrain_subphases": TERRAIN_SUBPHASES,
    }


def write_manifest(output_dir: str | Path, manifest: dict[str, Any]) -> None:
    path = ensure_layout(output_dir)
    manifest["updated_at"] = now_stamp()
    manifest["pipeline_version"] = PIPELINE_VERSION
    manifest["terrain_subphases"] = TERRAIN_SUBPHASES
    (path / "worldgen_run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def mark_stage(output_dir: str | Path, stage: str, status: str, elapsed_s: float | None = None, note: str = "") -> None:
    stage = normalize_stage(stage)
    manifest = read_manifest(output_dir)
    manifest.setdefault("stages", {})[stage] = {
        "status": status,
        "completed_at": now_stamp() if status == "complete" else None,
        "elapsed_s": round(elapsed_s, 3) if elapsed_s is not None else None,
        "files": STAGE_FILES.get(stage, []),
        "note": note,
    }
    write_manifest(output_dir, manifest)
    append_stage_log(output_dir, stage, status, elapsed_s, note)


def append_stage_log(output_dir: str | Path, stage: str, status: str, elapsed_s: float | None, note: str) -> None:
    path = ensure_layout(output_dir) / "diagnostics" / "pipeline_stage_log.csv"
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        if not exists:
            writer.writerow(["timestamp", "stage", "status", "elapsed_s", "note"])
        writer.writerow([now_stamp(), stage, status, "" if elapsed_s is None else f"{elapsed_s:.3f}", note])


def write_stage_graph(output_dir: str | Path) -> None:
    path = ensure_layout(output_dir) / "diagnostics" / "pipeline_stage_graph.txt"
    lines = [
        "WorldGen staged pipeline",
        "========================",
        "",
        "Executable stages:",
    ]
    for i, stage in enumerate(STAGE_ORDER, start=1):
        lines.append(f"  {i:02d}. {stage}")
    lines += [
        "",
        "Terrain/geology planned sub-phases:",
    ]
    for i, stage in enumerate(TERRAIN_SUBPHASES, start=1):
        lines.append(f"  04{chr(96+i)}. {stage}")
    lines += [
        "",
        "Update65 note: terrain sub-phases now include review diagnostics and user-addressable pipeline stages.",
        "The current terrain synthesizer is still internally monolithic; running any terrain sub-phase generates the full terrain checkpoint once, then writes sub-phase marker files and diagnostic maps from available data.",
        "Later terrain-overhaul updates can move logic into these same checkpoint names without changing user commands.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def status_rows(output_dir: str | Path) -> list[tuple[str, str, str]]:
    manifest = read_manifest(output_dir)
    stages = manifest.get("stages", {})
    rows = []
    for stage in STAGE_ORDER:
        exists = stage_exists(output_dir, stage)
        state = stages.get(stage, {})
        status = "complete" if exists else "missing"
        if state.get("status") and not exists:
            status = f"{state.get('status')} (files missing)"
        completed = state.get("completed_at") or ""
        rows.append((stage, status, completed))
    return rows


# ---------------------------------------------------------------------------
# JSON/dataclass conversion helpers
# ---------------------------------------------------------------------------


def _vs_dict(data: dict[str, Any] | None) -> dict[str, ValueSource]:
    if not data:
        return {}
    out: dict[str, ValueSource] = {}
    for key, value in data.items():
        if isinstance(value, ValueSource):
            out[key] = value
        elif isinstance(value, dict):
            out[key] = ValueSource(**value)
    return out


def _composition(data: dict[str, Any]) -> Composition:
    return Composition(**data)


def _orbit(data: dict[str, Any]) -> Orbit:
    return Orbit(**data)


def _moon_orbit(data: dict[str, Any]) -> MoonOrbit:
    return MoonOrbit(**data)


def _moon(data: dict[str, Any] | None) -> Moon | None:
    if not data:
        return None
    return Moon(
        name=data["name"],
        moon_class=data["moon_class"],
        mass_earth=data["mass_earth"],
        radius_earth=data["radius_earth"],
        density_relative_earth=data["density_relative_earth"],
        composition=_composition(data["composition"]),
        orbit=_moon_orbit(data["orbit"]),
        surface_gravity_g=data["surface_gravity_g"],
        tidal_strength_relative_earth_moon=data["tidal_strength_relative_earth_moon"],
        angular_diameter_degrees=data["angular_diameter_degrees"],
        value_sources=_vs_dict(data.get("value_sources")),
        moon_origin=data.get("moon_origin", "unknown"),
        tidal_effect_level=data.get("tidal_effect_level", "moderate"),
        axial_stability_effect=data.get("axial_stability_effect", "moderate"),
        notes=list(data.get("notes", [])),
    )


def star_from_dict(data: dict[str, Any]) -> Star:
    return Star(
        stellar_class=data["stellar_class"],
        mass_solar=data["mass_solar"],
        age_gyr=data["age_gyr"],
        metallicity=data["metallicity"],
        luminosity_solar=data["luminosity_solar"],
        radius_solar=data["radius_solar"],
        temperature_k=data["temperature_k"],
        main_sequence_lifetime_gyr=data["main_sequence_lifetime_gyr"],
        habitable_zone_inner_au=data["habitable_zone_inner_au"],
        habitable_zone_outer_au=data["habitable_zone_outer_au"],
        snow_line_au=data["snow_line_au"],
        value_sources=_vs_dict(data.get("value_sources")),
        stellar_subclass=data.get("stellar_subclass"),
        spectral_type=data.get("spectral_type"),
        stellar_description=data.get("stellar_description", ""),
    )


def planet_from_dict(data: dict[str, Any]) -> Planet:
    return Planet(
        name=data["name"],
        planet_class=data["planet_class"],
        mass_earth=data["mass_earth"],
        radius_earth=data["radius_earth"],
        density_relative_earth=data["density_relative_earth"],
        composition=_composition(data["composition"]),
        orbit=_orbit(data["orbit"]),
        stellar_flux_earth=data["stellar_flux_earth"],
        equilibrium_temperature_k=data["equilibrium_temperature_k"],
        surface_gravity_g=data["surface_gravity_g"],
        escape_velocity_relative_earth=data["escape_velocity_relative_earth"],
        hill_radius_au=data["hill_radius_au"],
        habitability_score=data.get("habitability_score", 0.0),
        is_main_planet=data.get("is_main_planet", False),
        moon=_moon(data.get("moon")),
        selection_notes=list(data.get("selection_notes", [])),
        value_sources=_vs_dict(data.get("value_sources")),
        formation_context=dict(data.get("formation_context", {})),
        architecture_role=data.get("architecture_role", "ordinary_planet"),
    )


def save_system_state(output_dir: str | Path, system: StarSystem) -> None:
    path = ensure_layout(output_dir) / "state" / "01_solar_system.json"
    data = {
        "schema_version": SCHEMA_VERSION,
        "stage": "solar-system",
        "seed": system.seed,
        "star": system.star.to_dict(),
        "planets": [p.to_dict() for p in system.planets],
        "notes": list(system.notes),
        "architecture": system.architecture,
        "diagnostics": system.diagnostics,
        "edit_note": "This file is safe to inspect/edit before downstream stages. Keep exactly one planet with is_main_planet=true.",
    }
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def load_system_state(output_dir: str | Path) -> StarSystem:
    path = Path(output_dir) / "state" / "01_solar_system.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    return StarSystem(
        seed=data.get("seed"),
        star=star_from_dict(data["star"]),
        planets=[planet_from_dict(p) for p in data["planets"]],
        notes=list(data.get("notes", [])),
        main_planet_profile=None,
        architecture=data.get("architecture", data.get("diagnostics", {}).get("architecture", "unspecified")),
        diagnostics=dict(data.get("diagnostics", {})),
    )


def save_physics_state(
    output_dir: str | Path,
    rotation: RotationState,
    atmosphere: Atmosphere,
    hydrosphere: Hydrosphere,
    geology: GeologyState,
) -> None:
    path = ensure_layout(output_dir) / "state" / "02_planet_physics.json"
    data = {
        "schema_version": SCHEMA_VERSION,
        "stage": "planet-physics",
        "rotation": rotation.to_dict(),
        "atmosphere": atmosphere.to_dict(),
        "hydrosphere": hydrosphere.to_dict(),
        "geology": geology.to_dict(),
        "edit_note": "Editable physical state. Changes here invalidate terrain and every downstream stage.",
    }
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def load_physics_state(output_dir: str | Path) -> tuple[RotationState, Atmosphere, Hydrosphere, GeologyState]:
    data = json.loads((Path(output_dir) / "state" / "02_planet_physics.json").read_text(encoding="utf-8"))
    return (
        RotationState(**data["rotation"]),
        Atmosphere(**data["atmosphere"]),
        Hydrosphere(**data["hydrosphere"]),
        GeologyState(**data["geology"]),
    )


def terrain_subphase_label(stage: str) -> str:
    stage = normalize_stage(stage)
    if stage not in TERRAIN_SUBPHASES:
        raise ValueError(f"{stage} is not a terrain sub-phase")
    return stage.replace("terrain-", "").replace("-", " ").title()


def terrain_subphase_marker_path(output_dir: str | Path, stage: str) -> Path:
    stage = normalize_stage(stage)
    # The final terrain stage also owns the shared raster files; its marker is
    # the final JSON entry, not the NPZ.
    rel = TERRAIN_SUBPHASE_FILES[stage][-1]
    return ensure_layout(output_dir) / rel


def save_terrain_subphase_checkpoint(
    output_dir: str | Path,
    stage: str,
    terrain: TerrainMap,
    *,
    elapsed_s: float | None = None,
    note: str = "",
) -> None:
    """Write a small, human-readable marker for a terrain sub-phase.

    Update65 avoids duplicating huge terrain arrays for every sub-phase.  The
    final terrain raster remains in 03_terrain.npz, while each sub-phase gets a
    marker JSON that records the available diagnostic fields and points at the
    shared raster checkpoint.
    """
    stage = normalize_stage(stage)
    data = {
        "schema_version": SCHEMA_VERSION,
        "pipeline_version": PIPELINE_VERSION,
        "stage": stage,
        "label": terrain_subphase_label(stage),
        "completed_at": now_stamp(),
        "elapsed_s": None if elapsed_s is None else round(elapsed_s, 3),
        "width": terrain.width,
        "height": terrain.height,
        "source_terrain_npz": "state/03_terrain.npz",
        "source_terrain_metadata": "state/03_terrain_metadata.json",
        "available_arrays": {
            "elevation_m": True,
            "is_land": True,
            "tectonic_plate_id": terrain.tectonic_plate_id is not None,
            "tectonic_boundary_class": terrain.tectonic_boundary_class is not None,
            "crust_type": terrain.crust_type is not None,
        },
        "terrain_stats": {
            "ocean_fraction": terrain.ocean_fraction,
            "land_fraction": terrain.land_fraction,
            "min_elevation_m": terrain.min_elevation_m,
            "max_elevation_m": terrain.max_elevation_m,
            "mean_land_elevation_m": terrain.mean_land_elevation_m,
            "mean_ocean_depth_m": terrain.mean_ocean_depth_m,
            "planet_radius_earth": terrain.planet_radius_earth,
        },
        "subphase_diagnostics": (getattr(terrain, "terrain_diagnostics", None) or {}).get("subphases", {}).get(stage, {}),
        "diagnostic_folder": "terrain_diagnostics/" + str(((getattr(terrain, "terrain_diagnostics", None) or {}).get("subphases", {}).get(stage, {}) or {}).get("folder", "")),
        "note": note or "Checkpoint marker references the shared terrain raster and the Stage 3 diagnostic folder for this sub-stage.",
    }
    terrain_subphase_marker_path(output_dir, stage).write_text(json.dumps(data, indent=2), encoding="utf-8")


def save_all_terrain_subphase_checkpoints(output_dir: str | Path, terrain: TerrainMap, *, elapsed_s: float | None = None) -> None:
    for stage in TERRAIN_SUBPHASES:
        save_terrain_subphase_checkpoint(
            output_dir,
            stage,
            terrain,
            elapsed_s=elapsed_s,
            note=(
                "Shared monolithic terrain pass total. This sub-stage is addressable now, but internal terrain timing will become exact after the terrain generator is split."
            ),
        )


def stage_is_terrain_subphase(stage: str) -> bool:
    return normalize_stage(stage) in TERRAIN_SUBPHASES


def final_terrain_stage() -> str:
    return FINAL_TERRAIN_STAGE


def save_terrain_state(output_dir: str | Path, terrain: TerrainMap) -> None:
    import numpy as np

    path = ensure_layout(output_dir)
    arrays = {
        "elevation_m": np.asarray(terrain.elevation_m, dtype=np.int16),
        "is_land": np.asarray(terrain.is_land, dtype=np.bool_),
    }
    if terrain.tectonic_plate_id is not None:
        arrays["tectonic_plate_id"] = np.asarray(terrain.tectonic_plate_id, dtype=np.int16)
    if terrain.tectonic_boundary_class is not None:
        arrays["tectonic_boundary_class"] = np.asarray(terrain.tectonic_boundary_class, dtype=np.int16)
    if getattr(terrain, "tectonic_province_type", None) is not None:
        arrays["tectonic_province_type"] = np.asarray(terrain.tectonic_province_type, dtype=np.int16)
    if getattr(terrain, "tectonic_province_age_x1000", None) is not None:
        arrays["tectonic_province_age_x1000"] = np.asarray(terrain.tectonic_province_age_x1000, dtype=np.int16)
    if getattr(terrain, "tectonic_boundary_strength_x1000", None) is not None:
        arrays["tectonic_boundary_strength_x1000"] = np.asarray(terrain.tectonic_boundary_strength_x1000, dtype=np.int16)
    if getattr(terrain, "tectonic_boundary_width_x1000", None) is not None:
        arrays["tectonic_boundary_width_x1000"] = np.asarray(terrain.tectonic_boundary_width_x1000, dtype=np.int16)
    if getattr(terrain, "plate_tectonic_plate_type", None) is not None:
        arrays["plate_tectonic_plate_type"] = np.asarray(terrain.plate_tectonic_plate_type, dtype=np.int16)
    for plate_field_name in [
        "plate_tectonic_continental_crust_x1000",
        "plate_tectonic_craton_core_x1000",
        "plate_tectonic_microplate_x1000",
        "plate_tectonic_sedimentary_plain_x1000",
        "plate_tectonic_velocity_x_x1000",
        "plate_tectonic_velocity_y_x1000",
        "plate_tectonic_speed_x1000",
        "plate_tectonic_convergence_x1000",
        "plate_tectonic_divergence_x1000",
        "plate_tectonic_transform_x1000",
        "plate_tectonic_ocean_crust_age_x1000",
        "plate_tectonic_mid_ocean_ridge_x1000",
        "plate_tectonic_trench_x1000",
        "plate_tectonic_fracture_zone_x1000",
        "plate_tectonic_abyssal_plain_x1000",
        "plate_tectonic_seamount_x1000",
        "plate_tectonic_orogeny_strength_x1000",
        "plate_tectonic_volcanic_arc_x1000",
        "plate_tectonic_continental_rift_x1000",
        "plate_tectonic_foreland_basin_x1000",
        "plate_tectonic_craton_shield_x1000",
        "plate_tectonic_accreted_terrane_x1000",
        "plate_tectonic_plateau_uplift_x1000",
        "plate_tectonic_shelf_width_x1000",
        "plate_tectonic_active_margin_x1000",
        "plate_tectonic_passive_margin_x1000",
        "plate_tectonic_rifted_margin_x1000",
        "plate_tectonic_island_arc_x1000",
        "plate_tectonic_coastal_plain_x1000",
        "plate_tectonic_coast_ruggedness_x1000",
        "plate_tectonic_backend_integration_x1000",
        "plate_tectonic_hydrology_readiness_x1000",
        "plate_tectonic_legacy_dependency_x1000",
    ]:
        value = getattr(terrain, plate_field_name, None)
        if value is not None:
            arrays[plate_field_name] = np.asarray(value, dtype=np.int16)
    for relief_name in [
        "terrain_mountain_strength_x1000",
        "terrain_basin_field_x1000",
        "terrain_rift_field_x1000",
        "terrain_interior_relief_x1000",
        "terrain_shield_highland_x1000",
        "terrain_plateau_x1000",
        "terrain_shelf_width_x1000",
        "terrain_submerged_continental_crust_x1000",
        "terrain_continental_shelf_support_x1000",
        "terrain_shelf_depth_target_x1000",
        "terrain_lake_depth_limit_x1000",
        "terrain_ripple_artifact_risk_x1000",
        "terrain_v4_boundary_deformation_x1000",
        "terrain_v4_volcanic_island_support_x1000",
        "terrain_v4_rift_cut_support_x1000",
        "terrain_v4_mountain_branch_support_x1000",
        "terrain_v4_elevation_delta_m",
        "terrain_coast_ruggedness_x1000",
        "terrain_mid_ocean_ridge_x1000",
        "terrain_trench_x1000",
        "terrain_fracture_zone_x1000",
        "terrain_seamount_x1000",
        "terrain_island_shape_complexity_x1000",
        "terrain_erosion_strength_x1000",
        "terrain_deposition_field_x1000",
        "terrain_valley_corridor_x1000",
        "terrain_sediment_supply_x1000",
        "terrain_coastal_plain_x1000",
        "terrain_alluvial_fan_x1000",
        "terrain_floodplain_x1000",
        "terrain_maturity_x1000",
    ]:
        value = getattr(terrain, relief_name, None)
        if value is not None:
            arrays[relief_name] = np.asarray(value, dtype=np.int16)
    for class_name in [
        "terrain_coast_style_class",
        "terrain_island_origin_class",
        "terrain_ocean_floor_class",
        "terrain_shelf_zone_class",
        "terrain_final_plate_component_class",
        "terrain_v4_topology_class",
        "terrain_v4_island_chain_class",
        "terrain_v4_boundary_network_class",
        "terrain_v4_orogen_network_class",
        "terrain_v4_control_response_class",
        "terrain_v4_landform_change_class",
        "plate_tectonic_continent_assembly_id",
        "plate_tectonic_plate_topology_problem_class",
        "plate_tectonic_landform_class",
        "plate_tectonic_margin_profile_class",
        "plate_tectonic_boundary_class",
        "plate_tectonic_subduction_polarity",
        "plate_tectonic_ocean_floor_class",
        "plate_tectonic_margin_class",
        "plate_tectonic_island_origin_class",
        "plate_tectonic_problem_class",
        "plate_tectonic_valley_corridor_x1000",
        "plate_tectonic_inland_basin_x1000",
        "plate_tectonic_lake_candidate_x1000",
        "plate_tectonic_terrain_detail_x1000",
        "plate_tectonic_drainage_ready_delta_m",
    ]:
        value = getattr(terrain, class_name, None)
        if value is not None:
            arrays[class_name] = np.asarray(value, dtype=np.int16)
    if getattr(terrain, "terrain_relief_delta_m", None) is not None:
        arrays["terrain_relief_delta_m"] = np.asarray(terrain.terrain_relief_delta_m, dtype=np.int16)
    if getattr(terrain, "plate_tectonic_relief_delta_m", None) is not None:
        arrays["plate_tectonic_relief_delta_m"] = np.asarray(terrain.plate_tectonic_relief_delta_m, dtype=np.int16)
    if getattr(terrain, "plate_tectonic_coast_delta_m", None) is not None:
        arrays["plate_tectonic_coast_delta_m"] = np.asarray(terrain.plate_tectonic_coast_delta_m, dtype=np.int16)
    if terrain.crust_type is not None:
        arrays["crust_type"] = np.asarray(terrain.crust_type, dtype=np.int16)
    np.savez_compressed(path / "state" / "03_terrain.npz", **arrays)
    meta = {
        "schema_version": SCHEMA_VERSION,
        "stage": "terrain",
        "width": terrain.width,
        "height": terrain.height,
        "min_elevation_m": terrain.min_elevation_m,
        "max_elevation_m": terrain.max_elevation_m,
        "mean_land_elevation_m": terrain.mean_land_elevation_m,
        "mean_ocean_depth_m": terrain.mean_ocean_depth_m,
        "ocean_fraction": terrain.ocean_fraction,
        "land_fraction": terrain.land_fraction,
        "source": terrain.source,
        "planet_radius_earth": terrain.planet_radius_earth,
        "terrain_subphases": TERRAIN_SUBPHASES,
        "terrain_diagnostics": getattr(terrain, "terrain_diagnostics", None),
        "edit_note": "Large raster arrays are in 03_terrain.npz. Metadata edits may invalidate climate and downstream stages.",
    }
    (path / "state" / "03_terrain_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def load_terrain_state(output_dir: str | Path) -> TerrainMap:
    import numpy as np

    path = Path(output_dir)
    meta = json.loads((path / "state" / "03_terrain_metadata.json").read_text(encoding="utf-8"))
    data = np.load(path / "state" / "03_terrain.npz", allow_pickle=True)
    def opt(name: str):
        return data[name].tolist() if name in data.files else None
    return TerrainMap(
        width=int(meta["width"]),
        height=int(meta["height"]),
        elevation_m=data["elevation_m"].astype(int).tolist(),
        is_land=data["is_land"].astype(bool).tolist(),
        min_elevation_m=int(meta["min_elevation_m"]),
        max_elevation_m=int(meta["max_elevation_m"]),
        mean_land_elevation_m=float(meta["mean_land_elevation_m"]),
        mean_ocean_depth_m=float(meta["mean_ocean_depth_m"]),
        ocean_fraction=float(meta["ocean_fraction"]),
        land_fraction=float(meta["land_fraction"]),
        source=meta.get("source", "procedural"),
        planet_radius_earth=float(meta.get("planet_radius_earth", 1.0)),
        tectonic_plate_id=opt("tectonic_plate_id"),
        tectonic_boundary_class=opt("tectonic_boundary_class"),
        tectonic_province_type=opt("tectonic_province_type"),
        tectonic_province_age_x1000=opt("tectonic_province_age_x1000"),
        tectonic_boundary_strength_x1000=opt("tectonic_boundary_strength_x1000"),
        tectonic_boundary_width_x1000=opt("tectonic_boundary_width_x1000"),
        plate_tectonic_plate_type=opt("plate_tectonic_plate_type"),
        plate_tectonic_continental_crust_x1000=opt("plate_tectonic_continental_crust_x1000"),
        plate_tectonic_craton_core_x1000=opt("plate_tectonic_craton_core_x1000"),
        plate_tectonic_microplate_x1000=opt("plate_tectonic_microplate_x1000"),
        plate_tectonic_continent_assembly_id=opt("plate_tectonic_continent_assembly_id"),
        plate_tectonic_plate_topology_problem_class=opt("plate_tectonic_plate_topology_problem_class"),
        plate_tectonic_velocity_x_x1000=opt("plate_tectonic_velocity_x_x1000"),
        plate_tectonic_velocity_y_x1000=opt("plate_tectonic_velocity_y_x1000"),
        plate_tectonic_speed_x1000=opt("plate_tectonic_speed_x1000"),
        plate_tectonic_convergence_x1000=opt("plate_tectonic_convergence_x1000"),
        plate_tectonic_divergence_x1000=opt("plate_tectonic_divergence_x1000"),
        plate_tectonic_transform_x1000=opt("plate_tectonic_transform_x1000"),
        plate_tectonic_boundary_class=opt("plate_tectonic_boundary_class"),
        plate_tectonic_subduction_polarity=opt("plate_tectonic_subduction_polarity"),
        plate_tectonic_ocean_floor_class=opt("plate_tectonic_ocean_floor_class"),
        plate_tectonic_ocean_crust_age_x1000=opt("plate_tectonic_ocean_crust_age_x1000"),
        plate_tectonic_mid_ocean_ridge_x1000=opt("plate_tectonic_mid_ocean_ridge_x1000"),
        plate_tectonic_trench_x1000=opt("plate_tectonic_trench_x1000"),
        plate_tectonic_fracture_zone_x1000=opt("plate_tectonic_fracture_zone_x1000"),
        plate_tectonic_abyssal_plain_x1000=opt("plate_tectonic_abyssal_plain_x1000"),
        plate_tectonic_seamount_x1000=opt("plate_tectonic_seamount_x1000"),
        plate_tectonic_orogeny_strength_x1000=opt("plate_tectonic_orogeny_strength_x1000"),
        plate_tectonic_volcanic_arc_x1000=opt("plate_tectonic_volcanic_arc_x1000"),
        plate_tectonic_continental_rift_x1000=opt("plate_tectonic_continental_rift_x1000"),
        plate_tectonic_foreland_basin_x1000=opt("plate_tectonic_foreland_basin_x1000"),
        plate_tectonic_craton_shield_x1000=opt("plate_tectonic_craton_shield_x1000"),
        plate_tectonic_accreted_terrane_x1000=opt("plate_tectonic_accreted_terrane_x1000"),
        plate_tectonic_plateau_uplift_x1000=opt("plate_tectonic_plateau_uplift_x1000"),
        plate_tectonic_sedimentary_plain_x1000=opt("plate_tectonic_sedimentary_plain_x1000"),
        plate_tectonic_landform_class=opt("plate_tectonic_landform_class"),
        plate_tectonic_relief_delta_m=opt("plate_tectonic_relief_delta_m"),
        plate_tectonic_margin_class=opt("plate_tectonic_margin_class"),
        plate_tectonic_shelf_width_x1000=opt("plate_tectonic_shelf_width_x1000"),
        plate_tectonic_active_margin_x1000=opt("plate_tectonic_active_margin_x1000"),
        plate_tectonic_passive_margin_x1000=opt("plate_tectonic_passive_margin_x1000"),
        plate_tectonic_rifted_margin_x1000=opt("plate_tectonic_rifted_margin_x1000"),
        plate_tectonic_island_arc_x1000=opt("plate_tectonic_island_arc_x1000"),
        plate_tectonic_coastal_plain_x1000=opt("plate_tectonic_coastal_plain_x1000"),
        plate_tectonic_coast_ruggedness_x1000=opt("plate_tectonic_coast_ruggedness_x1000"),
        plate_tectonic_island_origin_class=opt("plate_tectonic_island_origin_class"),
        plate_tectonic_margin_profile_class=opt("plate_tectonic_margin_profile_class"),
        plate_tectonic_coast_delta_m=opt("plate_tectonic_coast_delta_m"),
        plate_tectonic_backend_integration_x1000=opt("plate_tectonic_backend_integration_x1000"),
        plate_tectonic_hydrology_readiness_x1000=opt("plate_tectonic_hydrology_readiness_x1000"),
        plate_tectonic_legacy_dependency_x1000=opt("plate_tectonic_legacy_dependency_x1000"),
        plate_tectonic_problem_class=opt("plate_tectonic_problem_class"),
        plate_tectonic_valley_corridor_x1000=opt("plate_tectonic_valley_corridor_x1000"),
        plate_tectonic_inland_basin_x1000=opt("plate_tectonic_inland_basin_x1000"),
        plate_tectonic_lake_candidate_x1000=opt("plate_tectonic_lake_candidate_x1000"),
        plate_tectonic_terrain_detail_x1000=opt("plate_tectonic_terrain_detail_x1000"),
        plate_tectonic_drainage_ready_delta_m=opt("plate_tectonic_drainage_ready_delta_m"),
        terrain_mountain_strength_x1000=opt("terrain_mountain_strength_x1000"),
        terrain_basin_field_x1000=opt("terrain_basin_field_x1000"),
        terrain_rift_field_x1000=opt("terrain_rift_field_x1000"),
        terrain_interior_relief_x1000=opt("terrain_interior_relief_x1000"),
        terrain_shield_highland_x1000=opt("terrain_shield_highland_x1000"),
        terrain_plateau_x1000=opt("terrain_plateau_x1000"),
        terrain_coast_style_class=opt("terrain_coast_style_class"),
        terrain_shelf_width_x1000=opt("terrain_shelf_width_x1000"),
        terrain_submerged_continental_crust_x1000=opt("terrain_submerged_continental_crust_x1000"),
        terrain_continental_shelf_support_x1000=opt("terrain_continental_shelf_support_x1000"),
        terrain_shelf_depth_target_x1000=opt("terrain_shelf_depth_target_x1000"),
        terrain_shelf_zone_class=opt("terrain_shelf_zone_class"),
        terrain_lake_depth_limit_x1000=opt("terrain_lake_depth_limit_x1000"),
        terrain_final_plate_component_class=opt("terrain_final_plate_component_class"),
        terrain_ripple_artifact_risk_x1000=opt("terrain_ripple_artifact_risk_x1000"),
        terrain_v4_boundary_deformation_x1000=opt("terrain_v4_boundary_deformation_x1000"),
        terrain_v4_volcanic_island_support_x1000=opt("terrain_v4_volcanic_island_support_x1000"),
        terrain_v4_rift_cut_support_x1000=opt("terrain_v4_rift_cut_support_x1000"),
        terrain_v4_mountain_branch_support_x1000=opt("terrain_v4_mountain_branch_support_x1000"),
        terrain_v4_topology_class=opt("terrain_v4_topology_class"),
        terrain_v4_island_chain_class=opt("terrain_v4_island_chain_class"),
        terrain_v4_boundary_network_class=opt("terrain_v4_boundary_network_class"),
        terrain_v4_orogen_network_class=opt("terrain_v4_orogen_network_class"),
        terrain_v4_control_response_class=opt("terrain_v4_control_response_class"),
        terrain_v4_elevation_delta_m=opt("terrain_v4_elevation_delta_m"),
        terrain_v4_landform_change_class=opt("terrain_v4_landform_change_class"),
        terrain_coast_ruggedness_x1000=opt("terrain_coast_ruggedness_x1000"),
        terrain_island_origin_class=opt("terrain_island_origin_class"),
        terrain_ocean_floor_class=opt("terrain_ocean_floor_class"),
        terrain_mid_ocean_ridge_x1000=opt("terrain_mid_ocean_ridge_x1000"),
        terrain_trench_x1000=opt("terrain_trench_x1000"),
        terrain_fracture_zone_x1000=opt("terrain_fracture_zone_x1000"),
        terrain_seamount_x1000=opt("terrain_seamount_x1000"),
        terrain_island_shape_complexity_x1000=opt("terrain_island_shape_complexity_x1000"),
        terrain_erosion_strength_x1000=opt("terrain_erosion_strength_x1000"),
        terrain_deposition_field_x1000=opt("terrain_deposition_field_x1000"),
        terrain_valley_corridor_x1000=opt("terrain_valley_corridor_x1000"),
        terrain_sediment_supply_x1000=opt("terrain_sediment_supply_x1000"),
        terrain_coastal_plain_x1000=opt("terrain_coastal_plain_x1000"),
        terrain_alluvial_fan_x1000=opt("terrain_alluvial_fan_x1000"),
        terrain_floodplain_x1000=opt("terrain_floodplain_x1000"),
        terrain_maturity_x1000=opt("terrain_maturity_x1000"),
        terrain_relief_delta_m=opt("terrain_relief_delta_m"),
        crust_type=opt("crust_type"),
        terrain_diagnostics=meta.get("terrain_diagnostics"),
    )


def save_climate_state(output_dir: str | Path, climate: ClimateMap) -> None:
    import numpy as np

    path = ensure_layout(output_dir)
    np.savez_compressed(
        path / "state" / "04_climate.npz",
        annual_mean_temp_c_x10=np.asarray(climate.annual_mean_temp_c_x10, dtype=np.int16),
        warmest_month_temp_c_x10=np.asarray(climate.warmest_month_temp_c_x10, dtype=np.int16),
        coldest_month_temp_c_x10=np.asarray(climate.coldest_month_temp_c_x10, dtype=np.int16),
        annual_precip_mm=np.asarray(climate.annual_precip_mm, dtype=np.uint16),
        koppen_classification=np.asarray(climate.koppen_classification),
    )
    driver_maps = getattr(climate, "climate_driver_maps", None) or {}
    if driver_maps:
        np.savez_compressed(
            path / "state" / "04_climate_drivers.npz",
            **{str(k): np.asarray(v) for k, v in driver_maps.items()},
        )
    else:
        driver_path = path / "state" / "04_climate_drivers.npz"
        if driver_path.exists():
            try:
                driver_path.unlink()
            except Exception:
                pass
    meta = {
        "schema_version": SCHEMA_VERSION,
        "stage": "climate",
        "width": climate.width,
        "height": climate.height,
        "mean_land_temp_c": climate.mean_land_temp_c,
        "mean_ocean_temp_c": climate.mean_ocean_temp_c,
        "mean_land_precip_mm": climate.mean_land_precip_mm,
        "mean_ocean_precip_mm": climate.mean_ocean_precip_mm,
        "min_temp_c": climate.min_temp_c,
        "max_temp_c": climate.max_temp_c,
        "min_precip_mm": climate.min_precip_mm,
        "max_precip_mm": climate.max_precip_mm,
        "koppen_summary": climate.koppen_summary,
        "notes": climate.notes,
        "climate_mode": getattr(climate, "climate_mode", "legacy"),
        "climate_driver_map_info": getattr(climate, "climate_driver_map_info", None),
        "climate_driver_map_keys": sorted((getattr(climate, "climate_driver_maps", None) or {}).keys()),
    }
    (path / "state" / "04_climate_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def load_climate_state(output_dir: str | Path) -> ClimateMap:
    import numpy as np

    path = Path(output_dir)
    meta = json.loads((path / "state" / "04_climate_metadata.json").read_text(encoding="utf-8"))
    data = np.load(path / "state" / "04_climate.npz", allow_pickle=True)
    driver_maps = None
    driver_path = path / "state" / "04_climate_drivers.npz"
    if driver_path.exists():
        try:
            with np.load(driver_path, allow_pickle=True) as driver_data:
                # Keep driver rasters as NumPy arrays.  Some climate-overhaul
                # diagnostics are now native-resolution and monthly; converting
                # them to nested Python lists would waste memory and slow the UI.
                driver_maps = {str(k): driver_data[k].astype(int) for k in driver_data.files}
        except Exception:
            driver_maps = None
    return ClimateMap(
        width=int(meta["width"]),
        height=int(meta["height"]),
        annual_mean_temp_c_x10=data["annual_mean_temp_c_x10"].astype(int).tolist(),
        warmest_month_temp_c_x10=data["warmest_month_temp_c_x10"].astype(int).tolist(),
        coldest_month_temp_c_x10=data["coldest_month_temp_c_x10"].astype(int).tolist(),
        annual_precip_mm=data["annual_precip_mm"].astype(int).tolist(),
        koppen_classification=data["koppen_classification"].astype(str).tolist(),
        mean_land_temp_c=float(meta["mean_land_temp_c"]),
        mean_ocean_temp_c=float(meta["mean_ocean_temp_c"]),
        mean_land_precip_mm=float(meta["mean_land_precip_mm"]),
        mean_ocean_precip_mm=float(meta["mean_ocean_precip_mm"]),
        min_temp_c=float(meta["min_temp_c"]),
        max_temp_c=float(meta["max_temp_c"]),
        min_precip_mm=int(meta["min_precip_mm"]),
        max_precip_mm=int(meta["max_precip_mm"]),
        koppen_summary={str(k): int(v) for k, v in meta.get("koppen_summary", {}).items()},
        notes=list(meta.get("notes", [])),
        climate_mode=str(meta.get("climate_mode", "legacy")),
        climate_driver_maps=driver_maps,
        climate_driver_map_info=meta.get("climate_driver_map_info"),
    )


def save_hydrology_state(output_dir: str | Path, hydrology: HydrologyMap) -> None:
    import numpy as np

    path = ensure_layout(output_dir)
    np.savez_compressed(
        path / "state" / "05_hydrology.npz",
        runoff_mm=np.asarray(hydrology.runoff_mm, dtype=np.uint16),
        flow_accumulation=np.asarray(hydrology.flow_accumulation, dtype=np.uint32),
        river_intensity=np.asarray(hydrology.river_intensity, dtype=np.uint8),
        lake_mask=np.asarray(hydrology.lake_mask, dtype=np.bool_),
        drainage_basin_id=np.asarray(hydrology.drainage_basin_id, dtype=np.int32),
    )
    meta = {
        "schema_version": SCHEMA_VERSION,
        "stage": "hydrology",
        "width": hydrology.width,
        "height": hydrology.height,
        "river_cell_count": hydrology.river_cell_count,
        "lake_cell_count": hydrology.lake_cell_count,
        "max_flow_accumulation": hydrology.max_flow_accumulation,
        "river_threshold": hydrology.river_threshold,
        "estimated_major_river_count": hydrology.estimated_major_river_count,
        "drainage_basin_count": hydrology.drainage_basin_count,
        "major_drainage_basin_count": hydrology.major_drainage_basin_count,
        "coastal_basin_count": hydrology.coastal_basin_count,
        "endorheic_basin_count": hydrology.endorheic_basin_count,
        "minor_coastal_basin_cell_count": hydrology.minor_coastal_basin_cell_count,
        "delta_cell_count": hydrology.delta_cell_count,
        "notes": hydrology.notes or [],
    }
    (path / "state" / "05_hydrology_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def load_hydrology_state(output_dir: str | Path) -> HydrologyMap:
    import numpy as np

    path = Path(output_dir)
    meta = json.loads((path / "state" / "05_hydrology_metadata.json").read_text(encoding="utf-8"))
    data = np.load(path / "state" / "05_hydrology.npz", allow_pickle=True)
    return HydrologyMap(
        width=int(meta["width"]),
        height=int(meta["height"]),
        runoff_mm=data["runoff_mm"].astype(int).tolist(),
        flow_accumulation=data["flow_accumulation"].astype(int).tolist(),
        river_intensity=data["river_intensity"].astype(int).tolist(),
        lake_mask=data["lake_mask"].astype(bool).tolist(),
        drainage_basin_id=data["drainage_basin_id"].astype(int).tolist(),
        river_cell_count=int(meta["river_cell_count"]),
        lake_cell_count=int(meta["lake_cell_count"]),
        max_flow_accumulation=int(meta["max_flow_accumulation"]),
        river_threshold=int(meta["river_threshold"]),
        estimated_major_river_count=int(meta["estimated_major_river_count"]),
        drainage_basin_count=int(meta["drainage_basin_count"]),
        major_drainage_basin_count=int(meta["major_drainage_basin_count"]),
        coastal_basin_count=int(meta.get("coastal_basin_count", 0)),
        endorheic_basin_count=int(meta.get("endorheic_basin_count", 0)),
        minor_coastal_basin_cell_count=int(meta.get("minor_coastal_basin_cell_count", 0)),
        delta_cell_count=int(meta.get("delta_cell_count", 0)),
        notes=list(meta.get("notes", [])),
    )


def save_biome_state(output_dir: str | Path, biomes: BiomeMap) -> None:
    import numpy as np

    path = ensure_layout(output_dir)
    np.savez_compressed(path / "state" / "06_biomes.npz", biome_classification=np.asarray(biomes.biome_classification))
    meta = {
        "schema_version": SCHEMA_VERSION,
        "stage": "biomes",
        "width": biomes.width,
        "height": biomes.height,
        "biome_summary": biomes.biome_summary,
        "dominant_biome": biomes.dominant_biome,
        "land_biome_count": biomes.land_biome_count,
        "notes": biomes.notes,
    }
    (path / "state" / "06_biomes_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def load_biome_state(output_dir: str | Path) -> BiomeMap:
    import numpy as np

    path = Path(output_dir)
    meta = json.loads((path / "state" / "06_biomes_metadata.json").read_text(encoding="utf-8"))
    data = np.load(path / "state" / "06_biomes.npz", allow_pickle=True)
    return BiomeMap(
        width=int(meta["width"]),
        height=int(meta["height"]),
        biome_classification=data["biome_classification"].astype(str).tolist(),
        biome_summary={str(k): int(v) for k, v in meta.get("biome_summary", {}).items()},
        dominant_biome=str(meta.get("dominant_biome", "unknown")),
        land_biome_count=int(meta.get("land_biome_count", 0)),
        notes=list(meta.get("notes", [])),
    )


def save_regions_state(output_dir: str | Path, regions: RegionAnalysis) -> None:
    path = ensure_layout(output_dir) / "state" / "07_regions.json"
    data = {
        "schema_version": SCHEMA_VERSION,
        "stage": "regions",
        "rows": regions.rows,
        "cols": regions.cols,
        "regions": [r.to_dict() for r in regions.regions],
        "top_productive_region_ids": regions.top_productive_region_ids,
        "notes": regions.notes,
    }
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def load_regions_state(output_dir: str | Path) -> RegionAnalysis:
    data = json.loads((Path(output_dir) / "state" / "07_regions.json").read_text(encoding="utf-8"))
    return RegionAnalysis(
        rows=int(data["rows"]),
        cols=int(data["cols"]),
        regions=[RegionSummary(**r) for r in data.get("regions", [])],
        top_productive_region_ids=list(data.get("top_productive_region_ids", [])),
        notes=list(data.get("notes", [])),
    )


# ---------------------------------------------------------------------------
# Placeholder maps for partial-profile visualization
# ---------------------------------------------------------------------------


def placeholder_climate(terrain: TerrainMap) -> ClimateMap:
    zeros = [[0 for _ in range(terrain.width)] for _ in range(terrain.height)]
    koppen = [["--" for _ in range(terrain.width)] for _ in range(terrain.height)]
    return ClimateMap(
        width=terrain.width,
        height=terrain.height,
        annual_mean_temp_c_x10=zeros,
        warmest_month_temp_c_x10=zeros,
        coldest_month_temp_c_x10=zeros,
        annual_precip_mm=zeros,
        koppen_classification=koppen,
        mean_land_temp_c=math.nan,
        mean_ocean_temp_c=math.nan,
        mean_land_precip_mm=math.nan,
        mean_ocean_precip_mm=math.nan,
        min_temp_c=math.nan,
        max_temp_c=math.nan,
        min_precip_mm=0,
        max_precip_mm=0,
        koppen_summary={},
        notes=["Placeholder climate used for partial-stage visualization."],
        climate_mode="placeholder",
    )


def placeholder_hydrology(terrain: TerrainMap) -> HydrologyMap:
    zeros = [[0 for _ in range(terrain.width)] for _ in range(terrain.height)]
    falses = [[False for _ in range(terrain.width)] for _ in range(terrain.height)]
    return HydrologyMap(
        width=terrain.width,
        height=terrain.height,
        runoff_mm=zeros,
        flow_accumulation=zeros,
        river_intensity=zeros,
        lake_mask=falses,
        drainage_basin_id=zeros,
        river_cell_count=0,
        lake_cell_count=0,
        max_flow_accumulation=0,
        river_threshold=0,
        estimated_major_river_count=0,
        drainage_basin_count=0,
        major_drainage_basin_count=0,
        notes=["Placeholder hydrology used for partial-stage visualization."],
    )


def placeholder_biomes(terrain: TerrainMap) -> BiomeMap:
    grid = [["ocean" if not terrain.is_land[r][c] else "unclassified land" for c in range(terrain.width)] for r in range(terrain.height)]
    return BiomeMap(
        width=terrain.width,
        height=terrain.height,
        biome_classification=grid,
        biome_summary={"ocean": int(terrain.ocean_fraction * terrain.width * terrain.height), "unclassified land": int(terrain.land_fraction * terrain.width * terrain.height)},
        dominant_biome="unclassified land",
        land_biome_count=int(terrain.land_fraction * terrain.width * terrain.height),
        notes=["Placeholder biomes used for partial-stage visualization."],
    )


def placeholder_regions() -> RegionAnalysis:
    return RegionAnalysis(rows=0, cols=0, regions=[], top_productive_region_ids=[], notes=["Placeholder regions used for partial-stage visualization."])


def assemble_available_system(output_dir: str | Path) -> StarSystem:
    system = load_system_state(output_dir)
    if not stage_exists(output_dir, "planet-physics") or not stage_exists(output_dir, "terrain"):
        return system
    rotation, atmosphere, hydrosphere, geology = load_physics_state(output_dir)
    terrain = load_terrain_state(output_dir)
    climate = load_climate_state(output_dir) if stage_exists(output_dir, "climate") else placeholder_climate(terrain)
    hydrology = load_hydrology_state(output_dir) if stage_exists(output_dir, "hydrology") else placeholder_hydrology(terrain)
    biomes = load_biome_state(output_dir) if stage_exists(output_dir, "biomes") else placeholder_biomes(terrain)
    regions = load_regions_state(output_dir) if stage_exists(output_dir, "regions") else placeholder_regions()
    main = system.main_planet
    if main is None:
        return system
    system.main_planet_profile = MainPlanetProfile(
        planet_name=main.name,
        rotation=rotation,
        atmosphere=atmosphere,
        hydrosphere=hydrosphere,
        geology=geology,
        terrain=terrain,
        climate=climate,
        hydrology=hydrology,
        biomes=biomes,
        regions=regions,
        notes=["Assembled from staged pipeline state."],
    )
    return system


# ---------------------------------------------------------------------------
# Stage dependency / staleness / validation helpers
# ---------------------------------------------------------------------------


def stage_file_paths(output_dir: str | Path, stage: str) -> list[Path]:
    path = Path(output_dir)
    return [path / rel for rel in STAGE_FILES[normalize_stage(stage)]]


def stage_dependency_stages(stage: str) -> list[str]:
    """Return stage dependencies for staleness checks.

    Dependencies are intentionally conservative.  Manual edits to earlier JSON
    state files should make downstream stages stale so users do not accidentally
    keep climate/hydrology/biomes from an older planet or terrain state.
    """
    stage = normalize_stage(stage)
    if stage == "solar-system":
        return []
    if stage == "planet-physics":
        return ["solar-system"]
    if stage in TERRAIN_SUBPHASES:
        i = TERRAIN_SUBPHASES.index(stage)
        return ["planet-physics"] if i == 0 else [TERRAIN_SUBPHASES[i - 1]]
    if stage == "climate":
        return [FINAL_TERRAIN_STAGE, "planet-physics"]
    if stage == "hydrology":
        return ["climate", FINAL_TERRAIN_STAGE]
    if stage == "biomes":
        return ["hydrology", "climate", FINAL_TERRAIN_STAGE]
    if stage == "regions":
        return ["biomes", "hydrology", "climate", FINAL_TERRAIN_STAGE]
    if stage == "outputs":
        return ["regions", "biomes", "hydrology", "climate", FINAL_TERRAIN_STAGE]
    return []


def _existing_mtimes(paths: list[Path]) -> list[float]:
    return [p.stat().st_mtime for p in paths if p.exists()]


def newest_stage_input_mtime(output_dir: str | Path, stage: str) -> float | None:
    path = Path(output_dir)
    mtimes: list[float] = []
    # The resolved config is a global input. If it is manually edited after a
    # stage completes, that stage should be treated as stale. This is safer than
    # trying to infer which exact config field matters before we have a full
    # dependency metadata system.
    config_path = path / "config" / "resolved_config.json"
    if config_path.exists():
        mtimes.append(config_path.stat().st_mtime)
    # Visual editor changes are first stored in config/stage_overrides.json.
    # Planet-physics and later stages must become stale when that file changes,
    # otherwise rerunning from terrain can silently reuse the old physics JSON.
    # Do not use this for solar-system itself; a hydrosphere edit should not
    # force a new star/system unless the user explicitly rerolls Stage 1.
    override_path = path / "config" / "stage_overrides.json"
    if stage != "solar-system" and override_path.exists():
        mtimes.append(override_path.stat().st_mtime)
    for dep in stage_dependency_stages(stage):
        mtimes.extend(_existing_mtimes(stage_file_paths(path, dep)))
    return max(mtimes) if mtimes else None


def stage_output_mtime(output_dir: str | Path, stage: str) -> float | None:
    mtimes = _existing_mtimes(stage_file_paths(output_dir, stage))
    return max(mtimes) if mtimes else None


def stage_stale_reason(output_dir: str | Path, stage: str) -> str:
    stage = normalize_stage(stage)
    if not stage_exists(output_dir, stage):
        return ""
    for dep in stage_dependency_stages(stage):
        if stage_exists(output_dir, dep) and stage_is_stale(output_dir, dep):
            return f"dependency {dep} is stale"
    own_mtime = stage_output_mtime(output_dir, stage)
    input_mtime = newest_stage_input_mtime(output_dir, stage)
    if own_mtime is None or input_mtime is None:
        return ""
    # Use a tiny tolerance for filesystems with coarse timestamp precision.
    if input_mtime > own_mtime + 0.001:
        return "input changed after this stage completed"
    return ""


def stage_is_stale(output_dir: str | Path, stage: str) -> bool:
    return bool(stage_stale_reason(output_dir, stage))


def stage_needs_rerun(output_dir: str | Path, stage: str) -> bool:
    stage = normalize_stage(stage)
    return (not stage_exists(output_dir, stage)) or stage_is_stale(output_dir, stage)


def _override_has_active_values(value: Any) -> bool:
    if isinstance(value, dict):
        for key, child in value.items():
            if str(key).startswith("_"):
                continue
            if _override_has_active_values(child):
                return True
        return False
    if isinstance(value, list):
        return any(_override_has_active_values(child) for child in value)
    return value is not None


def unapplied_override_reason(output_dir: str | Path) -> str:
    path = Path(output_dir)
    override_path = path / "config" / "stage_overrides.json"
    report_path = path / "diagnostics" / "override_application_report.json"
    if not override_path.exists():
        return ""
    try:
        overrides = json.loads(override_path.read_text(encoding="utf-8"))
    except Exception:
        return "override file cannot be parsed"
    if not _override_has_active_values(overrides):
        return ""
    if not report_path.exists():
        return "override file has active values that have not been applied"
    if override_path.stat().st_mtime > report_path.stat().st_mtime + 0.001:
        return "override file changed after last apply-overrides"
    return ""


def status_detail_rows(output_dir: str | Path) -> list[dict[str, str]]:
    manifest = read_manifest(output_dir)
    stages = manifest.get("stages", {})
    rows: list[dict[str, str]] = []
    for stage in STAGE_ORDER:
        exists = stage_exists(output_dir, stage)
        state = stages.get(stage, {})
        if exists:
            status = "stale" if stage_is_stale(output_dir, stage) else "complete"
        else:
            status = "missing"
            if state.get("status"):
                status = f"{state.get('status')} (files missing)"
        rows.append(
            {
                "stage": stage,
                "status": status,
                "completed_at": state.get("completed_at") or "",
                "reason": stage_stale_reason(output_dir, stage),
            }
        )
    return rows


def write_status_report(output_dir: str | Path) -> None:
    path = ensure_layout(output_dir)
    rows = status_detail_rows(path)
    override_reason = unapplied_override_reason(path)
    report = {
        "schema_version": SCHEMA_VERSION,
        "pipeline_version": PIPELINE_VERSION,
        "generated_at": now_stamp(),
        "output_dir": str(path),
        "unapplied_overrides": override_reason,
        "stages": rows,
    }
    (path / "diagnostics" / "pipeline_status.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    csv_path = path / "diagnostics" / "pipeline_status.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["stage", "status", "completed_at", "reason"])
        writer.writeheader()
        writer.writerows(rows)


# Preserve the original public function name, but include stale status.
def status_rows(output_dir: str | Path) -> list[tuple[str, str, str, str]]:  # type: ignore[override]
    return [(row["stage"], row["status"], row["completed_at"], row["reason"]) for row in status_detail_rows(output_dir)]


def validate_staged_run(output_dir: str | Path) -> list[str]:
    """Validate editable staged state and write a human-readable report."""
    path = ensure_layout(output_dir)
    warnings: list[str] = []

    # Config.
    config_path = path / "config" / "resolved_config.json"
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
        pp = config.get("planet_profile", {})
        width = int(pp.get("map_width", 0))
        height = int(pp.get("map_height", 0))
        if width <= 0 or height <= 0:
            warnings.append("config/resolved_config.json: map_width and map_height must be positive.")
        if width != height * 2:
            warnings.append("config/resolved_config.json: map_width is usually expected to be exactly 2 × map_height for equirectangular maps.")
    except Exception as exc:
        warnings.append(f"config/resolved_config.json: could not read config: {exc}")

    # Solar-system state.
    system_path = path / "state" / "01_solar_system.json"
    if system_path.exists():
        try:
            system = json.loads(system_path.read_text(encoding="utf-8"))
            planets = system.get("planets", [])
            main = [p for p in planets if p.get("is_main_planet")]
            if len(main) != 1:
                warnings.append(f"state/01_solar_system.json: expected exactly one is_main_planet=true, found {len(main)}.")
            for p in planets:
                name = p.get("name", "<unnamed>")
                if float(p.get("mass_earth", 0)) <= 0:
                    warnings.append(f"state/01_solar_system.json: planet {name} has non-positive mass_earth.")
                if float(p.get("radius_earth", 0)) <= 0:
                    warnings.append(f"state/01_solar_system.json: planet {name} has non-positive radius_earth.")
                orbit = p.get("orbit", {})
                if float(orbit.get("semi_major_axis_au", 0)) <= 0:
                    warnings.append(f"state/01_solar_system.json: planet {name} has non-positive semi_major_axis_au.")
        except Exception as exc:
            warnings.append(f"state/01_solar_system.json: could not validate: {exc}")

    # Physics state.
    physics_path = path / "state" / "02_planet_physics.json"
    if physics_path.exists():
        try:
            physics = json.loads(physics_path.read_text(encoding="utf-8"))
            rotation = physics.get("rotation", {})
            atmosphere = physics.get("atmosphere", {})
            hydrosphere = physics.get("hydrosphere", {})
            if float(rotation.get("rotation_period_hours", 0)) <= 0:
                warnings.append("state/02_planet_physics.json: rotation_period_hours must be positive.")
            tilt = float(rotation.get("axial_tilt_degrees", 0))
            if not (0 <= tilt <= 90):
                warnings.append("state/02_planet_physics.json: axial_tilt_degrees should be between 0 and 90.")
            if float(atmosphere.get("pressure_bar", 0)) <= 0:
                warnings.append("state/02_planet_physics.json: pressure_bar must be positive.")
            ocean_target = float(hydrosphere.get("ocean_fraction_target", 0))
            if not (0 <= ocean_target <= 1):
                warnings.append("state/02_planet_physics.json: ocean_fraction_target should be between 0 and 1.")
        except Exception as exc:
            warnings.append(f"state/02_planet_physics.json: could not validate: {exc}")

    # Terrain metadata.
    terrain_meta = path / "state" / "03_terrain_metadata.json"
    if terrain_meta.exists():
        try:
            meta = json.loads(terrain_meta.read_text(encoding="utf-8"))
            if int(meta.get("width", 0)) <= 0 or int(meta.get("height", 0)) <= 0:
                warnings.append("state/03_terrain_metadata.json: terrain width/height must be positive.")
            ocean = float(meta.get("ocean_fraction", -1))
            land = float(meta.get("land_fraction", -1))
            if not (0 <= ocean <= 1 and 0 <= land <= 1):
                warnings.append("state/03_terrain_metadata.json: land/ocean fractions should be between 0 and 1.")
            if abs((ocean + land) - 1.0) > 0.02:
                warnings.append("state/03_terrain_metadata.json: land_fraction + ocean_fraction does not sum close to 1.")
        except Exception as exc:
            warnings.append(f"state/03_terrain_metadata.json: could not validate: {exc}")

    # Status/staleness warnings.
    for row in status_detail_rows(path):
        if row["status"] == "stale":
            warnings.append(f"{row['stage']}: stale because {row['reason']}.")
    override_reason = unapplied_override_reason(path)
    if override_reason:
        warnings.append(f"config/stage_overrides.json: {override_reason}.")

    lines = [
        "WorldGen staged-run validation",
        "==============================",
        "",
        f"Generated at: {now_stamp()}",
        f"Output directory: {path}",
        "",
    ]
    if warnings:
        lines.append("Warnings / action items:")
        lines.extend(f"  - {w}" for w in warnings)
    else:
        lines.append("No validation warnings found.")
    (path / "diagnostics" / "pipeline_validation_report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_status_report(path)
    return warnings
