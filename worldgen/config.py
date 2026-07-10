"""Configuration loading and normalization."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class StarConfig:
    stellar_class: str | None = None
    mass_solar: float | None = None
    age_gyr: float | None = None
    metallicity: float | None = None


@dataclass(frozen=True)
class SystemConfig:
    planet_count: int | None = None
    min_planets: int = 4
    max_planets: int = 10
    architecture_type: str | None = None
    main_planet_preference: str = "earthlike"
    require_major_moon: bool = True
    moon_strength_preference: str = "moderate"


@dataclass(frozen=True)
class PlanetProfileConfig:
    map_width: int = 1024
    map_height: int = 512
    min_map_width: int = 1024
    min_map_height: int = 512
    generate_hydrology: bool = True
    generate_biomes: bool = True
    generate_regions: bool = True
    fast_mode: bool = False
    no_accelerated_terrain: bool = True
    no_accelerated_climate: bool = True
    koppen_detail: str = "local4"
    # Climate backend selection. seasonal_v1 is the stable first overhaul backend;
    # seasonal_v2 is the preserved structured-atmosphere review mode; seasonal_v3
    # is the first structured atmosphere + basin-aware ocean-current review mode;
    # seasonal_v4 refines basin routing/current heat/upwelling/coastal feedback;
    # seasonal_v5 adds component-based moisture/rainfall coupling on top of v4;
    # legacy preserves the previous annual heuristic model for comparison.
    climate_generation_mode: str = "seasonal_v1"
    # Terrain backend selection. procedural_legacy keeps the current proven
    # generator. plate_tectonic_v1 now creates native plate setup, motion vectors, ocean-floor diagnostics,
    # relative-boundary diagnostics, and first plate-derived continental relief
    # while preserving downstream compatibility until later updates replace full Stage 3 terrain ownership.
    # real_world_stage3 is used by
    # Earth presets that source terrain from real-world data through Stage 3.
    terrain_generation_mode: str = "plate_history_v4"
    # Optional terrain-quality control: bias generated crust away from high latitudes.
    # In plate_history_v1/v2 this is applied as a smooth crust-potential penalty during
    # land formation; in plate_tectonic_v1 older cleanup passes may still use it as
    # a final corrective option.
    suppress_polar_land: bool = False
    # Time-evolved plate-history terrain mode controls.  These are ignored by the
    # legacy/procedural backend and by real-world Earth presets.
    tectonic_history_myr: float | None = None
    tectonic_timestep_myr: float = 2.5
    # plate_history_v1/v2 simulation-grid controls. The default "legacy" exactly
    # preserves the Update 17 internal grid choice; higher settings are opt-in.
    tectonic_grid_scale: str = "legacy"  # legacy|preview|normal|high|native|custom
    # stable = macro plate-history grid plus full-resolution detail (recommended).
    # raw is a developer-only research path and is not exposed in the normal UI.
    tectonic_grid_policy: str = "stable"  # stable|raw
    allow_experimental_tectonic_grid: bool = False
    tectonic_grid_width: int | None = None
    tectonic_grid_height: int | None = None


@dataclass(frozen=True)
class WorldGenConfig:
    seed: int | None = None
    star: StarConfig = StarConfig()
    system: SystemConfig = SystemConfig()
    planet_profile: PlanetProfileConfig = PlanetProfileConfig()


def load_config(path: str | Path | None) -> WorldGenConfig:
    if path is None:
        return WorldGenConfig()

    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return WorldGenConfig(
        seed=data.get("seed"),
        star=StarConfig(**data.get("star", {})),
        system=SystemConfig(**data.get("system", {})),
        planet_profile=PlanetProfileConfig(**data.get("planet_profile", {})),
    )


def merge_cli_overrides(config: WorldGenConfig, overrides: dict[str, Any]) -> WorldGenConfig:
    """Apply simple command-line overrides without mutating the original config."""
    seed = overrides.get("seed", config.seed)
    planet_count = overrides.get("planet_count", config.system.planet_count)
    map_width = overrides.get("map_width") if overrides.get("map_width") is not None else config.planet_profile.map_width
    map_height = overrides.get("map_height") if overrides.get("map_height") is not None else config.planet_profile.map_height
    generate_hydrology = overrides.get("generate_hydrology", config.planet_profile.generate_hydrology)
    generate_biomes = overrides.get("generate_biomes", config.planet_profile.generate_biomes)
    generate_regions = overrides.get("generate_regions", config.planet_profile.generate_regions)
    fast_mode = overrides.get("fast_mode", config.planet_profile.fast_mode)
    no_accelerated_terrain = overrides.get("no_accelerated_terrain", config.planet_profile.no_accelerated_terrain)
    no_accelerated_climate = overrides.get("no_accelerated_climate", config.planet_profile.no_accelerated_climate)
    koppen_detail = overrides.get("koppen_detail") if overrides.get("koppen_detail") is not None else config.planet_profile.koppen_detail
    climate_generation_mode = overrides.get("climate_generation_mode") if overrides.get("climate_generation_mode") is not None else config.planet_profile.climate_generation_mode
    terrain_generation_mode = overrides.get("terrain_generation_mode") if overrides.get("terrain_generation_mode") is not None else config.planet_profile.terrain_generation_mode
    suppress_polar_land = overrides.get("suppress_polar_land") if overrides.get("suppress_polar_land") is not None else config.planet_profile.suppress_polar_land
    tectonic_history_myr = overrides.get("tectonic_history_myr") if overrides.get("tectonic_history_myr") is not None else config.planet_profile.tectonic_history_myr
    tectonic_timestep_myr = overrides.get("tectonic_timestep_myr") if overrides.get("tectonic_timestep_myr") is not None else config.planet_profile.tectonic_timestep_myr
    tectonic_grid_scale = overrides.get("tectonic_grid_scale") if overrides.get("tectonic_grid_scale") is not None else config.planet_profile.tectonic_grid_scale
    tectonic_grid_policy = overrides.get("tectonic_grid_policy") if overrides.get("tectonic_grid_policy") is not None else config.planet_profile.tectonic_grid_policy
    allow_experimental_tectonic_grid = overrides.get("allow_experimental_tectonic_grid") if overrides.get("allow_experimental_tectonic_grid") is not None else config.planet_profile.allow_experimental_tectonic_grid
    tectonic_grid_width = overrides.get("tectonic_grid_width") if overrides.get("tectonic_grid_width") is not None else config.planet_profile.tectonic_grid_width
    tectonic_grid_height = overrides.get("tectonic_grid_height") if overrides.get("tectonic_grid_height") is not None else config.planet_profile.tectonic_grid_height

    def override_or_current(key: str, current: Any) -> Any:
        value = overrides.get(key)
        return current if value is None else value

    return WorldGenConfig(
        seed=seed,
        star=config.star,
        system=SystemConfig(
            planet_count=planet_count,
            min_planets=config.system.min_planets,
            max_planets=config.system.max_planets,
            architecture_type=override_or_current("architecture_type", config.system.architecture_type),
            main_planet_preference=override_or_current("main_planet_preference", config.system.main_planet_preference),
            require_major_moon=override_or_current("require_major_moon", config.system.require_major_moon),
            moon_strength_preference=override_or_current("moon_strength_preference", config.system.moon_strength_preference),
        ),
        planet_profile=PlanetProfileConfig(
            map_width=map_width,
            map_height=map_height,
            min_map_width=config.planet_profile.min_map_width,
            min_map_height=config.planet_profile.min_map_height,
            generate_hydrology=generate_hydrology,
            generate_biomes=generate_biomes,
            generate_regions=generate_regions,
            fast_mode=fast_mode,
            no_accelerated_terrain=no_accelerated_terrain,
            no_accelerated_climate=no_accelerated_climate,
            koppen_detail=koppen_detail,
            climate_generation_mode=str(climate_generation_mode or config.planet_profile.climate_generation_mode),
            terrain_generation_mode=terrain_generation_mode,
            suppress_polar_land=bool(suppress_polar_land),
            tectonic_history_myr=tectonic_history_myr,
            tectonic_timestep_myr=float(tectonic_timestep_myr if tectonic_timestep_myr is not None else config.planet_profile.tectonic_timestep_myr),
            tectonic_grid_scale=str(tectonic_grid_scale or config.planet_profile.tectonic_grid_scale),
            tectonic_grid_policy=str(tectonic_grid_policy or config.planet_profile.tectonic_grid_policy),
            allow_experimental_tectonic_grid=bool(allow_experimental_tectonic_grid),
            tectonic_grid_width=int(tectonic_grid_width) if tectonic_grid_width not in (None, "") else None,
            tectonic_grid_height=int(tectonic_grid_height) if tectonic_grid_height not in (None, "") else None,
        ),
    )
