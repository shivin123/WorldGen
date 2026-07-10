"""Generate the first deep-dive physical profile for the Main Planet.

This module does not yet run a real climate model. It creates the physical
foundation that later terrain, geology, and climate systems will use.
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import replace

from worldgen.config import PlanetProfileConfig
from worldgen.generators.climate_generator import generate_climate
from worldgen.generators.hydrology_generator import generate_hydrology
from worldgen.generators.biome_generator import generate_biomes
from worldgen.generators.region_generator import generate_regions
from worldgen.models.bodies import Planet, Star
from worldgen.models.planet_profile import (
    Atmosphere,
    BiomeMap,
    GeologyState,
    HydrologyMap,
    Hydrosphere,
    MainPlanetProfile,
    RegionAnalysis,
    RotationState,
    TerrainMap,
)
from worldgen.random_utils import clamp
from worldgen import performance


def generate_main_planet_physical_states(
    rng: random.Random,
    star: Star,
    planet: Planet,
    config: PlanetProfileConfig,
) -> tuple[RotationState, Atmosphere, Hydrosphere, GeologyState]:
    """Generate the editable physical states that feed terrain and climate.

    This is a public stage wrapper for the pipeline runner. Keeping it separate
    lets a user inspect/edit rotation, atmosphere, hydrosphere, and geology
    before committing to the expensive terrain stage.
    """
    _progress("Generating rotation/atmosphere/hydrosphere/geology...")
    rotation = _generate_rotation(rng, planet)
    atmosphere = _generate_atmosphere(rng, planet)
    hydrosphere = _generate_hydrosphere(rng, planet, atmosphere)
    geology = _generate_geology(rng, star, planet)
    return rotation, atmosphere, hydrosphere, geology


def generate_terrain_stage(
    rng: random.Random,
    planet: Planet,
    hydrosphere: Hydrosphere,
    geology: GeologyState,
    config: PlanetProfileConfig,
    *,
    output_dir: str | None = None,
) -> TerrainMap:
    """Generate the expensive terrain/geology raster stage.

    Update45 exposes this as a standalone pipeline stage. Internally the terrain
    algorithm still contains several sub-passes; the pipeline names those
    sub-phases so future updates can split and persist them one at a time.
    """
    terrain_start = time.perf_counter()
    _progress(f"Generating terrain grid {config.map_width} x {config.map_height}...")
    terrain = _generate_terrain(rng, planet, hydrosphere, geology, config, output_dir=output_dir)
    terrain_elapsed = time.perf_counter() - terrain_start
    performance.record_stage("generate terrain grid", terrain_elapsed)
    _progress(f"Terrain grid complete in {terrain_elapsed:.1f}s.")
    hydrosphere.ocean_fraction_actual = terrain.ocean_fraction
    return terrain


def generate_climate_stage(
    rotation: RotationState,
    atmosphere: Atmosphere,
    terrain: TerrainMap,
    config: PlanetProfileConfig,
) -> ClimateMap:
    climate_start = time.perf_counter()
    _progress(f"Generating climate, winds, and ocean-current effects ({config.climate_generation_mode})...")
    climate = generate_climate(
        rotation,
        atmosphere,
        terrain,
        use_accelerated=not config.no_accelerated_climate,
        koppen_detail=config.koppen_detail,
        climate_mode=config.climate_generation_mode,
    )
    climate_elapsed = time.perf_counter() - climate_start
    performance.record_stage("generate climate", climate_elapsed)
    _progress(f"Climate complete in {climate_elapsed:.1f}s.")
    return climate


def generate_hydrology_stage(terrain: TerrainMap, climate: ClimateMap, config: PlanetProfileConfig) -> HydrologyMap:
    if not config.generate_hydrology:
        return _empty_hydrology(terrain)
    hydrology_start = time.perf_counter()
    _progress("Generating hydrology, rivers, basins, and delta candidates...")
    hydrology = generate_hydrology(terrain, climate)
    delta_cells = _apply_delta_deposition(terrain, hydrology)
    if delta_cells > 0:
        _progress(f"Deposited {delta_cells:,} new/raised river-mouth delta cells.")
        hydrology.delta_cell_count = max(getattr(hydrology, "delta_cell_count", 0), delta_cells)
    hydrology_elapsed = time.perf_counter() - hydrology_start
    performance.record_stage("generate hydrology and delta deposition", hydrology_elapsed)
    _progress(f"Hydrology complete in {hydrology_elapsed:.1f}s.")
    return hydrology


def generate_biome_stage(terrain: TerrainMap, climate: ClimateMap, hydrology: HydrologyMap, config: PlanetProfileConfig) -> BiomeMap:
    if not config.generate_biomes:
        return _empty_biomes(terrain)
    biome_start = time.perf_counter()
    _progress("Generating biomes...")
    biomes = generate_biomes(terrain, climate, hydrology)
    biome_elapsed = time.perf_counter() - biome_start
    performance.record_stage("generate biomes", biome_elapsed)
    _progress(f"Biomes complete in {biome_elapsed:.1f}s.")
    return biomes


def generate_region_stage(terrain: TerrainMap, climate: ClimateMap, hydrology: HydrologyMap, biomes: BiomeMap, config: PlanetProfileConfig) -> RegionAnalysis:
    if not config.generate_regions:
        return _empty_regions()
    region_start = time.perf_counter()
    _progress("Generating regional summaries...")
    regions = generate_regions(terrain, climate, hydrology, biomes)
    region_elapsed = time.perf_counter() - region_start
    performance.record_stage("generate regional summaries", region_elapsed)
    _progress(f"Regional summaries complete in {region_elapsed:.1f}s.")
    return regions


def assemble_main_planet_profile(
    planet_name: str,
    rotation: RotationState,
    atmosphere: Atmosphere,
    hydrosphere: Hydrosphere,
    geology: GeologyState,
    terrain: TerrainMap,
    climate: ClimateMap,
    hydrology: HydrologyMap,
    biomes: BiomeMap,
    regions: RegionAnalysis,
) -> MainPlanetProfile:
    notes = [
        "Land-stage life and atmospheric oxygen are assumed by project scope.",
        "Terrain is procedural and uses a plate-inspired uplift/rift/crust diagnostic layer; it is not a full geologic-time plate reconstruction.",
        f"Terrain source/mode: {terrain.source}.",
        "Climate is long-term average climate only; weather is not simulated.",
    ]
    return MainPlanetProfile(
        planet_name=planet_name,
        rotation=rotation,
        atmosphere=atmosphere,
        hydrosphere=hydrosphere,
        geology=geology,
        terrain=terrain,
        climate=climate,
        hydrology=hydrology,
        biomes=biomes,
        regions=regions,
        notes=notes,
    )


def generate_main_planet_profile(
    rng: random.Random,
    star: Star,
    planet: Planet,
    config: PlanetProfileConfig,
) -> MainPlanetProfile:
    """Generate rotation, atmosphere, hydrosphere, geology, terrain, and downstream maps."""
    rotation, atmosphere, hydrosphere, geology = generate_main_planet_physical_states(rng, star, planet, config)
    terrain = generate_terrain_stage(rng, planet, hydrosphere, geology, config)
    climate = generate_climate_stage(rotation, atmosphere, terrain, config)
    hydrology = generate_hydrology_stage(terrain, climate, config)
    biomes = generate_biome_stage(terrain, climate, hydrology, config)
    regions = generate_region_stage(terrain, climate, hydrology, biomes, config)
    return assemble_main_planet_profile(
        planet.name, rotation, atmosphere, hydrosphere, geology, terrain, climate, hydrology, biomes, regions
    )



def _progress(message: str) -> None:
    performance.mark(message)
    print(f"[progress] {message}", flush=True)


def _empty_hydrology(terrain: TerrainMap) -> HydrologyMap:
    width = terrain.width
    height = terrain.height
    zeros = [[0 for _ in range(width)] for _ in range(height)]
    falses = [[False for _ in range(width)] for _ in range(height)]
    return HydrologyMap(
        width=width,
        height=height,
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
        coastal_basin_count=0,
        endorheic_basin_count=0,
        minor_coastal_basin_cell_count=0,
        delta_cell_count=0,
        notes=["Hydrology skipped by CLI/config option."],
    )


def _empty_biomes(terrain: TerrainMap) -> BiomeMap:
    grid = [["ocean" if not terrain.is_land[r][c] else "unclassified land" for c in range(terrain.width)] for r in range(terrain.height)]
    summary: dict[str, int] = {}
    for row in grid:
        for value in row:
            summary[value] = summary.get(value, 0) + 1
    return BiomeMap(
        width=terrain.width,
        height=terrain.height,
        biome_classification=grid,
        biome_summary=summary,
        dominant_biome="unclassified land",
        land_biome_count=summary.get("unclassified land", 0),
        notes=["Biomes skipped by CLI/config option."],
    )


def _empty_regions() -> RegionAnalysis:
    return RegionAnalysis(
        rows=0,
        cols=0,
        regions=[],
        top_productive_region_ids=[],
        notes=["Region analysis skipped by CLI/config option."],
    )


def _apply_delta_deposition(terrain: TerrainMap, hydrology: HydrologyMap) -> int:
    """Create conservative, fan-shaped delta/floodplain deposition.

    Previous builds let every strong coastal river raise nearby shallow shelf
    cells. On broad shelves this created trapezoid-like coastal plains with many
    parallel drainage basins. This version first identifies distinct river
    mouths, keeps only the strongest outlets, and builds a small irregular fan in
    the seaward direction. Most sediment now raises floodplains/estuaries rather
    than converting long shelf strips into new land.
    """
    width = terrain.width
    height = terrain.height

    mouths: list[tuple[int, int, int, float, float]] = []
    for r in range(height):
        for c in range(width):
            intensity = hydrology.river_intensity[r][c]
            if not terrain.is_land[r][c] or intensity < 205:
                continue
            seaward_r = 0.0
            seaward_c = 0.0
            ocean_count = 0
            for dr in (-1, 0, 1):
                rr = r + dr
                if rr < 0 or rr >= height:
                    continue
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    cc = (c + dc) % width
                    if not terrain.is_land[rr][cc]:
                        ocean_count += 1
                        seaward_r += dr
                        seaward_c += dc
            if ocean_count:
                norm = max(1.0e-6, (seaward_r * seaward_r + seaward_c * seaward_c) ** 0.5)
                mouths.append((intensity, r, c, seaward_r / norm, seaward_c / norm))

    if not mouths:
        return 0

    mouths.sort(reverse=True, key=lambda x: x[0])
    # Keep delta-building rare and visually meaningful. Other mouths are
    # estuaries/floodplains, not new protruding land.
    max_mouths = max(8, min(48, (width * height) // 180_000))
    selected: list[tuple[int, int, int, float, float]] = []
    occupied: set[tuple[int, int]] = set()
    separation = max(6, min(28, width // 180))
    for item in mouths:
        _intensity, r, c, _sr, _sc = item
        key = (r // separation, c // separation)
        if key in occupied:
            continue
        selected.append(item)
        occupied.add(key)
        if len(selected) >= max_mouths:
            break

    candidates: dict[tuple[int, int], int] = {}
    for intensity, r, c, seaward_r, seaward_c in selected:
        radius = 1
        if intensity >= 220:
            radius = 2
        if intensity >= 238:
            radius = 3
        if intensity >= 248:
            radius = 4
        # Only very shallow shelves can become new delta land.
        shelf_limit = max(-180, -45 - max(0, intensity - 205) * 0.9)

        for dr in range(-radius, radius + 1):
            rr = r + dr
            if rr < 0 or rr >= height:
                continue
            for dc in range(-radius, radius + 1):
                cc = (c + dc) % width
                dist = (dr * dr + dc * dc) ** 0.5
                if dist > radius + 0.45:
                    continue
                # Fan is biased seaward. Landward cells become floodplain only.
                dot = dr * seaward_r + dc * seaward_c
                if terrain.is_land[rr][cc]:
                    if dot <= 0.7 and 0 < terrain.elevation_m[rr][cc] < 120:
                        candidates[(rr, cc)] = max(candidates.get((rr, cc), 0), 8)
                    continue
                if dot < 0.15:
                    continue
                depth = terrain.elevation_m[rr][cc]
                if depth >= shelf_limit:
                    taper = max(0.0, 1.0 - dist / (radius + 0.55))
                    build = int(2 + taper * 16 + (intensity - 205) / 18)
                    candidates[(rr, cc)] = max(candidates.get((rr, cc), 0), build)

    if not candidates:
        return 0

    new_land = 0
    raised_land = 0
    for (rr, cc), build in candidates.items():
        if terrain.is_land[rr][cc]:
            terrain.elevation_m[rr][cc] = max(terrain.elevation_m[rr][cc], min(28, terrain.elevation_m[rr][cc] + build))
            raised_land += 1
        else:
            # New delta land is low and local; it should not produce broad,
            # trapezoid shelf plains.
            terrain.is_land[rr][cc] = True
            terrain.elevation_m[rr][cc] = max(1, min(18, build))
            new_land += 1

    land_values: list[int] = []
    ocean_values: list[int] = []
    land_count = 0
    for r in range(height):
        for c in range(width):
            if terrain.is_land[r][c]:
                land_count += 1
                land_values.append(terrain.elevation_m[r][c])
            else:
                ocean_values.append(terrain.elevation_m[r][c])

    total = width * height
    terrain.ocean_fraction = 1.0 - land_count / total
    terrain.land_fraction = land_count / total
    terrain.min_elevation_m = min(min(land_values) if land_values else 0, min(ocean_values) if ocean_values else 0)
    terrain.max_elevation_m = max(max(land_values) if land_values else 0, max(ocean_values) if ocean_values else 0)
    terrain.mean_land_elevation_m = sum(land_values) / len(land_values) if land_values else 0.0
    terrain.mean_ocean_depth_m = sum(ocean_values) / len(ocean_values) if ocean_values else 0.0
    return new_land + raised_land

def _generate_rotation(rng: random.Random, planet: Planet) -> RotationState:
    moon = planet.moon
    ctx = planet.formation_context or {}
    tide_factor = 0.0 if moon is None else clamp(math.log10(max(moon.tidal_strength_relative_earth_moon, 0.05)) + 1.0, 0.0, 3.0)
    tide_label = str((ctx.get("tidal_effect_level") or getattr(moon, "tidal_effect_level", "") if moon else "")).lower()
    if tide_label in {"strong", "extreme"}:
        tide_factor += 0.55
    elif tide_label == "weak":
        tide_factor *= 0.65

    base_day = rng.uniform(14.0, 34.0)
    gravity_modifier = (planet.surface_gravity_g - 1.0) * 2.0
    tidal_modifier = tide_factor * rng.uniform(1.5, 4.5)
    rotation_period = clamp(base_day + gravity_modifier + tidal_modifier, 10.0, 72.0)

    stability = str(ctx.get("axial_stability_effect") or (getattr(moon, "axial_stability_effect", "") if moon else "")).lower()
    if stability in {"high", "strong"}:
        axial_tilt = rng.triangular(8.0, 30.0, 22.0)
    elif stability in {"low", "weak", "none"}:
        axial_tilt = rng.triangular(0.0, 55.0, 28.0)
    else:
        axial_tilt = rng.triangular(5.0, 35.0, 23.5)

    return RotationState(
        rotation_period_hours=rotation_period,
        axial_tilt_degrees=axial_tilt,
        solar_day_hours=rotation_period,
        year_length_days=planet.orbit.orbital_period_days,
    )



def _generate_atmosphere(rng: random.Random, planet: Planet) -> Atmosphere:
    """Generate a pressure-bearing atmosphere biased toward liquid-water stability.

    The project assumes land-stage life, so the Main Planet atmosphere is not a
    random Venus/Mars-like atmosphere. Pressure and greenhouse strength are
    modeled from gravity, escape velocity, volatile inventory, and irradiation,
    then gently regulated into a liquid-water temperature range.
    """
    escape = planet.escape_velocity_relative_earth
    gravity = planet.surface_gravity_g
    volatile = planet.composition.water_ice_fraction
    equilibrium_k = planet.equilibrium_temperature_k
    ctx = planet.formation_context or {}
    volatile_delivery = str(ctx.get("volatile_delivery", "moderate")).lower()
    tectonic_bias = str(ctx.get("tectonic_energy_bias", "earth_like")).lower()

    delivery_multiplier = {"dry": 0.72, "low": 0.78, "moderate": 1.0, "wet": 1.18, "high": 1.24, "heavy_bombardment": 1.15}.get(volatile_delivery, 1.0)
    outgassing_multiplier = 1.12 if tectonic_bias in {"high", "moderate_high"} else (0.88 if tectonic_bias in {"low", "quiet"} else 1.0)
    volatile_factor = clamp((0.85 + volatile * 16.0) * delivery_multiplier, 0.70, 1.95)
    gravity_factor = clamp(0.72 + 0.34 * gravity, 0.72, 1.45)
    retention_factor = clamp(0.82 + 0.22 * escape, 0.78, 1.35)
    thermal_loss_factor = clamp(1.08 - max(0.0, equilibrium_k - 260.0) / 180.0, 0.72, 1.10)
    pressure = clamp(rng.uniform(0.82, 1.18) * volatile_factor * gravity_factor * retention_factor * thermal_loss_factor * outgassing_multiplier, 0.50, 3.20)

    # Oxygen is assumed because the project scope starts after land life exists.
    oxygen = clamp(rng.triangular(0.17, 0.27, 0.21), 0.15, 0.30)
    argon = rng.uniform(0.006, 0.018)

    # CO2 is kept in an Earth-like-to-warm-Earth band; pressure controls the
    # broader greenhouse effect.
    co2_ppm = rng.triangular(260.0, 1200.0, 520.0)
    nitrogen = max(0.50, 1.0 - oxygen - argon - (co2_ppm / 1_000_000.0))

    water_vapor_factor = clamp((0.65 + volatile * 12.0 + (planet.stellar_flux_earth - 0.85) * 0.16) * delivery_multiplier, 0.35, 1.65)
    raw_greenhouse = 25.0 + (pressure - 1.0) * 7.5 + math.log10(max(co2_ppm, 1.0) / 280.0) * 4.0 + water_vapor_factor * 3.4

    # Since the Main Planet is required to be liquid-water eligible, nudge the
    # greenhouse value toward a temperate surface instead of allowing runaway hot
    # or globally frozen outcomes.
    target_surface_k = rng.triangular(282.0, 293.0, 288.0)
    needed_greenhouse = target_surface_k - equilibrium_k
    greenhouse = clamp(raw_greenhouse, max(18.0, needed_greenhouse - 6.0), min(48.0, needed_greenhouse + 8.0))
    surface_temp_k = equilibrium_k + greenhouse

    notes = [
        "Oxygen-bearing atmosphere assumed; oxygen fraction is generated within an Earth-like band.",
        "Atmospheric pressure is estimated from gravity, escape velocity, volatile inventory, and thermal retention.",
        "Greenhouse warming is regulated toward the liquid-water range because the Main Planet selector requires ocean-capable worlds.",
    ]
    if surface_temp_k < 273.15:
        notes.append("Global estimate is cool; equatorial and lowland regions may still support liquid water.")
    elif surface_temp_k > 303.0:
        notes.append("Global estimate is warm; climate model may create expanded tropical/arid zones.")

    return Atmosphere(
        pressure_bar=pressure,
        nitrogen_fraction=nitrogen,
        oxygen_fraction=oxygen,
        carbon_dioxide_ppm=co2_ppm,
        argon_fraction=argon,
        water_vapor_factor=water_vapor_factor,
        greenhouse_warming_k=greenhouse,
        estimated_mean_surface_temp_k=surface_temp_k,
        estimated_mean_surface_temp_c=surface_temp_k - 273.15,
        notes=notes,
    )



def _generate_hydrosphere(rng: random.Random, planet: Planet, atmosphere: Atmosphere) -> Hydrosphere:
    volatile = planet.composition.water_ice_fraction
    ctx = planet.formation_context or {}
    volatile_delivery = str(ctx.get("volatile_delivery", "moderate")).lower()
    preference = str(ctx.get("main_planet_preference", "earthlike")).lower()
    delivery_ocean_bias = {"dry": -0.12, "low": -0.08, "moderate": 0.0, "wet": 0.08, "high": 0.11, "heavy_bombardment": 0.06}.get(volatile_delivery, 0.0)
    preference_bias = {"dry_terrestrial": -0.16, "oceanic": 0.13, "super_earth": 0.04, "colder_world": -0.02, "warmer_world": -0.04}.get(preference, 0.0)
    # Treat composition water/ice fraction as a broad volatile indicator, not a
    # literal surface-ocean mass fraction. Because Main Planet selection requires
    # ocean-capable candidates, the target ocean fraction is deliberately biased
    # toward significant oceans rather than marginal puddle worlds.
    base_ocean = 0.46 + math.log10(1.0 + volatile * 320.0) * 0.24
    temp_c = atmosphere.estimated_mean_surface_temp_c
    if temp_c < 0.0:
        base_ocean -= 0.04
    elif temp_c > 30.0:
        base_ocean -= 0.03
    target = clamp(base_ocean + delivery_ocean_bias + preference_bias + rng.uniform(-0.08, 0.11), 0.28, 0.88)

    if target < 0.52:
        water_class = "continent-rich ocean world"
    elif target < 0.66:
        water_class = "mixed continents and significant oceans"
    elif target < 0.74:
        water_class = "ocean-rich world"
    else:
        water_class = "near-ocean world with scattered continents"

    if temp_c < -3.0:
        ice = "moderate to high polar ice tendency"
    elif temp_c < 5.0:
        ice = "moderate polar ice tendency"
    elif temp_c > 32.0:
        ice = "low polar ice, high evaporation"
    else:
        ice = "Earth-like polar ice tendency"

    return Hydrosphere(
        volatile_fraction=volatile,
        ocean_fraction_target=target,
        ocean_fraction_actual=0.0,
        water_inventory_class=water_class,
        ice_cap_tendency=ice,
    )



def _generate_geology(rng: random.Random, star: Star, planet: Planet) -> GeologyState:
    # Larger planets retain internal heat longer; older systems cool down.
    ctx = planet.formation_context or {}
    tectonic_bias = str(ctx.get("tectonic_energy_bias", "earth_like")).lower()
    asymmetry_bias = str(ctx.get("crustal_asymmetry_bias", "medium")).lower()
    impact_history = str(ctx.get("impact_history", "normal")).lower()
    energy_multiplier = {"low": 0.78, "quiet": 0.72, "earth_like": 1.0, "moderate": 1.0, "moderate_high": 1.14, "high": 1.28}.get(tectonic_bias, 1.0)
    age_cooling = clamp(1.0 - (star.age_gyr - 2.0) / 9.0, 0.25, 1.0)
    mass_heat = clamp(planet.mass_earth ** 0.28, 0.65, 1.45)
    tidal_heat = 0.0
    if planet.moon is not None:
        tidal_heat = clamp(math.log10(max(planet.moon.tidal_strength_relative_earth_moon, 0.05)) / 5.0, 0.0, 0.45)

    internal_heat = clamp((0.55 * mass_heat + 0.45 * age_cooling + tidal_heat) * energy_multiplier * rng.uniform(0.85, 1.15), 0.16, 1.95)
    volcanism = clamp(internal_heat * rng.uniform(0.55, 1.20), 0.05, 1.70)
    erosion = clamp(rng.uniform(1.05, 1.85) * (1.0 / max(planet.surface_gravity_g, 0.5)), 0.70, 2.25)
    mountain_factor = clamp((1.15 / max(planet.surface_gravity_g, 0.5)) * (0.75 + volcanism * 0.35), 0.35, 1.55)
    impact_multiplier = {"calm": 0.72, "normal": 1.0, "battered": 1.45, "heavy_bombardment": 1.35}.get(impact_history, 1.0)
    asymmetry_multiplier = {"low": 0.90, "medium": 1.0, "moderate": 1.0, "high": 1.15}.get(asymmetry_bias, 1.0)
    crater_density = clamp((star.age_gyr / 8.0) * (1.15 - erosion * 0.45) * impact_multiplier * rng.uniform(0.65, 1.25), 0.03, 1.55)
    roughness = clamp((0.28 + mountain_factor * 0.24 + volcanism * 0.14 - erosion * 0.26) * asymmetry_multiplier, 0.08, 1.35)

    if volcanism > 1.15:
        geology_class = "volcanically active rugged world"
    elif internal_heat > 0.75:
        geology_class = "geologically active terrestrial world"
    elif internal_heat > 0.42:
        geology_class = "moderately active aging terrestrial world"
    else:
        geology_class = "geologically quiet old terrestrial world"

    return GeologyState(
        internal_heat=internal_heat,
        volcanism=volcanism,
        erosion=erosion,
        mountain_factor=mountain_factor,
        crater_density=crater_density,
        surface_roughness=roughness,
        geology_class=geology_class,
    )




def _terrain_style_modifiers(style: str) -> dict[str, float]:
    """Return generation-level biases for optional terrain style presets."""
    style = str(style or "derived_from_planet_physics").lower()
    presets = {
        "earth_like_mixed_continents": {"fragmentation": 0.08, "island_density": 0.02, "supercontinent": -0.18, "relief": 0.04},
        "supercontinent_world": {"fragmentation": -0.32, "island_density": -0.10, "supercontinent": 0.34, "relief": 0.02},
        "archipelago_world": {"fragmentation": 0.34, "island_density": 0.34, "supercontinent": -0.32, "relief": 0.04},
        "ocean_world": {"fragmentation": 0.18, "island_density": 0.22, "supercontinent": -0.20, "relief": -0.02},
        "rugged_tectonic_world": {"fragmentation": 0.16, "island_density": 0.10, "supercontinent": -0.08, "relief": 0.18},
        "old_eroded_shield_world": {"fragmentation": -0.10, "island_density": -0.06, "supercontinent": 0.08, "relief": -0.14},
        "volcanic_island_arc_world": {"fragmentation": 0.20, "island_density": 0.30, "supercontinent": -0.18, "relief": 0.12},
        "dry_highland_world": {"fragmentation": -0.02, "island_density": -0.10, "supercontinent": 0.04, "relief": 0.16},
    }
    return presets.get(style, {})


def _effective_supercontinent_score(controls: dict) -> float:
    """Convert derived/user supercontinent control into a 0..1 model bias."""
    derived = clamp(float(controls.get("derived_supercontinent_score", 0.45) or 0.45), 0.0, 1.0)
    mode = str(controls.get("supercontinent_tendency", "derived") or "derived").lower()
    if mode in {"suppressed", "suppress", "none"}:
        return min(derived, 0.08)
    if mode == "rare":
        return min(max(derived * 0.45, 0.08), 0.28)
    if mode == "occasional":
        return min(max(derived, 0.22), 0.52)
    if mode == "common":
        return min(max(derived, 0.50), 0.78)
    if mode in {"forced", "force"}:
        return 0.94
    return derived

def _generate_terrain(
    rng: random.Random,
    planet: Planet,
    hydrosphere: Hydrosphere,
    geology: GeologyState,
    config: PlanetProfileConfig,
    *,
    output_dir: str | None = None,
) -> TerrainMap:
    """Generate terrain directly at the requested resolution.

    Earlier builds silently generated an intermediate feature grid and upscaled
    large maps. That made iteration faster but also created square inland seas,
    shelf-edge land ribbons, and misleading progress messages. From this build
    onward, normal terrain synthesis is always performed at the requested grid
    size. Preview/fast mode changes the requested map size before this point; it
    does not upscale a smaller terrain field.
    """
    width = max(int(config.min_map_width), int(config.map_width))
    height = max(int(config.min_map_height), int(config.map_height))
    mode = str(getattr(config, "terrain_generation_mode", "procedural_legacy") or "procedural_legacy")
    if mode == "legacy":
        mode = "procedural_legacy"
    if mode == "plate_tectonic_v1":
        _progress(f"Using terrain mode plate_tectonic_v1 at {width} x {height}; generating native plate setup, motion/boundary diagnostics, plate-derived ocean-floor fields, continental relief, plate-derived landform belts, margin profiles, coasts/shelves/islands, drainage-ready valleys/basins, and final plate-mode QA.")
        return _generate_plate_tectonic_v1_scaffold(rng, planet, hydrosphere, geology, config, output_dir=output_dir)
    if mode == "plate_history_v1":
        _progress(f"Using terrain mode plate_history_v1 at {width} x {height}; running compact time-evolved plate history and deriving terrain from accumulated crust fields.")
        return _generate_plate_history_v1_scaffold(rng, planet, hydrosphere, geology, config, output_dir=output_dir)
    if mode == "plate_history_v2":
        _progress(f"Using terrain mode plate_history_v2 at {width} x {height}; deriving terrain from the stable plate-history macro layout plus structural crust, ridge, arc, and erosion reconstruction.")
        return _generate_plate_history_v2_scaffold(rng, planet, hydrosphere, geology, config, output_dir=output_dir)
    if mode == "plate_history_v3":
        _progress(f"Using terrain mode plate_history_v3 at {width} x {height}; running unified continuous-field tectonic reconstruction with deformable diagnostics, isostasy, bathymetry, lakes, age-aware erosion, and readable history snapshots.")
        return _generate_plate_history_v3_scaffold(rng, planet, hydrosphere, geology, config, output_dir=output_dir)
    if mode == "plate_history_v4":
        _progress(f"Using terrain mode plate_history_v4 at {width} x {height}; starting from stable v3 and applying experimental non-Voronoi topology, microplate/sliver, and volcanic-island-chain shaping.")
        return _generate_plate_history_v4_scaffold(rng, planet, hydrosphere, geology, config, output_dir=output_dir)
    if mode not in {"procedural_legacy", "real_world_stage3"}:
        _progress(f"Unknown terrain mode {mode!r}; falling back to procedural_legacy.")
    _progress(f"Using direct full-resolution terrain synthesis {width} x {height}; no terrain upscaling. Terrain mode: procedural_legacy.")
    return _generate_terrain_core(rng, planet, hydrosphere, geology, config, output_dir=output_dir)



def _resize_float_field(field, width: int, height: int):
    """Resize a float raster with bicubic filtering."""
    try:
        import numpy as np
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("NumPy and Pillow are required for terrain resizing. Install with: pip install -r requirements.txt") from exc
    arr = np.asarray(field, dtype=np.float32)
    if arr.shape == (height, width):
        return arr.astype(np.float32, copy=False)
    return np.asarray(Image.fromarray(arr, mode="F").resize((width, height), Image.Resampling.BICUBIC), dtype=np.float32).copy()


def _resize_int_field(field, width: int, height: int):
    """Resize a class/id raster with nearest-neighbor semantics."""
    try:
        import numpy as np
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("NumPy and Pillow are required for terrain resizing. Install with: pip install -r requirements.txt") from exc
    arr = np.asarray(field, dtype=np.int32)
    if arr.shape == (height, width):
        return arr.astype(np.int32, copy=False)
    return np.asarray(Image.fromarray(arr, mode="I").resize((width, height), Image.Resampling.NEAREST), dtype=np.int32).copy()


def _norm01(field, pct: float = 99.0):
    """Robust 0..1 normalization for tectonic influence fields."""
    import numpy as np
    arr = np.asarray(field, dtype=np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    arr = arr - min(0.0, float(arr.min()))
    hi = float(np.percentile(arr, pct)) if arr.size else 0.0
    if hi <= 1.0e-6:
        hi = float(arr.max()) if arr.size else 0.0
    if hi > 1.0e-6:
        arr = arr / hi
    return np.clip(arr, 0.0, 1.0).astype(np.float32, copy=False)


def _periodic_plate_ids(xx, yy, centers_x, centers_y, plate_scale, warp=None):
    """Assign cells to moving plate centers with wrapped longitude distance."""
    import numpy as np
    h, w = xx.shape
    best = np.full((h, w), np.inf, dtype=np.float32)
    ids = np.zeros((h, w), dtype=np.int32)
    for i, (cx, cy, scale) in enumerate(zip(centers_x, centers_y, plate_scale)):
        dx = np.abs(xx - cx)
        dx = np.minimum(dx, w - dx)
        dy = yy - cy
        d2 = (dx * dx + 1.35 * dy * dy) / max(float(scale), 1.0)
        if warp is not None:
            d2 = d2 + warp[i]
        mask = d2 < best
        ids[mask] = i
        best[mask] = d2[mask]
    return ids


def _plate_history_legacy_grid(width: int, height: int) -> tuple[int, int]:
    """Return the last-known-good Update 17 plate-history macro grid.

    The raw high/native path introduced after Update 17 lets the kinematic plate
    model operate directly at output resolution. That exposed scale-dependent
    artifacts: overly neat blocks, dense checker patterns, and unstable land
    thresholds. Until the history model is rebuilt around deformable plates, all
    public grid-scale choices use this stable macro grid and reserve requested
    resolution for final terrain rendering/diagnostics.
    """
    hist_w = int(max(192, min(512, round(width / 8))))
    if hist_w % 2:
        hist_w += 1
    hist_h = max(64, hist_w // 2)
    return int(hist_w), int(hist_h)


def _resolve_plate_history_grid(width: int, height: int, config) -> tuple[int, int, str, int | None]:
    """Resolve the internal plate-history grid.

    The default policy keeps the large-scale plate history on the proven macro
    grid and applies high-resolution detail afterward.  Raw requested grids are
    a developer-only research path because current high/raw history grids can
    degrade terrain quality.
    """
    scale = str(getattr(config, "tectonic_grid_scale", "legacy") or "legacy").strip().lower()
    if scale not in {"legacy", "preview", "normal", "high", "native", "custom"}:
        scale = "legacy"
    policy = str(getattr(config, "tectonic_grid_policy", "stable") or "stable").strip().lower()
    if policy not in {"stable", "raw"}:
        policy = "stable"
    allow_raw = bool(getattr(config, "allow_experimental_tectonic_grid", False))
    if policy == "raw" and not allow_raw:
        policy = "stable"

    stable_w, stable_h = _plate_history_legacy_grid(width, height)
    if scale == "legacy":
        return stable_w, stable_h, scale, None

    custom_w = getattr(config, "tectonic_grid_width", None)
    custom_h = getattr(config, "tectonic_grid_height", None)
    divisor = {"preview": 8.0, "normal": 4.0, "high": 2.0, "native": 1.0, "custom": 4.0}[scale]
    if scale == "custom" and custom_w:
        requested_w = int(custom_w)
        requested_h = int(custom_h) if custom_h else int(round(requested_w * height / max(width, 1)))
    else:
        requested_w = int(round(width / divisor))
        requested_h = int(round(height / divisor))
    requested_w = int(max(1, requested_w))
    requested_h = int(max(1, requested_h))
    requested_cells = requested_w * requested_h
    stable_cells = stable_w * stable_h

    if policy == "raw":
        return requested_w, requested_h, scale, None

    capped_from = int(requested_cells) if requested_cells != stable_cells else None
    return stable_w, stable_h, scale, capped_from


def _generate_plate_history_v1_scaffold(
    rng: random.Random,
    planet: Planet,
    hydrosphere: Hydrosphere,
    geology: GeologyState,
    config: PlanetProfileConfig,
    *,
    output_dir: str | None = None,
) -> TerrainMap:
    """Generate terrain from a compact time-evolved plate-history model.

    This mode intentionally preserves plate_tectonic_v1.  It is a separate
    terrain generation mode that runs an approximate kinematic history at a
    low/medium tectonic grid, accumulates crust fields over geological time,
    then derives full-resolution elevation from those fields.  It is not a
    mantle-convection solver; it is a practical procedural reconstruction model
    aimed at land formation, shelves, arcs, sutures, plateaus, and ocean age.
    """
    try:
        import numpy as np
        from scipy import ndimage
    except ImportError as exc:
        raise RuntimeError("NumPy and SciPy are required for plate_history_v1. Install with: pip install -r requirements.txt") from exc

    width = max(int(config.min_map_width), int(config.map_width))
    height = max(int(config.min_map_height), int(config.map_height))
    np_rng = np.random.default_rng(rng.randint(1, 2_147_483_647))

    # Resolve the plate-history grid.  The default ``legacy`` path preserves
    # Update 17 exactly; higher-resolution modes are opt-in via CLI/Web UI.
    hist_w, hist_h, tectonic_grid_scale, tectonic_grid_capped_from_cells = _resolve_plate_history_grid(width, height, config)
    tectonic_grid_policy = str(getattr(config, "tectonic_grid_policy", "stable") or "stable").strip().lower()
    if tectonic_grid_policy not in {"stable", "raw"}:
        tectonic_grid_policy = "stable"
    if tectonic_grid_policy == "raw" and not bool(getattr(config, "allow_experimental_tectonic_grid", False)):
        tectonic_grid_policy = "stable"
    if tectonic_grid_scale != "legacy" and tectonic_grid_capped_from_cells:
        _progress(
            "Plate History v1 hybrid grid: "
            f"requested {tectonic_grid_scale!r} would use {tectonic_grid_capped_from_cells:,} raw cells; "
            f"using stable macro history grid {hist_w} x {hist_h}, then applying high-resolution tectonic detail at {width} x {height}. "
            "Higher-resolution raw history grids are deferred terrain research."
        )
    elif tectonic_grid_scale != "legacy" and tectonic_grid_policy == "raw":
        _progress(
            "Plate History raw grid: "
            f"running requested {tectonic_grid_scale!r} historical grid at {hist_w} x {hist_h} ({hist_w * hist_h:,} cells)."
        )
    yy, xx = np.mgrid[0:hist_h, 0:hist_w].astype(np.float32)
    lat = 90.0 - ((yy + 0.5) / hist_h) * 180.0
    abs_lat = np.abs(lat)

    requested_history = getattr(config, "tectonic_history_myr", None)
    if requested_history is None:
        # Mature, but not always ancient: enough time to make shelves, sutures,
        # ocean-age gradients, and collisions without forcing every world into a
        # single overprocessed supercontinent cycle.
        requested_history = rng.uniform(350.0, 1300.0) * clamp(0.75 + geology.internal_heat * 0.35, 0.75, 1.35)
    history_myr = float(clamp(float(requested_history), 50.0, 2500.0))
    timestep_myr = float(getattr(config, "tectonic_timestep_myr", 2.5) or 2.5)
    timestep_myr = float(clamp(timestep_myr, 0.5, 20.0))
    requested_steps = max(1, int(round(history_myr / timestep_myr)))
    # Sampling cap keeps the new mode usable from the Web UI.  The stored history
    # length remains the user's requested length; each internal epoch represents
    # a larger geological interval when needed.
    history_steps = max(32, min(180, requested_steps))
    epoch_myr = history_myr / float(history_steps)

    ocean_target = clamp(float(hydrosphere.ocean_fraction_target), 0.45, 0.82)
    land_target = 1.0 - ocean_target
    heat = clamp(float(geology.internal_heat), 0.05, 1.95)
    activity = clamp(0.55 + heat * 0.45 + geology.volcanism * 0.16, 0.45, 1.75)
    plate_count = int(round(clamp(12 + activity * 7 + rng.uniform(-2, 5), 10, 30)))

    centers_x = np_rng.uniform(0, hist_w, plate_count).astype(np.float32)
    centers_y = np_rng.uniform(hist_h * 0.06, hist_h * 0.94, plate_count).astype(np.float32)
    # Use varied plate sizes to avoid evenly spaced Voronoi cells.
    plate_scale = np_rng.uniform(0.65, 1.85, plate_count).astype(np.float32)
    plate_type = np.zeros(plate_count, dtype=np.int32)
    cont_prob = clamp(land_target * 0.95 + 0.08, 0.22, 0.44)
    mixed_prob = 0.22
    micro_prob = 0.08 + min(0.08, heat * 0.04)
    for i in range(plate_count):
        r = rng.random()
        if r < cont_prob:
            plate_type[i] = 1  # continental
        elif r < cont_prob + mixed_prob:
            plate_type[i] = 2  # mixed / margin-rich
        elif r < cont_prob + mixed_prob + micro_prob:
            plate_type[i] = 3  # microplate / terrane
        else:
            plate_type[i] = 0  # oceanic
    if not np.any(plate_type == 1):
        plate_type[rng.randrange(plate_count)] = 1

    angles = np_rng.uniform(0.0, math.tau, plate_count).astype(np.float32)
    base_speed = np.where(plate_type == 0, 0.11, np.where(plate_type == 1, 0.055, 0.082)).astype(np.float32)
    speed = base_speed * np_rng.uniform(0.55, 1.45, plate_count).astype(np.float32) * activity
    vx = np.cos(angles).astype(np.float32) * speed
    vy = np.sin(angles).astype(np.float32) * speed * 0.55
    # Moving noise perturbations make old boundaries/sutures differ from present-day boundaries.
    warp = np_rng.normal(0.0, 0.7, (plate_count, hist_h, hist_w)).astype(np.float32)
    for i in range(plate_count):
        warp[i] = ndimage.gaussian_filter(warp[i], sigma=(hist_h / 18.0, hist_w / 18.0), mode=("nearest", "wrap"))

    convergence_acc = np.zeros((hist_h, hist_w), dtype=np.float32)
    divergence_acc = np.zeros((hist_h, hist_w), dtype=np.float32)
    transform_acc = np.zeros((hist_h, hist_w), dtype=np.float32)
    collision_acc = np.zeros((hist_h, hist_w), dtype=np.float32)
    subduction_acc = np.zeros((hist_h, hist_w), dtype=np.float32)
    ridge_acc = np.zeros((hist_h, hist_w), dtype=np.float32)
    trench_acc = np.zeros((hist_h, hist_w), dtype=np.float32)
    rift_acc = np.zeros((hist_h, hist_w), dtype=np.float32)
    continental_acc = np.zeros((hist_h, hist_w), dtype=np.float32)
    ocean_age = np.zeros((hist_h, hist_w), dtype=np.float32)
    present_ids = np.zeros((hist_h, hist_w), dtype=np.int32)
    present_boundary = np.zeros((hist_h, hist_w), dtype=np.int32)

    for step in range(history_steps):
        centers_x = (centers_x + vx * epoch_myr) % hist_w
        centers_y = centers_y + vy * epoch_myr
        high = centers_y > hist_h * 0.96
        low = centers_y < hist_h * 0.04
        vy[high | low] *= -1.0
        centers_y = np.clip(centers_y, hist_h * 0.04, hist_h * 0.96)
        if step and step % max(18, history_steps // 6) == 0:
            # Occasional velocity drift approximates ridge jumps, plate reorganization,
            # and changing slab-pull without requiring full topology remeshing.
            turn = np_rng.normal(0.0, 0.28, plate_count).astype(np.float32)
            ca, sa = np.cos(turn), np.sin(turn)
            vx, vy = vx * ca - vy * sa, vx * sa + vy * ca

        ids = _periodic_plate_ids(xx, yy, centers_x, centers_y, plate_scale, warp=warp * 0.55)
        types = plate_type[ids]
        cont_here = (types == 1).astype(np.float32) + (types == 2).astype(np.float32) * 0.58 + (types == 3).astype(np.float32) * 0.34
        continental_acc += cont_here * epoch_myr

        east = np.roll(ids, -1, axis=1)
        south = np.vstack((ids[1:, :], ids[-1:, :]))
        b_e = ids != east
        b_s = ids != south
        vx_here, vy_here = vx[ids], vy[ids]
        vx_e, vy_e = vx[east], vy[east]
        vx_s, vy_s = vx[south], vy[south]
        rel_x_e = vx_e - vx_here
        rel_y_e = vy_e - vy_here
        rel_y_s = vy_s - vy_here
        rel_x_s = vx_s - vx_here

        conv_e = np.where(b_e, np.maximum(-rel_x_e, 0.0), 0.0)
        div_e = np.where(b_e, np.maximum(rel_x_e, 0.0), 0.0)
        shear_e = np.where(b_e, np.abs(rel_y_e), 0.0)
        conv_s = np.where(b_s, np.maximum(-rel_y_s, 0.0), 0.0)
        div_s = np.where(b_s, np.maximum(rel_y_s, 0.0), 0.0)
        shear_s = np.where(b_s, np.abs(rel_x_s), 0.0)
        conv = conv_e + conv_s
        div = div_e + div_s
        shear = shear_e + shear_s

        type_e = plate_type[east]
        type_s = plate_type[south]
        cont_pair_e = ((types == 1) | (types == 2)) & ((type_e == 1) | (type_e == 2))
        cont_pair_s = ((types == 1) | (types == 2)) & ((type_s == 1) | (type_s == 2))
        ocean_pair_e = (types == 0) & (type_e == 0)
        ocean_pair_s = (types == 0) & (type_s == 0)
        mixed_ocean_pair_e = ((types == 0) & ((type_e == 1) | (type_e == 2) | (type_e == 3))) | (((types == 1) | (types == 2) | (types == 3)) & (type_e == 0))
        mixed_ocean_pair_s = ((types == 0) & ((type_s == 1) | (type_s == 2) | (type_s == 3))) | (((types == 1) | (types == 2) | (types == 3)) & (type_s == 0))

        convergence_acc += conv * epoch_myr
        divergence_acc += div * epoch_myr
        transform_acc += shear * epoch_myr
        collision_acc += (conv_e * cont_pair_e + conv_s * cont_pair_s) * epoch_myr
        subduction_acc += (conv_e * (ocean_pair_e | mixed_ocean_pair_e) + conv_s * (ocean_pair_s | mixed_ocean_pair_s)) * epoch_myr
        ridge_acc += (div_e * ocean_pair_e + div_s * ocean_pair_s) * epoch_myr
        trench_acc += (conv_e * (ocean_pair_e | mixed_ocean_pair_e) + conv_s * (ocean_pair_s | mixed_ocean_pair_s)) * epoch_myr
        rift_acc += (div_e * cont_pair_e + div_s * cont_pair_s) * epoch_myr

        oceanic = types == 0
        ocean_age[oceanic] += epoch_myr
        ridge_now = (div > np.percentile(div[b_e | b_s], 70) if np.any(b_e | b_s) else div > 0.0) & oceanic
        ocean_age[ridge_now] = 0.0
        present_ids = ids

    # Update 19 redo: derive a non-invasive plate-topology/junction diagnostic from
    # the final macro plate ownership.  This does not change plate IDs or terrain;
    # it lets us measure the remaining Voronoi-like topology problems safely.
    east_ids = np.roll(present_ids, -1, axis=1)
    west_ids = np.roll(present_ids, 1, axis=1)
    north_ids = np.vstack((present_ids[:1, :], present_ids[:-1, :]))
    south_ids = np.vstack((present_ids[1:, :], present_ids[-1:, :]))
    b_e = present_ids != east_ids
    b_w = present_ids != west_ids
    b_n = present_ids != north_ids
    b_s = present_ids != south_ids
    boundary_dir_count = b_e.astype(np.int16) + b_w.astype(np.int16) + b_n.astype(np.int16) + b_s.astype(np.int16)
    plate_junction_class = np.zeros((hist_h, hist_w), dtype=np.int16)
    plate_junction_class[boundary_dir_count >= 1] = 1  # ordinary plate edge / bend
    plate_junction_class[boundary_dir_count == 3] = 2  # T-like / triple junction candidate
    plate_junction_class[boundary_dir_count >= 4] = 3  # suspicious plus/cross junction
    plus_junction_count = int(np.count_nonzero(plate_junction_class == 3))
    t_triple_junction_count = int(np.count_nonzero(plate_junction_class == 2))
    ordinary_boundary_junction_count = int(np.count_nonzero(plate_junction_class == 1))

    cont_score = _norm01(continental_acc, 98.5)
    convergence = _norm01(convergence_acc, 99.0)
    divergence = _norm01(divergence_acc, 99.0)
    transform = _norm01(transform_acc, 99.0)
    collision = _norm01(collision_acc, 99.0)
    subduction = _norm01(subduction_acc, 99.0)
    ridge = _norm01(ridge_acc, 99.0)
    trench = _norm01(trench_acc, 99.0)
    rift = _norm01(rift_acc, 99.0)
    age_norm = np.clip(ocean_age / max(history_myr * 0.8, 1.0), 0.0, 1.0).astype(np.float32)

    # Broken island arcs: create separated volcanic centers from subduction support,
    # not a continuous thresholded ribbon.  Most support remains submerged as seamounts.
    arc_support = _norm01(ndimage.gaussian_filter(subduction * (1.0 - cont_score), sigma=(1.0, 2.2), mode=("nearest", "wrap")), 99.0)
    local_max = arc_support >= ndimage.maximum_filter(arc_support, size=(5, 9), mode=("nearest", "wrap")) - 1.0e-6
    candidates = np.argwhere(local_max & (arc_support > 0.34) & (np_rng.random(arc_support.shape) > 0.22))
    max_beads = int(clamp(plate_count * history_myr / 55.0, 18, 180))
    if len(candidates) > max_beads:
        take = np_rng.choice(len(candidates), size=max_beads, replace=False)
        candidates = candidates[take]
    arc_beads = np.zeros_like(arc_support, dtype=np.float32)
    for cy, cx in candidates:
        rad_y = int(np_rng.integers(1, 4))
        rad_x = int(np_rng.integers(2, 6))
        amp = float(np_rng.uniform(0.55, 1.0) * arc_support[cy, cx])
        for dy in range(-rad_y * 2, rad_y * 2 + 1):
            yy_i = cy + dy
            if yy_i < 0 or yy_i >= hist_h:
                continue
            for dx_i in range(-rad_x * 2, rad_x * 2 + 1):
                xx_i = (cx + dx_i) % hist_w
                d2 = (dy / max(rad_y, 1)) ** 2 + (dx_i / max(rad_x, 1)) ** 2
                if d2 <= 5.0:
                    arc_beads[yy_i, xx_i] = max(arc_beads[yy_i, xx_i], amp * math.exp(-0.55 * d2))
    arc_beads = _norm01(ndimage.gaussian_filter(arc_beads, sigma=(0.55, 0.9), mode=("nearest", "wrap")), 99.5)

    # Broad tectonic texture and coast irregularity.  This applies to all land,
    # so volcanic islands and continents share the same coastline-generation family.
    broad_noise = np_rng.normal(0.0, 1.0, (hist_h, hist_w)).astype(np.float32)
    broad_noise = ndimage.gaussian_filter(broad_noise, sigma=(hist_h / 14.0, hist_w / 14.0), mode=("nearest", "wrap"))
    broad_noise = (_norm01(broad_noise, 99.0) - 0.5) * 2.0
    medium_noise = np_rng.normal(0.0, 1.0, (hist_h, hist_w)).astype(np.float32)
    medium_noise = ndimage.gaussian_filter(medium_noise, sigma=(3.0, 5.0), mode=("nearest", "wrap"))
    medium_noise = (_norm01(medium_noise, 99.0) - 0.5) * 2.0

    polar_penalty = np.zeros((hist_h, hist_w), dtype=np.float32)
    if bool(getattr(config, "suppress_polar_land", False)):
        t = np.clip((abs_lat - 54.0) / 34.0, 0.0, 1.0)
        smooth = t * t * (3.0 - 2.0 * t)
        polar_noise = np_rng.normal(0.0, 1.0, (hist_h, hist_w)).astype(np.float32)
        polar_noise = ndimage.gaussian_filter(polar_noise, sigma=(4.0, 9.0), mode=("nearest", "wrap"))
        polar_noise = 0.72 + 0.46 * _norm01(polar_noise, 99.0)
        polar_penalty = smooth * polar_noise

    land_score = (
        cont_score * 1.25
        + collision * 0.38
        + arc_beads * 0.55
        + transform * 0.10
        - age_norm * 0.36
        - ridge * 0.16
        + broad_noise * 0.26
        + medium_noise * 0.30
        - polar_penalty * 0.95
    ).astype(np.float32)
    threshold = float(np.quantile(land_score, 1.0 - land_target))
    land = land_score >= threshold
    # Keep rare volcanic arc islands as beads, not continuous bands.
    land |= (arc_beads > 0.62) & (arc_support > 0.38)
    land = ndimage.binary_closing(land, structure=np.ones((3, 3), dtype=bool), iterations=1)
    land = ndimage.binary_opening(land, structure=np.ones((2, 2), dtype=bool), iterations=1)

    labels, ncomp = ndimage.label(land)
    objects = ndimage.find_objects(labels)
    component_area = np.bincount(labels.ravel(), minlength=ncomp + 1).astype(np.float32)
    snake_repairs = 0
    removed_tiny = 0
    for lab in range(1, ncomp + 1):
        area = int(component_area[lab])
        if area <= 0:
            continue
        comp = labels == lab
        sl = objects[lab - 1]
        if sl is None:
            continue
        h_box = sl[0].stop - sl[0].start
        w_box = sl[1].stop - sl[1].start
        # Components crossing the wrapped seam may have a misleading wide bbox;
        # this is still acceptable because the repair is conservative.
        aspect = max(w_box, h_box) / max(1, min(w_box, h_box))
        if area < 4 and float(arc_beads[comp].max(initial=0.0)) < 0.75:
            land[comp] = False
            removed_tiny += 1
        elif aspect >= 7.0 and area < hist_w * hist_h * 0.065 and min(w_box, h_box) <= max(3, int(hist_h * 0.08)):
            # Long thin islands are almost always artifacts.  Split them into
            # rugged bead-like highs using arc/volcanic peaks where available,
            # otherwise use the strongest local land potential.
            arc_peak = float(arc_beads[comp].max(initial=0.0))
            if arc_peak > 0.25:
                peak_cut = max(0.35, float(np.percentile(arc_beads[comp], 58)))
                keep = comp & (arc_beads >= peak_cut)
            else:
                peak_cut = float(np.percentile(land_score[comp], 62))
                keep = comp & (land_score >= peak_cut)
            keep = ndimage.binary_opening(keep, structure=np.ones((2, 2), dtype=bool), iterations=1)
            if int(keep.sum()) >= 3:
                land[comp] = False
                land[keep] = True
                snake_repairs += 1

    labels, ncomp = ndimage.label(land)
    component_area = np.bincount(labels.ravel(), minlength=ncomp + 1).astype(np.float32)
    land = labels > 0
    ocean = ~land

    # Recompute coast distances on a horizontally tiled map so the map edges are
    # not treated as impassable seams.
    ocean_tiled = np.tile(ocean, (1, 3))
    dist_to_land_tiled, nearest = ndimage.distance_transform_edt(ocean_tiled, return_indices=True)
    crop = slice(hist_w, hist_w * 2)
    dist_to_land = dist_to_land_tiled[:, crop].astype(np.float32)
    labels_tiled = np.tile(labels, (1, 3))
    nearest_label = labels_tiled[nearest[0][:, crop], nearest[1][:, crop]]
    nearest_area = component_area[np.clip(nearest_label, 0, len(component_area) - 1)]
    small_landmass_factor = np.clip((nearest_area - 12.0) / 420.0, 0.0, 1.0)
    land_tiled = np.tile(land, (1, 3))
    dist_to_ocean = ndimage.distance_transform_edt(land_tiled)[:, crop].astype(np.float32)

    active_margin = _norm01(subduction + trench + transform * 0.35, 99.0)
    passive_margin = np.clip(cont_score * (1.0 - active_margin) * (0.4 + age_norm * 0.6), 0.0, 1.0)
    shelf_width_cells = 2.0 + 16.0 * passive_margin + 4.0 * rift - 7.0 * active_margin
    shelf_width_cells = np.clip(shelf_width_cells, 1.2, 20.0)
    shelf_width_cells = shelf_width_cells * (0.18 + 0.82 * small_landmass_factor)

    orogeny = _norm01(collision * 0.82 + subduction * cont_score * 0.48 + transform * cont_score * 0.18, 99.0)
    plateau = _norm01(ndimage.gaussian_filter(orogeny * (0.45 + cont_score), sigma=(4.5, 8.0), mode=("nearest", "wrap")) * cont_score, 99.2)
    suture = _norm01(ndimage.gaussian_filter(convergence * cont_score, sigma=(2.0, 4.0), mode=("nearest", "wrap")), 99.0)
    volcanic = _norm01(arc_beads * 1.35 + arc_support * 0.16, 99.5)
    detail = np_rng.normal(0.0, 1.0, (hist_h, hist_w)).astype(np.float32)
    detail = ndimage.gaussian_filter(detail, sigma=(1.2, 2.2), mode=("nearest", "wrap"))
    detail = (_norm01(detail, 99.0) - 0.5) * 2.0

    elevation = np.zeros((hist_h, hist_w), dtype=np.float32)
    land_elev = (
        90.0
        + cont_score * 980.0
        + orogeny * (1500.0 + geology.mountain_factor * 720.0)
        + plateau * 1500.0
        + volcanic * 2100.0
        + suture * 520.0
        + transform * cont_score * 340.0
        - rift * cont_score * 620.0
        + detail * (130.0 + geology.surface_roughness * 90.0)
    )
    ocean_elev = (
        -3100.0
        - age_norm * 2100.0
        + ridge * 1450.0
        - trench * 1900.0
        + volcanic * 680.0
        + detail * 80.0
    )
    shelf = ocean & (dist_to_land <= np.maximum(shelf_width_cells, 0.1))
    shelf_depth = -45.0 - np.power(np.clip(dist_to_land / np.maximum(shelf_width_cells, 0.1), 0.0, 1.0), 1.45) * 900.0
    ocean_elev[shelf] = np.maximum(ocean_elev[shelf], shelf_depth[shelf])
    coastal_land = land & (dist_to_ocean <= 4.0)
    coast_cap = 45.0 + np.power(np.clip(dist_to_ocean, 0.0, 4.0) / 4.0, 1.25) * 820.0
    land_elev[coastal_land] = np.minimum(land_elev[coastal_land], coast_cap[coastal_land] + volcanic[coastal_land] * 900.0)
    elevation[land] = land_elev[land]
    elevation[ocean] = ocean_elev[ocean]
    elevation[land] = np.maximum(elevation[land], 1.0)
    elevation[ocean] = np.minimum(elevation[ocean], -1.0)

    # Upscale score and elevation separately.  Thresholding the upscaled score
    # gives coastlines smoother than nearest-neighbor masks while preserving the
    # same tectonic source fields for islands and continents.
    score_full_base = _resize_float_field(land_score, width, height)
    arc_full = _resize_float_field(arc_beads, width, height)

    # Update 19 redo: non-legacy grid choices are hybrid, not raw-native.  The
    # macro plate history remains stable, while high/native/custom choices add
    # deterministic full-resolution tectonic/coastal detail.  Legacy keeps the
    # exact Update 17 terrain path.
    hybrid_strength = {
        "legacy": 0.0,
        "preview": 0.18,
        "normal": 0.36,
        "high": 0.58,
        "native": 0.72,
        "custom": 0.50,
    }.get(str(tectonic_grid_scale), 0.0)
    score_full = score_full_base.copy()
    baseline_threshold = float(np.quantile(score_full_base, 1.0 - float(land.mean())))
    baseline_land_full = score_full_base >= baseline_threshold
    baseline_land_full |= arc_full > 0.68
    hybrid_land_changed_cells = 0
    hybrid_score_jitter_max = 0.0
    hybrid_arc_split_components = 0
    hybrid_arc_split_cells_removed = 0
    hybrid_ridge_spine_cells = 0
    hybrid_foreland_cells = 0
    hybrid_deposition_cells = 0
    final_mountain_strength_full = _resize_float_field(orogeny, width, height)
    if hybrid_strength > 0.0:
        boundary_strength_macro = np.maximum.reduce([convergence, divergence, transform])
        tectonic_focus = np.clip(_resize_float_field(
            np.clip(boundary_strength_macro * 0.45 + orogeny * 0.35 + rift * 0.20 + arc_support * 0.28, 0.0, 1.0),
            width,
            height,
        ), 0.0, 1.0)
        # Jitter only near the eventual coast; this avoids changing continental
        # interiors while giving native/high runs more realistic local coastlines.
        score_sigma = max(0.04, float(np.std(score_full_base)))
        shore_focus = np.clip(1.0 - np.abs(score_full_base - baseline_threshold) / (score_sigma * 0.90), 0.0, 1.0)
        noise_a = np_rng.normal(0.0, 1.0, (height, width)).astype(np.float32)
        noise_b = np_rng.normal(0.0, 1.0, (height, width)).astype(np.float32)
        noise_a = ndimage.gaussian_filter(noise_a, sigma=(max(1.0, height / 640.0), max(1.0, width / 640.0)), mode=("nearest", "wrap"))
        noise_b = ndimage.gaussian_filter(noise_b, sigma=(max(2.0, height / 300.0), max(3.0, width / 300.0)), mode=("nearest", "wrap"))
        native_noise = _norm01(noise_a * 0.55 + noise_b * 0.45, 99.0) * 2.0 - 1.0
        jitter = native_noise * hybrid_strength * (0.035 + 0.055 * shore_focus) * (0.45 + 0.55 * tectonic_focus)
        score_full = score_full_base + jitter.astype(np.float32)
        hybrid_score_jitter_max = float(np.max(np.abs(jitter))) if jitter.size else 0.0

    # Keep target close to the low-resolution result but do not force exact area.
    full_threshold = float(np.quantile(score_full, 1.0 - float(land.mean())))
    land_full = score_full >= full_threshold
    land_full |= arc_full > 0.68
    if hybrid_strength > 0.0:
        hybrid_land_changed_cells = int(np.count_nonzero(land_full != baseline_land_full))

        # Conservative arc cleanup: if full-resolution jitter connects volcanic
        # support into a long ribbon, remove the weak neck cells and keep separated
        # bead/high-center cells.  This only applies to small arc-dominated
        # components, not continents or large peninsulas.
        cont_for_arc_cleanup = _resize_float_field(cont_score, width, height)
        arc_ribbon = land_full & (arc_full > 0.34) & (cont_for_arc_cleanup < 0.38)
        arc_labels, arc_n = ndimage.label(arc_ribbon)
        if arc_n:
            arc_areas = np.bincount(arc_labels.ravel(), minlength=arc_n + 1)
            arc_objs = ndimage.find_objects(arc_labels)
            max_arc_area = int(width * height * 0.035)
            for arc_lab in range(1, arc_n + 1):
                area_a = int(arc_areas[arc_lab])
                if area_a <= 0 or area_a > max_arc_area:
                    continue
                sla = arc_objs[arc_lab - 1]
                if sla is None:
                    continue
                ha = sla[0].stop - sla[0].start
                wa = sla[1].stop - sla[1].start
                aspect_a = max(ha, wa) / max(1, min(ha, wa))
                if aspect_a < 5.8 or min(ha, wa) > max(10, int(height * 0.075)):
                    continue
                comp_a = arc_labels == arc_lab
                local_arc = arc_full[comp_a]
                if local_arc.size == 0:
                    continue
                peak_cut = max(0.40, float(np.percentile(local_arc, 68)))
                bead_keep = comp_a & (arc_full >= peak_cut)
                bead_keep |= comp_a & (arc_full >= ndimage.maximum_filter(arc_full, size=(9, 13), mode=("nearest", "wrap")) - 1.0e-6)
                bead_keep = ndimage.binary_dilation(bead_keep, structure=np.ones((3, 3), dtype=bool), iterations=1) & comp_a
                if int(bead_keep.sum()) < 8:
                    continue
                remove_a = comp_a & (~bead_keep) & (score_full < baseline_threshold + 0.10)
                if int(remove_a.sum()) <= 0:
                    continue
                land_full[remove_a] = False
                hybrid_arc_split_components += 1
                hybrid_arc_split_cells_removed += int(remove_a.sum())
        hybrid_land_changed_cells = int(np.count_nonzero(land_full != baseline_land_full))
    # Final full-resolution repair for thin snake-like islands produced by the
    # score threshold.  This is deliberately limited to small/narrow components
    # so continents and real peninsulas are not erased.
    labels_full, n_full = ndimage.label(land_full)
    if n_full:
        areas_full = np.bincount(labels_full.ravel(), minlength=n_full + 1)
        objs_full = ndimage.find_objects(labels_full)
        max_small_area = int(width * height * 0.055)
        for lab in range(1, n_full + 1):
            area_f = int(areas_full[lab])
            if area_f <= 0 or area_f > max_small_area:
                continue
            slf = objs_full[lab - 1]
            if slf is None:
                continue
            hb = slf[0].stop - slf[0].start
            wb = slf[1].stop - slf[1].start
            aspect_f = max(wb, hb) / max(1, min(wb, hb))
            if aspect_f >= 8.0 and min(wb, hb) <= max(5, int(height * 0.055)):
                comp_f = labels_full == lab
                cut = float(np.percentile(np.maximum(score_full, arc_full * 2.0)[comp_f], 63))
                keep_f = comp_f & (np.maximum(score_full, arc_full * 2.0) >= cut)
                keep_f = ndimage.binary_opening(keep_f, structure=np.ones((3, 3), dtype=bool), iterations=1)
                if int(keep_f.sum()) >= 8:
                    land_full[comp_f] = False
                    land_full[keep_f] = True
                    snake_repairs += 1
    elev_full = _resize_float_field(elevation, width, height)

    hybrid_relief_added_max_m = 0.0
    hybrid_erosion_cells = 0
    if hybrid_strength > 0.0:
        orogeny_full_detail = np.clip(_resize_float_field(orogeny, width, height), 0.0, 1.0)
        plateau_full_detail = np.clip(_resize_float_field(plateau, width, height), 0.0, 1.0)
        rift_full_detail = np.clip(_resize_float_field(rift, width, height), 0.0, 1.0)
        active_full_detail = np.clip(_resize_float_field(active_margin, width, height), 0.0, 1.0)
        transform_full_detail = np.clip(_resize_float_field(transform, width, height), 0.0, 1.0)
        land_tiled_full = np.tile(land_full, (1, 3))
        crop_full = slice(width, width * 2)
        dist_ocean_full = ndimage.distance_transform_edt(land_tiled_full)[:, crop_full].astype(np.float32)

        # Build visible ridge spines from macro orogeny/active-margin fields while
        # adding only full-resolution expression, not new large landmasses.  The
        # axis detector favors local maxima in the broad orogenic envelope, then
        # grows short branches from deterministic noise.
        ridge_noise = np_rng.normal(0.0, 1.0, (height, width)).astype(np.float32)
        ridge_noise = ndimage.gaussian_filter(ridge_noise, sigma=(max(1.0, height / 900.0), max(1.4, width / 900.0)), mode=("nearest", "wrap"))
        ridge_noise = _norm01(ridge_noise, 99.0)
        ridge_envelope = np.clip(orogeny_full_detail * 0.90 + active_full_detail * 0.28 + transform_full_detail * 0.10, 0.0, 1.0)
        local_window = (max(5, int(round(height / 170))), max(9, int(round(width / 170))))
        local_max = ndimage.maximum_filter(ridge_envelope, size=local_window, mode=("nearest", "wrap"))
        ridge_axis = land_full & (ridge_envelope > 0.24) & (ridge_envelope >= local_max - 0.020)
        branch_seed = land_full & (ridge_envelope > 0.30) & (ridge_noise > 0.73)
        branch_seed &= ndimage.binary_dilation(ridge_axis, structure=np.ones((5, 5), dtype=bool), iterations=2)
        ridge_axis |= branch_seed
        ridge_axis = ndimage.binary_dilation(ridge_axis, structure=np.ones((3, 3), dtype=bool), iterations=1)
        hybrid_ridge_spine_cells = int(np.count_nonzero(ridge_axis))
        ridge_dist = ndimage.distance_transform_edt(~ridge_axis).astype(np.float32)
        ridge_spine = np.exp(-np.square(ridge_dist / (1.6 + 1.2 * hybrid_strength))) * ridge_envelope
        ridge_spine = np.clip(ridge_spine, 0.0, 1.0)
        final_mountain_strength_full = np.clip(np.maximum(orogeny_full_detail, ridge_spine * 1.25), 0.0, 1.0)

        ridge_gain = (ridge_spine ** 1.20) * (460.0 + 420.0 * float(getattr(geology, "mountain_factor", 0.8))) * hybrid_strength
        plateau_gain = (plateau_full_detail ** 1.35) * 170.0 * hybrid_strength
        coastal_allow = np.clip((dist_ocean_full - 1.5) / 7.0, 0.25, 1.0)
        coastal_allow = np.maximum(coastal_allow, active_full_detail * 0.95)
        relief_gain = (ridge_gain + plateau_gain) * coastal_allow

        # Add mild foreland/depositional lows alongside major ridge belts.  This
        # makes mountain belts read as ridges with adjacent basins instead of one
        # broad raised blur.
        foreland_ring = (ridge_dist > 4.0) & (ridge_dist < 18.0) & land_full & (ridge_envelope > 0.22)
        foreland_ring &= (plateau_full_detail < 0.42) & (active_full_detail < 0.70)
        foreland_cut = (70.0 + 120.0 * hybrid_strength) * ridge_envelope * foreland_ring.astype(np.float32)
        relief_gain = np.nan_to_num(relief_gain - foreland_cut, nan=0.0, posinf=0.0, neginf=0.0)
        relief_gain[~land_full] = 0.0
        elev_full = np.nan_to_num(elev_full + relief_gain.astype(np.float32), nan=0.0, posinf=9000.0, neginf=-11000.0)
        hybrid_relief_added_max_m = float(np.max(relief_gain)) if relief_gain.size else 0.0
        hybrid_foreland_cells = int(np.count_nonzero(foreland_ring))

        # Gentle maturity pass: smooth artificial lowland ripples/pits and deposit
        # sediment in basins/forelands, but keep young mountains, arcs, and rugged
        # coasts sharp.
        smooth_elev = ndimage.gaussian_filter(elev_full, sigma=(1.1, 1.1), mode=("nearest", "wrap"))
        broader_smooth = ndimage.gaussian_filter(elev_full, sigma=(2.0, 2.0), mode=("nearest", "wrap"))
        lowland_mask = land_full & (final_mountain_strength_full < 0.24) & (active_full_detail < 0.28) & (dist_ocean_full > 3.0)
        basin_mask = lowland_mask & ((rift_full_detail > 0.20) | (plateau_full_detail < 0.20))
        blend = 0.11 + 0.15 * hybrid_strength
        elev_full[basin_mask] = elev_full[basin_mask] * (1.0 - blend) + smooth_elev[basin_mask] * blend
        depositional_mask = (foreland_ring | (land_full & (rift_full_detail > 0.34) & (final_mountain_strength_full < 0.35)))
        deposition = np.maximum(0.0, broader_smooth - elev_full) * (0.20 + 0.20 * hybrid_strength)
        elev_full[depositional_mask] = elev_full[depositional_mask] + deposition[depositional_mask]
        hybrid_erosion_cells = int(np.count_nonzero(basin_mask))
        hybrid_deposition_cells = int(np.count_nonzero(depositional_mask))

    elev_full = np.nan_to_num(elev_full, nan=0.0, posinf=9000.0, neginf=-11000.0)
    elev_full[land_full] = np.maximum(elev_full[land_full], 1.0)
    elev_full[~land_full] = np.minimum(elev_full[~land_full], -1.0)
    elev_full = np.rint(np.clip(elev_full, -11000, 9000)).astype(np.int32)

    min_elev = int(elev_full.min())
    max_elev = int(elev_full.max())
    land_vals = elev_full[land_full]
    ocean_vals = elev_full[~land_full]
    mean_land = float(land_vals.mean()) if land_vals.size else 0.0
    mean_ocean_depth = float((-ocean_vals).mean()) if ocean_vals.size else 0.0
    ocean_fraction = float((~land_full).mean())
    exact_1m_land_cells = int(((elev_full == 1) & land_full).sum())

    # Full-resolution surface-crust diagnostic.  This redo is intentionally
    # diagnostic-only: it does not change land, ocean, or elevation.  It prevents
    # the crust map from claiming that continent-sized final land is ordinary
    # oceanic plateau while still exposing the mismatch in problem_class.
    cont_full = _resize_float_field(cont_score, width, height)
    orogeny_full = _resize_float_field(orogeny, width, height)
    rift_full = _resize_float_field(rift, width, height)
    active_full = _resize_float_field(active_margin, width, height)
    passive_full = _resize_float_field(passive_margin, width, height)
    volcanic_full = _resize_float_field(volcanic, width, height)
    plate_type_full = _resize_int_field(plate_type[present_ids], width, height)
    shelf_width_full = np.maximum(_resize_float_field(shelf_width_cells, width, height), 0.1)
    ocean_full = ~land_full
    ocean_tiled_full = np.tile(ocean_full, (1, 3))
    crop_full = slice(width, width * 2)
    dist_land_full = ndimage.distance_transform_edt(ocean_tiled_full)[:, crop_full].astype(np.float32)

    surface_crust = np.zeros((height, width), dtype=np.int16)
    surface_crust[ocean_full & (dist_land_full <= shelf_width_full)] = 1
    surface_crust[ocean_full & (volcanic_full > 0.52) & (dist_land_full > shelf_width_full)] = 8
    surface_crust[land_full & (cont_full > 0.46)] = 2
    surface_crust[land_full & (cont_full > 0.24) & (cont_full <= 0.46)] = 4
    surface_crust[land_full & ((orogeny_full > 0.36) | (active_full > 0.50))] = 3
    surface_crust[land_full & (rift_full > 0.44) & (cont_full > 0.18)] = 4
    surface_crust[land_full & (plate_type_full == 3)] = 5
    surface_crust[land_full & (arc_full > 0.42)] = 6
    surface_crust[land_full & (volcanic_full > 0.62) & (arc_full <= 0.42) & (cont_full <= 0.24)] = 7

    land_capable = np.isin(surface_crust, [2, 3, 4, 5, 6, 7])
    conflict_initial = land_full & (~land_capable)
    # Assign a plausible surface crust class for display without changing the
    # underlying tectonic fields.  The conflict map preserves where this happened.
    surface_crust[conflict_initial & (arc_full > 0.32)] = 6
    surface_crust[conflict_initial & (volcanic_full > 0.42)] = 7
    surface_crust[conflict_initial & (plate_type_full == 3)] = 5
    still_conflict = land_full & (~np.isin(surface_crust, [2, 3, 4, 5, 6, 7]))
    labels_land_full, n_land_full = ndimage.label(land_full)
    land_areas_full = np.bincount(labels_land_full.ravel(), minlength=n_land_full + 1)
    continent_area_cut = max(128, int(width * height * 0.0025))
    component_promotions = 0
    continent_sized_conflict_cells = 0
    small_island_conflict_cells = 0
    for lab in range(1, n_land_full + 1):
        comp = labels_land_full == lab
        bad = comp & still_conflict
        if not np.any(bad):
            continue
        area_lab = int(land_areas_full[lab])
        if area_lab >= continent_area_cut:
            med_cont = float(np.median(cont_full[comp]))
            med_oro = float(np.median(orogeny_full[comp]))
            cls = 3 if med_oro > 0.34 else (4 if med_cont > 0.20 else 5)
            surface_crust[bad] = cls
            continent_sized_conflict_cells += int(bad.sum())
            component_promotions += 1
        else:
            surface_crust[bad] = 7
            small_island_conflict_cells += int(bad.sum())
    final_unresolved_crust_conflicts = int((land_full & (~np.isin(surface_crust, [2, 3, 4, 5, 6, 7]))).sum())
    crust_conflict_class = np.zeros((height, width), dtype=np.int16)
    crust_conflict_class[conflict_initial] = 1
    if n_land_full:
        safe_labels = np.clip(labels_land_full, 0, len(land_areas_full) - 1)
        crust_conflict_class[(land_areas_full[safe_labels] >= continent_area_cut) & conflict_initial] = 2
    crust_conflict_class[land_full & (~np.isin(surface_crust, [2, 3, 4, 5, 6, 7]))] = 3

    # Diagnostic-only checker/ripple score; no smoothing is applied in this redo.
    checkerboard_transition_score = float(np.mean(land_full != np.roll(land_full, 1, axis=1)) + np.mean(land_full != np.vstack((land_full[0:1, :], land_full[:-1, :])))) / 2.0

    boundary_class = np.zeros((hist_h, hist_w), dtype=np.int32)
    boundary_strength = np.maximum.reduce([convergence, divergence, transform])
    boundary_class[(divergence > convergence * 1.15) & (divergence > transform * 0.75) & (boundary_strength > 0.08)] = 2
    boundary_class[(convergence >= divergence) & (convergence > transform * 0.70) & (boundary_strength > 0.08)] = 1
    boundary_class[(transform > convergence * 0.85) & (transform > divergence * 0.85) & (boundary_strength > 0.08)] = 3
    boundary_class[(subduction > 0.26) & (arc_support > 0.18)] = 6
    margin_class = np.zeros((hist_h, hist_w), dtype=np.int32)
    coast_zone = _land_ocean_transition_zone(land, radius=2)
    margin_class[coast_zone & (passive_margin > 0.35)] = 1
    margin_class[coast_zone & (active_margin > 0.25)] = 2
    margin_class[coast_zone & (rift > 0.30)] = 3
    margin_class[coast_zone & (arc_support > 0.35)] = 4
    crust_class = np.zeros((hist_h, hist_w), dtype=np.int32)
    crust_class[cont_score > 0.42] = 1
    crust_class[(cont_score > 0.22) & (cont_score <= 0.42)] = 2
    crust_class[arc_beads > 0.36] = 4
    crust_class[(plate_type[present_ids] == 3) & (cont_score > 0.18)] = 5
    crust_class[ocean & (crust_class == 0)] = 0
    island_origin = np.zeros((hist_h, hist_w), dtype=np.int32)
    island_origin[land & (arc_beads > 0.40)] = 2
    island_origin[land & (plate_type[present_ids] == 3)] = 3
    island_origin[land & (volcanic > 0.42)] = 4

    def diagf(field):
        return _diagnostic_float_x1000(field, diag_w=width, diag_h=height)

    def diagc(field):
        return _diagnostic_class_raster(field, diag_w=width, diag_h=height)

    terrain = TerrainMap(
        width=width,
        height=height,
        elevation_m=elev_full.astype(int).tolist(),
        is_land=land_full.astype(bool).tolist(),
        min_elevation_m=min_elev,
        max_elevation_m=max_elev,
        mean_land_elevation_m=mean_land,
        mean_ocean_depth_m=mean_ocean_depth,
        ocean_fraction=ocean_fraction,
        land_fraction=1.0 - ocean_fraction,
        source="plate_history_v1 compact time-evolved kinematic plate history",
        planet_radius_earth=float(getattr(planet, "radius_earth", 1.0) or 1.0),
        tectonic_plate_id=diagc(present_ids),
        tectonic_boundary_class=diagc(boundary_class),
        tectonic_boundary_strength_x1000=diagf(boundary_strength),
        tectonic_boundary_width_x1000=diagf(ndimage.gaussian_filter(boundary_strength, sigma=(1.5, 3.0), mode=("nearest", "wrap"))),
        plate_tectonic_plate_type=diagc(plate_type[present_ids]),
        plate_tectonic_plate_topology_problem_class=diagc(plate_junction_class),
        plate_tectonic_continental_crust_x1000=diagf(cont_score),
        plate_tectonic_convergence_x1000=diagf(convergence),
        plate_tectonic_divergence_x1000=diagf(divergence),
        plate_tectonic_transform_x1000=diagf(transform),
        plate_tectonic_boundary_class=diagc(boundary_class),
        plate_tectonic_subduction_polarity=diagc((subduction > 0.25).astype(np.int32) * 2),
        plate_tectonic_ocean_crust_age_x1000=diagf(age_norm),
        plate_tectonic_mid_ocean_ridge_x1000=diagf(ridge),
        plate_tectonic_trench_x1000=diagf(trench),
        plate_tectonic_fracture_zone_x1000=diagf(transform * (1.0 - cont_score)),
        plate_tectonic_seamount_x1000=diagf(volcanic * (1.0 - cont_score)),
        plate_tectonic_orogeny_strength_x1000=diagf(orogeny),
        plate_tectonic_volcanic_arc_x1000=diagf(arc_beads),
        plate_tectonic_continental_rift_x1000=diagf(rift * cont_score),
        plate_tectonic_foreland_basin_x1000=diagf(ndimage.gaussian_filter(orogeny, sigma=(2.0, 4.0), mode=("nearest", "wrap")) * (1.0 - plateau)),
        plate_tectonic_craton_shield_x1000=diagf(cont_score * (1.0 - boundary_strength)),
        plate_tectonic_accreted_terrane_x1000=diagf(subduction * cont_score + arc_beads * 0.4),
        plate_tectonic_plateau_uplift_x1000=diagf(plateau),
        plate_tectonic_sedimentary_plain_x1000=diagf(passive_margin * (1.0 - orogeny)),
        plate_tectonic_margin_class=diagc(margin_class),
        plate_tectonic_shelf_width_x1000=diagf(shelf_width_cells),
        plate_tectonic_active_margin_x1000=diagf(active_margin),
        plate_tectonic_passive_margin_x1000=diagf(passive_margin),
        plate_tectonic_rifted_margin_x1000=diagf(rift),
        plate_tectonic_island_arc_x1000=diagf(arc_beads),
        plate_tectonic_coastal_plain_x1000=diagf(passive_margin * coast_zone.astype(np.float32)),
        plate_tectonic_coast_ruggedness_x1000=diagf(np.abs(medium_noise) * coast_zone.astype(np.float32)),
        plate_tectonic_island_origin_class=diagc(island_origin),
        terrain_mountain_strength_x1000=_diagnostic_float_x1000(final_mountain_strength_full, diag_w=width, diag_h=height),
        terrain_basin_field_x1000=diagf(rift * cont_score + passive_margin * 0.25),
        terrain_rift_field_x1000=diagf(rift),
        terrain_interior_relief_x1000=diagf(suture + detail * 0.15),
        terrain_plateau_x1000=diagf(plateau),
        terrain_shelf_width_x1000=diagf(shelf_width_cells),
        terrain_coast_ruggedness_x1000=diagf(np.abs(medium_noise) * coast_zone.astype(np.float32)),
        terrain_island_origin_class=diagc(island_origin),
        terrain_ocean_floor_class=diagc(((ridge > 0.32).astype(np.int32) * 2) + ((trench > 0.32).astype(np.int32) * 3)),
        terrain_mid_ocean_ridge_x1000=diagf(ridge),
        terrain_trench_x1000=diagf(trench),
        terrain_fracture_zone_x1000=diagf(transform * (1.0 - cont_score)),
        terrain_seamount_x1000=diagf(volcanic * (1.0 - cont_score)),
        crust_type=surface_crust.astype(int).tolist(),
        plate_tectonic_problem_class=crust_conflict_class.astype(int).tolist(),
        terrain_diagnostics={
            "terrain_mode": "plate_history_v1",
            "plate_history_v1": {
                "history_myr": round(history_myr, 3),
                "requested_timestep_myr": round(timestep_myr, 3),
                "requested_steps": int(requested_steps),
                "internal_history_steps": int(history_steps),
                "internal_epoch_myr": round(epoch_myr, 3),
                "tectonic_grid_width": int(hist_w),
                "tectonic_grid_height": int(hist_h),
                "tectonic_grid_scale": str(tectonic_grid_scale),
                "tectonic_grid_policy": str(tectonic_grid_policy),
                "tectonic_grid_capped_from_cells": int(tectonic_grid_capped_from_cells) if tectonic_grid_capped_from_cells else None,
                "tectonic_grid_runtime_mode": "raw_requested_history_grid" if tectonic_grid_policy == "raw" and tectonic_grid_scale != "legacy" else ("stable_macro_hybrid_full_resolution_detail" if float(hybrid_strength) > 0.0 else "legacy_macro_full_resolution_render"),
                "raw_native_history_grid_enabled": bool(tectonic_grid_policy == "raw" and tectonic_grid_scale != "legacy" and bool(getattr(config, "allow_experimental_tectonic_grid", False))),
                "allow_experimental_tectonic_grid": bool(getattr(config, "allow_experimental_tectonic_grid", False)),
                "hybrid_high_resolution_detail_enabled": bool(float(hybrid_strength) > 0.0),
                "hybrid_high_resolution_detail_strength": round(float(hybrid_strength), 3),
                "hybrid_land_changed_cells": int(hybrid_land_changed_cells),
                "hybrid_score_jitter_max": round(float(hybrid_score_jitter_max), 6),
                "hybrid_relief_added_max_m": round(float(hybrid_relief_added_max_m), 3),
                "hybrid_erosion_cells": int(hybrid_erosion_cells),
                "hybrid_deposition_cells": int(hybrid_deposition_cells),
                "hybrid_foreland_cells": int(hybrid_foreland_cells),
                "hybrid_ridge_spine_cells": int(hybrid_ridge_spine_cells),
                "hybrid_arc_split_components": int(hybrid_arc_split_components),
                "hybrid_arc_split_cells_removed": int(hybrid_arc_split_cells_removed),
                "plus_junction_count": int(plus_junction_count),
                "t_triple_junction_count": int(t_triple_junction_count),
                "ordinary_boundary_junction_count": int(ordinary_boundary_junction_count),
                "plate_count": int(plate_count),
                "arc_bead_count": int(len(candidates)),
                "snake_arc_repairs": int(snake_repairs),
                "removed_tiny_components": int(removed_tiny),
                "suppress_polar_land": bool(getattr(config, "suppress_polar_land", False)),
                "exact_1m_land_cells": int(exact_1m_land_cells),
                "checkerboard_transition_score": round(float(checkerboard_transition_score), 6),
                "initial_crust_land_conflict_cells": int(conflict_initial.sum()),
                "continent_sized_crust_conflict_cells": int(continent_sized_conflict_cells),
                "small_island_crust_conflict_cells": int(small_island_conflict_cells),
                "crust_conflict_component_promotions": int(component_promotions),
                "final_unresolved_crust_conflicts": int(final_unresolved_crust_conflicts),
                "surface_crust_rule": "Diagnostic-only final surface-crust map. Final land is displayed with land-capable crust classes while conflicts are preserved in plate_tectonic_problem_class; no land/elevation rewrite is applied.",
                "hybrid_grid_rule": "Non-legacy grid scales keep the stable macro plate history but add high-resolution coastline jitter, visible ridge spines, arc splitting, foreland/deposition relief, and gentle lowland erosion at the requested output resolution. Legacy remains Update 17-compatible.",
                "description": "Compact kinematic plate-history terrain mode: moves synthetic plates over geological time, accumulates crust/boundary fields, and derives final terrain from accumulated tectonic history rather than the post-processing stack used by plate_tectonic_v1.",
            },
        },
    )
    return terrain



def _generate_plate_history_v2_scaffold(
    rng: random.Random,
    planet: Planet,
    hydrosphere: Hydrosphere,
    geology: GeologyState,
    config: PlanetProfileConfig,
    *,
    output_dir: str | None = None,
) -> TerrainMap:
    """Experimental second-generation plate-history terrain mode.

    v2 deliberately does not replace v1. It uses the stable v1 macro plate
    history as the coarse geological reconstruction, then derives a stronger
    structural terrain from the accumulated fields: coast migration, surface
    crust coherence, broken volcanic island centers, explicit mountain ridge
    spines, foreland basins, plateau uplift, and erosion/deposition maturity.
    The important guardrail is that v2 does *not* run the old raw-native plate
    reseeding path that damaged Update 18.
    """
    try:
        import numpy as np
        from scipy import ndimage
    except ImportError as exc:
        raise RuntimeError("NumPy and SciPy are required for plate_history_v2. Install with: pip install -r requirements.txt") from exc

    # Reuse the stable v1 tectonic history and diagnostics as the macro state.
    base = _generate_plate_history_v1_scaffold(rng, planet, hydrosphere, geology, config, output_dir=output_dir)
    width, height = int(base.width), int(base.height)
    np_rng = np.random.default_rng(rng.randint(1, 2_147_483_647))

    elev0 = np.asarray(base.elevation_m, dtype=np.float32)
    land0 = np.asarray(base.is_land, dtype=bool)
    elev = elev0.copy()
    land = land0.copy()

    def _field01(value, default: float = 0.0):
        if value is None:
            return np.full((height, width), float(default), dtype=np.float32)
        arr = np.asarray(value, dtype=np.float32)
        if arr.shape != (height, width):
            arr = _resize_float_field(arr, width, height)
        return np.clip(arr / 1000.0, 0.0, 1.0).astype(np.float32, copy=False)

    def _class_field(value, default: int = 0):
        if value is None:
            return np.full((height, width), int(default), dtype=np.int32)
        arr = np.asarray(value, dtype=np.int32)
        if arr.shape != (height, width):
            arr = _resize_int_field(arr, width, height)
        return arr.astype(np.int32, copy=False)

    def _distance_to_feature_xwrap(feature_mask):
        """Distance to True cells with east/west wrap and polar rows as edges.

        scipy's EDT has no cylindrical topology.  Tiling three copies in X and
        slicing the center keeps continents/islands crossing the date-line from
        becoming artificial map-edge features, while north/south remain poles.
        """
        feature = np.asarray(feature_mask, dtype=bool)
        if feature.shape != (height, width):
            feature = np.asarray(feature, dtype=bool).reshape((height, width))
        if not np.any(feature):
            return np.full((height, width), float(max(width, height)), dtype=np.float32)
        tiled_nonfeature = np.concatenate([~feature, ~feature, ~feature], axis=1)
        dist = ndimage.distance_transform_edt(tiled_nonfeature)
        return dist[:, width:2 * width].astype(np.float32, copy=False)

    def _label_xwrap(mask):
        """Connected components with east/west wrap.

        North/south are not wrapped; east/west are joined so seam-crossing
        continents, oceans, lakes and islands are handled as one component.
        """
        labels, n = ndimage.label(np.asarray(mask, dtype=bool))
        if n <= 1 or width <= 1:
            return labels.astype(np.int32, copy=False), int(n)
        parent = list(range(n + 1))
        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a
        def union(a, b):
            if a == 0 or b == 0:
                return
            ra, rb = find(int(a)), find(int(b))
            if ra != rb:
                parent[rb] = ra
        for yy in range(height):
            union(labels[yy, 0], labels[yy, width - 1])
        remap = {0: 0}
        next_id = 1
        out = np.zeros_like(labels, dtype=np.int32)
        flat_labels = labels.ravel()
        flat_out = out.ravel()
        for i, lab in enumerate(flat_labels):
            if lab == 0:
                continue
            root = find(int(lab))
            if root not in remap:
                remap[root] = next_id
                next_id += 1
            flat_out[i] = remap[root]
        return out, next_id - 1

    convergence = _field01(base.plate_tectonic_convergence_x1000)
    divergence = _field01(base.plate_tectonic_divergence_x1000)
    transform = _field01(base.plate_tectonic_transform_x1000)
    cont = _field01(base.plate_tectonic_continental_crust_x1000)
    orogeny = _field01(base.plate_tectonic_orogeny_strength_x1000)
    arc = np.maximum(_field01(base.plate_tectonic_volcanic_arc_x1000), _field01(base.plate_tectonic_island_arc_x1000))
    rift = np.maximum(_field01(base.plate_tectonic_continental_rift_x1000), _field01(base.terrain_rift_field_x1000))
    plateau = _field01(base.plate_tectonic_plateau_uplift_x1000)
    passive = _field01(base.plate_tectonic_passive_margin_x1000)
    active = _field01(base.plate_tectonic_active_margin_x1000)
    boundary_strength = np.maximum.reduce([convergence, divergence, transform, orogeny * 0.85, arc * 0.7])

    # The scale chooser now controls structural intensity instead of dangerous
    # raw-native plate reseeding.  v2 is intentionally stronger than v1.
    scale = str(getattr(config, "tectonic_grid_scale", "legacy") or "legacy").strip().lower()
    if scale not in {"legacy", "preview", "normal", "high", "native", "custom"}:
        scale = "legacy"
    intensity = {
        "legacy": 0.55,
        "preview": 0.70,
        "normal": 0.90,
        "high": 1.10,
        "native": 1.25,
        "custom": 1.00,
    }.get(scale, 0.75)

    land_dist = _distance_to_feature_xwrap(land)
    ocean_dist = _distance_to_feature_xwrap(~land)
    coast_band = (land_dist <= 5.0) | (ocean_dist <= 5.0)
    near_ocean_land = land & (ocean_dist <= 5.0)
    near_land_ocean = (~land) & (land_dist <= 5.0)

    broad = np_rng.normal(0.0, 1.0, (height, width)).astype(np.float32)
    broad = ndimage.gaussian_filter(broad, sigma=(max(4.0, height / 42.0), max(4.0, width / 42.0)), mode=("nearest", "wrap"))
    broad = (_norm01(broad, 99.0) - 0.5) * 2.0
    detail = np_rng.normal(0.0, 1.0, (height, width)).astype(np.float32)
    detail = ndimage.gaussian_filter(detail, sigma=(2.0, 3.5), mode=("nearest", "wrap"))
    detail = (_norm01(detail, 99.0) - 0.5) * 2.0
    structural_noise = 0.55 * broad + 0.45 * detail

    # Coasts migrate only near existing coasts. This makes v2 visibly different
    # while keeping continents recognizable and avoiding full-world rethresholding.
    coast_push_score = (
        cont * 0.42
        + arc * 0.38
        + active * 0.15
        + structural_noise * 0.22
        - passive * 0.08
        - land_dist.astype(np.float32) * 0.018
    )
    coast_pull_score = (
        -cont * 0.28
        + passive * 0.20
        - active * 0.12
        - structural_noise * 0.24
        + ocean_dist.astype(np.float32) * 0.012
    )
    max_changes = int(max(128, round(width * height * (0.0025 + 0.0035 * intensity))))
    promote_candidates = np.argwhere(near_land_ocean & (coast_push_score > np.percentile(coast_push_score[near_land_ocean], 84) if np.any(near_land_ocean) else False))
    demote_candidates = np.argwhere(near_ocean_land & (coast_pull_score > np.percentile(coast_pull_score[near_ocean_land], 88) if np.any(near_ocean_land) else False) & (elev < 180.0))
    promoted = 0
    demoted = 0
    if len(promote_candidates):
        take_n = min(len(promote_candidates), max_changes // 2)
        scores = coast_push_score[promote_candidates[:, 0], promote_candidates[:, 1]]
        idx = np.argsort(scores)[-take_n:]
        pts = promote_candidates[idx]
        land[pts[:, 0], pts[:, 1]] = True
        elev[pts[:, 0], pts[:, 1]] = np.maximum(elev[pts[:, 0], pts[:, 1]], 1.0 + 90.0 * np.clip(coast_push_score[pts[:, 0], pts[:, 1]], 0.0, 1.0))
        promoted = int(take_n)
    if len(demote_candidates):
        take_n = min(len(demote_candidates), max_changes // 2)
        scores = coast_pull_score[demote_candidates[:, 0], demote_candidates[:, 1]]
        idx = np.argsort(scores)[-take_n:]
        pts = demote_candidates[idx]
        land[pts[:, 0], pts[:, 1]] = False
        elev[pts[:, 0], pts[:, 1]] = np.minimum(elev[pts[:, 0], pts[:, 1]], -25.0 - 85.0 * np.clip(scores[idx], 0.0, 1.0))
        demoted = int(take_n)

    # Arc islands are generated as separated volcanic centers, never as ribbons.
    arc_smooth = ndimage.gaussian_filter(arc, sigma=(1.0, 1.6), mode=("nearest", "wrap"))
    local_arc_max = arc_smooth >= ndimage.maximum_filter(arc_smooth, size=(9, 15), mode=("nearest", "wrap")) - 1.0e-6
    arc_threshold = max(0.28, float(np.percentile(arc_smooth[arc_smooth > 0.0], 72)) if np.any(arc_smooth > 0.0) else 1.0)
    centers = np.argwhere(local_arc_max & (arc_smooth >= arc_threshold))
    max_centers = int(max(12, min(240, (width * height) / 18000.0 * intensity)))
    if len(centers) > max_centers:
        weights = arc_smooth[centers[:, 0], centers[:, 1]]
        order = np.argsort(weights)[-max_centers:]
        centers = centers[order]
    arc_added_cells = 0
    for cy, cx in centers:
        # Mostly add/raise oceanic volcanic centers; avoid painting continuous arcs.
        ry = int(np_rng.integers(2, 5))
        rx = int(np_rng.integers(3, 8))
        amp = float(520.0 + 760.0 * arc_smooth[cy, cx] * intensity)
        for dy in range(-ry * 2, ry * 2 + 1):
            y = cy + dy
            if y < 0 or y >= height:
                continue
            for dx in range(-rx * 2, rx * 2 + 1):
                x = (cx + dx) % width
                d2 = (dy / max(ry, 1)) ** 2 + (dx / max(rx, 1)) ** 2
                if d2 > 3.2:
                    continue
                val = math.exp(-0.72 * d2)
                if val < 0.18:
                    continue
                if not land[y, x] and val > 0.36 and np_rng.random() < 0.72:
                    land[y, x] = True
                    arc_added_cells += 1
                if land[y, x]:
                    elev[y, x] = max(elev[y, x], 20.0 + amp * val)
                else:
                    elev[y, x] = max(elev[y, x], -420.0 + amp * val * 0.55)

    # Visible mountain construction: extract a ridge spine from accumulated
    # collision/subduction and then add narrow ridges plus broader plateaus.
    structural_orogeny = np.maximum(orogeny, np.maximum(convergence * cont, active * 0.75))
    ridge_threshold = max(0.24, float(np.percentile(structural_orogeny[land], 82)) if np.any(land) else 0.7)
    ridge_axis = land & (structural_orogeny >= ridge_threshold) & (structural_orogeny >= ndimage.maximum_filter(structural_orogeny, size=(5, 11), mode=("nearest", "wrap")) - 0.035)
    # Add diagonal/branching ridge segments where transform/convergence interact.
    branch_seed = land & (structural_orogeny > max(0.18, ridge_threshold * 0.72)) & (np.abs(detail) > 0.52) & (boundary_strength > 0.14)
    ridge_axis |= branch_seed & (np_rng.random((height, width)) < 0.18 * intensity)
    # Update 23: add a mountain hierarchy instead of a single sharpened line.
    # Secondary folds form parallel/oblique ridges around the main ridge spine;
    # structural valleys cut between them.  This is still conservative and uses
    # the existing stable macro fields rather than reseeding plates.
    prelim_ridge_dist = ndimage.distance_transform_edt(~ridge_axis)
    fold_noise = ndimage.gaussian_filter(np_rng.normal(0.0, 1.0, (height, width)).astype(np.float32), sigma=(1.4, 3.2), mode=("nearest", "wrap"))
    fold_noise = (_norm01(fold_noise, 98.0) - 0.5) * 2.0
    secondary_fold_seed = (
        land
        & (prelim_ridge_dist >= 4.0)
        & (prelim_ridge_dist <= 18.0)
        & (structural_orogeny > max(0.15, ridge_threshold * 0.55))
        & (boundary_strength > 0.10)
        & (fold_noise > 0.22)
    )
    ridge_axis |= secondary_fold_seed & (np_rng.random((height, width)) < (0.10 + 0.05 * intensity))
    mountain_suture_seed = land & (plateau > 0.18) & (prelim_ridge_dist > 10.0) & (fold_noise < -0.46) & (structural_orogeny > 0.12)
    ridge_axis |= mountain_suture_seed & (np_rng.random((height, width)) < 0.045 * intensity)
    secondary_fold_cells = int(np.count_nonzero(secondary_fold_seed))
    suture_ridge_cells = int(np.count_nonzero(mountain_suture_seed))
    if not np.any(ridge_axis) and np.any(land):
        fallback = land & (structural_orogeny >= (float(np.percentile(structural_orogeny[land], 94)) if np.any(land) else 1.0))
        ridge_axis |= fallback
    ridge_dist = ndimage.distance_transform_edt(~ridge_axis)
    ridge_envelope = ndimage.gaussian_filter(structural_orogeny, sigma=(2.0, 4.0), mode=("nearest", "wrap"))
    narrow_ridge = np.exp(-np.square(ridge_dist / (1.25 + 0.45 * intensity))) * ridge_envelope
    broad_belt = np.exp(-np.square(ridge_dist / (5.5 + 2.0 * intensity))) * ridge_envelope
    ridge_gain = (narrow_ridge ** 1.15) * (820.0 + 420.0 * intensity) * (0.65 + 0.55 * float(getattr(geology, "mountain_factor", 0.8)))
    plateau_gain = np.maximum(plateau, ndimage.gaussian_filter(collision if 'collision' in locals() else structural_orogeny, sigma=(4.0, 8.0), mode=("nearest", "wrap")) if False else plateau)  # kept simple; plateau field comes from v1 diagnostics
    plateau_gain = (np.maximum(plateau, broad_belt * 0.55) ** 1.35) * (260.0 + 220.0 * intensity)
    foreland_ring = land & (ridge_dist > 4.0) & (ridge_dist < 14.0) & (broad_belt > 0.10) & (elev < 900.0)
    foreland_cut = (90.0 + 80.0 * intensity) * broad_belt * foreland_ring.astype(np.float32)
    rift_cut = (rift ** 1.20) * (160.0 + 160.0 * intensity) * land.astype(np.float32)
    ridge_delta = (ridge_gain + plateau_gain - foreland_cut - rift_cut) * land.astype(np.float32)
    elev += ridge_delta
    # Cut structural valleys between ridge families.  These are not hydrology
    # rivers yet; they are tectonic/erosional weaknesses that give mountain
    # systems branching structure and internal relief.
    ridge_shadow = ndimage.gaussian_filter(ridge_axis.astype(np.float32), sigma=(2.4, 3.4), mode=("nearest", "wrap"))
    valley_seed = land & (broad_belt > 0.12) & (narrow_ridge < 0.18) & (ridge_shadow > 0.025) & (fold_noise < -0.18)
    valley_field = ndimage.gaussian_filter(valley_seed.astype(np.float32), sigma=(1.1, 1.8), mode=("nearest", "wrap"))
    valley_field = np.clip(valley_field * (0.7 + 1.2 * broad_belt) * (1.0 - narrow_ridge * 0.75), 0.0, 1.0)
    valley_cut = (80.0 + 135.0 * intensity) * valley_field * land.astype(np.float32)
    elev -= valley_cut
    structural_valley_cells = int(np.count_nonzero(valley_field > 0.08))

    # Mature erosion/deposition: smooth lowland checker/ripple patterns while
    # preserving young ridge spines and volcanic cones.
    smooth3 = ndimage.gaussian_filter(elev, sigma=(1.15, 1.15), mode=("nearest", "wrap"))
    smooth8 = ndimage.gaussian_filter(elev, sigma=(2.2, 2.2), mode=("nearest", "wrap"))
    lowland = land & (elev < 520.0) & (narrow_ridge < 0.12) & (arc < 0.22)
    basin = land & (rift + passive + broad_belt * 0.25 > 0.24) & (elev < 760.0)
    blend = np.clip(0.10 + 0.12 * intensity + passive * 0.12 + rift * 0.08, 0.0, 0.38)
    elev[lowland] = elev[lowland] * (1.0 - blend[lowland]) + smooth3[lowland] * blend[lowland]
    deposition = np.maximum(0.0, smooth8 - elev) * np.clip(0.14 + 0.18 * passive + 0.15 * rift + 0.06 * intensity, 0.0, 0.42)
    elev[basin] += deposition[basin]

    # Update 21: geomorphic transition profiles.  The previous v2 pass still
    # made short, clamp-like shelf and lake rims.  This pass treats coastlines,
    # inland water margins, and ocean floors as age/profile-dependent surfaces
    # instead of hard halos.
    ocean_age = _field01(base.plate_tectonic_ocean_crust_age_x1000, 0.65)
    ridge_field = _field01(base.plate_tectonic_mid_ocean_ridge_x1000)
    trench_field = _field01(base.plate_tectonic_trench_x1000)
    fracture_field = _field01(base.plate_tectonic_fracture_zone_x1000)
    seamount_field = np.maximum(_field01(base.plate_tectonic_seamount_x1000), arc * 0.65)

    land_pre_profile = elev >= 1.0
    ocean_pre_profile = ~land_pre_profile
    coast_to_land = _distance_to_feature_xwrap(land_pre_profile)
    coast_to_ocean = _distance_to_feature_xwrap(~land_pre_profile)

    # Margin profile codes: 0 background, 1 passive/broad shelf, 2 active,
    # 3 volcanic-island apron, 4 microcontinent/transitional, 5 rifted/irregular.
    # Update 22: this is now authoritative inside the profile band.  It blends
    # the old shallow-halo terrain toward one smooth shelf/slope/rise profile
    # instead of adding a second lobe on top of the old halo.
    margin_profile = np.zeros((height, width), dtype=np.int16)
    land_labels_for_profile, n_land_components = _label_xwrap(land_pre_profile)
    land_component_sizes = np.bincount(land_labels_for_profile.ravel()) if n_land_components > 0 else np.array([0])
    small_island_land = np.zeros((height, width), dtype=bool)
    if len(land_component_sizes) > 1:
        # Components smaller than about 0.25% of the map or smaller than a few
        # thousand cells are treated as islands for apron/shelf purposes.  They
        # can still be microcontinents if the crust field supports that.
        island_cutoff = max(64, min(int(width * height * 0.0025), int(width * height * 0.018)))
        small_ids = np.where((land_component_sizes > 0) & (land_component_sizes <= island_cutoff))[0]
        if len(small_ids):
            small_island_land = np.isin(land_labels_for_profile, small_ids)
    small_island_dist = _distance_to_feature_xwrap(small_island_land) if np.any(small_island_land) else np.full((height, width), 1.0e6, dtype=np.float32)

    volcanic_margin = ((arc > 0.23) & (cont < 0.48)) | ((small_island_dist <= 18.0) & (cont < 0.48) & (passive < 0.36))
    passive_margin = (passive > 0.24) & ~volcanic_margin
    active_margin = (active > 0.26) & ~volcanic_margin
    rifted_margin = (rift > 0.26) & ~volcanic_margin
    micro_margin = (cont > 0.16) & (cont < 0.48) & ~passive_margin & ~active_margin & ~volcanic_margin
    margin_profile[passive_margin] = 1
    margin_profile[active_margin] = 2
    margin_profile[volcanic_margin] = 3
    margin_profile[micro_margin] = 4
    margin_profile[rifted_margin] = 5

    d = coast_to_land.astype(np.float32)
    profile_width = np.full((height, width), 18.0, dtype=np.float32)
    profile_width = np.where(passive_margin, 38.0, profile_width)
    profile_width = np.where(active_margin, 15.0, profile_width)
    profile_width = np.where(volcanic_margin, 10.0, profile_width)
    profile_width = np.where(micro_margin, 24.0, profile_width)
    profile_width = np.where(rifted_margin, 26.0, profile_width)
    ocean_profile_zone = ocean_pre_profile & (d <= profile_width)

    abyssal_background = (
        -3650.0
        - 1250.0 * ocean_age
        - 520.0 * np.clip((d - profile_width) / 40.0, 0.0, 1.0)
        + 680.0 * ridge_field
        + 360.0 * seamount_field
        - 1050.0 * trench_field
    ).astype(np.float32)
    default_profile = -85.0 - 265.0 * d - 1180.0 * (1.0 - np.exp(-np.maximum(d - 4.0, 0.0) / 10.0))
    passive_profile = np.where(d <= 9.0, -45.0 - 21.0 * d, -235.0 - 116.0 * (d - 9.0) - 900.0 * (1.0 - np.exp(-np.maximum(d - 9.0, 0.0) / 14.0)))
    active_profile = -80.0 - 360.0 * d - 650.0 * trench_field
    volcanic_profile = -70.0 - 470.0 * np.power(np.maximum(d, 0.0), 0.82) - 110.0 * np.maximum(d - 5.0, 0.0)
    micro_profile = -60.0 - 96.0 * d - 980.0 * (1.0 - np.exp(-np.maximum(d - 8.0, 0.0) / 16.0))
    rifted_profile = -65.0 - 140.0 * d - 690.0 * (1.0 - np.exp(-np.maximum(d - 6.0, 0.0) / 13.0))
    near_profile = default_profile.astype(np.float32)
    near_profile = np.where(passive_margin, passive_profile, near_profile)
    near_profile = np.where(active_margin, active_profile, near_profile)
    near_profile = np.where(volcanic_margin, volcanic_profile, near_profile)
    near_profile = np.where(micro_margin, micro_profile, near_profile)
    near_profile = np.where(rifted_margin, rifted_profile, near_profile)
    blend_to_abyss = np.clip((d / np.maximum(profile_width, 1.0)) ** 1.45, 0.0, 1.0)
    target_bathy = near_profile * (1.0 - blend_to_abyss) + abyssal_background * blend_to_abyss
    # Authoritative but not perfectly flat: keep a small amount of original
    # ocean-floor detail, less around volcanic islands where the old halo was most visible.
    author_blend = np.where(volcanic_margin, 0.88, np.where(active_margin, 0.78, np.where(passive_margin, 0.70, 0.74))).astype(np.float32)
    before_ocean_profile = elev.copy()
    elev[ocean_profile_zone] = elev[ocean_profile_zone] * (1.0 - author_blend[ocean_profile_zone]) + target_bathy[ocean_profile_zone] * author_blend[ocean_profile_zone]
    # Avoid shallow shelf shelves around volcanic islands unless they are true
    # microcontinents: volcanic aprons should rapidly descend to deep water.
    volcanic_outer = ocean_profile_zone & volcanic_margin & (d > 5.0)
    elev[volcanic_outer] = np.minimum(elev[volcanic_outer], volcanic_profile[volcanic_outer] + 120.0)
    ocean_profile_adjusted_cells = int(np.count_nonzero(ocean_profile_zone & (np.abs(elev - before_ocean_profile) > 0.5)))

    # Ocean-floor maturity: old abyssal plains smooth and collect sediment;
    # young ridges, trenches, fracture zones, and seamount provinces stay rough.
    ocean_after_profile = elev < 1.0
    protected_ocean_relief = (ridge_field > 0.20) | (trench_field > 0.18) | (seamount_field > 0.23) | (fracture_field > 0.32)
    old_abyssal = ocean_after_profile & (coast_to_land > 10.0) & (ocean_age > 0.38) & ~protected_ocean_relief
    # Stronger multi-scale smoothing for deep-ocean ripples.  This is applied
    # only where oceanic crust is old/inactive, so ridges/trenches/seamounts stay crisp.
    ocean_smooth_a = ndimage.gaussian_filter(elev, sigma=(2.2, 2.2), mode=("nearest", "wrap"))
    ocean_smooth_b = ndimage.gaussian_filter(elev, sigma=(5.0, 5.0), mode=("nearest", "wrap"))
    ocean_smooth = 0.55 * ocean_smooth_a + 0.45 * ocean_smooth_b
    ocean_maturity_blend = np.clip(0.12 + (ocean_age - 0.25) * 0.62 + passive * 0.14 - ridge_field * 0.32 - trench_field * 0.30, 0.0, 0.62)
    elev[old_abyssal] = elev[old_abyssal] * (1.0 - ocean_maturity_blend[old_abyssal]) + ocean_smooth[old_abyssal] * ocean_maturity_blend[old_abyssal]
    # Sediment apron near broad passive/rifted/microcontinental margins, but not around volcanic islands.
    offshore_sediment = ocean_after_profile & (coast_to_land <= 26.0) & (passive_margin | micro_margin | rifted_margin) & ~volcanic_margin
    sediment_ocean_gain = np.clip(145.0 * passive + 70.0 * rift + 46.0 * cont, 0.0, 210.0) * np.exp(-coast_to_land / 15.0)
    elev[offshore_sediment] += sediment_ocean_gain[offshore_sediment]
    ocean_smoothing_cells = int(np.count_nonzero(old_abyssal))
    ocean_sediment_cells = int(np.count_nonzero(offshore_sediment))
    ocean_ripple_reduced_cells = int(np.count_nonzero(old_abyssal & (np.abs(elev - before_ocean_profile) > 0.5)))

    # Inland water classification and basin profiles.  Use component size to
    # separate the main ocean from lakes/enclosed seas, then smooth only the
    # lowland/endorheic class; rift and mountain lakes may keep steeper walls.
    water_for_components = elev < 1.0
    labels, n_labels = _label_xwrap(water_for_components)
    inland_water = np.zeros_like(water_for_components, dtype=bool)
    main_ocean_label = 0
    if n_labels > 0:
        counts = np.bincount(labels.ravel())
        if len(counts) > 1:
            main_ocean_label = int(np.argmax(counts[1:]) + 1)
            inland_water = water_for_components & (labels != main_ocean_label)
    inland_dist = _distance_to_feature_xwrap(inland_water)
    inland_margin = land_pre_profile & (inland_dist > 0.0) & (inland_dist <= 18.0)
    mountain_lake_margin = inland_margin & ((structural_orogeny > 0.34) | (narrow_ridge > 0.16))
    rift_lake_margin = inland_margin & (rift > 0.25)
    lowland_lake_margin = inland_margin & ~(mountain_lake_margin | rift_lake_margin)
    lake_profile_ceiling = 240.0 + 115.0 * np.power(np.maximum(inland_dist, 0.0), 1.25) + 260.0 * rift + 520.0 * structural_orogeny
    before_lake_profile = elev.copy()
    elev[lowland_lake_margin] = np.minimum(elev[lowland_lake_margin], lake_profile_ceiling[lowland_lake_margin])
    # Very old/low-energy lake margins get a mild sediment fill rather than a perfect rim.
    lake_smooth = ndimage.gaussian_filter(elev, sigma=(1.35, 1.35), mode=("nearest", "wrap"))
    lake_blend = np.clip(0.12 + 0.08 * intensity + passive * 0.05, 0.0, 0.26)
    elev[lowland_lake_margin] = elev[lowland_lake_margin] * (1.0 - lake_blend[lowland_lake_margin]) + lake_smooth[lowland_lake_margin] * lake_blend[lowland_lake_margin]
    inland_lake_profile_cells = int(np.count_nonzero(inland_margin))
    inland_lake_cliff_reduced_cells = int(np.count_nonzero(lowland_lake_margin & (elev < before_lake_profile - 0.5)))
    # Avoid all enclosed terrain-water bodies becoming a flat -1m clamp.
    # This is still a below-sea-level terrain-water approximation, not a full
    # lake-surface model, but it makes lake/seaway floors varied and diagnosable.
    lake_floor_noise = ndimage.gaussian_filter(np_rng.normal(0.0, 1.0, (height, width)).astype(np.float32), sigma=(1.5, 2.5), mode=("nearest", "wrap"))
    lake_floor_noise = _norm01(lake_floor_noise, 98.0)
    lake_depth_target = -6.0 - 42.0 * lake_floor_noise - 85.0 * rift - 36.0 * structural_orogeny
    deep_inland_water = inland_water & (elev > lake_depth_target)
    elev[deep_inland_water] = lake_depth_target[deep_inland_water]
    varied_lake_depth_cells = int(np.count_nonzero(deep_inland_water))

    # Age-aware erosion/deposition and sliding.  Approximate age from history
    # length and current activity: old, inactive belts are rounded; active ridges
    # and volcanic cones remain sharper.
    history_myr = getattr(config, "tectonic_history_myr", None)
    try:
        history_norm = min(1.0, max(0.0, float(history_myr or 750.0) / 1500.0))
    except (TypeError, ValueError):
        history_norm = 0.5
    young_activity = np.maximum.reduce([structural_orogeny, arc_smooth * 0.9, ridge_field * 0.75, trench_field * 0.65, divergence * 0.55])
    erosion_maturity = np.clip(history_norm * (1.0 - young_activity * 0.75) + passive * 0.18 + ocean_age * 0.10, 0.0, 1.0)
    old_mountain = land_pre_profile & (broad_belt > 0.14) & (narrow_ridge < 0.22) & (young_activity < 0.34)
    young_mountain = land_pre_profile & ((narrow_ridge > 0.18) | (arc_smooth > 0.30)) & (young_activity >= 0.22)
    old_smooth = ndimage.gaussian_filter(elev, sigma=(2.0, 1.7), mode=("nearest", "wrap"))
    old_blend = np.clip(0.10 + 0.22 * erosion_maturity, 0.0, 0.32)
    elev[old_mountain] = elev[old_mountain] * (1.0 - old_blend[old_mountain]) + old_smooth[old_mountain] * old_blend[old_mountain]
    # Sliding on non-active steep low/intermediate slopes.
    gy, gx = np.gradient(elev.astype(np.float32))
    slope_proxy = np.hypot(gx, gy)
    slide_zone = land_pre_profile & (slope_proxy > 360.0) & (young_activity < 0.30) & (narrow_ridge < 0.18)
    elev[slide_zone] = elev[slide_zone] * 0.72 + old_smooth[slide_zone] * 0.28
    age_erosion_cells = int(np.count_nonzero(old_mountain | slide_zone | old_abyssal))
    young_mountain_preserved_cells = int(np.count_nonzero(young_mountain))

    # Diagnostic terrain/ocean classes for the new geomorphic pass.
    ocean_floor_class = np.zeros((height, width), dtype=np.int16)
    ocean_floor_class[ocean_after_profile] = 1
    ocean_floor_class[ocean_after_profile & (ridge_field > 0.24)] = 2
    ocean_floor_class[ocean_after_profile & (trench_field > 0.24)] = 3
    ocean_floor_class[ocean_after_profile & (fracture_field > 0.28)] = 4
    ocean_floor_class[ocean_after_profile & (seamount_field > 0.25)] = 5
    ocean_floor_class[ocean_profile_zone] = np.where(ocean_floor_class[ocean_profile_zone] == 1, 6, ocean_floor_class[ocean_profile_zone])

    geomorphic_erosion = np.clip(lowland.astype(np.float32) * blend + old_mountain.astype(np.float32) * old_blend + old_abyssal.astype(np.float32) * ocean_maturity_blend + slide_zone.astype(np.float32) * 0.28, 0.0, 1.0)
    geomorphic_deposition = np.clip(deposition / 450.0 + offshore_sediment.astype(np.float32) * (sediment_ocean_gain / 300.0) + lowland_lake_margin.astype(np.float32) * 0.16, 0.0, 1.0)

    # Re-apply land/ocean bounds and keep coastal cliffs moderate unless a real
    # active mountain arc is present.
    land = elev >= 1.0
    ocean = ~land
    if np.any(land):
        elev[land] = np.maximum(elev[land], 1.0)
    if np.any(ocean):
        elev[ocean] = np.minimum(elev[ocean], -1.0)
    sea_land = land & (_distance_to_feature_xwrap(~land) <= 3.0) & (active < 0.33) & (narrow_ridge < 0.18)
    if np.any(sea_land):
        elev[sea_land] = np.minimum(elev[sea_land], 360.0 + 140.0 * structural_orogeny[sea_land])
    near_coast_ocean = ocean & (_distance_to_feature_xwrap(land) <= 4.0)
    if np.any(near_coast_ocean):
        # Gentle near-coast bound only; volcanic island aprons and active margins
        # may stay steep and deep, avoiding a renewed uniform shelf halo.
        gentle_near = near_coast_ocean & ~volcanic_margin & (trench_field < 0.30)
        elev[gentle_near] = np.maximum(elev[gentle_near], -520.0 - 260.0 * active[gentle_near])

    # Final post-deposition validator for volcanic/arc snake islands.  Deposition
    # can reconnect bead chains; sink low connector cells back into shallow
    # seamount ridges while preserving the volcanic centers.
    snake_arc_repairs_v2 = 0
    land_for_snake = elev >= 1.0
    island_labels, island_n = _label_xwrap(land_for_snake)
    for lab in range(1, island_n + 1):
        comp = island_labels == lab
        size = int(np.count_nonzero(comp))
        if size < 80:
            continue
        ys, xs = np.nonzero(comp)
        if len(xs) == 0:
            continue
        span_y = max(1, int(ys.max() - ys.min() + 1))
        # seam-safe approximate X span: use sorted largest gap.
        xsort = np.sort(xs)
        gaps = np.diff(np.concatenate([xsort, xsort[:1] + width]))
        span_x = max(1, int(width - gaps.max())) if len(gaps) else 1
        aspect = max(span_x / span_y, span_y / span_x)
        mean_arc = float(np.mean(arc_smooth[comp])) if np.any(comp) else 0.0
        if aspect < 7.5 or mean_arc < 0.16:
            continue
        comp_elev = elev[comp]
        keep_cut = max(120.0, float(np.percentile(comp_elev, 72)))
        keep = comp & ((elev >= keep_cut) | (arc_smooth >= max(0.24, float(np.percentile(arc_smooth[comp], 78)))))
        keep = ndimage.binary_dilation(keep, iterations=2) & comp
        sink = comp & ~keep & (elev < 700.0)
        if np.count_nonzero(sink) < max(8, int(size * 0.08)):
            continue
        elev[sink] = np.minimum(elev[sink], -35.0 - 120.0 * arc_smooth[sink])
        snake_arc_repairs_v2 += int(np.count_nonzero(sink))

    elev_int = np.rint(elev).astype(np.int32)
    land = elev_int >= 1
    ocean = ~land
    if np.any(ocean):
        elev_int[ocean] = np.minimum(elev_int[ocean], -1)
    if np.any(land):
        elev_int[land] = np.maximum(elev_int[land], 1)

    # Surface crust is diagnostic, but v2 makes it coherent with final terrain.
    surface_crust = _class_field(base.crust_type)
    surface_crust[land & (cont > 0.45)] = 1       # continental
    surface_crust[land & (cont <= 0.45) & (cont > 0.18)] = 2  # transitional/rifted
    surface_crust[land & (arc > 0.28)] = 4        # volcanic arc/island crust
    surface_crust[land & (cont <= 0.18) & (arc <= 0.28)] = 5  # microcontinent/accreted terrane
    surface_crust[ocean & (arc > 0.28)] = 6       # submerged seamount/arc crust
    surface_crust[ocean & (arc <= 0.28)] = 0

    # Diagnostic-only plate boundary deformation: roughen the plate map shown to
    # the user without changing the stable macro ownership used to seed v1.
    plate_id = _class_field(base.tectonic_plate_id)
    boundary = plate_id != np.roll(plate_id, -1, axis=1)
    boundary |= plate_id != np.roll(plate_id, 1, axis=1)
    boundary |= plate_id != np.vstack((plate_id[:1, :], plate_id[:-1, :]))
    boundary |= plate_id != np.vstack((plate_id[1:, :], plate_id[-1:, :]))
    # Update 23: stronger non-Voronoi diagnostic deformation.  Plate ownership
    # is not globally reseeded, but boundary cells are exchanged along warped
    # salients and transform-like slivers so displayed plates are less clean.
    rough_noise = ndimage.gaussian_filter(np_rng.normal(0.0, 1.0, (height, width)).astype(np.float32), sigma=(2.2, 4.0), mode=("nearest", "wrap"))
    fault_noise = ndimage.gaussian_filter(np_rng.normal(0.0, 1.0, (height, width)).astype(np.float32), sigma=(0.9, 8.0), mode=("nearest", "wrap"))
    active_boundary = boundary & (boundary_strength > (float(np.percentile(boundary_strength[boundary], 40)) if np.any(boundary) else 1.0))
    exchange = active_boundary & (np.abs(rough_noise) > np.percentile(np.abs(rough_noise[active_boundary]), 52) if np.any(active_boundary) else False)
    sliver_exchange = active_boundary & (np.abs(fault_noise) > np.percentile(np.abs(fault_noise[active_boundary]), 84) if np.any(active_boundary) else False)
    rough_plate_id = plate_id.copy()
    east_plate = np.roll(plate_id, -1, axis=1)
    west_plate = np.roll(plate_id, 1, axis=1)
    rough_plate_id[exchange & (rough_noise > 0)] = east_plate[exchange & (rough_noise > 0)]
    rough_plate_id[exchange & (rough_noise <= 0)] = west_plate[exchange & (rough_noise <= 0)]
    north_plate = np.vstack((plate_id[:1, :], plate_id[:-1, :]))
    south_plate = np.vstack((plate_id[1:, :], plate_id[-1:, :]))
    rough_plate_id[sliver_exchange & (fault_noise > 0)] = north_plate[sliver_exchange & (fault_noise > 0)]
    rough_plate_id[sliver_exchange & (fault_noise <= 0)] = south_plate[sliver_exchange & (fault_noise <= 0)]

    b_e = rough_plate_id != np.roll(rough_plate_id, -1, axis=1)
    b_w = rough_plate_id != np.roll(rough_plate_id, 1, axis=1)
    b_n = rough_plate_id != np.vstack((rough_plate_id[:1, :], rough_plate_id[:-1, :]))
    b_s = rough_plate_id != np.vstack((rough_plate_id[1:, :], rough_plate_id[-1:, :]))
    dir_count = b_e.astype(np.int16) + b_w.astype(np.int16) + b_n.astype(np.int16) + b_s.astype(np.int16)
    topology = np.zeros((height, width), dtype=np.int16)
    topology[dir_count >= 1] = 1
    topology[dir_count == 3] = 2
    topology[dir_count >= 4] = 3
    plus_junction_raw = topology == 3
    # Convert a subset of clean + intersections to T/triple diagnostic class by
    # favoring the stronger relative-motion direction. This does not yet move
    # continents, but it prevents the diagnostic map from implying perfect
    # four-way plate crosses everywhere.
    plus_break = plus_junction_raw & (np.abs(rough_noise) > 0.15)
    topology[plus_break] = 2
    plus_junction_resolved = int(np.count_nonzero(plus_break))

    min_elev = int(elev_int.min())
    max_elev = int(elev_int.max())
    ocean_fraction = float(np.mean(~land))
    mean_land = float(elev_int[land].mean()) if np.any(land) else 0.0
    mean_ocean_depth = float((-elev_int[ocean]).mean()) if np.any(ocean) else 0.0

    terrain = replace(
        base,
        elevation_m=elev_int.astype(int).tolist(),
        is_land=land.astype(bool).tolist(),
        min_elevation_m=min_elev,
        max_elevation_m=max_elev,
        mean_land_elevation_m=mean_land,
        mean_ocean_depth_m=mean_ocean_depth,
        ocean_fraction=ocean_fraction,
        land_fraction=1.0 - ocean_fraction,
        source="plate_history_v2 structural crust/ridge/arc terrain reconstruction",
        tectonic_plate_id=_diagnostic_class_raster(rough_plate_id, diag_w=width, diag_h=height),
        plate_tectonic_plate_topology_problem_class=_diagnostic_class_raster(topology, diag_w=width, diag_h=height),
        crust_type=surface_crust.astype(int).tolist(),
        terrain_mountain_strength_x1000=_diagnostic_float_x1000(np.maximum(narrow_ridge, broad_belt), diag_w=width, diag_h=height),
        terrain_valley_corridor_x1000=_diagnostic_float_x1000(valley_field, diag_w=width, diag_h=height),
        plate_tectonic_valley_corridor_x1000=_diagnostic_float_x1000(valley_field, diag_w=width, diag_h=height),
        plate_tectonic_orogeny_strength_x1000=_diagnostic_float_x1000(structural_orogeny, diag_w=width, diag_h=height),
        plate_tectonic_volcanic_arc_x1000=_diagnostic_float_x1000(arc_smooth, diag_w=width, diag_h=height),
        plate_tectonic_foreland_basin_x1000=_diagnostic_float_x1000(foreland_ring.astype(np.float32) * broad_belt, diag_w=width, diag_h=height),
        plate_tectonic_plateau_uplift_x1000=_diagnostic_float_x1000(np.maximum(plateau, broad_belt * 0.55), diag_w=width, diag_h=height),
        plate_tectonic_margin_profile_class=_diagnostic_class_raster(margin_profile, diag_w=width, diag_h=height),
        terrain_ocean_floor_class=_diagnostic_class_raster(ocean_floor_class, diag_w=width, diag_h=height),
        terrain_erosion_strength_x1000=_diagnostic_float_x1000(geomorphic_erosion, diag_w=width, diag_h=height),
        terrain_deposition_field_x1000=_diagnostic_float_x1000(geomorphic_deposition, diag_w=width, diag_h=height),
        terrain_sediment_supply_x1000=_diagnostic_float_x1000(np.clip(sediment_ocean_gain / 300.0 + passive * 0.35 + rift * 0.20, 0.0, 1.0), diag_w=width, diag_h=height),
    )
    # Lightweight tectonic-history stage snapshots.  These are diagnostic
    # reconstructions from the accumulated final fields, not full climate/
    # hydrology reruns at every epoch.  They make the simulated history visible
    # without multiplying runtime by five.
    snapshot_files: list[str] = []
    if output_dir:
        try:
            from pathlib import Path
            from PIL import Image, ImageDraw

            snap_dir = Path(output_dir) / "tectonic_history"
            snap_dir.mkdir(parents=True, exist_ok=True)
            snap_w = min(1024, width)
            snap_h = max(1, int(round(snap_w * height / max(width, 1))))

            def _save_snap(name: str, rgb_arr) -> None:
                img = Image.fromarray(np.asarray(rgb_arr, dtype=np.uint8), mode="RGB")
                if img.size != (snap_w, snap_h):
                    img = img.resize((snap_w, snap_h), Image.Resampling.BILINEAR)
                img.save(snap_dir / name)
                snapshot_files.append(f"tectonic_history/{name}")

            base_boundary = np.clip(boundary_strength, 0.0, 1.0)
            uplift = np.clip(structural_orogeny + narrow_ridge * 0.8 + arc_smooth * 0.45, 0.0, 1.0)
            subsidence = np.clip(rift * 0.70 + trench_field * 0.85 + foreland_ring.astype(np.float32) * broad_belt * 0.45, 0.0, 1.0)
            strong_boundary_cut = float(np.percentile(base_boundary[base_boundary > 0.0], 88)) if np.any(base_boundary > 0.0) else 1.0
            strong_boundary = base_boundary >= max(0.24, strong_boundary_cut)
            conv_edge = strong_boundary & (convergence >= np.maximum(divergence, transform))
            div_edge = strong_boundary & (divergence > np.maximum(convergence, transform))
            trans_edge = strong_boundary & ~(conv_edge | div_edge)
            for pct, factor in [(0, 0.00), (25, 0.25), (50, 0.50), (75, 0.75), (100, 1.00)]:
                c = np.clip(cont * (0.35 + 0.65 * factor), 0.0, 1.0)
                oa = np.clip(ocean_age * factor, 0.0, 1.0)
                crust_rgb = np.zeros((height, width, 3), dtype=np.uint8)
                crust_rgb[..., 0] = np.where(land, (70 + 130 * c).astype(np.uint8), (8 + 25 * oa).astype(np.uint8))
                crust_rgb[..., 1] = np.where(land, (88 + 108 * c).astype(np.uint8), (28 + 70 * (1.0 - oa)).astype(np.uint8))
                crust_rgb[..., 2] = np.where(land, (42 + 45 * c).astype(np.uint8), (92 + 130 * oa).astype(np.uint8))
                _save_snap(f"stage_{pct:03d}_crust_type.png", crust_rgb)

                boundary_rgb = np.zeros((height, width, 3), dtype=np.uint8)
                boundary_rgb[land] = np.array([82, 93, 55], dtype=np.uint8)
                boundary_rgb[~land] = np.array([18, 47, 88], dtype=np.uint8)
                # Show only active/strong boundaries, not the full diffuse influence web.
                boundary_rgb[conv_edge] = np.array([238, 92, 74], dtype=np.uint8)
                boundary_rgb[div_edge] = np.array([250, 204, 84], dtype=np.uint8)
                boundary_rgb[trans_edge] = np.array([190, 130, 245], dtype=np.uint8)
                _save_snap(f"stage_{pct:03d}_active_boundaries.png", boundary_rgb)

                age_rgb = np.zeros((height, width, 3), dtype=np.uint8)
                age_rgb[land] = np.array([86, 92, 58], dtype=np.uint8)
                age_rgb[..., 0] = np.where(~land, (15 + 30 * (1.0 - oa)).astype(np.uint8), age_rgb[..., 0])
                age_rgb[..., 1] = np.where(~land, (70 + 50 * (1.0 - oa)).astype(np.uint8), age_rgb[..., 1])
                age_rgb[..., 2] = np.where(~land, (105 + 135 * oa).astype(np.uint8), age_rgb[..., 2])
                _save_snap(f"stage_{pct:03d}_ocean_crust_age.png", age_rgb)

                relief_rgb = np.zeros((height, width, 3), dtype=np.uint8)
                up = np.clip(uplift * factor, 0.0, 1.0)
                down = np.clip(subsidence * factor, 0.0, 1.0)
                relief_rgb[..., 0] = (35 + 205 * up).astype(np.uint8)
                relief_rgb[..., 1] = (40 + 150 * np.clip(1.0 - np.maximum(up, down) * 0.5, 0.0, 1.0)).astype(np.uint8)
                relief_rgb[..., 2] = (50 + 190 * down).astype(np.uint8)
                relief_rgb[~land] = np.maximum(relief_rgb[~land], np.array([20, 45, 95], dtype=np.uint8))
                _save_snap(f"stage_{pct:03d}_uplift_subsidence.png", relief_rgb)

            # Seam diagnostic: red/blue mark high mismatch between first and last columns.
            seam_rgb = np.zeros((height, width, 3), dtype=np.uint8)
            seam_rgb[land] = np.array([80, 95, 55], dtype=np.uint8)
            seam_rgb[~land] = np.array([22, 58, 100], dtype=np.uint8)
            seam_delta = np.abs(elev_int[:, 0].astype(np.float32) - elev_int[:, -1].astype(np.float32))
            seam_bad = seam_delta > 250.0
            seam_rgb[:, 0][seam_bad] = np.array([255, 70, 60], dtype=np.uint8)
            seam_rgb[:, -1][seam_bad] = np.array([255, 70, 60], dtype=np.uint8)
            seam_land_mismatch = land[:, 0] != land[:, -1]
            seam_rgb[:, 0][seam_land_mismatch] = np.array([255, 220, 60], dtype=np.uint8)
            seam_rgb[:, -1][seam_land_mismatch] = np.array([255, 220, 60], dtype=np.uint8)
            _save_snap("stage_100_seam_diagnostics.png", seam_rgb)
        except Exception as exc:  # diagnostics must never fail terrain generation
            snapshot_files.append(f"snapshot_generation_failed:{type(exc).__name__}:{exc}")

    diag = dict(getattr(base, "terrain_diagnostics", None) or {})
    v1_diag = dict(diag.get("plate_history_v1", {}))
    diag["terrain_mode"] = "plate_history_v2"
    diag["plate_history_v2"] = {
        "base_mode": "plate_history_v1",
        "tectonic_grid_scale": scale,
        "structural_intensity": round(float(intensity), 3),
        "coastal_promoted_cells": int(promoted),
        "coastal_demoted_cells": int(demoted),
        "arc_centers": int(len(centers)),
        "arc_added_cells": int(arc_added_cells),
        "ridge_axis_cells": int(np.count_nonzero(ridge_axis)),
        "secondary_fold_seed_cells": int(secondary_fold_cells),
        "suture_ridge_seed_cells": int(suture_ridge_cells),
        "structural_valley_cells": int(structural_valley_cells),
        "foreland_cells": int(np.count_nonzero(foreland_ring)),
        "erosion_lowland_cells": int(np.count_nonzero(lowland)),
        "deposition_basin_cells": int(np.count_nonzero(basin)),
        "margin_profile_counts": {
            "passive_broad_shelf": int(np.count_nonzero(margin_profile == 1)),
            "active_narrow_shelf": int(np.count_nonzero(margin_profile == 2)),
            "volcanic_island_apron": int(np.count_nonzero(margin_profile == 3)),
            "microcontinent_transitional": int(np.count_nonzero(margin_profile == 4)),
            "rifted_irregular": int(np.count_nonzero(margin_profile == 5)),
        },
        "ocean_profile_adjusted_cells": int(ocean_profile_adjusted_cells),
        "ocean_floor_smoothed_cells": int(ocean_smoothing_cells),
        "ocean_sediment_cells": int(ocean_sediment_cells),
        "ocean_ripple_reduced_cells": int(ocean_ripple_reduced_cells),
        "inland_water_component_count": int(max(n_labels - 1, 0)),
        "inland_lake_profile_cells": int(inland_lake_profile_cells),
        "inland_lake_cliff_reduced_cells": int(inland_lake_cliff_reduced_cells),
        "varied_lake_depth_cells": int(varied_lake_depth_cells),
        "rift_lake_margin_cells_preserved": int(np.count_nonzero(rift_lake_margin)),
        "mountain_lake_margin_cells_preserved": int(np.count_nonzero(mountain_lake_margin)),
        "age_erosion_cells": int(age_erosion_cells),
        "young_mountain_cells_preserved": int(young_mountain_preserved_cells),
        "old_mountain_cells_eroded": int(np.count_nonzero(old_mountain)),
        "sliding_cells": int(np.count_nonzero(slide_zone)),
        "tectonic_history_snapshot_count": int(len([s for s in snapshot_files if not s.startswith('snapshot_generation_failed:')])),
        "tectonic_history_snapshot_files": list(snapshot_files),
        "post_deposition_snake_arc_cells_sunk": int(snake_arc_repairs_v2),
        "plate_boundary_exchange_cells": int(np.count_nonzero(exchange)),
        "plate_boundary_sliver_exchange_cells": int(np.count_nonzero(sliver_exchange)),
        "plus_junctions_resolved_to_t_candidates": int(plus_junction_resolved),
        "plus_junction_count": int(np.count_nonzero(topology == 3)),
        "t_triple_junction_count": int(np.count_nonzero(topology == 2)),
        "land_changed_cells_from_v1": int(np.count_nonzero(land != land0)),
        "elevation_changed_cells_from_v1": int(np.count_nonzero(elev_int != np.rint(elev0).astype(np.int32))),
        "mean_abs_elevation_delta_m": round(float(np.mean(np.abs(elev_int.astype(np.float32) - elev0))), 3),
        "max_abs_elevation_delta_m": int(np.max(np.abs(elev_int.astype(np.float32) - elev0))) if elev_int.size else 0,
        "raw_native_history_grid_enabled": False,
        "description": "Experimental v2 mode with Update 22 artifact cleanup and Update 23 structural topology/mountain refinement: stable v1 macro plate history plus structural coasts, bead volcanic arcs, hierarchical mountain ridges/folds/valleys, more deformed diagnostic plate boundaries, authoritative margin bathymetry profiles, x-wrap-aware distance/component logic, deep-ocean maturity smoothing, varied lake floors, post-deposition snake-island cleanup, and split/readable tectonic-history diagnostics.",
        "v1_macro_history": v1_diag,
    }
    terrain.terrain_diagnostics = diag
    return terrain



def _generate_plate_history_v3_scaffold(
    rng: random.Random,
    planet: Planet,
    hydrosphere: Hydrosphere,
    geology: GeologyState,
    config: PlanetProfileConfig,
    *,
    output_dir: str | None = None,
) -> TerrainMap:
    """Experimental third-generation plate-history terrain mode.

    v3 is intentionally isolated from v1/v2.  It uses v2 only as a macro-field
    source, then rebuilds final terrain from continuous circumstances instead of
    assigning separate hard rule sets to different cell classes.  Every cell is
    processed by the same field equations: crust thickness, buoyancy, uplift,
    subsidence, sediment, age/maturity, slope, water proximity, and plate motion
    determine the outcome.  Discrete rasters written by this function are
    diagnostics explaining the dominant circumstance, not switches that choose
    separate terrain rules.
    """
    try:
        import numpy as np
        from scipy import ndimage
    except ImportError as exc:
        raise RuntimeError("NumPy and SciPy are required for plate_history_v3. Install with: pip install -r requirements.txt") from exc

    # Use v2 as a macro state provider, not as the final authority.  This keeps
    # the mode isolated while allowing v3 to replace the conflicting downstream
    # halo/erosion/lake logic with a single continuous reconstruction.
    base = _generate_plate_history_v2_scaffold(rng, planet, hydrosphere, geology, config, output_dir=output_dir)
    width, height = int(base.width), int(base.height)
    np_rng = np.random.default_rng(rng.randint(1, 2_147_483_647))

    try:
        from worldgen.terrain_review import derive_terrain_controls
        terrain_controls = derive_terrain_controls(planet, hydrosphere, geology, config, output_dir=output_dir)
    except Exception:
        terrain_controls = {}

    def _control_float(name: str, default: float, low: float, high: float) -> float:
        try:
            value = terrain_controls.get(name, default) if isinstance(terrain_controls, dict) else default
            if value is None or value == "":
                value = default
            return clamp(float(value), low, high)
        except Exception:
            return clamp(float(default), low, high)

    # Update 27C: keep the old derived 0..1 terrain controls, but add
    # multiplier-style v3 knobs that are easy to sweep from the CLI/Web UI.
    # Defaults are intentionally stronger than 27B because feedback still showed
    # undersized shelves and too little erosion/deposition.
    v3_erosion_deposition_multiplier = _control_float("erosion_deposition_multiplier", 1.35, 0.0, 3.0)
    v3_continental_shelf_strength = _control_float("continental_shelf_strength", 1.65, 0.0, 3.5)
    v3_shelf_width_factor = _control_float("shelf_width_factor", 0.90, 0.0, 2.5)

    elev_base = np.asarray(base.elevation_m, dtype=np.float32)
    land_base = np.asarray(base.is_land, dtype=bool)

    def _field01(value, default: float = 0.0):
        if value is None:
            return np.full((height, width), float(default), dtype=np.float32)
        arr = np.asarray(value, dtype=np.float32)
        if arr.shape != (height, width):
            arr = _resize_float_field(arr, width, height)
        # Most diagnostics are 0..1000, but already-normalized fields are also accepted.
        if float(np.nanmax(arr)) > 1.5 or float(np.nanmin(arr)) < -0.05:
            arr = arr / 1000.0
        return np.clip(np.nan_to_num(arr, nan=0.0), 0.0, 1.0).astype(np.float32, copy=False)

    def _class_field(value, default: int = 0):
        if value is None:
            return np.full((height, width), int(default), dtype=np.int32)
        arr = np.asarray(value, dtype=np.int32)
        if arr.shape != (height, width):
            arr = _resize_int_field(arr, width, height)
        return arr.astype(np.int32, copy=False)

    def _dist_xwrap(feature_mask):
        feature = np.asarray(feature_mask, dtype=bool)
        if not np.any(feature):
            return np.full((height, width), float(max(width, height)), dtype=np.float32)
        tiled = np.concatenate([~feature, ~feature, ~feature], axis=1)
        return ndimage.distance_transform_edt(tiled)[:, width:2 * width].astype(np.float32, copy=False)

    def _smooth(arr, sy: float, sx: float):
        return ndimage.gaussian_filter(np.asarray(arr, dtype=np.float32), sigma=(float(sy), float(sx)), mode=("nearest", "wrap")).astype(np.float32, copy=False)

    def _var_smooth(current, weight, smooth_field):
        w = np.clip(weight, 0.0, 1.0).astype(np.float32, copy=False)
        return (current * (1.0 - w) + smooth_field * w).astype(np.float32, copy=False)

    def _label_xwrap(mask):
        labels, n = ndimage.label(np.asarray(mask, dtype=bool))
        if n <= 1 or width <= 1:
            return labels.astype(np.int32, copy=False), int(n)
        parent = list(range(n + 1))
        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a
        def union(a, b):
            if a == 0 or b == 0:
                return
            ra, rb = find(int(a)), find(int(b))
            if ra != rb:
                parent[rb] = ra
        for yy in range(height):
            union(labels[yy, 0], labels[yy, width - 1])
        remap = {0: 0}
        next_id = 1
        out = np.zeros_like(labels, dtype=np.int32)
        for idx, lab in np.ndenumerate(labels):
            if lab == 0:
                continue
            root = find(int(lab))
            if root not in remap:
                remap[root] = next_id
                next_id += 1
            out[idx] = remap[root]
        return out, next_id - 1

    def _dominant_adjacent_plate(plate_arr, comp_mask, fallback: int) -> int:
        """Most common neighboring plate around a component, with x-wrap."""
        vals = []
        for rolled in (
            np.roll(plate_arr, 1, axis=1),
            np.roll(plate_arr, -1, axis=1),
        ):
            neigh = rolled[comp_mask]
            vals.append(neigh[neigh != fallback])
        north_mask = np.zeros_like(comp_mask, dtype=bool)
        south_mask = np.zeros_like(comp_mask, dtype=bool)
        north_mask[1:, :] = comp_mask[:-1, :]
        south_mask[:-1, :] = comp_mask[1:, :]
        if np.any(north_mask):
            neigh = plate_arr[north_mask]
            vals.append(neigh[neigh != fallback])
        if np.any(south_mask):
            neigh = plate_arr[south_mask]
            vals.append(neigh[neigh != fallback])
        if vals:
            all_vals = np.concatenate([v.astype(np.int64, copy=False) for v in vals if v.size])
            if all_vals.size:
                unique, counts = np.unique(all_vals, return_counts=True)
                return int(unique[int(np.argmax(counts))])
        return int(fallback)

    def _cohere_plate_ids_xwrap(plate_arr):
        """Make final diagnostic plate IDs contiguous without changing terrain equations.

        Small disconnected fragments are reassigned to adjacent plates; large
        fragments are promoted to new microplate IDs.  The companion component
        class raster marks where cleanup happened so impossible non-contiguous
        plate diagnostics are easier to audit.
        """
        arr = np.asarray(plate_arr, dtype=np.int32).copy()
        comp_class = np.zeros_like(arr, dtype=np.int16)
        next_plate = int(arr.max(initial=0)) + 1
        world_cells = max(1, int(arr.size))
        # Work on a stable list of original IDs so newly promoted microplates are
        # already contiguous and do not need a second pass.
        original_ids = [int(v) for v in np.unique(arr) if int(v) != 0]
        for pid in original_ids:
            mask = arr == pid
            if not np.any(mask):
                continue
            labels_pid, n_pid = _label_xwrap(mask)
            if n_pid <= 1:
                continue
            counts = np.bincount(labels_pid.ravel())
            counts[0] = 0
            largest = int(np.argmax(counts))
            largest_size = int(counts[largest]) if largest < len(counts) else 0
            promote_threshold = max(96, int(world_cells * 0.0025), int(largest_size * 0.34))
            for lab in range(1, n_pid + 1):
                if lab == largest:
                    continue
                comp = labels_pid == lab
                size = int(counts[lab]) if lab < len(counts) else int(np.count_nonzero(comp))
                if size >= promote_threshold:
                    arr[comp] = next_plate
                    comp_class[comp] = 2  # large disconnected fragment promoted to microplate
                    next_plate += 1
                else:
                    replacement = _dominant_adjacent_plate(arr, comp, pid)
                    if replacement == pid:
                        replacement = next_plate
                        next_plate += 1
                        comp_class[comp] = 2
                    else:
                        comp_class[comp] = 1  # small disconnected fragment reassigned
                    arr[comp] = replacement
        return arr.astype(np.int32, copy=False), comp_class

    cont = _field01(base.plate_tectonic_continental_crust_x1000)
    conv = _field01(base.plate_tectonic_convergence_x1000)
    div = _field01(base.plate_tectonic_divergence_x1000)
    trans = _field01(base.plate_tectonic_transform_x1000)
    orogeny = _field01(base.plate_tectonic_orogeny_strength_x1000)
    arc = np.maximum(_field01(base.plate_tectonic_volcanic_arc_x1000), _field01(base.plate_tectonic_island_arc_x1000))
    rift = np.maximum(_field01(base.plate_tectonic_continental_rift_x1000), _field01(base.terrain_rift_field_x1000))
    plateau = _field01(base.plate_tectonic_plateau_uplift_x1000)
    passive = np.maximum(_field01(base.plate_tectonic_passive_margin_x1000), _field01(base.terrain_coastal_plain_x1000) * 0.45)
    active = _field01(base.plate_tectonic_active_margin_x1000)
    ridge = np.maximum(_field01(base.plate_tectonic_mid_ocean_ridge_x1000), _field01(base.terrain_mid_ocean_ridge_x1000))
    trench = np.maximum(_field01(base.plate_tectonic_trench_x1000), _field01(base.terrain_trench_x1000))
    fracture = np.maximum(_field01(base.plate_tectonic_fracture_zone_x1000), _field01(base.terrain_fracture_zone_x1000))
    seamount = np.maximum(_field01(base.plate_tectonic_seamount_x1000), _field01(base.terrain_seamount_x1000))
    ocean_age = _field01(base.plate_tectonic_ocean_crust_age_x1000, 0.45)
    # v3 uses smoothed structural circumstances for final elevation so diagnostic
    # plate-boundary polygons do not print through as hard ocean-floor triangles.
    ridge = _smooth(ridge, 3.0, 5.0)
    trench = _smooth(trench, 2.6, 4.2)
    fracture = _smooth(fracture, 2.0, 4.5)
    seamount = np.maximum(seamount, _smooth(seamount, 1.4, 2.2) * 0.75)
    ocean_age = _smooth(ocean_age, 2.4, 4.0)
    plate_id0 = _class_field(base.tectonic_plate_id)

    yy = np.linspace(-1.0, 1.0, height, dtype=np.float32)[:, None]
    abs_lat = np.abs(yy)
    polar_penalty = np.clip((abs_lat - 0.64) / 0.30, 0.0, 1.0)
    polar_penalty = polar_penalty * polar_penalty * (3.0 - 2.0 * polar_penalty)
    if bool(getattr(config, "suppress_polar_land", False)):
        cont = np.clip(cont * (1.0 - 0.58 * polar_penalty), 0.0, 1.0)
        plateau = np.clip(plateau * (1.0 - 0.45 * polar_penalty), 0.0, 1.0)

    # Continuous structural texture.  This adds natural variation without using
    # different logic branches for different cell classes.
    broad = np_rng.normal(0.0, 1.0, (height, width)).astype(np.float32)
    broad = _smooth(broad, max(5.0, height / 36.0), max(5.0, width / 36.0))
    broad = (_norm01(broad, 99.0) - 0.5) * 2.0
    mid = np_rng.normal(0.0, 1.0, (height, width)).astype(np.float32)
    mid = _smooth(mid, max(2.0, height / 120.0), max(2.0, width / 120.0))
    mid = (_norm01(mid, 99.0) - 0.5) * 2.0
    fine = np_rng.normal(0.0, 1.0, (height, width)).astype(np.float32)
    fine = _smooth(fine, 1.2, 1.8)
    fine = (_norm01(fine, 99.5) - 0.5) * 2.0
    texture = (0.48 * broad + 0.36 * mid + 0.16 * fine).astype(np.float32)

    # Unified continuous tectonic circumstances.  These are not terrain classes;
    # they are influences in one equation evaluated everywhere.
    compression = np.clip(conv * (0.30 + 0.70 * cont) + active * 0.42 + trans * conv * 0.28, 0.0, 1.0)
    extension = np.clip(div * (0.35 + 0.45 * cont) + rift * 0.75, 0.0, 1.0)
    volcanism = np.clip(arc * 0.92 + seamount * 0.42 + ridge * 0.28 + rift * 0.20, 0.0, 1.0)
    uplift = np.clip(orogeny * 0.82 + compression * 0.68 + plateau * 0.52 + volcanism * 0.32, 0.0, 1.0)
    subsidence = np.clip(extension * 0.56 + trench * 0.88 + ocean_age * (1.0 - cont) * 0.38, 0.0, 1.0)
    sediment = np.clip(passive * 0.55 + rift * 0.22 + compression * 0.18 + (1.0 - np.clip(np.abs(texture), 0.0, 1.0)) * 0.10, 0.0, 1.0)

    # Update 27D: clean cross-shaped Voronoi junctions should not print as
    # equally strong mountain-range intersections.  This is a continuous damping
    # field derived from the diagnostic plate geometry, not a separate terrain
    # class.  It weakens impossible four-way crossings while preserving normal
    # boundary belts and natural T/Y junctions.
    b0_e = plate_id0 != np.roll(plate_id0, 1, axis=1)
    b0_w = plate_id0 != np.roll(plate_id0, -1, axis=1)
    b0_n = np.zeros_like(b0_e); b0_s = np.zeros_like(b0_e)
    b0_n[1:, :] = plate_id0[1:, :] != plate_id0[:-1, :]
    b0_s[:-1, :] = plate_id0[:-1, :] != plate_id0[1:, :]
    initial_junction_count = b0_e.astype(np.int16) + b0_w.astype(np.int16) + b0_n.astype(np.int16) + b0_s.astype(np.int16)
    plus_junction_risk = _smooth((initial_junction_count >= 4).astype(np.float32), 1.1, 1.8)
    plus_junction_risk = np.clip(plus_junction_risk * (0.45 + 0.55 * (conv + div + trans)), 0.0, 1.0)
    compression = np.clip(compression * (1.0 - 0.20 * plus_junction_risk), 0.0, 1.0)
    uplift = np.clip(uplift * (1.0 - 0.28 * plus_junction_risk), 0.0, 1.0)
    orogeny = np.clip(orogeny * (1.0 - 0.18 * plus_junction_risk), 0.0, 1.0)

    crust_thickness = np.clip(
        0.18
        + 0.70 * cont
        + 0.18 * compression
        + 0.14 * plateau
        + 0.10 * volcanism
        + 0.06 * np.clip(texture, 0.0, 1.0)
        - 0.24 * trench * (1.0 - 0.45 * cont)
        - 0.17 * extension * (1.0 - 0.35 * cont),
        0.04,
        1.25,
    )
    if bool(getattr(config, "suppress_polar_land", False)):
        crust_thickness = np.clip(crust_thickness - 0.32 * polar_penalty, 0.02, 1.25)

    raw = (
        -4550.0
        + 6500.0 * crust_thickness
        + 1350.0 * uplift
        - 1600.0 * subsidence
        + 540.0 * sediment
        + 520.0 * texture * (0.18 + 0.82 * np.clip(cont + uplift + volcanism, 0.0, 1.0))
        + 620.0 * ridge * (1.0 - 0.55 * cont)
        - 1200.0 * trench * (1.0 - 0.25 * cont)
    ).astype(np.float32)

    # Match the requested hydrosphere by moving sea level, not by cutting cells
    # into special cases.  All cells remain governed by the same elevation field.
    ocean_target = clamp(float(getattr(hydrosphere, "ocean_fraction_target", 0.62) or 0.62), 0.40, 0.82)
    sea_level = float(np.percentile(raw, ocean_target * 100.0))
    elev = (raw - sea_level).astype(np.float32)

    # Preserve useful high-frequency detail from the previous stable mode without
    # preserving its broad halos: only the high-pass residual is blended in.
    base_smooth = _smooth(elev_base, max(6.0, height / 90.0), max(6.0, width / 90.0))
    residual = np.clip(elev_base - base_smooth, -380.0, 420.0)
    residual_weight = np.clip(0.15 + 0.28 * uplift + 0.18 * volcanism + 0.12 * trans, 0.08, 0.42)
    # Preserve land/coastal micro-detail from the stable mode, but do not import
    # broad old bathymetry halos or Voronoi-like ocean boundary textures.
    residual_weight = residual_weight * (0.92 * land_base.astype(np.float32) + 0.055 * (~land_base).astype(np.float32))
    elev = elev + residual * residual_weight

    # First land/water estimate from unified elevation.
    land = elev > 0.0
    water = ~land
    land_dist = _dist_xwrap(land)
    water_dist = _dist_xwrap(water)
    water_proximity = np.exp(-water_dist / 10.0).astype(np.float32)
    land_proximity = np.exp(-land_dist / 10.0).astype(np.float32)

    # A single continuous bathymetry equation is blended into water cells.  Update
    # 27 restores shelves by increasing continuous continental shelf support, not
    # by painting a fixed-width coastal halo.  Low-gradient continental/passive/
    # sediment-rich margins become broad; active/trench/volcanic-island margins
    # remain steep unless continental support is genuinely present.
    support_near = _smooth(cont * land.astype(np.float32), 3.0, 5.0)
    arc_near = _smooth(volcanism * land.astype(np.float32), 2.0, 3.0)
    early_slope_y, early_slope_x = np.gradient(elev)
    early_slope_n = _norm01(np.hypot(early_slope_x, early_slope_y).astype(np.float32), 98.5)
    low_margin_slope = np.clip(1.0 - _smooth(early_slope_n, 2.0, 3.2), 0.0, 1.0)
    margin_sediment_support = np.clip(sediment * (0.42 + 0.58 * passive + 0.26 * rift) * (1.0 - 0.55 * active), 0.0, 1.0)

    # Update 27C: build a wider submerged continental apron from true
    # continent-scale land, not from all final coastline cells.  This is what
    # lets broad continental shelves appear while keeping volcanic island chains
    # steep.  The apron is still a continuous field and only feeds the same
    # bathymetry equation used everywhere.
    initial_land_labels, initial_land_count = _label_xwrap(land)
    continent_scale_land = np.zeros_like(land, dtype=bool)
    if initial_land_count:
        counts_init = np.bincount(initial_land_labels.ravel())
        land_cells_init = max(1, int(np.count_nonzero(land)))
        world_cells_init = max(1, int(width * height))
        continent_min_cells = max(96, int(min(land_cells_init * 0.018, world_cells_init * 0.009)))
        for lab in range(1, initial_land_count + 1):
            comp = initial_land_labels == lab
            size = int(counts_init[lab]) if lab < len(counts_init) else int(np.count_nonzero(comp))
            if size < continent_min_cells:
                continue
            comp_cont = float(np.mean(cont[comp])) if size else 0.0
            comp_passive = float(np.mean(passive[comp])) if size else 0.0
            comp_volcanic = float(np.mean(volcanism[comp])) if size else 0.0
            if comp_cont > 0.30 or (comp_cont + comp_passive > 0.52 and comp_volcanic < 0.42):
                continent_scale_land |= comp
    if not np.any(continent_scale_land):
        continent_scale_land = land & (cont > 0.42)
    if not np.any(continent_scale_land):
        continent_scale_land = land

    dist_to_continental_land = _dist_xwrap(continent_scale_land)
    map_scale_px = max(0.55, float(width) / 2048.0)
    continental_apron_width = np.clip(
        map_scale_px
        * (
            18.0
            + 28.0 * v3_shelf_width_factor
            + 26.0 * max(0.0, v3_continental_shelf_strength - 1.0)
            + 22.0 * np.clip(passive + margin_sediment_support, 0.0, 1.0)
            + 10.0 * rift
            - 14.0 * active
            - 18.0 * trench
        ),
        2.5 * map_scale_px,
        190.0 * map_scale_px,
    )
    continental_apron = np.exp(-((dist_to_continental_land / np.maximum(1.0, continental_apron_width)) ** 1.16)).astype(np.float32)
    continental_apron *= np.clip(
        0.18 + 0.68 * _smooth(cont * continent_scale_land.astype(np.float32), 4.0, 7.0) + 0.30 * passive + 0.18 * margin_sediment_support + 0.10 * rift,
        0.0,
        1.0,
    )
    continental_apron *= np.clip(1.0 - 0.70 * trench - 0.50 * active - 0.42 * arc_near * (1.0 - passive), 0.0, 1.0)

    # Continental shelf support should come from submerged continental affinity,
    # not simply distance to the present coastline.  This lets drowned continental
    # margins become broad shallow shelves while unsupported volcanic islands stay
    # steep and deep.  The same continuous field is evaluated everywhere.
    submerged_continental_affinity = np.clip(
        np.maximum.reduce([cont, support_near * 1.12, continental_apron * 1.18])
        + 0.28 * passive
        + 0.18 * rift
        + 0.18 * margin_sediment_support
        - 0.20 * ridge * (1.0 - np.clip(cont + continental_apron, 0.0, 1.0))
        - 0.38 * trench
        - 0.14 * ocean_age * (1.0 - np.clip(cont + support_near + continental_apron, 0.0, 1.0)),
        0.0,
        1.0,
    )
    continental_margin_carrier = np.clip(
        0.54 * submerged_continental_affinity
        + 0.30 * support_near
        + 0.40 * continental_apron
        + 0.24 * passive
        + 0.12 * rift
        + 0.14 * margin_sediment_support,
        0.0,
        1.0,
    )
    volcanic_island_suppression = np.clip(
        1.0 - 0.76 * arc_near * (1.0 - np.clip(continental_margin_carrier * 1.55 + passive * 0.45, 0.0, 1.0)),
        0.12,
        1.0,
    )
    shelf_strength_scale = 0.82 + 0.54 * v3_continental_shelf_strength
    shelf_width_scale = 0.82 + 0.70 * v3_shelf_width_factor + 0.26 * max(0.0, v3_continental_shelf_strength - 1.0)
    shelf_support = np.clip(
        shelf_strength_scale
        * (
            0.44 * continental_margin_carrier
            + 0.34 * submerged_continental_affinity
            + 0.22 * continental_apron
            + 0.26 * passive
            + 0.26 * margin_sediment_support
            + 0.17 * rift
            + 0.12 * low_margin_slope * np.maximum(land_proximity, continental_apron)
            - 0.34 * active
            - 0.42 * trench
            - 0.22 * arc_near * (1.0 - continental_margin_carrier)
        ),
        0.0,
        1.0,
    ) * volcanic_island_suppression
    shelf_breadth = np.clip(
        shelf_width_scale
        * (
            5.0 * map_scale_px
            + 58.0 * shelf_support * map_scale_px
            + 36.0 * continental_margin_carrier * map_scale_px
            + 32.0 * continental_apron * map_scale_px
            + 22.0 * passive * np.clip(submerged_continental_affinity + sediment, 0.0, 1.0) * map_scale_px
            + 12.0 * rift * np.clip(submerged_continental_affinity + 0.45 * passive, 0.0, 1.0) * map_scale_px
            - 12.0 * active
            - 13.0 * trench
            - 7.0 * arc_near * (1.0 - continental_margin_carrier)
        ),
        1.0 * map_scale_px,
        190.0 * map_scale_px,
    )
    shelf_t = land_dist / np.maximum(1.0, shelf_breadth)
    shelf_decay = np.exp(-(shelf_t ** 1.10)).astype(np.float32)
    shelf_gate = np.clip(shelf_decay * (0.18 + 0.82 * shelf_support), 0.0, 1.0)
    slope_rise = np.clip(np.exp(-shelf_t / 2.95) - shelf_decay * 0.58, 0.0, 1.0).astype(np.float32)
    sediment_apron_support = np.clip(
        sediment * land_proximity * (0.14 + 0.78 * submerged_continental_affinity + 0.34 * passive + 0.18 * rift)
        * (1.0 - 0.60 * arc_near * (1.0 - continental_margin_carrier))
        * (1.0 - 0.48 * active)
        * (1.0 - 0.42 * trench),
        0.0,
        1.0,
    )
    abyssal_target = (
        -4850.0
        + 620.0 * ridge
        + 420.0 * seamount
        + 460.0 * sediment_apron_support
        - 1780.0 * trench
        - 840.0 * ocean_age * (1.0 - shelf_gate * 0.55)
    ).astype(np.float32)
    continental_shelf_profile = (
        -38.0
        - 240.0 * np.minimum(shelf_t, 1.0) ** 1.72
        - 2020.0 * np.clip(shelf_t - 1.0, 0.0, 2.6) ** 1.16
        + 230.0 * sediment_apron_support
        + 165.0 * passive
        + 120.0 * continental_apron
        - 640.0 * active
        - 1080.0 * trench
        - 460.0 * arc_near * (1.0 - continental_margin_carrier)
    ).astype(np.float32)
    shelf_profile_blend = np.clip(shelf_gate * (0.24 + 0.76 * np.maximum(continental_margin_carrier, continental_apron * 0.92)), 0.0, 1.0)
    bathy_target = abyssal_target * (1.0 - shelf_profile_blend) + np.maximum(abyssal_target, continental_shelf_profile) * shelf_profile_blend
    bathy_target += (680.0 * slope_rise * np.clip(shelf_support + passive * 0.35, 0.0, 1.0)).astype(np.float32)
    true_shelf_gate = np.clip(shelf_profile_blend * shelf_support * np.maximum(continental_margin_carrier, continental_apron), 0.0, 1.0)
    bathy_blend = np.clip((water.astype(np.float32)) * (0.28 + 0.52 * land_proximity + 0.18 * ocean_age + 0.42 * true_shelf_gate), 0.0, 0.97)
    elev = _var_smooth(elev, bathy_blend, bathy_target)
    land = elev > 0.0
    water = ~land

    # Unified maturity-driven erosion/deposition.  Every cell uses the same
    # variable-strength smoothing equation; outcomes differ only because the
    # circumstance fields differ.
    slope_y, slope_x = np.gradient(elev)
    slope = np.hypot(slope_x, slope_y).astype(np.float32)
    slope_n = _norm01(slope, 98.5)
    young_activity = np.clip(uplift * 0.75 + volcanism * 0.55 + ridge * 0.25 + trench * 0.18, 0.0, 1.0)
    old_orogen_maturity = np.clip(orogeny * cont * (1.0 - young_activity) * (0.45 + 0.38 * passive + 0.22 * sediment), 0.0, 1.0)
    old_maturity = np.clip(
        ocean_age * water.astype(np.float32) * (1.0 - ridge) * (1.0 - trench)
        + passive * 0.50
        + sediment * 0.34
        + (1.0 - young_activity) * cont * 0.22
        + old_orogen_maturity * 0.24,
        0.0,
        1.0,
    )
    erosion_multiplier = float(v3_erosion_deposition_multiplier)
    erosion_safety_floor = 0.10 if erosion_multiplier <= 0.0 else 0.0
    # Update 27D: high user erosion should emphasize old-mountain lowering,
    # basin fill, and shelf sedimentation; it should not globally blur coastlines
    # or erase islands.  Keep the master control, but soften its effect on direct
    # smoothing and protect final coast/micro-island detail more aggressively.
    erosion_scale = np.clip(0.20 + 0.70 * min(erosion_multiplier, 2.0) + 0.22 * max(0.0, erosion_multiplier - 2.0), 0.0, 2.10)
    erosion_strength = np.clip(
        (0.08 + 0.54 * old_maturity + 0.18 * slope_n + 0.18 * old_orogen_maturity - 0.48 * young_activity)
        * erosion_scale
        + erosion_safety_floor * old_maturity,
        0.0,
        0.78,
    )
    # Protect coastline shape but still smooth elevation inside impossible rims.
    coast_detail_preserve = np.clip((np.abs(_smooth(land.astype(np.float32), 1.1, 1.6) - land.astype(np.float32)) * 3.2), 0.0, 1.0)
    island_detail_preserve = np.clip(volcanism * (1.0 - cont) * (1.0 - passive) + seamount * 0.45, 0.0, 1.0)
    erosion_strength = np.clip(erosion_strength * (1.0 - 0.58 * coast_detail_preserve) * (1.0 - 0.38 * island_detail_preserve), 0.0, 0.78)
    smooth1 = _smooth(elev, 1.4, 2.0)
    smooth2 = _smooth(elev, 3.0, 4.6)
    smooth3 = _smooth(elev, 5.0, 7.8)
    maturity_extra = np.clip((erosion_multiplier - 1.0) * 0.22, 0.0, 0.36)
    smooth_mix = (
        smooth1 * (1.0 - old_maturity)
        + smooth2 * old_maturity * (1.0 - maturity_extra)
        + smooth3 * old_maturity * maturity_extra
    ).astype(np.float32)
    elev = _var_smooth(elev, erosion_strength, smooth_mix)

    # Passive-margin and old-basin deposition fills low-energy shelves, rises,
    # foreland-like basins, and broad coastal lowlands.  The term is continuous
    # and is suppressed around active, trench, and unsupported volcanic-island
    # coasts so sediment does not pile up in open ocean around peninsulas/islands.
    post_slope_y, post_slope_x = np.gradient(elev)
    post_slope_n = _norm01(np.hypot(post_slope_x, post_slope_y).astype(np.float32), 98.5)
    deposition_scale = np.clip(0.18 + 0.86 * erosion_multiplier, 0.0, 3.15)
    low_energy_deposition = np.clip(
        sediment
        * (0.30 + 0.48 * passive + 0.18 * rift + 0.16 * old_orogen_maturity)
        * (1.0 - 0.62 * post_slope_n)
        * (0.18 + 0.82 * np.clip(submerged_continental_affinity + passive + 0.45 * rift, 0.0, 1.0))
        * (1.0 - 0.60 * active)
        * (1.0 - 0.48 * trench)
        * (1.0 - 0.58 * arc_near * (1.0 - continental_margin_carrier)),
        0.0,
        1.0,
    ) * deposition_scale
    low_energy_deposition = np.clip(low_energy_deposition, 0.0, 1.0)
    shallow_water = water.astype(np.float32) * np.clip((2200.0 + elev) / 2200.0, 0.0, 1.0)
    elev += (
        (48.0 + 20.0 * erosion_multiplier) * low_energy_deposition * land.astype(np.float32)
        + (135.0 + 65.0 * erosion_multiplier) * low_energy_deposition * shallow_water
    ).astype(np.float32)

    # Basin/lake rims are not handled by separate categories.  A continuous
    # steep-rim reducer looks at water proximity, slope and youth: old/low-energy
    # rims relax more; active rift/mountain rims remain steeper.
    land = elev > 0.0
    water = ~land
    water_dist = _dist_xwrap(water)
    rim_influence = np.exp(-water_dist / 8.0).astype(np.float32) * land.astype(np.float32)
    rim_relax = np.clip(rim_influence * slope_n * (0.55 + 0.45 * old_maturity) * (1.0 - 0.65 * young_activity), 0.0, 0.65)
    elev = _var_smooth(elev, rim_relax, _smooth(elev, 3.2, 5.0))

    # A continuous ocean-floor maturity pass reduces residual Voronoi/plate-grid
    # ripples in old deep ocean while keeping young ridges, trenches and seamount
    # peaks readable.  This is still the same smoothing equation; only the local
    # maturity/preservation circumstances change the weight.
    land_tmp = elev > 0.0
    water_tmp = ~land_tmp
    ripple_artifact_risk = np.clip(
        water_tmp.astype(np.float32)
        * (0.22 + 0.48 * ocean_age + 0.22 * np.abs(texture))
        * (0.42 + 0.58 * (conv + div + trans))
        * (1.0 - 0.48 * ridge)
        * (1.0 - 0.42 * trench)
        * (1.0 - 0.36 * seamount)
        * (1.0 - 0.34 * shelf_support),
        0.0,
        1.0,
    )
    deep_ocean_maturity = np.clip(water_tmp.astype(np.float32) * (0.30 + 0.52 * ocean_age + 0.26 * ripple_artifact_risk) * (1.0 - 0.55 * ridge) * (1.0 - 0.45 * trench) * (1.0 - 0.35 * seamount), 0.0, 0.84)
    elev = _var_smooth(elev, deep_ocean_maturity, _smooth(elev, 4.2, 7.6))

    # Update 27C final shelf reinforcement: after the ocean-floor maturity pass,
    # broad supported continental shelves are raised again toward shallow water.
    # The cap below sea level preserves coastline detail and avoids simply
    # converting shelves into extra land.
    land_tmp = elev > 0.0
    water_tmp = ~land_tmp
    shelf_shallow_target = np.minimum(-4.0, np.maximum(continental_shelf_profile, -180.0 - 470.0 * np.clip(shelf_t, 0.0, 1.0) ** 1.65))
    shelf_reinforcement = np.clip(
        water_tmp.astype(np.float32)
        * np.maximum(shelf_profile_blend, true_shelf_gate)
        * (0.32 + 0.62 * np.maximum(continental_margin_carrier, continental_apron))
        * (1.0 - 0.58 * trench)
        * (1.0 - 0.46 * active)
        * (1.0 - 0.44 * arc_near * (1.0 - continental_margin_carrier)),
        0.0,
        0.88,
    )
    elev = _var_smooth(elev, shelf_reinforcement, shelf_shallow_target.astype(np.float32, copy=False))

    lake_depth_limited_cells = 0
    lake_depth_limit_min_m = 0.0
    deep_lake_supported_cells = 0
    lake_tiny_filled_cells = 0
    coastal_lagoon_filled_cells = 0
    lake_depth_limit_field = np.zeros((height, width), dtype=np.float32)

    def _wrapped_span(cols):
        if cols.size == 0:
            return 0
        xs_sorted = np.sort(cols.astype(np.int32, copy=False))
        gaps = np.diff(np.r_[xs_sorted, xs_sorted[0] + width])
        return int(width - gaps.max()) if gaps.size else int(xs_sorted[-1] - xs_sorted[0] + 1)

    def _component_center_factor(comp, ys, xs, texture_field):
        """0 at edge, 1 near basin center, cropped for speed."""
        if ys.size == 0 or xs.size == 0:
            return np.zeros(0, dtype=np.float32)
        # For seam-crossing components, the ordinary bbox can cover the world;
        # use a lightweight texture-only fallback rather than an expensive full
        # distance transform on every lagoon/lake.
        if _wrapped_span(xs) < (int(xs.max() - xs.min() + 1) if xs.size else 0):
            return np.clip(0.35 + 0.45 * texture_field[comp], 0.0, 1.0).astype(np.float32, copy=False)
        y0 = max(0, int(ys.min()) - 2); y1 = min(height, int(ys.max()) + 3)
        x0 = max(0, int(xs.min()) - 2); x1 = min(width, int(xs.max()) + 3)
        crop = comp[y0:y1, x0:x1]
        if not crop.size:
            return np.zeros(int(np.count_nonzero(comp)), dtype=np.float32)
        dist = ndimage.distance_transform_edt(crop).astype(np.float32, copy=False)
        vals = dist[(ys - y0, xs - x0)]
        denom = float(np.percentile(vals, 92.0)) if vals.size else 1.0
        return np.clip(vals / max(1.0, denom), 0.0, 1.0).astype(np.float32, copy=False)

    def _condition_enclosed_water_depths(current_elev):
        nonlocal lake_depth_limited_cells, lake_depth_limit_min_m, deep_lake_supported_cells, lake_tiny_filled_cells, coastal_lagoon_filled_cells, lake_depth_limit_field
        current_elev = np.asarray(current_elev, dtype=np.float32).copy()
        current_land = current_elev > 0.0
        current_water = ~current_land
        labels_local, n_local = _label_xwrap(current_water)
        if not n_local:
            return current_elev
        counts_local = np.bincount(labels_local.ravel())
        counts_local[0] = 0
        ocean_label_local = int(np.argmax(counts_local))
        ocean_mask_local = labels_local == ocean_label_local
        ocean_dist_local = _dist_xwrap(ocean_mask_local)
        tiny_local = np.zeros_like(current_water, dtype=bool)
        world_cells = max(1, width * height)
        basin_driver = np.clip(0.44 * extension + 0.34 * rift + 0.28 * subsidence + 0.20 * (1.0 - sediment) + 0.10 * np.abs(texture), 0.0, 1.0)
        local_texture = np.clip(_smooth(np.abs(texture), 1.2, 1.8), 0.0, 1.0)
        for lab in range(1, n_local + 1):
            if lab == ocean_label_local:
                continue
            comp = labels_local == lab
            size = int(counts_local[lab]) if lab < len(counts_local) else int(np.count_nonzero(comp))
            if size <= max(4, int(world_cells * 0.000015)):
                tiny_local |= comp
                continue
            ys, xs = np.where(comp)
            y_span = int(ys.max() - ys.min() + 1) if ys.size else 1
            x_span_wrapped = max(1, _wrapped_span(xs))
            long_axis = max(y_span, x_span_wrapped)
            short_axis = max(1, min(y_span, x_span_wrapped))
            elong = float(long_axis / short_axis)
            size_factor = float(np.clip(np.sqrt(size / max(1.0, world_cells * 0.0045)), 0.0, 1.0))
            basin_support_mean = float(np.mean(basin_driver[comp])) if size else 0.0
            sediment_fill_mean = float(np.mean(sediment[comp])) if size else 0.0
            mean_ocean_dist = float(np.mean(ocean_dist_local[comp])) if size else 999.0

            # Oversized one/few-cell-wide coast-parallel lakes are usually an
            # artifact of shelf/lake interaction at this raster scale.  Fill
            # unsupported ones into coastal lowland instead of leaving them as
            # endorheic sinks that distort drainage basins.  True rift/inland
            # seas are protected by basin_support_mean and distance from ocean.
            lagoon_like = (
                elong > 8.0
                and short_axis <= max(3.0, 5.5 * map_scale_px)
                and mean_ocean_dist <= max(4.0, 9.0 * map_scale_px)
                and basin_support_mean < 0.50
                and float(np.mean(rift[comp])) < 0.36
            )
            if lagoon_like:
                lowland = 1.5 + 14.0 * sediment[comp] + 5.0 * local_texture[comp]
                current_elev[comp] = np.maximum(current_elev[comp], lowland.astype(np.float32, copy=False))
                lake_depth_limit_field[comp] = 0.0
                coastal_lagoon_filled_cells += size
                continue

            strong_rift_basin = max(0.0, basin_support_mean - 0.36) / 0.64
            fill_modifier = np.clip(1.0 - 0.20 * max(0.0, erosion_multiplier - 1.0) - 0.30 * sediment_fill_mean, 0.48, 1.10)
            allowed_depth = (
                50.0
                + 260.0 * size_factor
                + 420.0 * basin_support_mean * max(0.16, size_factor)
                + 2050.0 * (strong_rift_basin ** 1.70) * max(0.22, size_factor)
                + 260.0 * max(0.0, 0.45 - sediment_fill_mean)
            ) * fill_modifier
            # Large inland seas get broader depth allowance, but only rift/
            # subsidence-supported ones can retain very deep troughs.
            if size_factor > 0.72 and basin_support_mean > 0.46:
                allowed_depth += 620.0 * (size_factor - 0.72) / 0.28
            if size_factor < 0.22 and basin_support_mean < 0.54:
                allowed_depth = min(allowed_depth, 240.0 + 390.0 * basin_support_mean)
            elif size_factor < 0.42 and basin_support_mean < 0.50:
                allowed_depth = min(allowed_depth, 500.0 + 520.0 * basin_support_mean)
            allowed_depth = float(np.clip(allowed_depth, 45.0, 3800.0))
            if allowed_depth > 1200.0:
                deep_lake_supported_cells += size

            center_factor = _component_center_factor(comp, ys, xs, local_texture)
            support_vals = basin_driver[comp].astype(np.float32, copy=False)
            # Edges are shallow, centers may retain rift/subsidence troughs.
            # This turns the clamp into a spatially varying limit instead of a
            # flat bathymetry replacement.
            allowed_field = allowed_depth * (0.26 + 0.74 * center_factor)
            allowed_field *= (0.82 + 0.24 * local_texture[comp] + 0.20 * support_vals)
            allowed_field = np.clip(allowed_field, 28.0, allowed_depth * 1.10).astype(np.float32, copy=False)
            before = current_elev[comp].copy()
            current_elev[comp] = np.maximum(current_elev[comp], -allowed_field)
            lake_depth_limit_field[comp] = np.maximum(lake_depth_limit_field[comp], np.clip(allowed_field / 4000.0, 0.0, 1.0))
            changed = before < current_elev[comp]
            if np.any(changed):
                lake_depth_limited_cells += int(np.count_nonzero(changed))
                lake_depth_limit_min_m = min(lake_depth_limit_min_m, -float(np.max(allowed_field)))
            # Keep floors from snapping to exactly -1 m; shallow lakes still get
            # varied, sediment-filled bathymetry and deeper centers.
            basin_floor = -6.0 - 90.0 * sediment[comp] - 155.0 * support_vals * center_factor - 45.0 * local_texture[comp]
            current_elev[comp] = np.minimum(current_elev[comp], basin_floor.astype(np.float32, copy=False))
        if np.any(tiny_local):
            fill = _smooth(current_elev, 2.0, 3.0)
            current_elev[tiny_local] = np.maximum(current_elev[tiny_local], np.minimum(6.0, fill[tiny_local] + 8.0))
            lake_depth_limit_field[tiny_local] = 0.0
            lake_tiny_filled_cells += int(np.count_nonzero(tiny_local))
        return current_elev

    # Final connected-water pass: enclosed water keeps varied depths, but lake
    # floors are limited by continuous basin support rather than allowed to keep
    # oceanic depths.
    elev = _condition_enclosed_water_depths(elev)

    land = elev > 0.0
    water = ~land

    # Post-deposition arc snake suppression as a continuous topology guard:
    # weak, low, long arc connectors are lowered into seamount ridges while peak
    # centers are preserved by volcanism/uplift.  Update 27 makes this stricter
    # for unsupported volcanic/island chains but leaves continental and
    # microcontinental fragments alone.
    labels_land, n_land = _label_xwrap(land)
    snake_sunk = 0
    if n_land:
        neighbor_kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.int16)
        land_neighbors = ndimage.convolve(land.astype(np.int16), neighbor_kernel, mode="constant", cval=0)
        # Restore east-west wrap for the two seam-adjacent columns without
        # wrapping north/south across the poles.
        if width > 1:
            land_neighbors[:, 0] += land[:, width - 1].astype(np.int16)
            land_neighbors[:, width - 1] += land[:, 0].astype(np.int16)
        for lab in range(1, n_land + 1):
            comp = labels_land == lab
            size = int(np.count_nonzero(comp))
            if size < max(20, int(width * height * 0.000035)):
                continue
            ys, xs = np.where(comp)
            y_span = int(ys.max() - ys.min() + 1)
            x_span = int(xs.max() - xs.min() + 1)
            # Account roughly for seam-crossing components by checking shorter wrap span.
            xs_sorted = np.sort(xs)
            gaps = np.diff(np.r_[xs_sorted, xs_sorted[0] + width])
            x_wrap_span = width - int(gaps.max()) if len(gaps) else x_span
            long_axis = max(x_wrap_span, y_span)
            short_axis = max(1, min(x_wrap_span, y_span))
            elong = long_axis / short_axis
            arc_mean = float(np.mean(volcanism[comp]))
            cont_mean = float(np.mean(cont[comp]))
            shelf_mean = float(np.mean(shelf_support[comp]))
            if elong > 7.2 and arc_mean > 0.18 and cont_mean < 0.50 and shelf_mean < 0.42:
                volc_cut = float(np.percentile(volcanism[comp], 76))
                elev_cut = float(np.percentile(elev[comp], 66))
                thin = comp & (land_neighbors <= 4)
                weak_bridge = comp & (cont < 0.40) & (volcanism < volc_cut) & (elev < elev_cut)
                connector = (thin | weak_bridge) & comp & (uplift < np.percentile(uplift[comp], 82))
                if np.any(connector):
                    sink = -42.0 - 260.0 * volcanism - 110.0 * np.clip(1.0 - cont, 0.0, 1.0)
                    elev[connector] = np.minimum(elev[connector], sink[connector])
                    snake_sunk += int(np.count_nonzero(connector))
    land = elev > 0.0
    water = ~land

    # Rebalance ocean fraction gently after smoothing without hard masks.
    ocean_now = float(np.mean(water))
    if abs(ocean_now - ocean_target) > 0.015:
        shift = float(np.percentile(elev, ocean_target * 100.0))
        elev = elev - shift * 0.75
        land = elev > 0.0
        water = ~land

    # The sea-level rebalance can create or deepen enclosed basins, so run the
    # same continuous lake-floor limiter one last time before integer export.
    elev = _condition_enclosed_water_depths(elev)
    land = elev > 0.0
    water = ~land

    # Update 27D: apply the explicit map-centering rule as a pure east/west
    # translation before final diagnostics are derived.  The shift preserves all
    # shapes and rolls the continuous fields and diagnostic plate IDs together:
    # first prefer an all-ocean longitude seam, otherwise center the land-rich
    # hemisphere.
    final_longitude_shift_cols = int(_best_longitude_shift_to_ocean_gap(land))
    if final_longitude_shift_cols:
        def _roll_field(arr):
            return np.roll(arr, final_longitude_shift_cols, axis=1)
        elev = _roll_field(elev)
        elev_base = _roll_field(elev_base)
        land_base = _roll_field(land_base)
        cont = _roll_field(cont); conv = _roll_field(conv); div = _roll_field(div); trans = _roll_field(trans)
        orogeny = _roll_field(orogeny); arc = _roll_field(arc); rift = _roll_field(rift); plateau = _roll_field(plateau)
        passive = _roll_field(passive); active = _roll_field(active); ridge = _roll_field(ridge); trench = _roll_field(trench)
        fracture = _roll_field(fracture); seamount = _roll_field(seamount); ocean_age = _roll_field(ocean_age); plate_id0 = _roll_field(plate_id0)
        compression = _roll_field(compression); extension = _roll_field(extension); volcanism = _roll_field(volcanism)
        uplift = _roll_field(uplift); subsidence = _roll_field(subsidence); sediment = _roll_field(sediment)
        crust_thickness = _roll_field(crust_thickness); texture = _roll_field(texture); old_maturity = _roll_field(old_maturity)
        erosion_strength = _roll_field(erosion_strength); low_energy_deposition = _roll_field(low_energy_deposition)
        continent_scale_land = _roll_field(continent_scale_land); continental_apron = _roll_field(continental_apron)
        submerged_continental_affinity = _roll_field(submerged_continental_affinity); continental_margin_carrier = _roll_field(continental_margin_carrier)
        shelf_support = _roll_field(shelf_support); shelf_breadth = _roll_field(shelf_breadth); shelf_t = _roll_field(shelf_t)
        shelf_profile_blend = _roll_field(shelf_profile_blend); true_shelf_gate = _roll_field(true_shelf_gate)
        shelf_shallow_target = _roll_field(shelf_shallow_target); continental_shelf_profile = _roll_field(continental_shelf_profile)
        ripple_artifact_risk = _roll_field(ripple_artifact_risk); lake_depth_limit_field = _roll_field(lake_depth_limit_field)
        plus_junction_risk = _roll_field(plus_junction_risk)
        land = elev > 0.0
        water = ~land

    elev_int = np.rint(np.clip(elev, -11000.0, 9200.0)).astype(np.int32)
    min_elev = int(elev_int.min())
    max_elev = int(elev_int.max())
    ocean_fraction = float(np.mean(~land))
    mean_land = float(elev_int[land].mean()) if np.any(land) else 0.0
    mean_ocean_depth = float((-elev_int[~land]).mean()) if np.any(~land) else 0.0

    # Deformable-looking plate diagnostics: exchange thin boundary/sliver zones
    # from the stable macro IDs without using the result to choose terrain rules.
    rough_plate = plate_id0.copy()
    boundary = np.zeros_like(rough_plate, dtype=bool)
    boundary |= rough_plate != np.roll(rough_plate, 1, axis=1)
    boundary |= rough_plate != np.roll(rough_plate, -1, axis=1)
    boundary[:-1, :] |= rough_plate[:-1, :] != rough_plate[1:, :]
    boundary[1:, :] |= rough_plate[1:, :] != rough_plate[:-1, :]
    exchange_noise = _smooth(np_rng.normal(0.0, 1.0, (height, width)).astype(np.float32), 1.6, 2.5)
    exch = boundary & (np.abs(exchange_noise) > 0.42)
    rough_plate[exch & (exchange_noise > 0)] = np.roll(rough_plate, 1, axis=1)[exch & (exchange_noise > 0)]
    rough_plate[exch & (exchange_noise < 0)] = np.roll(rough_plate, -1, axis=1)[exch & (exchange_noise < 0)]

    rough_plate, final_plate_component_class = _cohere_plate_ids_xwrap(rough_plate)

    b_e = rough_plate != np.roll(rough_plate, 1, axis=1)
    b_w = rough_plate != np.roll(rough_plate, -1, axis=1)
    b_n = np.zeros_like(b_e); b_s = np.zeros_like(b_e)
    b_n[1:, :] = rough_plate[1:, :] != rough_plate[:-1, :]
    b_s[:-1, :] = rough_plate[:-1, :] != rough_plate[1:, :]
    junction_count = b_e.astype(np.int16) + b_w.astype(np.int16) + b_n.astype(np.int16) + b_s.astype(np.int16)
    topology = np.zeros((height, width), dtype=np.int16)
    topology[junction_count >= 1] = 1
    topology[junction_count == 3] = 2
    topology[junction_count >= 4] = 3
    plus_to_t = (topology == 3) & (np.abs(exchange_noise) > 0.25)
    topology[plus_to_t] = 2

    # Update 26: v3-only diagnostics are recomputed from the final terrain mask
    # and continuous v3 circumstance fields.  These rasters are explanatory
    # diagnostics, not terrain-rule switches.
    land_float = land.astype(np.float32)
    water_float = water.astype(np.float32)
    final_land_dist = _dist_xwrap(land)
    final_water_dist = _dist_xwrap(water)
    final_coast_band = (final_land_dist <= 6.0) | (final_water_dist <= 6.0)
    final_coast_land = land & (final_water_dist <= 1.5)
    final_coast_water = water & (final_land_dist <= 2.5)
    final_slope_y, final_slope_x = np.gradient(elev)
    final_slope = np.hypot(final_slope_x, final_slope_y).astype(np.float32)
    final_slope_n = _norm01(final_slope, 98.5)
    support_near_final = _smooth(cont * land_float, 3.0, 5.0)
    final_dist_to_continental_land = _dist_xwrap(continent_scale_land)
    final_shelf_decay = np.exp(-(np.minimum(final_land_dist, final_dist_to_continental_land * 1.08) / np.maximum(1.0, shelf_breadth)) ** 1.05).astype(np.float32)
    final_low_slope = np.clip(1.0 - _smooth(final_slope_n, 2.0, 3.2), 0.0, 1.0)
    final_submerged_continental_affinity = np.clip(
        np.maximum.reduce([submerged_continental_affinity, support_near_final * 1.05, continental_apron * 1.10])
        + 0.14 * passive
        + 0.10 * sediment
        - 0.26 * trench
        - 0.16 * ridge * (1.0 - cont),
        0.0,
        1.0,
    )
    final_margin_carrier = np.clip(
        0.56 * final_submerged_continental_affinity
        + 0.26 * support_near_final
        + 0.36 * continental_apron
        + 0.24 * passive
        + 0.12 * rift
        + 0.12 * sediment,
        0.0,
        1.0,
    )
    final_shelf_support = np.clip(
        shelf_strength_scale
        * (
            0.44 * final_margin_carrier
            + 0.34 * final_submerged_continental_affinity
            + 0.20 * continental_apron
            + 0.26 * passive
            + 0.22 * sediment * (0.45 + 0.55 * passive)
            + 0.15 * rift
            + 0.13 * final_low_slope * np.exp(-final_land_dist / 18.0)
            - 0.34 * active
            - 0.42 * trench
            - 0.22 * volcanism * (1.0 - final_margin_carrier)
        ),
        0.0,
        1.0,
    )
    final_shelf_potential = np.clip(
        final_shelf_decay
        * (0.06 + 0.94 * final_shelf_support)
        * (0.66 + 0.34 * np.maximum(final_margin_carrier, continental_apron))
        * (1.0 - 0.50 * active)
        * (1.0 - 0.48 * trench)
        * (1.0 - 0.48 * volcanism * (1.0 - np.clip(final_margin_carrier * 1.55 + passive * 0.4, 0.0, 1.0))),
        0.0,
        1.0,
    )
    final_shelf_depth_target = np.clip((-shelf_shallow_target - 5.0) / 2200.0, 0.0, 1.0)
    final_shelf_zone = np.zeros((height, width), dtype=np.int16)
    final_shelf_zone[water & (final_shelf_potential > 0.40) & (elev > -320.0)] = 1  # shallow continental shelf sea
    final_shelf_zone[water & (final_shelf_potential > 0.26) & (elev <= -320.0) & (elev > -1450.0)] = 2  # shelf edge / upper slope
    final_shelf_zone[water & (final_shelf_potential > 0.14) & (elev <= -1450.0) & (elev > -2600.0)] = 3  # continental rise
    final_shelf_zone[water & (final_shelf_potential <= 0.14)] = 4  # abyssal/open ocean
    final_shelf_zone[water & (trench > 0.26)] = 5  # active trench suppression
    final_shelf_zone[land] = 6

    final_coastal_plain = np.clip(
        (0.42 * passive + 0.28 * sediment + 0.24 * support_near_final + 0.18 * (1.0 - final_slope_n)
         - 0.26 * active - 0.20 * volcanism - 0.16 * trench)
        * final_coast_band.astype(np.float32),
        0.0,
        1.0,
    )
    final_coast_ruggedness = np.clip(
        (0.38 * active + 0.30 * compression + 0.24 * volcanism + 0.22 * trans + 0.24 * final_slope_n
         - 0.24 * passive - 0.16 * sediment)
        * final_coast_band.astype(np.float32),
        0.0,
        1.0,
    )
    final_coast_style = np.zeros((height, width), dtype=np.int16)
    final_coast_style[final_coast_land & (final_coastal_plain > 0.34)] = 1
    final_coast_style[final_coast_land & (final_coast_ruggedness > 0.38)] = 2
    final_coast_style[final_coast_land & (extension > 0.42) & (rift > 0.22)] = 3
    final_coast_style[final_coast_land & (volcanism > 0.42) & (active + compression > 0.30)] = 4
    final_coast_style[final_coast_land & (final_shelf_potential > 0.44) & (sediment + passive > 0.32)] = 5
    mixed_coast = final_coast_land & (final_coast_style == 0)
    final_coast_style[mixed_coast] = 6

    final_island_origin = np.zeros((height, width), dtype=np.int16)
    final_island_origin[land] = 1
    final_land_labels, final_land_count = _label_xwrap(land)
    final_land_cells = max(1, int(np.count_nonzero(land)))
    final_world_cells = max(1, height * width)
    continent_threshold = max(256, int(final_land_cells * 0.028))
    island_threshold = max(8, int(final_world_cells * 0.0065))
    if final_land_count:
        counts = np.bincount(final_land_labels.ravel())
        for lab in range(1, final_land_count + 1):
            comp = final_land_labels == lab
            size = int(counts[lab]) if lab < len(counts) else int(np.count_nonzero(comp))
            if size >= continent_threshold:
                final_island_origin[comp] = 1
                continue
            comp_cont = float(np.mean(cont[comp]))
            comp_volc = float(np.mean(volcanism[comp]))
            comp_shelf = float(np.mean(final_shelf_potential[comp]))
            comp_passive = float(np.mean(passive[comp]))
            comp_elev = float(np.mean(elev[comp]))
            if size > island_threshold and comp_cont > 0.48:
                code = 4  # microcontinent / accreted terrane
            elif comp_volc > 0.34 and comp_cont <= 0.50:
                code = 3  # volcanic / island arc
            elif comp_shelf > 0.30 or comp_passive > 0.32:
                code = 2  # shelf island / drowned-margin remnant
            elif comp_elev > 850.0 or comp_volc > 0.24:
                code = 5  # hotspot / high island
            else:
                code = 2
            final_island_origin[comp] = code

    final_boundary_mask = junction_count >= 1
    final_boundary_class = np.zeros((height, width), dtype=np.int16)
    dominance = np.argmax(np.stack([compression, extension, trans, trench + active, volcanism], axis=0), axis=0) + 1
    final_boundary_class[final_boundary_mask] = dominance[final_boundary_mask].astype(np.int16)
    final_boundary_history_density = np.clip(conv * 0.35 + div * 0.30 + trans * 0.22 + uplift * 0.20 + trench * 0.18 + volcanism * 0.12, 0.0, 1.0)
    final_orogeny_history = np.clip(uplift * (0.40 + 0.60 * compression) + orogeny * 0.28, 0.0, 1.0)
    final_rift_history = np.clip(extension * (0.45 + 0.55 * div) + rift * 0.35, 0.0, 1.0)
    final_suture_history = np.clip((orogeny * 0.55 + compression * 0.35 + trans * 0.18) * cont * old_maturity * (1.0 - 0.55 * final_boundary_mask.astype(np.float32)), 0.0, 1.0)

    # Richer diagnostic dominant-circumstance crust classes for v3.  The codes
    # intentionally summarize continuous-field dominance and do not feed back
    # into the terrain equations.
    surface_crust = np.zeros((height, width), dtype=np.int16)
    surface_crust[water] = 1  # abyssal / generic oceanic crust
    surface_crust[water & (ocean_age > 0.62) & (elev < -3200.0)] = 3
    surface_crust[water & (ridge > 0.20)] = 2
    surface_crust[water & (fracture > 0.25)] = 5
    surface_crust[water & (seamount > 0.30)] = 6
    surface_crust[water & (trench > 0.24)] = 4
    # v3 now separates shallow shelf, slope, and rise/deep continental margin so
    # the crust map no longer marks very deep water as ordinary shelf.
    surface_crust[water & (final_shelf_zone == 1)] = 7
    surface_crust[water & (final_shelf_zone == 2)] = 19
    surface_crust[water & (final_shelf_zone == 3)] = 20
    surface_crust[water & (final_shelf_potential > 0.12) & (final_submerged_continental_affinity > 0.28) & (elev <= -2600.0)] = 21

    craton_like = land & (cont > 0.72) & (compression < 0.36) & (extension < 0.36) & (volcanism < 0.32)
    surface_crust[land] = 9
    surface_crust[craton_like] = 8
    surface_crust[land & (cont > 0.56) & (passive > 0.30) & final_coast_band] = 13
    surface_crust[land & (sediment * (0.45 + compression) > 0.30) & (final_slope_n < 0.48)] = 14
    surface_crust[land & (final_suture_history > 0.24)] = 11
    surface_crust[land & (compression > 0.48) & (uplift > 0.38)] = 10
    surface_crust[land & (extension > 0.44) & (cont > 0.38)] = 12
    surface_crust[land & (cont > 0.38) & (cont <= 0.62) & (compression + volcanism > 0.38)] = 15
    surface_crust[land & (cont > 0.45) & (volcanism > 0.42) & (active + compression > 0.26)] = 16
    surface_crust[land & (cont <= 0.45) & (volcanism > 0.42) & (active + compression > 0.20)] = 17
    surface_crust[land & (cont <= 0.45) & (volcanism > 0.28) & ~(active + compression > 0.20)] = 18

    terrain = replace(
        base,
        elevation_m=elev_int.astype(int).tolist(),
        is_land=land.astype(bool).tolist(),
        min_elevation_m=min_elev,
        max_elevation_m=max_elev,
        mean_land_elevation_m=mean_land,
        mean_ocean_depth_m=mean_ocean_depth,
        ocean_fraction=ocean_fraction,
        land_fraction=1.0 - ocean_fraction,
        source="plate_history_v3 unified continuous-field tectonic reconstruction",
        tectonic_plate_id=_diagnostic_class_raster(rough_plate, diag_w=width, diag_h=height),
        tectonic_boundary_class=_diagnostic_class_raster(final_boundary_class, diag_w=width, diag_h=height),
        tectonic_boundary_strength_x1000=_diagnostic_float_x1000(final_boundary_history_density, diag_w=width, diag_h=height),
        tectonic_boundary_width_x1000=_diagnostic_float_x1000(final_boundary_mask.astype(np.float32), diag_w=width, diag_h=height),
        plate_tectonic_boundary_class=_diagnostic_class_raster(final_boundary_class, diag_w=width, diag_h=height),
        plate_tectonic_plate_topology_problem_class=_diagnostic_class_raster(topology, diag_w=width, diag_h=height),
        crust_type=surface_crust.astype(int).tolist(),
        plate_tectonic_continental_crust_x1000=_diagnostic_float_x1000(cont, diag_w=width, diag_h=height),
        plate_tectonic_orogeny_strength_x1000=_diagnostic_float_x1000(uplift, diag_w=width, diag_h=height),
        plate_tectonic_volcanic_arc_x1000=_diagnostic_float_x1000(volcanism, diag_w=width, diag_h=height),
        plate_tectonic_continental_rift_x1000=_diagnostic_float_x1000(extension, diag_w=width, diag_h=height),
        plate_tectonic_plateau_uplift_x1000=_diagnostic_float_x1000(plateau + compression * 0.35, diag_w=width, diag_h=height),
        plate_tectonic_foreland_basin_x1000=_diagnostic_float_x1000(sediment * compression, diag_w=width, diag_h=height),
        plate_tectonic_accreted_terrane_x1000=_diagnostic_float_x1000(final_suture_history, diag_w=width, diag_h=height),
        plate_tectonic_craton_shield_x1000=_diagnostic_float_x1000(cont * old_maturity * (1.0 - final_orogeny_history * 0.35), diag_w=width, diag_h=height),
        plate_tectonic_shelf_width_x1000=_diagnostic_float_x1000(final_shelf_potential, diag_w=width, diag_h=height),
        plate_tectonic_coastal_plain_x1000=_diagnostic_float_x1000(final_coastal_plain, diag_w=width, diag_h=height),
        plate_tectonic_coast_ruggedness_x1000=_diagnostic_float_x1000(final_coast_ruggedness, diag_w=width, diag_h=height),
        plate_tectonic_island_origin_class=_diagnostic_class_raster(final_island_origin, diag_w=width, diag_h=height),
        terrain_mountain_strength_x1000=_diagnostic_float_x1000(uplift, diag_w=width, diag_h=height),
        terrain_erosion_strength_x1000=_diagnostic_float_x1000(erosion_strength, diag_w=width, diag_h=height),
        terrain_deposition_field_x1000=_diagnostic_float_x1000(sediment, diag_w=width, diag_h=height),
        terrain_maturity_x1000=_diagnostic_float_x1000(old_maturity, diag_w=width, diag_h=height),
        terrain_coast_style_class=_diagnostic_class_raster(final_coast_style, diag_w=width, diag_h=height),
        terrain_shelf_width_x1000=_diagnostic_float_x1000(final_shelf_potential, diag_w=width, diag_h=height),
        terrain_submerged_continental_crust_x1000=_diagnostic_float_x1000(final_submerged_continental_affinity, diag_w=width, diag_h=height),
        terrain_continental_shelf_support_x1000=_diagnostic_float_x1000(final_shelf_support, diag_w=width, diag_h=height),
        terrain_shelf_depth_target_x1000=_diagnostic_float_x1000(final_shelf_depth_target, diag_w=width, diag_h=height),
        terrain_shelf_zone_class=_diagnostic_class_raster(final_shelf_zone, diag_w=width, diag_h=height),
        terrain_lake_depth_limit_x1000=_diagnostic_float_x1000(lake_depth_limit_field, diag_w=width, diag_h=height),
        terrain_final_plate_component_class=_diagnostic_class_raster(final_plate_component_class, diag_w=width, diag_h=height),
        terrain_ripple_artifact_risk_x1000=_diagnostic_float_x1000(ripple_artifact_risk, diag_w=width, diag_h=height),
        terrain_coast_ruggedness_x1000=_diagnostic_float_x1000(final_coast_ruggedness, diag_w=width, diag_h=height),
        terrain_island_origin_class=_diagnostic_class_raster(final_island_origin, diag_w=width, diag_h=height),
        terrain_coastal_plain_x1000=_diagnostic_float_x1000(final_coastal_plain, diag_w=width, diag_h=height),
        terrain_basin_field_x1000=_diagnostic_float_x1000(final_suture_history + sediment * 0.35, diag_w=width, diag_h=height),
        terrain_ocean_floor_class=_diagnostic_class_raster(np.where(water, np.argmax(np.stack([ocean_age, ridge, trench, fracture, seamount], axis=0), axis=0) + 1, 0), diag_w=width, diag_h=height),
        terrain_sediment_supply_x1000=_diagnostic_float_x1000(sediment, diag_w=width, diag_h=height),
    )

    snapshot_files: list[str] = []
    if output_dir:
        try:
            from pathlib import Path
            from PIL import Image
            snap_dir = Path(output_dir) / "tectonic_history_v3"
            snap_dir.mkdir(parents=True, exist_ok=True)
            snap_w = min(1200, width)
            snap_h = max(1, int(round(snap_w * height / max(width, 1))))
            vx = _field01(base.plate_tectonic_velocity_x_x1000, 0.0) * 2.0 - 1.0
            vy = _field01(base.plate_tectonic_velocity_y_x1000, 0.0) * 2.0 - 1.0
            y_grid, x_grid = np.mgrid[0:height, 0:width].astype(np.float32)

            def _advect(field, factor: float):
                # Approximate moving-plate snapshots.  This is diagnostic only.
                shift_x = vx * factor * max(6.0, width / 26.0)
                shift_y = vy * factor * max(3.0, height / 34.0)
                coords = np.array([np.clip(y_grid - shift_y, 0, height - 1), (x_grid - shift_x) % width])
                return ndimage.map_coordinates(np.asarray(field, dtype=np.float32), coords, order=1, mode="wrap")

            def _advect_nearest(field, factor: float):
                shift_x = vx * factor * max(6.0, width / 26.0)
                shift_y = vy * factor * max(3.0, height / 34.0)
                coords = np.array([np.clip(y_grid - shift_y, 0, height - 1), (x_grid - shift_x) % width])
                return ndimage.map_coordinates(np.asarray(field, dtype=np.float32), coords, order=0, mode="wrap")

            def _heat_rgb(value, base=(28, 46, 70), hot=(235, 90, 62)):
                v = np.clip(np.asarray(value, dtype=np.float32), 0.0, 1.0)
                rgb = np.zeros((height, width, 3), dtype=np.uint8)
                for ch in range(3):
                    rgb[..., ch] = (base[ch] * (1.0 - v) + hot[ch] * v).astype(np.uint8)
                return rgb

            def _save(name: str, rgb) -> None:
                img = Image.fromarray(np.asarray(rgb, dtype=np.uint8), mode="RGB")
                if img.size != (snap_w, snap_h):
                    img = img.resize((snap_w, snap_h), Image.Resampling.BILINEAR)
                img.save(snap_dir / name)
                snapshot_files.append(f"tectonic_history_v3/{name}")

            summary_rows = ["stage,history_fraction,land_fraction,active_boundary_fraction,mean_continental_support,mean_orogeny,mean_rifting,mean_volcanism,mean_sediment,mean_subsidence\n"]
            boundary_strength = np.clip(conv + div + trans + uplift * 0.5, 0.0, 1.0)
            strong_cut = float(np.percentile(boundary_strength[boundary_strength > 0], 84)) if np.any(boundary_strength > 0) else 1.0
            for pct, factor in [(0, -0.75), (25, -0.35), (50, 0.0), (75, 0.35), (100, 0.75)]:
                c = np.clip(_advect(cont, factor), 0.0, 1.0)
                u = np.clip(_advect(uplift, factor), 0.0, 1.0)
                d = np.clip(_advect(subsidence, factor), 0.0, 1.0)
                b = np.clip(_advect(boundary_strength, factor), 0.0, 1.0)
                l = _advect(elev, factor) > 0.0
                simple = np.zeros((height, width, 3), dtype=np.uint8)
                simple[l] = np.array([82, 105, 58], dtype=np.uint8)
                simple[~l] = np.array([18, 52, 98], dtype=np.uint8)
                simple[..., 0] = np.where(l, np.maximum(simple[..., 0], (70 + 130 * c).astype(np.uint8)), simple[..., 0])
                simple[..., 1] = np.where(l, np.maximum(simple[..., 1], (90 + 80 * c).astype(np.uint8)), simple[..., 1])
                strong = b > max(0.18, strong_cut)
                simple[strong & (u >= d)] = np.array([230, 88, 70], dtype=np.uint8)
                simple[strong & (d > u)] = np.array([95, 135, 245], dtype=np.uint8)
                _save(f"stage_{pct:03d}_summary.png", simple)

                plate_stage = _advect_nearest(rough_plate, factor).astype(np.int32)
                plate_rgb = np.zeros((height, width, 3), dtype=np.uint8)
                max_plate = max(1, int(np.max(rough_plate)))
                for pid in range(max_plate + 1):
                    color = np.array([(53 + pid * 73) % 190 + 45, (97 + pid * 41) % 170 + 55, (139 + pid * 29) % 160 + 65], dtype=np.uint8)
                    plate_rgb[plate_stage == pid] = color
                stage_boundary = np.zeros((height, width), dtype=bool)
                stage_boundary |= plate_stage != np.roll(plate_stage, 1, axis=1)
                stage_boundary |= plate_stage != np.roll(plate_stage, -1, axis=1)
                stage_boundary[:-1, :] |= plate_stage[:-1, :] != plate_stage[1:, :]
                stage_boundary[1:, :] |= plate_stage[1:, :] != plate_stage[:-1, :]
                plate_rgb[stage_boundary] = np.array([20, 20, 24], dtype=np.uint8)
                _save(f"stage_{pct:03d}_plate_positions.png", plate_rgb)

                _save(f"stage_{pct:03d}_active_orogeny.png", _heat_rgb(_advect(final_orogeny_history, factor), base=(62, 72, 78), hot=(232, 82, 60)))
                _save(f"stage_{pct:03d}_active_rifting.png", _heat_rgb(_advect(final_rift_history, factor), base=(38, 58, 82), hot=(90, 145, 245)))
                _save(f"stage_{pct:03d}_active_volcanism.png", _heat_rgb(_advect(volcanism, factor), base=(42, 48, 66), hot=(245, 120, 52)))
                _save(f"stage_{pct:03d}_sediment_accumulation.png", _heat_rgb(_advect(sediment, factor), base=(52, 70, 82), hot=(95, 205, 150)))

                relief = np.zeros((height, width, 3), dtype=np.uint8)
                relief[..., 0] = (35 + 210 * u).astype(np.uint8)
                relief[..., 1] = (45 + 120 * (1.0 - np.maximum(u, d) * 0.45)).astype(np.uint8)
                relief[..., 2] = (55 + 190 * d).astype(np.uint8)
                _save(f"stage_{pct:03d}_uplift_subsidence_detail.png", relief)
                summary_rows.append(f"{pct},{(factor+0.75)/1.5:.3f},{float(np.mean(l)):.5f},{float(np.mean(strong)):.5f},{float(np.mean(c)):.5f},{float(np.mean(_advect(final_orogeny_history, factor))):.5f},{float(np.mean(_advect(final_rift_history, factor))):.5f},{float(np.mean(_advect(volcanism, factor))):.5f},{float(np.mean(_advect(sediment, factor))):.5f},{float(np.mean(d)):.5f}\n")
            (snap_dir / "stage_summary.csv").write_text("".join(summary_rows), encoding="utf-8")
            snapshot_files.append("tectonic_history_v3/stage_summary.csv")
        except Exception as exc:
            snapshot_files.append(f"snapshot_generation_failed:{type(exc).__name__}:{exc}")

    diag = dict(getattr(base, "terrain_diagnostics", None) or {})
    diag["terrain_mode"] = "plate_history_v3"
    diag["plate_history_v3"] = {
        "base_mode": "plate_history_v2_macro_field_source",
        "rule_model": "unified_continuous_fields_same_equations_all_cells",
        "ocean_fraction_target": round(float(ocean_target), 5),
        "ocean_fraction_actual": round(float(ocean_fraction), 5),
        "sea_level_shift_m": round(float(sea_level), 3),
        "mean_crust_thickness_proxy": round(float(np.mean(crust_thickness)), 5),
        "mean_uplift_proxy": round(float(np.mean(uplift)), 5),
        "mean_subsidence_proxy": round(float(np.mean(subsidence)), 5),
        "v3_erosion_deposition_multiplier": round(float(v3_erosion_deposition_multiplier), 3),
        "v3_continental_shelf_strength": round(float(v3_continental_shelf_strength), 3),
        "v3_shelf_width_factor": round(float(v3_shelf_width_factor), 3),
        "mean_erosion_strength": round(float(np.mean(erosion_strength)), 5),
        "mean_low_energy_deposition": round(float(np.mean(low_energy_deposition)), 5),
        "mean_submerged_continental_affinity_water": round(float(np.mean(final_submerged_continental_affinity[water])) if np.any(water) else 0.0, 5),
        "mean_shelf_support": round(float(np.mean(final_shelf_support[water])) if np.any(water) else 0.0, 5),
        "continental_apron_water_share_gt_025": round(float(np.mean((continental_apron > 0.25)[water])) if np.any(water) else 0.0, 5),
        "broad_shelf_ocean_share": round(float(np.mean((final_shelf_potential > 0.42)[water])) if np.any(water) else 0.0, 5),
        "shallow_shelf_ocean_share_depth_lt_250m": round(float(np.mean(((final_shelf_potential > 0.34) & (elev > -250.0))[water])) if np.any(water) else 0.0, 5),
        "shelf_or_slope_ocean_share_depth_lt_1000m": round(float(np.mean(((final_shelf_potential > 0.22) & (elev > -1000.0))[water])) if np.any(water) else 0.0, 5),
        "shelf_zone_shallow_share_of_ocean": round(float(np.mean((final_shelf_zone == 1)[water])) if np.any(water) else 0.0, 5),
        "shelf_zone_slope_rise_share_of_ocean": round(float(np.mean(((final_shelf_zone == 2) | (final_shelf_zone == 3))[water])) if np.any(water) else 0.0, 5),
        "snake_arc_cells_sunk": int(snake_sunk),
        "lake_depth_limited_cells": int(lake_depth_limited_cells),
        "deep_lake_supported_cells": int(deep_lake_supported_cells),
        "lake_depth_limit_min_m": round(float(lake_depth_limit_min_m), 2),
        "tiny_artifact_water_cells_filled": int(lake_tiny_filled_cells),
        "coastal_lagoon_artifact_cells_filled": int(coastal_lagoon_filled_cells),
        "plate_boundary_exchange_cells": int(np.count_nonzero(exch)),
        "plate_fragments_reassigned_cells": int(np.count_nonzero(final_plate_component_class == 1)),
        "plate_fragments_promoted_microplate_cells": int(np.count_nonzero(final_plate_component_class == 2)),
        "ripple_artifact_risk_mean_ocean": round(float(np.mean(ripple_artifact_risk[water])) if np.any(water) else 0.0, 5),
        "final_active_boundary_cells": int(np.count_nonzero(final_boundary_mask)),
        "final_land_component_count": int(final_land_count),
        "final_island_component_count": int(sum(1 for lab in range(1, final_land_count + 1) if np.count_nonzero(final_land_labels == lab) < continent_threshold)),
        "mean_final_shelf_potential": round(float(np.mean(final_shelf_potential[final_coast_water])) if np.any(final_coast_water) else 0.0, 5),
        "mean_final_coast_ruggedness": round(float(np.mean(final_coast_ruggedness[final_coast_land])) if np.any(final_coast_land) else 0.0, 5),
        "plus_junction_count": int(np.count_nonzero(topology == 3)),
        "initial_plus_junction_risk_cells": int(np.count_nonzero(plus_junction_risk > 0.12)),
        "t_triple_junction_candidate_count": int(np.count_nonzero(topology == 2)),
        "land_changed_cells_from_v2": int(np.count_nonzero(land != land_base)),
        "elevation_changed_cells_from_v2": int(np.count_nonzero(elev_int != np.rint(elev_base).astype(np.int32))),
        "mean_abs_elevation_delta_from_v2_m": round(float(np.mean(np.abs(elev_int.astype(np.float32) - elev_base))), 3),
        "tectonic_history_snapshot_files": list(snapshot_files),
        "description": "Experimental v3 mode: final terrain is rebuilt from continuous crust thickness/buoyancy, uplift, subsidence, sediment, age/maturity, slope, water proximity and motion fields. Diagnostic classes summarize dominant circumstances only; they do not choose separate terrain rule sets.",
    }

    terrain.terrain_diagnostics = diag
    return terrain


def _generate_plate_history_v4_scaffold(
    rng: random.Random,
    planet: Planet,
    hydrosphere: Hydrosphere,
    geology: GeologyState,
    config: PlanetProfileConfig,
    *,
    output_dir: str | None = None,
) -> TerrainMap:
    """Experimental v4 terrain branch built on the stable v3 model.

    v4 deliberately keeps plate_history_v3 stable.  It starts from the v3
    continuous-field terrain, then applies isolated experiments for the pending
    topology/island work: wavy non-Voronoi final plate diagnostics, small
    microplate/sliver promotion along deformed boundaries, and physically
    supported volcanic island-chain uplift.  The terrain equations remain
    continuous; diagnostic classes summarize what dominated a cell after the
    v4 shaping pass.
    """
    try:
        import numpy as np
        from scipy import ndimage
    except ImportError as exc:
        raise RuntimeError("NumPy and SciPy are required for plate_history_v4. Install with: pip install -r requirements.txt") from exc

    base = _generate_plate_history_v3_scaffold(rng, planet, hydrosphere, geology, config, output_dir=output_dir)
    width, height = int(base.width), int(base.height)
    np_rng = np.random.default_rng(rng.randint(1, 2_147_483_647))

    try:
        from worldgen.terrain_review import derive_terrain_controls
        terrain_controls = derive_terrain_controls(planet, hydrosphere, geology, config, output_dir=output_dir)
    except Exception:
        terrain_controls = {}

    def _v4_control_float(name: str, default: float, low: float, high: float) -> float:
        try:
            value = terrain_controls.get(name, default) if isinstance(terrain_controls, dict) else default
            if value is None or value == "":
                value = default
            return clamp(float(value), low, high)
        except Exception:
            return clamp(float(default), low, high)

    v4_topology_strength = _v4_control_float("v4_topology_strength", 1.0, 0.0, 2.5)
    v4_island_strength = _v4_control_float("v4_island_strength", 1.0, 0.0, 2.8)
    v4_rift_strength = _v4_control_float("v4_rift_strength", 1.0, 0.0, 2.5)

    elev = np.asarray(base.elevation_m, dtype=np.float32).copy()
    v3_elev_reference = elev.copy()
    original_land = np.asarray(base.is_land, dtype=bool)

    def _field01(value, default: float = 0.0):
        if value is None:
            return np.full((height, width), float(default), dtype=np.float32)
        arr = np.asarray(value, dtype=np.float32)
        if arr.shape != (height, width):
            arr = _resize_float_field(arr, width, height)
        if arr.size and (float(np.nanmax(arr)) > 1.5 or float(np.nanmin(arr)) < -0.05):
            arr = arr / 1000.0
        return np.clip(np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0), 0.0, 1.0).astype(np.float32, copy=False)

    def _class_field(value, default: int = 0):
        if value is None:
            return np.full((height, width), int(default), dtype=np.int32)
        arr = np.asarray(value, dtype=np.int32)
        if arr.shape != (height, width):
            arr = _resize_int_field(arr, width, height)
        return arr.astype(np.int32, copy=False)

    def _dist_xwrap(feature_mask):
        feature = np.asarray(feature_mask, dtype=bool)
        if not np.any(feature):
            return np.full((height, width), float(max(width, height)), dtype=np.float32)
        tiled = np.concatenate([~feature, ~feature, ~feature], axis=1)
        return ndimage.distance_transform_edt(tiled)[:, width:2 * width].astype(np.float32, copy=False)

    def _label_xwrap_v4(mask):
        labels, n = ndimage.label(np.asarray(mask, dtype=bool))
        if n <= 1 or width <= 1:
            return labels.astype(np.int32, copy=False), int(n)
        parent = list(range(n + 1))
        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a
        def union(a, b):
            if a == 0 or b == 0:
                return
            ra, rb = find(int(a)), find(int(b))
            if ra != rb:
                parent[rb] = ra
        for yy in range(height):
            union(labels[yy, 0], labels[yy, width - 1])
        remap = {0: 0}
        next_id = 1
        out = np.zeros_like(labels, dtype=np.int32)
        for idx, lab in np.ndenumerate(labels):
            if lab == 0:
                continue
            root = find(int(lab))
            if root not in remap:
                remap[root] = next_id
                next_id += 1
            out[idx] = remap[root]
        return out, next_id - 1

    def _dominant_adjacent_plate_v4(plate_arr, comp_mask, fallback: int) -> int:
        vals = []
        for rolled in (np.roll(plate_arr, 1, axis=1), np.roll(plate_arr, -1, axis=1)):
            neigh = rolled[comp_mask]
            vals.append(neigh[neigh != fallback])
        north_mask = np.zeros_like(comp_mask, dtype=bool)
        south_mask = np.zeros_like(comp_mask, dtype=bool)
        north_mask[1:, :] = comp_mask[:-1, :]
        south_mask[:-1, :] = comp_mask[1:, :]
        if np.any(north_mask):
            neigh = plate_arr[north_mask]
            vals.append(neigh[neigh != fallback])
        if np.any(south_mask):
            neigh = plate_arr[south_mask]
            vals.append(neigh[neigh != fallback])
        nonempty = [v.astype(np.int64, copy=False) for v in vals if v.size]
        if nonempty:
            all_vals = np.concatenate(nonempty)
            if all_vals.size:
                unique, counts = np.unique(all_vals, return_counts=True)
                return int(unique[int(np.argmax(counts))])
        return int(fallback)

    def _cohere_plate_ids_xwrap_v4(plate_arr):
        arr = np.asarray(plate_arr, dtype=np.int32).copy()
        comp_class = np.zeros_like(arr, dtype=np.int16)
        next_pid = int(arr.max(initial=0)) + 1
        world_cells = max(1, int(arr.size))
        original_ids = [int(v) for v in np.unique(arr) if int(v) != 0]
        for pid in original_ids:
            mask = arr == pid
            if not np.any(mask):
                continue
            labels_pid, n_pid = _label_xwrap_v4(mask)
            if n_pid <= 1:
                continue
            counts = np.bincount(labels_pid.ravel())
            counts[0] = 0
            largest = int(np.argmax(counts))
            largest_size = int(counts[largest]) if largest < len(counts) else 0
            promote_threshold = max(72, int(world_cells * 0.0014), int(largest_size * 0.24))
            for lab in range(1, n_pid + 1):
                if lab == largest:
                    continue
                comp = labels_pid == lab
                size = int(counts[lab]) if lab < len(counts) else int(np.count_nonzero(comp))
                if size >= promote_threshold:
                    arr[comp] = next_pid
                    comp_class[comp] = 2
                    next_pid += 1
                else:
                    replacement = _dominant_adjacent_plate_v4(arr, comp, pid)
                    if replacement == pid:
                        replacement = next_pid
                        next_pid += 1
                        comp_class[comp] = 2
                    else:
                        comp_class[comp] = 1
                    arr[comp] = replacement
        return arr.astype(np.int32, copy=False), comp_class

    cont = _field01(getattr(base, "plate_tectonic_continental_crust_x1000", None), 0.0)
    volcanism = _field01(getattr(base, "plate_tectonic_volcanic_arc_x1000", None), 0.0)
    convergence = _field01(getattr(base, "plate_tectonic_convergence_x1000", None), 0.0)
    divergence = _field01(getattr(base, "plate_tectonic_divergence_x1000", None), 0.0)
    transform = _field01(getattr(base, "plate_tectonic_transform_x1000", None), 0.0)
    rift = np.maximum(
        _field01(getattr(base, "plate_tectonic_continental_rift_x1000", None), 0.0),
        _field01(getattr(base, "terrain_rift_field_x1000", None), 0.0),
    )
    trench = _field01(getattr(base, "plate_tectonic_trench_x1000", None), 0.0)
    ridge = np.maximum(
        _field01(getattr(base, "plate_tectonic_mid_ocean_ridge_x1000", None), 0.0),
        _field01(getattr(base, "terrain_mid_ocean_ridge_x1000", None), 0.0),
    )
    seamount = _field01(getattr(base, "terrain_seamount_x1000", None), 0.0)
    shelf_support = _field01(getattr(base, "terrain_continental_shelf_support_x1000", None), 0.0)
    submerged_cont = _field01(getattr(base, "terrain_submerged_continental_crust_x1000", None), 0.0)
    sediment = _field01(getattr(base, "terrain_deposition_field_x1000", None), 0.0)
    boundary_history = _field01(getattr(base, "tectonic_boundary_strength_x1000", None), 0.0)
    old_ripple = _field01(getattr(base, "terrain_ripple_artifact_risk_x1000", None), 0.0)

    plates = _class_field(getattr(base, "tectonic_plate_id", None), 0)
    boundary_base = _class_field(getattr(base, "tectonic_boundary_class", None), 0)
    crust = _class_field(getattr(base, "crust_type", None), 0)
    island_origin = _class_field(getattr(base, "terrain_island_origin_class", None), 0)

    y_grid, x_grid = np.mgrid[0:height, 0:width]
    # Continuous low-frequency displacement: this makes the final v4 diagnostic
    # plate boundaries less Voronoi-perfect without changing v3 or using hard
    # per-cell terrain rule sets.  The displacement is intentionally small and
    # globally smooth so plate IDs remain coherent.
    nx = np_rng.normal(0.0, 1.0, size=(height, width)).astype(np.float32)
    ny = np_rng.normal(0.0, 1.0, size=(height, width)).astype(np.float32)
    sigma_y = max(4.0, height / 28.0)
    sigma_x = max(8.0, width / 28.0)
    nx = ndimage.gaussian_filter(nx, sigma=(sigma_y, sigma_x), mode="wrap")
    ny = ndimage.gaussian_filter(ny, sigma=(sigma_y, sigma_x), mode="wrap")
    for arr in (nx, ny):
        arr -= float(np.mean(arr))
        std = float(np.std(arr))
        if std > 1.0e-6:
            arr /= std
    # Update 32: make v4 strength controls visibly effective.  Earlier builds
    # accidentally omitted v4_* controls from derive_terrain_controls, so these
    # values stayed at 1.0.  Keep neutral behavior near 1.0, but make values near
    # 2.0 produce measurably stronger deformation/islands/rifts for A/B runs.
    topology_scale = 0.25 + 0.85 * v4_topology_strength
    amp_x = max(1.0, min(44.0, (width / 170.0) * topology_scale))
    amp_y = max(0.6, min(22.0, (height / 195.0) * topology_scale))
    coords = np.array([np.clip(y_grid + ny * amp_y, 0, height - 1), (x_grid + nx * amp_x) % width])
    plates_v4 = ndimage.map_coordinates(plates.astype(np.float32), coords, order=0, mode="wrap").astype(np.int32)
    boundary_v4 = ndimage.map_coordinates(boundary_base.astype(np.float32), coords, order=0, mode="wrap").astype(np.int32)
    plates_v4, v4_component_cleanup = _cohere_plate_ids_xwrap_v4(plates_v4)

    final_boundary = np.zeros((height, width), dtype=bool)
    final_boundary |= plates_v4 != np.roll(plates_v4, 1, axis=1)
    final_boundary |= plates_v4 != np.roll(plates_v4, -1, axis=1)
    final_boundary[:-1, :] |= plates_v4[:-1, :] != plates_v4[1:, :]
    final_boundary[1:, :] |= plates_v4[1:, :] != plates_v4[:-1, :]
    boundary_deformation = np.clip(
        ndimage.gaussian_filter(final_boundary.astype(np.float32), sigma=(1.0, 1.6), mode="wrap")
        * (1.55 + 0.55 * v4_topology_strength),
        0.0,
        1.0,
    )
    boundary_deformation = np.clip(
        np.maximum(boundary_deformation, boundary_history * (0.36 + 0.18 * v4_topology_strength)),
        0.0,
        1.0,
    )

    # Prefer T/Y-style junctions over clean four-way crosses by damping the
    # exact crossing core and spreading relief into nearby oblique branches.
    neighbor_count = (
        (plates_v4 != np.roll(plates_v4, 1, axis=1)).astype(np.int16)
        + (plates_v4 != np.roll(plates_v4, -1, axis=1)).astype(np.int16)
        + np.vstack(((plates_v4[0:1, :] != plates_v4[0:1, :]).astype(np.int16), (plates_v4[1:, :] != plates_v4[:-1, :]).astype(np.int16)))
        + np.vstack(((plates_v4[:-1, :] != plates_v4[1:, :]).astype(np.int16), (plates_v4[-1:, :] != plates_v4[-1:, :]).astype(np.int16)))
    )
    plus_risk = (neighbor_count >= 4) & final_boundary
    plus_soft = ndimage.gaussian_filter(plus_risk.astype(np.float32), sigma=(1.2, 1.8), mode="wrap")
    elev -= plus_soft * np.clip(convergence + boundary_deformation, 0.0, 1.0) * (95.0 + 55.0 * v4_topology_strength)

    # Microplates/sliver diagnostics.  This does not yet replace the full future
    # plate-topology model, but it marks coherent small boundary-belt fragments
    # and lets the diagnostic plate map stop showing impossible fragments as if
    # they belonged to a distant parent plate.
    topology_class = np.zeros((height, width), dtype=np.int32)
    micro_threshold = max(0.16, 0.38 - 0.085 * v4_topology_strength)
    candidate_micro = final_boundary & (boundary_deformation > 0.28) & ((divergence + transform + volcanism) > micro_threshold) & (cont < 0.74)
    cand_labels, cand_count = ndimage.label(candidate_micro, structure=np.ones((3, 3), dtype=np.uint8))
    next_plate = int(np.max(plates_v4)) + 1
    micro_cells = 0
    sliver_cells = 0
    for lab in range(1, cand_count + 1):
        cells = cand_labels == lab
        size = int(np.count_nonzero(cells))
        if size < max(8, int(width * height * (0.000006 / max(0.65, v4_topology_strength)))):
            continue
        if size > max(80, int(width * height * (0.0015 + 0.00075 * v4_topology_strength))):
            continue
        dil = ndimage.binary_dilation(cells, structure=np.ones((3, 3), dtype=np.uint8), iterations=1)
        local = dil & ~cells
        neighbor_ids = plates_v4[local]
        neighbor_ids = neighbor_ids[neighbor_ids >= 0]
        if neighbor_ids.size < 1:
            continue
        # Promote larger coherent boundary-belt fragments into local microplates;
        # mark very thin pieces as sliver plates but keep them contiguous.
        if size >= max(28, int(width * height * 0.000035)):
            plates_v4[cells] = next_plate
            topology_class[cells] = 2
            next_plate += 1
            micro_cells += size
        else:
            topology_class[cells] = 1
            sliver_cells += size

    # Re-cohere final diagnostic plate IDs after microplate promotion.  This is
    # still diagnostic/topology-only: terrain equations are continuous, but the
    # final plate map should not show physically impossible disconnected plates.
    plates_v4, v4_component_cleanup_2 = _cohere_plate_ids_xwrap_v4(plates_v4)
    v4_component_cleanup = np.maximum(v4_component_cleanup, v4_component_cleanup_2)
    topology_class[(v4_component_cleanup == 1) & (topology_class == 0)] = 3
    topology_class[v4_component_cleanup == 2] = np.maximum(topology_class[v4_component_cleanup == 2], 4)

    # Recompute final boundaries after microplate promotion and contiguity cleanup.
    final_boundary = np.zeros((height, width), dtype=bool)
    final_boundary |= plates_v4 != np.roll(plates_v4, 1, axis=1)
    final_boundary |= plates_v4 != np.roll(plates_v4, -1, axis=1)
    final_boundary[:-1, :] |= plates_v4[:-1, :] != plates_v4[1:, :]
    final_boundary[1:, :] |= plates_v4[1:, :] != plates_v4[:-1, :]

    # Volcanic island support: v4 adds medium/small oceanic island chains only
    # where oceanic volcanism, active boundaries, ridge/seamount support, and low
    # continental-shelf support agree.  This avoids using random islands as a
    # substitute for true plate/island circumstances.
    water0 = elev < 0.0
    oceanic = water0 & (cont < 0.48) & (submerged_cont < 0.55)
    boundary_activity = np.clip(np.maximum.reduce([convergence, divergence, transform, boundary_deformation]), 0.0, 1.0)

    # Update 29: v4 rift-cut support.  This is a continuous field, not a hard
    # cell rule: rifting/divergence, transform shear, deformed boundaries, and
    # continental/transitional crust jointly lower supported corridors into
    # rift valleys, gulfs, narrow seas, or lake-prone basins.  Sediment and
    # shelf support soften the cut so passive shelves do not become trenches.
    rift_seed = np.clip(0.42 * divergence + 0.26 * rift + 0.18 * transform + 0.14 * boundary_deformation - 0.18 * trench, 0.0, 1.0)
    water_distance = _dist_xwrap(water0)
    lowland_reach = np.exp(-np.minimum(water_distance, 80.0) / 24.0).astype(np.float32)
    rift_cut_support = np.clip(
        rift_seed
        * (0.28 + 0.72 * np.clip(cont + 0.35 * submerged_cont, 0.0, 1.0))
        * (0.62 + 0.38 * lowland_reach)
        * (1.0 - 0.34 * shelf_support)
        * (1.0 - 0.22 * sediment),
        0.0,
        1.0,
    )
    rift_cut_support = ndimage.gaussian_filter(rift_cut_support, sigma=(0.85, 1.55), mode="wrap").astype(np.float32, copy=False)
    # Update 35: keep the useful Update 34 rift improvement, but keep it
    # bounded so rifts become more readable without slicing continents into
    # artificial canals.  This is intentionally the only stronger v4 effect
    # carried forward from the rejected Update 34 terrain balance.
    rift_cut_threshold = max(0.075, 0.275 - 0.082 * v4_rift_strength)
    rift_cut_strength = np.clip((rift_cut_support - rift_cut_threshold) / max(0.14, 1.0 - rift_cut_threshold), 0.0, 1.0) ** max(0.72, 1.05 - 0.10 * v4_rift_strength)
    rift_lowering = rift_cut_strength * (0.54 + 0.92 * v4_rift_strength) * (220.0 + 470.0 * divergence + 350.0 * rift + 205.0 * transform)
    # Protect high young mountains from being sliced into graphic-looking canals;
    # old/low relief crust and margin zones are easier to cut.
    high_relief_protection = np.clip((elev - 1200.0) / 2600.0, 0.0, 0.55)
    elev -= rift_lowering * (1.0 - high_relief_protection)
    low_weak_land = original_land & (elev < 520.0)
    elev[low_weak_land] -= (rift_cut_strength[low_weak_land] * (72.0 + 120.0 * v4_rift_strength + 145.0 * lowland_reach[low_weak_land])).astype(np.float32)
    topology_class[(rift_cut_strength > 0.20) & (topology_class == 0)] = 5

    # Update 33: turn the strongest coherent v4 rift/sliver corridors into
    # actual local final plate IDs, not only diagnostic paint.  This is still
    # conservative and continuous-field driven: a corridor must be supported by
    # rifting/divergence/shear, deformation, weak shelf suppression, and limited
    # continental core strength.  It moves v4 toward native topology while v3
    # remains untouched.
    native_sliver_seed = (
        (rift_cut_strength > max(0.16, 0.40 - 0.115 * v4_rift_strength))
        & (boundary_deformation > max(0.10, 0.30 - 0.07 * v4_topology_strength))
        & ((divergence + transform + rift) > max(0.34, 0.78 - 0.13 * v4_topology_strength))
        & (cont < max(0.76, 0.88 - 0.04 * v4_topology_strength))
        & (shelf_support < 0.82)
    )
    # Thicken just enough to become coherent plate strips rather than one-cell
    # dashes, then restrict back to supported corridors.
    native_sliver_seed = ndimage.binary_closing(native_sliver_seed, structure=np.ones((3, 3), dtype=np.uint8), iterations=1)
    native_sliver_seed &= (rift_cut_strength > 0.14) | (boundary_deformation > 0.22)
    native_labels, native_count = ndimage.label(native_sliver_seed, structure=np.ones((3, 3), dtype=np.uint8))
    native_sliver_cells = 0
    native_microplate_cells = 0
    min_native = max(10, int(width * height * 0.000010))
    max_native = max(260, int(width * height * (0.0026 + 0.0008 * max(0.0, v4_topology_strength - 1.0))))
    for lab in range(1, native_count + 1):
        cells = native_labels == lab
        size = int(np.count_nonzero(cells))
        if size < min_native or size > max_native:
            continue
        # Avoid turning entire coastline shelves into plates; prefer elongated
        # corridors by checking bounding-box aspect in map-cell space.
        rr, cc = np.where(cells)
        if rr.size == 0:
            continue
        span_r = max(1, int(rr.max() - rr.min() + 1))
        span_c = max(1, int(cc.max() - cc.min() + 1))
        aspect = max(span_r / span_c, span_c / span_r)
        if aspect < 1.55 and size < max(42, min_native * 3):
            continue
        plates_v4[cells] = next_plate
        if size >= max(48, int(width * height * 0.000055)):
            topology_class[cells] = 6  # native rift-generated micro/sliver plate
            native_microplate_cells += size
        else:
            topology_class[cells] = 1
            native_sliver_cells += size
        next_plate += 1
    if native_sliver_cells or native_microplate_cells:
        plates_v4, v4_component_cleanup_3 = _cohere_plate_ids_xwrap_v4(plates_v4)
        v4_component_cleanup = np.maximum(v4_component_cleanup, v4_component_cleanup_3)
        final_boundary = np.zeros((height, width), dtype=bool)
        final_boundary |= plates_v4 != np.roll(plates_v4, 1, axis=1)
        final_boundary |= plates_v4 != np.roll(plates_v4, -1, axis=1)
        final_boundary[:-1, :] |= plates_v4[:-1, :] != plates_v4[1:, :]
        final_boundary[1:, :] |= plates_v4[1:, :] != plates_v4[:-1, :]
        boundary_deformation = np.clip(
            np.maximum(
                boundary_deformation,
                ndimage.gaussian_filter(final_boundary.astype(np.float32), sigma=(0.85, 1.35), mode="wrap") * (0.75 + 0.16 * v4_topology_strength),
            ),
            0.0,
            1.0,
        )

    # Update 30: v4 mountain-branch support.  Use the final deformed boundary
    # fabric plus convergence/volcanism to distribute uplift into oblique
    # branches instead of only raising a single neat boundary ribbon.  This is
    # still a continuous field; the diagnostic map explains where the extra
    # branch support came from.
    branch_noise = np_rng.normal(0.0, 1.0, size=(height, width)).astype(np.float32)
    branch_noise = ndimage.gaussian_filter(branch_noise, sigma=(max(1.8, height / 145.0), max(3.2, width / 130.0)), mode="wrap")
    branch_noise -= float(np.mean(branch_noise))
    branch_std = float(np.std(branch_noise))
    if branch_std > 1.0e-6:
        branch_noise /= branch_std
    branch_noise = np.clip(0.55 + 0.20 * branch_noise, 0.0, 1.0)
    branch_spread = ndimage.gaussian_filter(final_boundary.astype(np.float32), sigma=(1.7, 2.8), mode="wrap")
    mountain_branch_support = np.clip(
        branch_spread
        * (0.50 * convergence + 0.24 * volcanism + 0.18 * boundary_deformation + 0.08 * transform)
        * branch_noise
        * (0.42 + 0.58 * np.clip(cont + 0.28 * submerged_cont, 0.0, 1.0))
        * (1.0 - 0.46 * rift_cut_strength)
        * (1.0 - 0.28 * sediment),
        0.0,
        1.0,
    ).astype(np.float32, copy=False)
    mountain_branch_support = ndimage.gaussian_filter(mountain_branch_support, sigma=(0.75, 1.25), mode="wrap").astype(np.float32, copy=False)
    branch_lift = mountain_branch_support * (0.40 + 0.72 * v4_topology_strength) * (235.0 + 690.0 * convergence + 390.0 * volcanism + 160.0 * boundary_deformation)
    elev += branch_lift * original_land.astype(np.float32)
    # Slightly deepen adjacent weak basins so branching mountains create drainage
    # and foreland contrast rather than just globally raising land.
    branch_foreland = ndimage.gaussian_filter(mountain_branch_support, sigma=(2.2, 3.4), mode="wrap") - mountain_branch_support
    branch_foreland = np.clip(branch_foreland, 0.0, 1.0)
    elev -= branch_foreland * original_land.astype(np.float32) * np.clip(1.0 - cont * 0.35, 0.35, 1.0) * (34.0 + 20.0 * v4_topology_strength)

    island_chain_noise = np_rng.normal(0.0, 1.0, size=(height, width)).astype(np.float32)
    island_chain_noise = ndimage.gaussian_filter(island_chain_noise, sigma=(max(1.5, height / 170.0), max(3.0, width / 150.0)), mode="wrap")
    island_chain_noise -= float(np.mean(island_chain_noise))
    noise_std = float(np.std(island_chain_noise))
    if noise_std > 1.0e-6:
        island_chain_noise /= noise_std
    island_chain_noise = np.clip(0.5 + 0.22 * island_chain_noise, 0.0, 1.0)
    ridge_or_seamount = np.maximum(seamount, ridge * 0.72)
    island_chain_spine = np.clip(0.48 * volcanism + 0.24 * ridge_or_seamount + 0.20 * boundary_activity + 0.08 * divergence, 0.0, 1.0)
    chain_coherence = ndimage.gaussian_filter(island_chain_spine * oceanic.astype(np.float32), sigma=(0.85, 2.1), mode="wrap")
    volcanic_island_support = np.clip(
        (0.78 * island_chain_spine + 0.22 * chain_coherence)
        * (0.36 + 0.64 * island_chain_noise)
        * (1.0 - 0.54 * shelf_support)
        * (1.0 - 0.45 * cont)
        * (1.0 - 0.18 * trench),
        0.0,
        1.0,
    )
    # Update 33: make island strength alter both cutoff and component size.
    # Higher settings lower the percentile and add archipelago-shoulder support,
    # creating medium chains where the field already supports them.
    if np.any(oceanic):
        # Update 35: modestly lower the cutoff versus Update 33 so supported
        # island chains appear, but avoid the Update 34 behavior where volcanic
        # uplift dominated ordinary continents.
        thresh = float(np.percentile(volcanic_island_support[oceanic], max(54.0, 84.0 - 8.5 * v4_island_strength)))
    else:
        thresh = 1.0
    # Lower than v3-style cleanup thresholds because v4 explicitly needs more
    # volcanic small/medium islands and archipelagos.  Support is still
    # continuous and suppressed on continental shelves, so this should not add
    # random islands everywhere.
    island_cut = max(0.050, min(0.62, thresh - 0.095 * max(0.0, v4_island_strength - 1.0)))
    island_raw = np.clip((volcanic_island_support - island_cut) / max(0.07, 1.0 - island_cut), 0.0, 1.0)
    island_raw *= oceanic.astype(np.float32)
    # Smooth and close the support so islands can become medium archipelagos, not
    # just isolated dots.  At high island strength, use a larger along-chain
    # shoulder but still require nonzero support, so islands are not random.
    island_lift_field = ndimage.gaussian_filter(island_raw, sigma=(0.95, 1.85 + 0.35 * v4_island_strength), mode="wrap")
    island_lift_field = np.clip(island_lift_field + 0.22 * chain_coherence * island_raw.max(initial=0.0), 0.0, 1.0)
    island_lift_field = island_lift_field ** max(0.58, 1.08 - 0.17 * v4_island_strength)
    volcanic_lift_m = island_lift_field * (0.30 + 0.66 * v4_island_strength) * (1540.0 + 2920.0 * volcanism + 1420.0 * boundary_activity + 980.0 * ridge_or_seamount)
    # Keep shelves from becoming volcanic island carpets, and preserve deep-ocean
    # gaps inside island arcs.
    volcanic_lift_m *= np.clip(1.0 - 0.38 * shelf_support - 0.24 * sediment, 0.0, 1.0)
    # Update 35: volcanic/uplifted terrain must not be outside geomorphology.
    # Apply a conservative local weathering term to supported volcanic uplift
    # before it becomes final elevation.  This is not blanket continent shaving:
    # it only trims steep/high volcanic additions and leaves water gaps intact.
    volcanic_weathering = np.clip(
        island_lift_field
        * (0.26 + 0.36 * volcanism + 0.18 * boundary_activity + 0.10 * ridge_or_seamount)
        * np.clip(0.65 + 0.22 * sediment + 0.16 * shelf_support, 0.55, 1.05),
        0.0,
        1.0,
    ).astype(np.float32, copy=False)
    volcanic_lift_m *= np.clip(1.0 - 0.24 * volcanic_weathering, 0.72, 1.0)
    elev += volcanic_lift_m

    # Mild oceanic arc shoulder uplift makes island chains look less like perfect
    # beads, but keep it below sea level unless support is strong.
    arc_shoulder = ndimage.gaussian_filter(island_lift_field, sigma=(1.8, 3.0), mode="wrap")
    elev += arc_shoulder * np.clip(volcanism + boundary_activity, 0.0, 1.0) * (70.0 + 70.0 * v4_island_strength) * oceanic.astype(np.float32)

    land = elev > 0.0
    new_island = land & ~original_land & (volcanic_island_support > 0.22)
    island_origin[new_island] = 4
    crust[new_island & (convergence + volcanism > 0.55)] = 17
    crust[new_island & ~(convergence + volcanism > 0.55)] = 18

    island_chain_class = np.zeros((height, width), dtype=np.int32)
    island_chain_class[original_land] = 1
    support_area = (volcanic_island_support > max(0.10, island_cut * 0.82)) & oceanic
    island_chain_class[support_area & (convergence + volcanism > 0.72)] = 2
    island_chain_class[support_area & (ridge_or_seamount >= np.maximum(volcanism, convergence))] = 3
    island_chain_class[support_area & (divergence > 0.42) & (rift_cut_support > 0.25)] = 5
    island_chain_class[support_area & (island_chain_class == 0)] = 4
    existing_island_like = original_land & (island_origin > 0)
    island_chain_class[existing_island_like] = 6
    island_chain_class[new_island] = 7

    # Update 33: component-aware archipelago shaping.  Supported chain cells get
    # slightly raised cores and lowered gaps, increasing medium island groups
    # while preserving ocean gaps inside arcs.
    chain_core = (island_lift_field > max(0.12, 0.34 - 0.06 * v4_island_strength)) & oceanic
    chain_labels, chain_count = ndimage.label(chain_core, structure=np.ones((3, 3), dtype=np.uint8))
    archipelago_core_cells = 0
    for lab in range(1, chain_count + 1):
        cells = chain_labels == lab
        size = int(np.count_nonzero(cells))
        if size < max(6, int(width * height * 0.000004)):
            continue
        if size > max(900, int(width * height * 0.008)):
            continue
        local_core = cells & (volcanic_island_support > island_cut * 0.92)
        if not np.any(local_core):
            continue
        core_boost = ndimage.gaussian_filter(local_core.astype(np.float32), sigma=(0.65, 1.15), mode="wrap")
        boost_m = core_boost * (65.0 + 95.0 * v4_island_strength) * np.clip(volcanism + ridge_or_seamount + 0.35 * boundary_activity, 0.0, 1.0)
        elev += boost_m * oceanic.astype(np.float32)
        gap = ndimage.gaussian_filter(cells.astype(np.float32), sigma=(1.4, 2.6), mode="wrap") - core_boost
        elev -= np.clip(gap, 0.0, 1.0) * oceanic.astype(np.float32) * (18.0 + 28.0 * v4_island_strength)
        archipelago_core_cells += int(np.count_nonzero(local_core))
    land = elev > 0.0
    new_island = land & ~original_land & (volcanic_island_support > 0.18)
    island_origin[new_island] = 4
    island_chain_class[new_island] = 7

    # Remove a few one-cell volcanic flecks while preserving irregular chains.
    labels, count = ndimage.label(land, structure=np.ones((3, 3), dtype=np.uint8))
    if count:
        sizes = np.bincount(labels.ravel())
        min_keep = max(4, int(width * height * 0.000002))
        tiny_new = np.zeros_like(land, dtype=bool)
        for lab in range(1, count + 1):
            if sizes[lab] < min_keep and np.any(new_island & (labels == lab)):
                tiny_new |= labels == lab
        if np.any(tiny_new):
            elev[tiny_new] = np.minimum(elev[tiny_new], -8.0 - 42.0 * volcanic_island_support[tiny_new])
            land = elev > 0.0
            new_island &= ~tiny_new
            island_chain_class[tiny_new] = 0

    # Update 11: artifact cleanup and diagnostic reclassification.  This pass is
    # deliberately conservative: it only adjusts the experimental v4 branch and
    # does not change plate_history_v3 or older stable modes.  The goal is to
    # remove unsupported snake islands/spurs, reduce checkerboard land/water
    # artifacts, tighten submerged-crust/trench/ocean-floor diagnostics, and
    # make coast-style classes more interpretable without doing a full plate
    # reconstruction rewrite.
    def _neighbor_count_bool(mask: np.ndarray) -> np.ndarray:
        m = np.asarray(mask, dtype=bool)
        north = np.vstack((m[0:1, :], m[:-1, :]))
        south = np.vstack((m[1:, :], m[-1:, :]))
        west = np.roll(m, 1, axis=1)
        east = np.roll(m, -1, axis=1)
        nw = np.roll(north, 1, axis=1)
        ne = np.roll(north, -1, axis=1)
        sw = np.roll(south, 1, axis=1)
        se = np.roll(south, -1, axis=1)
        return (
            north.astype(np.int16)
            + south.astype(np.int16)
            + west.astype(np.int16)
            + east.astype(np.int16)
            + nw.astype(np.int16)
            + ne.astype(np.int16)
            + sw.astype(np.int16)
            + se.astype(np.int16)
        )

    # Start from the latest v4 land mask and trim only cells with weak geologic
    # support.  This catches checkerboard islands and low unsupported tendrils
    # while protecting barrier islands on shelves and volcanic/ridge-supported
    # island arcs.
    land = elev > 0.0
    land_neighbors = _neighbor_count_bool(land)
    volcanic_or_arc_support = np.clip(0.58 * volcanic_island_support + 0.24 * volcanism + 0.18 * boundary_activity, 0.0, 1.0)
    barrier_or_lagoon_support = np.clip(0.56 * shelf_support + 0.30 * sediment + 0.14 * (1.0 - trench), 0.0, 1.0)
    mountain_or_peninsula_support = np.clip(0.54 * mountain_branch_support + 0.24 * convergence + 0.16 * cont + 0.06 * boundary_deformation, 0.0, 1.0)
    protected_thin_land = (
        (volcanic_or_arc_support > 0.30)
        | ((barrier_or_lagoon_support > 0.58) & (elev < 80.0))
        | (mountain_or_peninsula_support > 0.34)
        | (island_chain_class >= 2)
    )
    checker_land = land & (land_neighbors <= 2) & (elev < 180.0) & (~protected_thin_land)
    thin_spur_cells = land & (land_neighbors <= 3) & (elev < 260.0) & (volcanic_or_arc_support < 0.22) & (mountain_or_peninsula_support < 0.24) & (barrier_or_lagoon_support < 0.46)
    cleanup_mask = checker_land | thin_spur_cells

    # Component-level snake-island cleanup: high-aspect islands without volcanic,
    # shelf/barrier, or orogenic support are either lowered or broken into short
    # archipelago segments.  Valid volcanic arcs and barrier islands are left
    # alone, because the user explicitly wants real barrier islands preserved.
    comp_labels, comp_count = _label_xwrap_v4(land)
    snake_island_cells = 0
    spur_cleanup_cells = int(np.count_nonzero(thin_spur_cells))
    checker_cleanup_cells = int(np.count_nonzero(checker_land))
    world_cells = max(1, width * height)
    for lab in range(1, comp_count + 1):
        cells = comp_labels == lab
        size = int(np.count_nonzero(cells))
        if size <= 0:
            continue
        rr, cc = np.where(cells)
        span_r = max(1, int(rr.max() - rr.min() + 1))
        span_c = max(1, int(cc.max() - cc.min() + 1))
        aspect = float(max(span_r / span_c, span_c / span_r))
        area_frac = size / float(world_cells)
        mean_arc = float(np.mean(volcanic_or_arc_support[cells]))
        mean_barrier = float(np.mean(barrier_or_lagoon_support[cells]))
        mean_mountain = float(np.mean(mountain_or_peninsula_support[cells]))
        # Tiny one/two-cell chains and very high-aspect components are usually
        # the unwanted snake-island artifact unless they have strong support.
        unsupported_snake = (
            area_frac < 0.0018
            and aspect > 7.0
            and mean_arc < 0.28
            and mean_barrier < 0.56
            and mean_mountain < 0.27
        )
        broken_chain = (
            area_frac < 0.0030
            and aspect > 10.0
            and mean_arc < 0.22
            and mean_barrier < 0.48
        )
        if unsupported_snake or broken_chain:
            cleanup_mask |= cells
            snake_island_cells += size

    if np.any(cleanup_mask):
        local_support = np.maximum.reduce([volcanic_or_arc_support, barrier_or_lagoon_support * 0.86, mountain_or_peninsula_support])
        lower = cleanup_mask & (local_support < 0.34)
        if np.any(lower):
            elev[lower] = np.minimum(elev[lower], -10.0 - 55.0 * np.clip(0.34 - local_support[lower], 0.0, 0.34) / 0.34)
        # Weakly-supported cells become low coastal/wetland breaks rather than
        # hard cliffs, helping long chains break into island groups naturally.
        soften = cleanup_mask & (~lower)
        if np.any(soften):
            elev[soften] = np.minimum(elev[soften], np.maximum(-6.0, elev[soften] * 0.42))

    # Fill isolated one-cell water holes on land only when they are not supported
    # by lake/depression diagnostics.  This reduces checkerboard land/water
    # noise without eliminating real lakes; lake breaching/above-sea lakes are
    # left for the hydrology workstream.
    land = elev > 0.0
    water_neighbors = _neighbor_count_bool(~land)
    land_neighbors = _neighbor_count_bool(land)
    unsupported_hole = (~land) & (land_neighbors >= 7) & (elev > -35.0) & (sediment < 0.38) & (rift_cut_strength < 0.26)
    hole_fill_cells = int(np.count_nonzero(unsupported_hole))
    if np.any(unsupported_hole):
        local_land_mean = (
            np.roll(elev, 1, axis=1)
            + np.roll(elev, -1, axis=1)
            + np.vstack((elev[0:1, :], elev[:-1, :]))
            + np.vstack((elev[1:, :], elev[-1:, :]))
        ) / 4.0
        elev[unsupported_hole] = np.maximum(2.0, local_land_mean[unsupported_hole] * 0.45)

    land = elev > 0.0
    water = ~land

    # Tighten trench support: keep the diagnostic trench core close to active
    # convergent/subduction boundaries, and avoid broad smeared trough blankets.
    trench_core_seed = water & (trench > 0.34) & ((convergence + 0.72 * boundary_deformation + 0.45 * volcanism) > 0.45)
    if np.any(trench_core_seed):
        trench_core = ndimage.gaussian_filter(trench_core_seed.astype(np.float32), sigma=(0.45, 0.85), mode="wrap")
        narrow_trench = np.clip(np.maximum(trench_core, trench * 0.35 * (convergence + boundary_deformation)), 0.0, 1.0)
    else:
        narrow_trench = np.zeros_like(trench, dtype=np.float32)
    broad_trench_excess = water & (trench > 0.24) & (narrow_trench < 0.10)
    if np.any(broad_trench_excess):
        elev[broad_trench_excess] += (70.0 * np.clip(trench[broad_trench_excess] - narrow_trench[broad_trench_excess], 0.0, 1.0)).astype(np.float32)
    land = elev > 0.0
    water = ~land

    # Reclassify underwater crust from final depth/coast/shelf support so deep
    # oceans are not dominated by continental colors.  Keep explicit shelves,
    # slopes, and rises, but require proximity/depth/support for continental
    # affinity; otherwise classify as oceanic/ridge/trench/fracture/seamount.
    dist_to_land = _dist_xwrap(land)
    depth = np.maximum(-elev, 0.0).astype(np.float32)
    shelf_like = water & (depth < 260.0) & ((shelf_support > 0.34) | (submerged_cont > 0.40) | (dist_to_land < 3.2))
    upper_slope = water & (~shelf_like) & (depth < 1050.0) & ((shelf_support > 0.22) | (submerged_cont > 0.32) | (dist_to_land < 6.5))
    continental_rise = water & (~shelf_like) & (~upper_slope) & (depth < 2100.0) & ((shelf_support > 0.18) | (submerged_cont > 0.38)) & (dist_to_land < 11.0)
    deep_margin = water & (~shelf_like) & (~upper_slope) & (~continental_rise) & (depth < 2700.0) & (submerged_cont > 0.54) & (dist_to_land < 14.0)
    ridge_zone = water & (ridge > 0.34) & (narrow_trench < 0.18)
    fracture_zone = water & (transform > 0.42) & (boundary_deformation > 0.18) & (ridge < 0.56) & (narrow_trench < 0.22) & (shelf_support < 0.42)
    seamount_zone = water & (ridge_or_seamount > 0.44) & (ridge < 0.45) & (narrow_trench < 0.20)
    old_oceanic = water & (depth > 3600.0) & (ridge < 0.20) & (narrow_trench < 0.12) & (transform < 0.26)

    crust[water] = 1
    crust[old_oceanic] = 3
    crust[ridge_zone] = 2
    crust[fracture_zone] = 5
    crust[seamount_zone] = 6
    crust[shelf_like] = 7
    crust[upper_slope] = 19
    crust[continental_rise] = 20
    crust[deep_margin] = 21
    crust[narrow_trench > 0.30] = 4
    # Restore land classes where the earlier water pass may have overwritten
    # cells that became land after cleanup/fill.
    crust[land & (crust < 8)] = np.where(cont[land & (crust < 8)] > 0.58, 9, 13)
    crust[new_island & land & (convergence + volcanism > 0.55)] = 17
    crust[new_island & land & ~(convergence + volcanism > 0.55)] = 18

    ocean_floor_class = np.zeros((height, width), dtype=np.int32)
    ocean_floor_class[water] = 1  # abyssal/generic oceanic
    ocean_floor_class[ridge_zone] = 2
    ocean_floor_class[narrow_trench > 0.26] = 3
    ocean_floor_class[fracture_zone] = 4
    ocean_floor_class[seamount_zone] = 5
    # Keep transform corridors bounded to boundary fabrics; if they dominate the
    # ocean, demote the weakest ones to abyssal so the diagnostic cannot show a
    # world-scale fracture province.
    transform_cells = int(np.count_nonzero(ocean_floor_class == 4))
    ocean_cells = max(1, int(np.count_nonzero(water)))
    if transform_cells > int(ocean_cells * 0.08):
        strength = (transform * boundary_deformation * (1.0 - 0.45 * shelf_support))[ocean_floor_class == 4]
        cutoff = float(np.percentile(strength, 100.0 * (1.0 - 0.08 * ocean_cells / max(1, transform_cells)))) if strength.size else 1.0
        weak_transform = (ocean_floor_class == 4) & ((transform * boundary_deformation * (1.0 - 0.45 * shelf_support)) < cutoff)
        ocean_floor_class[weak_transform] = 1

    # More informative boundary/polarity classes.  This is still diagnostic, but
    # it prevents ambiguous boundaries from all being painted as ocean-ocean
    # subduction and reduces discontinuous speckle by using the final fields.
    final_boundary = np.zeros((height, width), dtype=bool)
    final_boundary |= plates_v4 != np.roll(plates_v4, 1, axis=1)
    final_boundary |= plates_v4 != np.roll(plates_v4, -1, axis=1)
    final_boundary[:-1, :] |= plates_v4[:-1, :] != plates_v4[1:, :]
    final_boundary[1:, :] |= plates_v4[1:, :] != plates_v4[:-1, :]
    active_boundary = final_boundary | (boundary_deformation > 0.22)
    near_continent = (dist_to_land < 7.0) | (shelf_support > 0.28) | (submerged_cont > 0.32) | (cont > 0.42)
    subduction_polarity = np.zeros((height, width), dtype=np.int32)
    convergent_like = active_boundary & ((convergence + narrow_trench + volcanism * 0.35) > np.maximum(transform, divergence + rift) + 0.08)
    subduction_polarity[convergent_like & water & (~near_continent)] = 1  # ocean-ocean subduction
    subduction_polarity[convergent_like & water & near_continent] = 2      # ocean-continent subduction
    subduction_polarity[convergent_like & land & (cont > 0.45)] = 3        # continent-continent collision/orogeny
    subduction_polarity[active_boundary & (transform >= np.maximum(convergence, divergence + rift)) & (transform > 0.26)] = 4
    subduction_polarity[active_boundary & ((divergence + rift) > np.maximum(convergence, transform) + 0.05) & ((divergence + rift) > 0.25)] = 5
    uncertain = active_boundary & (subduction_polarity == 0) & ((convergence + transform + divergence + rift + volcanism) > 0.42)
    subduction_polarity[uncertain] = 6

    # Plate thin-neck cleanup.  This affects final v4 plate diagnostics only; it
    # does not attempt to reconstruct real plate chronology, but it prevents
    # two-lobed plates from being connected by fragile one/two-cell bridges.
    plate_neck_cells = 0
    for pid in [int(v) for v in np.unique(plates_v4) if int(v) != 0]:
        pmask = plates_v4 == pid
        if int(np.count_nonzero(pmask)) < max(140, int(world_cells * 0.00022)):
            continue
        same4 = (
            (np.roll(plates_v4, 1, axis=1) == pid).astype(np.int16)
            + (np.roll(plates_v4, -1, axis=1) == pid).astype(np.int16)
            + np.vstack(((plates_v4[0:1, :] == pid).astype(np.int16), (plates_v4[:-1, :] == pid).astype(np.int16)))
            + np.vstack(((plates_v4[1:, :] == pid).astype(np.int16), (plates_v4[-1:, :] == pid).astype(np.int16)))
        )
        neck = pmask & (same4 <= 2) & ((boundary_deformation + transform + rift_cut_strength) > 0.52)
        neck_size = int(np.count_nonzero(neck))
        if 0 < neck_size < max(400, int(world_cells * 0.0015)):
            replacement = _dominant_adjacent_plate_v4(plates_v4, neck, pid)
            if replacement == pid:
                replacement = next_plate
                next_plate += 1
            plates_v4[neck] = replacement
            topology_class[neck] = np.maximum(topology_class[neck], 4)
            plate_neck_cells += neck_size
    if plate_neck_cells:
        plates_v4, v4_component_cleanup_4 = _cohere_plate_ids_xwrap_v4(plates_v4)
        v4_component_cleanup = np.maximum(v4_component_cleanup, v4_component_cleanup_4)
        final_boundary = np.zeros((height, width), dtype=bool)
        final_boundary |= plates_v4 != np.roll(plates_v4, 1, axis=1)
        final_boundary |= plates_v4 != np.roll(plates_v4, -1, axis=1)
        final_boundary[:-1, :] |= plates_v4[:-1, :] != plates_v4[1:, :]
        final_boundary[1:, :] |= plates_v4[1:, :] != plates_v4[:-1, :]

    # Coast diagnostics: recompute ruggedness and margin classes from final coast
    # geometry instead of allowing inland/deep-ocean support fields to paint the
    # map.  Class 5 is now reserved for true deltaic/wet coastal plains with
    # sediment support; ordinary shelf coasts become passive/plain or mixed.
    coast_water_dist = _dist_xwrap(water)
    coast_land_dist = _dist_xwrap(land)
    coast_band_land = land & (coast_water_dist <= 4.0)
    elev_n = np.vstack((elev[0:1, :], elev[:-1, :]))
    elev_s = np.vstack((elev[1:, :], elev[-1:, :]))
    elev_w = np.roll(elev, 1, axis=1)
    elev_e = np.roll(elev, -1, axis=1)
    local_relief = np.maximum.reduce([np.abs(elev - elev_n), np.abs(elev - elev_s), np.abs(elev - elev_w), np.abs(elev - elev_e)]).astype(np.float32)
    coast_ruggedness = np.clip((local_relief / 520.0) * coast_band_land.astype(np.float32) * np.clip(1.35 - coast_water_dist / 5.0, 0.0, 1.0), 0.0, 1.0)
    # Fjord/active-margin support extends ruggedness only where there is actual
    # mountain/convergent support near the coast.
    coast_ruggedness = np.maximum(coast_ruggedness, coast_band_land.astype(np.float32) * np.clip(0.58 * mountain_branch_support + 0.34 * convergence + 0.20 * narrow_trench, 0.0, 1.0))
    terrain_coast_style = np.zeros((height, width), dtype=np.int32)
    active_margin = coast_band_land & ((narrow_trench > 0.18) | (convergence > 0.34) | (coast_ruggedness > 0.45))
    rifted_margin = coast_band_land & ((rift_cut_strength > 0.24) | (rift > 0.30)) & (~active_margin)
    volcanic_arc_coast = coast_band_land & ((volcanic_or_arc_support > 0.38) | (island_chain_class >= 2))
    true_deltaic_plain = coast_band_land & (elev < 125.0) & (sediment > 0.55) & (coast_ruggedness < 0.34) & (~active_margin)
    passive_plain = coast_band_land & (elev < 180.0) & (shelf_support > 0.32) & (~active_margin) & (~rifted_margin) & (~true_deltaic_plain) & (~volcanic_arc_coast)
    mixed_margin = coast_band_land & ~(active_margin | rifted_margin | volcanic_arc_coast | true_deltaic_plain | passive_plain)
    terrain_coast_style[passive_plain] = 1
    terrain_coast_style[active_margin] = 2
    terrain_coast_style[rifted_margin] = 3
    terrain_coast_style[volcanic_arc_coast] = 4
    terrain_coast_style[true_deltaic_plain] = 5
    terrain_coast_style[mixed_margin] = 6

    # Add modest foothill/valley texture tied to orogen networks and final land,
    # not global noise.  This aims to give mountains better structure without
    # reintroducing the old ripple fields.
    foothill_support = np.clip(ndimage.gaussian_filter(mountain_branch_support, sigma=(2.8, 4.6), mode="wrap") - mountain_branch_support * 0.35, 0.0, 1.0) * land.astype(np.float32)
    valley_seed_noise = np_rng.normal(0.0, 1.0, size=(height, width)).astype(np.float32)
    valley_seed_noise = ndimage.gaussian_filter(valley_seed_noise, sigma=(max(2.0, height / 180.0), max(3.8, width / 155.0)), mode="wrap")
    valley_seed_noise -= float(np.mean(valley_seed_noise))
    valley_std = float(np.std(valley_seed_noise))
    if valley_std > 1.0e-6:
        valley_seed_noise /= valley_std
    valley_corridor = land & (foothill_support > 0.08) & (valley_seed_noise < -0.22) & (elev > 180.0) & (coast_water_dist > 2.0)
    if np.any(foothill_support > 0.05):
        elev += foothill_support * np.clip(1.0 - coast_ruggedness * 0.35, 0.55, 1.0) * (26.0 + 34.0 * v4_topology_strength)
    if np.any(valley_corridor):
        valley_lowering = np.clip(foothill_support * (0.35 + 0.45 * mountain_branch_support), 0.0, 1.0) * (34.0 + 62.0 * v4_topology_strength)
        elev[valley_corridor] -= valley_lowering[valley_corridor]
    land = elev > 0.0
    water = ~land

    # Boundary class for the v4 plate map: recompute from continuous fields at
    # final boundary cells instead of inheriting historical class clutter.
    boundary_v4 = np.zeros((height, width), dtype=np.int32)
    dom = np.argmax(np.stack([convergence, divergence, transform, narrow_trench, volcanism], axis=0), axis=0) + 1
    boundary_v4[final_boundary] = dom[final_boundary].astype(np.int32)

    # Update 31: interpretable v4 topology network classes.  These are
    # diagnostic classes only; the terrain equations above remain continuous.
    # They make it easier to tell whether v4 created convergent branches,
    # rift cuts, transforms, volcanic arcs/island chains, triple junctions, or
    # micro/sliver fragments without relying on one plate-color image.
    boundary_network_class = np.zeros((height, width), dtype=np.int32)
    boundary_network_class[final_boundary & (dom == 1)] = 1  # convergent/orogenic boundary
    boundary_network_class[final_boundary & (dom == 2)] = 2  # divergent/rift boundary
    boundary_network_class[final_boundary & (dom == 3)] = 3  # transform/shear boundary
    boundary_network_class[final_boundary & (dom == 4)] = 4  # trench/subduction boundary
    boundary_network_class[final_boundary & (dom == 5)] = 5  # volcanic/arc boundary
    triple_or_complex = final_boundary & (neighbor_count >= 3)
    boundary_network_class[triple_or_complex] = 6
    boundary_network_class[(topology_class == 1) | (topology_class == 2) | (topology_class == 4)] = 7
    boundary_network_class[(rift_cut_strength > 0.26) & (boundary_network_class == 0)] = 8
    boundary_network_class[new_island & (boundary_network_class == 0)] = 9

    orogen_network_class = np.zeros((height, width), dtype=np.int32)
    branch_active = (mountain_branch_support > 0.18) & land
    primary_orogen = branch_active & (convergence >= np.maximum.reduce([volcanism, transform, divergence]))
    volcanic_orogen = branch_active & (volcanism > 0.34) & (volcanism >= convergence * 0.72)
    oblique_branch = branch_active & (transform + boundary_deformation > 0.48) & ~primary_orogen
    foreland_branch = (branch_foreland > 0.08) & land & (sediment + cont > 0.48)
    rifted_highland = (rift_cut_strength > 0.18) & land & (elev > 160.0)
    orogen_network_class[primary_orogen] = 1
    orogen_network_class[oblique_branch] = 2
    orogen_network_class[volcanic_orogen] = 3
    orogen_network_class[foreland_branch & (orogen_network_class == 0)] = 4
    orogen_network_class[rifted_highland & (orogen_network_class == 0)] = 5
    orogen_network_class[triple_or_complex & land] = np.maximum(orogen_network_class[triple_or_complex & land], 6)

    # Update 32: verification map for v4 controls.  It answers: if the user
    # changes topology/island/rift strength, which cells should respond?  This
    # makes it immediately visible when a UI/CLI control is not being consumed.
    topology_response = np.clip(0.55 * boundary_deformation + 0.45 * mountain_branch_support, 0.0, 1.0) * max(0.0, v4_topology_strength) / 2.5
    island_response = np.clip(0.50 * volcanic_island_support + 0.50 * island_lift_field, 0.0, 1.0) * max(0.0, v4_island_strength) / 2.8
    rift_response = np.clip(0.55 * rift_cut_support + 0.45 * rift_cut_strength, 0.0, 1.0) * max(0.0, v4_rift_strength) / 2.5
    control_response_class = np.zeros((height, width), dtype=np.int32)
    response_stack = np.stack([topology_response, island_response, rift_response], axis=0)
    response_max = np.max(response_stack, axis=0)
    response_arg = np.argmax(response_stack, axis=0) + 1
    active = response_max > 0.13
    control_response_class[active] = response_arg[active].astype(np.int32)
    mixed_count = ((topology_response > 0.13).astype(np.int8) + (island_response > 0.13).astype(np.int8) + (rift_response > 0.13).astype(np.int8))
    control_response_class[(mixed_count >= 2) & active] = 4
    control_response_class[(mixed_count >= 3) & active] = 5

    # Update 33: direct effect diagnostics.  Unlike support maps, these show
    # what v4 actually changed relative to stable v3 in meters/classes.
    v4_elevation_delta = np.rint(elev - v3_elev_reference).astype(np.int32)
    landform_change_class = np.zeros((height, width), dtype=np.int32)
    strong_up = v4_elevation_delta > 90
    strong_down = v4_elevation_delta < -70
    landform_change_class[strong_up & (mountain_branch_support >= np.maximum(island_lift_field, rift_cut_strength))] = 1
    landform_change_class[strong_up & ((island_lift_field > 0.08) | new_island)] = 2
    landform_change_class[strong_down & (rift_cut_strength > 0.12)] = 3
    landform_change_class[((topology_class == 1) | (topology_class == 2) | (topology_class == 6)) & (landform_change_class == 0)] = 4
    mixed_v4_change = (np.abs(v4_elevation_delta) > 70) & (mixed_count >= 2)
    landform_change_class[mixed_v4_change] = 5

    elev_int = np.rint(elev).astype(np.int32)
    water = ~land
    min_elev = int(elev_int.min())
    max_elev = int(elev_int.max())
    ocean_fraction = float(np.mean(water))
    mean_land = float(np.mean(elev_int[land])) if np.any(land) else 0.0
    mean_ocean_depth = float(-np.mean(elev_int[water])) if np.any(water) else 0.0

    terrain = replace(
        base,
        elevation_m=elev_int.astype(int).tolist(),
        is_land=land.astype(bool).tolist(),
        min_elevation_m=min_elev,
        max_elevation_m=max_elev,
        mean_land_elevation_m=mean_land,
        mean_ocean_depth_m=mean_ocean_depth,
        ocean_fraction=ocean_fraction,
        land_fraction=1.0 - ocean_fraction,
        source="plate_history_v4 experimental topology/island branch built on stable v3",
        tectonic_plate_id=_diagnostic_class_raster(plates_v4, diag_w=width, diag_h=height),
        tectonic_boundary_class=_diagnostic_class_raster(boundary_v4, diag_w=width, diag_h=height),
        plate_tectonic_boundary_class=_diagnostic_class_raster(boundary_v4, diag_w=width, diag_h=height),
        plate_tectonic_subduction_polarity=_diagnostic_class_raster(subduction_polarity, diag_w=width, diag_h=height),
        plate_tectonic_ocean_floor_class=_diagnostic_class_raster(ocean_floor_class, diag_w=width, diag_h=height),
        plate_tectonic_trench_x1000=_diagnostic_float_x1000(narrow_trench, diag_w=width, diag_h=height),
        terrain_ocean_floor_class=_diagnostic_class_raster(ocean_floor_class, diag_w=width, diag_h=height),
        terrain_trench_x1000=_diagnostic_float_x1000(narrow_trench, diag_w=width, diag_h=height),
        terrain_fracture_zone_x1000=_diagnostic_float_x1000((ocean_floor_class == 4).astype(np.float32), diag_w=width, diag_h=height),
        crust_type=crust.astype(int).tolist(),
        terrain_coast_style_class=_diagnostic_class_raster(terrain_coast_style, diag_w=width, diag_h=height),
        terrain_coast_ruggedness_x1000=_diagnostic_float_x1000(coast_ruggedness, diag_w=width, diag_h=height),
        terrain_valley_corridor_x1000=_diagnostic_float_x1000(valley_corridor.astype(np.float32), diag_w=width, diag_h=height),
        terrain_island_origin_class=_diagnostic_class_raster(island_origin, diag_w=width, diag_h=height),
        terrain_v4_boundary_deformation_x1000=_diagnostic_float_x1000(boundary_deformation, diag_w=width, diag_h=height),
        terrain_v4_volcanic_island_support_x1000=_diagnostic_float_x1000(volcanic_island_support, diag_w=width, diag_h=height),
        terrain_v4_rift_cut_support_x1000=_diagnostic_float_x1000(rift_cut_support, diag_w=width, diag_h=height),
        terrain_v4_mountain_branch_support_x1000=_diagnostic_float_x1000(mountain_branch_support, diag_w=width, diag_h=height),
        terrain_v4_topology_class=_diagnostic_class_raster(topology_class, diag_w=width, diag_h=height),
        terrain_v4_island_chain_class=_diagnostic_class_raster(island_chain_class, diag_w=width, diag_h=height),
        terrain_v4_boundary_network_class=_diagnostic_class_raster(boundary_network_class, diag_w=width, diag_h=height),
        terrain_v4_orogen_network_class=_diagnostic_class_raster(orogen_network_class, diag_w=width, diag_h=height),
        terrain_v4_control_response_class=_diagnostic_class_raster(control_response_class, diag_w=width, diag_h=height),
        terrain_v4_elevation_delta_m=_diagnostic_class_raster(v4_elevation_delta, diag_w=width, diag_h=height),
        terrain_v4_landform_change_class=_diagnostic_class_raster(landform_change_class, diag_w=width, diag_h=height),
    )
    diag = dict(getattr(base, "terrain_diagnostics", None) or {})
    diag["terrain_mode"] = "plate_history_v4"
    diag["plate_history_v4"] = {
        "base_mode": "plate_history_v3_stable",
        "rule_model": "v3_continuous_fields_plus_experimental_topology_and_volcanic_island_shaping",
        "plate_boundary_displacement_amp_x_cells": round(float(amp_x), 3),
        "plate_boundary_displacement_amp_y_cells": round(float(amp_y), 3),
        "microplate_cells_promoted": int(micro_cells),
        "sliver_plate_cells_marked": int(sliver_cells),
        "new_volcanic_island_cells": int(np.count_nonzero(new_island)),
        "v4_rift_cut_cells": int(np.count_nonzero(rift_cut_strength > 0.20)),
        "v4_plate_fragments_reassigned_cells": int(np.count_nonzero(v4_component_cleanup == 1)),
        "v4_plate_fragments_promoted_microplate_cells": int(np.count_nonzero(v4_component_cleanup == 2)),
        "v4_volcanic_island_cutoff": round(float(island_cut), 5),
        "mean_v4_volcanic_island_support_ocean": round(float(np.mean(volcanic_island_support[water])) if np.any(water) else 0.0, 5),
        "mean_v4_boundary_deformation": round(float(np.mean(boundary_deformation)), 5),
        "mean_v4_rift_cut_support": round(float(np.mean(rift_cut_support)), 5),
        "mean_v4_mountain_branch_support_land": round(float(np.mean(mountain_branch_support[land])) if np.any(land) else 0.0, 5),
        "v4_topology_strength": round(float(v4_topology_strength), 3),
        "v4_island_strength": round(float(v4_island_strength), 3),
        "v4_rift_strength": round(float(v4_rift_strength), 3),
        "v4_controls_read_from": "derive_terrain_controls/stage_overrides",
        "v4_effective_topology_scale": round(float(topology_scale), 5),
        "v4_effective_boundary_amp_x_cells": round(float(amp_x), 5),
        "v4_effective_boundary_amp_y_cells": round(float(amp_y), 5),
        "v4_effective_rift_cut_threshold": round(float(rift_cut_threshold), 5),
        "v4_effective_island_cutoff": round(float(island_cut), 5),
        "v4_island_chain_support_cells": int(np.count_nonzero(island_chain_class > 1)),
        "v4_boundary_network_active_cells": int(np.count_nonzero(boundary_network_class > 0)),
        "v4_orogen_network_active_land_cells": int(np.count_nonzero(orogen_network_class > 0)),
        "v4_control_response_active_cells": int(np.count_nonzero(control_response_class > 0)),
        "v4_control_response_topology_cells": int(np.count_nonzero(control_response_class == 1)),
        "v4_control_response_island_cells": int(np.count_nonzero(control_response_class == 2)),
        "v4_control_response_rift_cells": int(np.count_nonzero(control_response_class == 3)),
        "v4_control_response_mixed_cells": int(np.count_nonzero(control_response_class >= 4)),
        "v4_complex_or_triple_boundary_cells": int(np.count_nonzero(boundary_network_class == 6)),
        "v4_rift_corridor_cells": int(np.count_nonzero(boundary_network_class == 8)),
        "v4_native_sliver_cells": int(native_sliver_cells),
        "v4_native_microplate_cells": int(native_microplate_cells),
        "v4_archipelago_core_cells": int(archipelago_core_cells),
        "mean_v4_volcanic_weathering": round(float(np.mean(volcanic_weathering[oceanic])) if np.any(oceanic) else 0.0, 5),
        "max_v4_volcanic_uplift_m_after_weathering": int(np.max(volcanic_lift_m)) if volcanic_lift_m.size else 0,
        "v4_changed_cells_gt_50m": int(np.count_nonzero(np.abs(v4_elevation_delta) > 50)),
        "v4_changed_cells_gt_150m": int(np.count_nonzero(np.abs(v4_elevation_delta) > 150)),
        "v4_mean_abs_elevation_delta_m": round(float(np.mean(np.abs(v4_elevation_delta))), 3),
        "v4_max_uplift_m": int(np.max(v4_elevation_delta)) if v4_elevation_delta.size else 0,
        "v4_max_lowering_m": int(np.min(v4_elevation_delta)) if v4_elevation_delta.size else 0,
        "v4_landform_change_active_cells": int(np.count_nonzero(landform_change_class > 0)),
        "update11_checkerboard_land_cells_removed_or_softened": int(checker_cleanup_cells),
        "update11_thin_spur_cells_removed_or_softened": int(spur_cleanup_cells),
        "update11_snake_island_cells_removed_or_softened": int(snake_island_cells),
        "update11_isolated_water_holes_filled": int(hole_fill_cells),
        "update11_plate_neck_cells_reassigned": int(plate_neck_cells),
        "update11_narrow_trench_cells": int(np.count_nonzero(narrow_trench > 0.26)),
        "update11_transform_ocean_floor_fraction": round(float(np.count_nonzero(ocean_floor_class == 4)) / max(1, int(np.count_nonzero(water))), 5),
        "update11_shelf_like_water_cells": int(np.count_nonzero(shelf_like)),
        "update11_true_deltaic_plain_cells": int(np.count_nonzero(true_deltaic_plain)),
        "update11_coast_ruggedness_cells": int(np.count_nonzero(coast_ruggedness > 0.35)),
        "update11_valley_corridor_cells": int(np.count_nonzero(valley_corridor)),
        "plus_junction_core_cells_damped": int(np.count_nonzero(plus_risk)),
        "land_fraction_before_v4": round(float(np.mean(original_land)), 5),
        "land_fraction_after_v4": round(float(np.mean(land)), 5),
        "description": "v4 starts from stable v3, then warps/coheres final diagnostic plates, adds native rift/sliver plate ownership where supported, modestly strengthens rifts, raises weathered volcanic island chains, and in Update 11 applies conservative artifact cleanup for unsupported snake islands/spurs, checkerboard cells, underwater crust dominance, broad trenches, detached coast ruggedness, and ambiguous boundary/ocean-floor classes.",
    }
    terrain.terrain_diagnostics = diag
    return terrain


def _create_plate_tectonic_v1_workspace_terrain(
    rng: random.Random,
    planet: Planet,
    hydrosphere: Hydrosphere,
    geology: GeologyState,
    config: PlanetProfileConfig,
) -> TerrainMap:
    """Create a native plate-mode workspace without running legacy terrain.

    Plate Terrain 10 intentionally stops bootstrapping plate mode from the legacy
    generator.  The workspace is a coherent domain/continent seed used only to
    guide native plate seeding; the visible terrain is rebuilt by the plate
    foundation, relief, coast, and final QA passes that follow.
    """
    try:
        import numpy as np
        from scipy import ndimage
    except ImportError as exc:
        raise RuntimeError("NumPy and SciPy are required for plate terrain generation. Install with: pip install -r requirements.txt") from exc

    width = max(int(config.min_map_width), int(config.map_width))
    height = max(int(config.min_map_height), int(config.map_height))
    np_rng = np.random.default_rng(int(rng.randrange(1, 2**31 - 1)))
    lats = np.linspace(90.0 - 90.0 / height, -90.0 + 90.0 / height, height, dtype=np.float32)
    lons = np.linspace(-180.0 + 180.0 / width, 180.0 - 180.0 / width, width, dtype=np.float32)
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    cos_lat = np.maximum(0.18, np.cos(np.radians(lat_grid))).astype(np.float32)

    ocean_target = clamp(float(getattr(hydrosphere, "ocean_fraction_target", 0.65) or 0.65), 0.08, 0.94)
    requested_land = clamp(1.0 - ocean_target, 0.05, 0.82)
    heat = clamp(float(getattr(geology, "internal_heat", 0.75) or 0.75), 0.0, 2.0) / 2.0
    roughness = clamp(float(getattr(geology, "surface_roughness", 0.35) or 0.35), 0.0, 1.5) / 1.5

    # Long-term plate mode begins from continental domains, not per-cell land
    # thresholds.  Domains have irregular, warped support fields so later plate
    # IDs do not become perfect circular pedestals.
    # Plate Terrain 12: bias the native workspace toward fewer, larger continental
    # domains.  Earlier builds produced too many separate land assemblies before
    # plate setup, which later passes could only decorate as island-worlds.
    continent_count = int(round(1.4 + 2.35 * requested_land + 0.70 * heat + rng.uniform(-0.7, 0.55)))
    continent_count = max(1, min(5, continent_count))
    score = np.full((height, width), -0.35, dtype=np.float32)
    domain_field = np.zeros((height, width), dtype=np.float32)
    for idx in range(continent_count):
        lon0 = rng.uniform(-180.0, 180.0)
        lat0 = math.degrees(math.asin(rng.uniform(-0.94, 0.94)))
        major = rng.uniform(28.0, 76.0) * (1.0 + 0.60 * requested_land)
        minor = rng.uniform(17.0, 48.0) * (1.0 + 0.28 * requested_land)
        angle = rng.uniform(0.0, math.tau)
        # Component domains are chains of lobes, not single ellipses.
        nodes = rng.randint(4, 9)
        node_lon = lon0
        node_lat = lat0
        heading = angle
        for node in range(nodes):
            heading += rng.uniform(-0.62, 0.62)
            if node:
                step = rng.uniform(major * 0.15, major * 0.38)
                node_lon += math.cos(heading) * step / max(0.28, math.cos(math.radians(node_lat)))
                node_lat = clamp(node_lat + math.sin(heading) * step, -86.0, 86.0)
            dlon = ((lon_grid - node_lon + 180.0) % 360.0 - 180.0) * cos_lat
            dlat = lat_grid - node_lat
            along = dlon * math.cos(angle) + dlat * math.sin(angle)
            across = -dlon * math.sin(angle) + dlat * math.cos(angle)
            lobe = np.exp(-((along / max(3.0, major * rng.uniform(0.42, 0.95))) ** 2 + (across / max(3.0, minor * rng.uniform(0.45, 1.15))) ** 2))
            score += lobe * rng.uniform(0.62, 1.08)
            domain_field = np.maximum(domain_field, lobe.astype(np.float32))

    # Add rift/terrane texture as detail only; it should wrinkle continent
    # outlines and interiors without punching random inland seas everywhere.
    broad = np_rng.normal(0.0, 1.0, size=(height, width)).astype(np.float32)
    broad = ndimage.gaussian_filter(broad, sigma=max(2.0, width / 80.0), mode="wrap")
    mid = np_rng.normal(0.0, 1.0, size=(height, width)).astype(np.float32)
    mid = ndimage.gaussian_filter(mid, sigma=max(1.0, width / 260.0), mode="wrap")
    for arr in (broad, mid):
        std = float(np.std(arr))
        if std > 1.0e-6:
            arr -= float(np.mean(arr)); arr /= std
    score += 0.16 * broad + 0.09 * mid + 0.10 * roughness * mid

    threshold = float(np.quantile(score, clamp(1.0 - requested_land, 0.04, 0.96)))
    seed_land = score >= threshold
    structure = np.array([[0,1,0],[1,1,1],[0,1,0]], dtype=np.uint8)
    seed_land = ndimage.binary_closing(seed_land, structure=structure, iterations=3)
    seed_land = ndimage.binary_fill_holes(seed_land)
    seed_land = ndimage.binary_opening(seed_land, structure=structure, iterations=1)

    # Preserve coherent continent domains and a controlled number of terranes.
    labels, count = ndimage.label(seed_land, structure=structure)
    if count:
        sizes = np.bincount(labels.ravel()); sizes[0] = 0
        keep = sizes >= max(8, int(seed_land.size * 0.00011))
        seed_land = keep[labels]

    ocean = ~seed_land
    land_distance = ndimage.distance_transform_edt(seed_land).astype(np.float32)
    ocean_distance = ndimage.distance_transform_edt(ocean).astype(np.float32)
    land_base = 18.0 + 110.0 * np.clip(land_distance / max(1.0, width / 42.0), 0.0, 1.0) + 420.0 * domain_field + 46.0 * mid
    ocean_base = -80.0 - 850.0 * np.clip(ocean_distance / max(2.0, width / 36.0), 0.0, 1.0) - 2300.0 * np.clip(ocean_distance / max(4.0, width / 13.0), 0.0, 1.0) ** 1.2
    elevation = np.where(seed_land, land_base, ocean_base).astype(np.float32)
    elevation[seed_land] = np.maximum(elevation[seed_land], 1.0)
    elevation[ocean] = np.minimum(elevation[ocean], -1.0)
    elevation_i = np.rint(np.clip(elevation, -11000, 10000)).astype(np.int32)
    land_values = elevation_i[seed_land]
    ocean_values = elevation_i[ocean]
    ocean_fraction = 1.0 - float(np.mean(seed_land))
    diagnostics = {
        "generation_controls": {
            "terrain_generation_mode": "plate_tectonic_v1",
            "plate_backend_stage": "plate_domain_workspace_v1",
        },
        "terrain_mode": {
            "mode": "plate_tectonic_v1",
            "backend_status": "native_domain_workspace_no_legacy_core",
            "description": "Plate Terrain 10 creates a native domain/continent workspace and no longer runs the legacy terrain backend before plate setup.",
        },
        "plate_domain_workspace": {
            "continent_domain_count": int(continent_count),
            "seed_ocean_fraction": round(float(ocean_fraction), 4),
            "requested_ocean_fraction": round(float(ocean_target), 4),
            "legacy_core_used": False,
        },
    }
    return TerrainMap(
        width=width,
        height=height,
        elevation_m=elevation_i.astype(int).tolist(),
        is_land=seed_land.astype(bool).tolist(),
        min_elevation_m=int(elevation_i.min()),
        max_elevation_m=int(elevation_i.max()),
        mean_land_elevation_m=float(land_values.mean()) if land_values.size else 0.0,
        mean_ocean_depth_m=float(ocean_values.mean()) if ocean_values.size else 0.0,
        ocean_fraction=ocean_fraction,
        land_fraction=1.0 - ocean_fraction,
        source="plate_tectonic_v1 native domain workspace",
        planet_radius_earth=float(planet.radius_earth),
        terrain_diagnostics=diagnostics,
    )


def _generate_plate_tectonic_v1_scaffold(
    rng: random.Random,
    planet: Planet,
    hydrosphere: Hydrosphere,
    geology: GeologyState,
    config: PlanetProfileConfig,
    *,
    output_dir: str | None = None,
) -> TerrainMap:
    """Plate Terrain 11 drainage-ready tectonic landform backend.

    Plate Terrain 11 keeps the native plate-domain foundation and adds drainage-ready valley corridors, inland basins, lake candidates, terrain detail, continent consolidation, graph-aware plate topology repair, and margin-profile shelves/coastal plains.
    """
    terrain = _create_plate_tectonic_v1_workspace_terrain(rng, planet, hydrosphere, geology, config)
    terrain.source = "plate_tectonic_v1 native domain-based plate terrain workspace"

    diagnostics = terrain.terrain_diagnostics if isinstance(terrain.terrain_diagnostics, dict) else {}
    controls = diagnostics.get("generation_controls") if isinstance(diagnostics.get("generation_controls"), dict) else {}
    controls = {
        **controls,
        "terrain_generation_mode": "plate_tectonic_v1",
        "suppress_polar_land": bool(getattr(config, "suppress_polar_land", False)),
    }
    plate_setup = _generate_plate_tectonic_v1_plate_setup(rng, terrain, hydrosphere, geology, controls)

    # Plate Terrain 10: replace the old diagnostic plate/province IDs and boundary
    # classes with native plate setup + relative motion diagnostics while
    # retaining the legacy elevation backend for downstream compatibility.
    terrain.tectonic_plate_id = plate_setup["plate_id"]
    terrain.tectonic_province_type = plate_setup["province_type"]
    terrain.tectonic_province_age_x1000 = plate_setup["plate_age_x1000"]
    terrain.plate_tectonic_plate_type = plate_setup["plate_type"]
    terrain.plate_tectonic_continental_crust_x1000 = plate_setup["continental_crust_x1000"]
    terrain.plate_tectonic_craton_core_x1000 = plate_setup["craton_core_x1000"]
    terrain.plate_tectonic_microplate_x1000 = plate_setup["microplate_x1000"]
    terrain.plate_tectonic_velocity_x_x1000 = plate_setup["velocity_x_x1000"]
    terrain.plate_tectonic_velocity_y_x1000 = plate_setup["velocity_y_x1000"]
    terrain.plate_tectonic_speed_x1000 = plate_setup["speed_x1000"]
    terrain.plate_tectonic_convergence_x1000 = plate_setup["convergence_x1000"]
    terrain.plate_tectonic_divergence_x1000 = plate_setup["divergence_x1000"]
    terrain.plate_tectonic_transform_x1000 = plate_setup["transform_x1000"]
    terrain.plate_tectonic_boundary_class = plate_setup["boundary_class"]
    terrain.plate_tectonic_subduction_polarity = plate_setup["subduction_polarity"]
    terrain.plate_tectonic_ocean_floor_class = plate_setup["ocean_floor_class"]
    terrain.plate_tectonic_ocean_crust_age_x1000 = plate_setup["ocean_crust_age_x1000"]
    terrain.plate_tectonic_mid_ocean_ridge_x1000 = plate_setup["mid_ocean_ridge_x1000"]
    terrain.plate_tectonic_trench_x1000 = plate_setup["trench_x1000"]
    terrain.plate_tectonic_fracture_zone_x1000 = plate_setup["fracture_zone_x1000"]
    terrain.plate_tectonic_abyssal_plain_x1000 = plate_setup["abyssal_plain_x1000"]
    terrain.plate_tectonic_seamount_x1000 = plate_setup["seamount_x1000"]
    # In plate_tectonic_v1, expose the native ocean-floor fields through the
    # generic terrain_ocean_floor_* review slots too. Elevation still comes from
    # the legacy compatibility backend; these fields make the plate-derived
    # bathymetry plan visible and reviewable before it fully owns elevation.
    terrain.terrain_ocean_floor_class = plate_setup["ocean_floor_class"]
    terrain.terrain_mid_ocean_ridge_x1000 = plate_setup["mid_ocean_ridge_x1000"]
    terrain.terrain_trench_x1000 = plate_setup["trench_x1000"]
    terrain.terrain_fracture_zone_x1000 = plate_setup["fracture_zone_x1000"]
    terrain.terrain_seamount_x1000 = plate_setup["seamount_x1000"]

    plate_foundation = _apply_plate_tectonic_v1_owned_foundation(rng, terrain, plate_setup, hydrosphere, geology, controls)
    terrain.plate_tectonic_continent_assembly_id = plate_foundation.get("continent_assembly_id")
    terrain.plate_tectonic_plate_topology_problem_class = plate_setup.get("topology_problem_class")

    plate_relief = _apply_plate_tectonic_v1_continental_relief(terrain, plate_setup, geology, controls)
    terrain.plate_tectonic_orogeny_strength_x1000 = plate_relief["orogeny_strength_x1000"]
    terrain.plate_tectonic_volcanic_arc_x1000 = plate_relief["volcanic_arc_x1000"]
    terrain.plate_tectonic_continental_rift_x1000 = plate_relief["continental_rift_x1000"]
    terrain.plate_tectonic_foreland_basin_x1000 = plate_relief["foreland_basin_x1000"]
    terrain.plate_tectonic_craton_shield_x1000 = plate_relief["craton_shield_x1000"]
    terrain.plate_tectonic_accreted_terrane_x1000 = plate_relief["accreted_terrane_x1000"]
    terrain.plate_tectonic_plateau_uplift_x1000 = plate_relief["plateau_uplift_x1000"]
    terrain.plate_tectonic_sedimentary_plain_x1000 = plate_relief.get("sedimentary_plain_x1000")
    terrain.plate_tectonic_landform_class = plate_relief.get("landform_class")
    terrain.plate_tectonic_relief_delta_m = plate_relief["relief_delta_m"]
    terrain.terrain_mountain_strength_x1000 = plate_relief["combined_mountain_strength_x1000"]
    terrain.terrain_rift_field_x1000 = plate_relief["continental_rift_x1000"]
    terrain.terrain_basin_field_x1000 = plate_relief["foreland_basin_x1000"]
    terrain.terrain_shield_highland_x1000 = plate_relief["craton_shield_x1000"]
    terrain.terrain_plateau_x1000 = plate_relief["plateau_uplift_x1000"]
    terrain.terrain_interior_relief_x1000 = plate_relief["interior_relief_x1000"]

    plate_coasts = _apply_plate_tectonic_v1_coasts_shelves_islands(terrain, plate_setup, geology, controls)
    terrain.plate_tectonic_margin_class = plate_coasts["margin_class"]
    terrain.plate_tectonic_shelf_width_x1000 = plate_coasts["shelf_width_x1000"]
    terrain.plate_tectonic_active_margin_x1000 = plate_coasts["active_margin_x1000"]
    terrain.plate_tectonic_passive_margin_x1000 = plate_coasts["passive_margin_x1000"]
    terrain.plate_tectonic_rifted_margin_x1000 = plate_coasts["rifted_margin_x1000"]
    terrain.plate_tectonic_island_arc_x1000 = plate_coasts["island_arc_x1000"]
    terrain.plate_tectonic_coastal_plain_x1000 = plate_coasts["coastal_plain_x1000"]
    terrain.plate_tectonic_coast_ruggedness_x1000 = plate_coasts["coast_ruggedness_x1000"]
    terrain.plate_tectonic_island_origin_class = plate_coasts["island_origin_class"]
    terrain.plate_tectonic_coast_delta_m = plate_coasts["coast_delta_m"]
    terrain.plate_tectonic_margin_profile_class = plate_coasts.get("margin_profile_class")
    terrain.terrain_coast_style_class = plate_coasts["coast_style_class"]
    terrain.terrain_shelf_width_x1000 = plate_coasts["shelf_width_x1000"]
    terrain.terrain_coast_ruggedness_x1000 = plate_coasts["coast_ruggedness_x1000"]
    terrain.terrain_island_origin_class = plate_coasts["island_origin_class"]
    terrain.crust_type = _build_plate_tectonic_v1_crust_diagnostic(terrain, plate_coasts)

    plate_drainage = _apply_plate_tectonic_v1_drainage_ready_landforms(rng, terrain, plate_setup, plate_relief, plate_coasts, geology, controls)
    terrain.plate_tectonic_valley_corridor_x1000 = plate_drainage["valley_corridor_x1000"]
    terrain.plate_tectonic_inland_basin_x1000 = plate_drainage["inland_basin_x1000"]
    terrain.plate_tectonic_lake_candidate_x1000 = plate_drainage["lake_candidate_x1000"]
    terrain.plate_tectonic_terrain_detail_x1000 = plate_drainage["terrain_detail_x1000"]
    terrain.plate_tectonic_drainage_ready_delta_m = plate_drainage["drainage_ready_delta_m"]
    terrain.terrain_basin_field_x1000 = _combine_x1000_fields(terrain.terrain_basin_field_x1000, plate_drainage["inland_basin_x1000"])
    terrain.terrain_interior_relief_x1000 = _combine_x1000_fields(terrain.terrain_interior_relief_x1000, plate_drainage["terrain_detail_x1000"])

    # Plate Terrain 13: user-feedback corrective integration.  This pass is
    # intentionally stronger than the previous diagnostic-style refinements: it
    # rewrites the visible mask/elevation where the generated world still shows
    # shelf halos, island-world fragmentation, concentric continental ramps, weak
    # landforms, and too few lake basins.
    plate_corrective = _apply_plate_tectonic_v1_user_feedback_correction_u13(
        rng,
        terrain,
        hydrosphere,
        plate_setup,
        plate_relief,
        plate_coasts,
        plate_drainage,
        geology,
        controls,
    )
    terrain.plate_tectonic_valley_corridor_x1000 = _combine_x1000_fields(terrain.plate_tectonic_valley_corridor_x1000, plate_corrective.get("valley_corridor_x1000"))
    terrain.plate_tectonic_inland_basin_x1000 = _combine_x1000_fields(terrain.plate_tectonic_inland_basin_x1000, plate_corrective.get("inland_basin_x1000"))
    terrain.plate_tectonic_lake_candidate_x1000 = _combine_x1000_fields(terrain.plate_tectonic_lake_candidate_x1000, plate_corrective.get("lake_candidate_x1000"))
    terrain.plate_tectonic_terrain_detail_x1000 = _combine_x1000_fields(terrain.plate_tectonic_terrain_detail_x1000, plate_corrective.get("terrain_detail_x1000"))
    terrain.terrain_mountain_strength_x1000 = _combine_x1000_fields(terrain.terrain_mountain_strength_x1000, plate_corrective.get("mountain_strength_x1000"))
    terrain.terrain_plateau_x1000 = _combine_x1000_fields(terrain.terrain_plateau_x1000, plate_corrective.get("plateau_x1000"))
    terrain.terrain_rift_field_x1000 = _combine_x1000_fields(terrain.terrain_rift_field_x1000, plate_corrective.get("rift_x1000"))
    terrain.terrain_basin_field_x1000 = _combine_x1000_fields(terrain.terrain_basin_field_x1000, plate_corrective.get("inland_basin_x1000"))
    terrain.terrain_interior_relief_x1000 = _combine_x1000_fields(terrain.terrain_interior_relief_x1000, plate_corrective.get("terrain_detail_x1000"))
    terrain.terrain_shelf_width_x1000 = plate_corrective.get("shelf_width_x1000", terrain.terrain_shelf_width_x1000)
    terrain.plate_tectonic_shelf_width_x1000 = plate_corrective.get("shelf_width_x1000", terrain.plate_tectonic_shelf_width_x1000)

    # Plate Terrain 14: rebalance the strong U13 correction. U13 fixed continent
    # scale but removed too many islands/shelves and still lacked branching
    # on-land features. This pass restores controlled islands and variable shelves,
    # adds legacy-style structural land detail, and optionally suppresses polar land.
    plate_balance = _apply_plate_tectonic_v1_feature_balance_u14(
        rng,
        terrain,
        hydrosphere,
        plate_setup,
        plate_relief,
        plate_coasts,
        plate_drainage,
        geology,
        controls,
    )
    terrain.plate_tectonic_valley_corridor_x1000 = _combine_x1000_fields(terrain.plate_tectonic_valley_corridor_x1000, plate_balance.get("valley_corridor_x1000"))
    terrain.plate_tectonic_inland_basin_x1000 = _combine_x1000_fields(terrain.plate_tectonic_inland_basin_x1000, plate_balance.get("inland_basin_x1000"))
    terrain.plate_tectonic_lake_candidate_x1000 = _combine_x1000_fields(terrain.plate_tectonic_lake_candidate_x1000, plate_balance.get("lake_candidate_x1000"))
    terrain.plate_tectonic_terrain_detail_x1000 = _combine_x1000_fields(terrain.plate_tectonic_terrain_detail_x1000, plate_balance.get("terrain_detail_x1000"))
    terrain.terrain_mountain_strength_x1000 = _combine_x1000_fields(terrain.terrain_mountain_strength_x1000, plate_balance.get("mountain_strength_x1000"))
    terrain.terrain_plateau_x1000 = _combine_x1000_fields(terrain.terrain_plateau_x1000, plate_balance.get("plateau_x1000"))
    terrain.terrain_rift_field_x1000 = _combine_x1000_fields(terrain.terrain_rift_field_x1000, plate_balance.get("rift_x1000"))
    terrain.terrain_basin_field_x1000 = _combine_x1000_fields(terrain.terrain_basin_field_x1000, plate_balance.get("inland_basin_x1000"))
    terrain.terrain_interior_relief_x1000 = _combine_x1000_fields(terrain.terrain_interior_relief_x1000, plate_balance.get("terrain_detail_x1000"))
    terrain.terrain_shelf_width_x1000 = plate_balance.get("shelf_width_x1000", terrain.terrain_shelf_width_x1000)
    terrain.plate_tectonic_shelf_width_x1000 = plate_balance.get("shelf_width_x1000", terrain.plate_tectonic_shelf_width_x1000)
    if plate_balance.get("island_origin_class") is not None:
        terrain.plate_tectonic_island_origin_class = plate_balance.get("island_origin_class")
        terrain.terrain_island_origin_class = plate_balance.get("island_origin_class")


    # Plate Terrain 15: structural crust model.  This is not a visual island
    # sprinkle; it classifies continental/oceanic crust first, removes unsupported
    # decorative islands, then lets microcontinents, volcanic arcs, shelves,
    # mountain systems, plateaus, rifts, and valleys follow that crust layer.
    plate_crust_model = _apply_plate_tectonic_v1_crust_model_u15(
        rng,
        terrain,
        hydrosphere,
        plate_setup,
        plate_relief,
        plate_coasts,
        plate_drainage,
        geology,
        controls,
    )
    terrain.plate_tectonic_valley_corridor_x1000 = _combine_x1000_fields(terrain.plate_tectonic_valley_corridor_x1000, plate_crust_model.get("valley_corridor_x1000"))
    terrain.plate_tectonic_inland_basin_x1000 = _combine_x1000_fields(terrain.plate_tectonic_inland_basin_x1000, plate_crust_model.get("inland_basin_x1000"))
    terrain.plate_tectonic_lake_candidate_x1000 = _combine_x1000_fields(terrain.plate_tectonic_lake_candidate_x1000, plate_crust_model.get("lake_candidate_x1000"))
    terrain.plate_tectonic_terrain_detail_x1000 = _combine_x1000_fields(terrain.plate_tectonic_terrain_detail_x1000, plate_crust_model.get("terrain_detail_x1000"))
    terrain.terrain_mountain_strength_x1000 = _combine_x1000_fields(terrain.terrain_mountain_strength_x1000, plate_crust_model.get("mountain_strength_x1000"))
    terrain.terrain_plateau_x1000 = _combine_x1000_fields(terrain.terrain_plateau_x1000, plate_crust_model.get("plateau_x1000"))
    terrain.terrain_rift_field_x1000 = _combine_x1000_fields(terrain.terrain_rift_field_x1000, plate_crust_model.get("rift_x1000"))
    terrain.terrain_basin_field_x1000 = _combine_x1000_fields(terrain.terrain_basin_field_x1000, plate_crust_model.get("inland_basin_x1000"))
    terrain.terrain_interior_relief_x1000 = _combine_x1000_fields(terrain.terrain_interior_relief_x1000, plate_crust_model.get("terrain_detail_x1000"))
    terrain.terrain_shelf_width_x1000 = plate_crust_model.get("shelf_width_x1000", terrain.terrain_shelf_width_x1000)
    terrain.plate_tectonic_shelf_width_x1000 = plate_crust_model.get("shelf_width_x1000", terrain.plate_tectonic_shelf_width_x1000)
    if plate_crust_model.get("island_origin_class") is not None:
        terrain.plate_tectonic_island_origin_class = plate_crust_model.get("island_origin_class")
        terrain.terrain_island_origin_class = plate_crust_model.get("island_origin_class")
    if plate_crust_model.get("crust_class") is not None:
        terrain.crust_type = plate_crust_model.get("crust_class")
        terrain.plate_tectonic_crust_class = plate_crust_model.get("crust_class")


    # Plate Terrain 16: user-feedback cleanup.  This pass addresses the observed
    # U15 artifacts directly: uniform shelf bands and island halos, coastal cliff
    # jumps, polar land distortion, thin high snake islands, missing broken
    # volcanic island arcs, weak branching relief, missing plateaus, huge lake
    # candidates, and long straight coastline diagnostics.
    plate_cleanup = _apply_plate_tectonic_v1_feedback_cleanup_u16(
        rng,
        terrain,
        plate_setup,
        plate_relief,
        plate_coasts,
        geology,
        controls,
    )
    terrain.terrain_mountain_strength_x1000 = _combine_x1000_fields(terrain.terrain_mountain_strength_x1000, plate_cleanup.get("mountain_strength_x1000"))
    terrain.terrain_plateau_x1000 = _combine_x1000_fields(terrain.terrain_plateau_x1000, plate_cleanup.get("plateau_x1000"))
    terrain.plate_tectonic_terrain_detail_x1000 = _combine_x1000_fields(terrain.plate_tectonic_terrain_detail_x1000, plate_cleanup.get("terrain_detail_x1000"))
    if plate_cleanup.get("lake_candidate_x1000") is not None:
        terrain.plate_tectonic_lake_candidate_x1000 = plate_cleanup.get("lake_candidate_x1000")
    terrain.terrain_shelf_width_x1000 = plate_cleanup.get("shelf_width_x1000", terrain.terrain_shelf_width_x1000)
    terrain.plate_tectonic_shelf_width_x1000 = plate_cleanup.get("shelf_width_x1000", terrain.plate_tectonic_shelf_width_x1000)
    if plate_cleanup.get("island_origin_class") is not None:
        terrain.plate_tectonic_island_origin_class = plate_cleanup.get("island_origin_class")
        terrain.terrain_island_origin_class = plate_cleanup.get("island_origin_class")
    if plate_cleanup.get("crust_class") is not None:
        terrain.crust_type = plate_cleanup.get("crust_class")
        terrain.plate_tectonic_crust_class = plate_cleanup.get("crust_class")

    plate_final = _build_plate_tectonic_v1_final_integration_qa(terrain, plate_setup, plate_relief, plate_coasts, geology, controls)
    terrain.plate_tectonic_backend_integration_x1000 = plate_final["backend_integration_x1000"]
    terrain.plate_tectonic_hydrology_readiness_x1000 = plate_final["hydrology_readiness_x1000"]
    terrain.plate_tectonic_legacy_dependency_x1000 = plate_final["legacy_dependency_x1000"]
    terrain.plate_tectonic_problem_class = plate_final["problem_class"]

    terrain.tectonic_boundary_class = plate_setup["boundary_class"]
    terrain.tectonic_boundary_strength_x1000 = plate_setup["boundary_strength_x1000"]
    terrain.tectonic_boundary_width_x1000 = plate_setup["boundary_width_x1000"]

    controls = {**controls, "terrain_generation_mode": "plate_tectonic_v1", "plate_backend_stage": "plate_tectonic_drainage_ready_landforms_v1"}
    diagnostics["generation_controls"] = controls
    diagnostics["terrain_mode"] = {
        "mode": "plate_tectonic_v1",
        "backend_status": "plate_tectonic_drainage_ready_landforms_no_legacy_core_v1",
        "compatibility_backend": "native_plate_domain_workspace_no_legacy_core",
        "description": "Plate Terrain 11 creates native synthetic plate IDs with topology repair, continent/domain assemblies, motion vectors, relative-motion boundary classes, ocean-floor diagnostics, plate-owned foundation/mask/base elevation, tectonic landform belts, margin-profile shelves/coastal plains, drainage-ready valley/basin/lake-candidate fields, and final plate-mode integration/readiness QA. Legacy terrain is not used as a terrain backend in plate mode.",
        "gplates_role": "Design reference and future import/export target; not a required runtime dependency.",
        "completed_true_plate_substages": [
            "plate seeding and crust allocation",
            "plate motion vectors / Euler-like rotation",
            "relative-motion boundary classification",
            "ocean crust age, ridges, trenches, abyssal plains",
            "continental collision, rifts, volcanic arcs, foreland basins, craton/shield relief",
            "continent assemblies, sedimentary plains, inland plains, and tectonic landform classes",
            "margin-profile shelves/coastal plains, islands, and coast-style fields",
            "structural continental/oceanic crust-class layer with microcontinents and tectonic island arcs",
            "drainage-ready valley corridors, inland basins, lake candidates, and terrain detail fields",
            "final plate-mode integration QA and hydrology readiness fields",
            "feedback cleanup for variable shelves, polar land suppression, cliffs, arcs, plateaus, 1m diagnostics, and coastline artifacts",
        ],
        "planned_true_plate_substages": [
            "hydrology consumption of plate readiness fields",
        ],
    }
    plate_meta = dict(plate_setup["metadata"])
    plate_meta["plate_owned_foundation"] = plate_foundation["metadata"]
    plate_meta["continental_relief"] = plate_relief["metadata"]
    plate_meta["coasts_shelves_islands"] = plate_coasts["metadata"]
    plate_meta["drainage_ready_landforms"] = plate_drainage["metadata"]
    plate_meta["user_feedback_correction_u13"] = plate_corrective.get("metadata", {})
    plate_meta["feature_balance_u14"] = plate_balance.get("metadata", {})
    plate_meta["crust_model_u15"] = plate_crust_model.get("metadata", {})
    plate_meta["feedback_cleanup_u16"] = plate_cleanup.get("metadata", {})
    plate_meta["final_integration_qa"] = plate_final["metadata"]
    diagnostics["plate_tectonic_v1"] = plate_meta
    diagnostics["mode_transition_warnings"] = [
        {
            "level": "info",
            "message": "plate_tectonic_v1 now uses a native plate-domain foundation, graph-aware topology repair, continent assemblies, tectonic landform belts, margin-profile shelves/coastal plains, and final integration/readiness diagnostics; the legacy terrain backend is not run in plate mode.",
        }
    ]
    terrain.terrain_diagnostics = diagnostics
    return terrain




def _apply_plate_tectonic_v1_owned_foundation(
    rng: random.Random,
    terrain: TerrainMap,
    plate_setup: dict,
    hydrosphere: Hydrosphere,
    geology: GeologyState,
    controls: dict,
) -> dict:
    """Replace the legacy visible foundation with a native plate-owned one.

    Plate Terrain 10 keeps the legacy backend available for fallback texture, but
    plate_tectonic_v1 now decides the broad land/ocean mask, island scale mix,
    base continental elevation, and base bathymetry from native plate crust,
    boundary, and ocean-floor fields. Later relief/coast passes refine this base.
    """
    try:
        import numpy as np
        from PIL import Image
        from scipy import ndimage
    except ImportError as exc:
        raise RuntimeError("NumPy, Pillow, and SciPy are required for plate terrain diagnostics. Install with: pip install -r requirements.txt") from exc

    old_elev = np.asarray(terrain.elevation_m, dtype=np.float32)
    old_land = np.asarray(terrain.is_land, dtype=bool)
    height, width = old_elev.shape
    np_rng = np.random.default_rng(int(rng.randrange(1, 2**31 - 1)))
    island_density = clamp(float(controls.get("island_density", 0.45) or 0.45), 0.0, 1.0)
    fragmentation = clamp(float(controls.get("fragmentation_tendency", 0.50) or 0.50), 0.0, 1.0)

    def _full_float(name: str, default: float = 0.0):
        value = plate_setup.get(name)
        if value is None:
            return np.full((height, width), float(default), dtype=np.float32)
        arr = np.asarray(value, dtype=np.float32)
        if arr.size == 0:
            return np.full((height, width), float(default), dtype=np.float32)
        arr = arr / 1000.0
        if arr.shape != (height, width):
            img = Image.fromarray(arr.astype(np.float32), mode="F")
            arr = np.asarray(img.resize((width, height), Image.Resampling.BICUBIC), dtype=np.float32)
        return np.nan_to_num(arr, nan=default, posinf=default, neginf=default).astype(np.float32)

    def _full_class(name: str, default: int = 0):
        value = plate_setup.get(name)
        if value is None:
            return np.full((height, width), int(default), dtype=np.int32)
        arr = np.asarray(value, dtype=np.int32)
        if arr.size == 0:
            return np.full((height, width), int(default), dtype=np.int32)
        if arr.shape != (height, width):
            img = Image.fromarray(arr.astype(np.int16), mode="I;16")
            arr = np.asarray(img.resize((width, height), Image.Resampling.NEAREST), dtype=np.int32)
        return arr.astype(np.int32)

    continental = np.clip(_full_float("continental_crust_x1000"), 0.0, 1.0)
    craton = np.clip(_full_float("craton_core_x1000"), 0.0, 1.0)
    microplate = np.clip(_full_float("microplate_x1000"), 0.0, 1.0)
    convergence = np.clip(_full_float("convergence_x1000"), 0.0, 1.0)
    divergence = np.clip(_full_float("divergence_x1000"), 0.0, 1.0)
    transform = np.clip(_full_float("transform_x1000"), 0.0, 1.0)
    ridge = np.clip(_full_float("mid_ocean_ridge_x1000"), 0.0, 1.0)
    trench = np.clip(_full_float("trench_x1000"), 0.0, 1.0)
    fracture = np.clip(_full_float("fracture_zone_x1000"), 0.0, 1.0)
    abyssal = np.clip(_full_float("abyssal_plain_x1000"), 0.0, 1.0)
    seamount = np.clip(_full_float("seamount_x1000"), 0.0, 1.0)
    ocean_age = np.clip(_full_float("ocean_crust_age_x1000"), 0.0, 1.0)
    boundary_class = _full_class("boundary_class")

    # Broad plate-owned foundation potential. Legacy land is only a weak texture
    # hint so plate mode does not visually clone procedural_legacy.
    coarse_noise = np_rng.normal(0.0, 1.0, size=(height, width)).astype(np.float32)
    coarse_noise = ndimage.gaussian_filter(coarse_noise, sigma=max(2.0, width / 120.0), mode="wrap")
    fine_noise = np_rng.normal(0.0, 1.0, size=(height, width)).astype(np.float32)
    fine_noise = ndimage.gaussian_filter(fine_noise, sigma=max(0.9, width / 620.0), mode="wrap")
    def _norm_noise(arr):
        std = float(np.std(arr))
        if std > 1.0e-6:
            return (arr - float(np.mean(arr))) / std
        return np.zeros_like(arr, dtype=np.float32)
    coarse_noise = _norm_noise(coarse_noise)
    fine_noise = _norm_noise(fine_noise)

    island_arc_seed = np.clip((boundary_class == 6).astype(np.float32) * (0.35 + 0.85 * convergence) + 0.55 * seamount + 0.28 * microplate, 0.0, 1.0)
    arc_chain = ndimage.gaussian_filter(island_arc_seed, sigma=max(0.75, width / 950.0), mode="wrap")
    if float(arc_chain.max()) > 1.0e-6:
        arc_chain = np.clip(arc_chain / max(1.0e-6, float(np.percentile(arc_chain[arc_chain > 0], 96))), 0.0, 1.0)

    continent_potential = (
        0.86 * continental
        + 0.30 * craton
        + 0.12 * microplate
        + 0.18 * np.clip(divergence * continental, 0.0, 1.0)
        + 0.12 * old_land.astype(np.float32)
        + 0.15 * coarse_noise
        + 0.055 * fine_noise
    )
    island_potential = (
        0.26 * microplate
        + 0.28 * arc_chain
        + 0.18 * seamount
        + 0.12 * np.maximum(convergence, transform)
        + 0.05 * fine_noise
    )
    potential = np.clip(continent_potential + 0.34 * island_potential, -1.0, 2.5)

    ocean_target = clamp(float(getattr(hydrosphere, "ocean_fraction_target", terrain.ocean_fraction) or terrain.ocean_fraction), 0.08, 0.94)
    current_land = float(np.mean(old_land)) if old_land.size else 0.30
    requested_land = clamp(1.0 - ocean_target, 0.045, 0.82)
    # Do not hard-force the target; plate mode gets close, then reports any miss.
    effective_land_target = clamp(0.90 * requested_land + 0.10 * current_land, 0.06, 0.84)
    threshold = float(np.quantile(potential, clamp(1.0 - effective_land_target, 0.02, 0.98)))
    land = potential >= threshold

    # Ensure strong cratons/terranes survive thresholding, and isolated deep-ocean
    # noise does not create too many tiny islets.
    land |= (continental > 0.78) & (potential > threshold - 0.14)
    land |= (microplate > 0.72) & (arc_chain > 0.38) & (potential > threshold - 0.03)
    land |= (seamount > 0.88) & (arc_chain > 0.46) & (potential > threshold - 0.01)
    land &= ~((trench > 0.55) & (continental < 0.34))

    # Domain-based morphological cleanup.  Plate mode should produce coherent
    # continents first; rifts, lakes, and inland seas are later processes.  The
    # cleanup therefore preserves tectonically justified island chains but does
    # not allow threshold noise to pock-mark every continent interior.
    structure = np.array([[0,1,0],[1,1,1],[0,1,0]], dtype=np.uint8)
    land = ndimage.binary_closing(land, structure=structure, iterations=3)
    filled_land = ndimage.binary_fill_holes(land)
    holes = filled_land & (~land)
    if np.any(holes):
        hole_labels, hole_count = ndimage.label(holes, structure=structure)
        hole_sizes = np.bincount(hole_labels.ravel()) if hole_count else np.array([], dtype=np.int64)
        if hole_count:
            hole_sizes[0] = 0
            for lid in range(1, hole_count + 1):
                comp = hole_labels == lid
                tectonic_opening = float(np.mean(0.70 * divergence[comp] + 0.42 * trench[comp] + 0.30 * arc_chain[comp])) if np.any(comp) else 0.0
                scale_share = float(hole_sizes[lid]) / max(1.0, float(land.size))
                # Keep large/rift-supported inland seas; fill the many small
                # accidental holes that made plate mode look like Swiss cheese.
                if tectonic_opening < 0.38 and scale_share < max(0.0009, 0.018 * effective_land_target):
                    land[comp] = True
    land = ndimage.binary_opening(land, structure=structure, iterations=1)
    labels, count = ndimage.label(land, structure=structure)
    if count:
        sizes = np.bincount(labels.ravel())
        sizes[0] = 0
        component_score = np.zeros_like(sizes, dtype=np.float32)
        for lid in range(1, count + 1):
            comp = labels == int(lid)
            if np.any(comp):
                component_score[lid] = float(np.mean(0.52 * continental[comp] + 0.34 * craton[comp] + 0.45 * microplate[comp] + 0.52 * arc_chain[comp] + 0.30 * seamount[comp]))
        area_share = sizes / max(1.0, float(land.size))
        # Smooth size/score filter: continent and microcontinent components are
        # kept by scale and tectonic support rather than a single brightline.
        keep_strength = np.clip(area_share / max(0.00012, 0.00105 * (0.7 + island_density)) + component_score * 1.12, 0.0, 2.0)
        keep = keep_strength > 0.82
        if np.any(sizes > 0):
            largest = int(np.argmax(sizes))
            if largest > 0:
                keep[largest] = True
        # Preserve a bounded number of arc/hotspot islands as genuine tectonic
        # islands, not an unconstrained islet swarm.
        small_ids = np.where((sizes > 0) & (~keep))[0]
        if small_ids.size:
            ranked = sorted(((float(component_score[lid]), int(lid)) for lid in small_ids), reverse=True)
            budget = max(2, min(14, int(2 + 7 * island_density + width / 420)))
            for score_v, lid in ranked[:budget]:
                if score_v > 0.66:
                    keep[lid] = True
        land = keep[labels]

    # Re-target gently after cleanup if morphology removed too much land.
    actual_land = float(np.mean(land)) if land.size else 0.0
    if actual_land < effective_land_target * 0.82:
        threshold2 = float(np.quantile(potential, clamp(1.0 - effective_land_target * 0.98, 0.02, 0.98)))
        land |= potential >= threshold2
        land = ndimage.binary_closing(land, structure=structure, iterations=2)
    elif actual_land > effective_land_target * 1.28:
        threshold2 = float(np.quantile(potential, clamp(1.0 - effective_land_target * 1.06, 0.02, 0.98)))
        land &= potential >= threshold2

    # Plate Terrain 12: roughen true coastlines at the mask level.  This creates
    # bays, capes, gulfs, and peninsulas without relying on low-elevation shelf
    # strips that later show up as halos.  Reclose afterwards so the change adds
    # coastline diversity rather than renewed global fragmentation.
    try:
        land = _roughen_structural_coastlines(
            rng,
            land.astype(bool),
            potential.astype(np.float32),
            threshold,
            np_rng,
            strength=0.18 + 0.18 * fragmentation + 0.08 * island_density,
        )
        land = ndimage.binary_closing(land, structure=structure, iterations=2)
        land = ndimage.binary_fill_holes(land)
    except Exception:
        # Coast roughening is cosmetic/structural; do not fail terrain generation
        # if an unusual numeric edge case appears.
        pass

    # Plate Terrain 10: consolidate land into continent assemblies before any
    # elevation profile is derived.  This is not a hard supercontinent rule; it is
    # a graph/domain cleanup that treats high continental crust, craton support,
    # and component scale as the organizing units, then keeps separate tectonic
    # island chains where their arc/microplate support is real.
    labels2, count2 = ndimage.label(land, structure=structure)
    continent_assembly = np.zeros_like(labels2, dtype=np.int32)
    topology_problem = np.zeros_like(labels2, dtype=np.uint8)
    if count2:
        sizes2 = np.bincount(labels2.ravel()); sizes2[0] = 0
        total = max(1.0, float(land.size))
        component_records = []
        for lid in range(1, count2 + 1):
            comp = labels2 == lid
            if not np.any(comp):
                continue
            comp_area = float(sizes2[lid]) / total
            crust_support = float(np.mean(continental[comp]))
            craton_support = float(np.mean(craton[comp]))
            arc_support = float(np.mean(arc_chain[comp]))
            micro_support = float(np.mean(microplate[comp]))
            scale_support = np.clip(comp_area / max(0.0007, effective_land_target * 0.020), 0.0, 1.0)
            continent_score = 0.50 * crust_support + 0.24 * craton_support + 0.22 * scale_support - 0.18 * arc_support
            island_score = 0.45 * arc_support + 0.30 * micro_support + 0.22 * crust_support
            component_records.append((continent_score, island_score, comp_area, lid))
        component_records.sort(reverse=True)
        assembly_id = 1
        for continent_score, island_score, comp_area, lid in component_records:
            comp = labels2 == lid
            # Smoothly distinguish continent assemblies from island/terrane belts.
            is_continent = continent_score >= max(0.24, island_score * 0.72) or comp_area > max(0.006, effective_land_target * 0.040)
            if is_continent:
                continent_assembly[comp] = assembly_id; assembly_id += 1
                # Fill non-tectonic internal holes around major continental assemblies.
                envelope = ndimage.binary_closing(comp, structure=structure, iterations=2)
                new_interior = envelope & (~land) & (continental > 0.34) & (divergence < 0.52)
                land[new_interior] = True
                continent_assembly[new_interior] = assembly_id - 1
            else:
                # Keep supported island/terrane chains but mark them as separate
                # from continent assemblies. They should not become shelf carriers.
                continent_assembly[comp] = 1000 + assembly_id; assembly_id += 1
                topology_problem[comp & (arc_chain < 0.18) & (microplate < 0.22)] = 2
        labels2, count2 = ndimage.label(land, structure=structure)
        # Rebuild assembly ids for newly filled continent interiors while keeping
        # IDs compact enough for diagnostics.
        if count2:
            old_assembly = continent_assembly.copy()
            continent_assembly = np.zeros_like(labels2, dtype=np.int32)
            for lid in range(1, count2 + 1):
                comp = labels2 == lid
                vals = old_assembly[comp]
                vals = vals[vals > 0]
                if vals.size:
                    continent_assembly[comp] = int(np.bincount(vals.astype(np.int32)).argmax())
                else:
                    continent_assembly[comp] = lid

    ocean = ~land
    coast_zone = _land_ocean_transition_zone(land, radius=max(2, min(12, width // 330)))
    inland_distance = ndimage.distance_transform_edt(land).astype(np.float32)
    ocean_distance = ndimage.distance_transform_edt(ocean).astype(np.float32)
    coast_land_factor = np.exp(-inland_distance / max(3.0, width / 190.0)).astype(np.float32) * land.astype(np.float32)

    # Base plate-owned elevation. Later plate relief and coast passes modify it.
    rough = np.clip(0.55 * coarse_noise + 0.22 * fine_noise, -2.2, 2.2)
    # Base elevation is no longer simply a distance-to-coast ramp. Continental
    # assemblies receive broad but asymmetric interiors so later tectonic belts,
    # basins, and plains can break the old concentric pattern.
    assembly_noise = np_rng.normal(0.0, 1.0, size=(height, width)).astype(np.float32)
    assembly_noise = ndimage.gaussian_filter(assembly_noise, sigma=max(3.0, width / 95.0), mode="wrap")
    if float(np.std(assembly_noise)) > 1.0e-6:
        assembly_noise = (assembly_noise - float(np.mean(assembly_noise))) / float(np.std(assembly_noise))
    land_elev = (
        85.0
        + 560.0 * continental
        + 330.0 * craton
        + 180.0 * microplate
        + 145.0 * np.clip(divergence * continental, 0.0, 1.0)
        + 155.0 * rough
        + 120.0 * assembly_noise
    )
    # Plate-owned coasts must begin as low margins rather than as high plateaus
    # that later passes try to repair.  This keeps ordinary coast cells in a
    # coastal-hill/lowland range before active margins or collision belts add
    # tectonic relief.
    initial_coast_target = 12.0 + 120.0 * continental + 55.0 * microplate + 42.0 * rough
    land_elev = land_elev * (1.0 - 0.64 * coast_land_factor) + initial_coast_target * (0.64 * coast_land_factor)
    land_elev = np.maximum(land_elev, 1.0 + 18.0 * np.clip(coast_land_factor + 0.18 * fine_noise, 0.0, 1.0))

    ocean_elev = (
        -3900.0
        - 620.0 * ocean_age
        - 1280.0 * trench
        + 940.0 * ridge
        + 380.0 * fracture
        + 620.0 * seamount
        + 240.0 * (1.0 - abyssal)
    )
    # Keep oceanic features smooth, not blocky diagnostic chunks.
    ocean_elev = ndimage.gaussian_filter(ocean_elev, sigma=max(0.65, width / 1150.0), mode="wrap")
    # A first-order coastal bathymetry ramp is applied before explicit shelf
    # handling.  This avoids immediate -4000m water beside ordinary coastlines;
    # active trenches can deepen again in the margin pass.
    ocean_coast_factor = np.exp(-ocean_distance / max(2.2, width / 260.0)).astype(np.float32) * ocean.astype(np.float32)
    shallow_target = -45.0 - 580.0 * np.clip(ocean_distance / max(2.0, width / 145.0), 0.0, 1.0) ** 1.45
    shallow_target -= 520.0 * np.clip(trench + 0.35 * convergence, 0.0, 1.0)
    ocean_elev = np.where(ocean_coast_factor > 0.015, np.maximum(ocean_elev, shallow_target), ocean_elev)
    ocean_elev = np.minimum(ocean_elev, -18.0)

    new_elev = np.where(land, land_elev, ocean_elev).astype(np.float32)
    # Prevent accidental shelves/positive bathymetry before the explicit shelf pass.
    new_elev[land] = np.maximum(new_elev[land], 1.0)
    new_elev[ocean] = np.minimum(new_elev[ocean], -8.0)
    # Long-term plate foundation should expose continent assembly diagnostics.
    new_elev_i = np.rint(np.clip(new_elev, -11000, 10000)).astype(np.int32)

    terrain.elevation_m = new_elev_i.astype(int).tolist()
    terrain.is_land = land.astype(bool).tolist()
    terrain.land_fraction = float(np.mean(land))
    terrain.ocean_fraction = 1.0 - terrain.land_fraction
    terrain.min_elevation_m = int(new_elev_i.min())
    terrain.max_elevation_m = int(new_elev_i.max())
    terrain.mean_land_elevation_m = float(np.mean(new_elev_i[land])) if np.any(land) else 0.0
    terrain.mean_ocean_depth_m = float(np.mean(new_elev_i[ocean])) if np.any(ocean) else 0.0

    delta = new_elev_i.astype(np.float32) - old_elev
    meta = {
        "applied": True,
        "stage": "plate-tectonic-v1-owned-foundation",
        "backend_status": "native plate-owned foundation/mask/elevation base",
        "requested_ocean_fraction": round(float(ocean_target), 4),
        "effective_land_target": round(float(effective_land_target), 4),
        "legacy_land_fraction_before": round(float(current_land), 4),
        "plate_land_fraction_after": round(float(terrain.land_fraction), 4),
        "plate_ocean_fraction_after": round(float(terrain.ocean_fraction), 4),
        "mean_abs_foundation_delta_m": round(float(np.mean(np.abs(delta))), 2),
        "legacy_overlap_land_share": round(float(np.mean(land & old_land)) / max(1.0e-6, float(np.mean(land))) if np.any(land) else 0.0, 4),
        "domain_cleanup_model": "component scale plus tectonic support",
        "legacy_core_used": False,
        "notes": [
            "Plate Terrain 10 starts plate_tectonic_v1 from a native domain workspace instead of the legacy terrain backend.",
            "Continental interiors are stabilized before rifts/inland seas so threshold noise no longer pock-marks every landmass.",
        ],
    }
    return {"metadata": meta, "continent_assembly_id": continent_assembly.astype(int).tolist()}


def _apply_plate_tectonic_v1_continental_relief(
    terrain: TerrainMap,
    plate_setup: dict,
    geology: GeologyState,
    controls: dict,
) -> dict:
    """Apply the first plate-derived continental relief layer.

    Plate Terrain 10 now owns the broad land/ocean
    foundation, coastline cleanup, and downstream compatibility.  This pass is
    the first native plate stage that modifies elevation: convergence/collision
    uplifts mountains, subduction builds volcanic arcs, divergence cuts rifts,
    convergence shadows create foreland basins, cratons become broad shields,
    and microplates/accreted terranes become broken uplands.
    """
    try:
        import numpy as np
        from PIL import Image
        from scipy import ndimage
    except ImportError as exc:
        raise RuntimeError("NumPy, Pillow, and SciPy are required for plate terrain diagnostics. Install with: pip install -r requirements.txt") from exc

    elev = np.asarray(terrain.elevation_m, dtype=np.float32)
    land = np.asarray(terrain.is_land, dtype=bool)
    height, width = elev.shape

    def _arr(name: str, dtype=np.float32, scale: float = 1000.0):
        value = plate_setup.get(name)
        if value is None:
            return None
        arr = np.asarray(value, dtype=dtype)
        if arr.size == 0:
            return None
        if dtype is np.float32 or dtype == np.float32:
            return np.clip(arr / scale, -1.0, 1.0)
        return arr

    continental = _arr("continental_crust_x1000")
    craton = _arr("craton_core_x1000")
    microplate = _arr("microplate_x1000")
    convergence = _arr("convergence_x1000")
    divergence = _arr("divergence_x1000")
    transform = _arr("transform_x1000")
    trench = _arr("trench_x1000")
    seamount = _arr("seamount_x1000")
    boundary_class = np.asarray(plate_setup.get("boundary_class"), dtype=np.int32)
    subduction = np.asarray(plate_setup.get("subduction_polarity"), dtype=np.int32)
    plate_type = np.asarray(plate_setup.get("plate_type"), dtype=np.int32)

    if continental is None or convergence is None or divergence is None or boundary_class.size == 0:
        zeros = np.zeros((max(1, height // 4), max(1, width // 4)), dtype=np.float32)
        zero_i = (zeros * 1000).astype(int).tolist()
        return {
            "orogeny_strength_x1000": zero_i,
            "volcanic_arc_x1000": zero_i,
            "continental_rift_x1000": zero_i,
            "foreland_basin_x1000": zero_i,
            "craton_shield_x1000": zero_i,
            "accreted_terrane_x1000": zero_i,
            "plateau_uplift_x1000": zero_i,
            "combined_mountain_strength_x1000": zero_i,
            "interior_relief_x1000": zero_i,
            "relief_delta_m": [[0 for _ in range(width)] for __ in range(height)],
            "metadata": {"applied": False, "reason": "missing plate setup fields"},
        }

    diag_h, diag_w = continental.shape
    land_low = np.asarray(
        Image.fromarray(land.astype(np.uint8) * 255, mode="L").resize((diag_w, diag_h), Image.Resampling.BICUBIC),
        dtype=np.float32,
    ) / 255.0
    land_like = np.clip(land_low, 0.0, 1.0)
    continental = np.clip(continental, 0.0, 1.0)
    craton = np.clip(craton if craton is not None else np.zeros_like(continental), 0.0, 1.0)
    microplate = np.clip(microplate if microplate is not None else np.zeros_like(continental), 0.0, 1.0)
    convergence = np.clip(convergence, 0.0, 1.0)
    divergence = np.clip(divergence, 0.0, 1.0)
    transform = np.clip(transform if transform is not None else np.zeros_like(continental), 0.0, 1.0)
    trench = np.clip(trench if trench is not None else np.zeros_like(continental), 0.0, 1.0)
    seamount = np.clip(seamount if seamount is not None else np.zeros_like(continental), 0.0, 1.0)

    diag_scale = max(0.70, math.sqrt(float(diag_w * diag_h)) / 420.0)
    active_continent = np.clip(0.58 * land_like + 0.42 * continental, 0.0, 1.0)
    convergent = np.isin(boundary_class, [1, 6]).astype(np.float32)
    divergent = (boundary_class == 2).astype(np.float32)
    transform_mask = (boundary_class == 3).astype(np.float32)
    collision_polarity = (subduction == 3).astype(np.float32)
    oc_subduction = np.isin(subduction, [1, 2]).astype(np.float32)

    # Collision/orogeny: strongest where continental crust participates in
    # convergence or explicit continent-continent collision exists.
    orogeny_seed = convergence * (0.50 * convergent + 0.95 * collision_polarity) * np.clip(0.25 + 0.95 * active_continent, 0.0, 1.0)
    orogeny = ndimage.gaussian_filter(orogeny_seed, sigma=max(0.65, 1.15 * diag_scale), mode="wrap")

    # Volcanic arcs: oceanic subduction and active convergent margins adjacent
    # to land/microplates; keep them narrower and more broken than collision belts.
    arc_seed = (0.72 * oc_subduction + 0.28 * (boundary_class == 6).astype(np.float32)) * np.maximum(convergence, trench) * np.clip(0.18 + land_like + 0.55 * microplate, 0.0, 1.0)
    volcanic_arc = ndimage.gaussian_filter(arc_seed, sigma=max(0.55, 0.85 * diag_scale), mode="wrap")

    # Continental rifts: divergent boundaries through continental or mixed crust.
    rift_seed = divergence * (0.58 * divergent + 0.22) * np.clip(0.22 + 0.98 * continental + 0.24 * land_like, 0.0, 1.0)
    continental_rift = ndimage.gaussian_filter(rift_seed, sigma=max(0.60, 0.95 * diag_scale), mode="wrap")
    rift_shoulder = np.clip(ndimage.gaussian_filter(continental_rift, sigma=max(1.20, 2.2 * diag_scale), mode="wrap") - 0.55 * continental_rift, 0.0, 1.0)

    # Foreland basins lie adjacent to strong young mountain belts, not directly
    # on top of the range.
    mountain_envelope = ndimage.gaussian_filter(np.maximum(orogeny, volcanic_arc * 0.75), sigma=max(1.2, 2.1 * diag_scale), mode="wrap")
    foreland_basin = np.clip((mountain_envelope - 0.55 * np.maximum(orogeny, volcanic_arc)) * active_continent * (1.0 - 0.48 * craton), 0.0, 1.0)

    # Stable cratons create low, broad shield/highland relief; terranes create
    # broken coastal uplands and microplate-collision relief.
    craton_shield = np.clip(ndimage.gaussian_filter(craton * land_like, sigma=max(1.0, 2.0 * diag_scale), mode="wrap") * (1.0 - 0.35 * convergence), 0.0, 1.0)
    accreted_terrane = np.clip(ndimage.gaussian_filter(microplate * (0.35 + 0.65 * land_like) * (0.45 + 0.55 * (convergence + transform)), sigma=max(0.55, 0.85 * diag_scale), mode="wrap"), 0.0, 1.0)
    transform_uplift = np.clip(ndimage.gaussian_filter(transform * transform_mask * active_continent, sigma=max(0.65, 0.90 * diag_scale), mode="wrap"), 0.0, 1.0)
    plateau_uplift = np.clip(0.36 * craton_shield + 0.28 * rift_shoulder + 0.22 * transform_uplift + 0.18 * accreted_terrane, 0.0, 1.0)

    # Plate Terrain 10: continent-scale sedimentary/inland plains are generated
    # as landforms in their own right, not just low coastal rims.  They form in
    # foreland shadows, craton margins, rift-adjacent accommodation zones, and
    # low-convergence interiors. This prepares plausible Gangetic-style plains,
    # interior basins, and future lake/river corridors.
    low_convergence_interior = np.clip((1.0 - convergence) * active_continent * (0.35 + 0.65 * (boundary_class == 0).astype(np.float32)), 0.0, 1.0)
    interior_lowland_seed = np.clip((
        0.68 * foreland_basin
        + 0.36 * continental_rift
        + 0.20 * low_convergence_interior * (1.0 - 0.55 * craton_shield)
    ) * active_continent * (1.0 - 0.72 * np.maximum(orogeny, volcanic_arc)), 0.0, 1.0)
    sedimentary_plain = ndimage.gaussian_filter(interior_lowland_seed, sigma=max(1.2, 2.2 * diag_scale), mode="wrap")
    # Avoid converting every stable interior into a plain. The field is normalized
    # only against its stronger positive tail so plains stay as belts/basins.
    positive_plain = sedimentary_plain[sedimentary_plain > 0.02]
    if positive_plain.size:
        sedimentary_plain = np.clip((sedimentary_plain - float(np.percentile(positive_plain, 38))) / max(1.0e-6, float(np.percentile(positive_plain, 94) - np.percentile(positive_plain, 38))), 0.0, 1.0)
    sedimentary_plain = np.clip(sedimentary_plain, 0.0, 1.0)

    def _norm(arr):
        arr = np.clip(arr, 0.0, None).astype(np.float32)
        positive = arr[arr > 0]
        if positive.size:
            arr = arr / max(1.0e-6, float(np.percentile(positive, 96)))
        return np.clip(arr, 0.0, 1.0)

    orogeny = _norm(orogeny)
    volcanic_arc = _norm(volcanic_arc)
    continental_rift = _norm(continental_rift)
    foreland_basin = _norm(foreland_basin)
    craton_shield = _norm(craton_shield)
    accreted_terrane = _norm(accreted_terrane)
    plateau_uplift = _norm(plateau_uplift)
    rift_shoulder = _norm(rift_shoulder)
    sedimentary_plain = _norm(sedimentary_plain)

    heat = clamp(float(getattr(geology, "internal_heat", 0.75) or 0.75), 0.0, 2.0) / 2.0
    volcanism = clamp(float(getattr(geology, "volcanism", 0.55) or 0.55), 0.0, 1.8) / 1.8
    mountain_control = clamp(float(controls.get("mountain_belt_strength", 0.55) or 0.55), 0.0, 1.0)
    rift_control = clamp(float(controls.get("rift_strength", 0.45) or 0.45), 0.0, 1.0)
    basin_control = clamp(float(controls.get("basin_strength", controls.get("deposition_strength", 0.45)) or 0.45), 0.0, 1.0)
    relief_gain = 0.55 + 0.75 * mountain_control + 0.24 * heat

    def _up(arr, *, order=1):
        image = Image.fromarray(np.clip(arr, 0.0, 1.0).astype(np.float32), mode="F")
        return np.asarray(image.resize((width, height), Image.Resampling.BICUBIC if order else Image.Resampling.NEAREST), dtype=np.float32)

    orogeny_f = _up(orogeny)
    arc_f = _up(volcanic_arc)
    rift_f = _up(continental_rift)
    basin_f = _up(foreland_basin)
    shield_f = _up(craton_shield)
    terrane_f = _up(accreted_terrane)
    plateau_f = _up(plateau_uplift)
    sedimentary_plain_f = _up(sedimentary_plain)
    rift_shoulder_f = _up(rift_shoulder)
    transform_f = _up(transform_uplift)

    raw_delta = (
        1450.0 * relief_gain * orogeny_f
        + 980.0 * (0.70 + 0.55 * volcanism) * arc_f
        + 680.0 * terrane_f
        + 520.0 * shield_f
        + 620.0 * plateau_f
        + 520.0 * (0.55 + rift_control) * rift_shoulder_f
        + 300.0 * transform_f
        - 620.0 * (0.45 + rift_control) * rift_f
        - 560.0 * (0.55 + basin_control) * basin_f
        - 420.0 * (0.55 + basin_control) * sedimentary_plain_f
    )
    # Keep the first native relief update meaningful but bounded. Later plate
    # terrain updates can own the entire elevation synthesis; for now we avoid
    # breaking downstream climate/hydrology by limiting extreme deltas.
    raw_delta = np.clip(raw_delta, -1250.0, 2850.0) * land.astype(np.float32)
    # Smooth only a little so ridge/arc fields remain visible but do not form
    # hard diagnostic lines through islands or continents. Landform diversity now
    # comes from belts/basins/plains rather than coast distance.
    raw_delta = ndimage.gaussian_filter(raw_delta, sigma=0.55, mode="wrap") * land.astype(np.float32)

    new_elev = elev + raw_delta
    # Maintain land/ocean sign without re-fitting ocean fraction.
    if np.any(land):
        new_elev[land] = np.maximum(new_elev[land], 1.0)
    new_elev[~land] = elev[~land]
    new_elev_i = np.rint(np.clip(new_elev, -11000, 10000)).astype(np.int32)
    delta_i = np.rint(new_elev_i.astype(np.float32) - elev).astype(np.int32)

    terrain.elevation_m = new_elev_i.astype(int).tolist()
    if np.any(land):
        terrain.max_elevation_m = int(new_elev_i.max())
        terrain.mean_land_elevation_m = float(np.mean(new_elev_i[land]))
    if np.any(~land):
        terrain.min_elevation_m = int(new_elev_i.min())
        terrain.mean_ocean_depth_m = float(np.mean(new_elev_i[~land]))

    def _x1000(arr):
        return np.rint(np.clip(arr, 0.0, 1.0) * 1000.0).astype(np.int16).astype(int).tolist()

    combined_mountain = np.clip(np.maximum.reduce([orogeny, volcanic_arc * 0.85, accreted_terrane * 0.70, plateau_uplift * 0.55]), 0.0, 1.0)
    interior_relief = np.clip(np.maximum(craton_shield * 0.75, plateau_uplift * 0.85) + 0.25 * rift_shoulder + 0.18 * sedimentary_plain, 0.0, 1.0)
    # Diagnostic landform class: 0 water/background, 1 shield/craton, 2 orogen,
    # 3 volcanic arc, 4 rift valley, 5 foreland/sedimentary plain, 6 plateau,
    # 7 accreted terrane/transform upland.
    landform_class = np.zeros_like(orogeny, dtype=np.uint8)
    landform_class[land_like > 0.20] = 1
    landform_class[(sedimentary_plain > 0.32) & (land_like > 0.20)] = 5
    landform_class[(plateau_uplift > 0.38) & (land_like > 0.20)] = 6
    landform_class[(continental_rift > 0.34) & (land_like > 0.20)] = 4
    landform_class[(accreted_terrane > 0.36) & (land_like > 0.20)] = 7
    landform_class[(volcanic_arc > 0.35) & (land_like > 0.20)] = 3
    landform_class[(orogeny > 0.36) & (land_like > 0.20)] = 2
    meta = {
        "applied": True,
        "stage": "plate-tectonic-v1-continental-relief",
        "backend_status": "native plate continental relief applied to plate-owned foundation/mask",
        "mean_orogeny_strength": round(float(np.mean(orogeny)), 4),
        "strong_orogeny_share": round(float(np.mean(orogeny > 0.55)), 4),
        "mean_volcanic_arc_strength": round(float(np.mean(volcanic_arc)), 4),
        "mean_continental_rift_strength": round(float(np.mean(continental_rift)), 4),
        "mean_foreland_basin_strength": round(float(np.mean(foreland_basin)), 4),
        "mean_craton_shield_strength": round(float(np.mean(craton_shield)), 4),
        "mean_accreted_terrane_strength": round(float(np.mean(accreted_terrane)), 4),
        "mean_plateau_uplift_strength": round(float(np.mean(plateau_uplift)), 4),
        "mean_sedimentary_plain_strength": round(float(np.mean(sedimentary_plain)), 4),
        "sedimentary_plain_land_share": round(float(np.mean((_up(sedimentary_plain) > 0.30)[land])) if np.any(land) else 0.0, 4),
        "mean_land_relief_delta_m": round(float(np.mean(delta_i[land])) if np.any(land) else 0.0, 2),
        "mean_abs_land_relief_delta_m": round(float(np.mean(np.abs(delta_i[land]))) if np.any(land) else 0.0, 2),
        "strong_uplift_land_share": round(float(np.mean(delta_i[land] > 350)) if np.any(land) else 0.0, 4),
        "strong_subsidence_land_share": round(float(np.mean(delta_i[land] < -180)) if np.any(land) else 0.0, 4),
        "notes": [
            "Plate Terrain 10 treats mountains, rifts, foreland basins, sedimentary plains, plateaus, shields, and terranes as landform systems rather than a concentric coast-to-interior height ramp.",
            "Sedimentary/inland plains are now explicit pre-hydrology terrain structures for later river valleys, through-flow lakes, and endorheic basins.",
        ],
    }
    return {
        "orogeny_strength_x1000": _x1000(orogeny),
        "volcanic_arc_x1000": _x1000(volcanic_arc),
        "continental_rift_x1000": _x1000(continental_rift),
        "foreland_basin_x1000": _x1000(foreland_basin),
        "craton_shield_x1000": _x1000(craton_shield),
        "accreted_terrane_x1000": _x1000(accreted_terrane),
        "plateau_uplift_x1000": _x1000(plateau_uplift),
        "sedimentary_plain_x1000": _x1000(sedimentary_plain),
        "landform_class": landform_class.astype(int).tolist(),
        "combined_mountain_strength_x1000": _x1000(combined_mountain),
        "interior_relief_x1000": _x1000(interior_relief),
        "relief_delta_m": delta_i.astype(int).tolist(),
        "metadata": meta,
    }


def _apply_plate_tectonic_v1_coasts_shelves_islands(
    terrain: TerrainMap,
    plate_setup: dict,
    geology: GeologyState,
    controls: dict,
) -> dict:
    """Apply Plate Terrain 10 coast/shelf/island interpretation.

    Plate mode now owns the land/ocean foundation before this pass. This layer
    converts native plate margins into variable-width shelves, active-margin
    deep water, coastal plains, and island-origin diagnostics without reverting
    to distance halos around every island.
    """
    try:
        import numpy as np
        from PIL import Image
        from scipy import ndimage
    except ImportError as exc:
        raise RuntimeError("NumPy, Pillow, and SciPy are required for plate terrain diagnostics. Install with: pip install -r requirements.txt") from exc

    elev = np.asarray(terrain.elevation_m, dtype=np.float32)
    land = np.asarray(terrain.is_land, dtype=bool)
    height, width = elev.shape
    ocean = ~land

    def _diag_float(name: str):
        value = plate_setup.get(name)
        if value is None:
            return None
        arr = np.asarray(value, dtype=np.float32)
        if arr.size == 0:
            return None
        return np.clip(arr / 1000.0, -1.0, 1.0)

    def _diag_class(name: str):
        value = plate_setup.get(name)
        if value is None:
            return None
        arr = np.asarray(value, dtype=np.int32)
        return arr if arr.size else None

    def _up_float(arr, default=0.0):
        if arr is None or not getattr(arr, "size", 0):
            return np.full((height, width), float(default), dtype=np.float32)
        img = Image.fromarray(np.asarray(arr, dtype=np.float32), mode="F")
        return np.asarray(img.resize((width, height), Image.Resampling.BICUBIC), dtype=np.float32)

    def _up_class(arr, default=0):
        if arr is None or not getattr(arr, "size", 0):
            return np.full((height, width), int(default), dtype=np.int32)
        img = Image.fromarray(np.asarray(arr, dtype=np.int32).astype(np.int16), mode="I;16")
        return np.asarray(img.resize((width, height), Image.Resampling.NEAREST), dtype=np.int32)

    continental = np.clip(_up_float(_diag_float("continental_crust_x1000")), 0.0, 1.0)
    microplate = np.clip(_up_float(_diag_float("microplate_x1000")), 0.0, 1.0)
    convergence = np.clip(_up_float(_diag_float("convergence_x1000")), 0.0, 1.0)
    divergence = np.clip(_up_float(_diag_float("divergence_x1000")), 0.0, 1.0)
    transform = np.clip(_up_float(_diag_float("transform_x1000")), 0.0, 1.0)
    trench = np.clip(_up_float(_diag_float("trench_x1000")), 0.0, 1.0)
    volcanic_arc_relief = getattr(terrain, "plate_tectonic_volcanic_arc_x1000", None)
    if volcanic_arc_relief is not None:
        volcanic_arc = np.clip(np.asarray(volcanic_arc_relief, dtype=np.float32), 0.0, 1000.0) / 1000.0
        if volcanic_arc.shape != elev.shape:
            volcanic_arc = _up_float(volcanic_arc)
    else:
        volcanic_arc = np.zeros_like(elev, dtype=np.float32)
    boundary_class = _up_class(_diag_class("boundary_class"))
    subduction = _up_class(_diag_class("subduction_polarity"))

    # Coasts and nearby offshore zones.
    north = np.vstack((land[0:1, :], land[:-1, :]))
    south = np.vstack((land[1:, :], land[-1:, :]))
    west = np.roll(land, 1, axis=1)
    east = np.roll(land, -1, axis=1)
    coast = land & ((~north) | (~south) | (~west) | (~east))
    coast_buffer = _land_ocean_transition_zone(land, radius=max(2, min(16, width // 280)))

    # Continent-scale crust mask for margin ownership. Small volcanic islands
    # are not shelf carriers, so they no longer grow continental shelf halos.
    # Shelf carriers must be true continent-scale land components.  Earlier
    # plate revisions still let small microcontinents/islands carry shelves,
    # which produced huge circular shelf halos around islands.
    continent_carrier = land & (continental > 0.50)
    continent_carrier &= _continent_scale_land_mask(continent_carrier | (land & (continental > 0.66)), min_fraction=0.012)
    if int(continent_carrier.sum()) < max(12, int(land.size * 0.010)):
        continent_carrier = _continent_scale_land_mask(land & (continental > 0.44), min_fraction=0.018)
    if not continent_carrier.any():
        continent_carrier = _continent_scale_land_mask(land, min_fraction=0.025)
    land_dist = ndimage.distance_transform_edt(land).astype(np.float32)
    ocean_dist_to_land = ndimage.distance_transform_edt(ocean).astype(np.float32)

    active_seed = np.clip(0.50 * convergence + 0.30 * trench + 0.25 * volcanic_arc + 0.18 * transform, 0.0, 1.0)
    passive_seed = np.clip((continental * (1.0 - np.clip(active_seed + 0.55 * divergence, 0.0, 1.0))) * (0.35 + 0.65 * (land_dist < max(10, width // 90))), 0.0, 1.0)
    rift_seed = np.clip(divergence * (0.36 + 0.64 * continental), 0.0, 1.0)
    transform_seed = np.clip(transform * (0.35 + 0.65 * (continental + microplate)), 0.0, 1.0)
    arc_seed = np.clip(np.maximum(volcanic_arc, (subduction > 0).astype(np.float32) * convergence) * (0.38 + 0.62 * (microplate + 0.35 * coast_buffer)), 0.0, 1.0)

    # Margin influence fields. Keep them broad enough to be visible, but tied to
    # plate margins rather than generic coast distance.
    margin_sigma = max(1.0, min(6.5, width / 360.0))
    active_margin = np.clip(ndimage.gaussian_filter(active_seed * coast_buffer.astype(np.float32), sigma=margin_sigma, mode="wrap"), 0.0, 1.0)
    passive_margin = np.clip(ndimage.gaussian_filter(passive_seed * coast_buffer.astype(np.float32), sigma=margin_sigma * 1.25, mode="wrap"), 0.0, 1.0)
    rifted_margin = np.clip(ndimage.gaussian_filter(rift_seed * coast_buffer.astype(np.float32), sigma=margin_sigma, mode="wrap"), 0.0, 1.0)
    island_arc = np.clip(ndimage.gaussian_filter(arc_seed * (coast_buffer.astype(np.float32) + 0.35 * ocean.astype(np.float32)), sigma=max(0.8, margin_sigma * 0.75), mode="wrap"), 0.0, 1.0)
    transform_margin = np.clip(ndimage.gaussian_filter(transform_seed * coast_buffer.astype(np.float32), sigma=margin_sigma, mode="wrap"), 0.0, 1.0)

    shelf_control = clamp(float(controls.get("shelf_width_factor", 0.55) or 0.55), 0.0, 2.0)
    base_shelf_radius = max(2.2, min(34.0, (3.5 + 11.0 * shelf_control) * max(0.45, width / 2048.0)))
    # Plate Terrain 11 shelves are not radial distance halos.  They start only
    # from ocean cells adjacent to continent-scale passive/rifted margin
    # segments, then are gated by local margin support and an ocean-distance
    # profile.  Tiny islands and volcanic arcs therefore do not grow oversized
    # continental shelf aprons.
    ocean_next_to_continent = ocean & (
        np.roll(continent_carrier, 1, axis=1)
        | np.roll(continent_carrier, -1, axis=1)
        | np.vstack((continent_carrier[0:1, :], continent_carrier[:-1, :]))
        | np.vstack((continent_carrier[1:, :], continent_carrier[-1:, :]))
    )
    shelf_seed = ocean_next_to_continent & ((passive_margin > 0.08) | (rifted_margin > 0.10)) & (active_margin < 0.30) & (island_arc < 0.32)
    # Plate Terrain 11 margin profiles: shelves are segment-controlled bands, not
    # radial halos. A shelf cell must be closer to a continent-scale margin than
    # to a non-carrier island, and width varies along the segment.
    if np.any(shelf_seed):
        dist_to_shelf_seed = ndimage.distance_transform_edt(~shelf_seed).astype(np.float32)
        dist_to_continent = ndimage.distance_transform_edt(~continent_carrier).astype(np.float32)
        island_land = land & (~continent_carrier)
        dist_to_island = ndimage.distance_transform_edt(~island_land).astype(np.float32) if np.any(island_land) else np.full_like(dist_to_shelf_seed, 1.0e6)
        yy, xx = np.indices(elev.shape)
        # Plate Terrain 11: shelf profiles vary along margin segments using broad
        # deterministic segment texture. This keeps shelves as geological margin
        # bands instead of equal-width rings around every coast.
        segment_noise = (
            0.62 * np.sin(xx * 0.041 + yy * 0.067)
            + 0.38 * np.sin(xx * 0.097 - yy * 0.035 + np.sin(yy * 0.021))
        ).astype(np.float32)
        segment_noise = ndimage.gaussian_filter(segment_noise, sigma=max(1.1, width / 520.0), mode="wrap")
        if float(np.std(segment_noise)) > 1.0e-6:
            segment_noise = (segment_noise - float(np.mean(segment_noise))) / float(np.std(segment_noise))
        segment_texture = np.clip(0.55 + 0.28 * segment_noise, 0.0, 1.0)
        local_width_factor = np.clip(
            0.32
            + 0.86 * passive_margin
            + 0.46 * rifted_margin
            + 0.30 * segment_texture
            - 0.82 * active_margin
            - 0.76 * island_arc,
            0.05,
            1.72,
        )
        segment_width = np.maximum(0.75, base_shelf_radius * local_width_factor)
        raw_shelf = np.clip(1.0 - dist_to_shelf_seed / segment_width, 0.0, 1.0) ** (1.45 + 0.55 * (1.0 - segment_texture))
        carrier_gate = (dist_to_continent <= dist_to_island + 0.35).astype(np.float32)
        # Suppress smooth continuous shelves in weakly supported segment gaps; this
        # is a continuous support texture rather than a hard cutoff.
        margin_gap_gate = np.clip(0.18 + 0.82 * segment_texture + 0.70 * passive_margin + 0.42 * rifted_margin - 0.88 * active_margin - 0.72 * island_arc, 0.0, 1.0)
        raw_shelf *= carrier_gate * margin_gap_gate * ocean.astype(np.float32)
    else:
        dist_to_continent = ndimage.distance_transform_edt(~continent_carrier).astype(np.float32)
        dist_to_island = np.full_like(dist_to_continent, 1.0e6)
        raw_shelf = np.zeros_like(elev, dtype=np.float32)
    shelf_support = np.clip(1.00 * passive_margin + 0.74 * rifted_margin + 0.03 * continental - 1.30 * active_margin - 1.18 * island_arc, 0.0, 1.0)
    # Plate Terrain 11 intentionally removes the old continent-distance shelf gate
    # that could still create same-width halos.  Width comes from seeded margin
    # segments and support texture; distance-to-continent is now only used to keep
    # non-carrier islands from owning broad shelves.
    shelf_field = np.clip(raw_shelf * shelf_support, 0.0, 1.0)
    shelf_field *= np.clip(0.25 + 1.10 * passive_margin + 0.74 * rifted_margin - 0.82 * active_margin - 0.72 * island_arc, 0.0, 1.0)
    shelf_field[(microplate > 0.30) & (continental < 0.68) & (passive_margin < 0.48)] *= 0.018
    shelf_field[dist_to_island < np.maximum(1.4, dist_to_continent * 1.05)] *= 0.025
    shelf_field[(continent_carrier == 0) & (ocean_dist_to_land > max(1.0, base_shelf_radius * 0.24))] *= 0.10
    shelf_field[shelf_field < 0.045] = 0.0
    # Active margins can have narrow shelves but should not become broad halos.
    shelf_field[active_margin > 0.24] *= 0.035

    # Ruggedness comes from active/subduction/arc/transform margins and local
    # relief. Passive shelf plains stay smoother.
    elev_n = np.vstack((elev[0:1, :], elev[:-1, :]))
    elev_s = np.vstack((elev[1:, :], elev[-1:, :]))
    elev_w = np.roll(elev, 1, axis=1)
    elev_e = np.roll(elev, -1, axis=1)
    relief = np.maximum.reduce([np.abs(elev - elev_n), np.abs(elev - elev_s), np.abs(elev - elev_w), np.abs(elev - elev_e)])
    relief_norm = np.clip(relief / max(1.0, float(np.percentile(relief[coast], 92)) if coast.any() else 550.0), 0.0, 1.0)
    coast_ruggedness = np.clip((0.38 * active_margin + 0.28 * island_arc + 0.18 * transform_margin + 0.18 * relief_norm - 0.16 * passive_margin) * coast_buffer.astype(np.float32), 0.0, 1.0)
    coastal_lowland_band = np.exp(-land_dist / max(3.4, width / 165.0)).astype(np.float32) * land.astype(np.float32)
    # Coastal plains must be actively created on passive/rifted margins, not
    # merely preserved where legacy terrain happened to be low already.
    coastal_plain = np.clip((0.82 * passive_margin + 0.52 * rifted_margin + 0.34 * shelf_field + 0.16 * continental) * (1.0 - 0.78 * active_margin) * coastal_lowland_band, 0.0, 1.0) * land.astype(np.float32)

    margin_class = np.zeros_like(elev, dtype=np.uint8)
    margin_zone = coast_buffer | (shelf_field > 0.08)
    margin_class[margin_zone] = 6  # mixed/transition margin
    margin_class[margin_zone & (passive_margin >= np.maximum.reduce([active_margin, rifted_margin, island_arc, transform_margin]))] = 1
    margin_class[margin_zone & (active_margin > np.maximum.reduce([passive_margin, rifted_margin, transform_margin]))] = 2
    margin_class[margin_zone & (rifted_margin > np.maximum(active_margin, passive_margin))] = 3
    margin_class[margin_zone & (island_arc > np.maximum(passive_margin, rifted_margin)) & (island_arc > 0.12)] = 4
    margin_class[margin_zone & (transform_margin > 0.32) & (transform_margin >= active_margin)] = 5

    # Margin-profile class: 0 background, 1 passive shelf/plain profile,
    # 2 active/trench profile, 3 rifted gulf profile, 4 volcanic/island-arc
    # profile, 5 transform/escarpment profile, 6 mixed.
    margin_profile_class = np.zeros_like(margin_class, dtype=np.uint8)
    margin_profile_class[margin_zone] = 6
    margin_profile_class[margin_zone & (passive_margin >= np.maximum.reduce([active_margin, rifted_margin, island_arc, transform_margin]))] = 1
    margin_profile_class[margin_zone & (active_margin > np.maximum.reduce([passive_margin, rifted_margin, island_arc]))] = 2
    margin_profile_class[margin_zone & (rifted_margin > np.maximum(active_margin, passive_margin))] = 3
    margin_profile_class[margin_zone & (island_arc > np.maximum(passive_margin, rifted_margin)) & (island_arc > 0.12)] = 4
    margin_profile_class[margin_zone & (transform_margin > 0.32) & (transform_margin >= active_margin)] = 5

    coast_style = np.zeros_like(margin_class, dtype=np.uint8)
    coast_style[coast] = 6
    coast_style[coast & (margin_class == 1)] = 1
    coast_style[coast & (margin_class == 2)] = 2
    coast_style[coast & (margin_class == 3)] = 3
    coast_style[coast & (margin_class == 4)] = 4
    coast_style[coast & (shelf_field > 0.22) & (coastal_plain > 0.12)] = 5
    coast_style[coast & (coast_ruggedness > 0.44)] = np.maximum(coast_style[coast & (coast_ruggedness > 0.44)], 2)

    # Classify existing island components by plate context. This intentionally
    # does not create new land yet; Plate Terrain 5 interprets and modestly
    # reshapes/upgrades current islands without breaking the legacy mask.
    island_origin = np.zeros_like(margin_class, dtype=np.uint8)
    try:
        labels, count = ndimage.label(land, structure=np.array([[0,1,0],[1,1,1],[0,1,0]], dtype=np.uint8))
        sizes = np.bincount(labels.ravel()) if count else np.array([], dtype=np.int64)
    except Exception:
        labels = np.zeros_like(margin_class, dtype=np.int32); count = 0; sizes = np.array([], dtype=np.int64)
    if count:
        sizes[0] = 0
        island_limit = max(4, int(land.size * 0.0038))
        small_limit = max(3, int(land.size * 0.00065))
        island_origin[land] = 1
        for label_id in range(1, count + 1):
            size = int(sizes[label_id]) if label_id < len(sizes) else 0
            if size <= 0:
                continue
            comp = labels == label_id
            if size > island_limit:
                island_origin[comp] = 1
                continue
            comp_arc = float(np.mean(island_arc[comp])) if np.any(comp) else 0.0
            comp_micro = float(np.mean(microplate[comp])) if np.any(comp) else 0.0
            comp_shelf = float(np.mean(shelf_field[comp])) if np.any(comp) else 0.0
            comp_rift = float(np.mean(rifted_margin[comp])) if np.any(comp) else 0.0
            high = float(np.max(elev[comp])) if np.any(comp) else 0.0
            if comp_arc > 0.12:
                code = 3  # volcanic/arc island
            elif comp_micro > 0.18 or size > small_limit * 3 or comp_rift > 0.12:
                code = 4  # microcontinent/terrane
            elif high > 850 or float(np.mean(active_margin[comp])) > 0.18:
                code = 5  # hotspot/high island
            elif comp_shelf > 0.10:
                code = 2
            else:
                code = 2
            island_origin[comp] = code

    # Plate Terrain 11 coastal-continuity elevation pass.  This replaces the
    # previous modest coast adjustment with explicit shore profiles.  Ordinary
    # passive/rifted coasts get a smooth land -> coastal plain -> shelf -> slope
    # transition; active/subduction coasts can stay steeper, but even they avoid
    # immediate abyssal water in the first ocean cell unless a trench is present.
    new_elev = elev.copy()

    if np.any(ocean):
        # Plate Terrain 12: only shelf-supported margin segments get broad shallow
        # water.  Earlier versions applied a coast-distance max-depth profile to
        # almost every ocean cell near land; visually this produced continental
        # shelf halos.  Non-shelf coasts now get only a very narrow shore apron.
        supported_shelf = ocean & (shelf_field > 0.035)
        narrow_nearshore = ocean & (ocean_dist_to_land <= 1.25) & (
            (passive_margin > 0.14)
            | (rifted_margin > 0.16)
            | ((active_margin < 0.18) & (dist_to_continent <= dist_to_island + 0.35))
        )
        active_nearshore = ocean & (active_margin > 0.24) & (ocean_dist_to_land <= 0.85)
        near_ocean = supported_shelf | narrow_nearshore | active_nearshore
        active_factor_o = np.clip(active_margin + 0.75 * island_arc + 0.45 * trench, 0.0, 1.0)
        passive_factor_o = np.clip(passive_margin + 0.75 * rifted_margin + 0.70 * shelf_field, 0.0, 1.0)
        profile_width = np.maximum(1.2, base_shelf_radius * (0.44 + 1.26 * passive_factor_o - 0.76 * active_factor_o))
        profile_distance = np.where(supported_shelf, dist_to_continent, ocean_dist_to_land)
        t = np.clip(profile_distance / profile_width, 0.0, 1.0)
        passive_profile = -58.0 - 315.0 * t - 1420.0 * (t ** 1.85)
        active_profile = -150.0 - 690.0 * t - 2100.0 * (t ** 1.45) - 820.0 * trench
        shelf_profile = passive_profile * (1.0 - active_factor_o) + active_profile * active_factor_o
        shelf_profile += 430.0 * shelf_field * (1.0 - 0.82 * active_factor_o)
        new_elev[near_ocean] = np.maximum(new_elev[near_ocean], shelf_profile[near_ocean])
        # Non-carrier island shores get a narrow volcanic apron, not a continental shelf.
        island_near_ocean = ocean & (dist_to_island <= np.minimum(dist_to_continent, max(2.5, base_shelf_radius * 0.40)))
        if np.any(island_near_ocean):
            island_apron = -90.0 - 520.0 * np.clip(dist_to_island / max(1.4, base_shelf_radius * 0.34), 0.0, 1.0) ** 1.25 - 280.0 * island_arc
            new_elev[island_near_ocean] = np.minimum(new_elev[island_near_ocean], np.maximum(island_apron[island_near_ocean], -1050.0))

        active_deep = ocean & (active_margin > 0.20) & (ocean_dist_to_land > 1.8)
        if np.any(active_deep):
            new_elev[active_deep] -= 220.0 * active_margin[active_deep] + 340.0 * trench[active_deep]
        rift_gulf = ocean & (rifted_margin > 0.18)
        if np.any(rift_gulf):
            new_elev[rift_gulf] = np.maximum(new_elev[rift_gulf], -980.0 + 560.0 * rifted_margin[rift_gulf] - 220.0 * np.clip(ocean_dist_to_land[rift_gulf] / max(2.0, base_shelf_radius), 0.0, 1.0))

    if np.any(land):
        active_factor_l = np.clip(active_margin + 0.68 * island_arc + 0.38 * transform_margin, 0.0, 1.0)
        passive_factor_l = np.clip(passive_margin + 0.72 * rifted_margin + 0.38 * coastal_plain, 0.0, 1.0)
        land_profile_width = np.maximum(2.0, base_shelf_radius * (0.85 + 1.75 * passive_factor_l - 0.52 * active_factor_l))
        land_t = np.clip(land_dist / land_profile_width, 0.0, 1.0)
        lowland_profile = 8.0 + 72.0 * land_t + 310.0 * (land_t ** 1.65) + 80.0 * continental + 42.0 * relief_norm
        active_profile_l = 28.0 + 260.0 * land_t + 980.0 * (land_t ** 1.35) + 540.0 * active_factor_l + 360.0 * island_arc
        target_land_profile = lowland_profile * (1.0 - active_factor_l) + active_profile_l * active_factor_l

        coastal_transition = np.clip(np.exp(-land_dist / np.maximum(2.0, land_profile_width)) * (0.40 + 0.72 * passive_factor_l + 0.24 * coastal_plain - 0.32 * active_factor_l), 0.0, 0.86) * land.astype(np.float32)
        # Smooth only within the transition zone to avoid terracing the entire
        # continent.  Then pull passive/rifted coast cells toward the physical
        # profile; active coast cells keep more relief.
        smooth_source = ndimage.gaussian_filter(new_elev, sigma=1.65, mode="wrap")
        new_elev[land] = new_elev[land] * (1.0 - 0.24 * coastal_transition[land]) + smooth_source[land] * (0.24 * coastal_transition[land])
        new_elev[land] = new_elev[land] * (1.0 - coastal_transition[land]) + target_land_profile[land] * coastal_transition[land]

        # Hard cap the worst passive/rifted coastal cliffs.  This intentionally
        # affects only the first few cells from the sea and avoids exact 1m flats.
        immediate_passive = land & (land_dist <= 3.0) & (active_factor_l < 0.48)
        if np.any(immediate_passive):
            cap = 42.0 + 115.0 * land_dist + 190.0 * np.clip(1.0 - passive_factor_l, 0.0, 1.0) + 70.0 * relief_norm
            new_elev[immediate_passive] = np.minimum(new_elev[immediate_passive], cap[immediate_passive])

        # Active margins and volcanic arcs can be high near the sea, but keep
        # them as coastal mountains rather than uniform vertical cliffs.
        active_land = land & (active_factor_l >= 0.48) & (land_dist <= 4.0)
        if np.any(active_land):
            active_cap = 180.0 + 265.0 * land_dist + 920.0 * active_factor_l + 320.0 * relief_norm
            new_elev[active_land] = np.minimum(new_elev[active_land], active_cap[active_land])

        # Add subtle microrelief so newly formed coastal plains are not flat
        # one-meter surfaces.
        micro = ndimage.gaussian_filter(np.sin((np.indices(elev.shape)[1] * 0.37 + np.indices(elev.shape)[0] * 0.19)).astype(np.float32), sigma=0.8, mode="wrap")
        new_elev[land] += (2.5 + 8.0 * coastal_plain[land]) * micro[land] * np.clip(coastal_transition[land] + 0.35 * coastal_plain[land], 0.0, 1.0)
        new_elev[land] = np.maximum(new_elev[land], 1.0 + 7.0 * coastal_plain[land] + 3.0 * relief_norm[land])

    # One final shoreline safety pass, but only where a shelf/passive/rifted
    # margin actually supports shallow water.  Active, volcanic, and non-carrier
    # island coasts may drop away steeply; that removes the universal halo.
    immediate_ocean = ocean & (ocean_dist_to_land <= 1.25) & (
        (shelf_field > 0.035)
        | (passive_margin > 0.18)
        | (rifted_margin > 0.18)
        | ((active_margin > 0.42) & (trench < 0.38))
    )
    if np.any(immediate_ocean):
        immediate_active = np.clip(active_margin + 0.65 * trench + 0.45 * island_arc, 0.0, 1.0)
        nearshore_floor = -82.0 - 320.0 * immediate_active - 210.0 * trench + 130.0 * shelf_field
        new_elev[immediate_ocean] = np.maximum(new_elev[immediate_ocean], nearshore_floor[immediate_ocean])

    new_elev[ocean] = np.minimum(new_elev[ocean], -1.0)
    new_elev_i = np.rint(np.clip(new_elev, -11000, 10000)).astype(np.int32)
    delta_i = np.rint(new_elev_i.astype(np.float32) - elev).astype(np.int32)

    terrain.elevation_m = new_elev_i.astype(int).tolist()
    if np.any(land):
        terrain.max_elevation_m = int(new_elev_i.max())
        terrain.mean_land_elevation_m = float(np.mean(new_elev_i[land]))
    if np.any(ocean):
        terrain.min_elevation_m = int(new_elev_i.min())
        terrain.mean_ocean_depth_m = float(np.mean(new_elev_i[ocean]))

    def _x1000(arr):
        return np.rint(np.clip(arr, 0.0, 1.0) * 1000.0).astype(np.int16).astype(int).tolist()

    meta = {
        "applied": True,
        "stage": "plate-tectonic-v1-coasts-shelves-islands",
        "backend_status": "native plate margins applied to plate-owned foundation/mask",
        "mean_plate_shelf_width": round(float(np.mean(shelf_field[ocean])) if np.any(ocean) else 0.0, 4),
        "margin_profile_model": "segment-controlled shelf bands; island halos suppressed by continent-carrier distance",
        "broad_plate_shelf_ocean_share": round(float(np.mean(shelf_field[ocean] > 0.32)) if np.any(ocean) else 0.0, 4),
        "active_margin_coast_share": round(float(np.mean((active_margin > 0.18)[coast])) if np.any(coast) else 0.0, 4),
        "passive_margin_coast_share": round(float(np.mean((passive_margin > 0.18)[coast])) if np.any(coast) else 0.0, 4),
        "rifted_margin_coast_share": round(float(np.mean((rifted_margin > 0.18)[coast])) if np.any(coast) else 0.0, 4),
        "island_arc_land_share": round(float(np.mean((island_arc > 0.20)[land])) if np.any(land) else 0.0, 4),
        "mean_abs_coast_delta_m": round(float(np.mean(np.abs(delta_i[coast_buffer]))) if np.any(coast_buffer) else 0.0, 2),
        "mean_coast_land_elevation_m": round(float(np.mean(new_elev_i[coast])) if np.any(coast) else 0.0, 2),
        "p95_coast_land_elevation_m": round(float(np.percentile(new_elev_i[coast], 95)) if np.any(coast) else 0.0, 2),
        "mean_adjacent_ocean_depth_m": round(float(np.mean(new_elev_i[ocean & (ocean_dist_to_land <= 1.25)])) if np.any(ocean & (ocean_dist_to_land <= 1.25)) else 0.0, 2),
        "adjacent_ocean_deeper_than_1000m_share": round(float(np.mean(new_elev_i[ocean & (ocean_dist_to_land <= 1.25)] < -1000)) if np.any(ocean & (ocean_dist_to_land <= 1.25)) else 0.0, 4),
        "plate_terrain_stage": "Plate Terrain 10 coastal-continuity repair",
        "notes": [
            "Plate Terrain 10 replaces shelf halos with segment-controlled margin profiles and explicitly suppresses broad continental shelves around non-carrier islands.",
            "Passive/rifted continental segments can form shelves and coastal plains; active and volcanic island margins remain narrow/steep.",
            "Coastal transition is now driven by margin profile class rather than a uniform distance-from-land halo.",
        ],
    }
    return {
        "margin_class": margin_class.astype(int).tolist(),
        "shelf_width_x1000": _x1000(shelf_field),
        "active_margin_x1000": _x1000(active_margin),
        "passive_margin_x1000": _x1000(passive_margin),
        "rifted_margin_x1000": _x1000(rifted_margin),
        "island_arc_x1000": _x1000(island_arc),
        "coastal_plain_x1000": _x1000(coastal_plain),
        "coast_ruggedness_x1000": _x1000(coast_ruggedness),
        "island_origin_class": island_origin.astype(int).tolist(),
        "coast_style_class": coast_style.astype(int).tolist(),
        "margin_profile_class": margin_profile_class.astype(int).tolist(),
        "coast_delta_m": delta_i.astype(int).tolist(),
        "metadata": meta,
    }


def _apply_plate_tectonic_v1_feature_balance_u14(
    rng: random.Random,
    terrain: TerrainMap,
    hydrosphere: Hydrosphere,
    plate_setup: dict,
    plate_relief: dict,
    plate_coasts: dict,
    plate_drainage: dict,
    geology: GeologyState,
    controls: dict,
) -> dict:
    """Plate Terrain 14 balance pass after the aggressive U13 correction.

    U13 intentionally stopped the old island-world / shelf-halo failure, but it
    over-corrected: islands and continental shelves became scarce, and visible
    on-land features still looked too smooth.  This pass adds a controlled island
    budget, physically varied shelves, branching mountain/rift/valley structures,
    broader coastal plains, more plateau edges, and an optional polar-land
    suppression switch for distorted equirectangular polar caps.
    """
    try:
        import numpy as np
        from PIL import Image
        from scipy import ndimage
    except ImportError as exc:
        raise RuntimeError("NumPy, Pillow, and SciPy are required for plate terrain generation. Install with: pip install -r requirements.txt") from exc

    elev0 = np.asarray(terrain.elevation_m, dtype=np.float32)
    land = np.asarray(terrain.is_land, dtype=bool).copy()
    height, width = elev0.shape
    total_cells = max(1, height * width)
    np_rng = np.random.default_rng(int(rng.randrange(1, 2**31 - 1)))
    structure = np.array([[0,1,0],[1,1,1],[0,1,0]], dtype=np.uint8)

    def _full_field(value, *, default: float = 0.0, scale: float = 1000.0):
        if value is None:
            return np.full((height, width), float(default), dtype=np.float32)
        arr = np.asarray(value, dtype=np.float32)
        if arr.size == 0:
            return np.full((height, width), float(default), dtype=np.float32)
        if scale:
            arr = arr / scale
        if arr.shape != (height, width):
            img = Image.fromarray(arr.astype(np.float32), mode="F")
            arr = np.asarray(img.resize((width, height), Image.Resampling.BICUBIC), dtype=np.float32)
        return np.nan_to_num(arr, nan=default, posinf=default, neginf=default).astype(np.float32)

    def _full_class(value, *, default: int = 0):
        if value is None:
            return np.full((height, width), int(default), dtype=np.int32)
        arr = np.asarray(value, dtype=np.int32)
        if arr.size == 0:
            return np.full((height, width), int(default), dtype=np.int32)
        if arr.shape != (height, width):
            img = Image.fromarray(arr.astype(np.int16), mode="I;16")
            arr = np.asarray(img.resize((width, height), Image.Resampling.NEAREST), dtype=np.int32)
        return arr.astype(np.int32)

    def _norm(arr):
        arr = np.nan_to_num(np.asarray(arr, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        positive = arr[arr > 0.0]
        if positive.size:
            lo = float(np.percentile(positive, 8))
            hi = float(np.percentile(positive, 96))
            arr = (arr - lo) / max(1.0e-6, hi - lo)
        return np.clip(arr, 0.0, 1.0)

    def _x1000(arr):
        return np.rint(np.clip(arr, 0.0, 1.0) * 1000.0).astype(np.int16).astype(int).tolist()

    continental = np.clip(_full_field(plate_setup.get("continental_crust_x1000")), 0.0, 1.0)
    craton = np.clip(_full_field(plate_setup.get("craton_core_x1000")), 0.0, 1.0)
    microplate = np.clip(_full_field(plate_setup.get("microplate_x1000")), 0.0, 1.0)
    convergence = np.clip(_full_field(plate_setup.get("convergence_x1000")), 0.0, 1.0)
    divergence = np.clip(_full_field(plate_setup.get("divergence_x1000")), 0.0, 1.0)
    transform = np.clip(_full_field(plate_setup.get("transform_x1000")), 0.0, 1.0)
    trench = np.clip(_full_field(plate_setup.get("trench_x1000")), 0.0, 1.0)
    ridge = np.clip(_full_field(plate_setup.get("mid_ocean_ridge_x1000")), 0.0, 1.0)
    seamount = np.clip(_full_field(plate_setup.get("seamount_x1000")), 0.0, 1.0)
    orogeny0 = np.clip(_full_field(plate_relief.get("orogeny_strength_x1000")), 0.0, 1.0)
    arc0 = np.clip(_full_field(plate_relief.get("volcanic_arc_x1000")), 0.0, 1.0)
    rift0 = np.clip(_full_field(plate_relief.get("continental_rift_x1000")), 0.0, 1.0)
    basin0 = np.clip(_full_field(plate_relief.get("foreland_basin_x1000")), 0.0, 1.0)
    sediment0 = np.clip(_full_field(plate_relief.get("sedimentary_plain_x1000")), 0.0, 1.0)
    plateau0 = np.clip(_full_field(plate_relief.get("plateau_uplift_x1000")), 0.0, 1.0)
    passive0 = np.clip(_full_field(plate_coasts.get("passive_margin_x1000")), 0.0, 1.0)
    rifted0 = np.clip(_full_field(plate_coasts.get("rifted_margin_x1000")), 0.0, 1.0)
    active0 = np.clip(_full_field(plate_coasts.get("active_margin_x1000")), 0.0, 1.0)
    coastal_plain0 = np.clip(_full_field(plate_coasts.get("coastal_plain_x1000")), 0.0, 1.0)
    valley0 = np.clip(_full_field(plate_drainage.get("valley_corridor_x1000")), 0.0, 1.0)
    inland_basin0 = np.clip(_full_field(plate_drainage.get("inland_basin_x1000")), 0.0, 1.0)
    lake0 = np.clip(_full_field(plate_drainage.get("lake_candidate_x1000")), 0.0, 1.0)
    island_origin0 = _full_class(plate_coasts.get("island_origin_class"), default=0)

    lats = np.linspace(90.0 - 90.0 / height, -90.0 + 90.0 / height, height, dtype=np.float32)
    lons = np.linspace(-180.0 + 180.0 / width, 180.0 - 180.0 / width, width, dtype=np.float32)
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    abs_lat = np.abs(lat_grid)
    cos_lat = np.maximum(0.18, np.cos(np.radians(lat_grid))).astype(np.float32)

    suppress_polar = bool(controls.get("suppress_polar_land", False))
    polar_removed = np.zeros_like(land, dtype=bool)
    if suppress_polar:
        # Short-term visual fix: remove polar land cleanly instead of leaving
        # equirectangular-stretched caps.  Use a simple latitude cutoff rather
        # than per-cell random taper, because random taper made a noisy polar
        # fringe that looked worse than a clean temporary ocean cap.
        polar_removed = land & (abs_lat >= 72.0)
        land[polar_removed] = False
        land = ndimage.binary_opening(land, structure=structure, iterations=1)
        land = _remove_one_cell_land_water_needles(land)

    # Restore a controlled island budget.  Islands are generated from arc,
    # seamount, microplate, and hotspot support, but kept away from the broadest
    # continent carriers so they read as islands rather than renewed fragmentation.
    labels_pre, count_pre = ndimage.label(land, structure=structure)
    carrier = np.zeros_like(land, dtype=bool)
    if count_pre:
        sizes = np.bincount(labels_pre.ravel()); sizes[0] = 0
        land_cells = max(1.0, float(sizes.sum()))
        for lid in [int(i) for i in np.argsort(sizes)[::-1] if i > 0 and sizes[i] > 0][:5]:
            if sizes[lid] / land_cells > 0.075 or sizes[lid] / total_cells > 0.006:
                carrier |= labels_pre == lid
        if not carrier.any() and sizes.size > 1:
            carrier |= labels_pre == int(np.argmax(sizes))
    dist_to_carrier_land = ndimage.distance_transform_edt(~carrier).astype(np.float32) if carrier.any() else np.full_like(elev0, width, dtype=np.float32)
    dist_to_any_land_before = ndimage.distance_transform_edt(~land).astype(np.float32)
    ocean_before = ~land
    island_noise = np_rng.normal(0.0, 1.0, size=(height, width)).astype(np.float32)
    island_noise = ndimage.gaussian_filter(island_noise, sigma=max(1.2, width / 380.0), mode="wrap")
    if float(np.std(island_noise)) > 1.0e-6:
        island_noise = (island_noise - float(np.mean(island_noise))) / float(np.std(island_noise))
    island_potential = _norm(
        0.54 * seamount
        + 0.45 * microplate
        + 0.40 * arc0
        + 0.24 * convergence
        + 0.18 * ridge
        + 0.12 * np.maximum(island_noise, 0.0)
        - 0.34 * continental
        - 0.24 * trench
    )
    if suppress_polar:
        island_potential[abs_lat > 68.0] *= 0.05
    island_candidates = np.argwhere(ocean_before & (island_potential > 0.42) & (dist_to_carrier_land > max(3.0, width / 260.0)))
    added_islands = np.zeros_like(land, dtype=bool)
    island_budget = max(6, min(36, int(round(7 + 15 * clamp(float(controls.get("island_density", 0.35) or 0.35), 0.0, 1.0) + width / 190.0))))
    if island_candidates.size:
        # Sample top potential candidates, then grow island chains/lobes around them.
        cand_scores = island_potential[island_candidates[:, 0], island_candidates[:, 1]]
        order = np.argsort(cand_scores)[::-1]
        accepted_centers: list[tuple[int, int]] = []
        for idx in order[: min(len(order), island_budget * 18)]:
            rr, cc = island_candidates[int(idx)]
            if any(abs(int(rr) - ar) < max(5, height // 70) and min(abs(int(cc) - ac), width - abs(int(cc) - ac)) < max(8, width // 90) for ar, ac in accepted_centers):
                continue
            accepted_centers.append((int(rr), int(cc)))
            if len(accepted_centers) >= island_budget:
                break
        for rr, cc in accepted_centers:
            lon0 = float(lons[cc]); lat0 = float(lats[rr])
            angle = rng.uniform(0.0, math.tau)
            chain_len = rng.randint(1, 4 if island_potential[rr, cc] < 0.78 else 6)
            for k in range(chain_len):
                offset = (k - (chain_len - 1) / 2.0) * rng.uniform(1.7, 4.8)
                side = rng.uniform(-1.2, 1.2)
                lonc = lon0 + (offset * math.cos(angle) - side * math.sin(angle)) / max(0.20, math.cos(math.radians(lat0)))
                latc = clamp(lat0 + offset * math.sin(angle) + side * math.cos(angle), -66.0 if suppress_polar else -82.0, 66.0 if suppress_polar else 82.0)
                dlon = _wrapped_lon_delta_array(lon_grid, lonc) * cos_lat
                dlat = lat_grid - latc
                along = dlon * math.cos(angle) + dlat * math.sin(angle)
                across = -dlon * math.sin(angle) + dlat * math.cos(angle)
                major = rng.uniform(0.75, 2.4) * (1.0 + 0.35 * island_potential[rr, cc])
                minor = rng.uniform(0.45, 1.35) * (1.0 + 0.25 * island_potential[rr, cc])
                lobe = np.exp(-((along / max(0.32, major)) ** 2 + (across / max(0.24, minor)) ** 2))
                added_islands |= (lobe > rng.uniform(0.42, 0.58)) & ocean_before & (dist_to_carrier_land > max(2.0, width / 360.0))
    land |= added_islands
    land = _remove_one_cell_land_water_needles(land)

    ocean = ~land
    coast = land & ((~np.roll(land, 1, axis=1)) | (~np.roll(land, -1, axis=1)) | (~np.roll(land, 1, axis=0)) | (~np.roll(land, -1, axis=0)))
    land_dist = ndimage.distance_transform_edt(land).astype(np.float32)
    ocean_dist = ndimage.distance_transform_edt(ocean).astype(np.float32)
    inland = np.clip(land_dist / max(3.0, width / 24.0), 0.0, 1.0) * land.astype(np.float32)

    def _curve_belts(count: int, width_deg_range: tuple[float, float], length_range: tuple[float, float], *, coast_bias: bool = False, branchiness: float = 0.0, seed_field=None):
        field = np.zeros((height, width), dtype=np.float32)
        land_points = np.argwhere(land)
        coast_points = np.argwhere(coast) if np.any(coast) else land_points
        if land_points.size == 0:
            return field
        weighted_points = land_points
        if seed_field is not None:
            sf = np.asarray(seed_field, dtype=np.float32)
            hot = land & (sf > max(0.15, float(np.percentile(sf[land], 65)) if np.any(land) else 0.2))
            hp = np.argwhere(hot)
            if hp.size:
                weighted_points = hp
        for _ in range(max(0, int(count))):
            pts = coast_points if (coast_bias and coast_points.size and rng.random() < 0.68) else weighted_points
            rr, cc = pts[rng.randrange(len(pts))]
            lon0 = float(lons[int(cc)]); lat0 = float(lats[int(rr)])
            angle = rng.uniform(0.0, math.tau)
            belt_width = rng.uniform(*width_deg_range)
            belt_length = rng.uniform(*length_range)

            def add_segment(lonc: float, latc: float, seg_angle: float, seg_length: float, seg_width: float, strength: float):
                nonlocal field
                dlon = _wrapped_lon_delta_array(lon_grid, lonc) * cos_lat
                dlat = lat_grid - latc
                along = dlon * math.cos(seg_angle) + dlat * math.sin(seg_angle)
                across = -dlon * math.sin(seg_angle) + dlat * math.cos(seg_angle)
                wave = np.sin(np.radians(along * rng.uniform(1.6, 4.2) + rng.uniform(-180.0, 180.0))) * rng.uniform(1.0, 4.8)
                wave += np.sin(np.radians(along * rng.uniform(4.5, 8.0) + rng.uniform(-180.0, 180.0))) * rng.uniform(0.20, 1.10)
                dist = across + wave
                belt = np.exp(-((dist / max(0.32, seg_width)) ** 2)) * np.exp(-((along / max(5.0, seg_length)) ** 2))
                beads = 0.62 + 0.38 * np.sin(np.radians(along * rng.uniform(2.2, 7.0) + rng.uniform(-180.0, 180.0)))
                field = np.maximum(field, (strength * belt * np.clip(beads, 0.10, 1.0)).astype(np.float32))

            add_segment(lon0, lat0, angle, belt_length, belt_width, 1.0)
            branch_count = int(round(branchiness * rng.randint(1, 4)))
            for _b in range(branch_count):
                branch_offset = rng.uniform(-0.42, 0.42) * belt_length
                branch_angle = angle + rng.choice([-1.0, 1.0]) * rng.uniform(0.45, 1.15)
                lonb = lon0 + (branch_offset * math.cos(angle)) / max(0.20, math.cos(math.radians(lat0)))
                latb = clamp(lat0 + branch_offset * math.sin(angle), -78.0, 78.0)
                add_segment(lonb, latb, branch_angle, belt_length * rng.uniform(0.22, 0.55), belt_width * rng.uniform(0.45, 0.82), rng.uniform(0.42, 0.78))
        return _norm(field) * land.astype(np.float32)

    mountain_belts = _curve_belts(max(5, min(13, int(round(6 + width / 330)))), (0.75, 2.8), (34.0, 118.0), coast_bias=True, branchiness=1.0, seed_field=np.maximum(orogeny0, convergence))
    rift_belts = _curve_belts(max(3, min(8, int(round(3 + width / 520)))), (0.55, 1.75), (30.0, 112.0), coast_bias=False, branchiness=0.55, seed_field=np.maximum(rift0, divergence))
    river_valley_belts = _curve_belts(max(8, min(24, int(round(8 + width / 135)))), (0.28, 0.95), (18.0, 82.0), coast_bias=False, branchiness=0.85, seed_field=np.maximum(basin0, np.maximum(valley0, mountain_belts)))

    mountain_envelope = ndimage.gaussian_filter(np.maximum(mountain_belts, orogeny0), sigma=max(1.3, width / 410.0), mode="wrap")
    mountain_field = _norm(0.64 * mountain_belts + 0.42 * mountain_envelope + 0.44 * orogeny0 + 0.30 * arc0 + 0.26 * convergence) * land.astype(np.float32)
    rift_field = _norm(0.70 * rift_belts + 0.45 * rift0 + 0.34 * divergence + 0.16 * transform) * land.astype(np.float32)
    foreland_shadow = np.clip(ndimage.gaussian_filter(mountain_field, sigma=max(2.8, width / 58.0), mode="wrap") - 0.58 * mountain_field, 0.0, 1.0)

    broad_noise = np_rng.normal(0.0, 1.0, size=(height, width)).astype(np.float32)
    broad_noise = ndimage.gaussian_filter(broad_noise, sigma=max(3.5, width / 70.0), mode="wrap")
    if float(np.std(broad_noise)) > 1.0e-6:
        broad_noise = (broad_noise - float(np.mean(broad_noise))) / float(np.std(broad_noise))
    plateau_seed = _norm(0.42 * plateau0 + 0.35 * craton + 0.28 * inland + 0.24 * mountain_envelope + 0.18 * np.maximum(broad_noise, 0.0) - 0.32 * rift_field - 0.26 * foreland_shadow) * land.astype(np.float32)
    plateau_region = ndimage.gaussian_filter(plateau_seed, sigma=max(1.8, width / 150.0), mode="wrap")
    if np.any(land):
        pcut = float(np.percentile(plateau_region[land], 62))
        plateau_region = _norm(np.maximum(0.0, plateau_region - pcut)) * land.astype(np.float32)
    plateau_edge = np.clip(plateau_region - ndimage.minimum_filter(plateau_region, size=max(3, min(17, width // 170))), 0.0, 1.0)

    passive_like = np.clip(passive0 + 0.72 * coastal_plain0 + 0.45 * sediment0 + 0.34 * rifted0 - 0.55 * active0 - 0.42 * trench, 0.0, 1.0)
    coastal_plain_field = _norm(np.exp(-land_dist / max(2.6, width / 92.0)) * passive_like * (1.0 - 0.58 * mountain_field) * land.astype(np.float32))
    plain_field = _norm(0.52 * foreland_shadow + 0.48 * basin0 + 0.42 * sediment0 + 0.35 * inland_basin0 + 0.42 * coastal_plain_field) * land.astype(np.float32)
    valley_field = _norm(0.50 * river_valley_belts + 0.35 * valley0 + 0.28 * plain_field + 0.24 * rift_field) * land.astype(np.float32)
    lake_field = _norm((0.45 * lake0 + 0.42 * plain_field + 0.34 * rift_field + 0.30 * inland_basin0 + 0.24 * valley_field) * (0.30 + 0.78 * inland) * (1.0 - 0.50 * mountain_field) * land.astype(np.float32))
    lake_field = _norm(ndimage.gaussian_filter(lake_field, sigma=max(0.9, width / 560.0), mode="wrap") * land.astype(np.float32))
    terrain_detail = _norm(0.42 * mountain_field + 0.26 * plateau_edge + 0.22 * rift_field + 0.16 * np.abs(broad_noise) - 0.20 * plain_field) * land.astype(np.float32)

    new_elev = elev0.copy()
    new_island_land = added_islands & land
    if np.any(new_island_land):
        island_height = 45.0 + 720.0 * island_potential + 460.0 * seamount + 320.0 * arc0
        new_elev[new_island_land] = island_height[new_island_land]
    if np.any(polar_removed):
        new_elev[polar_removed] = -900.0

    base_land = 80.0 + 230.0 * continental + 150.0 * craton + 85.0 * inland + 110.0 * np.clip(broad_noise, -0.4, 1.6)
    new_elev[land] = 0.78 * new_elev[land] + 0.22 * base_land[land]
    relief_gain = 1.05 + 0.36 * clamp(float(controls.get("mountain_belt_strength", 0.72) or 0.72), 0.0, 3.0)
    new_elev[land] += 1720.0 * relief_gain * mountain_field[land]
    new_elev[land] += 420.0 * np.clip(mountain_envelope[land] - mountain_field[land] * 0.34, 0.0, 1.0)
    new_elev[land] += 420.0 * plateau_edge[land]

    plateau_target = 860.0 + 1040.0 * plateau_region + 165.0 * np.clip(broad_noise, -0.3, 1.2)
    plateau_blend = np.clip(0.58 * plateau_region * (1.0 - 0.42 * mountain_field), 0.0, 0.72) * land.astype(np.float32)
    new_elev[land] = new_elev[land] * (1.0 - plateau_blend[land]) + plateau_target[land] * plateau_blend[land]

    # Plains are deliberately common on passive coasts and forelands, but they do
    # not erase mountain belts or plateau escarpments.
    coastal_plain_target = 8.0 + 105.0 * np.clip(land_dist, 0.0, max(1.0, width / 95.0)) / max(1.0, width / 95.0) + 50.0 * np.clip(broad_noise, -0.4, 1.0)
    inland_plain_target = 45.0 + 185.0 * inland + 74.0 * sediment0 + 65.0 * np.clip(broad_noise, -0.5, 1.2)
    plain_target = np.minimum(inland_plain_target, coastal_plain_target + 210.0 * (1.0 - coastal_plain_field))
    plain_blend = np.clip(0.50 * plain_field + 0.58 * coastal_plain_field, 0.0, 0.82) * (1.0 - 0.62 * mountain_field) * land.astype(np.float32)
    new_elev[land] = new_elev[land] * (1.0 - plain_blend[land]) + plain_target[land] * plain_blend[land]

    # Rift troughs and river valleys are carved after plains/plateaus so they are
    # visible as linear depressions.  Rift shoulders restore the raised flanks.
    rift_shoulder = np.clip(ndimage.gaussian_filter(rift_field, sigma=max(1.2, width / 390.0), mode="wrap") - 0.52 * rift_field, 0.0, 1.0)
    new_elev[land] += 340.0 * rift_shoulder[land]
    new_elev[land] -= 760.0 * rift_field[land]
    new_elev[land] -= 360.0 * valley_field[land] * (0.35 + 0.65 * (plain_field[land] + mountain_field[land]))
    new_elev[land] -= 360.0 * lake_field[land]

    fine_noise = np_rng.normal(0.0, 1.0, size=(height, width)).astype(np.float32)
    fine_noise = ndimage.gaussian_filter(fine_noise, sigma=max(0.55, width / 1050.0), mode="wrap")
    if float(np.std(fine_noise)) > 1.0e-6:
        fine_noise = (fine_noise - float(np.mean(fine_noise))) / float(np.std(fine_noise))
    new_elev[land] += (30.0 + 92.0 * terrain_detail[land]) * fine_noise[land] * (1.0 - 0.65 * plain_blend[land])
    new_elev[land] = np.maximum(new_elev[land], 1.0 + 4.0 * coastal_plain_field[land])

    # Variable shelves.  Continental carriers get wide shelves on passive/rifted
    # margins, active margins get narrow/steep shelves, and islands receive only
    # small volcanic aprons.
    labels, count = ndimage.label(land, structure=structure)
    carrier = np.zeros_like(land, dtype=bool)
    if count:
        sizes = np.bincount(labels.ravel()); sizes[0] = 0
        land_cells = max(1.0, float(sizes.sum()))
        for lid in [int(i) for i in np.argsort(sizes)[::-1] if i > 0 and sizes[i] > 0][:5]:
            if sizes[lid] / land_cells > 0.08 or sizes[lid] / total_cells > 0.006:
                carrier |= labels == lid
        if not carrier.any() and sizes.size > 1:
            carrier |= labels == int(np.argmax(sizes))
    ocean = ~land
    dist_to_land = ndimage.distance_transform_edt(ocean).astype(np.float32)
    dist_to_carrier = ndimage.distance_transform_edt(~carrier).astype(np.float32) if carrier.any() else np.full_like(elev0, width, dtype=np.float32)
    island_land = land & (~carrier)
    dist_to_island = ndimage.distance_transform_edt(~island_land).astype(np.float32) if island_land.any() else np.full_like(elev0, 1.0e6, dtype=np.float32)
    carrier_coast = carrier & ((~np.roll(carrier, 1, axis=1)) | (~np.roll(carrier, -1, axis=1)) | (~np.roll(carrier, 1, axis=0)) | (~np.roll(carrier, -1, axis=0)))
    carrier_margin_influence = ndimage.gaussian_filter(carrier_coast.astype(np.float32), sigma=max(1.2, width / 430.0), mode="wrap")
    if carrier_margin_influence.max() > 1.0e-6:
        carrier_margin_influence = carrier_margin_influence / float(carrier_margin_influence.max())
    shelf_noise = np_rng.normal(0.0, 1.0, size=(height, width)).astype(np.float32)
    shelf_noise = ndimage.gaussian_filter(shelf_noise, sigma=max(2.2, width / 125.0), mode="wrap")
    if float(np.std(shelf_noise)) > 1.0e-6:
        shelf_noise = (shelf_noise - float(np.mean(shelf_noise))) / float(np.std(shelf_noise))
    segment_texture = np.clip(0.58 + 0.26 * shelf_noise, 0.0, 1.0)
    active_margin = np.clip(active0 + 0.70 * trench + 0.38 * arc0 + 0.22 * convergence, 0.0, 1.0)
    passive_margin = np.clip(0.35 * carrier_margin_influence + passive0 + 0.58 * coastal_plain0 + 0.38 * sediment0 - 0.55 * active_margin, 0.0, 1.0)
    rifted_margin = np.clip(rifted0 + 0.42 * divergence + 0.18 * rift_field - 0.34 * active_margin, 0.0, 1.0)
    base_width = max(4.5, width / 94.0) * (0.82 + 0.55 * clamp(float(controls.get("shelf_width_factor", 0.55) or 0.55), 0.0, 2.0))
    local_width = base_width * np.clip(0.34 + 1.85 * passive_margin + 1.20 * rifted_margin + 0.55 * segment_texture - 1.35 * active_margin, 0.18, 3.25)
    carrier_gate = (dist_to_carrier <= dist_to_island + max(1.0, width / 420.0)).astype(np.float32)
    support = np.clip(0.42 + 0.82 * passive_margin + 0.58 * rifted_margin + 0.25 * segment_texture - 1.06 * active_margin, 0.0, 1.0)
    shelf_field = np.clip(1.0 - dist_to_carrier / np.maximum(1.0, local_width), 0.0, 1.0) ** 1.18
    shelf_field *= carrier_gate * support * ocean.astype(np.float32)
    shelf_field[(active_margin > 0.45) & (dist_to_carrier > np.maximum(1.6, base_width * 0.24))] *= 0.04
    shelf_field[(microplate > 0.36) & (continental < 0.62)] *= 0.10
    shelf_field[shelf_field < 0.08] = 0.0
    shelf_field = np.clip(shelf_field, 0.0, 1.0)

    deep_ocean = -3550.0 - 760.0 * np.clip(dist_to_land / max(8.0, width / 30.0), 0.0, 1.0) - 1000.0 * trench + 620.0 * ridge + 410.0 * seamount
    non_shelf_nearshore = -230.0 - 780.0 * np.clip(dist_to_land / max(2.0, width / 210.0), 0.0, 1.0) - 1550.0 * np.clip(dist_to_land / max(4.0, width / 48.0), 0.0, 1.0) ** 1.12
    # Let island coasts fall away quickly, but not as a universal -4000m cliff.
    island_apron = -80.0 - 650.0 * np.clip(dist_to_island / max(1.2, width / 330.0), 0.0, 1.0) ** 1.3 - 220.0 * arc0
    ocean_target = np.minimum(deep_ocean, non_shelf_nearshore)
    island_apron_cells = ocean & (dist_to_island < dist_to_carrier) & (dist_to_island <= max(3.0, width / 210.0))
    ocean_target[island_apron_cells] = np.maximum(ocean_target[island_apron_cells], island_apron[island_apron_cells])
    new_elev[ocean] = np.minimum(new_elev[ocean], ocean_target[ocean])
    shelf_cells = ocean & (shelf_field > 0.0)
    if np.any(shelf_cells):
        t = np.clip(dist_to_carrier / np.maximum(1.0, local_width), 0.0, 1.0)
        passive_depth = -42.0 - 210.0 * t - 640.0 * (t ** 1.85)
        active_depth = -155.0 - 640.0 * t - 1540.0 * (t ** 1.35)
        active_factor = np.clip(active_margin * 1.35, 0.0, 1.0)
        shelf_depth = passive_depth * (1.0 - active_factor) + active_depth * active_factor
        shelf_depth += 75.0 * segment_texture
        new_elev[shelf_cells] = np.maximum(new_elev[shelf_cells], shelf_depth[shelf_cells])
    new_elev[ocean] = np.minimum(new_elev[ocean], -1.0)

    # Update island origin diagnostics after restoring islands.
    island_origin = island_origin0.copy()
    island_origin[carrier] = 1
    island_origin[added_islands & (seamount > 0.46)] = 5
    island_origin[added_islands & (arc0 > 0.32)] = 3
    island_origin[added_islands & (microplate > 0.36)] = 4
    island_origin[ocean] = 0

    new_elev_i = np.rint(np.clip(new_elev, -11000, 10000)).astype(np.int32)
    terrain.elevation_m = new_elev_i.astype(int).tolist()
    terrain.is_land = land.astype(bool).tolist()
    terrain.land_fraction = float(np.mean(land))
    terrain.ocean_fraction = 1.0 - terrain.land_fraction
    terrain.min_elevation_m = int(new_elev_i.min())
    terrain.max_elevation_m = int(new_elev_i.max())
    terrain.mean_land_elevation_m = float(np.mean(new_elev_i[land])) if np.any(land) else 0.0
    terrain.mean_ocean_depth_m = float(np.mean(new_elev_i[ocean])) if np.any(ocean) else 0.0
    terrain.source = str(getattr(terrain, "source", "plate_tectonic_v1")) + "; Plate Terrain 14 balanced shelves/islands/landforms"

    labels_final, count_final = ndimage.label(land, structure=structure)
    sizes_final = np.bincount(labels_final.ravel()) if count_final else np.array([0], dtype=np.int64)
    if sizes_final.size:
        sizes_final[0] = 0
    land_cells_final = int(np.sum(sizes_final)) if sizes_final.size else 0
    largest_share = float(np.max(sizes_final) / max(1, land_cells_final)) if land_cells_final else 0.0
    island_components = 0
    if count_final and land_cells_final:
        for lid in range(1, count_final + 1):
            share = float(sizes_final[lid]) / max(1.0, float(land_cells_final))
            if 0.00008 <= share <= 0.025:
                island_components += 1
    shelf_ocean_share = float(np.mean((shelf_field > 0.12)[ocean])) if np.any(ocean) else 0.0
    broad_shelf_share = float(np.mean((shelf_field > 0.45)[ocean])) if np.any(ocean) else 0.0
    coastal_plain_share = float(np.mean((coastal_plain_field > 0.35)[land])) if np.any(land) else 0.0
    valley_share = float(np.mean((valley_field > 0.38)[land])) if np.any(land) else 0.0
    rift_share = float(np.mean((rift_field > 0.38)[land])) if np.any(land) else 0.0
    mountain_share = float(np.mean((mountain_field > 0.42)[land])) if np.any(land) else 0.0
    plateau_share = float(np.mean((plateau_region > 0.35)[land])) if np.any(land) else 0.0
    polar_land_share = float(np.mean(land[abs_lat > 70.0])) if np.any(abs_lat > 70.0) else 0.0

    meta = {
        "applied": True,
        "stage": "plate-terrain-14-feature-balance",
        "land_fraction_after": round(float(terrain.land_fraction), 4),
        "largest_landmass_share_of_land_after": round(largest_share, 4),
        "landmass_count_after": int(count_final),
        "island_components_after": int(island_components),
        "added_island_cells": int(np.sum(added_islands)),
        "shelf_ocean_share_after": round(shelf_ocean_share, 4),
        "broad_shelf_ocean_share_after": round(broad_shelf_share, 4),
        "mountain_land_share_after": round(mountain_share, 4),
        "plateau_land_share_after": round(plateau_share, 4),
        "coastal_plain_land_share_after": round(coastal_plain_share, 4),
        "rift_land_share_after": round(rift_share, 4),
        "valley_land_share_after": round(valley_share, 4),
        "lake_candidate_land_share_after": round(float(np.mean((lake_field > 0.46)[land])) if np.any(land) else 0.0, 4),
        "polar_land_suppression_enabled": bool(suppress_polar),
        "polar_land_share_after_70deg": round(polar_land_share, 4),
        "polar_cells_oceanized": int(np.sum(polar_removed)),
        "mean_abs_elevation_delta_m": round(float(np.mean(np.abs(new_elev_i.astype(np.float32) - elev0))), 2),
        "fixes": [
            "restored a controlled volcanic/arc/microcontinent island budget after U13 continent consolidation",
            "rebuilt shelves as variable-width passive/rifted continental margins instead of all-or-nothing halos",
            "added branching mountain belts and branch rift/valley linework",
            "made coastal plains more common on passive/rifted low-relief margins",
            "reshaped plateaus into broad elevated regions with escarpment-like edges",
            "added optional polar land suppression through --suppress-polar-land",
        ],
    }
    try:
        diagnostics = terrain.terrain_diagnostics if isinstance(terrain.terrain_diagnostics, dict) else {}
        plate_diag = diagnostics.setdefault("plate_tectonic_v1", {})
        plate_diag["feature_balance_u14"] = meta
        terrain.terrain_diagnostics = diagnostics
    except Exception:
        pass

    return {
        "metadata": meta,
        "mountain_strength_x1000": _x1000(mountain_field),
        "plateau_x1000": _x1000(plateau_region),
        "rift_x1000": _x1000(rift_field),
        "valley_corridor_x1000": _x1000(valley_field),
        "inland_basin_x1000": _x1000(np.maximum(inland_basin0, np.maximum(plain_field, lake_field * 0.8))),
        "lake_candidate_x1000": _x1000(np.maximum(lake0, lake_field)),
        "terrain_detail_x1000": _x1000(terrain_detail),
        "shelf_width_x1000": _x1000(shelf_field),
        "island_origin_class": island_origin.astype(int).tolist(),
    }

def _apply_plate_tectonic_v1_crust_model_u15(
    rng: random.Random,
    terrain: TerrainMap,
    hydrosphere: Hydrosphere,
    plate_setup: dict,
    plate_relief: dict,
    plate_coasts: dict,
    plate_drainage: dict,
    geology: GeologyState,
    controls: dict,
) -> dict:
    """Plate Terrain 15 structural crust and landform pass.

    Update 14 restored shelves, but the islands were still effectively painted on
    top of the corrected continent mask and landforms were still post-hoc noise.
    This pass makes a first explicit crust-class layer: continental core,
    continental margin, stretched/rifted crust, microcontinent, oceanic crust,
    oceanic plateau, volcanic arc, and hotspot chain.  The visible mask and
    elevation are then rebuilt from those crust classes so islands/small
    continents, shelves, mountains, plateaus, rifts, valleys, and plains have a
    tectonic reason.
    """
    try:
        import numpy as np
        from PIL import Image
        from scipy import ndimage
    except ImportError as exc:
        raise RuntimeError("NumPy, Pillow, and SciPy are required for plate terrain generation. Install with: pip install -r requirements.txt") from exc

    elev0 = np.asarray(terrain.elevation_m, dtype=np.float32)
    land = np.asarray(terrain.is_land, dtype=bool).copy()
    height, width = elev0.shape
    total_cells = max(1, height * width)
    np_rng = np.random.default_rng(int(rng.randrange(1, 2**31 - 1)))
    structure = np.array([[0,1,0],[1,1,1],[0,1,0]], dtype=np.uint8)

    def _full_field(value, *, default: float = 0.0, scale: float = 1000.0):
        if value is None:
            return np.full((height, width), float(default), dtype=np.float32)
        arr = np.asarray(value, dtype=np.float32)
        if arr.size == 0:
            return np.full((height, width), float(default), dtype=np.float32)
        if scale:
            arr = arr / scale
        if arr.shape != (height, width):
            img = Image.fromarray(arr.astype(np.float32), mode="F")
            arr = np.asarray(img.resize((width, height), Image.Resampling.BICUBIC), dtype=np.float32)
        return np.nan_to_num(arr, nan=default, posinf=default, neginf=default).astype(np.float32)

    def _full_class(value, *, default: int = 0):
        if value is None:
            return np.full((height, width), int(default), dtype=np.int32)
        arr = np.asarray(value, dtype=np.int32)
        if arr.size == 0:
            return np.full((height, width), int(default), dtype=np.int32)
        if arr.shape != (height, width):
            img = Image.fromarray(arr.astype(np.int16), mode="I;16")
            arr = np.asarray(img.resize((width, height), Image.Resampling.NEAREST), dtype=np.int32)
        return arr.astype(np.int32)

    def _norm(arr):
        arr = np.nan_to_num(np.asarray(arr, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        positive = arr[arr > 0.0]
        if positive.size:
            lo = float(np.percentile(positive, 7))
            hi = float(np.percentile(positive, 97))
            arr = (arr - lo) / max(1.0e-6, hi - lo)
        return np.clip(arr, 0.0, 1.0)

    def _x1000(arr):
        return np.rint(np.clip(arr, 0.0, 1.0) * 1000.0).astype(np.int16).astype(int).tolist()

    def _wrapped_cells_delta(c_grid, c0):
        return ((c_grid - c0 + width / 2.0) % width) - width / 2.0

    continental = np.clip(_full_field(plate_setup.get("continental_crust_x1000")), 0.0, 1.0)
    craton = np.clip(_full_field(plate_setup.get("craton_core_x1000")), 0.0, 1.0)
    microplate = np.clip(_full_field(plate_setup.get("microplate_x1000")), 0.0, 1.0)
    plate_type = _full_class(plate_setup.get("plate_type"), default=0)
    convergence = np.clip(_full_field(plate_setup.get("convergence_x1000")), 0.0, 1.0)
    divergence = np.clip(_full_field(plate_setup.get("divergence_x1000")), 0.0, 1.0)
    transform = np.clip(_full_field(plate_setup.get("transform_x1000")), 0.0, 1.0)
    trench = np.clip(_full_field(plate_setup.get("trench_x1000")), 0.0, 1.0)
    ridge = np.clip(_full_field(plate_setup.get("mid_ocean_ridge_x1000")), 0.0, 1.0)
    fracture = np.clip(_full_field(plate_setup.get("fracture_zone_x1000")), 0.0, 1.0)
    seamount = np.clip(_full_field(plate_setup.get("seamount_x1000")), 0.0, 1.0)
    ocean_age = np.clip(_full_field(plate_setup.get("ocean_crust_age_x1000")), 0.0, 1.0)

    orogeny0 = np.clip(_full_field(plate_relief.get("orogeny_strength_x1000")), 0.0, 1.0)
    arc0 = np.clip(_full_field(plate_relief.get("volcanic_arc_x1000")), 0.0, 1.0)
    rift0 = np.clip(_full_field(plate_relief.get("continental_rift_x1000")), 0.0, 1.0)
    foreland0 = np.clip(_full_field(plate_relief.get("foreland_basin_x1000")), 0.0, 1.0)
    sediment0 = np.clip(_full_field(plate_relief.get("sedimentary_plain_x1000")), 0.0, 1.0)
    plateau0 = np.clip(_full_field(plate_relief.get("plateau_uplift_x1000")), 0.0, 1.0)
    passive0 = np.clip(_full_field(plate_coasts.get("passive_margin_x1000")), 0.0, 1.0)
    rifted0 = np.clip(_full_field(plate_coasts.get("rifted_margin_x1000")), 0.0, 1.0)
    active0 = np.clip(_full_field(plate_coasts.get("active_margin_x1000")), 0.0, 1.0)
    coastal_plain0 = np.clip(_full_field(plate_coasts.get("coastal_plain_x1000")), 0.0, 1.0)
    shelf0 = np.clip(_full_field(getattr(terrain, "terrain_shelf_width_x1000", None)), 0.0, 1.0)
    valley0 = np.clip(_full_field(getattr(terrain, "plate_tectonic_valley_corridor_x1000", None)), 0.0, 1.0)
    basin0 = np.clip(_full_field(getattr(terrain, "plate_tectonic_inland_basin_x1000", None)), 0.0, 1.0)
    lake0 = np.clip(_full_field(getattr(terrain, "plate_tectonic_lake_candidate_x1000", None)), 0.0, 1.0)
    island_origin0 = _full_class(getattr(terrain, "terrain_island_origin_class", None), default=0)

    lats = np.linspace(90.0 - 90.0 / height, -90.0 + 90.0 / height, height, dtype=np.float32)
    lons = np.linspace(-180.0 + 180.0 / width, 180.0 - 180.0 / width, width, dtype=np.float32)
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    row_grid, col_grid = np.mgrid[0:height, 0:width]
    abs_lat = np.abs(lat_grid)
    cos_lat = np.maximum(0.18, np.cos(np.radians(lat_grid))).astype(np.float32)
    suppress_polar = bool(controls.get("suppress_polar_land", False))

    # Remove U14-style decorative islands that are not supported by a crust or
    # margin signal.  Keep genuine continental fragments and arc/hotspot islands.
    labels, count = ndimage.label(land, structure=structure)
    sizes = np.bincount(labels.ravel()) if count else np.array([0], dtype=np.int64)
    if sizes.size:
        sizes[0] = 0
    land_cells = max(1.0, float(np.sum(sizes)))
    component_share = np.zeros_like(sizes, dtype=np.float32)
    if sizes.size:
        component_share = sizes.astype(np.float32) / land_cells
    ranked = [int(i) for i in np.argsort(sizes)[::-1] if i > 0 and sizes[i] > 0]
    carrier = np.zeros_like(land, dtype=bool)
    for lid in ranked[:5]:
        if component_share[lid] > 0.045 or sizes[lid] / total_cells > 0.006:
            carrier |= labels == lid
    if not carrier.any() and ranked:
        carrier |= labels == ranked[0]

    tectonic_island_support = _norm(
        0.82 * arc0
        + 0.72 * seamount
        + 0.58 * microplate
        + 0.34 * ridge
        + 0.26 * convergence
        + 0.20 * fracture
        - 0.30 * trench
    )
    continental_fragment_support = _norm(
        0.78 * continental
        + 0.42 * (plate_type == 2).astype(np.float32)
        + 0.34 * microplate
        + 0.26 * rifted0
        + 0.20 * passive0
        + 0.16 * craton
        - 0.30 * trench
    )
    remove_mask = np.zeros_like(land, dtype=bool)
    removed_components = 0
    for lid in ranked:
        comp = labels == lid
        if not comp.any() or np.any(carrier & comp):
            continue
        share = float(component_share[lid]) if lid < component_share.size else 0.0
        mean_island = float(np.mean(tectonic_island_support[comp]))
        max_island = float(np.max(tectonic_island_support[comp]))
        mean_frag = float(np.mean(continental_fragment_support[comp]))
        max_frag = float(np.max(continental_fragment_support[comp]))
        # Very tiny pieces need very strong support; larger fragments can remain
        # if they are coherent microcontinental crust.
        if (share < 0.00055 and max(max_island, max_frag) < 0.58) or (share < 0.018 and mean_island < 0.24 and mean_frag < 0.30 and max(max_island, max_frag) < 0.66):
            remove_mask |= comp
            removed_components += 1
    if remove_mask.any():
        land[remove_mask] = False
        land = _remove_one_cell_land_water_needles(land)

    # Rebuild carrier continents after unsupported-island removal.
    labels, count = ndimage.label(land, structure=structure)
    sizes = np.bincount(labels.ravel()) if count else np.array([0], dtype=np.int64)
    if sizes.size:
        sizes[0] = 0
    land_cells = max(1.0, float(np.sum(sizes)))
    ranked = [int(i) for i in np.argsort(sizes)[::-1] if i > 0 and sizes[i] > 0]
    carrier = np.zeros_like(land, dtype=bool)
    for lid in ranked[:5]:
        if sizes[lid] / land_cells > 0.045 or sizes[lid] / total_cells > 0.006:
            carrier |= labels == lid
    if not carrier.any() and ranked:
        carrier |= labels == ranked[0]

    ocean = ~land
    dist_to_carrier = ndimage.distance_transform_edt(~carrier).astype(np.float32) if carrier.any() else np.full_like(elev0, width, dtype=np.float32)
    dist_to_any_land = ndimage.distance_transform_edt(ocean).astype(np.float32)
    far_from_carrier = dist_to_carrier > max(4.0, width / 145.0)
    polar_ok = abs_lat < (68.0 if suppress_polar else 82.0)

    # Dynamic microcontinents and small continents: threshold coherent crustal
    # fragments in mixed/continental crust fields, instead of sprinkling dots.
    coarse_noise = np_rng.normal(0.0, 1.0, size=(height, width)).astype(np.float32)
    coarse_noise = ndimage.gaussian_filter(coarse_noise, sigma=max(2.0, width / 105.0), mode="wrap")
    if float(np.std(coarse_noise)) > 1.0e-6:
        coarse_noise = (coarse_noise - float(np.mean(coarse_noise))) / float(np.std(coarse_noise))
    microcontinent_potential = _norm(
        0.70 * continental_fragment_support
        + 0.34 * microplate
        + 0.25 * rifted0
        + 0.20 * sediment0
        + 0.13 * np.maximum(coarse_noise, 0.0)
        - 0.24 * active0
        - 0.20 * trench
    )
    micro_seed = ocean & far_from_carrier & polar_ok & (
        ((plate_type == 2) & (continental > 0.25) & (microcontinent_potential > 0.42))
        | ((plate_type == 3) & (continental > 0.18) & (microcontinent_potential > 0.48))
        | (microcontinent_potential > 0.76)
    )
    micro_seed = ndimage.binary_closing(micro_seed, structure=structure, iterations=1)
    micro_labels, micro_count = ndimage.label(micro_seed, structure=structure)
    added_microcontinents = np.zeros_like(land, dtype=bool)
    micro_added_components = 0
    if micro_count:
        micro_sizes = np.bincount(micro_labels.ravel()); micro_sizes[0] = 0
        candidates = []
        for lid in range(1, micro_count + 1):
            comp = micro_labels == lid
            area = int(micro_sizes[lid])
            if area < max(5, total_cells // 36000):
                continue
            if area > max(1400, total_cells // 42):
                continue
            score = float(np.mean(microcontinent_potential[comp])) + 0.18 * float(np.max(continental[comp])) + 0.08 * math.log1p(area)
            candidates.append((score, lid, area))
        candidates.sort(reverse=True)
        for _score, lid, area in candidates[: max(2, min(7, width // 185 + 2))]:
            comp = micro_labels == lid
            grow = ndimage.binary_dilation(comp, structure=structure, iterations=1)
            grow &= ocean & polar_ok & (microcontinent_potential > 0.34)
            # Keep these as substantial fragments, not pinprick islands.
            if int(np.sum(grow)) >= max(8, total_cells // 52000):
                added_microcontinents |= grow
                ocean[grow] = False
                micro_added_components += 1
    land |= added_microcontinents

    # Dynamic oceanic islands: connected chains/provinces from arc, hotspot,
    # microplate, and ridge signals.  No random center sprinkling; only local
    # maxima in tectonic support fields can become island land.
    ocean = ~land
    dist_to_any_land = ndimage.distance_transform_edt(ocean).astype(np.float32)
    dist_to_carrier = ndimage.distance_transform_edt(~carrier).astype(np.float32) if carrier.any() else np.full_like(elev0, width, dtype=np.float32)
    arc_chain_potential = _norm(
        0.72 * arc0
        + 0.66 * seamount
        + 0.36 * microplate
        + 0.28 * convergence
        + 0.18 * ridge
        + 0.14 * fracture
        - 0.48 * continental
        - 0.18 * trench
    )
    if np.any(ocean & polar_ok):
        dynamic_cut = max(0.58, float(np.percentile(arc_chain_potential[ocean & polar_ok], 98.2)))
    else:
        dynamic_cut = 0.70
    arc_seed = ocean & polar_ok & (dist_to_carrier > max(3.0, width / 250.0)) & (arc_chain_potential > dynamic_cut)
    # Join adjacent high-support cells into chains, but keep the chain thin.
    arc_seed = ndimage.binary_dilation(arc_seed, structure=structure, iterations=1) & ocean & polar_ok & (arc_chain_potential > dynamic_cut - 0.10)
    arc_labels, arc_count = ndimage.label(arc_seed, structure=structure)
    added_arcs = np.zeros_like(land, dtype=bool)
    arc_added_components = 0
    if arc_count:
        arc_sizes = np.bincount(arc_labels.ravel()); arc_sizes[0] = 0
        candidates = []
        for lid in range(1, arc_count + 1):
            comp = arc_labels == lid
            area = int(arc_sizes[lid])
            if area < max(3, total_cells // 90000) or area > max(600, total_cells // 150):
                continue
            mean_support = float(np.mean(arc_chain_potential[comp]))
            if mean_support < 0.58:
                continue
            candidates.append((mean_support + 0.04 * math.log1p(area), lid, area))
        candidates.sort(reverse=True)
        max_chains = max(3, min(14, int(round(4 + width / 170.0 + 10.0 * clamp(float(controls.get("island_density", 0.35) or 0.35), 0.0, 1.0)))))
        for _score, lid, _area in candidates[:max_chains]:
            comp = arc_labels == lid
            # A volcanic chain normally has a summit line and a very small apron.
            core = comp & (arc_chain_potential > dynamic_cut)
            apron = ndimage.binary_dilation(core, structure=structure, iterations=1) & ocean & (arc_chain_potential > dynamic_cut - 0.16)
            patch = core | apron
            if patch.any():
                added_arcs |= patch
                ocean[patch] = False
                arc_added_components += 1
    land |= added_arcs

    if suppress_polar:
        polar_removed = land & (abs_lat >= 72.0)
        land[polar_removed] = False
    else:
        polar_removed = np.zeros_like(land, dtype=bool)
    land = _remove_one_cell_land_water_needles(land)
    ocean = ~land

    # Components and crust classes after dynamic fragments/islands.
    labels, count = ndimage.label(land, structure=structure)
    sizes = np.bincount(labels.ravel()) if count else np.array([0], dtype=np.int64)
    if sizes.size:
        sizes[0] = 0
    land_cells = max(1.0, float(np.sum(sizes)))
    ranked = [int(i) for i in np.argsort(sizes)[::-1] if i > 0 and sizes[i] > 0]
    carrier = np.zeros_like(land, dtype=bool)
    component_kind = np.zeros_like(land, dtype=np.uint8)
    for lid in ranked:
        comp = labels == lid
        share_land = float(sizes[lid]) / land_cells
        share_total = float(sizes[lid]) / total_cells
        mean_cont = float(np.mean(continental[comp])) if np.any(comp) else 0.0
        mean_arc = float(np.mean(arc_chain_potential[comp])) if np.any(comp) else 0.0
        mean_micro = float(np.mean(microplate[comp])) if np.any(comp) else 0.0
        if share_land > 0.045 or share_total > 0.006:
            carrier |= comp
            component_kind[comp] = 1  # major continent
        elif mean_cont > 0.25 or mean_micro > 0.35 or np.any(added_microcontinents & comp):
            component_kind[comp] = 2  # microcontinent / continental fragment
        elif mean_arc > 0.50 or np.any(added_arcs & comp):
            component_kind[comp] = 3  # volcanic/hotspot island chain
        else:
            component_kind[comp] = 3

    dist_land = ndimage.distance_transform_edt(land).astype(np.float32)
    dist_ocean = ndimage.distance_transform_edt(ocean).astype(np.float32)
    dist_to_carrier = ndimage.distance_transform_edt(~carrier).astype(np.float32) if carrier.any() else np.full_like(elev0, width, dtype=np.float32)
    carrier_coast = carrier & ((~np.roll(carrier, 1, axis=1)) | (~np.roll(carrier, -1, axis=1)) | (~np.vstack((carrier[0:1, :], carrier[:-1, :]))) | (~np.vstack((carrier[1:, :], carrier[-1:, :]))))
    carrier_margin = ndimage.gaussian_filter(carrier_coast.astype(np.float32), sigma=max(1.2, width / 390.0), mode="wrap")
    if carrier_margin.max() > 1.0e-6:
        carrier_margin /= float(carrier_margin.max())

    active_margin = np.clip(active0 + 0.70 * trench + 0.32 * convergence + 0.24 * arc0, 0.0, 1.0)
    passive_margin = np.clip(passive0 + 0.56 * coastal_plain0 + 0.32 * sediment0 + 0.25 * carrier_margin - 0.60 * active_margin, 0.0, 1.0)
    rifted_margin = np.clip(rifted0 + 0.44 * divergence + 0.28 * rift0 - 0.32 * active_margin, 0.0, 1.0)

    crust_class = np.zeros((height, width), dtype=np.uint8)
    crust_class[ocean] = 0  # abyssal/oceanic crust by default
    crust_class[ocean & ((shelf0 > 0.18) | ((dist_ocean < max(3.0, width / 115.0)) & ((passive_margin + rifted_margin) > 0.22)))] = 1
    crust_class[ocean & (seamount > 0.56) & (ridge < 0.55)] = 8  # oceanic plateau / seamount province
    crust_class[land & (component_kind == 1)] = 2
    crust_class[land & ((component_kind == 1) & ((orogeny0 > 0.30) | (active_margin > 0.35) | (convergence > 0.36)))] = 3
    crust_class[land & ((passive_margin + rifted_margin + divergence) > 0.72) & (component_kind == 1)] = 4
    crust_class[land & (component_kind == 2)] = 5
    crust_class[land & (component_kind == 3) & ((arc0 + convergence) >= seamount)] = 6
    crust_class[land & (component_kind == 3) & ((arc0 + convergence) < seamount)] = 7

    # Tectonic mountain systems.  Main trunks are seeded from active/collision
    # margins, then branches use the local tangent to the tectonic signal.
    orogenic_signal = _norm(0.58 * orogeny0 + 0.42 * convergence + 0.36 * active_margin + 0.25 * arc0 + 0.18 * transform)
    rift_signal = _norm(0.62 * rift0 + 0.46 * divergence + 0.18 * transform + 0.18 * rifted_margin)
    gy, gx = np.gradient(orogenic_signal)
    ry, rx = np.gradient(rift_signal)
    mountain_lines = np.zeros((height, width), dtype=np.float32)
    rift_lines = np.zeros((height, width), dtype=np.float32)
    valley_lines = np.zeros((height, width), dtype=np.float32)

    def add_belt(field, rr, cc, angle, length_cells, width_cells, strength, branches=0):
        nonlocal mountain_lines, rift_lines, valley_lines
        dx = _wrapped_cells_delta(col_grid.astype(np.float32), float(cc)) * cos_lat
        dy = row_grid.astype(np.float32) - float(rr)
        ca = math.cos(angle); sa = math.sin(angle)
        along = dx * ca + dy * sa
        across = -dx * sa + dy * ca
        waviness = math.sin(float(rr) * 0.17 + float(cc) * 0.11) * 0.7
        wave = np.sin(along / max(4.0, length_cells * 0.16) + waviness) * width_cells * 0.42
        wave += np.sin(along / max(3.0, length_cells * 0.055) + waviness * 2.1) * width_cells * 0.16
        core = np.exp(-((across + wave) / max(0.65, width_cells)) ** 2) * np.exp(-((along / max(2.5, length_cells)) ** 2))
        field[:] = np.maximum(field, (strength * core).astype(np.float32))
        for bi in range(branches):
            off = (bi - (branches - 1) / 2.0) * length_cells * 0.22 + rng.uniform(-0.18, 0.18) * length_cells
            b_angle = angle + rng.choice([-1.0, 1.0]) * rng.uniform(0.45, 1.05)
            b_len = length_cells * rng.uniform(0.25, 0.48)
            b_width = max(0.45, width_cells * rng.uniform(0.42, 0.68))
            bdx = dx - off * ca
            bdy = dy - off * sa
            bca = math.cos(b_angle); bsa = math.sin(b_angle)
            balong = bdx * bca + bdy * bsa
            bacross = -bdx * bsa + bdy * bca
            branch = np.exp(-((bacross) / b_width) ** 2) * np.exp(-((balong / max(2.0, b_len)) ** 2))
            field[:] = np.maximum(field, (strength * rng.uniform(0.34, 0.62) * branch).astype(np.float32))

    def seed_belts(signal, land_mask, count_target, min_sep, length_base, width_base, field, use_grad="orogeny", branches=1):
        candidates = np.argwhere(land_mask & (signal > max(0.34, float(np.percentile(signal[land_mask], 82)) if np.any(land_mask) else 0.5)))
        if candidates.size == 0:
            return 0
        scores = signal[candidates[:, 0], candidates[:, 1]]
        order = np.argsort(scores)[::-1]
        centers: list[tuple[int, int]] = []
        for idx in order[: min(len(order), count_target * 25)]:
            rr, cc = int(candidates[int(idx), 0]), int(candidates[int(idx), 1])
            if any(abs(rr - ar) < min_sep and min(abs(cc - ac), width - abs(cc - ac)) < min_sep for ar, ac in centers):
                continue
            centers.append((rr, cc))
            if len(centers) >= count_target:
                break
        for rr, cc in centers:
            if use_grad == "rift":
                angle = math.atan2(float(ry[rr, cc]), float(rx[rr, cc])) + math.pi / 2.0
            else:
                angle = math.atan2(float(gy[rr, cc]), float(gx[rr, cc])) + math.pi / 2.0
            if not math.isfinite(angle):
                angle = rng.uniform(0.0, math.tau)
            angle += rng.uniform(-0.34, 0.34)
            add_belt(field, rr, cc, angle, length_base * rng.uniform(0.68, 1.35), width_base * rng.uniform(0.65, 1.45), float(signal[rr, cc]), branches=branches)
        return len(centers)

    mountain_seed_mask = land & ((crust_class == 3) | ((orogenic_signal > 0.42) & (component_kind == 1)))
    mountain_count = seed_belts(orogenic_signal, mountain_seed_mask, max(5, min(18, width // 115)), max(8, width // 70), max(18.0, width / 12.0), max(1.0, width / 520.0), mountain_lines, branches=2)
    rift_seed_mask = land & ((crust_class == 4) | ((rift_signal > 0.36) & (component_kind != 3)))
    rift_count = seed_belts(rift_signal, rift_seed_mask, max(3, min(10, width // 190)), max(10, width // 80), max(20.0, width / 10.5), max(0.65, width / 760.0), rift_lines, use_grad="rift", branches=1)

    # River-valley corridors start in highlands and trend down the elevation
    # gradient; they are deliberately weaker/longer than rifts.
    rough_base = ndimage.gaussian_filter(elev0, sigma=max(1.0, width / 360.0), mode="wrap")
    ey, ex = np.gradient(rough_base)
    valley_seed = land & (component_kind != 3) & (_norm(0.45 * mountain_lines + 0.25 * plateau0 + 0.22 * basin0 + 0.18 * sediment0) > 0.30)
    candidates = np.argwhere(valley_seed)
    if candidates.size:
        scores = (_norm(mountain_lines + 0.45 * plateau0 + 0.25 * basin0))[candidates[:, 0], candidates[:, 1]]
        order = np.argsort(scores)[::-1]
        centers = []
        for idx in order[: min(len(order), max(8, width // 75) * 18)]:
            rr, cc = int(candidates[int(idx), 0]), int(candidates[int(idx), 1])
            if any(abs(rr - ar) < max(5, height // 80) and min(abs(cc - ac), width - abs(cc - ac)) < max(8, width // 85) for ar, ac in centers):
                continue
            centers.append((rr, cc))
            if len(centers) >= max(8, min(24, width // 65)):
                break
        for rr, cc in centers:
            # Downhill direction is negative gradient.
            angle = math.atan2(float(-ey[rr, cc]), float(-ex[rr, cc] * cos_lat[rr, cc]))
            if not math.isfinite(angle):
                angle = rng.uniform(0.0, math.tau)
            add_belt(valley_lines, rr, cc, angle + rng.uniform(-0.28, 0.28), max(14.0, width / 18.0) * rng.uniform(0.55, 1.15), max(0.40, width / 1050.0), 0.62, branches=1)

    mountain_field = _norm(0.66 * mountain_lines + 0.38 * orogenic_signal + 0.18 * arc0) * land.astype(np.float32)
    rift_field = _norm(0.72 * rift_lines + 0.45 * rift_signal + 0.16 * transform) * land.astype(np.float32)
    valley_field = _norm(0.68 * valley_lines + 0.24 * valley0 + 0.20 * basin0 + 0.14 * rift_field) * land.astype(np.float32)

    # Plateaus by type: collision plateau, old craton/highland, and rift/volcanic
    # provinces.  Threshold and fill so plateaus have coherent edges, not just
    # raised Perlin noise.
    plateau_potential = _norm(
        0.38 * plateau0
        + 0.28 * craton
        + 0.24 * ndimage.gaussian_filter(mountain_field, sigma=max(2.0, width / 130.0), mode="wrap")
        + 0.18 * seamount * (component_kind != 3)
        + 0.16 * rifted_margin
        - 0.24 * valley_field
        - 0.20 * passive_margin
    ) * land.astype(np.float32)
    plateau_region = np.zeros_like(plateau_potential, dtype=np.float32)
    if np.any(land):
        pcut = max(0.40, float(np.percentile(plateau_potential[land], 72)))
        preg = (plateau_potential > pcut) & land & (component_kind != 3)
        preg = ndimage.binary_closing(preg, structure=structure, iterations=2)
        preg = ndimage.binary_fill_holes(preg)
        plateau_region = ndimage.gaussian_filter(preg.astype(np.float32), sigma=max(0.9, width / 520.0), mode="wrap")
        plateau_region = np.clip(plateau_region * _norm(plateau_potential + 0.25 * preg.astype(np.float32)), 0.0, 1.0)
    plateau_edge = np.clip(plateau_region - ndimage.minimum_filter(plateau_region, size=max(3, min(19, width // 150))), 0.0, 1.0)

    plain_field = _norm(
        0.36 * sediment0
        + 0.34 * foreland0
        + 0.30 * coastal_plain0
        + 0.28 * passive_margin * np.exp(-dist_land / max(3.0, width / 95.0))
        + 0.18 * basin0
        - 0.32 * mountain_field
        - 0.18 * plateau_edge
    ) * land.astype(np.float32)
    coastal_plain_field = _norm((passive_margin + 0.65 * rifted_margin + 0.45 * coastal_plain0 - 0.65 * active_margin) * np.exp(-dist_land / max(3.0, width / 85.0))) * land.astype(np.float32)
    inland_plain_field = _norm(0.52 * plain_field + 0.30 * basin0 + 0.20 * foreland0 - 0.35 * coastal_plain_field) * land.astype(np.float32)
    lake_field = _norm((0.38 * lake0 + 0.36 * basin0 + 0.30 * rift_field + 0.25 * valley_field + 0.18 * inland_plain_field) * (1.0 - 0.54 * mountain_field) * land.astype(np.float32))
    terrain_detail = _norm(0.42 * mountain_field + 0.26 * plateau_edge + 0.18 * rift_field + 0.16 * valley_field + 0.12 * np.abs(coarse_noise)) * land.astype(np.float32)

    # Rebuild elevation from crust classes.  Continental terrain is broad and
    # low-gradient unless uplifted; oceanic islands are steep and localized.
    new_elev = elev0.copy()
    continental_land = land & np.isin(crust_class, [2, 3, 4, 5])
    oceanic_island_land = land & np.isin(crust_class, [6, 7])
    base_continent = 120.0 + 260.0 * continental + 150.0 * craton + 70.0 * (component_kind == 2).astype(np.float32)
    new_elev[continental_land] = 0.60 * elev0[continental_land] + 0.40 * base_continent[continental_land]
    new_elev[oceanic_island_land] = 55.0 + 900.0 * arc_chain_potential[oceanic_island_land] + 330.0 * seamount[oceanic_island_land]
    relief_gain = 1.00 + 0.36 * clamp(float(controls.get("mountain_belt_strength", 0.72) or 0.72), 0.0, 3.0)
    new_elev[land] += 1450.0 * relief_gain * mountain_field[land]
    new_elev[land] += 320.0 * ndimage.gaussian_filter(mountain_field, sigma=max(1.6, width / 310.0), mode="wrap")[land]
    # Plateau target with escarpment; plains can later cut into it.
    plateau_target = 780.0 + 980.0 * plateau_region + 170.0 * craton + 120.0 * np.clip(coarse_noise, -0.3, 1.2)
    plateau_blend = np.clip(0.62 * plateau_region * (1.0 - 0.28 * mountain_field), 0.0, 0.78) * continental_land.astype(np.float32)
    new_elev[continental_land] = new_elev[continental_land] * (1.0 - plateau_blend[continental_land]) + plateau_target[continental_land] * plateau_blend[continental_land]
    new_elev[land] += 380.0 * plateau_edge[land]

    coast_plain_target = 10.0 + 120.0 * np.clip(dist_land, 0.0, max(2.0, width / 86.0)) / max(2.0, width / 86.0) + 40.0 * np.clip(coarse_noise, -0.4, 0.9)
    inland_plain_target = 70.0 + 190.0 * inland_plain_field + 80.0 * sediment0 + 65.0 * np.clip(coarse_noise, -0.5, 1.0)
    plain_target = np.minimum(inland_plain_target, coast_plain_target + 250.0 * (1.0 - coastal_plain_field))
    plain_blend = np.clip(0.46 * inland_plain_field + 0.58 * coastal_plain_field, 0.0, 0.82) * (1.0 - 0.65 * mountain_field) * continental_land.astype(np.float32)
    new_elev[continental_land] = new_elev[continental_land] * (1.0 - plain_blend[continental_land]) + plain_target[continental_land] * plain_blend[continental_land]

    rift_shoulder = np.clip(ndimage.gaussian_filter(rift_field, sigma=max(1.2, width / 420.0), mode="wrap") - 0.44 * rift_field, 0.0, 1.0)
    new_elev[land] += 300.0 * rift_shoulder[land]
    new_elev[land] -= 780.0 * rift_field[land]
    new_elev[land] -= 330.0 * valley_field[land] * (0.45 + 0.55 * np.maximum(mountain_field[land], inland_plain_field[land]))
    new_elev[land] -= 300.0 * lake_field[land]
    new_elev[land] = np.maximum(new_elev[land], 1.0 + 3.0 * coastal_plain_field[land])

    # Preserve Update14's successful shelf idea, but tie it to crust classes and
    # carrier continents. Wide shelves are submerged continental crust; active
    # margins and oceanic island chains stay narrow/steep.
    ocean = ~land
    dist_to_land_ocean = ndimage.distance_transform_edt(ocean).astype(np.float32)
    dist_to_carrier = ndimage.distance_transform_edt(~carrier).astype(np.float32) if carrier.any() else np.full_like(elev0, width, dtype=np.float32)
    island_land = land & np.isin(crust_class, [6, 7])
    dist_to_island = ndimage.distance_transform_edt(~island_land).astype(np.float32) if island_land.any() else np.full_like(elev0, 1.0e6, dtype=np.float32)
    shelf_texture = ndimage.gaussian_filter(coarse_noise, sigma=max(1.4, width / 210.0), mode="wrap")
    shelf_texture = np.clip(0.64 + 0.22 * shelf_texture, 0.22, 1.0)
    base_width = max(4.5, width / 92.0) * (0.82 + 0.50 * clamp(float(controls.get("shelf_width_factor", 0.55) or 0.55), 0.0, 2.0))
    shelf_width = base_width * np.clip(0.36 + 1.80 * passive_margin + 1.18 * rifted_margin + 0.35 * shelf_texture - 1.28 * active_margin, 0.16, 3.35)
    carrier_gate = (dist_to_carrier <= dist_to_island + max(1.0, width / 420.0)).astype(np.float32)
    submerged_continent = (continental > 0.20) | (plate_type == 2) | (shelf0 > 0.16)
    shelf_support = np.clip(0.35 + 0.78 * passive_margin + 0.58 * rifted_margin + 0.22 * shelf_texture - 1.02 * active_margin, 0.0, 1.0)
    shelf_field = np.clip(1.0 - dist_to_carrier / np.maximum(1.0, shelf_width), 0.0, 1.0) ** 1.16
    shelf_field *= carrier_gate * shelf_support * ocean.astype(np.float32) * submerged_continent.astype(np.float32)
    shelf_field[(active_margin > 0.45) & (dist_to_carrier > np.maximum(1.5, base_width * 0.25))] *= 0.08
    shelf_field[shelf_field < 0.07] = 0.0
    crust_class[ocean & (shelf_field > 0.08)] = 1
    crust_class[ocean & (seamount > 0.56) & (shelf_field <= 0.08)] = 8

    deep_ocean = -3700.0 - 700.0 * np.clip(dist_to_land_ocean / max(8.0, width / 28.0), 0.0, 1.0) - 980.0 * trench + 610.0 * ridge + 420.0 * seamount
    active_nearshore = -165.0 - 620.0 * np.clip(dist_to_land_ocean / max(2.0, width / 220.0), 0.0, 1.0) - 1500.0 * np.clip(dist_to_land_ocean / max(4.0, width / 54.0), 0.0, 1.0) ** 1.16
    new_elev[ocean] = np.minimum(new_elev[ocean], np.minimum(deep_ocean[ocean], active_nearshore[ocean]))
    shelf_cells = ocean & (shelf_field > 0.0)
    if np.any(shelf_cells):
        t = np.clip(dist_to_carrier / np.maximum(1.0, shelf_width), 0.0, 1.0)
        passive_depth = -36.0 - 190.0 * t - 620.0 * (t ** 1.9)
        active_depth = -150.0 - 610.0 * t - 1520.0 * (t ** 1.32)
        active_factor = np.clip(active_margin * 1.30, 0.0, 1.0)
        shelf_depth = passive_depth * (1.0 - active_factor) + active_depth * active_factor + 55.0 * shelf_texture
        new_elev[shelf_cells] = np.maximum(new_elev[shelf_cells], shelf_depth[shelf_cells])
    island_apron_cells = ocean & (dist_to_island < dist_to_carrier) & (dist_to_island <= max(3.0, width / 215.0))
    if np.any(island_apron_cells):
        apron = -85.0 - 650.0 * np.clip(dist_to_island / max(1.2, width / 335.0), 0.0, 1.0) ** 1.34 - 220.0 * arc0
        new_elev[island_apron_cells] = np.maximum(new_elev[island_apron_cells], apron[island_apron_cells])
    new_elev[ocean] = np.minimum(new_elev[ocean], -1.0)

    # Fine texture, reduced on plains/plateaus so landforms remain coherent.
    fine = np_rng.normal(0.0, 1.0, size=(height, width)).astype(np.float32)
    fine = ndimage.gaussian_filter(fine, sigma=max(0.55, width / 1150.0), mode="wrap")
    if float(np.std(fine)) > 1.0e-6:
        fine = (fine - float(np.mean(fine))) / float(np.std(fine))
    texture_mask = (1.0 - 0.62 * plain_blend) * (1.0 - 0.35 * plateau_region)
    new_elev[land] += (24.0 + 78.0 * terrain_detail[land]) * fine[land] * texture_mask[land]
    new_elev[land] = np.maximum(new_elev[land], 1.0)

    island_origin = island_origin0.copy()
    island_origin[:] = 0
    island_origin[land & (component_kind == 1)] = 1
    island_origin[land & (component_kind == 2)] = 4
    island_origin[land & (crust_class == 6)] = 3
    island_origin[land & (crust_class == 7)] = 5

    elev_i = np.rint(np.clip(new_elev, -11000, 10000)).astype(np.int32)
    terrain.elevation_m = elev_i.astype(int).tolist()
    terrain.is_land = land.astype(bool).tolist()
    terrain.land_fraction = float(np.mean(land))
    terrain.ocean_fraction = 1.0 - terrain.land_fraction
    terrain.min_elevation_m = int(elev_i.min())
    terrain.max_elevation_m = int(elev_i.max())
    terrain.mean_land_elevation_m = float(np.mean(elev_i[land])) if np.any(land) else 0.0
    terrain.mean_ocean_depth_m = float(np.mean(elev_i[ocean])) if np.any(ocean) else 0.0
    terrain.source = str(getattr(terrain, "source", "plate_tectonic_v1")) + "; Plate Terrain 15 crust-class terrain model"

    final_labels, final_count = ndimage.label(land, structure=structure)
    final_sizes = np.bincount(final_labels.ravel()) if final_count else np.array([0], dtype=np.int64)
    if final_sizes.size:
        final_sizes[0] = 0
    final_land_cells = max(1.0, float(np.sum(final_sizes)))
    largest_share = float(np.max(final_sizes) / final_land_cells) if final_land_cells > 0 else 0.0
    small_components = 0
    microcontinent_components = 0
    island_chain_components = 0
    for lid in range(1, final_count + 1):
        comp = final_labels == lid
        share = float(final_sizes[lid]) / final_land_cells if lid < final_sizes.size else 0.0
        if 0.00006 <= share <= 0.025:
            small_components += 1
        if np.any((component_kind == 2) & comp):
            microcontinent_components += 1
        if np.any((component_kind == 3) & comp):
            island_chain_components += 1
    shelf_ocean_share = float(np.mean((shelf_field > 0.12)[ocean])) if np.any(ocean) else 0.0
    broad_shelf_share = float(np.mean((shelf_field > 0.45)[ocean])) if np.any(ocean) else 0.0
    active_shelf_conflict = float(np.mean(((shelf_field > 0.35) & (active_margin > 0.45))[ocean])) if np.any(ocean) else 0.0
    mountain_share = float(np.mean((mountain_field > 0.42)[land])) if np.any(land) else 0.0
    rift_share = float(np.mean((rift_field > 0.38)[land])) if np.any(land) else 0.0
    valley_share = float(np.mean((valley_field > 0.36)[land])) if np.any(land) else 0.0
    plateau_share = float(np.mean((plateau_region > 0.35)[land])) if np.any(land) else 0.0
    coastal_plain_share = float(np.mean((coastal_plain_field > 0.35)[land])) if np.any(land) else 0.0
    lake_share = float(np.mean((lake_field > 0.46)[land])) if np.any(land) else 0.0
    oceanic_land_share = float(np.mean(np.isin(crust_class, [6, 7])[land])) if np.any(land) else 0.0
    micro_land_share = float(np.mean((crust_class == 5)[land])) if np.any(land) else 0.0

    meta = {
        "applied": True,
        "stage": "plate-terrain-15-crust-class-model",
        "land_fraction_after": round(float(terrain.land_fraction), 4),
        "landmass_count_after": int(final_count),
        "largest_landmass_share_of_land_after": round(largest_share, 4),
        "small_island_or_fragment_components_after": int(small_components),
        "microcontinent_components_after": int(microcontinent_components),
        "island_chain_components_after": int(island_chain_components),
        "unsupported_island_components_removed": int(removed_components),
        "dynamic_microcontinent_components_added": int(micro_added_components),
        "dynamic_arc_or_hotspot_components_added": int(arc_added_components),
        "shelf_ocean_share_after": round(shelf_ocean_share, 4),
        "broad_shelf_ocean_share_after": round(broad_shelf_share, 4),
        "active_margin_broad_shelf_conflict_share": round(active_shelf_conflict, 4),
        "mountain_belt_count_seeded": int(mountain_count),
        "rift_belt_count_seeded": int(rift_count),
        "mountain_land_share_after": round(mountain_share, 4),
        "plateau_land_share_after": round(plateau_share, 4),
        "coastal_plain_land_share_after": round(coastal_plain_share, 4),
        "rift_land_share_after": round(rift_share, 4),
        "valley_land_share_after": round(valley_share, 4),
        "lake_candidate_land_share_after": round(lake_share, 4),
        "microcontinental_land_share_after": round(micro_land_share, 4),
        "oceanic_island_land_share_after": round(oceanic_land_share, 4),
        "mean_abs_elevation_delta_m": round(float(np.mean(np.abs(elev_i.astype(np.float32) - elev0))), 2),
        "crust_class_codes": {
            "0": "abyssal/oceanic crust",
            "1": "submerged continental shelf / marginal sea",
            "2": "continental interior/core",
            "3": "active/orogenic continental crust",
            "4": "stretched/rifted continental margin",
            "5": "microcontinent / continental fragment",
            "6": "volcanic island arc",
            "7": "hotspot/oceanic island chain",
            "8": "oceanic plateau / seamount province",
        },
        "fixes": [
            "removes unsupported decorative islands introduced by the previous balancing pass",
            "adds microcontinents and islands from crust/margin support fields instead of random centers",
            "separates continental, stretched-margin, microcontinental, volcanic-arc, hotspot, and oceanic crust classes",
            "ties wide continental shelves to submerged continental crust while keeping active margins narrow",
            "rebuilds mountains, plateaus, rifts, valleys, plains, and lake basins from the crust/plate signals",
        ],
    }
    try:
        diagnostics = terrain.terrain_diagnostics if isinstance(terrain.terrain_diagnostics, dict) else {}
        plate_diag = diagnostics.setdefault("plate_tectonic_v1", {})
        plate_diag["crust_model_u15"] = meta
        terrain.terrain_diagnostics = diagnostics
    except Exception:
        pass

    return {
        "metadata": meta,
        "crust_class": crust_class.astype(int).tolist(),
        "mountain_strength_x1000": _x1000(mountain_field),
        "plateau_x1000": _x1000(plateau_region),
        "rift_x1000": _x1000(rift_field),
        "valley_corridor_x1000": _x1000(valley_field),
        "inland_basin_x1000": _x1000(np.maximum.reduce([basin0, inland_plain_field, lake_field * 0.82, foreland0 * 0.72])),
        "lake_candidate_x1000": _x1000(np.maximum(lake0, lake_field)),
        "terrain_detail_x1000": _x1000(terrain_detail),
        "shelf_width_x1000": _x1000(shelf_field),
        "island_origin_class": island_origin.astype(int).tolist(),
    }


def _combine_x1000_fields(a, b):
    """Combine two 0..1000 diagnostic fields with max semantics."""
    if a is None:
        return b
    if b is None:
        return a
    try:
        import numpy as np
        aa = np.asarray(a, dtype=np.float32)
        bb = np.asarray(b, dtype=np.float32)
        if aa.shape != bb.shape:
            from PIL import Image
            img = Image.fromarray(bb.astype(np.float32), mode="F")
            bb = np.asarray(img.resize((aa.shape[1], aa.shape[0]), Image.Resampling.BICUBIC), dtype=np.float32)
        return np.maximum(aa, bb).clip(0, 1000).round().astype(int).tolist()
    except Exception:
        return a


def _apply_plate_tectonic_v1_drainage_ready_landforms(
    rng: random.Random,
    terrain: TerrainMap,
    plate_setup: dict,
    plate_relief: dict,
    plate_coasts: dict,
    geology: GeologyState,
    controls: dict,
) -> dict:
    """Add Plate Terrain 11 drainage-ready relief structure.

    Plate Terrain 10 introduced landform belts and margin profiles, but many
    continents could still read as broad concentric surfaces: low coast, high
    interior, and few enclosed basins.  This pass is deliberately long-term: it
    creates explicit valley corridors, inland basins, lake-candidate depressions,
    and non-concentric terrain detail fields for future hydrology to consume. It
    does not call or reintroduce the legacy terrain generator.
    """
    try:
        import numpy as np
        from PIL import Image
        from scipy import ndimage
    except ImportError as exc:
        raise RuntimeError("NumPy, Pillow, and SciPy are required for plate terrain diagnostics. Install with: pip install -r requirements.txt") from exc

    elev = np.asarray(terrain.elevation_m, dtype=np.float32)
    land = np.asarray(terrain.is_land, dtype=bool)
    ocean = ~land
    height, width = elev.shape
    np_rng = np.random.default_rng(int(rng.randrange(1, 2**31 - 1)))

    def _field(value, *, default: float = 0.0, scale: float = 1000.0):
        if value is None:
            return np.full((height, width), float(default), dtype=np.float32)
        arr = np.asarray(value, dtype=np.float32)
        if arr.size == 0:
            return np.full((height, width), float(default), dtype=np.float32)
        arr = arr / scale
        if arr.shape != (height, width):
            img = Image.fromarray(arr.astype(np.float32), mode="F")
            arr = np.asarray(img.resize((width, height), Image.Resampling.BICUBIC), dtype=np.float32)
        return np.nan_to_num(arr, nan=default, posinf=default, neginf=default).astype(np.float32)

    orogeny = _field(plate_relief.get("orogeny_strength_x1000"))
    volcanic_arc = _field(plate_relief.get("volcanic_arc_x1000"))
    rift = _field(plate_relief.get("continental_rift_x1000"))
    foreland = _field(plate_relief.get("foreland_basin_x1000"))
    shield = _field(plate_relief.get("craton_shield_x1000"))
    plateau = _field(plate_relief.get("plateau_uplift_x1000"))
    sediment = _field(plate_relief.get("sedimentary_plain_x1000"))
    terrane = _field(plate_relief.get("accreted_terrane_x1000"))
    shelf = _field(plate_coasts.get("shelf_width_x1000"))
    coastal_plain = _field(plate_coasts.get("coastal_plain_x1000"))
    passive = _field(plate_coasts.get("passive_margin_x1000"))
    active = _field(plate_coasts.get("active_margin_x1000"))
    rifted_margin = _field(plate_coasts.get("rifted_margin_x1000"))
    convergence = _field(plate_setup.get("convergence_x1000"))
    divergence = _field(plate_setup.get("divergence_x1000"))
    transform = _field(plate_setup.get("transform_x1000"))

    land_f = land.astype(np.float32)
    land_dist = ndimage.distance_transform_edt(land).astype(np.float32)
    coast_decay = np.exp(-land_dist / max(4.0, width / 115.0)).astype(np.float32) * land_f
    inland_factor = np.clip(1.0 - coast_decay, 0.0, 1.0)

    # Broad landform supports.  These are continuous fields, not brightline masks.
    mountain_source = np.clip(np.maximum.reduce([orogeny, volcanic_arc * 0.82, plateau * 0.64, terrane * 0.58]), 0.0, 1.0)
    mountain_envelope = ndimage.gaussian_filter(mountain_source, sigma=max(2.0, width / 90.0), mode="wrap")
    plain_support = np.clip(np.maximum.reduce([sediment, foreland * 0.92, rift * 0.72, coastal_plain * 0.34]), 0.0, 1.0)

    smooth_elev = ndimage.gaussian_filter(elev, sigma=max(2.0, width / 85.0), mode="wrap")
    local_low = np.clip((smooth_elev - elev + 120.0) / 640.0, 0.0, 1.0) * land_f
    low_relief = np.clip(1.0 - np.abs(elev - smooth_elev) / 1100.0, 0.0, 1.0) * land_f

    # Interior basins are accommodation zones: foreland shadows, rift lows,
    # sedimentary plains, and low-convergence interiors.  They are not random
    # inland seas; they remain land for hydrology to decide later.
    basin_seed = np.clip(
        (0.76 * foreland + 0.64 * sediment + 0.48 * rift + 0.34 * local_low + 0.22 * low_relief)
        * (0.30 + 0.78 * inland_factor)
        * (1.0 - 0.46 * mountain_source)
        * land_f,
        0.0,
        1.0,
    )
    inland_basin = ndimage.gaussian_filter(basin_seed, sigma=max(1.5, width / 175.0), mode="wrap")

    # Meandering valley corridors.  Use band-limited noise as a corridor selector,
    # then gate it by mountain sources, plains, rifts, and basins so valleys form
    # as drainage-ready terrain networks rather than arbitrary scratches.
    n1 = np_rng.normal(0.0, 1.0, size=(height, width)).astype(np.float32)
    n2 = np_rng.normal(0.0, 1.0, size=(height, width)).astype(np.float32)
    n1 = ndimage.gaussian_filter(n1, sigma=max(1.1, width / 360.0), mode="wrap")
    n2 = ndimage.gaussian_filter(n2, sigma=max(3.0, width / 120.0), mode="wrap")
    for arr in (n1, n2):
        std = float(np.std(arr))
        if std > 1.0e-6:
            arr -= float(np.mean(arr)); arr /= std
    corridor_texture = np.exp(-np.abs(0.72 * n1 + 0.44 * n2) / 0.48).astype(np.float32)
    valley_seed = np.clip(
        corridor_texture
        * (0.36 + 0.52 * plain_support + 0.44 * inland_basin + 0.42 * mountain_envelope + 0.34 * rift)
        * (1.0 - 0.48 * shield)
        * (1.0 - 0.38 * active)
        * land_f,
        0.0,
        1.0,
    )
    valley_corridor = ndimage.gaussian_filter(valley_seed, sigma=max(0.65, width / 780.0), mode="wrap")
    pos = valley_corridor[valley_corridor > 0.01]
    if pos.size:
        valley_corridor = np.clip((valley_corridor - float(np.percentile(pos, 30))) / max(1.0e-6, float(np.percentile(pos, 95) - np.percentile(pos, 30))), 0.0, 1.0)

    inland_basin = np.clip(inland_basin, 0.0, 1.0)
    pos = inland_basin[inland_basin > 0.01]
    if pos.size:
        inland_basin = np.clip((inland_basin - float(np.percentile(pos, 26))) / max(1.0e-6, float(np.percentile(pos, 94) - np.percentile(pos, 26))), 0.0, 1.0)

    lake_candidate = np.clip(
        inland_basin
        * (0.48 + 0.54 * local_low + 0.36 * rift + 0.34 * valley_corridor)
        * (0.24 + 0.86 * inland_factor)
        * (1.0 - 0.62 * coastal_plain)
        * land_f,
        0.0,
        1.0,
    )
    lake_candidate = ndimage.gaussian_filter(lake_candidate, sigma=max(0.75, width / 520.0), mode="wrap")
    pos = lake_candidate[lake_candidate > 0.015]
    if pos.size:
        lake_candidate = np.clip((lake_candidate - float(np.percentile(pos, 36))) / max(1.0e-6, float(np.percentile(pos, 97) - np.percentile(pos, 36))), 0.0, 1.0)

    # Terrain detail restores natural texture without importing legacy topology.
    detail_noise = np_rng.normal(0.0, 1.0, size=(height, width)).astype(np.float32)
    detail_noise = ndimage.gaussian_filter(detail_noise, sigma=max(0.65, width / 900.0), mode="wrap")
    std = float(np.std(detail_noise))
    if std > 1.0e-6:
        detail_noise = (detail_noise - float(np.mean(detail_noise))) / std
    terrain_detail = np.clip(
        0.30 * mountain_source
        + 0.22 * plateau
        + 0.20 * shield
        + 0.18 * terrane
        + 0.16 * np.maximum(divergence, transform)
        + 0.10 * np.abs(detail_noise)
        - 0.22 * sediment
        - 0.18 * coastal_plain,
        0.0,
        1.0,
    ) * land_f

    # Elevation synthesis.  Basins and valleys carve real negative relief, but the
    # pass keeps land above sea level; hydrology will later decide where standing
    # water appears.  Detail is asymmetric so interiors do not become concentric.
    delta = np.zeros_like(elev, dtype=np.float32)
    delta -= 470.0 * inland_basin * (0.42 + 0.58 * low_relief)
    delta -= 285.0 * valley_corridor * (0.30 + 0.70 * plain_support)
    delta -= 265.0 * lake_candidate
    delta -= 205.0 * rift * inland_factor
    delta += (42.0 + 138.0 * terrain_detail) * detail_noise * land_f * (1.0 - 0.72 * lake_candidate) * (1.0 - 0.48 * valley_corridor)
    # Slightly raise shoulders and old uplands next to basins/valleys so drainage
    # corridors have visible relief instead of flat stains.
    basin_edge = np.clip(ndimage.gaussian_filter(inland_basin, sigma=max(1.2, width / 300.0), mode="wrap") - inland_basin, 0.0, 1.0)
    delta += 145.0 * basin_edge * (0.35 + 0.65 * plateau + 0.25 * shield) * land_f

    new_elev = elev + delta
    # Smooth only drainage depressions, not mountains. This avoids terracing and
    # keeps orogens/rift shoulders crisp.
    smooth_lowlands = ndimage.gaussian_filter(new_elev, sigma=0.80, mode="wrap")
    lowland_blend = np.clip(0.46 * inland_basin + 0.36 * valley_corridor + 0.28 * lake_candidate, 0.0, 0.66) * land_f
    new_elev[land] = new_elev[land] * (1.0 - lowland_blend[land]) + smooth_lowlands[land] * lowland_blend[land]
    new_elev[land] = np.maximum(new_elev[land], 1.0 + 2.5 * valley_corridor[land] + 4.0 * (1.0 - lake_candidate[land]))
    new_elev[ocean] = elev[ocean]

    new_elev_i = np.rint(np.clip(new_elev, -11000, 10000)).astype(np.int32)
    delta_i = np.rint(new_elev_i.astype(np.float32) - elev).astype(np.int32)
    terrain.elevation_m = new_elev_i.astype(int).tolist()
    if np.any(land):
        terrain.max_elevation_m = int(new_elev_i.max())
        terrain.mean_land_elevation_m = float(np.mean(new_elev_i[land]))
    if np.any(ocean):
        terrain.min_elevation_m = int(new_elev_i.min())
        terrain.mean_ocean_depth_m = float(np.mean(new_elev_i[ocean]))

    def _x1000(arr):
        return np.rint(np.clip(arr, 0.0, 1.0) * 1000.0).astype(np.int16).astype(int).tolist()

    meta = {
        "applied": True,
        "stage": "plate-tectonic-v1-drainage-ready-landforms",
        "backend_status": "native plate drainage-ready valleys, basins, lake candidates, and terrain detail applied",
        "mean_valley_corridor_strength": round(float(np.mean(valley_corridor[land])) if np.any(land) else 0.0, 4),
        "strong_valley_corridor_land_share": round(float(np.mean((valley_corridor > 0.42)[land])) if np.any(land) else 0.0, 4),
        "mean_inland_basin_strength": round(float(np.mean(inland_basin[land])) if np.any(land) else 0.0, 4),
        "strong_inland_basin_land_share": round(float(np.mean((inland_basin > 0.45)[land])) if np.any(land) else 0.0, 4),
        "lake_candidate_land_share": round(float(np.mean((lake_candidate > 0.48)[land])) if np.any(land) else 0.0, 4),
        "mean_terrain_detail_strength": round(float(np.mean(terrain_detail[land])) if np.any(land) else 0.0, 4),
        "mean_drainage_ready_delta_m": round(float(np.mean(delta_i[land])) if np.any(land) else 0.0, 2),
        "mean_abs_drainage_ready_delta_m": round(float(np.mean(np.abs(delta_i[land]))) if np.any(land) else 0.0, 2),
        "notes": [
            "Plate Terrain 12 strengthens drainage-ready valley corridors, inland basins, lake-candidate depressions, and terrain detail without reintroducing the legacy terrain backend.",
            "These fields are intended to break concentric continent elevations and prepare future hydrology for through-flow lakes, rift lakes, foreland plains, and interior drainage basins.",
        ],
    }
    return {
        "valley_corridor_x1000": _x1000(valley_corridor),
        "inland_basin_x1000": _x1000(inland_basin),
        "lake_candidate_x1000": _x1000(lake_candidate),
        "terrain_detail_x1000": _x1000(terrain_detail),
        "drainage_ready_delta_m": delta_i.astype(int).tolist(),
        "metadata": meta,
    }



def _apply_plate_tectonic_v1_user_feedback_correction_u13(
    rng: random.Random,
    terrain: TerrainMap,
    hydrosphere: Hydrosphere,
    plate_setup: dict,
    plate_relief: dict,
    plate_coasts: dict,
    plate_drainage: dict,
    geology: GeologyState,
    controls: dict,
) -> dict:
    """Aggressive Plate Terrain 13 corrective pass for the specific visual failures.

    Previous plate-terrain updates mostly exposed diagnostics and small deltas.  This
    pass intentionally changes the visible result.  It targets seven observed
    failures: universal shelf halos, concentric land elevation, weak mountains /
    plateaus / plains / rifts / valleys, smooth coasts, weak lakes, too-smooth
    plates, and excessive island-world fragmentation.
    """
    try:
        import numpy as np
        from PIL import Image
        from scipy import ndimage
    except ImportError as exc:
        raise RuntimeError("NumPy, Pillow, and SciPy are required for plate terrain generation. Install with: pip install -r requirements.txt") from exc

    elev0 = np.asarray(terrain.elevation_m, dtype=np.float32)
    land0 = np.asarray(terrain.is_land, dtype=bool)
    height, width = elev0.shape
    total_cells = max(1, height * width)
    np_rng = np.random.default_rng(int(rng.randrange(1, 2**31 - 1)))

    def _full_field(value, *, default: float = 0.0, scale: float = 1000.0):
        if value is None:
            return np.full((height, width), float(default), dtype=np.float32)
        arr = np.asarray(value, dtype=np.float32)
        if arr.size == 0:
            return np.full((height, width), float(default), dtype=np.float32)
        if scale:
            arr = arr / scale
        if arr.shape != (height, width):
            img = Image.fromarray(arr.astype(np.float32), mode="F")
            arr = np.asarray(img.resize((width, height), Image.Resampling.BICUBIC), dtype=np.float32)
        return np.nan_to_num(arr, nan=default, posinf=default, neginf=default).astype(np.float32)

    def _norm(arr):
        arr = np.nan_to_num(np.asarray(arr, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        positive = arr[arr > 0.0]
        if positive.size:
            lo = float(np.percentile(positive, 8))
            hi = float(np.percentile(positive, 96))
            arr = (arr - lo) / max(1.0e-6, hi - lo)
        return np.clip(arr, 0.0, 1.0)

    def _x1000(arr):
        return np.rint(np.clip(arr, 0.0, 1.0) * 1000.0).astype(np.int16).astype(int).tolist()

    continental = np.clip(_full_field(plate_setup.get("continental_crust_x1000")), 0.0, 1.0)
    craton = np.clip(_full_field(plate_setup.get("craton_core_x1000")), 0.0, 1.0)
    microplate = np.clip(_full_field(plate_setup.get("microplate_x1000")), 0.0, 1.0)
    convergence = np.clip(_full_field(plate_setup.get("convergence_x1000")), 0.0, 1.0)
    divergence = np.clip(_full_field(plate_setup.get("divergence_x1000")), 0.0, 1.0)
    transform = np.clip(_full_field(plate_setup.get("transform_x1000")), 0.0, 1.0)
    trench = np.clip(_full_field(plate_setup.get("trench_x1000")), 0.0, 1.0)
    ridge = np.clip(_full_field(plate_setup.get("mid_ocean_ridge_x1000")), 0.0, 1.0)
    seamount = np.clip(_full_field(plate_setup.get("seamount_x1000")), 0.0, 1.0)
    orogeny0 = np.clip(_full_field(plate_relief.get("orogeny_strength_x1000")), 0.0, 1.0)
    arc0 = np.clip(_full_field(plate_relief.get("volcanic_arc_x1000")), 0.0, 1.0)
    rift0 = np.clip(_full_field(plate_relief.get("continental_rift_x1000")), 0.0, 1.0)
    basin0 = np.clip(_full_field(plate_relief.get("foreland_basin_x1000")), 0.0, 1.0)
    sediment0 = np.clip(_full_field(plate_relief.get("sedimentary_plain_x1000")), 0.0, 1.0)
    plateau0 = np.clip(_full_field(plate_relief.get("plateau_uplift_x1000")), 0.0, 1.0)
    passive0 = np.clip(_full_field(plate_coasts.get("passive_margin_x1000")), 0.0, 1.0)
    rifted0 = np.clip(_full_field(plate_coasts.get("rifted_margin_x1000")), 0.0, 1.0)
    active0 = np.clip(_full_field(plate_coasts.get("active_margin_x1000")), 0.0, 1.0)
    coastal_plain0 = np.clip(_full_field(plate_coasts.get("coastal_plain_x1000")), 0.0, 1.0)
    valley0 = np.clip(_full_field(plate_drainage.get("valley_corridor_x1000")), 0.0, 1.0)
    inland_basin0 = np.clip(_full_field(plate_drainage.get("inland_basin_x1000")), 0.0, 1.0)
    lake0 = np.clip(_full_field(plate_drainage.get("lake_candidate_x1000")), 0.0, 1.0)

    lats = np.linspace(90.0 - 90.0 / height, -90.0 + 90.0 / height, height, dtype=np.float32)
    lons = np.linspace(-180.0 + 180.0 / width, 180.0 - 180.0 / width, width, dtype=np.float32)
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    cos_lat = np.maximum(0.18, np.cos(np.radians(lat_grid))).astype(np.float32)

    # 1) Stop the island-world failure by adding continent-scale support before
    # any elevation profile is rebuilt.  This is a floor, not a universal Earth
    # ocean target: water-rich worlds can still be ocean-dominated, but not just
    # scattered ovals unless the controls explicitly force that in a future UI.
    requested_land = clamp(1.0 - float(getattr(hydrosphere, "ocean_fraction_target", terrain.ocean_fraction) or terrain.ocean_fraction), 0.03, 0.86)
    current_land = float(np.mean(land0)) if land0.size else requested_land
    target_land = clamp(max(current_land, requested_land, 0.24), 0.06, 0.58)
    if requested_land > 0.32:
        target_land = clamp(max(target_land, requested_land * 0.98), 0.06, 0.62)

    labels0, count0 = ndimage.label(land0, structure=np.array([[0,1,0],[1,1,1],[0,1,0]], dtype=np.uint8))
    sizes0 = np.bincount(labels0.ravel()) if count0 else np.array([0], dtype=np.int64)
    if sizes0.size:
        sizes0[0] = 0
    top_ids = [int(i) for i in np.argsort(sizes0)[::-1] if i > 0 and sizes0[i] > 0][:4]
    continent_support = ndimage.gaussian_filter(land0.astype(np.float32), sigma=max(3.0, width / 48.0), mode="wrap")
    if float(continent_support.max()) > 1.0e-6:
        continent_support = continent_support / float(continent_support.max())

    broad = np_rng.normal(0.0, 1.0, size=(height, width)).astype(np.float32)
    broad = ndimage.gaussian_filter(broad, sigma=max(4.0, width / 42.0), mode="wrap")
    if float(np.std(broad)) > 1.0e-6:
        broad = (broad - float(np.mean(broad))) / float(np.std(broad))

    lobe_field = np.zeros((height, width), dtype=np.float32)
    centroid_points: list[tuple[float, float]] = []
    for lid in top_ids:
        rr, cc = np.where(labels0 == lid)
        if rr.size:
            centroid_points.append((float(lons[int(np.median(cc))]), float(lats[int(np.median(rr))])))
    while len(centroid_points) < max(2, min(4, int(round(2.0 + 2.5 * target_land)))):
        centroid_points.append((rng.uniform(-180.0, 180.0), math.degrees(math.asin(rng.uniform(-0.86, 0.86)))))
    for lon0, lat0 in centroid_points[:5]:
        angle = rng.uniform(0.0, math.tau)
        major = rng.uniform(46.0, 94.0) * (0.86 + 0.82 * target_land)
        minor = rng.uniform(16.0, 38.0) * (0.90 + 0.42 * target_land)
        dlon = _wrapped_lon_delta_array(lon_grid, lon0) * cos_lat
        dlat = lat_grid - lat0
        along = dlon * math.cos(angle) + dlat * math.sin(angle)
        across = -dlon * math.sin(angle) + dlat * math.cos(angle)
        lobe = np.exp(-((along / max(4.0, major)) ** 2 + (across / max(3.5, minor)) ** 2))
        # Add attached secondary lobes so continents are compound, not ovals.
        for _ in range(rng.randint(2, 5)):
            off = rng.uniform(-0.75, 0.75) * major
            side = rng.uniform(-0.85, 0.85) * minor
            along2 = along - off
            across2 = across - side
            lobe = np.maximum(lobe, 0.75 * np.exp(-((along2 / max(4.0, major * rng.uniform(0.35, 0.72))) ** 2 + (across2 / max(3.0, minor * rng.uniform(0.45, 0.95))) ** 2)))
        lobe_field = np.maximum(lobe_field, lobe.astype(np.float32))

    continent_field = _norm(
        1.10 * lobe_field
        + 0.72 * continent_support
        + 0.58 * continental
        + 0.22 * craton
        - 0.34 * microplate
        - 0.22 * trench
        + 0.10 * broad
    )
    threshold = float(np.quantile(continent_field, clamp(1.0 - target_land, 0.025, 0.975)))
    land = (continent_field >= threshold) | (land0 & (continent_field >= threshold - 0.13))
    land |= (land0 & (continental > 0.36) & (microplate < 0.72))

    structure = np.array([[0,1,0],[1,1,1],[0,1,0]], dtype=np.uint8)
    land = ndimage.binary_closing(land, structure=structure, iterations=3)
    land = ndimage.binary_fill_holes(land)
    land = _roughen_structural_coastlines(
        rng,
        land.astype(bool),
        continent_field.astype(np.float32),
        threshold,
        np_rng,
        strength=0.34 + 0.16 * clamp(float(controls.get("coastline_complexity", 0.50) or 0.50), 0.0, 1.0),
    )
    land = ndimage.binary_closing(land, structure=structure, iterations=1)
    land = _remove_one_cell_land_water_needles(land)

    # Keep a controlled island budget and force most land into continent-scale
    # bodies.  This directly targets the "no large continents" complaint.
    labels, count = ndimage.label(land, structure=structure)
    if count:
        sizes = np.bincount(labels.ravel()); sizes[0] = 0
        land_cells = max(1.0, float(np.sum(sizes)))
        keep = np.zeros_like(sizes, dtype=bool)
        ranked = [int(i) for i in np.argsort(sizes)[::-1] if i > 0 and sizes[i] > 0]
        continent_budget = max(1, min(4, int(round(1.5 + 6.0 * target_land))))
        for lid in ranked[:continent_budget]:
            if sizes[lid] / land_cells > 0.045 or sizes[lid] / total_cells > 0.006:
                keep[lid] = True
        if ranked:
            keep[ranked[0]] = True
        island_budget = max(3, min(14, int(round(4 + 12 * clamp(float(controls.get("island_density", 0.35) or 0.35), 0.0, 1.0)))))
        kept_islands = 0
        for lid in ranked[continent_budget:]:
            comp = labels == lid
            if not np.any(comp):
                continue
            support = float(np.mean(0.55 * microplate[comp] + 0.30 * seamount[comp] + 0.25 * convergence[comp]))
            area_land = float(sizes[lid]) / land_cells
            if kept_islands < island_budget and support > 0.34 and area_land > 0.0012:
                keep[lid] = True
                kept_islands += 1
        land = keep[labels]
        land = ndimage.binary_closing(land, structure=structure, iterations=1)

    # 2) Rebuild elevation with explicit non-concentric landforms.  Existing
    # plate fields remain inputs, but we add visible belts and basins at full
    # resolution so maps show mountains, plateaus, plains, rifts, valleys, and
    # lake-friendly depressions.
    ocean = ~land
    coast = land & ((~np.roll(land, 1, axis=1)) | (~np.roll(land, -1, axis=1)) | (~np.roll(land, 1, axis=0)) | (~np.roll(land, -1, axis=0)))
    land_dist = ndimage.distance_transform_edt(land).astype(np.float32)
    ocean_dist = ndimage.distance_transform_edt(ocean).astype(np.float32)
    inland = np.clip(land_dist / max(3.0, width / 22.0), 0.0, 1.0) * land.astype(np.float32)

    def _curve_belts(count: int, width_deg_range: tuple[float, float], length_range: tuple[float, float], *, coast_bias: bool = False):
        field = np.zeros((height, width), dtype=np.float32)
        land_points = np.argwhere(land)
        coast_points = np.argwhere(coast) if np.any(coast) else land_points
        if land_points.size == 0:
            return field
        for _ in range(count):
            pts = coast_points if (coast_bias and coast_points.size and rng.random() < 0.70) else land_points
            rr, cc = pts[rng.randrange(len(pts))]
            lon0 = float(lons[int(cc)]); lat0 = float(lats[int(rr)])
            angle = rng.uniform(0.0, math.tau)
            belt_width = rng.uniform(*width_deg_range)
            belt_length = rng.uniform(*length_range)
            dlon = _wrapped_lon_delta_array(lon_grid, lon0) * cos_lat
            dlat = lat_grid - lat0
            along = dlon * math.cos(angle) + dlat * math.sin(angle)
            across = -dlon * math.sin(angle) + dlat * math.cos(angle)
            wave = math.sin(rng.uniform(0.0, math.tau)) + np.sin(np.radians(along * rng.uniform(1.2, 3.4) + rng.uniform(-180.0, 180.0))) * rng.uniform(1.6, 5.5)
            dist = across + wave
            belt = np.exp(-((dist / max(0.6, belt_width)) ** 2)) * np.exp(-((along / max(8.0, belt_length)) ** 2))
            # Break belts into ranges with gaps rather than one uniform stripe.
            beads = 0.55 + 0.45 * np.sin(np.radians(along * rng.uniform(2.0, 5.5) + rng.uniform(-180.0, 180.0)))
            field = np.maximum(field, (belt * np.clip(beads, 0.12, 1.0)).astype(np.float32))
        return _norm(field) * land.astype(np.float32)

    mountain_belts = _curve_belts(max(4, min(9, int(round(5 + width / 420)))), (1.2, 4.7), (28.0, 105.0), coast_bias=True)
    rift_belts = _curve_belts(max(2, min(6, int(round(2 + width / 620)))), (0.9, 3.4), (24.0, 92.0), coast_bias=False)
    mountain_field = _norm(0.54 * mountain_belts + 0.48 * orogeny0 + 0.38 * arc0 + 0.30 * convergence + 0.20 * plateau0) * land.astype(np.float32)
    rift_field = _norm(0.60 * rift_belts + 0.46 * rift0 + 0.34 * divergence + 0.18 * transform) * land.astype(np.float32)
    plateau_field = _norm(
        ndimage.gaussian_filter(mountain_field, sigma=max(2.2, width / 95.0), mode="wrap") * (0.40 + 0.75 * inland)
        + 0.42 * plateau0
        + 0.26 * craton
        + 0.12 * np.maximum(0.0, broad)
    ) * land.astype(np.float32)
    foreland_shadow = np.clip(ndimage.gaussian_filter(mountain_field, sigma=max(3.0, width / 55.0), mode="wrap") - 0.52 * mountain_field, 0.0, 1.0)
    plain_field = _norm(
        0.62 * foreland_shadow
        + 0.50 * basin0
        + 0.48 * sediment0
        + 0.30 * inland_basin0
        + 0.20 * coastal_plain0
    ) * land.astype(np.float32)

    noise_fine = np_rng.normal(0.0, 1.0, size=(height, width)).astype(np.float32)
    noise_fine = ndimage.gaussian_filter(noise_fine, sigma=max(0.65, width / 920.0), mode="wrap")
    if float(np.std(noise_fine)) > 1.0e-6:
        noise_fine = (noise_fine - float(np.mean(noise_fine))) / float(np.std(noise_fine))
    noise_mid = np_rng.normal(0.0, 1.0, size=(height, width)).astype(np.float32)
    noise_mid = ndimage.gaussian_filter(noise_mid, sigma=max(1.6, width / 260.0), mode="wrap")
    if float(np.std(noise_mid)) > 1.0e-6:
        noise_mid = (noise_mid - float(np.mean(noise_mid))) / float(np.std(noise_mid))

    corridor_texture = np.exp(-np.abs(0.62 * noise_fine + 0.44 * noise_mid) / 0.55).astype(np.float32)
    valley_field = _norm((0.48 * valley0 + 0.52 * corridor_texture) * (0.36 + 0.52 * plain_field + 0.35 * mountain_field + 0.35 * rift_field) * land.astype(np.float32))
    lake_field = _norm((0.48 * lake0 + 0.52 * plain_field + 0.54 * rift_field + 0.34 * valley_field) * (0.28 + 0.82 * inland) * (1.0 - 0.55 * mountain_field) * land.astype(np.float32))
    # Make some basins coherent enough for lakes to be seen, not just single cells.
    lake_field = _norm(ndimage.gaussian_filter(lake_field, sigma=max(0.95, width / 540.0), mode="wrap") * land.astype(np.float32))

    terrain_detail = _norm(0.38 * mountain_field + 0.25 * plateau_field + 0.18 * np.abs(noise_fine) + 0.15 * np.abs(noise_mid) - 0.24 * plain_field) * land.astype(np.float32)

    new_elev = elev0.copy()
    added_land = land & (~land0)
    retained_land = land & land0
    # Reset new land and partially de-concentric old land before adding landforms.
    base_land = (
        90.0
        + 270.0 * continental
        + 190.0 * craton
        + 160.0 * np.clip(broad, -0.2, 1.9)
        + 130.0 * inland
        + 110.0 * terrain_detail
    )
    new_elev[added_land] = base_land[added_land]
    new_elev[retained_land] = 0.70 * new_elev[retained_land] + 0.30 * base_land[retained_land]

    relief_gain = 0.95 + 0.45 * clamp(float(controls.get("mountain_belt_strength", 0.70) or 0.70), 0.0, 3.0)
    new_elev[land] += (1550.0 * relief_gain * mountain_field[land])
    new_elev[land] += 760.0 * plateau_field[land]
    new_elev[land] -= 430.0 * plain_field[land]
    new_elev[land] -= 620.0 * rift_field[land]
    new_elev[land] -= 220.0 * valley_field[land]
    new_elev[land] -= 420.0 * lake_field[land]
    new_elev[land] += (54.0 + 132.0 * terrain_detail[land]) * noise_fine[land] * (1.0 - 0.68 * plain_field[land])

    # Gangetic-style inland/coastal plains: broad, low, smooth belts downstream
    # of mountain fronts.  These are plains, not just coast-distance ramps.
    plain_target = 35.0 + 185.0 * np.clip(noise_mid, -0.4, 1.0) + 95.0 * inland + 80.0 * sediment0
    plain_blend = np.clip(0.55 * plain_field + 0.25 * coastal_plain0, 0.0, 0.82) * land.astype(np.float32)
    new_elev[land] = new_elev[land] * (1.0 - plain_blend[land]) + plain_target[land] * plain_blend[land]
    # Plateaus should read as high, broad surfaces with rough edges.
    plateau_target = 720.0 + 880.0 * plateau_field + 260.0 * np.clip(noise_mid, -0.3, 1.4)
    plateau_blend = np.clip(plateau_field * (1.0 - 0.55 * mountain_field), 0.0, 0.60) * land.astype(np.float32)
    new_elev[land] = new_elev[land] * (1.0 - plateau_blend[land]) + plateau_target[land] * plateau_blend[land]
    new_elev[land] = np.maximum(new_elev[land], 1.0 + 3.0 * valley_field[land])

    # 3) Remove universal continental shelf halos.  Ocean cells are made steep by
    # default; only segmented passive/rifted margins adjacent to continent-sized
    # carriers are allowed to be shallow shelves.
    labels, count = ndimage.label(land, structure=structure)
    carrier = np.zeros_like(land, dtype=bool)
    if count:
        sizes = np.bincount(labels.ravel()); sizes[0] = 0
        land_cells = max(1.0, float(np.sum(sizes)))
        for lid in [int(i) for i in np.argsort(sizes)[::-1] if i > 0 and sizes[i] > 0][:5]:
            if sizes[lid] / land_cells > 0.10 or sizes[lid] / total_cells > 0.009:
                carrier |= labels == lid
        if not np.any(carrier) and sizes.size > 1:
            carrier |= labels == int(np.argmax(sizes))
    ocean = ~land
    dist_to_land = ndimage.distance_transform_edt(ocean).astype(np.float32)
    dist_to_carrier = ndimage.distance_transform_edt(~carrier).astype(np.float32)
    near_any_land = ocean & (dist_to_land <= max(10.0, width / 52.0))
    near_carrier = ocean & (dist_to_carrier <= max(12.0, width / 44.0))
    shelf_radius = max(3.5, width / 115.0) * (0.65 + 0.55 * clamp(float(controls.get("shelf_width_factor", 0.45) or 0.45), 0.0, 2.0))
    shelf_noise = np_rng.normal(0.0, 1.0, size=(height, width)).astype(np.float32)
    shelf_noise = ndimage.gaussian_filter(shelf_noise, sigma=max(2.0, width / 145.0), mode="wrap")
    if float(np.std(shelf_noise)) > 1.0e-6:
        shelf_noise = (shelf_noise - float(np.mean(shelf_noise))) / float(np.std(shelf_noise))
    margin_support = np.clip(0.66 * passive0 + 0.52 * rifted0 + 0.24 * coastal_plain0 + 0.22 * sediment0 - 0.72 * active0 - 0.58 * trench, 0.0, 1.0)
    segmented = (shelf_noise > -0.05).astype(np.float32) * (0.55 + 0.45 * _norm(shelf_noise))
    shelf_field = np.exp(-dist_to_carrier / max(1.0, shelf_radius)).astype(np.float32) * segmented * (0.22 + 0.94 * margin_support)
    shelf_field *= near_carrier.astype(np.float32)
    shelf_field[(microplate > 0.28) & (continental < 0.62)] *= 0.015
    shelf_field[active0 > 0.30] *= 0.05
    shelf_field[shelf_field < 0.18] = 0.0
    shelf_field = np.clip(shelf_field, 0.0, 1.0)

    # Deep default nearshore for islands and active margins: this is the actual
    # anti-halo fix.  Existing shallow water from earlier passes is overwritten.
    deep_nearshore = -820.0 - 540.0 * np.clip(dist_to_land / max(1.0, width / 210.0), 0.0, 1.0) - 1450.0 * np.clip(dist_to_land / max(4.0, width / 42.0), 0.0, 1.0) ** 0.82
    ocean_base = (
        -3500.0
        - 700.0 * np.clip(dist_to_land / max(10.0, width / 30.0), 0.0, 1.0)
        - 1050.0 * trench
        + 740.0 * ridge
        + 430.0 * seamount
    )
    ocean_target_elev = np.minimum(ocean_base, deep_nearshore)
    new_elev[ocean] = np.minimum(new_elev[ocean], ocean_target_elev[ocean])
    shelf_cells = ocean & (shelf_field > 0.0)
    if np.any(shelf_cells):
        shelf_depth = -70.0 - 520.0 * np.clip(dist_to_carrier / max(1.0, shelf_radius), 0.0, 1.0) ** 1.35 - 80.0 * np.clip(shelf_noise, -0.2, 1.4)
        new_elev[shelf_cells] = np.maximum(new_elev[shelf_cells], shelf_depth[shelf_cells])
    new_elev[ocean] = np.minimum(new_elev[ocean], -18.0)

    # Recompute final bookkeeping.
    new_elev_i = np.rint(np.clip(new_elev, -11000, 10000)).astype(np.int32)
    terrain.elevation_m = new_elev_i.astype(int).tolist()
    terrain.is_land = land.astype(bool).tolist()
    terrain.land_fraction = float(np.mean(land))
    terrain.ocean_fraction = 1.0 - terrain.land_fraction
    terrain.min_elevation_m = int(new_elev_i.min())
    terrain.max_elevation_m = int(new_elev_i.max())
    terrain.mean_land_elevation_m = float(np.mean(new_elev_i[land])) if np.any(land) else 0.0
    terrain.mean_ocean_depth_m = float(np.mean(new_elev_i[ocean])) if np.any(ocean) else 0.0
    terrain.source = str(getattr(terrain, "source", "plate_tectonic_v1")) + "; Plate Terrain 13 user-feedback corrective landforms"

    labels_final, count_final = ndimage.label(land, structure=structure)
    sizes_final = np.bincount(labels_final.ravel()) if count_final else np.array([0], dtype=np.int64)
    if sizes_final.size:
        sizes_final[0] = 0
    land_cells_final = int(np.sum(sizes_final)) if sizes_final.size else 0
    largest_share = float(np.max(sizes_final) / max(1, land_cells_final)) if land_cells_final else 0.0
    shelf_ocean_share = float(np.mean((shelf_field > 0.18)[ocean])) if np.any(ocean) else 0.0
    nearshore_shallow_share = float(np.mean((new_elev_i[near_any_land] > -500))) if np.any(near_any_land) else 0.0
    lake_share = float(np.mean((lake_field > 0.48)[land])) if np.any(land) else 0.0
    mountain_share = float(np.mean((mountain_field > 0.42)[land])) if np.any(land) else 0.0
    plain_share = float(np.mean((plain_field > 0.42)[land])) if np.any(land) else 0.0
    plateau_share = float(np.mean((plateau_field > 0.38)[land])) if np.any(land) else 0.0
    rift_share = float(np.mean((rift_field > 0.42)[land])) if np.any(land) else 0.0

    meta = {
        "applied": True,
        "stage": "plate-terrain-13-user-feedback-correction",
        "requested_land_fraction": round(float(requested_land), 4),
        "land_fraction_before": round(float(current_land), 4),
        "land_fraction_after": round(float(terrain.land_fraction), 4),
        "largest_landmass_share_of_land_after": round(largest_share, 4),
        "landmass_count_after": int(count_final),
        "shelf_ocean_share_after": round(shelf_ocean_share, 4),
        "nearshore_shallow_ocean_share_after": round(nearshore_shallow_share, 4),
        "mountain_land_share_after": round(mountain_share, 4),
        "plateau_land_share_after": round(plateau_share, 4),
        "plain_land_share_after": round(plain_share, 4),
        "rift_land_share_after": round(rift_share, 4),
        "lake_candidate_land_share_after": round(lake_share, 4),
        "mean_abs_elevation_delta_m": round(float(np.mean(np.abs(new_elev_i.astype(np.float32) - elev0))), 2),
        "fixes": [
            "universal shallow shelf halos overwritten with steep default nearshore bathymetry",
            "shelves limited to segmented passive/rifted margins adjacent to continent-sized carriers",
            "minimum visible continent-scale land floor applied to avoid pure island-world output",
            "full-resolution mountain belts, plateaus, plains, rifts, valleys, and lake basins added",
            "coastline mask roughened structurally rather than by low-elevation halos",
        ],
    }
    try:
        diagnostics = terrain.terrain_diagnostics if isinstance(terrain.terrain_diagnostics, dict) else {}
        plate_diag = diagnostics.setdefault("plate_tectonic_v1", {})
        plate_diag["user_feedback_correction_u13"] = meta
        terrain.terrain_diagnostics = diagnostics
    except Exception:
        pass

    return {
        "metadata": meta,
        "mountain_strength_x1000": _x1000(mountain_field),
        "plateau_x1000": _x1000(plateau_field),
        "rift_x1000": _x1000(rift_field),
        "valley_corridor_x1000": _x1000(valley_field),
        "inland_basin_x1000": _x1000(np.maximum(inland_basin0, plain_field)),
        "lake_candidate_x1000": _x1000(np.maximum(lake0, lake_field)),
        "terrain_detail_x1000": _x1000(terrain_detail),
        "shelf_width_x1000": _x1000(shelf_field),
    }

def _build_plate_tectonic_v1_final_integration_qa(
    terrain: TerrainMap,
    plate_setup: dict,
    plate_relief: dict,
    plate_coasts: dict,
    geology: GeologyState,
    controls: dict,
) -> dict:
    """Build Plate Terrain 10 final integration/readiness diagnostics.

    This scores where native plate fields now drive the terrain, where weak legacy/texture fallback remains, and where hydrology can safely consume the terrain in later updates.
    """
    try:
        import numpy as np
        from PIL import Image
        from scipy import ndimage
    except ImportError as exc:
        raise RuntimeError("NumPy, Pillow, and SciPy are required for plate terrain diagnostics. Install with: pip install -r requirements.txt") from exc

    elev = np.asarray(terrain.elevation_m, dtype=np.float32)
    land = np.asarray(terrain.is_land, dtype=bool)
    ocean = ~land
    height, width = elev.shape

    def _full_float(value, *, scale: float = 1000.0, default: float = 0.0):
        if value is None:
            return np.full((height, width), float(default), dtype=np.float32)
        arr = np.asarray(value, dtype=np.float32)
        if arr.size == 0:
            return np.full((height, width), float(default), dtype=np.float32)
        if scale:
            arr = arr / scale
        if arr.shape != (height, width):
            img = Image.fromarray(arr.astype(np.float32), mode="F")
            arr = np.asarray(img.resize((width, height), Image.Resampling.BICUBIC), dtype=np.float32)
        return np.nan_to_num(arr, nan=default, posinf=default, neginf=default).astype(np.float32)

    def _full_class(value, default: int = 0):
        if value is None:
            return np.full((height, width), int(default), dtype=np.int32)
        arr = np.asarray(value, dtype=np.int32)
        if arr.size == 0:
            return np.full((height, width), int(default), dtype=np.int32)
        if arr.shape != (height, width):
            img = Image.fromarray(arr.astype(np.int16), mode="I;16")
            arr = np.asarray(img.resize((width, height), Image.Resampling.NEAREST), dtype=np.int32)
        return arr

    continental = np.clip(_full_float(plate_setup.get("continental_crust_x1000")), 0.0, 1.0)
    microplate = np.clip(_full_float(plate_setup.get("microplate_x1000")), 0.0, 1.0)
    speed = np.clip(_full_float(plate_setup.get("speed_x1000")), 0.0, 1.0)
    convergence = np.clip(_full_float(plate_setup.get("convergence_x1000")), 0.0, 1.0)
    divergence = np.clip(_full_float(plate_setup.get("divergence_x1000")), 0.0, 1.0)
    transform = np.clip(_full_float(plate_setup.get("transform_x1000")), 0.0, 1.0)
    ocean_age = np.clip(_full_float(plate_setup.get("ocean_crust_age_x1000")), 0.0, 1.0)
    ridge = np.clip(_full_float(plate_setup.get("mid_ocean_ridge_x1000")), 0.0, 1.0)
    trench = np.clip(_full_float(plate_setup.get("trench_x1000")), 0.0, 1.0)
    fracture = np.clip(_full_float(plate_setup.get("fracture_zone_x1000")), 0.0, 1.0)
    seamount = np.clip(_full_float(plate_setup.get("seamount_x1000")), 0.0, 1.0)
    boundary_class = _full_class(plate_setup.get("boundary_class"))
    margin_class = _full_class(plate_coasts.get("margin_class"))

    orogeny = np.clip(_full_float(plate_relief.get("orogeny_strength_x1000")), 0.0, 1.0)
    volcanic_arc = np.clip(_full_float(plate_relief.get("volcanic_arc_x1000")), 0.0, 1.0)
    rift = np.clip(_full_float(plate_relief.get("continental_rift_x1000")), 0.0, 1.0)
    foreland = np.clip(_full_float(plate_relief.get("foreland_basin_x1000")), 0.0, 1.0)
    shield = np.clip(_full_float(plate_relief.get("craton_shield_x1000")), 0.0, 1.0)
    terrane = np.clip(_full_float(plate_relief.get("accreted_terrane_x1000")), 0.0, 1.0)
    plateau = np.clip(_full_float(plate_relief.get("plateau_uplift_x1000")), 0.0, 1.0)
    relief_delta = _full_float(plate_relief.get("relief_delta_m"), scale=1.0)

    shelf = np.clip(_full_float(plate_coasts.get("shelf_width_x1000")), 0.0, 1.0)
    active_margin = np.clip(_full_float(plate_coasts.get("active_margin_x1000")), 0.0, 1.0)
    passive_margin = np.clip(_full_float(plate_coasts.get("passive_margin_x1000")), 0.0, 1.0)
    rifted_margin = np.clip(_full_float(plate_coasts.get("rifted_margin_x1000")), 0.0, 1.0)
    island_arc = np.clip(_full_float(plate_coasts.get("island_arc_x1000")), 0.0, 1.0)
    coastal_plain = np.clip(_full_float(plate_coasts.get("coastal_plain_x1000")), 0.0, 1.0)
    coast_rugged = np.clip(_full_float(plate_coasts.get("coast_ruggedness_x1000")), 0.0, 1.0)
    coast_delta = _full_float(plate_coasts.get("coast_delta_m"), scale=1.0)

    valley = np.clip(_full_float(getattr(terrain, "terrain_valley_corridor_x1000", None)), 0.0, 1.0)
    basin = np.clip(_full_float(getattr(terrain, "terrain_basin_field_x1000", None)), 0.0, 1.0)
    deposition = np.clip(_full_float(getattr(terrain, "terrain_deposition_field_x1000", None)), 0.0, 1.0)
    maturity = np.clip(_full_float(getattr(terrain, "terrain_maturity_x1000", None)), 0.0, 1.0)

    # Native coverage means a cell has a plate-native reason for its terrain or
    # coast/ocean interpretation. It is deliberately broader than only direct
    # elevation modification because plate mode is still replacing pieces stage by stage.
    plate_motion_signal = np.clip(0.45 * speed + 0.35 * np.maximum.reduce([convergence, divergence, transform]) + 0.20 * (boundary_class > 0), 0.0, 1.0)
    plate_crust_signal = np.clip(0.62 * continental + 0.24 * microplate + 0.14 * shield, 0.0, 1.0)
    plate_relief_signal = np.clip(0.34 * orogeny + 0.22 * volcanic_arc + 0.16 * rift + 0.12 * foreland + 0.10 * terrane + 0.06 * plateau + np.clip(np.abs(relief_delta) / 1200.0, 0.0, 1.0) * 0.24, 0.0, 1.0)
    plate_ocean_signal = np.clip(0.28 * ocean_age + 0.28 * ridge + 0.22 * trench + 0.12 * fracture + 0.10 * seamount, 0.0, 1.0)
    plate_coast_signal = np.clip(0.25 * shelf + 0.20 * active_margin + 0.18 * passive_margin + 0.15 * rifted_margin + 0.12 * island_arc + 0.10 * coast_rugged + np.clip(np.abs(coast_delta) / 280.0, 0.0, 1.0) * 0.18, 0.0, 1.0)

    land_integration = np.clip(0.28 * plate_crust_signal + 0.26 * plate_motion_signal + 0.30 * plate_relief_signal + 0.16 * plate_coast_signal, 0.0, 1.0)
    ocean_integration = np.clip(0.22 * plate_motion_signal + 0.48 * plate_ocean_signal + 0.20 * plate_coast_signal + 0.10 * np.clip(1.0 - continental, 0.0, 1.0), 0.0, 1.0)
    integration = np.where(land, land_integration, ocean_integration).astype(np.float32)

    # Legacy dependency should be high where the present plate backend still has
    # little native support for the final mask/elevation. It is not inherently a
    # problem: it identifies where future plate-owned foundation work matters most.
    legacy_dependency = np.clip(1.0 - integration, 0.0, 1.0)
    legacy_dependency[land & (continental > 0.55)] *= 0.70
    legacy_dependency[ocean & (ocean_age > 0.35)] *= 0.72
    legacy_dependency = np.clip(legacy_dependency, 0.0, 1.0)

    # Hydrology readiness combines plate valleys/rifts/basins with existing Stage
    # 3C.5 fields. It is intentionally land-focused; oceans keep low values.
    gy, gx = np.gradient(elev)
    slope = np.sqrt(gx * gx + gy * gy)
    land_slope = slope[land]
    slope_norm = np.zeros_like(elev, dtype=np.float32)
    if land_slope.size:
        slope_norm = np.clip(slope / max(1.0, float(np.percentile(land_slope, 92))), 0.0, 1.0)
    coast = land & (
        (~np.vstack((land[0:1, :], land[:-1, :]))) |
        (~np.vstack((land[1:, :], land[-1:, :]))) |
        (~np.roll(land, 1, axis=1)) |
        (~np.roll(land, -1, axis=1))
    )
    coast_influence = ndimage.gaussian_filter(coast.astype(np.float32), sigma=max(1.0, min(8.0, width / 384.0)), mode="wrap")
    hydrology_readiness = np.clip(
        0.24 * np.maximum(valley, rift)
        + 0.18 * np.maximum(basin, foreland)
        + 0.14 * deposition
        + 0.12 * coastal_plain
        + 0.12 * np.clip(slope_norm * 0.7 + np.maximum(orogeny, volcanic_arc) * 0.3, 0.0, 1.0)
        + 0.10 * coast_influence
        + 0.10 * maturity,
        0.0,
        1.0,
    ) * land.astype(np.float32)

    # Problem classes for final inspection. 0 = no specific issue, 1 = legacy
    # dependency, 2 = weak plate boundary expression, 3 = weak hydrology readiness,
    # 4 = shelf/active-margin conflict, 5 = ocean-floor underexpression.
    problem = np.zeros((height, width), dtype=np.uint8)
    weak_bound = (boundary_class > 0) & (plate_motion_signal < 0.18)
    weak_hydro = land & (hydrology_readiness < 0.23) & (integration > 0.20)
    shelf_conflict = ocean & (shelf > 0.40) & (active_margin > 0.28)
    weak_ocean = ocean & (plate_ocean_signal < 0.18) & (integration < 0.45)
    high_legacy = legacy_dependency > 0.66
    problem[high_legacy] = 1
    problem[weak_bound] = 2
    problem[weak_hydro] = 3
    problem[shelf_conflict] = 4
    problem[weak_ocean] = 5

    def _mean_active(arr, mask):
        vals = arr[mask]
        return float(np.mean(vals)) if vals.size else 0.0

    metadata = {
        "backend_stage": "plate_domain_foundation_v1",
        "native_plate_integration_mean": round(float(np.mean(integration)), 3),
        "native_plate_land_integration_mean": round(_mean_active(integration, land), 3),
        "native_plate_ocean_integration_mean": round(_mean_active(integration, ocean), 3),
        "legacy_dependency_mean": round(float(np.mean(legacy_dependency)), 3),
        "legacy_dependency_strong_share": round(float(np.mean(legacy_dependency > 0.66)), 4),
        "plate_hydrology_readiness_mean_land": round(_mean_active(hydrology_readiness, land), 3),
        "plate_hydrology_readiness_good_land_share": round(float(np.mean(hydrology_readiness[land] > 0.42)) if np.any(land) else 0.0, 4),
        "plate_problem_legacy_dependency_share": round(float(np.mean(problem == 1)), 4),
        "plate_problem_weak_boundary_share": round(float(np.mean(problem == 2)), 4),
        "plate_problem_weak_hydrology_share": round(float(np.mean(problem == 3)), 4),
        "plate_problem_shelf_active_conflict_share": round(float(np.mean(problem == 4)), 4),
        "plate_problem_ocean_floor_gap_share": round(float(np.mean(problem == 5)), 4),
    }

    return {
        "backend_integration_x1000": np.rint(np.clip(integration, 0.0, 1.0) * 1000).astype(np.int16).tolist(),
        "hydrology_readiness_x1000": np.rint(np.clip(hydrology_readiness, 0.0, 1.0) * 1000).astype(np.int16).tolist(),
        "legacy_dependency_x1000": np.rint(np.clip(legacy_dependency, 0.0, 1.0) * 1000).astype(np.int16).tolist(),
        "problem_class": problem.astype(int).tolist(),
        "metadata": metadata,
    }



def _apply_plate_tectonic_v1_feedback_cleanup_u16(
    rng: random.Random,
    terrain: TerrainMap,
    plate_setup: dict,
    plate_relief: dict,
    plate_coasts: dict,
    geology: GeologyState,
    controls: dict,
) -> dict:
    """Final user-feedback cleanup pass for plate terrain.

    This pass is intentionally practical rather than a full geodynamic model. It
    uses the native plate fields already generated by plate_tectonic_v1 to fix
    artifacts reported in map review: same-width shelves, shelf halos around
    islands, coastal cliff jumps, polar land distortion, high snake-like islands,
    weak plateaus, weak volcanic island arcs, huge lake-candidate fields, and
    straight coastline diagnostics.  Longitude is treated as wrapped for all
    distance fields so the map seam is not a hard simulation edge.
    """
    try:
        import numpy as np
        from scipy import ndimage
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("NumPy, SciPy, and Pillow are required for plate terrain feedback cleanup. Install with: pip install -r requirements.txt") from exc

    elev0 = np.asarray(terrain.elevation_m, dtype=np.float32)
    land = np.asarray(terrain.is_land, dtype=bool).copy()
    height, width = elev0.shape
    if height <= 0 or width <= 0:
        return {"metadata": {"applied": False, "reason": "empty terrain"}}
    new_elev = elev0.copy()
    total_cells = max(1, height * width)
    structure = np.ones((3, 3), dtype=np.uint8)
    np_rng = np.random.default_rng(int(rng.randrange(1, 2**31 - 1)))

    def _resize_arr(value, *, dtype=np.float32, default=0.0, scale=1000.0):
        if value is None:
            return np.full((height, width), default, dtype=dtype)
        arr = np.asarray(value, dtype=np.float32)
        if arr.ndim != 2 or arr.size == 0:
            return np.full((height, width), default, dtype=dtype)
        if arr.shape != (height, width):
            img = Image.fromarray(arr.astype(np.float32), mode="F")
            arr = np.asarray(img.resize((width, height), Image.Resampling.NEAREST), dtype=np.float32)
        if scale:
            arr = arr / float(scale)
        return arr.astype(dtype, copy=False)

    def _resize_class(value, default=0):
        if value is None:
            return np.full((height, width), default, dtype=np.int16)
        arr = np.asarray(value, dtype=np.int16)
        if arr.ndim != 2 or arr.size == 0:
            return np.full((height, width), default, dtype=np.int16)
        if arr.shape != (height, width):
            img = Image.fromarray(arr.astype(np.int16), mode="I;16")
            arr = np.asarray(img.resize((width, height), Image.Resampling.NEAREST), dtype=np.int16)
        return arr

    def _x1000(arr):
        return np.rint(np.clip(arr, 0.0, 1.0) * 1000.0).astype(np.int16).astype(int).tolist()

    def _xwrap_distance_to_true(target: np.ndarray) -> np.ndarray:
        target = np.asarray(target, dtype=bool)
        if not bool(target.any()):
            return np.full((height, width), float(width + height), dtype=np.float32)
        # distance_transform_edt measures distance to the nearest zero. Triplicate
        # columns so the left/right map seam behaves like continuous longitude.
        tiled = np.concatenate([~target, ~target, ~target], axis=1)
        dist = ndimage.distance_transform_edt(tiled)
        return dist[:, width:2 * width].astype(np.float32)

    def _norm(arr):
        arr = np.asarray(arr, dtype=np.float32)
        pos = arr[np.isfinite(arr)]
        if pos.size == 0:
            return np.zeros_like(arr, dtype=np.float32)
        lo = float(np.percentile(pos, 4))
        hi = float(np.percentile(pos, 96))
        if hi <= lo + 1.0e-6:
            return np.zeros_like(arr, dtype=np.float32)
        return np.clip((arr - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)

    continental = np.clip(_resize_arr(plate_setup.get("continental_crust_x1000")), 0.0, 1.0)
    microplate = np.clip(_resize_arr(plate_setup.get("microplate_x1000")), 0.0, 1.0)
    convergence = np.clip(_resize_arr(plate_setup.get("convergence_x1000")), 0.0, 1.0)
    divergence = np.clip(_resize_arr(plate_setup.get("divergence_x1000")), 0.0, 1.0)
    transform = np.clip(_resize_arr(plate_setup.get("transform_x1000")), 0.0, 1.0)
    trench = np.clip(_resize_arr(plate_setup.get("trench_x1000")), 0.0, 1.0)
    ridge = np.clip(_resize_arr(plate_setup.get("mid_ocean_ridge_x1000")), 0.0, 1.0)
    seamount = np.clip(_resize_arr(plate_setup.get("seamount_x1000")), 0.0, 1.0)
    ocean_age = np.clip(_resize_arr(plate_setup.get("ocean_crust_age_x1000")), 0.0, 1.0)
    boundary_class = _resize_class(plate_setup.get("boundary_class"), default=0)

    orogeny0 = np.clip(_resize_arr(plate_relief.get("orogeny_strength_x1000")), 0.0, 1.0)
    arc0 = np.clip(_resize_arr(plate_relief.get("volcanic_arc_x1000")), 0.0, 1.0)
    rift0 = np.clip(_resize_arr(plate_relief.get("continental_rift_x1000")), 0.0, 1.0)
    foreland0 = np.clip(_resize_arr(plate_relief.get("foreland_basin_x1000")), 0.0, 1.0)
    plateau0 = np.clip(_resize_arr(plate_relief.get("plateau_uplift_x1000")), 0.0, 1.0)
    sediment0 = np.clip(_resize_arr(plate_relief.get("sedimentary_plain_x1000")), 0.0, 1.0)

    passive0 = np.clip(_resize_arr(plate_coasts.get("passive_margin_x1000")), 0.0, 1.0)
    rifted0 = np.clip(_resize_arr(plate_coasts.get("rifted_margin_x1000")), 0.0, 1.0)
    active0 = np.clip(_resize_arr(plate_coasts.get("active_margin_x1000")), 0.0, 1.0)
    coastal_plain0 = np.clip(_resize_arr(plate_coasts.get("coastal_plain_x1000")), 0.0, 1.0)
    island_arc0 = np.clip(_resize_arr(plate_coasts.get("island_arc_x1000")), 0.0, 1.0)
    shelf_existing = np.clip(_resize_arr(getattr(terrain, "terrain_shelf_width_x1000", None)), 0.0, 1.0)
    coast_rugged0 = np.clip(_resize_arr(plate_coasts.get("coast_ruggedness_x1000")), 0.0, 1.0)
    crust_class = _resize_class(getattr(terrain, "crust_type", None), default=0)
    island_origin = _resize_class(getattr(terrain, "terrain_island_origin_class", None), default=0)

    lats = np.linspace(90.0 - 90.0 / height, -90.0 + 90.0 / height, height, dtype=np.float32)
    lat_grid = np.repeat(lats[:, None], width, axis=1)
    abs_lat = np.abs(lat_grid)

    polar_removed_count = 0
    if bool(controls.get("suppress_polar_land", False)):
        polar_mask = land & (abs_lat >= 72.0)
        polar_removed_count = int(polar_mask.sum())
        if polar_removed_count:
            land[polar_mask] = False
            new_elev[polar_mask] = np.minimum(new_elev[polar_mask], -180.0 - 900.0 * np.clip((abs_lat[polar_mask] - 72.0) / 18.0, 0.0, 1.0))
            crust_class[polar_mask] = 0
            island_origin[polar_mask] = 0

    # Work out continent-sized carriers. Shelves should grow from these carriers,
    # not from every small island component.
    labels, count = ndimage.label(land, structure=structure)
    sizes = np.bincount(labels.ravel()) if count else np.asarray([0], dtype=np.int64)
    if sizes.size:
        sizes[0] = 0
    land_cells = max(1, int(sizes.sum()))
    carrier = np.zeros_like(land, dtype=bool)
    large_component_threshold = max(96, int(land_cells * 0.055), int(total_cells * 0.0025))
    for lid in [int(i) for i in np.argsort(sizes)[::-1] if i > 0 and sizes[i] >= large_component_threshold][:7]:
        carrier |= labels == lid
    if not bool(carrier.any()) and sizes.size > 1:
        carrier |= labels == int(np.argmax(sizes))

    dist_to_any_land = _xwrap_distance_to_true(land)
    dist_to_ocean = _xwrap_distance_to_true(~land)
    dist_to_carrier = _xwrap_distance_to_true(carrier)
    small_island = land & (~carrier)
    dist_to_small_island = _xwrap_distance_to_true(small_island)
    ocean = ~land

    # Add broken ocean-ocean volcanic arcs before shelf/bathymetry so they get
    # steep nearshore profiles instead of continental shelf halos.
    arc_added = np.zeros_like(land, dtype=bool)
    arc_support = np.clip(0.72 * arc0 + 0.42 * island_arc0 + 0.30 * convergence + 0.18 * seamount - 0.28 * continental - 0.18 * passive0, 0.0, 1.0)
    arc_noise = np_rng.normal(0.0, 1.0, size=(height, width)).astype(np.float32)
    arc_noise = ndimage.gaussian_filter(arc_noise, sigma=max(0.9, width / 900.0), mode="wrap")
    if float(np.std(arc_noise)) > 1.0e-6:
        arc_noise = (arc_noise - float(np.mean(arc_noise))) / float(np.std(arc_noise))
    oceanic_arc_band = ocean & (arc_support > 0.38) & (dist_to_carrier > max(4.0, width / 180.0)) & (abs_lat < 72.0)
    if bool(oceanic_arc_band.any()):
        score = arc_support + 0.18 * _norm(arc_noise) + 0.10 * seamount
        threshold = float(np.percentile(score[oceanic_arc_band], 78.0))
        arc_added = oceanic_arc_band & (score >= max(0.42, threshold))
        # Keep the islands broken/chained rather than a continuous snake.
        arc_added = ndimage.binary_opening(arc_added, structure=structure, iterations=1)
        arc_added = ndimage.binary_dilation(arc_added, structure=structure, iterations=1) & oceanic_arc_band
        max_arc_cells = max(0, int(total_cells * 0.0065))
        if int(arc_added.sum()) > max_arc_cells:
            cutoff = float(np.percentile(score[arc_added], 100.0 * (1.0 - max_arc_cells / max(1, int(arc_added.sum())))))
            arc_added &= score >= cutoff
        if bool(arc_added.any()):
            land[arc_added] = True
            ocean[arc_added] = False
            crust_class[arc_added] = 6
            island_origin[arc_added] = 3
            peak = np.clip(score, 0.0, 1.0)
            new_elev[arc_added] = np.maximum(new_elev[arc_added], 24.0 + 620.0 * peak[arc_added] + 1050.0 * np.clip(arc0[arc_added], 0.0, 1.0))

    # Recompute after adding arcs.
    labels, count = ndimage.label(land, structure=structure)
    sizes = np.bincount(labels.ravel()) if count else np.asarray([0], dtype=np.int64)
    if sizes.size:
        sizes[0] = 0
    land_cells = max(1, int(sizes.sum()))
    carrier = np.zeros_like(land, dtype=bool)
    for lid in [int(i) for i in np.argsort(sizes)[::-1] if i > 0 and sizes[i] >= large_component_threshold][:7]:
        carrier |= labels == lid
    if not bool(carrier.any()) and sizes.size > 1:
        carrier |= labels == int(np.argmax(sizes))
    small_island = land & (~carrier)
    ocean = ~land
    dist_to_any_land = _xwrap_distance_to_true(land)
    dist_to_ocean = _xwrap_distance_to_true(ocean)
    dist_to_carrier = _xwrap_distance_to_true(carrier)
    dist_to_small_island = _xwrap_distance_to_true(small_island)

    # Variable continental shelves: wide on passive/rifted/sediment-rich carrier
    # margins, narrow on active/subduction margins, almost absent around small
    # volcanic islands. A coarse texture changes width along the same margin.
    shelf_factor = clamp(float(controls.get("shelf_width_factor", 0.55) or 0.55), 0.0, 2.0)
    coarse = np_rng.normal(0.0, 1.0, size=(height, width)).astype(np.float32)
    coarse = ndimage.gaussian_filter(coarse, sigma=max(2.0, width / 135.0), mode="wrap")
    if float(np.std(coarse)) > 1.0e-6:
        coarse = (coarse - float(np.mean(coarse))) / float(np.std(coarse))
    shelf_texture = np.clip(0.58 + 0.26 * coarse, 0.12, 1.25)
    margin_support = np.clip(0.72 * passive0 + 0.58 * rifted0 + 0.34 * sediment0 + 0.22 * coastal_plain0 + 0.12 * shelf_existing - 0.82 * active0 - 0.64 * trench - 0.32 * island_arc0, 0.0, 1.0)
    base_width = max(3.6, width / 105.0) * (0.76 + 0.54 * shelf_factor)
    local_width = base_width * np.clip(0.20 + 2.65 * margin_support + 0.46 * shelf_texture, 0.10, 4.25)
    shelf_field = np.clip(1.0 - dist_to_carrier / np.maximum(1.0, local_width), 0.0, 1.0) ** 1.22
    carrier_gate = (dist_to_carrier <= dist_to_small_island + max(1.0, width / 420.0)).astype(np.float32)
    submerged_continent = (continental > 0.18) | (shelf_existing > 0.18) | (passive0 > 0.20) | (rifted0 > 0.18)
    shelf_field *= ocean.astype(np.float32) * carrier_gate * submerged_continent.astype(np.float32) * np.clip(0.20 + 0.94 * margin_support, 0.0, 1.0)
    shelf_field[(active0 > 0.42) & (dist_to_carrier > max(1.25, base_width * 0.22))] *= 0.06
    # Suppress halos around islands: allow only a tiny volcanic apron, not a broad
    # continental shelf ring.
    shelf_field[(dist_to_small_island < dist_to_carrier) & (dist_to_small_island < max(3.0, width / 190.0))] *= 0.03
    shelf_field[shelf_field < 0.055] = 0.0

    ocean = ~land
    if bool(ocean.any()):
        deep_ocean = -3650.0 - 860.0 * ocean_age - 760.0 * np.clip(dist_to_any_land / max(8.0, width / 30.0), 0.0, 1.0) - 1250.0 * trench + 620.0 * ridge + 360.0 * seamount
        steep_nearshore = -96.0 - 640.0 * np.clip(dist_to_any_land / max(1.6, width / 230.0), 0.0, 1.0) - 1500.0 * np.clip(dist_to_any_land / max(4.0, width / 58.0), 0.0, 1.0) ** 1.15
        new_elev[ocean] = np.minimum(new_elev[ocean], np.minimum(deep_ocean[ocean], steep_nearshore[ocean]))
        shelf_cells = ocean & (shelf_field > 0.0)
        if bool(shelf_cells.any()):
            t = np.clip(dist_to_carrier / np.maximum(1.0, local_width), 0.0, 1.0)
            passive_depth = -24.0 - 155.0 * t - 610.0 * (t ** 2.05)
            active_depth = -135.0 - 600.0 * t - 1500.0 * (t ** 1.35)
            active_factor = np.clip(active0 * 1.40, 0.0, 1.0)
            shelf_depth = passive_depth * (1.0 - active_factor) + active_depth * active_factor + 34.0 * shelf_texture
            new_elev[shelf_cells] = np.maximum(new_elev[shelf_cells], shelf_depth[shelf_cells])
        apron = ocean & (dist_to_small_island <= max(2.2, width / 360.0)) & (dist_to_small_island < dist_to_carrier)
        if bool(apron.any()):
            apron_depth = -95.0 - 760.0 * np.clip(dist_to_small_island / max(1.0, width / 420.0), 0.0, 1.0) ** 1.20 - 220.0 * arc0
            new_elev[apron] = np.maximum(new_elev[apron], apron_depth[apron])
        new_elev[ocean] = np.minimum(new_elev[ocean], -1.0)

    # Reduce coastal cliff jumps on passive/rifted/coastal-plain margins while
    # preserving steep active/subduction mountain coasts.
    land = new_elev >= 1.0
    ocean = ~land
    dist_to_ocean = _xwrap_distance_to_true(ocean)
    near_coast_land = land & (dist_to_ocean <= max(2.2, width / 240.0))
    if bool(near_coast_land.any()):
        passive_like = np.clip(passive0 + 0.68 * rifted0 + 0.52 * coastal_plain0 + 0.22 * sediment0 - 0.62 * active0 - 0.42 * orogeny0, 0.0, 1.0)
        land_profile_width = max(2.0, width / 260.0) * (0.9 + 1.4 * passive_like + 0.5 * coastal_plain0)
        coast_blend = np.clip(np.exp(-dist_to_ocean / np.maximum(1.0, land_profile_width)) * (0.20 + 0.68 * passive_like), 0.0, 0.78) * land.astype(np.float32)
        coastal_target = 8.0 + 42.0 * dist_to_ocean + 170.0 * coastal_plain0 + 520.0 * active0 + 420.0 * orogeny0 + 220.0 * coast_rugged0
        new_elev[near_coast_land] = new_elev[near_coast_land] * (1.0 - coast_blend[near_coast_land]) + coastal_target[near_coast_land] * coast_blend[near_coast_land]
        new_elev[near_coast_land] = np.maximum(new_elev[near_coast_land], 1.0)

    # Make plateaus visible: broad elevated low-slope areas behind collision and
    # active-margin belts, plus older interior uplands where cratons/forelands exist.
    rough = np_rng.normal(0.0, 1.0, size=(height, width)).astype(np.float32)
    rough = ndimage.gaussian_filter(rough, sigma=max(2.2, width / 170.0), mode="wrap")
    rough_n = _norm(rough)
    broad_orogen = ndimage.gaussian_filter(np.maximum(orogeny0, active0), sigma=max(1.4, width / 320.0), mode="wrap")
    plateau_field = np.clip(np.maximum(plateau0, broad_orogen * (0.62 + 0.32 * rough_n) - 0.48 * orogeny0) + 0.20 * foreland0 + 0.12 * continental, 0.0, 1.0) * land.astype(np.float32)
    plateau_gain = (260.0 + 980.0 * clamp(float(controls.get("plateau_strength", 0.55) or 0.55), 0.0, 1.5)) * plateau_field
    new_elev[land] += plateau_gain[land]

    # Add branching relief texture along mountain/arc systems so ranges no longer
    # read as fixed-width streaks. Branches are weaker than the main belts.
    branch_seed = np.clip(0.65 * orogeny0 + 0.36 * arc0 + 0.26 * convergence + 0.18 * transform, 0.0, 1.0)
    branch_noise = np_rng.normal(0.0, 1.0, size=(height, width)).astype(np.float32)
    branch_noise = ndimage.gaussian_filter(branch_noise, sigma=max(0.8, width / 760.0), mode="wrap")
    branch_field = np.clip(branch_seed * (0.55 + 0.42 * _norm(branch_noise)), 0.0, 1.0) * land.astype(np.float32)
    # Strong local highs, but avoid raising all belt edges uniformly.
    new_elev[land] += (90.0 + 310.0 * branch_field[land]) * branch_field[land]

    # Repair thin high snake-like islands and steep 2700m island edges. Keep true
    # volcanic peaks in the interior, but edges should fall to low coastal slopes.
    labels, count = ndimage.label(land, structure=structure)
    snake_components = 0
    snake_edge_cells = 0
    high_island_edge_cells = 0
    if count:
        sizes = np.bincount(labels.ravel()); sizes[0] = 0
        for lid in [int(i) for i in np.argsort(sizes)[::-1] if i > 0 and sizes[i] > 0]:
            comp = labels == lid
            size = int(sizes[lid])
            if size <= 0:
                continue
            if np.any(carrier & comp):
                continue
            rr, cc = np.where(comp)
            bbox_h = int(rr.max() - rr.min() + 1) if rr.size else 1
            bbox_w = int(cc.max() - cc.min() + 1) if cc.size else 1
            aspect = max(bbox_h, bbox_w) / max(1, min(bbox_h, bbox_w))
            mean_elev = float(np.mean(new_elev[comp]))
            comp_dist = ndimage.distance_transform_edt(comp)
            edge = comp & (comp_dist <= 1.35)
            high_edge = edge & (new_elev > 650.0)
            if bool(high_edge.any()):
                high_island_edge_cells += int(high_edge.sum())
                edge_cap = 70.0 + 430.0 * np.clip(arc0, 0.0, 1.0) + 260.0 * np.clip(seamount, 0.0, 1.0)
                new_elev[high_edge] = np.minimum(new_elev[high_edge], edge_cap[high_edge])
            if aspect >= 8.0 and size <= max(420, int(total_cells * 0.0065)) and mean_elev > 420.0:
                snake_components += 1
                weak_neck = comp & (comp_dist <= 1.05) & (arc0 < 0.34) & (seamount < 0.34)
                snake_edge_cells += int(weak_neck.sum())
                new_elev[weak_neck] = -140.0 - 520.0 * np.clip(dist_to_any_land[weak_neck] / max(1.0, width / 330.0), 0.0, 1.0)
                island_origin[weak_neck] = 0
                crust_class[weak_neck] = 0
    land = new_elev >= 1.0
    ocean = ~land
    new_elev[land] = np.maximum(new_elev[land], 1.0)
    new_elev[ocean] = np.minimum(new_elev[ocean], -1.0)

    # Prevent plate lake-candidate fields from taking over continents. Keep rift,
    # foreland, and basin candidates only where they are local depressions.
    lake0 = np.clip(_resize_arr(getattr(terrain, "plate_tectonic_lake_candidate_x1000", None)), 0.0, 1.0)
    slope_y, slope_x = np.gradient(new_elev)
    slope = np.sqrt(slope_x * slope_x + slope_y * slope_y)
    local_mean = ndimage.gaussian_filter(new_elev, sigma=max(1.2, width / 360.0), mode="wrap")
    depression = np.clip((local_mean - new_elev) / 420.0, 0.0, 1.0)
    lake_field = np.clip(lake0 * (0.35 + 0.65 * depression) * (1.0 - np.clip(slope / 600.0, 0.0, 1.0)), 0.0, 1.0) * land.astype(np.float32)
    candidate = lake_field > 0.48
    lake_rejected_cells = 0
    lake_rejected_components = 0
    if bool(candidate.any()):
        lab, n = ndimage.label(candidate, structure=structure)
        sz = np.bincount(lab.ravel()) if n else np.asarray([0], dtype=np.int64)
        if sz.size:
            sz[0] = 0
        max_lake_component = max(64, int(max(land.sum() * 0.018, total_cells * 0.0018)))
        for lid in [int(i) for i in np.where(sz > max_lake_component)[0] if i > 0]:
            mask = lab == lid
            lake_rejected_cells += int(mask.sum())
            lake_rejected_components += 1
            lake_field[mask] *= 0.18

    # Straight-coastline diagnostics: count long horizontal/vertical coastline
    # runs in full-resolution cells. This does not over-carve the coast, but it
    # makes the artifact visible in summaries and QA.
    north = np.vstack((land[0:1, :], land[:-1, :]))
    south = np.vstack((land[1:, :], land[-1:, :]))
    west = np.roll(land, 1, axis=1)
    east = np.roll(land, -1, axis=1)
    coast = land & ((~north) | (~south) | (~west) | (~east))

    def _longest_run(mask: np.ndarray, axis: int) -> int:
        best = 0
        arr = mask if axis == 1 else mask.T
        for row in arr:
            if not bool(row.any()):
                continue
            doubled = np.r_[row, row] if axis == 1 else row
            limit = width if axis == 1 else height
            i = 0
            while i < len(doubled):
                if not bool(doubled[i]):
                    i += 1
                    continue
                j = i
                while j < len(doubled) and bool(doubled[j]):
                    j += 1
                best = max(best, min(j - i, limit))
                i = j
                if axis == 1 and i >= limit * 2:
                    break
        return int(best)

    longest_h = _longest_run(coast, axis=1)
    longest_v = _longest_run(coast, axis=0)
    exact_1m_count = int(np.sum(land & (np.rint(new_elev).astype(np.int32) == 1)))

    elev_i = np.rint(np.clip(new_elev, -11000, 10000)).astype(np.int32)
    final_land = elev_i >= 1
    final_ocean = ~final_land
    terrain.elevation_m = elev_i.astype(int).tolist()
    terrain.is_land = final_land.astype(bool).tolist()
    terrain.land_fraction = float(np.mean(final_land))
    terrain.ocean_fraction = 1.0 - terrain.land_fraction
    terrain.min_elevation_m = int(elev_i.min())
    terrain.max_elevation_m = int(elev_i.max())
    terrain.mean_land_elevation_m = float(np.mean(elev_i[final_land])) if bool(final_land.any()) else 0.0
    terrain.mean_ocean_depth_m = float(np.mean(elev_i[final_ocean])) if bool(final_ocean.any()) else 0.0
    terrain.source = str(getattr(terrain, "source", "plate_tectonic_v1")) + "; Plate Terrain 16 feedback cleanup"

    crust_class[final_ocean & (shelf_field > 0.06)] = 1
    crust_class[final_ocean & (shelf_field <= 0.06) & (seamount > 0.56)] = 8
    missing_crust = final_land & (crust_class == 0)
    if bool(missing_crust.any()):
        crust_class[missing_crust] = np.where(arc0[missing_crust] > 0.45, 6, 2)
    island_origin[~final_land] = 0
    missing_island_origin = final_land & (island_origin == 0) & (~carrier)
    if bool(missing_island_origin.any()):
        island_origin[missing_island_origin] = np.where(arc0[missing_island_origin] > 0.38, 3, 4)

    mountain_field = np.clip(np.maximum(_resize_arr(getattr(terrain, "terrain_mountain_strength_x1000", None)), branch_field), 0.0, 1.0)
    plateau_field = np.clip(np.maximum(_resize_arr(getattr(terrain, "terrain_plateau_x1000", None)), plateau_field), 0.0, 1.0)
    terrain_detail = np.clip(_resize_arr(getattr(terrain, "plate_tectonic_terrain_detail_x1000", None)) + 0.20 * branch_field, 0.0, 1.0)

    meta = {
        "applied": True,
        "stage": "plate-terrain-16-user-feedback-cleanup",
        "suppress_polar_land": bool(controls.get("suppress_polar_land", False)),
        "polar_land_cells_removed": polar_removed_count,
        "volcanic_arc_land_cells_added": int(arc_added.sum()),
        "land_fraction_after": round(float(terrain.land_fraction), 4),
        "ocean_fraction_after": round(float(terrain.ocean_fraction), 4),
        "mean_shelf_strength_ocean": round(float(np.mean(shelf_field[final_ocean])) if bool(final_ocean.any()) else 0.0, 4),
        "shelf_ocean_share": round(float(np.mean((shelf_field > 0.08)[final_ocean])) if bool(final_ocean.any()) else 0.0, 4),
        "snake_like_high_island_components_repaired": snake_components,
        "high_island_edge_cells_lowered": high_island_edge_cells,
        "snake_neck_cells_sunk": snake_edge_cells,
        "lake_candidate_rejected_components": lake_rejected_components,
        "lake_candidate_rejected_cells": lake_rejected_cells,
        "land_exactly_1m_count": exact_1m_count,
        "land_exactly_1m_share_of_land": round(float(exact_1m_count) / max(1, int(final_land.sum())), 6),
        "longest_horizontal_coast_run_cells": longest_h,
        "longest_vertical_coast_run_cells": longest_v,
        "notes": [
            "Longitude-wrapped distance transforms are used for shelves, coastal blending, and nearshore bathymetry.",
            "Shelves are now tied to carrier continents and passive/rifted/sedimentary margins, not to every island component.",
            "Small volcanic/arc islands get steep aprons instead of continental shelf halos.",
            "The exact-1m land diagnostic count is written for terrain QA and map output.",
        ],
    }
    diagnostics = terrain.terrain_diagnostics if isinstance(getattr(terrain, "terrain_diagnostics", None), dict) else {}
    plate_diag = diagnostics.setdefault("plate_tectonic_v1", {})
    plate_diag["feedback_cleanup_u16"] = meta
    diagnostics["land_exactly_1m"] = {
        "count": exact_1m_count,
        "share_of_land": meta["land_exactly_1m_share_of_land"],
        "description": "Cells where final terrain is land and rounded elevation is exactly +1 m.",
    }
    diagnostics["straight_coastline"] = {
        "longest_horizontal_run_cells": longest_h,
        "longest_vertical_run_cells": longest_v,
        "description": "Diagnostic for long row/column-aligned coast segments after final cleanup.",
    }
    terrain.terrain_diagnostics = diagnostics

    return {
        "metadata": meta,
        "shelf_width_x1000": _x1000(shelf_field),
        "island_origin_class": island_origin.astype(int).tolist(),
        "crust_class": crust_class.astype(int).tolist(),
        "mountain_strength_x1000": _x1000(mountain_field),
        "plateau_x1000": _x1000(plateau_field),
        "lake_candidate_x1000": _x1000(lake_field),
        "terrain_detail_x1000": _x1000(terrain_detail),
    }

def _generate_plate_tectonic_v1_plate_setup(
    rng: random.Random,
    terrain: TerrainMap,
    hydrosphere: Hydrosphere,
    geology: GeologyState,
    controls: dict,
) -> dict:
    """Generate the first native plate-tectonic v1 state layer.

    This is a synthetic plate setup, not yet a time-evolved GPlates-style
    reconstruction. It deliberately stops at plate seeding and crust allocation
    so later updates can add motion, boundary classification, and plate-derived
    relief without changing the downstream terrain contract.
    """
    try:
        import numpy as np
        from PIL import Image
        from scipy import ndimage
    except ImportError as exc:
        raise RuntimeError("NumPy, Pillow, and SciPy are required for plate terrain diagnostics. Install with: pip install -r requirements.txt") from exc

    land_full = np.asarray(terrain.is_land, dtype=bool)
    height = int(terrain.height)
    width = int(terrain.width)
    diag_w = max(256, min(768, width // 3 if width >= 768 else 512))
    diag_h = max(128, min(384, height // 3 if height >= 384 else 256))
    land_low = np.asarray(Image.fromarray(land_full.astype(np.uint8) * 255, mode="L").resize((diag_w, diag_h), Image.Resampling.BICUBIC), dtype=np.float32) / 255.0
    land_bool = land_low >= 0.50

    lats = np.linspace(90.0 - 90.0 / diag_h, -90.0 + 90.0 / diag_h, diag_h, dtype=np.float32)
    lons = np.linspace(-180.0 + 180.0 / diag_w, 180.0 - 180.0 / diag_w, diag_w, dtype=np.float32)
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    cos_grid = np.maximum(0.18, np.cos(np.radians(lat_grid)))

    target_count = controls.get("target_plate_count", 32)
    try:
        target_count = int(round(float(target_count)))
    except Exception:
        target_count = 32
    fragmentation = clamp(float(controls.get("fragmentation_tendency", 0.5) or 0.5), 0.0, 1.0)
    supercontinent = clamp(float(controls.get("effective_supercontinent_score", controls.get("derived_supercontinent_score", 0.45)) or 0.45), 0.0, 1.0)
    island_density = clamp(float(controls.get("island_density", 0.45) or 0.45), 0.0, 1.0)
    ocean_target = clamp(float(getattr(hydrosphere, "ocean_fraction_target", terrain.ocean_fraction) or terrain.ocean_fraction), 0.05, 0.95)
    heat = clamp(float(getattr(geology, "internal_heat", 0.75) or 0.75), 0.0, 2.0) / 2.0
    volcanism = clamp(float(getattr(geology, "volcanism", 0.55) or 0.55), 0.0, 1.8) / 1.8

    # Plate Terrain 12: reduce default plate fragmentation.  Keep enough plates
    # for useful boundaries, but avoid letting microplates dominate the land mask.
    effective_count = int(round(target_count - 4.0 + 6.0 * fragmentation - 8.0 * supercontinent + 3.0 * max(0.0, ocean_target - 0.62) + 2.5 * heat))
    effective_count = max(7, min(72, effective_count))
    microplate_count = int(round(effective_count * (0.035 + 0.125 * fragmentation + 0.055 * island_density + 0.045 * volcanism)))
    microplate_count = max(0, min(max(4, effective_count // 2), microplate_count))
    macroplate_count = max(5, effective_count - microplate_count)

    centers = []
    # Area-corrected latitude sampling; unlike old terrain land seeding, plate
    # centers are allowed near poles so polar land/plate structure is possible.
    for idx in range(effective_count):
        is_micro = idx >= macroplate_count
        lon0 = rng.uniform(-180.0, 180.0)
        lat0 = math.degrees(math.asin(rng.uniform(-0.985, 0.985)))
        if is_micro and rng.random() < 0.68 and land_bool.any():
            # Bias microplates toward continental margins and island-rich belts.
            coast = land_bool & (~(np.roll(land_bool, 1, axis=0) & np.roll(land_bool, -1, axis=0) & np.roll(land_bool, 1, axis=1) & np.roll(land_bool, -1, axis=1)))
            candidates = np.argwhere(coast | ((land_low > 0.20) & (land_low < 0.80)))
            if candidates.size:
                rr, cc = candidates[rng.randrange(len(candidates))]
                lat0 = float(lats[int(rr)])
                lon0 = float(lons[int(cc)])
        weight = rng.uniform(0.88, 1.55) if not is_micro else rng.uniform(0.28, 0.72)
        age = clamp(rng.betavariate(1.6, 1.6) + (0.10 if not is_micro else -0.10), 0.0, 1.0)
        centers.append((lon0, lat0, weight, age, is_micro))

    # Plate Terrain 10: generate plates in a warped, domain-aware coordinate
    # space rather than pure circular/Voronoi cells.  This is a long-term
    # structural change: plate IDs become irregular carriers of crustal domains
    # instead of smooth disks that later turn into pedestal landmasses.
    warp_rng = np.random.default_rng(int(rng.randrange(1, 2**31 - 1)))
    warp_lon = warp_rng.normal(0.0, 1.0, size=(diag_h, diag_w)).astype(np.float32)
    warp_lat = warp_rng.normal(0.0, 1.0, size=(diag_h, diag_w)).astype(np.float32)
    warp_lon = ndimage.gaussian_filter(warp_lon, sigma=max(1.2, diag_w / 38.0), mode="wrap")
    warp_lat = ndimage.gaussian_filter(warp_lat, sigma=max(1.2, diag_w / 42.0), mode="wrap")
    for arr in (warp_lon, warp_lat):
        std = float(np.std(arr))
        if std > 1.0e-6:
            arr -= float(np.mean(arr)); arr /= std
    warped_lon_grid = lon_grid + warp_lon * (4.2 + 7.2 * fragmentation)
    warped_lat_grid = np.clip(lat_grid + warp_lat * (2.6 + 4.8 * fragmentation) * np.sqrt(cos_grid), -89.4, 89.4)
    best = np.full((diag_h, diag_w), 1.0e12, dtype=np.float32)
    second = np.full((diag_h, diag_w), 1.0e12, dtype=np.float32)
    best_id = np.zeros((diag_h, diag_w), dtype=np.int32)
    for idx, (lon0, lat0, weight, _age, is_micro) in enumerate(centers):
        dlon = ((warped_lon_grid - lon0 + 180.0) % 360.0 - 180.0) * cos_grid
        dlat = warped_lat_grid - lat0
        anis = 1.0 + 0.28 * math.sin(math.radians(lon0 * 2.0 + lat0))
        dist = (dlon * dlon * anis + dlat * dlat / max(0.50, anis)) / max(0.08, weight)
        # Soft ribbing and local roughness break perfectly smooth cells without
        # creating random fragmented plates.
        dist *= 1.0 + 0.105 * np.sin(np.radians((warped_lon_grid - lon0) * rng.uniform(1.4, 3.8) + (warped_lat_grid - lat0) * rng.uniform(0.8, 3.0)))
        dist += 0.055 * warp_lon * rng.uniform(-1.0, 1.0) + 0.055 * warp_lat * rng.uniform(-1.0, 1.0)
        if is_micro:
            dist *= 1.45
        better = dist < best
        second = np.where(better, best, np.minimum(second, dist))
        best = np.where(better, dist, best)
        best_id = np.where(better, idx, best_id)

    # Compress ids so empty/squeezed seeds do not create phantom plates.
    unique_ids = np.unique(best_id)
    id_map = {int(old): new for new, old in enumerate(unique_ids.tolist())}
    plate_id = np.vectorize(lambda x: id_map[int(x)], otypes=[np.int32])(best_id).astype(np.int32)
    plate_count = int(len(unique_ids))
    # Plate Terrain 13: roughen plate boundaries at the ID level before motion
    # diagnostics are derived.  This prevents smooth Voronoi-like plates from
    # surviving as visibly artificial arcs in the plate maps.
    plate_id = _roughen_plate_boundaries_u13(plate_id, warp_rng, iterations=7)

    # Plate Terrain 10 topology repair: remove bullseye-style plates that are
    # completely surrounded by one neighbor and soften plate roundness before
    # those IDs are allowed to drive terrain. This is a graph/topology repair,
    # not a visual brightline patch.
    topology_problem_class = np.zeros_like(plate_id, dtype=np.uint8)
    if plate_count > 1:
        for _repair_iter in range(4):
            changed = False
            sizes_tmp = np.bincount(plate_id.ravel(), minlength=max(plate_count, int(plate_id.max()) + 1)).astype(np.float64)
            total_cells = max(1.0, float(plate_id.size))
            for pid in np.unique(plate_id).astype(int).tolist():
                comp = plate_id == pid
                if not np.any(comp):
                    continue
                nbs = []
                for shifted in (np.roll(plate_id, 1, axis=1), np.roll(plate_id, -1, axis=1), np.vstack((plate_id[0:1, :], plate_id[:-1, :])), np.vstack((plate_id[1:, :], plate_id[-1:, :]))):
                    vals = shifted[comp & (shifted != pid)]
                    if vals.size:
                        nbs.extend([int(v) for v in np.unique(vals)])
                uniq_nbs = sorted(set(nbs))
                area = float(sizes_tmp[pid]) / total_cells if pid < sizes_tmp.size else 0.0
                # Plate Terrain 13: a plate wholly surrounded by a single neighbor
                # is almost always a Voronoi/topology artifact, even if it is not
                # tiny. Merge it unless it is a continent-scale carrier.
                if len(uniq_nbs) == 1 and area < max(0.18, 4.2 / max(8, plate_count)):
                    topology_problem_class[comp] = 1
                    plate_id[comp] = uniq_nbs[0]
                    changed = True
            if not changed:
                break
        # Keep the original compact ID space so seeded plate metadata (age/type,
        # original microplate flag, centers) remains aligned with plate ids. Empty
        # merged-away IDs simply have size zero and are ignored downstream.
        topology_problem_class = topology_problem_class.astype(np.uint8)

    sizes = np.bincount(plate_id.ravel(), minlength=plate_count).astype(np.float64)
    land_sums = np.bincount(plate_id.ravel(), weights=land_low.ravel(), minlength=plate_count).astype(np.float64)
    land_share_by_plate = land_sums / np.maximum(1.0, sizes)
    area_share = sizes / max(1.0, float(plate_id.size))
    age_by_plate = np.zeros(plate_count, dtype=np.float32)
    original_micro = np.zeros(plate_count, dtype=bool)
    for old, new_id in id_map.items():
        age_by_plate[new_id] = float(centers[old][3])
        original_micro[new_id] = bool(centers[old][4])

    plate_type_by_plate = np.zeros(plate_count, dtype=np.int32)  # 0 oceanic
    # Plate Terrain 10: plate type allocation is no longer derived mainly from
    # the legacy land mask. The existing land share is kept as a weak continuity
    # clue, but continental/mixed plates are selected from native plate size,
    # age, and seeded randomness so plate_tectonic_v1 can own the foundation.
    crust_rng = np.random.default_rng(int(rng.randrange(1, 2**31 - 1)))
    area_norm = area_share / max(1.0e-6, float(np.max(area_share)) if area_share.size else 1.0)
    land_target = clamp(1.0 - ocean_target, 0.05, 0.70)
    crust_score = (
        0.26 * land_share_by_plate
        + 0.24 * area_norm
        + 0.22 * age_by_plate
        + 0.18 * crust_rng.random(plate_count)
        + 0.10 * (1.0 - original_micro.astype(np.float32))
    )
    order = np.argsort(-crust_score)
    cumulative_area = 0.0
    continental_area_goal = clamp(land_target * (1.25 + 0.55 * supercontinent - 0.25 * fragmentation), 0.10, 0.72)
    mixed_area_goal = clamp(continental_area_goal + 0.16 + 0.18 * fragmentation + 0.10 * island_density, continental_area_goal + 0.06, 0.92)
    for pid in order:
        if original_micro[pid]:
            continue
        cumulative_area += float(area_share[pid])
        if cumulative_area <= continental_area_goal:
            plate_type_by_plate[pid] = 1  # continental plate
        elif cumulative_area <= mixed_area_goal:
            plate_type_by_plate[pid] = 2  # mixed plate / continental margin carrier
        else:
            plate_type_by_plate[pid] = 0
    if plate_count and not np.any(plate_type_by_plate == 1):
        plate_type_by_plate[int(order[0])] = 1
    # Microplates/terranes should be small crustal domains, not full-size
    # primary plates.  Earlier plate revisions let every originally micro seed
    # claim a Voronoi-sized cell, which helped create island-world maps.
    small_area = area_share < max(0.0035, 0.16 / max(1, plate_count))
    modest_area = area_share < max(0.009, 0.30 / max(1, plate_count))
    micro_mask_by_plate = (small_area & (crust_score > float(np.median(crust_score)) * 0.70)) | (original_micro & modest_area & (crust_score > float(np.percentile(crust_score, 45))))
    plate_type_by_plate[micro_mask_by_plate] = 3  # microplate/terrane
    # Original micro seeds that grew into large cells are demoted to mixed or
    # oceanic carrier plates; they can host terranes along boundaries later, but
    # should not turn the whole map into islands.
    oversized_original_micro = original_micro & (~micro_mask_by_plate)
    if np.any(oversized_original_micro):
        for pid in np.where(oversized_original_micro)[0]:
            plate_type_by_plate[int(pid)] = 2 if crust_score[int(pid)] > float(np.median(crust_score)) else 0
    plate_type = plate_type_by_plate[plate_id].astype(np.int32)

    boundary_seed = np.zeros_like(plate_id, dtype=bool)
    boundary_seed |= plate_id != np.roll(plate_id, 1, axis=1)
    boundary_seed |= plate_id != np.roll(plate_id, -1, axis=1)
    boundary_seed[:-1, :] |= plate_id[:-1, :] != plate_id[1:, :]
    boundary_seed[1:, :] |= plate_id[1:, :] != plate_id[:-1, :]
    boundary_distance = ndimage.distance_transform_edt(~boundary_seed)
    coast = land_bool & ((~np.roll(land_bool, 1, axis=1)) | (~np.roll(land_bool, -1, axis=1)) | (~np.roll(land_bool, 1, axis=0)) | (~np.roll(land_bool, -1, axis=0)))
    land_distance = ndimage.distance_transform_edt(land_bool)
    ocean_distance = ndimage.distance_transform_edt(~land_bool)

    continental_base = np.zeros((diag_h, diag_w), dtype=np.float32)
    continental_base[plate_type == 1] = 0.76
    continental_base[plate_type == 2] = 0.46
    continental_base[plate_type == 3] = 0.30
    crust_noise = crust_rng.normal(0.0, 1.0, size=(diag_h, diag_w)).astype(np.float32)
    crust_noise = ndimage.gaussian_filter(crust_noise, sigma=max(1.2, diag_w / 70.0), mode="wrap")
    if float(np.std(crust_noise)) > 1.0e-6:
        crust_noise = (crust_noise - float(np.mean(crust_noise))) / float(np.std(crust_noise))
    crust_lobes = np.clip(0.50 + 0.18 * crust_noise + 0.18 * age_by_plate[plate_id] + 0.08 * area_norm[plate_id], 0.0, 1.0)
    continental_crust = np.clip(continental_base * crust_lobes + 0.10 * land_low + 0.05 * (plate_type == 3), 0.0, 1.0)
    margin_zone = (continental_crust > 0.20) & (continental_crust < 0.58)
    continental_crust[margin_zone] = np.maximum(continental_crust[margin_zone], 0.24 + 0.10 * (plate_type[margin_zone] == 2))

    craton = (plate_type == 1) & (continental_crust > 0.48) & (boundary_distance > max(3.0, diag_w / 80.0))
    craton_strength = np.zeros((diag_h, diag_w), dtype=np.float32)
    if craton.any():
        craton_strength[craton] = np.clip(boundary_distance[craton] / max(4.0, diag_w / 26.0), 0.0, 1.0)
        craton_strength *= np.clip(0.55 + 0.55 * age_by_plate[plate_id], 0.0, 1.0)
        craton_strength = np.clip(ndimage.gaussian_filter(craton_strength, sigma=max(0.65, diag_w / 420.0), mode="wrap"), 0.0, 1.0)

    microplate_field = np.zeros((diag_h, diag_w), dtype=np.float32)
    microplate_field[plate_type == 3] = 1.0
    microplate_field[(plate_type == 2) & (margin_zone | (ocean_distance < max(3.0, diag_w / 120.0)))] = np.maximum(microplate_field[(plate_type == 2) & (margin_zone | (ocean_distance < max(3.0, diag_w / 120.0)))], 0.45)

    province_type = np.zeros((diag_h, diag_w), dtype=np.int32)
    province_type[(plate_type == 0) & (age_by_plate[plate_id] < 0.35)] = 1
    province_type[(plate_type == 1) & (craton_strength > 0.40)] = 2
    province_type[(plate_type == 1) & (craton_strength <= 0.40)] = 8
    province_type[(plate_type == 2) & (land_low > 0.30)] = 3
    province_type[(plate_type == 3)] = 6
    province_type[(plate_type == 0) & (land_low > 0.12) & (land_low < 0.45)] = 5
    province_type[margin_zone & (plate_type != 3)] = np.where(province_type[margin_zone & (plate_type != 3)] == 0, 3, province_type[margin_zone & (plate_type != 3)])

    plate_age = np.clip(age_by_plate[plate_id] * 0.65 + craton_strength * 0.30 + (plate_type == 0) * 0.10, 0.0, 1.0)

    def x1000(arr):
        return np.rint(np.clip(arr, 0.0, 1.0) * 1000.0).astype(np.int32).tolist()

    def signed_x1000(arr):
        return np.rint(np.clip(arr, -1.0, 1.0) * 1000.0).astype(np.int32).tolist()

    # Plate Terrain 2: assign Euler-like synthetic plate motion and classify
    # boundaries from relative motion. These fields are native plate diagnostics
    # but final elevation still uses the legacy compatibility backend.
    motion_speed_control = clamp(float(controls.get("plate_motion_speed", 0.50) or 0.50), 0.0, 1.5)
    motion_chaos = clamp(float(controls.get("plate_motion_chaos", 0.35) or 0.35), 0.0, 1.0)
    convergence_bias = clamp(float(controls.get("convergence_bias", 0.50) or 0.50), 0.0, 1.0)
    divergence_bias = clamp(float(controls.get("divergence_bias", 0.45) or 0.45), 0.0, 1.0)
    transform_bias = clamp(float(controls.get("transform_bias", 0.32) or 0.32), 0.0, 1.0)

    center_lon_by_plate = np.zeros(plate_count, dtype=np.float32)
    center_lat_by_plate = np.zeros(plate_count, dtype=np.float32)
    for old, new_id in id_map.items():
        center_lon_by_plate[new_id] = float(centers[old][0])
        center_lat_by_plate[new_id] = float(centers[old][1])

    pole_lon = rng.uniform(-180.0, 180.0)
    pole_lat = math.degrees(math.asin(rng.uniform(-0.92, 0.92)))
    vx_by_plate = np.zeros(plate_count, dtype=np.float32)
    vy_by_plate = np.zeros(plate_count, dtype=np.float32)
    speed_by_plate = np.zeros(plate_count, dtype=np.float32)
    for pid in range(plate_count):
        dlon = ((float(center_lon_by_plate[pid]) - pole_lon + 180.0) % 360.0 - 180.0) * max(0.25, math.cos(math.radians(float(center_lat_by_plate[pid]))))
        dlat = float(center_lat_by_plate[pid]) - pole_lat
        # Tangential velocity around a synthetic Euler pole, blended with local
        # plate-specific chaos so every run is not a single pinwheel.
        euler_angle = math.atan2(dlat, dlon) + math.pi / 2.0
        local_angle = rng.uniform(-math.pi, math.pi)
        angle = (1.0 - motion_chaos) * euler_angle + motion_chaos * local_angle
        ptype = int(plate_type_by_plate[pid]) if pid < plate_type_by_plate.size else 0
        type_factor = {0: 1.08, 1: 0.72, 2: 0.88, 3: 1.22}.get(ptype, 1.0)
        age_factor = 1.12 - 0.34 * float(age_by_plate[pid])
        speed = clamp((0.16 + 0.84 * motion_speed_control) * type_factor * age_factor * rng.uniform(0.62, 1.28), 0.02, 1.0)
        speed_by_plate[pid] = speed
        vx_by_plate[pid] = math.cos(angle) * speed
        vy_by_plate[pid] = math.sin(angle) * speed

    velocity_x = vx_by_plate[plate_id].astype(np.float32)
    velocity_y = vy_by_plate[plate_id].astype(np.float32)
    speed_field = np.clip(np.sqrt(velocity_x * velocity_x + velocity_y * velocity_y), 0.0, 1.0)

    convergence = np.zeros_like(speed_field, dtype=np.float32)
    divergence = np.zeros_like(speed_field, dtype=np.float32)
    transform = np.zeros_like(speed_field, dtype=np.float32)
    oceanic_pair = np.zeros_like(plate_id, dtype=bool)
    continental_pair = np.zeros_like(plate_id, dtype=bool)
    both_continental_pair = np.zeros_like(plate_id, dtype=bool)
    mixed_pair = np.zeros_like(plate_id, dtype=bool)

    def apply_pair(mask, a_slice, b_slice, normal_x, normal_y, tangent_x, tangent_y):
        if not np.any(mask):
            return
        rel_x = velocity_x[b_slice] - velocity_x[a_slice]
        rel_y = velocity_y[b_slice] - velocity_y[a_slice]
        rel_normal = rel_x * normal_x + rel_y * normal_y
        rel_tangent = rel_x * tangent_x + rel_y * tangent_y
        conv = np.clip(-rel_normal, 0.0, 1.0) * (0.55 + 0.75 * convergence_bias)
        div = np.clip(rel_normal, 0.0, 1.0) * (0.55 + 0.75 * divergence_bias)
        sh = np.clip(np.abs(rel_tangent), 0.0, 1.0) * (0.55 + 0.75 * transform_bias)
        ta = plate_type[a_slice]
        tb = plate_type[b_slice]
        oc = ((ta == 0) | (tb == 0)) & mask
        cont = (np.isin(ta, [1, 2]) | np.isin(tb, [1, 2])) & mask
        both_cont = (np.isin(ta, [1, 2]) & np.isin(tb, [1, 2])) & mask
        mix = ((ta != tb) | (ta == 3) | (tb == 3)) & mask
        for target_slice in (a_slice, b_slice):
            convergence[target_slice] = np.maximum(convergence[target_slice], np.where(mask, conv, 0.0))
            divergence[target_slice] = np.maximum(divergence[target_slice], np.where(mask, div, 0.0))
            transform[target_slice] = np.maximum(transform[target_slice], np.where(mask, sh, 0.0))
            oceanic_pair[target_slice] |= oc
            continental_pair[target_slice] |= cont
            both_continental_pair[target_slice] |= both_cont
            mixed_pair[target_slice] |= mix

    east_a = (slice(None), slice(None))
    east_b = (slice(None), np.r_[1:diag_w, 0])
    east_mask = plate_id != np.roll(plate_id, -1, axis=1)
    apply_pair(east_mask, east_a, east_b, 1.0, 0.0, 0.0, 1.0)

    south_a = (slice(0, diag_h - 1), slice(None))
    south_b = (slice(1, diag_h), slice(None))
    south_mask = plate_id[:-1, :] != plate_id[1:, :]
    apply_pair(south_mask, south_a, south_b, 0.0, -1.0, 1.0, 0.0)

    raw_strength = np.maximum.reduce([convergence, divergence, transform])
    boundary_class = np.zeros_like(plate_id, dtype=np.int32)
    weak_boundary = (raw_strength > 0.025) & (raw_strength < 0.11)
    passive = weak_boundary & oceanic_pair & continental_pair
    diffuse = weak_boundary & ~passive
    boundary_class[passive] = 4
    boundary_class[diffuse] = 5
    transform_dom = (transform > 0.055) & (transform >= np.maximum(convergence, divergence) * 1.14)
    divergent_dom = (divergence > 0.05) & (divergence >= convergence * 0.92) & ~transform_dom
    convergent_dom = (convergence > 0.05) & (convergence > divergence) & ~transform_dom
    boundary_class[divergent_dom] = 2
    boundary_class[transform_dom] = 3
    collision = convergent_dom & both_continental_pair
    volcanic_arc = convergent_dom & ~both_continental_pair & oceanic_pair
    boundary_class[collision] = 1
    boundary_class[volcanic_arc] = 6
    boundary_class[(convergent_dom & ~(collision | volcanic_arc))] = 1

    boundary_active = boundary_class > 0
    boundary_strength = np.clip(raw_strength * (0.85 + 0.25 * boundary_active.astype(np.float32)), 0.0, 1.0)
    boundary_distance_native = ndimage.distance_transform_edt(~boundary_active)
    boundary_width_factor = clamp(float(controls.get("boundary_width_factor", 1.0) or 1.0), 0.15, 3.0)
    boundary_width = np.clip(np.exp(-boundary_distance_native / max(1.0, 2.2 * boundary_width_factor + 1.5 * motion_chaos)), 0.0, 1.0)
    boundary_width[~boundary_active & (boundary_width < 0.10)] = 0.0

    subduction_polarity = np.zeros_like(plate_id, dtype=np.int32)
    subduction_polarity[volcanic_arc & continental_pair] = 1
    subduction_polarity[volcanic_arc & ~continental_pair] = 2
    subduction_polarity[collision] = 3

    # Feed native relative-motion classes back into the diagnostic province map.
    province_type[(boundary_class == 2) & (plate_type != 0)] = 4
    province_type[(boundary_class == 2) & (plate_type == 0)] = 1
    province_type[boundary_class == 6] = 5
    province_type[(boundary_class == 1) & np.isin(plate_type, [1, 2])] = 8
    province_type[(boundary_class == 3) & (plate_type == 3)] = 6

    # Plate Terrain 3: derive the first native ocean-floor model from native
    # plate motion. Divergent oceanic boundaries become ridge/spreading centers;
    # convergent oceanic/continental margins become trenches; transform motion
    # becomes fracture zones. Crust-age deepening then creates abyssal plains
    # away from ridges. These diagnostic fields currently drive review/QA first;
    # later plate-terrain updates will let them own final bathymetry elevation.
    ocean = ~land_bool
    oceanic = plate_type == 0
    mixed_ocean = (plate_type == 2) & ocean
    continental_edge_ocean = ocean & (continental_crust > 0.20) & (continental_crust < 0.58)
    ridge_seed_native = ocean & ((boundary_class == 2) | ((divergence > 0.14) & oceanic))
    trench_seed_native = ocean & ((subduction_polarity > 0) | ((convergence > 0.16) & oceanic_pair & (continental_pair | mixed_pair | continental_edge_ocean)))
    fracture_seed_native = ocean & ((boundary_class == 3) | ((transform > 0.18) & oceanic_pair))
    ridge_field_native = ndimage.gaussian_filter(ridge_seed_native.astype(np.float32) * (0.55 + 0.90 * divergence), sigma=max(0.75, diag_w / 420.0), mode="wrap") * ocean.astype(np.float32)
    trench_field_native = ndimage.gaussian_filter(trench_seed_native.astype(np.float32) * (0.55 + 0.90 * convergence), sigma=max(0.55, diag_w / 520.0), mode="wrap") * ocean.astype(np.float32)
    fracture_field_native = ndimage.gaussian_filter(fracture_seed_native.astype(np.float32) * (0.45 + 0.85 * transform), sigma=max(0.55, diag_w / 620.0), mode="wrap") * ocean.astype(np.float32)
    for _name, _arr in [("ridge", ridge_field_native), ("trench", trench_field_native), ("fracture", fracture_field_native)]:
        if float(_arr.max()) > 1.0e-6:
            _arr /= max(1.0e-6, float(np.percentile(_arr[_arr > 0], 96 if _name == "ridge" else 94)))
            np.clip(_arr, 0.0, 1.0, out=_arr)

    if np.any(ridge_field_native > 0.18):
        dist_to_native_ridge = ndimage.distance_transform_edt(~(ridge_field_native > 0.18))
        ocean_crust_age = np.clip(dist_to_native_ridge / max(6.0, min(88.0, diag_w / 12.0)), 0.0, 1.0) * ocean.astype(np.float32)
    else:
        ocean_crust_age = np.clip(ocean_distance / max(6.0, min(88.0, diag_w / 12.0)), 0.0, 1.0) * ocean.astype(np.float32)
    # Fast/micro/oceanic plates and high volcanism create more hotspot/seamount
    # chains, but keep them separate from trenches so trenches remain visually
    # readable.
    seamount_noise = np.zeros((diag_h, diag_w), dtype=np.float32)
    chain_sources = max(2, min(12, int(round(2 + 7 * volcanism + 4 * island_density + 4 * float(np.mean(speed_field[ocean])) if np.any(ocean) else 3))))
    for _chain in range(chain_sources):
        ocean_choices = np.argwhere(ocean & (ocean_crust_age > 0.20) & ~(trench_field_native > 0.22))
        if ocean_choices.size == 0:
            break
        rr, cc = ocean_choices[rng.randrange(len(ocean_choices))]
        lon = float(lons[int(cc)]); lat = float(lats[int(rr)]); heading = rng.uniform(-math.pi, math.pi)
        for _node in range(rng.randint(3, 8)):
            heading += rng.uniform(-0.45, 0.45)
            lon += math.cos(heading) * rng.uniform(3.0, 11.0) / max(0.25, math.cos(math.radians(lat)))
            lat = clamp(lat + math.sin(heading) * rng.uniform(1.8, 7.5), -78.0, 78.0)
            dlon = _wrapped_lon_delta_array(lon_grid, lon) * cos_grid
            dlat = lat_grid - lat
            radius = rng.uniform(0.55, 1.80)
            seamount_noise += np.exp(-((dlon / max(0.12, radius)) ** 2 + (dlat / max(0.12, radius * rng.uniform(0.70, 1.35))) ** 2)) * rng.uniform(0.30, 1.0)
    seamount_field_native = np.clip(seamount_noise, 0.0, None) * ocean.astype(np.float32)
    if float(seamount_field_native.max()) > 1.0e-6:
        seamount_field_native = np.clip(seamount_field_native / max(1.0e-6, float(np.percentile(seamount_field_native[seamount_field_native > 0], 97))), 0.0, 1.0)

    abyssal_plain = np.clip(ocean_crust_age * (1.0 - 0.42 * ridge_field_native) * (1.0 - 0.25 * seamount_field_native), 0.0, 1.0) * ocean.astype(np.float32)
    ocean_floor_class_native = np.zeros((diag_h, diag_w), dtype=np.int32)
    ocean_floor_class_native[ocean] = 1
    ocean_floor_class_native[ocean & (ridge_field_native > 0.32)] = 2
    ocean_floor_class_native[ocean & (trench_field_native > 0.25)] = 3
    ocean_floor_class_native[ocean & (fracture_field_native > 0.34) & ~(trench_field_native > 0.25)] = 4
    ocean_floor_class_native[ocean & (seamount_field_native > 0.42) & ~(trench_field_native > 0.25)] = 5

    type_counts = {
        "oceanic": int(np.sum(plate_type_by_plate == 0)),
        "continental": int(np.sum(plate_type_by_plate == 1)),
        "mixed": int(np.sum(plate_type_by_plate == 2)),
        "microplate_terrane": int(np.sum(plate_type_by_plate == 3)),
    }
    metadata = {
        "schema_version": 1,
        "stage": "plate-tectonic-v1-ocean-floor",
        "backend_status": "native plate setup + motion/boundary/ocean-floor diagnostics + continental relief shaping",
        "requested_plate_count": int(target_count),
        "effective_plate_count": int(plate_count),
        "macroplate_count": int(max(0, plate_count - type_counts["microplate_terrane"])),
        "microplate_count": type_counts["microplate_terrane"],
        "plate_type_counts": type_counts,
        "mean_plate_area_share": round(float(np.mean(area_share)) if area_share.size else 0.0, 5),
        "largest_plate_area_share": round(float(np.max(area_share)) if area_share.size else 0.0, 4),
        "single_neighbor_plate_repair_cell_share": round(float(np.mean(topology_problem_class == 1)), 4),
        "mean_plate_land_share": round(float(np.mean(land_share_by_plate)) if land_share_by_plate.size else 0.0, 4),
        "mean_continental_crust_fraction": round(float(np.mean(continental_crust)), 4),
        "craton_core_share": round(float(np.mean(craton_strength > 0.35)), 4),
        "microplate_cell_share": round(float(np.mean(microplate_field > 0.35)), 4),
        "mean_plate_motion_speed": round(float(np.mean(speed_by_plate)) if speed_by_plate.size else 0.0, 4),
        "max_plate_motion_speed": round(float(np.max(speed_by_plate)) if speed_by_plate.size else 0.0, 4),
        "boundary_activity_share": round(float(np.mean(boundary_active)), 4),
        "convergent_boundary_share": round(float(np.mean(np.isin(boundary_class, [1, 6]))), 4),
        "divergent_boundary_share": round(float(np.mean(boundary_class == 2)), 4),
        "transform_boundary_share": round(float(np.mean(boundary_class == 3)), 4),
        "passive_or_diffuse_boundary_share": round(float(np.mean(np.isin(boundary_class, [4, 5]))), 4),
        "mean_convergence_field": round(float(np.mean(convergence)), 4),
        "mean_divergence_field": round(float(np.mean(divergence)), 4),
        "mean_transform_field": round(float(np.mean(transform)), 4),
        "mean_ocean_crust_age": round(float(np.mean(ocean_crust_age[ocean])) if np.any(ocean) else 0.0, 4),
        "mid_ocean_ridge_share_of_ocean": round(float(np.mean((ridge_field_native > 0.32)[ocean])) if np.any(ocean) else 0.0, 4),
        "trench_share_of_ocean": round(float(np.mean((trench_field_native > 0.25)[ocean])) if np.any(ocean) else 0.0, 4),
        "fracture_zone_share_of_ocean": round(float(np.mean((fracture_field_native > 0.34)[ocean])) if np.any(ocean) else 0.0, 4),
        "seamount_share_of_ocean": round(float(np.mean((seamount_field_native > 0.42)[ocean])) if np.any(ocean) else 0.0, 4),
        "abyssal_plain_strong_share_of_ocean": round(float(np.mean((abyssal_plain > 0.62)[ocean])) if np.any(ocean) else 0.0, 4),
        "motion_controls": {
            "plate_motion_speed": round(float(motion_speed_control), 3),
            "plate_motion_chaos": round(float(motion_chaos), 3),
            "convergence_bias": round(float(convergence_bias), 3),
            "divergence_bias": round(float(divergence_bias), 3),
            "transform_bias": round(float(transform_bias), 3),
        },
        "polar_plate_center_allowed": True,
        "notes": [
            "Plate Terrain 10 creates plate IDs, crust allocation, plate motion vectors, relative-motion fields, native boundary classes, native ocean-floor diagnostics, a plate-owned foundation/mask/base elevation, plate-derived continental relief, native plate-margin coast/shelf/island fields, and final plate-mode integration/readiness QA.",
            "Plate Terrain 10 no longer runs the procedural legacy terrain core before plate setup; the broad land/ocean foundation comes from native domain and crust fields.",
        ],
    }
    return {
        "plate_id": plate_id.astype(int).tolist(),
        "plate_type": plate_type.astype(int).tolist(),
        "continental_crust_x1000": x1000(continental_crust),
        "craton_core_x1000": x1000(craton_strength),
        "microplate_x1000": x1000(microplate_field),
        "province_type": province_type.astype(int).tolist(),
        "plate_age_x1000": x1000(plate_age),
        "velocity_x_x1000": signed_x1000(velocity_x),
        "velocity_y_x1000": signed_x1000(velocity_y),
        "speed_x1000": x1000(speed_field),
        "convergence_x1000": x1000(convergence),
        "divergence_x1000": x1000(divergence),
        "transform_x1000": x1000(transform),
        "boundary_class": boundary_class.astype(int).tolist(),
        "boundary_strength_x1000": x1000(boundary_strength),
        "boundary_width_x1000": x1000(boundary_width),
        "subduction_polarity": subduction_polarity.astype(int).tolist(),
        "ocean_floor_class": ocean_floor_class_native.astype(int).tolist(),
        "ocean_crust_age_x1000": x1000(ocean_crust_age),
        "mid_ocean_ridge_x1000": x1000(ridge_field_native),
        "trench_x1000": x1000(trench_field_native),
        "fracture_zone_x1000": x1000(fracture_field_native),
        "abyssal_plain_x1000": x1000(abyssal_plain),
        "seamount_x1000": x1000(seamount_field_native),
        "topology_problem_class": topology_problem_class.astype(int).tolist(),
        "metadata": metadata,
    }


def _generate_terrain_core(
    rng: random.Random,
    planet: Planet,
    hydrosphere: Hydrosphere,
    geology: GeologyState,
    config: PlanetProfileConfig,
    *,
    output_dir: str | None = None,
) -> TerrainMap:
    """Generate a structured equirectangular terrain grid.

    This pass deliberately separates large structure from detail:
    - continent/ocean masks come from broad continental kernels;
    - mountain ranges are generated as land-weighted orogenic belts;
    - lone volcanic peaks are added as compact cones;
    - fractal roughness adds texture without creating global stripes;
    - a simple erosion/deposition pass carves valleys and smooths lowlands.

    This is still procedural terrain, not plate-tectonic history, but it gives
    rivers and drainage basins a more natural topographic field to follow.
    """
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("NumPy is required for detailed terrain generation. Install it with: pip install numpy") from exc

    width = max(int(config.min_map_width), int(config.map_width))
    height = max(int(config.min_map_height), int(config.map_height))
    large_direct = bool(config.no_accelerated_terrain and width * height > 1_200_000)

    lats = np.linspace(90.0 - 90.0 / height, -90.0 + 90.0 / height, height, dtype=np.float32)
    lons = np.linspace(-180.0 + 180.0 / width, 180.0 - 180.0 / width, width, dtype=np.float32)
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    cos_lat = np.maximum(0.18, np.cos(np.radians(lat_grid)))
    np_rng = np.random.default_rng(rng.randrange(1, 2**63 - 1))

    try:
        from worldgen.terrain_review import derive_terrain_controls
        terrain_controls = derive_terrain_controls(planet, hydrosphere, geology, config, output_dir=output_dir)
    except Exception:
        terrain_controls = {}

    ocean_target = clamp(float(getattr(hydrosphere, "ocean_fraction_target", 0.62) or 0.62), 0.05, 0.95)
    fragmentation = clamp(float(terrain_controls.get("fragmentation_tendency", 0.50) or 0.50), 0.0, 1.0)
    island_density = clamp(float(terrain_controls.get("island_density", 0.45) or 0.45), 0.0, 1.0)
    coastline_complexity_control = clamp(float(terrain_controls.get("coastline_complexity", 0.45) or 0.45), 0.0, 1.0)
    super_score = _effective_supercontinent_score(terrain_controls)
    terrain_style = str(terrain_controls.get("terrain_style", "derived_from_planet_physics") or "derived_from_planet_physics")
    style_mod = _terrain_style_modifiers(terrain_style)
    fragmentation = clamp(fragmentation + style_mod.get("fragmentation", 0.0), 0.0, 1.0)
    island_density = clamp(island_density + style_mod.get("island_density", 0.0), 0.0, 1.0)
    super_score = clamp(super_score + style_mod.get("supercontinent", 0.0), 0.0, 1.0)
    terrain_controls = {**terrain_controls, "effective_supercontinent_score": round(super_score, 3)}
    continent_target = int(round(2 + 8 * fragmentation + 3 * (1.0 - super_score) - 2 * max(0.0, ocean_target - 0.70)))
    continent_target = max(2, min(11, continent_target))
    if terrain_style == "earth_like_mixed_continents":
        continent_target = max(5, min(8, continent_target))
    if super_score > 0.78:
        continent_target = max(2, min(continent_target, rng.randint(2, 5)))
    elif fragmentation > 0.70:
        continent_target = max(continent_target, rng.randint(6, 9))

    continental = np.full((height, width), -0.42, dtype=np.float32)

    # Large continents and archipelago-capable lobes. These create the main
    # structure; later detail should modify, not replace, this structure.
    continent_count = continent_target
    super_anchor_lon = rng.uniform(-180.0, 180.0)
    super_anchor_lat = rng.triangular(-38.0, 38.0, 0.0)
    for _ in range(continent_count):
        if rng.random() < super_score * 0.72:
            lon0 = super_anchor_lon + rng.uniform(-70.0, 70.0) / max(0.30, math.cos(math.radians(super_anchor_lat)))
            lat0 = clamp(super_anchor_lat + rng.uniform(-30.0, 30.0), -65.0, 65.0)
        else:
            lon0 = rng.uniform(-180.0, 180.0)
            lat0 = rng.triangular(-50.0, 50.0, 0.0)
        amp = rng.uniform(1.05, 2.0) * (1.0 + 0.18 * super_score - 0.10 * fragmentation)
        lon_scale = rng.uniform(22.0, 64.0) * (1.0 + 0.18 * super_score - 0.12 * fragmentation)
        lat_scale = rng.uniform(16.0, 40.0) * (1.0 + 0.12 * super_score - 0.10 * fragmentation)
        dlon = _wrapped_lon_delta_array(lon_grid, lon0) * cos_lat
        dlat = lat_grid - lat0
        continental += amp * np.exp(-((dlon / lon_scale) ** 2 + (dlat / lat_scale) ** 2))

        # Add nearby lobes/peninsulas so continents are not perfect ovals.
        for _lobe in range(rng.randint(2, 5)):
            lon1 = lon0 + rng.uniform(-45.0, 45.0)
            lat1 = clamp(lat0 + rng.uniform(-24.0, 24.0), -68.0, 68.0)
            amp1 = amp * rng.uniform(0.22, 0.58)
            lon_scale1 = lon_scale * rng.uniform(0.35, 0.70)
            lat_scale1 = lat_scale * rng.uniform(0.35, 0.75)
            dlon = _wrapped_lon_delta_array(lon_grid, lon1) * cos_lat
            dlat = lat_grid - lat1
            continental += amp1 * np.exp(-((dlon / lon_scale1) ** 2 + (dlat / lat_scale1) ** 2))

    # Secondary microcontinents and rifted shelf fragments. These are driven by
    # fragmentation/ocean style controls and make the foundation layer less
    # dependent on a few smooth continental blobs.
    micro_count = int(round((4 + 18 * fragmentation + 12 * island_density) * (0.7 + 0.6 * ocean_target)))
    if super_score > 0.75:
        micro_count = max(2, int(micro_count * 0.45))
    for _micro in range(max(0, min(24, micro_count))):
        lon0 = rng.uniform(-180.0, 180.0)
        lat0 = rng.triangular(-58.0, 58.0, 0.0)
        amp = rng.uniform(0.22, 0.66) * (0.75 + 0.60 * fragmentation)
        lon_scale = rng.uniform(5.0, 20.0) * (0.8 + 0.5 * (1.0 - ocean_target))
        lat_scale = rng.uniform(3.5, 14.0) * (0.85 + 0.35 * (1.0 - ocean_target))
        dlon = _wrapped_lon_delta_array(lon_grid, lon0) * cos_lat
        dlat = lat_grid - lat0
        continental += amp * np.exp(-((dlon / lon_scale) ** 2 + (dlat / lat_scale) ** 2))

    # Long peninsula / pseudopodia chains grow from continent seeds so landmasses
    # are not just amoeba-like blobs. They can produce Florida/Baja/SE-Asia style
    # projections, narrow isthmuses, and island-producing tapering chains.
    peninsula_count = int(round((8 + 12 * coastline_complexity_control + 8 * fragmentation) * (0.75 + 0.30 * island_density)))
    if large_direct:
        peninsula_count = max(8, int(peninsula_count * 0.75))
    peninsula_count = max(6, min(26, peninsula_count))
    for _ in range(peninsula_count):
        lon_p = rng.uniform(-180.0, 180.0)
        lat_p = rng.triangular(-55.0, 55.0, 0.0)
        heading = rng.uniform(0.0, math.tau)
        amp = rng.uniform(0.20, 0.50) if large_direct else rng.uniform(0.20, 0.52)
        node_count = rng.randint(6, 14) if large_direct else rng.randint(6, 15)
        for node in range(node_count):
            heading += rng.uniform(-0.62, 0.62)
            step = rng.uniform(6.0, 18.0)
            lon_p += math.cos(heading) * step / max(0.28, math.cos(math.radians(lat_p)))
            lat_p = clamp(lat_p + math.sin(heading) * step, -72.0, 72.0)
            taper = 1.0 - node / max(1, node_count)
            lon_scale = rng.uniform(4.0, 17.0) * (0.55 + taper)
            lat_scale = rng.uniform(3.0, 14.0) * (0.55 + taper)
            dlon = _wrapped_lon_delta_array(lon_grid, lon_p) * cos_lat
            dlat = lat_grid - lat_p
            node_blob = np.exp(-((dlon / lon_scale) ** 2 + (dlat / lat_scale) ** 2))
            continental += amp * taper * node_blob * rng.uniform(0.75, 1.15)

    # Broad ocean basins keep continents separated and reduce artificial global arcs.
    ocean_basin_count = int(round(4 + 6 * fragmentation + 3 * ocean_target - 3 * super_score))
    for _ in range(max(3, min(11, ocean_basin_count))):
        lon0 = rng.uniform(-180.0, 180.0)
        lat0 = rng.uniform(-60.0, 60.0)
        amp = rng.uniform(0.42, 1.05)
        lon_scale = rng.uniform(28.0, 80.0)
        lat_scale = rng.uniform(18.0, 45.0)
        dlon = _wrapped_lon_delta_array(lon_grid, lon0) * cos_lat
        dlat = lat_grid - lat0
        continental -= amp * np.exp(-((dlon / lon_scale) ** 2 + (dlat / lat_scale) ** 2))

    # Allow occasional polar land/caps/archipelagos.  Earlier versions strongly
    # suppressed high-latitude continent seeds and then subtracted an additional
    # polar land penalty, so poles were almost always ocean.  Polar terrain is
    # now allowed as a profile-dependent possibility; climate/ice stages decide
    # what that land becomes.
    polar_chance = clamp(0.14 + 0.22 * fragmentation + 0.10 * (1.0 - ocean_target) - 0.08 * super_score, 0.08, 0.46)
    for hemi in (-1.0, 1.0):
        if rng.random() < polar_chance:
            lon0 = rng.uniform(-180.0, 180.0)
            lat0 = hemi * rng.uniform(62.0, 82.0)
            amp = rng.uniform(0.22, 0.74) * (0.85 + 0.45 * fragmentation)
            lon_scale = rng.uniform(18.0, 64.0)
            lat_scale = rng.uniform(5.0, 18.0)
            dlon = _wrapped_lon_delta_array(lon_grid, lon0) * cos_lat
            dlat = lat_grid - lat0
            continental += amp * np.exp(-((dlon / lon_scale) ** 2 + (dlat / lat_scale) ** 2))
            for _polar_lobe in range(rng.randint(1, 4)):
                lon1 = lon0 + rng.uniform(-55.0, 55.0)
                lat1 = clamp(lat0 + hemi * rng.uniform(-4.0, 10.0), -86.0, 86.0)
                dlon = _wrapped_lon_delta_array(lon_grid, lon1) * cos_lat
                dlat = lat_grid - lat1
                continental += amp * rng.uniform(0.18, 0.42) * np.exp(-((dlon / (lon_scale * rng.uniform(0.35, 0.75))) ** 2 + (dlat / (lat_scale * rng.uniform(0.45, 0.95))) ** 2))

    # Preserve the broad continent/ocean scaffold before adding texture. This
    # scaffold, not later valley/ridge/detail fields, defines where continents
    # broadly exist.
    continent_shape = continental.copy()

    # Very low-amplitude fractal structure, smoothed to avoid stripes.
    detail_large = _smooth_wrapped_array(np_rng.uniform(-1.0, 1.0, size=(height, width)).astype(np.float32), passes=7)
    detail_medium = _smooth_wrapped_array(np_rng.uniform(-1.0, 1.0, size=(height, width)).astype(np.float32), passes=3)
    continental += (0.12 + 0.08 * fragmentation) * detail_large + (0.04 + 0.05 * coastline_complexity_control) * detail_medium

    # Low-resolution province mosaic, upsampled and smoothed. This creates
    # continent-scale asymmetry and avoids the "round blob continent" look while
    # staying much cheaper than evaluating many full-resolution feature blobs.
    low_h = max(24, height // 96)
    low_w = max(48, width // 96)
    mosaic = np_rng.uniform(-1.0, 1.0, size=(low_h, low_w)).astype(np.float32)
    mosaic = _smooth_wrapped_array(mosaic, passes=4)
    mosaic = np.repeat(np.repeat(mosaic, int(math.ceil(height / low_h)), axis=0), int(math.ceil(width / low_w)), axis=1)[:height, :width]
    mosaic = _smooth_wrapped_array(mosaic, passes=9)
    continental += (0.12 + 0.18 * fragmentation + 0.06 * coastline_complexity_control) * mosaic

    # Keep a foundation copy before fracture/embayment carving. This foundation
    # controls the continent/ocean mask. Later negative fracture fields may carve
    # rift valleys and lowlands, but they should not accidentally open long blue
    # inland seas through every continent.
    continent_foundation = continental.copy()

    # Long but warped continental fractures/embayments. These are not mountains;
    # they reshape landmasses and coasts at medium-to-large scale.
    fracture_count = int(round(3 + 6 * fragmentation + 3 * float(terrain_controls.get("rift_strength", 0.45) or 0.45)))
    for _ in range(max(3, min(11, fracture_count))):
        lon0 = rng.uniform(-180.0, 180.0)
        lat0 = rng.uniform(-58.0, 58.0)
        heading = rng.uniform(0.0, math.tau)
        amp = rng.choice([-1.0, 1.0]) * rng.uniform(0.08, 0.22)
        for node in range(rng.randint(7, 13)):
            heading += rng.uniform(-0.75, 0.75)
            lon0 += math.cos(heading) * rng.uniform(8.0, 20.0) / max(0.28, math.cos(math.radians(lat0)))
            lat0 = clamp(lat0 + math.sin(heading) * rng.uniform(5.0, 15.0), -72.0, 72.0)
            dlon = _wrapped_lon_delta_array(lon_grid, lon0) * cos_lat
            dlat = lat_grid - lat0
            along = dlon * math.cos(heading) + dlat * math.sin(heading)
            across = -dlon * math.sin(heading) + dlat * math.cos(heading)
            fracture = np.exp(-((across / rng.uniform(3.5, 10.5)) ** 2 + (along / rng.uniform(10.0, 26.0)) ** 2))
            continental += amp * fracture * rng.uniform(0.55, 1.15)

    latitude_cooling_shape = 0.035 * (np.abs(lat_grid) / 90.0) ** 2
    continental -= latitude_cooling_shape
    continent_foundation -= latitude_cooling_shape
    continent_shape -= latitude_cooling_shape

    # Sea level is chosen from the broad continent/ocean scaffold. This keeps
    # detail fields from turning continental lowlands into ocean networks.
    # Coasts can still change later inside the coastal transition zone.
    foundation_for_mask = _lowpass_field_for_landmask(continent_shape)
    # Closing, island, and repair passes can move final land/ocean balance. Start
    # from a slightly more ocean-rich broad mask, but do not clamp away dry/ocean
    # test cases: the final target-fit pass below will measure and correct the
    # actual ocean fraction after terrain is assembled.
    compensation = 0.025 + 0.030 * coastline_complexity_control - 0.025 * max(0.0, island_density - 0.55)
    mask_ocean_target = clamp(ocean_target + compensation, 0.12, 0.92)
    sea_threshold = float(np.quantile(foundation_for_mask, mask_ocean_target))
    initial_structural_land = foundation_for_mask >= sea_threshold
    solid_land_field = _smooth_wrapped_array(initial_structural_land.astype(np.float32), passes=max(5, min(12, width // 260)))
    solid_threshold = float(np.quantile(solid_land_field, mask_ocean_target))
    structural_land = solid_land_field >= solid_threshold
    structural_land = _roughen_structural_coastlines(rng, structural_land, foundation_for_mask, sea_threshold, np_rng, strength=0.42 + 0.48 * coastline_complexity_control)
    # Remove unintentional land-mask holes here, before any elevation, erosion,
    # or shelf bathymetry exists. Rounded-rectangle lakes and strange inland
    # seas were mostly caused by low-frequency masks crossing the sea-level
    # threshold. We fill those artifacts at the source, then add deliberate
    # irregular rift/endorheic water bodies below.
    structural_land = _fill_unintended_inland_water_holes(structural_land)
    structural_land = _carve_irregular_seaways_and_marginal_seas(rng, structural_land, lons, lats, cos_lat, np_rng)
    structural_land = _add_structural_archipelago_chains(rng, structural_land, lons, lats, cos_lat, np_rng)
    if island_density > 0.58 or terrain_style in {"archipelago_world", "ocean_world", "volcanic_island_arc_world"}:
        structural_land = _add_structural_archipelago_chains(rng, structural_land, lons, lats, cos_lat, np_rng)
    structural_land = _close_narrow_water_channels(structural_land, radius=max(1, min(2, width // 1800)))
    # Do not carve standalone lakes/seas into the structural land mask.
    # Earlier terrain-stage lake carving could create pedestal lakes: water
    # bodies surrounded by artificial raised rims with no hydrologic outlet.
    # Lakes are now left to the hydrology stage, where they can be associated
    # with drainage basins and terminal sinks instead of being arbitrary holes
    # in the continent mask.
    intentional_inland_water = np.zeros_like(structural_land, dtype=bool)
    land_support = _smooth_wrapped_array(structural_land.astype(np.float32), passes=4)

    raw = continental.copy()
    mountain_field = np.zeros((height, width), dtype=np.float32)
    basin_field = np.zeros((height, width), dtype=np.float32)
    rift_field = np.zeros((height, width), dtype=np.float32)
    interior_relief_field = np.zeros((height, width), dtype=np.float32)
    shield_highland_field = np.zeros((height, width), dtype=np.float32)
    plateau_field = np.zeros((height, width), dtype=np.float32)

    # Continental interior uplift and basins.
    interior_feature_count = int(round((6 if large_direct else 10) + 10 * float(terrain_controls.get("interior_relief", 0.45) or 0.45) + 5 * fragmentation))
    for _ in range(interior_feature_count):
        lon0 = rng.uniform(-180.0, 180.0)
        lat0 = rng.uniform(-65.0, 65.0)
        amp = rng.uniform(-0.22, 0.34)
        lon_scale = rng.uniform(12.0, 38.0)
        lat_scale = rng.uniform(8.0, 26.0)
        dlon = _wrapped_lon_delta_array(lon_grid, lon0) * cos_lat
        dlat = lat_grid - lat0
        blob = np.exp(-((dlon / lon_scale) ** 2 + (dlat / lat_scale) ** 2))
        support_blob = blob * np.clip(land_support * 1.8, 0.0, 1.0)
        raw += amp * support_blob
        interior_relief_field += abs(amp) * support_blob
        if amp < 0:
            basin_field += (-amp) * support_blob
        else:
            shield_highland_field += amp * support_blob
            plateau_field += max(0.0, amp - 0.12) * support_blob

    # Land-weighted mountain ranges. Ranges are built from curved, broken
    # segments instead of single straight ellipses, then softened with foothills.
    # This avoids the artificial ruler-straight belts that earlier versions made.
    mountain_range_count = int(round(10 + 16 * float(terrain_controls.get("mountain_belt_strength", 1.0) or 1.0) / 2.0 + 5 * fragmentation))
    mountain_range_count = max(8, min(30, mountain_range_count))
    for _ in range(mountain_range_count):
        for _attempt in range(40):
            rr = rng.randrange(height)
            cc = rng.randrange(width)
            if land_support[rr, cc] > 0.32:
                lat0 = float(lats[rr])
                lon0 = float(lons[cc])
                break
        else:
            lon0 = rng.uniform(-180.0, 180.0)
            lat0 = rng.uniform(-60.0, 60.0)

        angle = rng.uniform(0.0, math.pi)
        amp_base = rng.uniform(0.24, 0.74) * geology.mountain_factor * (1.0 + style_mod.get("relief", 0.0))
        width_deg = rng.uniform(1.3, 4.8)
        segment_count = rng.randint(4, 7) if large_direct else rng.randint(6, 11)
        step = rng.uniform(4.5, 11.5)
        center_lon = lon0
        center_lat = lat0
        for seg in range(segment_count):
            angle += rng.uniform(-0.72, 0.72)
            center_lon += math.cos(angle) * step / max(0.25, math.cos(math.radians(center_lat)))
            center_lat = clamp(center_lat + math.sin(angle) * step, -72.0, 72.0)
            seg_amp = amp_base * rng.uniform(0.55, 1.15)
            seg_len = rng.uniform(5.0, 14.0)
            seg_width = width_deg * rng.uniform(0.75, 1.35)
            dlon = _wrapped_lon_delta_array(lon_grid, center_lon) * cos_lat
            dlat = lat_grid - center_lat
            along = dlon * math.cos(angle) + dlat * math.sin(angle)
            across = -dlon * math.sin(angle) + dlat * math.cos(angle)
            core = np.exp(-((across / seg_width) ** 2 + (along / seg_len) ** 2))
            foothills = 0.35 * np.exp(-((across / (seg_width * 3.0)) ** 2 + (along / (seg_len * 1.7)) ** 2))
            gap = rng.uniform(0.72, 1.0)
            mountain_field += seg_amp * gap * (core + foothills) * np.clip(land_support * 2.3, 0.0, 1.0)

            # Occasional short branch range.
            if rng.random() < (0.32 if large_direct else 0.52):
                branch_angle = angle + rng.choice([-1, 1]) * rng.uniform(0.55, 1.05)
                branch_len = rng.uniform(5.0, 18.0)
                branch_width = seg_width * rng.uniform(0.65, 1.05)
                branch_lon = center_lon + math.cos(branch_angle) * rng.uniform(3.0, 8.0)
                branch_lat = clamp(center_lat + math.sin(branch_angle) * rng.uniform(3.0, 8.0), -72.0, 72.0)
                dlon_b = _wrapped_lon_delta_array(lon_grid, branch_lon) * cos_lat
                dlat_b = lat_grid - branch_lat
                along_b = dlon_b * math.cos(branch_angle) + dlat_b * math.sin(branch_angle)
                across_b = -dlon_b * math.sin(branch_angle) + dlat_b * math.cos(branch_angle)
                branch = np.exp(-((across_b / branch_width) ** 2 + (along_b / branch_len) ** 2))
                mountain_field += seg_amp * rng.uniform(0.28, 0.62) * branch * np.clip(land_support * 2.2, 0.0, 1.0)

    # Compound orogenic provinces: longer mountain systems built from curved
    # chains of overlapping highland nodes, plus offset foothill/high-plateau
    # provinces. These are intentionally less linear than the segment belts
    # above and add medium-to-large scale mountain complexity.
    compound_count = rng.randint(10, 18)
    for _ in range(compound_count):
        for _attempt in range(50):
            rr = rng.randrange(height)
            cc = rng.randrange(width)
            if land_support[rr, cc] > 0.45:
                start_lat = float(lats[rr])
                start_lon = float(lons[cc])
                break
        else:
            start_lon = rng.uniform(-180.0, 180.0)
            start_lat = rng.uniform(-58.0, 58.0)

        node_count = rng.randint(4, 8) if large_direct else rng.randint(7, 14)
        heading = rng.uniform(0.0, math.pi * 2.0)
        curvature = rng.uniform(-0.28, 0.28)
        lon_node = start_lon
        lat_node = start_lat
        province_amp = rng.uniform(0.18, 0.52) * geology.mountain_factor
        side_bias = rng.choice([-1.0, 1.0])
        for node in range(node_count):
            heading += curvature + rng.uniform(-0.65, 0.65)
            step_deg = rng.uniform(5.0, 13.5)
            lon_node += math.cos(heading) * step_deg / max(0.28, math.cos(math.radians(lat_node)))
            lat_node = clamp(lat_node + math.sin(heading) * step_deg, -70.0, 70.0)

            local_width = rng.uniform(2.0, 8.0)
            local_len = rng.uniform(5.0, 16.0)
            lateral_offset = rng.uniform(-3.5, 3.5)
            offset_lon = lon_node + side_bias * math.cos(heading + math.pi / 2.0) * lateral_offset
            offset_lat = clamp(lat_node + side_bias * math.sin(heading + math.pi / 2.0) * lateral_offset, -72.0, 72.0)
            dlon = _wrapped_lon_delta_array(lon_grid, offset_lon) * cos_lat
            dlat = lat_grid - offset_lat
            along = dlon * math.cos(heading) + dlat * math.sin(heading)
            across = -dlon * math.sin(heading) + dlat * math.cos(heading)
            broken = rng.uniform(0.55, 1.15)
            core = np.exp(-((across / local_width) ** 2 + (along / local_len) ** 2))
            plateau = 0.28 * np.exp(-((across / (local_width * 4.4)) ** 2 + (along / (local_len * 2.8)) ** 2))
            mountain_field += province_amp * broken * (core + plateau) * np.clip(land_support * 2.4, 0.0, 1.0)

            # Foreland basins/valleys alongside some mountain systems.
            if rng.random() < 0.45:
                basin_offset = rng.uniform(3.0, 10.0) * -side_bias
                basin_lon = lon_node + math.cos(heading + math.pi / 2.0) * basin_offset
                basin_lat = clamp(lat_node + math.sin(heading + math.pi / 2.0) * basin_offset, -72.0, 72.0)
                dlon_b = _wrapped_lon_delta_array(lon_grid, basin_lon) * cos_lat
                dlat_b = lat_grid - basin_lat
                along_b = dlon_b * math.cos(heading) + dlat_b * math.sin(heading)
                across_b = -dlon_b * math.sin(heading) + dlat_b * math.cos(heading)
                foreland = np.exp(-((across_b / (local_width * 3.2)) ** 2 + (along_b / (local_len * 2.6)) ** 2))
                foreland_support = foreland * np.clip(land_support * 2.0, 0.0, 1.0)
                raw -= province_amp * 0.16 * foreland_support
                basin_field += province_amp * 0.16 * foreland_support

    raw += mountain_field

    # Large-scale tectonic provinces: shields, broad basins, and high plateaus
    # give continents more structure than simple blobs plus mountain ranges.
    for _ in range(rng.randint(12, 22)):
        lon0 = rng.uniform(-180.0, 180.0)
        lat0 = rng.uniform(-66.0, 66.0)
        amp = rng.choice([-1.0, 1.0]) * rng.uniform(0.055, 0.18)
        lon_scale = rng.uniform(18.0, 70.0)
        lat_scale = rng.uniform(12.0, 46.0)
        dlon = _wrapped_lon_delta_array(lon_grid, lon0) * cos_lat
        dlat = lat_grid - lat0
        province = np.exp(-((dlon / lon_scale) ** 2 + (dlat / lat_scale) ** 2))
        province_support = province * np.clip(land_support * 1.9, 0.0, 1.0)
        raw += amp * province_support
        interior_relief_field += abs(amp) * province_support
        if amp < 0:
            basin_field += (-amp) * province_support
        else:
            plateau_field += amp * province_support

    # Very broad continental provinces add medium/large-scale variation: old
    # cratons, sag basins, interior seas-in-waiting, and high plateaus. These
    # deliberately operate at a larger scale than the mountain strokes so the
    # map does not look like simple coast blobs plus straight ranges.
    for _ in range(rng.randint(8, 14)):
        lon0 = rng.uniform(-180.0, 180.0)
        lat0 = rng.uniform(-60.0, 60.0)
        amp = rng.choice([-1.0, 1.0]) * rng.uniform(0.10, 0.28)
        lon_scale = rng.uniform(35.0, 105.0)
        lat_scale = rng.uniform(18.0, 58.0)
        skew = rng.uniform(-0.75, 0.75)
        dlon = _wrapped_lon_delta_array(lon_grid, lon0) * cos_lat
        dlat = lat_grid - lat0
        rotated = dlon + skew * dlat
        province = np.exp(-((rotated / lon_scale) ** 2 + (dlat / lat_scale) ** 2))
        province_support = province * np.clip(land_support * 1.7, 0.0, 1.0)
        raw += amp * province_support
        interior_relief_field += abs(amp) * province_support
        if amp < 0:
            basin_field += (-amp) * province_support
        else:
            plateau_field += amp * province_support
            if lon_scale > 45.0 or lat_scale > 34.0:
                shield_highland_field += amp * province_support * 0.6

    # Arc-shaped highland/island-arc provinces. They add curved, broken
    # structure that is harder to mistake for straight drawn mountain lines.
    for _ in range(rng.randint(4, 8)):
        arc_lon = rng.uniform(-180.0, 180.0)
        arc_lat = rng.uniform(-55.0, 55.0)
        radius = rng.uniform(16.0, 48.0)
        start = rng.uniform(0.0, math.tau)
        sweep = rng.uniform(0.8, 2.2) * rng.choice([-1.0, 1.0])
        arc_amp = rng.uniform(0.10, 0.34) * geology.mountain_factor
        for node in range(rng.randint(6, 13)):
            t = start + sweep * node / 12.0 + rng.uniform(-0.12, 0.12)
            lon_n = arc_lon + math.cos(t) * radius / max(0.3, math.cos(math.radians(arc_lat)))
            lat_n = clamp(arc_lat + math.sin(t) * radius * rng.uniform(0.55, 1.05), -72.0, 72.0)
            scale_lon = rng.uniform(2.5, 9.0)
            scale_lat = rng.uniform(2.5, 10.0)
            dlon = _wrapped_lon_delta_array(lon_grid, lon_n) * cos_lat
            dlat = lat_grid - lat_n
            node_blob = np.exp(-((dlon / scale_lon) ** 2 + (dlat / scale_lat) ** 2))
            arc_support = node_blob * np.clip(land_support * 2.2, 0.0, 1.0)
            arc_local = arc_amp * rng.uniform(0.55, 1.25) * arc_support
            raw += arc_local
            mountain_field += arc_local * 0.28

    # First-pass pseudo plate drift/uplift model. Plate boundaries add warped
    # orogens, rifts, shelves, and uplift provinces that are not simply random
    # blobs. This is intentionally geometric rather than a full plate-tectonic
    # simulation, but it makes continents less smooth and gives mountains a
    # reason to form where plates collide or shear.
    plate_uplift, plate_rift, tectonic_diagnostics = _generate_plate_uplift_and_rift_fields(np_rng, lons, lats, cos_lat, rng, land_support, controls=terrain_controls, geology=geology)
    raw += plate_uplift * (0.22 + 0.26 * geology.mountain_factor)
    raw -= plate_rift * (0.10 + 0.14 * geology.erosion)
    mountain_field += np.clip(plate_uplift, 0.0, None) * (0.22 + 0.18 * geology.mountain_factor)
    rift_field += np.clip(plate_rift, 0.0, None)
    basin_field += np.clip(plate_rift, 0.0, None) * (0.18 + 0.10 * geology.erosion)
    interior_relief_field += np.clip(np.abs(plate_uplift) + plate_rift * 0.45, 0.0, None) * np.clip(land_support * 1.8, 0.0, 1.0)

    # Stage 3C.3 boundary-aware relief: use the low-resolution province and
    # boundary classes to add collision belts, rift valleys/shoulders, old
    # sutures, shields, sedimentary basins, and accreted terrane uplifts.
    try:
        from PIL import Image
        bc = np.asarray(tectonic_diagnostics.get("boundary_class"), dtype=np.uint8) if tectonic_diagnostics.get("boundary_class") is not None else None
        pt = np.asarray(tectonic_diagnostics.get("province_type"), dtype=np.uint8) if tectonic_diagnostics.get("province_type") is not None else None
        pa = np.asarray(tectonic_diagnostics.get("province_age_x1000"), dtype=np.float32) / 1000.0 if tectonic_diagnostics.get("province_age_x1000") is not None else None
        boundary_class_full = np.asarray(Image.fromarray(bc, mode="L").resize((width, height), Image.Resampling.NEAREST), dtype=np.uint8) if bc is not None else None
        province_type_full = np.asarray(Image.fromarray(pt, mode="L").resize((width, height), Image.Resampling.NEAREST), dtype=np.uint8) if pt is not None else None
        province_age_full = np.asarray(Image.fromarray(pa.astype(np.float32), mode="F").resize((width, height), Image.Resampling.BICUBIC), dtype=np.float32) if pa is not None else None
    except Exception:
        boundary_class_full = None
        province_type_full = None
        province_age_full = None

    if boundary_class_full is not None:
        collision = _smooth_wrapped_array(np.isin(boundary_class_full, [1, 6]).astype(np.float32), passes=max(1, min(5, width // 900))) * np.clip(land_support * 2.0, 0.0, 1.0)
        old_suture = _smooth_wrapped_array((boundary_class_full == 5).astype(np.float32), passes=max(1, min(6, width // 800))) * np.clip(land_support * 1.8, 0.0, 1.0)
        rift_zone = _smooth_wrapped_array((boundary_class_full == 2).astype(np.float32), passes=max(1, min(5, width // 950))) * np.clip(land_support * 1.7, 0.0, 1.0)
        transform_zone = _smooth_wrapped_array((boundary_class_full == 3).astype(np.float32), passes=max(1, min(4, width // 1000))) * np.clip(land_support * 1.6, 0.0, 1.0)
        segmented_noise = _smooth_wrapped_array(np_rng.uniform(0.55, 1.35, size=(height, width)).astype(np.float32), passes=2)
        collision_relief = collision * segmented_noise * (0.16 + 0.18 * geology.mountain_factor)
        suture_relief = old_suture * (0.055 + 0.06 * geology.mountain_factor)
        rift_valley = rift_zone * (0.12 + 0.08 * float(terrain_controls.get("rift_strength", 0.45) or 0.45))
        rift_shoulders = _smooth_wrapped_array(rift_zone, passes=3) * (0.065 + 0.055 * geology.mountain_factor)
        shear_highlands = transform_zone * 0.035
        raw += collision_relief + suture_relief + rift_shoulders + shear_highlands
        raw -= rift_valley
        mountain_field += collision_relief * 0.78 + suture_relief * 0.45 + shear_highlands * 0.35
        rift_field += rift_zone + plate_rift * 0.35
        basin_field += rift_valley * 0.95
        plateau_field += rift_shoulders * 0.55
        interior_relief_field += collision_relief + suture_relief + rift_shoulders + rift_valley * 0.65

    if province_type_full is not None:
        stable_shield = np.isin(province_type_full, [2, 8]).astype(np.float32)
        sedimentary = (province_type_full == 7).astype(np.float32)
        terrane = (province_type_full == 6).astype(np.float32)
        age_boost = province_age_full if province_age_full is not None else 0.5
        shield = _smooth_wrapped_array(stable_shield * np.asarray(age_boost, dtype=np.float32), passes=max(3, min(9, width // 500))) * np.clip(land_support * 1.4, 0.0, 1.0)
        sediment = _smooth_wrapped_array(sedimentary, passes=max(3, min(8, width // 600))) * np.clip(land_support * 1.6, 0.0, 1.0)
        terrane_uplift = _smooth_wrapped_array(terrane, passes=max(2, min(5, width // 850))) * np.clip(land_support * 1.8, 0.0, 1.0)
        raw += shield * (0.050 + 0.05 * float(terrain_controls.get("interior_relief", 0.45) or 0.45))
        raw -= sediment * (0.045 + 0.06 * geology.erosion)
        raw += terrane_uplift * (0.075 + 0.07 * geology.mountain_factor)
        shield_highland_field += shield
        basin_field += sediment
        plateau_field += shield * 0.55 + terrane_uplift * 0.35
        interior_relief_field += shield * 0.65 + sediment * 0.45 + terrane_uplift

    # Lone volcanic peaks or old shield peaks, again restricted to land.
    for _ in range(rng.randint(28, 54)):
        for _attempt in range(30):
            rr = rng.randrange(height)
            cc = rng.randrange(width)
            if land_support[rr, cc] > 0.45:
                lat0 = float(lats[rr])
                lon0 = float(lons[cc])
                break
        else:
            continue
        amp = rng.uniform(0.22, 0.72) * (0.65 + geology.volcanism * 0.35)
        scale = rng.uniform(1.0, 3.8)
        dlon = _wrapped_lon_delta_array(lon_grid, lon0) * cos_lat
        dlat = lat_grid - lat0
        peak = np.exp(-((dlon / scale) ** 2 + (dlat / scale) ** 2))
        raw += amp * peak * np.clip(land_support * 1.8, 0.0, 1.0)

    # Fractal detail: enough for texture, not enough to dominate structure.
    detail_coarse = _smooth_wrapped_array(np_rng.uniform(-1.0, 1.0, size=(height, width)).astype(np.float32), passes=5)
    detail_fine = _smooth_wrapped_array(np_rng.uniform(-1.0, 1.0, size=(height, width)).astype(np.float32), passes=1)
    raw += (0.13 * geology.surface_roughness * detail_coarse + 0.035 * geology.surface_roughness * detail_fine) * (0.45 + 0.75 * land_support)
    raw = _smooth_wrapped_array(raw, passes=3)

    # Different crust/material zones erode differently. Soft sedimentary-style
    # regions become smoother and more dissected; hard shield/igneous regions
    # preserve rough coasts and high relief. This also gives coastlines regional
    # character instead of one uniform roughness everywhere.
    erodibility = _generate_erodibility_field(np_rng, lon_grid, lat_grid, cos_lat, rng)
    coastal_zone = np.exp(-((raw - sea_threshold) / 0.18) ** 2).astype(np.float32)
    coast_roughness = np.clip(1.55 - erodibility + 0.28 * detail_coarse, 0.25, 1.85)
    rugged_coast_detail = _smooth_wrapped_array(np_rng.uniform(-1.0, 1.0, size=(height, width)).astype(np.float32), passes=1)
    smooth_raw = _smooth_wrapped_array(raw, passes=4)
    smooth_weight = coastal_zone * np.clip(erodibility - 0.85, 0.0, 0.75)
    raw = raw * (1.0 - smooth_weight * 0.78) + smooth_raw * (smooth_weight * 0.78)
    raw += coastal_zone * np.clip(coast_roughness - 0.70, 0.0, 1.25) * rugged_coast_detail * 0.105

    # Medium-scale coastline structure: broad bays, gulfs, peninsulas, and
    # rugged/smooth coastal provinces. These are larger than pixel noise but
    # smaller than whole continents, so coastlines vary like southwest Africa vs.
    # Norway/Chile/Alaska rather than having one uniform roughness.
    coastal_candidates = np.argwhere(coastal_zone > 0.34)
    if coastal_candidates.size:
        for _ in range(rng.randint(42, 76)):
            rr, cc = coastal_candidates[rng.randrange(len(coastal_candidates))]
            lon0 = float(lons[int(cc)])
            lat0 = float(lats[int(rr)])
            amp = rng.choice([-1.0, 1.0]) * rng.uniform(0.040, 0.215)
            lon_scale = rng.uniform(7.0, 34.0)
            lat_scale = rng.uniform(5.0, 24.0)
            dlon = _wrapped_lon_delta_array(lon_grid, lon0) * cos_lat
            dlat = lat_grid - lat0
            blob = np.exp(-((dlon / lon_scale) ** 2 + (dlat / lat_scale) ** 2))
            raw += amp * blob * np.clip(coastal_zone * 1.8 + 0.15 * land_support, 0.0, 1.0)

        # Larger embayments and continental shelves/peninsulas. These change
        # coastlines on a medium-to-large scale without turning them into pixel noise.
        for _ in range(rng.randint(8, 16)):
            rr, cc = coastal_candidates[rng.randrange(len(coastal_candidates))]
            lon0 = float(lons[int(cc)])
            lat0 = float(lats[int(rr)])
            amp = rng.choice([-1.0, 1.0]) * rng.uniform(0.09, 0.25)
            lon_scale = rng.uniform(18.0, 54.0)
            lat_scale = rng.uniform(10.0, 34.0)
            dlon = _wrapped_lon_delta_array(lon_grid, lon0) * cos_lat
            dlat = lat_grid - lat0
            blob = np.exp(-((dlon / lon_scale) ** 2 + (dlat / lat_scale) ** 2))
            raw += amp * blob * np.clip(coastal_zone * 1.15 + 0.30 * land_support, 0.0, 1.0)

        # Coastal province chains create larger fjorded coasts, island arcs,
        # shelves, and broad gulf/peninsula patterns. This specifically targets
        # the issue where every coast had good fine detail but simple medium and
        # large scale geometry.
        for _ in range(rng.randint(10, 20)):
            rr, cc = coastal_candidates[rng.randrange(len(coastal_candidates))]
            lon_c = float(lons[int(cc)])
            lat_c = float(lats[int(rr)])
            heading = rng.uniform(0.0, math.pi * 2.0)
            sign = rng.choice([-1.0, 1.0])
            chain_amp = sign * rng.uniform(0.055, 0.18)
            for _node in range(rng.randint(4, 9)):
                heading += rng.uniform(-0.55, 0.55)
                lon_c += math.cos(heading) * rng.uniform(5.0, 15.0) / max(0.28, math.cos(math.radians(lat_c)))
                lat_c = clamp(lat_c + math.sin(heading) * rng.uniform(4.0, 11.0), -72.0, 72.0)
                lon_scale = rng.uniform(5.0, 22.0)
                lat_scale = rng.uniform(4.0, 18.0)
                dlon = _wrapped_lon_delta_array(lon_grid, lon_c) * cos_lat
                dlat = lat_grid - lat_c
                chain_blob = np.exp(-((dlon / lon_scale) ** 2 + (dlat / lat_scale) ** 2))
                raw += chain_amp * rng.uniform(0.55, 1.25) * chain_blob * np.clip(coastal_zone * 1.9 + 0.22 * land_support, 0.0, 1.0)

        # Smooth-coast provinces remove medium detail in some places, creating
        # long simple coasts like Namibia/South Africa/Baja-style margins.
        coast_smooth = _smooth_wrapped_array(raw, passes=7)
        for _ in range(rng.randint(5, 9)):
            rr, cc = coastal_candidates[rng.randrange(len(coastal_candidates))]
            lon0 = float(lons[int(cc)])
            lat0 = float(lats[int(rr)])
            lon_scale = rng.uniform(18.0, 56.0)
            lat_scale = rng.uniform(9.0, 34.0)
            dlon = _wrapped_lon_delta_array(lon_grid, lon0) * cos_lat
            dlat = lat_grid - lat0
            mask = np.exp(-((dlon / lon_scale) ** 2 + (dlat / lat_scale) ** 2)) * coastal_zone
            mix = np.clip(mask * rng.uniform(0.25, 0.72), 0.0, 0.72)
            raw = raw * (1.0 - mix) + coast_smooth * mix


    # Continental-scale warped provinces and rift/basin systems. These operate
    # above the mountain/coastline-detail scale so each world has more varied
    # large landforms: sweeping highlands, broken shield interiors, rift valleys,
    # and broad sedimentary basins rather than simple smooth continents.
    for _ in range(rng.randint(7, 13)):
        start_lon = rng.uniform(-180.0, 180.0)
        start_lat = rng.uniform(-58.0, 58.0)
        heading = rng.uniform(0.0, math.tau)
        province_amp = rng.choice([-1.0, 1.0]) * rng.uniform(0.055, 0.20)
        chain_width = rng.uniform(8.0, 28.0)
        node_count = rng.randint(7, 15)
        lon_p = start_lon
        lat_p = start_lat
        for node in range(node_count):
            # Wander strongly; avoid straight province strokes.
            heading += rng.uniform(-0.72, 0.72)
            step = rng.uniform(6.0, 18.0)
            lon_p += math.cos(heading) * step / max(0.28, math.cos(math.radians(lat_p)))
            lat_p = clamp(lat_p + math.sin(heading) * step, -72.0, 72.0)
            local_width = chain_width * rng.uniform(0.55, 1.55)
            local_len = rng.uniform(8.0, 26.0)
            dlon = _wrapped_lon_delta_array(lon_grid, lon_p) * cos_lat
            dlat = lat_grid - lat_p
            blob = np.exp(-((dlon / local_len) ** 2 + (dlat / local_width) ** 2))
            jitter = 0.72 + 0.55 * detail_large
            raw += province_amp * rng.uniform(0.65, 1.35) * blob * jitter * np.clip(land_support * 1.8, 0.0, 1.0)

    # Large cratonic shields and sedimentary plains. These flatten or lift very
    # broad areas, giving continents recognizable interiors instead of uniform
    # medium relief everywhere.
    interior_feature_count = int(round((6 if large_direct else 10) + 10 * float(terrain_controls.get("interior_relief", 0.45) or 0.45) + 5 * fragmentation))
    for _ in range(interior_feature_count):
        lon0 = rng.uniform(-180.0, 180.0)
        lat0 = rng.uniform(-62.0, 62.0)
        uplift = rng.choice([-1.0, -0.65, 0.75, 1.0]) * rng.uniform(0.035, 0.13)
        lon_scale = rng.uniform(28.0, 86.0)
        lat_scale = rng.uniform(18.0, 62.0)
        rotation = rng.uniform(-0.95, 0.95)
        dlon = _wrapped_lon_delta_array(lon_grid, lon0) * cos_lat
        dlat = lat_grid - lat0
        x = dlon + rotation * dlat
        y = dlat - rotation * dlon * 0.18
        shield = np.exp(-((x / lon_scale) ** 2 + (y / lat_scale) ** 2))
        shield_support = shield * np.clip(land_support * 1.6, 0.0, 1.0)
        raw += uplift * shield_support
        interior_relief_field += abs(uplift) * shield_support
        if uplift < 0:
            basin_field += (-uplift) * shield_support
            # Soft basin floors should become smoother and lower.
            basin_smooth = _smooth_wrapped_array(raw, passes=5)
            mix = np.clip(shield * land_support * rng.uniform(0.10, 0.34), 0.0, 0.36)
            raw = raw * (1.0 - mix) + basin_smooth * mix
        else:
            shield_highland_field += uplift * shield_support
            plateau_field += max(0.0, uplift - 0.03) * shield_support

    # Tectonic texture provinces: broad escarpments, tilted blocks, and broken
    # basin chains. These add continental-scale shape without returning to
    # straight mountain strokes. They create more interesting medium/large land
    # without relying only on small-scale noise.
    for _ in range(rng.randint(5, 9)):
        start_lon = rng.uniform(-180.0, 180.0)
        start_lat = rng.uniform(-58.0, 58.0)
        heading = rng.uniform(0.0, math.tau)
        width_deg = rng.uniform(5.0, 16.0)
        amp = rng.choice([-1.0, 1.0]) * rng.uniform(0.045, 0.135)
        lon_node = start_lon
        lat_node = start_lat
        for _node in range(rng.randint(5, 10)):
            heading += rng.uniform(-0.80, 0.80)
            lon_node += math.cos(heading) * rng.uniform(8.0, 22.0) / max(0.28, math.cos(math.radians(lat_node)))
            lat_node = clamp(lat_node + math.sin(heading) * rng.uniform(6.0, 18.0), -72.0, 72.0)
            dlon = _wrapped_lon_delta_array(lon_grid, lon_node) * cos_lat
            dlat = lat_grid - lat_node
            along = dlon * math.cos(heading) + dlat * math.sin(heading)
            across = -dlon * math.sin(heading) + dlat * math.cos(heading)
            block = np.exp(-((along / rng.uniform(9.0, 28.0)) ** 2 + (across / width_deg) ** 2))
            asym = np.tanh(across / max(1.5, width_deg * 0.45))
            block_support = block * np.clip(land_support * 1.9, 0.0, 1.0)
            local_relief = amp * block * (0.55 + 0.45 * asym) * np.clip(land_support * 1.9, 0.0, 1.0)
            raw += local_relief
            interior_relief_field += np.abs(local_relief)
            if amp < 0:
                basin_field += (-amp) * block_support
            else:
                plateau_field += amp * block_support * 0.55

    # Large sediment aprons and interior plains downstream of highlands. This
    # makes some continents contain broad low-gradient surfaces instead of only
    # rough uplands and mountains.
    plains_seed = _smooth_wrapped_array(np.clip(raw - _smooth_wrapped_array(raw, passes=9), 0.0, None), passes=6)
    if float(plains_seed.max()) > 0:
        plains = np.clip(plains_seed / float(plains_seed.max()), 0.0, 1.0)
        plain_mask = np.clip((1.0 - plains) * land_support, 0.0, 1.0)
        raw = raw * (1.0 - 0.055 * plain_mask) + _smooth_wrapped_array(raw, passes=5) * (0.055 * plain_mask)

    # Use the original structural threshold as the authoritative continent mask,
    # then only allow land/sea flips near the actual structural coastline.
    #
    # Without this guard, later rift/basin/coastal-detail fields can push narrow
    # eroded valleys below sea level across continental interiors. On high-res
    # maps those become unrealistic blue shallow-sea networks. True inland seas
    # should be generated intentionally as broad basins, not as accidental river
    # valley flooding.
    # Keep intentional lakes/seas as water and keep shelf bathymetry from adding
    # new one-cell land ribbons outside the structural coastline. Candidate land
    # may erode/cut existing coast into bays, but it may not promote shallow
    # shelf cells into land. Oceanic islands/arcs are generated explicitly later.
    raw = np.where(intentional_inland_water, np.minimum(raw, sea_threshold - 0.08), raw)
    candidate_land = raw >= sea_threshold
    candidate_land[intentional_inland_water] = False
    coastal_transition = _land_ocean_transition_zone(structural_land, radius=max(3, min(12, width // 300)))
    land_mask = structural_land.copy()
    coastal_cut = coastal_transition & structural_land & (~candidate_land)
    land_mask[coastal_cut] = False
    land_mask[intentional_inland_water] = False
    land_mask = _close_narrow_water_channels(land_mask, radius=max(1, min(5, width // 1000)))
    land_mask[intentional_inland_water] = False

    land_scale = 4300.0 * geology.mountain_factor / max(planet.surface_gravity_g, 0.55)
    ocean_scale = 5200.0 * (0.88 + hydrosphere.ocean_fraction_target * 0.30)

    max_above = max(float(raw[land_mask].max() - sea_threshold) if land_mask.any() else 1.0, 1e-6)
    max_below = max(float(sea_threshold - raw[~land_mask].min()) if (~land_mask).any() else 1.0, 1e-6)
    above = np.clip((raw - sea_threshold) / max_above, 0.0, 1.0)
    below = np.clip((sea_threshold - raw) / max_below, 0.0, 1.0)
    elevation = np.where(
        land_mask,
        np.rint((above ** 0.74) * land_scale),
        -np.rint((below ** 0.90) * ocean_scale),
    ).astype(np.int32)

    # Add range/peak height after base scaling so mountains remain legible.
    mountain_extra = np.rint((mountain_field / max(0.001, float(mountain_field.max()))) * (620.0 * geology.mountain_factor)).astype(np.int32) if float(mountain_field.max()) > 0 else 0
    elevation = np.where(land_mask, elevation + mountain_extra, elevation)

    # Terrain Revisit 2: shape the ocean floor before final coastal/island
    # passes. This adds mid-ocean ridges, trenches, fracture zones, seamounts,
    # and abyssal-plain deepening from the existing province/boundary skeleton.
    elevation, ocean_floor_diagnostics = _apply_ocean_floor_tectonics(
        elevation,
        land_mask,
        boundary_class_full,
        province_type_full,
        lons,
        lats,
        cos_lat,
        rng,
        np_rng,
        geology,
        terrain_controls,
    )

    # Erosion/deposition makes river valleys and gentler lowlands. Keep the
    # structural coastline from the land/ocean model; otherwise channel erosion
    # can flood every small valley and turn all coasts into artificial fjords.
    structural_land_mask = land_mask.copy()
    elevation, erosion_diagnostics, erosion_meta = _apply_erosion_and_deposition(
        elevation,
        land_mask,
        geology,
        rng,
        erodibility,
        terrain_controls=terrain_controls,
        mountain_field=mountain_field,
        basin_field=basin_field,
        rift_field=rift_field,
        shield_highland_field=shield_highland_field,
        plateau_field=plateau_field,
    )
    land_mask = structural_land_mask

    # Remove very tiny islands as terrain noise; keep them submerged so coastlines
    # and drainage basins focus on meaningful landmasses.
    land_mask = _remove_small_land_components(land_mask, min_size=max(12, int(width * height * 0.000035)))
    land_mask, elevation = _repair_artificial_coast_and_inland_water(land_mask, elevation)
    land_mask, elevation = _add_oceanic_islands_and_arcs(rng, land_mask, elevation, lons, lats, cos_lat, sea_threshold, max_islands=max(80, width // 18))
    land_mask, elevation = _add_inland_sea_islands(rng, land_mask, elevation, lons, lats, cos_lat)
    land_mask, elevation = _repair_artificial_coast_and_inland_water(land_mask, elevation)
    land_mask, elevation = _remove_low_coastal_filaments(land_mask, elevation)
    land_mask, elevation = _remove_extreme_ribbon_land_components(land_mask, elevation)
    land_mask, elevation = _apply_coast_shelf_island_style_pass(
        rng,
        land_mask,
        elevation,
        lons,
        lats,
        cos_lat,
        terrain_controls,
        geology,
        np_rng,
        boundary_class_full=boundary_class_full,
        province_type_full=province_type_full,
    )
    land_mask, elevation, ocean_fit_info = _fit_ocean_fraction_target(elevation, land_mask, ocean_target, np_rng, max_error=0.025)
    elevation = _enforce_land_ocean_sign_with_lowland_variation(land_mask, elevation.astype(np.float32), np_rng)

    shift_cols = _best_longitude_shift_to_ocean_gap(land_mask)
    if shift_cols:
        land_mask = np.roll(land_mask, shift_cols, axis=1)
        elevation = np.roll(elevation, shift_cols, axis=1)
        if boundary_class_full is not None:
            boundary_class_full = np.roll(boundary_class_full, shift_cols, axis=1)
        if province_type_full is not None:
            province_type_full = np.roll(province_type_full, shift_cols, axis=1)
        for diag_key in [
            "plate_id",
            "boundary_class",
            "province_type",
            "province_age_x1000",
            "boundary_strength_x1000",
            "boundary_width_x1000",
        ]:
            if tectonic_diagnostics.get(diag_key) is not None:
                diag_arr = np.asarray(tectonic_diagnostics[diag_key], dtype=np.int32)
                diag_shift = int(round(shift_cols * diag_arr.shape[1] / max(1, width)))
                tectonic_diagnostics[diag_key] = np.roll(diag_arr, diag_shift, axis=1).astype(int).tolist()
        for diag_key, diag_value in list(erosion_diagnostics.items()):
            if diag_value is not None:
                diag_arr = np.asarray(diag_value, dtype=np.int32)
                diag_shift = int(round(shift_cols * diag_arr.shape[1] / max(1, width)))
                erosion_diagnostics[diag_key] = np.roll(diag_arr, diag_shift, axis=1).astype(int).tolist()
        for diag_key in [
            "ocean_floor_class_full",
            "ridge_field_full",
            "trench_field_full",
            "fracture_field_full",
            "seamount_field_full",
        ]:
            if ocean_floor_diagnostics.get(diag_key) is not None:
                ocean_floor_diagnostics[diag_key] = np.roll(np.asarray(ocean_floor_diagnostics[diag_key]), shift_cols, axis=1)

    terrain_coast_style_class, terrain_shelf_width_x1000, terrain_coast_ruggedness_x1000, terrain_island_origin_class = _coast_shelf_island_diagnostic_fields(
        land_mask,
        elevation,
        terrain_controls,
        boundary_class_full=boundary_class_full,
        province_type_full=province_type_full,
    )
    terrain_island_shape_complexity_x1000, island_shape_meta = _island_shape_complexity_diagnostic(land_mask, elevation)

    land_values = elevation[land_mask]
    ocean_values = elevation[~land_mask]
    ocean_fraction = 1.0 - (int(land_mask.sum()) / float(width * height))

    return TerrainMap(
        width=width,
        height=height,
        elevation_m=elevation.tolist(),
        is_land=land_mask.tolist(),
        min_elevation_m=int(elevation.min()),
        max_elevation_m=int(elevation.max()),
        mean_land_elevation_m=float(land_values.mean()) if land_values.size else 0.0,
        mean_ocean_depth_m=float(ocean_values.mean()) if ocean_values.size else 0.0,
        ocean_fraction=ocean_fraction,
        land_fraction=1.0 - ocean_fraction,
        source="procedural plate-inspired structured terrain",
        planet_radius_earth=float(planet.radius_earth),
        tectonic_plate_id=tectonic_diagnostics.get("plate_id"),
        tectonic_boundary_class=tectonic_diagnostics.get("boundary_class"),
        tectonic_province_type=tectonic_diagnostics.get("province_type"),
        tectonic_province_age_x1000=tectonic_diagnostics.get("province_age_x1000"),
        tectonic_boundary_strength_x1000=tectonic_diagnostics.get("boundary_strength_x1000"),
        tectonic_boundary_width_x1000=tectonic_diagnostics.get("boundary_width_x1000"),
        terrain_mountain_strength_x1000=diagf(mountain_field * land_mask.astype(np.float32)),
        terrain_basin_field_x1000=diagf(basin_field * land_mask.astype(np.float32)),
        terrain_rift_field_x1000=diagf(rift_field * land_mask.astype(np.float32)),
        terrain_interior_relief_x1000=diagf(interior_relief_field * land_mask.astype(np.float32)),
        terrain_shield_highland_x1000=diagf(shield_highland_field * land_mask.astype(np.float32)),
        terrain_plateau_x1000=diagf(plateau_field * land_mask.astype(np.float32)),
        terrain_coast_style_class=terrain_coast_style_class,
        terrain_shelf_width_x1000=terrain_shelf_width_x1000,
        terrain_coast_ruggedness_x1000=terrain_coast_ruggedness_x1000,
        terrain_island_origin_class=terrain_island_origin_class,
        terrain_ocean_floor_class=diagc(ocean_floor_diagnostics.get("ocean_floor_class_full")),
        terrain_mid_ocean_ridge_x1000=diagf(ocean_floor_diagnostics.get("ridge_field_full")),
        terrain_trench_x1000=diagf(ocean_floor_diagnostics.get("trench_field_full")),
        terrain_fracture_zone_x1000=diagf(ocean_floor_diagnostics.get("fracture_field_full")),
        terrain_seamount_x1000=diagf(ocean_floor_diagnostics.get("seamount_field_full")),
        terrain_island_shape_complexity_x1000=terrain_island_shape_complexity_x1000,
        terrain_erosion_strength_x1000=erosion_diagnostics.get("terrain_erosion_strength_x1000"),
        terrain_deposition_field_x1000=erosion_diagnostics.get("terrain_deposition_field_x1000"),
        terrain_valley_corridor_x1000=erosion_diagnostics.get("terrain_valley_corridor_x1000"),
        terrain_sediment_supply_x1000=erosion_diagnostics.get("terrain_sediment_supply_x1000"),
        terrain_coastal_plain_x1000=erosion_diagnostics.get("terrain_coastal_plain_x1000"),
        terrain_alluvial_fan_x1000=erosion_diagnostics.get("terrain_alluvial_fan_x1000"),
        terrain_floodplain_x1000=erosion_diagnostics.get("terrain_floodplain_x1000"),
        terrain_maturity_x1000=erosion_diagnostics.get("terrain_maturity_x1000"),
        terrain_relief_delta_m=erosion_diagnostics.get("terrain_relief_delta_m"),
        crust_type=_build_crust_diagnostic(land_mask, elevation, boundary_class_full=boundary_class_full, province_type_full=province_type_full),
        terrain_diagnostics={
            "generation_controls": terrain_controls,
            "ocean_target_fit": ocean_fit_info,
            "ocean_floor": ocean_floor_diagnostics.get("meta", {}),
            "island_shape": island_shape_meta,
            "erosion_deposition": erosion_meta,
        },
    )






def _island_shape_complexity_diagnostic(land_mask, elevation):
    """Return a compact 0..1000 raster showing non-oval island complexity.

    High values mark island coastlines with lobes, chains, and irregular edges.
    Low values on islands usually indicate oval/blob-like geometry and are used
    by the Stage 3.5 review page to diagnose remaining island-shape problems.
    """
    try:
        import numpy as np
        from scipy import ndimage
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("NumPy, SciPy, and Pillow are required for island complexity diagnostics. Install with: pip install -r requirements.txt") from exc
    land = land_mask.astype(bool, copy=False)
    h, w = land.shape
    labels, count = ndimage.label(land, structure=np.array([[0,1,0],[1,1,1],[0,1,0]], dtype=np.uint8))
    if count <= 0:
        return None, {"island_shape_complexity_mean": 0.0, "low_complexity_island_count": 0}
    sizes = np.bincount(labels.ravel()); sizes[0] = 0
    island_limit = max(4, int(h * w * 0.0038))
    complexity = np.zeros((h, w), dtype=np.float32)
    values = []
    low_count = 0
    objects = ndimage.find_objects(labels)
    for label_id, slc in enumerate(objects, start=1):
        if slc is None:
            continue
        size = int(sizes[label_id]) if label_id < len(sizes) else 0
        if size <= 0 or size > island_limit:
            continue
        # Use a local mask padded by one cell so perimeter is measured against ocean.
        local = labels[slc] == label_id
        area = float(size)
        north = np.vstack((local[0:1, :], local[:-1, :]))
        south = np.vstack((local[1:, :], local[-1:, :]))
        west = np.roll(local, 1, axis=1)
        east = np.roll(local, -1, axis=1)
        edge = local & ((~north) | (~south) | (~west) | (~east))
        perimeter = float(edge.sum())
        compactness = perimeter / max(1.0, math.sqrt(area))
        bbox_h = max(1, slc[0].stop - slc[0].start)
        bbox_w = max(1, slc[1].stop - slc[1].start)
        elongation = max(bbox_h, bbox_w) / max(1.0, min(bbox_h, bbox_w))
        value = clamp((compactness - 3.35) / 4.0 + min(1.0, (elongation - 1.0) / 3.0) * 0.35, 0.0, 1.0)
        if value < 0.22 and size > 5:
            low_count += 1
        values.append(value)
        complexity[labels == label_id] = value
    resized = np.asarray(Image.fromarray(complexity, mode="F").resize((512, 256), Image.Resampling.BICUBIC), dtype=np.float32)
    meta = {
        "island_shape_complexity_mean": round(float(sum(values) / len(values)) if values else 0.0, 4),
        "low_complexity_island_count": int(low_count),
        "island_shape_metric_note": "0=oval/blob-like island components; 1=lobed, chained, or irregular island components.",
    }
    return np.rint(np.clip(resized, 0.0, 1.0) * 1000.0).astype(int).tolist(), meta


def _coast_shelf_island_diagnostic_fields(land_mask, elevation, terrain_controls: dict | None = None, boundary_class_full=None, province_type_full=None):
    """Classify final coast, shelf, and island behavior for Stage 3C.4 review.

    Classes are deliberately broad and diagnostic-oriented, not a full coastal
    process model:
      coast style: 0 non-coast/background, 1 passive/smooth plain, 2 rugged/fjorded,
                   3 rifted/gulf margin, 4 volcanic/arc margin, 5 shelf/delta plain,
                   6 mixed/irregular margin.
      island origin: 0 water/non-island, 1 continent/large land, 2 shelf island,
                     3 volcanic/arc island, 4 microcontinent/terrane, 5 hotspot/high island.
    """
    try:
        import numpy as np
        from scipy import ndimage
    except ImportError as exc:
        raise RuntimeError("NumPy and SciPy are required for coastline diagnostics. Install with: pip install -r requirements.txt") from exc

    controls = terrain_controls or {}
    land = land_mask.astype(bool, copy=False)
    elev = elevation.astype(np.float32, copy=False)
    h, w = land.shape
    ocean = ~land
    north = np.vstack((land[0:1, :], land[:-1, :]))
    south = np.vstack((land[1:, :], land[-1:, :]))
    west = np.roll(land, 1, axis=1)
    east = np.roll(land, -1, axis=1)
    coast = land & ((~north) | (~south) | (~west) | (~east))

    # Shelf diagnostics use continent-scale source land, not any small island.
    # This avoids bullseye shelf halos around volcanic islands and lets deep
    # ocean approach active continental margins.
    continent_support = _continent_scale_land_mask(land, min_fraction=0.0025)
    if not continent_support.any():
        continent_support = land
    ocean_dist = ndimage.distance_transform_edt(~continent_support)
    land_dist = ndimage.distance_transform_edt(land)
    shelf_factor = clamp(float(controls.get("shelf_width_factor", 0.55) or 0.55), 0.0, 2.0)
    shelf_radius = max(2.0, min(46.0, (5.0 + 16.0 * shelf_factor) * max(0.45, w / 2048.0)))
    margin_factor = np.full((h, w), 0.45, dtype=np.float32)

    elev_n = np.vstack((elev[0:1, :], elev[:-1, :]))
    elev_s = np.vstack((elev[1:, :], elev[-1:, :]))
    elev_w = np.roll(elev, 1, axis=1)
    elev_e = np.roll(elev, -1, axis=1)
    relief = np.maximum.reduce([np.abs(elev - elev_n), np.abs(elev - elev_s), np.abs(elev - elev_w), np.abs(elev - elev_e)])
    ruggedness = np.clip(relief / max(1.0, np.percentile(relief[coast], 92) if coast.any() else 550.0), 0.0, 1.0).astype(np.float32)

    active = np.zeros_like(land, dtype=bool)
    rift = np.zeros_like(land, dtype=bool)
    arc = np.zeros_like(land, dtype=bool)
    passive = np.zeros_like(land, dtype=bool)
    if boundary_class_full is not None:
        b = np.asarray(boundary_class_full, dtype=np.uint8)
        if b.shape == land.shape:
            active = np.isin(b, [1, 3, 5, 6])
            rift = b == 2
            arc = b == 6
            passive = b == 4
    if province_type_full is not None:
        pt = np.asarray(province_type_full, dtype=np.uint8)
        if pt.shape == land.shape:
            arc |= pt == 5
            rift |= pt == 4
            passive |= pt == 3

    margin_factor[passive] = 1.15
    margin_factor[rift] = np.maximum(margin_factor[rift], 0.95)
    margin_factor[active | arc] = np.minimum(margin_factor[active | arc], 0.28)
    shelf_field = np.exp(-ocean_dist / shelf_radius).astype(np.float32) * ocean.astype(np.float32)
    shelf_field *= np.clip(margin_factor, 0.15, 1.35)
    shelf_field *= np.clip((elev + 2600.0) / 2600.0, 0.0, 1.0)

    style = np.zeros_like(elev, dtype=np.uint8)
    low_coast = coast & (elev < 180)
    high_relief_coast = coast & ((ruggedness > 0.48) | (elev > 900))
    style[coast] = 6
    style[coast & passive & ~high_relief_coast] = 1
    style[coast & low_coast & (shelf_field > 0.16)] = 5
    style[coast & rift] = 3
    style[coast & high_relief_coast] = 2
    style[coast & arc] = 4

    island_origin = np.zeros_like(style, dtype=np.uint8)
    labels, count = ndimage.label(land, structure=np.array([[0,1,0],[1,1,1],[0,1,0]], dtype=np.uint8))
    if count > 0:
        sizes = np.bincount(labels.ravel())
        sizes[0] = 0
        world = h * w
        island_limit = max(4, int(world * 0.0038))
        small_limit = max(3, int(world * 0.00065))
        island_origin[land] = 1
        objects = ndimage.find_objects(labels)
        for label_id, slc in enumerate(objects, start=1):
            if slc is None:
                continue
            size = int(sizes[label_id]) if label_id < len(sizes) else 0
            if size <= 0:
                continue
            mask = labels[slc] == label_id
            full_mask = labels == label_id
            if size > island_limit:
                island_origin[full_mask] = 1
                continue
            comp_elev = elev[full_mask]
            comp_arc = bool(np.mean(arc[full_mask]) > 0.04) if arc.any() else False
            comp_rift = bool(np.mean(rift[full_mask]) > 0.06) if rift.any() else False
            comp_shelf = bool(np.mean(shelf_field[full_mask]) > 0.10)
            high = float(comp_elev.max()) if comp_elev.size else 0.0
            if comp_arc:
                code = 3
            elif comp_rift or size > small_limit * 3:
                code = 4
            elif comp_shelf:
                code = 2
            elif high > 850:
                code = 5
            else:
                code = 2
            island_origin[full_mask] = code

    shelf_x1000 = np.rint(np.clip(shelf_field, 0.0, 1.0) * 1000.0).astype(np.int16)
    rugged_x1000 = np.rint(np.clip(ruggedness * coast.astype(np.float32), 0.0, 1.0) * 1000.0).astype(np.int16)
    return style.tolist(), shelf_x1000.tolist(), rugged_x1000.tolist(), island_origin.tolist()


def _apply_coast_shelf_island_style_pass(
    rng: random.Random,
    land_mask,
    elevation,
    lons,
    lats,
    cos_lat,
    terrain_controls: dict | None,
    geology: GeologyState,
    np_rng,
    boundary_class_full=None,
    province_type_full=None,
):
    """Stage 3C.4 procedural coast/shelf/island pass.

    This pass acts after broad relief and before final ocean-target fitting.  It
    does not yet simulate waves, glaciation, or sediment transport, but it gives
    coasts regionally distinct behavior: passive shelves smooth out, active/high
    relief margins get narrow inlets, rifted margins get gulfs, and islands are
    generated as chains/fragments rather than mostly isolated ovals.
    """
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("NumPy is required for coastline generation. Install it with: pip install numpy") from exc

    controls = terrain_controls or {}
    h, w = land_mask.shape
    land = land_mask.astype(bool, copy=True)
    elev = elevation.astype(np.int32, copy=True)
    if not land.any() or land.all():
        return land, elev

    lon_grid, lat_grid = np.meshgrid(lons, lats)
    coastline_complexity = clamp(float(controls.get("coastline_complexity", 0.45) or 0.45), 0.0, 1.0)
    island_density = clamp(float(controls.get("island_density", 0.45) or 0.45), 0.0, 1.0)
    shelf_width = clamp(float(controls.get("shelf_width_factor", 0.55) or 0.55), 0.0, 2.0)
    rugged_control = clamp(float(controls.get("coastal_ruggedness", 0.45) or 0.45), 0.0, 1.0)
    fjord = clamp(float(controls.get("fjord_tendency", 0.35) or 0.35), 0.0, 1.0)
    coastal_plain = clamp(float(controls.get("coastal_plain_bias", 0.45) or 0.45), 0.0, 1.0)
    island_irregularity = clamp(float(controls.get("island_shape_irregularity", 0.45) or 0.45), 0.0, 1.0)

    active = np.zeros_like(land, dtype=bool)
    rift = np.zeros_like(land, dtype=bool)
    arc = np.zeros_like(land, dtype=bool)
    passive = np.zeros_like(land, dtype=bool)
    if boundary_class_full is not None:
        b = np.asarray(boundary_class_full, dtype=np.uint8)
        if b.shape == land.shape:
            active = np.isin(b, [1, 3, 5, 6])
            rift = b == 2
            arc = b == 6
            passive = b == 4
    if province_type_full is not None:
        pt = np.asarray(province_type_full, dtype=np.uint8)
        if pt.shape == land.shape:
            arc |= pt == 5
            rift |= pt == 4
            passive |= pt == 3

    transition_radius = max(3, min(22, int(round(w / 360 * (0.7 + 0.55 * coastline_complexity)))))
    coast_zone = _land_ocean_transition_zone(land, radius=transition_radius)
    coast_land = coast_zone & land
    coast_ocean = coast_zone & (~land)
    if not coast_land.any():
        return land, elev

    # Shallow shelves: do not create land, and do not radiate shelves from every
    # island.  Broad shelves are attached mainly to continent-scale land and
    # passive/rifted margins; active/arc coasts keep narrow shelves and can have
    # deep water nearby.
    try:
        from scipy import ndimage
        continent_support = _continent_scale_land_mask(land, min_fraction=0.0025)
        if not continent_support.any():
            continent_support = land
        ocean_dist_continent = ndimage.distance_transform_edt(~continent_support)
        margin_factor = np.full((h, w), 0.45, dtype=np.float32)
        margin_factor[passive] = 1.15
        margin_factor[rift] = np.maximum(margin_factor[rift], 0.95)
        margin_factor[active | arc] = np.minimum(margin_factor[active | arc], 0.28)
        broad_radius = max(2.5, min(54.0, (5.0 + 16.0 * shelf_width) * max(0.45, w / 2048.0)))
        shelf_influence = np.exp(-ocean_dist_continent / broad_radius).astype(np.float32) * (~land).astype(np.float32)
        shelf_influence *= np.clip(margin_factor, 0.15, 1.35)
        shelf_noise = _smooth_wrapped_array(np_rng.uniform(0.72, 1.22, size=(h, w)).astype(np.float32), passes=2)
        shelf_target = -np.rint((170.0 + 560.0 * shelf_width) * shelf_influence * shelf_noise).astype(np.int32)
        shelf_cells = (~land) & (shelf_influence > 0.035)
        # Never let shelves flatten all near-coast ocean to a uniform shallow
        # band; retain depth variation and let active margins remain steep.
        elev[shelf_cells] = np.maximum(elev[shelf_cells], np.minimum(-8, shelf_target[shelf_cells]))
    except Exception:
        shelf_influence = np.zeros_like(elev, dtype=np.float32)

    # Passive/coastal-plain margins are smoother and lower. This creates South-
    # Africa/Namibia/Baja-style long simple margins where the profile supports it.
    passive_zone = coast_land & (passive | (~active & (np_rng.random((h, w)) < coastal_plain * 0.18)))
    if passive_zone.any():
        smooth_elev = _smooth_wrapped_array(elev.astype(np.float32), passes=max(2, min(6, w // 700)))
        mix = np.clip((0.18 + 0.34 * coastal_plain) * passive_zone.astype(np.float32), 0.0, 0.48)
        elev = np.rint(elev * (1.0 - mix) + smooth_elev * mix).astype(np.int32)
        elev[passive_zone] = np.minimum(elev[passive_zone], np.maximum(20, elev[passive_zone]))

    # Carve high-relief inlets, fjords, and rift gulfs with elongated, wandering
    # masks. Restrict to coastal transition zones so interiors are not randomly
    # cut apart.
    candidate_noise = np_rng.random((h, w)) < (0.08 + 0.24 * coastline_complexity)
    candidate = np.argwhere(coast_land & (active | rift | (elev > 650) | candidate_noise))
    carve_count = int(round(5 + 13 * coastline_complexity + 9 * fjord + 6 * rugged_control + 6 * float(np.mean(rift[coast_land])) if coast_land.any() else 0))
    carve_count = max(4, min(22, carve_count))
    carve_allowed_zone = _land_ocean_transition_zone(land, radius=max(2, min(14, w // 340)))
    for _ in range(carve_count):
        if candidate.size == 0:
            break
        rr, cc = candidate[rng.randrange(len(candidate))]
        lon = float(lons[int(cc)])
        lat = float(lats[int(rr)])
        local_rift = bool(rift[int(rr), int(cc)])
        local_active = bool(active[int(rr), int(cc)])
        nodes = rng.randint(4, 11 if local_rift else 8)
        heading = rng.uniform(0.0, math.tau)
        base_width = rng.uniform(0.22, 0.85) * (1.0 + 0.9 * local_rift)
        base_len = rng.uniform(0.9, 3.6) * (1.0 + 0.8 * local_rift)
        for _node in range(nodes):
            heading += rng.uniform(-0.72, 0.72)
            lon += math.cos(heading) * rng.uniform(0.9, 4.8) / max(0.28, math.cos(math.radians(lat)))
            lat = clamp(lat + math.sin(heading) * rng.uniform(0.7, 3.8), -74.0, 74.0)
            dlon = _wrapped_lon_delta_array(lon_grid, lon) * cos_lat
            dlat = lat_grid - lat
            along = dlon * math.cos(heading) + dlat * math.sin(heading)
            across = -dlon * math.sin(heading) + dlat * math.cos(heading)
            width_deg = base_width * rng.uniform(0.55, 1.65)
            len_deg = base_len * rng.uniform(0.55, 1.80)
            blob = np.exp(-((along / len_deg) ** 2 + (across / max(0.08, width_deg)) ** 2))
            carve = (blob > rng.uniform(0.45, 0.74)) & land & carve_allowed_zone
            # High mountains resist full drowning; carve only lower slots through them.
            carve &= (elev < (780 + 450 * float(local_active) + 550 * float(local_rift)))
            if int(carve.sum()) <= 0:
                continue
            land[carve] = False
            elev[carve] = np.minimum(elev[carve], -rng.randint(8, 220 if local_rift else 140))

    land = _remove_one_cell_land_water_needles(land)

    # Add irregular shelf islands, rift fragments, and volcanic/arc chains. These
    # intentionally use several linked lobes so islands are less oval.
    ocean = ~land
    island_candidates = np.argwhere((_land_ocean_transition_zone(land, radius=max(8, min(46, w // 120))) & ocean) | (arc & ocean) | (rift & ocean))
    island_chain_count = int(round(4 + 16 * island_density + 8 * coastline_complexity))
    island_chain_count = max(3, min(24, island_chain_count))
    for _ in range(island_chain_count):
        if island_candidates.size == 0:
            break
        rr, cc = island_candidates[rng.randrange(len(island_candidates))]
        lon = float(lons[int(cc)])
        lat = float(lats[int(rr)])
        heading = rng.uniform(0.0, math.tau)
        nodes = rng.randint(4, 9)
        arc_like = bool(arc[int(rr), int(cc)]) or rng.random() < 0.34
        for node in range(nodes):
            heading += rng.uniform(-0.58, 0.58)
            lon += math.cos(heading) * rng.uniform(0.7, 4.8) / max(0.28, math.cos(math.radians(lat)))
            lat = clamp(lat + math.sin(heading) * rng.uniform(0.5, 3.5), -72.0, 72.0)
            dlon = _wrapped_lon_delta_array(lon_grid, lon) * cos_lat
            dlat = lat_grid - lat
            rad_base = rng.uniform(0.16, 0.95) * (1.0 + 0.85 * island_irregularity)
            rad_lon = rad_base * rng.uniform(0.55, 1.85)
            rad_lat = rad_base * rng.uniform(0.45, 1.55)
            blob = np.exp(-((dlon / max(0.07, rad_lon)) ** 2 + (dlat / max(0.07, rad_lat)) ** 2))
            # Add a shoulder lobe to avoid circular dots.
            if rng.random() < 0.72:
                lon2 = lon + rng.uniform(-1.1, 1.1)
                lat2 = clamp(lat + rng.uniform(-0.9, 0.9), -72.0, 72.0)
                dlon2 = _wrapped_lon_delta_array(lon_grid, lon2) * cos_lat
                dlat2 = lat_grid - lat2
                blob += rng.uniform(0.22, 0.52) * np.exp(-((dlon2 / max(0.06, rad_lon * rng.uniform(0.45, 1.05))) ** 2 + (dlat2 / max(0.06, rad_lat * rng.uniform(0.45, 1.05))) ** 2))
            blob = _ragged_island_field(blob, np_rng, strength=0.55 + 0.40 * island_irregularity)
            rr_guess = int(clamp(round((90.0 - lat) / 180.0 * h - 0.5), 0, h - 1))
            cc_guess = int(((lon + 180.0) / 360.0 * w) % w)
            if not ocean[rr_guess, cc_guess]:
                continue
            depth = abs(int(elev[rr_guess, cc_guess]))
            uplift = min(rng.uniform(500.0, 2600.0) + depth * rng.uniform(0.42, 0.92), rng.uniform(900.0, 4400.0))
            island_elev = elev.astype(np.float32) + blob * uplift
            threshold = rng.uniform(0.30, 0.58 if arc_like else 0.68)
            add = ocean & (blob > threshold) & (island_elev > 0.0)
            if int(add.sum()) < max(1, (h * w) // 7_500_000):
                continue
            land[add] = True
            elev[add] = np.maximum(elev[add], np.rint(island_elev[add]).astype(np.int32))
            ocean = ~land

    land = _remove_one_cell_land_water_needles(land)
    land, elev = _remove_low_coastal_filaments(land, elev)
    land, elev = _remove_extreme_ribbon_land_components(land, elev)
    return land, elev

def _fit_ocean_fraction_target(elevation, land_mask, target_ocean_fraction: float, np_rng, *, max_error: float = 0.025):
    """Softly nudge final sea level toward the requested ocean target.

    Ocean fraction is now treated as a preference, not an override that can
    destroy the generated terrain.  Earlier builds used the requested quantile
    as a new sea level even when that required a large datum shift; this could
    create 1m coastal plains, artificial land bridges, drowned interiors, and
    shelf/island discontinuities.  This version caps final sea-level movement
    and reports any remaining miss instead of forcing the target.
    """
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("NumPy is required for terrain sea-level fitting. Install it with: pip install numpy") from exc

    target = clamp(float(target_ocean_fraction), 0.05, 0.95)
    elev = elevation.astype(np.float32, copy=True)
    before = 1.0 - float(np.mean(land_mask))
    before_elev = _enforce_land_ocean_sign_with_lowland_variation(land_mask.astype(bool, copy=True), elev, np_rng)
    if abs(before - target) <= max_error:
        return land_mask.astype(bool, copy=True), before_elev, {
            "requested_ocean_fraction": round(target, 4),
            "before_ocean_fraction": round(before, 4),
            "after_ocean_fraction": round(before, 4),
            "difference_after_minus_target": round(before - target, 4),
            "applied": False,
            "soft_targeting": True,
            "sea_level_shift_m": 0.0,
            "max_allowed_shift_m": 0.0,
            "target_forced": False,
        }

    requested_shift = float(np.quantile(elev, target))
    # The finalizer may make a modest sea-level correction, but the foundation
    # pass owns land/ocean structure.  Keep the cap small enough to avoid moving
    # entire shelves/low continents across sea level in one late step.
    miss = abs(before - target)
    max_shift = 120.0 + 520.0 * min(1.0, miss / 0.22)
    max_shift = clamp(max_shift, 140.0, 650.0)
    sea_level = clamp(requested_shift, -max_shift, max_shift)
    adjusted = elev - sea_level
    candidate_land = adjusted > 0.0

    # Only permit late land/water changes near the existing coastline.  Interior
    # continents and deep oceans should not flip because ocean_fraction_target is
    # edited after the terrain skeleton has already been built.
    transition = _land_ocean_transition_zone(land_mask.astype(bool, copy=False), radius=4)
    keep_original = ~transition
    candidate_land[keep_original] = land_mask[keep_original]
    candidate_land = _irregularize_land_water_edges(candidate_land, adjusted, np_rng, amount=0.045)
    candidate_land[keep_original] = land_mask[keep_original]
    candidate_land = _fill_unintended_inland_water_holes(candidate_land) if target < 0.72 else candidate_land
    candidate_land[keep_original] = land_mask[keep_original]
    candidate_land = _remove_small_land_components(candidate_land, min_size=max(8, int(candidate_land.size * 0.000012)))
    candidate_elev = _enforce_land_ocean_sign_with_lowland_variation(candidate_land, adjusted, np_rng)
    candidate_land, candidate_elev = _repair_artificial_coast_and_inland_water(candidate_land, candidate_elev)
    candidate_land, candidate_elev = _remove_low_coastal_filaments(candidate_land, candidate_elev)
    candidate_land, candidate_elev = _remove_extreme_ribbon_land_components(candidate_land, candidate_elev)
    candidate_elev = _enforce_land_ocean_sign_with_lowland_variation(candidate_land, candidate_elev.astype(np.float32), np_rng)
    after = 1.0 - float(np.mean(candidate_land))

    return candidate_land.astype(bool, copy=False), candidate_elev, {
        "requested_ocean_fraction": round(target, 4),
        "before_ocean_fraction": round(before, 4),
        "after_ocean_fraction": round(after, 4),
        "difference_after_minus_target": round(after - target, 4),
        "applied": True,
        "soft_targeting": True,
        "sea_level_shift_m": round(sea_level, 2),
        "requested_quantile_shift_m": round(requested_shift, 2),
        "max_allowed_shift_m": round(max_shift, 2),
        "target_forced": abs(requested_shift) <= max_shift,
        "note": "Ocean target is a soft preference; remaining error is reported rather than forcing destructive terrain changes.",
    }


def _enforce_land_ocean_sign_with_lowland_variation(land_mask, elevation, np_rng):
    """Keep land above water and ocean below water without creating flat 1m shelves."""
    try:
        import numpy as np
        from scipy import ndimage
    except ImportError as exc:
        raise RuntimeError("NumPy and SciPy are required for terrain finalization. Install with: pip install -r requirements.txt") from exc

    land = land_mask.astype(bool, copy=False)
    elev = elevation.astype(np.float32, copy=True)
    h, w = elev.shape
    noise = _smooth_wrapped_array(np_rng.uniform(0.0, 1.0, size=(h, w)).astype(np.float32), passes=2)
    land_dist = ndimage.distance_transform_edt(land)
    ocean_dist = ndimage.distance_transform_edt(~land)

    low_land = land & (elev < 3.0)
    if low_land.any():
        # Near the coast, keep tidal flats and marshy plains low but varied;
        # inland newly exposed areas rise more quickly so large exact-1m slabs do
        # not appear after ocean-target fitting.
        target = 2.0 + np.clip(land_dist * 2.8, 0.0, 38.0) + noise * 14.0
        elev[low_land] = np.maximum(elev[low_land], target[low_land])

    shallow_ocean = (~land) & (elev > -3.0)
    if shallow_ocean.any():
        depth = 2.0 + np.clip(ocean_dist * 3.2, 0.0, 42.0) + (1.0 - noise) * 16.0
        elev[shallow_ocean] = np.minimum(elev[shallow_ocean], -depth[shallow_ocean])

    elev[land] = np.maximum(elev[land], 1.0)
    elev[~land] = np.minimum(elev[~land], -1.0)
    return np.rint(elev).astype(np.int32)

def _upsample_terrain_with_fine_detail(
    rng: random.Random,
    coarse: TerrainMap,
    width: int,
    height: int,
    geology: GeologyState,
) -> TerrainMap:
    try:
        import numpy as np
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("NumPy and Pillow are required for large-map terrain upscaling. Install with: pip install -r requirements.txt") from exc

    elev_small = np.asarray(coarse.elevation_m, dtype=np.float32)
    elev_img = Image.fromarray(elev_small, mode="F")
    elev_big = np.asarray(elev_img.resize((width, height), Image.Resampling.BICUBIC), dtype=np.float32)

    # Add full-resolution relief so high-res maps are not just enlarged
    # lower-res grids. Use cheap multi-scale fields instead of re-running the
    # whole terrain generator. The medium fields add bays, shelves, blocks, and
    # broad relief; fine fields add local texture.
    np_rng = np.random.default_rng(rng.randrange(1, 2**63 - 1))
    fine = np_rng.uniform(-1.0, 1.0, size=(height, width)).astype(np.float32)
    medium = _smooth_wrapped_array(fine, passes=2)
    fine = _smooth_wrapped_array(fine, passes=1)

    province_h = max(48, min(192, height // 18))
    province_w = max(96, min(384, width // 18))
    province = np_rng.uniform(-1.0, 1.0, size=(province_h, province_w)).astype(np.float32)
    province = _smooth_wrapped_array(province, passes=5)
    province_img = Image.fromarray(province, mode="F")
    province_big = np.asarray(province_img.resize((width, height), Image.Resampling.BICUBIC), dtype=np.float32)
    province_big = _smooth_wrapped_array(province_big, passes=2)

    belt_h = max(32, min(128, height // 28))
    belt_w = max(64, min(256, width // 28))
    belts = np_rng.uniform(-1.0, 1.0, size=(belt_h, belt_w)).astype(np.float32)
    belts = _smooth_wrapped_array(belts, passes=2)
    belts_img = Image.fromarray(belts, mode="F")
    belts_big = np.asarray(belts_img.resize((width, height), Image.Resampling.BICUBIC), dtype=np.float32)
    belts_big = np.clip(belts_big - _smooth_wrapped_array(belts_big, passes=8), -1.0, 1.0)

    # Preserve the coarse structural land/sea decision when upscaling.
    #
    # The earlier high-resolution detail pass used `elev_big > 0` after adding
    # relief/warping. That let eroded river valleys that happened to be near sea
    # level flip into water, creating huge artificial inland shallow-sea networks
    # across continents. Real low river valleys can be close to sea level without
    # becoming ocean unless they are actually connected to the coast. Therefore,
    # only cells close to an existing structural coastline are allowed to change
    # land/sea state during the high-resolution finishing pass.
    structural_land = elev_big > 0.0
    coastal_transition = _land_ocean_transition_zone(structural_land, radius=max(4, min(14, width // 384)))

    # `near_sea_level` is useful for low coastal shelves, but it is not the same
    # thing as a coastline. It must not drive inland valley flooding by itself.
    near_sea_level = np.exp(-((elev_big) / 420.0) ** 2).astype(np.float32)
    shelf = np.exp(-((elev_big + 120.0) / 520.0) ** 2).astype(np.float32)
    coast_weight = near_sea_level * coastal_transition.astype(np.float32)

    macro_relief = (105.0 + 185.0 * geology.surface_roughness) * province_big
    belt_relief = (55.0 + 135.0 * geology.surface_roughness) * belts_big
    relief = (42.0 + 95.0 * geology.surface_roughness) * medium + (10.0 + 22.0 * geology.surface_roughness) * fine
    coast_warp = coast_weight * province_big * (95.0 + 115.0 * geology.surface_roughness) + shelf * coastal_transition.astype(np.float32) * medium * 55.0
    elev_big = elev_big + macro_relief * structural_land.astype(np.float32) + belt_relief * structural_land.astype(np.float32) + relief * (0.35 + 0.65 * structural_land.astype(np.float32)) + coast_warp + coast_weight * fine * 34.0

    candidate_land = elev_big > 0.0
    land_mask = np.where(coastal_transition, candidate_land, structural_land)
    land_mask = _irregularize_land_water_edges(land_mask, elev_big, np_rng, amount=0.20)
    land_mask = _close_narrow_water_channels(land_mask, radius=max(1, min(4, width // 1200)))
    land_mask, elev_big_int = _repair_artificial_coast_and_inland_water(land_mask, np.rint(elev_big).astype(np.int32))
    elev_big = elev_big_int.astype(np.float32)

    # Interior structural land may be very low or eroded, but it should remain
    # land. Clamp it just above sea level so valleys/plains render as lowlands
    # rather than blue inland seas. Coastal transition cells can still form bays,
    # estuaries, islands, and small shoreline changes.
    interior_land = land_mask & structural_land & (~coastal_transition)
    elev_big = np.where(interior_land, np.maximum(elev_big, 8.0), elev_big)

    # Do not run the expensive component cleanup at full upscaled size; the
    # feature grid has already removed tiny islands. This keeps 4096+ terrain
    # runs responsive.
    elevation = np.where(land_mask, np.maximum(elev_big, 1), np.minimum(elev_big, -1)).round().astype(np.int32)
    # Add some small volcanic/island-arc land after upscaling so oceans are not empty.
    lats = np.linspace(90.0 - 90.0 / height, -90.0 + 90.0 / height, height, dtype=np.float32)
    lons = np.linspace(-180.0 + 180.0 / width, 180.0 - 180.0 / width, width, dtype=np.float32)
    cos_lat = np.maximum(0.18, np.cos(np.radians(lats)))[:, None]
    land_mask, elevation = _add_oceanic_islands_and_arcs(rng, land_mask, elevation, lons, lats, cos_lat, sea_threshold=0.0, max_islands=max(26, width // 48))
    land_mask, elevation = _add_inland_sea_islands(rng, land_mask, elevation, lons, lats, cos_lat)
    land_mask, elevation = _repair_artificial_coast_and_inland_water(land_mask, elevation)
    elevation = np.where(land_mask, np.maximum(elevation, 1), np.minimum(elevation, -1)).round().astype(np.int32)

    land_values = elevation[land_mask]
    ocean_values = elevation[~land_mask]
    ocean_fraction = 1.0 - (int(land_mask.sum()) / float(width * height))
    return TerrainMap(
        width=width,
        height=height,
        elevation_m=elevation.tolist(),
        is_land=land_mask.tolist(),
        min_elevation_m=int(elevation.min()),
        max_elevation_m=int(elevation.max()),
        mean_land_elevation_m=float(land_values.mean()) if land_values.size else 0.0,
        mean_ocean_depth_m=float(ocean_values.mean()) if ocean_values.size else 0.0,
        ocean_fraction=ocean_fraction,
        land_fraction=1.0 - ocean_fraction,
        source=f"procedural accelerated feature grid from {coarse.width}x{coarse.height}",
        planet_radius_earth=float(getattr(coarse, "planet_radius_earth", 1.0) or 1.0),
    )




def _roughen_plate_boundaries_u13(plate_id, np_rng, iterations: int = 6):
    """Make synthetic plate cells less smooth while preserving plate identities.

    This is deliberately conservative: only cells in a narrow boundary band can
    switch to an already-adjacent plate ID.  It breaks circular/Voronoi borders
    and also creates more multi-neighbor junctions, which reduces the chance of
    plates being wholly surrounded by one neighbor.
    """
    try:
        import numpy as np
        from scipy import ndimage
    except ImportError as exc:
        raise RuntimeError("NumPy and SciPy are required for plate terrain generation. Install with: pip install -r requirements.txt") from exc
    pid = np.asarray(plate_id, dtype=np.int32).copy()
    h, w = pid.shape
    if h < 4 or w < 4 or np.unique(pid).size <= 2:
        return pid
    noise = np_rng.normal(0.0, 1.0, size=(h, w)).astype(np.float32)
    noise = ndimage.gaussian_filter(noise, sigma=max(0.85, w / 180.0), mode="wrap")
    if float(np.std(noise)) > 1.0e-6:
        noise = (noise - float(np.mean(noise))) / float(np.std(noise))
    for it in range(max(1, int(iterations))):
        north = np.vstack((pid[0:1, :], pid[:-1, :]))
        south = np.vstack((pid[1:, :], pid[-1:, :]))
        west = np.roll(pid, 1, axis=1)
        east = np.roll(pid, -1, axis=1)
        boundary = (pid != north) | (pid != south) | (pid != west) | (pid != east)
        if not np.any(boundary):
            break
        # Select a neighboring plate based on a rotated noise field so boundaries
        # migrate unevenly instead of diffusing into a blur.
        selector = np.roll(noise, it + 1, axis=1) + 0.35 * np.roll(noise, -(it + 2), axis=0)
        candidates = [north, south, west, east]
        chosen = candidates[it % 4].copy()
        chosen = np.where(selector > 0.65, east, chosen)
        chosen = np.where(selector < -0.65, west, chosen)
        chosen = np.where((selector >= -0.15) & (selector <= 0.15), south, chosen)
        flip = boundary & (chosen != pid) & (np.abs(selector) > (0.22 + 0.06 * (it % 3)))
        # Do not allow every boundary cell to change in one iteration.
        checker = ((np.indices(pid.shape)[0] + np.indices(pid.shape)[1] + it) % 3) != 0
        pid[flip & checker] = chosen[flip & checker]
        # Remove single-cell plate needles created by the roughening.
        for neigh in candidates:
            same_count = (
                (north == pid).astype(np.int16)
                + (south == pid).astype(np.int16)
                + (west == pid).astype(np.int16)
                + (east == pid).astype(np.int16)
            )
            isolated = same_count <= 1
            pid[isolated] = neigh[isolated]
            break
    return pid.astype(np.int32)

def _roughen_structural_coastlines(rng: random.Random, land_mask, foundation_field, sea_threshold: float, np_rng, strength: float = 0.30):
    """Perturb the structural land mask along coasts, not via elevation.

    This is the source-level coastline variation pass. It changes the continent
    mask itself near the sea-level boundary using multi-scale noise, bay/gulf
    bites, peninsulas, and island-fragment chains. Because it operates before
    erosion/deposition and before shelf bathymetry, it avoids the old failure
    mode where low valleys or shelf edges became thin ribbon land.
    """
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("NumPy is required for structural coastline generation. Install it with: pip install numpy") from exc
    land = land_mask.astype(bool, copy=True)
    h, w = land.shape
    coast = _land_ocean_transition_zone(land, radius=max(2, min(11, w // 420)))
    if not coast.any():
        return land

    noise1 = _smooth_wrapped_array(np_rng.uniform(-1.0, 1.0, size=(h, w)).astype(np.float32), passes=1)
    noise2 = _smooth_wrapped_array(np_rng.uniform(-1.0, 1.0, size=(h, w)).astype(np.float32), passes=4)
    noise3 = _smooth_wrapped_array(np_rng.uniform(-1.0, 1.0, size=(h, w)).astype(np.float32), passes=10)
    perturb = (0.40 * noise1 + 0.38 * noise2 + 0.22 * noise3) * strength
    candidate = (foundation_field + perturb * coast.astype(np.float32)) >= sea_threshold

    # Only allow flips close to true coasts. This creates bays, capes, barrier
    # margins, and archipelago fragments while preserving continental interiors.
    land = np.where(coast, candidate, land)

    # Add a few chained coastal bites and capes at map resolution. These are
    # larger than pixel noise and give coasts Norway/SE-Asia/Alaska-style medium
    # structure in some provinces while leaving other margins smooth.
    ys, xs = np.where(coast)
    if len(xs) > 0:
        lon_grid = None
        lat_grid = None
        lats = np.linspace(90.0 - 90.0 / h, -90.0 + 90.0 / h, h, dtype=np.float32)
        lons = np.linspace(-180.0 + 180.0 / w, 180.0 - 180.0 / w, w, dtype=np.float32)
        lon_grid, lat_grid = np.meshgrid(lons, lats)
        cos_lat = np.maximum(0.18, np.cos(np.radians(lat_grid)))
        active_coastal_edit_zone = _land_ocean_transition_zone(land, radius=max(4, min(24, w // 240)))
        for _ in range(rng.randint(54, 92)):
            k = rng.randrange(len(xs))
            lon0 = float(lons[xs[k]]); lat0 = float(lats[ys[k]])
            heading = rng.uniform(0.0, math.tau)
            sign = rng.choice([-1, 1])
            for _node in range(rng.randint(6, 16)):
                heading += rng.uniform(-0.8, 0.8)
                lon0 += math.cos(heading) * rng.uniform(2.0, 8.0) / max(0.28, math.cos(math.radians(lat0)))
                lat0 = clamp(lat0 + math.sin(heading) * rng.uniform(1.5, 6.0), -74.0, 74.0)
                dlon = _wrapped_lon_delta_array(lon_grid, lon0) * cos_lat
                dlat = lat_grid - lat0
                blob = np.exp(-((dlon / rng.uniform(0.55, 4.6)) ** 2 + (dlat / rng.uniform(0.45, 3.8)) ** 2))
                local = blob > rng.uniform(0.32, 0.58)
                local &= active_coastal_edit_zone
                if sign > 0:
                    land[local] = True
                else:
                    land[local] = False

    # Avoid coastline-only one-cell bridges without smoothing all coasts.
    return _remove_one_cell_land_water_needles(land)


def _remove_one_cell_land_water_needles(land_mask):
    """Remove needle-like single-cell land/water artifacts while preserving bays."""
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("NumPy is required for land-mask cleanup. Install it with: pip install numpy") from exc
    land = land_mask.astype(bool, copy=True)
    for _ in range(2):
        north = np.vstack((land[0:1, :], land[:-1, :]))
        south = np.vstack((land[1:, :], land[-1:, :]))
        west = np.roll(land, 1, axis=1)
        east = np.roll(land, -1, axis=1)
        nw = np.roll(north, 1, axis=1); ne = np.roll(north, -1, axis=1)
        sw = np.roll(south, 1, axis=1); se = np.roll(south, -1, axis=1)
        land_n = north.astype(np.int16)+south.astype(np.int16)+west.astype(np.int16)+east.astype(np.int16)+nw.astype(np.int16)+ne.astype(np.int16)+sw.astype(np.int16)+se.astype(np.int16)
        ocean_n = 8 - land_n
        land[land & (ocean_n >= 7)] = False
        land[(~land) & (land_n >= 7)] = True
    return land


def _best_longitude_shift_to_ocean_gap(land_mask) -> int:
    """Roll longitudes so an ocean gap sits on the map edge when possible.

    The display problem has two different cases:
    * if a mostly-ocean longitude band exists, put that gap at the left/right
      edge so the main continental hemisphere is visually centered;
    * if no meaningful gap exists, fall back to centering the land-rich half of
      the world.

    Earlier builds used only the land-rich-half heuristic. That centered dense
    land well, but on some worlds it left one continent visibly split across the
    map seam. This hybrid keeps the useful recentering behavior while honoring
    the explicit "push ocean bands to the edge" goal.
    """
    try:
        import numpy as np
    except ImportError:
        return 0

    land = land_mask.astype(bool, copy=False)
    h, w = land.shape
    if w <= 1:
        return 0

    col_land = land.mean(axis=0).astype(np.float64)

    # Exact first rule requested by the user: if any longitude column is pure
    # ocean, place the widest pure-ocean run at the map seam.  This is a pure
    # translation and conserves all raster shapes.
    exact_ocean = col_land <= 0.0
    if bool(exact_ocean.any()):
        doubled_exact = np.r_[exact_ocean, exact_ocean]
        best_start_exact = None
        best_len_exact = 0
        i = 0
        while i < w * 2:
            if not doubled_exact[i]:
                i += 1
                continue
            j = i
            while j < w * 2 and doubled_exact[j]:
                j += 1
            if i < w:
                run_len = min(j - i, w)
                if run_len > best_len_exact:
                    best_start_exact = i
                    best_len_exact = run_len
            i = j
        if best_start_exact is not None:
            gap_center = (best_start_exact + best_len_exact // 2) % w
            return int((-gap_center) % w)

    # Smooth only along longitude; this makes the gap finder tolerant of small
    # islands and ragged coasts inside a broadly oceanic band.
    smooth = col_land.copy()
    for _ in range(max(2, min(10, w // 360))):
        smooth = (smooth * 2.0 + np.roll(smooth, 1) + np.roll(smooth, -1)) / 4.0

    mean_col = float(col_land.mean())
    ocean_gap_threshold = max(0.018, min(0.085, mean_col * 0.30))
    low = smooth <= ocean_gap_threshold
    best_start = None
    best_len = 0
    best_avg = 1.0
    if bool(low.any()):
        doubled = np.r_[low, low]
        values = np.r_[smooth, smooth]
        i = 0
        while i < w * 2:
            if not doubled[i]:
                i += 1
                continue
            j = i
            while j < w * 2 and doubled[j]:
                j += 1
            run_len = j - i
            # Only consider canonical circular runs whose start is inside the
            # first world copy.
            if i < w:
                run_len = min(run_len, w)
                avg = float(values[i:i + run_len].mean()) if run_len > 0 else 1.0
                if run_len > best_len or (run_len == best_len and avg < best_avg):
                    best_start = i
                    best_len = run_len
                    best_avg = avg
            i = j

    # A gap of ~6% of the map or wider is visually meaningful at global scale.
    if best_start is not None and best_len >= max(12, int(w * 0.060)):
        gap_center = (best_start + best_len // 2) % w
        return int((-gap_center) % w)

    # Fallback: center the densest continental hemisphere.
    window = max(16, min(w - 1, int(round(w * 0.52))))
    doubled_land = np.r_[col_land, col_land]
    csum = np.r_[0.0, np.cumsum(doubled_land)]
    scores = csum[window:window + w] - csum[:w]
    best_window_start = int(np.argmax(scores))
    best_center = (best_window_start + window // 2) % w
    target_center = w // 2
    return int((target_center - best_center) % w)

def _remove_low_coastal_filaments(land_mask, elevation):
    """Remove attached low shelf filaments before finalizing terrain.

    This targets the source pattern where a coastal shelf or peninsula-chain
    feature leaves a one/two-cell-wide low strip attached to a continent. It is
    local and conservative: rugged/high relief peninsulas survive, but low
    shelf-edge ribbons are returned to shallow water.
    """
    try:
        import numpy as np
    except ImportError:
        return land_mask, elevation
    land = land_mask.astype(bool, copy=True)
    elev = elevation.astype(np.int32, copy=True)
    for pass_no in range(3):
        north = np.vstack((land[0:1, :], land[:-1, :]))
        south = np.vstack((land[1:, :], land[-1:, :]))
        west = np.roll(land, 1, axis=1)
        east = np.roll(land, -1, axis=1)
        nw = np.roll(north, 1, axis=1); ne = np.roll(north, -1, axis=1)
        sw = np.roll(south, 1, axis=1); se = np.roll(south, -1, axis=1)
        land_neighbors = (north.astype(np.int16)+south.astype(np.int16)+west.astype(np.int16)+east.astype(np.int16)+nw.astype(np.int16)+ne.astype(np.int16)+sw.astype(np.int16)+se.astype(np.int16))
        cardinal_land = north.astype(np.int16)+south.astype(np.int16)+west.astype(np.int16)+east.astype(np.int16)
        elev_n = np.vstack((elev[0:1, :], elev[:-1, :]))
        elev_s = np.vstack((elev[1:, :], elev[-1:, :]))
        elev_w = np.roll(elev, 1, axis=1)
        elev_e = np.roll(elev, -1, axis=1)
        rugged_neighbor = (elev_n > 260) | (elev_s > 260) | (elev_w > 260) | (elev_e > 260)
        filament = land & (elev < (145 - pass_no * 18)) & (land_neighbors <= 5) & (cardinal_land <= 3) & (~rugged_neighbor)
        if not filament.any():
            break
        land[filament] = False
        elev[filament] = np.minimum(elev[filament], -10)
    return land, elev

def _remove_extreme_ribbon_land_components(land_mask, elevation):
    """Remove source-generated long, thin shelf/peninsula artifacts.

    This is intentionally not a shoreline smoothing pass. It only removes land
    components whose bounding box is extremely long and thin and whose elevation
    is low, which are almost always accidental shelf-edge ribbons rather than
    legitimate peninsulas, mountain chains, or island arcs.
    """
    try:
        import numpy as np
        from scipy import ndimage
    except ImportError:
        return land_mask, elevation
    land = land_mask.astype(bool, copy=True)
    elev = elevation.astype(np.int32, copy=True)
    labels, count = ndimage.label(land)
    if count <= 0:
        return land, elev
    objects = ndimage.find_objects(labels)
    world_area = land.shape[0] * land.shape[1]
    for label_id, slc in enumerate(objects, start=1):
        if slc is None:
            continue
        rs, cs = slc
        hh = rs.stop - rs.start
        ww = cs.stop - cs.start
        if hh <= 0 or ww <= 0:
            continue
        component = labels[slc] == label_id
        area = int(component.sum())
        if area < max(20, int(world_area * 0.00001)):
            continue
        aspect = max(hh / max(1, ww), ww / max(1, hh))
        fill_ratio = area / float(hh * ww)
        comp_elev = elev[slc][component]
        mean_elev = float(comp_elev.mean()) if comp_elev.size else 0.0
        max_elev = int(comp_elev.max()) if comp_elev.size else 0
        if aspect >= 9.0 and fill_ratio <= 0.26 and mean_elev < 95.0 and max_elev < 420:
            mask = labels == label_id
            land[mask] = False
            elev[mask] = np.minimum(elev[mask], -12)
    return land, elev


def _carve_irregular_seaways_and_marginal_seas(rng: random.Random, land_mask, lons, lats, cos_lat, np_rng):
    """Carve source-level seaways, gulfs, and semi-enclosed seas.

    This is deliberately not post-processing. It changes the structural land
    mask before elevation and erosion are calculated, so partial inland seas and
    embayments are real coastline features rather than flooded erosion scars.
    Chains start near existing coasts and wander inland/along-margin with
    irregular widths, which avoids the old rounded-rectangle lake artifact.
    """
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("NumPy is required for seaway generation. Install it with: pip install numpy") from exc

    land = land_mask.astype(bool, copy=True)
    h, w = land.shape
    coast = _land_ocean_transition_zone(land, radius=max(5, min(30, w // 150)))
    coast_cells = np.argwhere(coast & land)
    if coast_cells.size == 0:
        return land

    lon_grid, lat_grid = np.meshgrid(lons, lats)
    carve = np.zeros_like(land, dtype=bool)
    chain_count = rng.randint(5, 10)
    carve_zone = _land_ocean_transition_zone(land, radius=max(8, min(46, w // 110)))
    carve_noise = _smooth_wrapped_array(np_rng.uniform(-1.0, 1.0, size=(h, w)).astype(np.float32), passes=1)
    for _ in range(chain_count):
        rr, cc = coast_cells[rng.randrange(len(coast_cells))]
        lon = float(lons[int(cc)])
        lat = float(lats[int(rr)])
        heading = rng.uniform(0.0, math.tau)
        nodes = rng.randint(5, 14)
        base_width = rng.choice([rng.uniform(0.45, 1.4), rng.uniform(1.2, 3.6)])
        base_len = rng.uniform(1.4, 5.8)
        for _node in range(nodes):
            heading += rng.uniform(-0.72, 0.72)
            step = rng.uniform(1.4, 6.8)
            lon += math.cos(heading) * step / max(0.28, math.cos(math.radians(lat)))
            lat = clamp(lat + math.sin(heading) * step, -72.0, 72.0)
            dlon = _wrapped_lon_delta_array(lon_grid, lon) * cos_lat
            dlat = lat_grid - lat
            along = dlon * math.cos(heading) + dlat * math.sin(heading)
            across = -dlon * math.sin(heading) + dlat * math.cos(heading)
            width_deg = base_width * rng.uniform(0.55, 1.75)
            len_deg = base_len * rng.uniform(0.55, 1.65)
            blob = np.exp(-((along / len_deg) ** 2 + (across / width_deg) ** 2))
            local = (blob + 0.11 * carve_noise) > rng.uniform(0.58, 0.86)
            carve |= local & land & carve_zone

    if carve.any():
        land[carve] = False
        land = _remove_one_cell_land_water_needles(land)
    return land


def _choose_cross_landmass_coast_points(rng: random.Random, coast_cells, width: int, height: int):
    """Pick two coastline cells that are likely to cut across a landmass.

    A farthest-pair choice often follows the long axis of an elongated continent,
    creating a long fjord/inlet without actually splitting the landmass. This
    helper first tries to connect opposite sides across the short axis. If the
    component is compact or the sampled sides are sparse, it falls back to the
    far-apart coast picker.
    """
    if len(coast_cells) < 2:
        return None, None
    sample_count = min(520, len(coast_cells))
    sampled = [tuple(int(v) for v in coast_cells[rng.randrange(len(coast_cells))]) for _ in range(sample_count)]
    rows = [p[0] for p in sampled]
    cols = [p[1] for p in sampled]
    row_min, row_max = min(rows), max(rows)
    col_min, col_max = min(cols), max(cols)
    row_span = max(1, row_max - row_min)
    col_span = max(1, col_max - col_min)

    best_pair = None
    best_score = -1.0
    if col_span >= row_span * 1.18:
        # Wide continent: cut north-south, preferably near the middle longitude
        # rather than along the long coastline.
        north_cut = row_min + row_span * 0.28
        south_cut = row_max - row_span * 0.28
        north = [p for p in sampled if p[0] <= north_cut]
        south = [p for p in sampled if p[0] >= south_cut]
        for a in north[:180]:
            for b in south[::max(1, len(south) // 80)]:
                dx_raw = abs(a[1] - b[1])
                dx = min(dx_raw, width - dx_raw) / max(1, width)
                dy = abs(a[0] - b[0]) / max(1, height)
                # Strongly prefer a real north-south crossing, but allow some
                # diagonal drift for natural seaways.
                score = dy * 2.2 - dx * 0.52 + rng.uniform(-0.025, 0.025)
                if score > best_score:
                    best_score = score
                    best_pair = (a, b)
    elif row_span >= col_span * 1.18:
        # Tall continent: cut west-east.
        west_cut = col_min + col_span * 0.28
        east_cut = col_max - col_span * 0.28
        west = [p for p in sampled if p[1] <= west_cut]
        east = [p for p in sampled if p[1] >= east_cut]
        for a in west[:180]:
            for b in east[::max(1, len(east) // 80)]:
                dx_raw = abs(a[1] - b[1])
                dx = min(dx_raw, width - dx_raw) / max(1, width)
                dy = abs(a[0] - b[0]) / max(1, height)
                score = dx * 2.2 - dy * 0.52 + rng.uniform(-0.025, 0.025)
                if score > best_score:
                    best_score = score
                    best_pair = (a, b)

    if best_pair is not None and best_score >= 0.20:
        return best_pair
    return _choose_far_apart_coast_points(rng, coast_cells, width, height)


def _choose_far_apart_coast_points(rng: random.Random, coast_cells, width: int, height: int):
    """Pick two coast cells likely to cut across, not nibble along, a landmass."""
    if len(coast_cells) < 2:
        return None, None
    sample_count = min(320, len(coast_cells))
    # Convert a random sample to Python tuples so repeated scoring is cheap.
    sampled = [tuple(int(v) for v in coast_cells[rng.randrange(len(coast_cells))]) for _ in range(sample_count)]
    best_pair = None
    best_score = -1.0
    for _ in range(64):
        a = sampled[rng.randrange(sample_count)]
        # Score against a subset to avoid O(n^2) on huge coastlines.
        for b in sampled[::max(1, sample_count // 48)]:
            dy = abs(a[0] - b[0]) / max(1, height)
            dx_raw = abs(a[1] - b[1])
            dx = min(dx_raw, width - dx_raw) / max(1, width)
            if dx < 0.035 and dy < 0.035:
                continue
            # Favor opposite or widely separated coastline sections. A little
            # randomness prevents all splits from selecting the same axis.
            score = math.hypot(dx * 1.8, dy * 1.25) + rng.uniform(-0.025, 0.025)
            if score > best_score:
                best_score = score
                best_pair = (a, b)
    if best_pair is None or best_score < 0.12:
        return None, None
    return best_pair



def _add_structural_archipelago_chains(rng: random.Random, land_mask, lons, lats, cos_lat, np_rng):
    """Add source-level islands and arc fragments before elevation scaling."""
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("NumPy is required for structural island generation. Install it with: pip install numpy") from exc

    land = land_mask.astype(bool, copy=True)
    h, w = land.shape
    ocean = ~land
    if not ocean.any():
        return land
    lon_grid, lat_grid = np.meshgrid(lons, lats)

    arc_count = rng.randint(10, 18)
    coast_ocean = _land_ocean_transition_zone(land, radius=max(10, min(60, w // 92))) & ocean
    candidates = np.argwhere(coast_ocean)
    if candidates.size == 0:
        candidates = np.argwhere(ocean)
    near_land = _land_ocean_transition_zone(land, radius=max(2, min(8, w // 650)))
    for _ in range(arc_count):
        rr, cc = candidates[rng.randrange(len(candidates))]
        lon = float(lons[int(cc)])
        lat = float(lats[int(rr)])
        heading = rng.uniform(0.0, math.tau)
        nodes = rng.randint(5, 16)
        for _node in range(nodes):
            heading += rng.uniform(-0.62, 0.62)
            lon += math.cos(heading) * rng.uniform(1.5, 7.5) / max(0.28, math.cos(math.radians(lat)))
            lat = clamp(lat + math.sin(heading) * rng.uniform(1.0, 5.5), -72.0, 72.0)
            dlon = _wrapped_lon_delta_array(lon_grid, lon) * cos_lat
            dlat = lat_grid - lat
            rad_lon = rng.uniform(0.35, 2.25)
            rad_lat = rng.uniform(0.28, 1.85)
            blob = np.exp(-((dlon / rad_lon) ** 2 + (dlat / rad_lat) ** 2))
            island = blob > rng.uniform(0.50, 0.78)
            # Most structural arc islands should sit just offshore, but allow a
            # few barrier/arc fragments close to continents so SE-Asia/Aegean-
            # style margins are possible.
            allow_near_margin = rng.random() < 0.38
            add = island & ocean & ((~near_land) | allow_near_margin)
            if int(add.sum()) >= max(3, (h * w) // 1_100_000):
                land[add] = True
                ocean = ~land
                near_land = _land_ocean_transition_zone(land, radius=max(2, min(8, w // 650)))
    return _remove_one_cell_land_water_needles(land)

def _fill_unintended_inland_water_holes(land_mask):
    """Fill accidental holes in the structural continent mask.

    Inland lakes/seas should come from deliberate rift/endorheic basin logic,
    not from the coarse land-mask threshold. Filling holes here prevents the
    rounded-rectangle inland seas seen when blocky low-resolution fields are
    smoothed and thresholded.
    """
    try:
        import numpy as np
        from scipy import ndimage
    except ImportError:
        return land_mask

    land = land_mask.astype(bool, copy=True)
    # binary_fill_holes treats edge-connected water as ocean and fills enclosed
    # inland water. That is what we want at the structural-mask stage.
    filled = ndimage.binary_fill_holes(land)
    return filled.astype(bool, copy=False)


def _add_intentional_inland_water_bodies(
    rng: random.Random,
    land_mask,
    lons,
    lats,
    cos_lat,
    np_rng,
):
    """Create deliberate irregular inland lakes/seas and rift lakes.

    These are generated from wandering chains of overlapping basins rather than
    rectangular coarse-grid holes. They are kept away from the immediate coast,
    are sparse, and are large enough to survive later cleanup.
    """
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("NumPy is required for inland water generation. Install it with: pip install numpy") from exc

    land = land_mask.astype(bool, copy=True)
    h, w = land.shape
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    coast_zone = _land_ocean_transition_zone(land, radius=max(8, min(46, w // 130)))
    interior = land & (~coast_zone) & (np.abs(lat_grid) < 72.0)
    water = np.zeros_like(land, dtype=bool)

    if not bool(interior.any()):
        return land, water

    # A few rift lakes / inland seas. Use attempts rather than a giant argwhere
    # candidate array to keep memory stable on 4K+ maps.
    body_count = rng.randint(2, 6)
    for _ in range(body_count):
        center_found = False
        for _attempt in range(300):
            rr = rng.randrange(h)
            cc = rng.randrange(w)
            if interior[rr, cc]:
                lon0 = float(lons[cc])
                lat0 = float(lats[rr])
                center_found = True
                break
        if not center_found:
            continue

        kind = rng.choice(("rift_lake", "endorheic_sea", "chain_lakes"))
        heading = rng.uniform(0.0, math.tau)
        nodes = rng.randint(3, 7) if kind != "endorheic_sea" else rng.randint(2, 4)
        base_len = rng.uniform(1.0, 3.8) if kind != "endorheic_sea" else rng.uniform(2.5, 7.5)
        base_width = rng.uniform(0.35, 1.25) if kind != "endorheic_sea" else rng.uniform(1.0, 3.2)
        body_field = np.zeros((h, w), dtype=np.float32)
        lon_n = lon0
        lat_n = lat0
        for node in range(nodes):
            heading += rng.uniform(-0.48, 0.48)
            if node > 0:
                step = rng.uniform(1.2, 4.8) if kind != "endorheic_sea" else rng.uniform(1.0, 3.2)
                lon_n += math.cos(heading) * step / max(0.28, math.cos(math.radians(lat_n)))
                lat_n = clamp(lat_n + math.sin(heading) * step, -70.0, 70.0)
            dlon = _wrapped_lon_delta_array(lon_grid, lon_n) * cos_lat
            dlat = lat_grid - lat_n
            along = dlon * math.cos(heading) + dlat * math.sin(heading)
            across = -dlon * math.sin(heading) + dlat * math.cos(heading)
            local_len = base_len * rng.uniform(0.75, 1.55)
            local_width = base_width * rng.uniform(0.65, 1.45)
            body_field += np.exp(-((along / local_len) ** 2 + (across / local_width) ** 2)) * rng.uniform(0.75, 1.25)

        # Irregular edge perturbation. This makes lakes look drowned/rifted, not
        # grid-block or rounded-rectangle generated.
        noise = _smooth_wrapped_array(np_rng.uniform(-1.0, 1.0, size=(h, w)).astype(np.float32), passes=2)
        threshold = rng.uniform(0.70, 1.05)
        lake = (body_field + 0.16 * noise) > threshold
        lake &= interior
        # Keep very tiny water bodies out of the structural terrain map.
        if int(lake.sum()) < max(16, int(h * w * 0.000004)):
            continue
        # Avoid one-pixel ribbons; only keep cells with local water support.
        lake = _stabilize_water_body_shape(lake)
        water |= lake
        land[lake] = False

    return land, water


def _stabilize_water_body_shape(water_mask):
    """Round off isolated water pixels without making blocky lakes."""
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("NumPy is required for lake shaping. Install it with: pip install numpy") from exc
    water = water_mask.astype(bool, copy=True)
    for _ in range(2):
        north = np.vstack((water[0:1, :], water[:-1, :]))
        south = np.vstack((water[1:, :], water[-1:, :]))
        west = np.roll(water, 1, axis=1)
        east = np.roll(water, -1, axis=1)
        nw = np.roll(north, 1, axis=1)
        ne = np.roll(north, -1, axis=1)
        sw = np.roll(south, 1, axis=1)
        se = np.roll(south, -1, axis=1)
        count = (north.astype(np.int16) + south.astype(np.int16) + west.astype(np.int16) + east.astype(np.int16) +
                 nw.astype(np.int16) + ne.astype(np.int16) + sw.astype(np.int16) + se.astype(np.int16))
        water = (water & (count >= 2)) | ((~water) & (count >= 6))
    return water


def _lowpass_field_for_landmask(field):
    """Return a very low-pass continent mask field for solid landmasses.

    The terrain height field can have rifts, valleys, basins, and detail. The
    land/ocean mask should come from a much broader continent-scale field;
    otherwise thresholding turns every low valley into a shallow sea.
    """
    try:
        import numpy as np
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("NumPy and Pillow are required for land-mask low-pass filtering. Install with: pip install -r requirements.txt") from exc

    h, w = field.shape
    # Keep enough resolution in the structural land mask to preserve
    # continent-scale peninsulas, gulfs, and archipelagos. Earlier values were
    # intentionally very low-pass, but that made continents look like smooth
    # amoebas and also created shelf-edge ribbon artefacts after upscaling.
    low_w = max(128, min(640, w // 4))
    low_h = max(64, min(320, h // 4))
    img = Image.fromarray(field.astype(np.float32, copy=False), mode="F")
    small = np.asarray(img.resize((low_w, low_h), Image.Resampling.BICUBIC), dtype=np.float32)
    small = _smooth_wrapped_array(small, passes=2)
    big = np.asarray(Image.fromarray(small, mode="F").resize((w, h), Image.Resampling.BICUBIC), dtype=np.float32)
    return _smooth_wrapped_array(big, passes=3)


def _close_narrow_water_channels(land_mask, radius: int = 2):
    """Fill narrow accidental seaways while preserving broad oceans/seas.

    Procedural continent masks can develop dendritic, river-like blue water
    channels when lowland/rift features cross the sea-level threshold. A small
    morphological close removes these thin artefacts but leaves broad bays and
    major inland seas intact.
    """
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("NumPy is required for land-mask cleanup. Install it with: pip install numpy") from exc

    def dilate(mask):
        north = np.vstack((mask[0:1, :], mask[:-1, :]))
        south = np.vstack((mask[1:, :], mask[-1:, :]))
        west = np.roll(mask, 1, axis=1)
        east = np.roll(mask, -1, axis=1)
        nw = np.roll(north, 1, axis=1)
        ne = np.roll(north, -1, axis=1)
        sw = np.roll(south, 1, axis=1)
        se = np.roll(south, -1, axis=1)
        return mask | north | south | west | east | nw | ne | sw | se

    def erode(mask):
        north = np.vstack((mask[0:1, :], mask[:-1, :]))
        south = np.vstack((mask[1:, :], mask[-1:, :]))
        west = np.roll(mask, 1, axis=1)
        east = np.roll(mask, -1, axis=1)
        nw = np.roll(north, 1, axis=1)
        ne = np.roll(north, -1, axis=1)
        sw = np.roll(south, 1, axis=1)
        se = np.roll(south, -1, axis=1)
        return mask & north & south & west & east & nw & ne & sw & se

    cleaned = land_mask.astype(bool, copy=True)
    for _ in range(max(0, int(radius))):
        cleaned = dilate(cleaned)
    for _ in range(max(0, int(radius))):
        cleaned = erode(cleaned)
    return cleaned


def _land_ocean_transition_zone(land_mask, radius: int = 6):
    """Return cells close to the real structural coastline.

    This is deliberately based on land/ocean adjacency, not elevation. A deep
    inland valley can be near sea level, but it should not be allowed to become
    an ocean cell unless it is actually close to the existing coast.
    """
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("NumPy is required for coastline transition detection. Install it with: pip install numpy") from exc

    land = land_mask.astype(bool, copy=False)
    h, w = land.shape
    north = np.vstack((land[0:1, :], land[:-1, :]))
    south = np.vstack((land[1:, :], land[-1:, :]))
    west = np.roll(land, 1, axis=1)
    east = np.roll(land, -1, axis=1)
    boundary = (land != north) | (land != south) | (land != west) | (land != east)
    zone = boundary.copy()
    current = boundary.copy()
    for _ in range(max(0, int(radius))):
        north = np.vstack((current[0:1, :], current[:-1, :]))
        south = np.vstack((current[1:, :], current[-1:, :]))
        west = np.roll(current, 1, axis=1)
        east = np.roll(current, -1, axis=1)
        current = current | north | south | west | east
        zone |= current
    return zone


def _wrapped_lon_delta(lon_a: float, lon_b: float) -> float:
    delta = (lon_a - lon_b + 180.0) % 360.0 - 180.0
    return delta


def _wrapped_lon_delta_array(lon_grid, lon_b: float):
    return (lon_grid - lon_b + 180.0) % 360.0 - 180.0


def _smooth_wrapped_array(array, passes: int = 1):
    """Fast wrapped 8-neighbor smoothing.

    Earlier versions used fancy indexing lists each pass, which became expensive
    at 4096x2048+. np.roll keeps the same wrapped-column behavior while avoiding
    repeated Python list/index construction.
    """
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("NumPy is required for smoothing. Install it with: pip install numpy") from exc

    dtype = array.dtype
    current = array.astype(np.float32, copy=False)
    for _ in range(max(0, passes)):
        north = np.vstack((current[0:1, :], current[:-1, :]))
        south = np.vstack((current[1:, :], current[-1:, :]))
        west = np.roll(current, 1, axis=1)
        east = np.roll(current, -1, axis=1)
        nw = np.roll(north, 1, axis=1)
        ne = np.roll(north, -1, axis=1)
        sw = np.roll(south, 1, axis=1)
        se = np.roll(south, -1, axis=1)
        current = (current * 4.0 + 1.5 * (north + south + west + east) + 0.7 * (nw + ne + sw + se)) / 13.8
    return current.astype(dtype, copy=False)


def _diagnostic_float_x1000(field, *, diag_w: int = 512, diag_h: int = 256):
    """Downsample a float influence field to a compact 0..1000 diagnostic raster."""
    try:
        import numpy as np
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("NumPy and Pillow are required for terrain diagnostics. Install with: pip install -r requirements.txt") from exc
    arr = np.asarray(field, dtype=np.float32)
    if arr.size == 0:
        return None
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    arr = arr - min(0.0, float(arr.min()))
    max_v = float(arr.max())
    if max_v > 1.0e-6:
        arr = arr / max_v
    arr = np.clip(arr, 0.0, 1.0)
    resized = np.asarray(Image.fromarray(arr.astype(np.float32), mode="F").resize((diag_w, diag_h), Image.Resampling.BICUBIC), dtype=np.float32)
    return np.rint(np.clip(resized, 0.0, 1.0) * 1000.0).astype(int).tolist()






def _diagnostic_class_raster(field, *, diag_w: int = 512, diag_h: int = 256):
    """Downsample an integer class raster with nearest-neighbor semantics."""
    try:
        import numpy as np
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("NumPy and Pillow are required for terrain diagnostics. Install with: pip install -r requirements.txt") from exc
    arr = np.asarray(field, dtype=np.int32)
    if arr.size == 0:
        return None
    resized = np.asarray(Image.fromarray(arr, mode="I").resize((diag_w, diag_h), Image.Resampling.NEAREST), dtype=np.int32)
    return resized.astype(int).tolist()


# Backwards-compatible short aliases used by older terrain-update snippets.
diagf = _diagnostic_float_x1000
diagc = _diagnostic_class_raster


def _apply_ocean_floor_tectonics(elevation, land_mask, boundary_class_full, province_type_full, lons, lats, cos_lat, rng: random.Random, np_rng, geology: GeologyState, terrain_controls: dict | None = None):
    """Shape ocean bathymetry with ridges, trenches, fracture zones, and seamounts.

    This is a procedural ocean-floor pass, not a full plate reconstruction.  It
    gives the bathymetry recognizable large-scale tectonic structure so oceans
    do not remain generic basins with only shelves: divergent ocean provinces
    become mid-ocean ridges, convergent active margins become trenches, transform
    boundaries become fracture-zone lineaments, and volcanic/hotspot provinces
    produce seamount chains.
    """
    try:
        import numpy as np
        from scipy import ndimage
    except ImportError as exc:
        raise RuntimeError("NumPy and SciPy are required for ocean-floor terrain generation. Install with: pip install -r requirements.txt") from exc

    controls = terrain_controls or {}
    elev = elevation.astype(np.float32, copy=True)
    land = land_mask.astype(bool, copy=False)
    ocean = ~land
    h, w = elev.shape
    if not ocean.any():
        zeros = np.zeros((h, w), dtype=np.float32)
        return elev.astype(np.int32), {
            "ocean_floor_class_full": np.zeros((h, w), dtype=np.uint8),
            "ridge_field_full": zeros,
            "trench_field_full": zeros,
            "fracture_field_full": zeros,
            "seamount_field_full": zeros,
            "meta": {"note": "No ocean cells present."},
        }

    b = np.asarray(boundary_class_full, dtype=np.uint8) if boundary_class_full is not None and getattr(boundary_class_full, "shape", None) == elev.shape else np.zeros((h, w), dtype=np.uint8)
    pt = np.asarray(province_type_full, dtype=np.uint8) if province_type_full is not None and getattr(province_type_full, "shape", None) == elev.shape else np.zeros((h, w), dtype=np.uint8)

    ridge_seed = ocean & ((b == 2) | (pt == 1))
    fracture_seed = ocean & (b == 3)
    active_margin = np.isin(b, [1, 6])
    continent_support = _continent_scale_land_mask(land, min_fraction=0.0025)
    if not continent_support.any():
        continent_support = land
    dist_to_continent = ndimage.distance_transform_edt(~continent_support)
    dist_to_land = ndimage.distance_transform_edt(~land)
    near_continent = ocean & (dist_to_continent < max(4, min(56, w // 58)))
    trench_seed = ocean & near_continent & active_margin

    # If the pseudo-plate skeleton did not create enough oceanic ridges/trenches,
    # draw a few sinuous global-scale ridges in deep ocean. They are broad enough
    # to be visible but remain below sea level.
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    synthetic_ridges = np.zeros((h, w), dtype=np.float32)
    ridge_count = int(round(2 + 5 * clamp(float(controls.get("rift_strength", 0.45) or 0.45), 0.0, 1.0) + 2 * clamp(float(getattr(geology, "internal_heat", 0.75) or 0.75), 0.0, 2.0) / 2.0))
    for _ in range(max(2, min(9, ridge_count))):
        for _attempt in range(60):
            rr = rng.randrange(h)
            cc = rng.randrange(w)
            if ocean[rr, cc] and dist_to_land[rr, cc] > max(2, min(30, w // 140)) and abs(float(lats[rr])) < 76.0:
                break
        else:
            continue
        lon = float(lons[cc]); lat = float(lats[rr])
        heading = rng.uniform(0.0, 2.0 * math.pi)
        nodes = rng.randint(9, 20)
        for _node in range(nodes):
            heading += rng.uniform(-0.42, 0.42)
            lon += math.cos(heading) * rng.uniform(5.0, 16.0) / max(0.25, math.cos(math.radians(lat)))
            lat = clamp(lat + math.sin(heading) * rng.uniform(3.0, 10.0), -76.0, 76.0)
            dlon = _wrapped_lon_delta_array(lon_grid, lon) * cos_lat
            dlat = lat_grid - lat
            width = rng.uniform(0.55, 1.75) * (0.70 + 0.60 * w / 2048.0)
            synthetic_ridges += np.exp(-((dlon / max(0.12, width)) ** 2 + (dlat / max(0.12, width * rng.uniform(0.75, 1.35))) ** 2)) * rng.uniform(0.45, 1.0)

    ridge_field = _smooth_wrapped_array(ridge_seed.astype(np.float32), passes=max(2, min(7, w // 420)))
    ridge_field = np.maximum(ridge_field, np.clip(synthetic_ridges, 0.0, 1.0)) * ocean.astype(np.float32)
    if float(ridge_field.max()) > 0:
        ridge_field = np.clip(ridge_field / max(1.0e-6, float(np.percentile(ridge_field[ridge_field > 0], 96))), 0.0, 1.0)

    trench_field = _smooth_wrapped_array(trench_seed.astype(np.float32), passes=max(1, min(5, w // 520))) * ocean.astype(np.float32)
    if float(trench_field.max()) > 0:
        trench_field = np.clip(trench_field / max(1.0e-6, float(np.percentile(trench_field[trench_field > 0], 94))), 0.0, 1.0)
    fracture_field = _smooth_wrapped_array(fracture_seed.astype(np.float32), passes=max(1, min(4, w // 650))) * ocean.astype(np.float32)
    if float(fracture_field.max()) > 0:
        fracture_field = np.clip(fracture_field / max(1.0e-6, float(np.percentile(fracture_field[fracture_field > 0], 94))), 0.0, 1.0)

    # Oceanic crust age proxy: abyssal plains deepen away from ridges. This is
    # what makes ridge-centered basins and broad deep plains visible.
    if (ridge_field > 0.18).any():
        dist_to_ridge = ndimage.distance_transform_edt(~(ridge_field > 0.18))
        age = np.clip(dist_to_ridge / max(8.0, min(96.0, w / 18.0)), 0.0, 1.0) * ocean.astype(np.float32)
    else:
        age = np.clip(dist_to_land / max(8.0, min(96.0, w / 16.0)), 0.0, 1.0) * ocean.astype(np.float32)

    seamount_field = np.zeros((h, w), dtype=np.float32)
    chain_count = max(3, min(14, int(round(3 + 8 * clamp(float(getattr(geology, "volcanism", 0.55) or 0.55), 0.0, 1.8) / 1.8))))
    for _ in range(chain_count):
        for _attempt in range(50):
            rr = rng.randrange(h); cc = rng.randrange(w)
            if ocean[rr, cc] and elev[rr, cc] < -500 and abs(float(lats[rr])) < 76.0:
                break
        else:
            continue
        lon = float(lons[cc]); lat = float(lats[rr]); heading = rng.uniform(0, 2 * math.pi)
        for node in range(rng.randint(4, 10)):
            heading += rng.uniform(-0.35, 0.35)
            lon += math.cos(heading) * rng.uniform(1.8, 7.5) / max(0.28, math.cos(math.radians(lat)))
            lat = clamp(lat + math.sin(heading) * rng.uniform(1.2, 5.5), -76.0, 76.0)
            dlon = _wrapped_lon_delta_array(lon_grid, lon) * cos_lat
            dlat = lat_grid - lat
            radius = rng.uniform(0.22, 0.95)
            seamount_field += np.exp(-((dlon / radius) ** 2 + (dlat / (radius * rng.uniform(0.75, 1.35))) ** 2)) * rng.uniform(0.35, 1.0)
    if float(seamount_field.max()) > 0:
        seamount_field = np.clip(seamount_field / max(1.0e-6, float(np.percentile(seamount_field[seamount_field > 0], 97))), 0.0, 1.0) * ocean.astype(np.float32)

    # Apply bathymetric shaping. Ridges and seamounts raise the floor but mostly
    # remain submerged; trenches carve narrow deep troughs; abyssal age deepens
    # plains away from ridges. Preserve the shelf cap from the previous pass by
    # limiting changes very near continent-scale passive margins.
    abyssal_deepen = (260.0 + 780.0 * age) * ocean.astype(np.float32)
    ridge_uplift = (650.0 + 1500.0 * ridge_field) * ridge_field
    fracture_offset = 180.0 * fracture_field * (0.45 + 0.55 * age)
    trench_cut = (900.0 + 2200.0 * trench_field) * trench_field
    seamount_uplift = (360.0 + 1600.0 * seamount_field) * seamount_field
    new_ocean = elev - abyssal_deepen + ridge_uplift + seamount_uplift - trench_cut - fracture_offset
    # Keep ocean below sea level unless an explicit later island pass lifts it.
    new_ocean = np.minimum(new_ocean, -35.0)
    elev[ocean] = new_ocean[ocean]

    floor_class = np.zeros((h, w), dtype=np.uint8)
    floor_class[ocean] = 1
    floor_class[ocean & (ridge_field > 0.32)] = 2
    floor_class[ocean & (trench_field > 0.25)] = 3
    floor_class[ocean & (fracture_field > 0.35)] = 4
    floor_class[ocean & (seamount_field > 0.38) & ~(trench_field > 0.25)] = 5
    meta = {
        "mean_ridge_strength": round(float(np.mean(ridge_field[ocean])) if ocean.any() else 0.0, 4),
        "mean_trench_strength": round(float(np.mean(trench_field[ocean])) if ocean.any() else 0.0, 4),
        "ridge_cell_share_of_ocean": round(float(np.mean((ridge_field > 0.32)[ocean])) if ocean.any() else 0.0, 4),
        "trench_cell_share_of_ocean": round(float(np.mean((trench_field > 0.25)[ocean])) if ocean.any() else 0.0, 4),
        "seamount_cell_share_of_ocean": round(float(np.mean((seamount_field > 0.38)[ocean])) if ocean.any() else 0.0, 4),
        "note": "Procedural ocean-floor tectonic shaping; ridges/trenches are generated from the Stage 3 boundary/province skeleton plus fallback sinuous ridge chains.",
    }
    return np.rint(elev).astype(np.int32), {
        "ocean_floor_class_full": floor_class,
        "ridge_field_full": ridge_field.astype(np.float32),
        "trench_field_full": trench_field.astype(np.float32),
        "fracture_field_full": fracture_field.astype(np.float32),
        "seamount_field_full": seamount_field.astype(np.float32),
        "meta": meta,
    }


def _diagnostic_signed_delta_m(field, *, diag_w: int = 512, diag_h: int = 256, clip_m: int = 2200):
    """Downsample a signed elevation-change field to compact meters."""
    try:
        import numpy as np
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("NumPy and Pillow are required for terrain diagnostics. Install with: pip install -r requirements.txt") from exc
    arr = np.asarray(field, dtype=np.float32)
    if arr.size == 0:
        return None
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    arr = np.clip(arr, -float(clip_m), float(clip_m))
    resized = np.asarray(Image.fromarray(arr.astype(np.float32), mode="F").resize((diag_w, diag_h), Image.Resampling.BICUBIC), dtype=np.float32)
    return np.rint(np.clip(resized, -float(clip_m), float(clip_m))).astype(int).tolist()


def _generate_plate_uplift_and_rift_fields(np_rng, lons, lats, cos_lat, rng: random.Random, land_support, *, controls: dict | None = None, geology: GeologyState | None = None):
    """Generate pseudo-plate uplift/rift fields from drifting province seeds.

    This remains procedural rather than a geologic-time plate reconstruction,
    but Stage 3C.2 gives the skeleton more geological meaning: province types,
    age proxies, richer boundary classes, boundary strength/width diagnostics,
    and target-plate-count control from the terrain review settings.
    """
    try:
        import numpy as np
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("NumPy and Pillow are required for plate-field terrain generation. Install with: pip install -r requirements.txt") from exc

    controls = controls or {}
    h = len(lats)
    w = len(lons)
    low_w = max(256, min(1024, w // 5))
    low_h = max(128, min(512, h // 5))
    low_lats = np.linspace(90.0 - 90.0 / low_h, -90.0 + 90.0 / low_h, low_h, dtype=np.float32)
    low_lons = np.linspace(-180.0 + 180.0 / low_w, 180.0 - 180.0 / low_w, low_w, dtype=np.float32)
    lon_grid, lat_grid = np.meshgrid(low_lons, low_lats)
    cos_grid = np.maximum(0.18, np.cos(np.radians(lat_grid)))

    target_plate_count = controls.get("target_plate_count")
    try:
        target_plate_count = int(round(float(target_plate_count)))
    except Exception:
        target_plate_count = rng.randint(28, 46)
    fragmentation = clamp(float(controls.get("fragmentation_tendency", 0.50) or 0.50), 0.0, 1.0)
    rift_strength_control = clamp(float(controls.get("rift_strength", 0.45) or 0.45), 0.0, 1.0)
    island_density = clamp(float(controls.get("island_density", 0.45) or 0.45), 0.0, 1.0)
    mountain_strength = clamp(float(controls.get("mountain_belt_strength", 1.0) or 1.0), 0.0, 3.0) / 3.0
    volcanism = clamp(float(getattr(geology, "volcanism", 0.65) if geology is not None else 0.65), 0.0, 1.8) / 1.8
    internal_heat = clamp(float(getattr(geology, "internal_heat", 0.75) if geology is not None else 0.75), 0.0, 2.0) / 2.0

    plate_count = int(round(target_plate_count + rng.uniform(-4, 5) + fragmentation * 8 - max(0.0, controls.get("effective_supercontinent_score", 0.45) - 0.55) * 8))
    plate_count = max(8, min(72, plate_count))
    centers: list[tuple[float, float, float, float, float, float]] = []
    for _ in range(plate_count):
        lon0 = rng.uniform(-180.0, 180.0)
        lat0 = rng.triangular(-62.0, 62.0, 0.0)
        heading = rng.uniform(0.0, math.tau)
        speed = rng.uniform(0.35, 1.25) * (0.75 + 0.45 * internal_heat + 0.20 * fragmentation)
        age_bias = rng.random()
        centers.append((lon0, lat0, math.cos(heading) * speed, math.sin(heading) * speed, rng.uniform(0.75, 1.40), age_bias))

    best = np.full((low_h, low_w), 1.0e9, dtype=np.float32)
    second = np.full((low_h, low_w), 1.0e9, dtype=np.float32)
    best_id = np.zeros((low_h, low_w), dtype=np.int16)
    second_id = np.zeros((low_h, low_w), dtype=np.int16)
    for i, (lon0, lat0, _vx, _vy, weight, _age) in enumerate(centers):
        dlon = _wrapped_lon_delta_array(lon_grid, lon0) * cos_grid
        dlat = lat_grid - lat0
        # Anisotropic weighted plates/provinces keep the map from looking like a
        # perfect radial Voronoi diagram.
        dist = (dlon * dlon * rng.uniform(0.80, 1.30) + dlat * dlat * rng.uniform(0.70, 1.25)) / weight
        better = dist < best
        second = np.where(better, best, np.minimum(second, dist))
        second_id = np.where(better, best_id, np.where(dist < second, i, second_id))
        best = np.where(better, dist, best)
        best_id = np.where(better, i, best_id)

    boundary = 1.0 - np.clip((second - best) / np.maximum(second + best, 1.0e-6) * (6.0 + 2.2 * fragmentation), 0.0, 1.0)
    boundary = _smooth_wrapped_array(boundary.astype(np.float32), passes=2)

    vx = np.asarray([p[2] for p in centers], dtype=np.float32)
    vy = np.asarray([p[3] for p in centers], dtype=np.float32)
    province_age_seed = np.asarray([p[5] for p in centers], dtype=np.float32)
    rel_speed = np.sqrt((vx[best_id] - vx[second_id]) ** 2 + (vy[best_id] - vy[second_id]) ** 2)
    rel_speed = np.clip(rel_speed / max(float(rel_speed.max()), 1.0e-6), 0.0, 1.0)

    mode_noise = _smooth_wrapped_array(np_rng.uniform(-1.0, 1.0, size=(low_h, low_w)).astype(np.float32), passes=3)
    compression = boundary * np.clip(0.40 + 0.74 * rel_speed + 0.25 * mode_noise + 0.22 * mountain_strength, 0.0, 1.55)
    rift = boundary * np.clip(0.30 + 0.45 * rel_speed - 0.32 * mode_noise + 0.48 * rift_strength_control, 0.0, 1.45)
    shear = boundary * np.clip(0.35 + 0.78 * rel_speed - np.abs(mode_noise) * 0.18 + 0.22 * fragmentation, 0.0, 1.40)

    low_land = np.asarray(Image.fromarray(np.asarray(land_support, dtype=np.float32), mode="F").resize((low_w, low_h), Image.Resampling.BICUBIC), dtype=np.float32)
    low_land = np.clip(low_land, 0.0, 1.0)
    gy_land, gx_land = np.gradient(low_land)
    land_edge = np.clip(np.sqrt(gx_land * gx_land + gy_land * gy_land) * 8.0, 0.0, 1.0)

    # Diagnostic boundary classes:
    # 0 intraplate, 1 convergent/collision, 2 divergent/rift,
    # 3 transform/shear, 4 passive margin, 5 diffuse/suture, 6 volcanic arc.
    boundary_class = np.zeros((low_h, low_w), dtype=np.uint8)
    active = boundary > (0.33 - 0.04 * fragmentation)
    convergent = active & (compression >= rift) & (compression >= shear * 0.85)
    divergent = active & (rift > compression) & (rift >= shear * 0.85)
    transform = active & (shear > compression * 1.05) & (shear > rift * 1.05)
    volcanic_arc = convergent & (low_land > 0.12) & (low_land < 0.76) & (mode_noise > (0.05 - 0.38 * volcanism))
    diffuse_suture = active & ~(convergent | divergent | transform) & (rel_speed < 0.55)
    passive_margin = (~active) & (land_edge > 0.24) & (boundary < 0.48)
    boundary_class[convergent] = 1
    boundary_class[divergent] = 2
    boundary_class[transform] = 3
    boundary_class[passive_margin] = 4
    boundary_class[diffuse_suture] = 5
    boundary_class[volcanic_arc] = 6

    boundary_width = _smooth_wrapped_array(np.clip(boundary * (0.55 + 0.40 * rel_speed + 0.25 * fragmentation), 0.0, 1.0), passes=3)
    boundary_strength = np.clip(np.maximum.reduce([compression / 1.55, rift / 1.45, shear / 1.40]) * (0.55 + 0.45 * boundary_width), 0.0, 1.0)

    # Province age and type proxy. This is not literal age; it marks old/stable
    # versus young/active crust so the review pages can diagnose the geologic
    # skeleton before the later mountain/coast passes are rewritten.
    age = province_age_seed[best_id]
    age = np.clip(0.50 * age + 0.25 * (1.0 - boundary_strength) + 0.18 * (low_land > 0.72) + 0.10 * (low_land < 0.15), 0.0, 1.0)
    province_type = np.zeros((low_h, low_w), dtype=np.uint8)  # old oceanic basin
    province_type[(low_land < 0.22) & ((age < 0.38) | (boundary_class == 2))] = 1  # young oceanic/ridge
    province_type[low_land > 0.72] = 2  # continental core
    province_type[(low_land > 0.25) & (low_land <= 0.72)] = 3  # shelf/margin
    province_type[(boundary_class == 2) & (low_land > 0.22)] = 4  # rifted margin
    province_type[(boundary_class == 6)] = 5  # volcanic arc
    province_type[(low_land > 0.38) & (low_land < 0.78) & (fragmentation + island_density + mode_noise * 0.20 > 1.0)] = 6  # terrane/microcontinent
    province_type[(low_land > 0.72) & (age < 0.42) & (mode_noise < -0.12)] = 7  # sedimentary basin
    province_type[(low_land > 0.80) & (age > 0.63) & (boundary_strength < 0.32)] = 8  # shield/highland

    # Upscale to target grid and apply land support: ocean ridges stay subtle;
    # continental collisions dominate where land already exists.
    comp = np.asarray(Image.fromarray(compression.astype(np.float32), mode="F").resize((w, h), Image.Resampling.BICUBIC), dtype=np.float32)
    rift_big = np.asarray(Image.fromarray(rift.astype(np.float32), mode="F").resize((w, h), Image.Resampling.BICUBIC), dtype=np.float32)
    support = np.clip(land_support * 2.1, 0.0, 1.0)
    comp = _smooth_wrapped_array(np.clip(comp, 0.0, 1.5), passes=1) * support
    rift_big = _smooth_wrapped_array(np.clip(rift_big, 0.0, 1.2), passes=1) * (0.35 + 0.65 * support)

    # Keep fixed-size low-resolution diagnostics to avoid storing full-size
    # plate rasters. They are enough to inspect plate layout and boundary types.
    diag_w = 512
    diag_h = 256
    def diag_nearest(arr):
        return np.asarray(Image.fromarray(arr.astype(np.int32), mode="I").resize((diag_w, diag_h), Image.Resampling.NEAREST), dtype=np.int32)
    def diag_float_x1000(arr):
        resized = np.asarray(Image.fromarray(np.clip(arr, 0.0, 1.0).astype(np.float32), mode="F").resize((diag_w, diag_h), Image.Resampling.BICUBIC), dtype=np.float32)
        return np.rint(np.clip(resized, 0.0, 1.0) * 1000.0).astype(np.int32)

    plate_diag = diag_nearest(best_id)
    boundary_diag = diag_nearest(boundary_class)
    diagnostics = {
        "plate_id": plate_diag.astype(int).tolist(),
        "boundary_class": boundary_diag.astype(int).tolist(),
        "province_type": diag_nearest(province_type).astype(int).tolist(),
        "province_age_x1000": diag_float_x1000(age).astype(int).tolist(),
        "boundary_strength_x1000": diag_float_x1000(boundary_strength).astype(int).tolist(),
        "boundary_width_x1000": diag_float_x1000(boundary_width).astype(int).tolist(),
        "plate_model": {
            "requested_target_plate_count": int(target_plate_count),
            "effective_plate_count": int(plate_count),
            "fragmentation_tendency": round(float(fragmentation), 3),
            "rift_strength": round(float(rift_strength_control), 3),
            "note": "Procedural province skeleton; not a full time-evolved plate reconstruction.",
        },
    }
    return comp.astype(np.float32), rift_big.astype(np.float32), diagnostics



def _build_plate_tectonic_v1_crust_diagnostic(terrain: TerrainMap, plate_coasts: dict):
    """Build crust diagnostic from active plate-mode shelf fields only.

    This prevents the main crust map from reintroducing the old shallow-depth
    shelf-halo diagnostic around every island.  In plate mode, shelf/marginal
    sea class means the native margin model created shelf support.
    """
    try:
        import numpy as np
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("NumPy and Pillow are required for plate crust diagnostics. Install with: pip install -r requirements.txt") from exc
    elev = np.asarray(terrain.elevation_m, dtype=np.int32)
    land = np.asarray(terrain.is_land, dtype=bool)
    shelf = np.asarray(plate_coasts.get("shelf_width_x1000"), dtype=np.float32)
    if shelf.shape != elev.shape:
        shelf = np.asarray(Image.fromarray(np.clip(shelf / 1000.0, 0.0, 1.0).astype(np.float32), mode="F").resize((elev.shape[1], elev.shape[0]), Image.Resampling.BICUBIC), dtype=np.float32) * 1000.0
    shelf_support = np.clip(shelf / 1000.0, 0.0, 1.0)
    codes = np.zeros(elev.shape, dtype=np.uint8)
    codes[(~land) & (shelf_support > 0.18)] = 1
    codes[land] = 2
    codes[land & (elev > 1600)] = 3
    diag = np.asarray(Image.fromarray(codes, mode="L").resize((512, 256), Image.Resampling.NEAREST), dtype=np.uint8)
    return diag.astype(int).tolist()


def _build_crust_diagnostic(land_mask, elevation, boundary_class_full=None, province_type_full=None):
    """Build a lightweight 512x256 crust-type diagnostic grid.

    Codes:
      0 deep oceanic crust
      1 continental shelf / marginal sea
      2 continental crust
      3 high continental / orogenic crust

    Shelf/marginal sea is no longer based on shallow depth around any land.  It
    is restricted to shallow water adjacent to continent-scale land or passive /
    rifted continental margins, so volcanic islands can have deep water nearby.
    """
    try:
        import numpy as np
        from PIL import Image
        from scipy import ndimage
    except ImportError as exc:
        raise RuntimeError("NumPy, SciPy, and Pillow are required for crust diagnostics. Install with: pip install -r requirements.txt") from exc
    land = land_mask.astype(bool, copy=False)
    elev = elevation.astype(np.int32, copy=False)
    h, w = elev.shape
    codes = np.zeros(elev.shape, dtype=np.uint8)

    continent_support = _continent_scale_land_mask(land, min_fraction=0.0025)
    if not continent_support.any():
        continent_support = land
    dist_to_continent = ndimage.distance_transform_edt(~continent_support)
    margin_support = np.zeros_like(land, dtype=bool)
    if boundary_class_full is not None:
        b = np.asarray(boundary_class_full, dtype=np.uint8)
        if b.shape == land.shape:
            margin_support |= np.isin(b, [2, 4])  # rift / passive margins
    if province_type_full is not None:
        pt = np.asarray(province_type_full, dtype=np.uint8)
        if pt.shape == land.shape:
            margin_support |= np.isin(pt, [3, 4])  # shelf/margin and rifted margin provinces
    continental_shelf = (~land) & (elev > -850) & (dist_to_continent < max(4, min(42, w // 95)))
    continental_shelf &= (margin_support | (dist_to_continent < max(3, min(18, w // 210))))
    codes[continental_shelf] = 1
    codes[land] = 2
    codes[land & (elev > 1600)] = 3
    diag = np.asarray(Image.fromarray(codes, mode="L").resize((512, 256), Image.Resampling.NEAREST), dtype=np.uint8)
    return diag.astype(int).tolist()


def _continent_scale_land_mask(land, min_fraction: float = 0.0025):
    """Return a mask of land components large enough to support continental shelves."""
    try:
        import numpy as np
        from scipy import ndimage
    except ImportError as exc:
        raise RuntimeError("NumPy and SciPy are required for component diagnostics. Install with: pip install -r requirements.txt") from exc
    labels, count = ndimage.label(land.astype(bool, copy=False), structure=np.array([[0,1,0],[1,1,1],[0,1,0]], dtype=np.uint8))
    if count <= 0:
        return np.zeros_like(land, dtype=bool)
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    threshold = max(32, int(land.size * min_fraction))
    keep = sizes >= threshold
    return keep[labels]

def _generate_erodibility_field(np_rng, lon_grid, lat_grid, cos_lat, rng: random.Random):
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("NumPy is required for erodibility generation. Install it with: pip install numpy") from exc

    h, w = lon_grid.shape
    field = np.full((h, w), 1.0, dtype=np.float32)
    for _ in range(rng.randint(8, 14)):
        lon0 = rng.uniform(-180.0, 180.0)
        lat0 = rng.uniform(-70.0, 70.0)
        amp = rng.uniform(-0.45, 0.58)
        lon_scale = rng.uniform(18.0, 64.0)
        lat_scale = rng.uniform(12.0, 42.0)
        dlon = _wrapped_lon_delta_array(lon_grid, lon0) * cos_lat
        dlat = lat_grid - lat0
        field += amp * np.exp(-((dlon / lon_scale) ** 2 + (dlat / lat_scale) ** 2))

    # Add broad but not global patches so adjacent coasts can have very
    # different resistance/roughness.
    field += 0.18 * _smooth_wrapped_array(np_rng.uniform(-1.0, 1.0, size=(h, w)).astype(np.float32), passes=6)
    return np.clip(field, 0.45, 1.85).astype(np.float32)


def _apply_erosion_and_deposition(
    elevation,
    land_mask,
    geology: GeologyState,
    rng: random.Random,
    erodibility,
    *,
    terrain_controls: dict | None = None,
    mountain_field=None,
    basin_field=None,
    rift_field=None,
    shield_highland_field=None,
    plateau_field=None,
):
    """Stage 3C.5 terrain maturity, erosion, deposition, and valley corridors.

    The previous pass only carved channels from a flow-accumulation proxy and
    gently smoothed lowlands.  This version still avoids a full hydrology model,
    but it prepares the terrain for hydrology by producing reviewable internal
    fields: erosion strength, deposition sinks, valley corridors, sediment
    supply, coastal plains, alluvial fans, floodplain tendency, terrain maturity,
    and before/after relief delta.
    """
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("NumPy is required for erosion/deposition. Install it with: pip install numpy") from exc

    controls = terrain_controls or {}
    h, w = elevation.shape
    elev = elevation.astype(np.float32, copy=True)
    pre_elev = elev.copy()
    land = land_mask.astype(bool, copy=False)

    def _field_or_zero(field):
        if field is None:
            return np.zeros((h, w), dtype=np.float32)
        arr = np.asarray(field, dtype=np.float32)
        if arr.shape != (h, w):
            return np.zeros((h, w), dtype=np.float32)
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0, copy=False)
        if arr.size == 0:
            return arr
        min_v = float(arr.min())
        max_v = float(arr.max())
        # Most generator fields are already 0..1 float32 arrays.  Reuse them
        # without copying so high-resolution terrain runs do not pay for five
        # additional full-map duplicates.
        if min_v >= 0.0 and max_v <= 1.05:
            return arr
        max_abs = max(abs(min_v), abs(max_v))
        if max_abs > 1.0e-6:
            arr = arr / max_abs
        return np.clip(arr, 0.0, 1.0).astype(np.float32, copy=False)

    mountain_f = _field_or_zero(mountain_field)
    basin_f = _field_or_zero(basin_field)
    rift_f = _field_or_zero(rift_field)
    shield_f = _field_or_zero(shield_highland_field)
    plateau_f = _field_or_zero(plateau_field)

    erosion_control = clamp(float(controls.get("erosion_deposition_strength", 0.45) or 0.45), 0.0, 1.0)
    deposition_control = clamp(float(controls.get("deposition_strength", erosion_control) or erosion_control), 0.0, 1.0)
    valley_control = clamp(float(controls.get("valley_carving_strength", erosion_control) or erosion_control), 0.0, 1.0)
    sediment_control = clamp(float(controls.get("sediment_supply_strength", 0.45 + 0.25 * erosion_control) or (0.45 + 0.25 * erosion_control)), 0.0, 1.0)
    coastal_plain_control = clamp(float(controls.get("coastal_plain_strength", controls.get("coastal_plain_bias", 0.45)) or 0.45), 0.0, 1.0)
    alluvial_fan_control = clamp(float(controls.get("alluvial_fan_strength", 0.34 + 0.25 * erosion_control) or (0.34 + 0.25 * erosion_control)), 0.0, 1.0)
    floodplain_control = clamp(float(controls.get("floodplain_strength", 0.32 + 0.25 * erosion_control) or (0.32 + 0.25 * erosion_control)), 0.0, 1.0)
    maturity_control = clamp(float(controls.get("terrain_maturity", 0.32 + 0.30 * erosion_control) or (0.32 + 0.30 * erosion_control)), 0.0, 1.0)
    ocean_target = 0.62
    try:
        ocean_target = clamp(float((controls.get("derivation_inputs") or {}).get("ocean_fraction_target", 0.62)), 0.05, 0.95)
    except Exception:
        pass

    # Compute steepest downhill neighbor with vectorized rolled neighbor grids.
    # Rows do not wrap over poles; columns wrap around longitude.
    invalid_high = np.full((h, w), 1.0e9, dtype=np.float32)
    best_drop = np.zeros((h, w), dtype=np.float32)
    best_target = np.full((h, w), -1, dtype=np.int32)
    flat_index = np.arange(h * w, dtype=np.int32).reshape(h, w)

    for dr, dc, dist in (
        (-1, -1, 1.4142), (-1, 0, 1.0), (-1, 1, 1.4142),
        (0, -1, 1.0),                 (0, 1, 1.0),
        (1, -1, 1.4142),  (1, 0, 1.0),  (1, 1, 1.4142),
    ):
        neigh_elev = np.roll(elev, -dc, axis=1)
        neigh_idx = np.roll(flat_index, -dc, axis=1)
        if dr < 0:
            neigh_elev = np.vstack((invalid_high[0:1, :], neigh_elev[:-1, :]))
            neigh_idx = np.vstack((np.full((1, w), -1, dtype=np.int32), neigh_idx[:-1, :]))
        elif dr > 0:
            neigh_elev = np.vstack((neigh_elev[1:, :], invalid_high[-1:, :]))
            neigh_idx = np.vstack((neigh_idx[1:, :], np.full((1, w), -1, dtype=np.int32)))
        drop = (elev - neigh_elev) / dist
        better = land & (drop > best_drop) & (neigh_idx >= 0)
        best_drop[better] = drop[better]
        best_target[better] = neigh_idx[better]

    flat_land = land.ravel()
    flat_elev = elev.ravel()
    downstream = best_target.ravel()

    # Accumulate flow from high to low. This remains a simple terrain-flow proxy,
    # not the later hydrology solver. Its purpose is valley readiness.
    accum = np.where(flat_land, 1.0, 0.0).astype(np.float32)
    order = np.argsort(flat_elev)[::-1]
    for idx in order:
        if not flat_land[idx]:
            continue
        target = int(downstream[idx])
        if target >= 0 and flat_land[target]:
            accum[target] += accum[idx]

    land_acc = accum[flat_land]
    if land_acc.size == 0:
        empty_diag = {k: None for k in (
            "terrain_erosion_strength_x1000", "terrain_deposition_field_x1000", "terrain_valley_corridor_x1000",
            "terrain_sediment_supply_x1000", "terrain_coastal_plain_x1000", "terrain_alluvial_fan_x1000",
            "terrain_floodplain_x1000", "terrain_maturity_x1000", "terrain_relief_delta_m",
        )}
        return elevation, empty_diag, {"stage": "3C.5", "note": "no land cells"}

    acc_scale = max(float(np.quantile(land_acc, 0.996)), 1.0)
    river_power = np.clip(np.log1p(accum) / max(1e-6, math.log1p(acc_scale)), 0.0, 1.0).reshape(h, w)

    smoothed = _smooth_wrapped_array(elev, passes=4)
    local_relief = np.maximum(0.0, elev - smoothed)
    relief_scale = max(float(np.quantile(local_relief[land], 0.97)) if land.any() else 1.0, 1.0)
    relief_factor = np.clip(local_relief / relief_scale, 0.0, 1.0)

    gy, gx = np.gradient(elev)
    slope = np.sqrt(gx * gx + gy * gy)
    slope_scale = max(float(np.quantile(slope[land], 0.96)) if land.any() else 1.0, 1.0)
    slope_factor = np.clip(slope / slope_scale, 0.0, 1.0)

    coastal_transition = _land_ocean_transition_zone(land, radius=max(3, min(18, w // 260)))
    coastal_zone = coastal_transition.astype(np.float32)
    lowland = land & (elev > 0) & (elev < (760 + 280 * deposition_control))
    mid_low = land & (elev > 0) & (elev < 1450)

    wetness_proxy = clamp(0.30 + 0.45 * ocean_target + 0.10 * erosion_control, 0.15, 0.95)
    dry_preservation = 1.0 - wetness_proxy

    # Valley corridors combine flow, rift lines, low basin exits, and steep
    # mountain-front drains.  This is intentionally pre-hydrology: it shapes the
    # terrain so the hydrology pass later has plausible routes to discover.
    valley_corridor = np.clip(
        0.52 * river_power
        + 0.18 * rift_f
        + 0.14 * basin_f * (0.45 + river_power)
        + 0.12 * mountain_f * slope_factor
        + 0.06 * plateau_f * river_power,
        0.0,
        1.0,
    ) * land.astype(np.float32)
    valley_corridor = _smooth_wrapped_array(valley_corridor, passes=1)

    sediment_supply = np.clip(
        0.38 * valley_corridor * (0.35 + relief_factor)
        + 0.26 * mountain_f * (0.35 + slope_factor)
        + 0.14 * plateau_f
        + 0.10 * rift_f
        + 0.12 * np.clip(erodibility, 0.45, 1.85) / 1.85,
        0.0,
        1.0,
    ) * land.astype(np.float32) * (0.55 + 0.65 * sediment_control)

    coastal_plain = np.clip((coastal_zone * lowland.astype(np.float32)) * (0.45 + 0.85 * coastal_plain_control) * (0.65 + 0.35 * wetness_proxy), 0.0, 1.0)
    floodplain = np.clip((river_power ** 0.72) * mid_low.astype(np.float32) * (0.40 + 0.95 * floodplain_control), 0.0, 1.0)
    # Fans appear where high relief or rift shoulders drop into basins/lowlands.
    fan_source = np.clip((mountain_f + 0.55 * rift_f + 0.35 * plateau_f) * (0.40 + slope_factor), 0.0, 1.0)
    fan_sink = np.clip((basin_f + lowland.astype(np.float32) * 0.55 + coastal_zone * 0.25) * (1.0 - np.clip(slope_factor, 0.0, 0.75)), 0.0, 1.0)
    alluvial_fan = np.clip(_smooth_wrapped_array(fan_source, passes=1) * _smooth_wrapped_array(fan_sink, passes=2) * (0.35 + 1.05 * alluvial_fan_control), 0.0, 1.0) * land.astype(np.float32)

    deposition_field = np.clip(
        0.32 * floodplain
        + 0.24 * basin_f * lowland.astype(np.float32)
        + 0.18 * coastal_plain
        + 0.18 * alluvial_fan
        + 0.08 * sediment_supply * (lowland.astype(np.float32) + 0.30 * coastal_zone),
        0.0,
        1.0,
    ) * land.astype(np.float32) * (0.55 + 0.75 * deposition_control)

    # Maturity is spatial, not just a single slider: old shields and wet lower
    # relief mature faster; young rugged mountain belts preserve sharper relief.
    maturity_field = np.clip(
        maturity_control * (0.52 + 0.30 * wetness_proxy + 0.18 * shield_f)
        + 0.16 * basin_f
        + 0.10 * coastal_plain
        - 0.12 * mountain_f * (1.0 - geology.erosion / 2.5),
        0.0,
        1.0,
    ) * land.astype(np.float32)

    erosion_strength_field = np.clip(
        (0.30 + 0.70 * erosion_control)
        * (0.35 + 0.65 * wetness_proxy)
        * (0.32 + 0.42 * valley_corridor + 0.26 * relief_factor)
        * np.clip(erodibility, 0.45, 1.85) / 1.45,
        0.0,
        1.25,
    ) * land.astype(np.float32)

    base_erosion_m = 520.0 + 1120.0 * geology.erosion * (0.45 + 0.85 * erosion_control)
    valley_cut = valley_corridor * (0.50 + 1.10 * relief_factor + 0.30 * slope_factor) * base_erosion_m * (0.38 + 0.92 * valley_control)
    # Preserve very dry scarps and old shields by reducing some smoothing/cutting,
    # while still allowing major valleys to form.
    preservation = np.clip(0.20 * dry_preservation * shield_f + 0.12 * dry_preservation * plateau_f, 0.0, 0.32)
    valley_cut *= (1.0 - preservation)

    mature_smooth = _smooth_wrapped_array(elev, passes=3)
    smooth_mix = np.clip((0.08 + 0.32 * maturity_field) * (0.35 + 0.65 * relief_factor), 0.0, 0.55)
    eroded = np.where(land, elev * (1.0 - smooth_mix) + mature_smooth * smooth_mix - valley_cut, elev)

    deposition_m = deposition_field * (140.0 + 560.0 * deposition_control + 160.0 * wetness_proxy)
    deposition_m += floodplain * river_power * (90.0 + 260.0 * floodplain_control)
    deposition_m += alluvial_fan * (120.0 + 360.0 * alluvial_fan_control)
    eroded = np.where(land, eroded + deposition_m, eroded)

    # Broader plain smoothing: floodplains and filled basins should be readable
    # at medium scale, but high mountain belts should remain intact.
    plain_smooth = _smooth_wrapped_array(eroded, passes=2)
    plain_mix = np.clip((0.10 * floodplain + 0.12 * deposition_field + 0.08 * coastal_plain) * (0.75 + deposition_control), 0.0, 0.42)
    plain_mix *= (1.0 - 0.45 * mountain_f)
    eroded = eroded * (1.0 - plain_mix) + plain_smooth * plain_mix

    # Keep land above water; sea-level fitting and coast styling happen later.
    eroded = np.where(land, np.maximum(eroded, 1.0), elev)
    out = np.rint(eroded).astype(np.int32)
    relief_delta = np.where(land, out.astype(np.float32) - pre_elev, 0.0)

    diagnostics = {
        "terrain_erosion_strength_x1000": diagf(erosion_strength_field),
        "terrain_deposition_field_x1000": diagf(deposition_field),
        "terrain_valley_corridor_x1000": diagf(valley_corridor),
        "terrain_sediment_supply_x1000": diagf(sediment_supply),
        "terrain_coastal_plain_x1000": diagf(coastal_plain),
        "terrain_alluvial_fan_x1000": diagf(alluvial_fan),
        "terrain_floodplain_x1000": diagf(floodplain),
        "terrain_maturity_x1000": diagf(maturity_field),
        "terrain_relief_delta_m": _diagnostic_signed_delta_m(relief_delta),
    }
    meta = {
        "stage": "3C.5 erosion/deposition/valley-corridors",
        "erosion_control": round(float(erosion_control), 3),
        "deposition_control": round(float(deposition_control), 3),
        "valley_carving_strength": round(float(valley_control), 3),
        "sediment_supply_strength": round(float(sediment_control), 3),
        "coastal_plain_strength": round(float(coastal_plain_control), 3),
        "alluvial_fan_strength": round(float(alluvial_fan_control), 3),
        "floodplain_strength": round(float(floodplain_control), 3),
        "terrain_maturity": round(float(maturity_control), 3),
        "mean_valley_corridor": round(float(np.mean(valley_corridor[land])) if land.any() else 0.0, 4),
        "mean_deposition_field": round(float(np.mean(deposition_field[land])) if land.any() else 0.0, 4),
        "mean_erosion_strength": round(float(np.mean(erosion_strength_field[land])) if land.any() else 0.0, 4),
        "mean_relief_delta_m": round(float(np.mean(relief_delta[land])) if land.any() else 0.0, 2),
        "max_valley_corridor": round(float(np.max(valley_corridor[land])) if land.any() else 0.0, 4),
        "max_deposition_field": round(float(np.max(deposition_field[land])) if land.any() else 0.0, 4),
        "note": "Pre-hydrology terrain-conditioning diagnostics; final rivers are generated later in the hydrology stage.",
    }
    return out, diagnostics, meta

def _remove_small_land_components(land_mask, min_size: int):
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("NumPy is required for land cleanup. Install it with: pip install numpy") from exc
    from collections import deque

    h, w = land_mask.shape
    # Full component labeling over multi-million-cell maps is expensive and
    # unnecessary now that oceanic islands are generated deliberately after this
    # cleanup. For large full-resolution runs, keep the mask and rely on the
    # later ribbon/water repair passes.
    if h * w > 1_200_000:
        return land_mask
    visited = np.zeros((h, w), dtype=bool)
    cleaned = land_mask.copy()
    for r in range(h):
        for c in range(w):
            if visited[r, c] or not land_mask[r, c]:
                continue
            q = deque([(r, c)])
            visited[r, c] = True
            cells = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr = rr + dr
                    if nr < 0 or nr >= h:
                        continue
                    nc = (cc + dc) % w
                    if not visited[nr, nc] and land_mask[nr, nc]:
                        visited[nr, nc] = True
                        q.append((nr, nc))
            if len(cells) < min_size:
                for rr, cc in cells:
                    cleaned[rr, cc] = False
    return cleaned


def _irregularize_land_water_edges(land_mask, elevation, np_rng, amount: float = 0.18):
    """Break blocky upscaled land/water edges with smoothed noise.

    This is only applied to cells already close to an existing land/water edge,
    so it cannot flood continental interiors or create global shallow-sea
    networks. It mainly removes square/block artefacts introduced by resizing
    lower-resolution feature masks.
    """
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("NumPy is required for edge irregularization. Install it with: pip install numpy") from exc
    h, w = land_mask.shape
    edge_zone = _land_ocean_transition_zone(land_mask, radius=max(2, min(7, w // 900)))
    noise = np_rng.uniform(-1.0, 1.0, size=(h, w)).astype(np.float32)
    noise = _smooth_wrapped_array(noise, passes=3)
    near_level = np.exp(-((elevation.astype(np.float32)) / 520.0) ** 2)
    flip_to_land = edge_zone & (~land_mask) & (noise > (0.70 - amount)) & (near_level > 0.35)
    flip_to_water = edge_zone & land_mask & (noise < (-0.72 + amount)) & (elevation < 85)
    out = land_mask.copy()
    out[flip_to_land] = True
    out[flip_to_water] = False
    return out



def _repair_artificial_coast_and_inland_water(land_mask, elevation):
    """Remove shelf ribbons and square inland water artefacts.

    This deliberately avoids using elevation alone as the land/sea decision.
    Low valleys may be near sea level while still being land; conversely a thin
    ring of low land on the continental shelf is usually an artefact. The repair
    pass combines neighborhood shape, component size, and shallowness so broad
    inland seas can survive while dendritic/seam-like shallow seas are filled.
    """
    land, elev = _remove_shelf_strip_artifacts(land_mask, elevation)
    land, elev = _remove_blocky_inland_water_artifacts(land, elev)
    land, elev = _remove_shelf_strip_artifacts(land, elev)
    return land, elev

def _remove_shelf_strip_artifacts(land_mask, elevation):
    """Remove low, narrow shelf-edge land ribbons without flattening real coasts."""
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("NumPy is required for shelf strip cleanup. Install it with: pip install numpy") from exc
    land = land_mask.astype(bool, copy=True)
    elev = elevation.astype(np.int32, copy=True)

    for pass_no in range(4):
        north = np.vstack((land[0:1, :], land[:-1, :]))
        south = np.vstack((land[1:, :], land[-1:, :]))
        west = np.roll(land, 1, axis=1)
        east = np.roll(land, -1, axis=1)
        nw = np.roll(north, 1, axis=1)
        ne = np.roll(north, -1, axis=1)
        sw = np.roll(south, 1, axis=1)
        se = np.roll(south, -1, axis=1)
        ocean_neighbors = ((~north).astype(np.int16) + (~south).astype(np.int16) + (~west).astype(np.int16) + (~east).astype(np.int16) +
                           (~nw).astype(np.int16) + (~ne).astype(np.int16) + (~sw).astype(np.int16) + (~se).astype(np.int16))
        cardinal_ocean = ((~north).astype(np.int16) + (~south).astype(np.int16) + (~west).astype(np.int16) + (~east).astype(np.int16))
        # Remove cells that are almost surrounded by water, or very low cells
        # forming a one/two-cell-wide shelf ring. The highland-neighbor check
        # protects real coastal cliffs and rugged headlands.
        high_neighbor = False
        for neigh in (north, south, west, east, nw, ne, sw, se):
            pass
        elev_n = np.vstack((elev[0:1, :], elev[:-1, :]))
        elev_s = np.vstack((elev[1:, :], elev[-1:, :]))
        elev_w = np.roll(elev, 1, axis=1)
        elev_e = np.roll(elev, -1, axis=1)
        high_neighbor = (elev_n > 140) | (elev_s > 140) | (elev_w > 140) | (elev_e > 140)
        ribbon = land & (elev < (82 if pass_no < 2 else 58)) & (ocean_neighbors >= 5) & (cardinal_ocean >= 2) & (~high_neighbor)
        isolated = land & (elev < 120) & (ocean_neighbors >= 7)
        strip = ribbon | isolated
        if not strip.any():
            break
        land[strip] = False
        elev[strip] = np.minimum(elev[strip], -10)
    return land, elev

def _remove_blocky_inland_water_artifacts(land_mask, elevation):
    """Fast cleanup for square/shallow inland water artifacts.

    This is intentionally vectorized. The earlier component-label cleanup was
    accurate but too slow for full-resolution 2K+ terrain. Repeated local rules
    fill shallow water cells that are mostly surrounded by land and break up
    square ponds/straight channels, while broad lakes/seas remain water.
    """
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("NumPy is required for inland water cleanup. Install it with: pip install numpy") from exc
    land = land_mask.astype(bool, copy=True)
    elev = elevation.astype(np.int32, copy=True)

    for pass_no in range(5):
        north = np.vstack((land[0:1, :], land[:-1, :]))
        south = np.vstack((land[1:, :], land[-1:, :]))
        west = np.roll(land, 1, axis=1)
        east = np.roll(land, -1, axis=1)
        nw = np.roll(north, 1, axis=1)
        ne = np.roll(north, -1, axis=1)
        sw = np.roll(south, 1, axis=1)
        se = np.roll(south, -1, axis=1)
        land_neighbors = (north.astype(np.int16) + south.astype(np.int16) + west.astype(np.int16) + east.astype(np.int16) +
                          nw.astype(np.int16) + ne.astype(np.int16) + sw.astype(np.int16) + se.astype(np.int16))
        cardinal_land = north.astype(np.int16) + south.astype(np.int16) + west.astype(np.int16) + east.astype(np.int16)
        water = ~land
        # Fill blocky/shallow water enclosed by land. Deeper/broader water bodies
        # survive because their interior cells do not meet the neighborhood rule.
        fill = water & ((land_neighbors >= 6) | ((land_neighbors >= 5) & (elev > -160)) | ((cardinal_land >= 3) & (elev > -240)))
        if not fill.any():
            break
        land[fill] = True
        elev[fill] = np.maximum(2, np.rint(np.abs(elev[fill]) * (0.08 + 0.03 * pass_no)).astype(np.int32))

    # One more pass to remove tiny square water corners and checkerboard holes.
    north = np.vstack((land[0:1, :], land[:-1, :]))
    south = np.vstack((land[1:, :], land[-1:, :]))
    west = np.roll(land, 1, axis=1)
    east = np.roll(land, -1, axis=1)
    cardinal_land = north.astype(np.int16) + south.astype(np.int16) + west.astype(np.int16) + east.astype(np.int16)
    fill = (~land) & (cardinal_land >= 3) & (elev > -420)
    land[fill] = True
    elev[fill] = np.maximum(2, np.rint(np.abs(elev[fill]) * 0.08).astype(np.int32))
    return land, elev

def _add_inland_sea_islands(rng: random.Random, land_mask, elevation, lons, lats, cos_lat):
    """Add a few islands to preserved large inland seas.

    Large enclosed seas without any islands look artificial. This only works on
    preserved water components that are clearly smaller than the world ocean, so
    normal oceans are handled by the separate oceanic island/arc generator.
    """
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("NumPy is required for inland-sea island generation. Install it with: pip install numpy") from exc
    from collections import deque

    h, w = land_mask.shape
    # Component tracing is intentionally skipped at high resolutions; oceanic
    # islands/arcs are still generated separately and this keeps full-resolution
    # terrain practical.
    if h * w > 1_200_000:
        return land_mask, elevation
    land = land_mask.astype(bool, copy=True)
    elev = elevation.astype(np.int32, copy=True)
    water = ~land
    visited = np.zeros((h, w), dtype=bool)
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    world_area = h * w
    components: list[list[tuple[int, int]]] = []

    for r in range(h):
        for c in range(w):
            if visited[r, c] or not water[r, c]:
                continue
            q = deque([(r, c)])
            visited[r, c] = True
            cells: list[tuple[int, int]] = []
            while q and len(cells) <= max(120_000, world_area // 30):
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for dr, dc in ((-1,0),(1,0),(0,-1),(0,1)):
                    nr = rr + dr
                    if nr < 0 or nr >= h:
                        continue
                    nc = (cc + dc) % w
                    if not visited[nr, nc] and water[nr, nc]:
                        visited[nr, nc] = True
                        q.append((nr, nc))
            area = len(cells)
            # Preserved inland seas: bigger than ponds, much smaller than oceans.
            if int(world_area * 0.00015) <= area <= int(world_area * 0.018):
                components.append(cells)

    for cells in components[:8]:
        island_count = rng.randint(1, 4)
        for _ in range(island_count):
            rr, cc = rng.choice(cells)
            lon0 = float(lons[cc]); lat0 = float(lats[rr])
            radius = rng.uniform(0.18, 0.65)
            dlon = _wrapped_lon_delta_array(lon_grid, lon0) * cos_lat
            dlat = lat_grid - lat0
            blob = np.exp(-((dlon / radius) ** 2 + (dlat / radius) ** 2))
            island_elev = elev.astype(np.float32) + blob * rng.uniform(90.0, 520.0)
            mask = blob > rng.uniform(0.42, 0.58)
            # Only place islands inside the same water component neighborhood.
            water_mask = ~land
            add = mask & water_mask & (island_elev > -25)
            land[add] = True
            elev[add] = np.maximum(2, np.rint(island_elev[add]).astype(np.int32))
    return land, elev



def _ragged_island_field(blob, np_rng, *, strength: float = 0.55):
    """Break circular/elliptical island kernels into lobed, eroded outlines."""
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("NumPy is required for island shape generation. Install it with: pip install numpy") from exc
    arr = np.asarray(blob, dtype=np.float32)
    if arr.size == 0:
        return arr
    noise = _smooth_wrapped_array(np_rng.uniform(0.0, 1.0, size=arr.shape).astype(np.float32), passes=1)
    bite = _smooth_wrapped_array(np_rng.uniform(0.0, 1.0, size=arr.shape).astype(np.float32), passes=2)
    # Enhance lobes near the edge and bite holes into the margin. Keep central
    # volcanic cores intact so islands still rise above sea level naturally.
    edge_weight = np.clip((arr - 0.10) / 0.55, 0.0, 1.0) * np.clip((0.92 - arr) / 0.62, 0.0, 1.0)
    shaped = arr * (0.76 + 0.58 * noise * strength)
    shaped -= edge_weight * bite * (0.20 + 0.42 * strength)
    shaped += np.maximum(0.0, arr - 0.72) * (0.25 + 0.20 * strength)
    return np.clip(shaped, 0.0, None).astype(np.float32)


def _add_oceanic_islands_and_arcs(rng: random.Random, land_mask, elevation, lons, lats, cos_lat, sea_threshold: float = 0.0, max_islands: int = 40):
    """Add volcanic islands, hotspot chains, and island arcs to oceans.

    Islands are generated after the main elevation scaling so they can rise from
    actual local bathymetry. This makes island topography depend on shelf/deep-
    ocean context instead of simply painting small green dots on water.
    """
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("NumPy is required for island generation. Install it with: pip install numpy") from exc
    h, w = land_mask.shape
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    land = land_mask.astype(bool, copy=True)
    elev = elevation.astype(np.int32, copy=True)
    ocean = ~land
    if not ocean.any():
        return land, elev
    local_np_rng = np.random.default_rng(rng.randrange(1, 2**63 - 1))

    def choose_ocean_cell(prefer_shallow: bool = False):
        for _attempt in range(80):
            rr = rng.randrange(h)
            cc = rng.randrange(w)
            if not ocean[rr, cc] or not (-72.0 < float(lats[rr]) < 72.0):
                continue
            depth = int(elev[rr, cc])
            if prefer_shallow:
                if -2200 <= depth <= -25:
                    return rr, cc
            else:
                if depth <= -45:
                    return rr, cc
        return None

    # Sparse isolated volcanic islands and small island groups. Sample shallow
    # shelves/margins more often than abyssal plains, but keep some deep-ocean
    # hotspots for Hawaii-style chains.
    isolated_count = max(14, min(max_islands, 120))
    for _ in range(isolated_count):
        chosen = choose_ocean_cell(prefer_shallow=(rng.random() < 0.68))
        if chosen is None:
            continue
        rr, cc = chosen
        lat0 = float(lats[rr])
        lon0 = float(lons[cc])
        local_depth = abs(int(elev[rr, cc]))
        radius = rng.uniform(0.24, 1.35)
        if local_depth < 900:
            radius *= rng.uniform(0.85, 1.75)
        height = local_depth * rng.uniform(0.72, 1.18) + rng.uniform(120.0, 1450.0)
        height = min(height, rng.uniform(1100.0, 4200.0))
        dlon = _wrapped_lon_delta_array(lon_grid, lon0) * cos_lat
        dlat = lat_grid - lat0
        cone = np.exp(-((dlon / radius) ** 2 + (dlat / (radius * rng.uniform(0.75, 1.25))) ** 2))
        # Secondary shoulders make larger islands less perfectly circular.
        if rng.random() < 0.46:
            lon1 = lon0 + rng.uniform(-1.4, 1.4)
            lat1 = clamp(lat0 + rng.uniform(-1.1, 1.1), -73.0, 73.0)
            dlon1 = _wrapped_lon_delta_array(lon_grid, lon1) * cos_lat
            dlat1 = lat_grid - lat1
            cone += 0.38 * np.exp(-((dlon1 / (radius * rng.uniform(0.55, 1.20))) ** 2 + (dlat1 / (radius * rng.uniform(0.55, 1.20))) ** 2))
        cone = _ragged_island_field(cone, local_np_rng, strength=rng.uniform(0.45, 0.90))
        island_elev = elev.astype(np.float32) + cone * height
        mask = ocean & (island_elev > 0)
        if int(mask.sum()) < max(1, (h * w) // 5_000_000):
            continue
        land[mask] = True
        elev[mask] = np.maximum(elev[mask], np.rint(island_elev[mask]).astype(np.int32))
        ocean = ~land

    # Curved island arcs and hotspot chains.
    arc_count = rng.randint(5, 10)
    for _ in range(arc_count):
        chosen = choose_ocean_cell(prefer_shallow=(rng.random() < 0.55))
        if chosen is None:
            continue
        rr, cc = chosen
        lon_c = float(lons[cc])
        lat_c = float(lats[rr])
        heading = rng.uniform(0.0, math.tau)
        node_count = rng.randint(8, 18)
        base_radius = rng.uniform(0.28, 1.25)
        for node in range(node_count):
            heading += rng.uniform(-0.56, 0.56)
            lon_c += math.cos(heading) * rng.uniform(1.2, 5.8) / max(0.3, math.cos(math.radians(lat_c)))
            lat_c = clamp(lat_c + math.sin(heading) * rng.uniform(0.9, 4.8), -72.0, 72.0)
            # Skip nodes that wandered onto continents.
            rr_guess = int(clamp(round((90.0 - lat_c) / 180.0 * h - 0.5), 0, h - 1))
            cc_guess = int(((lon_c + 180.0) / 360.0 * w) % w)
            if not ocean[rr_guess, cc_guess]:
                continue
            local_depth = abs(int(elev[rr_guess, cc_guess]))
            rad_lon = base_radius * rng.uniform(0.65, 1.75)
            rad_lat = base_radius * rng.uniform(0.55, 1.55)
            dlon = _wrapped_lon_delta_array(lon_grid, lon_c) * cos_lat
            dlat = lat_grid - lat_c
            blob = np.exp(-((dlon / rad_lon) ** 2 + (dlat / rad_lat) ** 2))
            blob = _ragged_island_field(blob, local_np_rng, strength=rng.uniform(0.50, 0.95))
            uplift = local_depth * rng.uniform(0.62, 1.06) + rng.uniform(220.0, 1850.0)
            if node in (0, node_count - 1):
                uplift *= rng.uniform(0.65, 0.95)
            island_elev = elev.astype(np.float32) + blob * min(uplift, rng.uniform(1400.0, 5200.0))
            mask = ocean & (island_elev > 0) & (blob > rng.uniform(0.18, 0.42))
            if int(mask.sum()) <= 0:
                continue
            land[mask] = True
            elev[mask] = np.maximum(elev[mask], np.rint(island_elev[mask]).astype(np.int32))
            ocean = ~land

    return land, elev

def _quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    q = clamp(q, 0.0, 1.0)
    ordered = sorted(values)
    index = int(q * (len(ordered) - 1))
    return ordered[index]


# ---------------------------------------------------------------------------
# Stage 2 review / explanation helpers
# ---------------------------------------------------------------------------

def _band(value: float, low: float, high: float, low_label: str, mid_label: str, high_label: str) -> str:
    if value < low:
        return low_label
    if value > high:
        return high_label
    return mid_label


def _stage2_rotation_metadata(rotation: RotationState, planet: Planet) -> dict:
    day = rotation.rotation_period_hours
    tilt = rotation.axial_tilt_degrees
    moon = planet.moon
    tide_level = "none" if moon is None else getattr(moon, "tidal_effect_level", "moderate")
    stability = "low" if moon is None else getattr(moon, "axial_stability_effect", "moderate")
    return {
        "rotation_class": _band(day, 18.0, 32.0, "fast", "earth_like", "slow"),
        "coriolis_strength": _band(day, 18.0, 34.0, "strong", "moderate", "weak"),
        "seasonality_class": _band(tilt, 10.0, 35.0, "low", "moderate", "high"),
        "axial_stability_class": "stable" if stability in {"high", "strong"} else ("unstable" if stability in {"low", "weak", "none"} else "moderate"),
        "tidal_braking": tide_level,
        "explanation": f"{day:.1f} h rotation gives {_band(day, 18.0, 34.0, 'strong', 'moderate', 'weak')} Coriolis behavior; {tilt:.1f}° tilt gives {_band(tilt, 10.0, 35.0, 'low', 'moderate', 'high')} seasons.",
    }


def _stage2_atmosphere_metadata(star: Star, planet: Planet, atmosphere: Atmosphere) -> dict:
    retention = clamp((planet.escape_velocity_relative_earth * 0.55 + planet.surface_gravity_g * 0.45) / max(0.85, planet.equilibrium_temperature_k / 255.0), 0.0, 2.0)
    pressure = atmosphere.pressure_bar
    co2 = atmosphere.carbon_dioxide_ppm
    greenhouse = atmosphere.greenhouse_warming_k
    temp_c = atmosphere.estimated_mean_surface_temp_c
    greenhouse_work = "heavy_lifting" if greenhouse > 40.0 else ("low" if greenhouse < 23.0 else "normal")
    return {
        "retention_score": round(retention, 3),
        "retention_class": _band(retention, 0.85, 1.35, "weak", "moderate", "strong"),
        "pressure_class": _band(pressure, 0.8, 1.8, "thin", "earthlike_to_moderate", "thick"),
        "co2_class": "low" if co2 < 350 else ("moderate" if co2 < 1200 else "high"),
        "greenhouse_workload": greenhouse_work,
        "climate_risk": "cold_edge" if temp_c < 4.0 else ("hot_risky" if temp_c > 30.0 else ("warm" if temp_c > 22.0 else "temperate")),
        "explanation": f"{pressure:.2f} bar atmosphere with {greenhouse:.1f} K greenhouse warming produces an estimated mean surface temperature of {temp_c:.1f} °C.",
    }


def _stage2_hydrosphere_metadata(planet: Planet, hydrosphere: Hydrosphere) -> dict:
    target = hydrosphere.ocean_fraction_target
    volatile = hydrosphere.volatile_fraction
    return {
        "target_land_fraction": round(1.0 - target, 3),
        "waterworld_risk": "high" if target >= 0.78 else ("moderate" if target >= 0.68 else "low"),
        "dry_world_risk": "high" if target <= 0.36 else ("moderate" if target <= 0.48 else "low"),
        "sea_level_sensitivity": "high" if target > 0.72 or target < 0.40 else "moderate",
        "continental_exposure_tendency": "low" if target > 0.72 else ("high" if target < 0.48 else "moderate"),
        "ice_storage_tendency": hydrosphere.ice_cap_tendency,
        "expected_coastline_complexity": "high" if 0.45 <= target <= 0.70 else "lower unless terrain is rugged",
        "volatile_inventory_note": f"Volatile fraction {volatile:.3f} is a formation inventory signal; ocean target {target:.2f} is the terrain sea-level goal.",
    }


def _stage2_geology_metadata(star: Star, planet: Planet, geology: GeologyState) -> dict:
    heat = geology.internal_heat
    volcanism = geology.volcanism
    rough = geology.surface_roughness
    ctx = planet.formation_context or {}
    if heat < 0.42:
        regime = "stagnant_lid_or_quiet"
    elif heat < 0.75:
        regime = "weak_plate_tectonics"
    elif volcanism < 1.15:
        regime = "earth_like_plate_tectonics"
    elif volcanism < 1.55:
        regime = "active_mobile_lid"
    else:
        regime = "volcanic_resurfacing"
    return {
        "tectonic_regime": regime,
        "orogenic_intensity": _band(geology.mountain_factor, 0.65, 1.25, "low", "moderate", "high"),
        "rift_tendency": _band(heat, 0.45, 1.05, "low", "moderate", "high"),
        "island_arc_tendency": _band(volcanism, 0.45, 1.10, "low", "moderate", "high"),
        "hotspot_tendency": _band(volcanism, 0.65, 1.35, "low", "moderate", "high"),
        "basin_formation_tendency": "high" if str(ctx.get("impact_history", "")).lower() in {"battered", "heavy_bombardment"} else _band(rough, 0.25, 0.75, "low", "moderate", "high"),
        "continental_fragmentation_tendency": "high" if str(ctx.get("crustal_asymmetry_bias", "")).lower() == "high" or heat > 1.05 else "moderate",
        "shelf_deposition_tendency": _band(geology.erosion, 0.85, 1.55, "low", "moderate", "high"),
        "crustal_contrast_strength": str(ctx.get("crustal_asymmetry_bias", "medium")),
        "explanation": f"Internal heat {heat:.2f}, volcanism {volcanism:.2f}, and roughness {rough:.2f} imply {regime.replace('_', ' ')}.",
    }


def _stage2_archetype(planet: Planet, atmosphere: Atmosphere, hydrosphere: Hydrosphere, geology: GeologyState) -> str:
    temp = atmosphere.estimated_mean_surface_temp_c
    ocean = hydrosphere.ocean_fraction_target
    if ocean > 0.74 and geology.volcanism > 1.0:
        return "active volcanic ocean world"
    if ocean > 0.70:
        return "warm ocean super-Earth" if planet.radius_earth > 1.2 else "ocean-rich temperate world"
    if ocean < 0.42 and geology.mountain_factor > 1.0:
        return "dry rugged highland world"
    if temp < 5.0:
        return "cold edge terrestrial world"
    if atmosphere.pressure_bar > 1.9:
        return "thick-atmosphere super-Earth"
    if geology.internal_heat < 0.45:
        return "quiet old continental world"
    return "Earth-like temperate world"


def _stage2_warnings(star: Star, planet: Planet, rotation: RotationState, atmosphere: Atmosphere, hydrosphere: Hydrosphere, geology: GeologyState) -> list[dict]:
    warnings: list[dict] = []
    def add(level: str, msg: str) -> None:
        warnings.append({"level": level, "message": msg})
    if planet.surface_gravity_g > 1.45:
        add("warning", f"Surface gravity is high at {planet.surface_gravity_g:.2f} g; human-comfort assumptions may be strained.")
    if atmosphere.pressure_bar < 0.65 or atmosphere.pressure_bar > 2.3:
        add("warning", f"Atmospheric pressure is outside the comfortable review band: {atmosphere.pressure_bar:.2f} bar.")
    if atmosphere.greenhouse_warming_k > 42.0:
        add("warning", "Greenhouse warming is doing heavy lifting to keep the world temperate.")
    if hydrosphere.ocean_fraction_target > 0.78:
        add("notice", "Ocean target is very high; terrain may need extra fragmentation/exposure to avoid a bland waterworld.")
    if rotation.axial_tilt_degrees > 45.0:
        add("warning", "Axial tilt is extreme; climate should show strong seasonality.")
    if rotation.rotation_period_hours > 48.0:
        add("notice", "Slow rotation may weaken Coriolis effects and broaden climate bands.")
    if geology.internal_heat < 0.35:
        add("notice", "Low internal heat implies subdued mountain building unless terrain overrides it.")
    if geology.volcanism > 1.25 and geology.internal_heat < 0.65:
        add("warning", "Volcanism is high relative to internal heat; review geology consistency.")
    if geology.erosion > 1.75 and geology.crater_density > 0.8:
        add("notice", "High erosion and high crater density pull in opposite directions; terrain should decide which dominates.")
    if not warnings:
        add("ok", "No major Stage 2 validation warnings recorded.")
    return warnings


def build_planet_physics_review(star: Star, planet: Planet, rotation: RotationState, atmosphere: Atmosphere, hydrosphere: Hydrosphere, geology: GeologyState) -> dict:
    """Build explanatory Stage 2 metadata for JSON state and Web UI review."""
    rotation_meta = _stage2_rotation_metadata(rotation, planet)
    atmosphere_meta = _stage2_atmosphere_metadata(star, planet, atmosphere)
    hydrosphere_meta = _stage2_hydrosphere_metadata(planet, hydrosphere)
    geology_meta = _stage2_geology_metadata(star, planet, geology)
    archetype = _stage2_archetype(planet, atmosphere, hydrosphere, geology)
    warnings = _stage2_warnings(star, planet, rotation, atmosphere, hydrosphere, geology)
    comfort = clamp(1.0 - abs(planet.surface_gravity_g - 1.0) * 0.35 - abs(atmosphere.pressure_bar - 1.0) * 0.10 - max(0.0, hydrosphere.ocean_fraction_target - 0.72) * 0.35, 0.0, 1.0)
    stability = clamp(0.55 + (0.18 if rotation_meta["axial_stability_class"] == "stable" else -0.12 if rotation_meta["axial_stability_class"] == "unstable" else 0.0) - abs(atmosphere.estimated_mean_surface_temp_c - 15.0) / 120.0, 0.0, 1.0)
    climate_mod = clamp(0.50 + atmosphere.pressure_bar * 0.12 + hydrosphere.ocean_fraction_target * 0.20 - abs(rotation.axial_tilt_degrees - 23.5) / 160.0, 0.0, 1.0)
    report = [
        f"This world reads as a {archetype}: {planet.radius_earth:.2f} R⊕, {planet.surface_gravity_g:.2f} g, {atmosphere.pressure_bar:.2f} bar, and an ocean target of {hydrosphere.ocean_fraction_target:.2f}.",
        rotation_meta["explanation"],
        atmosphere_meta["explanation"],
        hydrosphere_meta["volatile_inventory_note"],
        geology_meta["explanation"],
    ]
    downstream = {
        "terrain": [
            f"Tectonic regime should bias terrain toward {geology_meta['tectonic_regime'].replace('_', ' ')}.",
            f"Orogenic intensity is {geology_meta['orogenic_intensity']}; mountain chains should reflect this.",
            f"Crustal contrast/asymmetry is {geology_meta['crustal_contrast_strength']}; use this later to avoid overly uniform continents.",
            f"Shelf/deposition tendency is {geology_meta['shelf_deposition_tendency']}; coastal plains and deltas should be checked against it.",
        ],
        "climate": [
            f"Coriolis strength is {rotation_meta['coriolis_strength']}; circulation-cell width should respond downstream.",
            f"Seasonality is {rotation_meta['seasonality_class']} from axial tilt.",
            f"Pressure class is {atmosphere_meta['pressure_class']}; heat redistribution should respond downstream.",
            f"Climate risk is {atmosphere_meta['climate_risk']}.",
        ],
        "hydrology": [
            f"Ocean target {hydrosphere.ocean_fraction_target:.2f} implies target land fraction {1.0 - hydrosphere.ocean_fraction_target:.2f}.",
            f"Expected coastline complexity is {hydrosphere_meta['expected_coastline_complexity']}.",
            f"Erosion {geology.erosion:.2f} should affect river valleys, floodplains, and sediment deposition.",
        ],
    }
    return {
        "archetype": archetype,
        "scores": {
            "human_comfort": round(comfort, 3),
            "surface_stability": round(stability, 3),
            "climate_moderation": round(climate_mod, 3),
        },
        "report": report,
        "rotation": rotation_meta,
        "atmosphere": atmosphere_meta,
        "hydrosphere": hydrosphere_meta,
        "geology": geology_meta,
        "warnings": warnings,
        "downstream_implications": downstream,
        "stage1_context_used": planet.formation_context or {},
    }
