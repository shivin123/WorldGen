"""First-pass climate generation for the Main Planet.

This is a long-term climate approximation, not a weather model. It creates
cell-level annual mean temperature, seasonal temperature range, precipitation,
and a simplified Köppen-style climate classification for land cells.

The precipitation model explicitly includes:
- global circulation bands (ITCZ / subtropical subsidence / storm tracks)
- prevailing global wind belts by latitude
- local moisture advection from upwind ocean fetch
- local orographic uplift and leeward rain shadowing
"""

from __future__ import annotations

import math
from collections import deque

from worldgen.models.planet_profile import Atmosphere, ClimateMap, RotationState, TerrainMap
from worldgen.random_utils import clamp
from worldgen.physics.map_scale import map_scale_for_terrain, reference_cells_to_km, km_to_cells

OCEAN_CODE = "O"

# Climate distance constants are expressed in kilometers using the old
# 4096x2048 Earth-tuned cell constants as the compatibility baseline. This
# keeps climate behavior broadly similar at normal 4K Earth-scale maps while
# making high-resolution and large-radius runs physically scale-aware.
CONTINENTALITY_SATURATION_KM = reference_cells_to_km(80.0)
SEASONAL_CONTINENTALITY_KM = reference_cells_to_km(65.0)
COASTAL_MARITIME_EFOLD_KM = reference_cells_to_km(42.0)
COAST_MODERATION_EFOLD_KM = reference_cells_to_km(18.0)
CURRENT_COAST_EFOLD_KM = reference_cells_to_km(12.0)
MOISTURE_COAST_EFOLD_KM = reference_cells_to_km(105.0)
MOISTURE_FETCH_EFOLD_KM = reference_cells_to_km(44.0)
UPWIND_FETCH_SCAN_KM = reference_cells_to_km(44.0)
UPWIND_FETCH_WEIGHT_EFOLD_KM = reference_cells_to_km(18.0)
LOCAL_COAST_SCAN_KM = reference_cells_to_km(24.0)
LOCAL_WIND_SCAN_KM = reference_cells_to_km(14.0)


def _distance_cells_to_km(terrain: TerrainMap, cell_distance: float) -> float:
    return float(cell_distance) * map_scale_for_terrain(terrain).representative_km_per_cell


def generate_climate(
    rotation: RotationState,
    atmosphere: Atmosphere,
    terrain: TerrainMap,
    use_accelerated: bool = True,
    koppen_detail: str = "local4",
    climate_mode: str = "seasonal_v1",
) -> ClimateMap:
    """Generate climate using the requested backend.

    ``seasonal_v1`` is the stable climate-overhaul backend. ``seasonal_v2`` is
    the preserved structured atmospheric-circulation backend for review.
    ``seasonal_v3`` adds the first basin-aware ocean-circulation review layer.
    ``seasonal_v4`` refines basin routing, heat transport, upwelling, and coastal feedback.
    ``seasonal_v5`` preserves v4 ocean routing but adds component-based moisture/rainfall coupling.
    ``legacy`` preserves the previous long-term-average heuristic model for
    rollback/comparison, matching the terrain-mode workflow.
    """
    mode = (climate_mode or "seasonal_v1").strip().lower()
    if mode in {"legacy", "legacy_v1", "classic", "heuristic"}:
        climate = _generate_legacy_climate(rotation, atmosphere, terrain, use_accelerated=use_accelerated, koppen_detail=koppen_detail)
        climate.climate_mode = "legacy"
        climate.notes = ["Climate mode: legacy."] + list(climate.notes)
        return climate
    if mode in {"seasonal_v5", "circulation_v5", "structured_moisture_v1", "overhaul_v5"}:
        from worldgen.generators.climate_circulation_v2 import generate_seasonal_v5_climate

        return generate_seasonal_v5_climate(rotation, atmosphere, terrain, koppen_detail=koppen_detail)
    if mode in {"seasonal_v4", "circulation_v4", "structured_ocean_v2", "overhaul_v4"}:
        from worldgen.generators.climate_circulation_v2 import generate_seasonal_v4_climate

        return generate_seasonal_v4_climate(rotation, atmosphere, terrain, koppen_detail=koppen_detail)
    if mode in {"seasonal_v3", "circulation_v3", "structured_ocean_v1", "overhaul_v3"}:
        from worldgen.generators.climate_circulation_v2 import generate_seasonal_v3_climate

        return generate_seasonal_v3_climate(rotation, atmosphere, terrain, koppen_detail=koppen_detail)
    if mode in {"seasonal_v2", "circulation_v2", "structured_v1", "overhaul_v2"}:
        from worldgen.generators.climate_circulation_v2 import generate_seasonal_v2_climate

        return generate_seasonal_v2_climate(rotation, atmosphere, terrain, koppen_detail=koppen_detail)
    if mode in {"seasonal_v1", "seasonal", "overhaul_v1"}:
        from worldgen.generators.climate_seasonal import generate_seasonal_v1_climate

        return generate_seasonal_v1_climate(rotation, atmosphere, terrain, koppen_detail=koppen_detail)
    raise ValueError("Unknown climate mode %r. Valid modes: seasonal_v5, seasonal_v4, seasonal_v3, seasonal_v2, seasonal_v1, legacy." % climate_mode)


def _generate_legacy_climate(
    rotation: RotationState,
    atmosphere: Atmosphere,
    terrain: TerrainMap,
    use_accelerated: bool = True,
    koppen_detail: str = "local4",
) -> ClimateMap:
    """Generate climate with the pre-overhaul legacy backend.

    This keeps the old model available for comparison and rollback.
    """
    cells = terrain.width * terrain.height
    if (not use_accelerated) or cells <= 800_000:
        if not use_accelerated and cells > 2_600_000:
            print(f"[progress] Computing full-resolution legacy climate/Köppen at {terrain.width} x {terrain.height}; this can be slower but preserves local detail.", flush=True)
        return _generate_climate_core(rotation, atmosphere, terrain, koppen_detail=koppen_detail)

    aspect = terrain.width / max(1, terrain.height)
    feature_pixels = 180_000
    feature_w = min(terrain.width, max(512, int(round(math.sqrt(feature_pixels * aspect)))))
    feature_h = min(terrain.height, max(256, int(round(feature_w / aspect))))
    feature_w = max(512, (feature_w // 2) * 2)
    feature_h = max(256, (feature_h // 2) * 2)
    if feature_w == terrain.width and feature_h == terrain.height:
        return _generate_climate_core(rotation, atmosphere, terrain, koppen_detail=koppen_detail)

    print(f"[progress] Using accelerated legacy climate feature grid {feature_w} x {feature_h}, then upscaling to {terrain.width} x {terrain.height}.", flush=True)
    feature_terrain = _resample_terrain_for_climate(terrain, feature_w, feature_h)
    feature_climate = _generate_climate_core(rotation, atmosphere, feature_terrain, koppen_detail="local4")
    return _upsample_climate(feature_climate, terrain, koppen_detail=koppen_detail)


def _generate_climate_core(
    rotation: RotationState,
    atmosphere: Atmosphere,
    terrain: TerrainMap,
    koppen_detail: str = "local4",
) -> ClimateMap:
    height = terrain.height
    width = terrain.width
    if width * height > 180_000:
        return _generate_climate_core_vectorized(rotation, atmosphere, terrain, koppen_detail=koppen_detail)
    distance_to_ocean = _distance_to_ocean_cells(terrain.is_land)
    scale = map_scale_for_terrain(terrain)
    prevailing_winds = [_prevailing_wind_vector(90.0 - (row + 0.5) * 180.0 / height, rotation, atmosphere) for row in range(height)]
    real_earth = _is_real_earth_terrain(terrain)
    synthetic_earth = _is_synthetic_earth_terrain(terrain)

    mean_temp_grid: list[list[int]] = []
    warmest_grid: list[list[int]] = []
    coldest_grid: list[list[int]] = []
    precip_grid: list[list[int]] = []

    land_temp_sum = 0.0
    ocean_temp_sum = 0.0
    land_precip_sum = 0.0
    ocean_precip_sum = 0.0
    land_count = 0.0
    ocean_count = 0.0
    min_temp = 999.0
    max_temp = -999.0
    min_precip = 10**9
    max_precip = 0

    for row in range(height):
        lat = 90.0 - (row + 0.5) * 180.0 / height
        wind_dr, wind_dc = prevailing_winds[row]

        mean_row: list[int] = []
        warm_row: list[int] = []
        cold_row: list[int] = []
        precip_row: list[int] = []

        for col in range(width):
            land = terrain.is_land[row][col]
            elevation_m = terrain.elevation_m[row][col]
            coast_dist = distance_to_ocean[row][col]
            coast_dist_km = coast_dist * scale.representative_km_per_cell

            lon = -180.0 + (col + 0.5) * 360.0 / width
            current_temp_delta, current_precip_mult = _coastal_current_effect(terrain, row, col, lat, lon, coast_dist)
            annual_temp = _annual_temperature_c(atmosphere, lat, elevation_m, land, coast_dist_km) + current_temp_delta
            # Real Earth terrain calibration intentionally uses the same climate calculation
            # as procedural worlds. Do not apply Earth-specific geographic nudges here;
            # calibration diagnostics should reveal model errors rather than hide them.
            if synthetic_earth:
                # Synthetic Earth terrain often over-represents low-latitude land;
                # apply a small calibration offset so Earth mode remains a useful
                # broad temperature check without cooling generated exoplanets.
                annual_temp -= 2.8 if land else 1.2
            seasonal_amp = _seasonal_amplitude_c(rotation, lat, land, coast_dist_km)
            warmest = annual_temp + seasonal_amp
            coldest = annual_temp - seasonal_amp
            local_wind_dr, local_wind_dc = _local_wind_vector(terrain, row, col, wind_dr, wind_dc, coast_dist)
            precip = _annual_precipitation_mm(
                atmosphere=atmosphere,
                terrain=terrain,
                lat_degrees=lat,
                row=row,
                col=col,
                elevation_m=elevation_m,
                is_land=land,
                coast_dist_cells=coast_dist,
                wind_dr=local_wind_dr,
                wind_dc=local_wind_dc,
                rotation=rotation,
            )
            precip *= current_precip_mult
            # Real Earth terrain calibration intentionally uses the same precipitation model
            # as procedural worlds. Earth-specific regional precipitation boxes caused
            # straight-line artifacts and masked model calibration errors.

            mean_row.append(int(round(annual_temp * 10.0)))
            warm_row.append(int(round(warmest * 10.0)))
            cold_row.append(int(round(coldest * 10.0)))
            precip_int = int(round(precip))
            precip_row.append(precip_int)

            min_temp = min(min_temp, annual_temp)
            max_temp = max(max_temp, annual_temp)
            min_precip = min(min_precip, precip_int)
            max_precip = max(max_precip, precip_int)

            area_weight = max(0.01, math.cos(math.radians(lat)))
            if land:
                land_temp_sum += annual_temp * area_weight
                land_precip_sum += precip * area_weight
                land_count += area_weight
            else:
                ocean_temp_sum += annual_temp * area_weight
                ocean_precip_sum += precip * area_weight
                ocean_count += area_weight

        mean_temp_grid.append(mean_row)
        warmest_grid.append(warm_row)
        coldest_grid.append(cold_row)
        precip_grid.append(precip_row)

    # Köppen is a regional climate classification, not a micro-valley classifier.
    # Use smoothed climate fields for classification while preserving raw maps for
    # temperature/precipitation output. This removes latitude-straight and
    # one-cell valley stripes without hiding local weathering effects in the raw maps.
    # Smoothing radius must scale with map resolution. On a small grid, one cell
    # represents a large region and should be allowed to classify differently. On
    # high-resolution maps, a one-cell valley is microclimate noise for Köppen,
    # so classification uses a broader regional climate field.
    smoothing_passes = _koppen_smoothing_passes(width, height, koppen_detail)
    class_temp = _smooth_numeric_grid(mean_temp_grid, passes=smoothing_passes, scale=0.1)
    class_warm = _smooth_numeric_grid(warmest_grid, passes=smoothing_passes, scale=0.1)
    class_cold = _smooth_numeric_grid(coldest_grid, passes=smoothing_passes, scale=0.1)
    class_precip = _smooth_numeric_grid(precip_grid, passes=smoothing_passes, scale=1.0)

    koppen_grid: list[list[str]] = []
    koppen_summary: dict[str, int] = {}
    for row in range(height):
        koppen_row: list[str] = []
        for col in range(width):
            land = terrain.is_land[row][col]
            code = _koppen_code(
                annual_temp_c=class_temp[row][col],
                warmest_c=class_warm[row][col],
                coldest_c=class_cold[row][col],
                annual_precip_mm=class_precip[row][col],
                is_land=land,
            )
            koppen_row.append(code)
            if land:
                koppen_summary[code] = koppen_summary.get(code, 0) + 1
        koppen_grid.append(koppen_row)

    return ClimateMap(
        width=width,
        height=height,
        annual_mean_temp_c_x10=mean_temp_grid,
        warmest_month_temp_c_x10=warmest_grid,
        coldest_month_temp_c_x10=coldest_grid,
        annual_precip_mm=precip_grid,
        koppen_classification=koppen_grid,
        mean_land_temp_c=land_temp_sum / land_count if land_count else 0.0,
        mean_ocean_temp_c=ocean_temp_sum / ocean_count if ocean_count else 0.0,
        mean_land_precip_mm=land_precip_sum / land_count if land_count else 0.0,
        mean_ocean_precip_mm=ocean_precip_sum / ocean_count if ocean_count else 0.0,
        min_temp_c=min_temp,
        max_temp_c=max_temp,
        min_precip_mm=min_precip,
        max_precip_mm=max_precip,
        koppen_summary=dict(sorted(koppen_summary.items(), key=lambda item: item[0])),
        notes=[
            "Climate is long-term average climate, not weather.",
            "Precipitation includes latitude bands, prevailing global winds, upwind ocean fetch, and local orographic effects.",
            f"Distance-to-coast and moisture fetch use physical scaling: {scale.representative_km_per_cell:.2f} km/cell on a {scale.planet_radius_earth:.2f} R_earth planet.",
            "Köppen classes use map-scale-aware smoothed regional climate fields to avoid high-resolution micro-valley stripes while preserving small-map regional cells.",
            "Ocean cells are marked O; Köppen classes are intended for land cells.",
        ],
    )



def _generate_climate_core_vectorized(
    rotation: RotationState,
    atmosphere: Atmosphere,
    terrain: TerrainMap,
    koppen_detail: str = "local4",
) -> ClimateMap:
    """Vectorized full-resolution climate path.

    This replaces the old per-cell Python loop for large requested maps. It
    still works at the requested resolution; it does not use a lower-resolution
    feature grid or upscale the climate result.
    """
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("NumPy is required for full-resolution vectorized climate. Install it with: pip install numpy") from exc

    height = terrain.height
    width = terrain.width
    real_earth = _is_real_earth_terrain(terrain)
    synthetic_earth = _is_synthetic_earth_terrain(terrain)

    land = np.asarray(terrain.is_land, dtype=bool)
    elev = np.asarray(terrain.elevation_m, dtype=np.float32)
    dist = np.asarray(_distance_to_ocean_cells(terrain.is_land), dtype=np.float32)
    scale = map_scale_for_terrain(terrain)
    dist_km = dist * np.float32(scale.representative_km_per_cell)
    lats = np.linspace(90.0 - 90.0 / height, -90.0 + 90.0 / height, height, dtype=np.float32)
    lons = np.linspace(-180.0 + 180.0 / width, 180.0 - 180.0 / width, width, dtype=np.float32)
    lat_grid = lats[:, None]
    lon_grid = lons[None, :]
    abs_lat = np.abs(lat_grid)
    lat_rad = np.radians(lat_grid)

    # Temperature: same core physics as the scalar path, vectorized.
    latitude_effect = 45.0 * (np.cos(lat_rad) - 0.63)
    lapse = np.maximum(elev, 0.0) * 0.0063
    continentality = np.where(land, np.clip(dist_km / CONTINENTALITY_SATURATION_KM, 0.0, 1.0), 0.0)
    inland_warming = 1.8 * continentality * np.maximum(0.0, 1.0 - abs_lat / 70.0)
    ocean_buffer = np.where((~land) & (abs_lat < 50.0), -1.8, 0.0)
    coast_moderation = np.where(land, -1.6 * np.exp(-dist_km / COAST_MODERATION_EFOLD_KM), 0.0)
    annual_temp = atmosphere.estimated_mean_surface_temp_c + latitude_effect - lapse + inland_warming + ocean_buffer + coast_moderation

    # Approximate local current effects at full resolution without per-cell ocean scans.
    west_ocean = (~np.roll(land, 9, axis=1)).astype(np.float32)
    east_ocean = (~np.roll(land, -9, axis=1)).astype(np.float32)
    north_ocean = np.vstack([(~land[0:1, :]), (~land[:-1, :])]).astype(np.float32)
    south_ocean = np.vstack([(~land[1:, :]), (~land[-1:, :])]).astype(np.float32)
    coast_strength = np.where(land, np.exp(-dist_km / CURRENT_COAST_EFOLD_KM), 0.0)
    warm_east_coast = east_ocean * np.exp(-(((abs_lat - 35.0) / 22.0) ** 2))
    cold_west_coast = west_ocean * np.exp(-(((abs_lat - 27.0) / 16.0) ** 2))
    meridional_warm = south_ocean * (lat_grid > 0) + north_ocean * (lat_grid < 0)
    meridional_cold = north_ocean * (lat_grid > 0) + south_ocean * (lat_grid < 0)
    current_temp_delta = coast_strength * (1.7 * warm_east_coast - 1.55 * cold_west_coast + 0.45 * meridional_warm - 0.55 * meridional_cold)
    current_precip_mult = 1.0 + coast_strength * (0.20 * warm_east_coast - 0.20 * cold_west_coast + 0.08 * meridional_warm - 0.05 * meridional_cold)
    annual_temp = annual_temp + current_temp_delta

    # Real Earth terrain calibration intentionally uses the same climate calculation
    # as procedural worlds. Do not apply Earth-specific geographic nudges here;
    # calibration diagnostics should reveal model errors rather than hide them.
    if synthetic_earth:
        annual_temp = annual_temp - np.where(land, 2.8, 1.2)

    tilt_factor = rotation.axial_tilt_degrees / 23.5
    land_factor = np.where(land, 1.0, 0.42)
    seasonal_cont = np.where(land, np.clip(dist_km / SEASONAL_CONTINENTALITY_KM, 0.0, 1.0), 0.0)
    seasonal_amp = (abs_lat / 90.0) ** 1.20 * (7.0 + 12.0 * land_factor + 14.0 * seasonal_cont) * tilt_factor
    seasonal_amp = np.clip(seasonal_amp, np.where(land, 1.0, 0.4), 38.0)
    warmest = annual_temp + seasonal_amp
    coldest = annual_temp - seasonal_amp

    # Precipitation: rotation-aware global bands + vectorized moisture access + orographic effects.
    coriolis = clamp(24.0 / max(8.0, rotation.rotation_period_hours), 0.42, 2.2)
    hadley_edge = clamp(30.0 / math.sqrt(coriolis), 18.0, 48.0)
    itcz_lat = clamp(rotation.axial_tilt_degrees * 0.18, 1.5, 8.0)
    storm_lat = clamp(hadley_edge + 22.0 + 4.5 * (coriolis - 1.0), 40.0, 68.0)
    subtropical_lat = clamp(hadley_edge * 0.92, 18.0, 42.0)
    itcz_wet = 1080.0 * np.exp(-(((abs_lat - itcz_lat) / max(8.0, hadley_edge * 0.48)) ** 2))
    monsoon_wet = 300.0 * np.exp(-(((abs_lat - hadley_edge * 0.55) / 10.0) ** 2))
    subtropical_dry = (135.0 + 55.0 * coriolis) * np.exp(-(((abs_lat - subtropical_lat) / 9.5) ** 2))
    storm_track_wet = (505.0 + 60.0 * coriolis) * np.exp(-(((abs_lat - storm_lat) / 15.0) ** 2))
    polar_dry = 210.0 * np.exp(-(((abs_lat - 78.0) / 11.0) ** 2))
    circulation_base = 585.0 + itcz_wet * 0.86 + monsoon_wet * 0.66 + storm_track_wet * 0.80 - subtropical_dry * 0.88 - polar_dry

    coast_factor = np.exp(-dist_km / MOISTURE_COAST_EFOLD_KM)
    fetch = np.exp(-dist_km / MOISTURE_FETCH_EFOLD_KM)
    # Orographic proxy: use slope magnitude and directional gradient relative to prevailing belt.
    gy, gx = np.gradient(np.maximum(elev, 0.0))
    # Fast rotators have stronger zonal winds and stronger rain shadows.
    zonal_strength = clamp(0.72 + 0.38 * coriolis, 0.55, 1.55) / clamp(0.85 + 0.10 * atmosphere.pressure_bar, 0.8, 1.25)
    trades = abs_lat < hadley_edge
    westerlies = (abs_lat >= hadley_edge) & (abs_lat < hadley_edge + 30.0)
    wind_dc = np.where(trades, -zonal_strength, np.where(westerlies, zonal_strength, -0.72 * zonal_strength))
    wind_dr = np.where(trades, np.sign(lat_grid) * 0.38, np.where(westerlies, -np.sign(lat_grid) * 0.30, np.sign(lat_grid) * 0.34))
    directional_slope = -(gx * wind_dc + gy * wind_dr)
    slope_scale = np.percentile(np.abs(directional_slope[land]), 90) if np.any(land) else 1.0
    slope_scale = max(float(slope_scale), 1.0)
    uplift_factor = 1.0 + 0.24 * np.clip(directional_slope / slope_scale, 0.0, 1.0) * (0.45 + 0.55 * fetch)
    shadow_factor = 1.0 - 0.13 * np.clip(-directional_slope / slope_scale, 0.0, 1.0) * (0.25 + 0.45 * (1.0 - coast_factor))
    shadow_factor = np.clip(shadow_factor, 0.82, 1.12)
    convective_bonus = 1.0 + 0.18 * np.exp(-(((abs_lat - 12.0) / 15.0) ** 2)) + 0.08 * np.exp(-(((abs_lat - 42.0) / 18.0) ** 2))
    vapor_factor = 0.76 + 0.34 * atmosphere.water_vapor_factor
    pressure_factor = 0.88 + 0.14 * atmosphere.pressure_bar
    advection_factor = 0.58 + 0.28 * coast_factor + 0.62 * fetch
    land_precip = circulation_base * advection_factor * uplift_factor * shadow_factor * convective_bonus
    ocean_convergence = 1.02 + 0.18 * np.exp(-(((abs_lat - 6.0) / 11.0) ** 2)) + 0.12 * np.exp(-(((abs_lat - 52.0) / 13.0) ** 2)) - 0.12 * np.exp(-(((abs_lat - 28.0) / 8.0) ** 2))
    ocean_precip = circulation_base * 1.12 * ocean_convergence
    precip = np.where(land, land_precip, ocean_precip) * vapor_factor * pressure_factor * current_precip_mult
    # Real Earth terrain calibration intentionally uses the same precipitation model
    # as procedural worlds. Earth-specific regional precipitation boxes caused
    # straight-line artifacts and masked model calibration errors.
    precip = np.clip(precip, 25.0, 5200.0)

    mean_temp_grid = np.rint(annual_temp * 10.0).astype(np.int16)
    warmest_grid = np.rint(warmest * 10.0).astype(np.int16)
    coldest_grid = np.rint(coldest * 10.0).astype(np.int16)
    precip_grid = np.rint(precip).astype(np.int32)

    passes = _koppen_smoothing_passes(width, height, koppen_detail)
    class_temp = _smooth_numeric_grid_np(mean_temp_grid.astype(np.float32) * 0.1, passes=passes)
    class_warm = _smooth_numeric_grid_np(warmest_grid.astype(np.float32) * 0.1, passes=passes)
    class_cold = _smooth_numeric_grid_np(coldest_grid.astype(np.float32) * 0.1, passes=passes)
    class_precip = _smooth_numeric_grid_np(precip_grid.astype(np.float32), passes=passes)

    code_arr = _koppen_code_array(class_temp, class_warm, class_cold, class_precip, land)
    code_labels = ["O", "Af", "Am", "Aw", "BWh", "BWk", "BSh", "BSk", "Cfa", "Cfb", "Csa", "Csb", "Dfa", "Dfb", "Dfc", "ET", "EF"]
    koppen_grid = [[code_labels[int(v)] for v in row] for row in code_arr]
    koppen_summary: dict[str, int] = {}
    unique, counts = np.unique(code_arr[land], return_counts=True)
    for value, count in zip(unique, counts):
        label = code_labels[int(value)]
        if label != "O":
            koppen_summary[label] = int(count)

    area_weight = np.maximum(0.01, np.cos(np.radians(lat_grid))).astype(np.float64)
    land_weight = np.where(land, area_weight, 0.0)
    ocean_weight = np.where(~land, area_weight, 0.0)
    land_count = float(land_weight.sum())
    ocean_count = float(ocean_weight.sum())

    return ClimateMap(
        width=width,
        height=height,
        annual_mean_temp_c_x10=mean_temp_grid.astype(int).tolist(),
        warmest_month_temp_c_x10=warmest_grid.astype(int).tolist(),
        coldest_month_temp_c_x10=coldest_grid.astype(int).tolist(),
        annual_precip_mm=precip_grid.astype(int).tolist(),
        koppen_classification=koppen_grid,
        mean_land_temp_c=float((annual_temp * land_weight).sum() / land_count) if land_count else 0.0,
        mean_ocean_temp_c=float((annual_temp * ocean_weight).sum() / ocean_count) if ocean_count else 0.0,
        mean_land_precip_mm=float((precip * land_weight).sum() / land_count) if land_count else 0.0,
        mean_ocean_precip_mm=float((precip * ocean_weight).sum() / ocean_count) if ocean_count else 0.0,
        min_temp_c=float(np.min(annual_temp)),
        max_temp_c=float(np.max(annual_temp)),
        min_precip_mm=int(np.min(precip_grid)),
        max_precip_mm=int(np.max(precip_grid)),
        koppen_summary=dict(sorted(koppen_summary.items(), key=lambda item: item[0])),
        notes=[
            "Climate is long-term average climate, not weather.",
            "Large-map climate uses vectorized full-resolution fields rather than lower-resolution upscaling.",
            f"Distance-to-coast and moisture fetch use physical scaling: {scale.representative_km_per_cell:.2f} km/cell on a {scale.planet_radius_earth:.2f} R_earth planet.",
            "Wind belts respond to rotation period, Coriolis strength, atmospheric pressure, and axial tilt.",
            "Köppen classes use map-scale-aware smoothed regional climate fields.",
        ],
    )



def _smooth_numeric_grid_np(arr, passes: int):
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("NumPy is required for vectorized climate smoothing.") from exc
    current = arr.astype(np.float32, copy=False)
    for _ in range(max(0, passes)):
        north = np.vstack([current[0:1, :], current[:-1, :]])
        south = np.vstack([current[1:, :], current[-1:, :]])
        west = np.roll(current, 1, axis=1)
        east = np.roll(current, -1, axis=1)
        current = (current * 4.0 + north + south + west + east) / 8.0
    return current

def _koppen_code_array(annual_temp_c, warmest_c, coldest_c, annual_precip_mm, is_land):
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("NumPy is required for vectorized Köppen classification.") from exc
    code = np.zeros(is_land.shape, dtype=np.uint8)  # O
    land = is_land
    code[land & (warmest_c < 0.0)] = 16  # EF
    tundra = land & (code == 0) & (warmest_c < 10.0)
    code[tundra] = 15  # ET
    remaining = land & (code == 0)
    aridity_threshold = np.maximum(90.0, 20.0 * np.maximum(annual_temp_c, 0.0) + 260.0)
    desert = remaining & (annual_precip_mm < aridity_threshold)
    code[desert & (annual_temp_c >= 18.0)] = 4
    code[desert & (annual_temp_c < 18.0)] = 5
    steppe = remaining & (code == 0) & (annual_precip_mm < aridity_threshold * 1.85)
    code[steppe & (annual_temp_c >= 18.0)] = 6
    code[steppe & (annual_temp_c < 18.0)] = 7
    remaining = land & (code == 0)
    tropical = remaining & (coldest_c >= 18.0)
    code[tropical & (annual_precip_mm >= 2200.0)] = 1
    code[tropical & (annual_precip_mm >= 1250.0) & (annual_precip_mm < 2200.0)] = 2
    code[tropical & (annual_precip_mm < 1250.0)] = 3
    remaining = land & (code == 0)
    temperate = remaining & (coldest_c > 0.0)
    # Dry-summer inferred from precipitation and hot/warm summers, not latitude.
    dry_summer = temperate & (annual_precip_mm < 900.0) & (warmest_c >= 16.0)
    code[dry_summer & (warmest_c >= 22.0)] = 10
    code[dry_summer & (warmest_c < 22.0)] = 11
    code[temperate & (~dry_summer) & (warmest_c >= 22.0)] = 8
    code[temperate & (~dry_summer) & (warmest_c < 22.0)] = 9
    remaining = land & (code == 0)
    code[remaining & (warmest_c >= 22.0)] = 12
    code[remaining & (warmest_c >= 15.0) & (warmest_c < 22.0)] = 13
    code[remaining & (warmest_c < 15.0)] = 14
    return code


def _earth_temperature_adjustment_array(temp_c, lat, lon, land, elevation_m):
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("NumPy is required for Earth climate calibration arrays.") from exc
    out = temp_c.copy()

    # lat is normally shaped (height, 1) and lon is normally shaped (1, width)
    # so most masks broadcast naturally to (height, width).  The Antarctic
    # cooling gradient must also be broadcast before boolean indexing; indexing
    # the unbroadcast (height, 1) array with a (height, width) mask crashes at
    # full map widths.
    lat_full = np.broadcast_to(lat, out.shape)
    lon_full = np.broadcast_to(lon, out.shape)

    antarctica = land & (lat_full < -60.0)
    antarctic_gradient = np.clip((-lat_full - 60.0) / 22.0, 0.0, 1.0)
    out[antarctica] -= 15.0 + 10.0 * antarctic_gradient[antarctica]

    greenland = land & (lat_full >= 58.0) & (lat_full <= 84.0) & (lon_full >= -75.0) & (lon_full <= -10.0)
    out[greenland] -= 8.0

    tibet = land & (lat_full >= 25.0) & (lat_full <= 38.0) & (lon_full >= 70.0) & (lon_full <= 105.0) & (elevation_m > 2200)
    out[tibet] -= 3.5
    return out


def _earth_precipitation_adjustment_array(precip, lat, lon, land):
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("NumPy is required for Earth precipitation calibration arrays.") from exc
    mult = np.ones(precip.shape, dtype=np.float32)
    def box(lat_min, lat_max, lon_min, lon_max, factor):
        nonlocal mult
        mask = land & (lat >= lat_min) & (lat <= lat_max) & (lon >= lon_min) & (lon <= lon_max)
        mult[mask] *= factor
    box(-90, -60, -180, 180, 0.20)
    box(58, 84, -75, -10, 0.35)
    box(24, 48, -97, -62, 1.45)
    box(41, 57, -132, -116, 1.28)
    box(25, 41, -126, -105, 0.62)
    box(12, 31, 38, 62, 0.32)
    box(21, 31, 67, 78, 0.42)
    box(10, 22, 73, 79, 0.70)
    box(20, 45, 105, 123, 1.50)
    box(31, 46, 123, 132, 1.42)
    box(30, 47, 129, 147, 1.62)
    box(7, 24, 78, 105, 1.22)
    box(5, 24, 88, 112, 1.25)
    box(-30, -15, -76, -68, 0.30)
    box(-31, -16, 11, 18, 0.38)
    return np.clip(precip * mult, 15.0, 6000.0)

def _resample_terrain_for_climate(terrain: TerrainMap, width: int, height: int) -> TerrainMap:
    try:
        import numpy as np
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("NumPy and Pillow are required for accelerated climate generation. Install with: pip install -r requirements.txt") from exc
    elev = np.asarray(terrain.elevation_m, dtype=np.float32)
    land = np.asarray(terrain.is_land, dtype=np.uint8) * 255
    elev_small = np.asarray(Image.fromarray(elev, mode="F").resize((width, height), Image.Resampling.BILINEAR), dtype=np.float32)
    land_small = np.asarray(Image.fromarray(land, mode="L").resize((width, height), Image.Resampling.BILINEAR), dtype=np.uint8) >= 128
    elevation = np.where(land_small, np.maximum(elev_small, 1), np.minimum(elev_small, -1)).round().astype(np.int32)
    land_values = elevation[land_small]
    ocean_values = elevation[~land_small]
    ocean_fraction = 1.0 - int(land_small.sum()) / float(width * height)
    return TerrainMap(
        width=width,
        height=height,
        elevation_m=elevation.tolist(),
        is_land=land_small.tolist(),
        min_elevation_m=int(elevation.min()),
        max_elevation_m=int(elevation.max()),
        mean_land_elevation_m=float(land_values.mean()) if land_values.size else 0.0,
        mean_ocean_depth_m=float(ocean_values.mean()) if ocean_values.size else 0.0,
        ocean_fraction=ocean_fraction,
        land_fraction=1.0 - ocean_fraction,
        source=f"{terrain.source}; climate feature grid {width}x{height}",
        planet_radius_earth=float(getattr(terrain, "planet_radius_earth", 1.0) or 1.0),
    )


def _resize_numeric_grid(grid: list[list[int]], width: int, height: int) -> list[list[int]]:
    try:
        import numpy as np
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("NumPy and Pillow are required for accelerated climate generation. Install with: pip install -r requirements.txt") from exc
    arr = np.asarray(grid, dtype=np.float32)
    out = np.asarray(Image.fromarray(arr, mode="F").resize((width, height), Image.Resampling.BILINEAR), dtype=np.float32)
    return np.rint(out).astype(np.int32).tolist()


def _resize_string_grid_nearest(grid: list[list[str]], width: int, height: int) -> list[list[str]]:
    try:
        import numpy as np
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("NumPy and Pillow are required for accelerated climate generation. Install with: pip install -r requirements.txt") from exc
    codes = sorted({value for row in grid for value in row})
    code_to_id = {code: i for i, code in enumerate(codes)}
    id_to_code = {i: code for code, i in code_to_id.items()}
    arr = np.asarray([[code_to_id[value] for value in row] for row in grid], dtype=np.uint8)
    out = np.asarray(Image.fromarray(arr, mode="L").resize((width, height), Image.Resampling.NEAREST), dtype=np.uint8)
    return [[id_to_code[int(value)] for value in row] for row in out]


def _upsample_climate(feature: ClimateMap, terrain: TerrainMap, koppen_detail: str = "local4") -> ClimateMap:
    width = terrain.width
    height = terrain.height
    mean_temp = _resize_numeric_grid(feature.annual_mean_temp_c_x10, width, height)
    warm = _resize_numeric_grid(feature.warmest_month_temp_c_x10, width, height)
    cold = _resize_numeric_grid(feature.coldest_month_temp_c_x10, width, height)
    precip = _resize_numeric_grid(feature.annual_precip_mm, width, height)
    # Do not upscale coarse Köppen codes directly; that removes local detail and
    # creates large artificial blocks. Upscale the numeric climate fields and
    # reclassify at the target resolution using the requested local/regional
    # smoothing scale.
    smoothing_passes = _koppen_smoothing_passes(width, height, koppen_detail)
    class_temp = _smooth_numeric_grid(mean_temp, passes=smoothing_passes, scale=0.1)
    class_warm = _smooth_numeric_grid(warm, passes=smoothing_passes, scale=0.1)
    class_cold = _smooth_numeric_grid(cold, passes=smoothing_passes, scale=0.1)
    class_precip = _smooth_numeric_grid(precip, passes=smoothing_passes, scale=1.0)
    koppen = [[OCEAN_CODE for _ in range(width)] for _ in range(height)]

    land_temp_sum = ocean_temp_sum = land_precip_sum = ocean_precip_sum = 0.0
    land_count = ocean_count = 0.0
    min_temp = 999.0
    max_temp = -999.0
    min_precip = 10**9
    max_precip = 0
    koppen_summary: dict[str, int] = {}
    for r in range(height):
        lat = 90.0 - (r + 0.5) * 180.0 / height
        area_weight = max(0.01, math.cos(math.radians(lat)))
        for c in range(width):
            t = mean_temp[r][c] / 10.0
            p = int(precip[r][c])
            min_temp = min(min_temp, t)
            max_temp = max(max_temp, t)
            min_precip = min(min_precip, p)
            max_precip = max(max_precip, p)
            if terrain.is_land[r][c]:
                land_temp_sum += t * area_weight
                land_precip_sum += p * area_weight
                land_count += area_weight
                code = _koppen_code(class_temp[r][c], class_warm[r][c], class_cold[r][c], class_precip[r][c], True)
                koppen[r][c] = code
                koppen_summary[code] = koppen_summary.get(code, 0) + 1
            else:
                ocean_temp_sum += t * area_weight
                ocean_precip_sum += p * area_weight
                ocean_count += area_weight
                koppen[r][c] = OCEAN_CODE
    notes = list(feature.notes)
    notes.append(f"Climate computed on accelerated feature grid {feature.width}x{feature.height}, numeric fields upscaled, and Köppen reclassified at {width}x{height} using '{koppen_detail}' detail.")
    return ClimateMap(
        width=width,
        height=height,
        annual_mean_temp_c_x10=mean_temp,
        warmest_month_temp_c_x10=warm,
        coldest_month_temp_c_x10=cold,
        annual_precip_mm=precip,
        koppen_classification=koppen,
        mean_land_temp_c=land_temp_sum / land_count if land_count else 0.0,
        mean_ocean_temp_c=ocean_temp_sum / ocean_count if ocean_count else 0.0,
        mean_land_precip_mm=land_precip_sum / land_count if land_count else 0.0,
        mean_ocean_precip_mm=ocean_precip_sum / ocean_count if ocean_count else 0.0,
        min_temp_c=min_temp,
        max_temp_c=max_temp,
        min_precip_mm=min_precip,
        max_precip_mm=max_precip,
        koppen_summary=dict(sorted(koppen_summary.items(), key=lambda item: item[0])),
        notes=notes,
    )

def _smooth_numeric_grid(grid: list[list[int]], passes: int, scale: float) -> list[list[float]]:
    try:
        import numpy as np
    except ImportError:
        # Fallback: no smoothing if NumPy is unavailable, though requirements include it.
        return [[value * scale for value in row] for row in grid]
    arr = np.asarray(grid, dtype=np.float32) * scale
    for _ in range(max(0, passes)):
        north = np.vstack([arr[0:1, :], arr[:-1, :]])
        south = np.vstack([arr[1:, :], arr[-1:, :]])
        west = np.roll(arr, 1, axis=1)
        east = np.roll(arr, -1, axis=1)
        arr = (arr * 4.0 + north + south + west + east) / 8.0
    return arr.tolist()


def _koppen_smoothing_passes(width: int, height: int, detail: str) -> int:
    """Map-size-aware Köppen smoothing.

    Köppen is a climate-region classifier, but previous regional smoothing hid
    too much local variation at high resolution. These modes let the user trade
    detail against broad regional stability:
      cell     = raw per-cell climate
      local4   = ~2x2/4-cell neighborhood feel
      local9   = ~3x3/9-cell neighborhood feel
      regional = older broad smoothing for very clean maps
    """
    detail = (detail or "local4").lower()
    if detail == "cell":
        return 0
    if detail == "local4":
        return 1
    if detail == "local9":
        return 2
    # Regional smoothing still scales with map size, but is less aggressive than
    # the old high-res setting.
    return max(2, min(7, round(min(width, height) / 768.0 * 2.0)))



def _coastal_current_effect(terrain: TerrainMap, row: int, col: int, lat: float, lon: float, coast_dist_cells: int) -> tuple[float, float]:
    """Approximate ocean-current influence on nearby coasts.

    This is still not a full ocean model, but it now samples local land/ocean
    geometry in four directions. Warm poleward boundary currents moisten/warm
    many east coasts, cold equatorward currents cool/dry many subtropical west
    coasts, and high-latitude windward coasts get an added maritime wetting
    effect.
    """
    abs_lat = abs(lat)
    coast_dist_km = _distance_cells_to_km(terrain, coast_dist_cells)
    if coast_dist_km > max(reference_cells_to_km(10.0), reference_cells_to_km(98.0)):
        return 0.0, 1.0

    width = terrain.width
    height = terrain.height
    max_scan = km_to_cells(terrain, LOCAL_COAST_SCAN_KM, minimum=4, maximum=80)
    west_ocean = east_ocean = north_ocean = south_ocean = 0.0
    for step in range(1, max_scan + 1):
        weight = math.exp(-(step - 1) / 5.0)
        if not terrain.is_land[row][(col - step) % width]:
            west_ocean += weight
        if not terrain.is_land[row][(col + step) % width]:
            east_ocean += weight
        nr = row - step
        sr = row + step
        if nr >= 0 and not terrain.is_land[nr][col]:
            north_ocean += weight
        if sr < height and not terrain.is_land[sr][col]:
            south_ocean += weight

    total = west_ocean + east_ocean + north_ocean + south_ocean
    if total <= 0.0:
        return 0.0, 1.0
    west = west_ocean / total
    east = east_ocean / total
    north = north_ocean / total
    south = south_ocean / total
    coast_strength = math.exp(-coast_dist_km / CURRENT_COAST_EFOLD_KM)

    # East coasts tend to receive warm western-boundary currents; west coasts
    # in the subtropics tend to receive cold eastern-boundary currents.
    warm_east_coast = east * math.exp(-(((abs_lat - 34.0) / 22.0) ** 2))
    cold_west_coast = west * math.exp(-(((abs_lat - 25.0) / 15.5) ** 2))

    # Meridional ocean connection: equatorward water is warmer, poleward water
    # is cooler. It modulates coast climates where N/S ocean exposure dominates.
    if lat >= 0:
        warm_from_equator = south
        cold_from_pole = north
    else:
        warm_from_equator = north
        cold_from_pole = south
    meridional_warm = warm_from_equator * math.exp(-(((abs_lat - 28.0) / 24.0) ** 2))
    meridional_cold = cold_from_pole * math.exp(-(((abs_lat - 50.0) / 22.0) ** 2))

    high_lat_windward = (west + 0.35 * east) * math.exp(-(((abs_lat - 52.0) / 16.0) ** 2))

    temp_delta = coast_strength * (2.0 * warm_east_coast - 1.8 * cold_west_coast + 0.7 * meridional_warm - 0.8 * meridional_cold + 0.45 * high_lat_windward)
    precip_mult = 1.0 + coast_strength * (0.26 * warm_east_coast - 0.30 * cold_west_coast + 0.12 * meridional_warm - 0.10 * meridional_cold + 0.30 * high_lat_windward)
    return temp_delta, clamp(precip_mult, 0.66, 1.42)


def _local_wind_vector(terrain: TerrainMap, row: int, col: int, wind_dr: float, wind_dc: float, coast_dist_cells: int) -> tuple[float, float]:
    """Blend global winds with local coast/terrain steering.

    Close to coasts, winds pick up an onshore component from nearby ocean. Over
    steep terrain, a small component follows the easiest local valley direction.
    The adjustment is intentionally weak so global circulation still dominates.
    """
    coast_dist_km = _distance_cells_to_km(terrain, coast_dist_cells)
    if coast_dist_km > max(reference_cells_to_km(10.0), reference_cells_to_km(51.0)):
        return wind_dr, wind_dc
    h = terrain.height
    w = terrain.width
    ocean_dr = 0.0
    ocean_dc = 0.0
    max_scan = km_to_cells(terrain, LOCAL_WIND_SCAN_KM, minimum=3, maximum=60)
    for step in range(1, max_scan + 1):
        weight = math.exp(-(step - 1) / 4.0)
        for dr, dc in ((-step, 0), (step, 0), (0, -step), (0, step)):
            rr = row + dr
            if rr < 0 or rr >= h:
                continue
            cc = (col + dc) % w
            if not terrain.is_land[rr][cc]:
                # vector from ocean sample toward this cell
                ocean_dr += (-dr / max(1, step)) * weight
                ocean_dc += (-dc / max(1, step)) * weight
    strength = math.hypot(ocean_dr, ocean_dc)
    if strength <= 0.0:
        return wind_dr, wind_dc
    ocean_dr /= strength
    ocean_dc /= strength
    coast_weight = math.exp(-coast_dist_km / CURRENT_COAST_EFOLD_KM)
    return wind_dr * (1.0 - 0.24 * coast_weight) + ocean_dr * (0.24 * coast_weight), wind_dc * (1.0 - 0.24 * coast_weight) + ocean_dc * (0.24 * coast_weight)


def _annual_temperature_c(
    atmosphere: Atmosphere,
    lat_degrees: float,
    elevation_m: int,
    is_land: bool,
    coast_dist_km: float,
) -> float:
    base = atmosphere.estimated_mean_surface_temp_c
    abs_lat = abs(lat_degrees)
    lat_rad = math.radians(lat_degrees)

    latitude_effect = 34.0 * (math.cos(lat_rad) - 0.76)
    lapse = max(0, elevation_m) * 0.0063
    continentality = clamp(coast_dist_km / CONTINENTALITY_SATURATION_KM, 0.0, 1.0) if is_land else 0.0
    inland_warming = 1.8 * continentality * max(0.0, 1.0 - abs_lat / 70.0)
    ocean_buffer = -1.8 if not is_land and abs_lat < 50.0 else 0.0
    # Oceans moderate nearby land temperatures. Coastal tropics are slightly
    # cooler than continental interiors, while high-latitude coasts are warmer
    # than equally high-latitude inland areas. This is an annual-mean effect;
    # seasonal moderation is handled separately in _seasonal_amplitude_c.
    maritime = math.exp(-coast_dist_km / COASTAL_MARITIME_EFOLD_KM) if is_land else 1.0
    coastal_moderation = maritime * (-1.1 * max(0.0, 1.0 - abs_lat / 35.0) + 2.4 * clamp((abs_lat - 38.0) / 34.0, 0.0, 1.0))

    return base + latitude_effect - lapse + inland_warming + ocean_buffer + coastal_moderation



def _seasonal_amplitude_c(
    rotation: RotationState,
    lat_degrees: float,
    is_land: bool,
    coast_dist_km: float,
) -> float:
    abs_lat = abs(lat_degrees)
    tilt_factor = rotation.axial_tilt_degrees / 23.5
    land_factor = 1.0 if is_land else 0.42
    continentality = clamp(coast_dist_km / SEASONAL_CONTINENTALITY_KM, 0.0, 1.0) if is_land else 0.0
    amplitude = (abs_lat / 90.0) ** 1.20 * (7.0 + 12.0 * land_factor + 14.0 * continentality) * tilt_factor
    return clamp(amplitude, 1.0 if is_land else 0.4, 38.0)



def _annual_precipitation_mm(
    atmosphere: Atmosphere,
    terrain: TerrainMap,
    lat_degrees: float,
    row: int,
    col: int,
    elevation_m: int,
    is_land: bool,
    coast_dist_cells: int,
    wind_dr: float,
    wind_dc: float,
    rotation: RotationState,
) -> float:
    abs_lat = abs(lat_degrees)

    # Global circulation bands. Rotation controls Coriolis strength and
    # atmospheric cell width: fast rotators have narrower Hadley cells and
    # stronger westerlies/trades; slow rotators have broader tropical cells.
    coriolis = clamp(24.0 / max(8.0, rotation.rotation_period_hours), 0.42, 2.2)
    hadley_edge = clamp(30.0 / math.sqrt(coriolis), 18.0, 48.0)
    itcz_lat = clamp(rotation.axial_tilt_degrees * 0.18, 1.5, 8.0)
    storm_lat = clamp(hadley_edge + 22.0 + 4.5 * (coriolis - 1.0), 40.0, 68.0)
    subtropical_lat = clamp(hadley_edge * 0.92, 18.0, 42.0)

    itcz_wet = 1080.0 * math.exp(-(((abs_lat - itcz_lat) / max(8.0, hadley_edge * 0.48)) ** 2))
    monsoon_wet = 300.0 * math.exp(-(((abs_lat - hadley_edge * 0.55) / 10.0) ** 2))
    subtropical_dry = (135.0 + 55.0 * coriolis) * math.exp(-(((abs_lat - subtropical_lat) / 9.5) ** 2))
    storm_track_wet = (505.0 + 60.0 * coriolis) * math.exp(-(((abs_lat - storm_lat) / 15.0) ** 2))
    polar_dry = 210.0 * math.exp(-(((abs_lat - 78.0) / 11.0) ** 2))
    circulation_base = 585.0 + itcz_wet * 0.86 + monsoon_wet * 0.66 + storm_track_wet * 0.80 - subtropical_dry * 0.88 - polar_dry

    coast_dist_km = _distance_cells_to_km(terrain, coast_dist_cells)
    fetch, mean_upwind_elev, max_upwind_elev = _upwind_fetch_and_relief(terrain, row, col, wind_dr, wind_dc)
    coast_factor = math.exp(-coast_dist_km / MOISTURE_COAST_EFOLD_KM)

    vapor_factor = 0.76 + 0.34 * atmosphere.water_vapor_factor
    pressure_factor = 0.88 + 0.14 * atmosphere.pressure_bar

    if is_land:
        # Moisture supply from local winds carrying ocean moisture inland.
        advection_factor = 0.58 + 0.28 * coast_factor + 0.62 * fetch

        # Orographic uplift on the windward side and rain shadow on the lee side.
        upslope = max(0.0, max(elevation_m, 0) - mean_upwind_elev)
        uplift_factor = 1.0 + 0.30 * clamp(upslope / 2200.0, 0.0, 1.0) * (0.55 + 0.45 * fetch)

        # Strong upstream barriers dry the lee side, especially for inland cells.
        barrier_height = max(0.0, max_upwind_elev - max(elevation_m, 0))
        inland_shadow_weight = 0.25 + 0.48 * (1.0 - coast_factor)
        rain_shadow_factor = 1.0 - 0.16 * clamp((barrier_height - 550.0) / 3200.0, 0.0, 1.0) * inland_shadow_weight
        rain_shadow_factor = clamp(rain_shadow_factor, 0.78, 1.14)

        # Slight convective bonus for low-latitude warm land.
        convective_bonus = 1.0 + 0.18 * math.exp(-(((abs_lat - 12.0) / 15.0) ** 2)) + 0.08 * math.exp(-(((abs_lat - 42.0) / 18.0) ** 2))

        precip = circulation_base * advection_factor * uplift_factor * rain_shadow_factor * convective_bonus
    else:
        # Over oceans, convergence/subsidence bands dominate.
        convergence_factor = (
            1.02
            + 0.18 * math.exp(-(((abs_lat - 6.0) / 11.0) ** 2))
            + 0.12 * math.exp(-(((abs_lat - 52.0) / 13.0) ** 2))
            - 0.14 * math.exp(-(((abs_lat - 28.0) / 8.0) ** 2))
        )
        precip = circulation_base * 1.12 * convergence_factor

    return clamp(precip * vapor_factor * pressure_factor, 55.0, 5400.0)



def _koppen_code(
    annual_temp_c: float,
    warmest_c: float,
    coldest_c: float,
    annual_precip_mm: float,
    is_land: bool,
) -> str:
    if not is_land:
        return OCEAN_CODE

    if warmest_c < 0.0:
        return "EF"
    if warmest_c < 10.0:
        return "ET"

    # Final classification avoids direct latitude cutoffs. Latitude affects
    # temperature, seasonality, global winds, and rainfall upstream; Köppen itself
    # is decided from the resulting regional climate fields.
    aridity_threshold = max(70.0, 16.0 * max(annual_temp_c, 0.0) + 185.0)
    if annual_precip_mm < aridity_threshold:
        return "BWh" if annual_temp_c >= 18.0 else "BWk"
    if annual_precip_mm < aridity_threshold * 1.25:
        return "BSh" if annual_temp_c >= 18.0 else "BSk"

    if coldest_c >= 18.0:
        if annual_precip_mm >= 2200.0:
            return "Af"
        if annual_precip_mm >= 1250.0:
            return "Am"
        return "Aw"

    if coldest_c > 0.0:
        # Mediterranean-like climates are inferred from mild winters, warm/dry
        # annual conditions, and limited precipitation, not from a hard latitude band.
        if 420.0 <= annual_precip_mm < 920.0 and warmest_c >= 18.0:
            return "Csa" if warmest_c >= 22.0 else "Csb"
        return "Cfa" if warmest_c >= 22.0 else "Cfb"

    if warmest_c >= 22.0:
        return "Dfa"
    if warmest_c >= 15.0:
        return "Dfb"
    return "Dfc"


def _prevailing_wind_vector(lat_degrees: float, rotation: RotationState, atmosphere: Atmosphere) -> tuple[float, float]:
    """Return average downwind direction (dr, dc) for the latitude band.

    Planet characteristics matter here. Shorter days mean stronger Coriolis,
    narrower circulation cells, and stronger zonal winds. Longer days broaden
    the tropical cell and weaken the east/west component. Pressure slightly
    damps winds in thick atmospheres.

    dr > 0 means toward the south (larger row index), dr < 0 toward the north.
    dc > 0 means eastward, dc < 0 westward.
    """
    abs_lat = abs(lat_degrees)
    hemisphere = 1.0 if lat_degrees >= 0.0 else -1.0
    coriolis = clamp(24.0 / max(8.0, rotation.rotation_period_hours), 0.42, 2.2)
    zonal = clamp(0.72 + 0.38 * coriolis, 0.55, 1.55) / clamp(0.85 + 0.10 * atmosphere.pressure_bar, 0.8, 1.25)
    hadley_edge = clamp(30.0 / math.sqrt(coriolis), 18.0, 48.0)
    ferrel_edge = clamp(hadley_edge + 30.0, 48.0, 72.0)

    if abs_lat < hadley_edge * 0.62:
        # Deep tropical trades: toward equator and east-to-west.
        return (0.48 * hemisphere, -1.00 * zonal)
    if abs_lat < hadley_edge:
        # Poleward edge of Hadley cell / subtropical highs.
        return (0.24 * hemisphere, -0.82 * zonal)
    if abs_lat < ferrel_edge:
        # Westerlies. Faster rotators have stronger zonal flow.
        return (-0.34 * hemisphere, 0.95 * zonal)
    # Polar easterlies: east-to-west and equatorward.
    return (0.35 * hemisphere, -0.72 * zonal)

def _upwind_fetch_and_relief(
    terrain: TerrainMap,
    row: int,
    col: int,
    wind_dr: float,
    wind_dc: float,
    max_steps: int | None = None,
) -> tuple[float, float, float]:
    """Estimate ocean fetch and upstream relief in the opposite direction of the prevailing wind."""
    height = terrain.height
    width = terrain.width
    scale = map_scale_for_terrain(terrain)
    if max_steps is None:
        max_steps = km_to_cells(terrain, UPWIND_FETCH_SCAN_KM, minimum=4, maximum=90)
    sample_r = row + 0.5
    sample_c = col + 0.5

    ocean_score = 0.0
    weight_sum = 0.0
    mean_elev_weighted = 0.0
    mean_elev_weights = 0.0
    max_upwind_elev = max(0, terrain.elevation_m[row][col])

    for step in range(1, max_steps + 1):
        sample_r -= wind_dr
        sample_c -= wind_dc
        ri = int(round(sample_r))
        ci = int(round(sample_c)) % width
        if ri < 0 or ri >= height:
            break

        step_km = (step - 1) * scale.representative_km_per_cell
        weight = math.exp(-step_km / UPWIND_FETCH_WEIGHT_EFOLD_KM)
        weight_sum += weight

        if terrain.is_land[ri][ci]:
            elev = max(0, terrain.elevation_m[ri][ci])
            mean_elev_weighted += elev * weight
            mean_elev_weights += weight
            if elev > max_upwind_elev:
                max_upwind_elev = elev
        else:
            ocean_score += weight

    fetch = ocean_score / weight_sum if weight_sum > 0.0 else 0.0
    mean_upwind_elev = mean_elev_weighted / mean_elev_weights if mean_elev_weights > 0.0 else 0.0
    return fetch, mean_upwind_elev, float(max_upwind_elev)




def _is_real_earth_terrain(terrain: TerrainMap) -> bool:
    return str(getattr(terrain, "source", "")).startswith("real_earth")


def _is_synthetic_earth_terrain(terrain: TerrainMap) -> bool:
    return str(getattr(terrain, "source", "")).startswith("synthetic_earth")


def _earth_temperature_adjustment(temp_c: float, lat: float, lon: float, is_land: bool, elevation_m: int) -> float:
    """Regional temperature nudges for Real Earth calibration mode.

    These are intentionally limited to known large-scale calibration failures:
    Antarctic/Greenland ice should remain ice, and monsoon/subtropical
    corrections should not rewrite the whole planet.
    """
    if not is_land:
        return temp_c

    # Antarctica must remain an ice sheet, not boreal forest. Elevation and
    # polar night make it much colder than a simple latitude curve suggests.
    if lat < -60.0:
        coldness = clamp((-lat - 60.0) / 22.0, 0.0, 1.0)
        temp_c -= 15.0 + 10.0 * coldness

    # Greenland ice cap.
    if 58.0 <= lat <= 84.0 and -75.0 <= lon <= -10.0:
        temp_c -= 8.0

    # Tibetan Plateau / Himalaya cold highlands: prevent warm forest over the
    # highest real-Earth relief where coarse relief underestimates cold stress.
    if 25.0 <= lat <= 38.0 and 70.0 <= lon <= 105.0 and elevation_m > 2200:
        temp_c -= 3.5

    return temp_c



def _earth_precipitation_adjustment(precip: float, lat: float, lon: float, is_land: bool) -> float:
    """Earth regional moisture corrections for calibration mode.

    The generic climate model has no explicit ocean currents, seasonal monsoon
    reversal, Gulf/Atlantic moisture transport, or subtropical continental
    high-pressure cells. These broad regional factors correct the largest known
    Real Earth failures while keeping generated worlds fully procedural.

    Corrections use feathered geographic masks rather than hard rectangles so
    the precipitation map does not show blocky artificial edges.
    """
    if not is_land:
        return precip

    mult = 1.0

    def apply(factor: float, weight: float) -> None:
        nonlocal mult
        if weight <= 0.0:
            return
        mult *= 1.0 + (factor - 1.0) * clamp(weight, 0.0, 1.0)

    # Antarctica and Greenland: cold polar deserts / ice sheets, not forests.
    apply(0.20, _lat_weight(lat, -90.0, -60.0, 5.0))
    apply(0.35, _box_weight(lat, lon, 58.0, 84.0, -75.0, -10.0, 4.0, 6.0))

    # Eastern North America receives Atlantic/Gulf moisture and summer
    # convection; the generic westerly-only model was making it too dry.
    apply(1.45, _box_weight(lat, lon, 24.0, 48.0, -97.0, -62.0, 5.0, 7.0))
    apply(1.18, _box_weight(lat, lon, 26.0, 38.0, -92.0, -75.0, 4.0, 5.0))

    # North American west: wet Pacific Northwest, drier California/Southwest.
    apply(1.28, _box_weight(lat, lon, 41.0, 57.0, -132.0, -116.0, 4.0, 4.0))
    apply(0.62, _box_weight(lat, lon, 25.0, 41.0, -126.0, -105.0, 4.0, 5.0))

    # Arabia, Persian Gulf desert belt, and the Thar / northwest India dry zone.
    apply(0.32, _box_weight(lat, lon, 12.0, 31.0, 38.0, 62.0, 4.0, 5.0))
    apply(0.42, _box_weight(lat, lon, 21.0, 31.0, 67.0, 78.0, 3.0, 3.5))
    # Deccan rain shadow / western interior India. Keep the immediate Western
    # Ghats coast from becoming totally dry but reduce broad false wetness.
    apply(0.70, _box_weight(lat, lon, 10.0, 22.0, 73.0, 79.0, 3.0, 2.5))

    # East Asian monsoon: eastern China, Korea, and Japan should be wetter than
    # the generic annual prevailing-wind model.
    apply(1.50, _box_weight(lat, lon, 20.0, 45.0, 105.0, 123.0, 5.0, 5.0))
    apply(1.42, _box_weight(lat, lon, 31.0, 46.0, 123.0, 132.0, 4.0, 3.5))
    apply(1.62, _box_weight(lat, lon, 30.0, 46.5, 129.0, 146.5, 4.0, 4.0))

    # Mainland Southeast Asia and south/east India receive strong monsoonal
    # moisture; this helps separate them from the dry Arabia/Thar zones.
    apply(1.22, _box_weight(lat, lon, 7.0, 24.0, 78.0, 105.0, 4.0, 5.0))
    apply(1.25, _box_weight(lat, lon, 5.0, 24.0, 88.0, 112.0, 4.0, 5.0))

    # Atacama and Namib-style coastal deserts.
    apply(0.30, _box_weight(lat, lon, -30.0, -15.0, -76.0, -68.0, 3.5, 3.0))
    apply(0.38, _box_weight(lat, lon, -31.0, -16.0, 11.0, 18.0, 3.5, 3.0))

    return clamp(precip * mult, 15.0, 6000.0)


def _smoothstep(edge0: float, edge1: float, x: float) -> float:
    if edge0 == edge1:
        return 1.0 if x >= edge1 else 0.0
    t = clamp((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _lat_weight(lat: float, lat_min: float, lat_max: float, edge: float) -> float:
    south = _smoothstep(lat_min - edge, lat_min + edge, lat)
    north = 1.0 - _smoothstep(lat_max - edge, lat_max + edge, lat)
    return south * north


def _lon_weight(lon: float, lon_min: float, lon_max: float, edge: float) -> float:
    # Current use cases do not cross the dateline. Keep this simple and clear.
    west = _smoothstep(lon_min - edge, lon_min + edge, lon)
    east = 1.0 - _smoothstep(lon_max - edge, lon_max + edge, lon)
    return west * east


def _box_weight(lat: float, lon: float, lat_min: float, lat_max: float, lon_min: float, lon_max: float, lat_edge: float, lon_edge: float) -> float:
    return _lat_weight(lat, lat_min, lat_max, lat_edge) * _lon_weight(lon, lon_min, lon_max, lon_edge)


def _distance_to_ocean_cells(is_land: list[list[bool]]) -> list[list[int]]:
    """Approximate distance-to-ocean in grid cells using a fast distance transform.

    The original BFS was correct but became very slow on full-resolution 2K/4K
    runs. SciPy's distance transform is much faster and handles the common case;
    columns are triplicated before the transform so longitude wrapping is
    respected. A pure-Python BFS remains as a fallback if SciPy is unavailable.
    """
    height = len(is_land)
    width = len(is_land[0]) if height else 0
    if not height or not width:
        return []
    try:
        import numpy as np
        from scipy import ndimage
        land = np.asarray(is_land, dtype=bool)
        # distance_transform_edt returns distance to the nearest False cell.
        tiled = np.concatenate([land, land, land], axis=1)
        dist = ndimage.distance_transform_edt(tiled)
        core = dist[:, width:2 * width]
        return np.rint(core).astype(np.int32).tolist()
    except Exception:
        pass

    large = width + height + 10
    dist = [[large for _ in range(width)] for _ in range(height)]
    queue: deque[tuple[int, int]] = deque()

    for r in range(height):
        for c in range(width):
            if not is_land[r][c]:
                dist[r][c] = 0
                queue.append((r, c))

    if not queue:
        return dist

    while queue:
        r, c = queue.popleft()
        next_d = dist[r][c] + 1
        for nr, nc in ((r - 1, c), (r + 1, c), (r, (c - 1) % width), (r, (c + 1) % width)):
            if nr < 0 or nr >= height:
                continue
            if next_d < dist[nr][nc]:
                dist[nr][nc] = next_d
                queue.append((nr, nc))

    return dist
