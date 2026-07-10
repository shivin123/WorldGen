"""Structured atmospheric-circulation climate engine for WorldGen.

This backend is the opt-in climate-overhaul v2 path. It keeps the public ClimateMap
contract used by hydrology/biomes/outputs, but builds annual fields from three
seasonal circulation states and stores native-resolution driver rasters for review.

The model is intentionally lightweight rather than a full GCM:

    radiation -> pressure -> winds -> ocean currents -> heat/moisture transport
    -> seasonal temperature/precipitation -> monthly Köppen.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from worldgen.models.planet_profile import Atmosphere, ClimateMap, RotationState, TerrainMap
from worldgen.physics.map_scale import map_scale_for_terrain
from worldgen.random_utils import clamp

OCEAN_CODE = "O"
SEASON_NAMES = ("nh_summer", "equinox", "nh_winter")
SEASON_DECLINATION_FACTOR = (1.0, 0.0, -1.0)
SEASON_WEIGHTS = (0.25, 0.50, 0.25)


@dataclass(frozen=True)
class SeasonalClimateDiagnostics:
    hadley_edge_degrees: float
    pressure_range_hpa: tuple[float, float]
    wind_speed_range: tuple[float, float]
    current_heat_range_c: tuple[float, float]
    moisture_range: tuple[float, float]
    mean_cloud_factor: float
    mean_land_moisture: float
    mean_land_orographic_lift: float
    mean_land_rain_shadow: float
    large_water_component_count: int
    inland_water_component_count: int


def generate_seasonal_v2_climate(
    rotation: RotationState,
    atmosphere: Atmosphere,
    terrain: TerrainMap,
    koppen_detail: str = "local4",
) -> ClimateMap:
    """Generate climate using the preserved seasonal_v2 atmospheric-circulation path."""
    return _generate_seasonal_structured_climate(
        rotation,
        atmosphere,
        terrain,
        koppen_detail=koppen_detail,
        climate_mode_name="seasonal_v2",
        ocean_model="loop_v1",
    )


def generate_seasonal_v3_climate(
    rotation: RotationState,
    atmosphere: Atmosphere,
    terrain: TerrainMap,
    koppen_detail: str = "local4",
) -> ClimateMap:
    """Generate climate using structured atmosphere plus first basin-aware ocean circulation."""
    return _generate_seasonal_structured_climate(
        rotation,
        atmosphere,
        terrain,
        koppen_detail=koppen_detail,
        climate_mode_name="seasonal_v3",
        ocean_model="basin_v2",
    )


def generate_seasonal_v4_climate(
    rotation: RotationState,
    atmosphere: Atmosphere,
    terrain: TerrainMap,
    koppen_detail: str = "local4",
) -> ClimateMap:
    """Generate climate using structured atmosphere plus refined basin-aware ocean routing."""
    return _generate_seasonal_structured_climate(
        rotation,
        atmosphere,
        terrain,
        koppen_detail=koppen_detail,
        climate_mode_name="seasonal_v4",
        ocean_model="basin_v3",
    )


def generate_seasonal_v5_climate(
    rotation: RotationState,
    atmosphere: Atmosphere,
    terrain: TerrainMap,
    koppen_detail: str = "local4",
) -> ClimateMap:
    """Generate climate using v4 ocean routing plus component-based moisture/rainfall coupling."""
    return _generate_seasonal_structured_climate(
        rotation,
        atmosphere,
        terrain,
        koppen_detail=koppen_detail,
        climate_mode_name="seasonal_v5",
        ocean_model="basin_v3",
        moisture_model="component_v2",
    )


def _generate_seasonal_structured_climate(
    rotation: RotationState,
    atmosphere: Atmosphere,
    terrain: TerrainMap,
    koppen_detail: str = "local4",
    *,
    climate_mode_name: str,
    ocean_model: str,
    moisture_model: str = "component_v1",
) -> ClimateMap:
    """Generate climate using three seasonal circulation states."""
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - environment guard
        raise RuntimeError(f"NumPy is required for {climate_mode_name} climate generation. Install it with: pip install numpy") from exc

    from worldgen.generators.climate_generator import _koppen_smoothing_passes, _smooth_numeric_grid_np

    height = terrain.height
    width = terrain.width
    land = np.asarray(terrain.is_land, dtype=bool)
    elev = np.asarray(terrain.elevation_m, dtype=np.float32)
    positive_elev = np.maximum(elev, 0.0)
    scale = map_scale_for_terrain(terrain)

    water = ~land
    water_info = _water_body_fields(np, water, scale)
    ocean_like = water_info["ocean_like"]
    inland_water = water & ~ocean_like
    small_lake_mask = inland_water & (water_info["inland_source_factor"] < 0.075)
    small_lake_buffer = np.where(land, np.exp(-_distance_to_source_cells(np, small_lake_mask) / 2.25), 0.0).astype(np.float32)
    ocean_dist_cells = _distance_to_source_cells(np, ocean_like)
    dist_km = ocean_dist_cells.astype(np.float32) * np.float32(scale.representative_km_per_cell)

    lats = np.linspace(90.0 - 90.0 / height, -90.0 + 90.0 / height, height, dtype=np.float32)
    lons = np.linspace(-180.0 + 180.0 / width, 180.0 - 180.0 / width, width, dtype=np.float32)
    lat_grid = lats[:, None]
    lon_grid = lons[None, :]
    lat_rad = np.radians(lat_grid)
    abs_lat = np.abs(lat_grid)
    lon_rad = np.radians(lon_grid)

    coriolis = clamp(24.0 / max(8.0, rotation.rotation_period_hours), 0.42, 2.2)
    hadley_edge = clamp(30.0 / math.sqrt(coriolis), 18.0, 48.0)
    pressure_factor = clamp(atmosphere.pressure_bar, 0.45, 3.5)
    vapor_factor = clamp(atmosphere.water_vapor_factor, 0.25, 2.2)
    tilt = clamp(rotation.axial_tilt_degrees, 0.0, 75.0)

    ocean_distance_factor = np.exp(-dist_km / max(250.0, 1250.0 * scale.planet_radius_earth))
    continentality = np.where(land, np.clip(dist_km / max(300.0, 2100.0 * scale.planet_radius_earth), 0.0, 1.0), 0.0)
    coastal_moderation = np.where(land, ocean_distance_factor, 1.0)
    lapse_c = positive_elev * 0.0062
    gy, gx = np.gradient(positive_elev)
    relief = np.sqrt(gx * gx + gy * gy)
    relief_scale = max(1.0, float(np.quantile(relief[land], 0.97)) if np.any(land) else 1.0)
    relief_norm = np.clip(relief / relief_scale, 0.0, 1.0)

    global_baseline_c = float(atmosphere.estimated_mean_surface_temp_c)
    reference_insolation = _daily_insolation(np, lat_rad, 0.0)
    reference_albedo = _surface_albedo_field(np, land, abs_lat, positive_elev)

    seasonal_temps: list = []
    seasonal_precips: list = []
    seasonal_pressures: list = []
    seasonal_winds_u: list = []
    seasonal_winds_v: list = []
    seasonal_current_u: list = []
    seasonal_current_v: list = []
    seasonal_currents_heat: list = []
    seasonal_moisture: list = []
    seasonal_orographic: list = []
    seasonal_shadow: list = []
    seasonal_itcz: list = []
    seasonal_thermal_equator: list = []
    seasonal_pressure_belts: list = []
    seasonal_moisture_winds_u: list = []
    seasonal_moisture_winds_v: list = []
    seasonal_storm_tracks: list = []
    seasonal_ocean_path_classes: list = []
    seasonal_ocean_gyre_classes: list = []
    seasonal_coastal_upwelling: list = []
    seasonal_warm_current_influence: list = []
    seasonal_cold_current_influence: list = []
    seasonal_coastal_desert_potential: list = []
    seasonal_trade_wind_moisture: list = []
    seasonal_monsoon_moisture: list = []
    seasonal_frontal_moisture: list = []
    seasonal_orographic_precip_potential: list = []
    seasonal_coastal_dryness: list = []
    ocean_basin_id = np.zeros((height, width), dtype=np.int32)
    ocean_basin_kind = np.zeros((height, width), dtype=np.int16)

    previous_cloud = np.zeros((height, width), dtype=np.float32)
    previous_current_heat = np.zeros((height, width), dtype=np.float32)

    for decl_factor in SEASON_DECLINATION_FACTOR:
        declination = math.radians(tilt * decl_factor)
        declination_degrees = tilt * decl_factor
        insolation = _daily_insolation(np, lat_rad, declination)
        absorbed = insolation * (1.0 - reference_albedo) * (1.0 - 0.08 * previous_cloud)
        absorbed_anomaly = absorbed - np.mean(reference_insolation * (1.0 - reference_albedo))

        latitude_gradient = 38.0 * (np.cos(lat_rad) - 0.70)
        seasonal_gain = 42.0 * absorbed_anomaly
        land_season_gain = np.where(land, 1.02 + 0.48 * continentality, 0.40)
        ocean_buffer = np.where(water, -1.3 + 0.8 * np.cos(lat_rad), 0.0)
        continental_heat = np.where(land, 1.7 * continentality * np.maximum(0.0, 1.0 - abs_lat / 70.0), 0.0)
        maritime_high_lat_warm = np.where(land, 2.5 * coastal_moderation * np.clip((abs_lat - 38.0) / 34.0, 0.0, 1.0), 0.0)
        temp_c = (
            global_baseline_c
            + latitude_gradient
            + seasonal_gain * land_season_gain
            - lapse_c
            + ocean_buffer
            + continental_heat
            + maritime_high_lat_warm
            + previous_current_heat
        )

        thermal_equator_strength, thermal_equator_lat = _thermal_equator_field(
            np=np,
            lat_grid=lat_grid,
            declination_degrees=declination_degrees,
            hadley_edge=hadley_edge,
            land=land,
            temp_c=temp_c,
            ocean_like=ocean_like,
        )
        itcz_strength = _itcz_strength_field(np, lat_grid, declination_degrees, hadley_edge, land, temp_c, ocean_like, thermal_equator_lat)
        pressure = _seasonal_pressure_field(
            np=np,
            temp_c=temp_c,
            land=land,
            elev_m=positive_elev,
            lat_grid=lat_grid,
            abs_lat=abs_lat,
            lon_rad=lon_rad,
            declination_degrees=declination_degrees,
            hadley_edge=hadley_edge,
            pressure_bar=pressure_factor,
            itcz_strength=itcz_strength,
            thermal_equator_lat=thermal_equator_lat,
        )
        pressure_belt_class = _pressure_belt_class(np, lat_grid, hadley_edge, itcz_strength, thermal_equator_lat)
        wind_u, wind_v = _wind_from_pressure(np, pressure, lat_grid, rotation, atmosphere, hadley_edge, itcz_strength, thermal_equator_lat)
        moisture_wind_u, moisture_wind_v = _moisture_transport_wind(
            np=np,
            wind_u=wind_u,
            wind_v=wind_v,
            pressure=pressure,
            lat_grid=lat_grid,
            abs_lat=abs_lat,
            hadley_edge=hadley_edge,
            itcz_strength=itcz_strength,
            thermal_equator_lat=thermal_equator_lat,
            land=land,
            dist_km=dist_km,
        )
        storm_track_moisture = _storm_track_moisture_field(
            np=np,
            lat_grid=lat_grid,
            abs_lat=abs_lat,
            hadley_edge=hadley_edge,
            thermal_equator_lat=thermal_equator_lat,
            pressure=pressure,
            land=land,
            dist_km=dist_km,
            ocean_distance_factor=ocean_distance_factor,
        )
        if ocean_model == "basin_v3":
            from worldgen.generators.ocean_circulation_v3 import build_ocean_circulation_v3

            ocean_state = build_ocean_circulation_v3(
                np=np,
                wind_u=wind_u,
                wind_v=wind_v,
                moisture_wind_u=moisture_wind_u,
                moisture_wind_v=moisture_wind_v,
                land=land,
                ocean_like=ocean_like,
                lat_grid=lat_grid,
                abs_lat=abs_lat,
                dist_km=dist_km,
                terrain=terrain,
                hadley_edge=hadley_edge,
                itcz_strength=itcz_strength,
            )
            current_u = ocean_state.current_u
            current_v = ocean_state.current_v
            current_heat = ocean_state.current_heat
            ocean_basin_id = ocean_state.ocean_basin_id
            ocean_basin_kind = getattr(ocean_state, "ocean_basin_kind", np.where(ocean_basin_id > 0, 1, 0).astype(np.int16))
            ocean_path_class = ocean_state.ocean_current_path_class
            ocean_gyre_class = ocean_state.ocean_gyre_class
            coastal_upwelling = ocean_state.coastal_upwelling
            warm_current_influence = ocean_state.warm_current_influence
            cold_current_influence = ocean_state.cold_current_influence
            coastal_desert_potential = ocean_state.coastal_desert_potential
        elif ocean_model == "basin_v2":
            from worldgen.generators.ocean_circulation_v2 import build_ocean_circulation_v2

            ocean_state = build_ocean_circulation_v2(
                np=np,
                wind_u=wind_u,
                wind_v=wind_v,
                land=land,
                ocean_like=ocean_like,
                lat_grid=lat_grid,
                abs_lat=abs_lat,
                dist_km=dist_km,
                terrain=terrain,
                hadley_edge=hadley_edge,
                itcz_strength=itcz_strength,
            )
            current_u = ocean_state.current_u
            current_v = ocean_state.current_v
            current_heat = ocean_state.current_heat
            ocean_basin_id = ocean_state.ocean_basin_id
            ocean_basin_kind = getattr(ocean_state, "ocean_basin_kind", np.where(ocean_basin_id > 0, 1, 0).astype(np.int16))
            ocean_path_class = ocean_state.ocean_current_path_class
            ocean_gyre_class = ocean_state.ocean_gyre_class
            coastal_upwelling = ocean_state.coastal_upwelling
            warm_current_influence = ocean_state.warm_current_influence
            cold_current_influence = ocean_state.cold_current_influence
            coastal_desert_potential = ocean_state.coastal_desert_potential
        else:
            current_u, current_v, current_heat = _ocean_currents_and_heat(
                np=np,
                wind_u=wind_u,
                wind_v=wind_v,
                land=land,
                ocean_like=ocean_like,
                lat_grid=lat_grid,
                abs_lat=abs_lat,
                dist_km=dist_km,
                terrain=terrain,
            )
            ocean_path_class = _ocean_current_path_class(np, current_u, current_v, current_heat, ocean_like, lat_grid)
            ocean_gyre_class = _ocean_gyre_class(np, current_u, current_v, ocean_like, lat_grid)
            coastal_upwelling = np.zeros((height, width), dtype=np.float32)
            warm_current_influence = np.zeros((height, width), dtype=np.float32)
            cold_current_influence = np.zeros((height, width), dtype=np.float32)
            coastal_desert_potential = np.zeros((height, width), dtype=np.float32)
        temp_c = temp_c + current_heat + np.where(land, 1.15 * warm_current_influence - 1.35 * cold_current_influence, 0.0)

        if moisture_model == "component_v2":
            moisture, precip, cloud, orographic, rain_shadow, moisture_components = _moisture_and_precipitation_v2(
                np=np,
                temp_c=temp_c,
                pressure=pressure,
                wind_u=moisture_wind_u,
                wind_v=moisture_wind_v,
                current_heat=current_heat,
                warm_current_influence=warm_current_influence,
                cold_current_influence=cold_current_influence,
                coastal_upwelling=coastal_upwelling,
                coastal_desert_potential=coastal_desert_potential,
                storm_track_moisture=storm_track_moisture,
                land=land,
                water=water,
                ocean_like=ocean_like,
                inland_water=inland_water,
                inland_water_source_factor=water_info["inland_source_factor"],
                small_lake_buffer=small_lake_buffer,
                elev_m=positive_elev,
                gx=gx,
                gy=gy,
                relief_norm=relief_norm,
                lat_grid=lat_grid,
                abs_lat=abs_lat,
                dist_km=dist_km,
                atmosphere=atmosphere,
                vapor_factor=vapor_factor,
                pressure_factor=pressure_factor,
                hadley_edge=hadley_edge,
                itcz_strength=itcz_strength,
            )
        else:
            moisture, precip, cloud, orographic, rain_shadow = _moisture_and_precipitation(
                np=np,
                temp_c=temp_c,
                pressure=pressure,
                wind_u=moisture_wind_u,
                wind_v=moisture_wind_v,
                current_heat=current_heat,
                cold_current_influence=cold_current_influence,
                coastal_upwelling=coastal_upwelling,
                coastal_desert_potential=coastal_desert_potential,
                storm_track_moisture=storm_track_moisture,
                land=land,
                water=water,
                ocean_like=ocean_like,
                inland_water=inland_water,
                inland_water_source_factor=water_info["inland_source_factor"],
                small_lake_buffer=small_lake_buffer,
                elev_m=positive_elev,
                gx=gx,
                gy=gy,
                relief_norm=relief_norm,
                lat_grid=lat_grid,
                abs_lat=abs_lat,
                dist_km=dist_km,
                atmosphere=atmosphere,
                vapor_factor=vapor_factor,
                pressure_factor=pressure_factor,
                hadley_edge=hadley_edge,
                itcz_strength=itcz_strength,
            )
            moisture_components = _empty_moisture_components(np, height, width)

        previous_cloud = cloud
        previous_current_heat = current_heat * 0.45
        seasonal_temps.append(temp_c.astype(np.float32))
        seasonal_precips.append(precip.astype(np.float32))
        seasonal_pressures.append(pressure.astype(np.float32))
        seasonal_winds_u.append(wind_u.astype(np.float32))
        seasonal_winds_v.append(wind_v.astype(np.float32))
        seasonal_current_u.append(current_u.astype(np.float32))
        seasonal_current_v.append(current_v.astype(np.float32))
        seasonal_currents_heat.append(current_heat.astype(np.float32))
        seasonal_moisture.append(moisture.astype(np.float32))
        seasonal_orographic.append(orographic.astype(np.float32))
        seasonal_shadow.append(rain_shadow.astype(np.float32))
        seasonal_itcz.append(itcz_strength.astype(np.float32))
        seasonal_thermal_equator.append(thermal_equator_strength.astype(np.float32))
        seasonal_pressure_belts.append(pressure_belt_class.astype(np.int16))
        seasonal_moisture_winds_u.append(moisture_wind_u.astype(np.float32))
        seasonal_moisture_winds_v.append(moisture_wind_v.astype(np.float32))
        seasonal_storm_tracks.append(storm_track_moisture.astype(np.float32))
        seasonal_ocean_path_classes.append(ocean_path_class.astype(np.int16))
        seasonal_ocean_gyre_classes.append(ocean_gyre_class.astype(np.int16))
        seasonal_coastal_upwelling.append(coastal_upwelling.astype(np.float32))
        seasonal_warm_current_influence.append(warm_current_influence.astype(np.float32))
        seasonal_cold_current_influence.append(cold_current_influence.astype(np.float32))
        seasonal_coastal_desert_potential.append(coastal_desert_potential.astype(np.float32))
        seasonal_trade_wind_moisture.append(moisture_components["trade_wind_moisture"].astype(np.float32))
        seasonal_monsoon_moisture.append(moisture_components["monsoon_moisture"].astype(np.float32))
        seasonal_frontal_moisture.append(moisture_components["frontal_moisture"].astype(np.float32))
        seasonal_orographic_precip_potential.append(moisture_components["orographic_precip_potential"].astype(np.float32))
        seasonal_coastal_dryness.append(moisture_components["coastal_dryness"].astype(np.float32))

    seasonal_temp_stack = np.stack(seasonal_temps, axis=0)
    seasonal_precip_stack = np.stack(seasonal_precips, axis=0)
    seasonal_pressure_stack = np.stack(seasonal_pressures, axis=0)
    seasonal_wind_u_stack = np.stack(seasonal_winds_u, axis=0)
    seasonal_wind_v_stack = np.stack(seasonal_winds_v, axis=0)
    seasonal_current_u_stack = np.stack(seasonal_current_u, axis=0)
    seasonal_current_v_stack = np.stack(seasonal_current_v, axis=0)
    seasonal_current_heat_stack = np.stack(seasonal_currents_heat, axis=0)
    seasonal_moisture_stack = np.stack(seasonal_moisture, axis=0)
    seasonal_orographic_stack = np.stack(seasonal_orographic, axis=0)
    seasonal_shadow_stack = np.stack(seasonal_shadow, axis=0)
    seasonal_itcz_stack = np.stack(seasonal_itcz, axis=0)
    seasonal_thermal_equator_stack = np.stack(seasonal_thermal_equator, axis=0)
    seasonal_pressure_belt_stack = np.stack(seasonal_pressure_belts, axis=0)
    seasonal_moisture_wind_u_stack = np.stack(seasonal_moisture_winds_u, axis=0)
    seasonal_moisture_wind_v_stack = np.stack(seasonal_moisture_winds_v, axis=0)
    seasonal_storm_track_stack = np.stack(seasonal_storm_tracks, axis=0)
    seasonal_ocean_path_stack = np.stack(seasonal_ocean_path_classes, axis=0)
    seasonal_ocean_gyre_stack = np.stack(seasonal_ocean_gyre_classes, axis=0)
    seasonal_coastal_upwelling_stack = np.stack(seasonal_coastal_upwelling, axis=0)
    seasonal_warm_current_stack = np.stack(seasonal_warm_current_influence, axis=0)
    seasonal_cold_current_stack = np.stack(seasonal_cold_current_influence, axis=0)
    seasonal_coastal_desert_stack = np.stack(seasonal_coastal_desert_potential, axis=0)
    seasonal_trade_wind_moisture_stack = np.stack(seasonal_trade_wind_moisture, axis=0)
    seasonal_monsoon_moisture_stack = np.stack(seasonal_monsoon_moisture, axis=0)
    seasonal_frontal_moisture_stack = np.stack(seasonal_frontal_moisture, axis=0)
    seasonal_orographic_precip_potential_stack = np.stack(seasonal_orographic_precip_potential, axis=0)
    seasonal_coastal_dryness_stack = np.stack(seasonal_coastal_dryness, axis=0)

    weights = np.asarray(SEASON_WEIGHTS, dtype=np.float32)[:, None, None]
    annual_temp = np.sum(seasonal_temp_stack * weights, axis=0)
    annual_precip = np.sum(seasonal_precip_stack * weights, axis=0)
    annual_moisture = np.sum(seasonal_moisture_stack * weights, axis=0)
    annual_orographic = np.sum(seasonal_orographic_stack * weights, axis=0)
    annual_shadow = np.sum(seasonal_shadow_stack * weights, axis=0)
    annual_wind_u = np.sum(seasonal_wind_u_stack * weights, axis=0)
    annual_wind_v = np.sum(seasonal_wind_v_stack * weights, axis=0)
    annual_current_u = np.sum(seasonal_current_u_stack * weights, axis=0)
    annual_current_v = np.sum(seasonal_current_v_stack * weights, axis=0)
    annual_current_heat = np.sum(seasonal_current_heat_stack * weights, axis=0)
    annual_itcz = np.max(seasonal_itcz_stack, axis=0)
    annual_thermal_equator = np.max(seasonal_thermal_equator_stack, axis=0)
    annual_moisture_wind_u = np.sum(seasonal_moisture_wind_u_stack * weights, axis=0)
    annual_moisture_wind_v = np.sum(seasonal_moisture_wind_v_stack * weights, axis=0)
    annual_storm_track = np.sum(seasonal_storm_track_stack * weights, axis=0)
    annual_coastal_upwelling = np.sum(seasonal_coastal_upwelling_stack * weights, axis=0)
    annual_warm_current_influence = np.sum(seasonal_warm_current_stack * weights, axis=0)
    annual_cold_current_influence = np.sum(seasonal_cold_current_stack * weights, axis=0)
    annual_coastal_desert_potential = np.sum(seasonal_coastal_desert_stack * weights, axis=0)
    annual_trade_wind_moisture = np.sum(seasonal_trade_wind_moisture_stack * weights, axis=0)
    annual_monsoon_moisture = np.sum(seasonal_monsoon_moisture_stack * weights, axis=0)
    annual_frontal_moisture = np.sum(seasonal_frontal_moisture_stack * weights, axis=0)
    annual_orographic_precip_potential = np.sum(seasonal_orographic_precip_potential_stack * weights, axis=0)
    annual_coastal_dryness = np.sum(seasonal_coastal_dryness_stack * weights, axis=0)
    annual_ocean_path_class = _weighted_mode_class(np, seasonal_ocean_path_stack, weights[:, 0, 0])
    annual_ocean_gyre_class = _weighted_mode_class(np, seasonal_ocean_gyre_stack, weights[:, 0, 0])

    monthly_temp, monthly_precip = _synthesize_monthly_fields(np, seasonal_temp_stack, seasonal_precip_stack, lat_grid, land, dist_km)
    warmest = np.max(monthly_temp, axis=0)
    coldest = np.min(monthly_temp, axis=0)

    annual_temp_x10 = np.rint(annual_temp * 10.0).astype(np.int16)
    warmest_x10 = np.rint(warmest * 10.0).astype(np.int16)
    coldest_x10 = np.rint(coldest * 10.0).astype(np.int16)
    annual_precip_int = np.rint(np.clip(annual_precip, 15.0, 7500.0)).astype(np.int32)

    passes = _koppen_smoothing_passes(width, height, koppen_detail)
    class_monthly_temp = np.stack([_smooth_numeric_grid_np(monthly_temp[i], passes=passes) for i in range(12)], axis=0)
    class_monthly_precip = np.stack([_smooth_numeric_grid_np(monthly_precip[i], passes=passes) for i in range(12)], axis=0)
    code_arr = _koppen_from_monthly(np, class_monthly_temp, class_monthly_precip, land, lat_grid)
    code_labels = ["O", "Af", "Am", "Aw", "BWh", "BWk", "BSh", "BSk", "Cfa", "Cfb", "Cfc", "Csa", "Csb", "Csc", "Cwa", "Cwb", "Cwc", "Dfa", "Dfb", "Dfc", "Dfd", "Dsa", "Dsb", "Dsc", "Dsd", "Dwa", "Dwb", "Dwc", "Dwd", "ET", "EF"]
    koppen_grid = [[code_labels[int(v)] for v in row] for row in code_arr]

    koppen_summary: dict[str, int] = {}
    if np.any(land):
        unique, counts = np.unique(code_arr[land], return_counts=True)
        for value, count in zip(unique, counts):
            label = code_labels[int(value)]
            if label != OCEAN_CODE:
                koppen_summary[label] = int(count)

    area_weight = np.maximum(0.01, np.cos(np.radians(lat_grid))).astype(np.float64)
    land_weight = np.where(land, area_weight, 0.0)
    water_weight = np.where(water, area_weight, 0.0)
    land_count = float(land_weight.sum())
    water_count = float(water_weight.sum())

    wind_speed = np.sqrt(seasonal_wind_u_stack ** 2 + seasonal_wind_v_stack ** 2)
    diagnostics = SeasonalClimateDiagnostics(
        hadley_edge_degrees=float(hadley_edge),
        pressure_range_hpa=(float(np.min(seasonal_pressure_stack)), float(np.max(seasonal_pressure_stack))),
        wind_speed_range=(float(np.min(wind_speed)), float(np.max(wind_speed))),
        current_heat_range_c=(float(np.min(seasonal_current_heat_stack)), float(np.max(seasonal_current_heat_stack))),
        moisture_range=(float(np.min(seasonal_moisture_stack)), float(np.max(seasonal_moisture_stack))),
        mean_cloud_factor=float(np.mean(previous_cloud)),
        mean_land_moisture=float((annual_moisture * land_weight).sum() / land_count) if land_count else 0.0,
        mean_land_orographic_lift=float((annual_orographic * land_weight).sum() / land_count) if land_count else 0.0,
        mean_land_rain_shadow=float((annual_shadow * land_weight).sum() / land_count) if land_count else 0.0,
        large_water_component_count=int(water_info["large_water_component_count"]),
        inland_water_component_count=int(water_info["inland_water_component_count"]),
    )

    aridity_threshold = _koppen_aridity_threshold(np, monthly_temp, monthly_precip, lat_grid)
    aridity_index = np.where(land, annual_precip / np.maximum(aridity_threshold, 1.0), 0.0)
    driver_maps, driver_info = _build_driver_maps(
        np=np,
        terrain=terrain,
        land=land,
        seasonal_temp_stack=seasonal_temp_stack,
        seasonal_precip_stack=seasonal_precip_stack,
        seasonal_pressure_stack=seasonal_pressure_stack,
        seasonal_wind_u_stack=seasonal_wind_u_stack,
        seasonal_wind_v_stack=seasonal_wind_v_stack,
        seasonal_current_u_stack=seasonal_current_u_stack,
        seasonal_current_v_stack=seasonal_current_v_stack,
        seasonal_current_heat_stack=seasonal_current_heat_stack,
        seasonal_moisture_stack=seasonal_moisture_stack,
        seasonal_orographic_stack=seasonal_orographic_stack,
        seasonal_shadow_stack=seasonal_shadow_stack,
        seasonal_itcz_stack=seasonal_itcz_stack,
        seasonal_thermal_equator_stack=seasonal_thermal_equator_stack,
        seasonal_pressure_belt_stack=seasonal_pressure_belt_stack,
        seasonal_moisture_wind_u_stack=seasonal_moisture_wind_u_stack,
        seasonal_moisture_wind_v_stack=seasonal_moisture_wind_v_stack,
        seasonal_storm_track_stack=seasonal_storm_track_stack,
        monthly_temp=monthly_temp,
        monthly_precip=monthly_precip,
        circulation_zone_class=_circulation_zone_class(np, lat_grid, tilt, hadley_edge, seasonal_itcz_stack),
        ocean_basin_id=ocean_basin_id,
        ocean_basin_kind=ocean_basin_kind,
        ocean_current_path_class=annual_ocean_path_class,
        ocean_gyre_class=annual_ocean_gyre_class,
        coastal_upwelling=annual_coastal_upwelling,
        warm_current_influence=annual_warm_current_influence,
        cold_current_influence=annual_cold_current_influence,
        coastal_desert_potential=annual_coastal_desert_potential,
        trade_wind_moisture=annual_trade_wind_moisture,
        monsoon_moisture=annual_monsoon_moisture,
        frontal_moisture=annual_frontal_moisture,
        orographic_precip_potential=annual_orographic_precip_potential,
        coastal_dryness=annual_coastal_dryness,
        annual_moisture=annual_moisture,
        annual_orographic=annual_orographic,
        annual_shadow=annual_shadow,
        annual_wind_u=annual_wind_u,
        annual_wind_v=annual_wind_v,
        annual_current_u=annual_current_u,
        annual_current_v=annual_current_v,
        annual_current_heat=annual_current_heat,
        annual_itcz=annual_itcz,
        annual_thermal_equator=annual_thermal_equator,
        annual_moisture_wind_u=annual_moisture_wind_u,
        annual_moisture_wind_v=annual_moisture_wind_v,
        annual_storm_track=annual_storm_track,
        seasonal_circulation_zone_stack=np.stack([
            _circulation_zone_class_for_season(np, lat_grid, tilt, hadley_edge, seasonal_itcz_stack[idx])
            for idx in range(len(SEASON_NAMES))
        ], axis=0),
        aridity_index=aridity_index,
        inland_water_source_factor=water_info["inland_source_factor"],
        small_lake_buffer=small_lake_buffer,
        ocean_like=ocean_like,
        climate_mode_name=climate_mode_name,
    )

    notes = [
        f"Climate mode: {climate_mode_name}.",
        f"{climate_mode_name} uses explicit structured NH summer, equinox, and NH winter atmospheric-circulation states before rainfall is generated.",
        (
            "Climate Overhaul09 adds seasonal_v5 component-based moisture/rainfall coupling on top of the seasonal_v4 ocean driver."
            if moisture_model == "component_v2"
            else (
                "Climate Overhaul08 refines seasonal_v4 basin routing, current heat transport, upwelling, and coastal-desert feedback."
                if ocean_model == "basin_v3"
                else (
                    "Climate Overhaul07 adds basin-aware ocean basin/current/upwelling/coastal-desert driver layers."
                    if ocean_model == "basin_v2"
                    else "Climate Overhaul06 adds structured thermal-equator, pressure-belt, surface-wind, moisture-wind, and storm-track layers while preserving seasonal_v1 and legacy as rollback modes."
                )
            )
        ),
        "Annual maps are weighted summaries of seasonal states; Köppen classification is derived from synthesized monthly temperature and precipitation.",
        f"Hadley edge estimate: {diagnostics.hadley_edge_degrees:.1f} degrees; pressure range {diagnostics.pressure_range_hpa[0]:.0f}-{diagnostics.pressure_range_hpa[1]:.0f} hPa.",
        f"Mean land moisture index {diagnostics.mean_land_moisture:.2f}; mean orographic lift {diagnostics.mean_land_orographic_lift:.2f}; mean rain-shadow index {diagnostics.mean_land_rain_shadow:.2f}.",
        f"Large water components treated as ocean-like: {diagnostics.large_water_component_count}; smaller inland-water components: {diagnostics.inland_water_component_count}.",
        f"Map scale: {scale.representative_km_per_cell:.2f} km/cell on a {scale.planet_radius_earth:.2f} R_earth planet.",
        "Rollback/comparison modes remain available: --climate-mode seasonal_v4, --climate-mode seasonal_v3, --climate-mode seasonal_v2, --climate-mode seasonal_v1, and --climate-mode legacy.",
    ]

    return ClimateMap(
        width=width,
        height=height,
        annual_mean_temp_c_x10=annual_temp_x10.astype(int).tolist(),
        warmest_month_temp_c_x10=warmest_x10.astype(int).tolist(),
        coldest_month_temp_c_x10=coldest_x10.astype(int).tolist(),
        annual_precip_mm=annual_precip_int.astype(int).tolist(),
        koppen_classification=koppen_grid,
        mean_land_temp_c=float((annual_temp * land_weight).sum() / land_count) if land_count else 0.0,
        mean_ocean_temp_c=float((annual_temp * water_weight).sum() / water_count) if water_count else 0.0,
        mean_land_precip_mm=float((annual_precip * land_weight).sum() / land_count) if land_count else 0.0,
        mean_ocean_precip_mm=float((annual_precip * water_weight).sum() / water_count) if water_count else 0.0,
        min_temp_c=float(np.min(annual_temp)),
        max_temp_c=float(np.max(annual_temp)),
        min_precip_mm=int(np.min(annual_precip_int)),
        max_precip_mm=int(np.max(annual_precip_int)),
        koppen_summary=dict(sorted(koppen_summary.items(), key=lambda item: item[0])),
        notes=notes,
        climate_mode=climate_mode_name,
        climate_driver_maps=driver_maps,
        climate_driver_map_info=driver_info,
    )


def _daily_insolation(np, lat_rad, declination_rad: float):
    sin_lat = np.sin(lat_rad)
    cos_lat = np.cos(lat_rad)
    sin_dec = math.sin(declination_rad)
    cos_dec = math.cos(declination_rad)
    cos_h0 = -np.tan(lat_rad) * math.tan(declination_rad)
    polar_day = cos_h0 <= -1.0
    polar_night = cos_h0 >= 1.0
    h0 = np.arccos(np.clip(cos_h0, -1.0, 1.0))
    daily = (h0 * sin_lat * sin_dec + cos_lat * cos_dec * np.sin(h0)) / math.pi
    daily = np.where(polar_day, sin_lat * sin_dec, daily)
    daily = np.where(polar_night, 0.0, daily)
    return np.clip(daily / 0.318, 0.0, 2.25).astype(np.float32)


def _surface_albedo_field(np, land, abs_lat, elev_m):
    ocean_albedo = 0.08 + 0.08 * np.clip((abs_lat - 55.0) / 35.0, 0.0, 1.0)
    land_albedo = 0.24 + 0.10 * np.clip((abs_lat - 45.0) / 45.0, 0.0, 1.0) + 0.05 * np.clip((elev_m - 1800.0) / 2800.0, 0.0, 1.0)
    return np.where(land, land_albedo, ocean_albedo).astype(np.float32)


def _blur_periodic(np, arr, passes: int = 3, lat_weight: float = 0.18, lon_weight: float = 0.22):
    out = np.asarray(arr, dtype=np.float32)
    for _ in range(max(0, int(passes))):
        north = np.vstack([out[0:1, :], out[:-1, :]])
        south = np.vstack([out[1:, :], out[-1:, :]])
        west = np.roll(out, 1, axis=1)
        east = np.roll(out, -1, axis=1)
        center = max(0.0, 1.0 - 2.0 * lat_weight - 2.0 * lon_weight)
        out = center * out + lat_weight * (north + south) + lon_weight * (west + east)
    return out.astype(np.float32)


def _thermal_equator_field(*, np, lat_grid, declination_degrees: float, hadley_edge: float, land, temp_c, ocean_like):
    """Return a smooth thermal-equator/convergence target field and its local latitude.

    seasonal_v1 used a mostly global latitude shift plus a small per-cell warm-land
    nudge.  seasonal_v2 makes the thermal equator an explicit structured layer:
    warm continents pull it poleward in their summer hemisphere, warm tropical seas
    anchor it over ocean basins, and the final target is smoothed so pressure/rainfall
    bands do not become straight hard-latitude cuts.
    """
    base_lat = declination_degrees * 0.62
    tropical = np.exp(-((lat_grid / max(12.0, hadley_edge * 0.52)) ** 2))
    temp_anom = temp_c - float(np.mean(temp_c))
    warm_pool = np.clip(temp_anom / 16.0, -0.8, 1.6)
    summer_hemi = 1.0 if declination_degrees >= 0.0 else -1.0
    same_hemi = np.clip(summer_hemi * lat_grid / max(8.0, hadley_edge), 0.0, 1.0)
    warm_land_pull = np.where(land, np.clip(warm_pool, 0.0, 1.5) * (0.35 + 0.65 * same_hemi), 0.0)
    warm_ocean_anchor = np.where(ocean_like, np.clip(warm_pool + 0.15, 0.0, 1.2) * tropical, 0.0)
    pull = _blur_periodic(np, 0.72 * warm_land_pull + 0.34 * warm_ocean_anchor, passes=7, lat_weight=0.16, lon_weight=0.25)
    sign_target = np.where(declination_degrees >= 0.0, 1.0, -1.0)
    local_shift = np.clip(sign_target * pull * 12.0, -10.5, 10.5)
    # Keep the equinox close to the equator but still let persistent tropical warm pools distort it.
    if abs(declination_degrees) < 1.0:
        local_shift *= 0.35
    target_lat = np.clip(base_lat + local_shift, -hadley_edge * 0.72, hadley_edge * 0.72)
    width = max(5.4, hadley_edge * 0.24)
    strength = np.exp(-(((lat_grid - target_lat) / width) ** 2))
    strength = np.clip(strength * (0.82 + 0.30 * pull), 0.0, 1.8)
    return strength.astype(np.float32), target_lat.astype(np.float32)


def _itcz_strength_field(np, lat_grid, declination_degrees: float, hadley_edge: float, land, temp_c, ocean_like, thermal_equator_lat):
    temp_anom = temp_c - float(np.mean(temp_c))
    warm_pool = np.clip(temp_anom / 18.0, -0.5, 1.5)
    land_conv = np.where(land, np.clip(warm_pool, 0.0, 1.35), 0.0)
    ocean_conv = np.where(ocean_like, np.clip(warm_pool + 0.10, 0.0, 1.15), 0.0)
    conv_pool = _blur_periodic(np, 0.55 * land_conv + 0.34 * ocean_conv, passes=5, lat_weight=0.17, lon_weight=0.24)
    width = max(6.8, hadley_edge * 0.30)
    base = np.exp(-(((lat_grid - thermal_equator_lat) / width) ** 2))
    monsoon_pull = np.where(land, 0.28 * conv_pool, 0.0)
    ocean_anchor = np.where(ocean_like, 0.13 * conv_pool, 0.0)
    return np.clip(base * (0.78 + monsoon_pull + ocean_anchor) + 0.18 * conv_pool * base, 0.0, 1.85).astype(np.float32)


def _seasonal_pressure_field(*, np, temp_c, land, elev_m, lat_grid, abs_lat, lon_rad, declination_degrees, hadley_edge, pressure_bar, itcz_strength, thermal_equator_lat):
    lat_rel = lat_grid - thermal_equator_lat
    abs_rel = np.abs(lat_rel)
    itcz = -16.5 * itcz_strength
    subtropical_center = hadley_edge + 0.10 * thermal_equator_lat * np.sign(lat_grid)
    subpolar_center = hadley_edge + 27.0 + 0.06 * thermal_equator_lat * np.sign(lat_grid)
    subtropical_high = 11.4 * np.exp(-(((abs_rel - subtropical_center) / 13.4) ** 2))
    subpolar_low = -8.4 * np.exp(-(((abs_rel - subpolar_center) / 18.5) ** 2))
    polar_high = 7.1 * np.exp(-(((abs_lat - 80.5) / 12.5) ** 2))
    thermal_anomaly = temp_c - float(np.mean(temp_c))
    # Warm summer land forms thermal lows; cold winter continents form highs.
    land_thermal = np.where(land, -0.90 * thermal_anomaly, -0.20 * thermal_anomaly)
    wave_lat = np.exp(-(((abs_rel - 38.0) / 25.0) ** 2))
    planetary_wave = 1.7 * np.sin(2.0 * lon_rad + math.radians(declination_degrees)) * wave_lat
    monsoon_wave = np.where(land, -1.4 * np.cos(lon_rad - math.radians(declination_degrees * 1.1)) * np.exp(-(((abs_rel - 21.0) / 20.0) ** 2)), 0.0)
    elevation_high = np.where(land, 0.00115 * elev_m, 0.0)
    pressure = 1013.25 * pressure_bar + itcz + subtropical_high + subpolar_low + polar_high + land_thermal + planetary_wave + monsoon_wave + elevation_high
    pressure = _blur_periodic(np, pressure, passes=2, lat_weight=0.12, lon_weight=0.16)
    return pressure.astype(np.float32)


def _smoothstep_np(np, edge0, edge1, x):
    t = np.clip((x - edge0) / max(1.0e-6, edge1 - edge0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _wind_from_pressure(np, pressure, lat_grid, rotation, atmosphere, hadley_edge, itcz_strength, thermal_equator_lat):
    gy, gx = np.gradient(pressure)
    down_u = -gx
    down_v = -gy
    coriolis = clamp(24.0 / max(8.0, rotation.rotation_period_hours), 0.42, 2.2)
    f = np.sin(np.radians(lat_grid)) * coriolis
    turn = np.clip(np.abs(f), 0.05, 0.92)
    u_geo = down_u * (1.0 - 0.56 * turn) + (-down_v * np.sign(f)) * (0.56 * turn)
    v_geo = down_v * (1.0 - 0.56 * turn) + (down_u * np.sign(f)) * (0.56 * turn)

    rel_lat = lat_grid - thermal_equator_lat
    abs_rel = np.abs(rel_lat)
    trades_w = 1.0 - _smoothstep_np(np, hadley_edge - 5.0, hadley_edge + 7.0, abs_rel)
    westerly_w = _smoothstep_np(np, hadley_edge - 6.0, hadley_edge + 8.0, abs_rel) * (1.0 - _smoothstep_np(np, hadley_edge + 27.0, hadley_edge + 43.0, abs_rel))
    polar_easterly_w = _smoothstep_np(np, hadley_edge + 31.0, hadley_edge + 48.0, abs_rel)
    zonal_strength = clamp(0.76 + 0.33 * coriolis, 0.5, 1.55) / clamp(0.85 + 0.10 * atmosphere.pressure_bar, 0.8, 1.25)
    hemi = np.sign(np.where(np.abs(lat_grid) < 0.1, rel_lat + 0.1, lat_grid))
    background_u = zonal_strength * (-1.05 * trades_w + 0.98 * westerly_w - 0.66 * polar_easterly_w)
    # Surface branch converges into the distorted ITCZ rather than into a fixed equator.
    convergence_v = -np.sign(rel_lat) * (0.28 + 0.22 * itcz_strength) * trades_w
    mid_v = -hemi * 0.16 * westerly_w + hemi * 0.12 * polar_easterly_w
    background_v = convergence_v + mid_v
    u_out = 0.46 * background_u + 0.54 * u_geo
    v_out = 0.46 * background_v + 0.54 * v_geo
    speed = np.sqrt(u_out * u_out + v_out * v_out)
    scale = np.maximum(0.38, np.quantile(speed, 0.92))
    u_out = np.clip(u_out / scale, -2.25, 2.25)
    v_out = np.clip(v_out / scale, -2.25, 2.25)
    u_out = _blur_periodic(np, u_out, passes=1, lat_weight=0.10, lon_weight=0.14)
    v_out = _blur_periodic(np, v_out, passes=1, lat_weight=0.10, lon_weight=0.14)
    return u_out.astype(np.float32), v_out.astype(np.float32)


def _moisture_transport_wind(*, np, wind_u, wind_v, pressure, lat_grid, abs_lat, hadley_edge, itcz_strength, thermal_equator_lat, land, dist_km):
    """Derive moisture-bearing lower-tropospheric flow from surface wind.

    Moisture transport is smoother and more ocean-recharge-oriented than the
    surface wind map.  This separation prevents a single hard latitude transition
    from simultaneously defining the wind diagnostic, moisture advection, and
    rainfall/storm-track field.
    """
    gy, gx = np.gradient(pressure)
    rel_lat = lat_grid - thermal_equator_lat
    abs_rel = np.abs(rel_lat)
    trades_w = 1.0 - _smoothstep_np(np, hadley_edge - 3.0, hadley_edge + 9.0, abs_rel)
    westerly_w = _smoothstep_np(np, hadley_edge - 4.0, hadley_edge + 10.0, abs_rel) * (1.0 - _smoothstep_np(np, hadley_edge + 28.0, hadley_edge + 47.0, abs_rel))
    # Gentle low-pressure inflow and ITCZ convergence for moisture; no hard band edge.
    inflow_u = -gx
    inflow_v = -gy - 0.22 * np.sign(rel_lat) * itcz_strength * trades_w
    # Mid-latitude storms carry moisture mostly west-to-east but with poleward/equatorward meanders.
    storm_u = 0.72 * westerly_w
    storm_v = -0.18 * np.sign(lat_grid) * westerly_w
    coast_fetch = np.clip(np.exp(-dist_km / 1500.0), 0.0, 1.0)
    u = 0.70 * wind_u + 0.22 * inflow_u + storm_u * (0.35 + 0.45 * coast_fetch)
    v = 0.70 * wind_v + 0.22 * inflow_v + storm_v * (0.35 + 0.45 * coast_fetch)
    # Land friction damps speed but should not erase direction.\n
    friction = np.where(land, 0.90 - 0.10 * np.clip(dist_km / 1600.0, 0.0, 1.0), 1.0)
    u *= friction; v *= friction
    speed = np.sqrt(u*u + v*v)
    scale = np.maximum(0.35, np.quantile(speed, 0.94))
    u = np.clip(u / scale, -2.2, 2.2)
    v = np.clip(v / scale, -2.2, 2.2)
    return _blur_periodic(np, u, passes=2, lat_weight=0.11, lon_weight=0.16), _blur_periodic(np, v, passes=2, lat_weight=0.11, lon_weight=0.16)


def _storm_track_moisture_field(*, np, lat_grid, abs_lat, hadley_edge, thermal_equator_lat, pressure, land, dist_km, ocean_distance_factor):
    rel_lat = lat_grid - thermal_equator_lat * 0.40
    # Broad storm-track envelope centered near the subpolar low / polar front.
    north_track = np.exp(-(((lat_grid - (hadley_edge + 25.0 + np.maximum(thermal_equator_lat, 0.0) * 0.10)) / 16.5) ** 2))
    south_track = np.exp(-(((lat_grid + (hadley_edge + 25.0 + np.maximum(-thermal_equator_lat, 0.0) * 0.10)) / 16.5) ** 2))
    envelope = north_track + south_track
    low_pressure = np.clip((float(np.mean(pressure)) - pressure) / 18.0, 0.0, 1.6)
    ocean_recharge = np.where(land, 0.35 + 0.65 * np.exp(-dist_km / 2600.0), 1.0)
    field = envelope * (0.55 + 0.45 * low_pressure) * ocean_recharge
    field = _blur_periodic(np, field, passes=5, lat_weight=0.14, lon_weight=0.22)
    return np.clip(field, 0.0, 1.8).astype(np.float32)


def _pressure_belt_class(np, lat_grid, hadley_edge: float, itcz_strength, thermal_equator_lat):
    rel_lat = lat_grid - thermal_equator_lat
    abs_rel = np.abs(rel_lat)
    cls = np.zeros(itcz_strength.shape, dtype=np.int16)
    cls[:, :] = 5  # background / weak belt
    cls[abs_rel < 1.5] = 1  # thermal equator axis
    cls[itcz_strength >= 0.70] = 2  # ITCZ low / convergence
    cls[(abs_rel >= hadley_edge - 8.0) & (abs_rel <= hadley_edge + 10.0)] = 3  # subtropical high
    cls[(abs_rel >= hadley_edge + 18.0) & (abs_rel <= hadley_edge + 42.0)] = 4  # subpolar low / storm track
    cls[np.abs(lat_grid[:, 0]) >= 70.0, :] = 6  # polar high
    cls[itcz_strength >= 0.85] = 2
    return cls.astype(np.int16)


def _ocean_currents_and_heat(*, np, wind_u, wind_v, land, ocean_like, lat_grid, abs_lat, dist_km, terrain):
    """Lightweight loop-first surface-current model.

    The field is built from broad zonal flow bands (equatorial flow,
    countercurrent, westerlies) and then nudged into basin-like gyre loops by
    weak rotational structure plus coastal boundary intensification.
    """
    h, w = ocean_like.shape
    lon_phase = np.linspace(-math.pi, math.pi, w, dtype=np.float32)[None, :]
    hemi = np.where(lat_grid >= 0.0, 1.0, -1.0)
    poleward_v_sign = np.where(lat_grid >= 0.0, -1.0, 1.0)
    equatorward_v_sign = -poleward_v_sign
    coslat = np.clip(np.cos(np.radians(lat_grid)), 0.0, 1.0)

    trade_band = np.exp(-(((abs_lat - 10.0) / 8.5) ** 2))
    counter_band = np.exp(-((lat_grid / 3.0) ** 2))
    westerly_band = np.exp(-(((abs_lat - 40.0) / 14.0) ** 2))
    subpolar_band = np.exp(-(((abs_lat - 58.0) / 10.5) ** 2))

    equatorial_u = -1.08 * trade_band + 0.68 * counter_band
    loop_u = hemi * (0.55 * np.cos(lon_phase) * trade_band + 0.82 * np.cos(lon_phase) * westerly_band - 0.36 * np.cos(lon_phase) * subpolar_band)
    loop_v = coslat * (-0.42 * np.sin(lon_phase) * trade_band - 0.74 * np.sin(lon_phase) * westerly_band + 0.30 * np.sin(lon_phase) * subpolar_band)

    gyre_u = 0.42 * wind_u + equatorial_u + loop_u
    gyre_v = 0.36 * wind_v + loop_v
    gyre_u = np.where(ocean_like, gyre_u, 0.0)
    gyre_v = np.where(ocean_like, gyre_v, 0.0)

    scale = map_scale_for_terrain(terrain)
    coastal_weight = np.where(ocean_like, np.exp(-dist_km / max(180.0, 520.0 * scale.planet_radius_earth)), 0.0)
    west_land = np.roll(land, 3, axis=1).astype(np.float32) * coastal_weight
    east_land = np.roll(land, -3, axis=1).astype(np.float32) * coastal_weight
    western_boundary = west_land * np.exp(-(((abs_lat - 31.0) / 18.0) ** 2))
    eastern_boundary = east_land * np.exp(-(((abs_lat - 25.0) / 14.0) ** 2))
    gyre_v += 0.95 * western_boundary * poleward_v_sign
    gyre_v += 0.82 * eastern_boundary * equatorward_v_sign
    gyre_u += 0.18 * (east_land - west_land)

    for _ in range(7):
        neigh_u = (np.roll(gyre_u, 1, axis=1) + np.roll(gyre_u, -1, axis=1) + np.vstack([gyre_u[0:1, :], gyre_u[:-1, :]]) + np.vstack([gyre_u[1:, :], gyre_u[-1:, :]])) / 4.0
        neigh_v = (np.roll(gyre_v, 1, axis=1) + np.roll(gyre_v, -1, axis=1) + np.vstack([gyre_v[0:1, :], gyre_v[:-1, :]]) + np.vstack([gyre_v[1:, :], gyre_v[-1:, :]])) / 4.0
        gyre_u = np.where(ocean_like, 0.76 * gyre_u + 0.24 * neigh_u, 0.0)
        gyre_v = np.where(ocean_like, 0.76 * gyre_v + 0.24 * neigh_v, 0.0)

    poleward_strength = np.clip(gyre_v * poleward_v_sign, 0.0, 2.4)
    equatorward_strength = np.clip(gyre_v * equatorward_v_sign, 0.0, 2.4)
    warm_branch = western_boundary * poleward_strength
    cold_branch = eastern_boundary * equatorward_strength
    warm_equatorial = np.clip(-gyre_u, 0.0, 2.0) * np.exp(-((abs_lat - 8.0) / 11.0) ** 2)
    current_heat = 2.55 * warm_branch - 2.35 * cold_branch + 0.55 * warm_equatorial
    for _ in range(3):
        current_heat = 0.52 * current_heat + 0.24 * np.roll(current_heat, 1, axis=1) + 0.24 * np.roll(current_heat, -1, axis=1)
    current_heat = np.where(ocean_like, current_heat, 0.0)
    return gyre_u.astype(np.float32), gyre_v.astype(np.float32), np.clip(current_heat, -5.4, 5.4).astype(np.float32)

def _moisture_and_precipitation(*, np, temp_c, pressure, wind_u, wind_v, current_heat, storm_track_moisture, land, water, ocean_like, inland_water, inland_water_source_factor, small_lake_buffer, elev_m, gx, gy, relief_norm, lat_grid, abs_lat, dist_km, atmosphere, vapor_factor, pressure_factor, hadley_edge, itcz_strength, cold_current_influence=None, coastal_upwelling=None, coastal_desert_potential=None):
    if cold_current_influence is None:
        cold_current_influence = np.zeros_like(temp_c, dtype=np.float32)
    if coastal_upwelling is None:
        coastal_upwelling = np.zeros_like(temp_c, dtype=np.float32)
    if coastal_desert_potential is None:
        coastal_desert_potential = np.zeros_like(temp_c, dtype=np.float32)
    warm_water = np.clip((temp_c + 5.0) / 34.0, 0.0, 1.5)
    ocean_source = np.where(ocean_like, 0.96 + 0.92 * warm_water + 0.14 * np.clip(current_heat, 0.0, 5.0), 0.0)
    # Small inland lakes must not act like mini-oceans.  Their source strength is
    # capped by water-body area/fetch and is then buffered by background aridity.
    inland_source = np.where(inland_water, (0.16 + 0.62 * warm_water) * inland_water_source_factor, 0.0)
    land_source = np.where(land, 0.082 * np.clip((temp_c - 1.0) / 28.0, 0.0, 1.0), 0.0)
    source = (ocean_source + inland_source + land_source) * vapor_factor

    moisture = _advect_scalar(np, source.astype(np.float32), wind_u, wind_v, land, iterations=82)
    coast_recharge = np.exp(-dist_km / max(760.0, 2450.0))
    storm_lift = np.clip(storm_track_moisture, 0.0, 1.8)
    long_range_recharge = storm_lift * np.exp(-dist_km / max(1700.0, 5200.0))
    # Small lakes should be nearly climate-neutral: enough local humidity to
    # avoid an artificial dry moat, but far below the strength needed to create
    # a wet biome island in a desert.
    neutral_lake_buffer = np.where(land, 0.030 * small_lake_buffer * vapor_factor, 0.0)
    moisture = moisture + np.where(land, (0.34 * coast_recharge + 0.17 * long_range_recharge) * vapor_factor, 0.0) + neutral_lake_buffer
    moisture = moisture * (0.97 + 0.13 * pressure_factor)

    low_pressure_lift = np.clip((np.mean(pressure) - pressure) / 18.0, 0.0, 1.8)
    subtropical_subsidence = np.exp(-(((abs_lat - hadley_edge) / 9.5) ** 2))
    directional_slope = -(gx * wind_u + gy * wind_v)
    slope_scale = max(1.0, float(np.quantile(np.abs(directional_slope[land]), 0.92)) if np.any(land) else 1.0)
    windward = np.clip(directional_slope / slope_scale, 0.0, 1.0)
    leeward = np.clip(-directional_slope / slope_scale, 0.0, 1.0)
    moisture_available = np.clip(moisture / 0.95, 0.0, 1.8)
    orographic = 0.18 + 1.34 * windward * relief_norm * (0.42 + moisture_available)
    ridge_extraction = np.clip(windward * relief_norm * moisture_available, 0.0, 1.8)
    local_leeward = np.clip(leeward * relief_norm, 0.0, 1.0)
    # A rain shadow is a downwind drying plume after moisture is extracted on
    # the windward side of a barrier.  The previous local-only leeward test was
    # too hard to interpret near high windward slopes.
    shadow_source = np.clip(0.68 * local_leeward + 0.88 * ridge_extraction, 0.0, 2.3)
    shadow_plume = _advect_scalar(np, shadow_source.astype(np.float32), wind_u, wind_v, land, iterations=28)
    inland_weight = np.clip(1.0 - np.exp(-dist_km / 700.0), 0.0, 1.0)
    rain_shadow_strength = np.clip((0.34 * local_leeward + 0.66 * shadow_plume) * inland_weight, 0.0, 1.0)
    rain_shadow_strength = np.clip(rain_shadow_strength * (1.0 - 0.55 * small_lake_buffer), 0.0, 1.0)
    rain_shadow_factor = np.clip(1.0 - 0.39 * rain_shadow_strength, 0.56, 1.12)
    monsoon = np.where(land, np.clip((1013.25 * pressure_factor - pressure) / 18.0, 0.0, 1.2) * np.clip(moisture - 0.50, 0.0, 1.5), 0.0)
    cold_current_coastal = np.where(land, np.clip(cold_current_influence, 0.0, 1.8) * np.exp(-(((abs_lat - 24.0) / 13.0) ** 2)), 0.0)
    upwelling_coastal = np.where(land, np.clip(coastal_upwelling, 0.0, 1.5) * np.exp(-(((abs_lat - 23.0) / 14.0) ** 2)), 0.0)
    coastal_desert = np.where(land, np.clip(coastal_desert_potential, 0.0, 1.4), 0.0)
    lift = 0.56 + 0.66 * low_pressure_lift + 0.56 * itcz_strength + 0.42 * storm_lift + orographic + 0.42 * monsoon
    suppression = (1.0 - 0.31 * subtropical_subsidence) * (1.0 - 0.24 * cold_current_coastal) * (1.0 - 0.18 * upwelling_coastal) * (1.0 - 0.22 * coastal_desert)
    convection = np.clip((temp_c + 8.0) / 34.0, 0.12, 1.35)
    precip = 980.0 * moisture * lift * suppression * convection * (0.90 + 0.15 * pressure_factor) * rain_shadow_factor
    ocean_convergence = 1.0 + 0.20 * itcz_strength + 0.12 * storm_lift - 0.14 * subtropical_subsidence
    precip = np.where(water, precip * ocean_convergence * 0.95, precip)
    precip = np.clip(precip, 10.0, 7500.0)
    cloud = np.clip((moisture * 0.38 + precip / 4300.0), 0.0, 1.0)
    return moisture.astype(np.float32), precip.astype(np.float32), cloud.astype(np.float32), np.clip(orographic, 0.0, 2.0).astype(np.float32), rain_shadow_strength.astype(np.float32)


def _empty_moisture_components(np, height: int, width: int):
    zero = np.zeros((height, width), dtype=np.float32)
    return {
        "trade_wind_moisture": zero,
        "monsoon_moisture": zero,
        "frontal_moisture": zero,
        "orographic_precip_potential": zero,
        "coastal_dryness": zero,
    }


def _moisture_and_precipitation_v2(*, np, temp_c, pressure, wind_u, wind_v, current_heat, warm_current_influence, cold_current_influence, coastal_upwelling, coastal_desert_potential, storm_track_moisture, land, water, ocean_like, inland_water, inland_water_source_factor, small_lake_buffer, elev_m, gx, gy, relief_norm, lat_grid, abs_lat, dist_km, atmosphere, vapor_factor, pressure_factor, hadley_edge, itcz_strength):
    """Component-based seasonal_v5 moisture and rainfall pass.

    This keeps the old annual ClimateMap contract, but internally separates the
    moisture budget into named review layers so rainfall errors can be diagnosed
    without guessing which helper field caused them: trade-wind moisture, monsoon
    moisture, frontal/storm-track moisture, orographic extraction, and coastal
    cold-current dryness.
    """
    warm_water = np.clip((temp_c + 5.0) / 34.0, 0.0, 1.5)
    warm_current = np.clip(warm_current_influence, 0.0, 1.8)
    cold_current = np.clip(cold_current_influence, 0.0, 1.8)
    upwelling = np.clip(coastal_upwelling, 0.0, 1.6)
    coastal_desert = np.clip(coastal_desert_potential, 0.0, 1.5)

    # Cold eastern-boundary currents and upwelling reduce evaporation/fetch while
    # warm western-boundary currents boost it.  This is deliberately applied at
    # the source before advection, not only as a final rainfall multiplier.
    ocean_evap_modifier = np.clip(1.0 + 0.18 * warm_current - 0.22 * cold_current - 0.18 * upwelling, 0.50, 1.35)
    ocean_source = np.where(ocean_like, (0.94 + 0.92 * warm_water + 0.12 * np.clip(current_heat, 0.0, 5.0)) * ocean_evap_modifier, 0.0)
    inland_source = np.where(inland_water, (0.14 + 0.56 * warm_water) * inland_water_source_factor, 0.0)
    land_source = np.where(land, 0.075 * np.clip((temp_c - 1.0) / 28.0, 0.0, 1.0), 0.0)
    source = (ocean_source + inland_source + land_source) * vapor_factor

    low_pressure_pull = np.clip((1013.25 * pressure_factor - pressure) / 20.0, 0.0, 1.5)
    subtropical_subsidence = np.exp(-(((abs_lat - hadley_edge) / 10.0) ** 2)).astype(np.float32)
    trade_band = np.exp(-(((abs_lat - 15.0) / 13.0) ** 2)).astype(np.float32) * np.clip(1.0 - 0.38 * itcz_strength, 0.34, 1.0)
    frontal_band = np.clip(storm_track_moisture, 0.0, 1.8)
    coast_recharge = np.exp(-dist_km / max(720.0, 2400.0)).astype(np.float32)
    long_range_recharge = frontal_band * np.exp(-dist_km / max(1800.0, 5600.0)).astype(np.float32)

    base_moisture = _advect_scalar(np, source.astype(np.float32), wind_u, wind_v, land, iterations=72)
    trade_moisture = _advect_scalar(np, (source * trade_band).astype(np.float32), wind_u, wind_v, land, iterations=64)
    frontal_moisture = _advect_scalar(np, (source * frontal_band * (0.72 + 0.28 * ocean_like.astype(np.float32))).astype(np.float32), wind_u, wind_v, land, iterations=78)

    monsoon_source = np.where(land, low_pressure_pull * coast_recharge * np.clip((temp_c + 3.0) / 30.0, 0.0, 1.2), 0.0)
    monsoon_moisture = np.clip(0.48 * base_moisture * monsoon_source + 0.28 * trade_moisture * np.clip(low_pressure_pull, 0.0, 1.2), 0.0, 2.4)

    neutral_lake_buffer = np.where(land, 0.028 * small_lake_buffer * vapor_factor, 0.0)
    moisture = (
        0.66 * base_moisture
        + 0.24 * trade_moisture
        + 0.22 * frontal_moisture
        + 0.28 * monsoon_moisture
        + np.where(land, (0.30 * coast_recharge + 0.19 * long_range_recharge) * vapor_factor, 0.0)
        + neutral_lake_buffer
    )
    moisture = np.clip(moisture * (0.96 + 0.13 * pressure_factor), 0.0, 4.5)

    low_pressure_lift = np.clip((np.mean(pressure) - pressure) / 18.0, 0.0, 1.8)
    directional_slope = -(gx * wind_u + gy * wind_v)
    slope_scale = max(1.0, float(np.quantile(np.abs(directional_slope[land]), 0.92)) if np.any(land) else 1.0)
    windward = np.clip(directional_slope / slope_scale, 0.0, 1.0)
    leeward = np.clip(-directional_slope / slope_scale, 0.0, 1.0)
    moisture_available = np.clip(moisture / 0.95, 0.0, 1.9)

    # Stronger, more localized inland windward precipitation than seasonal_v4.
    orographic = 0.14 + 1.58 * windward * relief_norm * (0.36 + moisture_available)
    orographic_precip_potential = np.clip(orographic * moisture_available * (0.55 + 0.45 * relief_norm), 0.0, 3.0)

    ridge_extraction = np.clip(windward * relief_norm * moisture_available, 0.0, 2.0)
    local_leeward = np.clip(leeward * relief_norm, 0.0, 1.0)
    shadow_source = np.clip(0.48 * local_leeward + 0.98 * ridge_extraction, 0.0, 2.5)
    shadow_plume = _advect_scalar(np, shadow_source.astype(np.float32), wind_u, wind_v, land, iterations=36)
    inland_weight = np.clip(1.0 - np.exp(-dist_km / 650.0), 0.0, 1.0)
    rain_shadow_strength = np.clip((0.25 * local_leeward + 0.75 * shadow_plume) * inland_weight, 0.0, 1.0)
    rain_shadow_strength = np.clip(rain_shadow_strength * (1.0 - 0.50 * small_lake_buffer), 0.0, 1.0)

    cold_subtropical = np.exp(-(((abs_lat - 23.0) / 13.5) ** 2)).astype(np.float32)
    coastal_dryness = np.where(land, np.clip((0.58 * cold_current + 0.72 * upwelling + 0.82 * coastal_desert) * cold_subtropical, 0.0, 1.6), 0.0)
    coastal_dryness = np.clip(coastal_dryness * np.exp(-dist_km / 520.0), 0.0, 1.6)

    monsoon_lift = np.clip(monsoon_moisture * (0.72 + 0.28 * low_pressure_pull), 0.0, 1.7)
    trade_lift = np.clip(trade_moisture * (0.36 + 0.64 * itcz_strength), 0.0, 1.45)
    frontal_lift = np.clip(frontal_band * (0.70 + 0.30 * frontal_moisture), 0.0, 1.9)
    lift = 0.50 + 0.68 * low_pressure_lift + 0.60 * itcz_strength + 0.42 * frontal_lift + 0.30 * trade_lift + orographic + 0.36 * monsoon_lift
    suppression = (1.0 - 0.29 * subtropical_subsidence) * (1.0 - 0.34 * coastal_dryness) * (1.0 - 0.29 * rain_shadow_strength)
    suppression = np.clip(suppression, 0.32, 1.12)
    convection = np.clip((temp_c + 8.0) / 34.0, 0.12, 1.35)
    precip = 905.0 * moisture * lift * suppression * convection * (0.90 + 0.15 * pressure_factor)
    ocean_convergence = 1.0 + 0.20 * itcz_strength + 0.13 * frontal_band - 0.14 * subtropical_subsidence
    precip = np.where(water, precip * ocean_convergence * 0.95, precip)
    precip = np.clip(precip, 10.0, 7800.0)
    cloud = np.clip((moisture * 0.35 + precip / 4500.0), 0.0, 1.0)

    components = {
        "trade_wind_moisture": np.clip(trade_moisture, 0.0, 4.5).astype(np.float32),
        "monsoon_moisture": np.clip(monsoon_moisture, 0.0, 4.5).astype(np.float32),
        "frontal_moisture": np.clip(frontal_moisture, 0.0, 4.5).astype(np.float32),
        "orographic_precip_potential": np.clip(orographic_precip_potential, 0.0, 4.5).astype(np.float32),
        "coastal_dryness": np.clip(coastal_dryness, 0.0, 2.0).astype(np.float32),
    }
    return moisture.astype(np.float32), precip.astype(np.float32), cloud.astype(np.float32), np.clip(orographic, 0.0, 2.2).astype(np.float32), rain_shadow_strength.astype(np.float32), components


def _advect_scalar(np, source, wind_u, wind_v, land, iterations: int = 48):
    moisture = source.copy()
    rough_decay = np.where(land, 0.972, 0.992).astype(np.float32)
    for _ in range(iterations):
        west = np.roll(moisture, 1, axis=1)
        east = np.roll(moisture, -1, axis=1)
        north = np.vstack([moisture[0:1, :], moisture[:-1, :]])
        south = np.vstack([moisture[1:, :], moisture[-1:, :]])
        wu_pos = np.clip(wind_u, 0.0, None)
        wu_neg = np.clip(-wind_u, 0.0, None)
        wv_pos = np.clip(wind_v, 0.0, None)
        wv_neg = np.clip(-wind_v, 0.0, None)
        total = wu_pos + wu_neg + wv_pos + wv_neg + 0.35
        upstream = (west * wu_pos + east * wu_neg + north * wv_pos + south * wv_neg + moisture * 0.35) / total
        moisture = np.maximum(source * 0.82, source * 0.24 + upstream * rough_decay)
    return np.clip(moisture, 0.0, 4.2)


def _synthesize_monthly_fields(np, seasonal_temp_stack, seasonal_precip_stack, lat_grid, land, dist_km):
    summer = seasonal_temp_stack[0]
    equinox = seasonal_temp_stack[1]
    winter = seasonal_temp_stack[2]
    p_summer = seasonal_precip_stack[0]
    p_equinox = seasonal_precip_stack[1]
    p_winter = seasonal_precip_stack[2]
    thermal_lag = np.where(land, 0.15 + 0.18 * np.clip(dist_km / 1600.0, 0.0, 1.0), 0.70)
    temp_months = []
    precip_months = []
    for month in range(12):
        phase = np.sin(2.0 * math.pi * (month - 2.0 - 1.2 * thermal_lag) / 12.0)
        summer_w = np.clip(phase, 0.0, 1.0)
        winter_w = np.clip(-phase, 0.0, 1.0)
        equinox_w = 1.0 - np.maximum(summer_w, winter_w)
        temp = equinox * equinox_w + summer * summer_w + winter * winter_w
        precip = (p_equinox * equinox_w + p_summer * summer_w + p_winter * winter_w) / 12.0
        temp_months.append(temp.astype(np.float32))
        precip_months.append(np.clip(precip, 1.0, 1250.0).astype(np.float32))
    return np.stack(temp_months, axis=0), np.stack(precip_months, axis=0)


def _koppen_from_monthly(np, monthly_temp, monthly_precip, land, lat_grid):
    # Numeric codes correspond to code_labels defined in generate_seasonal_v2_climate.
    code = np.zeros(land.shape, dtype=np.uint8)  # O
    annual_temp = np.mean(monthly_temp, axis=0)
    annual_precip = np.sum(monthly_precip, axis=0)
    warmest = np.max(monthly_temp, axis=0)
    coldest = np.min(monthly_temp, axis=0)
    driest = np.min(monthly_precip, axis=0)
    months_ge_10 = np.sum(monthly_temp >= 10.0, axis=0)

    nh = lat_grid >= 0.0
    summer_idx_nh = [3, 4, 5, 6, 7, 8]
    winter_idx_nh = [9, 10, 11, 0, 1, 2]
    summer_stack = np.where(nh[None, :, :], monthly_precip[summer_idx_nh], monthly_precip[winter_idx_nh])
    winter_stack = np.where(nh[None, :, :], monthly_precip[winter_idx_nh], monthly_precip[summer_idx_nh])
    summer_p = np.sum(summer_stack, axis=0)
    winter_p = np.sum(winter_stack, axis=0)
    summer_fraction = summer_p / np.maximum(annual_precip, 1.0)
    driest_summer = np.min(summer_stack, axis=0)
    wettest_summer = np.max(summer_stack, axis=0)
    driest_winter = np.min(winter_stack, axis=0)
    wettest_winter = np.max(winter_stack, axis=0)

    code[land & (warmest < 0.0)] = 30  # EF
    code[land & (code == 0) & (warmest < 10.0)] = 29  # ET

    remaining = land & (code == 0)
    arid_adjust = np.where(summer_fraction >= 0.70, 280.0, np.where(summer_fraction >= 0.30, 140.0, 0.0))
    arid_threshold = np.maximum(90.0, 20.0 * np.maximum(annual_temp, 0.0) + arid_adjust)
    desert = remaining & (annual_precip < 0.50 * arid_threshold)
    steppe = remaining & (~desert) & (annual_precip < arid_threshold)
    code[desert & (annual_temp >= 18.0)] = 4
    code[desert & (annual_temp < 18.0)] = 5
    code[steppe & (annual_temp >= 18.0)] = 6
    code[steppe & (annual_temp < 18.0)] = 7

    remaining = land & (code == 0)
    tropical = remaining & (coldest >= 18.0)
    monsoon_limit = np.maximum(0.0, 100.0 - annual_precip / 25.0)
    code[tropical & (driest >= 60.0)] = 1
    code[tropical & (driest < 60.0) & (driest >= monsoon_limit)] = 2
    code[tropical & (code == 0)] = 3

    remaining = land & (code == 0)
    temperate = remaining & (coldest > 0.0) & (warmest >= 10.0)
    continental = remaining & (~temperate) & (warmest >= 10.0)
    dry_summer_t = (driest_summer < 40.0) & (driest_summer < wettest_winter / 3.0)
    dry_winter_t = (driest_winter < wettest_summer / 10.0)
    precip_f = ~(dry_summer_t | dry_winter_t)

    warm_a = warmest >= 22.0
    warm_b = (~warm_a) & (months_ge_10 >= 4)
    warm_c = (~warm_a) & (months_ge_10 >= 1) & (months_ge_10 <= 3)
    warm_d = coldest <= -38.0

    code[temperate & precip_f & warm_a] = 8   # Cfa
    code[temperate & precip_f & warm_b] = 9   # Cfb
    code[temperate & precip_f & warm_c] = 10  # Cfc
    code[temperate & dry_summer_t & warm_a] = 11  # Csa
    code[temperate & dry_summer_t & warm_b] = 12  # Csb
    code[temperate & dry_summer_t & warm_c] = 13  # Csc
    code[temperate & dry_winter_t & warm_a] = 14  # Cwa
    code[temperate & dry_winter_t & warm_b] = 15  # Cwb
    code[temperate & dry_winter_t & warm_c] = 16  # Cwc

    code[continental & precip_f & warm_a] = 17  # Dfa
    code[continental & precip_f & warm_b] = 18  # Dfb
    code[continental & precip_f & warm_c] = 19  # Dfc
    code[continental & precip_f & warm_d] = 20  # Dfd
    code[continental & dry_summer_t & warm_a] = 21  # Dsa
    code[continental & dry_summer_t & warm_b] = 22  # Dsb
    code[continental & dry_summer_t & warm_c] = 23  # Dsc
    code[continental & dry_summer_t & warm_d] = 24  # Dsd
    code[continental & dry_winter_t & warm_a] = 25  # Dwa
    code[continental & dry_winter_t & warm_b] = 26  # Dwb
    code[continental & dry_winter_t & warm_c] = 27  # Dwc
    code[continental & dry_winter_t & warm_d] = 28  # Dwd

    leftovers = land & (code == 0)
    code[leftovers & (annual_temp >= 18.0)] = 3
    code[leftovers & (annual_temp < 18.0)] = 9
    return code


def _koppen_aridity_threshold(np, monthly_temp, monthly_precip, lat_grid):
    annual_temp = np.mean(monthly_temp, axis=0)
    annual_precip = np.sum(monthly_precip, axis=0)
    nh = lat_grid >= 0.0
    nh_summer_p = np.sum(monthly_precip[3:9], axis=0)
    nh_winter_p = annual_precip - nh_summer_p
    summer_p = np.where(nh, nh_summer_p, nh_winter_p)
    summer_fraction = summer_p / np.maximum(annual_precip, 1.0)
    arid_adjust = np.where(summer_fraction >= 0.70, 280.0, np.where(summer_fraction >= 0.30, 140.0, 0.0))
    return np.maximum(90.0, 20.0 * np.maximum(annual_temp, 0.0) + arid_adjust).astype(np.float32)


def _water_body_fields(np, water, scale):
    h, w = water.shape
    total = max(1, h * w)
    km2_per_cell = max(1.0, scale.representative_km_per_cell * scale.representative_km_per_cell)
    try:
        from scipy import ndimage
        labels, count = ndimage.label(water, structure=np.array([[0,1,0],[1,1,1],[0,1,0]], dtype=np.uint8))
        # Merge components crossing the equirectangular seam.
        parent = list(range(count + 1))
        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        def union(a, b):
            if a and b:
                ra, rb = find(int(a)), find(int(b))
                if ra != rb:
                    parent[rb] = ra
        for r in range(h):
            union(labels[r, 0], labels[r, w - 1])
        if count:
            flat = labels.ravel()
            roots = np.zeros(count + 1, dtype=np.int32)
            for i in range(1, count + 1):
                roots[i] = find(i)
            flat_roots = roots[flat]
            counts = np.bincount(flat_roots, minlength=count + 1)
            area_cells = counts[flat_roots].reshape((h, w)).astype(np.float32)
        else:
            area_cells = np.zeros((h, w), dtype=np.float32)
    except Exception:
        # Conservative fallback: treat all water as one ocean-like body.
        count = 1 if bool(water.any()) else 0
        area_cells = np.where(water, float(water.sum()), 0.0).astype(np.float32)

    area_km2 = area_cells * km2_per_cell
    ocean_like_threshold_cells = max(2048.0, total * 0.010)
    ocean_like_threshold_km2 = 1_500_000.0 * max(0.65, scale.planet_radius_earth)
    ocean_like = water & ((area_cells >= ocean_like_threshold_cells) | (area_km2 >= ocean_like_threshold_km2))
    if water.any() and not ocean_like.any():
        ocean_like = water & (area_cells >= float(area_cells[water].max()))
    inland_water = water & ~ocean_like
    # Inland-water evaporation: tiny lakes have tiny climate influence; large
    # inland seas can matter, but still less than open oceans.
    size_t = np.clip((np.log10(np.maximum(area_km2, 1.0)) - math.log10(25_000.0)) / (math.log10(850_000.0) - math.log10(25_000.0)), 0.0, 1.0)
    inland_source_factor = np.where(inland_water, 0.015 + 0.55 * (size_t * size_t * (3.0 - 2.0 * size_t)), 0.0).astype(np.float32)
    large_components = int(len(set(np.unique(area_cells[ocean_like]).astype(int).tolist()))) if ocean_like.any() else 0
    inland_components = int(len(set(np.unique(area_cells[inland_water]).astype(int).tolist()))) if inland_water.any() else 0
    return {
        "ocean_like": ocean_like.astype(bool),
        "inland_source_factor": inland_source_factor,
        "water_area_cells": area_cells,
        "large_water_component_count": large_components,
        "inland_water_component_count": inland_components,
    }


def _distance_to_source_cells(np, source_mask):
    try:
        from scipy import ndimage
        source = np.asarray(source_mask, dtype=bool)
        h, w = source.shape
        if not source.any():
            return np.full((h, w), max(h, w), dtype=np.float32)
        # Wrap horizontally so cells near the date-line see ocean across the seam.
        inverted = ~source
        tiled = np.concatenate([inverted, inverted, inverted], axis=1)
        dist = ndimage.distance_transform_edt(tiled)
        return dist[:, w:2*w].astype(np.float32)
    except Exception:
        return np.where(source_mask, 0.0, 9999.0).astype(np.float32)



def _circulation_zone_class_for_season(np, lat_grid, tilt: float, hadley_edge: float, itcz_strength):
    """Season-specific circulation guide. Bands are guide latitudes, while ITCZ follows seasonal convergence."""
    abs_lat = np.abs(lat_grid)
    zone = np.zeros((lat_grid.shape[0], itcz_strength.shape[1]), dtype=np.int16)
    zone[:, :] = 5  # westerlies/default mid-lat flow
    zone[abs_lat[:, 0] < 1.0, :] = 1  # equator
    zone[(abs_lat[:, 0] >= 1.0) & (abs_lat[:, 0] < max(6.0, abs(tilt))), :] = 3  # tropics
    zone[(abs_lat[:, 0] >= hadley_edge - 5.0) & (abs_lat[:, 0] <= hadley_edge + 6.0), :] = 4  # subtropical highs
    zone[(abs_lat[:, 0] >= hadley_edge + 24.0) & (abs_lat[:, 0] <= hadley_edge + 38.0), :] = 6  # storm track / polar front
    zone[abs_lat[:, 0] >= 70.0, :] = 7  # polar cap
    zone[itcz_strength >= 0.74] = 2
    zone[abs_lat[:, 0] < 1.0, :] = 1
    return zone.astype(np.int16)


def _circulation_zone_class(np, lat_grid, tilt: float, hadley_edge: float, seasonal_itcz_stack):
    """Annual circulation guide using all three seasonal ITCZ/convergence states."""
    annual_itcz = np.max(seasonal_itcz_stack, axis=0)
    return _circulation_zone_class_for_season(np, lat_grid, tilt, hadley_edge, annual_itcz)


def _ocean_gyre_class(np, current_u, current_v, ocean_like, lat_grid):
    """Categorical ocean current/gyre guide for diagnostics."""
    abs_lat = np.abs(lat_grid)
    speed = np.sqrt(current_u * current_u + current_v * current_v)
    cls = np.zeros(ocean_like.shape, dtype=np.int16)
    active = ocean_like & (speed > 0.08)
    cls[active & (abs_lat < 8.0)] = 1  # equatorial current belt
    cls[active & (abs_lat >= 8.0) & (abs_lat < 45.0)] = 2  # subtropical gyre
    cls[active & (abs_lat >= 45.0) & (abs_lat < 67.0)] = 3  # subpolar gyre
    cls[active & (abs_lat >= 67.0)] = 4  # polar current
    cls[active & (current_v * np.sign(lat_grid) < -0.16) & (abs_lat < 55.0)] = 5  # poleward warm branch
    cls[active & (current_v * np.sign(lat_grid) > 0.16) & (abs_lat < 55.0)] = 6  # equatorward cold branch
    return cls.astype(np.int16)



def _ocean_current_path_class(np, current_u, current_v, current_heat, ocean_like, lat_grid):
    """Fallback current-path classes for seasonal_v2 loop currents."""
    abs_lat = np.abs(lat_grid)
    speed = np.sqrt(current_u * current_u + current_v * current_v)
    cls = np.zeros(ocean_like.shape, dtype=np.int16)
    active = ocean_like & (speed > 0.08)
    cls[ocean_like & ~active] = 8
    cls[active & (abs_lat < 13.0) & (current_u < -0.12)] = 1
    cls[active & (abs_lat < 5.0) & (current_u > 0.12)] = 2
    cls[active & (abs_lat >= 8.0) & (abs_lat < 45.0)] = np.where(cls[active & (abs_lat >= 8.0) & (abs_lat < 45.0)] == 0, 3, cls[active & (abs_lat >= 8.0) & (abs_lat < 45.0)])
    cls[active & (abs_lat >= 45.0) & (abs_lat < 67.0)] = 6
    cls[active & (current_v * np.sign(lat_grid) < -0.16) & (abs_lat < 55.0) & (current_heat > 0.15)] = 4
    cls[active & (current_v * np.sign(lat_grid) > 0.16) & (abs_lat < 55.0) & (current_heat < -0.15)] = 5
    return cls.astype(np.int16)


def _weighted_mode_class(np, stack, weights):
    """Return weighted modal class for a small seasonal class stack."""
    stack = np.asarray(stack, dtype=np.int16)
    weights = np.asarray(weights, dtype=np.float32)
    max_code = int(np.max(stack)) if stack.size else 0
    if max_code <= 0:
        return np.zeros(stack.shape[1:], dtype=np.int16)
    scores = []
    for code in range(max_code + 1):
        scores.append(np.sum((stack == code).astype(np.float32) * weights[:, None, None], axis=0))
    return np.argmax(np.stack(scores, axis=0), axis=0).astype(np.int16)

def _build_driver_maps(**kwargs):
    np = kwargs["np"]
    terrain = kwargs["terrain"]
    h, w = terrain.height, terrain.width
    stride = 1
    maps: dict[str, object] = {}

    def add(name: str, arr, scale: float = 1.0, dtype=np.int16, clip_min=None, clip_max=None):
        a = np.asarray(arr, dtype=np.float32) * float(scale)
        if clip_min is not None or clip_max is not None:
            lo = -32768 if clip_min is None else clip_min
            hi = 32767 if clip_max is None else clip_max
            a = np.clip(a, lo, hi)
        maps[name] = np.rint(a).astype(dtype)

    for idx, season in enumerate(SEASON_NAMES):
        add(f"temperature_{season}_c_x10", kwargs["seasonal_temp_stack"][idx], 10.0, np.int16)
        add(f"precipitation_{season}_mm", kwargs["seasonal_precip_stack"][idx], 1.0, np.int32, 0, 20000)
        add(f"pressure_{season}_hpa_x10", kwargs["seasonal_pressure_stack"][idx], 10.0, np.int32, 0, 50000)
        add(f"wind_u_{season}_x1000", kwargs["seasonal_wind_u_stack"][idx], 1000.0, np.int16, -5000, 5000)
        add(f"wind_v_{season}_x1000", kwargs["seasonal_wind_v_stack"][idx], 1000.0, np.int16, -5000, 5000)
        add(f"current_u_{season}_x1000", kwargs["seasonal_current_u_stack"][idx], 1000.0, np.int16, -5000, 5000)
        add(f"current_v_{season}_x1000", kwargs["seasonal_current_v_stack"][idx], 1000.0, np.int16, -5000, 5000)
        add(f"moisture_{season}_x1000", kwargs["seasonal_moisture_stack"][idx], 1000.0, np.int16, 0, 6000)
        add(f"itcz_{season}_x1000", kwargs["seasonal_itcz_stack"][idx], 1000.0, np.int16, 0, 3000)
        add(f"thermal_equator_{season}_x1000", kwargs["seasonal_thermal_equator_stack"][idx], 1000.0, np.int16, 0, 3000)
        add(f"pressure_belt_{season}_class", kwargs["seasonal_pressure_belt_stack"][idx], 1.0, np.int16, 0, 12)
        add(f"moisture_wind_u_{season}_x1000", kwargs["seasonal_moisture_wind_u_stack"][idx], 1000.0, np.int16, -5000, 5000)
        add(f"moisture_wind_v_{season}_x1000", kwargs["seasonal_moisture_wind_v_stack"][idx], 1000.0, np.int16, -5000, 5000)
        add(f"storm_track_moisture_{season}_x1000", kwargs["seasonal_storm_track_stack"][idx], 1000.0, np.int16, 0, 3000)
        add(f"circulation_zone_{season}_class", kwargs["seasonal_circulation_zone_stack"][idx], 1.0, np.int16, 0, 12)

    monthly_temp = kwargs["monthly_temp"]
    monthly_precip = kwargs["monthly_precip"]
    for idx in range(12):
        month = idx + 1
        add(f"monthly_temperature_{month:02d}_c_x10", monthly_temp[idx], 10.0, np.int16, -900, 900)
        add(f"monthly_precipitation_{month:02d}_mm", monthly_precip[idx], 1.0, np.int16, 0, 3000)

    add("warmest_month_index", np.argmax(monthly_temp, axis=0) + 1, 1.0, np.int16, 1, 12)
    add("wettest_month_index", np.argmax(monthly_precip, axis=0) + 1, 1.0, np.int16, 1, 12)
    add("driest_month_index", np.argmin(monthly_precip, axis=0) + 1, 1.0, np.int16, 1, 12)
    add("circulation_zone_class", kwargs["circulation_zone_class"], 1.0, np.int16, 0, 12)
    add("ocean_gyre_class", kwargs["ocean_gyre_class"], 1.0, np.int16, 0, 12)
    if kwargs.get("climate_mode_name") in {"seasonal_v3", "seasonal_v4", "seasonal_v5"}:
        add("ocean_basin_id", kwargs["ocean_basin_id"], 1.0, np.int32, 0, 100000)
        if "ocean_basin_kind" in kwargs:
            add("ocean_basin_kind", kwargs["ocean_basin_kind"], 1.0, np.int16, 0, 3)
        add("ocean_current_path_class", kwargs["ocean_current_path_class"], 1.0, np.int16, 0, 12)

    add("wind_u_annual_x1000", kwargs["annual_wind_u"], 1000.0, np.int16, -5000, 5000)
    add("wind_v_annual_x1000", kwargs["annual_wind_v"], 1000.0, np.int16, -5000, 5000)
    add("current_u_annual_x1000", kwargs["annual_current_u"], 1000.0, np.int16, -5000, 5000)
    add("current_v_annual_x1000", kwargs["annual_current_v"], 1000.0, np.int16, -5000, 5000)
    add("current_heat_annual_c_x10", kwargs["annual_current_heat"], 10.0, np.int16, -200, 200)
    if kwargs.get("climate_mode_name") in {"seasonal_v3", "seasonal_v4", "seasonal_v5"}:
        add("coastal_upwelling_x1000", kwargs["coastal_upwelling"], 1000.0, np.int16, 0, 3000)
        add("warm_current_influence_x1000", kwargs["warm_current_influence"], 1000.0, np.int16, 0, 3000)
        add("cold_current_influence_x1000", kwargs["cold_current_influence"], 1000.0, np.int16, 0, 3000)
        add("coastal_desert_potential_x1000", kwargs["coastal_desert_potential"], 1000.0, np.int16, 0, 3000)
    if kwargs.get("climate_mode_name") == "seasonal_v5":
        add("trade_wind_moisture_annual_x1000", kwargs["trade_wind_moisture"], 1000.0, np.int16, 0, 6000)
        add("monsoon_moisture_annual_x1000", kwargs["monsoon_moisture"], 1000.0, np.int16, 0, 6000)
        add("frontal_moisture_annual_x1000", kwargs["frontal_moisture"], 1000.0, np.int16, 0, 6000)
        add("orographic_precip_potential_x1000", kwargs["orographic_precip_potential"], 1000.0, np.int16, 0, 6000)
        add("coastal_dryness_x1000", kwargs["coastal_dryness"], 1000.0, np.int16, 0, 3000)
    add("moisture_annual_x1000", kwargs["annual_moisture"], 1000.0, np.int16, 0, 6000)
    add("orographic_lift_annual_x1000", kwargs["annual_orographic"], 1000.0, np.int16, 0, 5000)
    add("rain_shadow_annual_x1000", kwargs["annual_shadow"], 1000.0, np.int16, 0, 3000)
    add("itcz_annual_x1000", kwargs["annual_itcz"], 1000.0, np.int16, 0, 3000)
    add("thermal_equator_annual_x1000", kwargs["annual_thermal_equator"], 1000.0, np.int16, 0, 3000)
    add("moisture_wind_u_annual_x1000", kwargs["annual_moisture_wind_u"], 1000.0, np.int16, -5000, 5000)
    add("moisture_wind_v_annual_x1000", kwargs["annual_moisture_wind_v"], 1000.0, np.int16, -5000, 5000)
    add("storm_track_moisture_annual_x1000", kwargs["annual_storm_track"], 1000.0, np.int16, 0, 3000)
    add("aridity_index_x1000", kwargs["aridity_index"], 1000.0, np.int16, 0, 5000)
    add("inland_water_source_x1000", kwargs["inland_water_source_factor"], 1000.0, np.int16, 0, 2000)
    add("small_lake_neutral_buffer_x1000", kwargs["small_lake_buffer"], 1000.0, np.int16, 0, 1000)
    add("ocean_like_water_x1000", kwargs["ocean_like"].astype(np.float32), 1000.0, np.int16, 0, 1000)
    info = {
        "schema_version": 5,
        "source_width": int(w),
        "source_height": int(h),
        "width": int(w),
        "height": int(h),
        "stride": int(stride),
        "climate_mode": kwargs.get("climate_mode_name", "seasonal_v2"),
        "resolution_policy": "native climate-grid resolution unless future performance testing requires a cap",
        "value_encoding": "integer scaled; see map key suffixes",
        "seasons": list(SEASON_NAMES),
        "monthly_temperature_keys": [f"monthly_temperature_{m:02d}_c_x10" for m in range(1, 13)],
        "monthly_precipitation_keys": [f"monthly_precipitation_{m:02d}_mm" for m in range(1, 13)],
    }
    return maps, info
