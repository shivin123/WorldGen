"""Main Planet deep-dive profile models.

These objects summarize the physical, terrain, and first-pass climate state of
our selected Main Planet. The models are deliberately approximate: terrain is
procedural and climate is long-term average climate, not weather.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any


@dataclass
class RotationState:
    rotation_period_hours: float
    axial_tilt_degrees: float
    solar_day_hours: float
    year_length_days: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Atmosphere:
    pressure_bar: float
    nitrogen_fraction: float
    oxygen_fraction: float
    carbon_dioxide_ppm: float
    argon_fraction: float
    water_vapor_factor: float
    greenhouse_warming_k: float
    estimated_mean_surface_temp_k: float
    estimated_mean_surface_temp_c: float
    notes: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Hydrosphere:
    volatile_fraction: float
    ocean_fraction_target: float
    ocean_fraction_actual: float
    water_inventory_class: str
    ice_cap_tendency: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class GeologyState:
    internal_heat: float
    volcanism: float
    erosion: float
    mountain_factor: float
    crater_density: float
    surface_roughness: float
    geology_class: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TerrainMap:
    width: int
    height: int
    elevation_m: list[list[int]]
    is_land: list[list[bool]]
    min_elevation_m: int
    max_elevation_m: int
    mean_land_elevation_m: float
    mean_ocean_depth_m: float
    ocean_fraction: float
    land_fraction: float
    source: str = "procedural"
    # Radius of the body this equirectangular raster represents. Climate and
    # diagnostics use this to convert cell distances into physical kilometers.
    planet_radius_earth: float = 1.0
    # Optional low-resolution diagnostic rasters. These are intentionally
    # lightweight compared with the full terrain grid and are used for plate /
    # crust visual diagnostics. Codes are documented in visualization labels.
    tectonic_plate_id: list[list[int]] | None = None
    tectonic_boundary_class: list[list[int]] | None = None
    # Additional Stage 3C.2 low-resolution province/boundary diagnostics.
    # Values are integer-coded rasters kept at diagnostic resolution so large
    # terrain arrays remain compact in normal state files.
    tectonic_province_type: list[list[int]] | None = None
    tectonic_province_age_x1000: list[list[int]] | None = None
    tectonic_boundary_strength_x1000: list[list[int]] | None = None
    tectonic_boundary_width_x1000: list[list[int]] | None = None
    # Plate Terrain v1 native setup diagnostics. These are independent plate
    # seeding/crust-allocation fields used when terrain_generation_mode is
    # plate_tectonic_v1. They intentionally coexist with the legacy/procedural
    # province fields so the old backend can remain available.
    plate_tectonic_plate_type: list[list[int]] | None = None
    plate_tectonic_continental_crust_x1000: list[list[int]] | None = None
    plate_tectonic_craton_core_x1000: list[list[int]] | None = None
    plate_tectonic_microplate_x1000: list[list[int]] | None = None
    # Plate Terrain 10 native domain/topology diagnostics. Continent assembly IDs
    # distinguish coherent continents from island/terrane belts; topology problem
    # classes identify graph repairs such as enclosed single-neighbor plates.
    plate_tectonic_continent_assembly_id: list[list[int]] | None = None
    plate_tectonic_plate_topology_problem_class: list[list[int]] | None = None
    # Plate Terrain 2 native motion/boundary diagnostics. Velocity components
    # are signed -1000..1000 diagnostic vectors; speed and relative-motion
    # fields are 0..1000. Boundary class mirrors tectonic_boundary_class when
    # plate_tectonic_v1 is active, but is retained separately so the UI can
    # distinguish native plate-derived classes from legacy/proxy classes.
    plate_tectonic_velocity_x_x1000: list[list[int]] | None = None
    plate_tectonic_velocity_y_x1000: list[list[int]] | None = None
    plate_tectonic_speed_x1000: list[list[int]] | None = None
    plate_tectonic_convergence_x1000: list[list[int]] | None = None
    plate_tectonic_divergence_x1000: list[list[int]] | None = None
    plate_tectonic_transform_x1000: list[list[int]] | None = None
    plate_tectonic_boundary_class: list[list[int]] | None = None
    plate_tectonic_subduction_polarity: list[list[int]] | None = None
    # Plate Terrain 3 native ocean-floor diagnostics. These are derived from
    # the native plate-motion/boundary fields in plate_tectonic_v1 rather than
    # from the legacy/proxy bathymetry skeleton. They intentionally coexist with
    # the generic terrain_ocean_floor_* fields so the review UI can distinguish
    # native plate bathymetry from legacy procedural ocean-floor shaping.
    plate_tectonic_ocean_floor_class: list[list[int]] | None = None
    plate_tectonic_ocean_crust_age_x1000: list[list[int]] | None = None
    plate_tectonic_mid_ocean_ridge_x1000: list[list[int]] | None = None
    plate_tectonic_trench_x1000: list[list[int]] | None = None
    plate_tectonic_fracture_zone_x1000: list[list[int]] | None = None
    plate_tectonic_abyssal_plain_x1000: list[list[int]] | None = None
    plate_tectonic_seamount_x1000: list[list[int]] | None = None
    # Plate Terrain 4 native continental-terrain diagnostics. These are the
    # first plate-derived relief fields that can directly modify elevation in
    # plate_tectonic_v1 instead of merely annotating the legacy backend.
    plate_tectonic_orogeny_strength_x1000: list[list[int]] | None = None
    plate_tectonic_volcanic_arc_x1000: list[list[int]] | None = None
    plate_tectonic_continental_rift_x1000: list[list[int]] | None = None
    plate_tectonic_foreland_basin_x1000: list[list[int]] | None = None
    plate_tectonic_craton_shield_x1000: list[list[int]] | None = None
    plate_tectonic_accreted_terrane_x1000: list[list[int]] | None = None
    plate_tectonic_plateau_uplift_x1000: list[list[int]] | None = None
    plate_tectonic_sedimentary_plain_x1000: list[list[int]] | None = None
    plate_tectonic_landform_class: list[list[int]] | None = None
    plate_tectonic_relief_delta_m: list[list[int]] | None = None
    # Plate Terrain 5 native coast/shelf/island diagnostics. These are derived
    # from native plate continental crust, plate-margin class, subduction/arc,
    # rift, transform, and microplate fields. They let plate_tectonic_v1 begin
    # owning shelves and coast/island interpretation instead of using only
    # distance-from-land legacy coast heuristics.
    plate_tectonic_margin_class: list[list[int]] | None = None
    plate_tectonic_shelf_width_x1000: list[list[int]] | None = None
    plate_tectonic_active_margin_x1000: list[list[int]] | None = None
    plate_tectonic_passive_margin_x1000: list[list[int]] | None = None
    plate_tectonic_rifted_margin_x1000: list[list[int]] | None = None
    plate_tectonic_island_arc_x1000: list[list[int]] | None = None
    plate_tectonic_coastal_plain_x1000: list[list[int]] | None = None
    plate_tectonic_coast_ruggedness_x1000: list[list[int]] | None = None
    plate_tectonic_island_origin_class: list[list[int]] | None = None
    plate_tectonic_margin_profile_class: list[list[int]] | None = None
    plate_tectonic_coast_delta_m: list[list[int]] | None = None
    # Plate Terrain 10 final integration/readiness diagnostics. These summarize
    # how much of plate_tectonic_v1 is native versus still dependent on the
    # legacy/texture fallback, where hydrology is ready to consume the terrain, and
    # what class of remaining plate-mode problem a cell belongs to.
    plate_tectonic_backend_integration_x1000: list[list[int]] | None = None
    plate_tectonic_hydrology_readiness_x1000: list[list[int]] | None = None
    plate_tectonic_legacy_dependency_x1000: list[list[int]] | None = None
    plate_tectonic_problem_class: list[list[int]] | None = None

    # Plate Terrain 11 drainage-ready landform refinement. These fields are
    # produced after margin profiles so hydrology can later consume explicit
    # valley corridors, inland basins, and lake-candidate depressions instead of
    # relying on concentric coast-to-interior elevation gradients.
    plate_tectonic_valley_corridor_x1000: list[list[int]] | None = None
    plate_tectonic_inland_basin_x1000: list[list[int]] | None = None
    plate_tectonic_lake_candidate_x1000: list[list[int]] | None = None
    plate_tectonic_terrain_detail_x1000: list[list[int]] | None = None
    plate_tectonic_drainage_ready_delta_m: list[list[int]] | None = None
    # Stage 3C.3 low-resolution relief diagnostics. These represent the
    # generator's internal mountain, basin, rift, shield/highland, and plateau
    # influence fields rather than classes inferred only after final elevation.
    terrain_mountain_strength_x1000: list[list[int]] | None = None
    terrain_basin_field_x1000: list[list[int]] | None = None
    terrain_rift_field_x1000: list[list[int]] | None = None
    terrain_interior_relief_x1000: list[list[int]] | None = None
    terrain_shield_highland_x1000: list[list[int]] | None = None
    terrain_plateau_x1000: list[list[int]] | None = None
    # Stage 3C.4 coastline/shelf/island diagnostics.  The style and origin
    # rasters are integer-coded classes; the shelf/coast ruggedness fields are
    # 0..1000 influence fields.  They let the coast review page explain why a
    # margin is smooth/passive, rugged/fjorded, rifted/gulfed, volcanic/arc, or
    # island-rich rather than only showing a final coastline mask.
    terrain_coast_style_class: list[list[int]] | None = None
    terrain_shelf_width_x1000: list[list[int]] | None = None
    # v3 shelf/apron diagnostics.  These are continuous-field diagnostics, not
    # terrain-rule switches: they explain where the final model thinks
    # continental crust continues offshore, where shelf support exists, and
    # whether water is shelf, slope/rise, or abyssal.
    terrain_submerged_continental_crust_x1000: list[list[int]] | None = None
    terrain_continental_shelf_support_x1000: list[list[int]] | None = None
    terrain_shelf_depth_target_x1000: list[list[int]] | None = None
    terrain_shelf_zone_class: list[list[int]] | None = None
    # Update 27D terrain-correction diagnostics. Lake depth limit stores the
    # local allowed enclosed-water depth as 0..1000 normalized against 4000 m;
    # plate component class marks disconnected/reassigned/promoted final plate
    # fragments after the diagnostic plate-ID contiguity cleanup.
    terrain_lake_depth_limit_x1000: list[list[int]] | None = None
    terrain_final_plate_component_class: list[list[int]] | None = None
    terrain_ripple_artifact_risk_x1000: list[list[int]] | None = None
    # v4 experimental topology/island diagnostics.  These are diagnostic rasters
    # for the isolated plate_history_v4 branch; v3 and older modes leave them unset.
    terrain_v4_boundary_deformation_x1000: list[list[int]] | None = None
    terrain_v4_volcanic_island_support_x1000: list[list[int]] | None = None
    terrain_v4_rift_cut_support_x1000: list[list[int]] | None = None
    terrain_v4_mountain_branch_support_x1000: list[list[int]] | None = None
    terrain_v4_topology_class: list[list[int]] | None = None
    terrain_v4_island_chain_class: list[list[int]] | None = None
    # Update 31 v4 interpretability diagnostics. These classify the experimental
    # v4 boundary/orogen network after deformation, rift cuts, microplate cleanup,
    # and volcanic-island shaping so topology feedback is not inferred from one
    # color map alone.
    terrain_v4_boundary_network_class: list[list[int]] | None = None
    terrain_v4_orogen_network_class: list[list[int]] | None = None
    # Update 32: shows which v4 user-facing control is actually dominating a
    # cell. This is mainly a verification map for A/B runs.
    terrain_v4_control_response_class: list[list[int]] | None = None
    # Update 33: direct v4 effect diagnostics. Elevation delta is the signed
    # v4-after-v3 height change in meters; landform change class summarizes the
    # dominant v4 terrain effect (branch uplift, island uplift, rift lowering,
    # sliver/microplate corridor, or mixed).
    terrain_v4_elevation_delta_m: list[list[int]] | None = None
    terrain_v4_landform_change_class: list[list[int]] | None = None
    terrain_coast_ruggedness_x1000: list[list[int]] | None = None
    terrain_island_origin_class: list[list[int]] | None = None
    # Terrain Revisit 2 ocean-floor and island-shape diagnostics. These make
    # bathymetry and island morphology inspectable instead of inferred only
    # from the final terrain image. Ocean-floor classes: 0 land/background,
    # 1 abyssal plain, 2 mid-ocean ridge, 3 trench/subduction trough,
    # 4 fracture/transform zone, 5 seamount/hotspot province.
    terrain_ocean_floor_class: list[list[int]] | None = None
    terrain_mid_ocean_ridge_x1000: list[list[int]] | None = None
    terrain_trench_x1000: list[list[int]] | None = None
    terrain_fracture_zone_x1000: list[list[int]] | None = None
    terrain_seamount_x1000: list[list[int]] | None = None
    terrain_island_shape_complexity_x1000: list[list[int]] | None = None
    # Stage 3C.5 erosion/deposition diagnostics. These are compact diagnostic
    # rasters, not full hydrology outputs. They describe terrain maturity,
    # valley-corridor readiness, sediment source/sink fields, and the before/
    # after relief change applied by the erosion/deposition conditioning pass.
    terrain_erosion_strength_x1000: list[list[int]] | None = None
    terrain_deposition_field_x1000: list[list[int]] | None = None
    terrain_valley_corridor_x1000: list[list[int]] | None = None
    terrain_sediment_supply_x1000: list[list[int]] | None = None
    terrain_coastal_plain_x1000: list[list[int]] | None = None
    terrain_alluvial_fan_x1000: list[list[int]] | None = None
    terrain_floodplain_x1000: list[list[int]] | None = None
    terrain_maturity_x1000: list[list[int]] | None = None
    terrain_relief_delta_m: list[list[int]] | None = None
    crust_type: list[list[int]] | None = None
    # Stage 3 review metadata and sub-stage diagnostic summaries. Kept small
    # and JSON-serializable; large rasters remain in state/03_terrain.npz.
    terrain_diagnostics: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ClimateMap:
    width: int
    height: int
    annual_mean_temp_c_x10: list[list[int]]
    warmest_month_temp_c_x10: list[list[int]]
    coldest_month_temp_c_x10: list[list[int]]
    annual_precip_mm: list[list[int]]
    koppen_classification: list[list[str]]
    mean_land_temp_c: float
    mean_ocean_temp_c: float
    mean_land_precip_mm: float
    mean_ocean_precip_mm: float
    min_temp_c: float
    max_temp_c: float
    min_precip_mm: int
    max_precip_mm: int
    koppen_summary: dict[str, int]
    notes: list[str]
    climate_mode: str = "legacy"
    # Optional low/diagnostic-resolution climate-driver rasters used by the
    # seasonal climate backend.  Values are integer-scaled and saved to state;
    # they are deliberately kept out of compact system.json.
    climate_driver_maps: dict[str, Any] | None = None
    climate_driver_map_info: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)




@dataclass
class HydrologyMap:
    width: int
    height: int
    runoff_mm: list[list[int]]
    flow_accumulation: list[list[int]]
    river_intensity: list[list[int]]
    lake_mask: list[list[bool]]
    drainage_basin_id: list[list[int]]
    river_cell_count: int
    lake_cell_count: int
    max_flow_accumulation: int
    river_threshold: int
    estimated_major_river_count: int
    drainage_basin_count: int
    major_drainage_basin_count: int
    coastal_basin_count: int = 0
    endorheic_basin_count: int = 0
    minor_coastal_basin_cell_count: int = 0
    delta_cell_count: int = 0
    notes: list[str] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class BiomeMap:
    width: int
    height: int
    biome_classification: list[list[str]]
    biome_summary: dict[str, int]
    dominant_biome: str
    land_biome_count: int
    notes: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RegionSummary:
    region_id: str
    row: int
    col: int
    lat_south: float
    lat_north: float
    lon_west: float
    lon_east: float
    cell_count: int
    land_fraction: float
    ocean_fraction: float
    mean_elevation_m: float
    mean_land_elevation_m: float
    mean_temp_c: float
    mean_land_temp_c: float
    mean_precip_mm: float
    mean_land_precip_mm: float
    river_cell_count: int
    lake_cell_count: int
    dominant_koppen: str
    dominant_biome: str
    region_type: str
    biological_productivity_score: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RegionAnalysis:
    rows: int
    cols: int
    regions: list[RegionSummary]
    top_productive_region_ids: list[str]
    notes: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class MainPlanetProfile:
    planet_name: str
    rotation: RotationState
    atmosphere: Atmosphere
    hydrosphere: Hydrosphere
    geology: GeologyState
    terrain: TerrainMap
    climate: ClimateMap
    hydrology: HydrologyMap
    biomes: BiomeMap
    regions: RegionAnalysis
    notes: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
