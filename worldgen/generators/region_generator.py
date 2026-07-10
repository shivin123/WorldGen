"""Regional analysis for the Main Planet.

This layer converts the high-resolution map grids into a smaller set of named
latitude/longitude regions. It does not simulate cultures or civilization. The
purpose is to make the generated world easier to inspect, compare, and later use
as input for climate/terrain reports.
"""

from __future__ import annotations

from collections import Counter

from worldgen.models.planet_profile import (
    BiomeMap,
    ClimateMap,
    HydrologyMap,
    RegionAnalysis,
    RegionSummary,
    TerrainMap,
)
from worldgen.random_utils import clamp

DEFAULT_REGION_ROWS = 12
DEFAULT_REGION_COLS = 24


REGION_TYPE_BY_BIOME = {
    "ocean": "oceanic region",
    "lake / inland water": "inland water region",
    "ice sheet": "polar ice region",
    "tundra": "tundra region",
    "alpine highlands": "highland region",
    "boreal forest": "boreal forest region",
    "temperate forest": "temperate forest region",
    "temperate rainforest": "wet temperate forest region",
    "mediterranean scrubland": "mediterranean region",
    "temperate grassland": "grassland region",
    "semi-arid steppe": "steppe region",
    "cold desert": "cold desert region",
    "hot desert": "hot desert region",
    "savanna": "savanna region",
    "tropical seasonal forest": "tropical seasonal forest region",
    "tropical rainforest": "tropical rainforest region",
    "wetland / river corridor": "river corridor / wetland region",
}


def generate_regions(
    terrain: TerrainMap,
    climate: ClimateMap,
    hydrology: HydrologyMap,
    biomes: BiomeMap,
    rows: int = DEFAULT_REGION_ROWS,
    cols: int = DEFAULT_REGION_COLS,
) -> RegionAnalysis:
    """Aggregate full-resolution cells into regional summaries."""
    rows = max(2, rows)
    cols = max(4, cols)
    regions: list[RegionSummary] = []

    for region_row in range(rows):
        y0 = int(round(region_row * terrain.height / rows))
        y1 = int(round((region_row + 1) * terrain.height / rows))
        if y1 <= y0:
            y1 = min(terrain.height, y0 + 1)
        lat_north = 90.0 - y0 * 180.0 / terrain.height
        lat_south = 90.0 - y1 * 180.0 / terrain.height

        for region_col in range(cols):
            x0 = int(round(region_col * terrain.width / cols))
            x1 = int(round((region_col + 1) * terrain.width / cols))
            if x1 <= x0:
                x1 = min(terrain.width, x0 + 1)
            lon_west = -180.0 + x0 * 360.0 / terrain.width
            lon_east = -180.0 + x1 * 360.0 / terrain.width

            region = _summarize_region(
                terrain,
                climate,
                hydrology,
                biomes,
                region_row,
                region_col,
                y0,
                y1,
                x0,
                x1,
                lat_south,
                lat_north,
                lon_west,
                lon_east,
            )
            regions.append(region)

    top_regions = sorted(regions, key=lambda region: region.biological_productivity_score, reverse=True)[:8]

    return RegionAnalysis(
        rows=rows,
        cols=cols,
        regions=regions,
        top_productive_region_ids=[region.region_id for region in top_regions],
        notes=[
            "Regions are coarse latitude/longitude summaries of terrain, climate, hydrology, and biome grids.",
            "The biological productivity score is a broad environmental index, not a civilization or population model.",
        ],
    )


def _summarize_region(
    terrain: TerrainMap,
    climate: ClimateMap,
    hydrology: HydrologyMap,
    biomes: BiomeMap,
    region_row: int,
    region_col: int,
    y0: int,
    y1: int,
    x0: int,
    x1: int,
    lat_south: float,
    lat_north: float,
    lon_west: float,
    lon_east: float,
) -> RegionSummary:
    total_cells = max(1, (y1 - y0) * (x1 - x0))
    land_cells = 0
    ocean_cells = 0
    elevation_sum = 0.0
    land_elevation_sum = 0.0
    temp_sum = 0.0
    land_temp_sum = 0.0
    precip_sum = 0.0
    land_precip_sum = 0.0
    river_cells = 0
    lake_cells = 0
    biome_counts: Counter[str] = Counter()
    koppen_counts: Counter[str] = Counter()

    for r in range(y0, y1):
        for c in range(x0, x1):
            is_land = terrain.is_land[r][c]
            elevation = terrain.elevation_m[r][c]
            temp = climate.annual_mean_temp_c_x10[r][c] / 10.0
            precip = climate.annual_precip_mm[r][c]
            biome = biomes.biome_classification[r][c]
            koppen = climate.koppen_classification[r][c]

            elevation_sum += elevation
            temp_sum += temp
            precip_sum += precip
            biome_counts[biome] += 1
            koppen_counts[koppen] += 1

            if is_land:
                land_cells += 1
                land_elevation_sum += elevation
                land_temp_sum += temp
                land_precip_sum += precip
                if hydrology.river_intensity[r][c] > 0:
                    river_cells += 1
                if hydrology.lake_mask[r][c]:
                    lake_cells += 1
            else:
                ocean_cells += 1

    land_fraction = land_cells / total_cells
    ocean_fraction = ocean_cells / total_cells
    mean_elevation = elevation_sum / total_cells
    mean_land_elevation = land_elevation_sum / land_cells if land_cells else 0.0
    mean_temp = temp_sum / total_cells
    mean_land_temp = land_temp_sum / land_cells if land_cells else 0.0
    mean_precip = precip_sum / total_cells
    mean_land_precip = land_precip_sum / land_cells if land_cells else 0.0
    dominant_biome = biome_counts.most_common(1)[0][0] if biome_counts else "unknown"
    dominant_koppen = koppen_counts.most_common(1)[0][0] if koppen_counts else "unknown"
    region_type = _region_type(dominant_biome, dominant_koppen, land_fraction, river_cells, lake_cells)
    productivity_score = _productivity_score(
        land_fraction=land_fraction,
        mean_land_temp=mean_land_temp,
        mean_land_precip=mean_land_precip,
        mean_land_elevation=mean_land_elevation,
        dominant_biome=dominant_biome,
        river_cells=river_cells,
        lake_cells=lake_cells,
        land_cells=land_cells,
    )

    return RegionSummary(
        region_id=f"R{region_row + 1:02d}-{region_col + 1:02d}",
        row=region_row,
        col=region_col,
        lat_south=lat_south,
        lat_north=lat_north,
        lon_west=lon_west,
        lon_east=lon_east,
        cell_count=total_cells,
        land_fraction=land_fraction,
        ocean_fraction=ocean_fraction,
        mean_elevation_m=mean_elevation,
        mean_land_elevation_m=mean_land_elevation,
        mean_temp_c=mean_temp,
        mean_land_temp_c=mean_land_temp,
        mean_precip_mm=mean_precip,
        mean_land_precip_mm=mean_land_precip,
        river_cell_count=river_cells,
        lake_cell_count=lake_cells,
        dominant_koppen=dominant_koppen,
        dominant_biome=dominant_biome,
        region_type=region_type,
        biological_productivity_score=productivity_score,
    )


def _region_type(
    dominant_biome: str,
    dominant_koppen: str,
    land_fraction: float,
    river_cells: int,
    lake_cells: int,
) -> str:
    if land_fraction < 0.10:
        return "oceanic region"
    if lake_cells > 0 and land_fraction >= 0.10:
        return "lake district"
    if river_cells > 0 and dominant_biome not in {"ice sheet", "hot desert", "cold desert"}:
        return "river-influenced " + REGION_TYPE_BY_BIOME.get(dominant_biome, "mixed region")
    if dominant_koppen in {"EF", "ET"}:
        return "polar/cold-climate region"
    return REGION_TYPE_BY_BIOME.get(dominant_biome, "mixed region")


def _productivity_score(
    land_fraction: float,
    mean_land_temp: float,
    mean_land_precip: float,
    mean_land_elevation: float,
    dominant_biome: str,
    river_cells: int,
    lake_cells: int,
    land_cells: int,
) -> float:
    if land_cells <= 0:
        return 0.0

    # Temperature preference: broad land-life productivity band.
    if 8.0 <= mean_land_temp <= 26.0:
        temp_score = 1.0
    elif -2.0 <= mean_land_temp < 8.0:
        temp_score = (mean_land_temp + 2.0) / 10.0
    elif 26.0 < mean_land_temp <= 34.0:
        temp_score = 1.0 - (mean_land_temp - 26.0) / 16.0
    else:
        temp_score = 0.15

    # Precipitation preference: avoid both deserts and saturated extremes.
    if 650.0 <= mean_land_precip <= 1800.0:
        precip_score = 1.0
    elif 200.0 <= mean_land_precip < 650.0:
        precip_score = (mean_land_precip - 200.0) / 450.0
    elif 1800.0 < mean_land_precip <= 3200.0:
        precip_score = 1.0 - (mean_land_precip - 1800.0) / 2800.0
    else:
        precip_score = 0.12

    elevation_score = 1.0 if mean_land_elevation <= 1800.0 else max(0.15, 1.0 - (mean_land_elevation - 1800.0) / 3000.0)
    river_bonus = min(0.18, (river_cells + lake_cells * 1.5) / max(land_cells, 1) * 0.9)
    land_mix_score = 0.55 + 0.45 * min(1.0, land_fraction / 0.45)

    biome_modifier = {
        "tropical rainforest": 1.10,
        "tropical seasonal forest": 1.04,
        "temperate rainforest": 1.08,
        "temperate forest": 1.06,
        "wetland / river corridor": 1.08,
        "savanna": 0.92,
        "temperate grassland": 0.88,
        "mediterranean scrubland": 0.78,
        "semi-arid steppe": 0.56,
        "boreal forest": 0.62,
        "tundra": 0.30,
        "alpine highlands": 0.28,
        "cold desert": 0.16,
        "hot desert": 0.14,
        "ice sheet": 0.05,
        "lake / inland water": 0.65,
    }.get(dominant_biome, 0.55)

    raw = 100.0 * temp_score * precip_score * elevation_score * land_mix_score * biome_modifier
    return round(clamp(raw + river_bonus * 100.0, 0.0, 100.0), 1)
