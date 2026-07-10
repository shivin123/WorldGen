"""Biome/ecoregion classification for the Main Planet.

This is not an evolution model. The project assumes land-stage life already
exists, so this layer translates climate, terrain, and hydrology into broad
surface environments that can be used by later reporting or map layers.
"""

from __future__ import annotations

from worldgen.models.planet_profile import BiomeMap, ClimateMap, HydrologyMap, TerrainMap

OCEAN = "ocean"
ICE_SHEET = "ice sheet"
TUNDRA = "tundra"
ALPINE = "alpine highlands"
BOREAL_FOREST = "boreal forest"
TEMPERATE_FOREST = "temperate forest"
TEMPERATE_RAINFOREST = "temperate rainforest"
MEDITERRANEAN = "mediterranean scrubland"
GRASSLAND = "temperate grassland"
STEPPE = "semi-arid steppe"
COLD_DESERT = "cold desert"
HOT_DESERT = "hot desert"
SAVANNA = "savanna"
TROPICAL_SEASONAL_FOREST = "tropical seasonal forest"
TROPICAL_RAINFOREST = "tropical rainforest"
WETLAND = "wetland / river corridor"
LAKE = "lake / inland water"


def generate_biomes(
    terrain: TerrainMap,
    climate: ClimateMap,
    hydrology: HydrologyMap,
) -> BiomeMap:
    width = terrain.width
    height = terrain.height
    grid: list[list[str]] = []
    summary: dict[str, int] = {}

    for row in range(height):
        biome_row: list[str] = []
        for col in range(width):
            biome = _classify_cell(terrain, climate, hydrology, row, col)
            biome_row.append(biome)
            summary[biome] = summary.get(biome, 0) + 1
        grid.append(biome_row)

    land_summary = {name: count for name, count in summary.items() if name != OCEAN}
    dominant = max(land_summary.items(), key=lambda item: item[1])[0] if land_summary else OCEAN

    return BiomeMap(
        width=width,
        height=height,
        biome_classification=grid,
        biome_summary=dict(sorted(summary.items(), key=lambda item: item[0])),
        dominant_biome=dominant,
        land_biome_count=sum(land_summary.values()),
        notes=[
            "Biomes are broad surface-environment classes, not an evolution simulation.",
            "Land-stage life and atmospheric oxygen are assumed by project scope.",
            "Classification uses temperature, precipitation, Köppen class, elevation, rivers, and lake candidates.",
        ],
    )


def _classify_cell(
    terrain: TerrainMap,
    climate: ClimateMap,
    hydrology: HydrologyMap,
    row: int,
    col: int,
) -> str:
    if not terrain.is_land[row][col]:
        return OCEAN

    if hydrology.lake_mask[row][col]:
        return LAKE

    temp = climate.annual_mean_temp_c_x10[row][col] / 10.0
    warmest = climate.warmest_month_temp_c_x10[row][col] / 10.0
    coldest = climate.coldest_month_temp_c_x10[row][col] / 10.0
    precip = climate.annual_precip_mm[row][col]
    koppen = climate.koppen_classification[row][col]
    elevation = terrain.elevation_m[row][col]
    river = hydrology.river_intensity[row][col]

    lat = 90.0 - (row + 0.5) * 180.0 / terrain.height
    lon = -180.0 + (col + 0.5) * 360.0 / terrain.width

    # Very high wet corridors can support riparian/wetland ecosystems even in
    # otherwise dry climates. Keep ice/desert extremes dominant first.
    is_strong_river = river >= 180
    is_moderate_river = river >= 90

    # Real Earth calibration: Antarctica and Greenland should classify as ice
    # sheets even when broad annual metrics would otherwise produce boreal
    # forest on coarse terrain.
    if str(getattr(terrain, "source", "")).startswith("real_earth"):
        if lat < -60.0:
            return ICE_SHEET
        if 58.0 <= lat <= 84.0 and -75.0 <= lon <= -10.0 and warmest < 8.0:
            return ICE_SHEET

    if koppen == "EF" or warmest < 0.0:
        return ICE_SHEET
    if elevation >= 3200 and temp < 8.0:
        return ALPINE
    if koppen == "ET" or warmest < 10.0:
        return TUNDRA
    if elevation >= 4200:
        return ALPINE

    if koppen == "BWh":
        if is_strong_river:
            return WETLAND
        return HOT_DESERT
    if koppen == "BWk":
        if is_strong_river:
            return WETLAND
        return COLD_DESERT
    if koppen in {"BSh", "BSk"}:
        if is_moderate_river and precip > 240:
            return WETLAND
        return SAVANNA if temp >= 18.0 else STEPPE

    if koppen in {"Af", "Am"}:
        if precip >= 1800:
            return TROPICAL_RAINFOREST
        return TROPICAL_SEASONAL_FOREST
    if koppen == "Aw":
        if is_strong_river and precip > 900:
            return WETLAND
        return SAVANNA

    if koppen in {"Csa", "Csb"}:
        if precip > 900:
            return TEMPERATE_FOREST
        return MEDITERRANEAN

    if koppen in {"Cfa", "Cfb"}:
        if is_strong_river and precip > 800:
            return WETLAND
        if precip >= 1600:
            return TEMPERATE_RAINFOREST
        if precip < 520:
            return GRASSLAND
        return TEMPERATE_FOREST

    if koppen in {"Dfa", "Dfb", "Dfc"}:
        if precip < 360:
            return STEPPE if coldest > -18.0 else COLD_DESERT
        if temp < 2.0 or coldest < -22.0:
            return BOREAL_FOREST
        if precip > 1000 and is_moderate_river:
            return WETLAND
        return BOREAL_FOREST if temp < 7.0 else TEMPERATE_FOREST

    # Fallback for odd transition cells.
    if temp >= 22.0:
        if precip >= 1500:
            return TROPICAL_RAINFOREST
        if precip >= 700:
            return SAVANNA
        return HOT_DESERT
    if temp <= 0.0:
        return TUNDRA if precip >= 180 else COLD_DESERT
    if precip >= 1500:
        return TEMPERATE_RAINFOREST
    if precip >= 520:
        return TEMPERATE_FOREST
    if precip >= 220:
        return GRASSLAND
    return COLD_DESERT if temp < 12.0 else HOT_DESERT
