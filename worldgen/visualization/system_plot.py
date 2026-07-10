"""PNG visualizations for generated star systems."""

from __future__ import annotations

import math
from pathlib import Path
import os
import json

from worldgen.models.bodies import Planet
from worldgen.models.system import StarSystem


PLANET_CLASS_COLORS = {
    "rocky": "#b86b43",
    "super_earth": "#d49a58",
    "mini_neptune": "#6fb4c6",
    "ice_giant": "#8fbbe8",
    "gas_giant": "#d6b06f",
    "icy_dwarf": "#c9d8e6",
}

MOON_CLASS_COLORS = {
    "rocky_moon": "#b0b0b0",
    "mixed_ice_rock_moon": "#8ec2da",
    "icy_moon": "#cfe7f7",
}


KOPPEN_COLORS = {
    # Deep navy is intentionally distinct from every land Köppen color.
    "O": "#071a2f",
    "Af": "#145a32",
    "Am": "#1e8449",
    "Aw": "#58d68d",
    "BWh": "#d68910",
    "BWk": "#caa767",
    "BSh": "#f0b94e",
    "BSk": "#d8c27a",
    "Cfa": "#7dcea0",
    "Cfb": "#82e0aa",
    "Cfc": "#a3e4d7",
    "Csa": "#f7dc6f",
    "Csb": "#f8c471",
    "Csc": "#f5cba7",
    "Cwa": "#52be80",
    "Cwb": "#73c6b6",
    "Cwc": "#a2d9ce",
    "Dfa": "#85c1e9",
    "Dfb": "#5dade2",
    "Dfc": "#2e86c1",
    "Dfd": "#1b4f72",
    "Dsa": "#bb8fce",
    "Dsb": "#af7ac5",
    "Dsc": "#9b59b6",
    "Dsd": "#76448a",
    "Dwa": "#5499c7",
    "Dwb": "#2980b9",
    "Dwc": "#2471a3",
    "Dwd": "#1f618d",
    "ET": "#d6eaf8",
    "EF": "#f8f9f9",
}




KOPPEN_FULL_NAMES = {
    "O": "Ocean",
    "Af": "Tropical rainforest climate",
    "Am": "Tropical monsoon climate",
    "Aw": "Tropical savanna climate (dry winter)",
    "BWh": "Hot desert climate",
    "BWk": "Cold desert climate",
    "BSh": "Hot semi-arid steppe climate",
    "BSk": "Cold semi-arid steppe climate",
    "Cfa": "Humid subtropical climate",
    "Cfb": "Temperate oceanic climate",
    "Cfc": "Subpolar oceanic climate",
    "Csa": "Hot-summer Mediterranean climate",
    "Csb": "Warm-summer Mediterranean climate",
    "Csc": "Cold-summer Mediterranean climate",
    "Cwa": "Monsoon-influenced humid subtropical climate (dry winter)",
    "Cwb": "Subtropical highland climate with dry winter",
    "Cwc": "Cold subtropical highland climate with dry winter",
    "Dfa": "Hot-summer humid continental climate",
    "Dfb": "Warm-summer humid continental climate",
    "Dfc": "Subarctic climate",
    "Dfd": "Severely cold subarctic climate",
    "Dsa": "Hot-summer Mediterranean continental climate",
    "Dsb": "Warm-summer Mediterranean continental climate",
    "Dsc": "Dry-summer subarctic climate",
    "Dsd": "Severely cold dry-summer subarctic climate",
    "Dwa": "Monsoon-influenced hot-summer humid continental climate",
    "Dwb": "Monsoon-influenced warm-summer humid continental climate",
    "Dwc": "Monsoon-influenced subarctic climate",
    "Dwd": "Severely cold monsoon-influenced subarctic climate",
    "ET": "Tundra climate",
    "EF": "Ice cap climate",
}


def _koppen_label(code: str) -> str:
    return f"{code} — {KOPPEN_FULL_NAMES.get(code, code)}"


BIOME_COLORS = {
    # More distinct biome palette. Previous greens/tans were too close together,
    # making the biome map difficult to read at a glance.
    "ocean": "#2f6fa7",
    "lake / inland water": "#23a9e1",
    "ice sheet": "#ffffff",
    "tundra": "#b9d6e8",
    "alpine highlands": "#8d8d8d",
    "boreal forest": "#174f36",
    "temperate forest": "#2f8f46",
    "temperate rainforest": "#006d5b",
    "mediterranean scrubland": "#c47f2c",
    "temperate grassland": "#c7d958",
    "semi-arid steppe": "#d6b04c",
    "cold desert": "#cfc0a2",
    "hot desert": "#e08b18",
    "savanna": "#f2cd49",
    "tropical seasonal forest": "#55b948",
    "tropical rainforest": "#004d24",
    "wetland / river corridor": "#7a4fb3",
}


def save_system_orbit_map(system: StarSystem, output_path: str | Path) -> None:
    """Save the orbital architecture figure."""
    plt, Circle, Wedge = _import_matplotlib()
    output_path = _prepare_output_path(output_path)

    fig, ax = plt.subplots(figsize=(8.5, 8.5))
    _draw_orbital_map(ax, system, Circle, Wedge)
    fig.suptitle(f"Star System Orbital Map - Seed {system.seed}", fontsize=15, fontweight="bold")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)



def save_system_size_chart(system: StarSystem, output_path: str | Path) -> None:
    """Save the body size comparison figure."""
    plt, Circle, _ = _import_matplotlib()
    output_path = _prepare_output_path(output_path)

    fig = plt.figure(figsize=(14, 7))
    grid = fig.add_gridspec(1, 2, width_ratios=[0.9, 1.6])
    star_ax = fig.add_subplot(grid[0, 0])
    planet_ax = fig.add_subplot(grid[0, 1])

    _draw_star_size_panel(star_ax, system, Circle)
    _draw_planet_size_panel(planet_ax, system, Circle)

    fig.suptitle(f"Body Size Comparison - Seed {system.seed}", fontsize=15, fontweight="bold")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)



def save_main_planet_moon_view(system: StarSystem, output_path: str | Path) -> None:
    """Save a dedicated visualization for the Main Planet and its moon."""
    main_planet = system.main_planet
    if main_planet is None or main_planet.moon is None:
        raise RuntimeError("Main planet moon visualization requested, but no main planet moon exists.")

    plt, Circle, _ = _import_matplotlib()
    output_path = _prepare_output_path(output_path)

    fig = plt.figure(figsize=(15, 8.5))
    grid = fig.add_gridspec(2, 2, width_ratios=[1.25, 1.0], height_ratios=[1.15, 0.85])
    orbit_ax = fig.add_subplot(grid[:, 0])
    detail_ax = fig.add_subplot(grid[0, 1])
    size_ax = fig.add_subplot(grid[1, 1])

    _draw_main_planet_moon_orbit(orbit_ax, system, Circle)
    _draw_main_planet_moon_text(detail_ax, system)
    _draw_main_planet_moon_size(size_ax, system, Circle)

    fig.suptitle(
        f"Main Planet and Moon - {main_planet.name} & {main_planet.moon.name}",
        fontsize=15,
        fontweight="bold",
    )
    fig.savefig(output_path, dpi=180)
    plt.close(fig)



def save_main_planet_terrain_view(system: StarSystem, output_path: str | Path) -> None:
    """Save elevation and land/ocean terrain using Pillow."""
    if system.main_planet_profile is None:
        raise RuntimeError("Terrain visualization requested, but no Main Planet profile exists.")

    from PIL import Image, ImageDraw

    output_path = _prepare_output_path(output_path)
    profile = system.main_planet_profile
    terrain = profile.terrain
    width = terrain.width
    height = terrain.height
    legend_height = 132

    image = Image.new("RGB", (width, height + legend_height), "white")
    min_elev = terrain.min_elevation_m
    max_elev = terrain.max_elevation_m
    coastline_mask = _cached_large_coastline_mask(terrain.is_land)
    pixels = []
    idx = 0
    for r in range(height):
        elev_row = terrain.elevation_m[r]
        land_row = terrain.is_land[r]
        for c in range(width):
            elev = elev_row[c]
            if coastline_mask[idx]:
                pixels.append((28, 28, 24))
            elif land_row[c]:
                t = elev / max(1.0, max_elev)
                pixels.append(_land_elevation_color(t))
            else:
                t = abs(elev) / max(1.0, abs(min_elev))
                pixels.append(_ocean_depth_color(t))
            idx += 1

    map_img = Image.new("RGB", (width, height))
    map_img.putdata(pixels)
    image.paste(map_img, (0, 0))

    draw = ImageDraw.Draw(image)
    y0 = height + 12
    terrain_title = "artifact-repaired real Earth terrain" if str(getattr(terrain, "source", "")).startswith("real_earth") else "detailed procedural terrain"
    draw.text((16, y0), f"{profile.planet_name} {terrain_title}", fill=(0, 0, 0))
    draw.text(
        (16, y0 + 22),
        f"map {terrain.width} x {terrain.height} | ocean {terrain.ocean_fraction:.1%}, land {terrain.land_fraction:.1%} | "
        f"elevation {terrain.min_elevation_m:,} to {terrain.max_elevation_m:,} m | mean land {terrain.mean_land_elevation_m:,.0f} m | mean ocean {terrain.mean_ocean_depth_m:,.0f} m",
        fill=(0, 0, 0),
    )
    legend = [
        ("deep ocean", _ocean_depth_color(1.0)),
        ("shallow sea", _ocean_depth_color(0.15)),
        ("lowland", _land_elevation_color(0.12)),
        ("highland", _land_elevation_color(0.55)),
        ("mountain", _land_elevation_color(1.0)),
        ("major coastline", (28, 28, 24)),
    ]
    x = 16
    for label, color in legend:
        draw.rectangle((x, y0 + 55, x + 20, y0 + 75), fill=color, outline=(50, 50, 50))
        draw.text((x + 28, y0 + 57), label, fill=(0, 0, 0))
        x += 145
    draw.text((16, y0 + 96), "Terrain includes real/calibration or procedural relief, water masks, elevation/depth coloring, and marked major coastlines.", fill=(0, 0, 0))

    image.save(output_path, optimize=False, compress_level=0)


def save_main_planet_temperature_view(system: StarSystem, output_path: str | Path) -> None:
    """Save a dedicated annual mean temperature map using Pillow."""
    profile, climate, terrain = _require_main_planet_climate(system)
    from PIL import Image, ImageDraw
    output_path = _prepare_output_path(output_path)

    width = climate.width
    height = climate.height
    legend_height = 96
    image = Image.new("RGB", (width, height + legend_height), "white")
    min_t = climate.min_temp_c
    max_t = climate.max_temp_c
    span = max(1.0, max_t - min_t)

    pixels = []
    for r in range(height):
        for c in range(width):
            temp_c = climate.annual_mean_temp_c_x10[r][c] / 10.0
            t = (temp_c - min_t) / span
            pixels.append(_temperature_color(t))

    map_img = Image.new("RGB", (width, height))
    map_img.putdata(pixels)
    image.paste(map_img, (0, 0))

    draw = ImageDraw.Draw(image)
    y0 = height + 12
    draw.text((16, y0), f"{profile.planet_name} annual mean temperature", fill=(0, 0, 0))
    draw.text(
        (16, y0 + 22),
        f"mean land {climate.mean_land_temp_c:.1f} °C | mean ocean {climate.mean_ocean_temp_c:.1f} °C | "
        f"range {climate.min_temp_c:.1f} to {climate.max_temp_c:.1f} °C | ocean {terrain.ocean_fraction:.1%}",
        fill=(0, 0, 0),
    )
    _draw_gradient_legend(draw, 16, y0 + 52, 260, climate.min_temp_c, climate.max_temp_c, _temperature_color, "°C")
    image.save(output_path, optimize=False, compress_level=0)


def save_main_planet_precipitation_view(system: StarSystem, output_path: str | Path) -> None:
    """Save a dedicated annual precipitation map using Pillow."""
    profile, climate, _terrain = _require_main_planet_climate(system)
    from PIL import Image, ImageDraw
    output_path = _prepare_output_path(output_path)

    width = climate.width
    height = climate.height
    legend_height = 96
    image = Image.new("RGB", (width, height + legend_height), "white")
    max_p = max(1, climate.max_precip_mm)

    pixels = []
    for r in range(height):
        for c in range(width):
            precip = climate.annual_precip_mm[r][c]
            t = math.log1p(precip) / math.log1p(max_p)
            pixels.append(_precipitation_color(t))

    map_img = Image.new("RGB", (width, height))
    map_img.putdata(pixels)
    image.paste(map_img, (0, 0))

    draw = ImageDraw.Draw(image)
    y0 = height + 12
    draw.text((16, y0), f"{profile.planet_name} annual precipitation", fill=(0, 0, 0))
    draw.text(
        (16, y0 + 22),
        f"mean land {climate.mean_land_precip_mm:,.0f} mm/year | mean ocean {climate.mean_ocean_precip_mm:,.0f} mm/year | "
        f"range {climate.min_precip_mm:,} to {climate.max_precip_mm:,} mm/year | {profile.hydrosphere.water_inventory_class}",
        fill=(0, 0, 0),
    )
    _draw_gradient_legend(draw, 16, y0 + 52, 260, 0, climate.max_precip_mm, _precipitation_color, "mm/year")
    image.save(output_path, optimize=False, compress_level=0)


def save_main_planet_koppen_view(system: StarSystem, output_path: str | Path) -> None:
    """Save a dedicated simplified Köppen classification map using Pillow."""
    if system.main_planet_profile is None:
        raise RuntimeError("Köppen visualization requested, but no Main Planet profile exists.")

    from PIL import Image, ImageDraw

    output_path = _prepare_output_path(output_path)
    profile = system.main_planet_profile
    climate = profile.climate
    terrain = profile.terrain
    width = climate.width
    height = climate.height
    legend_height = 170

    image = Image.new("RGB", (width, height + legend_height), "white")
    pixels = []
    present_counts: dict[str, int] = {}
    for row in climate.koppen_classification:
        for code in row:
            pixels.append(_hex_to_rgb(KOPPEN_COLORS.get(code, "#777777")))
            present_counts[code] = present_counts.get(code, 0) + 1

    map_img = Image.new("RGB", (width, height))
    map_img.putdata(pixels)
    try:
        import numpy as np
        coast = np.frombuffer(_cached_large_coastline_mask(terrain.is_land), dtype=np.uint8).reshape((height, width)).astype(bool)
        rgb = np.asarray(map_img, dtype=np.uint8)
        rgb[coast] = (8, 8, 8)
        map_img = Image.fromarray(rgb, mode="RGB")
    except Exception:
        pass
    image.paste(map_img, (0, 0))

    draw = ImageDraw.Draw(image)
    y0 = height + 10
    draw.text((16, y0), f"{profile.planet_name} simplified Köppen climate classes", fill=(0, 0, 0))
    dominant = sorted(climate.koppen_summary.items(), key=lambda item: item[1], reverse=True)[:8]
    dominant_text = ", ".join(f"{_koppen_label(code)} {count:,}" for code, count in dominant) if dominant else "no land climate cells"
    draw.text((16, y0 + 22), f"Dominant land climates: {dominant_text}", fill=(0, 0, 0))
    draw.text((16, y0 + 44), "Ocean cells are stored as O and excluded from the dominant land-climate summary.", fill=(0, 0, 0))

    x = 16
    y = y0 + 72
    for code in KOPPEN_COLORS:
        if code == "O" or code not in present_counts:
            continue
        color = _hex_to_rgb(KOPPEN_COLORS[code])
        draw.rectangle((x, y, x + 18, y + 18), fill=color, outline=(50, 50, 50))
        draw.text((x + 24, y + 2), _koppen_label(code), fill=(0, 0, 0))
        x += 270
        if x + 260 > width:
            x = 16
            y += 28

    draw.text((16, height + legend_height - 28), "Köppen classes are simplified approximations from annual and seasonal values.", fill=(0, 0, 0))
    image.save(output_path, optimize=False, compress_level=0)


def save_main_planet_hydrology_view(system: StarSystem, output_path: str | Path) -> None:
    """Save a dedicated river, lake, and runoff map using vectorized raster writing."""
    if system.main_planet_profile is None:
        raise RuntimeError("Hydrology visualization requested, but no Main Planet profile exists.")

    from PIL import Image, ImageDraw
    import numpy as np

    output_path = _prepare_output_path(output_path)
    profile = system.main_planet_profile
    terrain = profile.terrain
    hydrology = profile.hydrology

    width = hydrology.width
    height = hydrology.height
    legend_height = 120

    land = np.asarray(terrain.is_land, dtype=bool)
    lakes = np.asarray(hydrology.lake_mask, dtype=bool)
    rivers = np.asarray(hydrology.river_intensity, dtype=np.uint16)
    runoff = np.asarray(hydrology.runoff_mm, dtype=np.uint16)
    coast = np.frombuffer(_cached_large_coastline_mask(terrain.is_land), dtype=np.uint8).reshape((height, width)).astype(bool)

    rgb = np.zeros((height, width, 3), dtype=np.uint8)
    rgb[~land] = (79, 134, 198)
    rgb[land] = (217, 198, 138)

    # Wetter land is slightly greener.
    green_boost = np.minimum(55, runoff // 18).astype(np.uint8)
    land_rows, land_cols = np.where(land)
    rgb[land_rows, land_cols, 0] = np.maximum(0, 217 - (green_boost[land_rows, land_cols] // 3))
    rgb[land_rows, land_cols, 2] = np.maximum(0, 138 - (green_boost[land_rows, land_cols] // 5))

    river_mask = rivers > 0
    if river_mask.any():
        t = np.clip(rivers.astype(np.float32) / 255.0, 0.0, 1.0)
        rgb[river_mask, 0] = (90 * (1.0 - t[river_mask])).astype(np.uint8)
        rgb[river_mask, 1] = (165 * (1.0 - t[river_mask]) + 35 * t[river_mask]).astype(np.uint8)
        rgb[river_mask, 2] = (215 * (1.0 - t[river_mask]) + 130 * t[river_mask]).astype(np.uint8)

    rgb[lakes] = (31, 120, 180)
    rgb[coast] = (28, 28, 24)

    image = Image.new("RGB", (width, height + legend_height), "white")
    image.paste(Image.fromarray(rgb, mode="RGB"), (0, 0))

    draw = ImageDraw.Draw(image)
    y0 = height + 12
    draw.text((16, y0), f"{profile.planet_name} hydrology | rivers, lake candidates, and runoff", fill=(0, 0, 0))
    draw.text(
        (16, y0 + 22),
        f"river cells {hydrology.river_cell_count:,} | major river systems {hydrology.estimated_major_river_count} | "
        f"lakes {hydrology.lake_cell_count:,} | coastal basins {getattr(hydrology, 'coastal_basin_count', 0):,} | "
        f"endorheic basins {getattr(hydrology, 'endorheic_basin_count', 0):,}",
        fill=(0, 0, 0),
    )

    legend_items = [
        ("ocean", (79, 134, 198)),
        ("land", (217, 198, 138)),
        ("river", (20, 80, 160)),
        ("lake candidate", (31, 120, 180)),
        ("major coastline", (28, 28, 24)),
    ]
    x = 16
    for label, color in legend_items:
        draw.rectangle((x, y0 + 54, x + 20, y0 + 74), fill=color, outline=(50, 50, 50))
        draw.text((x + 28, y0 + 56), label, fill=(0, 0, 0))
        x += 165

    image.save(output_path, optimize=False, compress_level=0)



def save_main_planet_drainage_basins_view(system: StarSystem, output_path: str | Path) -> None:
    """Save a drainage basin map with rivers overlaid using vectorized raster writing."""
    if system.main_planet_profile is None:
        raise RuntimeError("Drainage basin visualization requested, but no Main Planet profile exists.")

    from PIL import Image, ImageDraw
    import numpy as np

    output_path = _prepare_output_path(output_path)
    profile = system.main_planet_profile
    terrain = profile.terrain
    hydrology = profile.hydrology

    width = hydrology.width
    height = hydrology.height
    legend_height = 140

    land = np.asarray(terrain.is_land, dtype=bool)
    basin_ids = np.asarray(hydrology.drainage_basin_id, dtype=np.int32)
    rivers = np.asarray(hydrology.river_intensity, dtype=np.uint16)
    lakes = np.asarray(hydrology.lake_mask, dtype=bool)
    coast = np.frombuffer(_cached_large_coastline_mask(terrain.is_land), dtype=np.uint8).reshape((height, width)).astype(bool)

    rgb = np.zeros((height, width, 3), dtype=np.uint8)
    rgb[~land] = (79, 134, 198)
    rgb[land] = (207, 194, 145)

    # Mark every real basin with a deterministic color. Tiny coastal catchments
    # have already been merged into COASTAL_MINOR_BASIN_ID by the hydrology model.
    rgb[land & (basin_ids == -2)] = (224, 214, 166)
    positive_mask = land & (basin_ids > 0)
    if positive_mask.any():
        b = basin_ids.astype(np.int64)
        xhash = (b * 1103515245 + 12345) & 0x7FFFFFFF
        rgb[:, :, 0][positive_mask] = (115 + (xhash % 95))[positive_mask].astype(np.uint8)
        rgb[:, :, 1][positive_mask] = (105 + ((xhash // 97) % 105))[positive_mask].astype(np.uint8)
        rgb[:, :, 2][positive_mask] = (75 + ((xhash // 7919) % 95))[positive_mask].astype(np.uint8)

    # Light basin-boundary overlay between different neighboring basin ids.
    boundary = np.zeros((height, width), dtype=bool)
    boundary[:-1, :] |= (basin_ids[:-1, :] != basin_ids[1:, :]) & land[:-1, :] & land[1:, :]
    boundary[:, :-1] |= (basin_ids[:, :-1] != basin_ids[:, 1:]) & land[:, :-1] & land[:, 1:]
    rgb[boundary] = (55, 48, 38)

    rgb[rivers > 0] = (25, 85, 170)
    rgb[lakes] = (20, 105, 170)
    rgb[coast] = (28, 28, 24)

    image = Image.new("RGB", (width, height + legend_height), "white")
    image.paste(Image.fromarray(rgb, mode="RGB"), (0, 0))

    draw = ImageDraw.Draw(image)
    y0 = height + 12
    draw.text((16, y0), f"{profile.planet_name} drainage basins | watershed ids with rivers overlaid", fill=(0, 0, 0))
    draw.text(
        (16, y0 + 22),
        f"basins {hydrology.drainage_basin_count:,} | major basins {hydrology.major_drainage_basin_count:,} | "
        f"coastal {getattr(hydrology, 'coastal_basin_count', 0):,} | endorheic {getattr(hydrology, 'endorheic_basin_count', 0):,} | "
        f"minor coastal cells {getattr(hydrology, 'minor_coastal_basin_cell_count', 0):,}",
        fill=(0, 0, 0),
    )

    legend_items = [
        ("ocean", (79, 134, 198)),
        ("basin colors", (181, 163, 105)),
        ("merged coastal basins", (224, 214, 166)),
        ("river", (25, 85, 170)),
        ("lake candidate", (20, 105, 170)),
        ("major coastline", (28, 28, 24)),
    ]
    x = 16
    for label, color in legend_items:
        draw.rectangle((x, y0 + 54, x + 20, y0 + 74), fill=color, outline=(50, 50, 50))
        draw.text((x + 28, y0 + 56), label, fill=(0, 0, 0))
        x += 170
    draw.text((16, y0 + 92), "Colored areas are flow-derived drainage basins. Small coastal catchments are merged into the coastal-basins class; inland terminal basins remain endorheic.", fill=(0, 0, 0))

    image.save(output_path, optimize=False, compress_level=0)



_COASTLINE_CACHE: dict[int, bytearray] = {}


def _cached_large_coastline_mask(is_land: list[list[bool]]) -> bytearray:
    key = id(is_land)
    cached = _COASTLINE_CACHE.get(key)
    if cached is not None:
        return cached
    mask = _large_coastline_mask(is_land)
    _COASTLINE_CACHE[key] = mask
    return mask


def _large_coastline_mask(is_land: list[list[bool]]) -> bytearray:
    """Return coastline cells between large land and large ocean components.

    This is used by several diagnostic maps, including 4K views.  The older
    implementation walked every component in Python, which became expensive once
    Update42 began producing many more islands and fragmented coastlines.  The
    vectorized path keeps the same visual intent while making coastline overlays
    scale with full-resolution diagnostic runs.
    """
    try:
        import numpy as np
        from scipy import ndimage
    except Exception:  # pragma: no cover - fallback for minimal installs
        from collections import deque

        height = len(is_land)
        width = len(is_land[0]) if height else 0
        total = width * height
        if total == 0:
            return bytearray()

        min_land = max(180, int(total * 0.00055))
        min_ocean = max(800, int(total * 0.0040))
        large_land = bytearray(total)
        large_ocean = bytearray(total)
        visited = bytearray(total)

        def scan(target_land: bool, min_size: int, output: bytearray) -> None:
            for rr in range(height):
                for cc in range(width):
                    idx = rr * width + cc
                    if visited[idx] or is_land[rr][cc] != target_land:
                        continue
                    q = deque([(rr, cc)])
                    visited[idx] = 1
                    cells: list[int] = []
                    while q:
                        r, c = q.popleft()
                        cell_idx = r * width + c
                        cells.append(cell_idx)
                        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                            nr = r + dr
                            if nr < 0 or nr >= height:
                                continue
                            nc = (c + dc) % width
                            nidx = nr * width + nc
                            if not visited[nidx] and is_land[nr][nc] == target_land:
                                visited[nidx] = 1
                                q.append((nr, nc))
                    if len(cells) >= min_size:
                        for cell_idx in cells:
                            output[cell_idx] = 1

        scan(True, min_land, large_land)
        visited = bytearray(total)
        scan(False, min_ocean, large_ocean)
        coast = bytearray(total)
        for r in range(height):
            for c in range(width):
                idx = r * width + c
                if not large_land[idx]:
                    continue
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr = r + dr
                    if nr < 0 or nr >= height:
                        continue
                    nc = (c + dc) % width
                    nidx = nr * width + nc
                    if large_ocean[nidx]:
                        coast[idx] = 1
                        break
        return coast

    land = np.asarray(is_land, dtype=bool)
    if land.ndim != 2 or land.size == 0:
        return bytearray()
    height, width = land.shape
    total = int(width * height)

    min_land = max(180, int(total * 0.00055))
    min_ocean = max(800, int(total * 0.0040))
    structure = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)

    land_labels, land_count = ndimage.label(land, structure=structure)
    if land_count:
        land_sizes = np.bincount(land_labels.ravel())
        large_land = land & (land_sizes[land_labels] >= min_land)
    else:
        large_land = np.zeros_like(land, dtype=bool)

    water = ~land
    ocean_labels, ocean_count = ndimage.label(water, structure=structure)
    if ocean_count:
        ocean_sizes = np.bincount(ocean_labels.ravel())
        large_ocean = water & (ocean_sizes[ocean_labels] >= min_ocean)
    else:
        large_ocean = np.zeros_like(water, dtype=bool)

    ocean_north = np.vstack((large_ocean[0:1, :], large_ocean[:-1, :]))
    ocean_south = np.vstack((large_ocean[1:, :], large_ocean[-1:, :]))
    ocean_west = np.roll(large_ocean, 1, axis=1)
    ocean_east = np.roll(large_ocean, -1, axis=1)
    coast = large_land & (ocean_north | ocean_south | ocean_west | ocean_east)
    return bytearray(coast.astype(np.uint8, copy=False).ravel().tobytes())


def _basin_color(basin_id: int) -> tuple[int, int, int]:
    if basin_id <= 0:
        return (79, 134, 198)
    # Deterministic color hash. Keep colors earthy but varied.
    x = (basin_id * 1103515245 + 12345) & 0x7FFFFFFF
    r = 115 + (x % 95)
    g = 105 + ((x // 97) % 105)
    b = 75 + ((x // 7919) % 95)
    return (r, g, b)


def _require_main_planet_climate(system: StarSystem):
    if system.main_planet_profile is None:
        raise RuntimeError("Climate visualization requested, but no Main Planet profile exists.")
    profile = system.main_planet_profile
    return profile, profile.climate, profile.terrain




def save_main_planet_biome_view(system: StarSystem, output_path: str | Path) -> None:
    """Save a broad biome/ecoregion map using Pillow."""
    if system.main_planet_profile is None:
        raise RuntimeError("Biome visualization requested, but no Main Planet profile exists.")

    from PIL import Image, ImageDraw

    output_path = _prepare_output_path(output_path)
    profile = system.main_planet_profile
    biomes = profile.biomes
    width = biomes.width
    height = biomes.height
    legend_height = 175

    image = Image.new("RGB", (width, height + legend_height), "white")
    pixels = []
    for row in biomes.biome_classification:
        for cell in row:
            pixels.append(_hex_to_rgb(BIOME_COLORS.get(cell, "#777777")))
    map_img = Image.new("RGB", (width, height))
    map_img.putdata(pixels)
    image.paste(map_img, (0, 0))

    draw = ImageDraw.Draw(image)
    y0 = height + 10
    draw.text((16, y0), f"{profile.planet_name} biomes / ecoregions", fill=(0, 0, 0))
    dominant = sorted(
        ((name, count) for name, count in biomes.biome_summary.items() if name != "ocean"),
        key=lambda item: item[1],
        reverse=True,
    )[:6]
    dominant_text = ", ".join(f"{name} {count:,}" for name, count in dominant) if dominant else "no land biome cells"
    draw.text((16, y0 + 22), f"Dominant land biomes: {dominant_text}", fill=(0, 0, 0))
    draw.text((16, y0 + 44), "Classification layer only: assumes land-stage life; does not simulate evolution or species.", fill=(0, 0, 0))

    present = [name for name in BIOME_COLORS if name in biomes.biome_summary and name != "ocean"]
    x = 16
    y = y0 + 72
    for i, name in enumerate(present[:18]):
        color = _hex_to_rgb(BIOME_COLORS[name])
        draw.rectangle((x, y, x + 18, y + 18), fill=color, outline=(50, 50, 50))
        draw.text((x + 24, y + 2), name, fill=(0, 0, 0))
        x += 245
        if x + 230 > width:
            x = 16
            y += 28

    image.save(output_path, optimize=False, compress_level=0)



def save_main_planet_regions_view(system: StarSystem, output_path: str | Path) -> None:
    """Save a coarse regional-analysis map using Pillow."""
    if system.main_planet_profile is None:
        raise RuntimeError("Regional visualization requested, but no Main Planet profile exists.")

    from PIL import Image, ImageDraw

    output_path = _prepare_output_path(output_path)
    profile = system.main_planet_profile
    analysis = profile.regions
    terrain = profile.terrain

    map_width = max(1200, analysis.cols * 54)
    map_height = int(round(map_width * 0.50))
    legend_height = 190
    image = Image.new("RGB", (map_width, map_height + legend_height), "white")
    draw = ImageDraw.Draw(image)

    top_ids = set(analysis.top_productive_region_ids[:8])

    for region in analysis.regions:
        x0 = int(round(region.col * map_width / analysis.cols))
        x1 = int(round((region.col + 1) * map_width / analysis.cols))
        y0 = int(round(region.row * map_height / analysis.rows))
        y1 = int(round((region.row + 1) * map_height / analysis.rows))
        color = _region_color(region)
        outline = (30, 30, 30) if region.region_id in top_ids else (130, 130, 130)
        width = 3 if region.region_id in top_ids else 1
        draw.rectangle((x0, y0, x1, y1), fill=color, outline=outline, width=width)

        if (x1 - x0) >= 44 and (y1 - y0) >= 30:
            score = int(round(region.biological_productivity_score))
            label = f"{region.region_id}\n{score}" if region.land_fraction >= 0.10 else region.region_id
            draw.text((x0 + 4, y0 + 4), label, fill=(0, 0, 0))

    # Latitude/longitude reference lines.
    for i in range(1, analysis.cols):
        x = int(round(i * map_width / analysis.cols))
        draw.line((x, 0, x, map_height), fill=(105, 105, 105), width=1)
    for i in range(1, analysis.rows):
        y = int(round(i * map_height / analysis.rows))
        draw.line((0, y, map_width, y), fill=(105, 105, 105), width=1)

    y0 = map_height + 12
    draw.text((16, y0), f"{profile.planet_name} regional analysis | {analysis.cols} x {analysis.rows} coarse regions", fill=(0, 0, 0))
    draw.text((16, y0 + 22), "Colors show broad biological productivity score; top regions are outlined. This is not a civilization/population model.", fill=(0, 0, 0))

    legend = [
        ("ocean / mostly water", (79, 134, 198)),
        ("very low", (192, 119, 93)),
        ("low", (219, 171, 91)),
        ("moderate", (226, 210, 111)),
        ("high", (141, 185, 92)),
        ("very high", (58, 132, 75)),
    ]
    x = 16
    for label, color in legend:
        draw.rectangle((x, y0 + 54, x + 22, y0 + 76), fill=color, outline=(50, 50, 50))
        draw.text((x + 30, y0 + 57), label, fill=(0, 0, 0))
        x += 185

    top_regions = [region for region in analysis.regions if region.region_id in top_ids]
    top_regions.sort(key=lambda region: region.biological_productivity_score, reverse=True)
    top_text_parts = []
    for region in top_regions[:3]:
        top_text_parts.append(
            f"{region.region_id} {region.biological_productivity_score:.0f} ({region.region_type})"
        )
    top_text = "; ".join(top_text_parts) if top_text_parts else "no productive land regions"
    draw.text((16, y0 + 94), f"Top regions: {top_text}", fill=(0, 0, 0))
    draw.text((16, y0 + 116), "See main_planet_regions.csv for every region, biome, climate, river/lake count, and score.", fill=(0, 0, 0))
    draw.text(
        (16, y0 + 138),
        f"Source grid: {terrain.width} x {terrain.height}. Regional analysis is environmental only, not population/civilization simulation.",
        fill=(0, 0, 0),
    )

    image.save(output_path, optimize=False, compress_level=0)


def _region_color(region) -> tuple[int, int, int]:
    if region.land_fraction < 0.10:
        return (79, 134, 198)
    score = region.biological_productivity_score
    if score < 15:
        return (192, 119, 93)
    if score < 35:
        return (219, 171, 91)
    if score < 55:
        return (226, 210, 111)
    if score < 75:
        return (141, 185, 92)
    return (58, 132, 75)


def _draw_orbital_map(ax, system: StarSystem, Circle, Wedge) -> None:
    star = system.star
    planets = sorted(system.planets, key=lambda p: p.orbit.semi_major_axis_au)
    outermost_orbit = max((p.orbit.semi_major_axis_au for p in planets), default=1.0)
    plot_radius = max(outermost_orbit, star.snow_line_au, star.habitable_zone_outer_au) * 1.15

    hz_outer = Wedge(
        center=(0, 0),
        r=star.habitable_zone_outer_au,
        theta1=0,
        theta2=360,
        width=max(0.001, star.habitable_zone_outer_au - star.habitable_zone_inner_au),
        facecolor="#74c476",
        alpha=0.22,
        edgecolor="none",
        label="Habitable zone",
    )
    ax.add_patch(hz_outer)

    snow = Circle((0, 0), star.snow_line_au, fill=False, edgecolor="#3182bd", linestyle="--", linewidth=1.6)
    ax.add_patch(snow)
    ax.text(star.snow_line_au, 0, " snow/frost line", color="#3182bd", fontsize=8, va="bottom")

    golden_angle = math.radians(137.507764)
    for index, planet in enumerate(planets, start=1):
        orbit = Circle((0, 0), planet.orbit.semi_major_axis_au, fill=False, edgecolor="#999999", linewidth=0.8, alpha=0.7)
        ax.add_patch(orbit)

        angle = golden_angle * index
        x = planet.orbit.semi_major_axis_au * math.cos(angle)
        y = planet.orbit.semi_major_axis_au * math.sin(angle)
        color = PLANET_CLASS_COLORS.get(planet.planet_class, "#777777")
        marker_size = _planet_marker_size(planet)
        edge_color = "#111111" if planet.is_main_planet else "#555555"
        line_width = 2.2 if planet.is_main_planet else 0.8
        ax.scatter([x], [y], s=marker_size, color=color, edgecolor=edge_color, linewidth=line_width, zorder=5)
        label = f"{index}: {planet.name}"
        if planet.is_main_planet:
            label += " MAIN"
        ax.text(x, y, "  " + label, fontsize=8, va="center")

    ax.scatter([0], [0], s=260, color="#ffd34d", edgecolor="#8c6d1f", linewidth=1.2, zorder=10)
    ax.text(0, 0, f"  {star.stellar_class} star", fontsize=9, fontweight="bold", va="center")

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-plot_radius, plot_radius)
    ax.set_ylim(-plot_radius, plot_radius)
    ax.set_xlabel("AU")
    ax.set_ylabel("AU")
    ax.set_title("Orbital architecture\n(distances to scale; body sizes symbolic)")
    ax.grid(True, linewidth=0.4, alpha=0.35)

    note = (
        f"HZ {star.habitable_zone_inner_au:.2f}-{star.habitable_zone_outer_au:.2f} AU | "
        f"Snow line {star.snow_line_au:.2f} AU"
    )
    ax.text(0.02, 0.02, note, transform=ax.transAxes, fontsize=8, va="bottom")



def _draw_star_size_panel(ax, system: StarSystem, Circle) -> None:
    star = system.star
    sun_radius = 1.0
    generated_radius = star.radius_solar
    max_radius = max(sun_radius, generated_radius) * 1.30

    sun_circle = Circle((0, 0), sun_radius, facecolor="#fdd24d", edgecolor="#a96d00", linewidth=1.2, alpha=0.95)
    ax.add_patch(sun_circle)
    ax.text(0, -(max_radius + 0.25), "Sun\n1.00 R☉", ha="center", va="top", fontsize=9)

    generated_x = max_radius * 2.4
    generated_circle = Circle((generated_x, 0), generated_radius, facecolor="#ffd34d", edgecolor="#8c6d1f", linewidth=1.4, alpha=0.95)
    ax.add_patch(generated_circle)
    ax.text(generated_x, -(max_radius + 0.25), f"Generated {star.stellar_class} star\n{generated_radius:.2f} R☉", ha="center", va="top", fontsize=9)

    details = (
        f"Mass: {star.mass_solar:.3f} M☉\n"
        f"Age: {star.age_gyr:.2f} Gyr\n"
        f"Metallicity [Fe/H]: {star.metallicity:+.2f}\n"
        f"Luminosity: {star.luminosity_solar:.3f} L☉\n"
        f"Temperature: {star.temperature_k:.0f} K"
    )
    ax.text(0.02, 0.95, details, transform=ax.transAxes, fontsize=9, va="top",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#fff7dd", edgecolor="#c9aa45", alpha=0.95))

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-max_radius * 2.0, generated_x + max_radius * 2.0)
    ax.set_ylim(-max_radius * 1.8, max_radius * 1.8)
    ax.set_title("Star size comparison\n(solar-radius scale)")
    ax.axis("off")



def _draw_planet_size_panel(ax, system: StarSystem, Circle) -> None:
    planets = sorted(system.planets, key=lambda p: p.orbit.semi_major_axis_au)
    max_radius = max((p.radius_earth for p in planets), default=1.0)
    spacing = max(3.3, max_radius * 2.5)

    x = 0.0
    for index, planet in enumerate(planets, start=1):
        radius = planet.radius_earth
        color = PLANET_CLASS_COLORS.get(planet.planet_class, "#777777")
        circle = Circle(
            (x, 0),
            radius,
            facecolor=color,
            edgecolor="#111111" if planet.is_main_planet else "#555555",
            linewidth=2.0 if planet.is_main_planet else 0.8,
            alpha=0.95,
        )
        ax.add_patch(circle)
        label = f"{index}. {planet.name}\n{planet.radius_earth:.2f} R⊕\n{planet.planet_class}"
        ax.text(x, -max_radius * 1.65, label, ha="center", va="top", fontsize=8)
        if planet.is_main_planet:
            ax.text(x, radius + max_radius * 0.25, "MAIN", ha="center", va="bottom", fontsize=8, fontweight="bold")
        x += spacing

    note = "Planet radii are to scale with each other in Earth radii."
    ax.text(0.02, 0.95, note, transform=ax.transAxes, fontsize=9, va="top")

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-spacing, max(spacing, x))
    ax.set_ylim(-max_radius * 2.1, max_radius * 2.1)
    ax.set_title("Planet size comparison\n(Earth-radius scale)")
    ax.axis("off")



def _draw_main_planet_moon_orbit(ax, system: StarSystem, Circle) -> None:
    planet = system.main_planet
    assert planet is not None and planet.moon is not None
    moon = planet.moon
    orbit = moon.orbit

    plot_radius = orbit.safe_hill_limit_km * 1.08
    roche = Circle((0, 0), orbit.roche_limit_km, fill=False, edgecolor="#d62728", linestyle="--", linewidth=1.7)
    safe_hill = Circle((0, 0), orbit.safe_hill_limit_km, fill=False, edgecolor="#3182bd", linestyle=":", linewidth=1.8)
    moon_orbit = Circle((0, 0), orbit.semi_major_axis_km, fill=False, edgecolor="#777777", linewidth=1.1)
    ax.add_patch(roche)
    ax.add_patch(safe_hill)
    ax.add_patch(moon_orbit)

    moon_angle = math.radians(48.0)
    moon_x = orbit.semi_major_axis_km * math.cos(moon_angle)
    moon_y = orbit.semi_major_axis_km * math.sin(moon_angle)

    ax.scatter([0], [0], s=420, color="#d49a58", edgecolor="#6b4f1f", linewidth=1.4, zorder=10)
    ax.scatter([moon_x], [moon_y], s=90, color=MOON_CLASS_COLORS.get(moon.moon_class, "#cccccc"), edgecolor="#444444", linewidth=1.0, zorder=10)

    ax.text(0, 0, f"  {planet.name}", fontsize=9, fontweight="bold", va="center")
    ax.text(moon_x, moon_y, f"  {moon.name}", fontsize=8, va="center")
    ax.text(orbit.roche_limit_km, 0, " Roche limit", color="#d62728", fontsize=8, va="bottom")
    ax.text(orbit.safe_hill_limit_km, 0, " Safe Hill limit", color="#3182bd", fontsize=8, va="bottom")

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-plot_radius, plot_radius)
    ax.set_ylim(-plot_radius, plot_radius)
    ax.set_xlabel("km")
    ax.set_ylabel("km")
    ax.set_title("Main planet - moon orbital layout\n(distances to scale; body sizes symbolic)")
    ax.grid(True, linewidth=0.4, alpha=0.35)
    ax.text(
        0.02,
        0.02,
        f"Moon orbit: {orbit.semi_major_axis_km:,.0f} km | Month: {orbit.orbital_period_days:.2f} days",
        transform=ax.transAxes,
        fontsize=8,
        va="bottom",
    )



def _draw_main_planet_moon_text(ax, system: StarSystem) -> None:
    planet = system.main_planet
    assert planet is not None and planet.moon is not None
    moon = planet.moon
    ax.set_title("Key details")
    ax.axis("off")
    details = (
        f"Main Planet: {planet.name}\n"
        f"  Class: {planet.planet_class}\n"
        f"  Orbit: {planet.orbit.semi_major_axis_au:.3f} AU\n"
        f"  Year: {planet.orbit.orbital_period_days:.1f} days\n"
        f"  Mass: {planet.mass_earth:.2f} M⊕\n"
        f"  Radius: {planet.radius_earth:.2f} R⊕\n"
        f"  Gravity: {planet.surface_gravity_g:.2f} g\n"
        f"  Flux: {planet.stellar_flux_earth:.2f} F⊕\n"
        f"  Equilibrium temp: {planet.equilibrium_temperature_k:.1f} K\n"
        f"  Composition: {planet.composition.composition_class}\n"
        f"  Habitability score: {planet.habitability_score:.1f}/100\n\n"
        f"Moon: {moon.name}\n"
        f"  Class: {moon.moon_class}\n"
        f"  Mass: {moon.mass_earth:.4f} M⊕\n"
        f"  Radius: {moon.radius_earth:.3f} R⊕\n"
        f"  Gravity: {moon.surface_gravity_g:.2f} g\n"
        f"  Month length: {moon.orbit.orbital_period_days:.2f} days\n"
        f"  Eccentricity: {moon.orbit.eccentricity:.3f}\n"
        f"  Tidal strength: {moon.tidal_strength_relative_earth_moon:.2f} × Earth-Moon\n"
        f"  Angular diameter: {moon.angular_diameter_degrees:.2f}°"
    )
    ax.text(
        0.02,
        0.98,
        details,
        transform=ax.transAxes,
        fontsize=8.2,
        va="top",
        family="monospace",
        bbox=dict(boxstyle="round,pad=0.45", facecolor="#f7f7f7", edgecolor="#bbbbbb", alpha=0.96),
    )



def _draw_main_planet_moon_size(ax, system: StarSystem, Circle) -> None:
    planet = system.main_planet
    assert planet is not None and planet.moon is not None
    moon = planet.moon
    ax.set_title("Size comparison\n(Earth-radius scale)")
    max_radius = max(planet.radius_earth, moon.radius_earth) * 1.35
    planet_x = 0.0
    moon_x = max_radius * 2.8
    planet_circle = Circle((planet_x, 0.0), planet.radius_earth, facecolor="#d49a58", edgecolor="#6b4f1f", linewidth=1.4)
    moon_circle = Circle((moon_x, 0.0), moon.radius_earth, facecolor=MOON_CLASS_COLORS.get(moon.moon_class, "#cccccc"), edgecolor="#555555", linewidth=1.1)
    ax.add_patch(planet_circle)
    ax.add_patch(moon_circle)
    ax.text(planet_x, -(max_radius + 0.25), f"{planet.name}\n{planet.radius_earth:.2f} R⊕", ha="center", va="top", fontsize=8)
    ax.text(moon_x, -(max_radius + 0.25), f"{moon.name}\n{moon.radius_earth:.3f} R⊕", ha="center", va="top", fontsize=8)
    ax.text(0.02, 0.06, "Circles are to scale with each other; orbit panel uses symbolic body markers.", transform=ax.transAxes, fontsize=7.8, va="bottom")
    ax.set_xlim(-max_radius * 1.5, moon_x + max_radius * 1.5)
    ax.set_ylim(-max_radius * 1.65, max_radius * 1.65)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")





def _hex_to_rgb(value: str) -> tuple[int, int, int]:
    value = value.lstrip("#")
    return int(value[0:2], 16), int(value[2:4], 16), int(value[4:6], 16)


def _lerp_color(a: tuple[int, int, int], b: tuple[int, int, int], t: float) -> tuple[int, int, int]:
    t = max(0.0, min(1.0, t))
    return (
        int(round(a[0] + (b[0] - a[0]) * t)),
        int(round(a[1] + (b[1] - a[1]) * t)),
        int(round(a[2] + (b[2] - a[2]) * t)),
    )


def _land_elevation_color(t: float) -> tuple[int, int, int]:
    t = max(0.0, min(1.0, t))
    if t < 0.22:
        return _lerp_color((200, 185, 116), (108, 155, 84), t / 0.22)
    if t < 0.55:
        return _lerp_color((108, 155, 84), (143, 125, 82), (t - 0.22) / 0.33)
    if t < 0.82:
        return _lerp_color((143, 125, 82), (180, 174, 156), (t - 0.55) / 0.27)
    return _lerp_color((180, 174, 156), (245, 245, 240), (t - 0.82) / 0.18)


def _ocean_depth_color(t: float) -> tuple[int, int, int]:
    t = max(0.0, min(1.0, t))
    return _lerp_color((104, 168, 211), (18, 61, 122), t)


def _temperature_color(t: float) -> tuple[int, int, int]:
    # Blue -> pale -> orange/red.
    if t < 0.5:
        return _lerp_color((45, 90, 200), (235, 238, 230), t * 2.0)
    return _lerp_color((235, 238, 230), (190, 30, 30), (t - 0.5) * 2.0)


def _precipitation_color(t: float) -> tuple[int, int, int]:
    # Dry tan -> green -> blue.
    if t < 0.5:
        return _lerp_color((230, 210, 135), (95, 175, 105), t * 2.0)
    return _lerp_color((95, 175, 105), (45, 95, 185), (t - 0.5) * 2.0)


def _draw_gradient_legend(draw, x: int, y: int, width: int, minimum: float, maximum: float, color_func, suffix: str) -> None:
    height = 14
    for i in range(width):
        color = color_func(i / max(1, width - 1))
        draw.line((x + i, y, x + i, y + height), fill=color)
    draw.rectangle((x, y, x + width, y + height), outline=(50, 50, 50))
    draw.text((x, y + height + 4), f"{minimum:,.0f} {suffix}", fill=(0, 0, 0))
    draw.text((x + width - 75, y + height + 4), f"{maximum:,.0f} {suffix}", fill=(0, 0, 0))


def _planet_marker_size(planet: Planet) -> float:
    if planet.planet_class == "gas_giant":
        return 150
    if planet.planet_class == "ice_giant":
        return 115
    if planet.planet_class == "mini_neptune":
        return 85
    if planet.planet_class == "super_earth":
        return 65
    if planet.planet_class == "icy_dwarf":
        return 28
    return 48



def _hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    value = hex_color.lstrip("#")
    return int(value[0:2], 16), int(value[2:4], 16), int(value[4:6], 16)


def _prepare_output_path(output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path



def _import_matplotlib():
    try:
        import matplotlib
        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
        from matplotlib.patches import Circle, Wedge
    except ImportError as exc:
        raise RuntimeError(
            "Matplotlib is required for PNG output. Install it with: pip install matplotlib"
        ) from exc
    return plt, Circle, Wedge

# ---------------------------------------------------------------------------
# Final fast raster renderers
#
# These definitions intentionally appear late in the module so they override the
# earlier experimental versions. They use NumPy arrays instead of per-cell Python
# loops, which keeps full output generation stable for 1024x512+ maps.
# ---------------------------------------------------------------------------


def _save_image_fast(image, output_path: str | Path) -> None:
    """Save a raster image, optionally downsampling very large PNGs.

    Full-resolution 4096+ maps are useful for inspection, but writing every
    diagnostic at native resolution is often slower than the simulation itself.
    The WORLDGEN_IMAGE_MAX_WIDTH environment variable lets the CLI cap raster
    PNG width without changing the underlying generated grid.
    """
    max_width_raw = os.environ.get("WORLDGEN_IMAGE_MAX_WIDTH", "").strip()
    if max_width_raw:
        try:
            max_width = int(max_width_raw)
        except ValueError:
            max_width = 0
        if max_width > 0 and image.width > max_width:
            ratio = max_width / image.width
            new_size = (max_width, max(1, int(round(image.height * ratio))))
            # Use a high-quality filter so legends/text remain readable when
            # image_max_width is used. The underlying simulation grids are not
            # downsampled; this only affects exported PNG display size.
            from PIL import Image as _PILImage
            image = image.resize(new_size, _PILImage.Resampling.LANCZOS)
    image.save(output_path, optimize=False, compress_level=0)


def _terrain_rgb_array(terrain):
    import numpy as np
    elevation = np.asarray(terrain.elevation_m, dtype=np.float32)
    land = np.asarray(terrain.is_land, dtype=bool)
    height, width = elevation.shape
    rgb = np.zeros((height, width, 3), dtype=np.uint8)
    min_elev = min(-1.0, float(terrain.min_elevation_m))
    max_elev = max(1.0, float(terrain.max_elevation_m))

    ocean_t = np.clip(np.abs(elevation) / abs(min_elev), 0.0, 1.0)
    rgb[:, :, 0] = (65 - 36 * ocean_t).astype(np.uint8)
    rgb[:, :, 1] = (135 - 66 * ocean_t).astype(np.uint8)
    rgb[:, :, 2] = (195 - 78 * ocean_t).astype(np.uint8)

    land_t = np.clip(elevation / max_elev, 0.0, 1.0)
    low = land & (land_t < 0.35)
    mid = land & (land_t >= 0.35) & (land_t < 0.72)
    high = land & (land_t >= 0.72)
    t = np.zeros_like(land_t)
    t[low] = land_t[low] / 0.35
    rgb[low, 0] = (143 + 56 * t[low]).astype(np.uint8)
    rgb[low, 1] = (165 + 28 * t[low]).astype(np.uint8)
    rgb[low, 2] = (98 + 18 * t[low]).astype(np.uint8)
    t[mid] = (land_t[mid] - 0.35) / 0.37
    rgb[mid, 0] = (199 - 55 * t[mid]).astype(np.uint8)
    rgb[mid, 1] = (193 - 44 * t[mid]).astype(np.uint8)
    rgb[mid, 2] = (116 - 48 * t[mid]).astype(np.uint8)
    t[high] = (land_t[high] - 0.72) / 0.28
    rgb[high, 0] = (144 + 100 * t[high]).astype(np.uint8)
    rgb[high, 1] = (149 + 95 * t[high]).astype(np.uint8)
    rgb[high, 2] = (68 + 120 * t[high]).astype(np.uint8)

    coast = np.frombuffer(_cached_large_coastline_mask(terrain.is_land), dtype=np.uint8).reshape((height, width)).astype(bool)
    rgb[coast] = (24, 24, 20)
    return rgb


def save_main_planet_terrain_view(system: StarSystem, output_path: str | Path) -> None:
    if system.main_planet_profile is None:
        raise RuntimeError("Terrain visualization requested, but no Main Planet profile exists.")
    from PIL import Image, ImageDraw
    output_path = _prepare_output_path(output_path)
    profile = system.main_planet_profile
    terrain = profile.terrain
    width, height = terrain.width, terrain.height
    legend_height = 132
    image = Image.new("RGB", (width, height + legend_height), "white")
    image.paste(Image.fromarray(_terrain_rgb_array(terrain), mode="RGB"), (0, 0))
    draw = ImageDraw.Draw(image)
    y0 = height + 12
    terrain_title = "artifact-repaired real Earth terrain" if str(getattr(terrain, "source", "")).startswith("real_earth") else "detailed procedural terrain"
    draw.text((16, y0), f"{profile.planet_name} {terrain_title}", fill=(0, 0, 0))
    draw.text((16, y0 + 22), f"map {terrain.width} x {terrain.height} | ocean {terrain.ocean_fraction:.1%}, land {terrain.land_fraction:.1%} | elevation {terrain.min_elevation_m:,} to {terrain.max_elevation_m:,} m | mean land {terrain.mean_land_elevation_m:,.0f} m | mean ocean {terrain.mean_ocean_depth_m:,.0f} m", fill=(0, 0, 0))
    legend = [("deep ocean", (29, 69, 117)), ("shallow sea", (65, 135, 195)), ("lowland", (143, 165, 98)), ("highland", (144, 149, 68)), ("mountain", (244, 244, 188)), ("major coastline", (24, 24, 20))]
    x = 16
    for label, color in legend:
        draw.rectangle((x, y0 + 55, x + 20, y0 + 75), fill=color, outline=(50, 50, 50))
        draw.text((x + 28, y0 + 57), label, fill=(0, 0, 0))
        x += 145
    draw.text((16, y0 + 96), "Terrain includes real/calibration or procedural relief, water masks, elevation/depth coloring, and marked major coastlines.", fill=(0, 0, 0))
    _save_image_fast(image, output_path)


def _temperature_rgb_array(values, min_t: float, max_t: float):
    import numpy as np
    arr = np.asarray(values, dtype=np.float32) / 10.0
    t = np.clip((arr - min_t) / max(1.0, max_t - min_t), 0.0, 1.0)
    rgb = np.zeros((*arr.shape, 3), dtype=np.uint8)
    low = t < 0.5
    f = np.zeros_like(t)
    f[low] = t[low] * 2.0
    rgb[low, 0] = (45 + (235 - 45) * f[low]).astype(np.uint8)
    rgb[low, 1] = (90 + (238 - 90) * f[low]).astype(np.uint8)
    rgb[low, 2] = (200 + (230 - 200) * f[low]).astype(np.uint8)
    high = ~low
    f[high] = (t[high] - 0.5) * 2.0
    rgb[high, 0] = (235 + (190 - 235) * f[high]).astype(np.uint8)
    rgb[high, 1] = (238 + (30 - 238) * f[high]).astype(np.uint8)
    rgb[high, 2] = (230 + (30 - 230) * f[high]).astype(np.uint8)
    return rgb


def save_main_planet_temperature_view(system: StarSystem, output_path: str | Path) -> None:
    profile, climate, terrain = _require_main_planet_climate(system)
    from PIL import Image, ImageDraw
    output_path = _prepare_output_path(output_path)
    width, height = climate.width, climate.height
    legend_height = 96
    image = Image.new("RGB", (width, height + legend_height), "white")
    image.paste(Image.fromarray(_temperature_rgb_array(climate.annual_mean_temp_c_x10, climate.min_temp_c, climate.max_temp_c), mode="RGB"), (0, 0))
    draw = ImageDraw.Draw(image)
    y0 = height + 12
    draw.text((16, y0), f"{profile.planet_name} annual mean temperature", fill=(0, 0, 0))
    draw.text((16, y0 + 22), f"mean land {climate.mean_land_temp_c:.1f} °C | mean ocean {climate.mean_ocean_temp_c:.1f} °C | range {climate.min_temp_c:.1f} to {climate.max_temp_c:.1f} °C | ocean {terrain.ocean_fraction:.1%}", fill=(0, 0, 0))
    _draw_gradient_legend(draw, 16, y0 + 52, 260, climate.min_temp_c, climate.max_temp_c, _temperature_color, "°C")
    _save_image_fast(image, output_path)


def _precip_rgb_array(values, max_p: int, land_mask=None):
    import numpy as np
    arr = np.asarray(values, dtype=np.float32)
    t = np.clip(np.log1p(arr) / max(1e-6, math.log1p(max(1, max_p))), 0.0, 1.0)
    rgb = np.zeros((*arr.shape, 3), dtype=np.uint8)
    low = t < 0.5
    f = np.zeros_like(t)
    rgb[:, :, :] = (70, 110, 150)  # ocean/background when land mask is supplied
    active = np.ones_like(t, dtype=bool) if land_mask is None else np.asarray(land_mask, dtype=bool)
    dry = low & active
    f[dry] = t[dry] * 2.0
    rgb[dry, 0] = (230 + (95 - 230) * f[dry]).astype(np.uint8)
    rgb[dry, 1] = (210 + (175 - 210) * f[dry]).astype(np.uint8)
    rgb[dry, 2] = (135 + (105 - 135) * f[dry]).astype(np.uint8)
    wet = (~low) & active
    f[wet] = (t[wet] - 0.5) * 2.0
    rgb[wet, 0] = (95 + (45 - 95) * f[wet]).astype(np.uint8)
    rgb[wet, 1] = (175 + (95 - 175) * f[wet]).astype(np.uint8)
    rgb[wet, 2] = (105 + (185 - 105) * f[wet]).astype(np.uint8)
    return rgb



def save_main_planet_precipitation_view(system: StarSystem, output_path: str | Path) -> None:
    profile, climate, terrain = _require_main_planet_climate(system)
    from PIL import Image, ImageDraw
    import numpy as np
    output_path = _prepare_output_path(output_path)
    width, height = climate.width, climate.height
    legend_height = 96
    image = Image.new("RGB", (width, height + legend_height), "white")
    land = np.asarray(terrain.is_land, dtype=bool)
    rgb = _precip_rgb_array(climate.annual_precip_mm, climate.max_precip_mm, land_mask=land)
    coast = np.frombuffer(_cached_large_coastline_mask(terrain.is_land), dtype=np.uint8).reshape((height, width)).astype(bool)
    rgb[coast] = (25, 25, 22)
    image.paste(Image.fromarray(rgb, mode="RGB"), (0, 0))
    draw = ImageDraw.Draw(image)
    y0 = height + 12
    draw.text((16, y0), f"{profile.planet_name} annual land precipitation", fill=(0, 0, 0))
    draw.text((16, y0 + 22), f"land only shown | mean land {climate.mean_land_precip_mm:,.0f} mm/year | land range shown with global max {climate.max_precip_mm:,} mm/year | oceans masked blue-gray", fill=(0, 0, 0))
    _draw_gradient_legend(draw, 16, y0 + 52, 260, 0, climate.max_precip_mm, _precipitation_color, "mm/year")
    _save_image_fast(image, output_path)


def save_main_planet_koppen_view(system: StarSystem, output_path: str | Path) -> None:
    if system.main_planet_profile is None:
        raise RuntimeError("Köppen visualization requested, but no Main Planet profile exists.")
    from PIL import Image, ImageDraw
    import numpy as np
    output_path = _prepare_output_path(output_path)
    profile = system.main_planet_profile
    climate = profile.climate
    terrain = profile.terrain
    width, height = climate.width, climate.height
    legend_height = 170
    codes_arr = np.asarray(climate.koppen_classification, dtype=object)
    rgb = np.zeros((height, width, 3), dtype=np.uint8)
    present_counts: dict[str, int] = {}
    for code, hex_color in KOPPEN_COLORS.items():
        mask = codes_arr == code
        count = int(mask.sum())
        if count:
            rgb[mask] = _hex_to_rgb(hex_color)
            present_counts[code] = count
    coast = np.frombuffer(_cached_large_coastline_mask(terrain.is_land), dtype=np.uint8).reshape((height, width)).astype(bool)
    rgb[coast] = (8, 8, 8)
    image = Image.new("RGB", (width, height + legend_height), "white")
    image.paste(Image.fromarray(rgb, mode="RGB"), (0, 0))
    draw = ImageDraw.Draw(image)
    y0 = height + 10
    draw.text((16, y0), f"{profile.planet_name} simplified Köppen climate classes", fill=(0, 0, 0))
    dominant = sorted(climate.koppen_summary.items(), key=lambda item: item[1], reverse=True)[:8]
    dominant_text = ", ".join(f"{_koppen_label(code)} {count:,}" for code, count in dominant) if dominant else "no land climate cells"
    draw.text((16, y0 + 22), f"Dominant land climates: {dominant_text}", fill=(0, 0, 0))
    draw.text((16, y0 + 44), "Ocean cells use deep navy; major coasts are outlined in black and excluded from the dominant land-climate summary.", fill=(0, 0, 0))
    x = 16; y = y0 + 72
    draw.rectangle((x, y, x + 18, y + 18), fill=_hex_to_rgb(KOPPEN_COLORS["O"]), outline=(50, 50, 50))
    draw.text((x + 24, y + 2), _koppen_label("O"), fill=(0, 0, 0))
    x += 120
    draw.rectangle((x, y, x + 18, y + 18), fill=(8, 8, 8), outline=(50, 50, 50))
    draw.text((x + 24, y + 2), "coast", fill=(0, 0, 0))
    x += 120
    for code in KOPPEN_COLORS:
        if code == "O" or code not in present_counts:
            continue
        color = _hex_to_rgb(KOPPEN_COLORS[code])
        draw.rectangle((x, y, x + 18, y + 18), fill=color, outline=(50, 50, 50))
        draw.text((x + 24, y + 2), _koppen_label(code), fill=(0, 0, 0))
        x += 270
        if x + 260 > width:
            x = 16; y += 28
    draw.text((16, height + legend_height - 28), "Köppen classes are simplified approximations from annual and seasonal values.", fill=(0, 0, 0))
    _save_image_fast(image, output_path)


def save_main_planet_biome_view(system: StarSystem, output_path: str | Path) -> None:
    if system.main_planet_profile is None:
        raise RuntimeError("Biome visualization requested, but no Main Planet profile exists.")
    from PIL import Image, ImageDraw
    import numpy as np
    output_path = _prepare_output_path(output_path)
    profile = system.main_planet_profile
    biomes = profile.biomes
    width, height = biomes.width, biomes.height
    legend_height = 175
    biome_arr = np.asarray(biomes.biome_classification, dtype=object)
    rgb = np.zeros((height, width, 3), dtype=np.uint8)
    for name, hex_color in BIOME_COLORS.items():
        mask = biome_arr == name
        if mask.any():
            rgb[mask] = _hex_to_rgb(hex_color)
    image = Image.new("RGB", (width, height + legend_height), "white")
    image.paste(Image.fromarray(rgb, mode="RGB"), (0, 0))
    draw = ImageDraw.Draw(image)
    y0 = height + 10
    draw.text((16, y0), f"{profile.planet_name} biomes / ecoregions", fill=(0, 0, 0))
    dominant = sorted(((name, count) for name, count in biomes.biome_summary.items() if name != "ocean"), key=lambda item: item[1], reverse=True)[:6]
    dominant_text = ", ".join(f"{name} {count:,}" for name, count in dominant) if dominant else "no land biome cells"
    draw.text((16, y0 + 22), f"Dominant land biomes: {dominant_text}", fill=(0, 0, 0))
    draw.text((16, y0 + 44), "Classification layer only: assumes land-stage life; does not simulate evolution or species.", fill=(0, 0, 0))
    present = [name for name in BIOME_COLORS if name in biomes.biome_summary and name != "ocean"]
    x = 16; y = y0 + 72
    for name in present[:18]:
        color = _hex_to_rgb(BIOME_COLORS[name])
        draw.rectangle((x, y, x + 18, y + 18), fill=color, outline=(50, 50, 50))
        draw.text((x + 24, y + 2), name, fill=(0, 0, 0))
        x += 245
        if x + 230 > width:
            x = 16; y += 28
    _save_image_fast(image, output_path)


def save_main_planet_moon_view(system: StarSystem, output_path: str | Path) -> None:
    """Save a dedicated Main Planet and moon view using Pillow only."""
    if system.main_planet is None or system.main_planet.moon is None:
        raise RuntimeError("Main planet moon visualization requested, but no main planet moon exists.")
    from PIL import Image, ImageDraw
    import math as _math
    output_path = _prepare_output_path(output_path)
    planet = system.main_planet
    moon = planet.moon
    width, height = 1400, 760
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.text((24, 18), f"Main Planet and Moon - {planet.name} & {moon.name}", fill=(0, 0, 0))

    # Orbit panel: distances scaled to the safe Hill limit; body sizes symbolic.
    cx, cy = 360, 370
    orbit_max = max(1.0, moon.orbit.safe_hill_limit_km)
    scale = 300.0 / orbit_max
    roche_r = max(8, int(moon.orbit.roche_limit_km * scale))
    safe_r = int(moon.orbit.safe_hill_limit_km * scale)
    orbit_r = int(moon.orbit.semi_major_axis_km * scale)
    draw.ellipse((cx - safe_r, cy - safe_r, cx + safe_r, cy + safe_r), outline=(45, 125, 190), width=2)
    draw.ellipse((cx - roche_r, cy - roche_r, cx + roche_r, cy + roche_r), outline=(210, 45, 45), width=2)
    draw.ellipse((cx - orbit_r, cy - orbit_r, cx + orbit_r, cy + orbit_r), outline=(105, 105, 105), width=2)
    draw.ellipse((cx - 22, cy - 22, cx + 22, cy + 22), fill=(212, 154, 88), outline=(90, 65, 25), width=2)
    angle = _math.radians(45)
    mx = int(cx + orbit_r * _math.cos(angle)); my = int(cy - orbit_r * _math.sin(angle))
    draw.ellipse((mx - 9, my - 9, mx + 9, my + 9), fill=(185, 195, 200), outline=(70, 70, 70), width=1)
    draw.text((cx + 28, cy - 8), planet.name, fill=(0, 0, 0))
    draw.text((mx + 12, my - 6), moon.name, fill=(0, 0, 0))
    draw.text((80, 690), f"Moon orbit {moon.orbit.semi_major_axis_km:,.0f} km | month {moon.orbit.orbital_period_days:.2f} days | Roche {moon.orbit.roche_limit_km:,.0f} km | safe Hill {moon.orbit.safe_hill_limit_km:,.0f} km", fill=(0, 0, 0))

    # Size comparison panel: planet and moon on Earth-radius relative scale.
    sx, sy = 920, 520
    max_r = max(planet.radius_earth, moon.radius_earth)
    size_scale = 110.0 / max_r
    pr = int(planet.radius_earth * size_scale)
    mr = max(5, int(moon.radius_earth * size_scale))
    draw.ellipse((sx - pr, sy - pr, sx + pr, sy + pr), fill=(212, 154, 88), outline=(90, 65, 25), width=2)
    draw.ellipse((sx + 300 - mr, sy - mr, sx + 300 + mr, sy + mr), fill=(185, 195, 200), outline=(70, 70, 70), width=1)
    draw.text((sx - 70, sy + pr + 16), f"{planet.name}\n{planet.radius_earth:.2f} R⊕", fill=(0, 0, 0))
    draw.text((sx + 250, sy + max(pr, mr) + 16), f"{moon.name}\n{moon.radius_earth:.3f} R⊕", fill=(0, 0, 0))

    details = [
        f"Main Planet: {planet.name}",
        f"  Class: {planet.planet_class}",
        f"  Orbit: {planet.orbit.semi_major_axis_au:.3f} AU; year {planet.orbit.orbital_period_days:.1f} days",
        f"  Mass: {planet.mass_earth:.2f} M⊕; radius {planet.radius_earth:.2f} R⊕; gravity {planet.surface_gravity_g:.2f} g",
        f"  Flux: {planet.stellar_flux_earth:.2f} F⊕; equilibrium temp {planet.equilibrium_temperature_k:.1f} K",
        f"  Habitability score: {planet.habitability_score:.1f}/100",
        "",
        f"Moon: {moon.name}",
        f"  Class: {moon.moon_class}",
        f"  Mass: {moon.mass_earth:.4f} M⊕; radius {moon.radius_earth:.3f} R⊕; gravity {moon.surface_gravity_g:.2f} g",
        f"  Tidal strength: {moon.tidal_strength_relative_earth_moon:.2f} × Earth-Moon",
        f"  Angular diameter: {moon.angular_diameter_degrees:.2f}°",
    ]
    y = 88
    for line in details:
        draw.text((780, y), line, fill=(0, 0, 0))
        y += 24
    _save_image_fast(image, output_path)


def save_system_size_chart(system: StarSystem, output_path: str | Path) -> None:
    """Save star/planet size comparison using Pillow only."""
    from PIL import Image, ImageDraw
    output_path = _prepare_output_path(output_path)
    planets = sorted(system.planets, key=lambda p: p.orbit.semi_major_axis_au)
    width, height = 1500, 720
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.text((24, 18), f"Body Size Comparison - Seed {system.seed}", fill=(0, 0, 0))
    draw.text((24, 55), "Star uses solar-radius comparison; planets use Earth-radius comparison.", fill=(0, 0, 0))
    # Star comparison
    star_y = 235
    sun_r = 95
    gen_r = max(15, int(sun_r * system.star.radius_solar))
    draw.ellipse((80 - sun_r, star_y - sun_r, 80 + sun_r, star_y + sun_r), fill=(250, 210, 75), outline=(130, 90, 20), width=2)
    draw.text((45, star_y + sun_r + 12), "Sun\n1.00 R☉", fill=(0, 0, 0))
    draw.ellipse((320 - gen_r, star_y - gen_r, 320 + gen_r, star_y + gen_r), fill=(255, 211, 77), outline=(130, 90, 20), width=2)
    draw.text((270, star_y + sun_r + 12), f"Generated {system.star.stellar_class} star\n{system.star.radius_solar:.2f} R☉", fill=(0, 0, 0))
    star_details = f"Mass {system.star.mass_solar:.3f} M☉ | Age {system.star.age_gyr:.2f} Gyr | Luminosity {system.star.luminosity_solar:.3f} L☉ | Temp {system.star.temperature_k:.0f} K"
    draw.text((24, 420), star_details, fill=(0, 0, 0))
    # Planet row
    max_radius = max((p.radius_earth for p in planets), default=1.0)
    scale = 75.0 / max_radius
    base_x = 610
    spacing = max(110, int(820 / max(1, len(planets) - 1))) if len(planets) > 1 else 130
    py = 310
    for i, planet in enumerate(planets, start=1):
        x = base_x + (i - 1) * spacing
        r = max(4, int(planet.radius_earth * scale))
        color = _hex_to_rgb(PLANET_CLASS_COLORS.get(planet.planet_class, "#888888"))
        outline = (0, 0, 0) if planet.is_main_planet else (80, 80, 80)
        draw.ellipse((x - r, py - r, x + r, py + r), fill=color, outline=outline, width=3 if planet.is_main_planet else 1)
        if planet.is_main_planet:
            draw.text((x - 24, py - r - 28), "MAIN", fill=(0, 0, 0))
        draw.text((x - 42, py + max(r, 75) + 18), f"{i}. {planet.name}\n{planet.radius_earth:.2f} R⊕\n{planet.planet_class}", fill=(0, 0, 0))
    _save_image_fast(image, output_path)

# ---------------------------------------------------------------------------
# Final Pillow replacements for the small system figures.
# These override earlier Matplotlib versions to avoid backend/layout hangs when
# combined with large raster map generation.
# ---------------------------------------------------------------------------


def _safe_text(draw, xy, text, fill=(0, 0, 0)):
    draw.text(xy, str(text), fill=fill)


def save_system_orbit_map(system: StarSystem, output_path: str | Path) -> None:  # type: ignore[override]
    from PIL import Image, ImageDraw
    import math
    output_path = _prepare_output_path(output_path)
    size = 1100
    legend_h = 110
    cx = cy = size // 2
    img = Image.new("RGB", (size, size + legend_h), "white")
    draw = ImageDraw.Draw(img)
    star = system.star
    planets = sorted(system.planets, key=lambda p: p.orbit.semi_major_axis_au)
    outer = max([p.orbit.semi_major_axis_au for p in planets] + [star.snow_line_au, star.habitable_zone_outer_au, 1.0])
    scale = (size * 0.43) / (outer * 1.08)

    def rr(au: float) -> int:
        return int(round(au * scale))

    # HZ annulus approximated by two filled circles.
    outer_r = rr(star.habitable_zone_outer_au)
    inner_r = rr(star.habitable_zone_inner_au)
    draw.ellipse((cx - outer_r, cy - outer_r, cx + outer_r, cy + outer_r), fill=(219, 242, 218), outline=None)
    draw.ellipse((cx - inner_r, cy - inner_r, cx + inner_r, cy + inner_r), fill="white", outline=None)

    snow_r = rr(star.snow_line_au)
    _draw_dashed_circle(draw, cx, cy, snow_r, fill=(40, 115, 190))

    golden = math.radians(137.507764)
    for i, p in enumerate(planets, start=1):
        r = rr(p.orbit.semi_major_axis_au)
        draw.ellipse((cx - r, cy - r, cx + r, cy + r), outline=(175, 175, 175), width=1)
        a = golden * i
        x = int(cx + r * math.cos(a)); y = int(cy + r * math.sin(a))
        color = _hex_to_rgb(PLANET_CLASS_COLORS.get(p.planet_class, "#777777"))
        rad = 8 if not p.is_main_planet else 11
        draw.ellipse((x - rad, y - rad, x + rad, y + rad), fill=color, outline=(15, 15, 15), width=2 if p.is_main_planet else 1)
        _safe_text(draw, (x + 12, y - 7), f"{i}: {p.name}" + (" MAIN" if p.is_main_planet else ""))

    draw.ellipse((cx - 18, cy - 18, cx + 18, cy + 18), fill=(255, 211, 77), outline=(120, 90, 20), width=2)
    _safe_text(draw, (cx + 22, cy - 8), f"{star.stellar_class} star")

    y0 = size + 12
    _safe_text(draw, (16, y0), f"Star System Orbital Map - Seed {system.seed}")
    _safe_text(draw, (16, y0 + 24), f"Orbital distances to scale in AU; bodies symbolic. HZ {star.habitable_zone_inner_au:.2f}-{star.habitable_zone_outer_au:.2f} AU; snow/frost line {star.snow_line_au:.2f} AU")
    draw.rectangle((16, y0 + 54, 36, y0 + 74), fill=(219, 242, 218), outline=(80, 120, 80)); _safe_text(draw, (44, y0 + 56), "habitable zone")
    draw.line((190, y0 + 64, 230, y0 + 64), fill=(40, 115, 190), width=2); _safe_text(draw, (238, y0 + 56), "snow/frost line")
    _save_image_fast(img, output_path)
    _write_map_legend_sidecar(
        output_path,
        title="Star system orbital map",
        description="Orbital layout with habitable zone and snow/frost line. This overview image includes labels for readability.",
        items=[("habitable zone", (219, 242, 218)), ("snow/frost line", (40, 115, 190)), ("planet markers", (120, 160, 210))],
        stats={"seed": system.seed, "planet_count": len(system.planets), "architecture": getattr(system, "architecture", "unspecified")},
        notes=["Overview figure; not a terrain raster."],
    )


def save_system_size_chart(system: StarSystem, output_path: str | Path) -> None:  # type: ignore[override]
    from PIL import Image, ImageDraw
    output_path = _prepare_output_path(output_path)
    width, height = 1500, 620
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    star = system.star
    planets = sorted(system.planets, key=lambda p: p.orbit.semi_major_axis_au)
    _safe_text(draw, (20, 18), f"Body Size Comparison - Seed {system.seed}")
    _safe_text(draw, (20, 48), "Star panel uses solar radii; planet panel uses Earth radii.")

    # Star comparison.
    sx = 150; sy = 230; star_scale = 95
    sun_r = int(star_scale)
    gen_r = int(star.radius_solar * star_scale)
    draw.ellipse((sx - sun_r, sy - sun_r, sx + sun_r, sy + sun_r), fill=(253, 210, 77), outline=(160, 100, 0), width=2)
    _safe_text(draw, (sx - 38, sy + sun_r + 12), "Sun\n1.00 R_sun")
    gx = 390
    draw.ellipse((gx - gen_r, sy - gen_r, gx + gen_r, sy + gen_r), fill=(255, 211, 77), outline=(120, 90, 20), width=2)
    _safe_text(draw, (gx - 70, sy + sun_r + 12), f"Generated {star.stellar_class}\n{star.radius_solar:.2f} R_sun")
    _safe_text(draw, (20, 420), f"Mass {star.mass_solar:.3f} M_sun | Luminosity {star.luminosity_solar:.3f} L_sun | Temperature {star.temperature_k:.0f} K")

    # Planet comparison.
    max_r = max([p.radius_earth for p in planets] + [1.0])
    pscale = min(52, 115 / max_r)
    x = 650
    baseline = 300
    spacing = max(115, int(max_r * pscale * 2.4))
    for i, p in enumerate(planets, start=1):
        r = max(5, int(p.radius_earth * pscale))
        color = _hex_to_rgb(PLANET_CLASS_COLORS.get(p.planet_class, "#777777"))
        draw.ellipse((x - r, baseline - r, x + r, baseline + r), fill=color, outline=(15, 15, 15), width=3 if p.is_main_planet else 1)
        if p.is_main_planet:
            _safe_text(draw, (x - 20, baseline - r - 28), "MAIN")
        _safe_text(draw, (x - 45, baseline + max(70, r + 16)), f"{i}. {p.name}\n{p.radius_earth:.2f} R_earth\n{p.planet_class}")
        x += spacing
    _save_image_fast(img, output_path)
    _write_map_legend_sidecar(
        output_path,
        title="System body size comparison",
        description="Star and planet symbolic size comparison. This overview image includes labels for readability.",
        items=[("star", (253, 210, 77)), ("planet markers", (120, 160, 210)), ("Main Planet outline", (20, 20, 20))],
        stats={"seed": system.seed, "planet_count": len(system.planets)},
        notes=["Overview figure; not a terrain raster."],
    )


def save_main_planet_moon_view(system: StarSystem, output_path: str | Path) -> None:  # type: ignore[override]
    from PIL import Image, ImageDraw
    import math
    planet = system.main_planet
    if planet is None or planet.moon is None:
        raise RuntimeError("Main planet moon visualization requested, but no main planet moon exists.")
    moon = planet.moon
    output_path = _prepare_output_path(output_path)
    width, height = 1500, 820
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    _safe_text(draw, (20, 18), f"Main Planet and Moon - {planet.name} & {moon.name}")

    cx, cy = 390, 390
    max_km = moon.orbit.safe_hill_limit_km * 1.08
    scale = 330 / max_km
    roche_r = int(moon.orbit.roche_limit_km * scale)
    safe_r = int(moon.orbit.safe_hill_limit_km * scale)
    orbit_r = int(moon.orbit.semi_major_axis_km * scale)
    draw.ellipse((cx - safe_r, cy - safe_r, cx + safe_r, cy + safe_r), outline=(50, 130, 205), width=2)
    draw.ellipse((cx - orbit_r, cy - orbit_r, cx + orbit_r, cy + orbit_r), outline=(115, 115, 115), width=2)
    draw.ellipse((cx - roche_r, cy - roche_r, cx + roche_r, cy + roche_r), outline=(210, 45, 45), width=2)
    draw.ellipse((cx - 18, cy - 18, cx + 18, cy + 18), fill=(212, 154, 88), outline=(100, 70, 20), width=2)
    ma = math.radians(48)
    mx = int(cx + orbit_r * math.cos(ma)); my = int(cy + orbit_r * math.sin(ma))
    draw.ellipse((mx - 8, my - 8, mx + 8, my + 8), fill=(175, 190, 200), outline=(50, 50, 50))
    _safe_text(draw, (cx + 24, cy - 8), planet.name)
    _safe_text(draw, (mx + 12, my - 8), moon.name)
    _safe_text(draw, (20, 735), f"Distances to scale in km; body markers symbolic. Moon orbit {moon.orbit.semi_major_axis_km:,.0f} km | Roche {moon.orbit.roche_limit_km:,.0f} km | Safe Hill {moon.orbit.safe_hill_limit_km:,.0f} km")

    details = [
        f"Main Planet: {planet.name}",
        f"Class: {planet.planet_class}",
        f"Orbit: {planet.orbit.semi_major_axis_au:.3f} AU | Year: {planet.orbit.orbital_period_days:.1f} days",
        f"Mass: {planet.mass_earth:.2f} M_earth | Radius: {planet.radius_earth:.2f} R_earth | Gravity: {planet.surface_gravity_g:.2f} g",
        f"Flux: {planet.stellar_flux_earth:.2f} F_earth | Equilibrium temp: {planet.equilibrium_temperature_k:.1f} K",
        f"Composition: {planet.composition.composition_class}",
        "",
        f"Moon: {moon.name}",
        f"Class: {moon.moon_class}",
        f"Mass: {moon.mass_earth:.4f} M_earth | Radius: {moon.radius_earth:.3f} R_earth | Gravity: {moon.surface_gravity_g:.2f} g",
        f"Month: {moon.orbit.orbital_period_days:.2f} days | Eccentricity: {moon.orbit.eccentricity:.3f}",
        f"Tidal strength: {moon.tidal_strength_relative_earth_moon:.2f} x Earth-Moon | Angular diameter: {moon.angular_diameter_degrees:.2f} deg",
    ]
    y = 100
    for line in details:
        _safe_text(draw, (820, y), line)
        y += 28

    # Size comparison.
    py = 610; px = 940
    s = 55
    pr = int(planet.radius_earth * s); mr = max(5, int(moon.radius_earth * s))
    draw.ellipse((px - pr, py - pr, px + pr, py + pr), fill=(212, 154, 88), outline=(80, 55, 15), width=2)
    draw.ellipse((px + 270 - mr, py - mr, px + 270 + mr, py + mr), fill=(180, 190, 200), outline=(50, 50, 50), width=1)
    _safe_text(draw, (px - 50, py + pr + 12), f"{planet.name}\n{planet.radius_earth:.2f} R_earth")
    _safe_text(draw, (px + 220, py + pr + 12), f"{moon.name}\n{moon.radius_earth:.3f} R_earth")
    _save_image_fast(img, output_path)
    _write_map_legend_sidecar(
        output_path,
        title="Main Planet and Moon overview",
        description="Main Planet, moon orbit, Roche limit, and safe Hill limit overview. This image includes labels for readability.",
        items=[("Main Planet", (212, 154, 88)), ("moon", (175, 190, 200)), ("safe Hill limit", (50, 130, 205)), ("Roche limit", (210, 45, 45))],
        stats={"main_planet": planet.name, "moon": moon.name, "moon_orbit_km": round(moon.orbit.semi_major_axis_km, 1)},
        notes=["Overview figure; not a terrain raster."],
    )


def _draw_dashed_circle(draw, cx: int, cy: int, r: int, fill=(0, 0, 0), dash_degrees: int = 8) -> None:
    import math
    pts = []
    for start in range(0, 360, dash_degrees * 2):
        end = start + dash_degrees
        seg = []
        for deg in range(start, end + 1, 2):
            a = math.radians(deg)
            seg.append((cx + r * math.cos(a), cy + r * math.sin(a)))
        if len(seg) >= 2:
            draw.line(seg, fill=fill, width=2)

# ---------------------------------------------------------------------------
# Update 27 diagnostic maps
# ---------------------------------------------------------------------------


def _diagnostic_downsample_dimensions(width: int, height: int) -> tuple[int, int, int]:
    """Return drawing dimensions and stride for diagnostic overlays.

    The underlying simulation may be 4096x2048 or larger. Diagnostic maps keep
    the same aspect ratio but cap raw image size so progress remains practical.
    """
    max_w = 2048
    if width <= max_w:
        return width, height, 1
    stride = max(1, int(round(width / max_w)))
    return max(1, width // stride), max(1, height // stride), stride


def _simple_terrain_background(terrain, stride: int = 1):
    import numpy as np
    elev = np.asarray(terrain.elevation_m, dtype=np.float32)[::stride, ::stride]
    land = np.asarray(terrain.is_land, dtype=bool)[::stride, ::stride]
    h, w = elev.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    rgb[:, :, :] = (45, 89, 128)
    low = land & (elev < 450)
    mid = land & (elev >= 450) & (elev < 1500)
    high = land & (elev >= 1500)
    rgb[low] = (142, 162, 104)
    rgb[mid] = (166, 153, 92)
    rgb[high] = (178, 174, 145)
    return rgb


def _draw_arrow(draw, x1: float, y1: float, x2: float, y2: float, fill, width: int = 2) -> None:
    import math
    draw.line((x1, y1, x2, y2), fill=fill, width=width)
    ang = math.atan2(y2 - y1, x2 - x1)
    size = max(5, width * 3)
    for delta in (2.55, -2.55):
        ax = x2 + math.cos(ang + delta) * size
        ay = y2 + math.sin(ang + delta) * size
        draw.line((x2, y2, ax, ay), fill=fill, width=width)


def _diagnostic_wind_vector(lat_degrees: float) -> tuple[float, float]:
    abs_lat = abs(lat_degrees)
    hemisphere = 1.0 if lat_degrees >= 0.0 else -1.0
    if abs_lat < 18.0:
        return 0.55 * hemisphere, -1.0
    if abs_lat < 30.0:
        return 0.30 * hemisphere, -0.95
    if abs_lat < 60.0:
        return -0.35 * hemisphere, 1.0
    return 0.35 * hemisphere, -0.8


def save_main_planet_wind_currents_view(system: StarSystem, output_path: str | Path) -> None:
    if system.main_planet_profile is None:
        raise RuntimeError("Wind diagnostic requested, but no Main Planet profile exists.")
    from PIL import Image, ImageDraw
    output_path = _prepare_output_path(output_path)
    terrain = system.main_planet_profile.terrain
    out_w, out_h, stride = _diagnostic_downsample_dimensions(terrain.width, terrain.height)
    image = Image.fromarray(_simple_terrain_background(terrain, stride), mode="RGB")
    draw = ImageDraw.Draw(image)
    step_x = max(64, out_w // 28)
    step_y = max(48, out_h // 16)
    scale = min(step_x, step_y) * 0.42
    for y in range(step_y // 2, out_h, step_y):
        lat = 90.0 - (y + 0.5) * 180.0 / out_h
        dr, dc = _diagnostic_wind_vector(lat)
        for x in range(step_x // 2, out_w, step_x):
            _draw_arrow(draw, x, y, x + dc * scale, y + dr * scale, fill=(245, 245, 245), width=2)
    draw.rectangle((8, 8, min(out_w - 8, 760), 54), fill=(255, 255, 255), outline=(30, 30, 30))
    draw.text((18, 18), "Prevailing wind diagnostic: trades, westerlies, polar easterlies", fill=(0, 0, 0))
    _save_image_fast(image, output_path)


def save_main_planet_ocean_currents_view(system: StarSystem, output_path: str | Path) -> None:
    if system.main_planet_profile is None:
        raise RuntimeError("Ocean-current diagnostic requested, but no Main Planet profile exists.")
    from PIL import Image, ImageDraw
    import numpy as np
    output_path = _prepare_output_path(output_path)
    terrain = system.main_planet_profile.terrain
    out_w, out_h, stride = _diagnostic_downsample_dimensions(terrain.width, terrain.height)
    land = np.asarray(terrain.is_land, dtype=bool)[::stride, ::stride]
    rgb = np.zeros((out_h, out_w, 3), dtype=np.uint8)
    rgb[:, :, :] = (35, 88, 140)
    rgb[land] = (115, 130, 94)
    image = Image.fromarray(rgb, mode="RGB")
    draw = ImageDraw.Draw(image)
    step_x = max(80, out_w // 24)
    step_y = max(60, out_h // 14)
    scale = min(step_x, step_y) * 0.45
    for y in range(step_y // 2, out_h, step_y):
        lat = 90.0 - (y + 0.5) * 180.0 / out_h
        if abs(lat) < 5:
            continue
        for x in range(step_x // 2, out_w, step_x):
            if land[y, x]:
                continue
            # Simplified subtropical/polar gyre direction. Warm western-boundary
            # currents and cold eastern-boundary currents are implied by coast side.
            hemi = 1 if lat >= 0 else -1
            gyre = 1 if abs(lat) < 45 else -1
            dc = gyre * (1 if (x / out_w) < 0.5 else -1)
            dr = -hemi * 0.45 * gyre
            color = (255, 170, 70) if dc > 0 else (90, 185, 255)
            _draw_arrow(draw, x, y, x + dc * scale, y + dr * scale, fill=color, width=2)
    draw.rectangle((8, 8, min(out_w - 8, 780), 58), fill=(255, 255, 255), outline=(30, 30, 30))
    draw.text((18, 18), "Ocean-current diagnostic: simplified gyres and warm/cold current influence", fill=(0, 0, 0))
    _save_image_fast(image, output_path)


def save_main_planet_moisture_transport_view(system: StarSystem, output_path: str | Path) -> None:
    if system.main_planet_profile is None:
        raise RuntimeError("Moisture diagnostic requested, but no Main Planet profile exists.")
    from PIL import Image, ImageDraw
    import numpy as np
    output_path = _prepare_output_path(output_path)
    profile = system.main_planet_profile
    terrain = profile.terrain
    climate = profile.climate
    out_w, out_h, stride = _diagnostic_downsample_dimensions(terrain.width, terrain.height)
    precip = np.asarray(climate.annual_precip_mm, dtype=np.float32)[::stride, ::stride]
    land = np.asarray(terrain.is_land, dtype=bool)[::stride, ::stride]
    t = np.clip(np.log1p(precip) / max(1e-6, np.log1p(max(1, float(np.nanmax(precip))))), 0.0, 1.0)
    rgb = np.zeros((out_h, out_w, 3), dtype=np.uint8)
    rgb[:, :, :] = (52, 93, 130)
    rgb[land, 0] = (235 - 150 * t[land]).astype(np.uint8)
    rgb[land, 1] = (214 - 60 * t[land]).astype(np.uint8)
    rgb[land, 2] = (126 + 92 * t[land]).astype(np.uint8)
    image = Image.fromarray(rgb, mode="RGB")
    draw = ImageDraw.Draw(image)
    step_x = max(90, out_w // 22)
    step_y = max(68, out_h // 12)
    scale = min(step_x, step_y) * 0.36
    for y in range(step_y // 2, out_h, step_y):
        lat = 90.0 - (y + 0.5) * 180.0 / out_h
        dr, dc = _diagnostic_wind_vector(lat)
        for x in range(step_x // 2, out_w, step_x):
            moisture = t[y, x]
            color = (210, 240, 255) if moisture > 0.55 else (245, 230, 160)
            _draw_arrow(draw, x, y, x + dc * scale, y + dr * scale, fill=color, width=2)
    draw.rectangle((8, 8, min(out_w - 8, 850), 58), fill=(255, 255, 255), outline=(30, 30, 30))
    draw.text((18, 18), "Moisture transport diagnostic: land precipitation field with prevailing transport arrows", fill=(0, 0, 0))
    _save_image_fast(image, output_path)


def save_main_planet_rain_shadow_view(system: StarSystem, output_path: str | Path) -> None:
    if system.main_planet_profile is None:
        raise RuntimeError("Rain-shadow diagnostic requested, but no Main Planet profile exists.")
    from PIL import Image, ImageDraw
    import numpy as np
    output_path = _prepare_output_path(output_path)
    profile = system.main_planet_profile
    terrain = profile.terrain
    climate = profile.climate
    out_w, out_h, stride = _diagnostic_downsample_dimensions(terrain.width, terrain.height)
    elev = np.asarray(terrain.elevation_m, dtype=np.float32)[::stride, ::stride]
    land = np.asarray(terrain.is_land, dtype=bool)[::stride, ::stride]
    precip = np.asarray(climate.annual_precip_mm, dtype=np.float32)[::stride, ::stride]
    # Dry leeward candidate: high local relief plus lower-than-nearby precip.
    gy, gx = np.gradient(elev)
    relief = np.sqrt(gx * gx + gy * gy)
    local_p = precip
    local_mean = (local_p + np.roll(local_p, 8, axis=1) + np.roll(local_p, -8, axis=1) + np.roll(local_p, 8, axis=0) + np.roll(local_p, -8, axis=0)) / 5.0
    shadow = np.clip((local_mean - local_p) / 650.0, 0.0, 1.0) * np.clip(relief / max(1.0, np.quantile(relief[land], 0.98) if land.any() else 1.0), 0.0, 1.0)
    windward = np.clip((local_p - local_mean) / 650.0, 0.0, 1.0) * np.clip(relief / max(1.0, np.quantile(relief[land], 0.98) if land.any() else 1.0), 0.0, 1.0)
    rgb = np.zeros((out_h, out_w, 3), dtype=np.uint8)
    rgb[:, :, :] = (58, 92, 120)
    base = land
    rgb[base] = (140, 142, 116)
    sh = base & (shadow > 0.08)
    wi = base & (windward > 0.08)
    rgb[sh, 0] = (190 + 55 * shadow[sh]).astype(np.uint8)
    rgb[sh, 1] = (155 - 70 * shadow[sh]).astype(np.uint8)
    rgb[sh, 2] = (80 - 35 * shadow[sh]).astype(np.uint8)
    rgb[wi, 0] = (75 - 20 * windward[wi]).astype(np.uint8)
    rgb[wi, 1] = (150 + 70 * windward[wi]).astype(np.uint8)
    rgb[wi, 2] = (120 + 90 * windward[wi]).astype(np.uint8)
    image = Image.fromarray(rgb, mode="RGB")
    draw = ImageDraw.Draw(image)
    draw.rectangle((8, 8, min(out_w - 8, 790), 58), fill=(255, 255, 255), outline=(30, 30, 30))
    draw.text((18, 18), "Rain-shadow diagnostic: green windward/wet relief, brown dry leeward relief", fill=(0, 0, 0))
    _save_image_fast(image, output_path)


def save_main_planet_terrain_provinces_view(system: StarSystem, output_path: str | Path) -> None:
    if system.main_planet_profile is None:
        raise RuntimeError("Terrain-province diagnostic requested, but no Main Planet profile exists.")
    from PIL import Image, ImageDraw
    import numpy as np
    output_path = _prepare_output_path(output_path)
    profile = system.main_planet_profile
    terrain = profile.terrain
    hydrology = profile.hydrology
    out_w, out_h, stride = _diagnostic_downsample_dimensions(terrain.width, terrain.height)
    elev = np.asarray(terrain.elevation_m, dtype=np.float32)[::stride, ::stride]
    land = np.asarray(terrain.is_land, dtype=bool)[::stride, ::stride]
    rivers = np.asarray(hydrology.river_intensity, dtype=np.uint8)[::stride, ::stride]
    gy, gx = np.gradient(elev)
    slope = np.sqrt(gx * gx + gy * gy)
    slope_q = np.quantile(slope[land], 0.88) if land.any() else 1.0
    rgb = np.zeros((out_h, out_w, 3), dtype=np.uint8)
    rgb[:, :, :] = (45, 86, 128)  # ocean
    low = land & (elev < 220)
    basin = land & (elev >= 220) & (elev < 650) & (slope < slope_q * 0.35)
    shield = land & (elev >= 650) & (elev < 1450) & (slope < slope_q * 0.55)
    plateau = land & (elev >= 1450) & (slope < slope_q * 0.70)
    rugged = land & (slope >= slope_q * 0.75) & (elev >= 600)
    mountain = land & ((elev >= 1850) | (slope >= slope_q * 1.15))
    alluvial = land & (rivers > 0) & (elev < 520)
    rgb[low] = (152, 176, 112)
    rgb[basin] = (188, 176, 124)
    rgb[shield] = (154, 143, 101)
    rgb[plateau] = (180, 132, 92)
    rgb[rugged] = (129, 116, 96)
    rgb[mountain] = (220, 220, 188)
    rgb[alluvial] = (98, 174, 122)
    image = Image.fromarray(rgb, mode="RGB")
    draw = ImageDraw.Draw(image)
    draw.rectangle((8, 8, min(out_w - 8, 900), 74), fill=(255, 255, 255), outline=(30, 30, 30))
    draw.text((18, 18), "Terrain provinces diagnostic", fill=(0, 0, 0))
    draw.text((18, 40), "ocean, coastal plain, basin/plain, shield/upland, plateau, rugged highland, mountain, alluvial river plain", fill=(0, 0, 0))
    _save_image_fast(image, output_path)


def save_main_planet_erosion_deposition_view(system: StarSystem, output_path: str | Path) -> None:
    if system.main_planet_profile is None:
        raise RuntimeError("Erosion/deposition diagnostic requested, but no Main Planet profile exists.")
    from PIL import Image, ImageDraw
    import numpy as np
    output_path = _prepare_output_path(output_path)
    profile = system.main_planet_profile
    terrain = profile.terrain
    hydrology = profile.hydrology
    out_w, out_h, stride = _diagnostic_downsample_dimensions(terrain.width, terrain.height)
    elev = np.asarray(terrain.elevation_m, dtype=np.float32)[::stride, ::stride]
    land = np.asarray(terrain.is_land, dtype=bool)[::stride, ::stride]
    rivers = np.asarray(hydrology.river_intensity, dtype=np.float32)[::stride, ::stride]
    flow = np.asarray(hydrology.flow_accumulation, dtype=np.float32)[::stride, ::stride]
    gy, gx = np.gradient(elev)
    slope = np.sqrt(gx * gx + gy * gy)
    slope_norm = np.clip(slope / max(1.0, np.quantile(slope[land], 0.985) if land.any() else 1.0), 0.0, 1.0)
    flow_norm = np.clip(np.log1p(flow) / max(1e-6, np.log1p(max(1.0, float(np.nanmax(flow))))), 0.0, 1.0)
    erosion = land & (slope_norm > 0.22) & (flow_norm > 0.28)
    deposition = land & (flow_norm > 0.36) & (slope_norm < 0.35) & (elev < 700)
    deltas = land & (rivers > 170) & (elev < 45)
    rgb = np.zeros((out_h, out_w, 3), dtype=np.uint8)
    rgb[:, :, :] = (48, 83, 115)
    rgb[land] = (132, 132, 120)
    rgb[erosion] = (190, 72, 48)
    rgb[deposition] = (86, 164, 86)
    rgb[deltas] = (55, 205, 155)
    image = Image.fromarray(rgb, mode="RGB")
    draw = ImageDraw.Draw(image)
    draw.rectangle((8, 8, min(out_w - 8, 840), 74), fill=(255, 255, 255), outline=(30, 30, 30))
    draw.text((18, 18), "Erosion/deposition diagnostic", fill=(0, 0, 0))
    draw.text((18, 40), "red: likely incision/erosion | green: floodplain/deposition | teal: low river-mouth/delta land", fill=(0, 0, 0))
    _save_image_fast(image, output_path)



def save_main_planet_land_exactly_1m_view(system: StarSystem, output_path: str | Path) -> None:
    """Diagnostic map for land cells whose final integer elevation is exactly 1 m.

    These cells are useful because large numbers of exact-1m land cells usually
    mean a shoreline clamp, mask conversion, or sea-level cleanup is creating an
    artificial coastal shelf/halo instead of true low coastal terrain.
    """
    if system.main_planet_profile is None:
        raise RuntimeError("Exact-1m land diagnostic requested, but no Main Planet profile exists.")
    import numpy as np
    profile = system.main_planet_profile
    terrain = profile.terrain
    out_w, out_h, stride = _diagnostic_downsample_dimensions(terrain.width, terrain.height)
    land = np.asarray(terrain.is_land, dtype=bool)
    elev = np.asarray(terrain.elevation_m, dtype=np.int32)
    exact = land & (elev == 1)
    land_ds = land[::stride, ::stride]
    elev_ds = elev[::stride, ::stride]
    exact_ds = land_ds & (elev_ds == 1)
    rgb = np.zeros((out_h, out_w, 3), dtype=np.uint8)
    rgb[:, :, :] = (36, 72, 118)
    rgb[land_ds] = (118, 138, 96)
    rgb[exact_ds] = (245, 40, 190)
    total_exact = int(exact.sum())
    land_total = max(1, int(land.sum()))
    stats = {
        "land_exactly_1m_cells": total_exact,
        "land_exactly_1m_share_of_land": total_exact / land_total,
        "source_stride": int(stride),
    }
    if hasattr(terrain, "terrain_diagnostics"):
        diag = getattr(terrain, "terrain_diagnostics", {}) or {}
        diag["land_exactly_1m"] = dict(stats)
        terrain.terrain_diagnostics = diag
    _save_clean_rgb_map(
        rgb,
        output_path,
        title="Exact 1 m land diagnostic",
        description="Highlights land cells whose final integer elevation is exactly 1 m. Large clusters usually indicate sea-level clamp artifacts, artificial coastal shelves, or mask conversion problems.",
        items=[("ocean/background", (36, 72, 118)), ("land", (118, 138, 96)), ("land exactly 1 m", (245, 40, 190))],
        stats=stats,
        scale=_world_scale_meta(profile, out_w, out_h, kind="diagnostic_downsample", stride=stride),
    )

def save_main_planet_delta_mouths_view(system: StarSystem, output_path: str | Path) -> None:
    """Diagnostic map showing likely river mouths, estuaries, and delta deposition."""
    if system.main_planet_profile is None:
        raise RuntimeError("Delta/mouth diagnostic requested, but no Main Planet profile exists.")
    from PIL import Image, ImageDraw
    import numpy as np
    output_path = _prepare_output_path(output_path)
    profile = system.main_planet_profile
    terrain = profile.terrain
    hydrology = profile.hydrology
    out_w, out_h, stride = _diagnostic_downsample_dimensions(terrain.width, terrain.height)
    if getattr(terrain, "terrain_coast_style_class", None) is not None:
        style = np.asarray(terrain.terrain_coast_style_class, dtype=np.uint8)[::stride, ::stride]
        colors = {
            0: (34, 78, 128),
            1: (112, 184, 128),
            2: (205, 84, 62),
            3: (78, 130, 212),
            4: (232, 108, 54),
            5: (88, 205, 170),
            6: (235, 220, 112),
        }
        rgb = np.zeros((style.shape[0], style.shape[1], 3), dtype=np.uint8)
        for code, color in colors.items():
            rgb[style == code] = color
        land_hint = np.asarray(terrain.is_land, dtype=bool)[::stride, ::stride]
        rgb[(style == 0) & land_hint] = (132, 146, 105)
        _save_clean_rgb_map(rgb, output_path, title="Coastline / margin-type diagnostic", description="Coast style classes: passive shelf/coastal plains, rugged active/fjorded margins, rifted gulfs, volcanic arc coasts, true deltaic plains with sediment support, and mixed irregular coasts.", items=[("ocean/background", (34, 78, 128)), ("passive smooth coastal plain", (112, 184, 128)), ("rugged active/fjorded margin", (205, 84, 62)), ("rifted gulf margin", (78, 130, 212)), ("volcanic arc coast", (232, 108, 54)), ("true deltaic plain / sediment coast", (88, 205, 170)), ("mixed irregular coast", (235, 220, 112))], scale=_world_scale_meta(profile, out_w, out_h, kind="diagnostic_downsample", stride=stride))
        return
    land = np.asarray(terrain.is_land, dtype=bool)[::stride, ::stride]
    elev = np.asarray(terrain.elevation_m, dtype=np.int32)[::stride, ::stride]
    rivers = np.asarray(hydrology.river_intensity, dtype=np.uint8)[::stride, ::stride]

    rgb = np.zeros((out_h, out_w, 3), dtype=np.uint8)
    rgb[:, :, :] = (44, 82, 130)
    rgb[land] = (128, 151, 105)
    lowland = land & (elev < 120)
    rgb[lowland] = (170, 184, 118)

    river_mask = land & (rivers > 0)
    rgb[river_mask] = (40, 115, 210)

    mouth = np.zeros((out_h, out_w), dtype=bool)
    estuary = np.zeros((out_h, out_w), dtype=bool)
    delta = np.zeros((out_h, out_w), dtype=bool)
    for r in range(out_h):
        for c in range(out_w):
            if not land[r, c] or rivers[r, c] < 120:
                continue
            touches_ocean = False
            shallow = -99999
            for dr in (-1, 0, 1):
                rr = r + dr
                if rr < 0 or rr >= out_h:
                    continue
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    cc = (c + dc) % out_w
                    if not land[rr, cc]:
                        touches_ocean = True
                        shallow = max(shallow, int(elev[rr, cc]))
            if not touches_ocean:
                continue
            mouth[r, c] = True
            if shallow > -180 and rivers[r, c] >= 170:
                delta[r, c] = True
            else:
                estuary[r, c] = True

    # Dilate markers slightly so they remain visible after downsampling.
    for mask, color in ((estuary, (235, 225, 90)), (delta, (40, 220, 160))):
        ys, xs = np.where(mask)
        for r, c in zip(ys, xs):
            radius = 1 if out_w < 1800 else 2
            for dr in range(-radius, radius + 1):
                rr = r + dr
                if rr < 0 or rr >= out_h:
                    continue
                for dc in range(-radius, radius + 1):
                    cc = (c + dc) % out_w
                    rgb[rr, cc] = color

    image = Image.fromarray(rgb, mode="RGB")
    draw = ImageDraw.Draw(image)
    draw.rectangle((8, 8, min(out_w - 8, 900), 78), fill=(255, 255, 255), outline=(30, 30, 30))
    draw.text((18, 18), "River mouth and delta diagnostic", fill=(0, 0, 0))
    draw.text((18, 40), "blue: rivers | yellow: estuary mouths | teal: likely delta/deposition mouths | pale green: low coastal plains", fill=(0, 0, 0))
    _save_image_fast(image, output_path)

def save_main_planet_tectonic_plates_view(system: StarSystem, output_path: str | Path) -> None:
    """Diagnostic map showing low-resolution procedural plate IDs."""
    if system.main_planet_profile is None:
        raise RuntimeError("Plate diagnostic requested, but no Main Planet profile exists.")
    from PIL import Image, ImageDraw
    import numpy as np
    output_path = _prepare_output_path(output_path)
    terrain = system.main_planet_profile.terrain
    if terrain.tectonic_plate_id is None:
        raise RuntimeError("Terrain has no tectonic plate diagnostic grid.")
    plates = np.asarray(terrain.tectonic_plate_id, dtype=np.int32)
    h, w = plates.shape
    rng = np.random.default_rng(14321)
    max_plate = max(int(plates.max()), 0)
    colors = rng.integers(45, 235, size=(max_plate + 1, 3), dtype=np.uint8)
    rgb = colors[np.clip(plates, 0, max_plate)]
    image = Image.fromarray(rgb, mode="RGB").resize((max(1024, w * 2), max(512, h * 2)), Image.Resampling.NEAREST)
    draw = ImageDraw.Draw(image)
    draw.rectangle((8, 8, min(image.width - 8, 620), 58), fill=(255, 255, 255), outline=(30, 30, 30))
    draw.text((18, 18), "Procedural tectonic plate diagnostic", fill=(0, 0, 0))
    draw.text((18, 38), "colored cells show generated plate regions; terrain uses boundary interactions", fill=(0, 0, 0))
    _save_image_fast(image, output_path)


def _terrain_is_plate_history_v3(profile) -> bool:
    terrain = getattr(profile, "terrain", None)
    source = str(getattr(terrain, "source", "") or "").lower()
    diag = getattr(terrain, "terrain_diagnostics", None) or {}
    mode = str(diag.get("terrain_mode", "") or "").lower() if isinstance(diag, dict) else ""
    return ("plate_history_v3" in source) or ("plate_history_v4" in source) or mode in {"plate_history_v3", "plate_history_v4"}


def _field01_from_optional(value, shape):
    import numpy as np
    if value is None:
        return np.zeros(shape, dtype=np.float32)
    arr = np.asarray(value, dtype=np.float32)
    if arr.shape != shape:
        arr = np.resize(arr, shape).astype(np.float32)
    if arr.size and (float(np.nanmax(arr)) > 1.5 or float(np.nanmin(arr)) < -0.05):
        arr = arr / 1000.0
    return np.clip(np.nan_to_num(arr, nan=0.0), 0.0, 1.0).astype(np.float32, copy=False)


def _label_xwrap_bool(mask):
    import numpy as np
    from scipy import ndimage
    feature = np.asarray(mask, dtype=bool)
    labels, n = ndimage.label(feature, structure=np.ones((3, 3), dtype=np.uint8))
    h, w = feature.shape
    if n <= 1 or w <= 1:
        return labels.astype(np.int32, copy=False), int(n)
    parent = list(range(n + 1))
    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    def union(a: int, b: int) -> None:
        if a == 0 or b == 0:
            return
        ra, rb = find(int(a)), find(int(b))
        if ra != rb:
            parent[rb] = ra
    for yy in range(h):
        union(int(labels[yy, 0]), int(labels[yy, w - 1]))
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


def _save_v3_heat_field(system: StarSystem, output_path: str | Path, attr_name: str, *, title: str, description: str, hot_color=(235, 92, 64), cool_color=(35, 68, 102), label="high") -> None:
    if system.main_planet_profile is None:
        raise RuntimeError(f"{title} requested, but no Main Planet profile exists.")
    import numpy as np
    profile = system.main_planet_profile
    terrain = profile.terrain
    out_w, out_h, stride = _diagnostic_downsample_dimensions(terrain.width, terrain.height)
    land = np.asarray(terrain.is_land, dtype=bool)[::stride, ::stride]
    shape = land.shape
    field = _field01_from_optional(getattr(terrain, attr_name, None), (terrain.height, terrain.width))[::stride, ::stride]
    rgb = np.zeros((out_h, out_w, 3), dtype=np.uint8)
    base = np.array(cool_color, dtype=np.float32)
    hot = np.array(hot_color, dtype=np.float32)
    for ch in range(3):
        rgb[..., ch] = (base[ch] * (1.0 - field) + hot[ch] * field).astype(np.uint8)
    # Keep land/water readable under low signal.
    low = field < 0.06
    rgb[low & land] = (118, 132, 100)
    rgb[low & ~land] = (31, 68, 112)
    _save_clean_rgb_map(
        rgb,
        output_path,
        title=title,
        description=description,
        items=[("low/background ocean", (31, 68, 112)), ("low/background land", (118, 132, 100)), (label, hot_color)],
        stats={"mean_signal": round(float(np.mean(field)), 5), "max_signal": round(float(np.max(field)), 5)},
        scale=_world_scale_meta(profile, out_w, out_h, kind="diagnostic_downsample", stride=stride),
    )


def save_main_planet_plate_boundaries_view(system: StarSystem, output_path: str | Path) -> None:
    if system.main_planet_profile is None:
        raise RuntimeError("Plate-boundary diagnostic requested, but no Main Planet profile exists.")
    from PIL import Image
    import numpy as np
    profile = system.main_planet_profile
    terrain = profile.terrain
    if terrain.tectonic_boundary_class is None and terrain.tectonic_plate_id is None:
        raise RuntimeError("Terrain has no tectonic boundary diagnostic grid.")
    is_v3 = _terrain_is_plate_history_v3(profile)
    if is_v3:
        land = np.asarray(terrain.is_land, dtype=bool)
        plates = np.asarray(terrain.tectonic_plate_id, dtype=np.int32) if terrain.tectonic_plate_id is not None else None
        if plates is not None and plates.shape == land.shape:
            final_mask = np.zeros_like(land, dtype=bool)
            final_mask |= plates != np.roll(plates, 1, axis=1)
            final_mask |= plates != np.roll(plates, -1, axis=1)
            final_mask[:-1, :] |= plates[:-1, :] != plates[1:, :]
            final_mask[1:, :] |= plates[1:, :] != plates[:-1, :]
            conv = _field01_from_optional(getattr(terrain, "plate_tectonic_convergence_x1000", None), land.shape)
            div = _field01_from_optional(getattr(terrain, "plate_tectonic_divergence_x1000", None), land.shape)
            trans = _field01_from_optional(getattr(terrain, "plate_tectonic_transform_x1000", None), land.shape)
            trench = _field01_from_optional(getattr(terrain, "plate_tectonic_trench_x1000", None), land.shape)
            active = _field01_from_optional(getattr(terrain, "plate_tectonic_active_margin_x1000", None), land.shape)
            volcanism = _field01_from_optional(getattr(terrain, "plate_tectonic_volcanic_arc_x1000", None), land.shape)
            dominance = np.argmax(np.stack([conv, div, trans, trench + active, volcanism], axis=0), axis=0) + 1
            b = np.zeros_like(plates, dtype=np.uint8)
            b[final_mask] = dominance[final_mask].astype(np.uint8)
        else:
            b = np.asarray(terrain.tectonic_boundary_class, dtype=np.uint8)
            if land.shape != b.shape:
                land = np.resize(land, b.shape).astype(bool)
        rgb = np.zeros((b.shape[0], b.shape[1], 3), dtype=np.uint8)
        rgb[~land] = (32, 72, 116)
        rgb[land] = (128, 140, 102)
        colors = {
            1: (205, 62, 50),   # compression / collision
            2: (70, 126, 230),  # extension / rift
            3: (148, 80, 178),  # transform / shear
            4: (105, 64, 150),  # active subduction / trench
            5: (236, 128, 58),  # volcanic active boundary
        }
        for code, color in colors.items():
            rgb[b == code] = color
        image = Image.fromarray(rgb, mode="RGB")
        output_path = _prepare_output_path(output_path)
        _save_image_fast(image, output_path)
        _write_map_legend_sidecar(
            output_path,
            title="Final v3 plate boundaries",
            description="Final active plate-boundary positions only, derived from the final plate-ID raster when available. Historical boundary crossings are separated into boundary-history density, orogeny-history, and suture-history maps.",
            items=[("ocean/background", (32, 72, 116)), ("land/background", (128, 140, 102)), ("convergent/collision", colors[1]), ("divergent/rift", colors[2]), ("transform/shear", colors[3]), ("subduction/trench", colors[4]), ("volcanic active boundary", colors[5])],
            stats={"source_width": b.shape[1], "source_height": b.shape[0], "active_boundary_cells": int(np.count_nonzero(b))},
            scale=_world_scale_meta(profile, image.width, image.height, kind="diagnostic_downsample"),
        )
        return
    b = np.asarray(terrain.tectonic_boundary_class, dtype=np.uint8)
    rgb = np.zeros((b.shape[0], b.shape[1], 3), dtype=np.uint8)
    rgb[:, :, :] = (225, 225, 215)
    rgb[b == 1] = (165, 55, 45)
    rgb[b == 2] = (55, 105, 190)
    rgb[b == 3] = (128, 70, 165)
    image = Image.fromarray(rgb, mode="RGB").resize((max(1024, b.shape[1] * 2), max(512, b.shape[0] * 2)), Image.Resampling.NEAREST)
    output_path = _prepare_output_path(output_path)
    _save_image_fast(image, output_path)
    _write_map_legend_sidecar(output_path, title="Plate boundary diagnostic", description="Procedural convergent, divergent, and transform boundary classes.", items=[("intraplate", (225, 225, 215)), ("convergent/uplift", (165, 55, 45)), ("divergent/rift", (55, 105, 190)), ("transform/shear", (128, 70, 165))], stats={"source_width": b.shape[1], "source_height": b.shape[0]}, scale=_world_scale_meta(profile, image.width, image.height, kind="diagnostic_downsample"))

def save_main_planet_final_plate_boundaries_view(system: StarSystem, output_path: str | Path) -> None:
    save_main_planet_plate_boundaries_view(system, output_path)


def save_main_planet_boundary_history_density_view(system: StarSystem, output_path: str | Path) -> None:
    _save_v3_heat_field(
        system,
        output_path,
        "tectonic_boundary_strength_x1000",
        title="Boundary history density",
        description="Accumulated v3 boundary/deformation signal. Unlike final plate boundaries, this intentionally shows where boundary-related deformation passed through over time.",
        hot_color=(230, 78, 62),
        label="high historical boundary density",
    )


def save_main_planet_orogeny_history_view(system: StarSystem, output_path: str | Path) -> None:
    _save_v3_heat_field(
        system,
        output_path,
        "plate_tectonic_orogeny_strength_x1000",
        title="Orogeny history",
        description="Accumulated mountain-building/uplift signal derived from v3 continuous compression, uplift, volcanic, and plateau fields.",
        hot_color=(238, 92, 64),
        label="strong orogeny/uplift history",
    )


def save_main_planet_suture_history_view(system: StarSystem, output_path: str | Path) -> None:
    _save_v3_heat_field(
        system,
        output_path,
        "plate_tectonic_accreted_terrane_x1000",
        title="Suture history",
        description="Old/inactive collision and accretion scars. This is separated from final active boundaries so ancient sutures do not clutter the main boundary map.",
        hot_color=(184, 104, 208),
        cool_color=(36, 58, 82),
        label="old suture/accretion signal",
    )


def save_main_planet_submerged_continental_crust_view(system: StarSystem, output_path: str | Path) -> None:
    _save_v3_heat_field(
        system,
        output_path,
        "terrain_submerged_continental_crust_x1000",
        title="Submerged continental crust support",
        description="v3 shelf diagnostic: where continental affinity continues offshore. This should be broad around true passive continental margins, weak around unsupported volcanic islands, and suppressed at active trenches.",
        hot_color=(106, 205, 178),
        cool_color=(28, 62, 102),
        label="strong submerged continental support",
    )


def save_main_planet_continental_shelf_support_view(system: StarSystem, output_path: str | Path) -> None:
    _save_v3_heat_field(
        system,
        output_path,
        "terrain_continental_shelf_support_x1000",
        title="Continental shelf support",
        description="v3 shelf diagnostic: combined support for shallow continental shelf water from submerged continental affinity, passive-margin maturity, sediment supply, low slope, and active-margin suppression.",
        hot_color=(86, 218, 188),
        cool_color=(28, 64, 112),
        label="high shelf support",
    )


def save_main_planet_shelf_depth_target_view(system: StarSystem, output_path: str | Path) -> None:
    _save_v3_heat_field(
        system,
        output_path,
        "terrain_shelf_depth_target_x1000",
        title="Shelf depth target",
        description="v3 shelf diagnostic: normalized target depth for shelf/slope conditioning. Brighter values indicate deeper shelf-edge/slope targets; compare with shelf support to identify where support exists but bathymetry is still too deep.",
        hot_color=(236, 196, 92),
        cool_color=(30, 72, 118),
        label="deeper shelf/slope target",
    )


def save_main_planet_shelf_zones_view(system: StarSystem, output_path: str | Path) -> None:
    if system.main_planet_profile is None:
        raise RuntimeError("Shelf-zone diagnostic requested, but no Main Planet profile exists.")
    import numpy as np
    profile = system.main_planet_profile
    terrain = profile.terrain
    if getattr(terrain, "terrain_shelf_zone_class", None) is None:
        raise RuntimeError("Terrain has no v3 shelf-zone class grid.")
    out_w, out_h, stride = _diagnostic_downsample_dimensions(terrain.width, terrain.height)
    zone = np.asarray(terrain.terrain_shelf_zone_class, dtype=np.uint8)[::stride, ::stride]
    palette = {
        0: (28, 64, 106),
        1: (92, 204, 188),   # shallow shelf sea
        2: (72, 150, 204),   # shelf edge / upper slope
        3: (58, 104, 166),   # continental rise
        4: (22, 52, 104),    # abyssal/open ocean
        5: (82, 54, 126),    # trench/active suppression
        6: (128, 144, 104),  # land
    }
    rgb = np.zeros((zone.shape[0], zone.shape[1], 3), dtype=np.uint8)
    for code, color in palette.items():
        rgb[zone == code] = color
    _save_clean_rgb_map(
        rgb,
        output_path,
        title="v3 shelf / slope / rise zones",
        description="Final v3 ocean-margin class derived from final land/ocean state, shelf support, depth, and active-margin suppression. This separates true shallow continental shelf seas from slope/rise and abyssal ocean.",
        items=[
            ("background/unknown water", palette[0]),
            ("shallow continental shelf sea", palette[1]),
            ("shelf edge / upper continental slope", palette[2]),
            ("continental rise", palette[3]),
            ("abyssal/open ocean", palette[4]),
            ("active trench suppression", palette[5]),
            ("land", palette[6]),
        ],
        stats={"source_width": int(zone.shape[1]), "source_height": int(zone.shape[0]), "shallow_shelf_cells": int(np.count_nonzero(zone == 1)), "slope_rise_cells": int(np.count_nonzero((zone == 2) | (zone == 3)))},
        scale=_world_scale_meta(profile, out_w, out_h, kind="diagnostic_downsample", stride=stride),
    )


def save_main_planet_lake_depth_limit_view(system: StarSystem, output_path: str | Path) -> None:
    _save_v3_heat_field(
        system,
        output_path,
        "terrain_lake_depth_limit_x1000",
        title="v3 enclosed-water depth limit",
        description="Update 27D diagnostic: local allowed lake/inland-sea depth after basin-size, rift/subsidence support, sediment fill, distance from basin edge, and coastal-lagoon cleanup. Brighter centers are allowed to stay deeper; dark/fill areas were shallow or converted to coastal lowland.",
        hot_color=(236, 196, 92),
        cool_color=(28, 58, 104),
        label="deeper allowed lake/inland-sea floor",
    )


def save_main_planet_ripple_artifact_risk_view(system: StarSystem, output_path: str | Path) -> None:
    _save_v3_heat_field(
        system,
        output_path,
        "terrain_ripple_artifact_risk_x1000",
        title="v3 ripple artifact risk",
        description="Update 27D diagnostic: old deep-ocean cells where plate-history boundary accumulation and texture are most likely to print as visible ripples. The terrain pass damps this field while preserving ridges, trenches, seamounts, and supported shelves.",
        hot_color=(232, 94, 74),
        cool_color=(30, 62, 104),
        label="higher ripple artifact risk",
    )


def save_main_planet_final_plate_components_view(system: StarSystem, output_path: str | Path) -> None:
    if system.main_planet_profile is None:
        raise RuntimeError("Final plate component diagnostic requested, but no Main Planet profile exists.")
    import numpy as np
    profile = system.main_planet_profile
    terrain = profile.terrain
    if getattr(terrain, "terrain_final_plate_component_class", None) is None:
        raise RuntimeError("Terrain has no final plate component cleanup grid.")
    out_w, out_h, stride = _diagnostic_downsample_dimensions(terrain.width, terrain.height)
    cls = np.asarray(terrain.terrain_final_plate_component_class, dtype=np.uint8)[::stride, ::stride]
    if getattr(terrain, "tectonic_plate_id", None) is not None:
        plate = np.asarray(terrain.tectonic_plate_id, dtype=np.int32)[::stride, ::stride]
    else:
        plate = np.zeros_like(cls, dtype=np.int32)
    base = np.zeros((cls.shape[0], cls.shape[1], 3), dtype=np.uint8)
    # Muted deterministic plate-color background, with cleanup classes overlaid.
    base[..., 0] = ((plate * 37 + 70) % 120 + 45).astype(np.uint8)
    base[..., 1] = ((plate * 53 + 95) % 120 + 45).astype(np.uint8)
    base[..., 2] = ((plate * 71 + 125) % 120 + 45).astype(np.uint8)
    base[cls == 1] = np.array([236, 184, 72], dtype=np.uint8)   # reassigned fragment
    base[cls == 2] = np.array([226, 86, 74], dtype=np.uint8)    # promoted microplate
    _save_clean_rgb_map(
        base,
        output_path,
        title="v3 final plate component cleanup",
        description="Final diagnostic plate IDs after x-wrapped contiguity cleanup. Small disconnected fragments are reassigned to neighboring plates; large fragments are promoted to microplates so final plate diagnostics do not show physically impossible non-contiguous plates.",
        items=[
            ("ordinary contiguous final plate", (92, 128, 146)),
            ("small disconnected fragment reassigned", (236, 184, 72)),
            ("large disconnected fragment promoted to microplate", (226, 86, 74)),
        ],
        stats={
            "source_width": int(cls.shape[1]),
            "source_height": int(cls.shape[0]),
            "reassigned_cells": int(np.count_nonzero(cls == 1)),
            "promoted_microplate_cells": int(np.count_nonzero(cls == 2)),
        },
        scale=_world_scale_meta(profile, out_w, out_h, kind="diagnostic_downsample", stride=stride),
    )



def save_main_planet_v4_boundary_deformation_view(system: StarSystem, output_path: str | Path) -> None:
    _save_v3_heat_field(
        system,
        output_path,
        "terrain_v4_boundary_deformation_x1000",
        title="v4 boundary deformation field",
        description="plate_history_v4 diagnostic: smooth displacement/warping applied to the final diagnostic plate positions to reduce neat Voronoi-looking boundaries while keeping v3 stable.",
        hot_color=(235, 118, 72),
        cool_color=(30, 64, 104),
        label="high boundary deformation",
    )


def save_main_planet_v4_volcanic_island_support_view(system: StarSystem, output_path: str | Path) -> None:
    _save_v3_heat_field(
        system,
        output_path,
        "terrain_v4_volcanic_island_support_x1000",
        title="v4 volcanic island support",
        description="plate_history_v4 diagnostic: oceanic cells where volcanism, active boundaries, seamount/ridge support, and low continental-shelf support agree. This is where v4 may raise volcanic island chains or archipelagos.",
        hot_color=(245, 132, 58),
        cool_color=(24, 58, 104),
        label="high volcanic island support",
    )


def save_main_planet_v4_rift_cut_support_view(system: StarSystem, output_path: str | Path) -> None:
    _save_v3_heat_field(
        system,
        output_path,
        "terrain_v4_rift_cut_support_x1000",
        title="v4 rift-cut support",
        description="plate_history_v4 diagnostic: continuous support for rift cuts, gulf/seaway openings, rift-valley basins, and lake-prone extensional corridors. High values lower terrain only where divergence/rifting, transform shear, deformed boundaries, and continental/transitional crust agree.",
        hot_color=(232, 92, 84),
        cool_color=(26, 60, 104),
        label="high rift-cut support",
    )


def save_main_planet_v4_mountain_branch_support_view(system: StarSystem, output_path: str | Path) -> None:
    _save_v3_heat_field(
        system,
        output_path,
        "terrain_v4_mountain_branch_support_x1000",
        title="v4 mountain-branch support",
        description="plate_history_v4 diagnostic: oblique mountain-branch support derived from the deformed final plate fabric, convergence, volcanism, boundary deformation, and continental affinity. This highlights where v4 spreads uplift into branching orogens instead of only a neat boundary ribbon.",
        hot_color=(230, 166, 74),
        cool_color=(54, 72, 82),
        label="high mountain-branch support",
    )


def save_main_planet_v4_island_chain_view(system: StarSystem, output_path: str | Path) -> None:
    if system.main_planet_profile is None:
        raise RuntimeError("v4 island-chain diagnostic requested, but no Main Planet profile exists.")
    import numpy as np
    profile = system.main_planet_profile
    terrain = profile.terrain
    if getattr(terrain, "terrain_v4_island_chain_class", None) is None:
        raise RuntimeError("Terrain has no v4 island-chain class grid.")
    out_w, out_h, stride = _diagnostic_downsample_dimensions(terrain.width, terrain.height)
    cls = np.asarray(terrain.terrain_v4_island_chain_class, dtype=np.uint8)[::stride, ::stride]
    palette = {
        0: (24, 54, 98),
        1: (132, 150, 106),
        2: (230, 86, 70),
        3: (245, 150, 64),
        4: (214, 110, 190),
        5: (112, 206, 196),
        6: (178, 166, 94),
        7: (252, 222, 96),
    }
    rgb = np.zeros((cls.shape[0], cls.shape[1], 3), dtype=np.uint8)
    for code, color in palette.items():
        rgb[cls == code] = color
    _save_clean_rgb_map(
        rgb,
        output_path,
        title="v4 volcanic island-chain classes",
        description="plate_history_v4 diagnostic: classifies where supported island-chain terrain came from: volcanic arc, ridge/seamount chain, hotspot/oceanic chain, rift-margin chain, existing island, or newly raised v4 island. This is diagnostic-only; v3 is unchanged.",
        items=[
            ("open ocean / no island-chain support", palette[0]),
            ("pre-existing continent/large land", palette[1]),
            ("volcanic arc chain support", palette[2]),
            ("ridge or seamount chain support", palette[3]),
            ("hotspot/oceanic volcanic chain support", palette[4]),
            ("rift-margin/narrow-sea island support", palette[5]),
            ("pre-existing island retained", palette[6]),
            ("new v4 volcanic island", palette[7]),
        ],
        stats={
            "source_width": int(cls.shape[1]),
            "source_height": int(cls.shape[0]),
            "support_cells": int(np.count_nonzero((cls >= 2) & (cls <= 5))),
            "new_v4_island_cells": int(np.count_nonzero(cls == 7)),
        },
        scale=_world_scale_meta(profile, out_w, out_h, kind="diagnostic_downsample", stride=stride),
    )


def save_main_planet_v4_plate_topology_view(system: StarSystem, output_path: str | Path) -> None:
    if system.main_planet_profile is None:
        raise RuntimeError("v4 topology diagnostic requested, but no Main Planet profile exists.")
    import numpy as np
    profile = system.main_planet_profile
    terrain = profile.terrain
    if getattr(terrain, "terrain_v4_topology_class", None) is None:
        raise RuntimeError("Terrain has no v4 topology class grid.")
    out_w, out_h, stride = _diagnostic_downsample_dimensions(terrain.width, terrain.height)
    cls = np.asarray(terrain.terrain_v4_topology_class, dtype=np.uint8)[::stride, ::stride]
    if getattr(terrain, "tectonic_plate_id", None) is not None:
        plate = np.asarray(terrain.tectonic_plate_id, dtype=np.int32)[::stride, ::stride]
    else:
        plate = np.zeros_like(cls, dtype=np.int32)
    rgb = np.zeros((cls.shape[0], cls.shape[1], 3), dtype=np.uint8)
    rgb[..., 0] = ((plate * 37 + 70) % 120 + 45).astype(np.uint8)
    rgb[..., 1] = ((plate * 53 + 95) % 120 + 45).astype(np.uint8)
    rgb[..., 2] = ((plate * 71 + 125) % 120 + 45).astype(np.uint8)
    rgb[cls == 1] = np.array([238, 192, 74], dtype=np.uint8)
    rgb[cls == 2] = np.array([225, 82, 72], dtype=np.uint8)
    rgb[cls == 3] = np.array([112, 210, 196], dtype=np.uint8)
    rgb[cls == 4] = np.array([190, 92, 220], dtype=np.uint8)
    rgb[cls == 5] = np.array([232, 94, 84], dtype=np.uint8)
    rgb[cls == 6] = np.array([252, 150, 76], dtype=np.uint8)
    _save_clean_rgb_map(
        rgb,
        output_path,
        title="v4 experimental plate topology",
        description="plate_history_v4 diagnostic: deformed/cohered final plates with highlighted sliver candidates, promoted microplates, repaired disconnected fragments, and rift-cut corridors. v3 remains unchanged.",
        items=[
            ("ordinary deformed plate", (92, 128, 146)),
            ("sliver-plate candidate", (238, 192, 74)),
            ("promoted microplate candidate", (225, 82, 72)),
            ("reassigned disconnected fragment", (112, 210, 196)),
            ("promoted disconnected fragment", (190, 92, 220)),
            ("rift-cut corridor", (232, 94, 84)),
            ("native rift/sliver plate", (252, 150, 76)),
        ],
        stats={"sliver_cells": int(np.count_nonzero(cls == 1)), "microplate_cells": int(np.count_nonzero(cls == 2)), "reassigned_fragment_cells": int(np.count_nonzero(cls == 3)), "promoted_fragment_cells": int(np.count_nonzero(cls == 4)), "rift_cut_cells": int(np.count_nonzero(cls == 5)), "native_rift_sliver_cells": int(np.count_nonzero(cls == 6))},
        scale=_world_scale_meta(profile, out_w, out_h, kind="diagnostic_downsample", stride=stride),
    )



def save_main_planet_v4_boundary_network_view(system: StarSystem, output_path: str | Path) -> None:
    if system.main_planet_profile is None:
        raise RuntimeError("v4 boundary-network diagnostic requested, but no Main Planet profile exists.")
    import numpy as np
    profile = system.main_planet_profile
    terrain = profile.terrain
    if getattr(terrain, "terrain_v4_boundary_network_class", None) is None:
        raise RuntimeError("Terrain has no v4 boundary-network class grid.")
    out_w, out_h, stride = _diagnostic_downsample_dimensions(terrain.width, terrain.height)
    cls = np.asarray(terrain.terrain_v4_boundary_network_class, dtype=np.uint8)[::stride, ::stride]
    palette = {
        0: (24, 54, 92),
        1: (224, 174, 80),   # convergent/orogenic
        2: (92, 206, 198),   # divergent/rift
        3: (168, 124, 216),  # transform/shear
        4: (72, 78, 142),    # trench/subduction
        5: (232, 92, 68),    # volcanic/arc
        6: (250, 236, 116),  # complex/triple junction
        7: (232, 70, 150),   # micro/sliver boundary
        8: (74, 176, 226),   # rift corridor away from final boundary
        9: (246, 204, 84),   # new v4 island boundary/support
    }
    rgb = np.zeros((cls.shape[0], cls.shape[1], 3), dtype=np.uint8)
    for code, color in palette.items():
        rgb[cls == code] = color
    _save_clean_rgb_map(
        rgb,
        output_path,
        title="v4 boundary network classes",
        description="Update 31 diagnostic for plate_history_v4: classifies the final deformed boundary network into convergent, divergent/rift, transform, trench, volcanic/arc, complex junction, micro/sliver, rift-corridor, and new-island contexts. This is a readable topology summary, not a separate terrain rule set.",
        items=[
            ("background/no active v4 network", palette[0]),
            ("convergent/orogenic boundary", palette[1]),
            ("divergent/rift boundary", palette[2]),
            ("transform/shear boundary", palette[3]),
            ("trench/subduction boundary", palette[4]),
            ("volcanic/arc boundary", palette[5]),
            ("complex or triple-junction cell", palette[6]),
            ("microplate/sliver boundary", palette[7]),
            ("rift-cut corridor away from final boundary", palette[8]),
            ("new v4 volcanic-island context", palette[9]),
        ],
        stats={
            "source_width": int(cls.shape[1]),
            "source_height": int(cls.shape[0]),
            "network_cells": int(np.count_nonzero(cls > 0)),
            "complex_junction_cells": int(np.count_nonzero(cls == 6)),
            "micro_sliver_cells": int(np.count_nonzero(cls == 7)),
            "rift_corridor_cells": int(np.count_nonzero(cls == 8)),
        },
        scale=_world_scale_meta(profile, out_w, out_h, kind="diagnostic_downsample", stride=stride),
    )


def save_main_planet_v4_orogen_network_view(system: StarSystem, output_path: str | Path) -> None:
    if system.main_planet_profile is None:
        raise RuntimeError("v4 orogen-network diagnostic requested, but no Main Planet profile exists.")
    import numpy as np
    profile = system.main_planet_profile
    terrain = profile.terrain
    if getattr(terrain, "terrain_v4_orogen_network_class", None) is None:
        raise RuntimeError("Terrain has no v4 orogen-network class grid.")
    out_w, out_h, stride = _diagnostic_downsample_dimensions(terrain.width, terrain.height)
    cls = np.asarray(terrain.terrain_v4_orogen_network_class, dtype=np.uint8)[::stride, ::stride]
    palette = {
        0: (42, 66, 72),
        1: (232, 202, 132),  # primary convergent orogen
        2: (206, 146, 92),   # oblique branch
        3: (226, 88, 64),    # volcanic arc branch
        4: (150, 128, 88),   # foreland/sedimentary flank
        5: (132, 178, 172),  # rifted shoulder/highland
        6: (250, 232, 104),  # complex junction orogen
    }
    rgb = np.zeros((cls.shape[0], cls.shape[1], 3), dtype=np.uint8)
    for code, color in palette.items():
        rgb[cls == code] = color
    _save_clean_rgb_map(
        rgb,
        output_path,
        title="v4 orogen / mountain-branch network",
        description="Update 31 diagnostic for plate_history_v4: classifies mountain-branching cells into primary convergent belts, oblique branches, volcanic-arc branches, foreland flanks, rifted shoulders, and complex junction orogens. This makes mountain feedback easier than inspecting a heat map alone.",
        items=[
            ("no v4 branch/orogen class", palette[0]),
            ("primary convergent orogen", palette[1]),
            ("oblique mountain branch", palette[2]),
            ("volcanic-arc mountain branch", palette[3]),
            ("foreland/sedimentary flank", palette[4]),
            ("rifted shoulder/highland", palette[5]),
            ("complex-junction orogen", palette[6]),
        ],
        stats={
            "source_width": int(cls.shape[1]),
            "source_height": int(cls.shape[0]),
            "orogen_network_cells": int(np.count_nonzero(cls > 0)),
            "oblique_branch_cells": int(np.count_nonzero(cls == 2)),
            "volcanic_branch_cells": int(np.count_nonzero(cls == 3)),
            "foreland_flank_cells": int(np.count_nonzero(cls == 4)),
        },
        scale=_world_scale_meta(profile, out_w, out_h, kind="diagnostic_downsample", stride=stride),
    )


def save_main_planet_v4_control_response_view(system: StarSystem, output_path: str | Path) -> None:
    if system.main_planet_profile is None:
        raise RuntimeError("v4 control-response diagnostic requested, but no Main Planet profile exists.")
    import numpy as np
    profile = system.main_planet_profile
    terrain = profile.terrain
    if getattr(terrain, "terrain_v4_control_response_class", None) is None:
        raise RuntimeError("Terrain has no v4 control-response class grid.")
    out_w, out_h, stride = _diagnostic_downsample_dimensions(terrain.width, terrain.height)
    cls = np.asarray(terrain.terrain_v4_control_response_class, dtype=np.uint8)[::stride, ::stride]
    palette = {
        0: (30, 48, 64),
        1: (232, 174, 78),
        2: (244, 126, 72),
        3: (74, 184, 220),
        4: (194, 124, 220),
        5: (250, 230, 106),
    }
    rgb = np.zeros((cls.shape[0], cls.shape[1], 3), dtype=np.uint8)
    for code, color in palette.items():
        rgb[cls == code] = color
    _save_clean_rgb_map(
        rgb,
        output_path,
        title="v4 control-response classes",
        description="Update 32 diagnostic for plate_history_v4: shows which user-facing v4 control should visibly affect each cell. If topology/island/rift strength changes do not alter this map or related terrain, the control path is broken.",
        items=[
            ("weak/no v4 control response", palette[0]),
            ("topology / boundary / mountain-branch response", palette[1]),
            ("volcanic island-chain response", palette[2]),
            ("rift-cut / gulf / basin response", palette[3]),
            ("mixed two-control response", palette[4]),
            ("mixed topology + island + rift response", palette[5]),
        ],
        stats={
            "source_width": int(cls.shape[1]),
            "source_height": int(cls.shape[0]),
            "active_response_cells": int(np.count_nonzero(cls > 0)),
            "topology_response_cells": int(np.count_nonzero(cls == 1)),
            "island_response_cells": int(np.count_nonzero(cls == 2)),
            "rift_response_cells": int(np.count_nonzero(cls == 3)),
            "mixed_response_cells": int(np.count_nonzero(cls >= 4)),
        },
        scale=_world_scale_meta(profile, out_w, out_h, kind="diagnostic_downsample", stride=stride),
    )

def save_main_planet_v4_elevation_delta_view(system: StarSystem, output_path: str | Path) -> None:
    if system.main_planet_profile is None:
        raise RuntimeError("v4 elevation-delta diagnostic requested, but no Main Planet profile exists.")
    import numpy as np
    profile = system.main_planet_profile
    terrain = profile.terrain
    if getattr(terrain, "terrain_v4_elevation_delta_m", None) is None:
        raise RuntimeError("Terrain has no v4 elevation-delta grid.")
    out_w, out_h, stride = _diagnostic_downsample_dimensions(terrain.width, terrain.height)
    delta = np.asarray(terrain.terrain_v4_elevation_delta_m, dtype=np.float32)[::stride, ::stride]
    scale_m = max(250.0, float(np.nanpercentile(np.abs(delta), 98.5)) if delta.size else 500.0)
    t = np.clip(delta / scale_m, -1.0, 1.0)
    rgb = np.zeros((delta.shape[0], delta.shape[1], 3), dtype=np.uint8)
    # Lowering: blue/cyan. Uplift: orange/white. Near-zero: dark neutral.
    neg = t < -0.04
    pos = t > 0.04
    zero = ~(neg | pos)
    rgb[zero] = (42, 50, 56)
    nt = np.abs(t[neg])
    rgb[neg, 0] = (34 + 18 * nt).astype(np.uint8)
    rgb[neg, 1] = (86 + 100 * nt).astype(np.uint8)
    rgb[neg, 2] = (138 + 96 * nt).astype(np.uint8)
    pt = t[pos]
    rgb[pos, 0] = (126 + 124 * pt).astype(np.uint8)
    rgb[pos, 1] = (82 + 118 * pt).astype(np.uint8)
    rgb[pos, 2] = (48 + 70 * pt).astype(np.uint8)
    _save_clean_rgb_map(
        rgb,
        output_path,
        title="v4 elevation delta from stable v3",
        description="Update 33 diagnostic for plate_history_v4: signed terrain height change in meters after v4 topology, island-chain, rift, and mountain-branch shaping is applied to the stable v3 baseline. This is the most direct A/B map for whether v4 controls changed terrain.",
        items=[
            ("near-zero v4 change", (42, 50, 56)),
            ("v4 lowering / rift or basin cut", (44, 166, 220)),
            ("v4 uplift / mountain or island growth", (238, 180, 92)),
        ],
        stats={
            "source_width": int(delta.shape[1]),
            "source_height": int(delta.shape[0]),
            "scale_m_98p5_abs": round(float(scale_m), 2),
            "mean_abs_delta_m": round(float(np.mean(np.abs(delta))), 3) if delta.size else 0,
            "changed_cells_gt_50m": int(np.count_nonzero(np.abs(delta) > 50)),
            "max_uplift_m": int(np.max(delta)) if delta.size else 0,
            "max_lowering_m": int(np.min(delta)) if delta.size else 0,
        },
        scale=_world_scale_meta(profile, out_w, out_h, kind="diagnostic_downsample", stride=stride),
    )


def save_main_planet_v4_landform_change_view(system: StarSystem, output_path: str | Path) -> None:
    if system.main_planet_profile is None:
        raise RuntimeError("v4 landform-change diagnostic requested, but no Main Planet profile exists.")
    import numpy as np
    profile = system.main_planet_profile
    terrain = profile.terrain
    if getattr(terrain, "terrain_v4_landform_change_class", None) is None:
        raise RuntimeError("Terrain has no v4 landform-change class grid.")
    out_w, out_h, stride = _diagnostic_downsample_dimensions(terrain.width, terrain.height)
    cls = np.asarray(terrain.terrain_v4_landform_change_class, dtype=np.uint8)[::stride, ::stride]
    palette = {
        0: (38, 48, 56),
        1: (232, 196, 112),
        2: (246, 132, 68),
        3: (74, 176, 224),
        4: (216, 86, 166),
        5: (246, 230, 104),
    }
    rgb = np.zeros((cls.shape[0], cls.shape[1], 3), dtype=np.uint8)
    for code, color in palette.items():
        rgb[cls == code] = color
    _save_clean_rgb_map(
        rgb,
        output_path,
        title="v4 dominant landform-change classes",
        description="Update 33 diagnostic for plate_history_v4: summarizes the dominant actual terrain change applied by v4 relative to v3: mountain branching, volcanic island uplift, rift lowering, native sliver/microplate corridor, or mixed control response.",
        items=[
            ("weak/no actual v4 terrain change", palette[0]),
            ("mountain/orogen branch uplift", palette[1]),
            ("volcanic island-chain uplift", palette[2]),
            ("rift/gulf/basin lowering", palette[3]),
            ("native sliver/microplate corridor", palette[4]),
            ("mixed v4 terrain response", palette[5]),
        ],
        stats={
            "source_width": int(cls.shape[1]),
            "source_height": int(cls.shape[0]),
            "active_change_cells": int(np.count_nonzero(cls > 0)),
            "mountain_branch_cells": int(np.count_nonzero(cls == 1)),
            "island_uplift_cells": int(np.count_nonzero(cls == 2)),
            "rift_lowering_cells": int(np.count_nonzero(cls == 3)),
            "native_sliver_corridor_cells": int(np.count_nonzero(cls == 4)),
            "mixed_change_cells": int(np.count_nonzero(cls == 5)),
        },
        scale=_world_scale_meta(profile, out_w, out_h, kind="diagnostic_downsample", stride=stride),
    )

def save_main_planet_crust_type_view(system: StarSystem, output_path: str | Path) -> None:
    if system.main_planet_profile is None:
        raise RuntimeError("Crust diagnostic requested, but no Main Planet profile exists.")
    from PIL import Image
    import numpy as np
    profile = system.main_planet_profile
    terrain = profile.terrain
    if terrain.crust_type is None:
        raise RuntimeError("Terrain has no crust diagnostic grid.")
    c = np.asarray(terrain.crust_type, dtype=np.uint8)
    is_v3 = _terrain_is_plate_history_v3(profile) or int(c.max(initial=0)) > 8
    if is_v3:
        palette = {
            0: (20, 46, 82),
            1: (18, 36, 78),     # abyssal/generic oceanic
            2: (84, 196, 232),    # young ridge zone
            3: (8, 24, 58),       # old oceanic
            4: (92, 52, 138),     # trench/subduction trough
            5: (118, 108, 202),   # fracture/transform oceanic
            6: (66, 184, 172),    # seamount/oceanic plateau
            7: (116, 222, 212),   # shallow submerged shelf
            8: (122, 106, 68),    # craton/core
            9: (164, 148, 92),    # continental interior
            10: (238, 220, 164),  # orogenic belt
            11: (164, 96, 86),    # old suture
            12: (218, 142, 64),   # rifted continent
            13: (188, 184, 112),  # transitional/passive margin
            14: (204, 174, 116),  # sedimentary/foreland basin
            15: (186, 112, 96),   # accreted terrane/microcontinent
            16: (222, 70, 56),    # continental volcanic arc
            17: (244, 118, 62),   # oceanic island arc
            18: (246, 184, 76),   # hotspot/oceanic island
            19: (64, 142, 196),   # upper continental slope
            20: (38, 98, 158),    # continental rise
            21: (24, 64, 118),    # deep submerged continental margin
        }
        rgb = np.zeros((c.shape[0], c.shape[1], 3), dtype=np.uint8)
        for code, color in palette.items():
            rgb[c == code] = color
        image = Image.fromarray(rgb, mode="RGB")
        output_path = _prepare_output_path(output_path)
        _save_image_fast(image, output_path)
        _write_map_legend_sidecar(
            output_path,
            title="v3 crust circumstance diagnostic",
            description="Richer crust labels derived from final v3 continuous fields and final land/ocean state. Labels summarize dominant causes; they do not select separate terrain equations.",
            items=[
                ("abyssal/generic oceanic crust", palette[1]), ("young oceanic/ridge zone", palette[2]), ("old oceanic crust", palette[3]),
                ("trench/subduction trough", palette[4]), ("fracture/transform oceanic crust", palette[5]), ("seamount/oceanic plateau", palette[6]),
                ("shallow submerged continental shelf", palette[7]), ("upper continental slope", palette[19]), ("continental rise", palette[20]), ("deep submerged continental margin", palette[21]),
                ("continental craton/core", palette[8]), ("continental interior/shield", palette[9]),
                ("young orogenic belt", palette[10]), ("old suture/eroded orogen", palette[11]), ("rifted continental crust", palette[12]),
                ("transitional/passive margin", palette[13]), ("sedimentary/foreland basin", palette[14]), ("accreted terrane/microcontinent", palette[15]),
                ("continental volcanic arc", palette[16]), ("oceanic island arc", palette[17]), ("hotspot/oceanic island", palette[18]),
            ],
            stats={"source_width": c.shape[1], "source_height": c.shape[0], "class_count": int(len(set(np.unique(c).tolist()) - {0}))},
            scale=_world_scale_meta(profile, image.width, image.height, kind="diagnostic_downsample"),
        )
        return
    rgb = np.zeros((c.shape[0], c.shape[1], 3), dtype=np.uint8)
    rgb[c == 0] = (25, 60, 120)
    rgb[c == 1] = (75, 135, 175)
    rgb[c == 2] = (150, 142, 96)
    rgb[c == 3] = (215, 205, 170)
    rgb[c == 4] = (190, 165, 105)
    rgb[c == 5] = (196, 150, 96)
    rgb[c == 6] = (210, 92, 72)
    rgb[c == 7] = (225, 145, 74)
    rgb[c == 8] = (108, 155, 185)
    image = Image.fromarray(rgb, mode="RGB").resize((max(1024, c.shape[1] * 2), max(512, c.shape[0] * 2)), Image.Resampling.NEAREST)
    output_path = _prepare_output_path(output_path)
    _save_image_fast(image, output_path)
    _write_map_legend_sidecar(output_path, title="Crust type diagnostic", description="Plate Terrain 15 crust classes: oceanic crust, shelves, continental interiors, active margins, rifted margins, microcontinents, volcanic arcs, hotspot chains, and oceanic plateaus.", items=[("abyssal/oceanic crust", (25, 60, 120)), ("submerged continental shelf", (75, 135, 175)), ("continental core/interior", (150, 142, 96)), ("active/orogenic continent", (215, 205, 170)), ("rifted continental margin", (190, 165, 105)), ("microcontinent/fragment", (196, 150, 96)), ("volcanic island arc", (210, 92, 72)), ("hotspot/oceanic island", (225, 145, 74)), ("oceanic plateau", (108, 155, 185))], stats={"source_width": c.shape[1], "source_height": c.shape[0]}, scale=_world_scale_meta(profile, image.width, image.height, kind="diagnostic_downsample"))


def save_main_planet_coastline_margin_types_view(system: StarSystem, output_path: str | Path) -> None:
    """Diagnostic map showing broad coastline/margin styles."""
    if system.main_planet_profile is None:
        raise RuntimeError("Coastline margin diagnostic requested, but no Main Planet profile exists.")
    from PIL import Image, ImageDraw
    import numpy as np
    output_path = _prepare_output_path(output_path)
    terrain = system.main_planet_profile.terrain
    hydrology = system.main_planet_profile.hydrology
    out_w, out_h, stride = _diagnostic_downsample_dimensions(terrain.width, terrain.height)
    land = np.asarray(terrain.is_land, dtype=bool)[::stride, ::stride]
    elev = np.asarray(terrain.elevation_m, dtype=np.int32)[::stride, ::stride]
    rivers = np.asarray(hydrology.river_intensity, dtype=np.uint8)[::stride, ::stride]
    rgb = np.zeros((out_h, out_w, 3), dtype=np.uint8)
    rgb[:] = (30, 70, 120)
    rgb[land] = (132, 146, 105)

    north = np.vstack((land[0:1, :], land[:-1, :]))
    south = np.vstack((land[1:, :], land[-1:, :]))
    west = np.roll(land, 1, axis=1)
    east = np.roll(land, -1, axis=1)
    coast = land & ((~north) | (~south) | (~west) | (~east))
    # Local relief proxy.
    elev_n = np.vstack((elev[0:1, :], elev[:-1, :]))
    elev_s = np.vstack((elev[1:, :], elev[-1:, :]))
    elev_w = np.roll(elev, 1, axis=1)
    elev_e = np.roll(elev, -1, axis=1)
    relief = np.maximum.reduce([abs(elev - elev_n), abs(elev - elev_s), abs(elev - elev_w), abs(elev - elev_e)])
    low_delta = coast & (elev < 90) & (rivers > 125)
    rugged = coast & (relief > 220)
    shelf_plain = coast & (elev < 95) & (~low_delta)
    moderate = coast & (~rugged) & (~shelf_plain) & (~low_delta)
    rgb[moderate] = (240, 230, 120)      # passive/mixed margin
    rgb[shelf_plain] = (90, 190, 120)    # low shelf/coastal plain
    rgb[rugged] = (205, 85, 55)          # rugged active/fjord-like margin
    rgb[low_delta] = (40, 220, 170)      # deltaic mouth/coastal wetland
    image = Image.fromarray(rgb, mode="RGB")
    draw = ImageDraw.Draw(image)
    draw.rectangle((8, 8, min(out_w - 8, 940), 82), fill=(255, 255, 255), outline=(30, 30, 30))
    draw.text((18, 18), "Coastline / margin-type diagnostic", fill=(0, 0, 0))
    draw.text((18, 42), "yellow: passive/mixed | green: coastal plain/shelf | red: rugged active/fjord margin | teal: deltaic", fill=(0, 0, 0))
    _save_image_fast(image, output_path)


def save_main_planet_inland_lakes_view(system: StarSystem, output_path: str | Path) -> None:
    """Diagnostic map distinguishing ocean water from enclosed inland lakes/seas."""
    if system.main_planet_profile is None:
        raise RuntimeError("Inland lakes diagnostic requested, but no Main Planet profile exists.")
    from PIL import Image, ImageDraw
    import numpy as np
    from collections import deque
    output_path = _prepare_output_path(output_path)
    terrain = system.main_planet_profile.terrain
    hydrology = system.main_planet_profile.hydrology
    out_w, out_h, stride = _diagnostic_downsample_dimensions(terrain.width, terrain.height)
    land = np.asarray(terrain.is_land, dtype=bool)[::stride, ::stride]
    elev = np.asarray(terrain.elevation_m, dtype=np.int32)[::stride, ::stride]
    lake_candidates = np.asarray(hydrology.lake_mask, dtype=bool)[::stride, ::stride]
    water = ~land
    comp = np.full((out_h, out_w), -1, dtype=np.int32)
    sizes: dict[int, int] = {}
    cid = 0
    for r in range(out_h):
        for c in range(out_w):
            if not water[r, c] or comp[r, c] >= 0:
                continue
            q = deque([(r, c)]); comp[r, c] = cid; size = 0
            while q:
                rr, cc = q.popleft(); size += 1
                for dr, dc in ((-1,0),(1,0),(0,-1),(0,1)):
                    nr = rr + dr
                    if nr < 0 or nr >= out_h: continue
                    nc = (cc + dc) % out_w
                    if water[nr, nc] and comp[nr, nc] < 0:
                        comp[nr, nc] = cid; q.append((nr, nc))
            sizes[cid] = size; cid += 1
    ocean_id = max(sizes.items(), key=lambda item: item[1])[0] if sizes else -1
    inland = water & (comp >= 0) & (comp != ocean_id)
    rgb = np.zeros((out_h, out_w, 3), dtype=np.uint8)
    rgb[:] = (30, 75, 125)
    rgb[land] = (145, 150, 100)
    rgb[inland] = (105, 70, 180)
    rgb[lake_candidates] = (50, 205, 235)
    # shallow coastal/inland plains
    rgb[land & (elev < 120)] = (175, 178, 110)
    image = Image.fromarray(rgb, mode="RGB")
    draw = ImageDraw.Draw(image)
    draw.rectangle((8, 8, min(out_w - 8, 900), 82), fill=(255, 255, 255), outline=(30, 30, 30))
    draw.text((18, 18), "Inland lakes and enclosed seas diagnostic", fill=(0, 0, 0))
    draw.text((18, 42), "purple: enclosed water body | cyan: hydrologic lake/sink candidate | blue: world ocean", fill=(0, 0, 0))
    _save_image_fast(image, output_path)


def save_main_planet_islands_archipelago_view(system: StarSystem, output_path: str | Path) -> None:
    if system.main_planet_profile is None:
        raise RuntimeError("Island diagnostic requested, but no Main Planet profile exists.")
    import numpy as np
    from scipy import ndimage
    profile = system.main_planet_profile
    terrain = profile.terrain
    out_w, out_h, stride = _diagnostic_downsample_dimensions(terrain.width, terrain.height)
    is_v3 = _terrain_is_plate_history_v3(profile)
    if is_v3:
        land = np.asarray(terrain.is_land, dtype=bool)[::stride, ::stride]
        elev = np.asarray(terrain.elevation_m, dtype=np.int32)[::stride, ::stride]
        origin = np.asarray(getattr(terrain, "terrain_island_origin_class", None), dtype=np.int16)[::stride, ::stride] if getattr(terrain, "terrain_island_origin_class", None) is not None else None
        labels, count = _label_xwrap_bool(land)
        sizes = np.bincount(labels.ravel()) if count else np.asarray([0])
        land_cells = max(1, int(land.sum()))
        continent_threshold = max(32, int(land_cells * 0.028))
        rgb = np.zeros((out_h, out_w, 3), dtype=np.uint8)
        rgb[:] = (35, 82, 135)
        rgb[land] = (132, 146, 105)
        component_island_cells = 0
        archipelago_cells = 0
        for lab in range(1, count + 1):
            comp = labels == lab
            size = int(sizes[lab]) if lab < len(sizes) else int(comp.sum())
            if size >= continent_threshold:
                continue
            component_island_cells += size
            code = int(round(float(np.median(origin[comp])))) if origin is not None and np.any(comp) else 2
            if code == 3:
                color = (225, 92, 58)
            elif code == 4:
                color = (164, 106, 190)
            elif code == 5 or (np.any(comp) and float(np.mean(elev[comp])) > 900.0):
                color = (242, 238, 190)
            else:
                color = (230, 185, 74)
            rgb[comp] = color
            if size <= max(4, int(land_cells * 0.0025)):
                archipelago_cells += size
        _save_clean_rgb_map(
            rgb,
            output_path,
            title="Island and archipelago diagnostic",
            description="v3 island map recomputed from the final land mask with east/west wrapping. Component labels are diagnostic only and no longer depend on stale pre-v3 island-origin rasters.",
            items=[("ocean", (35, 82, 135)), ("large landmass / continent", (132, 146, 105)), ("shelf island / small island", (230, 185, 74)), ("volcanic / island arc", (225, 92, 58)), ("microcontinent / terrane", (164, 106, 190)), ("hotspot / high island", (242, 238, 190))],
            stats={"land_component_count": int(count), "continent_threshold_cells": int(continent_threshold), "island_cells": int(component_island_cells), "small_archipelago_cells": int(archipelago_cells)},
            scale=_world_scale_meta(profile, out_w, out_h, kind="diagnostic_downsample", stride=stride),
        )
        return
    if getattr(terrain, "terrain_island_origin_class", None) is not None:
        origin = np.asarray(terrain.terrain_island_origin_class, dtype=np.uint8)[::stride, ::stride]
        colors = {
            0: (35, 82, 135),
            1: (132, 146, 105),
            2: (230, 185, 74),
            3: (225, 92, 58),
            4: (164, 106, 190),
            5: (242, 238, 190),
        }
        rgb = np.zeros((origin.shape[0], origin.shape[1], 3), dtype=np.uint8)
        for code, color in colors.items():
            rgb[origin == code] = color
        _save_clean_rgb_map(rgb, output_path, title="Island and archipelago diagnostic", description="Stage 3C.4 island-origin classes: shelf islands, volcanic/arc islands, microcontinents/terranes, hotspot/high islands, and large landmasses.", items=[("water/non-island", (35, 82, 135)), ("continent / large land", (132, 146, 105)), ("shelf island", (230, 185, 74)), ("volcanic/arc island", (225, 92, 58)), ("microcontinent/terrane", (164, 106, 190)), ("hotspot/high island", (242, 238, 190))], scale=_world_scale_meta(profile, out_w, out_h, kind="diagnostic_downsample", stride=stride))
        return
    land = np.asarray(terrain.is_land, dtype=bool)[::stride, ::stride]
    elev = np.asarray(terrain.elevation_m, dtype=np.int32)[::stride, ::stride]
    rgb = np.zeros((out_h, out_w, 3), dtype=np.uint8)
    rgb[:] = (35, 80, 135)
    rgb[land] = (136, 147, 100)
    world = out_h * out_w
    island_limit = max(4, int(world * 0.0038))
    small_island_limit = max(3, int(world * 0.00065))
    structure = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)
    labels, count = ndimage.label(land, structure=structure)
    if count:
        sizes = np.bincount(labels.ravel())
        island_mask = land & (sizes[labels] <= island_limit)
        small_island_mask = land & (sizes[labels] <= small_island_limit)
    else:
        island_mask = np.zeros_like(land, dtype=bool)
        small_island_mask = np.zeros_like(land, dtype=bool)
    rgb[island_mask] = (230, 180, 65)
    rgb[small_island_mask] = (245, 105, 70)
    rgb[island_mask & (elev > 900)] = (245, 245, 210)
    _save_clean_rgb_map(rgb, output_path, title="Island and archipelago diagnostic", description="Highlights small islands, archipelago chains, and high volcanic/island-arc terrain.", items=[("ocean", (35, 80, 135)), ("large land", (136, 147, 100)), ("small islands / archipelago", (230, 180, 65)), ("very small islands", (245, 105, 70)), ("high volcanic/island-arc terrain", (245, 245, 210))], stats={"land_component_count": int(count), "island_limit_cells": int(island_limit), "small_island_limit_cells": int(small_island_limit)}, scale=_world_scale_meta(profile, out_w, out_h, kind="diagnostic_downsample", stride=stride))


def save_main_planet_terrain_region_maps(system: StarSystem, output_dir: str | Path, rows: int = 8, cols: int = 16) -> None:
    """Write locally rescaled terrain crops for regional topography inspection.

    The full-world terrain map uses one global elevation/depth scale, which can
    hide subtle regional relief. This diagnostic set cuts the world into an
    rows x cols grid and rescales land elevations and water depths separately
    inside each crop.
    """
    if system.main_planet_profile is None:
        raise RuntimeError("Terrain regional maps requested, but no Main Planet profile exists.")

    from pathlib import Path as _Path
    from PIL import Image, ImageDraw
    import numpy as np

    out = _Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    terrain = system.main_planet_profile.terrain
    elevation = np.asarray(terrain.elevation_m, dtype=np.float32)
    land = np.asarray(terrain.is_land, dtype=bool)
    height, width = elevation.shape

    row_edges = [round(i * height / rows) for i in range(rows + 1)]
    col_edges = [round(i * width / cols) for i in range(cols + 1)]

    index_lines = [
        "WorldGen regional terrain map set",
        "=================================",
        "",
        f"Source terrain grid: {width} x {height}",
        f"Grid: {rows} rows x {cols} columns = {rows * cols} regional maps",
        "Each regional PNG uses local land elevation scaling and local water-depth scaling.",
        "",
        "filename,row,col,row_start,row_end_exclusive,col_start,col_end_exclusive,land_min_m,land_max_m,water_min_depth_m,water_max_depth_m,land_fraction",
    ]

    for rr in range(rows):
        r0, r1 = row_edges[rr], row_edges[rr + 1]
        for cc in range(cols):
            c0, c1 = col_edges[cc], col_edges[cc + 1]
            elev_crop = elevation[r0:r1, c0:c1]
            land_crop = land[r0:r1, c0:c1]
            ch, cw = elev_crop.shape

            rgb = np.zeros((ch, cw, 3), dtype=np.uint8)

            land_values = elev_crop[land_crop]
            water_values = elev_crop[~land_crop]
            if land_values.size:
                land_min = float(np.min(land_values))
                land_max = float(np.max(land_values))
                land_span = max(1.0, land_max - land_min)
                land_t = np.clip((elev_crop - land_min) / land_span, 0.0, 1.0)
            else:
                land_min = 0.0
                land_max = 0.0
                land_t = np.zeros_like(elev_crop, dtype=np.float32)

            if water_values.size:
                depths = np.abs(water_values)
                water_min = float(np.min(depths))
                water_max = float(np.max(depths))
                water_span = max(1.0, water_max - water_min)
                water_t = np.clip((np.abs(elev_crop) - water_min) / water_span, 0.0, 1.0)
            else:
                water_min = 0.0
                water_max = 0.0
                water_t = np.zeros_like(elev_crop, dtype=np.float32)

            # Water: shallow cyan/blue to deep navy, rescaled per region.
            rgb[:, :, 0] = (104 - 86 * water_t).astype(np.uint8)
            rgb[:, :, 1] = (168 - 107 * water_t).astype(np.uint8)
            rgb[:, :, 2] = (211 - 89 * water_t).astype(np.uint8)

            # Land: same general terrain palette as main map, but locally stretched.
            low = land_crop & (land_t < 0.22)
            mid = land_crop & (land_t >= 0.22) & (land_t < 0.55)
            high = land_crop & (land_t >= 0.55) & (land_t < 0.82)
            peak = land_crop & (land_t >= 0.82)
            f = np.zeros_like(land_t, dtype=np.float32)
            f[low] = land_t[low] / 0.22
            rgb[low, 0] = (200 + (108 - 200) * f[low]).astype(np.uint8)
            rgb[low, 1] = (185 + (155 - 185) * f[low]).astype(np.uint8)
            rgb[low, 2] = (116 + (84 - 116) * f[low]).astype(np.uint8)
            f[mid] = (land_t[mid] - 0.22) / 0.33
            rgb[mid, 0] = (108 + (143 - 108) * f[mid]).astype(np.uint8)
            rgb[mid, 1] = (155 + (125 - 155) * f[mid]).astype(np.uint8)
            rgb[mid, 2] = (84 + (82 - 84) * f[mid]).astype(np.uint8)
            f[high] = (land_t[high] - 0.55) / 0.27
            rgb[high, 0] = (143 + (180 - 143) * f[high]).astype(np.uint8)
            rgb[high, 1] = (125 + (174 - 125) * f[high]).astype(np.uint8)
            rgb[high, 2] = (82 + (156 - 82) * f[high]).astype(np.uint8)
            f[peak] = (land_t[peak] - 0.82) / 0.18
            rgb[peak, 0] = (180 + (245 - 180) * f[peak]).astype(np.uint8)
            rgb[peak, 1] = (174 + (245 - 174) * f[peak]).astype(np.uint8)
            rgb[peak, 2] = (156 + (240 - 156) * f[peak]).astype(np.uint8)

            # Mark local land/water boundaries without hiding the regional relief.
            if ch > 2 and cw > 2:
                coast = np.zeros((ch, cw), dtype=bool)
                coast[1:-1, 1:-1] = land_crop[1:-1, 1:-1] & (
                    (~land_crop[:-2, 1:-1]) | (~land_crop[2:, 1:-1]) |
                    (~land_crop[1:-1, :-2]) | (~land_crop[1:-1, 2:])
                )
                rgb[coast] = (30, 30, 26)

            legend_height = 74
            image = Image.new("RGB", (cw, ch + legend_height), "white")
            image.paste(Image.fromarray(rgb, mode="RGB"), (0, 0))
            draw = ImageDraw.Draw(image)
            y0 = ch + 8
            filename = f"terrain_region_r{rr + 1:02d}_c{cc + 1:02d}.png"
            draw.text((8, y0), f"Terrain region r{rr + 1:02d} c{cc + 1:02d} | local land/water scale", fill=(0, 0, 0))
            draw.text(
                (8, y0 + 18),
                f"rows {r0}-{r1 - 1}, cols {c0}-{c1 - 1} | land {float(np.mean(land_crop)):.1%}",
                fill=(0, 0, 0),
            )
            draw.text(
                (8, y0 + 36),
                f"land {land_min:,.0f}..{land_max:,.0f} m | water depth {water_min:,.0f}..{water_max:,.0f} m",
                fill=(0, 0, 0),
            )
            # Save regional maps at their native crop size; image max width only
            # matters for very large future map sizes.
            _save_image_fast(image, out / filename)
            index_lines.append(
                f"{filename},{rr + 1},{cc + 1},{r0},{r1},{c0},{c1},{land_min:.0f},{land_max:.0f},{water_min:.0f},{water_max:.0f},{float(np.mean(land_crop)):.5f}"
            )

    (out / "terrain_region_index.csv").write_text("\n".join(index_lines) + "\n", encoding="utf-8")

# ---------------------------------------------------------------------------
# Clean map-only raster renderers with separate legend sidecars
#
# These definitions intentionally appear at the end of the module so they
# override earlier renderers that embedded bottom legend bands or in-map title
# boxes.  Generated map PNGs now contain only map pixels; per-map legend/scale
# details are written next to each image as <mapname>.legend.json and also merged
# into map_legends.json in the same folder for the Web UI.
# ---------------------------------------------------------------------------


def _legend_hex(color) -> str:
    if isinstance(color, str):
        value = color.strip()
        if value.startswith("#") and len(value) in (4, 7):
            return value
        return value
    try:
        r, g, b = color[:3]
        return f"#{int(r):02x}{int(g):02x}{int(b):02x}"
    except Exception:
        return "#000000"


def _legend_entries(items):
    entries = []
    for item in items or []:
        if isinstance(item, dict):
            entry = dict(item)
            if "color" in entry:
                entry["color"] = _legend_hex(entry["color"])
            if "colors" in entry:
                entry["colors"] = [_legend_hex(c) for c in entry.get("colors", [])]
            entries.append(entry)
            continue
        if len(item) >= 2:
            label, color = item[0], item[1]
            entries.append({"label": str(label), "color": _legend_hex(color)})
    return entries


def _write_map_legend_sidecar(output_path: str | Path, *, title: str, description: str = "", items=None, stats=None, scale=None, notes=None) -> None:
    import json as _json
    from pathlib import Path as _Path
    from PIL import Image as _PILImage

    path = _Path(output_path)
    sidecar = path.with_suffix(".legend.json")
    data = {
        "schema_version": 1,
        "map_file": path.name,
        "map_has_embedded_legend": False,
        "title": title,
        "description": description,
        "legend": _legend_entries(items or []),
        "stats": stats or {},
        "scale": scale or {},
        "notes": notes or [],
    }
    try:
        with _PILImage.open(path) as img:
            exported_w = int(img.width)
            exported_h = int(img.height)
            data["exported_image"] = {"width": exported_w, "height": exported_h}
            radius_earth = data.get("scale", {}).get("planet_radius_earth")
            if radius_earth and exported_w > 0 and exported_h > 0:
                radius_km = float(radius_earth) * 6371.0
                data["scale"]["exported_equator_km_per_pixel"] = (2.0 * math.pi * radius_km) / exported_w
                data["scale"]["exported_north_south_km_per_pixel"] = (math.pi * radius_km) / exported_h
    except Exception:
        pass
    sidecar.write_text(_json.dumps(data, indent=2), encoding="utf-8")

    manifest_path = path.parent / "map_legends.json"
    try:
        manifest = _json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
        if not isinstance(manifest, dict):
            manifest = {}
    except Exception:
        manifest = {}
    manifest[path.name] = data
    manifest_path.write_text(_json.dumps(manifest, indent=2), encoding="utf-8")


def _world_scale_meta(profile, width: int, height: int, *, kind: str = "full_world", stride: int = 1) -> dict:
    terrain = getattr(profile, "terrain", None)
    radius_earth = float(getattr(terrain, "planet_radius_earth", 1.0) or 1.0)
    radius_km = radius_earth * 6371.0
    return {
        "kind": kind,
        "projection": "equirectangular" if kind in {"full_world", "diagnostic_downsample"} else "local_region",
        "data_width": int(width),
        "data_height": int(height),
        "source_stride": int(stride),
        "planet_radius_earth": radius_earth,
        "equator_km_per_pixel": (2.0 * math.pi * radius_km) / max(1, width),
        "north_south_km_per_pixel": (math.pi * radius_km) / max(1, height),
    }


def _save_clean_rgb_map(rgb, output_path: str | Path, *, title: str, description: str = "", items=None, stats=None, scale=None, notes=None) -> None:
    from PIL import Image
    output_path = _prepare_output_path(output_path)
    image = Image.fromarray(rgb, mode="RGB")
    _save_image_fast(image, output_path)
    _write_map_legend_sidecar(output_path, title=title, description=description, items=items, stats=stats, scale=scale, notes=notes)


def _continuous_legend(label: str, color_stops, minimum=None, maximum=None, unit: str = "") -> list[dict]:
    return [{
        "kind": "gradient",
        "label": label,
        "colors": [_legend_hex(c) for c in color_stops],
        "min": minimum,
        "max": maximum,
        "unit": unit,
    }]


TERRAIN_LEGEND = [
    ("deep ocean", (29, 69, 117)),
    ("shallow sea", (65, 135, 195)),
    ("lowland", (143, 165, 98)),
    ("highland", (144, 149, 68)),
    ("mountain", (244, 244, 188)),
    ("major coastline", (24, 24, 20)),
]


def save_main_planet_terrain_view(system: StarSystem, output_path: str | Path) -> None:
    if system.main_planet_profile is None:
        raise RuntimeError("Terrain visualization requested, but no Main Planet profile exists.")
    profile = system.main_planet_profile
    terrain = profile.terrain
    stats = {
        "planet_name": profile.planet_name,
        "width": terrain.width,
        "height": terrain.height,
        "ocean_fraction": terrain.ocean_fraction,
        "land_fraction": terrain.land_fraction,
        "min_elevation_m": terrain.min_elevation_m,
        "max_elevation_m": terrain.max_elevation_m,
        "mean_land_elevation_m": terrain.mean_land_elevation_m,
        "mean_ocean_depth_m": terrain.mean_ocean_depth_m,
    }
    terrain_title = "artifact-repaired real Earth terrain" if str(getattr(terrain, "source", "")).startswith("real_earth") else "detailed procedural terrain"
    _save_clean_rgb_map(
        _terrain_rgb_array(terrain),
        output_path,
        title=f"{profile.planet_name} {terrain_title}",
        description="Terrain elevation/depth map with major coastline overlay. The PNG contains map pixels only; legend and scale are stored here.",
        items=TERRAIN_LEGEND,
        stats=stats,
        scale=_world_scale_meta(profile, terrain.width, terrain.height),
    )


def save_main_planet_temperature_view(system: StarSystem, output_path: str | Path) -> None:
    profile, climate, terrain = _require_main_planet_climate(system)
    rgb = _temperature_rgb_array(climate.annual_mean_temp_c_x10, climate.min_temp_c, climate.max_temp_c)
    _save_clean_rgb_map(
        rgb,
        output_path,
        title=f"{profile.planet_name} annual mean temperature",
        description="Annual mean temperature map. The PNG contains map pixels only.",
        items=_continuous_legend("temperature", [(45, 90, 200), (235, 238, 230), (190, 30, 30)], climate.min_temp_c, climate.max_temp_c, "°C"),
        stats={
            "mean_land_temp_c": climate.mean_land_temp_c,
            "mean_ocean_temp_c": climate.mean_ocean_temp_c,
            "min_temp_c": climate.min_temp_c,
            "max_temp_c": climate.max_temp_c,
            "ocean_fraction": terrain.ocean_fraction,
        },
        scale=_world_scale_meta(profile, climate.width, climate.height),
    )


def save_main_planet_precipitation_view(system: StarSystem, output_path: str | Path) -> None:
    profile, climate, terrain = _require_main_planet_climate(system)
    import numpy as np
    land = np.asarray(terrain.is_land, dtype=bool)
    rgb = _precip_rgb_array(climate.annual_precip_mm, climate.max_precip_mm, land_mask=land)
    coast = np.frombuffer(_cached_large_coastline_mask(terrain.is_land), dtype=np.uint8).reshape((climate.height, climate.width)).astype(bool)
    rgb[coast] = (25, 25, 22)
    _save_clean_rgb_map(
        rgb,
        output_path,
        title=f"{profile.planet_name} annual land precipitation",
        description="Annual precipitation map. Oceans are masked blue-gray; land rainfall uses a logarithmic dry-to-wet scale.",
        items=_continuous_legend("annual land precipitation", [(230, 210, 135), (95, 175, 105), (45, 95, 185)], 0, climate.max_precip_mm, "mm/year") + [
            {"label": "masked ocean", "color": (70, 110, 150)},
            {"label": "major coastline", "color": (25, 25, 22)},
        ],
        stats={
            "mean_land_precip_mm": climate.mean_land_precip_mm,
            "mean_ocean_precip_mm": climate.mean_ocean_precip_mm,
            "min_precip_mm": climate.min_precip_mm,
            "max_precip_mm": climate.max_precip_mm,
            "water_inventory_class": profile.hydrosphere.water_inventory_class,
        },
        scale=_world_scale_meta(profile, climate.width, climate.height),
    )


def save_main_planet_koppen_view(system: StarSystem, output_path: str | Path) -> None:
    if system.main_planet_profile is None:
        raise RuntimeError("Köppen visualization requested, but no Main Planet profile exists.")
    import numpy as np
    profile = system.main_planet_profile
    climate = profile.climate
    terrain = profile.terrain
    width, height = climate.width, climate.height
    codes_arr = np.asarray(climate.koppen_classification, dtype=object)
    rgb = np.zeros((height, width, 3), dtype=np.uint8)
    present_counts: dict[str, int] = {}
    for code, hex_color in KOPPEN_COLORS.items():
        mask = codes_arr == code
        count = int(mask.sum())
        if count:
            rgb[mask] = _hex_to_rgb(hex_color)
            present_counts[code] = count
    coast = np.frombuffer(_cached_large_coastline_mask(terrain.is_land), dtype=np.uint8).reshape((height, width)).astype(bool)
    rgb[coast] = (8, 8, 8)
    items = [{"label": _koppen_label(code), "color": color, "count": present_counts.get(code, 0)} for code, color in KOPPEN_COLORS.items() if code in present_counts]
    items.append({"label": "major coastline", "color": (8, 8, 8)})
    _save_clean_rgb_map(
        rgb,
        output_path,
        title=f"{profile.planet_name} simplified Köppen climate classes",
        description="Simplified Köppen classification map. Ocean cells use class O; major coasts are outlined.",
        items=items,
        stats={"koppen_summary": dict(climate.koppen_summary), "present_counts": present_counts},
        scale=_world_scale_meta(profile, width, height),
    )


def save_main_planet_hydrology_view(system: StarSystem, output_path: str | Path) -> None:
    if system.main_planet_profile is None:
        raise RuntimeError("Hydrology visualization requested, but no Main Planet profile exists.")
    import numpy as np
    profile = system.main_planet_profile
    terrain = profile.terrain
    hydrology = profile.hydrology
    width, height = hydrology.width, hydrology.height
    land = np.asarray(terrain.is_land, dtype=bool)
    lakes = np.asarray(hydrology.lake_mask, dtype=bool)
    rivers = np.asarray(hydrology.river_intensity, dtype=np.uint16)
    runoff = np.asarray(hydrology.runoff_mm, dtype=np.uint16)
    coast = np.frombuffer(_cached_large_coastline_mask(terrain.is_land), dtype=np.uint8).reshape((height, width)).astype(bool)
    rgb = np.zeros((height, width, 3), dtype=np.uint8)
    rgb[~land] = (79, 134, 198)
    rgb[land] = (217, 198, 138)
    green_boost = np.minimum(55, runoff // 18).astype(np.uint8)
    land_rows, land_cols = np.where(land)
    rgb[land_rows, land_cols, 0] = np.maximum(0, 217 - (green_boost[land_rows, land_cols] // 3))
    rgb[land_rows, land_cols, 2] = np.maximum(0, 138 - (green_boost[land_rows, land_cols] // 5))
    river_mask = rivers > 0
    if river_mask.any():
        t = np.clip(rivers.astype(np.float32) / 255.0, 0.0, 1.0)
        rgb[river_mask, 0] = (90 * (1.0 - t[river_mask])).astype(np.uint8)
        rgb[river_mask, 1] = (165 * (1.0 - t[river_mask]) + 35 * t[river_mask]).astype(np.uint8)
        rgb[river_mask, 2] = (215 * (1.0 - t[river_mask]) + 130 * t[river_mask]).astype(np.uint8)
    rgb[lakes] = (31, 120, 180)
    rgb[coast] = (28, 28, 24)
    _save_clean_rgb_map(
        rgb,
        output_path,
        title=f"{profile.planet_name} hydrology",
        description="Rivers, lake candidates, runoff-tinted land, ocean, and major coastline.",
        items=[
            ("ocean", (79, 134, 198)),
            ("land / low runoff", (217, 198, 138)),
            ("wetter land", (199, 198, 127)),
            ("river", (20, 80, 160)),
            ("lake candidate", (31, 120, 180)),
            ("major coastline", (28, 28, 24)),
        ],
        stats={
            "river_cell_count": hydrology.river_cell_count,
            "estimated_major_river_count": hydrology.estimated_major_river_count,
            "lake_cell_count": hydrology.lake_cell_count,
            "coastal_basin_count": getattr(hydrology, "coastal_basin_count", 0),
            "endorheic_basin_count": getattr(hydrology, "endorheic_basin_count", 0),
        },
        scale=_world_scale_meta(profile, width, height),
    )


def save_main_planet_drainage_basins_view(system: StarSystem, output_path: str | Path) -> None:
    if system.main_planet_profile is None:
        raise RuntimeError("Drainage basin visualization requested, but no Main Planet profile exists.")
    import numpy as np
    profile = system.main_planet_profile
    terrain = profile.terrain
    hydrology = profile.hydrology
    width, height = hydrology.width, hydrology.height
    land = np.asarray(terrain.is_land, dtype=bool)
    basin_ids = np.asarray(hydrology.drainage_basin_id, dtype=np.int32)
    rivers = np.asarray(hydrology.river_intensity, dtype=np.uint16)
    lakes = np.asarray(hydrology.lake_mask, dtype=bool)
    coast = np.frombuffer(_cached_large_coastline_mask(terrain.is_land), dtype=np.uint8).reshape((height, width)).astype(bool)
    rgb = np.zeros((height, width, 3), dtype=np.uint8)
    rgb[~land] = (79, 134, 198)
    rgb[land] = (207, 194, 145)
    rgb[land & (basin_ids == -2)] = (224, 214, 166)
    positive_mask = land & (basin_ids > 0)
    if positive_mask.any():
        b = basin_ids.astype(np.int64)
        xhash = (b * 1103515245 + 12345) & 0x7FFFFFFF
        rgb[:, :, 0][positive_mask] = (115 + (xhash % 95))[positive_mask].astype(np.uint8)
        rgb[:, :, 1][positive_mask] = (105 + ((xhash // 97) % 105))[positive_mask].astype(np.uint8)
        rgb[:, :, 2][positive_mask] = (75 + ((xhash // 7919) % 95))[positive_mask].astype(np.uint8)
    boundary = np.zeros((height, width), dtype=bool)
    boundary[:-1, :] |= (basin_ids[:-1, :] != basin_ids[1:, :]) & land[:-1, :] & land[1:, :]
    boundary[:, :-1] |= (basin_ids[:, :-1] != basin_ids[:, 1:]) & land[:, :-1] & land[:, 1:]
    rgb[boundary] = (55, 48, 38)
    rgb[rivers > 0] = (25, 85, 170)
    rgb[lakes] = (20, 105, 170)
    rgb[coast] = (28, 28, 24)
    _save_clean_rgb_map(
        rgb,
        output_path,
        title=f"{profile.planet_name} drainage basins",
        description="Flow-derived drainage basins with rivers, lakes, basin boundaries, and major coastline.",
        items=[
            ("ocean", (79, 134, 198)),
            ("basin colors", (181, 163, 105)),
            ("merged coastal basins", (224, 214, 166)),
            ("basin boundary", (55, 48, 38)),
            ("river", (25, 85, 170)),
            ("lake candidate", (20, 105, 170)),
            ("major coastline", (28, 28, 24)),
        ],
        stats={
            "drainage_basin_count": hydrology.drainage_basin_count,
            "major_drainage_basin_count": hydrology.major_drainage_basin_count,
            "coastal_basin_count": getattr(hydrology, "coastal_basin_count", 0),
            "endorheic_basin_count": getattr(hydrology, "endorheic_basin_count", 0),
            "minor_coastal_basin_cell_count": getattr(hydrology, "minor_coastal_basin_cell_count", 0),
        },
        scale=_world_scale_meta(profile, width, height),
    )


def save_main_planet_biome_view(system: StarSystem, output_path: str | Path) -> None:
    if system.main_planet_profile is None:
        raise RuntimeError("Biome visualization requested, but no Main Planet profile exists.")
    import numpy as np
    profile = system.main_planet_profile
    biomes = profile.biomes
    width, height = biomes.width, biomes.height
    biome_arr = np.asarray(biomes.biome_classification, dtype=object)
    rgb = np.zeros((height, width, 3), dtype=np.uint8)
    for name, hex_color in BIOME_COLORS.items():
        mask = biome_arr == name
        if mask.any():
            rgb[mask] = _hex_to_rgb(hex_color)
    items = [{"label": name, "color": color, "count": biomes.biome_summary.get(name, 0)} for name, color in BIOME_COLORS.items() if name in biomes.biome_summary]
    _save_clean_rgb_map(
        rgb,
        output_path,
        title=f"{profile.planet_name} biomes / ecoregions",
        description="Biome/ecoregion classification layer. Assumes land-stage life; does not simulate species or evolution.",
        items=items,
        stats={"biome_summary": dict(biomes.biome_summary), "dominant_biome": biomes.dominant_biome},
        scale=_world_scale_meta(profile, width, height),
    )


def save_main_planet_regions_view(system: StarSystem, output_path: str | Path) -> None:
    if system.main_planet_profile is None:
        raise RuntimeError("Region visualization requested, but no Main Planet profile exists.")
    from PIL import Image, ImageDraw
    profile = system.main_planet_profile
    terrain = profile.terrain
    analysis = profile.regions
    cell_w, cell_h = 54, 54
    map_width = max(analysis.cols * cell_w, 640)
    map_height = int(round(map_width * 0.50))
    image = Image.new("RGB", (map_width, map_height), "white")
    draw = ImageDraw.Draw(image)
    top_ids = set(analysis.top_productive_region_ids[:8])
    for region in analysis.regions:
        x0 = int(round(region.col * map_width / analysis.cols))
        x1 = int(round((region.col + 1) * map_width / analysis.cols))
        y0 = int(round(region.row * map_height / analysis.rows))
        y1 = int(round((region.row + 1) * map_height / analysis.rows))
        outline = (30, 30, 30) if region.region_id in top_ids else (130, 130, 130)
        width = 3 if region.region_id in top_ids else 1
        draw.rectangle((x0, y0, x1, y1), fill=_region_color(region), outline=outline, width=width)
    for i in range(1, analysis.cols):
        x = int(round(i * map_width / analysis.cols))
        draw.line((x, 0, x, map_height), fill=(105, 105, 105), width=1)
    for i in range(1, analysis.rows):
        y = int(round(i * map_height / analysis.rows))
        draw.line((0, y, map_width, y), fill=(105, 105, 105), width=1)
    output_path = _prepare_output_path(output_path)
    _save_image_fast(image, output_path)
    _write_map_legend_sidecar(
        output_path,
        title=f"{profile.planet_name} regional analysis",
        description="Coarse environmental region grid. Colors show broad biological productivity score; top regions are outlined. This is not a civilization/population model.",
        items=[
            ("ocean / mostly water", (79, 134, 198)),
            ("very low productivity", (192, 119, 93)),
            ("low productivity", (219, 171, 91)),
            ("moderate productivity", (226, 210, 111)),
            ("high productivity", (141, 185, 92)),
            ("very high productivity", (58, 132, 75)),
            ("top productive region outline", (30, 30, 30)),
        ],
        stats={
            "rows": analysis.rows,
            "cols": analysis.cols,
            "top_productive_region_ids": list(analysis.top_productive_region_ids),
            "source_grid_width": terrain.width,
            "source_grid_height": terrain.height,
        },
        scale={"kind": "coarse_region_grid", "data_width": map_width, "data_height": map_height, "rows": analysis.rows, "cols": analysis.cols},
    )


def save_main_planet_wind_currents_view(system: StarSystem, output_path: str | Path) -> None:
    if system.main_planet_profile is None:
        raise RuntimeError("Wind diagnostic requested, but no Main Planet profile exists.")
    from PIL import Image, ImageDraw
    profile = system.main_planet_profile
    terrain = profile.terrain
    out_w, out_h, stride = _diagnostic_downsample_dimensions(terrain.width, terrain.height)
    image = Image.fromarray(_simple_terrain_background(terrain, stride), mode="RGB")
    draw = ImageDraw.Draw(image)
    step_x = max(64, out_w // 28)
    step_y = max(48, out_h // 16)
    scale = min(step_x, step_y) * 0.42
    for y in range(step_y // 2, out_h, step_y):
        lat = 90.0 - (y + 0.5) * 180.0 / out_h
        dr, dc = _diagnostic_wind_vector(lat)
        for x in range(step_x // 2, out_w, step_x):
            _draw_arrow(draw, x, y, x + dc * scale, y + dr * scale, fill=(245, 245, 245), width=2)
    output_path = _prepare_output_path(output_path)
    _save_image_fast(image, output_path)
    _write_map_legend_sidecar(output_path, title="Prevailing wind diagnostic", description="Simplified prevailing wind belts: trades, westerlies, and polar easterlies over terrain background.", items=[("terrain background", (142, 162, 104)), ("wind arrows", (245, 245, 245))], scale=_world_scale_meta(profile, out_w, out_h, kind="diagnostic_downsample", stride=stride))


def save_main_planet_ocean_currents_view(system: StarSystem, output_path: str | Path) -> None:
    if system.main_planet_profile is None:
        raise RuntimeError("Ocean-current diagnostic requested, but no Main Planet profile exists.")
    from PIL import Image, ImageDraw
    import numpy as np
    profile = system.main_planet_profile
    terrain = profile.terrain
    out_w, out_h, stride = _diagnostic_downsample_dimensions(terrain.width, terrain.height)
    if getattr(terrain, "terrain_island_origin_class", None) is not None:
        origin = np.asarray(terrain.terrain_island_origin_class, dtype=np.uint8)[::stride, ::stride]
        colors = {
            0: (35, 82, 135),
            1: (132, 146, 105),
            2: (230, 185, 74),
            3: (225, 92, 58),
            4: (164, 106, 190),
            5: (242, 238, 190),
        }
        rgb = np.zeros((origin.shape[0], origin.shape[1], 3), dtype=np.uint8)
        for code, color in colors.items():
            rgb[origin == code] = color
        _save_clean_rgb_map(rgb, output_path, title="Island and archipelago diagnostic", description="Stage 3C.4 island-origin classes: shelf islands, volcanic/arc islands, microcontinents/terranes, hotspot/high islands, and large landmasses.", items=[("water/non-island", (35, 82, 135)), ("continent / large land", (132, 146, 105)), ("shelf island", (230, 185, 74)), ("volcanic/arc island", (225, 92, 58)), ("microcontinent/terrane", (164, 106, 190)), ("hotspot/high island", (242, 238, 190))], scale=_world_scale_meta(profile, out_w, out_h, kind="diagnostic_downsample", stride=stride))
        return
    land = np.asarray(terrain.is_land, dtype=bool)[::stride, ::stride]
    rgb = np.zeros((out_h, out_w, 3), dtype=np.uint8)
    rgb[:, :, :] = (35, 88, 140)
    rgb[land] = (115, 130, 94)
    image = Image.fromarray(rgb, mode="RGB")
    draw = ImageDraw.Draw(image)
    step_x = max(80, out_w // 24)
    step_y = max(60, out_h // 14)
    scale_len = min(step_x, step_y) * 0.45
    for y in range(step_y // 2, out_h, step_y):
        lat = 90.0 - (y + 0.5) * 180.0 / out_h
        if abs(lat) < 5:
            continue
        for x in range(step_x // 2, out_w, step_x):
            if land[y, x]:
                continue
            hemi = 1 if lat >= 0 else -1
            gyre = 1 if abs(lat) < 45 else -1
            dc = gyre * (1 if (x / out_w) < 0.5 else -1)
            dr = -hemi * 0.45 * gyre
            color = (255, 170, 70) if dc > 0 else (90, 185, 255)
            _draw_arrow(draw, x, y, x + dc * scale_len, y + dr * scale_len, fill=color, width=2)
    output_path = _prepare_output_path(output_path)
    _save_image_fast(image, output_path)
    _write_map_legend_sidecar(output_path, title="Ocean-current diagnostic", description="Simplified gyres and warm/cold current influence over land/ocean mask.", items=[("ocean", (35, 88, 140)), ("land", (115, 130, 94)), ("warm/current arrow", (255, 170, 70)), ("cold/current arrow", (90, 185, 255))], scale=_world_scale_meta(profile, out_w, out_h, kind="diagnostic_downsample", stride=stride))


def save_main_planet_moisture_transport_view(system: StarSystem, output_path: str | Path) -> None:
    if system.main_planet_profile is None:
        raise RuntimeError("Moisture diagnostic requested, but no Main Planet profile exists.")
    from PIL import Image, ImageDraw
    import numpy as np
    profile = system.main_planet_profile
    terrain = profile.terrain
    climate = profile.climate
    out_w, out_h, stride = _diagnostic_downsample_dimensions(terrain.width, terrain.height)
    precip = np.asarray(climate.annual_precip_mm, dtype=np.float32)[::stride, ::stride]
    land = np.asarray(terrain.is_land, dtype=bool)[::stride, ::stride]
    t = np.clip(np.log1p(precip) / max(1e-6, np.log1p(max(1, float(np.nanmax(precip))))), 0.0, 1.0)
    rgb = np.zeros((out_h, out_w, 3), dtype=np.uint8)
    rgb[:, :, :] = (52, 93, 130)
    rgb[land, 0] = (235 - 150 * t[land]).astype(np.uint8)
    rgb[land, 1] = (214 - 60 * t[land]).astype(np.uint8)
    rgb[land, 2] = (126 + 92 * t[land]).astype(np.uint8)
    image = Image.fromarray(rgb, mode="RGB")
    draw = ImageDraw.Draw(image)
    step_x = max(90, out_w // 22)
    step_y = max(68, out_h // 12)
    scale_len = min(step_x, step_y) * 0.36
    for y in range(step_y // 2, out_h, step_y):
        lat = 90.0 - (y + 0.5) * 180.0 / out_h
        dr, dc = _diagnostic_wind_vector(lat)
        for x in range(step_x // 2, out_w, step_x):
            moisture = t[y, x]
            color = (210, 240, 255) if moisture > 0.55 else (245, 230, 160)
            _draw_arrow(draw, x, y, x + dc * scale_len, y + dr * scale_len, fill=color, width=2)
    output_path = _prepare_output_path(output_path)
    _save_image_fast(image, output_path)
    _write_map_legend_sidecar(output_path, title="Moisture transport diagnostic", description="Land precipitation field with prevailing transport arrows.", items=[("ocean/background", (52, 93, 130)), ("dry land", (235, 214, 126)), ("wet land", (85, 154, 218)), ("dry transport arrow", (245, 230, 160)), ("moist transport arrow", (210, 240, 255))], scale=_world_scale_meta(profile, out_w, out_h, kind="diagnostic_downsample", stride=stride))


def save_main_planet_rain_shadow_view(system: StarSystem, output_path: str | Path) -> None:
    if system.main_planet_profile is None:
        raise RuntimeError("Rain-shadow diagnostic requested, but no Main Planet profile exists.")
    import numpy as np
    profile = system.main_planet_profile
    terrain = profile.terrain
    climate = profile.climate
    out_w, out_h, stride = _diagnostic_downsample_dimensions(terrain.width, terrain.height)
    elev = np.asarray(terrain.elevation_m, dtype=np.float32)[::stride, ::stride]
    land = np.asarray(terrain.is_land, dtype=bool)[::stride, ::stride]
    precip = np.asarray(climate.annual_precip_mm, dtype=np.float32)[::stride, ::stride]
    gy, gx = np.gradient(elev)
    relief = np.sqrt(gx * gx + gy * gy)
    local_p = precip
    local_mean = (local_p + np.roll(local_p, 8, axis=1) + np.roll(local_p, -8, axis=1) + np.roll(local_p, 8, axis=0) + np.roll(local_p, -8, axis=0)) / 5.0
    shadow = np.clip((local_mean - local_p) / 650.0, 0.0, 1.0) * np.clip(relief / max(1.0, np.quantile(relief[land], 0.98) if land.any() else 1.0), 0.0, 1.0)
    windward = np.clip((local_p - local_mean) / 650.0, 0.0, 1.0) * np.clip(relief / max(1.0, np.quantile(relief[land], 0.98) if land.any() else 1.0), 0.0, 1.0)
    rgb = np.zeros((out_h, out_w, 3), dtype=np.uint8)
    rgb[:, :, :] = (58, 92, 120)
    base = land
    rgb[base] = (140, 142, 116)
    sh = base & (shadow > 0.08)
    wi = base & (windward > 0.08)
    rgb[sh, 0] = (190 + 55 * shadow[sh]).astype(np.uint8)
    rgb[sh, 1] = (155 - 70 * shadow[sh]).astype(np.uint8)
    rgb[sh, 2] = (80 - 35 * shadow[sh]).astype(np.uint8)
    rgb[wi, 0] = (75 - 20 * windward[wi]).astype(np.uint8)
    rgb[wi, 1] = (150 + 70 * windward[wi]).astype(np.uint8)
    rgb[wi, 2] = (120 + 90 * windward[wi]).astype(np.uint8)
    _save_clean_rgb_map(rgb, output_path, title="Rain-shadow diagnostic", description="Green areas mark wetter windward relief; brown areas mark dry leeward relief candidates.", items=[("ocean/background", (58, 92, 120)), ("neutral land", (140, 142, 116)), ("windward/wet relief", (55, 220, 210)), ("leeward/dry relief", (245, 85, 45))], scale=_world_scale_meta(profile, out_w, out_h, kind="diagnostic_downsample", stride=stride))


def save_main_planet_terrain_provinces_view(system: StarSystem, output_path: str | Path) -> None:
    if system.main_planet_profile is None:
        raise RuntimeError("Terrain-province diagnostic requested, but no Main Planet profile exists.")
    import numpy as np
    profile = system.main_planet_profile
    terrain = profile.terrain
    hydrology = profile.hydrology
    out_w, out_h, stride = _diagnostic_downsample_dimensions(terrain.width, terrain.height)
    elev = np.asarray(terrain.elevation_m, dtype=np.float32)[::stride, ::stride]
    land = np.asarray(terrain.is_land, dtype=bool)[::stride, ::stride]
    rivers = np.asarray(hydrology.river_intensity, dtype=np.uint8)[::stride, ::stride]
    gy, gx = np.gradient(elev)
    slope = np.sqrt(gx * gx + gy * gy)
    slope_q = np.quantile(slope[land], 0.88) if land.any() else 1.0
    rgb = np.zeros((out_h, out_w, 3), dtype=np.uint8)
    rgb[:, :, :] = (45, 86, 128)
    low = land & (elev < 220)
    basin = land & (elev >= 220) & (elev < 650) & (slope < slope_q * 0.35)
    shield = land & (elev >= 650) & (elev < 1450) & (slope < slope_q * 0.55)
    plateau = land & (elev >= 1450) & (slope < slope_q * 0.70)
    rugged = land & (slope >= slope_q * 0.75) & (elev >= 600)
    mountain = land & ((elev >= 1850) | (slope >= slope_q * 1.15))
    alluvial = land & (rivers > 0) & (elev < 520)
    rgb[low] = (152, 176, 112)
    rgb[basin] = (188, 176, 124)
    rgb[shield] = (154, 143, 101)
    rgb[plateau] = (180, 132, 92)
    rgb[rugged] = (129, 116, 96)
    rgb[mountain] = (220, 220, 188)
    rgb[alluvial] = (98, 174, 122)
    _save_clean_rgb_map(rgb, output_path, title="Terrain provinces diagnostic", description="Broad terrain province classes derived from elevation, slope, and river/alluvial context.", items=[("ocean", (45, 86, 128)), ("coastal plain/lowland", (152, 176, 112)), ("basin/plain", (188, 176, 124)), ("shield/upland", (154, 143, 101)), ("plateau", (180, 132, 92)), ("rugged highland", (129, 116, 96)), ("mountain", (220, 220, 188)), ("alluvial river plain", (98, 174, 122))], scale=_world_scale_meta(profile, out_w, out_h, kind="diagnostic_downsample", stride=stride))


def save_main_planet_erosion_deposition_view(system: StarSystem, output_path: str | Path) -> None:
    if system.main_planet_profile is None:
        raise RuntimeError("Erosion/deposition diagnostic requested, but no Main Planet profile exists.")
    import numpy as np
    profile = system.main_planet_profile
    terrain = profile.terrain
    hydrology = profile.hydrology
    out_w, out_h, stride = _diagnostic_downsample_dimensions(terrain.width, terrain.height)
    elev = np.asarray(terrain.elevation_m, dtype=np.float32)[::stride, ::stride]
    land = np.asarray(terrain.is_land, dtype=bool)[::stride, ::stride]
    rivers = np.asarray(hydrology.river_intensity, dtype=np.float32)[::stride, ::stride]
    flow = np.asarray(hydrology.flow_accumulation, dtype=np.float32)[::stride, ::stride]
    gy, gx = np.gradient(elev)
    slope = np.sqrt(gx * gx + gy * gy)
    slope_norm = np.clip(slope / max(1.0, np.quantile(slope[land], 0.985) if land.any() else 1.0), 0.0, 1.0)
    flow_norm = np.clip(np.log1p(flow) / max(1e-6, np.log1p(max(1.0, float(np.nanmax(flow))))), 0.0, 1.0)
    erosion = land & (slope_norm > 0.22) & (flow_norm > 0.28)
    deposition = land & (flow_norm > 0.36) & (slope_norm < 0.35) & (elev < 700)
    deltas = land & (rivers > 170) & (elev < 45)
    rgb = np.zeros((out_h, out_w, 3), dtype=np.uint8)
    rgb[:, :, :] = (48, 83, 115)
    rgb[land] = (132, 132, 120)
    rgb[erosion] = (190, 72, 48)
    rgb[deposition] = (86, 164, 86)
    rgb[deltas] = (55, 205, 155)
    _save_clean_rgb_map(rgb, output_path, title="Erosion/deposition diagnostic", description="Likely incision/erosion, floodplain/deposition, and low river-mouth/delta land candidates.", items=[("ocean/background", (48, 83, 115)), ("neutral land", (132, 132, 120)), ("likely incision/erosion", (190, 72, 48)), ("floodplain/deposition", (86, 164, 86)), ("low river-mouth/delta land", (55, 205, 155))], scale=_world_scale_meta(profile, out_w, out_h, kind="diagnostic_downsample", stride=stride))


def save_main_planet_delta_mouths_view(system: StarSystem, output_path: str | Path) -> None:
    if system.main_planet_profile is None:
        raise RuntimeError("Delta/mouth diagnostic requested, but no Main Planet profile exists.")
    import numpy as np
    profile = system.main_planet_profile
    terrain = profile.terrain
    hydrology = profile.hydrology
    out_w, out_h, stride = _diagnostic_downsample_dimensions(terrain.width, terrain.height)
    if getattr(terrain, "terrain_coast_style_class", None) is not None:
        style = np.asarray(terrain.terrain_coast_style_class, dtype=np.uint8)[::stride, ::stride]
        colors = {
            0: (34, 78, 128),
            1: (112, 184, 128),
            2: (205, 84, 62),
            3: (78, 130, 212),
            4: (232, 108, 54),
            5: (88, 205, 170),
            6: (235, 220, 112),
        }
        rgb = np.zeros((style.shape[0], style.shape[1], 3), dtype=np.uint8)
        for code, color in colors.items():
            rgb[style == code] = color
        land_hint = np.asarray(terrain.is_land, dtype=bool)[::stride, ::stride]
        rgb[(style == 0) & land_hint] = (132, 146, 105)
        _save_clean_rgb_map(rgb, output_path, title="Coastline / margin-type diagnostic", description="Coast style classes: passive shelf/coastal plains, rugged active/fjorded margins, rifted gulfs, volcanic arc coasts, true deltaic plains with sediment support, and mixed irregular coasts.", items=[("ocean/background", (34, 78, 128)), ("passive smooth coastal plain", (112, 184, 128)), ("rugged active/fjorded margin", (205, 84, 62)), ("rifted gulf margin", (78, 130, 212)), ("volcanic arc coast", (232, 108, 54)), ("true deltaic plain / sediment coast", (88, 205, 170)), ("mixed irregular coast", (235, 220, 112))], scale=_world_scale_meta(profile, out_w, out_h, kind="diagnostic_downsample", stride=stride))
        return
    land = np.asarray(terrain.is_land, dtype=bool)[::stride, ::stride]
    elev = np.asarray(terrain.elevation_m, dtype=np.int32)[::stride, ::stride]
    rivers = np.asarray(hydrology.river_intensity, dtype=np.uint8)[::stride, ::stride]
    rgb = np.zeros((out_h, out_w, 3), dtype=np.uint8)
    rgb[:, :, :] = (44, 82, 130)
    rgb[land] = (128, 151, 105)
    lowland = land & (elev < 120)
    rgb[lowland] = (170, 184, 118)
    river_mask = land & (rivers > 0)
    rgb[river_mask] = (40, 115, 210)
    estuary = np.zeros((out_h, out_w), dtype=bool)
    delta = np.zeros((out_h, out_w), dtype=bool)
    for r in range(out_h):
        for c in range(out_w):
            if not land[r, c] or rivers[r, c] < 120:
                continue
            touches_ocean = False
            shallow = -99999
            for dr in (-1, 0, 1):
                rr = r + dr
                if rr < 0 or rr >= out_h:
                    continue
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    cc = (c + dc) % out_w
                    if not land[rr, cc]:
                        touches_ocean = True
                        shallow = max(shallow, int(elev[rr, cc]))
            if not touches_ocean:
                continue
            if shallow > -180 and rivers[r, c] >= 170:
                delta[r, c] = True
            else:
                estuary[r, c] = True
    for mask, color in ((estuary, (235, 225, 90)), (delta, (40, 220, 160))):
        ys, xs = np.where(mask)
        for r, c in zip(ys, xs):
            radius = 1 if out_w < 1800 else 2
            for dr in range(-radius, radius + 1):
                rr = r + dr
                if rr < 0 or rr >= out_h:
                    continue
                for dc in range(-radius, radius + 1):
                    cc = (c + dc) % out_w
                    rgb[rr, cc] = color
    _save_clean_rgb_map(rgb, output_path, title="River mouth and delta diagnostic", description="Rivers, estuary-mouth markers, likely delta/deposition mouths, and low coastal plains.", items=[("ocean", (44, 82, 130)), ("land", (128, 151, 105)), ("low coastal plain", (170, 184, 118)), ("river", (40, 115, 210)), ("estuary mouth", (235, 225, 90)), ("likely delta/deposition mouth", (40, 220, 160))], scale=_world_scale_meta(profile, out_w, out_h, kind="diagnostic_downsample", stride=stride))


def save_main_planet_tectonic_plates_view(system: StarSystem, output_path: str | Path) -> None:
    if system.main_planet_profile is None:
        raise RuntimeError("Plate diagnostic requested, but no Main Planet profile exists.")
    from PIL import Image
    import numpy as np
    profile = system.main_planet_profile
    terrain = profile.terrain
    if terrain.tectonic_plate_id is None:
        raise RuntimeError("Terrain has no tectonic plate diagnostic grid.")
    plates = np.asarray(terrain.tectonic_plate_id, dtype=np.int32)
    h, w = plates.shape
    rng = np.random.default_rng(14321)
    max_plate = max(int(plates.max()), 0)
    colors = rng.integers(45, 235, size=(max_plate + 1, 3), dtype=np.uint8)
    rgb = colors[np.clip(plates, 0, max_plate)]
    image = Image.fromarray(rgb, mode="RGB").resize((max(1024, w * 2), max(512, h * 2)), Image.Resampling.NEAREST)
    output_path = _prepare_output_path(output_path)
    _save_image_fast(image, output_path)
    _write_map_legend_sidecar(output_path, title="Procedural tectonic plate diagnostic", description="Colored cells show generated plate regions; terrain uses boundary interactions.", items=[("random plate colors", (120, 180, 220))], stats={"plate_count": max_plate + 1, "source_width": w, "source_height": h}, scale=_world_scale_meta(profile, image.width, image.height, kind="diagnostic_downsample"))


def save_main_planet_plate_boundaries_view(system: StarSystem, output_path: str | Path) -> None:
    if system.main_planet_profile is None:
        raise RuntimeError("Plate-boundary diagnostic requested, but no Main Planet profile exists.")
    from PIL import Image
    import numpy as np
    profile = system.main_planet_profile
    terrain = profile.terrain
    if terrain.tectonic_boundary_class is None and terrain.tectonic_plate_id is None:
        raise RuntimeError("Terrain has no tectonic boundary diagnostic grid.")
    is_v3 = _terrain_is_plate_history_v3(profile)
    if is_v3:
        land = np.asarray(terrain.is_land, dtype=bool)
        plates = np.asarray(terrain.tectonic_plate_id, dtype=np.int32) if terrain.tectonic_plate_id is not None else None
        if plates is not None and plates.shape == land.shape:
            final_mask = np.zeros_like(land, dtype=bool)
            final_mask |= plates != np.roll(plates, 1, axis=1)
            final_mask |= plates != np.roll(plates, -1, axis=1)
            final_mask[:-1, :] |= plates[:-1, :] != plates[1:, :]
            final_mask[1:, :] |= plates[1:, :] != plates[:-1, :]
            conv = _field01_from_optional(getattr(terrain, "plate_tectonic_convergence_x1000", None), land.shape)
            div = _field01_from_optional(getattr(terrain, "plate_tectonic_divergence_x1000", None), land.shape)
            trans = _field01_from_optional(getattr(terrain, "plate_tectonic_transform_x1000", None), land.shape)
            trench = _field01_from_optional(getattr(terrain, "plate_tectonic_trench_x1000", None), land.shape)
            active = _field01_from_optional(getattr(terrain, "plate_tectonic_active_margin_x1000", None), land.shape)
            volcanism = _field01_from_optional(getattr(terrain, "plate_tectonic_volcanic_arc_x1000", None), land.shape)
            dominance = np.argmax(np.stack([conv, div, trans, trench + active, volcanism], axis=0), axis=0) + 1
            b = np.zeros_like(plates, dtype=np.uint8)
            b[final_mask] = dominance[final_mask].astype(np.uint8)
        else:
            b = np.asarray(terrain.tectonic_boundary_class, dtype=np.uint8)
            if land.shape != b.shape:
                land = np.resize(land, b.shape).astype(bool)
        rgb = np.zeros((b.shape[0], b.shape[1], 3), dtype=np.uint8)
        rgb[~land] = (32, 72, 116)
        rgb[land] = (128, 140, 102)
        colors = {
            1: (205, 62, 50),   # compression / collision
            2: (70, 126, 230),  # extension / rift
            3: (148, 80, 178),  # transform / shear
            4: (105, 64, 150),  # active subduction / trench
            5: (236, 128, 58),  # volcanic active boundary
        }
        for code, color in colors.items():
            rgb[b == code] = color
        image = Image.fromarray(rgb, mode="RGB")
        output_path = _prepare_output_path(output_path)
        _save_image_fast(image, output_path)
        _write_map_legend_sidecar(
            output_path,
            title="Final v3 plate boundaries",
            description="Final active plate-boundary positions only, derived from the final plate-ID raster when available. Historical boundary crossings are separated into boundary-history density, orogeny-history, and suture-history maps.",
            items=[("ocean/background", (32, 72, 116)), ("land/background", (128, 140, 102)), ("convergent/collision", colors[1]), ("divergent/rift", colors[2]), ("transform/shear", colors[3]), ("subduction/trench", colors[4]), ("volcanic active boundary", colors[5])],
            stats={"source_width": b.shape[1], "source_height": b.shape[0], "active_boundary_cells": int(np.count_nonzero(b))},
            scale=_world_scale_meta(profile, image.width, image.height, kind="diagnostic_downsample"),
        )
        return
    b = np.asarray(terrain.tectonic_boundary_class, dtype=np.uint8)
    rgb = np.zeros((b.shape[0], b.shape[1], 3), dtype=np.uint8)
    rgb[:, :, :] = (225, 225, 215)
    rgb[b == 1] = (165, 55, 45)
    rgb[b == 2] = (55, 105, 190)
    rgb[b == 3] = (128, 70, 165)
    image = Image.fromarray(rgb, mode="RGB").resize((max(1024, b.shape[1] * 2), max(512, b.shape[0] * 2)), Image.Resampling.NEAREST)
    output_path = _prepare_output_path(output_path)
    _save_image_fast(image, output_path)
    _write_map_legend_sidecar(output_path, title="Plate boundary diagnostic", description="Procedural convergent, divergent, and transform boundary classes.", items=[("intraplate", (225, 225, 215)), ("convergent/uplift", (165, 55, 45)), ("divergent/rift", (55, 105, 190)), ("transform/shear", (128, 70, 165))], stats={"source_width": b.shape[1], "source_height": b.shape[0]}, scale=_world_scale_meta(profile, image.width, image.height, kind="diagnostic_downsample"))

def save_main_planet_final_plate_boundaries_view(system: StarSystem, output_path: str | Path) -> None:
    save_main_planet_plate_boundaries_view(system, output_path)


def save_main_planet_boundary_history_density_view(system: StarSystem, output_path: str | Path) -> None:
    _save_v3_heat_field(
        system,
        output_path,
        "tectonic_boundary_strength_x1000",
        title="Boundary history density",
        description="Accumulated v3 boundary/deformation signal. Unlike final plate boundaries, this intentionally shows where boundary-related deformation passed through over time.",
        hot_color=(230, 78, 62),
        label="high historical boundary density",
    )


def save_main_planet_orogeny_history_view(system: StarSystem, output_path: str | Path) -> None:
    _save_v3_heat_field(
        system,
        output_path,
        "plate_tectonic_orogeny_strength_x1000",
        title="Orogeny history",
        description="Accumulated mountain-building/uplift signal derived from v3 continuous compression, uplift, volcanic, and plateau fields.",
        hot_color=(238, 92, 64),
        label="strong orogeny/uplift history",
    )


def save_main_planet_suture_history_view(system: StarSystem, output_path: str | Path) -> None:
    _save_v3_heat_field(
        system,
        output_path,
        "plate_tectonic_accreted_terrane_x1000",
        title="Suture history",
        description="Old/inactive collision and accretion scars. This is separated from final active boundaries so ancient sutures do not clutter the main boundary map.",
        hot_color=(184, 104, 208),
        cool_color=(36, 58, 82),
        label="old suture/accretion signal",
    )


def save_main_planet_lake_depth_limit_view(system: StarSystem, output_path: str | Path) -> None:
    _save_v3_heat_field(
        system,
        output_path,
        "terrain_lake_depth_limit_x1000",
        title="v3 enclosed-water depth limit",
        description="Update 27D diagnostic: local allowed lake/inland-sea depth after basin-size, rift/subsidence support, sediment fill, distance from basin edge, and coastal-lagoon cleanup. Brighter centers are allowed to stay deeper; dark/fill areas were shallow or converted to coastal lowland.",
        hot_color=(236, 196, 92),
        cool_color=(28, 58, 104),
        label="deeper allowed lake/inland-sea floor",
    )


def save_main_planet_ripple_artifact_risk_view(system: StarSystem, output_path: str | Path) -> None:
    _save_v3_heat_field(
        system,
        output_path,
        "terrain_ripple_artifact_risk_x1000",
        title="v3 ripple artifact risk",
        description="Update 27D diagnostic: old deep-ocean cells where plate-history boundary accumulation and texture are most likely to print as visible ripples. The terrain pass damps this field while preserving ridges, trenches, seamounts, and supported shelves.",
        hot_color=(232, 94, 74),
        cool_color=(30, 62, 104),
        label="higher ripple artifact risk",
    )


def save_main_planet_final_plate_components_view(system: StarSystem, output_path: str | Path) -> None:
    if system.main_planet_profile is None:
        raise RuntimeError("Final plate component diagnostic requested, but no Main Planet profile exists.")
    import numpy as np
    profile = system.main_planet_profile
    terrain = profile.terrain
    if getattr(terrain, "terrain_final_plate_component_class", None) is None:
        raise RuntimeError("Terrain has no final plate component cleanup grid.")
    out_w, out_h, stride = _diagnostic_downsample_dimensions(terrain.width, terrain.height)
    cls = np.asarray(terrain.terrain_final_plate_component_class, dtype=np.uint8)[::stride, ::stride]
    if getattr(terrain, "tectonic_plate_id", None) is not None:
        plate = np.asarray(terrain.tectonic_plate_id, dtype=np.int32)[::stride, ::stride]
    else:
        plate = np.zeros_like(cls, dtype=np.int32)
    base = np.zeros((cls.shape[0], cls.shape[1], 3), dtype=np.uint8)
    # Muted deterministic plate-color background, with cleanup classes overlaid.
    base[..., 0] = ((plate * 37 + 70) % 120 + 45).astype(np.uint8)
    base[..., 1] = ((plate * 53 + 95) % 120 + 45).astype(np.uint8)
    base[..., 2] = ((plate * 71 + 125) % 120 + 45).astype(np.uint8)
    base[cls == 1] = np.array([236, 184, 72], dtype=np.uint8)   # reassigned fragment
    base[cls == 2] = np.array([226, 86, 74], dtype=np.uint8)    # promoted microplate
    _save_clean_rgb_map(
        base,
        output_path,
        title="v3 final plate component cleanup",
        description="Final diagnostic plate IDs after x-wrapped contiguity cleanup. Small disconnected fragments are reassigned to neighboring plates; large fragments are promoted to microplates so final plate diagnostics do not show physically impossible non-contiguous plates.",
        items=[
            ("ordinary contiguous final plate", (92, 128, 146)),
            ("small disconnected fragment reassigned", (236, 184, 72)),
            ("large disconnected fragment promoted to microplate", (226, 86, 74)),
        ],
        stats={
            "source_width": int(cls.shape[1]),
            "source_height": int(cls.shape[0]),
            "reassigned_cells": int(np.count_nonzero(cls == 1)),
            "promoted_microplate_cells": int(np.count_nonzero(cls == 2)),
        },
        scale=_world_scale_meta(profile, out_w, out_h, kind="diagnostic_downsample", stride=stride),
    )



def save_main_planet_v4_control_response_view(system: StarSystem, output_path: str | Path) -> None:
    if system.main_planet_profile is None:
        raise RuntimeError("v4 control-response diagnostic requested, but no Main Planet profile exists.")
    import numpy as np
    profile = system.main_planet_profile
    terrain = profile.terrain
    if getattr(terrain, "terrain_v4_control_response_class", None) is None:
        raise RuntimeError("Terrain has no v4 control-response class grid.")
    out_w, out_h, stride = _diagnostic_downsample_dimensions(terrain.width, terrain.height)
    cls = np.asarray(terrain.terrain_v4_control_response_class, dtype=np.uint8)[::stride, ::stride]
    palette = {
        0: (30, 48, 64),
        1: (232, 174, 78),
        2: (244, 126, 72),
        3: (74, 184, 220),
        4: (194, 124, 220),
        5: (250, 230, 106),
    }
    rgb = np.zeros((cls.shape[0], cls.shape[1], 3), dtype=np.uint8)
    for code, color in palette.items():
        rgb[cls == code] = color
    _save_clean_rgb_map(
        rgb,
        output_path,
        title="v4 control-response classes",
        description="Update 32 diagnostic for plate_history_v4: shows which user-facing v4 control should visibly affect each cell. If topology/island/rift strength changes do not alter this map or related terrain, the control path is broken.",
        items=[
            ("weak/no v4 control response", palette[0]),
            ("topology / boundary / mountain-branch response", palette[1]),
            ("volcanic island-chain response", palette[2]),
            ("rift-cut / gulf / basin response", palette[3]),
            ("mixed two-control response", palette[4]),
            ("mixed topology + island + rift response", palette[5]),
        ],
        stats={
            "source_width": int(cls.shape[1]),
            "source_height": int(cls.shape[0]),
            "active_response_cells": int(np.count_nonzero(cls > 0)),
            "topology_response_cells": int(np.count_nonzero(cls == 1)),
            "island_response_cells": int(np.count_nonzero(cls == 2)),
            "rift_response_cells": int(np.count_nonzero(cls == 3)),
            "mixed_response_cells": int(np.count_nonzero(cls >= 4)),
        },
        scale=_world_scale_meta(profile, out_w, out_h, kind="diagnostic_downsample", stride=stride),
    )

def save_main_planet_crust_type_view(system: StarSystem, output_path: str | Path) -> None:
    if system.main_planet_profile is None:
        raise RuntimeError("Crust diagnostic requested, but no Main Planet profile exists.")
    from PIL import Image
    import numpy as np
    profile = system.main_planet_profile
    terrain = profile.terrain
    if terrain.crust_type is None:
        raise RuntimeError("Terrain has no crust diagnostic grid.")
    c = np.asarray(terrain.crust_type, dtype=np.uint8)
    is_v3 = _terrain_is_plate_history_v3(profile) or int(c.max(initial=0)) > 8
    if is_v3:
        palette = {
            0: (20, 46, 82),
            1: (18, 36, 78),     # abyssal/generic oceanic
            2: (84, 196, 232),    # young ridge zone
            3: (8, 24, 58),       # old oceanic
            4: (92, 52, 138),     # trench/subduction trough
            5: (118, 108, 202),   # fracture/transform oceanic
            6: (66, 184, 172),    # seamount/oceanic plateau
            7: (116, 222, 212),   # shallow submerged shelf
            8: (122, 106, 68),    # craton/core
            9: (164, 148, 92),    # continental interior
            10: (238, 220, 164),  # orogenic belt
            11: (164, 96, 86),    # old suture
            12: (218, 142, 64),   # rifted continent
            13: (188, 184, 112),  # transitional/passive margin
            14: (204, 174, 116),  # sedimentary/foreland basin
            15: (186, 112, 96),   # accreted terrane/microcontinent
            16: (222, 70, 56),    # continental volcanic arc
            17: (244, 118, 62),   # oceanic island arc
            18: (246, 184, 76),   # hotspot/oceanic island
            19: (64, 142, 196),   # upper continental slope
            20: (38, 98, 158),    # continental rise
            21: (24, 64, 118),    # deep submerged continental margin
        }
        rgb = np.zeros((c.shape[0], c.shape[1], 3), dtype=np.uint8)
        for code, color in palette.items():
            rgb[c == code] = color
        image = Image.fromarray(rgb, mode="RGB")
        output_path = _prepare_output_path(output_path)
        _save_image_fast(image, output_path)
        _write_map_legend_sidecar(
            output_path,
            title="v3 crust circumstance diagnostic",
            description="Richer crust labels derived from final v3 continuous fields and final land/ocean state. Labels summarize dominant causes; they do not select separate terrain equations.",
            items=[
                ("abyssal/generic oceanic crust", palette[1]), ("young oceanic/ridge zone", palette[2]), ("old oceanic crust", palette[3]),
                ("trench/subduction trough", palette[4]), ("fracture/transform oceanic crust", palette[5]), ("seamount/oceanic plateau", palette[6]),
                ("shallow submerged continental shelf", palette[7]), ("upper continental slope", palette[19]), ("continental rise", palette[20]), ("deep submerged continental margin", palette[21]),
                ("continental craton/core", palette[8]), ("continental interior/shield", palette[9]),
                ("young orogenic belt", palette[10]), ("old suture/eroded orogen", palette[11]), ("rifted continental crust", palette[12]),
                ("transitional/passive margin", palette[13]), ("sedimentary/foreland basin", palette[14]), ("accreted terrane/microcontinent", palette[15]),
                ("continental volcanic arc", palette[16]), ("oceanic island arc", palette[17]), ("hotspot/oceanic island", palette[18]),
            ],
            stats={"source_width": c.shape[1], "source_height": c.shape[0], "class_count": int(len(set(np.unique(c).tolist()) - {0}))},
            scale=_world_scale_meta(profile, image.width, image.height, kind="diagnostic_downsample"),
        )
        return
    rgb = np.zeros((c.shape[0], c.shape[1], 3), dtype=np.uint8)
    rgb[c == 0] = (25, 60, 120)
    rgb[c == 1] = (75, 135, 175)
    rgb[c == 2] = (150, 142, 96)
    rgb[c == 3] = (215, 205, 170)
    rgb[c == 4] = (190, 165, 105)
    rgb[c == 5] = (196, 150, 96)
    rgb[c == 6] = (210, 92, 72)
    rgb[c == 7] = (225, 145, 74)
    rgb[c == 8] = (108, 155, 185)
    image = Image.fromarray(rgb, mode="RGB").resize((max(1024, c.shape[1] * 2), max(512, c.shape[0] * 2)), Image.Resampling.NEAREST)
    output_path = _prepare_output_path(output_path)
    _save_image_fast(image, output_path)
    _write_map_legend_sidecar(output_path, title="Crust type diagnostic", description="Plate Terrain 15 crust classes: oceanic crust, shelves, continental interiors, active margins, rifted margins, microcontinents, volcanic arcs, hotspot chains, and oceanic plateaus.", items=[("abyssal/oceanic crust", (25, 60, 120)), ("submerged continental shelf", (75, 135, 175)), ("continental core/interior", (150, 142, 96)), ("active/orogenic continent", (215, 205, 170)), ("rifted continental margin", (190, 165, 105)), ("microcontinent/fragment", (196, 150, 96)), ("volcanic island arc", (210, 92, 72)), ("hotspot/oceanic island", (225, 145, 74)), ("oceanic plateau", (108, 155, 185))], stats={"source_width": c.shape[1], "source_height": c.shape[0]}, scale=_world_scale_meta(profile, image.width, image.height, kind="diagnostic_downsample"))


def save_main_planet_coastline_margin_types_view(system: StarSystem, output_path: str | Path) -> None:
    if system.main_planet_profile is None:
        raise RuntimeError("Coastline margin diagnostic requested, but no Main Planet profile exists.")
    import numpy as np
    profile = system.main_planet_profile
    terrain = profile.terrain
    hydrology = profile.hydrology
    out_w, out_h, stride = _diagnostic_downsample_dimensions(terrain.width, terrain.height)
    if getattr(terrain, "terrain_coast_style_class", None) is not None:
        style = np.asarray(terrain.terrain_coast_style_class, dtype=np.uint8)[::stride, ::stride]
        colors = {
            0: (34, 78, 128),
            1: (112, 184, 128),
            2: (205, 84, 62),
            3: (78, 130, 212),
            4: (232, 108, 54),
            5: (88, 205, 170),
            6: (235, 220, 112),
        }
        rgb = np.zeros((style.shape[0], style.shape[1], 3), dtype=np.uint8)
        for code, color in colors.items():
            rgb[style == code] = color
        land_hint = np.asarray(terrain.is_land, dtype=bool)[::stride, ::stride]
        rgb[(style == 0) & land_hint] = (132, 146, 105)
        _save_clean_rgb_map(rgb, output_path, title="Coastline / margin-type diagnostic", description="Coast style classes: passive shelf/coastal plains, rugged active/fjorded margins, rifted gulfs, volcanic arc coasts, true deltaic plains with sediment support, and mixed irregular coasts.", items=[("ocean/background", (34, 78, 128)), ("passive smooth coastal plain", (112, 184, 128)), ("rugged active/fjorded margin", (205, 84, 62)), ("rifted gulf margin", (78, 130, 212)), ("volcanic arc coast", (232, 108, 54)), ("true deltaic plain / sediment coast", (88, 205, 170)), ("mixed irregular coast", (235, 220, 112))], scale=_world_scale_meta(profile, out_w, out_h, kind="diagnostic_downsample", stride=stride))
        return
    land = np.asarray(terrain.is_land, dtype=bool)[::stride, ::stride]
    elev = np.asarray(terrain.elevation_m, dtype=np.int32)[::stride, ::stride]
    rivers = np.asarray(hydrology.river_intensity, dtype=np.uint8)[::stride, ::stride]
    rgb = np.zeros((out_h, out_w, 3), dtype=np.uint8)
    rgb[:] = (30, 70, 120)
    rgb[land] = (132, 146, 105)
    north = np.vstack((land[0:1, :], land[:-1, :]))
    south = np.vstack((land[1:, :], land[-1:, :]))
    west = np.roll(land, 1, axis=1)
    east = np.roll(land, -1, axis=1)
    coast = land & ((~north) | (~south) | (~west) | (~east))
    elev_n = np.vstack((elev[0:1, :], elev[:-1, :]))
    elev_s = np.vstack((elev[1:, :], elev[-1:, :]))
    elev_w = np.roll(elev, 1, axis=1)
    elev_e = np.roll(elev, -1, axis=1)
    relief = np.maximum.reduce([abs(elev - elev_n), abs(elev - elev_s), abs(elev - elev_w), abs(elev - elev_e)])
    low_delta = coast & (elev < 90) & (rivers > 125)
    rugged = coast & (relief > 220)
    shelf_plain = coast & (elev < 95) & (~low_delta)
    moderate = coast & (~rugged) & (~shelf_plain) & (~low_delta)
    rgb[moderate] = (240, 230, 120)
    rgb[shelf_plain] = (90, 190, 120)
    rgb[rugged] = (205, 85, 55)
    rgb[low_delta] = (40, 220, 170)
    _save_clean_rgb_map(rgb, output_path, title="Coastline / margin-type diagnostic", description="Broad coastline and margin style classes.", items=[("ocean", (30, 70, 120)), ("land", (132, 146, 105)), ("passive/mixed margin", (240, 230, 120)), ("coastal plain/shelf", (90, 190, 120)), ("rugged active/fjord margin", (205, 85, 55)), ("deltaic/coastal wetland", (40, 220, 170))], scale=_world_scale_meta(profile, out_w, out_h, kind="diagnostic_downsample", stride=stride))


def save_main_planet_inland_lakes_view(system: StarSystem, output_path: str | Path) -> None:
    if system.main_planet_profile is None:
        raise RuntimeError("Inland lakes diagnostic requested, but no Main Planet profile exists.")
    import numpy as np
    from scipy import ndimage
    from collections import deque
    profile = system.main_planet_profile
    terrain = profile.terrain
    hydrology = profile.hydrology
    out_w, out_h, stride = _diagnostic_downsample_dimensions(terrain.width, terrain.height)
    if getattr(terrain, "terrain_coast_style_class", None) is not None:
        style = np.asarray(terrain.terrain_coast_style_class, dtype=np.uint8)[::stride, ::stride]
        colors = {
            0: (34, 78, 128),
            1: (112, 184, 128),
            2: (205, 84, 62),
            3: (78, 130, 212),
            4: (232, 108, 54),
            5: (88, 205, 170),
            6: (235, 220, 112),
        }
        rgb = np.zeros((style.shape[0], style.shape[1], 3), dtype=np.uint8)
        for code, color in colors.items():
            rgb[style == code] = color
        land_hint = np.asarray(terrain.is_land, dtype=bool)[::stride, ::stride]
        rgb[(style == 0) & land_hint] = (132, 146, 105)
        _save_clean_rgb_map(rgb, output_path, title="Coastline / margin-type diagnostic", description="Coast style classes: passive shelf/coastal plains, rugged active/fjorded margins, rifted gulfs, volcanic arc coasts, true deltaic plains with sediment support, and mixed irregular coasts.", items=[("ocean/background", (34, 78, 128)), ("passive smooth coastal plain", (112, 184, 128)), ("rugged active/fjorded margin", (205, 84, 62)), ("rifted gulf margin", (78, 130, 212)), ("volcanic arc coast", (232, 108, 54)), ("true deltaic plain / sediment coast", (88, 205, 170)), ("mixed irregular coast", (235, 220, 112))], scale=_world_scale_meta(profile, out_w, out_h, kind="diagnostic_downsample", stride=stride))
        return
    land = np.asarray(terrain.is_land, dtype=bool)[::stride, ::stride]
    elev = np.asarray(terrain.elevation_m, dtype=np.int32)[::stride, ::stride]
    lake_candidates = np.asarray(hydrology.lake_mask, dtype=bool)[::stride, ::stride]
    water = ~land
    comp = np.full((out_h, out_w), -1, dtype=np.int32)
    sizes: dict[int, int] = {}
    cid = 0
    for r in range(out_h):
        for c in range(out_w):
            if not water[r, c] or comp[r, c] >= 0:
                continue
            q = deque([(r, c)]); comp[r, c] = cid; size = 0
            while q:
                rr, cc = q.popleft(); size += 1
                for dr, dc in ((-1,0),(1,0),(0,-1),(0,1)):
                    nr = rr + dr
                    if nr < 0 or nr >= out_h:
                        continue
                    nc = (cc + dc) % out_w
                    if water[nr, nc] and comp[nr, nc] < 0:
                        comp[nr, nc] = cid; q.append((nr, nc))
            sizes[cid] = size; cid += 1
    ocean_id = max(sizes.items(), key=lambda item: item[1])[0] if sizes else -1
    inland = water & (comp >= 0) & (comp != ocean_id)
    rgb = np.zeros((out_h, out_w, 3), dtype=np.uint8)
    rgb[:] = (30, 75, 125)
    rgb[land] = (145, 150, 100)
    rgb[inland] = (105, 70, 180)
    # Lowland labels are diagnostic-only and topology-aware. "Coastal" now
    # means near the largest world-ocean component with east/west wrap, not merely
    # near the image edge or below 120 m.
    ocean_water = water & (comp == ocean_id)
    ocean_tiled = np.tile(ocean_water, (1, 3))
    dist_to_world_ocean = ndimage.distance_transform_edt(~ocean_tiled)[:, out_w:out_w * 2]
    coastal_low_plain = land & (elev < 120) & (dist_to_world_ocean <= max(2, int(round(4 / max(stride, 1)))))
    inland_low_plain = land & (elev < 120) & (~coastal_low_plain)
    rgb[inland_low_plain] = (196, 178, 104)
    rgb[coastal_low_plain] = (175, 210, 120)
    rgb[lake_candidates] = (50, 205, 235)
    _save_clean_rgb_map(rgb, output_path, title="Inland lakes and enclosed seas diagnostic", description="Distinguishes ocean water from enclosed inland lakes/seas, true coastal low plains, inland low plains, and hydrologic lake/sink candidates using east/west wrap-aware topology.", items=[("world ocean", (30, 75, 125)), ("land", (145, 150, 100)), ("true coastal low plain", (175, 210, 120)), ("inland low plain / basin floor", (196, 178, 104)), ("enclosed water body", (105, 70, 180)), ("hydrologic lake/sink candidate", (50, 205, 235))], stats={"enclosed_water_components": max(0, len(sizes) - 1), "lake_candidate_cells": int(lake_candidates.sum()), "coastal_low_plain_cells": int(coastal_low_plain.sum()), "inland_low_plain_cells": int(inland_low_plain.sum())}, scale=_world_scale_meta(profile, out_w, out_h, kind="diagnostic_downsample", stride=stride))


def save_main_planet_islands_archipelago_view(system: StarSystem, output_path: str | Path) -> None:
    if system.main_planet_profile is None:
        raise RuntimeError("Island diagnostic requested, but no Main Planet profile exists.")
    import numpy as np
    from scipy import ndimage
    profile = system.main_planet_profile
    terrain = profile.terrain
    out_w, out_h, stride = _diagnostic_downsample_dimensions(terrain.width, terrain.height)
    is_v3 = _terrain_is_plate_history_v3(profile)
    if is_v3:
        land = np.asarray(terrain.is_land, dtype=bool)[::stride, ::stride]
        elev = np.asarray(terrain.elevation_m, dtype=np.int32)[::stride, ::stride]
        origin = np.asarray(getattr(terrain, "terrain_island_origin_class", None), dtype=np.int16)[::stride, ::stride] if getattr(terrain, "terrain_island_origin_class", None) is not None else None
        labels, count = _label_xwrap_bool(land)
        sizes = np.bincount(labels.ravel()) if count else np.asarray([0])
        land_cells = max(1, int(land.sum()))
        continent_threshold = max(32, int(land_cells * 0.028))
        rgb = np.zeros((out_h, out_w, 3), dtype=np.uint8)
        rgb[:] = (35, 82, 135)
        rgb[land] = (132, 146, 105)
        component_island_cells = 0
        archipelago_cells = 0
        for lab in range(1, count + 1):
            comp = labels == lab
            size = int(sizes[lab]) if lab < len(sizes) else int(comp.sum())
            if size >= continent_threshold:
                continue
            component_island_cells += size
            code = int(round(float(np.median(origin[comp])))) if origin is not None and np.any(comp) else 2
            if code == 3:
                color = (225, 92, 58)
            elif code == 4:
                color = (164, 106, 190)
            elif code == 5 or (np.any(comp) and float(np.mean(elev[comp])) > 900.0):
                color = (242, 238, 190)
            else:
                color = (230, 185, 74)
            rgb[comp] = color
            if size <= max(4, int(land_cells * 0.0025)):
                archipelago_cells += size
        _save_clean_rgb_map(
            rgb,
            output_path,
            title="Island and archipelago diagnostic",
            description="v3 island map recomputed from the final land mask with east/west wrapping. Component labels are diagnostic only and no longer depend on stale pre-v3 island-origin rasters.",
            items=[("ocean", (35, 82, 135)), ("large landmass / continent", (132, 146, 105)), ("shelf island / small island", (230, 185, 74)), ("volcanic / island arc", (225, 92, 58)), ("microcontinent / terrane", (164, 106, 190)), ("hotspot / high island", (242, 238, 190))],
            stats={"land_component_count": int(count), "continent_threshold_cells": int(continent_threshold), "island_cells": int(component_island_cells), "small_archipelago_cells": int(archipelago_cells)},
            scale=_world_scale_meta(profile, out_w, out_h, kind="diagnostic_downsample", stride=stride),
        )
        return
    if getattr(terrain, "terrain_island_origin_class", None) is not None:
        origin = np.asarray(terrain.terrain_island_origin_class, dtype=np.uint8)[::stride, ::stride]
        colors = {
            0: (35, 82, 135),
            1: (132, 146, 105),
            2: (230, 185, 74),
            3: (225, 92, 58),
            4: (164, 106, 190),
            5: (242, 238, 190),
        }
        rgb = np.zeros((origin.shape[0], origin.shape[1], 3), dtype=np.uint8)
        for code, color in colors.items():
            rgb[origin == code] = color
        _save_clean_rgb_map(rgb, output_path, title="Island and archipelago diagnostic", description="Stage 3C.4 island-origin classes: shelf islands, volcanic/arc islands, microcontinents/terranes, hotspot/high islands, and large landmasses.", items=[("water/non-island", (35, 82, 135)), ("continent / large land", (132, 146, 105)), ("shelf island", (230, 185, 74)), ("volcanic/arc island", (225, 92, 58)), ("microcontinent/terrane", (164, 106, 190)), ("hotspot/high island", (242, 238, 190))], scale=_world_scale_meta(profile, out_w, out_h, kind="diagnostic_downsample", stride=stride))
        return
    land = np.asarray(terrain.is_land, dtype=bool)[::stride, ::stride]
    elev = np.asarray(terrain.elevation_m, dtype=np.int32)[::stride, ::stride]
    rgb = np.zeros((out_h, out_w, 3), dtype=np.uint8)
    rgb[:] = (35, 80, 135)
    rgb[land] = (136, 147, 100)
    world = out_h * out_w
    island_limit = max(4, int(world * 0.0038))
    small_island_limit = max(3, int(world * 0.00065))
    structure = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)
    labels, count = ndimage.label(land, structure=structure)
    if count:
        sizes = np.bincount(labels.ravel())
        island_mask = land & (sizes[labels] <= island_limit)
        small_island_mask = land & (sizes[labels] <= small_island_limit)
    else:
        island_mask = np.zeros_like(land, dtype=bool)
        small_island_mask = np.zeros_like(land, dtype=bool)
    rgb[island_mask] = (230, 180, 65)
    rgb[small_island_mask] = (245, 105, 70)
    rgb[island_mask & (elev > 900)] = (245, 245, 210)
    _save_clean_rgb_map(rgb, output_path, title="Island and archipelago diagnostic", description="Highlights small islands, archipelago chains, and high volcanic/island-arc terrain.", items=[("ocean", (35, 80, 135)), ("large land", (136, 147, 100)), ("small islands / archipelago", (230, 180, 65)), ("very small islands", (245, 105, 70)), ("high volcanic/island-arc terrain", (245, 245, 210))], stats={"land_component_count": int(count), "island_limit_cells": int(island_limit), "small_island_limit_cells": int(small_island_limit)}, scale=_world_scale_meta(profile, out_w, out_h, kind="diagnostic_downsample", stride=stride))


def save_main_planet_terrain_region_maps(system: StarSystem, output_dir: str | Path, rows: int = 8, cols: int = 16) -> None:
    if system.main_planet_profile is None:
        raise RuntimeError("Terrain regional maps requested, but no Main Planet profile exists.")
    from pathlib import Path as _Path
    import numpy as np
    from PIL import Image, ImageDraw

    out = _Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    profile = system.main_planet_profile
    terrain = profile.terrain
    elevation = np.asarray(terrain.elevation_m, dtype=np.float32)
    land = np.asarray(terrain.is_land, dtype=bool)
    height, width = elevation.shape

    gy, gx = np.gradient(elevation)
    slope = np.sqrt(gx * gx + gy * gy)
    coast_full = np.zeros_like(land, dtype=bool)
    if height > 2 and width > 2:
        coast_full[1:-1, 1:-1] = land[1:-1, 1:-1] & ((~land[:-2, 1:-1]) | (~land[2:, 1:-1]) | (~land[1:-1, :-2]) | (~land[1:-1, 2:]))

    def _fit_optional_field(value, *, scale: float = 1.0):
        if value is None:
            return None
        arr = np.asarray(value, dtype=np.float32) * scale
        if arr.size == 0:
            return None
        if arr.shape == elevation.shape:
            return arr
        try:
            from scipy import ndimage
            zy = height / max(1, arr.shape[0]); zx = width / max(1, arr.shape[1])
            return ndimage.zoom(arr, (zy, zx), order=1)[:height, :width]
        except Exception:
            return np.resize(arr, elevation.shape).astype(np.float32)

    mountain_field = _fit_optional_field(getattr(terrain, "terrain_mountain_strength_x1000", None), scale=0.001)
    basin_field = _fit_optional_field(getattr(terrain, "terrain_basin_field_x1000", None), scale=0.001)
    valley_field = _fit_optional_field(getattr(terrain, "terrain_valley_corridor_x1000", None), scale=0.001)
    erosion_field = _fit_optional_field(getattr(terrain, "terrain_erosion_strength_x1000", None), scale=0.001)

    row_edges = [round(i * height / rows) for i in range(rows + 1)]
    col_edges = [round(i * width / cols) for i in range(cols + 1)]
    region_summaries: list[dict[str, object]] = []
    contact_tiles: list[tuple[str, np.ndarray, dict[str, object]]] = []
    index_lines = [
        "filename,row,col,row_start,row_end_exclusive,col_start,col_end_exclusive,land_min_m,land_max_m,water_min_depth_m,water_max_depth_m,land_fraction",
    ]
    for rr in range(rows):
        r0, r1 = row_edges[rr], row_edges[rr + 1]
        for cc in range(cols):
            c0, c1 = col_edges[cc], col_edges[cc + 1]
            elev_crop = elevation[r0:r1, c0:c1]
            land_crop = land[r0:r1, c0:c1]
            ch, cw = elev_crop.shape
            rgb = np.zeros((ch, cw, 3), dtype=np.uint8)
            land_values = elev_crop[land_crop]
            water_values = elev_crop[~land_crop]
            if land_values.size:
                land_min = float(np.min(land_values)); land_max = float(np.max(land_values)); land_span = max(1.0, land_max - land_min)
                land_t = np.clip((elev_crop - land_min) / land_span, 0.0, 1.0)
            else:
                land_min = 0.0; land_max = 0.0; land_t = np.zeros_like(elev_crop, dtype=np.float32)
            if water_values.size:
                depths = np.abs(water_values); water_min = float(np.min(depths)); water_max = float(np.max(depths)); water_span = max(1.0, water_max - water_min)
                water_t = np.clip((np.abs(elev_crop) - water_min) / water_span, 0.0, 1.0)
            else:
                water_min = 0.0; water_max = 0.0; water_t = np.zeros_like(elev_crop, dtype=np.float32)
            rgb[:, :, 0] = (104 - 86 * water_t).astype(np.uint8)
            rgb[:, :, 1] = (168 - 107 * water_t).astype(np.uint8)
            rgb[:, :, 2] = (211 - 89 * water_t).astype(np.uint8)
            low = land_crop & (land_t < 0.22)
            mid = land_crop & (land_t >= 0.22) & (land_t < 0.55)
            high = land_crop & (land_t >= 0.55) & (land_t < 0.82)
            peak = land_crop & (land_t >= 0.82)
            f = np.zeros_like(land_t, dtype=np.float32)
            f[low] = land_t[low] / 0.22
            rgb[low, 0] = (200 + (108 - 200) * f[low]).astype(np.uint8); rgb[low, 1] = (185 + (155 - 185) * f[low]).astype(np.uint8); rgb[low, 2] = (116 + (84 - 116) * f[low]).astype(np.uint8)
            f[mid] = (land_t[mid] - 0.22) / 0.33
            rgb[mid, 0] = (108 + (143 - 108) * f[mid]).astype(np.uint8); rgb[mid, 1] = (155 + (125 - 155) * f[mid]).astype(np.uint8); rgb[mid, 2] = (84 + (82 - 84) * f[mid]).astype(np.uint8)
            f[high] = (land_t[high] - 0.55) / 0.27
            rgb[high, 0] = (143 + (180 - 143) * f[high]).astype(np.uint8); rgb[high, 1] = (125 + (174 - 125) * f[high]).astype(np.uint8); rgb[high, 2] = (82 + (156 - 82) * f[high]).astype(np.uint8)
            f[peak] = (land_t[peak] - 0.82) / 0.18
            rgb[peak, 0] = (180 + (245 - 180) * f[peak]).astype(np.uint8); rgb[peak, 1] = (174 + (245 - 174) * f[peak]).astype(np.uint8); rgb[peak, 2] = (156 + (240 - 156) * f[peak]).astype(np.uint8)
            if ch > 2 and cw > 2:
                coast = np.zeros((ch, cw), dtype=bool)
                coast[1:-1, 1:-1] = land_crop[1:-1, 1:-1] & ((~land_crop[:-2, 1:-1]) | (~land_crop[2:, 1:-1]) | (~land_crop[1:-1, :-2]) | (~land_crop[1:-1, 2:]))
                rgb[coast] = (30, 30, 26)
            filename = f"terrain_region_r{rr + 1:02d}_c{cc + 1:02d}.png"
            region_path = out / filename

            coast_crop = coast_full[r0:r1, c0:c1]
            slope_crop = slope[r0:r1, c0:c1]
            land_fraction = float(np.mean(land_crop))
            mean_land_elev = float(np.mean(land_values)) if land_values.size else 0.0
            max_elev = float(np.max(land_values)) if land_values.size else 0.0
            coast_share = float(np.mean(coast_crop)) if coast_crop.size else 0.0
            slope_land = slope_crop[land_crop]
            slope_proxy = float(np.mean(slope_land)) if slope_land.size else 0.0
            mountain_strength = float(np.mean(mountain_field[r0:r1, c0:c1][land_crop])) if mountain_field is not None and land_values.size else 0.0
            basin_strength = float(np.mean(basin_field[r0:r1, c0:c1][land_crop])) if basin_field is not None and land_values.size else 0.0
            valley_strength = float(np.mean(valley_field[r0:r1, c0:c1][land_crop])) if valley_field is not None and land_values.size else 0.0
            erosion_strength = float(np.mean(erosion_field[r0:r1, c0:c1][land_crop])) if erosion_field is not None and land_values.size else 0.0
            if land_fraction < 0.08:
                character = "open ocean or tiny-island region"
            elif coast_share > 0.035 and mountain_strength > 0.28:
                character = "rugged coastal highlands"
            elif coast_share > 0.035 and land_fraction < 0.55:
                character = "fragmented coastal or island region"
            elif valley_strength > 0.26 and basin_strength > 0.18:
                character = "basin-and-valley terrain"
            elif mountain_strength > 0.32 or max_elev > 1800:
                character = "mountain or highland region"
            elif basin_strength > 0.22 or (land_values.size and mean_land_elev < 350):
                character = "lowland or sedimentary basin region"
            elif land_fraction > 0.82:
                character = "continental interior"
            else:
                character = "mixed terrain region"
            region_stats = {"row": rr + 1, "col": cc + 1, "row_start": r0, "row_end_exclusive": r1, "col_start": c0, "col_end_exclusive": c1, "land_fraction": land_fraction, "ocean_fraction": 1.0 - land_fraction, "land_min_m": land_min, "land_max_m": land_max, "water_min_depth_m": water_min, "water_max_depth_m": water_max, "mean_land_elevation_m": round(mean_land_elev, 1), "max_elevation_m": round(max_elev, 1), "coast_cell_share": round(coast_share, 4), "mean_land_slope_proxy": round(slope_proxy, 2), "mountain_strength": round(mountain_strength, 3), "basin_strength": round(basin_strength, 3), "valley_corridor_strength": round(valley_strength, 3), "erosion_deposition_strength": round(erosion_strength, 3), "terrain_character": character}
            _save_clean_rgb_map(
                rgb,
                region_path,
                title=f"Terrain region r{rr + 1:02d} c{cc + 1:02d}",
                description="Local terrain crop with land elevations and water depths rescaled separately within this region. The PNG contains map pixels only.",
                items=[
                    {"kind": "gradient", "label": "local water depth", "colors": ["#68a8d3", "#123d7a"], "min": water_min, "max": water_max, "unit": "m"},
                    {"kind": "gradient", "label": "local land elevation", "colors": ["#c8b974", "#6c9b54", "#8f7d52", "#f5f5f0"], "min": land_min, "max": land_max, "unit": "m"},
                    {"label": "local coastline", "color": (30, 30, 26)},
                ],
                stats=region_stats,
                scale={"kind": "local_region", "source_world_width": width, "source_world_height": height, "data_width": cw, "data_height": ch, "row_start": r0, "row_end_exclusive": r1, "col_start": c0, "col_end_exclusive": c1},
            )
            region_summary = {"region_id": f"R{rr + 1:02d}-C{cc + 1:02d}", "filename": filename, **region_stats}
            region_summaries.append(region_summary)
            contact_tiles.append((f"R{rr + 1:02d}-C{cc + 1:02d}", rgb, region_summary))
            index_lines.append(f"{filename},{rr + 1},{cc + 1},{r0},{r1},{c0},{c1},{land_min:.0f},{land_max:.0f},{water_min:.0f},{water_max:.0f},{land_fraction:.5f}")
    header = [
        "WorldGen regional terrain map set",
        "=================================",
        "",
        f"Source terrain grid: {width} x {height}",
        f"Grid: {rows} rows x {cols} columns = {rows * cols} regional maps",
        "Each regional PNG is map-only; legend/scale details are in .legend.json sidecars and map_legends.json.",
        "",
    ]
    (out / "terrain_region_index.csv").write_text("\n".join(header + index_lines) + "\n", encoding="utf-8")
    (out / "terrain_region_summary.json").write_text(json.dumps({"schema_version": 1, "source_width": width, "source_height": height, "rows": rows, "cols": cols, "regions": region_summaries}, indent=2), encoding="utf-8")
    summary_header = ["region_id", "filename", "row", "col", "land_fraction", "ocean_fraction", "mean_land_elevation_m", "max_elevation_m", "coast_cell_share", "mountain_strength", "basin_strength", "valley_corridor_strength", "erosion_deposition_strength", "terrain_character"]
    summary_lines = [",".join(summary_header)]
    for item in region_summaries:
        summary_lines.append(",".join(str(item.get(key, "")) for key in summary_header))
    (out / "terrain_region_summary.csv").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    try:
        tile_w, tile_h = 180, 96
        label_h = 22
        sheet = Image.new("RGB", (cols * tile_w, rows * (tile_h + label_h)), (24, 28, 34))
        draw = ImageDraw.Draw(sheet)
        for label, rgb_tile, stats in contact_tiles:
            rr0 = int(stats.get("row", 1)) - 1; cc0 = int(stats.get("col", 1)) - 1
            img = Image.fromarray(rgb_tile.astype("uint8"), mode="RGB").resize((tile_w, tile_h), Image.BILINEAR)
            x = cc0 * tile_w; y = rr0 * (tile_h + label_h)
            sheet.paste(img, (x, y))
            draw.rectangle((x, y + tile_h, x + tile_w, y + tile_h + label_h), fill=(30, 34, 42))
            text = f"{label} land {float(stats.get('land_fraction', 0.0)):.0%}"
            draw.text((x + 4, y + tile_h + 4), text, fill=(232, 235, 238))
        sheet.save(out / "terrain_region_contact_sheet.png")
        _write_map_legend_sidecar(out / "terrain_region_contact_sheet.png", title="Regional terrain contact sheet", description="Compact 8×16 browse sheet for the regional terrain crop set. Use terrain_region_summary.json/csv for per-region metrics and the individual PNGs for local terrain inspection.", items=[{"label": "regional terrain thumbnails", "color": (136, 154, 102)}], stats={"rows": rows, "cols": cols, "region_count": len(region_summaries), "source_world_width": width, "source_world_height": height}, scale={"kind": "regional_contact_sheet", "source_world_width": width, "source_world_height": height, "rows": rows, "cols": cols})
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Climate overhaul seasonal-driver diagnostic maps (appended definitions override
# older proxy diagnostics with the actual seasonal_v1 driver fields when present).
# ---------------------------------------------------------------------------

def _climate_driver_maps(profile):
    return getattr(profile.climate, "climate_driver_maps", None) or {}


def _climate_driver_info(profile):
    return getattr(profile.climate, "climate_driver_map_info", None) or {}


def _climate_driver_array(profile, key: str):
    import numpy as np
    maps = _climate_driver_maps(profile)
    if key not in maps:
        return None
    try:
        return np.asarray(maps[key])
    except Exception:
        return None


def _driver_land_mask(profile, shape):
    import numpy as np
    terrain = profile.terrain
    info = _climate_driver_info(profile)
    stride = max(1, int(info.get("stride", 1) or 1)) if isinstance(info, dict) else 1
    land = np.asarray(terrain.is_land, dtype=bool)[::stride, ::stride]
    h, w = shape[:2]
    return land[:h, :w]


def _driver_scale_meta(profile, arr, *, kind: str = "climate_driver"):
    info = _climate_driver_info(profile)
    stride = max(1, int(info.get("stride", 1) or 1)) if isinstance(info, dict) else 1
    h, w = arr.shape[:2]
    meta = _world_scale_meta(profile, w, h, kind="diagnostic_downsample" if stride > 1 else "full_world", stride=stride)
    meta["driver_kind"] = kind
    meta["source_width"] = int(info.get("source_width", profile.terrain.width)) if isinstance(info, dict) else profile.terrain.width
    meta["source_height"] = int(info.get("source_height", profile.terrain.height)) if isinstance(info, dict) else profile.terrain.height
    return meta


def _gradient_rgb(arr, color_stops, *, vmin=None, vmax=None, mask=None, background=(55, 88, 125)):
    import numpy as np
    a = np.asarray(arr, dtype=np.float32)
    active = np.ones(a.shape, dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    if vmin is None:
        vmin = float(np.nanpercentile(a[active], 2)) if active.any() else float(np.nanmin(a))
    if vmax is None:
        vmax = float(np.nanpercentile(a[active], 98)) if active.any() else float(np.nanmax(a))
    if vmax <= vmin:
        vmax = vmin + 1.0
    t = np.clip((a - vmin) / (vmax - vmin), 0.0, 1.0)
    stops = [(float(pos), tuple(color)) for pos, color in color_stops]
    rgb = np.zeros((*a.shape, 3), dtype=np.uint8)
    rgb[:, :, :] = background
    for (p0, c0), (p1, c1) in zip(stops[:-1], stops[1:]):
        m = active & (t >= p0) & (t <= p1)
        if not m.any():
            continue
        f = (t[m] - p0) / max(1e-6, p1 - p0)
        for ch in range(3):
            rgb[m, ch] = (c0[ch] + (c1[ch] - c0[ch]) * f).astype(np.uint8)
    below = active & (t < stops[0][0])
    above = active & (t > stops[-1][0])
    if below.any():
        rgb[below] = stops[0][1]
    if above.any():
        rgb[above] = stops[-1][1]
    return rgb, float(vmin), float(vmax)


def _save_driver_scalar(system: StarSystem, output_path: str | Path, *, key: str, divisor: float, title: str, description: str, unit: str, mask_to_land: bool = False, color_stops=None, background=(55, 88, 125), fixed_min=None, fixed_max=None, display_transform: str | None = None):
    if system.main_planet_profile is None:
        raise RuntimeError("Climate driver diagnostic requested, but no Main Planet profile exists.")
    import numpy as np
    profile = system.main_planet_profile
    arr_i = _climate_driver_array(profile, key)
    if arr_i is None:
        raise RuntimeError(f"Climate driver map {key!r} is not available.")
    arr = arr_i.astype(np.float32) / float(divisor)
    land = _driver_land_mask(profile, arr.shape)
    mask = land if mask_to_land else np.ones(arr.shape, dtype=bool)
    if color_stops is None:
        color_stops = [(0.0, (230, 210, 135)), (0.50, (95, 175, 105)), (1.0, (45, 95, 185))]
    actual_vals = arr[mask] if mask.any() else arr.reshape(-1)
    if fixed_min is None:
        actual_min = float(np.nanpercentile(actual_vals, 1)) if actual_vals.size else 0.0
    else:
        actual_min = float(fixed_min)
    if fixed_max is None:
        actual_max = float(np.nanpercentile(actual_vals, 99)) if actual_vals.size else 1.0
    else:
        actual_max = float(fixed_max)
    render_arr = arr
    render_min = actual_min
    render_max = actual_max
    if display_transform == "log1p":
        safe_arr = np.maximum(arr, 0.0)
        render_arr = np.log1p(safe_arr)
        render_min = math.log1p(max(0.0, actual_min))
        render_max = math.log1p(max(0.0, actual_max))
    rgb, _, _ = _gradient_rgb(render_arr, color_stops, vmin=render_min, vmax=render_max, mask=mask, background=background)
    if mask_to_land:
        rgb[~land] = background
    stats = {"driver_key": key, "min": actual_min, "max": actual_max, "climate_mode": getattr(profile.climate, "climate_mode", "legacy")}
    if display_transform:
        stats["display_transform"] = display_transform
    _save_clean_rgb_map(
        rgb,
        output_path,
        title=title,
        description=description,
        items=_continuous_legend(unit, [c for _, c in color_stops], actual_min, actual_max, unit),
        stats=stats,
        scale=_driver_scale_meta(profile, arr, kind=key),
    )


def _save_driver_vector(system: StarSystem, output_path: str | Path, *, u_key: str, v_key: str, title: str, description: str, heat_key: str | None = None, moisture_key: str | None = None, ocean_only: bool = False):
    if system.main_planet_profile is None:
        raise RuntimeError("Climate vector diagnostic requested, but no Main Planet profile exists.")
    from PIL import Image, ImageDraw
    import numpy as np
    profile = system.main_planet_profile
    u_i = _climate_driver_array(profile, u_key)
    v_i = _climate_driver_array(profile, v_key)
    if u_i is None or v_i is None:
        raise RuntimeError(f"Climate vector driver maps {u_key!r}/{v_key!r} are not available.")
    u = u_i.astype(np.float32) / 1000.0
    v = v_i.astype(np.float32) / 1000.0
    land = _driver_land_mask(profile, u.shape)
    if heat_key:
        heat_i = _climate_driver_array(profile, heat_key)
        bg = heat_i.astype(np.float32) / 10.0 if heat_i is not None else np.zeros_like(u)
        stops = [(0.0, (60, 135, 210)), (0.50, (55, 95, 145)), (1.0, (240, 150, 70))]
        rgb, _, _ = _gradient_rgb(bg, stops, vmin=-4.5, vmax=4.5, mask=~land, background=(118, 130, 94))
        rgb[land] = (118, 130, 94)
    elif moisture_key:
        m_i = _climate_driver_array(profile, moisture_key)
        bg = m_i.astype(np.float32) / 1000.0 if m_i is not None else np.sqrt(u*u+v*v)
        stops = [(0.0, (230, 210, 135)), (0.55, (95, 175, 105)), (1.0, (45, 95, 185))]
        rgb, _, _ = _gradient_rgb(bg, stops, vmin=0.0, vmax=max(1.0, float(np.nanpercentile(bg, 98))), mask=land, background=(55, 88, 125))
    else:
        speed = np.sqrt(u * u + v * v)
        stops = [(0.0, (220, 220, 210)), (1.0, (70, 110, 210))]
        rgb, _, _ = _gradient_rgb(speed, stops, vmin=0.0, vmax=max(1.0, float(np.nanpercentile(speed, 98))), mask=np.ones(u.shape, dtype=bool), background=(55, 88, 125))
        rgb[land] = (135, 145, 105)
    image = Image.fromarray(rgb, mode="RGB")
    draw = ImageDraw.Draw(image)
    h, w = u.shape
    step_x = max(40, w // 28)
    step_y = max(30, h // 16)
    scale_len = min(step_x, step_y) * 0.42
    for y in range(step_y // 2, h, step_y):
        for x in range(step_x // 2, w, step_x):
            if ocean_only and land[y, x]:
                continue
            uu = float(u[y, x]); vv = float(v[y, x])
            mag = (uu * uu + vv * vv) ** 0.5
            if mag < 0.05:
                continue
            color = (245, 245, 245) if not ocean_only else ((255, 175, 75) if uu > 0 else (100, 190, 255))
            _draw_arrow(draw, x, y, x + uu / mag * scale_len, y + vv / mag * scale_len, fill=color, width=2)
    output_path = _prepare_output_path(output_path)
    _save_image_fast(image, output_path)
    _write_map_legend_sidecar(
        output_path,
        title=title,
        description=description,
        items=[("land", (118, 130, 94)), ("driver background", (55, 88, 125)), ("vector arrows", (245, 245, 245))],
        stats={"u_driver_key": u_key, "v_driver_key": v_key, "climate_mode": getattr(profile.climate, "climate_mode", "legacy")},
        scale=_driver_scale_meta(profile, u, kind=f"{u_key}+{v_key}"),
    )


def save_main_planet_wind_currents_view(system: StarSystem, output_path: str | Path) -> None:
    if system.main_planet_profile is not None and "wind_u_annual_x1000" in _climate_driver_maps(system.main_planet_profile):
        return _save_driver_vector(system, output_path, u_key="wind_u_annual_x1000", v_key="wind_v_annual_x1000", moisture_key="moisture_annual_x1000", title=f"{system.main_planet_profile.planet_name} seasonal_v1 annual wind and moisture transport", description="Actual seasonal_v1 annual mean wind vectors over the annual moisture field. This replaces the old latitude-only wind proxy.")
    # Fallback to the pre-overhaul proxy if legacy climate is selected.
    if system.main_planet_profile is None:
        raise RuntimeError("Wind diagnostic requested, but no Main Planet profile exists.")
    from PIL import Image, ImageDraw
    profile = system.main_planet_profile
    terrain = profile.terrain
    out_w, out_h, stride = _diagnostic_downsample_dimensions(terrain.width, terrain.height)
    image = Image.fromarray(_simple_terrain_background(terrain, stride), mode="RGB")
    draw = ImageDraw.Draw(image)
    step_x = max(64, out_w // 28); step_y = max(48, out_h // 16); scale = min(step_x, step_y) * 0.42
    for y in range(step_y // 2, out_h, step_y):
        lat = 90.0 - (y + 0.5) * 180.0 / out_h
        dr, dc = _diagnostic_wind_vector(lat)
        for x in range(step_x // 2, out_w, step_x):
            _draw_arrow(draw, x, y, x + dc * scale, y + dr * scale, fill=(245, 245, 245), width=2)
    output_path = _prepare_output_path(output_path); _save_image_fast(image, output_path)
    _write_map_legend_sidecar(output_path, title="Prevailing wind diagnostic", description="Legacy simplified prevailing wind belts.", items=[("terrain background", (142, 162, 104)), ("wind arrows", (245, 245, 245))], scale=_world_scale_meta(profile, out_w, out_h, kind="diagnostic_downsample", stride=stride))


def save_main_planet_ocean_currents_view(system: StarSystem, output_path: str | Path) -> None:
    if system.main_planet_profile is not None and "current_u_annual_x1000" in _climate_driver_maps(system.main_planet_profile):
        return _save_driver_vector(system, output_path, u_key="current_u_annual_x1000", v_key="current_v_annual_x1000", heat_key="current_heat_annual_c_x10", ocean_only=True, title=f"{system.main_planet_profile.planet_name} seasonal_v1 ocean currents", description="Actual seasonal_v1 annual mean ocean-current vectors over warm/cold current heat influence. Island-origin diagnostics now use main_planet_islands_archipelago.png instead.")
    # Do not reuse island-origin diagnostics here anymore; this file is always currents.
    if system.main_planet_profile is None:
        raise RuntimeError("Ocean-current diagnostic requested, but no Main Planet profile exists.")
    from PIL import Image, ImageDraw
    import numpy as np
    profile = system.main_planet_profile; terrain = profile.terrain
    out_w, out_h, stride = _diagnostic_downsample_dimensions(terrain.width, terrain.height)
    land = np.asarray(terrain.is_land, dtype=bool)[::stride, ::stride]
    rgb = np.zeros((out_h, out_w, 3), dtype=np.uint8); rgb[:, :, :] = (35, 88, 140); rgb[land] = (115, 130, 94)
    image = Image.fromarray(rgb, mode="RGB"); draw = ImageDraw.Draw(image)
    step_x = max(80, out_w // 24); step_y = max(60, out_h // 14); scale_len = min(step_x, step_y) * 0.45
    for y in range(step_y // 2, out_h, step_y):
        lat = 90.0 - (y + 0.5) * 180.0 / out_h
        if abs(lat) < 5: continue
        for x in range(step_x // 2, out_w, step_x):
            if land[y, x]: continue
            hemi = 1 if lat >= 0 else -1; gyre = 1 if abs(lat) < 45 else -1
            dc = gyre * (1 if (x / out_w) < 0.5 else -1); dr = -hemi * 0.45 * gyre
            color = (255, 170, 70) if dc > 0 else (90, 185, 255)
            _draw_arrow(draw, x, y, x + dc * scale_len, y + dr * scale_len, fill=color, width=2)
    output_path = _prepare_output_path(output_path); _save_image_fast(image, output_path)
    _write_map_legend_sidecar(output_path, title="Ocean-current diagnostic", description="Legacy simplified gyres. Island-origin diagnostics are no longer written into this map.", items=[("ocean", (35, 88, 140)), ("land", (115, 130, 94)), ("warm/current arrow", (255, 170, 70)), ("cold/current arrow", (90, 185, 255))], scale=_world_scale_meta(profile, out_w, out_h, kind="diagnostic_downsample", stride=stride))


def save_main_planet_moisture_transport_view(system: StarSystem, output_path: str | Path) -> None:
    if system.main_planet_profile is not None and "moisture_wind_u_annual_x1000" in _climate_driver_maps(system.main_planet_profile):
        return _save_driver_vector(system, output_path, u_key="moisture_wind_u_annual_x1000", v_key="moisture_wind_v_annual_x1000", moisture_key="moisture_annual_x1000", title=f"{system.main_planet_profile.planet_name} seasonal_v2 moisture transport", description="Actual seasonal_v2 annual moisture-bearing wind vectors over annual moisture. This is distinct from surface wind and is the driver used for rainfall advection.")
    if system.main_planet_profile is not None and "moisture_annual_x1000" in _climate_driver_maps(system.main_planet_profile):
        return _save_driver_vector(system, output_path, u_key="wind_u_annual_x1000", v_key="wind_v_annual_x1000", moisture_key="moisture_annual_x1000", title=f"{system.main_planet_profile.planet_name} seasonal_v1 moisture transport", description="Actual seasonal_v1 annual moisture field with annual wind vectors. This is the driver field behind inland rainfall, not only final precipitation.")
    # Legacy fallback: use final precipitation plus diagnostic winds.
    return save_main_planet_wind_currents_view(system, output_path)


def _save_legacy_rain_shadow_proxy(system: StarSystem, output_path: str | Path) -> None:
    """Render the pre-overhaul rain-shadow proxy when seasonal driver rasters are unavailable.

    This keeps main_planet_rain_shadow.png available for legacy climate mode and
    for partially migrated/stale staged runs.  The seasonal_v1 actual driver is
    still preferred whenever rain_shadow_annual_x1000 is present.
    """
    if system.main_planet_profile is None:
        raise RuntimeError("Rain-shadow diagnostic requested, but no Main Planet profile exists.")
    import numpy as np

    profile = system.main_planet_profile
    terrain = profile.terrain
    climate = profile.climate
    out_w, out_h, stride = _diagnostic_downsample_dimensions(terrain.width, terrain.height)
    elev = np.asarray(terrain.elevation_m, dtype=np.float32)[::stride, ::stride]
    land = np.asarray(terrain.is_land, dtype=bool)[::stride, ::stride]
    precip = np.asarray(climate.annual_precip_mm, dtype=np.float32)[::stride, ::stride]
    gy, gx = np.gradient(elev)
    relief = np.sqrt(gx * gx + gy * gy)
    local_p = precip
    local_mean = (
        local_p
        + np.roll(local_p, 8, axis=1)
        + np.roll(local_p, -8, axis=1)
        + np.roll(local_p, 8, axis=0)
        + np.roll(local_p, -8, axis=0)
    ) / 5.0
    relief_scale = max(1.0, np.quantile(relief[land], 0.98) if land.any() else 1.0)
    relief_factor = np.clip(relief / relief_scale, 0.0, 1.0)
    shadow = np.clip((local_mean - local_p) / 650.0, 0.0, 1.0) * relief_factor
    windward = np.clip((local_p - local_mean) / 650.0, 0.0, 1.0) * relief_factor

    rgb = np.zeros((out_h, out_w, 3), dtype=np.uint8)
    rgb[:, :, :] = (58, 92, 120)
    rgb[land] = (140, 142, 116)
    sh = land & (shadow > 0.08)
    wi = land & (windward > 0.08)
    rgb[sh, 0] = (190 + 55 * shadow[sh]).astype(np.uint8)
    rgb[sh, 1] = (155 - 70 * shadow[sh]).astype(np.uint8)
    rgb[sh, 2] = (80 - 35 * shadow[sh]).astype(np.uint8)
    rgb[wi, 0] = (75 - 20 * windward[wi]).astype(np.uint8)
    rgb[wi, 1] = (150 + 70 * windward[wi]).astype(np.uint8)
    rgb[wi, 2] = (120 + 90 * windward[wi]).astype(np.uint8)

    _save_clean_rgb_map(
        rgb,
        output_path,
        title="Rain-shadow diagnostic — legacy proxy",
        description=(
            "Fallback proxy used when seasonal_v1 rain-shadow driver rasters are not available. "
            "Green areas mark wetter windward relief; brown areas mark dry leeward-relief candidates. "
            "Rerun from the climate stage with --climate-mode seasonal_v1 to generate the actual downwind rain-shadow driver."
        ),
        items=[
            ("ocean/background", (58, 92, 120)),
            ("neutral land", (140, 142, 116)),
            ("windward/wet relief", (55, 220, 210)),
            ("leeward/dry relief candidate", (245, 85, 45)),
        ],
        scale=_world_scale_meta(profile, out_w, out_h, kind="diagnostic_downsample", stride=stride),
        stats={
            "climate_mode": getattr(profile.climate, "climate_mode", "legacy"),
            "fallback": "legacy_rain_shadow_proxy",
            "missing_driver_key": "rain_shadow_annual_x1000",
        },
    )


def save_main_planet_rain_shadow_view(system: StarSystem, output_path: str | Path) -> None:
    if system.main_planet_profile is not None and "rain_shadow_annual_x1000" in _climate_driver_maps(system.main_planet_profile):
        return save_main_planet_rain_shadow_actual_view(system, output_path)
    return _save_legacy_rain_shadow_proxy(system, output_path)


def save_main_planet_itcz_position_view(system: StarSystem, output_path: str | Path) -> None:
    return _save_driver_scalar(system, output_path, key="itcz_annual_x1000", divisor=1000.0, title="Seasonal_v1 ITCZ / convergence strength — annual composite", description="Maximum seasonal ITCZ/convergence strength. Use the seasonal ITCZ maps to inspect NH summer, equinox, and NH winter migration separately.", unit="ITCZ index", color_stops=[(0.0, (50, 80, 125)), (0.45, (140, 190, 130)), (1.0, (245, 225, 95))], fixed_min=0.0, fixed_max=1.6)


def _itcz_view(season, label, system, output_path):
    return _save_driver_scalar(system, output_path, key=f"itcz_{season}_x1000", divisor=1000.0, title=f"Seasonal_v1 ITCZ / convergence — {label}", description="Seasonal ITCZ/convergence strength for this seasonal anchor. This is the field that feeds pressure, winds, moisture convergence, and monsoon-like rainfall.", unit="ITCZ index", color_stops=[(0.0, (50, 80, 125)), (0.45, (140, 190, 130)), (1.0, (245, 225, 95))], fixed_min=0.0, fixed_max=1.6)


def save_main_planet_itcz_nh_summer_view(system, output_path): return _itcz_view("nh_summer", "NH summer", system, output_path)
def save_main_planet_itcz_equinox_view(system, output_path): return _itcz_view("equinox", "equinox", system, output_path)
def save_main_planet_itcz_nh_winter_view(system, output_path): return _itcz_view("nh_winter", "NH winter", system, output_path)



def save_main_planet_thermal_equator_view(system: StarSystem, output_path: str | Path) -> None:
    return _save_driver_scalar(system, output_path, key="thermal_equator_annual_x1000", divisor=1000.0, title="Seasonal_v2 thermal-equator driver — annual composite", description="Annual composite of the explicit seasonal_v2 thermal-equator target. It should bend toward warm seasonal land/ocean pools instead of remaining a straight latitude line.", unit="thermal-equator index", color_stops=[(0.0, (50, 80, 125)), (0.50, (125, 190, 150)), (1.0, (255, 235, 100))], fixed_min=0.0, fixed_max=1.6)


def _thermal_equator_view(season, label, system, output_path):
    return _save_driver_scalar(system, output_path, key=f"thermal_equator_{season}_x1000", divisor=1000.0, title=f"Seasonal_v2 thermal-equator driver — {label}", description="Seasonal thermal-equator/convergence target used before ITCZ and pressure belts are derived. Warm continents and tropical warm pools should distort this field smoothly.", unit="thermal-equator index", color_stops=[(0.0, (50, 80, 125)), (0.50, (125, 190, 150)), (1.0, (255, 235, 100))], fixed_min=0.0, fixed_max=1.6)


def save_main_planet_thermal_equator_nh_summer_view(system, output_path): return _thermal_equator_view("nh_summer", "NH summer", system, output_path)
def save_main_planet_thermal_equator_equinox_view(system, output_path): return _thermal_equator_view("equinox", "equinox", system, output_path)
def save_main_planet_thermal_equator_nh_winter_view(system, output_path): return _thermal_equator_view("nh_winter", "NH winter", system, output_path)


_PRESSURE_BELT_CLASS_ITEMS = {
    1: ("thermal-equator axis", (245, 245, 245)),
    2: ("ITCZ low / convergence", (245, 220, 80)),
    3: ("subtropical high / subsidence", (230, 160, 85)),
    4: ("subpolar low / storm track", (155, 120, 215)),
    5: ("weak/interbelt background", (90, 135, 185)),
    6: ("polar high", (210, 230, 245)),
}


def _pressure_belts_view(season, label, system, output_path):
    return _save_driver_class(
        system,
        output_path,
        key=f"pressure_belt_{season}_class",
        title=f"Seasonal_v2 pressure-belt structure — {label}",
        description=(
            "Categorical pressure-belt diagnostic derived from the explicit seasonal_v2 thermal equator and pressure field. "
            "Subtropical highs, storm-track/subpolar lows, and polar highs should migrate coherently with the ITCZ instead of sitting on fixed hard latitude bands."
        ),
        class_items=_PRESSURE_BELT_CLASS_ITEMS,
    )


def save_main_planet_pressure_belts_nh_summer_view(system, output_path): return _pressure_belts_view("nh_summer", "NH summer", system, output_path)
def save_main_planet_pressure_belts_equinox_view(system, output_path): return _pressure_belts_view("equinox", "equinox", system, output_path)
def save_main_planet_pressure_belts_nh_winter_view(system, output_path): return _pressure_belts_view("nh_winter", "NH winter", system, output_path)


def save_main_planet_storm_track_moisture_view(system: StarSystem, output_path: str | Path) -> None:
    return _save_driver_scalar(system, output_path, key="storm_track_moisture_annual_x1000", divisor=1000.0, title="Seasonal_v2 storm-track moisture — annual composite", description="Annual mean storm-track/frontal moisture driver. This replaces the previous hard latitude envelope that could create straight rainfall transitions.", unit="storm moisture index", color_stops=[(0.0, (70, 90, 125)), (0.50, (100, 165, 190)), (1.0, (215, 225, 245))], fixed_min=0.0, fixed_max=1.5)


def _storm_track_moisture_view(season, label, system, output_path):
    return _save_driver_scalar(system, output_path, key=f"storm_track_moisture_{season}_x1000", divisor=1000.0, title=f"Seasonal_v2 storm-track moisture — {label}", description="Seasonal storm-track/frontal moisture driver, smoothed and pressure-aware to avoid hard straight rainfall edges.", unit="storm moisture index", color_stops=[(0.0, (70, 90, 125)), (0.50, (100, 165, 190)), (1.0, (215, 225, 245))], fixed_min=0.0, fixed_max=1.5)


def save_main_planet_storm_track_moisture_nh_summer_view(system, output_path): return _storm_track_moisture_view("nh_summer", "NH summer", system, output_path)
def save_main_planet_storm_track_moisture_equinox_view(system, output_path): return _storm_track_moisture_view("equinox", "equinox", system, output_path)
def save_main_planet_storm_track_moisture_nh_winter_view(system, output_path): return _storm_track_moisture_view("nh_winter", "NH winter", system, output_path)


def save_main_planet_moisture_wind_nh_summer_view(system, output_path): return _save_driver_vector(system, output_path, u_key="moisture_wind_u_nh_summer_x1000", v_key="moisture_wind_v_nh_summer_x1000", moisture_key="moisture_nh_summer_x1000", title="Seasonal_v2 moisture-transport wind — NH summer", description="Moisture-bearing flow used for rainfall advection, separated from the surface wind field so rainfall does not inherit hard wind-belt transitions.")
def save_main_planet_moisture_wind_equinox_view(system, output_path): return _save_driver_vector(system, output_path, u_key="moisture_wind_u_equinox_x1000", v_key="moisture_wind_v_equinox_x1000", moisture_key="moisture_equinox_x1000", title="Seasonal_v2 moisture-transport wind — equinox", description="Moisture-bearing flow used for rainfall advection, separated from the surface wind field so rainfall does not inherit hard wind-belt transitions.")
def save_main_planet_moisture_wind_nh_winter_view(system, output_path): return _save_driver_vector(system, output_path, u_key="moisture_wind_u_nh_winter_x1000", v_key="moisture_wind_v_nh_winter_x1000", moisture_key="moisture_nh_winter_x1000", title="Seasonal_v2 moisture-transport wind — NH winter", description="Moisture-bearing flow used for rainfall advection, separated from the surface wind field so rainfall does not inherit hard wind-belt transitions.")

def _pressure_view(season, label, system, output_path):
    return _save_driver_scalar(system, output_path, key=f"pressure_{season}_hpa_x10", divisor=10.0, title=f"Seasonal_v1 pressure — {label}", description="Seasonal pressure field used to derive wind vectors. Includes pressure belts, ITCZ lows, continental thermal lows/highs, and terrain effects.", unit="hPa", color_stops=[(0.0, (70, 95, 180)), (0.5, (235, 235, 225)), (1.0, (190, 80, 60))])


def save_main_planet_pressure_nh_summer_view(system, output_path): return _pressure_view("nh_summer", "NH summer", system, output_path)
def save_main_planet_pressure_equinox_view(system, output_path): return _pressure_view("equinox", "equinox", system, output_path)
def save_main_planet_pressure_nh_winter_view(system, output_path): return _pressure_view("nh_winter", "NH winter", system, output_path)


def _wind_view(season, label, system, output_path):
    return _save_driver_vector(system, output_path, u_key=f"wind_u_{season}_x1000", v_key=f"wind_v_{season}_x1000", moisture_key=f"moisture_{season}_x1000", title=f"Seasonal_v1 winds — {label}", description="Seasonal wind vectors over seasonal moisture. These are the actual fields used for moisture transport.")


def save_main_planet_wind_nh_summer_view(system, output_path): return _wind_view("nh_summer", "NH summer", system, output_path)
def save_main_planet_wind_equinox_view(system, output_path): return _wind_view("equinox", "equinox", system, output_path)
def save_main_planet_wind_nh_winter_view(system, output_path): return _wind_view("nh_winter", "NH winter", system, output_path)


def _temperature_season_view(season, label, system, output_path):
    profile = system.main_planet_profile
    arr = _climate_driver_array(profile, f"temperature_{season}_c_x10") if profile else None
    fixed_min = None; fixed_max = None
    if arr is not None:
        import numpy as np
        vals = arr.astype(np.float32) / 10.0
        fixed_min = float(np.nanpercentile(vals, 1)); fixed_max = float(np.nanpercentile(vals, 99))
    return _save_driver_scalar(system, output_path, key=f"temperature_{season}_c_x10", divisor=10.0, title=f"Seasonal_v1 temperature — {label}", description="Seasonal mean temperature anchor used to synthesize monthly values for Köppen classification.", unit="°C", color_stops=[(0.0, (45, 90, 200)), (0.5, (235, 238, 230)), (1.0, (190, 30, 30))], fixed_min=fixed_min, fixed_max=fixed_max)


def save_main_planet_temperature_nh_summer_view(system, output_path): return _temperature_season_view("nh_summer", "NH summer", system, output_path)
def save_main_planet_temperature_equinox_view(system, output_path): return _temperature_season_view("equinox", "equinox", system, output_path)
def save_main_planet_temperature_nh_winter_view(system, output_path): return _temperature_season_view("nh_winter", "NH winter", system, output_path)


def _precip_season_view(season, label, system, output_path):
    return _save_driver_scalar(system, output_path, key=f"precipitation_{season}_mm", divisor=1.0, title=f"Seasonal_v1 precipitation — {label}", description="Seasonal annualized precipitation rate before monthly interpolation. Oceans are masked so land patterns are easier to inspect. Colors use a log-scaled display to preserve detail in both dry and wet climates.", unit="mm/year", mask_to_land=True, color_stops=[(0.0, (245, 225, 170)), (0.20, (201, 211, 118)), (0.45, (106, 182, 110)), (0.70, (76, 145, 169)), (1.0, (54, 87, 167))], fixed_min=0.0, display_transform="log1p")


def save_main_planet_precipitation_nh_summer_view(system, output_path): return _precip_season_view("nh_summer", "NH summer", system, output_path)
def save_main_planet_precipitation_equinox_view(system, output_path): return _precip_season_view("equinox", "equinox", system, output_path)
def save_main_planet_precipitation_nh_winter_view(system, output_path): return _precip_season_view("nh_winter", "NH winter", system, output_path)


def _moisture_season_view(season, label, system, output_path):
    return _save_driver_scalar(system, output_path, key=f"moisture_{season}_x1000", divisor=1000.0, title=f"Seasonal_v1 moisture — {label}", description="Seasonal advected atmospheric moisture index. Small inland-water bodies are capacity-limited before this field is computed.", unit="moisture index", color_stops=[(0.0, (220, 205, 135)), (0.55, (100, 175, 120)), (1.0, (65, 120, 210))], fixed_min=0.0)


def save_main_planet_moisture_nh_summer_view(system, output_path): return _moisture_season_view("nh_summer", "NH summer", system, output_path)
def save_main_planet_moisture_equinox_view(system, output_path): return _moisture_season_view("equinox", "equinox", system, output_path)
def save_main_planet_moisture_nh_winter_view(system, output_path): return _moisture_season_view("nh_winter", "NH winter", system, output_path)


def save_main_planet_orographic_lift_view(system: StarSystem, output_path: str | Path) -> None:
    return _save_driver_scalar(system, output_path, key="orographic_lift_annual_x1000", divisor=1000.0, title="Seasonal_v1 actual orographic lift", description="Annual mean windward relief/uplift driver used by the precipitation model. This is based on actual seasonal wind fields and moisture availability.", unit="lift index", mask_to_land=True, color_stops=[(0.0, (145, 135, 110)), (0.55, (85, 170, 130)), (1.0, (80, 190, 230))], fixed_min=0.0)


def save_main_planet_rain_shadow_actual_view(system: StarSystem, output_path: str | Path) -> None:
    return _save_driver_scalar(system, output_path, key="rain_shadow_annual_x1000", divisor=1000.0, title="Seasonal_v1 actual rain-shadow driver", description="Annual mean leeward rain-shadow driver based on seasonal wind direction and relief. Windward high slopes should no longer be classified as rain shadow just because they are drier than nearby foothills.", unit="shadow index", mask_to_land=True, color_stops=[(0.0, (135, 150, 115)), (0.50, (205, 150, 80)), (1.0, (120, 70, 45))], fixed_min=0.0, fixed_max=1.0)


def save_main_planet_aridity_index_view(system: StarSystem, output_path: str | Path) -> None:
    return _save_driver_scalar(system, output_path, key="aridity_index_x1000", divisor=1000.0, title="Seasonal_v1 Köppen aridity index", description="Annual precipitation divided by the local Köppen dry-climate threshold. Values below 1 tend toward steppe/desert; higher values are humid enough for non-B climates.", unit="P/P_threshold", mask_to_land=True, color_stops=[(0.0, (190, 105, 55)), (0.40, (230, 205, 120)), (1.0, (70, 160, 110))], fixed_min=0.0, fixed_max=2.5)


def save_main_planet_lake_moisture_sources_view(system: StarSystem, output_path: str | Path) -> None:
    return _save_driver_scalar(system, output_path, key="inland_water_source_x1000", divisor=1000.0, title="Seasonal_v1 inland-water moisture source cap", description="Capacity-limited evaporation influence for inland lakes/seas. Tiny lakes in deserts should show very low source strength instead of creating regional rainforest conditions.", unit="source cap", color_stops=[(0.0, (55, 88, 125)), (0.35, (120, 175, 180)), (1.0, (245, 230, 140))], fixed_min=0.0, fixed_max=0.60)

# ---------------------------------------------------------------------------
# Climate overhaul 03 diagnostics: native monthly progression, circulation
# zones, and gyre-class maps.
# ---------------------------------------------------------------------------

_MONTH_LABELS = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")


def _save_driver_class(system: StarSystem, output_path: str | Path, *, key: str, title: str, description: str, class_items: dict[int, tuple[str, tuple[int, int, int]]], water_background=(55, 88, 125), land_background=(118, 130, 94)) -> None:
    if system.main_planet_profile is None:
        raise RuntimeError("Climate class diagnostic requested, but no Main Planet profile exists.")
    import numpy as np
    profile = system.main_planet_profile
    arr_i = _climate_driver_array(profile, key)
    if arr_i is None:
        raise RuntimeError(f"Climate driver class map {key!r} is not available.")
    arr = arr_i.astype(np.int16)
    land = _driver_land_mask(profile, arr.shape)
    rgb = np.zeros((*arr.shape, 3), dtype=np.uint8)
    rgb[:, :, :] = water_background
    rgb[land] = land_background
    for code, (_label, color) in class_items.items():
        mask = arr == int(code)
        if mask.any():
            rgb[mask] = color
    _save_clean_rgb_map(
        rgb,
        output_path,
        title=title,
        description=description,
        items=[(label, color) for code, (label, color) in sorted(class_items.items())],
        stats={"driver_key": key, "climate_mode": getattr(profile.climate, "climate_mode", "legacy")},
        scale=_driver_scale_meta(profile, arr, kind=key),
    )


_CIRCULATION_CLASS_ITEMS = {
    1: ("equator", (245, 245, 245)),
    2: ("ITCZ / convergence", (245, 220, 80)),
    3: ("tropics", (100, 190, 120)),
    4: ("horse latitudes / subtropical highs", (230, 180, 95)),
    5: ("westerly belt", (95, 145, 210)),
    6: ("polar front / storm track", (170, 120, 210)),
    7: ("polar cap", (210, 230, 245)),
}


def _save_circulation_zone_map(system: StarSystem, output_path: str | Path, *, key: str, label: str) -> None:
    return _save_driver_class(
        system,
        output_path,
        key=key,
        title=f"Seasonal_v1 circulation-zone guide — {label}",
        description=(
            "Diagnostic guide to the seasonal_v1 circulation framework: equator, seasonal ITCZ/convergence, "
            "tropics, subtropical highs/horse latitudes, westerlies, polar fronts, and polar caps. "
            "The guide bands are latitude references; ITCZ/convergence follows the seasonal pressure/moisture state."
        ),
        class_items=_CIRCULATION_CLASS_ITEMS,
    )


def save_main_planet_circulation_zones_view(system: StarSystem, output_path: str | Path) -> None:
    return _save_circulation_zone_map(system, output_path, key="circulation_zone_class", label="annual composite")


def save_main_planet_circulation_zones_nh_summer_view(system: StarSystem, output_path: str | Path) -> None:
    return _save_circulation_zone_map(system, output_path, key="circulation_zone_nh_summer_class", label="NH summer")


def save_main_planet_circulation_zones_equinox_view(system: StarSystem, output_path: str | Path) -> None:
    return _save_circulation_zone_map(system, output_path, key="circulation_zone_equinox_class", label="equinox")


def save_main_planet_circulation_zones_nh_winter_view(system: StarSystem, output_path: str | Path) -> None:
    return _save_circulation_zone_map(system, output_path, key="circulation_zone_nh_winter_class", label="NH winter")


def save_main_planet_ocean_gyres_view(system: StarSystem, output_path: str | Path) -> None:
    # Draw a loop-style gyre diagnostic from the actual annual current vector field.
    if system.main_planet_profile is None:
        raise RuntimeError("Ocean gyre diagnostic requested, but no Main Planet profile exists.")
    from PIL import Image, ImageDraw
    import numpy as np
    profile = system.main_planet_profile
    cls_i = _climate_driver_array(profile, "ocean_gyre_class")
    u_i = _climate_driver_array(profile, "current_u_annual_x1000")
    v_i = _climate_driver_array(profile, "current_v_annual_x1000")
    heat_i = _climate_driver_array(profile, "current_heat_annual_c_x10")
    basin_i = _climate_driver_array(profile, "ocean_basin_id")
    if cls_i is None:
        raise RuntimeError("Ocean gyre class map is not available.")
    cls = cls_i.astype(np.int16)
    land = _driver_land_mask(profile, cls.shape)
    colors = {
        0: (42, 82, 135),
        1: (75, 155, 215),
        2: (45, 105, 175),
        3: (85, 130, 190),
        4: (145, 180, 210),
        5: (245, 155, 75),
        6: (70, 190, 230),
        7: (175, 235, 245),
    }
    rgb = np.zeros((*cls.shape, 3), dtype=np.uint8)
    rgb[:, :, :] = colors[0]
    for code, color in colors.items():
        mask = cls == code
        if mask.any():
            rgb[mask] = color
    rgb[land] = (118, 130, 94)
    image = Image.fromarray(rgb, mode="RGB")
    draw = ImageDraw.Draw(image)

    if u_i is not None and v_i is not None:
        u = u_i.astype(np.float32) / 1000.0
        v = v_i.astype(np.float32) / 1000.0
        heat = heat_i.astype(np.float32) / 10.0 if heat_i is not None else np.zeros_like(u)
        h, w = u.shape
        basin = basin_i.astype(np.int32) if basin_i is not None else None

        drew_basin_loops = False
        if basin is not None and np.any(basin > 0):
            # seasonal_v3/v4: draw basin-scale loop guides from the classified current layer.
            # This keeps the gyre diagnostic readable instead of turning into many
            # horizontal streamline bands.
            y_lat = 90.0 - (np.arange(h, dtype=np.float32) + 0.5) * 180.0 / float(h)
            basin_ids = [int(vv) for vv in np.unique(basin) if int(vv) > 0]
            basin_ids = sorted(basin_ids, key=lambda bid: int(np.sum(basin == bid)), reverse=True)[:12]
            for bid in basin_ids:
                for hemi_name, row_mask, clockwise, color in [
                    ("NH subtropical", (y_lat >= 8.0) & (y_lat <= 45.0), True, (250, 165, 80)),
                    ("SH subtropical", (y_lat <= -8.0) & (y_lat >= -45.0), False, (250, 165, 80)),
                    ("NH subpolar", (y_lat >= 45.0) & (y_lat <= 67.0), False, (210, 235, 245)),
                    ("SH subpolar", (y_lat <= -45.0) & (y_lat >= -67.0), True, (210, 235, 245)),
                ]:
                    mask = (basin == bid) & row_mask[:, None] & (~land)
                    if int(np.sum(mask)) < max(28, (h * w) // 900):
                        continue
                    ys, xs = np.nonzero(mask)
                    full_x0, full_x1 = int(xs.min()), int(xs.max())
                    full_span = full_x1 - full_x0 + 1
                    # Very wide basins otherwise draw one almost-horizontal global
                    # oval.  Split them into several basin windows so the diagnostic
                    # shows recognizable subtropical/subpolar loops.
                    if full_span > int(0.58 * w):
                        chunk_count = 3 if full_span > int(0.82 * w) else 2
                        x_chunks = []
                        for chunk_idx in range(chunk_count):
                            cx0 = int(round(chunk_idx * w / chunk_count))
                            cx1 = int(round((chunk_idx + 1) * w / chunk_count)) - 1
                            x_chunks.append((cx0, max(cx0, cx1)))
                    else:
                        x_chunks = [(full_x0, full_x1)]

                    for chunk_x0, chunk_x1 in x_chunks:
                        x_window = (np.arange(w) >= chunk_x0) & (np.arange(w) <= chunk_x1)
                        chunk_mask = mask & x_window[None, :]
                        if int(np.sum(chunk_mask)) < max(18, (h * w) // 1600):
                            continue
                        cys, cxs = np.nonzero(chunk_mask)
                        x0, x1 = int(cxs.min()), int(cxs.max())
                        y0, y1 = int(cys.min()), int(cys.max())
                        if (x1 - x0) < max(10, w // 42) or (y1 - y0) < max(6, h // 44):
                            continue
                        pad_x = max(2, int((x1 - x0) * 0.10)); pad_y = max(2, int((y1 - y0) * 0.10))
                        cx = 0.5 * (x0 + x1); cy = 0.5 * (y0 + y1)
                        rx = max(7.0, min(0.5 * (x1 - x0 - 2 * pad_x), 0.22 * w))
                        ry = max(5.0, min(0.5 * (y1 - y0 - 2 * pad_y), 0.16 * h))
                        pts = []
                        segments = []
                        steps = 96
                        for i in range(steps + 1):
                            t = 2.0 * math.pi * (i / steps)
                            if not clockwise:
                                t = -t
                            x = (cx + rx * math.cos(t)) % w
                            y = cy + ry * math.sin(t)
                            iy = int(round(y)); ix = int(round(x)) % w
                            if 0 <= iy < h and not land[iy, ix] and cls[iy, ix] != 0:
                                pts.append((x, y))
                            else:
                                if len(pts) >= 4:
                                    segments.append(pts)
                                pts = []
                        if len(pts) >= 4:
                            segments.append(pts)
                        for pts in segments:
                            if len(pts) >= 12:
                                draw.line(pts, fill=color, width=3)
                                _draw_arrow(draw, pts[-3][0], pts[-3][1], pts[-1][0], pts[-1][1], fill=color, width=3)
                                drew_basin_loops = True

        if not drew_basin_loops:
            seed_step_x = max(70, w // 22)
            seed_step_y = max(48, h // 14)
            step_len = max(4.0, min(seed_step_x, seed_step_y) * 0.22)
            max_steps = 58
            for y0 in range(seed_step_y // 2, h, seed_step_y):
                for x0 in range(seed_step_x // 2, w, seed_step_x):
                    if land[y0, x0] or cls[y0, x0] == 0:
                        continue
                    x = float(x0); y = float(y0)
                    pts = []
                    local_heat = []
                    for _ in range(max_steps):
                        ix = int(round(x)) % w
                        iy = int(round(y))
                        if iy < 0 or iy >= h or land[iy, ix] or cls[iy, ix] == 0:
                            break
                        uu = float(u[iy, ix]); vv = float(v[iy, ix])
                        mag = (uu * uu + vv * vv) ** 0.5
                        if mag < 0.035:
                            break
                        pts.append((x, y)); local_heat.append(float(heat[iy, ix]))
                        x = (x + uu / mag * step_len) % w
                        y = y + vv / mag * step_len
                    if len(pts) < 6:
                        continue
                    hmean = sum(local_heat) / max(1, len(local_heat))
                    color = (250, 165, 80) if hmean > 0.35 else ((85, 205, 245) if hmean < -0.35 else (245, 245, 235))
                    draw.line(pts, fill=color, width=2)
                    if len(pts) >= 2:
                        x1, y1 = pts[-2]; x2, y2 = pts[-1]
                        _draw_arrow(draw, x1, y1, x2, y2, fill=color, width=2)

    output_path = _prepare_output_path(output_path)
    _save_image_fast(image, output_path)
    _write_map_legend_sidecar(
        output_path,
        title="Ocean gyres and current branches",
        description=(
            "Loop-style diagnostic drawn from the annual ocean-current field. "
            "White loops are neutral currents, orange loops are warm-current branches, and cyan loops are cold/upwelling branches. "
            "In seasonal_v3/v4 the background classes come from the basin-aware ocean-current layer."
        ),
        items=[
            ("land", (118, 130, 94)),
            ("equatorial current background", (75, 155, 215)),
            ("subtropical gyre background", (45, 105, 175)),
            ("subpolar gyre background", (85, 130, 190)),
            ("polar current background", (145, 180, 210)),
            ("warm current loop", (250, 165, 80)),
            ("cold current loop", (85, 205, 245)),
            ("upwelling branch", (175, 235, 245)),
            ("neutral current loop", (245, 245, 235)),
        ],
        stats={"driver_key": "ocean_gyre_class", "climate_mode": getattr(profile.climate, "climate_mode", "legacy")},
        scale=_driver_scale_meta(profile, cls, kind="ocean_gyre_class"),
    )


_OCEAN_PATH_CLASS_ITEMS = {
    1: ("equatorial westward current", (70, 150, 220)),
    2: ("equatorial countercurrent", (235, 225, 110)),
    3: ("subtropical gyre interior", (45, 105, 175)),
    4: ("western-boundary warm current", (245, 145, 65)),
    5: ("eastern-boundary cold current", (75, 205, 235)),
    6: ("subpolar gyre", (95, 135, 195)),
    7: ("coastal upwelling", (175, 235, 245)),
    8: ("weak / blocked flow", (80, 95, 125)),
}


def save_main_planet_ocean_current_paths_view(system: StarSystem, output_path: str | Path) -> None:
    return _save_driver_class(
        system,
        output_path,
        key="ocean_current_path_class",
        title="Seasonal_v3 ocean current path classes",
        description=(
            "Categorical current-path diagnostic for the structured ocean layer: equatorial currents, "
            "countercurrents, subtropical gyres, western-boundary warm currents, eastern-boundary cold currents, "
            "subpolar gyres, upwelling zones, and weak/blocked flow."
        ),
        class_items=_OCEAN_PATH_CLASS_ITEMS,
        water_background=(35, 72, 120),
        land_background=(118, 130, 94),
    )


def save_main_planet_ocean_basins_view(system: StarSystem, output_path: str | Path) -> None:
    if system.main_planet_profile is None:
        raise RuntimeError("Ocean basin diagnostic requested, but no Main Planet profile exists.")
    import numpy as np
    profile = system.main_planet_profile
    arr_i = _climate_driver_array(profile, "ocean_basin_id")
    if arr_i is None:
        raise RuntimeError("Ocean basin ID map is not available.")
    basin = arr_i.astype(np.int32)
    kind_arr = _climate_driver_array(profile, "ocean_basin_kind")
    kind = kind_arr.astype(np.int16) if kind_arr is not None else np.where(basin > 0, 1, 0).astype(np.int16)
    land = _driver_land_mask(profile, basin.shape)
    rgb = np.zeros((*basin.shape, 3), dtype=np.uint8)
    rgb[:, :, :] = (35, 72, 120)
    ids = [int(v) for v in np.unique(basin) if int(v) > 0]
    open_ids = []
    enclosed_ids = []
    for bid in ids:
        mask = basin == bid
        k = int(np.bincount(kind[mask].astype(np.int16).clip(0, 2), minlength=3).argmax()) if np.any(mask) else 0
        if k == 2:
            enclosed_ids.append(bid)
            r = 45 + ((bid * 37) % 70)
            g = 145 + ((bid * 47) % 85)
            b = 180 + ((bid * 29) % 60)
        else:
            open_ids.append(bid)
            r = 55 + ((bid * 67) % 150)
            g = 85 + ((bid * 97) % 135)
            b = 120 + ((bid * 43) % 110)
        rgb[mask] = (r, g, b)
    rgb[land] = (118, 130, 94)
    _save_clean_rgb_map(
        rgb,
        output_path,
        title="Seasonal_v3/v4/v5 ocean basins and enclosed seas",
        description="Ocean basin IDs used by the structured current layer. Open connected world-ocean water is split into semi-isolated basins; closed/enclosed seas are retained as a separate basin type so they do not dominate interpretation of the world ocean.",
        items=[("land", (118, 130, 94)), ("open ocean basin colors", (80, 130, 210)), ("enclosed sea/lake basin colors", (70, 185, 220))],
        stats={"driver_key": "ocean_basin_id", "open_ocean_basin_count": len(open_ids), "enclosed_sea_count": len(enclosed_ids), "total_basin_count": len(ids), "climate_mode": getattr(profile.climate, "climate_mode", "legacy")},
        scale=_driver_scale_meta(profile, basin, kind="ocean_basin_id"),
    )


def save_main_planet_ocean_current_heat_view(system: StarSystem, output_path: str | Path) -> None:
    return _save_driver_scalar(
        system,
        output_path,
        key="current_heat_annual_c_x10",
        divisor=10.0,
        title="Seasonal_v3 ocean current heat transport",
        description="Annual current-carried warm/cold anomaly. Warm western-boundary branches should fade poleward; cold eastern-boundary/upwelling branches should cool adjacent coasts.",
        unit="°C anomaly",
        mask_to_land=False,
        color_stops=[(0.0, (60, 135, 210)), (0.50, (45, 80, 125)), (1.0, (245, 150, 70))],
        fixed_min=-5.5,
        fixed_max=5.5,
        background=(118, 130, 94),
    )


def save_main_planet_coastal_upwelling_view(system: StarSystem, output_path: str | Path) -> None:
    return _save_driver_scalar(
        system,
        output_path,
        key="coastal_upwelling_x1000",
        divisor=1000.0,
        title="Seasonal_v3 coastal upwelling influence",
        description="Land-side coastal upwelling influence spread from cold eastern-boundary current branches. This is one input to coastal desert potential.",
        unit="upwelling index",
        mask_to_land=True,
        color_stops=[(0.0, (230, 220, 150)), (0.55, (120, 190, 160)), (1.0, (70, 185, 230))],
        fixed_min=0.0,
        fixed_max=1.25,
    )


def save_main_planet_warm_current_influence_view(system: StarSystem, output_path: str | Path) -> None:
    return _save_driver_scalar(
        system,
        output_path,
        key="warm_current_influence_x1000",
        divisor=1000.0,
        title="Seasonal_v3 warm-current coastal influence",
        description="Land-side warm-current influence from poleward western-boundary currents. This can warm and moisten nearby coasts.",
        unit="warm-current index",
        mask_to_land=True,
        color_stops=[(0.0, (230, 220, 150)), (0.55, (235, 170, 85)), (1.0, (205, 75, 55))],
        fixed_min=0.0,
        fixed_max=1.25,
    )


def save_main_planet_cold_current_influence_view(system: StarSystem, output_path: str | Path) -> None:
    return _save_driver_scalar(
        system,
        output_path,
        key="cold_current_influence_x1000",
        divisor=1000.0,
        title="Seasonal_v3 cold-current coastal influence",
        description="Land-side cold-current influence from equatorward eastern-boundary currents. This cools/drys nearby coasts and supports coastal deserts.",
        unit="cold-current index",
        mask_to_land=True,
        color_stops=[(0.0, (230, 220, 150)), (0.55, (115, 175, 200)), (1.0, (55, 105, 205))],
        fixed_min=0.0,
        fixed_max=1.25,
    )


def save_main_planet_coastal_desert_potential_view(system: StarSystem, output_path: str | Path) -> None:
    return _save_driver_scalar(
        system,
        output_path,
        key="coastal_desert_potential_x1000",
        divisor=1000.0,
        title="Seasonal_v3 coastal desert potential",
        description="Combined land-side diagnostic from cold current influence, coastal upwelling, subtropical subsidence, and low ITCZ influence. It is a driver, not final Köppen classification.",
        unit="desert-potential index",
        mask_to_land=True,
        color_stops=[(0.0, (105, 175, 105)), (0.45, (225, 195, 105)), (1.0, (210, 120, 55))],
        fixed_min=0.0,
        fixed_max=1.0,
    )


def save_main_planet_trade_wind_moisture_view(system: StarSystem, output_path: str | Path) -> None:
    return _save_driver_scalar(
        system,
        output_path,
        key="trade_wind_moisture_annual_x1000",
        divisor=1000.0,
        title="Seasonal_v5 trade-wind moisture component",
        description="Annual trade-wind moisture component from the component-based seasonal_v5 rainfall pass. It shows tropical/subtropical moisture carried by the trade-wind branch before final precipitation is calculated.",
        unit="trade moisture index",
        color_stops=[(0.0, (85, 95, 130)), (0.45, (105, 180, 170)), (1.0, (215, 235, 185))],
        fixed_min=0.0,
        fixed_max=2.2,
    )


def save_main_planet_monsoon_moisture_view(system: StarSystem, output_path: str | Path) -> None:
    return _save_driver_scalar(
        system,
        output_path,
        key="monsoon_moisture_annual_x1000",
        divisor=1000.0,
        title="Seasonal_v5 monsoon moisture component",
        description="Annual monsoon moisture component from pressure-driven seasonal inflow over warm land. This is a driver layer, not final precipitation.",
        unit="monsoon moisture index",
        mask_to_land=True,
        color_stops=[(0.0, (100, 95, 125)), (0.50, (115, 170, 205)), (1.0, (210, 235, 255))],
        fixed_min=0.0,
        fixed_max=1.8,
    )


def save_main_planet_frontal_moisture_view(system: StarSystem, output_path: str | Path) -> None:
    return _save_driver_scalar(
        system,
        output_path,
        key="frontal_moisture_annual_x1000",
        divisor=1000.0,
        title="Seasonal_v5 frontal/storm-track moisture component",
        description="Annual mid-latitude frontal moisture component coupled to the explicit storm-track driver. It should be smoother than the older hard latitude transition.",
        unit="frontal moisture index",
        color_stops=[(0.0, (70, 80, 115)), (0.50, (125, 175, 210)), (1.0, (225, 235, 245))],
        fixed_min=0.0,
        fixed_max=2.4,
    )


def save_main_planet_orographic_precip_potential_view(system: StarSystem, output_path: str | Path) -> None:
    return _save_driver_scalar(
        system,
        output_path,
        key="orographic_precip_potential_x1000",
        divisor=1000.0,
        title="Seasonal_v5 orographic precipitation potential",
        description="Annual windward-slope rainfall extraction potential from moisture-bearing winds and relief. This diagnostic is intended to show localized inland windward rainfall before rain-shadow drying is applied.",
        unit="orographic precipitation potential",
        mask_to_land=True,
        color_stops=[(0.0, (95, 95, 95)), (0.45, (140, 185, 125)), (1.0, (235, 245, 180))],
        fixed_min=0.0,
        fixed_max=2.5,
    )


def save_main_planet_coastal_dryness_view(system: StarSystem, output_path: str | Path) -> None:
    return _save_driver_scalar(
        system,
        output_path,
        key="coastal_dryness_x1000",
        divisor=1000.0,
        title="Seasonal_v5 coastal dryness driver",
        description="Annual coastal dryness from cold currents, upwelling, and subtropical subsidence after ocean-current feedback is coupled into rainfall. This should highlight coastal-desert candidates.",
        unit="coastal dryness index",
        mask_to_land=True,
        color_stops=[(0.0, (90, 150, 110)), (0.55, (225, 205, 120)), (1.0, (200, 105, 55))],
        fixed_min=0.0,
        fixed_max=1.5,
    )

def _monthly_range(profile, prefix: str, divisor: float, suffix: str = "") -> tuple[float, float]:
    import numpy as np
    vals = []
    for month in range(1, 13):
        key = f"{prefix}_{month:02d}{suffix}"
        arr = _climate_driver_array(profile, key)
        if arr is not None:
            vals.append(arr.astype(np.float32) / divisor)
    if not vals:
        return 0.0, 1.0
    stack = np.stack(vals, axis=0)
    lo = float(np.nanpercentile(stack, 1))
    hi = float(np.nanpercentile(stack, 99))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return 0.0, 1.0
    return lo, hi


def save_main_planet_monthly_climate_maps(system: StarSystem, output_dir: str | Path) -> None:
    """Write grouped monthly temperature/precipitation maps for UI progression."""
    import json
    from pathlib import Path
    if system.main_planet_profile is None:
        raise RuntimeError("Monthly climate maps requested, but no Main Planet profile exists.")
    profile = system.main_planet_profile
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    temp_min, temp_max = _monthly_range(profile, "monthly_temperature", 10.0, "_c_x10")
    precip_min, precip_max = 0.0, max(75.0, _monthly_range(profile, "monthly_precipitation", 1.0, "_mm")[1])
    sequence = {"schema_version": 1, "kind": "monthly_climate_progression", "months": []}
    for month in range(1, 13):
        label = _MONTH_LABELS[month - 1]
        temp_name = f"main_planet_temperature_month_{month:02d}.png"
        precip_name = f"main_planet_precipitation_month_{month:02d}.png"
        _save_driver_scalar(
            system,
            out / temp_name,
            key=f"monthly_temperature_{month:02d}_c_x10",
            divisor=10.0,
            title=f"Monthly temperature — {label}",
            description="Synthesized monthly mean temperature from seasonal_v1 anchors. Use the Web UI monthly progression controls to step through the year.",
            unit="°C",
            color_stops=[(0.0, (45, 90, 200)), (0.5, (235, 238, 230)), (1.0, (190, 30, 30))],
            fixed_min=temp_min,
            fixed_max=temp_max,
        )
        _save_driver_scalar(
            system,
            out / precip_name,
            key=f"monthly_precipitation_{month:02d}_mm",
            divisor=1.0,
            title=f"Monthly precipitation — {label}",
            description="Synthesized monthly precipitation from seasonal_v1 anchors. Oceans are masked so land progression is easier to inspect. Colors use a log-scaled display to preserve detail in dry, moderate, and very wet climates.",
            unit="mm/month",
            mask_to_land=True,
            color_stops=[(0.0, (245, 225, 170)), (0.20, (201, 211, 118)), (0.45, (106, 182, 110)), (0.70, (76, 145, 169)), (1.0, (54, 87, 167))],
            fixed_min=precip_min,
            fixed_max=precip_max,
            display_transform="log1p",
        )
        sequence["months"].append({"month": month, "label": label, "temperature_map": temp_name, "precipitation_map": precip_name})
    (out / "monthly_sequence.json").write_text(json.dumps(sequence, indent=2), encoding="utf-8")


def save_main_planet_small_lake_neutral_buffer_view(system: StarSystem, output_path: str | Path) -> None:
    return _save_driver_scalar(
        system,
        output_path,
        key="small_lake_neutral_buffer_x1000",
        divisor=1000.0,
        title="Seasonal_v1 small-lake neutral buffer",
        description=(
            "Very local neutral humidity buffer around tiny inland lakes. It is designed to prevent artificial dry moats "
            "without allowing small desert lakes to create wet climate zones."
        ),
        unit="buffer index",
        mask_to_land=True,
        color_stops=[(0.0, (145, 130, 105)), (0.45, (120, 170, 165)), (1.0, (115, 205, 220))],
        fixed_min=0.0,
        fixed_max=1.0,
    )

# ---------------------------------------------------------------------------
# Update 10 final overrides: clearer system overview figures.
# ---------------------------------------------------------------------------


def save_system_orbit_map(system: StarSystem, output_path: str | Path) -> None:  # type: ignore[override]
    """Save a more readable orbital architecture map with an inner-system inset."""
    from PIL import Image, ImageDraw
    import math

    output_path = _prepare_output_path(output_path)
    planets = sorted(system.planets, key=lambda p: p.orbit.semi_major_axis_au)
    star = system.star
    width, height = 1700, 1120
    map_size = 1060
    cx, cy = map_size // 2 + 40, map_size // 2 + 20
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    outer = max([p.orbit.semi_major_axis_au for p in planets] + [star.snow_line_au, star.habitable_zone_outer_au, 1.0])
    scale = (map_size * 0.43) / max(0.001, outer * 1.08)

    def rr(au: float) -> int:
        return int(round(au * scale))

    draw.text((28, 18), f"Star System Orbital Map - Seed {system.seed}", fill=(0, 0, 0))
    draw.text((28, 46), "Orbits are to AU scale; bodies are symbolic. Inner inset separates close planets from the star.", fill=(80, 80, 80))

    # Habitable-zone annulus and frost line.
    outer_r = rr(star.habitable_zone_outer_au)
    inner_r = rr(star.habitable_zone_inner_au)
    draw.ellipse((cx - outer_r, cy - outer_r, cx + outer_r, cy + outer_r), fill=(219, 242, 218), outline=None)
    draw.ellipse((cx - inner_r, cy - inner_r, cx + inner_r, cy + inner_r), fill="white", outline=None)
    _draw_dashed_circle(draw, cx, cy, rr(star.snow_line_au), fill=(40, 115, 190))

    golden = math.radians(137.507764)
    min_label_gap = 24
    used_label_y: list[int] = []
    for i, p in enumerate(planets, start=1):
        r = rr(p.orbit.semi_major_axis_au)
        draw.ellipse((cx - r, cy - r, cx + r, cy + r), outline=(176, 184, 194), width=1)
        a = golden * i
        x = int(cx + r * math.cos(a)); y = int(cy + r * math.sin(a))
        color = _hex_to_rgb(PLANET_CLASS_COLORS.get(p.planet_class, "#777777"))
        rad = 7 if not p.is_main_planet else 11
        draw.ellipse((x - rad, y - rad, x + rad, y + rad), fill=color, outline=(15, 15, 15), width=3 if p.is_main_planet else 1)
        lx = x + (16 if x < cx else -170)
        ly = y - 8
        while any(abs(ly - old) < min_label_gap for old in used_label_y):
            ly += min_label_gap
        used_label_y.append(ly)
        label = f"{i}: {p.name}" + ("  MAIN" if p.is_main_planet else "")
        draw.line((x, y, lx + (0 if lx > x else 156), ly + 7), fill=(150, 150, 150), width=1)
        draw.text((lx, ly), label, fill=(0, 0, 0))

    # Smaller symbolic star to avoid hiding inner planets.
    draw.ellipse((cx - 13, cy - 13, cx + 13, cy + 13), fill=(255, 211, 77), outline=(120, 90, 20), width=2)
    draw.text((cx + 18, cy - 9), f"{star.stellar_class} star", fill=(0, 0, 0))

    # Inner inset, scaled to the innermost several planets / HZ inner edge.
    inner_planets = planets[: min(5, len(planets))]
    if inner_planets:
        ix0, iy0, isz = 1120, 80, 520
        icx, icy = ix0 + isz // 2, iy0 + isz // 2
        draw.rectangle((ix0, iy0, ix0 + isz, iy0 + isz), outline=(90, 100, 120), width=2)
        draw.text((ix0 + 12, iy0 + 10), "Inner-system inset", fill=(0, 0, 0))
        inner_outer = max([p.orbit.semi_major_axis_au for p in inner_planets] + [star.habitable_zone_inner_au, 0.08])
        iscale = (isz * 0.41) / max(0.001, inner_outer * 1.12)
        def ir(au: float) -> int:
            return int(round(au * iscale))
        hz_o = min(isz//2-8, ir(star.habitable_zone_outer_au))
        hz_i = min(isz//2-8, ir(star.habitable_zone_inner_au))
        if hz_o > 2:
            draw.ellipse((icx - hz_o, icy - hz_o, icx + hz_o, icy + hz_o), fill=(232, 246, 231), outline=None)
            draw.ellipse((icx - hz_i, icy - hz_i, icx + hz_i, icy + hz_i), fill="white", outline=None)
        for i, p in enumerate(inner_planets, start=1):
            r = ir(p.orbit.semi_major_axis_au)
            draw.ellipse((icx - r, icy - r, icx + r, icy + r), outline=(190, 190, 190), width=1)
            a = golden * (i + 1.7)
            x = int(icx + r * math.cos(a)); y = int(icy + r * math.sin(a))
            color = _hex_to_rgb(PLANET_CLASS_COLORS.get(p.planet_class, "#777777"))
            rad = 8 if p.is_main_planet else 6
            draw.ellipse((x - rad, y - rad, x + rad, y + rad), fill=color, outline=(0, 0, 0), width=2 if p.is_main_planet else 1)
            draw.text((x + 10, y - 6), str(i), fill=(0, 0, 0))
        draw.ellipse((icx - 9, icy - 9, icx + 9, icy + 9), fill=(255, 211, 77), outline=(120, 90, 20), width=2)

    # Side summary.
    sx, sy = 1120, 635
    lines = [
        f"Star: {star.stellar_class} | {star.mass_solar:.3f} M_sun | {star.radius_solar:.3f} R_sun",
        f"Luminosity {star.luminosity_solar:.3f} L_sun | Temp {star.temperature_k:.0f} K | Age {star.age_gyr:.2f} Gyr",
        f"HZ {star.habitable_zone_inner_au:.3f}-{star.habitable_zone_outer_au:.3f} AU | frost line {star.snow_line_au:.3f} AU",
        "",
        "Planets:",
    ]
    for i, p in enumerate(planets, start=1):
        mark = " MAIN" if p.is_main_planet else ""
        lines.append(f"{i}. {p.name}{mark}: {p.orbit.semi_major_axis_au:.3f} AU, {p.radius_earth:.2f} R_earth, {p.mass_earth:.2f} M_earth")
    for line in lines[:18]:
        draw.text((sx, sy), line, fill=(0, 0, 0))
        sy += 24

    draw.rectangle((28, height - 82, 48, height - 62), fill=(219, 242, 218), outline=(80, 120, 80)); draw.text((56, height - 80), "habitable zone", fill=(0,0,0))
    draw.line((230, height - 72, 280, height - 72), fill=(40, 115, 190), width=2); draw.text((290, height - 80), "snow/frost line", fill=(0,0,0))
    draw.text((500, height - 80), "Main Planet has a thicker outline and MAIN label.", fill=(80,80,80))
    _save_image_fast(img, output_path)
    _write_map_legend_sidecar(
        output_path,
        title="Star system orbital map",
        description="Readable orbital overview with a zoomed inner-system inset. Orbits are scaled by AU; star/planet marker sizes are symbolic so close inner planets are not hidden.",
        items=[("habitable zone", (219, 242, 218)), ("snow/frost line", (40, 115, 190)), ("symbolic planets", (120, 160, 210)), ("Main Planet outline", (20, 20, 20))],
        stats={"seed": system.seed, "planet_count": len(system.planets), "architecture": getattr(system, "architecture", "unspecified"), "inner_inset_planets": len(inner_planets)},
        notes=["Overview figure; not a terrain raster.", "Use the side summary and system.json for exact numeric values."],
    )


def save_system_size_chart(system: StarSystem, output_path: str | Path) -> None:  # type: ignore[override]
    """Save a non-clipped, non-overlapping star/planet size comparison."""
    from PIL import Image, ImageDraw
    import math as _math

    output_path = _prepare_output_path(output_path)
    star = system.star
    planets = sorted(system.planets, key=lambda p: p.orbit.semi_major_axis_au)

    # Update 10B: this figure must grow when the system is crowded instead of
    # forcing all bodies onto one row.  The image uses a fixed star panel plus a
    # wrapped planet grid and a compact data table.  Star/planet symbols remain
    # separately scaled because real stellar radii would make planets invisible.
    n = max(1, len(planets))
    cell_w = 210
    left_margin = 48
    right_margin = 48
    star_panel_w = 620
    max_cols = 6 if n <= 10 else 7
    cols = min(max_cols, n)
    rows = int(_math.ceil(n / cols))
    planet_panel_w = cols * cell_w + 80
    width = max(1500, star_panel_w + planet_panel_w + left_margin + right_margin)
    planet_top = 120
    row_h = 235
    table_top = planet_top + rows * row_h + 50
    table_rows = int(_math.ceil(n / 2))
    height = max(860, table_top + 56 + table_rows * 24 + 50)

    img = Image.new("RGB", (int(width), int(height)), "white")
    draw = ImageDraw.Draw(img)
    draw.text((24, 18), f"Body Size Comparison - Seed {system.seed}", fill=(0, 0, 0))
    draw.text((24, 48), "Star and planets use separate scales; the canvas wraps planets into rows when needed so labels do not overlap.", fill=(80, 80, 80))

    # Star panel, clamped so very large stars do not clip the canvas.
    sx, sy = 175, 260
    sun_r = 92
    gen_r = max(18, min(160, int(star.radius_solar * sun_r)))
    draw.rounded_rectangle((24, 90, star_panel_w - 28, 560), radius=18, fill=(248, 250, 252), outline=(200, 210, 220))
    draw.text((44, 110), "Star scale panel", fill=(0, 0, 0))
    draw.ellipse((sx - sun_r, sy - sun_r, sx + sun_r, sy + sun_r), fill=(253, 210, 77), outline=(160, 100, 0), width=2)
    draw.text((sx - 45, sy + sun_r + 14), "Sun\n1.00 R_sun", fill=(0, 0, 0))
    gx = 440
    draw.ellipse((gx - gen_r, sy - gen_r, gx + gen_r, sy + gen_r), fill=(255, 211, 77), outline=(120, 90, 20), width=2)
    draw.text((gx - 95, sy + max(sun_r, gen_r) + 14), f"Generated {star.stellar_class}\n{star.radius_solar:.2f} R_sun", fill=(0, 0, 0))
    draw.text((44, 490), f"Mass {star.mass_solar:.3f} M_sun | Radius {star.radius_solar:.3f} R_sun", fill=(0, 0, 0))
    draw.text((44, 516), f"Luminosity {star.luminosity_solar:.3f} L_sun | Temp {star.temperature_k:.0f} K", fill=(0, 0, 0))

    # Wrapped planet grid.
    planet_x0 = star_panel_w + 36
    max_r = max([p.radius_earth for p in planets] + [1.0])
    pscale = min(48.0, 72.0 / max(0.1, max_r))
    draw.rounded_rectangle((planet_x0 - 20, 90, width - 28, table_top - 15), radius=18, fill=(248, 250, 252), outline=(200, 210, 220))
    draw.text((planet_x0, 110), "Planet scale panel", fill=(0, 0, 0))
    for i, p in enumerate(planets, start=1):
        col = (i - 1) % cols
        row = (i - 1) // cols
        x = planet_x0 + 58 + col * cell_w
        baseline = planet_top + 82 + row * row_h
        r = max(5, int(p.radius_earth * pscale))
        color = _hex_to_rgb(PLANET_CLASS_COLORS.get(p.planet_class, "#777777"))
        # Cell guide keeps dense systems readable.
        draw.rounded_rectangle((x - 82, baseline - 80, x + 82, baseline + 112), radius=12, outline=(225, 230, 235), fill=(255, 255, 255))
        draw.ellipse((x - r, baseline - r, x + r, baseline + r), fill=color, outline=(15, 15, 15), width=3 if p.is_main_planet else 1)
        draw.text((x - 72, baseline - 72), f"{i}. {p.name}", fill=(0, 0, 0))
        if p.is_main_planet:
            draw.text((x + 24, baseline - 22), "MAIN", fill=(0, 90, 0))
        label_y = baseline + max(76, r + 14)
        draw.text((x - 72, label_y), f"{p.radius_earth:.2f} R_earth", fill=(0, 0, 0))
        draw.text((x - 72, label_y + 22), f"{p.mass_earth:.2f} M_earth", fill=(0, 0, 0))

    # Compact two-column exact-value table at the bottom.
    draw.rounded_rectangle((24, table_top, width - 28, height - 28), radius=18, fill=(248, 250, 252), outline=(200, 210, 220))
    draw.text((44, table_top + 18), "Planet details", fill=(0, 0, 0))
    for i, p in enumerate(planets, start=1):
        col = 0 if i <= table_rows else 1
        r_i = i if col == 0 else i - table_rows
        x = 44 + col * ((width - 96) / 2)
        y = table_top + 44 + (r_i - 1) * 24
        mark = " MAIN" if p.is_main_planet else ""
        txt = f"{i}. {p.name}{mark}: {p.orbit.semi_major_axis_au:.3f} AU | {p.radius_earth:.2f} R⊕ | {p.mass_earth:.2f} M⊕ | {p.planet_class}"
        draw.text((int(x), int(y)), txt, fill=(0, 0, 0))

    _save_image_fast(img, output_path)
    _write_map_legend_sidecar(
        output_path,
        title="System body size comparison",
        description="Non-clipped star/planet comparison. The canvas expands and wraps crowded planet systems into rows; exact values are shown in the image and sidebar instead of relying on a generic map legend.",
        items=[("star size markers", (253, 210, 77)), ("planet size markers", (120, 160, 210)), ("Main Planet outline", (20, 20, 20))],
        stats={"seed": system.seed, "planet_count": len(system.planets), "star_class": star.stellar_class, "star_radius_solar": round(star.radius_solar, 3), "star_mass_solar": round(star.mass_solar, 3)},
        notes=["Overview figure; not a terrain raster.", "Star and planet scales are separate to keep all bodies visible.", "Update 10B wraps dense systems instead of overlapping labels."],
    )
