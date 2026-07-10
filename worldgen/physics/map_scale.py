"""Map-scale helpers for equirectangular planetary rasters.

WorldGen maps are stored as 2:1 equirectangular grids. Several models need to
convert a distance measured in cells into a physical distance on the generated
planet. Keeping that conversion in one small module avoids accidentally tuning
climate or diagnostics to a particular map resolution.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from worldgen.constants import EARTH_RADIUS_KM

REFERENCE_MAP_WIDTH = 4096
REFERENCE_MAP_HEIGHT = 2048
REFERENCE_KM_PER_CELL = math.pi * EARTH_RADIUS_KM / REFERENCE_MAP_HEIGHT


@dataclass(frozen=True)
class MapScale:
    planet_radius_earth: float
    planet_radius_km: float
    equatorial_km_per_cell: float
    meridional_km_per_cell: float
    representative_km_per_cell: float
    reference_km_per_cell: float = REFERENCE_KM_PER_CELL


def planet_radius_earth_from_terrain(terrain) -> float:
    """Return the planet radius stored on a TerrainMap, falling back to Earth."""
    try:
        radius = float(getattr(terrain, "planet_radius_earth", 1.0) or 1.0)
    except (TypeError, ValueError):
        radius = 1.0
    if not math.isfinite(radius) or radius <= 0.0:
        radius = 1.0
    return radius


def map_scale_for_terrain(terrain) -> MapScale:
    """Physical cell scale for a generated terrain grid."""
    radius_earth = planet_radius_earth_from_terrain(terrain)
    radius_km = radius_earth * EARTH_RADIUS_KM
    width = max(1, int(getattr(terrain, "width", 1) or 1))
    height = max(1, int(getattr(terrain, "height", 1) or 1))
    equatorial = 2.0 * math.pi * radius_km / width
    meridional = math.pi * radius_km / height
    # Most raster distance transforms count row/column cells on a 2:1 grid, so
    # the meridional/equatorial-at-equator spacing is a good global first-order
    # scale. Latitude-aware geodesic distances can replace this later.
    representative = 0.5 * (equatorial + meridional)
    return MapScale(
        planet_radius_earth=radius_earth,
        planet_radius_km=radius_km,
        equatorial_km_per_cell=equatorial,
        meridional_km_per_cell=meridional,
        representative_km_per_cell=representative,
    )


def reference_cells_to_km(reference_cells: float) -> float:
    """Convert old 4096x2048 Earth-tuned cell constants to kilometers."""
    return float(reference_cells) * REFERENCE_KM_PER_CELL


def km_to_cells(terrain, km: float, *, minimum: int = 1, maximum: int | None = None) -> int:
    """Convert a physical kilometer distance into cells for this terrain."""
    scale = map_scale_for_terrain(terrain)
    cells = int(round(float(km) / max(1e-9, scale.representative_km_per_cell)))
    cells = max(int(minimum), cells)
    if maximum is not None:
        cells = min(int(maximum), cells)
    return cells
