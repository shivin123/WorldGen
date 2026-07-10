"""Structured basin-aware ocean circulation helpers for seasonal_v3 climate.

This is still a lightweight diagnostic model, not a full ocean solver.  The goal is
for ocean circulation to become a first-class climate driver with explicit basin
IDs, current-branch classes, heat transport, upwelling, and coastal influence maps.
"""

from __future__ import annotations

from dataclasses import dataclass

from worldgen.physics.map_scale import map_scale_for_terrain


@dataclass(frozen=True)
class OceanCirculationState:
    current_u: object
    current_v: object
    current_heat: object
    ocean_basin_id: object
    ocean_current_path_class: object
    ocean_gyre_class: object
    coastal_upwelling: object
    warm_current_influence: object
    cold_current_influence: object
    coastal_desert_potential: object


def build_ocean_circulation_v2(*, np, wind_u, wind_v, land, ocean_like, lat_grid, abs_lat, dist_km, terrain, hadley_edge: float, itcz_strength=None) -> OceanCirculationState:
    """Build basin-aware surface current and coastal influence diagnostics.

    Class encodings:
      ocean_current_path_class:
        0 land/non-ocean, 1 equatorial current, 2 equatorial countercurrent,
        3 subtropical gyre interior, 4 western-boundary warm current,
        5 eastern-boundary cold current, 6 subpolar gyre, 7 upwelling,
        8 weak/blocked flow.
      ocean_gyre_class:
        0 land/non-ocean, 1 equatorial, 2 subtropical, 3 subpolar, 4 polar,
        5 poleward warm branch, 6 equatorward cold branch, 7 upwelling.
    """
    h, w = ocean_like.shape
    ocean = np.asarray(ocean_like, dtype=bool)
    land = np.asarray(land, dtype=bool)
    if itcz_strength is None:
        itcz_strength = np.zeros((h, w), dtype=np.float32)

    basin_id = _label_ocean_basins(np, ocean)
    x_norm = _zonal_basin_coordinate(np, basin_id, ocean)

    hemi = np.where(lat_grid >= 0.0, 1.0, -1.0).astype(np.float32)
    poleward_v = np.where(lat_grid >= 0.0, -1.0, 1.0).astype(np.float32)
    equatorward_v = -poleward_v
    coslat = np.clip(np.cos(np.radians(lat_grid)), 0.0, 1.0).astype(np.float32)

    # Basin-relative boundary weights.  Western boundary = west side of an ocean
    # basin, usually next to the east coast of a continent.  Eastern boundary =
    # east side of an ocean basin, usually next to the west coast of a continent.
    western_edge = np.exp(-((x_norm - 0.04) / 0.18) ** 2).astype(np.float32)
    eastern_edge = np.exp(-((x_norm - 0.96) / 0.22) ** 2).astype(np.float32)
    interior_west = np.clip(1.0 - x_norm, 0.0, 1.0).astype(np.float32)
    interior_east = np.clip(x_norm, 0.0, 1.0).astype(np.float32)

    # Land adjacency sharpens coastal branches and supports upwelling detection.
    west_land = _directional_land_proximity(np, land, direction="west", max_cells=6)
    east_land = _directional_land_proximity(np, land, direction="east", max_cells=6)
    north_land = _directional_land_proximity(np, land, direction="north", max_cells=3)
    south_land = _directional_land_proximity(np, land, direction="south", max_cells=3)
    coastal_edge = np.clip(np.maximum.reduce([west_land, east_land, north_land, south_land]), 0.0, 1.0) * ocean

    equatorial_band = np.exp(-(((abs_lat - 9.0) / 7.5) ** 2)).astype(np.float32)
    counter_band = np.exp(-((lat_grid / 3.8) ** 2)).astype(np.float32)
    subtropical_band = np.exp(-(((abs_lat - 28.0) / 14.5) ** 2)).astype(np.float32)
    westerly_return_band = np.exp(-(((abs_lat - 39.0) / 9.5) ** 2)).astype(np.float32)
    subpolar_band = np.exp(-(((abs_lat - 57.0) / 9.5) ** 2)).astype(np.float32)
    polar_band = np.clip((abs_lat - 66.0) / 16.0, 0.0, 1.0).astype(np.float32)

    # Start with named branch currents, then add wind coupling.  u positive = east;
    # v positive = south in the map grid.
    current_u = np.zeros((h, w), dtype=np.float32)
    current_v = np.zeros((h, w), dtype=np.float32)

    # Equatorial circulation: westward trades on both sides of the equator and a
    # narrower eastward countercurrent along the convergence zone.
    current_u += -1.24 * equatorial_band * (1.0 - 0.70 * counter_band)
    current_u += 0.92 * counter_band

    # Subtropical gyre loop: westward low-latitude branch, eastward westerly return,
    # poleward western boundary, equatorward eastern boundary.
    current_u += -0.64 * equatorial_band * subtropical_band
    current_u += 0.94 * westerly_return_band * subtropical_band
    current_v += poleward_v * (1.20 * western_edge + 0.28 * interior_west) * subtropical_band
    current_v += equatorward_v * (0.88 * eastern_edge + 0.20 * interior_east) * subtropical_band

    # Subpolar gyres are weaker and usually rotate opposite the subtropical gyre.
    current_u += 0.58 * subpolar_band * (0.55 - x_norm)
    current_v += equatorward_v * (0.42 * western_edge) * subpolar_band
    current_v += poleward_v * (0.36 * eastern_edge) * subpolar_band
    current_u += 0.18 * polar_band * hemi

    # Surface wind coupling keeps the ocean tied to the atmospheric state but does
    # not dominate the named branches.
    current_u += 0.20 * np.asarray(wind_u, dtype=np.float32)
    current_v += 0.16 * np.asarray(wind_v, dtype=np.float32)

    # Intensify appropriate continental boundary currents.  Western-boundary warm
    # currents are usually poleward; eastern-boundary cold currents equatorward.
    western_boundary = np.clip(0.55 * western_edge + 0.70 * west_land, 0.0, 1.6) * subtropical_band * ocean
    eastern_boundary = np.clip(0.55 * eastern_edge + 0.70 * east_land, 0.0, 1.6) * subtropical_band * ocean
    current_v += 0.78 * western_boundary * poleward_v
    current_v += 0.70 * eastern_boundary * equatorward_v
    current_u += 0.13 * (east_land - west_land) * coastal_edge

    # Terrain/world-size aware smoothing.  Keep land as a hard barrier, but remove
    # striping and sharp current discontinuities inside basins.
    for _ in range(6):
        neigh_u = _masked_neighbor_mean(np, current_u, ocean)
        neigh_v = _masked_neighbor_mean(np, current_v, ocean)
        current_u = np.where(ocean, 0.77 * current_u + 0.23 * neigh_u, 0.0)
        current_v = np.where(ocean, 0.77 * current_v + 0.23 * neigh_v, 0.0)

    speed = np.sqrt(current_u * current_u + current_v * current_v)
    current_u = np.where(ocean, current_u, 0.0).astype(np.float32)
    current_v = np.where(ocean, current_v, 0.0).astype(np.float32)

    poleward_strength = np.clip(current_v * poleward_v, 0.0, 2.5)
    equatorward_strength = np.clip(current_v * equatorward_v, 0.0, 2.5)
    warm_branch = np.clip(western_boundary * poleward_strength, 0.0, 2.6)
    cold_branch = np.clip(eastern_boundary * equatorward_strength, 0.0, 2.6)
    equatorial_warm = np.clip(-current_u, 0.0, 2.0) * np.exp(-((abs_lat - 8.0) / 10.0) ** 2)

    upwelling = np.clip(cold_branch * np.exp(-(((abs_lat - 22.0) / 12.0) ** 2)) * (0.55 + 0.45 * east_land), 0.0, 1.0)

    heat_source = np.where(ocean, 2.9 * warm_branch + 0.50 * equatorial_warm - 2.7 * cold_branch - 2.0 * upwelling, 0.0).astype(np.float32)
    current_heat = _advect_ocean_scalar(np, heat_source, current_u, current_v, ocean, iterations=38, decay=0.988)
    # Poleward warm branches lose heat; equatorward cold branches retain a stronger
    # cold anomaly.  This makes the heat map more path-like than source-like.
    current_heat = current_heat - 0.018 * abs_lat * np.clip(poleward_strength, 0.0, 1.8)
    current_heat = np.where(ocean, np.clip(current_heat, -6.5, 6.5), 0.0).astype(np.float32)

    warm_influence = _spread_ocean_to_coastal_land(np, np.clip(current_heat / 3.2, 0.0, 1.6), land, ocean, iterations=18)
    cold_influence = _spread_ocean_to_coastal_land(np, np.clip(-current_heat / 3.2, 0.0, 1.6), land, ocean, iterations=18)
    upwelling_land = _spread_ocean_to_coastal_land(np, upwelling, land, ocean, iterations=16)

    subtropical_high = np.exp(-(((abs_lat - hadley_edge) / 10.0) ** 2)).astype(np.float32)
    itcz_suppression = np.clip(1.0 - 0.58 * np.asarray(itcz_strength, dtype=np.float32), 0.18, 1.0)
    coastal_decay = np.exp(-dist_km / 420.0).astype(np.float32)
    coastal_desert = np.where(
        land,
        np.clip((0.62 * cold_influence + 0.72 * upwelling_land) * subtropical_high * itcz_suppression * coastal_decay, 0.0, 1.0),
        0.0,
    ).astype(np.float32)

    path_class = np.zeros((h, w), dtype=np.int16)
    path_class[ocean & (speed < 0.075)] = 8
    path_class[ocean & (abs_lat >= 45.0) & (abs_lat < 68.0) & (speed >= 0.075)] = 6
    path_class[ocean & (abs_lat >= 8.0) & (abs_lat < 45.0) & (speed >= 0.075)] = 3
    path_class[ocean & (abs_lat < 13.0) & (current_u < -0.18)] = 1
    path_class[ocean & (abs_lat < 5.2) & (current_u > 0.18)] = 2
    path_class[ocean & (warm_branch > 0.12)] = 4
    path_class[ocean & (cold_branch > 0.12)] = 5
    path_class[ocean & (upwelling > 0.18)] = 7

    gyre_class = np.zeros((h, w), dtype=np.int16)
    gyre_class[ocean & (abs_lat < 8.0)] = 1
    gyre_class[ocean & (abs_lat >= 8.0) & (abs_lat < 45.0)] = 2
    gyre_class[ocean & (abs_lat >= 45.0) & (abs_lat < 67.0)] = 3
    gyre_class[ocean & (abs_lat >= 67.0)] = 4
    gyre_class[ocean & (warm_branch > 0.12)] = 5
    gyre_class[ocean & (cold_branch > 0.12)] = 6
    gyre_class[ocean & (upwelling > 0.18)] = 7

    return OceanCirculationState(
        current_u=current_u.astype(np.float32),
        current_v=current_v.astype(np.float32),
        current_heat=current_heat.astype(np.float32),
        ocean_basin_id=basin_id.astype(np.int32),
        ocean_current_path_class=path_class.astype(np.int16),
        ocean_gyre_class=gyre_class.astype(np.int16),
        coastal_upwelling=upwelling_land.astype(np.float32),
        warm_current_influence=warm_influence.astype(np.float32),
        cold_current_influence=cold_influence.astype(np.float32),
        coastal_desert_potential=coastal_desert.astype(np.float32),
    )


def _label_ocean_basins(np, ocean):
    h, w = ocean.shape
    try:
        from scipy import ndimage
        labels, count = ndimage.label(ocean, structure=np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8))
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

        for y in range(h):
            union(labels[y, 0], labels[y, w - 1])
        if count:
            roots = np.zeros(count + 1, dtype=np.int32)
            for i in range(1, count + 1):
                roots[i] = find(i)
            root_grid = roots[labels]
            unique_roots = [int(v) for v in np.unique(root_grid[ocean]) if int(v) != 0]
            remap = {root: idx + 1 for idx, root in enumerate(unique_roots)}
            out = np.zeros((h, w), dtype=np.int32)
            for root, idx in remap.items():
                out[root_grid == root] = idx
            return out
    except Exception:
        pass
    out = np.zeros((h, w), dtype=np.int32)
    if ocean.any():
        out[ocean] = 1
    return out


def _zonal_basin_coordinate(np, basin_id, ocean):
    h, w = basin_id.shape
    out = np.full((h, w), 0.5, dtype=np.float32)
    for y in range(h):
        x = 0
        while x < w:
            label = int(basin_id[y, x])
            if label <= 0:
                x += 1
                continue
            start = x
            x += 1
            while x < w and int(basin_id[y, x]) == label:
                x += 1
            end = x
            length = max(1, end - start)
            if length == 1:
                out[y, start] = 0.5
            else:
                out[y, start:end] = np.linspace(0.0, 1.0, length, dtype=np.float32)
    out[~ocean] = 0.5
    return out


def _directional_land_proximity(np, land, *, direction: str, max_cells: int):
    acc = np.zeros(land.shape, dtype=np.float32)
    for d in range(1, max_cells + 1):
        weight = (max_cells + 1 - d) / float(max_cells)
        if direction == "west":
            shifted = np.roll(land, d, axis=1)
        elif direction == "east":
            shifted = np.roll(land, -d, axis=1)
        elif direction == "north":
            shifted = np.vstack([land[0:1, :], land[:-1, :]]) if d == 1 else np.vstack([land[:d, :], land[:-d, :]])
        elif direction == "south":
            shifted = np.vstack([land[d:, :], land[-d:, :]]) if d < land.shape[0] else np.repeat(land[-1:, :], land.shape[0], axis=0)
        else:  # pragma: no cover - defensive guard
            shifted = land
        acc = np.maximum(acc, shifted.astype(np.float32) * weight)
    return np.clip(acc, 0.0, 1.0)


def _masked_neighbor_mean(np, arr, mask):
    vals = []
    weights = []
    for shifted_arr, shifted_mask in [
        (np.roll(arr, 1, axis=1), np.roll(mask, 1, axis=1)),
        (np.roll(arr, -1, axis=1), np.roll(mask, -1, axis=1)),
        (np.vstack([arr[0:1, :], arr[:-1, :]]), np.vstack([mask[0:1, :], mask[:-1, :]])),
        (np.vstack([arr[1:, :], arr[-1:, :]]), np.vstack([mask[1:, :], mask[-1:, :]])),
    ]:
        vals.append(np.where(shifted_mask, shifted_arr, 0.0))
        weights.append(shifted_mask.astype(np.float32))
    total_w = np.maximum(1.0, sum(weights))
    return (sum(vals) / total_w).astype(np.float32)


def _advect_ocean_scalar(np, source, current_u, current_v, ocean, *, iterations: int, decay: float):
    field = np.where(ocean, source, 0.0).astype(np.float32)
    source = field.copy()
    for _ in range(max(1, int(iterations))):
        west = np.roll(field, 1, axis=1)
        east = np.roll(field, -1, axis=1)
        north = np.vstack([field[0:1, :], field[:-1, :]])
        south = np.vstack([field[1:, :], field[-1:, :]])
        wu_pos = np.clip(current_u, 0.0, None)
        wu_neg = np.clip(-current_u, 0.0, None)
        wv_pos = np.clip(current_v, 0.0, None)
        wv_neg = np.clip(-current_v, 0.0, None)
        total = wu_pos + wu_neg + wv_pos + wv_neg + 0.35
        upstream = (west * wu_pos + east * wu_neg + north * wv_pos + south * wv_neg + field * 0.35) / total
        field = np.where(ocean, 0.30 * source + 0.70 * upstream * decay, 0.0)
    return field.astype(np.float32)


def _spread_ocean_to_coastal_land(np, ocean_source, land, ocean, *, iterations: int):
    field = np.where(ocean, ocean_source, 0.0).astype(np.float32)
    land_field = np.zeros_like(field, dtype=np.float32)
    active = ocean | land
    for _ in range(max(1, int(iterations))):
        neigh = (
            np.roll(field, 1, axis=1)
            + np.roll(field, -1, axis=1)
            + np.vstack([field[0:1, :], field[:-1, :]])
            + np.vstack([field[1:, :], field[-1:, :]])
        ) / 4.0
        land_field = np.where(land, np.maximum(land_field * 0.91, neigh * 0.86), 0.0)
        field = np.where(ocean, ocean_source, land_field)
        field = np.where(active, field, 0.0)
    return np.clip(land_field, 0.0, 2.0).astype(np.float32)
