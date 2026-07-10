"""Refined basin-aware ocean circulation helpers for seasonal_v4 climate.

This is still a lightweight diagnostic model, not a full ocean solver.  The goal is
to refine the first-class ocean driver with basin
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
    ocean_basin_kind: object
    ocean_current_path_class: object
    ocean_gyre_class: object
    coastal_upwelling: object
    warm_current_influence: object
    cold_current_influence: object
    coastal_desert_potential: object


def build_ocean_circulation_v3(*, np, wind_u, wind_v, land, ocean_like, lat_grid, abs_lat, dist_km, terrain, hadley_edge: float, itcz_strength=None, moisture_wind_u=None, moisture_wind_v=None) -> OceanCirculationState:
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
    if moisture_wind_u is None:
        moisture_wind_u = wind_u
    if moisture_wind_v is None:
        moisture_wind_v = wind_v
    wind_u = np.asarray(wind_u, dtype=np.float32)
    wind_v = np.asarray(wind_v, dtype=np.float32)
    moisture_wind_u = np.asarray(moisture_wind_u, dtype=np.float32)
    moisture_wind_v = np.asarray(moisture_wind_v, dtype=np.float32)

    basin_id, basin_kind = _label_ocean_basins(np, ocean)
    x_norm = _zonal_basin_coordinate(np, basin_id, ocean)
    basin_width_cells = _zonal_basin_width_cells(np, basin_id, ocean)
    openness = _ocean_openness(np, ocean)
    strait_factor = np.clip((basin_width_cells - 2.0) / 10.0, 0.16, 1.0).astype(np.float32)
    openness_factor = np.clip(0.35 + 0.65 * openness, 0.20, 1.0).astype(np.float32)
    routing_factor = np.clip(strait_factor * openness_factor, 0.12, 1.0).astype(np.float32)

    hemi = np.where(lat_grid >= 0.0, 1.0, -1.0).astype(np.float32)
    poleward_v = np.where(lat_grid >= 0.0, -1.0, 1.0).astype(np.float32)
    equatorward_v = -poleward_v
    coslat = np.clip(np.cos(np.radians(lat_grid)), 0.0, 1.0).astype(np.float32)

    # Basin-relative boundary weights.  Western boundary = west side of an ocean
    # basin, usually next to the east coast of a continent.  Eastern boundary =
    # east side of an ocean basin, usually next to the west coast of a continent.
    western_edge = np.exp(-((x_norm - 0.045) / 0.16) ** 2).astype(np.float32)
    eastern_edge = np.exp(-((x_norm - 0.955) / 0.18) ** 2).astype(np.float32)
    interior_west = np.clip(1.0 - x_norm, 0.0, 1.0).astype(np.float32)
    interior_east = np.clip(x_norm, 0.0, 1.0).astype(np.float32)
    gyre_core = np.clip(np.sin(np.pi * np.clip(x_norm, 0.0, 1.0)), 0.0, 1.0).astype(np.float32)

    # Land adjacency sharpens coastal branches and supports upwelling detection.
    west_land = _directional_land_proximity(np, land, direction="west", max_cells=8)
    east_land = _directional_land_proximity(np, land, direction="east", max_cells=8)
    north_land = _directional_land_proximity(np, land, direction="north", max_cells=4)
    south_land = _directional_land_proximity(np, land, direction="south", max_cells=4)
    coastal_edge = np.clip(np.maximum.reduce([west_land, east_land, north_land, south_land]), 0.0, 1.0) * ocean

    equatorial_band = np.exp(-(((abs_lat - 9.0) / 7.8) ** 2)).astype(np.float32)
    counter_band = np.exp(-((lat_grid / 3.9) ** 2)).astype(np.float32)
    subtropical_band = np.exp(-(((abs_lat - 29.0) / 15.5) ** 2)).astype(np.float32)
    westerly_return_band = np.exp(-(((abs_lat - 40.0) / 10.5) ** 2)).astype(np.float32)
    subpolar_band = np.exp(-(((abs_lat - 57.0) / 10.5) ** 2)).astype(np.float32)
    polar_band = np.clip((abs_lat - 66.0) / 16.0, 0.0, 1.0).astype(np.float32)

    # Start with named branch currents, then add wind coupling.  u positive = east;
    # v positive = south in the map grid.
    current_u = np.zeros((h, w), dtype=np.float32)
    current_v = np.zeros((h, w), dtype=np.float32)

    # Equatorial circulation: westward trades on both sides of the equator and a
    # narrower eastward countercurrent along the convergence zone.
    current_u += (-1.28 * equatorial_band * (1.0 - 0.66 * counter_band) + 0.94 * counter_band) * (0.58 + 0.42 * routing_factor)

    # Subtropical gyre loop: westward low-latitude branch, eastward westerly return,
    # poleward western boundary, equatorward eastern boundary.  The gyre-core
    # factor keeps the interior branch broad while the routing factor weakens
    # currents in narrow straits and behind islands.
    current_u += -0.78 * equatorial_band * subtropical_band * (0.45 + 0.55 * gyre_core)
    current_u += 1.02 * westerly_return_band * subtropical_band * (0.42 + 0.58 * gyre_core)
    current_v += poleward_v * (1.36 * western_edge + 0.24 * interior_west * gyre_core) * subtropical_band
    current_v += equatorward_v * (1.02 * eastern_edge + 0.18 * interior_east * gyre_core) * subtropical_band

    # Subpolar gyres are weaker and usually rotate opposite the subtropical gyre.
    current_u += 0.66 * subpolar_band * (0.55 - x_norm) * (0.55 + 0.45 * gyre_core)
    current_v += equatorward_v * (0.48 * western_edge) * subpolar_band
    current_v += poleward_v * (0.42 * eastern_edge) * subpolar_band
    current_u += 0.18 * polar_band * hemi

    # Surface wind coupling keeps the ocean tied to the atmospheric state but does
    # not dominate the named branches.
    current_u += 0.18 * wind_u + 0.06 * moisture_wind_u
    current_v += 0.14 * wind_v + 0.05 * moisture_wind_v

    current_u *= routing_factor
    current_v *= routing_factor

    # Intensify appropriate continental boundary currents.  Western-boundary warm
    # currents are usually poleward; eastern-boundary cold currents equatorward.
    western_boundary = np.clip(0.58 * western_edge + 0.82 * west_land, 0.0, 1.8) * subtropical_band * ocean * np.clip(0.55 + 0.45 * openness, 0.35, 1.0)
    eastern_boundary = np.clip(0.58 * eastern_edge + 0.82 * east_land, 0.0, 1.8) * subtropical_band * ocean * np.clip(0.50 + 0.50 * openness, 0.30, 1.0)
    current_v += 0.92 * western_boundary * poleward_v
    current_v += 0.84 * eastern_boundary * equatorward_v
    current_u += 0.16 * (east_land - west_land) * coastal_edge * routing_factor

    # Terrain/world-size aware smoothing.  Keep land as a hard barrier, but remove
    # striping and sharp current discontinuities inside basins.
    for _ in range(7):
        neigh_u = _masked_neighbor_mean(np, current_u, ocean)
        neigh_v = _masked_neighbor_mean(np, current_v, ocean)
        current_u = np.where(ocean, (0.80 * current_u + 0.20 * neigh_u) * np.clip(0.92 + 0.08 * routing_factor, 0.86, 1.0), 0.0)
        current_v = np.where(ocean, (0.80 * current_v + 0.20 * neigh_v) * np.clip(0.92 + 0.08 * routing_factor, 0.86, 1.0), 0.0)

    speed = np.sqrt(current_u * current_u + current_v * current_v)
    current_u = np.where(ocean, current_u, 0.0).astype(np.float32)
    current_v = np.where(ocean, current_v, 0.0).astype(np.float32)

    poleward_strength = np.clip(current_v * poleward_v, 0.0, 2.5)
    equatorward_strength = np.clip(current_v * equatorward_v, 0.0, 2.5)
    warm_branch = np.clip(western_boundary * poleward_strength, 0.0, 2.6)
    cold_branch = np.clip(eastern_boundary * equatorward_strength, 0.0, 2.6)
    equatorial_warm = np.clip(-current_u, 0.0, 2.0) * np.exp(-((abs_lat - 8.0) / 10.0) ** 2)

    wind_alongshore = np.clip(np.abs(moisture_wind_v), 0.0, 2.0) * eastern_boundary
    offshore_hint = np.clip((east_land - west_land) * (-moisture_wind_u), 0.0, 2.0) * eastern_boundary
    upwelling = np.clip((0.78 * cold_branch + 0.28 * wind_alongshore + 0.25 * offshore_hint) * np.exp(-(((abs_lat - 23.0) / 13.0) ** 2)) * (0.58 + 0.42 * east_land), 0.0, 1.0)

    equator_heat = np.clip((34.0 - abs_lat) / 34.0, 0.0, 1.0)
    cold_lat_gain = np.clip((abs_lat - 18.0) / 52.0, 0.0, 1.0)
    heat_source = np.where(
        ocean,
        3.35 * warm_branch * (0.75 + 0.55 * equator_heat)
        + 0.62 * equatorial_warm
        - 3.15 * cold_branch * (0.70 + 0.60 * cold_lat_gain)
        - 2.35 * upwelling,
        0.0,
    ).astype(np.float32)
    current_heat = _advect_ocean_scalar(np, heat_source, current_u, current_v, ocean, iterations=54, decay=0.982)
    # Poleward warm branches lose heat while subpolar/eastern-boundary branches keep
    # a clearer cold signature.  This makes heat transport look path-like rather
    # than only source-like.
    current_heat = current_heat - 0.022 * abs_lat * np.clip(poleward_strength, 0.0, 1.8) - 0.38 * upwelling
    current_heat = np.where(ocean, np.clip(current_heat, -6.5, 6.5), 0.0).astype(np.float32)

    warm_influence = _spread_ocean_to_coastal_land(np, np.clip(current_heat / 3.2, 0.0, 1.6), land, ocean, iterations=18)
    cold_influence = _spread_ocean_to_coastal_land(np, np.clip(-current_heat / 3.2, 0.0, 1.6), land, ocean, iterations=18)
    upwelling_land = _spread_ocean_to_coastal_land(np, upwelling, land, ocean, iterations=16)

    subtropical_high = np.exp(-(((abs_lat - hadley_edge) / 10.0) ** 2)).astype(np.float32)
    itcz_suppression = np.clip(1.0 - 0.62 * np.asarray(itcz_strength, dtype=np.float32), 0.16, 1.0)
    coastal_decay = np.exp(-dist_km / 460.0).astype(np.float32)
    offshore_dry_hint = np.where(land, np.clip(cold_influence + 0.8 * upwelling_land, 0.0, 2.0), 0.0)
    coastal_desert = np.where(
        land,
        np.clip((0.66 * cold_influence + 0.82 * upwelling_land + 0.18 * offshore_dry_hint) * subtropical_high * itcz_suppression * coastal_decay, 0.0, 1.0),
        0.0,
    ).astype(np.float32)

    path_class = np.zeros((h, w), dtype=np.int16)
    path_class[ocean & ((speed < 0.075) | (routing_factor < 0.30))] = 8
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
        ocean_basin_kind=basin_kind.astype(np.int16),
        ocean_current_path_class=path_class.astype(np.int16),
        ocean_gyre_class=gyre_class.astype(np.int16),
        coastal_upwelling=upwelling_land.astype(np.float32),
        warm_current_influence=warm_influence.astype(np.float32),
        cold_current_influence=cold_influence.astype(np.float32),
        coastal_desert_potential=coastal_desert.astype(np.float32),
    )


def _label_ocean_basins(np, ocean):
    """Label semi-isolated ocean basins for current routing.

    Earlier seasonal_v3/v4 diagnostics treated the connected world ocean as one
    basin.  That made the basin map look like a single missing legend class and
    gave the current model one global wraparound routing domain.  This function
    still starts from connected water components, but splits very large connected
    oceans into broad semi-isolated basins using land-dominated meridian barriers
    and chokepoint-like columns.  The result is closer to real-world usage: the
    Atlantic/Indian/Pacific are connected, but still useful as separate basins.
    """
    h, w = ocean.shape
    ocean = np.asarray(ocean, dtype=bool)
    base = np.zeros((h, w), dtype=np.int32)
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
            for root, idx in remap.items():
                base[root_grid == root] = idx
    except Exception:
        if ocean.any():
            base[ocean] = 1

    if not np.any(base > 0):
        return base, np.zeros_like(base, dtype=np.int16)

    out = np.zeros((h, w), dtype=np.int32)
    kind = np.zeros((h, w), dtype=np.int16)  # 0 land/non-ocean, 1 open ocean basin, 2 enclosed sea/lake
    next_id = 1
    total_ocean = int(np.sum(ocean))
    major_threshold = max(256, int(total_ocean * 0.18))

    # Avoid polar rows dominating meridian-open-water measures on highly distorted
    # equirectangular maps; ocean basins are mostly separated by continental
    # barriers in low/mid latitudes.
    row_idx = np.arange(h)
    lat = 90.0 - (row_idx + 0.5) * 180.0 / max(1, h)
    mid_rows = np.abs(lat) <= 68.0
    if not np.any(mid_rows):
        mid_rows = np.ones(h, dtype=bool)

    for comp_id in [int(v) for v in np.unique(base) if int(v) > 0]:
        comp = base == comp_id
        comp_count = int(np.sum(comp))
        if comp_count < major_threshold:
            out[comp] = next_id
            kind[comp] = 2
            next_id += 1
            continue

        split = _split_large_ocean_component_by_barriers(np, comp, mid_rows)
        split_ids = [int(v) for v in np.unique(split[comp]) if int(v) > 0]
        if len(split_ids) <= 1:
            out[comp] = next_id
            kind[comp] = 1
            next_id += 1
            continue
        # Stable west-to-east ordering by circular mean column.
        ordered = []
        for sid in split_ids:
            ys, xs = np.nonzero(split == sid)
            if len(xs) == 0:
                continue
            mean_x = float(np.mean(xs))
            ordered.append((mean_x, sid))
        for _, sid in sorted(ordered):
            mask = split == sid
            if int(np.sum(mask)) < max(32, total_ocean // 3000):
                # Tiny slivers are kept with nearest already-labeled neighbor later.
                continue
            out[mask] = next_id
            kind[mask] = 1
            next_id += 1

    # Any tiny unlabeled ocean sliver inherits the nearest neighboring basin by a
    # few relaxation passes; remaining cells become their own basin.
    unlabeled = ocean & (out <= 0)
    if np.any(unlabeled):
        for _ in range(8):
            grown = out.copy()
            for shifted in (np.roll(out, 1, axis=0), np.roll(out, -1, axis=0), np.roll(out, 1, axis=1), np.roll(out, -1, axis=1)):
                grown = np.where((grown <= 0) & ocean & (shifted > 0), shifted, grown)
            out = grown
            kind = np.where((kind <= 0) & ocean & (out > 0), 1, kind)
            unlabeled = ocean & (out <= 0)
            if not np.any(unlabeled):
                break
        if np.any(unlabeled):
            out[unlabeled] = next_id
            kind[unlabeled] = 2
            next_id += 1

    # Any remaining ocean cells with a basin ID but no kind inherit open-ocean.
    kind[(out > 0) & (kind <= 0)] = 1
    return out.astype(np.int32), kind.astype(np.int16)


def _split_large_ocean_component_by_barriers(np, comp, mid_rows):
    h, w = comp.shape
    split = np.zeros((h, w), dtype=np.int32)
    mid = comp[mid_rows, :] if np.any(mid_rows) else comp
    open_frac = np.mean(mid, axis=0).astype(np.float32)
    # Smooth across longitude, including the seam.
    smooth = open_frac.copy()
    for _ in range(10):
        smooth = (0.58 * smooth + 0.21 * np.roll(smooth, 1) + 0.21 * np.roll(smooth, -1)).astype(np.float32)
    median_open = float(np.median(smooth)) if smooth.size else 0.0
    barrier_threshold = max(0.10, min(0.46, median_open * 0.62))
    barrier = smooth <= barrier_threshold

    groups = []
    visited = np.zeros(w, dtype=bool)
    for x0 in range(w):
        if visited[x0] or not barrier[x0]:
            continue
        xs = []
        x = x0
        while not visited[x] and barrier[x]:
            visited[x] = True
            xs.append(x)
            x = (x + 1) % w
            if x == x0:
                break
        if len(xs) >= max(2, w // 90):
            # Center at the most closed column inside the barrier group.
            local = min(xs, key=lambda xx: float(smooth[xx]))
            groups.append(int(local))
    groups = sorted(set(groups))

    # If a world ocean has no strong meridian barriers, still split extremely wide
    # domains into 3 broad basins so current routing does not become one global oval.
    if len(groups) < 2:
        xs = np.nonzero(np.any(comp, axis=0))[0]
        if len(xs) > int(0.72 * w):
            groups = [int(round(w * frac)) % w for frac in (0.0, 1.0 / 3.0, 2.0 / 3.0)]
        else:
            split[comp] = 1
            return split

    # Segments are the water between successive barrier centers.  This deliberately
    # treats the seam as continuous.
    centers = sorted(groups)
    sector_id = np.zeros(w, dtype=np.int32)
    sid = 1
    for i, start in enumerate(centers):
        end = centers[(i + 1) % len(centers)]
        x = (start + 1) % w
        while x != (end + 1) % w:
            sector_id[x] = sid
            x = (x + 1) % w
        sid += 1
    # Barrier columns inherit the following sector so no ocean cells disappear.
    for c in centers:
        sector_id[c] = sector_id[(c + 1) % w] or 1
    split[comp] = sector_id[np.nonzero(comp)[1]]
    return split


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



def _zonal_basin_width_cells(np, basin_id, ocean):
    """Approximate row-local ocean-basin width for strait/chokepoint damping."""
    h, w = basin_id.shape
    out = np.zeros((h, w), dtype=np.float32)
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
            out[y, start:end] = float(length)
    out[~ocean] = 0.0
    return out.astype(np.float32)


def _ocean_openness(np, ocean):
    """Return 0..1 distance-from-land openness used to avoid strong currents in tiny straits."""
    try:
        from scipy import ndimage
        dist = ndimage.distance_transform_edt(ocean).astype(np.float32)
        return np.clip(dist / 8.0, 0.0, 1.0).astype(np.float32)
    except Exception:
        field = ocean.astype(np.float32)
        # Cheap fallback: repeated erosion-like averaging makes narrow passages low.
        for _ in range(6):
            neigh = (
                np.roll(field, 1, axis=1)
                + np.roll(field, -1, axis=1)
                + np.vstack([field[0:1, :], field[:-1, :]])
                + np.vstack([field[1:, :], field[-1:, :]])
            ) / 4.0
            field = np.where(ocean, np.minimum(field, neigh + 0.12), 0.0)
        return np.clip(field, 0.0, 1.0).astype(np.float32)

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
