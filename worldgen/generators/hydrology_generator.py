"""Surface hydrology and drainage basins for the Main Planet.

This is a simplified hydrology model, but the routing is now intended to behave
like a real watershed system:

1. runoff is generated from precipitation, temperature, and elevation;
2. water routes toward lower neighbors, with limited spillover/breaching for wet
   depressions so rivers can continue to the sea;
3. terminal basins are classified as ocean-draining or endorheic;
4. small coastal catchments are merged into a visual "coastal basins" class;
5. rivers are drawn from accumulated liquid runoff, so they can cross dry zones
   only when fed by wetter/highland source regions.
"""

from __future__ import annotations

import heapq
import math
from collections import defaultdict

from worldgen.models.planet_profile import ClimateMap, HydrologyMap, TerrainMap
from worldgen.random_utils import clamp

NO_DRAIN = -1
COASTAL_MINOR_BASIN_ID = -2


def generate_hydrology(terrain: TerrainMap, climate: ClimateMap) -> HydrologyMap:
    width = terrain.width
    height = terrain.height
    total = width * height

    elevations: list[int] = [0] * total
    is_land: list[bool] = [False] * total
    local_runoff: list[float] = [0.0] * total
    downstream: list[int] = [NO_DRAIN] * total

    # Plate terrain can expose pre-hydrology lake-basin candidates.  Hydrology
    # still routes flow normally, but these fields let rift/foreland/interior
    # depressions appear as conservative through-flow or endorheic lake masks
    # instead of requiring a perfect terminal sink cell after priority flooding.
    lake_candidate_strength: list[float] = [0.0] * total
    candidate_grid = getattr(terrain, "plate_tectonic_lake_candidate_x1000", None)
    if candidate_grid is not None:
        try:
            if len(candidate_grid) == height and all(len(row) == width for row in candidate_grid):
                for rr in range(height):
                    row = candidate_grid[rr]
                    for cc in range(width):
                        lake_candidate_strength[rr * width + cc] = max(0.0, min(1.0, float(row[cc]) / 1000.0))
        except Exception:
            lake_candidate_strength = [0.0] * total

    runoff_grid: list[list[int]] = []

    for r in range(height):
        runoff_row: list[int] = []
        for c in range(width):
            idx = r * width + c
            elev = terrain.elevation_m[r][c]
            land = terrain.is_land[r][c]
            elevations[idx] = elev
            is_land[idx] = land
            if land:
                temp_c = climate.annual_mean_temp_c_x10[r][c] / 10.0
                precip = climate.annual_precip_mm[r][c]
                runoff = _runoff_mm(precip, temp_c, elev)
            else:
                runoff = 0.0
            local_runoff[idx] = runoff
            runoff_row.append(int(round(runoff)))
        runoff_grid.append(runoff_row)

    # Priority-flood depression filling creates a spill surface that lets
    # wet watersheds escape shallow local pits and route to the ocean. This
    # fixes the common procedural-terrain failure where almost every small
    # depression becomes endorheic. Truly dry enclosed interiors can still
    # remain endorheic through the lake/sink classification layer.
    routing_elevations = _priority_flood_spill_surface(width, height, elevations, is_land)
    for r in range(height):
        for c in range(width):
            idx = r * width + c
            if not is_land[idx]:
                continue
            downstream[idx] = _choose_downstream_cell(
                width=width,
                height=height,
                r=r,
                c=c,
                idx=idx,
                elevations=routing_elevations,
                is_land=is_land,
                local_runoff=local_runoff,
                annual_precip=climate.annual_precip_mm[r][c],
            )

    accumulation = _accumulate_flow(total, is_land, downstream, local_runoff)
    water_component_id, ocean_component_ids, water_component_sizes, water_component_types = _water_component_info(width, height, is_land, elevations)

    land_indices = [idx for idx in range(total) if is_land[idx]]
    positive_accum = [accumulation[idx] for idx in land_indices if accumulation[idx] > 0.0]
    if positive_accum:
        threshold = max(420.0, _quantile(positive_accum, 0.985))
        max_accum = max(positive_accum)
    else:
        threshold = 420.0
        max_accum = 0.0

    (
        basin_grid_flat,
        basin_count,
        basin_sizes,
        basin_terminal_type,
        coastal_basin_count,
        endorheic_basin_count,
        minor_coastal_cells,
    ) = _compute_drainage_basins(width, height, is_land, downstream, water_component_id, ocean_component_ids)

    major_threshold = max(650, int(total * 0.0032))
    major_basin_count = sum(
        1
        for basin_id, size in basin_sizes.items()
        if basin_id > 0 and size >= major_threshold
    )

    # Make rivers easier to read and more hydrologically connected by including
    # all connected downstream cells once the river has formed upstream.
    river_flat = _build_river_network(width, height, is_land, downstream, accumulation, threshold, max_accum)

    river_grid: list[list[int]] = []
    flow_grid: list[list[int]] = []
    lake_grid: list[list[bool]] = []
    basin_grid: list[list[int]] = []
    river_cells = 0
    lake_cells = 0

    for r in range(height):
        river_row: list[int] = []
        flow_row: list[int] = []
        lake_row: list[bool] = []
        basin_row: list[int] = []
        for c in range(width):
            idx = r * width + c
            acc = accumulation[idx] if is_land[idx] else 0.0
            flow_int = int(round(min(999_999, acc)))
            flow_row.append(flow_int)
            basin_row.append(basin_grid_flat[idx] if is_land[idx] else 0)

            intensity = river_flat[idx]
            if intensity > 0:
                river_cells += 1
            river_row.append(intensity)

            basin_id = basin_grid_flat[idx]
            terminal_type = basin_terminal_type.get(basin_id, "")
            temp_c = climate.annual_mean_temp_c_x10[r][c] / 10.0
            precip = climate.annual_precip_mm[r][c]
            inland_water = (not is_land[idx]) and water_component_id[idx] >= 0 and water_component_id[idx] not in ocean_component_ids and water_component_types.get(water_component_id[idx], "lake") in {"lake", "inland_sea"}
            candidate_strength = lake_candidate_strength[idx] if is_land[idx] else 0.0
            closed_basin_lake = (
                is_land[idx]
                and terminal_type in {"endorheic", "lake"}
                and downstream[idx] == NO_DRAIN
                and acc >= threshold * 0.85
                and elevations[idx] <= max(1600, terrain.mean_land_elevation_m * 1.8)
                and temp_c > -4.0
                and precip >= 230
            )
            # Plate Terrain 13 produces explicit rift/foreland/plain lake-basin
            # candidates.  Use a lower threshold than the old conservative pass
            # so lakes appear as visible systems rather than rare terminal-pit
            # accidents, while still requiring water availability and plausible
            # temperature/elevation.
            plate_candidate_lake = (
                is_land[idx]
                and candidate_strength >= 0.50
                and (acc >= threshold * 0.12 or local_runoff[idx] >= 120.0 or terminal_type in {"endorheic", "lake"})
                and elevations[idx] <= max(2500, terrain.mean_land_elevation_m * 2.35)
                and temp_c > -8.0
                and precip >= 90
            )
            lake = bool(inland_water or closed_basin_lake or plate_candidate_lake)
            if lake:
                lake_cells += 1
            lake_row.append(lake)
        river_grid.append(river_row)
        flow_grid.append(flow_row)
        lake_grid.append(lake_row)
        basin_grid.append(basin_row)

    lake_grid, removed_lake_candidate_cells, removed_lake_candidate_components = _filter_oversized_lake_candidate_components(width, height, is_land, lake_grid)
    lake_cells = sum(1 for row in lake_grid for value in row if value)
    major_rivers = _estimate_major_river_count(width, height, river_grid)
    hydrology_notes = [
        "Hydrology is derived from terrain, precipitation, temperature, and annual liquid-runoff potential.",
        "Flow routing uses D8 downhill routing over a priority-flood spill surface, so wet rivers escape shallow local pits.",
        "Major rivers are extended downstream to ocean outlets, shared inland-lake terminals, or endorheic terminal basins.",
        "Only genuinely tiny coastal catchments are merged into a visual coastal-basins class; larger ocean-draining basins remain distinct.",
        "Lake cells are conservative closed-basin and plate-basin candidates, not final lake shorelines.",
    ]
    if removed_lake_candidate_cells > 0:
        hydrology_notes.append(
            f"Removed {removed_lake_candidate_cells:,} cells from {removed_lake_candidate_components:,} oversized continent-scale lake-candidate components."
        )

    return HydrologyMap(
        width=width,
        height=height,
        runoff_mm=runoff_grid,
        flow_accumulation=flow_grid,
        river_intensity=river_grid,
        lake_mask=lake_grid,
        drainage_basin_id=basin_grid,
        river_cell_count=river_cells,
        lake_cell_count=lake_cells,
        max_flow_accumulation=int(round(max_accum)),
        river_threshold=int(round(threshold)),
        estimated_major_river_count=major_rivers,
        drainage_basin_count=basin_count,
        major_drainage_basin_count=major_basin_count,
        coastal_basin_count=coastal_basin_count,
        endorheic_basin_count=endorheic_basin_count,
        minor_coastal_basin_cell_count=minor_coastal_cells,
        delta_cell_count=_estimate_delta_cell_count(width, height, is_land, downstream, river_flat),
        notes=hydrology_notes,
    )





def _filter_oversized_lake_candidate_components(width: int, height: int, is_land: list[bool], lake_grid: list[list[bool]]) -> tuple[list[list[bool]], int, int]:
    """Remove continent-scale plate lake candidates from the visible lake mask.

    Plate terrain produces broad basin-potential fields for rifts, forelands, and
    interiors.  Hydrology should turn parts of those fields into lakes, not color
    an entire continent as a lake candidate.  This pass removes only connected
    land-lake components that are far too large or span too much of the map.
    Existing water-body lakes/inland seas are preserved.
    """
    try:
        import numpy as np
        from scipy import ndimage
    except Exception:
        return lake_grid, 0, 0
    land = np.asarray(is_land, dtype=bool).reshape((height, width))
    lake = np.asarray(lake_grid, dtype=bool)
    if lake.shape != (height, width):
        return lake_grid, 0, 0
    candidate = lake & land
    if not bool(candidate.any()):
        return lake_grid, 0, 0
    labels, count = ndimage.label(candidate, structure=np.ones((3, 3), dtype=np.uint8))
    if count <= 0:
        return lake_grid, 0, 0
    land_total = max(1, int(land.sum()))
    world_total = max(1, width * height)
    # Genuine lakes can be huge, but a single pre-hydrology lake-candidate mask
    # covering a large share of all land is usually a basin-field artifact.
    max_cells = max(96, int(land_total * 0.018), int(world_total * 0.0025))
    max_width_span = max(12, int(width * 0.28))
    max_height_span = max(8, int(height * 0.34))
    remove = np.zeros_like(candidate, dtype=bool)
    removed_components = 0
    for comp_id in range(1, count + 1):
        mask = labels == comp_id
        size = int(mask.sum())
        if size <= 0:
            continue
        ys, xs = np.where(mask)
        span_y = int(ys.max() - ys.min() + 1) if ys.size else 0
        # Circular X span: choose the shorter longitude span across the seam.
        if xs.size:
            xs_sorted = np.sort(np.unique(xs))
            gaps = np.diff(np.r_[xs_sorted, xs_sorted[0] + width])
            span_x = int(width - gaps.max()) if gaps.size else 0
        else:
            span_x = 0
        too_large = size > max_cells or (span_x > max_width_span and span_y > 3) or span_y > max_height_span
        if too_large:
            remove |= mask
            removed_components += 1
    if not bool(remove.any()):
        return lake_grid, 0, 0
    lake[remove] = False
    return lake.tolist(), int(remove.sum()), int(removed_components)

def _water_component_info(width: int, height: int, is_land: list[bool], elevations: list[int] | None = None) -> tuple[list[int], set[int], dict[int, int], dict[int, str]]:
    """Label water components and classify ocean vs lake-like water.

    Earlier builds treated only the largest connected water component as the
    world ocean.  When land split the map into two large oceans (often northern
    and southern polar oceans), every non-largest ocean was incorrectly treated
    as a lake terminal.  This pass classifies multiple ocean-scale components as
    outlets while keeping genuinely enclosed lakes/inland seas as lake basins.
    """
    from collections import deque
    total = width * height
    comp = [-1] * total
    sizes: dict[int, int] = {}
    touches_pole: dict[int, bool] = {}
    touches_seam: dict[int, bool] = {}
    depth_sum: dict[int, float] = {}
    next_id = 0
    for idx in range(total):
        if is_land[idx] or comp[idx] >= 0:
            continue
        q: deque[int] = deque([idx])
        comp[idx] = next_id
        size = 0
        pole = False
        seam = False
        depth_total = 0.0
        while q:
            cur = q.popleft()
            size += 1
            r = cur // width
            c = cur % width
            pole = pole or r == 0 or r == height - 1
            seam = seam or c == 0 or c == width - 1
            if elevations is not None:
                depth_total += max(0.0, -float(elevations[cur]))
            for dr, dc in ((-1,0),(1,0),(0,-1),(0,1)):
                rr = r + dr
                if rr < 0 or rr >= height:
                    continue
                cc = (c + dc) % width
                n = rr * width + cc
                if is_land[n] or comp[n] >= 0:
                    continue
                comp[n] = next_id
                q.append(n)
        sizes[next_id] = size
        touches_pole[next_id] = pole
        touches_seam[next_id] = seam
        depth_sum[next_id] = depth_total
        next_id += 1
    if not sizes:
        return comp, set(), sizes, {}

    largest_id, largest_size = max(sizes.items(), key=lambda item: item[1])
    component_types: dict[int, str] = {}
    ocean_ids: set[int] = {largest_id}
    world = max(1, total)
    for cid, size in sizes.items():
        frac = size / world
        mean_depth = depth_sum.get(cid, 0.0) / max(1, size)
        if cid == largest_id:
            ctype = "ocean"
        elif frac >= 0.018 or size >= max(1_200, int(largest_size * 0.18)):
            ctype = "secondary_ocean"
        elif (touches_pole.get(cid, False) or touches_seam.get(cid, False)) and frac >= 0.004 and mean_depth > 60.0:
            ctype = "secondary_ocean"
        elif frac >= 0.0028 and mean_depth > 120.0:
            ctype = "inland_sea"
        else:
            ctype = "lake"
        component_types[cid] = ctype
        if ctype in {"ocean", "secondary_ocean"}:
            ocean_ids.add(cid)
    return comp, ocean_ids, sizes, component_types

def _priority_flood_spill_surface(
    width: int,
    height: int,
    elevations: list[int],
    is_land: list[bool],
) -> list[int]:
    """Return a hydrologically conditioned elevation surface.

    Land cells adjacent to the ocean are treated as outlets. A priority-flood
    then propagates inland and raises enclosed local pits only as much as needed
    to spill toward an outlet. This is not a full erosion model, but it is a
    standard DEM-conditioning step and makes generated river networks much less
    likely to terminate in thousands of tiny artificial endorheic pits.
    """
    total = width * height
    filled = list(elevations)
    visited = bytearray(total)
    heap: list[tuple[int, int]] = []

    for r in range(height):
        for c in range(width):
            idx = r * width + c
            if not is_land[idx]:
                continue
            ocean_neighbor = False
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)):
                rr = r + dr
                if rr < 0 or rr >= height:
                    ocean_neighbor = True
                    break
                cc = (c + dc) % width
                nidx = rr * width + cc
                if not is_land[nidx]:
                    ocean_neighbor = True
                    break
            if ocean_neighbor:
                visited[idx] = 1
                heapq.heappush(heap, (filled[idx], idx))

    if not heap:
        return filled

    while heap:
        spill_elev, idx = heapq.heappop(heap)
        r = idx // width
        c = idx % width
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)):
            rr = r + dr
            if rr < 0 or rr >= height:
                continue
            cc = (c + dc) % width
            nidx = rr * width + cc
            if visited[nidx] or not is_land[nidx]:
                continue
            visited[nidx] = 1
            # A tiny epsilon slope is represented as one metre to keep routing
            # deterministic across flats while preserving broad valley shapes.
            new_elev = max(filled[nidx], spill_elev + 1)
            filled[nidx] = new_elev
            heapq.heappush(heap, (new_elev, nidx))

    return filled

def _choose_downstream_cell(
    width: int,
    height: int,
    r: int,
    c: int,
    idx: int,
    elevations: list[int],
    is_land: list[bool],
    local_runoff: list[float],
    annual_precip: int,
) -> int:
    current_elev = elevations[idx]
    best_lower_idx = NO_DRAIN
    best_lower_drop = 0.0
    lowest_neighbor_idx = NO_DRAIN
    lowest_neighbor_elev = current_elev

    for dr in (-1, 0, 1):
        rr = r + dr
        if rr < 0 or rr >= height:
            continue
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            cc = (c + dc) % width
            nidx = rr * width + cc
            nelev = elevations[nidx]
            if nelev < lowest_neighbor_elev:
                lowest_neighbor_elev = nelev
                lowest_neighbor_idx = nidx

            distance = 1.4142 if dr and dc else 1.0
            drop = (current_elev - nelev) / distance
            if drop > best_lower_drop:
                best_lower_drop = drop
                best_lower_idx = nidx

    if best_lower_idx != NO_DRAIN:
        return best_lower_idx

    # No lower neighbor. Allow wet depressions and highland pits to overflow to
    # their lowest neighbor, simulating lake spillover/valley incision. Keep dry
    # depressions endorheic.
    if lowest_neighbor_idx != NO_DRAIN:
        spill_rise = max(0, lowest_neighbor_elev - current_elev)
        wet_enough = local_runoff[idx] >= 70.0 or annual_precip >= 620
        highland_spill = current_elev >= 700 and local_runoff[idx] >= 38.0
        very_wet_basin = local_runoff[idx] >= 150.0 or annual_precip >= 1050
        # Stronger erosion/deposition terrain tends to breach shallow divides;
        # this helps wet river systems continue to an ocean or a legitimate
        # inland terminal basin instead of ending in tiny local pits.
        if (wet_enough and spill_rise <= 420) or (highland_spill and spill_rise <= 620) or (very_wet_basin and spill_rise <= 880):
            return lowest_neighbor_idx

    return NO_DRAIN


def _accumulate_flow(
    total: int,
    is_land: list[bool],
    downstream: list[int],
    local_runoff: list[float],
) -> list[float]:
    """Accumulate upstream runoff without recursion.

    Earlier versions used a recursive depth-first sum over upstream cells. That
    works on small maps, but at 4096 x 2048 a perfectly valid long river valley
    can exceed Python's recursion limit. This implementation treats the flow
    network as a one-downstream-cell graph and processes it with Kahn-style
    topological propagation from headwaters to outlets.

    A tiny number of cells may remain unprocessed if spillover routing creates a
    flat loop. Those cells keep their accumulated upstream water and are treated
    as terminal/endoreic loop cells by the basin classifier rather than crashing
    the run.
    """
    from collections import deque

    indegree = bytearray(total)
    accumulation = [0.0] * total

    for idx in range(total):
        if not is_land[idx]:
            continue
        accumulation[idx] = local_runoff[idx]
        target = downstream[idx]
        if 0 <= target < total and is_land[target]:
            # Direct D8 indegree is at most 8, so a byte is enough and avoids a
            # very large Python int list on high-resolution maps.
            indegree[target] = min(255, indegree[target] + 1)

    queue: deque[int] = deque(idx for idx in range(total) if is_land[idx] and indegree[idx] == 0)
    processed = bytearray(total)

    while queue:
        idx = queue.popleft()
        if processed[idx]:
            continue
        processed[idx] = 1
        target = downstream[idx]
        if 0 <= target < total and is_land[target]:
            accumulation[target] += accumulation[idx]
            if indegree[target] > 0:
                indegree[target] -= 1
            if indegree[target] == 0:
                queue.append(target)

    # Any unprocessed land cells are in a loop/closed flat. Do not recurse; keep
    # the water already delivered to them. This is conservative and lets later
    # basin/lake logic classify them as endoreic sinks.
    for idx in range(total):
        if not is_land[idx]:
            accumulation[idx] = 0.0

    return accumulation


def _build_river_network(
    width: int,
    height: int,
    is_land: list[bool],
    downstream: list[int],
    accumulation: list[float],
    threshold: float,
    max_accum: float,
) -> list[int]:
    total = width * height
    river = [0] * total
    if max_accum <= 0.0:
        return river

    denom = max(1.0, math.log1p(max_accum / max(threshold, 1.0)))
    river_start_threshold = threshold
    continuation_threshold = threshold * 0.34

    starts = [idx for idx in range(total) if is_land[idx] and accumulation[idx] >= river_start_threshold]
    starts.sort(key=lambda idx: accumulation[idx], reverse=True)

    for start in starts:
        current = start
        seen: set[int] = set()
        while current >= 0 and current < total and is_land[current] and current not in seen:
            seen.add(current)
            acc = accumulation[current]
            if acc < continuation_threshold:
                break
            intensity = int(round(55 + 200 * (math.log1p(acc / max(threshold, 1.0)) / denom)))
            river[current] = max(river[current], int(clamp(intensity, 50, 255)))
            target = downstream[current]
            if target < 0:
                break
            if target >= total or not is_land[target]:
                break
            current = target

    return river


def _runoff_mm(precip_mm: int, temp_c: float, elevation_m: int) -> float:
    warmth = max(0.0, temp_c)
    evap = 210.0 + 13.5 * warmth
    if temp_c < -5.0:
        evap += 160.0
    if elevation_m > 1500:
        evap -= 85.0
    runoff = (precip_mm - evap) * 0.58
    return clamp(runoff, 0.0, max(0.0, precip_mm * 0.86))


def _compute_drainage_basins(
    width: int,
    height: int,
    is_land: list[bool],
    downstream: list[int],
    water_component_id: list[int],
    ocean_component_ids: set[int],
) -> tuple[list[int], int, dict[int, int], dict[int, str], int, int, int]:
    """Classify drainage basins by resolved outlet.

    Earlier versions used the exact ocean-adjacent target cell as the coastal
    terminal. On high-resolution maps that creates hundreds of thousands of
    tiny coastal basins, because every creek mouth gets a unique outlet id. This
    version groups ocean outlets into scale-aware coastal segments, while still
    keeping large trunk basins separate enough to be readable. Endorheic loops
    keep their own terminal ids.
    """
    total = width * height
    basin_by_terminal: dict[tuple[str, int], int] = {}
    basin_terminal_type: dict[int, str] = {}
    basin_sizes: dict[int, int] = {}
    basin_ids: list[int] = [0] * total
    next_basin_id = 1

    for idx in range(total):
        if not is_land[idx] or basin_ids[idx] != 0:
            continue

        path: list[int] = []
        seen_at: dict[int, int] = {}
        current = idx
        basin_id = 0
        terminal_key: tuple[str, int] | None = None

        while True:
            if current in seen_at:
                loop_cells = path[seen_at[current]:]
                terminal_key = ("endorheic", _sink_segment_key(width, height, min(loop_cells)))
                break

            seen_at[current] = len(path)
            path.append(current)

            existing = basin_ids[current]
            if existing > 0 or existing == COASTAL_MINOR_BASIN_ID:
                basin_id = existing
                break

            target = downstream[current]
            if target < 0:
                terminal_key = ("endorheic", _sink_segment_key(width, height, current))
                break
            if target >= total or not is_land[target]:
                if 0 <= target < total and water_component_id[target] >= 0 and water_component_id[target] not in ocean_component_ids:
                    terminal_key = ("lake", water_component_id[target])
                else:
                    terminal_key = ("coastal", _coastal_outlet_segment_key(width, height, current))
                break
            current = target

        if basin_id == 0:
            assert terminal_key is not None
            basin_id = basin_by_terminal.get(terminal_key, 0)
            if basin_id == 0:
                basin_id = next_basin_id
                next_basin_id += 1
                basin_by_terminal[terminal_key] = basin_id
                basin_terminal_type[basin_id] = terminal_key[0]
                basin_sizes[basin_id] = 0

        for cell in path:
            if basin_ids[cell] == 0:
                basin_ids[cell] = basin_id
                basin_sizes[basin_id] = basin_sizes.get(basin_id, 0) + 1

    initial_coastal_basin_count = sum(1 for _b, typ in basin_terminal_type.items() if typ == "coastal")
    endorheic_basin_count = sum(1 for _b, typ in basin_terminal_type.items() if typ == "endorheic")

    # Merge only truly tiny coastal cells. Because outlets are now already
    # grouped by coastal segment, this threshold can be smaller and avoids
    # swallowing medium coastal watersheds into a single visual class.
    minor_coastal_threshold = max(65, int(total * 0.000022))
    minor_coastal_ids = {
        basin_id
        for basin_id, typ in basin_terminal_type.items()
        if typ == "coastal" and basin_sizes.get(basin_id, 0) < minor_coastal_threshold
    }
    minor_coastal_cells = sum(basin_sizes.get(basin_id, 0) for basin_id in minor_coastal_ids)
    if minor_coastal_ids:
        minor_set = minor_coastal_ids
        for i, value in enumerate(basin_ids):
            if value in minor_set:
                basin_ids[i] = COASTAL_MINOR_BASIN_ID

    coastal_basin_count = max(0, initial_coastal_basin_count - len(minor_coastal_ids))
    basin_count = next_basin_id - 1
    return (
        basin_ids,
        basin_count,
        basin_sizes,
        basin_terminal_type,
        coastal_basin_count,
        endorheic_basin_count,
        minor_coastal_cells,
    )


def _coastal_outlet_segment_key(width: int, height: int, land_idx: int) -> int:
    """Scale-aware key for coastal drainage outlets.

    A segment is roughly 1-2 degrees wide on normal maps and larger on very high
    resolution maps. This merges adjacent tiny creek mouths but preserves broad
    regional ocean-draining basins.
    """
    r = land_idx // width
    c = land_idx % width
    lon_segments = max(48, min(256, width // 18))
    lat_segments = max(24, min(128, height // 18))
    lon_bin = int(c * lon_segments / max(1, width)) % lon_segments
    lat_bin = int(r * lat_segments / max(1, height))
    return lat_bin * lon_segments + lon_bin


def _sink_segment_key(width: int, height: int, idx: int) -> int:
    # Endorheic sinks are grouped more coarsely than coastal mouths so one
    # interior playa/lake basin does not fragment into many loop ids.
    r = idx // width
    c = idx % width
    lon_segments = max(64, min(256, width // 16))
    lat_segments = max(32, min(128, height // 16))
    lon_bin = int(c * lon_segments / max(1, width)) % lon_segments
    lat_bin = int(r * lat_segments / max(1, height))
    return lat_bin * lon_segments + lon_bin


def _estimate_delta_cell_count(width: int, height: int, is_land: list[bool], downstream: list[int], river: list[int]) -> int:
    total = width * height
    count = 0
    for idx in range(total):
        if not is_land[idx] or river[idx] < 180:
            continue
        target = downstream[idx]
        if target < 0 or target >= total or not is_land[target]:
            count += 1
    return count


def _quantile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    q = clamp(q, 0.0, 1.0)
    return ordered[int(q * (len(ordered) - 1))]


def _estimate_major_river_count(width: int, height: int, river_grid: list[list[int]]) -> int:
    visited = bytearray(width * height)
    major = 0
    min_component_size = max(20, int(width * height * 0.00007))

    for r in range(height):
        for c in range(width):
            idx = r * width + c
            if visited[idx] or river_grid[r][c] <= 0:
                continue
            stack = [(r, c)]
            visited[idx] = 1
            count = 0
            peak = 0
            while stack:
                rr, cc = stack.pop()
                count += 1
                peak = max(peak, river_grid[rr][cc])
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr = rr + dr
                    if nr < 0 or nr >= height:
                        continue
                    nc = (cc + dc) % width
                    nidx = nr * width + nc
                    if visited[nidx] or river_grid[nr][nc] <= 0:
                        continue
                    visited[nidx] = 1
                    stack.append((nr, nc))
            if count >= min_component_size or peak >= 230:
                major += 1
    return major
