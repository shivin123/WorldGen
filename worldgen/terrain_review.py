"""Terrain Stage 3 review diagnostics and control derivation.

This module intentionally does not replace the terrain generator yet.  It builds
review metadata, sub-stage checkpoint diagnostics, and map-only diagnostic files
from the current terrain raster so the Web UI can inspect terrain quality before
we do a deeper terrain-science rewrite.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

from worldgen.models.bodies import Planet
from worldgen.models.planet_profile import GeologyState, Hydrosphere, TerrainMap
from worldgen.config import PlanetProfileConfig




PROVINCE_TYPE_LABELS = {
    0: "old oceanic basin",
    1: "young oceanic crust / ridge province",
    2: "continental core / craton",
    3: "continental shelf or margin",
    4: "rifted continental margin",
    5: "volcanic arc province",
    6: "accreted terrane / microcontinent",
    7: "sedimentary basin",
    8: "shield / old highland",
}


PLATE_TECTONIC_TYPE_LABELS = {
    0: "oceanic plate",
    1: "continental plate",
    2: "mixed plate",
    3: "microplate / terrane",
}

PLATE_TECTONIC_TYPE_COLORS = {
    0: (42, 82, 150),
    1: (162, 134, 78),
    2: (126, 164, 112),
    3: (190, 105, 185),
}

PLATE_SUBDUCTION_POLARITY_LABELS = {
    0: "none / not convergent",
    1: "oceanic under continental or mixed",
    2: "ocean-ocean subduction",
    3: "continental collision",
}

PLATE_SUBDUCTION_POLARITY_COLORS = {
    0: (48, 66, 92),
    1: (232, 92, 54),
    2: (218, 132, 64),
    3: (188, 60, 72),
}

BOUNDARY_CLASS_LABELS = {
    0: "intraplate / inactive interior",
    1: "convergent or collision boundary",
    2: "divergent rift boundary",
    3: "transform or shear boundary",
    4: "passive margin",
    5: "diffuse or old suture boundary",
    6: "volcanic arc boundary",
}

PROVINCE_TYPE_COLORS = {
    0: (38, 68, 130),
    1: (58, 112, 185),
    2: (145, 124, 78),
    3: (185, 165, 105),
    4: (205, 128, 72),
    5: (178, 72, 54),
    6: (156, 104, 172),
    7: (126, 170, 108),
    8: (210, 196, 150),
}

BOUNDARY_CLASS_COLORS = {
    0: (225, 225, 215),
    1: (165, 55, 45),
    2: (55, 105, 190),
    3: (128, 70, 165),
    4: (80, 150, 135),
    5: (160, 135, 85),
    6: (215, 92, 55),
}

COAST_STYLE_LABELS = {
    0: "non-coast / background",
    1: "passive smooth coastal plain",
    2: "rugged active or fjorded margin",
    3: "rifted gulf margin",
    4: "volcanic arc coast",
    5: "shelf or low deltaic plain",
    6: "mixed irregular coast",
}

COAST_STYLE_COLORS = {
    0: (34, 78, 128),
    1: (112, 184, 128),
    2: (205, 84, 62),
    3: (78, 130, 212),
    4: (232, 108, 54),
    5: (88, 205, 170),
    6: (235, 220, 112),
}

PLATE_MARGIN_LABELS = {
    0: "background / non-margin",
    1: "passive continental margin",
    2: "active/subduction margin",
    3: "rifted margin or gulf",
    4: "volcanic island-arc margin",
    5: "transform/shear margin",
    6: "mixed/accreted margin",
}

PLATE_MARGIN_COLORS = {
    0: (34, 78, 128),
    1: (95, 178, 132),
    2: (205, 74, 58),
    3: (72, 124, 215),
    4: (230, 112, 58),
    5: (150, 88, 205),
    6: (228, 210, 104),
}

ISLAND_ORIGIN_LABELS = {
    0: "water / non-island",
    1: "continent or large landmass",
    2: "shelf island",
    3: "volcanic or arc island",
    4: "microcontinent or terrane",
    5: "hotspot or high island",
}

ISLAND_ORIGIN_COLORS = {
    0: (35, 82, 135),
    1: (132, 146, 105),
    2: (230, 185, 74),
    3: (225, 92, 58),
    4: (164, 106, 190),
    5: (242, 238, 190),
}

OCEAN_FLOOR_LABELS = {
    0: "land / background",
    1: "abyssal plain",
    2: "mid-ocean ridge",
    3: "trench / subduction trough",
    4: "fracture or transform zone",
    5: "seamount / hotspot province",
}

OCEAN_FLOOR_COLORS = {
    0: (126, 142, 102),
    1: (22, 54, 116),
    2: (88, 174, 210),
    3: (20, 22, 62),
    4: (94, 76, 156),
    5: (210, 130, 78),
}

PLATE_FINAL_PROBLEM_LABELS = {
    0: "no specific plate-mode issue",
    1: "legacy foundation dependency",
    2: "weak plate-boundary expression",
    3: "weak hydrology readiness",
    4: "active-margin shelf conflict",
    5: "ocean-floor underexpression",
}

PLATE_FINAL_PROBLEM_COLORS = {
    0: (112, 150, 112),
    1: (224, 168, 78),
    2: (184, 78, 190),
    3: (210, 88, 70),
    4: (232, 214, 76),
    5: (64, 96, 196),
}


def _class_share_stats(arr, labels: dict[int, str], prefix: str) -> dict[str, Any]:
    import numpy as np
    stats: dict[str, Any] = {}
    if arr is None or np.size(arr) == 0:
        return stats
    total = max(1, int(arr.size))
    for code, label in labels.items():
        key = label.lower().replace(" / ", "_").replace(" ", "_").replace("-", "_")
        stats[f"{prefix}_{code}_{key}_share"] = round(float(np.sum(arr == code)) / total, 4)
    return stats


def _shannon_diversity(arr) -> float:
    import numpy as np
    if arr is None or np.size(arr) == 0:
        return 0.0
    counts = np.bincount(np.asarray(arr, dtype=np.int32).ravel())
    counts = counts[counts > 0]
    if counts.size <= 1:
        return 0.0
    p = counts / counts.sum()
    return round(float(-(p * np.log(p)).sum() / max(np.log(counts.size), 1e-9)), 3)

def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _level(value: float, *, low: float, high: float) -> str:
    if value < low:
        return "low"
    if value > high:
        return "high"
    return "moderate"


def _read_overrides(output_dir: str | Path | None) -> dict[str, Any]:
    if output_dir is None:
        return {}
    path = Path(output_dir) / "config" / "stage_overrides.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _terrain_overrides(output_dir: str | Path | None) -> dict[str, Any]:
    data = _read_overrides(output_dir).get("terrain", {})
    return data if isinstance(data, dict) else {}


def derive_terrain_controls(
    planet: Planet,
    hydrosphere: Hydrosphere,
    geology: GeologyState,
    config: PlanetProfileConfig,
    *,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Derive terrain controls from Stage 1/2 profile plus optional UI overrides."""
    ctx = planet.formation_context or {}
    overrides = _terrain_overrides(output_dir)
    asym = str(ctx.get("crustal_asymmetry_bias", "medium")).lower()
    tectonic_bias = str(ctx.get("tectonic_energy_bias", "earth_like")).lower()
    impact = str(ctx.get("impact_history", "normal")).lower()
    volatile_delivery = str(ctx.get("volatile_delivery", "moderate")).lower()

    asym_score = {"low": 0.25, "medium": 0.5, "moderate": 0.55, "high": 0.82}.get(asym, 0.5)
    tectonic_score = {"low": 0.22, "quiet": 0.18, "earth_like": 0.58, "moderate": 0.55, "moderate_high": 0.68, "high": 0.82}.get(tectonic_bias, 0.55)
    heat = _clamp(float(getattr(geology, "internal_heat", 0.6) or 0.6), 0.0, 2.0) / 2.0
    volcanism = _clamp(float(getattr(geology, "volcanism", 0.5) or 0.5), 0.0, 1.8) / 1.8
    erosion = _clamp(float(getattr(geology, "erosion", 1.0) or 1.0), 0.0, 2.5) / 2.5
    ocean = _clamp(float(getattr(hydrosphere, "ocean_fraction_target", 0.62) or 0.62), 0.05, 0.95)

    fragmentation = _clamp(0.25 + 0.30 * tectonic_score + 0.20 * heat + 0.12 * volcanism - 0.18 * asym_score + 0.08 * (ocean - 0.55), 0.05, 0.95)
    supercontinent_score = _clamp(0.18 + 0.50 * asym_score + 0.20 * (1.0 - fragmentation) + 0.10 * (1.0 - heat), 0.02, 0.98)
    plate_count = int(round(8 + tectonic_score * 18 + fragmentation * 14 + max(0.0, ocean - 0.65) * 6))
    coastline_complexity = _clamp(0.28 + 0.28 * fragmentation + 0.16 * erosion + 0.12 * volcanism + 0.08 * (ocean - 0.55), 0.05, 0.95)
    island_density = _clamp(0.15 + 0.38 * volcanism + 0.22 * fragmentation + 0.16 * max(0.0, ocean - 0.55), 0.02, 0.98)
    shelf_width = _clamp(0.45 + 0.22 * ocean + 0.18 * erosion + 0.10 * (1.0 - tectonic_score), 0.05, 1.2)
    coastal_ruggedness = _clamp(0.18 + 0.30 * tectonic_score + 0.22 * volcanism + 0.18 * float(getattr(geology, "surface_roughness", 0.5) or 0.5) - 0.14 * erosion, 0.02, 0.98)
    fjord_tendency = _clamp(0.08 + 0.28 * coastal_ruggedness + 0.18 * float(getattr(geology, "mountain_factor", 1.0) or 1.0) + 0.10 * max(0.0, ocean - 0.58), 0.0, 1.0)
    coastal_plain_bias = _clamp(0.18 + 0.35 * shelf_width + 0.24 * erosion + 0.12 * (1.0 - tectonic_score), 0.0, 1.0)
    island_shape_irregularity = _clamp(0.22 + 0.34 * coastline_complexity + 0.20 * volcanism + 0.12 * fragmentation, 0.0, 1.0)
    mountain_strength = _clamp(float(getattr(geology, "mountain_factor", 1.0) or 1.0) * (0.65 + 0.55 * tectonic_score), 0.05, 2.2)
    rift_strength = _clamp(0.18 + 0.38 * heat + 0.22 * fragmentation + 0.12 * volcanism, 0.02, 0.98)
    interior_relief = _clamp(0.20 + 0.28 * asym_score + 0.22 * float(getattr(geology, "surface_roughness", 0.5) or 0.5) + 0.12 * (1.0 - erosion), 0.02, 1.0)
    deposition_strength = _clamp(0.16 + 0.34 * erosion + 0.18 * shelf_width + 0.12 * (1.0 if volatile_delivery in {"wet", "high", "heavy_bombardment"} else 0.0), 0.02, 1.0)
    valley_carving_strength = _clamp(0.12 + 0.32 * erosion + 0.20 * tectonic_score + 0.14 * max(0.0, ocean - 0.50) + 0.12 * float(getattr(geology, "surface_roughness", 0.5) or 0.5), 0.02, 1.0)
    sediment_supply_strength = _clamp(0.10 + 0.28 * erosion + 0.24 * mountain_strength / 2.2 + 0.16 * volcanism + 0.12 * shelf_width, 0.02, 1.0)
    coastal_plain_strength = _clamp(0.10 + 0.38 * shelf_width + 0.20 * erosion + 0.16 * max(0.0, ocean - 0.55) - 0.10 * tectonic_score, 0.02, 1.0)
    alluvial_fan_strength = _clamp(0.08 + 0.30 * mountain_strength / 2.2 + 0.22 * erosion + 0.12 * max(0.0, 0.65 - ocean), 0.02, 1.0)
    floodplain_strength = _clamp(0.10 + 0.30 * erosion + 0.18 * max(0.0, ocean - 0.45) + 0.14 * deposition_strength, 0.02, 1.0)
    terrain_maturity = _clamp(0.14 + 0.42 * erosion + 0.18 * (1.0 - heat) + 0.12 * max(0.0, ocean - 0.50) + 0.10 * (1.0 if impact in {"calm", "normal"} else 0.35), 0.02, 1.0)
    plate_motion_speed = _clamp(0.16 + 0.44 * heat + 0.22 * tectonic_score + 0.10 * volcanism + 0.06 * fragmentation, 0.03, 1.0)
    plate_motion_chaos = _clamp(0.12 + 0.28 * fragmentation + 0.18 * volcanism + 0.08 * asym_score, 0.02, 1.0)
    convergence_bias = _clamp(0.20 + 0.36 * tectonic_score + 0.16 * float(getattr(geology, "mountain_factor", 1.0) or 1.0) / 2.2 + 0.08 * supercontinent_score, 0.02, 1.0)
    divergence_bias = _clamp(0.18 + 0.34 * heat + 0.22 * fragmentation + 0.10 * max(0.0, ocean - 0.55), 0.02, 1.0)
    transform_bias = _clamp(0.12 + 0.22 * fragmentation + 0.16 * plate_motion_chaos + 0.10 * tectonic_score, 0.02, 1.0)

    style = "derived_from_planet_physics"
    if ocean > 0.78:
        style = "ocean_world"
    elif supercontinent_score > 0.72 and fragmentation < 0.38:
        style = "supercontinent_world"
    elif island_density > 0.70:
        style = "archipelago_world"
    elif mountain_strength > 1.25 and rift_strength > 0.58:
        style = "rugged_tectonic_world"
    elif erosion > 0.62 and heat < 0.35:
        style = "old_eroded_shield_world"
    elif volcanism > 0.68:
        style = "volcanic_island_arc_world"

    def override_float(key: str, current: float) -> float:
        value = overrides.get(key)
        if value is None or value == "":
            return current
        try:
            return float(value)
        except Exception:
            return current

    def override_text(key: str, current: str) -> str:
        value = overrides.get(key)
        return str(value) if value not in {None, ""} else current

    plate_count = int(round(override_float("target_plate_count", float(plate_count))))
    fragmentation = override_float("fragmentation_tendency", override_float("continent_fragmentation", fragmentation))
    supercontinent_mode = override_text("supercontinent_tendency", "derived")
    style = override_text("terrain_style", style)

    return {
        "schema_version": 1,
        "stage": "terrain-controls",
        "source": "derived from Stage 1/2 planet profile plus terrain overrides",
        "terrain_generation_mode": getattr(config, "terrain_generation_mode", "procedural_legacy"),
        "terrain_style": style,
        "supercontinent_tendency": supercontinent_mode,
        "derived_supercontinent_score": round(supercontinent_score, 3),
        "fragmentation_tendency": round(_clamp(fragmentation, 0.0, 1.0), 3),
        "target_plate_count": max(3, plate_count),
        "coastline_complexity": round(_clamp(override_float("coastline_complexity", coastline_complexity), 0.0, 1.0), 3),
        "island_density": round(_clamp(override_float("island_density", island_density), 0.0, 1.0), 3),
        "shelf_width_factor": round(_clamp(override_float("shelf_width_factor", shelf_width), 0.0, 2.0), 3),
        "coastal_ruggedness": round(_clamp(override_float("coastal_ruggedness", coastal_ruggedness), 0.0, 1.0), 3),
        "fjord_tendency": round(_clamp(override_float("fjord_tendency", fjord_tendency), 0.0, 1.0), 3),
        "coastal_plain_bias": round(_clamp(override_float("coastal_plain_bias", coastal_plain_bias), 0.0, 1.0), 3),
        "island_shape_irregularity": round(_clamp(override_float("island_shape_irregularity", island_shape_irregularity), 0.0, 1.0), 3),
        "mountain_belt_strength": round(_clamp(override_float("mountain_belt_strength", mountain_strength), 0.0, 3.0), 3),
        "rift_strength": round(_clamp(override_float("rift_strength", rift_strength), 0.0, 1.0), 3),
        "interior_relief": round(_clamp(override_float("interior_relief", interior_relief), 0.0, 1.0), 3),
        "erosion_deposition_strength": round(_clamp(override_float("erosion_deposition_strength", deposition_strength), 0.0, 1.0), 3),
        "erosion_deposition_multiplier": round(_clamp(override_float("erosion_deposition_multiplier", 1.35), 0.0, 3.0), 3),
        "continental_shelf_strength": round(_clamp(override_float("continental_shelf_strength", 1.65), 0.0, 3.5), 3),
        # Update 32: v4 controls must be part of the resolved terrain-control
        # contract, not only persisted in stage_overrides.json.  Earlier v4
        # builds wrote these CLI/UI values but derive_terrain_controls did not
        # return them, so plate_history_v4 fell back to neutral 1.0 and runs
        # with 1.0 vs 2.0 could be identical.
        "v4_topology_strength": round(_clamp(override_float("v4_topology_strength", 1.0), 0.0, 2.5), 3),
        "v4_island_strength": round(_clamp(override_float("v4_island_strength", 1.0), 0.0, 2.8), 3),
        "v4_rift_strength": round(_clamp(override_float("v4_rift_strength", 1.0), 0.0, 2.5), 3),
        "deposition_strength": round(_clamp(override_float("deposition_strength", deposition_strength), 0.0, 1.0), 3),
        "valley_carving_strength": round(_clamp(override_float("valley_carving_strength", valley_carving_strength), 0.0, 1.0), 3),
        "sediment_supply_strength": round(_clamp(override_float("sediment_supply_strength", sediment_supply_strength), 0.0, 1.0), 3),
        "coastal_plain_strength": round(_clamp(override_float("coastal_plain_strength", coastal_plain_strength), 0.0, 1.0), 3),
        "alluvial_fan_strength": round(_clamp(override_float("alluvial_fan_strength", alluvial_fan_strength), 0.0, 1.0), 3),
        "floodplain_strength": round(_clamp(override_float("floodplain_strength", floodplain_strength), 0.0, 1.0), 3),
        "terrain_maturity": round(_clamp(override_float("terrain_maturity", terrain_maturity), 0.0, 1.0), 3),
        "plate_motion_speed": round(_clamp(override_float("plate_motion_speed", plate_motion_speed), 0.0, 1.5), 3),
        "plate_motion_chaos": round(_clamp(override_float("plate_motion_chaos", plate_motion_chaos), 0.0, 1.0), 3),
        "convergence_bias": round(_clamp(override_float("convergence_bias", convergence_bias), 0.0, 1.0), 3),
        "divergence_bias": round(_clamp(override_float("divergence_bias", divergence_bias), 0.0, 1.0), 3),
        "transform_bias": round(_clamp(override_float("transform_bias", transform_bias), 0.0, 1.0), 3),
        "diagnostic_detail": override_text("diagnostic_detail", "standard"),
        "derivation_inputs": {
            "crustal_asymmetry_bias": asym,
            "tectonic_energy_bias": tectonic_bias,
            "impact_history": impact,
            "volatile_delivery": volatile_delivery,
            "ocean_fraction_target": ocean,
            "internal_heat": getattr(geology, "internal_heat", None),
            "volcanism": getattr(geology, "volcanism", None),
            "erosion": getattr(geology, "erosion", None),
            "surface_roughness": getattr(geology, "surface_roughness", None),
            "map_width": config.map_width,
            "map_height": config.map_height,
            "terrain_generation_mode": getattr(config, "terrain_generation_mode", "procedural_legacy"),
        },
        "notes": [
            "Terrain generation mode is now explicit: procedural_legacy is the current proven backend; plate_tectonic_v1 starts the new plate-tectonic architecture as a safe scaffold; plate_history_v3 is the stable continuous-field baseline; plate_history_v4 is the recommended conservative v4 terrain model built on stable v3.",
            "These controls are derived biases, not a hard geologic simulation. The current terrain generator remains procedural but now exposes a reviewable terrain contract.",
            "Supercontinents are not banned. They become more likely when profile-derived asymmetry is high and fragmentation is low, unless manually overridden.",
        ],
    }


def _terrain_arrays(terrain: TerrainMap):
    import numpy as np
    elev = np.asarray(terrain.elevation_m, dtype=np.float32)
    land = np.asarray(terrain.is_land, dtype=bool)
    return elev, land


def _coast_mask(land):
    import numpy as np
    north = np.vstack((land[0:1, :], land[:-1, :]))
    south = np.vstack((land[1:, :], land[-1:, :]))
    west = np.roll(land, 1, axis=1)
    east = np.roll(land, -1, axis=1)
    return land & ((~north) | (~south) | (~west) | (~east))


def _component_stats(land):
    import numpy as np
    from scipy import ndimage
    structure = np.array([[0,1,0],[1,1,1],[0,1,0]], dtype=np.uint8)
    labels, count = ndimage.label(land, structure=structure)
    if count <= 0:
        return labels, {"landmass_count": 0, "largest_landmass_share_of_land": 0.0, "largest_landmass_cells": 0, "island_count": 0, "small_island_count": 0}
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    land_cells = int(land.sum())
    largest = int(sizes.max()) if sizes.size else 0
    world_cells = int(land.size)
    island_limit = max(4, int(world_cells * 0.0038))
    small_limit = max(3, int(world_cells * 0.00065))
    island_count = int(np.sum((sizes > 0) & (sizes <= island_limit)))
    small_count = int(np.sum((sizes > 0) & (sizes <= small_limit)))
    nonzero = [int(v) for v in sizes.tolist() if int(v) > 0]
    top_sizes = sorted(nonzero, reverse=True)[:10]
    top_shares = [round(v / max(1, land_cells), 4) for v in top_sizes]
    continent_like = [v for v in nonzero if v > island_limit]
    island_like = [v for v in nonzero if v <= island_limit]
    return labels, {
        "landmass_count": int(count),
        "largest_landmass_share_of_land": round(largest / max(1, land_cells), 4),
        "largest_landmass_cells": largest,
        "top_10_landmass_cells": top_sizes,
        "top_10_landmass_share_of_land": top_shares,
        "continent_like_landmass_count": int(len(continent_like)),
        "continental_land_share": round(sum(continent_like) / max(1, land_cells), 4),
        "island_land_share": round(sum(island_like) / max(1, land_cells), 4),
        "island_count": island_count,
        "small_island_count": small_count,
        "island_limit_cells": int(island_limit),
        "small_island_limit_cells": int(small_limit),
    }


def _straightness_proxy(mask) -> float:
    import numpy as np
    if mask is None or not np.any(mask):
        return 0.0
    gy, gx = np.gradient(mask.astype(np.float32))
    angles = np.arctan2(gy, gx)
    # Concentration of doubled angles approximates directional alignment.
    c = float(np.mean(np.cos(2 * angles[mask])))
    s = float(np.mean(np.sin(2 * angles[mask])))
    return round(math.sqrt(c*c + s*s), 3)


def build_terrain_review(
    terrain: TerrainMap,
    planet: Planet,
    hydrosphere: Hydrosphere,
    geology: GeologyState,
    config: PlanetProfileConfig,
    *,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Build review metadata and sub-stage diagnostics for the current terrain."""
    import numpy as np

    elev, land = _terrain_arrays(terrain)
    coast = _coast_mask(land)
    labels, comp = _component_stats(land)
    gy, gx = np.gradient(elev)
    slope = np.sqrt(gx * gx + gy * gy)
    land_slope = slope[land]
    ocean = ~land
    north_land = float(np.mean(land[: land.shape[0] // 2, :])) if land.shape[0] > 1 else 0.0
    south_land = float(np.mean(land[land.shape[0] // 2 :, :])) if land.shape[0] > 1 else 0.0
    west_land = float(np.mean(land[:, : land.shape[1] // 2])) if land.shape[1] > 1 else 0.0
    east_land = float(np.mean(land[:, land.shape[1] // 2 :])) if land.shape[1] > 1 else 0.0
    coastline_cells = int(coast.sum())
    land_cells = int(land.sum())
    coastline_complexity = coastline_cells / max(1.0, math.sqrt(max(1, land_cells)))
    rows = land.shape[0]
    lat_values = np.linspace(90.0 - 90.0 / rows, -90.0 + 90.0 / rows, rows, dtype=np.float32)[:, None]
    tropical_land_share = round(float(np.sum(land & (np.abs(lat_values) <= 23.5))) / max(1, land_cells), 4)
    temperate_land_share = round(float(np.sum(land & (np.abs(lat_values) > 23.5) & (np.abs(lat_values) <= 66.5))) / max(1, land_cells), 4)
    polar_land_share = round(float(np.sum(land & (np.abs(lat_values) > 66.5))) / max(1, land_cells), 4)
    try:
        from scipy import ndimage
        coast_distance_cells = ndimage.distance_transform_edt(~coast)
        land_coast_distance = coast_distance_cells[land]
        avg_distance_from_coast_cells = round(float(np.mean(land_coast_distance)) if land_coast_distance.size else 0.0, 2)
        interior_land_share = round(float(np.mean(land_coast_distance > max(8, min(48, land.shape[1] // 64)))) if land_coast_distance.size else 0.0, 4)
    except Exception:
        avg_distance_from_coast_cells = 0.0
        interior_land_share = 0.0

    plate = None
    boundary = None
    province_type = None
    province_age = None
    boundary_strength = None
    boundary_width = None
    crust = None
    if terrain.tectonic_plate_id is not None:
        plate = np.asarray(terrain.tectonic_plate_id, dtype=np.int32)
    if terrain.tectonic_boundary_class is not None:
        boundary = np.asarray(terrain.tectonic_boundary_class, dtype=np.int32)
    if getattr(terrain, "tectonic_province_type", None) is not None:
        province_type = np.asarray(terrain.tectonic_province_type, dtype=np.int32)
    if getattr(terrain, "tectonic_province_age_x1000", None) is not None:
        province_age = np.asarray(terrain.tectonic_province_age_x1000, dtype=np.float32) / 1000.0
    if getattr(terrain, "tectonic_boundary_strength_x1000", None) is not None:
        boundary_strength = np.asarray(terrain.tectonic_boundary_strength_x1000, dtype=np.float32) / 1000.0
    if getattr(terrain, "tectonic_boundary_width_x1000", None) is not None:
        boundary_width = np.asarray(terrain.tectonic_boundary_width_x1000, dtype=np.float32) / 1000.0
    if terrain.crust_type is not None:
        crust = np.asarray(terrain.crust_type, dtype=np.int32)

    plate_tectonic_type = np.asarray(getattr(terrain, "plate_tectonic_plate_type", []), dtype=np.int32) if getattr(terrain, "plate_tectonic_plate_type", None) is not None else None

    def _relief_field(attr_name: str):
        value = getattr(terrain, attr_name, None)
        if value is None:
            return None
        arr = np.asarray(value, dtype=np.float32) / 1000.0
        return arr if arr.size else None

    relief_mountain = _relief_field("terrain_mountain_strength_x1000")
    relief_basin = _relief_field("terrain_basin_field_x1000")
    relief_rift = _relief_field("terrain_rift_field_x1000")
    relief_interior = _relief_field("terrain_interior_relief_x1000")
    relief_shield = _relief_field("terrain_shield_highland_x1000")
    relief_plateau = _relief_field("terrain_plateau_x1000")
    shelf_width_field = _relief_field("terrain_shelf_width_x1000")
    coast_ruggedness_field = _relief_field("terrain_coast_ruggedness_x1000")
    ocean_floor_class = np.asarray(getattr(terrain, "terrain_ocean_floor_class", []), dtype=np.int32) if getattr(terrain, "terrain_ocean_floor_class", None) is not None else None
    mid_ocean_ridge_field = _relief_field("terrain_mid_ocean_ridge_x1000")
    trench_field = _relief_field("terrain_trench_x1000")
    fracture_zone_field = _relief_field("terrain_fracture_zone_x1000")
    seamount_field = _relief_field("terrain_seamount_x1000")
    island_shape_complexity_field = _relief_field("terrain_island_shape_complexity_x1000")
    erosion_strength_field = _relief_field("terrain_erosion_strength_x1000")
    deposition_field = _relief_field("terrain_deposition_field_x1000")
    valley_corridor_field = _relief_field("terrain_valley_corridor_x1000")
    sediment_supply_field = _relief_field("terrain_sediment_supply_x1000")
    coastal_plain_field = _relief_field("terrain_coastal_plain_x1000")
    alluvial_fan_field = _relief_field("terrain_alluvial_fan_x1000")
    floodplain_field = _relief_field("terrain_floodplain_x1000")
    terrain_maturity_field = _relief_field("terrain_maturity_x1000")
    relief_delta_m = np.asarray(getattr(terrain, "terrain_relief_delta_m", []), dtype=np.float32) if getattr(terrain, "terrain_relief_delta_m", None) is not None else None
    coast_style = np.asarray(getattr(terrain, "terrain_coast_style_class", []), dtype=np.int32) if getattr(terrain, "terrain_coast_style_class", None) is not None else None
    island_origin = np.asarray(getattr(terrain, "terrain_island_origin_class", []), dtype=np.int32) if getattr(terrain, "terrain_island_origin_class", None) is not None else None
    plate_continental_crust = _relief_field("plate_tectonic_continental_crust_x1000")
    plate_craton_core = _relief_field("plate_tectonic_craton_core_x1000")
    plate_microplate = _relief_field("plate_tectonic_microplate_x1000")
    plate_motion_speed = _relief_field("plate_tectonic_speed_x1000")
    plate_convergence = _relief_field("plate_tectonic_convergence_x1000")
    plate_divergence = _relief_field("plate_tectonic_divergence_x1000")
    plate_transform = _relief_field("plate_tectonic_transform_x1000")
    plate_boundary_native = np.asarray(getattr(terrain, "plate_tectonic_boundary_class", []), dtype=np.int32) if getattr(terrain, "plate_tectonic_boundary_class", None) is not None else None
    plate_subduction = np.asarray(getattr(terrain, "plate_tectonic_subduction_polarity", []), dtype=np.int32) if getattr(terrain, "plate_tectonic_subduction_polarity", None) is not None else None
    plate_ocean_floor_class = np.asarray(getattr(terrain, "plate_tectonic_ocean_floor_class", []), dtype=np.int32) if getattr(terrain, "plate_tectonic_ocean_floor_class", None) is not None else None
    plate_ocean_crust_age = _relief_field("plate_tectonic_ocean_crust_age_x1000")
    plate_mid_ocean_ridge = _relief_field("plate_tectonic_mid_ocean_ridge_x1000")
    plate_trench = _relief_field("plate_tectonic_trench_x1000")
    plate_fracture_zone = _relief_field("plate_tectonic_fracture_zone_x1000")
    plate_abyssal_plain = _relief_field("plate_tectonic_abyssal_plain_x1000")
    plate_seamount = _relief_field("plate_tectonic_seamount_x1000")
    plate_orogeny = _relief_field("plate_tectonic_orogeny_strength_x1000")
    plate_volcanic_arc = _relief_field("plate_tectonic_volcanic_arc_x1000")
    plate_continental_rift = _relief_field("plate_tectonic_continental_rift_x1000")
    plate_foreland_basin = _relief_field("plate_tectonic_foreland_basin_x1000")
    plate_craton_shield = _relief_field("plate_tectonic_craton_shield_x1000")
    plate_accreted_terrane = _relief_field("plate_tectonic_accreted_terrane_x1000")
    plate_plateau_uplift = _relief_field("plate_tectonic_plateau_uplift_x1000")
    plate_relief_delta = np.asarray(getattr(terrain, "plate_tectonic_relief_delta_m", []), dtype=np.float32) if getattr(terrain, "plate_tectonic_relief_delta_m", None) is not None else None
    plate_margin_class = np.asarray(getattr(terrain, "plate_tectonic_margin_class", []), dtype=np.int32) if getattr(terrain, "plate_tectonic_margin_class", None) is not None else None
    plate_shelf_width = _relief_field("plate_tectonic_shelf_width_x1000")
    plate_active_margin = _relief_field("plate_tectonic_active_margin_x1000")
    plate_passive_margin = _relief_field("plate_tectonic_passive_margin_x1000")
    plate_rifted_margin = _relief_field("plate_tectonic_rifted_margin_x1000")
    plate_island_arc = _relief_field("plate_tectonic_island_arc_x1000")
    plate_coastal_plain = _relief_field("plate_tectonic_coastal_plain_x1000")
    plate_coast_ruggedness = _relief_field("plate_tectonic_coast_ruggedness_x1000")
    plate_island_origin = np.asarray(getattr(terrain, "plate_tectonic_island_origin_class", []), dtype=np.int32) if getattr(terrain, "plate_tectonic_island_origin_class", None) is not None else None
    plate_coast_delta = np.asarray(getattr(terrain, "plate_tectonic_coast_delta_m", []), dtype=np.float32) if getattr(terrain, "plate_tectonic_coast_delta_m", None) is not None else None
    plate_backend_integration = _relief_field("plate_tectonic_backend_integration_x1000")
    plate_hydrology_readiness = _relief_field("plate_tectonic_hydrology_readiness_x1000")
    plate_legacy_dependency = _relief_field("plate_tectonic_legacy_dependency_x1000")
    plate_problem_class = np.asarray(getattr(terrain, "plate_tectonic_problem_class", []), dtype=np.int32) if getattr(terrain, "plate_tectonic_problem_class", None) is not None else None

    generation_meta = getattr(terrain, "terrain_diagnostics", None) or {}
    if not isinstance(generation_meta, dict):
        generation_meta = {}
    ocean_fit = generation_meta.get("ocean_target_fit", {}) if isinstance(generation_meta.get("ocean_target_fit"), dict) else {}
    controls = derive_terrain_controls(planet, hydrosphere, geology, config, output_dir=output_dir)
    if isinstance(generation_meta.get("generation_controls"), dict):
        # Preserve the exact control values the terrain generator used.  This is
        # especially useful when a user changes overrides and reruns terrain.
        controls = {**controls, **generation_meta.get("generation_controls", {})}
    target_ocean = float(hydrosphere.ocean_fraction_target)
    ocean_target_error = float(terrain.ocean_fraction) - target_ocean
    warnings: list[dict[str, str]] = []
    largest_share = comp.get("largest_landmass_share_of_land", 0.0)
    if largest_share > 0.92 and controls.get("supercontinent_tendency") in {"suppressed", "rare"}:
        warnings.append({"level": "strong", "message": "Largest landmass contains more than 92% of all land even though supercontinent tendency is low/suppressed."})
    elif largest_share > 0.90:
        warnings.append({"level": "info", "message": "This terrain is effectively supercontinent-dominated. That is allowed, but review whether the planet profile supports it."})
    if abs(ocean_target_error) > 0.05:
        warnings.append({"level": "strong", "message": f"Final ocean fraction differs from target by {ocean_target_error:+.3f}. Review foundation target-fit diagnostics before continuing."})
    elif abs(ocean_target_error) > 0.025:
        warnings.append({"level": "info", "message": f"Final ocean fraction is close but not exact: {terrain.ocean_fraction:.3f} vs target {target_ocean:.3f}."})
    if terrain.ocean_fraction > 0.86:
        warnings.append({"level": "strong", "message": "Final ocean fraction is very high; exposed land and inland climate diagnostics may be limited."})
    if terrain.ocean_fraction < 0.35:
        warnings.append({"level": "strong", "message": "Final ocean fraction is low; climate/hydrology may produce large dry interiors unless atmosphere/terrain compensate."})
    if comp.get("island_count", 0) < 4 and float(controls.get("island_density", 0.0)) > 0.55:
        warnings.append({"level": "info", "message": "Derived island density is moderate/high, but final map has few small island components."})
    if land_slope.size and float(np.percentile(land_slope, 75)) < 55:
        warnings.append({"level": "info", "message": "Continental relief appears low; large interiors may produce straight/simple river courses."})

    province_stats: dict[str, Any] = {}
    if plate is not None and plate.size:
        plate_counts = np.bincount(plate.ravel())
        nonzero_plate_counts = plate_counts[plate_counts > 0]
        province_stats.update({
            "diagnostic_plate_count": int(nonzero_plate_counts.size),
            "largest_province_share": round(float(nonzero_plate_counts.max()) / max(1, int(plate.size)), 4) if nonzero_plate_counts.size else 0.0,
            "province_size_diversity_score": _shannon_diversity(plate),
        })
    if province_type is not None:
        province_stats.update(_class_share_stats(province_type, PROVINCE_TYPE_LABELS, "province"))
        continental_codes = {2, 3, 4, 6, 7, 8}
        oceanic_codes = {0, 1, 5}
        province_stats.update({
            "province_type_diversity_score": _shannon_diversity(province_type),
            "continental_or_margin_province_share": round(float(np.mean(np.isin(province_type, list(continental_codes)))), 4),
            "oceanic_or_arc_province_share": round(float(np.mean(np.isin(province_type, list(oceanic_codes)))), 4),
            "microcontinent_terrane_share": round(float(np.mean(province_type == 6)), 4),
        })
    if province_age is not None and province_age.size:
        province_stats.update({
            "mean_province_age_proxy": round(float(np.mean(province_age)), 3),
            "old_stable_province_share": round(float(np.mean(province_age > 0.65)), 4),
            "young_active_province_share": round(float(np.mean(province_age < 0.35)), 4),
        })
    plate_setup_meta = generation_meta.get("plate_tectonic_v1", {}) if isinstance(generation_meta.get("plate_tectonic_v1"), dict) else {}
    if plate_tectonic_type is not None and plate_tectonic_type.size:
        province_stats.update(_class_share_stats(plate_tectonic_type, PLATE_TECTONIC_TYPE_LABELS, "plate_type"))
        province_stats.update({
            "native_plate_setup_effective_plate_count": plate_setup_meta.get("effective_plate_count"),
            "native_plate_setup_microplate_count": plate_setup_meta.get("microplate_count"),
            "native_plate_setup_largest_plate_area_share": plate_setup_meta.get("largest_plate_area_share"),
            "native_plate_setup_craton_core_share": plate_setup_meta.get("craton_core_share"),
            "native_plate_setup_backend_status": plate_setup_meta.get("backend_status"),
        })
    if plate_continental_crust is not None:
        province_stats["native_mean_continental_crust_fraction"] = round(float(np.mean(plate_continental_crust)), 4)
        province_stats["native_continental_crust_high_share"] = round(float(np.mean(plate_continental_crust > 0.55)), 4)
    if plate_craton_core is not None:
        province_stats["native_craton_core_share"] = round(float(np.mean(plate_craton_core > 0.35)), 4)
    if plate_microplate is not None:
        province_stats["native_microplate_cell_share"] = round(float(np.mean(plate_microplate > 0.35)), 4)

    boundary_stats: dict[str, Any] = {}
    if boundary is not None:
        total = max(1, int(boundary.size))
        active_mask = boundary > 0
        boundary_stats = {
            **_class_share_stats(boundary, BOUNDARY_CLASS_LABELS, "boundary"),
            "boundary_activity_share": round(float(np.mean(active_mask)), 4),
            "active_margin_share": round(float(np.mean(np.isin(boundary, [1, 2, 3, 5, 6]))), 4),
            "passive_margin_share": round(float(np.mean(boundary == 4)), 4),
            "rift_boundary_share": round(float(np.mean(boundary == 2)), 4),
            "subduction_collision_boundary_share": round(float(np.mean(np.isin(boundary, [1, 6]))), 4),
            "diffuse_suture_share": round(float(np.mean(boundary == 5)), 4),
            "boundary_neatness_proxy": _straightness_proxy(active_mask),
            "source_cells": total,
        }
        if boundary_strength is not None and boundary_strength.size:
            boundary_stats.update({
                "mean_boundary_strength": round(float(np.mean(boundary_strength[active_mask])) if np.any(active_mask) else 0.0, 3),
                "strong_boundary_share": round(float(np.mean(boundary_strength > 0.62)), 4),
            })
        if boundary_width is not None and boundary_width.size:
            boundary_stats.update({
                "mean_boundary_width_proxy": round(float(np.mean(boundary_width[active_mask])) if np.any(active_mask) else 0.0, 3),
                "wide_boundary_share": round(float(np.mean(boundary_width > 0.55)), 4),
            })

    if plate_boundary_native is not None and plate_boundary_native.size:
        boundary_stats.update({
            **_class_share_stats(plate_boundary_native, BOUNDARY_CLASS_LABELS, "native_plate_boundary"),
            "native_plate_boundary_activity_share": round(float(np.mean(plate_boundary_native > 0)), 4),
            "native_plate_convergent_or_arc_share": round(float(np.mean(np.isin(plate_boundary_native, [1, 6]))), 4),
            "native_plate_divergent_share": round(float(np.mean(plate_boundary_native == 2)), 4),
            "native_plate_transform_share": round(float(np.mean(plate_boundary_native == 3)), 4),
            "native_plate_passive_or_diffuse_share": round(float(np.mean(np.isin(plate_boundary_native, [4, 5]))), 4),
        })
    for prefix, arr in [
        ("native_plate_motion_speed", plate_motion_speed),
        ("native_plate_convergence", plate_convergence),
        ("native_plate_divergence", plate_divergence),
        ("native_plate_transform", plate_transform),
    ]:
        if arr is not None and getattr(arr, "size", 0):
            boundary_stats[f"{prefix}_mean"] = round(float(np.mean(arr)), 3)
            boundary_stats[f"{prefix}_strong_share"] = round(float(np.mean(arr > 0.55)), 4)
    if plate_subduction is not None and plate_subduction.size:
        boundary_stats.update(_class_share_stats(plate_subduction, PLATE_SUBDUCTION_POLARITY_LABELS, "subduction_polarity"))

    crust_stats: dict[str, Any] = {}
    if crust is not None:
        crust_stats = {f"class_{i}_share": round(float(np.mean(crust == i)), 4) for i in range(4)}

    ocean_floor_stats: dict[str, Any] = {}
    if ocean_floor_class is not None and ocean_floor_class.size:
        ocean_floor_stats.update(_class_share_stats(ocean_floor_class, OCEAN_FLOOR_LABELS, "ocean_floor"))
        ocean_cells = ocean_floor_class > 0
        ocean_total = max(1, int(np.sum(ocean_cells)))
        ocean_floor_stats.update({
            "ocean_floor_class_diversity_score": _shannon_diversity(ocean_floor_class[ocean_cells]) if np.any(ocean_cells) else 0.0,
            "mid_ocean_ridge_share_of_ocean": round(float(np.sum(ocean_floor_class == 2)) / ocean_total, 4),
            "trench_share_of_ocean": round(float(np.sum(ocean_floor_class == 3)) / ocean_total, 4),
            "fracture_zone_share_of_ocean": round(float(np.sum(ocean_floor_class == 4)) / ocean_total, 4),
            "seamount_hotspot_share_of_ocean": round(float(np.sum(ocean_floor_class == 5)) / ocean_total, 4),
        })
    for prefix, arr in [
        ("mid_ocean_ridge_field", mid_ocean_ridge_field),
        ("trench_field", trench_field),
        ("fracture_zone_field", fracture_zone_field),
        ("seamount_field", seamount_field),
    ]:
        if arr is not None and getattr(arr, "size", 0):
            ocean_floor_stats[f"{prefix}_mean"] = round(float(np.mean(arr)), 3)
            ocean_floor_stats[f"{prefix}_strong_share"] = round(float(np.mean(arr > 0.55)), 4)
    if plate_ocean_floor_class is not None and plate_ocean_floor_class.size:
        plate_ocean_cells = plate_ocean_floor_class > 0
        plate_ocean_total = max(1, int(np.sum(plate_ocean_cells)))
        ocean_floor_stats.update({
            **_class_share_stats(plate_ocean_floor_class, OCEAN_FLOOR_LABELS, "native_plate_ocean_floor"),
            "native_plate_ocean_floor_diversity_score": _shannon_diversity(plate_ocean_floor_class[plate_ocean_cells]) if np.any(plate_ocean_cells) else 0.0,
            "native_plate_mid_ocean_ridge_share_of_ocean": round(float(np.sum(plate_ocean_floor_class == 2)) / plate_ocean_total, 4),
            "native_plate_trench_share_of_ocean": round(float(np.sum(plate_ocean_floor_class == 3)) / plate_ocean_total, 4),
            "native_plate_fracture_zone_share_of_ocean": round(float(np.sum(plate_ocean_floor_class == 4)) / plate_ocean_total, 4),
            "native_plate_seamount_share_of_ocean": round(float(np.sum(plate_ocean_floor_class == 5)) / plate_ocean_total, 4),
        })
    for prefix, arr in [
        ("native_plate_ocean_crust_age", plate_ocean_crust_age),
        ("native_plate_mid_ocean_ridge", plate_mid_ocean_ridge),
        ("native_plate_trench", plate_trench),
        ("native_plate_fracture_zone", plate_fracture_zone),
        ("native_plate_abyssal_plain", plate_abyssal_plain),
        ("native_plate_seamount", plate_seamount),
    ]:
        if arr is not None and getattr(arr, "size", 0):
            ocean_floor_stats[f"{prefix}_mean"] = round(float(np.mean(arr)), 3)
            ocean_floor_stats[f"{prefix}_strong_share"] = round(float(np.mean(arr > 0.55)), 4)

    relief_stats: dict[str, Any] = {}
    def _relief_metric(prefix: str, arr):
        if arr is None or not getattr(arr, "size", 0):
            return
        relief_stats[f"{prefix}_mean"] = round(float(np.mean(arr)), 3)
        relief_stats[f"{prefix}_strong_share"] = round(float(np.mean(arr > 0.62)), 4)
        relief_stats[f"{prefix}_moderate_share"] = round(float(np.mean(arr > 0.32)), 4)
    _relief_metric("mountain_field", relief_mountain)
    _relief_metric("basin_field", relief_basin)
    _relief_metric("rift_field", relief_rift)
    _relief_metric("interior_relief_field", relief_interior)
    _relief_metric("shield_highland_field", relief_shield)
    _relief_metric("plateau_field", relief_plateau)
    _relief_metric("native_plate_orogeny", plate_orogeny)
    _relief_metric("native_plate_volcanic_arc", plate_volcanic_arc)
    _relief_metric("native_plate_continental_rift", plate_continental_rift)
    _relief_metric("native_plate_foreland_basin", plate_foreland_basin)
    _relief_metric("native_plate_craton_shield", plate_craton_shield)
    _relief_metric("native_plate_accreted_terrane", plate_accreted_terrane)
    _relief_metric("native_plate_plateau_uplift", plate_plateau_uplift)
    if plate_relief_delta is not None and getattr(plate_relief_delta, "size", 0):
        land_delta = plate_relief_delta[land] if plate_relief_delta.shape == land.shape else plate_relief_delta.ravel()
        if getattr(land_delta, "size", 0):
            relief_stats["native_plate_mean_relief_delta_m"] = round(float(np.mean(land_delta)), 2)
            relief_stats["native_plate_mean_abs_relief_delta_m"] = round(float(np.mean(np.abs(land_delta))), 2)
            relief_stats["native_plate_strong_uplift_land_share"] = round(float(np.mean(land_delta > 350)), 4)
            relief_stats["native_plate_strong_subsidence_land_share"] = round(float(np.mean(land_delta < -180)), 4)

    if relief_stats.get("native_plate_orogeny_mean", 0.0) < 0.018 and boundary_stats.get("native_plate_convergent_or_arc_share", 0.0) > 0.04:
        warnings.append({"level": "info", "message": "Native plate convergence exists, but plate-derived orogeny is weak. Review Plate Terrain 4 relief controls before relying on mountain placement."})
    if relief_stats.get("native_plate_continental_rift_mean", 0.0) < 0.012 and boundary_stats.get("native_plate_divergent_share", 0.0) > 0.04:
        warnings.append({"level": "info", "message": "Native plate divergent boundaries exist, but continental rift expression is weak."})
    if relief_stats.get("native_plate_mean_abs_relief_delta_m", 0.0) > 650:
        warnings.append({"level": "info", "message": "Native plate relief changed elevation strongly. Inspect Plate Terrain 4 relief-delta diagnostics for over-uplift or over-subsidence."})

    if province_stats.get("largest_province_share", 0.0) > 0.22:
        warnings.append({"level": "info", "message": "Tectonic/province layout is dominated by one large diagnostic province. This may be plausible for quiet/asymmetric worlds but should be reviewed."})
    if province_stats.get("province_type_diversity_score", 1.0) < 0.45:
        warnings.append({"level": "info", "message": "Province type diversity is low; later terrain may lack geological variety unless this matches the terrain style."})
    if boundary_stats.get("boundary_neatness_proxy", 0.0) > 0.52:
        warnings.append({"level": "info", "message": "Boundary network has a high alignment/neatness proxy; watch for overly smooth arcs or straight mountain chains."})
    if boundary_stats.get("active_margin_share", 0.0) > 0.24 and str(controls.get("terrain_style", "")).lower() not in {"rugged_tectonic_world", "volcanic_island_arc_world"}:
        warnings.append({"level": "info", "message": "Active boundary share is high for the current terrain style. Review arc/rift density before continuing."})
    if boundary_stats.get("rift_boundary_share", 0.0) < 0.015 and float(controls.get("fragmentation_tendency", 0.0) or 0.0) > 0.65:
        warnings.append({"level": "info", "message": "Fragmentation tendency is high but rift boundary share is low; future continent breakup may be underrepresented."})
    if boundary_stats.get("native_plate_boundary_activity_share", 0.0) > 0 and boundary_stats.get("native_plate_motion_speed_mean", 0.0) < 0.08:
        warnings.append({"level": "info", "message": "Native plate motion speed is low; plate_tectonic_v1 boundaries may be too diffuse for active terrain until controls are raised."})
    if boundary_stats.get("native_plate_convergent_or_arc_share", 0.0) < 0.01 and float(controls.get("convergence_bias", 0.0) or 0.0) > 0.55:
        warnings.append({"level": "info", "message": "Convergence bias is moderate/high, but native plate convergence/collision boundary share is low."})
    if boundary_stats.get("native_plate_divergent_share", 0.0) < 0.01 and float(controls.get("divergence_bias", 0.0) or 0.0) > 0.55:
        warnings.append({"level": "info", "message": "Divergence bias is moderate/high, but native plate rift/spreading boundary share is low."})
    if ocean_floor_stats.get("mid_ocean_ridge_share_of_ocean", 0.0) < 0.004 and terrain.ocean_fraction > 0.45:
        warnings.append({"level": "info", "message": "Ocean-floor ridge expression is weak; inspect the ocean-floor class and ridge diagnostics."})
    if ocean_floor_stats.get("trench_share_of_ocean", 0.0) < 0.002 and boundary_stats.get("subduction_collision_boundary_share", 0.0) > 0.04:
        warnings.append({"level": "info", "message": "Subduction/collision boundaries exist but trench expression is weak in the ocean-floor diagnostics."})
    if ocean_floor_stats.get("native_plate_mid_ocean_ridge_share_of_ocean", 1.0) < 0.004 and boundary_stats.get("native_plate_divergent_share", 0.0) > 0.01 and terrain.ocean_fraction > 0.45:
        warnings.append({"level": "info", "message": "Native plate divergent boundaries exist, but native plate ridge expression is weak."})
    if ocean_floor_stats.get("native_plate_trench_share_of_ocean", 1.0) < 0.002 and boundary_stats.get("native_plate_convergent_or_arc_share", 0.0) > 0.01 and terrain.ocean_fraction > 0.45:
        warnings.append({"level": "info", "message": "Native plate convergent/arc boundaries exist, but native plate trench expression is weak."})
    if relief_stats.get("interior_relief_field_mean", 1.0) < 0.055 and float(terrain.land_fraction) > 0.28:
        warnings.append({"level": "info", "message": "Internal relief field is weak; large continents may still have flat interiors and simple river gradients."})
    if relief_stats.get("mountain_field_strong_share", 0.0) > 0.16:
        warnings.append({"level": "info", "message": "Mountain influence is widespread. Review whether mountain belts are too dominant before climate/hydrology stages."})
    if relief_stats.get("rift_field_moderate_share", 0.0) < 0.015 and float(controls.get("rift_strength", 0.0) or 0.0) > 0.60:
        warnings.append({"level": "info", "message": "Rift control is high but generated rift field is weak; continent-breakup corridors may be underrepresented."})

    coast_stats: dict[str, Any] = {}
    if coast_style is not None and coast_style.size:
        coast_cells_mask = coast_style > 0
        coast_total = max(1, int(np.sum(coast_cells_mask)))
        for code, label in COAST_STYLE_LABELS.items():
            if code == 0:
                continue
            key = label.lower().replace(" / ", "_").replace(" or ", "_").replace(" ", "_").replace("-", "_")
            coast_stats[f"coast_style_{code}_{key}_share"] = round(float(np.sum(coast_style == code)) / coast_total, 4)
        coast_stats.update({
            "coast_style_diversity_score": _shannon_diversity(coast_style[coast_cells_mask]) if np.any(coast_cells_mask) else 0.0,
            "rugged_or_fjorded_coast_share": round(float(np.sum(coast_style == 2)) / coast_total, 4),
            "rifted_gulf_coast_share": round(float(np.sum(coast_style == 3)) / coast_total, 4),
            "volcanic_arc_coast_share": round(float(np.sum(coast_style == 4)) / coast_total, 4),
            "smooth_passive_coast_share": round(float(np.sum(coast_style == 1)) / coast_total, 4),
            "shelf_deltaic_plain_coast_share": round(float(np.sum(coast_style == 5)) / coast_total, 4),
        })
    if shelf_width_field is not None and shelf_width_field.size:
        ocean_shelf_values = shelf_width_field[ocean] if shelf_width_field.shape == ocean.shape else shelf_width_field.ravel()
        coast_stats.update({
            "mean_shelf_width_proxy": round(float(np.mean(ocean_shelf_values)) if ocean_shelf_values.size else 0.0, 3),
            "broad_shelf_ocean_share": round(float(np.mean(ocean_shelf_values > 0.32)) if ocean_shelf_values.size else 0.0, 4),
        })
    if coast_ruggedness_field is not None and coast_ruggedness_field.size:
        rugged_values = coast_ruggedness_field[coast] if coast_ruggedness_field.shape == coast.shape else coast_ruggedness_field.ravel()
        coast_stats.update({
            "mean_coast_ruggedness_proxy": round(float(np.mean(rugged_values)) if rugged_values.size else 0.0, 3),
            "strong_coast_ruggedness_share": round(float(np.mean(rugged_values > 0.50)) if rugged_values.size else 0.0, 4),
        })
    if island_origin is not None and island_origin.size:
        island_cells = island_origin > 1
        island_total = max(1, int(np.sum(island_cells)))
        coast_stats.update({
            "island_origin_diversity_score": _shannon_diversity(island_origin[island_cells]) if np.any(island_cells) else 0.0,
            "shelf_island_cell_share": round(float(np.sum(island_origin == 2)) / island_total, 4),
            "volcanic_arc_island_cell_share": round(float(np.sum(island_origin == 3)) / island_total, 4),
            "microcontinent_terrane_island_cell_share": round(float(np.sum(island_origin == 4)) / island_total, 4),
            "hotspot_high_island_cell_share": round(float(np.sum(island_origin == 5)) / island_total, 4),
        })

    if island_shape_complexity_field is not None and island_shape_complexity_field.size:
        active_shape = island_shape_complexity_field[island_shape_complexity_field > 0.01]
        coast_stats.update({
            "mean_island_shape_complexity": round(float(np.mean(active_shape)) if active_shape.size else 0.0, 3),
            "low_complexity_island_cell_share": round(float(np.mean((island_shape_complexity_field > 0.01) & (island_shape_complexity_field < 0.22))), 4),
            "high_complexity_island_cell_share": round(float(np.mean(island_shape_complexity_field > 0.58)), 4),
        })
    if plate_margin_class is not None and plate_margin_class.size:
        margin_cells = plate_margin_class > 0
        margin_total = max(1, int(np.sum(margin_cells)))
        for code, label in PLATE_MARGIN_LABELS.items():
            if code == 0:
                continue
            key = label.lower().replace(" / ", "_").replace(" or ", "_").replace(" ", "_").replace("-", "_")
            coast_stats[f"native_plate_margin_{code}_{key}_share"] = round(float(np.sum(plate_margin_class == code)) / margin_total, 4)
        coast_stats["native_plate_margin_diversity_score"] = _shannon_diversity(plate_margin_class[margin_cells]) if np.any(margin_cells) else 0.0
    def _plate_coast_metric(name: str, arr):
        if arr is None or not getattr(arr, "size", 0):
            return
        values = arr.ravel()
        active_values = values[values > 0.01]
        coast_stats[f"native_plate_{name}_mean"] = round(float(np.mean(active_values)) if active_values.size else 0.0, 3)
        coast_stats[f"native_plate_{name}_coverage_share"] = round(float(np.mean(values > 0.01)), 4)
        coast_stats[f"native_plate_{name}_strong_share"] = round(float(np.mean(values > 0.45)), 4)
    _plate_coast_metric("shelf_width", plate_shelf_width)
    _plate_coast_metric("active_margin", plate_active_margin)
    _plate_coast_metric("passive_margin", plate_passive_margin)
    _plate_coast_metric("rifted_margin", plate_rifted_margin)
    _plate_coast_metric("island_arc", plate_island_arc)
    _plate_coast_metric("coastal_plain", plate_coastal_plain)
    _plate_coast_metric("coast_ruggedness", plate_coast_ruggedness)
    if plate_island_origin is not None and plate_island_origin.size:
        plate_island_cells = plate_island_origin > 1
        total = max(1, int(np.sum(plate_island_cells)))
        coast_stats.update({
            "native_plate_island_origin_diversity_score": _shannon_diversity(plate_island_origin[plate_island_cells]) if np.any(plate_island_cells) else 0.0,
            "native_plate_arc_island_cell_share": round(float(np.sum(plate_island_origin == 3)) / total, 4),
            "native_plate_microcontinent_island_cell_share": round(float(np.sum(plate_island_origin == 4)) / total, 4),
            "native_plate_high_island_cell_share": round(float(np.sum(plate_island_origin == 5)) / total, 4),
        })
    if plate_coast_delta is not None and plate_coast_delta.size:
        active_delta = plate_coast_delta[np.abs(plate_coast_delta) > 0]
        coast_stats.update({
            "native_plate_mean_abs_coast_delta_m": round(float(np.mean(np.abs(active_delta))) if active_delta.size else 0.0, 2),
            "native_plate_strong_coast_delta_share": round(float(np.mean(np.abs(plate_coast_delta) > 220)), 4),
        })
    if coast_stats.get("mean_island_shape_complexity", 1.0) < 0.24 and comp.get("island_count", 0) > 8:
        warnings.append({"level": "info", "message": "Island shape complexity is low; many islands may still look like ovals or overlapping blobs."})

    if coast_stats.get("coast_style_diversity_score", 1.0) < 0.42 and float(controls.get("coastline_complexity", 0.0) or 0.0) > 0.55:
        warnings.append({"level": "info", "message": "Coastline complexity control is moderate/high, but generated coast-style diversity is low."})
    if coast_stats.get("mean_shelf_width_proxy", 0.0) > 0.58 and terrain.ocean_fraction > 0.80:
        warnings.append({"level": "info", "message": "Broad shelves are common on a high-ocean world; inspect for excessive shallow-water margins or shelf islands."})
    if coast_stats.get("rugged_or_fjorded_coast_share", 0.0) < 0.04 and float(controls.get("fjord_tendency", 0.0) or 0.0) > 0.55:
        warnings.append({"level": "info", "message": "Fjord/drowned-valley control is high, but rugged/fjorded coast share is low."})
    if coast_stats.get("island_origin_diversity_score", 0.0) < 0.34 and comp.get("island_count", 0) > 8:
        warnings.append({"level": "info", "message": "Many islands exist, but island-origin diversity is low; islands may still feel too samey."})
    if coast_stats.get("native_plate_margin_diversity_score", 1.0) < 0.35 and coast_stats.get("native_plate_shelf_width_coverage_share", 0.0) > 0.02:
        warnings.append({"level": "info", "message": "Plate-derived margin diversity is low; plate_tectonic_v1 coasts may be too samey until plate margins are tuned."})
    if coast_stats.get("native_plate_shelf_width_mean", 0.0) > 0.42 and coast_stats.get("native_plate_active_margin_mean", 0.0) > 0.18:
        warnings.append({"level": "info", "message": "Plate-derived shelves and active margins overlap strongly; inspect whether active margins still have unrealistically broad shelves."})
    if coast_stats.get("native_plate_island_arc_mean", 0.0) < 0.025 and boundary_stats.get("native_plate_convergent_or_arc_share", 0.0) > 0.02:
        warnings.append({"level": "info", "message": "Native plate convergence exists, but island-arc coast expression is weak."})

    erosion_stats: dict[str, Any] = {}
    def _erosion_metric(prefix: str, arr):
        if arr is None or not getattr(arr, "size", 0):
            return
        active = arr[arr > 0.01]
        erosion_stats[f"{prefix}_mean"] = round(float(np.mean(active)) if active.size else 0.0, 3)
        erosion_stats[f"{prefix}_coverage_share"] = round(float(np.mean(arr > 0.01)), 4)
        erosion_stats[f"{prefix}_strong_share"] = round(float(np.mean(arr > 0.62)), 4)
        erosion_stats[f"{prefix}_moderate_share"] = round(float(np.mean(arr > 0.32)), 4)

    _erosion_metric("erosion_strength", erosion_strength_field)
    _erosion_metric("deposition_field", deposition_field)
    _erosion_metric("valley_corridor", valley_corridor_field)
    _erosion_metric("sediment_supply", sediment_supply_field)
    _erosion_metric("coastal_plain", coastal_plain_field)
    _erosion_metric("alluvial_fan", alluvial_fan_field)
    _erosion_metric("floodplain", floodplain_field)
    _erosion_metric("terrain_maturity", terrain_maturity_field)
    if relief_delta_m is not None and getattr(relief_delta_m, "size", 0):
        changed = np.abs(relief_delta_m) > 25
        erosion_stats.update({
            "mean_relief_delta_m": round(float(np.mean(relief_delta_m)), 2),
            "mean_abs_relief_delta_m": round(float(np.mean(np.abs(relief_delta_m))), 2),
            "strong_erosion_change_share": round(float(np.mean(relief_delta_m < -120)), 4),
            "strong_deposition_change_share": round(float(np.mean(relief_delta_m > 120)), 4),
            "changed_relief_share": round(float(np.mean(changed)), 4),
        })
    erosion_meta = generation_meta.get("erosion_deposition", {}) if isinstance(generation_meta.get("erosion_deposition"), dict) else {}
    if isinstance(erosion_meta, dict):
        for key in ("erosion_control", "deposition_control", "valley_carving_strength", "terrain_maturity"):
            if key in erosion_meta:
                erosion_stats[f"generator_{key}"] = erosion_meta[key]

    if erosion_stats.get("valley_corridor_mean", 1.0) < 0.035 and terrain.land_fraction > 0.18:
        warnings.append({"level": "info", "message": "Valley corridor field is weak; hydrology may still find overly simple or straight river paths."})
    if erosion_stats.get("deposition_field_mean", 0.0) < 0.025 and coast_stats.get("mean_shelf_width_proxy", 0.0) > 0.30:
        warnings.append({"level": "info", "message": "Shelves/coastal margins exist, but deposition field is weak; coastal plains and deltas may be underdeveloped."})
    if erosion_stats.get("mean_abs_relief_delta_m", 0.0) > 420:
        warnings.append({"level": "info", "message": "Erosion/deposition changed relief strongly. Review that mountain barriers were not over-smoothed."})
    if erosion_stats.get("terrain_maturity_mean", 0.0) < 0.12 and float(controls.get("erosion_deposition_strength", 0.0) or 0.0) > 0.55:
        warnings.append({"level": "info", "message": "Erosion/deposition control is high, but terrain maturity field is low; terrain may remain raw/artificial."})

    plate_final_stats: dict[str, Any] = {}
    def _plate_final_metric(prefix: str, arr):
        if arr is None or not getattr(arr, "size", 0):
            return
        values = np.asarray(arr, dtype=np.float32).ravel()
        active = values[values > 0.01]
        plate_final_stats[f"{prefix}_mean"] = round(float(np.mean(active)) if active.size else 0.0, 3)
        plate_final_stats[f"{prefix}_coverage_share"] = round(float(np.mean(values > 0.01)), 4)
        plate_final_stats[f"{prefix}_strong_share"] = round(float(np.mean(values > 0.62)), 4)

    _plate_final_metric("native_plate_backend_integration", plate_backend_integration)
    _plate_final_metric("native_plate_hydrology_readiness", plate_hydrology_readiness)
    _plate_final_metric("native_plate_legacy_dependency", plate_legacy_dependency)
    if plate_problem_class is not None and plate_problem_class.size:
        total_problem = max(1, int(plate_problem_class.size))
        for code, label in PLATE_FINAL_PROBLEM_LABELS.items():
            key = label.lower().replace(" / ", "_").replace(" ", "_").replace("-", "_")
            plate_final_stats[f"native_plate_problem_{code}_{key}_share"] = round(float(np.sum(plate_problem_class == code)) / total_problem, 4)
        plate_final_stats["native_plate_problem_any_share"] = round(float(np.mean(plate_problem_class > 0)), 4)
    plate_meta = generation_meta.get("plate_tectonic_v1", {}) if isinstance(generation_meta.get("plate_tectonic_v1"), dict) else {}
    plate_final_meta = plate_meta.get("final_integration_qa", {}) if isinstance(plate_meta.get("final_integration_qa"), dict) else {}
    if isinstance(plate_final_meta, dict):
        for key, value in plate_final_meta.items():
            if isinstance(value, (int, float, str, bool)):
                plate_final_stats[f"generator_{key}"] = value
    if plate_final_stats.get("native_plate_legacy_dependency_mean", 0.0) > 0.58 and str(controls.get("terrain_generation_mode")) == "plate_tectonic_v1":
        warnings.append({"level": "info", "message": "Plate mode still has high legacy-foundation dependency; this is expected until plate-owned foundation/mask replaces the compatibility backend."})
    if plate_final_stats.get("native_plate_hydrology_readiness_mean", 1.0) < 0.18 and terrain.land_fraction > 0.12 and str(controls.get("terrain_generation_mode")) == "plate_tectonic_v1":
        warnings.append({"level": "info", "message": "Plate-derived hydrology readiness is weak; hydrology should not yet rely only on native plate fields."})

    mountain_mask = land & ((elev > np.percentile(elev[land], 88) if land.any() else False) | (slope > (np.percentile(land_slope, 88) if land_slope.size else 1e9)))
    basin_mask = land & (elev < (np.percentile(elev[land], 25) if land.any() else 0)) & (slope < (np.percentile(land_slope, 45) if land_slope.size else 1e9))
    lowland_mask = land & (elev < 250)
    shelf_mask = ocean & (elev > -350)

    # Stage 3C.6 final QA/readiness scores. These scores deliberately combine
    # existing terrain-stage diagnostics instead of changing the terrain itself.
    # They are meant to make the post-terrain review concrete before hydrology.
    flat_interior_share = 0.0
    drainage_corridor_share = erosion_stats.get("valley_corridor_moderate_share", 0.0)
    try:
        from scipy import ndimage
        coast_distance = ndimage.distance_transform_edt(~coast)
        interior_threshold = max(8, min(64, land.shape[1] // 48))
        slope_threshold = float(np.percentile(land_slope, 32)) if land_slope.size else 1e9
        flat_interior_mask = land & (coast_distance > interior_threshold) & (slope < slope_threshold)
        flat_interior_share = round(float(np.mean(flat_interior_mask[land])) if land.any() else 0.0, 4)
    except Exception:
        flat_interior_mask = land & lowland_mask
        flat_interior_share = round(float(np.mean(flat_interior_mask[land])) if land.any() else 0.0, 4)

    ocean_fit_score = _clamp(1.0 - abs(ocean_target_error) / 0.08, 0.0, 1.0)
    landmass_balance_score = _clamp((0.96 - float(largest_share or 0.0)) / 0.55, 0.0, 1.0)
    # Supercontinents are allowed; avoid over-penalizing them when the derived profile supports one.
    if float(controls.get("effective_supercontinent_score", controls.get("derived_supercontinent_score", 0.0)) or 0.0) > 0.70:
        landmass_balance_score = max(landmass_balance_score, 0.55)
    province_diversity_score = float(province_stats.get("province_type_diversity_score", province_stats.get("province_size_diversity_score", 0.45)) or 0.0)
    boundary_quality_score = _clamp(1.0 - float(boundary_stats.get("boundary_neatness_proxy", 0.0) or 0.0), 0.0, 1.0)
    interior_relief_score = _clamp(float(relief_stats.get("interior_relief_field_mean", 0.0) or 0.0) * 2.2 + (1.0 - flat_interior_share) * 0.35, 0.0, 1.0)
    coast_quality_score = _clamp(0.38 * float(coast_stats.get("coast_style_diversity_score", 0.0) or 0.0) + 0.35 * min(1.0, coastline_complexity / 1.6) + 0.27 * float(coast_stats.get("mean_coast_ruggedness_proxy", 0.0) or 0.0), 0.0, 1.0)
    valley_score = _clamp(float(erosion_stats.get("valley_corridor_mean", 0.0) or 0.0) * 1.8 + drainage_corridor_share * 0.7, 0.0, 1.0)
    basin_deposition_score = _clamp(float(erosion_stats.get("deposition_field_mean", 0.0) or 0.0) * 2.0 + float(relief_stats.get("basin_field_mean", 0.0) or 0.0), 0.0, 1.0)
    erosion_effect_score = _clamp(float(erosion_stats.get("mean_abs_relief_delta_m", 0.0) or 0.0) / 180.0, 0.0, 1.0)
    if float(erosion_stats.get("mean_abs_relief_delta_m", 0.0) or 0.0) > 520:
        erosion_effect_score *= 0.72
    relief_variability_score = 0.0
    if land.any():
        relief_span = max(1.0, float(np.percentile(elev[land], 95) - np.percentile(elev[land], 5)))
        relief_variability_score = _clamp(relief_span / 4200.0, 0.0, 1.0)
    terrain_diversity_score = _clamp(0.24 * province_diversity_score + 0.22 * relief_variability_score + 0.20 * coast_quality_score + 0.17 * float(coast_stats.get("island_origin_diversity_score", 0.0) or 0.0) + 0.17 * interior_relief_score, 0.0, 1.0)
    drainage_readiness_score = _clamp(0.34 * valley_score + 0.22 * basin_deposition_score + 0.18 * coast_quality_score + 0.16 * ocean_fit_score + 0.10 * (1.0 - flat_interior_share), 0.0, 1.0)
    plate_integration_score = _clamp(float(plate_final_stats.get("native_plate_backend_integration_mean", 0.0) or 0.0), 0.0, 1.0)
    plate_legacy_dependency_score = _clamp(1.0 - float(plate_final_stats.get("native_plate_legacy_dependency_mean", 0.0) or 0.0), 0.0, 1.0)
    plate_native_hydrology_score = _clamp(float(plate_final_stats.get("native_plate_hydrology_readiness_mean", 0.0) or 0.0) * 1.8, 0.0, 1.0)
    if str(controls.get("terrain_generation_mode")) == "plate_tectonic_v1":
        overall_terrain_quality_score = _clamp(0.13 * ocean_fit_score + 0.10 * landmass_balance_score + 0.11 * boundary_quality_score + 0.14 * interior_relief_score + 0.13 * coast_quality_score + 0.12 * drainage_readiness_score + 0.10 * terrain_diversity_score + 0.10 * plate_integration_score + 0.07 * plate_legacy_dependency_score, 0.0, 1.0)
    else:
        overall_terrain_quality_score = _clamp(0.16 * ocean_fit_score + 0.14 * landmass_balance_score + 0.14 * boundary_quality_score + 0.16 * interior_relief_score + 0.15 * coast_quality_score + 0.13 * drainage_readiness_score + 0.12 * terrain_diversity_score, 0.0, 1.0)

    readiness_label = "ready"
    if overall_terrain_quality_score < 0.45 or drainage_readiness_score < 0.38 or abs(ocean_target_error) > 0.08:
        readiness_label = "not ready"
    elif overall_terrain_quality_score < 0.62 or drainage_readiness_score < 0.52 or warnings:
        readiness_label = "mostly ready with concerns"

    final_quality = {
        "overall_terrain_quality_score": round(overall_terrain_quality_score, 3),
        "hydrology_readiness_score": round(drainage_readiness_score, 3),
        "hydrology_readiness_label": readiness_label,
        "ocean_target_fit_score": round(ocean_fit_score, 3),
        "landmass_balance_score": round(landmass_balance_score, 3),
        "boundary_quality_score": round(boundary_quality_score, 3),
        "interior_relief_score": round(interior_relief_score, 3),
        "coast_quality_score": round(coast_quality_score, 3),
        "valley_corridor_readiness_score": round(valley_score, 3),
        "basin_deposition_score": round(basin_deposition_score, 3),
        "erosion_effect_score": round(erosion_effect_score, 3),
        "terrain_diversity_score": round(terrain_diversity_score, 3),
        "plate_backend_integration_score": round(plate_integration_score, 3),
        "plate_legacy_independence_score": round(plate_legacy_dependency_score, 3),
        "plate_native_hydrology_score": round(plate_native_hydrology_score, 3),
        "flat_interior_share_of_land": flat_interior_share,
        "drainage_corridor_moderate_share": round(float(drainage_corridor_share or 0.0), 4),
    }
    readiness_checks = [
        {"label": "Ocean target fit", "status": "pass" if ocean_fit_score >= 0.70 else ("review" if ocean_fit_score >= 0.45 else "fail"), "score": final_quality["ocean_target_fit_score"], "detail": f"Final ocean fraction {terrain.ocean_fraction:.3f}; target {target_ocean:.3f}."},
        {"label": "Landmass balance", "status": "pass" if landmass_balance_score >= 0.62 else ("review" if landmass_balance_score >= 0.42 else "fail"), "score": final_quality["landmass_balance_score"], "detail": f"Largest landmass share {largest_share}; landmass count {comp.get('landmass_count')}."},
        {"label": "Geologic skeleton", "status": "pass" if boundary_quality_score >= 0.55 and province_diversity_score >= 0.40 else "review", "score": round((boundary_quality_score + province_diversity_score) / 2.0, 3), "detail": "Province/boundary variety and neatness check."},
        {"label": "Interior relief", "status": "pass" if interior_relief_score >= 0.55 else ("review" if interior_relief_score >= 0.36 else "fail"), "score": final_quality["interior_relief_score"], "detail": f"Flat interior share of land {flat_interior_share:.3f}."},
        {"label": "Coasts and islands", "status": "pass" if coast_quality_score >= 0.50 else ("review" if coast_quality_score >= 0.32 else "fail"), "score": final_quality["coast_quality_score"], "detail": "Coast diversity, ruggedness, and complexity check."},
        {"label": "Drainage readiness", "status": "pass" if drainage_readiness_score >= 0.55 else ("review" if drainage_readiness_score >= 0.38 else "fail"), "score": final_quality["hydrology_readiness_score"], "detail": "Valley, basin, deposition, and coast/outlet readiness before hydrology."},
    ]
    if str(controls.get("terrain_generation_mode")) == "plate_tectonic_v1":
        readiness_checks.extend([
            {"label": "Plate backend integration", "status": "pass" if plate_integration_score >= 0.48 else ("review" if plate_integration_score >= 0.30 else "fail"), "score": final_quality["plate_backend_integration_score"], "detail": "How much native plate setup/motion/ocean-floor/relief/coast logic supports the final terrain."},
            {"label": "Legacy dependency", "status": "pass" if plate_legacy_dependency_score >= 0.45 else ("review" if plate_legacy_dependency_score >= 0.28 else "fail"), "score": final_quality["plate_legacy_independence_score"], "detail": "Low score means plate mode still has weak native support in that area even after plate-owned foundation replacement."},
            {"label": "Native plate hydrology readiness", "status": "pass" if plate_native_hydrology_score >= 0.45 else ("review" if plate_native_hydrology_score >= 0.25 else "fail"), "score": final_quality["plate_native_hydrology_score"], "detail": "How strongly native plate fields identify rifts, basins, outlets, coastal plains, and drainage corridors."},
        ])

    if flat_interior_share > 0.42 and terrain.land_fraction > 0.18:
        warnings.append({"level": "strong", "message": f"Flat interior risk is high: {flat_interior_share:.1%} of land is distant from coasts and low-slope. Review before hydrology."})
    elif flat_interior_share > 0.28 and terrain.land_fraction > 0.18:
        warnings.append({"level": "info", "message": f"Flat interior risk is moderate: {flat_interior_share:.1%} of land is distant from coasts and low-slope."})
    if drainage_readiness_score < 0.38 and terrain.land_fraction > 0.12:
        warnings.append({"level": "strong", "message": "Hydrology readiness is low; rivers may remain too straight or fail to find believable drainage corridors."})
    elif drainage_readiness_score < 0.52 and terrain.land_fraction > 0.12:
        warnings.append({"level": "info", "message": "Hydrology readiness is only moderate; inspect valley corridors and basin/deposition diagnostics."})
    if coast_quality_score < 0.30 and coastline_cells > 0:
        warnings.append({"level": "info", "message": "Final coast quality score is low; coastlines may still be too smooth/simple for this terrain style."})
    if terrain_diversity_score < 0.35 and terrain.land_fraction > 0.10:
        warnings.append({"level": "info", "message": "Final terrain diversity score is low; large regions may feel repetitive."})

    subphases = {
        "terrain-foundation-mask": {
            "folder": "01_foundation",
            "summary": "Broad land/ocean foundation and landmass distribution.",
            "metrics": {
                "ocean_fraction": round(float(terrain.ocean_fraction), 4),
                "land_fraction": round(float(terrain.land_fraction), 4),
                "target_ocean_fraction": round(float(hydrosphere.ocean_fraction_target), 4),
                "target_land_fraction": round(1.0 - float(hydrosphere.ocean_fraction_target), 4),
                "ocean_target_error": round(ocean_target_error, 4),
                "ocean_target_fit_applied": bool(ocean_fit.get("applied", False)),
                "sea_level_shift_m": ocean_fit.get("sea_level_shift_m"),
                **comp,
                "north_south_land_imbalance": round(abs(north_land - south_land), 4),
                "east_west_land_imbalance": round(abs(west_land - east_land), 4),
                "tropical_land_share": tropical_land_share,
                "temperate_land_share": temperate_land_share,
                "polar_land_share": polar_land_share,
                "interior_land_share": interior_land_share,
                "avg_distance_from_coast_cells": avg_distance_from_coast_cells,
                "derived_supercontinent_score": controls.get("derived_supercontinent_score"),
                "effective_supercontinent_score": controls.get("effective_supercontinent_score"),
                "fragmentation_tendency": controls.get("fragmentation_tendency"),
            },
        },
        "terrain-tectonic-provinces": {
            "folder": "02_provinces",
            "summary": "Procedural province/plate-style diagnostic fields with geological type and age proxies.",
            "metrics": {
                "target_plate_count": controls.get("target_plate_count"),
                "large_landmass_dominance": comp.get("largest_landmass_share_of_land"),
                **province_stats,
            },
        },
        "terrain-crust-and-boundaries": {
            "folder": "03_boundaries",
            "summary": "Crust classes, active/passive margins, and boundary activity diagnostics.",
            "metrics": {**boundary_stats, **crust_stats, **ocean_floor_stats},
        },
        "terrain-mountains-basins-rifts": {
            "folder": "04_mountains_basins",
            "summary": "Mountain, basin, highland, and interior relief diagnostics.",
            "metrics": {
                "max_elevation_m": int(terrain.max_elevation_m),
                "mean_land_elevation_m": round(float(terrain.mean_land_elevation_m), 2),
                "mountain_highland_share_of_land": round(float(np.mean(mountain_mask[land])) if land.any() else 0.0, 4),
                "basin_lowland_share_of_land": round(float(np.mean(basin_mask[land])) if land.any() else 0.0, 4),
                "interior_lowland_share_of_land": round(float(np.mean(lowland_mask[land])) if land.any() else 0.0, 4),
                "mean_land_slope_proxy": round(float(np.mean(land_slope)) if land_slope.size else 0.0, 3),
                "mountain_straightness_proxy": _straightness_proxy(mountain_mask),
                "derived_mountain_belt_strength": controls.get("mountain_belt_strength"),
                "derived_rift_strength": controls.get("rift_strength"),
                "derived_interior_relief": controls.get("interior_relief"),
                **relief_stats,
            },
        },
        "terrain-coasts-shelves-islands": {
            "folder": "05_coasts_islands",
            "summary": "Coastline, shelf, island, and archipelago diagnostics.",
            "metrics": {
                "coastline_cell_count": coastline_cells,
                "coastline_complexity_proxy": round(coastline_complexity, 3),
                "shelf_area_share_of_ocean": round(float(np.mean(shelf_mask[ocean])) if ocean.any() else 0.0, 4),
                "island_count": comp.get("island_count"),
                "small_island_count": comp.get("small_island_count"),
                "derived_coastline_complexity": controls.get("coastline_complexity"),
                "derived_island_density": controls.get("island_density"),
                "derived_shelf_width_factor": controls.get("shelf_width_factor"),
                "derived_coastal_ruggedness": controls.get("coastal_ruggedness"),
                "derived_fjord_tendency": controls.get("fjord_tendency"),
                "derived_coastal_plain_bias": controls.get("coastal_plain_bias"),
                "derived_island_shape_irregularity": controls.get("island_shape_irregularity"),
                **coast_stats,
            },
        },
        "terrain-erosion-deposition": {
            "folder": "06_erosion_deposition",
            "summary": "Erosion, deposition, valley corridors, sediment supply, and terrain maturity diagnostics.",
            "metrics": {
                "mean_land_slope_proxy": round(float(np.mean(land_slope)) if land_slope.size else 0.0, 3),
                "median_land_slope_proxy": round(float(np.median(land_slope)) if land_slope.size else 0.0, 3),
                "lowland_share_of_land": round(float(np.mean(lowland_mask[land])) if land.any() else 0.0, 4),
                "basin_share_of_land": round(float(np.mean(basin_mask[land])) if land.any() else 0.0, 4),
                "derived_erosion_deposition_strength": controls.get("erosion_deposition_strength"),
                "erosion_deposition_multiplier": controls.get("erosion_deposition_multiplier"),
                "continental_shelf_strength": controls.get("continental_shelf_strength"),
                "derived_deposition_strength": controls.get("deposition_strength"),
                "derived_valley_carving_strength": controls.get("valley_carving_strength"),
                "derived_sediment_supply_strength": controls.get("sediment_supply_strength"),
                "derived_coastal_plain_strength": controls.get("coastal_plain_strength"),
                "derived_alluvial_fan_strength": controls.get("alluvial_fan_strength"),
                "derived_floodplain_strength": controls.get("floodplain_strength"),
                "derived_terrain_maturity": controls.get("terrain_maturity"),
                **erosion_stats,
            },
        },
        "terrain-finalization-recentering": {
            "folder": "07_final",
            "summary": "Final terrain statistics and quality warnings.",
            "metrics": {
                "final_ocean_fraction": round(float(terrain.ocean_fraction), 4),
                "final_land_fraction": round(float(terrain.land_fraction), 4),
                "target_ocean_fraction": round(float(hydrosphere.ocean_fraction_target), 4),
                "ocean_target_error": round(ocean_target_error, 4),
                "min_elevation_m": int(terrain.min_elevation_m),
                "max_elevation_m": int(terrain.max_elevation_m),
                "mean_land_elevation_m": round(float(terrain.mean_land_elevation_m), 2),
                "mean_ocean_depth_m": round(float(terrain.mean_ocean_depth_m), 2),
                "landmass_count": comp.get("landmass_count"),
                "largest_landmass_share_of_land": comp.get("largest_landmass_share_of_land"),
                "coastline_complexity_proxy": round(coastline_complexity, 3),
                "terrain_style": controls.get("terrain_style"),
                **final_quality,
                **plate_final_stats,
            },
        },
    }

    report_lines = [
        "Stage 3 Terrain Review Report",
        "=============================",
        "",
        f"Terrain style: {controls.get('terrain_style')}",
        f"Supercontinent mode: {controls.get('supercontinent_tendency')} (derived score {controls.get('derived_supercontinent_score')}, effective {controls.get('effective_supercontinent_score', controls.get('derived_supercontinent_score'))})",
        f"Fragmentation tendency: {controls.get('fragmentation_tendency')}",
        f"Ocean fraction: {terrain.ocean_fraction:.3f} (target {hydrosphere.ocean_fraction_target:.3f}, error {ocean_target_error:+.3f})",
        f"Ocean target fit: applied={bool(ocean_fit.get('applied', False))}, sea-level shift={ocean_fit.get('sea_level_shift_m', 0)} m",
        f"Landmasses: {comp.get('landmass_count')} | largest share of land: {comp.get('largest_landmass_share_of_land')}",
        f"Top landmass shares: {comp.get('top_10_landmass_share_of_land')}",
        f"Latitudinal land share: tropical {tropical_land_share}, temperate {temperate_land_share}, polar {polar_land_share}",
        f"Interior land share: {interior_land_share} | average distance from coast: {avg_distance_from_coast_cells} cells",
        f"Coastline complexity proxy: {coastline_complexity:.3f}",
        f"Coast style diversity: {coast_stats.get('coast_style_diversity_score')} | shelf proxy {coast_stats.get('mean_shelf_width_proxy')} | island-origin diversity {coast_stats.get('island_origin_diversity_score')}",
        f"Province count/diversity: {province_stats.get('diagnostic_plate_count')} provinces, size diversity {province_stats.get('province_size_diversity_score')}",
        f"Boundary activity: active {boundary_stats.get('active_margin_share')}, passive {boundary_stats.get('passive_margin_share')}, neatness {boundary_stats.get('boundary_neatness_proxy')}",
        f"Ocean floor: ridges {ocean_floor_stats.get('mid_ocean_ridge_share_of_ocean')}, trenches {ocean_floor_stats.get('trench_share_of_ocean')}, seamounts {ocean_floor_stats.get('seamount_hotspot_share_of_ocean')}",
        f"Island shape complexity: mean {coast_stats.get('mean_island_shape_complexity')}, low-complexity cells {coast_stats.get('low_complexity_island_cell_share')}",
        f"Erosion/deposition: valley {erosion_stats.get('valley_corridor_mean')}, deposition {erosion_stats.get('deposition_field_mean')}, maturity {erosion_stats.get('terrain_maturity_mean')}, mean abs relief delta {erosion_stats.get('mean_abs_relief_delta_m')} m",
        f"Final terrain quality: {final_quality.get('overall_terrain_quality_score')} | hydrology readiness: {final_quality.get('hydrology_readiness_label')} ({final_quality.get('hydrology_readiness_score')})",
        f"Final QA scores: ocean {final_quality.get('ocean_target_fit_score')}, landmass {final_quality.get('landmass_balance_score')}, interior relief {final_quality.get('interior_relief_score')}, coasts {final_quality.get('coast_quality_score')}, diversity {final_quality.get('terrain_diversity_score')}",
        f"Plate mode integration: backend {final_quality.get('plate_backend_integration_score')}, legacy independence {final_quality.get('plate_legacy_independence_score')}, native hydrology {final_quality.get('plate_native_hydrology_score')}",
        "",
        "Hydrology readiness checklist:",
        *[f"- {item.get('status','review').upper()}: {item.get('label')} — score {item.get('score')}; {item.get('detail')}" for item in readiness_checks],
        "",
        "Downstream implications:",
        "- Climate will inherit the final elevation field, land/ocean mask, coastline exposure, and mountain barriers.",
        "- Hydrology will inherit river gradients, basin shape, lowland corridors, and endorheic risks from this terrain.",
        "- Biomes and regions will reflect terrain-driven precipitation, elevation, river access, and coast proximity.",
    ]
    if warnings:
        report_lines.extend(["", "Warnings:"])
        for item in warnings:
            report_lines.append(f"- {item.get('level','warning')}: {item.get('message','')}")

    return {
        "schema_version": 1,
        "stage": "terrain-review",
        "controls": controls,
        "summary": {
            "width": int(terrain.width),
            "height": int(terrain.height),
            "ocean_fraction": round(float(terrain.ocean_fraction), 4),
            "land_fraction": round(float(terrain.land_fraction), 4),
            "target_ocean_fraction": round(target_ocean, 4),
            "ocean_target_error": round(ocean_target_error, 4),
            "landmass_count": comp.get("landmass_count"),
            "largest_landmass_share_of_land": comp.get("largest_landmass_share_of_land"),
            "coastline_complexity_proxy": round(coastline_complexity, 3),
            "island_count": comp.get("island_count"),
            "small_island_count": comp.get("small_island_count"),
            "mean_land_slope_proxy": round(float(np.mean(land_slope)) if land_slope.size else 0.0, 3),
            "overall_terrain_quality_score": final_quality.get("overall_terrain_quality_score"),
            "hydrology_readiness_score": final_quality.get("hydrology_readiness_score"),
            "hydrology_readiness_label": final_quality.get("hydrology_readiness_label"),
            "flat_interior_share_of_land": final_quality.get("flat_interior_share_of_land"),
            "plate_backend_integration_score": final_quality.get("plate_backend_integration_score"),
            "plate_legacy_independence_score": final_quality.get("plate_legacy_independence_score"),
            "plate_native_hydrology_score": final_quality.get("plate_native_hydrology_score"),
        },
        "subphases": subphases,
        "final_quality": final_quality,
        "hydrology_readiness_checks": readiness_checks,
        "warnings": warnings,
        "ocean_target_fit": ocean_fit,
        "generation_metadata": generation_meta,
        "terrain_mode": generation_meta.get("terrain_mode"),
        "plate_tectonic_v1": generation_meta.get("plate_tectonic_v1"),
        "mode_transition_warnings": generation_meta.get("mode_transition_warnings", []),
        "report_lines": report_lines,
    }


def _downsample(arr, max_width: int = 2048):
    h, w = arr.shape[:2]
    stride = max(1, int(math.ceil(w / max_width)))
    return arr[::stride, ::stride], stride


def _save_rgb(path: Path, rgb, *, title: str, description: str, items: list[tuple[str, tuple[int, int, int]]], stats: dict[str, Any], terrain: TerrainMap, stride: int) -> None:
    from PIL import Image
    try:
        from worldgen.visualization.system_plot import _write_map_legend_sidecar
    except Exception:
        _write_map_legend_sidecar = None
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb.astype("uint8"), mode="RGB").save(path)
    if _write_map_legend_sidecar is not None:
        scale = {
            "kind": "terrain_diagnostic_downsample",
            "projection": "equirectangular",
            "source_stride": stride,
            "planet_radius_earth": terrain.planet_radius_earth,
            "data_width": int(rgb.shape[1]),
            "data_height": int(rgb.shape[0]),
        }
        _write_map_legend_sidecar(path, title=title, description=description, items=items, stats=stats, scale=scale)


def write_terrain_review_outputs(output_dir: str | Path, terrain: TerrainMap) -> None:
    """Write Stage 3 diagnostic folders, report files, and map-only PNGs."""
    import numpy as np
    from scipy import ndimage

    output_dir = Path(output_dir)
    review = getattr(terrain, "terrain_diagnostics", None) or {}
    if not isinstance(review, dict) or not review:
        return
    root = output_dir / "terrain_diagnostics"
    root.mkdir(parents=True, exist_ok=True)
    (root / "terrain_review.json").write_text(json.dumps(review, indent=2), encoding="utf-8")
    (root / "terrain_review_report.txt").write_text("\n".join(review.get("report_lines", [])) + "\n", encoding="utf-8")

    rows = [("subphase", "metric", "value")]
    for stage, info in (review.get("subphases") or {}).items():
        metrics = info.get("metrics", {}) if isinstance(info, dict) else {}
        for key, value in metrics.items():
            rows.append((stage, key, value))
    with (root / "terrain_subphase_metrics.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerows(rows)

    elev, land = _terrain_arrays(terrain)
    elev_ds, stride = _downsample(elev)
    land_ds = land[::stride, ::stride]
    coast_ds = _coast_mask(land_ds)
    ocean_ds = ~land_ds
    h, w = land_ds.shape

    labels, comp = _component_stats(land_ds)
    # Foundation mask + landmass components
    rgb = np.zeros((h, w, 3), dtype=np.uint8); rgb[:] = (35, 82, 135); rgb[land_ds] = (136, 154, 102); rgb[coast_ds] = (20, 20, 18)
    _save_rgb(root / "01_foundation" / "foundation_land_ocean_mask.png", rgb, title="Terrain foundation land/ocean mask", description="Final structural land/ocean mask used as a proxy for the foundation checkpoint until internal passes are split.", items=[("ocean", (35,82,135)), ("land", (136,154,102)), ("coastline", (20,20,18))], stats=review.get("subphases", {}).get("terrain-foundation-mask", {}).get("metrics", {}), terrain=terrain, stride=stride)
    if labels is not None and np.max(labels) > 0:
        rng = np.random.default_rng(4217); colors = rng.integers(45, 235, size=(int(labels.max()) + 1, 3), dtype=np.uint8); colors[0] = (35,82,135); rgb2 = colors[labels]
        _save_rgb(root / "01_foundation" / "landmass_components.png", rgb2, title="Landmass components", description="Connected land components. Large single-color dominance indicates supercontinent-style outcomes.", items=[("component colors", (140,180,120)), ("ocean", (35,82,135))], stats=comp, terrain=terrain, stride=stride)

    try:
        dist = ndimage.distance_transform_edt(~coast_ds).astype(np.float32)
        if float(dist.max()) > 0:
            norm = np.clip(dist / max(1.0, float(np.percentile(dist, 95))), 0, 1)
        else:
            norm = dist
        rgbd = np.zeros((h, w, 3), dtype=np.uint8); rgbd[:] = (26, 70, 120)
        rgbd[land_ds] = np.stack([
            (110 + 110 * norm[land_ds]).astype(np.uint8),
            (135 + 65 * norm[land_ds]).astype(np.uint8),
            (95 + 55 * norm[land_ds]).astype(np.uint8),
        ], axis=1)
        rgbd[coast_ds] = (245, 225, 100)
        _save_rgb(root / "01_foundation" / "foundation_distance_from_coast.png", rgbd, title="Foundation distance from coast", description="Coastal/interior exposure diagnostic. Large interior zones imply stronger continental climate and hydrology sensitivity later.", items=[("ocean", (26,70,120)), ("coast", (245,225,100)), ("interior land gradient", (205,190,130))], stats=review.get("subphases", {}).get("terrain-foundation-mask", {}).get("metrics", {}), terrain=terrain, stride=stride)

        core = land_ds & (dist > max(5, w // 80))
        lowland = land_ds & (elev_ds < 250)
        near_sea = np.abs(elev_ds) < 180
        rgbcore = np.zeros((h, w, 3), dtype=np.uint8); rgbcore[:] = (35,82,135); rgbcore[land_ds]=(135,150,105); rgbcore[core]=(205,180,110); rgbcore[lowland]=(120,180,110); rgbcore[near_sea & land_ds]=(230,215,120); rgbcore[coast_ds]=(30,30,22)
        _save_rgb(root / "01_foundation" / "foundation_continental_core_proxy.png", rgbcore, title="Continental core and lowland proxy", description="Proxy for exposed continental interiors, lowlands, and near-sea-level land most affected by ocean-target changes.", items=[("ocean", (35,82,135)), ("ordinary land", (135,150,105)), ("continental core", (205,180,110)), ("lowland", (120,180,110)), ("near sea level land", (230,215,120))], stats=review.get("subphases", {}).get("terrain-foundation-mask", {}).get("metrics", {}), terrain=terrain, stride=stride)

        ocean_fit = review.get("ocean_target_fit", {}) if isinstance(review.get("ocean_target_fit"), dict) else {}
        rgbfit = np.zeros((h, w, 3), dtype=np.uint8); rgbfit[:] = (35,82,135); rgbfit[land_ds]=(140,155,105); rgbfit[ocean_ds & (elev_ds > -350)] = (75,150,185); rgbfit[land_ds & near_sea]=(235,210,95); rgbfit[coast_ds]=(20,20,18)
        _save_rgb(root / "01_foundation" / "foundation_ocean_target_fit.png", rgbfit, title="Ocean target fit zones", description=f"Near-sea-level land and shallow shelves most likely to change when ocean target changes. Target fit: {ocean_fit}", items=[("deep ocean", (35,82,135)), ("shallow ocean/shelf", (75,150,185)), ("stable land", (140,155,105)), ("near sea level land", (235,210,95)), ("coastline", (20,20,18))], stats=review.get("subphases", {}).get("terrain-foundation-mask", {}).get("metrics", {}), terrain=terrain, stride=stride)
    except Exception:
        pass

    # Province map from diagnostic plate field, if present.
    province_metrics = review.get("subphases", {}).get("terrain-tectonic-provinces", {}).get("metrics", {})
    boundary_metrics = review.get("subphases", {}).get("terrain-crust-and-boundaries", {}).get("metrics", {})
    if terrain.tectonic_plate_id is not None:
        plate = np.asarray(terrain.tectonic_plate_id, dtype=np.int32)
        maxp = int(max(0, plate.max()))
        rng = np.random.default_rng(6543); colors = rng.integers(45, 230, size=(maxp + 1, 3), dtype=np.uint8)
        rgbp = colors[np.clip(plate, 0, maxp)]
        _save_rgb(root / "02_provinces" / "tectonic_province_map.png", rgbp, title="Terrain tectonic/province map", description="Procedural province/plate regions currently generated at diagnostic resolution.", items=[("province colors", (120,180,220))], stats=province_metrics, terrain=terrain, stride=1)

        counts = np.bincount(plate.ravel())
        if counts.size:
            largest_id = int(np.argmax(counts))
            rgbdom = np.zeros((*plate.shape, 3), dtype=np.uint8); rgbdom[:] = (70, 90, 115); rgbdom[plate == largest_id] = (235, 185, 75)
            rgbdom[(plate != largest_id) & (counts[plate] > np.percentile(counts[counts > 0], 75))] = (150, 165, 130)
            _save_rgb(root / "02_provinces" / "province_size_dominance.png", rgbdom, title="Province size dominance", description="Highlights the largest and other large diagnostic provinces. High dominance can produce uniform continents or repeated boundary geometry.", items=[("largest province", (235,185,75)), ("other large provinces", (150,165,130)), ("smaller provinces", (70,90,115))], stats=province_metrics, terrain=terrain, stride=1)

    if getattr(terrain, "tectonic_province_type", None) is not None:
        ptype = np.asarray(terrain.tectonic_province_type, dtype=np.int32)
        rgbt = np.zeros((*ptype.shape, 3), dtype=np.uint8)
        for code, color in PROVINCE_TYPE_COLORS.items():
            rgbt[ptype == code] = color
        _save_rgb(root / "02_provinces" / "province_type_map.png", rgbt, title="Province type map", description="Geological meaning assigned to the diagnostic province skeleton: old oceanic basins, continental cores, rifted margins, arcs, microcontinents, basins, and shields.", items=[(PROVINCE_TYPE_LABELS[k], PROVINCE_TYPE_COLORS[k]) for k in sorted(PROVINCE_TYPE_LABELS)], stats=province_metrics, terrain=terrain, stride=1)

        continental = np.isin(ptype, [2,3,4,6,7,8])
        micro = ptype == 6
        rgbmask = np.zeros((*ptype.shape, 3), dtype=np.uint8); rgbmask[:] = (34,72,130); rgbmask[continental] = (154,150,95); rgbmask[np.isin(ptype, [5])] = (210,90,58); rgbmask[micro] = (178,105,200)
        _save_rgb(root / "02_provinces" / "continental_oceanic_province_mask.png", rgbmask, title="Continental/oceanic province mask", description="Broad crustal identity proxy. This helps diagnose whether continents, shelves, arcs, and ocean basins match the planet profile.", items=[("oceanic province", (34,72,130)), ("continental/margin province", (154,150,95)), ("volcanic arc", (210,90,58)), ("microcontinent/terrane", (178,105,200))], stats=province_metrics, terrain=terrain, stride=1)
        rgbmicro = np.zeros((*ptype.shape, 3), dtype=np.uint8); rgbmicro[:] = (45,75,115); rgbmicro[np.isin(ptype, [3,4])] = (170,150,95); rgbmicro[micro] = (235,145,65); rgbmicro[ptype == 5] = (205,70,55)
        _save_rgb(root / "02_provinces" / "microcontinents_and_terranes.png", rgbmicro, title="Microcontinents and accreted terranes", description="Highlights smaller continental fragments, rifted margins, and arc/terrane provinces that should later feed island chains and irregular margins.", items=[("background/oceanic", (45,75,115)), ("shelf/rifted margin", (170,150,95)), ("microcontinent/terrane", (235,145,65)), ("arc", (205,70,55))], stats=province_metrics, terrain=terrain, stride=1)

    if getattr(terrain, "tectonic_province_age_x1000", None) is not None:
        age = np.asarray(terrain.tectonic_province_age_x1000, dtype=np.float32) / 1000.0
        rgba = np.zeros((*age.shape, 3), dtype=np.uint8)
        rgba[...,0] = (55 + 175 * age).astype(np.uint8)
        rgba[...,1] = (100 + 95 * (1.0 - np.abs(age - 0.55))).astype(np.uint8)
        rgba[...,2] = (195 - 110 * age).astype(np.uint8)
        _save_rgb(root / "02_provinces" / "province_age_proxy.png", rgba, title="Province age/stability proxy", description="Young/active versus old/stable crust proxy. Downstream, young provinces should favor rifts/arcs; old provinces should favor shields, eroded ranges, and stable interiors.", items=[("young/active", (55,130,195)), ("intermediate", (155,190,135)), ("old/stable", (230,120,85))], stats=province_metrics, terrain=terrain, stride=1)

    if getattr(terrain, "plate_tectonic_plate_type", None) is not None:
        ptype_native = np.asarray(terrain.plate_tectonic_plate_type, dtype=np.int32)
        rgb_native = np.zeros((*ptype_native.shape, 3), dtype=np.uint8)
        for code, color in PLATE_TECTONIC_TYPE_COLORS.items():
            rgb_native[ptype_native == code] = color
        _save_rgb(root / "02_provinces" / "plate_tectonic_v1_plate_types.png", rgb_native, title="Plate tectonic v1 native plate types", description="Native Plate Terrain 11 plate-type allocation. Plate mode now owns the visible foundation/mask/base elevation before applying native plate relief and coasts.", items=[(PLATE_TECTONIC_TYPE_LABELS[k], PLATE_TECTONIC_TYPE_COLORS[k]) for k in sorted(PLATE_TECTONIC_TYPE_LABELS)], stats=province_metrics, terrain=terrain, stride=1)

    def _save_plate_field(attr_name, file_name, title, description, color_a=(42,82,150), color_b=(230,190,85)):
        value = getattr(terrain, attr_name, None)
        if value is None:
            return
        arr = np.asarray(value, dtype=np.float32) / 1000.0
        arr = np.clip(arr, 0.0, 1.0)
        rgb_field = np.zeros((*arr.shape, 3), dtype=np.uint8)
        rgb_field[...,0] = (color_a[0] + (color_b[0] - color_a[0]) * arr).astype(np.uint8)
        rgb_field[...,1] = (color_a[1] + (color_b[1] - color_a[1]) * arr).astype(np.uint8)
        rgb_field[...,2] = (color_a[2] + (color_b[2] - color_a[2]) * arr).astype(np.uint8)
        _save_rgb(root / "02_provinces" / file_name, rgb_field, title=title, description=description, items=[("low", color_a), ("high", color_b)], stats=province_metrics, terrain=terrain, stride=1)

    _save_plate_field("plate_tectonic_continental_crust_x1000", "plate_tectonic_v1_continental_crust.png", "Plate tectonic v1 continental crust allocation", "Native Plate Terrain 4 continental-crust fraction field used by native plate relief to place craton/shield, collision, rift, and terrane terrain.", (38,68,130), (220,190,110))
    _save_plate_field("plate_tectonic_craton_core_x1000", "plate_tectonic_v1_craton_cores.png", "Plate tectonic v1 craton cores", "Stable old continental core field derived from native plate interiors and land support.", (45,72,110), (235,205,130))
    _save_plate_field("plate_tectonic_microplate_x1000", "plate_tectonic_v1_microplates.png", "Plate tectonic v1 microplates and terranes", "Native microplate/terrane field. These should later drive island arcs, accreted terranes, and fragmented margins.", (44,70,105), (220,110,210))
    if getattr(terrain, "plate_tectonic_continent_assembly_id", None) is not None:
        assembly = np.asarray(terrain.plate_tectonic_continent_assembly_id, dtype=np.int32)
        if assembly.size:
            rgba = np.zeros((*assembly.shape, 3), dtype=np.uint8)
            rgba[assembly == 0] = (36, 76, 128)
            ids = [int(x) for x in np.unique(assembly) if int(x) > 0]
            for cid in ids:
                if cid >= 1000:
                    color = ((180 + (cid * 17) % 70) % 256, 120 + (cid * 31) % 80, 68 + (cid * 19) % 120)
                else:
                    color = (92 + (cid * 43) % 120, 118 + (cid * 29) % 100, 78 + (cid * 61) % 120)
                rgba[assembly == cid] = color
            _save_rgb(root / "02_provinces" / "plate_tectonic_v1_continent_assemblies.png", rgba, title="Plate tectonic v1 continent assemblies", description="Plate Terrain 11 continent/domain assembly IDs. Values below 1000 are coherent continental assemblies; 1000+ values are island/terrane belts that should not carry broad continental shelves.", items=[("ocean/background", (36,76,128)), ("continent assemblies", (120,160,100)), ("island/terrane belts", (220,150,95))], stats=province_metrics, terrain=terrain, stride=1)

    if getattr(terrain, "plate_tectonic_plate_topology_problem_class", None) is not None:
        topo = np.asarray(terrain.plate_tectonic_plate_topology_problem_class, dtype=np.uint8)
        if topo.size:
            rgbt = np.zeros((*topo.shape, 3), dtype=np.uint8); rgbt[:] = (42, 72, 106)
            rgbt[topo == 1] = (235, 165, 72)
            rgbt[topo == 2] = (210, 98, 150)
            _save_rgb(root / "02_provinces" / "plate_tectonic_v1_topology_repairs.png", rgbt, title="Plate tectonic v1 topology repairs", description="Plate Terrain 11 topology diagnostic. Highlights cells affected by graph-aware plate repair such as single-neighbor/bullseye plates or unsupported island/terrane fragments.", items=[("normal plate domain", (42,72,106)), ("single-neighbor repair", (235,165,72)), ("unsupported island/terrane", (210,98,150))], stats=province_metrics, terrain=terrain, stride=1)

    def _save_motion_field(attr_name, file_name, title, description, low_color, high_color):
        value = getattr(terrain, attr_name, None)
        if value is None:
            return
        arr = np.asarray(value, dtype=np.float32) / 1000.0
        if arr.size == 0:
            return
        arr = np.clip(arr, 0.0, 1.0)
        rgb_field = np.zeros((*arr.shape, 3), dtype=np.uint8)
        for channel in range(3):
            rgb_field[..., channel] = (low_color[channel] + (high_color[channel] - low_color[channel]) * arr).astype(np.uint8)
        _save_rgb(root / "03_boundaries" / file_name, rgb_field, title=title, description=description, items=[("weak/none", low_color), ("strong", high_color)], stats=boundary_metrics, terrain=terrain, stride=1)

    if getattr(terrain, "plate_tectonic_velocity_x_x1000", None) is not None and getattr(terrain, "plate_tectonic_velocity_y_x1000", None) is not None:
        vx = np.asarray(terrain.plate_tectonic_velocity_x_x1000, dtype=np.float32) / 1000.0
        vy = np.asarray(terrain.plate_tectonic_velocity_y_x1000, dtype=np.float32) / 1000.0
        sp = np.clip(np.sqrt(vx * vx + vy * vy), 0.0, 1.0)
        angle = np.arctan2(vy, vx)
        rgbv = np.zeros((*sp.shape, 3), dtype=np.uint8)
        rgbv[..., 0] = ((0.5 + 0.5 * np.cos(angle)) * 70 + 185 * sp).clip(0,255).astype(np.uint8)
        rgbv[..., 1] = ((0.5 + 0.5 * np.sin(angle)) * 85 + 150 * sp).clip(0,255).astype(np.uint8)
        rgbv[..., 2] = ((1.0 - sp) * 125 + 75 * (0.5 + 0.5 * np.cos(angle - 2.1))).clip(0,255).astype(np.uint8)
        _save_rgb(root / "03_boundaries" / "plate_tectonic_v1_motion_vectors.png", rgbv, title="Plate tectonic v1 motion vectors", description="Native Plate Terrain 4 plate-motion direction and relative speed field. Hue indicates direction; brightness indicates speed. This is the first Euler-like motion layer, not yet a time reconstruction.", items=[("direction hue", (185,135,95)), ("faster/brighter", (235,210,110)), ("slower/darker", (45,70,125))], stats=boundary_metrics, terrain=terrain, stride=1)

    _save_motion_field("plate_tectonic_speed_x1000", "plate_tectonic_v1_plate_speed.png", "Plate tectonic v1 plate speed", "Native plate speed diagnostic. Oceanic and microplates tend to move faster; old/stable continental plates tend to move more slowly.", (42,70,115), (235,200,88))
    _save_motion_field("plate_tectonic_convergence_x1000", "plate_tectonic_v1_convergence_rate.png", "Plate tectonic v1 convergence field", "Relative-motion convergence field derived from adjacent plate velocities. Strong zones should later drive collision belts, subduction, arcs, and trenches.", (58,72,95), (225,76,55))
    _save_motion_field("plate_tectonic_divergence_x1000", "plate_tectonic_v1_divergence_rate.png", "Plate tectonic v1 divergence field", "Relative-motion divergence/spreading field derived from adjacent plate velocities. Strong zones should later drive rifts and mid-ocean ridges.", (46,70,105), (75,155,235))
    _save_motion_field("plate_tectonic_transform_x1000", "plate_tectonic_v1_transform_rate.png", "Plate tectonic v1 transform/shear field", "Relative-motion tangential/shear field derived from adjacent plate velocities. Strong zones should later drive offset ranges, fracture zones, and shear valleys.", (54,64,96), (178,96,220))

    if getattr(terrain, "plate_tectonic_subduction_polarity", None) is not None:
        pol = np.asarray(terrain.plate_tectonic_subduction_polarity, dtype=np.uint8)
        rgbp = np.zeros((*pol.shape, 3), dtype=np.uint8)
        for code, color in PLATE_SUBDUCTION_POLARITY_COLORS.items():
            rgbp[pol == code] = color
        _save_rgb(root / "03_boundaries" / "plate_tectonic_v1_subduction_polarity.png", rgbp, title="Plate tectonic v1 subduction/collision polarity", description="Native Plate Terrain 4 convergence subtype diagnostic. It distinguishes oceanic-under-continental/mixed margins, ocean-ocean subduction, and continental collision zones.", items=[(PLATE_SUBDUCTION_POLARITY_LABELS[k], PLATE_SUBDUCTION_POLARITY_COLORS[k]) for k in sorted(PLATE_SUBDUCTION_POLARITY_LABELS)], stats=boundary_metrics, terrain=terrain, stride=1)

    if terrain.tectonic_boundary_class is not None:
        b = np.asarray(terrain.tectonic_boundary_class, dtype=np.uint8)
        rgbb = np.zeros((b.shape[0], b.shape[1], 3), dtype=np.uint8)
        for code, color in BOUNDARY_CLASS_COLORS.items():
            rgbb[b == code] = color
        _save_rgb(root / "03_boundaries" / "boundary_class_map.png", rgbb, title="Terrain crust/boundary classes", description="Procedural boundary classes: convergent/collision, rift, transform, passive margin, diffuse suture, volcanic arc, or inactive interior.", items=[(BOUNDARY_CLASS_LABELS[k], BOUNDARY_CLASS_COLORS[k]) for k in sorted(BOUNDARY_CLASS_LABELS)], stats=boundary_metrics, terrain=terrain, stride=1)

        rgbap = np.zeros((b.shape[0], b.shape[1], 3), dtype=np.uint8); rgbap[:] = (38, 65, 100)
        rgbap[b == 4] = (85, 165, 145)
        rgbap[np.isin(b, [1,2,3,5,6])] = (210, 100, 70)
        rgbap[b == 2] = (70, 125, 210)
        rgbap[b == 6] = (230, 72, 45)
        _save_rgb(root / "03_boundaries" / "active_passive_margins.png", rgbap, title="Active and passive margins", description="Separates active boundaries from passive margins so the future mountain/coast passes can avoid treating every coastline as tectonically active.", items=[("intraplate/interior", (38,65,100)), ("passive margin", (85,165,145)), ("active boundary", (210,100,70)), ("rift", (70,125,210)), ("volcanic arc", (230,72,45))], stats=boundary_metrics, terrain=terrain, stride=1)

        rgbrift = np.zeros((b.shape[0], b.shape[1], 3), dtype=np.uint8); rgbrift[:] = (35, 75, 115); rgbrift[b == 2] = (60, 140, 235); rgbrift[b == 4] = (80, 160, 150); rgbrift[b == 5] = (185, 155, 80)
        _save_rgb(root / "03_boundaries" / "rift_zone_tendency.png", rgbrift, title="Rift zone tendency", description="Divergent/rift and related margin classes. High fragmentation worlds should usually show enough rift structure to explain broken continents.", items=[("background", (35,75,115)), ("rift/divergent", (60,140,235)), ("passive margin", (80,160,150)), ("old suture/diffuse", (185,155,80))], stats=boundary_metrics, terrain=terrain, stride=1)

        rgbconv = np.zeros((b.shape[0], b.shape[1], 3), dtype=np.uint8); rgbconv[:] = (48, 70, 92); rgbconv[b == 1] = (190, 60, 48); rgbconv[b == 6] = (240, 110, 55); rgbconv[b == 3] = (150, 90, 185)
        _save_rgb(root / "03_boundaries" / "subduction_collision_tendency.png", rgbconv, title="Subduction/collision tendency", description="Convergent and volcanic-arc boundary proxies that should later drive mountain belts, island arcs, trenches, and active margins.", items=[("background", (48,70,92)), ("collision/convergent", (190,60,48)), ("volcanic arc", (240,110,55)), ("transform/shear", (150,90,185))], stats=boundary_metrics, terrain=terrain, stride=1)

    if getattr(terrain, "tectonic_boundary_strength_x1000", None) is not None:
        bs = np.asarray(terrain.tectonic_boundary_strength_x1000, dtype=np.float32) / 1000.0
        rgbs = np.zeros((*bs.shape, 3), dtype=np.uint8)
        rgbs[...,0] = (45 + 205 * bs).astype(np.uint8)
        rgbs[...,1] = (70 + 115 * (1.0 - np.abs(bs - 0.55))).astype(np.uint8)
        rgbs[...,2] = (115 + 95 * (1.0 - bs)).astype(np.uint8)
        _save_rgb(root / "03_boundaries" / "boundary_neatness_strength.png", rgbs, title="Boundary strength field", description="Broad boundary strength diagnostic. Stronger zones should later have more mountain/rift/coast influence; weak/diffuse zones should be less visually dominant.", items=[("weak", (45,95,210)), ("moderate", (160,180,165)), ("strong", (250,105,115))], stats=boundary_metrics, terrain=terrain, stride=1)

    if getattr(terrain, "tectonic_boundary_width_x1000", None) is not None:
        bw = np.asarray(terrain.tectonic_boundary_width_x1000, dtype=np.float32) / 1000.0
        rgbw = np.zeros((*bw.shape, 3), dtype=np.uint8)
        rgbw[...,0] = (40 + 180 * bw).astype(np.uint8)
        rgbw[...,1] = (75 + 140 * bw).astype(np.uint8)
        rgbw[...,2] = (120 + 55 * (1.0 - bw)).astype(np.uint8)
        _save_rgb(root / "03_boundaries" / "boundary_width_field.png", rgbw, title="Boundary width field", description="Diffuse versus narrow boundary zones. Wider boundaries should produce broader orogenic/rift provinces rather than thin artificial lines.", items=[("narrow", (40,75,175)), ("moderate", (125,145,145)), ("wide/diffuse", (220,215,120))], stats=boundary_metrics, terrain=terrain, stride=1)

    if terrain.crust_type is not None:
        c = np.asarray(terrain.crust_type, dtype=np.uint8)
        crust_palette = {
            0: (50, 88, 132), 1: (44, 98, 156), 2: (64, 184, 230), 3: (34, 72, 132),
            4: (150, 86, 190), 5: (128, 126, 220), 6: (72, 202, 188), 7: (122, 228, 220),
            8: (122, 108, 72), 9: (168, 148, 86), 10: (246, 224, 156), 11: (166, 92, 86),
            12: (230, 142, 64), 13: (196, 188, 104), 14: (216, 186, 118), 15: (190, 112, 164),
            16: (226, 72, 56), 17: (250, 120, 58), 18: (250, 198, 76), 19: (84, 156, 214),
            20: (56, 120, 184), 21: (38, 86, 146),
        }
        rgbc = np.zeros((c.shape[0], c.shape[1], 3), dtype=np.uint8)
        rgbc[:] = crust_palette[0]
        for code, color in crust_palette.items():
            rgbc[c == code] = color
        _save_rgb(
            root / "03_boundaries" / "crust_type_map.png",
            rgbc,
            title="Terrain crust type classes",
            description="Readable broad crust classes derived from final elevation and land/ocean context. Update 35 colors all known v3/v4 crust classes instead of leaving most cells black.",
            items=[
                ("generic/deep ocean", crust_palette[1]), ("ridge/young ocean", crust_palette[2]), ("trench/fracture/seamount", crust_palette[4]),
                ("shelf / slope / rise", crust_palette[7]), ("continental interior", crust_palette[9]), ("orogenic / suture / rift", crust_palette[10]),
                ("volcanic arc / hotspot island", crust_palette[17]),
            ],
            stats=review.get("subphases", {}).get("terrain-crust-and-boundaries", {}).get("metrics", {}),
            terrain=terrain,
            stride=1,
        )

    boundary_metrics = review.get("subphases", {}).get("terrain-crust-and-boundaries", {}).get("metrics", {})
    if getattr(terrain, "terrain_ocean_floor_class", None) is not None:
        of = np.asarray(terrain.terrain_ocean_floor_class, dtype=np.uint8)
        rgbo = np.zeros((*of.shape, 3), dtype=np.uint8)
        for code, color in OCEAN_FLOOR_COLORS.items():
            rgbo[of == code] = color
        _save_rgb(root / "03_boundaries" / "ocean_floor_classes.png", rgbo, title="Ocean-floor classes", description="Terrain Revisit 2 ocean bathymetry classes: abyssal plains, mid-ocean ridges, trenches/subduction troughs, fracture zones, and seamount/hotspot provinces.", items=[(OCEAN_FLOOR_LABELS[k], OCEAN_FLOOR_COLORS[k]) for k in sorted(OCEAN_FLOOR_LABELS)], stats=boundary_metrics, terrain=terrain, stride=1)

    def _save_ocean_floor_field(attr_name: str, filename: str, title: str, description: str, high_color: tuple[int,int,int]):
        value = getattr(terrain, attr_name, None)
        if value is None:
            return
        arr = np.asarray(value, dtype=np.float32) / 1000.0
        if arr.size == 0:
            return
        arr = np.clip(arr, 0.0, 1.0)
        rgbx = np.zeros((*arr.shape, 3), dtype=np.uint8); rgbx[:] = (24, 56, 112)
        for channel in range(3):
            rgbx[..., channel] = (rgbx[..., channel].astype(np.float32) * (1.0 - arr) + high_color[channel] * arr).astype(np.uint8)
        _save_rgb(root / "03_boundaries" / filename, rgbx, title=title, description=description, items=[("weak/background", (24,56,112)), ("strong", high_color)], stats=boundary_metrics, terrain=terrain, stride=1)

    _save_ocean_floor_field("terrain_mid_ocean_ridge_x1000", "mid_ocean_ridge_field.png", "Mid-ocean ridge field", "Internal ocean-floor ridge influence. Strong zones should be raised oceanic ridges rather than generic deep basins.", (100, 220, 230))
    _save_ocean_floor_field("terrain_trench_x1000", "trench_subduction_field.png", "Trench / subduction field", "Internal trench influence near active ocean-continent and arc margins. Strong zones should be narrow, deep troughs.", (18, 18, 64))
    _save_ocean_floor_field("terrain_fracture_zone_x1000", "ocean_fracture_zone_field.png", "Ocean fracture-zone field", "Transform/fracture-zone bathymetry that offsets ridges and breaks up smooth ocean floors.", (160, 110, 220))
    _save_ocean_floor_field("terrain_seamount_x1000", "seamount_hotspot_field.png", "Seamount / hotspot field", "Submerged hotspot and volcanic seamount chains. Some later become high islands, but many should stay below sea level.", (230, 142, 70))

    if getattr(terrain, "plate_tectonic_ocean_floor_class", None) is not None:
        pof = np.asarray(terrain.plate_tectonic_ocean_floor_class, dtype=np.uint8)
        rgbpo = np.zeros((*pof.shape, 3), dtype=np.uint8)
        for code, color in OCEAN_FLOOR_COLORS.items():
            rgbpo[pof == code] = color
        _save_rgb(root / "03_boundaries" / "plate_tectonic_v1_ocean_floor_classes.png", rgbpo, title="Plate tectonic v1 native ocean-floor classes", description="Plate Terrain 4 native ocean-floor classes derived from native plate motion and boundary fields. These now coexist with native continental-relief shaping; later stages can let plate mode own full bathymetry/elevation.", items=[(OCEAN_FLOOR_LABELS[k], OCEAN_FLOOR_COLORS[k]) for k in sorted(OCEAN_FLOOR_LABELS)], stats=boundary_metrics, terrain=terrain, stride=1)

    _save_ocean_floor_field("plate_tectonic_ocean_crust_age_x1000", "plate_tectonic_v1_ocean_crust_age.png", "Plate tectonic v1 ocean-crust age", "Native ocean-crust age proxy: youngest near spreading ridges, older toward abyssal plains and subduction zones.", (215, 194, 124))
    _save_ocean_floor_field("plate_tectonic_mid_ocean_ridge_x1000", "plate_tectonic_v1_mid_ocean_ridge_field.png", "Plate tectonic v1 mid-ocean ridge field", "Native spreading-ridge field derived from divergent oceanic plate boundaries.", (100, 230, 232))
    _save_ocean_floor_field("plate_tectonic_trench_x1000", "plate_tectonic_v1_trench_field.png", "Plate tectonic v1 trench field", "Native trench/subduction field derived from convergent oceanic/continental and oceanic/oceanic plate boundaries.", (18, 18, 64))
    _save_ocean_floor_field("plate_tectonic_fracture_zone_x1000", "plate_tectonic_v1_fracture_zone_field.png", "Plate tectonic v1 fracture-zone field", "Native oceanic transform/fracture-zone field derived from tangential relative plate motion.", (166, 110, 226))
    _save_ocean_floor_field("plate_tectonic_abyssal_plain_x1000", "plate_tectonic_v1_abyssal_plain_field.png", "Plate tectonic v1 abyssal-plain field", "Native abyssal-plain/deepening proxy derived from older oceanic crust away from spreading ridges.", (54, 82, 148))
    _save_ocean_floor_field("plate_tectonic_seamount_x1000", "plate_tectonic_v1_seamount_field.png", "Plate tectonic v1 seamount/hotspot field", "Native seamount/hotspot field used to seed later oceanic island chains and submerged volcanic provinces.", (230, 142, 70))


    # Stage 3C.3 internal relief diagnostic rasters. These are generator fields,
    # not just final-elevation classifications, so they show why the relief was
    # created: mountains, basins, rifts, shields, plateaus, and interior energy.
    relief_metrics = review.get("subphases", {}).get("terrain-mountains-basins-rifts", {}).get("metrics", {})
    def _save_relief_field(attr_name: str, filename: str, title: str, description: str, low_color: tuple[int,int,int], high_color: tuple[int,int,int]):
        value = getattr(terrain, attr_name, None)
        if value is None:
            return
        arr = np.asarray(value, dtype=np.float32) / 1000.0
        if arr.size == 0:
            return
        arr = np.clip(arr, 0.0, 1.0)
        rgbx = np.zeros((*arr.shape, 3), dtype=np.uint8)
        for channel in range(3):
            rgbx[..., channel] = (low_color[channel] + (high_color[channel] - low_color[channel]) * arr).astype(np.uint8)
        _save_rgb(root / "04_mountains_basins" / filename, rgbx, title=title, description=description, items=[("weak/absent", low_color), ("strong", high_color)], stats=relief_metrics, terrain=terrain, stride=1)

    _save_relief_field("terrain_mountain_strength_x1000", "mountain_belt_strength.png", "Mountain belt strength field", "Internal mountain/orogenic influence field produced by collision, arc, suture, terrane, and procedural mountain systems. Strong zones should become major climate barriers and drainage divides.", (55, 78, 105), (238, 230, 190))
    _save_relief_field("terrain_basin_field_x1000", "basin_field.png", "Basin field", "Internal basin/lowland influence field produced by foreland, rift, sedimentary, and broad interior subsidence zones. Strong zones should later favor low-gradient rivers, lakes, plains, or endorheic basins.", (58, 88, 110), (210, 184, 112))
    _save_relief_field("terrain_rift_field_x1000", "rift_valley_field.png", "Rift valley field", "Internal rift influence field. Strong zones should cut long valleys, create rift shoulders, and provide natural continent-breakup corridors without forced supercontinent splitting.", (46, 76, 118), (82, 180, 238))
    _save_relief_field("terrain_interior_relief_x1000", "interior_relief_field.png", "Interior relief field", "Combined internal relief energy inside continents. This helps diagnose whether large landmasses have enough shields, plateaus, old ranges, basins, and escarpments to avoid flat interiors.", (70, 88, 86), (226, 190, 112))
    _save_relief_field("terrain_shield_highland_x1000", "shield_highland_field.png", "Shield and old highland field", "Stable craton/shield and old highland influence. Strong zones should become ancient continental interiors or eroded uplands rather than young sharp mountain belts.", (60, 82, 92), (210, 196, 150))
    _save_relief_field("terrain_plateau_x1000", "plateau_field.png", "Plateau field", "Broad uplift and plateau influence from rift shoulders, shields, terranes, and continental-scale provinces. Strong zones should make wide highlands rather than narrow linear ranges.", (62, 80, 98), (194, 132, 88))

    gy, gx = np.gradient(elev_ds); slope = np.sqrt(gx*gx + gy*gy); land_slope = slope[land_ds]
    slope_q = np.percentile(land_slope, 88) if land_slope.size else 1.0
    high_q = np.percentile(elev_ds[land_ds], 82) if land_ds.any() else 1000
    low_q = np.percentile(elev_ds[land_ds], 25) if land_ds.any() else 0
    mountain = land_ds & ((elev_ds > high_q) | (slope > slope_q))
    basin = land_ds & (elev_ds < low_q) & (slope < (np.percentile(land_slope, 45) if land_slope.size else 1.0))
    rgbm = np.zeros((h,w,3), dtype=np.uint8); rgbm[:] = (45,86,128); rgbm[land_ds]=(145,150,102); rgbm[basin]=(188,176,124); rgbm[mountain]=(220,220,188); rgbm[land_ds & (elev_ds>1300) & ~mountain]=(170,125,92)
    _save_rgb(root / "04_mountains_basins" / "mountains_basins_relief.png", rgbm, title="Mountains, basins, and interior relief", description="Proxy map for mountains/highlands, basins, plateaus, and broad interior relief before internal terrain passes are fully split.", items=[("ocean", (45,86,128)), ("land", (145,150,102)), ("basin/lowland", (188,176,124)), ("plateau/upland", (170,125,92)), ("mountain/rugged highland", (220,220,188))], stats=review.get("subphases", {}).get("terrain-mountains-basins-rifts", {}).get("metrics", {}), terrain=terrain, stride=stride)

    def _save_plate_relief_field(attr_name: str, filename: str, title: str, description: str, high_color: tuple[int, int, int]):
        value = getattr(terrain, attr_name, None)
        if value is None:
            return
        arr = np.asarray(value, dtype=np.float32) / 1000.0
        if arr.size == 0:
            return
        arr = np.clip(arr, 0.0, 1.0)
        rgbp = np.zeros((*arr.shape, 3), dtype=np.uint8)
        low_color = (46, 66, 94)
        for channel in range(3):
            rgbp[..., channel] = (low_color[channel] + (high_color[channel] - low_color[channel]) * arr).astype(np.uint8)
        _save_rgb(root / "04_mountains_basins" / filename, rgbp, title=title, description=description, items=[("weak/absent", low_color), ("strong", high_color)], stats=relief_metrics, terrain=terrain, stride=1)

    _save_plate_relief_field("plate_tectonic_orogeny_strength_x1000", "plate_tectonic_v1_orogeny_strength.png", "Plate tectonic v1 orogeny strength", "Plate Terrain 4 native collision/orogeny field. Strong zones are built from native convergence, continental crust participation, and continental-collision polarity; in plate_tectonic_v1 this now directly uplifts terrain.", (230, 214, 168))
    _save_plate_relief_field("plate_tectonic_volcanic_arc_x1000", "plate_tectonic_v1_volcanic_arc_field.png", "Plate tectonic v1 volcanic arc field", "Native subduction/arc relief field from oceanic subduction polarity, convergence, trench proximity, and microplate/land adjacency. This directly supports coastal arcs and island-arc uplands.", (230, 104, 64))
    _save_plate_relief_field("plate_tectonic_continental_rift_x1000", "plate_tectonic_v1_continental_rift_field.png", "Plate tectonic v1 continental rift field", "Native continental divergence/rift field. Strong zones lower rift valleys and raise shoulders in the plate-derived relief pass.", (96, 174, 236))
    _save_plate_relief_field("plate_tectonic_foreland_basin_x1000", "plate_tectonic_v1_foreland_basin_field.png", "Plate tectonic v1 foreland basin field", "Native basin field adjacent to collision/arc belts. Strong zones are lowered or sediment-accommodation regions downstream of uplift.", (112, 204, 124))
    _save_plate_relief_field("plate_tectonic_craton_shield_x1000", "plate_tectonic_v1_craton_shield_field.png", "Plate tectonic v1 craton/shield field", "Stable old continental core and shield-upland field. Strong zones create broad, lower-amplitude continental interior relief instead of blank flat interiors.", (218, 190, 116))
    _save_plate_relief_field("plate_tectonic_accreted_terrane_x1000", "plate_tectonic_v1_accreted_terrane_field.png", "Plate tectonic v1 accreted terrane field", "Native microplate/accreted-terrane relief field. Strong zones mark broken coastal uplands, accretionary belts, and microplate collision terrain.", (206, 124, 216))
    _save_plate_relief_field("plate_tectonic_plateau_uplift_x1000", "plate_tectonic_v1_plateau_uplift_field.png", "Plate tectonic v1 plateau uplift field", "Native broad uplift field combining craton/shield, rift-shoulder, transform, and terrane contributions.", (204, 148, 98))
    _save_plate_relief_field("plate_tectonic_sedimentary_plain_x1000", "plate_tectonic_v1_sedimentary_plain_field.png", "Plate tectonic v1 sedimentary / inland plain field", "Plate Terrain 11 explicit plain/accommodation field for foreland plains, sedimentary basins, rift-adjacent lowlands, and future lake/river corridors.", (116, 202, 132))

    _save_plate_relief_field("plate_tectonic_valley_corridor_x1000", "plate_tectonic_v1_valley_corridor_field.png", "Plate tectonic v1 valley corridor field", "Plate Terrain 11 drainage-ready valley corridors. These are terrain-carved corridors that should later guide rivers through plains, rifts, forelands, and mountain-front basins.", (78, 184, 216))
    _save_plate_relief_field("plate_tectonic_inland_basin_x1000", "plate_tectonic_v1_inland_basin_field.png", "Plate tectonic v1 inland basin field", "Plate Terrain 11 interior basin/accommodation field for foreland depressions, rift basins, sedimentary lowlands, and endorheic-prone regions.", (112, 174, 126))
    _save_plate_relief_field("plate_tectonic_lake_candidate_x1000", "plate_tectonic_v1_lake_candidate_field.png", "Plate tectonic v1 lake candidate field", "Plate Terrain 11 pre-hydrology lake-candidate depressions. These remain land terrain until the hydrology stage decides whether they become lakes, through-flow lakes, or dry basins.", (82, 164, 224))
    _save_plate_relief_field("plate_tectonic_terrain_detail_x1000", "plate_tectonic_v1_terrain_detail_field.png", "Plate tectonic v1 terrain detail field", "Plate Terrain 11 native texture/detail field restoring shields, uplands, rift shoulders, terranes, and non-concentric relief without calling the legacy terrain backend.", (202, 158, 104))

    if getattr(terrain, "plate_tectonic_landform_class", None) is not None:
        lf = np.asarray(terrain.plate_tectonic_landform_class, dtype=np.uint8)
        if lf.size:
            colors = {0:(40,76,128),1:(150,142,104),2:(230,220,178),3:(218,92,64),4:(88,176,230),5:(116,202,132),6:(194,136,88),7:(202,128,214)}
            labels_lf = {0:"water/background",1:"shield/craton",2:"orogen/mountain belt",3:"volcanic arc",4:"rift valley",5:"sedimentary/plain basin",6:"plateau/uplift",7:"accreted terrane"}
            rgblf = np.zeros((*lf.shape,3), dtype=np.uint8)
            for code,color in colors.items(): rgblf[lf == code] = color
            _save_rgb(root / "04_mountains_basins" / "plate_tectonic_v1_landform_classes.png", rgblf, title="Plate tectonic v1 landform classes", description="Plate Terrain 11 landform systems. Elevation is now driven by shields, orogens, arcs, rifts, plains/basins, plateaus, and accreted terranes rather than only distance from coast.", items=[(labels_lf[k], colors[k]) for k in sorted(colors)], stats=relief_metrics, terrain=terrain, stride=1)

    if getattr(terrain, "plate_tectonic_relief_delta_m", None) is not None:
        delta = np.asarray(terrain.plate_tectonic_relief_delta_m, dtype=np.float32)
        if delta.size:
            limit = max(150.0, float(np.percentile(np.abs(delta), 96)))
            norm = np.clip(delta / limit, -1.0, 1.0)
            rgbd = np.zeros((*norm.shape, 3), dtype=np.uint8)
            rgbd[..., 0] = np.where(norm >= 0, (92 + 140 * norm).astype(np.uint8), (48 + 55 * (1.0 + norm)).astype(np.uint8))
            rgbd[..., 1] = np.where(norm >= 0, (88 + 118 * (1.0 - norm)).astype(np.uint8), (82 + 120 * (-norm)).astype(np.uint8))
            rgbd[..., 2] = np.where(norm >= 0, (80 + 64 * (1.0 - norm)).astype(np.uint8), (126 + 80 * (-norm)).astype(np.uint8))
            _save_rgb(root / "04_mountains_basins" / "plate_tectonic_v1_relief_delta.png", rgbd, title="Plate tectonic v1 relief delta", description="Elevation change applied by Plate Terrain 4. Warm colors show plate-derived uplift; blue/green colors show rift or foreland-basin lowering. This is the first native plate field that directly changes terrain elevation.", items=[("subsidence/lowering", (48,202,206)), ("little change", (92,88,80)), ("uplift", (232,88,80))], stats=relief_metrics, terrain=terrain, stride=1)

    if getattr(terrain, "plate_tectonic_drainage_ready_delta_m", None) is not None:
        delta = np.asarray(terrain.plate_tectonic_drainage_ready_delta_m, dtype=np.float32)
        if delta.size:
            limit = max(80.0, float(np.percentile(np.abs(delta), 96)))
            norm = np.clip(delta / limit, -1.0, 1.0)
            rgbdr = np.zeros((*norm.shape, 3), dtype=np.uint8)
            rgbdr[..., 0] = np.where(norm >= 0, (86 + 130 * norm).astype(np.uint8), (38 + 52 * (1.0 + norm)).astype(np.uint8))
            rgbdr[..., 1] = np.where(norm >= 0, (92 + 88 * (1.0 - norm)).astype(np.uint8), (96 + 120 * (-norm)).astype(np.uint8))
            rgbdr[..., 2] = np.where(norm >= 0, (82 + 55 * (1.0 - norm)).astype(np.uint8), (138 + 80 * (-norm)).astype(np.uint8))
            _save_rgb(root / "04_mountains_basins" / "plate_tectonic_v1_drainage_ready_delta.png", rgbdr, title="Plate tectonic v1 drainage-ready delta", description="Plate Terrain 11 elevation change from valley corridors, interior basins, lake-candidate depressions, and native terrain detail. Blue/green indicates carved drainage-ready lowlands; warm tones show shoulders/upland detail.", items=[("carved basin/valley", (38,216,218)), ("little change", (86,92,82)), ("detail/upland shoulder", (216,92,82))], stats=relief_metrics, terrain=terrain, stride=1)

    # Coast/shelf/island map
    ocean = ~land_ds; shelf = ocean & (elev_ds > -350)
    island_limit = max(4, int(h*w*0.0038)); sizes = np.bincount(labels.ravel()) if labels is not None else np.array([])
    island_mask = land_ds & (sizes[labels] <= island_limit) if sizes.size else np.zeros_like(land_ds)
    coast_metrics = review.get("subphases", {}).get("terrain-coasts-shelves-islands", {}).get("metrics", {})
    rgbco = np.zeros((h,w,3), dtype=np.uint8); rgbco[:] = (28,72,125); rgbco[shelf]=(72,145,178); rgbco[land_ds]=(136,147,100); rgbco[coast_ds]=(235,220,110); rgbco[island_mask]=(235,155,65); rgbco[island_mask & (elev_ds>800)] = (245,235,190)
    _save_rgb(root / "05_coasts_islands" / "coasts_shelves_islands.png", rgbco, title="Coasts, shelves, and islands", description="Coastline, shallow shelf, and island-component diagnostic. Climate-aware fjord/delta typing will be added after climate review.", items=[("deep ocean", (28,72,125)), ("shallow shelf", (72,145,178)), ("large land", (136,147,100)), ("coastline", (235,220,110)), ("island/archipelago", (235,155,65)), ("high island/volcanic proxy", (245,235,190))], stats=coast_metrics, terrain=terrain, stride=stride)

    if getattr(terrain, "terrain_coast_style_class", None) is not None:
        style = np.asarray(terrain.terrain_coast_style_class, dtype=np.uint8)[::stride, ::stride]
        rgb_style = np.zeros((*style.shape, 3), dtype=np.uint8)
        for code, color in COAST_STYLE_COLORS.items():
            rgb_style[style == code] = color
        rgb_style[(style == 0) & land_ds] = (132, 146, 105)
        _save_rgb(root / "05_coasts_islands" / "coast_style_classes.png", rgb_style, title="Coast style classes", description="Stage 3C.4 coast style classes: passive smooth, rugged/fjorded, rifted gulf, volcanic arc, shelf/deltaic plain, and mixed irregular coasts.", items=[(COAST_STYLE_LABELS[k], COAST_STYLE_COLORS[k]) for k in sorted(COAST_STYLE_LABELS)], stats=coast_metrics, terrain=terrain, stride=stride)

    if getattr(terrain, "terrain_shelf_width_x1000", None) is not None:
        sf = np.asarray(terrain.terrain_shelf_width_x1000, dtype=np.float32)[::stride, ::stride] / 1000.0
        sf = np.clip(sf, 0.0, 1.0)
        rgbsf = np.zeros((*sf.shape, 3), dtype=np.uint8)
        rgbsf[...,0] = (28 + 86 * sf).astype(np.uint8)
        rgbsf[...,1] = (72 + 138 * sf).astype(np.uint8)
        rgbsf[...,2] = (125 + 80 * (1.0 - sf)).astype(np.uint8)
        rgbsf[land_ds] = (130, 145, 104)
        _save_rgb(root / "05_coasts_islands" / "shelf_width_field.png", rgbsf, title="Shelf width / shallow-margin field", description="Internal shelf influence produced by the Stage 3C.4 coast pass. Broad shelves should support coastal plains, shallow seas, shelf islands, and later sediment/deposition behavior.", items=[("deep ocean/background", (28,72,125)), ("broad shelf influence", (114,210,125)), ("land", (130,145,104))], stats=coast_metrics, terrain=terrain, stride=stride)

    if getattr(terrain, "terrain_coast_ruggedness_x1000", None) is not None:
        cr = np.asarray(terrain.terrain_coast_ruggedness_x1000, dtype=np.float32)[::stride, ::stride] / 1000.0
        cr = np.clip(cr, 0.0, 1.0)
        rgbcr = np.zeros((*cr.shape, 3), dtype=np.uint8); rgbcr[:] = (35,82,135); rgbcr[land_ds]=(132,146,105)
        mask = cr > 0
        rgbcr[...,0] = np.where(mask, (85 + 160 * cr).astype(np.uint8), rgbcr[...,0])
        rgbcr[...,1] = np.where(mask, (95 + 110 * (1.0 - cr)).astype(np.uint8), rgbcr[...,1])
        rgbcr[...,2] = np.where(mask, (95 + 55 * (1.0 - cr)).astype(np.uint8), rgbcr[...,2])
        _save_rgb(root / "05_coasts_islands" / "coast_ruggedness_field.png", rgbcr, title="Coast ruggedness field", description="Local relief contrast along coastlines. This helps inspect fjorded/rugged margins versus smooth passive or coastal-plain margins.", items=[("background/ocean", (35,82,135)), ("land", (132,146,105)), ("strong rugged coast", (245,95,95))], stats=coast_metrics, terrain=terrain, stride=stride)

    if getattr(terrain, "terrain_island_origin_class", None) is not None:
        origin = np.asarray(terrain.terrain_island_origin_class, dtype=np.uint8)[::stride, ::stride]
        rgbo = np.zeros((*origin.shape, 3), dtype=np.uint8)
        for code, color in ISLAND_ORIGIN_COLORS.items():
            rgbo[origin == code] = color
        _save_rgb(root / "05_coasts_islands" / "island_origin_classes.png", rgbo, title="Island origin classes", description="Diagnostic island-origin classes distinguishing shelf islands, volcanic/arc islands, microcontinents/terranes, hotspot/high islands, and large landmasses.", items=[(ISLAND_ORIGIN_LABELS[k], ISLAND_ORIGIN_COLORS[k]) for k in sorted(ISLAND_ORIGIN_LABELS)], stats=coast_metrics, terrain=terrain, stride=stride)

    if getattr(terrain, "plate_tectonic_margin_class", None) is not None:
        margin = np.asarray(terrain.plate_tectonic_margin_class, dtype=np.uint8)
        rgbmarg = np.zeros((*margin.shape, 3), dtype=np.uint8)
        for code, color in PLATE_MARGIN_COLORS.items():
            rgbmarg[margin == code] = color
        _save_rgb(root / "05_coasts_islands" / "plate_tectonic_v1_margin_classes.png", rgbmarg, title="Plate tectonic v1 margin classes", description="Native Plate Terrain 5 coast/margin classes derived from continental crust, active/passive/rift/transform boundaries, volcanic arcs, and microplates. This is the plate-mode replacement for generic coast-style inference.", items=[(PLATE_MARGIN_LABELS[k], PLATE_MARGIN_COLORS[k]) for k in sorted(PLATE_MARGIN_LABELS)], stats=coast_metrics, terrain=terrain, stride=1)
    if getattr(terrain, "plate_tectonic_margin_profile_class", None) is not None:
        mp = np.asarray(terrain.plate_tectonic_margin_profile_class, dtype=np.uint8)
        if mp.size:
            colors_mp = {0:(28,72,125),1:(104,196,156),2:(210,96,72),3:(82,172,220),4:(226,132,72),5:(188,120,210),6:(176,170,118)}
            labels_mp = {0:"background",1:"passive shelf/plain",2:"active trench/steep",3:"rifted gulf",4:"volcanic arc",5:"transform/escarpment",6:"mixed transition"}
            rgbmp = np.zeros((*mp.shape,3), dtype=np.uint8)
            for code,color in colors_mp.items(): rgbmp[mp == code] = color
            _save_rgb(root / "05_coasts_islands" / "plate_tectonic_v1_margin_profiles.png", rgbmp, title="Plate tectonic v1 margin-profile classes", description="Plate Terrain 11 margin profiles. These are the segment profiles that control shelf width, coastal plains, active steep margins, rifted gulfs, volcanic arcs, and transform escarpment coasts.", items=[(labels_mp[k], colors_mp[k]) for k in sorted(colors_mp)], stats=coast_metrics, terrain=terrain, stride=1)

    def _save_plate_coast_field(attr_name: str, filename: str, title: str, description: str, high_color: tuple[int, int, int]):
        value = getattr(terrain, attr_name, None)
        if value is None:
            return
        arr = np.asarray(value, dtype=np.float32) / 1000.0
        if arr.size == 0:
            return
        arr = np.clip(arr, 0.0, 1.0)
        rgbp = np.zeros((*arr.shape, 3), dtype=np.uint8)
        low_color = (36, 70, 112)
        for channel in range(3):
            rgbp[..., channel] = (low_color[channel] + (high_color[channel] - low_color[channel]) * arr).astype(np.uint8)
        _save_rgb(root / "05_coasts_islands" / filename, rgbp, title=title, description=description, items=[("weak/absent", low_color), ("strong", high_color)], stats=coast_metrics, terrain=terrain, stride=1)

    _save_plate_coast_field("plate_tectonic_shelf_width_x1000", "plate_tectonic_v1_shelf_width.png", "Plate tectonic v1 shelf width", "Native plate shelf field. Broad shelves now come mainly from continental-crust carrier plates and passive/rifted margins rather than distance from every island.", (114, 215, 156))
    _save_plate_coast_field("plate_tectonic_active_margin_x1000", "plate_tectonic_v1_active_margins.png", "Plate tectonic v1 active margins", "Active/subduction/arc/transform margin influence. Strong active margins should allow deep water near land and rugged coastal relief rather than broad passive shelves.", (226, 88, 64))
    _save_plate_coast_field("plate_tectonic_passive_margin_x1000", "plate_tectonic_v1_passive_margins.png", "Plate tectonic v1 passive margins", "Passive continental margin influence. Strong passive margins support broad shelves, smoother coasts, coastal plains, and later sediment/deposition behavior.", (112, 196, 128))
    _save_plate_coast_field("plate_tectonic_rifted_margin_x1000", "plate_tectonic_v1_rifted_margins.png", "Plate tectonic v1 rifted margins", "Rifted/gulf margin influence derived from plate divergence across continental or mixed crust. These should support gulfs, narrow seas, and rifted coastlines.", (96, 158, 238))
    _save_plate_coast_field("plate_tectonic_island_arc_x1000", "plate_tectonic_v1_island_arc_field.png", "Plate tectonic v1 island arc field", "Volcanic/island-arc coast field derived from subduction polarity, convergence, trenches, microplates, and arc relief.", (232, 120, 62))
    _save_plate_coast_field("plate_tectonic_coastal_plain_x1000", "plate_tectonic_v1_coastal_plain_field.png", "Plate tectonic v1 coastal plain field", "Passive/rifted shelf-adjacent coastal plain field. Strong zones should be low, smooth coastal margins rather than exact one-meter flats.", (102, 210, 150))
    _save_plate_coast_field("plate_tectonic_coast_ruggedness_x1000", "plate_tectonic_v1_coast_ruggedness.png", "Plate tectonic v1 coast ruggedness", "Native plate coast ruggedness from active margins, arcs, transform margins, and local relief. This should separate active rugged coasts from passive smooth ones.", (238, 92, 92))

    if getattr(terrain, "plate_tectonic_island_origin_class", None) is not None:
        porigin = np.asarray(terrain.plate_tectonic_island_origin_class, dtype=np.uint8)
        rgbpo = np.zeros((*porigin.shape, 3), dtype=np.uint8)
        for code, color in ISLAND_ORIGIN_COLORS.items():
            rgbpo[porigin == code] = color
        _save_rgb(root / "05_coasts_islands" / "plate_tectonic_v1_island_origin_classes.png", rgbpo, title="Plate tectonic v1 island origin classes", description="Native plate island-origin classes derived from shelf/margin, volcanic-arc, microplate, rift, and high-island context.", items=[(ISLAND_ORIGIN_LABELS[k], ISLAND_ORIGIN_COLORS[k]) for k in sorted(ISLAND_ORIGIN_LABELS)], stats=coast_metrics, terrain=terrain, stride=1)

    if getattr(terrain, "plate_tectonic_coast_delta_m", None) is not None:
        cdelta = np.asarray(terrain.plate_tectonic_coast_delta_m, dtype=np.float32)
        if cdelta.size:
            limit = max(80.0, float(np.percentile(np.abs(cdelta), 96)))
            norm = np.clip(cdelta / limit, -1.0, 1.0)
            rgbcd = np.zeros((*norm.shape, 3), dtype=np.uint8); rgbcd[:] = (90, 92, 102)
            rgbcd[...,0] = np.where(norm >= 0, (95 + 140 * norm).astype(np.uint8), (50 + 55 * (1.0 + norm)).astype(np.uint8))
            rgbcd[...,1] = np.where(norm >= 0, (94 + 80 * (1.0 - norm)).astype(np.uint8), (112 + 95 * (-norm)).astype(np.uint8))
            rgbcd[...,2] = np.where(norm >= 0, (96 + 40 * (1.0 - norm)).astype(np.uint8), (140 + 80 * (-norm)).astype(np.uint8))
            _save_rgb(root / "05_coasts_islands" / "plate_tectonic_v1_coast_delta.png", rgbcd, title="Plate tectonic v1 coast/shelf elevation delta", description="Elevation changes from Plate Terrain 5 coast/shelf/island pass. Warm colors are coastal/island uplift; blue colors are active-margin deepening or rift/shelf bathymetry adjustment.", items=[("deepening/lowering", (50,207,220)), ("little change", (90,92,102)), ("uplift/shallower", (235,94,96))], stats=coast_metrics, terrain=terrain, stride=1)

    if getattr(terrain, "terrain_island_shape_complexity_x1000", None) is not None:
        isc = np.asarray(terrain.terrain_island_shape_complexity_x1000, dtype=np.float32) / 1000.0
        isc = np.clip(isc, 0.0, 1.0)
        rgbi = np.zeros((*isc.shape, 3), dtype=np.uint8); rgbi[:] = (35,82,135)
        mask = isc > 0
        rgbi[...,0] = np.where(mask, (120 + 120 * isc).astype(np.uint8), rgbi[...,0])
        rgbi[...,1] = np.where(mask, (105 + 115 * isc).astype(np.uint8), rgbi[...,1])
        rgbi[...,2] = np.where(mask, (72 + 80 * (1.0 - isc)).astype(np.uint8), rgbi[...,2])
        _save_rgb(root / "05_coasts_islands" / "island_shape_complexity.png", rgbi, title="Island shape complexity", description="Terrain Revisit 2 island morphology diagnostic. Low values indicate oval/blob-like islands; high values mark lobed, chained, or irregular island components.", items=[("ocean/background", (35,82,135)), ("low/oval", (120,105,152)), ("high/lobed or chained", (240,220,72))], stats=coast_metrics, terrain=terrain, stride=1)

    # Stage 3C.5 erosion/deposition diagnostic rasters. These are generator
    # fields, not final hydrology products. They show how terrain was conditioned
    # before rivers are generated.
    erosion_metrics = review.get("subphases", {}).get("terrain-erosion-deposition", {}).get("metrics", {})

    def _save_stage6_field(attr_name: str, filename: str, title: str, description: str, low_color: tuple[int,int,int], high_color: tuple[int,int,int]):
        value = getattr(terrain, attr_name, None)
        if value is None:
            return
        arr = np.asarray(value, dtype=np.float32) / 1000.0
        if arr.size == 0:
            return
        arr = np.clip(arr, 0.0, 1.0)
        rgbx = np.zeros((*arr.shape, 3), dtype=np.uint8)
        for channel in range(3):
            rgbx[..., channel] = (low_color[channel] + (high_color[channel] - low_color[channel]) * arr).astype(np.uint8)
        _save_rgb(root / "06_erosion_deposition" / filename, rgbx, title=title, description=description, items=[("weak/absent", low_color), ("strong", high_color)], stats=erosion_metrics, terrain=terrain, stride=1)

    _save_stage6_field("terrain_erosion_strength_x1000", "erosion_strength_field.png", "Erosion strength field", "Stage 3C.5 internal erosion strength field. Strong zones combine slope, erodibility, valley routing, wetness proxy, relief, and terrain maturity.", (62, 82, 94), (232, 188, 108))
    _save_stage6_field("terrain_deposition_field_x1000", "deposition_field.png", "Deposition field", "Stage 3C.5 sediment sink field. Strong zones should correspond to basin floors, coastal plains, floodplains, alluvial fans, and low-gradient sediment accommodation areas.", (52, 82, 116), (95, 205, 126))
    _save_stage6_field("terrain_valley_corridor_x1000", "valley_corridor_field.png", "Valley corridor field", "Pre-hydrology drainage-corridor readiness field. Strong zones indicate terrain corridors likely to guide future rivers through mountains, rifts, basins, and lowlands.", (48, 74, 112), (92, 188, 238))
    _save_stage6_field("terrain_sediment_supply_x1000", "sediment_supply_field.png", "Sediment supply field", "Terrain sediment source field from mountain/upland relief, rifts, valley power, and erodibility. Strong source zones should feed plains, fans, and basins.", (72, 78, 84), (226, 168, 92))
    _save_stage6_field("terrain_coastal_plain_x1000", "coastal_plain_tendency.png", "Coastal plain tendency", "Passive-margin and shelf-adjacent lowland tendency. Strong zones should support smooth coastal plains and later delta/floodplain development.", (45, 82, 120), (104, 210, 154))
    _save_stage6_field("terrain_alluvial_fan_x1000", "alluvial_fan_tendency.png", "Alluvial fan tendency", "Fan-like deposition tendency where high relief or rift shoulders drop into basins, lowlands, or coastal plains.", (75, 74, 88), (220, 176, 96))
    _save_stage6_field("terrain_floodplain_x1000", "floodplain_tendency.png", "Floodplain tendency", "Broad low-gradient floodplain tendency before final hydrology. Strong zones should help future rivers produce wider plains instead of narrow forced lines.", (50, 78, 116), (112, 202, 132))
    _save_stage6_field("terrain_maturity_x1000", "terrain_maturity_field.png", "Terrain maturity field", "How worked-over the terrain became during Stage 3C.5. Mature zones are smoother, more sediment-filled, or more eroded; immature zones preserve sharper raw relief.", (72, 82, 92), (214, 202, 142))

    if getattr(terrain, "terrain_relief_delta_m", None) is not None:
        delta = np.asarray(terrain.terrain_relief_delta_m, dtype=np.float32)
        if delta.size:
            limit = max(100.0, float(np.percentile(np.abs(delta), 96)))
            norm_pos = np.clip(delta / limit, 0.0, 1.0)
            norm_neg = np.clip(-delta / limit, 0.0, 1.0)
            rgbd = np.zeros((*delta.shape, 3), dtype=np.uint8); rgbd[:] = (115, 116, 108)
            # Red/orange = cut down by erosion; green = filled/deposited.
            rgbd[...,0] = np.where(norm_neg > 0, (120 + 120 * norm_neg).astype(np.uint8), rgbd[...,0])
            rgbd[...,1] = np.where(norm_neg > 0, (105 - 25 * norm_neg).astype(np.uint8), rgbd[...,1])
            rgbd[...,2] = np.where(norm_neg > 0, (90 - 35 * norm_neg).astype(np.uint8), rgbd[...,2])
            rgbd[...,0] = np.where(norm_pos > 0, (90 - 30 * norm_pos).astype(np.uint8), rgbd[...,0])
            rgbd[...,1] = np.where(norm_pos > 0, (130 + 100 * norm_pos).astype(np.uint8), rgbd[...,1])
            rgbd[...,2] = np.where(norm_pos > 0, (92 + 35 * norm_pos).astype(np.uint8), rgbd[...,2])
            _save_rgb(root / "06_erosion_deposition" / "before_after_relief_delta.png", rgbd, title="Before/after relief delta", description="Signed terrain change introduced by Stage 3C.5. Orange/red indicates erosion or valley incision; green indicates deposition or basin/plain fill.", items=[("erosion / cut down", (230,80,55)), ("little change", (115,116,108)), ("deposition / fill", (70,220,120))], stats=erosion_metrics, terrain=terrain, stride=1)

    # Erosion/deposition proxy
    lowland = land_ds & (elev_ds < 250); gentle = land_ds & (slope < (np.percentile(land_slope, 35) if land_slope.size else 1.0)); depos = lowland & gentle
    rgbe = np.zeros((h,w,3), dtype=np.uint8); rgbe[:] = (35,80,130); rgbe[land_ds]=(145,145,100); rgbe[gentle]=(190,180,120); rgbe[depos]=(90,180,115); rgbe[mountain]=(215,215,185)
    _save_rgb(root / "06_erosion_deposition" / "erosion_deposition_proxy.png", rgbe, title="Erosion and deposition proxy", description="Gentle lowlands, sediment-accommodation zones, basins, and rugged source terrain inferred from final elevation/slope.", items=[("ocean", (35,80,130)), ("ordinary land", (145,145,100)), ("gentle plain", (190,180,120)), ("likely deposition/lowland", (90,180,115)), ("rugged source terrain", (215,215,185))], stats=erosion_metrics, terrain=terrain, stride=stride)

    # Final terrain class map
    final_metrics = review.get("subphases", {}).get("terrain-finalization-recentering", {}).get("metrics", {})
    final_quality = review.get("final_quality", {}) if isinstance(review.get("final_quality"), dict) else {}
    readiness_checks = review.get("hydrology_readiness_checks", []) if isinstance(review.get("hydrology_readiness_checks"), list) else []
    (root / "07_final").mkdir(parents=True, exist_ok=True)
    (root / "07_final" / "final_terrain_quality_report.json").write_text(json.dumps({"final_quality": final_quality, "hydrology_readiness_checks": readiness_checks, "warnings": review.get("warnings", [])}, indent=2), encoding="utf-8")

    rgbf = np.zeros((h,w,3), dtype=np.uint8); rgbf[:] = (30,70,120); rgbf[ocean & (elev_ds>-450)] = (65,135,195); rgbf[land_ds & (elev_ds<250)] = (156,176,112); rgbf[land_ds & (elev_ds>=250) & (elev_ds<1200)] = (143,135,86); rgbf[land_ds & (elev_ds>=1200)] = (218,210,176); rgbf[coast_ds] = (25,25,20)
    _save_rgb(root / "07_final" / "final_terrain_classes.png", rgbf, title="Final terrain class diagnostic", description="Simplified final terrain classes for quick quality review.", items=[("deep ocean", (30,70,120)), ("shallow sea/shelf", (65,135,195)), ("lowland", (156,176,112)), ("upland", (143,135,86)), ("mountain/highland", (218,210,176)), ("coastline", (25,25,20))], stats=final_metrics, terrain=terrain, stride=stride)

    def _fit_field(value, *, scale: float = 1.0):
        if value is None:
            return np.zeros((h, w), dtype=np.float32)
        arr = np.asarray(value, dtype=np.float32) * scale
        if arr.size == 0:
            return np.zeros((h, w), dtype=np.float32)
        if arr.shape == (h, w):
            return arr
        try:
            zoom_y = h / max(1, arr.shape[0]); zoom_x = w / max(1, arr.shape[1])
            return ndimage.zoom(arr, (zoom_y, zoom_x), order=1)[:h, :w]
        except Exception:
            return np.resize(arr, (h, w)).astype(np.float32)

    valley = np.clip(_fit_field(getattr(terrain, "terrain_valley_corridor_x1000", None), scale=0.001), 0.0, 1.0)
    deposit = np.clip(_fit_field(getattr(terrain, "terrain_deposition_field_x1000", None), scale=0.001), 0.0, 1.0)
    maturity = np.clip(_fit_field(getattr(terrain, "terrain_maturity_x1000", None), scale=0.001), 0.0, 1.0)
    coast_rug = np.clip(_fit_field(getattr(terrain, "terrain_coast_ruggedness_x1000", None), scale=0.001), 0.0, 1.0)
    relief_delta = _fit_field(getattr(terrain, "terrain_relief_delta_m", None), scale=1.0)
    slope_norm = np.zeros_like(elev_ds, dtype=np.float32)
    if land_slope.size:
        slope_norm = np.clip(slope / max(1.0, float(np.percentile(land_slope, 92))), 0.0, 1.0)

    dist = ndimage.distance_transform_edt(~coast_ds)
    interior = land_ds & (dist > max(5, min(64, w // 48)))
    flat_risk = np.clip((1.0 - slope_norm) * interior.astype(np.float32) * (1.0 - np.clip(valley * 1.3, 0.0, 1.0)), 0.0, 1.0)
    local_coast_density = ndimage.uniform_filter(coast_ds.astype(np.float32), size=max(5, min(35, w // 96)), mode="nearest")
    coast_complexity_local = np.clip(local_coast_density * 8.0 + coast_rug * 0.55, 0.0, 1.0)
    drainage = np.clip(0.45 * valley + 0.25 * deposit + 0.16 * (1.0 - flat_risk) + 0.14 * coast_complexity_local, 0.0, 1.0)
    relief_std = ndimage.uniform_filter(elev_ds * elev_ds, size=max(5, min(31, w // 100))) - ndimage.uniform_filter(elev_ds, size=max(5, min(31, w // 100))) ** 2
    relief_std = np.sqrt(np.clip(relief_std, 0.0, None))
    relief_div = np.clip(relief_std / max(1.0, float(np.percentile(relief_std[land_ds], 92)) if land_ds.any() else 1.0), 0.0, 1.0)
    diversity = np.clip(0.38 * relief_div + 0.24 * coast_complexity_local + 0.20 * valley + 0.18 * maturity, 0.0, 1.0)
    quality = np.clip(0.28 * drainage + 0.24 * diversity + 0.18 * (1.0 - flat_risk) + 0.16 * maturity + 0.14 * slope_norm, 0.0, 1.0)
    quality[ocean] = np.clip(0.35 + 0.40 * (elev_ds[ocean] > -450), 0.0, 1.0)

    def _gradient_map(arr, filename, title, description, low_color, high_color, *, ocean_color=None):
        arr = np.clip(arr, 0.0, 1.0)
        rgb = np.zeros((*arr.shape, 3), dtype=np.uint8)
        for channel in range(3):
            rgb[..., channel] = (low_color[channel] + (high_color[channel] - low_color[channel]) * arr).astype(np.uint8)
        if ocean_color is not None:
            rgb[ocean] = ocean_color
        _save_rgb(root / "07_final" / filename, rgb, title=title, description=description, items=[("low", low_color), ("high", high_color)] + ([("ocean/background", ocean_color)] if ocean_color else []), stats=final_metrics, terrain=terrain, stride=stride)

    _gradient_map(quality, "final_terrain_quality_score.png", "Final terrain quality score", "Cell-level QA score blending relief variety, drainage readiness, terrain maturity, and problem-area risk. This is an inspection aid, not a physical field used by later stages.", (105, 62, 62), (98, 205, 122), ocean_color=(38, 82, 132))
    _gradient_map(flat_risk, "flat_interior_risk.png", "Flat interior risk", "Highlights inland low-slope zones that may produce straight/simple rivers or visually empty continental interiors.", (75, 105, 95), (225, 96, 72), ocean_color=(35, 82, 135))
    _gradient_map(coast_complexity_local, "coastline_complexity_score.png", "Local coastline complexity score", "Local coastline density and ruggedness score. Low values along long coasts suggest simple/smooth margins that may need review.", (72, 105, 126), (232, 210, 92), ocean_color=(30, 76, 128))
    _gradient_map(drainage, "drainage_readiness_score.png", "Drainage readiness score", "Pre-hydrology readiness score using valley corridors, deposition/basin fields, flat-interior risk, and coast/outlet complexity.", (86, 78, 104), (90, 188, 235), ocean_color=(30, 76, 128))
    _gradient_map(diversity, "terrain_diversity_score.png", "Terrain diversity score", "Local terrain variety score from relief variation, coast complexity, valley fields, and terrain maturity.", (82, 86, 92), (220, 184, 100), ocean_color=(35, 82, 135))

    plate_integration = np.clip(_fit_field(getattr(terrain, "plate_tectonic_backend_integration_x1000", None), scale=0.001), 0.0, 1.0)
    plate_legacy = np.clip(_fit_field(getattr(terrain, "plate_tectonic_legacy_dependency_x1000", None), scale=0.001), 0.0, 1.0)
    plate_hydro = np.clip(_fit_field(getattr(terrain, "plate_tectonic_hydrology_readiness_x1000", None), scale=0.001), 0.0, 1.0)
    if np.any(plate_integration > 0):
        _gradient_map(plate_integration, "plate_tectonic_v1_backend_integration.png", "Plate tectonic v1 backend integration", "Plate Terrain 11 score showing where native plate setup, motion, ocean-floor, plate-owned foundation, relief, and coast/shelf layers support the final terrain.", (92, 78, 116), (94, 205, 132), ocean_color=None)
        _gradient_map(plate_legacy, "plate_tectonic_v1_legacy_dependency.png", "Plate tectonic v1 legacy dependency", "Plate Terrain 11 score showing remaining weak fallback/texture dependency after plate-owned foundation replacement. High values are targets for future plate refinement.", (74, 116, 96), (230, 166, 70), ocean_color=None)
        _gradient_map(plate_hydro, "plate_tectonic_v1_hydrology_readiness.png", "Plate tectonic v1 hydrology readiness", "Native plate-mode pre-hydrology readiness from plate rifts, basins, coastal plains, valleys, slopes, and terrain maturity. This helps decide when hydrology can consume plate fields directly.", (86, 78, 112), (80, 188, 230), ocean_color=(34, 78, 128))

        pclass = getattr(terrain, "plate_tectonic_problem_class", None)
        if pclass is not None:
            pcl = _fit_field(pclass, scale=1.0).astype(np.uint8)
            rgbpc = np.zeros((h, w, 3), dtype=np.uint8); rgbpc[:] = PLATE_FINAL_PROBLEM_COLORS[0]
            for code, color in PLATE_FINAL_PROBLEM_COLORS.items():
                rgbpc[pcl == code] = color
            items = [(PLATE_FINAL_PROBLEM_LABELS[k], PLATE_FINAL_PROBLEM_COLORS[k]) for k in sorted(PLATE_FINAL_PROBLEM_LABELS)]
            _save_rgb(root / "07_final" / "plate_tectonic_v1_problem_classes.png", rgbpc, title="Plate tectonic v1 problem classes", description="Plate Terrain 11 classified QA map. It separates remaining legacy dependency, weak plate boundaries, weak hydrology readiness, shelf/active-margin conflicts, and ocean-floor underexpression.", items=items, stats=final_metrics, terrain=terrain, stride=stride)

    # Problem class map for quick triage before the terrain-wide revisit.
    problem = np.zeros((h, w), dtype=np.uint8)
    problem[flat_risk > 0.58] = 1
    problem[land_ds & (drainage < 0.28)] = 2
    problem[coast_ds & (coast_complexity_local < 0.22)] = 3
    problem[land_ds & (quality < 0.34)] = 4
    problem[np.abs(relief_delta) > 520] = 5
    rgbp = np.zeros((h, w, 3), dtype=np.uint8); rgbp[:] = (35, 82, 135); rgbp[land_ds] = (132, 146, 105)
    colors = {1: (215, 94, 65), 2: (118, 72, 180), 3: (240, 214, 70), 4: (170, 78, 74), 5: (66, 174, 126)}
    labels_items = [("ordinary ocean", (35,82,135)), ("ordinary land", (132,146,105))]
    for code, color in colors.items():
        rgbp[problem == code] = color
    labels_items += [("flat interior risk", colors[1]), ("weak drainage readiness", colors[2]), ("simple/smooth coast risk", colors[3]), ("low local terrain quality", colors[4]), ("large erosion/deposition change", colors[5])]
    _save_rgb(root / "07_final" / "terrain_problem_areas.png", rgbp, title="Terrain problem areas", description="Classified terrain QA triage map. Use this to decide which terrain systems to revisit before hydrology and climate tuning.", items=labels_items, stats=final_metrics, terrain=terrain, stride=stride)

    # Largest-landmass dominance map. Supercontinents are allowed, but this map
    # makes their spatial dominance explicit.
    rgbl = np.zeros((h, w, 3), dtype=np.uint8); rgbl[:] = (35, 82, 135); rgbl[land_ds] = (132, 146, 105)
    if labels is not None and np.max(labels) > 0:
        sizes = np.bincount(labels.ravel()); sizes[0] = 0
        largest_label = int(np.argmax(sizes))
        rgbl[labels == largest_label] = (224, 164, 72)
        rgbl[coast_ds] = (35, 35, 28)
    _save_rgb(root / "07_final" / "landmass_dominance_map.png", rgbl, title="Landmass dominance map", description="Highlights the largest landmass against all other land. Useful for checking supercontinent dominance and secondary landmass balance.", items=[("ocean", (35,82,135)), ("other land", (132,146,105)), ("largest landmass", (224,164,72)), ("coastline", (35,35,28))], stats=final_metrics, terrain=terrain, stride=stride)
