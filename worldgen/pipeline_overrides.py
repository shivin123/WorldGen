"""Manual override helpers for the staged WorldGen pipeline.

The staged workflow intentionally stores small JSON state files between major
phases.  Users may edit those files directly.  This module adds a safer middle
path: place targeted edits in ``config/stage_overrides.json`` and run
``python -m worldgen.pipeline apply-overrides``.

Only existing, well-scoped state/config fields are changed.  Unknown fields are
reported as warnings instead of silently inventing new model parameters.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from worldgen.pipeline_state import ensure_layout, now_stamp


class OverrideResult:
    def __init__(self) -> None:
        self.changes: list[str] = []
        self.warnings: list[str] = []
        self.touched_files: set[str] = set()

    @property
    def changed(self) -> bool:
        return bool(self.changes)

    def add_change(self, file_label: str, field_path: str, old: Any, new: Any) -> None:
        self.changes.append(f"{file_label}: {field_path}: {old!r} -> {new!r}")
        self.touched_files.add(file_label)

    def add_warning(self, message: str) -> None:
        self.warnings.append(message)


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json_if_changed(path: Path, original: dict[str, Any] | None, data: dict[str, Any], result: OverrideResult, label: str) -> None:
    if original != data:
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        result.touched_files.add(label)


def _usable_items(mapping: Any):
    if not isinstance(mapping, dict):
        return []
    return [(k, v) for k, v in mapping.items() if not str(k).startswith("_") and v is not None]


def _patch_mapping(target: dict[str, Any], patch: dict[str, Any], result: OverrideResult, file_label: str, prefix: str = "") -> bool:
    changed = False
    for key, value in _usable_items(patch):
        field_path = f"{prefix}.{key}" if prefix else key
        if key not in target:
            result.add_warning(f"{file_label}: ignored unknown field '{field_path}'.")
            continue
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            changed = _patch_mapping(target[key], value, result, file_label, field_path) or changed
            continue
        old = target.get(key)
        if old != value:
            target[key] = value
            result.add_change(file_label, field_path, old, value)
            changed = True
    return changed


def _patch_first_main_planet(system_state: dict[str, Any], patch: dict[str, Any], result: OverrideResult) -> bool:
    planets = system_state.get("planets", [])
    main_indices = [i for i, p in enumerate(planets) if p.get("is_main_planet")]
    if len(main_indices) != 1:
        result.add_warning(f"state/01_solar_system.json: expected exactly one main planet, found {len(main_indices)}; main_planet overrides skipped.")
        return False
    main = planets[main_indices[0]]
    return _patch_mapping(main, patch, result, "state/01_solar_system.json", "main_planet")


def _patch_named_planets(system_state: dict[str, Any], patch: dict[str, Any], result: OverrideResult) -> bool:
    if not isinstance(patch, dict):
        result.add_warning("planet_overrides must be an object keyed by planet name.")
        return False
    changed = False
    planets = system_state.get("planets", [])
    by_name = {str(p.get("name")): p for p in planets if p.get("name") is not None}
    for name, planet_patch in _usable_items(patch):
        planet = by_name.get(str(name))
        if planet is None:
            result.add_warning(f"state/01_solar_system.json: no planet named '{name}' for planet_overrides.")
            continue
        if isinstance(planet_patch, dict):
            changed = _patch_mapping(planet, planet_patch, result, "state/01_solar_system.json", f"planet_overrides.{name}") or changed
    return changed


def _patch_config(config_state: dict[str, Any], overrides: dict[str, Any], result: OverrideResult) -> bool:
    changed = False
    # Direct config section: {"config": {"seed": ..., "planet_profile": {...}}}
    config_patch = overrides.get("config") if isinstance(overrides.get("config"), dict) else {}
    changed = _patch_mapping(config_state, config_patch, result, "config/resolved_config.json", "config") or changed

    # These top-level sections are also accepted because the placeholder file is
    # deliberately simple and editable.
    for section in ("system", "planet_profile"):
        patch = overrides.get(section)
        if isinstance(patch, dict) and section in config_state:
            changed = _patch_mapping(config_state[section], patch, result, "config/resolved_config.json", section) or changed
    return changed


def apply_stage_overrides(output_dir: str | Path) -> OverrideResult:
    """Apply ``config/stage_overrides.json`` to existing staged state.

    Supported sections:
      * ``star`` -> generated star in state/01_solar_system.json and config star defaults
      * ``main_planet`` -> selected planet in state/01_solar_system.json
      * ``planet_overrides`` -> named planets in state/01_solar_system.json
      * ``planet_physics`` -> nested rotation/atmosphere/hydrosphere/geology in state/02_planet_physics.json
      * ``system`` / ``planet_profile`` / ``config`` -> config/resolved_config.json

    Terrain override keys are recorded as Stage 3 review controls. Some are
    currently used for diagnostics/workflow and will be progressively wired
    into the terrain generator as the terrain model is refactored.
    """
    output_dir = ensure_layout(output_dir)
    result = OverrideResult()
    override_path = output_dir / "config" / "stage_overrides.json"
    overrides = _load_json(override_path)
    if overrides is None:
        result.add_warning(f"No override file found at {override_path}.")
        return result

    config_path = output_dir / "config" / "resolved_config.json"
    config_original = _load_json(config_path)
    if config_original is not None:
        config_state = json.loads(json.dumps(config_original))
        if _patch_config(config_state, overrides, result):
            _write_json_if_changed(config_path, config_original, config_state, result, "config/resolved_config.json")

    system_path = output_dir / "state" / "01_solar_system.json"
    system_original = _load_json(system_path)
    if system_original is not None:
        system_state = json.loads(json.dumps(system_original))
        changed = False
        if isinstance(overrides.get("star"), dict):
            changed = _patch_mapping(system_state.get("star", {}), overrides["star"], result, "state/01_solar_system.json", "star") or changed
            if config_original is not None:
                # Keep star defaults in resolved config in sync for future reruns.
                config_state = _load_json(config_path) or {}
                if "star" in config_state:
                    cfg_orig = json.loads(json.dumps(config_state))
                    if _patch_mapping(config_state["star"], overrides["star"], result, "config/resolved_config.json", "star"):
                        _write_json_if_changed(config_path, cfg_orig, config_state, result, "config/resolved_config.json")
        if isinstance(overrides.get("main_planet"), dict):
            changed = _patch_first_main_planet(system_state, overrides["main_planet"], result) or changed
        if isinstance(overrides.get("planet_overrides"), dict):
            changed = _patch_named_planets(system_state, overrides["planet_overrides"], result) or changed
        if changed:
            _write_json_if_changed(system_path, system_original, system_state, result, "state/01_solar_system.json")
    elif any(isinstance(overrides.get(section), dict) and overrides.get(section) for section in ("star", "main_planet", "planet_overrides")):
        result.add_warning("Solar-system overrides were provided, but state/01_solar_system.json does not exist yet.")

    physics_path = output_dir / "state" / "02_planet_physics.json"
    physics_original = _load_json(physics_path)
    physics_patch = overrides.get("planet_physics")
    if physics_original is not None and isinstance(physics_patch, dict):
        physics_state = json.loads(json.dumps(physics_original))
        changed = False
        for section in ("rotation", "atmosphere", "hydrosphere", "geology"):
            if isinstance(physics_patch.get(section), dict) and section in physics_state:
                changed = _patch_mapping(physics_state[section], physics_patch[section], result, "state/02_planet_physics.json", f"planet_physics.{section}") or changed
        if changed:
            _write_json_if_changed(physics_path, physics_original, physics_state, result, "state/02_planet_physics.json")
    elif isinstance(physics_patch, dict) and physics_patch:
        result.add_warning("planet_physics overrides were provided, but state/02_planet_physics.json does not exist yet.")

    terrain_patch = overrides.get("terrain")
    if isinstance(terrain_patch, dict) and any(v is not None for k, v in terrain_patch.items() if not str(k).startswith("_")):
        result.add_warning("terrain overrides were recorded as Stage 3 review controls. They affect derived terrain diagnostics/workflow now; deeper direct terrain synthesis wiring will happen during the terrain model refactor.")

    report = {
        "schema_version": 1,
        "applied_at": now_stamp(),
        "changed": result.changed,
        "changes": result.changes,
        "warnings": result.warnings,
        "touched_files": sorted(result.touched_files),
    }
    report_path = output_dir / "diagnostics" / "override_application_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return result
