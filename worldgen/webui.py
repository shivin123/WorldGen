"""Local browser UI for staged WorldGen runs.

Run with:

    python -m worldgen.webui

Then open http://127.0.0.1:8765/.

The UI is intentionally lightweight and dependency-free.  It wraps the staged
pipeline command, shows state/status/diagnostics, lets users edit
config/stage_overrides.json, and starts long pipeline actions as background
subprocesses with log files under diagnostics/webui_jobs/.  Update61 fixes globe drag direction, adds globe drag-mode controls, and adds additional globe UI polish.
"""

from __future__ import annotations

import argparse
import csv
import html
import io
import json
import math
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import uuid
import urllib.parse
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from worldgen.output.map_registry import MAP_REGISTRY, MAP_REGISTRY_INDEX

from worldgen.pipeline_state import (
    STAGE_ORDER,
    TERRAIN_SUBPHASES,
    ensure_layout,
    terrain_subphase_marker_path,
    normalize_stage,
    read_manifest,
    status_detail_rows,
    unapplied_override_reason,
    write_status_report,
)

DEFAULT_PORT = 8765
RECENT_RUNS_FILE = Path.home() / ".worldgen_recent_runs.json"


@dataclass
class WebJob:
    id: str
    label: str
    command: list[str]
    output_dir: Path | None
    log_path: Path
    created_at: str
    status: str = "queued"
    returncode: int | None = None
    started_at: str | None = None
    finished_at: str | None = None
    process: subprocess.Popen[str] | None = field(default=None, repr=False)
    cancel_requested: bool = False


JOBS: dict[str, WebJob] = {}
JOBS_LOCK = threading.Lock()


def now_stamp() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _safe_text(value: Any) -> str:
    return html.escape("" if value is None else str(value), quote=True)


def _read_json(path: Path, default: Any) -> Any:
    try:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        pass
    return default


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def load_recent_runs() -> list[str]:
    data = _read_json(RECENT_RUNS_FILE, [])
    return [str(item) for item in data if isinstance(item, str)]


def _save_recent_runs(runs: list[str]) -> None:
    cleaned: list[str] = []
    seen: set[str] = set()
    for item in runs:
        value = str(Path(item).expanduser())
        if value in seen:
            continue
        seen.add(value)
        cleaned.append(value)
    _write_json(RECENT_RUNS_FILE, cleaned[:200])


def remember_run(path: str | Path) -> None:
    run = str(Path(path).expanduser())
    recent = [item for item in load_recent_runs() if item != run]
    recent.insert(0, run)
    _save_recent_runs(recent[:200])


def forget_run(path: str | Path) -> None:
    run = str(Path(path).expanduser())
    _save_recent_runs([item for item in load_recent_runs() if item != run])


def purge_missing_recent_runs() -> int:
    recent = load_recent_runs()
    kept = [item for item in recent if Path(item).expanduser().exists()]
    _save_recent_runs(kept)
    return len(recent) - len(kept)


def _run_manifest_summary(output_dir: Path) -> dict[str, Any]:
    try:
        manifest = read_manifest(output_dir) if output_dir.exists() else {}
    except Exception:
        manifest = {}
    if not isinstance(manifest, dict):
        manifest = {}
    try:
        rows = status_detail_rows(output_dir) if output_dir.exists() else []
    except Exception:
        rows = []
    counts = {"complete": 0, "stale": 0, "missing": 0}
    first_incomplete = ""
    last_complete = ""
    for row in rows:
        status = str(row.get("status", "missing"))
        if status in counts:
            counts[status] += 1
        if not first_incomplete and status in {"stale", "missing"}:
            first_incomplete = str(row.get("stage", ""))
        if status == "complete":
            last_complete = str(row.get("stage", last_complete))
    return {
        "pipeline_version": manifest.get("pipeline_version", ""),
        "updated_at": manifest.get("updated_at", ""),
        "counts": counts,
        "first_incomplete": first_incomplete,
        "last_complete": last_complete,
    }


def _format_bytes(num_bytes: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(max(0, num_bytes))
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1024.0
    return f"{num_bytes} B"


def _dir_size_bytes(path: Path) -> int:
    total = 0
    try:
        for item in path.rglob("*"):
            try:
                if item.is_file():
                    total += item.stat().st_size
            except OSError:
                continue
    except OSError:
        return 0
    return total


def _run_management_rows(paths: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw in paths:
        output_dir = Path(raw).expanduser()
        exists = output_dir.exists()
        is_dir = output_dir.is_dir()
        summary = _run_manifest_summary(output_dir) if exists and is_dir else {
            "pipeline_version": "",
            "updated_at": "",
            "counts": {"complete": 0, "stale": 0, "missing": 0},
            "first_incomplete": "",
            "last_complete": "",
        }
        rows.append({
            "path": str(output_dir),
            "exists": exists,
            "is_dir": is_dir,
            "size_bytes": 0,
            **summary,
        })
    return rows


def _is_probable_run_dir(path: Path) -> bool:
    if not path.exists() or not path.is_dir():
        return False
    markers = [
        path / "state",
        path / "diagnostics",
        path / "config",
        path / "manifest.json",
    ]
    return any(marker.exists() for marker in markers)


def delete_run_directory(path: str | Path) -> tuple[bool, str]:
    output_dir = Path(path).expanduser().resolve()
    dangerous = {
        output_dir.anchor,
        str(Path.home().resolve()),
        str(Path.cwd().resolve()),
    }
    if str(output_dir) in dangerous:
        return False, f"Refusing to delete protected path: {output_dir}"
    if not output_dir.exists():
        forget_run(output_dir)
        return True, f"Run folder already missing; removed {output_dir} from recent runs."
    if not output_dir.is_dir():
        return False, f"Path is not a directory: {output_dir}"
    if not _is_probable_run_dir(output_dir):
        return False, f"Refusing to delete {output_dir} because it does not look like a WorldGen run directory."
    try:
        shutil.rmtree(output_dir)
    except Exception as exc:
        return False, f"Failed to delete {output_dir}: {exc}"
    forget_run(output_dir)
    return True, f"Deleted run directory {output_dir}"


def _parse_form(body: bytes) -> dict[str, list[str]]:
    return urllib.parse.parse_qs(body.decode("utf-8"), keep_blank_values=True)


def _first(form: dict[str, list[str]], key: str, default: str = "") -> str:
    values = form.get(key)
    if not values:
        return default
    return values[0]


def _checkbox(form: dict[str, list[str]], key: str) -> bool:
    return key in form


def _int_arg(form: dict[str, list[str]], key: str) -> str | None:
    value = _first(form, key).strip()
    if not value:
        return None
    try:
        return str(int(value))
    except ValueError:
        return None


def _float_form_value(form: dict[str, list[str]], key: str) -> float | None:
    value = _first(form, key).strip()
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _set_nested(mapping: dict[str, Any], path: list[str], value: Any) -> None:
    cur = mapping
    for key in path[:-1]:
        nxt = cur.get(key)
        if not isinstance(nxt, dict):
            nxt = {}
            cur[key] = nxt
        cur = nxt
    cur[path[-1]] = value


def _get_nested(mapping: dict[str, Any], path: list[str], default: Any = "") -> Any:
    cur: Any = mapping
    for key in path:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(key, default)
    return cur


def _stage_options(selected: str | None = None) -> str:
    out: list[str] = []
    for stage in STAGE_ORDER:
        sel = " selected" if stage == selected else ""
        out.append(f'<option value="{_safe_text(stage)}"{sel}>{_safe_text(stage)}</option>')
    return "\n".join(out)


def _job_dir(output_dir: Path | None) -> Path:
    if output_dir is not None:
        path = output_dir / "diagnostics" / "webui_jobs"
    else:
        path = Path.cwd() / "webui_jobs"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _job_record(job: WebJob) -> dict[str, Any]:
    return {
        "id": job.id,
        "label": job.label,
        "status": job.status,
        "created_at": job.created_at,
        "started_at": job.started_at,
        "finished_at": job.finished_at,
        "returncode": job.returncode,
        "cancel_requested": job.cancel_requested,
        "output_dir": str(job.output_dir) if job.output_dir is not None else None,
        "log_path": str(job.log_path),
        "command": job.command,
    }


def _persist_job(job: WebJob) -> None:
    """Persist job metadata in the run folder so the UI can group job history."""
    if job.output_dir is None:
        return
    index_path = _job_dir(job.output_dir) / "jobs_index.json"
    data = _read_json(index_path, [])
    if not isinstance(data, list):
        data = []
    record = _job_record(job)
    updated = False
    for idx, item in enumerate(data):
        if isinstance(item, dict) and item.get("id") == job.id:
            data[idx] = record
            updated = True
            break
    if not updated:
        data.append(record)
    _write_json(index_path, data[-500:])


def _historical_jobs_for_run(output_dir: Path) -> list[dict[str, Any]]:
    path = output_dir / "diagnostics" / "webui_jobs" / "jobs_index.json"
    data = _read_json(path, [])
    if not isinstance(data, list):
        return []
    return [item for item in data if isinstance(item, dict)]


def start_job(label: str, command: list[str], output_dir: Path | None) -> WebJob:
    if output_dir is not None:
        remember_run(output_dir)
    jid = uuid.uuid4().hex[:12]
    log_path = _job_dir(output_dir) / f"{jid}.log"
    job = WebJob(
        id=jid,
        label=label,
        command=command,
        output_dir=output_dir,
        log_path=log_path,
        created_at=now_stamp(),
    )
    with JOBS_LOCK:
        JOBS[jid] = job
    _persist_job(job)

    def runner() -> None:
        with JOBS_LOCK:
            job.status = "running"
            job.started_at = now_stamp()
        _persist_job(job)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w", encoding="utf-8", errors="replace") as log:
            log.write(f"WorldGen Web UI job {job.id}\n")
            log.write(f"Started: {job.started_at}\n")
            log.write("Command: " + " ".join(command) + "\n\n")
            log.flush()
            try:
                env = os.environ.copy()
                env.setdefault("PYTHONUNBUFFERED", "1")
                proc = subprocess.Popen(
                    command,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    text=True,
                    cwd=Path.cwd(),
                    env=env,
                )
                with JOBS_LOCK:
                    job.process = proc
                rc = proc.wait()
                with JOBS_LOCK:
                    job.returncode = rc
                    if job.cancel_requested or job.status == "cancelling":
                        job.status = "cancelled"
                    else:
                        job.status = "complete" if rc == 0 else "failed"
                    job.finished_at = now_stamp()
                _persist_job(job)
                log.write(f"\nFinished: {job.finished_at}\nReturn code: {rc}\n")
                if job.cancel_requested:
                    log.write("Job was cancelled from the Web UI. Previously completed stage files were left intact.\n")
            except Exception as exc:  # pragma: no cover - defensive UI guard.
                with JOBS_LOCK:
                    job.status = "failed"
                    job.returncode = -1
                    job.finished_at = now_stamp()
                _persist_job(job)
                log.write(f"\nWeb UI failed to start job: {exc}\n")

    threading.Thread(target=runner, daemon=True).start()
    return job


def _tail(path: Path, max_chars: int = 16000) -> str:
    try:
        data = path.read_text(encoding="utf-8", errors="replace")
        return data[-max_chars:]
    except Exception as exc:
        return f"Could not read log: {exc}"


def _available_maps(output_dir: Path) -> list[Path]:
    """Return curated top-level run maps.

    This intentionally stays small for legacy callers that expect only the
    primary output maps. The full Web UI inventory uses _all_image_maps() and
    _run_image_inventory() so diagnostics and regional maps are not hidden.
    """
    candidates: dict[str, Path] = {}
    for folder in (output_dir / "maps", output_dir):
        if folder.exists():
            for path in sorted(folder.glob("*.png")):
                if path.is_file():
                    candidates[path.name] = path
    return list(candidates.values())


def _terrain_region_maps(output_dir: Path) -> list[Path]:
    terrain_regions = output_dir / "terrain_regions"
    if not terrain_regions.exists():
        return []
    return sorted([p for p in terrain_regions.glob("*.png") if p.is_file()])


def _solar_system_state(output_dir: Path) -> dict[str, Any] | None:
    path = output_dir / "state" / "01_solar_system.json"
    data = _read_json(path, None)
    return data if isinstance(data, dict) else None

def _main_planet_state(output_dir: Path) -> dict[str, Any]:
    solar = _solar_system_state(output_dir) or {}
    planets = solar.get("planets", []) if isinstance(solar.get("planets"), list) else []
    for planet in planets:
        if isinstance(planet, dict) and planet.get("is_main_planet"):
            return planet
    return {}


def _planet_physics_state(output_dir: Path) -> dict[str, Any]:
    data = _read_json(output_dir / "state" / "02_planet_physics.json", {})
    return data if isinstance(data, dict) else {}


def _axial_tilt_degrees_for_run(output_dir: Path | None) -> float:
    if output_dir is None:
        return 0.0
    physics = _planet_physics_state(output_dir)
    rotation = physics.get("rotation", {}) if isinstance(physics.get("rotation"), dict) else {}
    for value in (rotation.get("axial_tilt_degrees"), physics.get("axial_tilt_degrees")):
        try:
            if value is not None:
                return float(value)
        except Exception:
            pass
    return 0.0


def _fmt_num(value: Any, digits: int = 3) -> str:
    try:
        return f"{float(value):,.{digits}f}"
    except Exception:
        return ""




def _fmt_compact(value: Any, digits: int = 2, suffix: str = "") -> str:
    text = _fmt_num(value, digits)
    return f"{text}{suffix}" if text else "—"


def _progress_bar(value: Any, min_v: float, max_v: float, label: str = "") -> str:
    try:
        f = float(value)
        pct = (f - min_v) / max(1e-9, max_v - min_v) * 100.0
        pct = max(0.0, min(100.0, pct))
        shown = _fmt_num(f, 2)
    except Exception:
        pct = 0.0
        shown = "—"
    return f"<div class='mini-bar' title='{_safe_text(label)}'><span style='width:{pct:.1f}%'></span></div><span class='muted small'>{_safe_text(shown)}</span>"


def _safe_float(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except Exception:
        return None


def _classification_pill(label: str, tone: str = "info") -> str:
    return f"<span class='class-pill { _safe_text(tone) }'>{_safe_text(label)}</span>"


def _metric_gauge(label: str, value: Any, *, unit: str = "", min_v: float = 0.0, max_v: float = 1.0, low: str = "low", mid: str = "earthlike", high: str = "high") -> str:
    f = _safe_float(value)
    if f is None:
        shown = "—"
        pct = 0.0
        band = "unknown"
    else:
        pct = max(0.0, min(100.0, (f - min_v) / max(1e-9, max_v - min_v) * 100.0))
        shown = f"{f:,.2f}".rstrip("0").rstrip(".") + (f" {unit}" if unit else "")
        band = low if pct < 33 else (mid if pct < 66 else high)
    return f"""
      <div class='visual-metric' title='{_safe_text(label)}: {_safe_text(shown)}'>
        <div class='metric-head'><b>{_safe_text(label)}</b><span>{_safe_text(shown)}</span></div>
        <div class='metric-track'><span style='width:{pct:.1f}%'></span></div>
        <small>{_safe_text(band)}</small>
      </div>
    """


def _hz_position_indicator(star: dict[str, Any], planet: dict[str, Any]) -> str:
    inner = _safe_float(star.get("habitable_zone_inner_au"))
    outer = _safe_float(star.get("habitable_zone_outer_au"))
    orbit = planet.get("orbit", {}) if isinstance(planet.get("orbit"), dict) else {}
    au = _safe_float(orbit.get("semi_major_axis_au"))
    if inner is None or outer is None or au is None or outer <= inner:
        return "<p class='muted small'>Habitable-zone position unavailable.</p>"
    low = max(0.01, inner * 0.55)
    high = max(outer * 1.7, au * 1.05, 0.1)
    pct = max(0.0, min(100.0, (au - low) / (high - low) * 100.0))
    h1 = max(0.0, min(100.0, (inner - low) / (high - low) * 100.0))
    h2 = max(0.0, min(100.0, (outer - low) / (high - low) * 100.0))
    status = "inside habitable zone" if inner <= au <= outer else ("inside inner edge" if au < inner else "outside outer edge")
    return f"""
      <div class='hz-strip' title='Orbit {au:.3f} AU; habitable zone {inner:.3f}–{outer:.3f} AU'>
        <div class='hz-band' style='left:{h1:.1f}%;width:{max(1.5,h2-h1):.1f}%'></div>
        <div class='hz-marker' style='left:{pct:.1f}%'></div>
      </div>
      <p class='muted small'>Main Planet orbit: {au:.3f} AU · {status}</p>
    """


def _planet_mix_html(planets: list[Any]) -> str:
    counts: dict[str, tuple[int, dict[str, str]]] = {}
    for planet in planets:
        if not isinstance(planet, dict):
            continue
        style = _planet_type_style(planet.get("planet_class"))
        key = style["group"]
        count, _ = counts.get(key, (0, style))
        counts[key] = (count + 1, style)
    if not counts:
        return "<p class='muted small'>No planet classes available yet.</p>"
    parts = []
    for group, (count, style) in sorted(counts.items()):
        parts.append(f"<span class='mix-chip' style='--planet-color:{_safe_text(style['color'])}'>{_safe_text(style['icon'])} {count}× {_safe_text(group)}</span>")
    return "<div class='planet-mix'>" + "".join(parts) + "</div>"


def _moon_summary_html(output_dir: Path) -> str:
    # The exact moon state has changed during development, so this is deliberately
    # tolerant.  It adds a visual indicator when any recognizable moon data exists
    # and otherwise explains that moon detail has not been generated/saved yet.
    solar = _solar_system_state(output_dir) or {}
    planets = solar.get("planets", []) if isinstance(solar.get("planets"), list) else []
    main = next((p for p in planets if isinstance(p, dict) and p.get("is_main_planet")), {})
    moons = []
    for key in ("moons", "moon", "major_moons"):
        val = main.get(key) if isinstance(main, dict) else None
        if isinstance(val, list):
            moons.extend([m for m in val if isinstance(m, dict)])
        elif isinstance(val, dict):
            moons.append(val)
    if not moons:
        return "<div class='moon-card'><div class='moon-icon'>☾</div><div><b>Moon system</b><span class='muted'>No major moon saved for the selected planet.</span></div></div>"
    row_parts = []
    for moon in moons[:5]:
        mass = _fmt_compact(moon.get('mass_earth'), 4, ' M⊕')
        if mass == '—':
            mass = _fmt_compact(moon.get('mass_moon'), 2, ' lunar masses')
        details = [mass]
        if moon.get('moon_origin'):
            details.append(_humanize_key(moon.get('moon_origin')))
        if moon.get('tidal_effect_level'):
            details.append(f"tides: {_humanize_key(moon.get('tidal_effect_level')).lower()}")
        if moon.get('axial_stability_effect'):
            details.append(f"stability: {_humanize_key(moon.get('axial_stability_effect')).lower()}")
        row_parts.append(f"<li>{_safe_text(moon.get('name','moon'))} · {_safe_text(' · '.join(details))}</li>")
    rows = "".join(row_parts)
    return f"<div class='moon-card'><div class='moon-icon'>☾</div><div><b>Moon system</b><span>{len(moons)} moon(s)</span><ul>{rows}</ul></div></div>"


def _slider_control(name: str, label: str, value: str, *, min_v: float, max_v: float, step: float, unit: str = "", help_text: str = "", suggested: str = "") -> str:
    val = value if value not in {"", None} else ""
    safe_val = _safe_text(val)
    safe_name = _safe_text(name)
    range_help = f"Suggested: {suggested}. " if suggested else ""
    range_help += f"Range shown: {min_v:g}–{max_v:g}{(' ' + unit) if unit else ''}."
    if help_text:
        range_help = help_text + " " + range_help
    return f"""
      <div class='slider-field visual-editor-field'>
        <label>{_safe_text(label)}</label>
        <div class='value-ruler'><span>{min_v:g}</span><span>{max_v:g}</span></div>
        <div class='row slider-row'>
          <input class='range' type='range' name='{safe_name}_range' min='{min_v:g}' max='{max_v:g}' step='{step:g}' value='{safe_val or min_v}' oninput="document.querySelector('[name={safe_name}]').value=this.value; this.closest('.visual-editor-field').querySelector('.live-value').textContent=this.value;">
          <input class='number' name='{safe_name}' value='{safe_val}' placeholder='auto' oninput="const r=this.parentElement.querySelector('.range'); if(this.value!=='') r.value=this.value; this.closest('.visual-editor-field').querySelector('.live-value').textContent=this.value||'auto';">
          <span class='live-value'>{safe_val or 'auto'}</span><span class='muted small'>{_safe_text(unit)}</span>
        </div>
        <p class='field-help'>{_safe_text(range_help)}</p>
      </div>
    """


def _humanize_key(value: Any) -> str:
    text = str(value or "").strip().replace("_", " ").replace("-", " ")
    return " ".join(part.capitalize() for part in text.split()) or "—"


def _option_tags(options: list[tuple[str, str]], selected: Any) -> str:
    selected_text = str(selected or "")
    tags = []
    for value, label in options:
        sel = " selected" if str(value) == selected_text else ""
        tags.append(f"<option value='{_safe_text(value)}'{sel}>{_safe_text(label)}</option>")
    return "".join(tags)


def _select_control(name: str, label: str, value: Any, options: list[tuple[str, str]], *, help_text: str = "") -> str:
    return f"""
      <div class='visual-editor-field'>
        <label>{_safe_text(label)}</label>
        <select name='{_safe_text(name)}'>{_option_tags(options, value)}</select>
        <p class='field-help'>{_safe_text(help_text)}</p>
      </div>
    """


SYSTEM_ARCHITECTURE_OPTIONS = [
    ("", "Random / weighted architecture"),
    ("compact_rocky_inner", "Compact rocky inner system"),
    ("solar_like_mixed", "Solar-like mixed system"),
    ("outer_giant_dominated", "Outer giant dominated"),
    ("low_mass_quiet", "Low-mass quiet system"),
    ("volatile_rich", "Volatile-rich system"),
    ("sparse_old", "Sparse old system"),
]

MAIN_PLANET_PREFERENCE_OPTIONS = [
    ("earthlike", "Earth-like"),
    ("dry_terrestrial", "Dry terrestrial"),
    ("oceanic", "Oceanic"),
    ("super_earth", "Super-Earth"),
    ("colder_world", "Colder world"),
    ("warmer_world", "Warmer world"),
]

MOON_STRENGTH_OPTIONS = [
    ("weak", "Weak tides"),
    ("moderate", "Moderate / Earth-like tides"),
    ("strong", "Strong tides"),
]

TERRAIN_MODE_OPTIONS = [
    ("plate_history_v4", "Plate history v4 — recommended conservative terrain model"),
    ("plate_history_v3", "Plate history v3 — stable fallback"),
    ("procedural_legacy", "Procedural legacy — archived baseline"),
    ("plate_tectonic_v1", "Plate tectonic v1 — plate-owned terrain"),
    ("plate_history_v1", "Plate history v1 — time-evolved plates"),
    ("plate_history_v2", "Plate history v2 — legacy experimental"),
]

EXPLAIN_FIELD = {
    "type": "Debris-belt type: the broad kind of asteroid/icy remnant structure inferred from the system architecture.",
    "activity": "Debris activity: how dynamically active the belt/reservoir is, used as context for impacts and volatile delivery.",
    "mode": "Giant-planet history mode: a simplified story for whether giant planets shield, stir, or deliver material.",
    "perturbation_level": "Perturbation level: how strongly giant planets are expected to disturb smaller bodies and the Main Planet neighborhood.",
    "resonance_flavor": "Resonance flavor: a qualitative estimate of orbital sculpting by giant planets; this is not a full N-body resonance calculation.",
    "architecture": "Architecture: the overall layout pattern used to place planets and bias system context.",
    "main_planet_preference": "Main Planet preference: the type of world the selector was asked to favor when scoring candidates.",
    "giant_planet_influence": "Giant-planet influence: estimated effect of gas/ice giants on impact history, volatile delivery, and long-term orbital calm.",
    "climate_stability_outlook": "Climate stability outlook: Stage 1 estimate based mainly on moon stabilization, stellar/orbital context, and giant-planet disturbance.",
    "volatile_delivery": "Volatile delivery: inferred supply of water/ices/volatiles from formation zone, composition, and debris/giant context.",
    "crustal_asymmetry_bias": "Crustal asymmetry bias: how likely later terrain should have uneven hemispheres, basins, or disrupted crustal provinces.",
    "moon_origin": "Moon origin: simplified formation story for the major moon.",
    "axial_stability_effect": "Axial stability effect: how much the moon is expected to stabilize the planet's axial tilt over long timescales.",
    "tidal_effect_level": "Tidal effect level: qualitative strength of lunar tides relative to the planet.",
    "tectonic_energy_bias": "Tectonic energy bias: broad expectation for internal heat and plate/mountain-building potential.",
    "impact_history": "Impact history: qualitative estimate of bombardment/scarring context from system architecture and giant planets.",
    "formation_zone": "Formation zone: where the Main Planet appears to have formed or been enriched relative to the habitable zone and snow line.",
    "rotation_class": "Rotation class: qualitative day-length band used by climate circulation later.",
    "coriolis_strength": "Coriolis strength: expected strength of rotational steering for winds and currents.",
    "seasonality_class": "Seasonality class: expected seasonal contrast from axial tilt.",
    "axial_stability_class": "Axial stability class: long-term tilt stability inferred from moon and system context.",
    "tidal_braking": "Tidal braking: qualitative slowing of rotation by the major moon or tides.",
    "retention_score": "Retention score: rough atmosphere-retention strength from gravity, escape velocity, and temperature.",
    "retention_class": "Retention class: qualitative ability to keep an atmosphere over geologic time.",
    "pressure_class": "Pressure class: whether the surface atmosphere is thin, moderate, or thick.",
    "co2_class": "CO₂ class: broad greenhouse-gas band for review, not a full atmospheric chemistry model.",
    "greenhouse_workload": "Greenhouse workload: how hard greenhouse warming is working to keep surface temperatures temperate.",
    "climate_risk": "Climate risk: high-level risk band for cold-edge, temperate, warm, or hot conditions.",
    "target_land_fraction": "Target land fraction: complement of the ocean target before terrain generation.",
    "waterworld_risk": "Waterworld risk: likelihood that terrain may become too ocean-dominated without exposed continents.",
    "dry_world_risk": "Dry-world risk: likelihood that exposed land and aridity dominate the final surface.",
    "sea_level_sensitivity": "Sea-level sensitivity: how much small terrain/sea-level changes may alter land exposure.",
    "continental_exposure_tendency": "Continental exposure tendency: how much land the terrain stage should try to expose.",
    "expected_coastline_complexity": "Expected coastline complexity: whether coastlines should be naturally intricate or need terrain help.",
    "tectonic_regime": "Tectonic regime: broad plate/lid behavior expected from heat, mass, age, and tides.",
    "orogenic_intensity": "Orogenic intensity: expected strength of mountain-building.",
    "rift_tendency": "Rift tendency: likelihood of rifts, spreading centers, or stretched crust.",
    "island_arc_tendency": "Island-arc tendency: likelihood of volcanic island arcs or arc-like chains.",
    "hotspot_tendency": "Hotspot tendency: likelihood of intraplate volcanic island chains or provinces.",
    "basin_formation_tendency": "Basin formation tendency: likelihood of large basins from impacts, crustal sagging, or tectonics.",
    "continental_fragmentation_tendency": "Continental fragmentation tendency: how strongly terrain should break up large landmasses.",
    "shelf_deposition_tendency": "Shelf/deposition tendency: likely strength of shelves, coastal plains, sediment, and deltas.",
    "crustal_contrast_strength": "Crustal contrast strength: expected difference between continental, oceanic, basin, and highland provinces.",
    "explanation": "Explanation: plain-language reason for the category values in this card.",
}

EXPLAIN_VALUE = {
    "architecture": {
        "compact_rocky_inner": "A tight inner rocky system. Good for many terrestrial planets, but orbital spacing can be cramped.",
        "solar_like_mixed": "A mixed layout with inner rocky worlds and outer giants/icy bodies; closest to a Solar-System-like template.",
        "outer_giant_dominated": "Outer giants strongly shape the system. Expect stronger perturbation and impact/volatile context.",
        "low_mass_quiet": "A calmer, lower-mass architecture with fewer disruptive giants and usually gentler dynamics.",
        "volatile_rich": "A system biased toward icy reservoirs and volatile delivery; useful for ocean-world candidates.",
        "sparse_old": "An older, sparse system. Usually dynamically quiet but potentially lower in geologic energy.",
    },
    "main_planet_preference": {
        "earthlike": "Scores for a balanced rocky world with moderate flux, gravity, and water fraction.",
        "dry_terrestrial": "Favors lower water/volatile fractions; useful for dry continents or Mars-like testing.",
        "oceanic": "Favors high water/volatile fractions; useful for ocean-world or archipelago-heavy testing.",
        "super_earth": "Favors larger rocky/super-Earth worlds, but gravity and scale should be reviewed carefully.",
        "colder_world": "Favors worlds toward the outer/cooler habitable-zone side.",
        "warmer_world": "Favors worlds toward the inner/warmer habitable-zone side.",
    },
    "type": {
        "asteroid_and_outer_icy_belts": "Both rocky asteroid-like and icy outer reservoirs are inferred; impacts and volatile supply may be elevated.",
        "inner_asteroid_belt": "A rocky belt is inferred between inner planets and the snow line.",
        "outer_icy_belt": "The dominant reservoir is beyond the snow line, mostly icy material.",
        "faint_outer_dust": "Only a weak remnant belt is inferred; impact delivery should be low.",
    },
    "activity": {
        "low": "Quiet belt/reservoir; fewer impacts and less volatile delivery.",
        "moderate": "Some active debris delivery without dominating the system.",
        "high": "Active scattering reservoir; later impact history and volatile supply should be reviewed.",
    },
    "mode": {
        "no_major_giant_perturbers": "No large gas/ice giants were generated, so giant-driven shielding/disruption is weak.",
        "protective_outer_giant_history": "Outer giants are present but mostly act as stable sculptors/shields.",
        "disruptive_outer_giant_history": "Outer giants likely stir debris and increase impact/volatile delivery.",
        "volatile_delivery_giant_assisted": "Giant planets help scatter icy material inward.",
        "quiet_outer_giant_architecture": "Giants exist but are not treated as violently migrating.",
    },
    "perturbation_level": {
        "weak": "Little giant-driven orbital disturbance expected.",
        "moderate": "Noticeable but not extreme giant-planet influence.",
        "strong": "Large or nearby giants may strongly disturb debris and planetary orbits.",
    },
    "resonance_flavor": {
        "weak": "Only weak orbital sculpting is implied.",
        "moderate": "Moderate orbital sculpting; belts and gaps may be structured.",
        "strong": "Strong sculpting; expect pronounced belts/gaps and disturbance context.",
    },
    "giant_planet_influence": {
        "weak": "Fewer giant-driven impacts and less volatile scattering.",
        "moderate": "Useful middle ground: some shielding/scattering without extreme instability.",
        "strong": "High-impact/volatile context; review climate and crustal assumptions.",
    },
    "climate_stability_outlook": {
        "favorable": "Stage 1 sees mostly stable inputs: suitable orbit/flux plus stabilizing moon or low perturbation.",
        "stable": "Generally stable, with no major Stage 1 warning dominating the outlook.",
        "moderate": "Usable, but one or more factors such as weak moon, edge flux, or perturbations need review.",
        "mixed": "Some stabilizing and destabilizing signals conflict; review before expensive stages.",
        "challenging": "Stage 1 context suggests climate stability may need manual tuning downstream.",
    },
    "volatile_delivery": {
        "dry": "Low volatile supply; expect drier worlds unless later hydrosphere overrides it.",
        "moderate": "Balanced volatile supply; good for mixed land/ocean worlds.",
        "enhanced": "More volatile supply than Earthlike baseline; can bias toward wetter climates.",
        "wet": "High volatile supply; ocean coverage may be high downstream.",
        "heavy": "Very strong volatile delivery; review for ocean-world behavior.",
    },
    "crustal_asymmetry_bias": {
        "low": "Later terrain can be comparatively balanced between hemispheres/provinces.",
        "medium": "Some asymmetry is expected from impacts or giant-planet context.",
        "high": "Later terrain should allow major hemispheric/basin/crustal contrasts.",
    },
    "moon_origin": {
        "giant_impact": "A large impact formed the moon; can imply crustal asymmetry and strong early heating.",
        "captured": "A captured moon; orbit/tides may be less Earthlike.",
        "co_accreted": "Moon formed alongside the planet; generally calmer origin context.",
    },
    "axial_stability_effect": {
        "low": "Axial tilt may wander more; seasons/climate stability deserve review.",
        "moderate": "Some stabilizing influence; a reasonable default.",
        "high": "Strong stabilizing influence on axial tilt.",
    },
}



EXPLAIN_VALUE.update({
    "rotation_class": {"fast": "Short day length; stronger wind/current deflection.", "earth_like": "Moderate day length similar enough for Earthlike circulation assumptions.", "slow": "Long day length; weaker Coriolis and broader circulation bands."},
    "coriolis_strength": {"strong": "Fast rotation should strongly bend winds and currents.", "moderate": "Balanced rotational steering.", "weak": "Slow rotation should produce broader climate bands and weaker steering."},
    "seasonality_class": {"low": "Low axial tilt, mild seasons.", "moderate": "Moderate axial tilt, familiar seasonal structure.", "high": "High axial tilt, strong seasonality."},
    "axial_stability_class": {"stable": "Moon/system context supports stable obliquity.", "moderate": "No extreme stabilizing or destabilizing signal.", "unstable": "Tilt may vary more; review climate assumptions."},
    "tidal_braking": {"none": "No major moon/tidal slowing signal.", "weak": "Tides only weakly slow rotation.", "moderate": "Earthlike tidal influence.", "strong": "Strong tides can slow rotation and shape coasts/oceans."},
    "pressure_class": {"thin": "Lower pressure, weaker heat redistribution and less atmospheric buffering.", "earthlike_to_moderate": "Broadly comfortable pressure band for Earthlike climates.", "thick": "High pressure, stronger heat redistribution and greenhouse buffering."},
    "greenhouse_workload": {"low": "Greenhouse warming is relatively weak.", "normal": "Greenhouse warming is doing normal temperate-world work.", "heavy_lifting": "Greenhouse warming is high; habitability may be fragile or tuned."},
    "climate_risk": {"cold_edge": "World is near the cold side of temperate assumptions.", "temperate": "Mean surface estimate is in a comfortable temperate band.", "warm": "Warm but not necessarily runaway; watch aridity/tropics.", "hot_risky": "Hot enough to deserve careful climate review."},
    "waterworld_risk": {"low": "Ocean target is unlikely to drown most land.", "moderate": "Ocean target is high enough to review land exposure.", "high": "Ocean target may create too little land unless terrain counters it."},
    "dry_world_risk": {"low": "Enough water for oceans or seas.", "moderate": "Dryness may be important regionally.", "high": "World may be land/dryness dominated."},
    "tectonic_regime": {"stagnant_lid_or_quiet": "Low heat; subdued plate motion and mountain building.", "weak_plate_tectonics": "Some tectonic activity but less vigorous than Earthlike.", "earth_like_plate_tectonics": "Good default for active continents, ranges, trenches, and rifts.", "active_mobile_lid": "Very active tectonics; expect rugged young terrain.", "volcanic_resurfacing": "Volcanism dominates and may overwrite older crust."},
})

EXPLAIN_DOWNSTREAM = {
    "architecture": "Downstream effect: biases planet spacing, giant/debris context, volatile delivery, and the starting assumptions passed into planet physics.",
    "main_planet_preference": "Downstream effect: changes which planet becomes the detailed world, so every later terrain, climate, hydrology, and biome stage inherits this choice.",
    "type": "Downstream effect: debris belts influence impact-history and volatile-delivery context, which later affects hydrosphere and geology expectations.",
    "activity": "Downstream effect: higher activity should increase impact/bombardment expectations and may raise volatile-delivery variability.",
    "mode": "Downstream effect: the giant-planet story influences impact history, volatile delivery, orbital calm, and climate-stability warnings.",
    "perturbation_level": "Downstream effect: stronger perturbation can raise impact history, reduce orbital calm, and increase review warnings for climate stability.",
    "resonance_flavor": "Downstream effect: stronger resonance flavor can justify belts, gaps, delivery pathways, and disturbed-system diagnostics later.",
    "giant_planet_influence": "Downstream effect: affects volatile delivery, impact history, and the stability context passed into planet physics.",
    "climate_stability_outlook": "Downstream effect: should influence rotation/tilt confidence, climate warnings, and whether later climate maps should be treated as stable or fragile.",
    "volatile_delivery": "Downstream effect: strongly affects ocean target, water-vapor feedback, dry/ocean-world risk, and later hydrology assumptions.",
    "crustal_asymmetry_bias": "Downstream effect: terrain should use this to allow or suppress hemispheric imbalance, dominant continents, uneven basins, and irregular crustal provinces.",
    "moon_origin": "Downstream effect: moon origin affects tidal strength, axial stability, impact context, and potentially coastal/tidal assumptions.",
    "axial_stability_effect": "Downstream effect: influences axial-tilt confidence, seasonality warnings, and long-term climate-stability interpretation.",
    "tidal_effect_level": "Downstream effect: affects rotation/tidal braking, coast/tide expectations, and sometimes internal-heating/geology context.",
    "tectonic_energy_bias": "Downstream effect: feeds internal heat, volcanism, mountain factor, rift/island-arc tendency, and terrain ruggedness.",
    "impact_history": "Downstream effect: should influence crater density, basin formation, rough ancient crust, and early volatile delivery context.",
    "formation_zone": "Downstream effect: affects volatile inventory, composition assumptions, and whether the planet is treated as dry, wet, or migrated/enriched.",
    "rotation_class": "Downstream effect: controls expected Coriolis strength and climate circulation width.",
    "coriolis_strength": "Downstream effect: later wind/current and precipitation models should use this to shape bands, storm tracks, and ocean currents.",
    "seasonality_class": "Downstream effect: controls seasonal temperature contrast and high-latitude climate behavior.",
    "axial_stability_class": "Downstream effect: tells climate review whether tilt-driven climate bands are likely stable over geologic time.",
    "tidal_braking": "Downstream effect: affects rotation period, climate circulation, tidal/coastal assumptions, and possibly tidal-heating context.",
    "retention_score": "Downstream effect: informs whether the atmosphere/pressure values are plausible or forced, and whether future climate should warn about atmospheric stability.",
    "retention_class": "Downstream effect: affects pressure plausibility, greenhouse confidence, and long-term surface habitability interpretation.",
    "pressure_class": "Downstream effect: pressure affects heat redistribution, evaporation, precipitation potential, and day/night temperature buffering.",
    "co2_class": "Downstream effect: CO₂ affects greenhouse warming, climate risk, aridity, and temperature-map interpretation.",
    "greenhouse_workload": "Downstream effect: heavy workload means climate may be tuned/fragile; climate maps should show warnings if other values do not support it.",
    "climate_risk": "Downstream effect: sets expectations for temperature maps, ice tendency, aridity, and climate warning severity.",
    "target_land_fraction": "Downstream effect: terrain sea level and continental exposure should aim toward this land/ocean balance.",
    "waterworld_risk": "Downstream effect: terrain should preserve enough relief/exposed continents if this risk is high, or warn about ocean-world outcomes.",
    "dry_world_risk": "Downstream effect: climate and hydrology should expect weaker humidity, fewer rivers, and stronger aridity if this is high.",
    "sea_level_sensitivity": "Downstream effect: terrain should show warnings when small elevation shifts could dramatically change land exposure.",
    "continental_exposure_tendency": "Downstream effect: terrain should bias landmass emergence and shelf exposure around this expectation.",
    "expected_coastline_complexity": "Downstream effect: terrain/coast generation should produce coastlines consistent with this complexity level.",
    "tectonic_regime": "Downstream effect: terrain should use this to shape mountain chains, rifts, trenches, basins, and volcanic provinces.",
    "orogenic_intensity": "Downstream effect: directly affects mountain range height, frequency, and ruggedness expectations.",
    "rift_tendency": "Downstream effect: terrain should create or suppress rifts, stretched crust, and breakup structures accordingly.",
    "island_arc_tendency": "Downstream effect: terrain should use this to tune volcanic arcs and avoid overusing neat arcs when tendency is low.",
    "hotspot_tendency": "Downstream effect: terrain should create or suppress intraplate volcanic island chains and hotspot provinces.",
    "basin_formation_tendency": "Downstream effect: terrain should use this for large basins, inland lowlands, impact basins, and sediment accommodation.",
    "continental_fragmentation_tendency": "Downstream effect: terrain should use this to break or preserve large landmasses and avoid unwanted single-supercontinent bias.",
    "shelf_deposition_tendency": "Downstream effect: terrain and hydrology should use this for shelves, coastal plains, deltas, and sedimentary lowlands.",
    "crustal_contrast_strength": "Downstream effect: terrain should use this to distinguish continents, ocean basins, shelves, highlands, and lowlands.",
    "explanation": "Downstream effect: this text should clarify why the generated value matters before expensive later stages are run.",
}


def _downstream_explanation(key: str) -> str:
    return EXPLAIN_DOWNSTREAM.get(str(key), "Downstream effect: this value is passed forward as context, so later generated stages should either use it directly or warn if their outputs contradict it.")


def _possible_values_html(key: str, current: Any) -> str:
    options = EXPLAIN_VALUE.get(str(key), {})
    if not options:
        return ""
    items = []
    current_s = str(current or "").strip()
    for option, explanation in options.items():
        marker = " <span class='pill'>current</span>" if str(option) == current_s else ""
        downstream = _downstream_explanation(str(key))
        items.append(f"<li><b>{_safe_text(_humanize_key(option))}</b>{marker}: {_safe_text(explanation)} <span class='muted'>{_safe_text(downstream)}</span></li>")
    return "<details class='possible-values'><summary>Other possible values</summary><ul>" + "".join(items) + "</ul></details>"


def _value_explanation(key: str, value: Any, mapping: dict[str, Any] | None = None) -> str:
    key_s = str(key or "")
    value_s = str(value or "").strip()
    direct = EXPLAIN_VALUE.get(key_s, {}).get(value_s)
    if key_s == "climate_stability_outlook" and mapping and not direct:
        direct = "This is derived from Stage 1 orbital, moon, and giant-planet context. Use the warnings and moon/giant influence cards to see what pushed it up or down."
    if not direct:
        direct = "This value should be reviewed as part of the stage context."
    downstream = _downstream_explanation(key_s)
    if "Downstream effect:" not in direct:
        return f"{direct} {downstream}"
    return direct


def _explainable_stat(key: str, value: Any, *, mapping: dict[str, Any] | None = None) -> str:
    label = _humanize_key(key)
    if isinstance(value, dict):
        value_text = ", ".join(f"{_humanize_key(k)}: {v}" for k, v in value.items())
    elif isinstance(value, list):
        value_text = ", ".join(str(v) for v in value[:6])
    else:
        value_text = str(value)
    field_help = EXPLAIN_FIELD.get(str(key), "")
    value_help = _value_explanation(str(key), value, mapping)
    possible_values = _possible_values_html(str(key), value)
    details = f"<details class='stat-help'><summary>Explain</summary>{'<p><b>Category:</b> ' + _safe_text(field_help) + '</p>' if field_help else ''}<p><b>Value:</b> {_safe_text(value_help)}</p>{possible_values}</details>"
    title = (field_help + (" " if field_help else "") + value_help).strip()
    return f"<div class='stat' title='{_safe_text(title)}'><b>{_safe_text(label)}</b><span class='stat-value'>{_safe_text(value_text)}</span>{details}</div>"


def _mapping_stat_items(mapping: dict[str, Any], keys: list[str] | None = None) -> str:
    if not isinstance(mapping, dict) or not mapping:
        return "<p class='muted small'>No structured values available yet.</p>"
    ordered = keys or list(mapping.keys())
    parts = []
    for key in ordered:
        if key not in mapping:
            continue
        parts.append(_explainable_stat(str(key), mapping.get(key), mapping=mapping))
    return "".join(parts) or "<p class='muted small'>No structured values available yet.</p>"


def _run_mode_label(mode: str) -> str:
    return {
        "generated": "Generated procedural world",
        "synthetic-earth": "Synthetic Earth calibration",
        "real-earth-terrain": "Real Earth terrain calibration",
    }.get(mode, mode)


def _unique_output_dir_name(base: str = "staged_web_run") -> str:
    """Return a default output directory name that is not already taken.

    The browser form uses this so a new run does not accidentally target an
    existing folder.  It deliberately returns a relative name, matching the old
    UI behavior while adding collision avoidance.
    """
    root = Path(base).expanduser()
    if not root.exists():
        return str(root)
    for idx in range(2, 10000):
        candidate = Path(f"{base}_{idx}").expanduser()
        if not candidate.exists():
            return str(candidate)
    return f"{base}_{int(time.time())}"


def _star_visual(stellar_class: Any, temperature_k: Any = None) -> str:
    """Return a compact SVG-like star class visual for the dashboard."""
    text = _safe_text(stellar_class or "G")
    letter = str(stellar_class or "G").strip().upper()[:1] or "G"
    colors = {
        "O": ("#93c5fd", "blue O star"),
        "B": ("#bfdbfe", "blue-white B star"),
        "A": ("#e0f2fe", "white A star"),
        "F": ("#fef9c3", "yellow-white F star"),
        "G": ("#facc15", "yellow G star"),
        "K": ("#fb923c", "orange K star"),
        "M": ("#ef4444", "red M star"),
    }
    color, desc = colors.get(letter, ("#facc15", "main-sequence star"))
    temp = _fmt_compact(temperature_k, 0, " K") if temperature_k is not None else ""
    return f"""
    <div class='star-visual' title='{_safe_text(desc)}'>
      <div class='star-disk' style='--star-color:{_safe_text(color)}'></div>
      <div><strong>{text}</strong><span>{_safe_text(desc)}</span><small>{_safe_text(temp)}</small></div>
    </div>
    """


def _planet_type_style(planet_class: Any) -> dict[str, str]:
    """Single source of truth for planet-type color/icon labels across the UI."""
    raw = str(planet_class or "planet").strip() or "planet"
    cls = raw.lower().replace(" ", "_").replace("-", "_")
    style = {"color": "#94a3b8", "icon": "●", "label": raw, "group": "Other planet"}
    if any(token in cls for token in ("gas_giant", "jovian", "gas")):
        style.update({"color": "#c084fc", "icon": "♃", "group": "Gas giant"})
    elif any(token in cls for token in ("ice_giant", "mini_neptune", "neptune")):
        style.update({"color": "#7dd3fc", "icon": "♆", "group": "Ice / mini-Neptune"})
    elif any(token in cls for token in ("super_earth", "super-earth")):
        style.update({"color": "#22c55e", "icon": "⊕", "group": "Super-Earth"})
    elif any(token in cls for token in ("rocky", "terrestrial", "earthlike", "earth_like")):
        style.update({"color": "#a3e635", "icon": "◉", "group": "Rocky / terrestrial"})
    elif any(token in cls for token in ("icy_dwarf", "dwarf", "ice")):
        style.update({"color": "#94a3b8", "icon": "◌", "group": "Icy / dwarf"})
    return style


def _planet_visual(planet_class: Any, is_main: bool = False) -> str:
    style = _planet_type_style(planet_class)
    ring = " main" if is_main else ""
    title = f"{style['label']} — {style['group']}"
    return (
        f"<span class='planet-dot{ring}' title='{_safe_text(title)}' "
        f"style='--planet-color:{_safe_text(style['color'])}'><span>{_safe_text(style['icon'])}</span></span>"
    )


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
    "Csa": "Hot-summer Mediterranean climate",
    "Csb": "Warm-summer Mediterranean climate",
    "Dfa": "Hot-summer humid continental climate",
    "Dfb": "Warm-summer humid continental climate",
    "Dfc": "Subarctic climate",
    "ET": "Tundra climate",
    "EF": "Ice cap climate",
}




def _clean_string_value(value: Any) -> str:
    if isinstance(value, bytes):
        try:
            return value.decode("utf-8")
        except Exception:
            return str(value)
    text = str(value)
    if text.startswith("b'") and text.endswith("'"):
        return text[2:-1]
    return text


def _koppen_label(code: Any) -> str:
    text = str(code)
    return f"{text} — {KOPPEN_FULL_NAMES.get(text, text)}"


def _image_dimensions(path: Path) -> tuple[int, int] | None:
    try:
        from PIL import Image

        with Image.open(path) as img:
            return int(img.width), int(img.height)
    except Exception:
        return None


def _planet_radius_earth_for_run(output_dir: Path | None) -> float | None:
    if output_dir is None:
        return None
    for candidate in (
        output_dir / "state" / "03_terrain_metadata.json",
        output_dir / "state" / "02_planet_physics.json",
    ):
        data = _read_json(candidate, {})
        if isinstance(data, dict):
            for key in ("planet_radius_earth", "radius_earth"):
                if key in data:
                    try:
                        return float(data[key])
                    except Exception:
                        pass
            profile = data.get("planet_profile")
            if isinstance(profile, dict) and "radius_earth" in profile:
                try:
                    return float(profile["radius_earth"])
                except Exception:
                    pass
    return None


def _map_scale_summary(path: Path, output_dir: Path | None) -> str:
    dims = _image_dimensions(path)
    if dims is None:
        return "Image size unavailable."
    width, height = dims
    parts = [f"{width:,} × {height:,} pixels"]
    radius_earth = _planet_radius_earth_for_run(output_dir)
    if radius_earth and width > 0 and height > 0:
        radius_km = radius_earth * 6371.0
        equator_km_per_pixel = (2.0 * math.pi * radius_km) / width
        ns_km_per_pixel = (math.pi * radius_km) / height
        parts.append(f"≈ {equator_km_per_pixel:,.2f} km/pixel at equator")
        parts.append(f"≈ {ns_km_per_pixel:,.2f} km/pixel north-south")
        if width / max(1, height) < 1.7 or width / max(1, height) > 2.3:
            parts.append("scale assumes this image spans the full equirectangular world")
    else:
        parts.append("real-size scale unavailable until planet/terrain state exists")
    return " · ".join(parts)



def _world_grid_dimensions(output_dir: Path | None) -> tuple[int, int] | None:
    if output_dir is None:
        return None
    for candidate in (
        output_dir / "state" / "03_terrain_metadata.json",
        output_dir / "state" / "04_climate_metadata.json",
        output_dir / "state" / "06_biomes_metadata.json",
    ):
        data = _read_json(candidate, {})
        if isinstance(data, dict) and "width" in data and "height" in data:
            try:
                return int(data["width"]), int(data["height"])
            except Exception:
                pass
    return None


def _legend_items_for_map(path: Path) -> list[tuple[str, str]]:
    """Return best-effort legend swatches for common generated map types."""
    name = path.name.lower()
    # Check specific diagnostic layers before generic "terrain" catches names like
    # main_planet_terrain_provinces.png.
    if "koppen" in name:
        try:
            from worldgen.visualization.system_plot import KOPPEN_COLORS
            return [(color, _koppen_label(code)) for code, color in KOPPEN_COLORS.items()]
        except Exception:
            return []
    if "biome" in name:
        try:
            from worldgen.visualization.system_plot import BIOME_COLORS
            return [(color, biome) for biome, color in BIOME_COLORS.items()]
        except Exception:
            return []
    if "ocean_basins" in name:
        return [("#1d4ed8", "ocean basin IDs"), ("#0f766e", "semi-isolated sea"), ("#64748b", "land / non-ocean")]
    if "ocean_current_paths" in name:
        return [("#2563eb", "equatorial westward current"), ("#22c55e", "equatorial countercurrent"), ("#f97316", "subtropical gyre interior"), ("#dc2626", "western-boundary warm current"), ("#38bdf8", "eastern-boundary cold current"), ("#7c3aed", "subpolar gyre"), ("#facc15", "coastal upwelling"), ("#94a3b8", "weak / blocked flow")]
    if "ocean_gyres" in name:
        return [("#2563eb", "equatorial flow"), ("#f97316", "subtropical gyre"), ("#7c3aed", "subpolar gyre"), ("#dc2626", "poleward warm branch"), ("#38bdf8", "equatorward cold branch")]
    if "ocean_current_heat" in name or "warm_current" in name or "cold_current" in name:
        return [("#2563eb", "cold anomaly / influence"), ("#e5e7eb", "neutral"), ("#dc2626", "warm anomaly / influence")]
    if "coastal_upwelling" in name:
        return [("#e5e7eb", "little upwelling"), ("#93c5fd", "moderate upwelling"), ("#1d4ed8", "strong upwelling")]
    if "coastal_desert" in name or "coastal_dryness" in name:
        return [("#e5e7eb", "weak dryness"), ("#fde68a", "moderate dryness"), ("#d97706", "strong coastal desert potential")]
    if "pressure_belts" in name:
        return [("#7dd3fc", "low pressure / convergence"), ("#facc15", "subtropical high"), ("#a78bfa", "subpolar low / storm belt"), ("#cbd5e1", "polar high")]
    if "pressure_" in name:
        return [("#2563eb", "lower pressure"), ("#e5e7eb", "near normal"), ("#dc2626", "higher pressure")]
    if "itcz" in name or "thermal_equator" in name:
        return [("#fef3c7", "weak convergence"), ("#fb923c", "seasonal ITCZ / thermal-equator influence"), ("#b91c1c", "strong convergence")]
    if "wind" in name:
        return [("#94a3b8", "weak/variable wind"), ("#38bdf8", "trade/moisture transport"), ("#f97316", "westerly/storm transport")]
    if "storm_track" in name or "frontal_moisture" in name:
        return [("#e0f2fe", "weak storm-track moisture"), ("#38bdf8", "moderate storm-track moisture"), ("#0f766e", "strong storm-track moisture")]
    if "trade_wind_moisture" in name:
        return [("#f8fafc", "weak trade moisture"), ("#93c5fd", "moderate trade moisture"), ("#1d4ed8", "strong trade moisture")]
    if "monsoon_moisture" in name:
        return [("#fef3c7", "weak monsoon"), ("#60a5fa", "moderate monsoon"), ("#14532d", "strong monsoon")]
    if "orographic" in name:
        return [("#f8fafc", "low lift"), ("#86efac", "moderate windward lift"), ("#166534", "strong windward lift")]
    if "aridity" in name:
        return [("#8c510a", "arid / dry"), ("#f6e8c3", "semi-arid"), ("#80cdc1", "humid"), ("#01665e", "very humid")]
    if "crust_type" in name or re.search(r"(^|_)crust(_|\.)", name):
        return [("#2457a6", "oceanic crust"), ("#9a6b2f", "continental crust"), ("#7f5530", "thickened/orogenic crust"), ("#564232", "rifted/transition crust")]
    if "coastline_margin" in name or "coast_style" in name:
        return [("#38bdf8", "shelf coast"), ("#22c55e", "deltaic/plain coast"), ("#f97316", "rugged/active coast"), ("#7c3aed", "volcanic/island-arc coast")]
    if "plate_boundaries" in name or "boundary" in name:
        return [("#111111", "plate boundary"), ("#e11d48", "convergent / collision"), ("#f97316", "subduction / arc"), ("#22c55e", "rift / divergent"), ("#38bdf8", "transform / shear")]
    if "tectonic_plate" in name or "plate" in name:
        return [("#38bdf8", "plate IDs / plate regions"), ("#111827", "boundary emphasis")]
    if "province" in name or "region" in name:
        return [("#94a3b8", "province / region classes"), ("#111827", "region boundaries")]
    if "hydrology" in name or "river" in name or "basin" in name or "drainage" in name:
        return [("#1d4ed8", "rivers"), ("#38bdf8", "lakes / inland water"), ("#0f766e", "wetlands / deltas"), ("#f8fafc", "snow / ice")]
    if "temperature" in name:
        return [("#2b55a1", "cold"), ("#8ed1f0", "cool"), ("#f7e08b", "warm"), ("#c2410c", "hot")]
    if "precipitation" in name or "rain" in name:
        return [("#f7fbff", "dry"), ("#9ecae1", "moderate"), ("#3182bd", "wet"), ("#08519c", "very wet")]
    if "terrain" in name or "elevation" in name:
        return [("#071a2f", "deep ocean"), ("#1e6091", "shallow ocean"), ("#6aa84f", "lowland"), ("#d9b36c", "upland"), ("#8b5e3c", "mountains"), ("#ffffff", "highest peaks")]
    return [("#64748b", "map-specific values"), ("#e5e7eb", "see hover/sidebar for exact interpretation")]

def _legend_html(path: Path) -> str:
    sidecar = path.with_suffix(".legend.json")
    if sidecar.exists():
        data = _read_json(sidecar, {})
        if isinstance(data, dict):
            title = _safe_text(data.get("title") or "Legend")
            desc = _safe_text(data.get("description") or "")
            rows = []
            for item in data.get("legend", []):
                if not isinstance(item, dict):
                    continue
                label = _safe_text(item.get("label") or item.get("name") or "")
                kind = str(item.get("kind") or "swatch")
                unit = _safe_text(item.get("unit") or "")
                min_v = item.get("min")
                max_v = item.get("max")
                count = item.get("count")
                detail_bits = []
                if min_v is not None or max_v is not None:
                    detail_bits.append(f"{_safe_text(min_v)}–{_safe_text(max_v)}{(' ' + unit) if unit else ''}")
                if count not in (None, "", 0):
                    detail_bits.append(f"{int(count):,} cells" if isinstance(count, int) else _safe_text(count))
                detail = f"<small>{' · '.join(detail_bits)}</small>" if detail_bits else ""
                if kind == "gradient" and item.get("colors"):
                    colors = [_safe_text(c) for c in item.get("colors", []) if c]
                    css = ",".join(colors)
                    rows.append(f"<div class='legend-item gradient-item' data-legend-label='{label.lower()}'><span class='gradient-swatch' style='background:linear-gradient(90deg,{css})'></span><span>{label}{detail}</span></div>")
                else:
                    color = _safe_text(item.get("color") or "#777777")
                    rows.append(f"<div class='legend-item' data-legend-label='{label.lower()}'><span class='swatch' style='background:{color}'></span><span>{label}{detail}</span></div>")
            stats = data.get("stats") if isinstance(data.get("stats"), dict) else {}
            scale = data.get("scale") if isinstance(data.get("scale"), dict) else {}
            stat_rows = []
            for key in ("width", "height", "land_fraction", "ocean_fraction", "min_elevation_m", "max_elevation_m", "mean_land_temp_c", "mean_land_precip_mm", "river_cell_count", "drainage_basin_count", "dominant_biome"):
                if key in stats:
                    stat_rows.append(f"<div class='legend-stat'><b>{_safe_text(_humanize_key(key))}</b><span>{_safe_text(stats.get(key))}</span></div>")
            scale_rows = []
            for key in ("kind", "projection", "data_width", "data_height", "source_stride", "exported_equator_km_per_pixel", "exported_north_south_km_per_pixel", "equator_km_per_pixel", "north_south_km_per_pixel"):
                if key in scale:
                    value = scale.get(key)
                    if isinstance(value, float):
                        value = f"{value:,.3f}"
                    scale_rows.append(f"<div class='legend-stat'><b>{_safe_text(_humanize_key(key))}</b><span>{_safe_text(value)}</span></div>")
            body = "".join(rows) or "<p class='muted small'>No legend entries were written for this map.</p>"
            stat_html = f"<h4>Stats</h4><div class='legend-stats'>{''.join(stat_rows)}</div>" if stat_rows else ""
            scale_html = f"<h4>Scale</h4><div class='legend-stats'>{''.join(scale_rows)}</div>" if scale_rows else ""
            desc_html = f"<p class='muted small'>{desc}</p>" if desc else ""
            return f"<h3>{title}</h3>{desc_html}{body}{stat_html}{scale_html}"
    items = _legend_items_for_map(path)
    if not items:
        return "<p class='muted small'>No built-in legend is available for this map yet.</p>"
    rows = "".join(f"<div class='legend-item' data-legend-label='{_safe_text(label.lower())}'><span class='swatch' style='background:{_safe_text(color)}'></span><span>{_safe_text(label)}</span></div>" for color, label in items)
    return f"<h3>Legend</h3>{rows}"


def _truthy_meta(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _registry_row_for_path(path: Path, output_dir: Path | None) -> dict[str, Any] | None:
    name = path.name
    rel = _rel_label(path, output_dir).replace("\\", "/") if output_dir is not None else name
    for row in MAP_REGISTRY:
        filename = str(row.get("filename", "") or "")
        pattern = str(row.get("pattern", "") or "")
        if filename and (filename == name or filename == rel):
            return row
        if pattern:
            # Keep this deliberately simple and dependency-free; it covers the
            # registry's glob-style families for monthly/terrain folders.
            try:
                import fnmatch
                if fnmatch.fnmatch(rel, pattern) or fnmatch.fnmatch(name, pattern):
                    return row
            except Exception:
                pass
    return None


def _is_hidden_map_path(path: Path, output_dir: Path | None) -> bool:
    row = _registry_row_for_path(path, output_dir)
    if row is None:
        return False
    return _truthy_meta(row.get("hidden")) or _truthy_meta(row.get("deprecated"))


def _map_display_title(path: Path, output_dir: Path | None) -> str:
    row = _registry_row_for_path(path, output_dir)
    if row is not None:
        filename = str(row.get("filename") or "")
        if filename:
            key = str(row.get("key") or path.stem)
            return _humanize_key(key.replace("main_planet_", "").replace("seasonal_v3_", ""))
    stem = path.stem
    stem = re.sub(r"^main_planet_", "", stem)
    return _humanize_key(stem)


def _planet_name_for_run(output_dir: Path | None) -> str:
    if output_dir is None:
        return ""
    system = _read_json(output_dir / "system.json", {})
    profile = system.get("main_planet_profile") if isinstance(system, dict) else None
    if isinstance(profile, dict):
        return str(profile.get("planet_name") or "")
    state = _read_json(output_dir / "state" / "02_planet_physics.json", {})
    if isinstance(state, dict):
        for key in ("planet_name", "name"):
            if state.get(key):
                return str(state.get(key))
    return ""


def _system_sidebar_html(output_dir: Path | None, path: Path) -> str:
    if output_dir is None or path.name.lower() not in {"system_orbits.png", "system_sizes.png", "main_planet_moon.png"}:
        return ""
    system = _read_json(output_dir / "system.json", {})
    if not isinstance(system, dict) or not system:
        return "<h3>System data</h3><p class='muted small'>System metadata not available. Rerun outputs to write system.json.</p>"
    star = system.get("star", {}) if isinstance(system.get("star"), dict) else {}
    planets = system.get("planets", []) if isinstance(system.get("planets"), list) else []
    star_rows = []
    for key in ("stellar_class", "mass_solar", "radius_solar", "luminosity_solar", "temperature_k", "age_gyr"):
        if key in star:
            star_rows.append(f"<div class='legend-stat'><b>{_safe_text(_humanize_key(key))}</b><span>{_safe_text(star.get(key))}</span></div>")
    planet_rows = []
    for idx, planet in enumerate([p for p in planets if isinstance(p, dict)], start=1):
        name = planet.get("name", f"Planet {idx}")
        main = " <span class='pill ok-pill'>main</span>" if planet.get("is_main_planet") else ""
        orbit = planet.get("orbit", {}) if isinstance(planet.get("orbit"), dict) else {}
        au = orbit.get("semi_major_axis_au", planet.get("semi_major_axis_au", ""))
        radius = planet.get("radius_earth", "")
        mass = planet.get("mass_earth", "")
        cls = planet.get("planet_class", "")
        planet_rows.append(f"<li><b>{_safe_text(idx)}. {_safe_text(name)}</b>{main}<br><span class='muted small'>{_safe_text(cls)} · orbit {_safe_text(au)} AU · radius {_safe_text(radius)} R⊕ · mass {_safe_text(mass)} M⊕</span></li>")
    return f"<h3>System data</h3><div class='legend-stats'>{''.join(star_rows)}</div><h4>Planets</h4><ol class='small map-related-list'>{''.join(planet_rows)}</ol>"


def _related_maps_html(row: dict[str, Any] | None, output_dir: Path | None) -> str:
    if row is None or output_dir is None:
        return ""
    raw = row.get("related_maps") or []
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
            raw = parsed if isinstance(parsed, list) else [raw]
        except Exception:
            raw = [part.strip() for part in raw.split(",") if part.strip()]
    links = []
    for key in raw:
        item = MAP_REGISTRY_INDEX.get(str(key))
        if not item:
            continue
        filename = str(item.get("filename") or "")
        pattern = str(item.get("pattern") or "")
        target = output_dir / filename if filename else None
        if target is not None and target.exists():
            url = _map_view_url(target, output_dir)
            links.append(f"<li><a href='{url}'>{_safe_text(_map_display_title(target, output_dir))}</a></li>")
        elif pattern:
            matches = sorted([p for p in output_dir.glob(pattern) if p.is_file()])
            if matches:
                url = _map_view_url(matches[0], output_dir)
                links.append(f"<li><a href='{url}'>{_safe_text(str(item.get('description') or key))}</a></li>")
    if not links:
        return ""
    return f"<h3>Related maps</h3><ul class='small map-related-list'>{''.join(links[:18])}</ul>"


def _map_context_sidebar_html(path: Path, output_dir: Path | None) -> str:
    """Compact sidebar context: identity, hover focus, related links.

    Update 10B moves long explanations below the map so the sidebar can stay
    useful while inspecting/zooming the raster.
    """
    row = _registry_row_for_path(path, output_dir)
    title = _map_display_title(path, output_dir)
    planet = _planet_name_for_run(output_dir)
    bits = [f"<h3>{_safe_text(title)}</h3>"]
    if planet:
        bits.append(f"<p class='muted small'><b>Planet:</b> {_safe_text(planet)}</p>")
    if row is not None:
        if _truthy_meta(row.get("deprecated")):
            bits.append("<p class='warn small'><b>Deprecated/hidden:</b> retained for old runs or advanced review.</p>")
        desc = str(row.get("description") or "")
        hover = str(row.get("hover_focus") or "")
        sidebar = str(row.get("sidebar_info") or "")
        if desc:
            bits.append(f"<p class='small'>{_safe_text(desc)}</p>")
        if hover:
            bits.append(f"<p class='help'><b>Hover focus:</b> {_safe_text(hover)}</p>")
        if sidebar:
            bits.append(f"<p class='muted small'><b>Group:</b> {_safe_text(sidebar)}</p>")
        bits.append(_related_maps_html(row, output_dir))
    else:
        bits.append("<p class='help'>Fallback viewer metadata: exact registry entry not found. Hover still shows active sampled color plus shared terrain/climate/hydrology context when state files are available.</p>")
    system_html = _system_sidebar_html(output_dir, path)
    if system_html:
        bits.append(system_html)
    return "".join(bits)


def _map_explanation_html(path: Path, output_dir: Path | None) -> str:
    row = _registry_row_for_path(path, output_dir)
    title = _map_display_title(path, output_dir)
    if row is None:
        return f"""
<section class='card map-explanation-card'>
  <h2>How to read this map</h2>
  <p>This image is not fully registered yet, so the viewer uses fallback color sampling plus any shared run state available for hover. Add it to the map registry if it should become a first-class diagnostic.</p>
</section>
"""
    desc = str(row.get("description") or "")
    interp = str(row.get("interpretation") or "")
    hover = str(row.get("hover_focus") or "")
    legend_hint = str(row.get("legend_hint") or "")
    related = _related_maps_html(row, output_dir)
    body = []
    if desc:
        body.append(f"<p>{_safe_text(desc)}</p>")
    if interp:
        body.append(f"<h3>Interpretation</h3><p>{_safe_text(interp)}</p>")
    if hover:
        body.append(f"<h3>Hover readout</h3><p>The active value for this map is <b>{_safe_text(hover)}</b>. Update 10B highlights this active value in the hover panel, then keeps shared context below it: coordinates, cell scale, elevation, land/water, temperature, rainfall, biome/Köppen, hydrology, plate/crust and basin data where available.</p>")
    if legend_hint:
        body.append(f"<h3>Legend note</h3><p>{_safe_text(legend_hint)}</p>")
    if _truthy_meta(row.get("deprecated")):
        body.append("<p class='warn'><b>Deprecated/hidden map:</b> this is retained for comparison or legacy diagnostics, but it should not drive the main review unless specifically needed.</p>")
    if related:
        body.append(f"<h3>Related diagnostics</h3>{related}")
    return f"""
<section class='card map-explanation-card'>
  <h2>How to read: {_safe_text(title)}</h2>
  {''.join(body) or '<p>No explanation has been added yet.</p>'}
</section>
"""

def _row_cell_areas_km2(output_dir: Path | None, world_h: int, world_w: int):
    radius = _planet_radius_earth_for_run(output_dir)
    if not radius or world_h <= 0 or world_w <= 0:
        return None
    import numpy as np
    radius_km = float(radius) * 6371.0
    dlon = 2.0 * math.pi / float(world_w)
    edges = np.linspace(math.pi / 2.0, -math.pi / 2.0, world_h + 1)
    # Area of a spherical lat/lon cell band: R^2 * dlon * |sin(phi_n)-sin(phi_s)|.
    band = (radius_km * radius_km) * dlon * np.abs(np.sin(edges[:-1]) - np.sin(edges[1:]))
    return band.astype(float)


def _array_area_stats(arr: Any, output_dir: Path | None, *, max_rows: int = 18, label_kind: str = "value") -> list[dict[str, Any]]:
    try:
        import numpy as np
        a = np.asarray(arr)
        if a.ndim < 2:
            return []
        h, w = a.shape[:2]
        areas = _row_cell_areas_km2(output_dir, h, w)
        uniq, counts = np.unique(a, return_counts=True)
        total_area = None
        rows = []
        if areas is not None:
            total_area = float(np.sum(areas) * w)
        for val, count in zip(uniq, counts):
            if isinstance(val, bytes):
                sval = val.decode("utf-8", errors="replace")
            else:
                sval = str(val)
            if sval in {"", "0", "0.0", "False"} and label_kind not in {"crust", "ocean_basin", "drainage_basin"}:
                # Keep water/ocean/zero values for basin/crust maps; skip blank string placeholders elsewhere.
                pass
            mask = a == val
            area = None
            if areas is not None:
                per_row = np.sum(mask, axis=1).astype(float) * areas
                area = float(np.sum(per_row))
            rows.append({"value": val, "label": sval, "count": int(count), "area_km2": area, "area_pct": None if area is None or not total_area else area / total_area * 100.0})
        rows.sort(key=lambda r: r["count"], reverse=True)
        return rows[:max_rows]
    except Exception:
        return []


def _format_area(area: float | None) -> str:
    if area is None:
        return "area unavailable"
    if area >= 1_000_000:
        return f"{area/1_000_000:.2f} million km²"
    return f"{area:,.0f} km²"


def _class_label_for_stats(kind: str, value: Any) -> str:
    try:
        iv = int(value)
    except Exception:
        return str(value)
    if kind == "crust":
        return _class_lookup("crust", iv)
    if kind == "shelf_zone":
        return _class_lookup("shelf_zone", iv)
    if kind == "ocean_path":
        return {0: "land/non-ocean", 1: "equatorial westward current", 2: "equatorial countercurrent", 3: "subtropical gyre interior", 4: "western-boundary warm current", 5: "eastern-boundary cold current", 6: "subpolar gyre", 7: "coastal upwelling", 8: "weak / blocked flow"}.get(iv, str(iv))
    if kind == "ocean_gyre":
        return {0: "land/non-ocean", 1: "equatorial current", 2: "subtropical gyre", 3: "subpolar gyre", 4: "polar current", 5: "poleward warm branch", 6: "equatorward cold branch", 7: "upwelling"}.get(iv, str(iv))
    if kind == "ocean_basin":
        return "land/non-ocean" if iv <= 0 else f"ocean basin {iv}"
    if kind == "drainage_basin":
        return "no basin/water" if iv <= 0 else f"drainage basin {iv}"
    if kind == "landmass_component":
        return "water/non-land" if iv <= 0 else f"landmass component {iv}"
    return str(value)


def _landmass_component_array(output_dir: Path | None):
    if output_dir is None:
        return None
    land = _cached_npz_array(output_dir, "03_terrain.npz", "is_land")
    if land is None:
        return None
    try:
        import numpy as np
        from scipy import ndimage
        key = (str((output_dir / "state" / "03_terrain.npz").resolve()), "computed_landmass_component_id", "landmass", (output_dir / "state" / "03_terrain.npz").stat().st_mtime)
        if key in _STATE_ARRAY_CACHE:
            return _STATE_ARRAY_CACHE[key]
        mask = np.asarray(land, dtype=bool)
        labels, count = ndimage.label(mask, structure=np.array([[0,1,0],[1,1,1],[0,1,0]], dtype=np.uint8))
        # Merge seam-crossing components so land wrapping across the map edge is one component.
        if count:
            parent = list(range(count + 1))
            def find(x):
                while parent[x] != x:
                    parent[x] = parent[parent[x]]
                    x = parent[x]
                return x
            def union(a,b):
                if a and b:
                    ra, rb = find(int(a)), find(int(b))
                    if ra != rb:
                        parent[rb] = ra
            for y in range(mask.shape[0]):
                union(labels[y,0], labels[y,mask.shape[1]-1])
            roots = np.zeros(count + 1, dtype=np.int32)
            for i in range(1, count + 1):
                roots[i] = find(i)
            root_grid = roots[labels]
            unique = [int(v) for v in np.unique(root_grid[mask]) if int(v) > 0]
            sizes = [(int(np.sum(root_grid == v)), v) for v in unique]
            remap = {root: idx + 1 for idx, (_size, root) in enumerate(sorted(sizes, reverse=True))}
            out = np.zeros_like(labels, dtype=np.int32)
            for root, idx in remap.items():
                out[root_grid == root] = idx
        else:
            out = labels.astype(np.int32)
        _STATE_ARRAY_CACHE[key] = out
        return out
    except Exception:
        return None

def _stats_array_for_map(path: Path, output_dir: Path | None) -> tuple[Any | None, str, str]:
    if output_dir is None:
        return None, "", ""
    name = path.name.lower()
    if "biome" in name:
        return _cached_npz_array(output_dir, "06_biomes.npz", "biome_classification"), "biome", "Biome area"
    if "koppen" in name:
        return _cached_npz_array(output_dir, "04_climate.npz", "koppen_classification"), "koppen", "Köppen area"
    if "crust_type" in name:
        return _cached_npz_array(output_dir, "03_terrain.npz", "crust_type"), "crust", "Crust-class area"
    if "ocean_basins" in name:
        return _cached_npz_array(output_dir, "04_climate_drivers.npz", "ocean_basin_id"), "ocean_basin", "Ocean basin area"
    if "ocean_current_paths" in name:
        return _cached_npz_array(output_dir, "04_climate_drivers.npz", "ocean_current_path_class"), "ocean_path", "Current-class area"
    if "ocean_gyres" in name:
        return _cached_npz_array(output_dir, "04_climate_drivers.npz", "ocean_gyre_class"), "ocean_gyre", "Gyre-class area"
    if "landmass_components" in name:
        return _landmass_component_array(output_dir), "landmass_component", "Landmass component area"
    if "drainage_basins" in name:
        return _cached_npz_array(output_dir, "05_hydrology.npz", "drainage_basin_id"), "drainage_basin", "Drainage basin area"
    if "shelf_zones" in name:
        return _cached_npz_array(output_dir, "03_terrain.npz", "terrain_shelf_zone_class"), "shelf_zone", "Shelf-zone area"
    return None, "", ""


def _ocean_basin_kind_label(output_dir: Path | None, basin_id: int) -> str:
    if output_dir is None or basin_id <= 0:
        return ""
    basin = _cached_npz_array(output_dir, "04_climate_drivers.npz", "ocean_basin_id")
    kind = _cached_npz_array(output_dir, "04_climate_drivers.npz", "ocean_basin_kind")
    if basin is None or kind is None:
        return "open ocean basin"
    try:
        import numpy as np
        mask = basin.astype(int) == int(basin_id)
        vals = kind[mask].astype(int)
        if vals.size == 0:
            return "open ocean basin"
        k = int(np.bincount(vals.clip(0, 2), minlength=3).argmax())
        return "enclosed sea/lake" if k == 2 else "open ocean basin"
    except Exception:
        return "open ocean basin"

def _map_area_stats_html(path: Path, output_dir: Path | None) -> str:
    arr, kind, title = _stats_array_for_map(path, output_dir)
    if arr is None:
        return ""
    rows = _array_area_stats(arr, output_dir, max_rows=20, label_kind=kind)
    if not rows:
        return ""
    body = []
    for row in rows:
        label = row["label"]
        if kind in {"crust", "shelf_zone", "ocean_path", "ocean_gyre", "ocean_basin", "drainage_basin", "landmass_component"}:
            label = _class_label_for_stats(kind, row["value"])
            if kind == "ocean_basin" and int(row["value"]) > 0:
                label = f"{_ocean_basin_kind_label(output_dir, int(row['value']))} {int(row['value'])}"
        area_txt = _format_area(row.get("area_km2"))
        pct = row.get("area_pct")
        pct_txt = "" if pct is None else f" · {pct:.2f}%"
        body.append(f"<tr><td>{_safe_text(label)}</td><td>{int(row['count']):,}</td><td>{_safe_text(area_txt)}{_safe_text(pct_txt)}</td></tr>")
    return f"<h3>{_safe_text(title)}</h3><p class='help'>Area uses latitude-aware spherical cell areas, so polar cells count less than equatorial cells.</p><div class='table-wrap'><table class='small'><tr><th>Value</th><th>Cells</th><th>Approx. area</th></tr>{''.join(body)}</table></div>"


def _map_focus_label(path: Path) -> str:
    name = path.name.lower()
    if "koppen" in name:
        return "Köppen class"
    if "biome" in name:
        return "Biome"
    if "temperature" in name:
        return "Temperature"
    if "precip" in name or "rain" in name:
        return "Rainfall"
    if "hydrology" in name or "river" in name or "basin" in name or "drainage" in name:
        return "Hydrology"
    if "plate" in name or "crust" in name or "province" in name:
        return "Tectonics / crust"
    if "terrain" in name or "elevation" in name:
        return "Elevation / landform"
    return "Map sample"


WORLD_MAP_NAME_HINTS = (
    "terrain", "elevation", "koppen", "biome", "precipitation", "rain", "temperature",
    "hydrology", "drainage", "basin", "river", "plate", "crust", "wind", "pressure",
    "climate", "landform", "island", "archipelago", "province", "region",
    "shelf", "slope", "rise", "rift", "volcanic", "orogeny", "suture", "ripple",
)
NON_WORLD_MAP_NAME_HINTS = ("system_", "moon", "orbit", "sizes", "solar")
_STATE_ARRAY_CACHE: dict[tuple[str, str, str, float], Any] = {}
_IMAGE_SAMPLE_CACHE: dict[tuple[str, float], Any] = {}


def _is_probable_world_map(path: Path) -> bool:
    name = path.name.lower()
    if any(token in name for token in NON_WORLD_MAP_NAME_HINTS):
        return False
    if "terrain_region" in name:
        # Regional crops have their own local extent; do not pretend they span the full world.
        return False
    return any(token in name for token in WORLD_MAP_NAME_HINTS)


def _map_data_geometry(output_dir: Path | None, image_path: Path) -> dict[str, Any]:
    """Return the image area that actually corresponds to the world raster.

    Current map PNGs are exported as map-only rasters with separate legend
    sidecars.  This function still understands legacy images with embedded
    legend bands so older runs keep lining up in the viewer.
    """
    dims = _image_dimensions(image_path) or (0, 0)
    img_w, img_h = dims
    world = _world_grid_dimensions(output_dir)
    out: dict[str, Any] = {
        "image_width": int(img_w),
        "image_height": int(img_h),
        "data_width": int(img_w),
        "data_height": int(img_h),
        "world_width": 0,
        "world_height": 0,
        "is_world_map": False,
        "note": "image coordinates only",
    }
    if output_dir is None or world is None or img_w <= 0 or img_h <= 0 or not _is_probable_world_map(image_path):
        return out
    world_w, world_h = world
    if world_w <= 0 or world_h <= 0:
        return out
    expected_h = int(round(float(img_w) * float(world_h) / float(world_w)))
    # Accept a map-data band when it fits inside the image.  A small tolerance
    # handles antialiasing/export padding; if the image is not close to a world
    # map, fall back to pixel-only behavior.
    if expected_h <= 0 or expected_h > img_h + 3:
        return out
    out.update({
        "data_width": int(img_w),
        "data_height": int(min(img_h, expected_h)),
        "world_width": int(world_w),
        "world_height": int(world_h),
        "is_world_map": True,
        "note": "map area",
    })
    return out




def _map_data_image_bytes(output_dir: Path | None, image_path: Path) -> bytes:
    """Return only the map-data band for world maps, excluding appended legend bands.

    The generated PNGs often include a bottom legend inside the same image.  That
    is useful for standalone viewing but bad for compare/overlay math.  This
    endpoint creates a temporary map-only image for browser comparison without
    changing the saved output file.
    """
    from PIL import Image

    with Image.open(image_path) as img:
        geom = _map_data_geometry(output_dir, image_path)
        data_w = int(geom.get("data_width") or img.width)
        data_h = int(geom.get("data_height") or img.height)
        if geom.get("is_world_map") and data_h > 0 and data_h < img.height:
            crop = img.crop((0, 0, min(img.width, data_w), min(img.height, data_h)))
        else:
            crop = img.copy()
        buf = io.BytesIO()
        # Preserve alpha when present.
        crop.save(buf, format="PNG")
        return buf.getvalue()


def _map_data_url(path: Path, output_dir: Path | None = None) -> str:
    params = {"path": str(path.resolve())}
    if output_dir is not None:
        params["output_dir"] = str(output_dir)
    return "/map-data-image?" + urllib.parse.urlencode(params)


def _cached_npz_array(output_dir: Path, filename: str, key: str):
    path = output_dir / "state" / filename
    if not path.exists():
        return None
    try:
        mtime = path.stat().st_mtime
    except Exception:
        mtime = 0.0
    cache_key = (str(path.resolve()), key, filename, mtime)
    if cache_key in _STATE_ARRAY_CACHE:
        return _STATE_ARRAY_CACHE[cache_key]
    try:
        import numpy as np
        with np.load(path, allow_pickle=True) as data:
            if key not in data.files:
                return None
            arr = data[key]
            # Keep a real ndarray in memory.  This makes repeated hover queries
            # fast; the local UI is intentionally a workstation tool.
            arr = arr.copy()
            _STATE_ARRAY_CACHE[cache_key] = arr
            # Simple guard against unbounded cache growth after many runs.
            if len(_STATE_ARRAY_CACHE) > 36:
                for old_key in list(_STATE_ARRAY_CACHE.keys())[:12]:
                    _STATE_ARRAY_CACHE.pop(old_key, None)
            return arr
    except Exception:
        return None


def _sample_image_rgb(path: Path, x: float, y: float) -> str | None:
    try:
        mtime = path.stat().st_mtime
        key = (str(path.resolve()), mtime)
        img = _IMAGE_SAMPLE_CACHE.get(key)
        if img is None:
            from PIL import Image
            img = Image.open(path).convert("RGBA")
            _IMAGE_SAMPLE_CACHE[key] = img
            if len(_IMAGE_SAMPLE_CACHE) > 8:
                for old_key in list(_IMAGE_SAMPLE_CACHE.keys())[:3]:
                    old = _IMAGE_SAMPLE_CACHE.pop(old_key, None)
                    try:
                        old.close()
                    except Exception:
                        pass
        ix = int(max(0, min(img.width - 1, math.floor(x))))
        iy = int(max(0, min(img.height - 1, math.floor(y))))
        r, g, b, a = img.getpixel((ix, iy))
        return f"rgba({r}, {g}, {b}, {a})"
    except Exception:
        return None


def _cell_value(arr: Any, row: int, col: int) -> Any:
    try:
        h, w = arr.shape[:2]
        rr = int(max(0, min(h - 1, row)))
        cc = int(max(0, min(w - 1, col)))
        return arr[rr, cc]
    except Exception:
        return None


def _class_lookup(kind: str, value: int) -> str:
    tables = {
        "crust": {
            1: "abyssal/generic oceanic crust", 2: "young oceanic/ridge zone", 3: "old oceanic crust",
            4: "trench/subduction trough", 5: "fracture/transform oceanic crust", 6: "seamount/oceanic plateau",
            7: "shallow submerged continental shelf", 8: "continental craton/core", 9: "continental interior/shield",
            10: "young orogenic belt", 11: "old suture/eroded orogen", 12: "rifted continental crust",
            13: "transitional/passive margin", 14: "sedimentary/foreland basin", 15: "accreted terrane/microcontinent",
            16: "continental volcanic arc", 17: "oceanic island arc", 18: "hotspot/oceanic island",
            19: "upper continental slope", 20: "continental rise", 21: "deep submerged continental margin",
        },
        "shelf_zone": {
            0: "background/unknown water", 1: "shallow continental shelf sea", 2: "shelf edge / upper continental slope",
            3: "continental rise", 4: "abyssal/open ocean", 5: "active trench suppression", 6: "land",
        },
        "plate_component": {
            0: "unchanged contiguous plate", 1: "small disconnected fragment reassigned", 2: "large disconnected fragment promoted to microplate",
        },
        "v4_topology": {
            0: "ordinary deformed plate", 1: "sliver-plate candidate", 2: "promoted microplate candidate",
            3: "reassigned disconnected fragment", 4: "promoted disconnected fragment", 5: "rift-cut corridor", 6: "native rift/sliver plate",
        },
        "v4_island_chain": {
            0: "open ocean / no island-chain support", 1: "pre-existing continent/large land",
            2: "volcanic arc chain support", 3: "ridge or seamount chain support",
            4: "hotspot/oceanic volcanic chain support", 5: "rift-margin/narrow-sea island support",
            6: "pre-existing island retained", 7: "new v4 volcanic island",
        },
        "v4_boundary_network": {
            0: "no active v4 network", 1: "convergent/orogenic boundary", 2: "divergent/rift boundary",
            3: "transform/shear boundary", 4: "trench/subduction boundary", 5: "volcanic/arc boundary",
            6: "complex or triple-junction cell", 7: "microplate/sliver boundary",
            8: "rift-cut corridor away from final boundary", 9: "new v4 volcanic-island context",
        },
        "v4_orogen_network": {
            0: "no v4 branch/orogen class", 1: "primary convergent orogen", 2: "oblique mountain branch",
            3: "volcanic-arc mountain branch", 4: "foreland/sedimentary flank",
            5: "rifted shoulder/highland", 6: "complex-junction orogen",
        },
        "v4_control_response": {
            0: "weak/no v4 control response", 1: "topology/boundary/mountain response",
            2: "volcanic island-chain response", 3: "rift-cut/gulf/basin response",
            4: "mixed two-control response", 5: "mixed topology + island + rift response",
        },
        "v4_landform_change": {
            0: "weak/no actual v4 terrain change", 1: "mountain/orogen branch uplift",
            2: "volcanic island-chain uplift", 3: "rift/gulf/basin lowering",
            4: "native sliver/microplate corridor", 5: "mixed v4 terrain response",
        },
        "island_origin": {
            0: "not island / continent", 1: "continental fragment / microcontinent", 2: "volcanic arc island",
            3: "hotspot/oceanic island", 4: "v4 volcanic island chain", 5: "depositional/coastal island",
        },
        "coast_style": {
            0: "not coast / background", 1: "passive/smooth margin", 2: "rugged/fjorded coast", 3: "rifted/gulfed margin",
            4: "active/subduction coast", 5: "volcanic/arc coast", 6: "depositional/deltaic coast",
        },
    }
    label = tables.get(kind, {}).get(int(value))
    return f"{int(value)} — {label}" if label else str(int(value))


def _field_x1000_hover(output_dir: Path, row: int, col: int, key: str, label: str, *, unit: str = "support") -> str | None:
    arr = _cached_npz_array(output_dir, "03_terrain.npz", key)
    if arr is None:
        return None
    v = _cell_value(arr, row, col)
    if v is None:
        return None
    try:
        value = float(v) / 1000.0
    except Exception:
        return None
    if unit == "lake_depth_m":
        return f"ACTIVE {label} {value * 4000.0:,.0f} m allowed depth"
    if unit == "depth_target":
        return f"ACTIVE {label} {value:.3f} normalized depth target"
    if unit == "signed_m":
        try:
            raw = float(v)
        except Exception:
            raw = value
        return f"ACTIVE {label} {raw:,.0f} m"
    return f"ACTIVE {label} {value:.3f}"


def _climate_driver_hover(output_dir: Path, row: int, col: int, key: str, label: str, *, divisor: float = 1000.0, unit: str = "index") -> str | None:
    arr = _cached_npz_array(output_dir, "04_climate_drivers.npz", key)
    if arr is None:
        # Some core climate rasters may be saved with the main climate state in
        # older/staged runs.  Try it before falling back to image color only.
        arr = _cached_npz_array(output_dir, "04_climate.npz", key)
    if arr is None:
        return None
    v = _cell_value(arr, row, col)
    if v is None:
        return None
    try:
        value = float(v) / float(divisor)
    except Exception:
        return None
    if unit:
        return f"ACTIVE {label} {value:.3f} {unit}"
    return f"ACTIVE {label} {value:.3f}"


def _active_map_hover_value(output_dir: Path | None, image_path: Path, row: int, col: int) -> str | None:
    if output_dir is None:
        return None
    name = image_path.name.lower()

    # Primary output maps get first-class active values, then the generic hover
    # context below adds elevation, rain, temperature, biome, hydrology, etc.
    if name == "main_planet_temperature.png":
        arr = _cached_npz_array(output_dir, "04_climate.npz", "annual_mean_temp_c_x10")
        v = _cell_value(arr, row, col) if arr is not None else None
        if v is not None:
            return f"ACTIVE temperature {float(v) / 10.0:.1f} °C"
    if name == "main_planet_precipitation.png":
        arr = _cached_npz_array(output_dir, "04_climate.npz", "annual_precip_mm")
        v = _cell_value(arr, row, col) if arr is not None else None
        if v is not None:
            return f"ACTIVE precipitation {int(v):,} mm/yr"
    if name == "main_planet_koppen.png":
        arr = _cached_npz_array(output_dir, "04_climate.npz", "koppen_classification")
        v = _cell_value(arr, row, col) if arr is not None else None
        if v is not None:
            return f"ACTIVE Köppen {_koppen_label(_clean_string_value(v))}"
    if name == "main_planet_biomes.png":
        arr = _cached_npz_array(output_dir, "06_biomes.npz", "biome_classification")
        v = _cell_value(arr, row, col) if arr is not None else None
        if v is not None:
            return f"ACTIVE biome {_clean_string_value(v)}"
    if "drainage_basins" in name:
        arr = _cached_npz_array(output_dir, "05_hydrology.npz", "drainage_basin_id")
        v = _cell_value(arr, row, col) if arr is not None else None
        if v is not None:
            return f"ACTIVE drainage basin {int(v)}"
    if "landmass_components" in name:
        arr = _landmass_component_array(output_dir)
        v = _cell_value(arr, row, col) if arr is not None else None
        if v is not None:
            return "ACTIVE water / non-land" if int(v) <= 0 else f"ACTIVE landmass component {int(v)}"
    if "ocean_basins" in name:
        arr = _cached_npz_array(output_dir, "04_climate_drivers.npz", "ocean_basin_id")
        kind_arr = _cached_npz_array(output_dir, "04_climate_drivers.npz", "ocean_basin_kind")
        v = _cell_value(arr, row, col) if arr is not None else None
        if v is not None:
            if int(v) <= 0:
                return "ACTIVE land / non-ocean"
            k = _cell_value(kind_arr, row, col) if kind_arr is not None else 1
            kind_label = "enclosed sea/lake" if int(k or 0) == 2 else "open ocean basin"
            return f"ACTIVE {kind_label} {int(v)}"
    if "ocean_current_paths" in name:
        arr = _cached_npz_array(output_dir, "04_climate_drivers.npz", "ocean_current_path_class")
        v = _cell_value(arr, row, col) if arr is not None else None
        labels = {0: "land/non-ocean", 1: "equatorial westward current", 2: "equatorial countercurrent", 3: "subtropical gyre interior", 4: "western-boundary warm current", 5: "eastern-boundary cold current", 6: "subpolar gyre", 7: "coastal upwelling", 8: "weak / blocked flow"}
        if v is not None:
            return f"ACTIVE ocean current path {labels.get(int(v), int(v))}"

    class_specs = [
        ("crust_type", "crust_type", "crust", "crust class"),
        ("shelf_zones", "terrain_shelf_zone_class", "shelf_zone", "shelf zone"),
        ("final_plate_components", "terrain_final_plate_component_class", "plate_component", "plate component"),
        ("v4_plate_topology", "terrain_v4_topology_class", "v4_topology", "v4 topology"),
        ("v4_island_chains", "terrain_v4_island_chain_class", "v4_island_chain", "v4 island-chain class"),
        ("v4_boundary_network", "terrain_v4_boundary_network_class", "v4_boundary_network", "v4 boundary-network class"),
        ("v4_orogen_network", "terrain_v4_orogen_network_class", "v4_orogen_network", "v4 orogen-network class"),
        ("v4_control_response", "terrain_v4_control_response_class", "v4_control_response", "v4 control-response class"),
        ("v4_landform_change", "terrain_v4_landform_change_class", "v4_landform_change", "v4 landform-change class"),
        ("islands_archipelago", "terrain_island_origin_class", "island_origin", "island origin"),
        ("coastline_margin_types", "terrain_coast_style_class", "coast_style", "coast style"),
    ]
    for token, key, table, label in class_specs:
        if token in name:
            arr = _cached_npz_array(output_dir, "03_terrain.npz", key)
            if arr is None:
                return None
            v = _cell_value(arr, row, col)
            if v is None:
                return None
            return f"ACTIVE {label} {_class_lookup(table, int(v))}"
    field_specs = [
        ("submerged_continental_crust", "terrain_submerged_continental_crust_x1000", "submerged continental crust"),
        ("continental_shelf_support", "terrain_continental_shelf_support_x1000", "continental shelf support"),
        ("shelf_depth_target", "terrain_shelf_depth_target_x1000", "shelf depth target", "depth_target"),
        ("lake_depth_limit", "terrain_lake_depth_limit_x1000", "lake depth limit", "lake_depth_m"),
        ("ripple_artifact_risk", "terrain_ripple_artifact_risk_x1000", "ripple artifact risk"),
        ("v4_boundary_deformation", "terrain_v4_boundary_deformation_x1000", "v4 boundary deformation"),
        ("v4_volcanic_island_support", "terrain_v4_volcanic_island_support_x1000", "v4 volcanic island support"),
        ("v4_rift_cut_support", "terrain_v4_rift_cut_support_x1000", "v4 rift-cut support"),
        ("v4_mountain_branch_support", "terrain_v4_mountain_branch_support_x1000", "v4 mountain-branch support"),
        ("v4_elevation_delta", "terrain_v4_elevation_delta_m", "v4 elevation delta", "signed_m"),
        ("erosion_deposition", "terrain_deposition_field_x1000", "deposition field"),
    ]
    for spec in field_specs:
        token, key, label = spec[:3]
        unit = spec[3] if len(spec) > 3 else "support"
        if token in name:
            return _field_x1000_hover(output_dir, row, col, key, label, unit=unit)

    # Monthly progression maps use grouped filenames rather than fixed tokens.
    m_month = re.match(r"main_planet_(temperature|precipitation)_month_(\d\d)\.png$", name)
    if m_month:
        kind = m_month.group(1)
        month = int(m_month.group(2))
        if kind == "temperature":
            return _climate_driver_hover(output_dir, row, col, f"monthly_temperature_{month:02d}_c_x10", f"month {month:02d} temperature", divisor=10.0, unit="°C")
        return _climate_driver_hover(output_dir, row, col, f"monthly_precipitation_{month:02d}_mm", f"month {month:02d} precipitation", divisor=1.0, unit="mm/month")

    if "circulation_zones" in name:
        arr = _cached_npz_array(output_dir, "04_climate_drivers.npz", "circulation_zone_class")
        if arr is not None:
            labels = {1: "equator", 2: "ITCZ / convergence", 3: "tropics", 4: "horse latitudes / subtropical high", 5: "westerly belt", 6: "polar front / storm track", 7: "polar cap"}
            v = _cell_value(arr, row, col)
            if v is not None:
                return f"ACTIVE circulation zone {labels.get(int(v), 'other')}"
    if "ocean_gyres" in name:
        arr = _cached_npz_array(output_dir, "04_climate_drivers.npz", "ocean_gyre_class")
        if arr is not None:
            labels = {0: "weak/no gyre", 1: "equatorial current", 2: "subtropical gyre", 3: "subpolar gyre", 4: "polar current", 5: "poleward warm branch", 6: "equatorward cold branch"}
            v = _cell_value(arr, row, col)
            if v is not None:
                return f"ACTIVE ocean gyre {labels.get(int(v), 'other')}"
    if "pressure_belts" in name:
        season = None
        for suffix in ("nh_summer", "equinox", "nh_winter"):
            if suffix in name:
                season = suffix
                break
        key = f"pressure_belt_{season}_class" if season else "pressure_belt_equinox_class"
        arr = _cached_npz_array(output_dir, "04_climate_drivers.npz", key)
        if arr is not None:
            labels = {1: "equatorial low / ITCZ", 2: "trade-wind belt", 3: "subtropical high", 4: "westerly belt", 5: "subpolar low / storm track", 6: "polar high"}
            v = _cell_value(arr, row, col)
            if v is not None:
                return f"ACTIVE pressure belt {labels.get(int(v), int(v))}"

    climate_specs = [
        ("trade_wind_moisture", "trade_wind_moisture_annual_x1000", "trade-wind moisture", 1000.0, "index"),
        ("monsoon_moisture", "monsoon_moisture_annual_x1000", "monsoon moisture", 1000.0, "index"),
        ("frontal_moisture", "frontal_moisture_annual_x1000", "frontal/storm-track moisture", 1000.0, "index"),
        ("orographic_precip_potential", "orographic_precip_potential_x1000", "orographic precipitation potential", 1000.0, "index"),
        ("coastal_dryness", "coastal_dryness_x1000", "coastal dryness", 1000.0, "index"),
        ("coastal_desert_potential", "coastal_desert_potential_x1000", "coastal desert potential", 1000.0, "index"),
        ("coastal_upwelling", "coastal_upwelling_x1000", "coastal upwelling", 1000.0, "index"),
        ("warm_current_influence", "warm_current_influence_x1000", "warm-current influence", 1000.0, "index"),
        ("cold_current_influence", "cold_current_influence_x1000", "cold-current influence", 1000.0, "index"),
        ("ocean_current_heat", "current_heat_annual_c_x10", "ocean-current heat anomaly", 10.0, "°C"),
        ("thermal_equator", "thermal_equator_annual_x1000", "thermal equator", 1000.0, "index"),
        ("storm_track_moisture", "storm_track_moisture_annual_x1000", "storm-track moisture", 1000.0, "index"),
        ("itcz_position", "itcz_annual_x1000", "ITCZ/convergence", 1000.0, "index"),
        ("pressure_nh_summer", "pressure_nh_summer_hpa_x10", "NH summer pressure", 10.0, "hPa"),
        ("pressure_equinox", "pressure_equinox_hpa_x10", "equinox pressure", 10.0, "hPa"),
        ("pressure_nh_winter", "pressure_nh_winter_hpa_x10", "NH winter pressure", 10.0, "hPa"),
        ("temperature_nh_summer", "temperature_nh_summer_c_x10", "NH summer temperature", 10.0, "°C"),
        ("temperature_equinox", "temperature_equinox_c_x10", "equinox temperature", 10.0, "°C"),
        ("temperature_nh_winter", "temperature_nh_winter_c_x10", "NH winter temperature", 10.0, "°C"),
        ("precipitation_nh_summer", "precipitation_nh_summer_mm", "NH summer precipitation", 1.0, "mm/year"),
        ("precipitation_equinox", "precipitation_equinox_mm", "equinox precipitation", 1.0, "mm/year"),
        ("precipitation_nh_winter", "precipitation_nh_winter_mm", "NH winter precipitation", 1.0, "mm/year"),
        ("moisture_nh_summer", "moisture_nh_summer_x1000", "NH summer moisture", 1000.0, "index"),
        ("moisture_equinox", "moisture_equinox_x1000", "equinox moisture", 1000.0, "index"),
        ("moisture_nh_winter", "moisture_nh_winter_x1000", "NH winter moisture", 1000.0, "index"),
        ("orographic_lift", "orographic_lift_annual_x1000", "orographic lift", 1000.0, "index"),
        ("rain_shadow_actual", "rain_shadow_annual_x1000", "rain shadow", 1000.0, "index"),
        ("rain_shadow", "rain_shadow_annual_x1000", "rain shadow", 1000.0, "index"),
        ("aridity_index", "aridity_index_x1000", "aridity index", 1000.0, "P/P_threshold"),
        ("lake_moisture_sources", "inland_water_source_x1000", "inland-water source cap", 1000.0, "index"),
        ("small_lake_neutral", "small_lake_neutral_buffer_x1000", "small-lake neutral buffer", 1000.0, "index"),
        ("wind_currents", "moisture_annual_x1000", "annual moisture under wind", 1000.0, "index"),
        ("moisture_transport", "moisture_annual_x1000", "annual moisture", 1000.0, "index"),
        ("ocean_currents", "current_heat_annual_c_x10", "ocean-current heat effect", 10.0, "°C"),
    ]
    for token, key, label, divisor, unit in climate_specs:
        if token in name:
            return _climate_driver_hover(output_dir, row, col, key, label, divisor=divisor, unit=unit)
    return None


def _cell_km_scale(output_dir: Path | None, lat_deg: float, world_w: int, world_h: int) -> tuple[float | None, float | None]:
    radius = _planet_radius_earth_for_run(output_dir)
    if not radius or world_w <= 0 or world_h <= 0:
        return None, None
    radius_km = float(radius) * 6371.0
    ew = (2.0 * math.pi * radius_km * max(0.0, math.cos(math.radians(lat_deg)))) / float(world_w)
    ns = (math.pi * radius_km) / float(world_h)
    return ew, ns


def _format_latlon(lat: float, lon: float) -> str:
    ns = "N" if lat >= 0 else "S"
    ew = "E" if lon >= 0 else "W"
    return f"{abs(lat):.2f}°{ns}, {abs(lon):.2f}°{ew}"

def _map_pixel_to_world(output_dir: Path | None, image_path: Path, x: float, y: float) -> tuple[int | None, int | None, str]:
    geom = _map_data_geometry(output_dir, image_path)
    img_w = float(geom.get("image_width") or 0)
    data_w = float(geom.get("data_width") or 0)
    data_h = float(geom.get("data_height") or 0)
    world_w = int(geom.get("world_width") or 0)
    world_h = int(geom.get("world_height") or 0)
    if not geom.get("is_world_map") or data_w <= 0 or data_h <= 0 or world_w <= 0 or world_h <= 0:
        return None, None, "image coordinates only"
    if y < 0 or x < 0 or x >= data_w or y >= data_h:
        return None, None, "legend / non-map area"
    col = int(max(0, min(world_w - 1, math.floor(float(x) * world_w / data_w))))
    row = int(max(0, min(world_h - 1, math.floor(float(y) * world_h / data_h))))
    return row, col, "map area"


def _map_hover_info(output_dir: Path | None, image_path: Path, x: float, y: float) -> dict[str, Any]:
    geom = _map_data_geometry(output_dir, image_path)
    row, col, area = _map_pixel_to_world(output_dir, image_path, x, y)
    payload: dict[str, Any] = {
        "x": int(round(x)),
        "y": int(round(y)),
        "area": area,
        "geometry": geom,
    }
    rgb = _sample_image_rgb(image_path, x, y)
    parts = [f"pixel {payload['x']:,}, {payload['y']:,}"]
    if rgb:
        payload["color"] = rgb
        parts.append(f"color {rgb}")
    if row is None or col is None or output_dir is None:
        if rgb:
            payload["active_value"] = f"sampled map color {rgb}"
            payload["context"] = parts + [area]
            payload["text"] = f"sampled map color {rgb} · " + " · ".join(parts + [area])
        else:
            parts.append(area)
            payload["context"] = parts
            payload["text"] = " · ".join(parts)
        return payload

    world_w = int(geom.get("world_width") or 0)
    world_h = int(geom.get("world_height") or 0)
    lon = (col + 0.5) / max(1, world_w) * 360.0 - 180.0
    lat = 90.0 - (row + 0.5) / max(1, world_h) * 180.0
    ew_km, ns_km = _cell_km_scale(output_dir, lat, world_w, world_h)
    payload.update({
        "row": row,
        "col": col,
        "lon": round(lon, 5),
        "lat": round(lat, 5),
        "ew_km_per_cell": None if ew_km is None else round(ew_km, 4),
        "ns_km_per_cell": None if ns_km is None else round(ns_km, 4),
    })
    context_parts = [f"cell r{row:,} c{col:,}", _format_latlon(lat, lon)]
    active_value = _active_map_hover_value(output_dir, image_path, row, col)
    if active_value:
        active_value = str(active_value).replace("ACTIVE ", "", 1)
        payload["active_value"] = active_value
    elif rgb:
        active_value = f"sampled map color {rgb}"
        payload["active_value"] = active_value
    parts = context_parts
    if rgb:
        parts.append(f"color {rgb}")
    if ew_km is not None and ns_km is not None:
        parts.append(f"cell scale ≈ {ew_km:.2f} km E-W × {ns_km:.2f} km N-S")

    # Show broad state info for any world-map image, not just maps whose
    # filename matches a specific layer.  This makes hover useful on composites,
    # diagnostics, and newly added maps before custom decoders exist.
    try:
        elev = _cached_npz_array(output_dir, "03_terrain.npz", "elevation_m")
        if elev is not None:
            v = _cell_value(elev, row, col)
            if v is not None:
                parts.append(f"elevation {int(v):,} m")
        land = _cached_npz_array(output_dir, "03_terrain.npz", "is_land")
        if land is not None:
            v = _cell_value(land, row, col)
            if v is not None:
                parts.append("land" if bool(v) else "water")
        plate = _cached_npz_array(output_dir, "03_terrain.npz", "tectonic_plate_id")
        if plate is not None:
            v = _cell_value(plate, row, col)
            if v is not None:
                parts.append(f"plate {int(v)}")
        boundary = _cached_npz_array(output_dir, "03_terrain.npz", "tectonic_boundary_class")
        if boundary is not None:
            v = _cell_value(boundary, row, col)
            if v is not None and int(v) != 0:
                parts.append(f"boundary class {int(v)}")
        crust = _cached_npz_array(output_dir, "03_terrain.npz", "crust_type")
        if crust is not None:
            v = _cell_value(crust, row, col)
            if v is not None:
                parts.append(f"crust {int(v)}")

        temp = _cached_npz_array(output_dir, "04_climate.npz", "annual_mean_temp_c_x10")
        if temp is not None:
            v = _cell_value(temp, row, col)
            if v is not None:
                parts.append(f"temp {float(v) / 10.0:.1f} °C")
        precip = _cached_npz_array(output_dir, "04_climate.npz", "annual_precip_mm")
        if precip is not None:
            v = _cell_value(precip, row, col)
            if v is not None:
                parts.append(f"rain {int(v):,} mm/yr")
        koppen = _cached_npz_array(output_dir, "04_climate.npz", "koppen_classification")
        if koppen is not None:
            v = _cell_value(koppen, row, col)
            if v is not None:
                parts.append(f"Köppen {_koppen_label(_clean_string_value(v))}")

        biome = _cached_npz_array(output_dir, "06_biomes.npz", "biome_classification")
        if biome is not None:
            v = _cell_value(biome, row, col)
            if v is not None:
                parts.append(f"biome {_clean_string_value(v)}")

        runoff = _cached_npz_array(output_dir, "05_hydrology.npz", "runoff_mm")
        if runoff is not None:
            v = _cell_value(runoff, row, col)
            if v is not None:
                parts.append(f"runoff {int(v):,} mm/yr")
        flow = _cached_npz_array(output_dir, "05_hydrology.npz", "flow_accumulation")
        if flow is not None:
            v = _cell_value(flow, row, col)
            if v is not None:
                parts.append(f"flow {int(v):,}")
        basin = _cached_npz_array(output_dir, "05_hydrology.npz", "drainage_basin_id")
        if basin is not None:
            v = _cell_value(basin, row, col)
            if v is not None:
                parts.append(f"basin {int(v)}")
    except Exception as exc:
        parts.append(f"details unavailable: {type(exc).__name__}")
    payload["context"] = parts
    payload["text"] = (str(payload.get("active_value")) + " · " if payload.get("active_value") else "") + " · ".join(parts)
    return payload


def _contour_kind_label(kind: str) -> str:
    return {
        "elevation": "Elevation contours",
        "temperature": "Temperature contours",
        "precipitation": "Rainfall contours",
    }.get(kind, kind)


def _load_contour_array(output_dir: Path, kind: str):
    import numpy as np

    if kind == "elevation":
        path = output_dir / "state" / "03_terrain.npz"
        if not path.exists():
            raise FileNotFoundError("state/03_terrain.npz is not available")
        data = np.load(path, allow_pickle=True)
        return data["elevation_m"].astype(float), "m"
    if kind == "temperature":
        path = output_dir / "state" / "04_climate.npz"
        if not path.exists():
            raise FileNotFoundError("state/04_climate.npz is not available")
        data = np.load(path, allow_pickle=True)
        return data["annual_mean_temp_c_x10"].astype(float) / 10.0, "°C"
    if kind == "precipitation":
        path = output_dir / "state" / "04_climate.npz"
        if not path.exists():
            raise FileNotFoundError("state/04_climate.npz is not available")
        data = np.load(path, allow_pickle=True)
        return data["annual_precip_mm"].astype(float), "mm/yr"
    raise ValueError(f"Unsupported contour kind: {kind}")


def _make_contour_png(output_dir: Path, kind: str, *, max_dim: int = 2048, levels: int = 12) -> bytes:
    import numpy as np

    arr, _unit = _load_contour_array(output_dir, kind)
    h, w = arr.shape[:2]
    max_dim = max(256, min(4096, int(max_dim)))
    step = max(1, int(math.ceil(max(w, h) / max_dim)))
    arr2 = arr[::step, ::step]
    # Avoid oceans dominating land elevation contours too much: quantile levels
    # are stable for all three supported fields and do not require metadata.
    finite = arr2[np.isfinite(arr2)]
    if finite.size < 10:
        raise ValueError("Not enough finite data for contours")
    lo = float(np.nanpercentile(finite, 4))
    hi = float(np.nanpercentile(finite, 96))
    if not math.isfinite(lo) or not math.isfinite(hi) or abs(hi - lo) < 1e-9:
        lo, hi = float(np.nanmin(finite)), float(np.nanmax(finite))
    if abs(hi - lo) < 1e-9:
        raise ValueError("Contour field is nearly constant")
    level_values = np.linspace(lo, hi, max(4, min(30, int(levels))))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_w = max(1, int(round(w / step)))
    out_h = max(1, int(round(h / step)))
    dpi = 100
    fig = plt.figure(figsize=(out_w / dpi, out_h / dpi), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    rows, cols = arr2.shape[:2]
    # Use pixel-edge limits so the transparent PNG aligns with the same
    # full-world raster extent as the generated maps.  The previous contour
    # image used center-coordinate limits, which could appear slightly shifted
    # when stretched over legend-bearing map PNGs in the browser.
    x = np.arange(cols) + 0.5
    y = np.arange(rows) + 0.5
    ax.contour(x, y, arr2, levels=level_values, colors="black", linewidths=0.45, alpha=0.72)
    ax.set_xlim(0, cols)
    ax.set_ylim(rows, 0)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", transparent=True, dpi=dpi, pad_inches=0)
    plt.close(fig)
    return buf.getvalue()


def _solar_system_svg(data: dict[str, Any], *, scale_mode: str = "log", show_labels: bool = True, show_hz: bool = True, show_snow: bool = True) -> str:
    star = data.get("star", {}) if isinstance(data.get("star"), dict) else {}
    planets = data.get("planets", []) if isinstance(data.get("planets"), list) else []
    width = 1280
    height = 520
    cx = 92
    cy = 250
    right = width - 76
    orbit_values: list[float] = []
    for planet in planets:
        if not isinstance(planet, dict):
            continue
        orbit = planet.get("orbit", {}) if isinstance(planet.get("orbit"), dict) else {}
        try:
            orbit_values.append(float(orbit.get("semi_major_axis_au")))
        except Exception:
            pass
    hz_inner = star.get("habitable_zone_inner_au")
    hz_outer = star.get("habitable_zone_outer_au")
    snow = star.get("snow_line_au")
    max_au = max(orbit_values + [float(snow or 0.0), float(hz_outer or 0.0), 1.0])
    log_min = math.log10(0.05)
    log_max = math.log10(max_au + 0.12)
    linear_max = max(max_au * 1.05, 0.2)

    def x_for_au(au: Any) -> float:
        try:
            value = max(0.0, float(au))
        except Exception:
            value = 0.0
        if scale_mode == "linear":
            return cx + value / max(0.001, linear_max) * (right - cx)
        return cx + (math.log10(value + 0.08) - log_min) / max(0.001, (log_max - log_min)) * (right - cx)

    star_class = str(star.get("stellar_class") or "G").upper()[:1]
    star_color = {"O":"#93c5fd","B":"#bfdbfe","A":"#e0f2fe","F":"#fef9c3","G":"#facc15","K":"#fb923c","M":"#ef4444"}.get(star_class, "#facc15")
    pieces = [
        f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="Interactive solar system overview">',
        f'<defs><radialGradient id="starGlow"><stop offset="0" stop-color="#fff"/><stop offset="0.35" stop-color="{star_color}"/><stop offset="1" stop-color="{star_color}" stop-opacity="0"/></radialGradient></defs>',
        f'<rect x="0" y="0" width="{width}" height="{height}" rx="18" fill="#020617"/>',
        '<text x="24" y="34" fill="#e5e7eb" font-size="20" font-weight="700">Interactive orbital overview</text>',
        f'<text x="24" y="58" fill="#94a3b8" font-size="13">{_safe_text(scale_mode.title())}-scaled orbital distance with optional habitable zone, snow line, labels, planet class color, and main-planet highlight.</text>',
        f'<line x1="{cx}" y1="{cy}" x2="{right}" y2="{cy}" stroke="#334155" stroke-width="2"/>',
        f'<circle cx="{cx}" cy="{cy}" r="42" fill="url(#starGlow)" opacity=".95"><title>Star: {_safe_text(star.get("stellar_class", ""))}</title></circle>',
        f'<circle cx="{cx}" cy="{cy}" r="24" fill="{star_color}" stroke="#fde68a" stroke-width="2"/>',
        f'<text x="{cx-18}" y="{cy+58}" fill="#fde68a" font-size="13">{_safe_text(star.get("stellar_class", "star"))}</text>',
    ]
    if show_hz and hz_inner is not None and hz_outer is not None:
        x1 = x_for_au(hz_inner)
        x2 = x_for_au(hz_outer)
        pieces.append(f'<rect x="{x1:.1f}" y="90" width="{max(2.0, x2-x1):.1f}" height="310" fill="#14532d" opacity="0.25"><title>Habitable zone: {_safe_text(_fmt_num(hz_inner))}-{_safe_text(_fmt_num(hz_outer))} AU</title></rect>')
        pieces.append(f'<text x="{x1+6:.1f}" y="112" fill="#86efac" font-size="14">habitable zone</text>')
    if show_snow and snow is not None:
        xs = x_for_au(snow)
        pieces.append(f'<line x1="{xs:.1f}" y1="78" x2="{xs:.1f}" y2="416" stroke="#bae6fd" stroke-width="2" stroke-dasharray="7 7"><title>Snow line: {_safe_text(_fmt_num(snow))} AU</title></line>')
        pieces.append(f'<text x="{xs+6:.1f}" y="90" fill="#bae6fd" font-size="14">snow line</text>')
    # AU guide ticks
    for au in [0.05, 0.1, 0.3, 1, 3, 10, 30, 100]:
        if au <= max_au * 1.2:
            x = x_for_au(au)
            pieces.append(f'<line x1="{x:.1f}" y1="{cy-8}" x2="{x:.1f}" y2="{cy+8}" stroke="#64748b"/>')
            pieces.append(f'<text x="{x-10:.1f}" y="{cy+30}" fill="#94a3b8" font-size="12">{au:g}</text>')
    for idx, planet in enumerate(planets):
        if not isinstance(planet, dict):
            continue
        orbit = planet.get("orbit", {}) if isinstance(planet.get("orbit"), dict) else {}
        au = orbit.get("semi_major_axis_au", 0.0)
        x = x_for_au(au)
        radius = planet.get("radius_earth", 1.0) or 1.0
        try:
            pr = max(5.0, min(25.0, 5.5 + math.sqrt(float(radius)) * 5.0))
        except Exception:
            pr = 9.0
        y = cy + ((idx % 2) * 2 - 1) * (48 + (idx % 3) * 22)
        style = _planet_type_style(planet.get("planet_class"))
        color = "#38bdf8" if planet.get("is_main_planet") else style["color"]
        icon = style["icon"]
        name = _safe_text(planet.get("name", f"planet {idx+1}"))
        title = f"{name}: {_safe_text(planet.get('planet_class',''))}, {_safe_text(_fmt_num(au))} AU, {_safe_text(_fmt_num(planet.get('mass_earth'), 2))} M⊕, {_safe_text(_fmt_num(planet.get('radius_earth'), 2))} R⊕, flux {_safe_text(_fmt_num(planet.get('stellar_flux_earth'), 2))} S⊕"
        pieces.append(f'<line x1="{x:.1f}" y1="{cy}" x2="{x:.1f}" y2="{y:.1f}" stroke="#475569" stroke-width="1"/>')
        pieces.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{pr:.1f}" fill="{color}" stroke="#e2e8f0" stroke-width="1.5"><title>{title}</title></circle>')
        pieces.append(f'<text x="{x:.1f}" y="{y+4:.1f}" text-anchor="middle" fill="#020617" font-size="{max(9, pr*0.9):.1f}" font-weight="900">{_safe_text(icon)}</text>')
        if show_labels:
            label_y = y - pr - 8 if y < cy else y + pr + 18
            pieces.append(f'<text x="{x+6:.1f}" y="{label_y:.1f}" fill="#e5e7eb" font-size="12">{name}</text>')
        if planet.get("is_main_planet"):
            pieces.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{pr+7:.1f}" fill="none" stroke="#22c55e" stroke-width="3"/>')
    pieces.append('<g transform="translate(24 452)">')
    legend = [("#38bdf8", "Main planet"), ("#a3e635", "rocky/terrestrial"), ("#22c55e", "super-Earth"), ("#c084fc", "gas giant"), ("#7dd3fc", "ice/mini-Neptune"), ("#94a3b8", "icy/dwarf/other")]
    if show_hz:
        legend.append(("#14532d", "habitable zone"))
    if show_snow:
        legend.append(("#bae6fd", "snow line"))
    lx = 0
    for color, label in legend:
        pieces.append(f'<circle cx="{lx+8}" cy="8" r="7" fill="{color}" opacity=".9"/><text x="{lx+22}" y="13" fill="#cbd5e1" font-size="12">{label}</text>')
        lx += 145
    pieces.append('</g>')
    pieces.append('</svg>')
    return "".join(pieces)

def _run_overview_html(output_dir: Path) -> str:
    """Summarize available solar-system and Main Planet state on the run dashboard."""
    solar = _solar_system_state(output_dir) or {}
    if not solar:
        return ""
    star = solar.get("star", {}) if isinstance(solar.get("star"), dict) else {}
    diagnostics_state = solar.get("diagnostics", {}) if isinstance(solar.get("diagnostics"), dict) else {}
    planets = solar.get("planets", []) if isinstance(solar.get("planets"), list) else []
    main_planet = next((p for p in planets if isinstance(p, dict) and p.get("is_main_planet")), {})
    formation_context = main_planet.get("formation_context", {}) if isinstance(main_planet.get("formation_context"), dict) else {}
    physics = _read_json(output_dir / "state" / "02_planet_physics.json", {})
    if not isinstance(physics, dict):
        physics = {}
    rotation = physics.get("rotation", {}) if isinstance(physics.get("rotation"), dict) else {}
    atmosphere = physics.get("atmosphere", {}) if isinstance(physics.get("atmosphere"), dict) else {}
    hydrosphere = physics.get("hydrosphere", {}) if isinstance(physics.get("hydrosphere"), dict) else {}
    geology = physics.get("geology", {}) if isinstance(physics.get("geology"), dict) else {}
    planet_rows = ""
    for idx, planet in enumerate(planets, start=1):
        if not isinstance(planet, dict):
            continue
        orbit = planet.get("orbit", {}) if isinstance(planet.get("orbit"), dict) else {}
        comp = planet.get("composition", {}) if isinstance(planet.get("composition"), dict) else {}
        cls = "main-planet" if planet.get("is_main_planet") else ""
        main_badge = " <span class='pill'>Main</span>" if planet.get("is_main_planet") else ""
        planet_rows += (
            f"<tr class='{cls}'><td>{idx}</td><td>{_planet_visual(planet.get('planet_class'), bool(planet.get('is_main_planet')))}<strong>{_safe_text(planet.get('name',''))}</strong>{main_badge}</td>"
            f"<td>{_safe_text(planet.get('planet_class',''))}</td>"
            f"<td>{_safe_text(_humanize_key(planet.get('architecture_role','')))}</td>"
            f"<td>{_safe_text(_fmt_compact(orbit.get('semi_major_axis_au'), 3, ' AU'))}</td>"
            f"<td>{_safe_text(_fmt_compact(planet.get('mass_earth'), 2, ' M⊕'))}</td>"
            f"<td>{_safe_text(_fmt_compact(planet.get('radius_earth'), 2, ' R⊕'))}</td>"
            f"<td>{_safe_text(_fmt_compact(planet.get('stellar_flux_earth'), 2, ' S⊕'))}</td>"
            f"<td>{_safe_text(comp.get('composition_class',''))}</td></tr>"
        )
    planet_table = "<p class='muted'>No planet list found yet.</p>" if not planet_rows else f"""
      <table class='planet-table'><thead><tr><th>#</th><th>Planet</th><th>Class</th><th>Role</th><th>Orbit</th><th>Mass</th><th>Radius</th><th>Flux</th><th>Composition</th></tr></thead><tbody>{planet_rows}</tbody></table>
    """
    orbit = main_planet.get("orbit", {}) if isinstance(main_planet.get("orbit"), dict) else {}
    star_cards = f"""
      <div class='stat' style='grid-column:span 2'>{_star_visual(star.get('spectral_type') or star.get('stellar_class','—'), star.get('temperature_k'))}<span class='muted small'>{_safe_text(star.get('stellar_description',''))}</span></div>
      <div class='stat'><b>Architecture</b>{_safe_text(_humanize_key(solar.get('architecture') or diagnostics_state.get('architecture')))}</div>
      <div class='stat'><b>Candidate quality</b>{_safe_text(_humanize_key(diagnostics_state.get('main_planet_candidate_quality')))}</div>
      <div class='stat'><b>Mass</b>{_safe_text(_fmt_compact(star.get('mass_solar'), 2, ' M☉'))}{_progress_bar(star.get('mass_solar'), 0.6, 1.3, 'stellar mass')}</div>
      <div class='stat'><b>Luminosity</b>{_safe_text(_fmt_compact(star.get('luminosity_solar'), 2, ' L☉'))}{_progress_bar(star.get('luminosity_solar'), 0.2, 2.0, 'stellar luminosity')}</div>
      <div class='stat'><b>Temperature</b>{_safe_text(_fmt_compact(star.get('temperature_k'), 0, ' K'))}</div>
      <div class='stat'><b>Habitable zone</b>{_safe_text(_fmt_compact(star.get('habitable_zone_inner_au'), 2, ' AU'))} – {_safe_text(_fmt_compact(star.get('habitable_zone_outer_au'), 2, ' AU'))}</div>
      <div class='stat'><b>Snow line</b>{_safe_text(_fmt_compact(star.get('snow_line_au'), 2, ' AU'))}</div>
    """
    main_cards = ""
    if main_planet:
        main_cards += f"""
          <div class='stat'><b>Main planet</b>{_safe_text(main_planet.get('name','—'))}</div>
          <div class='stat'><b>Class</b>{_safe_text(main_planet.get('planet_class','—'))}</div>
          <div class='stat'><b>Orbit</b>{_safe_text(_fmt_compact(orbit.get('semi_major_axis_au'), 3, ' AU'))}</div>
          <div class='stat'><b>Eccentricity</b>{_safe_text(_fmt_compact(orbit.get('eccentricity'), 3))}</div>
          <div class='stat'><b>Mass</b>{_safe_text(_fmt_compact(main_planet.get('mass_earth'), 2, ' M⊕'))}{_progress_bar(main_planet.get('mass_earth'), 0.2, 5.0, 'planet mass')}</div>
          <div class='stat'><b>Radius</b>{_safe_text(_fmt_compact(main_planet.get('radius_earth'), 2, ' R⊕'))}{_progress_bar(main_planet.get('radius_earth'), 0.4, 2.5, 'planet radius')}</div>
          <div class='stat'><b>Gravity</b>{_safe_text(_fmt_compact(main_planet.get('surface_gravity_g'), 2, ' g'))}{_progress_bar(main_planet.get('surface_gravity_g'), 0.4, 2.0, 'surface gravity')}</div>
          <div class='stat'><b>Habitability score</b>{_safe_text(_fmt_compact(main_planet.get('habitability_score'), 2))}{_progress_bar(main_planet.get('habitability_score'), 0, 100, 'habitability score')}</div>
        """
    if rotation or atmosphere or hydrosphere or geology:
        main_cards += f"""
          <div class='stat'><b>Rotation</b>{_safe_text(_fmt_compact(rotation.get('rotation_period_hours'), 1, ' h'))}</div>
          <div class='stat'><b>Axial tilt</b>{_safe_text(_fmt_compact(rotation.get('axial_tilt_degrees'), 1, '°'))}</div>
          <div class='stat'><b>Pressure</b>{_safe_text(_fmt_compact(atmosphere.get('pressure_bar'), 2, ' bar'))}</div>
          <div class='stat'><b>CO₂</b>{_safe_text(_fmt_compact(atmosphere.get('carbon_dioxide_ppm'), 0, ' ppm'))}</div>
          <div class='stat'><b>Ocean target</b>{_safe_text(_fmt_compact(hydrosphere.get('ocean_fraction_target'), 2))}{_progress_bar(hydrosphere.get('ocean_fraction_target'), 0, 1, 'ocean fraction target')}</div>
          <div class='stat'><b>Volcanism</b>{_safe_text(_fmt_compact(geology.get('volcanism'), 2))}{_progress_bar(geology.get('volcanism'), 0, 1, 'volcanism')}</div>
        """
    solar_context_html = ""
    if diagnostics_state or formation_context:
        diagnostics_keys = [
            "architecture", "main_planet_preference", "generation_attempts", "planet_count",
            "habitable_zone_planet_count", "outer_giant_count", "giant_planet_influence",
            "main_planet_candidate_quality", "climate_stability_outlook",
        ]
        formation_keys = [
            "formation_zone", "volatile_delivery", "giant_planet_influence", "impact_history",
            "tectonic_energy_bias", "crustal_asymmetry_bias", "moon_origin",
            "tidal_effect_level", "axial_stability_effect",
        ]
        solar_context_html = f"""
        <section class='card' style='margin-top:14px'>
          <h3>Solar-system diagnostics and formation context</h3>
          <div class='grid compact-grid'>{_mapping_stat_items(diagnostics_state, diagnostics_keys)}</div>
          <h4>Main Planet formation context</h4>
          <div class='grid compact-grid'>{_mapping_stat_items(formation_context, formation_keys)}</div>
        </section>
        """
    visual_indicators = ""
    if main_planet:
        visual_indicators = f"""
        <section class='card' style='margin-top:14px'>
          <h3>Visual habitability / system indicators</h3>
          <div class='grid compact-grid'>
            {_metric_gauge('Stellar flux', main_planet.get('stellar_flux_earth'), unit='S⊕', min_v=0.25, max_v=2.2, low='cold/dim', mid='temperate range', high='hot/bright')}
            {_metric_gauge('Gravity', main_planet.get('surface_gravity_g'), unit='g', min_v=0.25, max_v=2.2, low='low gravity', mid='comfortable', high='high gravity')}
            {_metric_gauge('Planet radius', main_planet.get('radius_earth'), unit='R⊕', min_v=0.35, max_v=2.5, low='small', mid='Earth-like/super-Earth', high='large')}
            {_metric_gauge('Habitability score', main_planet.get('habitability_score'), min_v=0.0, max_v=100.0, low='poor', mid='possible', high='strong')}
            {_metric_gauge('Ocean target', hydrosphere.get('ocean_fraction_target'), min_v=0.0, max_v=1.0, low='dry', mid='mixed land/ocean', high='ocean world')}
            {_metric_gauge('Volcanism', geology.get('volcanism'), min_v=0.0, max_v=1.0, low='quiet', mid='active', high='very active')}
          </div>
          <h4>Orbital position</h4>
          {_hz_position_indicator(star, main_planet)}
          <h4>Planet type mix</h4>
          {_planet_mix_html(planets)}
          <h4>Moon indicator</h4>
          {_moon_summary_html(output_dir)}
        </section>
        """
    return f"""
<section class='card' style='margin-top:18px'>
  <h2>System and planet overview</h2>
  <h3>Star</h3>
  <div class='stat-grid'>{star_cards}</div>
  <h3 style='margin-top:16px'>Main Planet / physics</h3>
  <div class='stat-grid'>{main_cards or '<p class="muted">Main Planet state not available yet.</p>'}</div>
  {solar_context_html}
  {visual_indicators}
  <details style='margin-top:16px' open><summary>Planet list</summary>{planet_table}</details>
</section>
"""


def _diagnostic_files(output_dir: Path) -> list[Path]:
    files: list[Path] = []
    for folder in (output_dir / "diagnostics", output_dir, output_dir / "terrain_diagnostics"):
        if folder.exists():
            iterator = folder.rglob("*") if folder.name == "terrain_diagnostics" else folder.glob("*")
            for path in sorted(iterator):
                if path.is_file() and path.suffix.lower() in {".txt", ".csv", ".json", ".zip"}:
                    files.append(path)
    return files


def _state_files(output_dir: Path) -> list[Path]:
    state = output_dir / "state"
    if not state.exists():
        return []
    return sorted([p for p in state.iterdir() if p.is_file() and p.suffix.lower() in {".json", ".npz"}])


def _mini_stat_card(label: str, value: Any) -> str:
    return f"<div class='stat'><b>{_safe_text(label)}</b>{_safe_text(value)}</div>"


def _terrain_review_state(output_dir: Path) -> dict[str, Any]:
    data = _read_json(output_dir / "terrain_diagnostics" / "terrain_review.json", {})
    if isinstance(data, dict) and data:
        return data
    meta = _read_json(output_dir / "state" / "03_terrain_metadata.json", {})
    if isinstance(meta, dict) and isinstance(meta.get("terrain_diagnostics"), dict):
        return meta["terrain_diagnostics"]
    return {}


def _terrain_diagnostic_image_groups(output_dir: Path) -> dict[str, list[Path]]:
    root = output_dir / "terrain_diagnostics"
    if not root.exists():
        return {}
    groups: dict[str, list[Path]] = {}
    for folder in sorted([p for p in root.iterdir() if p.is_dir()]):
        images = sorted([p for p in folder.glob("*.png") if p.is_file()])
        if images:
            groups[folder.name] = images
    return groups




def _landmass_components_table_html(output_dir: Path, *, limit: int = 12) -> str:
    path = output_dir / "landmass_components.csv"
    if not path.exists():
        return "<p class='muted'>Landmass component CSV is not available yet.</p>"
    try:
        rows: list[dict[str, str]] = []
        with path.open("r", encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                rows.append({str(k): str(v) for k, v in row.items()})
                if len(rows) >= max(1, limit):
                    break
        if not rows:
            return "<p class='muted'>Landmass component CSV is empty.</p>"
        cols = [
            "component_id", "type", "cell_count", "area_km2", "area_percent_of_land",
            "mean_elevation_m", "max_elevation_m", "dominant_surface_crust", "is_seam_crossing",
        ]
        head = "".join(f"<th>{_safe_text(_humanize_key(c))}</th>" for c in cols)
        body = ""
        for row in rows:
            body += "<tr>" + "".join(f"<td>{_safe_text(row.get(c, ''))}</td>" for c in cols) + "</tr>"
        return f"""
<details class='folder-card landmass-table-card' open>
  <summary>Largest landmasses — top {len(rows)} components</summary>
  <p class='help'>Area is latitude-weighted for the equirectangular map, so polar cells count less than equatorial cells. Use the CSV for the full list.</p>
  <div class='table-scroll'><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>
</details>
"""
    except Exception as exc:
        return f"<p class='muted'>Could not preview landmass components: {_safe_text(exc)}</p>"

def _terrain_diagnostic_folder_html(output_dir: Path) -> str:
    review = _terrain_review_state(output_dir)
    groups = _terrain_diagnostic_image_groups(output_dir)
    if not review and not groups:
        return "<p class='muted'>No terrain diagnostic folders are available yet. Generate terrain to create Stage 3 diagnostics.</p>"
    subphases = review.get("subphases", {}) if isinstance(review.get("subphases"), dict) else {}
    sections: list[str] = []
    for folder, images in groups.items():
        stage_info = next((info for info in subphases.values() if isinstance(info, dict) and info.get("folder") == folder), {})
        metrics = stage_info.get("metrics", {}) if isinstance(stage_info, dict) and isinstance(stage_info.get("metrics"), dict) else {}
        metric_rows = "".join(f"<tr><td>{_safe_text(_humanize_key(k))}</td><td>{_safe_text(v)}</td></tr>" for k, v in list(metrics.items())[:18])
        image_cards = ""
        for img in images:
            rel = _rel_label(img, output_dir)
            raw = _file_url(img)
            view = _map_view_url(img, output_dir)
            image_cards += f"<div class='mapcard'><a href='{view}'><img src='{raw}' loading='lazy'></a><p>{_safe_text(rel)}</p><div class='row small'><a href='{view}'>View</a><a href='{raw}' target='_blank'>Raw</a></div></div>"
        sections.append(f"""
<details class='folder-card'>
  <summary>{_safe_text(folder)} — {len(images)} maps</summary>
  <p class='help'>{_safe_text(stage_info.get('summary', 'Terrain diagnostic folder.'))}</p>
  <div class='maps'>{image_cards}</div>
  <details><summary>Metrics</summary><table><tbody>{metric_rows or '<tr><td class="muted">No metrics recorded.</td></tr>'}</tbody></table></details>
</details>
""")
    report = output_dir / "terrain_diagnostics" / "terrain_review_report.txt"
    report_link = f"<a class='button secondary' href='{_file_url(report)}' target='_blank'>Terrain review report</a>" if report.exists() else ""
    metrics = output_dir / "terrain_diagnostics" / "terrain_subphase_metrics.csv"
    metrics_link = f"<a class='button secondary' href='{_file_url(metrics)}' target='_blank'>Subphase metrics CSV</a>" if metrics.exists() else ""
    landmass_csv = output_dir / "landmass_components.csv"
    landmass_link = f"<a class='button secondary' href='{_file_url(landmass_csv)}' target='_blank'>Landmass components CSV</a>" if landmass_csv.exists() else ""
    v4_csv = output_dir / "v4_topology_diagnostics.csv"
    v4_link = f"<a class='button secondary' href='{_file_url(v4_csv)}' target='_blank'>v4 topology CSV</a>" if v4_csv.exists() else ""
    summary = review.get("summary", {}) if isinstance(review.get("summary"), dict) else {}
    summary_cards = "".join(_mini_stat_card(_humanize_key(k), v) for k, v in summary.items()) if summary else ""
    warnings = review.get("warnings", []) if isinstance(review.get("warnings"), list) else []
    warning_html = _stage_warnings_html(warnings) if warnings else "<p class='muted'>No terrain-specific warnings recorded.</p>"
    return f"""
<section class='card terrain-review-panel'>
  <h2>Terrain review diagnostics</h2>
  <p class='help'>These diagnostics are organized by terrain sub-stage. They are derived from the current terrain raster and checkpoint metadata so we can review terrain quality before deeper terrain-science refactoring.</p>
  <div class='row'>{report_link}{metrics_link}{landmass_link}{v4_link}</div>
  {_landmass_components_table_html(output_dir)}
  <div class='stat-grid'>{summary_cards}</div>
  <details><summary>Terrain warnings</summary>{warning_html}</details>
  {''.join(sections)}
</section>
"""


TERRAIN_SUBPHASE_REVIEW_SPEC: dict[str, dict[str, Any]] = {
    "terrain-foundation-mask": {
        "title": "3.1 Foundation mask",
        "focus": "Broad land/ocean foundation, landmass distribution, supercontinent tendency, and fragmentation tendency.",
        "review_questions": [
            "Does the ocean/land split roughly match the Stage 2 ocean target?",
            "Is the largest landmass share plausible for the derived supercontinent score?",
            "Are hemisphere and latitude imbalances interesting without feeling railroaded?",
        ],
        "primary_metrics": ["ocean_fraction", "target_ocean_fraction", "landmass_count", "largest_landmass_share_of_land", "derived_supercontinent_score", "fragmentation_tendency"],
    },
    "terrain-tectonic-provinces": {
        "title": "3.2 Tectonic provinces",
        "focus": "Procedural province/plate-style regions, province count, diversity, and large-province dominance.",
        "review_questions": [
            "Do provinces avoid overly neat arcs and identical spacing?",
            "Does the province count fit the tectonic regime and fragmentation tendency?",
            "Are large provinces balanced by smaller terranes or microcontinents?",
        ],
        "primary_metrics": ["target_plate_count", "diagnostic_plate_count", "largest_province_share", "province_size_diversity_score", "province_type_diversity_score", "continental_or_margin_province_share", "microcontinent_terrane_share"],
    },
    "terrain-crust-and-boundaries": {
        "title": "3.3 Crust and boundaries",
        "focus": "Crust classes, active/passive margins, rift-like zones, convergent zones, and boundary neatness.",
        "review_questions": [
            "Are boundary classes varied rather than one repeated shape?",
            "Do active margins, rifts, and transforms line up with terrain style and Stage 2 geology?",
            "Do crust classes distinguish deep ocean, shelves, ordinary continents, and high/orogenic land?",
        ],
        "primary_metrics": ["native_plate_motion_speed_mean", "native_plate_convergence_mean", "native_plate_divergence_mean", "native_plate_transform_mean", "native_plate_convergent_or_arc_share", "native_plate_divergent_share", "native_plate_transform_share", "mean_boundary_strength", "mean_boundary_width_proxy"],
    },
    "terrain-mountains-basins-rifts": {
        "title": "3.4 Mountains, basins, and rifts",
        "focus": "Mountain belts, interior relief, basins, highlands, rifts, and non-flat continental interiors.",
        "review_questions": [
            "Are mountains segmented and regionally varied instead of long straight ribbons?",
            "Do continental interiors include shields, basins, plateaus, or old ranges?",
            "Will relief provide useful gradients for climate and rivers later?",
        ],
        "primary_metrics": ["max_elevation_m", "mean_land_elevation_m", "mountain_highland_share_of_land", "basin_lowland_share_of_land", "mean_land_slope_proxy", "mountain_straightness_proxy"],
    },
    "terrain-coasts-shelves-islands": {
        "title": "3.5 Coasts, shelves, and islands",
        "focus": "Coastline complexity, shelves, island abundance, archipelago zones, and island shape diversity.",
        "review_questions": [
            "Do coasts vary regionally instead of using one global roughness style?",
            "Are islands diverse enough, or still too oval/blob-like?",
            "Do shelves and shallow seas create useful coastal plains without drowning all relief?",
        ],
        "primary_metrics": ["coastline_cell_count", "coastline_complexity_proxy", "coast_style_diversity_score", "mean_shelf_width_proxy", "rugged_or_fjorded_coast_share", "island_origin_diversity_score", "island_count", "derived_island_density"],
    },
    "terrain-erosion-deposition": {
        "title": "3.6 Erosion and deposition",
        "focus": "Slope smoothing, lowlands, sediment accommodation, alluvial basins, and valley-corridor readiness.",
        "review_questions": [
            "Are plains and basins present without flattening entire continents?",
            "Does erosion/deposition preserve enough mountain relief for climate barriers?",
            "Are there likely corridors for rivers to reach coasts in the hydrology stage?",
        ],
        "primary_metrics": ["mean_land_slope_proxy", "valley_corridor_mean", "deposition_field_mean", "sediment_supply_mean", "terrain_maturity_mean", "mean_abs_relief_delta_m", "derived_valley_carving_strength", "derived_deposition_strength"],
    },
    "terrain-finalization-recentering": {
        "title": "3.7 Final terrain",
        "focus": "Final terrain raster, final land/ocean balance, quality warnings, regional terrain maps, and downstream readiness.",
        "review_questions": [
            "Is the final world worth sending into climate/hydrology, or should terrain be rerolled/tuned first?",
            "Do landmass, coast, island, and relief diagnostics match the intended planet profile?",
            "Are any warnings severe enough to rerun at low resolution before committing high resolution?",
        ],
        "primary_metrics": ["overall_terrain_quality_score", "hydrology_readiness_label", "hydrology_readiness_score", "final_ocean_fraction", "target_ocean_fraction", "ocean_target_error", "landmass_count", "largest_landmass_share_of_land", "flat_interior_share_of_land", "drainage_corridor_moderate_share", "terrain_diversity_score", "terrain_style"],
    },
}


def _terrain_single_subphase_html(output_dir: Path, stage: str) -> str:
    stage = normalize_stage(stage)
    if stage not in TERRAIN_SUBPHASE_REVIEW_SPEC:
        return _terrain_diagnostic_folder_html(output_dir)
    review = _terrain_review_state(output_dir)
    groups = _terrain_diagnostic_image_groups(output_dir)
    if not review and not groups:
        return "<p class='muted'>No terrain diagnostics are available yet. Generate terrain to create Stage 3 sub-stage diagnostics.</p>"
    spec = TERRAIN_SUBPHASE_REVIEW_SPEC[stage]
    subphases = review.get("subphases", {}) if isinstance(review.get("subphases"), dict) else {}
    info = subphases.get(stage, {}) if isinstance(subphases.get(stage), dict) else {}
    metrics = info.get("metrics", {}) if isinstance(info.get("metrics"), dict) else {}
    folder = str(info.get("folder") or "")
    images = groups.get(folder, [])

    primary_rows = ""
    for key in spec.get("primary_metrics", []):
        if key in metrics:
            primary_rows += f"<tr><td>{_safe_text(_humanize_key(key))}</td><td>{_safe_text(metrics.get(key))}</td></tr>"
    if not primary_rows:
        primary_rows = "<tr><td colspan='2' class='muted'>Generate this terrain stage to populate metrics.</td></tr>"
    extra_rows = "".join(
        f"<tr><td>{_safe_text(_humanize_key(k))}</td><td>{_safe_text(v)}</td></tr>"
        for k, v in metrics.items()
        if k not in set(spec.get("primary_metrics", []))
    )
    question_items = "".join(f"<li>{_safe_text(q)}</li>" for q in spec.get("review_questions", []))
    image_cards = ""
    for img in images:
        rel = _rel_label(img, output_dir)
        raw = _file_url(img)
        view = _map_view_url(img, output_dir)
        compare = "/compare?" + urllib.parse.urlencode({"output_dir": str(output_dir), "left": rel})
        image_cards += f"<div class='mapcard'><a href='{view}'><img src='{raw}' loading='lazy'></a><p>{_safe_text(rel)}</p><div class='row small'><a href='{view}'>View</a><a href='{compare}'>Compare</a><a href='{raw}' target='_blank'>Raw</a></div></div>"
    if not image_cards:
        image_cards = "<p class='muted'>No diagnostic maps were written for this sub-stage yet.</p>"
    marker = terrain_subphase_marker_path(output_dir, stage) if 'terrain_subphase_marker_path' in globals() else None
    marker_link = f"<a class='button secondary' href='{_file_url(marker)}' target='_blank'>Checkpoint marker JSON</a>" if marker is not None and marker.exists() else ""
    final_panel = ""
    if stage == "terrain-finalization-recentering":
        final_quality = review.get("final_quality", {}) if isinstance(review.get("final_quality"), dict) else {}
        checks = review.get("hydrology_readiness_checks", []) if isinstance(review.get("hydrology_readiness_checks"), list) else []
        label = str(final_quality.get("hydrology_readiness_label", "unknown"))
        score = final_quality.get("hydrology_readiness_score", "—")
        cls = "complete" if label == "ready" else ("warn" if "concern" in label or "mostly" in label else "bad")
        rows = ""
        for item in checks:
            if isinstance(item, dict):
                status = str(item.get("status", "review"))
                status_cls = "complete" if status == "pass" else ("warn" if status == "review" else "bad")
                rows += f"<tr><td><span class='{status_cls}'>{_safe_text(status.upper())}</span></td><td>{_safe_text(item.get('label',''))}</td><td>{_safe_text(item.get('score','—'))}</td><td class='muted small'>{_safe_text(item.get('detail',''))}</td></tr>"
        if not rows:
            rows = "<tr><td colspan='4' class='muted'>No hydrology-readiness checks recorded yet.</td></tr>"
        final_report = output_dir / "terrain_diagnostics" / "07_final" / "final_terrain_quality_report.json"
        contact_sheet = output_dir / "terrain_regions" / "terrain_region_contact_sheet.png"
        final_links = ""
        if final_report.exists():
            final_links += f"<a class='button secondary' href='{_file_url(final_report)}' target='_blank'>Final QA JSON</a>"
        if contact_sheet.exists():
            final_links += f"<a class='button secondary' href='{_map_view_url(contact_sheet, output_dir)}'>Regional contact sheet</a>"
        final_panel = f"""
  <section class='card terrain-final-approval'>
    <div class='pipeline-stage-head'><h3>Ready for hydrology?</h3><span class='{cls}'>{_safe_text(label)} · {_safe_text(score)}</span></div>
    <p class='help'>This is a terrain-only readiness check. It does not guarantee good rivers; it tells us whether the final terrain has enough ocean-target fit, relief, coasts, basins, valley corridors, and diversity to be worth sending into hydrology.</p>
    <table><thead><tr><th>Status</th><th>Check</th><th>Score</th><th>Detail</th></tr></thead><tbody>{rows}</tbody></table>
    <div class='row'>{final_links}</div>
  </section>
"""
    all_folders = _terrain_diagnostic_folder_html(output_dir)
    return f"""
<section class='card terrain-substage-review'>
  <div class='pipeline-stage-head'><h2>{_safe_text(spec.get('title', _stage_label(stage)))}</h2><span class='pill'>terrain sub-stage</span></div>
  <p>{_safe_text(spec.get('focus', 'Terrain sub-stage diagnostics.'))}</p>
  <p class='warn'>Current implementation note: terrain is still synthesized by one shared terrain pass. This page focuses on the checkpoint diagnostics for this sub-stage; true independent pass timing/rerun behavior will come with the deeper terrain generator refactor.</p>
  <div class='row'>{marker_link}</div>
  {final_panel}
  <div class='grid'>
    <section class='card'><h3>Primary metrics</h3><table><tbody>{primary_rows}</tbody></table></section>
    <section class='card'><h3>What to inspect</h3><ul>{question_items}</ul></section>
  </div>
  <h3>Diagnostic maps for this sub-stage</h3>
  <div class='maps'>{image_cards}</div>
  <details><summary>All metrics for this sub-stage</summary><table><tbody>{extra_rows or '<tr><td class="muted">No additional metrics recorded.</td></tr>'}</tbody></table></details>
  <details><summary>All terrain diagnostic folders</summary>{all_folders}</details>
</section>
"""


def _terrain_lowres_preview_dir(output_dir: Path) -> Path:
    root = output_dir / "terrain_previews"
    root.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    return root / f"terrain_preview_2048x1024_{stamp}"


def _prepare_terrain_lowres_preview_run(output_dir: Path, preview_dir: Path, *, width: int = 2048, height: int = 1024) -> None:
    ensure_layout(preview_dir)
    # Preserve the current reviewed Stage 1/2 state, but use lower terrain resolution.
    for rel in ["config/resolved_config.json", "config/stage_overrides.json", "config/value_provenance.json", "state/01_solar_system.json", "state/02_planet_physics.json"]:
        src = output_dir / rel
        dst = preview_dir / rel
        if src.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
    cfg_path = preview_dir / "config" / "resolved_config.json"
    cfg = _read_json(cfg_path, {})
    if isinstance(cfg, dict):
        pp = cfg.setdefault("planet_profile", {})
        if isinstance(pp, dict):
            pp["map_width"] = width
            pp["map_height"] = height
            pp["min_map_width"] = min(int(pp.get("min_map_width", width) or width), width)
            pp["min_map_height"] = min(int(pp.get("min_map_height", height) or height), height)
        _write_json(cfg_path, cfg)
    note = {
        "created_at": now_stamp(),
        "source_run": str(output_dir),
        "purpose": "Low-resolution Stage 3 terrain test. Review this before committing terrain at the source run resolution.",
        "preview_width": width,
        "preview_height": height,
    }
    _write_json(preview_dir / "diagnostics" / "terrain_preview_note.json", note)


def _terrain_review_controls_html(output_dir: Path, stage: str) -> str:
    if not stage.startswith("terrain"):
        return ""
    return f"""
<section class='card terrain-workflow-card'>
  <h2>Terrain test / commit workflow</h2>
  <p class='help'>Terrain is expensive. Use a 2048×1024 preview to inspect landmasses, coasts, provinces, and relief before committing terrain at the run's full resolution. The preview is written to a separate folder and does not overwrite the current run.</p>
  <div class='row'>
    <form method='post' action='/stage-workflow'>
      <input type='hidden' name='output_dir' value='{_safe_text(output_dir)}'>
      <input type='hidden' name='stage' value='{_safe_text(stage)}'>
      <input type='hidden' name='action' value='terrain-lowres-test'>
      <button type='submit'>Generate low-res terrain test 2048×1024</button>
    </form>
    <form method='post' action='/stage-workflow' onsubmit="return confirm('Commit terrain at the current run resolution? This can be slow and will mark climate/hydrology/biomes/regions stale.');">
      <input type='hidden' name='output_dir' value='{_safe_text(output_dir)}'>
      <input type='hidden' name='stage' value='{_safe_text(stage)}'>
      <input type='hidden' name='action' value='terrain-full-commit'>
      <button class='secondary' type='submit'>Commit full-resolution terrain</button>
    </form>
  </div>
</section>
"""


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}
TEXT_SUFFIXES = {".txt", ".csv", ".json", ".md"}


def _rel_label(path: Path, output_dir: Path | None = None) -> str:
    try:
        if output_dir is not None:
            return str(path.relative_to(output_dir))
    except Exception:
        pass
    return path.name


def _file_url(path: Path) -> str:
    return "/file/" + urllib.parse.quote(str(path.resolve()))


def _map_view_url(path: Path, output_dir: Path | None = None) -> str:
    params = {"path": str(path.resolve())}
    if output_dir is not None:
        params["output_dir"] = str(output_dir)
    return "/map?" + urllib.parse.urlencode(params)


def _monthly_sequence_context(path: Path, output_dir: Path | None) -> dict[str, Any] | None:
    """Return sequence metadata for climate_monthly temperature/precip maps."""
    name = path.name.lower()
    m = re.match(r"main_planet_(temperature|precipitation)_month_(\d\d)\.png$", name)
    if not m:
        return None
    kind = m.group(1)
    month = int(m.group(2))
    folder = path.parent
    entries: list[dict[str, Any]] = []
    for idx in range(1, 13):
        candidate = folder / f"main_planet_{kind}_month_{idx:02d}.png"
        if not candidate.exists():
            return None
        label = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"][idx - 1]
        entries.append({
            "month": idx,
            "label": label,
            "path": str(candidate),
            "view_url": _map_view_url(candidate, output_dir),
            "raw_url": _file_url(candidate),
        })
    return {"kind": kind, "month": month, "entries": entries}


def _monthly_sequence_controls(path: Path, output_dir: Path | None) -> str:
    ctx = _monthly_sequence_context(path, output_dir)
    if not ctx:
        return ""
    month = int(ctx["month"])
    kind = str(ctx["kind"])
    entries = ctx["entries"]
    options = "".join(
        f"<option value='{item['month']}'{' selected' if int(item['month']) == month else ''}>{_safe_text(item['label'])}</option>"
        for item in entries
    )
    prev_month = ((month + 10) % 12) + 1
    next_month = (month % 12) + 1
    payload = json.dumps({
        str(item["month"]): {
            "view_url": item["view_url"],
            "raw_url": item["raw_url"],
            "path": item["path"],
            "label": f"climate_monthly/main_planet_{kind}_month_{int(item['month']):02d}.png",
            "month_label": item["label"],
        }
        for item in entries
    })
    other_kind = "precipitation" if kind == "temperature" else "temperature"
    other = path.parent / f"main_planet_{other_kind}_month_{month:02d}.png"
    other_link = f"<a class='button secondary' href='{_map_view_url(other, output_dir)}'>Switch to {other_kind}</a>" if other.exists() else ""
    return f"""
<section class='card monthly-sequence-card sticky-tools'>
  <div class='monthly-sequence-head'>
    <strong>Monthly {kind} progression</strong>
    <span id='monthly-sequence-readout' class='muted small'>Current: {month:02d} / {_safe_text(entries[month-1]['label'])}</span>
  </div>
  <div class='row monthly-sequence-row'>
    <button type='button' class='secondary' onclick='wgMonthlyStep(-1)'>Previous month</button>
    <button type='button' class='secondary' onclick='wgMonthlyStep(1)'>Next month</button>
    <label class='small'>Month
      <select id='monthly-sequence-select' onchange='wgSetMonthlyClimate(this.value)' style='width:120px'>{options}</select>
    </label>
    <input id='monthly-sequence-range' type='range' min='1' max='12' step='1' value='{month}' oninput='wgSetMonthlyClimate(this.value)' style='width:260px'>
    <button type='button' class='secondary' id='monthly-play-button' onclick='wgMonthlyTogglePlay()'>Play</button>
    <label class='small'>Speed
      <select id='monthly-play-speed' style='width:105px'>
        <option value='1400'>slow</option>
        <option value='800' selected>normal</option>
        <option value='350'>fast</option>
      </select>
    </label>
    {other_link}
    <span class='muted small'>Month changes now update the image in place, preserving zoom, pan, and scroll.</span>
  </div>
</section>
<script>
window.WG_MONTHLY_CONTEXT = {{kind: {json.dumps(kind)}, currentMonth: {month}, urls: {payload}}};
</script>
"""


def _monthly_viewer_overlay(path: Path, output_dir: Path | None) -> str:
    ctx = _monthly_sequence_context(path, output_dir)
    if not ctx:
        return ""
    kind = str(ctx["kind"])
    other_kind = "precipitation" if kind == "temperature" else "temperature"
    month = int(ctx["month"])
    other = path.parent / f"main_planet_{other_kind}_month_{month:02d}.png"
    other_link = f"<a class='button secondary small' href='{_map_view_url(other, output_dir)}'>Switch to {other_kind}</a>" if other.exists() else ""
    return f"""
<div class='viewer-overlay-tools'>
  <strong class='small'>Monthly {kind}</strong>
  <button type='button' class='secondary small' onclick='wgMonthlyStep(-1)'>◀</button>
  <button type='button' class='secondary small' onclick='wgMonthlyStep(1)'>▶</button>
  <button type='button' class='secondary small' onclick='wgMonthlyTogglePlay()'>Play / pause</button>
  {other_link}
  <span class='muted small'>Floating monthly controls stay reachable while scrolling inside the map.</span>
</div>
"""


def _is_inside_output_dir(path: Path, output_dir: Path) -> bool:
    try:
        path.resolve().relative_to(output_dir.resolve())
        return True
    except Exception:
        return False


def _run_image_inventory(output_dir: Path, *, include_regions: bool = True) -> list[Path]:
    """Recursively discover every generated image in a run folder.

    The old Available Maps panel only looked at the root/maps folder, which hid
    terrain diagnostics, regional terrain crops, preview images, and future map
    folders. This inventory is deliberately path-based and category-agnostic so
    newly added diagnostic images appear automatically.
    """
    if not output_dir.exists():
        return []
    skip_dirs = {"__pycache__", ".git", ".venv", "venv", "env", "node_modules"}
    images: list[Path] = []
    for path in output_dir.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        rel_parts = path.relative_to(output_dir).parts
        if any(part in skip_dirs for part in rel_parts):
            continue
        if not include_regions and rel_parts and rel_parts[0] == "terrain_regions":
            continue
        images.append(path)
    seen: set[str] = set()
    out: list[Path] = []
    for path in sorted(images, key=lambda item: _rel_label(item, output_dir).lower()):
        key = str(path.resolve())
        if key not in seen:
            seen.add(key)
            out.append(path)
    return out


def _all_image_maps(output_dir: Path, *, include_regions: bool = False) -> list[Path]:
    # Use the recursive inventory for selectors/viewers so every generated map
    # or diagnostic image can be viewed/compared without hand-editing a curated
    # list whenever a new output is added.
    return _run_image_inventory(output_dir, include_regions=include_regions)


def _image_category(path: Path, output_dir: Path) -> str:
    rel = _rel_label(path, output_dir).replace("\\", "/")
    name = path.name.lower()
    parts = rel.split("/")
    if parts and parts[0] == "terrain_diagnostics":
        if len(parts) >= 2:
            folder = parts[1]
            labels = {
                "01_foundation": "Terrain diagnostics / 01 foundation",
                "02_provinces": "Terrain diagnostics / 02 provinces",
                "03_boundaries": "Terrain diagnostics / 03 boundaries",
                "04_mountains_basins": "Terrain diagnostics / 04 mountains, basins, rifts",
                "05_coasts_islands": "Terrain diagnostics / 05 coasts, shelves, islands",
                "06_erosion_deposition": "Terrain diagnostics / 06 erosion and deposition",
                "07_final": "Terrain diagnostics / 07 final terrain",
            }
            return labels.get(folder, f"Terrain diagnostics / {folder}")
        return "Terrain diagnostics"
    if parts and parts[0] == "terrain_regions":
        return "Terrain regions"
    if parts and parts[0] == "climate_monthly":
        return "Climate monthly progression"
    if parts and parts[0] == "terrain_previews":
        return "Terrain previews"
    if name in {"system_orbits.png", "system_sizes.png", "main_planet_moon.png"} or name.startswith("system_"):
        return "Solar system"
    terrain_tokens = ("terrain", "tectonic", "plate", "crust", "coastline", "inland_lakes", "islands", "erosion")
    if any(token in name for token in terrain_tokens):
        return "Terrain main maps"
    climate_tokens = ("temperature", "precipitation", "koppen", "wind", "ocean_currents", "ocean_gyres", "moisture", "rain_shadow", "itcz", "pressure", "aridity", "orographic", "circulation", "lake_moisture", "small_lake")
    if any(token in name for token in climate_tokens):
        return "Climate"
    hydrology_tokens = ("hydrology", "drainage", "delta", "river", "basin")
    if any(token in name for token in hydrology_tokens):
        return "Hydrology"
    if "biome" in name:
        return "Biomes"
    if "region" in name:
        return "Regions"
    if parts and parts[0] == "maps":
        return "Maps folder"
    return "Other images"


def _grouped_image_inventory(output_dir: Path, *, include_regions: bool = True) -> dict[str, list[Path]]:
    groups: dict[str, list[Path]] = {}
    for path in _run_image_inventory(output_dir, include_regions=include_regions):
        if _is_hidden_map_path(path, output_dir):
            continue
        groups.setdefault(_image_category(path, output_dir), []).append(path)
    preferred = [
        "Solar system",
        "Terrain main maps",
        "Terrain diagnostics / 01 foundation",
        "Terrain diagnostics / 02 provinces",
        "Terrain diagnostics / 03 boundaries",
        "Terrain diagnostics / 04 mountains, basins, rifts",
        "Terrain diagnostics / 05 coasts, shelves, islands",
        "Terrain diagnostics / 06 erosion and deposition",
        "Terrain diagnostics / 07 final terrain",
        "Terrain previews",
        "Terrain regions",
        "Climate",
        "Climate monthly progression",
        "Hydrology",
        "Biomes",
        "Regions",
        "Maps folder",
        "Other images",
    ]
    ordered: dict[str, list[Path]] = {}
    for name in preferred:
        if name in groups:
            ordered[name] = sorted(groups.pop(name), key=lambda item: _rel_label(item, output_dir).lower())
    for name in sorted(groups):
        ordered[name] = sorted(groups[name], key=lambda item: _rel_label(item, output_dir).lower())
    return ordered


def _map_manifest_rows(output_dir: Path) -> list[dict[str, Any]]:
    rows = _read_json(output_dir / "diagnostics" / "map_manifest.json", [])
    if isinstance(rows, list) and rows:
        out: list[dict[str, Any]] = []
        for row in rows:
            if isinstance(row, dict):
                out.append(dict(row))
        if out:
            return out
    # Fallback for older runs: synthesize a minimal row set from the registry and recursive inventory.
    present = {str(_rel_label(path, output_dir)).replace('\\', '/'): path for path in _run_image_inventory(output_dir, include_regions=True)}
    fallback: list[dict[str, Any]] = []
    for item in MAP_REGISTRY:
        row = dict(item)
        filename = str(row.get('filename', '') or '')
        pattern = str(row.get('pattern', '') or '')
        if filename and filename in present:
            row.update({'status': 'generated', 'generated_count': 1, 'applicable': 'unknown', 'reason': 'Generated file found, but this run predates the saved manifest.'})
        elif pattern:
            matches = list(output_dir.glob(pattern))
            count = len([m for m in matches if m.is_file()])
            row.update({'status': 'generated' if count else 'unknown', 'generated_count': count, 'applicable': 'unknown', 'reason': 'Manifest missing; this is a best-effort fallback from the current filesystem.'})
        else:
            row.update({'status': 'unknown', 'generated_count': 0, 'applicable': 'unknown', 'reason': 'Manifest missing; rerun export to populate full map-index status.'})
        fallback.append(row)
    return fallback


def _map_status_badge(status: str) -> str:
    label = str(status or 'unknown')
    cls = 'ok' if label == 'generated' else ('warn' if label in {'partial', 'missing'} else ('muted' if label in {'not_applicable', 'unknown'} else 'muted'))
    return f"<span class='pill {cls}-pill'>{_safe_text(label.replace('_', ' '))}</span>"


def _map_index_link_for_row(output_dir: Path, row: dict[str, Any]) -> tuple[str, str]:
    filename = str(row.get('filename', '') or '')
    pattern = str(row.get('pattern', '') or '')
    if filename:
        path = output_dir / filename
        if path.exists():
            return 'View', _map_view_url(path, output_dir)
    if pattern:
        matches = sorted([p for p in output_dir.glob(pattern) if p.is_file()], key=lambda item: _rel_label(item, output_dir).lower())
        if matches:
            return 'View sample', _map_view_url(matches[0], output_dir)
    return '', ''


def _map_index_summary_html(output_dir: Path) -> str:
    rows = _map_manifest_rows(output_dir)
    if not rows:
        return "<p class='muted'>No map registry/manifest is available yet.</p>"
    counts = {'generated': 0, 'partial': 0, 'missing': 0, 'not_applicable': 0, 'unknown': 0}
    for row in rows:
        counts[str(row.get('status', 'unknown'))] = counts.get(str(row.get('status', 'unknown')), 0) + 1
    return (
        f"<div class='row'>"
        f"<span class='pill ok-pill'>generated {counts.get('generated', 0)}</span>"
        f"<span class='pill warn-pill'>partial {counts.get('partial', 0)}</span>"
        f"<span class='pill bad-pill'>missing {counts.get('missing', 0)}</span>"
        f"<span class='pill'>not applicable {counts.get('not_applicable', 0)}</span>"
        f"<span class='pill'>unknown {counts.get('unknown', 0)}</span>"
        f"</div>"
    )


def _map_index_table_html(output_dir: Path, *, stage_filter: str | None = None, show_hidden: bool = False) -> str:
    rows = _map_manifest_rows(output_dir)
    if stage_filter:
        rows = [row for row in rows if str(row.get('stage', '')) == stage_filter]
    if not show_hidden:
        rows = [row for row in rows if not (_truthy_meta(row.get('hidden')) or _truthy_meta(row.get('deprecated')))]
    if not rows:
        return "<p class='muted'>No registered maps match this filter.</p>"
    rows = sorted(rows, key=lambda row: (str(row.get('group', '')), str(row.get('stage', '')), str(row.get('key', ''))))
    parts = ["<div class='table-wrap'><table><tr><th>Map / family</th><th>Group</th><th>Stage</th><th>Status</th><th>Generated</th><th>Description</th><th>Reason</th><th>Open</th></tr>"]
    for row in rows:
        label = str(row.get('filename') or row.get('pattern') or row.get('key') or '')
        group = str(row.get('group', '') or '')
        stage = str(row.get('stage', '') or '')
        status = str(row.get('status', 'unknown') or 'unknown')
        family_count = row.get('family_count')
        generated_count = row.get('generated_count', 0)
        generated_txt = str(generated_count)
        if family_count not in {None, '', 0}:
            generated_txt += f" / {int(family_count)}"
        link_label, link_url = _map_index_link_for_row(output_dir, row)
        open_cell = f"<a class='button secondary small' href='{link_url}'>{_safe_text(link_label)}</a>" if link_url else "<span class='muted'>—</span>"
        parts.append(
            f"<tr>"
            f"<td><code>{_safe_text(label)}</code><div class='muted small'>{_safe_text(str(row.get('key', '')))}</div></td>"
            f"<td>{_safe_text(group)}</td>"
            f"<td>{_safe_text(stage)}</td>"
            f"<td>{_map_status_badge(status)}</td>"
            f"<td>{_safe_text(generated_txt)}</td>"
            f"<td>{_safe_text(str(row.get('description', '') or ''))}</td>"
            f"<td>{_safe_text(str(row.get('reason', '') or ''))}</td>"
            f"<td>{open_cell}</td>"
            f"</tr>"
        )
    parts.append('</table></div>')
    return ''.join(parts)


def _map_card_html(path: Path, output_dir: Path) -> str:
    rel = _rel_label(path, output_dir)
    raw_url = _file_url(path)
    view_url = _map_view_url(path, output_dir)
    compare_url = "/compare?" + urllib.parse.urlencode({"output_dir": str(output_dir), "left": rel})
    globe_url = "/globe?" + urllib.parse.urlencode({"output_dir": str(output_dir), "path": str(path)}) if _is_probable_world_map(path) else ""
    globe_link = f"<a href='{globe_url}'>Globe</a>" if globe_url else ""
    category = _image_category(path, output_dir)
    return (
        f"<div class='mapcard' data-map-name='{_safe_text(rel.lower())}' data-map-category='{_safe_text(category.lower())}'>"
        f"<a href='{view_url}'><img src='{raw_url}' loading='lazy'></a>"
        f"<p>{_safe_text(rel)}</p><span class='map-category-pill'>{_safe_text(category)}</span>"
        f"<div class='row small'><a href='{view_url}'>View</a><a href='{compare_url}'>Compare</a>{globe_link}<a href='{raw_url}' target='_blank'>Raw</a></div></div>"
    )


def _map_inventory_html(output_dir: Path, *, stage: str | None = None, full_inventory: bool = True) -> str:
    if not output_dir.exists():
        return "<p class='muted'>Run folder does not exist.</p>"
    groups = _grouped_image_inventory(output_dir, include_regions=True)
    if not groups:
        return "<p class='muted'>No maps/images available yet. Run solar-system or later stages with images enabled.</p>"
    if stage is not None and not full_inventory:
        wanted = _stage_map_categories(stage)
        groups = {name: paths for name, paths in groups.items() if name in wanted}
        if not groups:
            return "<p class='muted'>No stage-specific maps/images are available yet for this stage.</p>"
    total = sum(len(paths) for paths in groups.values())
    category_options = "".join(f"<option value='{_safe_text(name.lower())}'>{_safe_text(name)} ({len(paths)})</option>" for name, paths in groups.items())
    sections: list[str] = []
    for name, paths in groups.items():
        cards = "".join(_map_card_html(path, output_dir) for path in paths)
        # Keep high-cardinality folders closed by default so 128 regional maps do not swamp the main view.
        open_attr = " open" if (name in {"Climate", "Climate monthly progression"} or (name in {"Solar system", "Terrain main maps", "Hydrology", "Biomes", "Regions"} and len(paths) <= 30)) else ""
        sections.append(f"""
<details class='folder-card map-inventory-group'{open_attr} data-map-group='{_safe_text(name.lower())}'>
  <summary>{_safe_text(name)} — {len(paths)} image{'s' if len(paths) != 1 else ''}</summary>
  <div class='maps'>{cards}</div>
</details>
""")
    return f"""
<div class='map-browser' data-total-maps='{total}'>
  <div class='map-browser-controls'>
    <label>Search maps/images<input type='search' id='map-search' placeholder='terrain, koppen, r01_c04, boundary...' oninput='filterMapInventory()'></label>
    <label>Category<select id='map-category-filter' onchange='filterMapInventory()'><option value=''>All categories ({total})</option>{category_options}</select></label>
    <label class='inline-check'><input type='checkbox' id='map-show-diagnostics' checked onchange='filterMapInventory()' style='width:auto'> show diagnostics</label>
    <label class='inline-check'><input type='checkbox' id='map-show-regions' onchange='filterMapInventory()' style='width:auto'> show terrain regions</label>
  </div>
  <p class='help'>This inventory scans the whole run folder for generated images, not just the curated top-level map list. Diagnostic and regional folders are grouped and usually collapsed by default.</p>
  {''.join(sections)}
</div>
<script>
function filterMapInventory() {{
  const q = (document.getElementById('map-search')?.value || '').toLowerCase();
  const cat = (document.getElementById('map-category-filter')?.value || '').toLowerCase();
  const showDiag = document.getElementById('map-show-diagnostics')?.checked !== false;
  const showRegions = document.getElementById('map-show-regions')?.checked === true;
  document.querySelectorAll('.map-inventory-group').forEach(group => {{
    const groupName = (group.dataset.mapGroup || '').toLowerCase();
    const isDiag = groupName.includes('diagnostics');
    const isRegion = groupName.includes('terrain regions');
    let any = false;
    group.querySelectorAll('.mapcard').forEach(card => {{
      const name = (card.dataset.mapName || '').toLowerCase();
      const cardCat = (card.dataset.mapCategory || '').toLowerCase();
      const ok = (!q || name.includes(q)) && (!cat || cardCat === cat) && (showDiag || !isDiag) && (showRegions || !isRegion);
      card.style.display = ok ? '' : 'none';
      if (ok) any = true;
    }});
    group.style.display = any ? '' : 'none';
  }});
}}
filterMapInventory();
</script>
"""


def _stage_map_categories(stage: str) -> set[str]:
    stage = normalize_stage(stage)
    if stage == "solar-system":
        return {"Solar system"}
    if stage == "planet-physics":
        return {"Solar system"}
    if stage == "terrain-foundation-mask":
        return {"Terrain main maps", "Terrain diagnostics / 01 foundation"}
    if stage == "terrain-tectonic-provinces":
        return {"Terrain main maps", "Terrain diagnostics / 02 provinces"}
    if stage == "terrain-crust-and-boundaries":
        return {"Terrain main maps", "Terrain diagnostics / 03 boundaries"}
    if stage == "terrain-mountains-basins-rifts":
        return {"Terrain main maps", "Terrain diagnostics / 04 mountains, basins, rifts"}
    if stage == "terrain-coasts-shelves-islands":
        return {"Terrain main maps", "Terrain diagnostics / 05 coasts, shelves, islands"}
    if stage == "terrain-erosion-deposition":
        return {"Terrain main maps", "Terrain diagnostics / 06 erosion and deposition"}
    if stage == "terrain-finalization-recentering":
        return {"Terrain main maps", "Terrain diagnostics / 07 final terrain", "Terrain regions"}
    if stage == "climate":
        return {"Climate", "Climate monthly progression", "Terrain main maps"}
    if stage == "hydrology":
        return {"Hydrology", "Climate", "Climate monthly progression", "Terrain main maps"}
    if stage == "biomes":
        return {"Biomes", "Climate", "Climate monthly progression", "Hydrology", "Terrain main maps"}
    if stage == "regions":
        return {"Regions", "Biomes", "Hydrology", "Climate", "Climate monthly progression", "Terrain main maps"}
    return {"Solar system", "Terrain main maps", "Climate", "Climate monthly progression", "Hydrology", "Biomes", "Regions", "Other images"}


def _file_groups(output_dir: Path) -> dict[str, list[Path]]:
    all_images = _run_image_inventory(output_dir, include_regions=True)
    climate_images = [p for p in all_images if _image_category(p, output_dir) in {"Climate", "Climate monthly progression"}]
    groups: dict[str, list[Path]] = {
        "All generated images": all_images,
        "Climate diagnostic images": climate_images,
        "Primary/curated maps": _available_maps(output_dir),
        "Terrain diagnostics maps": [p for group in _terrain_diagnostic_image_groups(output_dir).values() for p in group],
        "Regional terrain maps": _terrain_region_maps(output_dir),
        "Diagnostics": _diagnostic_files(output_dir),
        "State": _state_files(output_dir),
        "Config": sorted((output_dir / "config").glob("*")) if (output_dir / "config").exists() else [],
        "Job logs": sorted((output_dir / "diagnostics" / "webui_jobs").glob("*")) if (output_dir / "diagnostics" / "webui_jobs").exists() else [],
    }
    return {name: [p for p in paths if p.is_file()] for name, paths in groups.items()}




def _fmt_duration(seconds: float | None) -> str:
    if seconds is None:
        return "—"
    try:
        seconds = float(seconds)
    except Exception:
        return "—"
    if seconds < 1:
        return f"{seconds:.2f}s"
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, sec = divmod(seconds, 60)
    if minutes < 60:
        return f"{int(minutes)}m {int(sec)}s"
    hours, minutes = divmod(minutes, 60)
    return f"{int(hours)}h {int(minutes)}m"


def _stage_timing_html(output_dir: Path) -> str:
    """Show latest per-stage durations from diagnostics/pipeline_stage_log.csv.

    Until terrain is truly internally split, the UI deliberately shows terrain as
    one row.  Showing seven identical terrain sub-stage timings looked more
    precise than the data really is.
    """
    path = output_dir / "diagnostics" / "pipeline_stage_log.csv"
    latest: dict[str, dict[str, Any]] = {}
    if path.exists():
        try:
            with path.open(newline="", encoding="utf-8") as fh:
                for row in csv.DictReader(fh):
                    stage = row.get("stage") or ""
                    if stage:
                        latest[stage] = row
        except Exception as exc:
            return f"<section class='card'><h2>Stage timings</h2><p class='bad'>Could not read stage log: {_safe_text(exc)}</p></section>"

    # Terrain is currently monolithic.  Pull the best available shared terrain
    # timing from either the terrain stage log or any terrain subphase marker.
    terrain_row: dict[str, Any] | None = latest.get("terrain")
    terrain_elapsed: float | None = None
    if terrain_row is not None:
        try:
            terrain_elapsed = float(terrain_row.get("elapsed_s", ""))
        except Exception:
            terrain_elapsed = None
    for stage in TERRAIN_SUBPHASES:
        marker = output_dir / ("state/" + {
            "terrain-foundation-mask": "03a_terrain_foundation_mask.json",
            "terrain-tectonic-provinces": "03b_terrain_tectonic_provinces.json",
            "terrain-crust-and-boundaries": "03c_terrain_crust_and_boundaries.json",
            "terrain-mountains-basins-rifts": "03d_terrain_mountains_basins_rifts.json",
            "terrain-coasts-shelves-islands": "03e_terrain_coasts_shelves_islands.json",
            "terrain-erosion-deposition": "03f_terrain_erosion_deposition.json",
            "terrain-finalization-recentering": "03g_terrain_finalization_recentering.json",
        }[stage])
        data = _read_json(marker, {})
        if isinstance(data, dict) and data.get("elapsed_s") is not None:
            try:
                val = float(data.get("elapsed_s"))
                if terrain_elapsed is None or val > terrain_elapsed:
                    terrain_elapsed = val
                    terrain_row = {
                        "timestamp": data.get("completed_at", ""),
                        "stage": "terrain",
                        "status": "complete",
                        "elapsed_s": str(val),
                        "note": "shared monolithic terrain pass; internal sub-stage timing not available yet",
                    }
            except Exception:
                pass
    if terrain_row is not None:
        latest["terrain"] = terrain_row
    for stage in TERRAIN_SUBPHASES:
        latest.pop(stage, None)

    if not latest:
        return "<section class='card'><h2>Stage timings</h2><p class='muted'>No stage timing log yet. Run stages through the pipeline to record elapsed time.</p></section>"

    durations: dict[str, float] = {}
    for stage, row in latest.items():
        try:
            val = row.get("elapsed_s", "")
            if val not in {"", None}:
                durations[stage] = max(0.0, float(val))
        except Exception:
            pass
    if not durations:
        return "<section class='card'><h2>Stage timings</h2><p class='muted'>Stage log exists, but no elapsed times have been recorded yet.</p></section>"

    total_s = sum(durations.values())
    max_s = max(durations.values()) or 1.0
    display_order = ["solar-system", "planet-physics", "terrain", "climate", "hydrology", "biomes", "regions", "outputs"]
    rows = ""
    for stage in display_order:
        if stage not in latest:
            continue
        row = latest.get(stage, {})
        elapsed = durations.get(stage)
        width = 0.0 if elapsed is None else max(2.0, min(100.0, elapsed / max_s * 100.0))
        pct = 0.0 if elapsed is None or total_s <= 0 else elapsed / total_s * 100.0
        status = row.get("status", "")
        note = row.get("note", "") or ""
        if stage == "terrain":
            note = (note + "; " if note else "") + "shown as one row until terrain processing is actually split"
        rows += (
            f"<tr><td>{_safe_text(stage)}</td><td>{_safe_text(status)}</td>"
            f"<td>{_safe_text(_fmt_duration(elapsed))}</td><td>{pct:.1f}%</td>"
            f"<td><div class='timing-bar'><span style='width:{width:.1f}%'></span></div></td>"
            f"<td class='muted small'>{_safe_text(note)}</td></tr>"
        )
    return f"""
<section class='card' style='margin-top:18px'>
  <h2>Stage timings</h2>
  <p class='muted small'>Total recorded stage time: {_safe_text(_fmt_duration(total_s))}. Terrain is intentionally shown as one row until the terrain generator is internally split into true timed sub-stages.</p>
  <table><thead><tr><th>Stage</th><th>Status</th><th>Time</th><th>Share</th><th>Visual</th><th>Note</th></tr></thead><tbody>{rows}</tbody></table>
</section>
"""

def _log_summary(path: Path) -> str:
    text = _tail(path, max_chars=6000)
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return "No log output yet."
    # Prefer the last meaningful pipeline line before the footer if available.
    for line in reversed(lines):
        if not line.lower().startswith(("finished:", "return code:")):
            return line[-300:]
    return lines[-1][-300:]


def _water_preference_range_text(preference: Any) -> str:
    pref = str(preference or "earthlike")
    if pref == "dry_terrestrial":
        return "Preferred water/volatile fraction for dry terrestrial: 0.003–0.018. Overall Main Planet accepted range: 0.002–0.090."
    if pref == "oceanic":
        return "Preferred water/volatile fraction for oceanic: 0.025–0.075. Overall Main Planet accepted range: 0.002–0.090."
    return f"Preferred water/volatile fraction for {pref}: 0.006–0.040. Overall Main Planet accepted range: 0.002–0.090."


def _stage1_factor_li(item: Any, preference: Any = None) -> str:
    text = str(item)
    extra = ""
    if "water/volatile fraction" in text and "preferred" not in text and "accepted" not in text:
        extra = f"<div class='field-help'>{_safe_text(_water_preference_range_text(preference))}</div>"
    if "gravity" in text:
        extra += "<div class='field-help'>Gravity is best near 0.85–1.25 g; the broader accepted Main Planet range is configured in the generator.</div>"
    if "flux" in text or "habitable zone" in text:
        extra += "<div class='field-help'>Flux near 1.0 S⊕ is Earthlike; edge-of-HZ values are allowed but should be reviewed downstream.</div>"
    return f"<li>{_safe_text(text)}{extra}</li>"

def _stage1_report_html(data: dict[str, Any]) -> str:
    diagnostics = data.get("diagnostics", {}) if isinstance(data.get("diagnostics"), dict) else {}
    report = diagnostics.get("system_report", []) if isinstance(diagnostics.get("system_report"), list) else []
    explanation = diagnostics.get("habitability_explanation", {}) if isinstance(diagnostics.get("habitability_explanation"), dict) else {}
    warnings = diagnostics.get("stage1_warnings", []) if isinstance(diagnostics.get("stage1_warnings"), list) else []
    debris = diagnostics.get("debris_belt_profile", {}) if isinstance(diagnostics.get("debris_belt_profile"), dict) else {}
    giant_history = diagnostics.get("giant_planet_history", {}) if isinstance(diagnostics.get("giant_planet_history"), dict) else {}
    report_items = "".join(f"<p>{_safe_text(line)}</p>" for line in report) or "<p class='muted'>No Stage 1 report is available for this run yet. Rerun the solar-system stage to generate one.</p>"
    positives = explanation.get("positive_factors", []) if isinstance(explanation.get("positive_factors"), list) else []
    concerns = explanation.get("concerns", []) if isinstance(explanation.get("concerns"), list) else []
    preference = diagnostics.get("main_planet_preference")
    pos_html = "".join(_stage1_factor_li(item, preference) for item in positives) or "<li class='muted'>No positive-factor details recorded.</li>"
    con_html = "".join(_stage1_factor_li(item, preference) for item in concerns) or "<li class='muted'>No candidate concerns recorded.</li>"
    warn_html = "".join(
        f"<li><span class='{ 'warn' if str(item.get('level','')).lower() == 'warning' else 'muted' }'>{_safe_text(item.get('level','notice'))}</span>: {_safe_text(item.get('message',''))}</li>"
        for item in warnings if isinstance(item, dict)
    ) or "<li class='muted'>No Stage 1 warnings/notices recorded.</li>"
    context_html = f"""
      <div class='grid compact-grid'>
        {_mapping_stat_items(debris, ['type', 'activity', 'inner_edge_au', 'outer_edge_au', 'impact_delivery_bias', 'note'])}
        {_mapping_stat_items(giant_history, ['mode', 'giant_count', 'perturbation_level', 'resonance_flavor', 'note'])}
      </div>
    """
    return f"""
<section class='card stage-report' style='margin-top:18px'>
  <h2>Stage 1 report</h2>
  {report_items}
  <div class='grid compact-grid' style='margin-top:12px'>
    <section class='card'><h3>Why this Main Planet?</h3><p>{_safe_text(explanation.get('summary','No explanation recorded.'))}</p><h4>Positive factors</h4><ul>{pos_html}</ul></section>
    <section class='card'><h3>Concerns / review flags</h3><ul>{con_html}</ul><h4>Warnings</h4><ul>{warn_html}</ul></section>
  </div>
  <h3>Optional system context</h3>
  {context_html}
</section>
"""


def _candidate_orbit_strip(star: dict[str, Any], cand: dict[str, Any]) -> str:
    inner = _safe_float(star.get("habitable_zone_inner_au"))
    outer = _safe_float(star.get("habitable_zone_outer_au"))
    au = _safe_float(cand.get("semi_major_axis_au"))
    if inner is None or outer is None or au is None or outer <= inner:
        return "<p class='muted small'>Orbit / habitable-zone placement unavailable.</p>"
    low = max(0.01, inner * 0.55)
    high = max(outer * 1.7, au * 1.05, 0.1)
    pct = max(0.0, min(100.0, (au - low) / (high - low) * 100.0))
    h1 = max(0.0, min(100.0, (inner - low) / (high - low) * 100.0))
    h2 = max(0.0, min(100.0, (outer - low) / (high - low) * 100.0))
    status = "inside habitable zone" if inner <= au <= outer else ("inside inner edge" if au < inner else "outside outer edge")
    return f"""
      <div class='hz-strip' title='Candidate orbit {au:.3f} AU; habitable zone {inner:.3f}–{outer:.3f} AU'>
        <div class='hz-band' style='left:{h1:.1f}%;width:{max(1.5,h2-h1):.1f}%'></div>
        <div class='hz-marker' style='left:{pct:.1f}%'></div>
      </div>
      <p class='muted small'>Orbit {au:.3f} AU · {status}</p>
    """


def _candidate_card_html(cand: dict[str, Any], star: dict[str, Any], preference: Any) -> str:
    selected = bool(cand.get("selected"))
    eligible = bool(cand.get("eligible"))
    cls = "main-candidate" if selected else ("eligible-candidate" if eligible else "rejected-candidate")
    status = str(cand.get("status") or ("selected" if selected else ("eligible" if eligible else "rejected")))
    score = _safe_float(cand.get("habitability_score"))
    score_text = "—" if score is None else f"{score:.1f}"
    reason = cand.get("selected_or_rejected_reason", "")
    positives = cand.get("positive_factors", []) if isinstance(cand.get("positive_factors"), list) else []
    concerns = cand.get("concerns", []) if isinstance(cand.get("concerns"), list) else []
    factor_html = ""
    if positives or concerns:
        factor_html = "<details><summary>Scoring factors</summary>"
        if positives:
            factor_html += "<b>Positive</b><ul>" + "".join(_stage1_factor_li(x, preference) for x in positives[:8]) + "</ul>"
        if concerns:
            factor_html += "<b>Concerns</b><ul>" + "".join(_stage1_factor_li(x, preference) for x in concerns[:8]) + "</ul>"
        factor_html += "</details>"
    main_badge = "<span class='pill complete'>Main Planet</span>" if selected else ""
    eligible_badge = "<span class='pill'>Eligible</span>" if eligible and not selected else ("<span class='pill missing'>Rejected</span>" if not selected else "")
    role = _humanize_key(cand.get("architecture_role"))
    hz_pos = cand.get("hz_position") or "unknown HZ position"
    composition = cand.get("composition_class") or "unknown composition"
    planet_class = cand.get("class") or cand.get("planet_class") or "planet"
    title = (
        f"{cand.get('name','Planet')}: score {score_text}. "
        f"Class {planet_class}; role {role}; HZ position {hz_pos}. "
        f"Selected as Main Planet." if selected else
        f"{cand.get('name','Planet')}: score {score_text}. Class {planet_class}; role {role}; HZ position {hz_pos}."
    )
    return f"""
      <article class='candidate-card {cls}' title='{_safe_text(title)}'>
        <div class='candidate-head'>
          <div class='candidate-title'>
            {_planet_visual(planet_class, selected)}
            <div><strong>{_safe_text(cand.get('name',''))}</strong><small>{_safe_text(_humanize_key(status))}</small></div>
          </div>
          <div class='candidate-score'><b>{_safe_text(score_text)}</b><small class='muted'>score / 100</small></div>
        </div>
        <div class='candidate-tags'>
          {main_badge}{eligible_badge}
          <span class='pill'>{_safe_text(planet_class)}</span>
          <span class='pill'>{_safe_text(role)}</span>
          <span class='pill'>{_safe_text(composition)}</span>
          <span class='pill'>{_safe_text(hz_pos)}</span>
        </div>
        <div class='candidate-viz'>
          {_candidate_orbit_strip(star, cand)}
          <div class='candidate-metrics'>
            {_metric_gauge('Score', cand.get('habitability_score'), min_v=0, max_v=100, low='poor', mid='possible', high='strong')}
            {_metric_gauge('Flux', cand.get('stellar_flux_earth'), unit='S⊕', min_v=0.25, max_v=2.2, low='cold/dim', mid='temperate', high='hot/bright')}
            {_metric_gauge('Gravity', cand.get('gravity_g'), unit='g', min_v=0.25, max_v=2.2, low='low', mid='comfortable', high='high')}
            {_metric_gauge('Water/volatile fraction', cand.get('water_fraction'), min_v=0, max_v=0.25, low='dry: ~0–0.02', mid='mixed: ~0.02–0.08', high='wet/oceanic: >0.08')}
          </div>
        </div>
        <div class='candidate-tags'>
          <span class='pill'>AU {_safe_text(_fmt_num(cand.get('semi_major_axis_au'), 3))}</span>
          <span class='pill'>M⊕ {_safe_text(_fmt_num(cand.get('mass_earth'), 2))}</span>
          <span class='pill'>R⊕ {_safe_text(_fmt_num(cand.get('radius_earth'), 2))}</span>
          <span class='pill'>T_eq {_safe_text(_fmt_num(cand.get('equilibrium_temperature_k'), 1))} K</span>
        </div>
        <div class='candidate-reason'><b>Selection note:</b> {_safe_text(reason or 'No reason recorded.')}</div>
        {factor_html}
      </article>
    """


def _stage1_candidate_table_html(data: dict[str, Any]) -> str:
    diagnostics = data.get("diagnostics", {}) if isinstance(data.get("diagnostics"), dict) else {}
    candidates = diagnostics.get("main_planet_candidates", []) if isinstance(diagnostics.get("main_planet_candidates"), list) else []
    star = data.get("star", {}) if isinstance(data.get("star"), dict) else {}
    if not candidates:
        planets = data.get("planets", []) if isinstance(data.get("planets"), list) else []
        candidates = []
        for planet in planets:
            if not isinstance(planet, dict):
                continue
            orbit = planet.get("orbit", {}) if isinstance(planet.get("orbit"), dict) else {}
            candidates.append({
                "name": planet.get("name"),
                "selected": bool(planet.get("is_main_planet")),
                "eligible": bool(planet.get("habitability_score", 0)),
                "status": "selected" if planet.get("is_main_planet") else "planet",
                "class": planet.get("planet_class"),
                "architecture_role": planet.get("architecture_role"),
                "composition_class": planet.get("composition", {}).get("class") if isinstance(planet.get("composition"), dict) else "",
                "hz_position": "unknown",
                "semi_major_axis_au": orbit.get("semi_major_axis_au"),
                "mass_earth": planet.get("mass_earth"),
                "radius_earth": planet.get("radius_earth"),
                "stellar_flux_earth": planet.get("stellar_flux_earth"),
                "equilibrium_temperature_k": planet.get("equilibrium_temperature_k"),
                "gravity_g": planet.get("surface_gravity_g"),
                "water_fraction": planet.get("water_fraction"),
                "habitability_score": planet.get("habitability_score"),
                "selected_or_rejected_reason": "Planet generated before candidate diagnostics were available.",
            })
    if not candidates:
        return "<p class='muted'>No planet/candidate cards are available for this run yet. Rerun the solar-system stage to generate them.</p>"
    ordered = sorted(
        [c for c in candidates if isinstance(c, dict)],
        key=lambda c: (not bool(c.get("selected")), -float(c.get("habitability_score") or 0.0)),
    )
    cards = "".join(_candidate_card_html(cand, star, diagnostics.get("main_planet_preference")) for cand in ordered)
    return f"""
<p class='help'>This card view merges the old Planets and Main Planet candidate tables. Each card shows orbit placement, score, climate-relevant gauges, key physical values, and the reason the world was selected or rejected. The selected Main Planet is highlighted in green.</p>
<div class='candidate-grid'>{cards}</div>
"""


def _solar_review_actions_html(output_dir: Path) -> str:
    q = urllib.parse.quote(str(output_dir))
    return f"""
<section class='card' style='margin-top:18px'>
  <h2>Stage 1 review workflow</h2>
  <p class='help'>Use these controls to iterate cheaply on Stage 1 before committing to planet physics, terrain, and climate.</p>
  <div class='row'>
    <form method='post' action='/solar-action'><input type='hidden' name='output_dir' value='{_safe_text(output_dir)}'><input type='hidden' name='action' value='accept-solar'><button type='submit'>Accept Solar System</button></form>
    <form method='post' action='/solar-action'><input type='hidden' name='output_dir' value='{_safe_text(output_dir)}'><input type='hidden' name='action' value='accept-continue'><button type='submit'>Accept and continue to Planet Physics</button></form>
    <a class='button secondary' href='/system-report?output_dir={q}'>Open Stage 1 report</a>
  </div>
  <h3>Reroll controls <span class='hover-help' title='All reroll buttons run only the solar-system stage. They should not continue into planet physics or terrain.'>?</span></h3>
  <div class='row'>
    <form method='post' action='/solar-action'><input type='hidden' name='output_dir' value='{_safe_text(output_dir)}'><input type='hidden' name='action' value='reroll-same'><button class='secondary' type='submit' title='Creates a new Stage 1 result using the same UI controls/configuration but advances to a new seed. Use this when you like the settings but want a different system.'>Reroll with same controls (new seed)</button></form>
    <form method='post' action='/solar-action'><input type='hidden' name='output_dir' value='{_safe_text(output_dir)}'><input type='hidden' name='action' value='reroll-star'><button class='secondary' type='submit' title='Rerolls the star and the whole system architecture/planet layout. Use this when the star itself is not what you want.'>Reroll star + system</button></form>
    <form method='post' action='/solar-action'><input type='hidden' name='output_dir' value='{_safe_text(output_dir)}'><input type='hidden' name='action' value='reroll-layout'><button class='secondary' type='submit' title='Keeps the current star values, then rerolls the planet orbits/layout and Main Planet candidates. Use this when the star is good but the planet system is not.'>Reroll planet layout, lock star</button></form>
    <form method='post' action='/solar-action'><input type='hidden' name='output_dir' value='{_safe_text(output_dir)}'><input type='hidden' name='action' value='reroll-selection'><button class='secondary' type='submit' title='Keeps the current star and architecture type, then rerolls the detailed layout/candidate scoring. Use this when the system style is right but the selected Main Planet or neighboring planets are not.'>Reroll layout/candidates, lock star+architecture</button></form>
  </div>
  <h3>Lock controls</h3>
  <div class='row'>
    <form method='post' action='/solar-action'><input type='hidden' name='output_dir' value='{_safe_text(output_dir)}'><input type='hidden' name='action' value='lock-star'><button class='secondary' type='submit'>Lock selected star</button></form>
    <form method='post' action='/solar-action'><input type='hidden' name='output_dir' value='{_safe_text(output_dir)}'><input type='hidden' name='action' value='lock-main-planet'><button class='secondary' type='submit'>Lock Main Planet values</button></form>
  </div>
</section>
"""


def _stage_acceptance_html(output_dir: Path) -> str:
    data = _read_json(output_dir / "config" / "stage_acceptance.json", {})
    if not isinstance(data, dict) or "solar-system" not in data:
        return "<p class='muted'>Solar System has not been explicitly accepted yet.</p>"
    item = data.get("solar-system", {}) if isinstance(data.get("solar-system"), dict) else {}
    return f"<p class='ok'>Solar System accepted at {_safe_text(item.get('accepted_at',''))}; Main Planet: {_safe_text(item.get('accepted_main_planet',''))}.</p>"



def _physics_review_state(output_dir: Path) -> dict[str, Any]:
    physics = _planet_physics_state(output_dir)
    review = physics.get("review", {}) if isinstance(physics.get("review"), dict) else {}
    return review


def _physics_warning_list(warnings: list[Any]) -> str:
    if not warnings:
        return "<li class='muted'>No Stage 2 warnings recorded yet.</li>"
    rows = []
    for item in warnings:
        if isinstance(item, dict):
            level = str(item.get("level", "notice")).lower()
            cls = "warn" if level == "warning" else ("ok" if level == "ok" else "muted")
            rows.append(f"<li><span class='{cls}'>{_safe_text(level)}</span>: {_safe_text(item.get('message',''))}</li>")
        else:
            rows.append(f"<li>{_safe_text(item)}</li>")
    return "".join(rows)


def _physics_report_html(output_dir: Path) -> str:
    physics = _planet_physics_state(output_dir)
    review = physics.get("review", {}) if isinstance(physics.get("review"), dict) else {}
    if not physics:
        return "<section class='card'><p class='muted'>No planet-physics state is available yet. Run to <code>planet-physics</code> first.</p></section>"
    rotation = physics.get("rotation", {}) if isinstance(physics.get("rotation"), dict) else {}
    atmosphere = physics.get("atmosphere", {}) if isinstance(physics.get("atmosphere"), dict) else {}
    hydrosphere = physics.get("hydrosphere", {}) if isinstance(physics.get("hydrosphere"), dict) else {}
    geology = physics.get("geology", {}) if isinstance(physics.get("geology"), dict) else {}
    scores = review.get("scores", {}) if isinstance(review.get("scores"), dict) else {}
    report = review.get("report", []) if isinstance(review.get("report"), list) else []
    report_html = "".join(f"<p>{_safe_text(line)}</p>" for line in report) or "<p class='muted'>No generated Stage 2 report found. Rerun planet-physics to create review metadata.</p>"
    return f"""
<section class='card stage-report' style='margin-top:18px'>
  <h2>Stage 2 Planet Physics report</h2>
  <p><span class='pill'>{_safe_text(review.get('archetype','archetype pending'))}</span></p>
  {report_html}
  <div class='grid compact-grid'>
    {_metric_gauge('Human comfort', scores.get('human_comfort'), min_v=0, max_v=1, low='strained', mid='workable', high='comfortable')}
    {_metric_gauge('Surface stability', scores.get('surface_stability'), min_v=0, max_v=1, low='unstable', mid='moderate', high='stable')}
    {_metric_gauge('Climate moderation', scores.get('climate_moderation'), min_v=0, max_v=1, low='variable', mid='moderate', high='well buffered')}
  </div>
  <div class='grid compact-grid' style='margin-top:12px'>
    <section class='physics-card'><h3>Rotation</h3><div class='grid compact-grid'>{_mapping_stat_items(review.get('rotation', {}), ['rotation_class','coriolis_strength','seasonality_class','axial_stability_class','tidal_braking','explanation'])}</div></section>
    <section class='physics-card'><h3>Atmosphere</h3><div class='grid compact-grid'>{_mapping_stat_items(review.get('atmosphere', {}), ['retention_score','retention_class','pressure_class','co2_class','greenhouse_workload','climate_risk','explanation'])}</div></section>
    <section class='physics-card'><h3>Hydrosphere</h3><div class='grid compact-grid'>{_mapping_stat_items(review.get('hydrosphere', {}), ['target_land_fraction','waterworld_risk','dry_world_risk','sea_level_sensitivity','continental_exposure_tendency','ice_storage_tendency','expected_coastline_complexity','volatile_inventory_note'])}</div></section>
    <section class='physics-card'><h3>Geology</h3><div class='grid compact-grid'>{_mapping_stat_items(review.get('geology', {}), ['tectonic_regime','orogenic_intensity','rift_tendency','island_arc_tendency','hotspot_tendency','basin_formation_tendency','continental_fragmentation_tendency','shelf_deposition_tendency','crustal_contrast_strength','explanation'])}</div></section>
  </div>
</section>
<section class='card' style='margin-top:18px'>
  <h2>Raw physical controls</h2>
  <div class='grid compact-grid'>
    {_metric_gauge('Rotation period', rotation.get('rotation_period_hours'), unit='h', min_v=8, max_v=96, low='fast', mid='moderate', high='slow')}
    {_metric_gauge('Axial tilt', rotation.get('axial_tilt_degrees'), unit='°', min_v=0, max_v=80, low='low seasons', mid='moderate', high='extreme seasons')}
    {_metric_gauge('Pressure', atmosphere.get('pressure_bar'), unit='bar', min_v=.1, max_v=5, low='thin', mid='moderate', high='thick')}
    {_metric_gauge('Greenhouse', atmosphere.get('greenhouse_warming_k'), unit='K', min_v=10, max_v=55, low='weak', mid='temperate work', high='heavy')}
    {_metric_gauge('Ocean target', hydrosphere.get('ocean_fraction_target'), min_v=0, max_v=1, low='dry', mid='mixed', high='ocean world')}
    {_metric_gauge('Volatile fraction', hydrosphere.get('volatile_fraction'), min_v=0, max_v=.1, low='dry inventory', mid='moderate', high='wet inventory')}
    {_metric_gauge('Internal heat', geology.get('internal_heat'), min_v=0, max_v=2, low='quiet', mid='active', high='hot')}
    {_metric_gauge('Volcanism', geology.get('volcanism'), min_v=0, max_v=2, low='quiet', mid='active', high='very active')}
    {_metric_gauge('Erosion', geology.get('erosion'), min_v=0, max_v=2.5, low='weak', mid='moderate', high='strong')}
    {_metric_gauge('Mountain factor', geology.get('mountain_factor'), min_v=0, max_v=2, low='low relief', mid='moderate', high='rugged')}
  </div>
</section>
"""


def _physics_downstream_html(output_dir: Path) -> str:
    review = _physics_review_state(output_dir)
    downstream = review.get("downstream_implications", {}) if isinstance(review.get("downstream_implications"), dict) else {}
    blocks = []
    for key in ("terrain", "climate", "hydrology"):
        items = downstream.get(key, []) if isinstance(downstream.get(key), list) else []
        lis = "".join(f"<li>{_safe_text(x)}</li>" for x in items) or "<li class='muted'>No implications recorded yet.</li>"
        blocks.append(f"<section class='physics-card'><h3>{_safe_text(_humanize_key(key))}</h3><ul class='implication-list'>{lis}</ul></section>")
    return "<section class='card' style='margin-top:18px'><h2>Downstream implications</h2><div class='grid compact-grid'>" + "".join(blocks) + "</div></section>"


def _physics_actions_html(output_dir: Path) -> str:
    q = urllib.parse.quote(str(output_dir))
    return f"""
<section class='card' style='margin-top:18px'>
  <h2>Stage 2 review workflow</h2>
  <p class='help'>These actions rerun only <code>planet-physics</code> unless the button says continue. Lock buttons save current values into <code>stage_overrides.json</code>; reroll-single-section buttons lock the other sections, rerun Stage 2, then apply the locks.</p>
  <div class='row'>
    <form method='post' action='/physics-action'><input type='hidden' name='output_dir' value='{_safe_text(output_dir)}'><input type='hidden' name='action' value='accept-physics'><button type='submit'>Accept Planet Physics</button></form>
    <form method='post' action='/physics-action'><input type='hidden' name='output_dir' value='{_safe_text(output_dir)}'><input type='hidden' name='action' value='accept-continue'><button type='submit'>Accept and continue to Terrain</button></form>
    <a class='button secondary' href='/planet?output_dir={q}'>Main Planet viewer</a>
    <a class='button secondary' href='/overrides?output_dir={q}'>Raw overrides</a>
  </div>
  <h3>Reroll controls <span class='hover-help' title='Rerolling Stage 2 preserves the accepted solar system and regenerates rotation/atmosphere/hydrosphere/geology only. It stops at planet-physics.'>?</span></h3>
  <div class='row'>
    <form method='post' action='/physics-action'><input type='hidden' name='output_dir' value='{_safe_text(output_dir)}'><input type='hidden' name='action' value='reroll-all'><button class='secondary' type='submit' title='Regenerates all planet-physics values and stops at Stage 2. Downstream terrain/climate stages are invalidated but not rerun.'>Reroll all planet physics</button></form>
    <form method='post' action='/physics-action'><input type='hidden' name='output_dir' value='{_safe_text(output_dir)}'><input type='hidden' name='action' value='reroll-rotation'><button class='secondary' type='submit' title='Locks atmosphere, hydrosphere, and geology; reruns Stage 2 so only rotation/tilt change after locks are applied.'>Reroll rotation only</button></form>
    <form method='post' action='/physics-action'><input type='hidden' name='output_dir' value='{_safe_text(output_dir)}'><input type='hidden' name='action' value='reroll-atmosphere'><button class='secondary' type='submit' title='Locks rotation, hydrosphere, and geology; reruns Stage 2 so atmosphere values change after locks are applied.'>Reroll atmosphere only</button></form>
    <form method='post' action='/physics-action'><input type='hidden' name='output_dir' value='{_safe_text(output_dir)}'><input type='hidden' name='action' value='reroll-hydrosphere'><button class='secondary' type='submit' title='Locks rotation, atmosphere, and geology; reruns Stage 2 so hydrosphere/ocean target change after locks are applied.'>Reroll hydrosphere only</button></form>
    <form method='post' action='/physics-action'><input type='hidden' name='output_dir' value='{_safe_text(output_dir)}'><input type='hidden' name='action' value='reroll-geology'><button class='secondary' type='submit' title='Locks rotation, atmosphere, and hydrosphere; reruns Stage 2 so geology values change after locks are applied.'>Reroll geology only</button></form>
  </div>
  <h3>Lock controls</h3>
  <div class='row'>
    <form method='post' action='/physics-action'><input type='hidden' name='output_dir' value='{_safe_text(output_dir)}'><input type='hidden' name='action' value='lock-rotation'><button class='secondary' type='submit'>Lock rotation</button></form>
    <form method='post' action='/physics-action'><input type='hidden' name='output_dir' value='{_safe_text(output_dir)}'><input type='hidden' name='action' value='lock-atmosphere'><button class='secondary' type='submit'>Lock atmosphere</button></form>
    <form method='post' action='/physics-action'><input type='hidden' name='output_dir' value='{_safe_text(output_dir)}'><input type='hidden' name='action' value='lock-hydrosphere'><button class='secondary' type='submit'>Lock hydrosphere</button></form>
    <form method='post' action='/physics-action'><input type='hidden' name='output_dir' value='{_safe_text(output_dir)}'><input type='hidden' name='action' value='lock-geology'><button class='secondary' type='submit'>Lock geology</button></form>
  </div>
  <h3>Presets</h3>
  <div class='row'>
    {''.join(f"<form method='post' action='/physics-action'><input type='hidden' name='output_dir' value='{_safe_text(output_dir)}'><input type='hidden' name='action' value='preset'><input type='hidden' name='preset' value='{p}'><button class='secondary' type='submit' title='Save the {p.replace('_',' ')} Stage 2 preset into overrides.'>{_humanize_key(p)}</button></form>" for p in ['earth_like_balanced','dry_rugged_world','ocean_world','high_volcanism_world','cold_high_tilt_world','thick_atmosphere_super_earth','low_gravity_thin_atmosphere'])}
  </div>
</section>
"""


def _stage2_acceptance_html(output_dir: Path) -> str:
    data = _read_json(output_dir / "config" / "stage_acceptance.json", {})
    if not isinstance(data, dict) or "planet-physics" not in data:
        return "<p class='muted'>Planet Physics has not been explicitly accepted yet.</p>"
    item = data.get("planet-physics", {}) if isinstance(data.get("planet-physics"), dict) else {}
    return f"<p class='ok'>Planet Physics accepted at {_safe_text(item.get('accepted_at',''))}; archetype: {_safe_text(item.get('archetype',''))}.</p>"



STAGE_REVIEW_LABELS = {
    "solar-system": "Stage 1: Solar System",
    "planet-physics": "Stage 2: Planet Physics",
    "terrain-foundation-mask": "Stage 3A: Terrain foundation/mask",
    "terrain-tectonic-provinces": "Stage 3B: Tectonic provinces",
    "terrain-crust-and-boundaries": "Stage 3C: Crust and boundaries",
    "terrain-mountains-basins-rifts": "Stage 3D: Mountains, basins, rifts",
    "terrain-coasts-shelves-islands": "Stage 3E: Coasts, shelves, islands",
    "terrain-erosion-deposition": "Stage 3F: Erosion/deposition",
    "terrain-finalization-recentering": "Stage 3G: Terrain finalization",
    "climate": "Stage 4: Climate",
    "hydrology": "Stage 5: Hydrology",
    "biomes": "Stage 6: Biomes",
    "regions": "Stage 7: Regions",
    "outputs": "Stage 8: Outputs",
}

STAGE_DEPENDENCIES = {
    "solar-system": ["run seed", "star-class/system controls", "planet-count controls"],
    "planet-physics": ["star mass/luminosity/age", "Main Planet mass/radius/orbit", "Stage 1 volatile delivery", "moon tide/stability metadata", "giant-planet influence"],
    "terrain-foundation-mask": ["ocean target", "planet radius", "geology class", "tectonic regime", "crustal asymmetry", "terrain resolution"],
    "terrain-tectonic-provinces": ["terrain foundation", "target plate count", "tectonic regime", "crustal contrast"],
    "terrain-crust-and-boundaries": ["tectonic provinces", "crustal asymmetry", "rift/island-arc tendencies"],
    "terrain-mountains-basins-rifts": ["mountain factor", "orogenic intensity", "rift tendency", "basin formation tendency"],
    "terrain-coasts-shelves-islands": ["ocean target", "shelf/deposition tendency", "island density", "coastline complexity"],
    "terrain-erosion-deposition": ["erosion", "precipitation/climate context", "shelf/deposition tendency", "surface roughness"],
    "terrain-finalization-recentering": ["all terrain subphases", "ocean target", "map projection/resolution"],
    "climate": ["terrain elevation/mask", "rotation class", "pressure", "greenhouse workload", "ocean fraction", "axial tilt"],
    "hydrology": ["terrain gradients", "precipitation", "evaporation/temperature", "erosion", "ocean mask"],
    "biomes": ["temperature", "precipitation", "hydrology", "Köppen classes", "elevation"],
    "regions": ["terrain", "climate", "hydrology", "biomes", "map scale"],
    "outputs": ["all generated stages", "map renderer settings", "diagnostics settings"],
}

STAGE_OUTPUT_SUMMARIES = {
    "solar-system": "Star, planets, selected Main Planet, moon, system architecture, and Stage 1 diagnostics.",
    "planet-physics": "Rotation, atmosphere, hydrosphere, geology, warnings, and downstream implications.",
    "terrain-foundation-mask": "Initial land/ocean/elevation foundation and terrain-mask checkpoint.",
    "terrain-tectonic-provinces": "Tectonic province checkpoint derived from the monolithic terrain pass.",
    "terrain-crust-and-boundaries": "Crust and boundary diagnostic checkpoint.",
    "terrain-mountains-basins-rifts": "Mountain, basin, and rift diagnostic checkpoint.",
    "terrain-coasts-shelves-islands": "Coast, shelf, and island diagnostic checkpoint.",
    "terrain-erosion-deposition": "Erosion/deposition diagnostic checkpoint.",
    "terrain-finalization-recentering": "Final terrain state used by climate/hydrology.",
    "climate": "Temperature, precipitation, wind/current diagnostics, and Köppen context.",
    "hydrology": "Rivers, drainage basins, lakes/sinks, runoff, and deltas.",
    "biomes": "Biome map and biome classification state.",
    "regions": "Regional summaries and labels.",
    "outputs": "Final maps, reports, diagnostics, and export files.",
}


def _stage_label(stage: str) -> str:
    return STAGE_REVIEW_LABELS.get(stage, _humanize_key(stage))


def _stage_detail_href(output_dir: Path, stage: str) -> str:
    q = urllib.parse.quote(str(output_dir))
    if stage == "solar-system":
        return f"/system-report?output_dir={q}"
    if stage == "planet-physics":
        return f"/physics?output_dir={q}"
    return f"/stage?output_dir={q}&stage={urllib.parse.quote(stage)}"


def _read_acceptance(output_dir: Path) -> dict[str, Any]:
    data = _read_json(output_dir / "config" / "stage_acceptance.json", {})
    return data if isinstance(data, dict) else {}


def _write_acceptance(output_dir: Path, acceptance: dict[str, Any]) -> None:
    _write_json(output_dir / "config" / "stage_acceptance.json", acceptance)


def _is_stage_accepted(output_dir: Path, stage: str) -> bool:
    item = _read_acceptance(output_dir).get(stage)
    return isinstance(item, dict)


def _stage_acceptance_record(output_dir: Path, stage: str) -> dict[str, Any]:
    item = _read_acceptance(output_dir).get(stage, {})
    return item if isinstance(item, dict) else {}


def _record_stage_acceptance(output_dir: Path, stage: str, note: str = "Accepted from the pipeline workspace.") -> None:
    acceptance = _read_acceptance(output_dir)
    acceptance[stage] = {
        "accepted_at": now_stamp(),
        "note": note,
        "stage_label": _stage_label(stage),
    }
    _write_acceptance(output_dir, acceptance)


def _read_provenance(output_dir: Path) -> dict[str, Any]:
    data = _read_json(output_dir / "config" / "value_provenance.json", {})
    return data if isinstance(data, dict) else {}


def _write_provenance(output_dir: Path, data: dict[str, Any]) -> None:
    _write_json(output_dir / "config" / "value_provenance.json", data)


def _flatten_leaf_paths(data: Any, prefix: str = "") -> list[tuple[str, Any]]:
    if isinstance(data, dict):
        out: list[tuple[str, Any]] = []
        for key, value in data.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            out.extend(_flatten_leaf_paths(value, path))
        return out
    return [(prefix, data)] if prefix else []


def _record_override_provenance(output_dir: Path, overrides: dict[str, Any], *, source: str = "manual", note: str = "Stored from the Web UI.") -> None:
    provenance = _read_provenance(output_dir)
    for path, value in _flatten_leaf_paths(overrides):
        provenance[path] = {
            "value": value,
            "source": source,
            "locked": source.startswith("locked"),
            "set_at": now_stamp(),
            "note": note,
        }
    _write_provenance(output_dir, provenance)


def _override_value_at(overrides: dict[str, Any], dotted: str) -> Any:
    cur: Any = overrides
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur.get(part)
    return cur


def _value_source_chip(output_dir: Path, dotted: str) -> str:
    provenance = _read_provenance(output_dir)
    if dotted in provenance and isinstance(provenance.get(dotted), dict):
        item = provenance[dotted]
        source = str(item.get("source", "manual"))
        cls = "manual" if "manual" in source else ("locked" if "locked" in source else "derived")
        title = f"{source}; {item.get('note','')}"
        return f"<span class='source-chip {cls}' title='{_safe_text(title)}'>{_safe_text(_humanize_key(source))}</span>"
    overrides = _read_json(output_dir / "config" / "stage_overrides.json", {})
    if isinstance(overrides, dict) and _override_value_at(overrides, dotted) is not None:
        return "<span class='source-chip manual' title='This value exists in stage_overrides.json, so it is treated as manually set even if older metadata is missing.'>Manual override</span>"
    return "<span class='source-chip generated' title='No manual override is recorded for this value.'>Generated/derived</span>"


def _normalize_warning_item(item: Any, default_level: str = "warning") -> dict[str, str]:
    """Convert legacy warning strings/dicts into a clean display record."""
    if isinstance(item, dict):
        level = str(item.get("level") or item.get("severity") or default_level)
        message_obj = item.get("message", item.get("text", item.get("reason", "")))
        if isinstance(message_obj, dict):
            message = str(message_obj.get("message") or message_obj.get("text") or message_obj.get("reason") or message_obj)
        elif isinstance(message_obj, list):
            message = "; ".join(str(part) for part in message_obj)
        else:
            message = str(message_obj) if message_obj is not None else ""
        details_obj = item.get("details") or item.get("hint") or item.get("source")
        details = ""
        if isinstance(details_obj, dict):
            details = "; ".join(f"{_humanize_key(k)}: {v}" for k, v in details_obj.items())
        elif isinstance(details_obj, list):
            details = "; ".join(str(part) for part in details_obj)
        elif details_obj is not None:
            details = str(details_obj)
        if not message or message == str(item):
            # Avoid rendering raw {'level':..., 'message':...} when possible.
            pairs = []
            for key, value in item.items():
                if key in {"level", "severity"}:
                    continue
                pairs.append(f"{_humanize_key(key)}: {value}")
            message = "; ".join(pairs) or str(item)
        return {"level": level, "message": message, "details": details}
    text = str(item)
    # Some older saved reports may contain a Python-dict-looking string. Do a
    # conservative parse by regex so the UI still renders the message cleanly.
    level_match = re.search(r"['\"]level['\"]\s*:\s*['\"]([^'\"]+)['\"]", text)
    message_match = re.search(r"['\"]message['\"]\s*:\s*['\"]([^'\"]+)['\"]", text)
    if message_match:
        return {"level": level_match.group(1) if level_match else default_level, "message": message_match.group(1), "details": ""}
    return {"level": default_level, "message": text, "details": ""}


def _incongruence_warnings(output_dir: Path) -> list[dict[str, str]]:
    warnings: list[dict[str, str]] = []
    overrides = _read_json(output_dir / "config" / "stage_overrides.json", {})
    if not isinstance(overrides, dict):
        return warnings
    solar = _solar_system_state(output_dir) or {}
    star = solar.get("star", {}) if isinstance(solar.get("star"), dict) else {}
    physics = _planet_physics_state(output_dir)
    review = physics.get("review", {}) if isinstance(physics.get("review"), dict) else {}
    hydro = physics.get("hydrosphere", {}) if isinstance(physics.get("hydrosphere"), dict) else {}
    geology = physics.get("geology", {}) if isinstance(physics.get("geology"), dict) else {}
    main = _main_planet_state(output_dir) or {}

    star_over = overrides.get("star", {}) if isinstance(overrides.get("star"), dict) else {}
    if "luminosity_solar" in star_over and not any(k in star_over for k in ["mass_solar", "age_gyr", "stellar_class"]):
        warnings.append({"level": "strong", "message": "Star luminosity is manually set while mass, age, and stellar class remain generated/fixed. Consider adjusting those preconditions instead if you want a physically derived brighter star."})
    if "mass_solar" in star_over and "luminosity_solar" not in star_over:
        warnings.append({"level": "info", "message": "Star mass is manually set. Luminosity may be derived from the changed mass on rerun, which is usually preferable to forcing luminosity directly."})

    main_over = overrides.get("main_planet", {}) if isinstance(overrides.get("main_planet"), dict) else {}
    mass = main_over.get("mass_earth", main.get("mass_earth"))
    radius = main_over.get("radius_earth", main.get("radius_earth"))
    try:
        density_rel = float(mass) / (float(radius) ** 3)
        if density_rel < 0.45 or density_rel > 2.2:
            warnings.append({"level": "strong", "message": f"Main Planet mass/radius imply unusual density ({density_rel:.2f} Earth density). Consider changing mass and radius together or adjusting composition/preference instead."})
    except Exception:
        pass

    pp_over = overrides.get("planet_physics", {}) if isinstance(overrides.get("planet_physics"), dict) else {}
    atm_over = pp_over.get("atmosphere", {}) if isinstance(pp_over.get("atmosphere"), dict) else {}
    hydro_over = pp_over.get("hydrosphere", {}) if isinstance(pp_over.get("hydrosphere"), dict) else {}
    geo_over = pp_over.get("geology", {}) if isinstance(pp_over.get("geology"), dict) else {}
    retention = review.get("atmosphere", {}).get("retention_class") if isinstance(review.get("atmosphere"), dict) else None
    if retention in {"weak", "low"} and float(atm_over.get("pressure_bar", 0) or 0) > 1.5:
        warnings.append({"level": "strong", "message": "Atmospheric pressure is manually high but the generated retention class is weak/low. Consider increasing gravity/volatile inventory or accepting a forced atmosphere."})
    volatile_delivery = (solar.get("diagnostics", {}) if isinstance(solar.get("diagnostics"), dict) else {}).get("volatile_delivery")
    ocean_target = hydro_over.get("ocean_fraction_target", hydro.get("ocean_fraction_target"))
    try:
        if volatile_delivery in {"low", "dry"} and float(ocean_target) > 0.65:
            warnings.append({"level": "strong", "message": "Ocean target is high despite low volatile delivery. Consider increasing volatile delivery/source context or mark the ocean target as intentionally forced."})
    except Exception:
        pass
    try:
        if float(geo_over.get("volcanism", geology.get("volcanism", 0)) or 0) > 1.2 and float(geo_over.get("internal_heat", geology.get("internal_heat", 0)) or 0) < 0.6:
            warnings.append({"level": "warning", "message": "Volcanism is high while internal heat is low. Consider increasing internal heat/tectonic energy or lowering volcanism."})
    except Exception:
        pass
    return warnings


def _stage_warning_items(output_dir: Path, stage: str) -> list[dict[str, str]]:
    warnings: list[dict[str, str]] = []
    if stage == "solar-system":
        solar = _solar_system_state(output_dir) or {}
        diag = solar.get("diagnostics", {}) if isinstance(solar.get("diagnostics"), dict) else {}
        for item in diag.get("stage1_warnings", []) if isinstance(diag.get("stage1_warnings"), list) else []:
            warnings.append(_normalize_warning_item(item))
    if stage == "planet-physics":
        review = _physics_review_state(output_dir)
        for item in review.get("warnings", []) if isinstance(review.get("warnings"), list) else []:
            warnings.append(_normalize_warning_item(item))
    if stage.startswith("terrain"):
        review = _terrain_review_state(output_dir)
        for item in review.get("warnings", []) if isinstance(review.get("warnings"), list) else []:
            warnings.append(_normalize_warning_item(item))
        subphases = review.get("subphases", {}) if isinstance(review.get("subphases"), dict) else {}
        info = subphases.get(stage, {}) if isinstance(subphases.get(stage), dict) else {}
        for item in info.get("warnings", []) if isinstance(info.get("warnings"), list) else []:
            warnings.append(_normalize_warning_item(item))
    # Manual/derived incongruence can affect several downstream stages; show it prominently on early stages and the generic workspace.
    if stage in {"solar-system", "planet-physics", "terrain-foundation-mask", "climate"}:
        warnings.extend(_incongruence_warnings(output_dir))
    return warnings


def _stage_status_map(output_dir: Path) -> dict[str, dict[str, Any]]:
    try:
        rows = status_detail_rows(output_dir) if output_dir.exists() else []
    except Exception:
        rows = []
    return {str(row.get("stage")): row for row in rows if isinstance(row, dict)}


def _pipeline_stage_nav_html(output_dir: Path, current_stage: str | None = None) -> str:
    status_by_stage = _stage_status_map(output_dir)
    links: list[str] = []
    for idx, stage in enumerate(STAGE_ORDER, start=1):
        row = status_by_stage.get(stage, {})
        status = str(row.get("status", "missing")) or "missing"
        status_cls = "complete" if status == "complete" else ("stale" if status == "stale" else "missing")
        current_cls = " current" if current_stage == stage else ""
        accepted = _is_stage_accepted(output_dir, stage)
        accepted_mark = " <span class='accepted-mark'>✓</span>" if accepted else ""
        url = "/stage?" + urllib.parse.urlencode({"output_dir": str(output_dir), "stage": stage})
        title = f"{_stage_label(stage)} — {status}" + ("; accepted" if accepted else "")
        links.append(
            f"<a class='{status_cls}{current_cls}' href='{url}' title='{_safe_text(title)}'>"
            f"<span class='nav-dot'></span><span>{idx}. {_safe_text(_stage_label(stage))}</span>{accepted_mark}</a>"
        )
    return "<nav class='pipeline-nav' aria-label='Pipeline stages'>" + "".join(links) + "</nav>"


def _warning_badge(warnings: list[dict[str, str]]) -> str:
    normalized = [_normalize_warning_item(w) for w in warnings]
    if not normalized:
        return "<span class='pill ok-pill'>0 warnings</span>"
    strong = sum(1 for w in normalized if str(w.get("level", "")).lower() in {"strong", "error", "hard", "severe", "block"})
    cls = "bad-pill" if strong else "warn-pill"
    label = f"{len(normalized)} warning" + ("s" if len(normalized) != 1 else "")
    return f"<span class='pill {cls}'>{_safe_text(label)}</span>"


def _stage_dependencies_html(output_dir: Path, stage: str) -> str:
    deps = STAGE_DEPENDENCIES.get(stage, [])
    if not deps:
        return "<p class='muted'>No dependency catalog has been defined for this stage yet.</p>"
    items = []
    for dep in deps:
        dotted_guess = dep.replace(" ", "_")
        items.append(f"<li><b>{_safe_text(dep)}</b> <span class='muted'>Used as a precondition for this stage. If you override a related upstream value, this stage should become stale or be rerun.</span></li>")
    return "<ul>" + "".join(items) + "</ul>"


def _stage_override_summary_html(output_dir: Path) -> str:
    overrides = _read_json(output_dir / "config" / "stage_overrides.json", {})
    if not isinstance(overrides, dict) or not overrides:
        return "<p class='muted'>No manual overrides are currently saved. Values are generated or derived from upstream stages.</p>"
    leaves = _flatten_leaf_paths(overrides)
    rows = ""
    for dotted, value in leaves[:80]:
        rows += f"<tr><td><code>{_safe_text(dotted)}</code></td><td>{_safe_text(value)}</td><td>{_value_source_chip(output_dir, dotted)}</td></tr>"
    if len(leaves) > 80:
        rows += f"<tr><td colspan='3' class='muted'>Showing first 80 of {len(leaves)} override values.</td></tr>"
    return f"<table><thead><tr><th>Value</th><th>Override</th><th>Source</th></tr></thead><tbody>{rows}</tbody></table>"


def _stage_warnings_html(warnings: list[dict[str, str]]) -> str:
    normalized = [_normalize_warning_item(w) for w in warnings]
    if not normalized:
        return "<p class='ok'>No review warnings for this stage.</p>"
    rows = ""
    for w in normalized:
        level = str(w.get("level", "warning")).lower()
        cls = "bad" if level in {"strong", "error", "hard", "severe", "block"} else ("warn" if level in {"warning", "soft", "caution"} else "muted")
        details = str(w.get("details", "") or "").strip()
        details_html = f"<div class='warning-details'>{_safe_text(details)}</div>" if details else ""
        rows += (
            f"<li class='warning-item {cls}'><span class='warning-level'>{_safe_text(_humanize_key(level))}</span>"
            f"<span class='warning-message'>{_safe_text(w.get('message',''))}</span>{details_html}</li>"
        )
    return f"<ul class='warning-list'>{rows}</ul>"


def _next_stage(stage: str) -> str | None:
    try:
        i = STAGE_ORDER.index(normalize_stage(stage))
        return STAGE_ORDER[i + 1] if i + 1 < len(STAGE_ORDER) else None
    except Exception:
        return None


def _stage_action_panel(output_dir: Path, stage: str) -> str:
    next_stage = _next_stage(stage)
    q_stage = _safe_text(stage)
    next_button = ""
    if next_stage:
        next_button = f"<form method='post' action='/stage-workflow'><input type='hidden' name='output_dir' value='{_safe_text(output_dir)}'><input type='hidden' name='stage' value='{q_stage}'><input type='hidden' name='action' value='continue-next'><button type='submit'>Continue to {_safe_text(_stage_label(next_stage))}</button></form>"
    stage_options = _stage_options("outputs")
    return f"""
<section class='card stage-actions-card'>
  <h2>Stage actions</h2>
  <p class='help'>These buttons are intentionally explicit. Reroll only this stage stops at this stage. Continue runs later stages only when you press a continue/run-to button. You can still run many or all stages without review using the run-to controls.</p>
  <div class='row'>
    <form method='post' action='/stage-workflow'><input type='hidden' name='output_dir' value='{_safe_text(output_dir)}'><input type='hidden' name='stage' value='{q_stage}'><input type='hidden' name='action' value='accept-stage'><button type='submit'>Accept this stage</button></form>
    <form method='post' action='/stage-workflow'><input type='hidden' name='output_dir' value='{_safe_text(output_dir)}'><input type='hidden' name='stage' value='{q_stage}'><input type='hidden' name='action' value='reroll-stage'><button class='secondary' type='submit' title='Regenerate only this stage and stop here. Downstream stages become stale but are not automatically regenerated.'>Reroll only this stage</button></form>
    {next_button}
  </div>
  <form method='post' action='/stage-workflow' class='inline-run-to'>
    <input type='hidden' name='output_dir' value='{_safe_text(output_dir)}'>
    <input type='hidden' name='stage' value='{q_stage}'>
    <input type='hidden' name='action' value='run-to-selected'>
    <label>Run without review to selected stage</label><select name='target_stage'>{stage_options}</select>
    <div class='row'><label><input type='checkbox' name='yes' checked style='width:auto'> auto-confirm</label><label><input type='checkbox' name='skip_json' checked style='width:auto'> skip JSON</label><button type='submit'>Run to selected stage</button></div>
  </form>
</section>
"""

def _stage_edit_values_html(output_dir: Path, stage: str) -> str:
    """Predictable per-stage edit area using the same override save path as the visual editor."""
    overrides = _read_json(output_dir / "config" / "stage_overrides.json", {})
    if not isinstance(overrides, dict):
        overrides = {}
    solar = _solar_system_state(output_dir) or {}
    planets = solar.get("planets", []) if isinstance(solar.get("planets"), list) else []
    main_planet = next((p for p in planets if isinstance(p, dict) and p.get("is_main_planet")), {})
    physics = _planet_physics_state(output_dir) or {}
    if not isinstance(physics, dict):
        physics = {}
    resolved_config = _read_json(output_dir / "config" / "resolved_config.json", {})
    if not isinstance(resolved_config, dict):
        resolved_config = {}
    planet_profile_cfg = resolved_config.get("planet_profile", {}) if isinstance(resolved_config.get("planet_profile"), dict) else {}

    def qv(form_key: str, override_path: list[str], actual: Any = "") -> str:
        ov = _get_nested(overrides, override_path, None)
        value = actual if ov is None else ov
        return _safe_text("" if value is None else value)

    star = solar.get("star", {}) if isinstance(solar.get("star"), dict) else {}
    diagnostics_state = solar.get("diagnostics", {}) if isinstance(solar.get("diagnostics"), dict) else {}
    moon_state = main_planet.get("moon", {}) if isinstance(main_planet.get("moon"), dict) else {}
    orbit = main_planet.get("orbit", {}) if isinstance(main_planet.get("orbit"), dict) else {}
    rotation = physics.get("rotation", {}) if isinstance(physics.get("rotation"), dict) else {}
    atmosphere = physics.get("atmosphere", {}) if isinstance(physics.get("atmosphere"), dict) else {}
    hydrosphere = physics.get("hydrosphere", {}) if isinstance(physics.get("hydrosphere"), dict) else {}
    geology = physics.get("geology", {}) if isinstance(physics.get("geology"), dict) else {}
    architecture_actual = solar.get("architecture") or diagnostics_state.get("architecture") or ""
    preference_actual = diagnostics_state.get("main_planet_preference") or "earthlike"
    moon_strength_actual = _get_nested(overrides, ["system", "moon_strength_preference"], None) or moon_state.get("tidal_effect_level") or "moderate"
    require_major_moon_value = _get_nested(overrides, ["system", "require_major_moon"], None)
    if require_major_moon_value is None:
        require_major_moon_value = bool(moon_state) if main_planet else True
    require_major_moon_checked = " checked" if bool(require_major_moon_value) else ""
    redirect_to = "/stage?" + urllib.parse.urlencode({"output_dir": str(output_dir), "stage": stage})

    solar_controls = f"""
      <h3>Solar-system values</h3>
      <p class='help'>Use these controls for Stage 1 and upstream preconditions. For physically derived values such as luminosity, changing the value directly records it as manual; prefer adjusting class, mass, age, or architecture when possible.</p>
      <div class='grid compact-grid'>
        {_select_control('star_stellar_class', 'Star class / subtype', qv('star_stellar_class', ['star','stellar_class'], star.get('spectral_type') or star.get('stellar_class') or ''), [("", "Auto / generated"), ("G", "G class"), ("K", "K class"), ("G0V", "G0V"), ("G2V", "G2V"), ("G5V", "G5V"), ("K0V", "K0V"), ("K3V", "K3V"), ("K5V", "K5V"), ("K7V", "K7V")], help_text='Changing class/subtype is the preferred way to shift mass, luminosity, and habitable-zone position naturally.')}
        {_select_control('system_architecture_type', 'System architecture', qv('system_architecture_type', ['system','architecture_type'], architecture_actual), SYSTEM_ARCHITECTURE_OPTIONS, help_text='Controls the orbital layout pattern and downstream formation context.')}
        {_select_control('main_planet_preference', 'Main Planet preference', qv('main_planet_preference', ['system','main_planet_preference'], preference_actual), MAIN_PLANET_PREFERENCE_OPTIONS, help_text='Controls candidate scoring and the kind of planet chosen for detailed simulation.')}
        {_select_control('moon_strength_preference', 'Moon tide strength', qv('moon_strength_preference', ['system','moon_strength_preference'], moon_strength_actual), MOON_STRENGTH_OPTIONS, help_text='Changes moon mass/orbit tendencies and downstream tide/axial-stability context.')}
        <div class='visual-editor-field'><label>Require major moon</label><input type='hidden' name='require_major_moon_present' value='1'><label><input type='checkbox' name='require_major_moon' style='width:auto'{require_major_moon_checked}> include one major moon</label><p class='field-help'>Affects tides, axial stability, and later coastal/climate assumptions.</p></div>
        <div class='visual-editor-field'><label>Star mass M☉</label><input name='star_mass_solar' value='{qv('star_mass_solar', ['star','mass_solar'], star.get('mass_solar'))}'><p class='field-help'>Preferred precondition for changing luminosity consistently.</p></div>
        <div class='visual-editor-field'><label>Star luminosity L☉</label><input name='star_luminosity_solar' value='{qv('star_luminosity_solar', ['star','luminosity_solar'], star.get('luminosity_solar'))}'><p class='field-help'>Direct edits are manual overrides and may warn if mass/class/age remain fixed.</p></div>
        <div class='visual-editor-field'><label>Main planet radius R⊕</label><input name='main_radius_earth' value='{qv('main_radius_earth', ['main_planet','radius_earth'], main_planet.get('radius_earth'))}'><p class='field-help'>Changes gravity, map scale, atmosphere retention, and terrain scale assumptions.</p></div>
        <div class='visual-editor-field'><label>Main planet mass M⊕</label><input name='main_mass_earth' value='{qv('main_mass_earth', ['main_planet','mass_earth'], main_planet.get('mass_earth'))}'><p class='field-help'>Use with radius to keep density and gravity plausible.</p></div>
        <div class='visual-editor-field'><label>Surface gravity g</label><input name='main_gravity_g' value='{qv('main_gravity_g', ['main_planet','surface_gravity_g'], main_planet.get('surface_gravity_g'))}'><p class='field-help'>Affects atmosphere, pressure, erosion, and habitability assumptions.</p></div>
        <div class='visual-editor-field'><label>Orbit semi-major axis AU</label><input name='orbit_semi_major_axis_au' value='{qv('orbit_semi_major_axis_au', ['main_planet','orbit','semi_major_axis_au'], orbit.get('semi_major_axis_au'))}'><p class='field-help'>Changes flux and equilibrium temperature; check habitable-zone warnings after edits.</p></div>
        <div class='visual-editor-field'><label>Orbit eccentricity</label><input name='orbit_eccentricity' value='{qv('orbit_eccentricity', ['main_planet','orbit','eccentricity'], orbit.get('eccentricity'))}'><p class='field-help'>Affects orbital seasonality and climate stress.</p></div>
      </div>
    """

    physics_controls = f"""
      <h3>Planet-physics values</h3>
      <p class='help'>These controls define the planet identity before terrain and climate. They are the predictable edit location for Stage 2 and terrain preconditions.</p>
      <div class='grid compact-grid'>
        {_slider_control('rotation_period_hours', 'Rotation period', qv('rotation_period_hours', ['planet_physics','rotation','rotation_period_hours'], rotation.get('rotation_period_hours')), min_v=8, max_v=96, step=0.5, unit='hours', help_text='Short days strengthen Coriolis behavior; long days broaden circulation cells.', suggested='18–36 h for many Earthlike tests')}
        {_slider_control('axial_tilt_degrees', 'Axial tilt', qv('axial_tilt_degrees', ['planet_physics','rotation','axial_tilt_degrees'], rotation.get('axial_tilt_degrees')), min_v=0, max_v=80, step=0.5, unit='degrees', help_text='Controls seasonal contrast and high-latitude climate.', suggested='10–35° for moderate seasons')}
        {_slider_control('pressure_bar', 'Pressure', qv('pressure_bar', ['planet_physics','atmosphere','pressure_bar'], atmosphere.get('pressure_bar')), min_v=0.1, max_v=5, step=0.05, unit='bar', help_text='Affects heat transport, water stability, and later precipitation behavior.', suggested='0.7–2.0 bar for many habitable worlds')}
        {_slider_control('co2_ppm', 'CO₂', qv('co2_ppm', ['planet_physics','atmosphere','carbon_dioxide_ppm'], atmosphere.get('carbon_dioxide_ppm')), min_v=0, max_v=5000, step=10, unit='ppm', help_text='One greenhouse control. High values can warm or stress climate depending on other settings.', suggested='250–1000 ppm for mild tests')}
        <div class='visual-editor-field'><label>Greenhouse warming K</label><input name='greenhouse_warming_k' value='{qv('greenhouse_warming_k', ['planet_physics','atmosphere','greenhouse_warming_k'], atmosphere.get('greenhouse_warming_k'))}'><p class='field-help'>Directly affects the temperature baseline used by climate.</p></div>
        <div class='visual-editor-field'><label>Water vapor factor</label><input name='water_vapor_factor' value='{qv('water_vapor_factor', ['planet_physics','atmosphere','water_vapor_factor'], atmosphere.get('water_vapor_factor'))}'><p class='field-help'>Affects humidity, precipitation potential, and greenhouse feedback.</p></div>
        {_slider_control('ocean_fraction_target', 'Ocean fraction target', qv('ocean_fraction_target', ['planet_physics','hydrosphere','ocean_fraction_target'], hydrosphere.get('ocean_fraction_target')), min_v=0.05, max_v=0.95, step=0.01, unit='fraction', help_text='Target ocean coverage before terrain/hydrology adjust details.', suggested='0.45–0.75 for mixed land/ocean')}
        {_slider_control('volcanism', 'Volcanism', qv('volcanism', ['planet_physics','geology','volcanism'], geology.get('volcanism')), min_v=0, max_v=1, step=0.01, unit='0–1', help_text='Influences arcs, hotspot islands, resurfacing, and atmospheric outgassing.', suggested='0.3–0.8')}
        {_slider_control('erosion', 'Erosion', qv('erosion', ['planet_physics','geology','erosion'], geology.get('erosion')), min_v=0, max_v=1, step=0.01, unit='0–1', help_text='Controls smoothing, sediment transfer, river maturity, and plains.', suggested='0.25–0.7')}
        {_slider_control('mountain_factor', 'Mountain factor', qv('mountain_factor', ['planet_physics','geology','mountain_factor'], geology.get('mountain_factor')), min_v=0, max_v=2, step=0.01, unit='factor', help_text='Scales mountain prominence and orographic effects downstream.', suggested='0.7–1.4')}
        {_slider_control('surface_roughness', 'Surface roughness', qv('surface_roughness', ['planet_physics','geology','surface_roughness'], geology.get('surface_roughness')), min_v=0, max_v=1, step=0.01, unit='0–1', help_text='Higher roughness preserves more local relief and drainage complexity.', suggested='0.25–0.75')}
      </div>
    """

    terrain_review = _terrain_review_state(output_dir)
    terrain_derived = terrain_review.get("controls", {}) if isinstance(terrain_review.get("controls"), dict) else {}

    terrain_controls = f"""
      <h3>Terrain precondition values</h3>
      <p class='help'>These controls prepare Stage 3. Defaults are derived from Stage 1/2 where available; non-empty values are recorded as manual terrain precondition overrides. The current update uses them for terrain review diagnostics/workflow and prepares them for deeper terrain synthesis wiring.</p>
      <div class='grid compact-grid'>
        {_select_control('terrain_generation_mode', 'Terrain generation mode', qv('terrain_generation_mode', ['planet_profile','terrain_generation_mode'], planet_profile_cfg.get('terrain_generation_mode', terrain_derived.get('terrain_generation_mode', 'procedural_legacy'))), TERRAIN_MODE_OPTIONS, help_text='Switches Stage 3 backend. plate_tectonic_v1 preserves the Update 16 plate-owned stack; plate_history_v1 runs a compact time-evolved plate-history model; plate_history_v2 is a stronger experimental structural reconstruction; plate_history_v3 is the stable unified continuous-field model; plate_history_v4 is the recommended conservative terrain model built on stable v3.')}
        <div class='visual-editor-field'><label>Suppress polar land</label><input type='hidden' name='suppress_polar_land_present' value='1'><label><input type='checkbox' name='suppress_polar_land' style='width:auto'{' checked' if bool(qv('suppress_polar_land', ['planet_profile','suppress_polar_land'], planet_profile_cfg.get('suppress_polar_land', False))) else ''}> keep generated land away from the poles</label><p class='field-help'>In plate_history_v1/v2/v3/v4 this is a smooth crust-potential penalty during land formation, not a hard latitude eraser.</p></div>
        <div class='visual-editor-field'><label>Tectonic history Myr</label><input name='tectonic_history_myr' value='{qv('tectonic_history_myr', ['planet_profile','tectonic_history_myr'], planet_profile_cfg.get('tectonic_history_myr', ''))}'><p class='field-help'>For plate_history_v1/v2/v3/v4. Leave blank for auto; try 250, 750, or 1500.</p></div>
        <div class='visual-editor-field'><label>Tectonic timestep Myr</label><input name='tectonic_timestep_myr' value='{qv('tectonic_timestep_myr', ['planet_profile','tectonic_timestep_myr'], planet_profile_cfg.get('tectonic_timestep_myr', 2.5))}'><p class='field-help'>For plate_history_v1/v2/v3/v4. Requested timestep; internal sampling is capped for speed.</p></div>
        {_select_control('tectonic_grid_scale', 'Tectonic simulation grid', qv('tectonic_grid_scale', ['planet_profile','tectonic_grid_scale'], planet_profile_cfg.get('tectonic_grid_scale', 'legacy')), [('legacy', 'Legacy — Update 17 default'), ('preview', 'Preview — requested / 8'), ('normal', 'Normal — structural detail'), ('high', 'High — strong structural detail'), ('native', 'Native — strongest structural detail'), ('custom', 'Custom detail')], help_text='For plate_history_v1/v2/v3/v4. This is a requested detail label. Normal runs always use the stable macro-history grid plus high-resolution tectonic detail; raw/high history grids are deferred terrain research.')}
        <div class='visual-editor-field'><label>Tectonic history resolution policy</label><p class='field-help'>Stable hybrid is locked for normal runs: WorldGen uses the proven macro plate-history grid, then applies full-resolution tectonic detail. Higher-resolution raw history grids are deferred terrain research.</p></div>
        <div class='visual-editor-field'><label>Custom tectonic grid width</label><input name='tectonic_grid_width' value='{qv('tectonic_grid_width', ['planet_profile','tectonic_grid_width'], planet_profile_cfg.get('tectonic_grid_width', ''))}'><p class='field-help'>Used when grid scale is custom to document requested detail. Normal runs still use stable hybrid history resolution.</p></div>
        <div class='visual-editor-field'><label>Custom tectonic grid height</label><input name='tectonic_grid_height' value='{qv('tectonic_grid_height', ['planet_profile','tectonic_grid_height'], planet_profile_cfg.get('tectonic_grid_height', ''))}'><p class='field-help'>Optional; leave blank to preserve map aspect.</p></div>
        {_select_control('terrain_style', 'Terrain style preset', qv('terrain_style', ['terrain','terrain_style'], terrain_derived.get('terrain_style', 'derived_from_planet_physics')), [("", "Derived from planet physics"), ("derived_from_planet_physics", "Derived from planet physics"), ("earth_like_mixed_continents", "Earth-like mixed continents"), ("supercontinent_world", "Supercontinent world"), ("archipelago_world", "Archipelago world"), ("ocean_world", "Ocean world"), ("rugged_tectonic_world", "Rugged tectonic world"), ("old_eroded_shield_world", "Old eroded shield world"), ("volcanic_island_arc_world", "Volcanic island-arc world"), ("dry_highland_world", "Dry highland world")], help_text='Preset bias now used by the foundation terrain generator. Default derives from planet physics.')}
        {_select_control('supercontinent_tendency', 'Supercontinent tendency', qv('supercontinent_tendency', ['terrain','supercontinent_tendency'], terrain_derived.get('supercontinent_tendency', 'derived')), [("", "Derived"), ("derived", "Derived from planet profile"), ("rare", "Rare"), ("occasional", "Occasional"), ("common", "Common"), ("forced", "Forced"), ("suppressed", "Suppressed")], help_text='Does not railroad by default. High crustal asymmetry and low fragmentation make supercontinents more likely.')}
        <div class='visual-editor-field'><label>Target plate / province count</label><input name='target_plate_count' value='{qv('target_plate_count', ['terrain','target_plate_count'], terrain_derived.get('target_plate_count', ''))}'><p class='field-help'>Higher values should increase tectonic/province fragmentation and boundary complexity.</p></div>
        {_slider_control('fragmentation_tendency', 'Fragmentation tendency', qv('fragmentation_tendency', ['terrain','fragmentation_tendency'], terrain_derived.get('fragmentation_tendency')), min_v=0, max_v=1, step=0.01, unit='0–1', help_text='Controls whether landmasses tend to stay connected or break into multiple continents, microcontinents, and archipelagos.', suggested='Derived unless testing supercontinent/archipelago behavior')}
        {_slider_control('coastline_complexity', 'Coastline complexity', qv('coastline_complexity', ['terrain','coastline_complexity'], terrain_derived.get('coastline_complexity')), min_v=0, max_v=1, step=0.01, unit='0–1', help_text='Higher values should support more bays, peninsulas, drowned valleys, and irregular margins.', suggested='Derived by region; dry passive coasts may remain smooth')}
        {_slider_control('island_density', 'Island density', qv('island_density', ['terrain','island_density'], terrain_derived.get('island_density')), min_v=0, max_v=1, step=0.01, unit='0–1', help_text='Controls volcanic islands, continental fragments, shelf islands, and archipelago abundance.', suggested='Derived from volcanism, ocean target, and fragmentation')}
        {_slider_control('shelf_width_factor', 'Shelf width factor', qv('shelf_width_factor', ['terrain','shelf_width_factor'], terrain_derived.get('shelf_width_factor')), min_v=0, max_v=2, step=0.01, unit='factor', help_text='Affects shelves, shallow seas, coastal plains, and later deposition/delta behavior.', suggested='Derived from ocean target, erosion, and tectonic setting')}
        {_slider_control('continental_shelf_strength', 'Continental shelf strength', qv('continental_shelf_strength', ['terrain','continental_shelf_strength'], terrain_derived.get('continental_shelf_strength', 1.65)), min_v=0, max_v=3, step=0.05, unit='×', help_text='plate_history_v3/v4 multiplier for submerged continental shelf support. Higher values make passive continental margins wider and shallower without giving volcanic islands broad shelf halos.', suggested='1.65 default; try 2.0–2.4 for broader shelves')}
        {_slider_control('mountain_belt_strength', 'Mountain belt strength', qv('mountain_belt_strength', ['terrain','mountain_belt_strength'], terrain_derived.get('mountain_belt_strength')), min_v=0, max_v=3, step=0.01, unit='factor', help_text='Controls orogenic prominence and ruggedness; later climate/hydrology use this for barriers and gradients.', suggested='Derived from mountain factor and tectonic energy')}
        {_slider_control('rift_strength', 'Rift strength', qv('rift_strength', ['terrain','rift_strength'], terrain_derived.get('rift_strength')), min_v=0, max_v=1, step=0.01, unit='0–1', help_text='Controls rifts, stretched crust, breakup structures, and long basin corridors.', suggested='Derived from internal heat and fragmentation')}
        {_slider_control('interior_relief', 'Interior relief', qv('interior_relief', ['terrain','interior_relief'], terrain_derived.get('interior_relief')), min_v=0, max_v=1, step=0.01, unit='0–1', help_text='Controls shields, plateaus, basins, old ranges, and non-flat continental interiors.', suggested='Important for avoiding flat interiors and straight rivers')}
        {_slider_control('erosion_deposition_strength', 'Derived erosion/deposition field', qv('erosion_deposition_strength', ['terrain','erosion_deposition_strength'], terrain_derived.get('erosion_deposition_strength')), min_v=0, max_v=1, step=0.01, unit='0–1', help_text='Derived terrain condition for smoothing, sediment plains, alluvial basins, lowlands, and future deltas.', suggested='Derived from erosion, shelf width, and volatile delivery')}
        {_slider_control('erosion_deposition_multiplier', 'Erosion/deposition multiplier', qv('erosion_deposition_multiplier', ['terrain','erosion_deposition_multiplier'], terrain_derived.get('erosion_deposition_multiplier', 1.35)), min_v=0, max_v=3, step=0.05, unit='×', help_text='plate_history_v3/v4 master multiplier for erosion, basin/shelf deposition, old mountain wearing, and sediment fill. 0 nearly disables it; 1.35 is default; 2+ is strong testing.', suggested='Try 1.4–1.8 for stronger erosion/deposition')}
        {_select_control('diagnostic_detail', 'Diagnostic detail', qv('diagnostic_detail', ['terrain','diagnostic_detail'], terrain_derived.get('diagnostic_detail', 'standard')), [("", "Standard"), ("standard", "Standard"), ("high", "High detail"), ("minimal", "Minimal")], help_text='Controls how many terrain diagnostics future updates should write. Standard is the default.')}
      </div>
    """

    foundation_controls = f"""
      <h3>Terrain foundation values</h3>
      <p class='help'>Stage 3.1 controls broad land/ocean exposure, continent grouping, fragmentation, and island/microcontinent abundance.</p>
      <div class='grid compact-grid'>
        {_select_control('terrain_generation_mode', 'Terrain generation mode', qv('terrain_generation_mode', ['planet_profile','terrain_generation_mode'], planet_profile_cfg.get('terrain_generation_mode', terrain_derived.get('terrain_generation_mode', 'procedural_legacy'))), TERRAIN_MODE_OPTIONS, help_text='Switches Stage 3 backend. plate_tectonic_v1 preserves the Update 16 plate-owned stack; plate_history_v1 runs a compact time-evolved plate-history model; plate_history_v2 is a stronger experimental structural reconstruction; plate_history_v3 is the stable unified continuous-field model; plate_history_v4 is the recommended conservative terrain model built on stable v3.')}
        <div class='visual-editor-field'><label>Suppress polar land</label><input type='hidden' name='suppress_polar_land_present' value='1'><label><input type='checkbox' name='suppress_polar_land' style='width:auto'{' checked' if bool(qv('suppress_polar_land', ['planet_profile','suppress_polar_land'], planet_profile_cfg.get('suppress_polar_land', False))) else ''}> keep generated land away from the poles</label><p class='field-help'>In plate_history_v1/v2/v3/v4 this is a smooth crust-potential penalty during land formation, not a hard latitude eraser.</p></div>
        <div class='visual-editor-field'><label>Tectonic history Myr</label><input name='tectonic_history_myr' value='{qv('tectonic_history_myr', ['planet_profile','tectonic_history_myr'], planet_profile_cfg.get('tectonic_history_myr', ''))}'><p class='field-help'>For plate_history_v1/v2/v3/v4. Leave blank for auto; try 250, 750, or 1500.</p></div>
        <div class='visual-editor-field'><label>Tectonic timestep Myr</label><input name='tectonic_timestep_myr' value='{qv('tectonic_timestep_myr', ['planet_profile','tectonic_timestep_myr'], planet_profile_cfg.get('tectonic_timestep_myr', 2.5))}'><p class='field-help'>For plate_history_v1/v2/v3/v4. Requested timestep; internal sampling is capped for speed.</p></div>
        {_select_control('tectonic_grid_scale', 'Tectonic simulation grid', qv('tectonic_grid_scale', ['planet_profile','tectonic_grid_scale'], planet_profile_cfg.get('tectonic_grid_scale', 'legacy')), [('legacy', 'Legacy — Update 17 default'), ('preview', 'Preview — requested / 8'), ('normal', 'Normal — structural detail'), ('high', 'High — strong structural detail'), ('native', 'Native — strongest structural detail'), ('custom', 'Custom detail')], help_text='For plate_history_v1/v2/v3/v4. This is a requested detail label. Normal runs always use the stable macro-history grid plus high-resolution tectonic detail; raw/high history grids are deferred terrain research.')}
        <div class='visual-editor-field'><label>Tectonic history resolution policy</label><p class='field-help'>Stable hybrid is locked for normal runs: WorldGen uses the proven macro plate-history grid, then applies full-resolution tectonic detail. Higher-resolution raw history grids are deferred terrain research.</p></div>
        <div class='visual-editor-field'><label>Custom tectonic grid width</label><input name='tectonic_grid_width' value='{qv('tectonic_grid_width', ['planet_profile','tectonic_grid_width'], planet_profile_cfg.get('tectonic_grid_width', ''))}'><p class='field-help'>Used when grid scale is custom to document requested detail. Normal runs still use stable hybrid history resolution.</p></div>
        <div class='visual-editor-field'><label>Custom tectonic grid height</label><input name='tectonic_grid_height' value='{qv('tectonic_grid_height', ['planet_profile','tectonic_grid_height'], planet_profile_cfg.get('tectonic_grid_height', ''))}'><p class='field-help'>Optional; leave blank to preserve map aspect.</p></div>
        {_select_control('terrain_style', 'Terrain style preset', qv('terrain_style', ['terrain','terrain_style'], terrain_derived.get('terrain_style', 'derived_from_planet_physics')), [("", "Derived from planet physics"), ("derived_from_planet_physics", "Derived from planet physics"), ("earth_like_mixed_continents", "Earth-like mixed continents"), ("supercontinent_world", "Supercontinent world"), ("archipelago_world", "Archipelago world"), ("ocean_world", "Ocean world"), ("rugged_tectonic_world", "Rugged tectonic world"), ("old_eroded_shield_world", "Old eroded shield world"), ("volcanic_island_arc_world", "Volcanic island-arc world"), ("dry_highland_world", "Dry highland world")], help_text='Sets the broad terrain behavior before detailed provinces, relief, and coasts.')}
        {_slider_control('ocean_fraction_target', 'Ocean fraction target', qv('ocean_fraction_target', ['planet_physics','hydrosphere','ocean_fraction_target'], hydrosphere.get('ocean_fraction_target')), min_v=0.05, max_v=0.95, step=0.01, unit='fraction', help_text='Target ocean coverage. Rerun from terrain foundation after saving/applying to regenerate land exposure.', suggested='0.45–0.75 for mixed land/ocean')}
        {_select_control('supercontinent_tendency', 'Supercontinent tendency', qv('supercontinent_tendency', ['terrain','supercontinent_tendency'], terrain_derived.get('supercontinent_tendency', 'derived')), [("", "Derived"), ("derived", "Derived from planet profile"), ("rare", "Rare"), ("occasional", "Occasional"), ("common", "Common"), ("forced", "Forced"), ("suppressed", "Suppressed")], help_text='High values let one landmass dominate; derived remains the default.')}
        {_slider_control('fragmentation_tendency', 'Fragmentation tendency', qv('fragmentation_tendency', ['terrain','fragmentation_tendency'], terrain_derived.get('fragmentation_tendency')), min_v=0, max_v=1, step=0.01, unit='0–1', help_text='Breaks land into more continents, microcontinents, and archipelagos.', suggested='Derived unless testing')}
        {_slider_control('island_density', 'Island density', qv('island_density', ['terrain','island_density'], terrain_derived.get('island_density')), min_v=0, max_v=1, step=0.01, unit='0–1', help_text='Foundation-level island and fragment abundance.', suggested='Derived from ocean target, volcanism, and fragmentation')}
        <div class='visual-editor-field'><label>Target continent count</label><input name='target_continent_count' value='{qv('target_continent_count', ['terrain','target_continent_count'], terrain_derived.get('target_continent_count', ''))}'><p class='field-help'>Optional future-facing control for continent core count; leave blank for derived.</p></div>
      </div>
    """

    province_controls = f"""
      <h3>Tectonic province values</h3>
      <p class='help'>Stage 3.2 controls province count, crust block diversity, microcontinents, and how uneven the geological skeleton is.</p>
      <div class='grid compact-grid'>
        <div class='visual-editor-field'><label>Target plate / province count</label><input name='target_plate_count' value='{qv('target_plate_count', ['terrain','target_plate_count'], terrain_derived.get('target_plate_count', ''))}'><p class='field-help'>Higher values increase province fragmentation and diagnostic plate count.</p></div>
        {_slider_control('fragmentation_tendency', 'Fragmentation tendency', qv('fragmentation_tendency', ['terrain','fragmentation_tendency'], terrain_derived.get('fragmentation_tendency')), min_v=0, max_v=1, step=0.01, unit='0–1', help_text='Controls whether provinces split into many blocks or stay broad.', suggested='Derived from terrain profile')}
        {_slider_control('province_type_diversity', 'Province type diversity', qv('province_type_diversity', ['terrain','province_type_diversity'], terrain_derived.get('province_type_diversity')), min_v=0, max_v=1, step=0.01, unit='0–1', help_text='Future-facing control for cratons, shelves, arcs, terranes, basins, and oceanic crust variety.', suggested='Derived')}
        {_slider_control('microcontinent_tendency', 'Microcontinent / terrane tendency', qv('microcontinent_tendency', ['terrain','microcontinent_tendency'], terrain_derived.get('microcontinent_tendency')), min_v=0, max_v=1, step=0.01, unit='0–1', help_text='Encourages accreted terranes, fragments, and small crustal blocks.', suggested='Derived from fragmentation and ocean target')}
        {_slider_control('crustal_asymmetry', 'Crustal asymmetry', qv('crustal_asymmetry', ['terrain','crustal_asymmetry'], terrain_derived.get('crustal_asymmetry')), min_v=0, max_v=1, step=0.01, unit='0–1', help_text='Higher values allow dominant crustal blocks and hemispheric imbalance.', suggested='Derived from Stage 1 formation context')}
      </div>
    """

    boundary_controls = f"""
      <h3>Crust and boundary values</h3>
      <p class='help'>Stage 3.3 controls active/passive margins, rifts, transforms, subduction/collision tendency, and boundary neatness.</p>
      <div class='grid compact-grid'>
        {_slider_control('plate_motion_speed', 'Plate motion speed', qv('plate_motion_speed', ['terrain','plate_motion_speed'], terrain_derived.get('plate_motion_speed')), min_v=0, max_v=1.5, step=0.01, unit='0–1.5', help_text='Plate Terrain v1 native velocity magnitude. Higher values make relative boundaries more active.', suggested='Derived from heat and tectonic energy')}
        {_slider_control('plate_motion_chaos', 'Plate motion chaos', qv('plate_motion_chaos', ['terrain','plate_motion_chaos'], terrain_derived.get('plate_motion_chaos')), min_v=0, max_v=1, step=0.01, unit='0–1', help_text='Blends coherent Euler-like rotation with local plate-specific directions. Higher values make less orderly plate motion.', suggested='Derived from fragmentation and volcanism')}
        {_slider_control('convergence_bias', 'Convergence bias', qv('convergence_bias', ['terrain','convergence_bias'], terrain_derived.get('convergence_bias')), min_v=0, max_v=1, step=0.01, unit='0–1', help_text='Bias for collision/subduction-like relative motion classification in plate_tectonic_v1.', suggested='Derived')}
        {_slider_control('divergence_bias', 'Divergence bias', qv('divergence_bias', ['terrain','divergence_bias'], terrain_derived.get('divergence_bias')), min_v=0, max_v=1, step=0.01, unit='0–1', help_text='Bias for rift/spreading-like relative motion classification in plate_tectonic_v1.', suggested='Derived')}
        {_slider_control('transform_bias', 'Transform bias', qv('transform_bias', ['terrain','transform_bias'], terrain_derived.get('transform_bias')), min_v=0, max_v=1, step=0.01, unit='0–1', help_text='Bias for shear/transform-like relative motion classification in plate_tectonic_v1.', suggested='Derived')}
        {_slider_control('active_margin_bias', 'Active margin bias', qv('active_margin_bias', ['terrain','active_margin_bias'], terrain_derived.get('active_margin_bias')), min_v=0, max_v=1, step=0.01, unit='0–1', help_text='Raises convergent/collision/arc-like boundary share.', suggested='Derived')}
        {_slider_control('passive_margin_bias', 'Passive margin bias', qv('passive_margin_bias', ['terrain','passive_margin_bias'], terrain_derived.get('passive_margin_bias')), min_v=0, max_v=1, step=0.01, unit='0–1', help_text='Raises passive margins, shelves, and smoother coastal margins.', suggested='Derived')}
        {_slider_control('rift_strength', 'Rift strength', qv('rift_strength', ['terrain','rift_strength'], terrain_derived.get('rift_strength')), min_v=0, max_v=1, step=0.01, unit='0–1', help_text='Controls divergent/rift boundary expression and breakup corridors.', suggested='Derived from heat and fragmentation')}
        {_slider_control('boundary_neatness', 'Boundary neatness', qv('boundary_neatness', ['terrain','boundary_neatness'], terrain_derived.get('boundary_neatness')), min_v=0, max_v=1, step=0.01, unit='0–1', help_text='Lower values should produce more broken, offset, and diffuse boundaries; high values are intentionally artificial/neat.', suggested='Keep low/moderate for natural terrain')}
        {_slider_control('boundary_width_factor', 'Boundary width factor', qv('boundary_width_factor', ['terrain','boundary_width_factor'], terrain_derived.get('boundary_width_factor')), min_v=0, max_v=2, step=0.01, unit='factor', help_text='Controls diffuse vs narrow boundary zones.', suggested='Derived')}
      </div>
    """

    relief_controls = f"""
      <h3>Mountains, basins, rifts, and interior relief values</h3>
      <p class='help'>Stage 3.4 controls visible relief before coast, erosion, climate, and rivers consume it.</p>
      <div class='grid compact-grid'>
        {_slider_control('mountain_belt_strength', 'Mountain belt strength', qv('mountain_belt_strength', ['terrain','mountain_belt_strength'], terrain_derived.get('mountain_belt_strength')), min_v=0, max_v=3, step=0.01, unit='factor', help_text='Scales collision belts, arcs, old sutures, and orographic barriers.', suggested='Derived from tectonic energy')}
        {_slider_control('rift_strength', 'Rift strength', qv('rift_strength', ['terrain','rift_strength'], terrain_derived.get('rift_strength')), min_v=0, max_v=1, step=0.01, unit='0–1', help_text='Controls rift valleys, rift shoulders, and breakup corridors.', suggested='Derived')}
        {_slider_control('interior_relief', 'Interior relief', qv('interior_relief', ['terrain','interior_relief'], terrain_derived.get('interior_relief')), min_v=0, max_v=1, step=0.01, unit='0–1', help_text='Adds shields, plateaus, old ranges, and non-flat interiors.', suggested='Important for avoiding straight rivers')}
        {_slider_control('basin_strength', 'Basin strength', qv('basin_strength', ['terrain','basin_strength'], terrain_derived.get('basin_strength')), min_v=0, max_v=1, step=0.01, unit='0–1', help_text='Future-facing control for foreland, rift, sedimentary, and interior basins.', suggested='Derived')}
        {_slider_control('plateau_strength', 'Plateau strength', qv('plateau_strength', ['terrain','plateau_strength'], terrain_derived.get('plateau_strength')), min_v=0, max_v=1, step=0.01, unit='0–1', help_text='Controls broad uplifted interiors and high plateaus.', suggested='Derived')}
        {_slider_control('shield_highland_strength', 'Shield/highland strength', qv('shield_highland_strength', ['terrain','shield_highland_strength'], terrain_derived.get('shield_highland_strength')), min_v=0, max_v=1, step=0.01, unit='0–1', help_text='Controls ancient eroded highlands and cratonic shields.', suggested='Derived')}
      </div>
    """

    coast_controls = f"""
      <h3>Coasts, shelves, and islands values</h3>
      <p class='help'>Stage 3.5 now feeds the coastline/shelf/island generator. These values affect shelf width, active/fjorded margins, smooth passive coasts, archipelago density, and island shape irregularity.</p>
      <div class='grid compact-grid'>
        {_slider_control('coastline_complexity', 'Coastline complexity', qv('coastline_complexity', ['terrain','coastline_complexity'], terrain_derived.get('coastline_complexity')), min_v=0, max_v=1, step=0.01, unit='0–1', help_text='Higher values support bays, peninsulas, fjords, drowned valleys, and rougher margins.', suggested='Derived regionally')}
        {_slider_control('shelf_width_factor', 'Shelf width factor', qv('shelf_width_factor', ['terrain','shelf_width_factor'], terrain_derived.get('shelf_width_factor')), min_v=0, max_v=2, step=0.01, unit='factor', help_text='Affects shelves, shallow seas, coastal plains, and shelf islands.', suggested='Derived')}
        {_slider_control('island_density', 'Island density', qv('island_density', ['terrain','island_density'], terrain_derived.get('island_density')), min_v=0, max_v=1, step=0.01, unit='0–1', help_text='Controls island/archipelago abundance.', suggested='Derived')}
        {_slider_control('coastal_ruggedness', 'Coastal ruggedness', qv('coastal_ruggedness', ['terrain','coastal_ruggedness'], terrain_derived.get('coastal_ruggedness')), min_v=0, max_v=1, step=0.01, unit='0–1', help_text='Controls active/rugged coastline expression and local shore relief.', suggested='Derived from tectonics, volcanism, roughness, and erosion')}
        {_slider_control('fjord_tendency', 'Fjord/drowned-valley tendency', qv('fjord_tendency', ['terrain','fjord_tendency'], terrain_derived.get('fjord_tendency')), min_v=0, max_v=1, step=0.01, unit='0–1', help_text='Controls narrow inlets, drowned valleys, and high-relief coastal cuts.', suggested='Derived from relief, water target, and ruggedness')}
        {_slider_control('coastal_plain_bias', 'Coastal plain bias', qv('coastal_plain_bias', ['terrain','coastal_plain_bias'], terrain_derived.get('coastal_plain_bias')), min_v=0, max_v=1, step=0.01, unit='0–1', help_text='Controls smooth low coastal plains and passive/sedimentary margins.', suggested='Derived from shelf width, erosion, and lower tectonic activity')}
        {_slider_control('island_shape_irregularity', 'Island shape irregularity', qv('island_shape_irregularity', ['terrain','island_shape_irregularity'], terrain_derived.get('island_shape_irregularity')), min_v=0, max_v=1, step=0.01, unit='0–1', help_text='Higher values generate more multi-lobed/chain-like islands instead of oval dots.', suggested='Derived from coastline complexity, volcanism, and fragmentation')}
      </div>
    """

    erosion_controls = f"""
      <h3>Erosion and deposition values</h3>
      <p class='help'>Stage 3.6 controls smoothing, sediment transfer, basins, lowlands, and drainage-readiness.</p>
      <div class='grid compact-grid'>
        {_slider_control('erosion', 'Planet erosion factor', qv('erosion', ['planet_physics','geology','erosion'], geology.get('erosion')), min_v=0, max_v=1.5, step=0.01, unit='factor', help_text='Stage 2 geology control used by terrain and hydrology.', suggested='0.25–1.1')}
        {_slider_control('erosion_deposition_strength', 'Derived erosion/deposition field', qv('erosion_deposition_strength', ['terrain','erosion_deposition_strength'], terrain_derived.get('erosion_deposition_strength')), min_v=0, max_v=1, step=0.01, unit='0–1', help_text='Derived terrain-specific condition for smoothing, alluvial basins, and lowlands.', suggested='Derived')}
        {_slider_control('erosion_deposition_multiplier', 'Erosion/deposition multiplier', qv('erosion_deposition_multiplier', ['terrain','erosion_deposition_multiplier'], terrain_derived.get('erosion_deposition_multiplier', 1.35)), min_v=0, max_v=3, step=0.05, unit='×', help_text='plate_history_v3/v4 master multiplier. 0 nearly disables extra erosion/deposition; 1.35 is default; 1.6–2.0 gives stronger old-mountain erosion, sediment fill, and shelf/basin deposition.', suggested='Try 1.4–1.8')}
        {_slider_control('deposition_strength', 'Deposition strength', qv('deposition_strength', ['terrain','deposition_strength'], terrain_derived.get('deposition_strength')), min_v=0, max_v=1, step=0.01, unit='0–1', help_text='Controls sediment plains, deltas, and basin fill.', suggested='Derived')}
        {_slider_control('valley_carving_strength', 'Valley carving strength', qv('valley_carving_strength', ['terrain','valley_carving_strength'], terrain_derived.get('valley_carving_strength')), min_v=0, max_v=1, step=0.01, unit='0–1', help_text='Controls pre-hydrology valley corridor incision and river-readiness.', suggested='Derived')}
        {_slider_control('sediment_supply_strength', 'Sediment supply strength', qv('sediment_supply_strength', ['terrain','sediment_supply_strength'], terrain_derived.get('sediment_supply_strength')), min_v=0, max_v=1, step=0.01, unit='0–1', help_text='Controls how strongly mountains/uplands feed sediment into plains and basins.', suggested='Derived')}
        {_slider_control('coastal_plain_strength', 'Coastal plain strength', qv('coastal_plain_strength', ['terrain','coastal_plain_strength'], terrain_derived.get('coastal_plain_strength')), min_v=0, max_v=1, step=0.01, unit='0–1', help_text='Controls broad passive-margin and shelf-adjacent lowland/plains expression.', suggested='Derived')}
        {_slider_control('alluvial_fan_strength', 'Alluvial fan strength', qv('alluvial_fan_strength', ['terrain','alluvial_fan_strength'], terrain_derived.get('alluvial_fan_strength')), min_v=0, max_v=1, step=0.01, unit='0–1', help_text='Controls fan-like deposition where high relief drops into basins or plains.', suggested='Derived')}
        {_slider_control('floodplain_strength', 'Floodplain strength', qv('floodplain_strength', ['terrain','floodplain_strength'], terrain_derived.get('floodplain_strength')), min_v=0, max_v=1, step=0.01, unit='0–1', help_text='Controls broad low-gradient floodplain tendency before the hydrology pass.', suggested='Derived')}
        {_slider_control('terrain_maturity', 'Terrain maturity', qv('terrain_maturity', ['terrain','terrain_maturity'], terrain_derived.get('terrain_maturity')), min_v=0, max_v=1, step=0.01, unit='0–1', help_text='Controls how worked-over, softened, and sediment-mature the terrain becomes.', suggested='Derived')}
      </div>
    """

    final_terrain_controls = f"""
      <h3>Final terrain review values</h3>
      <p class='help'>Stage 3.7 is for final acceptance before climate and hydrology. It shows the most important terrain-level knobs together.</p>
      {foundation_controls}
      {relief_controls}
      <h3>Diagnostics</h3>
      <div class='grid compact-grid'>
        {_select_control('diagnostic_detail', 'Diagnostic detail', qv('diagnostic_detail', ['terrain','diagnostic_detail'], terrain_derived.get('diagnostic_detail', 'standard')), [("", "Standard"), ("standard", "Standard"), ("high", "High detail"), ("minimal", "Minimal")], help_text='Controls how many terrain diagnostics future updates should write.')}
      </div>
    """

    if stage == "solar-system":
        controls = solar_controls
    elif stage == "planet-physics":
        controls = physics_controls
    elif stage == "terrain-foundation-mask":
        controls = foundation_controls
    elif stage == "terrain-tectonic-provinces":
        controls = province_controls
    elif stage == "terrain-crust-and-boundaries":
        controls = boundary_controls
    elif stage == "terrain-mountains-basins-rifts":
        controls = relief_controls
    elif stage == "terrain-coasts-shelves-islands":
        controls = coast_controls
    elif stage == "terrain-erosion-deposition":
        controls = erosion_controls
    elif stage == "terrain-finalization-recentering":
        controls = final_terrain_controls
    elif stage == "climate":
        controls = f"""
          <h3>Climate precondition values</h3>
          <p class='help'>Climate controls start with backend selection. <code>seasonal_v5</code> adds component-based moisture/rainfall coupling on top of the refined basin-ocean mode, <code>seasonal_v4</code> is the refined structured atmosphere + basin-ocean mode, <code>seasonal_v3</code> preserves the first basin-ocean mode, <code>seasonal_v2</code> preserves the structured atmosphere-only mode, <code>seasonal_v1</code> is the stable seasonal model, and <code>legacy</code> preserves the previous heuristic model for comparison.</p>
          <div class='grid compact-grid'>
            <div><label>Climate generation mode</label><select name='climate_generation_mode'><option value='seasonal_v5'>seasonal_v5 — component moisture + refined basin ocean</option><option value='seasonal_v4'>seasonal_v4 — refined atmosphere + basin ocean</option><option value='seasonal_v3'>seasonal_v3 — first basin ocean</option><option value='seasonal_v2'>seasonal_v2 — structured atmosphere only</option><option value='seasonal_v1'>seasonal_v1 — stable seasonal overhaul</option><option value='legacy'>legacy — previous heuristic model</option></select><p class='field-help'>Saved through stage overrides; rerun from climate to compare modes on the same terrain.</p></div>
          </div>
          {physics_controls}
        """
    elif stage == "hydrology":
        controls = f"""
          <h3>Hydrology precondition values</h3>
          <p class='help'>Dedicated hydrology controls will come with the hydrology review. For now these focus on terrain relief, erosion/deposition, and water target.</p>
          {foundation_controls}
          {relief_controls}
          {erosion_controls}
        """
    elif stage in {"biomes", "regions", "outputs"}:
        controls = f"""
          <h3>{_safe_text(_stage_label(stage))} upstream controls</h3>
          <p class='help'>Dedicated controls for this stage will be added when it is reviewed. These upstream values are the strongest current levers.</p>
          {physics_controls}
          {final_terrain_controls}
        """
    else:
        controls = "<p class='muted'>No guided edit controls are cataloged for this stage yet.</p>"

    return f"""
<section class='card stage-edit-card' id='edit-values'>
  <h2>Edit values</h2>
  <p class='help'>This is the predictable place to adjust stage values and preconditions. Saving records the values as manual/visual-editor overrides; apply them before rerolling if you want validation and stale-stage checks.</p>
  <form method='post' action='/quick-edit' class='quick-form'>
    <input type='hidden' name='output_dir' value='{_safe_text(output_dir)}'>
    <input type='hidden' name='redirect_to' value='{_safe_text(redirect_to)}'>
    {controls}
    <div class='row'>
      <button type='submit'>Save edited values</button>
      <label><input type='checkbox' name='apply_now' style='width:auto'> apply immediately and validate</label>
      <a class='button secondary' href='/overrides?output_dir={urllib.parse.quote(str(output_dir))}'>Advanced: raw override JSON</a>
    </div>
  </form>
</section>
"""


def _stage_support_tools_html(output_dir: Path, stage: str) -> str:
    q = urllib.parse.quote(str(output_dir))
    links: list[str] = []
    if stage == "solar-system":
        links.extend([
            f"<a class='button secondary' href='/system-report?output_dir={q}'>Stage 1 report</a>",
            f"<a class='button secondary' href='/system?output_dir={q}'>Solar-system viewer</a>",
            f"<a class='button secondary' href='/planet?output_dir={q}'>Main Planet viewer</a>",
        ])
        help_text = "Stage 1 tools are here because the report, orbital viewer, and Main Planet selection all belong to Solar System review."
    elif stage == "planet-physics":
        links.extend([
            f"<a class='button secondary' href='/physics?output_dir={q}'>Planet Physics review</a>",
            f"<a class='button secondary' href='/planet?output_dir={q}'>Upstream Main Planet details</a>",
        ])
        help_text = "Stage 2 tools are here because the physics review depends on the selected Main Planet and Stage 1 context."
    elif stage.startswith("terrain"):
        terrain_report = output_dir / "terrain_diagnostics" / "terrain_review_report.txt"
        if terrain_report.exists():
            links.append(f"<a class='button secondary' href='{_file_url(terrain_report)}' target='_blank'>Terrain review report</a>")
        links.extend([
            f"<a class='button secondary' href='/compare?output_dir={q}'>Compare terrain diagnostics</a>",
            f"<a class='button secondary' href='/files?output_dir={q}'>Open terrain diagnostic files</a>",
        ])
        help_text = "Terrain tools are grouped here because Stage 3 has multiple expensive sub-stages and diagnostic folders."
    else:
        links.extend([
            f"<a class='button secondary' href='/compare?output_dir={q}'>Compare maps</a>",
        ])
        help_text = "Map-oriented tools live here for generated terrain, climate, hydrology, biome, region, and output stages."
    advanced = " ".join([
        f"<a class='button secondary' href='/files?output_dir={q}'>File browser</a>",
        f"<a class='button secondary' href='/overrides?output_dir={q}'>Advanced: raw overrides</a>",
        f"<a class='button secondary' href='/jobs'>Jobs</a>",
    ])
    return f"""
<details class='stage-tools-card'>
  <summary><strong>Stage-specific tools</strong> <span class='muted'>viewers, reports, and utilities for this stage</span></summary>
  <p class='help'>{_safe_text(help_text)}</p>
  <div class='row'>{''.join(links)}</div>
  <details class='advanced-tools'>
    <summary>Advanced tools</summary>
    <p class='help'>These are fallback utilities for debugging, diagnostics, raw files, and emergency override editing. Normal value edits should happen through the stage controls.</p>
    <div class='row'>{advanced}</div>
  </details>
</details>
"""


def _pipeline_active_job_banner(output_dir: Path) -> str:
    with JOBS_LOCK:
        jobs = [j for j in JOBS.values() if j.output_dir is not None and Path(j.output_dir) == output_dir and j.status in {"queued", "running", "cancelling"}]
    if not jobs:
        return ""
    rows = ""
    for job in sorted(jobs, key=lambda item: item.created_at, reverse=True):
        cancel = ""
        if job.status in {"queued", "running", "cancelling"}:
            cancel = (
                f"<form method='post' action='/job-action' style='display:inline'>"
                f"<input type='hidden' name='action' value='cancel'>"
                f"<input type='hidden' name='job_id' value='{_safe_text(job.id)}'>"
                f"<button type='submit' class='danger small'>Cancel</button></form>"
            )
        rows += f"<li><a href='/job/{_safe_text(job.id)}'>{_safe_text(job.label)}</a> — <span class='stale'>{_safe_text(job.status)}</span> {cancel}</li>"
    return f"<section class='card active-job-banner'><h2>Active job</h2><ul>{rows}</ul></section>"


def _pipeline_advanced_tools_html(output_dir: Path) -> str:
    q = urllib.parse.quote(str(output_dir))
    return f"""
<details class='merged-dashboard advanced-tools'>
  <summary><strong>Advanced tools</strong> <span class='muted'>raw files, raw overrides, jobs, and run management</span></summary>
  <p class='help'>Most work should happen through the Pipeline and Stage Review pages. These tools remain available for debugging, recovery, and direct file inspection.</p>
  <div class='row'>
    <a class='button secondary' href='/files?output_dir={q}'>File browser</a>
    <a class='button secondary' href='/overrides?output_dir={q}'>Advanced: raw override JSON</a>
    <a class='button secondary' href='/jobs'>Jobs</a>
    <a class='button secondary' href='/compare?output_dir={q}'>Compare maps</a>
  </div>
  <div class='row'>
    <form method='post' action='/run-admin' style='display:inline'>
      <input type='hidden' name='action' value='forget-run'>
      <input type='hidden' name='output_dir' value='{_safe_text(output_dir)}'>
      <input type='hidden' name='redirect_to' value='/'>
      <button type='submit' class='secondary'>Forget from recent list</button>
    </form>
    <form method='post' action='/run-admin' style='display:inline' onsubmit="return confirm('Delete this run folder and all files? This cannot be undone.');">
      <input type='hidden' name='action' value='delete-run'>
      <input type='hidden' name='output_dir' value='{_safe_text(output_dir)}'>
      <input type='hidden' name='redirect_to' value='/pipeline?output_dir={q}'>
      <button type='submit' class='danger'>Delete run folder</button>
    </form>
  </div>
</details>
"""


def _card_action_form(output_dir: Path, stage: str, action: str, label: str, *, cls: str = "secondary", target_stage: str | None = None, title: str = "") -> str:
    target = f"<input type='hidden' name='target_stage' value='{_safe_text(target_stage)}'>" if target_stage else ""
    title_attr = f" title='{_safe_text(title)}'" if title else ""
    return (
        f"<form method='post' action='/stage-workflow' class='card-action-form'>"
        f"<input type='hidden' name='output_dir' value='{_safe_text(output_dir)}'>"
        f"<input type='hidden' name='stage' value='{_safe_text(stage)}'>"
        f"<input type='hidden' name='action' value='{_safe_text(action)}'>"
        f"{target}"
        f"<input type='hidden' name='yes' value='on'>"
        f"<input type='hidden' name='skip_json' value='on'>"
        f"<button type='submit' class='{_safe_text(cls)} small'{title_attr}>{_safe_text(label)}</button>"
        f"</form>"
    )


def _pipeline_stage_cards_html(output_dir: Path) -> str:
    status_by_stage = _stage_status_map(output_dir)
    cards = []
    for stage in STAGE_ORDER:
        row = status_by_stage.get(stage, {})
        status = str(row.get("status", "missing")) or "missing"
        status_cls = "complete" if status == "complete" else ("stale" if status == "stale" else "missing")
        accepted = _is_stage_accepted(output_dir, stage)
        warnings = _stage_warning_items(output_dir, stage)
        next_label = STAGE_OUTPUT_SUMMARIES.get(stage, "Stage outputs and diagnostics.")
        review_url = f"/stage?output_dir={urllib.parse.quote(str(output_dir))}&stage={urllib.parse.quote(stage)}"
        accepted_chip = "<span class='pill ok-pill'>accepted</span>" if accepted else "<span class='pill'>not accepted</span>"
        primary_label = "Review stage" if status != "missing" else "Open stage"
        action_buttons = [f"<a class='button secondary small' href='{review_url}' title='Open the unified stage review workspace for details, edit controls, preconditions, warnings, provenance, and stage-specific tools.'>{primary_label}</a>"]
        if status == "missing":
            action_buttons.append(_card_action_form(output_dir, stage, "run-to-selected", "Generate", target_stage=stage, title="Generate through this stage and stop. This is the quick path when you do not need to review earlier stages."))
        else:
            action_buttons.append(_card_action_form(output_dir, stage, "reroll-stage", "Reroll", title="Regenerate only this stage and stop here. Downstream stages may become stale but are not automatically regenerated."))
            if not accepted and status == "complete":
                action_buttons.append(_card_action_form(output_dir, stage, "accept-stage", "Accept", cls="secondary", title="Mark the current version of this stage as reviewed/approved."))
            if _next_stage(stage):
                action_buttons.append(_card_action_form(output_dir, stage, "continue-next", "Continue", cls="secondary", title="Run only to the next stage. Use run-to controls for longer unattended runs."))
        cards.append(f"""
<article class='pipeline-stage-card {status_cls}'>
  <div class='pipeline-stage-head'><h3>{_safe_text(_stage_label(stage))}</h3><span class='{status_cls}'>{_safe_text(status)}</span></div>
  <p class='muted small'>{_safe_text(next_label)}</p>
  <div class='row'>{accepted_chip}{_warning_badge(warnings)}</div>
  <div class='row card-action-row'>{''.join(action_buttons)}</div>
</article>
""")
    return "<div class='pipeline-grid'>" + "".join(cards) + "</div>"


class WorldGenUIHandler(BaseHTTPRequestHandler):
    server_version = "WorldGenWebUI/0.60"

    def do_GET(self) -> None:  # noqa: N802
        parsed = urllib.parse.urlparse(self.path)
        params = urllib.parse.parse_qs(parsed.query)
        try:
            if parsed.path == "/":
                self._send_html(self._page_home(params))
            elif parsed.path == "/pipeline":
                self._send_html(self._page_pipeline(params))
            elif parsed.path == "/stage":
                self._send_html(self._page_stage_workspace(params))
            elif parsed.path == "/run":
                output_dir = _first(params, "output_dir")
                suffix = f"?output_dir={urllib.parse.quote(output_dir)}" if output_dir else ""
                self._redirect(f"/pipeline{suffix}")
            elif parsed.path == "/overrides":
                self._send_html(self._page_overrides(params))
            elif parsed.path == "/system":
                self._send_html(self._page_system(params))
            elif parsed.path == "/system-report":
                self._send_html(self._page_system_report(params))
            elif parsed.path == "/planet":
                self._send_html(self._page_planet(params))
            elif parsed.path == "/physics":
                self._send_html(self._page_physics(params))
            elif parsed.path == "/map":
                self._send_html(self._page_map(params))
            elif parsed.path == "/map-index":
                self._send_html(self._page_map_index(params))
            elif parsed.path == "/globe":
                self._send_html(self._page_globe(params))
            elif parsed.path == "/compare":
                self._send_html(self._page_compare(params))
            elif parsed.path == "/files":
                self._send_html(self._page_files(params))
            elif parsed.path == "/csv":
                self._send_html(self._page_csv(params))
            elif parsed.path == "/jobs":
                self._send_html(self._page_jobs(params))
            elif parsed.path == "/map-info":
                self._send_map_info(params)
            elif parsed.path == "/contour-overlay":
                self._send_contour_overlay(params)
            elif parsed.path == "/map-data-image":
                self._send_map_data_image(params)
            elif parsed.path.startswith("/job/"):
                jid = parsed.path.rsplit("/", 1)[-1]
                self._send_html(self._page_job(jid))
            elif parsed.path.startswith("/file/"):
                self._send_file(parsed.path[len("/file/"):])
            else:
                self.send_error(404, "Not found")
        except Exception as exc:  # pragma: no cover - makes local UI recoverable.
            self._send_html(self._page_error(exc), status=500)

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length", "0"))
        form = _parse_form(self.rfile.read(length))
        parsed = urllib.parse.urlparse(self.path)
        try:
            if parsed.path == "/new-run":
                self._handle_new_run(form)
            elif parsed.path == "/stage-action":
                self._handle_stage_action(form)
            elif parsed.path == "/save-overrides":
                self._handle_save_overrides(form)
            elif parsed.path == "/quick-edit":
                self._handle_quick_edit(form)
            elif parsed.path == "/run-admin":
                self._handle_run_admin(form)
            elif parsed.path == "/solar-action":
                self._handle_solar_action(form)
            elif parsed.path == "/physics-action":
                self._handle_physics_action(form)
            elif parsed.path == "/job-action":
                self._handle_job_action(form)
            elif parsed.path == "/stage-workflow":
                self._handle_stage_workflow(form)
            else:
                self.send_error(404, "Not found")
        except Exception as exc:  # pragma: no cover - makes local UI recoverable.
            self._send_html(self._page_error(exc), status=500)

    # ------------------------------------------------------------------
    # POST handlers
    # ------------------------------------------------------------------

    def _handle_new_run(self, form: dict[str, list[str]]) -> None:
        output_dir = Path(_first(form, "output_dir", "worldgen_web_run")).expanduser()
        preset = _first(form, "preset", "generated").strip() or "generated"
        if preset not in {"generated", "synthetic-earth", "real-earth-terrain"}:
            preset = "generated"

        if preset == "generated":
            cmd = [sys.executable, "-m", "worldgen.pipeline", "new", "--output-dir", str(output_dir)]
        else:
            cmd = [sys.executable, "-m", "worldgen.main", "--preset", preset, "--output-dir", str(output_dir)]

        for key, flag in [
            ("seed", "--seed"),
            ("map_width", "--map-width"),
            ("map_height", "--map-height"),
            ("planet_count", "--planet-count"),
            ("image_max_width", "--image-max-width"),
        ]:
            if preset != "generated" and key == "planet_count":
                continue
            value = _int_arg(form, key)
            if value is not None:
                cmd += [flag, value]
        if preset == "generated":
            terrain_mode = _first(form, "terrain_mode", "procedural_legacy").strip() or "procedural_legacy"
            if terrain_mode in {"procedural_legacy", "plate_tectonic_v1", "plate_history_v1", "plate_history_v2", "plate_history_v3", "plate_history_v4"}:
                cmd += ["--terrain-mode", terrain_mode]
            for key, flag in [
                ("tectonic_history_myr", "--tectonic-history-myr"),
                ("tectonic_timestep_myr", "--tectonic-timestep-myr"),
            ]:
                value = _float_form_value(form, key)
                if value is not None:
                    cmd += [flag, str(value)]
            tectonic_grid_scale = _first(form, "tectonic_grid_scale", "").strip()
            if tectonic_grid_scale in {"legacy", "preview", "normal", "high", "native", "custom"}:
                cmd += ["--tectonic-grid-scale", tectonic_grid_scale]
            for key, flag in [
                ("tectonic_grid_width", "--tectonic-grid-width"),
                ("tectonic_grid_height", "--tectonic-grid-height"),
            ]:
                value = _int_arg(form, key)
                if value is not None:
                    cmd += [flag, value]
            for key, flag in [
                ("continental_shelf_strength", "--continental-shelf-strength"),
                ("erosion_deposition_multiplier", "--erosion-deposition-strength"),
                ("shelf_width_factor", "--shelf-width-factor"),
                ("v4_topology_strength", "--v4-topology-strength"),
                ("v4_island_strength", "--v4-island-strength"),
                ("v4_rift_strength", "--v4-rift-strength"),
            ]:
                value = _float_form_value(form, key)
                if value is not None:
                    cmd += [flag, str(value)]
            for key, flag in [
                ("star_class", "--star-class"),
                ("system_architecture", "--system-architecture"),
                ("main_planet_preference", "--main-planet-preference"),
                ("moon_strength", "--moon-strength"),
            ]:
                value = _first(form, key).strip()
                if value:
                    cmd += [flag, value]
            for key, flag in [
                ("star_mass", "--star-mass"),
                ("star_age", "--star-age"),
                ("metallicity", "--metallicity"),
            ]:
                value = _float_form_value(form, key)
                if value is not None:
                    cmd += [flag, str(value)]
            if _checkbox(form, "no_major_moon"):
                cmd.append("--no-major-moon")
        config = _first(form, "config").strip()
        if config:
            cmd += ["--config", config]
        run_to = normalize_stage(_first(form, "run_to", "solar-system"))
        if preset == "generated":
            cmd += ["--run-to", run_to]
        for checkbox, flag in [
            ("preview", "--preview"),
            ("fast", "--fast"),
            ("skip_hydrology", "--skip-hydrology"),
            ("skip_biomes", "--skip-biomes"),
            ("skip_regions", "--skip-regions"),
            ("skip_json", "--skip-json"),
            ("no_images", "--no-images"),
            ("yes", "--yes"),
            ("save_rasters", "--save-rasters"),
            ("skip_diagnostics", "--skip-diagnostics"),
            ("full_res_images", "--full-res-images"),
            ("suppress_polar_land", "--suppress-polar-land"),
        ]:
            if _checkbox(form, checkbox):
                cmd.append(flag)
        koppen = _first(form, "koppen_detail").strip()
        if koppen:
            cmd += ["--koppen-detail", koppen]
        climate_mode = _first(form, "climate_mode").strip()
        if climate_mode:
            cmd += ["--climate-mode", climate_mode]
        job_label = f"Create {_run_mode_label(preset)}"
        if preset == "generated":
            job_label += f" to {run_to}"
        job = start_job(job_label, cmd, output_dir)
        self._redirect(f"/job/{job.id}")

    def _handle_stage_action(self, form: dict[str, list[str]]) -> None:
        output_dir = Path(_first(form, "output_dir")).expanduser()
        action = _first(form, "action", "status")
        stage = _first(form, "stage", "solar-system")
        cmd = [sys.executable, "-m", "worldgen.pipeline"]
        label = action
        if action in {"run-to", "run-from", "run-stage"}:
            cmd += [action, normalize_stage(stage)]
            label = f"{action} {normalize_stage(stage)}"
        elif action in {"maps", "status", "validate", "apply-overrides"}:
            cmd.append(action)
            label = action
        else:
            raise ValueError(f"Unsupported action: {action}")
        cmd += ["--output-dir", str(output_dir)]
        if action == "apply-overrides" and _checkbox(form, "validate"):
            cmd.append("--validate")
        if action in {"run-to", "run-from", "run-stage", "maps"}:
            if _checkbox(form, "yes"):
                cmd.append("--yes")
            if _checkbox(form, "no_images"):
                cmd.append("--no-images")
            image_max_width = _int_arg(form, "image_max_width")
            if image_max_width is not None:
                cmd += ["--image-max-width", image_max_width]
            if _checkbox(form, "skip_json"):
                cmd.append("--skip-json")
            if _checkbox(form, "save_rasters"):
                cmd.append("--save-rasters")
            if _checkbox(form, "skip_diagnostics"):
                cmd.append("--skip-diagnostics")
        job = start_job(label, cmd, output_dir)
        self._redirect(f"/job/{job.id}")

    def _handle_save_overrides(self, form: dict[str, list[str]]) -> None:
        output_dir = Path(_first(form, "output_dir")).expanduser()
        text = _first(form, "overrides")
        # Validate JSON before writing so a paste mistake does not silently break the file.
        parsed = json.loads(text)
        path = ensure_layout(output_dir) / "config" / "stage_overrides.json"
        path.write_text(json.dumps(parsed, indent=2), encoding="utf-8")
        _record_override_provenance(output_dir, parsed, source="manual_raw_json", note="Saved from the raw JSON override editor.")
        remember_run(output_dir)
        if _checkbox(form, "apply_now"):
            cmd = [sys.executable, "-m", "worldgen.pipeline", "apply-overrides", "--output-dir", str(output_dir), "--validate"]
            job = start_job("apply-overrides", cmd, output_dir)
            self._redirect(f"/job/{job.id}")
        else:
            self._redirect(f"/overrides?output_dir={urllib.parse.quote(str(output_dir))}&saved=1")

    def _handle_job_action(self, form: dict[str, list[str]]) -> None:
        action = _first(form, "action", "").strip()
        jid = _first(form, "job_id", "").strip()
        if action != "cancel" or not jid:
            raise ValueError("Unsupported job action")
        with JOBS_LOCK:
            job = JOBS.get(jid)
            if job is None:
                raise ValueError(f"No active in-memory job with ID {jid}")
            if job.status not in {"queued", "running", "cancelling"}:
                self._redirect(f"/job/{jid}")
                return
            job.cancel_requested = True
            job.status = "cancelling"
            proc = job.process
        if proc is not None and proc.poll() is None:
            try:
                proc.terminate()
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
        _persist_job(job)
        self._redirect(f"/job/{jid}")

    def _handle_run_admin(self, form: dict[str, list[str]]) -> None:
        action = _first(form, "action", "").strip()
        output_dir = Path(_first(form, "output_dir")).expanduser() if _first(form, "output_dir").strip() else None
        redirect_to = _first(form, "redirect_to", "/") or "/"

        if action == "forget-run":
            if output_dir is not None:
                forget_run(output_dir)
            sep = "&" if "?" in redirect_to else "?"
            self._redirect(f"{redirect_to}{sep}run_message={urllib.parse.quote('Removed run from recent list.')}")
            return

        if action == "purge-missing":
            removed = purge_missing_recent_runs()
            sep = "&" if "?" in redirect_to else "?"
            self._redirect(f"{redirect_to}{sep}run_message={urllib.parse.quote(f'Purged {removed} missing run entries.')}")
            return

        if action == "delete-run":
            if output_dir is None:
                raise ValueError("delete-run requires output_dir")
            ok, message = delete_run_directory(output_dir)
            sep = "&" if "?" in redirect_to else "?"
            if redirect_to.startswith(("/run", "/pipeline")):
                redirect_base = "/"
            else:
                redirect_base = redirect_to
            self._redirect(f"{redirect_base}{sep}run_message={urllib.parse.quote(message)}")
            return

        raise ValueError(f"Unsupported run admin action: {action}")

    def _handle_solar_action(self, form: dict[str, list[str]]) -> None:
        output_dir = Path(_first(form, "output_dir")).expanduser()
        action = _first(form, "action", "").strip()
        ensure_layout(output_dir)
        config_path = output_dir / "config" / "resolved_config.json"
        config_data = _read_json(config_path, {})
        if not isinstance(config_data, dict):
            config_data = {}
        solar = _solar_system_state(output_dir) or {}
        star = solar.get("star", {}) if isinstance(solar.get("star"), dict) else {}
        planets = solar.get("planets", []) if isinstance(solar.get("planets"), list) else []
        main = next((p for p in planets if isinstance(p, dict) and p.get("is_main_planet")), {})

        def write_config() -> None:
            _write_json(config_path, config_data)

        def set_seed_offset(offset: int) -> None:
            seed = config_data.get("seed")
            try:
                seed_i = int(seed)
            except Exception:
                seed_i = int(time.time()) % 999_999_937
            config_data["seed"] = seed_i + offset

        def start_solar_rerun(label: str) -> None:
            write_config()
            cmd = [sys.executable, "-m", "worldgen.pipeline", "run-from", "solar-system", "--stop-at", "solar-system", "--output-dir", str(output_dir), "--yes", "--skip-json"]
            job = start_job(label, cmd, output_dir)
            self._redirect(f"/job/{job.id}")

        if action == "accept-solar" or action == "accept-continue":
            path = output_dir / "config" / "stage_acceptance.json"
            acceptance = _read_json(path, {})
            if not isinstance(acceptance, dict):
                acceptance = {}
            acceptance["solar-system"] = {
                "accepted_at": now_stamp(),
                "accepted_main_planet": main.get("name"),
                "architecture": solar.get("architecture") or (solar.get("diagnostics") or {}).get("architecture"),
                "note": "Accepted from the Web UI review workflow.",
            }
            _write_json(path, acceptance)
            if action == "accept-continue":
                cmd = [sys.executable, "-m", "worldgen.pipeline", "run-to", "planet-physics", "--output-dir", str(output_dir), "--yes", "--skip-json"]
                job = start_job("Accept Solar System and continue to planet-physics", cmd, output_dir)
                self._redirect(f"/job/{job.id}")
            else:
                self._redirect(f"/system?output_dir={urllib.parse.quote(str(output_dir))}&solar_message=Solar%20system%20accepted")
            return

        if action == "lock-star":
            config_data["star"] = {
                "stellar_class": star.get("spectral_type") or star.get("stellar_class"),
                "mass_solar": star.get("mass_solar"),
                "age_gyr": star.get("age_gyr"),
                "metallicity": star.get("metallicity"),
            }
            write_config()
            self._redirect(f"/system?output_dir={urllib.parse.quote(str(output_dir))}&solar_message=Current%20star%20locked%20into%20resolved_config.json")
            return

        if action == "lock-main-planet":
            overrides_path = output_dir / "config" / "stage_overrides.json"
            overrides = _read_json(overrides_path, {})
            if not isinstance(overrides, dict):
                overrides = {}
            mp = overrides.setdefault("main_planet", {})
            if isinstance(mp, dict) and main:
                mp["mass_earth"] = main.get("mass_earth")
                mp["radius_earth"] = main.get("radius_earth")
                mp["surface_gravity_g"] = main.get("surface_gravity_g")
                mp["stellar_flux_earth"] = main.get("stellar_flux_earth")
                orbit = main.get("orbit", {}) if isinstance(main.get("orbit"), dict) else {}
                mp["orbit"] = {
                    "semi_major_axis_au": orbit.get("semi_major_axis_au"),
                    "eccentricity": orbit.get("eccentricity"),
                    "orbital_period_days": orbit.get("orbital_period_days"),
                }
            _write_json(overrides_path, overrides)
            self._redirect(f"/system?output_dir={urllib.parse.quote(str(output_dir))}&solar_message=Current%20Main%20Planet%20values%20saved%20to%20stage_overrides.json")
            return

        if action == "reroll-same":
            set_seed_offset(1)
            start_solar_rerun("Reroll solar system with same settings")
            return

        if action == "reroll-star":
            config_data["star"] = {"stellar_class": None, "mass_solar": None, "age_gyr": None, "metallicity": None}
            set_seed_offset(101)
            start_solar_rerun("Reroll solar system with unlocked star")
            return

        if action == "reroll-layout":
            if star:
                config_data["star"] = {
                    "stellar_class": star.get("spectral_type") or star.get("stellar_class"),
                    "mass_solar": star.get("mass_solar"),
                    "age_gyr": star.get("age_gyr"),
                    "metallicity": star.get("metallicity"),
                }
            set_seed_offset(17)
            start_solar_rerun("Reroll planet layout with current star locked")
            return

        if action == "reroll-selection":
            # Main Planet selection is deterministic for a finished layout, so this
            # rerolls the layout while preserving star and high-level architecture.
            if star:
                config_data["star"] = {
                    "stellar_class": star.get("spectral_type") or star.get("stellar_class"),
                    "mass_solar": star.get("mass_solar"),
                    "age_gyr": star.get("age_gyr"),
                    "metallicity": star.get("metallicity"),
                }
            if solar.get("architecture"):
                system_cfg = config_data.setdefault("system", {})
                if isinstance(system_cfg, dict):
                    system_cfg["architecture_type"] = solar.get("architecture")
            set_seed_offset(29)
            start_solar_rerun("Reroll Main Planet candidates with star/architecture locked")
            return

        raise ValueError(f"Unsupported solar action: {action}")

    def _handle_physics_action(self, form: dict[str, list[str]]) -> None:
        output_dir = Path(_first(form, "output_dir")).expanduser()
        action = _first(form, "action", "").strip()
        ensure_layout(output_dir)
        physics_path = output_dir / "state" / "02_planet_physics.json"
        physics = _read_json(physics_path, {})
        if not isinstance(physics, dict):
            physics = {}
        review = physics.get("review", {}) if isinstance(physics.get("review"), dict) else {}
        stage_seed_offsets_path = output_dir / "config" / "stage_seed_offsets.json"
        stage_seed_offsets = _read_json(stage_seed_offsets_path, {})
        if not isinstance(stage_seed_offsets, dict):
            stage_seed_offsets = {}
        overrides_path = output_dir / "config" / "stage_overrides.json"
        overrides = _read_json(overrides_path, {})
        if not isinstance(overrides, dict):
            overrides = {}
        pp = overrides.setdefault("planet_physics", {})
        if not isinstance(pp, dict):
            pp = {}
            overrides["planet_physics"] = pp

        def save_overrides() -> None:
            _write_json(overrides_path, overrides)

        def bump_seed(offset: int) -> None:
            current = 0
            try:
                current = int(stage_seed_offsets.get("planet-physics", 0) or 0)
            except Exception:
                current = 0
            stage_seed_offsets["planet-physics"] = current + offset
            _write_json(stage_seed_offsets_path, stage_seed_offsets)

        def clear_physics_overrides() -> None:
            pp.clear()
            pp.update({"rotation": {}, "atmosphere": {}, "hydrosphere": {}, "geology": {}})
            overrides["planet_physics"] = pp

        def lock_section(section: str) -> None:
            current = physics.get(section, {}) if isinstance(physics.get(section), dict) else {}
            if not current:
                return
            # Copy only primitive editable values, not report/review metadata.
            allowed = {
                "rotation": ["rotation_period_hours", "axial_tilt_degrees", "solar_day_hours", "year_length_days"],
                "atmosphere": ["pressure_bar", "nitrogen_fraction", "oxygen_fraction", "carbon_dioxide_ppm", "argon_fraction", "water_vapor_factor", "greenhouse_warming_k", "estimated_mean_surface_temp_k", "estimated_mean_surface_temp_c"],
                "hydrosphere": ["volatile_fraction", "ocean_fraction_target", "ocean_fraction_actual", "water_inventory_class", "ice_cap_tendency"],
                "geology": ["internal_heat", "volcanism", "erosion", "mountain_factor", "crater_density", "surface_roughness", "geology_class"],
            }.get(section, list(current.keys()))
            pp[section] = {key: current.get(key) for key in allowed if key in current}

        def start_stage2_rerun(label: str) -> None:
            save_overrides()
            cmd = [sys.executable, "-m", "worldgen.pipeline", "run-from", "planet-physics", "--stop-at", "planet-physics", "--output-dir", str(output_dir), "--yes", "--skip-json"]
            job = start_job(label, cmd, output_dir)
            self._redirect(f"/job/{job.id}")

        if action == "accept-physics" or action == "accept-continue":
            path = output_dir / "config" / "stage_acceptance.json"
            acceptance = _read_json(path, {})
            if not isinstance(acceptance, dict):
                acceptance = {}
            acceptance["planet-physics"] = {
                "accepted_at": now_stamp(),
                "archetype": review.get("archetype"),
                "note": "Accepted from the Web UI review workflow.",
            }
            _write_json(path, acceptance)
            if action == "accept-continue":
                cmd = [sys.executable, "-m", "worldgen.pipeline", "run-to", "terrain-foundation-mask", "--output-dir", str(output_dir), "--yes", "--skip-json"]
                job = start_job("Accept Planet Physics and continue to terrain-foundation-mask", cmd, output_dir)
                self._redirect(f"/job/{job.id}")
            else:
                self._redirect(f"/physics?output_dir={urllib.parse.quote(str(output_dir))}&physics_message=Planet%20Physics%20accepted")
            return

        if action.startswith("lock-"):
            section = action.removeprefix("lock-")
            if section not in {"rotation", "atmosphere", "hydrosphere", "geology"}:
                raise ValueError(f"Unsupported lock section: {section}")
            lock_section(section)
            save_overrides()
            self._redirect(f"/physics?output_dir={urllib.parse.quote(str(output_dir))}&physics_message={urllib.parse.quote(_humanize_key(section) + ' locked into stage_overrides.json')}")
            return

        if action == "reroll-all":
            clear_physics_overrides()
            bump_seed(3)
            start_stage2_rerun("Reroll all planet physics")
            return

        reroll_sections = {
            "reroll-rotation": "rotation",
            "reroll-atmosphere": "atmosphere",
            "reroll-hydrosphere": "hydrosphere",
            "reroll-geology": "geology",
        }
        if action in reroll_sections:
            clear_physics_overrides()
            unlocked = reroll_sections[action]
            for section in ("rotation", "atmosphere", "hydrosphere", "geology"):
                if section != unlocked:
                    lock_section(section)
            bump_seed({"rotation": 5, "atmosphere": 7, "hydrosphere": 11, "geology": 13}.get(unlocked, 3))
            start_stage2_rerun(f"Reroll {unlocked} only")
            return

        if action == "preset":
            preset = _first(form, "preset", "").strip()
            presets = {
                "earth_like_balanced": {
                    "rotation": {"rotation_period_hours": 24.0, "axial_tilt_degrees": 23.5},
                    "atmosphere": {"pressure_bar": 1.0, "carbon_dioxide_ppm": 420, "water_vapor_factor": 1.0, "greenhouse_warming_k": 33.0},
                    "hydrosphere": {"ocean_fraction_target": 0.62},
                    "geology": {"internal_heat": 0.85, "volcanism": 0.75, "erosion": 1.05, "mountain_factor": 1.0, "surface_roughness": 0.45},
                },
                "dry_rugged_world": {"hydrosphere": {"ocean_fraction_target": 0.36}, "geology": {"mountain_factor": 1.22, "erosion": 0.78, "surface_roughness": 0.72}},
                "ocean_world": {"hydrosphere": {"ocean_fraction_target": 0.80}, "atmosphere": {"water_vapor_factor": 1.25}, "geology": {"erosion": 1.15, "surface_roughness": 0.42}},
                "high_volcanism_world": {"geology": {"internal_heat": 1.25, "volcanism": 1.35, "mountain_factor": 1.18, "surface_roughness": 0.78}},
                "cold_high_tilt_world": {"rotation": {"axial_tilt_degrees": 42.0}, "atmosphere": {"greenhouse_warming_k": 39.0}, "hydrosphere": {"ocean_fraction_target": 0.54}},
                "thick_atmosphere_super_earth": {"atmosphere": {"pressure_bar": 2.1, "greenhouse_warming_k": 38.0, "water_vapor_factor": 1.15}, "rotation": {"rotation_period_hours": 32.0}},
                "low_gravity_thin_atmosphere": {"atmosphere": {"pressure_bar": 0.62, "greenhouse_warming_k": 25.0, "water_vapor_factor": 0.65}, "hydrosphere": {"ocean_fraction_target": 0.48}},
            }
            if preset not in presets:
                raise ValueError(f"Unsupported Stage 2 preset: {preset}")
            clear_physics_overrides()
            overrides["planet_physics"] = presets[preset]
            save_overrides()
            self._redirect(f"/physics?output_dir={urllib.parse.quote(str(output_dir))}&physics_message={urllib.parse.quote('Saved Stage 2 preset: ' + _humanize_key(preset))}")
            return

        raise ValueError(f"Unsupported physics action: {action}")

    def _handle_stage_workflow(self, form: dict[str, list[str]]) -> None:
        output_dir = Path(_first(form, "output_dir")).expanduser()
        stage = normalize_stage(_first(form, "stage", "solar-system"))
        action = _first(form, "action", "").strip()
        ensure_layout(output_dir)

        if action == "accept-stage":
            _record_stage_acceptance(output_dir, stage)
            self._redirect(f"/stage?output_dir={urllib.parse.quote(str(output_dir))}&stage={urllib.parse.quote(stage)}&stage_message={urllib.parse.quote(_stage_label(stage) + ' accepted')}")
            return

        if action == "reroll-stage":
            cmd = [sys.executable, "-m", "worldgen.pipeline", "run-from", stage, "--stop-at", stage, "--output-dir", str(output_dir), "--yes", "--skip-json"]
            job = start_job(f"Reroll only {stage}", cmd, output_dir)
            self._redirect(f"/job/{job.id}")
            return

        if action == "continue-next":
            next_stage = _next_stage(stage)
            if not next_stage:
                self._redirect(f"/stage?output_dir={urllib.parse.quote(str(output_dir))}&stage={urllib.parse.quote(stage)}&stage_message=No%20later%20stage%20exists")
                return
            cmd = [sys.executable, "-m", "worldgen.pipeline", "run-to", next_stage, "--output-dir", str(output_dir), "--yes", "--skip-json"]
            job = start_job(f"Continue to {next_stage}", cmd, output_dir)
            self._redirect(f"/job/{job.id}")
            return

        if action == "terrain-lowres-test":
            preview_dir = _terrain_lowres_preview_dir(output_dir)
            _prepare_terrain_lowres_preview_run(output_dir, preview_dir, width=2048, height=1024)
            cmd = [sys.executable, "-m", "worldgen.pipeline", "run-from", "terrain-foundation-mask", "--stop-at", "terrain-finalization-recentering", "--output-dir", str(preview_dir), "--yes", "--skip-json", "--image-max-width", "2048"]
            job = start_job("Low-res terrain test 2048x1024", cmd, preview_dir)
            self._redirect(f"/job/{job.id}")
            return

        if action == "terrain-full-commit":
            cmd = [sys.executable, "-m", "worldgen.pipeline", "run-from", "terrain-foundation-mask", "--stop-at", "terrain-finalization-recentering", "--output-dir", str(output_dir), "--yes", "--skip-json"]
            job = start_job("Commit full-resolution terrain", cmd, output_dir)
            self._redirect(f"/job/{job.id}")
            return

        if action == "run-to-selected":
            target = normalize_stage(_first(form, "target_stage", "outputs"))
            cmd = [sys.executable, "-m", "worldgen.pipeline", "run-to", target, "--output-dir", str(output_dir)]
            if _checkbox(form, "yes"):
                cmd.append("--yes")
            if _checkbox(form, "skip_json"):
                cmd.append("--skip-json")
            job = start_job(f"Run to {target} without review", cmd, output_dir)
            self._redirect(f"/job/{job.id}")
            return

        raise ValueError(f"Unsupported stage workflow action: {action}")

    def _handle_quick_edit(self, form: dict[str, list[str]]) -> None:
        output_dir = Path(_first(form, "output_dir")).expanduser()
        ensure_layout(output_dir)
        path = output_dir / "config" / "stage_overrides.json"
        overrides = _read_json(path, {})
        if not isinstance(overrides, dict):
            overrides = {}

        field_map: dict[str, list[str]] = {
            "star_mass_solar": ["star", "mass_solar"],
            "star_luminosity_solar": ["star", "luminosity_solar"],
            "main_radius_earth": ["main_planet", "radius_earth"],
            "main_mass_earth": ["main_planet", "mass_earth"],
            "main_gravity_g": ["main_planet", "surface_gravity_g"],
            "orbit_semi_major_axis_au": ["main_planet", "orbit", "semi_major_axis_au"],
            "orbit_eccentricity": ["main_planet", "orbit", "eccentricity"],
            "rotation_period_hours": ["planet_physics", "rotation", "rotation_period_hours"],
            "axial_tilt_degrees": ["planet_physics", "rotation", "axial_tilt_degrees"],
            "pressure_bar": ["planet_physics", "atmosphere", "pressure_bar"],
            "co2_ppm": ["planet_physics", "atmosphere", "carbon_dioxide_ppm"],
            "greenhouse_warming_k": ["planet_physics", "atmosphere", "greenhouse_warming_k"],
            "water_vapor_factor": ["planet_physics", "atmosphere", "water_vapor_factor"],
            "ocean_fraction_target": ["planet_physics", "hydrosphere", "ocean_fraction_target"],
            "volcanism": ["planet_physics", "geology", "volcanism"],
            "erosion": ["planet_physics", "geology", "erosion"],
            "mountain_factor": ["planet_physics", "geology", "mountain_factor"],
            "surface_roughness": ["planet_physics", "geology", "surface_roughness"],
            "target_plate_count": ["terrain", "target_plate_count"],
            "target_continent_count": ["terrain", "target_continent_count"],
            "fragmentation_tendency": ["terrain", "fragmentation_tendency"],
            "province_type_diversity": ["terrain", "province_type_diversity"],
            "microcontinent_tendency": ["terrain", "microcontinent_tendency"],
            "crustal_asymmetry": ["terrain", "crustal_asymmetry"],
            "active_margin_bias": ["terrain", "active_margin_bias"],
            "passive_margin_bias": ["terrain", "passive_margin_bias"],
            "boundary_neatness": ["terrain", "boundary_neatness"],
            "boundary_width_factor": ["terrain", "boundary_width_factor"],
            "plate_motion_speed": ["terrain", "plate_motion_speed"],
            "plate_motion_chaos": ["terrain", "plate_motion_chaos"],
            "convergence_bias": ["terrain", "convergence_bias"],
            "divergence_bias": ["terrain", "divergence_bias"],
            "transform_bias": ["terrain", "transform_bias"],
            "coastline_complexity": ["terrain", "coastline_complexity"],
            "island_density": ["terrain", "island_density"],
            "shelf_width_factor": ["terrain", "shelf_width_factor"],
            "coastal_ruggedness": ["terrain", "coastal_ruggedness"],
            "fjord_tendency": ["terrain", "fjord_tendency"],
            "coastal_plain_bias": ["terrain", "coastal_plain_bias"],
            "island_shape_irregularity": ["terrain", "island_shape_irregularity"],
            "mountain_belt_strength": ["terrain", "mountain_belt_strength"],
            "rift_strength": ["terrain", "rift_strength"],
            "interior_relief": ["terrain", "interior_relief"],
            "basin_strength": ["terrain", "basin_strength"],
            "plateau_strength": ["terrain", "plateau_strength"],
            "shield_highland_strength": ["terrain", "shield_highland_strength"],
            "fjord_tendency": ["terrain", "fjord_tendency"],
            "coastal_plain_bias": ["terrain", "coastal_plain_bias"],
            "erosion_deposition_strength": ["terrain", "erosion_deposition_strength"],
            "erosion_deposition_multiplier": ["terrain", "erosion_deposition_multiplier"],
            "continental_shelf_strength": ["terrain", "continental_shelf_strength"],
            "v4_topology_strength": ["terrain", "v4_topology_strength"],
            "v4_island_strength": ["terrain", "v4_island_strength"],
            "v4_rift_strength": ["terrain", "v4_rift_strength"],
            "deposition_strength": ["terrain", "deposition_strength"],
            "valley_carving_strength": ["terrain", "valley_carving_strength"],
            "sediment_supply_strength": ["terrain", "sediment_supply_strength"],
            "coastal_plain_strength": ["terrain", "coastal_plain_strength"],
            "alluvial_fan_strength": ["terrain", "alluvial_fan_strength"],
            "floodplain_strength": ["terrain", "floodplain_strength"],
            "terrain_maturity": ["terrain", "terrain_maturity"],
            "tectonic_history_myr": ["planet_profile", "tectonic_history_myr"],
            "tectonic_timestep_myr": ["planet_profile", "tectonic_timestep_myr"],
            "tectonic_grid_width": ["planet_profile", "tectonic_grid_width"],
            "tectonic_grid_height": ["planet_profile", "tectonic_grid_height"],
        }
        for form_key, path_parts in field_map.items():
            value = _float_form_value(form, form_key)
            if value is not None:
                # Keep obvious integer controls tidy in JSON.
                if form_key in {"target_plate_count", "target_continent_count"}:
                    value = int(value)
                _set_nested(overrides, path_parts, value)

        string_field_map: dict[str, list[str]] = {
            "system_architecture_type": ["system", "architecture_type"],
            "main_planet_preference": ["system", "main_planet_preference"],
            "tectonic_grid_scale": ["planet_profile", "tectonic_grid_scale"],
            "moon_strength_preference": ["system", "moon_strength_preference"],
            "star_stellar_class": ["star", "stellar_class"],
            "terrain_generation_mode": ["planet_profile", "terrain_generation_mode"],
            "climate_generation_mode": ["planet_profile", "climate_generation_mode"],
            "terrain_style": ["terrain", "terrain_style"],
            "supercontinent_tendency": ["terrain", "supercontinent_tendency"],
            "diagnostic_detail": ["terrain", "diagnostic_detail"],
        }
        for form_key, path_parts in string_field_map.items():
            if form_key not in form:
                continue
            value = _first(form, form_key).strip()
            if form_key == "system_architecture_type" and value in {"", "random", "auto"}:
                _set_nested(overrides, path_parts, None)
            elif value:
                _set_nested(overrides, path_parts, value)

        if "require_major_moon_present" in form:
            _set_nested(overrides, ["system", "require_major_moon"], _checkbox(form, "require_major_moon"))
        if "suppress_polar_land_present" in form:
            _set_nested(overrides, ["planet_profile", "suppress_polar_land"], _checkbox(form, "suppress_polar_land"))

        path.write_text(json.dumps(overrides, indent=2), encoding="utf-8")
        _record_override_provenance(output_dir, overrides, source="manual_visual_editor", note="Saved from the visual parameter editor.")
        remember_run(output_dir)
        redirect_to = _first(form, "redirect_to").strip() or f"/pipeline?output_dir={urllib.parse.quote(str(output_dir))}"
        separator = "&" if "?" in redirect_to else "?"
        if _checkbox(form, "apply_now"):
            cmd = [sys.executable, "-m", "worldgen.pipeline", "apply-overrides", "--output-dir", str(output_dir), "--validate"]
            job = start_job("quick-edit apply-overrides", cmd, output_dir)
            self._redirect(f"/job/{job.id}")
        else:
            self._redirect(f"{redirect_to}{separator}quick_saved=1")

    # ------------------------------------------------------------------
    # Pages
    # ------------------------------------------------------------------

    def _page_shell(self, title: str, body: str, *, refresh: int | None = None) -> str:
        refresh_tag = f'<meta http-equiv="refresh" content="{refresh}">' if refresh else ""
        return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  {refresh_tag}
  <title>{_safe_text(title)} · WorldGen</title>
  <style>
    :root {{ --bg:#0f172a; --panel:#111827; --muted:#94a3b8; --text:#e5e7eb; --line:#334155; --accent:#38bdf8; --bad:#f87171; --ok:#4ade80; --warn:#facc15; }}
    * {{ box-sizing: border-box; }}
    body {{ margin:0; font-family: system-ui, -apple-system, Segoe UI, sans-serif; color:var(--text); background:linear-gradient(180deg,#020617,#0f172a); }}
    header {{ position:sticky; top:0; z-index:2; background:rgba(2,6,23,.94); border-bottom:1px solid var(--line); padding:12px 24px; display:flex; gap:18px; align-items:center; flex-wrap:wrap; }}
    header a {{ color:var(--text); text-decoration:none; padding:7px 10px; border-radius:8px; }}
    header a:hover {{ background:#1e293b; }}
    main {{ padding:18px; width:min(1960px, calc(100vw - 18px)); max-width:none; margin:auto; }}
    h1,h2,h3 {{ margin-top:0; }}
    .grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(320px,1fr)); gap:18px; }}
    .card {{ background:rgba(15,23,42,.92); border:1px solid var(--line); border-radius:14px; padding:18px; box-shadow:0 10px 30px rgba(0,0,0,.18); min-width:0; overflow-wrap:anywhere; }}
    label {{ display:block; margin:10px 0 4px; color:#cbd5e1; font-size:.92rem; }}
    input, select, textarea {{ width:100%; padding:9px 10px; border-radius:8px; border:1px solid var(--line); background:#020617; color:var(--text); }}
    textarea {{ min-height:520px; font-family: ui-monospace, SFMono-Regular, Consolas, monospace; font-size:.9rem; }}
    button, .button {{ display:inline-block; border:0; background:var(--accent); color:#00111c; padding:9px 13px; border-radius:9px; font-weight:700; cursor:pointer; text-decoration:none; margin-top:12px; }}
    button.secondary, .button.secondary {{ background:#1e293b; color:var(--text); border:1px solid var(--line); }}
    button.danger, .button.danger {{ background:#7f1d1d; color:#fee2e2; border:1px solid #ef4444; }}
    .muted {{ color:var(--muted); }}
    .ok {{ color:var(--ok); }} .bad {{ color:var(--bad); }} .warn {{ color:var(--warn); }}
    table {{ width:100%; border-collapse:collapse; table-layout:auto; }} th,td {{ border-bottom:1px solid var(--line); padding:8px; text-align:left; vertical-align:top; overflow-wrap:anywhere; word-break:break-word; }}
    th {{ color:#bfdbfe; }}
    pre {{ white-space:pre-wrap; background:#020617; border:1px solid var(--line); border-radius:12px; padding:14px; overflow:auto; }}
    .maps {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(240px,1fr)); gap:14px; }}
    .mapcard {{ background:#020617; border:1px solid var(--line); border-radius:12px; padding:10px; }}
    .mapcard img {{ width:100%; height:auto; border-radius:8px; display:block; background:#000; }}
    .map-category-pill {{ display:inline-block; margin:4px 0 7px; padding:2px 7px; border-radius:999px; background:rgba(56,189,248,.10); border:1px solid rgba(56,189,248,.30); color:#bae6fd; font-size:.75rem; }}
    .map-browser-controls {{ display:grid; grid-template-columns:minmax(260px,1fr) minmax(220px,320px) auto auto; gap:10px; align-items:end; margin-bottom:10px; }}
    .map-browser-controls .inline-check {{ margin-bottom:8px; white-space:nowrap; }}
    .warning-list {{ list-style:none; padding-left:0; display:grid; gap:8px; }}
    .warning-item {{ border:1px solid var(--line); background:#020617; border-radius:10px; padding:9px 10px; }}
    .warning-item .warning-level {{ display:inline-block; min-width:76px; font-weight:800; margin-right:8px; }}
    .warning-item.warn {{ border-color:rgba(250,204,21,.35); }}
    .warning-item.bad {{ border-color:rgba(248,113,113,.45); }}
    .warning-message {{ color:#e5e7eb; }}
    .warning-details {{ color:#94a3b8; font-size:.82rem; margin-top:4px; padding-left:84px; }}
    details {{ margin-top:10px; }}
    summary {{ cursor:pointer; color:#bfdbfe; }}
    .folder-card {{ margin-top:16px; background:#020617; border:1px dashed var(--line); border-radius:12px; padding:12px; }}
    .columns {{ columns:3 260px; }}
    .compact-grid {{ grid-template-columns:repeat(auto-fit,minmax(210px,1fr)); gap:12px; }}
    .quick-form h3 {{ margin-top:18px; margin-bottom:6px; color:#bfdbfe; }}
    .system-viewer svg {{ width:100%; height:auto; display:block; }}
    .help {{ color:var(--muted); font-size:.86rem; margin:.25rem 0 .75rem; }}
    .field-help {{ color:#94a3b8; font-size:.78rem; margin-top:3px; line-height:1.25; }}
    .resolution-panel {{ background:#020617; border:1px solid var(--line); border-radius:14px; padding:14px; margin:12px 0; }}
    .resolution-panel h3 {{ margin-bottom:4px; }}
    .resolution-inputs {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(190px,1fr)); gap:10px; }}
    .resolution-buttons {{ display:flex; gap:8px; flex-wrap:wrap; margin-top:10px; }}
    .star-visual {{ display:flex; gap:12px; align-items:center; background:#020617; border:1px solid var(--line); border-radius:14px; padding:12px; min-height:92px; }}
    .star-disk {{ width:68px; height:68px; border-radius:50%; background:radial-gradient(circle at 35% 30%, #fff 0%, var(--star-color) 35%, rgba(2,6,23,0) 72%); box-shadow:0 0 24px var(--star-color), inset -8px -9px 18px rgba(0,0,0,.22); flex:0 0 auto; }}
    .star-visual strong {{ display:block; font-size:1.35rem; color:#f8fafc; }}
    .star-visual span, .star-visual small {{ display:block; color:#94a3b8; }}
    .planet-dot {{ display:inline-flex; align-items:center; justify-content:center; width:22px; height:22px; border-radius:50%; background:radial-gradient(circle at 35% 30%, #fff 0%, var(--planet-color) 42%, #020617 100%); border:1px solid rgba(255,255,255,.35); vertical-align:middle; margin-right:6px; color:#020617; font-size:.72rem; font-weight:900; text-shadow:0 1px 0 rgba(255,255,255,.25); flex:0 0 auto; }}
    .planet-dot span {{ transform:translateY(-.5px); }}
    .planet-dot.main {{ box-shadow:0 0 0 3px rgba(34,197,94,.5); }}
    .orbit-details {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(220px,1fr)); gap:12px; margin-top:12px; }}
    .orbit-card {{ background:#020617; border:1px solid var(--line); border-radius:12px; padding:10px; }}
    .orbit-card b {{ color:#bfdbfe; }}
    .orbit-legend {{ display:flex; gap:10px; flex-wrap:wrap; color:#cbd5e1; font-size:.88rem; margin-top:8px; }}
    .orbit-legend span {{ display:inline-flex; gap:6px; align-items:center; }}
    .row {{ display:flex; gap:10px; flex-wrap:wrap; align-items:center; }}
    .pill {{ display:inline-block; padding:2px 8px; border-radius:999px; background:#1e293b; color:#cbd5e1; font-size:.85rem; }}
    .complete {{ color:var(--ok); }} .missing {{ color:var(--muted); }} .stale {{ color:var(--warn); }} .failed {{ color:var(--bad); }}
    .viewer-wrap {{ height:78vh; overflow:auto; background:#020617; border:1px solid var(--line); border-radius:14px; padding:12px; cursor:crosshair; position:relative; }}
    .viewer-overlay-tools {{ position:fixed; top:78px; right:330px; z-index:420; display:flex; gap:8px; align-items:center; flex-wrap:wrap; margin-bottom:10px; padding:8px 10px; border:1px solid rgba(148,163,184,.25); border-radius:10px; background:rgba(2,6,23,.94); backdrop-filter:blur(6px); box-shadow:0 10px 30px rgba(0,0,0,.32); max-width:calc(100vw - 380px); }}
    .hover-active-value {{ color:#fde68a; background:rgba(113,63,18,.55); border:1px solid rgba(251,191,36,.55); border-radius:8px; padding:5px 7px; margin-bottom:4px; font-weight:800; }}
    .hover-context-value {{ color:#cbd5e1; font-size:.86rem; }}
    .map-explanation-card {{ margin-top:16px; line-height:1.45; }}
    .viewer-shell {{ display:grid; grid-template-columns:minmax(0,1fr) 300px; gap:14px; align-items:start; }}
    .legend-panel {{ position:sticky; top:78px; background:#020617; border:1px solid var(--line); border-radius:12px; padding:12px; max-height:82vh; overflow:auto; }}
    .legend-item {{ display:flex; align-items:center; gap:8px; margin:6px 0; font-size:.86rem; cursor:pointer; border-radius:8px; padding:3px 4px; }}
    .legend-item:hover {{ background:rgba(148,163,184,.12); }}
    .legend-item.legend-active {{ outline:1px solid rgba(96,165,250,.8); background:rgba(37,99,235,.18); }}
    .legend-item small {{ display:block; color:var(--muted); font-size:.75rem; margin-top:2px; }}
    .swatch {{ width:18px; height:14px; border:1px solid rgba(255,255,255,.35); border-radius:3px; display:inline-block; flex:0 0 auto; }}
    .gradient-swatch {{ width:72px; height:14px; border:1px solid rgba(255,255,255,.35); border-radius:3px; display:inline-block; flex:0 0 auto; }}
    .legend-stats {{ display:grid; grid-template-columns:1fr; gap:5px; margin-top:6px; }}
    .legend-stat {{ display:flex; justify-content:space-between; gap:12px; border-top:1px solid rgba(148,163,184,.18); padding-top:5px; font-size:.8rem; }}
    .legend-stat span {{ color:var(--muted); text-align:right; }}
    .side-stack {{ position:relative; display:inline-block; transform-origin:top left; }}
    .side-stack img.zoom-image {{ display:block; }}
    .viewer-wrap.dragging, .viewer-wrap:active {{ cursor:grabbing; }}
    .zoom-image {{ max-width:none; image-rendering:auto; transform-origin:top left; }}
    .compare-grid {{ display:grid; grid-template-columns:1fr 1fr; gap:12px; }}
    .compare-pane {{ min-width:0; }}
    .compare-pane .viewer-wrap {{ height:70vh; }}
    .compare-workspace {{ margin-top:18px; }}
    .compare-stage {{ position:relative; overflow:hidden; background:#020617; border:1px solid var(--line); border-radius:14px; height:80vh; cursor:crosshair; }}
    .compare-stage.dragging, .compare-stage:active {{ cursor:grabbing; }}
    .compare-layer {{ position:absolute; left:0; top:0; transform-origin:top left; max-width:none; user-select:none; pointer-events:none; }}
    .compare-overlay-layer {{ opacity:.5; }}
    .compare-clip {{ position:absolute; left:0; top:0; overflow:hidden; height:100%; width:50%; pointer-events:none; }}
    .compare-clip.horizontal {{ width:100%; height:50%; }}
    .compare-split-line {{ position:absolute; background:#f8fafc; box-shadow:0 0 0 1px #020617, 0 0 10px rgba(0,0,0,.6); pointer-events:none; }}
    .compare-split-line.vertical {{ top:0; bottom:0; width:2px; left:50%; }}
    .compare-split-line.horizontal {{ left:0; right:0; height:2px; top:50%; }}
    .map-grid-layer {{ position:absolute; left:0; top:0; width:100%; height:100%; transform-origin:top left; pointer-events:none; display:none; z-index:3; background-image:linear-gradient(rgba(248,250,252,.30) 1px, transparent 1px), linear-gradient(90deg, rgba(248,250,252,.30) 1px, transparent 1px); background-size:10% 10%; }}
    .demarcation .map-grid-layer {{ display:block; }}
    .contour-layer {{ position:absolute; left:0; top:0; transform-origin:top left; max-width:none; opacity:.78; pointer-events:none; image-rendering:auto; z-index:2; }}
    .side-contour {{ z-index:2; }}
    .hover-readout {{ position:sticky; left:0; bottom:0; margin-top:8px; display:block; background:rgba(2,6,23,.92); border:1px solid var(--line); border-radius:8px; padding:6px 8px; color:#e5e7eb; font-size:.84rem; z-index:4; min-height:32px; }}
    .scale-readout {{ display:block; margin-top:6px; color:#bae6fd; font-size:.84rem; }}
    .timing-bar {{ height:14px; min-width:120px; background:#020617; border:1px solid var(--line); border-radius:999px; overflow:hidden; }}
    .timing-bar span {{ display:block; height:100%; background:linear-gradient(90deg,#38bdf8,#22c55e); }}
    .mini-bar {{ height:8px; background:#0f172a; border:1px solid #334155; border-radius:999px; overflow:hidden; min-width:90px; }}
    .mini-bar span {{ display:block; height:100%; background:linear-gradient(90deg,#38bdf8,#22c55e); }}
    .stat-grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(170px,1fr)); gap:10px; }}
    .stat {{ background:#020617; border:1px solid var(--line); border-radius:10px; padding:10px; }}
    .stat b {{ display:block; color:#bfdbfe; font-size:.85rem; margin-bottom:4px; }}
    .stat-value {{ display:block; }}
    .stat-help {{ margin-top:6px; }}
    .stat-help summary {{ font-size:.78rem; color:#93c5fd; }}
    .stat-help p {{ margin:.35rem 0; color:#cbd5e1; font-size:.8rem; line-height:1.3; }}
    .planet-table tr.main-planet {{ background:rgba(34,197,94,.10); }}
    .candidate-grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(360px,1fr)); gap:14px; align-items:stretch; }}
    .candidate-card {{ background:#020617; border:1px solid var(--line); border-radius:14px; padding:14px; position:relative; overflow:hidden; }}
    .candidate-card.main-candidate {{ border-color:#22c55e; box-shadow:0 0 0 1px rgba(34,197,94,.55), 0 0 26px rgba(34,197,94,.16); background:radial-gradient(circle at 12% 0%,rgba(34,197,94,.18),transparent 32%), #020617; }}
    .candidate-card.eligible-candidate:not(.main-candidate) {{ border-color:#64748b; }}
    .candidate-card.rejected-candidate {{ opacity:.88; }}
    .candidate-head {{ display:flex; justify-content:space-between; gap:12px; align-items:flex-start; margin-bottom:10px; }}
    .candidate-title {{ display:flex; gap:8px; align-items:center; min-width:0; }}
    .candidate-title strong {{ font-size:1.15rem; color:#e0f2fe; }}
    .candidate-title small {{ display:block; color:#94a3b8; margin-top:2px; }}
    .candidate-score {{ min-width:82px; text-align:right; }}
    .candidate-score b {{ display:block; color:#bfdbfe; font-size:1.05rem; }}
    .candidate-tags {{ display:flex; flex-wrap:wrap; gap:6px; margin:8px 0; }}
    .candidate-tags .pill {{ border:1px solid rgba(148,163,184,.25); }}
    .candidate-viz {{ display:grid; grid-template-columns:1fr; gap:8px; margin-top:10px; }}
    .candidate-metrics {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(130px,1fr)); gap:8px; }}
    .candidate-reason {{ margin-top:10px; padding:8px 10px; background:rgba(15,23,42,.75); border:1px solid rgba(148,163,184,.18); border-radius:10px; }}
    .candidate-card details {{ margin-top:9px; }}
    .physics-hero {{ display:grid; grid-template-columns:1.1fr .9fr; gap:18px; align-items:start; }}
    .physics-card {{ background:#020617; border:1px solid var(--line); border-radius:14px; padding:14px; }}
    .physics-card h3 {{ margin-bottom:6px; color:#bfdbfe; }}
    .implication-list li {{ margin-bottom:6px; }}
    @media (max-width:1000px) {{ .physics-hero {{ grid-template-columns:1fr; }} }}
    .hover-help {{ cursor:help; text-decoration:underline dotted rgba(191,219,254,.75); text-underline-offset:3px; }}
    .slider-field {{ background:#020617; border:1px solid var(--line); border-radius:10px; padding:10px; }}
    .slider-row {{ flex-wrap:nowrap; }}
    .slider-row input.range {{ flex:1 1 160px; min-width:120px; padding:0; }}
    .slider-row input.number {{ width:92px; }}
    .resolution-row {{ display:grid; grid-template-columns:1fr auto auto; gap:6px; align-items:end; }}
    .tiny-btn {{ padding:7px 9px; margin-top:0; min-width:44px; }}
    .preset-pills button {{ margin-top:6px; }}
    .viewer-map-stage {{ position:relative; display:inline-block; }}
    .full-compare {{ position:fixed !important; inset:0 !important; z-index:10; background:#020617; padding:12px; overflow:auto; }}
    .full-compare .compare-stage, .full-compare .viewer-wrap {{ height:calc(100vh - 155px); }}
    .full-exit {{ display:none; }}
    .full-compare .full-exit {{ display:inline-block; }}
    .sticky-tools {{ position:sticky; top:62px; z-index:50; background:rgba(15,23,42,.96); border:1px solid var(--line); border-radius:12px; padding:10px; margin-bottom:12px; }}

    code {{ overflow-wrap:anywhere; word-break:break-all; }}
    .stat, .orbit-card, .mapcard, .job-card {{ min-width:0; overflow-wrap:anywhere; }}
    .class-pill {{ display:inline-block; padding:3px 8px; border-radius:999px; background:#1e293b; border:1px solid var(--line); color:#bfdbfe; font-size:.8rem; }}
    .visual-metric {{ background:#020617; border:1px solid var(--line); border-radius:12px; padding:10px; min-width:0; }}
    .metric-head {{ display:flex; justify-content:space-between; gap:10px; align-items:baseline; }}
    .metric-head span {{ color:#e0f2fe; font-weight:700; }}
    .metric-track {{ height:10px; border-radius:999px; background:#0f172a; border:1px solid #1e293b; overflow:hidden; margin:7px 0 4px; }}
    .metric-track span {{ display:block; height:100%; background:linear-gradient(90deg,#38bdf8,#22c55e,#facc15); }}
    .hz-strip {{ position:relative; height:24px; background:linear-gradient(90deg,#7f1d1d,#1e293b,#0f5132,#1e293b,#1e3a8a); border:1px solid var(--line); border-radius:999px; overflow:hidden; margin-top:8px; }}
    .hz-band {{ position:absolute; top:0; bottom:0; background:rgba(34,197,94,.42); border-left:1px solid #86efac; border-right:1px solid #86efac; }}
    .hz-marker {{ position:absolute; top:-3px; bottom:-3px; width:3px; background:#f8fafc; box-shadow:0 0 8px #fff; }}
    .planet-mix {{ display:flex; flex-wrap:wrap; gap:8px; }}
    .mix-chip {{ display:inline-flex; align-items:center; gap:5px; padding:6px 9px; border-radius:999px; border:1px solid var(--line); background:linear-gradient(90deg, color-mix(in srgb, var(--planet-color) 30%, #020617), #020617); color:#e5e7eb; }}
    .moon-card {{ display:flex; align-items:center; gap:12px; background:#020617; border:1px solid var(--line); border-radius:12px; padding:10px; }}
    .moon-icon {{ width:46px; height:46px; border-radius:50%; background:radial-gradient(circle at 35% 30%,#f8fafc,#94a3b8 50%,#020617 70%); display:flex; align-items:center; justify-content:center; color:#020617; font-weight:900; }}
    .visual-editor {{ border:1px solid #1e3a8a; background:linear-gradient(180deg,rgba(15,23,42,.96),rgba(2,6,23,.72)); }}
    .stage-edit-card {{ margin-top:18px; border-color:#1e3a8a; background:linear-gradient(180deg,rgba(15,23,42,.98),rgba(2,6,23,.72)); }}
    .stage-edit-card h2 {{ margin-bottom:4px; }}
    .visual-editor-field {{ background:#020617; border:1px solid var(--line); border-radius:12px; padding:10px; }}
    .value-ruler {{ display:flex; justify-content:space-between; color:#64748b; font-size:.72rem; }}
    .live-value {{ min-width:52px; color:#bfdbfe; font-weight:700; text-align:right; }}
    .globe-wrap {{ display:grid; grid-template-columns:minmax(520px,1fr) minmax(300px,360px); gap:12px; align-items:start; }}
    .globe-stage.card {{ padding:6px; }}
    .globe-stage {{ background:#020617; border:1px solid var(--line); border-radius:14px; height:calc(100vh - 156px); min-height:560px; display:flex; align-items:stretch; justify-content:stretch; overflow:hidden; position:relative; }}
    .globe-stage canvas {{ width:100%; height:100%; display:block; cursor:crosshair; }}

    .pipeline-nav {{ position:sticky; top:58px; z-index:1; display:flex; gap:8px; flex-wrap:wrap; align-items:center; background:rgba(2,6,23,.92); border:1px solid var(--line); border-radius:14px; padding:10px; margin:12px 0 18px; backdrop-filter:blur(8px); }}
    .pipeline-nav a {{ text-decoration:none; color:var(--text); border:1px solid var(--line); background:#020617; border-radius:999px; padding:6px 9px; font-size:.82rem; display:inline-flex; gap:6px; align-items:center; }}
    .pipeline-nav a.current {{ border-color:#38bdf8; color:#bae6fd; box-shadow:0 0 0 1px rgba(56,189,248,.22); }}
    .pipeline-nav .nav-dot {{ width:8px; height:8px; border-radius:50%; background:#64748b; display:inline-block; }}
    .pipeline-nav .complete .nav-dot {{ background:#4ade80; }}
    .pipeline-nav .stale .nav-dot {{ background:#facc15; }}
    .pipeline-nav .missing .nav-dot {{ background:#64748b; }}
    .pipeline-nav .accepted-mark {{ color:#86efac; font-size:.75rem; }}
    .pipeline-grid {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(280px,1fr)); gap:14px; margin-top:18px; }}
    .pipeline-stage-card {{ background:#020617; border:1px solid var(--line); border-radius:14px; padding:14px; min-width:0; }}
    .pipeline-stage-card.complete {{ border-color:rgba(74,222,128,.45); }}
    .pipeline-stage-card.stale {{ border-color:rgba(250,204,21,.55); }}
    .pipeline-stage-card.missing {{ opacity:.86; }}
    .card-action-row {{ margin-top:10px; gap:7px; align-items:center; }}
    .card-action-row .button, .card-action-row button {{ margin-top:0; }}
    .card-action-form {{ display:inline; margin:0; }}
    button.small, .button.small {{ padding:6px 9px; font-size:.82rem; border-radius:8px; }}
    .stage-tools-card {{ margin-top:18px; background:rgba(15,23,42,.92); border:1px solid var(--line); border-radius:14px; padding:14px; }}
    .stage-tools-card > summary {{ font-size:1.02rem; }}
    .advanced-tools {{ margin-top:12px; border-top:1px solid rgba(148,163,184,.18); padding-top:10px; }}
    .active-job-banner {{ border-color:rgba(250,204,21,.55); background:rgba(113,63,18,.20); }}
    .provenance-drawer {{ margin-top:18px; }}
    .pipeline-stage-head {{ display:flex; justify-content:space-between; gap:10px; align-items:flex-start; }}
    .pipeline-stage-head h2, .pipeline-stage-head h3 {{ margin-bottom:6px; }}
    .merged-dashboard {{ margin-top:18px; background:rgba(15,23,42,.92); border:1px solid var(--line); border-radius:14px; padding:14px; }}
    .merged-dashboard > summary {{ font-size:1.05rem; }}
    .merged-dashboard[open] {{ box-shadow:0 10px 30px rgba(0,0,0,.15); }}
    .ok-pill {{ background:rgba(34,197,94,.18); color:#bbf7d0; border:1px solid rgba(34,197,94,.45); }}
    .warn-pill {{ background:rgba(250,204,21,.14); color:#fde68a; border:1px solid rgba(250,204,21,.45); }}
    .bad-pill {{ background:rgba(248,113,113,.14); color:#fecaca; border:1px solid rgba(248,113,113,.45); }}
    .source-chip {{ display:inline-block; border-radius:999px; padding:2px 8px; font-size:.78rem; border:1px solid var(--line); background:#1e293b; color:#cbd5e1; }}
    .source-chip.manual {{ color:#fed7aa; border-color:#f97316; background:rgba(249,115,22,.12); }}
    .source-chip.locked {{ color:#ddd6fe; border-color:#8b5cf6; background:rgba(139,92,246,.14); }}
    .source-chip.generated, .source-chip.derived {{ color:#bfdbfe; border-color:#38bdf8; background:rgba(56,189,248,.10); }}
    .stat-help, .possible-values {{ margin-top:8px; border-top:1px solid rgba(148,163,184,.18); padding-top:6px; }}
    .possible-values ul {{ margin:.4rem 0 0; padding-left:1.1rem; }}
    .possible-values li {{ margin:.35rem 0; }}
    .stage-actions-card form.inline-run-to {{ margin-top:12px; }}
    .globe-stage canvas.dragging {{ cursor:grabbing; }}
    .globe-details {{ max-height:calc(100vh - 156px); overflow:auto; }}
    .globe-hover-card {{ background:#020617; border:1px solid var(--line); border-radius:10px; overflow:hidden; }}
    .globe-hover-head {{ padding:8px 10px; background:linear-gradient(90deg,rgba(56,189,248,.18),rgba(34,197,94,.12)); border-bottom:1px solid var(--line); }}
    .globe-hover-head b {{ display:block; color:#e0f2fe; }}
    .globe-hover-head span {{ color:#94a3b8; font-size:.78rem; }}
    .globe-hover-table {{ width:100%; border-collapse:collapse; font-size:.82rem; }}
    .globe-hover-table th, .globe-hover-table td {{ padding:5px 8px; border-bottom:1px solid rgba(30,41,59,.8); vertical-align:top; }}
    .globe-hover-table th {{ width:42%; color:#93c5fd; font-weight:700; text-align:left; }}
    .globe-hover-table tr.focus-row th, .globe-hover-table tr.focus-row td {{ background:rgba(250,204,21,.10); color:#fde68a; }}
    .globe-control-grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:8px; align-items:center; }}
    .globe-action-row {{ display:flex; flex-wrap:wrap; gap:8px; margin-top:10px; }}
    .planet-dashboard {{ display:grid; grid-template-columns:1.1fr .9fr; gap:18px; }}
    .planet-hero {{ display:flex; align-items:center; gap:20px; background:radial-gradient(circle at 20% 20%,rgba(34,197,94,.22),transparent 36%), #020617; border:1px solid var(--line); border-radius:14px; padding:18px; }}
    .planet-orb {{ width:142px; height:142px; flex:0 0 auto; border-radius:50%; background:radial-gradient(circle at 32% 26%,#e0f2fe 0%,#22c55e 18%,#166534 34%,#1d4ed8 56%,#0f172a 82%); box-shadow:inset -24px -18px 34px rgba(2,6,23,.72), 0 0 28px rgba(34,197,94,.24); position:relative; }}
    .planet-orb::after {{ content:''; position:absolute; inset:10%; border-radius:50%; border:1px solid rgba(255,255,255,.2); transform:rotate(var(--tilt, 0deg)); }}
    .tilt-widget {{ height:110px; border:1px solid var(--line); border-radius:12px; background:#020617; display:flex; align-items:center; justify-content:center; position:relative; overflow:hidden; }}
    .tilt-axis {{ height:84px; width:3px; background:#38bdf8; transform:rotate(var(--tilt, 0deg)); box-shadow:0 0 12px #38bdf8; }}
    .orbit-mini {{ height:92px; border:1px solid var(--line); border-radius:12px; background:linear-gradient(90deg,#111827,#052e16,#111827); position:relative; overflow:hidden; }}
    .orbit-mini .hz {{ position:absolute; top:0; bottom:0; background:rgba(34,197,94,.25); border-left:1px solid #86efac; border-right:1px solid #86efac; }}
    .orbit-mini .planet-marker {{ position:absolute; top:50%; width:14px; height:14px; border-radius:50%; background:#38bdf8; transform:translate(-50%,-50%); box-shadow:0 0 12px #38bdf8; }}
    @media (max-width:1000px) {{ .planet-dashboard {{ grid-template-columns:1fr; }} }}
    @media (max-width:900px) {{ .map-browser-controls {{ grid-template-columns:1fr; }} }}
    @media (max-width:1000px) {{ .globe-wrap {{ grid-template-columns:1fr; }} }}
    .small {{ font-size:.88rem; }}
    .status-table form {{ margin:0; }}
    @media (max-width:1100px) {{ .compare-grid, .viewer-shell {{ grid-template-columns:1fr; }} .legend-panel {{ position:static; max-height:none; }} }}
  </style>
</head>
<body>
<header>
  <strong>WorldGen Local UI</strong>
  <a href="/">Home</a>
  <a href="/jobs">Jobs</a>
</header>
<main>{body}</main>
</body>
</html>"""

    def _page_pipeline(self, params: dict[str, list[str]]) -> str:
        output_dir = Path(_first(params, "output_dir")).expanduser()
        exists = output_dir.exists()
        message_value = _first(params, "pipeline_message")
        message = f"<p class='ok'>{_safe_text(message_value)}</p>" if message_value else ""
        if exists:
            ensure_layout(output_dir)
            try:
                write_status_report(output_dir)
            except Exception:
                pass
        else:
            body = f"""
<h1>Pipeline workspace</h1>
<p><code>{_safe_text(output_dir)}</code></p>
<section class='card'><p class='bad'>This run folder does not exist yet.</p><a class='button secondary' href='/'>Back home</a></section>
"""
            return self._page_shell("Pipeline", body)
        warnings = _incongruence_warnings(output_dir)
        warning_html = _stage_warnings_html(warnings)
        run_tools_html = self._page_run({"output_dir": [str(output_dir)], "embedded": ["1"]})
        body = f"""
<h1>Pipeline workspace</h1>
<p><code>{_safe_text(output_dir)}</code></p>
{message}
{_pipeline_active_job_banner(output_dir)}
{_pipeline_stage_nav_html(output_dir)}
<section class='card pipeline-intro'>
  <h2>Review workflow</h2>
  <p class='help'>This is the main run workspace. Use the stage cards for normal review. Each stage has one Review stage page with details, edit controls, preconditions, warnings, provenance, and stage-specific tools. Running many or all stages without review is still supported in the run-through controls below.</p>
  <details><summary>Current cross-stage warnings</summary>{warning_html}</details>
</section>
{_pipeline_stage_cards_html(output_dir)}
<section class='card' style='margin-top:18px'>
  <h2>Map index</h2>
  <p class='help'>Open the registry-backed map index to see every known map family, whether it was generated for this run, and why missing items are missing.</p>
  {_map_index_summary_html(output_dir)}
  <div class='row'><a class='button secondary' href='/map-index?output_dir={urllib.parse.quote(str(output_dir))}'>Open map index</a></div>
</section>
<details class='merged-dashboard'>
  <summary><strong>Run-through controls, maps, diagnostics, and generated files</strong> <span class='muted'>collapsed to keep Pipeline focused</span></summary>
  {run_tools_html}
</details>
<details class='merged-dashboard provenance-drawer'>
  <summary><strong>Manual/locked value provenance</strong> <span class='muted'>generated, derived, manual, locked, and raw-override sources</span></summary>
  <p class='help'>Values listed here are not plain generated values. They came from overrides, locks, presets, or raw JSON edits. If they conflict with derived values, warning panels will suggest adjusting preconditions instead of forcing the output.</p>
  {_stage_override_summary_html(output_dir)}
</details>
{_pipeline_advanced_tools_html(output_dir)}
"""
        return self._page_shell("Pipeline", body)

    def _page_stage_workspace(self, params: dict[str, list[str]]) -> str:
        output_dir = Path(_first(params, "output_dir")).expanduser()
        stage = normalize_stage(_first(params, "stage", "solar-system"))
        exists = output_dir.exists()
        message_value = _first(params, "stage_message")
        message = f"<p class='ok'>{_safe_text(message_value)}</p>" if message_value else ""
        if not exists:
            return self._page_shell("Stage workspace", f"<h1>{_safe_text(_stage_label(stage))}</h1><section class='card'><p class='bad'>Run folder does not exist: <code>{_safe_text(output_dir)}</code></p></section>")
        ensure_layout(output_dir)
        try:
            write_status_report(output_dir)
        except Exception:
            pass
        status_row = _stage_status_map(output_dir).get(stage, {})
        status = str(status_row.get("status", "missing"))
        accepted = _stage_acceptance_record(output_dir, stage)
        accepted_html = f"<p class='ok'>Accepted at {_safe_text(accepted.get('accepted_at',''))}. {_safe_text(accepted.get('note',''))}</p>" if accepted else "<p class='muted'>This stage has not been explicitly accepted yet.</p>"
        warnings = _stage_warning_items(output_dir, stage)
        output_summary = STAGE_OUTPUT_SUMMARIES.get(stage, "Stage output summary is not cataloged yet.")
        body = f"""
<h1>{_safe_text(_stage_label(stage))} review</h1>
<p><code>{_safe_text(output_dir)}</code></p>
{message}
<div class='row' style='margin-bottom:14px'>
  <a class='button secondary' href='/pipeline?output_dir={urllib.parse.quote(str(output_dir))}'>Back to Pipeline</a>
</div>
{_pipeline_stage_nav_html(output_dir, current_stage=stage)}
<section class='card stage-overview-card'>
  <div class='pipeline-stage-head'><h2>Stage status</h2><span class='{_safe_text(status)}'>{_safe_text(status)}</span></div>
  <p>{_safe_text(output_summary)}</p>
  <div class='row'>{_warning_badge(warnings)} {'<span class="pill ok-pill">accepted</span>' if accepted else '<span class="pill">not accepted</span>'}</div>
  {accepted_html}
</section>
{_stage_action_panel(output_dir, stage)}
{_terrain_review_controls_html(output_dir, stage)}
{_stage_edit_values_html(output_dir, stage)}
{_stage_support_tools_html(output_dir, stage)}
<section class='card stage-map-panel' style='margin-top:18px'>
  <h2>Stage maps and images</h2>
  <p class='help'>This stage-local view shows the generated images most relevant to the current stage first. The full recursive inventory remains in Run tools.</p>
  <div class='row'><a class='button secondary small' href='/map-index?output_dir={urllib.parse.quote(str(output_dir))}&stage={urllib.parse.quote(stage)}'>Stage map index</a></div>
  {_map_inventory_html(output_dir, stage=stage, full_inventory=False)}
</section>
{_terrain_single_subphase_html(output_dir, stage) if stage.startswith('terrain') else ''}
<div class='grid' style='margin-top:18px'>
  <section class='card'>
    <h2>Preconditions</h2>
    <p class='help'>These are the upstream assumptions this stage depends on. Editing a related upstream value should rerun or mark this stage stale. For derived quantities, prefer adjusting preconditions when you want a physically consistent target value.</p>
    {_stage_dependencies_html(output_dir, stage)}
  </section>
  <section class='card'>
    <h2>Warnings and congruence checks</h2>
    <p class='help'>Warnings do not usually block generation. They explain where manual values, locks, or derived values may be inconsistent.</p>
    {_stage_warnings_html(warnings)}
  </section>
</div>
<details class='merged-dashboard provenance-drawer'>
  <summary><strong>Manual values and source tracking</strong> <span class='muted'>closed by default</span></summary>
  <p class='help'>Manual/locked values are recorded separately from generated/derived values. Older overrides without metadata are still treated as manual for review purposes.</p>
  {_stage_override_summary_html(output_dir)}
</details>"""
        return self._page_shell("Stage workspace", body)

    def _page_home(self, params: dict[str, list[str]]) -> str:
        default_output_dir = _unique_output_dir_name("staged_web_run")
        recent = load_recent_runs()
        run_rows = _run_management_rows(recent)
        run_message_value = _first(params, "run_message")
        run_message = f"<p class='ok'>{_safe_text(run_message_value)}</p>" if run_message_value else ""
        recent_rows = ""
        for row in run_rows:
            path = row["path"]
            path_q = urllib.parse.quote(path)
            exists = bool(row.get("exists"))
            status = "Exists" if exists else "Missing"
            status_class = "complete" if exists else "bad"
            updated = row.get("updated_at") or ("—" if exists else "Folder removed")
            counts = row.get("counts", {}) if isinstance(row.get("counts"), dict) else {}
            stage_summary = "—"
            if exists:
                if row.get("first_incomplete"):
                    stage_summary = f"Next: {row.get('first_incomplete')}"
                elif row.get("last_complete"):
                    stage_summary = f"Complete through: {row.get('last_complete')}"
            summary_bits = []
            if exists and row.get("pipeline_version"):
                summary_bits.append(f"pipeline {row.get('pipeline_version')}")
            if exists and counts:
                summary_bits.append(f"{counts.get('complete',0)} complete / {counts.get('stale',0)} stale / {counts.get('missing',0)} missing")
            summary = " • ".join(summary_bits) if summary_bits else "—"
            open_link = f"<a class='button secondary' href='/pipeline?output_dir={path_q}'>Pipeline</a>" if exists else ""
            extras = ""
            forget_form = (
                f"<form method='post' action='/run-admin' style='display:inline'>"
                f"<input type='hidden' name='action' value='forget-run'>"
                f"<input type='hidden' name='output_dir' value='{_safe_text(path)}'>"
                f"<input type='hidden' name='redirect_to' value='/'>"
                f"<button type='submit' class='secondary small'>Forget</button></form>"
            )
            delete_form = ""
            if exists:
                delete_form = (
                    f"<form method='post' action='/run-admin' style='display:inline' onsubmit='return confirm(\"Delete this run folder and all files? This cannot be undone.\");'>"
                    f"<input type='hidden' name='action' value='delete-run'>"
                    f"<input type='hidden' name='output_dir' value='{_safe_text(path)}'>"
                    f"<input type='hidden' name='redirect_to' value='/'>"
                    f"<button type='submit' class='danger small'>Delete folder</button></form>"
                )
            recent_rows += f"<tr><td><code>{_safe_text(path)}</code></td><td class='{status_class}'>{_safe_text(status)}</td><td>{_safe_text(updated)}</td><td>{_safe_text(stage_summary)}<div class='muted small'>{_safe_text(summary)}</div></td><td>{open_link} {extras} {forget_form} {delete_form}</td></tr>"
        recent_rows = recent_rows or "<tr><td colspan='5' class='muted'>No recent runs yet.</td></tr>"
        body = f"""
<h1>WorldGen Local Web UI</h1>
{run_message}
<div class="grid">
  <section class="card">
    <h2>Create run</h2>
    <p class="help">Choose the run mode, resolution, and pipeline target. Hover or read the notes under each section for what the setting changes.</p>
    <form method="post" action="/new-run" id="new-run-form">
      <label title="Generated mode is the normal procedural/staged WorldGen pipeline. Earth modes are calibration presets that use the full-run engine.">Run mode / preset</label>
      <select name="preset" id="preset-select" onchange="wgPresetChanged()">
        <option value="generated" selected>Generated procedural world — staged pipeline</option>
        <option value="synthetic-earth">Synthetic Earth calibration — full run</option>
        <option value="real-earth-terrain">Real Earth terrain calibration — full run</option>
      </select>
      <p class="muted small" id="preset-help">Generated mode uses the staged pipeline. Earth calibration modes use real-world terrain data through Stage 3, then run downstream simulation.</p>
      <label title="Terrain backend for generated staged worlds. Earth modes ignore this because they use real-world Stage 3 terrain data.">Terrain generation mode</label>
      <select name="terrain_mode" id="terrain-mode-select">
        <option value="plate_history_v4" selected>Plate history v4 — recommended conservative terrain model</option>
        <option value="plate_history_v3">Plate history v3 — stable fallback</option>
        <option value="procedural_legacy">Procedural legacy — archived baseline</option>
        <option value="plate_tectonic_v1">Plate tectonic v1 — plate-owned terrain</option>
        <option value="plate_history_v1">Plate history v1 — time-evolved plates</option>
        <option value="plate_history_v2">Plate history v2 — legacy experimental</option>
      </select>
      <p class="muted small">Plate history v4 is now the recommended conservative terrain model for new generated worlds. It starts from stable v3, keeps v3 as fallback, and adds bounded topology, rift, and volcanic-island-chain behavior. Older terrain modes remain available for comparison/rollback.</p>
      <div class="grid compact-grid">
        <div><label title="For plate_history_v1/v2/v3/v4 only. Leave blank for a randomized mature history. Examples: 250, 750, 1500.">Tectonic history Myr</label><input name="tectonic_history_myr" placeholder="auto"><div class="field-help">Millions of years to simulate before final terrain is derived.</div></div>
        <div><label title="For plate_history_v1/v2/v3/v4 only. The engine may internally sample fewer epochs for speed while preserving total history length.">Tectonic timestep Myr</label><input name="tectonic_timestep_myr" placeholder="2.5"><div class="field-help">Requested geological timestep; 2–5 Myr is a good start.</div></div>
        <div><label title="For plate_history_v1/v2/v3/v4 only. This is a requested tectonic-history detail label; normal runs use the stable macro-history grid plus full-resolution detail.">Tectonic grid scale</label><select name="tectonic_grid_scale"><option value="legacy" selected>Legacy — Update 17 default</option><option value="preview">Preview — requested / 8</option><option value="normal">Normal — structural detail</option><option value="high">High — structural detail / 2</option><option value="native">Native — requested map resolution</option><option value="custom">Custom detail</option></select><div class="field-help">Raw/high history grids currently produce worse terrain, so the stable hybrid resolution is locked for normal runs.</div></div>
        <div><label title="Stable hybrid is locked for normal runs because raw/high history grids currently produce poorer terrain.">Tectonic history resolution policy</label><div class="readonly-pill">Stable hybrid — locked</div><div class="field-help">WorldGen uses the proven macro plate-history grid, then applies full-resolution tectonic detail. Higher-resolution raw history grids are deferred terrain research.</div></div>
        <div><label title="Only used when Tectonic grid scale is Custom.">Custom tectonic grid width</label><input name="tectonic_grid_width" placeholder="optional"><div class="field-help">Documents requested detail for future experiments. Normal runs still use stable hybrid history resolution.</div></div>
        <div><label title="Only used when Tectonic grid scale is Custom.">Custom tectonic grid height</label><input name="tectonic_grid_height" placeholder="optional"><div class="field-help">Example: 512. Leave blank to preserve map aspect ratio.</div></div>
      </div>
      <section class="resolution-panel v3-terrain-control-panel">
        <h3>plate_history_v3/v4 terrain tuning</h3>
        <p class="help">These controls are passed into the new run command and saved into <code>config/stage_overrides.json</code>. They affect <code>plate_history_v3</code> and <code>plate_history_v4</code>; older terrain modes are preserved.</p>
        <div class="grid compact-grid">
          <div><label title="Multiplier for submerged continental shelf support. Higher values should make passive continental margins wider and shallower without giving broad shelves to unsupported volcanic islands.">Continental shelf strength</label><input name="continental_shelf_strength" value="1.65"><div class="field-help">Default 1.65. Try 2.0–2.4 if shelves are still too narrow.</div></div>
          <div><label title="Master multiplier for erosion, old-mountain wearing, sediment fill, basin fill, and passive-margin deposition.">Erosion/deposition multiplier</label><input name="erosion_deposition_multiplier" value="1.35"><div class="field-help">Default 1.35. Try 1.6–2.0 for stronger erosion/deposition.</div></div>
          <div><label title="Broadness factor for the submerged continental apron and shelf-to-slope transition. This changes shelf width more than shelf shallowness.">Shelf width factor</label><input name="shelf_width_factor" value="0.90"><div class="field-help">Default 0.90. Try 1.2–1.6 for broader continental aprons.</div></div>
          <div><label title="plate_history_v4 only. Controls boundary deformation, sliver/microplate emphasis, and mountain-branching support without changing the stable v3 baseline.">v4 topology strength</label><input name="v4_topology_strength" value="1.00"><div class="field-help">Default 1.0. Try 1.2–1.6 for stronger topology.</div></div>
          <div><label title="plate_history_v4 only. Controls how strongly supported volcanic island chains are raised.">v4 island strength</label><input name="v4_island_strength" value="1.00"><div class="field-help">Default 1.0. Try 1.2–1.6 for more volcanic chains; use higher values only as stress tests.</div></div>
          <div><label title="plate_history_v4 only. Controls rift-cut/gulf/narrow-sea lowering in supported extensional corridors.">v4 rift strength</label><input name="v4_rift_strength" value="1.00"><div class="field-help">Default 1.0. Try 1.2–1.6 if rifts/gulfs are too weak.</div></div>
        </div>
      </section>
      <label title="In plate_history_v1/v2/v3/v4 this smoothly lowers polar continental-crust potential during land formation instead of erasing land at a hard latitude cutoff."><input type="checkbox" name="suppress_polar_land" style="width:auto"> suppress polar land</label>
      <p class="muted small">Use this when polar projection distortion is making terrain review hard or when testing non-polar continents.</p>
      <label title="The UI chooses a free default folder name so new runs do not overwrite existing runs.">Output directory</label><input name="output_dir" value="{_safe_text(default_output_dir)}">
      <div class="field-help">Automatically uses <code>staged_web_run_2</code>, <code>staged_web_run_3</code>, and so on if the base name already exists.</div>
      <label title="Optional JSON config file. Leave blank to use form values and defaults.">Config file, optional</label><input name="config" placeholder="examples/basic_config.json">
      <div class="field-help">Use this only when you want to start from a saved configuration file.</div>
      <div class="grid compact-grid">
        <div><label title="Base random seed. Reusing the same seed and settings should reproduce the same staged outputs.">Seed</label><input name="seed" value="143"><div class="field-help">Use this to reproduce or compare worlds.</div></div>
        <div><label title="Optional override for the number of generated planets in generated mode.">Planet count</label><input name="planet_count" placeholder="optional"><div class="field-help">Leave blank for the model default.</div></div>
        <div><label title="Climate backend. seasonal_v5 adds component-based moisture/rainfall coupling on top of seasonal_v4; seasonal_v4 is the refined structured atmosphere + basin-ocean mode; seasonal_v3 preserves the first basin-ocean mode; seasonal_v2 preserves the structured atmosphere-only mode; seasonal_v1 is the stable seasonal overhaul; legacy preserves the previous heuristic model for comparison.">Climate mode</label><select name="climate_mode"><option value="seasonal_v5">seasonal_v5 — component moisture + refined basin ocean</option><option value="seasonal_v4">seasonal_v4 — refined atmosphere + basin ocean</option><option value="seasonal_v3">seasonal_v3 — first basin ocean</option><option value="seasonal_v2">seasonal_v2 — structured atmosphere only</option><option value="seasonal_v1" selected>seasonal_v1 — stable seasonal overhaul</option><option value="legacy">legacy — previous heuristic model</option></select><div class="field-help">Use seasonal_v5 for component moisture/rainfall testing, seasonal_v4 for the refined basin-ocean circulation test, seasonal_v3 for first ocean comparison, seasonal_v2 for atmosphere-only comparison, seasonal_v1 for stable comparison, or legacy only for rollback/comparison.</div></div>
        <div><label title="How detailed the Köppen climate classification should be.">Köppen detail</label><select name="koppen_detail"><option value=""></option><option>regional</option><option>local9</option><option selected>local4</option><option>cell</option></select><div class="field-help">local4 is a good diagnostic default; cell is slower and noisier.</div></div>
      </div>
      <section class="resolution-panel solar-control-panel">
        <h3>Solar-system controls <span class="muted small">generated mode only</span></h3>
        <p class="help">These controls shape Stage 1 before terrain and climate exist: star subtype, orbital architecture, Main Planet preference, and moon/tide strength.</p>
        <div class="grid compact-grid">
          <div><label title="Choose a broad class or subtype. Leave random when you want the generator to pick a G/K main-sequence star.">Star class</label><select name="star_class"><option value="">Random G/K</option><option>G</option><option>K</option><option>G0V</option><option>G2V</option><option>G5V</option><option>K0V</option><option>K3V</option><option>K5V</option><option>K7V</option></select><div class="field-help">Subtypes improve flavor and derived star display.</div></div>
          <div><label title="Optional exact stellar mass in solar masses.">Star mass M☉</label><input name="star_mass" placeholder="optional"><div class="field-help">Leave blank for class-aware generation.</div></div>
          <div><label title="Optional stellar age in billions of years.">Star age Gyr</label><input name="star_age" placeholder="optional"><div class="field-help">Older systems bias quieter/sparser architecture.</div></div>
          <div><label title="Optional rough metallicity relative to solar.">Metallicity</label><input name="metallicity" placeholder="optional"><div class="field-help">Higher values bias giant/volatile-rich systems.</div></div>
          <div><label title="Overall orbital layout pattern.">Architecture</label><select name="system_architecture"><option value="random" selected>Random / weighted</option><option value="compact_rocky_inner">Compact rocky inner</option><option value="solar_like_mixed">Solar-like mixed</option><option value="outer_giant_dominated">Outer giant dominated</option><option value="low_mass_quiet">Low-mass quiet</option><option value="volatile_rich">Volatile-rich</option><option value="sparse_old">Sparse old</option></select><div class="field-help">Affects planet spacing, giants, and volatile delivery context.</div></div>
          <div><label title="What kind of Main Planet should be favored when selecting the deep-dive world.">Main Planet preference</label><select name="main_planet_preference"><option value="earthlike" selected>Earth-like</option><option value="dry_terrestrial">Dry terrestrial</option><option value="oceanic">Oceanic</option><option value="super_earth">Super-Earth</option><option value="colder_world">Colder world</option><option value="warmer_world">Warmer world</option></select><div class="field-help">Changes candidate scoring and HZ anchor bias.</div></div>
          <div><label title="Preferred strength of the single major moon's tidal influence.">Moon strength</label><select name="moon_strength"><option value="weak">Weak tides</option><option value="moderate" selected>Moderate / Earth-like</option><option value="strong">Strong tides</option></select><div class="field-help">Affects moon mass/orbit and tidal metadata.</div></div>
          <div><label>&nbsp;</label><label title="Disable the guaranteed major moon in generated systems."><input type="checkbox" name="no_major_moon" style="width:auto"> no major moon</label><div class="field-help">Useful for testing axial stability and tide assumptions.</div></div>
        </div>
        <h4>System presets</h4>
        <div class="row preset-pills">
          <button type="button" class="secondary" onclick="wgSystemPreset('earth')">Earth-like solar analog</button>
          <button type="button" class="secondary" onclick="wgSystemPreset('kstar')">K-star habitable world</button>
          <button type="button" class="secondary" onclick="wgSystemPreset('ocean')">Ocean-world candidate</button>
          <button type="button" class="secondary" onclick="wgSystemPreset('dry')">Dry terrestrial candidate</button>
          <button type="button" class="secondary" onclick="wgSystemPreset('super')">Super-Earth candidate</button>
          <button type="button" class="secondary" onclick="wgSystemPreset('cold')">Cold HZ edge world</button>
          <button type="button" class="secondary" onclick="wgSystemPreset('warm')">Warm inner-HZ world</button>
        </div>
      </section>
      <section class="resolution-panel">
        <h3>Resolution</h3>
        <p class="help">Width, height, and image max width usually move together. World maps are equirectangular, so height is normally half the width.</p>
        <div class="resolution-inputs">
          <div><label title="Number of simulation columns in the full-world raster.">Map width</label><input id="map_width" name="map_width" value="2048" oninput="wgSyncResolution('width')"><div class="field-help">Simulation cells east-west.</div></div>
          <div><label title="Number of simulation rows. Normally half of map width.">Map height</label><input id="map_height" name="map_height" value="1024" oninput="wgSyncResolution('height')"><div class="field-help">Simulation cells north-south.</div></div>
          <div><label title="Maximum rendered image width. Keep synced for full-detail map exports.">Image max width</label><input id="image_max_width" name="image_max_width" value="2048"><div class="field-help">Display/export cap for generated PNGs.</div></div>
        </div>
        <div class="resolution-buttons">
          <button type="button" class="secondary tiny-btn" onclick="wgScaleResolution(0.5)">½</button>
          <button type="button" class="secondary tiny-btn" onclick="wgScaleResolution(2)">×2</button>
          <button type="button" class="secondary tiny-btn" onclick="wgSetResolution(1024,512)">1K</button>
          <button type="button" class="secondary tiny-btn" onclick="wgSetResolution(2048,1024)">2K</button>
          <button type="button" class="secondary tiny-btn" onclick="wgSetResolution(4096,2048)">4K</button>
          <button type="button" class="secondary tiny-btn" onclick="wgSetResolution(8192,4096)">8K</button>
        </div>
      </section>
      <div class="row preset-pills">
        <button type="button" class="secondary" onclick="wgSetResolution(1024,512); document.querySelector('[name=run_to]').value='climate';">Preview climate</button>
        <button type="button" class="secondary" onclick="wgSetResolution(2048,1024); document.querySelector('[name=run_to]').value='terrain-finalization-recentering';">Terrain inspection</button>
        <button type="button" class="secondary" onclick="wgSetResolution(4096,2048); document.querySelector('[name=run_to]').value='outputs';">4K diagnostic</button>
        <button type="button" class="secondary" onclick="document.querySelector('[name=preset]').value='synthetic-earth'; wgPresetChanged(); wgSetResolution(4096,2048);">Synthetic Earth 4K</button>
      </div>
      <label>Run to stage <span class="muted small">generated mode only</span></label><select name="run_to" id="run_to_select">{_stage_options('solar-system')}</select>
      <div class="row">
        <label><input type="checkbox" name="sync_resolution" checked style="width:auto"> keep width / height / image width in tandem</label>
        <label><input type="checkbox" name="yes" checked style="width:auto"> auto-confirm large stages</label>
        <label><input type="checkbox" name="skip_json" checked style="width:auto"> skip JSON on final outputs</label>
        <label title="Intentionally lowers work for a quick look."><input type="checkbox" name="preview" style="width:auto"> preview <span class="muted small">quick low-cost run</span></label>
        <label title="Uses faster settings where supported."><input type="checkbox" name="fast" style="width:auto"> fast <span class="muted small">intentional shortcut</span></label>
        <label title="Skip PNG generation for state-only or speed tests."><input type="checkbox" name="no_images" style="width:auto"> no images</label>
        <label title="Do not compute rivers/lakes/drainage."><input type="checkbox" name="skip_hydrology" style="width:auto"> skip hydrology</label>
        <label><input type="checkbox" name="skip_biomes" style="width:auto"> skip biomes</label>
        <label><input type="checkbox" name="skip_regions" style="width:auto"> skip regions</label>
        <label><input type="checkbox" name="save_rasters" style="width:auto"> save rasters</label>
        <label><input type="checkbox" name="skip_diagnostics" style="width:auto"> skip diagnostics</label>
        <label title="Render images at full simulation resolution instead of the image max width cap."><input type="checkbox" name="full_res_images" style="width:auto"> full-res images</label>
      </div>
      <button type="submit">Create / run</button>
    </form>
  </section>
  <section class="card">
    <h2>Open existing run</h2>
    <form method="get" action="/pipeline">
      <label>Output directory</label><input name="output_dir" placeholder="staged_u47_workflow_test">
      <button type="submit">Open pipeline workspace</button>
    </form>
    <h3>Recent runs</h3>
    <p class="help">The table below checks whether each remembered run folder still exists. Missing entries are common after manual folder cleanup; use <strong>Forget</strong> for one entry or <strong>Purge missing</strong> for all missing entries.</p>
    <div class="row" style="margin-bottom:8px">
      <form method="post" action="/run-admin" style="display:inline">
        <input type="hidden" name="action" value="purge-missing">
        <input type="hidden" name="redirect_to" value="/">
        <button type="submit" class="secondary">Purge missing entries</button>
      </form>
    </div>
    <table><thead><tr><th>Path</th><th>Exists</th><th>Updated</th><th>Summary</th><th>Actions</th></tr></thead><tbody>{recent_rows}</tbody></table>
  </section>
</div>
<script>
function wgN(id) {{ return document.getElementById(id); }}
function wgSyncEnabled() {{ const e=document.querySelector('[name=sync_resolution]'); return !e || e.checked; }}
function wgRoundPow2(v) {{ return Math.max(64, Math.round(v)); }}
function wgSetResolution(w,h) {{ wgN('map_width').value=w; wgN('map_height').value=h; if (wgSyncEnabled()) wgN('image_max_width').value=w; }}
function wgScaleResolution(f) {{ const w=parseInt(wgN('map_width').value||'2048',10); const h=parseInt(wgN('map_height').value||'1024',10); wgSetResolution(wgRoundPow2(w*f), wgRoundPow2(h*f)); }}
function wgSyncResolution(src) {{ if(!wgSyncEnabled()) return; const w=parseInt(wgN('map_width').value||'0',10); const h=parseInt(wgN('map_height').value||'0',10); if(src==='width' && w>0) wgN('map_height').value=Math.max(64, Math.round(w/2)); if(src==='height' && h>0) wgN('map_width').value=Math.max(128, Math.round(h*2)); wgN('image_max_width').value=wgN('map_width').value; }}

function wgSelect(name, value) {{ const el=document.querySelector(`[name=${{name}}]`); if(el) el.value=value; }}
function wgInput(name, value) {{ const el=document.querySelector(`[name=${{name}}]`); if(el) el.value=value; }}
function wgSystemPreset(kind) {{
  const presets = {{
    earth: {{star_class:'G2V', architecture:'solar_like_mixed', pref:'earthlike', moon:'moderate', planets:''}},
    kstar: {{star_class:'K3V', architecture:'compact_rocky_inner', pref:'earthlike', moon:'moderate', planets:''}},
    ocean: {{star_class:'K3V', architecture:'volatile_rich', pref:'oceanic', moon:'moderate', planets:''}},
    dry: {{star_class:'G5V', architecture:'low_mass_quiet', pref:'dry_terrestrial', moon:'weak', planets:''}},
    super: {{star_class:'K0V', architecture:'outer_giant_dominated', pref:'super_earth', moon:'strong', planets:''}},
    cold: {{star_class:'K5V', architecture:'sparse_old', pref:'colder_world', moon:'moderate', planets:''}},
    warm: {{star_class:'G2V', architecture:'compact_rocky_inner', pref:'warmer_world', moon:'moderate', planets:''}}
  }};
  const p = presets[kind] || presets.earth;
  wgSelect('star_class', p.star_class); wgSelect('system_architecture', p.architecture); wgSelect('main_planet_preference', p.pref); wgSelect('moon_strength', p.moon); wgInput('planet_count', p.planets);
}}
function wgPresetChanged() {{ const p=document.getElementById('preset-select').value; const runTo=document.getElementById('run_to_select'); if(runTo) runTo.disabled = (p !== 'generated'); }}
wgPresetChanged();
</script>
"""
        return self._page_shell("Home", body)

    def _page_run(self, params: dict[str, list[str]]) -> str:
        embedded = _first(params, "embedded") == "1"
        output_dir = Path(_first(params, "output_dir")).expanduser()
        remember_run(output_dir)
        exists = output_dir.exists()
        status_rows_html = ""
        override_note = ""
        manifest_html = ""
        run_message_value = _first(params, "run_message")
        run_message = f"<p class='ok'>{_safe_text(run_message_value)}</p>" if run_message_value else ""
        quick_saved = "<p class='ok'>Visual editor values saved to config/stage_overrides.json.</p>" if _first(params, "quick_saved") else ""
        status_actions = ""
        if exists:
            ensure_layout(output_dir)
            write_status_report(output_dir)
            override_reason = unapplied_override_reason(output_dir)
            if override_reason:
                override_note = f"<p class='warn'>Override note: {_safe_text(override_reason)}</p>"
            rows = status_detail_rows(output_dir)
            first_stale = next((row for row in rows if row.get("status") == "stale"), None)
            first_missing = next((row for row in rows if row.get("status") == "missing"), None)
            for row in rows:
                cls = "stale" if row["status"] == "stale" else ("complete" if row["status"] == "complete" else "missing")
                mini = ""
                if row["status"] in {"stale", "missing"}:
                    action = "run-from" if row["status"] == "stale" else "run-to"
                    mini = (
                        f"<form method='post' action='/stage-action' style='display:inline'>"
                        f"<input type='hidden' name='output_dir' value='{_safe_text(output_dir)}'>"
                        f"<input type='hidden' name='action' value='{action}'>"
                        f"<input type='hidden' name='stage' value='{_safe_text(row['stage'])}'>"
                        f"<input type='hidden' name='image_max_width' value='2048'>"
                        f"<input type='hidden' name='yes' value='on'>"
                        f"<input type='hidden' name='skip_json' value='on'>"
                        f"<button class='secondary small' type='submit'>{_safe_text(action)}</button></form>"
                    )
                status_rows_html += f"<tr><td>{_safe_text(row['stage'])}</td><td class='{cls}'>{_safe_text(row['status'])}</td><td>{_safe_text(row['completed_at'])}</td><td>{_safe_text(row['reason'])} {mini}</td></tr>"
            if first_stale:
                st = str(first_stale.get("stage", ""))
                status_actions = (
                    f"<div class='sticky-tools'><strong>Suggested recovery:</strong> first stale stage is <code>{_safe_text(st)}</code>. "
                    f"<form method='post' action='/stage-action' style='display:inline'>"
                    f"<input type='hidden' name='output_dir' value='{_safe_text(output_dir)}'>"
                    f"<input type='hidden' name='action' value='run-from'>"
                    f"<input type='hidden' name='stage' value='{_safe_text(st)}'>"
                    f"<input type='hidden' name='image_max_width' value='2048'>"
                    f"<input type='hidden' name='yes' value='on'>"
                    f"<input type='hidden' name='skip_json' value='on'>"
                    f"<button type='submit'>Run from {_safe_text(st)}</button></form></div>"
                )
            elif first_missing:
                st = str(first_missing.get("stage", ""))
                status_actions = (
                    f"<div class='sticky-tools'><strong>Next stage:</strong> first missing stage is <code>{_safe_text(st)}</code>. "
                    f"<form method='post' action='/stage-action' style='display:inline'>"
                    f"<input type='hidden' name='output_dir' value='{_safe_text(output_dir)}'>"
                    f"<input type='hidden' name='action' value='run-to'>"
                    f"<input type='hidden' name='stage' value='{_safe_text(st)}'>"
                    f"<input type='hidden' name='image_max_width' value='2048'>"
                    f"<input type='hidden' name='yes' value='on'>"
                    f"<input type='hidden' name='skip_json' value='on'>"
                    f"<button type='submit'>Run to {_safe_text(st)}</button></form></div>"
                )
            manifest = read_manifest(output_dir)
            manifest_html = f"<p><span class='pill'>Pipeline: {_safe_text(manifest.get('pipeline_version',''))}</span> <span class='pill'>Updated: {_safe_text(manifest.get('updated_at',''))}</span></p>"
        else:
            status_rows_html = "<tr><td colspan='4' class='bad'>Directory does not exist yet.</td></tr>"

        map_cards = _map_inventory_html(output_dir, full_inventory=True) if exists else "<p class='muted'>No maps available yet.</p>"
        map_index_link = f"<a class='button secondary' href='/map-index?output_dir={urllib.parse.quote(str(output_dir))}'>Open map index</a>" if exists else ''
        region_folder = ""

        quick_editor = ""
        if exists:
            overrides = _read_json(output_dir / "config" / "stage_overrides.json", {})
            if not isinstance(overrides, dict):
                overrides = {}
            solar = _solar_system_state(output_dir) or {}
            planets = solar.get("planets", []) if isinstance(solar.get("planets"), list) else []
            main_planet = next((p for p in planets if isinstance(p, dict) and p.get("is_main_planet")), {})
            physics = _read_json(output_dir / "state" / "02_planet_physics.json", {})
            if not isinstance(physics, dict):
                physics = {}

            def qv(form_key: str, override_path: list[str], actual: Any = "") -> str:
                ov = _get_nested(overrides, override_path, None)
                value = actual if ov is None else ov
                return _safe_text("" if value is None else value)

            star = solar.get("star", {}) if isinstance(solar.get("star"), dict) else {}
            diagnostics_state = solar.get("diagnostics", {}) if isinstance(solar.get("diagnostics"), dict) else {}
            moon_state = main_planet.get("moon", {}) if isinstance(main_planet.get("moon"), dict) else {}
            orbit = main_planet.get("orbit", {}) if isinstance(main_planet.get("orbit"), dict) else {}
            rotation = physics.get("rotation", {}) if isinstance(physics.get("rotation"), dict) else {}
            atmosphere = physics.get("atmosphere", {}) if isinstance(physics.get("atmosphere"), dict) else {}
            hydrosphere = physics.get("hydrosphere", {}) if isinstance(physics.get("hydrosphere"), dict) else {}
            geology = physics.get("geology", {}) if isinstance(physics.get("geology"), dict) else {}
            require_major_moon_value = _get_nested(overrides, ["system", "require_major_moon"], None)
            if require_major_moon_value is None:
                require_major_moon_value = bool(moon_state) if main_planet else True
            require_major_moon_checked = " checked" if bool(require_major_moon_value) else ""
            architecture_actual = solar.get("architecture") or diagnostics_state.get("architecture") or ""
            preference_actual = diagnostics_state.get("main_planet_preference") or "earthlike"
            moon_strength_actual = _get_nested(overrides, ["system", "moon_strength_preference"], None) or moon_state.get("tidal_effect_level") or "moderate"
            quick_editor = f"""
  <section class="card visual-editor" style="margin-top:18px">
    <details open>
      <summary><strong>Full visual parameter editor</strong> <span class="muted">guided controls, suggested ranges, and override output</span></summary>
      <form method="post" action="/quick-edit" class="quick-form">
        <input type="hidden" name="output_dir" value="{_safe_text(output_dir)}">
        <h3>Solar-system generator controls</h3>
        <p class="help">These affect the next solar-system rerun. Saving actual generated values here intentionally fixes them as overrides.</p>
        <div class="grid compact-grid">
          {_select_control('star_stellar_class', 'Star class / subtype', qv('star_stellar_class', ['star','stellar_class'], star.get('spectral_type') or star.get('stellar_class') or ''), [("", "Auto / generated"), ("G", "G class"), ("K", "K class"), ("G0V", "G0V"), ("G2V", "G2V"), ("G5V", "G5V"), ("K0V", "K0V"), ("K3V", "K3V"), ("K5V", "K5V"), ("K7V", "K7V")], help_text='Stored as a star override for the next solar-system stage.')}
          {_select_control('system_architecture_type', 'System architecture', qv('system_architecture_type', ['system','architecture_type'], architecture_actual), SYSTEM_ARCHITECTURE_OPTIONS, help_text='Random/weighted stores null; specific choices fix the next orbital architecture.')}
          {_select_control('main_planet_preference', 'Main Planet preference', qv('main_planet_preference', ['system','main_planet_preference'], preference_actual), MAIN_PLANET_PREFERENCE_OPTIONS, help_text='Controls HZ candidate bias and main-world candidate scoring.')}
          {_select_control('moon_strength_preference', 'Moon tide strength', qv('moon_strength_preference', ['system','moon_strength_preference'], moon_strength_actual), MOON_STRENGTH_OPTIONS, help_text='Controls generated major moon mass/orbit tendency.')}
          <div class="visual-editor-field"><label>Require major moon</label><input type="hidden" name="require_major_moon_present" value="1"><label><input type="checkbox" name="require_major_moon" style="width:auto"{require_major_moon_checked}> include one major moon</label><p class="field-help">Unchecked worlds skip the guaranteed major moon and expose low axial-stability cases.</p></div>
        </div>
        <h3>Star / selected planet</h3>
        <p class="help">These values are saved as overrides. The current state remains visible elsewhere on the dashboard; use Apply immediately when you want downstream stages marked stale and validated.</p>
        <div class="grid compact-grid">
          <div class="visual-editor-field"><label>Star mass M☉</label><input name="star_mass_solar" value="{qv('star_mass_solar', ['star','mass_solar'], star.get('mass_solar'))}"><p class="field-help">G/K habitable-system stars are often roughly 0.7–1.1 M☉.</p></div>
          <div class="visual-editor-field"><label>Star luminosity L☉</label><input name="star_luminosity_solar" value="{qv('star_luminosity_solar', ['star','luminosity_solar'], star.get('luminosity_solar'))}"><p class="field-help">Controls habitable zone distance and received stellar flux.</p></div>
          <div class="visual-editor-field"><label>Main planet radius R⊕</label><input name="main_radius_earth" value="{qv('main_radius_earth', ['main_planet','radius_earth'], main_planet.get('radius_earth'))}"><p class="field-help">Affects map physical scale, gravity expectations, climate distances, and terrain scale.</p></div>
          <div class="visual-editor-field"><label>Main planet mass M⊕</label><input name="main_mass_earth" value="{qv('main_mass_earth', ['main_planet','mass_earth'], main_planet.get('mass_earth'))}"><p class="field-help">Use with radius and gravity; avoid impossible combinations unless testing.</p></div>
          <div class="visual-editor-field"><label>Surface gravity g</label><input name="main_gravity_g" value="{qv('main_gravity_g', ['main_planet','surface_gravity_g'], main_planet.get('surface_gravity_g'))}"><p class="field-help">Comfortable Earth-like range is roughly 0.7–1.4 g.</p></div>
          <div class="visual-editor-field"><label>Orbit semi-major axis AU</label><input name="orbit_semi_major_axis_au" value="{qv('orbit_semi_major_axis_au', ['main_planet','orbit','semi_major_axis_au'], orbit.get('semi_major_axis_au'))}"><p class="field-help">Compare with the habitable-zone strip above before changing.</p></div>
          <div class="visual-editor-field"><label>Orbit eccentricity</label><input name="orbit_eccentricity" value="{qv('orbit_eccentricity', ['main_planet','orbit','eccentricity'], orbit.get('eccentricity'))}"><p class="field-help">Lower values give more stable seasons; high values create strong orbital seasonality.</p></div>
        </div>
        <h3>Planet physics</h3>
        <div class="grid compact-grid">
          {_slider_control('rotation_period_hours', 'Rotation period', qv('rotation_period_hours', ['planet_physics','rotation','rotation_period_hours'], rotation.get('rotation_period_hours')), min_v=8, max_v=96, step=0.5, unit='hours', help_text='Short days strengthen Coriolis behavior; long days broaden circulation cells.', suggested='18–36 h for many Earthlike tests')}
          {_slider_control('axial_tilt_degrees', 'Axial tilt', qv('axial_tilt_degrees', ['planet_physics','rotation','axial_tilt_degrees'], rotation.get('axial_tilt_degrees')), min_v=0, max_v=80, step=0.5, unit='degrees', help_text='Controls seasonal contrast and high-latitude climate.', suggested='10–35° for moderate seasons')}
          {_slider_control('pressure_bar', 'Pressure', qv('pressure_bar', ['planet_physics','atmosphere','pressure_bar'], atmosphere.get('pressure_bar')), min_v=0.1, max_v=5, step=0.05, unit='bar', help_text='Affects heat transport and water stability.', suggested='0.7–2.0 bar for many habitable worlds')}
          {_slider_control('co2_ppm', 'CO₂', qv('co2_ppm', ['planet_physics','atmosphere','carbon_dioxide_ppm'], atmosphere.get('carbon_dioxide_ppm')), min_v=0, max_v=5000, step=10, unit='ppm', help_text='One greenhouse control. High values can warm or dry climate depending on other settings.', suggested='250–1000 ppm for mild tests')}
          <div><label>Greenhouse warming K</label><input name="greenhouse_warming_k" value="{qv('greenhouse_warming_k', ['planet_physics','atmosphere','greenhouse_warming_k'], atmosphere.get('greenhouse_warming_k'))}"></div>
          <div><label>Water vapor factor</label><input name="water_vapor_factor" value="{qv('water_vapor_factor', ['planet_physics','atmosphere','water_vapor_factor'], atmosphere.get('water_vapor_factor'))}"></div>
          {_slider_control('ocean_fraction_target', 'Ocean fraction target', qv('ocean_fraction_target', ['planet_physics','hydrosphere','ocean_fraction_target'], hydrosphere.get('ocean_fraction_target')), min_v=0.05, max_v=0.95, step=0.01, unit='fraction', help_text='Target ocean coverage before terrain/hydrology adjust details.', suggested='0.45–0.75 for mixed land/ocean')}
          {_slider_control('volcanism', 'Volcanism', qv('volcanism', ['planet_physics','geology','volcanism'], geology.get('volcanism')), min_v=0, max_v=1, step=0.01, unit='0–1', help_text='Should influence arcs, hotspot islands, resurfacing, and atmospheric outgassing.', suggested='0.3–0.8')}
          {_slider_control('erosion', 'Erosion', qv('erosion', ['planet_physics','geology','erosion'], geology.get('erosion')), min_v=0, max_v=1, step=0.01, unit='0–1', help_text='Controls smoothing and sediment transfer in later terrain/hydrology work.', suggested='0.25–0.7')}
          {_slider_control('mountain_factor', 'Mountain factor', qv('mountain_factor', ['planet_physics','geology','mountain_factor'], geology.get('mountain_factor')), min_v=0, max_v=2, step=0.01, unit='factor', help_text='Scales mountain prominence and orographic effects downstream.', suggested='0.7–1.4')}
          {_slider_control('surface_roughness', 'Surface roughness', qv('surface_roughness', ['planet_physics','geology','surface_roughness'], geology.get('surface_roughness')), min_v=0, max_v=1, step=0.01, unit='0–1', help_text='Higher roughness should preserve more local relief and drainage complexity.', suggested='0.25–0.75')}
        </div>
        <h3>Terrain controls</h3>
        <p class="muted">These are stored in overrides and consumed by <code>plate_history_v3</code> and <code>plate_history_v4</code> where noted. Use them to sweep terrain behavior without editing code.</p>
        <div class="grid compact-grid">
          <div><label>Target plate count</label><input name="target_plate_count" value="{qv('target_plate_count', ['terrain','target_plate_count'], '')}"></div>
          <div><label>Island density</label><input name="island_density" value="{qv('island_density', ['terrain','island_density'], '')}"></div>
          <div><label>Shelf width factor</label><input name="shelf_width_factor" value="{qv('shelf_width_factor', ['terrain','shelf_width_factor'], '')}"><p class="field-help">v3/v4: broadens or narrows the submerged continental apron.</p></div>
          <div><label>Continental shelf strength</label><input name="continental_shelf_strength" value="{qv('continental_shelf_strength', ['terrain','continental_shelf_strength'], '')}"><p class="field-help">v3/v4: shallower/wider passive continental shelves.</p></div>
          <div><label>Erosion/deposition multiplier</label><input name="erosion_deposition_multiplier" value="{qv('erosion_deposition_multiplier', ['terrain','erosion_deposition_multiplier'], '')}"><p class="field-help">v3/v4: master multiplier for erosion, sediment fill, and deposition.</p></div>
          <div><label>v4 topology strength</label><input name="v4_topology_strength" value="{qv('v4_topology_strength', ['terrain','v4_topology_strength'], '')}"><p class="field-help">v4 only: boundary deformation, microplates, and mountain branching.</p></div>
          <div><label>v4 island strength</label><input name="v4_island_strength" value="{qv('v4_island_strength', ['terrain','v4_island_strength'], '')}"><p class="field-help">v4 only: supported volcanic island-chain uplift.</p></div>
          <div><label>v4 rift strength</label><input name="v4_rift_strength" value="{qv('v4_rift_strength', ['terrain','v4_rift_strength'], '')}"><p class="field-help">v4 only: supported rift cuts, gulfs, and narrow seas.</p></div>
        </div>
        <div class="row">
          <button type="submit">Save visual editor values</button>
          <label><input type="checkbox" name="apply_now" style="width:auto"> apply immediately and validate</label>
          <a class="button secondary" href="/overrides?output_dir={urllib.parse.quote(str(output_dir))}">Raw JSON editor</a>
        </div>
      </form>
    </details>
  </section>
"""

        diagnostics = ""
        if exists:
            for path in _diagnostic_files(output_dir):
                rel = path.relative_to(output_dir) if path.is_relative_to(output_dir) else path.name
                url = "/file/" + urllib.parse.quote(str(path.resolve()))
                diagnostics += f"<li><a href='{url}' target='_blank'>{_safe_text(rel)}</a></li>"
        diagnostics = diagnostics or "<li class='muted'>No diagnostic files available yet.</li>"

        state_files = ""
        if exists:
            for path in _state_files(output_dir):
                rel = path.relative_to(output_dir)
                url = "/file/" + urllib.parse.quote(str(path.resolve()))
                state_files += f"<li><a href='{url}' target='_blank'>{_safe_text(rel)}</a></li>"
        state_files = state_files or "<li class='muted'>No state files available yet.</li>"

        run_jobs_html = ""
        stage_timing_html = _stage_timing_html(output_dir) if exists else ""
        if exists:
            with JOBS_LOCK:
                live_jobs = [j for j in JOBS.values() if j.output_dir is not None and Path(j.output_dir) == output_dir]
            history = {j.id: _job_record(j) for j in live_jobs}
            for item in _historical_jobs_for_run(output_dir):
                jid = str(item.get("id", ""))
                if jid and jid not in history:
                    history[jid] = item
            rows = ""
            for item in sorted(history.values(), key=lambda it: str(it.get("created_at", "")), reverse=True)[:12]:
                status = str(item.get("status", ""))
                cls = "complete" if status == "complete" else ("failed" if status == "failed" else ("missing" if status == "cancelled" else "stale"))
                jid = str(item.get("id", ""))
                if jid in JOBS:
                    id_link = f"<a href='/job/{_safe_text(jid)}'>{_safe_text(jid)}</a>"
                elif item.get("log_path"):
                    log_url = "/file/" + urllib.parse.quote(str(Path(str(item.get("log_path"))).resolve()))
                    id_link = f"<a href='{log_url}' target='_blank'>{_safe_text(jid)}</a>"
                else:
                    id_link = _safe_text(jid)
                rows += f"<tr><td>{id_link}</td><td>{_safe_text(item.get('label',''))}</td><td class='{cls}'>{_safe_text(status)}</td><td>{_safe_text(item.get('created_at',''))}</td></tr>"
            if rows:
                run_jobs_html = f"<section class='card' style='margin-top:18px'><h2>Jobs for this run</h2><table><thead><tr><th>ID/log</th><th>Label</th><th>Status</th><th>Created</th></tr></thead><tbody>{rows}</tbody></table></section>"

        body = f"""
<h1>Pipeline workspace</h1>
<p><code>{_safe_text(output_dir)}</code></p>
{run_message}
{manifest_html}
{override_note}
{quick_saved}
<p class="help">Use the stage cards above for normal review actions. This collapsed panel keeps run-through controls, generated maps, diagnostics, job history, and state files available without cluttering the main Pipeline view.</p>
{status_actions}
<div class="grid">
  <section class="card">
    <h2>Stage controls</h2>
    <form method="post" action="/stage-action">
      <input type="hidden" name="output_dir" value="{_safe_text(output_dir)}">
      <label>Action</label>
      <select name="action">
        <option>run-to</option><option>run-from</option><option>run-stage</option><option>maps</option><option>status</option><option>validate</option><option>apply-overrides</option>
      </select>
      <label>Stage</label><select name="stage">{_stage_options('solar-system')}</select>
      <div class="grid">
        <div><label>Image max width</label><input name="image_max_width" value="2048"></div>
      </div>
      <div class="row">
        <label><input type="checkbox" name="yes" checked style="width:auto"> auto-confirm large stages</label>
        <label><input type="checkbox" name="skip_json" checked style="width:auto"> skip JSON</label>
        <label><input type="checkbox" name="validate" checked style="width:auto"> validate after apply-overrides</label>
        <label title="Skip PNG generation for state-only or speed tests."><input type="checkbox" name="no_images" style="width:auto"> no images</label>
        <label><input type="checkbox" name="save_rasters" style="width:auto"> save rasters</label>
        <label><input type="checkbox" name="skip_diagnostics" style="width:auto"> skip diagnostics</label>
      </div>
      <button type="submit">Start job</button>
      <a class="button secondary" href="/overrides?output_dir={urllib.parse.quote(str(output_dir))}">Advanced: raw overrides</a>
    </form>
  </section>
  <section class="card">
    <h2>Status</h2>
    <table><thead><tr><th>Stage</th><th>Status</th><th>Completed</th><th>Reason</th></tr></thead><tbody>{status_rows_html}</tbody></table>
  </section>
</div>
{_run_overview_html(output_dir) if exists else ""}
{quick_editor}
{stage_timing_html}
{run_jobs_html}
<section class="card" style="margin-top:18px">
  <h2>Available maps</h2>
  {map_cards}
</section>
<div class="grid" style="margin-top:18px">
  <section class="card"><h2>Diagnostics</h2><ul>{diagnostics}</ul></section>
  <section class="card"><h2>State files</h2><ul>{state_files}</ul></section>
</div>
"""
        if embedded:
            body = body.replace("<h1>Pipeline workspace</h1>", "<h2>Run tools, maps, diagnostics, and files</h2>", 1)
            return body
        return self._page_shell("Run", body)

    def _page_overrides(self, params: dict[str, list[str]]) -> str:
        output_dir = Path(_first(params, "output_dir")).expanduser()
        ensure_layout(output_dir)
        path = output_dir / "config" / "stage_overrides.json"
        if path.exists():
            text = path.read_text(encoding="utf-8")
        else:
            text = "{}"
        saved = "<p class='ok'>Overrides saved.</p>" if _first(params, "saved") else ""
        body = f"""
<h1>Edit stage overrides</h1>
<p><code>{_safe_text(output_dir)}</code></p>
{saved}
<form method="post" action="/save-overrides">
  <input type="hidden" name="output_dir" value="{_safe_text(output_dir)}">
  <textarea name="overrides" spellcheck="false">{_safe_text(text)}</textarea>
  <div class="row">
    <button type="submit">Save overrides</button>
    <label><input type="checkbox" name="apply_now" style="width:auto"> apply immediately and validate</label>
    <a class="button secondary" href="/pipeline?output_dir={urllib.parse.quote(str(output_dir))}">Back to pipeline</a>
  </div>
</form>
"""
        return self._page_shell("Overrides", body)

    def _page_planet(self, params: dict[str, list[str]]) -> str:
        output_dir = Path(_first(params, "output_dir")).expanduser()
        solar = _solar_system_state(output_dir) or {}
        star = solar.get("star", {}) if isinstance(solar.get("star"), dict) else {}
        main = _main_planet_state(output_dir)
        physics = _planet_physics_state(output_dir)
        if not main:
            body = f"""
<h1>Main Planet viewer</h1>
<p><code>{_safe_text(output_dir)}</code></p>
<section class="card"><p class="muted">No selected Main Planet is available yet. Run to <code>solar-system</code> first.</p></section>
"""
            return self._page_shell("Main Planet", body)
        orbit = main.get("orbit", {}) if isinstance(main.get("orbit"), dict) else {}
        comp = main.get("composition", {}) if isinstance(main.get("composition"), dict) else {}
        rotation = physics.get("rotation", {}) if isinstance(physics.get("rotation"), dict) else {}
        atmosphere = physics.get("atmosphere", {}) if isinstance(physics.get("atmosphere"), dict) else {}
        hydrosphere = physics.get("hydrosphere", {}) if isinstance(physics.get("hydrosphere"), dict) else {}
        geology = physics.get("geology", {}) if isinstance(physics.get("geology"), dict) else {}
        tilt = _safe_float(rotation.get("axial_tilt_degrees"))
        if tilt is None:
            tilt = 0.0
        hz_html = _hz_position_indicator(star, main)
        maps = _available_maps(output_dir) if output_dir.exists() else []
        map_links = []
        for path in maps:
            name = path.name.lower()
            if any(token in name for token in ("terrain", "koppen", "biome", "precip", "temperature", "hydrology")):
                map_links.append(f"<a class='button secondary' href='{_map_view_url(path, output_dir)}'>{_safe_text(_rel_label(path, output_dir))}</a>")
        map_links_html = " ".join(map_links[:12]) or "<span class='muted'>Run terrain/climate/biome stages to populate planet maps.</span>"
        safe_tilt = max(-80.0, min(80.0, tilt))
        rows = ""
        for label, value, unit in [
            ("Orbit", orbit.get("semi_major_axis_au"), "AU"),
            ("Eccentricity", orbit.get("eccentricity"), ""),
            ("Orbital period", orbit.get("orbital_period_days"), "days"),
            ("Mass", main.get("mass_earth"), "M⊕"),
            ("Radius", main.get("radius_earth"), "R⊕"),
            ("Surface gravity", main.get("surface_gravity_g"), "g"),
            ("Escape velocity", main.get("escape_velocity_relative_earth"), "Earth"),
            ("Stellar flux", main.get("stellar_flux_earth"), "S⊕"),
            ("Equilibrium temp", main.get("equilibrium_temperature_k"), "K"),
            ("Habitability score", main.get("habitability_score"), ""),
        ]:
            rows += f"<tr><th>{_safe_text(label)}</th><td>{_safe_text(_fmt_compact(value, 3 if label in {'Orbit','Eccentricity'} else 2, (' ' + unit) if unit else ''))}</td></tr>"
        notes = main.get("selection_notes") if isinstance(main.get("selection_notes"), list) else []
        notes_html = "".join(f"<li>{_safe_text(note)}</li>" for note in notes[:10]) or "<li class='muted'>No selection notes saved.</li>"
        body = f"""
<h1>Main Planet viewer</h1>
<p><code>{_safe_text(output_dir)}</code></p>
<div class="row sticky-tools">
  <a class="button secondary" href="/pipeline?output_dir={urllib.parse.quote(str(output_dir))}">Back to pipeline</a>
  <a class="button secondary" href="/system?output_dir={urllib.parse.quote(str(output_dir))}">Solar-system viewer</a>
  <a class="button secondary" href="/physics?output_dir={urllib.parse.quote(str(output_dir))}">Planet Physics review</a>
  <a class="button secondary" href="/overrides?output_dir={urllib.parse.quote(str(output_dir))}">Edit raw overrides</a>
</div>
<div class="planet-dashboard">
  <section class="card planet-hero">
    <div class="planet-orb" style="--tilt:{safe_tilt:.1f}deg"></div>
    <div>
      <h2>{_safe_text(main.get('name','Main Planet'))}</h2>
      <p>{_planet_visual(main.get('planet_class'), True)} <strong>{_safe_text(main.get('planet_class','planet'))}</strong></p>
      <p class="muted">{_safe_text(comp.get('composition_class','composition not saved'))}</p>
      <div class="grid compact-grid">
        {_metric_gauge('Habitability', main.get('habitability_score'), min_v=0, max_v=1, low='poor', mid='possible', high='strong')}
        {_metric_gauge('Stellar flux', main.get('stellar_flux_earth'), unit='S⊕', min_v=.25, max_v=2.2, low='cold/dim', mid='temperate', high='hot')}
        {_metric_gauge('Gravity', main.get('surface_gravity_g'), unit='g', min_v=.25, max_v=2.2, low='low', mid='comfortable', high='high')}
      </div>
    </div>
  </section>
  <section class="card">
    <h2>Orbit and seasons</h2>
    {hz_html}
    <div class="tilt-widget" title="Axial tilt {safe_tilt:.1f}°"><div class="tilt-axis" style="--tilt:{safe_tilt:.1f}deg"></div></div>
    <p class="muted small">Axial tilt: {_safe_text(_fmt_compact(rotation.get('axial_tilt_degrees'), 1, '°')) or '—'} · Rotation: {_safe_text(_fmt_compact(rotation.get('rotation_period_hours'), 1, ' h')) or '—'}</p>
  </section>
</div>
<div class="grid" style="margin-top:18px">
  <section class="card">
    <h2>Physical summary</h2>
    <table><tbody>{rows}</tbody></table>
  </section>
  <section class="card">
    <h2>Atmosphere / hydrosphere / geology</h2>
    <div class="grid compact-grid">
      {_metric_gauge('Pressure', atmosphere.get('pressure_bar'), unit='bar', min_v=.1, max_v=5, low='thin', mid='moderate', high='thick')}
      {_metric_gauge('CO₂', atmosphere.get('carbon_dioxide_ppm'), unit='ppm', min_v=0, max_v=5000, low='low', mid='moderate', high='high')}
      {_metric_gauge('Ocean target', hydrosphere.get('ocean_fraction_target'), min_v=0, max_v=1, low='dry', mid='mixed', high='ocean world')}
      {_metric_gauge('Volcanism', geology.get('volcanism'), min_v=0, max_v=1, low='quiet', mid='active', high='very active')}
      {_metric_gauge('Erosion', geology.get('erosion'), min_v=0, max_v=1, low='weak', mid='moderate', high='strong')}
      {_metric_gauge('Mountain factor', geology.get('mountain_factor'), min_v=0, max_v=2, low='low relief', mid='moderate', high='rugged')}
    </div>
  </section>
</div>
<section class="card" style="margin-top:18px">
  <h2>Selection notes</h2>
  <ul>{notes_html}</ul>
</section>
<section class="card" style="margin-top:18px">
  <h2>Planet maps</h2>
  <p class="help">These open in the map viewer. Full-world maps can also be viewed as globes from the map page.</p>
  <div class="row">{map_links_html}</div>
</section>
"""
        return self._page_shell("Main Planet", body)


    def _page_physics(self, params: dict[str, list[str]]) -> str:
        output_dir = Path(_first(params, "output_dir")).expanduser()
        physics = _planet_physics_state(output_dir)
        physics_message_value = _first(params, "physics_message")
        physics_message = f"<p class='ok'>{_safe_text(physics_message_value)}</p>" if physics_message_value else ""
        if not physics:
            body = f"""
<h1>Stage 2 Planet Physics review</h1>
<p><code>{_safe_text(output_dir)}</code></p>
<div class="row"><a class="button secondary" href="/pipeline?output_dir={urllib.parse.quote(str(output_dir))}">Back to pipeline</a><a class="button secondary" href="/system?output_dir={urllib.parse.quote(str(output_dir))}">Stage 1 Solar System</a></div>
<section class="card" style="margin-top:18px"><p class="muted">No planet-physics state found yet. Run to <code>planet-physics</code> first.</p></section>
"""
            return self._page_shell("Planet Physics", body)
        review = physics.get("review", {}) if isinstance(physics.get("review"), dict) else {}
        warnings = review.get("warnings", []) if isinstance(review.get("warnings"), list) else []
        context = review.get("stage1_context_used", {}) if isinstance(review.get("stage1_context_used"), dict) else {}
        body = f"""
<h1>Stage 2 Planet Physics review</h1>
<p><code>{_safe_text(output_dir)}</code></p>
{physics_message}
<div class="row sticky-tools">
  <a class="button secondary" href="/pipeline?output_dir={urllib.parse.quote(str(output_dir))}">Back to pipeline</a>
  <a class="button secondary" href="/system?output_dir={urllib.parse.quote(str(output_dir))}">Stage 1 Solar System</a>
  <a class="button secondary" href="/planet?output_dir={urllib.parse.quote(str(output_dir))}">Main Planet viewer</a>
  <a class="button secondary" href="/overrides?output_dir={urllib.parse.quote(str(output_dir))}">Raw overrides</a>
</div>
{_stage2_acceptance_html(output_dir)}
{_physics_actions_html(output_dir)}
{_physics_report_html(output_dir)}
<section class="card" style="margin-top:18px">
  <h2>Stage 1 context used by Stage 2</h2>
  <p class="help">These are the solar-system/formation values carried forward into the physical profile.</p>
  <div class="grid compact-grid">{_mapping_stat_items(context, ["formation_zone", "volatile_delivery", "giant_planet_influence", "impact_history", "tectonic_energy_bias", "crustal_asymmetry_bias", "moon_origin", "tidal_effect_level", "axial_stability_effect", "main_planet_preference"])}</div>
</section>
<section class="card" style="margin-top:18px">
  <h2>Warnings and review notes</h2>
  <ul>{_physics_warning_list(warnings)}</ul>
</section>
{_physics_downstream_html(output_dir)}
"""
        return self._page_shell("Planet Physics", body)


    def _page_system(self, params: dict[str, list[str]]) -> str:
        output_dir = Path(_first(params, "output_dir")).expanduser()
        data = _solar_system_state(output_dir)
        if data is None:
            body = f"""
<h1>Solar-system viewer</h1>
<p><code>{_safe_text(output_dir)}</code></p>
<section class="card"><p class="muted">No solar-system state found yet. Run to <code>solar-system</code> first.</p></section>
"""
            return self._page_shell("Solar system", body)
        star = data.get("star", {}) if isinstance(data.get("star"), dict) else {}
        diagnostics_state = data.get("diagnostics", {}) if isinstance(data.get("diagnostics"), dict) else {}
        planets = data.get("planets", []) if isinstance(data.get("planets"), list) else []
        main_planet = next((p for p in planets if isinstance(p, dict) and p.get("is_main_planet")), {})
        main_context = main_planet.get("formation_context", {}) if isinstance(main_planet.get("formation_context"), dict) else {}
        scale_mode = _first(params, "scale", "log")
        if scale_mode not in {"log", "linear"}:
            scale_mode = "log"
        show_labels = _first(params, "labels", "show") != "hide"
        show_hz = _first(params, "hz", "show") != "hide"
        show_snow = _first(params, "snow", "show") != "hide"
        solar_message_value = _first(params, "solar_message")
        solar_message = f"<p class='ok'>{_safe_text(solar_message_value)}</p>" if solar_message_value else ""
        rows = ""
        for planet in planets:
            if not isinstance(planet, dict):
                continue
            orbit = planet.get("orbit", {}) if isinstance(planet.get("orbit"), dict) else {}
            cls = "complete" if planet.get("is_main_planet") else ""
            context = planet.get("formation_context", {}) if isinstance(planet.get("formation_context"), dict) else {}
            context_text = " · ".join(
                _humanize_key(context.get(key)) for key in ("formation_zone", "volatile_delivery", "impact_history") if context.get(key)
            )
            rows += (
                f"<tr class='{cls}'><td>{_planet_visual(planet.get('planet_class'), bool(planet.get('is_main_planet')))}{_safe_text(planet.get('name'))}</td>"
                f"<td>{_safe_text(planet.get('planet_class'))}</td>"
                f"<td>{_safe_text(_humanize_key(planet.get('architecture_role')))}</td>"
                f"<td>{_safe_text(_fmt_num(orbit.get('semi_major_axis_au'), 3))}</td>"
                f"<td>{_safe_text(_fmt_num(planet.get('mass_earth'), 2))}</td>"
                f"<td>{_safe_text(_fmt_num(planet.get('radius_earth'), 2))}</td>"
                f"<td>{_safe_text(_fmt_num(planet.get('stellar_flux_earth'), 2))}</td>"
                f"<td>{_safe_text(_fmt_num(planet.get('habitability_score'), 2))}</td>"
                f"<td>{_safe_text(context_text)}</td></tr>"
            )
        rows = rows or "<tr><td colspan='9' class='muted'>No planets listed.</td></tr>"
        diagnostic_html = ""
        if diagnostics_state or main_context:
            diagnostic_html = f"""
<section class="card" style="margin-top:18px">
  <h2>Stage 1 diagnostics</h2>
  <div class="grid compact-grid">{_mapping_stat_items(diagnostics_state, ["architecture", "main_planet_preference", "generation_attempts", "planet_count", "habitable_zone_planet_count", "outer_giant_count", "giant_planet_influence", "main_planet_candidate_quality", "climate_stability_outlook"])}</div>
  <h3>Main Planet formation context</h3>
  <div class="grid compact-grid">{_mapping_stat_items(main_context, ["formation_zone", "volatile_delivery", "giant_planet_influence", "impact_history", "tectonic_energy_bias", "crustal_asymmetry_bias", "moon_origin", "tidal_effect_level", "axial_stability_effect"])}</div>
</section>
"""
        body = f"""
<h1>Solar-system viewer</h1>
<p><code>{_safe_text(output_dir)}</code></p>
{solar_message}
<div class="row"><a class="button secondary" href="/pipeline?output_dir={urllib.parse.quote(str(output_dir))}">Back to pipeline</a><a class="button secondary" href="/planet?output_dir={urllib.parse.quote(str(output_dir))}">Main Planet viewer</a><a class="button secondary" href="/system-report?output_dir={urllib.parse.quote(str(output_dir))}">Stage 1 report</a></div>
{_stage_acceptance_html(output_dir)}
{_solar_review_actions_html(output_dir)}
<section class="card" style="margin-top:18px">
  <h2>Interactive orbital overview</h2>
  <p class="help">Use the controls to switch log/linear distance, hide labels, or toggle the habitable-zone and snow-line overlays. Hover over planet markers for orbit, size, mass, and flux details.</p>
  <form method="get" action="/system" class="row">
    <input type="hidden" name="output_dir" value="{_safe_text(output_dir)}">
    <label>Scale <select name="scale"><option value="log" {'selected' if scale_mode == 'log' else ''}>Log</option><option value="linear" {'selected' if scale_mode == 'linear' else ''}>Linear</option></select></label>
    <label>Labels <select name="labels"><option value="show" {'selected' if show_labels else ''}>Show</option><option value="hide" {'selected' if not show_labels else ''}>Hide</option></select></label>
    <label>Habitable zone <select name="hz"><option value="show" {'selected' if show_hz else ''}>Show</option><option value="hide" {'selected' if not show_hz else ''}>Hide</option></select></label>
    <label>Snow line <select name="snow"><option value="show" {'selected' if show_snow else ''}>Show</option><option value="hide" {'selected' if not show_snow else ''}>Hide</option></select></label>
    <button class="secondary" type="submit">Update view</button>
  </form>
  <div class="system-viewer">{_solar_system_svg(data, scale_mode=scale_mode, show_labels=show_labels, show_hz=show_hz, show_snow=show_snow)}</div>
  <div class="orbit-legend"><span>{_planet_visual('rocky')}rocky/terrestrial</span><span>{_planet_visual('gas_giant')}gas giant</span><span>{_planet_visual('ice_giant')}ice/mini-Neptune</span><span>{_planet_visual('rocky', True)}Main Planet outline</span></div>
</section>
{_stage1_report_html(data)}
{diagnostic_html}
<section class="card" style="margin-top:18px">
  <h2>Planets and Main Planet candidate scoring</h2>
  {_stage1_candidate_table_html(data)}
</section>
<div class="grid" style="margin-top:18px">
  <section class="card">
    <h2>Star</h2>
    {_star_visual(star.get('stellar_class'), star.get('temperature_k'))}
    <table style="margin-top:10px"><tbody>
      <tr><th>Class</th><td>{_safe_text(star.get('spectral_type') or star.get('stellar_class'))}</td></tr>
      <tr><th>Description</th><td>{_safe_text(star.get('stellar_description'))}</td></tr>
      <tr><th>Architecture</th><td>{_safe_text(_humanize_key(data.get('architecture') or diagnostics_state.get('architecture')))}</td></tr>
      <tr><th>Mass</th><td>{_safe_text(_fmt_num(star.get('mass_solar'), 3))} M☉</td></tr>
      <tr><th>Luminosity</th><td>{_safe_text(_fmt_num(star.get('luminosity_solar'), 3))} L☉</td></tr>
      <tr><th>Temperature</th><td>{_safe_text(_fmt_num(star.get('temperature_k'), 0))} K</td></tr>
      <tr><th>Habitable zone</th><td>{_safe_text(_fmt_num(star.get('habitable_zone_inner_au'), 3))}–{_safe_text(_fmt_num(star.get('habitable_zone_outer_au'), 3))} AU</td></tr>
      <tr><th>Snow line</th><td>{_safe_text(_fmt_num(star.get('snow_line_au'), 3))} AU</td></tr>
    </tbody></table>
  </section>
</div>
"""
        return self._page_shell("Solar system", body)

    def _page_system_report(self, params: dict[str, list[str]]) -> str:
        output_dir = Path(_first(params, "output_dir")).expanduser()
        data = _solar_system_state(output_dir)
        if data is None:
            return self._page_shell("Stage 1 report", f"<h1>Stage 1 report</h1><p><code>{_safe_text(output_dir)}</code></p><section class='card'><p class='muted'>No solar-system state found yet.</p></section>")
        body = f"""
<h1>Stage 1 Solar System report</h1>
<p><code>{_safe_text(output_dir)}</code></p>
<div class="row"><a class="button secondary" href="/system?output_dir={urllib.parse.quote(str(output_dir))}">Back to solar-system viewer</a><a class="button secondary" href="/pipeline?output_dir={urllib.parse.quote(str(output_dir))}">Pipeline workspace</a></div>
{_stage_acceptance_html(output_dir)}
{_solar_review_actions_html(output_dir)}
{_stage1_report_html(data)}
<section class="card" style="margin-top:18px">
  <h2>Planets and Main Planet candidate scoring</h2>
  {_stage1_candidate_table_html(data)}
</section>
"""
        return self._page_shell("Stage 1 report", body)

    def _resolve_run_map(self, output_dir: Path, value: str) -> Path | None:
        if not value:
            maps = _all_image_maps(output_dir, include_regions=False)
            return maps[0] if maps else None
        candidate = Path(value)
        if candidate.is_absolute() and candidate.exists() and candidate.is_file():
            return candidate
        for path in _all_image_maps(output_dir, include_regions=True):
            if _rel_label(path, output_dir) == value or path.name == value:
                return path
        rel = output_dir / value
        if rel.exists() and rel.is_file():
            return rel
        return None

    def _map_options(self, output_dir: Path, selected: Path | None = None, *, include_regions: bool = False) -> str:
        selected_key = str(selected.resolve()) if selected is not None else ""
        options = []
        for path in _all_image_maps(output_dir, include_regions=include_regions):
            rel = _rel_label(path, output_dir)
            sel = " selected" if str(path.resolve()) == selected_key else ""
            options.append(f'<option value="{_safe_text(rel)}"{sel}>{_safe_text(rel)}</option>')
        return "\n".join(options)

    def _zoom_script(self, image_id: str = "zoom-img") -> str:
        # Dependency-free pan/zoom with smooth wheel zoom, hover readouts,
        # image-space grids, single-map contours, and dynamic scale text.
        return f"""
<script>
(function() {{
  let scale = 1.0;
  let dragging = false;
  const img = document.getElementById('{image_id}');
  const box = img ? img.closest('.viewer-wrap') : null;
  const stack = document.getElementById('single-stack') || (img ? img.parentElement : null);
  const readout = document.getElementById('map-hover-readout');
  const scaleReadout = document.getElementById('map-scale-readout');
  let hoverTimer = null;
  let lastContourLevel = null;

  function dataHeight() {{
    if (!img || !img.naturalWidth) return 0;
    const host = stack || img;
    const ww = parseFloat(host.dataset.worldWidth || '0');
    const wh = parseFloat(host.dataset.worldHeight || '0');
    if (ww > 0 && wh > 0) return Math.min(img.naturalHeight || img.naturalWidth * wh / ww, img.naturalWidth * wh / ww);
    return img.naturalHeight || 0;
  }}
  function contourLevelsForScale() {{
    if (scale < 0.8) return 5;
    if (scale < 1.8) return 8;
    if (scale < 4.0) return 12;
    if (scale < 9.0) return 16;
    return 22;
  }}
  function updateContourDensity() {{
    const level = contourLevelsForScale();
    if (level === lastContourLevel) return;
    lastContourLevel = level;
    document.querySelectorAll('.contour-layer').forEach(el => {{
      const base = el.dataset.baseSrc || el.getAttribute('src') || '';
      if (!base) return;
      try {{
        const url = new URL(base, window.location.origin);
        const requestedLevels = url.searchParams.get('levels') || 'auto';
        if (requestedLevels === 'auto') url.searchParams.set('zoom', scale.toFixed(3));
        el.src = url.pathname + '?' + url.searchParams.toString();
      }} catch (e) {{}}
    }});
  }}
  function updateGridDensity() {{
    if (!img || !img.naturalWidth) return;
    const dh = dataHeight();
    let deg = 30;
    if (scale >= 1.4) deg = 15;
    if (scale >= 2.8) deg = 10;
    if (scale >= 5.5) deg = 5;
    if (scale >= 11) deg = 2;
    if (scale >= 20) deg = 1;
    const lonPx = Math.max(12, img.naturalWidth * scale * deg / 360.0);
    const latPx = Math.max(12, dh * scale * deg / 180.0);
    document.querySelectorAll('.map-grid-layer').forEach(el => {{
      el.style.backgroundSize = lonPx.toFixed(1) + 'px ' + latPx.toFixed(1) + 'px';
    }});
  }}
  function apply() {{
    if (!img) return;
    if (!img.dataset.baseW && img.naturalWidth) {{ img.dataset.baseW = img.naturalWidth; img.dataset.baseH = img.naturalHeight; }}
    const bw = parseFloat(img.dataset.baseW || img.naturalWidth || 1);
    const bh = parseFloat(img.dataset.baseH || img.naturalHeight || 1);
    const w = bw * scale;
    const h = bh * scale;
    img.style.width = w.toFixed(1) + 'px';
    img.style.height = h.toFixed(1) + 'px';
    if (stack) {{ stack.style.width = w.toFixed(1) + 'px'; stack.style.height = h.toFixed(1) + 'px'; }}
    const dh = dataHeight() * scale;
    document.querySelectorAll('.single-contour,.single-grid').forEach(el => {{
      el.style.width = w.toFixed(1) + 'px';
      el.style.height = dh.toFixed(1) + 'px';
    }});
    updateGridDensity();
    updateContourDensity();
  }}
  function zoomBy(factor, cx, cy) {{
    if (!box || !img) return;
    const old = scale;
    scale = Math.max(0.08, Math.min(32, scale * factor));
    const ratio = scale / old;
    const px = (cx === undefined ? box.clientWidth / 2 : cx);
    const py = (cy === undefined ? box.clientHeight / 2 : cy);
    box.scrollLeft = (box.scrollLeft + px) * ratio - px;
    box.scrollTop = (box.scrollTop + py) * ratio - py;
    apply();
    updateScaleReadout(null);
  }}
  function updateScaleReadout(data) {{
    if (!scaleReadout) return;
    let text = 'Zoom ' + scale.toFixed(2) + '×';
    if (data && data.ew_km_per_cell && data.ns_km_per_cell) {{
      text += ' · at cursor: one simulation cell ≈ ' + data.ew_km_per_cell + ' km E-W × ' + data.ns_km_per_cell + ' km N-S';
      text += ' · screen pixel ≈ ' + (data.ew_km_per_cell / scale).toFixed(3) + ' km E-W × ' + (data.ns_km_per_cell / scale).toFixed(3) + ' km N-S';
      text += ' · lat ' + data.lat + '°, lon ' + data.lon + '°';
    }} else {{
      text += ' · real-size scale appears when hovering over a full-world map with saved planet state';
    }}
    scaleReadout.textContent = text;
  }}
  window.wgZoomIn = function() {{ zoomBy(1.18); }};
  window.wgZoomOut = function() {{ zoomBy(1/1.18); }};
  window.wgZoomReset = function() {{ scale = 1.0; apply(); if (box) {{ box.scrollLeft = 0; box.scrollTop = 0; }} updateScaleReadout(null); }};
  window.wgToggleGrid = function() {{ if (box) box.classList.toggle('demarcation'); updateGridDensity(); }};
  window.wgSetMonthlyClimate = function(month) {{
    const ctx = window.WG_MONTHLY_CONTEXT;
    if (!ctx || !ctx.urls) return;
    const key = String(parseInt(month, 10));
    const item = ctx.urls[key];
    if (!item || !img) return;
    ctx.currentMonth = parseInt(key, 10);
    const select = document.getElementById('monthly-sequence-select');
    const range = document.getElementById('monthly-sequence-range');
    const readout = document.getElementById('monthly-sequence-readout');
    const label = document.getElementById('map-label-code');
    if (select) select.value = key;
    if (range) range.value = key;
    if (readout) readout.textContent = 'Current: ' + key.padStart(2, '0') + ' / ' + (item.month_label || '');
    if (label) label.textContent = item.label || label.textContent;
    img.dataset.path = item.path || img.dataset.path || '';
    img.src = item.raw_url;
    img.addEventListener('load', function once() {{ img.removeEventListener('load', once); apply(); }}, {{ once:true }});
    if (history && item.view_url) history.replaceState(null, '', item.view_url);
    updateScaleReadout(null);
  }};
  window.wgMonthlyStep = function(delta) {{
    const ctx = window.WG_MONTHLY_CONTEXT;
    if (!ctx) return;
    const current = parseInt(ctx.currentMonth || 1, 10);
    const next = ((current - 1 + delta + 12) % 12) + 1;
    window.wgSetMonthlyClimate(next);
  }};
  window.wgMonthlyTogglePlay = function() {{
    const btn = document.getElementById('monthly-play-button');
    if (window.WG_MONTHLY_TIMER) {{
      clearInterval(window.WG_MONTHLY_TIMER);
      window.WG_MONTHLY_TIMER = null;
      if (btn) btn.textContent = 'Play';
      return;
    }}
    const speed = document.getElementById('monthly-play-speed');
    const delay = Math.max(180, parseInt(speed ? speed.value : '800', 10) || 800);
    window.WG_MONTHLY_TIMER = setInterval(function() {{ window.wgMonthlyStep(1); }}, delay);
    if (btn) btn.textContent = 'Pause';
  }};
  function requestInfo(e) {{
    if (!img || !readout) return;
    const rect = img.getBoundingClientRect();
    const x = (e.clientX - rect.left) / Math.max(0.0001, rect.width) * (img.naturalWidth || rect.width);
    const y = (e.clientY - rect.top) / Math.max(0.0001, rect.height) * (img.naturalHeight || rect.height);
    if (x < 0 || y < 0 || x > (img.naturalWidth || rect.width) || y > (img.naturalHeight || rect.height)) return;
    clearTimeout(hoverTimer);
    hoverTimer = setTimeout(() => {{
      const url = '/map-info?output_dir=' + encodeURIComponent(img.dataset.outputDir || '') + '&path=' + encodeURIComponent(img.dataset.path || '') + '&x=' + encodeURIComponent(x.toFixed(1)) + '&y=' + encodeURIComponent(y.toFixed(1));
      fetch(url).then(r => r.json()).then(data => {{
        const esc = (s) => String(s || '').replace(/[&<>"']/g, ch => ({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[ch]));
        if (data.active_value) {{
          const ctx = Array.isArray(data.context) ? data.context.map(esc).join(' · ') : esc(data.text || '');
          readout.innerHTML = '<div class="hover-active-value">' + esc(data.active_value) + '</div><div class="hover-context-value">' + ctx + '</div>';
        }} else {{
          readout.textContent = data.text || '';
        }}
        updateScaleReadout(data);
      }}).catch(() => {{}});
    }}, 70);
  }}
  if (box) {{
    let sx = 0, sy = 0, sl = 0, st = 0;
    box.addEventListener('mousedown', function(e) {{ dragging = true; box.classList.add('dragging'); sx = e.clientX; sy = e.clientY; sl = box.scrollLeft; st = box.scrollTop; }});
    window.addEventListener('mouseup', function() {{ dragging = false; if (box) box.classList.remove('dragging'); }});
    window.addEventListener('mousemove', function(e) {{ if (!dragging) return; box.scrollLeft = sl - (e.clientX - sx); box.scrollTop = st - (e.clientY - sy); }});
    box.addEventListener('mousemove', requestInfo);
    box.addEventListener('wheel', function(e) {{ e.preventDefault(); const factor = Math.exp(-e.deltaY * 0.0015); zoomBy(factor, e.clientX - box.getBoundingClientRect().left, e.clientY - box.getBoundingClientRect().top); }}, {{ passive:false }});
  }}
  document.querySelectorAll('.legend-item').forEach(function(item) {{
    item.addEventListener('click', function() {{
      const active = item.classList.contains('legend-active');
      document.querySelectorAll('.legend-item.legend-active').forEach(function(el) {{ el.classList.remove('legend-active'); }});
      if (!active) item.classList.add('legend-active');
      if (readout) {{
        const label = (item.dataset.legendLabel || item.textContent || '').trim();
        readout.textContent = active ? 'Legend focus cleared. Hover the map for cell values.' : 'Legend focus: ' + label + ' — hover the map for the active value and shared context.';
      }}
    }});
  }});
  if (img && img.complete) apply(); else if (img) img.addEventListener('load', apply);
  updateScaleReadout(null);
}})();
</script>
"""


    def _page_map_index(self, params: dict[str, list[str]]) -> str:
        output_raw = _first(params, "output_dir")
        stage_filter = _first(params, "stage", "").strip() or None
        show_hidden = _first(params, "show_hidden", "").strip().lower() in {"1", "true", "yes", "on"}
        if not output_raw:
            return self._page_shell("Map index", "<h1>Map index</h1><p class='bad'>No run was selected.</p>")
        output_dir = Path(output_raw).expanduser()
        if not output_dir.exists():
            return self._page_shell("Map index", f"<h1>Map index</h1><p class='bad'>Run folder does not exist: <code>{_safe_text(output_dir)}</code></p>")
        options = ''.join(f"<option value='{_safe_text(st)}'{' selected' if stage_filter == st else ''}>{_safe_text(st)}</option>" for st in ["solar-system", "terrain-full", "climate", "hydrology", "biomes", "regions"])
        body = f"""
<h1>Registered map index</h1>
<p><code>{_safe_text(output_dir)}</code></p>
<div class='row sticky-tools'>
  <a class='button secondary' href='/pipeline?output_dir={urllib.parse.quote(str(output_dir))}'>Back to Pipeline</a>
  <a class='button secondary' href='/run?output_dir={urllib.parse.quote(str(output_dir))}'>Run tools</a>
  <a class='button secondary' href='/files?output_dir={urllib.parse.quote(str(output_dir))}'>Generated files</a>
</div>
<section class='card'>
  <h2>Map registry status</h2>
  <p class='help'>This index is based on the run-independent registry plus the current run folder. It helps prevent map outputs from becoming orphaned in the codebase. Each row shows whether the map was generated, is missing, or does not apply to this run/configuration.</p>
  {_map_index_summary_html(output_dir)}
  <form method='get' action='/map-index' class='row' style='margin-top:10px'>
    <input type='hidden' name='output_dir' value='{_safe_text(str(output_dir))}'>
    <label class='small'>Stage filter
      <select name='stage' style='width:180px'>
        <option value=''>all registered maps</option>
        {options}
      </select>
    </label>
    <label class='inline-check small'><input type='checkbox' name='show_hidden' value='1' {'checked' if show_hidden else ''} style='width:auto'> show hidden/deprecated maps</label>
    <button type='submit'>Apply filter</button>
  </form>
  {_map_index_table_html(output_dir, stage_filter=stage_filter, show_hidden=show_hidden)}
</section>
"""
        return self._page_shell("Map index", body)


    def _page_map(self, params: dict[str, list[str]]) -> str:
        raw_path = _first(params, "path")
        output_raw = _first(params, "output_dir")
        output_dir = Path(output_raw).expanduser() if output_raw else None
        path = Path(raw_path).expanduser()
        if not raw_path or not path.exists() or not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            return self._page_shell("Map not found", f"<h1 class='bad'>Map not found</h1><p><code>{_safe_text(raw_path)}</code></p>")
        label = _rel_label(path, output_dir) if output_dir else path.name
        display_title = _map_display_title(path, output_dir)
        planet_name = _planet_name_for_run(output_dir)
        raw_url = _file_url(path)
        scale_info = _map_scale_summary(path, output_dir)
        back = f"<a class='button secondary' href='/pipeline?output_dir={urllib.parse.quote(str(output_dir))}'>Back to pipeline</a>" if output_dir else ""
        compare = f"<a class='button secondary' href='/compare?{urllib.parse.urlencode({'output_dir': str(output_dir), 'left': label})}'>Compare this map</a>" if output_dir else ""
        globe = f"<a class='button secondary' href='/globe?{urllib.parse.urlencode({'output_dir': str(output_dir), 'path': str(path)})}'>View as globe</a>" if output_dir and _is_probable_world_map(path) else ""
        legend = _map_context_sidebar_html(path, output_dir) + _legend_html(path) + _map_area_stats_html(path, output_dir)
        explanation = _map_explanation_html(path, output_dir)
        output_attr = _safe_text(str(output_dir) if output_dir else "")
        contour = _first(params, "contour", "none")
        if contour not in {"none", "elevation", "temperature", "precipitation"}:
            contour = "none"
        contour_levels = _first(params, "contour_levels", "auto")
        if contour_levels != "auto":
            try:
                contour_levels = str(max(4, min(40, int(contour_levels))))
            except Exception:
                contour_levels = "auto"
        def contour_selected(value: str) -> str:
            return " selected" if contour == value else ""
        def contour_level_selected(value: str) -> str:
            return " selected" if contour_levels == value else ""
        geom = _map_data_geometry(output_dir, path)
        ww = int(geom.get("world_width") or 0)
        wh = int(geom.get("world_height") or 0)
        data_attrs = f'data-output-dir="{output_attr}" data-world-width="{ww}" data-world-height="{wh}"'
        monthly_controls = _monthly_sequence_controls(path, output_dir)
        monthly_overlay = _monthly_viewer_overlay(path, output_dir)
        contour_url = ""
        contour_img = ""
        if contour != "none" and output_dir is not None:
            contour_url = "/contour-overlay?" + urllib.parse.urlencode({"output_dir": str(output_dir), "kind": contour, "max_dim": "2048", "levels": contour_levels, "zoom": "1"})
            contour_img = f'<img class="contour-layer single-contour" src="{contour_url}" data-base-src="{contour_url}" alt="{_safe_text(_contour_kind_label(contour))}">'
        body = f"""
<h1>{_safe_text(display_title)}</h1>
<p><code id="map-label-code">{_safe_text(label)}</code></p>
<p class="muted small">{('Planet: ' + _safe_text(planet_name) + ' · ') if planet_name else ''}{_safe_text(scale_info)}</p>
{monthly_controls}
<form method="get" action="/map" class="sticky-tools row">
  <input type="hidden" name="output_dir" value="{_safe_text(str(output_dir) if output_dir else '')}">
  <input type="hidden" name="path" value="{_safe_text(str(path))}">
  {back}
  {compare}
  {globe}
  <a class="button secondary" href="{raw_url}" target="_blank">Open raw image</a>
  <button type="button" onclick="wgZoomIn()">Zoom in</button>
  <button type="button" onclick="wgZoomOut()">Zoom out</button>
  <button type="button" onclick="wgZoomReset()">Reset</button>
  <button type="button" onclick="wgToggleGrid()">Toggle grid</button>
  <label class="small">Contour <select name="contour" onchange="this.form.submit()" style="width:160px">
    <option value="none"{contour_selected('none')}>None</option>
    <option value="elevation"{contour_selected('elevation')}>Elevation</option>
    <option value="temperature"{contour_selected('temperature')}>Temperature</option>
    <option value="precipitation"{contour_selected('precipitation')}>Rainfall</option>
  </select></label>
  <label class="small">Contour density <select name="contour_levels" onchange="this.form.submit()" style="width:150px">
    <option value="auto"{contour_level_selected('auto')}>Auto with zoom</option>
    <option value="6"{contour_level_selected('6')}>Sparse</option>
    <option value="10"{contour_level_selected('10')}>Normal</option>
    <option value="16"{contour_level_selected('16')}>Dense</option>
    <option value="24"{contour_level_selected('24')}>Very dense</option>
  </select></label>
  <span class="muted small">Mouse wheel zooms. Drag pans. Crosshair cursor keeps pointing precise for hover info.</span>
</form>
<div class="viewer-shell">
  <div>
    <div id="single-viewer" class="viewer-wrap">{monthly_overlay}<div id="single-stack" class="viewer-map-stage" {data_attrs}><img id="zoom-img" class="zoom-image" src="{raw_url}" alt="{_safe_text(label)}" data-output-dir="{output_attr}" data-path="{_safe_text(str(path))}">{contour_img}<div class="map-grid-layer single-grid"></div></div></div>
    <div id="map-hover-readout" class="hover-readout">Hover over the map for pixel/cell info.</div>
    <div id="map-scale-readout" class="scale-readout">Scale will update as you move across the map.</div>
  </div>
  <aside class="legend-panel">{legend}</aside>
</div>
{explanation}
{self._zoom_script('zoom-img')}
"""
        return self._page_shell("Map viewer", body)


    def _page_globe(self, params: dict[str, list[str]]) -> str:
        raw_path = _first(params, "path")
        output_raw = _first(params, "output_dir")
        output_dir = Path(output_raw).expanduser() if output_raw else None
        path = Path(raw_path).expanduser()
        if not raw_path or not path.exists() or not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            return self._page_shell("Globe not found", f"<h1 class='bad'>Map not found</h1><p><code>{_safe_text(raw_path)}</code></p>")
        label = _rel_label(path, output_dir) if output_dir else path.name
        texture_url = _map_data_url(path, output_dir) if output_dir else _file_url(path)
        back = f"<a class='button secondary' href='/pipeline?output_dir={urllib.parse.quote(str(output_dir))}'>Back to pipeline</a>" if output_dir else ""
        map_link = f"<a class='button secondary' href='{_map_view_url(path, output_dir)}'>Flat map viewer</a>"
        compare_link = f"<a class='button secondary' href='/compare?{urllib.parse.urlencode({'output_dir': str(output_dir), 'left': label})}'>Compare this map</a>" if output_dir else ""
        scale_info = _map_scale_summary(path, output_dir)
        tilt_degrees = _axial_tilt_degrees_for_run(output_dir) if output_dir else 0.0
        map_info_base = "/map-info?" + urllib.parse.urlencode({"output_dir": str(output_dir) if output_dir else "", "path": str(path)})
        map_focus = _map_focus_label(path)
        body = f"""
<h1>3D globe viewer</h1>
<p><code>{_safe_text(label)}</code></p>
<p class="muted small">{_safe_text(scale_info)}</p>
<div class="row sticky-tools">
  {back}
  {map_link}
  {compare_link}
  <button type="button" onclick="wgGlobeToggleSpin()">Toggle spin</button>
  <button type="button" onclick="wgGlobeResetStraight()">Realign straight</button>
  <button type="button" onclick="wgGlobeResetTilted()">Realign tilted</button>
  <button type="button" onclick="wgGlobeSaveView()">Open current view PNG</button>
  <label class="small">Zoom <input id="globe-zoom" type="range" min="0.55" max="2.8" step="0.01" value="1" style="width:140px"></label>
  <label class="small">Light <input id="globe-light" type="range" min="0" max="1" step="0.01" value="0.35" style="width:120px"></label>
  <label class="small">Spin speed <input id="globe-spin-speed" type="range" min="0" max="3" step="0.05" value="1" style="width:120px"></label>
  <label class="small"><input id="globe-guides" type="checkbox" checked> Equator/tropics/polar circles</label>
  <label class="small"><input id="globe-smooth" type="checkbox" checked> Smooth texture</label>
  <label class="small"><input id="globe-seam-feather" type="checkbox" checked> Feather seam</label>
  <label class="small">Drag mode <select id="globe-drag-mode"><option value="grab" selected>grab surface</option><option value="camera">orbit camera</option></select></label>
  <label class="small"><input id="globe-invert-x" type="checkbox"> invert X</label>
  <label class="small"><input id="globe-invert-y" type="checkbox"> invert Y</label>
</div>
<div class="globe-wrap">
  <section class="globe-stage card"><canvas id="globe-canvas" width="1100" height="760"></canvas></section>
  <aside class="card globe-details">
    <h2>Globe controls</h2>
    <p class="help">This is a browser-side pseudo-3D globe made from the flat equirectangular map. Drag to rotate, use the mouse wheel or zoom slider to zoom, and hover to inspect the underlying cell details. Default drag mode is <b>grab surface</b>: the part of the globe under your cursor should move in the same direction as your drag.</p>
    <div id="globe-readout" class="hover-readout globe-hover-card">Hover over the globe for latitude/longitude and map details.</div>
    <p class="help small"><b>Drag check:</b> in default grab mode, dragging upward should pull the visible surface upward; dragging right should pull the visible surface right. Use the invert toggles only if your input device feels reversed.</p>
    <h3>Notes</h3>
    <ul class="muted small">
      <li>Uses clean map-only rasters when available; legacy embedded legend bands are cropped out where possible.</li>
      <li>Best with full-world maps. Regional maps and solar-system diagrams are not geographically meaningful as globes.</li>
      <li>Use Realign straight for map-style orientation, or Realign tilted to view the globe using the saved axial tilt ({tilt_degrees:.1f}°).</li>
      <li>The tropics and polar circles use the saved axial tilt when available; if the tilt is unknown, Earth-like 23.44° guides are used.</li>
      <li>Keyboard: Space toggles spin, 0 realigns straight, T realigns tilted.</li>
      <li>Later we can add true elevation displacement; this version is a visual texture globe.</li>
    </ul>
    <h3>Legend</h3>
    {_legend_html(path)}
  </aside>
</div>
<img id="globe-texture" src="{texture_url}" alt="texture" style="display:none">
<script>
(function() {{
  const canvas = document.getElementById('globe-canvas');
  const ctx = canvas.getContext('2d');
  const img = document.getElementById('globe-texture');
  const readout = document.getElementById('globe-readout');
  const zoomInput = document.getElementById('globe-zoom');
  const lightInput = document.getElementById('globe-light');
  const spinSpeedInput = document.getElementById('globe-spin-speed');
  const guidesInput = document.getElementById('globe-guides');
  const smoothInput = document.getElementById('globe-smooth');
  const seamFeatherInput = document.getElementById('globe-seam-feather');
  const dragModeInput = document.getElementById('globe-drag-mode');
  const invertXInput = document.getElementById('globe-invert-x');
  const invertYInput = document.getElementById('globe-invert-y');
  const mapInfoBase = {json.dumps(map_info_base)};
  const mapFocus = {json.dumps(map_focus)};
  const axialTiltDegrees = {tilt_degrees:.6f};
  let yaw = -0.55, pitch = 0.18, zoom = 1.0, spinning = true;
  let dragging = false, lastX = 0, lastY = 0;
  let hoverTimer = null, lastHover = null, lastFrameTime = performance.now();
  let texCanvas = null, texPixels = null, texWidth = 0, texHeight = 0;
  const out = document.createElement('canvas');
  const octx = out.getContext('2d');
  const guideTilt = Math.max(0, Math.min(89.0, Math.abs(axialTiltDegrees || 23.44) || 23.44));
  const polarGuideLat = Math.max(0.5, 90 - guideTilt);
  function resize() {{
    const box = canvas.parentElement.getBoundingClientRect();
    const targetW = Math.max(520, Math.min(1500, Math.floor(box.width - 12)));
    const targetH = Math.max(500, Math.min(1100, Math.floor(box.height - 12)));
    if (canvas.width !== targetW || canvas.height !== targetH) {{
      canvas.width = targetW; canvas.height = targetH;
      out.width = targetW; out.height = targetH;
    }}
  }}
  function ensureTexture() {{
    if (texPixels || !img.complete || !img.naturalWidth) return;
    texCanvas = document.createElement('canvas');
    texCanvas.width = img.naturalWidth; texCanvas.height = img.naturalHeight;
    const tctx = texCanvas.getContext('2d');
    tctx.drawImage(img,0,0);
    texWidth = texCanvas.width; texHeight = texCanvas.height;
    const idata = tctx.getImageData(0,0,texWidth,texHeight);
    texPixels = idata.data;
    // Update 24: optional texture seam feathering for the pseudo-3D globe.
    // The terrain generator should still be seam-safe, but many generated PNGs
    // have small first/last-column differences. Bilinear wrapping then displays
    // a visible meridian seam. Feather only a few edge columns and only in the
    // browser-side texture copy; the source map file is unchanged.
    if (seamFeatherInput && seamFeatherInput.checked && texWidth > 12) {{
      const feather = Math.min(5, Math.floor(texWidth / 160));
      for (let y=0; y<texHeight; y++) {{
        for (let k=0; k<feather; k++) {{
          const wl = (feather-k)/(feather+1);
          const wr = (k+1)/(feather+1);
          const li = (y*texWidth + k)*4;
          const ri = (y*texWidth + (texWidth-1-k))*4;
          for (let c=0; c<4; c++) {{
            const left = texPixels[li+c], right = texPixels[ri+c];
            const avg = (left + right) * 0.5;
            texPixels[li+c] = left*(1-wl) + avg*wl;
            texPixels[ri+c] = right*(1-wr) + avg*wr;
          }}
        }}
      }}
      tctx.putImageData(idata,0,0);
    }}
  }}
  function texel(ix, iy) {{
    ix = ((ix % texWidth) + texWidth) % texWidth;
    iy = Math.max(0, Math.min(texHeight-1, iy));
    const ti = (iy*texWidth + ix)*4;
    return [texPixels[ti], texPixels[ti+1], texPixels[ti+2], texPixels[ti+3]];
  }}
  function sampleTexture(u, v) {{
    if (!smoothInput.checked) return texel(Math.floor(u), Math.floor(v));
    const x0 = Math.floor(u), y0 = Math.floor(v);
    const fx = u - x0, fy = v - y0;
    const a = texel(x0, y0), b = texel(x0+1, y0), c = texel(x0, y0+1), d = texel(x0+1, y0+1);
    const outp = [0,0,0,0];
    for (let i=0;i<4;i++) outp[i] = a[i]*(1-fx)*(1-fy) + b[i]*fx*(1-fy) + c[i]*(1-fx)*fy + d[i]*fx*fy;
    return outp;
  }}
  function draw() {{
    if (!img.complete || !img.naturalWidth) return requestAnimationFrame(draw);
    ensureTexture();
    resize();
    const w = canvas.width, h = canvas.height;
    const cx = w/2, cy = h/2, r = Math.min(w,h) * 0.43 * zoom;
    const imageData = octx.createImageData(w,h);
    const data = imageData.data;
    const sinY = Math.sin(yaw), cosY = Math.cos(yaw), sinP = Math.sin(pitch), cosP = Math.cos(pitch);
    const ambient = parseFloat(lightInput.value || '0.35');
    for (let py=0; py<h; py++) {{
      const dy = (py - cy) / r;
      for (let px=0; px<w; px++) {{
        const dx = (px - cx) / r;
        const rr = dx*dx + dy*dy;
        const idx = (py*w + px)*4;
        if (rr > 1) {{ data[idx+3] = 0; continue; }}
        let z = Math.sqrt(1 - rr);
        // Canvas Y increases downward, but planet latitude increases northward.
        // Use a north-up local Y axis so maps and guide latitudes are not flipped.
        let x = dx, y = -dy;
        // inverse pitch then yaw
        let y1 = y*cosP + z*sinP;
        let z1 = -y*sinP + z*cosP;
        let x2 = x*cosY - z1*sinY;
        let z2 = x*sinY + z1*cosY;
        const lon = Math.atan2(x2, z2);
        const lat = Math.asin(Math.max(-1, Math.min(1, y1)));
        let u = (lon / (2*Math.PI) + 0.5) * texWidth;
        let v = (0.5 - lat / Math.PI) * texHeight;
        u = ((u % texWidth) + texWidth) % texWidth;
        v = Math.max(0, Math.min(texHeight-1, v));
        const pxl = sampleTexture(u, v);
        const shade = Math.max(ambient, Math.min(1, ambient + 0.65 * (0.35*x + -0.25*y + 0.9*z)));
        data[idx] = pxl[0] * shade;
        data[idx+1] = pxl[1] * shade;
        data[idx+2] = pxl[2] * shade;
        data[idx+3] = pxl[3];
      }}
    }}
    octx.putImageData(imageData,0,0);
    ctx.clearRect(0,0,w,h);
    ctx.fillStyle = '#020617'; ctx.fillRect(0,0,w,h);
    ctx.save();
    ctx.shadowColor = 'rgba(56,189,248,.35)'; ctx.shadowBlur = 24;
    ctx.drawImage(out,0,0);
    ctx.restore();
    ctx.beginPath(); ctx.arc(cx,cy,r,0,Math.PI*2); ctx.strokeStyle='rgba(226,232,240,.65)'; ctx.lineWidth=1.4; ctx.stroke();
    if (guidesInput.checked) drawGuideLines(cx, cy, r);
    const now = performance.now();
    const dt = Math.max(0.25, Math.min(3.0, (now - lastFrameTime) / 16.6667));
    lastFrameTime = now;
    if (spinning && !dragging) yaw += 0.0025 * parseFloat(spinSpeedInput.value || '1') * dt;
    requestAnimationFrame(draw);
  }}
  function projectLatLon(latDeg, lonDeg, cx, cy, r) {{
    const lat = latDeg * Math.PI/180, lon = lonDeg * Math.PI/180;
    const x2 = Math.sin(lon) * Math.cos(lat);
    const y1 = Math.sin(lat);
    const z2 = Math.cos(lon) * Math.cos(lat);
    const sinY=Math.sin(yaw), cosY=Math.cos(yaw), sinP=Math.sin(pitch), cosP=Math.cos(pitch);
    const x = x2*cosY + z2*sinY;
    const z1 = -x2*sinY + z2*cosY;
    const y = y1*cosP - z1*sinP;
    const z = y1*sinP + z1*cosP;
    if (z <= 0) return null;
    return [cx + x*r, cy - y*r];
  }}
  function drawLatitude(latDeg, color, width, label) {{
    ctx.beginPath();
    let started = false;
    for (let lon=-180; lon<=180; lon+=2) {{
      const p = projectLatLon(latDeg, lon, canvas.width/2, canvas.height/2, Math.min(canvas.width, canvas.height)*0.43*zoom);
      if (!p) {{ started=false; continue; }}
      if (!started) {{ ctx.moveTo(p[0], p[1]); started=true; }} else ctx.lineTo(p[0], p[1]);
    }}
    ctx.strokeStyle = color; ctx.lineWidth = width; ctx.stroke();
    const lp = projectLatLon(latDeg, 5, canvas.width/2, canvas.height/2, Math.min(canvas.width, canvas.height)*0.43*zoom);
    if (lp && label) {{ ctx.fillStyle=color; ctx.font='12px sans-serif'; ctx.fillText(label, lp[0]+6, lp[1]-4); }}
  }}
  function drawGuideLines(cx, cy, r) {{
    drawLatitude(0, 'rgba(250,204,21,.9)', 1.8, 'equator');
    drawLatitude(guideTilt, 'rgba(34,197,94,.75)', 1.1, 'N tropic');
    drawLatitude(-guideTilt, 'rgba(34,197,94,.75)', 1.1, 'S tropic');
    drawLatitude(polarGuideLat, 'rgba(125,211,252,.75)', 1.1, 'N polar circle');
    drawLatitude(-polarGuideLat, 'rgba(125,211,252,.75)', 1.1, 'S polar circle');
  }}
  function eventLatLon(ev) {{
    const rect = canvas.getBoundingClientRect();
    const w = canvas.width, h = canvas.height;
    const cx = w/2, cy = h/2, r = Math.min(w,h) * 0.43 * zoom;
    const px = (ev.clientX - rect.left) * w / rect.width;
    const py = (ev.clientY - rect.top) * h / rect.height;
    const dx = (px - cx) / r, dy = (py - cy) / r;
    if (dx*dx + dy*dy > 1) return null;
    let z = Math.sqrt(1 - dx*dx - dy*dy);
    let x=dx, y=-dy;
    const sinY=Math.sin(yaw), cosY=Math.cos(yaw), sinP=Math.sin(pitch), cosP=Math.cos(pitch);
    let y1 = y*cosP + z*sinP;
    let z1 = -y*sinP + z*cosP;
    let x2 = x*cosY - z1*sinY;
    let z2 = x*sinY + z1*cosY;
    return {{ lat: Math.asin(y1)*180/Math.PI, lon: Math.atan2(x2,z2)*180/Math.PI }};
  }}
  function esc(s) {{
    return String(s ?? '').replace(/[&<>"']/g, ch => ({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[ch]));
  }}
  function classifyDetail(item) {{
    const t = item.toLowerCase();
    if (t.startsWith('cell r')) return ['Cell', item];
    if (t.startsWith('active map')) return ['Active map value', item.replace(/^active map\s*/,'')];
    if (t.startsWith('pixel')) return ['Image pixel', item.replace(/^pixel\\s*/,'')];
    if (t.startsWith('color')) return ['Sample color', item.replace(/^color\\s*/,'')];
    if (t.includes('cell scale')) return ['Cell scale', item.replace(/^cell scale\\s*≈?\\s*/,'')];
    if (t.startsWith('elevation')) return ['Elevation', item.replace(/^elevation\\s*/,'')];
    if (t === 'land' || t === 'water') return ['Surface', item];
    if (t.startsWith('plate')) return ['Plate', item.replace(/^plate\\s*/,'')];
    if (t.startsWith('boundary')) return ['Boundary', item.replace(/^boundary class\\s*/,'class ')];
    if (t.startsWith('crust')) return ['Crust', item.replace(/^crust\\s*/,'')];
    if (t.startsWith('temp')) return ['Temperature', item.replace(/^temp\\s*/,'')];
    if (t.startsWith('rain')) return ['Rainfall', item.replace(/^rain\\s*/,'')];
    if (t.startsWith('köppen') || t.startsWith('koppen')) return ['Köppen', item.replace(/^k[öo]ppen\\s*/i,'')];
    if (t.startsWith('biome')) return ['Biome', item.replace(/^biome\\s*/,'')];
    if (t.startsWith('runoff')) return ['Runoff', item.replace(/^runoff\\s*/,'')];
    if (t.startsWith('flow')) return ['Flow accumulation', item.replace(/^flow\\s*/,'')];
    if (t.startsWith('basin')) return ['Drainage basin', item.replace(/^basin\\s*/,'')];
    return ['', item];
  }}
  function isFocusRow(label) {{
    const f = mapFocus.toLowerCase(), l = label.toLowerCase();
    return (f.includes('elevation') && l.includes('elevation')) ||
           (f.includes('rain') && l.includes('rain')) ||
           (f.includes('temperature') && l.includes('temperature')) ||
           (f.includes('biome') && l.includes('biome')) ||
           (f.includes('köppen') && l.includes('köppen')) ||
           (f.includes('hydrology') && (l.includes('flow') || l.includes('basin') || l.includes('runoff'))) ||
           (l.includes('active map')) ||
           (f.includes('tectonics') && (l.includes('plate') || l.includes('crust') || l.includes('boundary')));
  }}
  function formatGlobeReadout(ll, data) {{
    const text = data && data.text ? data.text : '';
    const items = text.split(' · ').filter(Boolean);
    const rows = [];
    rows.push(['Latitude / longitude', ll.lat.toFixed(2) + '°, ' + ll.lon.toFixed(2) + '°', true]);
    for (const item of items) {{
      const lower = item.toLowerCase();
      if (lower.includes('°n') || lower.includes('°s') || lower.includes('°e') || lower.includes('°w')) continue;
      const pair = classifyDetail(item);
      if (!pair[0]) continue;
      rows.push([pair[0], pair[1], isFocusRow(pair[0])]);
    }}
    const table = rows.map(r => '<tr class="' + (r[2] ? 'focus-row' : '') + '"><th>' + esc(r[0]) + '</th><td>' + esc(r[1]) + '</td></tr>').join('');
    return '<div class="globe-hover-head"><b>' + esc(mapFocus) + '</b><span>Focused values are highlighted. Other available state values are shown below.</span></div><table class="globe-hover-table">' + table + '</table>';
  }}
  function updateHover(ev) {{
    if (dragging) return;
    const ll = eventLatLon(ev);
    if (!ll) {{ return; }}
    lastHover = ll;
    const imgX = ((ll.lon + 180) / 360) * (img.naturalWidth || 1);
    const imgY = ((90 - ll.lat) / 180) * (img.naturalHeight || 1);
    clearTimeout(hoverTimer);
    hoverTimer = setTimeout(() => {{
      const url = mapInfoBase + '&x=' + encodeURIComponent(imgX.toFixed(1)) + '&y=' + encodeURIComponent(imgY.toFixed(1));
      fetch(url).then(r => r.json()).then(data => {{
        readout.innerHTML = formatGlobeReadout(ll, data);
      }}).catch(() => {{ readout.innerHTML = formatGlobeReadout(ll, {{text:''}}); }});
    }}, 90);
  }}
  canvas.addEventListener('mousedown', ev => {{ dragging=true; canvas.classList.add('dragging'); lastX=ev.clientX; lastY=ev.clientY; }});
  window.addEventListener('mouseup', () => {{ dragging=false; canvas.classList.remove('dragging'); }});
  window.addEventListener('mousemove', ev => {{
    if (dragging) {{
      const dx = ev.clientX - lastX;
      const dy = ev.clientY - lastY;
      const mode = dragModeInput ? dragModeInput.value : 'grab';
      // In grab-surface mode, visible map features follow the cursor.
      // In camera/orbit mode, controls behave like rotating a camera around the globe.
      let yawDelta = dx * 0.006;
      let pitchDelta = dy * 0.006;
      if (mode === 'camera') {{ yawDelta *= -1; pitchDelta *= -1; }}
      if (invertXInput && invertXInput.checked) yawDelta *= -1;
      if (invertYInput && invertYInput.checked) pitchDelta *= -1;
      yaw += yawDelta;
      pitch += pitchDelta;
      pitch = Math.max(-1.25, Math.min(1.25, pitch));
      lastX=ev.clientX; lastY=ev.clientY;
    }} else updateHover(ev);
  }});
  canvas.addEventListener('mousemove', updateHover);
  canvas.addEventListener('mouseleave', () => {{ clearTimeout(hoverTimer); }});
  canvas.addEventListener('wheel', ev => {{ ev.preventDefault(); zoom = Math.max(.55, Math.min(2.8, zoom * Math.exp(-ev.deltaY * 0.001))); zoomInput.value = zoom.toFixed(2); }}, {{passive:false}});
  zoomInput.addEventListener('input', () => zoom = parseFloat(zoomInput.value || '1'));
  if (seamFeatherInput) seamFeatherInput.addEventListener('change', () => {{ texPixels=null; texCanvas=null; }});
  window.wgGlobeToggleSpin = function() {{ spinning = !spinning; }};
  window.wgGlobeResetStraight = function() {{ yaw=0; pitch=0; zoom=1; zoomInput.value='1'; }};
  window.wgGlobeResetTilted = function() {{ yaw=0; pitch=Math.max(-80, Math.min(80, axialTiltDegrees)) * Math.PI/180; zoom=1; zoomInput.value='1'; }};
  window.wgGlobeReset = window.wgGlobeResetStraight;
  window.wgGlobeSaveView = function() {{ const url = canvas.toDataURL('image/png'); const w = window.open(''); if (w) w.document.write('<title>WorldGen globe view</title><img src="' + url + '" style="max-width:100%">'); }};
  window.addEventListener('keydown', ev => {{
    if (ev.target && ['INPUT','TEXTAREA','SELECT'].includes(ev.target.tagName)) return;
    if (ev.code === 'Space') {{ ev.preventDefault(); wgGlobeToggleSpin(); }}
    if (ev.key === '0') wgGlobeResetStraight();
    if (ev.key.toLowerCase() === 't') wgGlobeResetTilted();
  }});
  img.onload = draw; if (img.complete) draw();
}})();
</script>
"""
        return self._page_shell("Globe", body)


    def _page_compare(self, params: dict[str, list[str]]) -> str:
        output_dir = Path(_first(params, "output_dir")).expanduser()
        remember_run(output_dir)
        left = self._resolve_run_map(output_dir, _first(params, "left"))
        right = self._resolve_run_map(output_dir, _first(params, "right"))
        maps = _all_image_maps(output_dir, include_regions=False)
        if left is None and maps:
            left = maps[0]
        if right is None and len(maps) > 1:
            right = maps[1]
        elif right is None and maps:
            right = maps[0]
        if not maps:
            return self._page_shell("Compare maps", f"<h1>Compare maps</h1><p><code>{_safe_text(output_dir)}</code></p><section class='card'><p class='muted'>No maps available yet.</p></section>")

        mode = _first(params, "mode", "side")
        if mode not in {"side", "vertical-slider", "horizontal-slider", "overlay"}:
            mode = "side"
        contour = _first(params, "contour", "none")
        if contour not in {"none", "elevation", "temperature", "precipitation"}:
            contour = "none"
        contour_levels = _first(params, "contour_levels", "auto")
        if contour_levels != "auto":
            try:
                contour_levels = str(max(4, min(40, int(contour_levels))))
            except Exception:
                contour_levels = "auto"
        opacity = _first(params, "opacity", "50")
        try:
            opacity_f = max(0.0, min(1.0, float(opacity) / 100.0))
        except Exception:
            opacity_f = 0.5
        grid_checked = " checked" if _first(params, "grid") == "on" else ""
        grid_class = " demarcation" if grid_checked else ""
        left_label = _rel_label(left, output_dir) if left is not None else ""
        right_label = _rel_label(right, output_dir) if right is not None else ""
        left_raw = _file_url(left) if left is not None else ""
        right_raw = _file_url(right) if right is not None else ""
        # Compare layers use the map-data crop, not the full PNG, so appended
        # legend bands cannot distort alignment. Raw links still open the
        # original generated image with its embedded legend.
        left_map_src = _map_data_url(left, output_dir) if left is not None else ""
        right_map_src = _map_data_url(right, output_dir) if right is not None else ""
        contour_url = ""
        contour_label = ""
        if contour != "none":
            contour_url = "/contour-overlay?" + urllib.parse.urlencode({"output_dir": str(output_dir), "kind": contour, "max_dim": "2048", "levels": contour_levels, "zoom": "1"})
            contour_label = _contour_kind_label(contour)

        def selected(value: str) -> str:
            return " selected" if mode == value else ""

        def contour_selected(value: str) -> str:
            return " selected" if contour == value else ""
        def contour_level_selected(value: str) -> str:
            return " selected" if contour_levels == value else ""

        world_dims = _world_grid_dimensions(output_dir) or (0, 0)
        ww, wh = world_dims
        data_attrs = f'data-output-dir="{_safe_text(str(output_dir))}" data-world-width="{ww}" data-world-height="{wh}"'
        left_path_attr = _safe_text(str(left) if left else "")
        right_path_attr = _safe_text(str(right) if right else "")
        left_scale = _map_scale_summary(left, output_dir) if left is not None else ""
        right_scale = _map_scale_summary(right, output_dir) if right is not None else ""
        left_legend = _legend_html(left) if left is not None else ""
        right_legend = _legend_html(right) if right is not None else ""
        contour_img = f'<img class="contour-layer wg-layer" src="{contour_url}" data-base-src="{contour_url}" alt="{_safe_text(contour_label)}">' if contour_url else ""
        side_contour_left = f'<img class="contour-layer side-contour" src="{contour_url}" data-base-src="{contour_url}" alt="{_safe_text(contour_label)}">' if contour_url else ""
        side_contour_right = f'<img class="contour-layer side-contour" src="{contour_url}" data-base-src="{contour_url}" alt="{_safe_text(contour_label)}">' if contour_url else ""
        split_line_class = "horizontal" if mode == "horizontal-slider" else "vertical"

        select_form = f'''
<form method="get" action="/compare" class="card">
  <input type="hidden" name="output_dir" value="{_safe_text(output_dir)}">
  <div class="grid compact-grid">
    <div><label>Left/base map</label><select name="left">{self._map_options(output_dir, left)}</select></div>
    <div><label>Right/comparison map</label><select name="right">{self._map_options(output_dir, right)}</select></div>
    <div><label>Compare mode</label><select name="mode">
      <option value="side"{selected('side')}>Side by side, synced</option>
      <option value="vertical-slider"{selected('vertical-slider')}>Vertical before/after slider</option>
      <option value="horizontal-slider"{selected('horizontal-slider')}>Horizontal before/after slider</option>
      <option value="overlay"{selected('overlay')}>Overlay with opacity</option>
    </select></div>
    <div><label>Overlay opacity %</label><input name="opacity" value="{_safe_text(int(opacity_f*100))}"></div>
    <div><label>Contour overlay</label><select name="contour">
      <option value="none"{contour_selected('none')}>None</option>
      <option value="elevation"{contour_selected('elevation')}>Elevation</option>
      <option value="temperature"{contour_selected('temperature')}>Temperature</option>
      <option value="precipitation"{contour_selected('precipitation')}>Rainfall</option>
    </select></div>
    <div><label>Contour density</label><select name="contour_levels">
      <option value="auto"{contour_level_selected('auto')}>Auto with zoom</option>
      <option value="6"{contour_level_selected('6')}>Sparse</option>
      <option value="10"{contour_level_selected('10')}>Normal</option>
      <option value="16"{contour_level_selected('16')}>Dense</option>
      <option value="24"{contour_level_selected('24')}>Very dense</option>
    </select></div>
  </div>
  <div class="row">
    <label><input type="checkbox" name="grid" style="width:auto"{grid_checked}> demarcation grid</label>
    <button type="submit">Compare selected maps</button>
    <a class="button secondary" href="/pipeline?output_dir={urllib.parse.quote(str(output_dir))}">Back to pipeline</a>
  </div>
</form>
'''
        side_html = f'''
<div class="compare-grid" style="margin-top:18px">
  <section class="compare-pane card">
    <h2>Left: {_safe_text(left_label)}</h2>
    <p class="muted small">{_safe_text(left_scale)}</p>
    <div class="row small"><a href="{_map_view_url(left, output_dir) if left else '#'}">Open zoom viewer</a><a href="{left_raw}" target="_blank">Raw</a></div>
    <div id="left-box" class="viewer-wrap sync-box{grid_class}"><div class="side-stack" {data_attrs}><img id="left-img" class="zoom-image sync-img" src="{left_map_src}" alt="{_safe_text(left_label)}" data-path="{left_path_attr}">{side_contour_left}<div class="map-grid-layer side-grid"></div></div><div class="hover-readout map-hover-readout">Hover for map info.</div></div>
    <aside class="legend-panel" style="margin-top:10px">{left_legend}</aside>
  </section>
  <section class="compare-pane card">
    <h2>Right: {_safe_text(right_label)}</h2>
    <p class="muted small">{_safe_text(right_scale)}</p>
    <div class="row small"><a href="{_map_view_url(right, output_dir) if right else '#'}">Open zoom viewer</a><a href="{right_raw}" target="_blank">Raw</a></div>
    <div id="right-box" class="viewer-wrap sync-box{grid_class}"><div class="side-stack" {data_attrs}><img id="right-img" class="zoom-image sync-img" src="{right_map_src}" alt="{_safe_text(right_label)}" data-path="{right_path_attr}">{side_contour_right}<div class="map-grid-layer side-grid"></div></div><div class="hover-readout map-hover-readout">Hover for map info.</div></div>
    <aside class="legend-panel" style="margin-top:10px">{right_legend}</aside>
  </section>
</div>
'''
        slider_or_overlay_html = f'''
<section class="card compare-workspace" id="compare-workspace">
  <h2>{_safe_text(mode.replace('-', ' ').title())}</h2>
  <p class="muted small">Base: <code>{_safe_text(left_label)}</code> | Comparison: <code>{_safe_text(right_label)}</code></p>
  <p class="muted small">Base scale: {_safe_text(left_scale)}</p>
  <div id="compare-stage" class="compare-stage{grid_class}" data-mode="{_safe_text(mode)}" data-opacity="{opacity_f:.3f}" {data_attrs}>
    <img id="base-layer" class="compare-layer wg-layer" src="{left_map_src}" alt="{_safe_text(left_label)}" data-path="{left_path_attr}">
    <div id="clip-layer" class="compare-clip {'horizontal' if mode == 'horizontal-slider' else ''}"><img id="compare-layer" class="compare-layer wg-layer" src="{right_map_src}" alt="{_safe_text(right_label)}" data-path="{right_path_attr}"></div>
    <img id="overlay-layer" class="compare-layer compare-overlay-layer wg-layer" src="{right_map_src}" alt="{_safe_text(right_label)}" data-path="{right_path_attr}">
    {contour_img}
    <div class="map-grid-layer stage-grid wg-layer"></div>
    <div id="split-line" class="compare-split-line {split_line_class}"></div>
  </div>
  <div id="stage-hover-readout" class="hover-readout">Hover for map info.</div>
</section>
<div class="grid" style="margin-top:12px"><section class="card"><h2>Base legend</h2>{left_legend}</section><section class="card"><h2>Comparison legend</h2>{right_legend}</section></div>
'''
        display_html = side_html if mode == "side" else slider_or_overlay_html
        body = f'''
<h1>Compare maps</h1>
<p><code>{_safe_text(output_dir)}</code></p>
{select_form}
<div id="compare-controls" class="sticky-tools row">
  <button type="button" onclick="wgCompareZoomIn()">Zoom in</button>
  <button type="button" onclick="wgCompareZoomOut()">Zoom out</button>
  <button type="button" onclick="wgCompareReset()">Reset view</button>
  <button type="button" onclick="wgCompareToggleGrid()">Toggle demarcation grid</button>
  <button type="button" onclick="wgCompareFullscreen()">Full screen compare</button>
  <button class="secondary full-exit" type="button" onclick="wgCompareExitFullscreen()">Exit full-screen compare</button>
  <label class="small">Slider position <input id="split-control" type="range" min="0" max="100" value="50" style="width:180px"></label>
  <label class="small">Overlay opacity <input id="opacity-control" type="range" min="0" max="100" value="{_safe_text(int(opacity_f*100))}" style="width:180px"></label>
  <span class="muted small">Pan/zoom is synced. Mouse wheel zooms smoothly. Slider/overlay modes normalize differently sized images to the base map data area.</span>
</div>
<div id="compare-scale-readout" class="scale-readout">Scale will update as you move across the map.</div>
<div id="compare-workspace-root">
{display_html}
</div>
<p class="muted small">Contour and grid overlays are image-space layers, so they move with the map during pan/zoom. Grid density and contour density adapt as you zoom. Hover readouts use saved state when available.</p>
{self._compare_script(mode)}
'''
        return self._page_shell("Compare maps", body)

    def _compare_script(self, mode: str) -> str:
        return f'''
<script>
(function() {{
  let scale = 1.0;
  const mode = '{_safe_text(mode)}';
  const boxes = Array.from(document.querySelectorAll('.sync-box'));
  const sideStacks = Array.from(document.querySelectorAll('.side-stack'));
  const stage = document.getElementById('compare-stage');
  const splitControl = document.getElementById('split-control');
  const opacityControl = document.getElementById('opacity-control');
  const clip = document.getElementById('clip-layer');
  const overlay = document.getElementById('overlay-layer');
  const splitLine = document.getElementById('split-line');
  const scaleLine = document.getElementById('compare-scale-readout');
  let tx = 0, ty = 0, dragging = false, sx = 0, sy = 0, startX = 0, startY = 0, syncing = false, hoverTimer = null;
  let lastContourLevel = null;

  function naturalDataHeight(img) {{
    const host = img.closest('[data-world-width]') || stage;
    const ww = parseFloat(host && host.dataset.worldWidth || '0');
    const wh = parseFloat(host && host.dataset.worldHeight || '0');
    if (ww > 0 && wh > 0 && img.naturalWidth) return Math.min(img.naturalHeight || img.naturalWidth * wh / ww, img.naturalWidth * wh / ww);
    return img.naturalHeight || 0;
  }}
  function contourLevelsForScale() {{
    if (scale < 0.8) return 5;
    if (scale < 1.8) return 8;
    if (scale < 4.0) return 12;
    if (scale < 9.0) return 16;
    return 22;
  }}
  function updateContourDensity() {{
    const level = contourLevelsForScale();
    if (level === lastContourLevel) return;
    lastContourLevel = level;
    document.querySelectorAll('.contour-layer').forEach(el => {{
      const base = el.dataset.baseSrc || el.getAttribute('src') || '';
      if (!base) return;
      try {{
        const url = new URL(base, window.location.origin);
        const requestedLevels = url.searchParams.get('levels') || 'auto';
        if (requestedLevels === 'auto') url.searchParams.set('zoom', scale.toFixed(3));
        el.src = url.pathname + '?' + url.searchParams.toString();
      }} catch (e) {{}}
    }});
  }}
  function updateGridDensity() {{
    let baseW = 0, baseH = 0;
    const base = document.getElementById('base-layer') || document.getElementById('left-img');
    if (base && base.naturalWidth) {{ baseW = base.naturalWidth; baseH = naturalDataHeight(base); }}
    let deg = 30;
    if (scale >= 1.4) deg = 15;
    if (scale >= 2.8) deg = 10;
    if (scale >= 5.5) deg = 5;
    if (scale >= 11) deg = 2;
    if (scale >= 20) deg = 1;
    const lonPx = Math.max(12, baseW * scale * deg / 360.0);
    const latPx = Math.max(12, baseH * scale * deg / 180.0);
    document.querySelectorAll('.map-grid-layer').forEach(el => {{ el.style.backgroundSize = lonPx.toFixed(1) + 'px ' + latPx.toFixed(1) + 'px'; }});
  }}
  function sizeImageSpaceOverlays() {{
    document.querySelectorAll('.side-stack').forEach(stack => {{
      const img = stack.querySelector('img.sync-img');
      if (!img || !img.naturalWidth) return;
      const dataH = naturalDataHeight(img);
      const w = img.naturalWidth * scale;
      const h = img.naturalHeight * scale;
      stack.style.width = w.toFixed(1) + 'px';
      stack.style.height = h.toFixed(1) + 'px';
      img.style.width = w.toFixed(1) + 'px';
      img.style.height = h.toFixed(1) + 'px';
      stack.querySelectorAll('.contour-layer,.map-grid-layer').forEach(el => {{ el.style.width = w.toFixed(1) + 'px'; el.style.height = (dataH * scale).toFixed(1) + 'px'; }});
    }});
    const base = document.getElementById('base-layer');
    if (base && base.naturalWidth) {{
      const baseW = base.naturalWidth;
      const baseH = naturalDataHeight(base);
      document.querySelectorAll('#compare-stage .wg-layer').forEach(el => {{ el.style.width = baseW + 'px'; el.style.height = baseH + 'px'; }});
      document.querySelectorAll('#compare-stage > .contour-layer, #compare-stage > .map-grid-layer').forEach(el => {{ el.style.width = baseW + 'px'; el.style.height = baseH + 'px'; }});
    }}
    updateGridDensity();
    updateContourDensity();
  }}
  function applySide() {{ sizeImageSpaceOverlays(); sideStacks.forEach(stack => {{ stack.style.transform = 'none'; }}); }}
  function applyStage() {{
    sizeImageSpaceOverlays();
    document.querySelectorAll('#compare-stage .wg-layer').forEach(el => el.style.transform = 'translate(' + tx.toFixed(1) + 'px,' + ty.toFixed(1) + 'px) scale(' + scale.toFixed(4) + ')');
  }}
  function updateScaleLine(data) {{
    if (!scaleLine) return;
    let text = 'Zoom ' + scale.toFixed(2) + '×';
    if (data && data.ew_km_per_cell && data.ns_km_per_cell) {{
      text += ' · cursor cell ≈ ' + data.ew_km_per_cell + ' km E-W × ' + data.ns_km_per_cell + ' km N-S';
      text += ' · screen pixel ≈ ' + (data.ew_km_per_cell / scale).toFixed(3) + ' km E-W × ' + (data.ns_km_per_cell / scale).toFixed(3) + ' km N-S';
    }} else {{ text += ' · hover over a full-world map for local real-size scale'; }}
    scaleLine.textContent = text;
  }}
  function zoomBy(factor, anchorBox, cx, cy) {{
    const old = scale;
    scale = Math.max(0.06, Math.min(40, scale * factor));
    const ratio = scale / old;
    if (anchorBox) {{
      const px = cx === undefined ? anchorBox.clientWidth / 2 : cx;
      const py = cy === undefined ? anchorBox.clientHeight / 2 : cy;
      anchorBox.scrollLeft = (anchorBox.scrollLeft + px) * ratio - px;
      anchorBox.scrollTop = (anchorBox.scrollTop + py) * ratio - py;
    }} else if (stage) {{
      tx = tx * ratio;
      ty = ty * ratio;
    }}
    applySide(); applyStage(); updateScaleLine(null);
  }}
  function applySplit() {{
    const v = splitControl ? parseFloat(splitControl.value || '50') : 50;
    const op = opacityControl ? Math.max(0, Math.min(1, parseFloat(opacityControl.value || '50') / 100)) : 0.5;
    if (mode === 'vertical-slider' && clip) {{ clip.style.width = v + '%'; clip.style.height = '100%'; }}
    if (mode === 'horizontal-slider' && clip) {{ clip.style.height = v + '%'; clip.style.width = '100%'; }}
    if (splitLine) {{ if (mode === 'horizontal-slider') splitLine.style.top = v + '%'; else splitLine.style.left = v + '%'; }}
    if (overlay) overlay.style.opacity = (mode === 'overlay') ? op : '0';
    if (clip) clip.style.display = (mode === 'overlay') ? 'none' : 'block';
    if (splitLine) splitLine.style.display = (mode === 'overlay') ? 'none' : 'block';
  }}
  window.wgCompareZoomIn = function() {{ zoomBy(1.18, boxes[0]); }};
  window.wgCompareZoomOut = function() {{ zoomBy(1/1.18, boxes[0]); }};
  window.wgCompareReset = function() {{ scale = 1.0; tx = 0; ty = 0; boxes.forEach(b => {{ b.scrollLeft = 0; b.scrollTop = 0; }}); applySide(); applyStage(); updateScaleLine(null); }};
  window.wgCompareToggleGrid = function() {{ document.querySelectorAll('.viewer-wrap,.compare-stage').forEach(el => el.classList.toggle('demarcation')); updateGridDensity(); }};
  window.wgCompareFullscreen = function() {{
    const root = document.getElementById('compare-workspace-root'); if (!root) return;
    root.classList.add('full-compare');
    if (root.requestFullscreen && !document.fullscreenElement) root.requestFullscreen().catch(() => {{}});
  }};
  window.wgCompareExitFullscreen = function() {{
    const root = document.getElementById('compare-workspace-root'); if (root) root.classList.remove('full-compare');
    if (document.exitFullscreen && document.fullscreenElement) document.exitFullscreen().catch(() => {{}});
  }};
  document.addEventListener('fullscreenchange', () => {{ if (!document.fullscreenElement) {{ const root = document.getElementById('compare-workspace-root'); if (root) root.classList.remove('full-compare'); }} }});
  boxes.forEach(box => {{
    box.addEventListener('scroll', function() {{
      if (syncing || boxes.length < 2) return; syncing = true;
      const other = boxes.find(b => b !== box);
      const xr = box.scrollLeft / Math.max(1, box.scrollWidth - box.clientWidth);
      const yr = box.scrollTop / Math.max(1, box.scrollHeight - box.clientHeight);
      other.scrollLeft = xr * Math.max(1, other.scrollWidth - other.clientWidth);
      other.scrollTop = yr * Math.max(1, other.scrollHeight - other.clientHeight);
      syncing = false;
    }});
    let localDrag = false, bx=0, by=0, bl=0, bt=0;
    box.addEventListener('mousedown', e => {{ localDrag = true; box.classList.add('dragging'); bx=e.clientX; by=e.clientY; bl=box.scrollLeft; bt=box.scrollTop; }});
    window.addEventListener('mouseup', () => {{ localDrag = false; box.classList.remove('dragging'); }});
    window.addEventListener('mousemove', e => {{ if (!localDrag) return; box.scrollLeft = bl - (e.clientX - bx); box.scrollTop = bt - (e.clientY - by); }});
    box.addEventListener('wheel', e => {{ e.preventDefault(); zoomBy(Math.exp(-e.deltaY * 0.0015), box, e.clientX - box.getBoundingClientRect().left, e.clientY - box.getBoundingClientRect().top); }}, {{passive:false}});
    box.addEventListener('mousemove', e => requestHover(e, box.querySelector('img.sync-img'), box.querySelector('.map-hover-readout')));
  }});
  if (stage) {{
    stage.addEventListener('mousedown', e => {{ dragging=true; stage.classList.add('dragging'); sx=e.clientX; sy=e.clientY; startX=tx; startY=ty; }});
    window.addEventListener('mouseup', () => {{ dragging=false; if (stage) stage.classList.remove('dragging'); }});
    window.addEventListener('mousemove', e => {{ if (!dragging) return; tx=startX + e.clientX - sx; ty=startY + e.clientY - sy; applyStage(); }});
    stage.addEventListener('wheel', e => {{ e.preventDefault(); zoomBy(Math.exp(-e.deltaY * 0.0015), null); }}, {{passive:false}});
    stage.addEventListener('mousemove', e => requestHover(e, document.getElementById('base-layer'), document.getElementById('stage-hover-readout')));
  }}
  function requestHover(e, img, readout) {{
    if (!img || !readout) return;
    const rect = img.getBoundingClientRect();
    const x = (e.clientX - rect.left) / Math.max(0.0001, rect.width) * (img.naturalWidth || rect.width);
    const y = (e.clientY - rect.top) / Math.max(0.0001, rect.height) * (img.naturalHeight || rect.height);
    if (x < 0 || y < 0 || x > (img.naturalWidth || rect.width) || y > (img.naturalHeight || rect.height)) return;
    clearTimeout(hoverTimer);
    const host = img.closest('[data-output-dir]') || stage;
    hoverTimer = setTimeout(() => {{
      const url = '/map-info?output_dir=' + encodeURIComponent(host ? host.dataset.outputDir || '' : '') + '&path=' + encodeURIComponent(img.dataset.path || '') + '&x=' + encodeURIComponent(x.toFixed(1)) + '&y=' + encodeURIComponent(y.toFixed(1));
      fetch(url).then(r => r.json()).then(data => {{
      const esc = (s) => String(s || '').replace(/[&<>"']/g, ch => ({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[ch]));
      if (data.active_value) {{
        const ctx = Array.isArray(data.context) ? data.context.map(esc).join(' · ') : esc(data.text || '');
        readout.innerHTML = '<div class="hover-active-value">' + esc(data.active_value) + '</div><div class="hover-context-value">' + ctx + '</div>';
      }} else {{
        readout.textContent = data.text || '';
      }}
      updateScaleLine(data);
    }}).catch(() => {{}});
    }}, 70);
  }}
  if (splitControl) splitControl.addEventListener('input', applySplit);
  if (opacityControl) opacityControl.addEventListener('input', applySplit);
  window.addEventListener('load', () => {{ sizeImageSpaceOverlays(); applySide(); applyStage(); applySplit(); updateScaleLine(null); }});
  applySide(); applyStage(); applySplit(); updateScaleLine(null);
}})();
</script>
'''


    def _page_csv(self, params: dict[str, list[str]]) -> str:
        output_raw = _first(params, "output_dir")
        output_dir = Path(output_raw).expanduser() if output_raw else Path.cwd()
        raw_path = _first(params, "path")
        q = _first(params, "q").strip().lower()
        path = Path(raw_path).expanduser() if raw_path else output_dir / "landmass_components.csv"
        if not path.exists() or not path.is_file() or path.suffix.lower() != ".csv":
            return self._page_shell("CSV preview", f"<section class='card'><h1>CSV preview</h1><p class='muted'>CSV file not found: <code>{_safe_text(path)}</code></p><p><a class='button secondary' href='/files?output_dir={urllib.parse.quote(str(output_dir))}'>Back to files</a></p></section>")
        rows: list[dict[str, str]] = []
        fieldnames: list[str] = []
        total = 0
        matched = 0
        try:
            with path.open("r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                fieldnames = list(reader.fieldnames or [])
                for row in reader:
                    total += 1
                    hay = " ".join(str(row.get(k, "")) for k in fieldnames).lower()
                    if q and q not in hay:
                        continue
                    matched += 1
                    if len(rows) < 350:
                        rows.append({k: str(row.get(k, "")) for k in fieldnames})
        except Exception as exc:
            return self._page_shell("CSV preview", f"<section class='card'><h1>CSV preview</h1><p class='muted'>Could not read CSV: {_safe_text(exc)}</p></section>")
        head = "".join(f"<th>{_safe_text(_humanize_key(k))}</th>" for k in fieldnames)
        body_rows = []
        for row in rows:
            body_rows.append("<tr>" + "".join(f"<td>{_safe_text(row.get(k, ''))}</td>" for k in fieldnames) + "</tr>")
        body = "".join(body_rows) or f"<tr><td colspan='{max(1, len(fieldnames))}' class='muted'>No matching rows.</td></tr>"
        search_value = _safe_text(q)
        rel = _safe_text(_rel_label(path, output_dir))
        out_q = urllib.parse.quote(str(output_dir))
        raw_q = urllib.parse.quote(str(path.resolve()))
        page = f"""
<h1>CSV preview</h1>
<p><code>{rel}</code></p>
<div class='row'><a class='button secondary' href='/files?output_dir={out_q}'>Back to files</a><a class='button secondary' href='{_file_url(path)}' target='_blank'>Download/Open raw CSV</a></div>
<section class='card'>
  <form method='get' action='/csv' class='row'>
    <input type='hidden' name='output_dir' value='{_safe_text(str(output_dir))}'>
    <input type='hidden' name='path' value='{_safe_text(str(path.resolve()))}'>
    <input name='q' value='{search_value}' placeholder='Filter rows...'>
    <button>Filter</button>
    <a class='button secondary' href='/csv?output_dir={out_q}&path={raw_q}'>Clear</a>
  </form>
  <p class='help'>Showing {len(rows)} rows; matched {matched} of {total}. This preview is capped at 350 rows for browser speed.</p>
  <div class='table-scroll'><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>
</section>
"""
        return self._page_shell("CSV preview", page)

    def _page_files(self, params: dict[str, list[str]]) -> str:
        output_dir = Path(_first(params, "output_dir")).expanduser()
        remember_run(output_dir)
        groups = _file_groups(output_dir) if output_dir.exists() else {}
        sections = ""
        for name, files in groups.items():
            if not files:
                continue
            rows = ""
            for path in files:
                rel = _rel_label(path, output_dir)
                size = path.stat().st_size if path.exists() else 0
                if path.suffix.lower() in IMAGE_SUFFIXES:
                    primary = f"<a href='{_map_view_url(path, output_dir)}'>View</a>"
                elif path.suffix.lower() == ".csv":
                    primary = f"<a href='/csv?output_dir={urllib.parse.quote(str(output_dir))}&path={urllib.parse.quote(str(path.resolve()))}'>Preview CSV</a>"
                else:
                    primary = f"<a href='{_file_url(path)}' target='_blank'>Open</a>"
                rows += f"<tr><td><code>{_safe_text(rel)}</code></td><td>{_safe_text(path.suffix.lower())}</td><td>{size:,}</td><td>{primary} <a href='{_file_url(path)}' target='_blank'>Raw</a></td></tr>"
            sections += f"""
<section class="card" style="margin-top:18px">
  <h2>{_safe_text(name)} <span class="pill">{len(files)}</span></h2>
  <table><thead><tr><th>File</th><th>Type</th><th>Bytes</th><th>Actions</th></tr></thead><tbody>{rows}</tbody></table>
</section>
"""
        sections = sections or "<section class='card'><p class='muted'>No files found yet.</p></section>"
        body = f"""
<h1>Run file browser</h1>
<p><code>{_safe_text(output_dir)}</code></p>
<div class="row"><a class="button secondary" href="/pipeline?output_dir={urllib.parse.quote(str(output_dir))}">Back to pipeline</a><a class="button secondary" href="/compare?output_dir={urllib.parse.quote(str(output_dir))}">Compare maps</a></div>
{sections}
"""
        return self._page_shell("Files", body)

    def _page_jobs(self, params: dict[str, list[str]]) -> str:
        with JOBS_LOCK:
            live_jobs = list(JOBS.values())
        run_keys = list(load_recent_runs())
        for job in live_jobs:
            if job.output_dir is not None:
                key = str(job.output_dir)
                if key not in run_keys:
                    run_keys.insert(0, key)
        sections = ""
        for run in run_keys:
            output_dir = Path(run).expanduser()
            rows_by_id: dict[str, dict[str, Any]] = {}
            for item in _historical_jobs_for_run(output_dir):
                jid = str(item.get("id", ""))
                if jid:
                    rows_by_id[jid] = item
            for job in live_jobs:
                if job.output_dir is not None and Path(job.output_dir) == output_dir:
                    rows_by_id[job.id] = _job_record(job)
            if not rows_by_id:
                continue
            rows = ""
            for item in sorted(rows_by_id.values(), key=lambda it: str(it.get("created_at", "")), reverse=True)[:60]:
                status = str(item.get("status", ""))
                cls = "complete" if status == "complete" else ("failed" if status == "failed" else ("missing" if status == "cancelled" else "stale"))
                jid = str(item.get("id", ""))
                if jid in JOBS:
                    id_link = f"<a href='/job/{_safe_text(jid)}'>{_safe_text(jid)}</a>"
                elif item.get("log_path"):
                    log_url = "/file/" + urllib.parse.quote(str(Path(str(item.get("log_path"))).resolve()))
                    id_link = f"<a href='{log_url}' target='_blank'>{_safe_text(jid)}</a>"
                else:
                    id_link = _safe_text(jid)
                rows += f"<tr><td>{id_link}</td><td>{_safe_text(item.get('label',''))}</td><td class='{cls}'>{_safe_text(status)}</td><td>{_safe_text(item.get('created_at',''))}</td><td>{_safe_text(item.get('finished_at',''))}</td></tr>"
            sections += f"""
<section class="card" style="margin-top:18px">
  <h2><code>{_safe_text(run)}</code> <a class='button secondary' href='/pipeline?output_dir={urllib.parse.quote(run)}'>Open pipeline</a></h2>
  <table><thead><tr><th>ID/log</th><th>Label</th><th>Status</th><th>Created</th><th>Finished</th></tr></thead><tbody>{rows}</tbody></table>
</section>
"""
        orphan_jobs = [job for job in live_jobs if job.output_dir is None]
        if orphan_jobs:
            rows = ""
            for job in orphan_jobs:
                cls = "complete" if job.status == "complete" else ("failed" if job.status == "failed" else ("missing" if job.status == "cancelled" else "stale"))
                rows += f"<tr><td><a href='/job/{job.id}'>{_safe_text(job.id)}</a></td><td>{_safe_text(job.label)}</td><td class='{cls}'>{_safe_text(job.status)}</td><td>{_safe_text(job.created_at)}</td><td>{_safe_text(job.finished_at or '')}</td></tr>"
            sections += f"""
<section class="card" style="margin-top:18px">
  <h2>No run / UI job</h2>
  <table><thead><tr><th>ID</th><th>Label</th><th>Status</th><th>Created</th><th>Finished</th></tr></thead><tbody>{rows}</tbody></table>
</section>
"""
        sections = sections or "<section class='card'><p class='muted'>No jobs found yet. Job history is persisted per run after jobs are started from the web UI.</p></section>"
        return self._page_shell("Jobs", f"<h1>Jobs grouped by run</h1><p class='muted'>This page merges live jobs from the current server session with saved job history from recent run folders.</p>{sections}", refresh=5)

    def _page_job(self, jid: str) -> str:
        with JOBS_LOCK:
            job = JOBS.get(jid)
        if job is None:
            return self._page_shell("Job not found", f"<h1>Job not found</h1><p>No in-memory job with ID <code>{_safe_text(jid)}</code>.</p><p class='muted'>If the server restarted, open the run dashboard and inspect diagnostics/webui_jobs.</p>")
        cls = "complete" if job.status == "complete" else ("failed" if job.status == "failed" else ("missing" if job.status == "cancelled" else "stale"))
        log = _tail(job.log_path)
        latest = _log_summary(job.log_path)
        run_link = ""
        if job.output_dir is not None:
            run_link = f"<a class='button secondary' href='/pipeline?output_dir={urllib.parse.quote(str(job.output_dir))}'>Open pipeline workspace</a>"
        raw_log = _file_url(job.log_path) if job.log_path.exists() else "#"
        cancel_form = ""
        if job.status in {"queued", "running", "cancelling"}:
            cancel_form = f"""
<form method='post' action='/job-action' style='display:inline' onsubmit="return confirm('Cancel this running job? Completed stages stay on disk; the current incomplete stage may need rerun.');">
  <input type='hidden' name='action' value='cancel'>
  <input type='hidden' name='job_id' value='{_safe_text(job.id)}'>
  <button type='submit' class='danger'>Cancel job</button>
</form>
"""
        body = f"""
<h1>Job { _safe_text(job.id) }</h1>
<p><span class="pill">{_safe_text(job.label)}</span> <span class="{cls}">{_safe_text(job.status)}</span></p>
<section class="card">
  <h2>Current progress</h2>
  <p>{_safe_text(latest)}</p>
  <p class="muted small">Started: {_safe_text(job.started_at or '')} · Finished: {_safe_text(job.finished_at or '')} · Return code: {_safe_text(job.returncode if job.returncode is not None else '')}</p>
</section>
<p><code>{_safe_text(' '.join(job.command))}</code></p>
<div class="row">{run_link}<a class="button secondary" href="/jobs">All jobs</a><a class="button secondary" href="{raw_log}" target="_blank">Open full log</a>{cancel_form}</div>
<h2>Log tail</h2>
<pre>{_safe_text(log)}</pre>
"""
        refresh = 3 if job.status in {"queued", "running", "cancelling"} else None
        return self._page_shell("Job", body, refresh=refresh)

    def _page_error(self, exc: Exception) -> str:
        return self._page_shell("Error", f"<h1 class='bad'>Error</h1><pre>{_safe_text(type(exc).__name__ + ': ' + str(exc))}</pre>")

    # ------------------------------------------------------------------
    # File serving / response helpers
    # ------------------------------------------------------------------

    def _send_bytes(self, data: bytes, mime: str, *, cache: bool = True) -> None:
        self.send_response(200)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(data)))
        if cache:
            self.send_header("Cache-Control", "public, max-age=120")
        self.end_headers()
        self.wfile.write(data)

    def _send_json(self, data: dict[str, Any], *, status: int = 200) -> None:
        raw = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(raw)

    def _send_map_info(self, params: dict[str, list[str]]) -> None:
        output_raw = _first(params, "output_dir")
        output_dir = Path(output_raw).expanduser() if output_raw else None
        raw_path = _first(params, "path")
        try:
            path = Path(raw_path).expanduser()
            if not raw_path or not path.exists() or not path.is_file():
                self._send_json({"text": "map unavailable", "error": "not_found"}, status=404)
                return
            x = float(_first(params, "x", "0"))
            y = float(_first(params, "y", "0"))
            self._send_json(_map_hover_info(output_dir, path, x, y))
        except Exception as exc:
            self._send_json({"text": f"map info unavailable: {type(exc).__name__}", "error": str(exc)}, status=500)

    def _send_contour_overlay(self, params: dict[str, list[str]]) -> None:
        output_dir = Path(_first(params, "output_dir")).expanduser()
        kind = _first(params, "kind", "elevation")
        max_dim_raw = _first(params, "max_dim", "2048")
        levels_raw = _first(params, "levels", "12")
        zoom_raw = _first(params, "zoom", "1")
        try:
            max_dim = int(max_dim_raw)
        except Exception:
            max_dim = 2048
        try:
            zoom = max(0.05, min(64.0, float(zoom_raw)))
        except Exception:
            zoom = 1.0
        if str(levels_raw).lower() == "auto":
            if zoom < 0.8:
                levels = 5
            elif zoom < 1.8:
                levels = 8
            elif zoom < 4.0:
                levels = 12
            elif zoom < 9.0:
                levels = 16
            else:
                levels = 22
        else:
            try:
                levels = int(levels_raw)
            except Exception:
                levels = 12
        try:
            data = _make_contour_png(output_dir, kind, max_dim=max_dim, levels=levels)
        except Exception as exc:
            # Return a tiny transparent PNG rather than breaking the compare page.
            from PIL import Image

            buf = io.BytesIO()
            Image.new("RGBA", (1, 1), (0, 0, 0, 0)).save(buf, format="PNG")
            data = buf.getvalue()
        self._send_bytes(data, "image/png")


    def _send_map_data_image(self, params: dict[str, list[str]]) -> None:
        output_raw = _first(params, "output_dir")
        output_dir = Path(output_raw).expanduser() if output_raw else None
        raw_path = _first(params, "path")
        try:
            path = Path(raw_path).expanduser()
            if not raw_path or not path.exists() or not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
                self.send_error(404, "Map image not found")
                return
            data = _map_data_image_bytes(output_dir, path)
            self._send_bytes(data, "image/png")
        except Exception as exc:
            self.send_error(500, f"Could not build map-data image: {exc}")

    def _send_file(self, encoded_path: str) -> None:
        raw = urllib.parse.unquote(encoded_path)
        path = Path(raw)
        if not path.exists() or not path.is_file():
            self.send_error(404, "File not found")
            return
        suffix = path.suffix.lower()
        mime = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".csv": "text/csv; charset=utf-8",
            ".json": "application/json; charset=utf-8",
            ".txt": "text/plain; charset=utf-8",
            ".zip": "application/zip",
        }.get(suffix, "application/octet-stream")
        data = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Content-Disposition", f'inline; filename="{path.name}"')
        self.end_headers()
        self.wfile.write(data)

    def _send_html(self, html_text: str, status: int = 200) -> None:
        data = html_text.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _redirect(self, location: str) -> None:
        self.send_response(303)
        self.send_header("Location", location)
        self.end_headers()

    def log_message(self, fmt: str, *args: Any) -> None:
        sys.stderr.write("[webui] " + fmt % args + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Start the local WorldGen browser UI.")
    parser.add_argument("--host", default="127.0.0.1", help="Host/interface to bind. Default: 127.0.0.1")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help=f"Port to bind. Default: {DEFAULT_PORT}")
    parser.add_argument("--open", action="store_true", help="Open the UI in the default browser after starting.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    server = ThreadingHTTPServer((args.host, args.port), WorldGenUIHandler)
    url = f"http://{args.host}:{args.port}/"
    print(f"WorldGen Local Web UI running at {url}")
    print("Press Ctrl+C to stop.")
    if args.open:
        import webbrowser

        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping WorldGen Local Web UI.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
