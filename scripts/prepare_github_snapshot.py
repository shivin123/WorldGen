"""Create a clean WorldGen source snapshot for GitHub.

This script copies a project folder while excluding generated runs, caches,
local diagnostics, and temporary files that should not be committed.

Example:
    python scripts/prepare_github_snapshot.py --source . --target ../WorldGen_github_clean
"""

from __future__ import annotations

import argparse
import fnmatch
import shutil
from pathlib import Path

EXCLUDE_NAMES = {
    ".git",
    ".hg",
    ".svn",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".venv",
    "venv",
    "env",
    "ENV",
    ".idea",
    ".vscode",
    "diagnostics",
    "maps",
    "state",
    "terrain_regions",
    "regional_terrain",
}

EXCLUDE_PATTERNS = [
    "*.pyc",
    "*.pyo",
    "*.log",
    "*.tmp",
    "*.bak",
    "*.zip",
    "error.txt",
    "output*",
    "staged_web_run*",
    "staged_update*",
    "staged_*_test*",
    "staged_*_smoke*",
    "*_test",
    "*_smoke",
    "real_earth_*",
    "main_planet_*.png",
    "system_orbits.png",
    "system_sizes.png",
    "worldgen_diagnostic_bundle.zip",
]

# Keep source/reference data even if binary. Generated runs are excluded by folder name.
INCLUDE_BINARY_UNDER = {
    Path("worldgen") / "data",
}


def _is_under(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def should_exclude(path: Path, source_root: Path) -> bool:
    rel = path.relative_to(source_root)
    parts = rel.parts

    if any(part in EXCLUDE_NAMES for part in parts):
        # Keep docs/updates and source diagnostics modules; only root/run diagnostics dirs are excluded.
        if "docs" in parts:
            return False
        if rel.parts[:2] == ("worldgen", "output"):
            return False
        return True

    name = path.name
    for pattern in EXCLUDE_PATTERNS:
        if fnmatch.fnmatch(name, pattern) or fnmatch.fnmatch(str(rel), pattern):
            if any(_is_under(rel, keep) or rel == keep for keep in INCLUDE_BINARY_UNDER):
                return False
            return True

    return False


def copy_clean_tree(source: Path, target: Path) -> None:
    source = source.resolve()
    target = target.resolve()

    if source == target or source in target.parents:
        raise ValueError("Target must not be inside the source folder.")

    if target.exists():
        raise FileExistsError(f"Target already exists: {target}")

    target.mkdir(parents=True)

    copied = 0
    skipped = 0
    for path in source.rglob("*"):
        if should_exclude(path, source):
            skipped += 1
            if path.is_dir():
                # rglob will still walk it; this is acceptable for this small project.
                pass
            continue

        rel = path.relative_to(source)
        dest = target / rel
        if path.is_dir():
            dest.mkdir(parents=True, exist_ok=True)
        else:
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, dest)
            copied += 1

    print(f"Created clean snapshot: {target}")
    print(f"Copied files: {copied}")
    print(f"Skipped paths: {skipped}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a clean WorldGen source snapshot for GitHub.")
    parser.add_argument("--source", default=".", help="Source project folder. Default: current folder.")
    parser.add_argument("--target", required=True, help="Target folder to create. Must not already exist.")
    args = parser.parse_args()

    copy_clean_tree(Path(args.source), Path(args.target))


if __name__ == "__main__":
    main()
