"""Validation rules for generated moon orbits."""

from __future__ import annotations

from dataclasses import dataclass

from worldgen.models.bodies import Moon


@dataclass(frozen=True)
class MoonValidationResult:
    is_valid: bool
    messages: list[str]


def validate_moon_orbit(moon: Moon) -> MoonValidationResult:
    messages: list[str] = []
    if moon.orbit.periapsis_km <= moon.orbit.roche_limit_km:
        messages.append("Moon periapsis is inside or too near the Roche limit.")
    if moon.orbit.apoapsis_km >= moon.orbit.safe_hill_limit_km:
        messages.append("Moon apoapsis is outside the conservative Hill-sphere stability limit.")
    if moon.orbit.hill_fraction >= 0.33:
        messages.append("Moon orbit consumes too much of the planet's Hill sphere.")
    return MoonValidationResult(is_valid=not messages, messages=messages)
