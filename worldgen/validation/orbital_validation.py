"""Validation rules for plausibly stable generated systems."""

from __future__ import annotations

from dataclasses import dataclass

from worldgen.constants import MIN_MUTUAL_HILL_SEPARATION
from worldgen.models.bodies import Planet, Star
from worldgen.physics.orbital import mutual_hill_radius_au


@dataclass(frozen=True)
class ValidationResult:
    is_valid: bool
    messages: list[str]


def validate_planet_spacing(
    star: Star,
    planets: list[Planet],
    min_mutual_hill_separation: float = MIN_MUTUAL_HILL_SEPARATION,
) -> ValidationResult:
    messages: list[str] = []
    sorted_planets = sorted(planets, key=lambda p: p.orbit.semi_major_axis_au)

    for inner, outer in zip(sorted_planets, sorted_planets[1:]):
        separation = outer.orbit.semi_major_axis_au - inner.orbit.semi_major_axis_au
        mutual_hill = mutual_hill_radius_au(
            inner.mass_earth,
            outer.mass_earth,
            inner.orbit.semi_major_axis_au,
            outer.orbit.semi_major_axis_au,
            star.mass_solar,
        )
        separation_in_hill = separation / mutual_hill if mutual_hill > 0 else 0
        if separation_in_hill < min_mutual_hill_separation:
            messages.append(
                f"{inner.name} and {outer.name} are too close: "
                f"{separation_in_hill:.2f} mutual Hill radii; "
                f"minimum is {min_mutual_hill_separation:.2f}."
            )

    return ValidationResult(is_valid=not messages, messages=messages)
