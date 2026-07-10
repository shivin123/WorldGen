"""Simplified stellar physics for G/K main-sequence stars."""

from __future__ import annotations

import math


def luminosity_from_mass(mass_solar: float) -> float:
    """Approximate main-sequence mass-luminosity relation."""
    return mass_solar ** 3.5


def radius_from_mass_and_age(
    mass_solar: float,
    age_gyr: float,
    main_sequence_lifetime_gyr: float,
) -> float:
    """Approximate radius with mild main-sequence expansion.

    This is intentionally simple. It makes older main-sequence stars slightly
    larger while avoiding detailed stellar evolution.
    """
    base_radius = mass_solar ** 0.8
    age_fraction = max(0.0, min(1.0, age_gyr / main_sequence_lifetime_gyr))
    return base_radius * (1.0 + 0.15 * age_fraction)


def main_sequence_lifetime_gyr(mass_solar: float, luminosity_solar: float) -> float:
    return 10.0 * mass_solar / luminosity_solar


def temperature_from_luminosity_and_radius(
    luminosity_solar: float,
    radius_solar: float,
) -> float:
    """Estimate effective temperature using solar-relative scaling."""
    solar_temperature_k = 5772.0
    return solar_temperature_k * (luminosity_solar / (radius_solar ** 2)) ** 0.25


def habitable_zone_au(luminosity_solar: float) -> tuple[float, float]:
    """Conservative-ish first-pass habitable zone approximation."""
    inner = math.sqrt(luminosity_solar / 1.1)
    outer = math.sqrt(luminosity_solar / 0.53)
    return inner, outer


def snow_line_au(luminosity_solar: float) -> float:
    return 2.7 * math.sqrt(luminosity_solar)
