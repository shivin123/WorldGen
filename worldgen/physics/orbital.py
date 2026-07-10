"""Orbital mechanics helpers using simplified Keplerian assumptions."""

from __future__ import annotations

import math

from worldgen.constants import (
    AU_KM,
    DAYS_PER_EARTH_YEAR,
    EARTH_DENSITY_KG_M3,
    EARTH_MASS_IN_SOLAR_MASSES,
    EARTH_MASS_KG,
    EARTH_RADIUS_KM,
    G_GRAVITATIONAL,
)


def orbital_period_years(semi_major_axis_au: float, central_mass_solar: float) -> float:
    """Kepler's third law in solar units.

    P^2 = a^3 / M
    """
    return math.sqrt((semi_major_axis_au ** 3) / central_mass_solar)


def periapsis_apoapsis(semi_major_axis_au: float, eccentricity: float) -> tuple[float, float]:
    return (
        semi_major_axis_au * (1.0 - eccentricity),
        semi_major_axis_au * (1.0 + eccentricity),
    )


def hill_radius_au(
    semi_major_axis_au: float,
    body_mass_earth: float,
    central_mass_solar: float,
    eccentricity: float = 0.0,
) -> float:
    body_mass_solar = body_mass_earth * EARTH_MASS_IN_SOLAR_MASSES
    return semi_major_axis_au * (1.0 - eccentricity) * (body_mass_solar / (3.0 * central_mass_solar)) ** (1.0 / 3.0)


def mutual_hill_radius_au(
    mass1_earth: float,
    mass2_earth: float,
    a1_au: float,
    a2_au: float,
    star_mass_solar: float,
) -> float:
    total_mass_solar = (mass1_earth + mass2_earth) * EARTH_MASS_IN_SOLAR_MASSES
    average_a = (a1_au + a2_au) / 2.0
    return average_a * (total_mass_solar / (3.0 * star_mass_solar)) ** (1.0 / 3.0)


def orbital_period_days(semi_major_axis_au: float, central_mass_solar: float) -> float:
    return orbital_period_years(semi_major_axis_au, central_mass_solar) * DAYS_PER_EARTH_YEAR


def roche_limit_km(
    primary_radius_earth: float,
    primary_density_relative_earth: float,
    satellite_density_relative_earth: float,
    fluid_body: bool = True,
) -> float:
    """Approximate Roche limit around a planet.

    Uses d = C * R_primary * (rho_primary / rho_satellite)^(1/3).
    C is 2.44 for a fluid satellite and about 1.26 for a rigid satellite.
    We use the fluid value by default as a conservative safety boundary.
    """
    coefficient = 2.44 if fluid_body else 1.26
    primary_radius_km = primary_radius_earth * EARTH_RADIUS_KM
    density_ratio = max(primary_density_relative_earth, 1e-9) / max(satellite_density_relative_earth, 1e-9)
    return coefficient * primary_radius_km * (density_ratio ** (1.0 / 3.0))


def moon_orbital_period_days(semi_major_axis_km: float, planet_mass_earth: float, moon_mass_earth: float = 0.0) -> float:
    """Two-body orbital period for a moon around a planet."""
    semi_major_axis_m = semi_major_axis_km * 1000.0
    central_mass_kg = (planet_mass_earth + moon_mass_earth) * EARTH_MASS_KG
    period_seconds = 2.0 * math.pi * math.sqrt((semi_major_axis_m ** 3) / (G_GRAVITATIONAL * central_mass_kg))
    return period_seconds / 86_400.0


def km_to_au(distance_km: float) -> float:
    return distance_km / AU_KM


def au_to_km(distance_au: float) -> float:
    return distance_au * AU_KM


def density_kg_m3_from_relative(relative_earth_density: float) -> float:
    return relative_earth_density * EARTH_DENSITY_KG_M3
