"""Simplified planetary physics."""

from __future__ import annotations

import math

from worldgen.constants import MOON_MASS_EARTH, MOON_ORBIT_KM


def surface_gravity_g(mass_earth: float, radius_earth: float) -> float:
    return mass_earth / (radius_earth ** 2)


def density_relative_earth(mass_earth: float, radius_earth: float) -> float:
    return mass_earth / (radius_earth ** 3)


def escape_velocity_relative_earth(mass_earth: float, radius_earth: float) -> float:
    return (mass_earth / radius_earth) ** 0.5


def stellar_flux_earth(luminosity_solar: float, semi_major_axis_au: float) -> float:
    return luminosity_solar / (semi_major_axis_au ** 2)


def equilibrium_temperature_k(
    stellar_flux_relative_earth: float,
    albedo: float = 0.30,
) -> float:
    """Estimate blackbody equilibrium temperature.

    Earth's effective equilibrium temperature is about 255 K at albedo 0.30.
    """
    return 255.0 * (stellar_flux_relative_earth ** 0.25) * (((1.0 - albedo) / 0.70) ** 0.25)


def tidal_strength_relative_earth_moon(
    moon_mass_earth: float,
    moon_orbit_km: float,
    planet_radius_earth: float,
) -> float:
    """Approximate tide-raising strength relative to Earth's Moon.

    This is a simple proportional comparison: tide strength is roughly
    proportional to moon mass and inversely proportional to distance cubed.
    The planet radius term adjusts for the size of the world being flexed.
    """
    distance_ratio = MOON_ORBIT_KM / moon_orbit_km
    return (moon_mass_earth / MOON_MASS_EARTH) * (distance_ratio ** 3) * (planet_radius_earth ** 3)


def angular_diameter_degrees(body_radius_earth: float, distance_km: float) -> float:
    """Apparent angular diameter of a body in degrees."""
    from worldgen.constants import EARTH_RADIUS_KM

    radius_km = body_radius_earth * EARTH_RADIUS_KM
    return math.degrees(2.0 * math.atan(radius_km / distance_km))
