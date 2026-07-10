"""Shared constants for the WorldGen project.

The project intentionally uses normalized astronomy units where they make
inspection easier:
- solar masses, radii, luminosities for stars
- Earth masses and radii for planets
- AU for planet orbits
- kilometers for moon orbits
- Earth years / days for time
"""

from __future__ import annotations

# Unit conversion helpers.
DAYS_PER_EARTH_YEAR = 365.256
EARTH_MASS_IN_SOLAR_MASSES = 3.0034896149156e-6
EARTH_RADIUS_KM = 6371.0
EARTH_MASS_KG = 5.9722e24
EARTH_DENSITY_KG_M3 = 5514.0
SOLAR_RADIUS_KM = 695700.0
AU_KM = 149_597_870.7
G_GRAVITATIONAL = 6.67430e-11
MOON_MASS_EARTH = 0.0123000371
MOON_RADIUS_EARTH = 0.2727
MOON_ORBIT_KM = 384_400.0

# Star generation limits. These are intentionally conservative.
K_STAR_MIN_MASS = 0.65
K_STAR_MAX_MASS = 0.84
G_STAR_MIN_MASS = 0.84
G_STAR_MAX_MASS = 1.10

MIN_SYSTEM_AGE_GYR = 2.0
MAX_SYSTEM_AGE_GYR = 8.0

# Orbit generation limits.
MIN_PLANET_ECCENTRICITY = 0.0
MAX_PLANET_ECCENTRICITY = 0.08
MAX_MAIN_PLANET_ECCENTRICITY = 0.08

# Spacing. Higher values create more conservative planet spacing.
MIN_MUTUAL_HILL_SEPARATION = 10.0

# Moon placement. The safe Hill fraction is intentionally conservative for
# prograde moons and keeps the generated moon well inside the planet's domain.
MIN_MOON_ECCENTRICITY = 0.0
MAX_MOON_ECCENTRICITY = 0.04
MOON_ROCHE_SAFETY_FACTOR = 1.35
MOON_SAFE_HILL_FRACTION = 0.33
MIN_MOON_ORBIT_PLANET_RADII = 4.0
MAX_MOON_TO_PLANET_MASS_RATIO = 0.05

# Main Planet suitability limits.
MAIN_PLANET_MIN_GRAVITY_G = 0.75
MAIN_PLANET_MAX_GRAVITY_G = 1.60
MAIN_PLANET_MIN_STELLAR_FLUX = 0.70
MAIN_PLANET_MAX_STELLAR_FLUX = 1.20

# Basic planet class names.
PLANET_ROCKY = "rocky"
PLANET_SUPER_EARTH = "super_earth"
PLANET_MINI_NEPTUNE = "mini_neptune"
PLANET_ICE_GIANT = "ice_giant"
PLANET_GAS_GIANT = "gas_giant"
PLANET_ICY_DWARF = "icy_dwarf"

# Basic moon class names.
MOON_ROCKY = "rocky_moon"
MOON_ICY = "icy_moon"
MOON_MIXED = "mixed_ice_rock_moon"
