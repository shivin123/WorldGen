"""Moon generation for the selected Main Planet."""

from __future__ import annotations

import random

from worldgen.constants import (
    AU_KM,
    EARTH_RADIUS_KM,
    MAX_MOON_ECCENTRICITY,
    MAX_MOON_TO_PLANET_MASS_RATIO,
    MIN_MOON_ORBIT_PLANET_RADII,
    MOON_ICY,
    MOON_MIXED,
    MOON_ROCHE_SAFETY_FACTOR,
    MOON_ROCKY,
    MOON_SAFE_HILL_FRACTION,
)
from worldgen.generators.naming import generate_moon_name
from worldgen.models.bodies import Composition, Moon, MoonOrbit, Planet, Star, ValueSource
from worldgen.physics.orbital import moon_orbital_period_days, roche_limit_km
from worldgen.physics.planetary import angular_diameter_degrees, density_relative_earth, surface_gravity_g, tidal_strength_relative_earth_moon
from worldgen.random_utils import weighted_choice


def generate_main_planet_moon(
    rng: random.Random,
    star: Star,
    planet: Planet,
    moon_strength_preference: str = "moderate",
) -> Moon:
    """Generate exactly one moon for the Main Planet.

    The moon is placed outside the planet's Roche limit and inside a conservative
    fraction of the planet's Hill sphere.
    """
    strength = (moon_strength_preference or "moderate").strip().lower().replace("-", "_").replace(" ", "_")
    if strength not in {"weak", "moderate", "strong"}:
        strength = "moderate"

    for attempt in range(1, 101):
        moon_class = _choose_moon_class(rng, planet)
        mass = _choose_moon_mass(rng, planet, strength)
        radius = _choose_moon_radius(rng, moon_class, mass)
        density = density_relative_earth(mass, radius)
        composition = _choose_moon_composition(rng, moon_class)

        roche = roche_limit_km(
            primary_radius_earth=planet.radius_earth,
            primary_density_relative_earth=planet.density_relative_earth,
            satellite_density_relative_earth=density,
            fluid_body=True,
        )
        planet_radius_km = planet.radius_earth * EARTH_RADIUS_KM
        min_orbit = max(
            roche * MOON_ROCHE_SAFETY_FACTOR,
            planet_radius_km * MIN_MOON_ORBIT_PLANET_RADII,
        )
        safe_hill_limit = planet.hill_radius_au * AU_KM * MOON_SAFE_HILL_FRACTION

        if safe_hill_limit <= min_orbit:
            continue

        # Bias toward broad orbital bands rather than the Roche edge. Very close
        # moons create unrealistic thousand-fold tides for Earth-like worlds;
        # very distant moons approach the Hill stability limit.
        span = safe_hill_limit - min_orbit
        if strength == "weak":
            low, high = min_orbit + 0.50 * span, min_orbit + 0.94 * span
        elif strength == "strong":
            low, high = min_orbit + 0.08 * span, min_orbit + 0.50 * span
        else:
            low, high = min_orbit + 0.25 * span, min_orbit + 0.75 * span
        orbit_km = rng.uniform(low, high)
        eccentricity = rng.uniform(0.0, MAX_MOON_ECCENTRICITY)
        periapsis = orbit_km * (1.0 - eccentricity)
        apoapsis = orbit_km * (1.0 + eccentricity)

        if periapsis <= roche * MOON_ROCHE_SAFETY_FACTOR or apoapsis >= safe_hill_limit:
            continue

        period_days = moon_orbital_period_days(orbit_km, planet.mass_earth, mass)
        hill_fraction = orbit_km / (planet.hill_radius_au * AU_KM)
        tide = tidal_strength_relative_earth_moon(mass, orbit_km, planet.radius_earth)
        min_tide, max_tide = {
            "weak": (0.02, 2.5),
            "moderate": (0.18, 12.0),
            "strong": (0.8, 80.0),
        }[strength]
        if not (min_tide <= tide <= max_tide):
            continue
        mass_ratio = mass / max(1e-9, planet.mass_earth)
        origin = _choose_moon_origin(rng, moon_class, mass_ratio, eccentricity, planet)
        tidal_level = _tidal_effect_level(tide)
        stability_effect = _axial_stability_effect(mass_ratio, hill_fraction, tide)
        name = generate_moon_name(rng, {planet.name}, planet.name)

        return Moon(
            name=name,
            moon_class=moon_class,
            mass_earth=mass,
            radius_earth=radius,
            density_relative_earth=density,
            composition=composition,
            orbit=MoonOrbit(
                semi_major_axis_km=orbit_km,
                eccentricity=eccentricity,
                orbital_period_days=period_days,
                periapsis_km=periapsis,
                apoapsis_km=apoapsis,
                roche_limit_km=roche,
                safe_hill_limit_km=safe_hill_limit,
                hill_fraction=hill_fraction,
            ),
            surface_gravity_g=surface_gravity_g(mass, radius),
            tidal_strength_relative_earth_moon=tide,
            angular_diameter_degrees=angular_diameter_degrees(radius, orbit_km),
            value_sources={
                "mass_earth": ValueSource("RANDOM_GENERATED", "bounded as a fraction of Main Planet mass"),
                "radius_earth": ValueSource("RANDOM_GENERATED", "simple mass-radius relation by moon class"),
                "roche_limit_km": ValueSource("DERIVED", "fluid Roche limit using planet and moon densities"),
                "safe_hill_limit_km": ValueSource("DERIVED", "conservative fraction of planet Hill sphere"),
                "orbital_period_days": ValueSource("DERIVED", "two-body orbital period around planet"),
                "moon_origin": ValueSource("DERIVED", "broad origin class from moon mass, class, and orbit"),
                "tidal_effect_level": ValueSource("DERIVED", "relative tide compared with Earth-Moon"),
                "axial_stability_effect": ValueSource("DERIVED", "mass ratio and orbital placement proxy"),
            },
            moon_origin=origin,
            tidal_effect_level=tidal_level,
            axial_stability_effect=stability_effect,
            notes=[
                f"moon origin model: {origin.replace('_', ' ')}",
                f"tidal effect: {tidal_level}; axial stability effect: {stability_effect}",
            ],
        )

    raise RuntimeError(f"Could not generate a stable moon for {planet.name} after 100 attempts.")


def _choose_moon_class(rng: random.Random, planet: Planet) -> str:
    water = planet.composition.water_ice_fraction
    if water > 0.04:
        choices = [(MOON_ROCKY, 0.40), (MOON_MIXED, 0.45), (MOON_ICY, 0.15)]
    else:
        choices = [(MOON_ROCKY, 0.70), (MOON_MIXED, 0.25), (MOON_ICY, 0.05)]
    return weighted_choice(rng, choices)


def _choose_moon_mass(rng: random.Random, planet: Planet, moon_strength_preference: str = "moderate") -> float:
    # Earth-Moon is about 1.23% Earth mass. This allows smaller ordinary moons
    # and occasional large stabilizing moons while staying below the configured cap.
    cap = min(MAX_MOON_TO_PLANET_MASS_RATIO, 0.025)
    if moon_strength_preference == "weak":
        low, high = 0.0008, min(cap, 0.008)
    elif moon_strength_preference == "strong":
        low, high = 0.008, cap
    else:
        low, high = 0.0015, cap
    return planet.mass_earth * rng.uniform(low, max(low, high))


def _choose_moon_radius(rng: random.Random, moon_class: str, mass_earth: float) -> float:
    if moon_class == MOON_ROCKY:
        density_factor = rng.uniform(0.75, 1.05)
    elif moon_class == MOON_MIXED:
        density_factor = rng.uniform(0.45, 0.75)
    else:
        density_factor = rng.uniform(0.25, 0.50)

    # radius^3 = mass / density in Earth-relative units.
    return (mass_earth / density_factor) ** (1.0 / 3.0)


def _choose_moon_composition(rng: random.Random, moon_class: str) -> Composition:
    if moon_class == MOON_ROCKY:
        iron = rng.uniform(0.05, 0.18)
        water = rng.uniform(0.0001, 0.015)
        gas = 0.0
        silicate = max(0.0, 1.0 - iron - water)
        volatile = "low"
        label = "rocky silicate moon"
    elif moon_class == MOON_MIXED:
        iron = rng.uniform(0.02, 0.10)
        water = rng.uniform(0.08, 0.35)
        gas = 0.0
        silicate = max(0.0, 1.0 - iron - water)
        volatile = "moderate"
        label = "mixed rock-ice moon"
    else:
        iron = rng.uniform(0.00, 0.05)
        silicate = rng.uniform(0.15, 0.40)
        water = max(0.0, 1.0 - iron - silicate)
        gas = 0.0
        volatile = "high"
        label = "icy moon"

    total = iron + silicate + water + gas
    return Composition(
        iron_fraction=iron / total,
        silicate_fraction=silicate / total,
        water_ice_fraction=water / total,
        gas_envelope_fraction=gas / total,
        volatile_inventory=volatile,
        composition_class=label,
    )


def _choose_moon_origin(rng: random.Random, moon_class: str, mass_ratio: float, eccentricity: float, planet: Planet) -> str:
    if moon_class == MOON_ROCKY and mass_ratio >= 0.006:
        choices = [("giant_impact", 0.68), ("co_accreted", 0.20), ("captured", 0.12)]
    elif moon_class == MOON_ICY or planet.composition.water_ice_fraction > 0.04:
        choices = [("co_accreted", 0.48), ("captured", 0.32), ("giant_impact", 0.20)]
    elif eccentricity > 0.025:
        choices = [("captured", 0.52), ("giant_impact", 0.28), ("co_accreted", 0.20)]
    else:
        choices = [("giant_impact", 0.42), ("co_accreted", 0.38), ("captured", 0.20)]
    return weighted_choice(rng, choices)


def _tidal_effect_level(tide: float) -> str:
    if tide < 0.35:
        return "weak"
    if tide < 1.75:
        return "moderate"
    if tide < 15.0:
        return "strong"
    return "extreme"


def _axial_stability_effect(mass_ratio: float, hill_fraction: float, tide: float) -> str:
    if mass_ratio >= 0.008 and 0.025 <= hill_fraction <= 0.22 and tide >= 0.25:
        return "high"
    if mass_ratio >= 0.003 and hill_fraction <= 0.28:
        return "moderate"
    return "low"
