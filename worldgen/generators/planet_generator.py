"""Planet generation with simple composition, orbital properties, and system-architecture context."""

from __future__ import annotations

import math
import random

from worldgen.constants import (
    MAX_PLANET_ECCENTRICITY,
    PLANET_GAS_GIANT,
    PLANET_ICE_GIANT,
    PLANET_ICY_DWARF,
    PLANET_MINI_NEPTUNE,
    PLANET_ROCKY,
    PLANET_SUPER_EARTH,
)
from worldgen.generators.naming import generate_planet_name
from worldgen.models.bodies import Composition, Orbit, Planet, Star, ValueSource
from worldgen.physics.orbital import hill_radius_au, orbital_period_days, orbital_period_years, periapsis_apoapsis
from worldgen.physics.planetary import (
    density_relative_earth,
    equilibrium_temperature_k,
    escape_velocity_relative_earth,
    stellar_flux_earth,
    surface_gravity_g,
)
from worldgen.random_utils import weighted_choice


DEFAULT_ARCHITECTURE = "solar_like_mixed"
DEFAULT_MAIN_PLANET_PREFERENCE = "earthlike"


def generate_planets(
    rng: random.Random,
    star: Star,
    planet_count: int,
    architecture_type: str | None = None,
    main_planet_preference: str = DEFAULT_MAIN_PLANET_PREFERENCE,
) -> list[Planet]:
    """Generate a full orbital ladder.

    The ladder deliberately includes one habitable-zone candidate, then lets the
    selected system architecture tune spacing, planet classes, volatile inventory,
    and rough orbital temperament.  The downstream selector still performs the
    hard Main Planet eligibility check.
    """
    architecture = _normalize_architecture(architecture_type)
    preference = _normalize_preference(main_planet_preference)
    semi_major_axes, hz_anchor_index = _generate_orbital_ladder(rng, star, planet_count, architecture, preference)
    planets: list[Planet] = []
    used_names: set[str] = set()

    for index, semi_major_axis in enumerate(semi_major_axes):
        is_hz_anchor = index == hz_anchor_index
        planet_class = _choose_planet_class(rng, star, semi_major_axis, architecture, preference, is_hz_anchor)
        mass = _choose_planet_mass(rng, planet_class, architecture, preference, is_hz_anchor)
        radius = _choose_planet_radius(rng, planet_class, mass)
        composition = _choose_composition(rng, planet_class, semi_major_axis, star.snow_line_au, architecture, preference, is_hz_anchor)
        eccentricity = rng.uniform(0.0, _max_eccentricity_for_architecture(architecture, is_hz_anchor))

        period_years = orbital_period_years(semi_major_axis, star.mass_solar)
        period_days = orbital_period_days(semi_major_axis, star.mass_solar)
        periapsis, apoapsis = periapsis_apoapsis(semi_major_axis, eccentricity)
        flux = stellar_flux_earth(star.luminosity_solar, semi_major_axis)
        gravity = surface_gravity_g(mass, radius)
        density = density_relative_earth(mass, radius)
        escape_velocity = escape_velocity_relative_earth(mass, radius)
        hill = hill_radius_au(semi_major_axis, mass, star.mass_solar, eccentricity)

        name = generate_planet_name(rng, used_names)
        planets.append(
            Planet(
                name=name,
                planet_class=planet_class,
                mass_earth=mass,
                radius_earth=radius,
                density_relative_earth=density,
                composition=composition,
                orbit=Orbit(
                    semi_major_axis_au=semi_major_axis,
                    eccentricity=eccentricity,
                    orbital_period_years=period_years,
                    orbital_period_days=period_days,
                    periapsis_au=periapsis,
                    apoapsis_au=apoapsis,
                ),
                stellar_flux_earth=flux,
                equilibrium_temperature_k=equilibrium_temperature_k(flux),
                surface_gravity_g=gravity,
                escape_velocity_relative_earth=escape_velocity,
                hill_radius_au=hill,
                architecture_role="habitable_zone_candidate" if is_hz_anchor else _architecture_role(star, semi_major_axis, planet_class),
                value_sources={
                    "semi_major_axis_au": ValueSource("RANDOM_GENERATED", f"{architecture} orbital ladder with habitable-zone candidate"),
                    "eccentricity": ValueSource("RANDOM_GENERATED", f"low eccentricity tuned by {architecture}"),
                    "orbital_period": ValueSource("DERIVED", "Kepler's third law"),
                    "stellar_flux": ValueSource("DERIVED", "luminosity / distance^2"),
                    "surface_gravity": ValueSource("DERIVED", "mass / radius^2"),
                    "hill_radius": ValueSource("DERIVED", "planet-star Hill sphere"),
                    "architecture_role": ValueSource("DERIVED", "orbital zone and class"),
                },
            )
        )

    return planets


def _normalize_architecture(architecture_type: str | None) -> str:
    value = (architecture_type or DEFAULT_ARCHITECTURE).strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "auto": DEFAULT_ARCHITECTURE,
        "random": DEFAULT_ARCHITECTURE,
        "solar_like": "solar_like_mixed",
        "mixed": "solar_like_mixed",
        "compact": "compact_rocky_inner",
        "outer_giant": "outer_giant_dominated",
        "quiet": "low_mass_quiet",
        "volatile": "volatile_rich",
        "sparse": "sparse_old",
    }
    return aliases.get(value, value)


def _normalize_preference(preference: str | None) -> str:
    value = (preference or DEFAULT_MAIN_PLANET_PREFERENCE).strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "earth": "earthlike",
        "earth_like": "earthlike",
        "dry": "dry_terrestrial",
        "ocean": "oceanic",
        "ocean_world": "oceanic",
        "super": "super_earth",
        "cold": "colder_world",
        "warm": "warmer_world",
    }
    return aliases.get(value, value)


def _generate_orbital_ladder(
    rng: random.Random,
    star: Star,
    planet_count: int,
    architecture: str,
    preference: str,
) -> tuple[list[float], int]:
    """Generate sorted semi-major axes with an intentional habitable-zone candidate."""
    if planet_count <= 1:
        return [rng.uniform(star.habitable_zone_inner_au, star.habitable_zone_outer_au)], 0

    flux_target = {
        "earthlike": 1.00,
        "dry_terrestrial": 1.08,
        "oceanic": 0.92,
        "super_earth": 0.98,
        "colder_world": 0.78,
        "warmer_world": 1.14,
    }.get(preference, 1.0)
    preferred_anchor = math.sqrt(max(1e-9, star.luminosity_solar / flux_target))
    low = star.habitable_zone_inner_au * 1.015
    high = star.habitable_zone_outer_au * 0.985
    if low >= high:
        low, high = star.habitable_zone_inner_au, star.habitable_zone_outer_au
    anchor = max(low, min(high, preferred_anchor * rng.uniform(0.96, 1.04)))

    if architecture == "compact_rocky_inner":
        anchor_index = min(max(round(planet_count * 0.58), 1), planet_count - 2)
        inner_range = (1.25, 1.72)
        outer_range = (1.32, 1.92)
        outer_snow_range = (1.55, 2.15)
    elif architecture == "sparse_old":
        anchor_index = min(max(planet_count // 2, 1), planet_count - 2)
        inner_range = (1.70, 2.35)
        outer_range = (1.75, 2.65)
        outer_snow_range = (2.00, 3.10)
    elif architecture == "outer_giant_dominated":
        anchor_index = min(max(round(planet_count * 0.45), 1), planet_count - 2)
        inner_range = (1.45, 2.05)
        outer_range = (1.55, 2.25)
        outer_snow_range = (1.90, 2.75)
    elif architecture == "low_mass_quiet":
        anchor_index = min(max(planet_count // 2, 1), planet_count - 2)
        inner_range = (1.40, 1.95)
        outer_range = (1.50, 2.20)
        outer_snow_range = (1.75, 2.45)
    elif architecture == "volatile_rich":
        anchor_index = min(max(round(planet_count * 0.50), 1), planet_count - 2)
        inner_range = (1.38, 1.95)
        outer_range = (1.48, 2.10)
        outer_snow_range = (1.65, 2.35)
    else:
        anchor_index = min(max(planet_count // 2, 1), planet_count - 2)
        inner_range = (1.45, 2.05)
        outer_range = (1.45, 2.10)
        outer_snow_range = (1.70, 2.45)

    axes = [anchor]
    current = anchor
    for _ in range(anchor_index):
        current /= rng.uniform(*inner_range)
        axes.append(max(0.055 if star.stellar_class == "K" else 0.08, current))

    current = anchor
    outer_count = planet_count - anchor_index - 1
    for _ in range(outer_count):
        multiplier = rng.uniform(*(outer_snow_range if current > star.snow_line_au * 0.85 else outer_range))
        current *= multiplier
        axes.append(current)

    if planet_count >= 4 and max(axes) < star.snow_line_au * 1.10:
        axes[axes.index(max(axes))] = star.snow_line_au * rng.uniform(1.15, 2.35)
    if architecture == "outer_giant_dominated" and planet_count >= 5 and max(axes) < star.snow_line_au * 2.0:
        axes[axes.index(max(axes))] = star.snow_line_au * rng.uniform(2.0, 3.4)

    sorted_axes = sorted(axes)
    anchor_idx = min(range(len(sorted_axes)), key=lambda i: abs(sorted_axes[i] - anchor))
    return sorted_axes, anchor_idx


def _choose_planet_class(
    rng: random.Random,
    star: Star,
    semi_major_axis_au: float,
    architecture: str,
    preference: str,
    is_hz_anchor: bool,
) -> str:
    inside_snow_line = semi_major_axis_au < star.snow_line_au
    flux = stellar_flux_earth(star.luminosity_solar, semi_major_axis_au)

    if is_hz_anchor:
        if preference == "super_earth":
            choices = [(PLANET_SUPER_EARTH, 0.66), (PLANET_ROCKY, 0.32), (PLANET_MINI_NEPTUNE, 0.02)]
        else:
            rocky_weight = 0.78 if preference in {"earthlike", "dry_terrestrial", "warmer_world"} else 0.68
            choices = [(PLANET_ROCKY, rocky_weight), (PLANET_SUPER_EARTH, 1.0 - rocky_weight - 0.02), (PLANET_MINI_NEPTUNE, 0.02)]
        return weighted_choice(rng, choices)

    if inside_snow_line:
        if architecture == "compact_rocky_inner":
            choices = [(PLANET_ROCKY, 0.74), (PLANET_SUPER_EARTH, 0.23), (PLANET_MINI_NEPTUNE, 0.03)]
        elif architecture == "low_mass_quiet":
            choices = [(PLANET_ROCKY, 0.82), (PLANET_SUPER_EARTH, 0.15), (PLANET_MINI_NEPTUNE, 0.03)]
        elif 0.65 <= flux <= 1.45:
            choices = [(PLANET_ROCKY, 0.70), (PLANET_SUPER_EARTH, 0.26), (PLANET_MINI_NEPTUNE, 0.04)]
        elif flux > 3.0:
            choices = [(PLANET_ROCKY, 0.78), (PLANET_SUPER_EARTH, 0.18), (PLANET_MINI_NEPTUNE, 0.04)]
        else:
            choices = [(PLANET_ROCKY, 0.58), (PLANET_SUPER_EARTH, 0.32), (PLANET_MINI_NEPTUNE, 0.10)]
    else:
        if architecture == "outer_giant_dominated":
            choices = [(PLANET_GAS_GIANT, 0.42), (PLANET_ICE_GIANT, 0.25), (PLANET_MINI_NEPTUNE, 0.12), (PLANET_ICY_DWARF, 0.10), (PLANET_SUPER_EARTH, 0.07), (PLANET_ROCKY, 0.04)]
        elif architecture == "volatile_rich":
            choices = [(PLANET_ICE_GIANT, 0.28), (PLANET_GAS_GIANT, 0.24), (PLANET_MINI_NEPTUNE, 0.22), (PLANET_SUPER_EARTH, 0.12), (PLANET_ICY_DWARF, 0.09), (PLANET_ROCKY, 0.05)]
        elif architecture == "low_mass_quiet":
            choices = [(PLANET_MINI_NEPTUNE, 0.24), (PLANET_ICY_DWARF, 0.23), (PLANET_SUPER_EARTH, 0.22), (PLANET_ROCKY, 0.16), (PLANET_ICE_GIANT, 0.10), (PLANET_GAS_GIANT, 0.05)]
        else:
            choices = [(PLANET_GAS_GIANT, 0.28), (PLANET_ICE_GIANT, 0.24), (PLANET_MINI_NEPTUNE, 0.18), (PLANET_ICY_DWARF, 0.13), (PLANET_SUPER_EARTH, 0.10), (PLANET_ROCKY, 0.07)]

    if star.metallicity > 0.20 and not inside_snow_line:
        choices = [(name, weight * (1.45 if name in {PLANET_GAS_GIANT, PLANET_ICE_GIANT} else 1.0)) for name, weight in choices]
    elif star.metallicity < -0.25 and not inside_snow_line:
        choices = [(name, weight * (0.65 if name == PLANET_GAS_GIANT else 1.0)) for name, weight in choices]

    return weighted_choice(rng, choices)


def _choose_planet_mass(rng: random.Random, planet_class: str, architecture: str, preference: str, is_hz_anchor: bool) -> float:
    ranges = {
        PLANET_ROCKY: (0.25, 1.7),
        PLANET_SUPER_EARTH: (1.7, 4.4),
        PLANET_MINI_NEPTUNE: (4.0, 14.0),
        PLANET_ICE_GIANT: (10.0, 25.0),
        PLANET_GAS_GIANT: (40.0, 320.0),
        PLANET_ICY_DWARF: (0.02, 0.25),
    }
    low, high = ranges[planet_class]
    if is_hz_anchor:
        if preference == "super_earth" or planet_class == PLANET_SUPER_EARTH:
            low, high = 1.7, 3.4
        elif preference == "dry_terrestrial":
            low, high = 0.55, 1.25
        elif preference == "oceanic":
            low, high = 0.75, 1.9
        else:
            low, high = 0.65, 1.65
    elif architecture == "low_mass_quiet" and planet_class in {PLANET_ROCKY, PLANET_SUPER_EARTH, PLANET_GAS_GIANT}:
        high = low + (high - low) * 0.55
    elif architecture == "outer_giant_dominated" and planet_class == PLANET_GAS_GIANT:
        low, high = 80.0, 360.0
    return rng.uniform(low, high)


def _choose_planet_radius(rng: random.Random, planet_class: str, mass_earth: float) -> float:
    if planet_class == PLANET_ROCKY:
        return mass_earth ** 0.27 * rng.uniform(0.95, 1.08)
    if planet_class == PLANET_SUPER_EARTH:
        return mass_earth ** 0.28 * rng.uniform(1.00, 1.18)
    if planet_class == PLANET_MINI_NEPTUNE:
        return rng.uniform(1.8, 3.8)
    if planet_class == PLANET_ICE_GIANT:
        return rng.uniform(3.5, 4.6)
    if planet_class == PLANET_GAS_GIANT:
        return rng.uniform(8.0, 12.5)
    if planet_class == PLANET_ICY_DWARF:
        return rng.uniform(0.18, 0.55)
    raise ValueError(f"Unsupported planet class: {planet_class}")


def _choose_composition(
    rng: random.Random,
    planet_class: str,
    semi_major_axis_au: float,
    snow_line_au: float,
    architecture: str,
    preference: str,
    is_hz_anchor: bool,
) -> Composition:
    beyond_snow_line = semi_major_axis_au > snow_line_au

    if planet_class in {PLANET_ROCKY, PLANET_SUPER_EARTH}:
        if is_hz_anchor:
            if preference == "dry_terrestrial":
                water = rng.uniform(0.003, 0.012)
            elif preference == "oceanic":
                water = rng.uniform(0.025, 0.070)
            elif preference == "colder_world":
                water = rng.uniform(0.015, 0.060)
            elif preference == "warmer_world":
                water = rng.uniform(0.004, 0.026)
            else:
                water = rng.uniform(0.006, 0.040)
        else:
            upper = 0.035 if not beyond_snow_line else 0.12
            if architecture == "volatile_rich":
                upper *= 1.35
            if architecture == "sparse_old":
                upper *= 0.85
            water = rng.uniform(0.0025, min(0.15, upper))
        iron = rng.uniform(0.22, 0.38)
        silicate = max(0.0, 1.0 - iron - water)
        gas = rng.uniform(0.0, 0.002)
        volatile = "moderate" if 0.002 <= water <= 0.04 else "low" if water < 0.002 else "high"
    elif planet_class == PLANET_MINI_NEPTUNE:
        iron = rng.uniform(0.05, 0.18)
        silicate = rng.uniform(0.20, 0.45)
        water = rng.uniform(0.10, 0.35 if architecture != "volatile_rich" else 0.45)
        gas = max(0.0, 1.0 - iron - silicate - water)
        volatile = "high"
    elif planet_class in {PLANET_ICE_GIANT, PLANET_GAS_GIANT}:
        iron = rng.uniform(0.01, 0.08)
        silicate = rng.uniform(0.04, 0.18)
        water = rng.uniform(0.08, 0.35 if architecture != "volatile_rich" else 0.45)
        gas = max(0.0, 1.0 - iron - silicate - water)
        volatile = "very_high"
    else:
        iron = rng.uniform(0.05, 0.20)
        silicate = rng.uniform(0.20, 0.45)
        water = max(0.0, 1.0 - iron - silicate)
        gas = 0.0
        volatile = "icy"

    total = iron + silicate + water + gas
    iron /= total
    silicate /= total
    water /= total
    gas /= total

    return Composition(
        iron_fraction=iron,
        silicate_fraction=silicate,
        water_ice_fraction=water,
        gas_envelope_fraction=gas,
        volatile_inventory=volatile,
        composition_class=_composition_class(planet_class, water, gas, beyond_snow_line),
    )


def _composition_class(planet_class: str, water: float, gas: float, beyond_snow_line: bool) -> str:
    if planet_class in {PLANET_GAS_GIANT, PLANET_ICE_GIANT}:
        return "volatile-rich giant"
    if planet_class == PLANET_MINI_NEPTUNE:
        return "gas-enveloped volatile world"
    if planet_class == PLANET_ICY_DWARF:
        return "icy minor planet"
    if gas > 0.01:
        return "gas-enveloped rocky world"
    if water < 0.002:
        return "dry rocky world"
    if water <= 0.03 and not beyond_snow_line:
        return "temperate rocky volatile inventory"
    if water <= 0.08:
        return "water-rich rocky world"
    return "ice/water-rich rocky world"


def _max_eccentricity_for_architecture(architecture: str, is_hz_anchor: bool) -> float:
    max_e = MAX_PLANET_ECCENTRICITY
    if architecture == "low_mass_quiet":
        max_e *= 0.55
    elif architecture == "sparse_old":
        max_e *= 0.80
    elif architecture == "outer_giant_dominated":
        max_e *= 1.10
    if is_hz_anchor:
        max_e *= 0.65
    return max(0.0, min(0.11, max_e))


def _architecture_role(star: Star, semi_major_axis_au: float, planet_class: str) -> str:
    if semi_major_axis_au < star.habitable_zone_inner_au * 0.75:
        return "inner_hot_world"
    if star.habitable_zone_inner_au * 0.75 <= semi_major_axis_au <= star.habitable_zone_outer_au * 1.20:
        return "habitable_zone_neighbor"
    if semi_major_axis_au < star.snow_line_au:
        return "temperate_outer_terrestrial"
    if planet_class in {PLANET_GAS_GIANT, PLANET_ICE_GIANT}:
        return "outer_giant"
    return "cold_outer_body"
