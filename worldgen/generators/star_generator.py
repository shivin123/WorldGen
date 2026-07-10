"""Star generation for single G/K main-sequence systems."""

from __future__ import annotations

import random

from worldgen.config import StarConfig
from worldgen.constants import (
    G_STAR_MAX_MASS,
    G_STAR_MIN_MASS,
    K_STAR_MAX_MASS,
    K_STAR_MIN_MASS,
    MAX_SYSTEM_AGE_GYR,
    MIN_SYSTEM_AGE_GYR,
)
from worldgen.models.bodies import Star, ValueSource
from worldgen.physics.stellar import (
    habitable_zone_au,
    luminosity_from_mass,
    main_sequence_lifetime_gyr,
    radius_from_mass_and_age,
    snow_line_au,
    temperature_from_luminosity_and_radius,
)
from worldgen.random_utils import weighted_choice


def generate_star(rng: random.Random, config: StarConfig) -> Star:
    stellar_class, stellar_class_source = _choose_stellar_class(rng, config.stellar_class)
    mass_solar, mass_source = _choose_mass(rng, stellar_class, config.mass_solar)

    luminosity = luminosity_from_mass(mass_solar)
    lifetime = main_sequence_lifetime_gyr(mass_solar, luminosity)

    age_gyr, age_source = _choose_age(rng, lifetime, config.age_gyr)
    metallicity, metallicity_source = _choose_metallicity(rng, config.metallicity)

    radius = radius_from_mass_and_age(mass_solar, age_gyr, lifetime)
    temperature = temperature_from_luminosity_and_radius(luminosity, radius)
    hz_inner, hz_outer = habitable_zone_au(luminosity)
    snow_line = snow_line_au(luminosity)
    stellar_subclass = _stellar_subclass_from_mass(stellar_class, mass_solar)
    spectral_type = f"{stellar_class}{stellar_subclass}V"
    stellar_description = _stellar_description(stellar_class, stellar_subclass)

    return Star(
        stellar_class=stellar_class,
        mass_solar=mass_solar,
        age_gyr=age_gyr,
        metallicity=metallicity,
        luminosity_solar=luminosity,
        radius_solar=radius,
        temperature_k=temperature,
        main_sequence_lifetime_gyr=lifetime,
        habitable_zone_inner_au=hz_inner,
        habitable_zone_outer_au=hz_outer,
        snow_line_au=snow_line,
        value_sources={
            "stellar_class": stellar_class_source,
            "mass_solar": mass_source,
            "age_gyr": age_source,
            "metallicity": metallicity_source,
            "luminosity_solar": ValueSource("DERIVED", "mass^3.5 approximation"),
            "radius_solar": ValueSource("DERIVED", "mass^0.8 plus age expansion approximation"),
            "temperature_k": ValueSource("DERIVED", "luminosity-radius scaling"),
            "main_sequence_lifetime_gyr": ValueSource("DERIVED", "10 * mass / luminosity"),
            "habitable_zone": ValueSource("DERIVED", "simple luminosity scaling"),
            "snow_line_au": ValueSource("DERIVED", "2.7 * sqrt(luminosity)"),
            "stellar_subclass": ValueSource("DERIVED", "mass position within generated G/K range"),
            "spectral_type": ValueSource("DERIVED", "class + subclass + main-sequence luminosity class"),
        },
        stellar_subclass=stellar_subclass,
        spectral_type=spectral_type,
        stellar_description=stellar_description,
    )


def _choose_stellar_class(rng: random.Random, requested: str | None) -> tuple[str, ValueSource]:
    if requested is not None:
        stellar_class = requested.upper().strip()
        if len(stellar_class) >= 1 and stellar_class[0] in {"G", "K"}:
            # Accept both broad classes (G/K) and UI-friendly labels such as G2V or K5.
            return stellar_class[0], ValueSource("USER_SPECIFIED", f"requested {requested}")
        raise ValueError("Only G and K stars are supported in this version.")

    # K stars are more common and long-lived, so weight them higher.
    return weighted_choice(rng, [("K", 0.65), ("G", 0.35)]), ValueSource("RANDOM_GENERATED")


def _choose_mass(rng: random.Random, stellar_class: str, requested: float | None) -> tuple[float, ValueSource]:
    min_mass, max_mass = (K_STAR_MIN_MASS, K_STAR_MAX_MASS) if stellar_class == "K" else (G_STAR_MIN_MASS, G_STAR_MAX_MASS)

    if requested is not None:
        if not (min_mass <= requested <= max_mass):
            raise ValueError(f"{stellar_class}-class mass must be between {min_mass} and {max_mass} solar masses.")
        return requested, ValueSource("USER_SPECIFIED")

    return rng.uniform(min_mass, max_mass), ValueSource("RANDOM_GENERATED")


def _choose_age(rng: random.Random, lifetime_gyr: float, requested: float | None) -> tuple[float, ValueSource]:
    max_age = min(MAX_SYSTEM_AGE_GYR, lifetime_gyr * 0.92)
    min_age = min(MIN_SYSTEM_AGE_GYR, max_age)

    if requested is not None:
        if requested <= 0:
            raise ValueError("Star age must be positive.")
        if requested >= lifetime_gyr:
            raise ValueError("Star age exceeds main-sequence lifetime; evolved stars are not supported yet.")
        return requested, ValueSource("USER_SPECIFIED")

    return rng.uniform(min_age, max_age), ValueSource("RANDOM_GENERATED")


def _choose_metallicity(rng: random.Random, requested: float | None) -> tuple[float, ValueSource]:
    if requested is not None:
        return requested, ValueSource("USER_SPECIFIED")

    # [Fe/H]-like value. Keep conservative for planet-friendly systems.
    return rng.gauss(0.0, 0.18), ValueSource("RANDOM_GENERATED", "Gaussian around solar metallicity")


def _stellar_subclass_from_mass(stellar_class: str, mass_solar: float) -> int:
    """Approximate G0-G9/K0-K9 subclass from mass within the supported range."""
    if stellar_class == "G":
        min_mass, max_mass = G_STAR_MIN_MASS, G_STAR_MAX_MASS
    else:
        min_mass, max_mass = K_STAR_MIN_MASS, K_STAR_MAX_MASS
    # Hotter/brighter subtypes have lower subclass numbers.
    position = (max_mass - mass_solar) / max(1e-9, max_mass - min_mass)
    return int(max(0, min(9, round(position * 9.0))))


def _stellar_description(stellar_class: str, subclass: int) -> str:
    if stellar_class == "G":
        if subclass <= 2:
            return "warm yellow main-sequence star"
        if subclass <= 6:
            return "Sun-like yellow main-sequence star"
        return "cool yellow main-sequence star"
    if subclass <= 2:
        return "warm orange main-sequence star"
    if subclass <= 6:
        return "orange main-sequence star"
    return "cool orange main-sequence star"
