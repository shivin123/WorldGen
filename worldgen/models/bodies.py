"""Data models for stars, planets, and moons."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass
class ValueSource:
    """Tracks where a value came from.

    source should be one of: USER_SPECIFIED, RANDOM_GENERATED, DERIVED.
    """

    source: str
    note: str = ""


@dataclass
class Star:
    stellar_class: str
    mass_solar: float
    age_gyr: float
    metallicity: float
    luminosity_solar: float
    radius_solar: float
    temperature_k: float
    main_sequence_lifetime_gyr: float
    habitable_zone_inner_au: float
    habitable_zone_outer_au: float
    snow_line_au: float
    value_sources: dict[str, ValueSource] = field(default_factory=dict)
    stellar_subclass: int | None = None
    spectral_type: str | None = None
    stellar_description: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Composition:
    iron_fraction: float
    silicate_fraction: float
    water_ice_fraction: float
    gas_envelope_fraction: float
    volatile_inventory: str
    composition_class: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Orbit:
    semi_major_axis_au: float
    eccentricity: float
    orbital_period_years: float
    orbital_period_days: float
    periapsis_au: float
    apoapsis_au: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class MoonOrbit:
    semi_major_axis_km: float
    eccentricity: float
    orbital_period_days: float
    periapsis_km: float
    apoapsis_km: float
    roche_limit_km: float
    safe_hill_limit_km: float
    hill_fraction: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Moon:
    name: str
    moon_class: str
    mass_earth: float
    radius_earth: float
    density_relative_earth: float
    composition: Composition
    orbit: MoonOrbit
    surface_gravity_g: float
    tidal_strength_relative_earth_moon: float
    angular_diameter_degrees: float
    value_sources: dict[str, ValueSource] = field(default_factory=dict)
    moon_origin: str = "unknown"
    tidal_effect_level: str = "moderate"
    axial_stability_effect: str = "moderate"
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Planet:
    name: str
    planet_class: str
    mass_earth: float
    radius_earth: float
    density_relative_earth: float
    composition: Composition
    orbit: Orbit
    stellar_flux_earth: float
    equilibrium_temperature_k: float
    surface_gravity_g: float
    escape_velocity_relative_earth: float
    hill_radius_au: float
    habitability_score: float = 0.0
    is_main_planet: bool = False
    moon: Moon | None = None
    selection_notes: list[str] = field(default_factory=list)
    value_sources: dict[str, ValueSource] = field(default_factory=dict)
    formation_context: dict[str, Any] = field(default_factory=dict)
    architecture_role: str = "ordinary_planet"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
