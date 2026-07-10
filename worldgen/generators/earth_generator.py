"""Earth preset generation.

Both Earth presets use real Sun/Earth/Moon values and real-world terrain data
through Stage 3.  They are calibration/test worlds: terrain is fixed from the
Earth DEM loader, while later climate, hydrology, biome, and regional stages run
through the WorldGen simulation stack.
"""

from __future__ import annotations

import random

from worldgen.config import WorldGenConfig
from worldgen.constants import (
    AU_KM,
    MOON_MASS_EARTH,
    MOON_ORBIT_KM,
    MOON_RADIUS_EARTH,
    PLANET_GAS_GIANT,
    PLANET_ICE_GIANT,
    PLANET_ROCKY,
)
from worldgen.generators.biome_generator import generate_biomes
from worldgen.generators.climate_generator import generate_climate
from worldgen.generators.hydrology_generator import generate_hydrology
from worldgen.generators.real_earth_terrain import load_real_earth_terrain
from worldgen.generators.region_generator import generate_regions
from worldgen.models.bodies import Composition, Moon, MoonOrbit, Orbit, Planet, Star, ValueSource
from worldgen.models.planet_profile import Atmosphere, GeologyState, Hydrosphere, MainPlanetProfile, RotationState
from worldgen.models.system import StarSystem
from worldgen.physics.orbital import (
    hill_radius_au,
    moon_orbital_period_days,
    orbital_period_days,
    orbital_period_years,
    periapsis_apoapsis,
    roche_limit_km,
)
from worldgen.physics.planetary import (
    angular_diameter_degrees,
    density_relative_earth,
    equilibrium_temperature_k,
    escape_velocity_relative_earth,
    stellar_flux_earth,
    surface_gravity_g,
    tidal_strength_relative_earth_moon,
)


def generate_synthetic_earth_system(rng: random.Random, config: WorldGenConfig) -> StarSystem:
    """Generate a calibration world using real Earth/Sun/Moon characteristics."""
    star = _sun()
    planets = _solar_system_planets(star)
    earth = next(planet for planet in planets if planet.name == "Earth")
    earth.is_main_planet = True
    earth.habitability_score = 100.0
    earth.selection_notes = [
        "Synthetic Earth preset: real Earth bulk/orbital values",
        "uses real-world Earth terrain data through Stage 3; procedural simulation resumes after terrain",
    ]
    earth.moon = _moon(earth)
    earth.architecture_role = "habitable_zone_candidate"
    earth.formation_context = {
        "architecture": "solar_like_mixed",
        "main_planet_preference": "earthlike",
        "formation_zone": "habitable_zone",
        "volatile_delivery": "moderate",
        "giant_planet_influence": "moderate",
        "impact_history": "normal",
        "tectonic_energy_bias": "earth_like",
        "crustal_asymmetry_bias": "low",
        "moon_origin": earth.moon.moon_origin,
        "tidal_effect_level": earth.moon.tidal_effect_level,
        "axial_stability_effect": earth.moon.axial_stability_effect,
    }

    profile = _earth_profile(rng, star, earth, config, terrain_mode="synthetic")

    notes = [
        "Preset: Synthetic Earth",
        "Uses real Sun/Earth/Moon physical and orbital properties.",
        "Terrain source is the same Real Earth DEM loader used by real-earth-terrain mode, through Stage 3.",
        "Use the validation report to tune climate, hydrology, and biome behavior after the fixed terrain stage.",
    ]
    return StarSystem(
        seed=config.seed,
        star=star,
        planets=planets,
        notes=notes,
        main_planet_profile=profile,
        architecture="solar_like_mixed",
        diagnostics={
            "architecture": "solar_like_mixed",
            "main_planet_preference": "earthlike",
            "planet_count": len(planets),
            "habitable_zone_planet_count": 1,
            "outer_giant_count": 4,
            "giant_planet_influence": "moderate",
            "main_planet_candidate_quality": "strong",
            "climate_stability_outlook": "favorable",
        },
    )



def generate_real_earth_terrain_system(rng: random.Random, config: WorldGenConfig) -> StarSystem:
    """Generate a calibration world using Earth values and bundled real Earth terrain."""
    star = _sun()
    planets = _solar_system_planets(star)
    earth = next(planet for planet in planets if planet.name == "Earth")
    earth.is_main_planet = True
    earth.habitability_score = 100.0
    earth.selection_notes = [
        "Real Earth terrain preset: real Earth bulk/orbital values",
        "uses the Real Earth DEM loader: external ETOPO-style NPZ when provided, otherwise bundled fallback terrain",
        "uses the same climate calculation as procedural worlds so diagnostics reveal calibration errors",
    ]
    earth.moon = _moon(earth)
    earth.architecture_role = "habitable_zone_candidate"
    earth.formation_context = {
        "architecture": "solar_like_mixed",
        "main_planet_preference": "earthlike",
        "formation_zone": "habitable_zone",
        "volatile_delivery": "moderate",
        "giant_planet_influence": "moderate",
        "impact_history": "normal",
        "tectonic_energy_bias": "earth_like",
        "crustal_asymmetry_bias": "low",
        "moon_origin": earth.moon.moon_origin,
        "tidal_effect_level": earth.moon.tidal_effect_level,
        "axial_stability_effect": earth.moon.axial_stability_effect,
    }
    profile = _earth_profile(rng, star, earth, config, terrain_mode="real")
    notes = [
        "Preset: Real Earth Terrain",
        "Uses real Sun/Earth/Moon physical and orbital properties.",
        "Terrain source is loaded by the Real Earth DEM loader: WORLDGEN_EARTH_DEM_NPZ or packaged ETOPO-style NPZ when available, otherwise bundled fallback terrain.",
        "Real Earth climate now uses the same model path as procedural worlds; no Earth-specific regional climate nudges are applied.",
    ]
    return StarSystem(
        seed=config.seed,
        star=star,
        planets=planets,
        notes=notes,
        main_planet_profile=profile,
        architecture="solar_like_mixed",
        diagnostics={
            "architecture": "solar_like_mixed",
            "main_planet_preference": "earthlike",
            "planet_count": len(planets),
            "habitable_zone_planet_count": 1,
            "outer_giant_count": 4,
            "giant_planet_influence": "moderate",
            "main_planet_candidate_quality": "strong",
            "climate_stability_outlook": "favorable",
        },
    )

def _sun() -> Star:
    return Star(
        stellar_class="G",
        stellar_subclass=2,
        spectral_type="G2V",
        stellar_description="Sun-like yellow main-sequence star",
        mass_solar=1.0,
        age_gyr=4.57,
        metallicity=0.0,
        luminosity_solar=1.0,
        radius_solar=1.0,
        temperature_k=5772.0,
        main_sequence_lifetime_gyr=10.0,
        habitable_zone_inner_au=(1.0 / 1.1) ** 0.5,
        habitable_zone_outer_au=(1.0 / 0.53) ** 0.5,
        snow_line_au=2.7,
        value_sources={
            "mass_solar": ValueSource("USER_SPECIFIED", "Synthetic Earth preset Sun value"),
            "luminosity_solar": ValueSource("USER_SPECIFIED", "Synthetic Earth preset Sun value"),
            "radius_solar": ValueSource("USER_SPECIFIED", "Synthetic Earth preset Sun value"),
            "age_gyr": ValueSource("USER_SPECIFIED", "Solar age approximation"),
        },
    )


def _solar_system_planets(star: Star) -> list[Planet]:
    specs = [
        ("Mercury", PLANET_ROCKY, 0.0553, 0.383, 0.387, 0.206, _rocky_composition(0.70, 0.30, 0.0001, "metal-rich rocky world")),
        ("Venus", PLANET_ROCKY, 0.815, 0.949, 0.723, 0.007, _rocky_composition(0.30, 0.70, 0.0002, "dry rocky world")),
        ("Earth", PLANET_ROCKY, 1.0, 1.0, 1.0, 0.017, _rocky_composition(0.32, 0.67, 0.010, "ocean-bearing rocky world")),
        ("Mars", PLANET_ROCKY, 0.107, 0.532, 1.524, 0.093, _rocky_composition(0.25, 0.75, 0.001, "cold dry rocky world")),
        ("Jupiter", PLANET_GAS_GIANT, 317.8, 11.21, 5.203, 0.049, _giant_composition("hydrogen-helium gas giant")),
        ("Saturn", PLANET_GAS_GIANT, 95.2, 9.45, 9.537, 0.057, _giant_composition("hydrogen-helium gas giant")),
        ("Uranus", PLANET_ICE_GIANT, 14.5, 4.01, 19.191, 0.046, _giant_composition("ice giant")),
        ("Neptune", PLANET_ICE_GIANT, 17.1, 3.88, 30.07, 0.010, _giant_composition("ice giant")),
    ]
    return [_planet_from_spec(star, *spec) for spec in specs]


def _planet_from_spec(
    star: Star,
    name: str,
    planet_class: str,
    mass: float,
    radius: float,
    semi_major_axis: float,
    eccentricity: float,
    composition: Composition,
) -> Planet:
    period_years = orbital_period_years(semi_major_axis, star.mass_solar)
    period_days = orbital_period_days(semi_major_axis, star.mass_solar)
    periapsis, apoapsis = periapsis_apoapsis(semi_major_axis, eccentricity)
    flux = stellar_flux_earth(star.luminosity_solar, semi_major_axis)
    density = density_relative_earth(mass, radius)
    return Planet(
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
        surface_gravity_g=surface_gravity_g(mass, radius),
        escape_velocity_relative_earth=escape_velocity_relative_earth(mass, radius),
        hill_radius_au=hill_radius_au(semi_major_axis, mass, star.mass_solar, eccentricity),
        value_sources={
            "physical_values": ValueSource("USER_SPECIFIED", "Synthetic Earth preset / Solar System calibration values"),
            "derived_values": ValueSource("DERIVED", "Computed from preset masses, radii, and orbits"),
        },
    )


def _rocky_composition(iron: float, silicate: float, water: float, label: str) -> Composition:
    gas = 0.0
    total = iron + silicate + water + gas
    return Composition(
        iron_fraction=iron / total,
        silicate_fraction=silicate / total,
        water_ice_fraction=water / total,
        gas_envelope_fraction=gas,
        volatile_inventory="moderate" if water >= 0.003 else "low",
        composition_class=label,
    )


def _giant_composition(label: str) -> Composition:
    return Composition(
        iron_fraction=0.03,
        silicate_fraction=0.07,
        water_ice_fraction=0.18,
        gas_envelope_fraction=0.72,
        volatile_inventory="very_high",
        composition_class=label,
    )


def _moon(earth: Planet) -> Moon:
    moon_density = density_relative_earth(MOON_MASS_EARTH, MOON_RADIUS_EARTH)
    roche = roche_limit_km(earth.radius_earth, earth.density_relative_earth, moon_density, fluid_body=True)
    safe_hill = earth.hill_radius_au * AU_KM * 0.33
    period = moon_orbital_period_days(MOON_ORBIT_KM, earth.mass_earth, MOON_MASS_EARTH)
    eccentricity = 0.0549
    return Moon(
        name="Moon",
        moon_class="rocky_moon",
        mass_earth=MOON_MASS_EARTH,
        radius_earth=MOON_RADIUS_EARTH,
        density_relative_earth=moon_density,
        composition=Composition(
            iron_fraction=0.08,
            silicate_fraction=0.92,
            water_ice_fraction=0.0,
            gas_envelope_fraction=0.0,
            volatile_inventory="very_low",
            composition_class="rocky silicate moon",
        ),
        orbit=MoonOrbit(
            semi_major_axis_km=MOON_ORBIT_KM,
            eccentricity=eccentricity,
            orbital_period_days=period,
            periapsis_km=MOON_ORBIT_KM * (1.0 - eccentricity),
            apoapsis_km=MOON_ORBIT_KM * (1.0 + eccentricity),
            roche_limit_km=roche,
            safe_hill_limit_km=safe_hill,
            hill_fraction=MOON_ORBIT_KM / (earth.hill_radius_au * AU_KM),
        ),
        surface_gravity_g=surface_gravity_g(MOON_MASS_EARTH, MOON_RADIUS_EARTH),
        tidal_strength_relative_earth_moon=tidal_strength_relative_earth_moon(MOON_MASS_EARTH, MOON_ORBIT_KM, earth.radius_earth),
        angular_diameter_degrees=angular_diameter_degrees(MOON_RADIUS_EARTH, MOON_ORBIT_KM),
        moon_origin="giant_impact",
        tidal_effect_level="moderate",
        axial_stability_effect="high",
        notes=[
            "Earth calibration moon: giant-impact origin assumption.",
            "Provides strong axial-stability context for climate calibration.",
        ],
        value_sources={
            "physical_values": ValueSource("USER_SPECIFIED", "Synthetic Earth preset Moon values"),
            "orbital_period": ValueSource("DERIVED", "Computed from Earth and Moon mass and distance"),
        },
    )


def _earth_profile(rng: random.Random, star: Star, earth: Planet, config: WorldGenConfig, terrain_mode: str) -> MainPlanetProfile:
    rotation = RotationState(
        rotation_period_hours=23.934,
        axial_tilt_degrees=23.44,
        solar_day_hours=24.0,
        year_length_days=365.256,
    )
    atmosphere = Atmosphere(
        pressure_bar=1.0,
        nitrogen_fraction=0.7808,
        oxygen_fraction=0.2095,
        carbon_dioxide_ppm=420.0,
        argon_fraction=0.0093,
        water_vapor_factor=1.0,
        greenhouse_warming_k=33.0,
        estimated_mean_surface_temp_k=288.15,
        estimated_mean_surface_temp_c=15.0,
        notes=[
            "Synthetic Earth preset atmosphere.",
            "CO2 value is a modern rounded calibration value, not a fixed geologic baseline.",
        ],
    )
    # Both Earth presets use the same real-world terrain through Stage 3. Keep
    # the global baseline close to the canonical modern Earth mean and let the
    # normal climate model produce downstream diagnostics.
    atmosphere.estimated_mean_surface_temp_c = 15.0
    atmosphere.estimated_mean_surface_temp_k = 288.15

    hydrosphere = Hydrosphere(
        volatile_fraction=earth.composition.water_ice_fraction,
        ocean_fraction_target=0.708,
        ocean_fraction_actual=0.0,
        water_inventory_class="Earth-like ocean world",
        ice_cap_tendency="Earth-like polar ice tendency",
    )
    geology = GeologyState(
        internal_heat=1.0,
        volcanism=0.75,
        erosion=1.25,
        mountain_factor=1.0,
        crater_density=0.20,
        surface_roughness=0.70,
        geology_class="Earth-like active terrestrial world",
    )
    terrain = load_real_earth_terrain(
        target_width=config.planet_profile.map_width,
        target_height=config.planet_profile.map_height,
    )
    terrain.source = f"{terrain.source}; preset={terrain_mode}; real-world data through Stage 3 terrain"
    hydrosphere.ocean_fraction_actual = terrain.ocean_fraction
    climate = generate_climate(
        rotation,
        atmosphere,
        terrain,
        use_accelerated=not config.planet_profile.no_accelerated_climate,
        koppen_detail=config.planet_profile.koppen_detail,
    )
    hydrology = generate_hydrology(terrain, climate)
    biomes = generate_biomes(terrain, climate, hydrology)
    regions = generate_regions(terrain, climate, hydrology, biomes)
    return MainPlanetProfile(
        planet_name="Earth",
        rotation=rotation,
        atmosphere=atmosphere,
        hydrosphere=hydrosphere,
        geology=geology,
        terrain=terrain,
        climate=climate,
        hydrology=hydrology,
        biomes=biomes,
        regions=regions,
        notes=[
            "Earth calibration mode uses real Earth/Sun/Moon characteristics.",
            "Terrain mode: real-world Earth terrain data through Stage 3 (preset=" + terrain_mode + ")",
        ],
    )
