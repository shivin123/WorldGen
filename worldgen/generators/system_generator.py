"""Top-level star system generator."""

from __future__ import annotations

import math
import random
from collections import Counter
from typing import Any

from worldgen.config import WorldGenConfig
from worldgen.constants import (
    MAIN_PLANET_MAX_GRAVITY_G,
    MAIN_PLANET_MAX_STELLAR_FLUX,
    MIN_MUTUAL_HILL_SEPARATION,
    MAIN_PLANET_MIN_GRAVITY_G,
    MAIN_PLANET_MIN_STELLAR_FLUX,
    MAX_MAIN_PLANET_ECCENTRICITY,
    PLANET_GAS_GIANT,
    PLANET_ICE_GIANT,
    PLANET_MINI_NEPTUNE,
    PLANET_ROCKY,
    PLANET_SUPER_EARTH,
)
from worldgen.generators.moon_generator import generate_main_planet_moon
from worldgen.generators.planet_generator import generate_planets
from worldgen.generators.planet_profile_generator import generate_main_planet_profile
from worldgen.generators.star_generator import generate_star
from worldgen.models.bodies import Planet
from worldgen.models.system import StarSystem
from worldgen.physics.orbital import mutual_hill_radius_au
from worldgen.random_utils import weighted_choice
from worldgen.validation.orbital_validation import validate_planet_spacing


MAIN_PLANET_MIN_EQUILIBRIUM_TEMP_K = 225.0
MAIN_PLANET_MAX_EQUILIBRIUM_TEMP_K = 272.0
MAIN_PLANET_MIN_WATER_FRACTION = 0.003
MAIN_PLANET_MAX_WATER_FRACTION = 0.080
MAIN_PLANET_MIN_ESCAPE_VELOCITY_EARTH = 0.75

SYSTEM_ARCHITECTURES = {
    "compact_rocky_inner",
    "solar_like_mixed",
    "outer_giant_dominated",
    "low_mass_quiet",
    "volatile_rich",
    "sparse_old",
}

MAIN_PLANET_PREFERENCES = {
    "earthlike",
    "dry_terrestrial",
    "oceanic",
    "super_earth",
    "colder_world",
    "warmer_world",
}


def generate_star_system_shell(rng: random.Random, config: WorldGenConfig, include_moon: bool = True) -> StarSystem:
    """Generate star, planets, Main Planet selection, and optionally the moon.

    This stage deliberately stops before the expensive Main Planet profile. It is
    used by the staged pipeline so users can inspect/edit the orbital system and
    generate orbit/size maps before terrain generation begins.
    """
    star = generate_star(rng, config.star)
    architecture = _choose_system_architecture(rng, star, config)
    preference = _normalize_preference(config.system.main_planet_preference)
    requested_planet_count = _choose_planet_count(rng, config, architecture)

    notes: list[str] = [
        f"System architecture: {architecture.replace('_', ' ')}.",
        f"Main Planet preference: {preference.replace('_', ' ')}.",
    ]
    planets: list[Planet] | None = None
    last_errors: list[str] = []
    total_attempts = 0
    best_candidate_snapshot: dict[str, Any] = {}

    # High-count systems with massive outer planets can fail the mutual-Hill
    # spacing check repeatedly, especially when the UI asks for a dense explicit
    # planet count. Keep the requested count as the first target, but do not abort
    # the whole run: first try to prune only the bodies that break spacing, then
    # progressively step down if no eligible Main Planet survives.
    count_floor = 4 if requested_planet_count >= 4 else 1
    count_options = list(range(requested_planet_count, count_floor - 1, -1))
    accepted_repair_note = ""

    for planet_count in count_options:
        for attempt in range(1, 401):
            total_attempts += 1
            candidate_planets = generate_planets(rng, star, planet_count, architecture, preference)
            validation = validate_planet_spacing(star, candidate_planets)
            repair_note = ""
            if not validation.is_valid:
                last_errors = validation.messages
                repaired, removed = _repair_planet_spacing(star, candidate_planets)
                repaired_validation = validate_planet_spacing(star, repaired)
                if not repaired_validation.is_valid:
                    last_errors = repaired_validation.messages or validation.messages
                    continue
                candidate_planets = repaired
                if removed:
                    repair_note = (
                        f"Repaired dense orbital spacing by removing {len(removed)} unstable body/bodies "
                        f"({', '.join(removed[:8])}{'...' if len(removed) > 8 else ''})."
                    )

            selected, selection_error, candidate_snapshot = _select_main_planet(candidate_planets, preference)
            if candidate_snapshot:
                best_candidate_snapshot = candidate_snapshot
            if not selected:
                last_errors = [selection_error]
                continue

            planets = candidate_planets
            accepted_repair_note = repair_note
            notes.append(f"Planet spacing and liquid-water Main Planet eligibility accepted after {total_attempts} attempt(s).")
            if accepted_repair_note:
                notes.append(accepted_repair_note)
            if len(candidate_planets) != requested_planet_count:
                notes.append(
                    f"Final stable system contains {len(candidate_planets)} planet(s) instead of requested {requested_planet_count}; "
                    "the UI request was treated as a target, not a hard failure condition."
                )
            elif planet_count != requested_planet_count:
                notes.append(f"Reduced planet count from {requested_planet_count} to {planet_count} to satisfy stability and Main Planet constraints.")
            break
        if planets is not None:
            break

    if planets is None:
        raise RuntimeError(
            "Could not generate a stable system with a liquid-water eligible Main Planet. "
            + "Last errors: "
            + "; ".join(last_errors[:4])
        )

    main_planet = next(planet for planet in planets if planet.is_main_planet)
    if include_moon and config.system.require_major_moon:
        main_planet.moon = generate_main_planet_moon(rng, star, main_planet, config.system.moon_strength_preference)
        notes.append(
            f"Generated one {main_planet.moon.tidal_effect_level}-tide moon for Main Planet {main_planet.name} "
            f"using Roche/Hill limits; origin class {main_planet.moon.moon_origin}."
        )
    elif not config.system.require_major_moon:
        notes.append("Major moon generation disabled by system.require_major_moon=false.")

    _annotate_formation_context(planets, star, architecture, preference)
    diagnostics = _system_diagnostics(star, planets, architecture, preference, total_attempts, best_candidate_snapshot)
    notes.extend(_diagnostic_notes(diagnostics))

    return StarSystem(
        seed=config.seed,
        star=star,
        planets=planets,
        notes=notes,
        main_planet_profile=None,
        architecture=architecture,
        diagnostics=diagnostics,
    )


def generate_main_planet_profile_for_system(rng: random.Random, system: StarSystem, config: WorldGenConfig) -> StarSystem:
    """Attach the expensive Main Planet profile to an existing system shell."""
    main_planet = system.main_planet
    if main_planet is None:
        raise RuntimeError("Cannot generate Main Planet profile; no selected Main Planet exists in the system state.")
    if main_planet.moon is None and config.system.require_major_moon:
        main_planet.moon = generate_main_planet_moon(rng, system.star, main_planet, config.system.moon_strength_preference)
        system.notes.append(f"Generated one moon for Main Planet {main_planet.name} using Roche/Hill limits.")
    profile = generate_main_planet_profile(rng, system.star, main_planet, config.planet_profile)
    system.main_planet_profile = profile
    system.notes.append(f"Generated Main Planet physical profile, {profile.terrain.width}x{profile.terrain.height} terrain grid, climate, hydrology, biome, and regional-analysis layers.")
    return system


def generate_star_system(rng: random.Random, config: WorldGenConfig) -> StarSystem:
    system = generate_star_system_shell(rng, config, include_moon=True)
    return generate_main_planet_profile_for_system(rng, system, config)


def _choose_system_architecture(rng: random.Random, star, config: WorldGenConfig) -> str:
    requested = config.system.architecture_type
    if requested:
        requested_text = str(requested).strip().lower().replace("-", "_").replace(" ", "_")
        if requested_text not in {"", "random", "auto"}:
            normalized = _normalize_architecture(requested)
            if normalized not in SYSTEM_ARCHITECTURES:
                raise ValueError(f"Unknown system architecture '{requested}'. Valid options: {', '.join(sorted(SYSTEM_ARCHITECTURES))}")
            return normalized

    choices = [
        ("solar_like_mixed", 0.31),
        ("compact_rocky_inner", 0.20),
        ("outer_giant_dominated", 0.17),
        ("low_mass_quiet", 0.14),
        ("volatile_rich", 0.11),
        ("sparse_old", 0.07),
    ]
    if star.stellar_class == "K":
        choices = [(name, weight * (1.25 if name in {"compact_rocky_inner", "low_mass_quiet"} else 1.0)) for name, weight in choices]
    if star.metallicity > 0.20:
        choices = [(name, weight * (1.35 if name in {"outer_giant_dominated", "volatile_rich"} else 1.0)) for name, weight in choices]
    if star.age_gyr > 6.5:
        choices = [(name, weight * (1.45 if name == "sparse_old" else 1.0)) for name, weight in choices]
    return weighted_choice(rng, choices)


def _normalize_architecture(value: str | None) -> str:
    text = (value or "solar_like_mixed").strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "auto": "solar_like_mixed",
        "random": "solar_like_mixed",
        "solar_like": "solar_like_mixed",
        "mixed": "solar_like_mixed",
        "compact": "compact_rocky_inner",
        "outer_giant": "outer_giant_dominated",
        "quiet": "low_mass_quiet",
        "volatile": "volatile_rich",
        "sparse": "sparse_old",
    }
    return aliases.get(text, text)


def _normalize_preference(value: str | None) -> str:
    text = (value or "earthlike").strip().lower().replace("-", "_").replace(" ", "_")
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
    normalized = aliases.get(text, text)
    if normalized not in MAIN_PLANET_PREFERENCES:
        raise ValueError(f"Unknown Main Planet preference '{value}'. Valid options: {', '.join(sorted(MAIN_PLANET_PREFERENCES))}")
    return normalized


def _choose_planet_count(rng: random.Random, config: WorldGenConfig, architecture: str) -> int:
    if config.system.planet_count is not None:
        if config.system.planet_count < 1:
            raise ValueError("planet_count must be at least 1")
        return config.system.planet_count

    min_p = config.system.min_planets
    max_p = config.system.max_planets
    if architecture == "sparse_old":
        max_p = min(max_p, 7)
    elif architecture == "compact_rocky_inner":
        min_p = max(min_p, 5)
    elif architecture == "outer_giant_dominated":
        min_p = max(min_p, 5)
    elif architecture == "low_mass_quiet":
        max_p = min(max_p, 8)
    return rng.randint(min_p, max(min_p, max_p))


def _repair_planet_spacing(star, planets: list[Planet], min_remaining: int = 4) -> tuple[list[Planet], list[str]]:
    """Prune the least important body from invalid close pairs until spacing is valid.

    This is deliberately conservative: it does not mutate or shift orbits, so all
    derived orbital quantities remain internally consistent. The habitable-zone
    candidate is protected whenever possible.
    """
    working = list(planets)
    removed: list[str] = []

    def spacing_ratio(inner: Planet, outer: Planet) -> float:
        separation = outer.orbit.semi_major_axis_au - inner.orbit.semi_major_axis_au
        mutual = mutual_hill_radius_au(
            inner.mass_earth,
            outer.mass_earth,
            inner.orbit.semi_major_axis_au,
            outer.orbit.semi_major_axis_au,
            star.mass_solar,
        )
        return separation / mutual if mutual > 0 else 0.0

    while len(working) > max(1, min_remaining):
        ordered = sorted(working, key=lambda p: p.orbit.semi_major_axis_au)
        close_pairs: list[tuple[float, Planet, Planet]] = []
        for inner, outer in zip(ordered, ordered[1:]):
            ratio = spacing_ratio(inner, outer)
            if ratio < MIN_MUTUAL_HILL_SEPARATION:
                close_pairs.append((ratio, inner, outer))
        if not close_pairs:
            break
        _, inner, outer = min(close_pairs, key=lambda item: item[0])
        loser = _spacing_prune_loser(inner, outer)
        if loser not in working:
            loser = outer
        working.remove(loser)
        removed.append(loser.name)

    return working, removed


def _spacing_prune_loser(a: Planet, b: Planet) -> Planet:
    def importance(p: Planet) -> float:
        value = 0.0
        if p.architecture_role == "habitable_zone_candidate":
            value += 1000.0
        if p.planet_class in {PLANET_ROCKY, PLANET_SUPER_EARTH}:
            value += 120.0
        if MAIN_PLANET_MIN_STELLAR_FLUX <= p.stellar_flux_earth <= MAIN_PLANET_MAX_STELLAR_FLUX:
            value += 100.0
        if MAIN_PLANET_MIN_GRAVITY_G <= p.surface_gravity_g <= MAIN_PLANET_MAX_GRAVITY_G:
            value += 25.0
        if p.planet_class in {PLANET_GAS_GIANT, PLANET_ICE_GIANT}:
            value -= min(80.0, p.mass_earth / 4.0)
        value -= p.orbit.eccentricity * 10.0
        return value

    return a if importance(a) < importance(b) else b


def _select_main_planet(planets: list[Planet], preference: str = "earthlike") -> tuple[bool, str, dict[str, Any]]:
    """Score planets and select only from hard liquid-water candidates."""
    scored: list[tuple[float, Planet]] = []
    rejection_summaries: list[str] = []
    best_snapshot: dict[str, Any] = {}

    for planet in planets:
        planet.is_main_planet = False
        score, notes = _habitability_score(planet, preference)
        issues = _liquid_water_eligibility_issues(planet)
        if not best_snapshot or score > float(best_snapshot.get("score", -1)):
            best_snapshot = {
                "name": planet.name,
                "class": planet.planet_class,
                "score": round(score, 3),
                "issues": issues,
                "flux": round(planet.stellar_flux_earth, 3),
                "gravity": round(planet.surface_gravity_g, 3),
            }
        if issues:
            planet.habitability_score = min(score, 25.0)
            planet.selection_notes = notes + ["Rejected as Main Planet: " + "; ".join(issues)]
            rejection_summaries.append(f"{planet.name}: " + "; ".join(issues[:2]))
            continue

        # Eligible candidates get a meaningful score boost so the selector strongly
        # prefers planets that can plausibly support liquid water and oceans.
        score = max(score, 55.0)
        planet.habitability_score = score
        planet.selection_notes = notes + ["passes hard liquid-water Main Planet eligibility checks"]
        scored.append((score, planet))

    if not scored:
        return False, "No planet passed hard liquid-water Main Planet eligibility. " + " | ".join(rejection_summaries[:4]), best_snapshot

    best_score, best_planet = max(scored, key=lambda item: item[0])
    best_planet.is_main_planet = True
    return True, "", best_snapshot


def _liquid_water_eligibility_issues(planet: Planet) -> list[str]:
    """Hard filters for selecting the deep-dive Main Planet."""
    issues: list[str] = []
    if planet.planet_class not in {PLANET_ROCKY, PLANET_SUPER_EARTH}:
        issues.append(f"class {planet.planet_class} is not rocky/super-Earth")
    if not (MAIN_PLANET_MIN_STELLAR_FLUX <= planet.stellar_flux_earth <= MAIN_PLANET_MAX_STELLAR_FLUX):
        issues.append(f"stellar flux {planet.stellar_flux_earth:.2f} F⊕ outside {MAIN_PLANET_MIN_STELLAR_FLUX:.2f}-{MAIN_PLANET_MAX_STELLAR_FLUX:.2f}")
    if not (MAIN_PLANET_MIN_EQUILIBRIUM_TEMP_K <= planet.equilibrium_temperature_k <= MAIN_PLANET_MAX_EQUILIBRIUM_TEMP_K):
        issues.append(f"equilibrium temp {planet.equilibrium_temperature_k:.1f} K outside {MAIN_PLANET_MIN_EQUILIBRIUM_TEMP_K:.0f}-{MAIN_PLANET_MAX_EQUILIBRIUM_TEMP_K:.0f} K")
    if not (MAIN_PLANET_MIN_GRAVITY_G <= planet.surface_gravity_g <= MAIN_PLANET_MAX_GRAVITY_G):
        issues.append(f"gravity {planet.surface_gravity_g:.2f} g outside {MAIN_PLANET_MIN_GRAVITY_G:.2f}-{MAIN_PLANET_MAX_GRAVITY_G:.2f} g")
    if planet.orbit.eccentricity > MAX_MAIN_PLANET_ECCENTRICITY:
        issues.append(f"eccentricity {planet.orbit.eccentricity:.3f} above {MAX_MAIN_PLANET_ECCENTRICITY:.2f}")
    if not (MAIN_PLANET_MIN_WATER_FRACTION <= planet.composition.water_ice_fraction <= MAIN_PLANET_MAX_WATER_FRACTION):
        issues.append(f"water/volatile fraction {planet.composition.water_ice_fraction:.4f} outside {MAIN_PLANET_MIN_WATER_FRACTION:.3f}-{MAIN_PLANET_MAX_WATER_FRACTION:.3f}")
    if planet.escape_velocity_relative_earth < MAIN_PLANET_MIN_ESCAPE_VELOCITY_EARTH:
        issues.append(f"escape velocity {planet.escape_velocity_relative_earth:.2f} Earth below {MAIN_PLANET_MIN_ESCAPE_VELOCITY_EARTH:.2f}")
    return issues


def _habitability_score(planet: Planet, preference: str = "earthlike") -> tuple[float, list[str]]:
    """Return a normalized 0-100 Main Planet suitability score."""
    score = 0.0
    notes: list[str] = []

    if planet.planet_class in {PLANET_ROCKY, PLANET_SUPER_EARTH}:
        score += 25.0
        notes.append("rocky/super-Earth body")
    else:
        score -= 55.0
        notes.append(f"poor main-world class: {planet.planet_class}")

    if preference == "super_earth" and planet.planet_class == PLANET_SUPER_EARTH:
        score += 8.0
        notes.append("matches super-Earth preference")
    elif preference != "super_earth" and planet.planet_class == PLANET_ROCKY:
        score += 4.0
        notes.append("matches terrestrial preference")

    flux = planet.stellar_flux_earth
    target_flux = {
        "earthlike": 1.00,
        "dry_terrestrial": 1.08,
        "oceanic": 0.92,
        "super_earth": 0.98,
        "colder_world": 0.78,
        "warmer_world": 1.14,
    }.get(preference, 1.0)
    flux_delta = abs(flux - target_flux)
    if flux_delta <= 0.08:
        score += 36.0
        notes.append(f"excellent preference-adjusted stellar flux: {flux:.2f} Earth")
    elif 0.75 <= flux <= 1.20:
        score += 28.0
        notes.append(f"strong stellar flux: {flux:.2f} Earth")
    elif MAIN_PLANET_MIN_STELLAR_FLUX <= flux <= MAIN_PLANET_MAX_STELLAR_FLUX:
        score += 18.0
        notes.append(f"acceptable stellar flux: {flux:.2f} Earth")
    else:
        score -= 60.0
        notes.append(f"unsuitable stellar flux: {flux:.2f} Earth")

    eq_temp = planet.equilibrium_temperature_k
    if 240.0 <= eq_temp <= 262.0:
        score += 14.0
        notes.append(f"excellent equilibrium temperature: {eq_temp:.1f} K")
    elif MAIN_PLANET_MIN_EQUILIBRIUM_TEMP_K <= eq_temp <= MAIN_PLANET_MAX_EQUILIBRIUM_TEMP_K:
        score += 8.0
        notes.append(f"acceptable equilibrium temperature: {eq_temp:.1f} K")
    else:
        score -= 45.0
        notes.append(f"unsuitable equilibrium temperature: {eq_temp:.1f} K")

    gravity = planet.surface_gravity_g
    if 0.85 <= gravity <= 1.25:
        score += 18.0
        notes.append(f"excellent gravity: {gravity:.2f} g")
    elif MAIN_PLANET_MIN_GRAVITY_G <= gravity <= MAIN_PLANET_MAX_GRAVITY_G:
        score += 10.0
        notes.append(f"reasonable gravity: {gravity:.2f} g")
    else:
        score -= 25.0
        notes.append(f"less suitable gravity: {gravity:.2f} g")

    if planet.orbit.eccentricity <= MAX_MAIN_PLANET_ECCENTRICITY:
        score += 8.0
        notes.append(f"low eccentricity: {planet.orbit.eccentricity:.3f}")
    else:
        score -= 35.0
        notes.append(f"eccentricity above Main Planet preference: {planet.orbit.eccentricity:.3f}")

    water = planet.composition.water_ice_fraction
    if preference == "dry_terrestrial":
        preferred_water = (0.003, 0.018)
    elif preference == "oceanic":
        preferred_water = (0.025, 0.075)
    else:
        preferred_water = (0.006, 0.040)
    water_range_text = f"preferred {preferred_water[0]:.3f}–{preferred_water[1]:.3f} for {preference}"
    if preferred_water[0] <= water <= preferred_water[1]:
        score += 12.0
        notes.append(f"matches preference water/volatile fraction: {water:.3f} ({water_range_text})")
    elif MAIN_PLANET_MIN_WATER_FRACTION <= water <= MAIN_PLANET_MAX_WATER_FRACTION:
        score += 8.0
        notes.append(f"usable water/volatile fraction: {water:.3f} (accepted {MAIN_PLANET_MIN_WATER_FRACTION:.3f}–{MAIN_PLANET_MAX_WATER_FRACTION:.3f}; {water_range_text})")
    else:
        score -= 20.0
        notes.append(f"less suitable water/volatile fraction: {water:.4f} (accepted {MAIN_PLANET_MIN_WATER_FRACTION:.3f}–{MAIN_PLANET_MAX_WATER_FRACTION:.3f}; {water_range_text})")

    return max(0.0, min(100.0, score)), notes


def _annotate_formation_context(planets: list[Planet], star, architecture: str, preference: str) -> None:
    giant_influence = _giant_planet_influence(planets)
    giant_level = giant_influence["level"]
    for planet in planets:
        formation_zone = _formation_zone(star, planet)
        volatile_delivery = _volatile_delivery(planet, architecture, formation_zone)
        impact_history = _impact_history(architecture, giant_level, planet)
        tectonic_energy = _tectonic_energy_bias(planet, star)
        crust_asymmetry = _crustal_asymmetry_bias(impact_history, giant_level, planet)
        context = {
            "architecture": architecture,
            "main_planet_preference": preference,
            "formation_zone": formation_zone,
            "volatile_delivery": volatile_delivery,
            "giant_planet_influence": giant_level,
            "nearest_giant_distance_ratio": giant_influence.get("nearest_distance_ratio"),
            "impact_history": impact_history,
            "tectonic_energy_bias": tectonic_energy,
            "crustal_asymmetry_bias": crust_asymmetry,
        }
        if planet.is_main_planet and planet.moon is not None:
            context["moon_origin"] = planet.moon.moon_origin
            context["tidal_effect_level"] = planet.moon.tidal_effect_level
            context["axial_stability_effect"] = planet.moon.axial_stability_effect
            if planet.moon.tidal_effect_level in {"strong", "extreme"} and tectonic_energy != "high":
                context["tectonic_energy_bias"] = "moderate_high"
        planet.formation_context = context
        if planet.is_main_planet:
            planet.selection_notes.append(
                "formation context: "
                f"{formation_zone}, volatile delivery {volatile_delivery}, giant influence {giant_level}, impact history {impact_history}"
            )


def _formation_zone(star, planet: Planet) -> str:
    au = planet.orbit.semi_major_axis_au
    if au < star.habitable_zone_inner_au * 0.75:
        return "inner_system"
    if star.habitable_zone_inner_au * 0.75 <= au <= star.habitable_zone_outer_au * 1.15:
        return "habitable_zone"
    if au < star.snow_line_au:
        return "outer_rocky_zone"
    if au < star.snow_line_au * 1.8:
        return "near_snow_line"
    return "outer_volatile_zone"


def _volatile_delivery(planet: Planet, architecture: str, formation_zone: str) -> str:
    water = planet.composition.water_ice_fraction
    if architecture == "volatile_rich" or formation_zone in {"near_snow_line", "outer_volatile_zone"}:
        if water > 0.05:
            return "heavy"
        if water > 0.015:
            return "enhanced"
    if water < 0.006:
        return "dry"
    if water < 0.025:
        return "moderate"
    if water < 0.07:
        return "wet"
    return "heavy"


def _giant_planet_influence(planets: list[Planet]) -> dict[str, Any]:
    giants = [p for p in planets if p.planet_class in {PLANET_GAS_GIANT, PLANET_ICE_GIANT}]
    main = next((p for p in planets if p.is_main_planet), None)
    if not giants or main is None:
        return {"level": "weak", "count": len(giants), "nearest_distance_ratio": None}
    nearest_ratio = min(abs(g.orbit.semi_major_axis_au - main.orbit.semi_major_axis_au) / max(1e-9, main.orbit.semi_major_axis_au) for g in giants)
    total_giant_mass = sum(g.mass_earth for g in giants)
    if total_giant_mass > 260 and nearest_ratio < 5.0:
        level = "strong"
    elif total_giant_mass > 80 or nearest_ratio < 3.0:
        level = "moderate"
    else:
        level = "weak"
    return {"level": level, "count": len(giants), "nearest_distance_ratio": round(nearest_ratio, 3)}


def _impact_history(architecture: str, giant_level: str, planet: Planet) -> str:
    if architecture == "sparse_old" and giant_level == "weak":
        return "calm"
    if architecture in {"outer_giant_dominated", "volatile_rich"} and giant_level in {"moderate", "strong"}:
        return "battered"
    if planet.architecture_role == "inner_hot_world":
        return "battered"
    return "normal"


def _tectonic_energy_bias(planet: Planet, star) -> str:
    mass = planet.mass_earth
    if mass < 0.6 or star.age_gyr > 7.0:
        return "low"
    if mass > 1.8 or planet.planet_class == PLANET_SUPER_EARTH:
        return "high"
    return "earth_like"


def _crustal_asymmetry_bias(impact_history: str, giant_level: str, planet: Planet) -> str:
    if impact_history == "battered" or giant_level == "strong":
        return "high"
    if planet.mass_earth > 1.6 or giant_level == "moderate":
        return "medium"
    return "low"


def _system_diagnostics(star, planets: list[Planet], architecture: str, preference: str, attempts: int, best_candidate_snapshot: dict[str, Any]) -> dict[str, Any]:
    main = next((p for p in planets if p.is_main_planet), None)
    hz_planets = [p for p in planets if star.habitable_zone_inner_au <= p.orbit.semi_major_axis_au <= star.habitable_zone_outer_au]
    giants = [p for p in planets if p.planet_class in {PLANET_GAS_GIANT, PLANET_ICE_GIANT}]
    class_counts = Counter(p.planet_class for p in planets)
    giant_influence = _giant_planet_influence(planets)
    moon = main.moon if main is not None else None
    candidates = _candidate_table(star, planets, preference)
    selected_candidate = next((c for c in candidates if c.get("selected")), None)
    debris = _debris_belt_profile(star, planets, architecture)
    giant_history = _giant_planet_history(planets, architecture, giant_influence)
    warnings = _stage1_warnings(star, planets, main, giant_influence, debris)
    explanation = _habitability_explanation(main, selected_candidate, candidates, giant_influence, warnings)
    diagnostics: dict[str, Any] = {
        "architecture": architecture,
        "main_planet_preference": preference,
        "generation_attempts": attempts,
        "planet_count": len(planets),
        "planet_class_counts": dict(class_counts),
        "habitable_zone_planet_count": len(hz_planets),
        "outer_giant_count": len(giants),
        "giant_planet_influence": giant_influence["level"],
        "nearest_giant_distance_ratio": giant_influence.get("nearest_distance_ratio"),
        "main_planet_candidate_quality": _candidate_quality(main.habitability_score if main else 0.0),
        "selected_main_planet": None if main is None else {
            "name": main.name,
            "class": main.planet_class,
            "habitability_score": round(main.habitability_score, 3),
            "stellar_flux_earth": round(main.stellar_flux_earth, 3),
            "gravity_g": round(main.surface_gravity_g, 3),
            "water_fraction": round(main.composition.water_ice_fraction, 5),
        },
        "best_rejected_or_seen_candidate": best_candidate_snapshot,
        "main_planet_candidates": candidates,
        "habitability_explanation": explanation,
        "stage1_warnings": warnings,
        "debris_belt_profile": debris,
        "giant_planet_history": giant_history,
        "system_report": [],
        "moon": None if moon is None else {
            "name": moon.name,
            "origin": moon.moon_origin,
            "tidal_effect_level": moon.tidal_effect_level,
            "axial_stability_effect": moon.axial_stability_effect,
            "tidal_strength_relative_earth_moon": round(moon.tidal_strength_relative_earth_moon, 3),
        },
        "climate_stability_outlook": _climate_stability_outlook(main, giant_influence["level"]),
    }
    diagnostics["system_report"] = _system_report(star, planets, diagnostics)
    return diagnostics


def _candidate_table(star, planets: list[Planet], preference: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for planet in planets:
        score, notes = _habitability_score(planet, preference)
        issues = _liquid_water_eligibility_issues(planet)
        hz_position = _hz_position_label(star, planet)
        positives = [n for n in notes if not n.startswith(("poor", "unsuitable", "less suitable", "eccentricity above"))]
        concerns = [n for n in notes if n not in positives]
        concerns.extend(issues)
        status = "selected" if planet.is_main_planet else ("eligible" if not issues else "rejected")
        rows.append({
            "name": planet.name,
            "selected": bool(planet.is_main_planet),
            "status": status,
            "class": planet.planet_class,
            "architecture_role": planet.architecture_role,
            "semi_major_axis_au": round(planet.orbit.semi_major_axis_au, 4),
            "hz_position": hz_position,
            "stellar_flux_earth": round(planet.stellar_flux_earth, 3),
            "equilibrium_temperature_k": round(planet.equilibrium_temperature_k, 1),
            "mass_earth": round(planet.mass_earth, 3),
            "radius_earth": round(planet.radius_earth, 3),
            "gravity_g": round(planet.surface_gravity_g, 3),
            "water_fraction": round(planet.composition.water_ice_fraction, 5),
            "composition_class": planet.composition.composition_class,
            "habitability_score": round(planet.habitability_score if planet.habitability_score else score, 3),
            "eligible": not issues,
            "selected_or_rejected_reason": "selected Main Planet" if planet.is_main_planet else ("eligible candidate" if not issues else "; ".join(issues[:4])),
            "positive_factors": positives[:8],
            "concerns": concerns[:10],
        })
    rows.sort(key=lambda item: (0 if item.get("selected") else 1, -float(item.get("habitability_score", 0.0))))
    return rows


def _hz_position_label(star, planet: Planet) -> str:
    au = planet.orbit.semi_major_axis_au
    inner = star.habitable_zone_inner_au
    outer = star.habitable_zone_outer_au
    if au < inner * 0.85:
        return "interior hot zone"
    if au < inner:
        return "inner edge neighbor"
    if au <= inner + (outer - inner) * 0.33:
        return "inner habitable zone"
    if au <= inner + (outer - inner) * 0.67:
        return "middle habitable zone"
    if au <= outer:
        return "outer habitable zone"
    if au <= outer * 1.25:
        return "outer edge neighbor"
    return "beyond habitable zone"


def _debris_belt_profile(star, planets: list[Planet], architecture: str) -> dict[str, Any]:
    rocky_outer = [p for p in planets if p.orbit.semi_major_axis_au > star.habitable_zone_outer_au and p.orbit.semi_major_axis_au < star.snow_line_au and p.planet_class in {PLANET_ROCKY, PLANET_SUPER_EARTH}]
    giants = [p for p in planets if p.planet_class in {PLANET_GAS_GIANT, PLANET_ICE_GIANT}]
    if architecture in {"sparse_old", "low_mass_quiet"} and not giants:
        kind = "faint_outer_dust"
        activity = "low"
        note = "Old/quiet architecture leaves only a faint unresolved debris component."
    elif architecture == "compact_rocky_inner" and rocky_outer:
        kind = "inner_asteroid_belt"
        activity = "moderate"
        note = "Rocky material remains between the compact inner system and snow line."
    elif giants:
        kind = "asteroid_and_outer_icy_belts"
        activity = "high" if architecture in {"outer_giant_dominated", "volatile_rich"} else "moderate"
        note = "Outer giant planets plausibly sculpt both rocky and icy debris reservoirs."
    else:
        kind = "outer_icy_belt"
        activity = "moderate" if architecture == "volatile_rich" else "low"
        note = "No strong giant sculpting; outer icy remnants are the dominant debris reservoir."
    return {
        "type": kind,
        "activity": activity,
        "inner_edge_au": round(max(star.habitable_zone_outer_au * 1.15, star.snow_line_au * 0.55), 3),
        "outer_edge_au": round(max(star.snow_line_au * 1.8, star.habitable_zone_outer_au * 1.7), 3),
        "impact_delivery_bias": "elevated" if activity == "high" else ("moderate" if activity == "moderate" else "low"),
        "note": note,
    }


def _giant_planet_history(planets: list[Planet], architecture: str, giant_influence: dict[str, Any]) -> dict[str, Any]:
    giants = [p for p in planets if p.planet_class in {PLANET_GAS_GIANT, PLANET_ICE_GIANT}]
    if not giants:
        mode = "no_major_giant_perturbers"
        resonance = "weak"
        note = "No gas/ice giant class planets were generated, so impact shielding and disruption are weak."
    elif architecture == "outer_giant_dominated":
        mode = "disruptive_outer_giant_history" if giant_influence.get("level") == "strong" else "protective_outer_giant_history"
        resonance = "strong" if giant_influence.get("level") == "strong" else "moderate"
        note = "Outer giant planets dominate long-term debris scattering and volatile delivery."
    elif architecture == "volatile_rich":
        mode = "volatile_delivery_giant_assisted"
        resonance = "moderate"
        note = "Giant planets and icy debris reservoirs bias the system toward stronger volatile delivery."
    else:
        mode = "quiet_outer_giant_architecture"
        resonance = "moderate" if giant_influence.get("level") != "weak" else "weak"
        note = "Giant planets are present but not modeled as violently migrating."
    return {
        "mode": mode,
        "giant_count": len(giants),
        "perturbation_level": giant_influence.get("level"),
        "resonance_flavor": resonance,
        "note": note,
    }


def _stage1_warnings(star, planets: list[Planet], main: Planet | None, giant_influence: dict[str, Any], debris: dict[str, Any]) -> list[dict[str, str]]:
    warnings: list[dict[str, str]] = []
    remaining = star.main_sequence_lifetime_gyr - star.age_gyr
    if remaining < 1.5:
        warnings.append({"level": "warning", "message": f"Star has only about {remaining:.1f} Gyr of main-sequence lifetime remaining."})
    if star.age_gyr < 0.8:
        warnings.append({"level": "notice", "message": "Very young system; heavy bombardment and climate instability may be underrepresented."})
    if main is not None:
        if main.radius_earth > 1.5:
            warnings.append({"level": "notice", "message": f"Main Planet radius is large ({main.radius_earth:.2f} R⊕); downstream gravity/terrain scale should be reviewed."})
        if main.surface_gravity_g > 1.35:
            warnings.append({"level": "warning", "message": f"Main Planet gravity is high ({main.surface_gravity_g:.2f} g)."})
        if main.stellar_flux_earth < 0.70 or main.stellar_flux_earth > 1.25:
            warnings.append({"level": "warning", "message": f"Main Planet receives edge-case stellar flux ({main.stellar_flux_earth:.2f} S⊕)."})
        if main.orbit.eccentricity > 0.05:
            warnings.append({"level": "notice", "message": f"Main Planet eccentricity ({main.orbit.eccentricity:.3f}) may create stronger orbital seasonality."})
        if main.moon is None:
            warnings.append({"level": "notice", "message": "Main Planet has no major moon; axial stability is assumed low unless later stages override it."})
        else:
            hf = main.moon.orbit.hill_fraction
            if hf < 0.08 or hf > 0.45:
                warnings.append({"level": "notice", "message": f"Moon orbit uses an unusual Hill-fraction ({hf:.2f}); review tides/stability."})
    if giant_influence.get("level") == "strong":
        warnings.append({"level": "notice", "message": "Strong giant-planet influence may imply elevated impact history or orbital perturbations."})
    if debris.get("activity") == "high":
        warnings.append({"level": "notice", "message": "High debris-belt activity suggests stronger impact/volatile-delivery context."})
    return warnings


def _habitability_explanation(main: Planet | None, selected_candidate: dict[str, Any] | None, candidates: list[dict[str, Any]], giant_influence: dict[str, Any], warnings: list[dict[str, str]]) -> dict[str, Any]:
    if main is None:
        return {"summary": "No Main Planet selected.", "positive_factors": [], "concerns": ["No Main Planet selected."], "near_miss_count": 0}
    positives = list(selected_candidate.get("positive_factors", []) if selected_candidate else main.selection_notes[:])
    concerns = list(selected_candidate.get("concerns", []) if selected_candidate else [])
    if main.moon is not None:
        positives.append(f"major moon provides {main.moon.axial_stability_effect} axial-stability effect and {main.moon.tidal_effect_level} tides")
    else:
        concerns.append("no major moon for axial stabilization")
    if giant_influence.get("level") == "weak":
        positives.append("weak giant-planet perturbation")
    elif giant_influence.get("level") == "strong":
        concerns.append("strong giant-planet perturbation context")
    concerns.extend(w["message"] for w in warnings if w.get("level") == "warning")
    near_misses = [c for c in candidates if not c.get("selected") and c.get("eligible")]
    quality = _candidate_quality(main.habitability_score)
    return {
        "summary": f"{main.name} is a {quality} Main Planet candidate with {main.stellar_flux_earth:.2f} S⊕ flux, {main.surface_gravity_g:.2f} g gravity, and {main.composition.water_ice_fraction:.3f} water/volatile fraction.",
        "positive_factors": positives[:10],
        "concerns": concerns[:10],
        "near_miss_count": len(near_misses),
    }


def _system_report(star, planets: list[Planet], diagnostics: dict[str, Any]) -> list[str]:
    main = next((p for p in planets if p.is_main_planet), None)
    architecture = _human_label(diagnostics.get("architecture"))
    star_label = star.spectral_type or f"{star.stellar_class}-class"
    lines = [
        f"This is a {star_label} main-sequence system using a {architecture} architecture.",
        f"The generator produced {len(planets)} planets, including {diagnostics.get('habitable_zone_planet_count', 0)} planet(s) in the modeled habitable-zone band and {diagnostics.get('outer_giant_count', 0)} outer giant(s).",
    ]
    if main is not None:
        ctx = main.formation_context or {}
        lines.append(
            f"The selected Main Planet is {main.name}, a {main.planet_class.replace('_', ' ')} at {main.orbit.semi_major_axis_au:.3f} AU with {main.stellar_flux_earth:.2f} S⊕ flux and a {diagnostics.get('main_planet_candidate_quality')} candidate rating."
        )
        lines.append(
            "Formation context: "
            f"{_human_label(ctx.get('formation_zone'))}, volatile delivery {_human_label(ctx.get('volatile_delivery'))}, "
            f"impact history {_human_label(ctx.get('impact_history'))}, and giant-planet influence {_human_label(ctx.get('giant_planet_influence'))}."
        )
    debris = diagnostics.get("debris_belt_profile") or {}
    if isinstance(debris, dict):
        lines.append(f"Debris context: {_human_label(debris.get('type'))} with {debris.get('activity', 'unknown')} activity.")
    explanation = diagnostics.get("habitability_explanation") or {}
    if isinstance(explanation, dict) and explanation.get("summary"):
        lines.append(str(explanation["summary"]))
    warnings = diagnostics.get("stage1_warnings") or []
    if warnings:
        lines.append(f"Review flags: {len(warnings)} non-blocking Stage 1 warning(s)/notice(s) were generated.")
    return lines


def _human_label(value: Any) -> str:
    if value is None:
        return "unknown"
    return str(value).replace("_", " ")

def _candidate_quality(score: float) -> str:
    if score >= 86:
        return "strong"
    if score >= 70:
        return "good"
    if score >= 55:
        return "acceptable"
    return "weak"


def _climate_stability_outlook(main: Planet | None, giant_level: str) -> str:
    if main is None:
        return "unknown"
    moon_effect = main.moon.axial_stability_effect if main.moon else "low"
    if main.orbit.eccentricity <= 0.025 and moon_effect in {"moderate", "high"} and giant_level != "strong":
        return "favorable"
    if main.orbit.eccentricity <= 0.05 and giant_level != "strong":
        return "moderate"
    return "variable"


def _diagnostic_notes(diagnostics: dict[str, Any]) -> list[str]:
    return [
        "Solar-system diagnostics: "
        f"HZ planets={diagnostics.get('habitable_zone_planet_count')}, "
        f"outer giants={diagnostics.get('outer_giant_count')}, "
        f"giant influence={diagnostics.get('giant_planet_influence')}, "
        f"candidate quality={diagnostics.get('main_planet_candidate_quality')}, "
        f"climate stability outlook={diagnostics.get('climate_stability_outlook')}.",
    ]
