"""Top-level generated star system model."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any

from worldgen.models.bodies import Planet, Star
from worldgen.models.planet_profile import MainPlanetProfile


@dataclass
class StarSystem:
    seed: int | None
    star: Star
    planets: list[Planet]
    notes: list[str]
    main_planet_profile: MainPlanetProfile | None = None
    architecture: str = "unspecified"
    diagnostics: dict[str, Any] = field(default_factory=dict)

    @property
    def main_planet(self) -> Planet | None:
        for planet in self.planets:
            if planet.is_main_planet:
                return planet
        return None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
