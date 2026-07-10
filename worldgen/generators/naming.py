"""Name generation helpers for planets and moons.

The goal is not to emulate any one real naming culture perfectly, but to avoid
repeating the same small fixed list of names. These generators create
pronounceable pseudo-astronomical names while staying deterministic for a given
random seed.
"""

from __future__ import annotations

import random

PLANET_PREFIXES = [
    "A", "Ae", "Al", "An", "Ar", "Au", "Bel", "Br", "Cae", "Cal", "Cer", "Cor",
    "Da", "Del", "Dor", "El", "Eri", "Fae", "Gal", "Hel", "Io", "Iri", "Ka", "Kel",
    "La", "Ly", "Ma", "Mor", "Na", "Ner", "Or", "Oph", "Pa", "Py", "Qua", "Rhe",
    "Sa", "Sel", "Ta", "Tor", "Uri", "Va", "Vel", "Xa", "Yri", "Za",
]

PLANET_MIDDLES = [
    "la", "ra", "ri", "ro", "ta", "the", "na", "ne", "no", "lia", "vora", "cae",
    "dra", "pho", "myr", "syl", "ther", "vyn", "lune", "rion", "this", "vra",
    "mere", "dor", "qua", "sar", "lys", "tan", "vor", "zen",
]

PLANET_SUFFIXES = [
    "a", "ae", "ara", "aris", "ea", "eon", "era", "eron", "es", "ia", "ion", "ira",
    "is", "on", "or", "ora", "os", "um", "us", "yra", "ys",
]

MOON_PREFIXES = [
    "Al", "Ara", "Ari", "Bel", "Ca", "Celi", "Da", "Eli", "Eo", "Ila", "Iri", "Ka",
    "La", "Lu", "Ma", "Mira", "Ne", "Ny", "Ori", "Rae", "Sa", "Seli", "Ta", "Va", "Ve",
]

MOON_SUFFIXES = [
    "a", "ah", "el", "en", "eth", "i", "ia", "iel", "il", "in", "is", "or", "ra",
    "rin", "ta", "the", "u", "une", "ys",
]


def generate_planet_name(rng: random.Random, used_names: set[str]) -> str:
    """Generate a unique planet name."""
    return _generate_unique_name(rng, used_names, kind="planet")



def generate_moon_name(rng: random.Random, used_names: set[str], planet_name: str) -> str:
    """Generate a unique moon name.

    Most moons get a short standalone name. Some borrow a subtle relationship to
    the host planet by sharing the first letter, which adds variety without
    forcing a strict naming convention.
    """
    for _ in range(100):
        if rng.random() < 0.25 and planet_name:
            # A loosely planet-linked name.
            start = planet_name[0].upper()
            middle = rng.choice(["a", "e", "i", "o", "u", "el", "ir", "or", "yl"])
            end = rng.choice(MOON_SUFFIXES)
            name = _normalize_name(start + middle + end)
        else:
            name = _generate_base_name(rng, kind="moon")
        if name not in used_names:
            used_names.add(name)
            return name
    # Fallback with numeric suffix.
    base = _generate_base_name(rng, kind="moon")
    i = 2
    name = base
    while name in used_names:
        name = f"{base}-{i}"
        i += 1
    used_names.add(name)
    return name



def _generate_unique_name(rng: random.Random, used_names: set[str], kind: str) -> str:
    for _ in range(100):
        name = _generate_base_name(rng, kind=kind)
        if name not in used_names:
            used_names.add(name)
            return name
    base = _generate_base_name(rng, kind=kind)
    i = 2
    name = base
    while name in used_names:
        name = f"{base}-{i}"
        i += 1
    used_names.add(name)
    return name



def _generate_base_name(rng: random.Random, kind: str) -> str:
    if kind == "planet":
        prefix = rng.choice(PLANET_PREFIXES)
        suffix = rng.choice(PLANET_SUFFIXES)
        roll = rng.random()
        if roll < 0.58:
            raw = prefix + suffix
        elif roll < 0.93:
            raw = prefix + rng.choice(PLANET_MIDDLES) + suffix
        else:
            raw = prefix + rng.choice(PLANET_SUFFIXES) + "-" + rng.choice(PLANET_PREFIXES) + rng.choice(["a", "is", "on"])
    elif kind == "moon":
        prefix = rng.choice(MOON_PREFIXES)
        suffix = rng.choice(MOON_SUFFIXES)
        if rng.random() < 0.72:
            raw = prefix + suffix
        else:
            raw = prefix + rng.choice(["l", "m", "n", "r"]) + suffix
    else:
        raise ValueError(f"Unsupported name kind: {kind}")
    return _normalize_name(raw)



def _normalize_name(raw: str) -> str:
    pieces = [piece for piece in raw.split("-") if piece]
    normalized_pieces = []
    for piece in pieces:
        piece = piece.replace("aa", "a").replace("ee", "e").replace("ii", "i").replace("oo", "o")
        piece = piece.replace("uu", "u")
        piece = piece[0].upper() + piece[1:].lower()
        normalized_pieces.append(piece)
    return "-".join(normalized_pieces)
