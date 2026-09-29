"""Armies (the 8 soldier classes interleaved by slot), magazine sizes and map lookup."""
from __future__ import annotations

from typing import Any

from rush.forks12 import FORKS
from rush.maps_big import BIG_MAPS, get_map

# magazine per class (rounds), the values the networks were trained with
TARGET_CAPS: dict[str, int] = {"base": 22, "hunter": 25, "heavy": 38, "sniper": 17, "tank": 28, "scout": 20, "assault": 30, "guardian": 22}
THEMES = ("steppe", "urban", "forest", "delta", "plateau")


def parse_army(spec: str) -> list[str]:
    names = list(FORKS) if spec.strip().lower() == "all" else [s.strip() for s in spec.split(",") if s.strip()]
    for f in names:
        if f not in FORKS:
            raise SystemExit(f"unknown class {f!r}; known: {list(FORKS)} or 'all'")
    return names


def assign_forks(team_size: int, blue: list[str], red: list[str]) -> tuple[list[str], list[int]]:
    """Slot -> army entry (blue classes + red classes), classes interleaved within each team."""
    forks = blue + red
    agent_fork = [i % len(blue) for i in range(team_size)] + [len(blue) + (i % len(red)) for i in range(team_size)]
    return forks, agent_fork


def resolve_map(spec: str) -> Any:
    """A hand-made map (Warfront500 — 500 v 500, Front150, Crossing150, Front50) or a generated one:
    "<theme>:<seed>[:<team size>]" with theme in steppe, urban, forest, delta, plateau (500 per side by default)."""
    if spec in BIG_MAPS:
        return get_map(spec)
    parts = spec.split(":")
    if len(parts) in (2, 3) and parts[0] in THEMES:
        from rush.maps14 import themed_big
        return themed_big(parts[0], int(parts[2]) if len(parts) == 3 else 500, int(parts[1]))
    raise SystemExit(f"map {spec!r}: one of {list(BIG_MAPS)} or <theme>:<seed>[:<team>] with theme in {THEMES}")
