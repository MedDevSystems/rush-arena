from __future__ import annotations

# region MODULE_CONTRACT [DOMAIN(8): LevelDesign; CONCEPT(9): MapPool; TECH(7): Python]
## @modulecontract
## @purpose Generate 4 distinct maps with circle-based cover, varying dimensions, spawn zones, and CP positions. Returns MapDef objects ready for the env.
## @scope Map generation and MapDef data structure.
## @input None (map designs are hardcoded)
## @output List of MapDef instances
## @links LINKS_TO: config, spatial
## @invariants
## - Every map carries exactly 7 control points — the observation has 7 CP slots
## - Compact maps must set a smaller cp_radius, or their points overlap
## @changes
## LAST_CHANGE: [v0.3.0] 3 compact maps (Pit 40x40, Alley 50x28, Ring 45x45) with per-map cp_radius for the early curriculum.
## PREV: [v0.2.0] 4-map pool: Pillars(100x80), Arena(120x120), Diagonal(80x80), Highway(140x60).
## @modulemap
## CLASS 7[Map definition container] => MapDef
## FUNC 8[Generate the full 7-map pool] => build_map_pool
## BLOCK 8[Close-quarters maps for bootstrap] => _map_pit / _map_alley / _map_ring
def _module_contract():
    pass
# endregion MODULE_CONTRACT
# GREP_SUMMARY: map pool, procedural, circle pillars, 4 maps, varying sizes, spawn zones, control points

import math
from dataclasses import dataclass, field

import pygame


@dataclass
class MapDef:
    name: str
    cols: int
    rows: int
    tile_size: int
    grid: list[list[str]]
    cp_positions: list[tuple[int, int]]
    spawn_zone_a: tuple[str, int, int, int, int]  # (name, row_min, row_max, col_min, col_max)
    spawn_zone_b: tuple[str, int, int, int, int]
    cp_radius: float = 150.0
    compact: bool = False

    @property
    def arena_w(self) -> int:
        return self.cols * self.tile_size

    @property
    def arena_h(self) -> int:
        return self.rows * self.tile_size

    @property
    def arena_diag(self) -> float:
        return math.hypot(self.arena_w, self.arena_h)

    def to_ascii(self) -> str:
        return "\n".join("".join(row) for row in self.grid)

    def build_walls(self) -> list[pygame.Rect]:
        walls = []
        for r in range(self.rows):
            for c in range(self.cols):
                if self.grid[r][c] == "#":
                    walls.append(pygame.Rect(c * self.tile_size, r * self.tile_size, self.tile_size, self.tile_size))
        return walls


def _empty_grid(cols: int, rows: int) -> list[list[str]]:
    grid = [["." for _ in range(cols)] for _ in range(rows)]
    for x in range(cols):
        grid[0][x] = "#"
        grid[rows - 1][x] = "#"
    for y in range(rows):
        grid[y][0] = "#"
        grid[y][cols - 1] = "#"
    return grid


def _circle(grid: list[list[str]], cy: int, cx: int, r: int) -> None:
    rows, cols = len(grid), len(grid[0])
    for y in range(max(1, cy - r - 1), min(rows - 1, cy + r + 2)):
        for x in range(max(1, cx - r - 1), min(cols - 1, cx + r + 2)):
            if math.hypot(x - cx, y - cy) <= r:
                grid[y][x] = "#"


def _map_pillars() -> MapDef:
    W, H = 100, 80
    g = _empty_grid(W, H)
    _circle(g, 20, 25, 4); _circle(g, 20, 75, 4)
    _circle(g, 60, 25, 4); _circle(g, 60, 75, 4)
    _circle(g, 40, 50, 5)
    _circle(g, 15, 50, 3); _circle(g, 65, 50, 3)
    _circle(g, 40, 15, 3); _circle(g, 40, 85, 3)
    _circle(g, 10, 38, 2); _circle(g, 10, 62, 2)
    _circle(g, 30, 35, 2); _circle(g, 30, 65, 2)
    _circle(g, 50, 35, 2); _circle(g, 50, 65, 2)
    _circle(g, 70, 38, 2); _circle(g, 70, 62, 2)
    _circle(g, 25, 50, 1); _circle(g, 55, 50, 1)
    _circle(g, 40, 35, 1); _circle(g, 40, 65, 1)
    return MapDef(
        name="Pillars", cols=W, rows=H, tile_size=30, grid=g,
        cp_positions=[(10, 50), (30, 25), (30, 75), (40, 50), (50, 25), (50, 75), (70, 50)],
        spawn_zone_a=("blue", 25, 55, 2, 12),
        spawn_zone_b=("red", 25, 55, 87, 97),
    )


def _map_arena() -> MapDef:
    W, H = 120, 120
    g = _empty_grid(W, H)
    _circle(g, 60, 60, 6)
    for ad in range(0, 360, 45):
        a = math.radians(ad)
        _circle(g, 60 + int(30 * math.sin(a)), 60 + int(30 * math.cos(a)), 4)
    for ad in range(22, 360, 45):
        a = math.radians(ad)
        _circle(g, 60 + int(16 * math.sin(a)), 60 + int(16 * math.cos(a)), 2)
    _circle(g, 12, 30, 3); _circle(g, 12, 60, 3); _circle(g, 12, 90, 3)
    _circle(g, 108, 30, 3); _circle(g, 108, 60, 3); _circle(g, 108, 90, 3)
    _circle(g, 20, 15, 2); _circle(g, 20, 105, 2)
    _circle(g, 100, 15, 2); _circle(g, 100, 105, 2)
    return MapDef(
        name="Arena", cols=W, rows=H, tile_size=30, grid=g,
        cp_positions=[(20, 60), (40, 30), (40, 90), (60, 60), (80, 30), (80, 90), (100, 60)],
        spawn_zone_a=("blue", 100, 118, 2, 118),
        spawn_zone_b=("red", 2, 18, 2, 118),
    )


def _map_diagonal() -> MapDef:
    W, H = 80, 80
    g = _empty_grid(W, H)
    _circle(g, 20, 20, 5); _circle(g, 40, 40, 6); _circle(g, 60, 60, 5)
    _circle(g, 20, 55, 3); _circle(g, 55, 20, 3)
    _circle(g, 25, 40, 3); _circle(g, 40, 60, 3)
    _circle(g, 60, 25, 3); _circle(g, 40, 20, 3)
    _circle(g, 10, 40, 2); _circle(g, 40, 10, 2)
    _circle(g, 70, 40, 2); _circle(g, 40, 70, 2)
    _circle(g, 12, 65, 1); _circle(g, 65, 12, 1)
    _circle(g, 30, 30, 1); _circle(g, 50, 50, 1)
    return MapDef(
        name="Diagonal", cols=W, rows=H, tile_size=30, grid=g,
        cp_positions=[(10, 40), (25, 60), (35, 25), (40, 40), (45, 55), (55, 20), (70, 40)],
        spawn_zone_a=("blue", 60, 78, 2, 20),
        spawn_zone_b=("red", 2, 18, 60, 78),
    )


def _map_highway() -> MapDef:
    W, H = 140, 60
    g = _empty_grid(W, H)
    _circle(g, 30, 45, 5); _circle(g, 30, 70, 6); _circle(g, 30, 95, 5)
    _circle(g, 12, 35, 3); _circle(g, 12, 70, 3); _circle(g, 12, 105, 3)
    _circle(g, 48, 35, 3); _circle(g, 48, 70, 3); _circle(g, 48, 105, 3)
    _circle(g, 30, 18, 4); _circle(g, 30, 122, 4)
    _circle(g, 20, 55, 2); _circle(g, 20, 85, 2)
    _circle(g, 40, 55, 2); _circle(g, 40, 85, 2)
    _circle(g, 30, 30, 1); _circle(g, 30, 110, 1)
    _circle(g, 8, 55, 1); _circle(g, 52, 55, 1)
    _circle(g, 8, 85, 1); _circle(g, 52, 85, 1)
    return MapDef(
        name="Highway", cols=W, rows=H, tile_size=30, grid=g,
        cp_positions=[(15, 45), (15, 95), (30, 30), (30, 70), (30, 110), (45, 45), (45, 95)],
        spawn_zone_a=("blue", 10, 50, 2, 14),
        spawn_zone_b=("red", 10, 50, 126, 138),
    )


# region BLOCK_COMPACT_MAPS
## @purpose Close-quarters maps for the early curriculum: contact happens in seconds and a
## roughly aimed shot still lands, which is the only reason aiming is learnable from random weights.
## @rationale A player is 12 px wide. At 400 px that is 1.7 degrees of angular width, at 1400 px
## it is 0.5 — the difference between a hit every hundred random shots and effectively never.

def _map_pit() -> MapDef:
    """40x40 (1200x1200 px). One room, four pillars, spawns on opposite walls."""
    W, H = 40, 40
    g = _empty_grid(W, H)
    _circle(g, 12, 12, 3); _circle(g, 12, 27, 3)
    _circle(g, 27, 12, 3); _circle(g, 27, 27, 3)
    _circle(g, 20, 20, 2)
    return MapDef(
        name="Pit", cols=W, rows=H, tile_size=30, grid=g,
        cp_positions=[(6, 20), (20, 6), (20, 20), (20, 34), (34, 20), (12, 12), (28, 28)],
        spawn_zone_a=("blue", 33, 38, 2, 38),
        spawn_zone_b=("red", 2, 7, 2, 38),
        cp_radius=70.0, compact=True,
    )


def _map_alley() -> MapDef:
    """50x28 corridor: no way to avoid contact, every fight is a duel at close range."""
    W, H = 50, 28
    g = _empty_grid(W, H)
    _circle(g, 14, 12, 2); _circle(g, 14, 25, 3); _circle(g, 14, 38, 2)
    _circle(g, 6, 19, 2); _circle(g, 22, 19, 2)
    _circle(g, 6, 31, 1); _circle(g, 22, 31, 1)
    _circle(g, 6, 7, 1); _circle(g, 22, 7, 1)
    return MapDef(
        name="Alley", cols=W, rows=H, tile_size=30, grid=g,
        cp_positions=[(14, 8), (6, 15), (22, 15), (14, 25), (6, 35), (22, 35), (14, 42)],
        spawn_zone_a=("blue", 4, 24, 2, 6),
        spawn_zone_b=("red", 4, 24, 44, 48),
        cp_radius=70.0, compact=True,
    )


def _map_ring() -> MapDef:
    """45x45 with a solid core: everyone circles the same block, sightlines stay short."""
    W, H = 45, 45
    g = _empty_grid(W, H)
    _circle(g, 22, 22, 7)
    _circle(g, 8, 8, 2); _circle(g, 8, 36, 2)
    _circle(g, 36, 8, 2); _circle(g, 36, 36, 2)
    return MapDef(
        name="Ring", cols=W, rows=H, tile_size=30, grid=g,
        cp_positions=[(6, 22), (22, 6), (22, 38), (38, 22), (12, 12), (32, 32), (12, 32)],
        spawn_zone_a=("blue", 38, 43, 2, 43),
        spawn_zone_b=("red", 2, 7, 2, 43),
        cp_radius=70.0, compact=True,
    )
# endregion BLOCK_COMPACT_MAPS


def build_map_pool() -> list[MapDef]:
    """Full pool. Compact maps carry the early curriculum, large ones the late one."""
    return [
        _map_pit(), _map_alley(), _map_ring(),
        _map_pillars(), _map_arena(), _map_diagonal(), _map_highway(),
    ]
