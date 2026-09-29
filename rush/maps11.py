from __future__ import annotations

# region MODULE_CONTRACT [DOMAIN(8): LevelDesign; CONCEPT(9): FairMapPool; TECH(7): Python]
## @modulecontract
## @purpose Map pool of experiment 11: the seven legacy layouts with their dead control points fixed in the
## data, thirteen hand-designed layouts that each force a different tactic, and a seeded procedural
## generator of symmetric maps. Every map is fair between the teams by construction and checkable.
## @scope Map data, symmetry transforms, validation (symmetry, reachability, path-distance balance,
## openness). No torch, no rendering: world11 consumes MapDef objects, tools/map_check renders them.
## @input Pool name ("maps11", "legacy", "maps11+proc:N") and a seed for the procedural part
## @output list[MapDef11] — a MapDef subclass, so world11 and record11 read it unchanged
## @links LINKS_TO: map_gen (frozen, legacy grids are read from it), world11, tools/map_check, tests/test_maps11
## @invariants
## - Exactly 7 control points per map: world11 has 7 CP slots, the observation and the scoring assume all 7
## - One tile size (30 px) for the whole pool; every map fits in 121 rows x 141 cols
## - Each map is invariant under its team transform: the wall grid exactly, the CP set within PAIR_TOL tiles,
##   spawn zone B is the image of zone A
## - Every free tile is reachable (no sealed pockets), every CP zone holds free tiles reachable from both spawns
## - The seven legacy maps keep their names and their order at the head of the pool (indices 0..6)
## @rationale
## Q: Why fairness by symmetry instead of hand-balancing?
## A: A self-play league reads a team-side advantage as skill. A mirror or a 180-degree rotation makes the two
## A: sides the same game; the checker then only has to verify the transform, not judge a layout.
## Q: Why fix the legacy CPs in the data rather than keep world11's relocation?
## A: The relocation moves a point off the design axis and breaks the symmetry it sat on. Here the covering
## A: pillar is shrunk instead, so the point stays where the designer put it and becomes capturable.
## Q: Why odd dimensions for every new map?
## A: With an odd width the mirror axis runs through tile centres, so a point on the axis maps onto itself and
## A: the CP set is exactly symmetric. Even-width legacy maps keep a one-tile pairing error (documented below).
## @changes
## LAST_CHANGE: [v0.1.0] Initial pool: 7 legacy (fixed) + 13 designed + procedural generator.
## @modulemap
## CLASS 7[MapDef with symmetry and design notes] => MapDef11
## BLOCK 8[Grid drawing and symmetry helpers] => _canvas / _rect / _circle / _symmetrize / _image
## BLOCK 8[Legacy maps with fixed CPs] => _legacy_*
## BLOCK 9[Designed maps] => _map_*
## FUNC 9[Seeded symmetric generator] => procedural_map
## FUNC 9[Fairness and reachability checks] => validate_map
## FUNC 8[Pool selection] => build_pool
## @usecases
## - world11: World11(n, map_pool="maps11") -> build_pool("maps11", seed)
## - tools/map_check: validate_map(m) for every map, thumbnails
def _module_contract():
    pass
# endregion MODULE_CONTRACT
# GREP_SUMMARY: maps, map pool, symmetry, mirror, rot180, fairness, control points, procedural, validation, BFS
# STRUCTURE: ▶ design (half + symmetrize) → ⚡ CPs + images → ○ validate (symmetry, BFS balance, openness) → ⎋ MapDef11

import heapq
import logging
import math
import random
from collections import deque
from dataclasses import dataclass

from rush import map_gen
from rush.map_gen import MapDef

logger = logging.getLogger(__name__)

# region BLOCK_CONSTANTS
TILE: int = 30
N_CPS: int = 7
MAX_ROWS: int = 121
MAX_COLS: int = 141
PAIR_TOL_TILES: float = 1.5          # CP image must land within this many tiles of another CP
BALANCE_TOL_PX: float = 60.0         # team path distance to a CP and to its image may differ by this much
MIN_SAFE_SPAWN_TILES: int = 10
MIN_SPAWN_SEPARATION_TILES: float = 20.0
SYMMETRIES = ("mirror_lr", "mirror_tb", "rot180")
# endregion BLOCK_CONSTANTS


# region CLASS_MapDef11
## @purpose MapDef plus the fairness transform and a one-line design note. world11 reads only MapDef fields.
@dataclass
class MapDef11(MapDef):
    symmetry: str = "mirror_lr"
    tactic: str = ""
    kind: str = "designed"          # legacy | designed | procedural
    seed: int | None = None
# endregion CLASS_MapDef11


# region BLOCK_GRID_HELPERS
def _canvas(cols: int, rows: int, fill: str = ".") -> list[list[str]]:
    g = [[fill for _ in range(cols)] for _ in range(rows)]
    for x in range(cols):
        g[0][x] = g[rows - 1][x] = "#"
    for y in range(rows):
        g[y][0] = g[y][cols - 1] = "#"
    return g


def _rect(g: list[list[str]], r0: int, r1: int, c0: int, c1: int, ch: str = "#") -> None:
    """Fill rows r0..r1 and cols c0..c1 inclusive, never touching the border ring."""
    rows, cols = len(g), len(g[0])
    for r in range(max(1, min(r0, r1)), min(rows - 2, max(r0, r1)) + 1):
        for c in range(max(1, min(c0, c1)), min(cols - 2, max(c0, c1)) + 1):
            g[r][c] = ch


def _circle(g: list[list[str]], cy: int, cx: int, rad: float, ch: str = "#") -> None:
    rows, cols = len(g), len(g[0])
    ri = int(math.ceil(rad))
    for y in range(max(1, cy - ri), min(rows - 1, cy + ri + 1)):
        for x in range(max(1, cx - ri), min(cols - 1, cx + ri + 1)):
            if math.hypot(x - cx, y - cy) <= rad:
                g[y][x] = ch


def _image(r: float, c: float, rows: int, cols: int, sym: str) -> tuple[float, float]:
    if sym == "mirror_lr":
        return r, cols - 1 - c
    if sym == "mirror_tb":
        return rows - 1 - r, c
    if sym == "rot180":
        return rows - 1 - r, cols - 1 - c
    raise ValueError(f"unknown symmetry {sym}")


def _symmetrize(g: list[list[str]], sym: str) -> None:
    """Union of the grid with its image: a wall anywhere is a wall at its image too."""
    rows, cols = len(g), len(g[0])
    walls = [(r, c) for r in range(rows) for c in range(cols) if g[r][c] == "#"]
    for r, c in walls:
        ir, ic = _image(r, c, rows, cols, sym)
        g[int(ir)][int(ic)] = "#"


def _zone_image(zone: tuple[str, int, int, int, int], rows: int, cols: int, sym: str, name: str) -> tuple[str, int, int, int, int]:
    _, r0, r1, c0, c1 = zone
    a = _image(r0, c0, rows, cols, sym)
    b = _image(r1, c1, rows, cols, sym)
    return (name, int(min(a[0], b[0])), int(max(a[0], b[0])), int(min(a[1], b[1])), int(max(a[1], b[1])))


def _cps(seeds: list[tuple[int, int]], rows: int, cols: int, sym: str) -> list[tuple[int, int]]:
    """Seed points plus their images; a point that is its own image (on the axis / centre) counts once."""
    out: list[tuple[int, int]] = []
    for r, c in seeds:
        out.append((r, c))
        ir, ic = _image(r, c, rows, cols, sym)
        if (int(ir), int(ic)) != (r, c) and math.hypot(ir - r, ic - c) > PAIR_TOL_TILES:
            out.append((int(ir), int(ic)))
    if len(out) != N_CPS:
        raise ValueError(f"CP seeds {seeds} give {len(out)} points under {sym}, need {N_CPS}")
    return out


def _make(name: str, g: list[list[str]], sym: str, cp_seeds: list[tuple[int, int]], zone_a: tuple[int, int, int, int],
          cp_radius: float, compact: bool, tactic: str, kind: str = "designed", seed: int | None = None) -> MapDef11:
    rows, cols = len(g), len(g[0])
    _symmetrize(g, sym)
    za = ("blue", *zone_a)
    zb = _zone_image(za, rows, cols, sym, "red")
    # Sealed pockets (a free tile inside a ring of pillars) would be unreachable flow-field holes.
    # Both spawns seed the fill, so the filled set is symmetric too.
    _fill_pockets(g, _safe_tiles(g, za) + _safe_tiles(g, zb))
    return MapDef11(name=name, cols=cols, rows=rows, tile_size=TILE, grid=g, cp_positions=_cps(cp_seeds, rows, cols, sym),
                    spawn_zone_a=za, spawn_zone_b=zb, cp_radius=cp_radius, compact=compact,
                    symmetry=sym, tactic=tactic, kind=kind, seed=seed)
# endregion BLOCK_GRID_HELPERS


# region BLOCK_LEGACY
## @purpose The experiment 9/10 layouts, made exactly symmetric and with every CP capturable.
## @rationale The centre CP of six maps sat inside a pillar wider than its capture radius. The pillar is
## shrunk (Pit/Alley to radius 1, the large maps to radius 3), which keeps cover on the spot and puts free
## tiles inside the circle. Walls are unioned with their team image; CP pairs are snapped onto exact images.
def _legacy(src: MapDef, sym: str, cp_seeds: list[tuple[int, int]], shrink: list[tuple[int, int, float, float]], tactic: str) -> MapDef11:
    g = [row[:] for row in src.grid]
    for cy, cx, old_r, new_r in shrink:
        _circle(g, cy, cx, old_r, ".")
        _circle(g, cy, cx, new_r, "#")
    za = src.spawn_zone_a[1:]
    return _make(src.name, g, sym, cp_seeds, za, src.cp_radius, src.compact, tactic, kind="legacy")


def _legacy_pool() -> list[MapDef11]:
    return [
        _legacy(map_gen._map_pit(), "mirror_tb",
                [(20, 20), (20, 6), (20, 34), (6, 20), (12, 20)],
                [(20, 20, 2, 1)], "One room, four pillars: fights start in seconds."),
        _legacy(map_gen._map_alley(), "mirror_lr",
                [(14, 25), (14, 8), (6, 15), (22, 15)],
                [(14, 25, 3, 1)], "Corridor duel: no way around the enemy."),
        _legacy(map_gen._map_ring(), "mirror_tb",
                [(22, 6), (6, 22), (12, 12), (12, 32)],
                [], "Everyone circles one solid core; sightlines stay short."),
        _legacy(map_gen._map_pillars(), "mirror_lr",
                [(40, 50), (10, 50), (70, 50), (30, 25), (50, 25)],
                [(40, 50, 5, 3)], "Pillar forest: peek, shoot, rotate between columns."),
        _legacy(map_gen._map_arena(), "mirror_tb",
                [(60, 60), (20, 60), (40, 30), (40, 90)],
                [(60, 60, 6, 3)], "Big round arena with a ring of cover around the centre."),
        _legacy(map_gen._map_diagonal(), "rot180",
                [(40, 40), (10, 40), (25, 60), (35, 25)],
                [(40, 40, 6, 3)], "Diagonal axis: the front line runs corner to corner."),
        _legacy(map_gen._map_highway(), "mirror_lr",
                [(30, 70), (15, 45), (30, 30), (45, 45)],
                [(30, 70, 6, 3)], "Wide highway: three rows of cover, long lateral rotations."),
    ]
# endregion BLOCK_LEGACY


# region BLOCK_DESIGNED_COMPACT
## @purpose Close-quarters layouts for the early curriculum, each with its own dominant decision.

def _map_crossfire() -> MapDef11:
    """41x41: a plus of walls splits four quadrants around an open centre plaza."""
    W = H = 41
    g = _canvas(W, H)
    _rect(g, 6, 14, 20, 20); _rect(g, 26, 34, 20, 20)       # vertical arms
    _rect(g, 20, 20, 6, 14); _rect(g, 20, 20, 26, 34)       # horizontal arms
    _rect(g, 9, 10, 9, 10); _rect(g, 6, 7, 33, 34)           # quadrant crates, clear of the CP circles
    _rect(g, 14, 15, 13, 14); _rect(g, 14, 15, 26, 27)
    return _make("Crossfire", g, "rot180", [(20, 20), (10, 30), (4, 20), (10, 14)], (33, 38, 2, 12), 65.0, True,
                 "Four quadrants and a centre plaza: hold the cross or flank through the outer ring.")


def _map_warehouse() -> MapDef11:
    """45x35: loading bays on both sides, doors into a shelved main hall."""
    W, H = 45, 35
    g = _canvas(W, H)
    _rect(g, 1, 7, 11, 11); _rect(g, 11, 23, 11, 11); _rect(g, 27, 33, 11, 11)   # bay wall, doors at 8-10 and 24-26
    _rect(g, 11, 11, 16, 20); _rect(g, 11, 11, 24, 28)                            # shelving rows, gap in the middle
    _rect(g, 23, 23, 16, 20); _rect(g, 23, 23, 24, 28)
    _rect(g, 8, 9, 18, 19); _rect(g, 25, 26, 18, 19)                              # crates
    _rect(g, 16, 18, 18, 18)
    return _make("Warehouse", g, "mirror_lr", [(17, 22), (5, 17), (29, 17), (17, 15)], (3, 31, 2, 8), 60.0, True,
                 "Rooms and doors: clear the doorway, then fight shelf to shelf.")


def _map_maze() -> MapDef11:
    """41x41: blocks on a lattice joined into L-walls; corridors 3-5 tiles, loops and dead ends."""
    W = H = 41
    g = _canvas(W, H)
    lattice = [5, 13, 25, 33]
    for r in lattice:
        for c in lattice:
            _rect(g, r, r + 2, c, c + 2)
    _rect(g, 5, 7, 8, 12); _rect(g, 13, 15, 28, 32)            # joins -> L shapes
    _rect(g, 8, 12, 13, 15); _rect(g, 16, 20, 25, 27)
    _rect(g, 25, 27, 8, 12); _rect(g, 28, 32, 5, 7)
    _rect(g, 18, 22, 5, 7); _rect(g, 18, 22, 17, 17)           # centre stub
    return _make("Maze", g, "rot180", [(20, 20), (10, 20), (20, 10), (30, 3)], (34, 38, 2, 16), 60.0, True,
                 "Close-quarters maze: corners, ambushes, short trades at point-blank range.")


def _map_bunker() -> MapDef11:
    """39x39: king of the hill — a walled keep with four doors in the middle of the room."""
    W = H = 39
    g = _canvas(W, H)
    _rect(g, 12, 12, 12, 26); _rect(g, 26, 26, 12, 26)
    _rect(g, 12, 26, 12, 12); _rect(g, 12, 26, 26, 26)
    for d in (18, 19, 20):                                     # four doors
        g[12][d] = g[26][d] = g[d][12] = g[d][26] = "."
    _rect(g, 16, 16, 16, 17); _rect(g, 22, 22, 21, 22)         # cover inside the keep
    _rect(g, 6, 7, 6, 7); _rect(g, 6, 7, 31, 32)                # outer cover
    return _make("Bunker", g, "rot180", [(19, 19), (6, 19), (19, 6), (8, 30)], (33, 37, 1, 12), 60.0, True,
                 "King of the hill: the keep scores, but its four doors are kill zones.")
# endregion BLOCK_DESIGNED_COMPACT


# region BLOCK_DESIGNED_LARGE
## @purpose Large layouts for the late curriculum: rotations, sightlines and map control.

def _map_lanes() -> MapDef11:
    """121x61: three lanes (MOBA) split by long walls with two crossings each."""
    W, H = 121, 61
    g = _canvas(W, H)
    for row in (20, 40):
        _rect(g, row, row, 20, 100)
        for c0 in (40, 76):
            _rect(g, row, row, c0, c0 + 4, ".")
        _rect(g, row, row, 58, 62, ".")
    for c in (30, 50):
        _rect(g, 9, 11, c, c + 1); _rect(g, 49, 51, c, c + 1); _rect(g, 29, 31, c + 2, c + 3)
    return _make("Lanes", g, "mirror_lr", [(10, 60), (30, 60), (50, 60), (10, 34), (50, 34)], (10, 50, 2, 12), 110.0, False,
                 "Three lanes: push one, defend two, rotate through the crossings.")


def _map_longshot() -> MapDef11:
    """141x41: long sightlines — an open centre lane between galleries with windows."""
    W, H = 141, 41
    g = _canvas(W, H)
    for row in (12, 28):                                           # left half only; _make mirrors it
        _rect(g, row, row, 16, 70)
        for c0 in (26, 44, 62):
            _rect(g, row, row, c0, c0 + 1, ".")
    for c in (35, 55):
        _rect(g, 19, 21, c, c)
    _rect(g, 5, 6, 45, 46); _rect(g, 34, 35, 45, 46)
    return _make("Longshot", g, "mirror_lr", [(20, 70), (20, 38), (6, 55), (34, 55)], (4, 36, 2, 12), 110.0, False,
                 "Sniper lanes: whoever controls the open centre lane sees everything; galleries flank it.")


def _map_fortress() -> MapDef11:
    """101x101: a walled fortress in the centre with four gates, an inner keep, outposts in the corners."""
    W = H = 101
    g = _canvas(W, H)
    _rect(g, 34, 35, 34, 66); _rect(g, 65, 66, 34, 66)
    _rect(g, 34, 66, 34, 35); _rect(g, 34, 66, 65, 66)
    for d in range(48, 53):
        g[34][d] = g[35][d] = g[65][d] = g[66][d] = "."
        g[d][34] = g[d][35] = g[d][65] = g[d][66] = "."
    _rect(g, 44, 44, 44, 47); _rect(g, 44, 47, 44, 44)            # inner keep corners
    _rect(g, 56, 56, 53, 56); _rect(g, 53, 56, 56, 56)
    for rr, cc in ((20, 20), (20, 50), (50, 20), (28, 72), (15, 35)):
        _rect(g, rr, rr + 2, cc, cc + 2)
    return _make("Fortress", g, "rot180", [(50, 50), (50, 27), (27, 50), (20, 80)], (84, 98, 2, 16), 130.0, False,
                 "Siege: the keep is worth holding, the gates are chokepoints, outposts give a safe income.")


def _map_meadow() -> MapDef11:
    """101x81: an open field with scattered rocks — no walls to hide behind for long."""
    W, H = 101, 81
    g = _canvas(W, H)
    rng = random.Random(1101)
    for _ in range(26):
        r, c = rng.randint(4, H - 5), rng.randint(16, W // 2 - 2)
        _circle(g, r, c, rng.choice([1.0, 1.0, 1.5, 2.0]))
    for r in range(3, H - 3):
        for c in range(1, 14):
            g[r][c] = "."                                           # keep the spawn band open
    for cr, cc in ((40, 50), (15, 50), (65, 50), (40, 24), (20, 35)):
        _circle(g, cr, cc, 2.5, ".")                                # clear the CP centres
    return _make("Meadow", g, "mirror_lr", [(40, 50), (15, 50), (65, 50), (40, 24), (20, 35)], (10, 70, 2, 12), 140.0, False,
                 "Open field: long exchanges, focus fire and spreading out decide it.")


def _map_offices() -> MapDef11:
    """91x71: an office floor — rooms off corridors, doors everywhere, few long lines."""
    W, H = 91, 71
    g = _canvas(W, H, "#")
    # Carved on the left half (cols 1..45); the right half is its mirror image, copied below.
    _rect(g, 1, H - 2, 1, 10, ".")                                   # lobby / spawn
    for row in (6, 22, 35, 48, 64):                                  # horizontal corridors, 3 tall
        _rect(g, row - 1, row + 1, 1, 45, ".")
    for col in (20, 45):                                             # vertical corridors, 3 wide
        _rect(g, 1, H - 2, col - 1, col + 1, ".")
    rooms = [(10, 18, 24, 40), (26, 31, 24, 40), (39, 44, 24, 40), (52, 60, 24, 40),
             (10, 18, 12, 16), (52, 60, 12, 16), (26, 31, 12, 16), (39, 44, 12, 16)]
    for r0, r1, c0, c1 in rooms:
        _rect(g, r0, r1, c0, c1, ".")
    doors = [(8, 28, 9, 29), (8, 36, 9, 37), (19, 32, 20, 33), (24, 30, 25, 31), (32, 34, 33, 35),
             (37, 28, 38, 29), (45, 36, 46, 37), (50, 30, 51, 31), (61, 34, 62, 35),
             (14, 17, 15, 18), (56, 17, 57, 18), (28, 17, 29, 18), (41, 17, 42, 18),
             (14, 41, 15, 43), (28, 41, 29, 43), (41, 41, 42, 43), (56, 41, 57, 43),
             (8, 13, 9, 14), (61, 13, 62, 14)]
    for r0, c0, r1, c1 in doors:
        _rect(g, r0, r1, c0, c1, ".")
    _rect(g, 30, 40, 42, 45, ".")                                    # atrium around the centre axis
    for r in range(H):
        for c in range(46, W):
            g[r][c] = g[r][W - 1 - c]
    return _make("Offices", g, "mirror_lr", [(35, 45), (6, 45), (64, 45), (35, 33), (14, 30)], (4, 66, 2, 9), 100.0, False,
                 "Office floor: room clearing, door holds, corridor crossfire.")


def _map_bridges() -> MapDef11:
    """111x61: a river of walls across the middle, crossed by three narrow bridges."""
    W, H = 111, 61
    g = _canvas(W, H)
    _rect(g, 1, 59, 50, 60)
    for r0 in (8, 28, 49):
        _rect(g, r0, r0 + 4, 50, 60, ".")
    for r, c in ((6, 42), (14, 44), (26, 44), (34, 44), (47, 42), (55, 44)):
        _rect(g, r, r + 1, c, c + 1)
    _rect(g, 20, 40, 30, 30)
    return _make("Bridges", g, "mirror_lr", [(10, 55), (30, 55), (51, 55), (30, 38), (12, 25)], (8, 52, 2, 12), 110.0, False,
                 "Chokepoints: three bridges, and a team that splits badly loses all of them.")


def _map_flanks() -> MapDef11:
    """101x61: a solid central block with one long tunnel; the real routes are two flanks."""
    W, H = 101, 61
    g = _canvas(W, H)
    _rect(g, 18, 42, 30, 70)
    _rect(g, 29, 31, 30, 70, ".")                                  # tunnel
    _rect(g, 29, 31, 48, 52, ".")
    for c in (38, 46):
        _rect(g, 8, 10, c, c + 1); _rect(g, 50, 52, c, c + 1)
    return _make("Flanks", g, "mirror_lr", [(30, 50), (8, 50), (52, 50), (30, 22), (8, 26)], (12, 48, 2, 12), 110.0, False,
                 "Two flanks and a death tunnel: commit to a side or split and trade.")


def _map_hill() -> MapDef11:
    """81x81: an open hill in the centre inside two broken rings whose gaps do not line up."""
    W = H = 81
    g = _canvas(W, H)
    for rad, gaps in ((12, (0, 90, 180, 270)), (22, (45, 135, 225, 315))):
        for deg10 in range(0, 3600, 5):
            a = math.radians(deg10 / 10)
            if any(abs(((deg10 / 10 - gd + 180) % 360) - 180) < 14 for gd in gaps):
                continue
            g[40 + int(round(rad * math.sin(a)))][40 + int(round(rad * math.cos(a)))] = "#"
    return _make("Hill", g, "rot180", [(40, 40), (40, 10), (10, 40), (16, 64)], (70, 78, 2, 10), 110.0, False,
                 "King of the hill: the centre scores most, but the rings force a spiral approach.")


def _map_pinwheel() -> MapDef11:
    """91x91: four bent arms around the centre — looks asymmetric, plays identical for both teams."""
    W = H = 91
    g = _canvas(W, H)
    _rect(g, 18, 35, 44, 46); _rect(g, 18, 20, 47, 60)             # arm 1: up from the hub, bends right
    _rect(g, 44, 46, 55, 72); _rect(g, 47, 60, 70, 72)             # arm 2: right from the hub, bends down
    _rect(g, 8, 10, 30, 36); _rect(g, 30, 32, 80, 82)              # cover; the rotation adds arms 3 and 4
    return _make("Pinwheel", g, "rot180", [(45, 45), (28, 62), (30, 30), (15, 70)], (78, 88, 2, 14), 120.0, False,
                 "Asymmetric-looking but fair (180-degree symmetry): every lane has a mirrored twin.")


def _designed_pool() -> list[MapDef11]:
    return [_map_crossfire(), _map_warehouse(), _map_maze(), _map_bunker(),
            _map_lanes(), _map_longshot(), _map_fortress(), _map_meadow(), _map_offices(),
            _map_bridges(), _map_flanks(), _map_hill(), _map_pinwheel()]
# endregion BLOCK_DESIGNED_LARGE


# region FUNC_validate_map
## @purpose Everything that makes a map fair and playable, as numbers: symmetry, reachability, balance, openness.
## @io MapDef -> dict (stats + "problems": list[str]; empty list = valid)
## @complexity 8
def _safe_tiles(g: list[list[str]], zone: tuple) -> list[tuple[int, int]]:
    _, r0, r1, c0, c1 = zone
    rows, cols = len(g), len(g[0])
    out = []
    for r in range(max(1, r0), min(rows - 1, r1 + 1)):
        for c in range(max(1, c0), min(cols - 1, c1 + 1)):
            if all(g[r + dr][c + dc] != "#" for dr in (-1, 0, 1) for dc in (-1, 0, 1) if 0 <= r + dr < rows and 0 <= c + dc < cols):
                out.append((r, c))
    return out


def _dijkstra(g: list[list[str]], sources: list[tuple[int, int]]) -> list[list[float]]:
    """Tile distance, 8-connected, diagonal only when both orthogonal neighbours are free — world11's flow rule."""
    rows, cols = len(g), len(g[0])
    inf = float("inf")
    d = [[inf] * cols for _ in range(rows)]
    pq = []
    for r, c in sources:
        d[r][c] = 0.0
        pq.append((0.0, r, c))
    heapq.heapify(pq)
    s2 = math.sqrt(2.0)
    while pq:
        dist, r, c = heapq.heappop(pq)
        if dist > d[r][c]:
            continue
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)):
            nr, nc = r + dr, c + dc
            if not (0 <= nr < rows and 0 <= nc < cols) or g[nr][nc] == "#":
                continue
            if dr and dc and (g[r + dr][c] == "#" or g[r][c + dc] == "#"):
                continue
            nd = dist + (s2 if dr and dc else 1.0)
            if nd < d[nr][nc]:
                d[nr][nc] = nd
                heapq.heappush(pq, (nd, nr, nc))
    return d


def _zone_tiles(m: MapDef, k: int) -> list[tuple[int, int]]:
    r0, c0 = m.cp_positions[k]
    cy, cx = r0 * m.tile_size + m.tile_size / 2, c0 * m.tile_size + m.tile_size / 2
    rad = int(math.ceil(m.cp_radius / m.tile_size)) + 1
    out = []
    for r in range(max(0, r0 - rad), min(m.rows, r0 + rad + 1)):
        for c in range(max(0, c0 - rad), min(m.cols, c0 + rad + 1)):
            if m.grid[r][c] != "#" and math.hypot(c * m.tile_size + m.tile_size / 2 - cx, r * m.tile_size + m.tile_size / 2 - cy) <= m.cp_radius:
                out.append((r, c))
    return out


def validate_map(m: MapDef) -> dict:
    g, rows, cols = m.grid, m.rows, m.cols
    sym = getattr(m, "symmetry", None)
    problems: list[str] = []
    st: dict = {"name": m.name, "rows": rows, "cols": cols, "symmetry": sym, "compact": bool(m.compact),
                "cp_radius": float(m.cp_radius), "kind": getattr(m, "kind", "legacy")}

    if len(m.cp_positions) != N_CPS:
        problems.append(f"cps={len(m.cp_positions)} != {N_CPS}")
    if rows > MAX_ROWS or cols > MAX_COLS:
        problems.append(f"size {rows}x{cols} exceeds {MAX_ROWS}x{MAX_COLS}")
    if m.tile_size != TILE:
        problems.append(f"tile {m.tile_size} != {TILE}")

    # Symmetry of walls, CP set and spawn zones.
    asym = 0
    if sym in SYMMETRIES:
        for r in range(rows):
            for c in range(cols):
                ir, ic = _image(r, c, rows, cols, sym)
                asym += (g[r][c] == "#") != (g[int(ir)][int(ic)] == "#")
        pair_err = 0.0
        for r, c in m.cp_positions:
            ir, ic = _image(r, c, rows, cols, sym)
            pair_err = max(pair_err, min(math.hypot(ir - r2, ic - c2) for r2, c2 in m.cp_positions))
        zb = _zone_image(m.spawn_zone_a, rows, cols, sym, "red")
        zone_ok = zb[1:] == tuple(m.spawn_zone_b[1:])
        st.update(asym_tiles=asym, cp_pair_err_tiles=round(pair_err, 3), spawn_zone_image_ok=zone_ok)
        if asym:
            problems.append(f"walls not symmetric under {sym}: {asym} tiles")
        if pair_err > PAIR_TOL_TILES:
            problems.append(f"CP set not symmetric: worst pairing error {pair_err:.2f} tiles")
        if not zone_ok:
            problems.append("spawn zone B is not the image of zone A")
    else:
        problems.append(f"unknown symmetry {sym}")

    # Spawns.
    sa, sb = _safe_tiles(g, m.spawn_zone_a), _safe_tiles(g, m.spawn_zone_b)
    st.update(safe_spawn_a=len(sa), safe_spawn_b=len(sb))
    if min(len(sa), len(sb)) < MIN_SAFE_SPAWN_TILES:
        problems.append(f"too few safe spawn tiles: {len(sa)}/{len(sb)}")
    if not sa or not sb:
        st["problems"] = problems
        return st

    da, db = _dijkstra(g, sa), _dijkstra(g, sb)
    free = [(r, c) for r in range(rows) for c in range(cols) if g[r][c] != "#"]
    unreachable = sum(1 for r, c in free if math.isinf(da[r][c]))
    sep = min(da[r][c] for r, c in sb)
    st.update(free_tiles=len(free), unreachable_free=unreachable, spawn_separation_tiles=round(sep, 1))
    if unreachable:
        problems.append(f"{unreachable} free tiles unreachable from spawn A")
    if sep < MIN_SPAWN_SEPARATION_TILES:
        problems.append(f"spawns only {sep:.1f} tiles apart")

    # CP zones: free, reachable, balanced between teams, not overlapping.
    ts = m.tile_size
    dist_a, dist_b, zone_sizes = [], [], []
    for k in range(len(m.cp_positions)):
        zt = _zone_tiles(m, k)
        zone_sizes.append(len(zt))
        if not zt:
            problems.append(f"CP{k} {m.cp_positions[k]}: empty capture zone")
            dist_a.append(float("inf")); dist_b.append(float("inf"))
            continue
        dist_a.append(min(da[r][c] for r, c in zt) * ts)
        dist_b.append(min(db[r][c] for r, c in zt) * ts)
        if math.isinf(dist_a[-1]) or math.isinf(dist_b[-1]):
            problems.append(f"CP{k}: zone unreachable")
    imbalance = 0.0
    if sym in SYMMETRIES and all(math.isfinite(x) for x in dist_a + dist_b):
        for k, (r, c) in enumerate(m.cp_positions):
            ir, ic = _image(r, c, rows, cols, sym)
            j = min(range(len(m.cp_positions)), key=lambda j: math.hypot(ir - m.cp_positions[j][0], ic - m.cp_positions[j][1]))
            imbalance = max(imbalance, abs(dist_a[k] - dist_b[j]))
        if imbalance > BALANCE_TOL_PX:
            problems.append(f"path-distance imbalance {imbalance:.0f} px > {BALANCE_TOL_PX:.0f}")
    overlap = 0
    for i in range(len(m.cp_positions)):
        for j in range(i + 1, len(m.cp_positions)):
            (r1, c1), (r2, c2) = m.cp_positions[i], m.cp_positions[j]
            if math.hypot(r1 - r2, c1 - c2) * ts < 2 * m.cp_radius:
                overlap += 1
    st.update(cp_zone_tiles=zone_sizes, cp_dist_a_px=[round(x) if math.isfinite(x) else None for x in dist_a],
              cp_dist_b_px=[round(x) if math.isfinite(x) else None for x in dist_b],
              balance_err_px=round(imbalance, 1), cp_overlaps=overlap)
    if overlap and getattr(m, "kind", "legacy") != "legacy":
        problems.append(f"{overlap} pairs of CP circles overlap")

    # Openness: share of free interior tiles; free-run length as a sightline proxy.
    interior = (rows - 2) * (cols - 2)
    runs = []
    for r, c in free:
        h = 1
        x = c - 1
        while x >= 0 and g[r][x] != "#":
            h += 1; x -= 1
        x = c + 1
        while x < cols and g[r][x] != "#":
            h += 1; x += 1
        v = 1
        y = r - 1
        while y >= 0 and g[y][c] != "#":
            v += 1; y -= 1
        y = r + 1
        while y < rows and g[y][c] != "#":
            v += 1; y += 1
        runs.append(max(h, v))
    st.update(free_share=round(len(free) / max(1, interior), 3), mean_sightline_tiles=round(sum(runs) / max(1, len(runs)), 1),
              max_sightline_tiles=max(runs) if runs else 0)
    st["problems"] = problems
    return st
# endregion FUNC_validate_map


# region FUNC_procedural_map
## @purpose A seeded, symmetric, validated random map. Same seed -> same map, always valid.
## @io (seed, compact | None) -> MapDef11
## @complexity 8
## @rationale Design half, symmetrize, then repair instead of rejecting: sealed pockets are filled, a
## disconnected spawn gets a corridor carved along the middle row, CP centres are cleared. Rejection is
## kept only as the last resort (a new derived seed), so the retry count stays small and deterministic.
def _fill_pockets(g: list[list[str]], seeds: list[tuple[int, int]]) -> int:
    rows, cols = len(g), len(g[0])
    seen = [[False] * cols for _ in range(rows)]
    q = deque()
    for r, c in seeds:
        if g[r][c] != "#" and not seen[r][c]:
            seen[r][c] = True
            q.append((r, c))
    while q:
        r, c = q.popleft()
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and not seen[nr][nc] and g[nr][nc] != "#":
                seen[nr][nc] = True
                q.append((nr, nc))
    filled = 0
    for r in range(rows):
        for c in range(cols):
            if g[r][c] != "#" and not seen[r][c]:
                g[r][c] = "#"
                filled += 1
    return filled


def _attempt(seed: int, compact: bool | None) -> MapDef11 | None:
    rng = random.Random(seed)
    if compact is None:
        compact = rng.random() < 0.4
    if compact:
        W, H = rng.choice(range(35, 47, 2)), rng.choice(range(31, 45, 2))
        cp_r = 60.0
    else:
        W, H = rng.choice(range(71, 124, 2)), rng.choice(range(51, 92, 2))
        cp_r = float(rng.choice((100, 110, 120, 130)))
    sym = rng.choice(("mirror_lr", "rot180"))
    style = rng.choice(("scatter", "pillars", "rooms", "lanes"))
    g = _canvas(W, H)
    half = W // 2
    spawn_w = 8 if compact else 12
    x0 = spawn_w + 3
    if style == "scatter":
        for _ in range(int(W * H / (60 if compact else 90))):
            r, c = rng.randint(2, H - 3), rng.randint(x0, half)
            _rect(g, r, r + rng.randint(0, 2), c, c + rng.randint(0, 3))
    elif style == "pillars":
        for _ in range(int(W * H / (220 if compact else 260))):       # denser merged into one blob
            _circle(g, rng.randint(3, H - 4), rng.randint(x0, half), rng.choice((1.0, 1.5, 2.0, 3.0)))
    elif style == "rooms":
        for _ in range(rng.randint(3, 6 if compact else 9)):
            r0, c0 = rng.randint(2, H - 10), rng.randint(x0, max(x0 + 1, half - 6))
            h, w = rng.randint(5, 9), rng.randint(5, 10)
            _rect(g, r0, r0 + h, c0, c0, "#"); _rect(g, r0, r0 + h, c0 + w, c0 + w, "#")
            _rect(g, r0, r0, c0, c0 + w, "#"); _rect(g, r0 + h, r0 + h, c0, c0 + w, "#")
            for _d in range(2):                                     # two doors
                if rng.random() < 0.5:
                    rr = rng.randint(r0 + 1, r0 + h - 2); cc = rng.choice((c0, c0 + w))
                    _rect(g, rr, rr + 1, cc, cc, ".")
                else:
                    cc = rng.randint(c0 + 1, c0 + w - 2); rr = rng.choice((r0, r0 + h))
                    _rect(g, rr, rr, cc, cc + 1, ".")
    else:  # lanes
        for row in sorted(rng.sample(range(6, H - 6), k=min(2, max(1, H // 20)))):
            _rect(g, row, row, x0, half)
            for _ in range(rng.randint(1, 3)):
                c = rng.randint(x0, half - 3)
                _rect(g, row, row, c, c + 2, ".")
    # Open spawn band, then symmetrize.
    _rect(g, 1, H - 2, 1, spawn_w, ".")
    _symmetrize(g, sym)

    # CPs: one on the axis (centre for rot180), three seeded pairs in the left half.
    centre = (H // 2, W // 2)
    seeds = [centre]
    min_gap = 2 * cp_r / TILE + 1
    tries = 0
    while len(seeds) < 4 and tries < 400:
        tries += 1
        r, c = rng.randint(3, H - 4), rng.randint(x0 + 1, half - 2)
        ir, ic = _image(r, c, H, W, sym)
        cand = [(r, c), (int(ir), int(ic))]
        pts = seeds + [p for s in seeds[1:] for p in [(int(_image(s[0], s[1], H, W, sym)[0]), int(_image(s[0], s[1], H, W, sym)[1]))]]
        if all(math.hypot(a[0] - b[0], a[1] - b[1]) >= min_gap for a in cand for b in pts) and math.hypot(ir - r, ic - c) >= min_gap:
            seeds.append((r, c))
    if len(seeds) < 4:
        return None
    for r, c in seeds:
        _circle(g, r, c, 1.5, ".")
    _symmetrize(g, sym)          # wall union keeps symmetry; clearing was on seeds only — clear the images too
    for r, c in seeds:
        ir, ic = _image(r, c, H, W, sym)
        _circle(g, r, c, 1.5, "."); _circle(g, int(ir), int(ic), 1.5, ".")

    zone_a = (2, H - 3, 2, spawn_w - 1)
    za = ("blue", *zone_a)
    zb = _zone_image(za, H, W, sym, "red")
    # Connectivity: carve a middle corridor if the spawns do not see each other, then fill sealed pockets.
    sa = _safe_tiles(g, za)
    if not sa:
        return None
    d = _dijkstra(g, sa)
    if all(math.isinf(d[r][c]) for r, c in _safe_tiles(g, zb)):
        _rect(g, H // 2 - 1, H // 2 + 1, 1, W - 2, ".")
    _fill_pockets(g, _safe_tiles(g, za) + _safe_tiles(g, zb))
    name = f"Proc{seed % 100000:05d}"
    tactic = f"procedural {style}, {sym}"
    m = MapDef11(name=name, cols=W, rows=H, tile_size=TILE, grid=g, cp_positions=_cps(seeds, H, W, sym),
                 spawn_zone_a=za, spawn_zone_b=zb, cp_radius=cp_r, compact=compact,
                 symmetry=sym, tactic=tactic, kind="procedural", seed=seed)
    return m


def procedural_map(seed: int, compact: bool | None = None, max_attempts: int = 64) -> MapDef11:
    for attempt in range(max_attempts):
        derived = seed if attempt == 0 else (seed * 1_000_003 + attempt * 7919) % (1 << 31)
        m = _attempt(derived, compact)
        if m is None:
            continue
        v = validate_map(m)
        if not v["problems"]:
            m.name = f"Proc{seed % 100000:05d}"
            m.seed = seed
            m.tactic += f" (attempt {attempt})"
            return m
    raise RuntimeError(f"procedural_map: no valid map for seed {seed} in {max_attempts} attempts")
# endregion FUNC_procedural_map


# region FUNC_build_pool
## @purpose Pool by name. "maps11" = legacy (fixed) + designed; "legacy" = map_gen as frozen; "maps11+proc:N" adds
## N procedural maps derived from `seed`.
## @io (name, seed) -> list[MapDef]
def build_pool(name: str = "maps11", seed: int = 0) -> list[MapDef]:
    if name == "legacy":
        pool: list[MapDef] = map_gen.build_map_pool()
    elif name == "maps11" or name.startswith("maps11+proc:"):
        pool = _legacy_pool() + _designed_pool()
        if name.startswith("maps11+proc:"):
            n = int(name.split(":", 1)[1])
            rng = random.Random(seed ^ 0x5EED11)
            # Keep the curriculum's two classes both represented among procedural maps.
            pool += [procedural_map(rng.randrange(1 << 30), compact=(i % 3 == 0)) for i in range(n)]
    else:
        raise ValueError(f"unknown map pool {name!r}: use 'maps11', 'legacy' or 'maps11+proc:N'")
    logger.info(f"[IMP:9][build_pool][BUILD] pool={name}, maps={len(pool)}, compact={sum(bool(m.compact) for m in pool)}, "
                f"names={[m.name for m in pool]} [VALUE]")
    return pool
# endregion FUNC_build_pool
