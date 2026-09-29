from __future__ import annotations

# region MODULE_CONTRACT [DOMAIN(8): LevelDesign; CONCEPT(9): BigBattleMaps; TECH(7): Python+PIL]
## @modulecontract
## @purpose Maps for the big-battle show (50v50, 150v150, 500v500): large symmetric layouts whose structure reads
## at extreme zoom-out (fronts, rivers, ridges, a citadel, CP clusters), with the rules of each battle attached.
## @scope Map data, rules, validation (symmetry, reachability, multi-spawn path balance, density), PNG renders
## for docs/maps_big, and a flow-field cost probe for world_big. No world, no policy.
## @input Map name (BIG_MAPS key)
## @output BigMap objects (a maps11.MapDef11 subclass: world code reads the same fields plus the rules)
## @links LINKS_TO: maps11 (grid helpers, Dijkstra), docs/BIGBATTLE_CONTRACT.md, world_big (consumer)
## @invariants
## - Every map is exactly invariant under its team transform: walls, the CP set, the spawn-zone set
## - Every free tile is reachable from both teams' spawns; every CP zone holds free reachable tiles
## - Team path distance to a CP equals the other team's distance to its image within BALANCE_TOL_PX
## - CP capture circles never overlap; agents per CP stays within AGENTS_PER_CP
## - Odd width and height: the symmetry axis runs through tile centres, CPs on the axis are their own image
## @rationale
## Q: Why several spawn rectangles per team?
## A: 500 agents respawning in one box would queue behind each other for half a minute and fight one door.
## A: Spawn rectangles along the back edge put reinforcements on every front at once.
## Q: Why thick walls (>= 4 tiles) for the structural features?
## A: The 500v500 map viewed whole on a laptop is ~2 screen pixels per tile. A 1-tile wall is invisible there; a
## A: 4-8 tile ridge or river is a line the viewer can follow. Fine cover (rocks, houses) is for close zoom.
## Q: Why score_to_win = 180 x CPs?
## A: world11 pays 0.02 score per owned CP per frame. A team holding ~60% of N points earns 0.012 N per frame,
## A: so 180 N is reached in ~15000 frames = ~4 minutes of game time — the show's target of 3-6 minutes.
## Q: Why team_lives = 10 x team_size?
## A: exp11b 5v5 matches see ~1.4 deaths per agent-minute; six-minute battles need ~8 lives per agent before
## A: elimination can pre-empt the score race.
## @changes
## LAST_CHANGE: [v0.1.0] Front50, Front150, Crossing150, Warfront500; validation, renders, flow-field probe.
## @modulemap
## CLASS 8[MapDef11 plus battle rules, spawn sets, sectors, CP names] => BigMap
## BLOCK 8[Construction helpers] => _cps_any / _zones_image / _finish
## FUNC 9[Front: trench line, ridges, central redoubt] => _front
## FUNC 9[Crossing: river with five bridges] => _crossing150
## FUNC 9[Warfront: canyons, plains, citadel] => _warfront500
## FUNC 9[Validation] => validate_big
## FUNC 7[Renders] => render_full / render_zoom
## FUNC 8[Flow-field cost probe] => flow_probe
## FUNC 7[CLI] => main
## @usecases
## - world_big: from rush.maps_big import BIG_MAPS; m = BIG_MAPS["Front50"]()
## - python -m rush.maps_big --check --render --flow-probe Warfront500
def _module_contract():
    pass
# endregion MODULE_CONTRACT
# GREP_SUMMARY: big battle, maps, 500v500, symmetry, spawn zones, sectors, zoom-out, flow field, rules
# STRUCTURE: ▶ draw half-features → ⚡ symmetrize → ⚡ carve CPs + fill pockets → ○ validate → ⎋ BigMap (+ renders)

import argparse
import json
import logging
import math
import random
import sys
import time
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from rush.maps11 import (
    MapDef11,
    _canvas,
    _circle,
    _dijkstra,
    _fill_pockets,
    _image,
    _rect,
    _safe_tiles,
    _symmetrize,
    _zone_tiles,
)

logger = logging.getLogger(__name__)

# region BLOCK_CONSTANTS
TILE: int = 30
CP_SCORE_PER_FRAME: float = 0.02         # world11's income per owned CP per frame (not imported: no torch here)
SCORE_PER_CP: int = 180
LIVES_PER_AGENT: int = 10
MAX_FRAMES: int = 21600                  # 6 minutes at 60 FPS
AGENTS_PER_CP: tuple[float, float] = (7.0, 12.0)
TILES_PER_AGENT: tuple[float, float] = (200.0, 450.0)
BALANCE_TOL_PX: float = 30.0
PAIR_TOL_TILES: float = 0.5
MIN_SPAWN_SEPARATION_TILES: float = 60.0
CP_CARVE_TILES: int = 2                  # every CP centre gets at least this free disc
# Minimum width of every main route, by battle size (redesign 25.09.2026): ~10 / 15 / 20 agents abreast.
# Every blob, bastion and landform keeps at least this gap to its neighbours; checked by chokepoint_stats.
WMIN_BY_TEAM: dict[int, int] = {50: 8, 150: 12, 500: 16}
REPO = Path(__file__).resolve().parent
OUT_DEFAULT = REPO / "docs" / "maps_big"
# endregion BLOCK_CONSTANTS


# region CLASS_BigMap
## @purpose One battle: MapDef11 geometry plus the rules and the metadata the world and the viewer need.
@dataclass
class BigMap(MapDef11):
    team_size: int = 50
    score_to_win: int = 2400
    team_lives: int = 500
    max_frames: int = MAX_FRAMES
    spawn_zones_a: list[tuple[str, int, int, int, int]] = field(default_factory=list)
    spawn_zones_b: list[tuple[str, int, int, int, int]] = field(default_factory=list)
    sectors: list[dict] = field(default_factory=list)       # {"name", "rect": [r0, r1, c0, c1]}
    cp_names: list[str] = field(default_factory=list)
    concept: str = ""
# endregion CLASS_BigMap


# region BLOCK_HELPERS
def _cps_any(seeds: list[tuple[int, int]], centre: list[tuple[int, int]], rows: int, cols: int, sym: str) -> list[tuple[int, int]]:
    """Axis/centre points once, every seed plus its exact image. Seeds must be off the axis."""
    out = list(centre)
    for r, c in centre:
        ir, ic = _image(r, c, rows, cols, sym)
        if (int(ir), int(ic)) != (r, c):
            raise ValueError(f"centre CP {(r, c)} is not its own image under {sym}")
    for r, c in seeds:
        ir, ic = _image(r, c, rows, cols, sym)
        if (int(ir), int(ic)) == (r, c):
            raise ValueError(f"seed CP {(r, c)} lies on the axis; list it as a centre point")
        out += [(r, c), (int(ir), int(ic))]
    return out


def _zones_image(zones: list[tuple[str, int, int, int, int]], rows: int, cols: int, sym: str) -> list[tuple[str, int, int, int, int]]:
    out = []
    for _, r0, r1, c0, c1 in zones:
        a, b = _image(r0, c0, rows, cols, sym), _image(r1, c1, rows, cols, sym)
        out.append(("red", int(min(a[0], b[0])), int(max(a[0], b[0])), int(min(a[1], b[1])), int(max(a[1], b[1]))))
    return out


def _blob(g: list[list[str]], rng: random.Random, r: int, c: int, rad: float) -> None:
    """One solid rounded rock/crater rim: overlapping discs with no gaps inside — cover up close, a shape far away.
    (The old rock clumps were fields of 1-2-tile pebbles: every gap between them was a chokepoint.)"""
    _circle(g, r, c, rad * 0.8)
    for _ in range(rng.randint(2, 4)):
        a, d = rng.uniform(0, 2 * math.pi), rad * rng.uniform(0.2, 0.45)
        _circle(g, int(round(r + d * math.sin(a))), int(round(c + d * math.cos(a))), rad * rng.uniform(0.5, 0.7))


class _Scatter:
    """Places blobs so that every gap they leave is at least `gap` tiles: to other blobs, to keep-out discs (CP
    zones), keep-out rectangles (spawns, ridges, bastions) and to the blob's own symmetry image (the image is
    stamped by _finish, and two halves of a blob must not pinch a passage on the axis)."""

    def __init__(self, rows: int, cols: int, sym: str, gap: float) -> None:
        self.rows, self.cols, self.sym, self.gap = rows, cols, sym, gap
        self.blobs: list[tuple[float, float, float]] = []
        self.discs: list[tuple[float, float, float]] = []
        self.rects: list[tuple[int, int, int, int]] = []

    def ok(self, r: float, c: float, rad: float) -> bool:
        g = self.gap
        if min(r, c, self.rows - 1 - r, self.cols - 1 - c) < rad + 2:
            return False
        ir, ic = _image(r, c, self.rows, self.cols, self.sym)
        d_img = math.hypot(r - ir, c - ic)
        if 0 < d_img < 2 * rad + g:
            return False
        for br, bc, brad in self.blobs:
            if math.hypot(r - br, c - bc) < rad + brad + g:
                return False
        for dr, dc, drad in self.discs:
            if math.hypot(r - dr, c - dc) < rad + drad + g:
                return False
        for r0, r1, c0, c1 in self.rects:
            dy = max(r0 - r, 0, r - r1)
            dx = max(c0 - c, 0, c - c1)
            if math.hypot(dx, dy) < rad + g:
                return False
        return True

    def place(self, g: list[list[str]], rng: random.Random, n: int, rad: tuple[float, float],
              region: tuple[int, int, int, int], tries: int = 40) -> int:
        r0, r1, c0, c1 = region
        placed = 0
        for _ in range(n * tries):
            if placed >= n:
                break
            rd = rng.uniform(*rad)
            reach = rd * 1.15 + 1                           # _blob's discs reach 0.45·rad + 0.7·rad, plus rounding
            r, c = rng.randint(r0, r1), rng.randint(c0, c1)
            if self.ok(r, c, reach):
                _blob(g, rng, r, c, rd)
                self.blobs.append((r, c, reach))
                placed += 1
        return placed


def _cp_names(cps: list[tuple[int, int]], sectors: list[dict]) -> list[str]:
    """Sector initial + running number, in reading order within the sector: 'Ц1', 'К3'."""
    names = [""] * len(cps)
    for s in sectors:
        r0, r1, c0, c1 = s["rect"]
        inside = sorted((k for k, (r, c) in enumerate(cps) if r0 <= r <= r1 and c0 <= c <= c1 and not names[k]),
                        key=lambda k: (cps[k][0], cps[k][1]))
        for i, k in enumerate(inside, 1):
            names[k] = f"{s['tag']}{i}"
    return [n or f"T{k}" for k, n in enumerate(names)]


def _finish(name: str, g: list[list[str]], sym: str, centre_cps: list[tuple[int, int]], cp_seeds: list[tuple[int, int]],
            zones_a: list[tuple[int, int, int, int]], cp_radius: float, team_size: int, sectors: list[dict], concept: str) -> BigMap:
    """Symmetrize, carve CP centres, seal pockets, attach rules. Everything after this is exactly symmetric."""
    rows, cols = len(g), len(g[0])
    cps = _cps_any(cp_seeds, centre_cps, rows, cols, sym)
    _symmetrize(g, sym)
    for r, c in cps:                                   # the CP set is symmetric, so the carve is too
        _circle(g, r, c, CP_CARVE_TILES, ".")
    za = [("blue", *z) for z in zones_a]
    zb = _zones_image(za, rows, cols, sym)
    for _, r0, r1, c0, c1 in za + zb:                  # spawn boxes are open ground
        _rect(g, r0, r1, c0, c1, ".")
    seeds = [t for z in za + zb for t in _safe_tiles(g, z)]
    _fill_pockets(g, seeds)
    sealed = [rc for rc in cps if g[rc[0]][rc[1]] == "#"]
    if sealed:                                          # a carved CP disc was an enclosed pocket and got refilled
        raise ValueError(f"{name}: CP centres sealed in walls: {sealed}")
    n_cp = len(cps)
    m = BigMap(name=name, cols=cols, rows=rows, tile_size=TILE, grid=g, cp_positions=cps,
               spawn_zone_a=za[0], spawn_zone_b=zb[0], cp_radius=cp_radius, compact=False,
               symmetry=sym, tactic=concept.split(".")[0], kind="big",
               team_size=team_size, score_to_win=int(round(SCORE_PER_CP * n_cp, -2)),
               team_lives=LIVES_PER_AGENT * team_size, max_frames=MAX_FRAMES,
               spawn_zones_a=za, spawn_zones_b=zb, sectors=sectors, concept=concept)
    m.cp_names = _cp_names(cps, sectors)
    return m
# endregion BLOCK_HELPERS


# region FUNC__front
## @purpose "Front line": two massive ridges that split the field into three sectors, a map-wide no-man's belt of
## craters, four bastions around the centre redoubt, rock blobs for cover. Redesign 25.09.2026: no fences, no gates —
## every gap >= WMIN_BY_TEAM (the old trench lines and ring walls made 4-9-tile chokepoints).
## @rationale rot180: the ridge that shields blue's west flank is red's east ridge — both teams face the
## same asymmetric-looking but equal front. Scales by size; the CP grid scales with the team.
def _front(name: str, team_size: int, cols: int, rows: int, cp_radius: float, grid_rows: list[float], k_cols: int, seed: int) -> BigMap:
    sym = "rot180"
    g = _canvas(cols, rows)
    rng = random.Random(seed)
    mr, mc = (rows - 1) // 2, (cols - 1) // 2
    rr = int(math.ceil(cp_radius / TILE))
    wmin = WMIN_BY_TEAM[team_size]
    bl = int(rows * 0.16)                              # rear edge of the field (the base trench line is gone)
    sc = _Scatter(rows, cols, sym, wmin)

    # CPs first: every obstacle below keeps its distance from them.
    fwd = 2 * rr + 4
    seeds = [(mr - fwd, mc), (mr, int(cols * 0.2))]
    for fr in grid_rows:
        r = int(rows * fr)
        for i in range(k_cols):
            seeds.append((r, int(cols * (0.1 + 0.8 * (i + 0.5) / k_cols))))
    zones = []
    n_z = max(2, team_size // 25)
    zw = min(50, (cols - 20) // (n_z * 2))
    for i in range(n_z):
        cc = int(cols * (i + 0.5) / n_z)
        zones.append((3, 3 + max(8, rows // 16), cc - zw // 2, cc + zw // 2))
    sc.discs += [(r, c, rr + 1) for r, c in _cps_any(seeds, [(mr, mc)], rows, cols, sym)]
    sc.rects += [z for z in zones]

    # Two ridges split the field into three sectors. Massive landforms, not fences: they end far from the axis, so
    # the three sectors reconnect across a front as wide as the map.
    ridge_w = max(8, cols // 30)
    ridge_end = mr - (2 * rr + 10 + wmin)
    for cc in (cols // 3, 2 * cols // 3):
        box = (bl + 6, ridge_end, cc - ridge_w // 2, cc + ridge_w // 2)
        _rect(g, *box)
        sc.rects.append(box)

    # Centre redoubt: four corner bastions around the centre point, gaps of 2R-b >= wmin between them
    # (was a walled ring with 5-tile gates).
    R = rr + 6
    b = max(5, R // 2)
    for dr in (-1, 1):
        for dc in (-1, 1):
            box = (mr + dr * R - b // 2, mr + dr * R + b // 2, mc + dc * R - b // 2, mc + dc * R + b // 2)
            _rect(g, *box)
            sc.rects.append((min(box[:2]), max(box[:2]), min(box[2:]), max(box[2:])))

    # Cover: craters on the no-man's belt, rock blobs across the field. Upper half only — the image fills the rest;
    # _Scatter keeps every gap (blob-blob, blob-ridge, blob-point, blob-image) >= wmin.
    s = 1.0 if team_size <= 50 else 1.3
    sc.place(g, rng, cols // 12, (2.5 * s, 4.0 * s), (mr - 3 * rr - 8, mr - 4, 4, cols - 5))
    sc.place(g, rng, (mr - bl) * cols // 900, (3.0 * s, 6.0 * s), (bl + 4, mr - 6, 4, cols - 5))
    sectors = [
        {"name": "Западный фронт", "tag": "З", "rect": [0, rows - 1, 0, cols // 3 - 1]},
        {"name": "Редут", "tag": "Р", "rect": [mr - R - 3, mr + R + 3, mc - R - 3, mc + R + 3]},
        {"name": "Центр", "tag": "Ц", "rect": [0, rows - 1, cols // 3, 2 * cols // 3 - 1]},
        {"name": "Восточный фронт", "tag": "В", "rect": [0, rows - 1, 2 * cols // 3, cols - 1]},
    ]
    concept = ("Front line. Two massive ridges cut the field into three sectors that reconnect across a map-wide "
               "no-man's belt of craters; four bastions guard the centre redoubt; rock blobs give cover, and no route "
               f"to any point is narrower than {wmin} tiles.")
    return _finish(name, g, sym, [(mr, mc)], seeds, zones, cp_radius, team_size, sectors, concept)
# endregion FUNC__front


# region FUNC__crossing150
## @purpose "River crossing": a 17-tile river splits the map; seven 25-tile fords are the only way across. Each ford
## carries a CP and is guarded by blockhouses; villages between the fords and rear points behind them.
## Redesign 25.09.2026: 5 bridges of 11 tiles (8 effective behind the trench segments) -> 7 fords of 25.
def _crossing150() -> BigMap:
    sym = "mirror_tb"
    cols, rows = 381, 241
    g = _canvas(cols, rows)
    rng = random.Random(150)
    ax = (rows - 1) // 2                               # 120
    wmin = WMIN_BY_TEAM[150]
    fords = [int(cols * (i + 0.5) / 7) for i in range(7)]   # 27 81 136 190 244 299 353
    half_ford = 12                                     # 25-tile fords (were five 11-tile bridges, 8 effective)
    sc = _Scatter(rows, cols, sym, wmin)
    # River (on the axis, symmetric by itself) with seven wide fords.
    _rect(g, ax - 8, ax + 8, 1, cols - 2)
    for cb in fords:
        _rect(g, ax - 8, ax + 8, cb - half_ford, cb + half_ford, ".")
    sc.rects.append((ax - 8, ax + 8, 1, cols - 2))
    # Bridgeheads: two blockhouses set back from each ford exit — cover for the defenders, >= wmin to the bank and
    # to each other (were trench segments and a shield wall that pinched the exit to 8 tiles).
    for cb in fords:
        for dc in (-17, 17):
            box = (ax - 27, ax - 22, cb + dc - 3, cb + dc + 3)
            _rect(g, *box)
            sc.rects.append(box)
    # Villages between fords: four solid houses around a square, streets >= 17 tiles.
    villages, vr = [54, 163, 217, 326], 66             # row 66: houses end 14 rows above the blockhouses
    for vc in villages:
        for dr, dc in ((-11, -13), (-11, 13), (11, -13), (11, 13)):
            box = (vr + dr - 2, vr + dr + 2, vc + dc - 3, vc + dc + 3)
            _rect(g, *box)
            sc.rects.append(box)
    centre = [(ax, cb) for cb in fords]
    rear = [(34, 110), (34, 190), (34, 270)]
    seeds = [(ax - 20, cb) for cb in fords] + [(vr, vc) for vc in villages] + rear
    zones = [(4, 14, 20, 80), (4, 14, 160, 220), (4, 14, 300, 360)]
    rr = int(math.ceil(210.0 / TILE))
    sc.discs += [(r, c, rr + 1) for r, c in _cps_any(seeds, centre, rows, cols, sym)]
    sc.rects += list(zones)
    # Woods: solid rock blobs on the north bank, the mirror fills the south.
    sc.place(g, rng, 90, (4.0, 7.0), (20, ax - 12, 6, cols - 7))
    names = [f"Брод {t}" for t in "ABCDEFG"]
    edges = [0] + [(fords[i] + fords[i + 1]) // 2 for i in range(6)] + [cols]
    sectors = [{"name": names[i], "tag": "ABCDEFG"[i], "rect": [0, rows - 1, edges[i], edges[i + 1] - 1]} for i in range(7)]
    concept = ("River crossing. A 17-tile river cuts the map in two; seven 25-tile fords are the only crossings and "
               "every ford is a point. Blockhouses guard the ford exits, villages and rear points sit between them, "
               f"rock blobs give cover — no route narrower than {wmin} tiles.")
    return _finish("Crossing150", g, sym, centre, seeds, zones, 210.0, 150, sectors, concept)
# endregion FUNC__crossing150


# region FUNC__warfront500
## @purpose "Warfront": three sectors split by ridges — a canyon maze, a central citadel, open plains — joined by a
## wide no-man's belt across the middle. rot180 makes one team's canyon the other team's plains, so both armies
## attack through both terrains.
def _warfront500() -> BigMap:
    sym = "rot180"
    cols, rows = 721, 441
    g = _canvas(cols, rows)
    rng = random.Random(500)
    mr, mc = 220, 360
    wmin = WMIN_BY_TEAM[500]
    sc = _Scatter(rows, cols, sym, wmin)
    centre = [(mr, mc)]
    seeds = [(195, 335), (195, 385), (160, 360)]                                        # citadel + north gate
    seeds += [(r, c) for r in (40, 90, 140, 190) for c in (30, 90, 150, 210)]            # canyons
    seeds += [(r, c) for r in (45, 100, 155, 200) for c in (520, 580, 640, 700)]         # plains
    seeds += [(r, c) for r in (40, 90, 135) for c in (270, 330, 390, 450) if (r, c) not in ((90, 330), (90, 390))]
    seeds += [(90, 300), (90, 420)]                                                      # centre sector
    seeds += [(210, 60), (210, 180), (210, 540), (210, 660)]                             # the belt
    zones = [(4, 16, c, c + 60) for c in (25, 125, 225, 325, 425, 525, 625)]
    rr = int(math.ceil(240.0 / TILE))
    sc.discs += [(r, c, rr + 1) for r, c in _cps_any(seeds, centre, rows, cols, sym)]
    sc.rects += list(zones)
    # No base trench line (its 11-tile gates were the first chokepoint of every attack).
    # Ridges split sectors in the upper band (images close the lower band); they end 120 tiles before their images,
    # so the belt between them is a map-wide front.
    for c0 in (236, 477):
        box = (32, 160, c0, c0 + 9)
        _rect(g, *box)
        sc.rects.append(box)
    # West canyons: a grid of mesas with 24-34-tile lanes. The half-mesas against the ridge now touch it
    # (a 2-tile slot between them used to be a dead-end pocket).
    for rc in (65, 115, 165):
        for cc in (60, 120, 180):
            _rect(g, rc - 13, rc + 13, cc - 13, cc + 13)
            sc.rects.append((rc - 13, rc + 13, cc - 13, cc + 13))
    for rc in (65, 115, 165):
        _rect(g, rc - 10, rc + 10, 1, 8)
        _rect(g, rc - 10, rc + 10, 222, 236)
        sc.rects += [(rc - 10, rc + 10, 1, 8), (rc - 10, rc + 10, 222, 236)]
    # Centre sector outside the citadel: three hard-points.
    for rc, cc in ((65, 300), (65, 420), (110, 360)):
        _rect(g, rc - 6, rc + 6, cc - 6, cc + 6)
        sc.rects.append((rc - 6, rc + 6, cc - 6, cc + 6))
    # Citadel: four L-shaped corner bastions (half size 45, arms 22, 6 thick) — 46-tile openings on every side —
    # and a keep of four 8-tile pillars around the centre point, 28-tile passages between them (was a ring with
    # 9-tile gates and a keep with 7-tile diagonal gates).
    H, L, T = 45, 22, 6
    for dr in (-1, 1):
        for dc in (-1, 1):
            r_out, c_out = mr + dr * H, mc + dc * H
            arm_r = (r_out, r_out - dr * (T - 1), c_out, c_out - dc * (L - 1))
            arm_c = (r_out, r_out - dr * (L - 1), c_out, c_out - dc * (T - 1))
            for a in (arm_r, arm_c):
                _rect(g, *a)
                sc.rects.append((min(a[:2]), max(a[:2]), min(a[2:]), max(a[2:])))
    K, P = 18, 4
    for dr in (-1, 1):
        for dc in (-1, 1):
            _rect(g, mr + dr * K - P, mr + dr * K + P, mc + dc * K - P, mc + dc * K + P)
    sc.rects.append((mr - H, mr + H, mc - H, mc + H))                 # no blobs inside the citadel
    # East plains: solid rock blobs (were 420 pebbles of radius 1.5-5 — a field of 2-4-tile slots).
    sc.place(g, rng, 60, (4.0, 8.0), (34, 205, 495, cols - 6))
    # Canyons floor and the centre sector: a few boulders.
    sc.place(g, rng, 12, (3.0, 5.0), (34, 200, 10, 225))
    sc.place(g, rng, 14, (3.0, 5.0), (34, 170, 250, 470))
    # No-man's belt: craters (were staggered 3-tile trench segments).
    sc.place(g, rng, 40, (4.0, 7.0), (172, mr - 6, 6, cols - 7))
    sectors = [
        {"name": "Цитадель", "tag": "Ц", "rect": [mr - H, mr + H, mc - H, mc + H]},
        {"name": "Каньоны (север)", "tag": "К", "rect": [0, mr, 0, 235]},
        {"name": "Равнины (юг)", "tag": "Р", "rect": [mr + 1, rows - 1, 0, 235]},
        {"name": "Равнины (север)", "tag": "П", "rect": [0, mr, 485, cols - 1]},
        {"name": "Каньоны (юг)", "tag": "Ю", "rect": [mr + 1, rows - 1, 485, cols - 1]},
        {"name": "Центральный фронт", "tag": "Ф", "rect": [0, rows - 1, 236, 484]},
    ]
    concept = ("Warfront. Ridges split the field into three sectors — canyons between mesas, a citadel of four "
               "bastions around a pillared keep, open plains of boulders — joined by a map-wide no-man's belt of "
               "craters. rot180 turns one army's canyons into the other's plains: both armies attack through both "
               f"terrains. No route narrower than {wmin} tiles.")
    return _finish("Warfront500", g, sym, centre, seeds, zones, 240.0, 500, sectors, concept)
# endregion FUNC__warfront500


# region BLOCK_REGISTRY
BIG_MAPS = {
    "Front50": lambda: _front("Front50", 50, 201, 141, 180.0, [0.27], 4, seed=50),
    "Front150": lambda: _front("Front150", 150, 347, 243, 210.0, [0.26, 0.37], 7, seed=151),
    "Crossing150": _crossing150,
    "Warfront500": _warfront500,
}


@lru_cache(maxsize=None)
def get_map(name: str) -> BigMap:
    """Built once per process; callers must not mutate the grid."""
    return BIG_MAPS[name]()
# endregion BLOCK_REGISTRY


# region FUNC_validate_big
## @purpose Fairness, reachability, density and rule sanity of one big map. Returns stats + a problem list.
## @complexity 8
def validate_big(m: BigMap) -> dict:
    g, rows, cols, sym, ts = m.grid, m.rows, m.cols, m.symmetry, m.tile_size
    n_agents = 2 * m.team_size
    problems: list[str] = []
    st: dict = {"name": m.name, "rows": rows, "cols": cols, "symmetry": sym, "team_size": m.team_size,
                "cps": len(m.cp_positions), "cp_radius": m.cp_radius, "score_to_win": m.score_to_win,
                "team_lives": m.team_lives, "max_frames": m.max_frames}
    if rows % 2 == 0 or cols % 2 == 0:
        problems.append("even dimension")

    asym = sum((g[r][c] == "#") != (g[int(_image(r, c, rows, cols, sym)[0])][int(_image(r, c, rows, cols, sym)[1])] == "#")
               for r in range(rows) for c in range(cols))
    cps = m.cp_positions
    cpset = set(cps)
    pair_err = 0.0
    for r, c in cps:
        ir, ic = _image(r, c, rows, cols, sym)
        pair_err = max(pair_err, 0.0 if (int(ir), int(ic)) in cpset else min(math.hypot(ir - a, ic - b) for a, b in cps))
    zones_ok = sorted(z[1:] for z in _zones_image(m.spawn_zones_a, rows, cols, sym)) == sorted(z[1:] for z in m.spawn_zones_b)
    st.update(asym_tiles=asym, cp_pair_err_tiles=round(pair_err, 3), spawn_zones_image_ok=zones_ok)
    if asym:
        problems.append(f"walls not symmetric: {asym} tiles")
    if pair_err > PAIR_TOL_TILES:
        problems.append(f"CP pairing error {pair_err:.2f} tiles")
    if not zones_ok:
        problems.append("spawn zones B are not the image of A")

    sa = [t for z in m.spawn_zones_a for t in _safe_tiles(g, z)]
    sb = [t for z in m.spawn_zones_b for t in _safe_tiles(g, z)]
    st.update(safe_spawn_a=len(sa), safe_spawn_b=len(sb))
    if min(len(sa), len(sb)) < 2 * m.team_size:
        problems.append(f"safe spawn tiles {len(sa)}/{len(sb)} < 2 x team")

    t0 = time.perf_counter()
    da, db = _dijkstra(g, sa), _dijkstra(g, sb)
    st["dijkstra_s"] = round(time.perf_counter() - t0, 2)
    free = [(r, c) for r in range(rows) for c in range(cols) if g[r][c] != "#"]
    unreachable = sum(1 for r, c in free if math.isinf(da[r][c]) or math.isinf(db[r][c]))
    sep = min(da[r][c] for r, c in sb)
    st.update(free_tiles=len(free), unreachable_free=unreachable, spawn_separation_tiles=round(sep, 1),
              free_tiles_per_agent=round(len(free) / n_agents, 1), agents_per_cp=round(n_agents / len(cps), 2))
    if unreachable:
        problems.append(f"{unreachable} free tiles unreachable")
    if sep < MIN_SPAWN_SEPARATION_TILES:
        problems.append(f"spawns {sep:.1f} tiles apart")
    if not TILES_PER_AGENT[0] <= len(free) / n_agents <= TILES_PER_AGENT[1]:
        problems.append(f"density {len(free) / n_agents:.0f} tiles/agent outside {TILES_PER_AGENT}")
    if not AGENTS_PER_CP[0] <= n_agents / len(cps) <= AGENTS_PER_CP[1]:
        problems.append(f"{n_agents / len(cps):.1f} agents per CP outside {AGENTS_PER_CP}")

    dist_a, dist_b, zsz = [], [], []
    for k in range(len(cps)):
        zt = _zone_tiles(m, k)
        zsz.append(len(zt))
        if not zt:
            problems.append(f"CP {m.cp_names[k]} empty zone")
            dist_a.append(math.inf); dist_b.append(math.inf)
            continue
        dist_a.append(min(da[r][c] for r, c in zt) * ts)
        dist_b.append(min(db[r][c] for r, c in zt) * ts)
    imbalance = 0.0
    for k, (r, c) in enumerate(cps):
        ir, ic = _image(r, c, rows, cols, sym)
        j = cps.index((int(ir), int(ic))) if (int(ir), int(ic)) in cpset else k
        if math.isfinite(dist_a[k]) and math.isfinite(dist_b[j]):
            imbalance = max(imbalance, abs(dist_a[k] - dist_b[j]))
    overlap = sum(1 for i in range(len(cps)) for j in range(i + 1, len(cps))
                  if math.hypot(cps[i][0] - cps[j][0], cps[i][1] - cps[j][1]) * ts < 2 * m.cp_radius)
    st.update(balance_err_px=round(imbalance, 1), cp_overlaps=overlap, min_cp_zone_tiles=min(zsz),
              nearest_cp_from_spawn_px=round(min(dist_a)), farthest_cp_from_spawn_px=round(max(dist_a)))
    if imbalance > BALANCE_TOL_PX:
        problems.append(f"path-distance imbalance {imbalance:.0f} px")
    if overlap:
        problems.append(f"{overlap} overlapping CP circles")

    # Openness via row/column free-run lengths, O(tiles).
    hrun = [[0] * cols for _ in range(rows)]
    for r in range(rows):
        c = 0
        while c < cols:
            if g[r][c] == "#":
                c += 1
                continue
            s = c
            while c < cols and g[r][c] != "#":
                c += 1
            for x in range(s, c):
                hrun[r][x] = c - s
    vrun_max = 0
    runs_total = 0
    for c in range(cols):
        r = 0
        while r < rows:
            if g[r][c] == "#":
                r += 1
                continue
            s = r
            while r < rows and g[r][c] != "#":
                r += 1
            for y in range(s, r):
                runs_total += max(hrun[y][c], r - s)
            vrun_max = max(vrun_max, r - s)
    st.update(free_share=round(len(free) / ((rows - 2) * (cols - 2)), 3),
              mean_sightline_tiles=round(runs_total / max(1, len(free)), 1))
    est_frames = m.score_to_win / (CP_SCORE_PER_FRAME * 0.6 * len(cps))
    st["est_match_minutes_at_60pct"] = round(est_frames / 3600, 2)
    st["problems"] = problems
    return st
# endregion FUNC_validate_big


# region FUNC_expectations
## @purpose The designer's beliefs, written before the check ran; the CLI prints MATCH/MISMATCH per line.
def expectations(m: BigMap) -> dict:
    n_cp = {"Front50": 13, "Front150": 33, "Crossing150": 35, "Warfront500": 103}[m.name]
    return {
        "problems": ("==", 0), "asym_tiles": ("==", 0), "cp_pair_err_tiles": ("==", 0.0),
        "spawn_zones_image_ok": ("==", True), "unreachable_free": ("==", 0), "cp_overlaps": ("==", 0),
        "balance_err_px": ("<=", BALANCE_TOL_PX), "cps": ("==", n_cp),
        "free_tiles_per_agent": ("in", TILES_PER_AGENT), "agents_per_cp": ("in", AGENTS_PER_CP),
        "est_match_minutes_at_60pct": ("in", (3.0, 6.0)), "min_cp_zone_tiles": (">=", 20),
        "free_share": ("in", (0.80, 0.97)),
        # redesign 25.09.2026 — no fences, no route narrower than WMIN_BY_TEAM, little cramped ground
        "fence_walls": ("==", 0), "cps_bottleneck_lt_wmin": ("==", 0), "cps_need_lt_wmin_route": ("==", 0),
        "cramped_area_share_w_lt6": ("<=", 0.15),
    }


def _compare(val, op, ref) -> bool:
    if op == "==":
        return val == ref
    if op == "<=":
        return val <= ref
    if op == ">=":
        return val >= ref
    return ref[0] <= val <= ref[1]
# endregion FUNC_expectations


# region FUNC_render
## @purpose Neon PNGs: the whole map at "laptop" scale (fits 1440 px wide) and an 8 px/tile crop at the centre.
COL_BG, COL_WALL, COL_EDGE = (6, 9, 21), (58, 50, 140), (138, 123, 255)
COL_BLUE, COL_RED, COL_CP, COL_TXT = (53, 200, 255), (255, 132, 51), (185, 179, 227), (214, 208, 255)
FONT_PATHS = ("/System/Library/Fonts/Supplemental/Arial Bold.ttf", "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")


def _font(size: int):
    """A TTF with Cyrillic: PIL's built-in bitmap font draws Russian labels as boxes."""
    from PIL import ImageFont
    for p in FONT_PATHS:
        if Path(p).exists():
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def _draw(m: BigMap, s: float, box: tuple[int, int, int, int], labels: bool):
    from PIL import Image, ImageDraw
    r0, r1, c0, c1 = box
    w, h = int((c1 - c0 + 1) * s), int((r1 - r0 + 1) * s)
    img = Image.new("RGB", (w, h), COL_BG)
    d = ImageDraw.Draw(img)
    for zones, col in ((m.spawn_zones_a, COL_BLUE), (m.spawn_zones_b, COL_RED)):
        for _, a, b, c, e in zones:
            d.rectangle([(c - c0) * s, (a - r0) * s, (e - c0 + 1) * s - 1, (b - r0 + 1) * s - 1], fill=tuple(int(v * 0.3) for v in col))
    for r in range(r0, r1 + 1):
        row = m.grid[r]
        c = c0
        while c <= c1:
            if row[c] != "#":
                c += 1
                continue
            e = c
            while e + 1 <= c1 and row[e + 1] == "#":
                e += 1
            d.rectangle([(c - c0) * s, (r - r0) * s, (e - c0 + 1) * s - 1, (r - r0 + 1) * s - 1], fill=COL_WALL)
            c = e + 1
    if s >= 6:                                     # wall edges only where a tile is big enough to show them
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                if m.grid[r][c] == "#" and any(0 <= r + dr < m.rows and 0 <= c + dc < m.cols and m.grid[r + dr][c + dc] != "#"
                                               for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1))):
                    d.rectangle([(c - c0) * s, (r - r0) * s, (c - c0 + 1) * s - 1, (r - r0 + 1) * s - 1], outline=COL_EDGE)
    rad = m.cp_radius / TILE * s
    font = _font(max(11, int(s * 1.6)))
    for k, (r, c) in enumerate(m.cp_positions):
        if not (r0 <= r <= r1 and c0 <= c <= c1):
            continue
        cx, cy = (c - c0 + 0.5) * s, (r - r0 + 0.5) * s
        d.ellipse([cx - rad, cy - rad, cx + rad, cy + rad], outline=COL_CP, width=max(1, int(s // 3)))
        if labels:
            d.text((cx, cy), m.cp_names[k], fill=COL_TXT, font=font, anchor="mm")
    return img, d


def render_full(m: BigMap, path: Path) -> None:
    s = min(1440 / m.cols, 900 / m.rows)
    img, d = _draw(m, s, (0, m.rows - 1, 0, m.cols - 1), labels=False)
    font = _font(15)
    for sec in m.sectors:
        r0, r1, c0, c1 = sec["rect"]
        d.text(((c0 + c1) / 2 * s, (r0 + 3) * s + (18 if r0 == 0 else 6)), sec["name"], fill=COL_TXT, font=font, anchor="mt")
    img.save(path)


def render_zoom(m: BigMap, path: Path) -> None:
    mr, mc = (m.rows - 1) // 2, (m.cols - 1) // 2
    hw, hh = 60, 38
    img, _ = _draw(m, 8, (max(0, mr - hh), min(m.rows - 1, mr + hh), max(0, mc - hw), min(m.cols - 1, mc + hw)), labels=True)
    img.save(path)
# endregion FUNC_render


# region FUNC_chokepoint_stats
## @purpose Quantify how much a map funnels the armies through narrow passages ("fighting through gaps in a fence").
## @io (wall bool[R,C], spawn tiles, CP tiles, cross rows) -> dict of width statistics, all in tiles
## @rationale
## Q: Which widths?
## A: "Width W is passable" = a disc of diameter W fits: free tiles with clearance >= (W+1)/2 (Euclidean distance
## A: transform to the nearest wall centre) are connected. An agent is 0.8 tile wide; a 500-strong army needs
## A: routes an order of magnitude wider than that or it queues. Reported per CP: the bottleneck (widest W that
## A: still reaches it from the spawns) and the detour when routes narrower than 6 / 10 tiles are forbidden.
## A: Map-wide: share of free area whose local width 2*clearance-1 < 6, openings on cross lines, thin long walls.
CHOKE_WIDTHS: tuple[int, ...] = (2, 4, 6, 8, 10, 12, 16, 20, 30)


def _grid_graph(mask: "np.ndarray"):
    """8-connected graph over True tiles; diagonals only when both orthogonal neighbours are open (world rule)."""
    import numpy as np
    from scipy.sparse import coo_matrix
    R, C = mask.shape
    idx = -np.ones((R, C), np.int64)
    idx[mask] = np.arange(int(mask.sum()))
    P = np.pad(mask, 1)                                 # False ring: shifted views never leave the grid
    I = np.pad(idx, 1, constant_values=-1)

    def sh(a, dr, dc):
        return a[1 + dr:R + 1 + dr, 1 + dc:C + 1 + dc]

    rows, cols, w = [], [], []
    for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
        ok = mask & sh(P, dr, dc)
        if dr and dc:                                   # diagonal: both orthogonal steps open
            ok &= sh(P, dr, 0) & sh(P, 0, dc)
        ia, ib = idx[ok], sh(I, dr, dc)[ok]
        rows += [ia, ib]; cols += [ib, ia]
        w += [np.full(len(ia), math.sqrt(2) if dr and dc else 1.0)] * 2
    n = int(mask.sum())
    g = coo_matrix((np.concatenate(w), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n)).tocsr()
    return g, idx


def chokepoint_stats(wall: "np.ndarray", spawn: list[tuple[int, int]], cps: list[list[tuple[int, int]]],
                     cross_rows: list[int], wmin: int = 10) -> dict:
    import numpy as np
    from scipy import ndimage
    from scipy.sparse.csgraph import dijkstra
    free = ~wall
    R, C = wall.shape
    interior = wall.copy()
    interior[0, :] = interior[-1, :] = interior[:, 0] = interior[:, -1] = False
    rr, cc = np.mgrid[0:R, 0:C]
    to_border = np.minimum(np.minimum(rr, R - 1 - rr), np.minimum(cc, C - 1 - cc)).astype(float)
    # The map edge is where spawns live, not a passage wall: it only limits clearance weakly (2d+1), otherwise
    # every spawn box and every route along the edge would count as a chokepoint.
    clear = np.minimum(ndimage.distance_transform_edt(~interior), 2 * to_border + 1)
    clear[wall] = 0
    local_w = 2 * clear - 1
    out: dict = {"cramped_area_share_w_lt6": round(float((local_w[free] < 6).mean()), 3),
                 "cramped_area_share_w_lt10": round(float((local_w[free] < 10).mean()), 3)}

    # Spawn boxes and CP zones are always passable: they sit against the map edge or a wall by design, and the
    # question is how wide the way BETWEEN them is. A run of w free tiles has max clearance ceil(w/2), hence W/2.
    ends = np.zeros_like(free)
    for r, c in spawn:
        ends[r, c] = True
    for zone in cps:
        for r, c in zone:
            ends[r, c] = True

    def dists(W: int) -> "np.ndarray":
        mask = (free & (clear >= W / 2)) | (ends & free) if W > 1 else free
        g, idx = _grid_graph(mask)
        src = [idx[r, c] for r, c in spawn if idx[r, c] >= 0]
        if not src:
            return np.full(len(cps), np.inf)
        d = dijkstra(g, indices=src, min_only=True)
        res = []
        for zone in cps:
            ids = [idx[r, c] for r, c in zone if idx[r, c] >= 0]
            res.append(float(d[ids].min()) if ids else math.inf)
        return np.array(res)

    base = dists(1)
    bottleneck = np.zeros(len(cps))
    detour = {}
    for W in sorted(set(CHOKE_WIDTHS) | {wmin}):
        dW = dists(W)
        bottleneck = np.where(np.isfinite(dW), W, bottleneck)
        if W in (6, 10, wmin):
            ratio = dW / np.maximum(base, 1e-9)
            detour[W] = ratio
    out["bottleneck_w_median"] = float(np.median(bottleneck))
    out["bottleneck_w_min"] = float(bottleneck.min())
    out["cps_bottleneck_lt6"] = int((bottleneck < 6).sum())
    out["cps_bottleneck_lt10"] = int((bottleneck < 10).sum())
    for W, ratio in detour.items():
        out[f"cps_need_lt{W}_route"] = int((~np.isfinite(ratio) | (ratio > 1.10)).sum())   # >10% detour or unreachable
    out["wmin"] = wmin
    out["cps_bottleneck_lt_wmin"] = int((bottleneck < wmin).sum())
    out["cps_need_lt_wmin_route"] = out[f"cps_need_lt{wmin}_route"]
    out["cps"] = len(cps)

    openings = []
    for r in cross_rows:
        row = free[r]
        runs, c = [], 0
        while c < len(row):
            if row[c]:
                s = c
                while c < len(row) and row[c]:
                    c += 1
                runs.append(c - s)
            else:
                c += 1
        openings.append({"row": r, "openings": len(runs), "open_share": round(float(row.mean()), 3),
                         "narrow_lt6": sum(1 for x in runs if x < 6), "widest": max(runs) if runs else 0})
    out["cross_lines"] = openings

    # Thin long walls ("fence" segments): wall components no thicker than 3 tiles and at least 12 tiles long.
    lab, n = ndimage.label(wall[1:-1, 1:-1], structure=np.ones((3, 3)))
    wall_clear = ndimage.distance_transform_edt(wall[1:-1, 1:-1])
    fences = 0
    if n:
        thick = ndimage.maximum(wall_clear, lab, index=np.arange(1, n + 1))
        sl = ndimage.find_objects(lab)
        for k, s in enumerate(sl):
            length = max(s[0].stop - s[0].start, s[1].stop - s[1].start)
            if thick[k] * 2 - 1 <= 3 and length >= 12:
                fences += 1
    out["fence_walls"] = fences
    return out


def map_chokepoints(m: BigMap) -> dict:
    import numpy as np
    wall = np.array([[ch == "#" for ch in row] for row in m.grid])
    spawn = [t for z in m.spawn_zones_a for t in _safe_tiles(m.grid, z)]
    zones = [_zone_tiles(m, k) for k in range(len(m.cp_positions))]
    mr = (m.rows - 1) // 2
    return chokepoint_stats(wall, spawn, zones, cross_rows=[mr, int(m.rows * 0.3), int(m.rows * 0.4)],
                            wmin=WMIN_BY_TEAM.get(m.team_size, 10))
# endregion FUNC_chokepoint_stats


# region FUNC_flow_probe
## @purpose Cost of the per-CP flow fields world_big must build: exact per-CP Dijkstra time (sampled), the number of
## relaxation sweeps a world11-style tensor relaxation needs (= longest shortest path in steps), and its memory.
def flow_probe(m: BigMap, samples: int = 4) -> dict:
    t0 = time.perf_counter()
    far = 0.0
    idx = list(range(0, len(m.cp_positions), max(1, len(m.cp_positions) // samples)))[:samples]
    for k in idx:
        d = _dijkstra(m.grid, _zone_tiles(m, k))
        far = max(far, max(v for row in d for v in row if math.isfinite(v)))
    per_cp = (time.perf_counter() - t0) / len(idx)
    cells = m.rows * m.cols * len(m.cp_positions)
    out = {"cps": len(m.cp_positions), "tiles": m.rows * m.cols, "dijkstra_per_cp_s": round(per_cp, 2),
           "dijkstra_all_cps_s_python": round(per_cp * len(m.cp_positions), 1),
           "longest_path_tiles": round(far, 1), "relax_sweeps_needed": int(math.ceil(far)),
           "dist_tensor_MB_fp32": round(cells * 4 / 2**20, 1),
           "world11_style_peak_MB_fp32": round(cells * 4 * 10 / 2**20, 1)}
    try:
        import torch
        dist = torch.full((len(m.cp_positions), m.rows, m.cols), float("inf"))
        wall = torch.tensor([[ch == "#" for ch in row] for row in m.grid])
        for k, (r, c) in enumerate(m.cp_positions):
            dist[k, r, c] = 0.0
        t1, n_it = time.perf_counter(), 5
        for _ in range(n_it):
            best = dist.clone()
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                sh = torch.roll(dist, shifts=(dr, dc), dims=(1, 2)) + 1.0
                best = torch.minimum(best, sh)
            dist = torch.where(wall, torch.full_like(dist, float("inf")), best)
        out["torch_cpu_sweep_s"] = round((time.perf_counter() - t1) / n_it, 3)
        out["torch_cpu_total_est_s"] = round(out["torch_cpu_sweep_s"] * out["relax_sweeps_needed"] * 2, 1)
    except Exception as exc:  # torch is optional for the map module
        out["torch_error"] = str(exc)[:120]
    return out
# endregion FUNC_flow_probe


# region FUNC_main
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Big-battle maps: check, render, flow-field probe")
    ap.add_argument("--maps", nargs="*", default=list(BIG_MAPS))
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--render", action="store_true")
    ap.add_argument("--flow-probe", nargs="*", default=None)
    ap.add_argument("--out", default=str(OUT_DEFAULT))
    ap.add_argument("--chokepoints", action="store_true", help="passage-width statistics per map")
    ap.add_argument("--chokepoints-synth", type=int, nargs="*", default=None,
                    help="same statistics for the synthetic viewer fixture at these team sizes")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", handlers=[logging.StreamHandler(sys.stdout)])
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    report, bad = {}, 0
    for name in args.maps:
        t0 = time.perf_counter()
        m = get_map(name)
        logger.info(f"[IMP:9][main][BUILD] {name}: {m.rows}x{m.cols} tiles, cps={len(m.cp_positions)}, team={m.team_size}, "
                    f"zones={len(m.spawn_zones_a)}, score_to_win={m.score_to_win}, lives={m.team_lives}, build={time.perf_counter() - t0:.2f}s [VALUE]")
        if args.check:
            st = validate_big(m)
            st.update({k: v for k, v in map_chokepoints(m).items() if k != "cross_lines"})
            report[name] = st
            for key, (op, ref) in expectations(m).items():
                val = len(st["problems"]) if key == "problems" else st[key]
                ok = _compare(val, op, ref)
                bad += not ok
                logger.info(f"[IMP:9][main][BELIEF] {name}.{key}: expected {op} {ref}, got {val} -> {'MATCH' if ok else 'MISMATCH'} [VALUE]")
            logger.info(f"[IMP:9][main][RESULT] {name}: {json.dumps({k: v for k, v in st.items() if k != 'problems'}, ensure_ascii=False)} problems={st['problems']} [VALUE]")
        if args.render:
            render_full(m, out / f"{name}_full.png")
            render_zoom(m, out / f"{name}_zoom.png")
            logger.info(f"[IMP:8][main][EXEC] rendered {name}_full.png, {name}_zoom.png [VALUE]")
    if args.chokepoints:
        for name in args.maps:
            t0 = time.perf_counter()
            cs = map_chokepoints(get_map(name))
            report.setdefault(name, {})["chokepoints"] = cs
            logger.info(f"[IMP:9][main][RESULT] chokepoints {name} ({time.perf_counter() - t0:.1f}s): {json.dumps(cs, ensure_ascii=False)} [VALUE]")
    if args.chokepoints_synth is not None:
        import numpy as np
        from rush.tools.synth_big_replay import synth_chokepoints
        for n in args.chokepoints_synth or [500]:
            cs = synth_chokepoints(n)
            report.setdefault(f"synth{n}", {})["chokepoints"] = cs
            logger.info(f"[IMP:9][main][RESULT] chokepoints synth{n}: {json.dumps(cs, ensure_ascii=False)} [VALUE]")
    if args.flow_probe is not None:
        for name in args.flow_probe or ["Warfront500"]:
            fp = flow_probe(get_map(name))
            report.setdefault(name, {})["flow_probe"] = fp
            logger.info(f"[IMP:9][main][RESULT] flow probe {name}: {fp} [VALUE]")
    if report:
        (out / "map_check.json").write_text(json.dumps(report, ensure_ascii=False, indent=1))
    logger.info(f"[IMP:10][main][RESULT] mismatches={bad} [VALUE]")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
# endregion FUNC_main
