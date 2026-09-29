from __future__ import annotations

# region MODULE_CONTRACT [DOMAIN(8): Maps; CONCEPT(9): ThemedMaps; TECH(7): numpy, scipy]
## @modulecontract
## @purpose Themed procedural maps for experiment 14, small (7 CPs, batched World13 at 5/10/20 per side) and
## big (BigMap, WorldBig12 at 50/150/500 per side): steppe, delta, urban, forest, plateau. Plus the "mix14" pools.
## @scope Grid generation (left half + symmetrize), CP seeds, spawn zones, validation, preview renders, disk cache.
## @input theme, team size, seed
## @output MapDef11 (small) / maps_big.BigMap (big); pools by name "mix14:<ts>[:K]"
## @links USES_API(8): numpy, scipy.ndimage, PIL; LINKS_TO: maps11 (helpers, validate_map), maps13 (wide maps,
## EXTRA_POOLS hook), maps_big (BigMap, _finish, validate_big, chokepoint_stats, render_full), docs/EXP14_CONTRACT.md
## @invariants
## - every map: exactly symmetric (mirror_lr or rot180), no fence walls (thin long walls), no CP whose best route
##   needs a passage narrower than WMIN; small maps pass maps11.validate_map (size limit excepted), big maps
##   pass maps_big.validate_big with no problems
## - every obstacle is placed only where its gap to every wall, keep-out zone (CP discs, spawn boxes) and to its
##   own symmetry image is >= WMIN tiles; long obstacles are >= 5 tiles thick (fence_walls counts <= 3)
## - deterministic for (theme, team size, seed)
## @rationale
## Q: Why a clearance test on every candidate instead of fixed layouts?
## A: The redesign (docs/BIG_NOTES_redesign.md) showed that "gaps in a fence" come from obstacles placed without a
## A: minimum gap. Testing the gap per candidate keeps every theme's character (clusters, fords, streets, belts,
## A: rings) and still guarantees that an army of any size can move between them.
## @changes
## LAST_CHANGE: [v0.1.0] Initial: five themes, small and big, mix14 pools, previews.
## @modulemap
## FUNC 8[Candidate shapes per theme] => _shape
## FUNC 9[Themed grid: river / scatter with clearance] => _themed_grid
## FUNC 8[Small themed map] => themed_small
## FUNC 9[Big themed map] => themed_big
## FUNC 7[Pools] => build_pool14, big_pool
## FUNC 6[Preview PNGs] => render_small, main
def _module_contract():
    pass
# endregion MODULE_CONTRACT
# GREP_SUMMARY: maps14, themes, steppe, delta, urban, forest, plateau, procedural, big maps, mix14 pool, previews
# STRUCTURE: ▶ size from team → ⚡ CP seeds + keep-out → ○ theme shapes with clearance (left half) → ⚡ symmetrize → ◇ validate + chokepoints → ⎋ map

import argparse
import logging
import math
import pickle
import random
import sys
import time
from pathlib import Path

import numpy as np

from rush import maps11
from rush.maps11 import (
    N_CPS,
    TILE,
    MapDef11,
    _canvas,
    _circle,
    _cps,
    _fill_pockets,
    _image,
    _safe_tiles,
    _symmetrize,
    _zone_image,
    _zone_tiles,
    validate_map,
)

logger = logging.getLogger(__name__)

# region BLOCK_CONSTANTS
THEMES: tuple[str, ...] = ("steppe", "delta", "urban", "forest", "plateau")
THEME_RU: dict[str, str] = {"steppe": "Степь", "delta": "Дельта", "urban": "Город", "forest": "Лесополосы", "plateau": "Плато"}
SMALL_TILES_PER_AGENT: dict[int, float] = {5: 420.0, 10: 360.0, 20: 300.0}
SMALL_WMIN: dict[int, int] = {5: 6, 10: 8, 20: 8}   # 5v5: 6 tiles = 7 agents abreast; 8 left room for ~3 shapes
SMALL_CP_RADIUS: dict[int, float] = {5: 120.0, 10: 150.0, 20: 180.0}
BIG_TILES_PER_AGENT: float = 300.0
BIG_AGENTS_PER_CP: float = 9.0
BIG_CP_RADIUS: dict[int, float] = {50: 180.0, 150: 210.0, 500: 240.0}
# Target wall share of the interior per theme (steppe and forest are open fields, urban is dense).
WALL_SHARE: dict[str, float] = {"steppe": 0.11, "delta": 0.12, "urban": 0.16, "forest": 0.13, "plateau": 0.12}
MIN_THICK: int = 5                   # chokepoint_stats calls a wall component <= 3 thick and >= 12 long a fence
CACHE_DIR: Path = Path.home() / ".cache" / "rush" / "maps14"
PREVIEW_DIR: Path = Path(__file__).resolve().parent / "docs" / "maps14"
# endregion BLOCK_CONSTANTS


def _wmin_big(team_size: int) -> int:
    from rush.maps_big import WMIN_BY_TEAM
    return WMIN_BY_TEAM.get(team_size, 16 if team_size >= 300 else 12 if team_size >= 100 else 8)


# region FUNC__shape
## @purpose One candidate obstacle mask (bool H x W) of a theme around (r, c); s scales sizes with the map.
## @complexity 7
def _shape(theme: str, rng: random.Random, H: int, W: int, r: int, c: int, s: float) -> np.ndarray:
    yy, xx = np.ogrid[:H, :W]
    m = np.zeros((H, W), dtype=bool)

    def disc(rr: float, cc: float, rad: float) -> None:
        nonlocal m
        m |= np.hypot(xx - cc, yy - rr) <= rad

    if theme == "steppe":                                   # a rock cluster: overlapping discs, one solid mass
        pts = [(float(r), float(c), rng.uniform(2.5, 3.5) * s)]
        for _ in range(rng.randint(2, 5)):
            br, bc, brad = pts[rng.randrange(len(pts))]
            rad = rng.uniform(2.0, 3.2) * s
            a = rng.uniform(0, 2 * math.pi)
            d = 0.7 * (brad + rad) * rng.uniform(0.5, 1.0)
            pts.append((br + d * math.sin(a), bc + d * math.cos(a), rad))
        for pr, pc, rad in pts:
            disc(pr, pc, rad)
    elif theme == "forest":                                 # a thick tree belt with rounded ends, or a copse
        if rng.random() < 0.7:
            length = rng.uniform(12, 24) * s
            half = rng.uniform(2.6, 3.4) * max(1.0, s * 0.8)   # radius >= 2.6 -> belt >= 5 tiles thick
            a = rng.choice((0.0, math.pi / 2, math.pi / 4, -math.pi / 4)) + rng.uniform(-0.25, 0.25)
            n = max(2, int(length / max(1.0, half)))
            for k in range(n + 1):
                t = (k / n - 0.5) * length
                disc(r + t * math.sin(a), c + t * math.cos(a), half)
        else:
            for _ in range(rng.randint(2, 4)):
                disc(r + rng.uniform(-2, 2) * s, c + rng.uniform(-2, 2) * s, rng.uniform(2.6, 3.6) * s)
    elif theme == "urban":                                  # a building block: rectangle or L
        h, w = int(rng.uniform(5, 10) * s), int(rng.uniform(5, 11) * s)
        m[max(0, r - h // 2): r + h - h // 2, max(0, c - w // 2): c + w - w // 2] = True
        if rng.random() < 0.35:
            h2, w2 = int(rng.uniform(5, 7) * s), int(rng.uniform(8, 14) * s)
            m[max(0, r - h // 2): r - h // 2 + h2, max(0, c - w // 2): c - w // 2 + w2] = True
    elif theme == "plateau":                                # a mesa, or a ring arc (thick) with an opening
        if rng.random() < 0.55:
            disc(r, c, rng.uniform(3.5, 6.5) * s)
        else:
            R = rng.uniform(7, 11) * s
            thick = rng.uniform(5.0, 6.0)
            d = np.hypot(xx - c, yy - r)
            ang = np.arctan2(yy - r, xx - c)
            a0 = rng.uniform(-math.pi, math.pi)
            span = rng.uniform(math.pi * 0.6, math.pi * 1.2)
            da = (ang - a0 + math.pi) % (2 * math.pi) - math.pi
            m |= (np.abs(d - R) <= thick / 2) & (np.abs(da) <= span / 2)
    elif theme == "delta":                                  # islands and reed beds between the channels
        if rng.random() < 0.5:
            disc(r, c, rng.uniform(3.0, 5.5) * s)
        else:
            length = rng.uniform(8, 16) * s
            a = rng.uniform(-0.6, 0.6)
            for k in range(6):
                t = (k / 5 - 0.5) * length
                disc(r + t * math.sin(a), c + t * math.cos(a), 2.7 * max(1.0, s * 0.8))
    else:
        raise ValueError(f"unknown theme {theme!r}")
    m[0, :] = m[-1, :] = False
    m[:, 0] = m[:, -1] = False
    return m
# endregion FUNC__shape


# region FUNC__clear
## @purpose True if every tile of `cand` is at least `gap + 1` tiles (Euclidean) from every `blocked` tile.
## Evaluated in a window around the candidate: anything farther than the window margin cannot violate the gap.
def _clear(blocked: np.ndarray, cand: np.ndarray, gap: int) -> bool:
    from scipy import ndimage
    rr, cc = np.nonzero(cand)
    if rr.size == 0:
        return False
    m = gap + 2
    r0, r1 = max(0, rr.min() - m), min(cand.shape[0], rr.max() + m + 1)
    c0, c1 = max(0, cc.min() - m), min(cand.shape[1], cc.max() + m + 1)
    bw = blocked[r0:r1, c0:c1]
    if not bw.any():
        return True
    if (bw & cand[r0:r1, c0:c1]).any():
        return False
    d = ndimage.distance_transform_edt(~cand[r0:r1, c0:c1])
    return bool(d[bw].min() >= gap + 1)
# endregion FUNC__clear


# region FUNC__themed_grid
## @purpose Fill the LEFT half of a canvas with theme obstacles (the caller symmetrizes). Returns (grid, info).
## keep: bool mask of keep-out tiles (CP discs, spawn boxes) that obstacles must stay >= wmin away from;
## axis_margin: obstacles end this far before the symmetry axis (so an obstacle and its image leave >= wmin).
## @complexity 8
def _themed_grid(theme: str, rng: random.Random, H: int, W: int, sym: str, keep: np.ndarray, wmin: int, s: float,
                 x0: int, river: np.ndarray | None = None) -> tuple[list[list[str]], dict]:
    g = _canvas(W, H)
    wall = np.zeros((H, W), dtype=bool)                       # interior obstacles only; the border is a margin
    if river is not None:
        wall |= river
        for rr_, cc_ in zip(*np.nonzero(river)):
            g[rr_][cc_] = "#"
    half = (W - 1) // 2
    limit = half - (wmin // 2 + 1)                            # candidate columns end here (image gap >= wmin + 2)
    # The map edge is not a passage wall (maps_big.chokepoint_stats counts it weakly) and a CP zone is always
    # passable: obstacles keep half the gap from both; routes to the zones are verified by chokepoint_stats.
    bm, keep_gap = wmin // 2 + 1, wmin // 2
    target = WALL_SHARE[theme]
    interior = (H - 2) * (W - 2)
    placed, tries, max_tries = 0, 0, int(600 + 0.05 * H * W)
    while tries < max_tries:
        tries += 1
        if 2 * wall[1:-1, 1:-1].sum() / interior >= target:   # the image doubles what is placed
            break
        r, c = rng.randint(bm, H - 1 - bm), rng.randint(x0, max(x0 + 1, limit))
        cand = _shape(theme, rng, H, W, r, c, s)
        cand[:, limit + 1:] = False
        cand[:, :x0] = False
        cand[:bm, :] = False
        cand[H - bm:, :] = False
        if not cand.any():
            continue
        if not _clear(wall, cand, wmin) or not _clear(keep, cand, keep_gap):
            continue
        wall |= cand
        for rr_, cc_ in zip(*np.nonzero(cand)):
            g[rr_][cc_] = "#"
        placed += 1
    _symmetrize(g, sym)
    return g, {"placed": placed, "tries": tries, "wall_share_half": round(float(2 * wall[1:-1, 1:-1].sum() / interior), 3)}
# endregion FUNC__themed_grid


# region FUNC__river
## @purpose The delta's river on the symmetry axis (mirror_lr: a vertical band — its own image) with fords, and the
## rows of the fords. The centre row is always a ford (the centre CP sits on it).
def _river(H: int, W: int, wmin: int, rng: random.Random, rr_cp: int) -> tuple[np.ndarray, list[int]]:
    band = np.zeros((H, W), dtype=bool)
    mc = (W - 1) // 2
    half_w = 3                                               # 7 tiles wide: > 3, never a "fence"
    band[1:-1, mc - half_w: mc + half_w + 1] = True
    mr = (H - 1) // 2
    centre_half = max(wmin // 2 + 2, rr_cp + 2)              # the centre ford carries the centre CP
    side_half = wmin // 2 + 2                                # side fords: wmin + 5 wide
    seg = 2 * wmin                                           # river segments between fords at least this long
    fords = [(mr, centre_half)]
    edge = mr - centre_half                                  # rows above the centre ford (and below: symmetric)
    n_side = max(1, (edge - 4) // (2 * side_half + 1 + seg) - 0)
    n_side = min(n_side, 3)
    step = (edge - 2) / (n_side + 1)
    for k in range(1, n_side + 1):
        off = centre_half + int(step * k + rng.uniform(-0.15, 0.15) * step)
        if mr - off - side_half < 2:
            continue
        fords += [(mr - off, side_half), (mr + off, side_half)]   # symmetric about the centre row too (rot-safe)
    for fr, hh in fords:
        band[max(1, fr - hh): fr + hh + 1, :] = False
    return band, sorted(f for f, _ in fords)
# endregion FUNC__river


def _poisson_seeds(rng: random.Random, n: int, H: int, W: int, sym: str, rows: tuple[int, int], cols: tuple[int, int],
                   gap: float, fixed: list[tuple[int, int]], avoid: np.ndarray | None = None, tries: int = 4000) -> list[tuple[int, int]]:
    """n seeds in the given box; every point and its image keep `gap` from all others (and from `fixed`)."""
    pts = [p for f in fixed for p in (f, tuple(int(v) for v in _image(f[0], f[1], H, W, sym)))]
    out: list[tuple[int, int]] = []
    for _ in range(tries):
        if len(out) >= n:
            break
        r, c = rng.randint(*rows), rng.randint(*cols)
        if avoid is not None and avoid[r, c]:
            continue
        ir, ic = (int(v) for v in _image(r, c, H, W, sym))
        cand = [(r, c), (ir, ic)]
        if math.hypot(ir - r, ic - c) < gap:
            continue
        if all(math.hypot(a[0] - b[0], a[1] - b[1]) >= gap for a in cand for b in pts):
            out.append((r, c))
            pts += cand
    return out


# region FUNC_themed_small
## @purpose One small themed map (7 CPs) for team_size per side (5, 10, 20), or raise after max_attempts seeds.
## @complexity 8
def _attempt_small(theme: str, team_size: int, seed: int) -> tuple[MapDef11 | None, dict]:
    from rush.maps_big import chokepoint_stats
    rng = random.Random(seed)
    tier = min(SMALL_TILES_PER_AGENT, key=lambda k: abs(k - team_size))
    wmin = SMALL_WMIN[tier]
    total = SMALL_TILES_PER_AGENT[tier] * 2 * team_size / (1.0 - WALL_SHARE[theme] - 0.02)
    aspect = rng.uniform(1.45, 1.8)
    H = int(math.sqrt(total / aspect)) | 1
    W = int(total / H) | 1
    sym = "mirror_lr" if theme == "delta" else rng.choice(("mirror_lr", "rot180"))
    spawn_w = 10
    x0 = spawn_w + 3
    cp_r = SMALL_CP_RADIUS[tier]
    rr = int(math.ceil(cp_r / TILE))
    mr, mc = (H - 1) // 2, (W - 1) // 2
    river = None
    if theme == "delta":
        river, _ = _river(H, W, wmin, rng, rr)
    seeds = [(mr, mc)]
    avoid = None
    if river is not None:
        from scipy import ndimage
        avoid = ndimage.distance_transform_edt(~river) < rr + wmin + 2
    seeds += _poisson_seeds(rng, 3, H, W, sym, (rr + 3, H - rr - 4), (x0 + rr + 2, mc - rr - 3), 2 * rr + 3, [(mr, mc)], avoid)
    if len(seeds) < 4:
        return None, {"reason": "cp seeds"}
    keep = np.zeros((H, W), dtype=bool)
    yy, xx = np.ogrid[:H, :W]
    for r, c in seeds:
        for pr, pc in ((r, c), _image(r, c, H, W, sym)):
            keep |= np.hypot(xx - pc, yy - pr) <= rr + 1
    keep[:, :spawn_w + 1] = True
    s = {5: 0.8, 10: 1.0}.get(tier, 1.2)
    g, info = _themed_grid(theme, rng, H, W, sym, keep, wmin, s, x0, river)
    for r, c in seeds:
        ir, ic = _image(r, c, H, W, sym)
        _circle(g, r, c, 2.0, ".")
        _circle(g, int(ir), int(ic), 2.0, ".")
    zone_a = ("blue", 2, H - 3, 2, spawn_w - 1)
    zone_b = _zone_image(zone_a, H, W, sym, "red")
    _fill_pockets(g, _safe_tiles(g, zone_a) + _safe_tiles(g, zone_b))
    try:
        cps = _cps(seeds, H, W, sym)
    except ValueError as e:
        return None, {"reason": str(e)}
    name = f"{theme.capitalize()}{team_size}_{seed % 100000:05d}"
    m = MapDef11(name=name, cols=W, rows=H, tile_size=TILE, grid=g, cp_positions=cps, spawn_zone_a=zone_a,
                 spawn_zone_b=zone_b, cp_radius=cp_r, compact=False, symmetry=sym,
                 tactic=f"{theme}, {sym}, wmin {wmin}", kind="procedural", seed=seed)
    v = validate_map(m)
    problems = [p for p in v["problems"] if not p.startswith("size ")]
    if problems:
        return None, {"reason": "validate", "problems": problems}
    wall = np.array([[ch == "#" for ch in row] for row in g])
    spawn = _safe_tiles(g, zone_a)
    zones = [_zone_tiles(m, k) for k in range(len(cps))]
    ck = chokepoint_stats(wall, spawn, zones, cross_rows=[mr, int(H * 0.3)], wmin=wmin)
    info.update({"W": W, "H": H, "sym": sym, "free_per_agent": round(float((~wall).sum()) / (2 * team_size), 1),
                 "bottleneck_w_min": ck["bottleneck_w_min"], "cps_bottleneck_lt_wmin": ck["cps_bottleneck_lt_wmin"],
                 "fence_walls": ck["fence_walls"], "wmin": wmin})
    if ck["cps_bottleneck_lt_wmin"] or ck["fence_walls"]:
        return None, {"reason": "chokepoints", **info}
    return m, info


def themed_small(theme: str, team_size: int, seed: int, max_attempts: int = 30) -> MapDef11:
    last: dict = {}
    for attempt in range(max_attempts):
        derived = seed if attempt == 0 else (seed * 1_000_003 + attempt * 7919) % (1 << 31)
        m, info = _attempt_small(theme, team_size, derived)
        if m is not None:
            logger.info(f"[IMP:7][themed_small][RESULT] {m.name}: {info['W']}x{info['H']} {info['sym']} obstacles={info['placed']} "
                        f"wall_share={info['wall_share_half']} free/agent={info['free_per_agent']} bottleneck_min={info['bottleneck_w_min']} "
                        f"(wmin {info['wmin']}) fences={info['fence_walls']} attempt={attempt} [VALUE]")
            return m
        last = info
    raise RuntimeError(f"themed_small({theme}, {team_size}, {seed}): no valid map in {max_attempts} attempts; last {last}")
# endregion FUNC_themed_small


# region FUNC_themed_big
## @purpose One big themed BigMap for team_size per side (50, 150, 500): CPs ~ 1 per 9 agents, multi-box spawns,
## sectors by thirds, validated by validate_big + chokepoint_stats. Cached on disk (generation of a 500 map ~ 1 min).
## @complexity 9
def _attempt_big(theme: str, team_size: int, seed: int):
    from scipy import ndimage
    from rush.maps_big import _finish, map_chokepoints, validate_big
    rng = random.Random(seed)
    wmin = _wmin_big(team_size)
    total = BIG_TILES_PER_AGENT * 2 * team_size / (1.0 - WALL_SHARE[theme] - 0.02)
    aspect = rng.uniform(1.5, 1.7)
    H = int(math.sqrt(total / aspect)) | 1
    W = int(total / H) | 1
    sym = "mirror_lr" if theme == "delta" else rng.choice(("mirror_lr", "rot180"))
    cp_r = BIG_CP_RADIUS.get(team_size, 240.0)
    rr = int(math.ceil(cp_r / TILE))
    mr, mc = (H - 1) // 2, (W - 1) // 2
    depth = max(8, W // 40)                                  # spawn boxes along the left edge
    n_z = max(2, team_size // 25)
    zh = max(8, (H - 8) // n_z - 4)
    zones = []
    for i in range(n_z):
        rc = int(H * (i + 0.5) / n_z)
        zones.append((max(3, rc - zh // 2), min(H - 4, rc + zh // 2), 3, 3 + depth))
    x0 = 3 + depth + wmin + 2
    river = None
    avoid = None
    if theme == "delta":
        river, _ = _river(H, W, wmin, rng, rr)
        avoid = ndimage.distance_transform_edt(~river) < rr + wmin + 2
    n_pairs = max(2, int(round((2 * team_size / BIG_AGENTS_PER_CP - 1) / 2)))
    seeds = _poisson_seeds(rng, n_pairs, H, W, sym, (rr + 4, H - rr - 5), (x0 + rr, mc - rr - 3), 2 * rr + 4,
                           [(mr, mc)], avoid, tries=20000)
    if len(seeds) < n_pairs:
        return None, {"reason": f"cp seeds {len(seeds)}/{n_pairs}"}
    keep = np.zeros((H, W), dtype=bool)
    yy, xx = np.ogrid[:H, :W]
    for r, c in seeds + [(mr, mc)]:
        for pr, pc in ((r, c), _image(r, c, H, W, sym)):
            keep |= np.hypot(xx - pc, yy - pr) <= rr + 1
    for r0, r1, c0, c1 in zones:
        keep[r0:r1 + 1, c0:c1 + 1] = True
    s = {50: 1.2, 150: 1.6}.get(team_size, 2.0)
    g, info = _themed_grid(theme, rng, H, W, sym, keep, wmin, s, x0, river)
    thirds = [(0, W // 3 - 1), (W // 3, 2 * W // 3 - 1), (2 * W // 3, W - 1)]
    ru = THEME_RU[theme]
    names = [("Запад", "З"), ("Центр", "Ц"), ("Восток", "В")]
    sectors = [{"name": f"{ru}: {n}", "tag": t, "rect": [0, H - 1, a, b]} for (n, t), (a, b) in zip(names, thirds)]
    concept = (f"{ru} ({theme}), {team_size} per side: procedural {theme} obstacles with >= {wmin}-tile gaps, "
               f"{sym}, {1 + 2 * len(seeds)} points.")
    try:
        m = _finish(f"{theme.capitalize()}{team_size}_{seed % 100000:05d}", g, sym, [(mr, mc)], seeds, zones, cp_r,
                    team_size, sectors, concept)
    except ValueError as e:
        return None, {"reason": str(e)}
    v = validate_big(m)
    if v["problems"]:
        return None, {"reason": "validate", "problems": v["problems"]}
    ck = map_chokepoints(m)
    info.update({"W": W, "H": H, "sym": sym, "cps": len(m.cp_positions), "free_per_agent": v["free_tiles_per_agent"],
                 "agents_per_cp": v["agents_per_cp"], "bottleneck_w_min": ck["bottleneck_w_min"],
                 "cps_bottleneck_lt_wmin": ck["cps_bottleneck_lt_wmin"], "fence_walls": ck["fence_walls"], "wmin": wmin,
                 "balance_err_px": v["balance_err_px"], "est_match_min": v["est_match_minutes_at_60pct"]})
    if ck["cps_bottleneck_lt_wmin"] or ck["fence_walls"]:
        return None, {"reason": "chokepoints", **info}
    return m, info


def themed_big(theme: str, team_size: int, seed: int, max_attempts: int = 12, cache: bool = True):
    key = CACHE_DIR / f"{theme}_{team_size}_{seed}.pkl"
    if cache and key.exists():
        try:
            return pickle.loads(key.read_bytes())
        except Exception as exc:                              # a stale or partial cache file: rebuild
            logger.warning(f"[IMP:7][themed_big][WARN] cache {key} unreadable ({exc}); rebuilding [VALUE]")
    last: dict = {}
    t0 = time.perf_counter()
    for attempt in range(max_attempts):
        derived = seed if attempt == 0 else (seed * 1_000_003 + attempt * 7919) % (1 << 31)
        m, info = _attempt_big(theme, team_size, derived)
        if m is not None:
            logger.info(f"[IMP:8][themed_big][RESULT] {m.name}: {info['W']}x{info['H']} {info['sym']} cps={info['cps']} "
                        f"obstacles={info['placed']} wall_share={info['wall_share_half']} free/agent={info['free_per_agent']} "
                        f"agents/cp={info['agents_per_cp']} bottleneck_min={info['bottleneck_w_min']} (wmin {info['wmin']}) "
                        f"fences={info['fence_walls']} balance={info['balance_err_px']}px est_min={info['est_match_min']} "
                        f"attempt={attempt} {time.perf_counter() - t0:.1f}s [VALUE]")
            if cache:
                key.parent.mkdir(parents=True, exist_ok=True)
                tmp = key.with_suffix(".tmp")
                tmp.write_bytes(pickle.dumps(m))
                tmp.replace(key)
            return m
        last = info
    raise RuntimeError(f"themed_big({theme}, {team_size}, {seed}): no valid map in {max_attempts} attempts; last {last}")
# endregion FUNC_themed_big


# region FUNC_pools
## @purpose "mix14:<ts>[:K]" -> the batched pool for team size ts with K themed maps per theme (default 2).
## ts 5: every maps11 map (compact included) + 12 maps11-procedural + 4 maps13 wide + themed; ts 10/20: the maps13
## stage pool (big maps11 maps + wide) + themed. Registered in maps13.EXTRA_POOLS so World13 builds it.
def build_pool14(name: str, seed: int = 0) -> list:
    from rush import maps13
    parts = name.split(":")
    ts = int(parts[1])
    k = int(parts[2]) if len(parts) > 2 else 2
    rng = random.Random(seed ^ (0x14A7 + ts))
    if ts <= 5:
        pool = list(maps11.build_pool("maps11+proc:12", seed))
        pool += [maps13.wide_map(rng.randrange(1 << 30), 5) for _ in range(4)]
    else:
        pool = list(maps13.build_pool13(f"stage{ts}:12", seed))
    for th in THEMES:
        pool += [themed_small(th, ts, rng.randrange(1 << 30)) for _ in range(k)]
    assert all(len(m.cp_positions) == N_CPS for m in pool)
    logger.info(f"[IMP:9][build_pool14][BUILD] pool={name}: maps={len(pool)} compact={sum(bool(m.compact) for m in pool)} "
                f"themed={k * len(THEMES)} largest={max(m.cols for m in pool)}x{max(m.rows for m in pool)} [VALUE]")
    return pool


## @purpose Big maps of one scale: the maps_big maps of that team size + `k` themed maps per theme.
def big_pool(team_size: int, k: int = 1, seed: int = 0, themes: tuple[str, ...] = THEMES) -> list:
    from rush.maps_big import BIG_MAPS, get_map
    out = [get_map(n) for n in BIG_MAPS if get_map(n).team_size == team_size]
    rng = random.Random(seed ^ (0xB16 + team_size))
    for th in themes:
        for _ in range(k):
            out.append(themed_big(th, team_size, rng.randrange(1 << 30)))
    logger.info(f"[IMP:9][big_pool][BUILD] team={team_size}: maps={[m.name for m in out]} [VALUE]")
    return out


def register_pools() -> None:
    from rush import maps13
    maps13.EXTRA_POOLS["mix14:"] = build_pool14
# endregion FUNC_pools


# region FUNC_render_small
def render_small(m, path: Path, px: int = 6) -> None:
    from PIL import Image, ImageDraw
    from rush.maps_big import COL_BG, COL_BLUE, COL_CP, COL_RED, COL_WALL
    img = Image.new("RGB", (m.cols * px, m.rows * px), COL_BG)
    d = ImageDraw.Draw(img)
    for z, col in ((m.spawn_zone_a, COL_BLUE), (m.spawn_zone_b, COL_RED)):
        _, a, b, c, e = z
        d.rectangle([c * px, a * px, (e + 1) * px - 1, (b + 1) * px - 1], fill=tuple(int(v * 0.3) for v in col))
    for r, row in enumerate(m.grid):
        for c, ch in enumerate(row):
            if ch == "#":
                d.rectangle([c * px, r * px, (c + 1) * px - 1, (r + 1) * px - 1], fill=COL_WALL)
    rad = m.cp_radius / TILE * px
    for r, c in m.cp_positions:
        x, y = (c + 0.5) * px, (r + 0.5) * px
        d.ellipse([x - rad, y - rad, x + rad, y + rad], outline=COL_CP, width=2)
    img.save(path)
# endregion FUNC_render_small


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="exp14 themed maps: generate, validate, render previews")
    ap.add_argument("--small", default="5,10,20")
    ap.add_argument("--big", default="50,150")
    ap.add_argument("--seed", type=int, default=14)
    ap.add_argument("--out", default=str(PREVIEW_DIR))
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", handlers=[logging.StreamHandler(sys.stdout)], force=True)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    from rush.maps_big import render_full
    for ts in [int(x) for x in args.small.split(",") if x]:
        for th in THEMES:
            m = themed_small(th, ts, args.seed * 131 + ts)
            render_small(m, out / f"small{ts}_{th}.png")
    for ts in [int(x) for x in args.big.split(",") if x]:
        for th in THEMES:
            m = themed_big(th, ts, args.seed * 977 + ts)
            render_full(m, out / f"big{ts}_{th}.png")
    logger.info(f"[IMP:9][main][RESULT] previews in {out}: {sorted(p.name for p in out.glob('*.png'))} [VALUE]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
