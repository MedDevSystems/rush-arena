from __future__ import annotations

# region MODULE_CONTRACT [DOMAIN(8): Maps; CONCEPT(9): StageMapPools; TECH(7): numpy, scipy]
## @modulecontract
## @purpose Map pools for the experiment-13 team-size curriculum: world11's pools pass through; "stageN"
## pools give maps sized for N per side — procedural WIDE maps (solid cover blobs, a minimum passage width
## enforced while generating) plus the maps11 maps that are big enough.
## @scope Pool names, the wide-map generator, its validation (maps11.validate_map + maps_big.chokepoint_stats).
## @input pool name ("maps11", "maps11+proc:N", "stageN", "stageN:K"), seed
## @output list[MapDef11] with exactly 7 CPs each (world layout), symmetric, reachable, balanced
## @links USES_API(8): numpy, scipy.ndimage; LINKS_TO: maps11 (helpers, validation), maps_big (chokepoint_stats),
## world13 (routes world11's build_pool here), docs/EXP13_CONTRACT.md, docs/BIG_NOTES_redesign.md
## @invariants
## - every wide map: validate_map without problems (the 121x141 size limit excepted), no CP whose best route
##   needs a passage narrower than WMIN tiles, no thin "fence" walls
## - deterministic for (team_size, seed)
## @rationale
## Q: Why not scale maps11's procedural generator?
## A: Its "rooms" and "lanes" styles are thin walls with 2-tile doors — the "fighting through gaps in a fence"
## A: the redesign removed from the big maps. Clumps can only form and move where passages are wide; the
## A: generator therefore places solid blobs only where the gap to every other wall stays >= WMIN.
## @changes
## LAST_CHANGE: [v0.1.1] EXTRA_POOLS hook (empty by default): exp14 registers its "mix14:" pools here.
## PREV_CHANGE: [v0.1.0] Initial: stage pools, wide generator.
## @modulemap
## FUNC 8[one wide map] => wide_map
## FUNC 7[pool by name] => build_pool13
# endregion MODULE_CONTRACT
# GREP_SUMMARY: maps13, stage pool, wide map, team size, chokepoint, blob cover, curriculum
# STRUCTURE: ▶ size from team → ○ blobs with clearance → ⚡ symmetrize → ⚡ CP seeds → ◇ validate + chokepoints → ⎋ MapDef11

import logging
import math
import random

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
    _rect,
    _safe_tiles,
    _symmetrize,
    _zone_image,
    _zone_tiles,
    validate_map,
)

logger = logging.getLogger(__name__)

# region BLOCK_CONSTANTS
TILES_PER_AGENT: dict[int, float] = {5: 450.0, 10: 380.0, 20: 300.0, 50: 250.0}   # free tiles per agent
WMIN: dict[int, int] = {5: 8, 10: 8, 20: 8, 50: 10}                                # min passage width, tiles
CP_RADIUS: dict[int, float] = {5: 120.0, 10: 150.0, 20: 180.0, 50: 210.0}
FREE_SHARE: float = 0.87                                                             # expected free share
DEFAULT_PROC: int = 24
MAPS11_MIN_FREE_PER_AGENT: float = 200.0                                             # a maps11 map joins stageN if it gives this
EXTRA_POOLS: dict = {}          # name prefix -> builder(name, seed); registered by later experiments (maps14.register_pools)
# endregion BLOCK_CONSTANTS


def _tiers(ts: int) -> int:
    return min(TILES_PER_AGENT, key=lambda k: abs(k - ts))


def _free_tiles(m) -> int:
    return sum(ch != "#" for row in m.grid for ch in row)


# region FUNC_wide_map
## @purpose One wide procedural map for team_size per side, or None if this seed fails validation.
## @complexity 8
def _attempt_wide(seed: int, team_size: int) -> tuple[MapDef11 | None, dict]:
    from scipy import ndimage
    from rush.maps_big import chokepoint_stats
    rng = random.Random(seed)
    tier = _tiers(team_size)
    wmin = WMIN[tier]
    total = TILES_PER_AGENT[tier] * 2 * team_size / FREE_SHARE
    aspect = rng.uniform(1.45, 1.85)
    H = int(math.sqrt(total / aspect)) | 1
    W = int(total / H) | 1
    sym = rng.choice(("mirror_lr", "rot180"))
    spawn_w = 10
    x0 = spawn_w + 3
    half = W // 2
    g = _canvas(W, H)

    def walls() -> np.ndarray:
        return np.array([[ch == "#" for ch in row] for row in g])

    def image_mask(mask: np.ndarray) -> np.ndarray:
        out = np.zeros_like(mask)
        rr, cc = np.nonzero(mask)
        for r, c in zip(rr, cc):
            ir, ic = _image(r, c, H, W, sym)
            out[int(ir), int(ic)] = True
        return out

    target_wall_share = 1.0 - FREE_SHARE
    placed, tries = 0, 0
    while tries < 900:
        tries += 1
        wl = walls()
        if wl[1:-1, 1:-1].mean() >= target_wall_share:
            break
        shape = rng.random()
        r, c = rng.randint(3, H - 4), rng.randint(x0 + 2, half)
        cand = np.zeros((H, W), dtype=bool)
        if shape < 0.5:
            rad = rng.choice((2.0, 2.5, 3.0, 3.5, 4.5, 5.5))
            yy, xx = np.ogrid[:H, :W]
            cand = (np.hypot(xx - c, yy - r) <= rad)
        else:
            h, w = rng.randint(3, 8), rng.randint(3, 10)
            cand[max(1, r - h // 2): min(H - 1, r + h // 2 + 1), max(1, c - w // 2): min(W - 1, c + w // 2 + 1)] = True
        cand[:, :x0] = False
        cand[0, :] = cand[-1, :] = False
        cand[:, 0] = cand[:, -1] = False
        if not cand.any():
            continue
        img = image_mask(cand)
        both = cand | img
        # Gap to every existing wall (border included) >= wmin: distance from the new cover to the nearest wall.
        d_old = ndimage.distance_transform_edt(~wl)
        if d_old[both].min() < wmin + 1:
            continue
        # The blob and its own image: either one piece across the axis, or apart by >= wmin.
        if not (cand & img).any():
            d_self = ndimage.distance_transform_edt(~cand)
            if d_self[img].min() < wmin + 1:
                continue
        for rr, cc in zip(*np.nonzero(both)):
            g[rr][cc] = "#"
        placed += 1
    _symmetrize(g, sym)

    # CPs: centre (its own image) + 3 seeds in the left half; clear a small pad around each.
    cr, cc0 = (H - 1) // 2, (W - 1) // 2
    cp_r = CP_RADIUS[tier]
    seeds = [(cr, cc0)]
    min_gap = 2 * cp_r / TILE + 2
    t = 0
    while len(seeds) < 4 and t < 600:
        t += 1
        r, c = rng.randint(4, H - 5), rng.randint(x0 + 2, half - 3)
        ir, ic = _image(r, c, H, W, sym)
        pts = [p for s in seeds for p in (s, tuple(int(v) for v in _image(s[0], s[1], H, W, sym)))]
        cand_pts = [(r, c), (int(ir), int(ic))]
        if all(math.hypot(a[0] - b[0], a[1] - b[1]) >= min_gap for a in cand_pts for b in pts) and math.hypot(ir - r, ic - c) >= min_gap:
            seeds.append((r, c))
    if len(seeds) < 4:
        return None, {"reason": "cp seeds"}
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
    m = MapDef11(name=f"Wide{team_size}_{seed % 100000:05d}", cols=W, rows=H, tile_size=TILE, grid=g, cp_positions=cps,
                 spawn_zone_a=zone_a, spawn_zone_b=zone_b, cp_radius=cp_r, compact=False, symmetry=sym,
                 tactic=f"wide blobs, {sym}, {placed} blobs, wmin {wmin}", kind="procedural", seed=seed)
    v = validate_map(m)
    problems = [p for p in v["problems"] if not p.startswith("size ")]
    if problems:
        return None, {"reason": "validate", "problems": problems}
    wall = np.array([[ch == "#" for ch in row] for row in g])
    spawn = _safe_tiles(g, zone_a)
    zones = [_zone_tiles(m, k) for k in range(len(cps))]
    ck = chokepoint_stats(wall, spawn, zones, cross_rows=[H // 2, int(H * 0.3)], wmin=wmin)
    info = {"W": W, "H": H, "sym": sym, "blobs": placed, "free": int((~wall).sum()),
            "free_per_agent": round(float((~wall).sum()) / (2 * team_size), 1),
            "bottleneck_w_min": ck["bottleneck_w_min"], "cps_bottleneck_lt_wmin": ck["cps_bottleneck_lt_wmin"],
            "fence_walls": ck["fence_walls"], "cramped_lt6": ck["cramped_area_share_w_lt6"], "wmin": wmin,
            "spawn_sep": v.get("spawn_separation_tiles")}
    if ck["cps_bottleneck_lt_wmin"] or ck["fence_walls"]:
        return None, {"reason": "chokepoints", **info}
    return m, info


def wide_map(seed: int, team_size: int, max_attempts: int = 40) -> MapDef11:
    last: dict = {}
    for attempt in range(max_attempts):
        derived = seed if attempt == 0 else (seed * 1_000_003 + attempt * 7919) % (1 << 31)
        m, info = _attempt_wide(derived, team_size)
        if m is not None:
            logger.info(f"[IMP:7][wide_map][RESULT] {m.name}: {info['W']}x{info['H']} {info['sym']} blobs={info['blobs']} "
                        f"free/agent={info['free_per_agent']} bottleneck_min={info['bottleneck_w_min']} (wmin {info['wmin']}) "
                        f"fences={info['fence_walls']} cramped<6={info['cramped_lt6']} spawn_sep={info['spawn_sep']} "
                        f"attempt={attempt} [VALUE]")
            m.name = f"Wide{team_size}_{seed % 100000:05d}"
            return m
        last = info
    raise RuntimeError(f"wide_map: no valid map for seed {seed}, team {team_size}: last {last}")
# endregion FUNC_wide_map


# region FUNC_build_pool13
## @purpose Pool by name. world11 names pass through to maps11.build_pool. "stage5[:K]" = maps11 non-compact
## maps + K maps11-procedural non-compact; "stageN[:K]" (N >= 10) = K wide maps for N per side + maps11 maps
## with >= MAPS11_MIN_FREE_PER_AGENT free tiles per agent.
def build_pool13(name: str = "maps11", seed: int = 0) -> list:
    for prefix, builder in EXTRA_POOLS.items():           # later experiments' pools (exp14 "mix14:"); empty by default
        if name.startswith(prefix):
            return builder(name, seed)
    if not name.startswith("stage"):
        return maps11.build_pool(name, seed)
    head, _, k = name.partition(":")
    ts = int(head[len("stage"):])
    n_proc = int(k) if k else DEFAULT_PROC
    rng = random.Random(seed ^ (0x13A7 + ts))
    base = [m for m in maps11.build_pool("maps11", seed) if not m.compact]
    if ts <= 5:
        pool = base + [maps11.procedural_map(rng.randrange(1 << 30), compact=False) for _ in range(n_proc)]
    else:
        big = [m for m in base if _free_tiles(m) / (2 * ts) >= MAPS11_MIN_FREE_PER_AGENT]
        pool = big + [wide_map(rng.randrange(1 << 30), ts) for _ in range(n_proc)]
    fpa = [round(_free_tiles(m) / (2 * ts)) for m in pool]
    logger.info(f"[IMP:9][build_pool13][BUILD] pool={name}, team={ts}, maps={len(pool)} (maps11 {len(pool) - n_proc}, proc {n_proc}), "
                f"free tiles/agent min/median/max={min(fpa)}/{sorted(fpa)[len(fpa) // 2]}/{max(fpa)}, "
                f"largest={max(m.cols for m in pool)}x{max(m.rows for m in pool)}, names={[m.name for m in pool][:8]}... [VALUE]")
    assert all(len(m.cp_positions) == N_CPS for m in pool)
    return pool
# endregion FUNC_build_pool13
