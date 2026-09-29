from __future__ import annotations

# region MODULE_CONTRACT [DOMAIN(8): Spectating; CONCEPT(9): BigBattleWorld; TECH(9): torch+scipy]
## @modulecontract
## @purpose One large match (team_size per side, up to 500 v 500) with world11's physics and rules and
## world11's exact observation layout, so frozen experiment-11 policies play it unchanged.
## @scope A single world, no auto-reset, no training signals beyond what world11.step already computes.
## @input A BigMap (maps_big) or any MapDef-like object + optional rule fields; actions (1, 2T, 4)
## @output world11.StepOut per decision; snapshot / map_geometry / events for replay11
## @links USES_API(9): torch, scipy.sparse.csgraph; LINKS_TO: world11 (physics source), maps_big, record_big,
## docs/BIGBATTLE_CONTRACT.md
## @invariants
## - OBS_LAYOUT / OBS_SIZE are world11's objects, asserted at import
## - Physics (move, turn, dash, shoot, bullets, hits, shields, respawn, control points, score, rewards,
##   step bookkeeping) is world11's source code, executed in a per-battle clone of the module whose rule
##   constants (TEAM_SIZE, N_AGENTS, BULLET_SLOTS, N_CP_SLOTS, TEAM_LIVES, SCORE_TO_WIN, MAX_FRAMES)
##   are rebound to this battle. Layout constants (9 entity, 8 bullet, 7 CP, 5 last-seen slots) are not.
## - Observation normalisations use this battle's team_size / team_lives / score_to_win / max_frames,
##   and a capped map diagonal (DIAG_CAP_PX), so the values stay in the range the policies were trained on
## - With team_size = 5 on a world11 map the observation equals World11's (tests/test_world_big.py)
## @rationale
## Q: Why exec a clone of world11's source instead of importing its class?
## A: World11 reads team size, bullet capacity and the rule constants from module globals, so an import
## A: can only ever be 5 v 5. Editing world11 is out of scope (frozen for training); copying its physics
## A: would fork the rules. A clone executes the SAME source with the battle's constants — the rules
## A: cannot drift, and every physics method runs verbatim.
## Q: What is overridden, and why is it not physics?
## A: Map tables (one BigMap, multi-rectangle spawns), flow fields (scipy Dijkstra per CP: the min-plus
## A: sweep is ~700 iterations x 8 shifts over 100 CPs x 250k tiles), visibility (line of sight only for
## A: the 9 nearest allies and 32 nearest enemies in range — 3.2M samples instead of 96M), hit resolution
## A: (world11's own method on the compacted alive bullets), the observation builder (7 NEAREST CPs, 5
## A: nearest last-seen enemies, top-8 hostile bullets over alive bullets), snapshot (bulk transfers).
## Q: Where can the culled visibility differ from world11?
## A: Only when more than KE_CAP enemies are nearer than a visible enemy that would still rank among the
## A: 9 nearest present entities. With team_size <= KE_CAP (every test at 5 v 5) it is exact.
## @changes
## LAST_CHANGE: [v0.1.0] Initial big-battle world.
## @modulemap
## FUNC 9[Clone world11 with a battle's rule constants] => _world11_clone
## CLASS 9[Overrides on top of the clone's World11] => _BigMixin
## FUNC 9[Build a WorldBig for a map] => WorldBig
## FUNC 6[MapDef-like object -> rules] => battle_rules
## @usecases
## - w = WorldBig(get_map("Front50"), device="mps", seed=1); out = w.step(actions)
def _module_contract():
    pass
# endregion MODULE_CONTRACT
# GREP_SUMMARY: big battle, world_big, 500v500, clone world11, culled visibility, nearest CPs, dijkstra flow field, snapshot
# STRUCTURE: ▶ BigMap → ⚡ clone world11(constants) → ⚡ class(Mixin, clone.World11) → ⚡ tables + dijkstra flow → ○ step (world11 source) → ⎋ StepOut / snapshot

import importlib.util
import itertools
import logging
import math
import sys
import time
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import torch

from rush import world11 as W11

logger = logging.getLogger(__name__)

# region BLOCK_CONSTANTS
OBS_LAYOUT: list[dict] = W11.OBS_LAYOUT
OBS_SIZE: int = W11.OBS_SIZE
N_ENTITY_SLOTS: int = W11.N_ENTITY_SLOTS        # 9
N_BULLET_SLOTS: int = W11.N_BULLET_SLOTS        # 8
N_CP_SLOTS: int = W11.N_CP_SLOTS                # 7
N_LASTSEEN_SLOTS: int = W11.N_LASTSEEN_SLOTS    # 5
assert (N_ENTITY_SLOTS, N_BULLET_SLOTS, N_CP_SLOTS, N_LASTSEEN_SLOTS) == (9, 8, 7, 5), "world11 layout drifted"
assert OBS_SIZE == 813, f"world11 OBS_SIZE drifted: {OBS_SIZE}"

KE_CAP: int = 32                     # nearest enemies per agent that get a line-of-sight test
# Largest world11 map diagonal is ~4590 px (a 123x91 procedural map); every CP / last-seen offset and
# path length the policies saw was divided by <= this. A 500x500-tile map would divide by 21 000 px and
# push those inputs to ~0. The cap keeps them in the trained range (clamped as in world11).
DIAG_CAP_PX: float = 4600.0
BULLETS_PER_AGENT: int = W11.BULLET_SLOTS // W11.N_AGENTS     # 16: world11's ring capacity per agent
# Flow-field method switch (CPs x rows x cols): below it world11's own sweep (exact tie-breaks), above it
# Dijkstra per CP (a 500x500 map with ~100 CPs is 25M cells x ~700 sweep iterations x 8 shifts).
SWEEP_MAX_CELLS: int = 3_000_000
_WORLD11_PATH: Path = Path(W11.__file__)
_clone_counter = itertools.count()
# endregion BLOCK_CONSTANTS


# region FUNC_battle_rules
## @purpose Rule fields of a map: BigMap attributes, world11 defaults for a plain MapDef.
def battle_rules(m: Any) -> dict:
    zones_a = list(getattr(m, "spawn_zones_a", []) or [m.spawn_zone_a])
    zones_b = list(getattr(m, "spawn_zones_b", []) or [m.spawn_zone_b])
    return {
        "team_size": int(getattr(m, "team_size", W11.TEAM_SIZE)),
        "score_to_win": float(getattr(m, "score_to_win", W11.SCORE_TO_WIN)),
        "team_lives": int(getattr(m, "team_lives", W11.TEAM_LIVES)),
        "max_frames": int(getattr(m, "max_frames", W11.MAX_FRAMES)),
        "spawn_zones": [zones_a, zones_b],
    }
# endregion FUNC_battle_rules


# region FUNC__world11_clone
## @purpose Execute world11's source as a fresh module and rebind its rule constants to one battle.
## @io (team_size, n_cps, rules) -> module
def _world11_clone(team_size: int, n_cps: int, rules: dict) -> ModuleType:
    name = f"rush._world11_big_{next(_clone_counter)}"
    spec = importlib.util.spec_from_file_location(name, _WORLD11_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod          # @dataclass resolves annotations through sys.modules[cls.__module__]
    spec.loader.exec_module(mod)
    # Rule constants only. Layout constants (OBS_LAYOUT, _SEG, N_*_SLOTS) keep the values computed at
    # exec time, which are world11's — the observation shape cannot change.
    mod.TEAM_SIZE = team_size
    mod.N_AGENTS = 2 * team_size
    mod.BULLET_SLOTS = BULLETS_PER_AGENT * 2 * team_size
    mod.N_CP_SLOTS = n_cps
    mod.TEAM_LIVES = rules["team_lives"]
    mod.SCORE_TO_WIN = rules["score_to_win"]
    mod.MAX_FRAMES = rules["max_frames"]
    assert mod.OBS_SIZE == OBS_SIZE and mod.N_ENTITY_SLOTS == N_ENTITY_SLOTS
    return mod
# endregion FUNC__world11_clone


# region CLASS__BigMixin
## @purpose Everything a single large battle needs on top of the clone's World11.
## @complexity 9
class _BigMixin:
    _wm: ModuleType          # the clone module (set on the class by WorldBig)

    def __init__(self, big_map: Any, device: str = "cpu", seed: int = 0, record: bool = True,
                 reward_preset: str = W11.DEFAULT_REWARD_PRESET) -> None:
        self.big = big_map
        self.rules = battle_rules(big_map)
        self.T = self.rules["team_size"]
        self.A = 2 * self.T
        self._t_build = time.perf_counter()
        super().__init__(1, device=device, seed=seed, record=record, map_pool="big", reward_preset=reward_preset)
        self.auto_reset = False          # one match to its end
        wm = self._wm
        logger.info(f"[IMP:9][WorldBig.__init__][INIT] map={big_map.name}, team_size={self.T}, agents={self.A}, cps={self.n_cps}, "
                    f"tiles={self.big.cols}x{self.big.rows}, bullet_slots={wm.BULLET_SLOTS}, score_to_win={wm.SCORE_TO_WIN}, "
                    f"team_lives={wm.TEAM_LIVES}, max_frames={wm.MAX_FRAMES}, norm_diag={self.norm_diag:.0f}px "
                    f"(true {float(self.map_diag[0]):.0f}), build={time.perf_counter() - self._t_build:.1f}s [VALUE]")

    # region FUNC__build_map_tables
    ## @purpose world11's per-map tables for the one BigMap (n_maps = 1), multi-rectangle spawn zones.
    def _build_map_tables(self) -> None:
        m = self.big
        ts = float(m.tile_size)
        rows, cols = m.rows, m.cols
        grid = torch.from_numpy(np.array([[ch == "#" for ch in row] for row in m.grid], dtype=bool))
        self._pool = [m]
        self.n_maps = 1
        self.tile_size = ts
        c_n = len(m.cp_positions)
        self.n_cps = c_n
        cp_xy = torch.tensor([[c * ts + ts / 2, r * ts + ts / 2] for r, c in m.cp_positions], dtype=torch.float32)
        cp_rc = torch.tensor([[r, c] for r, c in m.cp_positions], dtype=torch.long)
        cp_r = float(m.cp_radius)

        # Dead-CP relocation exactly as world11 does it (a CP whose circle holds no free tile centre).
        self.relocated_cps = []
        fr = (~grid).nonzero().float()
        fx, fy = fr[:, 1] * ts + ts / 2, fr[:, 0] * ts + ts / 2
        for k in range(c_n):
            d = ((fx - cp_xy[k, 0]) ** 2 + (fy - cp_xy[k, 1]) ** 2).sqrt()
            if bool((d <= cp_r).any()):
                continue
            j = int(d.argmin())
            old = (float(cp_xy[k, 0]), float(cp_xy[k, 1]))
            cp_xy[k, 0], cp_xy[k, 1] = fx[j], fy[j]
            cp_rc[k, 0], cp_rc[k, 1] = int(fr[j, 0]), int(fr[j, 1])
            self.relocated_cps.append((m.name, k, old, (float(fx[j]), float(fy[j]))))

        # CP zone per tile (nearest covering CP), chunked over CPs to bound memory on 500x500 maps.
        rr = torch.arange(rows).view(rows, 1).float() * ts + ts / 2
        cc = torch.arange(cols).view(1, cols).float() * ts + ts / 2
        best = torch.full((rows, cols), float("inf"))
        zone = torch.full((rows, cols), -1, dtype=torch.long)
        for k0 in range(0, c_n, 16):
            xy = cp_xy[k0: k0 + 16]
            d = ((cc.unsqueeze(0) - xy[:, 0].view(-1, 1, 1)) ** 2 + (rr.unsqueeze(0) - xy[:, 1].view(-1, 1, 1)) ** 2).sqrt()
            dmin, arg = d.min(0)
            better = dmin < best
            best = torch.where(better, dmin, best)
            zone = torch.where(better, arg + k0, zone)
        zone = torch.where(best <= cp_r, zone, torch.full_like(zone, -1))

        pad = W11.GRID_RADIUS
        grid_pad = torch.ones(1, rows + 2 * pad, cols + 2 * pad, dtype=torch.bool)
        grid_pad[0, pad: pad + rows, pad: pad + cols] = grid
        zone_pad = torch.full((1, rows + 2 * pad, cols + 2 * pad), -1, dtype=torch.long)
        zone_pad[0, pad: pad + rows, pad: pad + cols] = zone

        # Safe spawn tiles = WallGrid.safe_spawn_tiles (free, 8 free neighbours, not on the border), row-major.
        g = grid.numpy()
        safe = np.zeros_like(g)
        inner = ~g[1:-1, 1:-1]
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                inner &= ~g[1 + dr: rows - 1 + dr, 1 + dc: cols - 1 + dc]
        safe[1:-1, 1:-1] = inner
        safe_rc = np.argwhere(safe)                                                # row-major (r, c)
        spawns, counts = [], []
        for team in (0, 1):
            sel = np.zeros(len(safe_rc), dtype=bool)
            for _, r0, r1, c0, c1 in self.rules["spawn_zones"][team]:
                sel |= (safe_rc[:, 0] >= r0) & (safe_rc[:, 0] <= r1) & (safe_rc[:, 1] >= c0) & (safe_rc[:, 1] <= c1)
            tiles = safe_rc[sel]
            if len(tiles) == 0:
                half = len(safe_rc) // 2
                tiles = safe_rc[:half] if team == 0 else safe_rc[half:]
            spawns.append(torch.tensor(np.stack([tiles[:, 1] * ts + ts / 2, tiles[:, 0] * ts + ts / 2], 1), dtype=torch.float32))
            counts.append(len(tiles))
        smax = max(counts)
        spawn_tbl = torch.zeros(1, 2, smax, 2)
        for team in (0, 1):
            spawn_tbl[0, team, : counts[team]] = spawns[team]

        dev = self.device
        self.map_grid = grid.unsqueeze(0).to(dev)
        self.map_grid_pad = grid_pad.to(dev)
        self.map_zone_pad = zone_pad.to(dev)
        self.map_arena = torch.tensor([[cols * ts, rows * ts]], dtype=torch.float32, device=dev)
        self.map_cp_xy = cp_xy.unsqueeze(0).to(dev)
        self.map_cp_rc = cp_rc.unsqueeze(0)
        self.map_cp_r = torch.tensor([cp_r], dtype=torch.float32, device=dev)
        self.map_compact = torch.tensor([bool(getattr(m, "compact", False))], device=dev)
        self.map_spawn = spawn_tbl.to(dev)
        self.map_spawn_n = torch.tensor([counts], dtype=torch.long, device=dev)
        self._map_spawn_n_cpu = self.map_spawn_n.cpu()
        self.map_diag = self.map_arena.pow(2).sum(1).sqrt()
        self.norm_diag = min(float(self.map_diag[0]), DIAG_CAP_PX)
        self.map_desc = torch.zeros(1, W11.N_MAPS_STATE, device=dev)
        self._compact_ids = self._large_ids = torch.zeros(1, dtype=torch.long, device=dev)
        logger.info(f"[IMP:9][WorldBig._build_map_tables][BUILD] map={m.name}, tiles={cols}x{rows}, free={int((~grid).sum())}, "
                    f"cps={c_n}, relocated_cps={len(self.relocated_cps)}, spawn_tiles={counts}, zones_a={len(self.rules['spawn_zones'][0])}, "
                    f"zones_b={len(self.rules['spawn_zones'][1])} [VALUE]")
    # endregion FUNC__build_map_tables

    # region FUNC__build_flow_fields
    ## @purpose Walkable distance into every CP zone and the first-step direction, like world11 but by
    ## multi-source Dijkstra on the 8-connected tile graph (no corner cutting, same costs).
    ## @rationale world11's min-plus sweep does (path length in tiles) x 8 full-tensor shifts over all CPs;
    ## on a 500x500 map with ~100 CPs that is ~700 x 8 passes over 25M cells. Dijkstra from each CP zone
    ## is ~0.1-0.5 s per CP on the CPU. The first-step direction reuses world11's exact rule.
    def _build_flow_fields(self) -> None:
        grid = self.map_grid[0].cpu().numpy()
        rows, cols = grid.shape
        cells = self.n_cps * rows * cols
        if cells <= SWEEP_MAX_CELLS:
            # world11's own min-plus sweep (inherited from the clone): bit-identical distances and
            # first-step tie-breaks. Dijkstra finds the same shortest paths, but equal-length first steps
            # round differently in float64 and flipped the direction on ~17% of tiles of a world11 map.
            t0 = time.perf_counter()
            super()._build_flow_fields()
            logger.info(f"[IMP:9][WorldBig._build_flow_fields][BUILD] method=world11_sweep, cells={cells}, {time.perf_counter() - t0:.1f}s [VALUE]")
            return
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import dijkstra

        t0 = time.perf_counter()
        ts = self.tile_size
        free = ~grid
        node = np.full(grid.shape, -1, dtype=np.int64)
        node[free] = np.arange(int(free.sum()))
        nbrs = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]

        def shifted(a: np.ndarray, dr: int, dc: int, fill) -> np.ndarray:
            out = np.full_like(a, fill)
            rs, re = max(0, -dr), rows - max(0, dr)
            cs, ce = max(0, -dc), cols - max(0, dc)
            out[rs:re, cs:ce] = a[rs + dr: re + dr, cs + dc: ce + dc]
            return out

        allowed, src, dst, wts = [], [], [], []
        for dr, dc in nbrs:
            ok = shifted(free, dr, dc, False)
            if dr != 0 and dc != 0:
                ok = ok & shifted(free, dr, 0, False) & shifted(free, 0, dc, False)
            ok = ok & free
            allowed.append(ok)
            r_idx, c_idx = np.nonzero(ok)
            src.append(node[r_idx, c_idx])
            dst.append(node[r_idx + dr, c_idx + dc])
            wts.append(np.full(len(r_idx), ts * (math.sqrt(2.0) if dr and dc else 1.0)))
        n_nodes = int(free.sum())
        graph = coo_matrix((np.concatenate(wts), (np.concatenate(src), np.concatenate(dst))), shape=(n_nodes, n_nodes)).tocsr()

        cp_xy = self.map_cp_xy[0].cpu().numpy()
        cp_r = float(self.map_cp_r[0])
        rr = (np.arange(rows) * ts + ts / 2).reshape(rows, 1)
        cc = (np.arange(cols) * ts + ts / 2).reshape(1, cols)
        k_n = self.n_cps
        dist = np.full((k_n, rows, cols), np.inf, dtype=np.float32)
        in_zone_all = np.zeros((k_n, rows, cols), dtype=bool)
        empty = 0
        for k in range(k_n):
            zone = (np.sqrt((cc - cp_xy[k, 0]) ** 2 + (rr - cp_xy[k, 1]) ** 2) <= cp_r) & free
            in_zone_all[k] = zone
            sources = node[zone]
            if len(sources) == 0:
                empty += 1
                continue
            d = dijkstra(graph, directed=True, indices=sources, min_only=True)
            dist[k][free] = d.astype(np.float32)
        t_dij = time.perf_counter() - t0

        costs = [ts * (math.sqrt(2.0) if dr and dc else 1.0) for dr, dc in nbrs]
        dvec = np.array([[dc, dr] for dr, dc in nbrs], dtype=np.float32)
        dvec = dvec / np.linalg.norm(dvec, axis=1, keepdims=True)
        flow = np.zeros((k_n, rows, cols, 2), dtype=np.float32)
        for k in range(k_n):
            cand = np.stack([np.where(ok, shifted(dist[k], dr, dc, np.inf) + np.float32(cost), np.inf)
                             for (dr, dc), ok, cost in zip(nbrs, allowed, costs)], 0).astype(np.float32)
            best_i = cand.argmin(0)
            best_v = np.take_along_axis(cand, best_i[None], 0)[0]
            stop = (dist[k] == 0) | np.isinf(best_v) | np.isinf(dist[k])
            f = dvec[best_i]
            f[stop] = 0.0
            flow[k] = f
        dev = self.device
        self.flow_zone = torch.from_numpy(in_zone_all).unsqueeze(0).to(dev)
        self.flow_dist = torch.from_numpy(dist).unsqueeze(0).to(dev)              # (1, K, R, C)
        self.flow_dir = torch.from_numpy(flow).unsqueeze(0).to(dev)               # (1, K, R, C, 2)
        reach = np.isfinite(dist) & free[None]
        logger.info(f"[IMP:9][WorldBig._build_flow_fields][BUILD] method=dijkstra, cps={k_n}, nodes={n_nodes}, edges={graph.nnz}, empty_zones={empty}, "
                    f"reachable_free_tiles={int(reach.sum())}/{n_nodes * k_n}, dijkstra={t_dij:.1f}s, total={time.perf_counter() - t0:.1f}s, "
                    f"flow_mb={(dist.nbytes + flow.nbytes) / 2**20:.0f} [VALUE]")
    # endregion FUNC__build_flow_fields

    # region FUNC_reset_worlds
    ## @purpose Start the match: world11's field resets, spawn tiles drawn from the team's zones.
    def reset_worlds(self, mask: torch.Tensor) -> None:
        self._vis_cache = None
        T = self.T
        sels = [(torch.rand(1, T, generator=self.gen) * self._map_spawn_n_cpu[0, team].float()).long().clamp(min=0)
                for team in (0, 1)]
        for team in (0, 1):
            self.pos[0, team * T: (team + 1) * T] = self.map_spawn[0, team, sels[team][0].to(self.device)]
        for name, val in (("vel", 0.0), ("angle", 0.0), ("hp", float(W11.PLAYER_HP)), ("shield", float(W11.SHIELD_MAX)),
                          ("since_dmg", 0), ("alive", True), ("waiting", False), ("cooldown", 0), ("respawn_timer", 0),
                          ("dash_left", 0), ("dash_cd", 0), ("b_alive", False), ("b_age", 0), ("cp_owner", 0),
                          ("cp_cap_team", 0), ("cp_progress", 0.0), ("score", 0.0), ("frame", 0), ("mem_valid", False),
                          ("mem_age", 0), ("mem_pos", 0.0), ("mem_vel", 0.0), ("mem_hp", 0.0)):
            getattr(self, name)[0] = val
        self.lives[0] = self._wm.TEAM_LIVES
        self.b_id = torch.full_like(self.b_owner, -1, dtype=torch.long)
        self._next_bullet_id = 0
    # endregion FUNC_reset_worlds

    # region FUNC__shoot
    ## @purpose world11's shooting; the new bullets get stable ids for the replay.
    def _shoot(self, shoot_a: torch.Tensor) -> None:
        before = self.b_alive.clone()
        super()._shoot(shoot_a)
        new = self.b_alive & ~before
        k = int(new.sum())
        if k:
            self.b_id[new] = torch.arange(self._next_bullet_id, self._next_bullet_id + k, device=self.device)
            self._next_bullet_id += k
    # endregion FUNC__shoot

    # region FUNC__resolve_hits
    ## @purpose world11's hit resolution on the compacted alive bullets (order preserved, so every argmax
    ## over bullets and agents picks the same element): (alive x agents) instead of (16*A x A).
    def _resolve_hits(self, dmg_dealt, dmg_taken, kills, deaths) -> None:
        ai = self.b_alive[0].nonzero().flatten()
        if ai.numel() == 0:
            return
        full_pos, full_owner, full_alive = self.b_pos, self.b_owner, self.b_alive
        self.b_pos, self.b_owner, self.b_alive = full_pos[:, ai], full_owner[:, ai], full_alive[:, ai]
        try:
            super()._resolve_hits(dmg_dealt, dmg_taken, kills, deaths)
            sub_alive = self.b_alive
        finally:
            self.b_pos, self.b_owner = full_pos, full_owner
            self.b_alive = full_alive
        self.b_alive = full_alive.clone()
        self.b_alive[:, ai] = sub_alive
    # endregion FUNC__resolve_hits

    # region FUNC__visibility
    ## @purpose Line of sight (world11 geometry, 96 samples) for the 9 nearest alive allies and the KE_CAP
    ## nearest alive enemies in vision range of every agent; every other pair reads as not visible.
    ## @io idx (ignored, one world) -> (1, A, A) bool
    def _visibility(self, idx: torch.Tensor | None) -> torch.Tensor:
        pos, alive = self.pos[0], self.alive[0]
        a_n, dev, T = self.A, self.device, self.T
        rel = pos.unsqueeze(0) - pos.unsqueeze(1)                                  # [i, j] = pos_j - pos_i
        dist = rel.pow(2).sum(-1).sqrt()
        team = self._agent_team
        enemy = team.view(-1, 1) != team.view(1, -1)
        eye = torch.eye(a_n, dtype=torch.bool, device=dev)
        inf = torch.full_like(dist, float("inf"))
        ke, ka = min(T, KE_CAP), min(T - 1, N_ENTITY_SLOTS)
        e_key = torch.where(enemy & alive.view(1, -1) & (dist <= W11.VISION_RANGE), dist, inf)
        e_d, e_i = e_key.topk(ke, dim=1, largest=False)
        cands, ok = [e_i], [torch.isfinite(e_d)]
        if ka > 0:
            a_key = torch.where(~enemy & alive.view(1, -1) & ~eye, dist, inf)
            a_d, a_i = a_key.topk(ka, dim=1, largest=False)
            cands.append(a_i)
            ok.append(torch.isfinite(a_d))
        cand, cand_ok = torch.cat(cands, 1), torch.cat(ok, 1)                      # (A, K)
        a = pos.view(a_n, 1, 2)
        b = pos[cand]                                                             # (A, K, 2)
        t = torch.linspace(0.0, 1.0, W11.LOS_SAMPLES, device=dev).view(1, 1, W11.LOS_SAMPLES, 1)
        pts = a.unsqueeze(2) + (b - a).unsqueeze(2) * t
        m = torch.zeros(pts.shape[:-1], dtype=torch.long, device=dev)
        blocked = self._wall_at(m, pts[..., 0], pts[..., 1]).any(-1)
        d_c = (b - a).pow(2).sum(-1).sqrt()
        vis_c = cand_ok & ~blocked & (d_c <= W11.VISION_RANGE)
        # OR-combine: when fewer candidates are valid than asked for, topk pads with inf entries whose index
        # can repeat an index of the OTHER list; a plain scatter_ let such a padded False overwrite a real True
        # (caught by the 5v5 equivalence test: red team lost sight of an enemy at 48 px).
        vis = torch.zeros(a_n, a_n, dtype=torch.uint8, device=dev)
        vis.scatter_reduce_(1, cand, vis_c.to(torch.uint8), reduce="amax")
        return vis.bool().unsqueeze(0)
    # endregion FUNC__visibility

    def global_state(self) -> torch.Tensor:
        # No critic in a show match; world11's state layout is sized for 10 agents and 7 CPs.
        return torch.zeros(1, 2, 1, device=self.device)

    # region FUNC__build_obs
    ## @purpose world11's observation, for one large world. Differences from world11._build_obs:
    ## rule normalisations from this battle, capped diagonal, 7 nearest CPs (index order) when a map has
    ## more, 5 nearest eligible last-seen enemies (index order) when a team has more, top-8 hostile
    ## bullets over the alive bullets. With team_size 5 on a 7-CP world11 map every choice below
    ## degenerates to world11's.
    ## @io (idx ignored, vis (1, A, A)) -> (1, A, OBS_SIZE)
    ## @complexity 9
    def _build_obs(self, idx: torch.Tensor | None, vis: torch.Tensor) -> torch.Tensor:
        wm, T, a = self._wm, self.T, self.A
        pos, vel, angle = self.pos, self.vel, self.angle
        hp, shield, alive = self.hp, self.shield, self.alive
        maps = self.map_idx
        n, dev = 1, self.device
        seg = {s["name"]: s for s in OBS_LAYOUT}
        obs = torch.zeros(n, a, OBS_SIZE, device=dev)
        arena = self.map_arena[maps].unsqueeze(1)
        diag = torch.full((n, 1, 1), self.norm_diag, device=dev)
        team = self._agent_team.view(1, a).expand(n, a)
        is_blue = team == 0
        facing = torch.stack([angle.cos(), angle.sin()], dim=-1)
        pm1 = lambda cond: torch.where(cond, 1.0, -1.0)                          # noqa: E731
        stw, lives_max, max_frames = wm.SCORE_TO_WIN, wm.TEAM_LIVES, wm.MAX_FRAMES

        # ---- core ----
        core = torch.zeros(n, a, W11.CORE_SIZE, device=dev)
        core[..., 0] = pos[..., 0] / arena[..., 0] * 2 - 1
        core[..., 1] = pos[..., 1] / arena[..., 1] * 2 - 1
        core[..., 2] = torch.where(alive, hp / W11.PLAYER_HP * 2 - 1, torch.full_like(hp, -1.0))
        core[..., 3] = torch.where(alive, shield / W11.SHIELD_MAX * 2 - 1, torch.full_like(shield, -1.0))
        core[..., 4] = pm1(alive)
        core[..., 5] = self.respawn_timer.float() / W11.RESPAWN_FRAMES * 2 - 1
        core[..., 6] = self.cooldown.float() / W11.SHOOT_COOLDOWN_FRAMES * 2 - 1
        core[..., 7:9] = facing
        core[..., 9:11] = (vel / W11.VEL_NORM).clamp(-1.0, 1.0)
        core[..., 11] = self.dash_cd.float() / W11.DASH_COOLDOWN_FRAMES * 2 - 1
        core[..., 12] = self.dash_left.float() / W11.DASH_FRAMES * 2 - 1
        core[..., 13:21] = self._cast_rays(pos, maps) * 2 - 1
        score = self.score
        my_score = torch.where(is_blue, score[:, :1], score[:, 1:])
        en_score = torch.where(is_blue, score[:, 1:], score[:, :1])
        core[..., 21] = (my_score / stw).clamp(max=1.0) * 2 - 1
        core[..., 22] = (en_score / stw).clamp(max=1.0) * 2 - 1
        alive_b = alive[:, :T].sum(1, keepdim=True).float()
        alive_r = alive[:, T:].sum(1, keepdim=True).float()
        core[..., 23] = torch.where(is_blue, alive_b, alive_r) / T * 2 - 1
        core[..., 24] = torch.where(is_blue, alive_r, alive_b) / T * 2 - 1
        lives = self.lives.float()
        core[..., 25] = torch.where(is_blue, lives[:, :1], lives[:, 1:]) / lives_max * 2 - 1
        core[..., 26] = torch.where(is_blue, lives[:, 1:], lives[:, :1]) / lives_max * 2 - 1
        core[..., 27] = (self.frame.float() / max_frames).clamp(max=1.0).view(n, 1) * 2 - 1
        core[..., 28] = pm1(self.map_compact[maps]).view(n, 1).expand(n, a)
        obs[..., :W11.CORE_SIZE] = core

        # ---- entity slots (world11 verbatim over the full A x A matrices) ----
        rel = pos.unsqueeze(1) - pos.unsqueeze(2)
        dist = rel.pow(2).sum(-1).sqrt()
        same_team = team.unsqueeze(2) == team.unsqueeze(1)
        is_self = torch.eye(a, dtype=torch.bool, device=dev).view(1, a, a)
        other_alive = alive.unsqueeze(1).expand(n, a, a)
        present = other_alive & ~is_self & (same_team | vis)
        k_ent = min(N_ENTITY_SLOTS, a - 1)
        order = torch.where(present, dist, torch.full_like(dist, float("inf"))).argsort(dim=2)[:, :, :k_ent]
        keep = present.gather(2, order).float().unsqueeze(-1)

        def g(t: torch.Tensor) -> torch.Tensor:
            src = t.unsqueeze(1).expand(n, a, *t.shape[1:])
            ix = order if t.dim() == 2 else order.unsqueeze(-1).expand(n, a, k_ent, t.shape[2])
            return src.gather(2, ix)

        g_rel = rel.gather(2, order.unsqueeze(-1).expand(-1, -1, -1, 2))
        g_dist = dist.gather(2, order)
        g_vel, g_face = g(vel), g(facing)
        g_hp, g_shield = g(hp), g(shield)
        g_same = same_team.gather(2, order)
        g_vis = vis.gather(2, order)
        g_dash = g(self.dash_left) > 0
        aim_err = torch.atan2(g_rel[..., 1], g_rel[..., 0]) - angle.unsqueeze(2)
        aim_err = (aim_err + math.pi) % (2 * math.pi) - math.pi
        their_ang = torch.atan2(-g_rel[..., 1], -g_rel[..., 0])
        their_err = their_ang - torch.atan2(g_face[..., 1], g_face[..., 0])
        their_err = (their_err + math.pi) % (2 * math.pi) - math.pi
        ones = torch.ones_like(g_dist)
        ent = torch.stack([
            ones,
            torch.where(g_same, torch.full_like(g_dist, W11.ETYPE_ALLY), torch.full_like(g_dist, W11.ETYPE_ENEMY)),
            (g_rel[..., 0] / W11.ENTITY_RANGE).clamp(-1, 1),
            (g_rel[..., 1] / W11.ENTITY_RANGE).clamp(-1, 1),
            (g_dist / W11.ENTITY_RANGE).clamp(max=1.0) * 2 - 1,
            (g_vel[..., 0] / W11.VEL_NORM).clamp(-1, 1),
            (g_vel[..., 1] / W11.VEL_NORM).clamp(-1, 1),
            g_face[..., 0], g_face[..., 1],
            g_hp / W11.PLAYER_HP * 2 - 1,
            g_shield / W11.SHIELD_MAX * 2 - 1,
            aim_err / math.pi,
            their_err / math.pi,
            torch.where(g_vis, ones, -ones),
            torch.where(g_dash, ones, -ones),
        ], dim=-1) * keep
        s = seg["entities"]
        obs[..., s["start"]: s["start"] + k_ent * W11.ENTITY_SLOT_SIZE] = ent.reshape(n, a, -1)

        # ---- bullet slots: nearest hostile among the alive bullets ----
        ab = self.b_alive[0].nonzero().flatten()
        if ab.numel():
            b_pos, b_vel = self.b_pos[:, ab], self.b_vel[:, ab]
            b_rel = b_pos.unsqueeze(1) - pos.unsqueeze(2)                         # (1, A, nb, 2)
            b_dist = b_rel.pow(2).sum(-1).sqrt()
            b_team = self._agent_team[self.b_owner[:, ab].long()].unsqueeze(1)
            hostile = b_team != team.unsqueeze(2)
            kb = min(N_BULLET_SLOTS, int(ab.numel()))
            _, b_order = torch.where(hostile, b_dist, torch.full_like(b_dist, float("inf"))).topk(kb, dim=2, largest=False)
            b_keep = hostile.gather(2, b_order).float().unsqueeze(-1)
            idx2 = b_order.unsqueeze(-1).expand(-1, -1, -1, 2)
            g_brel = b_rel.gather(2, idx2)
            g_bvel = b_vel.unsqueeze(1).expand(n, a, b_vel.shape[1], 2).gather(2, idx2)
            vv = g_bvel.pow(2).sum(-1).clamp(min=1e-6)
            t_star = (-(g_brel * g_bvel).sum(-1) / vv).clamp(min=0.0)
            closest = (g_brel + g_bvel * t_star.unsqueeze(-1)).pow(2).sum(-1).sqrt()
            bones = torch.ones_like(t_star)
            bul = torch.stack([
                bones,
                (g_brel[..., 0] / W11.BULLET_OBS_RANGE).clamp(-1, 1),
                (g_brel[..., 1] / W11.BULLET_OBS_RANGE).clamp(-1, 1),
                g_bvel[..., 0] / W11.BULLET_SPEED,
                g_bvel[..., 1] / W11.BULLET_SPEED,
                (closest / W11.BULLET_MISS_NORM).clamp(max=1.0) * 2 - 1,
                (t_star / W11.BULLET_TTC_NORM).clamp(max=1.0) * 2 - 1,
            ], dim=-1) * b_keep
            s = seg["bullets"]
            obs[..., s["start"]: s["start"] + kb * W11.BULLET_SLOT_SIZE] = bul.reshape(n, a, -1)

        # ---- control point slots: the 7 nearest, in CP index order ----
        c_all = self.n_cps
        cp_xy_all = self.map_cp_xy[maps]                                          # (1, C, 2)
        radius = self.map_cp_r[maps].view(n, 1, 1)
        c_rel_all = cp_xy_all.unsqueeze(1) - pos.unsqueeze(2)                     # (1, A, C, 2)
        c_dist_all = c_rel_all.pow(2).sum(-1).sqrt()
        kc = min(N_CP_SLOTS, c_all)
        if c_all <= N_CP_SLOTS:
            sel = torch.arange(c_all, device=dev).view(1, 1, -1).expand(n, a, -1)
        else:
            _, near_k = c_dist_all.topk(kc, dim=2, largest=False)
            sel = near_k.sort(dim=2).values
        c_rel = c_rel_all.gather(2, sel.unsqueeze(-1).expand(-1, -1, -1, 2))
        c_dist = c_dist_all.gather(2, sel)
        my_tid = (team + 1).unsqueeze(2)
        owner = self.cp_owner.unsqueeze(1).expand(n, a, c_all).gather(2, sel)
        cap = self.cp_cap_team.unsqueeze(1).expand(n, a, c_all).gather(2, sel)
        prog = self.cp_progress.unsqueeze(1).expand(n, a, c_all).gather(2, sel)
        owner_val = torch.where(owner == my_tid, 1.0, torch.where(owner == 0, 0.0, -1.0))
        cap_val = torch.where(cap == my_tid, 1.0, torch.where(cap == 0, 0.0, -1.0))
        near = (c_dist_all.transpose(1, 2) <= radius) & alive.unsqueeze(1)       # (1, C, A)
        blue_near = near[:, :, :T].any(2).unsqueeze(1).expand(n, a, c_all).gather(2, sel)
        red_near = near[:, :, T:].any(2).unsqueeze(1).expand(n, a, c_all).gather(2, sel)
        mine_near = torch.where(is_blue.unsqueeze(2), blue_near, red_near)
        foe_near = torch.where(is_blue.unsqueeze(2), red_near, blue_near)
        ts = self.tile_size
        row = (pos[..., 1] / ts).long().clamp(0, self.map_grid.shape[1] - 1)
        col = (pos[..., 0] / ts).long().clamp(0, self.map_grid.shape[2] - 1)
        mm = maps.view(n, 1, 1).expand(n, a, kc)
        rr, cc = row.unsqueeze(2).expand(-1, -1, kc), col.unsqueeze(2).expand(-1, -1, kc)
        p_dist = self.flow_dist[mm, sel, rr, cc]
        p_dir = self.flow_dir[mm, sel, rr, cc]
        path_norm = 1.5 * diag
        p_obs = torch.where(torch.isfinite(p_dist), (p_dist / path_norm).clamp(max=1.0) * 2 - 1, torch.ones_like(p_dist))
        cones = torch.ones_like(c_dist)
        cps = torch.stack([
            cones,
            owner_val,
            c_rel[..., 0] / diag,
            c_rel[..., 1] / diag,
            (c_dist / diag).clamp(max=1.0) * 2 - 1,
            p_obs,
            p_dir[..., 0], p_dir[..., 1],
            prog * 2 - 1,
            cap_val,
            torch.where(foe_near, 1.0, -1.0),
            torch.where(mine_near, 1.0, -1.0),
            torch.where(c_dist <= radius, 1.0, -1.0),
            (c_dist / radius).clamp(max=2.0) - 1.0,
        ], dim=-1)
        s = seg["cps"]
        obs[..., s["start"]: s["start"] + kc * W11.CP_SLOT_SIZE] = cps.reshape(n, a, -1)

        # ---- last-seen enemy memory (team-shared); only enemies I do not see now ----
        # mem tensors are (1, 2, T, ...): [team t, enemy k of team 1-t]
        mem_of = lambda t: torch.cat([t[:, 0:1].expand(n, T, *t.shape[2:]), t[:, 1:2].expand(n, T, *t.shape[2:])], dim=1)  # noqa: E731
        m_pos, m_vel = mem_of(self.mem_pos), mem_of(self.mem_vel)                 # (1, A, T, 2)
        m_hp, m_age, m_val = mem_of(self.mem_hp), mem_of(self.mem_age), mem_of(self.mem_valid)
        enemy_idx = torch.cat([torch.arange(T, 2 * T), torch.arange(0, T)]).to(dev).view(2, T)[self._agent_team]   # (A, T)
        i_see = vis.gather(2, enemy_idx.view(1, a, T).expand(n, -1, -1))
        eligible = m_val & ~i_see
        l_rel_all = m_pos - pos.unsqueeze(2)
        l_dist_all = l_rel_all.pow(2).sum(-1).sqrt()
        if T <= N_LASTSEEN_SLOTS:
            lsel = torch.arange(T, device=dev).view(1, 1, -1).expand(n, a, -1)
        else:
            key = torch.where(eligible, l_dist_all, torch.full_like(l_dist_all, float("inf")))
            _, lk = key.topk(N_LASTSEEN_SLOTS, dim=2, largest=False)
            lsel = lk.sort(dim=2).values
        kl = lsel.shape[2]
        gl = lambda t: t.gather(2, lsel) if t.dim() == 3 else t.gather(2, lsel.unsqueeze(-1).expand(-1, -1, -1, t.shape[3]))  # noqa: E731
        show = gl(eligible).float().unsqueeze(-1)
        l_rel, l_dist = gl(l_rel_all), gl(l_dist_all)
        lv, lh, la = gl(m_vel), gl(m_hp), gl(m_age)
        lones = torch.ones_like(l_dist)
        ls = torch.stack([
            lones,
            l_rel[..., 0] / diag.view(n, 1, 1),
            l_rel[..., 1] / diag.view(n, 1, 1),
            (l_dist / diag.view(n, 1, 1)).clamp(max=1.0) * 2 - 1,
            la.float() / W11.LASTSEEN_HORIZON_FRAMES * 2 - 1,
            torch.where(la == 0, lones, -lones),
            (lv[..., 0] / W11.VEL_NORM).clamp(-1, 1),
            (lv[..., 1] / W11.VEL_NORM).clamp(-1, 1),
            lh / W11.PLAYER_HP * 2 - 1,
        ], dim=-1) * show
        s = seg["lastseen"]
        obs[..., s["start"]: s["start"] + kl * W11.LASTSEEN_SLOT_SIZE] = ls.reshape(n, a, -1)

        # ---- egocentric grid (world11 verbatim) ----
        side, rad = W11.GRID_SIDE, W11.GRID_RADIUS
        gr = (row + rad).view(n, a, 1, 1) + self._grid_dr.view(1, 1, side, side)
        gc = (col + rad).view(n, a, 1, 1) + self._grid_dc.view(1, 1, side, side)
        gm = maps.view(n, 1, 1, 1).expand(n, a, side, side)
        walls = self.map_grid_pad[gm, gr, gc].float()
        zone = self.map_zone_pad[gm, gr, gc]
        z_owner = self.cp_owner.gather(1, zone.clamp(min=0).view(n, -1)).view(n, a, side, side)
        my_t = (team + 1).view(n, a, 1, 1)
        z_val = torch.where(z_owner == my_t, 1.0, torch.where(z_owner == 0, 0.5, -1.0))
        z_val = torch.where(zone >= 0, z_val, torch.zeros_like(z_val))
        s = seg["grid"]
        obs[..., s["start"]:] = torch.stack([walls, z_val], dim=2).reshape(n, a, -1)
        return obs.clamp(-1.0, 1.0)
    # endregion FUNC__build_obs

    # region FUNC_snapshot
    ## @purpose Render state for replay11 from bulk transfers; bullets carry their stable world id.
    ## columnar=True (default): agents / bullets / cps as dicts of numpy arrays — replay11.BigReplayWriter's
    ## fast path for 1000 agents. columnar=False: World11.snapshot's lists of dicts.
    def snapshot(self, world_idx: int = 0, columnar: bool = True) -> dict:
        T = self.T
        if columnar:
            np_ = lambda t: t[0].detach().cpu().numpy()                             # noqa: E731
            pos, vel = np_(self.pos), np_(self.vel)
            waiting, rt = np_(self.waiting), np_(self.respawn_timer)
            live = self.b_alive[0].nonzero().flatten()
            bp, bv = self.b_pos[0, live].cpu().numpy(), self.b_vel[0, live].cpu().numpy()
            bo = self.b_owner[0, live].cpu().numpy().astype(np.int64)
            agents = {"x": pos[:, 0], "y": pos[:, 1], "angle": np_(self.angle), "vx": vel[:, 0], "vy": vel[:, 1],
                      "hp": np_(self.hp), "shield": np_(self.shield), "alive": np_(self.alive), "waiting": waiting,
                      "respawn_in": np.where(waiting, rt, 0), "dashing": np_(self.dash_left) > 0, "dash_cd": np_(self.dash_cd),
                      "team": (np.arange(self.A) >= T).astype(np.int64)}
            bullets = {"id": self.b_id[0, live].cpu().numpy(), "x": bp[:, 0], "y": bp[:, 1], "vx": bv[:, 0], "vy": bv[:, 1],
                       "team": (bo >= T).astype(np.int64), "owner": bo}
            cxy = np_(self.map_cp_xy)
            cps = {"x": cxy[:, 0], "y": cxy[:, 1], "r": np.full(self.n_cps, float(self.map_cp_r[0])),
                   "owner": np_(self.cp_owner).astype(np.int64), "cap_team": np_(self.cp_cap_team).astype(np.int64),
                   "progress": np_(self.cp_progress)}
            return {"frame": int(self.frame[0]), "map_idx": 0, "score": self.score[0].cpu().tolist(),
                    "lives": [int(v) for v in self.lives[0].cpu().tolist()], "agents": agents, "bullets": bullets, "cps": cps}
        pos, vel = self.pos[0].cpu().tolist(), self.vel[0].cpu().tolist()
        ang, hp, sh = self.angle[0].cpu().tolist(), self.hp[0].cpu().tolist(), self.shield[0].cpu().tolist()
        alive, waiting = self.alive[0].cpu().tolist(), self.waiting[0].cpu().tolist()
        rt, dl, dcd = self.respawn_timer[0].cpu().tolist(), self.dash_left[0].cpu().tolist(), self.dash_cd[0].cpu().tolist()
        agents = [{
            "id": i, "team": 0 if i < T else 1, "x": pos[i][0], "y": pos[i][1], "angle": ang[i],
            "vx": vel[i][0], "vy": vel[i][1], "hp": hp[i], "shield": sh[i], "alive": bool(alive[i]),
            "respawn_in": int(rt[i]) if waiting[i] else 0, "dashing": dl[i] > 0, "dash_cd": int(dcd[i]),
        } for i in range(self.A)]
        live = self.b_alive[0].nonzero().flatten()
        bp, bv = self.b_pos[0, live].cpu().tolist(), self.b_vel[0, live].cpu().tolist()
        bo, bid = self.b_owner[0, live].cpu().tolist(), self.b_id[0, live].cpu().tolist()
        bullets = [{"id": int(bid[k]), "x": bp[k][0], "y": bp[k][1], "vx": bv[k][0], "vy": bv[k][1],
                    "team": 0 if bo[k] < T else 1, "owner": int(bo[k])} for k in range(len(bo))]
        cxy, r = self.map_cp_xy[0].cpu().tolist(), float(self.map_cp_r[0])
        own, cap, prog = self.cp_owner[0].cpu().tolist(), self.cp_cap_team[0].cpu().tolist(), self.cp_progress[0].cpu().tolist()
        cps = [{"x": cxy[k][0], "y": cxy[k][1], "r": r, "owner": int(own[k]), "progress": prog[k], "cap_team": int(cap[k])}
               for k in range(self.n_cps)]
        return {"frame": int(self.frame[0]), "map_idx": 0, "score": self.score[0].cpu().tolist(),
                "lives": [int(v) for v in self.lives[0].cpu().tolist()], "agents": agents, "bullets": bullets, "cps": cps}
    # endregion FUNC_snapshot

    ## @purpose Static geometry for the replay header. sectors: maps_big gives tile rects {"name", "rect":
    ## [r0, r1, c0, c1]}; the viewer wants world px {"name", x0, y0, x1, y1} — both are provided.
    def map_geometry(self, map_idx: int = 0) -> dict:
        m = self.big
        ts = float(m.tile_size)
        walls = [[int(c), int(r)] for r, c in self.map_grid[0].nonzero().cpu().tolist()]
        sectors = []
        for s in list(getattr(m, "sectors", []) or []):
            r0, r1, c0, c1 = s["rect"]
            sectors.append({"name": s["name"], "rect": [r0, r1, c0, c1],
                            "x0": c0 * ts, "y0": r0 * ts, "x1": (c1 + 1) * ts, "y1": (r1 + 1) * ts})
        return {"name": m.name, "arena_w": int(m.cols * m.tile_size), "arena_h": int(m.rows * m.tile_size),
                "tile_size": int(m.tile_size), "cols": int(m.cols), "rows": int(m.rows), "walls": walls,
                "compact": bool(getattr(m, "compact", False)), "cp_radius": float(m.cp_radius),
                "sectors": sectors, "cp_names": list(getattr(m, "cp_names", []) or []),
                "concept": getattr(m, "concept", "")}
# endregion CLASS__BigMixin


# region FUNC_WorldBig
## @purpose Build the world for one map: a world11 clone with this battle's rules, plus the mixin.
## @io (map, device, seed, record) -> world instance (a subclass of the clone's World11)
def WorldBig(big_map: Any, device: str = "cpu", seed: int = 0, record: bool = True,
             reward_preset: str = W11.DEFAULT_REWARD_PRESET):    # noqa: N802 — used like a class
    rules = battle_rules(big_map)
    mod = _world11_clone(rules["team_size"], len(big_map.cp_positions), rules)
    cls = type("WorldBig", (_BigMixin, mod.World11), {"_wm": mod})
    return cls(big_map, device=device, seed=seed, record=record, reward_preset=reward_preset)
# endregion FUNC_WorldBig
