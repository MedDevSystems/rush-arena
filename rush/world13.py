from __future__ import annotations

# region MODULE_CONTRACT [DOMAIN(9): Simulation; CONCEPT(10): VariableTeamBatchedWorld; TECH(9): torch, module cloning]
## @modulecontract
## @purpose Experiment 13: world12 (physics, masks, traits, rewards, teamwork) batched over many worlds with
## ANY team size (multiple of 5), same OBS_LAYOUT12 (1029) and the same 249-column team state, so model12
## weights run unchanged at 10v10, 20v20, 50v50.
## @scope World classes per team size, observation builder generalised to TS != 5, squad-pooled global state,
## uneven-numbers scenario hook, stage map pools (maps13) routed into world11's map tables.
## @input team_size, n_worlds, device, seed, Rules12-like rules, World12 kwargs (map_pool, compact_share, ...)
## @output a World12 subclass instance: step / reset_worlds / observe / action_mask / global_state as World12
## @links USES_API(9): torch; LINKS_TO: world11 (source cloned), world12 (source cloned against the clone),
## maps13 (pools), vec13, train13, docs/EXP13_CONTRACT.md, docs/EXP13_NOTES.md
## @invariants
## - world11.py and world12.py are NOT edited and NOT copied for physics: their SOURCE is executed as fresh
##   modules; world11's clone gets TEAM_SIZE / N_AGENTS / TEAM_LIVES / BULLET_SLOTS rebound after exec, world12's
##   source is executed with `import rush.world11` resolved to that clone
## - Layout constants keep their exec-time (5v5) values: OBS_SIZE12 = 1029, 9/8/7/5 token slots, STATE 249
## - team_size = 5: observations, rewards, masks, global state equal World12's bit for bit (tests/test_world13.py)
## @rationale
## Q: Why clone world12's source instead of subclassing world12.World12?
## A: World12 inherits from world11.World11 and both read TEAM_SIZE / N_AGENTS from module globals at call time;
## A: a subclass of the real classes can only be 5v5. Cloning runs the same code with the battle's constants
## A: (world_big did this for world11 and verified 5v5 bit-exactness) — the rules cannot drift.
## Q: What is overridden, and why is it not physics?
## A: _build_obs: world12 writes one last-seen token per enemy (TS tokens do not fit 5 slots) and divides local
## A: force counts by the team size; bullets are compacted to the alive ones before the top-8 (slots grow with
## A: the team). global_state: world12's is sized by the agent count; squads keep 249 columns. _reset_idx: the
## A: uneven-numbers hook. None of these touch movement, shooting, hits, respawn, capture or reward code.
## @changes
## LAST_CHANGE: [v0.1.0] Initial: clones, generalised obs, squad state, uneven hook.
## @modulemap
## FUNC 8[exec a module source with import overrides] => _exec_clone
## FUNC 9[world class for a team size, cached] => world13_class
## CLASS 10[overrides on the cloned World12] => _Mixin13
## FUNC 7[instance factory] => World13
# endregion MODULE_CONTRACT
# GREP_SUMMARY: world13, team size, big teams, clone, squads, global state, last seen, uneven, 10v10, 20v20
# STRUCTURE: ▶ world13_class(ts) → ⚡ clone world11 (rebind) → ⚡ clone world12 against it → ⚡ class(_Mixin13, clone.World12) → ⎋ cache

import builtins
import importlib.util
import itertools
import logging
import math
import sys
import types
from pathlib import Path
from typing import Any

import torch

from rush import world11 as _W11_REAL
from rush import world12 as _W12_REAL
from rush.world12 import (
    BULLET_SIZE12,
    CORE_SIZE12,
    ENTITY_SIZE12,
    FLIGHT_NORM_FRAMES,
    FORCE_FAR,
    FORCE_NEAR,
    LASTSEEN_SIZE12,
    LATERAL_NORM,
    N_TRAITS,
    NEAR_RANGE,
    OBS_LAYOUT12,
    OBS_SIZE12,
    REACH_NORM,
    STATE_SIZE12,
)

logger = logging.getLogger(__name__)

# region BLOCK_CONSTANTS
_SEG12 = {s["name"]: s for s in OBS_LAYOUT12}
N_ENTITY_SLOTS: int = int(_W11_REAL.N_ENTITY_SLOTS)       # 9
N_BULLET_SLOTS: int = int(_W11_REAL.N_BULLET_SLOTS)       # 8
N_CP_SLOTS: int = int(_W11_REAL.N_CP_SLOTS)               # 7
N_LASTSEEN_SLOTS: int = int(_W11_REAL.N_LASTSEEN_SLOTS)   # 5
BASE_TEAM: int = int(_W11_REAL.TEAM_SIZE)                 # 5
N_SQUADS: int = BASE_TEAM                                 # squads per team in the critic state
BULLETS_PER_AGENT: float = 25.6                           # world12: 256 slots for ten agents
LIVES_PER_AGENT: int = int(_W11_REAL.TEAM_LIVES) // BASE_TEAM   # 6 (world11: 30 for five)
FORCE_COUNT_NORM: float = 5.0                             # local force counts / this, clamped (absolute "how many")
STATE_AGENT: int = 12                                     # world12's per-agent block in the team state
_W11_PATH = Path(_W11_REAL.__file__)
_W12_PATH = Path(_W12_REAL.__file__)
_counter = itertools.count()
_CLASS_CACHE: dict[tuple, type] = {}
# endregion BLOCK_CONSTANTS


# region FUNC__exec_clone
## @purpose Execute a module's source as a fresh module whose `import` statements resolve the names in
## `overrides` (full dotted module names) to the given objects — at exec time AND inside its functions later
## (functions resolve builtins through their module globals, which carry the patched __builtins__).
def _exec_clone(path: Path, name: str, overrides: dict[str, Any]) -> types.ModuleType:
    real_import = builtins.__import__
    pkg_real = sys.modules["rush"]

    class _Pkg(types.ModuleType):
        def __getattr__(self, attr: str) -> Any:
            full = f"rush.{attr}"
            if full in overrides:
                return overrides[full]
            return getattr(pkg_real, attr)

    pkg = _Pkg("rush")

    def _import(nm, globals=None, locals=None, fromlist=(), level=0):  # noqa: A002
        if level == 0:
            if nm in overrides:
                return overrides[nm] if fromlist else pkg
            if nm == "rush" and fromlist and any(f"rush.{f}" in overrides for f in fromlist):
                return pkg
        return real_import(nm, globals, locals, fromlist, level)

    b = dict(builtins.__dict__)
    b["__import__"] = _import
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    mod.__dict__["__builtins__"] = b
    sys.modules[name] = mod          # @dataclass resolves annotations through sys.modules[cls.__module__]
    spec.loader.exec_module(mod)
    return mod
# endregion FUNC__exec_clone


# region CLASS__Mixin13
## @purpose Everything a team size != 5 needs on top of the cloned World12.
## @complexity 9
class _Mixin13:
    _W11: types.ModuleType            # world11 clone (rule constants of this team size)
    _W12: types.ModuleType            # world12 clone
    team_size: int

    def __init__(self, n_worlds: int, device: str = "cpu", seed: int = 0, rules: Any = None,
                 uneven_share: float = 0.0, uneven_max: int = 0, **kw: Any) -> None:
        ts = self.team_size
        if rules is None:
            rules = self._W12.Rules12(bullet_slots=int(round(BULLETS_PER_AGENT * 2 * ts)))
        self._uneven_share, self._uneven_max = 0.0, 0
        self._disabled: torch.Tensor | None = None
        super().__init__(n_worlds, device=device, seed=seed, rules=rules, **kw)   # type: ignore[call-arg]
        self._disabled = torch.zeros(n_worlds, 2 * ts, dtype=torch.bool, device=self.device)
        self.set_uneven(uneven_share, uneven_max)
        logger.info(f"[IMP:9][World13.__init__][INIT] team_size={ts} agents={2 * ts} worlds={n_worlds} "
                    f"bullet_slots={self.rules.bullet_slots} team_lives={self._W11.TEAM_LIVES} score_to_win={self._W11.SCORE_TO_WIN} "
                    f"max_frames={self._W11.MAX_FRAMES} map_pool={kw.get('map_pool', 'maps11')} maps={self.n_maps} "
                    f"obs={OBS_SIZE12} state={STATE_SIZE12} squads={N_SQUADS}x{ts // N_SQUADS} [VALUE]")

    @property
    def n_agents(self) -> int:
        return 2 * self.team_size

    ## @purpose world11's map tables; a stage pool without compact maps gets its large maps as the "compact"
    ## draw too (world11 draws from both lists on every reset and cannot draw from an empty one).
    def _build_map_tables(self) -> None:
        super()._build_map_tables()                                               # type: ignore[misc]
        if len(self._compact_ids_cpu) == 0:
            self._compact_ids, self._compact_ids_cpu = self._large_ids, self._large_ids_cpu
            logger.info(f"[IMP:8][World13._build_map_tables][CONFIG] pool has no compact maps: compact draw -> "
                        f"{len(self._large_ids_cpu)} large maps [VALUE]")

    # region FUNC_set_uneven
    ## @purpose Scenario hook: at each world reset, with probability `share`, one random team plays without
    ## 1..max_missing members for the whole match (dead, never respawn). 0 = off (World12 behaviour exactly).
    def set_uneven(self, share: float, max_missing: int) -> None:
        mx = max(0, min(int(max_missing), self.team_size - 1))
        self._uneven_share, self._uneven_max = (float(share), mx) if mx > 0 else (0.0, 0)
        logger.info(f"[IMP:8][World13.set_uneven][CONFIG] share={self._uneven_share} max_missing={self._uneven_max} [VALUE]")

    def _reset_idx(self, idx: torch.Tensor, k: int) -> None:
        super()._reset_idx(idx, k)                                               # type: ignore[misc]
        if self._disabled is None:
            return
        self._disabled[idx] = False
        if self._uneven_share <= 0 or k == 0:
            return
        ts = self.team_size
        r = torch.rand(k, 3, generator=self.gen)
        on = r[:, 0] < self._uneven_share
        if not bool(on.any()):
            return
        team = (r[:, 1] < 0.5).long()
        cnt = (r[:, 2] * self._uneven_max).long() + 1                           # 1..max
        slot = torch.arange(ts).view(1, ts)
        miss = on.view(-1, 1) & (slot >= (ts - cnt).view(-1, 1))                # the last cnt slots of that team
        full = torch.zeros(k, 2 * ts, dtype=torch.bool)
        full[:, :ts] = miss & (team == 0).view(-1, 1)
        full[:, ts:] = miss & (team == 1).view(-1, 1)
        full = full.to(self.device)
        self._disabled[idx] = full
        self.alive[idx] = self.alive[idx] & ~full
        self.waiting[idx] = self.waiting[idx] & ~full
        self.hp[idx] = torch.where(full, torch.zeros_like(self.hp[idx]), self.hp[idx])
        self.shield[idx] = torch.where(full, torch.zeros_like(self.shield[idx]), self.shield[idx])
    # endregion FUNC_set_uneven

    # region FUNC__build_obs
    ## @purpose world12's observation builder for any team size. Differences from world12 (each an identity at
    ## TS = 5, see the module rationale): force counts / FORCE_COUNT_NORM clamped; bullets compacted to the
    ## alive ones and top-8 by topk; last-seen = the 5 nearest remembered-and-not-seen enemies, in slot order.
    def _build_obs(self, idx: torch.Tensor | None, vis: torch.Tensor) -> torch.Tensor:
        W11 = self._W11
        S = lambda t: self._sel(t, idx)                                          # noqa: E731
        TS = self.team_size
        pos, vel, angle = S(self.pos), S(self.vel), S(self.angle)
        hp, shield, alive = S(self.hp), S(self.shield), S(self.alive)
        maps = S(self.map_idx)
        max_hp, max_sh = S(self._max_hp), S(self._max_sh)
        ltr = S(self._log_traits)
        spd_a, range_a = S(self._b_speed_a), S(self._range_a)
        n, a, dev = pos.shape[0], 2 * TS, self.device
        obs = torch.zeros(n, a, OBS_SIZE12, device=dev)
        arena = self.map_arena[maps].unsqueeze(1)
        diag = self.map_diag[maps].view(n, 1, 1)
        team = self._agent_team.view(1, a).expand(n, a)
        is_blue = (team == 0)
        facing = torch.stack([angle.cos(), angle.sin()], dim=-1)
        pm1 = lambda cond: torch.where(cond, 1.0, -1.0)                          # noqa: E731
        vision = self.rules.vision

        # ---- core ----
        core = torch.zeros(n, a, CORE_SIZE12, device=dev)
        core[..., 0] = pos[..., 0] / arena[..., 0] * 2 - 1
        core[..., 1] = pos[..., 1] / arena[..., 1] * 2 - 1
        core[..., 2] = torch.where(alive, hp / max_hp * 2 - 1, torch.full_like(hp, -1.0))
        core[..., 3] = torch.where(alive, shield / max_sh * 2 - 1, torch.full_like(shield, -1.0))
        core[..., 4] = pm1(alive)
        core[..., 5] = S(self.respawn_timer).float() / W11.RESPAWN_FRAMES * 2 - 1
        core[..., 6] = S(self.cooldown).float() / S(self._cd_frames).float() * 2 - 1
        core[..., 7:9] = facing
        core[..., 9:11] = (vel / W11.VEL_NORM).clamp(-1.0, 1.0)
        core[..., 11] = S(self.dash_cd).float() / S(self._dash_cd_a).float() * 2 - 1
        core[..., 12] = S(self.dash_left).float() / W11.DASH_FRAMES * 2 - 1
        core[..., 13:21] = self._cast_rays(pos, maps) * 2 - 1
        score = S(self.score)
        my_score = torch.where(is_blue, score[:, :1], score[:, 1:])
        en_score = torch.where(is_blue, score[:, 1:], score[:, :1])
        core[..., 21] = (my_score / W11.SCORE_TO_WIN).clamp(max=1.0) * 2 - 1
        core[..., 22] = (en_score / W11.SCORE_TO_WIN).clamp(max=1.0) * 2 - 1
        alive_b = alive[:, :TS].sum(1, keepdim=True).float()
        alive_r = alive[:, TS:].sum(1, keepdim=True).float()
        core[..., 23] = torch.where(is_blue, alive_b, alive_r) / TS * 2 - 1
        core[..., 24] = torch.where(is_blue, alive_r, alive_b) / TS * 2 - 1
        lives = S(self.lives).float()
        core[..., 25] = torch.where(is_blue, lives[:, :1], lives[:, 1:]) / W11.TEAM_LIVES * 2 - 1
        core[..., 26] = torch.where(is_blue, lives[:, 1:], lives[:, :1]) / W11.TEAM_LIVES * 2 - 1
        core[..., 27] = (S(self.frame).float() / W11.MAX_FRAMES).clamp(max=1.0).view(n, 1) * 2 - 1
        core[..., 28] = pm1(self.map_compact[maps]).view(n, 1).expand(n, a)
        n_old = W11.CORE_SIZE
        core[..., n_old: n_old + N_TRAITS] = ltr
        c = n_old + N_TRAITS
        core[..., c] = pm1(alive & (S(self.cooldown) <= W11.ACTION_REPEAT))
        core[..., c + 1] = pm1(alive & (S(self.dash_cd) <= 0) & (S(self.dash_left) <= 0))
        rel = pos.unsqueeze(1) - pos.unsqueeze(2)
        dist = rel.pow(2).sum(-1).sqrt()
        same_team = team.unsqueeze(2) == team.unsqueeze(1)
        is_self = torch.eye(a, dtype=torch.bool, device=dev).view(1, a, a)
        other_alive = alive.unsqueeze(1).expand(n, a, a)
        ally = other_alive & same_team & ~is_self
        foe = other_alive & ~same_team
        # CHANGE vs world12: absolute local counts (/5, clamped). At TS = 5 the count never exceeds 5 -> identity.
        cnt = lambda m: (m.sum(2).float() / FORCE_COUNT_NORM).clamp(max=1.0) * 2 - 1    # noqa: E731
        core[..., c + 2] = cnt(ally & (dist <= FORCE_NEAR))
        core[..., c + 3] = cnt(foe & (dist <= FORCE_NEAR))
        core[..., c + 4] = cnt(ally & (dist <= FORCE_FAR))
        core[..., c + 5] = cnt(foe & (dist <= FORCE_FAR))
        core[..., c + 6] = cnt(foe & vis)
        wv = self.rules.respawn_wave_frames
        horizon = float(W11.RESPAWN_FRAMES + max(wv, 0))
        core[..., c + 7] = (self._respawn_eta(idx).float() / horizon).clamp(max=1.0) * 2 - 1
        if wv > 0:
            f = S(self.frame).view(n, 1).expand(n, a)
            core[..., c + 8] = ((wv - f % wv) % wv).float() / wv * 2 - 1
        else:
            core[..., c + 8] = -1.0
        obs[..., :CORE_SIZE12] = core

        # ---- entity slots (unchanged) ----
        present = other_alive & ~is_self & (same_team | vis)
        order = torch.where(present, dist, torch.full_like(dist, float("inf"))).argsort(dim=2)[:, :, :N_ENTITY_SLOTS]
        keep = present.gather(2, order).float().unsqueeze(-1)

        def g(t: torch.Tensor) -> torch.Tensor:
            src = t.unsqueeze(1).expand(n, a, *t.shape[1:])
            ix = order if t.dim() == 2 else order.unsqueeze(-1).expand(n, a, N_ENTITY_SLOTS, t.shape[2])
            return src.gather(2, ix)

        g_rel = rel.gather(2, order.unsqueeze(-1).expand(-1, -1, -1, 2))
        g_dist = dist.gather(2, order)
        g_vel, g_face = g(vel), g(facing)
        g_hp, g_shield = g(hp), g(shield)
        g_same = same_team.gather(2, order)
        g_vis = vis.gather(2, order)
        g_dash = g(S(self.dash_left)) > 0
        aim_err = torch.atan2(g_rel[..., 1], g_rel[..., 0]) - angle.unsqueeze(2)
        aim_err = (aim_err + math.pi) % (2 * math.pi) - math.pi
        their_ang = torch.atan2(-g_rel[..., 1], -g_rel[..., 0])
        their_err = their_ang - torch.atan2(g_face[..., 1], g_face[..., 0])
        their_err = (their_err + math.pi) % (2 * math.pi) - math.pi
        ones = torch.ones_like(g_dist)
        f_ = facing.unsqueeze(2)
        cross = f_[..., 0] * g_rel[..., 1] - f_[..., 1] * g_rel[..., 0]
        along = (f_ * g_rel).sum(-1)
        lateral = torch.where(along > 0, (cross / LATERAL_NORM).clamp(-1, 1), torch.where(cross >= 0, ones, -ones))
        ent = torch.cat([
            torch.stack([
                ones,
                torch.where(g_same, torch.full_like(g_dist, W11.ETYPE_ALLY), torch.full_like(g_dist, W11.ETYPE_ENEMY)),
                (g_rel[..., 0] / NEAR_RANGE).clamp(-1, 1),
                (g_rel[..., 1] / NEAR_RANGE).clamp(-1, 1),
                (g_dist / NEAR_RANGE).clamp(max=1.0) * 2 - 1,
                (g_vel[..., 0] / W11.VEL_NORM).clamp(-1, 1),
                (g_vel[..., 1] / W11.VEL_NORM).clamp(-1, 1),
                g_face[..., 0], g_face[..., 1],
                g_hp / g(max_hp) * 2 - 1,
                g_shield / g(max_sh) * 2 - 1,
                aim_err / math.pi,
                their_err / math.pi,
                torch.where(g_vis, ones, -ones),
                torch.where(g_dash, ones, -ones),
            ], dim=-1),
            g(ltr),
            torch.stack([
                (g_rel[..., 0] / vision).clamp(-1, 1),
                (g_rel[..., 1] / vision).clamp(-1, 1),
                (g_dist / vision).clamp(max=1.0) * 2 - 1,
                lateral,
                (g_dist / spd_a.unsqueeze(2) / FLIGHT_NORM_FRAMES).clamp(max=1.0) * 2 - 1,
                pm1(g_dist <= range_a.unsqueeze(2)),
                pm1(g_dist <= g(range_a)),
            ], dim=-1),
        ], dim=-1) * keep
        s = _SEG12["entities"]
        obs[..., s["start"]: s["start"] + N_ENTITY_SLOTS * ENTITY_SIZE12] = ent.reshape(n, a, -1)

        # ---- bullet slots: CHANGE vs world12 — compact to the alive bullets, then top-8 by topk ----
        b_alive_all = S(self.b_alive)
        kb = max(N_BULLET_SLOTS, int(b_alive_all.sum(1).max()))                  # one host sync
        kb = min(kb, b_alive_all.shape[1])
        bsel = torch.topk(b_alive_all.to(torch.uint8), kb, dim=1, sorted=False).indices
        take = lambda t: t.gather(1, bsel if t.dim() == 2 else bsel.unsqueeze(-1).expand(-1, -1, t.shape[2]))  # noqa: E731
        b_pos, b_vel, b_alive = take(S(self.b_pos)), take(S(self.b_vel)), take(b_alive_all)
        nb = kb
        b_rel = b_pos.unsqueeze(1) - pos.unsqueeze(2)
        b_dist = b_rel.pow(2).sum(-1).sqrt()
        b_team = self._agent_team[take(S(self.b_owner)).long()].unsqueeze(1)
        hostile = (b_team != team.unsqueeze(2)) & b_alive.unsqueeze(1)
        b_key = torch.where(hostile, b_dist, torch.full_like(b_dist, float("inf")))
        b_order = torch.topk(b_key, N_BULLET_SLOTS, dim=2, largest=False, sorted=True).indices
        b_keep = hostile.gather(2, b_order).float().unsqueeze(-1)
        idx2 = b_order.unsqueeze(-1).expand(-1, -1, -1, 2)
        g_brel = b_rel.gather(2, idx2)
        g_bvel = b_vel.unsqueeze(1).expand(n, a, nb, 2).gather(2, idx2)
        gb = lambda t: t.unsqueeze(1).expand(n, a, nb).gather(2, b_order)        # noqa: E731
        g_bspd, g_bdmg = gb(take(S(self.b_speed))), gb(take(S(self.b_dmg)))
        g_left = (gb(take(S(self.b_life))) - gb(take(S(self.b_age)))).clamp(min=0).float()
        vv = g_bvel.pow(2).sum(-1).clamp(min=1e-6)
        t_star = (-(g_brel * g_bvel).sum(-1) / vv).clamp(min=0.0)
        closest = (g_brel + g_bvel * t_star.unsqueeze(-1)).pow(2).sum(-1).sqrt()
        bones = torch.ones_like(t_star)
        bul = torch.stack([
            bones,
            (g_brel[..., 0] / W11.BULLET_OBS_RANGE).clamp(-1, 1),
            (g_brel[..., 1] / W11.BULLET_OBS_RANGE).clamp(-1, 1),
            g_bvel[..., 0] / g_bspd,
            g_bvel[..., 1] / g_bspd,
            (closest / W11.BULLET_MISS_NORM).clamp(max=1.0) * 2 - 1,
            (t_star / W11.BULLET_TTC_NORM).clamp(max=1.0) * 2 - 1,
            (g_bspd / 36.0).log(),
            (g_bdmg / float(W11.BULLET_DAMAGE)).log(),
            (g_left * g_bspd / REACH_NORM).clamp(max=1.0) * 2 - 1,
        ], dim=-1) * b_keep
        s = _SEG12["bullets"]
        obs[..., s["start"]: s["start"] + N_BULLET_SLOTS * BULLET_SIZE12] = bul.reshape(n, a, -1)

        # ---- control point slots (unchanged) ----
        cp_xy = self.map_cp_xy[maps]
        radius = self.map_cp_r[maps].view(n, 1, 1)
        c_rel = cp_xy.unsqueeze(1) - pos.unsqueeze(2)
        c_dist = c_rel.pow(2).sum(-1).sqrt()
        my_tid = (team + 1).unsqueeze(2)
        owner = S(self.cp_owner).unsqueeze(1)
        cap = S(self.cp_cap_team).unsqueeze(1)
        owner_val = torch.where(owner == my_tid, 1.0, torch.where(owner == 0, 0.0, -1.0))
        cap_val = torch.where(cap == my_tid, 1.0, torch.where(cap == 0, 0.0, -1.0))
        d_cp = c_dist.transpose(1, 2)
        near = (d_cp <= radius) & alive.unsqueeze(1)
        blue_near = near[:, :, :TS].any(2).unsqueeze(1)
        red_near = near[:, :, TS:].any(2).unsqueeze(1)
        mine_near = torch.where(is_blue.unsqueeze(2), blue_near, red_near)
        foe_near = torch.where(is_blue.unsqueeze(2), red_near, blue_near)
        tsz = self.tile_size
        row = (pos[..., 1] / tsz).long().clamp(0, self.map_grid.shape[1] - 1)
        col = (pos[..., 0] / tsz).long().clamp(0, self.map_grid.shape[2] - 1)
        mm = maps.view(n, 1, 1).expand(n, a, N_CP_SLOTS)
        kk = torch.arange(N_CP_SLOTS, device=dev).view(1, 1, -1).expand(n, a, -1)
        rr, cc = row.unsqueeze(2).expand(-1, -1, N_CP_SLOTS), col.unsqueeze(2).expand(-1, -1, N_CP_SLOTS)
        p_dist = self.flow_dist[mm, kk, rr, cc]
        p_dir = self.flow_dir[mm, kk, rr, cc]
        path_norm = 1.5 * diag
        p_obs = torch.where(torch.isfinite(p_dist), (p_dist / path_norm).clamp(max=1.0) * 2 - 1, torch.ones_like(p_dist))
        cones = torch.ones_like(c_dist)
        cps = torch.stack([
            cones,
            owner_val.expand(n, a, N_CP_SLOTS),
            c_rel[..., 0] / diag,
            c_rel[..., 1] / diag,
            (c_dist / diag).clamp(max=1.0) * 2 - 1,
            p_obs,
            p_dir[..., 0], p_dir[..., 1],
            S(self.cp_progress).unsqueeze(1).expand(n, a, N_CP_SLOTS) * 2 - 1,
            cap_val.expand(n, a, N_CP_SLOTS),
            torch.where(foe_near, 1.0, -1.0).expand(n, a, N_CP_SLOTS),
            torch.where(mine_near, 1.0, -1.0).expand(n, a, N_CP_SLOTS),
            torch.where(c_dist <= radius, 1.0, -1.0),
            (c_dist / radius).clamp(max=2.0) - 1.0,
        ], dim=-1)
        s = _SEG12["cps"]
        obs[..., s["start"]: s["start"] + N_CP_SLOTS * s["size"]] = cps.reshape(n, a, -1)

        # ---- last-seen memory: CHANGE vs world12 — the 5 nearest shown enemies (slot order) ----
        mem_valid = S(self.mem_valid)
        my_team_mem = lambda t: torch.cat([t[:, 0:1].expand(n, TS, *t.shape[2:]), t[:, 1:2].expand(n, TS, *t.shape[2:])], dim=1)  # noqa: E731
        m_pos, m_vel = my_team_mem(S(self.mem_pos)), my_team_mem(S(self.mem_vel))
        m_hp, m_age, m_val = my_team_mem(S(self.mem_hp)), my_team_mem(S(self.mem_age)), my_team_mem(mem_valid)
        enemy_idx = torch.cat([torch.arange(TS, 2 * TS), torch.arange(0, TS)]).to(dev)
        enemy_idx = enemy_idx.view(2, TS)[self._agent_team]                       # (A, TS)
        i_see = vis.gather(2, enemy_idx.view(1, a, TS).expand(n, -1, -1))
        shown = m_val & ~i_see
        show = shown.float().unsqueeze(-1)
        en_max_hp = max_hp[:, enemy_idx]
        en_ltr = ltr[:, enemy_idx]
        l_rel = m_pos - pos.unsqueeze(2)
        l_dist = l_rel.pow(2).sum(-1).sqrt()
        lones = torch.ones_like(l_dist)
        ls = torch.cat([
            torch.stack([
                lones,
                (l_rel[..., 0] / diag.view(n, 1, 1)),
                (l_rel[..., 1] / diag.view(n, 1, 1)),
                (l_dist / diag.view(n, 1, 1)).clamp(max=1.0) * 2 - 1,
                m_age.float() / W11.LASTSEEN_HORIZON_FRAMES * 2 - 1,
                torch.where(m_age == 0, lones, -lones),
                (m_vel[..., 0] / W11.VEL_NORM).clamp(-1, 1),
                (m_vel[..., 1] / W11.VEL_NORM).clamp(-1, 1),
                m_hp / en_max_hp * 2 - 1,
            ], dim=-1),
            en_ltr,
        ], dim=-1) * show                                                          # (n, A, TS, 17)
        if TS > N_LASTSEEN_SLOTS:
            key = torch.where(shown, l_dist, torch.full_like(l_dist, float("inf")))
            pick = torch.topk(key, N_LASTSEEN_SLOTS, dim=2, largest=False).indices.sort(dim=2).values
            ls = ls.gather(2, pick.unsqueeze(-1).expand(-1, -1, -1, ls.shape[-1]))
        s = _SEG12["lastseen"]
        obs[..., s["start"]: s["start"] + N_LASTSEEN_SLOTS * LASTSEEN_SIZE12] = ls.reshape(n, a, -1)

        # ---- egocentric grid (unchanged) ----
        G, GS = W11.GRID_RADIUS, W11.GRID_SIDE
        pr = row + G
        pc = col + G
        gr = pr.view(n, a, 1, 1) + self._grid_dr.view(1, 1, GS, GS)
        gc = pc.view(n, a, 1, 1) + self._grid_dc.view(1, 1, GS, GS)
        gm = maps.view(n, 1, 1, 1).expand(n, a, GS, GS)
        walls = self.map_grid_pad[gm, gr, gc].float()
        zone = self.map_zone_pad[gm, gr, gc]
        z_owner = S(self.cp_owner).gather(1, zone.clamp(min=0).view(n, -1)).view(n, a, GS, GS)
        my_t = (team + 1).view(n, a, 1, 1)
        z_val = torch.where(z_owner == my_t, 1.0, torch.where(z_owner == 0, 0.5, -1.0))
        z_val = torch.where(zone >= 0, z_val, torch.zeros_like(z_val))
        s = _SEG12["grid"]
        obs[..., s["start"]:] = torch.stack([walls, z_val], dim=2).reshape(n, a, -1)
        return obs.clamp(-1.0, 1.0)
    # endregion FUNC__build_obs

    # region FUNC_global_state
    ## @purpose world12's 249-column team state for any team size: five squads per team (team_size/5
    ## consecutive slots each) replace the five agents. Kinematics are averaged over alive members (all members if
    ## none alive); the status columns (hp, shield, alive, timers) and traits are averaged over all members, i.e.
    ## world12's per-agent values pooled. At TS = 5 every squad is one agent -> world12's state exactly.
    def global_state(self) -> torch.Tensor:
        W11 = self._W11
        n, dev, TS = self.n, self.device, self.team_size
        k = TS // N_SQUADS
        arena = self.map_arena[self.map_idx].unsqueeze(1)
        per = torch.stack([
            self.pos[..., 0] / arena[..., 0] * 2 - 1,
            self.pos[..., 1] / arena[..., 1] * 2 - 1,
            (self.vel[..., 0] / W11.VEL_NORM).clamp(-1, 1),
            (self.vel[..., 1] / W11.VEL_NORM).clamp(-1, 1),
            self.angle.cos(), self.angle.sin(),
            torch.where(self.alive, self.hp / self._max_hp * 2 - 1, torch.full_like(self.hp, -1.0)),
            torch.where(self.alive, self.shield / self._max_sh * 2 - 1, torch.full_like(self.hp, -1.0)),
            torch.where(self.alive, 1.0, -1.0),
            self.respawn_timer.float() / W11.RESPAWN_FRAMES * 2 - 1,
            self.dash_cd.float() / self._dash_cd_a.float() * 2 - 1,
            self.cooldown.float() / self._cd_frames.float() * 2 - 1,
        ], dim=-1)                                                                # (n, A, 12)
        ltr = self._log_traits
        if k > 1:
            sq = per.view(n, 2 * N_SQUADS, k, STATE_AGENT)
            al = self.alive.view(n, 2 * N_SQUADS, k, 1).float()
            cnt = al.sum(2)
            kin = torch.where(cnt > 0, (sq[..., :6] * al).sum(2) / cnt.clamp(min=1.0), sq[..., :6].mean(2))
            per = torch.cat([kin, sq[..., 6:].mean(2)], dim=-1)                   # (n, 10, 12)
            ltr = ltr.view(n, 2 * N_SQUADS, k, N_TRAITS).mean(2)
        cp_xy = self.map_cp_xy[self.map_idx]
        out = torch.zeros(n, 2, STATE_SIZE12, device=dev)
        onehot = self.map_desc[self.map_idx]
        for t in (0, 1):
            me = slice(t * N_SQUADS, (t + 1) * N_SQUADS)
            en = slice((1 - t) * N_SQUADS, (2 - t) * N_SQUADS)
            agents = torch.cat([per[:, me], per[:, en]], dim=1).reshape(n, -1)
            tid = t + 1
            own = torch.where(self.cp_owner == tid, 1.0, torch.where(self.cp_owner == 0, 0.0, -1.0))
            capv = torch.where(self.cp_cap_team == tid, 1.0, torch.where(self.cp_cap_team == 0, 0.0, -1.0))
            cps = torch.stack([cp_xy[..., 0] / arena[..., 0] * 2 - 1, cp_xy[..., 1] / arena[..., 1] * 2 - 1, own,
                               self.cp_progress * 2 - 1, capv], dim=-1).reshape(n, -1)
            glob = torch.stack([
                (self.score[:, t] / W11.SCORE_TO_WIN).clamp(max=1.0) * 2 - 1,
                (self.score[:, 1 - t] / W11.SCORE_TO_WIN).clamp(max=1.0) * 2 - 1,
                self.lives[:, t].float() / W11.TEAM_LIVES * 2 - 1,
                self.lives[:, 1 - t].float() / W11.TEAM_LIVES * 2 - 1,
                (self.frame.float() / W11.MAX_FRAMES).clamp(max=1.0) * 2 - 1,
                torch.where(self.map_compact[self.map_idx], 1.0, -1.0),
            ], dim=-1)
            tr = torch.cat([ltr[:, me], ltr[:, en]], dim=1).reshape(n, -1)
            out[:, t] = torch.cat([agents, cps, glob, onehot, tr], dim=1)
        return out
    # endregion FUNC_global_state

    def snapshot(self, world_idx: int) -> dict:
        snap = super().snapshot(world_idx)                                        # type: ignore[misc]
        if self._disabled is not None:
            d = self._disabled[int(world_idx)].tolist()
            for ag in snap["agents"]:
                ag["disabled"] = bool(d[ag["id"]])
        return snap
# endregion CLASS__Mixin13


# region FUNC_world13_class
## @purpose The World13 class for a team size (multiple of 5): clones of world11 (rule constants rebound) and
## world12 (executed against that clone), then _Mixin13 on top. Cached per (team_size, max_frames, score).
## @io team_size -> class(n_worlds, device, seed, rules=None, uneven_share=0, uneven_max=0, **World12 kwargs)
def world13_class(team_size: int, max_frames: int | None = None, score_to_win: float | None = None) -> type:
    ts = int(team_size)
    if ts < BASE_TEAM or ts % N_SQUADS:
        raise ValueError(f"team_size={ts}: must be a multiple of {N_SQUADS} and >= {BASE_TEAM}")
    key = (ts, max_frames, score_to_win)
    if key in _CLASS_CACHE:
        return _CLASS_CACHE[key]
    from rush import maps13
    maps_ns = types.SimpleNamespace(build_pool=maps13.build_pool13)
    i = next(_counter)
    w11 = _exec_clone(_W11_PATH, f"rush._world11_t{ts}_{i}", {"rush.maps11": maps_ns})
    # Rule constants only; layout constants (OBS_LAYOUT, N_*_SLOTS, STATE_SIZE) keep their exec-time 5v5 values.
    w11.TEAM_SIZE = ts
    w11.N_AGENTS = 2 * ts
    w11.TEAM_LIVES = LIVES_PER_AGENT * ts
    w11.BULLET_SLOTS = int(round(BULLETS_PER_AGENT * 2 * ts))
    if max_frames is not None:
        w11.MAX_FRAMES = int(max_frames)
    if score_to_win is not None:
        w11.SCORE_TO_WIN = float(score_to_win)
    w12 = _exec_clone(_W12_PATH, f"rush._world12_t{ts}_{i}",
                      {"rush.world11": w11, "rush.maps11": maps_ns})
    w12.STATE_SIZE12 = STATE_SIZE12                    # squads keep world12's 249 columns
    assert w12.OBS_SIZE12 == OBS_SIZE12 and w11.N_ENTITY_SLOTS == N_ENTITY_SLOTS and w11.N_LASTSEEN_SLOTS == N_LASTSEEN_SLOTS
    cls = type(f"World13_T{ts}", (_Mixin13, w12.World12), {"_W11": w11, "_W12": w12, "team_size": ts})
    _CLASS_CACHE[key] = cls
    logger.info(f"[IMP:9][world13_class][BUILD] team_size={ts}: clones {w11.__name__}, {w12.__name__}; "
                f"TEAM_LIVES={w11.TEAM_LIVES} BULLET_SLOTS={w11.BULLET_SLOTS} MAX_FRAMES={w11.MAX_FRAMES} "
                f"SCORE_TO_WIN={w11.SCORE_TO_WIN} obs={w12.OBS_SIZE12} state={w12.STATE_SIZE12} [VALUE]")
    return cls
# endregion FUNC_world13_class


# region FUNC_World13
def World13(n_worlds: int, team_size: int = 5, device: str = "cpu", seed: int = 0, rules: Any = None, **kw: Any):
    return world13_class(team_size)(n_worlds, device=device, seed=seed, rules=rules, **kw)
# endregion FUNC_World13
