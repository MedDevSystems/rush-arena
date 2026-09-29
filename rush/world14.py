from __future__ import annotations

# region MODULE_CONTRACT [DOMAIN(9): Simulation; CONCEPT(9): LadderWorlds14; TECH(8): torch]
## @modulecontract
## @purpose The worlds experiment 14 trains in: batched World13(ts) and single big-battle training worlds on
## world_big12's physics — both with (a) per-agent deaths kept for the class x class kill matrix and (b) the
## commander hook: a respawn plan that changes an agent's class and spawn point at its next respawn.
## @scope Class construction (clones, mixins), the big world's rules overrides and critic state, the respawn plan.
## No trainer logic (vec14), no maps (maps14 / maps_big).
## @input team size (batched) or a BigMap (big); rules overrides (range_px, friendly_fire, ff_coef)
## @output world instances with world12's API (step -> StepOut with info["action_mask"], observe, global_state,
## set_traits, set_pending_traits, set_reward_weights) plus last_deaths, slot_fork, set_respawn_plan
## @links LINKS_TO: world13 (world13_class), world_big12 (_Big12Mixin, clones), world_big (_BigMixin, battle_rules),
## world12 (Rules12, TRAITS12), vec14, docs/EXP14_CONTRACT.md
## @invariants
## - Without a respawn plan the worlds behave exactly as their bases (tests/test_ladder14.py compares step by step)
## - A planned respawn happens at the slot's normal respawn time; only its traits (then HP/shield to the new maximum)
##   and position (a CP the team owns at that moment, else the zone spawn already drawn) change
## - The big world's team state has world12's 249 columns: 5 squads per team pooled as in world13, and the CPs
##   pooled into 7 groups by column (the critic layout has 7 CP slots)
## @rationale
## Q: Why a proxy for world12 in the big world?
## A: world_big12's mixin calls `self._wm12.World12.__init__` with its own Rules12 and no overrides. The proxy's
## A: World12.__init__ forwards the trainer's overrides (World12 accepts `rules_overrides`) and delegates everything
## A: else to the clone, so no world_big12 code is copied or edited.
## @changes
## LAST_CHANGE: [v0.1.0] Initial: respawn plans, deaths stash, big training world with squad state.
## @modulemap
## CLASS 8[Respawn plan (commander hook)] => RespawnPlanMixin
## CLASS 6[Per-agent deaths of the last decision] => _DeathsMixin
## FUNC 7[Batched class for a team size] => batched_world_class
## CLASS 8[Big-world critic state and overrides] => _BigTrainMixin
## FUNC 8[Big training world for a map] => BigTrainWorld
def _module_contract():
    pass
# endregion MODULE_CONTRACT
# GREP_SUMMARY: world14, respawn plan, commander hook, spawn at CP, class at respawn, big training world, squad state, deaths
# STRUCTURE: ▶ batched: (plan, deaths, World13(ts)) | big: clones + proxy(rules) → (plan, deaths, BigTrain, Big12, Big, World12) → ⎋ world

import logging
import math
import types
from typing import Any

import torch

from rush.forks12 import FORKS, TRAITS12, trait_vector

logger = logging.getLogger(__name__)

N_SQUADS: int = 5
STATE_AGENT: int = 12
STATE_CPS: int = 7


def score_to_win_for(ts: int) -> float | None:
    """Same scaling as vec13 (sqrt of team size / 5); None keeps world11's 200 at 5v5."""
    return None if ts == 5 else float(round(200.0 * math.sqrt(ts / 5.0)))


# region CLASS_RespawnPlanMixin
## @purpose Commander hook. set_respawn_plan(world_idx, team, [(slot, fork_or_traits, cp_or_zone)]): at that
## slot's next respawn the agent takes the class (traits; slot_fork records the fork name index) and spawns at the
## given control point if its team owns it then (else in its spawn zone, as always). Default: no plan, no change.
## @complexity 7
class RespawnPlanMixin:
    FORK_NAMES: list[str] = list(FORKS)

    def _plan_init(self) -> None:
        if getattr(self, "_plans", None) is None:
            self._plans: dict[tuple[int, int], tuple[torch.Tensor | None, int | None, int]] = {}
            self.slot_fork = torch.full((self.n, 2 * self._team_size_plan()), -1, dtype=torch.long, device=self.device)

    def _team_size_plan(self) -> int:
        return int(self.alive.shape[1] // 2)

    def set_slot_forks(self, fork_idx: torch.Tensor) -> None:
        """Which fork controls each slot (n, A) — bookkeeping for a future commander; physics reads traits only."""
        self._plan_init()
        self.slot_fork = fork_idx.to(self.device).long().clone()

    ## @io world_idx int, team 0/1, entries [(slot 0..ts-1, fork name | trait list/tensor | None, cp index | None)]
    def set_respawn_plan(self, world_idx: int, team: int, entries: list[tuple[int, Any, Any]]) -> None:
        self._plan_init()
        ts = self._team_size_plan()
        for slot, cls, target in entries:
            if not 0 <= int(slot) < ts:
                raise ValueError(f"slot {slot} outside 0..{ts - 1}")
            agent = int(team) * ts + int(slot)
            fork_i = -1
            if cls is None:
                tv = None
            elif isinstance(cls, str):
                tv = torch.tensor(trait_vector(cls), dtype=torch.float32, device=self.device)
                fork_i = self.FORK_NAMES.index(cls)
            else:
                tv = torch.as_tensor(cls, dtype=torch.float32, device=self.device).view(len(TRAITS12))
            cp = None if target in (None, "zone") else int(target)
            if cp is not None and not 0 <= cp < int(self.map_cp_xy.shape[1]):
                raise ValueError(f"cp {cp} outside 0..{int(self.map_cp_xy.shape[1]) - 1}")
            self._plans[(int(world_idx), agent)] = (tv, cp, fork_i)
        logger.info(f"[IMP:8][RespawnPlanMixin.set_respawn_plan][EXEC] world={world_idx} team={team} entries={len(entries)} "
                    f"pending={len(self._plans)} [VALUE]")

    def pending_plans(self) -> int:
        return len(getattr(self, "_plans", None) or {})

    def _respawn(self) -> torch.Tensor:
        plans = getattr(self, "_plans", None)
        ready = super()._respawn()                                                    # type: ignore[misc]
        if not plans or not bool(ready.any()):
            return ready
        hits = [(w, a) for (w, a) in list(plans) if bool(ready[w, a])]
        if not hits:
            return ready
        traits_changed = False
        for w, a in hits:
            tv, cp, fork_i = plans.pop((w, a))
            if tv is not None:
                self.traits[w, a] = tv
                traits_changed = True
                if fork_i >= 0:
                    self.slot_fork[w, a] = fork_i
            if cp is not None:
                team = 0 if a < self._team_size_plan() else 1
                if int(self.cp_owner[w, cp]) == team + 1:
                    xy = self.map_cp_xy[self.map_idx[w], cp]
                    jit = (torch.rand(2, generator=self.gen) - 0.5).to(self.device) * 30.0     # within the carved CP disc
                    self.pos[w, a] = xy + jit
        if traits_changed:
            self._derive()
            self.hp = torch.where(ready, self._max_hp, self.hp)
            self.shield = torch.where(ready, self._max_sh, self.shield)
        self._vis_cache = None
        return ready
# endregion CLASS_RespawnPlanMixin


# region CLASS__DeathsMixin
## @purpose Keep the per-agent deaths of the last decision (world12 passes them to _reward but not to info).
class _DeathsMixin:
    def _reward(self, dmg_dealt, dmg_taken, kills, deaths, *rest):                     # type: ignore[override]
        self.last_deaths = deaths.clone()
        return super()._reward(dmg_dealt, dmg_taken, kills, deaths, *rest)            # type: ignore[misc]
# endregion CLASS__DeathsMixin


# region FUNC_batched_world_class
_BATCHED: dict[tuple[int, bool], type] = {}
# World12.sync_guards for experiment 14's batched worlds: False skips the physics' early-exit host syncs (bit-exact,
# tests/test_perf14.py). train14 --sync-guards 1 restores them.
SYNC_GUARDS: bool = False


def batched_world_class(ts: int, sync_guards: bool | None = None) -> type:
    sg = SYNC_GUARDS if sync_guards is None else bool(sync_guards)
    key = (ts, sg)
    if key not in _BATCHED:
        from rush.world13 import world13_class
        base = world13_class(ts, score_to_win=score_to_win_for(ts))
        _BATCHED[key] = type(f"World14_ts{ts}", (RespawnPlanMixin, _DeathsMixin, base), {"sync_guards": sg})
        logger.info(f"[IMP:8][batched_world_class][BUILD] World14 ts={ts} on {base.__name__} score_to_win={score_to_win_for(ts)} "
                    f"sync_guards={sg} [VALUE]")
    return _BATCHED[key]
# endregion FUNC_batched_world_class


# region CLASS__BigTrainMixin
## @purpose Big-battle world for training: world12's 249-column team state (squads pooled, 7 CP groups).
class _BigTrainMixin:
    def global_state(self) -> torch.Tensor:
        from rush.world12 import N_TRAITS, STATE_SIZE12
        W11 = self._wm
        n, dev, TS = self.n, self.device, self.T
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
        ], dim=-1)                                                                    # (n, A, 12)
        sq = per.view(n, 2 * N_SQUADS, k, STATE_AGENT)
        al = self.alive.view(n, 2 * N_SQUADS, k, 1).float()
        cnt = al.sum(2)
        kin = torch.where(cnt > 0, (sq[..., :6] * al).sum(2) / cnt.clamp(min=1.0), sq[..., :6].mean(2))
        per = torch.cat([kin, sq[..., 6:].mean(2)], dim=-1)                           # (n, 10, 12)
        ltr = self._log_traits.view(n, 2 * N_SQUADS, k, N_TRAITS).mean(2)
        cp_xy = self.map_cp_xy[self.map_idx]                                          # (n, C, 2)
        grp = self._cp_groups                                                         # (7, C) weights
        out = torch.zeros(n, 2, STATE_SIZE12, device=dev)
        onehot = self.map_desc[self.map_idx]
        for t in (0, 1):
            me = slice(t * N_SQUADS, (t + 1) * N_SQUADS)
            en = slice((1 - t) * N_SQUADS, (2 - t) * N_SQUADS)
            agents = torch.cat([per[:, me], per[:, en]], dim=1).reshape(n, -1)
            tid = t + 1
            own = torch.where(self.cp_owner == tid, 1.0, torch.where(self.cp_owner == 0, 0.0, -1.0))
            capv = torch.where(self.cp_cap_team == tid, 1.0, torch.where(self.cp_cap_team == 0, 0.0, -1.0))
            feats = torch.stack([cp_xy[..., 0] / arena[..., 0] * 2 - 1, cp_xy[..., 1] / arena[..., 1] * 2 - 1, own,
                                 self.cp_progress * 2 - 1, capv], dim=-1)             # (n, C, 5)
            cps = torch.einsum("gc,ncf->ngf", grp, feats).reshape(n, -1)              # (n, 35)
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

    def _build_cp_groups(self) -> None:
        """7 groups of CPs by column (then row): the group feature is the mean of its CPs' features."""
        xy = self.map_cp_xy[0].detach().cpu()
        c = xy.shape[0]
        order = sorted(range(c), key=lambda i: (float(xy[i, 0]), float(xy[i, 1])))
        g = torch.zeros(STATE_CPS, c)
        for j, i in enumerate(order):
            g[min(STATE_CPS - 1, j * STATE_CPS // c), i] = 1.0
        g = g / g.sum(1, keepdim=True).clamp(min=1.0)
        self._cp_groups = g.to(self.device)
# endregion CLASS__BigTrainMixin


# region FUNC_BigTrainWorld
class _ClsProxy:
    """Stands in for the world12 clone's class inside a proxy module: __init__ forwards rules overrides."""

    def __init__(self, cls: type, overrides: dict) -> None:
        self.__dict__["_cls"] = cls

        def init(obj, n_worlds, **kw):
            return cls.__init__(obj, n_worlds, rules_overrides=overrides, **kw)
        self.__dict__["__init__"] = init

    def __getattr__(self, name: str) -> Any:
        return getattr(self.__dict__["_cls"], name)


class _ModProxy(types.ModuleType):
    def __init__(self, mod: types.ModuleType, overrides: dict) -> None:
        super().__init__(mod.__name__ + "_proxy")
        self.__dict__["_mod"] = mod
        self.__dict__["World12"] = _ClsProxy(mod.World12, overrides)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.__dict__["_mod"], name)


## @purpose One big-battle training world for a map: world_big12's clones, the trainer's rule overrides, the
## squad critic state, deaths stash and respawn plans. record=False (no events: training does not need them).
def BigTrainWorld(big_map: Any, device: str = "cpu", seed: int = 0, rules_overrides: dict | None = None):  # noqa: N802
    from rush.world_big import _BigMixin, _world11_clone, battle_rules
    from rush.world_big12 import _Big12Mixin, _world12_clone
    rules = battle_rules(big_map)
    if rules["team_size"] % N_SQUADS:
        raise ValueError(f"big training world needs team size divisible by {N_SQUADS}: {rules['team_size']}")
    mod11 = _world11_clone(rules["team_size"], len(big_map.cp_positions), rules)
    mod12 = _world12_clone(mod11)
    ov = {k: v for k, v in (rules_overrides or {}).items() if v is not None}
    cls = type("BigTrainWorld14", (RespawnPlanMixin, _DeathsMixin, _BigTrainMixin, _Big12Mixin, _BigMixin, mod12.World12),
               {"_wm": mod11, "_wm12": _ModProxy(mod12, ov), "sync_guards": SYNC_GUARDS})
    w = cls(big_map, device=device, seed=seed, record=False)
    w._build_cp_groups()
    return w
# endregion FUNC_BigTrainWorld
