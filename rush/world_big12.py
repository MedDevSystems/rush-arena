from __future__ import annotations

# region MODULE_CONTRACT [DOMAIN(8): Spectating; CONCEPT(9): BigBattleWorld12; TECH(9): torch+scipy]
## @modulecontract
## @purpose One large match (team_size per side, up to 500 v 500) with world12's physics, rules and
## observation (experiment 12: long range, exact line of sight, swept bullets, per-agent traits, wave
## respawn, action masks), so the exp12 forks play it unchanged — each agent with its own fork's traits.
## @scope A single world, no auto-reset. Show only: rewards are computed (world12 code) but nobody learns.
## @input A BigMap (maps_big) or any MapDef-like object; actions (1, 2T, 4); per-agent traits (1, 2T, 8)
## @output world11.StepOut with info["action_mask"]; snapshot / map_geometry / events for replay11
## @links USES_API(9): torch, scipy.sparse.csgraph; LINKS_TO: world12 (physics + obs source), world11 (base),
## world_big (map tables, flow fields, spawn resets, snapshot — reused), record_big12, docs/BIG12_NOTES.md
## @invariants
## - Observation is world12's OBS_LAYOUT12 (1029 wide) — asserted when a world is built
## - Physics (move, turn, dash, shoot with per-agent cooldown/speed/damage, substepped bullets, hits with
##   per-agent radius, shields scaled by max_hp, wave respawn, control points, score, rewards with
##   per-agent weights, teamwork bookkeeping, step order) is world12's and world11's SOURCE, executed in
##   per-battle clones of both modules whose rule constants are rebound to this battle
## - With team_size = 5 on a maps11 map the observation, rewards and physics equal World12(default rules)
##   bit for bit (tests/test_world_big12.py), including non-baseline traits and trained-policy matches
## @rationale
## Q: How is world12 cloned, given that it imports world11?
## A: First world11 is executed as a fresh module with the battle's constants (world_big._world11_clone).
## A: Then world12's source is executed as another fresh module while `rush.world11` — both the
## A: package attribute (used by `from rush import world11 as W11`) and the sys.modules entry (used
## A: by `from rush.world11 import ...`) — temporarily point to that clone. The clone's World12 thus
## A: subclasses the clone's World11 and every W11.X / N_AGENTS it reads is this battle's. The real modules
## A: are restored in a finally block and are never mutated.
## Q: What is overridden, and why is none of it physics?
## A: Map tables, flow fields, spawn resets, snapshot and geometry (world_big's _BigMixin, reused as is);
## A: visibility (exact tile-crossing LOS of world12, but only for the 9 nearest allies and KE_CAP nearest
## A: enemies in the viewer's vision — 41k segments instead of 500k); hit resolution (world12's own method
## A: run on the compacted alive bullets); the observation builder (world12's, with world_big's changes:
## A: battle normalisations, capped diagonal, 7 nearest CPs, 5 nearest last-seen enemies, top-8 alive
## A: bullets); global_state (no critic in a show).
## Q: Where can it differ from World12?
## A: Only through the culled visibility: when more than KE_CAP enemies are nearer than a visible enemy that
## A: would still rank among the 9 nearest present entities. At team_size <= KE_CAP it is exact.
## @changes
## LAST_CHANGE: [v0.1.0] Initial exp12 big-battle world.
## @modulemap
## FUNC 9[Clone world12 on a world11 clone] => _world12_clone
## CLASS 8[Rules12 + battle rule fields] => BigRules12
## CLASS 9[exp12 overrides on top of _BigMixin and the clone's World12] => _Big12Mixin
## FUNC 9[Build a WorldBig12 for a map] => WorldBig12
## @usecases
## - w = WorldBig12(get_map("Warfront500")); w.set_traits(t); w.refill(); obs = w.observe(); out = w.step(acts)
def _module_contract():
    pass
# endregion MODULE_CONTRACT
# GREP_SUMMARY: big battle, exp12, world_big12, clone world12, traits, masks, exact LOS culled, nearest CPs, 500v500
# STRUCTURE: ▶ BigMap → ⚡ clone world11(constants) → ⚡ clone world12 on it → ⚡ class(Big12, _BigMixin, clone.World12) → ○ step (world12 source) → ⎋ StepOut + mask / snapshot

import importlib.util
import itertools
import logging
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any

import torch

import rush
from rush import world11 as W11
from rush import world12 as W12
from rush.world_big import DIAG_CAP_PX, KE_CAP, _BigMixin, _world11_clone, battle_rules

logger = logging.getLogger(__name__)

# region BLOCK_CONSTANTS
OBS_LAYOUT12: list[dict] = W12.OBS_LAYOUT12
OBS_SIZE12: int = W12.OBS_SIZE12
assert OBS_SIZE12 == 1029, f"world12 OBS_SIZE12 drifted: {OBS_SIZE12}"
N_ENTITY_SLOTS: int = W11.N_ENTITY_SLOTS        # 9
N_BULLET_SLOTS: int = W11.N_BULLET_SLOTS        # 8
N_CP_SLOTS: int = W11.N_CP_SLOTS                # 7
N_LASTSEEN_SLOTS: int = W11.N_LASTSEEN_SLOTS    # 5
BULLETS_PER_AGENT12: int = 24                   # 240-frame life / 9-12 frame cooldown; drops are counted
LOS_CHUNK_PAIRS: int = 8192                     # segments per exact-LOS call (bounds the (P, span) temporaries)
HIT_CELL_PX: float = 32.0                       # hit prefilter grid; >= the largest hit radius (3 + 12 * 1.25 = 18 px)
HIT_PREFILTER_MIN: int = 512                    # below this many alive bullets the prefilter is not worth it
BULLET_TOPK_MIN: int = 512                      # above this many alive bullets: topk(8) instead of a full sort
_WORLD12_PATH: Path = Path(W12.__file__)
_clone_counter = itertools.count()
# endregion BLOCK_CONSTANTS


# region CLASS_BigRules12
## @purpose world12's Rules12 plus the battle's rule fields, readable as rules["spawn_zones"] — the key
## world_big's _BigMixin reads — while world12 reads rules.bullet_speed etc. from the same object.
@dataclass(frozen=True)
class BigRules12(W12.Rules12):
    battle: dict = field(default_factory=dict, compare=False, hash=False)

    def __getitem__(self, key: str) -> Any:
        return self.battle[key]
# endregion CLASS_BigRules12


# region FUNC__world12_clone
## @purpose Execute world12's source on top of a world11 clone (see @rationale in the module contract).
## @io world11 clone module -> world12 clone module
def _world12_clone(mod11: ModuleType) -> ModuleType:
    name = f"rush._world12_big_{next(_clone_counter)}"
    spec = importlib.util.spec_from_file_location(name, _WORLD12_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    real_attr = rush.world11
    real_mod = sys.modules["rush.world11"]
    try:
        rush.world11 = mod11
        sys.modules["rush.world11"] = mod11
        spec.loader.exec_module(mod)
    finally:
        rush.world11 = real_attr
        sys.modules["rush.world11"] = real_mod
    assert mod.W11 is mod11 and mod.N_AGENTS == mod11.N_AGENTS, "world12 clone did not bind the world11 clone"
    assert mod.OBS_SIZE12 == OBS_SIZE12, f"clone layout {mod.OBS_SIZE12} != {OBS_SIZE12}"
    assert W11.TEAM_SIZE == 5 and W12.N_AGENTS == 10, "real modules were mutated"
    return mod
# endregion FUNC__world12_clone


# region CLASS__Big12Mixin
## @purpose exp12 overrides for one large battle, above world_big's _BigMixin (tables, flow, spawns, snapshot).
## @complexity 9
class _Big12Mixin:
    _wm: ModuleType          # world11 clone
    _wm12: ModuleType        # world12 clone

    def __init__(self, big_map: Any, device: str = "cpu", seed: int = 0, record: bool = True,
                 reward_preset: str = W11.DEFAULT_REWARD_PRESET) -> None:
        self.big = big_map
        battle = battle_rules(big_map)
        self.T = battle["team_size"]
        self.A = 2 * self.T
        slots = max(W12.Rules12().bullet_slots, BULLETS_PER_AGENT12 * self.A)
        rules = BigRules12(bullet_slots=slots, battle=battle)
        t0 = time.perf_counter()
        self._wm12.World12.__init__(self, 1, device=device, seed=seed, rules=rules, record=record, map_pool="big",
                                    reward_preset=reward_preset)
        self.auto_reset = False          # one match to its end
        # _BigMixin.reset_worlds ran inside World11.__init__, before World12 re-sized the bullet ring
        self.b_id = torch.full_like(self.b_owner, -1, dtype=torch.long)
        wm = self._wm
        logger.info(f"[IMP:9][WorldBig12.__init__][INIT] map={big_map.name}, team_size={self.T}, agents={self.A}, cps={self.n_cps}, "
                    f"tiles={big_map.cols}x{big_map.rows}, bullet_slots={slots}, substeps={self._substeps}, range={rules.bullet_range:.0f}px, "
                    f"vision={rules.vision:.0f}px, wave={rules.respawn_wave_frames}, score_to_win={wm.SCORE_TO_WIN}, team_lives={wm.TEAM_LIVES}, "
                    f"max_frames={wm.MAX_FRAMES}, norm_diag={self.norm_diag:.0f}px (true {float(self.map_diag[0]):.0f}), "
                    f"obs={OBS_SIZE12}, build={time.perf_counter() - t0:.1f}s [VALUE]")

    # region FUNC_refill
    ## @purpose After set_traits at the start of a match: HP and shield to each agent's new maximum.
    def refill(self) -> None:
        self.hp = torch.where(self.alive, self._max_hp, self.hp)
        self.shield = torch.where(self.alive, self._max_sh, self.shield)
        self._vis_cache = None
    # endregion FUNC_refill

    # region FUNC_reset_worlds
    ## @purpose world_big's spawn reset, then world12's per-match extras (its _reset_idx is not on this path).
    def reset_worlds(self, mask: torch.Tensor) -> None:
        _BigMixin.reset_worlds(self, mask)
        if getattr(self, "_ready", False):
            self.hp[0] = self._max_hp[0]
            self.shield[0] = self._max_sh[0]
            self.last_hit[0] = -10**6
            self._focus_ring[0] = 0.0
    # endregion FUNC_reset_worlds

    # region FUNC__resolve_hits
    ## @purpose world12's hit resolution on the compacted alive bullets (order preserved, so every argmax over
    ## bullets and agents picks the same element): (alive x agents) instead of (24*A x A).
    ## A bullet can only hit an agent closer than hit radius (<= HIT_CELL_PX); such a pair lies in neighbouring
    ## cells of a HIT_CELL_PX grid, so only bullets in the 3x3 cells around a living agent are passed on — a
    ## conservative superset, order preserved: the result is the same as over all alive bullets.
    def _resolve_hits(self, dmg_dealt, dmg_taken, kills, deaths) -> None:
        ai = self.b_alive[0].nonzero().flatten()
        if ai.numel() == 0:
            return
        if ai.numel() > HIT_PREFILTER_MIN:
            cell = max(HIT_CELL_PX, float(self._hit_r.max()))
            gw, gh = int(self.map_arena[0, 0] // cell) + 3, int(self.map_arena[0, 1] // cell) + 3
            occ = torch.zeros(gh, gw, dtype=torch.bool, device=self.device)
            ap = self.pos[0, self.alive[0]]
            acx, acy = (ap[:, 0] / cell).long() + 1, (ap[:, 1] / cell).long() + 1
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    occ[(acy + dy).clamp(0, gh - 1), (acx + dx).clamp(0, gw - 1)] = True
            bp = self.b_pos[0, ai]
            bcx = (bp[:, 0] / cell).long().add(1).clamp(0, gw - 1)
            bcy = (bp[:, 1] / cell).long().add(1).clamp(0, gh - 1)
            ai = ai[occ[bcy, bcx]]
            if ai.numel() == 0:
                return
        full = (self.b_pos, self.b_owner, self.b_alive, self.b_dmg)
        self.b_pos, self.b_owner, self.b_alive, self.b_dmg = full[0][:, ai], full[1][:, ai], full[2][:, ai], full[3][:, ai]
        try:
            self._wm12.World12._resolve_hits(self, dmg_dealt, dmg_taken, kills, deaths)
            sub_alive = self.b_alive
        finally:
            self.b_pos, self.b_owner, self.b_alive, self.b_dmg = full
        self.b_alive = full[2].clone()
        self.b_alive[:, ai] = sub_alive
    # endregion FUNC__resolve_hits

    # region FUNC__visibility
    ## @purpose world12's exact tile-crossing line of sight within each viewer's vision, for the 9 nearest alive
    ## allies and the KE_CAP nearest alive enemies in vision; every other pair reads as not visible.
    ## @rationale A pair is tested as (lower index -> higher index), the orientation world12 uses for its
    ## unordered pairs, so the float path of every crossing is the same and the 5v5 result is bit-exact.
    ## @io idx (ignored, one world) -> (1, A, A) bool
    def _visibility(self, idx: torch.Tensor | None) -> torch.Tensor:
        pos, alive = self.pos[0], self.alive[0]
        a_n, dev, T = self.A, self.device, self.T
        vision = self._vision_a[0]
        rel = pos.unsqueeze(0) - pos.unsqueeze(1)                                  # [i, j] = pos_j - pos_i
        dist = rel.pow(2).sum(-1).sqrt()
        team = self._agent_team
        enemy = team.view(-1, 1) != team.view(1, -1)
        eye = torch.eye(a_n, dtype=torch.bool, device=dev)
        inf = torch.full_like(dist, float("inf"))
        ke, ka = min(T, KE_CAP), min(T - 1, N_ENTITY_SLOTS)
        e_key = torch.where(enemy & alive.view(1, -1) & (dist <= vision.view(-1, 1)), dist, inf)
        e_d, e_i = e_key.topk(ke, dim=1, largest=False)
        cands, ok = [e_i], [torch.isfinite(e_d)]
        if ka > 0:
            a_key = torch.where(~enemy & alive.view(1, -1) & ~eye, dist, inf)
            a_d, a_i = a_key.topk(ka, dim=1, largest=False)
            cands.append(a_i)
            ok.append(torch.isfinite(a_d))
        cand, cand_ok = torch.cat(cands, 1), torch.cat(ok, 1)                      # (A, K)
        me = torch.arange(a_n, device=dev).view(-1, 1).expand_as(cand)
        valid = cand_ok.flatten().nonzero().flatten()        # topk pads with inf entries: no LOS work for those
        lo = torch.minimum(me, cand).flatten()[valid]
        hi = torch.maximum(me, cand).flatten()[valid]
        blocked_v = torch.zeros(valid.numel(), dtype=torch.bool, device=dev)
        maps = self.map_idx[:1]
        # _segment_blocked pads every segment of a call to the call's longest tile span: sort by span so each
        # chunk holds similar lengths (short ally pairs no longer pay for 280-tile enemy pairs). Per-pair result
        # does not depend on the padding.
        ts = self.tile_size
        span = ((pos[lo] - pos[hi]).abs() / ts).amax(-1)
        order = span.argsort()
        for s in range(0, lo.numel(), LOS_CHUNK_PAIRS):
            sl = order[s: s + LOS_CHUNK_PAIRS]
            blocked_v[sl] = self._segment_blocked(pos[lo[sl]].unsqueeze(0), pos[hi[sl]].unsqueeze(0), maps)[0]
        blocked = torch.ones(cand.numel(), dtype=torch.bool, device=dev)
        blocked[valid] = blocked_v
        blocked = blocked.view_as(cand)
        d_c = dist.gather(1, cand)
        vis_c = cand_ok & ~blocked & (d_c <= vision.view(-1, 1))
        # OR-combine (world_big lesson): topk pads with inf entries whose index can repeat one of the other list
        vis = torch.zeros(a_n, a_n, dtype=torch.uint8, device=dev)
        vis.scatter_reduce_(1, cand, vis_c.to(torch.uint8), reduce="amax")
        return vis.bool().unsqueeze(0)
    # endregion FUNC__visibility

    def global_state(self) -> torch.Tensor:
        # No critic in a show match; world12's state layout is sized for 10 agents and 7 CPs.
        return torch.zeros(1, 2, 1, device=self.device)

    # region FUNC__build_obs
    ## @purpose world12's observation for one large world. Differences from world12._build_obs (the same as
    ## world_big's against world11): rule normalisations from this battle, capped diagonal, 7 nearest CPs
    ## (index order), 5 nearest eligible last-seen enemies (index order, with their traits and max HP), top-8
    ## hostile bullets over the alive bullets. At team_size 5 on a 7-CP map every choice degenerates to world12's.
    ## @io (idx ignored, vis (1, A, A)) -> (1, A, OBS_SIZE12)
    ## @complexity 10
    def _build_obs(self, idx: torch.Tensor | None, vis: torch.Tensor) -> torch.Tensor:
        wm, W = self._wm, self._wm12
        TS, a = self.T, self.A
        pos, vel, angle = self.pos, self.vel, self.angle
        hp, shield, alive = self.hp, self.shield, self.alive
        maps = self.map_idx
        max_hp, max_sh = self._max_hp, self._max_sh
        ltr = self._log_traits                                                    # (1, A, K)
        spd_a, range_a = self._b_speed_a, self._range_a
        n, dev = 1, self.device
        seg = W.OBS_LAYOUT12
        seg = {s["name"]: s for s in seg}
        obs = torch.zeros(n, a, OBS_SIZE12, device=dev)
        arena = self.map_arena[maps].unsqueeze(1)
        diag = torch.full((n, 1, 1), self.norm_diag, device=dev)
        team = self._agent_team.view(1, a).expand(n, a)
        is_blue = (team == 0)
        facing = torch.stack([angle.cos(), angle.sin()], dim=-1)
        pm1 = lambda cond: torch.where(cond, 1.0, -1.0)                          # noqa: E731
        vision = self.rules.vision
        stw, lives_max, max_frames = wm.SCORE_TO_WIN, wm.TEAM_LIVES, wm.MAX_FRAMES
        K = W.N_TRAITS

        # ---- core: world11's 29, then own traits, readiness, local force ratio, respawn timing ----
        core = torch.zeros(n, a, W.CORE_SIZE12, device=dev)
        core[..., 0] = pos[..., 0] / arena[..., 0] * 2 - 1
        core[..., 1] = pos[..., 1] / arena[..., 1] * 2 - 1
        core[..., 2] = torch.where(alive, hp / max_hp * 2 - 1, torch.full_like(hp, -1.0))
        core[..., 3] = torch.where(alive, shield / max_sh * 2 - 1, torch.full_like(shield, -1.0))
        core[..., 4] = pm1(alive)
        core[..., 5] = self.respawn_timer.float() / W11.RESPAWN_FRAMES * 2 - 1
        core[..., 6] = self.cooldown.float() / self._cd_frames.float() * 2 - 1
        core[..., 7:9] = facing
        core[..., 9:11] = (vel / W11.VEL_NORM).clamp(-1.0, 1.0)
        core[..., 11] = self.dash_cd.float() / self._dash_cd_a.float() * 2 - 1
        core[..., 12] = self.dash_left.float() / W11.DASH_FRAMES * 2 - 1
        core[..., 13:21] = self._cast_rays(pos, maps) * 2 - 1
        score = self.score
        my_score = torch.where(is_blue, score[:, :1], score[:, 1:])
        en_score = torch.where(is_blue, score[:, 1:], score[:, :1])
        core[..., 21] = (my_score / stw).clamp(max=1.0) * 2 - 1
        core[..., 22] = (en_score / stw).clamp(max=1.0) * 2 - 1
        alive_b = alive[:, :TS].sum(1, keepdim=True).float()
        alive_r = alive[:, TS:].sum(1, keepdim=True).float()
        core[..., 23] = torch.where(is_blue, alive_b, alive_r) / TS * 2 - 1
        core[..., 24] = torch.where(is_blue, alive_r, alive_b) / TS * 2 - 1
        lives = self.lives.float()
        core[..., 25] = torch.where(is_blue, lives[:, :1], lives[:, 1:]) / lives_max * 2 - 1
        core[..., 26] = torch.where(is_blue, lives[:, 1:], lives[:, :1]) / lives_max * 2 - 1
        core[..., 27] = (self.frame.float() / max_frames).clamp(max=1.0).view(n, 1) * 2 - 1
        core[..., 28] = pm1(self.map_compact[maps]).view(n, 1).expand(n, a)
        n_old = W11.CORE_SIZE
        core[..., n_old: n_old + K] = ltr
        c = n_old + K
        core[..., c] = pm1(alive & (self.cooldown <= W11.ACTION_REPEAT))
        core[..., c + 1] = pm1(alive & (self.dash_cd <= 0) & (self.dash_left <= 0))
        rel = pos.unsqueeze(1) - pos.unsqueeze(2)
        dist = rel.pow(2).sum(-1).sqrt()
        same_team = team.unsqueeze(2) == team.unsqueeze(1)
        is_self = torch.eye(a, dtype=torch.bool, device=dev).view(1, a, a)
        other_alive = alive.unsqueeze(1).expand(n, a, a)
        ally = other_alive & same_team & ~is_self
        foe = other_alive & ~same_team
        cnt = lambda m: m.sum(2).float() / TS * 2 - 1                            # noqa: E731
        core[..., c + 2] = cnt(ally & (dist <= W.FORCE_NEAR))
        core[..., c + 3] = cnt(foe & (dist <= W.FORCE_NEAR))
        core[..., c + 4] = cnt(ally & (dist <= W.FORCE_FAR))
        core[..., c + 5] = cnt(foe & (dist <= W.FORCE_FAR))
        core[..., c + 6] = cnt(foe & vis)
        wv = self.rules.respawn_wave_frames
        horizon = float(W11.RESPAWN_FRAMES + max(wv, 0))
        core[..., c + 7] = (self._respawn_eta(None).float() / horizon).clamp(max=1.0) * 2 - 1
        if wv > 0:
            f = self.frame.view(n, 1).expand(n, a)
            core[..., c + 8] = ((wv - f % wv) % wv).float() / wv * 2 - 1
        else:
            core[..., c + 8] = -1.0
        obs[..., :W.CORE_SIZE12] = core

        # ---- entity slots (world12 verbatim over the full A x A matrices) ----
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
        f_ = facing.unsqueeze(2)
        cross = f_[..., 0] * g_rel[..., 1] - f_[..., 1] * g_rel[..., 0]
        along = (f_ * g_rel).sum(-1)
        lateral = torch.where(along > 0, (cross / W.LATERAL_NORM).clamp(-1, 1), torch.where(cross >= 0, ones, -ones))
        ent = torch.cat([
            torch.stack([
                ones,
                torch.where(g_same, torch.full_like(g_dist, W11.ETYPE_ALLY), torch.full_like(g_dist, W11.ETYPE_ENEMY)),
                (g_rel[..., 0] / W.NEAR_RANGE).clamp(-1, 1),
                (g_rel[..., 1] / W.NEAR_RANGE).clamp(-1, 1),
                (g_dist / W.NEAR_RANGE).clamp(max=1.0) * 2 - 1,
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
                (g_dist / spd_a.unsqueeze(2) / W.FLIGHT_NORM_FRAMES).clamp(max=1.0) * 2 - 1,
                pm1(g_dist <= range_a.unsqueeze(2)),
                pm1(g_dist <= g(range_a)),
            ], dim=-1),
        ], dim=-1) * keep
        s = seg["entities"]
        obs[..., s["start"]: s["start"] + k_ent * W.ENTITY_SIZE12] = ent.reshape(n, a, -1)

        # ---- bullet slots: nearest hostile among the alive bullets ----
        ab = self.b_alive[0].nonzero().flatten()
        if ab.numel():
            b_pos, b_vel = self.b_pos[:, ab], self.b_vel[:, ab]
            nb = int(ab.numel())
            b_rel = b_pos.unsqueeze(1) - pos.unsqueeze(2)                         # (1, A, nb, 2)
            b_dist = b_rel.pow(2).sum(-1).sqrt()
            b_team = self._agent_team[self.b_owner[:, ab].long()].unsqueeze(1)
            hostile = b_team != team.unsqueeze(2)
            kb = min(N_BULLET_SLOTS, nb)
            key = torch.where(hostile, b_dist, torch.full_like(b_dist, float("inf")))
            if nb > BULLET_TOPK_MIN:
                b_order = key.topk(kb, dim=2, largest=False).indices
            else:
                b_order = key.argsort(dim=2)[:, :, :kb]                            # world12's own ordering
            b_keep = hostile.gather(2, b_order).float().unsqueeze(-1)
            idx2 = b_order.unsqueeze(-1).expand(-1, -1, -1, 2)
            g_brel = b_rel.gather(2, idx2)
            g_bvel = b_vel.unsqueeze(1).expand(n, a, nb, 2).gather(2, idx2)
            gb = lambda t: t[:, ab].unsqueeze(1).expand(n, a, nb).gather(2, b_order)   # noqa: E731
            g_bspd, g_bdmg = gb(self.b_speed), gb(self.b_dmg)
            g_left = (gb(self.b_life) - gb(self.b_age)).clamp(min=0).float()
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
                (g_left * g_bspd / W.REACH_NORM).clamp(max=1.0) * 2 - 1,
            ], dim=-1) * b_keep
            s = seg["bullets"]
            obs[..., s["start"]: s["start"] + kb * W.BULLET_SIZE12] = bul.reshape(n, a, -1)

        # ---- control point slots: the 7 nearest, in CP index order ----
        c_all = self.n_cps
        cp_xy_all = self.map_cp_xy[maps]
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
        blue_near = near[:, :, :TS].any(2).unsqueeze(1).expand(n, a, c_all).gather(2, sel)
        red_near = near[:, :, TS:].any(2).unsqueeze(1).expand(n, a, c_all).gather(2, sel)
        mine_near = torch.where(is_blue.unsqueeze(2), blue_near, red_near)
        foe_near = torch.where(is_blue.unsqueeze(2), red_near, blue_near)
        tsz = self.tile_size
        row = (pos[..., 1] / tsz).long().clamp(0, self.map_grid.shape[1] - 1)
        col = (pos[..., 0] / tsz).long().clamp(0, self.map_grid.shape[2] - 1)
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
        obs[..., s["start"]: s["start"] + kc * s["size"]] = cps.reshape(n, a, -1)

        # ---- last-seen enemy memory (team-shared); only enemies I do not see now; the 5 nearest ----
        mem_of = lambda t: torch.cat([t[:, 0:1].expand(n, TS, *t.shape[2:]), t[:, 1:2].expand(n, TS, *t.shape[2:])], dim=1)  # noqa: E731
        m_pos, m_vel = mem_of(self.mem_pos), mem_of(self.mem_vel)                 # (1, A, T, 2)
        m_hp, m_age, m_val = mem_of(self.mem_hp), mem_of(self.mem_age), mem_of(self.mem_valid)
        enemy_idx = torch.cat([torch.arange(TS, 2 * TS), torch.arange(0, TS)]).to(dev).view(2, TS)[self._agent_team]   # (A, T)
        i_see = vis.gather(2, enemy_idx.view(1, a, TS).expand(n, -1, -1))
        eligible = m_val & ~i_see
        l_rel_all = m_pos - pos.unsqueeze(2)
        l_dist_all = l_rel_all.pow(2).sum(-1).sqrt()
        if TS <= N_LASTSEEN_SLOTS:
            lsel = torch.arange(TS, device=dev).view(1, 1, -1).expand(n, a, -1)
        else:
            key = torch.where(eligible, l_dist_all, torch.full_like(l_dist_all, float("inf")))
            _, lk = key.topk(N_LASTSEEN_SLOTS, dim=2, largest=False)
            lsel = lk.sort(dim=2).values
        kl = lsel.shape[2]
        gl = lambda t: t.gather(2, lsel) if t.dim() == 3 else t.gather(2, lsel.unsqueeze(-1).expand(-1, -1, -1, t.shape[3]))  # noqa: E731
        show = gl(eligible).float().unsqueeze(-1)
        l_rel, l_dist = gl(l_rel_all), gl(l_dist_all)
        lv, lh, la = gl(m_vel), gl(m_hp), gl(m_age)
        en_sel = enemy_idx.view(1, a, TS).gather(2, lsel)                          # (1, A, kl) agent ids
        en_max_hp = max_hp.gather(1, en_sel.view(n, -1)).view(n, a, kl)
        en_ltr = ltr.gather(1, en_sel.view(n, -1, 1).expand(-1, -1, K)).view(n, a, kl, K)
        lones = torch.ones_like(l_dist)
        ls = torch.cat([
            torch.stack([
                lones,
                l_rel[..., 0] / diag.view(n, 1, 1),
                l_rel[..., 1] / diag.view(n, 1, 1),
                (l_dist / diag.view(n, 1, 1)).clamp(max=1.0) * 2 - 1,
                la.float() / W11.LASTSEEN_HORIZON_FRAMES * 2 - 1,
                torch.where(la == 0, lones, -lones),
                (lv[..., 0] / W11.VEL_NORM).clamp(-1, 1),
                (lv[..., 1] / W11.VEL_NORM).clamp(-1, 1),
                lh / en_max_hp * 2 - 1,
            ], dim=-1),
            en_ltr,
        ], dim=-1) * show
        s = seg["lastseen"]
        obs[..., s["start"]: s["start"] + kl * W.LASTSEEN_SIZE12] = ls.reshape(n, a, -1)

        # ---- egocentric grid (unchanged) ----
        G, GS = W11.GRID_RADIUS, W11.GRID_SIDE
        gr = (row + G).view(n, a, 1, 1) + self._grid_dr.view(1, 1, GS, GS)
        gc = (col + G).view(n, a, 1, 1) + self._grid_dc.view(1, 1, GS, GS)
        gm = maps.view(n, 1, 1, 1).expand(n, a, GS, GS)
        walls = self.map_grid_pad[gm, gr, gc].float()
        zone = self.map_zone_pad[gm, gr, gc]
        z_owner = self.cp_owner.gather(1, zone.clamp(min=0).view(n, -1)).view(n, a, GS, GS)
        my_t = (team + 1).view(n, a, 1, 1)
        z_val = torch.where(z_owner == my_t, 1.0, torch.where(z_owner == 0, 0.5, -1.0))
        z_val = torch.where(zone >= 0, z_val, torch.zeros_like(z_val))
        s = seg["grid"]
        obs[..., s["start"]:] = torch.stack([walls, z_val], dim=2).reshape(n, a, -1)
        return obs.clamp(-1.0, 1.0)
    # endregion FUNC__build_obs

    # region FUNC_snapshot
    ## @purpose world_big's columnar snapshot (bullet ids) plus each agent's body radius and max HP.
    def snapshot(self, world_idx: int = 0, columnar: bool = True) -> dict:
        snap = _BigMixin.snapshot(self, world_idx, columnar)
        if columnar:
            snap["agents"]["r"] = self._radius_a[0].cpu().numpy()
            snap["agents"]["max_hp"] = self._max_hp[0].cpu().numpy()
        return snap
    # endregion FUNC_snapshot
# endregion CLASS__Big12Mixin


# region FUNC_WorldBig12
## @purpose Build the world for one map: world11 and world12 clones with this battle's rules, plus the mixins.
## @io (map, device, seed, record) -> world instance (a subclass of the clone's World12)
def WorldBig12(big_map: Any, device: str = "cpu", seed: int = 0, record: bool = True,
               reward_preset: str = W11.DEFAULT_REWARD_PRESET):    # noqa: N802 — used like a class
    rules = battle_rules(big_map)
    mod11 = _world11_clone(rules["team_size"], len(big_map.cp_positions), rules)
    mod12 = _world12_clone(mod11)
    cls = type("WorldBig12", (_Big12Mixin, _BigMixin, mod12.World12), {"_wm": mod11, "_wm12": mod12})
    return cls(big_map, device=device, seed=seed, record=record, reward_preset=reward_preset)
# endregion FUNC_WorldBig12
