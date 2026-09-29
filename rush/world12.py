from __future__ import annotations

# region MODULE_CONTRACT [DOMAIN(9): RL; CONCEPT(10): BatchedWorld12; TECH(9): torch]
## @modulecontract
## @purpose Experiment-12 world: world11 plus longer range (bullet 36 px/frame, range = vision = Rules12.range_px,
## default 2400 px), a 7-way turn head (+-0.5 deg fine aim), per-agent body traits read by the physics, action
## masks for the shot and dash cooldowns, friendly fire with a mirrored penalty, an observation that extends
## every world11 token AT ITS END, and per-agent reward weights (forks with different rewards share a world).
## @scope Everything World11 does; no policy, no training.
## @input Actions (N, 10, 4): move 0..8, shoot 0..1, turn 0..6, dash 0..1
## @output StepOut (obs (N, 10, OBS_SIZE12)); info adds action_mask, r_team_agent, tau_mult, shots_dropped
## @links USES_API(9): torch; LINKS_TO: world11 (base class, frozen), docs/EXP12_CONTRACT.md,
## docs/EXP12_NOTES_world.md
## @invariants
## - OBS_LAYOUT12 lists world11's segments in world11's order; every token keeps world11's features in its
##   first n_old columns with world11's meaning and normalisation; new features follow them
## - Rules12.legacy() with baseline traits and baseline reward weights: the n_old columns of every token,
##   rewards, termination and all state equal World11 bit for bit (tests/test_world12.py)
## - Baseline traits (all 1.0) with default rules are the default-rules world exactly (traits multiply)
## - Line of sight with exact_los=True has no tunnelling: every tile the segment passes through is tested
## - Bullets never skip a 1-tile wall or a player: substeps keep a bullet's per-substep travel <= 20 px
## @rationale
## Q: Why subclass World11 instead of executing a clone of its source like world_big?
## A: world_big needed the SAME rules with different team sizes. World12 changes rules per agent (speed,
## A: radius, cooldowns, damage, range), which module constants cannot express. The physics methods that
## A: read those constants are overridden here with per-agent tensors; everything else (maps, flow fields,
## A: control points, memory, rays, reset bookkeeping) is inherited unchanged. The override is written so
## A: that trait 1.0 and legacy rules give the same float arithmetic as world11 — the legacy test proves it.
## Q: Why keep the old entity columns normalised by 1400 px instead of the new vision?
## A: The exp11b network learned near-range combat on those columns. Rescaling them would erase the learned
## A: resolution (a 15 px aim error becomes 0.002 at 8640). The old columns keep 1400 (clamped), the new far
## A: columns carry d / vision (Rules12.vision = range_px) — near-range skill survives the graft.
## Q: Why did the range drop from 8640 to 2400 px (v0.3.0)?
## A: The user saw 500v500 soldiers hitting from 1400-2600 px on average — "shooting across the whole map".
## A: 2400 px = 80 tiles (~1.7x world11's 1400 vision; ~1/9 of Warfront500's width). Populations trained at
## A: 8640 continue under 2400: the far columns (d / vision, flight time, in-range flags) rescale by 3.6x —
## A: a feature-scale shift the fine-tune absorbs; the near columns (the grafted skill) do not change.
## Q: How does friendly fire keep the legacy world bit-exact?
## A: Rules12.friendly_fire gates every change: off, the hit mask is world11's "hostile", the enemy credit
## A: equals the full credit and the team-kill mask is all False; legacy_rules() has it off.
## @changes
## LAST_CHANGE: [v0.3.0] Range 2400 px (Rules12.range_px; vision_px 0 = equals range), friendly fire
## (Rules12.friendly_fire / ff_coef): allies hittable, ff damage and team kills penalised with the agent's own
## damage_dealt / kill weights, left out of kills/assists/engagement stats, tw ff_damage / teamkills, aim
## label = first agent on the line is an enemy; rules_overrides kwarg for trainer flags.
## PREV_CHANGE: [v0.1.0] Initial experiment-12 world.
## @modulemap
## CONST 10[Observation layout with per-segment n_old] => OBS_LAYOUT12
## CLASS 9[Rule constants of a world] => Rules12
## CLASS 10[World11 with range, fine aim, traits, masks, weights] => World12
## @usecases
## - vec12: w = World12(n, device); w.set_traits(t); w.set_reward_weights(r); out = w.step(a); out.info["action_mask"]
## - model12: graft by OBS_LAYOUT12[*]["n_old"] / old_start (new columns zero-init)
def _module_contract():
    pass
# endregion MODULE_CONTRACT
# GREP_SUMMARY: world12, experiment 12, traits, action mask, fine aim, long range, exact line of sight, obs layout 12, reward weights

import dataclasses
import logging
import math
from dataclasses import dataclass

import torch

from rush import world11 as W11
from rush.world11 import (
    N_AGENTS,
    N_ENTITY_SLOTS,
    N_BULLET_SLOTS,
    N_CP_SLOTS,
    N_LASTSEEN_SLOTS,
    OBS_LAYOUT,
    OBS_SIZE,
    STATE_SIZE,
    World11,
)

logger = logging.getLogger(__name__)

# region BLOCK_CONSTANTS_TRAITS
TRAITS12: list[str] = ["fire_rate", "bullet_range", "bullet_speed", "move_speed",
                       "body_radius", "max_hp", "damage", "dash_cooldown"]
N_TRAITS: int = len(TRAITS12)
TRAIT_BOUNDS: dict[str, tuple[float, float]] = {
    "fire_rate": (0.5, 2.0), "bullet_range": (0.5, 2.0), "bullet_speed": (0.5, 2.0), "move_speed": (0.5, 2.0),
    # 1.25 x 12 px = 15 px = half a tile: a bigger box does not fit a spawn tile and gets stuck on walls
    "body_radius": (0.5, 1.25), "max_hp": (0.5, 2.0), "damage": (0.5, 2.0), "dash_cooldown": (0.5, 2.0),
}
REWARD_W12: list[str] = ["kill", "death", "damage_dealt", "damage_taken", "capture", "score_delta", "win", "team_spirit"]
N_REWARD_W: int = len(REWARD_W12)
# endregion BLOCK_CONSTANTS_TRAITS

# region BLOCK_CONSTANTS_ACTIONS
N_TURN12: int = 7          # 0..4 as world11, 5 = -fine2, 6 = +fine2
TURN_FINE2_DEG: float = 0.5
HEAD_SIZES12: list[int] = [W11.N_MOVES, W11.N_SHOOT, N_TURN12, W11.N_ABILITY]
MASK_SIZE12: int = sum(HEAD_SIZES12)
# endregion BLOCK_CONSTANTS_ACTIONS

# region BLOCK_CONSTANTS_OBS12
NEAR_RANGE: float = W11.ENTITY_RANGE      # 1400: world11's entity normalisation, kept for the old columns
LATERAL_NORM: float = 100.0               # signed miss distance of my barrel line / this
FLIGHT_NORM_FRAMES: float = 60.0          # bullet flight time to the entity / this
REACH_NORM: float = 2000.0                # remaining reach of a bullet / this

FORCE_NEAR: float = 600.0                # local force ratio radii (teamwork, contract W.9)
FORCE_FAR: float = 1500.0
ASSIST_WINDOW_FRAMES: int = 120           # damage within this window before a death earns an assist
FOCUS_WINDOW_DECISIONS: int = 15          # 60 frames of team damage for the focus index
LONELY_RADIUS: float = 600.0              # a death with no living ally this close is "lonely"

CORE_NEW: list[str] = [f"trait_{t}" for t in TRAITS12] + [
    "shoot_ready", "dash_ready", "allies_600", "enemies_600", "allies_1500", "enemies_1500", "enemies_visible",
    "respawn_eta", "wave_in"]
ENTITY_NEW: list[str] = [f"trait_{t}" for t in TRAITS12] + [
    "far_rx", "far_ry", "far_dist", "barrel_lateral", "flight_time", "in_my_range", "me_in_their_range"]
BULLET_NEW: list[str] = ["log_speed", "log_damage", "reach_left"]
CP_NEW: list[str] = []
LASTSEEN_NEW: list[str] = [f"trait_{t}" for t in TRAITS12]
_NEW = {"core": CORE_NEW, "entities": ENTITY_NEW, "bullets": BULLET_NEW, "cps": CP_NEW, "lastseen": LASTSEEN_NEW}


def _layout12() -> list[dict]:
    out, start = [], 0
    for s in OBS_LAYOUT:
        d = dict(s)
        d["old_start"] = s["start"]
        d["n_old"] = s["size"]
        if s["kind"] != "grid":
            d["size"] = s["size"] + len(_NEW[s["name"]])
            d["new_features"] = list(_NEW[s["name"]])
            flat = d["count"] * d["size"]
        else:
            d["new_features"] = []
            flat = s["size"][0] * s["size"][1] * s["size"][2]
        d["start"] = start
        start += flat
        out.append(d)
    return out


OBS_LAYOUT12: list[dict] = _layout12()
_SEG12 = {s["name"]: s for s in OBS_LAYOUT12}
OBS_SIZE12: int = sum((s["count"] * s["size"]) if s["kind"] != "grid" else s["size"][0] * s["size"][1] * s["size"][2]
                      for s in OBS_LAYOUT12)
CORE_SIZE12: int = _SEG12["core"]["size"]
ENTITY_SIZE12: int = _SEG12["entities"]["size"]
BULLET_SIZE12: int = _SEG12["bullets"]["size"]
LASTSEEN_SIZE12: int = _SEG12["lastseen"]["size"]

# Global state: world11's 169 values, then the traits (log) of all ten agents, own team first.
STATE_SIZE12: int = STATE_SIZE + N_AGENTS * N_TRAITS
STATE_LAYOUT12: dict = {"n_old": STATE_SIZE, "size": STATE_SIZE12, "new": "log traits (10 agents x 8, own team first)"}


def old_to_new_index() -> torch.Tensor:
    """(OBS_SIZE,) long: the column of OBS_SIZE12 that holds each world11 observation column."""
    idx = []
    for s in OBS_LAYOUT12:
        if s["kind"] == "grid":
            k = s["size"][0] * s["size"][1] * s["size"][2]
            idx.extend(range(s["start"], s["start"] + k))
            continue
        for slot in range(s["count"]):
            base = s["start"] + slot * s["size"]
            idx.extend(range(base, base + s["n_old"]))
    out = torch.tensor(idx, dtype=torch.long)
    assert out.numel() == OBS_SIZE
    return out


def embed_obs11(obs11: torch.Tensor) -> torch.Tensor:
    """World11 observation -> OBS_SIZE12 with the new columns zero (the graft's step-0 input)."""
    out = torch.zeros(*obs11.shape[:-1], OBS_SIZE12, dtype=obs11.dtype, device=obs11.device)
    out[..., old_to_new_index().to(obs11.device)] = obs11
    return out
# endregion BLOCK_CONSTANTS_OBS12


# region CLASS_Rules12
## @purpose Rule constants of a World12. default(): experiment 12. legacy(): world11's rules and methods.
@dataclass(frozen=True)
class Rules12:
    bullet_speed: float = 36.0            # px / frame
    range_px: float = 2400.0              # bullet range at trait 1.0 (v0.3: 8640 shot across whole maps)
    vision_px: float = 0.0                # 0 = vision equals the range (the exp12 rule); legacy: world11's 1400
    bullet_slots: int = 256
    fine_aim: bool = True                 # turn indices 5/6 legal
    exact_los: bool = True                # tile-crossing line of sight (False: world11's 96 samples)
    swept_bullets: bool = True            # substeps so a bullet never skips a wall or a player
    max_substep_px: float = 20.0
    respawn_wave_frames: int = 540        # the dead respawn together at the next wave tick (0 = off)
    assist_reward: float = 0.3            # per assist, in the zero-sum combat term (kill stays 1.0)
    friendly_fire: bool = True            # bullets hit allies (never the shooter)
    ff_coef: float = 1.0                  # penalty = ff_coef x (damage_dealt / kill weight) per ally HP / team kill
    legacy: bool = False

    @staticmethod
    def default() -> "Rules12":
        return Rules12()

    @staticmethod
    def legacy_rules() -> "Rules12":
        return Rules12(bullet_speed=float(W11.BULLET_SPEED),
                       range_px=float(W11.BULLET_SPEED * W11.BULLET_LIFETIME_FRAMES),
                       vision_px=float(W11.VISION_RANGE), bullet_slots=int(W11.BULLET_SLOTS), fine_aim=False,
                       exact_los=False, swept_bullets=False, respawn_wave_frames=0, assist_reward=0.0,
                       friendly_fire=False, ff_coef=0.0, legacy=True)

    @property
    def bullet_range(self) -> float:
        return float(self.range_px)

    @property
    def vision(self) -> float:
        return float(self.vision_px) if self.vision_px > 0 else float(self.range_px)

    @property
    def bullet_life_frames(self) -> int:
        return max(1, int(round(self.range_px / self.bullet_speed)))
# endregion CLASS_Rules12


# region CLASS_World12
## @purpose World11 + experiment-12 rules: per-agent traits in the physics, long range with exact line of
## sight, swept bullets, fine aim, action masks, the experiment-12 observation and per-agent reward weights.
## @complexity 10
class World12(World11):
    # Early-exit guards (`if not x.any(): return`) in the frame physics. Each is a host sync; with hundreds of worlds
    # they almost never fire, and the guarded code is an exact no-op when x is empty. False (set by experiment 14's
    # worlds) skips the syncs; results are bit-identical (tests/test_perf14.py). Guards that decide whether the CPU
    # generator is drawn from (_respawn) always stay.
    sync_guards: bool = True

    def __init__(self, n_worlds: int, device: str = "cpu", seed: int = 0, rules: Rules12 | None = None,
                 rules_overrides: dict | None = None, **kw) -> None:
        rules = rules or Rules12.default()
        if rules_overrides:
            # Trainer flags (--range-px, --friendly-fire, --ff-coef) patch whatever rules the caller built
            # (world13 sizes its bullet ring first); unknown names fail loudly in dataclasses.replace.
            rules = dataclasses.replace(rules, **{k: v for k, v in rules_overrides.items() if v is not None})
        self.rules = rules
        self._ready = False                    # world11's __init__ resets worlds before the trait tables exist
        super().__init__(n_worlds, device=device, seed=seed, **kw)
        n, a, bs, dev = n_worlds, N_AGENTS, self.rules.bullet_slots, self.device
        # Bullet ring sized by the rules (world11 allocated its own 160) plus per-bullet speed, reach, damage.
        self.b_pos, self.b_vel = torch.zeros(n, bs, 2, device=dev), torch.zeros(n, bs, 2, device=dev)
        self.b_owner = torch.zeros(n, bs, dtype=torch.int32, device=dev)
        self.b_age = torch.zeros(n, bs, dtype=torch.int32, device=dev)
        self.b_alive = torch.zeros(n, bs, dtype=torch.bool, device=dev)
        self.b_life = torch.ones(n, bs, dtype=torch.int32, device=dev)
        self.b_speed = torch.full((n, bs), float(self.rules.bullet_speed), device=dev)
        self.b_dmg = torch.full((n, bs), float(W11.BULLET_DAMAGE), device=dev)
        fine = torch.tensor([-math.radians(TURN_FINE2_DEG), math.radians(TURN_FINE2_DEG)], dtype=torch.float32, device=dev)
        self._turn_deltas = torch.cat([self._turn_deltas, fine])
        self._pairs = torch.triu_indices(a, a, 1, device=dev)                   # (2, 45) unordered agent pairs
        self._dropped = torch.zeros(n, dtype=torch.int32, device=dev)
        self._r_team_agent: torch.Tensor | None = None
        # Teamwork bookkeeping: frame of the last damage shooter -> victim, focus window of team damage.
        self.last_hit = torch.full((n, a, a), -10**6, dtype=torch.int32, device=dev)
        self._focus_ring = torch.zeros(n, FOCUS_WINDOW_DECISIONS, 2, W11.TEAM_SIZE, device=dev)
        self._focus_ptr = 0
        self._tw = self._new_tw()
        self.traits = torch.ones(n, a, N_TRAITS, device=dev)
        self._pending_traits = torch.ones(n, a, N_TRAITS, device=dev)
        self._pending_mask = torch.zeros(n, dtype=torch.bool, device=dev)
        self.reward_w = torch.ones(n, a, N_REWARD_W, device=dev)
        self._w_is_one = True
        self._derive()
        self._ready = True
        logger.info(f"[IMP:9][World12.__init__][INIT] worlds={n} obs={OBS_SIZE12} state={STATE_SIZE12} bullet_slots={bs} "
                    f"range={self.rules.bullet_range:.0f}px vision={self.rules.vision:.0f}px life={self.rules.bullet_life_frames}f "
                    f"substeps={self._substeps} exact_los={self.rules.exact_los} fine_aim={self.rules.fine_aim} "
                    f"friendly_fire={self.rules.friendly_fire} ff_coef={self.rules.ff_coef} legacy={self.rules.legacy} [VALUE]")

    # region FUNC_set_traits
    ## @purpose Per-agent trait multipliers (order TRAITS12) for all worlds or the worlds in idx; recomputes the
    ## per-agent physics tables. Bounds are enforced (TRAIT_BOUNDS) — a wrong multiplier fails loudly here.
    def set_traits(self, traits: torch.Tensor, idx: torch.Tensor | None = None) -> None:
        t = traits.to(self.device, torch.float32)
        for k, name in enumerate(TRAITS12):
            lo, hi = TRAIT_BOUNDS[name]
            v = t[..., k]
            if bool((v < lo).any()) or bool((v > hi).any()):
                raise ValueError(f"trait {name} out of bounds [{lo}, {hi}]: min={float(v.min()):.3f} max={float(v.max()):.3f}")
        if idx is None:
            self.traits = t.expand(self.n, N_AGENTS, N_TRAITS).clone()
        else:
            self.traits[idx] = t.expand(int(idx.numel()), N_AGENTS, N_TRAITS)
        self._derive()
        logger.debug(f"[IMP:6][World12.set_traits][EXEC] worlds={'all' if idx is None else int(idx.numel())} substeps={self._substeps} [VALUE]")

    ## @purpose Traits for the NEXT match of the worlds in world_idx. A world that has just been reset (frame 0:
    ## the trainer rebinds opponents after step() auto-reset it) gets them now, HP and shield refilled to the
    ## new maximum; any other world gets them inside its next reset. Call observe_idx() for fresh worlds if the
    ## current observation must show the new traits.
    def set_pending_traits(self, world_idx: torch.Tensor, traits: torch.Tensor) -> None:
        world_idx = torch.as_tensor(world_idx, dtype=torch.long, device=self.device).flatten()
        t = traits.to(self.device, torch.float32).expand(int(world_idx.numel()), N_AGENTS, N_TRAITS)
        fresh = self.frame[world_idx] == 0
        if bool(fresh.any()):
            fi = world_idx[fresh]
            self.set_traits(t[fresh], fi)
            self.hp[fi] = self._max_hp[fi]
            self.shield[fi] = self._max_sh[fi]
        if bool((~fresh).any()):
            self._pending_traits[world_idx[~fresh]] = t[~fresh]
            self._pending_mask[world_idx[~fresh]] = True

    ## @purpose Per-agent reward weights (order REWARD_W12, multipliers of the exp11b preset; team_spirit
    ## multiplies the trainer's tau).
    def set_reward_weights(self, w: torch.Tensor, idx: torch.Tensor | None = None) -> None:
        w = w.to(self.device, torch.float32)
        if bool((w < 0).any()):
            raise ValueError("reward weights must be >= 0 (they change emphasis, not sign)")
        if idx is None:
            self.reward_w = w.expand(self.n, N_AGENTS, N_REWARD_W).clone()
        else:
            self.reward_w[idx] = w.expand(int(idx.numel()), N_AGENTS, N_REWARD_W)
        self._w_is_one = bool((self.reward_w == 1.0).all())

    ## @purpose Physics tables from traits. Every value at trait 1.0 is the rule constant itself, computed in
    ## float32 so that x * 1.0 and x / 1.0 are exact (the legacy test relies on it).
    def _derive(self) -> None:
        r, t = self.rules, self.traits
        fr, br, bsp, ms, rad, mhp, dmg, dcd = t.unbind(-1)
        self._cd_frames = torch.round(W11.SHOOT_COOLDOWN_FRAMES / fr).clamp(min=1).to(torch.int32)
        self._b_speed_a = r.bullet_speed * bsp
        self._range_a = r.bullet_range * br
        self._b_life_a = torch.round(self._range_a / self._b_speed_a).clamp(min=1).to(torch.int32)
        self._vision_a = r.vision * br
        self._speed_a = W11.PLAYER_SPEED * ms
        self._radius_a = W11.PLAYER_RADIUS * rad
        self._hit_r = W11.BULLET_RADIUS + self._radius_a
        self._max_hp = float(W11.PLAYER_HP) * mhp
        self._max_sh = float(W11.SHIELD_MAX) * mhp
        self._dmg_a = float(W11.BULLET_DAMAGE) * dmg
        self._dash_cd_a = torch.round(W11.DASH_COOLDOWN_FRAMES * dcd).clamp(min=1).to(torch.int32)
        self._log_traits = t.log()
        if r.swept_bullets:
            lim = min(r.max_substep_px, 2.0 * float(self._hit_r.min()))
            self._substeps = max(1, math.ceil(float(self._b_speed_a.max()) / lim))
        else:
            self._substeps = 1
    # endregion FUNC_set_traits

    def _reset_idx(self, idx: torch.Tensor, k: int) -> None:
        super()._reset_idx(idx, k)
        if self._ready:
            pend = self._pending_mask[idx]
            if bool(pend.any()):
                pi = idx[pend]
                self.traits[pi] = self._pending_traits[pi]
                self._pending_mask[pi] = False
                self._derive()
            self.hp[idx] = self._max_hp[idx]
            self.shield[idx] = self._max_sh[idx]
            self.last_hit[idx] = -10**6
            self._focus_ring[idx] = 0.0

    ## @purpose Per-decision teamwork counters (per world, per team), summed over the decision's frames.
    def _new_tw(self) -> dict:
        z = lambda: torch.zeros(self.n, 2, device=self.device)                  # noqa: E731
        za = lambda: torch.zeros(self.n, N_AGENTS, device=self.device)          # noqa: E731
        return {"assists": za(), "hits": za(), "hit_dist": za(), "pair_dmg": torch.zeros(self.n, N_AGENTS, N_AGENTS, device=self.device),
                "deaths_lonely": z(), "deaths_team": z(), "kills_adv": z(), "kills_even": z(), "kills_team": z(),
                "ff_dmg": za(), "teamkills": za()}

    ## @purpose Frames until each agent respawns (0 if alive): the first frame k >= max(timer, 1) of the coming
    ## decision whose world frame is a wave tick (waves off: the timer itself).
    def _respawn_eta(self, idx: torch.Tensor | None = None) -> torch.Tensor:
        timer = self._sel(self.respawn_timer, idx).clamp(min=1)
        waiting = self._sel(self.waiting, idx)
        wv = self.rules.respawn_wave_frames
        if wv > 0:
            f = self._sel(self.frame, idx).view(-1, 1)
            timer = timer + (wv - (f + timer) % wv) % wv
        return torch.where(waiting, timer, torch.zeros_like(timer))

    # region FUNC_action_mask
    ## @purpose Legal actions for the NEXT decision, (N, A, 20) bool in head order move|shoot|turn|dash.
    ## @rationale A shot fires within a decision if the cooldown reaches 0 within its ACTION_REPEAT frames
    ## (the cooldown counts down before the trigger check), so shoot is legal at cooldown <= ACTION_REPEAT.
    ## The dash triggers before the frames, so it needs cooldown 0 and no running dash. A dead agent has only
    ## the no-ops (the trainer's contract, EXP12_NOTES_trainer): one that respawns inside the decision stands
    ## still for at most ACTION_REPEAT - 1 frames — the price of one simple rule for every dead agent.
    def action_mask(self) -> torch.Tensor:
        ar = W11.ACTION_REPEAT
        alive = self.alive
        act = alive
        shoot_ok = alive & (self.cooldown <= ar)
        dash_ok = alive & (self.dash_cd <= 0) & (self.dash_left <= 0)
        m = torch.zeros(self.n, N_AGENTS, MASK_SIZE12, dtype=torch.bool, device=self.device)
        m[..., 0] = True
        m[..., 1:9] = act.unsqueeze(-1)
        o = W11.N_MOVES
        m[..., o] = True
        m[..., o + 1] = shoot_ok
        o += W11.N_SHOOT
        m[..., o + 2] = True
        m[..., [o + 0, o + 1, o + 3, o + 4]] = alive.unsqueeze(-1)
        if self.rules.fine_aim:
            m[..., [o + 5, o + 6]] = alive.unsqueeze(-1)
        o += N_TURN12
        m[..., o] = True
        m[..., o + 1] = dash_ok
        return m
    # endregion FUNC_action_mask

    # region FUNC_step
    ## @purpose world11.step with substepped bullets, dropped-shot accounting, per-agent team reward and the
    ## action mask. Copied from world11 (frozen) because the frame loop itself changes; the order of every
    ## physics call is world11's, so legacy rules (1 substep) run the same sequence.
    def step(self, actions: torch.Tensor) -> W11.StepOut:
        n = self.n
        dev = self.device
        dmg_dealt = torch.zeros(n, N_AGENTS, device=dev)
        dmg_taken = torch.zeros(n, N_AGENTS, device=dev)
        kills = torch.zeros(n, N_AGENTS, device=dev)
        deaths = torch.zeros(n, N_AGENTS, device=dev)
        cp_gain = torch.zeros(n, 2, device=dev)
        cap_part = torch.zeros(n, N_AGENTS, device=dev)
        score_before = self.score.clone()
        self._dropped = torch.zeros(n, dtype=torch.int32, device=dev)
        self._tw = self._new_tw()
        if self.record:
            self._events = [[] for _ in range(n)]

        actions = actions.to(dev).long()
        move_a, shoot_a, turn_a, dash_a = actions[..., 0], actions[..., 1], actions[..., 2], actions[..., 3]
        aim_label = self._aim_label(self._vis_cache if self._vis_cache is not None else self._visibility(None))

        self._turn(turn_a)
        self._trigger_dash(dash_a)
        start_pos = self.pos.clone()
        respawned = torch.zeros(n, N_AGENTS, dtype=torch.bool, device=dev)
        sub = self._substeps
        for _ in range(W11.ACTION_REPEAT):
            self.frame += 1
            self.cooldown = (self.cooldown - 1).clamp(min=0)
            self.dash_cd = (self.dash_cd - 1).clamp(min=0)
            self._move(move_a)
            self._shoot(shoot_a)
            for s in range(sub):
                self._advance_bullets_sub(s, sub)
                self._resolve_hits(dmg_dealt, dmg_taken, kills, deaths)
            self._regen_shield()
            respawned |= self._respawn()
            self._update_control_points(cp_gain, cap_part)
            self._score_income()
        vel = (self.pos - start_pos) / W11.ACTION_REPEAT
        self.vel = torch.where((respawned | ~self.alive).unsqueeze(-1), torch.zeros_like(vel), vel)
        tw_metrics = self._teamwork_decision_metrics()

        ts = W11.TEAM_SIZE
        blue_dead = (~self.alive[:, :ts]).all(1) & (self.lives[:, 0] <= 0)
        red_dead = (~self.alive[:, ts:]).all(1) & (self.lives[:, 1] <= 0)
        won = self.score >= W11.SCORE_TO_WIN
        decided = won.any(1) | blue_dead | red_dead
        clock = (self.frame >= W11.MAX_FRAMES) & ~decided
        done = decided | clock
        if self.clock_is_truncation:
            terminated, truncated = decided, clock
        else:
            terminated, truncated = done, torch.zeros_like(done)

        winner = torch.zeros(n, dtype=torch.long, device=dev)
        winner = torch.where(self.score[:, 0] > self.score[:, 1], torch.ones_like(winner), winner)
        winner = torch.where(self.score[:, 1] > self.score[:, 0], torch.full_like(winner, 2), winner)
        winner = torch.where(red_dead, torch.ones_like(winner), winner)
        winner = torch.where(blue_dead, torch.full_like(winner, 2), winner)
        winner = torch.where(blue_dead & red_dead, torch.zeros_like(winner), winner)
        winner = torch.where(done, winner, torch.zeros_like(winner))

        r_indiv, r_team = self._reward(dmg_dealt, dmg_taken, kills, deaths, cp_gain, cap_part, score_before, winner, terminated, clock)
        flags = torch.stack([done, truncated]).cpu().numpy()
        done_cpu = flags[0]
        info = {"winner": winner, "score": self.score.clone(), "frame": self.frame.clone(), "done": done,
                "kills": kills, "captures": cp_gain, "done_cpu": done_cpu, "any_truncated": bool(flags[1].any()),
                "aim_label": aim_label, "damage": dmg_dealt, "r_team_agent": self._r_team_agent,
                "tau_mult": self.reward_w[..., REWARD_W12.index("team_spirit")].clone(), "shots_dropped": self._dropped,
                "assists": self._tw["assists"], "deaths_lonely": self._tw["deaths_lonely"], "deaths_team": self._tw["deaths_team"],
                "kills_adv": self._tw["kills_adv"], "kills_even": self._tw["kills_even"], "kills_team": self._tw["kills_team"],
                "hits": self._tw["hits"], "hit_dist": self._tw["hit_dist"],
                "ff_damage": self._tw["ff_dmg"], "teamkills": self._tw["teamkills"],
                **{k: v for k, v in tw_metrics.items() if not k.startswith("_")}}
        tw = self._tw
        # Per-decision SUMS in the trainer's x_num / x_den protocol (pooled ratio tw_x); (N, 2) per team,
        # (N, 10) per agent.
        info["tw"] = {
            "focus_num": tw_metrics["_focus_num"], "focus_den": tw_metrics["_focus_den"],
            "ally_dist_num": tw_metrics["_ally_num"], "ally_dist_den": tw_metrics["_ally_den"],
            "lonely_deaths_num": tw["deaths_lonely"], "lonely_deaths_den": tw["deaths_team"],
            "kills_adv_num": tw["kills_adv"], "kills_adv_den": tw["kills_team"],
            "kills_even_num": tw["kills_even"], "kills_even_den": tw["kills_team"],
            "engage_dist_num": tw["hit_dist"], "engage_dist_den": tw["hits"],
            "assists_num": tw["assists"].view(n, 2, W11.TEAM_SIZE).sum(-1), "assists_den": tw["kills_team"],
            "kills_adv": tw["kills_adv"], "kills_even": tw["kills_even"],
        }
        if self.rules.friendly_fire:
            # tw_ff_damage = share of the team's damage that hit its own; tw_teamkills = share of the team's
            # kills (enemy kills + team kills) that were team kills.
            per_team = lambda t: t.view(n, 2, W11.TEAM_SIZE).sum(-1)                       # noqa: E731
            ff_t, dealt_t, tk_t = per_team(tw["ff_dmg"]), per_team(info["damage"]), per_team(tw["teamkills"])
            info["tw"].update({"ff_damage_num": ff_t, "ff_damage_den": ff_t + dealt_t,
                               "teamkills_num": tk_t, "teamkills_den": tk_t + tw["kills_team"]})

        vis = self._visibility(None)
        self._update_memory(vis)
        obs = self._build_obs(None, vis)
        final_obs = obs.clone()
        final_state = self.global_state()
        events = self._events if self.record else None
        idx_cpu = torch.as_tensor(done_cpu).nonzero().flatten().numpy()
        if self.record:
            for w in idx_cpu.tolist():
                events[w].append({"type": "match_end", "frame": int(self.frame[w]), "winner": int(winner[w]),
                                  "score": [float(self.score[w, 0]), float(self.score[w, 1])], "truncated": bool(truncated[w])})
        if idx_cpu.size and self.auto_reset:
            idx = torch.as_tensor(idx_cpu, dtype=torch.long, device=dev)
            self._reset_idx(idx, int(idx_cpu.size))
            vis_new = self._visibility(idx)
            self._update_memory(vis_new, idx)
            obs[idx] = self._build_obs(idx, vis_new)
            vis[idx] = vis_new
        self._vis_cache = vis
        info["action_mask"] = self.action_mask()
        return W11.StepOut(obs=obs, r_indiv=r_indiv, r_team=r_team, terminated=terminated, truncated=truncated,
                           final_obs=final_obs, final_state=final_state, info=info, events=events)
    # endregion FUNC_step

    def observe_idx(self, idx: torch.Tensor) -> torch.Tensor:
        """Observation of the worlds in idx (e.g. after set_traits on freshly reset worlds)."""
        return self._build_obs(idx, self._visibility(idx))

    # region FUNC_physics_overrides
    ## @purpose world11's physics methods with the per-agent tables in place of the rule constants.
    def _trigger_dash(self, dash_a: torch.Tensor) -> None:
        go = self.alive & (dash_a == 1) & (self.dash_cd <= 0) & (self.dash_left <= 0)
        self.dash_left = torch.where(go, torch.full_like(self.dash_left, W11.DASH_FRAMES), self.dash_left)
        self.dash_cd = torch.where(go, self._dash_cd_a, self.dash_cd)
        if self.record:
            for w, ag in torch.nonzero(go).tolist():
                self._events[w].append({"type": "dash", "frame": int(self.frame[w]), "agent": ag})

    def _move(self, move_a: torch.Tensor) -> None:
        dashing = (self.dash_left > 0) & self.alive
        unit = self._dirs_unit[move_a]
        facing = torch.stack([self.angle.cos(), self.angle.sin()], dim=-1)
        standing = (move_a == 0).unsqueeze(-1)
        dash_dir = torch.where(standing, facing, unit)
        speed = torch.where(dashing, self._speed_a * W11.DASH_SPEED_MULT, self._speed_a)
        step = torch.where(dashing.unsqueeze(-1), dash_dir, unit) * speed.unsqueeze(-1)
        self.dash_left = (self.dash_left - 1).clamp(min=0)

        rad = self._radius_a
        w = torch.arange(self.n, device=self.device).unsqueeze(1).expand(-1, N_AGENTS)
        arena = self.map_arena[self.map_idx].unsqueeze(1)
        new_x = self.pos[..., 0] + step[..., 0]
        blocked = self._rect_hits_wall(w, new_x, self.pos[..., 1], rad)
        ok = self.alive & ~blocked
        self.pos[..., 0] = torch.where(ok, torch.minimum(torch.maximum(new_x, rad), arena[..., 0] - rad), self.pos[..., 0])
        new_y = self.pos[..., 1] + step[..., 1]
        blocked = self._rect_hits_wall(w, self.pos[..., 0], new_y, rad)
        ok = self.alive & ~blocked
        self.pos[..., 1] = torch.where(ok, torch.minimum(torch.maximum(new_y, rad), arena[..., 1] - rad), self.pos[..., 1])

    def _shoot(self, shoot_a: torch.Tensor) -> None:
        firing = self.alive & (shoot_a == 1) & (self.cooldown <= 0)
        if self.sync_guards and not firing.any():
            return
        bs = self.rules.bullet_slots
        self.cooldown = torch.where(firing, self._cd_frames, self.cooldown)
        muzzle = self._radius_a + float(W11.BULLET_RADIUS + 2)
        cos_a, sin_a = self.angle.cos(), self.angle.sin()
        bx = self.pos[..., 0] + cos_a * muzzle
        by = self.pos[..., 1] + sin_a * muzzle

        free_rank = torch.argsort(self.b_alive.int(), dim=1, stable=True)
        shooter_rank = firing.cumsum(1) - 1
        capacity = (~self.b_alive).sum(1, keepdim=True)
        placeable = firing & (shooter_rank < capacity) & (shooter_rank < bs)
        self._dropped += (firing & ~placeable).sum(1).int()
        slot = free_rank.gather(1, shooter_rank.clamp(min=0, max=bs - 1))
        if not self.sync_guards:
            self._place_bullets_dense(placeable, slot, bx, by, cos_a, sin_a)
            return
        if not placeable.any():
            return
        wi, ai = torch.nonzero(placeable, as_tuple=True)
        si = slot[wi, ai]
        spd = self._b_speed_a[wi, ai]
        self.b_pos[wi, si, 0] = bx[wi, ai]
        self.b_pos[wi, si, 1] = by[wi, ai]
        self.b_vel[wi, si, 0] = cos_a[wi, ai] * spd
        self.b_vel[wi, si, 1] = sin_a[wi, ai] * spd
        self.b_owner[wi, si] = ai.int()
        self.b_age[wi, si] = 0
        self.b_alive[wi, si] = True
        self.b_life[wi, si] = self._b_life_a[wi, ai]
        self.b_speed[wi, si] = spd
        self.b_dmg[wi, si] = self._dmg_a[wi, ai]
        if self.record:
            for w, ag in zip(wi.tolist(), ai.tolist()):
                self._events[w].append({"type": "shot", "frame": int(self.frame[w]), "agent": ag})

    ## @purpose _shoot's bullet placement without nonzero (no host sync): every agent scatters into its slot, the
    ## non-placeable ones into a spare column B that is dropped. Placeable agents of a world have distinct slots
    ## (shooter ranks are distinct), so every real slot gets exactly the value the sparse path writes.
    def _place_bullets_dense(self, placeable, slot, bx, by, cos_a, sin_a) -> None:
        bs = self.rules.bullet_slots
        idx = torch.where(placeable, slot, torch.full_like(slot, bs))                  # (n, A) -> slot or spare
        spd = self._b_speed_a

        def put(dst: torch.Tensor, val: torch.Tensor) -> torch.Tensor:
            ext = torch.cat([dst, dst[:, :1]], 1)                                    # spare column
            ext.scatter_(1, idx, val.to(dst.dtype))
            return ext[:, :bs]
        self.b_pos = torch.stack([put(self.b_pos[..., 0].contiguous(), bx), put(self.b_pos[..., 1].contiguous(), by)], -1)
        self.b_vel = torch.stack([put(self.b_vel[..., 0].contiguous(), cos_a * spd),
                                  put(self.b_vel[..., 1].contiguous(), sin_a * spd)], -1)
        agent = torch.arange(N_AGENTS, device=self.device).view(1, -1).expand_as(slot)
        self.b_owner = put(self.b_owner, agent)
        self.b_age = put(self.b_age, torch.zeros_like(slot))
        self.b_alive = put(self.b_alive, torch.ones_like(placeable))
        self.b_life = put(self.b_life, self._b_life_a)
        self.b_speed = put(self.b_speed, spd)
        self.b_dmg = put(self.b_dmg, self._dmg_a)

    def _advance_bullets(self) -> None:
        self._advance_bullets_sub(0, 1)

    ## @purpose One substep of bullet flight: 1/sub of the frame's displacement, age counted on the first
    ## substep only, wall test at the new position. With sub steps <= min(20 px, hit diameter) a bullet can
    ## neither jump a 30 px wall tile (box 6 px) nor a player.
    def _advance_bullets_sub(self, s: int, sub: int) -> None:
        if self.sync_guards and not self.b_alive.any():
            return
        self.b_pos = self.b_pos + (self.b_vel if sub == 1 else self.b_vel / sub)
        if s == 0:
            self.b_age = self.b_age + 1
        w = torch.arange(self.n, device=self.device).unsqueeze(1).expand(-1, self.rules.bullet_slots)
        wall = self._rect_hits_wall(w, self.b_pos[..., 0], self.b_pos[..., 1], W11.BULLET_RADIUS)
        expired = self.b_age > self.b_life
        self.b_alive = self.b_alive & ~wall & ~expired

    def _resolve_hits(self, dmg_dealt, dmg_taken, kills, deaths) -> None:
        if self.sync_guards and not self.b_alive.any():
            return
        ts = W11.TEAM_SIZE
        d = (self.b_pos.unsqueeze(2) - self.pos.unsqueeze(1)).pow(2).sum(-1).sqrt()
        owner_team = self._agent_team[self.b_owner.long()]
        target_team = self._agent_team.view(1, 1, N_AGENTS)
        same_side = owner_team.unsqueeze(2) == target_team                         # (n, B, A): bullet of the target's team
        if self.rules.friendly_fire:
            # Friendly fire: any agent but the shooter can be hit (the muzzle spawns outside the own hitbox).
            can_hit = self.b_owner.long().unsqueeze(2) != torch.arange(N_AGENTS, device=self.device).view(1, 1, N_AGENTS)
        else:
            can_hit = ~same_side
        hit = (d < self._hit_r.unsqueeze(1)) & can_hit & self.alive.unsqueeze(1) & self.b_alive.unsqueeze(2)
        if self.sync_guards and not hit.any():
            return
        first_target = hit.float().argmax(dim=2)
        bullet_hits = hit.any(2)
        onehot = torch.zeros_like(hit, dtype=torch.float32)
        onehot.scatter_(2, first_target.unsqueeze(2), bullet_hits.float().unsqueeze(2))

        incoming = (onehot * self.b_dmg.unsqueeze(2)).sum(1)
        sh_before, hp_before = self.shield, self.hp
        absorb = torch.minimum(incoming, sh_before)
        hp_dmg = (incoming - absorb).clamp(max=hp_before)
        self.shield = torch.where(self.alive, sh_before - absorb, sh_before)
        self.hp = torch.where(self.alive, hp_before - hp_dmg, hp_before)
        dealt = absorb + hp_dmg
        dmg_taken += dealt
        self.since_dmg = torch.where(incoming > 0, torch.zeros_like(self.since_dmg), self.since_dmg)

        credit = onehot * dealt.unsqueeze(1) / (onehot.sum(1, keepdim=True).clamp(min=1.0))
        shooter = self.b_owner.long()
        # Damage to enemies earns damage_dealt; damage to allies goes to ff_dmg (penalised in _reward). Without
        # friendly fire every credited (bullet, target) pair is hostile, so the enemy credit IS credit (bit-exact).
        enemy_credit = torch.where(same_side, torch.zeros_like(credit), credit)
        dmg_dealt.scatter_add_(1, shooter, enemy_credit.sum(2))
        ally_hit = bullet_hits & same_side.gather(2, first_target.unsqueeze(2)).squeeze(2)          # (n, B)
        if self.rules.friendly_fire:
            self._tw["ff_dmg"].scatter_add_(1, shooter, (credit - enemy_credit).sum(2))
        # Teamwork: damage per (shooter, victim) pair and the frame it last happened.
        pair = torch.zeros(self.n, N_AGENTS, N_AGENTS, device=self.device).scatter_add_(
            1, shooter.unsqueeze(2).expand(-1, -1, N_AGENTS), credit)
        self._tw["pair_dmg"] += pair
        self.last_hit = torch.where(pair > 0, self.frame.view(-1, 1, 1), self.last_hit)
        # Engagement distance: shooter -> target at the moment of the hit, summed per shooter (enemy hits only).
        bh = (bullet_hits & ~ally_hit).float()
        tgt_pos = self.pos.gather(1, first_target.unsqueeze(-1).expand(-1, -1, 2))
        own_pos = self.pos.gather(1, shooter.unsqueeze(-1).expand(-1, -1, 2))
        self._tw["hits"].scatter_add_(1, shooter, bh)
        self._tw["hit_dist"].scatter_add_(1, shooter, bh * (tgt_pos - own_pos).pow(2).sum(-1).sqrt())
        if self.record:
            for w, bi in torch.nonzero(bullet_hits).tolist():
                tgt = int(first_target[w, bi])
                self._events[w].append({"type": "hit", "frame": int(self.frame[w]), "shooter": int(self.b_owner[w, bi]),
                                        "target": tgt, "x": float(self.b_pos[w, bi, 0]), "y": float(self.b_pos[w, bi, 1])})
        self.b_alive = self.b_alive & ~bullet_hits

        died = self.alive & (self.hp <= 0)
        if not self.sync_guards or died.any():
            alive_before = self.alive
            deaths += died.float()
            self.alive = self.alive & ~died
            killer_bullet = (onehot * died.unsqueeze(1).float()).argmax(dim=1)
            killer = self.b_owner.long().gather(1, killer_bullet)
            # A team kill is not a kill: it is counted apart (penalised in _reward) and left out of kill stats.
            tk = died & (self._agent_team[killer] == self._agent_team.view(1, N_AGENTS))
            kills.scatter_add_(1, killer, (died & ~tk).float())
            if self.rules.friendly_fire:
                self._tw["teamkills"].scatter_add_(1, killer, tk.float())
            self._teamwork_on_deaths(died, killer, alive_before, tk)
            if self.record:
                for w, v in torch.nonzero(died).tolist():
                    ev = {"type": "kill", "frame": int(self.frame[w]), "killer": int(killer[w, v]), "victim": v}
                    if bool(tk[w, v]):
                        ev["teamkill"] = True
                    self._events[w].append(ev)
            for team in (0, 1):
                lo, hi = team * ts, (team + 1) * ts
                self.score[:, 1 - team] += died[:, lo:hi].sum(1).float()
                can = self.lives[:, team] > 0
                n_new = (died[:, lo:hi].sum(1) * can).int()
                self.lives[:, team] = (self.lives[:, team] - n_new).clamp(min=0)
            self.waiting = self.waiting | died
            self.respawn_timer = torch.where(died, torch.full_like(self.respawn_timer, W11.RESPAWN_FRAMES), self.respawn_timer)
            self.dash_left = torch.where(died, torch.zeros_like(self.dash_left), self.dash_left)

    # region FUNC__teamwork_on_deaths
    ## @purpose Assists (damage to the victim within ASSIST_WINDOW_FRAMES before its death, killer excluded),
    ## lonely deaths (no living ally within LONELY_RADIUS) and the local balance at every kill (agents of each
    ## side alive within FORCE_NEAR of the victim, the victim counted on its side): advantage >= 2:1, even or
    ## worse <= 1:1. Bookkeeping only — no state the physics reads. Friendly fire: damage to an ally earns no
    ## assist, and a team-killed victim (tk) counts as a death but not as a kill of the other side.
    def _teamwork_on_deaths(self, died: torch.Tensor, killer: torch.Tensor, alive_before: torch.Tensor,
                            tk: torch.Tensor | None = None) -> None:
        ts, tw = W11.TEAM_SIZE, self._tw
        same = (self._agent_team.view(1, -1, 1) == self._agent_team.view(1, 1, -1))
        recent = (self.frame.view(-1, 1, 1) - self.last_hit) <= ASSIST_WINDOW_FRAMES          # [w, shooter, victim]
        is_killer = torch.zeros(self.n, N_AGENTS, N_AGENTS, dtype=torch.bool, device=self.device)
        is_killer.scatter_(1, killer.unsqueeze(1), died.unsqueeze(1))
        assist = recent & died.unsqueeze(1) & ~is_killer & ~same
        tw["assists"] += assist.float().sum(2)
        self.last_hit = torch.where(died.unsqueeze(1), torch.full_like(self.last_hit, -10**6), self.last_hit)
        killed = died if tk is None else died & ~tk

        pd = (self.pos.unsqueeze(1) - self.pos.unsqueeze(2)).pow(2).sum(-1).sqrt()          # [w, v, j]
        eye = torch.eye(N_AGENTS, dtype=torch.bool, device=self.device).view(1, N_AGENTS, N_AGENTS)
        others = alive_before.unsqueeze(1)
        lonely = died & ~(others & same & ~eye & (pd <= LONELY_RADIUS)).any(2)
        own_near = (others & same & (pd <= FORCE_NEAR)).sum(2)                               # victim included
        foe_near = (others & ~same & (pd <= FORCE_NEAR)).sum(2)
        adv = killed & (foe_near >= 2 * own_near)
        even = killed & (foe_near <= own_near)
        for t in (0, 1):
            sl = slice(t * ts, (t + 1) * ts)
            tw["deaths_team"][:, t] += died[:, sl].sum(1).float()
            tw["deaths_lonely"][:, t] += lonely[:, sl].sum(1).float()
            tw["kills_team"][:, 1 - t] += killed[:, sl].sum(1).float()      # victims of team t are kills of 1 - t
            tw["kills_adv"][:, 1 - t] += adv[:, sl].sum(1).float()
            tw["kills_even"][:, 1 - t] += even[:, sl].sum(1).float()
    # endregion FUNC__teamwork_on_deaths

    ## @purpose Per-world team metrics at the end of a decision: focus index over the last 60 frames and the
    ## mean distance from each living agent to its nearest living ally.
    def _teamwork_decision_metrics(self) -> dict:
        ts = W11.TEAM_SIZE
        pair = self._tw["pair_dmg"]
        dmg_b = pair[:, :ts, ts:].sum(1)                                          # team 0 -> red targets
        dmg_r = pair[:, ts:, :ts].sum(1)
        self._focus_ring[:, self._focus_ptr] = torch.stack([dmg_b, dmg_r], 1)
        self._focus_ptr = (self._focus_ptr + 1) % FOCUS_WINDOW_DECISIONS
        win = self._focus_ring.sum(1)                                             # (n, 2, 5)
        tot = win.sum(-1)
        focus_valid = tot > 0
        focus = torch.where(focus_valid, win.max(-1).values / tot.clamp(min=1e-9), torch.full_like(tot, -1.0))
        pd = (self.pos.unsqueeze(1) - self.pos.unsqueeze(2)).pow(2).sum(-1).sqrt()
        same = (self._agent_team.view(1, -1, 1) == self._agent_team.view(1, 1, -1))
        eye = torch.eye(N_AGENTS, dtype=torch.bool, device=self.device).view(1, N_AGENTS, N_AGENTS)
        ok = self.alive.unsqueeze(1) & self.alive.unsqueeze(2) & same & ~eye
        nearest = torch.where(ok, pd, torch.full_like(pd, float("inf"))).min(2).values          # (n, A)
        has = torch.isfinite(nearest)
        near_sum, near_cnt = [], []
        for t in (0, 1):
            sl = slice(t * ts, (t + 1) * ts)
            near_cnt.append(has[:, sl].sum(1).float())
            near_sum.append(torch.where(has[:, sl], nearest[:, sl], torch.zeros_like(nearest[:, sl])).sum(1))
        near_sum, near_cnt = torch.stack(near_sum, 1), torch.stack(near_cnt, 1)
        return {"focus": focus, "focus_valid": focus_valid,
                "nearest_ally": torch.where(near_cnt > 0, near_sum / near_cnt.clamp(min=1), torch.full_like(near_sum, -1.0)),
                "nearest_ally_valid": near_cnt > 0,
                "_focus_num": torch.where(focus_valid, win.max(-1).values, torch.zeros_like(tot)), "_focus_den": tot,
                "_ally_num": near_sum, "_ally_den": near_cnt}

    def _regen_shield(self) -> None:
        self.since_dmg = self.since_dmg + 1
        can = self.alive & (self.since_dmg >= W11.SHIELD_REGEN_DELAY_FRAMES) & (self.shield < self._max_sh)
        if not self.sync_guards or can.any():
            self.shield = torch.where(can, torch.minimum(self.shield + W11.SHIELD_REGEN_PER_FRAME, self._max_sh), self.shield)

    def _respawn(self) -> torch.Tensor:
        ts = W11.TEAM_SIZE
        none = torch.zeros(self.n, N_AGENTS, dtype=torch.bool, device=self.device)
        if not self.waiting.any():
            return none
        self.respawn_timer = torch.where(self.waiting, self.respawn_timer - 1, self.respawn_timer)
        ready = self.waiting & (self.respawn_timer <= 0)
        wv = self.rules.respawn_wave_frames
        if wv > 0:
            # Wave respawn: a dead agent whose timer ran out waits for its world's next wave tick, so the
            # dead come back together instead of trickling in one by one.
            ready = ready & ((self.frame % wv) == 0).view(-1, 1)
        if not ready.any():
            return none
        m = self.map_idx
        for team in (0, 1):
            lo, hi = team * ts, (team + 1) * ts
            sub = ready[:, lo:hi]
            if not sub.any():
                continue
            n_tiles = self.map_spawn_n[m, team].unsqueeze(1).float()
            r = torch.rand(self.n, ts, generator=self.gen).to(self.device)
            sel = (r * n_tiles).long()
            pts = self.map_spawn[m.unsqueeze(1), team, sel]
            self.pos[:, lo:hi] = torch.where(sub.unsqueeze(2), pts, self.pos[:, lo:hi])
        self.hp = torch.where(ready, self._max_hp, self.hp)
        self.shield = torch.where(ready, self._max_sh, self.shield)
        self.since_dmg = torch.where(ready, torch.zeros_like(self.since_dmg), self.since_dmg)
        self.alive = self.alive | ready
        self.waiting = self.waiting & ~ready
        self.cooldown = torch.where(ready, torch.zeros_like(self.cooldown), self.cooldown)
        self.dash_cd = torch.where(ready, torch.zeros_like(self.dash_cd), self.dash_cd)
        if self.record:
            for w, ag in torch.nonzero(ready).tolist():
                self._events[w].append({"type": "respawn", "frame": int(self.frame[w]), "agent": ag})
        return ready

    def _aim_label(self, vis: torch.Tensor) -> torch.Tensor:
        d = torch.stack([self.angle.cos(), self.angle.sin()], dim=-1)
        rel = self.pos.unsqueeze(1) - self.pos.unsqueeze(2)
        along = (rel * d.unsqueeze(2)).sum(-1)
        perp = (rel[..., 0] * d.unsqueeze(2)[..., 1] - rel[..., 1] * d.unsqueeze(2)[..., 0]).abs()
        team = self._agent_team
        enemy = (team.view(1, -1, 1) != team.view(1, 1, -1))
        both_alive = self.alive.unsqueeze(2) & self.alive.unsqueeze(1)
        in_range = (along > 0) & (along <= self._range_a.unsqueeze(2))
        if not self.rules.friendly_fire:
            hit = enemy & both_alive & vis & in_range & (perp < self._hit_r.unsqueeze(1))
            return hit.any(2).float()
        # Friendly fire: the bullet stops at the FIRST agent on its line (walls: LOS to that agent's centre), so
        # the label is 1 only if that first agent is an enemy — an ally in front turns a good shot into a bad one.
        eye = torch.eye(N_AGENTS, dtype=torch.bool, device=self.device).view(1, N_AGENTS, N_AGENTS)
        on_line = both_alive & vis & in_range & (perp < self._hit_r.unsqueeze(1)) & ~eye
        first = torch.where(on_line, along, torch.full_like(along, float("inf"))).argmin(2, keepdim=True)
        return (on_line.any(2) & enemy.expand_as(on_line).gather(2, first).squeeze(2)).float()
    # endregion FUNC_physics_overrides

    # region FUNC__visibility
    ## @purpose Line of sight within each VIEWER's vision (rules.vision x its bullet_range trait).
    ## Legacy: world11's 96-sample test. Default: exact tile-crossing test (_segment_blocked).
    def _visibility(self, idx: torch.Tensor | None) -> torch.Tensor:
        pos = self._sel(self.pos, idx)
        maps = self._sel(self.map_idx, idx)
        vision = self._sel(self._vision_a, idx)
        n = pos.shape[0]
        a = pos.unsqueeze(2)
        b = pos.unsqueeze(1)
        if not self.rules.exact_los:
            t = torch.linspace(0.0, 1.0, W11.LOS_SAMPLES, device=self.device).view(1, 1, 1, W11.LOS_SAMPLES, 1)
            pts = a.unsqueeze(3) + (b - a).unsqueeze(3) * t
            m = maps.view(n, 1, 1, 1).expand(-1, N_AGENTS, N_AGENTS, W11.LOS_SAMPLES)
            blocked = self._wall_at(m, pts[..., 0], pts[..., 1]).any(-1)
        else:
            i0, i1 = self._pairs[0], self._pairs[1]
            bp = self._segment_blocked(pos[:, i0], pos[:, i1], maps)                # (n, 45)
            blocked = torch.zeros(n, N_AGENTS, N_AGENTS, dtype=torch.bool, device=self.device)
            blocked[:, i0, i1] = bp
            blocked[:, i1, i0] = bp
        dist = (b - a).pow(2).sum(-1).sqrt()
        return ~blocked & (dist <= vision.unsqueeze(2))

    ## @purpose Exact wall test of segments a->b (n, P, 2) px: every tile the open segment passes through.
    ## @rationale A segment enters a tile only across a vertical or a horizontal grid line (or starts in it).
    ## At every vertical line x = k*ts it crosses, both tiles (k-1, row) and (k, row) are tested; the same for
    ## horizontal lines; plus the two end tiles. Corner-only contacts count as blocking (conservative,
    ## measure zero). Cost ~ 2*(dx + dy) tile lookups per pair, bounded per call by the largest pair span.
    def _segment_blocked(self, a: torch.Tensor, b: torch.Tensor, maps: torch.Tensor) -> torch.Tensor:
        ts = self.tile_size
        n, p = a.shape[0], a.shape[1]
        mm = maps.view(n, 1)
        ax, ay, bx, by = a[..., 0], a[..., 1], b[..., 0], b[..., 1]
        ca, ra = (ax / ts).floor().long(), (ay / ts).floor().long()
        cb, rb = (bx / ts).floor().long(), (by / ts).floor().long()
        blocked = self._tile_wall(mm.expand(n, p), ra, ca) | self._tile_wall(mm.expand(n, p), rb, cb)
        nx, ny = (ca - cb).abs(), (ra - rb).abs()
        kx, ky = (int(v) for v in torch.stack([nx.max(), ny.max()]).tolist())     # one host sync per call
        if kx > 0:
            j = torch.arange(1, kx + 1, device=self.device).view(1, 1, kx)
            k = torch.minimum(ca, cb).unsqueeze(-1) + j
            valid = j <= nx.unsqueeze(-1)
            dx = (bx - ax).unsqueeze(-1)
            t = (k.float() * ts - ax.unsqueeze(-1)) / torch.where(dx == 0, torch.ones_like(dx), dx)
            y = ay.unsqueeze(-1) + t * (by - ay).unsqueeze(-1)
            row = (y / ts).floor().long()
            m3 = mm.view(n, 1, 1).expand(n, p, kx)
            hitv = self._tile_wall(m3, row, k - 1) | self._tile_wall(m3, row, k)
            blocked = blocked | (hitv & valid).any(-1)
        if ky > 0:
            j = torch.arange(1, ky + 1, device=self.device).view(1, 1, ky)
            k = torch.minimum(ra, rb).unsqueeze(-1) + j
            valid = j <= ny.unsqueeze(-1)
            dy = (by - ay).unsqueeze(-1)
            t = (k.float() * ts - ay.unsqueeze(-1)) / torch.where(dy == 0, torch.ones_like(dy), dy)
            x = ax.unsqueeze(-1) + t * (bx - ax).unsqueeze(-1)
            col = (x / ts).floor().long()
            m3 = mm.view(n, 1, 1).expand(n, p, ky)
            hith = self._tile_wall(m3, k - 1, col) | self._tile_wall(m3, k, col)
            blocked = blocked | (hith & valid).any(-1)
        return blocked

    def _tile_wall(self, m: torch.Tensor, row: torch.Tensor, col: torch.Tensor) -> torch.Tensor:
        row = row.clamp(0, self.map_grid.shape[1] - 1)
        col = col.clamp(0, self.map_grid.shape[2] - 1)
        return self.map_grid[m, row, col]
    # endregion FUNC__visibility

    # region FUNC__reward
    ## @purpose world11's rewards; with non-baseline weights each agent values every component with ITS OWN
    ## weights: combat_i = boost * sum_c w_ic * base_c * (e_ic - mean over the enemy team of e_jc) (zero-sum
    ## as in exp11b, but "what my enemy's kill costs me" is weighted by what I value); team terms per agent
    ## = w_capture * CP term + w_score * score-delta term + w_win * outcome term.
    ## @rationale With baseline weights world11's own _reward runs (bit-exact legacy); the weighted path
    ## equals it up to float rounding (test_reward_weights_ones_match).
    def _reward(self, dmg_dealt, dmg_taken, kills, deaths, cp_gain, cap_part, score_before, winner, terminated, clock):
        ts = W11.TEAM_SIZE
        delta = self.score - score_before
        by_timeout = clock & terminated
        cap_t = torch.zeros(self.n, 2, device=self.device)
        score_t = torch.zeros(self.n, 2, device=self.device)
        out_t = torch.zeros(self.n, 2, device=self.device)
        for team in (0, 1):
            other = 1 - team
            cap_t[:, team] = cp_gain[:, team] * W11.R_CP_CAPTURED + cp_gain[:, other] * W11.R_CP_LOST
            score_t[:, team] = (delta[:, team] - delta[:, other]) * W11.R_SCORE_DELTA
            my_win = (winner == team + 1) & terminated
            my_loss = (winner == other + 1) & terminated
            draw = (winner == 0) & terminated
            win_pay = torch.where(by_timeout, torch.full_like(cap_t[:, 0], W11.R_WIN_TIMEOUT), torch.full_like(cap_t[:, 0], W11.R_WIN))
            loss_pay = torch.where(by_timeout, torch.full_like(cap_t[:, 0], W11.R_LOSS_TIMEOUT), torch.full_like(cap_t[:, 0], W11.R_LOSS))
            out_t[:, team] = my_win.float() * win_pay + my_loss.float() * loss_pay + draw.float() * W11.R_TIMEOUT_DRAW
        team_of = self._agent_team
        assist_r = float(self.rules.assist_reward)
        if self._w_is_one and assist_r == 0.0 and not self.rules.friendly_fire:
            r_indiv, r_team = super()._reward(dmg_dealt, dmg_taken, kills, deaths, cp_gain, cap_part, score_before, winner, terminated, clock)
            self._r_team_agent = r_team[:, team_of]
            return r_indiv, r_team
        r_team = cap_t + score_t + out_t
        w = self.reward_w
        wi = {k: w[..., i] for i, k in enumerate(REWARD_W12)}
        rc = self.reward_cfg
        # An assist is a share of a kill: it is valued with the agent's kill weight (no 9th weight). Friendly
        # fire mirrors the gains: damage to an ally costs what damage to an enemy earns, a team kill what a kill
        # earns, both x ff_coef and valued with the agent's own damage_dealt / kill weights. Inside the zero-sum
        # term an enemy's friendly fire is, symmetrically, a gain for my side.
        ffc = float(self.rules.ff_coef) if self.rules.friendly_fire else 0.0
        ev = torch.stack([dmg_dealt * rc.damage_dealt, dmg_taken * rc.damage_taken, kills * rc.kill, deaths * rc.death,
                          self._tw["assists"] * assist_r,
                          self._tw["ff_dmg"] * (-ffc * rc.damage_dealt), self._tw["teamkills"] * (-ffc * rc.kill)], -1)
        wc = torch.stack([wi["damage_dealt"], wi["damage_taken"], wi["kill"], wi["death"], wi["kill"],
                          wi["damage_dealt"], wi["kill"]], -1)                                            # (N, A, 7)
        own = (ev * wc).sum(-1)
        if rc.zero_sum:
            mean_b, mean_r = ev[:, :ts].mean(1, keepdim=True), ev[:, ts:].mean(1, keepdim=True)
            enemy_mean = torch.cat([mean_r.expand(-1, ts, -1), mean_b.expand(-1, ts, -1)], 1)   # (N, A, 4)
            own = own - (enemy_mean * wc).sum(-1)
        combat = own * self.combat_boost
        r_indiv = combat + cap_part * W11.R_CP_CAPTURE_INDIV * wi["capture"]
        self._r_team_agent = (wi["capture"] * cap_t[:, team_of] + wi["score_delta"] * score_t[:, team_of]
                              + wi["win"] * out_t[:, team_of])
        return r_indiv, r_team
    # endregion FUNC__reward

    # region FUNC__build_obs
    ## @purpose Experiment-12 observation. Copied from world11._build_obs (frozen) because every token gains
    ## columns and four normalisations become per agent (hp, shield by the owner's max; shot and dash
    ## cooldowns by the owner's own cooldown; bullet velocity by the bullet's own speed) — each equals
    ## world11's constant at trait 1.0 / legacy rules. New columns are listed in OBS_LAYOUT12.
    ## @io (idx or None, vis (n, A, A)) -> (n, A, OBS_SIZE12)
    ## @complexity 10
    def _build_obs(self, idx: torch.Tensor | None, vis: torch.Tensor) -> torch.Tensor:
        S = lambda t: self._sel(t, idx)                                          # noqa: E731
        TS = W11.TEAM_SIZE
        pos, vel, angle = S(self.pos), S(self.vel), S(self.angle)
        hp, shield, alive = S(self.hp), S(self.shield), S(self.alive)
        maps = S(self.map_idx)
        max_hp, max_sh = S(self._max_hp), S(self._max_sh)
        ltr = S(self._log_traits)                                                 # (n, A, K)
        spd_a, range_a = S(self._b_speed_a), S(self._range_a)
        n, a, dev = pos.shape[0], N_AGENTS, self.device
        obs = torch.zeros(n, a, OBS_SIZE12, device=dev)
        arena = self.map_arena[maps].unsqueeze(1)
        diag = self.map_diag[maps].view(n, 1, 1)
        team = self._agent_team.view(1, a).expand(n, a)
        is_blue = (team == 0)
        facing = torch.stack([angle.cos(), angle.sin()], dim=-1)
        pm1 = lambda cond: torch.where(cond, 1.0, -1.0)                          # noqa: E731
        vision = self.rules.vision

        # ---- core: world11's 29, then own traits and readiness ----
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
        # local force ratio (teamwork): alive allies / enemies near me, visible enemies — counts / team size
        rel = pos.unsqueeze(1) - pos.unsqueeze(2)
        dist = rel.pow(2).sum(-1).sqrt()
        same_team = team.unsqueeze(2) == team.unsqueeze(1)
        is_self = torch.eye(a, dtype=torch.bool, device=dev).view(1, a, a)
        other_alive = alive.unsqueeze(1).expand(n, a, a)
        ally = other_alive & same_team & ~is_self
        foe = other_alive & ~same_team
        cnt = lambda m: m.sum(2).float() / TS * 2 - 1                            # noqa: E731
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

        # ---- entity slots ----
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
        # new: signed miss distance of my barrel line at their distance; behind me it saturates to +-1
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

        # ---- bullet slots ----
        b_pos, b_vel, b_alive = S(self.b_pos), S(self.b_vel), S(self.b_alive)
        nb = b_pos.shape[1]
        b_rel = b_pos.unsqueeze(1) - pos.unsqueeze(2)
        b_dist = b_rel.pow(2).sum(-1).sqrt()
        b_team = self._agent_team[S(self.b_owner).long()].unsqueeze(1)
        hostile = (b_team != team.unsqueeze(2)) & b_alive.unsqueeze(1)
        b_order = torch.where(hostile, b_dist, torch.full_like(b_dist, float("inf"))).argsort(dim=2)[:, :, :N_BULLET_SLOTS]
        b_keep = hostile.gather(2, b_order).float().unsqueeze(-1)
        idx2 = b_order.unsqueeze(-1).expand(-1, -1, -1, 2)
        g_brel = b_rel.gather(2, idx2)
        g_bvel = b_vel.unsqueeze(1).expand(n, a, nb, 2).gather(2, idx2)
        gb = lambda t: t.unsqueeze(1).expand(n, a, nb).gather(2, b_order)        # noqa: E731
        g_bspd, g_bdmg = gb(S(self.b_speed)), gb(S(self.b_dmg))
        g_left = (gb(S(self.b_life)) - gb(S(self.b_age))).clamp(min=0).float()
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

        # ---- last-seen enemy memory ----
        mem_valid = S(self.mem_valid)
        my_team_mem = lambda t: torch.cat([t[:, 0:1].expand(n, TS, *t.shape[2:]), t[:, 1:2].expand(n, TS, *t.shape[2:])], dim=1)  # noqa: E731
        m_pos, m_vel = my_team_mem(S(self.mem_pos)), my_team_mem(S(self.mem_vel))
        m_hp, m_age, m_val = my_team_mem(S(self.mem_hp)), my_team_mem(S(self.mem_age)), my_team_mem(mem_valid)
        enemy_idx = torch.cat([torch.arange(TS, 2 * TS), torch.arange(0, TS)]).to(dev)
        enemy_idx = enemy_idx.view(2, TS)[self._agent_team]                       # (A, 5)
        i_see = vis.gather(2, enemy_idx.view(1, a, TS).expand(n, -1, -1))
        show = (m_val & ~i_see).float().unsqueeze(-1)
        en_max_hp = max_hp[:, enemy_idx]                                          # (n, A, 5)
        en_ltr = ltr[:, enemy_idx]                                                # (n, A, 5, K)
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
        ], dim=-1) * show
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
    ## @purpose world11's global state with per-agent normalisations (hp, shield, cooldowns), then the log
    ## traits of the ten agents, own team first (STATE_SIZE12).
    def global_state(self) -> torch.Tensor:
        n, dev, TS = self.n, self.device, W11.TEAM_SIZE
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
        ], dim=-1)
        cp_xy = self.map_cp_xy[self.map_idx]
        out = torch.zeros(n, 2, STATE_SIZE12, device=dev)
        onehot = self.map_desc[self.map_idx]
        for t in (0, 1):
            me = slice(t * TS, (t + 1) * TS)
            en = slice((1 - t) * TS, (2 - t) * TS)
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
            tr = torch.cat([self._log_traits[:, me], self._log_traits[:, en]], dim=1).reshape(n, -1)
            out[:, t] = torch.cat([agents, cps, glob, onehot, tr], dim=1)
        return out
    # endregion FUNC_global_state

    def snapshot(self, world_idx: int) -> dict:
        snap = super().snapshot(world_idx)
        w = int(world_idx)
        for ag in snap["agents"]:
            i = ag["id"]
            ag["r"] = float(self._radius_a[w, i])
            ag["max_hp"] = float(self._max_hp[w, i])
            ag["traits"] = [float(v) for v in self.traits[w, i].tolist()]
        return snap
# endregion CLASS_World12
