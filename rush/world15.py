from __future__ import annotations

# region MODULE_CONTRACT [DOMAIN(9): Simulation; CONCEPT(9): AmmoAndEyes15; TECH(8): torch]
## @modulecontract
## @purpose Experiment 15's worlds: exp14's batched worlds (any team size) and big training / show worlds with
## (a) limited ammo refilled only inside an own, uncontested control point and (b) a wider «eyes» pool — up to 20
## nearest visible enemies + 12 nearest allies as entity tokens (two queues, so allies cannot crowd enemies out),
## with ammo features (own, allies'; enemies' hidden), refill cues on CP tokens and ammo in the critic state.
## @scope One mixin (_Exp15Mixin) layered on top of world14's classes, the OBS_LAYOUT15 / STATE_SIZE15 constants,
## Rules15 and three factories. No trainer logic (vec15/train15), no model (model15).
## @input team size (batched) or a BigMap (big); Rules15; per-agent ammo caps (set_ammo_caps)
## @output world instances with world12's API; observation OBS_SIZE15 (OBS_LAYOUT15), global_state STATE_SIZE15,
## info["action_mask"] with the ammo rule, info["tw"] ammo/aim metrics (num/den per team), info["shots"] etc.
## @links LINKS_TO: world14 (batched_world_class, BigTrainWorld pieces), world12 (OBS_LAYOUT12, entity token
## formula), world_big12 (clones, _Big12Mixin), docs/EXP15_CONTRACT.md, docs/EXP15_NOTES_world.md
## @invariants
## - Rules15(ammo=False, eyes="mixed9"): physics identical to the base world; OBS_LAYOUT15's old columns
##   (old12_to_new15_index) carry exactly the base observation, every new column and slot is 0
##   (tests/test_world15.py compares step by step)
## - A shot needs ammo >= 1 and consumes 1 (dropped shots too — the trigger was pulled); ammo refills at
##   cap / (refill_s * FPS) per frame only while inside the radius of a CP owned by the agent's team with no living
##   enemy inside that radius; full on respawn and at reset
## - Enemy ammo never enters an agent's observation (ammo_frac = ammo_known = 0 on enemy tokens)
## - No host syncs are added to the step path (Python branches on Rules15 only)
## @rationale
## Q: Why post-process the base observation instead of copying the builders?
## A: Three builders exist (world12, world13's clone, world_big12's) and they differ only in core normalisations,
## A: CP choice (7 nearest in big worlds) and last-seen tokens. Core, bullets, CPs, last-seen and grid of exp15 are
## A: the base blocks plus appended columns; only the entity block is rebuilt, with world12's per-token formula over
## A: a different selection. In mode "mixed9" the selection is world12's (argsort, 9 nearest present) — the test
## A: checks the rebuilt block equals the base block bit for bit, so the formula copy cannot drift silently.
## Q: Why derive the CP refill cues from the CP token columns?
## A: The token already holds «owned by me» (col 1), «living enemy inside» (col 10) and the flow path distance
## A: (col 5); refill_here = own & no enemy inside is exactly the refill rule, for whichever CPs the builder chose.
## @changes
## LAST_CHANGE: [v0.1.0] Initial: layout, ammo, eyes pool, metrics, factories.
## @modulemap
## CONST 10[Observation layout with n_old / count_old against OBS_LAYOUT12] => OBS_LAYOUT15
## CLASS 7[Experiment-15 rules] => Rules15
## CLASS 10[Ammo, eyes, metrics on top of any world12-lineage class] => _Exp15Mixin
## FUNC 8[Batched class for a team size] => batched_world15_class
## FUNC 8[Big training world for a map] => BigTrainWorld15
## FUNC 7[Big show world for a map] => WorldBig15
def _module_contract():
    pass
# endregion MODULE_CONTRACT
# GREP_SUMMARY: world15, ammo, refill, own uncontested control point, eyes pool, 20 enemies 12 allies, hidden enemy ammo, OBS_LAYOUT15
# STRUCTURE: ▶ base world (world14 lineage) → ⚡ _Exp15Mixin(_shoot gate, frame refill, respawn/reset refill, mask, obs15, state15, metrics) → ⎋ world

import dataclasses
import logging
import math
from dataclasses import dataclass
from typing import Any

import torch

from rush import world11 as W11
from rush import world12 as W12
from rush.config import FPS

logger = logging.getLogger(__name__)

# region BLOCK_CONSTANTS_LAYOUT15
E_EN: int = 20                            # nearest visible enemies in the eyes pool
E_AL: int = 12                            # nearest allies (no line of sight needed, as in world12)
E_EN_MORE: int = 20                       # v4 entities_more: the next 20 enemies (crowds of 500 v 500) ...
E_AL_MORE: int = 4                        # ... and the next 4 allies, a separate gated readout segment
N_ENTITY_MORE: int = E_EN_MORE + E_AL_MORE
N_ENTITY15: int = E_EN + E_AL
AMMO_REF: float = 60.0                    # absolute ammo normalisation (largest planned cap is ~5 x 38)
REFILL_ETA_NORM_S: float = 10.0           # seconds to the refill point / this, clipped
N_SQUADS: int = 5                         # critic pooling (world13/world14 convention)

CORE15_V1: list[str] = ["ammo_frac", "ammo_abs", "refilling", "refill_eta"]
# v2 (review §4.5): «under fire now» + the agent's own current spread (so it can learn when a shot is worth it)
CORE15_V2: list[str] = ["dmg_recent", "since_dmg", "incoming_n", "incoming_ttc", "spread_now"]
# v3 (28.09, owner: «те кто стоят на точках стоят потому что не видят ближайшего врага»): the commander's ORDER
# (target CP along the flow field) and the direction to the FRONT (nearest non-own CP by path) — what the
# commander decides with and the soldier could not see. Written by the team's commander into world.order_cp.
CORE15_V3: list[str] = ["ord_has", "ord_dx", "ord_dy", "ord_path", "ord_rx", "ord_ry", "ord_owner",
                        "front_has", "front_dx", "front_dy", "front_path"]
CORE15_NEW: list[str] = CORE15_V1 + CORE15_V2 + CORE15_V3
PATH_NORM: float = 6000.0                 # px of path distance at which ord_path / front_path saturate (+1)
ENTITY15_NEW: list[str] = ["ammo_frac", "ammo_known", "on_own_cp"]
CP15_NEW: list[str] = ["refill_here", "refill_eta"]
_NEW15 = {"core": CORE15_NEW, "entities": ENTITY15_NEW, "cps": CP15_NEW, "bullets": [], "lastseen": []}
_COUNT15 = {"entities": N_ENTITY15}
DMG_RECENT_TAU: float = 30.0              # frames: exponential window of «damage taken recently» (~0.5 s)
SINCE_DMG_NORM: float = 180.0             # frames since the last damage / this (3 s = shield regen delay)
INCOMING_TTC_FRAMES: float = 30.0         # a hostile bullet «incoming» = passes within my hit radius within this
GRID2_RADIUS: int = 10                    # v2 wide grid: 21 x 21 tiles (review §4.8), a separate new-path token
GRID2_SIDE: int = 2 * GRID2_RADIUS + 1
SPREAD_NORM_DEG: float = 4.0              # spread_now / this
# v4 (28.09, owner: «можем улучшить зрение сети»): a soldier in a crowd faces far more bullets and enemies than it saw; the net saw
# 8 nearest bullets and 32 entities. bullets_pool = the 32 most dangerous hostile bullets (predicted to pass within
# BULLET_MISS_NORM, soonest first; then the nearest others), world12's bullet token formula. battle = a coarse
# egocentric map of the fight: 17 x 17 cells of 13 tiles (390 px, ±3.3 k px), channels below.
N_BULLET_POOL: int = 32
BATTLE_TILES: int = 13
BATTLE_R: int = 8
BATTLE_SIDE: int = 2 * BATTLE_R + 1
BATTLE_NORM: float = 4.0                  # agents per cell at which a density channel saturates
BATTLE_CH: list[str] = ["allies", "enemies_seen", "cp_mine", "cp_theirs", "cp_neutral", "walls"]


## @purpose Segment table. Every segment carries `path`: "old" = what exp14's encoder saw (the graft source, widened
## tokens get zero-init columns), "new" = only through exp15's state-conditioned readout (zero-init), so the
## grafted network equals exp14 at ANY scale (v2 item 1). entities_old = world12's 9 nearest present entities,
## exactly exp14's token block; entities = the 20 enemies + 12 allies pool; grid21 = the wide egocentric grid.
def _layout15() -> list[dict]:
    out, start = [], 0

    def add(d: dict, flat: int) -> None:
        nonlocal start
        d["start"] = start
        start += flat
        out.append(d)

    for s in W12.OBS_LAYOUT12:
        base = {k: v for k, v in s.items() if k not in ("n_old", "old_start", "new_features", "start")}
        if s["kind"] == "grid":
            d = dict(base, n_old=s["size"], old_start=s["start"], count_old=1, new_features=[], path="old")
            add(d, s["size"][0] * s["size"][1] * s["size"][2])
            g2 = dict(base, name="grid21", size=[s["size"][0], GRID2_SIDE, GRID2_SIDE], n_old=None, old_start=None,
                      count_old=0, new_features=["walls", "cp_zone_signed"], path="new")
            add(g2, s["size"][0] * GRID2_SIDE * GRID2_SIDE)
            continue
        if s["name"] == "entities":
            old = dict(base, name="entities_old", n_old=s["size"], old_start=s["start"], count_old=s["count"],
                       new_features=[], path="old")
            add(old, s["count"] * s["size"])
            pool = dict(base, n_old=s["size"], old_start=None, count_old=0, count=N_ENTITY15,
                        new_features=list(ENTITY15_NEW), size=s["size"] + len(ENTITY15_NEW), path="new")
            add(pool, pool["count"] * pool["size"])
            continue
        d = dict(base, n_old=s["size"], old_start=s["start"], count_old=s["count"], count=s["count"],
                 new_features=list(_NEW15[s["name"]]), size=s["size"] + len(_NEW15[s["name"]]), path="old")
        add(d, d["count"] * d["size"])
    # v4: appended readout-only segments (every earlier start stays where it was)
    ent = next(d for d in out if d["name"] == "entities")
    add(dict(ent, name="entities_more", count=N_ENTITY_MORE), N_ENTITY_MORE * ent["size"])
    bs = next(s for s in W12.OBS_LAYOUT12 if s["name"] == "bullets")["size"]
    add({"name": "bullets_pool", "kind": "tokens", "count": N_BULLET_POOL, "size": bs, "n_old": None, "old_start": None,
         "count_old": 0, "new_features": ["danger_ordered"], "path": "new"}, N_BULLET_POOL * bs)
    add({"name": "battle", "kind": "grid", "count": 1, "size": [len(BATTLE_CH), BATTLE_SIDE, BATTLE_SIDE], "n_old": None,
         "old_start": None, "count_old": 0, "new_features": list(BATTLE_CH), "path": "new"},
        len(BATTLE_CH) * BATTLE_SIDE * BATTLE_SIDE)
    return out


OBS_LAYOUT15: list[dict] = _layout15()
SEG15: dict[str, dict] = {s["name"]: s for s in OBS_LAYOUT15}
OBS_SIZE15: int = sum((s["count"] * s["size"]) if s["kind"] != "grid" else s["size"][0] * s["size"][1] * s["size"][2]
                      for s in OBS_LAYOUT15)
LAYOUT_VERSION15: int = 4                 # 3: core + orders/front (CORE15_V3); 4: pool 56, bullets_pool, battle map
STATE15_NEW: list[str] = [f"ammo_squad_{i}" for i in range(2 * N_SQUADS)]   # own team's 5 squads first
STATE_SIZE15: int = W12.STATE_SIZE12 + len(STATE15_NEW)
STATE_LAYOUT15: dict = {"n_old": W12.STATE_SIZE12, "size": STATE_SIZE15,
                        "new": "ammo fraction per squad (5 own, then 5 enemy); 0 with ammo off"}


def old12_to_new15_index() -> torch.Tensor:
    """(OBS_SIZE12,) long: the OBS_SIZE15 column that holds each OBS_LAYOUT12 column. v2: world12's entity block
    maps to entities_old (the old path), its grid to grid; OBS_LAYOUT12 order is preserved (old_start ascending)."""
    idx = []
    for s in sorted((s for s in OBS_LAYOUT15 if s.get("old_start") is not None), key=lambda s: s["old_start"]):
        if s["kind"] == "grid":
            k = s["size"][0] * s["size"][1] * s["size"][2]
            idx.extend(range(s["start"], s["start"] + k))
            continue
        for slot in range(s["count_old"]):
            base = s["start"] + slot * s["size"]
            idx.extend(range(base, base + s["n_old"]))
    out = torch.tensor(idx, dtype=torch.long)
    assert out.numel() == W12.OBS_SIZE12, (out.numel(), W12.OBS_SIZE12)
    return out


def embed_obs12(obs12: torch.Tensor) -> torch.Tensor:
    """OBS_SIZE12 observation -> OBS_SIZE15 with every new column / slot zero (the graft's step-0 input)."""
    out = torch.zeros(*obs12.shape[:-1], OBS_SIZE15, dtype=obs12.dtype, device=obs12.device)
    out[..., old12_to_new15_index().to(obs12.device)] = obs12
    return out
# endregion BLOCK_CONSTANTS_LAYOUT15


# region CLASS_Rules15
## @purpose Experiment-15 switches, kept apart from Rules12 (the base worlds build and patch Rules12 themselves).
## ammo=False + eyes="mixed9" = the base world with the exp15 layout (new columns zero).
@dataclass(frozen=True)
class Rules15:
    ammo: bool = True
    refill_s: float = 3.5                 # seconds of standing in an own uncontested CP for a full cap
    cap_default: float = 30.0             # until set_ammo_caps
    eyes: str = "pool"                    # "pool" (20 enemies + 12 allies) | "mixed9" (world12's 9 nearest present)
    e_en: int = E_EN
    e_al: int = E_AL
    # ---- v2 (docs/PERCEPTION_REVIEW.md §4, contract «v2») ----
    spread: bool = True                   # random angular error of each shot (Gaussian, clipped at spread_clip sigma)
    spread_base_deg: float = 0.25         # sigma standing still, not turning
    spread_move_deg: float = 1.0          # + at full own speed (linear in speed / own max speed)
    spread_dash_deg: float = 1.5          # + while dashing
    spread_turn_deg: float = 1.0          # + for a turn of spread_turn_ref_deg or more in this decision
    spread_turn_ref_deg: float = 30.0
    spread_class_pow: float = 2.0         # class multiplier = bullet_range_trait ** -pow (sniper 1.2 -> 0.69)
    spread_clip: float = 2.5
    range_px: float | None = 1600.0       # bullet range at trait 1.0; None = Rules12's (2400). Vision stays Rules12's
    class_range: bool = False             # False: every class shoots range_px (the class trait works through spread
                                          # and vision); True: range_px x bullet_range trait (1280..1920)
    obs_v2: bool = True                   # under-fire / spread_now core columns and the 21x21 grid (0 when False)
    # ---- v2.1 (27.09, owner: «make sure bullets don't pass through them; bullets x2 faster») ----
    exact_hits: bool = True               # segment-vs-circle hit test over each bullet substep (no grazing tunnelling)
    bullet_speed_mult: float = 2.0        # x every class's bullet speed; range unchanged -> flight frames halve
    max_substep_exact_px: float = 30.0    # with exact hits the substep only has to stop wall tunnelling (<= 1 tile)
    # ---- v2.2 (27.09, owner: dodging must cost; «постепенное уменьшение боеспособности при потере здоровья») ----
    shoot_move_mult: float = 0.6          # x move speed (dash too) while the gun is cooling down after a shot
    hit_slow_mult: float = 0.5            # x move speed for hit_slow_s after taking any damage
    hit_slow_s: float = 0.4
    low_hp_cd_mult: float = 2.0           # shot cooldown x (1 + (mult-1) * lost share of hp+shield): x2 near death
    # ---- v2.3 (27.09 evening, owner: «обзоры и дальнобойности +15% для всех, урон +50%, отключим френдли фаер») ----
    reach_mult: float = 1.15              # x bullet range and x vision for every class
    dmg_mult: float = 1.5                 # x bullet damage for every class
    friendly_fire: bool | None = False    # overrides Rules12.friendly_fire (None = keep Rules12's); off = through allies
    # ---- v2.4 (27.09 evening, owner: longer matches; spawn on captured points by a heuristic, later a commander) ----
    match_mult: float = 2.0               # big maps: score_to_win and max_frames x this (lives unchanged)
    cp_spawn: bool = True                 # a respawning agent with spawn_plan = k appears at own safe CP k
    cp_spawn_safe_r: float = 1.5          # «safe» = no living enemy within this x CP radius of the CP centre
    cp_spawn_zone: float = 0.6            # spawn tiles within this x CP radius (3x3 free block around the tile)
    cp_spawn_only: bool = True            # v2.5: the base only at the match start; later only own safe CPs (or wait)
    # ---- v3 observation (28.09): the commander's order + the front direction in the core (CORE15_V3) ----
    obs_orders: bool = True               # False = those 11 columns stay 0 (the layout keeps them either way)
    obs_v4: bool = True                   # False = bullets_pool and battle stay 0 (the layout keeps them either way)
    _NO_FATIGUE = {"shoot_move_mult": 1.0, "hit_slow_mult": 1.0, "low_hp_cd_mult": 1.0}   # not a field
    _V2_2 = {"reach_mult": 1.0, "dmg_mult": 1.0, "friendly_fire": None,                   # not a field: 27.09 afternoon
             "match_mult": 1.0, "cp_spawn": False, "obs_orders": False, "obs_v4": False}
    _V2_3 = {"match_mult": 1.0, "cp_spawn": False, "obs_orders": False, "obs_v4": False}  # not a field: 27.09 17:20
    _V2_4 = {"cp_spawn_only": False, "obs_v4": False}                                     # not a field: 28.09 08:40

    @property
    def fatigue(self) -> bool:
        return self.shoot_move_mult < 1 or self.hit_slow_mult < 1 or self.low_hp_cd_mult > 1

    def __post_init__(self) -> None:
        if self.eyes not in ("pool", "mixed9"):
            raise ValueError(f"eyes must be 'pool' or 'mixed9': {self.eyes!r}")
        if self.e_en + self.e_al > N_ENTITY15:
            raise ValueError(f"eyes pool {self.e_en}+{self.e_al} exceeds {N_ENTITY15} slots")
        if self.range_px is not None and self.range_px <= 0:
            raise ValueError(f"range_px must be > 0 or None: {self.range_px}")
        if self.bullet_speed_mult <= 0:
            raise ValueError(f"bullet_speed_mult must be > 0: {self.bullet_speed_mult}")
        if not (0 < self.shoot_move_mult <= 1 and 0 < self.hit_slow_mult <= 1 and self.low_hp_cd_mult >= 1):
            raise ValueError("move mults must be in (0, 1], low_hp_cd_mult >= 1")

    @staticmethod
    def legacy() -> "Rules15":
        """exp14's physics and exp14's inputs on the old path: no ammo, no spread, Rules12's range, v2 columns 0."""
        return Rules15(ammo=False, eyes="mixed9", spread=False, range_px=None, obs_v2=False, exact_hits=False,
                       bullet_speed_mult=1.0, **Rules15._NO_FATIGUE, **Rules15._V2_2)

    @staticmethod
    def v1() -> "Rules15":
        """exp15 v1's rules (ammo + pool eyes, no spread, Rules12's range, no v2 columns) in the v2 layout."""
        return Rules15(spread=False, range_px=None, obs_v2=False, exact_hits=False, bullet_speed_mult=1.0,
                       **Rules15._NO_FATIGUE, **Rules15._V2_2)

    @staticmethod
    def v2_0() -> "Rules15":
        """exp15 v2 as trained until 27.09: point hit test at substep ends, class bullet speeds x1."""
        return Rules15(exact_hits=False, bullet_speed_mult=1.0, **Rules15._NO_FATIGUE, **Rules15._V2_2)

    @staticmethod
    def v2_1() -> "Rules15":
        """27.09 afternoon: exact hits + bullets x2, no combat fatigue."""
        return Rules15(**Rules15._NO_FATIGUE, **Rules15._V2_2)

    @staticmethod
    def v2_2() -> "Rules15":
        """27.09 late afternoon: v2.1 + combat fatigue."""
        return Rules15(**Rules15._V2_2)

    @staticmethod
    def v2_3() -> "Rules15":
        """27.09 17:20: v2.2 + reach x1.15, damage x1.5, no friendly fire (the approved crowd-fire battle)."""
        return Rules15(**Rules15._V2_3)
# endregion CLASS_Rules15


# region CLASS__Exp15Mixin
## @purpose Ammo rules, the eyes pool observation, ammo in the critic state and the ammo/aim metrics, on top of
## any world12-lineage class (batched world13/world14 clones, big training and show worlds).
## @complexity 10
class _Exp15Mixin:
    obs_layout = OBS_LAYOUT15
    obs_size = OBS_SIZE15
    state_size = STATE_SIZE15

    def __init__(self, *args: Any, rules15: Rules15 | None = None, **kw: Any) -> None:
        self.rules15 = rules15 or Rules15()
        self._ammo_ready = False
        super().__init__(*args, **kw)                                                 # type: ignore[call-arg]
        if self.rules15.friendly_fire is not None and self.rules.friendly_fire != self.rules15.friendly_fire:
            self.rules = dataclasses.replace(self.rules, friendly_fire=bool(self.rules15.friendly_fire))   # v2.3
        n, a, dev = self.n, int(self.alive.shape[1]), self.device
        self._A15, self._TS15 = a, a // 2
        self.ammo_cap = torch.full((n, a), float(self.rules15.cap_default), device=dev)
        self.ammo = self.ammo_cap.clone()
        self._refilling_prev = torch.zeros(n, a, dtype=torch.bool, device=dev)
        self._new_counters()
        # ---- v2 state: under-fire tracker, this decision's turn, the spread generator, the wide grid tables ----
        self._dmg_recent = torch.zeros(n, a, device=dev)
        self._since_hit = torch.full((n, a), SINCE_DMG_NORM, device=dev)   # own counter: the base's since_dmg is 0
        self._hs_prev = (self.hp + self.shield).clone()                    # at spawn, which would read «just hit»
        self._angle0 = self.angle.clone()
        self._slow_left = torch.zeros(n, a, device=dev)                    # v2.2: frames of «just hit» slowdown left
        self.spawn_plan = torch.full((n, a), -1, dtype=torch.long, device=dev)   # v2.4: CP to respawn at (-1 = base)
        self.order_cp = torch.full((n, a), -1, dtype=torch.long, device=dev)     # v3: the commander's target CP (-1 none)
        self.cp_spawns_total = torch.zeros(n, 2, device=dev)
        self._build_cp_spawn_tables()
        self._turn_deg = torch.zeros(n, a, device=dev)
        seed15 = (int(self.gen.initial_seed()) ^ 0x15AB) & 0x7FFFFFFF
        try:
            self._gen15 = torch.Generator(device=dev).manual_seed(seed15)
            self._gen15_dev = True
        except (RuntimeError, TypeError):                  # a backend without device generators: sample on CPU
            self._gen15 = torch.Generator(device="cpu").manual_seed(seed15)
            self._gen15_dev = False
        self._build_grid21_tables()
        self._ammo_ready = True
        logger.info(f"[IMP:9][_Exp15Mixin.__init__][INIT] {type(self).__name__}: worlds={n} agents={a} ammo={self.rules15.ammo} "
                    f"cap={self.rules15.cap_default} refill_s={self.rules15.refill_s} eyes={self.rules15.eyes} "
                    f"({self.rules15.e_en}+{self.rules15.e_al}) obs={OBS_SIZE15} state={STATE_SIZE15} [VALUE]")

    # region FUNC_set_ammo_caps
    ## @purpose Per-agent caps (tensor broadcastable to (k, A) for the worlds in idx, or (n, A) / (A,) / scalar for
    ## all). Ammo above a new cap is clipped; a larger cap is not topped up (it fills at the point or on respawn).
    def set_ammo_caps(self, caps: Any, idx: torch.Tensor | None = None) -> None:
        c = torch.as_tensor(caps, dtype=torch.float32, device=self.device)
        if bool((c < 1).any()):
            raise ValueError(f"ammo caps must be >= 1: min={float(c.min())}")
        if idx is None:
            self.ammo_cap = c.expand(self.n, self._A15).clone()
        else:
            idx = torch.as_tensor(idx, dtype=torch.long, device=self.device).flatten()
            self.ammo_cap[idx] = c.expand(int(idx.numel()), self._A15)
        self.ammo = torch.minimum(self.ammo, self.ammo_cap)
    # endregion FUNC_set_ammo_caps

    @property
    def refilling(self) -> torch.Tensor:
        """(n, A) bool: the agent received ammo in the last frame (for the viewer's refill glow)."""
        return self._refilling_prev.clone()

    def fill_ammo(self, idx: torch.Tensor | None = None) -> None:
        """Ammo to the cap (all worlds or the worlds in idx) — e.g. after set_ammo_caps at a match start."""
        if idx is None:
            self.ammo = self.ammo_cap.clone()
        else:
            self.ammo[idx] = self.ammo_cap[idx]

    def _new_counters(self) -> None:
        z = lambda: torch.zeros(self.n, self._A15, device=self.device)                # noqa: E731
        self._c_shots, self._c_empty, self._c_alive, self._c_trips = z(), z(), z(), z()
        self._c_dead_empty = torch.zeros(self.n, 2, device=self.device)

    # region FUNC__cp_occupancy
    ## @purpose (on_own (n,A), refill_ok (n,A)): inside the radius of a CP owned by the own team; and that CP has
    ## no living enemy inside its radius. idx selects worlds (observation of a subset).
    def _cp_occupancy(self, idx: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        S = lambda t: t if idx is None else t[idx]                                   # noqa: E731
        pos, alive, maps = S(self.pos), S(self.alive), S(self.map_idx)
        n, a, ts = pos.shape[0], self._A15, self._TS15
        cp_xy = self.map_cp_xy[maps]                                                  # (n, C, 2)
        r = self.map_cp_r[maps].view(n, 1, 1)
        d = (pos.unsqueeze(2) - cp_xy.unsqueeze(1)).pow(2).sum(-1).sqrt()             # (n, A, C)
        in_r = (d <= r) & alive.unsqueeze(-1)
        blue_in, red_in = in_r[:, :ts].any(1), in_r[:, ts:].any(1)                    # (n, C)
        is_blue = (self._agent_team == 0).view(1, a, 1)
        enemy_in = torch.where(is_blue, red_in.unsqueeze(1), blue_in.unsqueeze(1))
        own = S(self.cp_owner).unsqueeze(1) == (self._agent_team + 1).view(1, a, 1)
        mine = in_r & own
        return mine.any(-1), (mine & ~enemy_in).any(-1)
    # endregion FUNC__cp_occupancy

    # region FUNC__derive15
    ## @purpose world12's physics tables, then v2's range rule: every class shoots Rules15.range_px (or range_px x
    ## the bullet_range trait with class_range) — bullet lifetime recomputed; vision stays Rules12's (2400 x trait).
    ## The spread multiplier per agent = bullet_range_trait ** -spread_class_pow (the class difference moves from
    ## range to accuracy: sniper 1.2 -> 0.69, assault 0.8 -> 1.56).
    def _derive(self) -> None:
        super()._derive()                                                              # type: ignore[misc]
        r15 = getattr(self, "rules15", None)
        br = self.traits[..., 1]
        self._spread_mult = br.pow(-(r15.spread_class_pow if r15 is not None else 0.0))
        if r15 is None:
            return
        faster = float(r15.bullet_speed_mult) != 1.0
        if faster:
            self._b_speed_a = self._b_speed_a * float(r15.bullet_speed_mult)
            self._b_life_a = torch.round(self._range_a / self._b_speed_a).clamp(min=1).to(torch.int32)
        if r15.range_px is not None:
            rng = torch.full_like(br, float(r15.range_px))
            self._range_a = rng * br if r15.class_range else rng
            self._b_life_a = torch.round(self._range_a / self._b_speed_a).clamp(min=1).to(torch.int32)
        if r15.reach_mult != 1.0:                        # v2.3: range and vision x reach_mult (super() rebuilt them)
            self._range_a = self._range_a * float(r15.reach_mult)
            self._vision_a = self._vision_a * float(r15.reach_mult)
            self._b_life_a = torch.round(self._range_a / self._b_speed_a).clamp(min=1).to(torch.int32)
        if r15.dmg_mult != 1.0:
            self._dmg_a = self._dmg_a * float(r15.dmg_mult)
        if (faster or r15.exact_hits) and self.rules.swept_bullets:
            # exact hits: the substep only guards walls (a 30 px tile, +-3 px bullet box); point test: world12's limit
            lim = float(r15.max_substep_exact_px) if r15.exact_hits else min(self.rules.max_substep_px, 2.0 * float(self._hit_r.min()))
            self._substeps = max(1, math.ceil(float(self._b_speed_a.max()) / lim))
    # endregion FUNC__derive15

    # region FUNC_exact_hits
    ## @purpose Exact bullet hits (Rules15.exact_hits). world12 tests the bullet POINT at the end of each substep
    ## against the hit radius, so a bullet crossing a body near its edge (chord shorter than the substep) can pass
    ## through unregistered. Here each substep's SEGMENT is tested: for every live bullet the first body along the
    ## segment (smallest entry parameter) within its hit radius is found, the bullet is moved to its closest-approach
    ## point on that body, and world12's own resolution (damage, credit, friendly fire, events, kills) runs unchanged
    ## on it. A bullet stopped by a wall in this substep may still hit a body in front of the wall (resolution before
    ## the wall kill); bodies never stand inside walls. Bodies do not move during the bullet substeps (world12 moves
    ## agents once per frame before firing), so the segment test is exact for the frame.
    def _advance_bullets_sub(self, s: int, sub: int) -> None:
        if not self.rules15.exact_hits:
            super()._advance_bullets_sub(s, sub)                                       # type: ignore[misc]
            return
        prev, alive_before = self.b_pos, self.b_alive
        super()._advance_bullets_sub(s, sub)                                           # type: ignore[misc]
        self._b_prev = prev
        expired = self.b_age > self.b_life
        self._b_wall = alive_before & ~self.b_alive & ~expired

    def _resolve_hits(self, dmg_dealt, dmg_taken, kills, deaths) -> None:
        prev = getattr(self, "_b_prev", None)
        if not self.rules15.exact_hits or prev is None:
            super()._resolve_hits(dmg_dealt, dmg_taken, kills, deaths)               # type: ignore[misc]
            return
        wall = self._b_wall
        self._b_prev = None
        live = self.b_alive | wall
        idx = live.nonzero()                                                           # (M, 2): world, slot
        if idx.numel() == 0:
            return
        w, b = idx[:, 0], idx[:, 1]
        p0, p1 = prev[w, b], self.b_pos[w, b]                                          # (M, 2)
        seg = p1 - p0
        l2 = seg.pow(2).sum(-1).clamp(min=1e-9)
        a_n = int(self.pos.shape[1])
        hr = self._hit_r.expand(self.n, a_n)
        owner = self.b_owner[w, b].long()
        agents = torch.arange(a_n, device=self.device).view(1, a_n)
        m = int(w.numel())
        best_t = torch.full((m,), float("inf"), device=self.device)
        best_pt = p1.clone()
        chunk = max(1, 4_000_000 // max(1, a_n))
        for c0 in range(0, m, chunk):
            sl = slice(c0, min(m, c0 + chunk))
            ww = w[sl]
            q = self.pos[ww]                                                           # (m', A, 2)
            rel = q - p0[sl].unsqueeze(1)
            t = ((rel * seg[sl].unsqueeze(1)).sum(-1) / l2[sl].unsqueeze(1)).clamp(0.0, 1.0)
            closest = p0[sl].unsqueeze(1) + seg[sl].unsqueeze(1) * t.unsqueeze(-1)
            d2 = (q - closest).pow(2).sum(-1)
            r_ = hr[ww]
            if self.rules.friendly_fire:
                can = owner[sl].unsqueeze(1) != agents
            else:
                can = self._agent_team.view(1, a_n) != self._agent_team[owner[sl]].unsqueeze(1)
            hit = (d2 < r_.pow(2)) & can & self.alive[ww]
            t_in = t - (r_.pow(2) - d2).clamp(min=0.0).sqrt() / l2[sl].sqrt().unsqueeze(1)
            t_in = torch.where(hit, t_in, torch.full_like(t_in, float("inf")))
            tmin, j = t_in.min(1)
            got = torch.isfinite(tmin)
            best_t[sl] = tmin
            pt = closest.gather(1, j.view(-1, 1, 1).expand(-1, 1, 2)).squeeze(1)
            best_pt[sl] = torch.where(got.unsqueeze(-1), pt, p1[sl])
        got = torch.isfinite(best_t)
        pos_new = self.b_pos.clone()
        pos_new[w, b] = best_pt
        alive_new = self.b_alive.clone()
        alive_new[w[got], b[got]] = True                                               # wall-stopped bullets that hit first
        self.b_pos, self.b_alive = pos_new, alive_new
        super()._resolve_hits(dmg_dealt, dmg_taken, kills, deaths)                   # type: ignore[misc]
        self.b_alive = self.b_alive & ~wall
    # endregion FUNC_exact_hits

    # region FUNC_cp_spawn
    ## @purpose Spawn on captured points (Rules15 v2.4). A plan (spawn_plan[w, i] = CP index, written by the team's
    ## commander, rush.army_commander) is honoured at the respawn only if the CP is the
    ## agent's team's and no living enemy stands within cp_spawn_safe_r x radius; otherwise the base spawn stands.
    ## The tile is drawn from the CP's free tiles (3x3 free block) with the world's own generator, like the base.
    def _build_cp_spawn_tables(self) -> None:
        # big battles only (a world with its own rule clone _wm): 5v5 teams start with no CP and keep the base spawn
        self._cp_spawn_ok = bool(self.rules15.cp_spawn) and hasattr(self, "map_cp_xy") and hasattr(self, "_wm")
        if not self._cp_spawn_ok:
            return
        grid = self.map_grid.cpu()                                                    # (maps, rows, cols) bool walls
        ts = float(self.tile_size)
        cxy, cr = self.map_cp_xy.cpu(), self.map_cp_r.cpu().view(-1)
        n_maps, K = int(cxy.shape[0]), int(cxy.shape[1])
        rows, cols = int(grid.shape[1]), int(grid.shape[2])
        tiles: list[list[list[tuple[float, float]]]] = []
        for m in range(n_maps):
            g = grid[m]
            blk = torch.nn.functional.max_pool2d(g.float().view(1, 1, rows, cols), 3, stride=1, padding=1).view(rows, cols) > 0
            per = []
            for k in range(K):
                x, y, rad = float(cxy[m, k, 0]), float(cxy[m, k, 1]), float(cr[m]) * float(self.rules15.cp_spawn_zone)
                c0, c1 = max(0, int((x - rad) / ts)), min(cols - 1, int((x + rad) / ts))
                r0, r1 = max(0, int((y - rad) / ts)), min(rows - 1, int((y + rad) / ts))
                cand = [((c + 0.5) * ts, (r + 0.5) * ts) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)
                        if not bool(blk[r, c]) and ((c + 0.5) * ts - x) ** 2 + ((r + 0.5) * ts - y) ** 2 <= rad * rad]
                per.append(cand[:256])
            tiles.append(per)
        M = max(1, max(len(t) for per in tiles for t in per))
        tbl = torch.zeros(n_maps, K, M, 2)
        cnt = torch.zeros(n_maps, K, dtype=torch.long)
        for m in range(n_maps):
            for k in range(K):
                if tiles[m][k]:
                    tbl[m, k, :len(tiles[m][k])] = torch.tensor(tiles[m][k])
                    cnt[m, k] = len(tiles[m][k])
        self._cp_spawn_tbl, self._cp_spawn_n = tbl.to(self.device), cnt.to(self.device)
        logger.info(f"[IMP:8][_Exp15Mixin._build_cp_spawn_tables][INIT] maps={n_maps} cps={K} tiles/cp min={int(cnt.min())} "
                    f"max={int(cnt.max())} [VALUE]")

    ## @purpose Which CPs a team may spawn on now: (n, 2, K) = own and no living enemy near the centre.
    def cp_spawn_allowed(self) -> torch.Tensor:
        m = self.map_idx
        cxy = self.map_cp_xy[m]                                                        # (n, K, 2)
        safe_r = (self.map_cp_r[m].view(-1, 1) * float(self.rules15.cp_spawn_safe_r))  # (n, 1)
        d = (self.pos.unsqueeze(2) - cxy.unsqueeze(1)).pow(2).sum(-1).sqrt()           # (n, A, K)
        near = (d < safe_r.unsqueeze(1)) & self.alive.unsqueeze(-1)
        team = self._agent_team.view(1, -1, 1)
        near_t = torch.stack([(near & (team == t)).any(1) for t in (0, 1)], 1)         # (n, 2, K): team t near
        owner = self.cp_owner                                                          # 0 neutral, 1 blue, 2 red
        own = torch.stack([owner == 1, owner == 2], 1)
        return own & ~near_t.flip(1)

    ## @io ready (n, A) from the base respawn, died_at (n, A, 2) positions before it -> the agents that really came back.
    ## v2.5 (28.09, owner: «спавн на базе только на старте игры, остальные только на захваченных точках»):
    ## with cp_spawn_only the base is never used again — an agent without a valid plan goes to the own safe CP
    ## nearest to where it died; with no own safe CP at all it stays dead until the next wave (the base respawn is
    ## undone: position, alive, waiting and its event; lives were spent at death, so they are untouched).
    def _cp_respawn(self, ready: torch.Tensor, died_at: torch.Tensor | None = None) -> torch.Tensor:
        plan = self.spawn_plan
        only = bool(self.rules15.cp_spawn_only)
        ok_t = self.cp_spawn_allowed()                                                 # (n, 2, K)
        team = self._agent_team.view(1, -1).expand_as(plan)
        ar = torch.arange(self.n, device=self.device).view(-1, 1)
        m = self.map_idx.view(-1, 1).expand_as(plan)
        k = plan.clamp(min=0)
        ok_plan = (plan >= 0) & ok_t[ar, team, k] & (self._cp_spawn_n[m, k] > 0)
        if only:
            allowed = ok_t[ar, team] & (self._cp_spawn_n[self.map_idx] > 0).unsqueeze(1)   # (n, A, K)
            ref = died_at if died_at is not None else self.pos
            d = (ref.unsqueeze(2) - self.map_cp_xy[self.map_idx].unsqueeze(1)).norm(dim=-1)
            d = torch.where(allowed, d, torch.full_like(d, float("inf")))
            best, near_k = d.min(-1)
            k = torch.where(ok_plan, k, near_k)
            has = ok_plan | torch.isfinite(best)
            stay_dead = ready & ~has
            if bool(stay_dead.any()):
                self._undo_respawn(stay_dead, died_at)
                ready = ready & ~stay_dead
            go = ready
        else:
            go = ready & ok_plan
        if not bool(go.any()):
            return ready
        cnt = self._cp_spawn_n[m, k]
        r = torch.rand(plan.shape, generator=self.gen).to(self.device)
        sel = (r * cnt.clamp(min=1).float()).long().clamp(max=self._cp_spawn_tbl.shape[2] - 1)
        pts = self._cp_spawn_tbl[m, k, sel]                                            # (n, A, 2)
        self.pos = torch.where(go.unsqueeze(-1), pts, self.pos)
        ts = self._TS15
        self.cp_spawns_total += torch.stack([go[:, :ts].sum(1), go[:, ts:].sum(1)], 1)     # (n, 2) since reset
        return ready

    def _undo_respawn(self, mask: torch.Tensor, died_at: torch.Tensor | None) -> None:
        if died_at is not None:
            self.pos = torch.where(mask.unsqueeze(-1), died_at, self.pos)
        self.alive = self.alive & ~mask
        self.waiting = self.waiting | mask
        self.respawn_timer = torch.where(mask, torch.zeros_like(self.respawn_timer), self.respawn_timer)
        if self.record:
            drop = {(w, a) for w, a in torch.nonzero(mask).tolist()}
            for w in {w for w, _ in drop}:
                self._events[w] = [e for e in self._events[w]
                                   if not (e.get("type") == "respawn" and (w, e.get("agent")) in drop)]
    # endregion FUNC_cp_spawn

    # region FUNC_fatigue
    ## @purpose Combat fatigue (Rules15 v2.2): the frame's move is shortened while the gun cools down after a shot
    ## and for a moment after taking damage (the two multiply); the dash is scaled the same way. Scaling the
    ## displacement back towards the start keeps every wall/arena check the base already passed.
    def _move(self, move_a: torch.Tensor) -> None:
        r = self.rules15
        if not r.fatigue:
            super()._move(move_a)                                                      # type: ignore[misc]
            return
        before = self.pos.clone()
        super()._move(move_a)                                                          # type: ignore[misc]
        one = torch.ones_like(self._slow_left)
        mult = (torch.where(self.cooldown > 0, one * r.shoot_move_mult, one)
                * torch.where(self._slow_left > 0, one * r.hit_slow_mult, one))
        self.pos = before + (self.pos - before) * mult.unsqueeze(-1)
        self._slow_left = (self._slow_left - 1).clamp(min=0)

    ## @purpose Cooldown of the shots fired this frame, stretched by the lost share of hp+shield (x low_hp_cd_mult at 0).
    def _fatigue_cooldown(self, firing: torch.Tensor) -> None:
        k = self.rules15.low_hp_cd_mult
        if k <= 1:
            return
        lost = (1.0 - (self.hp + self.shield) / (self._max_hp + self._max_sh)).clamp(0.0, 1.0)
        cd = torch.round(self.cooldown.float() * (1.0 + (k - 1.0) * lost)).to(self.cooldown.dtype)
        self.cooldown = torch.where(firing, cd, self.cooldown)
    # endregion FUNC_fatigue

    # region FUNC_spread
    ## @purpose Spread sigma in degrees per agent (n, A) from the current frame's state: base + speed share + dash +
    ## this decision's turn, times the class multiplier. 0 when Rules15.spread is off.
    def spread_sigma_deg(self) -> torch.Tensor:
        r = self.rules15
        if not r.spread:
            return torch.zeros_like(self._turn_deg)
        spd = self.vel.pow(2).sum(-1).sqrt()
        move = (spd / self._speed_a).clamp(0.0, 1.0)                                  # dash speed counts as 1 + dash term
        dash = (self.dash_left > 0).float()
        turn = (self._turn_deg / r.spread_turn_ref_deg).clamp(max=1.0)
        s = r.spread_base_deg + r.spread_move_deg * move + r.spread_dash_deg * dash + r.spread_turn_deg * turn
        return s * self._spread_mult

    def _spread_noise(self) -> torch.Tensor:
        shape = tuple(self._turn_deg.shape)
        if self._gen15_dev:
            z = torch.randn(shape, generator=self._gen15, device=self.device)
        else:
            z = torch.randn(shape, generator=self._gen15).to(self.device)
        return z.clamp(-self.rules15.spread_clip, self.rules15.spread_clip)
    # endregion FUNC_spread

    # region FUNC_physics15
    ## @purpose The shot gate: shoot only with ammo >= 1; every fired trigger consumes 1 and is counted. v2: the barrel
    ## is rotated by the spread error for the base's bullet placement only (the agent's facing is restored).
    def _shoot(self, shoot_a: torch.Tensor) -> None:
        if self.rules15.ammo:
            shoot_a = torch.where(self.ammo >= 1.0, shoot_a, torch.zeros_like(shoot_a))
        firing = self.alive & (shoot_a == 1) & (self.cooldown <= 0)                  # the base's own firing rule
        d = self.angle - self._angle0
        self._turn_deg = ((d + math.pi) % (2 * math.pi) - math.pi).abs() * (180.0 / math.pi)
        if self.rules15.spread:
            facing = self.angle
            err = self.spread_sigma_deg() * (math.pi / 180.0) * self._spread_noise()
            self.angle = facing + err
            super()._shoot(shoot_a)                                                    # type: ignore[misc]
            self.angle = facing
        else:
            super()._shoot(shoot_a)                                                    # type: ignore[misc]
        self._fatigue_cooldown(firing)
        f = firing.float()
        self._c_shots = self._c_shots + f
        if self.rules15.ammo:
            self.ammo = self.ammo - f

    ## @purpose Once per frame (world11 calls it after hits): shield regen, then the ammo refill and its counters.
    def _regen_shield(self) -> None:
        super()._regen_shield()                                                        # type: ignore[misc]
        if getattr(self, "_ammo_ready", False):                                        # v2 under-fire tracker
            hs = self.hp + self.shield
            taken = (self._hs_prev - hs).clamp(min=0.0) * self.alive.float()
            self._dmg_recent = self._dmg_recent * math.exp(-1.0 / DMG_RECENT_TAU) + taken
            if self.rules15.hit_slow_mult < 1:
                self._slow_left = torch.where(taken > 0, torch.full_like(self._slow_left, round(self.rules15.hit_slow_s * FPS)),
                                              self._slow_left)
            self._since_hit = torch.where(taken > 0, torch.zeros_like(self._since_hit),
                                          (self._since_hit + 1.0).clamp(max=SINCE_DMG_NORM))
            self._hs_prev = hs
        if not self.rules15.ammo:
            return
        _, ok = self._cp_occupancy()
        filling = ok & (self.ammo < self.ammo_cap)
        rate = self.ammo_cap / (self.rules15.refill_s * FPS)
        self.ammo = torch.where(filling, torch.minimum(self.ammo + rate, self.ammo_cap), self.ammo)
        self._c_trips = self._c_trips + (filling & ~self._refilling_prev).float()
        self._refilling_prev = filling
        alive = self.alive.float()
        self._c_alive = self._c_alive + alive
        self._c_empty = self._c_empty + alive * (self.ammo < 1.0).float()

    def _respawn(self) -> torch.Tensor:
        died_at = self.pos.clone() if getattr(self, "_cp_spawn_ok", False) else None
        ready = super()._respawn()                                                     # type: ignore[misc]
        if getattr(self, "_cp_spawn_ok", False) and bool(ready.any()):
            ready = self._cp_respawn(ready, died_at)                                   # v2.4/v2.5: base -> a CP
        if self.rules15.ammo:
            self.ammo = torch.where(ready, self.ammo_cap, self.ammo)
        if getattr(self, "_ammo_ready", False):                                        # v2 under-fire: a fresh life
            self._dmg_recent = torch.where(ready, torch.zeros_like(self._dmg_recent), self._dmg_recent)
            self._since_hit = torch.where(ready, torch.full_like(self._since_hit, SINCE_DMG_NORM), self._since_hit)
            self._hs_prev = torch.where(ready, self.hp + self.shield, self._hs_prev)
            self._slow_left = torch.where(ready, torch.zeros_like(self._slow_left), self._slow_left)
        return ready

    def _reset_idx(self, idx: torch.Tensor, k: int) -> None:
        super()._reset_idx(idx, k)                                                     # type: ignore[misc]
        if getattr(self, "_ammo_ready", False):
            self.ammo[idx] = self.ammo_cap[idx]
            self._refilling_prev[idx] = False
            self._dmg_recent[idx] = 0.0
            self._since_hit[idx] = SINCE_DMG_NORM
            self._hs_prev[idx] = self.hp[idx] + self.shield[idx]
            self._slow_left[idx] = 0.0
            self.spawn_plan[idx] = -1
            self.order_cp[idx] = -1
            self.cp_spawns_total[idx] = 0.0
            self._turn_deg[idx] = 0.0
            self._angle0[idx] = self.angle[idx]

    def reset_worlds(self, mask: torch.Tensor) -> None:
        super().reset_worlds(mask)                                                     # type: ignore[misc]
        if getattr(self, "_ammo_ready", False):
            m = mask.to(self.device).view(-1, 1)
            self.ammo = torch.where(m, self.ammo_cap, self.ammo)
            self._refilling_prev = self._refilling_prev & ~m
            self._dmg_recent = torch.where(m, torch.zeros_like(self._dmg_recent), self._dmg_recent)
            self._since_hit = torch.where(m, torch.full_like(self._since_hit, SINCE_DMG_NORM), self._since_hit)
            self._hs_prev = torch.where(m, self.hp + self.shield, self._hs_prev)
            self._slow_left = torch.where(m, torch.zeros_like(self._slow_left), self._slow_left)
            self.spawn_plan = torch.where(m, torch.full_like(self.spawn_plan, -1), self.spawn_plan)
            self.order_cp = torch.where(m, torch.full_like(self.order_cp, -1), self.order_cp)
            self.cp_spawns_total = torch.where(m, torch.zeros_like(self.cp_spawns_total), self.cp_spawns_total)
            self._turn_deg = torch.where(m, torch.zeros_like(self._turn_deg), self._turn_deg)
            self._angle0 = torch.where(m, self.angle, self._angle0)

    def _teamwork_on_deaths(self, died: torch.Tensor, killer: torch.Tensor, alive_before: torch.Tensor,
                            tk: torch.Tensor | None = None) -> None:
        super()._teamwork_on_deaths(died, killer, alive_before, tk)                    # type: ignore[misc]
        if self.rules15.ammo and getattr(self, "_ammo_ready", False):
            empty = (died & (self.ammo < 1.0)).float()
            ts = self._TS15
            self._c_dead_empty = self._c_dead_empty + torch.stack([empty[:, :ts].sum(1), empty[:, ts:].sum(1)], 1)

    ## @purpose world12's mask; shooting additionally needs ammo >= 1.
    def action_mask(self) -> torch.Tensor:
        m = super().action_mask()                                                      # type: ignore[misc]
        if self.rules15.ammo and getattr(self, "_ammo_ready", False):
            o = W11.N_MOVES + 1
            m[..., o] = m[..., o] & (self.ammo >= 1.0)
        return m
    # endregion FUNC_physics15

    # region FUNC_step
    ## @purpose The base step, then the per-decision ammo / aim metrics in info (num/den per team, per agent).
    ## aimed = shots fired in the decision while the decision's aim label (computed by the base at its start) said
    ## the bullet would hit an enemy first.
    def step(self, actions: torch.Tensor) -> Any:
        self._new_counters()
        self._angle0 = self.angle.clone()          # the decision's turn is measured against this (spread, spread_now)
        out = super().step(actions)                                                    # type: ignore[misc]
        info, n, ts = out.info, self.n, self._TS15
        team = lambda t: t.view(n, 2, ts).sum(-1)                                     # noqa: E731
        shots = self._c_shots
        aimed = shots * (info["aim_label"].float() > 0.5).float()
        hits = info.get("hits", torch.zeros_like(shots))
        kills_team = info["tw"]["kills_adv_den"]
        deaths_team = info["tw"]["lonely_deaths_den"]
        info["shots"], info["aimed_shots"], info["ammo"] = shots, aimed, self.ammo.clone()
        info["tw"].update({
            "aimed_num": team(aimed), "aimed_den": team(shots),
            "accuracy_num": team(hits), "accuracy_den": team(shots),
            "shots_per_kill_num": team(shots), "shots_per_kill_den": kills_team,
        })
        if self.rules15.ammo:
            info["tw"].update({
                "empty_time_num": team(self._c_empty), "empty_time_den": team(self._c_alive),
                "deaths_empty_num": self._c_dead_empty, "deaths_empty_den": deaths_team,
                "refills_num": team(self._c_trips), "refills_den": deaths_team,
            })
        return out
    # endregion FUNC_step

    # region FUNC__build_obs
    ## @purpose The base observation (OBS_LAYOUT12) re-laid into OBS_LAYOUT15: core + ammo cues, the entity block
    ## rebuilt over the eyes pool (world12's per-token formula + ammo_frac/ammo_known/on_own_cp), bullets and
    ## last-seen as they are, CP tokens + refill_here/refill_eta, grid as it is.
    ## @io (idx or None, vis (n, A, A)) -> (n, A, OBS_SIZE15)
    def _build_obs(self, idx: torch.Tensor | None, vis: torch.Tensor) -> torch.Tensor:
        o12 = super()._build_obs(idx, vis)                                             # type: ignore[misc]
        n, a = o12.shape[0], o12.shape[1]
        dev = o12.device
        L12 = {s["name"]: s for s in W12.OBS_LAYOUT12}
        out = torch.zeros(n, a, OBS_SIZE15, device=dev, dtype=o12.dtype)
        ammo_on = self.rules15.ammo
        S = lambda t: t if idx is None else t[idx]                                   # noqa: E731

        def seg12(name: str) -> torch.Tensor:
            s = L12[name]
            return o12[..., s["start"]: s["start"] + s["count"] * s["size"]].view(n, a, s["count"], s["size"])

        def put(name: str, tok: torch.Tensor) -> None:
            s = SEG15[name]
            out[..., s["start"]: s["start"] + s["count"] * s["size"]] = tok.reshape(n, a, -1)

        # ---- CP tokens: base + refill cues ----
        cps12 = seg12("cps")                                                           # (n, A, 7, 14)
        cp_extra = torch.zeros(n, a, cps12.shape[2], len(CP15_NEW), device=dev)
        eta_core = torch.ones(n, a, device=dev)
        if ammo_on:
            own = cps12[..., 1] == 1.0
            foe_inside = cps12[..., 10] == 1.0
            present = cps12[..., 0] == 1.0
            here = own & ~foe_inside & present
            diag = (torch.full((n,), float(self.norm_diag), device=dev) if hasattr(self, "norm_diag")
                    else self.map_diag[S(self.map_idx)])
            p_dist = (cps12[..., 5] + 1.0) * 0.5 * (1.5 * diag.view(n, 1, 1))
            speed = S(self._speed_a).unsqueeze(-1)                                     # px per frame
            eta = ((p_dist / speed / FPS) / REFILL_ETA_NORM_S).clamp(max=1.0) * 2 - 1
            eta = torch.where(here, eta, torch.ones_like(eta))
            cp_extra[..., 0] = torch.where(here, 1.0, -1.0)
            cp_extra[..., 1] = eta
            eta_core = eta.min(-1).values
        put("cps", torch.cat([cps12, cp_extra], -1))

        # ---- core: base + ammo ----
        core12 = o12[..., : L12["core"]["size"]].clone()
        core_extra = torch.zeros(n, a, len(CORE15_NEW), device=dev)
        if ammo_on:
            ammo, cap = S(self.ammo), S(self.ammo_cap)
            _, ok = self._cp_occupancy(idx)
            c = W11.CORE_SIZE + W12.N_TRAITS                                           # shoot_ready column
            ready = S(self.alive) & (S(self.cooldown) <= W11.ACTION_REPEAT) & (ammo >= 1.0)
            core12[..., c] = torch.where(ready, 1.0, -1.0)
            core_extra[..., 0] = ammo / cap * 2 - 1
            core_extra[..., 1] = (ammo / AMMO_REF).clamp(max=1.0) * 2 - 1
            core_extra[..., 2] = torch.where(ok & (ammo < cap), 1.0, -1.0)
            core_extra[..., 3] = eta_core
        if self.rules15.obs_v2:
            v2_end = len(CORE15_V1) + len(CORE15_V2)
            core_extra[..., len(CORE15_V1):v2_end] = self._core_v2(idx, seg12("bullets"))
        if self.rules15.obs_orders and hasattr(self, "flow_dist") and hasattr(self, "order_cp"):
            core_extra[..., len(CORE15_V1) + len(CORE15_V2):] = self._core_v3(idx)
        out[..., : SEG15["core"]["size"]] = torch.cat([core12, core_extra], -1)

        # ---- old path: world12's 9 nearest present entities, exactly the base block (exp14's input) ----
        put("entities_old", seg12("entities"))
        # ---- new path: entity tokens over the eyes pool ----
        ent_main, ent_more = self._entity_tokens15(idx, vis)
        put("entities", ent_main)
        if self.rules15.obs_v4:
            put("entities_more", ent_more)

        # ---- bullets, last-seen: as they are (same sizes and counts) ----
        for name in ("bullets", "lastseen"):
            put(name, seg12(name))
        g12, g15 = L12["grid"], SEG15["grid"]
        k = g12["size"][0] * g12["size"][1] * g12["size"][2]
        out[..., g15["start"]: g15["start"] + k] = o12[..., g12["start"]: g12["start"] + k]
        if self.rules15.obs_v2:
            g2 = SEG15["grid21"]
            k2 = g2["size"][0] * g2["size"][1] * g2["size"][2]
            out[..., g2["start"]: g2["start"] + k2] = self._grid21(idx).reshape(n, a, -1)
        if self.rules15.obs_v4:
            put("bullets_pool", self._bullets_pool(idx))
            gb = SEG15["battle"]
            kb = gb["size"][0] * gb["size"][1] * gb["size"][2]
            out[..., gb["start"]: gb["start"] + kb] = self._battle_map(idx, vis).reshape(n, a, -1)
        return out.clamp(-1.0, 1.0)
    # endregion FUNC__build_obs

    # region FUNC__core_v2
    ## @purpose v2 core columns (n, A, 5), each in [-1, 1]:
    ## dmg_recent  = exponentially-windowed (tau 30 frames ~ 0.5 s) HP+shield lost / own max HP+shield
    ## since_dmg   = frames since the last damage / 180 (the shield-regen delay)
    ## incoming_n  = share of the 8 observed hostile bullets predicted to pass within my hit radius in < 30 frames
    ## incoming_ttc= frames to the soonest such bullet / 30 (1 when none)
    ## spread_now  = my current spread sigma / 4 deg (what a shot fired now would scatter by)
    ## The bullet prediction reads the observed bullet tokens (closest approach, time to it) — no new O(A x B) work.
    def _core_v2(self, idx: torch.Tensor | None, bullets12: torch.Tensor) -> torch.Tensor:
        S = lambda t: t if idx is None else t[idx]                                   # noqa: E731
        n, a, dev = bullets12.shape[0], bullets12.shape[1], bullets12.device
        tot = S(self._max_hp) + S(self._max_sh)
        dmg = (S(self._dmg_recent) / tot).clamp(max=1.0) * 2 - 1
        since = (S(self._since_hit) / SINCE_DMG_NORM).clamp(max=1.0) * 2 - 1
        present = bullets12[..., 0] == 1.0
        closest = (bullets12[..., 5] + 1.0) * 0.5 * W11.BULLET_MISS_NORM
        ttc = (bullets12[..., 6] + 1.0) * 0.5 * W11.BULLET_TTC_NORM
        inc = present & (closest <= S(self._hit_r).unsqueeze(-1)) & (ttc < INCOMING_TTC_FRAMES)
        inc_n = inc.float().sum(-1) / float(bullets12.shape[2]) * 2 - 1
        soon = torch.where(inc, ttc, torch.full_like(ttc, INCOMING_TTC_FRAMES)).min(-1).values
        inc_ttc = (soon / INCOMING_TTC_FRAMES).clamp(max=1.0) * 2 - 1
        sig = S(self.spread_sigma_deg())
        spread = (sig / SPREAD_NORM_DEG).clamp(max=1.0) * 2 - 1
        alive = S(self.alive)
        v2 = torch.stack([dmg, since, inc_n, inc_ttc, spread], -1)
        return torch.where(alive.unsqueeze(-1), v2, torch.zeros_like(v2)).to(dev)
    # endregion FUNC__core_v2

    # region FUNC__core_v3
    ## @purpose v3 core columns (n, A, 11) in [-1, 1] (0 where absent / dead):
    ## ord_has ±1; ord_dx, ord_dy = the flow-field step toward the ordered CP from my tile (the path, not the line);
    ## ord_path = path distance / PATH_NORM (clipped) * 2 - 1; ord_rx, ord_ry = straight offset / PATH_NORM;
    ## ord_owner = +1 mine, -1 the enemy's, 0 neutral; front_* = the same for the nearest non-own CP by path.
    def _core_v3(self, idx: torch.Tensor | None) -> torch.Tensor:
        S = lambda t: t if idx is None else t[idx]                                   # noqa: E731
        pos, alive, maps = S(self.pos), S(self.alive), S(self.map_idx)
        n, a = pos.shape[0], pos.shape[1]
        dev = pos.device
        out = torch.zeros(n, a, len(CORE15_V3), device=dev)
        ts = float(self.tile_size)
        fd, fdir = self.flow_dist, self.flow_dir                                       # (maps, K, rows, cols[, 2])
        K = int(fd.shape[1])
        row = (pos[..., 1] / ts).long().clamp(0, fd.shape[2] - 1)
        col = (pos[..., 0] / ts).long().clamp(0, fd.shape[3] - 1)
        mm = maps.view(n, 1).expand(n, a)
        team = self._agent_team.to(dev).view(1, a).expand(n, a)
        owner = S(self.cp_owner)                                                       # (n, K) 0 / 1 blue / 2 red
        cxy = self.map_cp_xy[maps]                                                     # (n, K, 2)

        def block(k: torch.Tensor, has: torch.Tensor) -> list[torch.Tensor]:
            kk = k.clamp(0, K - 1)
            dirv = fdir[mm, kk, row, col]                                              # (n, A, 2)
            dist = fd[mm, kk, row, col]
            dist = torch.where(torch.isfinite(dist), dist, torch.full_like(dist, PATH_NORM))
            rel = cxy.gather(1, kk.unsqueeze(-1).expand(n, a, 2)) - pos
            return [dirv[..., 0], dirv[..., 1], (dist / PATH_NORM).clamp(max=1.0) * 2 - 1,
                    (rel[..., 0] / PATH_NORM).clamp(-1, 1), (rel[..., 1] / PATH_NORM).clamp(-1, 1)]

        order = S(self.order_cp)
        has_o = (order >= 0) & alive
        ox, oy, op, orx, ory = block(order, has_o)
        own_k = owner.gather(1, order.clamp(0, K - 1))                                 # (n, A)
        o_owner = torch.where(own_k == 0, 0.0, torch.where(own_k == team + 1, 1.0, -1.0))
        # front: the nearest CP (by path from my tile) that is not my team's
        pd = fd[mm.unsqueeze(-1).expand(n, a, K), torch.arange(K, device=dev).view(1, 1, K).expand(n, a, K),
                row.unsqueeze(-1).expand(n, a, K), col.unsqueeze(-1).expand(n, a, K)]
        not_own = owner.unsqueeze(1) != (team + 1).unsqueeze(-1)
        pd = torch.where(not_own & torch.isfinite(pd), pd, torch.full_like(pd, float("inf")))
        best, front = pd.min(-1)
        has_f = torch.isfinite(best) & alive
        fx, fy, fp, _, _ = block(front, has_f)
        cols = [torch.where(has_o, 1.0, -1.0), ox, oy, op, orx, ory, o_owner, torch.where(has_f, 1.0, -1.0), fx, fy, fp]
        v3 = torch.stack(cols, -1)
        keep = torch.stack([torch.ones_like(has_o)] + [has_o] * 6 + [torch.ones_like(has_f)] + [has_f] * 3, -1)
        v3 = torch.where(keep, v3, torch.zeros_like(v3))
        return torch.where(alive.unsqueeze(-1), v3, torch.zeros_like(v3))
    # endregion FUNC__core_v3

    # region FUNC__grid21
    ## @purpose v2 wide egocentric grid (n, A, 2, 21, 21): walls and CP zones signed by owner — world11's formula
    ## over a 10-tile radius, from tables re-padded from the base's (padding counts as wall, zone -1).
    def _build_grid21_tables(self) -> None:
        p0, p1 = W11.GRID_RADIUS, GRID2_RADIUS
        walls = self.map_grid_pad[:, p0: self.map_grid_pad.shape[1] - p0, p0: self.map_grid_pad.shape[2] - p0]
        zone = self.map_zone_pad[:, p0: self.map_zone_pad.shape[1] - p0, p0: self.map_zone_pad.shape[2] - p0]
        m, r, c = walls.shape
        wp = torch.ones(m, r + 2 * p1, c + 2 * p1, dtype=torch.bool, device=self.device)
        wp[:, p1: p1 + r, p1: p1 + c] = walls
        zp = torch.full((m, r + 2 * p1, c + 2 * p1), -1, dtype=zone.dtype, device=self.device)
        zp[:, p1: p1 + r, p1: p1 + c] = zone
        self._g21_walls, self._g21_zone = wp, zp
        g = torch.arange(-p1, p1 + 1, device=self.device)
        self._g21_dr, self._g21_dc = torch.meshgrid(g, g, indexing="ij")

    def _grid21(self, idx: torch.Tensor | None) -> torch.Tensor:
        S = lambda t: t if idx is None else t[idx]                                   # noqa: E731
        pos, maps = S(self.pos), S(self.map_idx)
        n, a, G, GS = pos.shape[0], self._A15, GRID2_RADIUS, GRID2_SIDE
        ts = self.tile_size
        row = (pos[..., 1] / ts).long().clamp(0, self.map_grid.shape[1] - 1)
        col = (pos[..., 0] / ts).long().clamp(0, self.map_grid.shape[2] - 1)
        gr = (row + G).view(n, a, 1, 1) + self._g21_dr.view(1, 1, GS, GS)
        gc = (col + G).view(n, a, 1, 1) + self._g21_dc.view(1, 1, GS, GS)
        gm = maps.view(n, 1, 1, 1).expand(n, a, GS, GS)
        walls = self._g21_walls[gm, gr, gc].float()
        zone = self._g21_zone[gm, gr, gc]
        z_owner = S(self.cp_owner).gather(1, zone.clamp(min=0).view(n, -1)).view(n, a, GS, GS)
        my_t = (self._agent_team + 1).view(1, a, 1, 1)
        z_val = torch.where(z_owner == my_t, 1.0, torch.where(z_owner == 0, 0.5, -1.0))
        z_val = torch.where(zone >= 0, z_val, torch.zeros_like(z_val))
        return torch.stack([walls, z_val], dim=2)
    # endregion FUNC__grid21

    # region FUNC__bullets_pool
    ## @purpose v4 bullet tokens (n, A, 32, 10): hostile bullets, the dangerous ones first (closest approach within
    ## BULLET_MISS_NORM px, soonest first), then the nearest others; world12's per-bullet formula, empty slots zero.
    def _bullets_pool(self, idx: torch.Tensor | None) -> torch.Tensor:
        S = lambda t: t if idx is None else t[idx]                                   # noqa: E731
        pos = S(self.pos)
        n, a, dev = pos.shape[0], pos.shape[1], pos.device
        k = N_BULLET_POOL
        out = torch.zeros(n, a, k, SEG15["bullets_pool"]["size"], device=dev)
        b_alive = S(self.b_alive)
        ab = b_alive.any(0).nonzero().flatten()                                       # slots alive in any world
        if ab.numel() == 0:
            return out
        b_pos, b_vel = S(self.b_pos)[:, ab], S(self.b_vel)[:, ab]
        nb = int(ab.numel())
        team = self._agent_team.view(1, a)
        b_team = self._agent_team[S(self.b_owner)[:, ab].long()]                      # (n, nb)
        hostile = (b_team.unsqueeze(1) != team.unsqueeze(2)) & b_alive[:, ab].unsqueeze(1)
        rel = b_pos.unsqueeze(1) - pos.unsqueeze(2)                                   # (n, A, nb, 2) bullet - me
        dist = rel.pow(2).sum(-1).sqrt()
        vv = b_vel.pow(2).sum(-1).clamp(min=1e-6).unsqueeze(1)
        t_star = (-(rel * b_vel.unsqueeze(1)).sum(-1) / vv).clamp(min=0.0)
        closest = (rel + b_vel.unsqueeze(1) * t_star.unsqueeze(-1)).pow(2).sum(-1).sqrt()
        danger = closest < W11.BULLET_MISS_NORM
        key = torch.where(danger, t_star, 1e4 + dist)
        key = torch.where(hostile, key, torch.full_like(key, float("inf")))
        kk = min(k, nb)
        kv, order = key.topk(kk, dim=2, largest=False)
        keep = torch.isfinite(kv).float().unsqueeze(-1)
        i2 = order.unsqueeze(-1).expand(-1, -1, -1, 2)
        g_rel = rel.gather(2, i2)
        g_vel = b_vel.unsqueeze(1).expand(n, a, nb, 2).gather(2, i2)
        gb = lambda t: t[:, ab].unsqueeze(1).expand(n, a, nb).gather(2, order)       # noqa: E731
        spd, dmg = gb(S(self.b_speed)), gb(S(self.b_dmg))
        left = (gb(S(self.b_life)) - gb(S(self.b_age))).clamp(min=0).float()
        g_t, g_c = t_star.gather(2, order), closest.gather(2, order)
        tok = torch.stack([
            torch.ones_like(g_t),
            (g_rel[..., 0] / W11.BULLET_OBS_RANGE).clamp(-1, 1),
            (g_rel[..., 1] / W11.BULLET_OBS_RANGE).clamp(-1, 1),
            g_vel[..., 0] / spd,
            g_vel[..., 1] / spd,
            (g_c / W11.BULLET_MISS_NORM).clamp(max=1.0) * 2 - 1,
            (g_t / W11.BULLET_TTC_NORM).clamp(max=1.0) * 2 - 1,
            (spd / 36.0).log(),
            (dmg / float(W11.BULLET_DAMAGE)).log(),
            (left * spd / W12.REACH_NORM).clamp(max=1.0) * 2 - 1,
        ], dim=-1) * keep
        out[:, :, :kk] = tok
        return out
    # endregion FUNC__bullets_pool

    # region FUNC__battle_map
    ## @purpose v4 coarse egocentric battle map (n, A, 6, 17, 17), cells of 13 tiles on the map's fixed coarse grid
    ## (the agent's cell in the middle): own team's alive density, density of enemies SEEN by any alive team-mate
    ## (the team's shared sight, no omniscience), own / enemy / neutral CPs, wall fraction (outside the map = wall).
    def _battle_map(self, idx: torch.Tensor | None, vis: torch.Tensor) -> torch.Tensor:
        S = lambda t: t if idx is None else t[idx]                                   # noqa: E731
        pos, alive, maps = S(self.pos), S(self.alive), S(self.map_idx)
        n, a, dev = pos.shape[0], pos.shape[1], pos.device
        ts, p = float(self.tile_size), BATTLE_R
        cell = ts * BATTLE_TILES
        if getattr(self, "_battle_walls", None) is None:                              # (maps, Rc + 2p, Cc + 2p)
            g = self.map_grid.float().clone()                                         # (maps, rows, cols)
            rows, cols = g.shape[1], g.shape[2]
            for mi in range(g.shape[0]):                                              # beyond a map's own arena = wall
                mc = math.ceil(float(self.map_arena[mi, 0]) / ts)
                mr = math.ceil(float(self.map_arena[mi, 1]) / ts)
                g[mi, mr:, :] = 1.0
                g[mi, :, mc:] = 1.0
            pr, pc = (-rows) % BATTLE_TILES, (-cols) % BATTLE_TILES
            g = torch.nn.functional.pad(g, (0, pc, 0, pr), value=1.0).unsqueeze(1)
            wf = torch.nn.functional.avg_pool2d(g, BATTLE_TILES, stride=BATTLE_TILES).squeeze(1)
            self._battle_walls = torch.nn.functional.pad(wf, (p, p, p, p), value=1.0)
        walls = self._battle_walls
        rc, cc = walls.shape[1] - 2 * p, walls.shape[2] - 2 * p
        team = self._agent_team.view(1, a).expand(n, a).to(dev)
        r = (pos[..., 1] / cell).long().clamp(0, rc - 1)
        c = (pos[..., 0] / cell).long().clamp(0, cc - 1)
        flat = r * cc + c                                                             # (n, A)
        grid = torch.zeros(n, 2, len(BATTLE_CH) - 1, rc * cc, device=dev)
        for t in (0, 1):
            mine_t = alive & (team == t)
            seen = (vis & mine_t.unsqueeze(2)).any(1)                                 # (n, A): seen by team t
            grid[:, t, 0].scatter_add_(1, flat, mine_t.float())
            grid[:, t, 1].scatter_add_(1, flat, (alive & (team != t) & seen).float())
        grid[:, :, :2] = (grid[:, :, :2] / BATTLE_NORM).clamp(max=1.0)
        cxy = self.map_cp_xy[maps]                                                    # (n, K, 2)
        cr = (cxy[..., 1] / cell).long().clamp(0, rc - 1)
        ccol = (cxy[..., 0] / cell).long().clamp(0, cc - 1)
        cflat = cr * cc + ccol
        owner = S(self.cp_owner)
        for t in (0, 1):
            for ch, m in ((2, owner == t + 1), (3, (owner != 0) & (owner != t + 1)), (4, owner == 0)):
                grid[:, t, ch].scatter_add_(1, cflat, m.float())
        grid[:, :, 2:] = grid[:, :, 2:].clamp(max=1.0)
        grid = torch.nn.functional.pad(grid.view(n, 2, -1, rc, cc), (p, p, p, p))    # (n, 2, 5, Rp, Cp)
        wall = walls[maps].unsqueeze(1).unsqueeze(1).expand(n, 2, 1, rc + 2 * p, cc + 2 * p)
        grid = torch.cat([grid, wall], 2)                                             # (n, 2, 6, Rp, Cp)
        ar = torch.arange(BATTLE_SIDE, device=dev)
        rows = (r.view(n, a, 1, 1) + ar.view(1, 1, -1, 1))                            # padded: r - p + p
        cols = (c.view(n, a, 1, 1) + ar.view(1, 1, 1, -1))
        w = torch.arange(n, device=dev).view(n, 1, 1, 1)
        out = grid[w, team.view(n, a, 1, 1), :, rows, cols]                           # (n, A, 17, 17, 6)
        return out.permute(0, 1, 4, 2, 3).contiguous()
    # endregion FUNC__battle_map

    # region FUNC__entity_tokens15
    ## @purpose Entity tokens (n, A, N_ENTITY15, 33): world12's 30 features per selected entity (the formula of
    ## world12._build_obs, verbatim) + ammo_frac, ammo_known (allies only — enemy ammo is hidden), on_own_cp.
    ## Selection: "pool" = top-k nearest visible enemies (e_en) then nearest allies (e_al), slots after them empty;
    ## "mixed9" = world12's (argsort of present, 9 nearest), slots 9.. empty.
    def _entity_tokens15(self, idx: torch.Tensor | None, vis: torch.Tensor) -> torch.Tensor:
        S = lambda t: t if idx is None else t[idx]                                   # noqa: E731
        W = W12
        pos, vel, angle = S(self.pos), S(self.vel), S(self.angle)
        hp, shield, alive = S(self.hp), S(self.shield), S(self.alive)
        max_hp, max_sh = S(self._max_hp), S(self._max_sh)
        ltr = S(self._log_traits)
        spd_a, range_a = S(self._b_speed_a), S(self._range_a)
        n, a, dev = pos.shape[0], self._A15, self.device
        ts = self._TS15
        team = self._agent_team.view(1, a).expand(n, a)
        facing = torch.stack([angle.cos(), angle.sin()], dim=-1)
        pm1 = lambda cond: torch.where(cond, 1.0, -1.0)                              # noqa: E731
        vision = self.rules.vision
        rel = pos.unsqueeze(1) - pos.unsqueeze(2)
        dist = rel.pow(2).sum(-1).sqrt()
        same_team = team.unsqueeze(2) == team.unsqueeze(1)
        is_self = torch.eye(a, dtype=torch.bool, device=dev).view(1, a, a)
        other_alive = alive.unsqueeze(1).expand(n, a, a)
        inf = torch.full_like(dist, float("inf"))
        r15 = self.rules15
        if r15.eyes == "mixed9":
            present = other_alive & ~is_self & (same_team | vis)
            k = min(W11.N_ENTITY_SLOTS, a - 1)
            order = torch.where(present, dist, inf).argsort(dim=2)[:, :, :k]
            keep = present.gather(2, order)
        else:
            ke, ka = min(r15.e_en, ts), min(r15.e_al, ts - 1)
            if r15.obs_v4:                                   # v4: the next enemies / allies go to entities_more
                ke, ka = min(r15.e_en + E_EN_MORE, ts), min(r15.e_al + E_AL_MORE, ts - 1)
            foe_key = torch.where(other_alive & ~same_team & vis, dist, inf)
            e_d, e_i = foe_key.topk(ke, dim=2, largest=False)
            parts_i, parts_k = [e_i], [torch.isfinite(e_d)]
            if ka > 0:
                al_key = torch.where(other_alive & same_team & ~is_self, dist, inf)
                a_d, a_i = al_key.topk(ka, dim=2, largest=False)
                parts_i.append(a_i)
                parts_k.append(torch.isfinite(a_d))
            order, keep = torch.cat(parts_i, 2), torch.cat(parts_k, 2)
            k = order.shape[2]
        keep_f = keep.float().unsqueeze(-1)

        def g(t: torch.Tensor) -> torch.Tensor:
            src = t.unsqueeze(1).expand(n, a, *t.shape[1:])
            ix = order if t.dim() == 2 else order.unsqueeze(-1).expand(n, a, k, t.shape[2])
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
        lateral = torch.where(along > 0, (cross / W.LATERAL_NORM).clamp(-1, 1), torch.where(cross >= 0, ones, -ones))
        if r15.ammo:
            on_own, _ = self._cp_occupancy(idx)
            frac = S(self.ammo) / S(self.ammo_cap) * 2 - 1
            g_frac = torch.where(g_same, g(frac), torch.zeros_like(g_dist))
            new = torch.stack([g_frac, g_same.float(), pm1(g(on_own))], dim=-1)
        else:
            new = torch.zeros(n, a, k, len(ENTITY15_NEW), device=dev)
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
            new,
        ], dim=-1) * keep_f
        size = SEG15["entities"]["size"]
        tok = torch.zeros(n, a, N_ENTITY15, size, device=dev)
        more = torch.zeros(n, a, N_ENTITY_MORE, size, device=dev)
        if r15.eyes == "mixed9" or not r15.obs_v4:
            tok[:, :, :k] = ent
            return tok, more
        # v4 split: the pool is [ke enemies][ka allies]; the first e_en / e_al of each (exactly the v3 pool) stay
        # in entities, the rest go to entities_more
        ke0, ka0 = min(r15.e_en, ts), min(r15.e_al, ts - 1)
        en, al = ent[:, :, :ke], ent[:, :, ke:]
        main = torch.cat([en[:, :, :ke0], al[:, :, :ka0]], 2)
        rest = torch.cat([en[:, :, ke0:], al[:, :, ka0:]], 2)
        tok[:, :, :main.shape[2]] = main
        more[:, :, :rest.shape[2]] = rest
        return tok, more
    # endregion FUNC__entity_tokens15

    # region FUNC_global_state
    ## @purpose The base critic state + ammo fraction per squad (5 own squads, then 5 enemy; 0 with ammo off).
    def global_state(self) -> torch.Tensor:
        base = super().global_state()                                                  # type: ignore[misc]
        n = base.shape[0]
        extra = torch.zeros(n, 2, len(STATE15_NEW), device=base.device)
        if self.rules15.ammo and getattr(self, "_ammo_ready", False) and self._TS15 % N_SQUADS == 0:
            k = self._TS15 // N_SQUADS
            sq = (self.ammo / self.ammo_cap).view(n, 2 * N_SQUADS, k).mean(-1)          # (n, 10)
            extra[:, 0] = sq
            extra[:, 1] = torch.cat([sq[:, N_SQUADS:], sq[:, :N_SQUADS]], 1)
        return torch.cat([base, extra], -1)
    # endregion FUNC_global_state
# endregion CLASS__Exp15Mixin


# region FUNC_factories
_BATCHED15: dict[tuple, type] = {}


## @purpose Experiment 15's batched class for a team size: _Exp15Mixin over world14's batched class. Rules15 is
## passed per instance (rules15=...), the sync-guard choice follows world14 (exp14's perf path by default).
def batched_world15_class(ts: int, sync_guards: bool | None = None) -> type:
    from rush.world14 import batched_world_class
    base = batched_world_class(ts, sync_guards)
    key = (ts, base)
    if key not in _BATCHED15:
        _BATCHED15[key] = type(f"World15_ts{ts}", (_Exp15Mixin, base), {})
        logger.info(f"[IMP:8][batched_world15_class][BUILD] World15 ts={ts} on {base.__name__} [VALUE]")
    return _BATCHED15[key]


## @purpose Big training world (world14.BigTrainWorld's classes) with the exp15 mixin on top. The class assembly
## mirrors world14.BigTrainWorld (frozen, builds its class inside the function), with _Exp15Mixin first.
## @purpose Rules15 v2.4 match length on a BigMap: score_to_win and max_frames x match_mult (a copy; the caller's
## map object is shared across worlds and must not compound). A 5v5 MapDef11 has neither field and keeps W11's.
def _scale_match(big_map: Any, rules15: Rules15 | None) -> Any:
    r15 = rules15 or Rules15()
    if r15.match_mult == 1.0 or not hasattr(big_map, "score_to_win"):
        return big_map
    return dataclasses.replace(big_map, score_to_win=int(round(big_map.score_to_win * r15.match_mult)),
                               max_frames=int(round(big_map.max_frames * r15.match_mult)))


def BigTrainWorld15(big_map: Any, device: str = "cpu", seed: int = 0, rules_overrides: dict | None = None,  # noqa: N802
                    rules15: Rules15 | None = None):
    from rush import world14 as W14
    from rush.world_big import _BigMixin, _world11_clone, battle_rules
    from rush.world_big12 import _Big12Mixin, _world12_clone
    big_map = _scale_match(big_map, rules15)
    rules = battle_rules(big_map)
    if rules["team_size"] % W14.N_SQUADS:
        raise ValueError(f"big training world needs team size divisible by {W14.N_SQUADS}: {rules['team_size']}")
    mod11 = _world11_clone(rules["team_size"], len(big_map.cp_positions), rules)
    mod12 = _world12_clone(mod11)
    ov = {k: v for k, v in (rules_overrides or {}).items() if v is not None}
    cls = type("BigTrainWorld15", (_Exp15Mixin, W14.RespawnPlanMixin, W14._DeathsMixin, W14._BigTrainMixin, _Big12Mixin,
                                   _BigMixin, mod12.World12),
               {"_wm": mod11, "_wm12": W14._ModProxy(mod12, ov), "sync_guards": W14.SYNC_GUARDS})
    w = cls(big_map, device=device, seed=seed, record=False, rules15=rules15)
    w._build_cp_groups()
    return w


## @purpose Big show world (world_big12.WorldBig12's classes, events recorded) with the exp15 mixin on top.
def WorldBig15(big_map: Any, device: str = "cpu", seed: int = 0, record: bool = True,   # noqa: N802
               rules15: Rules15 | None = None):
    from rush.world_big import _BigMixin, _world11_clone, battle_rules
    from rush.world_big12 import _Big12Mixin, _world12_clone
    big_map = _scale_match(big_map, rules15)
    rules = battle_rules(big_map)
    mod11 = _world11_clone(rules["team_size"], len(big_map.cp_positions), rules)
    mod12 = _world12_clone(mod11)
    cls = type("WorldBig15", (_Exp15Mixin, _Big12Mixin, _BigMixin, mod12.World12), {"_wm": mod11, "_wm12": mod12})
    return cls(big_map, device=device, seed=seed, record=record, rules15=rules15)
# endregion FUNC_factories
