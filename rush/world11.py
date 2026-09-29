from __future__ import annotations

# region MODULE_CONTRACT [DOMAIN(9): RL; CONCEPT(10): BatchedWorld11; TECH(9): torch]
## @modulecontract
## @purpose Simulate N independent 5v5 matches at once as tensors (experiment 11): the batched
## world of experiment 10 plus per-agent rewards for parameter sharing, a full-information
## global state for a centralized critic, an observation an agent can actually aim and navigate
## with, a real dash ability, and a render snapshot for the replay recorder.
## @scope Physics, control points, scoring, observations, global state, rewards, auto-reset,
## optional event recording. No policy, no training, no rendering.
## @input Actions (N, 10, 4) for every agent of every world: move, shoot, turn, dash
## @output StepOut: obs (N, 10, OBS_SIZE), r_indiv (N, 10), r_team (N, 2), terminated/truncated (N,),
## final_obs / final_state for the worlds that ended, info, events
## @links USES_API(9): torch; LINKS_TO: config, map_gen, spatial, docs/EXP11_CONTRACT.md
## @invariants
## - Agent slots 0..4 are blue, 5..9 are red, in every world
## - OBS_SIZE == sum of OBS_LAYOUT segment sizes; the encoder reads only OBS_LAYOUT
## - In every "tokens" segment an unused slot is exact zeros and a used slot never is (field 0 = 1)
## - r_team is zero-sum on CP events and score delta; terminal payouts are symmetric
## - final_obs / final_state are the pre-reset values of the worlds that ended this step
## - info["score"], info["frame"], info["winner"] are taken BEFORE the auto-reset: for an ended
##   world they hold the final score / frame / result of that match (the ladder reads them)
## - Every CP zone contains walkable tiles (dead CPs of map_gen are relocated, see _build_map_tables)
## - The game rules of experiment 10 are unchanged except the dash (head 3)
## @rationale
## Q: Why is the match clock a termination and not a truncation by default?
## A: The clock ends the match by the rules and pays a timeout outcome (R_WIN_TIMEOUT). With the
## A: remaining time now IN the observation (core field time_frac), the state is Markov at the
## A: limit, so it is a genuine terminal. Reporting it as truncated would make a bootstrapping
## A: trainer add gamma*V(final_obs) on top of an outcome it was already paid — double counting.
## A: `clock_is_truncation=True` switches to the other reading: the limit becomes a truncation
## A: and the timeout outcome is NOT paid, which is the consistent pair.
## Q: Why world-aligned (not rotated) egocentric grid?
## A: Every other coordinate in the observation (relative positions, velocities, flow vectors)
## A: is world-aligned. Rotating only the grid would force the network to learn a rotation to
## A: join the two; a rotated frame everywhere is a separate experiment.
## Q: Why is enemy memory shared by the team?
## A: Allies are already always visible (ALLY_COMMS) — the game models a team with voice comms.
## A: A callout of a spotted enemy is the same channel. Private memory would ask the LSTM to
## A: rebuild what the team collectively knows, from 2-second BPTT windows.
## @changes
## LAST_CHANGE: [v0.2.0] Experiment 11b: RewardConfig presets (exp11a / exp11b zero-sum combat at
##   0.02 per HP), runtime combat boost, info["aim_label"] for the auxiliary aim head.
## PREV: [v0.1.0] Initial world for experiment 11, forked from batch_world v0.2.0.
## @modulemap
## CLASS 10[N matches as tensors] => World11
## FUNC 9[One decision = ACTION_REPEAT physics frames] => step
## FUNC 9[Observation for a subset of worlds] => _build_obs
## FUNC 8[Full-information team-relative state] => global_state
## FUNC 8[Per-map BFS distance and flow toward every CP] => _build_flow_fields
## FUNC 8[Team-shared last-seen enemy memory] => _update_memory
## FUNC 7[Render state of one world] => snapshot
## @usecases
## - vec11: out = world.step(actions); r = (1 - TAU) * out.r_indiv + TAU * out.r_team[own team]
## - record11: World11(1, record=True); world.snapshot(0) every decision; world.map_geometry(m)
def _module_contract():
    pass
# endregion MODULE_CONTRACT
# GREP_SUMMARY: world11, batched world, experiment 11, obs layout, global state, dash, flow field, last seen, egocentric grid, snapshot
# STRUCTURE: ▶ build maps + flow fields → ○ step: ⚡ turn+dash trigger → ⟳ x4 frames: ⚡ move → ⚡ shoot → ⚡ bullets → ⚡ hits → ⚡ regen → ⚡ respawn → ⚡ CPs → ⚡ score → ◇ done → ⚡ vis → ⚡ memory → ⊕ obs(all) → ⚡ final_obs/state → ⚡ reset(done) → ⊕ obs(done) → ⎋ StepOut

import logging
import math
from dataclasses import dataclass, field

import numpy as np
import torch

from rush.config import (
    ACTION_REPEAT,
    BULLET_DAMAGE,
    BULLET_LIFETIME_MS,
    BULLET_RADIUS,
    BULLET_SPEED,
    COMPACT_SHARE_END,
    COMPACT_SHARE_START,
    FPS,
    PLAYER_HP,
    PLAYER_RADIUS,
    PLAYER_SPEED,
    R_CP_CAPTURE_INDIV,
    R_CP_CAPTURED,
    R_CP_LOST,
    R_DAMAGE_DEALT,
    R_DAMAGE_TAKEN,
    R_DEATH,
    R_KILL,
    R_LOSS,
    R_LOSS_TIMEOUT,
    R_SCORE_DELTA,
    R_TIMEOUT_DRAW,
    R_WIN,
    R_WIN_TIMEOUT,
    SCORE_TO_WIN,
    SHIELD_MAX,
    SHIELD_REGEN_DELAY_FRAMES,
    SHIELD_REGEN_PER_FRAME,
    SHOOT_COOLDOWN_MS,
    TURN_COARSE_DEG,
    TURN_FINE_DEG,
    VISION_RANGE,
)
from rush.spatial import WallGrid

logger = logging.getLogger(__name__)

# region BLOCK_CONSTANTS_GAME
TEAM_SIZE: int = 5
N_AGENTS: int = 2 * TEAM_SIZE
TEAM_LIVES: int = 30
RESPAWN_FRAMES: int = 120
MAX_FRAMES: int = 10800
BULLET_SLOTS: int = 160

SHOOT_COOLDOWN_FRAMES: int = max(1, (SHOOT_COOLDOWN_MS * FPS) // 1000)
BULLET_LIFETIME_FRAMES: int = max(1, (BULLET_LIFETIME_MS * FPS) // 1000)
HIT_RADIUS: float = float(BULLET_RADIUS + PLAYER_RADIUS)

N_MOVES: int = 9
N_SHOOT: int = 2
N_TURN: int = 5        # 0 = -coarse, 1 = -fine, 2 = hold, 3 = +fine, 4 = +coarse
N_ABILITY: int = 2     # 0 = nothing, 1 = dash
_TURN_DELTAS_DEG = [-TURN_COARSE_DEG, -TURN_FINE_DEG, 0.0, TURN_FINE_DEG, TURN_COARSE_DEG]

CP_CAPTURE_FRAMES: int = 120
CP_SCORE_PER_FRAME: float = 0.02

# Move directions, index 0 = stand still. Same order as env_tdm.DIRECTIONS.
_DIRS = [(0, 0), (0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1)]
# endregion BLOCK_CONSTANTS_GAME

# region BLOCK_CONSTANTS_DASH
# Dash: 3x speed for 8 frames (two decisions), then a 3 s cooldown. 108 px against 36 px of a
# normal 8-frame walk: enough to cross a lane of fire or break line of sight behind one pillar,
# not enough to outrun a bullet (24 px/frame). Per-frame displacement 13.5 px stays under the
# 30 px wall thickness, so the destination-box wall test cannot be tunnelled through.
# Direction: the held move direction; standing still dashes along the barrel.
DASH_SPEED_MULT: float = 3.0
DASH_FRAMES: int = 8
DASH_COOLDOWN_FRAMES: int = 180
# endregion BLOCK_CONSTANTS_DASH

# region BLOCK_CONSTANTS_OBS
N_WALL_RAYS: int = 8
RAY_MAX_DIST: float = 500.0
RAY_STEP: float = 8.0
RAY_SAMPLES: int = int(RAY_MAX_DIST // RAY_STEP)      # 62
LOS_SAMPLES: int = 96

VEL_NORM: float = PLAYER_SPEED * DASH_SPEED_MULT      # per-frame speed that maps to 1.0
ENTITY_RANGE: float = VISION_RANGE                    # relative entity positions / this
BULLET_OBS_RANGE: float = 600.0
BULLET_MISS_NORM: float = 100.0                        # closest-approach distance / this
BULLET_TTC_NORM: float = 30.0                          # frames to closest approach / this

LASTSEEN_HORIZON_FRAMES: int = 600                     # 10 s: older memories are dropped
GRID_RADIUS: int = 7                                   # 15 x 15 tiles around the agent
GRID_SIDE: int = 2 * GRID_RADIUS + 1
GRID_CHANNELS: int = 2                                 # walls | CP zones (signed by owner)

N_ENTITY_SLOTS: int = N_AGENTS - 1                     # nine others, no permanently empty slots
ENTITY_SLOT_SIZE: int = 15
N_BULLET_SLOTS: int = 8
BULLET_SLOT_SIZE: int = 7
N_CP_SLOTS: int = 7
CP_SLOT_SIZE: int = 14
N_LASTSEEN_SLOTS: int = TEAM_SIZE
LASTSEEN_SLOT_SIZE: int = 9
CORE_SIZE: int = 29

ETYPE_ALLY: float = -1.0
ETYPE_ENEMY: float = 1.0


def _layout() -> list[dict]:
    segs = [
        ("core", "vector", 1, CORE_SIZE),
        ("entities", "tokens", N_ENTITY_SLOTS, ENTITY_SLOT_SIZE),
        ("bullets", "tokens", N_BULLET_SLOTS, BULLET_SLOT_SIZE),
        ("cps", "tokens", N_CP_SLOTS, CP_SLOT_SIZE),
        ("lastseen", "tokens", N_LASTSEEN_SLOTS, LASTSEEN_SLOT_SIZE),
        ("grid", "grid", 1, [GRID_CHANNELS, GRID_SIDE, GRID_SIDE]),
    ]
    out, start = [], 0
    for name, kind, count, size in segs:
        flat = size[0] * size[1] * size[2] if kind == "grid" else count * size
        out.append({"name": name, "kind": kind, "start": start, "count": count, "size": size})
        start += flat
    return out


OBS_LAYOUT: list[dict] = _layout()
_SEG = {s["name"]: s for s in OBS_LAYOUT}
OBS_SIZE: int = _SEG["grid"]["start"] + GRID_CHANNELS * GRID_SIDE * GRID_SIDE

# Global state (centralized critic): per agent, own team first.
STATE_AGENT_SIZE: int = 12
STATE_CP_SIZE: int = 5
# Map descriptor, not a map-id one-hot: a one-hot of 8 crashed (scatter_ out of bounds) on a pool of 20+
# maps and says nothing about a procedural map. 8 static floats per map instead — see _map_descriptor.
N_MAPS_STATE: int = 8
STATE_SIZE: int = N_AGENTS * STATE_AGENT_SIZE + N_CP_SLOTS * STATE_CP_SIZE + 6 + N_MAPS_STATE
# endregion BLOCK_CONSTANTS_OBS


# region CLASS_StepOut
@dataclass
class StepOut:
    obs: torch.Tensor                 # (N, 10, OBS_SIZE)
    r_indiv: torch.Tensor             # (N, 10)
    r_team: torch.Tensor              # (N, 2)
    terminated: torch.Tensor          # (N,) bool
    truncated: torch.Tensor           # (N,) bool
    final_obs: torch.Tensor           # (N, 10, OBS_SIZE), valid where done
    final_state: torch.Tensor         # (N, 2, STATE_SIZE), valid where done
    info: dict
    events: list | None = field(default=None)
# endregion CLASS_StepOut


# region CLASS_RewardConfig
## @purpose Weights of the individual combat channel. CP/score/terminal rewards (team channel and
## capture credit) are not part of it and stay as in experiment 10.
## @rationale
## Q: Why zero-sum combat and 0.02 per HP (experiment 11b)?
## A: exp11a (481M decisions) learned objectives but not combat: a full health bar of damage paid 0.5
## A: against 10 for a win and +3 per capture, so combat drowned. OpenAI Five (Dota 2, appendix G)
## A: pays ~2 per full health bar against 5 for a win, and makes every reward zero-sum by subtracting
## A: the enemy team's mean — dealing damage and avoiding it become worth the same.
@dataclass(frozen=True)
class RewardConfig:
    damage_dealt: float      # per HP (shield included) the agent dealt
    damage_taken: float      # per HP (shield included) the agent took
    kill: float
    death: float
    zero_sum: bool           # combat_i -= mean combat of the enemy team (OpenAI Five)


REWARD_PRESETS: dict[str, RewardConfig] = {
    "exp11a": RewardConfig(damage_dealt=R_DAMAGE_DEALT, damage_taken=R_DAMAGE_TAKEN, kill=R_KILL, death=R_DEATH, zero_sum=False),
    "exp11b": RewardConfig(damage_dealt=0.02, damage_taken=-0.02, kill=1.0, death=-1.0, zero_sum=True),
}
DEFAULT_REWARD_PRESET: str = "exp11b"
# endregion CLASS_RewardConfig


# region CLASS_World11
## @purpose N matches as tensors; one step() advances every match by one decision.
## @complexity 9
class World11:
    def __init__(
        self,
        n_worlds: int,
        device: str = "cpu",
        seed: int = 0,
        compact_share: float | None = None,
        record: bool = False,
        clock_is_truncation: bool = False,
        map_pool: str = "maps11",
        reward_preset: str | RewardConfig = DEFAULT_REWARD_PRESET,
    ) -> None:
        self.n = n_worlds
        self.reward_cfg = reward_preset if isinstance(reward_preset, RewardConfig) else REWARD_PRESETS[reward_preset]
        self.combat_boost: float = 1.0       # multiplies the combat channel; the trainer anneals it
        self._vis_cache: torch.Tensor | None = None   # visibility matching the last observation
        logger.info(f"[IMP:9][World11.__init__][INIT] reward preset={reward_preset if isinstance(reward_preset, str) else 'custom'} {self.reward_cfg} [VALUE]")
        # "maps11" (20 fair maps), "legacy" (map_gen as frozen for exp 9/10), "maps11+proc:N" (+N procedural from seed).
        self.map_pool = map_pool
        self._seed = seed
        self.device = torch.device(device)
        self.gen = torch.Generator(device="cpu").manual_seed(seed)
        self._fixed_compact_share = compact_share
        self.progress: float = 0.0
        self.record = record
        self.auto_reset = True
        self.clock_is_truncation = clock_is_truncation
        self._events: list[list[dict]] = [[] for _ in range(n_worlds)] if record else []

        self._build_map_tables()
        self._build_flow_fields()

        f = lambda *s: torch.zeros(*s, device=self.device)                      # noqa: E731
        b = lambda *s: torch.zeros(*s, dtype=torch.bool, device=self.device)    # noqa: E731
        i = lambda *s: torch.zeros(*s, dtype=torch.int32, device=self.device)   # noqa: E731

        n, a, bs, c = n_worlds, N_AGENTS, BULLET_SLOTS, N_CP_SLOTS
        self.map_idx = torch.zeros(n, dtype=torch.long, device=self.device)
        self.pos = f(n, a, 2)
        self.vel = f(n, a, 2)                # mean per-frame displacement over the last decision
        self.angle, self.hp = f(n, a), f(n, a)
        self.shield = f(n, a)
        self.since_dmg = i(n, a)
        self.alive, self.waiting = b(n, a), b(n, a)
        self.cooldown, self.respawn_timer = i(n, a), i(n, a)
        self.dash_left, self.dash_cd = i(n, a), i(n, a)

        self.b_pos, self.b_vel = f(n, bs, 2), f(n, bs, 2)
        self.b_owner, self.b_age = i(n, bs), i(n, bs)
        self.b_alive = b(n, bs)

        self.cp_owner, self.cp_cap_team = i(n, c), i(n, c)   # 0 neutral, 1 blue, 2 red
        self.cp_progress = f(n, c)

        self.score = f(n, 2)
        self.lives = i(n, 2)
        self.frame = i(n)

        # Team-shared memory of enemies: [:, t, k] = what team t knows about enemy k of team 1-t.
        self.mem_pos, self.mem_vel = f(n, 2, TEAM_SIZE, 2), f(n, 2, TEAM_SIZE, 2)
        self.mem_hp = f(n, 2, TEAM_SIZE)
        self.mem_age = i(n, 2, TEAM_SIZE)
        self.mem_valid = b(n, 2, TEAM_SIZE)

        self._dirs = torch.tensor(_DIRS, dtype=torch.float32, device=self.device)
        norm = self._dirs.norm(dim=1, keepdim=True).clamp(min=1.0)
        self._dirs_unit = self._dirs / norm
        self._turn_deltas = torch.tensor([math.radians(d) for d in _TURN_DELTAS_DEG], dtype=torch.float32, device=self.device)
        ray_ang = torch.arange(N_WALL_RAYS, device=self.device) * (2 * math.pi / N_WALL_RAYS)
        self._ray_dir = torch.stack([ray_ang.cos(), ray_ang.sin()], dim=1)
        self._agent_team = torch.zeros(N_AGENTS, dtype=torch.long, device=self.device)
        self._agent_team[TEAM_SIZE:] = 1
        g = torch.arange(-GRID_RADIUS, GRID_RADIUS + 1, device=self.device)
        self._grid_dr, self._grid_dc = torch.meshgrid(g, g, indexing="ij")     # (S, S)

        self.reset_worlds(torch.ones(n, dtype=torch.bool, device=self.device))
        self._last_vis: torch.Tensor | None = None
        logger.info(
            f"[IMP:9][World11.__init__][INIT] worlds={n}, device={self.device}, obs={OBS_SIZE}, "
            f"state={STATE_SIZE}, maps={self.n_maps}, record={record}, clock_is_truncation={clock_is_truncation} [VALUE]"
        )
    # region FUNC__build_map_tables
    ## @purpose Flatten the hand-built map pool into padded tensors indexable by map id.
    def _build_map_tables(self) -> None:
        from rush.maps11 import build_pool
        pool = build_pool(self.map_pool, self._seed)
        self._pool = pool
        self.n_maps = len(pool)
        rmax = max(m.rows for m in pool)
        cmax = max(m.cols for m in pool)
        smax = 0

        grids = torch.ones(self.n_maps, rmax, cmax, dtype=torch.bool)  # padding counts as wall
        counts = torch.zeros(self.n_maps, 2, dtype=torch.long)
        arena = torch.zeros(self.n_maps, 2)
        cp_xy = torch.zeros(self.n_maps, N_CP_SLOTS, 2)
        cp_rc = torch.zeros(self.n_maps, N_CP_SLOTS, 2, dtype=torch.long)
        cp_r = torch.zeros(self.n_maps)
        compact = torch.zeros(self.n_maps, dtype=torch.bool)

        raw_spawns: list[list[torch.Tensor]] = [[], []]
        for mi, m in enumerate(pool):
            walls = m.build_walls()
            grid = WallGrid(walls, m.cols, m.rows, m.tile_size)
            for r in range(m.rows):
                for c in range(m.cols):
                    grids[mi, r, c] = bool(grid.grid[r][c])
            arena[mi, 0], arena[mi, 1] = float(m.arena_w), float(m.arena_h)
            cp_r[mi] = float(m.cp_radius)
            compact[mi] = bool(m.compact)
            ts = m.tile_size
            for k, (r, c) in enumerate(m.cp_positions):
                cp_xy[mi, k, 0] = c * ts + ts / 2
                cp_xy[mi, k, 1] = r * ts + ts / 2
                cp_rc[mi, k, 0], cp_rc[mi, k, 1] = r, c

            safe = grid.safe_spawn_tiles()
            for team, zone in ((0, m.spawn_zone_a), (1, m.spawn_zone_b)):
                tiles = [(r, c) for r, c in safe if zone[1] <= r <= zone[2] and zone[3] <= c <= zone[4]]
                if not tiles:
                    tiles = safe[: len(safe) // 2] if team == 0 else safe[len(safe) // 2:]
                pts = torch.tensor([[c * ts + ts / 2, r * ts + ts / 2] for r, c in tiles], dtype=torch.float32)
                raw_spawns[team].append(pts)
                counts[mi, team] = len(tiles)
                smax = max(smax, len(tiles))

        spawn_tbl = torch.zeros(self.n_maps, 2, smax, 2)
        for team in (0, 1):
            for mi, pts in enumerate(raw_spawns[team]):
                spawn_tbl[mi, team, : pts.shape[0]] = pts

        self.tile_size = float(pool[0].tile_size)
        ts = self.tile_size
        # Relocate dead control points. Measured at build: 6 of 49 CPs (the centre point of every
        # map but Ring) sit inside a solid wall block — no agent centre can come within the capture
        # radius (Pit/Alley: 92 px vs r=70; Arena/Diagonal/Highway: ~180 px vs r=150), so in
        # experiments 9-10 that point could never be captured. map_gen is frozen for those
        # experiments; world11 moves such a centre to the nearest free tile centre instead.
        self.relocated_cps: list[tuple[str, int, tuple[float, float], tuple[float, float]]] = []
        for mi, m in enumerate(pool):
            fr = (~grids[mi, : m.rows, : m.cols]).nonzero().float()              # (F, 2) row, col
            fx, fy = fr[:, 1] * ts + ts / 2, fr[:, 0] * ts + ts / 2
            for k in range(N_CP_SLOTS):
                d = ((fx - cp_xy[mi, k, 0]) ** 2 + (fy - cp_xy[mi, k, 1]) ** 2).sqrt()
                if bool((d <= cp_r[mi]).any()):
                    continue
                j = int(d.argmin())
                old = (float(cp_xy[mi, k, 0]), float(cp_xy[mi, k, 1]))
                cp_xy[mi, k, 0], cp_xy[mi, k, 1] = fx[j], fy[j]
                cp_rc[mi, k, 0], cp_rc[mi, k, 1] = int(fr[j, 0]), int(fr[j, 1])
                self.relocated_cps.append((m.name, k, old, (float(fx[j]), float(fy[j]))))
        logger.info(f"[IMP:9][World11._build_map_tables][BUILD] relocated_cps={len(self.relocated_cps)}: "
                    + ", ".join(f"{nm}#{k} {o}->{nw}" for nm, k, o, nw in self.relocated_cps) + " [VALUE]")
        # CP zone index per tile: nearest CP whose circle covers the tile centre, else -1.
        rr = torch.arange(rmax).view(rmax, 1).float() * ts + ts / 2
        cc = torch.arange(cmax).view(1, cmax).float() * ts + ts / 2
        zone = torch.full((self.n_maps, rmax, cmax), -1, dtype=torch.long)
        for mi in range(self.n_maps):
            d = ((cc.unsqueeze(0) - cp_xy[mi, :, 0].view(-1, 1, 1)) ** 2 + (rr.unsqueeze(0) - cp_xy[mi, :, 1].view(-1, 1, 1)) ** 2).sqrt()
            dmin, arg = d.min(0)
            zone[mi] = torch.where(dmin <= cp_r[mi], arg, torch.full_like(arg, -1))

        # Padded copies so an egocentric window never indexes outside: walls / no zone around.
        pad = GRID_RADIUS
        grid_pad = torch.ones(self.n_maps, rmax + 2 * pad, cmax + 2 * pad, dtype=torch.bool)
        grid_pad[:, pad: pad + rmax, pad: pad + cmax] = grids
        zone_pad = torch.full((self.n_maps, rmax + 2 * pad, cmax + 2 * pad), -1, dtype=torch.long)
        zone_pad[:, pad: pad + rmax, pad: pad + cmax] = zone

        self.map_grid = grids.to(self.device)
        self.map_grid_pad = grid_pad.to(self.device)
        self.map_zone_pad = zone_pad.to(self.device)
        self.map_arena = arena.to(self.device)
        self.map_cp_xy = cp_xy.to(self.device)
        self.map_cp_rc = cp_rc
        self.map_cp_r = cp_r.to(self.device)
        self.map_compact = compact.to(self.device)
        self.map_spawn = spawn_tbl.to(self.device)
        self.map_spawn_n = counts.to(self.device)
        self.map_diag = self.map_arena.pow(2).sum(1).sqrt()
        self._compact_ids = torch.nonzero(self.map_compact).flatten()
        self._large_ids = torch.nonzero(~self.map_compact).flatten()
        # Host copies for _reset_idx: map pick and spawn selection are computed next to the CPU generator.
        self._compact_ids_cpu = self._compact_ids.cpu()
        self._large_ids_cpu = self._large_ids.cpu()
        self._map_spawn_n_cpu = counts.cpu()

        # Static per-map descriptor for the critic (global_state), all in [-1, 1].
        desc = torch.zeros(self.n_maps, N_MAPS_STATE)
        for mi, m in enumerate(pool):
            free = ~grids[mi, 1: m.rows - 1, 1: m.cols - 1]
            r3, c3 = m.rows // 3, m.cols // 3
            centre = ~grids[mi, r3: m.rows - r3, c3: m.cols - c3]
            desc[mi] = torch.tensor([
                m.arena_w / (141 * ts) * 2 - 1, m.arena_h / (121 * ts) * 2 - 1,
                1.0 if m.compact else -1.0,
                float(free.float().mean()) * 2 - 1, float(centre.float().mean()) * 2 - 1,
                min(1.0, m.cp_radius / 150.0) * 2 - 1,
                1.0 if getattr(m, "symmetry", "") == "rot180" else -1.0,
                1.0 if getattr(m, "kind", "") == "procedural" else -1.0,
            ])
        self.map_desc = desc.to(self.device)
    # endregion FUNC__build_map_tables

    # region FUNC__build_flow_fields
    ## @purpose Shortest walkable distance (px) from every tile into every CP capture zone, and the unit
    ## direction of the first step along that path, per map.
    ## @rationale 8-connected grid, diagonal allowed only when both orthogonal neighbours are
    ## free (no corner cutting through a wall). Relaxation is a batched min-plus sweep over all
    ## maps x CPs at once, run to a fixed point at construction; it is a one-off cost.
    def _build_flow_fields(self) -> None:
        grid = self.map_grid.cpu()                                               # (M, R, C) True = wall
        m, rows, cols = grid.shape
        ts = self.tile_size
        inf = float("inf")
        free = ~grid
        # Target = every free tile whose centre lies inside the capture circle, not the centre
        # tile: 13 of the 49 CP centres sit on a wall (a pillar in the middle of the circle).
        # Distance is therefore "walk until you are in the zone", which is what capturing needs.
        rr = torch.arange(rows).view(rows, 1).float() * ts + ts / 2
        cc = torch.arange(cols).view(1, cols).float() * ts + ts / 2
        cp_xy = self.map_cp_xy.cpu()
        cp_r = self.map_cp_r.cpu()
        in_zone = ((cc.view(1, 1, 1, cols) - cp_xy[..., 0].view(m, N_CP_SLOTS, 1, 1)) ** 2
                   + (rr.view(1, 1, rows, 1) - cp_xy[..., 1].view(m, N_CP_SLOTS, 1, 1)) ** 2).sqrt() <= cp_r.view(m, 1, 1, 1)
        in_zone = in_zone & free.unsqueeze(1)
        self.flow_zone = in_zone.to(self.device)                                 # (M, K, R, C)
        dist = torch.where(in_zone, torch.zeros(()), torch.full((), inf)).expand(m, N_CP_SLOTS, rows, cols).clone()
        empty_zones = int((in_zone.flatten(2).sum(-1) == 0).sum())

        nbrs = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]

        def shifted(t: torch.Tensor, dr: int, dc: int, fill) -> torch.Tensor:
            # value at (r + dr, c + dc), `fill` outside
            out = torch.full_like(t, fill)
            rs, re = max(0, -dr), rows - max(0, dr)
            cs, ce = max(0, -dc), cols - max(0, dc)
            out[..., rs:re, cs:ce] = t[..., rs + dr: re + dr, cs + dc: ce + dc]
            return out

        free_f = free.unsqueeze(1)                                               # (M, 1, R, C)
        allowed = []
        for dr, dc in nbrs:
            ok = shifted(free, dr, dc, False)
            if dr != 0 and dc != 0:
                ok = ok & shifted(free, dr, 0, False) & shifted(free, 0, dc, False)
            allowed.append((ok & free).unsqueeze(1))
        costs = [ts * (math.sqrt(2.0) if dr and dc else 1.0) for dr, dc in nbrs]

        iters = 0
        while True:
            iters += 1
            best = dist
            for (dr, dc), ok, cost in zip(nbrs, allowed, costs):
                cand = shifted(dist, dr, dc, inf) + cost
                best = torch.where(ok & (cand < best), cand, best)
            best = torch.where(free_f, best, torch.full_like(best, inf))
            if torch.equal(best, dist):
                break
            dist = best

        # First step: neighbour minimising cost + neighbour distance.
        cand_all = torch.stack(
            [torch.where(ok, shifted(dist, dr, dc, inf) + cost, torch.full_like(dist, inf)) for (dr, dc), ok, cost in zip(nbrs, allowed, costs)],
            dim=0,
        )                                                                         # (8, M, K, R, C)
        best_val, best_i = cand_all.min(0)
        dvec = torch.tensor([[dc, dr] for dr, dc in nbrs], dtype=torch.float32)
        dvec = dvec / dvec.norm(dim=1, keepdim=True)
        flow = dvec[best_i]                                                       # (M, K, R, C, 2)
        stop = (dist == 0) | torch.isinf(best_val) | torch.isinf(dist)
        flow = torch.where(stop.unsqueeze(-1), torch.zeros_like(flow), flow)

        reach = torch.isfinite(dist)
        self.flow_dist = dist.to(self.device)                                    # (M, K, R, C) px, inf unreachable
        self.flow_dir = flow.to(self.device)                                     # (M, K, R, C, 2)
        logger.info(
            f"[IMP:9][World11._build_flow_fields][BUILD] maps={m}, cps={N_CP_SLOTS}, iterations={iters}, empty_zones={empty_zones}, "
            f"reachable_free_tiles={int((reach & free_f).sum())}/{int(free_f.sum()) * N_CP_SLOTS} [VALUE]"
        )
    # endregion FUNC__build_flow_fields

    def set_progress(self, progress: float) -> None:
        self.progress = float(progress)

    def set_compact_share(self, share: float | None) -> None:
        """Pin the compact-map share for future resets (the trainer's schedule); None returns to the
        progress-driven COMPACT_SHARE_START -> END ramp."""
        self._fixed_compact_share = None if share is None else min(1.0, max(0.0, float(share)))

    # region FUNC_reset_worlds
    ## @purpose Restart the masked worlds: new map under the curriculum, fresh spawns, empty bullets, neutral points, empty memory.
    def reset_worlds(self, mask: torch.Tensor) -> None:
        idx_cpu = np.flatnonzero(mask.cpu().numpy())
        if idx_cpu.size == 0:
            return
        self._vis_cache = None          # positions changed outside step(): the next label recomputes
        self._reset_idx(torch.as_tensor(idx_cpu, dtype=torch.long, device=self.device), int(idx_cpu.size))

    ## @purpose Reset the given worlds. The random draws happen on the CPU generator in the same order
    ## as before (so seeded runs are unchanged), the map pick and spawn selection are computed there too,
    ## and the result crosses to the device in ONE copy — it used to be five blocking copies plus .item().
    def _reset_idx(self, idx: torch.Tensor, k: int) -> None:
        share = self._fixed_compact_share
        if share is None:
            t = min(1.0, max(0.0, self.progress))
            share = COMPACT_SHARE_START + (COMPACT_SHARE_END - COMPACT_SHARE_START) * t
        take_compact = torch.rand(k, generator=self.gen) < share
        pick_c = self._compact_ids_cpu[torch.randint(len(self._compact_ids_cpu), (k,), generator=self.gen)]
        pick_l = self._large_ids_cpu[torch.randint(len(self._large_ids_cpu), (k,), generator=self.gen)]
        m_cpu = torch.where(take_compact, pick_c, pick_l)
        sels = []
        for team in (0, 1):
            n_tiles = self._map_spawn_n_cpu[m_cpu, team]
            r = torch.rand(k, TEAM_SIZE, generator=self.gen)
            sels.append((r * n_tiles.unsqueeze(1).float()).long().clamp(min=0))
        packed = torch.cat([m_cpu.unsqueeze(1), sels[0], sels[1]], dim=1).to(self.device)   # (k, 1 + 2*TEAM_SIZE)
        m = packed[:, 0]
        self.map_idx[idx] = m
        for team in (0, 1):
            sel = packed[:, 1 + team * TEAM_SIZE: 1 + (team + 1) * TEAM_SIZE]
            pts = self.map_spawn[m.unsqueeze(1), team, sel]
            self.pos[idx, team * TEAM_SIZE: (team + 1) * TEAM_SIZE] = pts

        self.vel[idx] = 0.0
        self.angle[idx] = 0.0
        self.hp[idx] = float(PLAYER_HP)
        self.shield[idx] = float(SHIELD_MAX)
        self.since_dmg[idx] = 0
        self.alive[idx] = True
        self.waiting[idx] = False
        self.cooldown[idx] = 0
        self.respawn_timer[idx] = 0
        self.dash_left[idx] = 0
        self.dash_cd[idx] = 0
        self.b_alive[idx] = False
        self.b_age[idx] = 0
        self.cp_owner[idx] = 0
        self.cp_cap_team[idx] = 0
        self.cp_progress[idx] = 0.0
        self.score[idx] = 0.0
        self.lives[idx] = TEAM_LIVES
        self.frame[idx] = 0
        self.mem_valid[idx] = False
        self.mem_age[idx] = 0
        self.mem_pos[idx] = 0.0
        self.mem_vel[idx] = 0.0
        self.mem_hp[idx] = 0.0
    # endregion FUNC_reset_worlds

    # region FUNC__rect_hits_wall
    ## @purpose Wall test for an axis-aligned box around each point, through the tile grid.
    def _rect_hits_wall(self, w: torch.Tensor, x: torch.Tensor, y: torch.Tensor, half: float) -> torch.Tensor:
        ts = self.tile_size
        arena = self.map_arena[self.map_idx[w]]
        x0, x1 = x - half, x + half - 1
        y0, y1 = y - half, y + half - 1
        hit = (x0 < 0) | (y0 < 0) | (x1 >= arena[..., 0]) | (y1 >= arena[..., 1])
        for cx in (x0, x1):
            for cy in (y0, y1):
                col = (cx / ts).long().clamp(min=0, max=self.map_grid.shape[2] - 1)
                row = (cy / ts).long().clamp(min=0, max=self.map_grid.shape[1] - 1)
                hit = hit | self.map_grid[self.map_idx[w], row, col]
        return hit
    # endregion FUNC__rect_hits_wall

    def _wall_at(self, m: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        ts = self.tile_size
        col = (x / ts).long().clamp(min=0, max=self.map_grid.shape[2] - 1)
        row = (y / ts).long().clamp(min=0, max=self.map_grid.shape[1] - 1)
        return self.map_grid[m, row, col]

    # region FUNC_step
    ## @purpose One decision for every world: hold the action for ACTION_REPEAT frames, score it,
    ## observe, capture final values of the ended worlds, auto-reset them.
    ## @io actions (N, 10, 4) -> StepOut
    ## @complexity 9
    def step(self, actions: torch.Tensor) -> StepOut:
        n = self.n
        dev = self.device
        dmg_dealt = torch.zeros(n, N_AGENTS, device=dev)
        dmg_taken = torch.zeros(n, N_AGENTS, device=dev)
        kills = torch.zeros(n, N_AGENTS, device=dev)
        deaths = torch.zeros(n, N_AGENTS, device=dev)
        cp_gain = torch.zeros(n, 2, device=dev)
        cap_part = torch.zeros(n, N_AGENTS, device=dev)
        score_before = self.score.clone()
        if self.record:
            self._events = [[] for _ in range(n)]

        actions = actions.to(dev).long()
        move_a, shoot_a, turn_a, dash_a = actions[..., 0], actions[..., 1], actions[..., 2], actions[..., 3]
        # Label of the state the policy saw (pre-turn barrel, visibility of the last observation).
        aim_label = self._aim_label(self._vis_cache if self._vis_cache is not None else self._visibility(None))

        self._turn(turn_a)
        self._trigger_dash(dash_a)
        start_pos = self.pos.clone()
        respawned = torch.zeros(n, N_AGENTS, dtype=torch.bool, device=dev)
        for _ in range(ACTION_REPEAT):
            self.frame += 1
            self.cooldown = (self.cooldown - 1).clamp(min=0)
            self.dash_cd = (self.dash_cd - 1).clamp(min=0)
            self._move(move_a)
            self._shoot(shoot_a)
            self._advance_bullets()
            self._resolve_hits(dmg_dealt, dmg_taken, kills, deaths)
            self._regen_shield()
            respawned |= self._respawn()
            self._update_control_points(cp_gain, cap_part)
            self._score_income()
        # A respawn teleports: that displacement is not a velocity.
        vel = (self.pos - start_pos) / ACTION_REPEAT
        self.vel = torch.where((respawned | ~self.alive).unsqueeze(-1), torch.zeros_like(vel), vel)

        blue_dead = (~self.alive[:, :TEAM_SIZE]).all(1) & (self.lives[:, 0] <= 0)
        red_dead = (~self.alive[:, TEAM_SIZE:]).all(1) & (self.lives[:, 1] <= 0)
        won = self.score >= SCORE_TO_WIN
        decided = won.any(1) | blue_dead | red_dead
        clock = (self.frame >= MAX_FRAMES) & ~decided
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
        # kills (N, 10) and captures (N, 2) are this decision's counts — the trainer sums them per match.
        # The one host sync of a step: done and truncated travel together; the trainer side reuses them
        # (done_cpu, any_truncated) instead of asking the device again.
        flags = torch.stack([done, truncated]).cpu().numpy()
        done_cpu = flags[0]
        info = {"winner": winner, "score": self.score.clone(), "frame": self.frame.clone(), "done": done,
                "kills": kills, "captures": cp_gain, "done_cpu": done_cpu, "any_truncated": bool(flags[1].any()),
                "aim_label": aim_label}

        vis = self._visibility(None)
        self._update_memory(vis)
        obs = self._build_obs(None, vis)
        final_obs = obs.clone()
        final_state = self.global_state()
        events = self._events if self.record else None
        idx_cpu = np.flatnonzero(done_cpu)
        if self.record:
            for w in idx_cpu.tolist():
                events[w].append({"type": "match_end", "frame": int(self.frame[w]), "winner": int(winner[w]),
                                  "score": [float(self.score[w, 0]), float(self.score[w, 1])], "truncated": bool(truncated[w])})

        # auto_reset=False keeps a finished world frozen on its final state (the replay recorder); it used to
        # monkeypatch reset_worlds, which silently stopped working when the reset moved to _reset_idx.
        if idx_cpu.size and self.auto_reset:
            idx = torch.as_tensor(idx_cpu, dtype=torch.long, device=dev)
            self._reset_idx(idx, int(idx_cpu.size))
            vis_new = self._visibility(idx)
            self._update_memory(vis_new, idx)
            obs[idx] = self._build_obs(idx, vis_new)
            vis[idx] = vis_new
            logger.debug(f"[IMP:6][World11.step][EXEC] reset {idx.numel()} worlds [VALUE]")
        self._vis_cache = vis

        return StepOut(obs=obs, r_indiv=r_indiv, r_team=r_team, terminated=terminated, truncated=truncated,
                       final_obs=final_obs, final_state=final_state, info=info, events=events)
    # endregion FUNC_step

    # region FUNC__trigger_dash
    ## @purpose Start a dash for every alive agent that asks for one and is off cooldown.
    def _trigger_dash(self, dash_a: torch.Tensor) -> None:
        go = self.alive & (dash_a == 1) & (self.dash_cd <= 0) & (self.dash_left <= 0)
        # No early "if not go.any()": that was a host sync on every step; the where() below is a no-op
        # for an all-False mask.
        self.dash_left = torch.where(go, torch.full_like(self.dash_left, DASH_FRAMES), self.dash_left)
        self.dash_cd = torch.where(go, torch.full_like(self.dash_cd, DASH_COOLDOWN_FRAMES), self.dash_cd)
        if self.record:
            for w, ag in torch.nonzero(go).tolist():
                self._events[w].append({"type": "dash", "frame": int(self.frame[w]), "agent": ag})
    # endregion FUNC__trigger_dash

    def _move(self, move_a: torch.Tensor) -> None:
        dashing = (self.dash_left > 0) & self.alive
        unit = self._dirs_unit[move_a]                                           # (N, A, 2)
        facing = torch.stack([self.angle.cos(), self.angle.sin()], dim=-1)
        standing = (move_a == 0).unsqueeze(-1)
        dash_dir = torch.where(standing, facing, unit)
        speed = torch.where(dashing, torch.full_like(self.angle, PLAYER_SPEED * DASH_SPEED_MULT), torch.full_like(self.angle, PLAYER_SPEED))
        step = torch.where(dashing.unsqueeze(-1), dash_dir, unit) * speed.unsqueeze(-1)
        self.dash_left = (self.dash_left - 1).clamp(min=0)

        w = torch.arange(self.n, device=self.device).unsqueeze(1).expand(-1, N_AGENTS)
        arena = self.map_arena[self.map_idx].unsqueeze(1)
        new_x = self.pos[..., 0] + step[..., 0]
        blocked = self._rect_hits_wall(w, new_x, self.pos[..., 1], PLAYER_RADIUS)
        ok = self.alive & ~blocked
        self.pos[..., 0] = torch.where(ok, new_x.clamp(min=PLAYER_RADIUS).min(arena[..., 0] - PLAYER_RADIUS), self.pos[..., 0])
        new_y = self.pos[..., 1] + step[..., 1]
        blocked = self._rect_hits_wall(w, self.pos[..., 0], new_y, PLAYER_RADIUS)
        ok = self.alive & ~blocked
        self.pos[..., 1] = torch.where(ok, new_y.clamp(min=PLAYER_RADIUS).min(arena[..., 1] - PLAYER_RADIUS), self.pos[..., 1])

    def _turn(self, turn_a: torch.Tensor) -> None:
        delta = self._turn_deltas[turn_a]
        ang = self.angle + torch.where(self.alive, delta, torch.zeros_like(delta))
        self.angle = (ang + math.pi) % (2 * math.pi) - math.pi

    # region FUNC__shoot
    ## @purpose Spawn bullets for every agent that fired, into free slots of its world's ring.
    def _shoot(self, shoot_a: torch.Tensor) -> None:
        firing = self.alive & (shoot_a == 1) & (self.cooldown <= 0)
        if not firing.any():
            return
        self.cooldown = torch.where(firing, torch.full_like(self.cooldown, SHOOT_COOLDOWN_FRAMES), self.cooldown)

        muzzle = float(PLAYER_RADIUS + BULLET_RADIUS + 2)
        cos_a, sin_a = self.angle.cos(), self.angle.sin()
        bx = self.pos[..., 0] + cos_a * muzzle
        by = self.pos[..., 1] + sin_a * muzzle

        free_rank = torch.argsort(self.b_alive.int(), dim=1, stable=True)
        shooter_rank = firing.cumsum(1) - 1
        capacity = (~self.b_alive).sum(1, keepdim=True)
        placeable = firing & (shooter_rank < capacity) & (shooter_rank < BULLET_SLOTS)
        if not placeable.any():
            return

        slot = free_rank.gather(1, shooter_rank.clamp(min=0, max=BULLET_SLOTS - 1))
        wi, ai = torch.nonzero(placeable, as_tuple=True)
        si = slot[wi, ai]
        self.b_pos[wi, si, 0] = bx[wi, ai]
        self.b_pos[wi, si, 1] = by[wi, ai]
        self.b_vel[wi, si, 0] = cos_a[wi, ai] * BULLET_SPEED
        self.b_vel[wi, si, 1] = sin_a[wi, ai] * BULLET_SPEED
        self.b_owner[wi, si] = ai.int()
        self.b_age[wi, si] = 0
        self.b_alive[wi, si] = True
        if self.record:
            for w, ag in zip(wi.tolist(), ai.tolist()):
                self._events[w].append({"type": "shot", "frame": int(self.frame[w]), "agent": ag})
    # endregion FUNC__shoot

    def _advance_bullets(self) -> None:
        if not self.b_alive.any():
            return
        self.b_pos = self.b_pos + self.b_vel
        self.b_age = self.b_age + 1
        w = torch.arange(self.n, device=self.device).unsqueeze(1).expand(-1, BULLET_SLOTS)
        wall = self._rect_hits_wall(w, self.b_pos[..., 0], self.b_pos[..., 1], BULLET_RADIUS)
        expired = self.b_age > BULLET_LIFETIME_FRAMES
        self.b_alive = self.b_alive & ~wall & ~expired

    # region FUNC__resolve_hits
    ## @purpose Bullet-vs-player hits, damage through shield then HP, deaths and kill credit.
    def _resolve_hits(self, dmg_dealt, dmg_taken, kills, deaths) -> None:
        if not self.b_alive.any():
            return
        d = (self.b_pos.unsqueeze(2) - self.pos.unsqueeze(1)).pow(2).sum(-1).sqrt()
        owner_team = self._agent_team[self.b_owner.long()]
        target_team = self._agent_team.view(1, 1, N_AGENTS)
        hostile = owner_team.unsqueeze(2) != target_team
        hit = (d < HIT_RADIUS) & hostile & self.alive.unsqueeze(1) & self.b_alive.unsqueeze(2)
        if not hit.any():
            return

        first_target = hit.float().argmax(dim=2)
        bullet_hits = hit.any(2)
        onehot = torch.zeros_like(hit, dtype=torch.float32)
        onehot.scatter_(2, first_target.unsqueeze(2), bullet_hits.float().unsqueeze(2))

        incoming = onehot.sum(1) * float(BULLET_DAMAGE)
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
        dmg_dealt.scatter_add_(1, shooter, credit.sum(2))
        if self.record:
            for w, bi in torch.nonzero(bullet_hits).tolist():
                tgt = int(first_target[w, bi])
                self._events[w].append({"type": "hit", "frame": int(self.frame[w]), "shooter": int(self.b_owner[w, bi]),
                                        "target": tgt, "x": float(self.b_pos[w, bi, 0]), "y": float(self.b_pos[w, bi, 1])})

        self.b_alive = self.b_alive & ~bullet_hits

        died = self.alive & (self.hp <= 0)
        if died.any():
            deaths += died.float()
            self.alive = self.alive & ~died
            killer_bullet = (onehot * died.unsqueeze(1).float()).argmax(dim=1)
            killer = self.b_owner.long().gather(1, killer_bullet)
            kills.scatter_add_(1, killer, died.float())
            if self.record:
                for w, v in torch.nonzero(died).tolist():
                    self._events[w].append({"type": "kill", "frame": int(self.frame[w]), "killer": int(killer[w, v]), "victim": v})
            for team in (0, 1):
                lo, hi = team * TEAM_SIZE, (team + 1) * TEAM_SIZE
                self.score[:, 1 - team] += died[:, lo:hi].sum(1).float()
                can = self.lives[:, team] > 0
                n_new = (died[:, lo:hi].sum(1) * can).int()
                self.lives[:, team] = (self.lives[:, team] - n_new).clamp(min=0)
            self.waiting = self.waiting | died
            self.respawn_timer = torch.where(died, torch.full_like(self.respawn_timer, RESPAWN_FRAMES), self.respawn_timer)
            self.dash_left = torch.where(died, torch.zeros_like(self.dash_left), self.dash_left)
    # endregion FUNC__resolve_hits

    def _regen_shield(self) -> None:
        self.since_dmg = self.since_dmg + 1
        can = self.alive & (self.since_dmg >= SHIELD_REGEN_DELAY_FRAMES) & (self.shield < SHIELD_MAX)
        if can.any():
            self.shield = torch.where(can, (self.shield + SHIELD_REGEN_PER_FRAME).clamp(max=SHIELD_MAX), self.shield)

    # region FUNC__respawn
    ## @purpose Bring back agents whose timer ran out; returns the mask of agents respawned this frame.
    def _respawn(self) -> torch.Tensor:
        none = torch.zeros(self.n, N_AGENTS, dtype=torch.bool, device=self.device)
        if not self.waiting.any():
            return none
        self.respawn_timer = torch.where(self.waiting, self.respawn_timer - 1, self.respawn_timer)
        ready = self.waiting & (self.respawn_timer <= 0)
        if not ready.any():
            return none
        m = self.map_idx
        for team in (0, 1):
            lo, hi = team * TEAM_SIZE, (team + 1) * TEAM_SIZE
            sub = ready[:, lo:hi]
            if not sub.any():
                continue
            n_tiles = self.map_spawn_n[m, team].unsqueeze(1).float()
            r = torch.rand(self.n, TEAM_SIZE, generator=self.gen).to(self.device)
            sel = (r * n_tiles).long()
            pts = self.map_spawn[m.unsqueeze(1), team, sel]
            self.pos[:, lo:hi] = torch.where(sub.unsqueeze(2), pts, self.pos[:, lo:hi])
        self.hp = torch.where(ready, torch.full_like(self.hp, float(PLAYER_HP)), self.hp)
        self.shield = torch.where(ready, torch.full_like(self.shield, float(SHIELD_MAX)), self.shield)
        self.since_dmg = torch.where(ready, torch.zeros_like(self.since_dmg), self.since_dmg)
        self.alive = self.alive | ready
        self.waiting = self.waiting & ~ready
        self.cooldown = torch.where(ready, torch.zeros_like(self.cooldown), self.cooldown)
        self.dash_cd = torch.where(ready, torch.zeros_like(self.dash_cd), self.dash_cd)
        if self.record:
            for w, ag in torch.nonzero(ready).tolist():
                self._events[w].append({"type": "respawn", "frame": int(self.frame[w]), "agent": ag})
        return ready
    # endregion FUNC__respawn

    # region FUNC__update_control_points
    ## @purpose Capture state machine for all points of all worlds at once (unchanged from experiment 10).
    def _update_control_points(self, cp_gain, cap_part) -> None:
        cp_xy = self.map_cp_xy[self.map_idx]
        radius = self.map_cp_r[self.map_idx].unsqueeze(1)
        d = (cp_xy.unsqueeze(2) - self.pos.unsqueeze(1)).pow(2).sum(-1).sqrt()
        inside = (d <= radius.unsqueeze(2)) & self.alive.unsqueeze(1)
        blue_near = inside[:, :, :TEAM_SIZE].sum(2)
        red_near = inside[:, :, TEAM_SIZE:].sum(2)

        contested = (blue_near > 0) & (red_near > 0)
        active = torch.where(blue_near > 0, torch.ones_like(self.cp_owner), torch.zeros_like(self.cp_owner))
        active = torch.where(red_near > 0, torch.full_like(active, 2), active)
        active = torch.where(contested, torch.zeros_like(active), active)

        empty = active == 0
        decay = (self.cp_progress - 0.5 / CP_CAPTURE_FRAMES).clamp(min=0.0)
        owned_by_active = active == self.cp_owner

        switch = (~empty) & (~owned_by_active) & (self.cp_cap_team != active)
        prog = torch.where(switch, torch.zeros_like(self.cp_progress), self.cp_progress)
        cap_team = torch.where(switch, active, self.cp_cap_team)

        advancing = (~empty) & (~owned_by_active) & (~contested)
        prog = torch.where(advancing, prog + 1.0 / CP_CAPTURE_FRAMES, prog)

        captured = advancing & (prog >= 1.0)
        new_owner = torch.where(captured, active, self.cp_owner)
        prog = torch.where(captured, torch.zeros_like(prog), prog)
        cap_team = torch.where(captured, torch.zeros_like(cap_team), cap_team)

        prog = torch.where(empty, decay, prog)
        cap_team = torch.where(empty & (decay <= 0), torch.zeros_like(cap_team), cap_team)
        prog = torch.where(owned_by_active & (~empty), torch.zeros_like(prog), prog)
        cap_team = torch.where(owned_by_active & (~empty), torch.zeros_like(cap_team), cap_team)

        self.cp_progress, self.cp_cap_team, self.cp_owner = prog, cap_team, new_owner
        for team in (0, 1):
            cp_gain[:, team] += (captured & (active == team + 1)).sum(1).float()
        cap_part += (inside & captured.unsqueeze(2)).sum(1).float()
        if self.record and captured.any():
            for w, k in torch.nonzero(captured).tolist():
                self._events[w].append({"type": "capture", "frame": int(self.frame[w]), "cp": k, "team": int(active[w, k]) - 1})
    # endregion FUNC__update_control_points

    def _score_income(self) -> None:
        for team in (0, 1):
            self.score[:, team] += (self.cp_owner == team + 1).sum(1).float() * CP_SCORE_PER_FRAME

    # region FUNC__reward
    ## @purpose Individual reward per agent and team reward per team; the trainer mixes them.
    ## @rationale Same components and weights as experiment 10, split by channel instead of
    ## pre-mixed for one learner slot. The timeout payout is paid only when the clock is a
    ## termination (see module contract): a truncation must not carry an outcome.
    def _reward(self, dmg_dealt, dmg_taken, kills, deaths, cp_gain, cap_part, score_before, winner, terminated, clock):
        delta = self.score - score_before                                        # (N, 2)
        by_timeout = clock & terminated
        r_team = torch.zeros(self.n, 2, device=self.device)
        for team in (0, 1):
            other = 1 - team
            tr = cp_gain[:, team] * R_CP_CAPTURED + cp_gain[:, other] * R_CP_LOST
            tr = tr + (delta[:, team] - delta[:, other]) * R_SCORE_DELTA
            my_win = (winner == team + 1) & terminated
            my_loss = (winner == other + 1) & terminated
            draw = (winner == 0) & terminated
            win_pay = torch.where(by_timeout, torch.full_like(tr, R_WIN_TIMEOUT), torch.full_like(tr, R_WIN))
            loss_pay = torch.where(by_timeout, torch.full_like(tr, R_LOSS_TIMEOUT), torch.full_like(tr, R_LOSS))
            r_team[:, team] = tr + my_win.float() * win_pay + my_loss.float() * loss_pay + draw.float() * R_TIMEOUT_DRAW
        rc = self.reward_cfg
        combat = (dmg_dealt * rc.damage_dealt + dmg_taken * rc.damage_taken
                  + kills * rc.kill + deaths * rc.death) * self.combat_boost        # (N, 10)
        if rc.zero_sum:
            # OpenAI Five: subtract the enemy team's mean, so the sum over all ten agents is 0 for any boost.
            blue, red = combat[:, :TEAM_SIZE], combat[:, TEAM_SIZE:]
            combat = torch.cat([blue - red.mean(1, keepdim=True), red - blue.mean(1, keepdim=True)], dim=1)
        r_indiv = combat + cap_part * R_CP_CAPTURE_INDIV
        return r_indiv, r_team
    # endregion FUNC__reward

    def set_combat_boost(self, boost: float) -> None:
        """Multiplier on the combat channel (damage + kills/deaths) for future steps; >= 0."""
        self.combat_boost = max(0.0, float(boost))

    # region FUNC__aim_label
    ## @purpose Auxiliary-task label per agent: would a bullet fired NOW along the current barrel hit a
    ## visible enemy? Geometry only (enemy velocity ignored): the enemy centre lies ahead of the barrel,
    ## within HIT_RADIUS of its line, within bullet range, and in line of sight (LOS to the centre stands
    ## in for "before a wall" — the centre is at most HIT_RADIUS off the bullet path).
    ## @io vis (N, A, A) bool -> (N, A) float {0, 1}
    ## @rationale Lample & Chaplot (Doom, AAAI 2017): agents without an "enemy in view" signal fired at
    ## will; co-training that game feature fixed it. Computing it here costs O(N*A*A) on tensors the
    ## observation already built — the visibility matrix is reused, not recomputed.
    def _aim_label(self, vis: torch.Tensor) -> torch.Tensor:
        d = torch.stack([self.angle.cos(), self.angle.sin()], dim=-1)            # (N, A, 2)
        rel = self.pos.unsqueeze(1) - self.pos.unsqueeze(2)                      # [w, i, j] = pos_j - pos_i
        along = (rel * d.unsqueeze(2)).sum(-1)
        perp = (rel[..., 0] * d.unsqueeze(2)[..., 1] - rel[..., 1] * d.unsqueeze(2)[..., 0]).abs()
        team = self._agent_team
        enemy = (team.view(1, -1, 1) != team.view(1, 1, -1))
        both_alive = self.alive.unsqueeze(2) & self.alive.unsqueeze(1)
        in_range = (along > 0) & (along <= float(BULLET_SPEED * BULLET_LIFETIME_FRAMES))
        hit = enemy & both_alive & vis & in_range & (perp < HIT_RADIUS)
        return hit.any(2).float()
    # endregion FUNC__aim_label

    # region FUNC__select
    ## @purpose State tensors for a subset of worlds (None = all, as views).
    def _sel(self, t: torch.Tensor, idx: torch.Tensor | None) -> torch.Tensor:
        return t if idx is None else t[idx]
    # endregion FUNC__select

    # region FUNC__visibility
    ## @purpose Line of sight within vision range between every pair of agents.
    ## @io idx -> (n, A, A) bool; [w, i, j] = i can see j (both alive not required here)
    def _visibility(self, idx: torch.Tensor | None) -> torch.Tensor:
        pos = self._sel(self.pos, idx)
        maps = self._sel(self.map_idx, idx)
        n = pos.shape[0]
        a = pos.unsqueeze(2)
        b = pos.unsqueeze(1)
        t = torch.linspace(0.0, 1.0, LOS_SAMPLES, device=self.device).view(1, 1, 1, LOS_SAMPLES, 1)
        pts = a.unsqueeze(3) + (b - a).unsqueeze(3) * t
        m = maps.view(n, 1, 1, 1).expand(-1, N_AGENTS, N_AGENTS, LOS_SAMPLES)
        blocked = self._wall_at(m, pts[..., 0], pts[..., 1]).any(-1)
        dist = (b - a).pow(2).sum(-1).sqrt()
        return ~blocked & (dist <= VISION_RANGE)
    # endregion FUNC__visibility

    # region FUNC__update_memory
    ## @purpose Team-shared last-seen enemy memory: refresh what any alive teammate sees now,
    ## age the rest, forget past the horizon and on the enemy's death.
    def _update_memory(self, vis: torch.Tensor, idx: torch.Tensor | None = None) -> None:
        alive = self._sel(self.alive, idx)
        pos, vel, hp = self._sel(self.pos, idx), self._sel(self.vel, idx), self._sel(self.hp, idx)
        mp, mv, mh = self._sel(self.mem_pos, idx), self._sel(self.mem_vel, idx), self._sel(self.mem_hp, idx)
        ma, mval = self._sel(self.mem_age, idx), self._sel(self.mem_valid, idx)
        if idx is not None:
            mp, mv, mh, ma, mval = mp.clone(), mv.clone(), mh.clone(), ma.clone(), mval.clone()
        for t in (0, 1):
            me = slice(t * TEAM_SIZE, (t + 1) * TEAM_SIZE)
            en = slice((1 - t) * TEAM_SIZE, (2 - t) * TEAM_SIZE)
            seen = (vis[:, me, en] & alive[:, me].unsqueeze(2)).any(1) & alive[:, en]   # (n, 5)
            s3 = seen.unsqueeze(-1)
            mp[:, t] = torch.where(s3, pos[:, en], mp[:, t])
            mv[:, t] = torch.where(s3, vel[:, en], mv[:, t])
            mh[:, t] = torch.where(seen, hp[:, en], mh[:, t])
            ma[:, t] = torch.where(seen, torch.zeros_like(ma[:, t]), ma[:, t] + ACTION_REPEAT)
            mval[:, t] = (mval[:, t] | seen) & (ma[:, t] <= LASTSEEN_HORIZON_FRAMES) & alive[:, en]
        if idx is None:
            self.mem_pos, self.mem_vel, self.mem_hp, self.mem_age, self.mem_valid = mp, mv, mh, ma, mval
        else:
            self.mem_pos[idx], self.mem_vel[idx], self.mem_hp[idx], self.mem_age[idx], self.mem_valid[idx] = mp, mv, mh, ma, mval
    # endregion FUNC__update_memory

    # region FUNC__cast_rays
    def _cast_rays(self, pos: torch.Tensor, maps: torch.Tensor) -> torch.Tensor:
        n = pos.shape[0]
        steps = torch.arange(1, RAY_SAMPLES + 1, device=self.device, dtype=torch.float32) * RAY_STEP
        offs = self._ray_dir.view(1, 1, N_WALL_RAYS, 1, 2) * steps.view(1, 1, 1, RAY_SAMPLES, 1)
        pts = pos.view(n, N_AGENTS, 1, 1, 2) + offs
        m = maps.view(n, 1, 1, 1).expand(-1, N_AGENTS, N_WALL_RAYS, RAY_SAMPLES)
        arena = self.map_arena[maps].view(n, 1, 1, 1, 2)
        outside = (pts < 0).any(-1) | (pts >= arena).any(-1)
        blocked = self._wall_at(m, pts[..., 0], pts[..., 1]) | outside
        any_hit = blocked.any(-1)
        first = blocked.float().argmax(-1)
        dist = (first + 1).float() * RAY_STEP
        dist = torch.where(any_hit, dist, torch.full_like(dist, RAY_MAX_DIST + RAY_STEP))
        return (dist / RAY_MAX_DIST).clamp(max=1.0)
    # endregion FUNC__cast_rays

    # region FUNC_observe
    ## @purpose Observation of every agent of every world (no state change).
    def observe(self) -> torch.Tensor:
        vis = self._visibility(None)
        self._vis_cache = vis
        return self._build_obs(None, vis)
    # endregion FUNC_observe

    # region FUNC__build_obs
    ## @purpose Build the experiment-11 observation for a subset of worlds.
    ## @io (idx or None, vis (n, A, A)) -> (n, A, OBS_SIZE)
    ## @complexity 10
    def _build_obs(self, idx: torch.Tensor | None, vis: torch.Tensor) -> torch.Tensor:
        S = lambda t: self._sel(t, idx)                                          # noqa: E731
        pos, vel, angle = S(self.pos), S(self.vel), S(self.angle)
        hp, shield, alive = S(self.hp), S(self.shield), S(self.alive)
        maps = S(self.map_idx)
        n, a, dev = pos.shape[0], N_AGENTS, self.device
        obs = torch.zeros(n, a, OBS_SIZE, device=dev)
        arena = self.map_arena[maps].unsqueeze(1)                                # (n, 1, 2)
        diag = self.map_diag[maps].view(n, 1, 1)
        team = self._agent_team.view(1, a).expand(n, a)
        is_blue = (team == 0)
        facing = torch.stack([angle.cos(), angle.sin()], dim=-1)                  # (n, A, 2)
        pm1 = lambda cond: torch.where(cond, 1.0, -1.0)                          # noqa: E731

        # ---- core ----
        core = torch.zeros(n, a, CORE_SIZE, device=dev)
        core[..., 0] = pos[..., 0] / arena[..., 0] * 2 - 1
        core[..., 1] = pos[..., 1] / arena[..., 1] * 2 - 1
        core[..., 2] = torch.where(alive, hp / PLAYER_HP * 2 - 1, torch.full_like(hp, -1.0))
        core[..., 3] = torch.where(alive, shield / SHIELD_MAX * 2 - 1, torch.full_like(shield, -1.0))
        core[..., 4] = pm1(alive)
        core[..., 5] = S(self.respawn_timer).float() / RESPAWN_FRAMES * 2 - 1
        core[..., 6] = S(self.cooldown).float() / SHOOT_COOLDOWN_FRAMES * 2 - 1
        core[..., 7:9] = facing
        core[..., 9:11] = (vel / VEL_NORM).clamp(-1.0, 1.0)
        core[..., 11] = S(self.dash_cd).float() / DASH_COOLDOWN_FRAMES * 2 - 1
        core[..., 12] = S(self.dash_left).float() / DASH_FRAMES * 2 - 1
        core[..., 13:21] = self._cast_rays(pos, maps) * 2 - 1
        score = S(self.score)
        my_score = torch.where(is_blue, score[:, :1], score[:, 1:])
        en_score = torch.where(is_blue, score[:, 1:], score[:, :1])
        core[..., 21] = (my_score / SCORE_TO_WIN).clamp(max=1.0) * 2 - 1
        core[..., 22] = (en_score / SCORE_TO_WIN).clamp(max=1.0) * 2 - 1
        alive_b = alive[:, :TEAM_SIZE].sum(1, keepdim=True).float()
        alive_r = alive[:, TEAM_SIZE:].sum(1, keepdim=True).float()
        core[..., 23] = torch.where(is_blue, alive_b, alive_r) / TEAM_SIZE * 2 - 1
        core[..., 24] = torch.where(is_blue, alive_r, alive_b) / TEAM_SIZE * 2 - 1
        lives = S(self.lives).float()
        core[..., 25] = torch.where(is_blue, lives[:, :1], lives[:, 1:]) / TEAM_LIVES * 2 - 1
        core[..., 26] = torch.where(is_blue, lives[:, 1:], lives[:, :1]) / TEAM_LIVES * 2 - 1
        core[..., 27] = (S(self.frame).float() / MAX_FRAMES).clamp(max=1.0).view(n, 1) * 2 - 1
        core[..., 28] = pm1(self.map_compact[maps]).view(n, 1).expand(n, a)
        obs[..., :CORE_SIZE] = core

        # ---- entity slots: allies always, enemies when visible; nearest first ----
        rel = pos.unsqueeze(1) - pos.unsqueeze(2)                                 # (n, me, other, 2)
        dist = rel.pow(2).sum(-1).sqrt()
        same_team = team.unsqueeze(2) == team.unsqueeze(1)
        is_self = torch.eye(a, dtype=torch.bool, device=dev).view(1, a, a)
        other_alive = alive.unsqueeze(1).expand(n, a, a)
        present = other_alive & ~is_self & (same_team | vis)
        # Nine slots for nine others; absent ones sort last and are zeroed by `keep`.
        order = torch.where(present, dist, torch.full_like(dist, float("inf"))).argsort(dim=2)[:, :, :N_ENTITY_SLOTS]
        keep = present.gather(2, order).float().unsqueeze(-1)

        def g(t: torch.Tensor) -> torch.Tensor:
            # per-agent tensor (n, A[, k]) -> the ordered others as seen by each agent (n, A, 9[, k])
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
        ent = torch.stack(
            [
                ones,                                                            # 0 present
                torch.where(g_same, torch.full_like(g_dist, ETYPE_ALLY), torch.full_like(g_dist, ETYPE_ENEMY)),
                (g_rel[..., 0] / ENTITY_RANGE).clamp(-1, 1),                     # 2
                (g_rel[..., 1] / ENTITY_RANGE).clamp(-1, 1),                     # 3
                (g_dist / ENTITY_RANGE).clamp(max=1.0) * 2 - 1,                  # 4
                (g_vel[..., 0] / VEL_NORM).clamp(-1, 1),                         # 5
                (g_vel[..., 1] / VEL_NORM).clamp(-1, 1),                         # 6
                g_face[..., 0], g_face[..., 1],                                  # 7, 8
                g_hp / PLAYER_HP * 2 - 1,                                        # 9
                g_shield / SHIELD_MAX * 2 - 1,                                   # 10
                aim_err / math.pi,                                               # 11 my barrel vs them
                their_err / math.pi,                                             # 12 their barrel vs me
                torch.where(g_vis, ones, -ones),                                 # 13 in my line of sight
                torch.where(g_dash, ones, -ones),                                # 14 dashing
            ],
            dim=-1,
        ) * keep
        s = _SEG["entities"]
        obs[..., s["start"]: s["start"] + N_ENTITY_SLOTS * ENTITY_SLOT_SIZE] = ent.reshape(n, a, -1)

        # ---- bullet slots: nearest hostile bullets ----
        b_pos, b_vel, b_alive = S(self.b_pos), S(self.b_vel), S(self.b_alive)
        b_rel = b_pos.unsqueeze(1) - pos.unsqueeze(2)                             # (n, A, B, 2)
        b_dist = b_rel.pow(2).sum(-1).sqrt()
        b_team = self._agent_team[S(self.b_owner).long()].unsqueeze(1)
        hostile = (b_team != team.unsqueeze(2)) & b_alive.unsqueeze(1)
        b_order = torch.where(hostile, b_dist, torch.full_like(b_dist, float("inf"))).argsort(dim=2)[:, :, :N_BULLET_SLOTS]
        b_keep = hostile.gather(2, b_order).float().unsqueeze(-1)
        idx2 = b_order.unsqueeze(-1).expand(-1, -1, -1, 2)
        g_brel = b_rel.gather(2, idx2)
        g_bvel = b_vel.unsqueeze(1).expand(n, a, BULLET_SLOTS, 2).gather(2, idx2)
        vv = g_bvel.pow(2).sum(-1).clamp(min=1e-6)
        t_star = (-(g_brel * g_bvel).sum(-1) / vv).clamp(min=0.0)                 # frames to closest approach
        closest = (g_brel + g_bvel * t_star.unsqueeze(-1)).pow(2).sum(-1).sqrt()
        bones = torch.ones_like(t_star)
        bul = torch.stack(
            [
                bones,
                (g_brel[..., 0] / BULLET_OBS_RANGE).clamp(-1, 1),
                (g_brel[..., 1] / BULLET_OBS_RANGE).clamp(-1, 1),
                g_bvel[..., 0] / BULLET_SPEED,
                g_bvel[..., 1] / BULLET_SPEED,
                (closest / BULLET_MISS_NORM).clamp(max=1.0) * 2 - 1,
                (t_star / BULLET_TTC_NORM).clamp(max=1.0) * 2 - 1,
            ],
            dim=-1,
        ) * b_keep
        s = _SEG["bullets"]
        obs[..., s["start"]: s["start"] + N_BULLET_SLOTS * BULLET_SLOT_SIZE] = bul.reshape(n, a, -1)

        # ---- control point slots ----
        cp_xy = self.map_cp_xy[maps]                                              # (n, C, 2)
        radius = self.map_cp_r[maps].view(n, 1, 1)
        c_rel = cp_xy.unsqueeze(1) - pos.unsqueeze(2)                             # (n, A, C, 2)
        c_dist = c_rel.pow(2).sum(-1).sqrt()
        my_tid = (team + 1).unsqueeze(2)
        owner = S(self.cp_owner).unsqueeze(1)
        cap = S(self.cp_cap_team).unsqueeze(1)
        owner_val = torch.where(owner == my_tid, 1.0, torch.where(owner == 0, 0.0, -1.0))
        cap_val = torch.where(cap == my_tid, 1.0, torch.where(cap == 0, 0.0, -1.0))
        d_cp = c_dist.transpose(1, 2)                                             # (n, C, A)
        near = (d_cp <= radius) & alive.unsqueeze(1)
        blue_near = near[:, :, :TEAM_SIZE].any(2).unsqueeze(1)
        red_near = near[:, :, TEAM_SIZE:].any(2).unsqueeze(1)
        mine_near = torch.where(is_blue.unsqueeze(2), blue_near, red_near)
        foe_near = torch.where(is_blue.unsqueeze(2), red_near, blue_near)

        ts = self.tile_size
        row = (pos[..., 1] / ts).long().clamp(0, self.map_grid.shape[1] - 1)     # (n, A)
        col = (pos[..., 0] / ts).long().clamp(0, self.map_grid.shape[2] - 1)
        mm = maps.view(n, 1, 1).expand(n, a, N_CP_SLOTS)
        kk = torch.arange(N_CP_SLOTS, device=dev).view(1, 1, -1).expand(n, a, -1)
        rr, cc = row.unsqueeze(2).expand(-1, -1, N_CP_SLOTS), col.unsqueeze(2).expand(-1, -1, N_CP_SLOTS)
        p_dist = self.flow_dist[mm, kk, rr, cc]                                   # (n, A, C)
        p_dir = self.flow_dir[mm, kk, rr, cc]                                     # (n, A, C, 2)
        path_norm = 1.5 * diag
        p_obs = torch.where(torch.isfinite(p_dist), (p_dist / path_norm).clamp(max=1.0) * 2 - 1, torch.ones_like(p_dist))
        cones = torch.ones_like(c_dist)
        cps = torch.stack(
            [
                cones,                                                           # 0 present
                owner_val.expand(n, a, N_CP_SLOTS),                              # 1
                c_rel[..., 0] / diag,                                            # 2
                c_rel[..., 1] / diag,                                            # 3
                (c_dist / diag).clamp(max=1.0) * 2 - 1,                          # 4 straight-line
                p_obs,                                                           # 5 walkable path
                p_dir[..., 0], p_dir[..., 1],                                    # 6, 7 first step of the path
                S(self.cp_progress).unsqueeze(1).expand(n, a, N_CP_SLOTS) * 2 - 1,
                cap_val.expand(n, a, N_CP_SLOTS),                                # 9
                torch.where(foe_near, 1.0, -1.0).expand(n, a, N_CP_SLOTS),       # 10
                torch.where(mine_near, 1.0, -1.0).expand(n, a, N_CP_SLOTS),      # 11
                torch.where(c_dist <= radius, 1.0, -1.0),                        # 12 I am inside
                (c_dist / radius).clamp(max=2.0) - 1.0,                          # 13
            ],
            dim=-1,
        )
        s = _SEG["cps"]
        obs[..., s["start"]: s["start"] + N_CP_SLOTS * CP_SLOT_SIZE] = cps.reshape(n, a, -1)

        # ---- last-seen enemy memory (team-shared), only for enemies I do not see right now ----
        mem_valid = S(self.mem_valid)                                             # (n, 2, 5)
        my_team_mem = lambda t: torch.cat([t[:, 0:1].expand(n, TEAM_SIZE, *t.shape[2:]), t[:, 1:2].expand(n, TEAM_SIZE, *t.shape[2:])], dim=1)  # noqa: E731
        m_pos, m_vel = my_team_mem(S(self.mem_pos)), my_team_mem(S(self.mem_vel))  # (n, A, 5, 2)
        m_hp, m_age, m_val = my_team_mem(S(self.mem_hp)), my_team_mem(S(self.mem_age)), my_team_mem(mem_valid)
        enemy_idx = torch.cat([torch.arange(TEAM_SIZE, 2 * TEAM_SIZE), torch.arange(0, TEAM_SIZE)]).to(dev)
        enemy_idx = enemy_idx.view(2, TEAM_SIZE)[self._agent_team]                # (A, 5): enemy slot ids per agent
        i_see = vis.gather(2, enemy_idx.view(1, a, TEAM_SIZE).expand(n, -1, -1))   # (n, A, 5)
        show = (m_val & ~i_see).float().unsqueeze(-1)
        l_rel = m_pos - pos.unsqueeze(2)
        l_dist = l_rel.pow(2).sum(-1).sqrt()
        lones = torch.ones_like(l_dist)
        ls = torch.stack(
            [
                lones,
                (l_rel[..., 0] / diag.view(n, 1, 1)),
                (l_rel[..., 1] / diag.view(n, 1, 1)),
                (l_dist / diag.view(n, 1, 1)).clamp(max=1.0) * 2 - 1,
                m_age.float() / LASTSEEN_HORIZON_FRAMES * 2 - 1,
                torch.where(m_age == 0, lones, -lones),                          # a teammate sees it now
                (m_vel[..., 0] / VEL_NORM).clamp(-1, 1),
                (m_vel[..., 1] / VEL_NORM).clamp(-1, 1),
                m_hp / PLAYER_HP * 2 - 1,
            ],
            dim=-1,
        ) * show
        s = _SEG["lastseen"]
        obs[..., s["start"]: s["start"] + N_LASTSEEN_SLOTS * LASTSEEN_SLOT_SIZE] = ls.reshape(n, a, -1)

        # ---- egocentric grid: walls, CP zones signed by owner ----
        pr = row + GRID_RADIUS
        pc = col + GRID_RADIUS
        gr = pr.view(n, a, 1, 1) + self._grid_dr.view(1, 1, GRID_SIDE, GRID_SIDE)
        gc = pc.view(n, a, 1, 1) + self._grid_dc.view(1, 1, GRID_SIDE, GRID_SIDE)
        gm = maps.view(n, 1, 1, 1).expand(n, a, GRID_SIDE, GRID_SIDE)
        walls = self.map_grid_pad[gm, gr, gc].float()
        zone = self.map_zone_pad[gm, gr, gc]                                      # (n, A, S, S) cp idx or -1
        z_owner = S(self.cp_owner).gather(1, zone.clamp(min=0).view(n, -1)).view(n, a, GRID_SIDE, GRID_SIDE)
        my_t = (team + 1).view(n, a, 1, 1)
        z_val = torch.where(z_owner == my_t, 1.0, torch.where(z_owner == 0, 0.5, -1.0))
        z_val = torch.where(zone >= 0, z_val, torch.zeros_like(z_val))
        s = _SEG["grid"]
        obs[..., s["start"]:] = torch.stack([walls, z_val], dim=2).reshape(n, a, -1)

        return obs.clamp(-1.0, 1.0)
    # endregion FUNC__build_obs

    # region FUNC_global_state
    ## @purpose Full-information state per team perspective for a centralized critic.
    ## @io None -> (N, 2, STATE_SIZE); [:, t] lists team t's agents first, CP ownership signed for t
    def global_state(self) -> torch.Tensor:
        n, dev = self.n, self.device
        arena = self.map_arena[self.map_idx].unsqueeze(1)
        per = torch.stack(
            [
                self.pos[..., 0] / arena[..., 0] * 2 - 1,
                self.pos[..., 1] / arena[..., 1] * 2 - 1,
                (self.vel[..., 0] / VEL_NORM).clamp(-1, 1),
                (self.vel[..., 1] / VEL_NORM).clamp(-1, 1),
                self.angle.cos(), self.angle.sin(),
                torch.where(self.alive, self.hp / PLAYER_HP * 2 - 1, torch.full_like(self.hp, -1.0)),
                torch.where(self.alive, self.shield / SHIELD_MAX * 2 - 1, torch.full_like(self.hp, -1.0)),
                torch.where(self.alive, 1.0, -1.0),
                self.respawn_timer.float() / RESPAWN_FRAMES * 2 - 1,
                self.dash_cd.float() / DASH_COOLDOWN_FRAMES * 2 - 1,
                self.cooldown.float() / SHOOT_COOLDOWN_FRAMES * 2 - 1,
            ],
            dim=-1,
        )                                                                         # (N, A, 12)
        cp_xy = self.map_cp_xy[self.map_idx]
        out = torch.zeros(n, 2, STATE_SIZE, device=dev)
        onehot = self.map_desc[self.map_idx]                               # (N, N_MAPS_STATE) map descriptor
        for t in (0, 1):
            me = slice(t * TEAM_SIZE, (t + 1) * TEAM_SIZE)
            en = slice((1 - t) * TEAM_SIZE, (2 - t) * TEAM_SIZE)
            agents = torch.cat([per[:, me], per[:, en]], dim=1).reshape(n, -1)
            tid = t + 1
            own = torch.where(self.cp_owner == tid, 1.0, torch.where(self.cp_owner == 0, 0.0, -1.0))
            capv = torch.where(self.cp_cap_team == tid, 1.0, torch.where(self.cp_cap_team == 0, 0.0, -1.0))
            cps = torch.stack(
                [cp_xy[..., 0] / arena[..., 0] * 2 - 1, cp_xy[..., 1] / arena[..., 1] * 2 - 1, own, self.cp_progress * 2 - 1, capv],
                dim=-1,
            ).reshape(n, -1)
            glob = torch.stack(
                [
                    (self.score[:, t] / SCORE_TO_WIN).clamp(max=1.0) * 2 - 1,
                    (self.score[:, 1 - t] / SCORE_TO_WIN).clamp(max=1.0) * 2 - 1,
                    self.lives[:, t].float() / TEAM_LIVES * 2 - 1,
                    self.lives[:, 1 - t].float() / TEAM_LIVES * 2 - 1,
                    (self.frame.float() / MAX_FRAMES).clamp(max=1.0) * 2 - 1,
                    torch.where(self.map_compact[self.map_idx], 1.0, -1.0),
                ],
                dim=-1,
            )
            out[:, t] = torch.cat([agents, cps, glob, onehot], dim=1)
        return out
    # endregion FUNC_global_state

    # region FUNC_snapshot
    ## @purpose Render state of one world for the replay recorder (plain Python types).
    def snapshot(self, world_idx: int) -> dict:
        w = int(world_idx)
        agents = []
        for i in range(N_AGENTS):
            agents.append({
                "id": i, "team": int(self._agent_team[i]),
                "x": float(self.pos[w, i, 0]), "y": float(self.pos[w, i, 1]),
                "angle": float(self.angle[w, i]),
                "vx": float(self.vel[w, i, 0]), "vy": float(self.vel[w, i, 1]),
                "hp": float(self.hp[w, i]), "shield": float(self.shield[w, i]),
                "alive": bool(self.alive[w, i]), "respawn_in": int(self.respawn_timer[w, i]) if bool(self.waiting[w, i]) else 0,
                "dashing": bool(self.dash_left[w, i] > 0), "dash_cd": int(self.dash_cd[w, i]),
            })
        live = torch.nonzero(self.b_alive[w]).flatten().tolist()
        bullets = [{
            "x": float(self.b_pos[w, k, 0]), "y": float(self.b_pos[w, k, 1]),
            "vx": float(self.b_vel[w, k, 0]), "vy": float(self.b_vel[w, k, 1]),
            "team": int(self._agent_team[int(self.b_owner[w, k])]), "owner": int(self.b_owner[w, k]),
        } for k in live]
        m = int(self.map_idx[w])
        r = float(self.map_cp_r[m])
        cps = [{
            "x": float(self.map_cp_xy[m, k, 0]), "y": float(self.map_cp_xy[m, k, 1]), "r": r,
            "owner": int(self.cp_owner[w, k]), "progress": float(self.cp_progress[w, k]), "cap_team": int(self.cp_cap_team[w, k]),
        } for k in range(N_CP_SLOTS)]
        return {
            "frame": int(self.frame[w]), "map_idx": m,
            "score": [float(self.score[w, 0]), float(self.score[w, 1])],
            "lives": [int(self.lives[w, 0]), int(self.lives[w, 1])],
            "agents": agents, "bullets": bullets, "cps": cps,
        }
    # endregion FUNC_snapshot

    # region FUNC_map_geometry
    ## @purpose Static geometry of one map for the replay header.
    def map_geometry(self, map_idx: int) -> dict:
        m = self._pool[int(map_idx)]
        grid = self.map_grid[int(map_idx), : m.rows, : m.cols].cpu()
        walls = [[int(c), int(r)] for r, c in torch.nonzero(grid).tolist()]
        return {
            "name": m.name, "arena_w": int(m.arena_w), "arena_h": int(m.arena_h),
            "tile_size": int(m.tile_size), "cols": int(m.cols), "rows": int(m.rows),
            "walls": walls, "compact": bool(m.compact), "cp_radius": float(m.cp_radius),
        }
    # endregion FUNC_map_geometry
# endregion CLASS_World11
