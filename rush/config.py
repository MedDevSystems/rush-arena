# region MODULE_CONTRACT [DOMAIN(7): Configuration; CONCEPT(8): GameConstants; TECH(6): Python]
## @modulecontract
## @purpose Centralized game constants for the 2D top-down arena shooter: screen dimensions, physics, colors, entity parameters.
## @scope All numeric/visual constants consumed by other modules
## @input None (pure constants)
## @output Importable constant values
## @links LINKS_TO: entities, arena, renderer, game_loop
## @invariants
## - TILE_SIZE evenly divides ARENA_W and ARENA_H
## - All speeds are in pixels per frame at 60 FPS
## @rationale
## Q: Why a flat module instead of JSON config?
## A: Constants are tightly coupled with game logic (tile math, physics). A flat Python file gives type safety and zero parsing overhead.
## @changes
## LAST_CHANGE: [v0.6.0] Experiment 10: two-speed turn (the 3-step turn could only reach 6 barrel
##   directions), shield with out-of-combat regen, timeout wins at half price, individual capture
##   credit. Legacy TURN_STEP_DEG stays for env_tdm (experiment 8b).
## PREV: [v0.5.0] Fog of war, 15 Hz decisions, self-play league, attention+LSTM sizes, outcome-only reward weights.
## @modulemap
## BLOCK 9[Screen and arena geometry] => SCREEN / ARENA / TILE
## BLOCK 8[Entity parameters] => PLAYER / BULLET
## BLOCK 8[Shield over HP] => SHIELD_*
## BLOCK 8[Fog of war limits] => VISION_*
## BLOCK 9[Decision rate] => ACTION_REPEAT / DECISION_HZ
## BLOCK 9[Aiming steps] => TURN_*
## BLOCK 9[Self-play league policy] => LEAGUE_*
## BLOCK 8[Network sizing] => ATTN_* / FEATURES_DIM / LSTM_HIDDEN
## BLOCK 9[Outcome reward weights] => R_* / TAU_TEAM
## BLOCK 7[Colors palette] => COLOR_*
## @usecases
## - [any module]: import config -> use constants
def _module_contract():
    pass
# endregion MODULE_CONTRACT
# GREP_SUMMARY: config, constants, screen, arena, tile, player, bullet, colors, FPS
# STRUCTURE: ▶ constants only → ⎋ importable values

# region BLOCK_CONSTANTS_SCREEN
FPS: int = 60
SCREEN_W: int = 900
SCREEN_H: int = 900
# endregion BLOCK_CONSTANTS_SCREEN

# region BLOCK_CONSTANTS_ARENA
TILE_SIZE: int = 30
ARENA_COLS: int = 120
ARENA_ROWS: int = 120
ARENA_W: int = ARENA_COLS * TILE_SIZE   # 3600 logical px
ARENA_H: int = ARENA_ROWS * TILE_SIZE   # 3600 logical px
RENDER_SCALE: float = SCREEN_W / ARENA_W  # 0.25 — fit 3600 into 900
# endregion BLOCK_CONSTANTS_ARENA

# region BLOCK_CONSTANTS_CONTROL_POINTS
CP_CAPTURE_RADIUS: float = 150.0
CP_CAPTURE_FRAMES: int = 120            # 2 seconds to capture
CP_POSITIONS: list[tuple[int, int]] = [  # (row, col) on tile grid
    (20, 60),   # CP-A: near red spawn
    (40, 25),   # CP-B: left mid
    (40, 95),   # CP-C: right mid
    (60, 60),   # CP-D: center
    (80, 25),   # CP-E: left mid
    (80, 95),   # CP-F: right mid
    (100, 60),  # CP-G: near blue spawn
]
CP_COUNT: int = len(CP_POSITIONS)
SCORE_TO_WIN: int = 200
# endregion BLOCK_CONSTANTS_CONTROL_POINTS

# region BLOCK_CONSTANTS_VISION
# Fog of war: an enemy enters the observation only when it is both in line of
# sight and within visual range. Allies stay visible always (team comms).
VISION_RANGE: float = 1400.0
ALLY_COMMS: bool = True
# endregion BLOCK_CONSTANTS_VISION

# region BLOCK_CONSTANTS_DECISION
# Agents decide at 60/ACTION_REPEAT Hz; physics still runs at 60 FPS.
# Longer holds make behaviour readable to a spectator and stretch the discount
# horizon over the 120-frame capture without touching gamma.
ACTION_REPEAT: int = 4
DECISION_HZ: float = FPS / ACTION_REPEAT
# endregion BLOCK_CONSTANTS_DECISION

# region BLOCK_CONSTANTS_AIM
# Aiming is the policy's job. AIM_ASSIST is the share of the way the engine turns
# the barrel toward the nearest visible enemy each frame: 1.0 is the old auto-aim,
# 0.0 means nobody helps. It stays here as an emergency lever, not as a default —
# the compact early maps are what makes hitting learnable from random weights.
AIM_ASSIST: float = 0.0
TURN_STEP_DEG: float = 15.0                 # LEGACY (env_tdm / experiment 8b): per frame, x4 per decision
# Experiment 10 turn: five options [-coarse, -fine, 0, +fine, +coarse], applied ONCE
# per decision in batch_world. The old 15 deg/frame x ACTION_REPEAT=4 quantised the
# barrel to 60 deg steps — measured: exactly 6 reachable directions after 120 decisions.
# A 24 px target subtends 4.6 deg at 300 px, 2.3 at 600, 1.4 at 1000. Fine 2.5 deg
# leaves a residual error <= 1.25 deg after settling — inside the target out to ~550 px —
# and matches the 2.6 deg/decision angular speed of a perpendicular strafer at 400 px.
# Coarse 30 deg turns around in 6 decisions (0.4 s); gcd(30, 2.5) = 2.5, so every
# multiple of 2.5 deg is reachable: 144 barrel directions instead of 6.
TURN_COARSE_DEG: float = 30.0               # per decision
TURN_FINE_DEG: float = 2.5                  # per decision
# endregion BLOCK_CONSTANTS_AIM

# region BLOCK_CONSTANTS_LEAGUE
# Self-play league: the trainer drops policy snapshots here, env workers sample
# opponents from them. Generation 0 is a frozen random-weight network — both the
# first opponent and the permanent yardstick.
LEAGUE_DIRNAME: str = "opponents"
LEAGUE_SNAPSHOT_INTERVAL: int = 2_000_000   # env steps between snapshots
LEAGUE_MAX_GENERATIONS: int = 16            # hall of fame size after thinning
LEAGUE_RECENT_KEEP: int = 8                 # newest generations always kept
LEAGUE_LATEST_SHARE: float = 0.3            # play the newest generation
LEAGUE_SCRIPTED_SHARE: float = 0.0          # no scripted bot in TRAINING; eval only
LEAGUE_CACHE_SIZE: int = 3                  # loaded models kept per worker
# endregion BLOCK_CONSTANTS_LEAGUE

# region BLOCK_CONSTANTS_CURRICULUM
# Map mixture over training progress. Compact maps put agents in each other's
# faces so that a randomly aimed shot still lands; large maps are where fog of
# war, memory and rotation matter. Neither end ever drops to zero: a hard stage
# switch would strand the league in a different game.
CURRICULUM_STEPS: int = 60_000_000
COMPACT_SHARE_START: float = 0.8
COMPACT_SHARE_END: float = 0.2
EVAL_COMPACT_SHARE: float = 0.5             # arena_eval always uses this fixed mixture
# endregion BLOCK_CONSTANTS_CURRICULUM

# region BLOCK_CONSTANTS_NETWORK
ATTN_DIM: int = 128
ATTN_HEADS: int = 4
ATTN_LAYERS: int = 2
FEATURES_DIM: int = 256
LSTM_HIDDEN: int = 256
# endregion BLOCK_CONSTANTS_NETWORK

# region BLOCK_CONSTANTS_REWARD
# Outcome rewards only. No line-of-sight bonus, no proximity bonus, no cowardice
# penalty: with real opponents in the league, tactics have to emerge from winning,
# not from hand-written instructions about how to behave.
TAU_TEAM: float = 0.6
R_WIN: float = 10.0
R_LOSS: float = -10.0
# A win on the frame limit pays half of a win by score or elimination. Before this,
# any 1-point lead at timeout earned the full R_WIN — a leading team had no reason
# to close the match out, and turtling was subsidised at the outcome level.
R_WIN_TIMEOUT: float = 5.0
R_LOSS_TIMEOUT: float = -5.0
# CP events are zero-sum since experiment 10: a capture pays the capturer +3 and the
# other team -3, whether the point was neutral or owned. The old extra my_lost term
# made losing an owned point cost -6 against +3 for taking it — a defence premium the
# self-play meta would have absorbed as a bias, and a net-negative drift that polluted
# ep_rew_mean as an instrument.
R_CP_CAPTURED: float = 3.0
R_CP_LOST: float = -3.0
# Individual credit for being inside the circle at the capture frame. Still an outcome,
# not behaviour shaping: the capture happened and these agents are the ones who made it.
# Without it, kills were personal but objectives were not, and a free-rider collected
# 60% of the team stream from spawn.
R_CP_CAPTURE_INDIV: float = 1.0
R_SCORE_DELTA: float = 0.15
R_KILL: float = 1.0
R_DEATH: float = -1.0
R_DAMAGE_DEALT: float = 0.005      # per HP
R_DAMAGE_TAKEN: float = -0.002     # per HP
# A timeout draw is the mutual-passivity equilibrium of pure self-play: nobody
# acts, nobody scores, nobody learns. Making it mildly bad is an outcome rule —
# it says standing around is a poor result, not how to play. Also paid on the
# (rare) simultaneous elimination of both teams, which is an explicit draw now.
R_TIMEOUT_DRAW: float = -1.0
# endregion BLOCK_CONSTANTS_REWARD

# region BLOCK_CONSTANTS_PLAYER
PLAYER_RADIUS: int = 12
PLAYER_SPEED: float = 4.5
PLAYER_HP: int = 100
PLAYER_AMMO: int = 9999
SHOOT_COOLDOWN_MS: int = 200
# endregion BLOCK_CONSTANTS_PLAYER

# region BLOCK_CONSTANTS_SHIELD
# Shield over HP, Halo-model: absorbs damage first, regenerates after a pause out of
# combat. This is what gives a fight rounds — retreat has a payoff (shield comes back),
# pushing a broken shield before it regens is a timing decision, and focused fire kills
# while the same damage spread over five targets regenerates. Sized deliberately small:
# 40 over 100 HP keeps time-to-kill close to the old game so the combat bootstrap
# measured in experiment 9 still holds.
SHIELD_MAX: float = 40.0
SHIELD_REGEN_DELAY_FRAMES: int = 180        # 3 s without taking damage
SHIELD_REGEN_PER_FRAME: float = 0.335       # ~20 HP/s -> full shield in 2 s
# endregion BLOCK_CONSTANTS_SHIELD

# region BLOCK_CONSTANTS_BULLET
BULLET_RADIUS: int = 3
BULLET_SPEED: float = 24.0
BULLET_DAMAGE: int = 10
BULLET_LIFETIME_MS: int = 3000
# endregion BLOCK_CONSTANTS_BULLET

# region BLOCK_CONSTANTS_COLORS
COLOR_BG: tuple[int, int, int] = (30, 30, 35)
COLOR_WALL: tuple[int, int, int] = (80, 80, 95)
COLOR_WALL_BORDER: tuple[int, int, int] = (55, 55, 65)
COLOR_PLAYER_1: tuple[int, int, int] = (50, 180, 220)
COLOR_PLAYER_2: tuple[int, int, int] = (220, 80, 60)
COLOR_BULLET_1: tuple[int, int, int] = (120, 220, 255)
COLOR_BULLET_2: tuple[int, int, int] = (255, 160, 120)
COLOR_HP_BAR: tuple[int, int, int] = (60, 200, 80)
COLOR_HP_BG: tuple[int, int, int] = (60, 60, 60)
COLOR_HUD_TEXT: tuple[int, int, int] = (200, 200, 200)
COLOR_GRID: tuple[int, int, int] = (38, 38, 44)
COLOR_CROSSHAIR: tuple[int, int, int] = (255, 255, 100)
COLOR_CP_NEUTRAL: tuple[int, int, int] = (180, 180, 180)
COLOR_CP_BLUE: tuple[int, int, int] = (80, 160, 240)
COLOR_CP_RED: tuple[int, int, int] = (240, 100, 80)
COLOR_CP_CONTESTED: tuple[int, int, int] = (255, 200, 50)
# endregion BLOCK_CONSTANTS_COLORS
