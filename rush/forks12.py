from __future__ import annotations

# region MODULE_CONTRACT [DOMAIN(8): GameDesign; CONCEPT(9): ForkPopulation; TECH(6): Python]
## @modulecontract
## @purpose The population of experiment 12: eight forks of the exp11b network, each with its own body
## traits (multipliers over TRAITS12) and its own reward emphasis (multipliers over REWARD_KEYS12).
## @scope Pure data + helpers. No torch, no world: world12 reads trait_vector(), the trainer reads
## reward_weights() / team_spirit_mult(). The point-buy that keeps the forks fair lives in tools/balance12.py.
## @input none
## @output FORKS, TRAITS12, REWARD_KEYS12, BASE_REWARD12, trait_vector, reward_weights, validate
## @links LINKS_TO: docs/EXP12_CONTRACT.md (section D), docs/EXP12_FORKS.md, tools/balance12.py,
## world11.REWARD_PRESETS["exp11b"], config.R_* (base reward values)
## @invariants
## - every trait multiplier lies in [0.6, 1.4] (±40%), bullet_range in [0.8, 1.2] (±20%, v0.2.0), every
##   reward multiplier in [0.5, 2.0] and > 0
##   (emphasis changes, sign never does), fork "base" is all 1.0
## - exactly one fork differs from base ONLY in reward, exactly one ONLY in traits
## - the order of TRAITS12 is the contract's order (world12 indexes the trait tensor by it)
## @rationale
## Q: Why multipliers for rewards instead of absolute weights?
## A: exp11b's weights live in two channels (combat: kill/death/damage; objective: capture/score/win) and the
## A: trainer anneals a combat boost and team spirit on top. A multiplier composes with those schedules and
## A: keeps the zero-sum structure: each agent values its own events AND the enemy mean with its own weights.
## Q: What does "capture" scale?
## A: Both capture payouts of world11: the individual R_CP_CAPTURE_INDIV (agents inside the captured point)
## A: and the team R_CP_CAPTURED / R_CP_LOST. "score_delta" scales R_SCORE_DELTA, "win" scales R_WIN/R_LOSS
## A: (and the timeout variants), "team_spirit" scales the trainer's τ schedule (clamped to 1).
## Q: What does "bullet_range" mean next to "bullet_speed"?
## A: bullet_range multiplies the maximum travel distance (world12 Rules12.range_px, 2400 px at 1.0 since
## A: v0.2.0; 8640 before); bullet_speed multiplies the speed (36 px/frame at 1.0). Lifetime = range / speed,
## A: so the two are independent. Each agent's vision equals its own range (world12).
## @changes
## LAST_CHANGE: [v0.2.0] Range 2400 px, bullet_range within ±20% (sniper 1.3 -> 1.2, assault 0.7 -> 0.8);
## rebalanced at the new range: heavy dash_cooldown 1.2, tank max_hp 1.35 + body 1.15, scout body 0.85, guardian
## max_hp 1.15 (tools/balance12.py: max|power| 0.138, worst pair assault-tank 0.602, docs/exp12_beliefs_balance_r2.json).
## PREV_CHANGE: [v0.1.0] Eight forks, point-buy balanced in tools/balance12.py.
# endregion MODULE_CONTRACT
# GREP_SUMMARY: forks, population, traits, reward emphasis, archetypes, experiment 12

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# region BLOCK_KEYS
TRAITS12: list[str] = ["fire_rate", "bullet_range", "bullet_speed", "move_speed",
                       "body_radius", "max_hp", "damage", "dash_cooldown"]
REWARD_KEYS12: list[str] = ["kill", "death", "damage_dealt", "damage_taken",
                            "capture", "score_delta", "win", "team_spirit"]
# exp11b absolute values the multipliers apply to (world11 RewardConfig "exp11b" + config.R_*).
# capture = R_CP_CAPTURE_INDIV (individual; R_CP_CAPTURED=3.0 on the team channel scales with it),
# team_spirit = the trainer's τ schedule (0.2 -> 0.8), given as 1.0 = "use the schedule".
BASE_REWARD12: dict[str, float] = {"kill": 1.0, "death": -1.0, "damage_dealt": 0.02, "damage_taken": -0.02,
                                   "capture": 1.0, "score_delta": 0.15, "win": 10.0, "team_spirit": 1.0}
TRAIT_LIMITS: tuple[float, float] = (0.6, 1.4)
# Per-trait narrower limits (v0.2.0, user: "soldiers shoot across the whole map"): range within ±20%.
TRAIT_LIMITS_BY: dict[str, tuple[float, float]] = {"bullet_range": (0.8, 1.2)}
REWARD_LIMITS: tuple[float, float] = (0.5, 2.0)
# endregion BLOCK_KEYS


# region CLASS_ForkSpec
@dataclass(frozen=True)
class ForkSpec:
    name: str
    title_ru: str
    traits: dict[str, float] = field(default_factory=dict)   # missing trait = 1.0
    reward: dict[str, float] = field(default_factory=dict)   # missing key = 1.0 (multiplier on BASE_REWARD12)
    blurb_ru: str = ""

    def trait(self, k: str) -> float:
        return float(self.traits.get(k, 1.0))

    def reward_mult(self, k: str) -> float:
        return float(self.reward.get(k, 1.0))
# endregion CLASS_ForkSpec


# region BLOCK_FORKS
## Numbers come from the point-buy in tools/balance12.py: every fork's power index is within ±0.15 logit of
## base (≈ ±4 percentage points of duel win rate) — see docs/EXP12_FORKS.md for the table.
FORKS: dict[str, ForkSpec] = {f.name: f for f in [
    ForkSpec("base", "Контроль",
             blurb_ru="Сеть exp11b как есть: обычное тело, обычные награды. Эталон, от которого меряем остальных."),
    ForkSpec("hunter", "Охотник",
             reward={"kill": 1.6, "death": 1.2, "capture": 0.8, "score_delta": 0.8},
             blurb_ru="Тело обычное, отличается только наградой: больше за убийство, больнее за смерть, меньше за точки."),
    ForkSpec("heavy", "Пулемётчик",
             traits={"fire_rate": 1.3, "bullet_range": 1.1, "move_speed": 0.7, "body_radius": 1.2, "max_hp": 1.2,
                     "dash_cooldown": 1.2},
             blurb_ru="Пример пользователя: +30% скорострельность, +10% дальность, −30% скорость, +20% тело; "
                      "за это +20% здоровья и рывок реже. Награды обычные — отличается только телом."),
    ForkSpec("sniper", "Снайпер",
             traits={"bullet_speed": 1.4, "bullet_range": 1.2, "damage": 1.3, "fire_rate": 0.7, "max_hp": 0.85},
             reward={"damage_dealt": 1.5, "kill": 1.2, "death": 1.3, "capture": 0.8},
             blurb_ru="Быстрая дальняя пуля и тяжёлый урон, но стреляет редко и хрупок. Награда за урон и за то, чтобы выжить."),
    ForkSpec("tank", "Танк",
             traits={"max_hp": 1.35, "body_radius": 1.15, "move_speed": 0.8, "dash_cooldown": 1.3},
             reward={"capture": 1.45, "score_delta": 1.3, "damage_taken": 0.6, "kill": 0.8},
             blurb_ru="Много здоровья, крупный и медленный. Награда за точки и удержание; урон по себе почти не штрафуется."),
    ForkSpec("scout", "Разведчик",
             traits={"move_speed": 1.2, "dash_cooldown": 0.7, "body_radius": 0.85, "max_hp": 0.7},
             reward={"capture": 1.4, "score_delta": 1.2, "death": 0.8},
             blurb_ru="Быстрый, часто делает рывок, маленький и хрупкий. Награда за захват дальних точек."),
    ForkSpec("assault", "Штурмовик",
             traits={"fire_rate": 1.3, "bullet_range": 0.8, "bullet_speed": 0.8, "move_speed": 1.05, "damage": 0.9},
             reward={"kill": 1.3, "damage_dealt": 1.3, "capture": 0.9},
             blurb_ru="Скорострельный ближний бой, короткая и медленная пуля. Награда за убийства и урон."),
    ForkSpec("guardian", "Страж",
             traits={"max_hp": 1.15, "fire_rate": 0.85},
             reward={"team_spirit": 1.25, "win": 1.3, "score_delta": 1.3, "death": 1.5},
             blurb_ru="Чуть крепче и чуть медленнее. Играет на команду: больше общей награды, дорогая смерть."),
]}
# endregion BLOCK_FORKS


# region FUNC_helpers
def trait_vector(name: str) -> list[float]:
    """Multipliers in TRAITS12 order (what world12.set_traits expects per agent)."""
    f = FORKS[name]
    return [f.trait(k) for k in TRAITS12]


def reward_weights(name: str) -> dict[str, float]:
    """Absolute reward weights = BASE_REWARD12 × the fork's multipliers (sign kept; team_spirit is a τ multiplier)."""
    f = FORKS[name]
    return {k: BASE_REWARD12[k] * f.reward_mult(k) for k in REWARD_KEYS12}


def team_spirit_mult(name: str) -> float:
    return FORKS[name].reward_mult("team_spirit")


def validate() -> list[str]:
    """Problems with the fork table (empty = OK)."""
    probs: list[str] = []
    if set(FORKS["base"].traits) or set(FORKS["base"].reward):
        probs.append("base must be all 1.0")
    for f in FORKS.values():
        for k, v in f.traits.items():
            if k not in TRAITS12:
                probs.append(f"{f.name}: unknown trait {k}")
            elif not TRAIT_LIMITS_BY.get(k, TRAIT_LIMITS)[0] <= v <= TRAIT_LIMITS_BY.get(k, TRAIT_LIMITS)[1]:
                probs.append(f"{f.name}: trait {k}={v} outside {TRAIT_LIMITS_BY.get(k, TRAIT_LIMITS)}")
        for k, v in f.reward.items():
            if k not in REWARD_KEYS12:
                probs.append(f"{f.name}: unknown reward key {k}")
            elif not REWARD_LIMITS[0] <= v <= REWARD_LIMITS[1]:
                probs.append(f"{f.name}: reward {k}={v} outside {REWARD_LIMITS}")
    only_reward = [f.name for f in FORKS.values() if f.reward and not f.traits]
    only_traits = [f.name for f in FORKS.values() if f.traits and not f.reward]
    if len(only_reward) != 1:
        probs.append(f"need exactly one reward-only fork, got {only_reward}")
    if len(only_traits) != 1:
        probs.append(f"need exactly one traits-only fork, got {only_traits}")
    logger.info(f"[IMP:9][validate][RESULT] forks={len(FORKS)} reward_only={only_reward} traits_only={only_traits} "
                f"problems={len(probs)} [VALUE]")
    return probs
# endregion FUNC_helpers
