# FILE: rush/league_elo.py
# region MODULE_CONTRACT [DOMAIN(7): Evaluation; CONCEPT(8): LeagueRating; TECH(4): numpy]
## @modulecontract
## @purpose Ratings of every player of a self-play league on the Elo scale (a Bradley–Terry fit), from all its battles:
## gate / evaluation battles (league/matches.jsonl) and training battles (league/train_matches.jsonl, where the moving
## learner is credited to the candidate it became at the end of that block of iterations). A battle's result is soft:
## s = 0.5 + control margin / 2 (the average share of control points held, over the whole battle, minus the
## opponent's), which carries far more information than win / loss; battles without it fall back to 1 / 0 / 0.5.
## @output {player: rating} with the anchor (generation 0 by default) at 1000; 400 points = 10:1 odds
# endregion MODULE_CONTRACT

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

ANCHOR = "gen_000"
SCALE = 400.0 / math.log(10.0)               # Elo points per unit of the logistic's argument


def soft_result(margin: float | None, winner_is_a: bool | None) -> float:
    if margin is not None:
        return float(min(1.0, max(0.0, 0.5 + 0.5 * margin)))
    return 0.5 if winner_is_a is None else (1.0 if winner_is_a else 0.0)


def candidate_of(it: int, gen_every: int) -> str:
    """The candidate a training iteration's learner turns into (cand_<the next multiple of gen_every>)."""
    return f"cand_{int(math.ceil(max(1, it) / gen_every) * gen_every):05d}"


## @io league dir -> list of (player a, player b, result of a in [0, 1], weight)
def load_matches(league: Path, gen_every: int = 5, train_weight: float = 1.0) -> list[tuple[str, str, float, float]]:
    out: list[tuple[str, str, float, float]] = []
    alias: dict[str, str] = {}                                        # a promoted candidate IS its generation
    mp = league / "matches.jsonl"
    rows = [json.loads(x) for x in mp.read_text().splitlines() if x.strip()] if mp.exists() else []
    for m in rows:
        if m.get("kind") == "alias":
            alias[m["red"]] = m["blue"]
    name = lambda p: alias.get(p, p)                                  # noqa: E731
    for m in rows:
        if m.get("kind") == "alias":
            continue
        out.append((name(m["blue"]), name(m["red"]),
                    soft_result(m.get("control_margin"), None if m.get("winner") in (None, 0) else m["winner"] == 1), 1.0))
    tp = league / "train_matches.jsonl"
    if tp.exists() and train_weight > 0:
        for line in tp.read_text().splitlines():
            if not line.strip():
                continue
            m = json.loads(line)
            opp = m["opponent"]
            opp = f"gen_{int(opp[3:]):03d}" if opp.startswith("gen") and opp[3:].isdigit() else opp
            me = candidate_of(int(m["iter"]) + 1, gen_every)          # "iter" is the iteration being collected (0-based)
            lm = m.get("learner_control_margin")
            out.append((name(me), name(opp), soft_result(lm, bool(m["learner_won"])), train_weight))
    return out


## @io matches -> {player: Elo rating}, the anchor at 1000 (if present); a weak prior keeps unconnected players finite
def fit_ratings(matches: list[tuple[str, str, float, float]], anchor: str = ANCHOR, prior: float = 0.01,
                iters: int = 2000) -> dict[str, float]:
    players = sorted({p for a, b, _, _ in matches for p in (a, b)})
    if not players:
        return {}
    idx = {p: i for i, p in enumerate(players)}
    ia = np.array([idx[a] for a, _, _, _ in matches])
    ib = np.array([idx[b] for _, b, _, _ in matches])
    s = np.array([r for _, _, r, _ in matches], dtype=float)
    w = np.array([x for _, _, _, x in matches], dtype=float)
    r = np.zeros(len(players))
    lr = 0.5
    for _ in range(iters):                                           # gradient ascent on the weighted log-likelihood
        p = 1.0 / (1.0 + np.exp(-(r[ia] - r[ib])))
        g = np.zeros_like(r)
        np.add.at(g, ia, w * (s - p))
        np.add.at(g, ib, -w * (s - p))
        g -= prior * r
        r += lr * g / max(1.0, w.sum() / len(players))
    base = r[idx[anchor]] if anchor in idx else r.mean()
    return {p: round(1000.0 + SCALE * (r[i] - base), 1) for p, i in idx.items()}


def rate_league(league: Path, gen_every: int = 5) -> dict:
    m = load_matches(league, gen_every)
    ratings = fit_ratings(m)
    games: dict[str, int] = {}
    for a, b, _, _ in m:
        games[a] = games.get(a, 0) + 1
        games[b] = games.get(b, 0) + 1
    res = {"t": int(__import__("time").time()), "battles": len(m),
           "ratings": dict(sorted(ratings.items(), key=lambda kv: -kv[1])), "games": games}
    (league / "ratings.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
    return res
