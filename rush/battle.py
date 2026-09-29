"""Play one big battle between two players and record it for the viewer.

A player is a soldier network commanded by the army commander (rush.army_commander):
  hf:<repo>[/<variant>][@rev]   a network from the Hugging Face Hub
  net:<dir>                     a network from a local folder (config.json + model.safetensors, or ckpt_*_latest.pt)
  random                        an untrained network (a baseline and for tests)
optionally followed by "#Label" for the name shown in the viewer.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from rush.armies import TARGET_CAPS, assign_forks, parse_army, resolve_map
from rush.config import ACTION_REPEAT, FPS
from rush.forks12 import FORKS, trait_vector
from rush.replay11 import BigReplayWriter
from rush.world15 import Rules15, WorldBig15

logger = logging.getLogger(__name__)


def build_world(map_name: str, device: str, seed: int, record: bool = True) -> tuple[Any, Any, list, list]:
    """WorldBig15 with all 8 classes per army, their traits and magazines."""
    big_map = resolve_map(map_name)
    world = WorldBig15(big_map, device=device, seed=seed, record=record, rules15=Rules15())
    T, A = world.T, world.A
    blue, red = parse_army("all"), parse_army("all")
    forks, agent_fork = assign_forks(T, blue, red)
    world.set_traits(torch.tensor([trait_vector(forks[k]) for k in agent_fork], dtype=torch.float32).view(1, A, -1))
    world.refill()
    caps = torch.tensor([TARGET_CAPS[forks[k]] for k in agent_fork], dtype=torch.float32, device=device).view(1, A)
    world.set_ammo_caps(caps)
    world.fill_ammo()
    return big_map, world, forks, agent_fork


def make_player(spec: str, world: Any, seed: int) -> tuple[Any, str]:
    spec, _, label = spec.partition("#")
    from rush.hub import load_soldier, random_soldier
    from rush.net_player import NetPlayer
    if spec == "random":
        return NetPlayer(world, random_soldier(world.pos.device, seed)), label or "Untrained"
    if spec.startswith("hf:") or spec.startswith("net:"):
        policy, name = load_soldier(spec, world.pos.device)
        return NetPlayer(world, policy), label or name
    raise SystemExit(f"unknown player {spec!r}: use hf:<repo>, net:<dir> or random")


def _extras(world: Any, fired: np.ndarray, aimed: np.ndarray) -> dict:
    return {"fired": fired.astype(np.int64), "aimed": aimed.astype(np.int64),
            "ammo": world.ammo[0].detach().cpu().numpy(), "ammo_cap": world.ammo_cap[0].detach().cpu().numpy()}


def _writer(world: Any, forks: list, agent_fork: list, labels: list[str], seed: int) -> BigReplayWriter:
    T, A, wm = world.T, world.A, world._wm
    geo0, snap = world.map_geometry(0), world.snapshot(0)
    geo = {"map_idx": 0, "name": geo0["name"], "w": float(geo0["arena_w"]), "h": float(geo0["arena_h"]), "tile": geo0["tile_size"],
           "cols": geo0["cols"], "rows": geo0["rows"], "walls": geo0["walls"],
           "cps": list(zip(snap["cps"]["x"].tolist(), snap["cps"]["y"].tolist())), "cp_radius": geo0["cp_radius"],
           "sectors": geo0["sectors"]}
    rules = {"team_size": T, "score_to_win": float(wm.SCORE_TO_WIN), "team_lives": int(wm.TEAM_LIVES), "max_frames": int(wm.MAX_FRAMES),
             "exp": 15, "ammo": True, "refill_s": world.rules15.refill_s, "caps": dict(TARGET_CAPS)}
    w = BigReplayWriter(geo, seed=seed, labels=labels, source="rush", fps=FPS, frames_per_decision=ACTION_REPEAT,
                        rules=rules, decimate=2)
    w.header["map"]["cp_names"] = geo0["cp_names"]
    fork_of = np.array(agent_fork)
    n_blue = len(parse_army("all"))
    w.header["forks"] = [{"name": nm, "title_ru": FORKS[nm].title_ru, "blurb_ru": FORKS[nm].blurb_ru, "team": 0 if i < n_blue else 1,
                          "traits": dict(FORKS[nm].traits), "count": int((fork_of == i).sum()), "source": "rush"}
                         for i, nm in enumerate(forks)]
    w.header["agent_fork"] = agent_fork
    w.header["agent_radius"] = [round(float(v), 2) for v in world._radius_a[0].tolist()]
    w.header["agent_max_hp"] = [round(float(v), 1) for v in world._max_hp[0].tolist()]
    w.add(snap, None, _extras(world, np.zeros(A, bool), np.zeros(A, bool)))
    return w


## @io blue / red player specs, map, seed, device, max_decisions (0 = the whole match), out (replay path or None)
##   -> result dict (score, winner, kills, decisions, wall seconds)
def play(blue: str, red: str, map_name: str = "Warfront500", seed: int = 0, device: str = "cpu", max_decisions: int = 0,
         out: str | Path | None = None, log_every: int = 250) -> dict:
    t0 = time.perf_counter()
    _, world, forks, agent_fork = build_world(map_name, device, seed, record=out is not None)
    A = world.A
    pb, lb = make_player(blue, world, seed)
    pr, lr = make_player(red, world, seed + 1)
    for side, p in ((0, pb), (1, pr)):                          # each side's commander plans only its own team
        p.plan_team = side
    writer = _writer(world, forks, agent_fork, [lb, lr], seed) if out is not None else None
    is_blue = (world._agent_team == 0).to(world.pos.device).view(1, A, 1)
    kills = np.zeros(2)
    team_of = world._agent_team.cpu().numpy()
    d, done = 0, False
    while not done:
        with torch.no_grad():
            ab, _ = pb.act(world)
            ar, _ = pr.act(world)
        acts = torch.where(is_blue, ab, ar)
        o = world.step(acts)
        d += 1
        inf = o.info
        kills += np.bincount(team_of, weights=inf["kills"][0].cpu().numpy(), minlength=2)
        if writer is not None:
            evs = o.events[0] if o.events else []
            writer.add(world.snapshot(0), [e for e in evs if e.get("type") != "shot"],
                       _extras(world, inf["shots"][0].cpu().numpy() > 0, inf["aimed_shots"][0].cpu().numpy() > 0))
        if log_every and d % log_every == 0:
            logger.info(f"decision {d}: score {[round(float(s)) for s in world.score[0]]} kills {kills.astype(int).tolist()}")
        done = bool(inf["done_cpu"][0]) or bool(max_decisions and d >= max_decisions)
    score = [float(s) for s in world.score[0]]
    win = int(o.info["winner"][0])
    if win == 0:
        win = 1 if score[0] > score[1] else (2 if score[1] > score[0] else 0)
    res = {"blue": lb, "red": lr, "map": map_name, "seed": seed, "decisions": d, "winner": ["draw", "blue", "red"][win],
           "score": [round(s, 1) for s in score], "kills": kills.astype(int).tolist(), "wall_s": round(time.perf_counter() - t0, 1)}
    if writer is not None:
        writer.finish(win, "score")
        writer.save(Path(out))
        res["replay"] = str(out)
    logger.info("result: " + json.dumps(res, ensure_ascii=False))
    return res
