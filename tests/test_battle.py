"""Smoke tests: short battles play, are recorded and decoded; the commander issues orders; the gate is sound;
self-play runs; a network loads from the Hub."""
from __future__ import annotations

import gzip
import json
import os

import pytest

from rush.battle import play
from rush.replay11 import load_replay_v2


def test_short_battle_is_recorded(tmp_path):
    out = tmp_path / "b.arena.bin.gz"
    r = play("random#Blue", "random#Red", "Front50", seed=3, device="cpu", max_decisions=40, out=out, log_every=0)
    assert r["decisions"] == 40 and out.exists()
    d = load_replay_v2(gzip.open(out).read())
    assert d["n_agents"] == 100 and [t["label"] for t in d["teams"]] == ["Blue", "Red"]


def test_same_seed_same_battle():
    a = play("random", "random", "Front50", seed=5, device="cpu", max_decisions=120, out=None, log_every=0)
    b = play("random", "random", "Front50", seed=5, device="cpu", max_decisions=120, out=None, log_every=0)
    assert a["score"] == b["score"] and a["kills"] == b["kills"]


def test_commander_orders_every_soldier():
    import torch

    from rush.army_commander import ArmyCommander
    from rush.battle import build_world
    _, world, _, _ = build_world("Front50", "cpu", 1, record=False)
    world.observe()
    t = ArmyCommander(world).act(world)
    assert bool((t[world.alive] >= 0).all()) and bool((world.order_cp[world.alive] >= 0).all())
    assert int(torch.unique(t).numel()) > 1                  # squads spread over several points


def test_gate_equal_players_is_undecided(tmp_path, monkeypatch):
    """Identical margins of 0 (e.g. battles too short for contact) must not reject the candidate."""
    from rush import gate
    lg = tmp_path / "league"
    for d in ("gen_000", "cand_00005"):
        (lg / d).mkdir(parents=True)
    (lg / "current.json").write_text(json.dumps({"gen": 0, "dir": "gen_000"}))
    (lg / "candidates.jsonl").write_text(json.dumps({"iter": 5, "dir": "cand_00005"}) + "\n")
    monkeypatch.setattr(gate, "play", lambda *a, **k: {"score": [10.0, 10.0]})
    v = gate.run_gate(tmp_path, max_pairs=3)
    assert v["decision"] == "undecided" and len(v["pairs"]) == 3 and "new_gen" not in v


def test_selfplay_two_iterations(tmp_path):
    import torch

    from rush import selfplay
    from rush.hub import random_soldier
    from rush.model11 import policy_payload
    init = tmp_path / "init.pt"
    torch.save(policy_payload(random_soldier("cpu").model, 0), init)
    selfplay.main(["--init", str(init), "--run-dir", str(tmp_path / "run"), "--maps", "Front50", "--worlds", "2",
                   "--rollout", "20", "--bptt", "10", "--minibatch-rows", "32", "--max-iters", "2", "--gen-every", "2",
                   "--amp", "0", "--device", "cpu"])
    rows = [json.loads(x) for x in (tmp_path / "run" / "models15" / "metrics.jsonl").read_text().splitlines()]
    assert [r["warmup"] for r in rows] == [True, False]
    assert (tmp_path / "run" / "league" / "cand_00002" / "ckpt_base_latest.pt").exists()


@pytest.mark.skipif(os.environ.get("RUSH_OFFLINE") == "1", reason="needs the Hugging Face Hub")
def test_soldier_from_the_hub():
    from rush.hub import load_soldier
    try:
        load_soldier("hf:koskokos/rush-soldier", "cpu")
    except Exception as e:                                   # no network / model not published yet
        pytest.skip(f"hub unavailable: {e}")
    r = play("hf:koskokos/rush-soldier", "random", "Front50", seed=1, device="cpu", max_decisions=20, out=None, log_every=0)
    assert r["decisions"] == 20
