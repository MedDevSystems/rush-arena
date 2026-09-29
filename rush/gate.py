"""The generation gate of self-play, run locally: is the newest candidate stronger than the current generation?

A pair = two full battles on the same seed with the sides swapped (the same start cancels most of the map's side
bias); the pair's margin = the candidate's score minus the generation's, summed over both battles. Pairs are added
one by one; after each (from the second on) a one-sided paired t-test at 5 %: t > t_crit -> the candidate becomes
the next generation (league/gen_XXX + league/current.json, which the running trainer picks up), t < -t_crit -> it is
weaker; undecided after --max-pairs -> not promoted.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import shutil
import sys
import time
from pathlib import Path

from rush.battle import play

T_CRIT_95 = {1: 6.314, 2: 2.920, 3: 2.353, 4: 2.132, 5: 2.015, 6: 1.943, 7: 1.895, 8: 1.860, 9: 1.833, 10: 1.812}


def run_gate(run_dir: Path, candidate: str = "", map_name: str = "Warfront500", seed: int = 200, max_pairs: int = 6,
             device: str = "cpu", max_decisions: int = 0) -> dict:
    lg = run_dir / "league"
    cur = json.loads((lg / "current.json").read_text())
    gen = int(cur["gen"])
    cands = [json.loads(x) for x in (lg / "candidates.jsonl").read_text().splitlines() if x.strip()]
    cand = candidate or cands[-1]["dir"]
    cd, gd = lg / cand, lg / cur["dir"]
    lc, lgn = f"Candidate {cand}", f"Generation {gen}"
    diffs, pairs, decision, t = [], [], "undecided", None
    for k in range(max_pairs):
        rec = lg / f"{cand}_vs_gen{gen}_pair{k}" if k == 0 else None
        ra = play(f"net:{cd}#{lc}", f"net:{gd}#{lgn}", map_name, seed + k, device, max_decisions,
                  f"{rec}_a.arena.bin.gz" if rec else None, log_every=0)
        rb = play(f"net:{gd}#{lgn}", f"net:{cd}#{lc}", map_name, seed + k, device, max_decisions,
                  f"{rec}_b.arena.bin.gz" if rec else None, log_every=0)
        d = (ra["score"][0] - ra["score"][1]) + (rb["score"][1] - rb["score"][0])
        diffs.append(d)
        pairs.append({"seed": seed + k, "margin": round(d, 1), "cand_blue": ra["score"], "cand_red": rb["score"]})
        n = len(diffs)
        if n >= 2:
            mu = sum(diffs) / n
            sd = math.sqrt(sum((x - mu) ** 2 for x in diffs) / (n - 1))
            t = mu / (sd / math.sqrt(n)) if sd > 0 else (math.inf if mu > 0 else (-math.inf if mu < 0 else 0.0))
            tc = T_CRIT_95.get(n - 1, 1.645)
            print(f"pair {n}: mean margin {mu:.0f}, t = {t:.2f} (critical {tc})", flush=True)
            if t > tc:
                decision = "promote"
                break
            if t < -tc:
                decision = "reject"
                break
    verdict = {"t": int(time.time()), "candidate": cand, "vs_gen": gen, "pairs": pairs, "decision": decision,
               "t_stat": None if t is None or not math.isfinite(t) else round(t, 3)}
    if decision == "promote":
        new = max([int(p.name[4:]) for p in lg.glob("gen_*") if p.name[4:].isdigit()] + [gen]) + 1
        gdir = lg / f"gen_{new:03d}"
        gdir.mkdir(parents=True, exist_ok=True)
        shutil.copy(cd / "ckpt_base_latest.pt", gdir / "ckpt_base_latest.pt")
        (lg / "current.json").write_text(json.dumps({"gen": new, "dir": gdir.name, "from": cand, "t": int(time.time())}))
        verdict["new_gen"] = new
    with open(lg / "gates.jsonl", "a") as f:
        f.write(json.dumps(verdict) + "\n")
    return verdict


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="rush-gate", description="Gate the newest self-play candidate against the current generation.")
    p.add_argument("--run-dir", required=True)
    p.add_argument("--candidate", default="", help="a league/cand_XXXXX dir (default: the newest)")
    p.add_argument("--map", default="Warfront500")
    p.add_argument("--seed", type=int, default=200)
    p.add_argument("--max-pairs", type=int, default=6)
    p.add_argument("--max-decisions", type=int, default=0)
    p.add_argument("--device", default="cuda")
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.WARNING)
    v = run_gate(Path(a.run_dir), a.candidate, a.map, a.seed, a.max_pairs, a.device, a.max_decisions)
    print(json.dumps(v))
    return 0


if __name__ == "__main__":
    sys.exit(main())
