"""Command line: `rush-battle` plays and records a battle, `rush-view` opens replays in the browser."""
from __future__ import annotations

import argparse
import functools
import http.server
import logging
import shutil
import sys
import tempfile
import threading
import webbrowser
from pathlib import Path

VIEWER = Path(__file__).resolve().parent / "viewer"


def battle(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="rush-battle", description="Play a battle between two armies and record it.")
    p.add_argument("--blue", default="hf:koskokos/rush-soldier", help="hf:<repo>[/<variant>] | net:<dir> | random, optional #Label")
    p.add_argument("--red", default="hf:koskokos/rush-soldier")
    p.add_argument("--map", default="Warfront500", help="Warfront500 (500 v 500), Front150, Crossing150, Front50, or <theme>:<seed>")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if _cuda() else "cpu")
    p.add_argument("--max-decisions", type=int, default=0, help="stop early (a full Warfront500 match is ~7000)")
    p.add_argument("--out", default="battle.arena.bin.gz", help="replay file ('' = do not record)")
    p.add_argument("--view", action="store_true", help="open the replay in the browser when done")
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", stream=sys.stdout)
    for name in ("rush",):
        logging.getLogger(name).setLevel(logging.INFO)
    from rush.battle import play
    res = play(a.blue, a.red, a.map, a.seed, a.device, a.max_decisions, a.out or None)
    print(f"{res['blue']} {res['score'][0]:.0f} : {res['score'][1]:.0f} {res['red']} — winner {res['winner']}, "
          f"kills {res['kills'][0]} : {res['kills'][1]}, {res['decisions']} decisions, {res['wall_s']} s")
    if a.view and a.out:
        view([a.out])
    return 0


def view(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="rush-view", description="Open a replay in the browser (a local static server).")
    p.add_argument("replay", nargs="?", default="", help=".arena.bin.gz file (empty = the bundled demo)")
    p.add_argument("--port", type=int, default=8766)
    a = p.parse_args(argv)
    root = Path(tempfile.mkdtemp(prefix="rush-view-"))                    # a copy: the package dir may be read-only
    shutil.copytree(VIEWER, root, dirs_exist_ok=True)
    src = ""
    if a.replay:
        dst = root / "viewer" / "samples" / "big"
        dst.mkdir(parents=True, exist_ok=True)
        shutil.copy(a.replay, dst / Path(a.replay).name)
        src = "?src=samples/big/" + Path(a.replay).name
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(root))
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", a.port), handler)
    url = f"http://127.0.0.1:{a.port}/viewer/big.html{src}"
    print(f"viewer at {url} (Ctrl+C to stop)")
    threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


def selfplay(argv: list[str] | None = None) -> int:
    """rush-selfplay: generational self-play from a published soldier (or --resume a run). Unknown options go to the
    trainer (rush.selfplay: --maps, --worlds, --rollout, --gen-every, --lr, ...)."""
    p = argparse.ArgumentParser(prog="rush-selfplay", description=selfplay.__doc__)
    p.add_argument("--init", default="hf:koskokos/rush-soldier", help="hf:<repo>[/<variant>] or net:<dir> to start from")
    p.add_argument("--run-dir", default="runs/selfplay")
    p.add_argument("--resume", action="store_true", help="continue the run in --run-dir")
    a, rest = p.parse_known_args(argv)
    run = Path(a.run_dir)
    run.mkdir(parents=True, exist_ok=True)
    from rush import selfplay as sp
    if a.resume:
        return sp.main(["--resume", str(run), "--log-file", str(run / "selfplay.log"), *rest])
    from rush.hub import _local_dir
    src, _ = _local_dir(a.init)
    return sp.main(["--init", str(src), "--run-dir", str(run), "--log-file", str(run / "selfplay.log"), *rest])


def gate(argv: list[str] | None = None) -> int:
    from rush.gate import main
    return main(argv)


def _cuda() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False
