from __future__ import annotations

# region MODULE_CONTRACT [DOMAIN(8): Spectating; CONCEPT(9): ReplayFormat; TECH(7): json+gzip]
## @modulecontract
## @purpose Replay file format of experiment 11 (.arena.json.gz): header (map geometry, team ids,
## generation labels, seed, decision rate), per-decision frames and match events. Writer builds a
## replay from World11.snapshot() dicts, reader restores the same frames for tools and tests.
## @scope Format, quantisation, bullet identity, event derivation by snapshot diffing. No simulation,
## no policies, no rendering.
## @input Snapshot dicts (contract docs/EXP11_CONTRACT.md, section W) and map_geometry() dicts
## @output .arena.json.gz files; decoded dict with absolute integer frames
## @links LINKS_TO: record11 (producer), viewer/replay.js (consumer, mirrors decode()), world11 (snapshot source)
## @invariants
## - Teams are ids 0 and 1 only; colours are the viewer's business (theme), never stored here
## - CP owner / cap_team use the world encoding: 0 neutral, 1 = team 0, 2 = team 1
## - Agent rows are delta-encoded against the previous frame; frame 0 is absolute. decode() and
##   viewer/replay.js must stay byte-for-byte equivalent in how they undo it
## - Every bullet carries an id that is stable across the frames it lives in
## - Event time "t" is a frame index (one frame per decision), never a physics frame number
## @rationale
## Q: Why JSON + gzip instead of a binary format?
## A: The viewer is a static page with no build step. JSON parses natively, DecompressionStream
## A: undoes gzip natively, and a human can still open a replay with zcat | jq. Delta-encoded small
## A: integers compress to ~0.3 MB per 3-minute match — size is not the constraint that would
## A: justify a custom binary reader in two languages.
## Q: Why assign bullet ids here instead of trusting the world's slot index?
## A: The contract's snapshot does not promise ids, and a slot is reused the moment a bullet dies.
## A: Bullets fly straight, so "previous position + velocity x frames_per_decision" predicts the
## A: next position exactly; matching on that gives ids that survive slot reuse and any world.
## Q: Why derive events by diffing instead of instrumenting the world?
## A: The frozen experiment-9/10 worlds cannot be instrumented, and world11's event channel is
## A: still being built. Diffing works for every world; world-provided events (dash) merge in.
## Q: Why a second, binary version for the big battle (schema 2)?
## A: 1000 agents x ~3600 decisions is 25M agent rows: as JSON integers that is hundreds of MB to
## A: parse on the main thread. Schema 2 is one gzip stream of a small JSON header followed by
## A: typed-array chunks (planar, little-endian) that the browser views without parsing, frame
## A: decimation (every k-th decision stored, the player interpolates), and bullets stored once
## A: per bullet (birth, life, origin, velocity) instead of once per frame — a bullet flies straight.
## A: Chunks arrive in time order, so the player starts before the file has finished downloading.
## @changes
## LAST_CHANGE: [v2.0.0] Schema 2 (BigReplayWriter): binary chunked container for up to thousands of
##   agents, frame decimation, per-bullet records, columnar snapshots. Schema 1 unchanged and still read.
## PREV: [v1.0.0] Initial format, schema version 1.
## @modulemap
## CLASS 9[Accumulate snapshots into a replay] => ReplayWriter
## CLASS 9[Schema-2 writer for big battles] => BigReplayWriter
## FUNC 8[Derive hit/kill/capture/respawn events from two snapshots] => diff_events
## FUNC 7[Decode a replay file to absolute frames] => load_replay
## FUNC 7[Decode a schema-2 container] => load_replay_v2
## FUNC 6[Map geometry to wall runs] => wall_runs
## @usecases
## - record11: w = ReplayWriter(geometry, ...); w.add(snapshot); w.finish(result); w.save(path)
## - record_big: w = BigReplayWriter(geometry, ..., decimate=2); w.add(snapshot, world_events); w.finish(...); w.save(path)
## - tests/tools: rep = load_replay(path); rep["frames"][i]["A"]  (schema 1) / rep["x"][frame, agent] (schema 2)
def _module_contract():
    pass
# endregion MODULE_CONTRACT
# GREP_SUMMARY: replay, format, gzip, json, quantise, delta, bullet id, events, kill feed, snapshot
# STRUCTURE: ▶ snapshot → ⚡ quantise → ⚡ match bullets → ⚡ diff events → ⊕ frames → ⎋ .arena.json.gz

import datetime as _dt
import gzip
import json
import logging
import math
import struct
import time
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# region BLOCK_CONSTANTS
FORMAT_NAME: str = "arena-replay"
SCHEMA_VERSION: int = 1
POS_SCALE: int = 2                 # stored = round(px * 2): half-pixel precision
ANGLE_SCALE: float = 2.0           # stored = round(degrees * 2) in [0, 720)
VEL_SCALE: int = 4                 # bullet velocity px/frame * 4
PROGRESS_SCALE: int = 100          # CP progress 0..1 -> 0..100
SCORE_SCALE: int = 10              # score 0..200 -> 0..2000
AGENT_FIELDS: list[str] = ["x", "y", "a", "hp", "sh", "st", "rs"]
BULLET_FIELDS: list[str] = ["id", "x", "y", "vx", "vy", "team", "owner"]
CP_FIELDS: list[str] = ["owner", "cap", "prog"]
ST_ALIVE: int = 1
ST_WAITING: int = 2
ST_DASHING: int = 4                # world11: agent is mid-dash at the snapshot
BULLET_MATCH_TOL: float = 3.0      # px between predicted and observed bullet position
TELEPORT_PX: float = 60.0          # a jump longer than this between decisions is a respawn
# endregion BLOCK_CONSTANTS


# region FUNC_wall_runs
## @purpose Turn a wall tile list into horizontal runs [row, col_start, col_end_inclusive].
## @io list[[col, row]] -> list[[row, c0, c1]]
## @rationale One run per wall segment instead of one pair per tile: the Arena map has ~2 000
## wall tiles but a few hundred runs, and a run is exactly the rect the viewer fills.
def wall_runs(tiles: list[list[int]]) -> list[list[int]]:
    rows: dict[int, list[int]] = {}
    for c, r in tiles:
        rows.setdefault(int(r), []).append(int(c))
    runs: list[list[int]] = []
    for r in sorted(rows):
        cols = sorted(set(rows[r]))
        start = prev = cols[0]
        for c in cols[1:]:
            if c != prev + 1:
                runs.append([r, start, prev])
                start = c
            prev = c
        runs.append([r, start, prev])
    return runs
# endregion FUNC_wall_runs


def _q(v: float, scale: float) -> int:
    return int(round(float(v) * scale))


def _quant_angle(rad: float) -> int:
    deg = math.degrees(float(rad)) % 360.0
    return int(round(deg * ANGLE_SCALE)) % int(360 * ANGLE_SCALE)


# region FUNC_diff_events
## @purpose Events between two consecutive snapshots: hits, kills with killer, captures, respawns.
## @io (prev snapshot, cur snapshot, t, frames_per_decision) -> list[event dict]
## @complexity 7
## @rationale
## Q: How is the killer found without the world telling us?
## A: A bullet that existed in the previous snapshot and is gone now either hit something or hit a
## A: wall. Its straight path over the decision passes within hit range of the victim if it was the
## A: hit — closest such bullet wins. Needs bullet owners in the snapshot (the batch_world adapter
## A: provides them); without owners, killer stays -1 and the viewer shows "eliminated".
def diff_events(prev: dict, cur: dict, t: int, fpd: int, hit_radius: float) -> list[dict]:
    events: list[dict] = []
    pa, ca = prev["agents"], cur["agents"]

    gone = [b for b in prev["bullets"] if b.get("_matched_next") is None]

    def _culprit(victim: dict) -> tuple[int, int]:
        best, best_d = (-1, -1), float("inf")
        for b in gone:
            if b["team"] == victim["team"]:
                continue
            # closest approach of the bullet's segment over this decision to the victim's new position
            x0, y0 = b["x"], b["y"]
            dx, dy = b["vx"] * fpd, b["vy"] * fpd
            seg2 = dx * dx + dy * dy
            u = 0.0 if seg2 == 0 else max(0.0, min(1.0, ((victim["x"] - x0) * dx + (victim["y"] - y0) * dy) / seg2))
            px, py = x0 + dx * u, y0 + dy * u
            d = math.hypot(victim["x"] - px, victim["y"] - py)
            if d < best_d:
                best_d, best = d, (int(b.get("owner", -1)), int(b["id"]))
        # generous: the victim moved during the decision too
        return best if best_d <= hit_radius + 4.5 * fpd + 2 else (-1, -1)

    for i, (p, c) in enumerate(zip(pa, ca)):
        before = p["hp"] + p["shield"] if p["alive"] else 0.0
        after = c["hp"] + c["shield"] if c["alive"] else 0.0
        died = p["alive"] and not c["alive"]
        if died:
            killer, _ = _culprit(p)
            events.append({"t": t, "k": "kill", "killer": killer, "victim": i,
                           "x": _q(p["x"], 1), "y": _q(p["y"], 1)})
        elif p["alive"] and c["alive"] and after < before - 0.5:
            src, _ = _culprit(c)
            events.append({"t": t, "k": "hit", "src": src, "dst": i, "dmg": int(round(before - after)),
                           "brk": bool(p["shield"] > 0.5 and c["shield"] <= 0.5)})
        elif not p["alive"] and c["alive"]:
            events.append({"t": t, "k": "respawn", "agent": i, "x": _q(c["x"], 1), "y": _q(c["y"], 1)})

    for k, (pc, cc) in enumerate(zip(prev["cps"], cur["cps"])):
        if cc["owner"] != pc["owner"] and cc["owner"] != 0:
            events.append({"t": t, "k": "capture", "cp": k, "team": int(cc["owner"]) - 1, "prev": int(pc["owner"])})
    return events
# endregion FUNC_diff_events


# region FUNC_convert_world_events
## @purpose World11 event records (record=True) of one decision -> replay events.
## @io (world events, prev snapshot, cur snapshot, t) -> list[event dict]
## @rationale The world knows the truth the diff can only guess: who fired the killing bullet, every
## individual hit, dashes. Shots are not copied — the viewer derives them from new bullet ids, and a
## 3-minute match fires thousands. Damage per hit comes from the snapshot difference of the target,
## split evenly between the hits it took this decision.
def convert_world_events(world_events: list[dict], prev: dict, cur: dict, t: int) -> list[dict]:
    out: list[dict] = []
    hits_per_target: dict[int, int] = {}
    for ev in world_events:
        if ev.get("type") == "hit":
            hits_per_target[int(ev["target"])] = hits_per_target.get(int(ev["target"]), 0) + 1
    for ev in world_events:
        kind = ev.get("type") or ev.get("k")
        if kind == "hit":
            d = int(ev["target"])
            p, c = prev["agents"][d], cur["agents"][d]
            lost = (p["hp"] + p["shield"]) - ((c["hp"] + c["shield"]) if c["alive"] else 0.0)
            out.append({"t": t, "k": "hit", "src": int(ev["shooter"]), "dst": d,
                        "dmg": int(round(max(0.0, lost) / hits_per_target[d])),
                        "brk": bool(p["shield"] > 0.5 and c["shield"] <= 0.5)})
        elif kind == "kill":
            v = int(ev["victim"])
            a = cur["agents"][v]
            out.append({"t": t, "k": "kill", "killer": int(ev["killer"]), "victim": v, "x": _q(a["x"], 1), "y": _q(a["y"], 1)})
        elif kind == "respawn":
            g = int(ev["agent"])
            a = cur["agents"][g]
            out.append({"t": t, "k": "respawn", "agent": g, "x": _q(a["x"], 1), "y": _q(a["y"], 1)})
        elif kind == "capture":
            k = int(ev["cp"])
            out.append({"t": t, "k": "capture", "cp": k, "team": int(ev["team"]), "prev": int(prev["cps"][k]["owner"])})
        elif kind == "dash":
            out.append({"t": t, "k": "dash", "agent": int(ev["agent"])})
    return out
# endregion FUNC_convert_world_events


# region CLASS_ReplayWriter
## @purpose Accumulate snapshots of one match and serialise them.
## @complexity 7
class ReplayWriter:
    def __init__(self, geometry: dict, *, seed: int, labels: list[str], source: str,
                 fps: int, frames_per_decision: int, rules: dict, match_index: int = 0) -> None:
        self.geometry = geometry
        self.header: dict[str, Any] = {
            "format": FORMAT_NAME,
            "version": SCHEMA_VERSION,
            "meta": {
                "created": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
                "seed": int(seed),
                "source": source,
                "match_index": int(match_index),
            },
            "timing": {"fps": fps, "frames_per_decision": frames_per_decision,
                       "decision_hz": fps / frames_per_decision},
            "rules": rules,
            "quant": {"pos": POS_SCALE, "angle": ANGLE_SCALE, "vel": VEL_SCALE,
                      "progress": PROGRESS_SCALE, "score": SCORE_SCALE},
            "fields": {"A": AGENT_FIELDS, "B": BULLET_FIELDS, "C": CP_FIELDS},
            "encoding": {"A": "delta", "B": "abs", "C": "abs"},
            "map": {
                "idx": int(geometry["map_idx"]),
                "name": geometry.get("name", f"map{geometry['map_idx']}"),
                "w": geometry["w"], "h": geometry["h"], "tile": geometry["tile"],
                "cols": geometry["cols"], "rows": geometry["rows"],
                "walls": wall_runs(geometry["walls"]),
                "cps": [[_q(x, 1), _q(y, 1)] for x, y in geometry["cps"]],
                "cp_radius": geometry["cp_radius"],
            },
            "teams": [{"id": 0, "label": labels[0]}, {"id": 1, "label": labels[1]}],
        }
        self.frames: list[dict] = []
        self.events: list[dict] = []
        self._prev: dict | None = None
        self._prev_a: list[int] | None = None
        self._next_bullet_id = 0
        self._fpd = frames_per_decision
        self._hit_radius = float(rules["player_radius"] + rules["bullet_radius"])
        self.result: dict | None = None

    # region FUNC__assign_bullet_ids
    ## @purpose Carry ids across frames by exact straight-line prediction; new bullets get new ids.
    def _assign_bullet_ids(self, snap: dict) -> None:
        prev = self._prev["bullets"] if self._prev else []
        free = list(prev)
        for b in snap["bullets"]:
            b["id"] = None
            for j, p in enumerate(free):
                if p["team"] != b["team"]:
                    continue
                if abs(p["x"] + p["vx"] * self._fpd - b["x"]) <= BULLET_MATCH_TOL and \
                   abs(p["y"] + p["vy"] * self._fpd - b["y"]) <= BULLET_MATCH_TOL:
                    b["id"] = p["id"]
                    p["_matched_next"] = b["id"]
                    free.pop(j)
                    break
            if b["id"] is None:
                b["id"] = self._next_bullet_id
                self._next_bullet_id += 1
        for p in free:
            p["_matched_next"] = None
    # endregion FUNC__assign_bullet_ids

    # region FUNC_add
    ## @purpose Append one decision's snapshot; derives events against the previous one.
    ## @io snapshot dict, optional world events -> None
    def add(self, snap: dict, world_events: list[dict] | None = None) -> None:
        t = len(self.frames)
        self._assign_bullet_ids(snap)
        if self._prev is not None:
            if world_events is None:
                self.events.extend(diff_events(self._prev, snap, t, self._fpd, self._hit_radius))
            else:
                self.events.extend(convert_world_events(world_events, self._prev, snap, t))

        a_abs: list[int] = []
        for ag in snap["agents"]:
            st = (ST_ALIVE if ag["alive"] else 0) | (ST_WAITING if (not ag["alive"] and ag.get("respawn_in", 0) > 0) else 0) \
                | (ST_DASHING if ag.get("dashing") else 0)
            a_abs += [_q(ag["x"], POS_SCALE), _q(ag["y"], POS_SCALE), _quant_angle(ag["angle"]),
                      int(round(ag["hp"])), int(round(ag["shield"])), st, int(ag.get("respawn_in", 0))]
        a_enc = a_abs if self._prev_a is None else [v - p for v, p in zip(a_abs, self._prev_a)]
        self._prev_a = a_abs

        b_flat: list[int] = []
        for b in snap["bullets"]:
            b_flat += [int(b["id"]), _q(b["x"], 1), _q(b["y"], 1), _q(b["vx"], VEL_SCALE), _q(b["vy"], VEL_SCALE),
                       int(b["team"]), int(b.get("owner", -1))]
        c_flat: list[int] = []
        for c in snap["cps"]:
            c_flat += [int(c["owner"]), int(c["cap_team"]), _q(c["progress"], PROGRESS_SCALE)]

        self.frames.append({
            "f": int(snap["frame"]),
            "s": [_q(snap["score"][0], SCORE_SCALE), _q(snap["score"][1], SCORE_SCALE)],
            "l": [int(v) for v in snap.get("lives", [0, 0])],
            "A": a_enc, "B": b_flat, "C": c_flat,
        })
        self._prev = snap
    # endregion FUNC_add

    def finish(self, winner: int, reason: str) -> None:
        last = self._prev or {}
        self.result = {"winner": int(winner), "reason": reason,
                       "score": [round(float(s), 2) for s in last.get("score", [0, 0])],
                       "frames": int(last.get("frame", 0)), "decisions": len(self.frames)}
        self.events.append({"t": len(self.frames) - 1, "k": "end", "winner": int(winner), "reason": reason})

    def counts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for ev in self.events:
            out[ev["k"]] = out.get(ev["k"], 0) + 1
        return out

    # region FUNC_save
    ## @purpose Serialise to .arena.json.gz; returns the compressed size in bytes.
    def save(self, path: Path) -> int:
        doc = dict(self.header)
        doc["result"] = self.result
        doc["frames"] = self.frames
        doc["events"] = self.events
        raw = json.dumps(doc, separators=(",", ":")).encode()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(path, "wb", compresslevel=9) as fh:
            fh.write(raw)
        size = path.stat().st_size
        logger.info(f"[IMP:9][ReplayWriter.save][RESULT] {path.name}: frames={len(self.frames)}, events={len(self.events)}, raw={len(raw)} B, gz={size} B [VALUE]")
        return size
    # endregion FUNC_save
# endregion CLASS_ReplayWriter


# region FUNC_load_replay
## @purpose Read a replay and undo the delta encoding: every frame's "A" becomes absolute.
## @io Path -> dict (same document, frames absolute)
def load_replay(path: Path) -> dict:
    with gzip.open(path, "rb") as fh:
        raw = fh.read()
    if raw[:4] == V2_MAGIC:
        return load_replay_v2(raw)
    doc = json.loads(raw)
    if doc.get("format") != FORMAT_NAME or doc.get("version") != SCHEMA_VERSION:
        raise ValueError(f"Not an {FORMAT_NAME} v{SCHEMA_VERSION} file: {doc.get('format')} v{doc.get('version')}")
    if doc["encoding"]["A"] == "delta":
        prev: list[int] | None = None
        for fr in doc["frames"]:
            if prev is not None:
                fr["A"] = [d + p for d, p in zip(fr["A"], prev)]
            prev = fr["A"]
        doc["encoding"]["A"] = "abs"
    return doc
# endregion FUNC_load_replay


# =====================================================================================================
# Schema 2 — big battles
# =====================================================================================================
# Container (all little-endian, gzip over the whole thing):
#   "ARB2" | u32 header_len | header JSON (utf-8, padded to 4) | sections...
#   section: u8 type | 3 pad | u32 payload_len | payload (payload_len is a multiple of 4)
#   Inside a payload every array starts on a 4-byte boundary (pad after each array).
#   FRAMES  (1): u32 f0, u32 F | u32 dec[F] | u32 phys[F] | u32 score[F*2] (x10) | u32 lives[F*2]
#                | i16 x[F*N] | i16 y[F*N]   (row 0 of the chunk absolute, later rows delta vs the row above)
#                | u8 angle[F*N] (256 steps) | u8 hp[F*N] | u8 shield[F*N] | u8 st[F*N]
#                | u8 cp_state[F*C] (owner | cap_team << 2) | u8 cp_prog[F*C] (0..100)
#   BULLETS (2): u32 M | u16 birth[M] | u16 life[M] | u16 owner[M] | u16 x0[M] | u16 y0[M] | i8 vx[M] | i8 vy[M]
#                bullets born in the chunk's decision range, sorted by birth; position at decision t is
#                (x0, y0) + (vx, vy) * frames_per_decision * (t - birth), visible for birth <= t < birth + life
#   EVENTS  (3): u32 E | u16 t[E] | u16 a[E] | u16 b[E] | u16 x[E] | u16 y[E] | u8 kind[E] | u8 v[E]
#                kind: 1 kill (a killer|65535, b victim, x y) · 2 hit (a src|65535, b dst, v dmg | 128 shield break)
#                3 capture (a cp, b team, v prev owner) · 4 respawn (a agent, x y) · 5 dash (a agent) · 6 end (a winner, v reason)
#   Time is the DECISION index everywhere (events, bullets, dec[]); stored frames are every k-th decision
#   plus the last one, and the player interpolates between them.

# region BLOCK_V2_CONSTANTS
V2_VERSION: int = 2
V2_MAGIC: bytes = b"ARB2"
SEC_FRAMES, SEC_BULLETS, SEC_EVENTS = 1, 2, 3
# exp15 optional per-chunk sections, written BEFORE the chunk's events section (a player marks a chunk complete
# on its events): ammo = u8 ammo + u8 cap per frame per agent; attn = u16 attended target per frame per agent.
# Loaders that predate them skip unknown section types (both loops advance by the section length).
SEC_AMMO, SEC_ATTN = 4, 5
ATTN_CP_BASE: int = 0xF000          # attention code: < ATTN_CP_BASE agent index; ATTN_CP_BASE + c = control point c; NONE16 none
EV_CODE: dict[str, int] = {"kill": 1, "hit": 2, "capture": 3, "respawn": 4, "dash": 5, "end": 6, "refill": 7}
EV_NAME: dict[int, str] = {v: k for k, v in EV_CODE.items()}
REASONS: list[str] = ["score", "elimination", "timeout", "draw", "other"]
NONE16: int = 65535
V2_VEL_SCALE: int = 4               # bullet velocity px/frame x 4 fits int8 (24 px/frame -> 96)
BULLET_HASH_PX: float = 8.0         # spatial-hash bucket for id-less bullet matching
# endregion BLOCK_V2_CONSTANTS


def _pad4(buf: bytearray) -> None:
    while len(buf) % 4:
        buf.append(0)


def _put(buf: bytearray, arr: np.ndarray) -> None:
    buf += np.ascontiguousarray(arr).tobytes()
    _pad4(buf)


# region FUNC__snapshot_columns
## @purpose Snapshot (World11/WorldBig format, agents/bullets/cps as lists of dicts OR as dicts of arrays)
## -> plain numpy columns. The columnar form is the fast path for 1000 agents.
def _snapshot_columns(snap: dict) -> dict:
    def col(src: Any, key: str, dtype, default=0.0) -> np.ndarray:
        if isinstance(src, dict):
            v = src.get(key)
            return np.asarray(v if v is not None else np.full(len(next(iter(src.values()))), default), dtype=dtype)
        return np.fromiter((d.get(key, default) for d in src), dtype=dtype, count=len(src))

    ag, bl, cp = snap["agents"], snap["bullets"], snap["cps"]
    alive = col(ag, "alive", bool, False)
    respawn_in = col(ag, "respawn_in", np.float64, 0)
    waiting = col(ag, "waiting", bool, False) if isinstance(ag, dict) and "waiting" in ag else (~alive & (respawn_in > 0))
    n_b = len(bl["x"]) if isinstance(bl, dict) else len(bl)
    has_id = (isinstance(bl, dict) and bl.get("id") is not None) or (not isinstance(bl, dict) and n_b and "id" in bl[0])
    return {
        "x": col(ag, "x", np.float64), "y": col(ag, "y", np.float64), "angle": col(ag, "angle", np.float64),
        "hp": col(ag, "hp", np.float64), "shield": col(ag, "shield", np.float64),
        "alive": alive, "waiting": waiting, "dashing": col(ag, "dashing", bool, False),
        "bx": col(bl, "x", np.float64) if n_b else np.zeros(0), "by": col(bl, "y", np.float64) if n_b else np.zeros(0),
        "bvx": col(bl, "vx", np.float64) if n_b else np.zeros(0), "bvy": col(bl, "vy", np.float64) if n_b else np.zeros(0),
        "bteam": col(bl, "team", np.int64) if n_b else np.zeros(0, np.int64),
        "bowner": col(bl, "owner", np.int64, -1) if n_b else np.zeros(0, np.int64),
        "bid": (col(bl, "id", np.int64) if has_id else None),
        "cp_owner": col(cp, "owner", np.int64), "cp_cap": col(cp, "cap_team", np.int64),
        "cp_prog": col(cp, "progress", np.float64),
        "frame": int(snap["frame"]), "score": [float(s) for s in snap["score"]],
        "lives": [int(v) for v in snap.get("lives", [0, 0])],
    }
# endregion FUNC__snapshot_columns


# region CLASS_BigReplayWriter
## @purpose Accumulate one big match and write it as schema 2. Same call pattern as ReplayWriter.
## @complexity 8
## @rationale
## Q: Why keep everything in memory and write at the end?
## A: A bullet's lifetime and the final result are only known later, and the header carries counts the
## A: player uses to preallocate. 500v500 over ~3600 decisions is ~40 MB of numpy — affordable.
class BigReplayWriter:
    def __init__(self, geometry: dict, *, seed: int, labels: list[str], source: str, fps: int,
                 frames_per_decision: int, rules: dict, match_index: int = 0, decimate: int = 2,
                 chunk_frames: int = 64) -> None:
        self.geometry = geometry
        self._fpd = int(frames_per_decision)
        self._k = max(1, int(decimate))
        self._cf = max(1, int(chunk_frames))
        span = max(float(geometry["w"]), float(geometry["h"]))
        self._pos_q = 2 if span * 2 < 32000 else 1           # i16 absolute rows must fit
        self.header: dict[str, Any] = {
            "format": FORMAT_NAME, "version": V2_VERSION,
            "meta": {"created": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
                     "seed": int(seed), "source": source, "match_index": int(match_index)},
            "timing": {"fps": fps, "frames_per_decision": self._fpd, "decision_hz": fps / self._fpd},
            "rules": rules,
            "quant": {"pos": self._pos_q, "angle_steps": 256, "vel": V2_VEL_SCALE, "progress": 100, "score": SCORE_SCALE},
            "map": {
                "idx": int(geometry.get("map_idx", 0)), "name": geometry.get("name", "map"),
                "w": geometry["w"], "h": geometry["h"], "tile": geometry["tile"],
                "cols": geometry["cols"], "rows": geometry["rows"],
                "walls": wall_runs(geometry["walls"]) if geometry["walls"] else [],
                "cps": [[_q(x, 1), _q(y, 1)] for x, y in geometry["cps"]],
                "cp_radius": geometry["cp_radius"],
                "sectors": geometry.get("sectors", []),
            },
            "teams": [{"id": 0, "label": labels[0]}, {"id": 1, "label": labels[1]}],
            "decimate": self._k, "chunk_frames": self._cf,
        }
        self._frames: list[tuple] = []       # stored frames (encoded columns)
        self._pending: tuple | None = None    # last decision if it is not on the decimation grid
        self._n = 0                           # decisions added
        self._prev: dict | None = None
        # bullets
        self._b: dict[str, list] = {k: [] for k in ("birth", "death", "owner", "x0", "y0", "vx", "vy")}
        self._live_ids: dict[int, int] = {}   # world bullet id -> record index (id mode)
        self._live_rec: np.ndarray = np.zeros(0, np.int64)   # record index per bullet of the previous snapshot (match mode)
        # events (columns)
        self._ev: dict[str, list] = {k: [] for k in ("t", "k", "a", "b", "x", "y", "v")}
        self.result: dict | None = None
        self._t_add = 0.0
        # exp15 extras (ammo / attention): decided on the first add that carries them; per-agent stats for the
        # class panel; refills aggregated per CP between stored frames (one event per CP per stored frame at most)
        self._ext = {"ammo": False, "attn": False}
        self._st: dict[str, np.ndarray] | None = None
        self._prev_ammo: np.ndarray | None = None
        self._gaining: np.ndarray | None = None
        self._refill_acc: dict[int, list] = {}
        cps = geometry.get("cps") or []
        self._cp_xy = np.array([[float(x), float(y)] for x, y in cps], np.float64).reshape(-1, 2)
        self._cp_r = float(geometry.get("cp_radius", 0.0))

    # region FUNC__event
    def _event(self, t: int, kind: str, a: int = NONE16, b: int = NONE16, x: float = 0, y: float = 0, v: int = 0) -> None:
        e = self._ev
        e["t"].append(t); e["k"].append(EV_CODE[kind])
        e["a"].append(NONE16 if a is None or a < 0 else int(a)); e["b"].append(NONE16 if b is None or b < 0 else int(b))
        e["x"].append(max(0, min(65535, int(round(x))))); e["y"].append(max(0, min(65535, int(round(y)))))
        e["v"].append(int(v) & 255)
    # endregion FUNC__event

    # region FUNC__track_bullets
    ## @purpose Birth/death bookkeeping. With world ids: a dict. Without: predicted-position spatial hash.
    def _track_bullets(self, d: int, c: dict) -> None:
        n = len(c["bx"])
        rec = np.full(n, -1, np.int64)
        if c["bid"] is not None:
            seen: dict[int, int] = {}
            for j in range(n):
                bid = int(c["bid"][j])
                r = self._live_ids.get(bid)
                if r is not None:
                    rec[j] = r
                seen[bid] = j
            for bid, r in list(self._live_ids.items()):
                if bid not in seen:
                    self._b["death"][r] = d
                    del self._live_ids[bid]
        elif self._prev is not None and len(self._prev["bx"]):
            p = self._prev
            px = p["bx"] + p["bvx"] * self._fpd
            py = p["by"] + p["bvy"] * self._fpd
            buckets: dict[tuple, list[int]] = {}
            for i in range(len(px)):
                buckets.setdefault((int(px[i] // BULLET_HASH_PX), int(py[i] // BULLET_HASH_PX), int(p["bteam"][i])), []).append(i)
            used = np.zeros(len(px), bool)
            for j in range(n):
                kx, ky, tm = int(c["bx"][j] // BULLET_HASH_PX), int(c["by"][j] // BULLET_HASH_PX), int(c["bteam"][j])
                done = False
                for dx in (-1, 0, 1):
                    for dy in (-1, 0, 1):
                        for i in buckets.get((kx + dx, ky + dy, tm), ()):
                            if not used[i] and abs(px[i] - c["bx"][j]) <= BULLET_MATCH_TOL and abs(py[i] - c["by"][j]) <= BULLET_MATCH_TOL:
                                used[i] = True
                                rec[j] = self._live_rec[i]
                                done = True
                                break
                        if done:
                            break
                    if done:
                        break
            for i in np.flatnonzero(~used):
                self._b["death"][int(self._live_rec[i])] = d
        q = self._pos_q
        for j in np.flatnonzero(rec < 0):
            r = len(self._b["birth"])
            B = self._b
            B["birth"].append(d); B["death"].append(-1)
            B["owner"].append(int(c["bowner"][j]) if c["bowner"][j] >= 0 else NONE16)
            B["x0"].append(int(round(c["bx"][j] * q))); B["y0"].append(int(round(c["by"][j] * q)))
            B["vx"].append(int(np.clip(round(c["bvx"][j] * V2_VEL_SCALE), -127, 127)))
            B["vy"].append(int(np.clip(round(c["bvy"][j] * V2_VEL_SCALE), -127, 127)))
            rec[j] = r
            if c["bid"] is not None:
                self._live_ids[int(c["bid"][j])] = r
        self._live_rec = rec
    # endregion FUNC__track_bullets

    # region FUNC__events_from_world
    def _events_from_world(self, world_events: list[dict], p: dict, c: dict, t: int) -> None:
        hits_per: dict[int, int] = {}
        for ev in world_events:
            if (ev.get("type") or ev.get("k")) == "hit":
                hits_per[int(ev["target"])] = hits_per.get(int(ev["target"]), 0) + 1
        for ev in world_events:
            kind = ev.get("type") or ev.get("k")
            if kind == "hit":
                dst = int(ev["target"])
                before = p["hp"][dst] + p["shield"][dst]
                after = (c["hp"][dst] + c["shield"][dst]) if c["alive"][dst] else 0.0
                dmg = int(round(max(0.0, before - after) / hits_per[dst]))
                brk = 128 if (p["shield"][dst] > 0.5 and c["shield"][dst] <= 0.5) else 0
                self._event(t, "hit", int(ev["shooter"]), dst, v=min(127, dmg) | brk)
            elif kind == "kill":
                v = int(ev["victim"])
                self._event(t, "kill", int(ev["killer"]), v, p["x"][v], p["y"][v])
            elif kind == "respawn":
                g = int(ev["agent"])
                self._event(t, "respawn", g, None, c["x"][g], c["y"][g])
            elif kind == "capture":
                k = int(ev["cp"])
                self._event(t, "capture", k, int(ev["team"]), v=int(p["cp_owner"][k]))
            elif kind == "dash":
                self._event(t, "dash", int(ev["agent"]))
    # endregion FUNC__events_from_world

    # region FUNC__events_by_diff
    ## @purpose No world events: kills/respawns/captures/hits from state transitions (no culprit).
    def _events_by_diff(self, p: dict, c: dict, t: int) -> None:
        for v in np.flatnonzero(p["alive"] & ~c["alive"]):
            self._event(t, "kill", -1, int(v), p["x"][v], p["y"][v])
        for g in np.flatnonzero(~p["alive"] & c["alive"]):
            self._event(t, "respawn", int(g), None, c["x"][g], c["y"][g])
        lost = (p["hp"] + p["shield"]) - (c["hp"] + c["shield"])
        for dst in np.flatnonzero(p["alive"] & c["alive"] & (lost > 0.5)):
            brk = 128 if (p["shield"][dst] > 0.5 and c["shield"][dst] <= 0.5) else 0
            self._event(t, "hit", -1, int(dst), v=min(127, int(round(lost[dst]))) | brk)
        for k in np.flatnonzero((c["cp_owner"] != p["cp_owner"]) & (c["cp_owner"] != 0)):
            self._event(t, "capture", int(k), int(c["cp_owner"][k]) - 1, v=int(p["cp_owner"][k]))
    # endregion FUNC__events_by_diff

    def _encode_frame(self, d: int, c: dict, x: dict) -> tuple:
        q = self._pos_q
        st = (c["alive"].astype(np.uint8) * ST_ALIVE) | (c["waiting"].astype(np.uint8) * ST_WAITING) \
            | (c["dashing"].astype(np.uint8) * ST_DASHING)
        ang = (np.round((np.degrees(c["angle"]) % 360.0) / 360.0 * 256.0).astype(np.int64) % 256).astype(np.uint8)
        n = len(c["x"])
        am = np.clip(np.round(x["ammo"]), 0, 255).astype(np.uint8) if x.get("ammo") is not None else None
        cap = np.clip(np.round(x["ammo_cap"]), 0, 255).astype(np.uint8) if x.get("ammo_cap") is not None else None
        att = np.asarray(x["attn"]).astype(np.int64) if x.get("attn") is not None else None
        att = np.where((att < 0) | (att > NONE16), NONE16, att).astype(np.uint16) if att is not None else np.full(n, NONE16, np.uint16)
        return (d, c["frame"], [int(round(s * SCORE_SCALE)) for s in c["score"]], c["lives"],
                np.round(c["x"] * q).astype(np.int32), np.round(c["y"] * q).astype(np.int32), ang,
                np.clip(np.round(c["hp"]), 0, 255).astype(np.uint8), np.clip(np.round(c["shield"]), 0, 255).astype(np.uint8), st,
                ((c["cp_owner"] & 3) | ((c["cp_cap"] & 3) << 2)).astype(np.uint8),
                np.clip(np.round(c["cp_prog"] * 100), 0, 100).astype(np.uint8),
                am if am is not None else np.zeros(n, np.uint8),
                cap if cap is not None else (am if am is not None else np.zeros(n, np.uint8)),
                att)

    # region FUNC__extras
    ## @purpose exp15 per-decision extras, from the `extras` argument or the snapshot's agent columns (feature-detected):
    ## ammo, ammo_cap (N floats), attn (N ints: agent index, ATTN_CP_BASE + cp, <0 or NONE16 = none),
    ## aimed / fired (N 0/1: this decision's shot was fired with the aim label on / fired at all).
    @staticmethod
    def _extras(snap: dict, extras: dict | None) -> dict:
        ag = snap.get("agents")
        out: dict[str, Any] = {}
        for key in ("ammo", "ammo_cap", "attn", "aimed", "fired"):
            v = extras.get(key) if extras else None
            if v is None and isinstance(ag, dict):
                v = ag.get(key)
            if v is not None:
                v = v.detach().cpu().numpy() if hasattr(v, "detach") else np.asarray(v)
            out[key] = v
        return out
    # endregion FUNC__extras

    # region FUNC__ammo_stats
    ## @purpose Per-agent stats (alive / empty decisions, refill trips, aimed and fired shots) and refills per CP.
    ## A refill = ammo grew while the agent was alive in both decisions (a respawn refill is not one); the CP is the
    ## one whose radius holds the agent. Gains accumulate per CP until the next stored frame, then one event:
    ## a = cp, b = agents refilling, v = ammo gained (≤ 255), x/y = CP centre.
    def _ammo_stats(self, d: int, c: dict, x: dict) -> None:
        n = len(c["x"])
        if self._st is None:
            self._st = {k: np.zeros(n, np.int64) for k in ("alive_dec", "empty_dec", "refills", "aimed", "fired")}
            self._gaining = np.zeros(n, bool)
        st = self._st
        st["alive_dec"] += c["alive"]
        if x.get("aimed") is not None:
            st["aimed"] += np.asarray(x["aimed"]).astype(np.int64)
        if x.get("fired") is not None:
            st["fired"] += np.asarray(x["fired"]).astype(np.int64)
        am = x.get("ammo")
        if am is None:
            return
        am = np.asarray(am, np.float64)
        st["empty_dec"] += c["alive"] & (am < 0.5)
        gaining = np.zeros(n, bool)
        if self._prev_ammo is not None and self._prev is not None:
            gain = am - self._prev_ammo
            gaining = c["alive"] & self._prev["alive"] & (gain > 0.25)
            idx = np.flatnonzero(gaining)
            if len(idx) and len(self._cp_xy):
                dx = c["x"][idx, None] - self._cp_xy[None, :, 0]
                dy = c["y"][idx, None] - self._cp_xy[None, :, 1]
                dist = np.hypot(dx, dy)
                cp = np.argmin(dist, axis=1)
                near = dist[np.arange(len(idx)), cp] <= self._cp_r * 1.05 + 1.0
                for j in np.flatnonzero(near):
                    acc = self._refill_acc.setdefault(int(cp[j]), [0, 0.0])
                    acc[0] += 1
                    acc[1] += float(gain[idx[j]])
            st["refills"] += gaining & ~self._gaining
        self._gaining = gaining
        self._prev_ammo = am
        if d % self._k == 0:
            self._flush_refills(d)

    def _flush_refills(self, d: int) -> None:
        for cp, (cnt, gained) in sorted(self._refill_acc.items()):
            x, y = self._cp_xy[cp]
            self._event(d, "refill", cp, cnt, x, y, v=min(255, int(round(gained))))
        self._refill_acc.clear()
    # endregion FUNC__ammo_stats

    # region FUNC_add
    ## @purpose Append one decision. world_events: World11-style records of this decision, or None.
    ## extras (exp15, optional): see _extras — ammo/attention streams are written only if some decision carries them.
    def add(self, snap: dict, world_events: list[dict] | None = None, extras: dict | None = None) -> None:
        t0 = time.perf_counter()
        d = self._n
        c = _snapshot_columns(snap)
        x = self._extras(snap, extras)
        self._ext["ammo"] |= x["ammo"] is not None
        self._ext["attn"] |= x["attn"] is not None
        self._track_bullets(d, c)
        if self._prev is not None:
            if world_events is None:
                self._events_by_diff(self._prev, c, d)
            else:
                self._events_from_world(world_events, self._prev, c, d)
        self._ammo_stats(d, c, x)
        enc = self._encode_frame(d, c, x)
        if d % self._k == 0:
            self._frames.append(enc)
            self._pending = None
        else:
            self._pending = enc
        self._prev = c
        self._n += 1
        self._t_add += time.perf_counter() - t0
    # endregion FUNC_add

    def finish(self, winner: int, reason: str) -> None:
        if self._pending is not None:
            self._frames.append(self._pending)
            self._pending = None
        if self._refill_acc:
            self._flush_refills(max(0, self._n - 1))
        last = self._prev or {"score": [0, 0], "frame": 0}
        self.result = {"winner": int(winner), "reason": reason,
                       "score": [round(float(s), 2) for s in last["score"]],
                       "frames": int(last["frame"]), "decisions": self._n}
        self._event(self._n - 1, "end", int(winner), None, v=REASONS.index(reason) if reason in REASONS else 4)

    def counts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for k in self._ev["k"]:
            out[EV_NAME[k]] = out.get(EV_NAME[k], 0) + 1
        out["shot"] = len(self._b["birth"])
        return out

    # region FUNC_save
    ## @purpose Write the schema-2 container; returns the compressed size in bytes.
    ## @complexity 7
    def save(self, path: Path, compresslevel: int = 6) -> int:
        t0 = time.perf_counter()
        F = len(self._frames)
        N = len(self._frames[0][4]) if F else 0
        C = len(self._frames[0][10]) if F else 0
        dec = np.array([f[0] for f in self._frames], np.uint32)
        B = {k: np.array(v, np.int64) for k, v in self._b.items()}
        last_dec = self._n - 1
        if len(B["birth"]):
            B["death"][B["death"] < 0] = last_dec + 1
        life = (B["death"] - B["birth"]) if len(B["birth"]) else np.zeros(0, np.int64)
        E = {k: np.array(v, np.int64) for k, v in self._ev.items()}
        order = np.argsort(E["t"], kind="stable") if len(E["t"]) else np.zeros(0, np.int64)
        E = {k: v[order] for k, v in E.items()}

        body = bytearray()
        n_chunks = (F + self._cf - 1) // self._cf
        for ci in range(n_chunks):
            f0, f1 = ci * self._cf, min(F, (ci + 1) * self._cf)
            d_lo = 0 if ci == 0 else int(dec[f0])
            d_hi = int(dec[f1]) if f1 < F else last_dec + 1
            fr = self._frames[f0:f1]
            n = len(fr)
            x = np.stack([f[4] for f in fr]); y = np.stack([f[5] for f in fr])
            xd = np.vstack([x[:1], np.diff(x, axis=0)]).astype(np.int16)
            yd = np.vstack([y[:1], np.diff(y, axis=0)]).astype(np.int16)
            p = bytearray()
            _put(p, np.array([f0, n], np.uint32))
            _put(p, dec[f0:f1])
            _put(p, np.array([f[1] for f in fr], np.uint32))
            _put(p, np.array([f[2] for f in fr], np.uint32).reshape(-1))
            _put(p, np.array([f[3] for f in fr], np.uint32).reshape(-1))
            _put(p, xd.reshape(-1)); _put(p, yd.reshape(-1))
            for idx in (6, 7, 8, 9):
                _put(p, np.stack([f[idx] for f in fr]).reshape(-1))
            _put(p, np.stack([f[10] for f in fr]).reshape(-1)); _put(p, np.stack([f[11] for f in fr]).reshape(-1))
            body += struct.pack("<B3xI", SEC_FRAMES, len(p)) + p

            sel = np.flatnonzero((B["birth"] >= d_lo) & (B["birth"] < d_hi)) if len(B["birth"]) else np.zeros(0, np.int64)
            p = bytearray()
            _put(p, np.array([len(sel)], np.uint32))
            _put(p, B["birth"][sel].astype(np.uint16)); _put(p, np.minimum(life[sel], NONE16).astype(np.uint16))
            _put(p, B["owner"][sel].astype(np.uint16)); _put(p, B["x0"][sel].astype(np.uint16)); _put(p, B["y0"][sel].astype(np.uint16))
            _put(p, B["vx"][sel].astype(np.int8)); _put(p, B["vy"][sel].astype(np.int8))
            body += struct.pack("<B3xI", SEC_BULLETS, len(p)) + p

            if self._ext["ammo"]:
                p = bytearray()
                _put(p, np.array([f0, n], np.uint32))
                _put(p, np.stack([f[12] for f in fr]).reshape(-1)); _put(p, np.stack([f[13] for f in fr]).reshape(-1))
                body += struct.pack("<B3xI", SEC_AMMO, len(p)) + p
            if self._ext["attn"]:
                p = bytearray()
                _put(p, np.array([f0, n], np.uint32))
                _put(p, np.stack([f[14] for f in fr]).reshape(-1))
                body += struct.pack("<B3xI", SEC_ATTN, len(p)) + p

            sel = np.flatnonzero((E["t"] >= d_lo) & (E["t"] < d_hi)) if len(E["t"]) else np.zeros(0, np.int64)
            p = bytearray()
            _put(p, np.array([len(sel)], np.uint32))
            for k in ("t", "a", "b", "x", "y"):
                _put(p, E[k][sel].astype(np.uint16))
            _put(p, E["k"][sel].astype(np.uint8)); _put(p, E["v"][sel].astype(np.uint8))
            body += struct.pack("<B3xI", SEC_EVENTS, len(p)) + p

        header = dict(self.header)
        if self._ext["ammo"] or self._ext["attn"]:
            header["ext"] = {"ammo": bool(self._ext["ammo"]), "attn": bool(self._ext["attn"]), "attn_cp_base": ATTN_CP_BASE}
        if self._st is not None and (self._ext["ammo"] or self._st["aimed"].any() or self._st["fired"].any()):
            st = self._st
            fired = st["fired"] if st["fired"].any() else np.bincount(
                np.where(B["owner"] < N, B["owner"], N), minlength=N + 1)[:N] if len(B["owner"]) else np.zeros(N, np.int64)
            header["agent_stats"] = {"decision_s": self._fpd / float(self.header["timing"]["fps"]),
                                     "alive_dec": st["alive_dec"].tolist(), "empty_dec": st["empty_dec"].tolist(),
                                     "refills": st["refills"].tolist(), "aimed": st["aimed"].tolist(),
                                     "fired": [int(v) for v in fired]}
        header.update({"result": self.result, "n_agents": N, "team_size": self.header["rules"].get("team_size", N // 2),
                       "n_cps": C, "n_frames": F, "n_decisions": self._n, "n_chunks": n_chunks,
                       "n_bullets": int(len(B["birth"])), "max_bullet_life": int(life.max()) if len(life) else 0,
                       "n_events": int(len(E["t"])), "counts": self.counts(), "sections_bytes": len(body)})
        hj = bytearray(json.dumps(header, separators=(",", ":")).encode())
        _pad4(hj)
        blob = V2_MAGIC + struct.pack("<I", len(hj)) + bytes(hj) + bytes(body)
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(path, "wb", compresslevel=compresslevel) as fh:
            fh.write(blob)
        size = path.stat().st_size
        logger.info(f"[IMP:9][BigReplayWriter.save][RESULT] {path.name}: agents={N}, cps={C}, decisions={self._n}, "
                    f"frames={F} (decimate {self._k}), chunks={n_chunks}, bullets={len(B['birth'])}, events={len(E['t'])}, "
                    f"raw={len(blob)} B, gz={size} B, add={self._t_add:.1f}s, save={time.perf_counter() - t0:.1f}s [VALUE]")
        return size
    # endregion FUNC_save
# endregion CLASS_BigReplayWriter


# region FUNC_load_replay_v2
## @purpose Decode a schema-2 container (already gunzipped) into numpy arrays — the Python mirror of
## viewer/big/replay2.js, used by tests and tools.
## @io bytes -> dict: header fields + dec, phys, score (F,2), lives (F,2), x/y (F,N) px, angle rad, hp, sh, st,
##     cp_owner/cp_cap/cp_prog (F,C), bullets {birth, life, owner, x0, y0, vx, vy}, events {t, kind, a, b, x, y, v}
def load_replay_v2(raw: bytes) -> dict:
    if raw[:4] != V2_MAGIC:
        raise ValueError("not a schema-2 container")
    (hlen,) = struct.unpack_from("<I", raw, 4)
    doc = json.loads(raw[8:8 + hlen].rstrip(b"\x00").decode())
    N, C, F = doc["n_agents"], doc["n_cps"], doc["n_frames"]
    pq = doc["quant"]["pos"]
    out: dict[str, Any] = {k: [] for k in ("dec", "phys", "score", "lives", "x", "y", "angle", "hp", "sh", "st", "cps", "cpp",
                                           "ammo", "ammo_cap", "attn")}
    bullets: dict[str, list] = {k: [] for k in ("birth", "life", "owner", "x0", "y0", "vx", "vy")}
    events: dict[str, list] = {k: [] for k in ("t", "a", "b", "x", "y", "kind", "v")}
    off = 8 + hlen

    def take(buf: bytes, o: int, dtype, count: int) -> tuple[np.ndarray, int]:
        a = np.frombuffer(buf, dtype=dtype, count=count, offset=o)
        o += a.nbytes
        return a, o + (-o) % 4

    while off < len(raw):
        typ, plen = struct.unpack_from("<B3xI", raw, off)
        o = off + 8
        if typ == SEC_FRAMES:
            (f0, n), o = take(raw, o, np.uint32, 2)
            for key, dt, cnt in (("dec", np.uint32, n), ("phys", np.uint32, n), ("score", np.uint32, 2 * n), ("lives", np.uint32, 2 * n)):
                a, o = take(raw, o, dt, cnt)
                out[key].append(a.reshape(n, -1) if cnt == 2 * n else a)
            xd, o = take(raw, o, np.int16, n * N)
            yd, o = take(raw, o, np.int16, n * N)
            out["x"].append(np.cumsum(xd.reshape(n, N).astype(np.int64), axis=0) / pq)
            out["y"].append(np.cumsum(yd.reshape(n, N).astype(np.int64), axis=0) / pq)
            for key in ("angle", "hp", "sh", "st"):
                a, o = take(raw, o, np.uint8, n * N)
                out[key].append(a.reshape(n, N))
            a, o = take(raw, o, np.uint8, n * C); out["cps"].append(a.reshape(n, C))
            a, o = take(raw, o, np.uint8, n * C); out["cpp"].append(a.reshape(n, C))
        elif typ == SEC_BULLETS:
            (m,), o = take(raw, o, np.uint32, 1)
            m = int(m)
            for key, dt in (("birth", np.uint16), ("life", np.uint16), ("owner", np.uint16), ("x0", np.uint16), ("y0", np.uint16), ("vx", np.int8), ("vy", np.int8)):
                a, o = take(raw, o, dt, m)
                bullets[key].append(a)
        elif typ == SEC_EVENTS:
            (m,), o = take(raw, o, np.uint32, 1)
            m = int(m)
            for key, dt in (("t", np.uint16), ("a", np.uint16), ("b", np.uint16), ("x", np.uint16), ("y", np.uint16), ("kind", np.uint8), ("v", np.uint8)):
                a, o = take(raw, o, dt, m)
                events[key].append(a)
        elif typ == SEC_AMMO:
            (f0, n), o = take(raw, o, np.uint32, 2)
            a, o = take(raw, o, np.uint8, n * N); out["ammo"].append(a.reshape(n, N))
            a, o = take(raw, o, np.uint8, n * N); out["ammo_cap"].append(a.reshape(n, N))
        elif typ == SEC_ATTN:
            (f0, n), o = take(raw, o, np.uint32, 2)
            a, o = take(raw, o, np.uint16, n * N); out["attn"].append(a.reshape(n, N))
        off += 8 + plen                     # unknown section types are skipped

    cat = lambda lst: np.concatenate(lst) if lst else np.zeros(0)  # noqa: E731
    res = dict(doc)
    res.update({k: cat(v) for k, v in out.items() if k not in ("angle", "ammo", "ammo_cap", "attn")})
    for k in ("ammo", "ammo_cap", "attn"):          # exp15 streams: (F, N) or absent
        if out[k]:
            res[k] = np.concatenate(out[k])
    res["angle"] = cat(out["angle"]).astype(np.float64) / 256.0 * 2 * math.pi
    res["cp_owner"] = res["cps"] & 3
    res["cp_cap"] = res["cps"] >> 2
    res["cp_prog"] = res.pop("cpp") / 100.0
    res["bullets"] = {k: cat(v) for k, v in bullets.items()}
    res["events"] = {k: cat(v) for k, v in events.items()}
    if len(res["dec"]) != F:
        raise ValueError(f"frame count {len(res['dec'])} != header {F}")
    return res
# endregion FUNC_load_replay_v2
