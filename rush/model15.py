from __future__ import annotations

# region MODULE_CONTRACT [DOMAIN(9): RL; CONCEPT(10): FloatingAttention15; TECH(9): torch]
## @modulecontract
## @purpose Experiment-15 actor-critic: exp12/exp14's network (model12.ActorCritic12) plus a *floating attention*
## readout — a query built from the actor's memory (LSTM output h_t), the agent's own core token and raw
## «under fire» features attends over every observation token of the step, and its result is added to h_t before the
## action heads through a zero-initialised projection. v2 (contract «v2», docs/PERCEPTION_REVIEW.md §2): an HONEST
## graft — the old encoder path receives exactly exp14's inputs; segments exp14 did not have (the 20+12 eyes pool,
## the 21×21 grid, …) enter ONLY through the readout, so at step 0 the network equals exp14 at every scale.
## @scope Network, weight surgery, attention statistics, policy files. No rollouts, no optimizer, no world.
## @input exp12/exp14 payload {"state_dict", "config", "step"} (ActorCritic12); OBS_LAYOUT15 whose segments carry
##   "n_old" (features per token of the exp14 layout), optionally "new_features" (appended column names) and optionally
##   "path": "readout" (else: a segment whose name the exp14 layout lacks is readout-only)
## @output ActorCritic15; Policy15 (act with masks; also usable through encoder()+step_actor() like Policy12)
## @links USES_API(9): torch; LINKS_TO: model11 (LayoutEncoder, _AttnBlock), model12 (graft helpers, masks),
##   train15, vec15, docs/EXP15_CONTRACT.md, docs/EXP15_NOTES_model.md, docs/PERCEPTION_REVIEW.md
## @invariants
## - Graft (v2): every exp14 weight is copied; new input columns of OLD-path segments are zero; old-path token
##   segments keep exp14's token COUNT (strict when the layout has readout segments); readout-only segments feed only
##   the readout, whose output projection is zero -> exp14's logits and value exactly on every observation, at any
##   scale (tests/test_model15.py, 5/20/50 per side)
## - v1 layouts (no readout segments, entity count 9 -> 32) still build (comparison), with a WARN: not function-preserving
## - step_actor(feats, ...) after encoder(obs) reproduces act_rollout's logits: the encoder keeps the tokens and the
##   raw query features of its last call. Not thread-safe across callers sharing one model — parts step sequentially
## @rationale
## Q: Why did v1 drift (0.51 -> 0.15-0.33 vs its frozen start)?
## A: PERCEPTION_REVIEW §2: the old encoder mean-pools over all tokens; with 32 entity slots instead of 9 the pooled
## A: feature changed (KL 0.74 nat at 20v20) although the graft test (5v5, pool == old 9) passed. v2 keeps the old
## A: path's inputs identical to exp14's and gives the new information only to the zero-started readout.
## Q: Why is the attention query taken AFTER the LSTM (h_t) and not a memory token of h_{t-1} inside the encoder?
## A: A memory token makes the encoder depend on the previous step's LSTM output, so the update could no longer
## A: encode the T*B observations of a BPTT chunk in one batch. Querying with h_t keeps the encoder batched and still
## A: makes the attention depend on memory AND the current situation.
## Q: Why raw «under fire» columns in the query and not only through the core token?
## A: The core token reaches them through zero-initialised columns, i.e. only after the old path learns to read them;
## A: the query's own linear layer is freshly initialised, so the readout conditions on «hot fight» from step 1.
## Q: Why a zero-initialised projection instead of a learned scalar gate?
## A: Same function-preserving start with fewer moving parts: the projection's weight gets a gradient at once.
## @changes
## LAST_CHANGE: [v0.2.0] Honest graft: readout-only segments (eyes pool, 21×21 grid) with their own embeddings and a
##   context block; old path fed exactly exp14's inputs; raw under-fire features in the query; token order for the
##   viewer = old tokens then readout tokens.
## PREV_CHANGE: [v0.1.0] Initial: floating readout, graft from exp14 (ActorCritic12 payloads), attention statistics.
## @modulemap
## FUNC 6[Readout-only segment names of a layout] => readout_segments
## CLASS 9[Old encoder + readout-only branch, returns tokens and types] => LayoutEncoder15
## CLASS 9[State-conditioned attention readout] => FloatingReadout
## CLASS 10[Actor-critic with floating attention] => ActorCritic15
## FUNC 10[exp14 payload -> exp15 model] => from_exp14
## CLASS 8[Inference wrapper] => Policy15
## FUNC 7[Encoder token order (viewer mapping)] => token_order
## FUNC 8[Load an exp15 file, or graft an exp12/exp14 one] => load_policy
## @usecases
## - train15: model = from_exp14(torch.load(ckpt), OBS_LAYOUT15, STATE_SIZE15)
## - league / vec: pol = load_policy(path, device, layout=OBS_LAYOUT15, state_size=STATE_SIZE15)
def _module_contract():
    pass
# endregion MODULE_CONTRACT
# GREP_SUMMARY: model15, floating attention, attention readout, zero-init projection, honest graft, readout-only segments, eyes pool, grid 21x21, under fire query, attention mass
# STRUCTURE: ▶ obs → ⚡ LayoutEncoder15 (old path = exp14 encoder on old segments → feats; readout branch = pool/grid21 tokens + context block) → ⚡ LSTM → h_t → ⚡ FloatingReadout(q = [h_t, core, under_fire], K/V = old ∪ readout tokens) → h_t + P(attn) → ⚡ heads → ⎋ actions; ⊕ feats + global_state → critic

import logging
from typing import Any

import torch
import torch.nn as nn

from rush.model11 import DEFAULT_CONFIG as DEFAULT_CONFIG11
from rush.model11 import LayoutEncoder, _AttnBlock
from rush.model12 import (NVEC12, ActorCritic12, _copy_linear_cols, layout_column_map, policy_payload,  # noqa: F401
                                   seg_n_old)

logger = logging.getLogger(__name__)

# region BLOCK_CONSTANTS
# Token types for attention statistics.
T_CORE, T_ENEMY, T_ALLY, T_BULLET, T_CP, T_LASTSEEN, T_GRID, T_OTHER = range(8)
TYPE_NAMES: list[str] = ["core", "enemy", "ally", "bullet", "cp", "lastseen", "grid", "other"]
N_TYPES: int = len(TYPE_NAMES)
SEG_TYPE: dict[str, int] = {"bullets": T_BULLET, "bullets_pool": T_BULLET, "cps": T_CP, "lastseen": T_LASTSEEN}
ENTITY_SEGS: tuple[str, ...] = ("entities", "entities_old", "pool", "eyes", "eyes_pool", "entities_pool", "entities_more")   # world11-format entity tokens
ENTITY_TYPE_COL: int = 1              # world11 entity token: ETYPE_ALLY = -1, ETYPE_ENEMY = +1
CORE_HP_COL: int = 2                  # world11 core: hp / PLAYER_HP * 2 - 1
CORE_ALIVE_COL: int = 4               # world11 core: alive as ±1
AMMO_FEATURE: str = "ammo_frac"       # world15 core feature: ammo / cap * 2 - 1 (±1, like every world11 core feature)
# Core features that go RAW into the readout query (contract v2 item 5: «under fire now»). Matched by prefix.
QUERY_PREFIXES: tuple[str, ...] = ("under_fire", "dmg_recent", "damage_recent", "since_dmg", "incoming", "threat",
                                   "hit_recent", "spread_now", "ammo_frac")
AMMO_EDGES: tuple[float, float] = (0.25, 0.6)    # buckets: empty-ish [0, .25), low [.25, .6), full [.6, 1]
HP_EDGE: float = 0.5                  # hp fraction below this = "hurt"
GATE_INIT: float = -6.0               # attention-score bias of readout segments grafted onto a trained net (x_gate)
AMMO_BUCKETS: list[str] = ["ammo_lo", "ammo_mid", "ammo_hi"]
HP_BUCKETS: list[str] = ["hp_lo", "hp_hi"]
# endregion BLOCK_CONSTANTS


# region FUNC_readout_segments
## @purpose Names of readout-only segments. world15 v2 marks every segment with "path": "old" (exp14's encoder
## input, may be renamed — e.g. entities_old) or "new" (readout only). Layouts without the key fall back to «any
## segment whose name the exp14 layout lacks». The core (vector) segment is never readout-only.
def readout_segments(layout: list[dict], old_names: list[str] | None = None) -> list[str]:
    out = []
    has_path = any("path" in s for s in layout)
    for s in layout:
        if s["kind"] == "vector":
            continue
        if has_path:
            if s.get("path") in ("new", "readout"):
                out.append(s["name"])
        elif old_names is not None and s["name"] not in old_names:
            out.append(s["name"])
    return out


def query_columns(layout: list[dict]) -> list[int]:
    """Absolute obs columns of the core's appended features whose names start with a QUERY_PREFIXES entry."""
    core = next(s for s in layout if s["kind"] == "vector")
    names = list(core.get("new_features") or [])
    base = int(core["start"]) + seg_n_old(core)
    return [base + i for i, n in enumerate(names) if any(str(n).startswith(p) for p in QUERY_PREFIXES)]
# endregion FUNC_readout_segments


# region CLASS_LayoutEncoder15
## @purpose Old path = model11.LayoutEncoder over the segments exp14 had (same parameters and state_dict keys, so the
## graft copies exp14 verbatim); readout branch = per-segment embeddings (tokens) / CNN (grids) of the readout-only
## segments, contextualised by one attention block together with the old path's core token. forward returns the old
## path's features (LSTM and critic input — unchanged by the new segments) and, for the readout, the union of tokens.
class LayoutEncoder15(LayoutEncoder):
    def __init__(self, layout: list[dict], d_model: int, heads: int, layers: int, features: int,
                 readout_segs: list[str] | None = None, query_cols: list[int] | None = None) -> None:
        ro = set(readout_segs or [])
        super().__init__([s for s in layout if s["name"] not in ro], d_model, heads, layers, features)
        self.full_layout = [dict(s) for s in layout]
        self.readout_segs = [s["name"] for s in layout if s["name"] in ro]
        self.x_tok_segs = [s for s in layout if s["name"] in ro and s["kind"] == "tokens"]
        self.x_grid_segs = [s for s in layout if s["name"] in ro and s["kind"] == "grid"]
        self.query_cols = list(query_cols or [])
        n_x = len(self.x_tok_segs) + len(self.x_grid_segs)
        if n_x:
            self.x_embed = nn.ModuleList([nn.Linear(int(s["size"]), d_model) for s in self.x_tok_segs])
            self.x_type = nn.Parameter(torch.randn(n_x, d_model) * 0.02)
            self.x_cnn = nn.ModuleList()
            for s in self.x_grid_segs:
                c = int(s["size"][0])
                self.x_cnn.append(nn.Sequential(
                    nn.Conv2d(c, 32, 3, stride=2, padding=1), nn.GELU(),
                    nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.GELU(),
                    nn.AdaptiveAvgPool2d((2, 2)), nn.Flatten(), nn.Linear(256, d_model),
                ))
            self.x_block = _AttnBlock(d_model, heads)
            self.x_norm = nn.LayerNorm(d_model)
            # v4: per readout segment, an additive attention-score bias on its tokens (as keys) in x_block and the
            # readout. 0 = plain attention; a segment grafted onto a trained net starts at GATE_INIT (nearly unseen)
            # and opens as it learns, so the warm start keeps the old behaviour.
            self.x_gate = nn.Parameter(torch.zeros(n_x))
        self._last: tuple | None = None

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):  # type: ignore[override]
        k = prefix + "x_gate"
        if hasattr(self, "x_gate") and k not in state_dict:                           # pre-v4 checkpoints: open gates
            state_dict[k] = torch.zeros_like(self.x_gate)
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    @staticmethod
    def _entity_types(raw: torch.Tensor) -> torch.Tensor:
        return torch.where(raw[..., ENTITY_TYPE_COL] > 0, T_ENEMY, T_ALLY).long()

    def forward(self, obs: torch.Tensor, return_tokens: bool = False):  # type: ignore[override]
        b = obs.shape[0]
        dev = obs.device
        # ---- old path: exactly model11.LayoutEncoder.forward over the exp14 segments ----
        vec = torch.cat([obs[:, s["start"]: s["start"] + int(s["size"]) * int(s.get("count", 1))] for s in self.vec_segs], dim=1)
        core = self.core(vec) + self.type_embed[0]
        tokens = [core.unsqueeze(1)]
        used = [torch.ones(b, 1, dtype=torch.bool, device=dev)]
        types = [torch.full((b, 1), T_CORE, dtype=torch.long, device=dev)]
        for k, s in enumerate(self.tok_segs):
            cnt, size = int(s["count"]), int(s["size"])
            raw = obs[:, s["start"]: s["start"] + cnt * size].reshape(b, cnt, size)
            tokens.append(self.tok_embed[k](raw) + self.type_embed[1 + k])
            used.append(raw.abs().sum(-1) > 0)
            types.append(self._entity_types(raw) if s["name"] in ENTITY_SEGS
                         else torch.full((b, cnt), SEG_TYPE.get(s["name"], T_OTHER), dtype=torch.long, device=dev))
        for k, s in enumerate(self.grid_segs):
            c, h, w = (int(v) for v in s["size"])
            raw = obs[:, s["start"]: s["start"] + c * h * w].reshape(b, c, h, w)
            g = self.grid_cnn[k](raw) + self.type_embed[1 + len(self.tok_segs) + k]
            tokens.append(g.unsqueeze(1))
            used.append(torch.ones(b, 1, dtype=torch.bool, device=dev))
            types.append(torch.full((b, 1), T_GRID, dtype=torch.long, device=dev))
        x = torch.cat(tokens, dim=1)
        mask = torch.cat(used, dim=1)
        pad = ~mask
        for blk in self.blocks:
            x = blk(x, pad)
        x = self.norm(x)
        wts = mask.unsqueeze(-1).to(x.dtype)
        pooled = (x * wts).sum(1) / wts.sum(1).clamp(min=1.0)
        feats = self.head(torch.cat([x[:, 0], pooled], dim=1))
        types_t = torch.cat(types, dim=1)
        # ---- readout-only branch (never touches feats) ----
        if self.x_tok_segs or self.x_grid_segs:
            xt = [x[:, :1]]                                        # context: the agent's own (old-path) core token
            xu = [torch.ones(b, 1, dtype=torch.bool, device=dev)]
            xg = [torch.zeros(b, 1, device=dev)]
            xy = []
            gate = getattr(self, "x_gate", None)
            for k, s in enumerate(self.x_tok_segs):
                cnt, size = int(s["count"]), int(s["size"])
                raw = obs[:, s["start"]: s["start"] + cnt * size].reshape(b, cnt, size)
                xt.append(self.x_embed[k](raw) + self.x_type[k])
                xu.append(raw.abs().sum(-1) > 0)
                xg.append((gate[k] if gate is not None else torch.zeros((), device=dev)).float().expand(b, cnt))
                xy.append(self._entity_types(raw) if s["name"] in ENTITY_SEGS
                          else torch.full((b, cnt), SEG_TYPE.get(s["name"], T_OTHER), dtype=torch.long, device=dev))
            for k, s in enumerate(self.x_grid_segs):
                c, h, w = (int(v) for v in s["size"])
                raw = obs[:, s["start"]: s["start"] + c * h * w].reshape(b, c, h, w)
                xt.append((self.x_cnn[k](raw) + self.x_type[len(self.x_tok_segs) + k]).unsqueeze(1))
                xu.append(torch.ones(b, 1, dtype=torch.bool, device=dev))
                kg = len(self.x_tok_segs) + k
                xg.append((gate[kg] if gate is not None else torch.zeros((), device=dev)).float().expand(b, 1))
                xy.append(torch.full((b, 1), T_GRID, dtype=torch.long, device=dev))
            xx = torch.cat(xt, dim=1)
            xm = torch.cat(xu, dim=1)
            xb = torch.cat(xg, dim=1)
            kpm = xb.masked_fill(~xm, float("-inf"))                 # float key mask: pads -inf, gated tokens biased
            xx = self.x_norm(self.x_block(xx, kpm.to(xx.dtype)))[:, 1:]   # drop the context copy of the core token
            x = torch.cat([x, xx.to(x.dtype)], dim=1)
            bias = torch.cat([torch.zeros_like(pad, dtype=torch.float32), xb[:, 1:]], dim=1)
            pad = torch.cat([pad, ~xm[:, 1:]], dim=1)
            types_t = torch.cat([types_t] + xy, dim=1)
        else:
            bias = None
        rq = obs[:, self.query_cols] if self.query_cols else obs.new_zeros(b, 0)
        tok = (x, pad, types_t, rq, bias)
        self._last = tok
        return (feats, tok) if return_tokens else feats
# endregion CLASS_LayoutEncoder15


# region CLASS_FloatingReadout
## @purpose h' = h + P(MHA(q = Q([h, core, raw query features]), K = V = tokens, padding masked)); P zero-initialised.
## @io h (B, H), tokens (B, T, D), pad (B, T), rq (B, R) -> h' (B, H), attention weights averaged over heads (B, T)
class FloatingReadout(nn.Module):
    def __init__(self, d_model: int, hidden: int, heads: int, n_raw: int = 0) -> None:
        super().__init__()
        self.n_raw = int(n_raw)
        self.q = nn.Linear(hidden + d_model + self.n_raw, d_model)
        self.attn = nn.MultiheadAttention(d_model, heads, batch_first=True)
        self.out = nn.Linear(d_model, hidden)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, h: torch.Tensor, x: torch.Tensor, pad: torch.Tensor, rq: torch.Tensor | None = None,
                bias: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        parts = [h, x[:, 0].to(h.dtype)]
        if self.n_raw:
            parts.append(rq.to(h.dtype))
        q = self.q(torch.cat(parts, dim=-1)).unsqueeze(1).to(x.dtype)
        kpm = pad if bias is None else bias.masked_fill(pad, float("-inf")).to(x.dtype)   # v4 gates (see x_gate)
        a, w = self.attn(q, x, x, key_padding_mask=kpm, need_weights=True, average_attn_weights=True)
        return h + self.out(a.squeeze(1)).to(h.dtype), w.squeeze(1)
# endregion CLASS_FloatingReadout


# region CLASS_ActorCritic15
## @purpose ActorCritic12 + floating readout between the LSTM and the action heads (the aux aim head reads h' too,
## so the aim signal also trains the attention). Critic unchanged (old-path encoder features + global state).
## @complexity 9
class ActorCritic15(ActorCritic12):
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        cfg = self.config
        ro = list(cfg.get("readout_segs") or [])
        qc = list(cfg.get("query_cols") or [])
        # Replace the encoder by the exp15 one (old path: identical parameters and state_dict keys to exp14's).
        self.encoder = LayoutEncoder15(cfg["layout"], cfg["d_model"], cfg["heads"], cfg["layers"], cfg["features"], ro, qc)
        fa = dict(cfg.get("floating_attn") or {})
        self.readout = FloatingReadout(cfg["d_model"], cfg["lstm_hidden"], int(fa.get("heads", cfg["heads"])), len(qc))
        self._init_weights()
        nn.init.zeros_(self.readout.out.weight)
        nn.init.zeros_(self.readout.out.bias)
        core = next(s for s in cfg["layout"] if s["kind"] == "vector")
        names = list(core.get("new_features") or [])
        self.ammo_col = (int(core["start"]) + seg_n_old(core) + names.index(AMMO_FEATURE)) if AMMO_FEATURE in names else None
        self.core_start = int(core["start"])
        self.track_attn = False
        self.export_top = False
        self.last_top: tuple[torch.Tensor, torch.Tensor] | None = None
        self.reset_attn_stats()

    # region FUNC_attention_stats
    ## @purpose Attention mass per token type, split by own-ammo bucket x HP bucket, over alive agents. Accumulated
    ## on the device during collection (no host sync); read and zeroed by take_attn_stats().
    def reset_attn_stats(self) -> None:
        dev = next(self.parameters()).device
        self._attn_mass = torch.zeros(len(AMMO_BUCKETS), len(HP_BUCKETS), N_TYPES, device=dev)
        self._attn_n = torch.zeros(len(AMMO_BUCKETS), len(HP_BUCKETS), device=dev)

    @torch.no_grad()
    def _track(self, obs: torch.Tensor, w: torch.Tensor, types: torch.Tensor) -> None:
        mass = torch.zeros(w.shape[0], N_TYPES, device=w.device, dtype=torch.float32)
        mass.scatter_add_(1, types, w.float())
        alive = obs[:, self.core_start + CORE_ALIVE_COL] > 0
        hp = (obs[:, self.core_start + CORE_HP_COL] + 1.0) * 0.5
        hb = (hp >= HP_EDGE).long()
        if self.ammo_col is not None:
            am = (obs[:, self.ammo_col] + 1.0) * 0.5                       # ±1 -> fraction of the cap
            ab = (am >= AMMO_EDGES[0]).long() + (am >= AMMO_EDGES[1]).long()
        else:
            ab = torch.full_like(hb, len(AMMO_BUCKETS) - 1)
        cell = (ab * len(HP_BUCKETS) + hb)[alive]
        flat_m = self._attn_mass.view(-1, N_TYPES)
        flat_m.index_add_(0, cell, mass[alive])
        self._attn_n.view(-1).index_add_(0, cell, torch.ones_like(cell, dtype=self._attn_n.dtype))

    def take_attn_stats(self) -> dict[str, Any]:
        """{"attn_<type>": share over all alive decisions, "attn_<ammo>_<hp>_<type>": share in that cell, n per cell}."""
        m, n = self._attn_mass.cpu(), self._attn_n.cpu()
        self.reset_attn_stats()
        out: dict[str, Any] = {}
        tot_n = float(n.sum())
        if tot_n > 0:
            all_m = m.sum((0, 1)) / tot_n
            for t, name in enumerate(TYPE_NAMES):
                out[f"attn_{name}"] = round(float(all_m[t]), 5)
        for i, an in enumerate(AMMO_BUCKETS):
            for j, hn in enumerate(HP_BUCKETS):
                c = float(n[i, j])
                out[f"attn_n_{an}_{hn}"] = int(c)
                if c > 0:
                    for t in (T_ENEMY, T_ALLY, T_CP, T_BULLET, T_LASTSEEN, T_GRID):
                        out[f"attn_{an}_{hn}_{TYPE_NAMES[t]}"] = round(float(m[i, j, t]) / c, 5)
        return out
    # endregion FUNC_attention_stats

    # region FUNC_readout
    ## @purpose h' from h and the step's tokens; records the attention statistics / top token when enabled.
    def _readout(self, h: torch.Tensor, tok: tuple, obs: torch.Tensor | None) -> torch.Tensor:
        x, pad, types, rq = tok[:4]
        h2, w = self.readout(h, x, pad, rq, tok[4] if len(tok) > 4 else None)
        if obs is not None and (self.track_attn or self.export_top):
            with torch.no_grad():
                if self.track_attn:
                    self._track(obs, w, types)
                if self.export_top:
                    ww = w.masked_fill((types == T_CORE) | (types == T_GRID), -1.0)
                    top = ww.argmax(-1)
                    self.last_top = (types.gather(1, top.unsqueeze(1)).squeeze(1), top)
        return h2
    # endregion FUNC_readout

    # region FUNC_step_actor
    ## @purpose One recurrent step with the readout. The tokens come from `tok` or, when called as
    ## encoder(obs) -> step_actor(feats, ...) (Policy12.act, vec14.opponent_act), from the encoder's last call.
    def step_actor(self, feats: torch.Tensor, state: tuple[torch.Tensor, torch.Tensor], starts: torch.Tensor,  # type: ignore[override]
                   tok: tuple | None = None, obs: torch.Tensor | None = None):
        tok = tok if tok is not None else self.encoder._last
        if tok is None or tok[0].shape[0] != feats.shape[0]:
            raise RuntimeError("ActorCritic15.step_actor needs the tokens of the same batch: call encoder(obs) first")
        keep = (1.0 - starts.float()).view(1, -1, 1)
        h, c = state[0] * keep, state[1] * keep
        out, (h, c) = self.lstm(feats.unsqueeze(0), (h, c))
        h2 = self._readout(out.squeeze(0), tok, obs)
        return self.actor_head(h2), (h, c)
    # endregion FUNC_step_actor

    @torch.no_grad()
    def act_rollout(self, obs, gstate, state, starts, deterministic: bool = False, mask: torch.Tensor | None = None):  # type: ignore[override]
        feats, tok = self.encoder(obs, return_tokens=True)
        logits, state = self.step_actor(feats, state, starts, tok, obs)
        actions, logp, _ = self.sample(logits, deterministic, mask)
        v = self.value_from(feats, gstate)
        return actions, logp, v, state

    # region FUNC_evaluate_sequence
    ## @io as model12: obs (T, B, O), gstate, actions, starts, state0, mask -> logp (T, B), ent (T, B, heads), v [, aux]
    def evaluate_sequence(self, obs, gstate, actions, starts, state0, return_aux: bool = False, mask: torch.Tensor | None = None):  # type: ignore[override]
        t, b = obs.shape[0], obs.shape[1]
        feats, tok = self.encoder(obs.reshape(t * b, -1), return_tokens=True)
        v = self.value_from(feats, gstate.reshape(t * b, -1)).view(t, b)
        h, c = state0
        keep0 = (1.0 - starts[0].float()).view(1, -1, 1)
        seq = self._lstm_segments(feats.view(t, b, -1), starts, h * keep0, c * keep0)
        h2 = self._readout(seq.reshape(t * b, -1), tok, None)
        logits = self.actor_head(h2)
        mk = None if mask is None else mask.reshape(t * b, -1)
        logp, ent = self.evaluate_logits(logits, actions.reshape(t * b, -1), mk)
        if return_aux:
            if self.aux_aim_head is None:
                raise RuntimeError("evaluate_sequence(return_aux=True) on a model built without aux_aim")
            aux = self.aux_aim_head(h2).view(t, b)
            return logp.view(t, b), ent.view(t, b, -1), v, aux
        return logp.view(t, b), ent.view(t, b, -1), v
    # endregion FUNC_evaluate_sequence
# endregion CLASS_ActorCritic15


def build_model15(config: dict[str, Any], device: torch.device | str = "cpu") -> ActorCritic15:
    for key in ("layout", "state_size"):
        if key not in config:
            raise ValueError(f"model config lacks {key!r}")
    cfg = {**DEFAULT_CONFIG11, **config}
    cfg.setdefault("floating_attn", {"heads": cfg["heads"]})
    cfg.setdefault("readout_segs", [])
    cfg.setdefault("query_cols", [])
    model = ActorCritic15(cfg).to(device)
    model.reset_attn_stats()
    n_x = sum(p.numel() for n, p in model.encoder.named_parameters() if n.startswith("x_"))
    logger.info(f"[IMP:8][build_model15][BUILD] ActorCritic15 params={sum(p.numel() for p in model.parameters())} "
                f"(readout {sum(p.numel() for p in model.readout.parameters())}, readout-only branch {n_x}), nvec={model.nvec}, "
                f"state={cfg['state_size']}, readout_segs={cfg['readout_segs']}, query_cols={len(cfg['query_cols'])}, "
                f"ammo_col={model.ammo_col} [VALUE]")
    return model


def is_exp15(config: dict[str, Any]) -> bool:
    return bool(config.get("floating_attn"))


# region FUNC_upgrade_core15
## @purpose Honest graft of an exp15 payload onto a layout whose CORE gained appended features (world15 layout v3:
## the order/front columns, 28.09). Every other segment must match by name, kind, size and count (their starts may
## shift); the core's first columns keep their meaning. The only parameter that changes shape is the core's input
## projection; its new columns start at ZERO, so the upgraded net computes exactly what the old one did until it
## learns to read the new inputs. Returns (model, upgraded?) — the model as is when the layouts already agree.
## v4 (28.09): also token segments whose COUNT grew (the per-token weights do not depend on it) and readout-only
## segments appended to the layout (their embeddings / CNN start fresh, x_type rows are re-mapped by name). Those
## two change what the readout attends to, so v3 -> v4 is a warm start, not an exact graft.
def upgrade_core15(payload: dict[str, Any], layout: list[dict], device: torch.device | str = "cpu") -> tuple[Any, bool]:
    import copy
    cfg = payload["config"]
    old = {s["name"]: s for s in cfg["layout"]}
    new = {s["name"]: s for s in layout}
    oc = next(s for s in cfg["layout"] if s["kind"] == "vector")
    nc = next(s for s in layout if s["kind"] == "vector")
    sig = lambda L: [(s["name"], s["kind"], s["size"], s.get("count")) for s in L]         # noqa: E731
    if sig(cfg["layout"]) == sig(layout):
        m = build_model15(cfg, device)
        m.load_state_dict(payload["state_dict"])
        return m, False
    of, nf = list(oc.get("new_features") or []), list(nc.get("new_features") or [])
    if int(nc["size"]) < int(oc["size"]) or nf[:len(of)] != of or seg_n_old(oc) != seg_n_old(nc):
        raise ValueError(f"core {oc['size']}->{nc['size']}: new features must extend the old ones ({of} -> {nf})")
    changes = []
    for name, s in old.items():
        if s["kind"] == "vector":
            continue
        t = new.get(name)
        if t is None or (t["kind"], t["size"]) != (s["kind"], s["size"]):
            raise ValueError(f"segment {name!r} differs between the checkpoint and the layout: {s} vs {t}")
        if t.get("count") != s.get("count"):
            if s["kind"] != "tokens" or int(t["count"]) < int(s["count"]):
                raise ValueError(f"segment {name!r}: count {s.get('count')} -> {t.get('count')} is not a token pool growth")
            changes.append(f"{name} x{s['count']}->{t['count']}")
    added = [s for s in layout if s["name"] not in old]
    for s in added:
        if s.get("path") not in ("new", "readout"):
            raise ValueError(f"added segment {s['name']!r} must be readout-only (path new)")
    changes += [f"+{s['name']}" for s in added]
    old_ro = list(cfg.get("readout_segs") or [])
    new_ro = old_ro + [s["name"] for s in added]
    cfg2 = copy.deepcopy(cfg)
    cfg2["layout"] = copy.deepcopy(layout)
    cfg2["readout_segs"] = new_ro
    cfg2["query_cols"] = query_columns(layout)
    cfg2["upgraded_core"] = f"{oc['size']}->{nc['size']}" + (f" {' '.join(changes)}" if changes else "")
    m = build_model15(cfg2, device)
    sd = dict(payload["state_dict"])
    tgt = m.state_dict()

    def x_order(L: list[dict], ro: list[str]) -> list[str]:                          # x_type rows: tokens, then grids
        r = set(ro)
        return [s["name"] for s in L if s["name"] in r and s["kind"] == "tokens"] + \
               [s["name"] for s in L if s["name"] in r and s["kind"] == "grid"]
    for k, v in list(sd.items()):
        if k == "encoder.x_type" and k in tgt and tgt[k].shape != v.shape:
            ob, nb = x_order(cfg["layout"], old_ro), x_order(layout, new_ro)
            w = tgt[k].clone()
            for i, name in enumerate(ob):
                w[nb.index(name)] = v[i]
            sd[k] = w
        elif k in tgt and tgt[k].shape != v.shape:
            if v.dim() != 2 or tgt[k].shape[0] != v.shape[0] or tgt[k].shape[1] < v.shape[1]:
                raise ValueError(f"unexpected shape change {k}: {tuple(v.shape)} -> {tuple(tgt[k].shape)}")
            w = torch.zeros_like(tgt[k])
            w[:, : v.shape[1]] = v
            sd[k] = w
    fresh = [k for k in tgt if k not in sd]
    for k in fresh:
        sd[k] = tgt[k]
    m.load_state_dict(sd)
    if added and hasattr(m.encoder, "x_gate"):
        nb = x_order(layout, new_ro)
        with torch.no_grad():
            for s in added:
                m.encoder.x_gate[nb.index(s["name"])] = GATE_INIT
    logger.info(f"[IMP:9][upgrade_core15][EXEC] core {oc['size']} -> {nc['size']} (new zero columns: {nf[len(of):]}), "
                f"segments: {changes}, fresh params: {len(fresh)}, gates of added segments = {GATE_INIT} [VALUE]")
    return m, True
# endregion FUNC_upgrade_core15


# region FUNC_from_exp14
## @purpose Build an exp15 network from an ActorCritic12 payload (exp12 fork / exp14 class checkpoint). v2 layouts
## (with readout-only segments): exact on EVERY observation — the old path sees exp14's inputs (extra columns get
## zero weights), the rest only reaches the zero-started readout. v1 layouts (entity count changed, no readout
## segments): exact only when the enlarged pool holds exactly the exp14 tokens (WARN).
## @io payload (ActorCritic12), layout15 (segments with n_old = exp14 widths), state_size15, state_map (new column of
##   each old global-state column; None = prefix) -> ActorCritic15 on `device`
## @complexity 8
def from_exp14(payload: dict[str, Any], layout15: list[dict], state_size15: int, state_map: list[int] | None = None,
               device: torch.device | str = "cpu") -> ActorCritic15:
    cfg0 = dict(payload["config"])
    sd0 = payload["state_dict"]
    if is_exp15(cfg0):
        raise ValueError("payload is already exp15: use load_policy / build_model15")
    if list(cfg0.get("nvec", [])) != NVEC12:
        raise ValueError(f"from_exp14 needs an exp12/exp14 payload (nvec {NVEC12}), got {cfg0.get('nvec')}")
    old_layout = cfg0["layout"]
    old_names = [s["name"] for s in old_layout]
    ro = readout_segments(layout15, old_names)
    old_path = [s for s in layout15 if s["name"] not in ro]
    # Old-path segments map to exp14's by ORDER (world15 v2 renames exp14's entity block to entities_old); kinds must agree.
    if [s["kind"] for s in old_path] != [s["kind"] for s in old_layout]:
        raise ValueError(f"old-path segments {[(s['name'], s['kind']) for s in old_path]} do not line up with exp14's "
                         f"{[(s['name'], s['kind']) for s in old_layout]}")
    renamed = [(so["name"], sn["name"]) for so, sn in zip(old_layout, old_path) if so["name"] != sn["name"]]
    honest = bool(ro)
    for so, sn in zip(old_layout, old_path):
        n_old = seg_n_old(sn)
        want = so["size"][0] if so["kind"] == "grid" else int(so["size"])
        if n_old != want:
            raise ValueError(f"segment {sn['name']}: n_old={n_old} but the source had width {so['size']}")
        if sn["kind"] != "tokens" and int(so.get("count", 1)) != int(sn.get("count", 1)):
            raise ValueError(f"segment {sn['name']}: count {so.get('count')} -> {sn.get('count')} (only token segments may change count)")
        if sn["kind"] == "tokens" and int(so["count"]) != int(sn["count"]):
            if honest:
                raise ValueError(f"old-path token segment {sn['name']}: count {so['count']} -> {sn['count']} breaks the honest "
                                 f"graft (mean pooling changes); give the extra tokens their own readout-only segment")
            logger.warning(f"[IMP:9][from_exp14][WARN] v1 layout: {sn['name']} count {so['count']} -> {sn['count']} on the old "
                           f"path — function-preserving only when the extra slots are empty (PERCEPTION_REVIEW §2) [VALUE]")
        if so["kind"] == "grid" and list(so["size"][1:]) != list(sn["size"][1:]):
            raise ValueError(f"grid {sn['name']} spatial size changed {so['size']} -> {sn['size']}")

    qc = query_columns(layout15)
    cfg15 = {**cfg0, "layout": [dict(s) for s in layout15], "state_size": int(state_size15),
             "floating_attn": {"heads": int(cfg0.get("heads", DEFAULT_CONFIG11["heads"]))},
             "readout_segs": ro, "query_cols": qc,
             "grafted_from15": {"step": int(payload.get("step", 0)), "fork": payload.get("fork"), "state_size": int(cfg0["state_size"]),
                                "counts": {s["name"]: int(s.get("count", 1)) for s in old_layout}, "honest": honest}}
    model = build_model15(cfg15, "cpu")
    new_sd = model.state_dict()
    vec_map, tok_n_old, grid_old = layout_column_map(old_path)
    tok_names = [s["name"] for s in old_path if s["kind"] == "tokens"]
    grid_names = [s["name"] for s in old_path if s["kind"] == "grid"]
    special = {"encoder.core.0.weight", "state_mlp.0.weight"}
    special |= {f"encoder.tok_embed.{k}.weight" for k in range(len(tok_names))}
    special |= {f"encoder.grid_cnn.{k}.0.weight" for k in range(len(grid_names))}
    copied = 0
    for k, v in sd0.items():
        if k in special:
            continue
        if k not in new_sd:
            raise KeyError(f"source key {k} has no exp15 counterpart")
        if new_sd[k].shape != v.shape:
            raise ValueError(f"unexpected shape change at {k}: {tuple(v.shape)} -> {tuple(new_sd[k].shape)}")
        new_sd[k] = v.clone()
        copied += 1
    model.load_state_dict(new_sd)
    enc = model.encoder
    reshaped = []
    _copy_linear_cols(enc.core[0], sd0["encoder.core.0.weight"], None, vec_map)
    reshaped.append(("encoder.core.0", sd0["encoder.core.0.weight"].shape[1], enc.core[0].in_features))
    for k, name in enumerate(tok_names):
        w = sd0[f"encoder.tok_embed.{k}.weight"]
        _copy_linear_cols(enc.tok_embed[k], w, None, list(range(tok_n_old[name])))
        reshaped.append((f"tok_embed.{k}({name})", w.shape[1], enc.tok_embed[k].in_features))
    with torch.no_grad():
        for k, name in enumerate(grid_names):
            w = sd0[f"encoder.grid_cnn.{k}.0.weight"]
            conv = enc.grid_cnn[k][0]
            conv.weight.zero_()
            conv.weight[:, : grid_old[name]] = w
    s_old = int(cfg0["state_size"])
    _copy_linear_cols(model.state_mlp[0], sd0["state_mlp.0.weight"], None, list(state_map) if state_map is not None else list(range(s_old)))
    reshaped.append(("state_mlp.0", s_old, model.state_mlp[0].in_features))
    readout_zero = bool(model.readout.out.weight.abs().sum() == 0 and model.readout.out.bias.abs().sum() == 0)
    model = model.to(device)
    model.reset_attn_stats()
    logger.info(f"[IMP:9][from_exp14][RESULT] grafted fork={payload.get('fork')} step={payload.get('step', 0)}: copied={copied} tensors, "
                f"widened {reshaped}, honest={honest} readout_segs={ro} renamed={renamed} query_cols={len(qc)}, old-path token counts "
                f"{ {s['name']: int(s.get('count', 1)) for s in old_path if s['kind'] == 'tokens'} }, "
                f"readout out-projection zero={readout_zero} [VALUE]")
    return model
# endregion FUNC_from_exp14


# region CLASS_Policy15
## @purpose Frozen inference view (league / vec / recorder). Same interface as model12.Policy12.
## After every act(): `last_attn_top` (rows,) long — index of the top-attended token in the order of
## token_order(layout, readout_segs) (core and grids excluded), `last_attn_type` (rows,) its type (T_* constants).
## The replay side (tools/tourney14._exp15_extras, record_big15) maps the index to a fighter / CP via
## world.attention_targets.
class Policy15:
    def __init__(self, model: ActorCritic15, step: int = 0, fork: str | None = None) -> None:
        self.model = model.eval()
        self.model.export_top = True
        self.step = step
        self.fork = fork
        self.device = next(model.parameters()).device
        self.config = model.config
        self.token_order = token_order(model.config["layout"], model.config.get("readout_segs") or [])
        self.last_attn_top: torch.Tensor | None = None
        self.last_attn_type: torch.Tensor | None = None

    def initial_state(self, batch: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.model.initial_state(batch, self.device)

    @torch.no_grad()
    def act(self, obs: torch.Tensor, state: tuple[torch.Tensor, torch.Tensor], episode_starts: torch.Tensor,
            deterministic: bool = False, mask: torch.Tensor | None = None):
        obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        starts = torch.as_tensor(episode_starts, dtype=torch.float32, device=self.device)
        feats, tok = self.model.encoder(obs, return_tokens=True)
        logits, state = self.model.step_actor(feats, state, starts, tok, obs)
        actions, _, _ = self.model.sample(logits, deterministic, mask)
        if self.model.last_top is not None:
            self.last_attn_type, self.last_attn_top = self.model.last_top
        return actions, state
# endregion CLASS_Policy15


# region FUNC_token_order
## @purpose The encoder's token order: index -> (segment name, slot). 0 = core; then the OLD path's token segments'
## slots in layout order, then the old grids (one token each); then the readout-only token segments' slots, then the
## readout-only grids. world.attention_targets maps (agent, token index) -> fighter / CP code.
def token_order(layout: list[dict], readout_segs: list[str] | None = None) -> list[tuple[str, int]]:
    ro = set(readout_segs or [])
    out: list[tuple[str, int]] = [("core", 0)]
    for s in layout:
        if s["kind"] == "tokens" and s["name"] not in ro:
            out.extend((s["name"], j) for j in range(int(s["count"])))
    out.extend((s["name"], 0) for s in layout if s["kind"] == "grid" and s["name"] not in ro)
    for s in layout:
        if s["kind"] == "tokens" and s["name"] in ro:
            out.extend((s["name"], j) for j in range(int(s["count"])))
    out.extend((s["name"], 0) for s in layout if s["kind"] == "grid" and s["name"] in ro)
    return out
# endregion FUNC_token_order


# region FUNC_load_policy
## @purpose An exp15 file loads as is; an exp12/exp14 (ActorCritic12) file is grafted onto (layout, state_size).
def load_policy(path: str, device: torch.device | str = "cpu", layout: list[dict] | None = None,
                state_size: int | None = None, state_map: list[int] | None = None) -> Policy15:
    payload = torch.load(str(path), map_location="cpu", weights_only=False)
    return policy_from_payload(payload, device, layout, state_size, state_map, str(path))


def policy_from_payload(payload: dict[str, Any], device: torch.device | str = "cpu", layout: list[dict] | None = None,
                        state_size: int | None = None, state_map: list[int] | None = None, name: str = "?") -> Policy15:
    cfg = payload["config"]
    if is_exp15(cfg):
        if layout is not None:
            model, up = upgrade_core15(payload, layout, device)
            kind = "exp15 upgraded" if up else "exp15"
        else:
            model = build_model15(cfg, device)
            model.load_state_dict(payload["state_dict"])
            kind = "exp15"
    else:
        if layout is None or state_size is None:
            raise ValueError(f"{name} is not an exp15 file: pass layout and state_size to graft it")
        model = from_exp14(payload, layout, state_size, state_map, device)
        kind = "grafted"
    logger.info(f"[IMP:8][load_policy][EXEC] {name} ({kind}) step={payload.get('step', 0)} fork={payload.get('fork')} [VALUE]")
    return Policy15(model, int(payload.get("step", 0)), payload.get("fork"))
# endregion FUNC_load_policy
