from __future__ import annotations

# region MODULE_CONTRACT [DOMAIN(9): RL; CONCEPT(10): GraftedActorCritic12; TECH(9): torch]
## @modulecontract
## @purpose Experiment-12 actor-critic: model11's network with action masks, plus the graft that turns
## an exp11 checkpoint into an exp12 network (wider observation, 7-way turn head) that acts exactly
## like the exp11 network at step 0. Also the exp12 inference entrypoint load_policy().
## @scope Weight surgery, masked sampling/evaluation, policy files. No rollouts, no optimizer.
## @input exp11 payload {"state_dict", "config", "step"}; OBS_LAYOUT12 whose segments carry "n_old"
## @output ActorCritic12; Policy12 (act with an optional legal-action mask)
## @links USES_API(9): torch; LINKS_TO: model11 (network definition), world12 (OBS_LAYOUT12), ppo12, vec12, league12
## @invariants
## - Graft: every exp11 weight is copied; columns of new observation features are zero; a segment's
##   first n_old features (per token) are the exp11 features in the exp11 order
## - Turn head 5 -> 7: indices 0-4 keep their exp11 rows, 5 and 6 get weight 0 and bias NEW_TURN_BIAS
## - A mask marks LEGAL actions (True); illegal logits are -inf in sampling AND in evaluation, so the
##   stored log-prob and the update's log-prob come from the same distribution
## - A head with a single legal action has entropy exactly 0 there (masked entropy is not NaN)
## @rationale
## Q: Why graft instead of training exp12 from scratch?
## A: exp11b took 500M decisions to reach Elo 3567 and was still climbing. Zero-initialised new input
## A: columns make the new features invisible at step 0 (the network is exactly exp11b on the old part),
## A: and gradient opens them as they turn out useful — Net2Net / "function-preserving" growth.
## Q: Why bias -8 for the two new turn logits and not 0?
## A: With bias 0 the fine turns would get ~2/7 of the probability mass at once and change behaviour
## A: before the network knows what they do. At -8 they start at p ~ e^-8 relative to a typical logit,
## A: i.e. invisible, and the entropy bonus plus policy gradient raise them when they pay.
## Q: Why refuse new segments instead of zero-initialising them too?
## A: A token that is not all-zero joins attention: even with a zero embedding it changes the softmax
## A: over tokens (its key is the type embedding). Only features appended to existing segments are
## A: function-preserving; a new segment would silently change exp11b's behaviour.
## @changes
## LAST_CHANGE: [v0.1.0] Initial exp12 graft + masked actor-critic.
## @modulemap
## FUNC 9[Column map of an old flat vector inside a new layout] => layout_column_map
## FUNC 10[exp11 payload -> exp12 model] => from_exp11
## CLASS 9[Masked actor-critic] => ActorCritic12
## CLASS 8[Inference wrapper with masks] => Policy12
## FUNC 8[Load an exp12 or exp11 policy file] => load_policy
## @usecases
## - train12: model = from_exp11(torch.load(ckpt), OBS_LAYOUT12, STATE_SIZE12, state_map)
## - league12 / vec12: pol = load_policy(path, device, layout=..., state_size=...); a, st = pol.act(obs, st, starts, mask)
def _module_contract():
    pass
# endregion MODULE_CONTRACT
# GREP_SUMMARY: model12, graft, exp11 to exp12, zero-init columns, turn head 7, action mask, masked entropy, load_policy
# STRUCTURE: ▶ exp11 payload → ⚡ build ActorCritic12(layout12, nvec 9/2/7/2) → ⚡ copy weights (input columns by n_old map, head rows by turn map) → ⎋ model; act: logits → mask(-inf) → sample

import logging
import math
from typing import Any

import torch
import torch.nn as nn

from rush.model11 import DEFAULT_CONFIG as DEFAULT_CONFIG11
from rush.model11 import ActorCritic, policy_payload  # noqa: F401  (policy_payload re-exported)

logger = logging.getLogger(__name__)

# region BLOCK_CONSTANTS
NVEC11: list[int] = [9, 2, 5, 2]
NVEC12: list[int] = [9, 2, 7, 2]            # turn: 0-4 as exp11, 5 = -0.5 deg, 6 = +0.5 deg
TURN_HEAD: int = 2
SHOOT_HEAD: int = 1
DASH_HEAD: int = 3
NEW_TURN_BIAS: float = -8.0
MASK_FILL: float = -1e9                      # after .float(): finite, so masked entropy stays 0, not NaN
# endregion BLOCK_CONSTANTS


def _flat(size: Any) -> int:
    return math.prod(size) if isinstance(size, (list, tuple)) else int(size)


## @purpose A segment's old width: features per token / vector slot, or CHANNELS for a grid. world12 writes a grid's
## n_old as its whole old size [C, H, W]; a bare int means channels.
def seg_n_old(s: dict) -> int:
    v = s.get("n_old")
    if s["kind"] == "grid":
        v = s["size"] if v is None else v
        return int(v[0]) if isinstance(v, (list, tuple)) else int(v)
    return int(s["size"]) if v is None else int(v)


# region FUNC_layout_column_map
## @purpose Where the exp11 columns of the vector part / of each token live in the exp12 layout.
## @io layout12 (segments with n_old) -> (vec_map: list[int] new column of each old vector column,
##   tok_n_old: {segment name: n_old}, grid_old_channels: {segment name: channels})
## @rationale Vector segments are concatenated in layout order by LayoutEncoder, so the old vector column
## j of segment s, slot c maps to new_offset(s) + c * size_new(s) + j.
def layout_column_map(layout12: list[dict]) -> tuple[list[int], dict[str, int], dict[str, int]]:
    vec_map: list[int] = []
    tok_n_old: dict[str, int] = {}
    grid_old: dict[str, int] = {}
    off = 0
    for s in layout12:
        kind, cnt = s["kind"], int(s.get("count", 1))
        if kind == "vector":
            size = int(s["size"])
            n_old = seg_n_old(s)
            if n_old > 0:
                for c in range(cnt):
                    vec_map.extend(off + c * size + j for j in range(n_old))
            off += cnt * size
        elif kind == "tokens":
            tok_n_old[s["name"]] = seg_n_old(s)
        elif kind == "grid":
            grid_old[s["name"]] = seg_n_old(s)
    return vec_map, tok_n_old, grid_old
# endregion FUNC_layout_column_map


# region FUNC_obs_column_map
## @purpose New flat index of every exp11 flat observation column (segments in layout order; per token the
## first n_old features; per grid the first n_old channels). Lets a world11 observation be placed into an
## exp12 observation (tests, the legacy bridge world) and says which exp12 columns are "new".
## @io layout12 -> LongTensor (OBS_SIZE11,)
def obs_column_map(layout12: list[dict]) -> torch.Tensor:
    cols: list[int] = []
    for s in layout12:
        start, cnt = int(s["start"]), int(s.get("count", 1))
        if s["kind"] in ("vector", "tokens"):
            size = int(s["size"])
            n_old = seg_n_old(s)
            for c in range(cnt):
                cols.extend(start + c * size + j for j in range(n_old))
        else:
            _, h, w = (int(v) for v in s["size"])
            cols.extend(range(start, start + seg_n_old(s) * h * w))
    return torch.as_tensor(cols, dtype=torch.long)
# endregion FUNC_obs_column_map


# region FUNC__masked
def _masked(lg: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    lg = lg.float()
    return lg if mask is None else lg.masked_fill(~mask, MASK_FILL)
# endregion FUNC__masked


# region CLASS_ActorCritic12
## @purpose model11.ActorCritic with an optional legal-action mask (B, sum nvec) in sampling and evaluation,
## and per-head entropies that are 0 where a head has one legal action.
## @complexity 7
class ActorCritic12(ActorCritic):
    def _split_mask(self, mask: torch.Tensor | None) -> list[torch.Tensor | None]:
        if mask is None:
            return [None] * len(self.nvec)
        return list(torch.split(mask, self.nvec, dim=-1))

    def sample(self, logits: torch.Tensor, deterministic: bool = False, mask: torch.Tensor | None = None):  # type: ignore[override]
        acts, logps, ents = [], [], []
        for lg, mk in zip(self._split_logits(logits), self._split_mask(mask)):
            dist = torch.distributions.Categorical(logits=_masked(lg, mk))
            a = dist.logits.argmax(-1) if deterministic else dist.sample()
            acts.append(a)
            logps.append(dist.log_prob(a))
            ents.append(dist.entropy())
        return torch.stack(acts, -1), torch.stack(logps, -1).sum(-1), torch.stack(ents, -1)

    def evaluate_logits(self, logits: torch.Tensor, actions: torch.Tensor, mask: torch.Tensor | None = None):  # type: ignore[override]
        logps, ents = [], []
        for k, (lg, mk) in enumerate(zip(self._split_logits(logits), self._split_mask(mask))):
            dist = torch.distributions.Categorical(logits=_masked(lg, mk))
            logps.append(dist.log_prob(actions[..., k]))
            ents.append(dist.entropy())
        return torch.stack(logps, -1).sum(-1), torch.stack(ents, -1)

    @torch.no_grad()
    def act_rollout(self, obs, gstate, state, starts, deterministic: bool = False, mask: torch.Tensor | None = None):  # type: ignore[override]
        feats = self.encoder(obs)
        logits, state = self.step_actor(feats, state, starts)
        actions, logp, _ = self.sample(logits, deterministic, mask)
        v = self.value_from(feats, gstate)
        return actions, logp, v, state

    # region FUNC_evaluate_sequence
    ## @io as model11, plus mask (T, B, sum nvec) bool or None
    def evaluate_sequence(self, obs, gstate, actions, starts, state0, return_aux: bool = False, mask: torch.Tensor | None = None):  # type: ignore[override]
        t, b = obs.shape[0], obs.shape[1]
        feats = self.encoder(obs.reshape(t * b, -1))
        v = self.value_from(feats, gstate.reshape(t * b, -1)).view(t, b)
        feats = feats.view(t, b, -1)
        h, c = state0
        keep0 = (1.0 - starts[0].float()).view(1, -1, 1)
        seq = self._lstm_segments(feats, starts, h * keep0, c * keep0)
        logits = self.actor_head(seq.reshape(t * b, -1))
        mk = None if mask is None else mask.reshape(t * b, -1)
        logp, ent = self.evaluate_logits(logits, actions.reshape(t * b, -1), mk)
        if return_aux:
            if self.aux_aim_head is None:
                raise RuntimeError("evaluate_sequence(return_aux=True) on a model built without aux_aim")
            aux = self.aux_aim_head(seq.reshape(t * b, -1)).view(t, b)
            return logp.view(t, b), ent.view(t, b, -1), v, aux
        return logp.view(t, b), ent.view(t, b, -1), v
    # endregion FUNC_evaluate_sequence
# endregion CLASS_ActorCritic12


def build_model12(config: dict[str, Any], device: torch.device | str = "cpu") -> ActorCritic12:
    for key in ("layout", "state_size"):
        if key not in config:
            raise ValueError(f"model config lacks {key!r}")
    model = ActorCritic12({**DEFAULT_CONFIG11, **config}).to(device)
    logger.info(f"[IMP:8][build_model12][BUILD] ActorCritic12 params={sum(p.numel() for p in model.parameters())}, "
                f"nvec={model.nvec}, state={config['state_size']} [VALUE]")
    return model


# region FUNC__copy_linear_cols
## @purpose new.weight[:, cols[j]] = old.weight[:, j], other columns 0; bias copied. In place, no grad.
@torch.no_grad()
def _copy_linear_cols(new: nn.Linear, old_w: torch.Tensor, old_b: torch.Tensor | None, cols: list[int]) -> None:
    if old_w.shape[1] != len(cols):
        raise ValueError(f"column map has {len(cols)} entries for a {old_w.shape[1]}-column weight")
    new.weight.zero_()
    new.weight[:, torch.as_tensor(cols, dtype=torch.long)] = old_w.to(new.weight)
    if old_b is not None:
        new.bias.copy_(old_b)
# endregion FUNC__copy_linear_cols


# region FUNC_from_exp11
## @purpose Build an exp12 network from an exp11 payload so that, on the old part of the observation,
## it computes exactly exp11's logits (plus two new turn logits at NEW_TURN_BIAS) and value.
## @io payload11, layout12, state_size12, state_map (new column of each old global-state column; None = the
##   old columns are the prefix) -> ActorCritic12 on `device`
## @complexity 9
def from_exp11(payload11: dict[str, Any], layout12: list[dict], state_size12: int, state_map: list[int] | None = None,
               device: torch.device | str = "cpu", nvec12: list[int] | None = None) -> ActorCritic12:
    cfg11 = dict(payload11["config"])
    sd11 = payload11["state_dict"]
    nvec11 = [int(v) for v in cfg11.get("nvec", NVEC11)]
    nvec12 = list(nvec12 or NVEC12)
    old_layout = cfg11["layout"]
    old_names = [s["name"] for s in old_layout]
    new_names = [s["name"] for s in layout12]
    if old_names != new_names:
        raise ValueError(f"segments changed {old_names} -> {new_names}: only features appended to existing segments graft "
                         f"function-preservingly (see @rationale)")
    for so, sn in zip(old_layout, layout12):
        n_old = seg_n_old(sn)
        want = so["size"][0] if so["kind"] == "grid" else int(so["size"])
        if n_old != want or int(so.get("count", 1)) != int(sn.get("count", 1)):
            raise ValueError(f"segment {sn['name']}: n_old={n_old}/count={sn.get('count')} but exp11 had size {so['size']}/count {so.get('count')}")
        if so["kind"] == "grid" and list(so["size"][1:]) != list(sn["size"][1:]):
            raise ValueError(f"grid {sn['name']} spatial size changed {so['size']} -> {sn['size']}")

    cfg12 = {**cfg11, "layout": [dict(s) for s in layout12], "state_size": int(state_size12), "nvec": nvec12,
             "grafted_from": {"step": int(payload11.get("step", 0)), "nvec": nvec11, "state_size": int(cfg11["state_size"])}}
    model = build_model12(cfg12, "cpu")
    new_sd = model.state_dict()

    # 1) Everything whose shape did not change is copied verbatim.
    vec_map, tok_n_old, grid_old = layout_column_map(layout12)
    tok_names = [s["name"] for s in layout12 if s["kind"] == "tokens"]
    grid_names = [s["name"] for s in layout12 if s["kind"] == "grid"]
    special = {"encoder.core.0.weight", "state_mlp.0.weight", "actor_head.2.weight", "actor_head.2.bias"}
    special |= {f"encoder.tok_embed.{k}.weight" for k in range(len(tok_names))}
    special |= {f"encoder.grid_cnn.{k}.0.weight" for k in range(len(grid_names))}
    copied, reshaped = 0, []
    for k, v in sd11.items():
        if k in special:
            continue
        if k not in new_sd:
            raise KeyError(f"exp11 key {k} has no exp12 counterpart")
        if new_sd[k].shape != v.shape:
            raise ValueError(f"unexpected shape change at {k}: {tuple(v.shape)} -> {tuple(new_sd[k].shape)}")
        new_sd[k] = v.clone()
        copied += 1
    model.load_state_dict(new_sd)

    # 2) Input layers: exp11 columns at their new positions, new columns zero.
    enc = model.encoder
    _copy_linear_cols(enc.core[0], sd11["encoder.core.0.weight"], None, vec_map)
    reshaped.append(("encoder.core.0", sd11["encoder.core.0.weight"].shape[1], enc.core[0].in_features))
    for k, name in enumerate(tok_names):
        w = sd11[f"encoder.tok_embed.{k}.weight"]
        _copy_linear_cols(enc.tok_embed[k], w, None, list(range(tok_n_old[name])))
        reshaped.append((f"encoder.tok_embed.{k}({name})", w.shape[1], enc.tok_embed[k].in_features))
    with torch.no_grad():
        for k, name in enumerate(grid_names):
            w = sd11[f"encoder.grid_cnn.{k}.0.weight"]                           # (32, C_old, 3, 3)
            conv = enc.grid_cnn[k][0]
            conv.weight.zero_()
            conv.weight[:, : grid_old[name]] = w
            reshaped.append((f"encoder.grid_cnn.{k}.0({name})", w.shape[1], conv.in_channels))
    s_old = int(cfg11["state_size"])
    smap = list(state_map) if state_map is not None else list(range(s_old))
    _copy_linear_cols(model.state_mlp[0], sd11["state_mlp.0.weight"], None, smap)
    reshaped.append(("state_mlp.0", s_old, model.state_mlp[0].in_features))

    # 3) Action head: rows by head; the turn head's new actions start switched off.
    w_old, b_old = sd11["actor_head.2.weight"], sd11["actor_head.2.bias"]
    head = model.actor_head[2]
    with torch.no_grad():
        head.weight.zero_()
        head.bias.zero_()
        o_old, o_new = 0, 0
        for h, (n_old, n_new) in enumerate(zip(nvec11, nvec12)):
            if n_new < n_old:
                raise ValueError(f"head {h} shrinks {n_old} -> {n_new}")
            head.weight[o_new:o_new + n_old] = w_old[o_old:o_old + n_old]
            head.bias[o_new:o_new + n_old] = b_old[o_old:o_old + n_old]
            if n_new > n_old:
                head.bias[o_new + n_old:o_new + n_new] = NEW_TURN_BIAS
            o_old += n_old
            o_new += n_new
    model = model.to(device)
    logger.info(f"[IMP:9][from_exp11][RESULT] grafted exp11 step={payload11.get('step', 0)}: copied={copied} tensors verbatim, "
                f"input layers widened {reshaped}, heads {nvec11}->{nvec12} (new logits bias={NEW_TURN_BIAS}) [VALUE]")
    return model
# endregion FUNC_from_exp11


# region CLASS_Policy12
## @purpose Frozen inference view with masks: the object league12 and vec12 drive opponents with.
class Policy12:
    def __init__(self, model: ActorCritic12, step: int = 0, fork: str | None = None) -> None:
        self.model = model.eval()
        self.step = step
        self.fork = fork
        self.device = next(model.parameters()).device
        self.config = model.config

    def initial_state(self, batch: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.model.initial_state(batch, self.device)

    @torch.no_grad()
    def act(self, obs: torch.Tensor, state: tuple[torch.Tensor, torch.Tensor], episode_starts: torch.Tensor,
            deterministic: bool = False, mask: torch.Tensor | None = None):
        obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        starts = torch.as_tensor(episode_starts, dtype=torch.float32, device=self.device)
        feats = self.model.encoder(obs)
        logits, state = self.model.step_actor(feats, state, starts)
        actions, _, _ = self.model.sample(logits, deterministic, mask)
        return actions, state
# endregion CLASS_Policy12


# region FUNC_load_policy
## @purpose Load an exp12 file, or an exp11 file grafted on the fly (then layout/state_size are required).
def load_policy(path: str, device: torch.device | str = "cpu", layout: list[dict] | None = None,
                state_size: int | None = None, state_map: list[int] | None = None) -> Policy12:
    payload = torch.load(str(path), map_location="cpu", weights_only=False)
    cfg = payload["config"]
    if list(cfg.get("nvec", NVEC11)) == NVEC12:
        model = build_model12(cfg, device)
        model.load_state_dict(payload["state_dict"])
    else:
        if layout is None or state_size is None:
            raise ValueError(f"{path} is an exp11 file: pass layout and state_size to graft it")
        model = from_exp11(payload, layout, state_size, state_map, device)
    logger.info(f"[IMP:8][load_policy][EXEC] loaded {path} step={payload.get('step', 0)} fork={payload.get('fork')} [VALUE]")
    return Policy12(model, int(payload.get("step", 0)), payload.get("fork"))
# endregion FUNC_load_policy
