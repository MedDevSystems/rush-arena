from __future__ import annotations

# region MODULE_CONTRACT [DOMAIN(9): RL; CONCEPT(10): ActorCriticMAPPO; TECH(9): torch]
## @modulecontract
## @purpose Experiment-11 actor-critic: an encoder generated from OBS_LAYOUT, a recurrent actor
## with four categorical heads, and a centralized critic that also reads the fog-free global
## state (MAPPO). Also the one public inference entrypoint, load_policy(), that the league and
## the replay recorder use.
## @scope Network definition, value normalization, checkpoint payload format, inference wrapper.
## No rollouts, no optimizer, no environment.
## @input obs (B, OBS_SIZE), global_state (B, STATE_SIZE), LSTM state
## @output action logits / sampled actions (B, 4), normalized and raw values (B,)
## @links USES_API(9): torch; LINKS_TO: ppo11, vec11, train11, league11 (load_policy), record11 (load_policy)
## @invariants
## - The encoder reads offsets only from the layout stored in config — never hard-coded
## - A token slot that is exact zeros is masked out of attention; the core token is never masked,
##   so attention always has at least one key (no NaN rows)
## - The actor never sees global_state: fog of war holds at decision time
## - Policy files are torch.save({"state_dict", "config", "step"}); extra keys (optimizer) allowed
## - Value normalization statistics are module buffers, so they travel inside state_dict
## @rationale
## Q: Why is the critic not recurrent?
## A: It reads the global state — the full information the actor's LSTM tries to reconstruct.
## A: A feed-forward critic makes V(s) a function of (obs, state) only, so the value of a
## A: truncated episode's final observation is exact, with no hidden state to invent.
## Q: Why share the encoder between actor and critic?
## A: The encoder is the expensive part (attention over up to ~30 tokens, a CNN over the grid).
## A: Sharing halves the compute of every rollout and update. The critic gets its own head on
## A: top plus the global-state branch, so it is not bottlenecked by what the actor keeps.
## Q: Why running mean/std value normalization instead of PopArt?
## A: Returns here are dominated by ±10 terminal payouts and move slowly across training; the
## A: MAPPO study found plain running normalization of targets enough. PopArt's output
## A: re-scaling is a later knob if the value scale drifts fast.
## @changes
## LAST_CHANGE: [v0.2.0] Optional auxiliary aim head (config "aux_aim"), evaluate_sequence(return_aux=True).
## PREV: [v0.1.0] Initial experiment-11 actor-critic.
## @modulemap
## CLASS 7[Running mean/variance as buffers] => RunningNorm
## CLASS 7[Pre-norm attention block] => _AttnBlock
## CLASS 9[Layout-driven observation encoder] => LayoutEncoder
## CLASS 10[Actor-critic with centralized critic] => ActorCritic
## CLASS 8[Inference wrapper for league and recorder] => Policy
## FUNC 8[Build from config / load a policy file] => build_model, load_policy, policy_payload
## @usecases
## - ppo11: model.evaluate_sequence(...) during updates, model.act_rollout(...) during collection
## - league11 / record11: policy = load_policy(path, device); a, st = policy.act(obs, st, starts)
def _module_contract():
    pass
# endregion MODULE_CONTRACT
# GREP_SUMMARY: actor critic, MAPPO, centralized critic, layout encoder, attention, LSTM, value normalization, load_policy
# STRUCTURE: ▶ obs → ⚡ LayoutEncoder(vector|tokens|grid → attention → pool) → ⚡ LSTM → ⚡ heads → ⎋ actions; ⊕ features+global_state → ⚡ critic MLP → ⎋ value

import logging
from typing import Any

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# region BLOCK_DEFAULTS
DEFAULT_CONFIG: dict[str, Any] = {
    "d_model": 128,
    "heads": 4,
    "layers": 2,
    "features": 256,
    "lstm_hidden": 256,
    "critic_hidden": 256,
    "state_hidden": 256,
    "nvec": [9, 2, 5, 2],
    # Auxiliary "would a shot hit now" head (experiment 11b). Off by default so a config without the key
    # (every exp11a file) rebuilds exactly the network it was saved from.
    "aux_aim": False,
}
# endregion BLOCK_DEFAULTS


# region CLASS_RunningNorm
## @purpose Running mean/variance of scalar value targets (Welford-style batch merge), kept as buffers.
class RunningNorm(nn.Module):
    def __init__(self, eps: float = 1e-4) -> None:
        super().__init__()
        self.register_buffer("mean", torch.zeros(()))
        self.register_buffer("var", torch.ones(()))
        self.register_buffer("count", torch.tensor(eps))

    @torch.no_grad()
    def update(self, x: torch.Tensor) -> None:
        x = x.detach().float().flatten()
        b_mean, b_var, b_n = x.mean(), x.var(unbiased=False), float(x.numel())
        delta = b_mean - self.mean
        tot = self.count + b_n
        self.mean.copy_(self.mean + delta * b_n / tot)
        m2 = self.var * self.count + b_var * b_n + delta.pow(2) * self.count * b_n / tot
        self.var.copy_(m2 / tot)
        self.count.copy_(tot)

    @property
    def std(self) -> torch.Tensor:
        return self.var.clamp(min=1e-8).sqrt()

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.std + self.mean
# endregion CLASS_RunningNorm


# region CLASS__AttnBlock
## @purpose Pre-norm self-attention + feed-forward. (B, T, D), pad (B, T) -> (B, T, D)
class _AttnBlock(nn.Module):
    def __init__(self, dim: int, heads: int) -> None:
        super().__init__()
        self.n1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.n2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(nn.Linear(dim, dim * 2), nn.GELU(), nn.Linear(dim * 2, dim))

    def forward(self, x: torch.Tensor, pad: torch.Tensor) -> torch.Tensor:
        h = self.n1(x)
        a, _ = self.attn(h, h, h, key_padding_mask=pad, need_weights=False)
        x = x + a
        return x + self.ff(self.n2(x))
# endregion CLASS__AttnBlock


# region CLASS_LayoutEncoder
## @purpose Turn a flat observation into features by the segment layout: vector segments form the
## always-present core token, token segments become typed embedded tokens (exact-zero slots
## masked), grid segments pass a small CNN and become one token each. Attention mixes all tokens.
## @io (B, OBS_SIZE) -> (B, features)
## @complexity 8
class LayoutEncoder(nn.Module):
    def __init__(self, layout: list[dict], d_model: int, heads: int, layers: int, features: int) -> None:
        super().__init__()
        self.layout = [dict(s) for s in layout]
        self.vec_segs = [s for s in self.layout if s["kind"] == "vector"]
        self.tok_segs = [s for s in self.layout if s["kind"] == "tokens"]
        self.grid_segs = [s for s in self.layout if s["kind"] == "grid"]
        if not self.vec_segs:
            raise ValueError("OBS_LAYOUT needs at least one vector segment (the core)")
        for s in self.layout:
            if s["kind"] not in ("vector", "tokens", "grid"):
                raise ValueError(f"Unknown segment kind {s['kind']!r} in {s['name']}")

        vec_size = sum(int(s["size"]) * int(s.get("count", 1)) for s in self.vec_segs)
        self.core = nn.Sequential(nn.Linear(vec_size, d_model), nn.GELU(), nn.Linear(d_model, d_model))
        self.tok_embed = nn.ModuleList([nn.Linear(int(s["size"]), d_model) for s in self.tok_segs])
        n_types = 1 + len(self.tok_segs) + len(self.grid_segs)
        self.type_embed = nn.Parameter(torch.randn(n_types, d_model) * 0.02)

        self.grid_cnn = nn.ModuleList()
        for s in self.grid_segs:
            c, _, _ = (int(v) for v in s["size"])
            self.grid_cnn.append(nn.Sequential(
                nn.Conv2d(c, 32, 3, stride=2, padding=1), nn.GELU(),
                nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.GELU(),
                nn.AdaptiveAvgPool2d((2, 2)), nn.Flatten(), nn.Linear(256, d_model),
            ))

        self.blocks = nn.ModuleList([_AttnBlock(d_model, heads) for _ in range(layers)])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(nn.Linear(d_model * 2, features), nn.GELU())
        self.features = features

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        b = obs.shape[0]
        vec = torch.cat([obs[:, s["start"]: s["start"] + int(s["size"]) * int(s.get("count", 1))] for s in self.vec_segs], dim=1)
        core = self.core(vec) + self.type_embed[0]
        tokens = [core.unsqueeze(1)]
        used = [torch.ones(b, 1, dtype=torch.bool, device=obs.device)]

        for k, s in enumerate(self.tok_segs):
            cnt, size = int(s["count"]), int(s["size"])
            raw = obs[:, s["start"]: s["start"] + cnt * size].reshape(b, cnt, size)
            tokens.append(self.tok_embed[k](raw) + self.type_embed[1 + k])
            used.append(raw.abs().sum(-1) > 0)

        for k, s in enumerate(self.grid_segs):
            c, h, w = (int(v) for v in s["size"])
            raw = obs[:, s["start"]: s["start"] + c * h * w].reshape(b, c, h, w)
            g = self.grid_cnn[k](raw) + self.type_embed[1 + len(self.tok_segs) + k]
            tokens.append(g.unsqueeze(1))
            used.append(torch.ones(b, 1, dtype=torch.bool, device=obs.device))

        x = torch.cat(tokens, dim=1)
        mask = torch.cat(used, dim=1)
        pad = ~mask
        for blk in self.blocks:
            x = blk(x, pad)
        x = self.norm(x)
        wts = mask.unsqueeze(-1).to(x.dtype)
        pooled = (x * wts).sum(1) / wts.sum(1).clamp(min=1.0)
        return self.head(torch.cat([x[:, 0], pooled], dim=1))
# endregion CLASS_LayoutEncoder


# region CLASS_ActorCritic
## @purpose Shared encoder; recurrent actor over observations; feed-forward centralized critic over
## encoder features + global state; value normalization for the critic's targets.
## @complexity 9
class ActorCritic(nn.Module):
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        cfg = {**DEFAULT_CONFIG, **config}
        self.config = cfg
        self.nvec = [int(v) for v in cfg["nvec"]]
        self.encoder = LayoutEncoder(cfg["layout"], cfg["d_model"], cfg["heads"], cfg["layers"], cfg["features"])
        self.lstm = nn.LSTM(cfg["features"], cfg["lstm_hidden"], num_layers=1)
        self.actor_head = nn.Sequential(nn.Linear(cfg["lstm_hidden"], 256), nn.GELU(), nn.Linear(256, sum(self.nvec)))
        self.state_mlp = nn.Sequential(nn.Linear(int(cfg["state_size"]), cfg["state_hidden"]), nn.GELU(),
                                       nn.Linear(cfg["state_hidden"], cfg["state_hidden"]), nn.GELU())
        self.critic = nn.Sequential(nn.Linear(cfg["features"] + cfg["state_hidden"], cfg["critic_hidden"]), nn.GELU(),
                                    nn.Linear(cfg["critic_hidden"], 1))
        self.value_norm = RunningNorm()
        # Reads the actor's LSTM output, so the aim signal shapes the same memory the policy acts from.
        self.aux_aim_head = nn.Linear(cfg["lstm_hidden"], 1) if cfg.get("aux_aim") else None
        self._init_weights()

    def _init_weights(self) -> None:
        # Small final layers: near-uniform initial policy, near-zero initial value (standard PPO practice).
        nn.init.orthogonal_(self.actor_head[-1].weight, gain=0.01)
        nn.init.zeros_(self.actor_head[-1].bias)
        nn.init.orthogonal_(self.critic[-1].weight, gain=1.0)
        nn.init.zeros_(self.critic[-1].bias)

    @property
    def hidden(self) -> int:
        return int(self.config["lstm_hidden"])

    def initial_state(self, batch: int, device: torch.device | str | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        dev = device if device is not None else next(self.parameters()).device
        z = torch.zeros(1, batch, self.hidden, device=dev)
        return z, z.clone()

    # region FUNC_value
    ## @purpose Normalized value from encoder features and global state. (B, F), (B, S) -> (B,)
    def value_from(self, feats: torch.Tensor, gstate: torch.Tensor) -> torch.Tensor:
        return self.critic(torch.cat([feats, self.state_mlp(gstate)], dim=1)).squeeze(-1)

    def value(self, obs: torch.Tensor, gstate: torch.Tensor) -> torch.Tensor:
        """Normalized value of (obs, global_state); feed-forward, no LSTM state needed."""
        return self.value_from(self.encoder(obs), gstate)
    # endregion FUNC_value

    def _split_logits(self, logits: torch.Tensor) -> list[torch.Tensor]:
        return list(torch.split(logits, self.nvec, dim=-1))

    # region FUNC_step_actor
    ## @purpose One recurrent actor step. Rows with episode_start=1 start from zero memory.
    ## @io feats (B, F), state, starts (B,) -> logits (B, sum nvec), state
    def step_actor(self, feats: torch.Tensor, state: tuple[torch.Tensor, torch.Tensor], starts: torch.Tensor):
        keep = (1.0 - starts.float()).view(1, -1, 1)
        h, c = state[0] * keep, state[1] * keep
        out, (h, c) = self.lstm(feats.unsqueeze(0), (h, c))
        return self.actor_head(out.squeeze(0)), (h, c)
    # endregion FUNC_step_actor

    # region FUNC_act_rollout
    ## @purpose Collection-time step: sample actions, log-prob, normalized value, new state.
    @torch.no_grad()
    def act_rollout(self, obs, gstate, state, starts, deterministic: bool = False):
        feats = self.encoder(obs)
        logits, state = self.step_actor(feats, state, starts)
        actions, logp, _ = self.sample(logits, deterministic)
        v = self.value_from(feats, gstate)
        return actions, logp, v, state
    # endregion FUNC_act_rollout

    def sample(self, logits: torch.Tensor, deterministic: bool = False):
        acts, logps, ents = [], [], []
        for lg in self._split_logits(logits):
            dist = torch.distributions.Categorical(logits=lg.float())
            a = lg.argmax(-1) if deterministic else dist.sample()
            acts.append(a)
            logps.append(dist.log_prob(a))
            ents.append(dist.entropy())
        return torch.stack(acts, -1), torch.stack(logps, -1).sum(-1), torch.stack(ents, -1)

    def evaluate_logits(self, logits: torch.Tensor, actions: torch.Tensor):
        """Log-prob of taken actions (sum over heads) and entropy per head. -> (B,), (B, n_heads)"""
        logps, ents = [], []
        for k, lg in enumerate(self._split_logits(logits)):
            dist = torch.distributions.Categorical(logits=lg.float())
            logps.append(dist.log_prob(actions[..., k]))
            ents.append(dist.entropy())
        return torch.stack(logps, -1).sum(-1), torch.stack(ents, -1)

    # region FUNC_evaluate_sequence
    ## @purpose Update-time pass over sequence chunks: encoder batched over all T*B, LSTM unrolled
    ## over T with per-step resets from `starts`, critic batched.
    ## @io obs (T, B, O), gstate (T, B, S), actions (T, B, 4), starts (T, B), state0 (h, c) of (1, B, H)
    ## ->  logp (T, B), entropy (T, B, heads), value_norm (T, B)
    ## @complexity 7
    def evaluate_sequence(self, obs, gstate, actions, starts, state0, return_aux: bool = False):
        """With return_aux=True (model built with aux_aim) a 4th value: aim logits (T, B)."""
        t, b = obs.shape[0], obs.shape[1]
        feats = self.encoder(obs.reshape(t * b, -1))
        v = self.value_from(feats, gstate.reshape(t * b, -1)).view(t, b)
        feats = feats.view(t, b, -1)
        h, c = state0
        keep0 = (1.0 - starts[0].float()).view(1, -1, 1)
        h, c = h * keep0, c * keep0
        seq = self._lstm_segments(feats, starts, h, c)
        logits = self.actor_head(seq.reshape(t * b, -1))
        logp, ent = self.evaluate_logits(logits, actions.reshape(t * b, -1))
        if return_aux:
            if self.aux_aim_head is None:
                raise RuntimeError("evaluate_sequence(return_aux=True) on a model built without aux_aim")
            aux = self.aux_aim_head(seq.reshape(t * b, -1)).view(t, b)
            return logp.view(t, b), ent.view(t, b, -1), v, aux
        return logp.view(t, b), ent.view(t, b, -1), v
    # endregion FUNC_evaluate_sequence

    # region FUNC__lstm_segments
    ## @purpose LSTM over a chunk with per-row episode starts in ONE cuDNN call: every episode of every
    ## row becomes its own packed sequence; a row's first episode starts from that row's state, every
    ## later one from zeros (an episode start zeroes the state).
    ## @io feats (T, B, F), starts (T, B), h/c (1, B, H) already masked for starts[0] -> (T, B, H)
    ## @rationale
    ## Q: Why packing and not "cut the chunk at every reset time" (previous version) or "one call per
    ## Q: episode index, padded to T" (tried first)?
    ## A: An LSTM's cost is its SEQUENTIAL steps, not its call count. Cutting at reset times keeps T steps
    ## A: but pays ~1 ms fixed cost per extra call (~7 calls per chunk once matches end); padding each
    ## A: episode index to T pays T steps per index. Measured on the V100 (320 rows, fwd+bwd, amp),
    ## A: see docs/EXP11_NOTES_perf.md. A packed batch is one call over at most T steps.
    def _lstm_segments(self, feats: torch.Tensor, starts: torch.Tensor, h: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        t, b, f = feats.shape
        if t == 1:
            return self.lstm(feats, (h, c))[0]
        dev = feats.device
        seg = torch.zeros(t, b, dtype=torch.long, device=dev)
        seg[1:] = (starts[1:] > 0).long().cumsum(0)                          # episode index within the chunk
        last = seg[-1]
        n_extra, k_max = torch.stack([last.sum(), last.max()]).tolist()       # one host sync for both
        if n_extra == 0:
            return self.lstm(feats, (h, c))[0]

        # Sequence table: row b's episode k is sequence offset[b] + k.
        n_ep = last + 1                                                       # (B,)
        offset = torch.cumsum(n_ep, 0) - n_ep
        s_total = b + n_extra
        seq_row = torch.repeat_interleave(torch.arange(b, device=dev), n_ep)  # (S,)
        seq_k = torch.arange(s_total, device=dev) - offset[seq_row]
        # begin/length of every episode: counts of steps with seg < k / == k, per row.
        ks = torch.arange(k_max + 1, device=dev).view(-1, 1, 1)              # (K, 1, 1)
        below = (seg.unsqueeze(0) < ks).sum(1)                                # (K, B)
        equal = (seg.unsqueeze(0) == ks).sum(1)                               # (K, B)
        seq_begin = below[seq_k, seq_row]
        seq_len = equal[seq_k, seq_row]

        ar = torch.arange(t, device=dev).view(-1, 1)
        src = (ar + seq_begin.view(1, -1)).clamp(max=t - 1)                   # left-align every episode
        x = feats[:, seq_row].gather(0, src.unsqueeze(-1).expand(-1, -1, f))  # (T, S, F)
        first = (seq_k == 0).view(1, -1, 1).to(h.dtype)
        h0, c0 = h[:, seq_row] * first, c[:, seq_row] * first
        packed = nn.utils.rnn.pack_padded_sequence(x, seq_len.cpu(), enforce_sorted=False)
        y, _ = self.lstm(packed, (h0, c0))
        y, _ = nn.utils.rnn.pad_packed_sequence(y, total_length=t)            # (T, S, H)

        sid = offset.view(1, -1) + seg                                        # (T, B) sequence of each step
        local = ar - seq_begin[sid]                                           # position inside it
        return y[local, sid]                                                  # (T, B, H)
    # endregion FUNC__lstm_segments
# endregion CLASS_ActorCritic


# region FUNC_build_model
## @purpose Construct an ActorCritic from a config that carries layout and state size.
def build_model(config: dict[str, Any], device: torch.device | str = "cpu") -> ActorCritic:
    for key in ("layout", "state_size"):
        if key not in config:
            raise ValueError(f"model config lacks {key!r}")
    model = ActorCritic(config).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"[IMP:8][build_model][BUILD] ActorCritic params={n_params}, obs segments={len(config['layout'])}, state={config['state_size']} [VALUE]")
    return model
# endregion FUNC_build_model


def policy_payload(model: ActorCritic, step: int) -> dict[str, Any]:
    """The policy file body: exactly what load_policy() reads. torch.compile's `_orig_mod.` key
    prefix is stripped, so a compiled trainer still writes files an uncompiled loader can read."""
    return {"state_dict": {k.replace("_orig_mod.", ""): v.detach().cpu() for k, v in model.state_dict().items()},
            "config": dict(model.config), "step": int(step)}


# region CLASS_Policy
## @purpose Frozen inference view of an ActorCritic: the contract object league11 and record11 use.
## @complexity 5
class Policy:
    def __init__(self, model: ActorCritic, step: int = 0) -> None:
        self.model = model.eval()
        self.step = step
        self.device = next(model.parameters()).device
        self.config = model.config

    def initial_state(self, batch: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.model.initial_state(batch, self.device)

    @torch.no_grad()
    def act(self, obs: torch.Tensor, state: tuple[torch.Tensor, torch.Tensor], episode_starts: torch.Tensor,
            deterministic: bool = False) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        starts = torch.as_tensor(episode_starts, dtype=torch.float32, device=self.device)
        feats = self.model.encoder(obs)
        logits, state = self.model.step_actor(feats, state, starts)
        actions, _, _ = self.model.sample(logits, deterministic)
        return actions, state
# endregion CLASS_Policy


# region FUNC_load_policy
## @purpose Load a policy file (league snapshot or training checkpoint) for inference.
## @io (path, device) -> Policy
def load_policy(path: str, device: torch.device | str = "cpu") -> Policy:
    payload = torch.load(str(path), map_location=device, weights_only=False)
    # The config travels with the file: an exp11a file has no "aux_aim" key and rebuilds without the
    # head, so the strict load below holds for old and new files alike.
    model = build_model(payload["config"], device)
    model.load_state_dict(payload["state_dict"])
    logger.info(f"[IMP:8][load_policy][EXEC] Loaded {path} step={payload.get('step', 0)} [VALUE]")
    return Policy(model, int(payload.get("step", 0)))
# endregion FUNC_load_policy

