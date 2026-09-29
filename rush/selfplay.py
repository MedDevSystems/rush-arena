"""Self-play by generations for the rush soldier network.

The learner plays one side of every world against a frozen generation (the latest with p = latest_p, else an older
one) and learns from the battle's outcome with PPO: per-soldier rewards (damage, kills, captures, score, win) in each
class's weights, GAE, clipped policy loss, a value head (trained alone for the first iteration), and a KL anchor to
the starting network that decays. Every army is commanded by rush.army_commander (respawn points and squad orders);
the learner's side flips after every finished match of its world.

Every --gen-every iterations a candidate is written to <run>/league/cand_<iter>/. `rush-gate` decides whether it
becomes the next generation (league/gen_XXX + league/current.json); the trainer picks new generations up by itself.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from rush import world12 as W12
from rush.armies import TARGET_CAPS, assign_forks, parse_army, resolve_map
from rush.army_commander import ArmyCommander
from rush.forks12 import FORKS, trait_vector
from rush.model11 import policy_payload
from rush.model12 import MASK_FILL
from rush.model15 import CORE_ALIVE_COL, upgrade_core15
from rush.world15 import OBS_LAYOUT15, BigTrainWorld15, Rules15

logger = logging.getLogger(__name__)
FORK_NAMES: list[str] = list(FORKS)
NVEC: list[int] = list(W12.HEAD_SIZES12)
MANIFEST = "population.json"
GATE_LR_MULT = 20.0                     # the readout gates (x_gate) learn this much faster


@dataclass
class SPConfig:
    maps: str = "Warfront500"
    worlds: int = 6
    rollout: int = 1000
    bptt: int = 50
    minibatch_rows: int = 256
    epochs: int = 2
    lr: float = 5e-5
    clip: float = 0.2
    vf_coef: float = 0.5
    ent_coef: float = 0.003
    gamma: float = 0.995
    gae_lambda: float = 0.95
    max_grad_norm: float = 0.5
    target_kl: float = 0.03
    anchor_coef: float = 0.5           # λ of KL(π_start ‖ π) at the start ...
    anchor_floor: float = 0.05         # ... falling linearly to this ...
    anchor_iters: int = 40             # ... over this many iterations
    critic_warmup_iters: int = 1
    tau: float = 0.8                   # team share of the reward
    latest_p: float = 0.8              # an opponent is the latest generation with this p, else an older one
    gen_every: int = 5
    amp: bool = True
    seed: int = 0
    hours: float = 48.0
    max_iters: int = 0
    ckpt_every: int = 1


# ---- helpers ------------------------------------------------------------------------------------------------------
def compute_gae(rewards, values, dones, next_value, gamma, lam):
    adv = torch.zeros_like(rewards)
    last = torch.zeros_like(next_value)
    nv = next_value
    for t in reversed(range(rewards.shape[0])):
        nonterm = 1.0 - dones[t].float()
        delta = rewards[t] + gamma * nv * nonterm - values[t]
        last = delta + gamma * lam * nonterm * last
        adv[t] = last
        nv = values[t]
    return adv, adv + values


def masked_logp(logits: torch.Tensor, mask: torch.Tensor) -> list[torch.Tensor]:
    return [torch.log_softmax(lg.float().masked_fill(~mk, MASK_FILL), -1)
            for lg, mk in zip(torch.split(logits, NVEC, -1), torch.split(mask, NVEC, -1))]


def kl_heads(logits_p: torch.Tensor, logits_q: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    kl = 0.0
    for lp, lq in zip(masked_logp(logits_p, mask), masked_logp(logits_q, mask)):
        kl = kl + (lp.exp() * (lp - lq)).sum(-1)
    return kl


def param_groups(m: nn.Module, lr: float) -> list[dict]:
    gates = [p for n, p in m.named_parameters() if n.endswith("x_gate")]
    rest = [p for n, p in m.named_parameters() if not n.endswith("x_gate")]
    groups = [{"params": rest, "lr": lr, "lr_mult": 1.0}]
    if gates:
        groups.append({"params": gates, "lr": lr * GATE_LR_MULT, "lr_mult": GATE_LR_MULT})
    return groups


def load_net(path: Path, device: Any) -> Any:
    """ActorCritic15 from a ckpt file or a folder (config.json + model.safetensors / ckpt_*_latest.pt)."""
    from rush.hub import load_payload
    payload = load_payload(path) if path.is_dir() else torch.load(str(path), map_location="cpu", weights_only=False)
    m, _ = upgrade_core15(payload, OBS_LAYOUT15, device)
    return m.eval(), payload


def seq_eval(model: Any, obs, gstate, starts, state0, detach_feats: bool = False):
    t, b = obs.shape[0], obs.shape[1]
    feats, tok = model.encoder(obs.reshape(t * b, -1), return_tokens=True)
    v = None
    if gstate is not None:
        v = model.value_from(feats.detach() if detach_feats else feats, gstate.reshape(t * b, -1)).view(t, b)
    h, c = state0
    keep0 = (1.0 - starts[0].float()).view(1, -1, 1)
    seq = model._lstm_segments(feats.view(t, b, -1), starts, h * keep0, c * keep0)
    h2 = model._readout(seq.reshape(t * b, -1), tok, None)
    return model.actor_head(h2), v


class ArenaWorld:
    """One big battle: all 8 classes per army, their traits, magazines and reward weights, an ArmyCommander for both
    sides. A finished match continues on a map drawn from the list."""

    def __init__(self, maps: list[Any], device: str, rng: random.Random, c_reward: torch.Tensor) -> None:
        self.maps, self.device, self.rng, self.c_reward = maps, device, rng, c_reward
        self.world = None
        self.matches = 0
        self._build(maps[rng.randrange(len(maps))])

    def _build(self, big_map: Any) -> None:
        if self.world is not None:
            del self.world, self.commander
            if torch.device(self.device).type == "cuda":
                torch.cuda.empty_cache()
        w = BigTrainWorld15(big_map, device=self.device, seed=self.rng.randrange(1 << 30), rules_overrides={"ff_coef": 1.3},
                            rules15=Rules15())
        self.world, self.map_name = w, big_map.name
        self.T, self.A = w.T, w.A
        forks, agent_fork = assign_forks(self.T, parse_army("all"), parse_army("all"))
        dev = self.device
        self.cls = torch.tensor([FORK_NAMES.index(forks[k]) for k in agent_fork], dtype=torch.long, device=dev)
        w.set_traits(torch.tensor([trait_vector(forks[k]) for k in agent_fork], dtype=torch.float32, device=dev).view(1, self.A, -1))
        w.refill()
        w.set_ammo_caps(torch.tensor([float(TARGET_CAPS[forks[k]]) for k in agent_fork], device=dev).view(1, self.A))
        w.fill_ammo()
        w.set_reward_weights(self.c_reward.to(dev)[self.cls].view(1, self.A, -1))
        self.commander = ArmyCommander(w)
        self.obs, self.mask = w.observe()[0], w.action_mask()[0]
        self.start = True
        self.team = torch.cat([torch.zeros(self.T, dtype=torch.long), torch.ones(self.T, dtype=torch.long)]).to(dev)

    def step(self, acts: torch.Tensor) -> dict:
        out = self.world.step(acts.view(1, self.A, 4))
        info = out.info
        done = bool(info["done_cpu"][0])
        if done:
            self.matches += 1
            logger.info(f"match {self.matches} on {self.map_name}: winner {int(info['winner'][0])}, "
                        f"score {[round(float(s)) for s in self.world.score[0]]}")
            self._build(self.maps[self.rng.randrange(len(self.maps))])
        else:
            self.obs, self.mask = out.obs[0], info["action_mask"][0]
            self.start = False
        return {"info": info, "done": done, "r_indiv": out.r_indiv[0]}


def fork_reward_vector(spec, names):
    r = dict(getattr(spec, "reward", {}) or {})
    return [float(r.get(k, 1.0)) for k in names], float(r.get("team_spirit", 1.0))


# ---- trainer ------------------------------------------------------------------------------------------------------
class SelfPlayTrainer:
    def __init__(self, cfg: SPConfig, model: Any, anchor: Any, run_dir: Path, device: str, state: dict | None = None,
                 opt_state: dict | None = None) -> None:
        self.cfg, self.model, self.anchor = cfg, model, anchor
        self.run_dir = run_dir
        self.models_dir, self.league = run_dir / "models15", run_dir / "league"
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.league.mkdir(parents=True, exist_ok=True)
        self.device = dev = torch.device(device)
        self.use_amp = bool(cfg.amp) and dev.type == "cuda"
        self.opt = torch.optim.Adam(param_groups(model, cfg.lr), lr=cfg.lr, eps=1e-5)
        self.scaler = torch.amp.GradScaler("cuda") if self.use_amp else None
        if opt_state:
            try:
                self.opt.load_state_dict(opt_state["opt"])
                if opt_state.get("scaler") and self.scaler is not None:
                    self.scaler.load_state_dict(opt_state["scaler"])
            except (ValueError, KeyError) as e:
                logger.info(f"optimizer state not loaded ({e})")
        for g in self.opt.param_groups:
            g["lr"] = cfg.lr * float(g.get("lr_mult", 1.0))
        st = dict(state or {})
        self.it, self.steps = int(st.get("iter", 0)), int(st.get("steps", 0))
        self.hours_done, self.start_step = float(st.get("hours", 0.0)), int(st.get("start_step", 0))
        rw = [fork_reward_vector(FORKS[n], list(W12.REWARD_W12)) for n in FORK_NAMES]
        c_reward = torch.tensor([r for r, _ in rw], dtype=torch.float32)
        self.c_tau = (torch.tensor([t for _, t in rw], device=dev) * cfg.tau).clamp(0.0, 1.0)
        rng = random.Random(cfg.seed)
        maps = [resolve_map(s) for s in cfg.maps.split(",") if s.strip()]
        self.worlds = [ArenaWorld(maps, str(dev), random.Random(rng.randrange(1 << 30)), c_reward) for _ in range(cfg.worlds)]
        self.A = self.worlds[0].A
        self.l_team = [i % 2 for i in range(cfg.worlds)]           # flips after every finished match of that world
        self.R = R = self.A * cfg.worlds
        self.row_world = torch.arange(cfg.worlds, device=dev).repeat_interleave(self.A)
        self.row_team = torch.cat([w.team for w in self.worlds])
        self.row_cls = torch.cat([w.cls for w in self.worlds])
        self.gen = -1
        self.opp_cache: dict[int, Any] = {}
        self.rng_opp = random.Random(cfg.seed + 7)
        self._ensure_gen0()
        self._load_opponent()
        self.w_opp: list[int] = [self.gen for _ in range(cfg.worlds)]
        self.pending: set[int] = set()
        self._assign_rows()
        self.NL = int(self.L.numel())
        self.st_l = model.initial_state(R, dev)
        self.st_a = anchor.initial_state(R, dev)
        self.st_o = model.initial_state(R, dev)
        T, O = cfg.rollout, self.worlds[0].obs.shape[-1]
        M, S = self.worlds[0].mask.shape[-1], self.worlds[0].world.global_state().shape[-1]
        if T % cfg.bptt:
            raise SystemExit(f"bptt {cfg.bptt} must divide rollout {T}")
        self.M = M
        self.b_obs = torch.zeros(T, self.NL, O, dtype=torch.float16)            # host RAM
        self.b_gs = torch.zeros(T, self.NL, S, dtype=torch.float16)
        self.b_mask = torch.zeros(T, self.NL, M, dtype=torch.bool, device=dev)
        self.b_act = torch.zeros(T, self.NL, 4, dtype=torch.long, device=dev)
        self.b_logp, self.b_val, self.b_rew, self.b_done, self.b_start = (torch.zeros(T, self.NL, device=dev) for _ in range(5))
        self.b_alive = torch.zeros(T, self.NL, dtype=torch.bool, device=dev)
        nc, H = T // cfg.bptt, self.st_l[0].shape[-1]
        self.h0 = torch.zeros(nc, 1, self.NL, H, device=dev)
        self.c0 = torch.zeros_like(self.h0)
        self.ah0 = torch.zeros(nc, 1, self.NL, self.st_a[0].shape[-1], device=dev)
        self.ac0 = torch.zeros_like(self.ah0)
        self.core_start = model.core_start
        logger.info(f"self-play: maps {[m.name for m in maps]}, {cfg.worlds} worlds × {self.A} soldiers, learner rows {self.NL}, "
                    f"rollout {T} -> {T * self.NL} decisions per iteration, observation buffer "
                    f"{T * self.NL * O * 2 / 2**30:.1f} GiB in RAM, generation {self.gen}, iteration {self.it}")

    # ---- league ----
    def _ensure_gen0(self) -> None:
        if (self.league / "current.json").exists():
            return
        g0 = self.league / "gen_000"
        g0.mkdir(parents=True, exist_ok=True)
        torch.save(policy_payload(self.anchor, self.start_step), g0 / "ckpt_base_latest.pt")
        (self.league / "current.json").write_text(json.dumps({"gen": 0, "dir": "gen_000", "from": "init", "t": int(time.time())}))

    def _opp_model(self, g: int) -> Any:
        if g not in self.opp_cache:
            m, _ = load_net(self.league / f"gen_{g:03d}" / "ckpt_base_latest.pt", self.device)
            for q in m.parameters():
                q.requires_grad_(False)
            self.opp_cache[g] = m
        return self.opp_cache[g]

    def _pool(self) -> list[int]:
        wd = self.league / "withdrawn.txt"
        withdrawn = set(wd.read_text().split()) if wd.exists() else set()
        gens = [int(d.name[4:]) for d in self.league.glob("gen_*")
                if d.name not in withdrawn and (d / "ckpt_base_latest.pt").exists() and int(d.name[4:]) <= self.gen]
        return sorted(gens) or [self.gen]

    def _load_opponent(self) -> bool:
        cur = json.loads((self.league / "current.json").read_text())
        g = int(cur["gen"])
        if g == self.gen:
            return False
        old, self.gen = self.gen, g
        self._opp_model(g)
        if hasattr(self, "w_opp"):
            for i, o in enumerate(self.w_opp):
                if o == old:
                    self.w_opp[i] = g
                    self.worlds[i].start = True
            self._assign_rows()
        logger.info(f"latest generation {g}, pool {self._pool()}")
        return True

    def _assign_rows(self) -> None:
        dev = self.device
        lt = torch.tensor(self.l_team, device=dev)
        is_l = self.row_team == lt[self.row_world]
        self.L = torch.nonzero(is_l).flatten()
        self.P = torch.nonzero(~is_l).flatten()
        self.L_world, self.L_team = self.row_world[self.L], self.row_team[self.L]
        pw = self.row_world[self.P]
        self.P_groups = {g: self.P[torch.isin(pw, torch.tensor([i for i, o in enumerate(self.w_opp) if o == g], device=dev))]
                         for g in sorted(set(self.w_opp))}

    def _reassign(self) -> None:
        if not self.pending:
            return
        older = [g for g in self._pool() if g != self.gen]
        for i in sorted(self.pending):
            self.l_team[i] ^= 1
            self.w_opp[i] = self.gen if (not older or self.rng_opp.random() < self.cfg.latest_p) else self.rng_opp.choice(older)
            self.worlds[i].start = True
        self.pending.clear()
        self._assign_rows()

    def _write_candidate(self) -> None:
        d = self.league / f"cand_{self.it:05d}"
        d.mkdir(parents=True, exist_ok=True)
        torch.save(policy_payload(self.model, self.start_step + self.steps), d / "ckpt_base_latest.pt")
        with open(self.league / "candidates.jsonl", "a") as f:
            f.write(json.dumps({"iter": self.it, "dir": d.name, "t": int(time.time()), "vs_gen": self.gen}) + "\n")
        logger.info(f"candidate {d.name} written")

    # ---- collection ----
    @torch.no_grad()
    def collect(self) -> dict:
        cfg, dev, T = self.cfg, self.device, self.cfg.rollout
        m, anc = self.model, self.anchor
        m.eval()
        results = []
        for t in range(T):
            for w in self.worlds:
                w.commander.act(w.world)                                   # orders and respawn plans of both armies
            obs = torch.cat([w.obs for w in self.worlds])
            mask = torch.cat([w.mask for w in self.worlds])
            starts = torch.cat([torch.full((self.A,), 1.0 if w.start else 0.0, device=dev) for w in self.worlds])
            gs_w = torch.stack([w.world.global_state()[0] for w in self.worlds])
            act = torch.zeros(self.R, 4, dtype=torch.long, device=dev)
            act[:, 2] = 2
            L = self.L
            oL, mL, sL = obs[L], mask[L], starts[L]
            gL = gs_w[self.L_world, self.L_team]
            sl_state = (self.st_l[0][:, L], self.st_l[1][:, L])
            sa_state = (self.st_a[0][:, L], self.st_a[1][:, L])
            if t % cfg.bptt == 0:
                k = t // cfg.bptt
                keep = (1.0 - sL).view(1, -1, 1)
                self.h0[k], self.c0[k] = sl_state[0] * keep, sl_state[1] * keep
                self.ah0[k], self.ac0[k] = sa_state[0] * keep, sa_state[1] * keep
            with torch.autocast("cuda", dtype=torch.float16, enabled=self.use_amp):
                feats, tok = m.encoder(oL, return_tokens=True)
                logits, (h, c) = m.step_actor(feats, sl_state, sL, tok, oL)
                v = m.value_from(feats, gL)
                fa, ta_ = anc.encoder(oL, return_tokens=True)
                _, (ha, ca) = anc.step_actor(fa, sa_state, sL, ta_, oL)
            self.st_l[0][:, L], self.st_l[1][:, L] = h.to(self.st_l[0].dtype), c.to(self.st_l[1].dtype)
            self.st_a[0][:, L], self.st_a[1][:, L] = ha.to(self.st_a[0].dtype), ca.to(self.st_a[1].dtype)
            a_l, logp, _ = m.sample(logits.float(), False, mL)
            act[L] = a_l
            for g, Pg in self.P_groups.items():
                if Pg.numel() == 0:
                    continue
                opp = self._opp_model(g)
                oP = obs[Pg]
                so = (self.st_o[0][:, Pg], self.st_o[1][:, Pg])
                with torch.autocast("cuda", dtype=torch.float16, enabled=self.use_amp):
                    fp, tp = opp.encoder(oP, return_tokens=True)
                    lp, (ho, co) = opp.step_actor(fp, so, starts[Pg], tp, oP)
                self.st_o[0][:, Pg], self.st_o[1][:, Pg] = ho.to(self.st_o[0].dtype), co.to(self.st_o[1].dtype)
                a_p, _, _ = opp.sample(lp.float(), False, mask[Pg])
                act[Pg] = a_p
            self.b_obs[t].copy_(oL.to(torch.float16))
            self.b_gs[t].copy_(gL.to(torch.float16))
            self.b_mask[t], self.b_act[t], self.b_start[t] = mL, a_l, sL
            self.b_logp[t] = logp.float()
            self.b_val[t] = m.value_norm.denormalize(v.float().view(-1))
            self.b_alive[t] = oL[:, self.core_start + CORE_ALIVE_COL] > 0
            rew = torch.zeros(self.R, device=dev)
            done = torch.zeros(self.R, device=dev)
            for i, w in enumerate(self.worlds):
                sl = slice(i * self.A, (i + 1) * self.A)
                score = [float(s) for s in w.world.score[0]]
                res = w.step(act[sl])
                tau = self.c_tau[self.row_cls[sl]]
                rew[sl] = (1.0 - tau) * res["r_indiv"].float() + tau * res["info"]["r_team_agent"][0].float()
                if res["done"]:
                    done[sl] = 1.0
                    win, lt = int(res["info"]["winner"][0]), int(self.l_team[i])
                    results.append({"t": int(time.time()), "iter": self.it, "world": i, "opponent": f"gen{self.w_opp[i]}",
                                    "learner_team": lt, "winner": win, "learner_won": win == lt + 1, "score": score})
                    self.pending.add(i)
            self.b_rew[t] = rew[L]
            self.b_done[t] = done[L]
        self.steps += T * self.NL
        if results:
            with open(self.league / "train_matches.jsonl", "a") as f:
                f.writelines(json.dumps(r) + "\n" for r in results)
        return {"matches": results}

    def anchor_lambda(self) -> float:
        c = self.cfg
        x = min(1.0, self.it / max(1, c.anchor_iters))
        return c.anchor_coef + (c.anchor_floor - c.anchor_coef) * x

    # ---- update ----
    def update(self) -> dict:
        cfg, dev, m = self.cfg, self.device, self.model
        warm = self.it < cfg.critic_warmup_iters
        with torch.no_grad():
            obs = torch.cat([w.obs for w in self.worlds])[self.L]
            gs_w = torch.stack([w.world.global_state()[0] for w in self.worlds])
            with torch.autocast("cuda", dtype=torch.float16, enabled=self.use_amp):
                nv = m.value_from(m.encoder(obs), gs_w[self.L_world, self.L_team]).float().view(-1)
            adv, ret = compute_gae(self.b_rew, self.b_val, self.b_done, m.value_norm.denormalize(nv), cfg.gamma, cfg.gae_lambda)
            m.value_norm.update(ret)
            ret_n = m.value_norm.normalize(ret)
        lam = self.anchor_lambda()
        nc = cfg.rollout // cfg.bptt
        st = {k: [] for k in ("pg", "vf", "kl", "clip", "ent", "anc")}
        epochs_done, stop = 0, False
        m.train()
        for _ in range(cfg.epochs):
            perm = torch.randperm(self.NL, device=dev)
            pairs = [(c, j) for c in range(nc) for j in range(0, self.NL, cfg.minibatch_rows)]
            random.shuffle(pairs)
            for c, j in pairs:
                ts = slice(c * cfg.bptt, (c + 1) * cfg.bptt)
                col = perm[j:j + cfg.minibatch_rows]
                cc = col.cpu()
                o = self.b_obs[ts][:, cc].to(dev, non_blocking=True).float()
                g = self.b_gs[ts][:, cc].to(dev, non_blocking=True).float()
                mk, sts = self.b_mask[ts][:, col], self.b_start[ts][:, col]
                with torch.autocast("cuda", dtype=torch.float16, enabled=self.use_amp):
                    lg, v = seq_eval(m, o, g, sts, (self.h0[c][:, col], self.c0[c][:, col]), detach_feats=warm)
                    if not warm and lam > 0:
                        with torch.no_grad():
                            lga, _ = seq_eval(self.anchor, o, None, sts, (self.ah0[c][:, col], self.ac0[c][:, col]))
                lg = lg.float()
                mkf = mk.reshape(-1, self.M)
                logp, ent = m.evaluate_logits(lg, self.b_act[ts][:, col].reshape(-1, 4), mkf)
                logp = logp.view(cfg.bptt, -1)
                alive = self.b_alive[ts][:, col].float()
                n_alive = alive.sum().clamp(min=1.0)
                a = adv[ts][:, col]
                a = (a - a.mean()) / (a.std() + 1e-8)
                ratio = (logp - self.b_logp[ts][:, col]).exp()
                pg = (-torch.min(ratio * a, ratio.clamp(1 - cfg.clip, 1 + cfg.clip) * a) * alive).sum() / n_alive
                vf = 0.5 * (v.float() - ret_n[ts][:, col]).pow(2).mean()
                ent_m = (ent.view(cfg.bptt, -1, ent.shape[-1]).sum(-1) * alive).sum() / n_alive
                if warm:
                    loss, anc = cfg.vf_coef * vf, torch.zeros((), device=dev)
                else:
                    anc = (kl_heads(lga.float(), lg, mkf).view(cfg.bptt, -1) * alive).sum() / n_alive if lam > 0 \
                        else torch.zeros((), device=dev)
                    loss = pg + cfg.vf_coef * vf - cfg.ent_coef * ent_m + lam * anc
                self.opt.zero_grad(set_to_none=True)
                if self.scaler is not None:
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.opt)
                    nn.utils.clip_grad_norm_(m.parameters(), cfg.max_grad_norm)
                    self.scaler.step(self.opt)
                    self.scaler.update()
                else:
                    loss.backward()
                    nn.utils.clip_grad_norm_(m.parameters(), cfg.max_grad_norm)
                    self.opt.step()
                with torch.no_grad():
                    logr = logp - self.b_logp[ts][:, col]
                    st["kl"].append(float((((logr.exp() - 1) - logr) * alive).sum() / n_alive))
                    st["clip"].append(float((((ratio - 1).abs() > cfg.clip).float() * alive).sum() / n_alive))
                    st["pg"].append(float(pg)); st["vf"].append(float(vf)); st["ent"].append(float(ent_m)); st["anc"].append(float(anc))
            epochs_done += 1
            if not warm and sum(st["kl"][-len(pairs):]) / len(pairs) > cfg.target_kl:
                stop = True
                break
        m.eval()
        y = ret.flatten()
        ev = float(1.0 - (y - self.b_val.flatten()).var() / y.var().clamp(min=1e-8))
        mean = lambda xs: round(sum(xs) / max(1, len(xs)), 5)                     # noqa: E731
        return {"warmup": warm, "epochs_done": epochs_done, "kl_stop": stop, "approx_kl": mean(st["kl"]),
                "clipfrac": mean(st["clip"]), "pg_loss": mean(st["pg"]), "vf_loss": mean(st["vf"]), "entropy": mean(st["ent"]),
                "anchor_kl": mean(st["anc"]), "anchor_lambda": round(lam, 4), "explained_var": round(ev, 4),
                "reward_mean": round(float(self.b_rew.mean()), 5)}

    # ---- checkpoints ----
    def save(self, reason: str) -> None:
        d = self.models_dir
        step = self.start_step + self.steps
        payload = policy_payload(self.model, step)
        payload["sp_optimizer"] = {"opt": self.opt.state_dict(), "scaler": self.scaler.state_dict() if self.scaler else None}
        tmp = d / ".tmp_ckpt_base.pt"
        torch.save(payload, tmp)
        if (d / "ckpt_base_latest.pt").exists():
            (d / "ckpt_base_latest.pt").replace(d / "ckpt_base_prev.pt")
        tmp.replace(d / "ckpt_base_latest.pt")
        man = {"global_step": step, "iter": self.it, "saved": time.strftime("%Y-%m-%dT%H:%M:%S"),
               "sp": {"iter": self.it, "steps": self.steps, "hours": self.hours_done, "start_step": self.start_step,
                      "gen": self.gen, "reason": reason, "cfg": dataclasses.asdict(self.cfg)}}
        (d / MANIFEST).write_text(json.dumps(man, indent=1))
        logger.info(f"saved ({reason}): iteration {self.it}, generation {self.gen}")

    def run(self) -> str:
        cfg = self.cfg
        mpath = self.models_dir / "metrics.jsonl"
        seg_t = time.time()
        while True:
            self._load_opponent()
            self._reassign()
            t0 = time.perf_counter()
            col = self.collect()
            t1 = time.perf_counter()
            upd = self.update()
            t2 = time.perf_counter()
            self.it += 1
            self.hours_done += (time.time() - seg_t) / 3600.0
            seg_t = time.time()
            won = [r["learner_won"] for r in col["matches"]]
            row = {"t": int(time.time()), "iter": self.it, "gen": self.gen, "steps": self.steps, "hours": round(self.hours_done, 3),
                   "collect_s": round(t1 - t0, 1), "update_s": round(t2 - t1, 1), "matches": len(won), "learner_wins": sum(won), **upd}
            with open(mpath, "a") as fh:
                fh.write(json.dumps(row) + "\n")
            logger.info("iteration " + json.dumps(row))
            if self.it % cfg.gen_every == 0:
                self._write_candidate()
            if self.it % cfg.ckpt_every == 0:
                self.save("periodic")
            if cfg.max_iters and self.it >= cfg.max_iters:
                self.save("max_iters")
                return "max_iters"
            if self.hours_done >= cfg.hours:
                self.save("hour cap")
                return "hour cap"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="rush self-play by generations")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--init", help="a network to start from: ckpt file or folder (config.json + model.safetensors)")
    src.add_argument("--resume", help="a self-play run dir to continue")
    ap.add_argument("--run-dir", default="runs/selfplay")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--log-file", default="")
    d = SPConfig()
    for f_ in dataclasses.fields(SPConfig):
        flag, val = "--" + f_.name.replace("_", "-"), getattr(d, f_.name)
        if isinstance(val, bool):
            ap.add_argument(flag, type=int, default=int(val), choices=(0, 1), dest=f_.name)
        else:
            ap.add_argument(flag, type=type(val), default=val, dest=f_.name)
    args = ap.parse_args(argv)
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if args.log_file:
        handlers.append(logging.FileHandler(args.log_file))
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", handlers=handlers, force=True)
    cfg = SPConfig(**{f_.name: getattr(args, f_.name) for f_ in dataclasses.fields(SPConfig)})
    cfg.amp = bool(cfg.amp)
    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    state, opt_state = None, None
    if args.resume:
        run_dir = Path(args.resume)
        state = json.loads((run_dir / "models15" / MANIFEST).read_text()).get("sp")
        model, payload = load_net(run_dir / "models15" / "ckpt_base_latest.pt", args.device)
        opt_state = payload.get("sp_optimizer")
        anchor, _ = load_net(run_dir / "league" / "gen_000" / "ckpt_base_latest.pt", args.device)
    else:
        run_dir = Path(args.run_dir)
        model, payload = load_net(Path(args.init), args.device)
        anchor, _ = load_net(Path(args.init), args.device)
        state = {"start_step": int(payload.get("step", 0))}
    for q in anchor.parameters():
        q.requires_grad_(False)
    SelfPlayTrainer(cfg, model, anchor, run_dir, args.device, state, opt_state).run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
