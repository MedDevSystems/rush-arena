"""A soldier network playing one army: every soldier acts with the network (greedy), while the army commander
(rush.army_commander) decides where the dead respawn and which control point each squad is ordered to take (the
network observes its order). The network is a soldier, not a commander.

The network runs one batch per soldier class, as in training: at the very start of a battle some actions are exact
ties, and a different batch shape can break a tie the other way."""
from __future__ import annotations

from typing import Any

import torch

from rush.army_commander import ArmyCommander
from rush.forks12 import FORKS, trait_vector


class NetPlayer:
    def __init__(self, world: Any, policy: Any, deterministic: bool = True) -> None:
        n, a = int(world.pos.shape[0]), int(world.pos.shape[1])
        dev = world.pos.device
        self.policy, self.deterministic, self.device = policy, deterministic, dev
        tv = torch.tensor([trait_vector(f) for f in FORKS], dtype=torch.float32, device=dev)          # (8, traits)
        cls = (world.traits.unsqueeze(-2) - tv.view(1, 1, len(FORKS), -1)).abs().sum(-1).argmin(-1)[0]  # (A,)
        self.groups = []
        for k in range(len(FORKS)):
            slots = torch.nonzero(cls == k).flatten()
            if slots.numel():
                self.groups.append([slots, policy.initial_state(n * int(slots.numel()))])
        self.first = True
        self.commander = ArmyCommander(world)
        self.plan_team: int | None = None

    def act(self, world: Any) -> tuple[torch.Tensor, dict]:
        n, a = int(world.pos.shape[0]), int(world.pos.shape[1])
        obs, mask = world.observe(), world.action_mask()
        acts = torch.zeros(n, a, 4, dtype=torch.long, device=self.device)
        acts[..., 2] = 2
        for g in self.groups:
            slots, state = g
            o = obs[:, slots].reshape(-1, obs.shape[-1])
            mk = mask[:, slots].reshape(-1, mask.shape[-1])
            starts = torch.full((o.shape[0],), 1.0 if self.first else 0.0, device=self.device)
            with torch.no_grad():
                act, g[1] = self.policy.act(o, state, starts, self.deterministic, mask=mk)
            acts[:, slots] = act.long().view(n, len(slots), 4)
        self.first = False
        acts = torch.where(world.alive.unsqueeze(-1), acts, torch.tensor([0, 0, 2, 0], device=self.device).view(1, 1, 4))
        self.commander.plan_team = self.plan_team
        with torch.no_grad():
            self.commander.act(world)
        return acts, {}
