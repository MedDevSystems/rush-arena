# FILE: rush/army_commander.py
# region MODULE_CONTRACT [DOMAIN(8): GameAI; CONCEPT(8): ArmyCommander; TECH(6): torch]
## @modulecontract
## @purpose The heuristic commander of an army (both sides or one): which control point each squad of 8 soldiers is
## ordered to take, who goes back to an own point to refill, and on which own point each dead soldier respawns. It
## writes world.order_cp (the order every soldier observes) and world.spawn_plan (used by the respawn rule).
## @scope Commands only — it never moves or fires a soldier.
## @invariants deterministic; squads are fixed by rank within the team; a full reassignment every reassign_every
## decisions, plus at once for squads without a target or sitting on a safe own point.
# endregion MODULE_CONTRACT

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch

SAFE_OWN_LAST = 1e6            # px added to a safe own point's cost: it is chosen only when no other point is left


@dataclass(frozen=True)
class CommanderRules:
    squad_size: int = 8
    reassign_every: int = 20
    own_cost: float = 5000.0          # px: an own point is a worse objective than a neutral / enemy one ...
    enemy_bonus: float = 1500.0       # ... an enemy point a better one ...
    contested_bonus: float = 1200.0   # ... and an own point with enemies near it needs defending
    front_w: float = 0.6              # cost per px of distance from the map's centre line
    crowd_w: float = 1500.0           # cost per squad already sent to a point
    low_ammo: float = 0.30            # below this share of the magazine a soldier is sent to refill ...
    refill_until: float = 0.95        # ... until this share
    spawn_plan: bool = True
    spawn_L_front: float = 1200.0     # px: how fast a point's appeal falls with its path distance to the front,
    spawn_L_mid: float = 2000.0       # for front-line classes (high max hp), others, and long-range classes
    spawn_L_rear: float = 4000.0
    spawn_soft_cap: float = 8.0       # appeal / (1 + already planned / this): spreads a wave over several points
    spawn_chunk: int = 16             # dead soldiers placed per greedy round (in rank order)
    spawn_base_appeal: float = -0.5   # < 0: any eligible point beats the base (the base is start-only)


class ArmyCommander:
    def __init__(self, world: Any, rules: CommanderRules = CommanderRules()) -> None:
        self.rules = rules
        self.device = world.pos.device
        self.n, self.A = int(world.pos.shape[0]), int(world.pos.shape[1])
        n, a, dev = self.n, self.A, self.device
        self.team = world._agent_team.to(dev)
        ids = torch.arange(a, device=dev)
        t_size = int((self.team == 0).sum())
        self.rank = torch.where(self.team == 0, ids, ids - t_size)                       # index within the team
        self.ts = float(getattr(world, "tile_size", 30.0))
        self.target = torch.full((n, a), -1, dtype=torch.long, device=dev)
        self.refill_mode = torch.zeros(n, a, dtype=torch.bool, device=dev)
        self.t = 0
        self.plan_team: int | None = None                                                # None = command both teams
        self._setup(world)

    # region FUNC_setup
    def _setup(self, world: Any) -> None:
        maps = world.map_idx.to(self.device)
        self.maps = maps
        self.cp_xy = world.map_cp_xy[maps].to(self.device)                               # (n, K, 2)
        cp_r = world.map_cp_r[maps].to(self.device).float()
        self.cp_r = cp_r.view(-1, 1) if cp_r.dim() == 1 else cp_r.view(cp_r.shape[0], -1)[:, :1]
        sp, sn = world.map_spawn[maps].to(self.device), world.map_spawn_n[maps].to(self.device)
        idx = torch.arange(sp.shape[2], device=self.device).view(1, 1, -1)
        valid = (idx < sn.unsqueeze(-1)).float().unsqueeze(-1)
        centre = (sp * valid).sum(2) / valid.sum(2).clamp(min=1.0)
        axis = centre[:, 1] - centre[:, 0]
        axis = axis / axis.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        mid = (centre[:, 0] + centre[:, 1]) / 2
        self.front = ((self.cp_xy - mid.unsqueeze(1)) * axis.unsqueeze(1)).sum(-1).abs()  # distance from the centre line
        K = self.cp_xy.shape[1]
        rc_r = (self.cp_xy[..., 1] / self.ts).long().clamp(0, world.flow_dist.shape[2] - 1)
        rc_c = (self.cp_xy[..., 0] / self.ts).long().clamp(0, world.flow_dist.shape[3] - 1)
        kk = torch.arange(K, device=self.device).view(1, K, 1).expand(maps.shape[0], K, K)
        mm = maps.view(-1, 1, 1).expand_as(kk)
        d = world.flow_dist[mm, kk, rc_r.unsqueeze(1).expand_as(kk), rc_c.unsqueeze(1).expand_as(kk)].to(self.device)
        self.cp_path = torch.where(torch.isfinite(d), d, torch.full_like(d, 1e7)).transpose(1, 2)   # [k, j]: from k to j
    # endregion FUNC_setup

    # region FUNC_act
    ## @purpose One decision of command: refill flags, (re)assign squad objectives, plan respawns, write the orders.
    ## @io world -> target point per soldier (n, A) long (also written to world.order_cp for the commanded team)
    def act(self, world: Any) -> torch.Tensor:
        r, n, a, dev = self.rules, self.n, self.A, self.device
        self.t += 1
        pos, alive = world.pos, world.alive
        ammo, cap = world.ammo, world.ammo_cap.clamp(min=1e-6)
        self.refill_mode = alive & ((self.refill_mode & (ammo < r.refill_until * cap)) | (ammo < r.low_ammo * cap))
        row = (pos[..., 1] / self.ts).long().clamp(0, world.flow_dist.shape[2] - 1)
        col = (pos[..., 0] / self.ts).long().clamp(0, world.flow_dist.shape[3] - 1)
        K = self.cp_xy.shape[1]
        kk = torch.arange(K, device=dev).view(1, 1, K)
        mm = self.maps.view(n, 1, 1).expand(n, a, K)
        pathd = world.flow_dist[mm, kk.expand(n, a, K), row.unsqueeze(2).expand(n, a, K), col.unsqueeze(2).expand(n, a, K)]
        pathd = torch.where(torch.isfinite(pathd), pathd, torch.full_like(pathd, 1e7))
        owner_cp = world.cp_owner.unsqueeze(1)
        mine = owner_cp == (self.team + 1).view(1, a, 1)
        theirs = owner_cp == (2 - self.team).view(1, a, 1)
        dcp = (pos.unsqueeze(2) - self.cp_xy.unsqueeze(1)).pow(2).sum(-1).sqrt()
        near_cp = (dcp < self.cp_r.view(n, 1, 1) * 3.0) & alive.unsqueeze(-1)
        blue_near, red_near = near_cp[:, self.team == 0].any(1), near_cp[:, self.team == 1].any(1)   # (n, K)
        enemy_near = torch.where((self.team == 0).view(1, a, 1), red_near.unsqueeze(1), blue_near.unsqueeze(1))
        tk = self.target.clamp(min=0).unsqueeze(2)
        cur_own = mine.gather(2, tk).squeeze(2) & ~enemy_near.gather(2, tk).squeeze(2)
        if self.t % r.reassign_every == 1 or bool((self.target < 0).any()) or bool((cur_own & alive).any()):
            self._assign(pathd, mine, theirs, enemy_near, alive,
                         force=(self.t % r.reassign_every == 1) | (self.target < 0) | cur_own)
        safe_home = mine & ~enemy_near
        home = torch.where(safe_home, pathd, torch.where(mine, pathd + 3000.0, torch.full_like(pathd, float("inf"))))
        home_d, home_k = home.min(2)
        go_home = self.refill_mode & torch.isfinite(home_d)
        target = torch.where(go_home, home_k, self.target.clamp(min=0))
        if r.spawn_plan and getattr(world, "_cp_spawn_ok", False):
            self._plan_spawns(world)
        if hasattr(world, "order_cp"):
            mine_t = alive if self.plan_team is None else alive & (self.team == self.plan_team).view(1, a)
            world.order_cp = torch.where(mine_t, target, torch.where(alive, world.order_cp, torch.full_like(target, -1)))
        return target
    # endregion FUNC_act

    # region FUNC_assign
    ## @purpose Squads (by rank) pick objectives greedily, cheapest squad first; each pick raises that point's cost
    ## for the next squads (crowding). Safe own points come last: holding them is not an objective.
    def _assign(self, pathd: torch.Tensor, mine: torch.Tensor, theirs: torch.Tensor, enemy_near: torch.Tensor,
                alive: torch.Tensor, force: torch.Tensor) -> None:
        r, n = self.rules, self.n
        K = pathd.shape[2]
        base = (pathd + mine.float() * r.own_cost - (mine & enemy_near).float() * (r.own_cost + r.contested_bonus)
                - theirs.float() * r.enemy_bonus + self.front.unsqueeze(1) * r.front_w)       # (n, A, K)
        safe_own = mine & ~enemy_near
        base = torch.where(safe_own, base + SAFE_OWN_LAST, base)
        for team in (0, 1):
            members = (self.team == team).nonzero().squeeze(-1)
            sq = (self.rank[members] // max(1, r.squad_size)).long()
            S = int(sq.max()) + 1
            w = alive[:, members].float()
            sums = torch.zeros(n, S, K, device=pathd.device).index_add_(1, sq, base[:, members] * w.unsqueeze(-1))
            cnt = torch.zeros(n, S, device=pathd.device).index_add_(1, sq, w).clamp(min=1.0)
            sq_cost = sums / cnt.unsqueeze(-1)
            order = sq_cost.min(-1).values.argsort(-1)
            load = torch.zeros(n, K, device=pathd.device)
            pick = torch.zeros(n, S, dtype=torch.long, device=pathd.device)
            ar_n = torch.arange(n, device=pathd.device)
            for k in range(S):
                s = order[:, k]
                c = sq_cost[ar_n, s] + load * r.crowd_w
                choice = c.argmin(-1)
                pick[ar_n, s] = choice
                load[ar_n, choice] += 1.0
            new_t = pick[:, sq]
            cur = self.target[:, members]
            self.target[:, members] = torch.where(force[:, members], new_t, cur)
    # endregion FUNC_assign

    # region FUNC_plan_spawns
    ## @purpose Every waiting soldier of the commanded team gets an eligible own point: appeal = exp(-front path / L)
    ## (L by class: front-line, others, long-range) × a balance term (enemies near it); placed in rank order,
    ## spawn_chunk at a time, each round dividing a point's appeal by (1 + already planned / soft cap).
    def _plan_spawns(self, world: Any) -> None:
        r = self.rules
        n, a = self.n, self.A
        dead = world.waiting if hasattr(world, "waiting") else ~world.alive
        teams = (0, 1) if self.plan_team is None else (int(self.plan_team),)
        allowed = world.cp_spawn_allowed()
        owner = world.cp_owner
        dcp = (world.pos.unsqueeze(2) - self.cp_xy.unsqueeze(1)).pow(2).sum(-1).sqrt()
        near3 = (dcp < 3.0 * self.cp_r.view(n, 1, 1)) & world.alive.unsqueeze(-1)
        br, mhp = world.traits[..., 1], world.traits[..., 5]
        L = torch.where(mhp >= 1.2, r.spawn_L_front, torch.where(br >= 1.15, r.spawn_L_rear, r.spawn_L_mid))
        plan = world.spawn_plan.clone()
        for t in teams:
            on_t = (self.team == t)
            plan[:, on_t] = -1
            mine = owner == (t + 1)
            path = torch.where((~mine).unsqueeze(1), self.cp_path, torch.full_like(self.cp_path, float("inf")))
            front = path.min(-1).values
            front = torch.where(torch.isfinite(front), front, torch.zeros_like(front))
            en = (near3 & (self.team != t).view(1, a, 1)).sum(1).float()
            al = (near3 & on_t.view(1, a, 1)).sum(1).float()
            bal = 1.0 + (en - al).clamp(min=0.0) / 5.0
            elig = allowed[:, t]
            for w in range(n):
                idx = torch.nonzero(dead[w] & on_t).flatten()
                if idx.numel() == 0 or not bool(elig[w].any()):
                    continue
                app = torch.exp(-front[w].unsqueeze(0) / L[w, idx].unsqueeze(-1)) * bal[w].unsqueeze(0)
                app = torch.where(elig[w].unsqueeze(0), app, torch.full_like(app, -1.0))
                load = torch.zeros(app.shape[1], device=app.device)
                for c0 in range(0, int(idx.numel()), max(1, r.spawn_chunk)):
                    sub = slice(c0, c0 + r.spawn_chunk)
                    best, k = (app[sub] / (1.0 + load / r.spawn_soft_cap).unsqueeze(0)).max(-1)
                    use = best > r.spawn_base_appeal
                    plan[w, idx[sub][use]] = k[use]
                    load = load + torch.bincount(k[use], minlength=app.shape[1]).float()
        world.spawn_plan = plan
    # endregion FUNC_plan_spawns
