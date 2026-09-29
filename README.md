# rush — 500 v 500 battles and the network that fights them

*[Русская версия](README.ru.md)*

**rush** is a 2D top-down team shooter built for very large battles: two armies of up to 500 soldiers fight over
~100 control points on a 21 000 × 13 000 px map. Soldiers come in 8 classes (tank, sniper, scout, …) with their own
speed, range, rate of fire, health and magazine; ammunition is refilled only on your own uncontested point; the
dead respawn on control points your side holds. The whole world is a batched PyTorch simulation, so it runs on a GPU.

This repository contains the game world, the army commander, the trained **soldier network**
([koskokos/rush-soldier](https://huggingface.co/koskokos/rush-soldier) on the Hugging Face Hub), self-play training
by generations, a tool to play and record a battle, and a browser viewer for the recordings.

## 1. Generate a battle

```bash
pip install -e .                       # Python ≥ 3.10, PyTorch ≥ 2.1
rush-battle --blue hf:koskokos/rush-soldier --red random --map Front50 --out battle.arena.bin.gz
```

`Front50` is a 50 v 50 map: a match takes minutes even on a CPU. The flagship map is `Warfront500` (500 v 500,
~7 000 decisions per match, about an hour on one GPU):

```bash
rush-battle --blue hf:koskokos/rush-soldier --red hf:koskokos/rush-soldier --map Warfront500 --device cuda --out warfront.arena.bin.gz
```

Players: `hf:<repo>[/<variant>]` (a network from the Hub), `net:<folder>` (a local `config.json` +
`model.safetensors`, or a training checkpoint folder), `random` (an untrained network); add `#Name` to set the name
shown in the viewer. Maps: `Warfront500`, `Front150`, `Crossing150`, `Front50`, or a generated one — `steppe:11`,
`urban:12`, `forest:13`, `delta:14`, `plateau:15` (`<theme>:<seed>[:<team size>]`). `--max-decisions N` stops early.

From Python:

```python
from rush.battle import play
result = play("hf:koskokos/rush-soldier", "random", map_name="Front50", out="battle.arena.bin.gz")
print(result["score"], result["winner"])
```

## 2. Watch the recording

```bash
rush-view battle.arena.bin.gz          # or: rush-battle ... --view
```

The viewer opens in the browser. Zoom with the wheel, pinch or the ± buttons; «Режиссёр» (director) follows the
hottest fight; «Классы» shows the class legend. The interface is in Russian for now.

## 3. Train it further: self-play by generations

```bash
# start from the published soldier; generation 0 = this network, it is also the KL anchor
rush-selfplay --init hf:koskokos/rush-soldier --run-dir runs/sp --device cuda \
    --maps Front150 --worlds 4 --rollout 500 --gen-every 5
```

Every iteration plays `--rollout` decisions in `--worlds` parallel battles — the learner on one side, a frozen
generation (the latest, sometimes an older one) on the other, sides swapped after every match — then updates the
network with PPO on the outcome (damage, kills, captures, score, win). Every `--gen-every` iterations a candidate is
written to `runs/sp/league/cand_XXXXX/`. In a second terminal, gate it:

```bash
rush-gate --run-dir runs/sp --map Front150 --device cuda      # pairs of side-swapped battles + a paired t-test
```

If the candidate is significantly stronger it becomes the next generation (`league/gen_XXX`, `league/current.json`)
and the running trainer starts meeting it. Metrics go to `runs/sp/models15/metrics.jsonl`; resume with
`rush-selfplay --run-dir runs/sp --resume`. The defaults (six 500 v 500 worlds, 1000-decision rollouts) need a large
GPU and ~35 GB of RAM; `Front150` with 4 worlds fits a single consumer GPU. Watch any candidate:

```bash
rush-battle --blue "net:runs/sp/league/cand_00005#Candidate" --red "net:runs/sp/league/gen_000#Generation 0" --map Front150 --view
```

## The soldier network

One network plays every soldier of an army, all 8 classes (the class is part of its observation). Per soldier and
decision: a transformer over entity tokens (nearest allies and enemies, bullets, control points, last-seen enemies,
wall grids), an LSTM, and a state-conditioned attention readout over further tokens (up to 56 soldiers, the 32 most
dangerous bullets, a 17 × 17 coarse map of the battle). Actions: 9 moves, fire, 7 turns, dash.

The army is commanded by `rush/army_commander.py` — which point each squad of 8 is ordered to take, who goes back
to refill, where the dead respawn. The network observes its order and fights; it is a soldier, not a commander.

## Repository

```
rush/world1*.py, world_big*.py   the world (each version extends the previous one; world15 is the current rules)
rush/maps*.py                    hand-made and generated maps
rush/model1*.py                  the network
rush/army_commander.py           the army commander
rush/battle.py, net_player.py    playing and recording a battle; rush/hub.py — networks from the Hub
rush/selfplay.py, gate.py        self-play training and the generation gate
rush/viewer/                     the browser viewer
```

## License

MIT — code and model weights.
