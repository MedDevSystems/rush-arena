---
license: mit
library_name: pytorch
tags:
  - reinforcement-learning
  - multi-agent
  - self-play
  - game
---

# rush-soldier

A soldier network for **rush**, a 2D team shooter for 500 v 500 battles with control points
(code, world, viewer and training: https://github.com/MedDevSystems/rush-arena).

One network plays every soldier of an army, all 8 classes (the class is part of the observation). The army is
commanded by the heuristic army commander of the rush package (respawn points, which control point each squad of 8
takes); the network observes its order and fights.

## Use

```bash
pip install -e git+https://github.com/MedDevSystems/rush-arena#egg=rush-arena
rush-battle --blue hf:koskokos/rush-soldier --red random --map Front50 --view
rush-battle --blue hf:koskokos/rush-soldier --red hf:koskokos/rush-soldier --map Warfront500 --device cuda
```

```python
from rush.battle import play
play("hf:koskokos/rush-soldier", "random", map_name="Front50", out="battle.arena.bin.gz")
```

## Model

Per soldier and decision (every 4 frames at 60 fps): a transformer over entity tokens (nearest allies and enemies,
bullets, control points, last-seen enemies, wall grids), an LSTM, and a state-conditioned attention readout over
further tokens (up to 56 soldiers, the 32 most dangerous bullets, a 17 × 17 coarse map of the battle); four action
heads — move (9), fire (2), turn (7, fine and coarse), dash (2) — masked by the rules. 1.7 M parameters.
`config.json` carries the full observation layout; the weights are in `model.safetensors`.

## Training

Self-play by generations: the network plays a pool of its past generations and learns from the battle outcome with
PPO (per-soldier rewards for damage, kills, captures, score and the win, weighted per class), with a KL anchor to its
starting point that decays over training. A candidate becomes the next generation only when a sequential one-sided
paired t-test (5 %) over side-swapped battle pairs on the same seed shows it is stronger. This checkpoint is the
latest accepted generation. The same procedure ships with the code (`rush-selfplay`, `rush-gate`).

## Limitations

- A soldier, not a commander: the strategy (respawns, squad orders) comes from the heuristic army commander.
- Trained on the rules of rush as published with this model (world version 15, observation layout 4); other rules
  or layouts need retraining.
- Greedy actions are used for play; sampling is used in training.

## License

MIT.
