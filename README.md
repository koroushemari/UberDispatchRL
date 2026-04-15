# UberDispatchRL

Open [UberDispatchRL.ipynb](UberDispatchRL.ipynb) first.

This repo is intentionally minimal:

- `UberDispatchRL.ipynb`: the full story-driven notebook version of the project
- `london_uber_layout.json`: the processed downtown London map used by the notebook

The notebook contains:

- the RL agents
- the London dispatch environment
- heuristic baselines
- training and evaluation helpers
- the DQN ablation logic
- commented cells that reproduce the final experiments

Main conclusion from the project:

- DQN is the strongest learned policy on this benchmark
- a nearest-stop heuristic is stronger overall
- the benchmark is useful because it reveals when simple geometric rules still beat learned dispatch
