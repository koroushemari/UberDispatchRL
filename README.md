# UberDispatchRL

Open [UberDispatchRL.ipynb](UberDispatchRL.ipynb) first and run the notebook from top to bottom.

This repo is intentionally minimal and course-project style:

- `UberDispatchRL.ipynb`: the full story-driven notebook version of the project
- `london_uber_layout.json`: the processed downtown London map used by the notebook

The notebook contains:

- separate code cells for utilities, agents, environment, heuristics, and experiments
- comments at the top of each code cell
- a few cells with displayed sample results already shown
- commented cells for the longer full experiment and ablation runs

Main conclusion from the project:

- DQN is the strongest learned policy on this benchmark
- a nearest-stop heuristic is stronger overall
- the benchmark is useful because it reveals when simple geometric rules still beat learned dispatch
