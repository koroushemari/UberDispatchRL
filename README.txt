Final Project Submission Contents
================================

This folder is the cleaned submission package.

Files
-----
- rideshare_dispatch_rl.py
  Single main Python file containing:
  - the London dispatch environment
  - Q-learning, SARSA, DQN, and PPO
  - training, evaluation, and plotting code

- london_uber_layout.json
  Processed map/layout file used by the code.

- submission_report.tex
  Final report source in the NeurIPS template.

- references.bib
  Bibliography for the report.

- neurips_2026.sty
- checklist.tex
  Official NeurIPS template assets used by the report source.

- map.png
- results_summary.png
- showdown.png
  Figures used in the report.

- results_summary.txt
  Text summary of the three-seed experiment used in the report.

How to run the code
-------------------
From this folder:

python3 rideshare_dispatch_rl.py --episodes 900 --seeds 3 --eval-episodes 20 --output-dir final_results

This will generate:
- final_results/results_summary.png
- final_results/summary.txt
- final_results/summary.json

Note
----
The report source is ready for Overleaf or any local LaTeX installation that supports the NeurIPS style.
There is no local LaTeX compiler installed on this machine, so the package includes the complete source rather than a locally compiled PDF.
