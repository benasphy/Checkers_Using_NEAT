"""Plot training progress from runs/<name>/training_metrics.csv:
fitness curves plus the periodic ladder-Elo probe of the generation best.

Usage: python plot_fitness.py [runs/path_a/training_metrics.csv] [out.png]
"""

import csv
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

path = sys.argv[1] if len(sys.argv) > 1 else "runs/path_a/training_metrics.csv"
out = sys.argv[2] if len(sys.argv) > 2 else "fitness_plot.png"

gens, best, mean = [], [], []
elo_gens, elos = [], []
with open(path) as f:
    for row in csv.DictReader(f):
        gens.append(int(row["gen"]))
        best.append(float(row["best_fitness"]))
        mean.append(float(row["mean_fitness"]))
        if row.get("elo"):
            elo_gens.append(int(row["gen"]))
            elos.append(float(row["elo"]))

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
ax1.plot(gens, best, label="Best fitness (mean points/game)")
ax1.plot(gens, mean, label="Population mean fitness")
ax1.axhline(0.5, color="gray", ls="--", lw=0.8, label="50% (peer parity)")
ax1.set_ylabel("Tournament score")
ax1.legend()
ax1.grid(True, alpha=0.4)

if elos:
    ax2.plot(elo_gens, elos, "o-", color="tab:red", label="Ladder Elo (random = 0)")
    ax2.legend()
ax2.set_xlabel("Generation")
ax2.set_ylabel("Performance Elo")
ax2.grid(True, alpha=0.4)

plt.tight_layout()
plt.savefig(out, dpi=140)
print(f"Saved {out}")
