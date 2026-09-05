"""Aggregate multi-seed results into paper-grade figures (mean +/- 95% CI).

Reads runs/<rq>/<cond>_s<seed>/training_metrics.csv for every cell, aligns
probe Elo by generation, and writes:

- paper/figures/<rq>_elo.pdf/.png     Elo learning curves, mean +/- 95% CI
- paper/figures/<rq>_complexity.pdf   champion/population size (RQ1 bloat)
- paper/figures/<rq>_summary.csv      per-generation stats used in the plots

Usage:
  python experiments/aggregate.py --rq rq1_topo
  python experiments/aggregate.py --rq rq2_amp --smooth 3
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiments.run_experiment import CONDITIONS

FIG_DIR = os.path.join("paper", "figures")


def moving_average(xs, w):
    if w <= 1 or len(xs) < w:
        return xs[:]
    out = []
    for i in range(len(xs)):
        lo = max(0, i - w + 1)
        out.append(sum(xs[lo:i + 1]) / (i + 1 - lo))
    return out


def ci95(vals):
    n = len(vals)
    if n < 2:
        return 0.0
    m = sum(vals) / n
    var = sum((v - m) ** 2 for v in vals) / (n - 1)
    return 1.96 * math.sqrt(var / n)


def load_cells(rq):
    """-> {condition: {gen: [elo_per_seed]}, ...} plus complexity series."""
    elos = defaultdict(lambda: defaultdict(list))
    nodes = defaultdict(lambda: defaultdict(list))
    conns = defaultdict(lambda: defaultdict(list))
    run_dir = os.path.join("runs", rq)
    for cond in CONDITIONS[rq]:
        if not os.path.isdir(run_dir):
            continue
        for cell in sorted(os.listdir(run_dir)):
            if not cell.startswith(f"{cond}_s"):
                continue
            path = os.path.join(run_dir, cell, "training_metrics.csv")
            if not os.path.exists(path):
                continue
            with open(path) as f:
                for row in csv.DictReader(f):
                    gen = int(row["gen"])
                    if row.get("elo"):
                        elos[cond][gen].append(float(row["elo"]))
                    if row.get("nodes"):
                        nodes[cond][gen].append(float(row["nodes"]))
                        conns[cond][gen].append(float(row["conns"]))
    return elos, nodes, conns


def plot_series(ax, series_by_cond, smooth, ylabel, title):
    for cond, by_gen in series_by_cond.items():
        gens = sorted(by_gen)
        if not gens:
            continue
        means = [sum(by_gen[g]) / len(by_gen[g]) for g in gens]
        means = moving_average(means, smooth)
        cis = [ci95(by_gen[g]) for g in gens]
        line, = ax.plot(gens, means, label=f"{cond} (n={max(len(by_gen[g]) for g in gens)})")
        ax.fill_between(gens, [m - c for m, c in zip(means, cis)],
                        [m + c for m, c in zip(means, cis)],
                        color=line.get_color(), alpha=0.2)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.4)


def write_summary(rq, elos):
    os.makedirs(FIG_DIR, exist_ok=True)
    out = os.path.join(FIG_DIR, f"{rq}_summary.csv")
    conds = sorted(elos)
    gens = sorted({g for c in conds for g in elos[c]})
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["gen"] + [f"{c}_mean" for c in conds] +
                   [f"{c}_ci95" for c in conds] +
                   [f"{c}_n" for c in conds])
        for g in gens:
            row = [g]
            for c in conds:
                v = elos[c].get(g, [])
                row.append(f"{sum(v) / len(v):.1f}" if v else "")
            for c in conds:
                v = elos[c].get(g, [])
                row.append(f"{ci95(v):.1f}" if v else "")
            for c in conds:
                row.append(len(elos[c].get(g, [])))
            w.writerow(row)
    print(f"Wrote {out}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rq", choices=sorted(CONDITIONS), required=True)
    p.add_argument("--smooth", type=int, default=3,
                   help="moving-average window (in probe points)")
    args = p.parse_args()

    elos, nodes, conns = load_cells(args.rq)
    if not elos:
        sys.exit(f"No data under runs/{args.rq}/ yet.")

    os.makedirs(FIG_DIR, exist_ok=True)
    titles = {"rq1_topo": "RQ1 TOPO: evolving vs fixed topology (equal compute)",
              "rq2_amp": "RQ2 AMP: search depth during evolution"}

    fig, ax = plt.subplots(figsize=(7, 4.5))
    plot_series(ax, elos, args.smooth,
                "Performance Elo (random = 0)", titles.get(args.rq, args.rq))
    ax.set_xlabel("Generation")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        out = os.path.join(FIG_DIR, f"{args.rq}_elo.{ext}")
        fig.savefig(out, dpi=200)
        print(f"Wrote {out}")
    plt.close(fig)

    if nodes:
        fig, (a1, a2) = plt.subplots(2, 1, figsize=(7, 7), sharex=True)
        plot_series(a1, nodes, args.smooth, "Champion nodes",
                    "Network complexity (RQ1 bloat check)")
        plot_series(a2, conns, args.smooth, "Champion connections", "")
        a2.set_xlabel("Generation")
        fig.tight_layout()
        out = os.path.join(FIG_DIR, f"{args.rq}_complexity.pdf")
        fig.savefig(out, dpi=200)
        print(f"Wrote {out}")
        plt.close(fig)

    write_summary(args.rq, elos)


if __name__ == "__main__":
    main()
