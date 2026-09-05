"""Final head-to-head evaluation at a COMMON deep search depth.

For every condition x seed cell of an RQ, loads the best-EVER measured
genome (best_ever_genome.pkl, falling back to best_value_genome.pkl) and
measures its performance Elo vs the calibrated ladder at one shared depth
(default 8). This is the number that goes in the paper's results table:
for RQ2 it removes the confound that deeper-trained agents also probe at
deeper depth during training.

Writes paper/figures/<rq>_final_eval.csv:
  condition, seed, depth, games_per_rung, elo

Usage:
  python experiments/final_eval.py --rq rq2_amp --depth 8 --games 40
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai.arena import (append_raw_records, pair_opening_tasks, pair_tasks,
                      run_tasks, score_for_a)
from ai.ladder import LADDER, neat_spec
from ai.openings import get_test_openings
from ai.train_path_a import (PROBE_RUNGS, calibrate_ladder,
                             performance_rating)
from experiments.run_experiment import CONDITIONS

FIG_DIR = os.path.join("paper", "figures")


def ci95(vals):
    n = len(vals)
    if n < 2:
        return 0.0
    m = sum(vals) / n
    var = sum((v - m) ** 2 for v in vals) / (n - 1)
    return 1.96 * math.sqrt(var / n)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rq", choices=sorted(CONDITIONS), required=True)
    p.add_argument("--depth", type=int, default=8,
                   help="common evaluation depth for all conditions")
    p.add_argument("--games", type=int, default=40, help="games per rung")
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--overwrite", action="store_true",
                   help="explicitly replace an existing held-out evaluation")
    args = p.parse_args()

    run_dir = os.path.join("runs", args.rq)
    os.makedirs(FIG_DIR, exist_ok=True)
    raw_out = os.path.join(FIG_DIR, f"{args.rq}_final_eval_games.jsonl")
    out = os.path.join(FIG_DIR, f"{args.rq}_final_eval.csv")
    if not args.overwrite and (os.path.exists(raw_out) or os.path.exists(out)):
        raise SystemExit("Held-out evaluation already exists; use --overwrite explicitly")
    test_openings = get_test_openings(max(1, args.games // 2))
    all_eval_records = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        ladder_elo = calibrate_ladder(
            executor, os.path.join("runs", "evaluation", "ladder_calibration_v1.json"))
        rung_specs = dict(LADDER)

        rows = []
        for cond in CONDITIONS[args.rq]:
            if not os.path.isdir(run_dir):
                continue
            for cell in sorted(os.listdir(run_dir)):
                if not cell.startswith(f"{cond}_s"):
                    continue
                seed = int(cell.rsplit("_s", 1)[1])
                cell_dir = os.path.join(run_dir, cell)
                genome_path = os.path.join(cell_dir, "best_ever_genome.pkl")
                if not os.path.exists(genome_path):
                    genome_path = os.path.join(cell_dir,
                                               "best_value_genome.pkl")
                if not os.path.exists(genome_path):
                    print(f"  skip {cell}: no genome yet")
                    continue
                with open(genome_path, "rb") as f:
                    genome = pickle.load(f)
                config_path = CONDITIONS[args.rq][cond]["config"]
                spec = neat_spec(genome, config_path, args.depth)
                results = []
                for i, rname in enumerate(PROBE_RUNGS):
                    tasks = pair_opening_tasks(spec, rung_specs[rname], test_openings,
                                               base_seed=seed * 1000 + 31 * i,
                                               meta={"rung": rname, "cond": cond, "seed": seed, "rq": args.rq})
                    records = run_tasks(tasks, executor)
                    strict, _ = score_for_a(records)
                    results.append((rname, strict, len(tasks)))
                    all_eval_records.extend(records)
                elo = performance_rating(results, ladder_elo)
                print(f"  {cell}: Elo {elo:.0f} at depth {args.depth}")
                rows.append((cond, seed, args.depth, len(test_openings) * 2,
                             f"{elo:.0f}"))

    if args.overwrite and os.path.exists(raw_out):
        os.remove(raw_out)
    append_raw_records(all_eval_records, raw_out)
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["condition", "seed", "eval_depth", "games_per_rung",
                    "elo"])
        w.writerows(rows)
    print(f"\nWrote {out}")

    by_cond = {}
    for cond, _s, _d, _g, elo in rows:
        by_cond.setdefault(cond, []).append(float(elo))
    print(f"\n=== {args.rq} final Elo at depth {args.depth} "
          f"(mean +/- 95% CI) ===")
    for cond, vals in sorted(by_cond.items()):
        m = sum(vals) / len(vals)
        print(f"  {cond:8s}: {m:7.1f} +/- {ci95(vals):5.1f}   "
              f"(n={len(vals)}, seeds: {[round(v) for v in vals]})")


if __name__ == "__main__":
    main()
