"""Diagnose the reference ladder: play material-dN rungs against each other
and report W/D/L per pair plus fitted Elo. A healthy ladder must be
monotonic in search depth (d1 < d2 < d4 ...). Run after changing search or
before trusting any Elo number.

Usage: python experiments/diagnose_ladder.py --games 100
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai.arena import pair_tasks, run_tasks
from ai.elo import fit_elo

RUNGS = [("random", ("random",)), ("material-d1", ("material", 1)),
         ("material-d2", ("material", 2)), ("material-d4", ("material", 4)),
         ("material-d6", ("material", 6))]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--games", type=int, default=100, help="games per pair")
    p.add_argument("--opening-plies", type=int, default=8)
    p.add_argument("--seed", type=int, default=7)
    args = p.parse_args()

    tasks = []
    for i, (na, sa) in enumerate(RUNGS):
        for j, (nb, sb) in enumerate(RUNGS):
            if i < j:
                tasks += pair_tasks(sa, sb, args.games,
                                    base_seed=args.seed + 97 * i + j,
                                    opening_plies=args.opening_plies,
                                    meta={"pair": (na, nb)})
    with ProcessPoolExecutor() as ex:
        records = run_tasks(tasks, ex, chunksize=2)

    stats = defaultdict(lambda: [0, 0, 0])  # pair -> [W_a, D, L_a]
    plies = defaultdict(list)
    games = []
    for rec in records:
        na, nb = rec["meta"]["pair"]
        a_won = rec["winner"] == 1
        draw = rec["winner"] == 0
        if not rec["meta"]["a_is_p1"]:
            a_won = rec["winner"] == 2
        s = stats[(na, nb)]
        if draw:
            s[1] += 1
        elif a_won:
            s[0] += 1
        else:
            s[2] += 1
        plies[(na, nb)].append(rec["plies"])
        s_a = rec["p1_strict"] if rec["meta"]["a_is_p1"] else 1.0 - rec["p1_strict"]
        games.append((na, nb, s_a))

    print(f"\n{args.games} games/pair, opening_plies={args.opening_plies}")
    print(f"{'pair':<24} {'W(A)':>5} {'D':>5} {'L(A)':>5} {'scoreA':>7} {'plies':>6}")
    for (na, nb), (w, d, l) in stats.items():
        n = w + d + l
        print(f"{na + ' vs ' + nb:<24} {w:>5} {d:>5} {l:>5} "
              f"{(w + 0.5 * d) / n:>7.3f} {sum(plies[(na, nb)]) / n:>6.1f}")

    ratings = fit_elo(games, anchor="random", anchor_rating=0.0)
    print("\nFitted Elo (random=0):")
    for name, _ in RUNGS:
        print(f"  {name:<12} {ratings.get(name, float('nan')):>7.0f}")
    vals = [ratings.get(n) for n in ("material-d1", "material-d2",
                                     "material-d4")]
    vals = [v for v in vals if v is not None]
    print("\nMonotonic d1<d2<d4 (probe ladder):",
          "YES" if vals == sorted(vals) else "NO <-- problem")
    print("Note: d4 and d6 are expected to be statistically tied (small",
          "depth returns beyond d4 with a simple eval); the probe ladder",
          "only uses d1/d2/d4.")


if __name__ == "__main__":
    main()
