"""Show progress of all training runs (percentage + latest probe results).

Usage:
    python3 progress.py              # all runs
    python3 progress.py validation   # one run dir name under runs/
"""

import csv
import glob
import os
import sys

# expected total generations per run dir (default if unknown)
TOTALS = {"validation": 60,
          "neat_s0": 200, "neat_s1": 200, "neat_s2": 200,
          "fixed_s0": 200, "fixed_s1": 200, "fixed_s2": 200,
          "d2_s0": 150, "d2_s1": 150, "d2_s2": 150,
          "d4_s0": 200, "d4_s1": 200, "d4_s2": 200,
          "d6_s0": 150, "d6_s1": 150, "d6_s2": 150}

dirs = sys.argv[1:] or None
pattern = "runs/*/training_metrics.csv"
files = sorted(glob.glob(pattern)) + sorted(glob.glob("runs/*/*/training_metrics.csv"))

seen = set()
for path in files:
    run = os.path.dirname(path)
    if run in seen:
        continue
    seen.add(run)
    name = os.path.basename(run)
    if dirs and name not in dirs and run not in dirs:
        continue
    try:
        rows = list(csv.DictReader(open(path)))
    except OSError:
        continue
    if not rows:
        continue
    last = int(rows[-1]["gen"]) + 1
    total = TOTALS.get(name)
    pct = f"  =  {100 * last / total:.0f}%" if total else ""
    print(f"\n{run}:  generation {last}{'/' + str(total) if total else ''}{pct}")
    probes = [r for r in rows if r.get("elo")]
    for r in probes[-6:]:  # last few probes only
        def g(k):
            return r.get(k) or "  -  "
        print(f"  gen {r['gen']:>3}  Elo {r['elo']:>5}  "
              f"wr_d1 {g('wr_material-d1'):>5}  wr_d2 {g('wr_material-d2'):>5}  "
              f"wr_d4 {g('wr_material-d4'):>5}  conns {g('mean_conns'):>5}  "
              f"species {g('species'):>2}")
