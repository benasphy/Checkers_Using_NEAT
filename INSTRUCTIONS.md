# INSTRUCTIONS — read this first

Everything you need to run the project, survive power outages, and find your
results. Nothing else is required reading.

---

## 0. What was broken and what is now fixed (Aug 12, 2026)

| Problem (your 17 flags) | Root cause | Fix |
|---|---|---|
| Ladder rated d4 BELOW d1/d2; Elo numbers meaningless | Pure-material eval has **game-tree pathology**: deeper search plays *weaker* (d1 beat d6 77–5). Verified: not a search bug. | Benchmark eval is now material + small positional terms (`benchmark_eval` in `ai/search.py`). Ladder is monotonic: d1<d2<d4. Diagnose anytime: `python3 experiments/diagnose_ladder.py` |
| Fitness 0.9–1.0 but Elo ~62; "best-ever Elo 248" was a gen-10 fluke | Champion = winner of a ~10-game **lottery** | Top-8 **playoff** each generation picks the real champion (logged in the `playoff` column) |
| Species explosion (23 species / 92 genomes), connections collapsing 36→4 | threshold 2.0 too fine + crossover shrinkage under noisy fitness | `compatibility_threshold = 2.5`, delete rates slightly below add rates, 4 HOF games/genome (anti-cycling anchor) |
| "Where are the results?" | They were never aggregated | See section 4 below |
| Old Elo comparisons invalid | Old calibration caches | Cache version bumped (`_v2_g300`); old caches ignored automatically |

Your old run is archived at `runs/pilot_v2_rq1_topo/` (cite it in the paper
as the pilot study that motivated these fixes). Nothing was deleted.

---

## 1. The validation run (running NOW in the background)

A 60-generation test run proves the fixes work before you spend ~50h.
It was launched with `nohup`, so it keeps running even if you close the
terminal or the chat session ends.

**Check if it's still running:**
```bash
ps aux | grep "main.py train" | grep -v grep
tail -5 runs/validation.log
```

**When it finishes (~1–1.5h), check the result:**
```bash
python3 - <<'EOF'
import csv
rows = [r for r in csv.DictReader(open("runs/validation/training_metrics.csv")) if r["elo"]]
for r in rows:
    print(f"gen {r['gen']:>3}  Elo {r['elo']:>5}  wr_d1 {r['wr_material-d1']:>5}  "
          f"wr_d2 {r['wr_material-d2']:>5}  mean_conns {r['mean_conns']:>5}  species {r['species']:>2}")
EOF
```

**PASS criteria** (compare first vs last probe):
- Elo is clearly RISING (e.g. from negative to positive), and
- `wr_material-d1` rises well above 0 (the old run stayed at 0.000 forever), and
- `mean_conns` does NOT collapse toward ~4 (stays roughly >15), and
- species count stays reasonable (roughly 3–12, not 23).

**If PASS → go to section 2.**
**If FAIL → do NOT start the big runs.** Open a new chat, share this file and
`runs/validation/training_metrics.csv`, and ask to iterate on the fix.

---

## 2. The real experiment: RQ1 only (Plan A — one cell at a time, ~4h each)

> Decision (Aug 12, after the 60-gen validation run): **RQ2 is postponed.**
> The validation run showed learning is real but slow; RQ2 (depth scaling)
> only makes sense if RQ1 produces healthy curves. RQ2 commands are kept in
> section 2b for later — skip them for now.
>
> Plan A scheduling: run **ONE cell at a time with all 16 cores** (~4h per
> cell = one night). Two cells in parallel was 4x slower per generation
> (thermal/memory contention) — same total compute, worse latency.

### FIRST: stop the old parallel runs and clear their 2 generations

```bash
# Ctrl+C the running jobs (or: kill %1 %2), then:
rm -rf runs/rq1_topo      # they only reached gen ~1-2 with old settings
```

### The protocol — 6 cells, one per night, in this order

```bash
python3 experiments/run_experiment.py --rq rq1_topo --only neat_s0  --generations 200
```
then (next night):
```bash
python3 experiments/run_experiment.py --rq rq1_topo --only fixed_s0 --generations 200
```
then `neat_s1`, `fixed_s1`, `neat_s2`, `fixed_s2` — same command, just
change the name. ~4h per cell, 6 nights total.

**Scientific standard of this protocol** (what a reviewer would ask):
- 2 conditions × 3 seeds, identical compute (generations × population ×
  games), everything seeded and resumable
- the two conditions differ ONLY in structural mutation rates (the
  treatment); all other settings shared
- strength measured against ONE fixed, calibrated, project-wide ladder
  (never used for fitness); best genome chosen by measured Elo, not fitness
- n=3 seeds gives wide confidence intervals — reported honestly as a
  limitation in the paper

**Safety valve:** check `python3 progress.py` as `neat_s0` passes gen ~100.
If its Elo has flattened below ~150, STOP and reassess before running the
remaining cells — don't spend nights on a flat curve.

### 2b. (LATER, only if RQ1 looks good) RQ2 — AMP: training depth 2/4/6

```bash
# Reuse the RQ1 neat runs as the d4 condition (identical config & depth):
mkdir -p runs/rq2_amp
for s in 0 1 2; do cp -r runs/rq1_topo/neat_s$s runs/rq2_amp/d4_s$s; done

# Then run d2 (fast, ~1.5h) and d6 (slow, ~10h) one cell at a time, 150 gens:
python3 experiments/run_experiment.py --rq rq2_amp --only d2_s0 --generations 150
python3 experiments/run_experiment.py --rq rq2_amp --only d6_s0 --generations 150
# ... then d2_s1, d6_s1, d2_s2, d6_s2 (same command, change the name)
```

(For RQ2, only generations 0–150 of the d4 runs are compared — equal compute.)

**Suggested schedule:** one pair per night. RQ1 = ~3 nights.

---

## 3. Power outage / stop / resume — the only rule you need

> **If anything interrupts a run (power cut, Ctrl+C, closed laptop):
> just rerun the exact same command. It resumes automatically from the
> newest checkpoint. You lose at most ~5 generations. Nothing else to do.**

This works for every command in section 2. Checkpoints are written
atomically every 5 generations in `runs/<rq>/<cell>/neat-checkpoint-N`.

To check progress of any cell:
```bash
tail -3 runs/rq1_topo/neat_s0/training_metrics.csv
```

---

## 4. Where the results are (and how to make paper figures)

| What | Where |
|---|---|
| Per-generation data for one run | `runs/<rq>/<cell>/training_metrics.csv` (gen, fitness, Elo, win rates vs each benchmark, W/D/L, topology, species) |
| Best genome of a run | `runs/<rq>/<cell>/best_ever_genome.pkl` (best by *measured* Elo, never overwritten by luck) |
| Paper figures (mean ± 95% CI) | created by you → `paper/figures/` |
| Headline results table | created by you → `paper/figures/*_final_eval.csv` |

**After all cells of an RQ finish:**
```bash
python3 experiments/aggregate.py --rq rq1_topo     # figures + summary CSV
python3 experiments/aggregate.py --rq rq2_amp
python3 experiments/final_eval.py --rq rq1_topo --depth 8 --games 40
python3 experiments/final_eval.py --rq rq2_amp --depth 8 --games 40
```

**Watch a run live (fitness / Elo / complexity / benchmark curves):**
```bash
python3 plot_fitness.py runs/rq1_topo/neat_s0/training_metrics.csv
```

---

## 5. Paper checklist

- [x] Validation run passes (section 1) — PASSED Aug 12: Elo −209→+62,
      wr_random 0.22→0.67, connections stable ~35, species healthy
- [ ] RQ1: 3/3 seeds per condition (neat + fixed)
- [ ] `aggregate.py` figures with CI bands
- [ ] `final_eval.py` table at depth 8 = headline numbers
- [ ] Methods section can be drafted any time (pipeline is final)
- [ ] Frame honestly: a controlled comparison at modest-but-real strength,
      NOT "we built a strong checkers player"
- [ ] Cite the pilot (`runs/pilot_v2_rq1_topo/`) + the game-tree pathology
      finding (material-only eval is non-monotonic in depth) as motivation
      for the measurement design
- [ ] Optional, only if RQ1 curves are strong: RQ2 (section 2b)
