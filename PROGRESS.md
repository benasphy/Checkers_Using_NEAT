# Project Progress & Guide

## 1. What happened (Aug 2026 rebuild)

The original NEAT agent was weak due to fatal bugs, all now fixed by a ground-up rebuild:

| Old problem | Fix |
|---|---|
| Wrong rules: no multi-jumps, crowning kept jumping, wrong loss condition, no draw rules | Bitboard engine (`checkers/bitboard.py`), exact American rules, validated by perft (7, 49, 302, 1469, 7361, 36768, 179740) + differential tests |
| Policy output = index into an arbitrary move list (unlearnable) | Value network + alpha-beta search (Blondie24 architecture) |
| Config started networks with ZERO connections | `initial_connection = full_direct` |
| Half of fitness signal inverted (side-swap credit bug), value nets trained on noise, parallel eval discarded | New arena with correct color-balanced credit, one population, outcome-only fitness |
| No real strength measure (only vs random) | Fixed ladder (random, material-d1/2/4) + calibrated Elo probes, never used for fitness |
| No seeds/reproducibility | Everything seeded; metrics in `runs/<name>/training_metrics.csv` |

Verification: 27 tests pass (`python -m pytest`); a hand-wired material genome scores 10/10 vs random (plumbing proof); end-to-end smoke training + resume tested.

## 2. Training: start / stop / resume (power outage safe)

```bash
# start (~1 min/gen, pop 96, 8 cores)
python main.py train --generations 300 --depth 4 --seed 0

# stop anytime: Ctrl+C  (or power outage - same thing)
# a checkpoint is saved every 5 generations to runs/path_a/neat-checkpoint-N

# resume from the newest checkpoint (you lose at most ~5 generations):
ls runs/path_a/ | grep checkpoint          # find highest N
python main.py train --generations 300 --depth 4 --seed 0 \
    --resume runs/path_a/neat-checkpoint-N
```

The hall of fame (`runs/path_a/hof.pkl`), metrics CSV, and ladder calibration reload automatically. `best_value_genome.pkl` (repo root) is always the latest best - the web app and `main.py play` use it even mid-training.

Monitor: `python plot_fitness.py` (fitness + Elo curves) or watch the `elo` column in `runs/path_a/training_metrics.csv`. Early generations losing to random is normal.

## 3. Next steps (in order)

1. **Run 1 long training** (300+ gens, seed 0). Check Elo curve rises past material-d2 (~800), then d4.
2. **Repeat with seeds 1-4** -> mean +/- CI curves (required for a paper).
3. **Deep-play evaluation**: `python main.py ladder --depth 8 --games 40 --d6` (train shallow, play deep).
4. **Speed** (optional): vectorized net evaluator or Rust/C engine -> bigger pops, depth 6 training.
5. **Path B (novelty)**: policy+value genome (128-logit fixed action space) + PUCT MCTS; use MCTS visit distributions as a dense distillation fitness. Value-net fitness = MSE predicting self-play outcomes.
6. **Baselines/ablations**: fixed-topology evolution (Blondie24 replica) vs NEAT under equal compute; fitness variants; depth-during-evolution sweep; alpha-beta vs MCTS at equal compute.
7. **Human study**: collect games via the web app.

## 4. Research paper: is it publishable?

**Yes - with the right framing.** Not "first checkers AI" (checkers is solved: Chinook, Science 2007; and Blondie24 already evolved an expert-level net in 1999). Publishable angles this codebase supports:

- **RQ1**: Does evolving topology (NEAT) beat fixed-topology evolution (Blondie24) at equal compute? (unanswered in literature at modern scale)
- **RQ2**: Search-amplified coevolution - how does search depth during evolution affect final strength/sample efficiency?
- **RQ3 (main novelty, Path B)**: MCTS visit-distribution distillation as a *gradient-free* dense fitness for neuroevolution ("AlphaZero targets without backprop").
- **RQ4**: Outcome-only vs shaped fitness ablation (you have the broken shaped version in git history as a case study).

Target venues: IEEE CoG / Transactions on Games, GECCO, EvoStar/EvoApplications; workshop track first is a realistic path. Must-cite: Stanley & Miikkulainen (NEAT), Chellapilla & Fogel (Blondie24), Schaeffer et al. (Chinook), Gauci & Stanley (HyperNEAT checkers), Silver et al. (AlphaZero).

Requirements to be credible: >= 5 seeds per condition, Elo with error bars, fixed compute budgets per comparison, honest "solved game / practical strength" framing. The engine correctness tests + seeded pipeline already satisfy the reproducibility bar.

## 5. Command reference

```bash
python -m pytest                    # test suite
python main.py perft --depth 7      # engine validation
python main.py train ...            # see flags: --pop --depth --workers --resume --out --seed
python main.py ladder --games 40    # Elo vs reference ladder
python main.py play --depth 6       # play in terminal
python web_visualize.py             # play in browser
python plot_fitness.py              # training curves
```
