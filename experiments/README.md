# Experiments — paper protocol (arXiv preprint, target: IEEE CoG / GECCO)

Working paper title: **"Does Topology Matter? Search-Amplified Neuroevolution
of Checkers Players"** — a Blondie24-style coevolution study with modern
rigor (multiple seeds, calibrated Elo, equal-compute comparisons).

## Research questions & naming

| RQ | Name | Question | Conditions |
|---|---|---|---|
| RQ1 | **TOPO** | Does evolving topology (NEAT) beat fixed-topology evolution (Blondie24-style) at equal compute? | `topo-neat` vs `topo-fixed` |
| RQ2 | **AMP** | How does search depth *during* evolution affect final strength and sample efficiency? | `amp-d2`, `amp-d4`, `amp-d6` |

- `topo-neat`: `neat_value_config.txt`, depth 4 (topology evolves)
- `topo-fixed`: `neat_fixed_config.txt`, depth 4 (identical except ALL
  structural mutation rates = 0; weights/biases only — the control)
- `amp-dN`: `neat_value_config.txt`, training depth N

Equal compute = same generations, population, games per generation, seeds.
The old `runs/path_a/` 300-generation run is the **pilot study** (it used the
old growth-biased config and a 40-game ladder calibration; treat as such).

## Protocol

```bash
# 1. RQ1: 2 conditions x 5 seeds (run cells in parallel shells if you have
#    the cores; --only runs a single cell)
python experiments/run_experiment.py --rq rq1_topo --seeds 0 1 2 3 4 \
    --generations 300
python experiments/run_experiment.py --rq rq1_topo --only neat_s3

# 2. RQ2: 3 conditions x 5 seeds
python experiments/run_experiment.py --rq rq2_amp --seeds 0 1 2 3 4 \
    --generations 300

# 3. Learning-curve figures (mean +/- 95% CI across seeds) -> paper/figures/
python experiments/aggregate.py --rq rq1_topo
python experiments/aggregate.py --rq rq2_amp

# 4. Final comparison at a COMMON deep depth (removes the RQ2 confound that
#    deeper-trained agents also probe deeper during training)
python experiments/final_eval.py --rq rq1_topo --depth 8 --games 40
python experiments/final_eval.py --rq rq2_amp --depth 8 --games 40
```

Outputs for the paper land in `paper/figures/`:
`*_elo.pdf/.png` (learning curves), `*_complexity.pdf` (bloat check),
`*_summary.csv` (per-generation means/CIs), `*_final_eval.csv` (results table).

## Measurement hygiene (what reviewers will ask about)

- Ladder calibrated once at **200 games/pair**, cache versioned by games
  count (`ladder_calibration_g200.json`); ratings sanity-checked for
  monotonicity in depth.
- Probe = 50 games/rung x 4 rungs every 5 generations (was 8: ~±100 Elo
  noise). Figures use a moving average; CIs come from seed replication.
- Frozen benchmark champions at gens 50/100/150/200/250 are probed every 5
  gens (`wr_bm*` columns) — a stationary measuring stick that detects
  coevolutionary forgetting/cycling.
- Best-ever genome is selected by **measured probe Elo**, never by noisy
  tournament fitness, and is never overwritten (`best_ever_genome.pkl`).
- Fitness games use seeded random openings (8 plies); color-swapped pairs
  share the opening, so colors are fairly balanced per pairing.
- Hall of fame members carry running Elo ratings (gate = 24 games at >=55%);
  when full, the *weakest* member is evicted, not the oldest.

## Paper checklist (what's left before writing Results)

- [ ] RQ1: 5/5 seeds per condition finished
- [ ] RQ2: 5/5 seeds per condition finished
- [ ] `aggregate.py` figures show CI bands (n=5)
- [ ] `final_eval.py` tables at depth 8 (the paper's headline numbers)
- [ ] Optional ablation: `--extra peer_rounds=2` rerun of one cell to show
      fitness-measurement noise matters
- [ ] Methods section can be drafted now — the pipeline is final
