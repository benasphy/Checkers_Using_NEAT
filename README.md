# Checkers AI: NEAT-Evolved Value Network + Alpha-Beta Search

A research-grade checkers (American/English draughts) agent in the spirit of
Blondie24: a **NEAT-evolved value network** provides the position evaluation,
and **iterative-deepening alpha-beta search** provides the tactics. Evolution
never sees hand-crafted strategic knowledge - fitness comes purely from game
outcomes in coevolutionary tournaments.

## Why this architecture?

Checkers is a highly tactical, forced-capture game. Every strong checkers
program in history (Chinook, Blondie24) couples an evaluation function with
deep search; a raw network choosing moves without search cannot compete.
Here NEAT supplies the *learned* component (the evaluation, with evolved
topology), and search supplies the tactical strength. Note that checkers is
weakly solved (perfect play draws), so the realistic target is superhuman
*practical* strength, not "always wins".

## Components

| Path | What it is |
|---|---|
| `checkers/bitboard.py` | 32-square bitboard engine: full American rules (forced captures, multi-jump sequences as single moves, crowning ends the move, no-move = loss), Zobrist hashing, perft validation |
| `checkers/game.py` | Game wrapper: threefold repetition, 40-move rule, legacy UI API |
| `ai/features.py` | Canonical side-to-move encoding (36 inputs, color-symmetric) |
| `ai/value_net.py` | NEAT genome -> cached evaluation function |
| `ai/search.py` | Negamax alpha-beta: TT, forced-capture quiescence, repetition-aware, seeded tie-breaking |
| `ai/ladder.py` | Reference opponents: random + material-only alpha-beta at fixed depths |
| `ai/arena.py` | Seeded, color-balanced match runner (parallelized) |
| `ai/elo.py` | Elo fitting for ladder calibration and progress measurement |
| `ai/train_path_a.py` | Evolution loop: outcome-only tournament fitness, Elo-gated hall of fame, periodic ladder probes |

## Quickstart

```bash
pip install -r requirements.txt

# 1. Validate the engine (perft matches published values)
python -m pytest                 # full test suite
python main.py perft --depth 7   # 7 49 302 1469 7361 36768 179740

# 2. Train (long-running; start small to see it work)
python main.py train --generations 200 --depth 4 --seed 0

# 3. Measure strength vs the fixed ladder (Elo, random anchored at 0)
python main.py ladder --genome best_value_genome.pkl --depth 6 --games 20

# 4. Play against it
python main.py play --depth 6          # terminal
python web_visualize.py                # browser at http://localhost:5000

# 5. Plot training progress
python plot_fitness.py runs/path_a/training_metrics.csv
```

## Training design (what makes this sound)

- **Outcome-only fitness.** Each genome's fitness is its mean score from
  color-balanced games vs randomly paired population peers and hall-of-fame
  members. No reward shaping; draws carry a bounded material tie-break
  (0.4-0.6) only so early populations have a gradient.
- **Elo-gated hall of fame.** The generation best enters the HOF only by
  scoring >= 55% in a strict gate match vs current HOF members - prevents
  coevolutionary cycling.
- **Measurement is separate from training.** Strength is probed against a
  fixed ladder (random, material-d1/d2/d4) that is never used for fitness,
  and reported as performance Elo with the ladder calibrated once.
- **Reproducibility.** Everything is seeded; per-generation metrics go to
  `runs/<name>/training_metrics.csv`; populations checkpoint every 10
  generations.

Useful flags: `--pop` (population size), `--depth` (search depth during
training; play deeper than you train), `--workers`, `--probe-every`,
`--peer-rounds`, `--out` (run directory).

## Rules implemented

American checkers: men capture forward only, captures mandatory, multi-jumps
must be completed with the same piece, crowning ends the move, kings move one
diagonal step, side with no legal move loses, draws by threefold repetition
or 80 plies without a capture/man move. The move generator is validated by
perft against published node counts and by differential testing against an
independent naive implementation (`tests/test_engine.py`).

## Research roadmap

1. **Path A (this code):** NEAT value net + alpha-beta. Baselines: fixed
   ladder Elo curves across seeds.
2. **Path B (planned):** policy+value genome with a fixed 128-logit action
   encoding + PUCT MCTS; MCTS visit-distribution distillation as a dense
   fitness signal.
3. **Ablations for the paper:** topology evolution vs fixed topology
   (Blondie24 replica), search depth during evolution, fitness variants,
   alpha-beta vs MCTS at equal compute.

Key related work: Chellapilla & Fogel (Blondie24), Schaeffer et al. (Chinook,
"Checkers Is Solved"), Stanley & Miikkulainen (NEAT), Gauci & Stanley
(HyperNEAT checkers), Silver et al. (AlphaZero).

## License

MIT (see LICENSE).
