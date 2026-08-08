# Checkers AI - Design Documentation

This document explains the architecture and the reasoning behind it. For
commands and quickstart, see `README.md`.

---

## 1. Overview

The agent follows the Blondie24 recipe, modernized: evolution (NEAT) learns a
**value function** with no human strategic knowledge, and **alpha-beta search**
turns that evaluation into strong tactical play. The project is structured for
research use: a rule-exact engine validated by perft, seeded reproducible
training, and strength measurement that is strictly separated from fitness.

```
game outcome fitness            measurement only
   (peers + HOF)                (never trains)
        |                             |
   NEAT population  --best-->  fixed ladder (random, material-d1/2/4)
        |                             |
  value network V(s)            performance Elo
        |
  alpha-beta search (depth d)  ->  moves
```

---

## 2. Engine (`checkers/bitboard.py`)

- 32 playable squares in four 32-bit masks (`m1, k1, m2, k2`), side to move,
  half-move clock, incremental Zobrist hash.
- American rules exactly: mandatory captures; multi-jump sequences generated
  as complete single moves (piece-locked continuation is therefore implicit);
  a man crowned mid-jump stops immediately; captured pieces stay on the board
  until the move completes (they block landing squares and cannot be jumped
  twice); men capture forward only; a side with no legal move loses.
- Validation: `perft(1..8)` equals the published sequence
  7, 49, 302, 1469, 7361, 36768, 179740, 845931, plus differential testing
  against an independent array-based reference implementation over random
  playouts, with incremental-hash verification.

`checkers/game.py` adds threefold repetition and the 40-move rule (80 plies
without capture or man move) and exposes the legacy tuple API used by the web
UI.

## 3. Value network (`ai/features.py`, `ai/value_net.py`)

- Input (36): the 32 squares canonicalized to the side to move (board rotated
  180 degrees for player 2, ownership swapped; +1 man / +1.3 king for the
  mover, negative for the opponent) plus 4 normalized piece counts. A position
  and its color-swapped mirror produce identical inputs by construction.
- Output: one tanh unit in [-1, 1] = expected outcome for the side to move,
  scaled by 600 inside search so it stays below mate scores.
- NEAT config: `initial_connection = full_direct` (fully wired minimal nets;
  the neat-python default `unconnected` would start with constant outputs),
  single output, growth-biased structural mutation rates.

## 4. Search (`ai/search.py`)

Iterative-deepening negamax alpha-beta with:

- transposition table (Zobrist-keyed, bound flags, TT-move ordering),
- trivial-but-correct quiescence: captures are forced in checkers, so at
  depth <= 0 the search simply continues while the side to move has captures
  and only evaluates quiet positions,
- single-reply extensions,
- repetition awareness (positions seen in the game history or search path
  score 0), 40-move-rule awareness,
- seeded random tie-breaking at the root so equal engines produce varied
  games (needed for meaningful match statistics).

Because multi-jumps are single moves, the side to move always alternates
between plies and plain negamax sign handling is exact.

## 5. Training (`ai/train_path_a.py`)

Per generation:

1. Every genome plays color-balanced games against randomly paired peers
   (`--peer-rounds` rounds of random perfect matchings) and against a random
   hall-of-fame member. All games run in parallel across processes with
   per-game seeds.
2. Fitness = mean points per game. Wins/losses are 1/0; draws are
   `0.5 + 0.1*tanh(material/300)` - a bounded tie-break inside (0.4, 0.6)
   that never reorders a win above a draw or a draw above a loss.
3. The generation best must score >= 55% in a strict gate match against
   sampled HOF members to enter the hall of fame (Elo-gating, prevents
   coevolutionary forgetting/cycling).
4. Every `--probe-every` generations the best genome plays the fixed ladder;
   a performance Elo (random anchored at 0, ladder calibrated once and
   cached) is logged to `runs/<name>/training_metrics.csv`.

Truncated games (150 plies) are adjudicated: a 3-man material edge wins,
otherwise draw.

Design rules honored throughout: no reward shaping; correct credit on both
colors; the measurement ladder never contributes fitness; every stochastic
component is seeded.

## 6. Known limitations / next steps

- Training-time search depth (default 4) bounds tactical quality of the
  fitness signal; play deeper than you train (e.g., depth 6-8).
- neat-python network evaluation is the hot loop; a vectorized/compiled
  evaluator (or Rust/C engine port) would enable bigger populations and
  deeper training search.
- Path B (planned): fixed 128-logit policy head + PUCT MCTS with
  visit-distribution distillation as dense fitness; endgame tablebases would
  push toward never-losing play.

## 7. Contact

Email: binidani1903@gmail.com | Telegram: @benasphy
License: MIT.
