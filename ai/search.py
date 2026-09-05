"""Iterative-deepening alpha-beta (negamax) search for checkers.

Designed around the properties of the game:

- Multi-jumps are single moves in the engine, so the side to move always
  flips between plies and plain negamax sign handling is correct.
- Captures are forced. "Quiescence" is therefore trivial: at depth <= 0 the
  search simply keeps going while the side to move has captures (capture
  chains strictly remove pieces, so this always terminates) and evaluates
  only quiet positions.
- Draw awareness: threefold repetition and the 40-move-rule counter are
  evaluated with the same thresholds as the game wrapper.

Features: transposition table, TT-move + capture-size move ordering,
single-reply extensions, seeded random tie-breaking at the root (so equal
engines produce varied games), optional time limit.

The evaluation function is pluggable: ``eval_fn(position) -> float`` from the
side-to-move perspective. Values must stay well below ``MATE``.
"""

from __future__ import annotations

import random
import time

from checkers.bitboard import Position, _popcount
from checkers.game import MAX_HMC_PLIES

MATE = 100_000.0
_MATE_BAND = MATE - 500.0
_INF = float("inf")

_EXACT, _LOWER, _UPPER = 0, 1, 2


class _Timeout(Exception):
    pass


class Searcher:
    def __init__(self, eval_fn, depth: int = 4, seed=None, tt_max: int = 1_000_000):
        self.eval_fn = eval_fn
        self.depth = depth
        self.rng = random.Random(seed)
        self.tt: dict[tuple, tuple] = {}
        self.tt_max = tt_max
        self.nodes = 0
        self.evaluations = 0
        self._rep: dict[int, int] = {}
        self._deadline = None

    # -- public API -----------------------------------------------------------

    def best_move(self, game_or_pos, hash_history=None, max_seconds=None):
        if hasattr(game_or_pos, "position"):
            pos: Position = game_or_pos.position
            history = list(game_or_pos.hash_history)
        else:
            pos = game_or_pos
            history = list(hash_history) if hash_history else [pos.hash]

        # Root histories are independent; never reuse values across games.
        self.tt.clear()
        moves = pos.legal_moves()
        if not moves:
            return None
        if len(moves) == 1:
            return moves[0]

        # Repetition table: occurrences of positions strictly before the
        # current node. Each node pushes its own hash while its children are
        # searched, so drop the current position's entry from the history.
        self._rep = {}
        for h in history[:-1]:
            self._rep[h] = self._rep.get(h, 0) + 1

        self._deadline = (time.monotonic() + max_seconds) if max_seconds else None
        order = moves[:]
        self.rng.shuffle(order)
        best = order[0]
        scores: dict[tuple, float] = {}

        try:
            for d in range(1, self.depth + 1):
                alpha = -_INF
                iter_scores: dict[tuple, float] = {}
                self._rep[pos.hash] = self._rep.get(pos.hash, 0) + 1
                try:
                    for mv in order:
                        child = pos.apply(mv)
                        v = -self._negamax(child, d - 1, -_INF, -alpha, 1)
                        iter_scores[(mv.fr, mv.to, mv.cap)] = v
                        if v > alpha:
                            alpha = v
                finally:
                    self._rep[pos.hash] -= 1
                scores = iter_scores
                order.sort(key=lambda m: scores.get((m.fr, m.to, m.cap), -_INF),
                           reverse=True)
                best = order[0]
                if alpha >= _MATE_BAND:  # forced win found; no need to go deeper
                    break
        except _Timeout:
            pass  # keep the best move of the last completed iteration

        if scores:
            bv = max(scores.values())
            top = [m for m in order if scores.get((m.fr, m.to, m.cap)) == bv]
            if top:
                best = self.rng.choice(top)
        return best

    # -- core -------------------------------------------------------------------

    def _negamax(self, pos: Position, depth: int, alpha: float, beta: float,
                 ply: int) -> float:
        self.nodes += 1
        if self._deadline is not None and (self.nodes & 0xFFF) == 0 \
                and time.monotonic() > self._deadline:
            raise _Timeout

        # Draw rules (never applied at the root, ply >= 1 here by construction).
        if pos.hmc >= MAX_HMC_PLIES:
            return 0.0
        if self._rep.get(pos.hash, 0) >= 2:
            return 0.0  # repetition scored as draw

        rep_context = frozenset((h, n) for h, n in self._rep.items() if n)
        tt_key = (pos.hash, pos.hmc, rep_context)
        entry = self.tt.get(tt_key)
        tt_move = None
        if entry is not None:
            e_depth, e_flag, e_val, tt_move = entry
            if e_depth >= depth:
                if e_flag == _EXACT:
                    return e_val
                if e_flag == _LOWER:
                    if e_val > alpha:
                        alpha = e_val
                elif e_val < beta:
                    beta = e_val
                if alpha >= beta:
                    return e_val

        moves = pos.legal_moves()
        if not moves:
            return -(MATE - ply)  # side to move has no move: loss

        is_jump = moves[0].cap != 0
        if (depth <= 0 and not is_jump) or ply >= 100:
            self.evaluations += 1
            return self.eval_fn(pos)

        # Move ordering: TT move first, then bigger captures first.
        if is_jump and len(moves) > 1:
            moves.sort(key=lambda m: _popcount(m.cap), reverse=True)
        if tt_move is not None:
            for i, m in enumerate(moves):
                if (m.fr, m.to, m.cap) == tt_move:
                    if i:
                        moves.insert(0, moves.pop(i))
                    break

        child_depth = depth if (len(moves) == 1 and depth > 0) else depth - 1

        best_val = -_INF
        best_mv = None
        orig_alpha = alpha
        self._rep[pos.hash] = self._rep.get(pos.hash, 0) + 1
        try:
            for mv in moves:
                v = -self._negamax(pos.apply(mv), child_depth, -beta, -alpha, ply + 1)
                if v > best_val:
                    best_val = v
                    best_mv = mv
                if v > alpha:
                    alpha = v
                if alpha >= beta:
                    break
        finally:
            self._rep[pos.hash] -= 1

        # Store in TT (skip mate-band scores to avoid ply-adjustment issues).
        if depth >= 1 and abs(best_val) < _MATE_BAND:
            flag = _EXACT
            if best_val <= orig_alpha:
                flag = _UPPER
            elif best_val >= beta:
                flag = _LOWER
            self.tt[tt_key] = (depth, flag, best_val,
                               (best_mv.fr, best_mv.to, best_mv.cap))
        return best_val


# ---------------------------------------------------------------------------
# Reference evaluation (material only) - used by the baseline ladder.
# ---------------------------------------------------------------------------

def material_eval(pos: Position) -> float:
    """Material balance from the side to move's perspective
    (man = 100, king = 140)."""
    m1, k1 = _popcount(pos.m1), _popcount(pos.k1)
    m2, k2 = _popcount(pos.m2), _popcount(pos.k2)
    diff = 100.0 * (m1 - m2) + 140.0 * (k1 - k2)
    return diff if pos.turn == 1 else -diff
