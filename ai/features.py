"""Canonical board encoding for value networks.

The board is always presented from the perspective of the side to move:

- If player 2 is to move the board is rotated 180 degrees (square ``s`` maps
  to ``31 - s``) and piece ownership is swapped, so "my men always advance
  toward canonical row 0".
- Inputs 0..31: +1.0 my man, +KING_VALUE my king, -1.0 / -KING_VALUE for the
  opponent, 0 empty.
- Inputs 32..35: my men, my kings, opponent men, opponent kings (each / 12).

This gives NUM_INPUTS = 36 and guarantees color symmetry: a position and its
color-swapped rotation produce identical feature vectors.
"""

from __future__ import annotations

from checkers.bitboard import NUM_SQUARES, Position, _popcount

KING_VALUE = 1.3
NUM_INPUTS = NUM_SQUARES + 4


def encode(pos: Position) -> list[float]:
    if pos.turn == 1:
        my_m, my_k, op_m, op_k = pos.m1, pos.k1, pos.m2, pos.k2
        flip = False
    else:
        my_m, my_k, op_m, op_k = pos.m2, pos.k2, pos.m1, pos.k1
        flip = True

    x = [0.0] * NUM_INPUTS
    for canon in range(NUM_SQUARES):
        s = (NUM_SQUARES - 1 - canon) if flip else canon
        b = 1 << s
        if my_m & b:
            x[canon] = 1.0
        elif my_k & b:
            x[canon] = KING_VALUE
        elif op_m & b:
            x[canon] = -1.0
        elif op_k & b:
            x[canon] = -KING_VALUE
    x[32] = _popcount(my_m) / 12.0
    x[33] = _popcount(my_k) / 12.0
    x[34] = _popcount(op_m) / 12.0
    x[35] = _popcount(op_k) / 12.0
    return x
