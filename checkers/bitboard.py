"""Bitboard engine for American (English) checkers / draughts.

Board layout
------------
Only the 32 dark squares are playable. Square index ``s`` in ``0..31`` maps to
the 8x8 board as::

    row = s // 4
    col = 2 * (s % 4) + (1 if row is even else 0)

Row 0 is the TOP of the displayed board. Player 2 starts on rows 0-2
(squares 0-11) and moves DOWN (toward row 7). Player 1 starts on rows 5-7
(squares 20-31) and moves UP (toward row 0). Player 1 moves first
(this matches the original project's convention; which color is labelled
"first" is cosmetic and does not affect play strength).

Rules implemented (American checkers):
- Men move/capture diagonally forward only; kings in all four directions.
- Captures are mandatory. Multi-jumps must be continued with the SAME piece
  until no further jump is available (any available jump sequence may be
  chosen; the "maximum capture" rule of international draughts does NOT apply).
- A man that reaches the crowning row is kinged and the move ends immediately,
  even if further jumps would be available.
- Captured pieces remain on the board until the move is complete: they block
  landing squares and cannot be jumped twice.
- A player with no legal move on their turn loses.

Draw bookkeeping (threefold repetition, 40-move rule) is the responsibility of
the caller (see ``checkers.game.CheckersGame``); the engine exposes a Zobrist
hash and a half-move clock (``hmc``: plies since the last capture or man move)
to support it.

Moves are complete sequences: a multi-jump is ONE move. ``Move.cap`` is the
bitmask of captured squares and ``Move.path`` the visited squares including
origin and final landing square.
"""

from __future__ import annotations

import random
from collections import namedtuple

# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

NUM_SQUARES = 32
FULL_MASK = (1 << NUM_SQUARES) - 1

# Directions: 0=NW, 1=NE, 2=SW, 3=SE  (N = toward row 0 = "up")
_DIR_DELTAS = ((-1, -1), (-1, 1), (1, -1), (1, 1))
ALL_DIRS = (0, 1, 2, 3)
# Men of player 1 move up (NW/NE); men of player 2 move down (SW/SE).
MAN_DIRS = {1: (0, 1), 2: (2, 3)}

CROWN_MASK = {1: 0x0000000F, 2: 0xF0000000}  # rows 0 and 7
CROWN_ROW = {1: 0, 2: 7}


def sq_to_rc(s: int) -> tuple[int, int]:
    row = s >> 2
    col = 2 * (s & 3) + (1 if row % 2 == 0 else 0)
    return row, col


def rc_to_sq(row: int, col: int) -> int:
    """Return square index for (row, col), or -1 if not a playable square."""
    if not (0 <= row < 8 and 0 <= col < 8):
        return -1
    if (row + col) % 2 == 0:  # light square, not playable
        return -1
    return row * 4 + (col - (1 if row % 2 == 0 else 0)) // 2


def _build_tables():
    step = [[-1] * NUM_SQUARES for _ in range(4)]
    jump = [[-1] * NUM_SQUARES for _ in range(4)]
    for s in range(NUM_SQUARES):
        r, c = sq_to_rc(s)
        for d, (dr, dc) in enumerate(_DIR_DELTAS):
            step[d][s] = rc_to_sq(r + dr, c + dc)
            jump[d][s] = rc_to_sq(r + 2 * dr, c + 2 * dc)
    return step, jump


STEP, JUMP = _build_tables()

# ---------------------------------------------------------------------------
# Zobrist hashing
# ---------------------------------------------------------------------------

_zrng = random.Random(0x5EED_C0DE)
# Piece kinds: 0 = man P1, 1 = king P1, 2 = man P2, 3 = king P2
Z_PIECE = [[_zrng.getrandbits(64) for _ in range(NUM_SQUARES)] for _ in range(4)]
Z_TURN = _zrng.getrandbits(64)  # xored in when it is player 2's turn


def _bits(mask: int):
    """Iterate over set bit indices of a mask."""
    while mask:
        lsb = mask & -mask
        yield lsb.bit_length() - 1
        mask ^= lsb


popcount = getattr(int, "bit_count", None)
if popcount is None:  # Python < 3.10 fallback
    def _popcount(x):  # pragma: no cover
        return bin(x).count("1")
else:
    def _popcount(x):
        return x.bit_count()


# ---------------------------------------------------------------------------
# Moves and positions
# ---------------------------------------------------------------------------

Move = namedtuple("Move", ("fr", "to", "cap", "path"))
# fr, to: square indices. cap: bitmask of captured squares.
# path: tuple of visited squares, path[0] == fr, path[-1] == to.


class Position:
    """Immutable-ish position: apply() returns a new Position."""

    __slots__ = ("m1", "k1", "m2", "k2", "turn", "hmc", "hash")

    def __init__(self, m1, k1, m2, k2, turn, hmc=0, h=None):
        self.m1 = m1
        self.k1 = k1
        self.m2 = m2
        self.k2 = k2
        self.turn = turn
        self.hmc = hmc
        self.hash = self._compute_hash() if h is None else h

    # -- construction -------------------------------------------------------

    @staticmethod
    def initial() -> "Position":
        m2 = 0x00000FFF  # squares 0-11  (top, player 2)
        m1 = 0xFFF00000  # squares 20-31 (bottom, player 1)
        return Position(m1, 0, m2, 0, turn=1, hmc=0)

    def _compute_hash(self) -> int:
        h = 0
        for kind, mask in enumerate((self.m1, self.k1, self.m2, self.k2)):
            zk = Z_PIECE[kind]
            for s in _bits(mask):
                h ^= zk[s]
        if self.turn == 2:
            h ^= Z_TURN
        return h

    # -- queries -------------------------------------------------------------

    @property
    def occupied(self) -> int:
        return self.m1 | self.k1 | self.m2 | self.k2

    def pieces_of(self, player: int) -> tuple[int, int]:
        return (self.m1, self.k1) if player == 1 else (self.m2, self.k2)

    def material(self, player: int) -> tuple[int, int]:
        men, kings = self.pieces_of(player)
        return _popcount(men), _popcount(kings)

    def with_turn(self, player: int) -> "Position":
        if player == self.turn:
            return self
        return Position(self.m1, self.k1, self.m2, self.k2, player, self.hmc,
                        self.hash ^ Z_TURN)

    # -- move generation ------------------------------------------------------

    def legal_moves(self) -> list[Move]:
        me = self.turn
        my_men, my_kings = self.pieces_of(me)
        opp_all = (self.m2 | self.k2) if me == 1 else (self.m1 | self.k1)
        occ = self.occupied
        crown = CROWN_MASK[me]
        man_dirs = MAN_DIRS[me]

        jumps: list[Move] = []
        for s in _bits(my_men):
            self._jump_dfs(s, False, occ & ~(1 << s), opp_all, man_dirs, crown, jumps)
        for s in _bits(my_kings):
            self._jump_dfs(s, True, occ & ~(1 << s), opp_all, man_dirs, crown, jumps)
        if jumps:
            return jumps

        moves: list[Move] = []
        empty = ~occ & FULL_MASK
        for s in _bits(my_men):
            for d in man_dirs:
                t = STEP[d][s]
                if t >= 0 and (empty >> t) & 1:
                    moves.append(Move(s, t, 0, (s, t)))
        for s in _bits(my_kings):
            for d in ALL_DIRS:
                t = STEP[d][s]
                if t >= 0 and (empty >> t) & 1:
                    moves.append(Move(s, t, 0, (s, t)))
        return moves

    def _jump_dfs(self, origin, is_king, occ_wo_origin, opp_all, man_dirs, crown, out):
        """Collect all complete jump sequences for the piece at ``origin``.

        - ``occ_wo_origin``: all occupied squares with the moving piece lifted.
          Captured pieces stay in it (they remain on the board during the move,
          blocking landing squares).
        - A sequence ends when no further jump exists, or immediately when a
          man lands on the crowning row.
        """
        dirs = ALL_DIRS if is_king else man_dirs
        step, jump = STEP, JUMP

        def rec(s, cap, path):
            extended = False
            for d in dirs:
                m = step[d][s]
                l = jump[d][s]
                if m < 0 or l < 0:
                    continue
                mb = 1 << m
                if not (opp_all & mb) or (cap & mb):
                    continue  # nothing to jump, or already captured
                if (occ_wo_origin >> l) & 1:
                    continue  # landing square blocked
                extended = True
                ncap = cap | mb
                npath = path + (l,)
                if not is_king and ((1 << l) & crown):
                    out.append(Move(origin, l, ncap, npath))  # crowning ends move
                else:
                    rec(l, ncap, npath)
            if not extended and cap:
                out.append(Move(origin, s, cap, path))

        rec(origin, 0, (origin,))

    def has_any_move(self) -> bool:
        return bool(self.legal_moves())

    # -- applying moves --------------------------------------------------------

    def apply(self, move: Move) -> "Position":
        me = self.turn
        fr_bit = 1 << move.fr
        to_bit = 1 << move.to
        m1, k1, m2, k2 = self.m1, self.k1, self.m2, self.k2
        h = self.hash

        if me == 1:
            was_king = bool(k1 & fr_bit)
            crowned = (not was_king) and bool(to_bit & CROWN_MASK[1])
            if was_king:
                k1 = (k1 & ~fr_bit) | to_bit
                h ^= Z_PIECE[1][move.fr] ^ Z_PIECE[1][move.to]
            else:
                m1 &= ~fr_bit
                h ^= Z_PIECE[0][move.fr]
                if crowned:
                    k1 |= to_bit
                    h ^= Z_PIECE[1][move.to]
                else:
                    m1 |= to_bit
                    h ^= Z_PIECE[0][move.to]
            if move.cap:
                for s in _bits(move.cap & m2):
                    h ^= Z_PIECE[2][s]
                for s in _bits(move.cap & k2):
                    h ^= Z_PIECE[3][s]
                m2 &= ~move.cap
                k2 &= ~move.cap
        else:
            was_king = bool(k2 & fr_bit)
            crowned = (not was_king) and bool(to_bit & CROWN_MASK[2])
            if was_king:
                k2 = (k2 & ~fr_bit) | to_bit
                h ^= Z_PIECE[3][move.fr] ^ Z_PIECE[3][move.to]
            else:
                m2 &= ~fr_bit
                h ^= Z_PIECE[2][move.fr]
                if crowned:
                    k2 |= to_bit
                    h ^= Z_PIECE[3][move.to]
                else:
                    m2 |= to_bit
                    h ^= Z_PIECE[2][move.to]
            if move.cap:
                for s in _bits(move.cap & m1):
                    h ^= Z_PIECE[0][s]
                for s in _bits(move.cap & k1):
                    h ^= Z_PIECE[1][s]
                m1 &= ~move.cap
                k1 &= ~move.cap

        h ^= Z_TURN
        # Half-move clock: resets on any capture or any man move.
        hmc = 0 if (move.cap or not was_king) else self.hmc + 1
        return Position(m1, k1, m2, k2, 2 if me == 1 else 1, hmc, h)

    # -- misc -------------------------------------------------------------------

    def to_array(self):
        """8x8 int array using the legacy encoding:
        0 empty, 1 P1 man, 2 P2 man, 3 P1 king, 4 P2 king."""
        import numpy as np
        arr = np.zeros((8, 8), dtype=int)
        for kind, mask, val in ((0, self.m1, 1), (1, self.k1, 3),
                                (2, self.m2, 2), (3, self.k2, 4)):
            for s in _bits(mask):
                r, c = sq_to_rc(s)
                arr[r, c] = val
        return arr

    def __str__(self):
        symbols = {0: ".", 1: "r", 2: "b", 3: "R", 4: "B"}
        arr = self.to_array()
        lines = ["  0 1 2 3 4 5 6 7"]
        for r in range(8):
            lines.append(str(r) + " " + " ".join(symbols[int(v)] for v in arr[r]))
        return "\n".join(lines)

    def __eq__(self, other):
        return (self.m1 == other.m1 and self.k1 == other.k1
                and self.m2 == other.m2 and self.k2 == other.k2
                and self.turn == other.turn)

    def __hash__(self):
        return self.hash


# ---------------------------------------------------------------------------
# Perft (for engine validation)
# ---------------------------------------------------------------------------

def perft(pos: Position, depth: int) -> int:
    """Count leaf nodes of the game tree (multi-jumps count as one move).

    Standard perft: no draw rules, a side with no moves is simply a leaf's
    parent (its node has zero children).
    """
    if depth == 0:
        return 1
    moves = pos.legal_moves()
    if depth == 1:
        return len(moves)
    total = 0
    for mv in moves:
        total += perft(pos.apply(mv), depth - 1)
    return total
