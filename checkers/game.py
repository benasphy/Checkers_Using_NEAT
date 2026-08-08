"""High-level game wrapper around the bitboard engine.

Adds the game-level rules the raw engine leaves to the caller:

- threefold repetition draw (Zobrist hash occurrence count >= 3),
- the 40-move rule (80 consecutive plies without a capture or a man move),
- loss for the side to move with no legal move.

Also provides a legacy-compatible surface for the web UI:
``current_player``, ``board.board`` (8x8 numpy array, 0 empty / 1 P1 man /
2 P2 man / 3 P1 king / 4 P2 king), ``get_legal_moves()`` returning
``(from_row, from_col, to_row, to_col, [(r, c), ...captures])`` tuples and
``make_move()`` accepting them.
"""

from __future__ import annotations

from .bitboard import Move, Position, sq_to_rc

MAX_HMC_PLIES = 80  # 40 moves per side without capture or man move -> draw


class _BoardView:
    """Read-only 8x8 array adapter kept for legacy callers/templates."""

    def __init__(self, game: "CheckersGame"):
        self._game = game

    @property
    def board(self):
        return self._game.position.to_array()

    def get_piece(self, row, col):
        return int(self.board[row, col])

    def __str__(self):
        return str(self._game.position)


class CheckersGame:
    def __init__(self):
        self.board = _BoardView(self)
        self.reset()

    # -- state ----------------------------------------------------------------

    def reset(self):
        self.position = Position.initial()
        self.hash_history = [self.position.hash]
        self._hash_counts = {self.position.hash: 1}
        self.move_history: list[Move] = []

    @property
    def current_player(self) -> int:
        return self.position.turn

    # -- moves ------------------------------------------------------------------

    def engine_moves(self) -> list[Move]:
        """Legal moves as engine ``Move`` namedtuples (multi-jump = one move)."""
        if self._is_drawn():
            return []
        return self.position.legal_moves()

    def get_legal_moves(self, player=None):
        """Legacy tuple format. If ``player`` differs from the side to move the
        moves for a hypothetical position with that side to move are returned
        (kept for backward compatibility)."""
        pos = self.position if player in (None, self.position.turn) \
            else self.position.with_turn(player)
        if self._is_drawn():
            return []
        return [self._to_ui(mv) for mv in pos.legal_moves()]

    @staticmethod
    def _to_ui(mv: Move):
        fr, fc = sq_to_rc(mv.fr)
        tr, tc = sq_to_rc(mv.to)
        caps = [sq_to_rc(s) for s in range(32) if (mv.cap >> s) & 1]
        return (fr, fc, tr, tc, caps)

    def find_engine_move(self, from_rc, to_rc):
        """Match a (row,col) -> (row,col) request against legal engine moves.

        If several jump sequences share origin and destination the first is
        returned (rare ambiguity; acceptable for casual UI play)."""
        for mv in self.position.legal_moves():
            if sq_to_rc(mv.fr) == tuple(from_rc) and sq_to_rc(mv.to) == tuple(to_rc):
                return mv
        return None

    def make_engine_move(self, mv: Move):
        self.position = self.position.apply(mv)
        h = self.position.hash
        self.hash_history.append(h)
        self._hash_counts[h] = self._hash_counts.get(h, 0) + 1
        self.move_history.append(mv)

    def make_move(self, move):
        """Accept an engine ``Move`` or a legacy 5-tuple."""
        if isinstance(move, Move):
            self.make_engine_move(move)
            return
        fr_r, fr_c, to_r, to_c = move[0], move[1], move[2], move[3]
        mv = self.find_engine_move((fr_r, fr_c), (to_r, to_c))
        if mv is None:
            raise ValueError(f"Illegal move: {move}")
        self.make_engine_move(mv)

    # -- termination ---------------------------------------------------------------

    def _is_drawn(self) -> bool:
        if self.position.hmc >= MAX_HMC_PLIES:
            return True
        return self._hash_counts.get(self.position.hash, 0) >= 3

    def is_game_over(self) -> bool:
        if self._is_drawn():
            return True
        return not self.position.legal_moves()

    def get_winner(self):
        """1 / 2 for a win, 0 for a draw, None if the game is not over."""
        if self._is_drawn():
            return 0
        if not self.position.legal_moves():
            return 2 if self.position.turn == 1 else 1
        return None

    def draw_reason(self):
        if self.position.hmc >= MAX_HMC_PLIES:
            return "40-move rule"
        if self._hash_counts.get(self.position.hash, 0) >= 3:
            return "threefold repetition"
        return None

    # -- misc ------------------------------------------------------------------------

    def material_diff(self, player: int = 1) -> int:
        """Material balance in centipawns from ``player``'s perspective
        (man = 100, king = 140)."""
        m1, k1 = self.position.material(1)
        m2, k2 = self.position.material(2)
        diff = 100 * (m1 - m2) + 140 * (k1 - k2)
        return diff if player == 1 else -diff

    def get_state(self):
        """Legacy helper: flattened 8x8 array and side to move."""
        return self.position.to_array().flatten(), self.position.turn
