"""Engine validation: perft against published values, differential testing
against an independent naive implementation, and rule-specific unit tests."""

import random

import pytest

from checkers.bitboard import (
    Move,
    Position,
    perft,
    rc_to_sq,
    sq_to_rc,
)

# ---------------------------------------------------------------------------
# Published perft values for the starting position of American checkers
# (multi-jump sequences counted as a single move).
# ---------------------------------------------------------------------------

KNOWN_PERFT = {1: 7, 2: 49, 3: 302, 4: 1469, 5: 7361, 6: 36768, 7: 179740,
               8: 845931}


def test_perft_shallow():
    pos = Position.initial()
    for depth in range(1, 7):
        assert perft(pos, depth) == KNOWN_PERFT[depth], f"perft({depth})"


def test_perft_depth7():
    assert perft(Position.initial(), 7) == KNOWN_PERFT[7]


@pytest.mark.slow
def test_perft_depth8():
    assert perft(Position.initial(), 8) == KNOWN_PERFT[8]


# ---------------------------------------------------------------------------
# Independent naive reference implementation (array-based, shares no code
# with the bitboard engine) used for differential testing.
# ---------------------------------------------------------------------------

def _ref_dirs(piece, player):
    if piece in (3, 4):
        return [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    return [(-1, -1), (-1, 1)] if player == 1 else [(1, -1), (1, 1)]


def ref_moves(arr, player):
    """Return set of (origin_rc, final_rc, frozenset(captured_rcs))."""
    own = (1, 3) if player == 1 else (2, 4)
    opp = (2, 4) if player == 1 else (1, 3)
    crown_row = 0 if player == 1 else 7
    jumps = []

    def dfs(r, c, piece, captured, origin):
        extended = False
        for dr, dc in _ref_dirs(piece, player):
            mr, mc = r + dr, c + dc
            lr, lc = r + 2 * dr, c + 2 * dc
            if not (0 <= lr < 8 and 0 <= lc < 8):
                continue
            if arr[mr][mc] not in opp or (mr, mc) in captured:
                continue
            # Landing must be empty; the origin square counts as empty because
            # the moving piece has left it. Captured pieces stay on the board
            # and block landing squares.
            if (lr, lc) != origin and arr[lr][lc] != 0:
                continue
            extended = True
            ncap = captured | {(mr, mc)}
            if piece in (1, 2) and lr == crown_row:
                jumps.append((origin, (lr, lc), frozenset(ncap)))
            else:
                dfs(lr, lc, piece, ncap, origin)
        if not extended and captured:
            jumps.append((origin, (r, c), frozenset(captured)))

    for r in range(8):
        for c in range(8):
            if arr[r][c] in own:
                dfs(r, c, arr[r][c], set(), (r, c))
    if jumps:
        return set(jumps)

    quiets = set()
    for r in range(8):
        for c in range(8):
            p = arr[r][c]
            if p in own:
                for dr, dc in _ref_dirs(p, player):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < 8 and 0 <= nc < 8 and arr[nr][nc] == 0:
                        quiets.add(((r, c), (nr, nc), frozenset()))
    return quiets


def _bb_moves_as_rc(pos):
    out = set()
    for mv in pos.legal_moves():
        caps = frozenset(sq_to_rc(s) for s in range(32) if (mv.cap >> s) & 1)
        out.add((sq_to_rc(mv.fr), sq_to_rc(mv.to), caps))
    return out


def test_differential_vs_reference():
    rng = random.Random(12345)
    for _ in range(150):
        pos = Position.initial()
        for _ply in range(120):
            arr = pos.to_array().tolist()
            assert _bb_moves_as_rc(pos) == ref_moves(arr, pos.turn), (
                f"move set mismatch at:\n{pos}"
            )
            moves = pos.legal_moves()
            if not moves:
                break
            nxt = pos.apply(rng.choice(moves))
            # Incremental Zobrist must match a from-scratch recomputation.
            assert nxt.hash == nxt._compute_hash()
            pos = nxt


# ---------------------------------------------------------------------------
# Rule-specific unit tests
# ---------------------------------------------------------------------------

def _pos(m1=0, k1=0, m2=0, k2=0, turn=1, hmc=0):
    return Position(m1, k1, m2, k2, turn, hmc)


def sq(r, c):
    s = rc_to_sq(r, c)
    assert s >= 0, f"({r},{c}) is not a playable square"
    return s


def test_captures_are_forced():
    # P1 man at (4,3) can capture P2 man at (3,2); P1 man at (6,1) has quiet
    # moves. Only capture moves must be returned.
    p = _pos(m1=(1 << sq(4, 3)) | (1 << sq(6, 1)), m2=1 << sq(3, 2))
    moves = p.legal_moves()
    assert moves and all(m.cap for m in moves)
    assert {(m.fr, m.to) for m in moves} == {(sq(4, 3), sq(2, 1))}


def test_multijump_single_move_and_piece_lock():
    # P1 man at (6,1); P2 men at (5,2) and (3,2): double jump (6,1)->(4,3)->(2,1)
    # must be generated as ONE move; stopping after the first jump is illegal.
    p = _pos(m1=1 << sq(6, 1), m2=(1 << sq(5, 2)) | (1 << sq(3, 2)))
    moves = p.legal_moves()
    assert len(moves) == 1
    mv = moves[0]
    assert (mv.fr, mv.to) == (sq(6, 1), sq(2, 1))
    assert mv.cap == (1 << sq(5, 2)) | (1 << sq(3, 2))
    assert mv.path == (sq(6, 1), sq(4, 3), sq(2, 1))


def test_branching_multijump_choice():
    # After first jump the man can continue two different ways; both complete
    # sequences must be offered (any may be chosen - no maximum-capture rule).
    m2 = (1 << sq(5, 2)) | (1 << sq(3, 2)) | (1 << sq(3, 4))
    p = _pos(m1=1 << sq(6, 1), m2=m2)
    moves = p.legal_moves()
    tos = {m.to for m in moves}
    assert tos == {sq(2, 1), sq(2, 5)}
    assert all(len(m.path) == 3 for m in moves)


def test_crowning_ends_move():
    # P1 man at (2,1) jumps P2 man at (1,2), landing on crown row (0,3).
    # A further jump over (1,4) would exist for a king, but the move MUST end.
    p = _pos(m1=1 << sq(2, 1), m2=(1 << sq(1, 2)) | (1 << sq(1, 4)))
    moves = p.legal_moves()
    assert len(moves) == 1
    mv = moves[0]
    assert (mv.fr, mv.to) == (sq(2, 1), sq(0, 3))
    assert mv.cap == 1 << sq(1, 2)
    nxt = p.apply(mv)
    assert nxt.k1 == 1 << sq(0, 3)  # crowned
    assert nxt.m1 == 0
    assert nxt.turn == 2


def test_king_must_continue_jumping():
    # Same geometry but the jumper is already a king: it must continue
    # (0,3) -> jump (1,4) -> (2,5), a single double-capture move.
    p = _pos(k1=1 << sq(2, 1), m2=(1 << sq(1, 2)) | (1 << sq(1, 4)))
    moves = p.legal_moves()
    assert len(moves) == 1
    mv = moves[0]
    assert (mv.fr, mv.to) == (sq(2, 1), sq(2, 5))
    assert mv.cap == (1 << sq(1, 2)) | (1 << sq(1, 4))


def test_men_do_not_capture_backwards():
    # P1 man at (4,3) with a P2 man BEHIND it at (5,2): no backward capture,
    # so quiet moves are returned.
    p = _pos(m1=1 << sq(4, 3), m2=1 << sq(5, 2))
    moves = p.legal_moves()
    assert all(m.cap == 0 for m in moves)
    assert {m.to for m in moves} == {sq(3, 2), sq(3, 4)}


def test_captured_piece_blocks_and_cannot_be_jumped_twice():
    # King triangle: captured pieces stay on the board until the move ends.
    # K1 at (4,1); P2 men at (3,2) and (3,4) and (5,2).
    # From (4,1): NE jump over (3,2) to (2,3); then SE over (3,4) to (4,5).
    # From (4,5) a jump back over (3,4) is illegal (already captured).
    p = _pos(k1=1 << sq(4, 1),
             m2=(1 << sq(3, 2)) | (1 << sq(3, 4)) | (1 << sq(5, 2)))
    moves = p.legal_moves()
    # Two sequences exist: NE-then-SE (2 caps) and SW over (5,2) (1 cap).
    by_to = {m.to: m for m in moves}
    assert sq(4, 5) in by_to and by_to[sq(4, 5)].cap == (
        (1 << sq(3, 2)) | (1 << sq(3, 4)))
    assert sq(6, 3) in by_to and by_to[sq(6, 3)].cap == (1 << sq(5, 2))
    assert len(moves) == 2


def test_no_legal_moves_when_blocked():
    # P1 man at (7,0) fully blocked by own pieces: no moves for P1.
    p = _pos(m1=(1 << sq(7, 0)) | (1 << sq(6, 1)) | (1 << sq(5, 0)) | (1 << sq(5, 2)),
             m2=1 << sq(0, 1))
    # P1's man at (5,0),(5,2),(6,1) still have moves; isolate the blocked case:
    blocked = _pos(m1=1 << sq(7, 0), k1=0,
                   m2=(1 << sq(6, 1)) | (1 << sq(5, 0)) | (1 << sq(5, 2)))
    # (7,0) can only go NE to (6,1), occupied by P2 -> but then jump? landing
    # (5,2) occupied by P2 -> blocked. No moves.
    assert blocked.legal_moves() == []
    assert p.legal_moves() != []


def test_halfmove_clock():
    p = Position.initial()
    mv = p.legal_moves()[0]
    p2 = p.apply(mv)
    assert p2.hmc == 0  # man move resets
    # King shuffle increments.
    kp = _pos(k1=1 << sq(4, 3), k2=1 << sq(0, 1), turn=1, hmc=0)
    kmv = [m for m in kp.legal_moves() if m.cap == 0][0]
    kp2 = kp.apply(kmv)
    assert kp2.hmc == 1


def test_apply_capture_removes_pieces():
    p = _pos(m1=1 << sq(4, 3), m2=(1 << sq(3, 2)), turn=1)
    mv = p.legal_moves()[0]
    nxt = p.apply(mv)
    assert nxt.m2 == 0
    assert nxt.m1 == 1 << sq(2, 1)
    assert nxt.hmc == 0
    assert nxt.turn == 2
