"""Tests for features, search behavior, arena, and Elo fitting."""

from checkers.bitboard import NUM_SQUARES, Position, rc_to_sq
from ai.arena import play_match
from ai.elo import fit_elo
from ai.features import encode
from ai.search import Searcher, material_eval


def sq(r, c):
    s = rc_to_sq(r, c)
    assert s >= 0
    return s


def _rot(mask):
    out = 0
    for s in range(NUM_SQUARES):
        if (mask >> s) & 1:
            out |= 1 << (NUM_SQUARES - 1 - s)
    return out


def test_feature_color_symmetry():
    # A position for P1-to-move and its 180-degree color-swapped mirror for
    # P2-to-move must produce identical canonical encodings.
    p1 = Position(m1=(1 << sq(5, 2)) | (1 << sq(6, 1)), k1=1 << sq(2, 3),
                  m2=(1 << sq(1, 2)) | (1 << sq(3, 4)), k2=0, turn=1)
    p2 = Position(m1=_rot(p1.m2), k1=_rot(p1.k2),
                  m2=_rot(p1.m1), k2=_rot(p1.k1), turn=2)
    assert encode(p1) == encode(p2)


def test_feature_initial_balance():
    x = encode(Position.initial())
    assert sum(x[:32]) == 0.0  # symmetric start
    assert x[32] == x[34] == 1.0  # 12/12 men each


def test_search_avoids_hanging_piece():
    # P1 king at (7,2): moving to (6,1) lets the P2 king at (5,0) capture it
    # (losing the game); (6,3) is safe. Depth-2 material search must pick it.
    pos = Position(m1=0, k1=1 << sq(7, 2), m2=0, k2=1 << sq(5, 0), turn=1)
    s = Searcher(material_eval, depth=2, seed=1)
    mv = s.best_move(pos)
    assert (mv.fr, mv.to) == (sq(7, 2), sq(6, 3))


def test_search_prefers_bigger_capture_line():
    # Both a single jump and a double jump are available; the double leaves
    # the opponent with nothing useful, the single hangs a man.
    pos = Position(
        m1=(1 << sq(6, 1)) | (1 << sq(6, 7)),
        k1=0,
        m2=(1 << sq(5, 2)) | (1 << sq(3, 2)) | (1 << sq(5, 6)),
        k2=0, turn=1)
    s = Searcher(material_eval, depth=3, seed=1)
    mv = s.best_move(pos)
    assert bin(mv.cap).count("1") == 2


def test_search_finds_win_over_draw():
    # Sanity: from the initial position search returns a legal move quickly.
    s = Searcher(material_eval, depth=4, seed=0)
    mv = s.best_move(Position.initial())
    assert mv is not None and mv.cap == 0


def test_material_d2_crushes_random():
    strict, records = play_match(("material", 2), ("random",),
                                 games=8, base_seed=42)
    assert strict >= 6.0, f"material-d2 scored only {strict}/8 vs random"
    assert all(rec["plies"] > 0 for rec in records)


def test_elo_fit_direction_and_scale():
    games = [("A", "B", 1.0)] * 9 + [("A", "B", 0.0)]
    ratings = fit_elo(games, anchor="B", anchor_rating=0.0)
    diff = ratings["A"] - ratings["B"]
    assert 280 < diff < 500, f"unexpected Elo gap {diff}"
