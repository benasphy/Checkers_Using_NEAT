"""Tests for the CheckersGame wrapper: draw rules, termination, legacy API."""

from checkers.bitboard import Position, rc_to_sq
from checkers.game import CheckersGame, MAX_HMC_PLIES


def sq(r, c):
    s = rc_to_sq(r, c)
    assert s >= 0
    return s


def _set_position(game, **kw):
    pos = Position(kw.get("m1", 0), kw.get("k1", 0), kw.get("m2", 0),
                   kw.get("k2", 0), kw.get("turn", 1), kw.get("hmc", 0))
    game.position = pos
    game.hash_history = [pos.hash]
    game._hash_counts = {pos.hash: 1}
    game.move_history = []
    return game


def test_initial_setup_matches_legacy_encoding():
    game = CheckersGame()
    arr = game.board.board
    assert (arr == 2).sum() == 12 and (arr == 1).sum() == 12
    assert game.current_player == 1
    assert len(game.get_legal_moves()) == 7


def test_side_with_no_moves_loses():
    game = CheckersGame()
    # P1 man at (7,0) is completely blocked; P1 to move -> P2 wins.
    _set_position(game, m1=1 << sq(7, 0),
                  m2=(1 << sq(6, 1)) | (1 << sq(5, 0)) | (1 << sq(5, 2)),
                  turn=1)
    assert game.is_game_over()
    assert game.get_winner() == 2


def test_no_pieces_is_a_loss():
    game = CheckersGame()
    _set_position(game, m1=0, m2=1 << sq(1, 2), turn=1)
    assert game.is_game_over()
    assert game.get_winner() == 2


def test_threefold_repetition_draw():
    game = CheckersGame()
    _set_position(game, k1=1 << sq(4, 3), k2=1 << sq(0, 1), turn=1)
    # Shuffle both kings back and forth twice -> third occurrence of start.
    seq = [((4, 3), (3, 2)), ((0, 1), (1, 0)),
           ((3, 2), (4, 3)), ((1, 0), (0, 1)),
           ((4, 3), (3, 2)), ((0, 1), (1, 0)),
           ((3, 2), (4, 3)), ((1, 0), (0, 1))]
    for frm, to in seq:
        assert not game.is_game_over()
        mv = game.find_engine_move(frm, to)
        assert mv is not None
        game.make_engine_move(mv)
    assert game.is_game_over()
    assert game.get_winner() == 0
    assert game.draw_reason() == "threefold repetition"


def test_forty_move_rule():
    game = CheckersGame()
    _set_position(game, k1=1 << sq(4, 3), k2=1 << sq(0, 1), turn=1,
                  hmc=MAX_HMC_PLIES)
    assert game.is_game_over()
    assert game.get_winner() == 0
    assert game.draw_reason() == "40-move rule"


def test_legacy_move_tuple_roundtrip():
    game = CheckersGame()
    moves = game.get_legal_moves()
    assert all(len(m) == 5 for m in moves)
    game.make_move(moves[0])
    assert game.current_player == 2
    assert len(game.hash_history) == 2


def test_multijump_exposed_as_single_ui_move():
    game = CheckersGame()
    _set_position(game, m1=1 << sq(6, 1),
                  m2=(1 << sq(5, 2)) | (1 << sq(3, 2)), turn=1)
    moves = game.get_legal_moves()
    assert len(moves) == 1
    fr_r, fr_c, to_r, to_c, caps = moves[0]
    assert (fr_r, fr_c, to_r, to_c) == (6, 1, 2, 1)
    assert set(caps) == {(5, 2), (3, 2)}
    game.make_move(moves[0])
    assert game.material_diff(1) == 100  # one man left vs none
    assert game.is_game_over() and game.get_winner() == 1
