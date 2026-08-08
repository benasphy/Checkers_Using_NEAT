"""Seeded, color-balanced match running (serial or across processes).

Every pairing is played an even number of games with colors swapped and
per-game seeds derived from a base seed, so experiments are reproducible.

Games truncated at ``max_plies`` are adjudicated: a material edge of at least
3 men decides the game, otherwise it is a draw. Two result scales are
reported for player 1:

- ``p1_strict``: 1 / 0.5 / 0   (used for Elo measurement and HOF gating)
- ``p1_shaped``: draws become ``0.5 + 0.1*tanh(mat/300)`` (in (0.4, 0.6)),
  a bounded tie-breaker that keeps W > any draw > L while giving early,
  weak populations a selection gradient. Used for fitness only.
"""

from __future__ import annotations

import math

from checkers.game import CheckersGame
from ai.ladder import make_agent

DEFAULT_MAX_PLIES = 150
ADJUDICATE_MATERIAL = 300  # centipawns (3 men)


def play_game(agent_p1, agent_p2, max_plies: int = DEFAULT_MAX_PLIES) -> dict:
    game = CheckersGame()
    plies = 0
    while plies < max_plies and not game.is_game_over():
        agent = agent_p1 if game.current_player == 1 else agent_p2
        mv = agent.select(game)
        if mv is None:
            break
        game.make_engine_move(mv)
        plies += 1

    winner = game.get_winner()
    mat = game.material_diff(1)
    reason = game.draw_reason() or ("natural" if winner is not None else "adjudicated")
    if winner is None:  # truncated -> adjudicate
        if mat >= ADJUDICATE_MATERIAL:
            winner = 1
        elif mat <= -ADJUDICATE_MATERIAL:
            winner = 2
        else:
            winner = 0

    strict = 1.0 if winner == 1 else (0.0 if winner == 2 else 0.5)
    shaped = strict if winner != 0 else 0.5 + 0.1 * math.tanh(mat / 300.0)
    return {"winner": winner, "p1_strict": strict, "p1_shaped": shaped,
            "plies": plies, "mat": mat, "reason": reason}


def play_task(task):
    """Worker entry point (top level: picklable for ProcessPoolExecutor).

    task = (spec_p1, spec_p2, seed, max_plies, meta); meta is passed through.
    """
    spec_p1, spec_p2, seed, max_plies, meta = task
    a1 = make_agent(spec_p1, seed=(seed * 2 + 1) & 0xFFFFFFFF)
    a2 = make_agent(spec_p2, seed=(seed * 2 + 2) & 0xFFFFFFFF)
    rec = play_game(a1, a2, max_plies)
    rec["meta"] = meta
    return rec


def pair_tasks(spec_a, spec_b, games: int, base_seed: int,
               max_plies: int = DEFAULT_MAX_PLIES, meta=None):
    """Build ``games`` tasks alternating colors (A starts as player 1)."""
    tasks = []
    for g in range(games):
        seed = (base_seed * 1_000_003 + g) & 0x7FFFFFFF
        if g % 2 == 0:
            tasks.append(((spec_a, spec_b, seed, max_plies,
                           {"a_is_p1": True, **(meta or {})})))
        else:
            tasks.append(((spec_b, spec_a, seed, max_plies,
                           {"a_is_p1": False, **(meta or {})})))
    return tasks


def run_tasks(tasks, executor=None, chunksize: int = 1):
    if executor is None:
        return [play_task(t) for t in tasks]
    return list(executor.map(play_task, tasks, chunksize=chunksize))


def score_for_a(records) -> tuple[float, float]:
    """Aggregate (strict, shaped) points for side A over pair_tasks records."""
    strict = shaped = 0.0
    for rec in records:
        if rec["meta"]["a_is_p1"]:
            strict += rec["p1_strict"]
            shaped += rec["p1_shaped"]
        else:
            strict += 1.0 - rec["p1_strict"]
            shaped += 1.0 - rec["p1_shaped"]
    return strict, shaped


def play_match(spec_a, spec_b, games: int, base_seed: int, executor=None,
               max_plies: int = DEFAULT_MAX_PLIES):
    """Play a color-balanced match; return (strict_points_A, records)."""
    tasks = pair_tasks(spec_a, spec_b, games, base_seed, max_plies)
    records = run_tasks(tasks, executor)
    strict, _ = score_for_a(records)
    return strict, records
