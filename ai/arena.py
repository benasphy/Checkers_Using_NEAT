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

import hashlib
import math
import random
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from checkers.game import CheckersGame
from ai.ladder import make_agent

DEFAULT_MAX_PLIES = 150
ADJUDICATE_MATERIAL = 300  # centipawns (3 men)
EVALUATION_PROTOCOL_VERSION = "paired-strict-v1"


@dataclass(frozen=True)
class MatchTask:
    spec_p1: tuple
    spec_p2: tuple
    seed: int
    max_plies: int
    meta: dict
    opening_plies: int = 0
    initial_pos: object | None = None
    initial_moves: tuple | None = None
    adjudicate: bool = True


@lru_cache(maxsize=None)
def _config_descriptor(config_path: str) -> dict:
    path = Path(config_path).resolve()
    return {"name": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def _agent_descriptor(spec: tuple) -> dict:
    if spec[0] == "neat":
        return {"kind": "neat", "genome_sha256": hashlib.sha256(spec[1]).hexdigest(),
                "config": _config_descriptor(spec[2]), "depth": spec[3]}
    if spec[0] == "material":
        return {"kind": "material", "depth": spec[1], "version": "material-100-140-v1"}
    return {"kind": spec[0], "version": "v1"}


def _compute_descriptor(agent) -> dict:
    searcher = getattr(agent, "searcher", None)
    return {
        "search_nodes": getattr(searcher, "nodes", 0),
        "network_evaluations": getattr(searcher, "evaluations", 0),
    }


def play_game(agent_p1, agent_p2, max_plies: int = DEFAULT_MAX_PLIES,
              opening_plies: int = 0, opening_seed: int | None = None,
              initial_pos=None, initial_moves=None, adjudicate: bool = True) -> dict:
    game = CheckersGame(initial_pos=initial_pos, initial_moves=initial_moves)
    plies = len(game.move_history)
    if opening_plies > 0:
        rng = random.Random(opening_seed)
        for _ in range(opening_plies):
            if game.is_game_over() or plies >= max_plies:
                break
            moves = game.engine_moves()
            if not moves:
                break
            mv = rng.choice(moves)
            game.make_engine_move(mv)
            plies += 1

    while plies < max_plies and not game.is_game_over():
        agent = agent_p1 if game.current_player == 1 else agent_p2
        mv = agent.select(game)
        if mv is None:
            break
        game.make_engine_move(mv)
        plies += 1

    winner = game.get_winner()
    mat = game.material_diff(1)
    reason = game.draw_reason() or ("natural" if winner is not None else "max_plies")
    if winner is None:
        if adjudicate and mat >= ADJUDICATE_MATERIAL:
            winner, reason = 1, "material_adjudication"
        elif adjudicate and mat <= -ADJUDICATE_MATERIAL:
            winner, reason = 2, "material_adjudication"
        else:
            winner = 0

    strict = 1.0 if winner == 1 else (0.0 if winner == 2 else 0.5)
    shaped = strict if winner != 0 else 0.5 + 0.1 * math.tanh(mat / 300.0)
    moves = [{"fr": mv.fr, "to": mv.to, "cap": mv.cap, "path": list(mv.path)}
             for mv in game.move_history]
    pos = game.position
    return {"winner": winner, "p1_strict": strict, "p1_shaped": shaped,
            "plies": plies, "mat": mat, "reason": reason, "moves": moves,
            "final_position": {"m1": pos.m1, "k1": pos.k1, "m2": pos.m2,
                               "k2": pos.k2, "turn": pos.turn, "hmc": pos.hmc,
                               "hash": pos.hash}}


def play_task(task: MatchTask):
    """Worker entry point (top level and dataclass are process-picklable)."""
    a1 = make_agent(task.spec_p1, seed=(task.seed * 2 + 1) & 0xFFFFFFFF)
    a2 = make_agent(task.spec_p2, seed=(task.seed * 2 + 2) & 0xFFFFFFFF)
    rec = play_game(a1, a2, max_plies=task.max_plies,
                    opening_plies=task.opening_plies,
                    opening_seed=task.seed, initial_pos=task.initial_pos,
                    initial_moves=task.initial_moves,
                    adjudicate=task.adjudicate)
    rec["meta"] = task.meta
    rec["protocol"] = EVALUATION_PROTOCOL_VERSION
    rec["seed"] = task.seed
    rec["max_plies"] = task.max_plies
    rec["adjudicate"] = task.adjudicate
    rec["agent_p1"] = _agent_descriptor(task.spec_p1)
    rec["agent_p2"] = _agent_descriptor(task.spec_p2)
    rec["compute_p1"] = _compute_descriptor(a1)
    rec["compute_p2"] = _compute_descriptor(a2)
    return rec


def pair_tasks(spec_a, spec_b, games: int, base_seed: int,
               max_plies: int = DEFAULT_MAX_PLIES, meta=None,
               opening_plies: int = 0, adjudicate: bool = True):
    """Build ``games`` tasks alternating colors (A starts as player 1)."""
    if games <= 0 or games % 2:
        raise ValueError("games must be a positive even number for color pairing")
    tasks = []
    for g in range(games):
        seed = (base_seed * 1_000_003 + g // 2) & 0x7FFFFFFF
        if g % 2 == 0:
            tasks.append(MatchTask(
                spec_a, spec_b, seed, max_plies,
                {"a_is_p1": True, **(meta or {})}, opening_plies,
                adjudicate=adjudicate))
        else:
            tasks.append(MatchTask(
                spec_b, spec_a, seed, max_plies,
                {"a_is_p1": False, **(meta or {})}, opening_plies,
                adjudicate=adjudicate))
    return tasks


def pair_opening_tasks(spec_a, spec_b, openings, base_seed: int,
                       max_plies: int = DEFAULT_MAX_PLIES, meta=None,
                       adjudicate: bool = False):
    """Build 2*len(openings) tasks: each opening position played twice with colors swapped."""
    tasks = []
    for i, op in enumerate(openings):
        seed = (base_seed * 1_000_003 + i) & 0x7FFFFFFF
        from ai.openings import OPENING_SUITE_VERSION
        op_meta = {"opening_id": op.id, "opening_name": op.name,
                   "opening_suite": OPENING_SUITE_VERSION, **(meta or {})}
        # Game 1: A is player 1
        tasks.append(MatchTask(
            spec_a, spec_b, seed, max_plies,
            {"a_is_p1": True, **op_meta}, initial_pos=op.position,
            initial_moves=op.move_history, adjudicate=adjudicate))
        # Game 2: B is player 1 (A is player 2)
        tasks.append(MatchTask(
            spec_b, spec_a, seed, max_plies,
            {"a_is_p1": False, **op_meta}, initial_pos=op.position,
            initial_moves=op.move_history, adjudicate=adjudicate))
    return tasks


def append_raw_records(records: list[dict], filepath: str):
    """Durably append structured game records to a JSONL file."""
    import json
    import os
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    with open(filepath, "a", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
        f.flush()
        os.fsync(f.fileno())


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
               max_plies: int = DEFAULT_MAX_PLIES, opening_plies: int = 0,
               adjudicate: bool = True):
    """Play a color-balanced match; return (strict_points_A, records)."""
    tasks = pair_tasks(spec_a, spec_b, games, base_seed, max_plies,
                       opening_plies=opening_plies, adjudicate=adjudicate)
    records = run_tasks(tasks, executor)
    strict, _ = score_for_a(records)
    return strict, records
