"""Elo rating estimation from game results.

Simple iterative maximum-likelihood fit of the logistic (Elo) model with
draws counted as half a win. Good enough for ladder measurement; anchor the
scale by fixing one player's rating (we anchor ``random`` at 0).
"""

from __future__ import annotations

from collections import defaultdict


def expected(ra: float, rb: float) -> float:
    return 1.0 / (1.0 + 10.0 ** ((rb - ra) / 400.0))


def fit_elo(games, anchor=None, anchor_rating: float = 0.0,
            iterations: int = 400) -> dict:
    """games: iterable of (player_a, player_b, score_a) with score in {0, .5, 1}
    (or any fraction). Returns {player: rating}."""
    games = list(games)
    players = sorted({g[0] for g in games} | {g[1] for g in games})
    ratings = {p: 0.0 for p in players}
    n_games = defaultdict(int)
    for a, b, _s in games:
        n_games[a] += 1
        n_games[b] += 1

    k = 40.0
    for it in range(iterations):
        grad = defaultdict(float)
        for a, b, s in games:
            e = expected(ratings[a], ratings[b])
            grad[a] += s - e
            grad[b] += (1.0 - s) - (1.0 - e)
        for p in players:
            if n_games[p]:
                ratings[p] += k * grad[p] / n_games[p]
        if it and it % 50 == 0:
            k = max(k * 0.7, 2.0)

    if anchor is not None and anchor in ratings:
        shift = anchor_rating - ratings[anchor]
        ratings = {p: r + shift for p, r in ratings.items()}
    return ratings
