"""Evaluation function compiled from a NEAT genome.

The network maps the canonical 36-feature encoding (see ``ai.features``) to a
single tanh output in [-1, 1], interpreted as the expected outcome for the
side to move. For search, the value is scaled to ``VALUE_SCALE`` so it lives
comfortably below mate scores.

A small hash-keyed cache avoids re-evaluating transpositions reached through
different move orders (quiescence revisits the same positions frequently).
"""

from __future__ import annotations

import neat

from ai.features import encode

VALUE_SCALE = 600.0
_CACHE_MAX = 400_000


class ValueNetwork:
    def __init__(self, genome, config):
        self.genome = genome
        self._net = neat.nn.FeedForwardNetwork.create(genome, config)
        self._cache: dict[int, float] = {}

    def value(self, pos) -> float:
        """Expected outcome in [-1, 1] from the side to move's perspective."""
        h = pos.hash
        v = self._cache.get(h)
        if v is None:
            raw = self._net.activate(encode(pos))[0]
            v = max(-1.0, min(1.0, raw))
            if len(self._cache) >= _CACHE_MAX:
                self._cache.clear()
            self._cache[h] = v
        return v

    def evaluate(self, pos) -> float:
        """Search-scaled evaluation (side-to-move perspective)."""
        return self.value(pos) * VALUE_SCALE
