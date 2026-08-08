"""Playing agents and the fixed reference-opponent ladder.

Agents are constructed from picklable *specs* so they can be shipped to
worker processes:

- ``('random',)``                                   uniform random legal move
- ``('material', depth)``                           alpha-beta, material-only eval
- ``('neat', genome_bytes, config_path, depth)``    alpha-beta, evolved value net

The ladder (Random, Material-1/2/4/6) is used for MEASUREMENT ONLY - never
for fitness - so the evolved players cannot overfit their benchmark.
"""

from __future__ import annotations

import pickle
import random

import neat

from ai.search import Searcher, material_eval
from ai.value_net import ValueNetwork

_CONFIG_CACHE: dict[str, neat.Config] = {}


def load_neat_config(path: str) -> neat.Config:
    cfg = _CONFIG_CACHE.get(path)
    if cfg is None:
        cfg = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                          neat.DefaultSpeciesSet, neat.DefaultStagnation, path)
        _CONFIG_CACHE[path] = cfg
    return cfg


class RandomAgent:
    name = "random"

    def __init__(self, seed=None):
        self.rng = random.Random(seed)

    def select(self, game):
        moves = game.engine_moves()
        return self.rng.choice(moves) if moves else None


class SearchAgent:
    def __init__(self, eval_fn, depth, seed=None, name="search"):
        self.searcher = Searcher(eval_fn, depth=depth, seed=seed)
        self.depth = depth
        self.name = name

    def select(self, game, max_seconds=None):
        return self.searcher.best_move(game, max_seconds=max_seconds)


def make_agent(spec: tuple, seed=None):
    kind = spec[0]
    if kind == "random":
        return RandomAgent(seed)
    if kind == "material":
        depth = spec[1]
        return SearchAgent(material_eval, depth, seed, name=f"material-d{depth}")
    if kind == "neat":
        _, genome_bytes, config_path, depth = spec
        genome = pickle.loads(genome_bytes)
        config = load_neat_config(config_path)
        net = ValueNetwork(genome, config)
        return SearchAgent(net.evaluate, depth, seed, name=f"neat-d{depth}")
    raise ValueError(f"Unknown agent spec: {spec!r}")


def neat_spec(genome, config_path: str, depth: int) -> tuple:
    return ("neat", pickle.dumps(genome), config_path, depth)


# Measurement ladder: name -> spec. Material-d1 is the classic greedy player.
LADDER = [
    ("random", ("random",)),
    ("material-d1", ("material", 1)),
    ("material-d2", ("material", 2)),
    ("material-d4", ("material", 4)),
    ("material-d6", ("material", 6)),
]
