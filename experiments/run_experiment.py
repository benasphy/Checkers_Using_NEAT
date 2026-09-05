"""Dry-run-first runner for corrected, provenance-bearing pilot experiments."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai.openings import (OPENING_SUITE_VERSION, get_test_openings,
                         get_training_openings, get_validation_openings)
from ai.persist import atomic_json_dump
from ai.train_path_a import (GATE_GAMES, RESEARCH_PROTOCOL_VERSION,
                             genome_complexity)

VALUE_CONFIG = "neat_value_config.txt"
FIXED_CONFIG = "neat_fixed_config.txt"
FIXED_NONLINEAR_CONFIG = "neat_fixed_nonlinear_config.txt"
DEFAULT_RUN_ROOT = os.path.join("runs", "corrected_pilot_v1")

CONDITIONS = {
    "rq1_topo": {
        "neat": {"config": VALUE_CONFIG, "depth": 4,
                 "parameter_budget": 305, "primary": True},
        "fixed_nonlinear": {"config": FIXED_NONLINEAR_CONFIG, "depth": 4,
                            "parameter_budget": 305, "primary": True},
        "fixed": {"config": FIXED_CONFIG, "depth": 4,
                  "parameter_budget": 37, "primary": False},
    },
    "rq2_amp": {
        "d2": {"config": VALUE_CONFIG, "depth": 2,
               "parameter_budget": 305, "primary": True},
        "d4": {"config": VALUE_CONFIG, "depth": 4,
               "parameter_budget": 305, "primary": True},
        "d6": {"config": VALUE_CONFIG, "depth": 6,
               "parameter_budget": 305, "primary": True},
    },
}


def _sha256(path: str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def selected_conditions(rq: str, include_sensitivity: bool = False) -> dict:
    return {
        name: values for name, values in CONDITIONS[rq].items()
        if values["primary"] or include_sensitivity
    }


def protocol_path(run_root: str, rq: str) -> str:
    return os.path.join(run_root, rq, "protocol_manifest.json")


def load_protocol_manifest(run_root: str, rq: str) -> dict:
    path = protocol_path(run_root, rq)
    if not os.path.exists(path):
        raise ValueError(f"Missing corrected protocol manifest: {path}")
    with open(path, encoding="utf-8") as f:
        manifest = json.load(f)
    if manifest.get("research_protocol") != RESEARCH_PROTOCOL_VERSION:
        raise ValueError(f"Incompatible protocol manifest: {path}")
    return manifest


def validate_cell_manifest(cell_dir: str, expected: dict) -> dict:
    path = os.path.join(cell_dir, "run_manifest.json")
    if not os.path.exists(path):
        raise ValueError(f"Missing run manifest: {path}")
    with open(path, encoding="utf-8") as f:
        manifest = json.load(f)
    params = manifest.get("parameters", {})
    problems = []
    checks = {
        "research_protocol": manifest.get("research_protocol"),
        "config_sha256": manifest.get("config_sha256"),
        "seed": params.get("seed"),
        "depth": params.get("depth"),
        "parameter_budget": params.get("parameter_budget"),
        "fitness_mode": params.get("fitness_mode"),
    }
    wanted = {
        "research_protocol": RESEARCH_PROTOCOL_VERSION,
        "config_sha256": expected["config_sha256"],
        "seed": expected["seed"],
        "depth": expected["depth"],
        "parameter_budget": expected["parameter_budget"],
        "fitness_mode": expected["fitness_mode"],
    }
    for key, value in wanted.items():
        if checks[key] != value:
            problems.append(key)
    if problems:
        raise ValueError(f"Invalid corrected run {cell_dir}: {', '.join(problems)}")
    return manifest


def build_protocol_manifest(args, conditions: dict, overrides: dict) -> dict:
    settings = {
        "generations": args.generations,
        "seeds": args.seeds,
        "peer_rounds": args.peer_rounds,
        "probe_every": args.probe_every,
        "probe_games": args.probe_games,
        "fitness_mode": args.fitness_mode,
        "fitness_openings": args.fitness_openings,
        "candidate_top_k": args.candidate_top_k,
        "candidate_openings": args.candidate_openings,
        "rank_reliability_threshold": args.min_rank_reliability,
        "ladder_min_gap": args.min_ladder_gap,
        "overrides": overrides,
    }
    condition_records = {
        name: {"config": values["config"],
               "config_sha256": _sha256(values["config"]),
               "depth": values["depth"],
               "parameter_budget": values["parameter_budget"]}
        for name, values in conditions.items()
    }
    cells = [
        {"condition": name, "seed": seed, "directory": f"{name}_s{seed}",
         "config_sha256": condition_records[name]["config_sha256"],
         "depth": values["depth"],
         "parameter_budget": values["parameter_budget"],
         "fitness_mode": args.fitness_mode}
        for name, values in conditions.items() for seed in args.seeds
    ]
    return {"schema": 1, "research_protocol": RESEARCH_PROTOCOL_VERSION,
            "opening_suite": OPENING_SUITE_VERSION, "rq": args.rq,
            "conditions": condition_records, "settings": settings,
            "cells": cells}


def estimate_cell_games(pop_size: int, generations: int, peer_rounds: int,
                        fitness_openings: int, candidate_top_k: int,
                        candidate_openings: int, probe_every: int,
                        probe_games: int) -> int:
    peer = (pop_size + pop_size % 2) * peer_rounds * fitness_openings
    later = (pop_size * 2 * fitness_openings +
             candidate_top_k * 2 * candidate_openings + GATE_GAMES)
    probes = sum(1 for generation in range(generations)
                 if generation % probe_every == 0) * 4 * probe_games
    return generations * peer + max(generations - 1, 0) * later + probes


def _parse_overrides(values) -> dict:
    overrides = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Override must be key=value: {value}")
        key, raw = value.split("=", 1)
        try:
            overrides[key] = int(raw)
        except ValueError:
            try:
                overrides[key] = float(raw)
            except ValueError:
                overrides[key] = raw
    protected = {"fitness_mode", "fitness_openings", "candidate_top_k",
                 "candidate_openings", "parameter_budget",
                 "rank_reliability_threshold", "ladder_min_gap"}
    conflict = protected & overrides.keys()
    if conflict:
        raise ValueError("Use dedicated flags for: " + ", ".join(sorted(conflict)))
    return overrides


def _preflight(args, conditions: dict, overrides: dict) -> list[str]:
    from ai.ladder import load_neat_config
    import neat

    if not 2 <= len(set(args.seeds)) <= 3:
        raise ValueError("Corrected pilot requires 2 or 3 distinct seeds")
    if not 25 <= args.generations <= 50:
        raise ValueError("Corrected pilot requires 25 to 50 generations")
    if args.fitness_mode != "strict":
        raise ValueError("Corrected pilot must use strict fitness; shaped is ablation-only")

    suites = [get_training_openings(), get_validation_openings(),
              get_test_openings()]
    hashes = [{opening.position.hash for opening in suite} for suite in suites]
    if any(hashes[i] & hashes[j] for i in range(3) for j in range(i + 1, 3)):
        raise ValueError("Opening suites are not disjoint")

    lines = []
    for name, condition in conditions.items():
        cfg = load_neat_config(condition["config"], fresh=True)
        pop_size = int(overrides.get("pop_size", cfg.pop_size))
        genome = neat.DefaultGenome(0)
        genome.configure_new(cfg.genome_config)
        complexity = genome_complexity(genome)
        if complexity["parameters"] > condition["parameter_budget"]:
            raise ValueError(f"{name} starts above its parameter budget")
        games = estimate_cell_games(
            pop_size, args.generations, args.peer_rounds,
            args.fitness_openings, args.candidate_top_k,
            args.candidate_openings, args.probe_every, args.probe_games)
        lines.append(f"{name}: pop={pop_size}, depth={condition['depth']}, "
                     f"initial_params={complexity['parameters']}, "
                     f"cap={condition['parameter_budget']}, games={games}")
    return lines


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rq", choices=sorted(CONDITIONS), required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--generations", type=int, default=25)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--only", nargs="*", default=None,
                        help="execute only listed <condition>_s<seed> cells")
    parser.add_argument("--run-root", default=DEFAULT_RUN_ROOT)
    parser.add_argument("--include-sensitivity", action="store_true",
                        help="also run the lower-capacity fixed-direct control")
    parser.add_argument("--peer-rounds", type=int, default=2)
    parser.add_argument("--probe-every", type=int, default=5)
    parser.add_argument("--probe-games", type=int, default=8)
    parser.add_argument("--fitness-mode", choices=("strict", "shaped"),
                        default="strict")
    parser.add_argument("--fitness-openings", type=int, default=1)
    parser.add_argument("--candidate-top-k", type=int, default=3)
    parser.add_argument("--candidate-openings", type=int, default=2)
    parser.add_argument("--min-ladder-gap", type=float,
                        help="preregistered minimum Elo gap d1<d2<d4")
    parser.add_argument("--min-rank-reliability", type=float,
                        help="preregistered pairwise rank-agreement threshold")
    parser.add_argument("--extra", nargs="*", default=[])
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--execute", action="store_true",
                      help="run games after preflight (default only prints plan)")
    mode.add_argument("--dry-run", action="store_true",
                      help="explicitly print the plan without running games")
    args = parser.parse_args(argv)

    try:
        overrides = _parse_overrides(args.extra)
        conditions = selected_conditions(args.rq, args.include_sensitivity)
        lines = _preflight(args, conditions, overrides)
    except ValueError as error:
        parser.error(str(error))

    print(f"Corrected pilot {args.rq}: {len(conditions)} conditions x "
          f"{len(set(args.seeds))} seeds x {args.generations} generations")
    for line in lines:
        print(f"  {line}")
    print("  estimates assume every evolving genome remains within the cap")

    if not args.execute:
        print("Dry run only. Supply preregistered thresholds and --execute to run.")
        return
    if args.min_ladder_gap is None or args.min_rank_reliability is None:
        parser.error("--execute requires --min-ladder-gap and --min-rank-reliability")
    if not -1.0 <= args.min_rank_reliability <= 1.0:
        parser.error("--min-rank-reliability must be in [-1, 1]")

    manifest = build_protocol_manifest(args, conditions, overrides)
    manifest_path = protocol_path(args.run_root, args.rq)
    if os.path.exists(manifest_path):
        if load_protocol_manifest(args.run_root, args.rq) != manifest:
            parser.error("Existing protocol manifest differs; choose a new --run-root")
    else:
        atomic_json_dump(manifest, manifest_path)

    from ai.persist import newest_checkpoint
    from ai.train_path_a import run_training

    for cell in manifest["cells"]:
        name = cell["condition"]
        seed = cell["seed"]
        cell_name = cell["directory"]
        if args.only and cell_name not in args.only:
            continue
        condition = conditions[name]
        out_dir = os.path.join(args.run_root, args.rq, cell_name)
        resume = newest_checkpoint(out_dir)
        print(f"\n=== {args.rq} / {cell_name} -> {out_dir} ===", flush=True)
        if resume:
            print(f"    auto-resuming from {resume}", flush=True)
        run_training(
            config_path=condition["config"], generations=args.generations,
            depth=condition["depth"], workers=args.workers, seed=seed,
            out_dir=out_dir, resume=resume, peer_rounds=args.peer_rounds,
            probe_every=args.probe_every, probe_games=args.probe_games,
            fitness_mode=args.fitness_mode,
            fitness_openings=args.fitness_openings,
            candidate_top_k=args.candidate_top_k,
            candidate_openings=args.candidate_openings,
            parameter_budget=condition["parameter_budget"],
            rank_reliability_threshold=args.min_rank_reliability,
            ladder_min_gap=args.min_ladder_gap, **overrides)


if __name__ == "__main__":
    main()
