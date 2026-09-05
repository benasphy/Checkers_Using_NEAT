"""Path A training: evolve a NEAT value network used inside alpha-beta search.

Design (see README / paper plan):

- ONE population. Each genome is a value network; the playing agent is
  alpha-beta search (fixed depth during training) using that network at the
  leaves.
- Fitness defaults to strict W/D/L from color-balanced games against peers
  and the hall of fame. Material-shaped draws are available only as an
  explicit ablation.
- Hall of fame is Elo-gated: the generation's best genome enters only by
  scoring >= 55% in a strict color-balanced gate match against HOF members.
- Strength is MEASURED (never trained on) against a fixed ladder of
  material-only alpha-beta players, reported as a performance Elo anchored at
  random = 0. The ladder is calibrated once and cached.

Everything is seeded and logged to CSV for reproducibility.
"""

from __future__ import annotations

import copy
import csv
import gzip
import hashlib
import importlib.metadata
import json
import os
import pickle
import platform
import random
import time
from concurrent.futures import ProcessPoolExecutor
from itertools import count
from pathlib import Path
from statistics import mean

import neat

from ai.arena import append_raw_records, pair_opening_tasks, run_tasks, score_for_a
from ai.elo import expected, fit_elo
from ai.ladder import LADDER, load_neat_config, neat_spec
from ai.openings import (OPENING_SUITE_VERSION, get_training_openings,
                         get_validation_openings)
from ai.persist import atomic_json_dump, atomic_pickle_dump, atomic_pickle_dump_gzip

HOF_MAX = 12
GATE_GAMES = 12
GATE_THRESHOLD = 0.55
PROBE_RUNGS = ["random", "material-d1", "material-d2", "material-d4"]
CHECKPOINT_SCHEMA = 2
RESEARCH_PROTOCOL_VERSION = "corrected-pilot-v1"
LADDER_CALIBRATION_VERSION = "validation-paired-strict-v2"
LADDER_CACHE_PATH = os.path.join(
    "runs", "evaluation", "ladder_calibration_paired_strict_v2.json")


# ---------------------------------------------------------------------------
# Ladder calibration (once, cached): fixed Elo for the reference opponents.
# ---------------------------------------------------------------------------

def ladder_calibration_manifest(seed: int = 7, games_per_pair: int = 32,
                                max_plies: int = 150) -> dict:
    if games_per_pair <= 0 or games_per_pair > 32 or games_per_pair % 2:
        raise ValueError("games_per_pair must be an even number from 2 to 32")
    openings = get_validation_openings(games_per_pair // 2)
    rungs = [(name, spec) for name, spec in LADDER if name in PROBE_RUNGS]
    return {
        "version": LADDER_CALIBRATION_VERSION,
        "game_protocol": "paired-strict-v1",
        "seed": seed,
        "games_per_pair": games_per_pair,
        "max_plies": max_plies,
        "adjudicate": False,
        "opening_suite": OPENING_SUITE_VERSION,
        "opening_ids": [opening.id for opening in openings],
        "rungs": [{"name": name, "spec": list(spec)} for name, spec in rungs],
    }


def calibrate_ladder(executor, cache_path: str, seed: int = 7,
                     games_per_pair: int = 32, max_plies: int = 150) -> dict:
    expected_manifest = ladder_calibration_manifest(
        seed=seed, games_per_pair=games_per_pair, max_plies=max_plies)
    if os.path.exists(cache_path):
        with open(cache_path, encoding="utf-8") as f:
            cached = json.load(f)
        if cached.get("manifest") != expected_manifest:
            raise ValueError(f"Calibration manifest mismatch: {cache_path}")
        return cached["ratings"]
    print("Calibrating reference ladder (one-off)...")
    rungs = [(n, s) for n, s in LADDER if n in PROBE_RUNGS]
    openings = get_validation_openings(games_per_pair // 2)
    tasks = []
    for i, (na, sa) in enumerate(rungs):
        for j, (nb, sb) in enumerate(rungs):
            if i < j:
                tasks += pair_opening_tasks(
                    sa, sb, openings, base_seed=seed + 97 * i + j,
                    max_plies=max_plies,
                    meta={"pair": (na, nb), "purpose": "ladder_calibration"})
    records = run_tasks(tasks, executor)
    games = []
    for rec in records:
        na, nb = rec["meta"]["pair"]
        s_a = rec["p1_strict"] if rec["meta"]["a_is_p1"] else 1.0 - rec["p1_strict"]
        games.append((na, nb, s_a))
    ratings = fit_elo(games, anchor="random", anchor_rating=0.0)
    atomic_json_dump({"manifest": expected_manifest, "ratings": ratings,
                      "games": records}, cache_path)
    print("Ladder Elo:", {k: round(v) for k, v in ratings.items()})
    return ratings


def require_monotonic_ladder(ratings: dict, min_gap: float = 0.0) -> None:
    ordered = [ratings[name] for name in ("material-d1", "material-d2",
                                          "material-d4")]
    gaps = [b - a for a, b in zip(ordered, ordered[1:])]
    if any(gap < min_gap for gap in gaps):
        raise ValueError(
            f"Ladder stability gate failed: gaps={gaps}, required>={min_gap}")


def performance_rating(results, opp_ratings, lo=-1500.0, hi=4000.0) -> float:
    """Rating R such that expected total score vs rated opponents matches the
    actual total. results: list of (opp_name, score_total, n_games)."""
    actual = sum(s for _, s, _ in results)
    n = sum(g for _, _, g in results)
    if n == 0:
        return 0.0
    actual = min(max(actual, 0.25), n - 0.25)  # avoid +/- infinity

    def diff(r):
        return sum(expected(r, opp_ratings[o]) * g for o, _, g in results) - actual

    for _ in range(80):
        mid = (lo + hi) / 2.0
        if diff(mid) < 0:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_run_manifest(config_path: str, **parameters) -> dict:
    """Fingerprint every input that can change a resumed evolutionary run."""
    root = Path(__file__).resolve().parent.parent
    source = hashlib.sha256()
    for directory in (root / "ai", root / "checkers"):
        for path in sorted(directory.glob("*.py")):
            source.update(str(path.relative_to(root)).encode())
            source.update(b"\0")
            source.update(path.read_bytes())
    config_file = Path(config_path).resolve()
    return {
        "schema": CHECKPOINT_SCHEMA,
        "research_protocol": RESEARCH_PROTOCOL_VERSION,
        "opening_suite": OPENING_SUITE_VERSION,
        "config_name": config_file.name,
        "config_sha256": _file_sha256(config_file),
        "source_sha256": source.hexdigest(),
        "python": platform.python_version(),
        "neat_python": importlib.metadata.version("neat-python"),
        "parameters": parameters,
    }


def _require_same_manifest(saved: dict, expected: dict) -> None:
    if saved != expected:
        changed = sorted(set(saved) | set(expected),
                         key=str)
        changed = [key for key in changed if saved.get(key) != expected.get(key)]
        raise ValueError("Checkpoint manifest mismatch: " + ", ".join(changed))


def _restore_log_offsets(out_dir: str, offsets: dict[str, int]) -> None:
    supported = {"training_metrics.csv", "raw_training_games.jsonl",
                 "raw_gate_games.jsonl", "raw_probe_games.jsonl"}
    for name, offset in offsets.items():
        if name not in supported:
            raise ValueError(f"Unsupported checkpoint log path: {name}")
        path = os.path.join(out_dir, name)
        size = os.path.getsize(path) if os.path.exists(path) else 0
        if size < offset:
            raise ValueError(f"Log {name} is shorter than its checkpoint offset")
        if size > offset:
            with open(path, "r+b") as f:
                f.truncate(offset)


def _flush_and_fsync(file_obj) -> None:
    file_obj.flush()
    os.fsync(file_obj.fileno())


def genome_complexity(genome) -> dict[str, int]:
    connections = sum(1 for gene in genome.connections.values() if gene.enabled)
    nodes = len(genome.nodes)
    return {"nodes": nodes, "connections": connections,
            "parameters": nodes + connections}


def aggregate_fitness(records, genome_ids, mode: str = "strict"):
    if mode not in {"strict", "shaped"}:
        raise ValueError("fitness mode must be 'strict' or 'shaped'")
    score_key = "p1_strict" if mode == "strict" else "p1_shaped"
    points = {gid: 0.0 for gid in genome_ids}
    played = {gid: 0 for gid in genome_ids}
    for record in records:
        meta = record["meta"]
        p1_score = record[score_key]
        a_score = p1_score if meta["a_is_p1"] else 1.0 - p1_score
        points[meta["a"]] += a_score
        played[meta["a"]] += 1
        if meta.get("type") == "peer":
            points[meta["b"]] += 1.0 - a_score
            played[meta["b"]] += 1
    return points, played


def pairwise_rank_reliability(first: dict, second: dict, genome_ids) -> float:
    """Pairwise rank agreement in [-1, 1], ignoring ties."""
    concordant = discordant = 0
    ids = list(genome_ids)
    for i, left in enumerate(ids):
        for right in ids[i + 1:]:
            a = first[left] - first[right]
            b = second[left] - second[right]
            if a == 0 or b == 0:
                continue
            if (a > 0) == (b > 0):
                concordant += 1
            else:
                discordant += 1
    compared = concordant + discordant
    return (concordant - discordant) / compared if compared else 0.0


def _genome_sha256(genome) -> str:
    return hashlib.sha256(pickle.dumps(genome)).hexdigest()


def _hof_entry(genome, generation: int, gate_score, fitness_mode: str,
               opponents=()) -> dict:
    return {
        "genome": genome,
        "genome_sha256": _genome_sha256(genome),
        "generation": generation,
        "gate_score": gate_score,
        "fitness_mode": fitness_mode,
        "opening_suite": OPENING_SUITE_VERSION,
        "opponents": list(opponents),
    }


def _connection_signature(entry: dict) -> frozenset:
    return frozenset(key for key, gene in entry["genome"].connections.items()
                     if gene.enabled)


def _prune_redundant_hof_entry(hall_of_fame: list[dict]) -> None:
    """Keep first/latest anchors and evict the least topologically distinct middle entry."""
    candidates = range(1, len(hall_of_fame) - 1)
    signatures = [_connection_signature(entry) for entry in hall_of_fame]

    def min_distance(index):
        distances = []
        for other, signature in enumerate(signatures):
            if other == index:
                continue
            union = signatures[index] | signature
            overlap = len(signatures[index] & signature)
            distances.append(1.0 - overlap / max(len(union), 1))
        return min(distances)

    hall_of_fame.pop(min(candidates, key=lambda index: (min_distance(index), index)))


def _compute_totals(records) -> tuple[int, int]:
    nodes = evaluations = 0
    for record in records:
        for side in ("compute_p1", "compute_p2"):
            compute = record.get(side, {})
            nodes += compute.get("search_nodes", 0)
            evaluations += compute.get("network_evaluations", 0)
    return nodes, evaluations


def _take_counter(owner, attribute):
    counter = getattr(owner, attribute)
    if counter is None:
        return None
    next_value = next(counter)
    setattr(owner, attribute, count(next_value))
    return next_value


class AtomicCheckpointer(neat.Checkpointer):
    """Atomic, generation-exact checkpoint including non-NEAT run state."""

    def __init__(self, *args, population_ref, manifest, snapshot_fn, **kwargs):
        super().__init__(*args, **kwargs)
        self.population_ref = population_ref
        self.manifest = manifest
        self.snapshot_fn = snapshot_fn

    def save_checkpoint(self, config, population, species_set, generation):
        # end_generation runs after reproduction, so this population is generation + 1.
        self._save(config, population, species_set, generation + 1)

    def _save(self, config, population, species_set, next_generation):
        filename = f"{self.filename_prefix}{next_generation}"
        reproduction = self.population_ref.reproduction
        species_indexer = _take_counter(species_set, "indexer")
        stored_species = copy.copy(species_set)
        stored_species.indexer = None
        stored_species.reporters = None
        payload = {
            "schema": CHECKPOINT_SCHEMA,
            "generation": next_generation,
            "population": population,
            "species_set": stored_species,
            "species_indexer": species_indexer,
            "node_indexer": _take_counter(config.genome_config, "node_indexer"),
            "random_state": random.getstate(),
            "reproduction_state": {
                "genome_indexer": _take_counter(reproduction, "genome_indexer"),
                "ancestors": reproduction.ancestors,
            },
            "training_state": self.snapshot_fn(),
            "manifest": self.manifest,
        }
        atomic_pickle_dump_gzip(payload, filename)
        print(f"Saved checkpoint to {filename}")

    def save_final(self):
        pop = self.population_ref
        self._save(pop.config, pop.population, pop.species, pop.generation)


def restore_training_checkpoint(filename: str, config, expected_manifest: dict):
    with gzip.open(filename, "rb") as f:
        payload = pickle.load(f)
    if not isinstance(payload, dict) or payload.get("schema") != CHECKPOINT_SCHEMA:
        raise ValueError("Legacy checkpoint lacks complete transactional run state")
    _require_same_manifest(payload["manifest"], expected_manifest)
    node_indexer = payload["node_indexer"]
    config.genome_config.node_indexer = count(node_indexer) if node_indexer is not None else None
    pop = neat.Population(
        config,
        (payload["population"], payload["species_set"], payload["generation"]),
    )
    pop.species.indexer = count(payload["species_indexer"])
    pop.species.reporters = pop.reporters
    reproduction_state = payload["reproduction_state"]
    pop.reproduction.genome_indexer = count(reproduction_state["genome_indexer"])
    pop.reproduction.ancestors = reproduction_state["ancestors"]
    return pop, payload["training_state"], payload["random_state"]


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def build_fitness_tasks(genome_ids, specs, hall_of_fame, rng,
                        peer_rounds: int, openings_per_pair: int,
                        max_plies: int, generation: int):
    """Build a deterministic schedule with shared openings and HOF anchors."""
    opening_pool = get_training_openings()
    tasks = []
    for round_index in range(peer_rounds):
        order = list(genome_ids)
        rng.shuffle(order)
        if len(order) % 2:
            order.append(rng.choice(order[:-1]))
        openings = rng.sample(opening_pool, openings_per_pair)
        for pair_index, (left, right) in enumerate(zip(order[::2], order[1::2])):
            tasks += pair_opening_tasks(
                specs[left], specs[right], openings,
                base_seed=rng.randrange(1 << 30), max_plies=max_plies,
                meta={"type": "peer", "a": left, "b": right,
                      "generation": generation, "round": round_index,
                      "pair": pair_index, "purpose": "fitness"})

    panel = rng.sample(hall_of_fame, 1) if hall_of_fame else []
    for anchor_index, entry in enumerate(panel):
        openings = rng.sample(opening_pool, openings_per_pair)
        shared_seed = rng.randrange(1 << 30)
        anchor_spec = neat_spec(entry["genome"], specs[next(iter(specs))][2],
                                specs[next(iter(specs))][3])
        for gid in genome_ids:
            tasks += pair_opening_tasks(
                specs[gid], anchor_spec, openings, base_seed=shared_seed,
                max_plies=max_plies,
                meta={"type": "hof", "a": gid, "b": None,
                      "generation": generation, "anchor": anchor_index,
                      "anchor_sha256": entry["genome_sha256"],
                      "purpose": "fitness"})
    return tasks, panel


def build_common_panel_tasks(genome_ids, specs, panel, rng,
                             openings_per_opponent: int, max_plies: int,
                             generation: int):
    tasks = []
    opening_pool = get_training_openings()
    for anchor_index, entry in enumerate(panel):
        openings = rng.sample(opening_pool, openings_per_opponent)
        shared_seed = rng.randrange(1 << 30)
        anchor_spec = neat_spec(entry["genome"], specs[next(iter(specs))][2],
                                specs[next(iter(specs))][3])
        for gid in genome_ids:
            tasks += pair_opening_tasks(
                specs[gid], anchor_spec, openings, base_seed=shared_seed,
                max_plies=max_plies,
                meta={"type": "candidate_panel", "a": gid, "b": None,
                      "generation": generation, "anchor": anchor_index,
                      "anchor_sha256": entry["genome_sha256"],
                      "purpose": "candidate_selection"})
    return tasks


def run_training(config_path: str, generations: int = 200, depth: int = 4,
                  workers=None, seed: int = 0, peer_rounds: int = 2,
                  probe_every: int = 5, probe_games: int = 8,
                  out_dir: str = "runs/path_a", pop_size=None,
                  max_plies: int = 150, resume: str = None,
                  fitness_mode: str = "strict", fitness_openings: int = 1,
                  candidate_top_k: int = 3, candidate_openings: int = 2,
                  parameter_budget: int | None = None,
                  rank_reliability_threshold: float | None = None,
                  ladder_min_gap: float | None = None,
                  publish_web_genome: bool = False):
    if generations <= 0 or peer_rounds <= 0 or probe_every <= 0:
        raise ValueError("generations, peer_rounds, and probe_every must be positive")
    if probe_games <= 0 or probe_games > 32 or probe_games % 2:
        raise ValueError("probe_games must be an even number from 2 to 32")
    if fitness_mode not in {"strict", "shaped"}:
        raise ValueError("fitness_mode must be 'strict' or 'shaped'")
    if not 1 <= fitness_openings <= len(get_training_openings()):
        raise ValueError("fitness_openings is outside the training suite")
    if not 1 <= candidate_openings <= len(get_training_openings()):
        raise ValueError("candidate_openings is outside the training suite")
    if candidate_top_k <= 0:
        raise ValueError("candidate_top_k must be positive")
    if parameter_budget is not None and parameter_budget <= 0:
        raise ValueError("parameter_budget must be positive")

    os.makedirs(out_dir, exist_ok=True)
    config = load_neat_config(config_path, fresh=True)
    if pop_size is not None:
        config.pop_size = pop_size
    manifest = build_run_manifest(
        config_path, seed=seed, depth=depth, pop_size=config.pop_size,
        peer_rounds=peer_rounds, probe_every=probe_every,
        probe_games=probe_games, max_plies=max_plies,
        fitness_mode=fitness_mode, fitness_openings=fitness_openings,
        candidate_top_k=candidate_top_k, candidate_openings=candidate_openings,
        parameter_budget=parameter_budget,
        rank_reliability_threshold=rank_reliability_threshold,
        ladder_min_gap=ladder_min_gap, gate_games=GATE_GAMES,
        gate_threshold=GATE_THRESHOLD,
    )
    manifest_path = os.path.join(out_dir, "run_manifest.json")

    if resume:
        if not os.path.exists(manifest_path):
            raise ValueError("Cannot resume without run_manifest.json")
        with open(manifest_path, encoding="utf-8") as f:
            _require_same_manifest(json.load(f), manifest)
        pop, restored, evolution_random_state = restore_training_checkpoint(
            resume, config, manifest)
        _restore_log_offsets(out_dir, restored["log_offsets"])
        print(f"Resumed from {resume} at generation {pop.generation}.")
    else:
        occupied = os.path.exists(manifest_path) or any(
            name.startswith("neat-checkpoint-") for name in os.listdir(out_dir)
        )
        if occupied or os.path.exists(os.path.join(out_dir, "training_metrics.csv")):
            raise ValueError("Run directory already contains state; resume it or use --fresh")
        random.seed(seed)
        pop = neat.Population(config)
        evolution_random_state = random.getstate()
        restored = None

    cur_gen = getattr(pop, "generation", 0) or 0
    if resume and cur_gen >= generations:
        print(f"Population is already at generation {cur_gen} >= target {generations}. Nothing to run.")
        best_path = os.path.join(out_dir, "best_value_genome.pkl")
        if os.path.exists(best_path):
            with open(best_path, "rb") as f:
                return pickle.load(f)
        return max(pop.population.values(), key=lambda g: getattr(g, "fitness", 0.0) or 0.0)

    n_gens_to_run = (generations - cur_gen) if resume else generations

    executor = ProcessPoolExecutor(max_workers=workers)
    try:
        ladder_elo = calibrate_ladder(executor, LADDER_CACHE_PATH,
                                      max_plies=max_plies)
        if ladder_min_gap is not None:
            require_monotonic_ladder(ladder_elo, ladder_min_gap)
    except BaseException:
        executor.shutdown()
        raise
    if not resume:
        atomic_json_dump(manifest, manifest_path)

    hof_path = os.path.join(out_dir, "hof.pkl")
    hall_of_fame: list = restored["hall_of_fame"] if restored else []
    if restored:
        print(f"Loaded hall of fame with {len(hall_of_fame)} members.")

    metrics_path = os.path.join(out_dir, "training_metrics.csv")
    new_csv = not os.path.exists(metrics_path)
    csv_file = open(metrics_path, "a", newline="")
    writer = csv.writer(csv_file)
    if new_csv:
        writer.writerow(["gen", "best_fitness", "candidate_fitness",
                         "mean_fitness", "species", "games",
                         "candidate_games", "avg_plies", "draw_rate",
                         "fitness_mode", "eligible", "rank_reliability",
                         "rank_gate_pass", "gate", "hof_size", "nodes",
                         "conns", "parameters", "mean_nodes", "mean_conns",
                         "mean_parameters", "fitness_search_nodes",
                         "fitness_network_evaluations", "elapsed_s", "elo"] +
                        [f"wr_{r}" for r in PROBE_RUNGS])
        _flush_and_fsync(csv_file)
    rung_specs = dict(LADDER)

    state = restored["state"] if restored else {
        "best_ever": None,
        "best_ever_fitness": -1.0,
        "best_ever_elo": -9999.0,
        "best_ever_elo_genome": None,
        "latest_best": None,
    }

    def checkpoint_snapshot():
        _flush_and_fsync(csv_file)
        offsets = {}
        for name in ("training_metrics.csv", "raw_training_games.jsonl",
                     "raw_gate_games.jsonl", "raw_probe_games.jsonl"):
            path = os.path.join(out_dir, name)
            offsets[name] = os.path.getsize(path) if os.path.exists(path) else 0
        return {"state": state, "hall_of_fame": hall_of_fame,
                "log_offsets": offsets}

    pop.add_reporter(neat.StdOutReporter(True))
    pop.add_reporter(neat.StatisticsReporter())
    checkpointer = AtomicCheckpointer(
        generation_interval=5, time_interval_seconds=None,
        filename_prefix=os.path.join(out_dir, "neat-checkpoint-"),
        population_ref=pop, manifest=manifest, snapshot_fn=checkpoint_snapshot)
    pop.add_reporter(checkpointer)

    def eval_genomes(genomes, cfg):
        gen = pop.generation
        t0 = time.time()
        gen_rng = random.Random((seed << 20) ^ gen)

        gids = [gid for gid, _ in genomes]
        genome_by_id = dict(genomes)
        complexities = {gid: genome_complexity(genome)
                        for gid, genome in genomes}
        eligible = [gid for gid in gids if parameter_budget is None or
                    complexities[gid]["parameters"] <= parameter_budget]
        if len(eligible) < 2:
            raise RuntimeError("Fewer than two genomes satisfy the parameter budget")
        specs = {gid: neat_spec(genome_by_id[gid], config_path, depth)
                 for gid in eligible}

        # ---- build tasks: peer pairings + hall-of-fame games ----------------
        tasks, panel = build_fitness_tasks(
            eligible, specs, hall_of_fame, gen_rng, peer_rounds,
            fitness_openings, max_plies, gen)
        records = run_tasks(tasks, executor, chunksize=4)

        # ---- aggregate explicitly selected fitness scale ---------------------
        points, played = aggregate_fitness(records, eligible, fitness_mode)
        for gid, genome in genomes:
            genome.fitness = points.get(gid, 0.0) / max(played.get(gid, 0), 1)

        ranked = sorted(eligible,
                        key=lambda gid: (-genome_by_id[gid].fitness, gid))
        best_fit = genome_by_id[ranked[0]].fitness

        # Re-rank noisy top genomes against an identical historical panel.
        candidate_ids = ranked[:min(candidate_top_k, len(ranked))]
        candidate_gid = candidate_ids[0]
        candidate_records = []
        reliability = None
        if panel and len(candidate_ids) > 1:
            candidate_tasks = build_common_panel_tasks(
                candidate_ids, specs, panel, gen_rng, candidate_openings,
                max_plies, gen)
            candidate_records = run_tasks(candidate_tasks, executor, chunksize=2)
            candidate_points, candidate_played = aggregate_fitness(
                candidate_records, candidate_ids, fitness_mode)
            candidate_scores = {
                gid: candidate_points[gid] / max(candidate_played[gid], 1)
                for gid in candidate_ids
            }
            initial_scores = {gid: genome_by_id[gid].fitness
                              for gid in candidate_ids}
            reliability = pairwise_rank_reliability(
                initial_scores, candidate_scores, candidate_ids)
            candidate_gid = max(
                candidate_ids,
                key=lambda gid: (candidate_scores[gid],
                                 genome_by_id[gid].fitness, -gid))

        candidate = pickle.loads(pickle.dumps(genome_by_id[candidate_gid]))
        candidate_fit = genome_by_id[candidate_gid].fitness
        state["latest_best"] = candidate
        append_raw_records(records + candidate_records,
                           os.path.join(out_dir, "raw_training_games.jsonl"))

        # ---- Elo-gated hall of fame ------------------------------------------
        gate_score = ""
        gate_records = []
        if not hall_of_fame:
            hall_of_fame.append(_hof_entry(
                candidate, gen, "seed", fitness_mode))
            gate_score = "seed"
            atomic_pickle_dump(hall_of_fame, hof_path)
        else:
            opponents = gen_rng.sample(hall_of_fame,
                                       min(3, len(hall_of_fame)))
            gtasks = []
            openings_per_opponent = GATE_GAMES // (2 * len(opponents))
            for index, entry in enumerate(opponents):
                gate_openings = gen_rng.sample(
                    get_training_openings(), openings_per_opponent)
                gtasks += pair_opening_tasks(
                    neat_spec(candidate, config_path, depth),
                    neat_spec(entry["genome"], config_path, depth),
                    gate_openings, base_seed=gen_rng.randrange(1 << 30),
                    max_plies=max_plies,
                    meta={"type": "gate", "a": candidate_gid,
                          "opponent_sha256": entry["genome_sha256"],
                          "generation": gen, "opponent_index": index,
                          "purpose": "hof_gate"})
            gate_records = run_tasks(gtasks, executor, chunksize=2)
            strict, _ = score_for_a(gate_records)
            frac = strict / len(gtasks)
            gate_score = f"{frac:.2f}"
            if frac >= GATE_THRESHOLD:
                hall_of_fame.append(_hof_entry(
                    candidate, gen, frac, fitness_mode,
                    (entry["genome_sha256"] for entry in opponents)))
                if len(hall_of_fame) > HOF_MAX:
                    _prune_redundant_hof_entry(hall_of_fame)
                atomic_pickle_dump(hall_of_fame, hof_path)
        if gate_records:
            append_raw_records(gate_records,
                               os.path.join(out_dir, "raw_gate_games.jsonl"))

        # ---- save generation best genome --------------------------------------
        if candidate_fit > state["best_ever_fitness"] or state["best_ever"] is None:
            state["best_ever"] = candidate
            state["best_ever_fitness"] = candidate_fit
        atomic_pickle_dump(candidate, os.path.join(out_dir, "best_value_genome.pkl"))
        if publish_web_genome:
            atomic_pickle_dump(candidate, "best_value_genome.pkl")

        # ---- periodic strength probe vs fixed ladder ----------------------------
        elo_str = ""
        winrates = ["" for _ in PROBE_RUNGS]
        if gen % probe_every == 0:
            results = []
            val_openings = get_validation_openings(max(1, probe_games // 2))
            all_probe_records = []
            for i, rname in enumerate(PROBE_RUNGS):
                ptasks = pair_opening_tasks(neat_spec(candidate, config_path, depth),
                                            rung_specs[rname], val_openings,
                                            base_seed=(seed << 8) ^ (gen * 131 + i),
                                            max_plies=max_plies,
                                            meta={"rung": rname, "gen": gen})
                precords = run_tasks(ptasks, executor, chunksize=2)
                strict, _ = score_for_a(precords)
                results.append((rname, strict, len(ptasks)))
                winrates[i] = f"{strict / len(ptasks):.3f}"
                all_probe_records.extend(precords)
            append_raw_records(all_probe_records, os.path.join(out_dir, "raw_probe_games.jsonl"))
            elo = performance_rating(results, ladder_elo)
            elo_str = f"{elo:.0f}"
            if elo > state["best_ever_elo"]:
                state["best_ever_elo"] = elo
                state["best_ever_elo_genome"] = candidate
                atomic_pickle_dump(
                    candidate, os.path.join(out_dir, "best_ever_genome.pkl"))
            print(f"  [probe] Elo ~{elo:.0f} | " +
                  " ".join(f"{r}:{w}" for r, w in zip(PROBE_RUNGS, winrates)))

        elapsed = time.time() - t0
        n_games = len(records)
        draws = sum(record["winner"] == 0 for record in records)
        plies = sum(record["plies"] for record in records)
        candidate_complexity = complexities[candidate_gid]
        mean_nodes = mean(item["nodes"] for item in complexities.values())
        mean_connections = mean(item["connections"]
                                for item in complexities.values())
        mean_parameters = mean(item["parameters"]
                               for item in complexities.values())
        search_nodes, network_evaluations = _compute_totals(
            records + candidate_records)
        rank_gate_pass = ""
        if reliability is not None and rank_reliability_threshold is not None:
            rank_gate_pass = int(reliability >= rank_reliability_threshold)
        writer.writerow([gen, f"{best_fit:.4f}", f"{candidate_fit:.4f}",
                         f"{mean(g.fitness for _, g in genomes):.4f}",
                         len(pop.species.species), n_games,
                         len(candidate_records),
                         f"{plies / max(n_games, 1):.1f}",
                         f"{draws / max(n_games, 1):.3f}",
                         fitness_mode, len(eligible),
                         "" if reliability is None else f"{reliability:.3f}",
                         rank_gate_pass, gate_score, len(hall_of_fame),
                         candidate_complexity["nodes"],
                         candidate_complexity["connections"],
                         candidate_complexity["parameters"],
                         f"{mean_nodes:.2f}", f"{mean_connections:.2f}",
                         f"{mean_parameters:.2f}", search_nodes,
                         network_evaluations, f"{elapsed:.1f}", elo_str] +
                        winrates)
        _flush_and_fsync(csv_file)
        print(f"  gen {gen}: best={best_fit:.3f} games={n_games} "
              f"draws={draws / max(n_games, 1):.2f} hof={len(hall_of_fame)} "
              f"gate={gate_score} ({elapsed:.0f}s)")
        if reliability is not None and rank_reliability_threshold is not None \
                and reliability < rank_reliability_threshold:
            print(f"  [gate warning] rank reliability {reliability:.3f} < "
                  f"{rank_reliability_threshold:.3f}")

    try:
        random.setstate(evolution_random_state)
        winner = pop.run(eval_genomes, n_gens_to_run)
        checkpointer.save_final()
    finally:
        executor.shutdown()
        csv_file.close()

    winner = state["best_ever"] or winner
    atomic_pickle_dump(winner, os.path.join(out_dir, "best_value_genome.pkl"))
    print(f"\nTraining complete. Best genome saved to {out_dir}/best_value_genome.pkl")
    return winner
