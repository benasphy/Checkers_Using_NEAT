"""Path A training: evolve a NEAT value network used inside alpha-beta search.

Design (see README / paper plan):

- ONE population. Each genome is a value network; the playing agent is
  alpha-beta search (fixed depth during training) using that network at the
  leaves.
- Fitness is OUTCOME-ONLY, from color-balanced games against population peers
  (random pairings) and against the hall of fame. Draws use a bounded
  material tie-break (in (0.4, 0.6)) purely to give early populations a
  gradient; wins/losses always dominate.
- Hall of fame is Elo-gated: the generation's best genome enters only by
  scoring >= 55% in a strict color-balanced gate match against HOF members.
- Strength is MEASURED (never trained on) against a fixed ladder of
  material-only alpha-beta players, reported as a performance Elo anchored at
  random = 0. The ladder is calibrated once and cached.

Everything is seeded and logged to CSV for reproducibility.
"""

from __future__ import annotations

import csv
import json
import os
import pickle
import random
import time
from concurrent.futures import ProcessPoolExecutor
from statistics import mean

import neat

from ai.arena import pair_tasks, run_tasks, score_for_a
from ai.elo import expected, fit_elo
from ai.ladder import LADDER, load_neat_config, neat_spec

HOF_MAX = 12
GATE_GAMES = 12
GATE_THRESHOLD = 0.55
PROBE_RUNGS = ["random", "material-d1", "material-d2", "material-d4"]


# ---------------------------------------------------------------------------
# Ladder calibration (once, cached): fixed Elo for the reference opponents.
# ---------------------------------------------------------------------------

def calibrate_ladder(executor, cache_path: str, seed: int = 7,
                     games_per_pair: int = 40) -> dict:
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            return json.load(f)
    print("Calibrating reference ladder (one-off)...")
    rungs = [(n, s) for n, s in LADDER if n in PROBE_RUNGS]
    tasks = []
    for i, (na, sa) in enumerate(rungs):
        for j, (nb, sb) in enumerate(rungs):
            if i < j:
                tasks += pair_tasks(sa, sb, games_per_pair,
                                    base_seed=seed + 97 * i + j,
                                    meta={"pair": (na, nb)})
    records = run_tasks(tasks, executor)
    games = []
    for rec in records:
        na, nb = rec["meta"]["pair"]
        s_a = rec["p1_strict"] if rec["meta"]["a_is_p1"] else 1.0 - rec["p1_strict"]
        games.append((na, nb, s_a))
    ratings = fit_elo(games, anchor="random", anchor_rating=0.0)
    with open(cache_path, "w") as f:
        json.dump(ratings, f, indent=2)
    print("Ladder Elo:", {k: round(v) for k, v in ratings.items()})
    return ratings


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


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def run_training(config_path: str, generations: int = 200, depth: int = 4,
                 workers=None, seed: int = 0, peer_rounds: int = 2,
                 probe_every: int = 5, probe_games: int = 8,
                 out_dir: str = "runs/path_a", pop_size=None,
                 max_plies: int = 150, resume: str = None):
    os.makedirs(out_dir, exist_ok=True)
    config = load_neat_config(config_path)
    if pop_size:
        config.pop_size = pop_size

    rng = random.Random(seed)
    if resume:
        pop = neat.Checkpointer.restore_checkpoint(resume)
        pop.config = config
        print(f"Resumed from {resume} at generation {pop.generation}.")
    else:
        pop = neat.Population(config)
    pop.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)
    pop.add_reporter(neat.Checkpointer(
        generation_interval=5, time_interval_seconds=None,
        filename_prefix=os.path.join(out_dir, "neat-checkpoint-")))

    hof_path = os.path.join(out_dir, "hof.pkl")
    hall_of_fame: list = []
    if os.path.exists(hof_path):
        with open(hof_path, "rb") as f:
            hall_of_fame = pickle.load(f)
        print(f"Loaded hall of fame with {len(hall_of_fame)} members.")

    metrics_path = os.path.join(out_dir, "training_metrics.csv")
    new_csv = not os.path.exists(metrics_path)
    csv_file = open(metrics_path, "a", newline="")
    writer = csv.writer(csv_file)
    if new_csv:
        writer.writerow(["gen", "best_fitness", "mean_fitness", "species",
                         "games", "avg_plies", "draw_rate", "gate", "hof_size",
                         "elapsed_s", "elo"] +
                        [f"wr_{r}" for r in PROBE_RUNGS])

    executor = ProcessPoolExecutor(max_workers=workers)
    ladder_elo = calibrate_ladder(
        executor, os.path.join(out_dir, "ladder_calibration.json"), seed=seed + 7)
    rung_specs = dict(LADDER)

    state = {"gen": getattr(pop, "generation", 0) or 0,
             "best_ever": None, "best_ever_fitness": -1.0}

    def eval_genomes(genomes, cfg):
        gen = state["gen"]
        state["gen"] += 1
        t0 = time.time()
        gen_rng = random.Random((seed << 20) ^ gen)

        gids = [gid for gid, _ in genomes]
        genome_by_id = dict(genomes)
        specs = {gid: neat_spec(g, config_path, depth) for gid, g in genomes}

        # ---- build tasks: peer pairings + hall-of-fame games ----------------
        tasks = []
        for rnd in range(peer_rounds):
            order = gids[:]
            gen_rng.shuffle(order)
            if len(order) % 2 == 1:
                order.append(gen_rng.choice(order[:-1]))
            for a, b in zip(order[::2], order[1::2]):
                tasks += pair_tasks(specs[a], specs[b], 2,
                                    base_seed=gen_rng.randrange(1 << 30),
                                    max_plies=max_plies,
                                    meta={"type": "peer", "a": a, "b": b})
        if hall_of_fame:
            for gid in gids:
                hof_genome = gen_rng.choice(hall_of_fame)
                tasks += pair_tasks(specs[gid],
                                    neat_spec(hof_genome, config_path, depth), 2,
                                    base_seed=gen_rng.randrange(1 << 30),
                                    max_plies=max_plies,
                                    meta={"type": "hof", "a": gid, "b": None})

        records = run_tasks(tasks, executor, chunksize=4)

        # ---- aggregate outcome-only fitness ---------------------------------
        points = {gid: 0.0 for gid in gids}
        played = {gid: 0 for gid in gids}
        draws = plies = 0
        for rec in records:
            m = rec["meta"]
            shaped_p1 = rec["p1_shaped"]
            a_pts = shaped_p1 if m["a_is_p1"] else 1.0 - shaped_p1
            points[m["a"]] += a_pts
            played[m["a"]] += 1
            if m["type"] == "peer":
                points[m["b"]] += 1.0 - a_pts
                played[m["b"]] += 1
            draws += 1 if rec["winner"] == 0 else 0
            plies += rec["plies"]

        for gid, genome in genomes:
            genome.fitness = points[gid] / max(played[gid], 1)

        best_gid = max(gids, key=lambda g: genome_by_id[g].fitness)
        best_genome = genome_by_id[best_gid]
        best_fit = best_genome.fitness

        # ---- Elo-gated hall of fame ------------------------------------------
        gate_score = ""
        candidate = pickle.loads(pickle.dumps(best_genome))
        if not hall_of_fame:
            hall_of_fame.append(candidate)
            gate_score = "seed"
        else:
            opponents = gen_rng.sample(hall_of_fame,
                                       min(3, len(hall_of_fame)))
            gtasks = []
            per_opp = max(GATE_GAMES // len(opponents) // 2 * 2, 2)
            for i, opp in enumerate(opponents):
                gtasks += pair_tasks(neat_spec(candidate, config_path, depth),
                                     neat_spec(opp, config_path, depth),
                                     per_opp,
                                     base_seed=gen_rng.randrange(1 << 30),
                                     max_plies=max_plies, meta={"i": i})
            grecords = run_tasks(gtasks, executor, chunksize=2)
            strict, _ = score_for_a(grecords)
            frac = strict / len(gtasks)
            gate_score = f"{frac:.2f}"
            if frac >= GATE_THRESHOLD:
                hall_of_fame.append(candidate)
                if len(hall_of_fame) > HOF_MAX:
                    hall_of_fame.pop(0)
                with open(hof_path, "wb") as f:
                    pickle.dump(hall_of_fame, f)

        # ---- save best genome -------------------------------------------------
        if best_fit > state["best_ever_fitness"] or state["best_ever"] is None:
            state["best_ever"] = candidate
            state["best_ever_fitness"] = best_fit
        with open(os.path.join(out_dir, "best_value_genome.pkl"), "wb") as f:
            pickle.dump(candidate, f)
        with open("best_value_genome.pkl", "wb") as f:  # web app default path
            pickle.dump(candidate, f)

        # ---- periodic strength probe vs fixed ladder ----------------------------
        elo_str = ""
        winrates = ["" for _ in PROBE_RUNGS]
        if gen % probe_every == 0:
            results = []
            for i, rname in enumerate(PROBE_RUNGS):
                ptasks = pair_tasks(neat_spec(candidate, config_path, depth),
                                    rung_specs[rname], probe_games,
                                    base_seed=(seed << 8) ^ (gen * 131 + i),
                                    max_plies=max_plies, meta={"rung": rname})
                precords = run_tasks(ptasks, executor, chunksize=2)
                strict, _ = score_for_a(precords)
                results.append((rname, strict, len(ptasks)))
                winrates[i] = f"{strict / len(ptasks):.3f}"
            elo = performance_rating(results, ladder_elo)
            elo_str = f"{elo:.0f}"
            print(f"  [probe] Elo ~{elo:.0f} | " +
                  " ".join(f"{r}:{w}" for r, w in zip(PROBE_RUNGS, winrates)))

        elapsed = time.time() - t0
        n_games = len(records)
        writer.writerow([gen, f"{best_fit:.4f}",
                         f"{mean(g.fitness for _, g in genomes):.4f}",
                         len(pop.species.species), n_games,
                         f"{plies / max(n_games, 1):.1f}",
                         f"{draws / max(n_games, 1):.3f}",
                         gate_score, len(hall_of_fame), f"{elapsed:.1f}",
                         elo_str] + winrates)
        csv_file.flush()
        print(f"  gen {gen}: best={best_fit:.3f} games={n_games} "
              f"draws={draws / max(n_games, 1):.2f} hof={len(hall_of_fame)} "
              f"gate={gate_score} ({elapsed:.0f}s)")

    try:
        winner = pop.run(eval_genomes, generations)
    finally:
        executor.shutdown()
        csv_file.close()

    with open(os.path.join(out_dir, "best_value_genome.pkl"), "wb") as f:
        pickle.dump(winner, f)
    print(f"\nTraining complete. Best genome saved to {out_dir}/best_value_genome.pkl")
    return winner
