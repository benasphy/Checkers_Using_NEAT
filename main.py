"""CLI for the NEAT-checkers project.

Commands:
  train   Evolve a NEAT value network (Path A: value net + alpha-beta search).
  ladder  Measure a saved genome against the fixed reference ladder (Elo).
  play    Play against a trained agent in the terminal.
  perft   Validate/benchmark the move generator.
"""

import argparse
import os
import pickle
import sys
import time

DEFAULT_CONFIG = os.path.join(os.path.dirname(__file__), "neat_value_config.txt")
DEFAULT_GENOME = "best_value_genome.pkl"


def cmd_train(args):
    from ai.train_path_a import run_training
    run_training(
        config_path=args.config,
        generations=args.generations,
        depth=args.depth,
        workers=args.workers,
        seed=args.seed,
        peer_rounds=args.peer_rounds,
        probe_every=args.probe_every,
        out_dir=args.out,
        pop_size=args.pop,
        resume=args.resume,
    )


def cmd_ladder(args):
    from concurrent.futures import ProcessPoolExecutor

    from ai.arena import pair_tasks, run_tasks, score_for_a
    from ai.ladder import LADDER, neat_spec
    from ai.train_path_a import PROBE_RUNGS, calibrate_ladder, performance_rating

    with open(args.genome, "rb") as f:
        genome = pickle.load(f)
    spec = neat_spec(genome, args.config, args.depth)
    rungs = [(n, s) for n, s in LADDER if n in PROBE_RUNGS]
    if args.d6:
        rungs.append(("material-d6", ("material", 6)))

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        cache = os.path.join(args.out, "ladder_calibration.json")
        os.makedirs(args.out, exist_ok=True)
        ladder_elo = calibrate_ladder(executor, cache, seed=args.seed + 7)

        print(f"\nEvaluating {args.genome} at depth {args.depth} "
              f"({args.games} games per rung)...")
        results = []
        for i, (rname, rspec) in enumerate(rungs):
            tasks = pair_tasks(spec, rspec, args.games,
                               base_seed=args.seed + 31 * i,
                               meta={"rung": rname})
            records = run_tasks(tasks, executor)
            strict, _ = score_for_a(records)
            wins = sum(1 for r in records
                       if (r["winner"] == 1) == r["meta"]["a_is_p1"]
                       and r["winner"] != 0)
            draws = sum(1 for r in records if r["winner"] == 0)
            print(f"  vs {rname:12s}: {strict:5.1f}/{len(tasks)} points  "
                  f"(W {wins}  D {draws}  L {len(tasks) - wins - draws})")
            if rname in ladder_elo:
                results.append((rname, strict, len(tasks)))
        elo = performance_rating(results, ladder_elo)
        print(f"\nPerformance Elo vs calibrated ladder (random=0): {elo:.0f}")
        print("Ladder anchors:",
              {k: round(v) for k, v in sorted(ladder_elo.items(),
                                              key=lambda kv: kv[1])})


def cmd_play(args):
    from checkers.game import CheckersGame
    from ai.ladder import make_agent, neat_spec

    if os.path.exists(args.genome):
        with open(args.genome, "rb") as f:
            genome = pickle.load(f)
        agent = make_agent(neat_spec(genome, args.config, args.depth),
                           seed=args.seed)
        print(f"Loaded {args.genome} (search depth {args.depth}).")
    else:
        agent = make_agent(("material", args.depth), seed=args.seed)
        print(f"No genome at {args.genome}; playing material-d{args.depth}.")

    game = CheckersGame()
    print("\nYou are 'r' (bottom), moving up. Enter moves as: fromRow fromCol toRow toCol")
    print("For multi-jumps enter origin and FINAL landing square.\n")
    while not game.is_game_over():
        print(game.position, "\n")
        if game.current_player == 1:
            moves = game.get_legal_moves()
            for i, m in enumerate(moves):
                caps = f" x{len(m[4])}" if m[4] else ""
                print(f"  [{i}] ({m[0]},{m[1]}) -> ({m[2]},{m[3]}){caps}")
            raw = input("Your move (index or 4 numbers, q to quit): ").strip()
            if raw.lower() == "q":
                return
            try:
                parts = raw.split()
                if len(parts) == 1:
                    game.make_move(moves[int(parts[0])])
                else:
                    fr_r, fr_c, to_r, to_c = map(int, parts)
                    game.make_move((fr_r, fr_c, to_r, to_c, []))
            except (ValueError, IndexError) as e:
                print(f"  Invalid move ({e}). Try again.")
                continue
        else:
            t0 = time.time()
            mv = agent.select(game)
            game.make_engine_move(mv)
            print(f"AI plays {game.move_history[-1].path} ({time.time() - t0:.1f}s)")
    print(game.position)
    winner = game.get_winner()
    msg = {1: "You win!", 2: "AI wins!", 0: f"Draw ({game.draw_reason()})."}
    print(msg[winner])


def cmd_perft(args):
    from checkers.bitboard import Position, perft
    pos = Position.initial()
    for d in range(1, args.depth + 1):
        t0 = time.time()
        n = perft(pos, d)
        dt = time.time() - t0
        rate = n / dt if dt > 0 else float("inf")
        print(f"perft({d}) = {n:>12,}   {dt:7.2f}s   {rate:,.0f} leaves/s")


def cli(argv=None):
    p = argparse.ArgumentParser(description="Checkers AI with NEAT + search")
    sub = p.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train", help="evolve a NEAT value network")
    t.add_argument("--generations", type=int, default=200)
    t.add_argument("--depth", type=int, default=4,
                   help="search depth during training")
    t.add_argument("--workers", type=int, default=None)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--pop", type=int, default=None,
                   help="override pop_size from the config")
    t.add_argument("--peer-rounds", type=int, default=2)
    t.add_argument("--probe-every", type=int, default=5)
    t.add_argument("--config", default=DEFAULT_CONFIG)
    t.add_argument("--out", default="runs/path_a")
    t.add_argument("--resume", default=None,
                   help="path to a runs/<name>/neat-checkpoint-N file")
    t.set_defaults(fn=cmd_train)

    l = sub.add_parser("ladder", help="measure a genome vs the reference ladder")
    l.add_argument("--genome", default=DEFAULT_GENOME)
    l.add_argument("--depth", type=int, default=6)
    l.add_argument("--games", type=int, default=20)
    l.add_argument("--d6", action="store_true", help="also play material-d6")
    l.add_argument("--workers", type=int, default=None)
    l.add_argument("--seed", type=int, default=0)
    l.add_argument("--config", default=DEFAULT_CONFIG)
    l.add_argument("--out", default="runs/path_a")
    l.set_defaults(fn=cmd_ladder)

    g = sub.add_parser("play", help="play vs the agent in the terminal")
    g.add_argument("--genome", default=DEFAULT_GENOME)
    g.add_argument("--depth", type=int, default=6)
    g.add_argument("--seed", type=int, default=None)
    g.add_argument("--config", default=DEFAULT_CONFIG)
    g.set_defaults(fn=cmd_play)

    f = sub.add_parser("perft", help="validate/benchmark move generation")
    f.add_argument("--depth", type=int, default=7)
    f.set_defaults(fn=cmd_perft)

    args = p.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    cli()
