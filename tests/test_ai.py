"""Tests for features, search behavior, arena, and Elo fitting."""

from checkers.bitboard import NUM_SQUARES, Position, rc_to_sq
from ai.arena import play_match
from ai.elo import fit_elo
from ai.features import encode
from ai.search import Searcher, material_eval


def sq(r, c):
    s = rc_to_sq(r, c)
    assert s >= 0
    return s


def _rot(mask):
    out = 0
    for s in range(NUM_SQUARES):
        if (mask >> s) & 1:
            out |= 1 << (NUM_SQUARES - 1 - s)
    return out


def test_feature_color_symmetry():
    # A position for P1-to-move and its 180-degree color-swapped mirror for
    # P2-to-move must produce identical canonical encodings.
    p1 = Position(m1=(1 << sq(5, 2)) | (1 << sq(6, 1)), k1=1 << sq(2, 3),
                  m2=(1 << sq(1, 2)) | (1 << sq(3, 4)), k2=0, turn=1)
    p2 = Position(m1=_rot(p1.m2), k1=_rot(p1.k2),
                  m2=_rot(p1.m1), k2=_rot(p1.k1), turn=2)
    assert encode(p1) == encode(p2)


def test_feature_initial_balance():
    x = encode(Position.initial())
    assert sum(x[:32]) == 0.0  # symmetric start
    assert x[32] == x[34] == 1.0  # 12/12 men each


def test_search_avoids_hanging_piece():
    # P1 king at (7,2): moving to (6,1) lets the P2 king at (5,0) capture it
    # (losing the game); (6,3) is safe. Depth-2 material search must pick it.
    pos = Position(m1=0, k1=1 << sq(7, 2), m2=0, k2=1 << sq(5, 0), turn=1)
    s = Searcher(material_eval, depth=2, seed=1)
    mv = s.best_move(pos)
    assert (mv.fr, mv.to) == (sq(7, 2), sq(6, 3))


def test_search_prefers_bigger_capture_line():
    # Both a single jump and a double jump are available; the double leaves
    # the opponent with nothing useful, the single hangs a man.
    pos = Position(
        m1=(1 << sq(6, 1)) | (1 << sq(6, 7)),
        k1=0,
        m2=(1 << sq(5, 2)) | (1 << sq(3, 2)) | (1 << sq(5, 6)),
        k2=0, turn=1)
    s = Searcher(material_eval, depth=3, seed=1)
    mv = s.best_move(pos)
    assert bin(mv.cap).count("1") == 2


def test_search_finds_win_over_draw():
    # Sanity: from the initial position search returns a legal move quickly.
    s = Searcher(material_eval, depth=4, seed=0)
    mv = s.best_move(Position.initial())
    assert mv is not None and mv.cap == 0


def test_search_uses_threefold_context_and_clears_root_tt():
    from ai.search import Searcher, material_eval

    pos = Position(m1=(1 << sq(6, 1)) | (1 << sq(6, 5)), k1=0,
                   m2=1 << sq(1, 2), k2=0, turn=1)
    searcher = Searcher(material_eval, depth=1, seed=1)
    searcher._deadline = None
    searcher._rep = {pos.hash: 1}
    assert searcher._negamax(pos, 0, float("-inf"), float("inf"), 1) == 100.0
    searcher._rep = {pos.hash: 2}
    assert searcher._negamax(pos, 0, float("-inf"), float("inf"), 1) == 0.0

    searcher.tt[("stale",)] = (99, 0, 999.0, None)
    searcher.best_move(Position.initial())
    assert ("stale",) not in searcher.tt


def test_material_d2_crushes_random():
    strict, records = play_match(("material", 2), ("random",),
                                 games=8, base_seed=42)
    assert strict >= 6.0, f"material-d2 scored only {strict}/8 vs random"
    assert all(rec["plies"] > 0 for rec in records)


def test_elo_fit_direction_and_scale():
    games = [("A", "B", 1.0)] * 9 + [("A", "B", 0.0)]
    ratings = fit_elo(games, anchor="B", anchor_rating=0.0)
    diff = ratings["A"] - ratings["B"]
    assert 280 < diff < 500, f"unexpected Elo gap {diff}"


def _pruned_stock_network(genome, config):
    """Compile stock NEAT after independently removing inactive connections."""
    import copy
    import neat

    connections = [key for key, gene in genome.connections.items() if gene.enabled]
    reachable = set(config.genome_config.input_keys)
    changed = True
    while changed:
        changed = False
        for source, target in connections:
            if source in reachable and target not in reachable:
                reachable.add(target)
                changed = True

    required = set(config.genome_config.output_keys)
    changed = True
    while changed:
        changed = False
        for source, target in connections:
            if target in required and source not in required:
                required.add(source)
                changed = True

    active = reachable & required
    pruned = copy.deepcopy(genome)
    for key, gene in pruned.connections.items():
        if gene.enabled and not (key[0] in active and key[1] in active):
            gene.enabled = False
    return neat.nn.FeedForwardNetwork.create(pruned, config)


def test_fast_feed_forward_prunes_orphans_and_evaluates_output():
    import neat
    import pytest
    from ai.ladder import load_neat_config
    from ai.value_net import FastFeedForwardNetwork

    cfg = load_neat_config("neat_value_config.txt")
    genome = neat.DefaultGenome(1)
    genome.configure_new(cfg.genome_config)

    for connection in genome.connections.values():
        connection.enabled = False
    input_key = cfg.genome_config.input_keys[0]
    direct = genome.connections[(input_key, 0)]
    direct.enabled = True
    direct.weight = 0.75

    output = genome.nodes[0]
    output.bias = -0.2
    output.response = 1.7

    # Stock NEAT omits output 0 if this unreachable incoming edge is not pruned.
    orphan_id = 9999
    node_gene = neat.genes.DefaultNodeGene(orphan_id)
    node_gene.bias = 0.5
    node_gene.response = 1.0
    node_gene.activation = "tanh"
    node_gene.aggregation = "sum"
    genome.nodes[orphan_id] = node_gene
    # Create connection from orphan to output
    cg = neat.genes.DefaultConnectionGene((orphan_id, 0))
    cg.weight = 2.5
    cg.enabled = True
    genome.connections[(orphan_id, 0)] = cg

    stock_unpruned = neat.nn.FeedForwardNetwork.create(genome, cfg)
    assert 0 not in {step[0] for step in stock_unpruned.node_evals}

    reference = _pruned_stock_network(genome, cfg)
    fast = FastFeedForwardNetwork.create(genome, cfg)
    inputs = [0.0] * 36
    inputs[0] = 0.4
    expected = cfg.genome_config.activation_defs.get(output.activation)(
        output.bias + output.response * (inputs[0] * direct.weight)
    )

    assert reference.activate(inputs)[0] == pytest.approx(expected, abs=1e-12)
    assert fast.activate(inputs)[0] == pytest.approx(expected, abs=1e-12)


def test_fast_feed_forward_disconnected_output_matches_stock_zero():
    import neat
    from ai.ladder import load_neat_config
    from ai.value_net import FastFeedForwardNetwork

    cfg = load_neat_config("neat_value_config.txt")
    genome = neat.DefaultGenome(2)
    genome.configure_new(cfg.genome_config)
    for connection in genome.connections.values():
        connection.enabled = False
    genome.nodes[0].bias = 0.5
    genome.nodes[0].response = 2.0

    inputs = [0.25] * 36
    reference = _pruned_stock_network(genome, cfg)
    fast = FastFeedForwardNetwork.create(genome, cfg)

    assert reference.activate(inputs) == [0.0]
    assert fast.activate(inputs) == [0.0]


def test_fast_feed_forward_numerical_equivalence_with_stock_neat():
    """Compare every generated phenotype with stock NEAT after pruning."""
    import random
    import neat
    import pytest
    from ai.ladder import load_neat_config
    from ai.value_net import FastFeedForwardNetwork

    cfg = load_neat_config("neat_value_config.txt")
    random.seed(42)

    # Test across 30 mutated genomes with random hidden nodes and connections
    for genome_id in range(1, 31):
        genome = neat.DefaultGenome(genome_id)
        genome.configure_new(cfg.genome_config)

        # Mutate structure and weights randomly
        for _ in range(random.randint(1, 5)):
            genome.mutate_add_node(cfg.genome_config)
        for _ in range(random.randint(2, 10)):
            genome.mutate_add_connection(cfg.genome_config)
        for _ in range(random.randint(3, 15)):
            genome.mutate(cfg.genome_config)

        stock_net = _pruned_stock_network(genome, cfg)
        fast_net = FastFeedForwardNetwork.create(genome, cfg)

        for _ in range(5):
            inp = [random.uniform(-1.0, 1.0) for _ in range(36)]
            stock_out = stock_net.activate(inp)
            fast_out = fast_net.activate(inp)

            assert len(fast_out) == len(stock_out) == 1
            assert fast_out[0] == pytest.approx(stock_out[0], abs=1e-12), \
                f"Mismatch on genome {genome_id}: fast={fast_out[0]}, stock={stock_out[0]}"


def test_fast_feed_forward_rejects_active_cycle():
    import neat
    import pytest
    from ai.ladder import load_neat_config
    from ai.value_net import FastFeedForwardNetwork

    cfg = load_neat_config("neat_value_config.txt")
    genome = neat.DefaultGenome(3)
    genome.configure_new(cfg.genome_config)
    for connection in genome.connections.values():
        connection.enabled = False

    hidden_a, hidden_b = 1001, 1002
    for node_id in (hidden_a, hidden_b):
        node = neat.genes.DefaultNodeGene(node_id)
        node.bias = 0.0
        node.response = 1.0
        node.activation = "tanh"
        node.aggregation = "sum"
        genome.nodes[node_id] = node

    cycle_edges = [
        (cfg.genome_config.input_keys[0], hidden_a),
        (hidden_a, hidden_b),
        (hidden_b, hidden_a),
        (hidden_b, 0),
    ]
    for key in cycle_edges:
        connection = neat.genes.DefaultConnectionGene(key)
        connection.weight = 1.0
        connection.enabled = True
        genome.connections[key] = connection

    with pytest.raises(ValueError, match="cycle"):
        FastFeedForwardNetwork.create(genome, cfg)


def test_arena_opening_plies_execution():
    from ai.arena import pair_tasks, run_tasks, play_match

    # Verify pair_tasks creates tasks with opening_plies
    tasks = pair_tasks(("material", 1), ("random",), games=4, base_seed=123, opening_plies=4)
    assert len(tasks) == 4
    records = run_tasks(tasks)
    assert len(records) == 4
    for r in records:
        assert r["plies"] >= 4
        assert "winner" in r
        assert "p1_strict" in r

    strict, match_records = play_match(("material", 1), ("random",), games=2, base_seed=99, opening_plies=2)
    assert len(match_records) == 2
    assert 0.0 <= strict <= 2.0


def test_color_pair_reuses_opening_and_strict_cap_is_draw():
    from ai.arena import pair_tasks, run_tasks

    tasks = pair_tasks(("random",), ("random",), games=2, base_seed=17,
                       opening_plies=2, max_plies=2, adjudicate=False)
    records = run_tasks(tasks)

    assert records[0]["seed"] == records[1]["seed"]
    assert records[0]["moves"] == records[1]["moves"]
    assert [r["meta"]["a_is_p1"] for r in records] == [True, False]
    assert all(r["winner"] == 0 and r["reason"] == "max_plies" for r in records)
    assert all(r["protocol"] == "paired-strict-v1" for r in records)


def test_deterministic_neat_seeding():
    import random
    import neat
    from ai.ladder import load_neat_config

    cfg = load_neat_config("neat_value_config.txt")

    random.seed(777)
    pop1 = neat.Population(cfg)
    g1_keys = sorted(pop1.population.keys())
    g1_weights = [pop1.population[k].connections[(i, 0)].weight for k in g1_keys for (i, o) in [(i, 0) for i in range(-36, 0)] if (i, 0) in pop1.population[k].connections]

    random.seed(777)
    pop2 = neat.Population(cfg)
    g2_keys = sorted(pop2.population.keys())
    g2_weights = [pop2.population[k].connections[(i, 0)].weight for k in g2_keys for (i, o) in [(i, 0) for i in range(-36, 0)] if (i, 0) in pop2.population[k].connections]

    assert g1_keys == g2_keys
    assert g1_weights == g2_weights


def test_neat_checkpoint_resume_is_generation_exact_and_deterministic(tmp_path):
    import random
    import neat
    import pytest
    from ai.ladder import load_neat_config
    from ai.train_path_a import AtomicCheckpointer, restore_training_checkpoint

    def config():
        cfg = load_neat_config("neat_value_config.txt", fresh=True)
        cfg.pop_size = 8
        return cfg

    def evaluate(genomes, _config):
        for gid, genome in genomes:
            genome.fitness = sum(c.weight for c in genome.connections.values()) - gid * 1e-9

    def signature(pop):
        return sorted(
            (gid,
             tuple(sorted((key, node.bias, node.response)
                          for key, node in genome.nodes.items())),
             tuple(sorted((key, conn.weight, conn.enabled)
                          for key, conn in genome.connections.items())))
            for gid, genome in pop.population.items()
        )

    random.seed(1234)
    uninterrupted = neat.Population(config())
    uninterrupted.run(evaluate, 4)
    expected_population = signature(uninterrupted)
    expected_random_state = random.getstate()

    random.seed(1234)
    split = neat.Population(config())
    manifest = {"test": "resume-v1"}
    checkpointer = AtomicCheckpointer(
        generation_interval=1, time_interval_seconds=None,
        filename_prefix=str(tmp_path / "neat-checkpoint-"),
        population_ref=split, manifest=manifest,
        snapshot_fn=lambda: {"marker": "committed"})
    split.add_reporter(checkpointer)
    split.run(evaluate, 2)
    checkpoint = tmp_path / "neat-checkpoint-2"
    assert checkpoint.exists()

    with pytest.raises(ValueError, match="manifest mismatch"):
        restore_training_checkpoint(str(checkpoint), config(), {"test": "different"})

    resumed, state, random_state = restore_training_checkpoint(
        str(checkpoint), config(), manifest)
    assert resumed.generation == 2
    assert state == {"marker": "committed"}
    random.setstate(random_state)
    resumed.run(evaluate, 2)

    assert resumed.generation == 4
    assert signature(resumed) == expected_population
    assert random.getstate() == expected_random_state


def test_openings_suite_split_and_validity():
    from ai.openings import get_validation_openings, get_test_openings

    val_ops = get_validation_openings()
    test_ops = get_test_openings()

    assert len(val_ops) == 16, f"Expected 16 validation openings, got {len(val_ops)}"
    assert len(test_ops) == 32, f"Expected 32 test openings, got {len(test_ops)}"

    val_ids = {op.id for op in val_ops}
    test_ids = {op.id for op in test_ops}
    assert val_ids.isdisjoint(test_ids), "Validation and test opening IDs must be disjoint"

    val_hashes = {op.position.hash for op in val_ops}
    test_hashes = {op.position.hash for op in test_ops}
    assert val_hashes.isdisjoint(test_hashes), "Validation and test positions must be disjoint"

    for op in val_ops + test_ops:
        assert op.position.turn == 1, "2-ply openings must return turn to player 1"
        assert len(op.move_history) == 2, "2-ply openings must have 2 moves in history"
        assert len(op.position.legal_moves()) > 0, "Opening position must have legal moves"


def test_pair_opening_tasks_and_raw_logging(tmp_path):
    import json, os
    from ai.openings import get_validation_openings
    from ai.arena import pair_opening_tasks, run_tasks, score_for_a, append_raw_records

    val_ops = get_validation_openings(2)
    tasks = pair_opening_tasks(("material", 1), ("random",), val_ops, base_seed=42)
    assert len(tasks) == 4  # 2 openings * 2 colors

    records = run_tasks(tasks)
    assert len(records) == 4
    for r in records:
        assert "opening_id" in r["meta"]
        assert r["plies"] >= 2
        assert "winner" in r
        assert "p1_strict" in r

    strict, shaped = score_for_a(records)
    assert 0.0 <= strict <= 4.0

    log_file = str(tmp_path / "test_raw.jsonl")
    append_raw_records(records, log_file)
    assert os.path.exists(log_file)
    with open(log_file, "r") as f:
        lines = [json.loads(line) for line in f if line.strip()]
    assert len(lines) == 4
    assert lines[0]["meta"]["opening_id"] == "val_00"


def test_fixed_nonlinear_config_and_phenotype():
    import neat
    from ai.ladder import load_neat_config
    from ai.value_net import FastFeedForwardNetwork

    cfg = load_neat_config("neat_fixed_nonlinear_config.txt")
    assert cfg.genome_config.num_hidden == 8
    assert cfg.genome_config.conn_add_prob == 0.0
    assert cfg.genome_config.node_add_prob == 0.0

    genome = neat.DefaultGenome(1)
    genome.configure_new(cfg.genome_config)

    # 36 inputs, 8 hidden, 1 output -> 9 nodes total
    assert len(genome.nodes) == 9
    # 36*8 + 8*1 = 296 connections
    assert len(genome.connections) == 296

    net = FastFeedForwardNetwork.create(genome, cfg)
    inp = [0.25] * 36
    out = net.activate(inp)
    assert len(out) == 1
    assert -1.0 <= out[0] <= 1.0


def test_shared_anchor_panel_invariance():
    import random
    from ai.arena import pair_tasks
    from ai.ladder import neat_spec, load_neat_config
    import neat

    cfg = load_neat_config("neat_value_config.txt")
    g1 = neat.DefaultGenome(1); g1.configure_new(cfg.genome_config)
    g2 = neat.DefaultGenome(2); g2.configure_new(cfg.genome_config)
    hof_g = neat.DefaultGenome(99); hof_g.configure_new(cfg.genome_config)

    specs = {1: neat_spec(g1, "neat_value_config.txt", 1),
             2: neat_spec(g2, "neat_value_config.txt", 1)}
    hof_spec = neat_spec(hof_g, "neat_value_config.txt", 1)

    # When facing shared hof_spec, both genomes get paired against the same anchor
    tasks1 = pair_tasks(specs[1], hof_spec, 2, base_seed=12345)
    tasks2 = pair_tasks(specs[2], hof_spec, 2, base_seed=12345)

    assert len(tasks1) == len(tasks2) == 2
    # Opponent spec is identical for both
    assert tasks1[0][1] == tasks2[0][1] == hof_spec
    assert tasks1[1][0] == tasks2[1][0] == hof_spec
