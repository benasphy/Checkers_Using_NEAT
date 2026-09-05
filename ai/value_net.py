"""Evaluation function compiled from a NEAT genome.

The network maps the canonical 36-feature encoding (see ``ai.features``) to a
single tanh output in [-1, 1], interpreted as the expected outcome for the
side to move. For search, the value is scaled to ``VALUE_SCALE`` so it lives
comfortably below mate scores.

Uses a DAG compiler (FastFeedForwardNetwork) that prunes unreachable and
dead-end nodes while preserving neat-python's configured node semantics.
A small hash-keyed cache avoids re-evaluating transpositions.
"""

from __future__ import annotations

from ai.features import encode

VALUE_SCALE = 600.0
_CACHE_MAX = 400_000


class FastFeedForwardNetwork:
    """Feed-forward NEAT evaluator with inactive connections pruned.

    Guarantees that:
    1. Only nodes reachable from inputs and reaching outputs are evaluated.
    2. Topological ordering is strictly respected.
    3. Disconnected outputs retain neat-python's initial value of zero.
    4. Configured activation and aggregation functions are preserved.
    """
    __slots__ = ("num_inputs", "output_idx", "eval_steps", "size")

    def __init__(self, num_inputs: int, output_idx: int, eval_steps: list, size: int):
        self.num_inputs = num_inputs
        self.output_idx = output_idx
        self.eval_steps = eval_steps
        self.size = size

    @classmethod
    def create(cls, genome, config) -> FastFeedForwardNetwork:
        input_keys = list(config.genome_config.input_keys)
        output_keys = list(config.genome_config.output_keys)
        output_key = output_keys[0]

        enabled_conns = {k: cg for k, cg in genome.connections.items() if cg.enabled}

        # 1. Forward reachability from inputs
        reachable_from_inputs = set(input_keys)
        changed = True
        while changed:
            changed = False
            for (i, o) in enabled_conns:
                if i in reachable_from_inputs and o not in reachable_from_inputs:
                    reachable_from_inputs.add(o)
                    changed = True

        # 2. Backward reachability to outputs
        reaches_outputs = set(output_keys)
        changed = True
        while changed:
            changed = False
            for (i, o) in enabled_conns:
                if o in reaches_outputs and i not in reaches_outputs:
                    reaches_outputs.add(i)
                    changed = True

        # Edges outside an input-to-output path must not block NEAT's scheduler.
        active_nodes = reachable_from_inputs & reaches_outputs

        # Build incoming edges for each active node
        incoming = {n: [] for n in active_nodes if n not in input_keys}
        for (i, o), cg in enabled_conns.items():
            if i in active_nodes and o in incoming:
                incoming[o].append((i, cg.weight))

        # Topological sorting
        evaluated = set(input_keys)
        remaining = set(incoming.keys())
        node_order = []

        while remaining:
            ready = sorted(n for n in remaining
                           if all(i in evaluated for i, _ in incoming[n]))
            if not ready:
                cycle_nodes = ", ".join(str(n) for n in sorted(remaining))
                raise ValueError(
                    f"Active NEAT phenotype contains a cycle involving nodes: {cycle_nodes}"
                )
            for n in ready:
                node_order.append(n)
                evaluated.add(n)
                remaining.remove(n)

        all_nodes = list(input_keys) + node_order
        node_to_idx = {k: idx for idx, k in enumerate(all_nodes)}
        if output_key not in node_to_idx:
            node_to_idx[output_key] = len(all_nodes)
            all_nodes.append(output_key)

        eval_steps = []
        for n in node_order:
            node_data = genome.nodes[n]
            act_func = config.genome_config.activation_defs.get(node_data.activation)
            agg_func = config.genome_config.aggregation_function_defs.get(node_data.aggregation)
            indexed_inputs = tuple((node_to_idx[i], w) for i, w in incoming[n])
            eval_steps.append((node_to_idx[n], act_func, agg_func,
                               node_data.bias, node_data.response, indexed_inputs))

        out_idx = node_to_idx[output_key]
        return cls(len(input_keys), out_idx, eval_steps, len(all_nodes))

    def activate(self, inputs: list[float]) -> list[float]:
        if len(inputs) != self.num_inputs:
            raise RuntimeError(f"Expected {self.num_inputs:n} inputs, got {len(inputs):n}")

        vals = [0.0] * self.size
        vals[:self.num_inputs] = inputs
        for target, act_func, agg_func, bias, resp, in_edges in self.eval_steps:
            weighted_inputs = [vals[src] * weight for src, weight in in_edges]
            vals[target] = act_func(bias + resp * agg_func(weighted_inputs))
        return [vals[self.output_idx]]


class ValueNetwork:
    def __init__(self, genome, config):
        self.genome = genome
        self._net = FastFeedForwardNetwork.create(genome, config)
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
