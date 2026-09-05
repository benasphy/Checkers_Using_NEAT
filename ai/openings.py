"""Versioned, disjoint opening suites for training, validation, and testing.

Provides deterministic, reproducible opening positions for paired matches:
- Training Openings (32 four-ply positions): used only for fitness and HOF games.
- Validation Openings (16 two-ply positions): used for progress probes.
- Test Openings (32 two-ply positions): strictly held out for final evaluation.

All positions are generated deterministically from legal 2-ply and 4-ply sequences,
ensuring symmetry and zero data leakage.
"""

from __future__ import annotations

from dataclasses import dataclass

from checkers.bitboard import Position, Move

OPENING_SUITE_VERSION = "american-checkers-split-v2"


@dataclass(frozen=True)
class Opening:
    id: str
    name: str
    position: Position
    move_history: tuple[Move, ...]

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "turn": self.position.turn,
            "moves": [(m.fr, m.to, m.cap, m.path) for m in self.move_history],
        }


def _generate_all_2ply_openings() -> list[tuple[str, tuple[Move, ...], Position]]:
    """Generate all 49 legal 2-ply opening positions from the initial board."""
    initial = Position.initial()
    openings = []
    p1_moves = sorted(initial.legal_moves(), key=lambda m: (m.fr, m.to))
    for m1 in p1_moves:
        p1_pos = initial.apply(m1)
        p2_moves = sorted(p1_pos.legal_moves(), key=lambda m: (m.fr, m.to))
        for m2 in p2_moves:
            p2_pos = p1_pos.apply(m2)
            name = f"{m1.fr}-{m1.to}/{m2.fr}-{m2.to}"
            openings.append((name, (m1, m2), p2_pos))
    return openings


def _generate_training_openings(count: int = 32) -> list[Opening]:
    candidates = []
    seen = set()
    for prefix, history, pos in _generate_all_2ply_openings():
        for m3 in sorted(pos.legal_moves(), key=lambda m: (m.fr, m.to, m.cap)):
            p3 = pos.apply(m3)
            for m4 in sorted(p3.legal_moves(), key=lambda m: (m.fr, m.to, m.cap)):
                p4 = p3.apply(m4)
                key = (p4.m1, p4.k1, p4.m2, p4.k2, p4.turn, p4.hmc)
                if key in seen or not p4.legal_moves():
                    continue
                seen.add(key)
                name = f"{prefix}/{m3.fr}-{m3.to}/{m4.fr}-{m4.to}"
                candidates.append((name, history + (m3, m4), p4))
    indices = [i * len(candidates) // count for i in range(count)]
    return [Opening(f"train_{n:02d}", candidates[i][0], candidates[i][2],
                    candidates[i][1])
            for n, i in enumerate(indices)]


def _build_suites() -> tuple[list[Opening], list[Opening], list[Opening]]:
    all_2ply = _generate_all_2ply_openings()
    # 49 total 2-ply positions.
    # We assign:
    # - 16 positions to Validation (indices 0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45)
    # - 32 positions to Test (all remaining indices up to 48)

    val_indices = set(range(0, 48, 3))  # 16 items
    test_indices = set(range(48)) - val_indices  # 32 items

    val_list = []
    for idx in sorted(val_indices):
        name, history, pos = all_2ply[idx]
        oid = f"val_{len(val_list):02d}"
        val_list.append(Opening(id=oid, name=name, position=pos, move_history=history))

    test_list = []
    for idx in sorted(test_indices):
        name, history, pos = all_2ply[idx]
        oid = f"test_{len(test_list):02d}"
        test_list.append(Opening(id=oid, name=name, position=pos, move_history=history))

    return _generate_training_openings(), val_list, test_list


_TRAIN_OPENINGS, _VAL_OPENINGS, _TEST_OPENINGS = _build_suites()
_ALL_OPENINGS = {
    op.id: op for op in _TRAIN_OPENINGS + _VAL_OPENINGS + _TEST_OPENINGS
}


def get_training_openings(count: int | None = None) -> list[Opening]:
    """Return frozen fitness openings (up to 32 positions)."""
    if count is None:
        return list(_TRAIN_OPENINGS)
    return list(_TRAIN_OPENINGS[:count])


def get_validation_openings(count: int | None = None) -> list[Opening]:
    """Return frozen validation openings (up to 16 positions)."""
    if count is None:
        return list(_VAL_OPENINGS)
    return list(_VAL_OPENINGS[:count])


def get_test_openings(count: int | None = None) -> list[Opening]:
    """Return frozen held-out test openings (up to 32 positions)."""
    if count is None:
        return list(_TEST_OPENINGS)
    return list(_TEST_OPENINGS[:count])


def get_opening_by_id(opening_id: str) -> Opening | None:
    return _ALL_OPENINGS.get(opening_id)
