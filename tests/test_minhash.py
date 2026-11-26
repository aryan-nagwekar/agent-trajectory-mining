"""Validate exact Jaccard, deterministic sketches, and retrieval."""

import random

from atm.clustering.minhash_lsh import Sketcher, brute_force_edges, lsh_edges
from atm.clustering.shingles import exact_jaccard, shingles


def test_minhash_is_close_to_exact_on_seeded_random_pairs() -> None:
    rng = random.Random(42)
    sketcher = Sketcher(128, 42)
    population = [(str(i),) for i in range(200)]
    for _ in range(10):
        a = frozenset(rng.sample(population, 85))
        b = frozenset(rng.sample(population, 85))
        assert abs(sketcher(a).jaccard(sketcher(b)) - exact_jaccard(a, b)) <= 0.10


def test_lsh_excludes_empty_sets_and_verifies_pairs() -> None:
    sets = [shingles(list(tokens), 3) for tokens in ("abcd", "abcd", "vwxyz", "", "")]
    signatures = [Sketcher(128, 42)(items) for items in sets]
    edges, _ = lsh_edges(sets, signatures, .6, 32, 4)
    assert edges == brute_force_edges(sets, .6) == {(0, 1)}
