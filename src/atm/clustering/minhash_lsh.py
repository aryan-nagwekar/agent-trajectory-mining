"""Retrieve MinHash-LSH candidates and verify edges with exact Jaccard."""

import json
from functools import lru_cache
from itertools import combinations

import numpy as np
from datasketch import MinHash, MinHashLSH

from atm.clustering.shingles import Shingle, exact_jaccard


@lru_cache(maxsize=65_536)
def encode(shingle: Shingle) -> bytes:
    return json.dumps(shingle, ensure_ascii=False, separators=(",", ":")).encode()


class Sketcher:
    """Share the seeded permutations across compatible MinHash signatures."""

    def __init__(self, num_perm: int, seed: int):
        self.num_perm = num_perm
        self.seed = seed
        self.permutations = MinHash(num_perm=num_perm, seed=seed).permutations

    def __call__(self, items: frozenset[Shingle]) -> MinHash:
        sketch = MinHash(num_perm=self.num_perm, seed=self.seed, permutations=self.permutations)
        if items:
            sketch.update_batch([encode(item) for item in sorted(items)])
        return sketch


def lsh_edges(sets: list[frozenset], signatures: list[MinHash], threshold: float,
              bands: int, rows: int) -> tuple[set[tuple[int, int]], int]:
    """Use LSH only for retrieval; accept an edge on exact set similarity."""
    if len(sets) != len(signatures):
        raise ValueError("shingle sets and signatures differ in length")
    if not sets:
        return set(), 0
    index = MinHashLSH(threshold=threshold, num_perm=len(signatures[0]), params=(bands, rows))
    for i, signature in enumerate(signatures):
        if sets[i]:
            index.insert(i, signature)
    edges = set()
    candidates = 0
    for i, signature in enumerate(signatures):
        if not sets[i]:
            continue
        for j in index.query(signature):
            if j > i:
                candidates += 1
                if exact_jaccard(sets[i], sets[j]) >= threshold:
                    edges.add((i, j))
    return edges, candidates


def brute_force_edges(sets: list[frozenset], threshold: float) -> set[tuple[int, int]]:
    """Enumerate every pair and compute exact Jaccard for the benchmark."""
    return {(i, j) for i, j in combinations(range(len(sets)), 2)
            if exact_jaccard(sets[i], sets[j]) >= threshold}


def approximate_medoid(signatures: list[MinHash], members: list[int]) -> int:
    """Return the member most similar on average to its community's sketches."""
    hashes = np.stack([signatures[i].hashvalues for i in members])
    totals = np.zeros(len(members))
    for col in range(hashes.shape[1]):
        _, indexes, counts = np.unique(hashes[:, col], return_inverse=True, return_counts=True)
        totals += counts[indexes]
    return members[int(np.argmax(totals))]
