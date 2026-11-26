"""Turn collapsed action tokens into sets of contiguous k-grams."""

from collections.abc import Sequence

Shingle = tuple[str, ...]


def shingles(tokens: Sequence[str], k: int) -> frozenset[Shingle]:
    """Short sequences have empty shingle sets."""
    if k < 1:
        raise ValueError("k must be positive")
    return frozenset(tuple(tokens[i:i + k]) for i in range(len(tokens) - k + 1))


def exact_jaccard(left: frozenset, right: frozenset) -> float:
    """Compare shingle sets; empty sets carry no matching evidence."""
    shared = len(left & right)
    union = len(left) + len(right) - shared
    return shared / union if union else 0.0
