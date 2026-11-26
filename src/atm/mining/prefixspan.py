"""PrefixSpan with bounded gaps and sequence-level support."""

from collections.abc import Iterable, Sequence
from math import ceil

Pattern = tuple[str, ...]
Projection = dict[int, int]


class SequenceIndex:
    """Store token positions as bit masks, keyed by sequence index."""

    def __init__(self, sequences: Sequence[Sequence[str]], max_gap: int):
        if max_gap < 0:
            raise ValueError("max_gap must be nonnegative")
        self.size = len(sequences)
        self.max_gap = max_gap
        self.positions: dict[str, Projection] = {}
        for instance, sequence in enumerate(sequences):
            for pos, token in enumerate(sequence):
                occurrences = self.positions.setdefault(token, {})
                occurrences[instance] = occurrences.get(instance, 0) | (1 << pos)

    def following(self, projection: Projection) -> Projection:
        """Find positions within max_gap intervening tokens of any endpoint."""
        following = {}
        for instance, ends in projection.items():
            reachable = 0
            for offset in range(1, self.max_gap + 2):
                reachable |= ends << offset
            following[instance] = reachable
        return following

    @staticmethod
    def extend(left: Projection, right: Projection) -> Projection:
        """Intersect endpoints with the positions of the next token."""
        if len(left) > len(right):
            left, right = right, left
        return {row: matched for row, mask in left.items() if (matched := mask & right.get(row, 0))}

    def supports(self, patterns: Iterable[Pattern]) -> Iterable[tuple[Pattern, Projection]]:
        """Evaluate candidates with shared prefix projections."""
        trie: dict = {}
        for pattern in patterns:
            if not pattern:
                raise ValueError("patterns must be nonempty")
            node = trie
            for token in pattern:
                node = node.setdefault(token, {})
            node[None] = {}

        def visit(node: dict, prefix: Pattern, projection: Projection | None):
            if None in node:
                yield prefix, projection
            reachable = None if projection is None else self.following(projection)
            for token, children in node.items():
                if token is None:
                    continue
                positions = self.positions.get(token, {})
                extended = positions if reachable is None else self.extend(reachable, positions)
                yield from visit(children, prefix + (token,), extended)

        yield from visit(trie, (), None)


def mine(sequences: Sequence[Sequence[str]], min_support: float, max_pattern_len: int, max_gap: int) -> dict[Pattern, int]:
    """Grow from frequent prefixes, preserving every possible ending position."""
    if not 0 < min_support <= 1 or max_pattern_len < 1:
        raise ValueError("min_support must be within (0, 1] and max_pattern_len must be positive")
    index = SequenceIndex(sequences, max_gap)
    threshold = max(1, ceil(min_support * index.size))
    roots = {token: positions for token, positions in sorted(index.positions.items()) if len(positions) >= threshold}
    patterns: dict[Pattern, int] = {}

    def visit(prefix: Pattern, projection: Projection) -> None:
        patterns[prefix] = len(projection)
        if len(prefix) == max_pattern_len:
            return
        reachable = index.following(projection)
        for token, positions in roots.items():
            projected = index.extend(reachable, positions)
            if len(projected) >= threshold:
                visit(prefix + (token,), projected)

    for token, positions in roots.items():
        visit((token,), positions)
    return patterns
