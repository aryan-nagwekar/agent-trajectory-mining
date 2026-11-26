"""Fit training-only reference features and transform observed prefixes."""

from collections import Counter
from dataclasses import dataclass
from itertools import groupby
from pathlib import Path

import numpy as np
import pandas as pd

from atm.clustering.archetypes import fit_archetypes
from atm.clustering.minhash_lsh import Sketcher, approximate_medoid
from atm.clustering.shingles import shingles
from atm.mining.discriminative import candidate_patterns
from atm.mining.prefixspan import Pattern, SequenceIndex
from atm.normalize import collapse

GROUPS = ("cost", "bag", "ngrams", "patterns", "dynamics", "archetypes")


@dataclass(frozen=True)
class Trace:
    tokens: tuple[str, ...]
    actions: tuple[str, ...]
    outcomes: tuple[str, ...]
    chars: tuple[int, ...]
    token_counts: tuple[int | None, ...]


@dataclass(frozen=True)
class Prefix:
    tokens: tuple[str, ...]
    collapsed: tuple[str, ...]
    cost: tuple[float, ...]
    dynamics: tuple[float, ...]
    bag: Counter
    ngrams: Counter
    signatures: dict[tuple[str, int], np.ndarray]


@dataclass(frozen=True)
class Reference:
    patterns: tuple[Pattern, ...]
    archetypes: tuple[tuple[tuple[str, int], np.ndarray], ...]


@dataclass
class FeatureSpace:
    vocabulary: tuple[str, ...]
    ngrams: tuple[Pattern, ...]
    reference: Reference
    max_gap: int

    @classmethod
    def fit(cls, train: list[Prefix], reference: Reference, ngram_limit: int, max_gap: int):
        grams = Counter()
        for prefix in train:
            grams.update(prefix.ngrams)
        vocabulary = tuple(sorted({token for prefix in train for token in prefix.bag}))
        ranking = sorted(grams, key=lambda token: (-grams[token], token))
        return cls(vocabulary, tuple(ranking[:ngram_limit]), reference, max_gap)

    def transform(self, rows: list[Prefix]) -> dict[str, np.ndarray]:
        n = len(rows)
        out = {
            "cost": np.array([p.cost for p in rows], dtype=np.float32),
            "bag": bag_matrix(rows, self.vocabulary),
            "ngrams": np.array([[p.ngrams[key] for key in self.ngrams] for p in rows], dtype=np.float32),
            "patterns": np.zeros((n, len(self.reference.patterns)), dtype=np.float32),
            "dynamics": np.array([p.dynamics for p in rows], dtype=np.float32),
            "archetypes": np.zeros((n, max(1, len(self.reference.archetypes))), dtype=np.float32),
        }
        if self.reference.patterns:
            indexes = {pattern: i for i, pattern in enumerate(self.reference.patterns)}
            projected = SequenceIndex([p.collapsed for p in rows], self.max_gap)
            for pattern, matches in projected.supports(self.reference.patterns):
                out["patterns"][list(matches), indexes[pattern]] = 1
        if self.reference.archetypes:
            for j, (view, medoid) in enumerate(self.reference.archetypes):
                for i, row in enumerate(rows):
                    signature = row.signatures.get(view)
                    if signature is not None:
                        out["archetypes"][i, j] = (medoid == signature).mean()
        return out


def load_traces(path: Path) -> dict[str, Trace]:
    frame = pd.read_parquet(path, columns=[
        "run_id", "step_idx", "token", "action", "outcome", "step_chars", "step_tokens",
    ])
    traces = {}
    for run_id, group in frame.groupby("run_id", sort=False):
        ordered = group.sort_values("step_idx")
        if list(ordered["step_idx"]) != list(range(len(ordered))):
            raise ValueError(f"noncontiguous step_idx for {run_id}")
        traces[run_id] = Trace(
            tuple(ordered["token"]), tuple(ordered["action"]), tuple(ordered["outcome"]),
            tuple(ordered["step_chars"]),
            tuple(None if pd.isna(value) else int(value) for value in ordered["step_tokens"]),
        )
    return traces


def ngrams(sequence: tuple, n: int) -> Counter:
    return Counter(tuple(sequence[i:i + n]) for i in range(len(sequence) - n + 1))


def build_prefix(trace: Trace, t: int, sketcher: Sketcher, views: tuple[tuple[str, int], ...]) -> Prefix:
    """Observe only step indices 0..t-1 and return per-run sufficient statistics."""
    if not 0 < t <= len(trace.tokens):
        raise ValueError("checkpoint outside the observed trajectory")
    tokens, base, outcomes = trace.tokens[:t], trace.actions[:t], trace.outcomes[:t]
    collapsed = tuple(collapse(list(tokens)))
    representations = {"tokens": collapsed, "actions": collapse(list(base))}
    signatures = {}
    for view, k in views:
        items = shingles(representations[view], k)
        if items:
            signatures[view, k] = sketcher(items).hashvalues
    chars, costs = trace.chars[:t], trace.token_counts[:t]
    token_missing = any(count is None for count in costs)
    examples = Counter(base)
    last_edit = max((i for i, action in enumerate(base) if action == "edit"), default=-1)
    last_test = max((i for i, action in enumerate(base) if action == "run_test"), default=-1)
    cycles = 0
    edited = False
    for action in base:
        if action == "edit":
            edited = True
        elif action == "run_test" and edited:
            cycles += 1
            edited = False
    dynamics = (
        max(sum(1 for _ in group) for _, group in groupby(base)),
        sum(value >= 3 for value in ngrams(base, 3).values()),
        cycles,
        sum(status == "error" for status in outcomes) / t,
        sum(action == "run_test" and status == "fail" for action, status in zip(base, outcomes, strict=True))
        / max(examples["run_test"], 1),
        t - 1 - last_edit if last_edit >= 0 else t, t - 1 - last_test if last_test >= 0 else t,
        float(last_edit >= 0), float(last_test >= 0),
    )
    cost = (sum(chars), sum(cost for cost in costs if cost is not None),
            float(token_missing), float("submit" in base))
    return Prefix(tokens, collapsed, cost, dynamics, Counter(tokens), ngrams(tokens, 2) + ngrams(tokens, 3),
                  signatures)


def bag_matrix(rows: list[Prefix], vocabulary: tuple[str, ...]) -> np.ndarray:
    counts = np.array([[prefix.bag[token] for token in vocabulary] for prefix in rows], dtype=np.float32)
    denom = np.array([len(prefix.tokens) for prefix in rows], dtype=np.float32)[:, None]
    return np.concatenate((counts, counts / denom), axis=1)


def matrix(groups: dict[str, np.ndarray], keep: tuple[str, ...] = GROUPS) -> np.ndarray:
    return np.concatenate([groups[name] for name in keep], axis=1)


def fit_reference(training: pd.DataFrame, config: dict) -> Reference:
    """Rank by absolute support difference irrespective of statistical significance."""
    candidates = candidate_patterns(training, config["mining"])
    n_fail = int(training["failed"].sum())
    n_success = len(training) - n_fail
    if not n_success or not n_fail:
        raise ValueError("both classes are required to fit pattern features")
    ordered = sorted(candidates, key=lambda pattern: (
        -abs(candidates[pattern][0] / n_fail - candidates[pattern][1] / n_success),
        len(pattern), pattern,
    ))
    patterns = tuple(ordered[:config["prediction"]["top_patterns"]])
    failed = training.loc[training["failed"] == 1]
    medoids = []
    for view, k in config["prediction"]["archetype_views"]:
        _, signatures, groups, _ = fit_archetypes(
            failed[view].tolist(), {**config["clustering"], "k": k}, config["seed"]
        )
        medoids.extend(((view, k), signatures[approximate_medoid(signatures, members)].hashvalues)
                       for members in groups)
    return Reference(patterns, tuple(medoids))
