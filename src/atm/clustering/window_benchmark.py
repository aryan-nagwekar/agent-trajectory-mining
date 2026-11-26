"""Benchmark LSH and exact all-pairs Jaccard on overlapping action windows."""

import argparse
import json
import logging
import math
import os
import subprocess
import tempfile
from collections import Counter
from fractions import Fraction
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import yaml
from datasketch import MinHashLSH

from atm.clustering.minhash_lsh import Sketcher
from atm.clustering.shingles import exact_jaccard, shingles
from atm.clustering.sweep import base_sequences
from atm.normalize import collapse


def windows(sequences, width: int, stride: int, k: int) -> list:
    """Create only full raw windows, then collapse base actions within each window."""
    return [
        shingles(collapse(list(sequence[start:start + width])), k)
        for sequence in sequences for start in range(0, len(sequence) - width + 1, stride)
    ]


def lsh_count(items: list, threshold: float, bands: int, rows: int, seed: int) -> tuple[int, int, int]:
    """Group identical nonempty sets, preserving exact window-pair multiplicity."""
    multiplicity = Counter(item for item in items if item)
    signatures = Sketcher(128, seed)
    index = MinHashLSH(threshold=threshold, num_perm=128, params=(bands, rows))
    unique = list(multiplicity)
    signed = [signatures(item) for item in unique]
    for i, signature in enumerate(signed):
        index.insert(i, signature)
    duplicate_pairs = sum(math.comb(count, 2) for count in multiplicity.values())
    candidates, accepted = duplicate_pairs, duplicate_pairs
    for i, signature in enumerate(signed):
        left = unique[i]
        for j in index.query(signature):
            if j <= i:
                continue
            right = unique[j]
            weight = multiplicity[left] * multiplicity[right]
            candidates += weight
            if exact_jaccard(left, right) >= threshold:
                accepted += weight
    return accepted, candidates, len(unique)


def exact_count(executable: Path, items: list, threshold: float, timeout: float) -> tuple[float, int | None]:
    vocabulary = {token: i for i, token in enumerate(sorted(set().union(*items)))}
    encoded = [tuple(sorted(vocabulary[token] for token in row)) for row in items]
    cut = Fraction(str(threshold))
    payload = f"{len(items)} {cut.numerator} {cut.denominator}\n" + "\n".join(
        " ".join(map(str, (len(row), *row))) for row in encoded
    )
    start = perf_counter()
    try:
        result = subprocess.run([str(executable)], input=payload, text=True, capture_output=True,
                                check=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return perf_counter() - start, None
    return perf_counter() - start, int(result.stdout)


def plot(frame: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import NullLocator, StrMethodFormatter

    fig, (time, recall) = plt.subplots(1, 2, figsize=(9, 3.5), layout="constrained")
    time.loglog(frame["windows"], frame["lsh_seconds"], "o-", label="MinHash LSH + verification", color="#0072B2")
    exact = frame.loc[frame["exact_status"] == "completed"]
    time.loglog(exact["windows"], exact["exact_seconds"], "s--", label="Exact all-pairs", color="#D55E00")
    time.set(xlabel="20-step windows", ylabel="Runtime (seconds)")
    time.set_xticks([5000, 10000, int(frame["windows"].max())])
    time.xaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    time.xaxis.set_minor_locator(NullLocator())
    time.legend(frameon=False, fontsize=9)
    recall.plot(exact["windows"], exact["recall"], "o-", color="#0072B2")
    recall.axhline(.8, color="#777777", linestyle=":")
    recall.set(xlabel="20-step windows", ylabel="Recall against exact pairs", ylim=(0, 1.04))
    recall.set_xticks([5000, 10000, 20000])
    recall.xaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    for ax in (time, recall):
        ax.grid(alpha=.22)
    Path("figures").mkdir(exist_ok=True)
    for extension in ("png", "pdf"):
        fig.savefig(f"figures/lsh_scaling.{extension}", dpi=300)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/expanded.yaml"))
    args = parser.parse_args()
    full = yaml.safe_load(args.config.read_text())
    cfg = full["window_benchmark"]
    choice = json.loads(Path("results/archetype_selection.json").read_text())
    threshold, k = choice["threshold"], choice["k"]
    seed = full["seed"]
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    sequences = base_sequences()
    material = windows(sequences.tolist(), cfg["width"], cfg["stride"], k)
    order = np.random.default_rng(seed).permutation(len(material))
    sizes = sorted({n for n in cfg["sizes"] if n <= len(order)} | {len(order)})
    records = []
    with tempfile.TemporaryDirectory() as staging:
        executable = Path(staging) / "exact_windows"
        subprocess.run(
            [os.environ.get("CXX", "c++"), "-std=c++17", "-O3", str(Path(__file__).with_name("exact_windows.cpp")),
             "-o", str(executable)], check=True, capture_output=True, text=True,
        )
        can_continue = True
        for size in sizes:
            sample = [material[i] for i in order[:size]]
            start = perf_counter()
            matched, candidates, unique = lsh_count(sample, threshold, cfg["bands"], cfg["rows"], seed)
            lsh_seconds = perf_counter() - start
            elapsed, hits = None, None
            status = "over_budget_previous_size"
            if can_continue:
                elapsed, hits = exact_count(executable, sample, threshold, cfg["exact_budget_seconds"])
                status = "completed" if hits is not None else "timed_out"
                can_continue = hits is not None
            if hits is not None and matched > hits:
                raise ValueError("LSH accepted more pairs than exact verification")
            records.append({
                "windows": size, "all_pairs": math.comb(size, 2), "unique_nonempty_sets": unique,
                "lsh_candidates": candidates, "lsh_matches": matched, "exact_matches": hits,
                "recall": matched / hits if hits else None, "lsh_seconds": lsh_seconds,
                "exact_seconds": elapsed, "exact_status": status,
            })
            logging.info("n=%d LSH %.3fs, exact %s [%s], recall %s", size, lsh_seconds, elapsed, status,
                         records[-1]["recall"])
    frame = pd.DataFrame(records)
    frame.to_csv("results/lsh_window_scaling.csv", index=False)
    backup = Path("figures/lsh_scaling_failed_runs.png")
    if not backup.exists():
        for suffix in ("png", "pdf"):
            original = Path(f"figures/lsh_scaling.{suffix}")
            if original.exists():
                original.rename(original.with_stem("lsh_scaling_failed_runs"))
    plot(frame)
    Path("results/lsh_window_config.json").write_text(json.dumps({
        "window_count": len(material), "source_runs": len(sequences), "width": cfg["width"], "stride": cfg["stride"],
        "k": k, "threshold": threshold, "bands": cfg["bands"], "rows": cfg["rows"],
        "unavailable_sizes": [n for n in cfg["sizes"] if n > len(material)],
        "brute_force_budget_seconds": cfg["exact_budget_seconds"], "seed": seed,
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
