"""Check that native exact search agrees with pairwise Python Jaccard."""

import shutil
import subprocess
from pathlib import Path

import pytest

from atm.clustering import window_benchmark as benchmark
from atm.clustering.minhash_lsh import brute_force_edges
from atm.clustering.shingles import shingles
from atm.transfer_patterns import summarize_overlap


def test_windows_include_only_full_strided_slices() -> None:
    rows = benchmark.windows([tuple("abcdef"), tuple("abc")], 4, 2, 2)
    assert rows == [shingles(tuple("abcd"), 2), shingles(tuple("cdef"), 2)]


@pytest.mark.skipif(shutil.which("c++") is None, reason="C++ compiler unavailable")
@pytest.mark.parametrize("threshold", [.3, .4, .5, .6])
def test_native_exact_matches_python_with_duplicates_and_empty_sets(tmp_path: Path, threshold: float) -> None:
    executable = tmp_path / "exact"
    native = Path(benchmark.__file__).with_name("exact_windows.cpp")
    subprocess.run(["c++", "-std=c++17", "-O2", str(native), "-o", str(executable)], check=True)
    items = [shingles(tuple(text), 2) for text in ("abcd", "abcd", "abcx", "ab", "bcd", "x", "")]
    _, matches = benchmark.exact_count(executable, items, threshold, 10)
    assert matches == len(brute_force_edges(items, threshold))


def test_undefined_overlap_when_both_lists_are_empty() -> None:
    overlap, _ = summarize_overlap({"a": set(), "b": set(), "c": {"x"}})
    assert overlap["jaccard"].isna().sum() == 1
    assert (overlap["jaccard"] == 0).sum() == 2
