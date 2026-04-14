"""Tests for stats.py."""
from __future__ import annotations

import numpy as np
import pytest

from stats import bootstrap_ci, cliffs_delta, paired_permutation_test, verdict


def test_bootstrap_ci_covers_known_mean():
    rng = np.random.default_rng(0)
    x = rng.normal(0.7, 1.0, size=300)
    m, lo, hi = bootstrap_ci(x, n_boot=2000, rng=rng)
    assert lo <= 0.7 <= hi
    assert lo < m < hi


def test_bootstrap_ci_empty_returns_nan():
    m, lo, hi = bootstrap_ci(np.array([]))
    assert np.isnan(m) and np.isnan(lo) and np.isnan(hi)


def test_permutation_test_calibrated_under_null():
    rng = np.random.default_rng(1)
    rejections = 0
    trials = 100
    for _ in range(trials):
        x = rng.normal(0, 1, size=50)
        y = rng.normal(0, 1, size=50)
        _, p = paired_permutation_test(x, y, n_perm=500, rng=rng)
        if p < 0.05:
            rejections += 1
    rate = rejections / trials
    assert 0.0 < rate < 0.15


def test_cliffs_delta_extremes():
    a = np.full(50, 1.0)
    b = np.full(50, 0.0)
    assert cliffs_delta(a, b) == pytest.approx(1.0)
    assert cliffs_delta(b, a) == pytest.approx(-1.0)


def test_verdict_buckets():
    # Confirmed: p<0.05, |delta|>0.147, direction matches.
    assert verdict(0.01, 0.2, True) == "confirmed"
    # Suggestive: p<0.05 but delta too small.
    assert verdict(0.01, 0.05, True) == "suggestive"
    # Null: direction does not match.
    assert verdict(0.01, 0.2, False) == "null"
    # Null: p not significant.
    assert verdict(0.5, 0.5, True) == "null"
