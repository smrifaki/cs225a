"""Bootstrap CI, paired permutation test, Cliff's delta on
per-episode metrics. Used by aggregate.py to compare modes.

All routines are deterministic given a numpy seed.
"""
from __future__ import annotations

import numpy as np


def bootstrap_ci(
    x: np.ndarray,
    n_boot: int = 10_000,
    alpha: float = 0.05,
    rng: np.random.Generator | None = None,
) -> tuple[float, float, float]:
    rng = rng if rng is not None else np.random.default_rng(0)
    n = len(x)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    boots = rng.choice(x, size=(n_boot, n), replace=True).mean(axis=1)
    lo = float(np.quantile(boots, alpha / 2))
    hi = float(np.quantile(boots, 1 - alpha / 2))
    return float(x.mean()), lo, hi


def paired_permutation_test(
    x: np.ndarray,
    y: np.ndarray,
    n_perm: int = 10_000,
    rng: np.random.Generator | None = None,
) -> tuple[float, float]:
    rng = rng if rng is not None else np.random.default_rng(0)
    assert x.shape == y.shape, "paired test requires equal-length arrays"
    d = x - y
    obs = float(d.mean())
    n = len(d)
    if n == 0:
        return float("nan"), float("nan")
    signs = rng.choice([-1.0, 1.0], size=(n_perm, n))
    null = (signs * d).mean(axis=1)
    p = float(np.mean(np.abs(null) >= abs(obs)))
    return obs, p


def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    nx, ny = len(x), len(y)
    if nx == 0 or ny == 0:
        return float("nan")
    gt = (x[:, None] > y[None, :]).sum()
    lt = (x[:, None] < y[None, :]).sum()
    return float((gt - lt) / (nx * ny))


def verdict(p_value: float, delta: float, direction_match: bool) -> str:
    """Verdict bucket given the test direction, p-value, and effect size."""
    if not direction_match:
        return "null"
    if p_value < 0.05 and abs(delta) > 0.147:
        return "confirmed"
    if p_value < 0.05:
        return "suggestive"
    return "null"
