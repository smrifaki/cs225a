"""Aggregate per-seed run logs into headline metrics.

Each `run_sim` invocation writes a CSV per episode under
runs/seed_<seed>/<mode>/. This script pools across seeds and modes
and emits the headline success-rate table plus per-sensor-mode
usage breakdown.
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

MODES = ("vision_only", "force_only", "always_fuse", "selector")
SEEDS = (42, 1337, 2024)


def _load_episode_csvs(run_dir: Path) -> list[dict]:
    rows: list[dict] = []
    for csv_path in sorted(run_dir.glob("episode_*.csv")):
        with csv_path.open() as f:
            rows.extend(csv.DictReader(f))
    return rows


def aggregate(runs_root: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    per_mode_success: dict[str, list[float]] = defaultdict(list)
    per_mode_sensor_reads: dict[str, list[int]] = defaultdict(list)
    per_mode_episode_time: dict[str, list[float]] = defaultdict(list)
    for seed in SEEDS:
        for mode in MODES:
            run_dir = runs_root / f"seed_{seed}" / mode
            if not run_dir.exists():
                continue
            for r in _load_episode_csvs(run_dir):
                per_mode_success[mode].append(int(r.get("success", 0)))
                per_mode_sensor_reads[mode].append(int(r.get("sensor_reads", 0)))
                per_mode_episode_time[mode].append(float(r.get("seconds", 0.0)))

    headline = out_dir / "headline.csv"
    with headline.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "mode", "n_episodes", "success_rate",
            "sensor_reads_mean", "sensor_reads_std",
            "episode_seconds_mean",
        ])
        for mode in MODES:
            s = per_mode_success.get(mode, [])
            r = per_mode_sensor_reads.get(mode, [])
            t = per_mode_episode_time.get(mode, [])
            if not s:
                continue
            w.writerow([
                mode,
                len(s),
                f"{mean(s):.4f}",
                f"{mean(r):.1f}" if r else "",
                f"{stdev(r):.1f}" if len(r) > 1 else "",
                f"{mean(t):.2f}" if t else "",
            ])
    print(f"wrote {headline}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--runs-root", type=Path, default=Path("runs"))
    p.add_argument("--out-dir", type=Path, default=Path("runs/aggregated"))
    args = p.parse_args()
    aggregate(args.runs_root, args.out_dir)


if __name__ == "__main__":
    main()
