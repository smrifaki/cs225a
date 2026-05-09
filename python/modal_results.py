"""Modal pipeline that produces the artefacts checked into `results/`.

The C++ controller runs in the lab's SCL/OpenSai simulator, which is
not pip-installable. This pipeline instead exercises the *decision
layer* of the controller, the BALD-style sensor-attention policy, on
a synthetic dual-modal forward model whose ground truth I control.

Outputs:
  - traces.csv         per-step sensor choice, prediction error, EIG
  - per_phase.csv      summary stats per peg-in-hole phase
  - figures/sensor_attention.{pdf,png}
  - figures/eig_vs_residual.{pdf,png}
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import modal  # type: ignore[import-not-found]

app = modal.App("cs225a-results")
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy==2.2.0", "matplotlib==3.10.0")
)


@app.function(image=image, timeout=600)
def simulate(seed: int = 0, horizon: int = 600) -> dict[str, Any]:
    import numpy as np

    rng = np.random.default_rng(seed)

    # Phase model. The peg-in-hole task has four canonical phases.
    # Three modalities: vision (eye-in-hand camera), force (wrist FT
    # sensor), proprioception (joint encoders). The ground-truth
    # per-modality residual scale is phase-dependent; the controller
    # does not see the phase label.
    #
    # phase tuple = (name, start, end, sigma_vision, sigma_force, sigma_proprio)
    phases = [
        ("approach", 0,   150, 1.4, 0.4, 0.5),
        ("align",    150, 280, 1.1, 0.7, 1.2),
        ("contact",  280, 420, 0.7, 1.3, 0.9),
        ("insert",   420, horizon, 0.4, 1.7, 0.6),
    ]

    # Forward dynamics: 1D feature per modality. Each tick the
    # controller has a learned Bayesian forward model with running
    # posterior precision per modality. Posterior precision starts
    # uniform across modalities and updates with each observation.
    var = {"vision": 1.0, "force": 1.0, "proprio": 1.0}

    # BALD-style expected information gain: pick the modality with
    # the larger expected entropy drop, ~ residual^2 / posterior_var,
    # softmaxed to keep some exploration.
    beta = 6.0
    modalities = ["vision", "force", "proprio"]

    rows: list[dict[str, Any]] = []
    phase_idx = 0
    for t in range(horizon):
        # advance the phase pointer
        while phase_idx + 1 < len(phases) and t >= phases[phase_idx][2]:
            phase_idx += 1
        name, _, _, sv, sf, sp = phases[phase_idx]

        # ground-truth residuals per modality
        r = {
            "vision":  rng.normal(0.0, sv),
            "force":   rng.normal(0.0, sf),
            "proprio": rng.normal(0.0, sp),
        }

        # expected information gain per modality, ignoring constants
        eig = {
            m: 0.5 * np.log1p(r[m] ** 2 / max(var[m], 1e-3))
            for m in modalities
        }

        # softmax pick
        logits = beta * np.array([eig[m] for m in modalities])
        probs = np.exp(logits - logits.max())
        probs /= probs.sum()
        pick_idx = int(rng.choice(len(modalities), p=probs))
        picked = modalities[pick_idx]

        # update posterior precision of the picked modality. The
        # update is residual-weighted: a large residual moves the
        # posterior more than a small one, which matches a Gaussian
        # likelihood with observation noise sigma_obs.
        sigma_obs2 = 1.0
        prec_pre = 1.0 / max(var[picked], 1e-9)
        prec_post = prec_pre + (r[picked] ** 2) / sigma_obs2
        var[picked] = 1.0 / prec_post

        rows.append({
            "step": t,
            "phase": name,
            "residual_vision":   float(r["vision"]),
            "residual_force":    float(r["force"]),
            "residual_proprio":  float(r["proprio"]),
            "eig_vision":   float(eig["vision"]),
            "eig_force":    float(eig["force"]),
            "eig_proprio":  float(eig["proprio"]),
            "pick": picked,
            "var_vision_post":  float(var["vision"]),
            "var_force_post":   float(var["force"]),
            "var_proprio_post": float(var["proprio"]),
        })

    # per-phase summary
    summary: list[dict[str, Any]] = []
    for name, lo, hi, sv, sf, sp in phases:
        chunk = rows[lo:hi]
        if not chunk:
            continue
        n = len(chunk)
        picks = {m: sum(1 for r in chunk if r["pick"] == m) for m in modalities}
        eig_means = {
            m: float(np.mean([r[f"eig_{m}"] for r in chunk]))
            for m in modalities
        }
        summary.append({
            "phase": name,
            "steps": n,
            "true_sigma_vision": sv,
            "true_sigma_force": sf,
            "true_sigma_proprio": sp,
            "vision_picks":  picks["vision"],
            "force_picks":   picks["force"],
            "proprio_picks": picks["proprio"],
            "vision_share":  picks["vision"] / n,
            "force_share":   picks["force"] / n,
            "proprio_share": picks["proprio"] / n,
            "mean_eig_vision":  eig_means["vision"],
            "mean_eig_force":   eig_means["force"],
            "mean_eig_proprio": eig_means["proprio"],
        })

    # Baselines for regret comparison:
    #   - random      : uniform-random modality each tick
    #   - vision_only : always pick vision
    #   - force_only  : always pick force
    #   - oracle      : pick the modality with the highest true sigma
    #     for the current phase (BALD agent does not see phase label).
    #
    # Score function = sum of EIG over the trajectory (each tick the
    # information actually drawn from the picked sensor).
    baselines: dict[str, float] = {}
    bald_eig = 0.0
    for r in rows:
        bald_eig += float(r[f"eig_{r['pick']}"])

    rng_b = np.random.default_rng(seed + 9999)
    baselines["random"] = float(sum(
        r[f"eig_{rng_b.choice(modalities)}"] for r in rows
    ))
    baselines["vision_only"] = float(sum(r["eig_vision"] for r in rows))
    baselines["force_only"]  = float(sum(r["eig_force"]  for r in rows))
    baselines["proprio_only"] = float(sum(r["eig_proprio"] for r in rows))

    # Phase-aware baseline: pick the modality with the largest true
    # sigma in the current phase (still does not see per-tick
    # residuals). This is a strong handcrafted policy.
    phase_aware_eig = 0.0
    for name, lo, hi, sv, sf, sp in phases:
        true_sigmas = {"vision": sv, "force": sf, "proprio": sp}
        best_mod = max(true_sigmas, key=lambda m: true_sigmas[m])
        chunk = rows[lo:hi]
        for r in chunk:
            phase_aware_eig += float(r[f"eig_{best_mod}"])

    # Per-tick oracle: omniscient picker that always selects the
    # modality with the largest realized EIG at this tick.
    tick_oracle_eig = float(sum(
        max(r["eig_vision"], r["eig_force"], r["eig_proprio"])
        for r in rows
    ))

    regret_vs_oracle = {
        "bald":         tick_oracle_eig - bald_eig,
        "phase_aware":  tick_oracle_eig - phase_aware_eig,
        "random":       tick_oracle_eig - baselines["random"],
        "vision_only":  tick_oracle_eig - baselines["vision_only"],
        "force_only":   tick_oracle_eig - baselines["force_only"],
        "proprio_only": tick_oracle_eig - baselines["proprio_only"],
    }

    # Per-phase oracle-match rate: fraction of ticks where the BALD
    # agent picked the modality with the largest ground-truth sigma
    # in that phase. The agent does not see phase labels; this is a
    # quality-of-inference diagnostic.
    oracle_match: list[dict[str, Any]] = []
    for name, lo, hi, sv, sf, sp in phases:
        true_sigmas = {"vision": sv, "force": sf, "proprio": sp}
        best_mod = max(true_sigmas, key=lambda m: true_sigmas[m])
        chunk = rows[lo:hi]
        if not chunk:
            continue
        match = sum(1 for r in chunk if r["pick"] == best_mod) / len(chunk)
        oracle_match.append({
            "phase": name,
            "best_modality": best_mod,
            "best_sigma": true_sigmas[best_mod],
            "match_rate": float(match),
            "n_ticks": len(chunk),
        })

    return {
        "seed": seed,
        "horizon": horizon,
        "rows": rows,
        "summary": summary,
        "bald_eig":         bald_eig,
        "phase_aware_eig":  phase_aware_eig,
        "tick_oracle_eig":  tick_oracle_eig,
        "baseline_eig":     baselines,
        "regret_vs_oracle": regret_vs_oracle,
        "oracle_match":     oracle_match,
    }


_PHASE_PALETTE = {
    "approach": "#3b6ea5", "align": "#3aa37a",
    "contact": "#d28244", "insert": "#a44a8a",
}

_MODALITY_PALETTE = {
    "vision":  "#3b6ea5",
    "force":   "#d28244",
    "proprio": "#3aa37a",
}


def _apply_style(plt) -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.4,
        "lines.markersize": 4,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    })


def write_outputs(payload: dict[str, Any], out_dir: Path) -> dict[str, Path]:
    import numpy as np
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _apply_style(plt)

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)

    rows = payload["rows"]
    summary = payload["summary"]

    traces_csv = out_dir / "traces.csv"
    with traces_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    per_phase_csv = out_dir / "per_phase.csv"
    with per_phase_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)

    regret_table = payload.get("regret_table") or []
    regret_csv = out_dir / "regret_vs_oracle.csv"
    if regret_table:
        with regret_csv.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(regret_table[0].keys()))
            writer.writeheader()
            writer.writerows(regret_table)

    om = payload.get("oracle_match") or []
    om_csv = out_dir / "oracle_match.csv"
    if om:
        with om_csv.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(om[0].keys()))
            writer.writeheader()
            writer.writerows(om)

    # Figure 1: sensor attention across time, three modalities.
    modalities = ("vision", "force", "proprio")
    by_step: dict[int, dict[str, list[int]]] = {}
    for r in rows:
        for m in modalities:
            by_step.setdefault(r["step"], {m: [] for m in modalities})
        for m in modalities:
            by_step[r["step"]][m].append(1 if r["pick"] == m else 0)
    steps_sorted = sorted(by_step.keys())
    window = 25
    kernel = np.ones(window) / window

    fig, ax = plt.subplots(figsize=(5.6, 2.8))
    for m in modalities:
        mean_share = np.array(
            [np.mean(by_step[t][m]) for t in steps_sorted]
        )
        smooth = np.convolve(mean_share, kernel, mode="same")
        ax.plot(steps_sorted, smooth, color=_MODALITY_PALETTE[m], label=m)
    ax.set_ylim(-0.05, 1.1)
    ax.set_xlabel("control tick")
    ax.set_ylabel("share of picks (smoothed)")

    cum = 0
    label_centers: list[tuple[int, str]] = []
    for s in summary:
        label_centers.append((cum + s["steps"] // 2, s["phase"]))
        cum += s["steps"]
        ax.axvline(cum, color="#888", linestyle=":", linewidth=0.8)
    for x, name in label_centers:
        ax.text(x, 1.04, name, ha="center", va="bottom",
                fontsize=8, color="#444")
    ax.legend(frameon=False, loc="upper right", ncol=3, columnspacing=1.2)
    fig.tight_layout()
    fig_a = out_dir / "figures" / "sensor_attention.pdf"
    fig.savefig(fig_a, bbox_inches="tight")
    fig.savefig(fig_a.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)

    # Figure 2: EIG scatter, color by phase.
    fig, ax = plt.subplots(figsize=(3.6, 3.0))
    for name, color in _PHASE_PALETTE.items():
        xs = [r["eig_vision"] for r in rows if r["phase"] == name]
        ys = [r["eig_force"] for r in rows if r["phase"] == name]
        ax.scatter(xs, ys, s=8, color=color, alpha=0.6, label=name,
                   edgecolor="none")
    lim = max(
        max(r["eig_vision"] for r in rows),
        max(r["eig_force"] for r in rows),
    )
    ax.plot([0, lim], [0, lim], color="#888", lw=0.8, linestyle="--")
    ax.set_xlabel("EIG, vision")
    ax.set_ylabel("EIG, force")
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig_b = out_dir / "figures" / "eig_vs_residual.pdf"
    fig.savefig(fig_b, bbox_inches="tight")
    fig.savefig(fig_b.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)

    # Figure 3: stacked modality share per phase with seed-std error bars.
    fig, ax = plt.subplots(figsize=(4.4, 2.8))
    phases = [s["phase"] for s in summary]
    xs = np.arange(len(phases))
    width = 0.27
    for i, m in enumerate(modalities):
        means = [s[f"{m}_share_mean"] for s in summary]
        stds  = [s.get(f"{m}_share_std", 0.0) for s in summary]
        ax.bar(xs + (i - 1) * width, means, width=width, yerr=stds,
               label=m, color=_MODALITY_PALETTE[m],
               capsize=3, edgecolor="white", error_kw={"lw": 1})
    ax.set_xticks(xs)
    ax.set_xticklabels(phases)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("modality share (mean over seeds)")
    ax.legend(frameon=False, loc="upper right", ncol=3, columnspacing=1.0)
    fig.tight_layout()
    fig_c = out_dir / "figures" / "modality_share_per_phase.pdf"
    fig.savefig(fig_c)
    fig.savefig(fig_c.with_suffix(".png"), dpi=200)
    plt.close(fig)

    # Figure: per-modality EIG distribution. Box plot per modality
    # across phases shows when each sensor's expected information
    # gain peaks. Vision is highest in approach, force in insert.
    fig, ax = plt.subplots(figsize=(4.4, 3.0))
    phase_names = [s["phase"] for s in summary]
    width = 0.27
    xs = np.arange(len(phase_names))
    for i, m in enumerate(modalities):
        means = []
        stds  = []
        for s in summary:
            phase_rows = [r for r in rows if r["phase"] == s["phase"]]
            vals = [float(r[f"eig_{m}"]) for r in phase_rows]
            means.append(float(np.mean(vals)))
            stds.append(float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0)
        ax.bar(xs + (i - 1) * width, means, width=width, yerr=stds,
               label=m, color=_MODALITY_PALETTE[m],
               capsize=3, edgecolor="white", error_kw={"lw": 0.8})
    ax.set_xticks(xs)
    ax.set_xticklabels(phase_names)
    ax.set_ylabel("mean EIG (over ticks in phase)")
    ax.legend(frameon=False, loc="upper left", ncol=3, columnspacing=1.0)
    fig.tight_layout()
    calib_fig = out_dir / "figures" / "eig_per_phase.pdf"
    fig.savefig(calib_fig)
    fig.savefig(calib_fig.with_suffix(".png"), dpi=200)
    plt.close(fig)

    # Figure 4: posterior precision evolution (1/var) per modality
    # across time, averaged over seeds. Reveals when each modality
    # gets sharp.
    by_step_var: dict[int, dict[str, list[float]]] = {}
    for r in rows:
        by_step_var.setdefault(r["step"], {m: [] for m in modalities})
        for m in modalities:
            by_step_var[r["step"]][m].append(1.0 / max(r[f"var_{m}_post"], 1e-6))
    fig, ax = plt.subplots(figsize=(5.6, 2.8))
    for m in modalities:
        prec = np.array([
            float(np.mean(by_step_var[t][m])) for t in steps_sorted
        ])
        ax.plot(steps_sorted, prec, color=_MODALITY_PALETTE[m], label=m)
    cum = 0
    for s in summary:
        cum += s["steps"]
        ax.axvline(cum, color="#888", linestyle=":", linewidth=0.8)
    ax.set_xlabel("control tick")
    ax.set_ylabel("posterior precision (1 / var)")
    ax.legend(frameon=False, loc="upper left", ncol=3, columnspacing=1.2)
    fig.tight_layout()
    fig_d = out_dir / "figures" / "posterior_precision.pdf"
    fig.savefig(fig_d)
    fig.savefig(fig_d.with_suffix(".png"), dpi=200)
    plt.close(fig)

    # Composite: 3 panels.
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.2))

    # Panel A: per-modality share over time
    ax = axes[0]
    for m in modalities:
        mean_share = np.array([np.mean(by_step[t][m]) for t in steps_sorted])
        smooth = np.convolve(mean_share, kernel, mode="same")
        ax.plot(steps_sorted, smooth, color=_MODALITY_PALETTE[m], label=m)
    ax.set_ylim(-0.05, 1.1)
    ax.set_xlabel("control tick")
    ax.set_ylabel("share of picks (smoothed)")
    ax.set_title("a) modality share over time")
    cum = 0
    for s in summary:
        cum += s["steps"]
        ax.axvline(cum, color="#888", linestyle=":", linewidth=0.8)
    ax.legend(frameon=False, loc="upper right", ncol=3, columnspacing=1.0)

    # Panel B: per-phase modality share with seed-std bars
    ax = axes[1]
    phases_list = [s["phase"] for s in summary]
    xs = np.arange(len(phases_list))
    width = 0.27
    for i, m in enumerate(modalities):
        ms = [s[f"{m}_share_mean"] for s in summary]
        ss = [s.get(f"{m}_share_std", 0.0) for s in summary]
        ax.bar(xs + (i - 1) * width, ms, width=width, yerr=ss,
               label=m, color=_MODALITY_PALETTE[m],
               capsize=3, edgecolor="white", error_kw={"lw": 1})
    ax.set_xticks(xs)
    ax.set_xticklabels(phases_list)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("modality share")
    ax.set_title("b) share per phase")

    # Panel C: posterior precision over time
    ax = axes[2]
    for m in modalities:
        prec = np.array([float(np.mean(by_step_var[t][m])) for t in steps_sorted])
        ax.plot(steps_sorted, prec, color=_MODALITY_PALETTE[m], label=m)
    cum = 0
    for s in summary:
        cum += s["steps"]
        ax.axvline(cum, color="#888", linestyle=":", linewidth=0.8)
    ax.set_xlabel("control tick")
    ax.set_ylabel("posterior precision")
    ax.set_title("c) posterior precision")

    fig.tight_layout()
    composite_fig = out_dir / "figures" / "composite.pdf"
    fig.savefig(composite_fig)
    fig.savefig(composite_fig.with_suffix(".png"), dpi=200)
    plt.close(fig)

    return {
        "traces": traces_csv,
        "per_phase": per_phase_csv,
        "fig_a": fig_a,
        "fig_b": fig_b,
        "fig_c": fig_c,
        "fig_d": fig_d,
        "fig_composite": composite_fig,
        "fig_eig_per_phase": calib_fig,
    }


def _aggregate(payloads: list[dict[str, Any]]) -> dict[str, Any]:
    import numpy as np
    # Concatenate traces across seeds (kept as separate rows, tagged
    # by seed) so per-step plots can show the spread.
    rows: list[dict[str, Any]] = []
    for p in payloads:
        for r in p["rows"]:
            rows.append({"seed": p["seed"], **r})

    # Per-phase aggregate stats over seeds.
    by_phase: dict[str, list[dict[str, Any]]] = {}
    for p in payloads:
        for s in p["summary"]:
            by_phase.setdefault(s["phase"], []).append(s)
    summary: list[dict[str, Any]] = []
    for phase, items in by_phase.items():
        n = len(items)
        shares = {
            m: np.array([it[f"{m}_share"] for it in items])
            for m in ("vision", "force", "proprio")
        }
        eigs = {
            m: np.array([it[f"mean_eig_{m}"] for it in items])
            for m in ("vision", "force", "proprio")
        }
        row = {
            "phase": phase,
            "steps": items[0]["steps"],
            "true_sigma_vision": items[0]["true_sigma_vision"],
            "true_sigma_force": items[0]["true_sigma_force"],
            "true_sigma_proprio": items[0]["true_sigma_proprio"],
            "n_seeds": n,
        }
        for m in ("vision", "force", "proprio"):
            row[f"{m}_share_mean"] = float(shares[m].mean())
            row[f"{m}_share_std"]  = float(shares[m].std(ddof=1) if n > 1 else 0.0)
            row[f"mean_eig_{m}"]   = float(eigs[m].mean())
        summary.append(row)
    # Aggregate oracle-match rate per phase.
    match_by_phase: dict[str, list[float]] = {}
    for p in payloads:
        for r in p.get("oracle_match", []):
            match_by_phase.setdefault(r["phase"], []).append(r["match_rate"])
    oracle_match = []
    for phase, vals in match_by_phase.items():
        arr = np.array(vals, dtype=float)
        oracle_match.append({
            "phase": phase,
            "match_rate_mean": float(arr.mean()),
            "match_rate_std": float(arr.std(ddof=1) if arr.size > 1 else 0.0),
            "n_seeds": int(arr.size),
        })

    # Aggregate regret vs oracle across seeds.
    regret_keys = list(payloads[0].get("regret_vs_oracle", {}).keys())
    by_method: dict[str, list[float]] = {k: [] for k in regret_keys}
    for p in payloads:
        for k, v in p.get("regret_vs_oracle", {}).items():
            by_method[k].append(float(v))
    regret_table = []
    for k, vals in by_method.items():
        arr = np.array(vals, dtype=float)
        regret_table.append({
            "method": k,
            "regret_mean": float(arr.mean()),
            "regret_std": float(arr.std(ddof=1) if arr.size > 1 else 0.0),
            "n_seeds": int(arr.size),
        })

    return {
        "seeds": [p["seed"] for p in payloads],
        "horizon": payloads[0]["horizon"],
        "rows": rows,
        "summary": summary,
        "regret_table": regret_table,
        "oracle_match": oracle_match,
    }


@app.local_entrypoint()
def main(
    seeds: str = "0,1,2,3,4,5,6,7",
    horizon: int = 600,
    out_dir: str = "results",
) -> None:
    seed_list = [int(s) for s in seeds.split(",") if s.strip()]
    args = [(s, int(horizon)) for s in seed_list]
    payloads = list(simulate.starmap(args))
    agg = _aggregate(payloads)
    paths = write_outputs(agg, Path(out_dir))
    print(
        f"wrote traces.csv ({len(agg['rows'])} rows), per_phase.csv "
        f"({len(agg['summary'])} phases), 4 figures"
    )
    for m in ("vision", "force", "proprio"):
        shares = ", ".join(
            f"{s['phase']}={s[f'{m}_share_mean']:.2f}"
            f"±{s.get(f'{m}_share_std', 0):.2f}"
            for s in agg["summary"]
        )
        print(f"{m:>7} share across {len(seed_list)} seeds: {shares}")
