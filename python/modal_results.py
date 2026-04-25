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

    # Phase model. The peg-in-hole task has four canonical phases:
    # approach (vision-dominant), align (vision still informative),
    # contact (force becomes informative as vision saturates), insert
    # (force dominant). The ground-truth per-modality precision is
    # phase-dependent; the controller does not see the phase label.
    phases = [
        ("approach", 0,   150, 1.4, 0.4),
        ("align",    150, 280, 1.1, 0.7),
        ("contact",  280, 420, 0.7, 1.3),
        ("insert",   420, horizon, 0.4, 1.7),
    ]

    # Forward dynamics: 2D feature per modality. Vision feature
    # variance = sigma_v^2; force feature variance = sigma_f^2. Each
    # tick the controller has a learned Bayesian forward model with
    # running posterior precision per modality. We initialize the
    # model precision uniform across modalities and update it with
    # each observation. The "true" task uncertainty is the entropy
    # of the joint posterior.
    prior_var_v = 1.0
    prior_var_f = 1.0
    var_v, var_f = prior_var_v, prior_var_f

    # BALD-style expected information gain: pick the modality with
    # the larger expected entropy drop, ~ residual^2 / posterior_var,
    # softmaxed to keep some exploration.
    beta = 6.0

    rows: list[dict[str, Any]] = []
    phase_idx = 0
    for t in range(horizon):
        # advance the phase pointer
        while phase_idx + 1 < len(phases) and t >= phases[phase_idx][2]:
            phase_idx += 1
        name, _, _, true_v, true_f = phases[phase_idx]

        # ground-truth residuals (the magnitude of "what the forward
        # model would be wrong by if it stayed at the prior")
        r_v = rng.normal(0.0, true_v)
        r_f = rng.normal(0.0, true_f)

        # expected information gain per modality, ignoring constants
        eig_v = 0.5 * np.log1p(r_v ** 2 / max(var_v, 1e-3))
        eig_f = 0.5 * np.log1p(r_f ** 2 / max(var_f, 1e-3))

        # softmax pick
        logits = beta * np.array([eig_v, eig_f])
        probs = np.exp(logits - logits.max())
        probs /= probs.sum()
        pick = int(rng.choice(2, p=probs))

        # update posterior precision of the picked modality
        if pick == 0:
            var_v = 1.0 / (1.0 / var_v + 1.0)
        else:
            var_f = 1.0 / (1.0 / var_f + 1.0)

        rows.append(
            {
                "step": t,
                "phase": name,
                "residual_vision": float(r_v),
                "residual_force": float(r_f),
                "eig_vision": float(eig_v),
                "eig_force": float(eig_f),
                "pick": "vision" if pick == 0 else "force",
                "var_vision_post": float(var_v),
                "var_force_post": float(var_f),
            }
        )

    # per-phase summary
    summary: list[dict[str, Any]] = []
    for name, lo, hi, true_v, true_f in phases:
        chunk = rows[lo:hi]
        if not chunk:
            continue
        n_v = sum(1 for r in chunk if r["pick"] == "vision")
        n_f = len(chunk) - n_v
        eig_v_mean = float(np.mean([r["eig_vision"] for r in chunk]))
        eig_f_mean = float(np.mean([r["eig_force"] for r in chunk]))
        summary.append(
            {
                "phase": name,
                "steps": len(chunk),
                "true_sigma_vision": true_v,
                "true_sigma_force": true_f,
                "vision_picks": n_v,
                "force_picks": n_f,
                "vision_share": n_v / len(chunk),
                "force_share": n_f / len(chunk),
                "mean_eig_vision": eig_v_mean,
                "mean_eig_force": eig_f_mean,
            }
        )

    return {
        "seed": seed,
        "horizon": horizon,
        "rows": rows,
        "summary": summary,
    }


_PHASE_PALETTE = {
    "approach": "#3b6ea5", "align": "#3aa37a",
    "contact": "#d28244", "insert": "#a44a8a",
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

    # Figure 1: sensor attention across time, with phase boundaries.
    # Use per-step vision picks aggregated across seeds.
    by_step: dict[int, list[int]] = {}
    for r in rows:
        by_step.setdefault(r["step"], []).append(1 if r["pick"] == "vision" else 0)
    steps_sorted = sorted(by_step.keys())
    mean_share = np.array([np.mean(by_step[t]) for t in steps_sorted])
    std_share = np.array([np.std(by_step[t], ddof=1) if len(by_step[t]) > 1 else 0.0
                          for t in steps_sorted])
    window = 25
    kernel = np.ones(window) / window
    smooth_mean = np.convolve(mean_share, kernel, mode="same")
    smooth_std = np.convolve(std_share, kernel, mode="same")

    fig, ax = plt.subplots(figsize=(5.4, 2.6))
    ax.plot(steps_sorted, smooth_mean, color="#3b6ea5", lw=1.4)
    ax.fill_between(steps_sorted, smooth_mean - smooth_std,
                    smooth_mean + smooth_std,
                    color="#3b6ea5", alpha=0.15, linewidth=0)
    ax.set_ylim(-0.05, 1.1)
    ax.set_xlabel("control tick")
    ax.set_ylabel("share of vision picks (smoothed)")

    # phase boundaries from cumulative steps
    cum = 0
    label_centers: list[tuple[int, str]] = []
    for s in summary:
        label_centers.append((cum + s["steps"] // 2, s["phase"]))
        cum += s["steps"]
        ax.axvline(cum, color="#888", linestyle=":", linewidth=0.8)
    for x, name in label_centers:
        ax.text(x, 1.04, name, ha="center", va="bottom",
                fontsize=8, color="#444")
    ax.spines[["top", "right"]].set_visible(False)
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

    # Figure 3: vision-share per phase with error bars across seeds.
    fig, ax = plt.subplots(figsize=(3.6, 2.6))
    phases = [s["phase"] for s in summary]
    means = [s["vision_share_mean"] for s in summary]
    stds = [s.get("vision_share_std", 0.0) for s in summary]
    xs = np.arange(len(phases))
    ax.bar(xs, means, yerr=stds, color="#3b6ea5",
           capsize=3, edgecolor="white", error_kw={"lw": 1})
    ax.axhline(0.5, color="#c0392b", lw=0.8, linestyle="--", label="50/50 split")
    ax.set_xticks(xs)
    ax.set_xticklabels(phases, fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("vision share (mean over seeds)")
    ax.legend(frameon=False, fontsize=7, loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig_c = out_dir / "figures" / "vision_share_per_phase.pdf"
    fig.savefig(fig_c, bbox_inches="tight")
    fig.savefig(fig_c.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)

    return {
        "traces": traces_csv,
        "per_phase": per_phase_csv,
        "fig_a": fig_a,
        "fig_b": fig_b,
        "fig_c": fig_c,
    }


def _aggregate(payloads: list[dict[str, Any]]) -> dict[str, Any]:
    import numpy as np
    # Concatenate traces across seeds (kept as separate rows, tagged
    # by seed) so per-step plots can show the spread.
    rows: list[dict[str, Any]] = []
    for p in payloads:
        for r in p["rows"]:
            rows.append({"seed": p["seed"], **r})

    # Per-phase aggregate stats over seeds: mean and stderr of
    # vision share and mean EIG.
    by_phase: dict[str, list[dict[str, Any]]] = {}
    for p in payloads:
        for s in p["summary"]:
            by_phase.setdefault(s["phase"], []).append(s)
    summary: list[dict[str, Any]] = []
    for phase, items in by_phase.items():
        n = len(items)
        vs = np.array([it["vision_share"] for it in items])
        eig_v = np.array([it["mean_eig_vision"] for it in items])
        eig_f = np.array([it["mean_eig_force"] for it in items])
        summary.append({
            "phase": phase,
            "steps": items[0]["steps"],
            "true_sigma_vision": items[0]["true_sigma_vision"],
            "true_sigma_force": items[0]["true_sigma_force"],
            "vision_share_mean": float(vs.mean()),
            "vision_share_std": float(vs.std(ddof=1) if n > 1 else 0.0),
            "mean_eig_vision": float(eig_v.mean()),
            "mean_eig_force": float(eig_f.mean()),
            "n_seeds": n,
        })
    return {
        "seeds": [p["seed"] for p in payloads],
        "horizon": payloads[0]["horizon"],
        "rows": rows,
        "summary": summary,
    }


@app.local_entrypoint()
def main(
    seeds: str = "0,1,2,3,4",
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
        f"({len(agg['summary'])} phases), "
        f"{paths['fig_a'].relative_to(Path(out_dir))}, "
        f"{paths['fig_b'].relative_to(Path(out_dir))}"
    )
    shares = ", ".join(
        f"{s['phase']}={s['vision_share_mean']:.2f}"
        f"±{s.get('vision_share_std', 0):.2f}"
        for s in agg["summary"]
    )
    print(f"vision-share across seeds {seed_list}: {shares}")
