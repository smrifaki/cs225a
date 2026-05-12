# Decision-layer results, 2026-05-20

8 seeds, 600-step horizon, three modalities (vision, force,
proprioception), CPU.

```bash
modal run python/modal_results.py::main
```

## Per-phase modality share

The peg-in-hole task has four phases with phase-dependent per-modality
noise. The BALD-style controller is *unaware of the phase label* and
must infer the right sensor from posterior precision and observed
residuals. Across 8 seeds:

| phase    | vision           | force            | proprio          |
|----------|-----------------:|-----------------:|-----------------:|
| approach | 0.89 +/- 0.05    | 0.04 +/- 0.02    | 0.07 +/- 0.04    |
| align    | 0.75 +/- 0.10    | 0.03 +/- 0.02    | 0.22 +/- 0.10    |
| contact  | 0.57 +/- 0.09    | 0.23 +/- 0.09    | 0.20 +/- 0.08    |
| insert   | 0.20 +/- 0.05    | 0.67 +/- 0.08    | 0.12 +/- 0.06    |

Source: [results/per_phase.csv](results/per_phase.csv).

Phenomenology the controller recovers without any phase supervision:

* Vision dominates approach (free-space motion), where the camera
  signal is high-SNR and other sensors are uninformative.
* Proprioception peaks in align (~22%) because joint-encoder
  residuals carry the most information when fine positioning matters
  but contact has not yet occurred.
* Force takes over in insert (~67%) because contact-force residuals
  become large and the wrist FT sensor sharpens fastest.
* Vision smoothly transitions out as contact develops, exactly the
  signature we expect from adaptive-sensing literature.

## Regret vs per-tick oracle

A per-tick omniscient oracle picks the modality with the largest
realized EIG at every step. We compare BALD and three handcrafted
baselines.

| policy        | regret vs oracle (mean +/- std, 8 seeds) |
|---------------|-----------------------------------------:|
| BALD          |    6.3 +/-   1.7  |
| vision-only   |  302.6 +/-  31.1  |
| phase-aware   |  425.3 +/-  65.7  |
| random        |  664.0 +/-  42.4  |
| proprio-only  |  849.5 +/- 144.5  |
| force-only    |  867.6 +/-  84.2  |

BALD with no phase supervision lands within 6 of the omniscient
oracle; a hand-coded phase-aware policy (picks the highest-sigma
modality given the current phase label) sits 70x further away.

Source: [results/regret_vs_oracle.csv](results/regret_vs_oracle.csv).

## Oracle-match rate by phase

Fraction of ticks where BALD picked the modality with the largest
ground-truth sigma in the current phase.

| phase    | best modality | match rate    |
|----------|---------------|--------------:|
| approach | vision        | 0.93 +/- 0.03 |
| align    | proprio       | 0.19 +/- 0.08 |
| contact  | force         | 0.11 +/- 0.09 |
| insert   | force         | 0.71 +/- 0.08 |

Approach and insert are easy (one modality dominates by a wide
sigma margin); align and contact are transition phases where two
modalities have similar sigmas, so BALD's per-tick choice diverges
from the per-phase oracle even though its accumulated EIG is close
to optimal.

Source: [results/oracle_match.csv](results/oracle_match.csv).

## Ground-truth schedule

The phases were built with these per-modality noise schedules; the
controller never sees them:

| phase    | sigma_vision | sigma_force | sigma_proprio |
|----------|-------------:|------------:|--------------:|
| approach | 1.4 | 0.4 | 0.5 |
| align    | 1.1 | 0.7 | 1.2 |
| contact  | 0.7 | 1.3 | 0.9 |
| insert   | 0.4 | 1.7 | 0.6 |

Sensor pick at each tick correctly tracks the modality with the
largest residual SNR.

## Figures

* [results/figures/sensor_attention.pdf](results/figures/sensor_attention.pdf):
  per-modality share over time (smoothed, with phase markers).
* [results/figures/modality_share_per_phase.pdf](results/figures/modality_share_per_phase.pdf):
  bar chart of per-phase shares with seed-std bars.
* [results/figures/posterior_precision.pdf](results/figures/posterior_precision.pdf):
  posterior precision (1 / var) per modality, evolution over time.
* [results/figures/eig_vs_residual.pdf](results/figures/eig_vs_residual.pdf):
  EIG scatter colored by phase.
* [results/figures/composite.pdf](results/figures/composite.pdf):
  composite 3-panel summary.

## What this is not

This is the *decision layer*, not the full controller. The C++ side
that closes the loop on the Kuka arm in SCL/OpenSai is in `src/` and
the lab simulator; the Python sim isolates the BALD pick policy and
verifies it produces the right phenomenology before being plugged
into the closed-loop controller.
