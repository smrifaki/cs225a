# Design decisions

Running log of the choices that shape the decision-layer model and
the headline numbers in [RESULTS.md](../RESULTS.md).

## Synthetic peg-in-hole MDP

* **Four phases (approach, align, contact, insert) with per-phase
  per-modality sigma.** The phase boundaries are proportional to
  horizon so the schedule scales linearly with task length; see
  the horizon sweep in `results/horizon_sweep.csv`.
* **Three modalities (vision, force, proprio).** A two-modality
  model has too little structure to test attention; three is the
  minimum that gives a non-trivial best-modality changing across
  phases.

## Posterior update rule

* **Residual-weighted Bayesian precision update:**
  `prec_post = prec_pre + r^2 / sigma_obs2`. The earlier "+1
  precision" update did not depend on the realized residual, which
  made the per-modality posterior identical across runs and broke
  any calibration analysis. The residual-weighted update is
  consistent with a Gaussian likelihood with known obs noise.

## Pick policy

* **Softmax over EIG with default beta=6.** Beta sweep in
  `results/beta_sweep.csv` shows beta=6 trades ~5 regret for a
  small exploration margin vs argmax (beta=20).
* **EIG approximation:** `0.5 * log1p(r^2 / posterior_var)`. The
  log1p form avoids divergence when posterior_var is small. Note
  that this is the *predicted* EIG given r, not the expected EIG
  marginalised over r; calibration between predicted and realised
  is a known limitation (see Sensor-dropout robustness section in
  RESULTS).

## Baselines

* **UCB1** with reward = realized residual squared. Sits between
  BALD and any fixed-policy baseline (regret ~176 vs 6 for BALD, vs
  425 for phase-aware).
* **Per-tick oracle** (omniscient) picks the modality with the
  largest realised EIG at each step. The "phase-aware" baseline is
  weaker than this oracle because it sees phase but not per-tick
  realisations.

## Compute envelope

* **CPU Modal, 8 seeds in parallel via `starmap`.** Each seed runs
  in seconds; the full pipeline (default + 3 sweeps) is under a
  minute end to end.
