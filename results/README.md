# results/ directory map

Every artefact under `results/` is produced by a single `modal run`
of `python/modal_results.py::main`. The full reproduction recipe is
in [REPRODUCE.md](REPRODUCE.md).

```
results/
  traces.csv                   per-tick row (seed, step, phase,
                               per-modality residual + EIG, pick,
                               posterior variances)
  per_phase.csv                per-phase summary across seeds:
                               steps, true sigmas, modality share
                               (mean + std), mean EIG per modality
  regret_vs_oracle.csv         BALD vs UCB1, vision-only, force-only,
                               proprio-only, random, phase-aware
  oracle_match.csv             fraction of ticks where BALD picked the
                               modality with the largest true sigma in
                               the current phase
  beta_sweep.csv               inverse-temperature sweep, regret vs
                               oracle at each beta in {0.5..20}
  horizon_sweep.csv            proportional-phase horizon sweep at
                               beta=6, horizons {200..1200}
  dropout_sweep.csv            per-modality sensor-dropout sweep at
                               horizon=600, dropout in {0..0.4}
  figures/
    sensor_attention.{pdf,png}   per-modality share over time
    modality_share_per_phase.{pdf,png}
    posterior_precision.{pdf,png}
    eig_per_phase.{pdf,png}
    eig_vs_residual.{pdf,png}
    beta_sweep.{pdf,png}
    horizon_sweep.{pdf,png}
    dropout_sweep.{pdf,png}
    composite.{pdf,png}        three-panel summary
```

Phase ground truth lives in `python/modal_results.py::simulate`
(sigmas per phase, default horizon = 600).
