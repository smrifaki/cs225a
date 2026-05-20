# Reproduce

The C++ controller runs in the lab simulator (SCL/OpenSai). The Python
decision-layer pipeline that produces every artefact under `results/` is
fully reproducible from a clean Modal account:

```bash
uv venv .venv --python 3.11 && source .venv/bin/activate
uv pip install modal numpy matplotlib
modal token set --token-id "$MODAL_TOKEN_ID" --token-secret "$MODAL_TOKEN_SECRET"
modal run python/modal_results.py::main
```

Default: 8 seeds (0..7), horizon = 600, beta = 6. CLI overrides
documented at the top of `python/modal_results.py`. The per-tick
update rule is residual-weighted Bayesian precision growth; the BALD
agent's softmax temperature is `beta`.

Headline checks after a run:

* `results/per_phase.csv` contains four rows (approach, align, contact,
  insert) with mean and std of per-modality share across seeds.
* `results/regret_vs_oracle.csv` lists BALD, UCB1, phase-aware, and the
  three fixed-modality baselines; BALD should land near 6 +/- 2 regret
  vs the per-tick oracle.
* `results/horizon_sweep.csv`, `results/beta_sweep.csv`,
  `results/dropout_sweep.csv` give the three ablations.
