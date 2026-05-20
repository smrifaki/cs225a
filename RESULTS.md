# Results

All numbers below come from real PyBullet Modal runs. No synthetic-
decision-layer outputs in this directory.

## Real Kuka iiwa peg-in-hole

Real PyBullet rigid-body dynamics, 7-DOF Kuka iiwa, peg = cylinder
fixed-constrained to the EE link with verified world-aligned
offset, receiver = 4 walls forming a square hole with 3 mm radial
clearance.

Two control modes:

* **v2 position control** (per-joint POSITION_CONTROL). Cleanest
  insertion: vision and proprio active in approach + align, then
  force jumps from 0 → 0.67 share in contact, 1.00 share in
  insert (contact force 99 – 316 N). 40 % of ticks have real
  contact.
* **v3 closed-loop OSC** (computed-torque control with Λ task
  inertia, null-space posture term, gravity compensation). Real τ
  commanded into the joints, τ_task = 75 – 95 N·m, contacts
  during the OSC approach transient.

Per-seed numbers in [results/real_pybullet/](results/real_pybullet/).

## What's missing (for the full proposal)

* Closed-loop OSC reaching the deep-insert configuration with
  contact throughout, not just the transient.
* Multiple peg geometries (e.g. tapered, hexagonal) for the
  geometry-robustness ablation.
* Replay against the actual lab simulator (SCL/OpenSai) — the
  lab binary is not pip-installable so the canonical sim here is
  PyBullet with the standard `pybullet_data` Kuka model.

Each extension is a Modal-runnable script change, not a structural
rewrite.
