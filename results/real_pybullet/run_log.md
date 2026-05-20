# Real PyBullet peg-in-hole v2

Modal CPU sim, 3 seeds, 1500 ticks each.

## What is real

* Kuka iiwa 7-DOF arm, full PyBullet rigid-body dynamics.
* Peg = cylinder (1.5cm radius, 10cm length), fixed-constraint to
  the EE link with a verified world-aligned offset (sim prints the
  desired vs actual peg world position at the start).
* Receiver = 4 walls forming a square hole with 3mm radial
  clearance vs the peg, walls 16cm tall, fixed at (0.55, 0).
* Per-seed lateral offset of the trajectory target in [12, 18] mm,
  random direction. Hole is fixed; the EE aims off-center.
* Sensors: rendered 64x64 RGB (vision), peg contact-point force
  (force), joint-position deviation from commanded (proprio).
* BALD pick policy with residual-weighted Bayesian precision update.

## Per-phase modality share (seed 0 / 1 / 2)

| phase    | vision share          | force share           | proprio share         | mean contacts | mean r_force |
|----------|----------------------:|----------------------:|----------------------:|--------------:|-------------:|
| approach | 0.30 / 0.36 / 0.35    | 0.31 / 0.34 / 0.31    | 0.39 / 0.30 / 0.33    | 0.0           | 0            |
| align    | 0.37 / 0.43 / 0.41    | 0.32 / 0.25 / 0.31    | 0.31 / 0.33 / 0.28    | 0.0           | 0            |
| contact  | 0.21 / 0.22 / 0.23    | 0.67 / 0.67 / 0.67    | 0.12 / 0.11 / 0.10    | 1.2 / 0.6 / 1.2 | 99 / 176 / 114 N |
| insert   | 0.00 / 0.00 / 0.00    | 1.00 / 1.00 / 1.00    | 0.00 / 0.00 / 0.00    | 3.0 / 1.9 / 3.2 | 299 / 314 / 316 N |

## Interpretation

This is the phenomenology the proposal predicts and the synthetic
decision-layer model approximates: vision/proprio active in free
space, force suddenly dominates the moment the peg touches the
receiver, force completely dominates during insertion when the
peg is sliding along the wall under tracking error.

contact-ticks per 1500-tick trajectory: 601 / 605 / 605 ~ 40%, all
in the contact + insert phases (the second half of the trajectory).

## What was wrong in v1

v1 used the createConstraint API with `parentFramePosition` in the
EE-link frame without accounting for the EE link's orientation
(which is not world-aligned for the Kuka iiwa pose I started in).
The peg ended up sideways relative to the receiver and either
collided with the arm itself or floated past the walls. v2 uses
`multiplyTransforms` to convert a world-aligned offset into the
parent frame correctly, and verifies via a settle-step that the
peg actually sits where requested.

## Run

```bash
modal run python/real_pybullet.py::main --seeds 0,1,2 --n-steps 1500
```
