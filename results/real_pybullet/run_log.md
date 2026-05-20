# Real PyBullet + OSC peg-in-hole

Modal CPU job in `python/real_pybullet.py`.

## What is real

* Kuka iiwa 7-DOF arm from `pybullet_data`, full rigid-body
  dynamics, ~240 Hz sim step.
* Closed-loop operational-space controller written from scratch:
  task-space inertia `Lambda = (J M^-1 J^T)^-1`, joint torque
  `tau = J^T Lambda (Kp e + Kv de) + N (Kp_post q_err + Kv_post dq_err)
  + tau_g` with gravity comp from `calculateInverseDynamics`.
* Peg = cylinder fixed-constraint to the EE link (collision filter
  off between peg and arm so the contact sensor only reports peg-
  vs-environment).
* Receiver = 4 thin walls forming a square hole with ~3mm radial
  clearance vs the peg.
* 5 seeds with per-seed lateral target perturbation in [12, 20] mm.

## Current behavior

OSC tracks the offset target with steady-state error ~5-8 mm. The
peg ends roughly 7-15 mm off the hole center and contacts the wall
geometry should follow, but in the current run PyBullet reports
zero contacts during the descent and the insert phase.

## Why this is honest, not theatre

Posting the actual values rather than the synthetic ones:

| phase    | vision | force | proprio | contacts | r_force | x_err |
|----------|-------:|------:|--------:|---------:|--------:|------:|
| approach | 0.00   | 0.00  | 1.00    | 0.0      | 0.00    | 0.008 |
| align    | 0.00   | 0.00  | 1.00    | 0.0      | 0.00    | 0.005 |
| contact  | 0.00   | 0.00  | 1.00    | 0.0      | 0.00    | 0.004 |
| insert   | 0.00   | 0.00  | 1.00    | 0.0      | 0.00    | 0.005 |

Proprio dominates everywhere because the joint deviation residual
is the only modality with nonzero signal: vision summary changes
slowly, contact stays zero. The synthetic decision-layer numbers
in `results/per_phase.csv` (vision dominates approach, force
dominates insert) were a designed-in narrative. Real PyBullet
with the current geometry does not reproduce that narrative.

## Next step (not yet shipped)

Plausible fix candidates:

1. Replace the 4-wall receiver with a thin top plate + a vertical
   cylindrical bore. The plate's top edge is hard to avoid.
2. Drive the peg into the plate corner-on, not face-on, so the
   tracking error is along the contact normal.
3. Tune OSC gains so steady-state error << geometric clearance.
4. Switch to direct torque control with a velocity profile instead
   of a single position target per phase.

Logged here rather than overwriting the synthetic decision-layer
narrative; the synthetic numbers are still in
[results/per_phase.csv](../per_phase.csv).
