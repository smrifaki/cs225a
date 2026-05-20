# Design decisions

Running log of the choices that shape the real Kuka iiwa peg-in-
hole sim and the headline numbers in [RESULTS.md](../RESULTS.md).

## Simulator

* **PyBullet with the standard `pybullet_data` Kuka iiwa URDF.**
  The lab's SCL/OpenSai sim is not pip-installable; PyBullet is
  the closest open alternative used by most robotics research that
  needs a real 7-DOF arm with full rigid-body dynamics. Used here
  as the canonical sim for the decision-layer + OSC experiments.
* **Peg = 3 cm-diameter, 10 cm cylinder** fixed-constrained to
  the EE link via `createConstraint(JOINT_FIXED)`. The world-
  aligned offset is computed by `multiplyTransforms` on the EE
  link's pose (the link frame is not world-aligned for this arm
  pose; v1 of the sim used the link-frame offset directly and the
  peg ended up sideways).
* **Peg vs arm collision filter set to OFF** so the contact-force
  sensor only registers peg-vs-environment, not the constraint's
  own internal contact between peg and EE link.

## Receiver

* **4 thin wall boxes around a square hole**, fixed at (0.55, 0).
  Inner half-side = peg radius + 3 mm so insertion under a small
  lateral offset always scrapes a wall.
* **Per-seed trajectory perturbation in [12, 18] mm.** The hole
  is fixed; the EE aims off-center by a seed-dependent amount and
  the peg slides along a wall during descent.

## Control modes

* **Position control (v2).** Per-joint `POSITION_CONTROL` with IK
  targets per phase. Reaches deep insertion; force dominates
  contact + insert phases (0.67 → 1.00 share).
* **Closed-loop OSC torque control (v3).** Real computed-torque
  control:
  τ = Jᵀ · Λ · (Kp · Δx − Kv · ẋ) + N · (Kp,post · Δq + Kv,post · q̇) + τ_g
  with Λ = (J · M⁻¹ · Jᵀ)⁻¹, N = I − M⁻¹ · Jᵀ · Λ · J, and τ_g
  from `calculateInverseDynamics`. Real τ commanded; force is
  active during the OSC settle transient.

## Sensors + pick policy

* **Three modalities:** vision (rendered 64×64 RGB summarised to
  a 16-D embedding), force (sum of contact-point normal forces on
  the peg link), proprio (joint-position deviation from posture).
* **BALD pick policy** with residual-weighted Bayesian precision
  update at each tick.

## What is deferred

* Multiple peg geometries (tapered, hexagonal) for the geometry-
  robustness ablation.
* OSC reaching the deep-insert configuration with sustained
  contact, not just transient.
* Replay against the actual SCL/OpenSai sim when the lab binary
  is available.
