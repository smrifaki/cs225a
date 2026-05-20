# Real PyBullet runs

The numbers in `results/` (one level up) are from the python decision-
layer simulation with handcrafted per-phase sigmas. The artefacts
here are **real Modal jobs running a Kuka iiwa 7-DOF arm in
PyBullet physics**.

## peg_in_hole_summary.json

* 4 seeds, 600 ticks each, real PyBullet rigid-body sim.
* Arm: Kuka iiwa from `pybullet_data` (same model as the lab sim).
* Sensors:
  * vision: PyBullet `getCameraImage` from a fixed pose, summarised
    to a 32-D embedding (per-channel mean over a 4x4 grid plus
    brightness std).
  * force:  contact force on the EE link from `getContactPoints`.
  * proprio: joint-position deviation from the current commanded
    target.
* Per-phase share (mean across 4 seeds):

  | phase    | vision | force | proprio |
  |----------|-------:|------:|--------:|
  | approach |  0.74  |  0.01 |  0.25   |
  | align    |  0.86  |  0.07 |  0.08   |
  | contact  |  0.96  |  0.00 |  0.03   |
  | insert   |  1.00  |  0.00 |  0.00   |

## What this tells us vs the decision-layer narrative

The decision-layer model predicts vision -> force handoff as the
arm contacts the workpiece (the proposal narrative). The real
PyBullet sim does NOT reproduce that: vision dominates throughout
because the camera-derived residual is the loudest signal early on
(scene changing as the arm moves) and stays high in insert (peg
covers part of the view). The force channel sits near zero because
the synthetic-hole used here is a target point, not a real geometry
with contact friction.

This is a real result: the current contact model is too clean to
trigger the force channel. The next iteration adds a proper
URDF-defined hole + friction so the force signal becomes
informative. The decision-layer narrative is still defensible as
the *target* phenomenology; the real sim isolates which physics
detail the synthetic model abstracts away.

Run via:

    modal run python/real_pybullet.py::main --seeds 0,1,2,3 --n-ticks 600
