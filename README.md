# Predictive visual servoing for contact-rich manipulation

CS 225A final project. Operational-space control on a 7-DOF Kuka arm
augmented with a prediction-error-driven sensor-attention layer that
decides at each control tick which of three sensors (eye-in-hand
camera, wrist force-torque, joint encoders) to attend to when
refining the end-effector pose.

**Real Kuka iiwa PyBullet sim on Modal** with closed-loop control,
4-wall receiver, verified contact firing during insertion.

| control mode      | contact-ticks / 1500 | r_force (insert) | force share (insert) |
|-------------------|---------------------:|-----------------:|---------------------:|
| position control  | 600                  |    299–316 N     |   1.00               |
| closed-loop OSC   | ~190                 |    471–902 N (approach transient) | 0.42–0.51 (approach) |

Per-seed numbers + raw traces in
[results/real_pybullet/](results/real_pybullet/).

The core claim: a robot doing a contact-rich task (peg-in-hole, soft
contact, slip-prone grasp) does not need every sensor every tick; it
needs the sensor whose prediction-error from a learned forward model
is largest right now. Cross-pollinates the adaptive-sensing
framework from the Arbabian Lab into a real manipulation setup
inside the SCL/OpenSai Control stack.

## Project shape

| Stage | Deliverable | Due |
|---|---|---|
| Proposal | one-page proposal in `docs/proposal.md` | Apr 23 |
| Outline | technical outline + planned milestones | May 12-14 |
| Progress | controller running on the sim Kuka with vision-only feedback | May 19-21 |
| Demo | full sim demo: contact-rich peg-in-hole with mixed sensor attention | Jun 9 |
| Report | `docs/final_report.md` + video | Jun 9 |

## Project structure

```
cs225a/
  src/
    controllers/    OSC + multi-modal feedback controller (C++)
    vision/         eye-in-hand camera pipeline (OpenCV)
    utils/          shared helpers (math, redis interface)
  include/          C++ headers
  configs/          YAML configs per experiment
  world/            SCL world files (Kuka, table, objects)
  python/
    real_pybullet.py      position-control PyBullet peg-in-hole
    real_pybullet_osc.py  closed-loop OSC torque control
    analysis/             post-run analysis, plots, calibration diagnostics
  tests/            C++ unit tests via CTest + Python tests
  docs/             theory note, design decisions, proposal, report
  scripts/          build, run, smoke
  .github/          CI workflows
```

## Software stack

* SCL / OpenSai Control (lab-provided, for the canonical demo)
* PyBullet 3.2.6 (Kuka iiwa from `pybullet_data`, real rigid-body
  dynamics; the canonical sim for the decision-layer experiments
  here when the lab binary is not available)
* Redis (state interface; SCL convention)
* C++17, CMake, Eigen, OpenCV, gtest
* Python 3.11 for analysis (uv-managed)
* CI: GitHub Actions (lint + unit tests on CPU)

## Quick start (simulation)

```bash
mkdir build && cd build
cmake ..
make -j
./run_sim configs/peg_in_hole.yaml
```

## Reproducing the real PyBullet peg-in-hole

```bash
# closed-loop OSC torque control (Λ task inertia, null-space posture,
# gravity compensation)
modal run python/real_pybullet_osc.py::main --seeds 0,1,2 --n-steps 1500

# position-control variant (per-joint POSITION_CONTROL with IK
# targets per phase; reaches deep insertion with sustained contact)
modal run python/real_pybullet.py::main --seeds 0,1,2 --n-steps 1500
```

Both runs use the same 4-wall receiver geometry (3 mm radial
clearance vs the peg), a per-seed lateral target offset of 12–20
mm so contact dynamics differ across seeds, and a peg-vs-arm
collision filter so the force sensor only registers peg-vs-
environment.

## Theory backing

* `docs/sensor_attention_theory.md`: BALD-style argument for
  prediction-error-driven sensor selection on top of operational-
  space control. Tight to the adaptive-sensing track in the
  Arbabian Lab.
* `docs/control_design.md`: OSC formulation with hierarchical task
  priorities (primary: end-effector pose; secondary: redundancy
  resolution; tertiary: joint-limit avoidance).

## License

MIT. See `LICENSE`.