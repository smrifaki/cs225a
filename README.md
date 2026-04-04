# Predictive visual servoing for contact-rich manipulation

CS 225A final project. Operational-space control on a 7-DOF Kuka arm
augmented with a prediction-error-driven sensor-attention layer that
decides at each control tick whether to attend to the eye-in-hand
camera or to the wrist force-torque sensor when refining the
end-effector pose.

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
  python/analysis/  post-run analysis, plots, calibration diagnostics
  tests/            C++ unit tests via CTest + Python tests
  docs/             theory note, design decisions, proposal, report
  scripts/          build, run, smoke
  .github/          CI workflows
```

## Software stack

* SCL / OpenSai Control (lab-provided)
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

## Reproducing the final results

See `scripts/run_all_experiments.sh` (locked to the three submission
seeds, mirrors the CS 224R repro convention).

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
