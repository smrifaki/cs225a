# Project proposal

CS 225A, spring 2026.
Author: Mouhssine Rifaki.

## Title

Predictive visual servoing for contact-rich manipulation: an
operational-space controller with prediction-error-driven sensor
attention.

## Motivation

Modern manipulation pipelines tend to fuse every available sensor at
every control tick: camera, force-torque, joint encoders, sometimes
tactile arrays. This is a deliberate over-allocation: most of the
time the policy can recover the same control from a subset of
sensors, and the additional sensor reads spend bandwidth and compute
that an embedded robot can ill afford.

The proposal asks a small, falsifiable version of this question.
Given a 7-DOF Kuka doing a contact-rich peg-in-hole task in
simulation, can a prediction-error signal from a learned forward
model drive online attention between two sensor channels (eye-in-
hand camera vs wrist force-torque) without degrading the success
rate of the controller? The hope is a controller that uses each
sensor only when needed and matches the always-on baseline on
success rate.

This builds on the Arbabian Lab adaptive-sensing track and ports
its central feature (prediction error as a routing signal) into the
SCL/OpenSai Control stack the course operates in.

## Concrete task

Peg-in-hole insertion with a 6mm clearance, peg held by a parallel-
jaw gripper, hole positioned within a 30cm x 30cm workspace.
Reference object pose is drawn from a small distribution at the
start of each episode. The controller has 30 seconds of wall-clock
to complete the insertion. Success metric is a binary indicator
(peg fully inserted within 30s) plus a continuous metric (final
end-effector pose error from the hole center).

## Two-channel sensor setup

* Channel A: eye-in-hand RGB-D camera (240 Hz, used for visual
  servoing toward the hole).
* Channel B: wrist force-torque sensor (1 kHz, used for contact-
  driven corrections).

Both channels are simulated through SCL.

## Controller architecture

Operational-space control with three priority levels:

1. Primary task: end-effector pose tracking toward the target,
   computed from one or both sensor channels per tick.
2. Secondary task: redundancy resolution toward a comfortable
   nominal joint configuration.
3. Tertiary task: joint-limit avoidance.

The novel piece is the per-tick selector that decides which
sensor channel (or both) feeds the primary task. Three baselines
plus the proposed selector:

* B0: vision-only. Camera drives the primary task. Force ignored.
* B1: force-only. Wrist sensor drives, camera ignored.
* B2: always-fuse. Vision and force are fused at every tick via
  fixed weights.
* Proposed: prediction-error-driven. A small forward model predicts
  the next sensor reading per channel; the channel with the larger
  precision-weighted residual is the one given primary-task weight
  that tick. (BALD-style intuition; see
  `docs/sensor_attention_theory.md`.)

## Hypotheses

H1. The proposed selector matches B2 (always-fuse) on success rate
    while using strictly less per-tick compute. Operationalized as:
    success rate within 2% of B2, integrated sensor-read count at
    least 30% lower than B2.

H2. The proposed selector beats B0 and B1 on success rate. Each
    of B0 and B1 is expected to fail on a known subset of the
    workspace (vision-only fails under occlusion; force-only fails
    when the peg is not yet in contact).

H3. The prediction-error signal is calibrated under the working
    distribution of object poses, i.e., ECE on the forward model's
    log-variance head is below 0.05.

## Timeline

Weeks 1-2: SCL world setup, Kuka URDF, OSC primary task on the bare
end-effector with no sensors-as-input (proprioception only).

Week 3: add vision channel; train forward model on random rollouts
in the workspace.

Week 4: add force channel + selector; calibration diagnostics.

Week 5: contact-rich peg-in-hole; baseline B0, B1, B2 numbers.

Weeks 6-7: proposed selector vs baselines; calibration; ablations.

Weeks 8-9: report, video, polish.

## Risk

Risk 1: SCL contact dynamics may be noisier than expected and the
force channel could dominate the prediction error in ways the
forward model cannot predict. Mitigation: pretrain the forward
model on random rollouts that include contact, so the
calibrated distribution covers contact noise.

Risk 2: redundancy resolution may interact badly with the selector,
because the secondary task is computed from joint angles only and
does not depend on the selected channel. Should be fine in practice
because OSC priority levels are decoupled, but worth verifying
empirically.

Risk 3: vision-only baseline B0 might be too strong. Mitigation:
seed-controlled occlusion is part of the workspace distribution so
B0 has predictable failure modes; this is the test for H2.

## Out of scope

* Real-robot deployment. All experiments stay in simulation.
* End-to-end learning of the controller. The OSC structure is
  fixed; only the selector and the forward model are learned.
* Multi-object manipulation. Single peg, single hole.
