# Design decisions

Running log of design choices and their rationale. Update as we go.

## D1. OSC over impedance control

The primary task is end-effector pose tracking with explicit
hierarchical priority levels. Impedance control with a virtual
stiffness would have been a viable alternative; we picked OSC
because the SCL framework exposes the operational-space formulation
natively and the course material centers on OSC. Trade-off: OSC
needs the dynamically consistent generalized inverse, which assumes
the Jacobian is full row-rank. Near singularities we use a damped
pseudo-inverse fallback (planned, not yet implemented).

## D2. Two sensor channels, not three

Vision and force-torque only. Tactile arrays and joint-torque
encoders are excluded for scope. The selector logic generalizes to
N channels with no architectural change; the demo just uses two.

## D3. Linear forward model, not a neural network

Each channel's forward model is a single linear layer
$y = W \phi + b$ with a per-coordinate log-variance head. Rationale:
the relevant nonlinearity (contact onset) is captured by the
prediction-error magnitude rather than by the predicted mean, and a
linear model on $(q, \dot q, \tau_{\text{last}})$ is sufficient to
get a calibrated baseline for free-space motion. If the empirical
calibration test fails we will revisit with a small MLP.

## D4. Hysteresis floor on the selector

The selector cannot leave its current mode until it has spent at
least `min_ticks_in_mode` ticks in it (default 50 ticks at 1 kHz,
i.e., 50 ms). This is the anti-chatter measure standard in
switching control. Empirical tuning will follow.

## D5. Greedy K-patch oracle analog: not applicable here

In CS 224R the oracle was a clairvoyant patch selector. Here the
analogous oracle would be a clairvoyant sensor selector that knows
the upcoming contact transition in advance. We do not implement
that oracle because the relevant comparison is against the
always-fuse baseline (B2), which uses both sensors every tick and
therefore upper-bounds any selector's success rate at the cost of
maximal sensor budget. We compare to B2 directly.

## D6. Locked experimental setup

Three seeds (42, 1337, 2024), four modes (B0, B1, B2, proposed),
one task (peg-in-hole), one workspace distribution. See
`docs/proposal.md` for the hypothesis statement.

## D7. Simulation only

The deliverable runs in SCL simulation. No real-robot deployment.
The course rubric allows simulation-only projects and the
contact-rich dynamics in SCL are sufficient to test the
hypotheses.

## D8. Vision pipeline is a single thresholded contour

Not a learned detector. The hole's annular fiducial is large and
well-lit in simulation; a simple threshold + largest-contour
heuristic recovers the centroid reliably. A learned detector adds
complexity that is not on the critical path for the selector
ablation.

## D9. C++17, not 20

The lab toolchain pins at C++17. We follow.

## D10. Eigen with compile-time shapes

`Eigen::Matrix<double, 7, 1>` not `Eigen::VectorXd`. Compile-time
shapes catch errors at build time and let the compiler unroll the
hot loop. The price is more verbose templates, which is worth it
for a real-time controller.
