# Sensor attention via prediction error

Applied-register note explaining why the selector picks the sensor
with the larger precision-weighted residual. Mirrors the
`sufficient_statistic_claim.md` argument from the CS 224R adaptive-
sensing track, adapted to a control setting.

## Setting

At each control tick $t$ the robot has two candidate sensor
channels: vision $V$ and force-torque $F$. A small forward model
$f_\theta$ predicts what the next reading of each channel will be,
given the current joint state and the last commanded torque. The
prediction is Gaussian with mean $\mu_c$ and per-coordinate
log-variance $\log \sigma_c^2$ for channel $c \in \{V, F\}$.

After the channel is read at the end of the tick, the realized
residual is
$$
r_c = y_c - \mu_c.
$$
The precision-weighted residual norm is
$$
\rho_c = \lVert r_c \cdot \exp(-\log\sigma_c^2 / 2) \rVert_2.
$$

## Why use prediction error for selection

In a stationary regime (free-space approach with a known scene)
the vision channel is well-predictable and $\rho_V$ is small. In a
contact transition the force-torque channel becomes informative;
$\rho_F$ rises sharply because the forward model cannot predict the
contact event from prior state alone.

The intuition: the channel with the larger precision-weighted
residual is the channel that just supplied new information the model
did not already have. That is exactly the BALD criterion specialized
to a control loop (Houlsby et al., 2011, generalized in Gal et al.
2017): pick the observation whose realized residual is most
informative under the current posterior.

## Selector rule

For each tick the selector computes $\rho_V$ and $\rho_F$ from the
previous tick's realized residuals. The current tick's mode is:

$$
\text{mode}_t =
\begin{cases}
\text{vision only} & \rho_V > \rho_F + \delta \\
\text{force only}  & \rho_F > \rho_V + \delta \\
\text{fuse}        & \text{otherwise}
\end{cases}
$$

with $\delta$ a hysteresis margin to avoid oscillation. Fusion
weights when in fuse mode are

$$
w_c = \frac{\rho_c}{\rho_V + \rho_F},
$$

normalized so $w_V + w_F = 1$.

## Why this can match always-fuse

When both channels are well-calibrated the prediction-error signal
identifies which channel is currently providing new information. In
free space, vision provides almost all the new information per tick;
in contact, force provides it. Always-fuse pays for both channels
every tick; the selector pays for only one when the other is
predictable.

The risk is that the forward model itself is miscalibrated, in which
case the selector's signal is noise. We test this by ECE diagnostics
on the forward model (`python/analysis/calibration.py`) and require
ECE below 0.05 before the selector is trusted (see H3 in
`proposal.md`).

## How this differs from the CS 224R foveation feature

The foveation work in CS 224R used the same prediction-error idea
to pick which patch of a static image to commit at high resolution.
Here the choice is over sensor CHANNELS at each control tick of a
1 kHz loop. Both are instances of BALD-style information-gain
selection; the loop rate and the action set differ.

## Failure modes anticipated

F1. Aleatoric noise dominance. If one channel has high irreducible
    noise (force-torque under stick-slip vibration), $\rho_F$ is
    chronically high and the selector under-uses vision. Mitigation
    (planned): a Laplace-style epistemic-only variant of the forward
    model that strips aleatoric noise from the prediction-error
    signal. Out of scope for the first deliverable; reported in
    `failure_modes.md` once we hit it.

F2. Selector oscillation. Without sufficient hysteresis the selector
    chatters at the contact transition. Mitigation: empirical $\delta$
    tuning, with hysteresis margin reported alongside.

F3. Forward-model staleness. If the model is trained on free-space
    random rollouts but evaluated under contact, its log-variance
    head is uncalibrated under contact (analogous to the
    ImageNet-C calibration breakdown in CS 224R). Mitigation:
    include contact in pretraining rollouts.

## Empirical correlates that should hold

1. Forward-model ECE on a held-out set is below 0.05 on free-space
   and below 0.10 in the contact regime.
2. The selector chooses force in at least 80% of contact ticks.
3. The selector chooses vision in at least 80% of free-space ticks.
4. Success rate on peg-in-hole matches always-fuse to within 2%.
5. Per-tick sensor-read count is at least 30% lower than always-
   fuse.

## What this is NOT

* A theorem. The applied-register note explains the intuition. The
  empirical demonstration is the deliverable.
* A claim about real-robot deployment. All experiments stay in SCL
  simulation.
* A claim about superiority over learned end-to-end controllers.
  The contribution is the principled selector, not the OSC backbone.
