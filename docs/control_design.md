# Controller design

Operational-space control with three priority levels and a per-tick
sensor selector for the primary task. Reference: Khatib (1987,
"A unified approach for motion and force control of robot
manipulators").

## Notation

* $q \in \mathbb{R}^7$ joint configuration of the Kuka.
* $\dot q$ joint velocities.
* $x \in \mathbb{R}^6$ end-effector pose (position $\in \mathbb{R}^3$,
  orientation as a 3-vector, the local body-fixed log-map of the
  rotation).
* $J(q) \in \mathbb{R}^{6 \times 7}$ end-effector Jacobian.
* $M(q) \in \mathbb{R}^{7 \times 7}$ joint-space inertia matrix.
* $\Lambda(q) = (J M^{-1} J^\top)^{-1}$ task-space inertia.
* $\bar J = M^{-1} J^\top \Lambda$ dynamically-consistent
  generalized inverse.
* $N = I - \bar J J$ null-space projector.
* $\tau \in \mathbb{R}^7$ joint torque command.

## Primary task: end-effector pose tracking

The primary task drives the end-effector toward the current pose
target $x^*$ provided by the sensor selector (see below). Following
the standard OSC structure:

$$
F_{\text{ee}} = \Lambda(q) \bigl[ k_p (x^* - x) - k_v \dot x \bigr] + \mu + p
$$

with $\mu = \Lambda(q) \dot J \dot q$ the centrifugal/Coriolis
compensation in task space and $p = J^{\top -1} g(q)$ the gravity
compensation. $k_p$ and $k_v$ are critically damped: $k_v = 2
\sqrt{k_p}$.

The corresponding joint torque is

$$
\tau_{\text{primary}} = J^\top F_{\text{ee}}.
$$

## Secondary task: posture in the null space

A nominal joint configuration $q_0$ (elbow-up, comfortable, away
from limits) drives the secondary task:

$$
\tau_{\text{secondary}} = N \bigl[ k_p^{\text{post}} (q_0 - q) - k_v^{\text{post}} \dot q \bigr].
$$

## Tertiary task: joint-limit avoidance

Soft repulsive potential at each joint:

$$
\tau_{\text{joint-limit}} = -k_{\text{lim}} \sum_i \nabla_q V_i(q),
$$

with $V_i$ growing rapidly near either limit of joint $i$. Implemented
via a smoothed barrier; gain tuned per-joint.

## Final torque

$$
\tau = \tau_{\text{primary}} + \tau_{\text{secondary}} + \tau_{\text{joint-limit}} + g(q),
$$

where $g(q)$ is the gravity vector. The gravity term is included
explicitly because SCL's torque interface expects the controller to
compensate.

## Sensor selector

The primary task target $x^*$ depends on which sensor channel
informs it. Two channels:

* Vision: produces an estimated hole pose $x^*_v$ from the eye-in-
  hand camera via a known calibration plus an image-space marker
  detection. Available at 240 Hz simulated.
* Force: produces a contact-driven target correction $x^*_f$ from
  the wrist force-torque reading. When in contact, $x^*_f$ regularizes
  toward the line of least resistance.

The selector chooses one of three modes per tick:

* Vision only: $x^* = x^*_v$.
* Force only: $x^* = x^*_f$.
* Fuse: $x^* = w_v x^*_v + w_f x^*_f$ with weights from the
  prediction-error signal (BALD-style; see
  `docs/sensor_attention_theory.md`).

The selector itself is implemented in
`src/controllers/sensor_attention.cpp` and the forward model that
provides the prediction-error in `src/utils/forward_model.cpp`.

## Loop rate

Control runs at 1 kHz (matching SCL's default). Camera is sampled
at 240 Hz and held between camera ticks. Force-torque at 1 kHz.

## State machine

Three regimes:

1. Free-space approach. Vision dominates; force-torque ignored
   except for safety stops.
2. Contact transition. Force-torque rises sharply; selector switches
   to fuse mode.
3. Insertion. Vision is occluded by the peg itself; force dominates.

The selector handles all three transitions without explicit hand-
coded thresholds; the prediction-error signal naturally picks the
right channel under each regime.

## Stability

Standard arguments. Critical damping on the primary task,
hierarchical null-space projection keeps secondary and tertiary
tasks from disturbing the primary. The sensor selector adds a
discrete switch; under sufficient hysteresis on the
prediction-error signal the closed loop remains stable (verified
empirically; formal analysis is out of scope for this course).

## Tuning ranges

Initial guesses (SCL standard ranges):

* $k_p = 400$, $k_v = 40$ on the primary task.
* $k_p^{\text{post}} = 50$, $k_v^{\text{post}} = 14$ on the secondary.
* $k_{\text{lim}} = 200$.
* Force-torque deadband: 0.5 N translational, 0.05 Nm rotational.

Final tuned values will be reported alongside the empirical results.

## What is NOT in the design

* Trajectory optimization upstream of the primary task. The target
  $x^*$ comes from the sensor selector each tick; there is no
  precomputed trajectory.
* Reinforcement learning of the controller gains. Gains are tuned
  by hand once.
* Impedance control with explicit virtual stiffness. We use position
  control with force-driven target updates instead.
