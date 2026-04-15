#include "pvs/osc.h"

namespace pvs {

OperationalSpaceController::OperationalSpaceController(const OscGains& gains)
    : gains_(gains) {}

Eigen::Matrix<double, 6, 6>
OperationalSpaceController::TaskInertia(const OscState& s) const {
  const auto Minv = s.M.inverse();
  return (s.J * Minv * s.J.transpose()).inverse();
}

Eigen::Matrix<double, 7, 6>
OperationalSpaceController::DynConsistentInverse(const OscState& s) const {
  const auto Minv = s.M.inverse();
  const auto Lambda = TaskInertia(s);
  return Minv * s.J.transpose() * Lambda;
}

Eigen::Matrix<double, 7, 7>
OperationalSpaceController::NullProjector(const OscState& s) const {
  const auto Jbar = DynConsistentInverse(s);
  return Eigen::Matrix<double, 7, 7>::Identity() - Jbar * s.J;
}

Eigen::Matrix<double, 7, 1>
OperationalSpaceController::Compute(const OscState& s) const {
  // Primary task: end-effector pose tracking.
  const Eigen::Matrix<double, 6, 6> Lambda = TaskInertia(s);
  const Eigen::Matrix<double, 6, 1> x_err = s.x_des - s.x_cur;
  // Task-space velocity estimate from J * dq (omit explicit dotJ * dq
  // term; SCL convention absorbs it via the bias compensation that
  // the integrator applies. See docs/control_design.md.)
  const Eigen::Matrix<double, 6, 1> v_task = s.J * s.dq;
  const Eigen::Matrix<double, 6, 1> F_ee =
      Lambda * (gains_.kp * x_err - gains_.kv * v_task);
  const Eigen::Matrix<double, 7, 1> tau_primary = s.J.transpose() * F_ee;

  // Secondary task: posture in the null space.
  const Eigen::Matrix<double, 7, 7> N = NullProjector(s);
  const Eigen::Matrix<double, 7, 1> tau_secondary =
      N * (gains_.kp_post * (s.q_nom - s.q) - gains_.kv_post * s.dq);

  // Tertiary task: joint-limit avoidance via a smoothed barrier.
  // Potential V(q) = 1/(q_limit - q) + 1/(q_limit + q) on each joint.
  // The negative gradient gives a repulsive torque
  //   tau_lim(i) = -k_lim * (1/(q_limit - q)^2 - 1/(q_limit + q)^2)
  // which vanishes by symmetry at q = 0 and blows up near either
  // limit. Each denominator is floored at eps to keep the loop
  // numerically safe if the planner ever steps the state past a
  // limit by accident.
  Eigen::Matrix<double, 7, 1> tau_lim;
  const double q_limit = 2.7;  // approx Kuka iiwa range, rad
  const double eps = 1e-3;
  for (int i = 0; i < 7; ++i) {
    const double upper = std::max(q_limit - s.q(i), eps);
    const double lower = std::max(q_limit + s.q(i), eps);
    tau_lim(i) = -gains_.k_lim *
                 (1.0 / (upper * upper) - 1.0 / (lower * lower));
  }

  // Final torque includes explicit gravity compensation.
  return tau_primary + tau_secondary + tau_lim + s.g;
}

}  // namespace pvs
