// Operational-space controller for a 7-DOF arm.
//
// Implements the three-priority OSC structure described in
// docs/control_design.md. Stateless aside from gain configuration;
// the per-tick state (q, dq, target) flows in via Compute().

#pragma once

#include <Eigen/Dense>

namespace pvs {

struct OscGains {
  double kp = 400.0;       // primary task proportional
  double kv = 40.0;        // primary task derivative (critically damped: kv = 2 sqrt(kp))
  double kp_post = 50.0;   // secondary task (posture) proportional
  double kv_post = 14.0;   // secondary task derivative
  double k_lim = 200.0;    // tertiary task (joint-limit) gain
};

struct OscState {
  Eigen::Matrix<double, 7, 1> q;       // joint config
  Eigen::Matrix<double, 7, 1> dq;      // joint velocity
  Eigen::Matrix<double, 6, 1> x_des;   // end-effector pose target
  Eigen::Matrix<double, 6, 1> x_cur;   // end-effector pose current
  Eigen::Matrix<double, 6, 7> J;       // end-effector Jacobian at q
  Eigen::Matrix<double, 7, 7> M;       // joint-space inertia at q
  Eigen::Matrix<double, 7, 1> g;       // gravity vector at q
  Eigen::Matrix<double, 7, 1> q_nom;   // posture target
};

class OperationalSpaceController {
 public:
  explicit OperationalSpaceController(const OscGains& gains);

  // Returns the joint torque command for one tick.
  Eigen::Matrix<double, 7, 1> Compute(const OscState& state) const;

 private:
  OscGains gains_;

  // Cached intermediates exposed for unit tests.
  Eigen::Matrix<double, 6, 6> TaskInertia(const OscState& s) const;
  Eigen::Matrix<double, 7, 6> DynConsistentInverse(const OscState& s) const;
  Eigen::Matrix<double, 7, 7> NullProjector(const OscState& s) const;
};

}  // namespace pvs
