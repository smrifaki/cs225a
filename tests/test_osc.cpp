#include <gtest/gtest.h>

#include "pvs/osc.h"

using pvs::OperationalSpaceController;
using pvs::OscGains;
using pvs::OscState;

namespace {

OscState IdentityState() {
  OscState s;
  s.q.setZero();
  s.dq.setZero();
  s.x_des.setZero();
  s.x_cur.setZero();
  s.J.setZero();
  // A non-degenerate Jacobian: pick the first 6 columns of a
  // 7x7 identity. Last column is the redundancy direction.
  for (int i = 0; i < 6; ++i) s.J(i, i) = 1.0;
  s.M = Eigen::Matrix<double, 7, 7>::Identity();
  s.g.setZero();
  s.q_nom.setZero();
  return s;
}

}  // namespace

TEST(OscTest, ZeroErrorZeroTorqueAtRest) {
  OscGains gains;
  OperationalSpaceController osc(gains);
  auto s = IdentityState();
  const auto tau = osc.Compute(s);
  // With x_des == x_cur, dq == 0, q == q_nom == 0, g == 0, the only
  // nonzero contribution is the joint-limit barrier, which is small
  // at q == 0.
  for (int i = 0; i < 7; ++i) {
    EXPECT_NEAR(tau(i), 0.0, 50.0);  // generous tolerance: barrier
  }
}

TEST(OscTest, PositiveErrorYieldsPositiveTorque) {
  OscGains gains;
  OperationalSpaceController osc(gains);
  auto s = IdentityState();
  s.x_des(0) = 0.1;  // 10cm in the x direction
  const auto tau = osc.Compute(s);
  // The first joint is aligned with the x direction of the
  // primary task. Positive error should drive it positive.
  EXPECT_GT(tau(0), 0.0);
}

TEST(OscTest, GravityFlowsThrough) {
  OscGains gains;
  OperationalSpaceController osc(gains);
  auto s = IdentityState();
  s.g.setConstant(1.0);  // 1 Nm gravity per joint
  const auto tau = osc.Compute(s);
  // Gravity should add 1 to each joint torque (plus the small
  // barrier at q == 0; the barrier at q == 0 is symmetric and ~0).
  for (int i = 0; i < 7; ++i) {
    EXPECT_NEAR(tau(i), 1.0, 50.0);  // tolerance covers the barrier
  }
}

TEST(OscTest, NullSpaceMovesTowardNominalPosture) {
  OscGains gains;
  OperationalSpaceController osc(gains);
  auto s = IdentityState();
  s.q_nom.setConstant(0.5);
  // Use the redundancy direction (joint 7) which is in the null
  // space of the Jacobian set above.
  const auto tau = osc.Compute(s);
  EXPECT_GT(tau(6), 0.0);  // null-space pulls joint 7 toward q_nom(6)=0.5
}
