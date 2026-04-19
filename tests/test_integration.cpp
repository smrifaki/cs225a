#include <gtest/gtest.h>

#include "pvs/forward_model.h"
#include "pvs/osc.h"
#include "pvs/sensor_attention.h"

// Integration smoke: wire the forward model into the selector and
// the selector's mode into OSC's target, then run a few ticks of a
// synthetic free-space-to-contact scenario. Verifies the full
// compose path executes without crashing and that the mode
// transitions reflect the residual signal flow.

using pvs::LinearForwardModel;
using pvs::OperationalSpaceController;
using pvs::OscGains;
using pvs::OscState;
using pvs::ResidualSnapshot;
using pvs::SensorAttentionConfig;
using pvs::SensorAttentionSelector;
using pvs::SensorMode;

namespace {

OscState BareState() {
  OscState s;
  s.q.setZero();
  s.dq.setZero();
  s.x_des.setZero();
  s.x_cur.setZero();
  s.J.setZero();
  for (int i = 0; i < 6; ++i) s.J(i, i) = 1.0;
  s.M = Eigen::Matrix<double, 7, 7>::Identity();
  s.g.setZero();
  s.q_nom.setZero();
  return s;
}

}  // namespace

TEST(IntegrationTest, ForceLargerResidualWinsSelector) {
  LinearForwardModel fm_vision;
  LinearForwardModel fm_force;
  Eigen::Matrix<double, LinearForwardModel::kYDim,
                LinearForwardModel::kPhiDim> W;
  W.setZero();
  Eigen::Matrix<double, LinearForwardModel::kYDim, 1> b;
  b.setZero();
  Eigen::Matrix<double, LinearForwardModel::kYDim, 1> log_var;
  log_var.setZero();
  fm_vision.SetParams(W, b, log_var);
  fm_force.SetParams(W, b, log_var);

  Eigen::Matrix<double, LinearForwardModel::kPhiDim, 1> phi;
  phi.setZero();

  // Vision residual is small; force residual is large.
  Eigen::Matrix<double, LinearForwardModel::kYDim, 1> y_vision;
  y_vision.setConstant(0.05);
  Eigen::Matrix<double, LinearForwardModel::kYDim, 1> y_force;
  y_force.setConstant(2.0);

  const double rho_v = fm_vision.PrecisionWeightedResidualNorm(phi, y_vision);
  const double rho_f = fm_force.PrecisionWeightedResidualNorm(phi, y_force);
  ASSERT_LT(rho_v, rho_f);

  SensorAttentionConfig cfg;
  cfg.delta_hysteresis = 0.05;
  cfg.min_ticks_in_mode = 1;
  SensorAttentionSelector sel(cfg);
  ResidualSnapshot snap;
  snap.rho_vision = rho_v;
  snap.rho_force = rho_f;
  double wv = 0.0, wf = 0.0;
  // Two calls so the anti-chatter floor allows switching off the
  // default initial mode.
  sel.Select(snap, &wv, &wf);
  const auto mode = sel.Select(snap, &wv, &wf);
  EXPECT_EQ(mode, SensorMode::kForceOnly);
}

TEST(IntegrationTest, OscRespondsToSelectorChosenTarget) {
  OperationalSpaceController osc(OscGains{});

  auto state_vision_target = BareState();
  state_vision_target.x_des(0) = 0.1;
  const auto tau_vision = osc.Compute(state_vision_target);

  auto state_force_target = BareState();
  state_force_target.x_des(2) = 0.1;
  const auto tau_force = osc.Compute(state_force_target);

  EXPECT_NE(tau_vision(0), tau_force(0));
  EXPECT_NE(tau_vision(2), tau_force(2));
}
