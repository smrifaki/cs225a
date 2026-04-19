#include <gtest/gtest.h>

#include "pvs/forward_model.h"

using pvs::LinearForwardModel;

TEST(ForwardModelTest, ZeroParamsPredictZero) {
  LinearForwardModel m;
  Eigen::Matrix<double, LinearForwardModel::kPhiDim, 1> phi;
  phi.setOnes();
  const auto y = m.Predict(phi);
  for (int i = 0; i < LinearForwardModel::kYDim; ++i) {
    EXPECT_NEAR(y(i), 0.0, 1e-9);
  }
}

TEST(ForwardModelTest, IdentityWeightsPropagatePhi) {
  LinearForwardModel m;
  Eigen::Matrix<double, LinearForwardModel::kYDim,
                LinearForwardModel::kPhiDim>
      W = Eigen::Matrix<double, LinearForwardModel::kYDim,
                        LinearForwardModel::kPhiDim>::Zero();
  for (int i = 0; i < LinearForwardModel::kYDim; ++i) {
    W(i, i) = 1.0;
  }
  Eigen::Matrix<double, LinearForwardModel::kYDim, 1> b;
  b.setZero();
  Eigen::Matrix<double, LinearForwardModel::kYDim, 1> log_var;
  log_var.setZero();
  m.SetParams(W, b, log_var);

  Eigen::Matrix<double, LinearForwardModel::kPhiDim, 1> phi;
  for (int i = 0; i < LinearForwardModel::kPhiDim; ++i) phi(i) = i;
  const auto y = m.Predict(phi);
  for (int i = 0; i < LinearForwardModel::kYDim; ++i) {
    EXPECT_NEAR(y(i), static_cast<double>(i), 1e-9);
  }
}

TEST(ForwardModelTest, PrecisionWeightedResidualNorm) {
  LinearForwardModel m;
  Eigen::Matrix<double, LinearForwardModel::kYDim,
                LinearForwardModel::kPhiDim>
      W = Eigen::Matrix<double, LinearForwardModel::kYDim,
                        LinearForwardModel::kPhiDim>::Zero();
  Eigen::Matrix<double, LinearForwardModel::kYDim, 1> b;
  b.setZero();
  Eigen::Matrix<double, LinearForwardModel::kYDim, 1> log_var;
  log_var.setZero();  // unit precision
  m.SetParams(W, b, log_var);

  Eigen::Matrix<double, LinearForwardModel::kPhiDim, 1> phi;
  phi.setZero();
  Eigen::Matrix<double, LinearForwardModel::kYDim, 1> y;
  y.setConstant(1.0);
  const double norm = m.PrecisionWeightedResidualNorm(phi, y);
  // ||(1,1,1,1,1,1) * exp(0)||_2 = sqrt(6)
  EXPECT_NEAR(norm, std::sqrt(6.0), 1e-9);
}

TEST(ForwardModelTest, HighPrecisionDeflatesResidual) {
  LinearForwardModel m;
  Eigen::Matrix<double, LinearForwardModel::kYDim,
                LinearForwardModel::kPhiDim>
      W = Eigen::Matrix<double, LinearForwardModel::kYDim,
                        LinearForwardModel::kPhiDim>::Zero();
  Eigen::Matrix<double, LinearForwardModel::kYDim, 1> b;
  b.setZero();
  Eigen::Matrix<double, LinearForwardModel::kYDim, 1> log_var;
  log_var.setConstant(-2.0);  // high precision per coord (large 1/sigma)
  m.SetParams(W, b, log_var);

  Eigen::Matrix<double, LinearForwardModel::kPhiDim, 1> phi;
  phi.setZero();
  Eigen::Matrix<double, LinearForwardModel::kYDim, 1> y;
  y.setConstant(1.0);
  const double norm = m.PrecisionWeightedResidualNorm(phi, y);
  // precision factor exp(-(-2)/2) = e, so weighted residual is
  // ||(e, e, e, e, e, e)||_2 = e * sqrt(6)
  EXPECT_NEAR(norm, std::exp(1.0) * std::sqrt(6.0), 1e-9);
}
