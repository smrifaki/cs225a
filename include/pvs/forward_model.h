// Linear forward model on (joint state, torque) -> sensor reading.
//
// Each channel (vision pose target, wrist force-torque) gets its own
// linear model with a residual log-variance. Trained offline from
// random-policy rollouts, frozen at run time. Provides the
// precision-weighted residual the selector consumes.

#pragma once

#include <Eigen/Dense>
#include <array>
#include <cstddef>

namespace pvs {

// Generic linear forward model:
//   y_hat = W * phi + b
// with phi = [q; dq; tau_last], y the 6-D channel reading (vision
// pose error or wrist wrench). The model also carries a per-coord
// log-variance estimate that drives the precision weighting.
class LinearForwardModel {
 public:
  static constexpr std::size_t kPhiDim = 7 + 7 + 7;  // q + dq + tau_last
  static constexpr std::size_t kYDim   = 6;

  LinearForwardModel();

  // Set the parameters; typically loaded from a trained checkpoint.
  void SetParams(const Eigen::Matrix<double, kYDim, kPhiDim>& W,
                 const Eigen::Matrix<double, kYDim, 1>& b,
                 const Eigen::Matrix<double, kYDim, 1>& log_var);

  // Predict the channel reading from state.
  Eigen::Matrix<double, kYDim, 1> Predict(
      const Eigen::Matrix<double, kPhiDim, 1>& phi) const;

  // Precision-weighted residual norm given observed y.
  double PrecisionWeightedResidualNorm(
      const Eigen::Matrix<double, kPhiDim, 1>& phi,
      const Eigen::Matrix<double, kYDim, 1>& y) const;

 private:
  Eigen::Matrix<double, kYDim, kPhiDim> W_ =
      Eigen::Matrix<double, kYDim, kPhiDim>::Zero();
  Eigen::Matrix<double, kYDim, 1> b_ = Eigen::Matrix<double, kYDim, 1>::Zero();
  Eigen::Matrix<double, kYDim, 1> log_var_ =
      Eigen::Matrix<double, kYDim, 1>::Zero();
};

}  // namespace pvs
