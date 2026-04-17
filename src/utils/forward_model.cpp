#include "pvs/forward_model.h"

#include <cmath>

namespace pvs {

LinearForwardModel::LinearForwardModel() = default;

void LinearForwardModel::SetParams(
    const Eigen::Matrix<double, kYDim, kPhiDim>& W,
    const Eigen::Matrix<double, kYDim, 1>& b,
    const Eigen::Matrix<double, kYDim, 1>& log_var) {
  W_ = W;
  b_ = b;
  // Clamp log-variance to a sensible range for numerical stability;
  // mirrors the convention in the CS 224R ForwardDynamics module.
  for (int i = 0; i < kYDim; ++i) {
    log_var_(i) = std::max(-3.0, std::min(3.0, log_var(i)));
  }
}

Eigen::Matrix<double, LinearForwardModel::kYDim, 1>
LinearForwardModel::Predict(
    const Eigen::Matrix<double, kPhiDim, 1>& phi) const {
  return W_ * phi + b_;
}

double LinearForwardModel::PrecisionWeightedResidualNorm(
    const Eigen::Matrix<double, kPhiDim, 1>& phi,
    const Eigen::Matrix<double, kYDim, 1>& y) const {
  const auto y_hat = Predict(phi);
  const auto residual = y - y_hat;
  // precision = exp(-log_var / 2); precision-weighted residual is
  // element-wise product.
  Eigen::Matrix<double, kYDim, 1> weighted;
  for (int i = 0; i < kYDim; ++i) {
    weighted(i) = residual(i) * std::exp(-log_var_(i) / 2.0);
  }
  return weighted.norm();
}

}  // namespace pvs
