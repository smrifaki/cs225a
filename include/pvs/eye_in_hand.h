// Eye-in-hand camera pipeline.
//
// Consumes a simulated RGB frame from SCL, detects a single marker
// (the hole's annular fiducial) via thresholding, and returns the
// estimated hole pose in the end-effector frame. The pipeline is
// intentionally simple; learned vision is out of scope for this
// course project.

#pragma once

#include <Eigen/Dense>
#include <opencv2/core.hpp>
#include <optional>

namespace pvs {

struct EyeInHandConfig {
  double focal_length_px = 600.0;
  double principal_point_x_px = 320.0;
  double principal_point_y_px = 240.0;
  double marker_diameter_m = 0.03;
  double detect_threshold = 0.6;  // [0, 1] grayscale threshold
};

class EyeInHandCamera {
 public:
  explicit EyeInHandCamera(const EyeInHandConfig& cfg);

  // Returns the estimated hole pose in the end-effector frame, or
  // std::nullopt if detection failed (e.g., the marker is occluded).
  // Pose convention matches OscState::x_des: position in R^3 plus a
  // body-fixed log-map of the rotation.
  std::optional<Eigen::Matrix<double, 6, 1>> DetectHolePose(
      const cv::Mat& rgb) const;

 private:
  EyeInHandConfig cfg_;
};

}  // namespace pvs
