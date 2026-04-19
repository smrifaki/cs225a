#include "pvs/eye_in_hand.h"

#include <opencv2/imgproc.hpp>

namespace pvs {

EyeInHandCamera::EyeInHandCamera(const EyeInHandConfig& cfg) : cfg_(cfg) {}

std::optional<Eigen::Matrix<double, 6, 1>>
EyeInHandCamera::DetectHolePose(const cv::Mat& rgb) const {
  if (rgb.empty()) {
    return std::nullopt;
  }
  cv::Mat gray;
  cv::cvtColor(rgb, gray, cv::COLOR_RGB2GRAY);
  cv::Mat bw;
  cv::threshold(gray, bw, cfg_.detect_threshold * 255.0, 255.0,
                cv::THRESH_BINARY_INV);
  std::vector<std::vector<cv::Point>> contours;
  cv::findContours(bw, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
  if (contours.empty()) {
    return std::nullopt;
  }
  // Pick the largest contour; assume it is the hole's fiducial.
  std::size_t best = 0;
  double best_area = 0.0;
  for (std::size_t i = 0; i < contours.size(); ++i) {
    const double a = cv::contourArea(contours[i]);
    if (a > best_area) {
      best_area = a;
      best = i;
    }
  }
  if (best_area < 30.0) {
    return std::nullopt;
  }
  cv::Moments m = cv::moments(contours[best]);
  if (m.m00 < 1e-3) {
    return std::nullopt;
  }
  const double cx = m.m10 / m.m00;
  const double cy = m.m01 / m.m00;
  // Pinhole back-projection assuming the marker plane is the table
  // surface at a known distance from the camera. Distance is
  // estimated from contour area via marker_diameter_m.
  const double area_m2 = (cfg_.marker_diameter_m / 2.0) *
                         (cfg_.marker_diameter_m / 2.0) * M_PI;
  const double area_px2 = best_area;
  const double z = cfg_.focal_length_px *
                   std::sqrt(area_m2 / std::max(area_px2, 1.0));
  const double x = (cx - cfg_.principal_point_x_px) * z / cfg_.focal_length_px;
  const double y = (cy - cfg_.principal_point_y_px) * z / cfg_.focal_length_px;

  Eigen::Matrix<double, 6, 1> pose;
  pose << x, y, z, 0.0, 0.0, 0.0;  // orientation handled separately
  return pose;
}

}  // namespace pvs
