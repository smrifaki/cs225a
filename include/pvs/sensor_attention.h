// Sensor selector for the OSC primary task.
//
// Picks per tick which sensor channel (vision, force, or fuse)
// drives the end-effector target. See
// docs/sensor_attention_theory.md for the rationale.

#pragma once

#include <Eigen/Dense>
#include <cstdint>

namespace pvs {

enum class SensorMode : std::uint8_t {
  kVisionOnly = 0,
  kForceOnly = 1,
  kFuse = 2,
};

struct SensorAttentionConfig {
  double delta_hysteresis = 0.05;  // margin between channels to switch
  int min_ticks_in_mode = 50;      // anti-chatter floor (ticks)
};

struct ResidualSnapshot {
  // Precision-weighted residual norms for each channel from the
  // previous tick.
  double rho_vision = 0.0;
  double rho_force = 0.0;
};

class SensorAttentionSelector {
 public:
  explicit SensorAttentionSelector(const SensorAttentionConfig& cfg);

  // Returns the mode for the current tick and (when in fuse mode)
  // populates the channel weights.
  SensorMode Select(const ResidualSnapshot& snap, double* w_vision, double* w_force);

  // Diagnostics for tests and logging.
  int ticks_in_current_mode() const { return ticks_in_mode_; }
  SensorMode current_mode() const { return current_mode_; }

 private:
  SensorAttentionConfig cfg_;
  SensorMode current_mode_ = SensorMode::kVisionOnly;
  int ticks_in_mode_ = 0;
};

}  // namespace pvs
