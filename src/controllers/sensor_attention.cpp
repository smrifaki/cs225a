#include "pvs/sensor_attention.h"

#include <algorithm>

namespace pvs {

SensorAttentionSelector::SensorAttentionSelector(const SensorAttentionConfig& cfg)
    : cfg_(cfg) {}

SensorMode SensorAttentionSelector::Select(
    const ResidualSnapshot& snap, double* w_vision, double* w_force) {
  // Hysteresis floor: do not switch out of the current mode until
  // we have spent at least min_ticks_in_mode in it.
  ticks_in_mode_++;
  const bool may_switch = ticks_in_mode_ >= cfg_.min_ticks_in_mode;

  const double v = snap.rho_vision;
  const double f = snap.rho_force;
  const double delta = cfg_.delta_hysteresis;

  SensorMode proposed;
  if (v > f + delta) {
    proposed = SensorMode::kVisionOnly;
  } else if (f > v + delta) {
    proposed = SensorMode::kForceOnly;
  } else {
    proposed = SensorMode::kFuse;
  }

  if (proposed != current_mode_) {
    if (may_switch) {
      current_mode_ = proposed;
      ticks_in_mode_ = 0;
    }
  }

  // Fuse weights when in fuse mode; else degenerate to one-hot.
  const double sum = std::max(v + f, 1e-9);
  switch (current_mode_) {
    case SensorMode::kVisionOnly:
      if (w_vision) *w_vision = 1.0;
      if (w_force)  *w_force  = 0.0;
      break;
    case SensorMode::kForceOnly:
      if (w_vision) *w_vision = 0.0;
      if (w_force)  *w_force  = 1.0;
      break;
    case SensorMode::kFuse:
      if (w_vision) *w_vision = v / sum;
      if (w_force)  *w_force  = f / sum;
      break;
  }
  return current_mode_;
}

}  // namespace pvs
