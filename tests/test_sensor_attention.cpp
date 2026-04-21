#include <gtest/gtest.h>

#include "pvs/sensor_attention.h"

using pvs::ResidualSnapshot;
using pvs::SensorAttentionConfig;
using pvs::SensorAttentionSelector;
using pvs::SensorMode;

TEST(SensorAttentionTest, VisionDominantOnLargeVisionResidual) {
  SensorAttentionConfig cfg;
  cfg.delta_hysteresis = 0.05;
  cfg.min_ticks_in_mode = 1;  // allow immediate switching for the test
  SensorAttentionSelector sel(cfg);
  ResidualSnapshot snap;
  snap.rho_vision = 1.0;
  snap.rho_force = 0.1;
  double wv = 0.0, wf = 0.0;
  const auto mode = sel.Select(snap, &wv, &wf);
  EXPECT_EQ(mode, SensorMode::kVisionOnly);
  EXPECT_NEAR(wv, 1.0, 1e-9);
  EXPECT_NEAR(wf, 0.0, 1e-9);
}

TEST(SensorAttentionTest, ForceDominantOnLargeForceResidual) {
  SensorAttentionConfig cfg;
  cfg.min_ticks_in_mode = 1;
  SensorAttentionSelector sel(cfg);
  ResidualSnapshot snap;
  snap.rho_vision = 0.1;
  snap.rho_force = 1.0;
  double wv = 0.0, wf = 0.0;
  // First call sets state; second call lets us cross the
  // anti-chatter floor.
  sel.Select(snap, &wv, &wf);
  const auto mode = sel.Select(snap, &wv, &wf);
  EXPECT_EQ(mode, SensorMode::kForceOnly);
  EXPECT_NEAR(wv, 0.0, 1e-9);
  EXPECT_NEAR(wf, 1.0, 1e-9);
}

TEST(SensorAttentionTest, FuseOnComparableResiduals) {
  SensorAttentionConfig cfg;
  cfg.delta_hysteresis = 0.5;  // wide margin; both within delta
  cfg.min_ticks_in_mode = 1;
  SensorAttentionSelector sel(cfg);
  ResidualSnapshot snap;
  snap.rho_vision = 0.3;
  snap.rho_force = 0.4;
  double wv = 0.0, wf = 0.0;
  const auto mode = sel.Select(snap, &wv, &wf);
  EXPECT_EQ(mode, SensorMode::kFuse);
  EXPECT_NEAR(wv + wf, 1.0, 1e-9);
  EXPECT_GT(wf, wv);  // larger residual gets larger weight
}

TEST(SensorAttentionTest, HysteresisFloorBlocksSwitching) {
  SensorAttentionConfig cfg;
  cfg.delta_hysteresis = 0.05;
  cfg.min_ticks_in_mode = 50;
  SensorAttentionSelector sel(cfg);

  // Vision dominant for 10 ticks; selector starts in vision mode.
  ResidualSnapshot vision_snap;
  vision_snap.rho_vision = 1.0;
  vision_snap.rho_force = 0.1;
  for (int i = 0; i < 10; ++i) {
    double wv, wf;
    sel.Select(vision_snap, &wv, &wf);
  }

  // Now force should dominate, but anti-chatter floor blocks the
  // switch.
  ResidualSnapshot force_snap;
  force_snap.rho_vision = 0.1;
  force_snap.rho_force = 1.0;
  double wv = 0.0, wf = 0.0;
  const auto mode = sel.Select(force_snap, &wv, &wf);
  EXPECT_EQ(mode, SensorMode::kVisionOnly)
      << "switch should be blocked by hysteresis floor";
}

TEST(SensorAttentionTest, HysteresisFloorReleasesAfterMinTicks) {
  SensorAttentionConfig cfg;
  cfg.delta_hysteresis = 0.05;
  cfg.min_ticks_in_mode = 5;
  SensorAttentionSelector sel(cfg);

  // Bring selector firmly into vision mode.
  ResidualSnapshot vision_snap;
  vision_snap.rho_vision = 1.0;
  vision_snap.rho_force = 0.0;
  double wv, wf;
  for (int i = 0; i < 5; ++i) sel.Select(vision_snap, &wv, &wf);

  // Now switch the residuals; after >= 5 ticks the selector may
  // change.
  ResidualSnapshot force_snap;
  force_snap.rho_vision = 0.0;
  force_snap.rho_force = 1.0;
  const auto mode = sel.Select(force_snap, &wv, &wf);
  EXPECT_EQ(mode, SensorMode::kForceOnly);
}
