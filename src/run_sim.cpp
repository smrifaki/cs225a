// Driver that wires the OSC controller, sensor selector, vision
// pipeline, and Redis interface together against the SCL simulator.
//
// Loop:
//   1. Read q, dq, ee_pose, ft, camera_rgb from Redis.
//   2. Run vision detection if a new camera frame arrived.
//   3. Compute forward-model predictions and residual norms for the
//      vision and force channels using the previous tick's targets.
//   4. Ask the sensor selector for the current mode and channel
//      weights.
//   5. Fuse the target x_des from the active channel(s).
//   6. Run OSC, get joint torques.
//   7. Write tau back to Redis.
//
// Build only when SCL is detected; the test target builds without
// it.

#include <Eigen/Dense>
#include <chrono>
#include <iostream>
#include <thread>

#include "pvs/eye_in_hand.h"
#include "pvs/forward_model.h"
#include "pvs/osc.h"
#include "pvs/redis_interface.h"
#include "pvs/sensor_attention.h"

namespace {

Eigen::Matrix<double, 21, 1> AssemblePhi(
    const Eigen::Matrix<double, 7, 1>& q,
    const Eigen::Matrix<double, 7, 1>& dq,
    const Eigen::Matrix<double, 7, 1>& tau_last) {
  Eigen::Matrix<double, 21, 1> phi;
  phi.head<7>() = q;
  phi.segment<7>(7) = dq;
  phi.tail<7>() = tau_last;
  return phi;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "usage: run_sim <config.yaml>\n";
    return 1;
  }
  const std::string config_path = argv[1];
  std::cout << "loading config from " << config_path << "\n";

  // Config loading (yaml-cpp wrapper omitted for brevity in this
  // skeleton; the YAML keys above map 1:1 to the struct fields).
  pvs::OscGains osc_gains;
  pvs::SensorAttentionConfig sel_cfg;
  pvs::EyeInHandConfig eye_cfg;

  pvs::OperationalSpaceController osc(osc_gains);
  pvs::SensorAttentionSelector selector(sel_cfg);
  pvs::EyeInHandCamera camera(eye_cfg);
  pvs::LinearForwardModel fm_vision;
  pvs::LinearForwardModel fm_force;
  // Forward-model weights are loaded from disk in a real run.

  pvs::RedisKeys keys;
  pvs::RedisClient redis("127.0.0.1", 6379);
  if (!redis.Connect()) {
    std::cerr << "redis connect failed; aborting\n";
    return 2;
  }

  Eigen::Matrix<double, 7, 1> tau_last;
  tau_last.setZero();

  const double dt = 0.001;
  for (std::size_t tick = 0; ; ++tick) {
    Eigen::Matrix<double, 7, 1> q, dq;
    Eigen::Matrix<double, 6, 1> ee_pose, ft;
    if (!redis.ReadVector(keys.q, &q) ||
        !redis.ReadVector(keys.dq, &dq) ||
        !redis.ReadVector(keys.ee_pose, &ee_pose) ||
        !redis.ReadVector(keys.ft, &ft)) {
      std::cerr << "redis read failed at tick " << tick << "\n";
      break;
    }

    const auto phi = AssemblePhi(q, dq, tau_last);

    // Vision channel: a 6-DOF "pose target correction" measured by
    // the eye-in-hand pipeline. In the actual loop the camera frame
    // would be deserialized from Redis; we elide that here.
    Eigen::Matrix<double, 6, 1> vision_target;
    vision_target.setZero();

    // Force channel: the wrist 6-DOF wrench treated as the
    // observation y.
    const Eigen::Matrix<double, 6, 1> y_force = ft;

    const double rho_vision =
        fm_vision.PrecisionWeightedResidualNorm(phi, vision_target);
    const double rho_force =
        fm_force.PrecisionWeightedResidualNorm(phi, y_force);

    pvs::ResidualSnapshot snap;
    snap.rho_vision = rho_vision;
    snap.rho_force = rho_force;
    double wv = 0.5, wf = 0.5;
    const auto mode = selector.Select(snap, &wv, &wf);
    (void)mode;

    pvs::OscState state;
    state.q = q;
    state.dq = dq;
    state.x_cur = ee_pose;
    // x_des is the convex combination of vision and force targets,
    // each transformed into the OSC primary task pose space.
    state.x_des = wv * vision_target + wf * y_force;
    state.J.setZero();   // populated from SCL kinematics in real run
    state.M.setIdentity();
    state.g.setZero();
    state.q_nom.setZero();

    const auto tau = osc.Compute(state);
    redis.WriteVector(keys.tau, tau);
    tau_last = tau;

    std::this_thread::sleep_for(std::chrono::duration<double>(dt));
  }
  return 0;
}
