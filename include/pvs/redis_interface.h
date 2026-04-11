// Thin Redis interface following the SCL convention.
//
// SCL uses Redis as the state-and-command bus: the simulator writes
// q, dq, sensor readings to known keys; the controller reads those,
// computes torques, and writes them back. This header wraps the
// minimal subset we need for the project.

#pragma once

#include <Eigen/Dense>
#include <string>

namespace pvs {

struct RedisKeys {
  std::string q       = "sai2::interfaces::kuka_iiwa::sensors::q";
  std::string dq      = "sai2::interfaces::kuka_iiwa::sensors::dq";
  std::string tau     = "sai2::interfaces::kuka_iiwa::actuators::tau";
  std::string ee_pose = "sai2::interfaces::kuka_iiwa::sensors::ee_pose";
  std::string ft      = "sai2::interfaces::kuka_iiwa::sensors::ft";
  std::string camera  = "sai2::interfaces::kuka_iiwa::sensors::camera_rgb";
};

class RedisClient {
 public:
  // host = "127.0.0.1", port = 6379 are the SCL defaults.
  RedisClient(const std::string& host, int port);
  ~RedisClient();

  bool Connect();

  // Read a vector-valued key; returns false on failure.
  template <int N>
  bool ReadVector(const std::string& key, Eigen::Matrix<double, N, 1>* out);

  // Write a vector-valued key; returns false on failure.
  template <int N>
  bool WriteVector(const std::string& key,
                   const Eigen::Matrix<double, N, 1>& v);

 private:
  std::string host_;
  int port_;
  void* impl_ = nullptr;  // hiredis context
};

}  // namespace pvs
