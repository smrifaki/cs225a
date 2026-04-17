// Redis interface implementation.
//
// The lab environment provides hiredis as the C client. We keep the
// implementation behind an opaque void* pointer in the header so the
// public header does not pull in hiredis transitively. The actual
// hiredis include is gated on a build flag because not every dev
// machine has it; on CI we compile with a stub.

#include "pvs/redis_interface.h"

#include <iostream>
#include <sstream>
#include <stdexcept>

#if defined(PVS_HAVE_HIREDIS)
#include <hiredis/hiredis.h>
#endif

namespace pvs {

RedisClient::RedisClient(const std::string& host, int port)
    : host_(host), port_(port) {}

RedisClient::~RedisClient() {
#if defined(PVS_HAVE_HIREDIS)
  if (impl_) {
    redisFree(static_cast<redisContext*>(impl_));
  }
#endif
}

bool RedisClient::Connect() {
#if defined(PVS_HAVE_HIREDIS)
  redisContext* c = redisConnect(host_.c_str(), port_);
  if (!c || c->err) {
    if (c) {
      std::cerr << "redis connect error: " << c->errstr << "\n";
      redisFree(c);
    }
    return false;
  }
  impl_ = c;
  return true;
#else
  (void)host_;
  (void)port_;
  std::cerr << "RedisClient::Connect() called without hiredis; "
            << "rebuild with PVS_HAVE_HIREDIS defined.\n";
  return false;
#endif
}

namespace {

std::string FormatVector(const double* data, int n) {
  std::ostringstream oss;
  oss << "[";
  for (int i = 0; i < n; ++i) {
    if (i > 0) oss << ", ";
    oss << data[i];
  }
  oss << "]";
  return oss.str();
}

bool ParseVector(const std::string& s, double* out, int n) {
  // SCL convention: comma-separated, bracketed: "[a, b, c]".
  std::size_t lb = s.find('[');
  std::size_t rb = s.find(']');
  if (lb == std::string::npos || rb == std::string::npos || rb <= lb) {
    return false;
  }
  std::string inner = s.substr(lb + 1, rb - lb - 1);
  std::istringstream iss(inner);
  for (int i = 0; i < n; ++i) {
    if (i > 0) {
      char comma;
      iss >> comma;
      if (comma != ',') return false;
    }
    if (!(iss >> out[i])) return false;
  }
  return true;
}

}  // namespace

template <int N>
bool RedisClient::ReadVector(const std::string& key,
                             Eigen::Matrix<double, N, 1>* out) {
#if defined(PVS_HAVE_HIREDIS)
  auto* c = static_cast<redisContext*>(impl_);
  if (!c) return false;
  auto* reply = static_cast<redisReply*>(redisCommand(c, "GET %s", key.c_str()));
  if (!reply || reply->type != REDIS_REPLY_STRING) {
    if (reply) freeReplyObject(reply);
    return false;
  }
  const std::string s(reply->str, reply->len);
  freeReplyObject(reply);
  return ParseVector(s, out->data(), N);
#else
  (void)key;
  (void)out;
  return false;
#endif
}

template <int N>
bool RedisClient::WriteVector(const std::string& key,
                              const Eigen::Matrix<double, N, 1>& v) {
#if defined(PVS_HAVE_HIREDIS)
  auto* c = static_cast<redisContext*>(impl_);
  if (!c) return false;
  const std::string s = FormatVector(v.data(), N);
  auto* reply = static_cast<redisReply*>(
      redisCommand(c, "SET %s %s", key.c_str(), s.c_str()));
  const bool ok = reply && reply->type == REDIS_REPLY_STATUS;
  if (reply) freeReplyObject(reply);
  return ok;
#else
  (void)key;
  (void)v;
  return false;
#endif
}

// Explicit instantiation for the shapes the controller uses.
template bool RedisClient::ReadVector<6>(const std::string&,
                                          Eigen::Matrix<double, 6, 1>*);
template bool RedisClient::ReadVector<7>(const std::string&,
                                          Eigen::Matrix<double, 7, 1>*);
template bool RedisClient::WriteVector<6>(const std::string&,
                                           const Eigen::Matrix<double, 6, 1>&);
template bool RedisClient::WriteVector<7>(const std::string&,
                                           const Eigen::Matrix<double, 7, 1>&);

}  // namespace pvs
