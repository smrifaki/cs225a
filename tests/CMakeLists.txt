find_package(GTest REQUIRED)
include(GoogleTest)

add_executable(pvs_tests
  test_osc.cpp
  test_sensor_attention.cpp
  test_forward_model.cpp
  test_integration.cpp
)
target_link_libraries(pvs_tests
  PRIVATE
  pvs_core
  GTest::gtest
  GTest::gtest_main
)
gtest_discover_tests(pvs_tests DISCOVERY_TIMEOUT 60)
