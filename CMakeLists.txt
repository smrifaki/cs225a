cmake_minimum_required(VERSION 3.16)
project(predictive_visual_servoing CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

add_compile_options(
  -Wall -Wextra -Wpedantic
  -Wno-unused-parameter
)

find_package(Eigen3 REQUIRED NO_MODULE)
find_package(OpenCV REQUIRED)

# SCL / OpenSai Control are installed lab-wide. The course makefile
# convention is to expose their headers and libs via env vars. We
# wrap them here so the build is portable to a fresh clone with the
# lab environment loaded.
set(SCL_INCLUDE_DIR $ENV{SCL_INCLUDE_DIR} CACHE PATH "SCL include dir")
set(SCL_LIB_DIR     $ENV{SCL_LIB_DIR}     CACHE PATH "SCL lib dir")

if(NOT SCL_INCLUDE_DIR OR NOT SCL_LIB_DIR)
  message(WARNING
    "SCL_INCLUDE_DIR / SCL_LIB_DIR not set; the simulation binary "
    "will not link. The unit-test target still builds.")
endif()

include_directories(
  ${CMAKE_SOURCE_DIR}/include
  ${EIGEN3_INCLUDE_DIR}
  ${OpenCV_INCLUDE_DIRS}
)
if(SCL_INCLUDE_DIR)
  include_directories(${SCL_INCLUDE_DIR})
  link_directories(${SCL_LIB_DIR})
endif()

set(LIB_SOURCES
  src/controllers/osc.cpp
  src/controllers/sensor_attention.cpp
  src/vision/eye_in_hand.cpp
  src/utils/forward_model.cpp
  src/utils/redis_interface.cpp
)

add_library(pvs_core STATIC ${LIB_SOURCES})
target_link_libraries(pvs_core
  PUBLIC
  Eigen3::Eigen
  ${OpenCV_LIBS}
)

if(SCL_INCLUDE_DIR)
  add_executable(run_sim src/run_sim.cpp)
  target_link_libraries(run_sim PRIVATE pvs_core scl_lib)
endif()

enable_testing()
add_subdirectory(tests)
