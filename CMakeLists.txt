cmake_minimum_required(VERSION 3.20)
project(pim_graphs CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Ramulator's yaml-cpp needs this on newer CMake (AppleClang builds).
set(CMAKE_POLICY_DEFAULT_MINIMUM 3.5)

add_subdirectory(third_party/ramulator2)

add_library(pim_sim STATIC
  src/ramulator_hbm.cpp
)
target_include_directories(pim_sim PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/src
  ${CMAKE_CURRENT_SOURCE_DIR}/third_party/ramulator2/src
  ${CMAKE_CURRENT_SOURCE_DIR}/third_party/ramulator2/ext/yaml-cpp/include
  ${CMAKE_CURRENT_SOURCE_DIR}/third_party/ramulator2/ext/spdlog/include
)
target_link_libraries(pim_sim PUBLIC ramulator)

# Tests (GoogleTest)
include(FetchContent)
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG v1.14.0
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

enable_testing()

add_executable(pim_tests tests/test_pseudo_channels.cpp)
target_link_libraries(pim_tests PRIVATE pim_sim GTest::gtest_main)
target_compile_definitions(pim_tests PRIVATE
  PIM_DEFAULT_CONFIG="${CMAKE_SOURCE_DIR}/configs/hbm_pnm.yaml"
)
# Ramulator builds as libramulator.dylib next to its CMakeLists; ensure loader finds it at test time.
set_target_properties(pim_tests PROPERTIES
  BUILD_RPATH "${CMAKE_SOURCE_DIR}/third_party/ramulator2"
)

include(GoogleTest)
gtest_discover_tests(pim_tests)

# Optional Python bindings
option(BUILD_PYBIND "Build pybind11 module pim_bindings" OFF)
if(BUILD_PYBIND)
  find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
  FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG v2.11.1
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
  )
  FetchContent_MakeAvailable(pybind11)
  pybind11_add_module(pim_bindings python/pim_bindings.cpp)
  target_link_libraries(pim_bindings PRIVATE pim_sim)
  set_target_properties(pim_bindings PROPERTIES
    BUILD_RPATH "${CMAKE_SOURCE_DIR}/third_party/ramulator2"
    INSTALL_RPATH "${CMAKE_SOURCE_DIR}/third_party/ramulator2"
  )
endif()
