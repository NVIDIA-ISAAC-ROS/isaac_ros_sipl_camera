# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.22.1)
project(isaac_ros_sipl_camera LANGUAGES C CXX)

find_package(ament_cmake_auto REQUIRED)
ament_auto_find_build_dependencies()

# SIPL camera package requires ARM64 Jetpack plus SIPL API.
if(NOT CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64)$")
  message(STATUS
    "Skipping build of package '${PROJECT_NAME}' because it requires ARM64 SIPL support. "
    "Detected architecture: ${CMAKE_SYSTEM_PROCESSOR}")
  return()
endif()

if(DEFINED ENV{ISAAC_ROS_PLATFORM} AND NOT "$ENV{ISAAC_ROS_PLATFORM}" STREQUAL "arm64-jetpack")
  message(STATUS
    "Skipping build of package '${PROJECT_NAME}' because it expects arm64-jetpack. "
    "ISAAC_ROS_PLATFORM is set to '$ENV{ISAAC_ROS_PLATFORM}'")
  return()
endif()

# SIPL library paths on Jetpack 7+
# Libraries are in /usr/lib/aarch64-linux-gnu/nvidia/
# Headers are in /usr/src/jetson_sipl_api/sipl/include/
set(NVIDIA_LIB_PATH "/usr/lib/aarch64-linux-gnu/nvidia")
set(SIPL_INCLUDE_PATH "/usr/src/jetson_sipl_api/sipl/include")

set(_sipl_required_libs
  nvsipl
  nvsipl_query
  nvsipl_pipeline
  nvscibuf
  nvscisync
  nvscicommon
)

# SIPL/NvSci libraries are installed with JetPack so we can expect to find them
# at /usr/lib/aarch64-linux-gnu/nvidia.
set(_sipl_missing_libs "")
foreach(_lib IN LISTS _sipl_required_libs)
  string(TOUPPER "${_lib}" _lib_upper)
  find_library(${_lib_upper}_LIB NAMES ${_lib} PATHS ${NVIDIA_LIB_PATH} NO_DEFAULT_PATH)
  if(NOT ${_lib_upper}_LIB)
    list(APPEND _sipl_missing_libs "${_lib} (${_lib_upper}_LIB)")
  endif()
endforeach()

if(_sipl_missing_libs)
  string(REPLACE ";" "\n  - " _missing_formatted "${_sipl_missing_libs}")
  message(FATAL_ERROR
    "Required SIPL/NvSci libraries not found. This package requires the full NVIDIA SIPL stack.\n"
    "Missing libraries:\n"
    "  - ${_missing_formatted}\n"
    "Expected library path: ${NVIDIA_LIB_PATH}\n"
    "Expected header path: ${SIPL_INCLUDE_PATH}\n"
    "Expected to be available on Jetpack 7+\n"
    "Verify with: ls ${NVIDIA_LIB_PATH}/lib{nvsipl,nvscibuf,nvscisync,nvscicommon}*"
    "See Jetson SIPL API installation guidance in development environment documentation")
endif()

# Sanity check these should exist to compile.
# NVIDIA SIPL headers use flat includes like #include "nvscibuf.h"
if(NOT EXISTS "${SIPL_INCLUDE_PATH}/nvsci/nvscibuf.h")
  message(FATAL_ERROR
    "NvSci headers not found at: ${SIPL_INCLUDE_PATH}/nvsci/\n"
    "See Jetson SIPL API installation guidance in development environment documentation")
endif()

if(NOT EXISTS "${SIPL_INCLUDE_PATH}/NvSIPLCamera.hpp")
  message(FATAL_ERROR
    "SIPL headers not found at: ${SIPL_INCLUDE_PATH}\n"
    "See Jetson SIPL API installation guidance in development environment documentation")
endif()

find_package(CUDAToolkit REQUIRED)
find_package(Eigen3 REQUIRED)

set(_sipl_public_includes
  ${SIPL_INCLUDE_PATH}
  ${SIPL_INCLUDE_PATH}/query/include
  ${SIPL_INCLUDE_PATH}/nvsci
)
set(_sipl_common_link_libs
  ${NVSCIBUF_LIB}
  ${NVSCISYNC_LIB}
  ${NVSCICOMMON_LIB}
  CUDA::cudart
  CUDA::cuda_driver
)

# Buffer manager library
ament_auto_add_library(sipl_buffer_manager SHARED
  src/sipl_buffer_manager.cpp
)

target_link_libraries(sipl_buffer_manager
  ${_sipl_common_link_libs}
)

# Base camera node library
ament_auto_add_library(sipl_camera_node SHARED
  src/sipl_camera_node.cpp
)

target_link_libraries(sipl_camera_node
  sipl_buffer_manager
  ${NVSIPL_LIB}
  ${NVSIPL_QUERY_LIB}
  ${NVSIPL_PIPELINE_LIB}
  ${_sipl_common_link_libs}
  magic_enum::magic_enum
)

# Stereo camera node library
ament_auto_add_library(sipl_stereo_camera_node SHARED
  src/sipl_stereo_camera_node.cpp
)

target_link_libraries(sipl_stereo_camera_node
  sipl_camera_node
)

foreach(_tgt IN ITEMS sipl_buffer_manager sipl_camera_node sipl_stereo_camera_node)
  target_include_directories(${_tgt} PUBLIC ${_sipl_public_includes})
endforeach()

foreach(_tgt IN ITEMS sipl_camera_node sipl_stereo_camera_node)
  target_include_directories(${_tgt} PUBLIC ${EIGEN3_INCLUDE_DIR})
endforeach()

# Set RPATH so NVIDIA SIPL libs and transitive GXF extension libs are
# discoverable at runtime.  The three properties below match the pattern
# used by every other NITROS-consuming package (e.g. custom_nitros_image).
foreach(_tgt IN ITEMS sipl_buffer_manager sipl_camera_node sipl_stereo_camera_node)
  set_target_properties(${_tgt} PROPERTIES
    BUILD_WITH_INSTALL_RPATH TRUE
    BUILD_RPATH_USE_ORIGIN TRUE
    INSTALL_RPATH_USE_LINK_PATH TRUE
    INSTALL_RPATH "${NVIDIA_LIB_PATH}"
  )
endforeach()


# Register components
rclcpp_components_register_node(sipl_camera_node
  PLUGIN "isaac_ros::sipl::SiplCameraNode"
  EXECUTABLE sipl_camera_node_exe
)

rclcpp_components_register_node(sipl_stereo_camera_node
  PLUGIN "isaac_ros::sipl::SiplStereoCameraNode"
  EXECUTABLE sipl_stereo_camera_node_exe
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()

  # The FindPythonInterp and FindPythonLibs modules are removed
  if(POLICY CMP0148)
    cmake_policy(SET CMP0148 OLD)
  endif()

  find_package(launch_testing_ament_cmake REQUIRED)

  # Integration tests require a SIPL-capable aarch64 target with a
  # connected Camera-over-Ethernet sensor. We only register them on aarch64 to avoid
  # build-time failures on x86 hosts that lack the SIPL stack.
  # Prepend test directory to PYTHONPATH so launch tests can import files
  # in the test directory
  set(_sipl_test_env "APPEND_ENV" "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/test")
  if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64)$")
    add_launch_test(test/sipl_camera_stereo_test.py ${_sipl_test_env})
    add_launch_test(test/sipl_camera_stereo_nv24_test.py ${_sipl_test_env})
    add_launch_test(test/sipl_camera_mono_test.py ${_sipl_test_env})
    add_launch_test(test/sipl_camera_mono_format_conversion_test.py ${_sipl_test_env})
    add_launch_test(test/sipl_camera_mono_nv24_test.py ${_sipl_test_env})
    add_launch_test(test/sipl_camera_bench_test.py ${_sipl_test_env})
    add_launch_test(test/sipl_camera_stereo_override_test.py ${_sipl_test_env})
    add_launch_test(test/sipl_camera_mono_override_test.py ${_sipl_test_env})
  endif()
endif()

ament_auto_package(USE_SCOPED_HEADER_INSTALL_DIR INSTALL_TO_SHARE launch config)

# Embed versioning information into installed files
ament_index_get_resource(ISAAC_ROS_COMMON_CMAKE_PATH isaac_ros_common_cmake_path isaac_ros_common)
include("${ISAAC_ROS_COMMON_CMAKE_PATH}/isaac_ros_common-version-info.cmake")
generate_version_info(${PROJECT_NAME})