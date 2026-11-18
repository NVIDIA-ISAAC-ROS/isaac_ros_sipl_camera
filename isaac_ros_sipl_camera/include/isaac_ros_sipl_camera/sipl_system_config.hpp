// SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// SPDX-License-Identifier: Apache-2.0

#ifndef ISAAC_ROS_SIPL_CAMERA__SIPL_SYSTEM_CONFIG_HPP_
#define ISAAC_ROS_SIPL_CAMERA__SIPL_SYSTEM_CONFIG_HPP_

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "rclcpp/rclcpp.hpp"

#include "isaac_ros_sipl_camera/sipl_definitions.hpp"

namespace isaac_ros
{
namespace sipl
{
namespace config
{

using SiplEnvOverrides = std::vector<std::pair<std::string, std::string>>;

struct YuvOverrideOptions
{
  /// Overridden by NVSIPL_FORCE_YUV_CFA when set.
  std::string cfa = "uyvy";
  std::string yuv_sensor_name = "GW5";
  /// Select the driver EEPROM address: 0x51 for YUV422, 0x50 for RAW.
  bool export_link_format_env = true;
  bool export_vb1940_restore_delay_env = true;
};

bool sensorIsYuv422(const nvsipl::sensorconfig::CommonSensorConfig & sensor);

/// Restore YUV422 fields dropped by the query parser and return driver overrides.
/// Run after GetSensorSystemConfig() and before ApplyMask()/SetPlatformCfg().
/// Based on samples/camera/CQueryAdapter.cpp::OverrideYuvInputFormats().
SiplEnvOverrides overrideYuvInputFormats(
  nvsipl::sensorconfig::SensorSystemConfig & config,
  const YuvOverrideOptions & options,
  const rclcpp::Logger & logger);

void applySiplEnvOverrides(
  const SiplEnvOverrides & env, const rclcpp::Logger & logger);

/// Make module-local sensorGroup values platform-unique before SetPlatformCfg().
void uniquifySensorGroups(
  nvsipl::sensorconfig::SensorSystemConfig & config,
  const rclcpp::Logger & logger);

std::vector<uint32_t> findDuplicateSensorIds(
  const nvsipl::sensorconfig::SensorSystemConfig & config);

void preflightGroupMembership();

}  // namespace config
}  // namespace sipl
}  // namespace isaac_ros

#endif  // ISAAC_ROS_SIPL_CAMERA__SIPL_SYSTEM_CONFIG_HPP_
