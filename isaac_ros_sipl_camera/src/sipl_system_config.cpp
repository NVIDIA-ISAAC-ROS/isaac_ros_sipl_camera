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

#include "isaac_ros_sipl_camera/sipl_system_config.hpp"

#include <fcntl.h>
#include <glob.h>
#include <grp.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <map>
#include <set>
#include <stdexcept>
#include <variant>

namespace isaac_ros
{
namespace sipl
{
namespace config
{

namespace
{

constexpr uint16_t kEepromAliasBase = 0x30U;

const std::map<std::string, uint32_t> & yuvPixelOrders()
{
  static const std::map<std::string, uint32_t> kOrders = {
    {"yuv", NVSIPL_PIXEL_ORDER_YUV},
    {"yvu", NVSIPL_PIXEL_ORDER_YVU},
    {"yuyv", NVSIPL_PIXEL_ORDER_YUYV},
    {"yvyu", NVSIPL_PIXEL_ORDER_YVYU},
    {"vyuy", NVSIPL_PIXEL_ORDER_VYUY},
    {"uyvy", NVSIPL_PIXEL_ORDER_UYVY},
  };
  return kOrders;
}

template<typename Fn>
void forEachSensorConst(
  const nvsipl::sensorconfig::SensorSystemConfig & config, Fn && fn)
{
  for (const auto & module : config.modules) {
    if (std::holds_alternative<nvsipl::sensorconfig::GmslModule>(module.moduleType)) {
      const auto & gm = std::get<nvsipl::sensorconfig::GmslModule>(module.moduleType);
      for (const auto & sv : gm.sensorConfigs) {
        fn(module, std::get<nvsipl::sensorconfig::GmslCameraSensorConfig>(sv));
      }
    } else if (std::holds_alternative<nvsipl::sensorconfig::CoEModule>(module.moduleType)) {
      const auto & cm = std::get<nvsipl::sensorconfig::CoEModule>(module.moduleType);
      for (const auto & sv : cm.sensorConfigs) {
        fn(module, std::get<nvsipl::sensorconfig::CoECameraSensorConfig>(sv));
      }
    }
  }
}

bool currentUserInGroup(const char * group_name)
{
  if (geteuid() == 0) {
    return true;
  }
  const group * required_group = getgrnam(group_name);  // NOLINT(runtime/threadsafe_fn)
  if (required_group == nullptr) {
    return false;
  }
  if (getegid() == required_group->gr_gid) {
    return true;
  }
  const int count = getgroups(0, nullptr);
  if (count < 0) {
    return false;
  }
  std::vector<gid_t> groups(static_cast<size_t>(count));
  if (count > 0 && getgroups(count, groups.data()) < 0) {
    return false;
  }
  return std::find(groups.begin(), groups.end(), required_group->gr_gid) != groups.end();
}

bool currentUserCanAccessDevices(
  const char * group_name, const char * pattern, int mode)
{
  glob_t devices{};
  const int result = glob(pattern, 0, nullptr, &devices);
  if (result != 0) {
    globfree(&devices);
    return result == GLOB_NOMATCH && currentUserInGroup(group_name);
  }
  bool accessible = true;
  for (size_t i = 0; i < devices.gl_pathc; ++i) {
    if (faccessat(AT_FDCWD, devices.gl_pathv[i], mode, AT_EACCESS) != 0) {
      accessible = false;
      break;
    }
  }
  globfree(&devices);
  return accessible;
}

}  // namespace

bool sensorIsYuv422(const nvsipl::sensorconfig::CommonSensorConfig & sensor)
{
  return !sensor.vcInfoList.empty() &&
         sensor.vcInfoList[0].inputFormat == NVSIPL_CAP_INPUT_FORMAT_TYPE_YUV422;
}

SiplEnvOverrides overrideYuvInputFormats(
  nvsipl::sensorconfig::SensorSystemConfig & config,
  const YuvOverrideOptions & options,
  const rclcpp::Logger & logger)
{
  // The NVSIPL_* environment overrides here are specific to the Sensing camera drivers.
  std::string cfa_name = options.cfa;
  if (const char * cfa_env = std::getenv("NVSIPL_FORCE_YUV_CFA"); cfa_env != nullptr) {
    cfa_name = cfa_env;
  }
  const auto & orders = yuvPixelOrders();
  const auto order_it = orders.find(cfa_name);
  if (order_it == orders.end()) {
    std::string supported;
    for (const auto & [name, unused] : orders) {
      supported += (supported.empty() ? "" : ", ") + name;
    }
    throw std::runtime_error(
      "Unknown YUV cfa '" + cfa_name + "' (from NVSIPL_FORCE_YUV_CFA); supported: " +
      supported);
  }

  bool has_yuv_sensor = false;
  std::map<uint32_t, bool> link_is_yuv;

  for (auto & module : config.modules) {
    if (!std::holds_alternative<nvsipl::sensorconfig::GmslModule>(module.moduleType)) {
      continue;
    }
    auto & gm = std::get<nvsipl::sensorconfig::GmslModule>(module.moduleType);

    bool module_is_yuv = false;
    for (auto & sv : gm.sensorConfigs) {
      auto & sensor = std::get<nvsipl::sensorconfig::GmslCameraSensorConfig>(sv);
      if (sensor.name != options.yuv_sensor_name) {
        continue;
      }
      module_is_yuv = true;
      has_yuv_sensor = true;
      for (auto & vc : sensor.vcInfoList) {
        RCLCPP_INFO(
          logger,
          "Forcing sensor %u (%s) to YUV422/%s (was cfa=0x%x inputFormat=%d)",
          sensor.id, sensor.name.c_str(), cfa_name.c_str(),
          vc.cfa, static_cast<int>(vc.inputFormat));
        vc.cfa = order_it->second;
        vc.inputFormat = NVSIPL_CAP_INPUT_FORMAT_TYPE_YUV422;
      }
    }

    if (gm.linkIndex != UINT32_MAX) {
      link_is_yuv[gm.linkIndex] = module_is_yuv;
    }
  }

  // Apply the Sensing EEPROM alias scheme to configurations containing YUV sensors.
  if (has_yuv_sensor) {
    for (auto & module : config.modules) {
      if (!std::holds_alternative<nvsipl::sensorconfig::GmslModule>(module.moduleType)) {
        continue;
      }
      auto & gm = std::get<nvsipl::sensorconfig::GmslModule>(module.moduleType);
      // Stereo sensors share an EEPROM; monocular modules need per-link aliases.
      const bool is_stereo = gm.sensorConfigs.size() > 1U;
      for (auto & eeprom : gm.eepromConfigs) {
        if (is_stereo || gm.linkIndex == UINT32_MAX) {
          eeprom.address.virtualI2CAddress.reset();
        } else {
          eeprom.address.virtualI2CAddress =
            static_cast<uint16_t>(kEepromAliasBase + gm.linkIndex);
        }
      }
    }
  }

  SiplEnvOverrides env;
  if (options.export_link_format_env) {
    for (const auto & [link, is_yuv] : link_is_yuv) {
      env.emplace_back(
        "NVSIPL_SENSOR_INPUT_FORMAT_LINK" + std::to_string(link),
        is_yuv ? "yuv422" : "raw");
    }
  }

  // Mixed rigs only: a GW5 on the bus shortens the VB1940 link restore window.
  if (options.export_vb1940_restore_delay_env && has_yuv_sensor) {
    env.emplace_back("NVSIPL_VB1940_LINK_RESTORE_DELAY_MS", "10");
  }
  return env;
}

void applySiplEnvOverrides(
  const SiplEnvOverrides & env, const rclcpp::Logger & logger)
{
  for (const auto & [key, value] : env) {
    if (setenv(key.c_str(), value.c_str(), 1) != 0) {
      throw std::runtime_error("Failed to set SIPL environment variable " +
                               key + ": " + std::strerror(errno));
    }
    RCLCPP_DEBUG(logger, "%s=%s", key.c_str(), value.c_str());
  }
}

void uniquifySensorGroups(
  nvsipl::sensorconfig::SensorSystemConfig & config,
  const rclcpp::Logger & logger)
{
  std::set<uint32_t> used_groups;
  uint32_t next_group = 1U;

  for (auto & module : config.modules) {
    uint32_t module_group = 0U;
    bool has_group = false;
    auto visit_group = [&](auto & sensors) {
        for (auto & sv : sensors) {
          std::visit([&](auto & sensor) {
              if (sensor.sensorGroup > 0U) {
                module_group = sensor.sensorGroup;
                has_group = true;
              }
            }, sv);
        }
      };

    if (std::holds_alternative<nvsipl::sensorconfig::GmslModule>(module.moduleType)) {
      visit_group(std::get<nvsipl::sensorconfig::GmslModule>(module.moduleType).sensorConfigs);
    } else if (std::holds_alternative<nvsipl::sensorconfig::CoEModule>(module.moduleType)) {
      visit_group(std::get<nvsipl::sensorconfig::CoEModule>(module.moduleType).sensorConfigs);
    }

    if (!has_group) {
      continue;
    }
    if (used_groups.insert(module_group).second) {
      next_group = std::max(next_group, module_group + 1U);
      continue;
    }

    while (used_groups.count(next_group) > 0U) {
      ++next_group;
    }
    const uint32_t new_group = next_group;
    used_groups.insert(new_group);

    auto reassign = [&](auto & sensors) {
        for (auto & sv : sensors) {
          std::visit([&](auto & sensor) {
              if (sensor.sensorGroup > 0U) {
                sensor.sensorGroup = new_group;
              }
            }, sv);
        }
      };
    if (std::holds_alternative<nvsipl::sensorconfig::GmslModule>(module.moduleType)) {
      reassign(std::get<nvsipl::sensorconfig::GmslModule>(module.moduleType).sensorConfigs);
    } else if (std::holds_alternative<nvsipl::sensorconfig::CoEModule>(module.moduleType)) {
      reassign(std::get<nvsipl::sensorconfig::CoEModule>(module.moduleType).sensorConfigs);
    }

    RCLCPP_INFO(
      logger, "Module '%s': sensorGroup %u already taken, remapped to %u",
      module.name.c_str(), module_group, new_group);
  }
}

std::vector<uint32_t> findDuplicateSensorIds(
  const nvsipl::sensorconfig::SensorSystemConfig & config)
{
  std::set<uint32_t> seen;
  std::set<uint32_t> duplicates;
  forEachSensorConst(
    config, [&](const auto & /*module*/, const auto & sensor) {
      if (!seen.insert(sensor.id).second) {
        duplicates.insert(sensor.id);
      }
    });
  return {duplicates.begin(), duplicates.end()};
}

void preflightGroupMembership()
{
  if (geteuid() == 0) {
    return;
  }

  std::vector<std::string> missing_groups;
  const std::array<std::pair<const char *, const char *>, 3> requirements{{
    {"video", "/dev/nvmap"}, {"i2c", "/dev/i2c-*"}, {"gpio", "/dev/gpiochip*"}}};
  for (const auto & [group_name, pattern] : requirements) {
    const int mode = std::strcmp(group_name, "video") == 0 ? R_OK : R_OK | W_OK;
    const bool accessible = currentUserCanAccessDevices(group_name, pattern, mode);
    if (!accessible) {
      missing_groups.emplace_back(group_name);
    }
  }
  if (missing_groups.empty()) {
    return;
  }

  std::string groups;
  for (const auto & group_name : missing_groups) {
    groups += (groups.empty() ? "" : ", ") + group_name;
  }
  throw std::runtime_error(
          "SIPL permission preflight failed: missing access for device group(s): " +
          groups + ". Set group with sudo usermod -aG <group> \"$USER\" and log out and back in. " +
          "Verify device permissions and group membership in the environment running the camera.");
}

}  // namespace config
}  // namespace sipl
}  // namespace isaac_ros
