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

#include "isaac_ros_sipl_camera/transport_adapter.hpp"

#include <arpa/inet.h>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <optional>
#include <stdexcept>
#include <string>
#include <variant>

#include "rclcpp/rclcpp.hpp"

namespace isaac_ros
{
namespace sipl
{

namespace
{

using SiplSensorCfgPtr = const nvsipl::sensorconfig::CommonSensorConfig *;

// Flattens the nested module -> sensor structure of `cfg` into a flat list,
// keeping only sensors whose module/sensor variants match <ModuleT, SensorT>
// (i.e. one transport family).
template<typename ModuleT, typename SensorT>
std::vector<SiplSensorCfgPtr> collectSensors(
  const nvsipl::sensorconfig::SensorSystemConfig & cfg)
{
  std::vector<SiplSensorCfgPtr> sensors;
  for (const auto & module_cfg : cfg.modules) {
    const auto * mod = std::get_if<ModuleT>(&module_cfg.moduleType);
    if (mod == nullptr) {continue;}  // module belongs to the other family
    for (const auto & sensor_v : mod->sensorConfigs) {
      const auto * sensor = std::get_if<SensorT>(&sensor_v);
      if (sensor == nullptr) {continue;}
      sensors.push_back(sensor);
    }
  }
  return sensors;
}

// Per SIPL's CommonSensorConfig, sensors sharing a `sensorGroup` form one
// logical camera and `deviceIndex` distinguishes sensors within a module. A
// stereo pair is therefore two sensors with the same non-zero sensorGroup and
// deviceIndex 0 (left) and 1 (right). SensorGroup 0 is the default/ungrouped
// value, so it is not treated as stereo. Returns the (left, right) pair, or
// nullopt if none exists.
std::optional<std::pair<SiplSensorCfgPtr, SiplSensorCfgPtr>> findStereoPair(
  const std::vector<SiplSensorCfgPtr> & sensors)
{
  for (const auto * left : sensors) {
    if (left->sensorGroup == 0U || left->deviceIndex != 0U) {
      continue;
    }
    for (const auto * right : sensors) {
      if (right->sensorGroup == left->sensorGroup && right->deviceIndex == 1U) {
        return std::make_pair(left, right);
      }
    }
  }
  return std::nullopt;
}

// Expect deviceIndex 0 to exist in any valid config. Throws if not found.
SiplSensorCfgPtr selectMonoSensor(const std::vector<SiplSensorCfgPtr> & sensors)
{
  const auto primary = std::find_if(
    sensors.begin(), sensors.end(),
    [](SiplSensorCfgPtr sensor) {return sensor->deviceIndex == 0U;});
  if (primary == sensors.end()) {
    throw std::runtime_error("mono request: no sensor with deviceIndex 0");
  }
  return *primary;
}

std::vector<uint32_t> selectSensorsFromList(
  const std::vector<SiplSensorCfgPtr> & sensors,
  bool stereo,
  const char * adapter_name,
  const rclcpp::Logger & logger)
{
  if (sensors.empty()) {
    throw std::runtime_error(
      std::string(adapter_name) + ": no sensors in filtered config");
  }

  if (!stereo) {
    return {selectMonoSensor(sensors)->id};
  }

  const auto pair = findStereoPair(sensors);
  if (pair.has_value()) {
    return {pair->first->id, pair->second->id};
  }

  // Fallback if no formal stereo pair (shared non-zero
  // sensorGroup with deviceIndex 0/1) exists, but we still want to stream two
  // independent pipelines (e.g. VB1940). Take the first two sensors.
  if (sensors.size() < 2U) {
    throw std::runtime_error(
      std::string(adapter_name) +
      ": stereo requested but fewer than two sensors available");
  }
  RCLCPP_WARN(
    logger,
    "%s: stereo requested but no formal stereo pair found; falling back to the "
    "first two sensors (ids %u, %u) as independent pipelines",
    adapter_name, sensors[0]->id, sensors[1]->id);
  return {sensors[0]->id, sensors[1]->id};
}

}  // namespace

namespace internal
{

std::optional<uint32_t> parseIpAddressV4(const std::string & s)
{
  if (s.empty()) {
    return std::nullopt;
  }
  in_addr addr{};
  if (inet_pton(AF_INET, s.c_str(), &addr) != 1) {
    return std::nullopt;
  }
  return addr.s_addr;
}

std::optional<std::array<uint8_t, 6>> parseMacAddress(const std::string & s)
{
  if (s.empty()) {
    return std::nullopt;
  }
  unsigned int parsed[6];
  char trailing = '\0';
  const int matched = std::sscanf(
    s.c_str(), "%2x:%2x:%2x:%2x:%2x:%2x%c",
    &parsed[0], &parsed[1], &parsed[2], &parsed[3], &parsed[4], &parsed[5],
    &trailing);
  if (matched != 6) {
    return std::nullopt;
  }
  std::array<uint8_t, 6> bytes{};
  for (size_t i = 0; i < bytes.size(); ++i) {
    bytes[i] = static_cast<uint8_t>(parsed[i]);
  }
  return bytes;
}

}  // namespace internal

std::unique_ptr<TransportAdapter> TransportAdapter::Create(
  nvsipl::sensorconfig::SensorSystemConfig & cfg,
  const CoeOverrides & coe_overrides,
  uint16_t link_mask,
  rclcpp::Logger logger)
{
  if (cfg.transports.empty()) {
    throw std::runtime_error(
      "TransportAdapter::Create: SensorSystemConfig has no transports");
  }

  // Dispatch on the first transport's variant; the chosen adapter applies its
  // overrides to `cfg` during construction.
  const auto & first = cfg.transports.front().transportType;
  if (std::holds_alternative<nvsipl::sensorconfig::CoETransportConfig>(first)) {
    return std::make_unique<CoeTransportAdapter>(cfg, coe_overrides, logger);
  }
  if (std::holds_alternative<nvsipl::sensorconfig::GmslTransportConfig>(first)) {
    return std::make_unique<GmslTransportAdapter>(cfg, link_mask, logger);
  }
  throw std::runtime_error(
    "TransportAdapter::Create: unrecognized transport variant");
}

// -----------------------------------------------------------------------------
// CoE adapter (Eagle, etc.)
// -----------------------------------------------------------------------------

// CoE override pass: mutates CoETransportConfig (interfaceName, ipAddress)
// and CoECameraSensorConfig (macAddress, ipAddress) entries whose
// transportId matches `coe_overrides.hsb_id`. Single-HSB Eagle is the only
// shape we test today; multi-HSB needs SIPL's --coeConfigOverridePath CSV
// path instead.
CoeTransportAdapter::CoeTransportAdapter(
  nvsipl::sensorconfig::SensorSystemConfig & cfg,
  const CoeOverrides & coe_overrides,
  rclcpp::Logger logger)
: TransportAdapter(logger)
{
  std::optional<std::array<uint8_t, 6>> mac_bytes;
  if (coe_overrides.mac_address.has_value() && !coe_overrides.mac_address->empty()) {
    mac_bytes = internal::parseMacAddress(*coe_overrides.mac_address);
    if (!mac_bytes.has_value()) {
      RCLCPP_WARN(
        logger_, "Invalid MAC address format: %s",
        coe_overrides.mac_address->c_str());
    } else {
      RCLCPP_INFO(
        logger_, "MAC address override: %s", coe_overrides.mac_address->c_str());
    }
  }

  std::optional<uint32_t> ip_value;
  if (coe_overrides.ip_address.has_value() && !coe_overrides.ip_address->empty()) {
    ip_value = internal::parseIpAddressV4(*coe_overrides.ip_address);
    if (!ip_value.has_value()) {
      RCLCPP_WARN(
        logger_, "Invalid IP address format: %s",
        coe_overrides.ip_address->c_str());
    } else {
      RCLCPP_INFO(
        logger_, "IP address override: %s", coe_overrides.ip_address->c_str());
    }
  }

  const bool has_iface = coe_overrides.interface_name.has_value() &&
    !coe_overrides.interface_name->empty();

  // Override matching CoE transports.
  for (auto & transport : cfg.transports) {
    auto * coe = std::get_if<nvsipl::sensorconfig::CoETransportConfig>(
      &transport.transportType);
    if (coe == nullptr) {continue;}
    if (transport.transportId != coe_overrides.hsb_id) {continue;}
    if (has_iface) {
      coe->interfaceName = *coe_overrides.interface_name;
    }
    if (ip_value.has_value()) {
      coe->ipAddress = *ip_value;
    }
    RCLCPP_INFO(
      logger_,
      "Applied CoE transport override: transportId=%u interface=%s",
      transport.transportId,
      has_iface ? coe_overrides.interface_name->c_str() : "(default)");
  }

  // Override matching CoE modules' sensor MAC / IP.
  bool applied_to_any_module = false;
  bool has_any_coe = false;
  for (auto & module : cfg.modules) {
    auto * coe_mod = std::get_if<nvsipl::sensorconfig::CoEModule>(&module.moduleType);
    if (coe_mod == nullptr) {continue;}
    has_any_coe = true;
    if (module.transportId != coe_overrides.hsb_id) {continue;}
    for (auto & sensor_v : coe_mod->sensorConfigs) {
      auto * coe_sensor =
        std::get_if<nvsipl::sensorconfig::CoECameraSensorConfig>(&sensor_v);
      if (coe_sensor == nullptr) {continue;}
      if (ip_value.has_value()) {coe_sensor->ipAddress = *ip_value;}
      if (mac_bytes.has_value()) {
        std::memcpy(coe_sensor->macAddress.data(), mac_bytes->data(), 6);
      }
    }
    applied_to_any_module = true;
    RCLCPP_INFO(
      logger_,
      "Applied CoE module overrides: name=%s transportId=%u",
      module.name.c_str(), module.transportId);
  }

  if (has_any_coe && !applied_to_any_module) {
    // A requested override that matches no module would be silently dropped, so
    // fail loudly. With nothing requested the mismatch is a harmless no-op.
    const bool overrides_requested =
      has_iface || ip_value.has_value() || mac_bytes.has_value();
    if (overrides_requested) {
      RCLCPP_ERROR(
        logger_,
        "CoeTransportAdapter: hsb_id=%u matched no CoE module; requested "
        "overrides were not applied",
        coe_overrides.hsb_id);
      throw std::runtime_error(
        "CoeTransportAdapter: hsb_id=" + std::to_string(coe_overrides.hsb_id) +
        " matched no CoE module; requested overrides were not applied");
    }
    RCLCPP_WARN(
      logger_, "CoeTransportAdapter: no CoE module matched hsb_id=%u",
      coe_overrides.hsb_id);
  }
}

std::vector<uint32_t> CoeTransportAdapter::selectSensors(
  const nvsipl::sensorconfig::SensorSystemConfig & cfg,
  bool stereo) const
{
  const auto sensors = collectSensors<
    nvsipl::sensorconfig::CoEModule,
    nvsipl::sensorconfig::CoECameraSensorConfig>(cfg);
  return selectSensorsFromList(
    sensors, stereo, "CoeTransportAdapter::selectSensors", logger_);
}

// -----------------------------------------------------------------------------
// GMSL adapter
// -----------------------------------------------------------------------------

GmslTransportAdapter::GmslTransportAdapter(
  nvsipl::sensorconfig::SensorSystemConfig & cfg,
  uint16_t link_mask,
  rclcpp::Logger logger)
: TransportAdapter(logger)
{
  if (link_mask == 0U) {
    throw std::runtime_error("GMSL link_mask must select at least one link; got 0x0000");
  }

  // Apply the link selection. SIPL returns one module per populated link, so
  // when multiple GMSL cameras are detected on different links this keeps only
  // the modules whose linkIndex is set in link_mask and drops the rest.
  size_t kept = 0;
  cfg.modules.erase(
    std::remove_if(
      cfg.modules.begin(), cfg.modules.end(),
      [&](const nvsipl::sensorconfig::ModuleConfig & m) {
        const auto * gmsl = std::get_if<nvsipl::sensorconfig::GmslModule>(
          &m.moduleType);
        if (gmsl == nullptr) {
          return false;  // leave non-GMSL modules untouched
        }
        const uint32_t link = gmsl->linkIndex;
        const bool enabled = (link < 16U) && ((link_mask >> link) & 0x1U);
        if (enabled) {
          ++kept;
        }
        return !enabled;
      }),
    cfg.modules.end());

  RCLCPP_INFO(
    logger_,
    "GmslTransportAdapter: link_mask=0x%04x kept %zu module(s)",
    static_cast<unsigned>(link_mask), kept);
}

std::vector<uint32_t> GmslTransportAdapter::selectSensors(
  const nvsipl::sensorconfig::SensorSystemConfig & cfg,
  bool stereo) const
{
  const auto sensors = collectSensors<
    nvsipl::sensorconfig::GmslModule,
    nvsipl::sensorconfig::GmslCameraSensorConfig>(cfg);
  return selectSensorsFromList(
    sensors, stereo, "GmslTransportAdapter::selectSensors", logger_);
}

}  // namespace sipl
}  // namespace isaac_ros
