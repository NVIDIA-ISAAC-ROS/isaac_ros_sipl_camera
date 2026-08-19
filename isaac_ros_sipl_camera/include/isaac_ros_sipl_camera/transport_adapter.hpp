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

#ifndef ISAAC_ROS_SIPL_CAMERA__TRANSPORT_ADAPTER_HPP_
#define ISAAC_ROS_SIPL_CAMERA__TRANSPORT_ADAPTER_HPP_

#include <array>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "rclcpp/logger.hpp"

#include "NvSIPLCameraTypes.hpp"

namespace isaac_ros
{
namespace sipl
{

// CoE override parameters to select which CoE target.
struct CoeOverrides
{
  uint32_t hsb_id{0};                          // matches transport.transportId / module.transportId
  std::optional<std::string> interface_name;   // e.g. "mgbe0_0"
  std::optional<std::string> ip_address;       // e.g. "192.168.0.2"
  std::optional<std::string> mac_address;      // e.g. "8c:1f:64:6d:70:21"
};

// Adapter abstracting the transport-type differences between CoE and GMSL.
// Build one with Create() after parsing the SensorSystemConfig.
class TransportAdapter
{
public:
  enum class Type
  {
    kCoe,
    kGmsl
  };

  explicit TransportAdapter(rclcpp::Logger logger)
  : logger_(logger) {}
  virtual ~TransportAdapter() = default;

  // Returns the SIPL global sensor id backing each pipeline for the requested
  // shape (mono = one id; stereo = left + right). Throws std::runtime_error if
  // the config cannot satisfy the requested shape (no sensors / no stereo pair).
  virtual std::vector<uint32_t> selectSensors(
    const nvsipl::sensorconfig::SensorSystemConfig & cfg,
    bool stereo) const = 0;

  virtual Type type() const = 0;

  // Selects the adapter family from `cfg`'s first transport and applies the
  // matching runtime overrides to `cfg` in place. Throws std::runtime_error if
  // `cfg` has no transports or the transport variant is unrecognized.
  static std::unique_ptr<TransportAdapter> Create(
    nvsipl::sensorconfig::SensorSystemConfig & cfg,
    const CoeOverrides & coe_overrides,
    uint16_t link_mask,
    rclcpp::Logger logger);

protected:
  rclcpp::Logger logger_;
};

// CoE-specific adapter. Applies MAC/IP/interface overrides to `cfg`.
class CoeTransportAdapter : public TransportAdapter
{
public:
  CoeTransportAdapter(
    nvsipl::sensorconfig::SensorSystemConfig & cfg,
    const CoeOverrides & overrides,
    rclcpp::Logger logger);

  std::vector<uint32_t> selectSensors(
    const nvsipl::sensorconfig::SensorSystemConfig & cfg,
    bool stereo) const override;

  Type type() const override {return Type::kCoe;}
};

// GMSL-specific adapter. Filters `cfg` modules by `link_mask`.
class GmslTransportAdapter : public TransportAdapter
{
public:
  // Throws std::runtime_error if `link_mask` selects no links.
  GmslTransportAdapter(
    nvsipl::sensorconfig::SensorSystemConfig & cfg,
    uint16_t link_mask,
    rclcpp::Logger logger);

  std::vector<uint32_t> selectSensors(
    const nvsipl::sensorconfig::SensorSystemConfig & cfg,
    bool stereo) const override;

  Type type() const override {return Type::kGmsl;}
};

// Internal parsing helpers.
namespace internal
{
// Parse "192.168.0.2" to uint32_t (network byte order). Returns nullopt if
// the input is malformed.
std::optional<uint32_t> parseIpAddressV4(const std::string & s);

// Parse "8c:1f:64:6d:70:21" into 6 bytes. Returns nullopt if malformed.
std::optional<std::array<uint8_t, 6>> parseMacAddress(const std::string & s);
}  // namespace internal

}  // namespace sipl
}  // namespace isaac_ros

#endif  // ISAAC_ROS_SIPL_CAMERA__TRANSPORT_ADAPTER_HPP_
