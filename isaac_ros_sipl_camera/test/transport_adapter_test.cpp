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

#include <gtest/gtest.h>

#include "rclcpp/rclcpp.hpp"

namespace
{

using isaac_ros::sipl::CoeTransportAdapter;
using isaac_ros::sipl::GmslTransportAdapter;
using isaac_ros::sipl::TransportAdapter;
using isaac_ros::sipl::internal::parseIpAddressV4;
using isaac_ros::sipl::internal::parseMacAddress;

namespace sc = nvsipl::sensorconfig;

// Build a minimal SensorSystemConfig with one Hawk-shaped GMSL module
// replicated across `num_links` links, two AR0234 sensors per module.
sc::SensorSystemConfig MakeGmslHawkFixture(uint32_t num_links)
{
  sc::SensorSystemConfig cfg;

  sc::TransportConfig t;
  t.transportId = 0;
  t.name = "transportSettings_kenari_AB";
  t.transportType = sc::GmslTransportConfig{};
  cfg.transports.push_back(t);

  uint32_t next_id = 0;
  for (uint32_t link = 0; link < num_links; ++link) {
    sc::GmslModule gm;
    gm.linkIndex = link;
    sc::GmslCameraSensorConfig s0;
    s0.name = "AR0234";
    s0.id = next_id++;
    s0.sensorGroup = 1;
    s0.deviceIndex = 0;
    sc::GmslCameraSensorConfig s1;
    s1.name = "AR0234";
    s1.id = next_id++;
    s1.sensorGroup = 1;
    s1.deviceIndex = 1;
    gm.sensorConfigs.push_back(s0);
    gm.sensorConfigs.push_back(s1);

    sc::ModuleConfig m;
    m.name = "AR0234CS_HAWK_SINGLE_CAMERA";
    m.platformConfig = "AR0234CS_HAWK_SINGLE_CAMERA";
    m.transportId = 0;
    m.moduleType = gm;
    cfg.modules.push_back(m);
  }
  return cfg;
}

sc::SensorSystemConfig MakeCoeEagleFixture()
{
  sc::SensorSystemConfig cfg;

  sc::TransportConfig t;
  t.transportId = 0;
  t.name = "HsbTransport";
  sc::CoETransportConfig coe;
  coe.interfaceName = "mgbe0_0";
  t.transportType = coe;
  cfg.transports.push_back(t);

  sc::ModuleConfig m;
  m.name = "VB1940";
  m.transportId = 0;
  sc::CoEModule cm;
  sc::CoECameraSensorConfig s0;
  s0.name = "VB1940";
  s0.id = 0;
  s0.sensorGroup = 1;
  s0.deviceIndex = 0;
  cm.sensorConfigs.push_back(s0);
  m.moduleType = cm;
  cfg.modules.push_back(m);
  return cfg;
}

TEST(TransportAdapterTest, CreateDispatchesByVariant)
{
  auto logger = rclcpp::get_logger("transport_adapter_test");
  auto gmsl_cfg = MakeGmslHawkFixture(/*num_links=*/4);
  auto adapter_g = TransportAdapter::Create(gmsl_cfg, {}, /*link_mask=*/0x0001, logger);
  ASSERT_NE(adapter_g, nullptr);
  EXPECT_NE(dynamic_cast<GmslTransportAdapter *>(adapter_g.get()), nullptr);
  EXPECT_EQ(adapter_g->type(), TransportAdapter::Type::kGmsl);

  auto coe_cfg = MakeCoeEagleFixture();
  auto adapter_c = TransportAdapter::Create(coe_cfg, {}, /*link_mask=*/0x0001, logger);
  ASSERT_NE(adapter_c, nullptr);
  EXPECT_NE(dynamic_cast<CoeTransportAdapter *>(adapter_c.get()), nullptr);
  EXPECT_EQ(adapter_c->type(), TransportAdapter::Type::kCoe);
}

TEST(TransportAdapterTest, GmslLinkMaskFiltersToRequestedLinks)
{
  auto cfg = MakeGmslHawkFixture(/*num_links=*/4);
  ASSERT_EQ(cfg.modules.size(), 4U);

  auto logger = rclcpp::get_logger("transport_adapter_test");

  // Mask 0x0001 should keep only link 0.
  GmslTransportAdapter adapter(cfg, /*link_mask=*/0x0001, logger);
  ASSERT_EQ(cfg.modules.size(), 1U);
  const auto & mod = cfg.modules.front();
  const auto * gmsl =
    std::get_if<nvsipl::sensorconfig::GmslModule>(&mod.moduleType);
  ASSERT_NE(gmsl, nullptr);
  EXPECT_EQ(gmsl->linkIndex, 0U);
}

TEST(TransportAdapterTest, GmslLinkMaskRejectsEmptyAndFiltersMultiLink)
{
  auto logger = rclcpp::get_logger("transport_adapter_test");

  // An empty mask is an error: nothing would be captured.
  auto cfg_empty = MakeGmslHawkFixture(/*num_links=*/4);
  EXPECT_THROW(
    GmslTransportAdapter(cfg_empty, /*link_mask=*/0x0000, logger), std::runtime_error);

  // A multi-link mask keeps exactly the selected links: multiple Hawk cameras on
  // one deserializer (multiple links) is a supported selection.
  auto cfg_multi = MakeGmslHawkFixture(/*num_links=*/4);
  EXPECT_NO_THROW(
    GmslTransportAdapter(cfg_multi, /*link_mask=*/0x0003, logger));
  EXPECT_EQ(cfg_multi.modules.size(), 2U);
}

TEST(TransportAdapterTest, GmslSelectSensorsStereoForOneHawk)
{
  auto cfg = MakeGmslHawkFixture(/*num_links=*/1);
  auto logger = rclcpp::get_logger("transport_adapter_test");
  GmslTransportAdapter adapter(cfg, /*link_mask=*/0x0001, logger);

  auto sel = adapter.selectSensors(cfg, /*stereo=*/true);

  ASSERT_EQ(sel.size(), 2U);
  EXPECT_EQ(sel[0], 0U);
  EXPECT_EQ(sel[1], 1U);
}

TEST(TransportAdapterTest, SelectSensorsStereoWithoutPairThrows)
{
  // Single-sensor config cannot satisfy a stereo request.
  auto cfg = MakeCoeEagleFixture();
  auto logger = rclcpp::get_logger("transport_adapter_test");
  CoeTransportAdapter adapter(cfg, {}, logger);

  EXPECT_THROW(adapter.selectSensors(cfg, /*stereo=*/true), std::runtime_error);
}

TEST(TransportAdapterTest, SelectSensorsMonoWithoutDeviceIndexZeroThrows)
{
  // A config whose only sensor has a non-zero deviceIndex cannot serve a mono
  // request, which expects the primary (deviceIndex 0) sensor.
  auto cfg = MakeCoeEagleFixture();
  std::get<sc::CoECameraSensorConfig>(
    std::get<sc::CoEModule>(cfg.modules[0].moduleType).sensorConfigs[0]).deviceIndex = 1;

  auto logger = rclcpp::get_logger("transport_adapter_test");
  CoeTransportAdapter adapter(cfg, {}, logger);

  EXPECT_THROW(adapter.selectSensors(cfg, /*stereo=*/false), std::runtime_error);
}

TEST(TransportAdapterTest, MacAddressParser)
{
  auto bytes = parseMacAddress("8c:1f:64:6d:70:21");
  ASSERT_TRUE(bytes.has_value());
  EXPECT_EQ((*bytes)[0], 0x8c);
  EXPECT_EQ((*bytes)[5], 0x21);

  EXPECT_FALSE(parseMacAddress("").has_value());
  EXPECT_FALSE(parseMacAddress("not a mac").has_value());
  EXPECT_FALSE(parseMacAddress("8c:1f:64:6d:70").has_value());
  // Trailing junk is rejected.
  EXPECT_FALSE(parseMacAddress("8c:1f:64:6d:70:21X").has_value());
}

TEST(TransportAdapterTest, IpAddressParser)
{
  auto ip = parseIpAddressV4("192.168.0.2");
  ASSERT_TRUE(ip.has_value());
  // 192.168.0.2 in network byte order = 0x020 0a8c0
  EXPECT_EQ(*ip, htonl(0xC0A80002U));

  EXPECT_FALSE(parseIpAddressV4("").has_value());
  EXPECT_FALSE(parseIpAddressV4("not.an.ip").has_value());
}

TEST(TransportAdapterTest, CoeApplyOverridesMutatesMatchingHsbOnly)
{
  // Two CoE transports / modules with distinct transportIds. Override should
  // only mutate the one matching coe_overrides.hsb_id; the other stays put.
  sc::SensorSystemConfig cfg;
  for (uint32_t hsb = 0; hsb < 2; ++hsb) {
    sc::TransportConfig t;
    t.transportId = hsb;
    sc::CoETransportConfig coe_t;
    coe_t.interfaceName = "default";
    coe_t.ipAddress = 0;
    t.transportType = coe_t;
    cfg.transports.push_back(t);

    sc::ModuleConfig m;
    m.transportId = hsb;
    sc::CoEModule cm;
    sc::CoECameraSensorConfig s;
    s.id = hsb;
    s.macAddress = {0, 0, 0, 0, 0, 0};
    s.ipAddress = 0;
    cm.sensorConfigs.push_back(s);
    m.moduleType = cm;
    cfg.modules.push_back(m);
  }

  auto logger = rclcpp::get_logger("transport_adapter_test");
  isaac_ros::sipl::CoeOverrides overrides;
  overrides.hsb_id = 1;
  overrides.interface_name = "mgbe0_0";
  overrides.ip_address = "192.168.0.2";
  overrides.mac_address = "8c:1f:64:6d:70:21";

  CoeTransportAdapter adapter(cfg, overrides, logger);

  // hsb 0 should be untouched.
  const auto & t0 = std::get<sc::CoETransportConfig>(cfg.transports[0].transportType);
  EXPECT_EQ(t0.interfaceName, "default");
  EXPECT_EQ(t0.ipAddress, 0U);
  const auto & s0 = std::get<sc::CoECameraSensorConfig>(
    std::get<sc::CoEModule>(cfg.modules[0].moduleType).sensorConfigs[0]);
  EXPECT_EQ(s0.ipAddress, 0U);

  // hsb 1 should have the override applied.
  const auto & t1 = std::get<sc::CoETransportConfig>(cfg.transports[1].transportType);
  EXPECT_EQ(t1.interfaceName, "mgbe0_0");
  EXPECT_EQ(t1.ipAddress, htonl(0xC0A80002U));
  const auto & s1 = std::get<sc::CoECameraSensorConfig>(
    std::get<sc::CoEModule>(cfg.modules[1].moduleType).sensorConfigs[0]);
  EXPECT_EQ(s1.ipAddress, htonl(0xC0A80002U));
  EXPECT_EQ(s1.macAddress[0], 0x8c);
  EXPECT_EQ(s1.macAddress[5], 0x21);
}

TEST(TransportAdapterTest, CoeApplyOverridesNoOpForGmslConfig)
{
  // GMSL fixture, but called via the CoE adapter — should be entirely
  // ignored (no exception, no modification, no module filtering).
  auto cfg = MakeGmslHawkFixture(/*num_links=*/2);
  const size_t modules_before = cfg.modules.size();

  auto logger = rclcpp::get_logger("transport_adapter_test");
  isaac_ros::sipl::CoeOverrides overrides;
  overrides.hsb_id = 0;
  overrides.ip_address = "192.168.0.2";
  CoeTransportAdapter adapter(cfg, overrides, logger);

  EXPECT_EQ(cfg.modules.size(), modules_before);
}

TEST(TransportAdapterTest, CoeUnmatchedHsbIdWithOverridesThrows)
{
  // Overrides target an hsb_id that no CoE module has, so they would be
  // silently dropped: fail fast instead.
  auto cfg = MakeCoeEagleFixture();
  auto logger = rclcpp::get_logger("transport_adapter_test");
  isaac_ros::sipl::CoeOverrides overrides;
  overrides.hsb_id = 99;
  overrides.ip_address = "192.168.0.2";

  EXPECT_THROW(CoeTransportAdapter(cfg, overrides, logger), std::runtime_error);
}

TEST(TransportAdapterTest, CoeUnmatchedHsbIdWithoutOverridesDoesNotThrow)
{
  // No overrides requested, so an unmatched hsb_id is a harmless no-op.
  auto cfg = MakeCoeEagleFixture();
  auto logger = rclcpp::get_logger("transport_adapter_test");
  isaac_ros::sipl::CoeOverrides overrides;
  overrides.hsb_id = 99;

  EXPECT_NO_THROW(CoeTransportAdapter(cfg, overrides, logger));
}

}  // namespace
