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


#include <gtest/gtest.h>

#include <algorithm>
#include <cstdlib>
#include <string>
#include <variant>
#include <vector>

#include "rclcpp/rclcpp.hpp"

#include "isaac_ros_sipl_camera/sipl_system_config.hpp"

namespace sc = nvsipl::sensorconfig;
namespace config = isaac_ros::sipl::config;

namespace
{

rclcpp::Logger testLogger()
{
  return rclcpp::get_logger("sipl_system_config_test");
}

sc::GmslCameraSensorConfig MakeSensor(
  const std::string & name, uint32_t id, uint32_t group, uint8_t device_index,
  NvSiplCapInputFormatType input_format, uint32_t cfa)
{
  sc::GmslCameraSensorConfig s;
  s.name = name;
  s.id = id;
  s.sensorGroup = group;
  s.deviceIndex = device_index;
  sc::CommonSensorConfig::VirtualChannelInfo vc;
  vc.inputFormat = input_format;
  vc.cfa = cfa;
  vc.resolution.width = 1920;
  vc.resolution.height = 1536;
  s.vcInfoList.push_back(vc);
  return s;
}

sc::ModuleConfig MakeModule(
  const std::string & name, uint32_t link,
  std::vector<sc::GmslCameraSensorConfig> sensors)
{
  sc::GmslModule gm;
  gm.linkIndex = link;
  for (auto & s : sensors) {
    gm.sensorConfigs.push_back(s);
  }
  sc::EEPROMConfig eeprom;
  eeprom.name = "BR24H128";
  eeprom.address.i2cAddress = 0x50;
  gm.eepromConfigs.push_back(eeprom);

  sc::ModuleConfig m;
  m.name = name;
  m.transportId = 0;
  m.moduleType = gm;
  return m;
}

/// Reproduce the query parser output: the GW5 YUV fields arrive as RAW defaults.
sc::SensorSystemConfig MakeMixedFixture()
{
  sc::SensorSystemConfig cfg;

  sc::TransportConfig t;
  t.transportId = 0;
  t.name = "csi-ab";
  t.transportType = sc::GmslTransportConfig{};
  cfg.transports.push_back(t);

  cfg.modules.push_back(
    MakeModule(
      "S56C_MIX", 0,
      {MakeSensor("VB1940", 0, 1, 0, NVSIPL_CAP_INPUT_FORMAT_TYPE_RAW10,
        NVSIPL_PIXEL_ORDER_GBRG),
        MakeSensor("VB1940", 1, 1, 1, NVSIPL_CAP_INPUT_FORMAT_TYPE_RAW10,
        NVSIPL_PIXEL_ORDER_GBRG)}));
  cfg.modules.push_back(
    MakeModule(
      "SHF3L_MIX_L2", 2,
      {MakeSensor("GW5", 2, 2, 0, NVSIPL_CAP_INPUT_FORMAT_TYPE_RAW12, 0U)}));
  cfg.modules.push_back(
    MakeModule(
      "SHF3L_MIX_L3", 3,
      {MakeSensor("GW5", 3, 3, 0, NVSIPL_CAP_INPUT_FORMAT_TYPE_RAW12, 0U)}));
  return cfg;
}

const sc::GmslCameraSensorConfig & SensorAt(
  const sc::SensorSystemConfig & cfg, size_t module_index, size_t sensor_index)
{
  const auto & gm = std::get<sc::GmslModule>(cfg.modules[module_index].moduleType);
  return std::get<sc::GmslCameraSensorConfig>(gm.sensorConfigs[sensor_index]);
}

const sc::GmslModule & ModuleAt(const sc::SensorSystemConfig & cfg, size_t index)
{
  return std::get<sc::GmslModule>(cfg.modules[index].moduleType);
}

}  // namespace

TEST(SiplConfigFixupsTest, OverrideRewritesOnlyYuvSensors)
{
  auto cfg = MakeMixedFixture();
  config::overrideYuvInputFormats(cfg, {}, testLogger());

  for (size_t module_index : {1U, 2U}) {
    const auto & sensor = SensorAt(cfg, module_index, 0);
    EXPECT_EQ(sensor.vcInfoList[0].inputFormat, NVSIPL_CAP_INPUT_FORMAT_TYPE_YUV422);
    EXPECT_EQ(sensor.vcInfoList[0].cfa, NVSIPL_PIXEL_ORDER_UYVY);
  }

  for (size_t sensor_index : {0U, 1U}) {
    const auto & sensor = SensorAt(cfg, 0, sensor_index);
    EXPECT_EQ(sensor.vcInfoList[0].inputFormat, NVSIPL_CAP_INPUT_FORMAT_TYPE_RAW10);
    EXPECT_EQ(sensor.vcInfoList[0].cfa, NVSIPL_PIXEL_ORDER_GBRG);
  }
}

TEST(SiplConfigFixupsTest, SensorIsYuv422DependsOnOverrideOrdering)
{
  auto cfg = MakeMixedFixture();
  EXPECT_FALSE(config::sensorIsYuv422(SensorAt(cfg, 1, 0)));

  config::overrideYuvInputFormats(cfg, {}, testLogger());

  EXPECT_TRUE(config::sensorIsYuv422(SensorAt(cfg, 1, 0)));
  EXPECT_TRUE(config::sensorIsYuv422(SensorAt(cfg, 2, 0)));
  EXPECT_FALSE(config::sensorIsYuv422(SensorAt(cfg, 0, 0)));
}

TEST(SiplConfigFixupsTest, OverrideAssignsPerLinkEepromAliases)
{
  auto cfg = MakeMixedFixture();
  config::overrideYuvInputFormats(cfg, {}, testLogger());

  EXPECT_FALSE(ModuleAt(cfg, 0).eepromConfigs[0].address.virtualI2CAddress.has_value());

  ASSERT_TRUE(ModuleAt(cfg, 1).eepromConfigs[0].address.virtualI2CAddress.has_value());
  EXPECT_EQ(*ModuleAt(cfg, 1).eepromConfigs[0].address.virtualI2CAddress, 0x32);
  ASSERT_TRUE(ModuleAt(cfg, 2).eepromConfigs[0].address.virtualI2CAddress.has_value());
  EXPECT_EQ(*ModuleAt(cfg, 2).eepromConfigs[0].address.virtualI2CAddress, 0x33);
}

TEST(SiplConfigFixupsTest, OverrideLeavesRawOnlyConfigsEepromUntouched)
{
  sc::SensorSystemConfig cfg;
  sc::TransportConfig t;
  t.transportId = 0;
  t.transportType = sc::GmslTransportConfig{};
  cfg.transports.push_back(t);
  cfg.modules.push_back(
    MakeModule(
      "AR0234CS_HAWK", 1,
      {MakeSensor("AR0234", 0, 1, 0, NVSIPL_CAP_INPUT_FORMAT_TYPE_RAW12,
        NVSIPL_PIXEL_ORDER_GBRG)}));

  config::overrideYuvInputFormats(cfg, {}, testLogger());

  EXPECT_FALSE(ModuleAt(cfg, 0).eepromConfigs[0].address.virtualI2CAddress.has_value());
  EXPECT_EQ(ModuleAt(cfg, 0).eepromConfigs[0].address.i2cAddress, 0x50);
  EXPECT_EQ(
    SensorAt(cfg, 0, 0).vcInfoList[0].inputFormat, NVSIPL_CAP_INPUT_FORMAT_TYPE_RAW12);
}

TEST(SiplConfigFixupsTest, OverrideRejectsUnknownCfa)
{
  auto cfg = MakeMixedFixture();
  config::YuvOverrideOptions options;
  options.cfa = "uyuv";  // typo for "uyvy"

  EXPECT_THROW(
    config::overrideYuvInputFormats(cfg, options, testLogger()), std::runtime_error);
  EXPECT_FALSE(config::sensorIsYuv422(SensorAt(cfg, 1, 0)));
}

TEST(SiplConfigFixupsTest, OverrideExportsLinkFormatEnvironment)
{
  auto cfg = MakeMixedFixture();
  const auto env = config::overrideYuvInputFormats(cfg, {}, testLogger());
  config::applySiplEnvOverrides(env, testLogger());

  EXPECT_STREQ(std::getenv("NVSIPL_SENSOR_INPUT_FORMAT_LINK0"), "raw");
  EXPECT_STREQ(std::getenv("NVSIPL_SENSOR_INPUT_FORMAT_LINK2"), "yuv422");
  EXPECT_STREQ(std::getenv("NVSIPL_SENSOR_INPUT_FORMAT_LINK3"), "yuv422");
  EXPECT_STREQ(std::getenv("NVSIPL_VB1940_LINK_RESTORE_DELAY_MS"), "10");
}

TEST(SiplConfigFixupsTest, ApplyEnvironmentRejectsInvalidVariableName)
{
  EXPECT_THROW(
    config::applySiplEnvOverrides({{"INVALID=NAME", "value"}}, testLogger()),
    std::runtime_error);
}

TEST(SiplConfigFixupsTest, OverrideOmitsRestoreDelayWithoutYuvSensors)
{
  sc::SensorSystemConfig cfg;
  sc::TransportConfig t;
  t.transportId = 0;
  t.transportType = sc::GmslTransportConfig{};
  cfg.transports.push_back(t);
  cfg.modules.push_back(
    MakeModule(
      "S56C_MIX", 0,
      {MakeSensor("VB1940", 0, 1, 0, NVSIPL_CAP_INPUT_FORMAT_TYPE_RAW10,
        NVSIPL_PIXEL_ORDER_GBRG)}));

  const auto env = config::overrideYuvInputFormats(cfg, {}, testLogger());
  EXPECT_EQ(
    std::find_if(
      env.begin(), env.end(),
      [](const auto & kv) {return kv.first == "NVSIPL_VB1940_LINK_RESTORE_DELAY_MS";}),
    env.end());
}

TEST(SiplConfigFixupsTest, FindsDuplicateSensorIds)
{
  auto cfg = MakeMixedFixture();
  const auto single_transport = cfg.modules;
  cfg.modules.insert(cfg.modules.end(), single_transport.begin(), single_transport.end());

  EXPECT_EQ(config::findDuplicateSensorIds(cfg), (std::vector<uint32_t>{0, 1, 2, 3}));

  cfg.modules = single_transport;
  EXPECT_TRUE(config::findDuplicateSensorIds(cfg).empty());
}

TEST(SiplConfigFixupsTest, UniquifiesCollidingSensorGroups)
{
  sc::SensorSystemConfig cfg;
  sc::TransportConfig t;
  t.transportId = 0;
  t.transportType = sc::GmslTransportConfig{};
  cfg.transports.push_back(t);
  cfg.modules.push_back(
    MakeModule(
      "S56C_MIX", 0,
      {MakeSensor("VB1940", 0, 1, 0, NVSIPL_CAP_INPUT_FORMAT_TYPE_RAW10, 0U),
        MakeSensor("VB1940", 1, 1, 1, NVSIPL_CAP_INPUT_FORMAT_TYPE_RAW10, 0U)}));
  cfg.modules.push_back(
    MakeModule(
      "S56C_MIX", 1,
      {MakeSensor("VB1940", 2, 1, 0, NVSIPL_CAP_INPUT_FORMAT_TYPE_RAW10, 0U),
        MakeSensor("VB1940", 3, 1, 1, NVSIPL_CAP_INPUT_FORMAT_TYPE_RAW10, 0U)}));

  config::uniquifySensorGroups(cfg, testLogger());

  EXPECT_EQ(SensorAt(cfg, 0, 0).sensorGroup, 1U);
  EXPECT_EQ(SensorAt(cfg, 0, 1).sensorGroup, 1U);
  const uint32_t second = SensorAt(cfg, 1, 0).sensorGroup;
  EXPECT_NE(second, 1U);
  EXPECT_EQ(SensorAt(cfg, 1, 1).sensorGroup, second);
}

TEST(SiplConfigFixupsTest, UniquifyIsNoOpForDistinctGroups)
{
  auto cfg = MakeMixedFixture();
  config::uniquifySensorGroups(cfg, testLogger());

  EXPECT_EQ(SensorAt(cfg, 0, 0).sensorGroup, 1U);
  EXPECT_EQ(SensorAt(cfg, 0, 1).sensorGroup, 1U);
  EXPECT_EQ(SensorAt(cfg, 1, 0).sensorGroup, 2U);
  EXPECT_EQ(SensorAt(cfg, 2, 0).sensorGroup, 3U);
}
