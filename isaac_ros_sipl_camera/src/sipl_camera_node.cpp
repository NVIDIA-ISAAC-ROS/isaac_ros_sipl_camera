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

#include "isaac_ros_sipl_camera/sipl_camera_node.hpp"

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <future>
#include <limits>
#include <sstream>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

#include "NvCamFsync.h"

#include "ament_index_cpp/get_package_share_directory.hpp"
#include "cuda_buffer/cuda_buffer_api.hpp"
#include "tf2/LinearMath/Quaternion.hpp"

#include "isaac_ros_sipl_camera/sipl_system_config.hpp"

using namespace std::chrono_literals;

namespace isaac_ros
{
namespace sipl
{

// CoE requires exactly four capture buffers. GMSL can use a deeper pool to
// absorb short consumer stalls without forcing ICP drops.
constexpr uint16_t kDefaultCaptureBufferCount = 4U;
constexpr uint16_t kMaxGmslCaptureBufferCount = 32U;
constexpr uint16_t kIsp0BufferCount = 4U;

namespace
{

// SIPL and its driver environment require a single owner per process.
std::atomic<bool> g_sipl_env_owned{false};

// SIPL's EnableMask spends a whole nibble per link (LINK_0=0x0001,
// LINK_1=0x0010, LINK_2=0x0100, LINK_3=0x1000) where `link_mask` uses plain bit
// positions, so only the low 8 links are expressible in the 32-bit mask.
uint32_t toSiplEnableMask(uint16_t link_mask)
{
  constexpr unsigned kMaxSiplLinks = 8U;
  if ((link_mask >> kMaxSiplLinks) != 0U) {
    throw std::runtime_error(
      "link_mask selects a link above " + std::to_string(kMaxSiplLinks - 1U) +
      ", which SIPL's EnableMask cannot encode; got " + std::to_string(link_mask));
  }
  uint32_t enable_mask = 0U;
  for (unsigned link = 0U; link < kMaxSiplLinks; ++link) {
    if (((link_mask >> link) & 0x1U) != 0U) {
      enable_mask |= 1U << (4U * link);
    }
  }
  return enable_mask;
}

bool configIsGmsl(const nvsipl::sensorconfig::SensorSystemConfig & cfg)
{
  return !cfg.transports.empty() &&
         std::holds_alternative<nvsipl::sensorconfig::GmslTransportConfig>(
    cfg.transports.front().transportType);
}

std::string siplNotificationName(uint16_t event_code)
{
  using NotifType = nvsipl::NvSIPLPipelineNotifier::NotificationType;
  switch (static_cast<NotifType>(event_code)) {
    case NotifType::NOTIF_WARN_ICP_FRAME_DROP:
      return "ICP_FRAME_DROP";
    case NotifType::NOTIF_WARN_ICP_FRAME_DISCONTINUITY:
      return "ICP_FRAME_DISCONTINUITY";
    case NotifType::NOTIF_WARN_ICP_CAPTURE_TIMEOUT:
      return "ICP_CAPTURE_TIMEOUT";
    case NotifType::NOTIF_WARN_ICP_APP_BACK_PRESSURE:
      return "ICP_APP_BACK_PRESSURE";
    case NotifType::NOTIF_ERROR_ICP_BAD_INPUT_STREAM:
      return "ICP_BAD_INPUT_STREAM";
    case NotifType::NOTIF_ERROR_ICP_CAPTURE_FAILURE:
      return "ICP_CAPTURE_FAILURE";
    case NotifType::NOTIF_ERROR_ICP_EMB_DATA_PARSE_FAILURE:
      return "ICP_EMB_DATA_PARSE_FAILURE";
    case NotifType::NOTIF_ERROR_ISP_PROCESSING_FAILURE:
      return "ISP_PROCESSING_FAILURE";
    case NotifType::NOTIF_ERROR_ACP_PROCESSING_FAILURE:
      return "ACP_PROCESSING_FAILURE";
    case NotifType::NOTIF_ERROR_CDI_SET_SENSOR_CTRL_FAILURE:
      return "CDI_SET_SENSOR_CTRL_FAILURE";
    case NotifType::NOTIF_ERROR_DESERIALIZER_FAILURE:
      return "DESERIALIZER_FAILURE";
    case NotifType::NOTIF_ERROR_SERIALIZER_FAILURE:
      return "SERIALIZER_FAILURE";
    case NotifType::NOTIF_ERROR_SENSOR_FAILURE:
      return "SENSOR_FAILURE";
    case NotifType::NOTIF_ERROR_ISP_PROCESSING_FAILURE_RECOVERABLE:
      return "ISP_PROCESSING_FAILURE_RECOVERABLE";
    case NotifType::NOTIF_ERROR_ICP_AUTH_FAILURE:
      return "ICP_AUTH_FAILURE";
    case NotifType::NOTIF_ERROR_ICP_AUTH_OUT_OF_ORDER:
      return "ICP_AUTH_OUT_OF_ORDER";
    case NotifType::NOTIF_ERROR_ACP_SETTINGS_DISCONTINUITY:
      return "ACP_SETTINGS_DISCONTINUITY";
    case NotifType::NOTIF_ERROR_INTR_LOCALIZATION_FAILURE:
      return "INTR_LOCALIZATION_FAILURE";
    case NotifType::NOTIF_ERROR_INTR_LOCALIZATION_TIMEOUT:
      return "INTR_LOCALIZATION_TIMEOUT";
    case NotifType::NOTIF_ERROR_ISP_PREFENCE_TIMEOUT:
      return "ISP_PREFENCE_TIMEOUT";
    case NotifType::NOTIF_INIT_ERROR_FAILURE:
      return "INIT_ERROR_FAILURE";
    case NotifType::NOTIF_ERROR_INTR_RECOVERY:
      return "INTR_RECOVERY";
    case NotifType::NOTIF_WARN_DEVICE_BLOCK_ERROR_INTR_DETECTED:
      return "DEVICE_BLOCK_ERROR_INTR_DETECTED";
    case NotifType::NOTIF_ERROR_POWER_LOAD_SWITCH_FAILURE:
      return "POWER_LOAD_SWITCH_FAILURE";
    default:
      return "EVENT_" + std::to_string(event_code);
  }
}

std::string resolvePackageUrl(const std::string & url)
{
  constexpr const char * kPackagePrefix = "package://";
  constexpr const char * kFilePrefix = "file://";
  if (url.rfind(kFilePrefix, 0) == 0U) {
    return url.substr(std::strlen(kFilePrefix));
  }
  if (url.rfind(kPackagePrefix, 0) != 0U) {
    return url;
  }
  const std::string remainder = url.substr(std::strlen(kPackagePrefix));
  const size_t slash = remainder.find('/');
  if (slash == std::string::npos) {
    throw std::runtime_error("Malformed package:// URL (no path component): " + url);
  }
  const std::string package = remainder.substr(0, slash);
  const std::string relative = remainder.substr(slash + 1);
  return ament_index_cpp::get_package_share_directory(package) + "/" + relative;
}

}  // namespace

SiplCameraNode::SiplCameraNode(const rclcpp::NodeOptions & options)
: SiplCameraNode(options, "sipl_camera")
{
  // Mono camera constructor initializes immediately.
  initialize(describePipelines());
}

SiplCameraNode::SiplCameraNode(
  const rclcpp::NodeOptions & options, const std::string & node_name)
: rclcpp::Node(node_name, options),
  sipl_camera_(nullptr),
  sipl_query_(nullptr),
  sci_buf_module_(nullptr),
  stop_capture_(false),
  output_qos_(::isaac_ros::common::AddQosParameter(*this, "DEFAULT", "output_qos", 1)),
  sci_sync_module_(nullptr),
  tsc_correlator_(this->get_clock(), this->get_logger())
{
  tsc_correlator_.setRecalibrationInterval(std::chrono::milliseconds{1000});

  // ---------------------------------------------------------------------------
  // Common parameters
  // ---------------------------------------------------------------------------
  {
    rcl_interfaces::msg::ParameterDescriptor d;
    d.description =
      "SIPL platform configuration name. Selects which camera plugin's "
      "platform-config the SIPL query API loads. Required. "
      "Examples: \"VB1940\" (Eagle CoE), "
      "\"AR0234CS_HAWK\" (Hawk GMSL stereo).";
    try {
      platform_config_ = declare_parameter<std::string>("platform_config", d);
    } catch (const rclcpp::exceptions::UninitializedStaticallyTypedParameterException &) {
      throw std::runtime_error(
              std::string("'platform_config' is required but was not set. Pass a parameter "
              "file whose top-level key is '") + get_name() +
              ":' (ROS 2 ignores parameter files keyed to any other node name), or set "
              "-p platform_config:=<name> directly. Examples: 'VB1940' (Eagle CoE), "
              "'AR0234CS_HAWK' (Hawk GMSL stereo).");
    }
  }
  {
    rcl_interfaces::msg::ParameterDescriptor d;
    d.description =
      "Optional platform-config JSON to parse in addition to SIPL's built-in "
      "database, for camera configurations the database does not ship. Accepts a "
      "package:// URL or a filesystem path. Empty = database only.";
    platform_config_file_ = declare_parameter<std::string>("platform_config_file", "", d);
  }

  // Get debug flag from the ROS logger level so that users control
  // verbosity with the standard --ros-args --log-level <node_name>:=<level>
  enable_debug_logs_ = rcutils_logging_logger_is_enabled_for(
    get_logger().get_name(), RCUTILS_LOG_SEVERITY_DEBUG);

  enable_statistics_ = declare_parameter<bool>("enable_statistics", false);
  statistics_interval_s_ = declare_parameter<double>("statistics_interval_s", 60.0);
  constexpr double kMinStatisticsIntervalS = 0.001;
  const double max_statistics_interval_s = std::chrono::duration<double>(
    std::chrono::nanoseconds::max()).count();
  if (!std::isfinite(statistics_interval_s_) ||
    statistics_interval_s_ < kMinStatisticsIntervalS ||
    statistics_interval_s_ >= max_statistics_interval_s)
  {
    throw std::runtime_error(
      "statistics_interval_s must be finite and in [0.001, " +
      std::to_string(max_statistics_interval_s) + "); got " +
      std::to_string(statistics_interval_s_));
  }

  {
    rcl_interfaces::msg::ParameterDescriptor d;
    d.description =
      "Number of registered ICP capture buffers per pipeline. CoE requires 4; "
      "GMSL accepts 4 through 32.";
    d.read_only = true;
    const int64_t count = declare_parameter<int64_t>(
      "capture_buffer_count", static_cast<int64_t>(kDefaultCaptureBufferCount), d);
    if (count < static_cast<int64_t>(kDefaultCaptureBufferCount) ||
      count > static_cast<int64_t>(kMaxGmslCaptureBufferCount))
    {
      throw std::runtime_error(
        "capture_buffer_count must be in [" +
        std::to_string(kDefaultCaptureBufferCount) + ", " +
        std::to_string(kMaxGmslCaptureBufferCount) + "]; got " +
        std::to_string(count));
    }
    capture_buffer_count_ = static_cast<uint16_t>(count);
  }

  // Output encoding
  rcl_interfaces::msg::ParameterDescriptor encoding_desc;
  encoding_desc.name = "encoding_desired";
  encoding_desc.description = "Desired output encoding. Allowed: nv12, nv24";
  encoding_desc.additional_constraints = "Must be 'nv12' or 'nv24'";
  encoding_desc.read_only = true;
  encoding_desc.type = rcl_interfaces::msg::ParameterType::PARAMETER_STRING;
  encoding_desired_ = declare_parameter<std::string>(
    "encoding_desired", "nv12", encoding_desc);

  // Timestamp configuration
  use_hw_timestamp_ = declare_parameter<bool>("use_hw_timestamp", true);
  expected_fps_ = declare_parameter<double>("expected_fps", 30.0);
  if (!std::isfinite(expected_fps_) || expected_fps_ <= 0.0) {
    throw std::runtime_error(
      "expected_fps must be finite and > 0; got " + std::to_string(expected_fps_));
  }
  {
    rcl_interfaces::msg::ParameterDescriptor d;
    d.description =
      "Maximum allowed time from the hardware capture SOF timestamp to entry into "
      "the ROS publisher, expressed in expected frame periods. Values above this "
      "threshold emit a per-camera throttled warning. 0 disables the guard.";
    max_capture_to_publish_latency_frame_periods_ = declare_parameter<double>(
      "max_capture_to_publish_latency_frame_periods", 2.0, d);
    if (!std::isfinite(max_capture_to_publish_latency_frame_periods_) ||
      max_capture_to_publish_latency_frame_periods_ < 0.0)
    {
      throw std::runtime_error(
        "max_capture_to_publish_latency_frame_periods must be finite and non-negative; got " +
        std::to_string(max_capture_to_publish_latency_frame_periods_));
    }
  }

  // TF frame names
  camera_link_frame_name_ = declare_parameter<std::string>("camera_link_frame_name", "camera");
  optical_frame_name_ = camera_link_frame_name_ + "_optical";

  {
    rcl_interfaces::msg::ParameterDescriptor d;
    d.description =
      "Broadcast the camera frames on /tf_static. Set false when the robot's URDF "
      "already places them: a rig whose cameras sit on different kinematic branches "
      "cannot be described by the single static tree this node emits, and two "
      "publishers of the same frame make it ambiguous which parent wins.";
    publish_tf_ = declare_parameter<bool>("publish_tf", true, d);
  }

  const std::string camera_info_url =
    declare_parameter<std::string>("camera_info_url", "");
  if (!camera_info_url.empty()) {
    camera_info_ = loadCameraInfoFromFile(camera_info_url);
    camera_info_loaded_ = true;
    RCLCPP_INFO(
      get_logger(), "Loaded camera info from \"%s\"",
      camera_info_url.c_str());
  }

  // Absolute path to the NITO (ISP calibration blob) for the active camera
  // module.
  nito_file_ = declare_parameter<std::string>("nito_file", "");

  // Max time to wait for each sensor's first frame after capture starts before
  // timeout. 0 disables the check.
  first_frame_timeout_s_ = declare_parameter<double>("first_frame_timeout_s", 5.0);

  // ---------------------------------------------------------------------------
  // CoE camera transport specific parameters. Ignored when transport is GMSL.
  // ---------------------------------------------------------------------------
  {
    rcl_interfaces::msg::ParameterDescriptor d;
    d.description =
      "[CoE only] HSB instance name for logging. No runtime effect.";
    hsb_name_ = declare_parameter<std::string>("hsb_name", "hsb_0", d);
  }
  {
    rcl_interfaces::msg::ParameterDescriptor d;
    d.description =
      "[CoE only] HSB id used to match the per-HSB MAC/IP/interface override "
      "below against the platform plugin's transport list.";
    hsb_id_ = declare_parameter<int>("hsb_id", 0, d);
  }
  {
    rcl_interfaces::msg::ParameterDescriptor d;
    d.description =
      "[CoE only] Ethernet interface name on the host side, e.g. \"mgbe0_0\".";
    interface_name_ = declare_parameter<std::string>("interface_name", "mgbe0_0", d);
  }
  {
    rcl_interfaces::msg::ParameterDescriptor d;
    d.description =
      "[CoE only] Override the camera IP address. Empty = use plugin default.";
    declare_parameter<std::string>("ip_address", "", d);
  }
  {
    rcl_interfaces::msg::ParameterDescriptor d;
    d.description =
      "[CoE only] Override the camera MAC address. Empty = use plugin default.";
    declare_parameter<std::string>("mac_address", "", d);
  }

  // ---------------------------------------------------------------------------
  // GMSL camera transport specific parameters. Ignored when transport is CoE.
  // ---------------------------------------------------------------------------
  {
    rcl_interfaces::msg::ParameterDescriptor d;
    d.description =
      "[GMSL only] Bitmask of GMSL links to enable on the deserializer. Bit i "
      "set = link i enabled, so 0x0001 = link 0, 0x0002 = link 1, 0x0004 = "
      "link 2, 0x0008 = link 3. Accepts a single value, or a list holding one "
      "value per transport for a platform configuration that declares several. "
      "Default 0x0001 = link 0 only.";
    d.dynamic_typing = true;
    declare_parameter("link_mask", rclcpp::ParameterValue(static_cast<int64_t>(0x0001)), d);

    const rclcpp::Parameter link_mask_param = get_parameter("link_mask");
    std::vector<int64_t> requested;
    if (link_mask_param.get_type() == rclcpp::ParameterType::PARAMETER_INTEGER) {
      requested.push_back(link_mask_param.as_int());
    } else if (link_mask_param.get_type() == rclcpp::ParameterType::PARAMETER_INTEGER_ARRAY) {
      requested = link_mask_param.as_integer_array();
    } else {
      throw std::runtime_error(
        "link_mask must be an integer or a list of integers; got " +
        std::string(link_mask_param.get_type_name()));
    }
    if (requested.empty()) {
      throw std::runtime_error("link_mask must select at least one link; got an empty list");
    }

    link_masks_.clear();
    link_masks_.reserve(requested.size());
    // A zero mask disables one transport in a multi-deserializer configuration.
    for (const int64_t mask : requested) {
      if (mask < 0 || mask > static_cast<int64_t>(std::numeric_limits<uint16_t>::max())) {
        throw std::runtime_error(
          "link_mask entries must be >= 0 and fit in uint16; got " +
          std::to_string(mask));
      }
      link_masks_.push_back(static_cast<uint16_t>(mask));
    }
    if (std::all_of(
        link_masks_.begin(), link_masks_.end(),
        [](uint16_t m) {return m == 0U;}))
    {
      throw std::runtime_error("link_mask must select at least one link across all transports");
    }
  }
  {
    rcl_interfaces::msg::ParameterDescriptor d;
    d.description =
      "[GMSL only] Device-tree camera FSYNC group used when starting capture.";
    const int64_t fsync_group_id = declare_parameter<int64_t>("fsync_group_id", 0, d);
    if (fsync_group_id < 0 ||
      fsync_group_id > static_cast<int64_t>(std::numeric_limits<uint32_t>::max()))
    {
      throw std::runtime_error(
        "fsync_group_id must fit in uint32; got " + std::to_string(fsync_group_id));
    }
    fsync_group_id_ = static_cast<uint32_t>(fsync_group_id);
  }
  {
    rcl_interfaces::msg::ParameterDescriptor d;
    d.description =
      "[GMSL only] Restart the Tegra FSYNC group before capture so sequentially "
      "initialized sensors share a frame trigger. Set false for rigs whose frame "
      "sync is generated by the deserializer (fsync mode osc_manual), where the "
      "Tegra FSYNC block is not wired to the sensors.";
    enable_fsync_restart_ = declare_parameter<bool>("enable_fsync_restart", true, d);
  }

  // Create TF broadcaster
  tf_static_broadcaster_ = std::make_unique<tf2_ros::StaticTransformBroadcaster>(this);

  if (enable_debug_logs_) {
    nvsipl::INvSIPLTrace::GetInstance()->SetLevel(nvsipl::INvSIPLTrace::TraceLevel::LevelDebug);
  }
}


SiplCameraNode::~SiplCameraNode()
{
  statistics_stopping_.store(true);
  if (statistics_timer_) {
    statistics_timer_->cancel();
  }
  statistics_timer_.reset();

  // SIPL/NvSci teardown can deadlock in the driver when capture is unhealthy,
  // hanging shutdown indefinitely. Run it on a worker and force-exit on timeout
  // so Ctrl-C always terminates the process.
  auto teardown = std::async(
    std::launch::async, [this]() {
      stopAllPipelines();
      if (enable_statistics_) {
        logStatistics("Final", true);
      }

      // Free per-pipeline NvSciSync objects before NvSciSyncModuleClose below;
      for (auto & pipeline : pipelines_) {
        if (pipeline.sci_sync_isp0) {
          NvSciSyncObjFree(pipeline.sci_sync_isp0);
          pipeline.sci_sync_isp0 = nullptr;
        }
        pipeline.cpu_wait_context.reset();
      }

      if (sci_sync_module_) {
        NvSciSyncModuleClose(sci_sync_module_);
      }

      // Clear pipelines explicitly so NvSciBufObj's owned by SiplBufferManager are
      // freed before NvSciBufModuleClose.
      pipelines_.clear();
    });

  constexpr int kShutdownTimeoutSec = 5;
  if (teardown.wait_for(std::chrono::seconds(kShutdownTimeoutSec)) !=
    std::future_status::ready)
  {
    // rclcpp logging may be gone by now; write to stderr.
    std::fprintf(
      stderr, "[sipl_camera] teardown deadlocked; force-exiting after %ds\n",
      kShutdownTimeoutSec);
    std::fflush(nullptr);
    std::_Exit(EXIT_FAILURE);
  }

  if (owns_sipl_env_) {
    for (auto it = previous_sipl_env_.rbegin(); it != previous_sipl_env_.rend(); ++it) {
      const auto & [key, value] = *it;
      const int result = value ?
        setenv(key.c_str(), value->c_str(), 1) :
        unsetenv(key.c_str());
      if (result != 0) {
        std::fprintf(
          stderr, "[sipl_camera] failed to restore environment variable %s: %s\n",
          key.c_str(), std::strerror(errno));
      }
    }
    previous_sipl_env_.clear();
    g_sipl_env_owned.store(false);
  }
}

void SiplCameraNode::declareTimestampAlignmentParams()
{
  {
    rcl_interfaces::msg::ParameterDescriptor d;
    d.description =
      "Publish identical timestamps for hardware-synced frame pairs whose capture "
      "times are within max_timestamp_diff_us of each other: the lagging frame "
      "adopts the first-arriving partner's timestamp. Pairs farther apart keep "
      "their own timestamps and are reported as out of sync.";
    align_stereo_timestamps_ = declare_parameter<bool>("align_stereo_timestamps", false, d);
  }
  max_timestamp_diff_us_ = declare_parameter<double>("max_timestamp_diff_us", 500.0);
  if (!std::isfinite(max_timestamp_diff_us_) || max_timestamp_diff_us_ <= 0.0) {
    throw std::runtime_error(
      "max_timestamp_diff_us must be finite and > 0; got " +
      std::to_string(max_timestamp_diff_us_));
  }
}

void SiplCameraNode::prepareCaptureStart()
{
  if (!enable_fsync_restart_) {
    RCLCPP_INFO(
      get_logger(),
      "Skipping Tegra FSYNC group restart (enable_fsync_restart=false)");
    return;
  }
  // Restarting a shared FSYNC group re-times every attached sensor.
  if (pipelines_.size() < 2U) {
    return;
  }
  restartFsyncGroup(fsync_group_id_);
}

void SiplCameraNode::restartFsyncGroup(uint32_t fsync_group_id)
{
  if (transport_adapter_->type() != TransportAdapter::Type::kGmsl) {
    return;
  }

  const auto read_tsc_ticks = []() {
      uint64_t ticks;
      asm volatile ("mrs %0, cntvct_el0" : "=r" (ticks));
      return ticks;
    };

  // Restart the Tegra FSYNC source when it drives the sensors' trigger inputs.
  // Starting with an active source can leave sequentially initialized sensors
  // one frame apart; schedule a common first pulse after startup.
  const CAM_FSYNC_STATUS stop_status = cam_fsync_stop_group(fsync_group_id);
  if (stop_status != CAM_FSYNC_OK && stop_status != CAM_FSYNC_GROUP_IDLE) {
    throw std::runtime_error(
      "Failed to stop camera FSYNC group " + std::to_string(fsync_group_id) +
      " (status=" + std::to_string(static_cast<int>(stop_status)) +
      "; verify fsync_group_id and read/write access to /dev/fsync-group)");
  }

  // Give camera startup time to finish before the first pulse.
  constexpr uint64_t kFsyncStartMarginNs = 500'000'000U;
  const uint64_t start_tsc_ticks =
    read_tsc_ticks() + nanosecondsToTscTicks(kFsyncStartMarginNs);
  RCLCPP_INFO(
    get_logger(),
    "Scheduling camera FSYNC group %u first edge at TSC tick %" PRIu64 " (+500 ms)",
    fsync_group_id, start_tsc_ticks);
  const CAM_FSYNC_STATUS start_status =
    cam_fsync_program_abs_start_value(fsync_group_id, start_tsc_ticks);
  if (start_status != CAM_FSYNC_OK) {
    throw std::runtime_error(
      "Failed to schedule camera FSYNC group " + std::to_string(fsync_group_id) +
      " (status=" + std::to_string(static_cast<int>(start_status)) + ")");
  }
}

void SiplCameraNode::setupSiplCamera(const std::vector<PipelineDescriptor> & descriptors)
{
  const size_t num_sensors = descriptors.size();
  const bool explicit_ids = !descriptors.empty() && std::all_of(
    descriptors.begin(), descriptors.end(),
    [](const PipelineDescriptor & d) {return d.sensor_id_explicit;});

  RCLCPP_INFO(get_logger(), "=== SIPL Camera Setup Begin ===");
  // Get SIPL Query instance and load platform configuration
  sipl_query_ = nvsipl::INvSIPLCameraQuery::GetInstance();
  if (!sipl_query_) {
    throw std::runtime_error("Failed to get SIPL Query instance");
  }

  RCLCPP_INFO(get_logger(), "  Parsing SIPL database...");
  nvsipl::SIPLStatus status = sipl_query_->ParseDatabase();
  if (status != nvsipl::NVSIPL_STATUS_OK) {
    RCLCPP_FATAL(
      get_logger(), "  ParseDatabase failed with status: %u",
      static_cast<uint32_t>(status));
    throw std::runtime_error("Failed to parse SIPL database");
  }

  if (!platform_config_file_.empty()) {
    const std::string path = resolvePackageUrl(platform_config_file_);
    RCLCPP_INFO(get_logger(), "  Parsing platform config file: %s", path.c_str());
    status = sipl_query_->ParseJsonFile(path);
    if (status != nvsipl::NVSIPL_STATUS_OK) {
      RCLCPP_FATAL(get_logger(), "  ParseJsonFile('%s') failed with status: %u",
        path.c_str(), static_cast<uint32_t>(status));
      throw std::runtime_error("Failed to parse platform config file: " + path);
    }
  }

  RCLCPP_INFO(get_logger(), "  Getting sensor system config for: %s", platform_config_.c_str());
  status = sipl_query_->GetSensorSystemConfig(platform_config_, sensor_system_config_);
  if (status != nvsipl::NVSIPL_STATUS_OK) {
    RCLCPP_FATAL(
      get_logger(),
      "  GetSensorSystemConfig failed with status: %u for platform: %s",
      static_cast<uint32_t>(status), platform_config_.c_str());
    const auto & available = sipl_query_->GetCameraConfigNames();
    for (const auto & name : available) {
      RCLCPP_FATAL(get_logger(), "    available config: %s", name.c_str());
    }
    throw std::runtime_error("Sensor system config not found");
  }

  const bool is_gmsl = configIsGmsl(sensor_system_config_);
  if (!is_gmsl && capture_buffer_count_ != kDefaultCaptureBufferCount) {
    throw std::runtime_error(
      "capture_buffer_count must be 4 for non-GMSL platform config '" +
      platform_config_ + "'; got " + std::to_string(capture_buffer_count_));
  }

  const config::SiplEnvOverrides sipl_env =
    config::overrideYuvInputFormats(sensor_system_config_, {}, get_logger());
  if (g_sipl_env_owned.exchange(true)) {
    throw std::runtime_error(
            "Another SIPL camera node already configured this process. SIPL "
            "allows one owner per process and the SIPL environment overrides "
            "are process-global, so run additional cameras in their own "
            "container, or use SiplMultiCameraNode for a shared deserializer.");
  }
  owns_sipl_env_ = true;
  previous_sipl_env_.reserve(sipl_env.size());
  for (const auto & [key, value] : sipl_env) {
    (void)value;
    const char * previous = std::getenv(key.c_str());
    previous_sipl_env_.emplace_back(
      key, previous == nullptr ?
      std::nullopt : std::optional<std::string>{previous});
  }
  config::applySiplEnvOverrides(sipl_env, get_logger());

  // JSON configurations use the query library mask; built-in configurations
  // retain the transport adapter filtering validated for Hawk.
  const bool mask_via_sipl = !platform_config_file_.empty() && is_gmsl;
  if (mask_via_sipl) {
    if (link_masks_.size() != sensor_system_config_.transports.size()) {
      throw std::runtime_error(
        "link_mask needs one entry per transport: platform config '" + platform_config_ +
        "' declares " + std::to_string(sensor_system_config_.transports.size()) +
        " transport(s) but link_mask has " + std::to_string(link_masks_.size()) + ".");
    }
    std::vector<uint32_t> enable_masks;
    enable_masks.reserve(link_masks_.size());
    for (const uint16_t link_mask : link_masks_) {
      enable_masks.push_back(toSiplEnableMask(link_mask));
    }
    RCLCPP_INFO(get_logger(), "  Applying %zu link mask(s)", enable_masks.size());
    status = sipl_query_->ApplyMask(sensor_system_config_, enable_masks);
    if (status != nvsipl::NVSIPL_STATUS_OK) {
      RCLCPP_FATAL(get_logger(), "  ApplyMask failed with status: %u",
        static_cast<uint32_t>(status));
      throw std::runtime_error("Failed to apply link_mask");
    }
  } else if (link_masks_.size() > 1U) {
    throw std::runtime_error(
      "link_mask takes a single value for platform config '" + platform_config_ +
      "'; a list applies only to a multi-transport configuration loaded from "
      "platform_config_file.");
  }

  const auto duplicate_ids = config::findDuplicateSensorIds(sensor_system_config_);
  if (!duplicate_ids.empty()) {
    std::string ids;
    for (const uint32_t id : duplicate_ids) {
      ids += (ids.empty() ? "" : ", ") + std::to_string(id);
    }
    throw std::runtime_error(
      "Platform config '" + platform_config_ + "' contains duplicate sensor id(s): " +
      ids + ". This usually means the config declares the same modules on more than "
      "one transport; narrow it with the link_mask parameter.");
  }

  CoeOverrides coe_overrides;
  coe_overrides.hsb_id = static_cast<uint32_t>(hsb_id_);
  if (!interface_name_.empty()) {coe_overrides.interface_name = interface_name_;}
  const std::string ip_param = get_parameter("ip_address").as_string();
  const std::string mac_param = get_parameter("mac_address").as_string();
  if (!ip_param.empty()) {coe_overrides.ip_address = ip_param;}
  if (!mac_param.empty()) {coe_overrides.mac_address = mac_param;}

  RCLCPP_INFO(get_logger(), "Applying transport overrides...");
  transport_adapter_ = TransportAdapter::Create(
    sensor_system_config_, coe_overrides, link_masks_.front(), get_logger(),
    /*erase_unmasked_modules=*/!mask_via_sipl);

  config::uniquifySensorGroups(sensor_system_config_, get_logger());

  RCLCPP_INFO(
    get_logger(), "Loaded platform config: %s with %zu module(s)",
    platform_config_.c_str(), sensor_system_config_.modules.size());
  for (size_t mi = 0; mi < sensor_system_config_.modules.size(); ++mi) {
    const auto & m = sensor_system_config_.modules[mi];
    RCLCPP_INFO(
      get_logger(), "    Module[%zu]: name=%s transportId=%u",
      mi, m.name.c_str(), m.transportId);
  }

  // A single module may expose multiple sensors (e.g. a Hawk GMSL module is one
  // dual-sensor module that provides the whole stereo pair), so reason about
  // sensor counts per module rather than treating module count as sensor count.
  const auto module_sensor_count =
    [](const nvsipl::sensorconfig::ModuleConfig & module) -> size_t {
      if (std::holds_alternative<nvsipl::sensorconfig::GmslModule>(module.moduleType)) {
        return std::get<nvsipl::sensorconfig::GmslModule>(module.moduleType).sensorConfigs.size();
      } else if (std::holds_alternative<nvsipl::sensorconfig::CoEModule>(module.moduleType)) {
        return std::get<nvsipl::sensorconfig::CoEModule>(module.moduleType).sensorConfigs.size();
      }
      return 0;
    };

  // Fail fast when the platform config cannot supply enough sensors for the
  // requested pipelines.
  size_t available_sensors = 0;
  for (const auto & module : sensor_system_config_.modules) {
    available_sensors += module_sensor_count(module);
  }
  if (!explicit_ids && available_sensors < num_sensors) {
    throw std::runtime_error(
            "Platform config '" + platform_config_ + "' exposes " +
            std::to_string(available_sensors) + " sensor(s) but " +
            std::to_string(num_sensors) + " sensor(s) were requested; check that the "
            "platform config matches the camera setup");
  }

  // Drop trailing modules SIPL would otherwise expect us to register. For example
  // the VB1940 "Dual Sensor" config exposes two single-sensor modules, but to
  // publish a single sensor (mono) we register only one pipeline. Keep just
  // enough leading modules to cover num_sensors, accumulating sensors per module
  // so multi-sensor modules (e.g. a Hawk dual-sensor module) are never split.
  //
  // Meaningless once sensor ids are named explicitly: keeping the *first* N
  // modules could delete the very module that owns a requested id.
  if (explicit_ids) {
    RCLCPP_DEBUG(
      get_logger(), "Explicit sensor ids requested; skipping module trimming");
  } else {
    size_t kept_modules = 0;
    size_t kept_sensors = 0;
    for (const auto & module : sensor_system_config_.modules) {
      if (kept_sensors >= num_sensors) {break;}
      kept_sensors += module_sensor_count(module);
      ++kept_modules;
    }
    if (kept_modules < sensor_system_config_.modules.size()) {
      RCLCPP_INFO(
        get_logger(),
        "Trimming SIPL modules from %zu to %zu to match registered pipelines",
        sensor_system_config_.modules.size(), kept_modules);
      sensor_system_config_.modules.resize(kept_modules);
    }
  }

  // Report missing permissions before SetPlatformCfg returns a generic error.
  config::preflightGroupMembership();

  // Get camera instance
  sipl_camera_ = nvsipl::INvSIPLCamera::GetInstance();
  if (!sipl_camera_) {
    throw std::runtime_error("Failed to get SIPL Camera instance");
  }

  status = sipl_camera_->SetPlatformCfg(sensor_system_config_);
  if (status != nvsipl::NVSIPL_STATUS_OK) {
    RCLCPP_FATAL(
      get_logger(), "  SetPlatformCfg failed with status: %u",
      static_cast<uint32_t>(status));
    throw std::runtime_error("Failed to set platform configuration");
  }

  RCLCPP_INFO(get_logger(), "SIPL Camera platform setup complete");
}

// Walk the nested SensorSystemConfig looking for a sensor with the given SIPL
// global id. CoE and GMSL sensor configs share CommonSensorConfig.
const nvsipl::sensorconfig::CommonSensorConfig &
SiplCameraNode::getSensorConfig(uint32_t sensor_id) const
{
  for (const auto & module : sensor_system_config_.modules) {
    if (std::holds_alternative<nvsipl::sensorconfig::GmslModule>(module.moduleType)) {
      const auto & gm = std::get<nvsipl::sensorconfig::GmslModule>(module.moduleType);
      for (const auto & sv : gm.sensorConfigs) {
        const auto & s =
          std::get<nvsipl::sensorconfig::GmslCameraSensorConfig>(sv);
        if (s.id == sensor_id) {return s;}
      }
    } else if (std::holds_alternative<nvsipl::sensorconfig::CoEModule>(module.moduleType)) {
      const auto & cm = std::get<nvsipl::sensorconfig::CoEModule>(module.moduleType);
      for (const auto & sv : cm.sensorConfigs) {
        const auto & s =
          std::get<nvsipl::sensorconfig::CoECameraSensorConfig>(sv);
        if (s.id == sensor_id) {return s;}
      }
    }
  }
  throw std::runtime_error(
          "getSensorConfig: no sensor with id=" + std::to_string(sensor_id));
}

std::vector<SiplCameraNode::PipelineDescriptor> SiplCameraNode::describePipelines()
{
  return {{
    "", 0,
    camera_link_frame_name_ + "_optical",
    camera_info_, camera_info_loaded_
  }};
}

std::vector<SiplCameraNode::PipelineDescriptor>
SiplCameraNode::resolvePipelineDescriptors(std::vector<PipelineDescriptor> requested_descriptors)
{
  const size_t explicit_count = std::count_if(
    requested_descriptors.begin(), requested_descriptors.end(),
    [](const PipelineDescriptor & d) {return d.sensor_id_explicit;});
  if (explicit_count == requested_descriptors.size() && explicit_count > 0U) {
    for (const auto & desc : requested_descriptors) {
      (void)getSensorConfig(desc.sensor_id);
      RCLCPP_INFO(
        get_logger(), "[%s] Using explicit SIPL sensor_id=%u",
        desc.display_name(), desc.sensor_id);
    }
    return requested_descriptors;
  }
  if (explicit_count != 0U) {
    throw std::runtime_error(
      "All pipelines must either specify an explicit sensor_id or none of them "
      "may; got " + std::to_string(explicit_count) + " of " +
      std::to_string(requested_descriptors.size()));
  }

  const bool stereo = requested_descriptors.size() == 2U;
  const auto sensor_ids =
    transport_adapter_->selectSensors(sensor_system_config_, stereo);
  if (sensor_ids.size() != requested_descriptors.size()) {
    throw std::runtime_error(
            "Requested " + std::to_string(requested_descriptors.size()) +
            " SIPL pipeline(s), but platform_config '" + platform_config_ +
            "' produced " + std::to_string(sensor_ids.size()) +
            " usable sensor selection(s)");
  }

  for (size_t i = 0; i < requested_descriptors.size(); ++i) {
    requested_descriptors[i].sensor_id = sensor_ids[i];
    RCLCPP_INFO(
      get_logger(),
      "[%s] Using SIPL sensor_id=%u",
      requested_descriptors[i].display_name(),
      sensor_ids[i]);
  }
  return requested_descriptors;
}

void SiplCameraNode::resolvePipelineOutputs(CameraPipeline & pipeline)
{
  const auto & sensor = getSensorConfig(pipeline.sensor_id);
  if (sensor.vcInfoList.empty()) {
    throw std::runtime_error(
      "[" + std::string(pipeline.desc.display_name()) +
      "] Sensor " + std::to_string(pipeline.sensor_id) + " has no virtual channels");
  }
  pipeline.image_width = sensor.vcInfoList[0].resolution.width;
  pipeline.image_height = sensor.vcInfoList[0].resolution.height;
  pipeline.encoding =
    pipeline.desc.encoding.empty() ? encoding_desired_ : pipeline.desc.encoding;

  const bool is_yuv422 = config::sensorIsYuv422(sensor);
  if (is_yuv422 &&
    ((pipeline.image_width & 1U) != 0U || (pipeline.image_height & 1U) != 0U))
  {
    throw std::runtime_error(
      "[" + std::string(pipeline.desc.display_name()) + "] YUV422 dimensions must be even "
      "for NV12 conversion; got " + std::to_string(pipeline.image_width) + "x" +
      std::to_string(pipeline.image_height));
  }
  bool publish_from_capture = is_yuv422;
  switch (pipeline.desc.requested_output) {
    case PipelineDescriptor::Output::kAuto: break;
    case PipelineDescriptor::Output::kIcp: publish_from_capture = true; break;
    case PipelineDescriptor::Output::kIsp0: publish_from_capture = false; break;
  }
  pipeline.publish_from_capture = publish_from_capture;
  pipeline.capture_enabled = publish_from_capture || pipeline.desc.raw_output;

  if (publish_from_capture && pipeline.encoding != "nv12") {
    throw std::runtime_error(
      "[" + std::string(pipeline.desc.display_name()) + "] YUV422 capture pipelines "
      "support encoding 'nv12' only (got '" + pipeline.encoding + "'). nv24 has no "
      "GPU downstream consumer and NPP has no 4:2:2->4:4:4 conversion.");
  }

  RCLCPP_INFO(
    get_logger(), "[%s] sensor %u is %s -> publishing from %s",
    pipeline.desc.display_name(), pipeline.sensor_id,
    is_yuv422 ? "YUV422" : "RAW",
    publish_from_capture ? "ICP (ISP bypassed)" : "ISP0");

  if (pipeline.desc.camera_info_loaded) {
    pipeline.camera_info_msg = pipeline.desc.camera_info;
  } else {
    pipeline.camera_info_msg = sensor_msgs::msg::CameraInfo{};
    pipeline.camera_info_msg.width = pipeline.image_width;
    pipeline.camera_info_msg.height = pipeline.image_height;
  }
}

void SiplCameraNode::buildPipelineConfig(CameraPipeline & pipeline)
{
  pipeline.cfg = nvsipl::NvSIPLPipelineConfiguration{
    .captureOutputRequested = pipeline.capture_enabled,
    .isp0OutputRequested = !pipeline.publish_from_capture,
    .isp1OutputRequested = false,
    .isp2OutputRequested = false,
    .disableSubframe = true,
    .bufferCfg = {
      .maxCaptureBufferCount = capture_buffer_count_,
      .maxIsp0BufferCount = 64U,
      .maxIsp1BufferCount = 64U,
      .maxIsp2BufferCount = 64U,
    }
  };
}

void SiplCameraNode::buildTimestampAligners()
{
  if (!align_stereo_timestamps_) {
    return;
  }
  if (!use_hw_timestamp_) {
    throw std::runtime_error(
      "align_stereo_timestamps requires use_hw_timestamp:=true (raw TSC tick domain)");
  }

  auto pairs = stereo_pair_names_;
  if (pairs.empty()) {
    if (pipelines_.size() != 2) {
      throw std::runtime_error(
        "align_stereo_timestamps requires exactly 2 pipelines (stereo) unless "
        "stereo_pairs names them; got " + std::to_string(pipelines_.size()));
    }
    pairs.emplace_back(pipelines_[0].desc.name, pipelines_[1].desc.name);
  }

  const int64_t max_diff_ns = static_cast<int64_t>(max_timestamp_diff_us_ * 1000.0 + 0.5);
  const int64_t half_period_ns = static_cast<int64_t>(0.5e9 / expected_fps_ + 0.5);

  const auto find_pipeline = [this](const std::string & name) -> CameraPipeline * {
      for (auto & pipeline : pipelines_) {
        if (pipeline.desc.name == name) {return &pipeline;}
      }
      return nullptr;
    };

  timestamp_aligners_.reserve(pairs.size());
  for (const auto & [left_name, right_name] : pairs) {
    CameraPipeline * left = find_pipeline(left_name);
    CameraPipeline * right = find_pipeline(right_name);
    if (left == nullptr || right == nullptr) {
      throw std::runtime_error(
        "stereo_pairs names unknown camera(s): '" + left_name + "', '" + right_name + "'");
    }
    if (left == right) {
      throw std::runtime_error(
        "stereo_pairs entry pairs camera '" + left_name + "' with itself");
    }
    if (left->aligner != nullptr || right->aligner != nullptr) {
      throw std::runtime_error(
        "stereo_pairs assigns camera '" + left_name + "' or '" + right_name +
        "' to more than one pair");
    }

    timestamp_aligners_.push_back(
      std::make_unique<StereoTimestampAligner>(max_diff_ns, half_period_ns));
    left->aligner = timestamp_aligners_.back().get();
    right->aligner = timestamp_aligners_.back().get();

    RCLCPP_INFO(get_logger(),
      "Timestamp alignment enabled for (%s, %s): max_diff=%.1f us, half_period=%.2f ms",
      left->desc.display_name(), right->desc.display_name(),
      max_timestamp_diff_us_, half_period_ns / 1.0e6);
  }
}

void SiplCameraNode::initialize(std::vector<PipelineDescriptor> descriptors)
{
  setupSiplCamera(descriptors);
  descriptors = resolvePipelineDescriptors(std::move(descriptors));

  RCLCPP_INFO(get_logger(), "Configuring pipelines for %zu sensor(s)", descriptors.size());

  // Phase 1: Configure all pipelines (must complete before Init())
  pipelines_.reserve(descriptors.size());
  for (auto & desc : descriptors) {
    pipelines_.emplace_back();
    CameraPipeline & pipeline = pipelines_.back();
    pipeline.desc = std::move(desc);
    // PipelineDescriptor.sensor_id is the SIPL global sensor id (resolved by
    // the transport adapter's selectSensors()).
    pipeline.sensor_id = pipeline.desc.sensor_id;
    pipeline.cuda_stream = ::nvidia::isaac_ros::common::createCudaStream(
      std::string("SiplCameraNode:") + pipeline.desc.display_name());

    resolvePipelineOutputs(pipeline);
    buildPipelineConfig(pipeline);

    RCLCPP_INFO(get_logger(), "[%s] Configuring pipeline (sensor_id=%u, %ux%u, %s)",
      pipeline.desc.display_name(), pipeline.sensor_id,
      pipeline.image_width, pipeline.image_height, pipeline.encoding.c_str());

    auto status = sipl_camera_->SetPipelineCfg(
      pipeline.sensor_id, pipeline.cfg, pipeline.queues);
    if (status != nvsipl::NVSIPL_STATUS_OK) {
      throw std::runtime_error(
              "Failed to set pipeline configuration for sensor " +
              std::to_string(pipeline.sensor_id));
    }
  }

  buildTimestampAligners();

  // Initialize NvSci modules (must precede buffer allocation: NvSciBufModule
  // is required for NvSciBufObjAlloc calls inside allocateBuffersForPipeline).
  NvSciError sci_err = NvSciSyncModuleOpen(&sci_sync_module_);
  if (sci_err != NvSciError_Success) {
    RCLCPP_FATAL(get_logger(), "Failed to open NvSciSync module: %d", sci_err);
    throw std::runtime_error("NvSciSync module initialization failed");
  }

  NvSciBufModule module_handle = nullptr;
  sci_err = NvSciBufModuleOpen(&module_handle);
  if (sci_err != NvSciError_Success) {
    RCLCPP_FATAL(get_logger(), "Failed to open NvSciBuf module: %d", sci_err);
    throw std::runtime_error("NvSciBuf module initialization failed");
  }
  sci_buf_module_.reset(
    module_handle, [](NvSciBufModuleRec * p) {
      if (p) {
        NvSciBufModuleClose(p);
      }
    });

  // SIPL HAL ordering: GetImageAttributes / NvSciBufObjAlloc must run before
  // Init(); RegisterImages and RegisterAutoControlPlugin must run after.
  // Out-of-order calls return NVSIPL_STATUS_INVALID_STATE (status 6).
  for (auto & pipeline : pipelines_) {
    NvSciSyncCpuWaitContext cpu_wait_context = nullptr;
    sci_err = NvSciSyncCpuWaitContextAlloc(sci_sync_module_, &cpu_wait_context);
    if (sci_err != NvSciError_Success) {
      RCLCPP_FATAL(
        get_logger(), "[%s] Failed to allocate CPU wait context: %d",
        pipeline.desc.display_name(), sci_err);
      throw std::runtime_error("CPU wait context allocation failed");
    }
    pipeline.cpu_wait_context.reset(cpu_wait_context);

    RCLCPP_INFO(
      get_logger(), "[%s] Allocating buffers...",
      pipeline.desc.display_name());
    allocateBuffersForPipeline(pipeline);

    // NvSciBuf reconciliation may introduce two types of padding depending on
    // the sensor resolution and color format.
    // (see nvscibuf.h NvSciBufImageAttrKey_PlanePitch and PlaneOffset):
    //
    // 1. INTER-PLANE GAP: PlaneOffset[1] > PlanePitch[0] * PlaneHeight[0].
    //    NvSciBuf aligns each plane's base address to satisfy the maximum
    //    start address alignment constraint of all HW engines accessing the
    //    buffer (NvSciBufImageAttrKey_PlaneBaseAddrAlign).
    //
    // 2. PER-ROW STRIDE PADDING: PlanePitch > PlaneWidth * (bpp / 8).
    //    Per nvscibuf.h, the pitch is first computed from width and color
    //    format, then "aligned to the maximum of the pitch alignment constraint
    //    value of all the HW engines that are going to operate on the buffer
    //    using extra padding bytes."
    //
    // Capture pipelines convert packed 4:2:2 into a fully packed NV12 block, so
    // their pool size follows from the published geometry, not an ISP layout.
    if (pipeline.publish_from_capture) {
      // All buffers share the reconciled layout; reject unsupported formats at startup.
      BufferAttributes capture_attrs{};
      auto attr_status =
        pipeline.buffer_manager_icp->queryAllocatedBufferAttributes(capture_attrs);
      if (attr_status != nvsipl::NVSIPL_STATUS_OK) {
        throw std::runtime_error(
          "[" + std::string(pipeline.desc.display_name()) +
          "] Failed to query reconciled capture buffer attributes");
      }
      if (!validateCaptureBufferFormat(capture_attrs, pipeline)) {
        throw std::runtime_error(
          "[" + std::string(pipeline.desc.display_name()) +
          "] Capture buffer format is not a supported packed 8-bit 4:2:2 surface; "
          "see the preceding error for the specific mismatch");
      }

      pipeline.needs_compaction = false;
      pipeline.compact_frame_bytes =
        static_cast<size_t>(pipeline.image_width) * pipeline.image_height * 3U / 2U;
      RCLCPP_INFO(get_logger(),
        "[%s] Capture pipeline: %s %ux%u pitch=%u -> nv12 %zu bytes",
        pipeline.desc.display_name(), toString(pipeline.packed_yuv_order),
        pipeline.image_width, pipeline.image_height,
        capture_attrs.plane_pitches[0], pipeline.compact_frame_bytes);
      continue;
    }

    // Probe the buffer layout once at init (all buffers in a pool share
    // the same NvSciBufAttrList) to decide whether gap removal is needed.
    // Size the buffer from reconciled plane pitches × heights.
    BufferAttributes init_attrs{};
    auto attr_status = pipeline.buffer_manager_isp0->queryAllocatedBufferAttributes(init_attrs);
    if (attr_status != nvsipl::NVSIPL_STATUS_OK || init_attrs.plane_count < 2) {
      throw std::runtime_error(
              "[" + std::string(pipeline.desc.display_name()) +
              "] Failed to query reconciled ISP0 buffer attributes; cannot size compact pool");
    }

    const size_t y_end =
      static_cast<size_t>(init_attrs.plane_pitches[0]) * init_attrs.plane_heights[0];
    pipeline.needs_compaction = (init_attrs.plane_offsets[1] != y_end);
    RCLCPP_DEBUG(
      get_logger(),
      "[%s] ISP0 buffer layout: Y ends at %zu, UV starts at %" PRIu64 " — %s",
      pipeline.desc.display_name(), y_end, init_attrs.plane_offsets[1],
      pipeline.needs_compaction ? "compaction required" : "already compact");

    const size_t y_plane_bytes =
      static_cast<size_t>(init_attrs.plane_pitches[0]) * init_attrs.plane_heights[0];
    const size_t uv_plane_bytes =
      static_cast<size_t>(init_attrs.plane_pitches[1]) * init_attrs.plane_heights[1];
    pipeline.compact_frame_bytes = y_plane_bytes + uv_plane_bytes;
  }

  auto status = sipl_camera_->Init();
  if (status != nvsipl::NVSIPL_STATUS_OK) {
    RCLCPP_FATAL(
      get_logger(), "SIPL Init() failed with status: %u",
      static_cast<uint32_t>(status));
    throw std::runtime_error("Failed to initialize SIPL camera");
  }

  // Register all sensors' buffers + sync before any AC plugin is registered on the primary call.
  for (auto & pipeline : pipelines_) {
    RCLCPP_INFO(
      get_logger(), "[%s] Registering buffers + sync...",
      pipeline.desc.display_name());
    registerBuffersForPipeline(pipeline);

    RCLCPP_INFO(
      get_logger(), "[%s] Creating publishers...",
      pipeline.desc.display_name());
    createPublisherForPipeline(pipeline);
  }

  for (auto & pipeline : pipelines_) {
    registerAutoControlPlugin(pipeline);
  }

  RCLCPP_INFO(get_logger(), "Starting capture...");
  startAllPipelines();

  if (enable_statistics_) {
    const auto interval = std::chrono::duration_cast<std::chrono::nanoseconds>(
      std::chrono::duration<double>(statistics_interval_s_));
    statistics_timer_ = create_wall_timer(
      interval, [this]() {logStatistics("Periodic");});
  }

  for (const auto & pipeline : pipelines_) {
    RCLCPP_INFO(
      get_logger(), "SIPL Camera initialized: %s [%s] %ux%u %s",
      platform_config_.c_str(), pipeline.desc.display_name(),
      pipeline.image_width, pipeline.image_height, pipeline.encoding.c_str());
  }
}


void SiplCameraNode::allocateBuffersForPipeline(CameraPipeline & pipeline)
{
  // Allocate (but don't yet register with SIPL) ICP + ISP0 NvSciBufObj pools.
  // Caller must invoke registerBuffersForPipeline() after sipl_camera_->Init().
  pipeline.buffer_manager_icp = std::make_shared<SiplBufferManager>(
    sci_buf_module_, capture_buffer_count_, get_logger(), enable_debug_logs_);

  auto status = pipeline.buffer_manager_icp->allocateBuffers(
    sipl_camera_.get(),
    pipeline.sensor_id,
    nvsipl::INvSIPLClient::ConsumerDesc::OutputType::ICP);
  if (status != nvsipl::NVSIPL_STATUS_OK) {
    throw std::runtime_error(
            "[" + std::string(pipeline.desc.display_name()) +
            "] Failed to allocate ICP buffers");
  }

  if (pipeline.publish_from_capture) {
    RCLCPP_INFO(
      get_logger(), "[%s] Allocated %u ICP buffer pool (sensor %u); ISP0 bypassed",
      pipeline.desc.display_name(), capture_buffer_count_, pipeline.sensor_id);
    return;
  }

  pipeline.buffer_manager_isp0 = std::make_shared<SiplBufferManager>(
    sci_buf_module_, kIsp0BufferCount, get_logger(), enable_debug_logs_);

  const auto sample_type = getSurfSampleTypeFromEncoding(pipeline.encoding);
  RCLCPP_DEBUG(
    get_logger(), "[%s] Requested ISP0 sample type for encoding '%s'",
    pipeline.desc.display_name(), pipeline.encoding.c_str());

  status = pipeline.buffer_manager_isp0->allocateBuffers(
    sipl_camera_.get(),
    pipeline.sensor_id,
    nvsipl::INvSIPLClient::ConsumerDesc::OutputType::ISP0,
    sample_type);
  if (status != nvsipl::NVSIPL_STATUS_OK) {
    throw std::runtime_error(
            "[" + std::string(pipeline.desc.display_name()) +
            "] Failed to allocate ISP0 buffers");
  }

  RCLCPP_INFO(
    get_logger(), "[%s] Allocated %u ICP and %u ISP0 buffers (sensor %u)",
    pipeline.desc.display_name(), capture_buffer_count_, kIsp0BufferCount,
    pipeline.sensor_id);
}

void SiplCameraNode::registerBuffersForPipeline(CameraPipeline & pipeline)
{
  // ICP must be registered before ISP* (NvSIPLCamera.hpp:RegisterImages
  // precondition). RegisterNvSciSyncObj must run after RegisterImages and
  // before Start().
  auto status = pipeline.buffer_manager_icp->registerBuffers(
    sipl_camera_.get(),
    pipeline.sensor_id,
    nvsipl::INvSIPLClient::ConsumerDesc::OutputType::ICP);
  if (status != nvsipl::NVSIPL_STATUS_OK) {
    throw std::runtime_error(
            "[" + std::string(pipeline.desc.display_name()) +
            "] Failed to register ICP buffers");
  }

  if (pipeline.publish_from_capture) {
    return;
  }

  status = pipeline.buffer_manager_isp0->registerBuffers(
    sipl_camera_.get(),
    pipeline.sensor_id,
    nvsipl::INvSIPLClient::ConsumerDesc::OutputType::ISP0);
  if (status != nvsipl::NVSIPL_STATUS_OK) {
    throw std::runtime_error(
            "[" + std::string(pipeline.desc.display_name()) +
            "] Failed to register ISP0 buffers");
  }

  allocateSync(
    pipeline.sensor_id,
    nvsipl::INvSIPLClient::ConsumerDesc::OutputType::ISP0,
    pipeline.sci_sync_isp0);
}

void SiplCameraNode::createPublisherForPipeline(CameraPipeline & pipeline)
{
  const auto & name = pipeline.desc.name;
  const std::string & topic_ns =
    pipeline.desc.topic_prefix.empty() ? name : pipeline.desc.topic_prefix;
  std::string image_topic = topic_ns.empty() ? "image_raw" : topic_ns + "/image_raw";
  std::string camera_info_topic =
    topic_ns.empty() ? "camera_info" : topic_ns + "/camera_info";

  rclcpp::PublisherOptions pub_options;
  pub_options.use_intra_process_comm = rclcpp::IntraProcessSetting::Enable;
  pipeline.image_pub = create_publisher<sensor_msgs::msg::Image>(
    image_topic, output_qos_, pub_options);
  RCLCPP_DEBUG(
    get_logger(), "[%s] Publisher QoS: depth=%zu, reliability=%s",
    pipeline.desc.display_name(), output_qos_.get_rmw_qos_profile().depth,
    output_qos_.get_rmw_qos_profile().reliability == RMW_QOS_POLICY_RELIABILITY_BEST_EFFORT ?
    "best_effort" : "reliable");

  pipeline.camera_info_pub = create_publisher<sensor_msgs::msg::CameraInfo>(
    camera_info_topic, output_qos_);
}


void SiplCameraNode::startAllPipelines()
{
  RCLCPP_INFO(get_logger(), "Starting capture...");
  prepareCaptureStart();
  nvsipl::SIPLStatus status = sipl_camera_->Start();
  if (status != nvsipl::NVSIPL_STATUS_OK) {
    throw std::runtime_error("Failed to start SIPL camera");
  }
  stop_capture_ = false;
  stop_notifications_ = false;

  // Publish static transforms before spawning capture threads so the TF tree
  // is available to downstream consumers by the time the first image arrives.
  if (publish_tf_) {
    publishStaticTransforms();
  } else {
    RCLCPP_INFO(get_logger(), "publish_tf is false; the camera frames must come from the URDF");
  }

  for (auto & pipeline : pipelines_) {
    pipeline.pipeline_thread = std::thread(
      &SiplCameraNode::pipelineThread, this, std::ref(pipeline));
    pipeline.event_thread = std::thread(
      &SiplCameraNode::handleNotificationQueue, this, std::ref(pipeline));
  }
  RCLCPP_INFO(get_logger(), "Pipeline threads started for %zu pipeline(s)", pipelines_.size());
}

void SiplCameraNode::stopAllPipelines()
{
  RCLCPP_INFO(get_logger(), "Stopping capture...");
  stop_capture_ = true;

  for (auto & pipeline : pipelines_) {
    if (pipeline.pipeline_thread.joinable()) {
      pipeline.pipeline_thread.join();
    }
  }

  if (sipl_camera_) {
    RCLCPP_INFO(get_logger(), "Stopping SIPL camera...");
    sipl_camera_->Stop();
  }
  stop_notifications_ = true;

  for (auto & pipeline : pipelines_) {
    if (pipeline.event_thread.joinable()) {
      pipeline.event_thread.join();
    }
  }

  if (sipl_camera_) {
    RCLCPP_INFO(get_logger(), "Deinitializing SIPL camera...");
    sipl_camera_->Deinit();
    sipl_camera_.reset();
  }
}

void SiplCameraNode::allocateSync(
  uint32_t sensor_id,
  nvsipl::INvSIPLClient::ConsumerDesc::OutputType output_type, NvSciSyncObj & sync)
{
  struct NvSciSyncAttrListGuard
  {
    NvSciSyncAttrList list{nullptr};
    ~NvSciSyncAttrListGuard()
    {
      if (list != nullptr) {
        NvSciSyncAttrListFree(list);
      }
    }
  };

  // Create the CPU waiter attribute list.
  NvSciSyncAttrListGuard waiter_attr_list;
  auto err = NvSciSyncAttrListCreate(sci_sync_module_, &waiter_attr_list.list);
  if (err != NvSciError_Success) {
    throw std::runtime_error("Failed to create waiter NvSciSyncAttrList");
  }

  NvSciSyncAttrKeyValuePair kv[2];
  memset(kv, 0, sizeof(kv));

  bool cpu_signaler_waiter = true;
  kv[0].attrKey = NvSciSyncAttrKey_NeedCpuAccess;
  kv[0].value = reinterpret_cast<void *>(&cpu_signaler_waiter);
  kv[0].len = sizeof(cpu_signaler_waiter);

  NvSciSyncAccessPerm cpu_perm = NvSciSyncAccessPerm_WaitOnly;
  kv[1].attrKey = NvSciSyncAttrKey_RequiredPerm;
  kv[1].value = reinterpret_cast<void *>(&cpu_perm);
  kv[1].len = sizeof(cpu_perm);

  err = NvSciSyncAttrListSetAttrs(waiter_attr_list.list, kv, 2);
  if (err != NvSciError_Success) {
    throw std::runtime_error("Failed to set NvSciSyncAttrList values");
  }

  // Get the camera signaler attribute list.
  NvSciSyncAttrListGuard signaler_attr_list;
  err = NvSciSyncAttrListCreate(sci_sync_module_, &signaler_attr_list.list);
  if (err != NvSciError_Success) {
    throw std::runtime_error("Failed to create signaler NvSciSyncAttrList");
  }

  auto status = sipl_camera_->FillNvSciSyncAttrList(
    sensor_id, output_type,
    signaler_attr_list.list, nvsipl::SIPL_SIGNALER);
  if (status != nvsipl::NVSIPL_STATUS_OK) {
    throw std::runtime_error("Failure in FillNvSciSyncAttrList");
  }

  // Reconcile the attributes.
  NvSciSyncAttrList unreconciled_list[2];
  unreconciled_list[0] = waiter_attr_list.list;
  unreconciled_list[1] = signaler_attr_list.list;

  NvSciSyncAttrListGuard reconciled_attr_list;
  NvSciSyncAttrListGuard conflict_attr_list;

  err = NvSciSyncAttrListReconcile(
    unreconciled_list, 2, &reconciled_attr_list.list,
    &conflict_attr_list.list);
  if (err != NvSciError_Success) {
    throw std::runtime_error("Failed to reconcile NvSciSync attributes");
  }

  // Allocate the sync object.
  err = NvSciSyncObjAlloc(reconciled_attr_list.list, &sync);
  if (err != NvSciError_Success) {
    throw std::runtime_error("Failed to allocate NvSciSyncObj");
  }

  // Register the sync object.
  status = sipl_camera_->RegisterNvSciSyncObj(
    sensor_id, output_type, nvsipl::NVSIPL_EOFSYNCOBJ,
    sync);
  if (status != nvsipl::NVSIPL_STATUS_OK) {
    throw std::runtime_error("Failed to register NvSciSyncObj");
  }

  RCLCPP_DEBUG(
    get_logger(), "Created and registered sync object for output type %u",
    static_cast<uint32_t>(output_type));
}

void SiplCameraNode::registerAutoControlPlugin(const CameraPipeline & pipeline)
{
  if (pipeline.publish_from_capture) {
    RCLCPP_INFO(get_logger(),
      "[%s] No ISP output on sensor %u; skipping NITO / auto-control plugin",
      pipeline.desc.display_name(), pipeline.sensor_id);
    return;
  }

  // For sensor-group pairs (sensorGroup > 0, two sensors sharing a module),
  // registering on the secondary returns NVSIPL_STATUS_INVALID_STATE.
  const auto & sensor = getSensorConfig(pipeline.sensor_id);
  if (sensor.sensorGroup > 0U && sensor.deviceIndex != 0U) {
    RCLCPP_INFO(get_logger(),
      "[%s] Skipping AC plugin for secondary sensor %u of group %u",
      pipeline.desc.display_name(), pipeline.sensor_id, sensor.sensorGroup);
    return;
  }

  const std::string & nito_file =
    pipeline.desc.nito_file.empty() ? nito_file_ : pipeline.desc.nito_file;
  if (nito_file.empty()) {
    RCLCPP_FATAL(get_logger(),
      "[%s] nito_file is required for ISP0 pipelines but is empty",
      pipeline.desc.display_name());
    throw std::runtime_error("nito_file parameter is required");
  }
  std::vector<uint8_t> nito_blob = loadNitoFile(nito_file);
  if (nito_blob.empty()) {
    RCLCPP_FATAL(get_logger(), "Failed to load NITO file: %s", nito_file.c_str());
    throw std::runtime_error("Failed to load NITO file");
  }

  RCLCPP_INFO(get_logger(), "[%s] Registering Auto Control Plugin (sensor %u)",
    pipeline.desc.display_name(), pipeline.sensor_id);
  auto status = sipl_camera_->RegisterAutoControlPlugin(
    pipeline.sensor_id, nvsipl::NV_PLUGIN, nullptr, nito_blob);
  if (status != nvsipl::NVSIPL_STATUS_OK) {
    RCLCPP_FATAL(
      get_logger(),
      "RegisterAutoControlPlugin failed for sensor %u (nito_file='%s', "
      "nito_size=%zu) with SIPL status %u",
      pipeline.sensor_id, nito_file.c_str(), nito_blob.size(),
      static_cast<uint32_t>(status));
    throw std::runtime_error(
            "Failed to register autocontrol plugin (status " +
            std::to_string(static_cast<uint32_t>(status)) + ")");
  }

  RCLCPP_INFO(
    get_logger(), "Registered Auto Control Plugin for sensor %u (nito_file: %s)",
    pipeline.sensor_id, nito_file.c_str());
}


void SiplCameraNode::pipelineThread(CameraPipeline & pipeline)
{
  RCLCPP_INFO(
    get_logger(), "[%s] Pipeline thread started (sensor %u)",
    pipeline.desc.display_name(), pipeline.sensor_id);

  auto last_frame_time = std::chrono::steady_clock::now();

  // First-frame watchdog: If none arrives within first_frame_timeout_s_,
  // the camera is not connected/streaming fail hard rather than warn forever.
  const auto capture_start = std::chrono::steady_clock::now();
  bool first_frame_received = false;

  const auto steady_now_ns = []() {
      return std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
    };
  const auto begin_output_stall = [&](std::chrono::steady_clock::time_point wait_start) {
      if (!enable_statistics_) {
        return;
      }
      auto & stats = *pipeline.statistics;
      const std::lock_guard<std::mutex> lock(stats.details_mutex);
      if (stats.output_stall_start_ns == 0) {
        stats.output_stall_start_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
          wait_start.time_since_epoch()).count();
        ++stats.output_stall_episodes;
      }
    };
  const auto end_output_stall = [&]() {
      if (!enable_statistics_) {
        return;
      }
      auto & stats = *pipeline.statistics;
      const int64_t now_ns = steady_now_ns();
      const std::lock_guard<std::mutex> lock(stats.details_mutex);
      if (stats.output_stall_start_ns != 0) {
        stats.output_stall_total_ns +=
          static_cast<uint64_t>(now_ns - stats.output_stall_start_ns);
        stats.output_stall_start_ns = 0;
      }
    };

  while (rclcpp::ok() && !stop_capture_) {
    if (!first_frame_received && first_frame_timeout_s_ > 0.0) {
      const double waited_s = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - capture_start).count();
      if (waited_s > first_frame_timeout_s_) {
        RCLCPP_FATAL(
          get_logger(),
          "[%s] No frames from sensor %u within %.1fs. Verify the camera connection.",
          pipeline.desc.display_name(), pipeline.sensor_id, first_frame_timeout_s_);
        // All pipeline threads run this watchdog so only the first to fire
        // logs a node-level log and triggers shutdown so the failing sensor is known.
        if (!first_frame_timeout_reported_.exchange(true)) {
          RCLCPP_FATAL(
            get_logger(),
            "Shutting down node: sensor %u (%s) produced no frames within %.1fs.",
            pipeline.sensor_id, pipeline.desc.display_name(), first_frame_timeout_s_);
          rclcpp::shutdown();
        }
        break;
      }
    }

    if (pipeline.capture_enabled && !pipeline.queues.captureCompletionQueue) {
      RCLCPP_ERROR(get_logger(), "[%s] Capture completion queue is null",
        pipeline.desc.display_name());
      break;
    }
    if (!pipeline.publish_from_capture && !pipeline.queues.isp0CompletionQueue) {
      RCLCPP_ERROR(get_logger(), "[%s] ISP0 completion queue is null",
        pipeline.desc.display_name());
      break;
    }

    if (pipeline.capture_enabled) {
      nvsipl::INvSIPLClient::INvSIPLBuffer * buffer_raw = nullptr;
      auto capture_start_time = std::chrono::steady_clock::now();
      auto status = pipeline.queues.captureCompletionQueue->Get(buffer_raw,
        SiplCameraNode::kSiplQueueGetTimeoutUs);
      if (status != nvsipl::NVSIPL_STATUS_OK) {
        if (status == nvsipl::NVSIPL_STATUS_TIMED_OUT) {
          if (first_frame_received && !stop_capture_) {
            begin_output_stall(capture_start_time);
            RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 1000,
              "[%s] Timeout getting RAW buffer",
              pipeline.desc.display_name());
          } else {
            RCLCPP_DEBUG_THROTTLE(get_logger(), *get_clock(), 1000,
              "[%s] Waiting for the first RAW buffer",
              pipeline.desc.display_name());
          }
          continue;
        }
        if (status == nvsipl::NVSIPL_STATUS_EOF) {
          RCLCPP_INFO(get_logger(), "[%s] Capture queue EOF received",
            pipeline.desc.display_name());
          break;
        }
        RCLCPP_ERROR_THROTTLE(get_logger(), *get_clock(), 1000,
          "[%s] Failed to get RAW buffer (status=%d)",
          pipeline.desc.display_name(), static_cast<int>(status));
        if (first_frame_received && !stop_capture_) {
          begin_output_stall(capture_start_time);
        }
        continue;
      }

      if (pipeline.publish_from_capture) {
        if (buffer_raw == nullptr) {
          continue;
        }
        end_output_stall();
        first_frame_received = true;
        if (enable_statistics_) {
          ++pipeline.statistics->frames_acquired;
        }
        auto * nvmm_buffer = dynamic_cast<nvsipl::INvSIPLClient::INvSIPLNvMBuffer *>(buffer_raw);
        if (nvmm_buffer != nullptr) {
          try {
            const bool published =
              processCaptureFrame(nvmm_buffer, pipeline, capture_start_time);
            if (enable_statistics_ && !published) {
              ++pipeline.statistics->app_frame_drops;
            }
          } catch (const std::exception & e) {
            if (enable_statistics_) {
              ++pipeline.statistics->app_frame_drops;
            }
            RCLCPP_ERROR(get_logger(), "[%s] Capture frame processing failed: %s",
              pipeline.desc.display_name(), e.what());
            buffer_raw->Release();
          }
        } else {
          if (enable_statistics_) {
            ++pipeline.statistics->app_frame_drops;
          }
          RCLCPP_ERROR(get_logger(),
            "[%s] dynamic_cast to INvSIPLNvMBuffer failed for capture buffer %p",
            pipeline.desc.display_name(), static_cast<void *>(buffer_raw));
          buffer_raw->Release();
        }
        continue;
      }

      if (buffer_raw != nullptr) {
        buffer_raw->Release();
      }
    }

    // Timestamp before blocking on the ISP0 completion queue. The ISP is processing
    // (or has already processed) the frame during this Get() call. This marks the
    // start of our ISP wait so we can measure the full consumer-visible ISP latency:
    // time spent blocked in Get() + remaining DMA flush (EOF fence).
    auto isp0_start_time = std::chrono::steady_clock::now();

    // Get ISP0 processed buffer.
    nvsipl::INvSIPLClient::INvSIPLBuffer * buffer = nullptr;
    auto status =
      pipeline.queues.isp0CompletionQueue->Get(buffer, SiplCameraNode::kSiplQueueGetTimeoutUs);

    if (status == nvsipl::NVSIPL_STATUS_OK && buffer != nullptr) {
      end_output_stall();
      first_frame_received = true;
      if (enable_statistics_) {
        ++pipeline.statistics->frames_acquired;
      }
      if (enable_debug_logs_) {
        auto now = std::chrono::steady_clock::now();
        auto elapsed_us = std::chrono::duration_cast<std::chrono::microseconds>(
          now - last_frame_time).count();
        if (elapsed_us > 0) {
          double fps = 1000000.0 / static_cast<double>(elapsed_us);
          RCLCPP_DEBUG(
            get_logger(), "[%s] Instantaneous FPS: %.2f",
            pipeline.desc.display_name(), fps);
        }
        last_frame_time = now;
      }

      auto * nvmm_buffer = dynamic_cast<nvsipl::INvSIPLClient::INvSIPLNvMBuffer *>(buffer);
      if (nvmm_buffer != nullptr) {
        try {
          const bool published = processIspFrame(nvmm_buffer, pipeline, isp0_start_time);
          if (enable_statistics_ && !published) {
            ++pipeline.statistics->app_frame_drops;
          }
        } catch (const std::exception & e) {
          if (enable_statistics_) {
            ++pipeline.statistics->app_frame_drops;
          }
          RCLCPP_ERROR(
            get_logger(), "[%s] Frame processing failed: %s",
            pipeline.desc.display_name(), e.what());
          buffer->Release();
        }
      } else {
        if (enable_statistics_) {
          ++pipeline.statistics->app_frame_drops;
        }
        RCLCPP_ERROR(
          get_logger(),
          "[%s] dynamic_cast to INvSIPLNvMBuffer failed for buffer %p",
          pipeline.desc.display_name(), static_cast<void *>(buffer));
        buffer->Release();
      }

      size_t queue_depth = pipeline.queues.isp0CompletionQueue->GetCount();
      if (queue_depth > 1) {
        RCLCPP_WARN_THROTTLE(
          get_logger(), *get_clock(), 1000,
          "[%s] ISP0 queue depth high (%zu). Possible backlog.",
          pipeline.desc.display_name(), queue_depth);
      }
    } else if (status == nvsipl::NVSIPL_STATUS_TIMED_OUT) {
      if (first_frame_received && !stop_capture_) {
        begin_output_stall(isp0_start_time);
      }
      RCLCPP_DEBUG_THROTTLE(
        get_logger(), *get_clock(), 1000,
        "[%s] Waiting for frames (queue timeout, no data from sensor)",
        pipeline.desc.display_name());
    } else if (status == nvsipl::NVSIPL_STATUS_EOF) {
      RCLCPP_INFO(
        get_logger(), "[%s] ISP0 queue EOF received",
        pipeline.desc.display_name());
      break;
    } else {
      if (first_frame_received && !stop_capture_) {
        begin_output_stall(isp0_start_time);
      }
      RCLCPP_WARN(
        get_logger(), "[%s] Failed to get ISP0 buffer (status=%u)",
        pipeline.desc.display_name(), static_cast<uint32_t>(status));
    }
  }

  end_output_stall();
  RCLCPP_INFO(get_logger(), "[%s] Pipeline thread exited", pipeline.desc.display_name());
}

void SiplCameraNode::handleNotificationQueue(CameraPipeline & pipeline)
{
  RCLCPP_INFO(get_logger(), "[%s] Event thread started", pipeline.desc.display_name());

  if (!pipeline.queues.notificationQueue) {
    RCLCPP_ERROR(
      get_logger(), "[%s] Notification queue is null",
      pipeline.desc.display_name());
    return;
  }

  nvsipl::NvSIPLPipelineNotifier::NotificationData event;

  while (!stop_notifications_ || pipeline.queues.notificationQueue->GetCount() > 0U) {
    nvsipl::SIPLStatus status = pipeline.queues.notificationQueue->Get(
      event, SiplCameraNode::kSiplQueueGetTimeoutUs);

    if (status == nvsipl::NVSIPL_STATUS_OK) {
      handlePipelineNotification(pipeline, event);
    } else if (status == nvsipl::NVSIPL_STATUS_TIMED_OUT) {
      // Timeout is normal
    } else if (status == nvsipl::NVSIPL_STATUS_EOF) {
      RCLCPP_INFO(
        get_logger(), "[%s] Notification queue EOF received",
        pipeline.desc.display_name());
      break;
    } else {
      RCLCPP_WARN(
        get_logger(), "[%s] Notification queue Get() failed (status=%u)",
        pipeline.desc.display_name(), static_cast<uint32_t>(status));
    }
  }

  RCLCPP_INFO(get_logger(), "[%s] Event thread exited", pipeline.desc.display_name());
}

bool SiplCameraNode::validateIsp0BufferFormat(
  const BufferAttributes & attrs,
  const CameraPipeline & pipeline) const
{
  const std::string & encoding = pipeline.encoding;
  if (encoding != "nv12" && encoding != "nv24") {
    RCLCPP_ERROR(
      get_logger(), "Unsupported encoding '%s' (expected nv12 or nv24)",
      encoding.c_str());
    return false;
  }

  if (attrs.plane_count != 2U) {
    RCLCPP_ERROR(
      get_logger(), "Unsupported plane count %u for encoding '%s' (expected 2)",
      attrs.plane_count, encoding.c_str());
    return false;
  }

  // Guard against a abnormal runtime buffer layout that does not match the layout expected when
  // initialized as the output buffer pool was sized from the reconciled plane pitches
  // and heights.
  const size_t runtime_y_bytes =
    static_cast<size_t>(attrs.plane_pitches[0]) * attrs.plane_heights[0];
  const size_t runtime_uv_bytes =
    static_cast<size_t>(attrs.plane_pitches[1]) * attrs.plane_heights[1];
  const size_t runtime_compact_bytes = runtime_y_bytes + runtime_uv_bytes;
  if (runtime_compact_bytes != pipeline.compact_frame_bytes) {
    RCLCPP_ERROR_THROTTLE(
      get_logger(), *get_clock(), 2000,
      "[%s] Runtime compact size %zu bytes differs from init-time size %zu bytes "
      "(Y pitch=%u height=%u, UV pitch=%u height=%u) — layout changed after init",
      pipeline.desc.display_name(),
      runtime_compact_bytes, pipeline.compact_frame_bytes,
      attrs.plane_pitches[0], attrs.plane_heights[0],
      attrs.plane_pitches[1], attrs.plane_heights[1]);
    return false;
  }

  const NvSciBufAttrValColorFmt expected_plane0 = NvSciColor_Y8;
  const NvSciBufAttrValColorFmt expected_plane1 = NvSciColor_V8U8;
  if (attrs.plane_color_formats[0] != expected_plane0 ||
    attrs.plane_color_formats[1] != expected_plane1)
  {
    RCLCPP_ERROR(
      get_logger(),
      "Unexpected plane color formats for encoding '%s': plane0=%u plane1=%u "
      "(expected %u/%u)",
      encoding.c_str(),
      static_cast<uint32_t>(attrs.plane_color_formats[0]),
      static_cast<uint32_t>(attrs.plane_color_formats[1]),
      static_cast<uint32_t>(expected_plane0),
      static_cast<uint32_t>(expected_plane1));
    return false;
  }


  const uint32_t y_width = attrs.plane_widths[0];
  const uint32_t y_height = attrs.plane_heights[0];
  const uint32_t uv_width = attrs.plane_widths[1];
  const uint32_t uv_height = attrs.plane_heights[1];
  const bool looks_nv12 = (static_cast<uint64_t>(uv_height) * 2U == y_height) &&
    (static_cast<uint64_t>(uv_width) * 2U == y_width);
  const bool looks_nv24 = (uv_height == y_height) && (uv_width == y_width);
  const char * detected = looks_nv24 ? "nv24" : (looks_nv12 ? "nv12" : "unknown");

  if (enable_debug_logs_) {
    for (uint32_t i = 0; i < attrs.plane_count; ++i) {
      RCLCPP_DEBUG(
        get_logger(),
        "Plane %u: width=%u, height=%u, pitch=%u, "
        "offset=%" PRIu64 ", bpp=%u",
        i, attrs.plane_widths[i], attrs.plane_heights[i], attrs.plane_pitches[i],
        attrs.plane_offsets[i], attrs.plane_bits_per_pixels[i]);
    }
  }

  RCLCPP_DEBUG_THROTTLE(
    get_logger(), *get_clock(), 5000,
    "Detected buffer layout: y=%ux%u uv=%ux%u (detected %s, desired %s)",
    y_width, y_height, uv_width, uv_height, detected, encoding.c_str());

  if (encoding == "nv12" && !looks_nv12) {
    RCLCPP_ERROR_THROTTLE(
      get_logger(), *get_clock(), 2000,
      "Buffer layout mismatch for nv12: y=%ux%u uv=%ux%u",
      y_width, y_height, uv_width, uv_height);
    return false;
  }
  if (encoding == "nv24" && !looks_nv24) {
    RCLCPP_ERROR_THROTTLE(
      get_logger(), *get_clock(), 2000,
      "Buffer layout mismatch for nv24: y=%ux%u uv=%ux%u",
      y_width, y_height, uv_width, uv_height);
    return false;
  }

  // ROS2 sensor_msgs/Image carries a single `step` field, defined as the
  // full row length in bytes (the luma row stride):
  //   https://github.com/ros2/common_interfaces/blob/rolling/sensor_msgs/msg/Image.msg
  // For the multi-planar in ROS2 sensor_msgs/image_encodings.hpp defines those encodings purely by
  // linking out to the V4L2 spec, which is therefore the normative source
  // of the chroma layout:
  //   https://www.kernel.org/doc/html/latest/userspace-api/media/v4l/pixfmt-yuv-planar.html
  // V4L2's contiguous-plane rule:
  //   NV12/NV21 -> chroma pitch == luma pitch
  //   NV24      -> chroma pitch == 2 * luma pitch
  // nvscibuf reconciles each plane's pitch independently against its own
  // HW alignment constraint (NvSciBufImageAttrKey_PlanePitchAlign), so a
  // valid GPU layout can still violate the ROS2/V4L2 stride relation. Catch potential
  // inconsistencies.
  const uint32_t expected_uv_pitch =
    (encoding == "nv24") ? attrs.plane_pitches[0] * 2U :
    attrs.plane_pitches[0];
  if (attrs.plane_pitches[1] != expected_uv_pitch) {
    RCLCPP_ERROR_THROTTLE(
      get_logger(), *get_clock(), 2000,
      "[%s] nvscibuf UV pitch (%u) violates the V4L2 %s stride relation that "
      "ROS2 sensor_msgs/Image inherits via image_encodings.hpp; expected UV "
      "pitch %u (Y pitch=%u). Adjust NvSciBufImageAttrKey_PlanePitchAlign "
      "so the chroma pitch matches the V4L2 rule.",
      pipeline.desc.display_name(),
      attrs.plane_pitches[1], encoding.c_str(),
      expected_uv_pitch, attrs.plane_pitches[0]);
    return false;
  }

  return true;
}

bool SiplCameraNode::validateCaptureBufferFormat(
  const BufferAttributes & attrs,
  CameraPipeline & pipeline) const
{
  // ICP exposes its format through per-plane NvSciBuf attributes.
  if (attrs.plane_count != 1U) {
    RCLCPP_ERROR_THROTTLE(
      get_logger(), *get_clock(), 2000,
      "[%s] Expected 1 plane for a packed YUV422 capture buffer, got %u",
      pipeline.desc.display_name(), attrs.plane_count);
    return false;
  }

  const auto order = packedYuvOrderFromColorFormat(attrs.plane_color_formats[0]);
  if (!order) {
    RCLCPP_ERROR_THROTTLE(
      get_logger(), *get_clock(), 2000,
      "[%s] Unsupported capture color format %u; expected packed 8-bit 4:2:2 "
      "(NvSciColor_U8Y8V8Y8 / NvSciColor_Y8U8Y8V8)",
      pipeline.desc.display_name(),
      static_cast<uint32_t>(attrs.plane_color_formats[0]));
    return false;
  }
  pipeline.packed_yuv_order = *order;

  if (attrs.plane_bits_per_pixels[0] != 16U) {
    RCLCPP_ERROR_THROTTLE(
      get_logger(), *get_clock(), 2000,
      "[%s] Expected 16 bits per pixel for packed 4:2:2, got %u",
      pipeline.desc.display_name(), attrs.plane_bits_per_pixels[0]);
    return false;
  }

  if (attrs.plane_widths[0] != pipeline.image_width ||
    attrs.plane_heights[0] != pipeline.image_height)
  {
    RCLCPP_ERROR_THROTTLE(
      get_logger(), *get_clock(), 2000,
      "[%s] Capture buffer geometry %ux%u does not match the configured %ux%u",
      pipeline.desc.display_name(),
      attrs.plane_widths[0], attrs.plane_heights[0],
      pipeline.image_width, pipeline.image_height);
    return false;
  }

  if (attrs.plane_pitches[0] < static_cast<uint32_t>(pipeline.image_width) * 2U) {
    RCLCPP_ERROR_THROTTLE(
      get_logger(), *get_clock(), 2000,
      "[%s] Capture pitch %u is smaller than the minimum packed 4:2:2 row (%u bytes)",
      pipeline.desc.display_name(), attrs.plane_pitches[0], pipeline.image_width * 2U);
    return false;
  }

  const size_t min_size =
    static_cast<size_t>(attrs.plane_pitches[0]) * attrs.plane_heights[0];
  if (attrs.size < min_size) {
    RCLCPP_ERROR_THROTTLE(
      get_logger(), *get_clock(), 2000,
      "[%s] Capture buffer size %" PRIu64 " is smaller than pitch x height (%zu)",
      pipeline.desc.display_name(), attrs.size, min_size);
    return false;
  }

  return true;
}

bool SiplCameraNode::processCaptureFrame(
  nvsipl::INvSIPLClient::INvSIPLNvMBuffer * sipl_buffer,
  CameraPipeline & pipeline,
  std::chrono::steady_clock::time_point capture_start_time)
{
  const auto process_start = enable_debug_logs_ ?
    std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};

  if (sipl_buffer == nullptr) {
    RCLCPP_ERROR(get_logger(), "[%s] Null SIPL capture buffer",
      pipeline.desc.display_name());
    return false;
  }

  CudaDevicePtr gpu_ptr = nullptr;
  size_t gpu_size = 0U;
  BufferAttributes attrs{};
  auto status = pipeline.buffer_manager_icp->mapNvmmToCudaNoFence(
    sipl_buffer, &gpu_ptr, &gpu_size, &attrs);
  if (status != nvsipl::NVSIPL_STATUS_OK || gpu_ptr == nullptr || gpu_size == 0U) {
    RCLCPP_ERROR(get_logger(), "[%s] Failed to map capture NVMM to CUDA",
      pipeline.desc.display_name());
    sipl_buffer->Release();
    return false;
  }

  const auto maybe_header = buildFrameHeader(pipeline, sipl_buffer);
  if (!maybe_header) {
    sipl_buffer->Release();
    return false;
  }
  const std_msgs::msg::Header & header = *maybe_header;

  bool buffer_release_attempted = false;
  bool buffer_release_ok = false;
  const auto release_buffer = [&]() -> bool {
      if (!buffer_release_attempted) {
        buffer_release_attempted = true;
        const auto release_status = sipl_buffer->Release();
        buffer_release_ok = release_status == nvsipl::NVSIPL_STATUS_OK;
        if (!buffer_release_ok) {
          RCLCPP_FATAL(
            get_logger(), "[%s] Failed to release capture buffer (status=%u); stopping node",
            pipeline.desc.display_name(), static_cast<uint32_t>(release_status));
          rclcpp::shutdown();
        }
      }
      return buffer_release_ok;
    };
  const bool published = publishFrame(
    pipeline, header, pipeline.image_width,
    [&](uint8_t * compact_ptr, size_t /*block_size*/) -> bool {
      const NppStatus npp_status = convertPackedYuv422ToNv12(
        static_cast<const uint8_t *>(gpu_ptr), attrs.plane_pitches[0],
        compact_ptr, pipeline.image_width, pipeline.image_height,
        pipeline.packed_yuv_order, *pipeline.cuda_stream);
      // A failed conversion may still have queued work that uses the capture buffer.
      const cudaError_t cuda_err = cudaStreamSynchronize(*pipeline.cuda_stream);
      if (npp_status != NPP_SUCCESS) {
        RCLCPP_ERROR_THROTTLE(get_logger(), *get_clock(), 1000,
          "[%s] %s -> nv12 conversion failed (NppStatus %d)",
          pipeline.desc.display_name(), toString(pipeline.packed_yuv_order),
          static_cast<int>(npp_status));
        return false;
      }
      if (cuda_err != cudaSuccess) {
        RCLCPP_ERROR(get_logger(), "[%s] cudaStreamSynchronize failed: %s",
          pipeline.desc.display_name(), cudaGetErrorString(cuda_err));
        return false;
      }
      return true;
    }, release_buffer);

  if (!buffer_release_attempted && !release_buffer()) {
    return false;
  }

  if (published && enable_debug_logs_) {
    const auto process_end = std::chrono::steady_clock::now();
    RCLCPP_DEBUG(get_logger(),
      "[Process] [%s] capture %s %ux%u pitch=%u -> nv12 %zu bytes; "
      "queue-to-publish=%ld us, convert=%ld us",
      pipeline.desc.display_name(), toString(pipeline.packed_yuv_order),
      pipeline.image_width, pipeline.image_height, attrs.plane_pitches[0],
      pipeline.compact_frame_bytes,
      std::chrono::duration_cast<std::chrono::microseconds>(
        process_end - capture_start_time).count(),
      std::chrono::duration_cast<std::chrono::microseconds>(
        process_end - process_start).count());
  }
  return published;
}

std::optional<std_msgs::msg::Header> SiplCameraNode::buildFrameHeader(
  CameraPipeline & pipeline,
  nvsipl::INvSIPLClient::INvSIPLNvMBuffer * buffer,
  uint64_t * out_raw_tsc)
{
  std_msgs::msg::Header header;
  header.frame_id = pipeline.desc.frame_id;

  if (!use_hw_timestamp_) {
    header.stamp = now();
    return header;
  }

  const uint64_t raw_tsc = extractTscTimestamp(buffer);
  if (out_raw_tsc != nullptr) {
    *out_raw_tsc = raw_tsc;
  }
  if (raw_tsc == 0U) {
    RCLCPP_ERROR_THROTTLE(
      get_logger(), *get_clock(), 1000,
      "[%s] SIPL returned an irregular zero capture SOF timestamp; dropping frame. "
      "Set use_hw_timestamp:=false to use ROS clock timestamps instead.",
      pipeline.desc.display_name());
    return std::nullopt;
  }

  // Convert each frame's capture time first, then align in the ROS-time (ns) domain so the
  // lagging frame is guaranteed to use exactly the leader's already converted timestamp and
  // avoid TSC->ROS time domain recalibration influence.
  int64_t stamp_ns = tsc_correlator_.tscToRos(raw_tsc).nanoseconds();
  if (pipeline.aligner != nullptr) {
    const auto r = pipeline.aligner->align(stamp_ns);
    stamp_ns = r.ns;
    if (r.desync) {
      const uint64_t desync_count = ++stereo_desync_count_;
      ++pipeline.statistics->stereo_desyncs;
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 1000,
        "[%s] stereo frame out of sync (> %.0f us): %" PRIu64 " total; publishing "
        "unaligned timestamp (gated downstream consumers drop this pair)",
        pipeline.desc.display_name(), max_timestamp_diff_us_, desync_count);
    }
  }
  header.stamp = rclcpp::Time(stamp_ns, get_clock()->get_clock_type());
  return header;
}

bool SiplCameraNode::processIspFrame(
  nvsipl::INvSIPLClient::INvSIPLNvMBuffer * sipl_buffer,
  CameraPipeline & pipeline,
  std::chrono::steady_clock::time_point isp0_start_time)
{
  auto process_start = std::chrono::steady_clock::now();

  if (enable_debug_logs_) {
    auto handoff_us = std::chrono::duration_cast<std::chrono::microseconds>(
      process_start - isp0_start_time).count();
    RCLCPP_DEBUG(get_logger(),
      "[Process] [%s] Time from ISP start to processIspFrame entry: %ld us",
      pipeline.desc.display_name(), handoff_us);
  }

  if (sipl_buffer == nullptr) {
    RCLCPP_ERROR(get_logger(), "[%s] Null SIPL buffer", pipeline.desc.display_name());
    return false;
  }

  // Retrieve the EOF fence from the SIPL buffer. This is a lightweight metadata
  // extraction (no waiting). mapNvmmToCuda() waits on it using this pipeline's
  // CPU wait context before accessing the completed ISP output.
  NvSciSyncFence fence = NvSciSyncFenceInitializer;
  nvsipl::SIPLStatus status = sipl_buffer->GetEOFNvSciSyncFence(&fence);
  if (status != nvsipl::NVSIPL_STATUS_OK) {
    RCLCPP_ERROR(get_logger(), "[%s] Failed to get EOF fence", pipeline.desc.display_name());
    sipl_buffer->Release();
    return false;
  }

  CudaDevicePtr gpu_ptr = nullptr;
  size_t gpu_size = 0U;
  BufferAttributes attrs{};
  status = pipeline.buffer_manager_isp0->mapNvmmToCuda(
    sipl_buffer, fence, pipeline.cpu_wait_context.get(), &gpu_ptr, &gpu_size, &attrs);

  if (status != nvsipl::NVSIPL_STATUS_OK || gpu_ptr == nullptr || gpu_size == 0U) {
    RCLCPP_ERROR(
      get_logger(), "[%s] Failed to map NVMM to CUDA",
      pipeline.desc.display_name());
    sipl_buffer->Release();
    return false;
  }

  if (enable_debug_logs_) {
    // ISP EOF fence latency: time from isp0CompletionQueue->Get() in captureThread
    // (when the ISP driver delivered the buffer to us) to after mapNvmmToCuda returns
    // (when the ISP hardware has finished all DMA writes and the EOF fence was signaled).
    // This is the full consumer-visible ISP residual latency including the mailbox
    // handoff between capture and process threads.
    auto isp_fence_done = std::chrono::steady_clock::now();
    auto isp_latency_us = std::chrono::duration_cast<std::chrono::microseconds>(
      isp_fence_done - isp0_start_time).count();
    RCLCPP_DEBUG(
      get_logger(),
      "[Process] [%s] ISP start to frame ready and cpu sync'ed latency: %ld us",
      pipeline.desc.display_name(), isp_latency_us);
  }

  if (!validateIsp0BufferFormat(attrs, pipeline)) {
    sipl_buffer->Release();
    return false;
  }

  uint64_t raw_tsc = 0;
  const auto maybe_header = buildFrameHeader(pipeline, sipl_buffer, &raw_tsc);
  if (!maybe_header) {
    sipl_buffer->Release();
    return false;
  }
  const std_msgs::msg::Header & header = *maybe_header;

  // Compact the Y and UV planes into a pre-allocated pool buffer, removing
  // the hardware inter-plane padding gap to make it compatible with ROS2 sensor_msgs/Image.
  //   NV12: Y stride=pitch_Y, UV stride=pitch_UV, UV height=H/2
  //   NV24: Y stride=pitch_Y, UV stride=pitch_UV, UV height=H
  const size_t y_size = static_cast<size_t>(attrs.plane_pitches[0]) * attrs.plane_heights[0];
  const size_t uv_size = static_cast<size_t>(attrs.plane_pitches[1]) * attrs.plane_heights[1];
  const size_t compact_size = y_size + uv_size;

  const bool published = publishFrame(
    pipeline, header, attrs.plane_pitches[0],
    [&](uint8_t * compact_ptr, size_t /*block_size*/) -> bool {
      cudaError_t cuda_err;
      if (pipeline.needs_compaction) {
        // Copy Y the UV planes into the compact buffer.
        cuda_err = cudaMemcpyAsync(
          compact_ptr, gpu_ptr, y_size,
          cudaMemcpyDeviceToDevice, *pipeline.cuda_stream);
        if (cuda_err != cudaSuccess) {
          RCLCPP_ERROR(
            get_logger(), "[%s] Y plane copy failed: %s",
            pipeline.desc.display_name(), cudaGetErrorString(cuda_err));
          return false;
        }
        cuda_err = cudaMemcpyAsync(
          compact_ptr + y_size,
          static_cast<const uint8_t *>(gpu_ptr) + attrs.plane_offsets[1],
          uv_size, cudaMemcpyDeviceToDevice, *pipeline.cuda_stream);
        if (cuda_err != cudaSuccess) {
          RCLCPP_ERROR(
            get_logger(), "[%s] UV plane copy failed: %s",
            pipeline.desc.display_name(), cudaGetErrorString(cuda_err));
          return false;
        }
      } else {
        cuda_err = cudaMemcpyAsync(
          compact_ptr, gpu_ptr, compact_size,
          cudaMemcpyDeviceToDevice, *pipeline.cuda_stream);
        if (cuda_err != cudaSuccess) {
          RCLCPP_ERROR(
            get_logger(), "[%s] Frame copy failed: %s",
            pipeline.desc.display_name(), cudaGetErrorString(cuda_err));
          return false;
        }
      }
      return true;
    });

  const cudaError_t copy_error = cudaStreamSynchronize(*pipeline.cuda_stream);
  sipl_buffer->Release();
  if (copy_error != cudaSuccess) {
    if (enable_statistics_ && published) {
      ++pipeline.statistics->published_frame_errors;
    }
    RCLCPP_ERROR(
      get_logger(), "[%s] cudaStreamSynchronize failed: %s",
      pipeline.desc.display_name(), cudaGetErrorString(copy_error));
    return published;
  }

  if (!published) {
    return false;
  }

  RCLCPP_DEBUG(
    get_logger(),
    "[Process] [%s] Compacted %s: Y %u×%u + UV %u×%u, "
    "gap removed %lu bytes, compact %zu bytes",
    pipeline.desc.display_name(), pipeline.encoding.c_str(),
    attrs.plane_pitches[0], attrs.plane_heights[0],
    attrs.plane_pitches[1], attrs.plane_heights[1],
    attrs.plane_offsets[1] - y_size, compact_size);

  if (enable_debug_logs_) {
    auto process_end = std::chrono::steady_clock::now();
    auto process_duration_us = std::chrono::duration_cast<std::chrono::microseconds>(
      process_end - process_start).count();

    auto now_ros = now();
    int64_t latency_us = 0;
    if (use_hw_timestamp_) {
      auto capture_timestamp = tsc_correlator_.tscToRos(raw_tsc);
      latency_us = (now_ros.nanoseconds() - capture_timestamp.nanoseconds()) / 1000;
    }

    RCLCPP_DEBUG(
      get_logger(),
      "[Process] [%s] image_size=%zu, "
      "process_duration=%ld us, capture_to_publish_latency=%ld us, "
      "raw_tsc=%" PRIu64 ", now_ros=%" PRId64,
      pipeline.desc.display_name(), compact_size,
      process_duration_us, latency_us,
      raw_tsc, now_ros.nanoseconds());
  }
  return true;
}

void SiplCameraNode::publishCameraInfo(
  CameraPipeline & pipeline, const std_msgs::msg::Header & header)
{
  pipeline.camera_info_msg.header = header;
  pipeline.camera_info_pub->publish(pipeline.camera_info_msg);
}

void SiplCameraNode::publishStaticTransforms()
{
  geometry_msgs::msg::TransformStamped transform_stamped;
  transform_stamped.header.stamp = now();
  transform_stamped.header.frame_id = camera_link_frame_name_;
  transform_stamped.child_frame_id = optical_frame_name_;

  tf2::Quaternion q;
  kCamLinkROptical.getRotation(q);

  transform_stamped.transform.translation.x = 0.0;
  transform_stamped.transform.translation.y = 0.0;
  transform_stamped.transform.translation.z = 0.0;
  transform_stamped.transform.rotation.x = q.x();
  transform_stamped.transform.rotation.y = q.y();
  transform_stamped.transform.rotation.z = q.z();
  transform_stamped.transform.rotation.w = q.w();

  tf_static_broadcaster_->sendTransform(transform_stamped);
  RCLCPP_INFO(
    get_logger(), "Published static transform: %s -> %s",
    camera_link_frame_name_.c_str(), optical_frame_name_.c_str());
}

sensor_msgs::msg::CameraInfo SiplCameraNode::loadCameraInfoFromFile(
  const std::string & camera_info_url)
{
  std::string camera_name = camera_info_url.substr(camera_info_url.find_last_of("/\\") + 1);
  camera_name = camera_name.substr(0, camera_name.find_last_of("."));

  camera_info_manager::CameraInfoManager cinfo(this, camera_name, camera_info_url);
  if (cinfo.validateURL(camera_info_url)) {
    if (cinfo.isCalibrated()) {
      auto camera_info = sensor_msgs::msg::CameraInfo(cinfo.getCameraInfo());
      if (camera_info.p[7] != 0.0 || camera_info.p[11] != 0.0) {
        const std::string warning_msg =
          "Camera info " + camera_name +
          " has non-zero values in p[7] or p[11] which are expected to be zero."
          " See https://docs.ros2.org/latest/api/sensor_msgs/msg/CameraInfo.html";
        RCLCPP_WARN(get_logger(), "%s", warning_msg.c_str());
      }
      return camera_info;
    } else {
      const std::string error_msg =
        "Camera info " + camera_name + " not calibrated";
      RCLCPP_FATAL(get_logger(), "%s", error_msg.c_str());
      throw std::runtime_error(error_msg);
    }
  } else {
    const std::string error_msg =
      "Unable to validate camera info URL: " + camera_name;
    RCLCPP_FATAL(get_logger(), "%s", error_msg.c_str());
    throw std::runtime_error(error_msg);
  }
}


uint64_t SiplCameraNode::extractTscTimestamp(
  nvsipl::INvSIPLClient::INvSIPLNvMBuffer * buffer)
{
  const nvsipl::INvSIPLClient::ImageMetaData & metadata = buffer->GetImageData();
  // Use start-of-frame (SOF) TSC as it is more consistent relative to capture time
  // for stereo pair timestamp alignment and is closer to the actual frame capture time.
  // TscCorrelator converts ticks to nanoseconds using CNTFRQ_EL0 and applies
  // the calibrated TSC-to-ROS offset.
  return metadata.frameCaptureStartTSC;
}

uint64_t SiplCameraNode::nanosecondsToTscTicks(uint64_t nanoseconds) const
{
  return tsc_correlator_.nanosecondsToTscTicks(nanoseconds);
}

NvSciBufSurfSampleType SiplCameraNode::getSurfSampleTypeFromEncoding(
  const std::string & encoding) const
{
  static const std::unordered_map<std::string, NvSciBufSurfSampleType>
  kEncodingToSampleType = {
    {"nv12", NvSciSurfSampleType_420},
    {"nv24", NvSciSurfSampleType_444}
  };

  const auto it = kEncodingToSampleType.find(encoding);
  if (it != kEncodingToSampleType.end()) {
    return it->second;
  }

  RCLCPP_ERROR(
    get_logger(), "Unsupported encoding: %s", encoding.c_str());
  throw std::invalid_argument("Unsupported encoding: " + encoding);
}

std::vector<uint8_t> SiplCameraNode::loadNitoFile(const std::string & path)
{
  FILE * fp = fopen(path.c_str(), "rb");
  if (!fp) {
    RCLCPP_ERROR(get_logger(), "Could not open NITO file: %s", path.c_str());
    return {};
  }

  fseek(fp, 0, SEEK_END);
  const int64_t fsize = static_cast<int64_t>(ftell(fp));
  rewind(fp);

  if (fsize <= 0) {
    RCLCPP_ERROR(get_logger(), "Invalid NITO file size: %" PRId64, fsize);
    fclose(fp);
    return {};
  }

  std::vector<uint8_t> nito_data(fsize);
  size_t result = fread(nito_data.data(), 1, fsize, fp);
  fclose(fp);

  if (result != static_cast<size_t>(fsize)) {
    RCLCPP_ERROR(
      get_logger(), "Failed to read NITO file (read %zu of %" PRId64 " bytes)",
      result, fsize);
    return {};
  }

  RCLCPP_INFO(get_logger(), "Loaded NITO file: %s", path.c_str());
  return nito_data;
}

void SiplCameraNode::logStatistics(const char * report_type, bool final_report)
{
  const std::lock_guard<std::mutex> lock(statistics_report_mutex_);
  if (statistics_stopping_.load() && !final_report) {
    return;
  }

  const int64_t report_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
    std::chrono::steady_clock::now().time_since_epoch()).count();
  for (auto & pipeline : pipelines_) {
    auto & stats = *pipeline.statistics;
    const uint64_t published = stats.frames_published.load();
    const int64_t first_ns = stats.first_publish_ns.load();
    const int64_t last_ns = stats.last_publish_ns.load();
    double average_fps = 0.0;
    if (published > 1U && last_ns > first_ns) {
      average_fps = static_cast<double>(published - 1U) * 1.0e9 /
        static_cast<double>(last_ns - first_ns);
    }

    const uint64_t previous_published =
      stats.previous_report_published.exchange(published);
    int64_t previous_report_ns = stats.previous_report_ns.exchange(report_ns);
    uint64_t interval_frames = published - previous_published;
    if (previous_report_ns == 0 && first_ns > 0) {
      previous_report_ns = first_ns;
      interval_frames = published > 0U ? published - 1U : 0U;
    }
    double interval_fps = 0.0;
    if (previous_report_ns > 0 && report_ns > previous_report_ns) {
      interval_fps = static_cast<double>(interval_frames) * 1.0e9 /
        static_cast<double>(report_ns - previous_report_ns);
    }

    uint64_t stall_episodes = 0U;
    uint64_t stall_total_ns = 0U;
    std::map<uint16_t, uint64_t> sipl_event_totals;
    {
      const std::lock_guard<std::mutex> details_lock(stats.details_mutex);
      stall_episodes = stats.output_stall_episodes;
      stall_total_ns = stats.output_stall_total_ns;
      // Include the elapsed part of an active stall without closing it.
      if (stats.output_stall_start_ns != 0 && report_ns > stats.output_stall_start_ns) {
        stall_total_ns += static_cast<uint64_t>(report_ns - stats.output_stall_start_ns);
      }
      sipl_event_totals = stats.sipl_event_totals;
    }

    const uint64_t app_drops = stats.app_frame_drops.load();
    const uint64_t published_errors = stats.published_frame_errors.load();
    std::ostringstream sipl_event_stream;
    if (!sipl_event_totals.empty()) {
      sipl_event_stream << '{';
      bool first_event = true;
      for (const auto & [event_code, count] : sipl_event_totals) {
        if (!first_event) {
          sipl_event_stream << ',';
        }
        first_event = false;
        sipl_event_stream << siplNotificationName(event_code) << ':' << count;
      }
      sipl_event_stream << '}';
    }
    const std::string sipl_events_total = sipl_event_totals.empty() ?
      "none" : sipl_event_stream.str();
    const double stall_total_ms = static_cast<double>(stall_total_ns) / 1.0e6;

    if (final_report) {
      RCLCPP_INFO(
        get_logger(),
        "[Statistics] %s [%s] sensor=%u frames_total={icp_done:%" PRIu64
        ",acquired:%" PRIu64 ",published:%" PRIu64 "} interval_fps=%.3f"
        " average_fps=%.3f timing_totals={stall_episodes:%" PRIu64
        ",stall_ms:%.3f,latency_samples:%" PRIu64 ",latency_violations:%" PRIu64
        ",stereo_desyncs:%" PRIu64
        "} app_drops_total=%" PRIu64 " published_frame_errors_total=%" PRIu64
        " sipl_events_total=%s",
        report_type, pipeline.desc.display_name(), pipeline.sensor_id,
        stats.icp_frames_completed.load(), stats.frames_acquired.load(), published,
        interval_fps, average_fps, stall_episodes, stall_total_ms,
        stats.latency_samples.load(), stats.latency_violations.load(),
        stats.stereo_desyncs.load(), app_drops, published_errors, sipl_events_total.c_str());
    } else {
      RCLCPP_INFO(
        get_logger(),
        "[Statistics] %s [%s] sensor=%u frames_total={icp_done:%" PRIu64
        ",acquired:%" PRIu64 ",published:%" PRIu64 "} interval_fps=%.3f"
        " timing_totals={stall_episodes:%" PRIu64
        ",stall_ms:%.3f,latency_samples:%" PRIu64 ",latency_violations:%" PRIu64
        ",stereo_desyncs:%" PRIu64
        "} app_drops_total=%" PRIu64 " published_frame_errors_total=%" PRIu64
        " sipl_events_total=%s",
        report_type, pipeline.desc.display_name(), pipeline.sensor_id,
        stats.icp_frames_completed.load(), stats.frames_acquired.load(), published,
        interval_fps, stall_episodes, stall_total_ms,
        stats.latency_samples.load(), stats.latency_violations.load(),
        stats.stereo_desyncs.load(), app_drops, published_errors, sipl_events_total.c_str());
    }
  }
}

void SiplCameraNode::handlePipelineNotification(
  CameraPipeline & pipeline,
  const nvsipl::NvSIPLPipelineNotifier::NotificationData & event)
{
  using NotifType = nvsipl::NvSIPLPipelineNotifier::NotificationType;
  const uint16_t event_code = static_cast<uint16_t>(event.eNotifType);
  constexpr uint16_t kFirstSiplWarningEvent = 100U;
  if (enable_statistics_ && event_code >= kFirstSiplWarningEvent) {
    const std::lock_guard<std::mutex> details_lock(pipeline.statistics->details_mutex);
    ++pipeline.statistics->sipl_event_totals[event_code];
  }

  switch (event.eNotifType) {
    case NotifType::NOTIF_INFO_ICP_PROCESSING_DONE: {
        if (enable_statistics_) {
          ++pipeline.statistics->icp_frames_completed;
        }
        RCLCPP_DEBUG(
          get_logger(), "[Sipl Notification] ICP processing done on sensor %u",
          event.uIndex);
        break;
      }
    case NotifType::NOTIF_WARN_ICP_FRAME_DROP: {
        RCLCPP_WARN(
          get_logger(),
          "[Sipl Notification] ICP Frame drop detected on sensor %u (pipeline full?)",
          event.uIndex);
        break;
      }
    case NotifType::NOTIF_WARN_ICP_FRAME_DISCONTINUITY: {
        RCLCPP_WARN(
          get_logger(), "[Sipl Notification] Frame discontinuity on sensor %u, frame %lu",
          event.uIndex, event.frameSeqNumber);
        break;
      }
    case NotifType::NOTIF_WARN_ICP_CAPTURE_TIMEOUT: {
        RCLCPP_WARN(get_logger(), "[Sipl Notification] Capture timeout on sensor %u", event.uIndex);
        break;
      }
    case NotifType::NOTIF_WARN_ICP_APP_BACK_PRESSURE: {
        RCLCPP_WARN(get_logger(), "[Sipl Notification] App back pressure on sensor %u",
          event.uIndex);
        break;
      }
    case NotifType::NOTIF_ERROR_ICP_CAPTURE_FAILURE: {
        RCLCPP_ERROR(get_logger(), "[Sipl Notification] Capture failure on sensor %u",
          event.uIndex);
        break;
      }
    case NotifType::NOTIF_ERROR_ICP_EMB_DATA_PARSE_FAILURE:
      RCLCPP_ERROR(
        get_logger(), "[Sipl Notification] Embedded data parse failure on sensor %u",
        event.uIndex);
      break;
    case NotifType::NOTIF_ERROR_ISP_PROCESSING_FAILURE:
      RCLCPP_ERROR(
        get_logger(), "[Sipl Notification] ISP processing failure on sensor %u",
        event.uIndex);
      break;
    case NotifType::NOTIF_ERROR_INTERNAL_FAILURE:
      RCLCPP_ERROR(get_logger(), "[Sipl Notification] Internal failure");
      break;
    case NotifType::NOTIF_ERROR_ACP_PROCESSING_FAILURE:
      RCLCPP_ERROR(
        get_logger(), "[Sipl Notification] Auto control processing failure on sensor %u",
        event.uIndex);
      break;
    case NotifType::NOTIF_ERROR_CDI_SET_SENSOR_CTRL_FAILURE:
      RCLCPP_ERROR(
        get_logger(), "[Sipl Notification] Sensor control write failure on sensor %u",
        event.uIndex);
      break;
    case NotifType::NOTIF_ERROR_ICP_BAD_INPUT_STREAM:
      RCLCPP_ERROR(
        get_logger(),
        "[Sipl Notification] ICP bad input stream on sensor %u", event.uIndex);
      break;
    case NotifType::NOTIF_ERROR_DESERIALIZER_FAILURE:
      RCLCPP_ERROR(
        get_logger(), "[Sipl Notification] Deserializer failure");
      break;
    case NotifType::NOTIF_ERROR_SERIALIZER_FAILURE:
      RCLCPP_ERROR(
        get_logger(),
        "[Sipl Notification] Serializer failure on link mask 0x%02x",
        event.uLinkMask);
      break;
    case NotifType::NOTIF_ERROR_SENSOR_FAILURE:
      RCLCPP_ERROR(
        get_logger(),
        "[Sipl Notification] Sensor failure on link mask 0x%02x",
        event.uLinkMask);
      break;
    case NotifType::NOTIF_ERROR_ISP_PROCESSING_FAILURE_RECOVERABLE:
      RCLCPP_WARN(
        get_logger(),
        "[Sipl Notification] Recoverable ISP processing failure on sensor %u",
        event.uIndex);
      break;
    case NotifType::NOTIF_ERROR_ICP_AUTH_FAILURE:
      RCLCPP_ERROR(
        get_logger(),
        "[Sipl Notification] ICP authentication failure on sensor %u",
        event.uIndex);
      break;
    case NotifType::NOTIF_ERROR_ICP_AUTH_OUT_OF_ORDER:
      RCLCPP_ERROR(
        get_logger(),
        "[Sipl Notification] ICP authentication out-of-order on sensor %u",
        event.uIndex);
      break;
    case NotifType::NOTIF_ERROR_ACP_SETTINGS_DISCONTINUITY:
      RCLCPP_ERROR(
        get_logger(),
        "[Sipl Notification] AC plugin settings discontinuity on sensor %u",
        event.uIndex);
      break;
    case NotifType::NOTIF_ERROR_INTR_LOCALIZATION_FAILURE:
      RCLCPP_ERROR(
        get_logger(), "[Sipl Notification] Interrupt localization failure");
      break;
    case NotifType::NOTIF_ERROR_INTR_LOCALIZATION_TIMEOUT:
      RCLCPP_ERROR(
        get_logger(), "[Sipl Notification] Interrupt localization timeout");
      break;
    case NotifType::NOTIF_ERROR_ISP_PREFENCE_TIMEOUT:
      RCLCPP_ERROR(
        get_logger(),
        "[Sipl Notification] ISP prefence timeout on sensor %u", event.uIndex);
      break;
    case NotifType::NOTIF_INIT_ERROR_FAILURE:
      RCLCPP_ERROR(
        get_logger(),
        "[Sipl Notification] Pipeline init error on sensor %u", event.uIndex);
      break;
    case NotifType::NOTIF_ERROR_INTR_RECOVERY:
      RCLCPP_WARN(
        get_logger(), "[Sipl Notification] Interrupt recovery underway");
      break;
    case NotifType::NOTIF_WARN_DEVICE_BLOCK_ERROR_INTR_DETECTED:
      RCLCPP_WARN(
        get_logger(),
        "[Sipl Notification] Device block error interrupt detected on sensor %u",
        event.uIndex);
      break;
    case NotifType::NOTIF_ERROR_POWER_LOAD_SWITCH_FAILURE:
      RCLCPP_ERROR(
        get_logger(),
        "[Sipl Notification] Power load switch failure on sensor %u",
        event.uIndex);
      break;
    default:
      // Preserve the severity of future SIPL event codes.
      if (event_code >= 200U) {
        RCLCPP_ERROR(
          get_logger(), "[Sipl Notification] Unhandled error event %u on sensor %u",
          static_cast<uint32_t>(event_code), event.uIndex);
      } else if (event_code >= kFirstSiplWarningEvent) {
        RCLCPP_WARN(
          get_logger(), "[Sipl Notification] Unhandled warning event %u on sensor %u",
          static_cast<uint32_t>(event_code), event.uIndex);
      } else {
        RCLCPP_DEBUG(
          get_logger(), "[Sipl Notification] Pipeline event %u on sensor %u",
          static_cast<uint32_t>(event_code), event.uIndex);
      }
      break;
  }
}

bool SiplCameraNode::publishFrame(
  CameraPipeline & pipeline,
  const std_msgs::msg::Header & header,
  uint32_t step_bytes,
  const std::function<bool(uint8_t * compact_ptr, size_t block_size)> & fill_fn,
  const std::function<bool()> & after_fill)
{
  auto image_msg = std::make_unique<sensor_msgs::msg::Image>();
  image_msg->header = header;
  image_msg->height = pipeline.image_height;
  image_msg->width = pipeline.image_width;
  image_msg->encoding = pipeline.encoding;
  image_msg->is_bigendian = 0;
  image_msg->step = step_bytes;
  try {
    image_msg->data = cuda_buffer_backend::allocate_buffer(pipeline.compact_frame_bytes);
    // The write handle records a CUDA event on the pipeline's stream when it
    // falls out of scope, which consumers await before reading.
    auto write_handle =
      cuda_buffer_backend::from_output_buffer(image_msg->data, *pipeline.cuda_stream);
    if (!fill_fn(write_handle.get_ptr(), image_msg->data.size())) {
      return false;
    }
  } catch (const std::exception & e) {
    RCLCPP_ERROR_THROTTLE(
      get_logger(), *get_clock(), 1000,
      "[%s] Failed to allocate or fill CUDA image: %s. Dropping frame",
      pipeline.desc.display_name(), e.what());
    return false;
  }

  if (after_fill && !after_fill()) {
    return false;
  }

  // Measure SOF-to-publisher latency before DDS delivery.
  if (use_hw_timestamp_) {
    const int64_t publication_ros_ns = now().nanoseconds();
    const int64_t capture_ros_ns =
      rclcpp::Time(header.stamp, get_clock()->get_clock_type()).nanoseconds();
    const int64_t latency_us = (publication_ros_ns - capture_ros_ns) / 1000;
    auto & stats = *pipeline.statistics;
    ++stats.latency_samples;
    const int64_t threshold_us = static_cast<int64_t>(
      max_capture_to_publish_latency_frame_periods_ * 1.0e6 / expected_fps_);
    if (threshold_us > 0 && latency_us > threshold_us) {
      const uint64_t violation_count = ++stats.latency_violations;
      const int64_t steady_now_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
      constexpr int64_t kLatencyErrorThrottleNs = 1'000'000'000LL;
      int64_t last_error_ns = stats.last_latency_error_ns.load();
      if (steady_now_ns - last_error_ns >= kLatencyErrorThrottleNs &&
        stats.last_latency_error_ns.compare_exchange_strong(last_error_ns, steady_now_ns))
      {
        RCLCPP_WARN(
          get_logger(),
          "[%s] Abnormal capture SOF-to-publisher latency: %.3f ms exceeds %.3f ms "
          "threshold (%.3f frame periods at %.3f FPS; violation=%" PRIu64 ")",
          pipeline.desc.display_name(), latency_us / 1000.0, threshold_us / 1000.0,
          max_capture_to_publish_latency_frame_periods_, expected_fps_, violation_count);
      }
    }
  }

  // Publish image and the matching CameraInfo with the same timestamp.
  try {
    pipeline.image_pub->publish(std::move(image_msg));
    publishCameraInfo(pipeline, header);
  } catch (const std::exception & e) {
    RCLCPP_ERROR_THROTTLE(
      get_logger(), *get_clock(), 1000,
      "[%s] Failed to publish image or camera_info: %s. Dropping frame",
      pipeline.desc.display_name(), e.what());
    return false;
  } catch (...) {
    RCLCPP_ERROR_THROTTLE(
      get_logger(), *get_clock(), 1000,
      "[%s] Failed to publish image or camera_info: unknown exception. Dropping frame",
      pipeline.desc.display_name());
    return false;
  }
  if (enable_statistics_) {
    const int64_t now_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
      std::chrono::steady_clock::now().time_since_epoch()).count();
    int64_t unset = 0;
    pipeline.statistics->first_publish_ns.compare_exchange_strong(unset, now_ns);
    pipeline.statistics->last_publish_ns.store(now_ns);
    ++pipeline.statistics->frames_published;
  }
  return true;
}


}  // namespace sipl
}  // namespace isaac_ros

// Register as component
#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(isaac_ros::sipl::SiplCameraNode)
