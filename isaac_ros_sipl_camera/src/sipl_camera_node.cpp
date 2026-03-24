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

#include <arpa/inet.h>
#include <chrono>
#include <cinttypes>
#include <fstream>
#include <sstream>
#include <unordered_map>
#include <variant>
#include <vector>

#include "magic_enum.hpp"
#include "tf2/LinearMath/Quaternion.h"
#include "gxf/multimedia/video.hpp"
#include "gxf/core/entity.hpp"
#include "isaac_ros_nitros/types/type_adapter_nitros_context.hpp"

using namespace std::chrono_literals;

namespace isaac_ros
{
namespace sipl
{

// COE capture driver (CoeChannelDriverLinux) limits up to 4 registered buffers per channel,
// defined by FUSA_MAX_PENDING_CAPTURES_COE in NvFusaCaptureCommon.h. SIPL also enforces 4 as
// the minimum for COE mode, making this the only valid value.
constexpr uint16_t kNumCaptureBuffers = 4U;

SiplCameraNode::SiplCameraNode(const rclcpp::NodeOptions & options)
: SiplCameraNode(options, true)
{
  // Mono camera constructor initializes immediately.
  initialize(describePipelines());
}

SiplCameraNode::SiplCameraNode(const rclcpp::NodeOptions & options, bool /*defer_initialization*/)
: rclcpp::Node("sipl_camera_node", options),
  sipl_camera_(nullptr),
  sipl_query_(nullptr),
  sci_buf_module_(nullptr),
  cpu_wait_context_(nullptr),
  stop_capture_(false),
  image_width_(0),
  image_height_(0),
  sci_sync_module_(nullptr)
{
  platform_config_ = declare_parameter<std::string>("platform_config", "VB1940_Camera");

  // CoE network configuration
  hsb_name_ = declare_parameter<std::string>("hsb_name", "hsb_0");
  hsb_id_ = declare_parameter<int>("hsb_id", 0);
  interface_name_ = declare_parameter<std::string>("interface_name", "mgbe0_0");
  declare_parameter<std::string>("ip_address", "");
  declare_parameter<std::string>("mac_address", "");

  // Get debug flag from the ROS logger level so that users control
  // verbosity with the standard --ros-args --log-level <node_name>:=<level>
  enable_debug_logs_ = rcutils_logging_logger_is_enabled_for(
    get_logger().get_name(), RCUTILS_LOG_SEVERITY_DEBUG);

  // Output encoding
  rcl_interfaces::msg::ParameterDescriptor encoding_desc;
  encoding_desc.name = "encoding_desired";
  encoding_desc.description = "Desired output encoding. Allowed: nv12, nv24";
  encoding_desc.additional_constraints = "Must be 'nv12' or 'nv24'";
  encoding_desc.read_only = true;
  encoding_desc.type = rcl_interfaces::msg::ParameterType::PARAMETER_STRING;
  encoding_desired_ = declare_parameter<std::string>(
    "encoding_desired", "nv12", encoding_desc);

  // Buffer configuration
  enable_cpu_access_ = declare_parameter<bool>("enable_cpu_access", false);

  // Timestamp configuration
  use_hw_timestamp_ = declare_parameter<bool>("use_hw_timestamp", true);

  // TF frame names
  camera_link_frame_name_ = declare_parameter<std::string>("camera_link_frame_name", "camera");
  optical_frame_name_ = camera_link_frame_name_ + "_optical";

  const std::string camera_info_url =
    declare_parameter<std::string>("camera_info_url", "");
  if (!camera_info_url.empty()) {
    camera_info_ = loadCameraInfoFromFile(camera_info_url);
    camera_info_loaded_ = true;
    RCLCPP_INFO(
      get_logger(), "Loaded camera info from \"%s\"",
      camera_info_url.c_str());
  }

  // NITO file directory
  nito_path_ = declare_parameter<std::string>(
    "nito_path", "/var/nvidia/nvcam/settings/sipl");

  // Create TF broadcaster
  tf_static_broadcaster_ = std::make_unique<tf2_ros::StaticTransformBroadcaster>(this);

  if (enable_debug_logs_) {
    nvsipl::INvSIPLTrace::GetInstance()->SetLevel(nvsipl::INvSIPLTrace::TraceLevel::LevelDebug);
  }
}


SiplCameraNode::~SiplCameraNode()
{
  stopAllPipelines();

  // Clean up per-pipeline sync objects
  for (auto & pipeline : pipelines_) {
    if (pipeline.sci_sync_isp0) {
      NvSciSyncObjFree(pipeline.sci_sync_isp0);
    }
  }

  // Clean up NvSci modules
  if (cpu_wait_context_) {
    NvSciSyncCpuWaitContextFree(cpu_wait_context_);
  }

  if (sci_sync_module_) {
    NvSciSyncModuleClose(sci_sync_module_);
  }
}

void SiplCameraNode::setupSiplCamera(size_t num_sensors)
{
  RCLCPP_INFO(get_logger(), "=== SIPL Camera Setup Begin ===");
  // Get SIPL Query instance and load platform configuration
  sipl_query_ = nvsipl::INvSIPLCameraQuery::GetInstance();
  if (!sipl_query_) {
    throw std::runtime_error("Failed to get SIPL Query instance");
  }

  RCLCPP_INFO(get_logger(), "  Parsing SIPL database...");
  nvsipl::SIPLStatus status = sipl_query_->ParseDatabase();
  if (status != nvsipl::NVSIPL_STATUS_OK) {
    RCLCPP_FATAL(get_logger(), "  ParseDatabase failed with status: %u",
      static_cast<uint32_t>(status));
    throw std::runtime_error("Failed to parse SIPL database");
  }

  RCLCPP_INFO(get_logger(), "  Getting camera system config for: %s", platform_config_.c_str());
  status = sipl_query_->GetCameraSystemConfig(platform_config_, camera_system_config_);
  if (status != nvsipl::NVSIPL_STATUS_OK) {
    RCLCPP_FATAL(get_logger(),
      "  GetCameraSystemConfig failed with status: %u for platform: %s",
      static_cast<uint32_t>(status), platform_config_.c_str());
    throw std::runtime_error("Camera system config not found");
  }

  RCLCPP_INFO(
    get_logger(), "Loaded platform config: %s with %zu camera(s)",
    platform_config_.c_str(), camera_system_config_.cameras.size());
  for (size_t i = 0; i < camera_system_config_.cameras.size(); ++i) {
    const auto & cam = camera_system_config_.cameras[i];
    RCLCPP_INFO(get_logger(), "    Camera[%zu]: sensor_id=%u, resolution=%ux%u",
      i, cam.sensorInfo.id,
      cam.sensorInfo.vcInfo.resolution.width,
      cam.sensorInfo.vcInfo.resolution.height);
  }

  // Filter to match the requested number of sensors
  if (camera_system_config_.cameras.size() > num_sensors) {
    RCLCPP_INFO(
      get_logger(), "Filtering platform config from %zu cameras to %zu",
      camera_system_config_.cameras.size(), num_sensors);
    camera_system_config_.cameras.resize(num_sensors);
  }

  if (camera_system_config_.cameras.size() < num_sensors) {
    throw std::runtime_error("Insufficient camera configs");
  }

  RCLCPP_INFO(get_logger(), "Applying network parameter overrides...");
  applyNetworkOverrides();

  // Get camera instance
  sipl_camera_ = nvsipl::INvSIPLCamera::GetInstance();
  if (!sipl_camera_) {
    throw std::runtime_error("Failed to get SIPL Camera instance");
  }

  status = sipl_camera_->SetPlatformCfg(camera_system_config_);
  if (status != nvsipl::NVSIPL_STATUS_OK) {
    RCLCPP_FATAL(get_logger(), "  SetPlatformCfg failed with status: %u",
      static_cast<uint32_t>(status));
    throw std::runtime_error("Failed to set platform configuration");
  }

  RCLCPP_INFO(get_logger(), "SIPL Camera platform setup complete");
}

std::vector<SiplCameraNode::PipelineDescriptor> SiplCameraNode::describePipelines()
{
  return {{
    "", 0,
    camera_link_frame_name_ + "_optical",
    camera_info_, camera_info_loaded_
  }};
}

void SiplCameraNode::initialize(std::vector<PipelineDescriptor> descriptors)
{
  setupSiplCamera(descriptors.size());

  // Configure pipeline (ISP0 only, no RAW)
  nvsipl::NvSIPLPipelineConfiguration pipeline_cfg = {
    .captureOutputRequested = true,
    .isp0OutputRequested = true,
    .isp1OutputRequested = false,
    .isp2OutputRequested = false,
    .disableSubframe = true,
    .bufferCfg = {
      .maxCaptureBufferCount = kNumCaptureBuffers,
      .maxIsp0BufferCount = 64U,
      .maxIsp1BufferCount = 64U,
      .maxIsp2BufferCount = 64U,
    }
  };

  RCLCPP_INFO(get_logger(), "Configuring pipelines for %zu sensor(s)", descriptors.size());

  // Phase 1: Configure all pipelines (must complete before Init())
  for (auto & desc : descriptors) {
    CameraPipeline pipeline;
    pipeline.desc = std::move(desc);
    pipeline.sensor_id =
      camera_system_config_.cameras[pipeline.desc.camera_index].sensorInfo.id;

    RCLCPP_INFO(get_logger(), "[%s] Configuring pipeline (sensor_id=%u)",
      pipeline.desc.display_name(), pipeline.sensor_id);

    auto status = sipl_camera_->SetPipelineCfg(
      pipeline.sensor_id, pipeline_cfg, pipeline.queues);
    if (status != nvsipl::NVSIPL_STATUS_OK) {
      throw std::runtime_error(
        "Failed to set pipeline configuration for sensor " +
        std::to_string(pipeline.sensor_id));
    }

    pipelines_.push_back(std::move(pipeline));
  }

  // Set image dimensions from first camera
  image_width_ = camera_system_config_.cameras[0].sensorInfo.vcInfo.resolution.width;
  image_height_ = camera_system_config_.cameras[0].sensorInfo.vcInfo.resolution.height;

  // Phase 2: SIPL Init
  auto status = sipl_camera_->Init();
  if (status != nvsipl::NVSIPL_STATUS_OK) {
    RCLCPP_FATAL(get_logger(), "SIPL Init() failed with status: %u",
      static_cast<uint32_t>(status));
    throw std::runtime_error("Failed to initialize SIPL camera");
  }

  // Phase 3: Initialize NvSci modules
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
  sci_buf_module_.reset(module_handle, [](NvSciBufModuleRec * p) {
      if (p) {
        NvSciBufModuleClose(p);
      }
    });

  sci_err = NvSciSyncCpuWaitContextAlloc(sci_sync_module_, &cpu_wait_context_);
  if (sci_err != NvSciError_Success) {
    RCLCPP_FATAL(get_logger(), "Failed to allocate CPU wait context: %d", sci_err);
    throw std::runtime_error("CPU wait context allocation failed");
  }

  // Phase 4: Per-pipeline buffer allocation, auto control, publishers
  for (auto & pipeline : pipelines_) {
    RCLCPP_INFO(get_logger(), "[%s] Allocating buffers...",
      pipeline.desc.display_name());
    allocateBuffersForPipeline(pipeline);

    RCLCPP_INFO(get_logger(), "[%s] Registering Auto Control Plugin (sensor %u)",
      pipeline.desc.display_name(), pipeline.sensor_id);
    registerAutoControlPlugin(pipeline.desc.camera_index);

    RCLCPP_INFO(get_logger(), "[%s] Creating publishers...",
      pipeline.desc.display_name());
    createPublisherForPipeline(pipeline);
  }

  RCLCPP_INFO(get_logger(), "Starting capture...");
  startAllPipelines();

  RCLCPP_INFO(
    get_logger(),
    "SIPL Camera initialized: %s, %dx%d",
    platform_config_.c_str(), image_width_, image_height_);
}


void SiplCameraNode::allocateBuffersForPipeline(CameraPipeline & pipeline)
{
  // ICP buffers
  pipeline.buffer_manager_icp = std::make_shared<SiplBufferManager>(
    sci_buf_module_, kNumCaptureBuffers, get_logger(), enable_debug_logs_);

  auto status = pipeline.buffer_manager_icp->allocateAndRegisterBuffers(
    sipl_camera_.get(),
    pipeline.sensor_id,
    nvsipl::INvSIPLClient::ConsumerDesc::OutputType::ICP);
  if (status != nvsipl::NVSIPL_STATUS_OK) {
    throw std::runtime_error(
      "[" + std::string(pipeline.desc.display_name()) +
      "] Failed to allocate and register ICP buffers");
  }

  // ISP0 buffers
  pipeline.buffer_manager_isp0 = std::make_shared<SiplBufferManager>(
    sci_buf_module_, kNumCaptureBuffers, get_logger(), enable_debug_logs_);

  const auto sample_type = getSurfSampleTypeFromEncoding();
  RCLCPP_DEBUG(
    get_logger(), "[%s] Requested ISP0 sample type for encoding '%s'",
    pipeline.desc.display_name(), encoding_desired_.c_str());

  if (enable_cpu_access_) {
    RCLCPP_INFO(get_logger(),
      "[%s] Enabling CPU access and ReadWrite permissions for ISP0 buffers",
      pipeline.desc.display_name());
  }

  status = pipeline.buffer_manager_isp0->allocateAndRegisterBuffers(
    sipl_camera_.get(),
    pipeline.sensor_id,
    nvsipl::INvSIPLClient::ConsumerDesc::OutputType::ISP0,
    enable_cpu_access_,
    sample_type);
  if (status != nvsipl::NVSIPL_STATUS_OK) {
    throw std::runtime_error(
      "[" + std::string(pipeline.desc.display_name()) +
      "] Failed to allocate and register ISP0 buffers");
  }

  RCLCPP_INFO(
    get_logger(), "[%s] Registered %u ICP and ISP0 buffers (sensor %u)",
    pipeline.desc.display_name(), kNumCaptureBuffers, pipeline.sensor_id);

  // Allocate and register sync objects (Required before starting capture)
  RCLCPP_INFO(get_logger(), "[%s] Allocating sync objects...",
    pipeline.desc.display_name());
  allocateSync(pipeline.desc.camera_index,
    nvsipl::INvSIPLClient::ConsumerDesc::OutputType::ISP0,
    pipeline.sci_sync_isp0);
}

void SiplCameraNode::createPublisherForPipeline(CameraPipeline & pipeline)
{
  const auto & name = pipeline.desc.name;
  std::string image_topic = name.empty() ? "image_raw" : name + "/image_raw";
  std::string camera_info_topic = name.empty() ? "camera_info" : name + "/camera_info";

  std::string nitros_format = getNitrosFormatFromEncoding();
  pipeline.image_pub = std::make_shared<nvidia::isaac_ros::nitros::ManagedNitrosPublisher<
        nvidia::isaac_ros::nitros::NitrosImage>>(
    this,
    image_topic,
    nitros_format,
    nvidia::isaac_ros::nitros::NitrosDiagnosticsConfig{},
    rclcpp::QoS(10));

  pipeline.camera_info_pub = create_publisher<sensor_msgs::msg::CameraInfo>(
    camera_info_topic, 10);
}


void SiplCameraNode::startAllPipelines()
{
  RCLCPP_INFO(get_logger(), "Starting capture...");
  nvsipl::SIPLStatus status = sipl_camera_->Start();
  if (status != nvsipl::NVSIPL_STATUS_OK) {
    throw std::runtime_error("Failed to start SIPL camera");
  }
  stop_capture_ = false;

  // Publish static transforms before spawning capture threads so the TF tree
  // is available to downstream consumers by the time the first image arrives.
  publishStaticTransforms();

  for (auto & pipeline : pipelines_) {
    pipeline.frame_state = std::make_shared<FrameState>();

    pipeline.capture_thread = std::thread(
      &SiplCameraNode::captureThread, this, std::ref(pipeline));
    pipeline.process_thread = std::thread(
      &SiplCameraNode::processThread, this, std::ref(pipeline));
    pipeline.event_thread = std::thread(
      &SiplCameraNode::handleNotificationQueue, this, std::ref(pipeline));
  }
  RCLCPP_INFO(get_logger(), "Capture threads started for %zu pipeline(s)", pipelines_.size());
}

void SiplCameraNode::stopAllPipelines()
{
  RCLCPP_INFO(get_logger(), "Stopping capture...");
  stop_capture_ = true;

  for (auto & pipeline : pipelines_) {
    if (pipeline.capture_thread.joinable()) {
      pipeline.capture_thread.join();
    }
    if (pipeline.process_thread.joinable()) {
      pipeline.process_thread.join();
    }
    if (pipeline.event_thread.joinable()) {
      pipeline.event_thread.join();
    }
  }

  for (const auto & pipeline : pipelines_) {
    RCLCPP_INFO(
      get_logger(),
      "[%s] Mail box drops=%" PRIu64,
      pipeline.desc.display_name(),
      pipeline.dropped_mail);
  }

  if (sipl_camera_) {
    RCLCPP_INFO(get_logger(), "Stopping and deinitializing SIPL camera...");
    sipl_camera_->Stop();
    sipl_camera_->Deinit();
    sipl_camera_.reset();
  }
}

void SiplCameraNode::allocateSync(
  uint32_t camera_index,
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

  const auto sensor_id = camera_system_config_.cameras[camera_index].sensorInfo.id;
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

void SiplCameraNode::registerAutoControlPlugin(uint32_t camera_index)
{
  std::string module_name = camera_system_config_.cameras[camera_index].sensorInfo.name;
  std::vector<uint8_t> nito_blob = loadNitoFile(module_name);
  if (nito_blob.empty()) {
    RCLCPP_FATAL(get_logger(), "Failed to load NITO file for module: %s", module_name.c_str());
    throw std::runtime_error("Failed to load NITO file");
  }

  const auto sensor_id = camera_system_config_.cameras[camera_index].sensorInfo.id;
  auto status = sipl_camera_->RegisterAutoControlPlugin(sensor_id, nvsipl::NV_PLUGIN, nullptr,
        nito_blob);
  if (status != nvsipl::NVSIPL_STATUS_OK) {
    throw std::runtime_error("Failed to register autocontrol plugin");
  }

  RCLCPP_INFO(
    get_logger(), "Registered Auto Control Plugin for sensor %u (module: %s)",
    sensor_id, camera_system_config_.cameras[camera_index].sensorInfo.name.c_str());
}


void SiplCameraNode::applyNetworkOverrides()
{
  // Get parameters directly
  std::string mac_address = get_parameter("mac_address").as_string();
  std::string ip_address = get_parameter("ip_address").as_string();

  // Parse MAC address if provided
  uint8_t mac_bytes[6] = {0};
  bool has_mac = false;
  if (!mac_address.empty()) {
    try {
      parseMacAddress(mac_address, mac_bytes);
      has_mac = true;
      RCLCPP_INFO(
        get_logger(), "MAC address override: %s", mac_address.c_str());
    } catch (const std::exception & e) {
      RCLCPP_WARN(get_logger(), "Invalid MAC address format: %s", mac_address.c_str());
    }
  }

  // Parse IP address if provided
  uint32_t ip_value = 0;
  bool has_ip = false;
  if (!ip_address.empty()) {
    try {
      ip_value = parseIpAddress(ip_address);
      has_ip = true;
      RCLCPP_INFO(get_logger(), "IP address override: %s", ip_address.c_str());
    } catch (const std::exception & e) {
      RCLCPP_WARN(get_logger(), "Invalid IP address format: %s", ip_address.c_str());
    }
  }

  // Apply to transport configurations
  for (auto & transport : camera_system_config_.transports) {
    if (std::holds_alternative<nvsipl::CoETransSettings>(transport)) {
      auto & coe_trans = std::get<nvsipl::CoETransSettings>(transport);
      if (coe_trans.hsbId == static_cast<uint32_t>(hsb_id_)) {
        coe_trans.interfaceName = interface_name_;
        if (has_ip) {
          coe_trans.ipAddress = ip_value;
        }
        RCLCPP_INFO(
          get_logger(), "Applied transport override: interface=%s",
          interface_name_.c_str());
      }
    }
  }

  // Apply to camera configurations
  for (auto & camera : camera_system_config_.cameras) {
    if (std::holds_alternative<nvsipl::CoECamera>(camera.cameratype)) {
      auto & coe_cam = std::get<nvsipl::CoECamera>(camera.cameratype);
      if (coe_cam.hsbId == static_cast<uint32_t>(hsb_id_)) {
        if (coe_cam.sensors == nullptr) {
          RCLCPP_FATAL(get_logger(),
            "CoE camera sensors pointer is null for hsb_id=%d; "
            "platform config '%s' may be corrupt",
            hsb_id_, platform_config_.c_str());
          throw std::runtime_error("CoE camera sensors pointer is null");
        }
        if (has_ip) {
          coe_cam.sensors->ipAddress = ip_value;
        }
        if (has_mac) {
          memcpy(coe_cam.sensors->macAddress, mac_bytes, 6);
        }
      }
    }
  }
}


void SiplCameraNode::captureThread(CameraPipeline & pipeline)
{
  RCLCPP_INFO(get_logger(), "[%s] Capture thread started (sensor %u)",
    pipeline.desc.display_name(), pipeline.sensor_id);

  nvsipl::INvSIPLClient::INvSIPLBuffer * buffer = nullptr;

  auto last_frame_time = std::chrono::steady_clock::now();

  while (rclcpp::ok() && !stop_capture_) {
    if (!pipeline.queues.captureCompletionQueue) {
      RCLCPP_ERROR(get_logger(), "[%s] Capture completion queue is null",
        pipeline.desc.display_name());
      break;
    }
    if (!pipeline.queues.isp0CompletionQueue) {
      RCLCPP_ERROR(get_logger(), "[%s] ISP0 completion queue is null",
        pipeline.desc.display_name());
      break;
    }

    // We must wait and read the raw frame even when we do not use it before we can
    // get the ISP0 buffer. Get buffer from ICP completion queue.
    nvsipl::INvSIPLClient::INvSIPLBuffer * buffer_raw = nullptr;
    auto status = pipeline.queues.captureCompletionQueue->Get(buffer_raw,
          SiplCameraNode::kSiplQueueGetTimeoutUs);
    if (status != nvsipl::NVSIPL_STATUS_OK) {
      if (status == nvsipl::NVSIPL_STATUS_TIMED_OUT) {
        // Timeout is expected, continue loop to check stop condition.
        RCLCPP_WARN(get_logger(), "[%s] Timeout getting RAW buffer",
          pipeline.desc.display_name());
        continue;
      }
      RCLCPP_ERROR(get_logger(), "[%s] Failed to get RAW buffer (status=%d)",
            pipeline.desc.display_name(), static_cast<int>(status));
      continue;
    }
    // We're not using the RAW buffer, so release it immediately.
    if (buffer_raw != nullptr) {
      buffer_raw->Release();
    }

    // Timestamp before blocking on the ISP0 completion queue. The ISP is processing
    // (or has already processed) the frame during this Get() call. This marks the
    // start of our ISP wait so we can measure the full consumer-visible ISP latency:
    // time spent blocked in Get() + mailbox handoff + remaining DMA flush (EOF fence).
    auto isp0_start_time = std::chrono::steady_clock::now();

    // Get buffer from ISP0 completion queue.
    status =
      pipeline.queues.isp0CompletionQueue->Get(buffer, SiplCameraNode::kSiplQueueGetTimeoutUs);

    if (status == nvsipl::NVSIPL_STATUS_OK && buffer != nullptr) {
      updateTscOffset();

      if (enable_debug_logs_) {
        // FPS Calculation
        auto now = std::chrono::steady_clock::now();
        auto elapsed_us = std::chrono::duration_cast<std::chrono::microseconds>(
          now - last_frame_time).count();

        if (elapsed_us > 0) {
          double fps = 1000000.0 / static_cast<double>(elapsed_us);
          RCLCPP_DEBUG(
            get_logger(), "[Capture] [%s] Instantaneous FPS: %.2f",
            pipeline.desc.display_name(), fps);
        }
        last_frame_time = now;
      }

      // Mailbox Logic:
      // 1. Lock mutex
      // 2. If pending buffer exists, release it (drop frame)
      // 3. Store new buffer
      // 4. Notify process thread
      {
        std::lock_guard<std::mutex> lock(pipeline.frame_state->mutex);
        if (pipeline.frame_state->pending_buffer != nullptr) {
          // Drop previous frame
          pipeline.frame_state->pending_buffer->Release();
          ++pipeline.dropped_mail;
          RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 100,
            "[%s] Frame skipped: likely downstream processing is not keeping up "
            "with the camera frame rate.",
            pipeline.desc.display_name());
        }
        pipeline.frame_state->pending_buffer = buffer;
        pipeline.frame_state->isp0_start_time = isp0_start_time;
      }
      pipeline.frame_state->cv.notify_one();

      size_t queue_depth = pipeline.queues.isp0CompletionQueue->GetCount();
      if (queue_depth > 1) {
        RCLCPP_WARN_THROTTLE(
          get_logger(), *get_clock(), 1000,
          "[Capture] [%s] ISP0 completion queue depth high (%zu). Possible backlog.",
          pipeline.desc.display_name(), queue_depth);
      }
      RCLCPP_DEBUG(get_logger(), "[Capture] [%s] Acquired buffer %p, queue_depth=%zu",
        pipeline.desc.display_name(), static_cast<void *>(buffer), queue_depth);

      buffer = nullptr;  // Ownership transferred to pending_buffer for processThread
    } else if (status == nvsipl::NVSIPL_STATUS_TIMED_OUT) {
      RCLCPP_DEBUG_THROTTLE(get_logger(), *get_clock(), 1000,
        "[Capture] [%s] Waiting for frames (queue timeout, no data from sensor)",
        pipeline.desc.display_name());
    } else if (status == nvsipl::NVSIPL_STATUS_EOF) {
      RCLCPP_INFO(get_logger(), "[%s] ISP0 queue EOF received",
        pipeline.desc.display_name());
      break;
    } else {
      RCLCPP_WARN(get_logger(), "[%s] Failed to get buffer (status=%u)",
        pipeline.desc.display_name(), static_cast<uint32_t>(status));
    }
  }

  // Set processing active to false to ensure process thread exits if it's waiting
  pipeline.frame_state->processing_active = false;
  pipeline.frame_state->cv.notify_all();

  RCLCPP_INFO(get_logger(), "[%s] Capture thread exited", pipeline.desc.display_name());
}

void SiplCameraNode::processThread(CameraPipeline & pipeline)
{
  RCLCPP_INFO(get_logger(), "[%s] Process thread started", pipeline.desc.display_name());

  while (rclcpp::ok() && !stop_capture_ && pipeline.frame_state->processing_active) {
    nvsipl::INvSIPLClient::INvSIPLBuffer * buffer = nullptr;
    std::chrono::steady_clock::time_point isp0_start_time;

    {
      std::unique_lock<std::mutex> lock(pipeline.frame_state->mutex);
      bool acquired = pipeline.frame_state->cv.wait_for(
        lock, std::chrono::microseconds(SiplCameraNode::kSiplProcessWaitTimeoutUs),
        [&pipeline]() {
          return pipeline.frame_state->pending_buffer != nullptr ||
                 !pipeline.frame_state->processing_active || !rclcpp::ok();
      });

      if (!acquired) {
        RCLCPP_DEBUG_THROTTLE(get_logger(), *get_clock(), 1000,
          "[Process] [%s] Waiting for buffer from capture thread",
          pipeline.desc.display_name());
        continue;
      }

      if (!pipeline.frame_state->processing_active || !rclcpp::ok()) {
        RCLCPP_DEBUG(get_logger(),
          "[Process] [%s] Exiting - processing_active=%d, rclcpp_ok=%d",
          pipeline.desc.display_name(),
          pipeline.frame_state->processing_active.load(), rclcpp::ok());
        break;
      }

      if (pipeline.frame_state->pending_buffer != nullptr) {
        buffer = pipeline.frame_state->pending_buffer;
        pipeline.frame_state->pending_buffer = nullptr;
        isp0_start_time = pipeline.frame_state->isp0_start_time;
        RCLCPP_DEBUG(get_logger(), "[Process] [%s] Received buffer %p from capture thread",
          pipeline.desc.display_name(), static_cast<void *>(buffer));
      } else {
        RCLCPP_DEBUG(get_logger(),
          "[Process] [%s] No buffer received from capture thread",
          pipeline.desc.display_name());
      }
    }

    if (buffer != nullptr) {
      auto * nvmm_buffer = dynamic_cast<nvsipl::INvSIPLClient::INvSIPLNvMBuffer *>(buffer);

      if (nvmm_buffer != nullptr) {
        try {
          const bool should_release = processFrame(nvmm_buffer, pipeline, isp0_start_time);
          if (should_release) {
            buffer->Release();
            buffer = nullptr;
          } else {
            // Ownership transferred to NitrosImage release callback.
            buffer = nullptr;
          }
        } catch (const std::exception & e) {
          RCLCPP_ERROR(get_logger(), "[%s] Frame processing failed: %s",
            pipeline.desc.display_name(), e.what());
        }
      } else {
        RCLCPP_ERROR(get_logger(),
          "[Process] [%s] dynamic_cast to INvSIPLNvMBuffer failed for buffer %p",
          pipeline.desc.display_name(), static_cast<void *>(buffer));
      }

      // If we still own the buffer (non-NvM, or exception path), release it.
      if (buffer != nullptr) {
        buffer->Release();
        buffer = nullptr;
      }
    }
  }

  // Cleanup any pending buffer if we are exiting
  {
    std::lock_guard<std::mutex> lock(pipeline.frame_state->mutex);
    if (pipeline.frame_state->pending_buffer != nullptr) {
      pipeline.frame_state->pending_buffer->Release();
      pipeline.frame_state->pending_buffer = nullptr;
    }
  }

  RCLCPP_INFO(get_logger(), "[%s] Process thread exited", pipeline.desc.display_name());
}

void SiplCameraNode::handleNotificationQueue(CameraPipeline & pipeline)
{
  RCLCPP_INFO(get_logger(), "[%s] Event thread started", pipeline.desc.display_name());

  nvsipl::NvSIPLPipelineNotifier::NotificationData event;

  while (rclcpp::ok() && !stop_capture_) {
    if (!pipeline.queues.notificationQueue) {
      RCLCPP_ERROR(get_logger(), "[%s] Notification queue is null",
        pipeline.desc.display_name());
      break;
    }

    nvsipl::SIPLStatus status = pipeline.queues.notificationQueue->Get(
      event, SiplCameraNode::kSiplQueueGetTimeoutUs);

    if (status == nvsipl::NVSIPL_STATUS_OK) {
      handlePipelineNotification(event);
    } else if (status == nvsipl::NVSIPL_STATUS_TIMED_OUT) {
      // Timeout is normal
    } else if (status == nvsipl::NVSIPL_STATUS_EOF) {
      RCLCPP_INFO(get_logger(), "[%s] Notification queue EOF received",
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

bool SiplCameraNode::validateBufferFormat(const BufferAttributes & attrs) const
{
  if (encoding_desired_ != "nv12" && encoding_desired_ != "nv24") {
    RCLCPP_ERROR(
      get_logger(), "Unsupported encoding '%s' (expected nv12 or nv24)",
      encoding_desired_.c_str());
    return false;
  }

  if (attrs.plane_count != 2U) {
    RCLCPP_ERROR(
      get_logger(), "Unsupported plane count %u for encoding '%s' (expected 2)",
      attrs.plane_count, encoding_desired_.c_str());
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
      encoding_desired_.c_str(),
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
  const bool looks_nv12 = (static_cast<uint64_t>(uv_height) * 2U == y_height);
  const bool looks_nv24 = (uv_height == y_height);
  const char * detected = looks_nv24 ? "nv24" : (looks_nv12 ? "nv12" : "unknown");

  RCLCPP_DEBUG_THROTTLE(
    get_logger(), *get_clock(), 5000,
    "Detected buffer layout: y=%ux%u uv=%ux%u (detected %s, desired %s)",
    y_width, y_height, uv_width, uv_height, detected, encoding_desired_.c_str());

  if (encoding_desired_ == "nv12" && !looks_nv12) {
    RCLCPP_ERROR_THROTTLE(
      get_logger(), *get_clock(), 2000,
      "Buffer layout mismatch for nv12: y=%ux%u uv=%ux%u",
      y_width, y_height, uv_width, uv_height);
    return false;
  }
  if (encoding_desired_ == "nv24" && !looks_nv24) {
    RCLCPP_ERROR_THROTTLE(
      get_logger(), *get_clock(), 2000,
      "Buffer layout mismatch for nv24: y=%ux%u uv=%ux%u",
      y_width, y_height, uv_width, uv_height);
    return false;
  }

  return true;
}

bool SiplCameraNode::processFrame(
  nvsipl::INvSIPLClient::INvSIPLNvMBuffer * sipl_buffer,
  CameraPipeline & pipeline,
  std::chrono::steady_clock::time_point isp0_start_time)
{
  auto process_start = std::chrono::steady_clock::now();

  if (enable_debug_logs_) {
    auto handoff_us = std::chrono::duration_cast<std::chrono::microseconds>(
      process_start - isp0_start_time).count();
    RCLCPP_DEBUG(get_logger(),
      "[Process] [%s] Time from ISP start to processFrame entry: %ld us",
      pipeline.desc.display_name(), handoff_us);
  }

  if (sipl_buffer == nullptr) {
    RCLCPP_ERROR(get_logger(), "[%s] Null SIPL buffer", pipeline.desc.display_name());
    return true;
  }

  // Retrieve the EOF fence from the SIPL buffer. This is a lightweight metadata
  // extraction (no waiting). The fence is passed to mapNvmmToCuda() which uses
  // it for GPU-side synchronization via cudaWaitExternalSemaphoresAsync().
  NvSciSyncFence fence = NvSciSyncFenceInitializer;
  nvsipl::SIPLStatus status = sipl_buffer->GetEOFNvSciSyncFence(&fence);
  if (status != nvsipl::NVSIPL_STATUS_OK) {
    RCLCPP_ERROR(get_logger(), "[%s] Failed to get EOF fence", pipeline.desc.display_name());
    return true;
  }

  CudaDevicePtr gpu_ptr = nullptr;
  size_t gpu_size = 0U;
  BufferAttributes attrs{};
  status = pipeline.buffer_manager_isp0->mapNvmmToCuda(
    sipl_buffer, fence, cpu_wait_context_, &gpu_ptr, &gpu_size, &attrs);

  if (status != nvsipl::NVSIPL_STATUS_OK || gpu_ptr == nullptr || gpu_size == 0U) {
    RCLCPP_ERROR(get_logger(), "[%s] Failed to map NVMM to CUDA",
      pipeline.desc.display_name());
    return true;
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
    RCLCPP_DEBUG(get_logger(),
      "[Process] [%s] ISP start to frame ready and cpu sync'ed latency: %ld us",
      pipeline.desc.display_name(), isp_latency_us);
  }

  if (!validateBufferFormat(attrs)) {
    return true;
  }

  std_msgs::msg::Header header;
  if (use_hw_timestamp_) {
    uint64_t timestamp = extractTscTimestamp(sipl_buffer);
    header.stamp = convertTscToRosTime(timestamp);
  } else {
    header.stamp = now();
  }
  header.frame_id = pipeline.desc.frame_id;

  // Build NitrosImage. Ownership of gpu_buffer transfers to NITROS.
  nvidia::isaac_ros::nitros::NitrosImage nitros_image =
    nvidia::isaac_ros::nitros::NitrosImageBuilder()
    .WithHeader(header)
    .WithEncoding(encoding_desired_)
    .WithDimensions(image_height_, image_width_)
    .WithGpuData(gpu_ptr)
    .WithReleaseCallback([]() {
      // No-op: prevent release callback to avoid cudaFree on NvSci-mapped ptr from
      // at VideoBuffer::wrapMemory.
      // Release handled by wrapMemory below
      })
    .Build();

  // The SIPL buffer has padding bytes between the Y plane and the UV plane for alignment.
  // The Managed NITROS library (used by the Type Adapter) assumes planes are packed contiguously.
  // It calculates the UV plane address as Base_Address + Y_Plane_Size. Correct this by manually
  // updating the VideoBufferInfo.
  {
    auto context = nvidia::isaac_ros::nitros::GetTypeAdapterNitrosContext().getContext();
    auto message = nvidia::gxf::Entity::Shared(context, nitros_image.handle);
    if (!message) {
      RCLCPP_ERROR(get_logger(), "[%s] Failed to get GXF entity from NitrosImage",
        pipeline.desc.display_name());
      return true;  // Caller releases SIPL buffer
    }
    auto video_buffer_handle = message.value().get<nvidia::gxf::VideoBuffer>();
    if (!video_buffer_handle) {
      RCLCPP_ERROR(get_logger(), "[%s] Failed to get VideoBuffer from GXF entity",
        pipeline.desc.display_name());
      return true;  // Caller releases SIPL buffer
    }
    const auto & video_buffer = video_buffer_handle.value();
    nvidia::gxf::VideoBufferInfo info = video_buffer->video_frame_info();
    RCLCPP_DEBUG(get_logger(),
      "[Process] NITROS video buffer attributes: width: %u, height: %u, "
      "color_format: %s, color_planes: %zu, surface_layout: %s",
      info.width, info.height,
      std::string(magic_enum::enum_name(info.color_format)).c_str(),
      info.color_planes.size(),
      std::string(magic_enum::enum_name(info.surface_layout)).c_str());
    for (size_t i = 0; i < info.color_planes.size(); ++i) {
      RCLCPP_DEBUG(get_logger(),
        "[Process] Color plane %zu: stride: %u, offset: %u, size: %lu", i,
        info.color_planes[i].stride, info.color_planes[i].offset,
        info.color_planes[i].size);
    }

    // Get attributes from SIPL Buffer (via attrs) and set the actual stride,
    // offset, and size to the video msg info.
    for (size_t i = 0; i < info.color_planes.size() && i < attrs.plane_count; ++i) {
      info.color_planes[i].stride = attrs.plane_pitches[i];
      info.color_planes[i].offset = attrs.plane_offsets[i];
      // Set the size of the Y plane (Plane 0) to be equal to the offset of the
      // UV plane (Plane 1). This ensures that the Managed NITROS library (which
      // uses Base + Size to find the next plane) calculates the correct starting
      // address for the UV data, accounting for any hardware padding.
      if (i == 0 && info.color_planes.size() > 1 && attrs.plane_count > 1) {
        info.color_planes[i].size = attrs.plane_offsets[1];
      } else {
      // NvSciBuf provides a per-plane pitch (stride in bytes per row). Pitch is the
      // authoritative row size because it includes alignment/padding the producer must honor
        info.color_planes[i].size = info.color_planes[i].stride *
          info.color_planes[i].height;
      }
      RCLCPP_DEBUG(get_logger(),
            "[Process] Updated color plane %zu: stride: %u, offset: %u, size: %lu",
          i, info.color_planes[i].stride, info.color_planes[i].offset,
          info.color_planes[i].size);
    }

    // Release callback to keep SIPL buffer alive until downstream is done with the GPU pointer.
    auto release_callback = [buffer =
        static_cast<nvsipl::INvSIPLClient::INvSIPLBuffer *>(sipl_buffer),
        buffer_manager = pipeline.buffer_manager_isp0](
      void *) -> nvidia::gxf::Expected<void> {  // NOSONAR
        if (buffer != nullptr) {
          buffer->Release();
        }
        return nvidia::gxf::Success;
      };

    // Re-wrap memory with updated info and a release callback.
    video_buffer->wrapMemory(
              info,
              gpu_size,
              nvidia::gxf::MemoryStorageType::kDevice,
              gpu_ptr,
              release_callback
    );
  }

  // Publish image via Managed NITROS
  pipeline.image_pub->publish(nitros_image);

  // Publish camera info with same timestamp
  publishCameraInfo(pipeline, header);

  if (enable_debug_logs_) {
    auto process_end = std::chrono::steady_clock::now();
    auto process_duration_us = std::chrono::duration_cast<std::chrono::microseconds>(
      process_end - process_start).count();

    auto now_ros = this->now();
    uint64_t raw_tsc = extractTscTimestamp(sipl_buffer);
    auto capture_timestamp = convertTscToRosTime(raw_tsc);
    int64_t latency_us = (now_ros.nanoseconds() - capture_timestamp.nanoseconds()) / 1000;

    RCLCPP_DEBUG(get_logger(),
      "[Process] [%s] gpu_ptr=%p, size=%zu, "
      "process_duration=%ld us, capture_to_publish_latency=%ld us, "
      "raw_tsc=%" PRIu64 ", now_ros=%" PRId64 ", converted_tsc=%" PRId64 ", offset=%" PRId64,
      pipeline.desc.display_name(), gpu_ptr, gpu_size,
      process_duration_us, latency_us,
      raw_tsc, now_ros.nanoseconds(), capture_timestamp.nanoseconds(),
      tsc_to_ros_offset_);
  }

  // Ownership of the SIPL buffer was transferred to the NitrosImage release callback.
  return false;
}

void SiplCameraNode::publishCameraInfo(
  const CameraPipeline & pipeline, const std_msgs::msg::Header & header)
{
  sensor_msgs::msg::CameraInfo info;
  if (pipeline.desc.camera_info_loaded) {
    info = pipeline.desc.camera_info;
  } else {
    info.width = image_width_;
    info.height = image_height_;
    info.distortion_model = "";
    info.d.clear();
    info.k.fill(0.0);
    info.r.fill(0.0);
    info.p.fill(0.0);
  }
  info.header = header;
  pipeline.camera_info_pub->publish(info);
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


void SiplCameraNode::updateTscOffset()
{
  const auto now_steady = std::chrono::steady_clock::now();
  if (now_steady - tsc_offset_last_calibration_ < kTscRecalibrationInterval) {
    return;
  }

  // frameCaptureTSC is stamped by the RTCPU firmware reading CNTVCT_EL0
  // (the ARM Generic Timer) directly which is passed downstream to the SIPL image metadata.
  // So we read the TSC and ROS timestamp closely to get the offset between the time domains
  // for timestamp conversion.
  // This clock read method is used by drivers in the linux-nv-oot repository.
  uint64_t cntvct;
  asm volatile ("mrs %0, cntvct_el0" : "=r" (cntvct));
  const int64_t ros_ns = now().nanoseconds();
  tsc_to_ros_offset_ = ros_ns - static_cast<int64_t>(cntvct);

  RCLCPP_INFO(
    get_logger(),
    "TSC offset updated offset=%" PRId64 " ns", tsc_to_ros_offset_);

  tsc_offset_last_calibration_ = now_steady;
}

rclcpp::Time SiplCameraNode::convertTscToRosTime(uint64_t tsc_timestamp)
{
  const int64_t ros_ns = static_cast<int64_t>(tsc_timestamp) + tsc_to_ros_offset_;
  return rclcpp::Time(ros_ns);
}

uint64_t SiplCameraNode::extractTscTimestamp(
  nvsipl::INvSIPLClient::INvSIPLNvMBuffer * buffer)
{
  const nvsipl::INvSIPLClient::ImageMetaData & metadata = buffer->GetImageData();
  // frameCaptureTSC is the end of frame timestamp and is in TSC time domain
  // with where each tick is 1 ns on Thor.
  // (Reference in fusa repo: capture/src/fusaCoeChannelLinux.cpp::getCaptureStatus for CoE)
  // todo: Handle Orin platform case to scale down ticks by 32 when Orin is supported.
  return metadata.frameCaptureTSC;
}

std::string SiplCameraNode::getNitrosFormatFromEncoding()
{
  // Map encoding to NITROS format string
  if (encoding_desired_ == "nv12") {
    return nvidia::isaac_ros::nitros::nitros_image_nv12_t::supported_type_name;
  } else if (encoding_desired_ == "nv24") {
    return nvidia::isaac_ros::nitros::nitros_image_nv24_t::supported_type_name;
  } else {
    RCLCPP_ERROR(
      get_logger(), "Unsupported encoding: %s", encoding_desired_.c_str());
    throw std::invalid_argument("Unsupported encoding: " + encoding_desired_);
  }
}

NvSciBufSurfSampleType SiplCameraNode::getSurfSampleTypeFromEncoding() const
{
  static const std::unordered_map<std::string, NvSciBufSurfSampleType>
  kEncodingToSampleType = {
    {"nv12", NvSciSurfSampleType_420},
    {"nv24", NvSciSurfSampleType_444}
  };

  const auto it = kEncodingToSampleType.find(encoding_desired_);
  if (it != kEncodingToSampleType.end()) {
    return it->second;
  }

  RCLCPP_ERROR(
    get_logger(), "Unsupported encoding: %s", encoding_desired_.c_str());
  throw std::invalid_argument("Unsupported encoding: " + encoding_desired_);
}

void SiplCameraNode::parseMacAddress(const std::string & mac_str, uint8_t mac_bytes[6])
{
  std::istringstream ss(mac_str);
  std::string byte_str;
  uint32_t i = 0;

  while (std::getline(ss, byte_str, ':') && i < 6) {
    mac_bytes[i++] = static_cast<uint8_t>(std::stoi(byte_str, nullptr, 16));
  }

  if (i != 6) {
    throw std::invalid_argument("Invalid MAC address format, expected XX:XX:XX:XX:XX:XX");
  }
}

uint32_t SiplCameraNode::parseIpAddress(const std::string & ip_str)
{
  struct in_addr addr;
  if (inet_aton(ip_str.c_str(), &addr) == 1) {
    return addr.s_addr;
  }

  throw std::invalid_argument("Invalid IP address format, expected XXX.XXX.XXX.XXX");
}

std::vector<uint8_t> SiplCameraNode::loadNitoFile(const std::string & module_name)
{
  if (nito_path_.empty()) {
    RCLCPP_ERROR(get_logger(), "NITO path is empty; set the 'nito_path' parameter");
    return {};
  }

  std::string module_name_lower;
  for (auto & c : module_name) {
    module_name_lower.push_back(std::tolower(c));
  }

  std::string base_path = nito_path_;
  if (base_path.back() != '/') {
    base_path += "/";
  }

  std::vector<std::string> candidates = {
    base_path + module_name + ".nito",
    base_path + module_name_lower + ".nito"
  };

  std::string selected_nito_file;
  FILE * fp = nullptr;

  for (const auto & candidate : candidates) {
    fp = fopen(candidate.c_str(), "rb");
    if (fp) {
      selected_nito_file = candidate;
      RCLCPP_INFO(get_logger(), "Found NITO file: %s", candidate.c_str());
      break;
    }
  }

  if (!fp) {
    RCLCPP_ERROR(
      get_logger(),
      "Could not find NITO file for module '%s' in directory: %s",
      module_name.c_str(), nito_path_.c_str());
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
    RCLCPP_ERROR(get_logger(), "Failed to read NITO file (read %zu of %" PRId64 " bytes)",
      result, fsize);
    return {};
  }

  return nito_data;
}

void SiplCameraNode::handlePipelineNotification(
  const nvsipl::NvSIPLPipelineNotifier::NotificationData & event)
{
  using NotifType = nvsipl::NvSIPLPipelineNotifier::NotificationType;

  switch (event.eNotifType) {
    case NotifType::NOTIF_INFO_ICP_PROCESSING_DONE:
      RCLCPP_DEBUG(get_logger(), "[Sipl Notification] ICP processing done on sensor %u",
          event.uIndex);
      break;
    case NotifType::NOTIF_INFO_ISP_PROCESSING_DONE:
      RCLCPP_DEBUG(get_logger(), "[Sipl Notification] ISP processing done on sensor %u",
          event.uIndex);
      break;
    case NotifType::NOTIF_WARN_ICP_FRAME_DROP:
      RCLCPP_WARN(get_logger(),
          "[Sipl Notification] ICP Frame drop detected on sensor %u (pipeline full?)",
          event.uIndex);
      break;
    case NotifType::NOTIF_WARN_ICP_FRAME_DISCONTINUITY:
      RCLCPP_WARN(
        get_logger(), "[Sipl Notification] Frame discontinuity on sensor %u, frame %lu",
        event.uIndex, event.frameSeqNumber);
      break;
    case NotifType::NOTIF_WARN_ICP_CAPTURE_TIMEOUT:
      RCLCPP_WARN(get_logger(), "[Sipl Notification] Capture timeout on sensor %u", event.uIndex);
      break;
    case NotifType::NOTIF_WARN_ICP_APP_BACK_PRESSURE:
      RCLCPP_WARN(get_logger(), "[Sipl Notification] App back pressure on sensor %u", event.uIndex);
      break;
    case NotifType::NOTIF_ERROR_ICP_CAPTURE_FAILURE:
      RCLCPP_ERROR(get_logger(), "[Sipl Notification] Capture failure on sensor %u", event.uIndex);
      break;
    case NotifType::NOTIF_ERROR_ICP_EMB_DATA_PARSE_FAILURE:
      RCLCPP_ERROR(get_logger(), "[Sipl Notification] Embedded data parse failure on sensor %u",
          event.uIndex);
      break;
    case NotifType::NOTIF_ERROR_ISP_PROCESSING_FAILURE:
      RCLCPP_ERROR(get_logger(), "[Sipl Notification] ISP processing failure on sensor %u",
          event.uIndex);
      break;
    case NotifType::NOTIF_ERROR_INTERNAL_FAILURE:
      RCLCPP_ERROR(get_logger(), "[Sipl Notification] Internal failure on sensor %u", event.uIndex);
      break;
    case NotifType::NOTIF_ERROR_ACP_PROCESSING_FAILURE:
      RCLCPP_ERROR(get_logger(), "[Sipl Notification] Auto control processing failure on sensor %u",
          event.uIndex);
      break;
    case NotifType::NOTIF_ERROR_CDI_SET_SENSOR_CTRL_FAILURE:
      RCLCPP_ERROR(get_logger(), "[Sipl Notification] Sensor control write failure on sensor %u",
          event.uIndex);
      break;
    default:
      RCLCPP_WARN(
        get_logger(), "[Sipl Notification] Pipeline event %u on sensor %u",
        static_cast<uint32_t>(event.eNotifType), event.uIndex);
      break;
  }
}

}  // namespace sipl
}  // namespace isaac_ros

// Register as component
#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(isaac_ros::sipl::SiplCameraNode)
