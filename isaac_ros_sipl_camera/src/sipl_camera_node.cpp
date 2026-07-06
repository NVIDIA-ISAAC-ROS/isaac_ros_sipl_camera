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

#include "tf2/LinearMath/Quaternion.h"

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
  output_qos_(::isaac_ros::common::AddQosParameter(*this, "DEFAULT", "output_qos", 1)),
  sci_sync_module_(nullptr),
  tsc_correlator_(this->get_clock(), this->get_logger())
{
  tsc_correlator_.setRecalibrationInterval(std::chrono::milliseconds{1000});
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

  output_buffer_pool_size_ = declare_parameter<int>("output_buffer_pool_size", 10);

  // Create TF broadcaster
  tf_static_broadcaster_ = std::make_unique<tf2_ros::StaticTransformBroadcaster>(this);

  cuda_stream_ = ::nvidia::isaac_ros::common::createCudaStream("SiplCameraNode");

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
    pipelines_.emplace_back();
    CameraPipeline & pipeline = pipelines_.back();
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
    RCLCPP_DEBUG(get_logger(),
      "[%s] ISP0 buffer layout: Y ends at %zu, UV starts at %" PRIu64 " — %s",
      pipeline.desc.display_name(), y_end, init_attrs.plane_offsets[1],
      pipeline.needs_compaction ? "compaction required" : "already compact");

    const size_t y_plane_bytes =
      static_cast<size_t>(init_attrs.plane_pitches[0]) * init_attrs.plane_heights[0];
    const size_t uv_plane_bytes =
      static_cast<size_t>(init_attrs.plane_pitches[1]) * init_attrs.plane_heights[1];
    pipeline.compact_frame_bytes = y_plane_bytes + uv_plane_bytes;

    RCLCPP_INFO(get_logger(), "[%s] Registering Auto Control Plugin (sensor %u)",
      pipeline.desc.display_name(), pipeline.sensor_id);
    registerAutoControlPlugin(pipeline.desc.camera_index);

    RCLCPP_INFO(get_logger(), "[%s] Creating publishers...",
      pipeline.desc.display_name());
    createPublisherForPipeline(pipeline);
  }

  // Pre-allocate a fixed pool of GPU padding removed frame buffers per pipeline.
  if (output_buffer_pool_size_ <= 0) {
    throw std::runtime_error(
      "output_buffer_pool_size must be > 0, got " +
      std::to_string(output_buffer_pool_size_));
  }
  const size_t pool_size = static_cast<size_t>(output_buffer_pool_size_);
  for (auto & pipeline : pipelines_) {
    const size_t compact_frame_bytes = pipeline.compact_frame_bytes;
    pipeline.compact_pool = std::make_unique<nvidia::isaac_ros::nitros::CUDAMemoryPool>();
    cudaError_t err = pipeline.compact_pool->create(
      compact_frame_bytes, pool_size,
      nvidia::isaac_ros::nitros::CUDAMemoryPool::MemoryType::Device);
    if (err != cudaSuccess) {
      throw std::runtime_error(
        "[" + std::string(pipeline.desc.display_name()) +
        "] Failed to create output buffer pool: " + cudaGetErrorString(err));
    }
    RCLCPP_INFO(get_logger(),
      "[%s] Pre-allocated output buffer pool: %zu x %zu bytes = %zu bytes",
      pipeline.desc.display_name(), pool_size, compact_frame_bytes,
      pool_size * compact_frame_bytes);
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

  status = pipeline.buffer_manager_isp0->allocateAndRegisterBuffers(
    sipl_camera_.get(),
    pipeline.sensor_id,
    nvsipl::INvSIPLClient::ConsumerDesc::OutputType::ISP0,
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

  rclcpp::PublisherOptions pub_options;
  pub_options.use_intra_process_comm = rclcpp::IntraProcessSetting::Enable;
  pipeline.image_pub = create_publisher<nvidia::isaac_ros::nitros::NitrosImage>(
    image_topic, output_qos_, pub_options);
  RCLCPP_DEBUG(get_logger(), "[%s] Publisher QoS: depth=%zu, reliability=%s",
    pipeline.desc.display_name(), output_qos_.get_rmw_qos_profile().depth,
    output_qos_.get_rmw_qos_profile().reliability == RMW_QOS_POLICY_RELIABILITY_BEST_EFFORT ?
    "best_effort" : "reliable");

  pipeline.camera_info_pub = create_publisher<sensor_msgs::msg::CameraInfo>(
    camera_info_topic, output_qos_);
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
    if (pipeline.event_thread.joinable()) {
      pipeline.event_thread.join();
    }
  }

  for (const auto & pipeline : pipelines_) {
    RCLCPP_INFO(
      get_logger(),
      "[%s] Pool exhaustion drops=%" PRIu64,
      pipeline.desc.display_name(),
      pipeline.dropped_pool_exhausted);
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
    sensor_id, module_name.c_str());
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

  // Apply to camera configurations to the target HSB
  bool applied_to_any_camera = false;
  for (auto & camera : camera_system_config_.cameras) {
    if (std::holds_alternative<nvsipl::CoECamera>(camera.cameratype)) {
      auto & coe_cam = std::get<nvsipl::CoECamera>(camera.cameratype);
      if (coe_cam.hsbId == static_cast<uint32_t>(hsb_id_)) {
        RCLCPP_INFO(get_logger(), "Applying network overrides to camera %s on HSB %d",
                    camera.sensorInfo.name.c_str(), hsb_id_);
        if (has_ip) {
          coe_cam.sensors->ipAddress = ip_value;
        }
        if (has_mac) {
          memcpy(coe_cam.sensors->macAddress, mac_bytes, 6);
        }
        applied_to_any_camera = true;
      }
    }
  }

  if (!applied_to_any_camera) {
    RCLCPP_ERROR(get_logger(), "No cameras found on HSB %d. Network overrides skipped.", hsb_id_);
  }
}


void SiplCameraNode::pipelineThread(CameraPipeline & pipeline)
{
  RCLCPP_INFO(get_logger(), "[%s] Pipeline thread started (sensor %u)",
    pipeline.desc.display_name(), pipeline.sensor_id);

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
    // time spent blocked in Get() + remaining DMA flush (EOF fence).
    auto isp0_start_time = std::chrono::steady_clock::now();

    // Get ISP0 processed buffer.
    nvsipl::INvSIPLClient::INvSIPLBuffer * buffer = nullptr;
    status =
      pipeline.queues.isp0CompletionQueue->Get(buffer, SiplCameraNode::kSiplQueueGetTimeoutUs);

    if (status == nvsipl::NVSIPL_STATUS_OK && buffer != nullptr) {
      if (enable_debug_logs_) {
        auto now = std::chrono::steady_clock::now();
        auto elapsed_us = std::chrono::duration_cast<std::chrono::microseconds>(
          now - last_frame_time).count();
        if (elapsed_us > 0) {
          double fps = 1000000.0 / static_cast<double>(elapsed_us);
          RCLCPP_DEBUG(get_logger(), "[%s] Instantaneous FPS: %.2f",
            pipeline.desc.display_name(), fps);
        }
        last_frame_time = now;
      }

      auto * nvmm_buffer = dynamic_cast<nvsipl::INvSIPLClient::INvSIPLNvMBuffer *>(buffer);
      if (nvmm_buffer != nullptr) {
        try {
          processFrame(nvmm_buffer, pipeline, isp0_start_time);
        } catch (const std::exception & e) {
          RCLCPP_ERROR(get_logger(), "[%s] Frame processing failed: %s",
            pipeline.desc.display_name(), e.what());
          buffer->Release();
        }
      } else {
        RCLCPP_ERROR(get_logger(),
          "[%s] dynamic_cast to INvSIPLNvMBuffer failed for buffer %p",
          pipeline.desc.display_name(), static_cast<void *>(buffer));
        buffer->Release();
      }

      size_t queue_depth = pipeline.queues.isp0CompletionQueue->GetCount();
      if (queue_depth > 1) {
        RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 1000,
          "[%s] ISP0 queue depth high (%zu). Possible backlog.",
          pipeline.desc.display_name(), queue_depth);
      }
    } else if (status == nvsipl::NVSIPL_STATUS_TIMED_OUT) {
      RCLCPP_DEBUG_THROTTLE(get_logger(), *get_clock(), 1000,
        "[%s] Waiting for frames (queue timeout, no data from sensor)",
        pipeline.desc.display_name());
    } else if (status == nvsipl::NVSIPL_STATUS_EOF) {
      RCLCPP_INFO(get_logger(), "[%s] ISP0 queue EOF received",
        pipeline.desc.display_name());
      break;
    } else {
      RCLCPP_WARN(get_logger(), "[%s] Failed to get ISP0 buffer (status=%u)",
        pipeline.desc.display_name(), static_cast<uint32_t>(status));
    }
  }

  RCLCPP_INFO(get_logger(), "[%s] Pipeline thread exited", pipeline.desc.display_name());
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

bool SiplCameraNode::validateBufferFormat(
  const BufferAttributes & attrs,
  const CameraPipeline & pipeline) const
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
  const bool looks_nv12 = (static_cast<uint64_t>(uv_height) * 2U == y_height) &&
    (static_cast<uint64_t>(uv_width) * 2U == y_width);
  const bool looks_nv24 = (uv_height == y_height) && (uv_width == y_width);
  const char * detected = looks_nv24 ? "nv24" : (looks_nv12 ? "nv12" : "unknown");

  if (enable_debug_logs_) {
    for (uint32_t i = 0; i < attrs.plane_count; ++i) {
      RCLCPP_DEBUG(get_logger(),
        "Plane %u: width=%u, height=%u, pitch=%u, "
        "offset=%" PRIu64 ", bpp=%u",
        i, attrs.plane_widths[i], attrs.plane_heights[i], attrs.plane_pitches[i],
        attrs.plane_offsets[i], attrs.plane_bits_per_pixels[i]);
    }
  }

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
    (encoding_desired_ == "nv24") ? attrs.plane_pitches[0] * 2U :
    attrs.plane_pitches[0];
  if (attrs.plane_pitches[1] != expected_uv_pitch) {
    RCLCPP_ERROR_THROTTLE(
      get_logger(), *get_clock(), 2000,
      "[%s] nvscibuf UV pitch (%u) violates the V4L2 %s stride relation that "
      "ROS2 sensor_msgs/Image inherits via image_encodings.hpp; expected UV "
      "pitch %u (Y pitch=%u). Adjust NvSciBufImageAttrKey_PlanePitchAlign "
      "so the chroma pitch matches the V4L2 rule.",
      pipeline.desc.display_name(),
      attrs.plane_pitches[1], encoding_desired_.c_str(),
      expected_uv_pitch, attrs.plane_pitches[0]);
    return false;
  }

  return true;
}

void SiplCameraNode::processFrame(
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
    return;
  }

  // Retrieve the EOF fence from the SIPL buffer. This is a lightweight metadata
  // extraction (no waiting). The fence is passed to mapNvmmToCuda() which uses
  // it for GPU-side synchronization via cudaWaitExternalSemaphoresAsync().
  NvSciSyncFence fence = NvSciSyncFenceInitializer;
  nvsipl::SIPLStatus status = sipl_buffer->GetEOFNvSciSyncFence(&fence);
  if (status != nvsipl::NVSIPL_STATUS_OK) {
    RCLCPP_ERROR(get_logger(), "[%s] Failed to get EOF fence", pipeline.desc.display_name());
    sipl_buffer->Release();
    return;
  }

  CudaDevicePtr gpu_ptr = nullptr;
  size_t gpu_size = 0U;
  BufferAttributes attrs{};
  status = pipeline.buffer_manager_isp0->mapNvmmToCuda(
    sipl_buffer, fence, cpu_wait_context_, &gpu_ptr, &gpu_size, &attrs);

  if (status != nvsipl::NVSIPL_STATUS_OK || gpu_ptr == nullptr || gpu_size == 0U) {
    RCLCPP_ERROR(get_logger(), "[%s] Failed to map NVMM to CUDA",
      pipeline.desc.display_name());
    sipl_buffer->Release();
    return;
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

  if (!validateBufferFormat(attrs, pipeline)) {
    sipl_buffer->Release();
    return;
  }

  std_msgs::msg::Header header;
  uint64_t raw_tsc = 0;
  if (use_hw_timestamp_) {
    raw_tsc = extractTscTimestamp(sipl_buffer);
    header.stamp = tsc_correlator_.tscToRos(raw_tsc);
  } else {
    header.stamp = now();
  }
  header.frame_id = pipeline.desc.frame_id;

  // Compact the Y and UV planes into a pre-allocated buffer, removing the
  // hardware inter-plane padding gap.
  //   NV12: Y stride=pitch_Y, UV stride=pitch_UV, UV height=H/2
  //   NV24: Y stride=pitch_Y, UV stride=pitch_UV, UV height=H
  const size_t y_size = static_cast<size_t>(attrs.plane_pitches[0]) * attrs.plane_heights[0];
  const size_t uv_size = static_cast<size_t>(attrs.plane_pitches[1]) * attrs.plane_heights[1];
  const size_t compact_size = y_size + uv_size;

  // Acquire a pre-allocated buffer from the per-pipeline pool via from_pool(),
  // copy the compacted planes, then release the SIPL source buffer.
  // If all buffers are held by downstream consumers, from_pool() throws
  // and we catch the exception to drop the frame.
  nvidia::isaac_ros::nitros::NitrosImage image_msg;
  try {
    auto write_handle = image_msg.from_pool(
      *pipeline.compact_pool, image_width_, image_height_,
      attrs.plane_pitches[0], encoding_desired_, *cuda_stream_);

    uint8_t * compact_ptr = write_handle.get_ptr();
    cudaError_t cuda_err;

    if (pipeline.needs_compaction) {
      cuda_err = cudaMemcpyAsync(compact_ptr, gpu_ptr, y_size,
        cudaMemcpyDeviceToDevice, *cuda_stream_);
      if (cuda_err != cudaSuccess) {
        RCLCPP_ERROR(get_logger(), "[%s] Y plane copy failed: %s",
          pipeline.desc.display_name(), cudaGetErrorString(cuda_err));
        sipl_buffer->Release();
        return;
      }

      cuda_err = cudaMemcpyAsync(
        compact_ptr + y_size,
        static_cast<const uint8_t *>(gpu_ptr) + attrs.plane_offsets[1],
        uv_size, cudaMemcpyDeviceToDevice, *cuda_stream_);
      if (cuda_err != cudaSuccess) {
        RCLCPP_ERROR(get_logger(), "[%s] UV plane copy failed: %s",
          pipeline.desc.display_name(), cudaGetErrorString(cuda_err));
        sipl_buffer->Release();
        return;
      }
    } else {
      cuda_err = cudaMemcpyAsync(compact_ptr, gpu_ptr, compact_size,
        cudaMemcpyDeviceToDevice, *cuda_stream_);
      if (cuda_err != cudaSuccess) {
        RCLCPP_ERROR(get_logger(), "[%s] Frame copy failed: %s",
          pipeline.desc.display_name(), cudaGetErrorString(cuda_err));
        sipl_buffer->Release();
        return;
      }
    }

    cuda_err = cudaStreamSynchronize(*cuda_stream_);
    if (cuda_err != cudaSuccess) {
      RCLCPP_ERROR(get_logger(), "[%s] cudaStreamSynchronize failed: %s",
        pipeline.desc.display_name(), cudaGetErrorString(cuda_err));
      sipl_buffer->Release();
      return;
    }
  } catch (const std::runtime_error &) {
    ++pipeline.dropped_pool_exhausted;
    RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 1000,
      "[%s] Output buffer pool exhausted (pool=%zu), dropping frame (total %" PRIu64 ")",
      pipeline.desc.display_name(), pipeline.compact_pool->block_count(),
      pipeline.dropped_pool_exhausted);
    sipl_buffer->Release();
    return;
  }

  // Data has been copied out of the SIPL buffer; release it immediately.
  sipl_buffer->Release();

  RCLCPP_DEBUG(get_logger(),
    "[Process] [%s] Compacted %s: Y %u×%u + UV %u×%u, "
    "gap removed %lu bytes, compact %zu bytes",
    pipeline.desc.display_name(), encoding_desired_.c_str(),
    attrs.plane_pitches[0], attrs.plane_heights[0],
    attrs.plane_pitches[1], attrs.plane_heights[1],
    attrs.plane_offsets[1] - y_size, compact_size);

  // Set metadata on the NitrosImage (from_pool doesn't accept a header).
  image_msg.timestamp_sec = header.stamp.sec;
  image_msg.timestamp_nsec = header.stamp.nanosec;
  image_msg.frame_id = header.frame_id;

  pipeline.image_pub->publish(image_msg);

  // Publish camera info with same timestamp
  publishCameraInfo(pipeline, header);

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

    RCLCPP_DEBUG(get_logger(),
      "[Process] [%s] image_size=%zu, "
      "process_duration=%ld us, capture_to_publish_latency=%ld us, "
      "raw_tsc=%" PRIu64 ", now_ros=%" PRId64,
      pipeline.desc.display_name(), compact_size,
      process_duration_us, latency_us,
      raw_tsc, now_ros.nanoseconds());
  }
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


uint64_t SiplCameraNode::extractTscTimestamp(
  nvsipl::INvSIPLClient::INvSIPLNvMBuffer * buffer)
{
  const nvsipl::INvSIPLClient::ImageMetaData & metadata = buffer->GetImageData();
  // frameCaptureTSC is the end of frame timestamp in raw TSC tick counts.
  // TscCorrelator converts ticks to nanoseconds using CNTFRQ_EL0 and applies
  // the calibrated TSC-to-ROS offset.
  return metadata.frameCaptureTSC;
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
