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

#ifndef ISAAC_ROS_SIPL_CAMERA__SIPL_CAMERA_NODE_HPP_
#define ISAAC_ROS_SIPL_CAMERA__SIPL_CAMERA_NODE_HPP_

#include <atomic>
#include <chrono>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include <cinttypes>

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/camera_info.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "camera_info_manager/camera_info_manager.hpp"
#include "tf2/LinearMath/Matrix3x3.h"
#include "tf2_ros/static_transform_broadcaster.h"

#include "isaac_ros_nitros_image_type/nitros_image.hpp"
#include "isaac_ros_nitros/types/cuda_memory_pool.hpp"

#include "isaac_ros_sipl_camera/sipl_buffer_manager.hpp"
#include "isaac_ros_sipl_camera/transport_adapter.hpp"
#include "isaac_ros_sipl_camera/tsc_correlator.hpp"
#include "isaac_ros_sipl_camera/stereo_timestamp_aligner.hpp"

#include "isaac_ros_common/cuda_stream.hpp"
#include "isaac_ros_common/qos.hpp"
#include "isaac_ros_sipl_camera/sipl_definitions.hpp"

namespace isaac_ros
{
namespace sipl
{

/**
 * @brief Base ROS2 node for SIPL camera integration
 *
 * This node provides direct SIPL API integration for monocular cameras,
 * publishing NITROS-accelerated images with GPU memory management.
 *
 * Subclasses override describePipelines() to
 * declare additional sensors and publishStaticTransforms() for custom TF trees.
 * All per-sensor runtime state is managed via the pipelines_ vector.
 */
class SiplCameraNode : public rclcpp::Node
{
public:
  explicit SiplCameraNode(const rclcpp::NodeOptions & options = rclcpp::NodeOptions());
  virtual ~SiplCameraNode();

protected:
  SiplCameraNode(const rclcpp::NodeOptions & options, bool defer_initialization);

  /// Struct to describe the pipeline.
  /// The name field doubles as the ROS topic prefix:
  /// "" for mono (topics are "image_raw", "camera_info"),
  /// "left"/"right" for stereo (topics are "left/image_raw", "left/camera_info").
  struct PipelineDescriptor
  {
    std::string name;
    uint32_t sensor_id;                   // SIPL global sensor id (CommonSensorConfig::id)
    std::string frame_id;                 // optical frame name for headers
    sensor_msgs::msg::CameraInfo camera_info;
    bool camera_info_loaded = false;

    /// Returns a human-readable name for log messages.
    const char * display_name() const {return name.empty() ? "camera" : name.c_str();}
  };

  /// Owns all per-sensor runtime state. The descriptor (moved in from
  /// describePipelines()) provides the configuration; remaining fields
  /// are populated during initialize().
  /// After initialize(), pipelines_ is never resized, so pointers/references
  /// to CameraPipeline members (e.g. queues) remain stable.
  struct CameraPipeline
  {
    PipelineDescriptor desc;
    uint32_t sensor_id = 0;
    nvsipl::NvSIPLPipelineQueues queues;
    NvSciSyncObj sci_sync_isp0 = nullptr;
    std::shared_ptr<SiplBufferManager> buffer_manager_icp;
    std::shared_ptr<SiplBufferManager> buffer_manager_isp0;
    std::thread pipeline_thread;
    std::thread event_thread;
    uint64_t dropped_pool_exhausted = 0U;
    std::unique_ptr<nvidia::isaac_ros::nitros::CUDAMemoryPool> compact_pool;
    rclcpp::Publisher<nvidia::isaac_ros::nitros::NitrosImage>::SharedPtr image_pub;
    rclcpp::Publisher<sensor_msgs::msg::CameraInfo>::SharedPtr camera_info_pub;
    // Determined at init from ISP0 buffer layout. When false the Y and UV
    // planes are already contiguous and a single memcpy replaces the
    // two-copy compaction path.
    bool needs_compaction = true;
    size_t compact_frame_bytes = 0;
  };

  // Shared constants
  static constexpr uint64_t kSiplQueueGetTimeoutUs = 100'000U;

  // REP-103 optical frame convention rotation (parent <- child).
  // Transform that converts from camera body frame to camera optical frame:
  // Camera Link   ->  Optical
  // x             ->  z
  // y             ->  -x
  // z             ->  -y
  // +x should point to the right in the image
  // +y should point down in the image
  // +z should point into the plane of the image
  static inline const tf2::Matrix3x3 kCamLinkROptical {
    0.0, 0.0, 1.0,
    -1.0, 0.0, 0.0,
    0.0, -1.0, 0.0
  };

  // --- Pipeline lifecycle (called by initialize()) ---

  /// Subclass override: declare which sensors to configure.
  /// Mono returns 1 descriptor; stereo returns 2.
  virtual std::vector<PipelineDescriptor> describePipelines();

  /// Shared init sequence called by leaf constructors after all params are declared.
  void initialize(std::vector<PipelineDescriptor> descriptors);

  /// Publish static TF transforms. Override in stereo for the 4-transform tree.
  virtual void publishStaticTransforms();

  /// Perform transport-specific preparation immediately before SIPL Start().
  virtual void prepareCaptureStart();

  // --- Shared camera info / utilities for derived nodes ---
  sensor_msgs::msg::CameraInfo loadCameraInfoFromFile(
    const std::string & camera_info_url);

  // Shared utilities
  std::vector<uint8_t> loadNitoFile(const std::string & path);
  uint64_t extractTscTimestamp(nvsipl::INvSIPLClient::INvSIPLNvMBuffer * buffer);
  uint64_t nanosecondsToTscTicks(uint64_t nanoseconds) const;
  NvSciBufSurfSampleType getSurfSampleTypeFromEncoding() const;

  void handlePipelineNotification(
    const nvsipl::NvSIPLPipelineNotifier::NotificationData & event);

  void processFrame(
    nvsipl::INvSIPLClient::INvSIPLNvMBuffer * buffer,
    CameraPipeline & pipeline,
    std::chrono::steady_clock::time_point isp0_start_time);

  // Acquires a compact pool buffer, populates it with NV12/NV24 bytes, and publishes
  // NitrosImage + matching CameraInfo.
  //
  // `step_bytes` is the row stride from `attrs.plane_pitches[0]`. This may not equal
  // image_width_ when the ISP produces per-row padding that the compactor preserves.
  bool publishFrame(
    CameraPipeline & pipeline,
    const std_msgs::msg::Header & header,
    uint32_t step_bytes,
    const std::function<bool(uint8_t * compact_ptr, size_t block_size)> & fill_fn);

  // --- Per-pipeline runtime state ---
  std::vector<CameraPipeline> pipelines_;

  // SIPL API instances
  std::unique_ptr<nvsipl::INvSIPLCamera> sipl_camera_;
  std::unique_ptr<nvsipl::INvSIPLCameraQuery> sipl_query_;
  nvsipl::sensorconfig::SensorSystemConfig sensor_system_config_;
  std::unique_ptr<TransportAdapter> transport_adapter_;
  // Return the CommonSensorConfig matching the given SIPL global sensor id.
  // Throws if not found.
  const nvsipl::sensorconfig::CommonSensorConfig &
  getSensorConfig(uint32_t sensor_id) const;

  // NvSci modules
  std::shared_ptr<NvSciBufModuleRec> sci_buf_module_;
  NvSciSyncCpuWaitContext cpu_wait_context_;

  // TF broadcasting
  std::unique_ptr<tf2_ros::StaticTransformBroadcaster> tf_static_broadcaster_;

  ::nvidia::isaac_ros::common::CudaStreamPtr cuda_stream_;

  // Threading
  std::atomic<bool> stop_capture_;

  // Camera parameters
  uint32_t image_width_;
  uint32_t image_height_;

  // ROS Parameters
  std::string platform_config_;
  std::string encoding_desired_;
  std::string camera_link_frame_name_;
  std::string nito_file_;
  bool enable_debug_logs_{false};
  rclcpp::QoS output_qos_;
  int output_buffer_pool_size_{0};
  uint16_t link_mask_{0x0001};
  double first_frame_timeout_s_{5.0};
  // Flag to mark when the first frame timeout has already been reported.
  std::atomic<bool> first_frame_timeout_reported_{false};

  // Stereo timestamp alignment (active only for 2-pipeline stereo with HW timestamps).
  bool align_stereo_timestamps_{false};
  double max_timestamp_diff_us_{500.0};
  double expected_fps_{30.0};
  // Count of stereo pairs that arrived skewed beyond max_timestamp_diff_us (kept
  // unaligned, so the downstream ExactTime gate drops them). Incremented from both
  // pipeline threads. Note: this counts skew-induced drops only, not pairs lost
  // because one frame of the pair never arrived.
  std::atomic<uint64_t> stereo_desync_count_{0};

  // Camera info caching (loaded in constructor, moved into pipeline by initialize())
  sensor_msgs::msg::CameraInfo camera_info_;
  bool camera_info_loaded_{false};

private:
  // --- Internal init helpers (called by initialize()) ---
  void setupSiplCamera(size_t num_sensors);
  std::vector<PipelineDescriptor> resolvePipelineDescriptors(
    std::vector<PipelineDescriptor> requested_descriptors);
  void allocateBuffersForPipeline(CameraPipeline & pipeline);
  void registerBuffersForPipeline(CameraPipeline & pipeline);
  void createPublisherForPipeline(CameraPipeline & pipeline);
  /// Allocate the per-pipeline compact GPU buffer pools used by publishFrame.
  void allocateCompactPools();
  void startAllPipelines();
  void stopAllPipelines();

  // --- Thread entry points ---
  void pipelineThread(CameraPipeline & pipeline);
  void handleNotificationQueue(CameraPipeline & pipeline);

  void publishCameraInfo(const CameraPipeline & pipeline, const std_msgs::msg::Header & header);

  void allocateSync(
    uint32_t sensor_id,
    nvsipl::INvSIPLClient::ConsumerDesc::OutputType output_type,
    NvSciSyncObj & sync);
  void registerAutoControlPlugin(uint32_t sensor_id);

  bool validateBufferFormat(
    const BufferAttributes & attrs,
    const CameraPipeline & pipeline) const;

  NvSciSyncModule sci_sync_module_;

  std::string hsb_name_;
  int hsb_id_;
  std::string interface_name_;
  bool use_hw_timestamp_;
  std::string optical_frame_name_;

  TscCorrelator tsc_correlator_;
  std::unique_ptr<StereoTimestampAligner> stereo_timestamp_aligner_;
};

}  // namespace sipl
}  // namespace isaac_ros

#endif  // ISAAC_ROS_SIPL_CAMERA__SIPL_CAMERA_NODE_HPP_
