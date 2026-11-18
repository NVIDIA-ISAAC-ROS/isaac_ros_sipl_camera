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
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <cinttypes>

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/camera_info.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "camera_info_manager/camera_info_manager.hpp"
#include "tf2/LinearMath/Matrix3x3.hpp"
#include "tf2_ros/static_transform_broadcaster.hpp"

#include "isaac_ros_sipl_camera/sipl_buffer_manager.hpp"
#include "isaac_ros_sipl_camera/transport_adapter.hpp"
#include "isaac_ros_sipl_camera/tsc_correlator.hpp"
#include "isaac_ros_sipl_camera/stereo_timestamp_aligner.hpp"
#include "isaac_ros_sipl_camera/yuv_convert.hpp"

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
 * publishing CUDA-backed ROS images with GPU memory management.
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
  /// Defer initialization until the derived node has declared its parameters.
  SiplCameraNode(const rclcpp::NodeOptions & options, const std::string & node_name);

  /// Struct to describe the pipeline.
  /// The name field doubles as the ROS topic prefix:
  /// "" for mono (topics are "image_raw", "camera_info"),
  /// "left"/"right" for stereo (topics are "left/image_raw", "left/camera_info").
  struct PipelineDescriptor
  {
    /// kAuto selects ICP for YUV422 sensors and ISP0 for RAW sensors.
    enum class Output { kAuto, kIsp0, kIcp };

    std::string name;
    uint32_t sensor_id;                   // SIPL global sensor id (CommonSensorConfig::id)
    std::string frame_id;                 // optical frame name for headers
    sensor_msgs::msg::CameraInfo camera_info;
    bool camera_info_loaded = false;

    /// Topic namespace relative to the node namespace; empty uses `name`.
    std::string topic_prefix;

    bool sensor_id_explicit = false;
    Output requested_output = Output::kAuto;
    /// Request ICP alongside ISP0 for the mono/stereo drain loop.
    bool raw_output = true;
    /// Empty values inherit the node-level parameters.
    std::string encoding;
    std::string nito_file;

    /// Returns a human-readable name for log messages.
    const char * display_name() const {return name.empty() ? "camera" : name.c_str();}
  };

  struct PipelineStatistics
  {
    std::atomic<uint64_t> icp_frames_completed{0};
    std::atomic<uint64_t> frames_acquired{0};
    std::atomic<uint64_t> frames_published{0};
    std::atomic<uint64_t> app_frame_drops{0};
    // CUDA errors detected after ROS publication.
    std::atomic<uint64_t> published_frame_errors{0};
    std::atomic<uint64_t> stereo_desyncs{0};
    std::atomic<int64_t> first_publish_ns{0};
    std::atomic<int64_t> last_publish_ns{0};
    std::atomic<uint64_t> previous_report_published{0};
    std::atomic<int64_t> previous_report_ns{0};
    std::atomic<uint64_t> latency_samples{0};
    std::atomic<uint64_t> latency_violations{0};
    std::atomic<int64_t> last_latency_error_ns{0};

    // Protect the stall state and notification totals across pipeline threads.
    std::mutex details_mutex;
    uint64_t output_stall_episodes{0};
    uint64_t output_stall_total_ns{0};
    int64_t output_stall_start_ns{0};
    std::map<uint16_t, uint64_t> sipl_event_totals;
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
    nvsipl::NvSIPLPipelineConfiguration cfg{};
    uint32_t image_width = 0;
    uint32_t image_height = 0;
    std::string encoding;
    sensor_msgs::msg::CameraInfo camera_info_msg;
    bool publish_from_capture = false;
    bool capture_enabled = true;
    PackedYuvOrder packed_yuv_order = PackedYuvOrder::kUyvy;
    // Owned by timestamp_aligners_.
    StereoTimestampAligner * aligner = nullptr;
    nvsipl::NvSIPLPipelineQueues queues;
    NvSciSyncObj sci_sync_isp0 = nullptr;
    // NvSciSync permits only one outstanding fence wait per CPU wait context.
    // Each pipeline owns its context, including during initialization failure.
    std::unique_ptr<NvSciSyncCpuWaitContextRec, decltype(&NvSciSyncCpuWaitContextFree)>
    cpu_wait_context{nullptr, &NvSciSyncCpuWaitContextFree};
    std::shared_ptr<SiplBufferManager> buffer_manager_icp;
    std::shared_ptr<SiplBufferManager> buffer_manager_isp0;
    std::thread pipeline_thread;
    std::thread event_thread;
    // Each pipeline owns its CUDA stream so they are decoupled.
    ::nvidia::isaac_ros::common::CudaStreamPtr cuda_stream;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr image_pub;
    rclcpp::Publisher<sensor_msgs::msg::CameraInfo>::SharedPtr camera_info_pub;
    // Determined at init from ISP0 buffer layout. When false the Y and UV
    // planes are already contiguous and a single memcpy replaces the
    // two-copy compaction path.
    bool needs_compaction = true;
    size_t compact_frame_bytes = 0;
    std::shared_ptr<PipelineStatistics> statistics =
      std::make_shared<PipelineStatistics>();
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

  void restartFsyncGroup(uint32_t fsync_group_id);

  // --- Shared camera info / utilities for derived nodes ---
  sensor_msgs::msg::CameraInfo loadCameraInfoFromFile(
    const std::string & camera_info_url);

  void declareTimestampAlignmentParams();

  // Shared utilities
  std::vector<uint8_t> loadNitoFile(const std::string & path);
  uint64_t extractTscTimestamp(nvsipl::INvSIPLClient::INvSIPLNvMBuffer * buffer);
  uint64_t nanosecondsToTscTicks(uint64_t nanoseconds) const;
  NvSciBufSurfSampleType getSurfSampleTypeFromEncoding(const std::string & encoding) const;

  void handlePipelineNotification(
    CameraPipeline & pipeline,
    const nvsipl::NvSIPLPipelineNotifier::NotificationData & event);

  /// Returns true when the frame was published.
  bool processIspFrame(
    nvsipl::INvSIPLClient::INvSIPLNvMBuffer * buffer,
    CameraPipeline & pipeline,
    std::chrono::steady_clock::time_point isp0_start_time);

  /// Convert a packed capture buffer to NV12; return whether it was published.
  bool processCaptureFrame(
    nvsipl::INvSIPLClient::INvSIPLNvMBuffer * buffer,
    CameraPipeline & pipeline,
    std::chrono::steady_clock::time_point capture_start_time);

  /// Return the SOF-derived, optionally aligned header, or nullopt for a zero TSC.
  std::optional<std_msgs::msg::Header> buildFrameHeader(
    CameraPipeline & pipeline,
    nvsipl::INvSIPLClient::INvSIPLNvMBuffer * buffer,
    uint64_t * out_raw_tsc = nullptr);

  // Fill a CUDA image and publish it with matching CameraInfo.
  // after_fill runs before publication; either callback can abort by returning false.
  // step_bytes is the luma row stride, including padding.
  bool publishFrame(
    CameraPipeline & pipeline,
    const std_msgs::msg::Header & header,
    uint32_t step_bytes,
    const std::function<bool(uint8_t * compact_ptr, size_t block_size)> & fill_fn,
    const std::function<bool()> & after_fill = {});

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

  // TF broadcasting
  std::unique_ptr<tf2_ros::StaticTransformBroadcaster> tf_static_broadcaster_;

  // Threading
  std::atomic<bool> stop_capture_;

  // ROS Parameters
  std::string platform_config_;
  std::string platform_config_file_;
  std::string encoding_desired_;
  std::string camera_link_frame_name_;
  std::string nito_file_;
  bool owns_sipl_env_{false};
  std::vector<std::pair<std::string, std::optional<std::string>>> previous_sipl_env_;
  bool publish_tf_{true};
  bool enable_debug_logs_{false};
  bool enable_statistics_{false};
  double statistics_interval_s_{60.0};
  std::atomic<bool> statistics_stopping_{false};
  std::atomic<bool> stop_notifications_{false};
  std::mutex statistics_report_mutex_;
  rclcpp::TimerBase::SharedPtr statistics_timer_;
  rclcpp::QoS output_qos_;
  uint16_t capture_buffer_count_{4U};
  // GMSL link selection in plain bit positions (bit i = link i), one entry per
  // transport in the platform configuration.
  std::vector<uint16_t> link_masks_{0x0001};
  uint32_t fsync_group_id_{0U};
  bool enable_fsync_restart_{true};
  double first_frame_timeout_s_{5.0};
  // Flag to mark when the first frame timeout has already been reported.
  std::atomic<bool> first_frame_timeout_reported_{false};

  // Stereo timestamp alignment (active only for 2-pipeline stereo with HW timestamps).
  bool align_stereo_timestamps_{false};
  double max_timestamp_diff_us_{500.0};
  double expected_fps_{30.0};
  // Pairs used for TF and timestamp alignment; set before initialize().
  std::vector<std::pair<std::string, std::string>> stereo_pair_names_;
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
  void setupSiplCamera(const std::vector<PipelineDescriptor> & descriptors);
  std::vector<PipelineDescriptor> resolvePipelineDescriptors(
    std::vector<PipelineDescriptor> requested_descriptors);
  void resolvePipelineOutputs(CameraPipeline & pipeline);
  void buildPipelineConfig(CameraPipeline & pipeline);
  void buildTimestampAligners();
  void allocateBuffersForPipeline(CameraPipeline & pipeline);
  void registerBuffersForPipeline(CameraPipeline & pipeline);
  void createPublisherForPipeline(CameraPipeline & pipeline);
  void startAllPipelines();
  void stopAllPipelines();

  // --- Thread entry points ---
  void pipelineThread(CameraPipeline & pipeline);
  void handleNotificationQueue(CameraPipeline & pipeline);
  void logStatistics(const char * report_type, bool final_report = false);

  void publishCameraInfo(CameraPipeline & pipeline, const std_msgs::msg::Header & header);

  void allocateSync(
    uint32_t sensor_id,
    nvsipl::INvSIPLClient::ConsumerDesc::OutputType output_type,
    NvSciSyncObj & sync);
  void registerAutoControlPlugin(const CameraPipeline & pipeline);

  bool validateIsp0BufferFormat(
    const BufferAttributes & attrs,
    const CameraPipeline & pipeline) const;

  bool validateCaptureBufferFormat(
    const BufferAttributes & attrs,
    CameraPipeline & pipeline) const;

  NvSciSyncModule sci_sync_module_;

  std::string hsb_name_;
  int hsb_id_;
  std::string interface_name_;
  bool use_hw_timestamp_;
  double max_capture_to_publish_latency_frame_periods_{2.0};
  std::string optical_frame_name_;

  TscCorrelator tsc_correlator_;
  std::vector<std::unique_ptr<StereoTimestampAligner>> timestamp_aligners_;
};

}  // namespace sipl
}  // namespace isaac_ros

#endif  // ISAAC_ROS_SIPL_CAMERA__SIPL_CAMERA_NODE_HPP_
