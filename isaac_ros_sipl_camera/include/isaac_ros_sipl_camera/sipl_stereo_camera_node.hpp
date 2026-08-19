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

#ifndef ISAAC_ROS_SIPL_CAMERA__SIPL_STEREO_CAMERA_NODE_HPP_
#define ISAAC_ROS_SIPL_CAMERA__SIPL_STEREO_CAMERA_NODE_HPP_

#include <string>
#include <vector>

#include "isaac_ros_sipl_camera/sipl_camera_node.hpp"

namespace isaac_ros
{
namespace sipl
{

/**
 * @brief ROS2 node for SIPL stereo camera integration
 *
 * Extends SiplCameraNode to support stereo camera pairs (like VB1940),
 * managing two sensors with synchronized capture and publishing.
 * Only overrides describePipelines() and publishStaticTransforms();
 * all per-sensor runtime state is managed by the base class via pipelines_.
 */
class SiplStereoCameraNode : public SiplCameraNode
{
public:
  explicit SiplStereoCameraNode(const rclcpp::NodeOptions & options = rclcpp::NodeOptions());

  // Base class destructor handles stopAllPipelines() and NvSci cleanup.
  ~SiplStereoCameraNode() override = default;

protected:
  std::vector<PipelineDescriptor> describePipelines() override;
  void publishStaticTransforms() override;
  void prepareCaptureStart() override;

private:
  std::string left_camera_frame_name_;
  std::string right_camera_frame_name_;
  uint32_t fsync_group_id_{0U};

  // Left camera info is inherited from the base class (camera_info_ / camera_info_loaded_).
  // Right camera info (loaded during construction, moved into pipeline by initialize())
  sensor_msgs::msg::CameraInfo right_camera_info_;
  bool right_camera_info_loaded_{false};
};

}  // namespace sipl
}  // namespace isaac_ros

#endif  // ISAAC_ROS_SIPL_CAMERA__SIPL_STEREO_CAMERA_NODE_HPP_
