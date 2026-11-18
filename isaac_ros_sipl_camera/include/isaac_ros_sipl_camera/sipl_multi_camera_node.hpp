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

#ifndef ISAAC_ROS_SIPL_CAMERA__SIPL_MULTI_CAMERA_NODE_HPP_
#define ISAAC_ROS_SIPL_CAMERA__SIPL_MULTI_CAMERA_NODE_HPP_

#include <string>
#include <vector>

#include "isaac_ros_sipl_camera/sipl_camera_node.hpp"

namespace isaac_ros
{
namespace sipl
{

/// Drive named RAW and YUV422 camera pipelines in one SIPL instance.
class SiplMultiCameraNode : public SiplCameraNode
{
public:
  explicit SiplMultiCameraNode(const rclcpp::NodeOptions & options = rclcpp::NodeOptions());

  ~SiplMultiCameraNode() override = default;

protected:
  std::vector<PipelineDescriptor> describePipelines() override;
  void publishStaticTransforms() override;

private:
  static PipelineDescriptor::Output parseOutput(
    const std::string & value, const std::string & camera_name);

  /// Per-camera configuration read from the `camera.<name>.*` parameters.
  struct CameraSpec
  {
    std::string name;
    std::string topic_namespace;
    std::string frame_name;
    uint32_t sensor_id = 0;
    PipelineDescriptor::Output output = PipelineDescriptor::Output::kAuto;
    bool raw_output = false;
    std::string encoding;
    std::string nito_file;
    sensor_msgs::msg::CameraInfo camera_info;
    bool camera_info_loaded = false;
  };

  std::vector<CameraSpec> cameras_;
};

}  // namespace sipl
}  // namespace isaac_ros

#endif  // ISAAC_ROS_SIPL_CAMERA__SIPL_MULTI_CAMERA_NODE_HPP_
