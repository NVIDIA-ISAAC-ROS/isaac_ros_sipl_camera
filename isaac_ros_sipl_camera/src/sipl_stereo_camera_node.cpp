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

#include "isaac_ros_sipl_camera/sipl_stereo_camera_node.hpp"

#include <cinttypes>
#include <cstdint>
#include <stdexcept>

#include "tf2/LinearMath/Quaternion.hpp"
#include "tf2/LinearMath/Transform.hpp"

#include "isaac_ros_sipl_camera/stereo_extrinsics.hpp"

namespace isaac_ros
{
namespace sipl
{

SiplStereoCameraNode::SiplStereoCameraNode(const rclcpp::NodeOptions & options)
: SiplCameraNode(options, "sipl_stereo_camera")
{
  // Declare stereo-specific parameters
  left_camera_frame_name_ = declare_parameter<std::string>(
    "left_camera_frame_name", "stereo_left");
  right_camera_frame_name_ = declare_parameter<std::string>(
    "right_camera_frame_name", "stereo_right");

  const std::string right_camera_info_url =
    declare_parameter<std::string>("right_camera_info_url", "");
  if (!right_camera_info_url.empty()) {
    right_camera_info_ = loadCameraInfoFromFile(right_camera_info_url);
    right_camera_info_loaded_ = true;
    RCLCPP_INFO(
      get_logger(), "[SiplStereoCameraNode] Loaded right camera info from \"%s\"",
      right_camera_info_url.c_str());
  }

  declareTimestampAlignmentParams();

  initialize(describePipelines());
}

std::vector<SiplCameraNode::PipelineDescriptor>
SiplStereoCameraNode::describePipelines()
{
  return {
    {"left", 0, left_camera_frame_name_ + "_optical",
      camera_info_, camera_info_loaded_},
    {"right", 1, right_camera_frame_name_ + "_optical",
      right_camera_info_, right_camera_info_loaded_}
  };
}


void SiplStereoCameraNode::publishStaticTransforms()
{
  geometry_msgs::msg::TransformStamped left_transform, right_transform;
  geometry_msgs::msg::TransformStamped left_optical, right_optical;

  auto stamp = now();

  tf2::Quaternion optical_q;
  kCamLinkROptical.getRotation(optical_q);

  // Left camera: camera_link → left_camera
  left_transform.header.stamp = stamp;
  left_transform.header.frame_id = camera_link_frame_name_;
  left_transform.child_frame_id = left_camera_frame_name_;
  left_transform.transform.translation.x = 0.0;
  left_transform.transform.translation.y = 0.0;
  left_transform.transform.translation.z = 0.0;
  left_transform.transform.rotation.x = 0.0;
  left_transform.transform.rotation.y = 0.0;
  left_transform.transform.rotation.z = 0.0;
  left_transform.transform.rotation.w = 1.0;

  // Left optical: left_camera → left_camera_optical
  left_optical.header.stamp = stamp;
  left_optical.header.frame_id = left_camera_frame_name_;
  left_optical.child_frame_id = left_camera_frame_name_ + "_optical";
  left_optical.transform.translation.x = 0.0;
  left_optical.transform.translation.y = 0.0;
  left_optical.transform.translation.z = 0.0;
  left_optical.transform.rotation.x = optical_q.x();
  left_optical.transform.rotation.y = optical_q.y();
  left_optical.transform.rotation.z = optical_q.z();
  left_optical.transform.rotation.w = optical_q.w();

  // Right camera: camera_link → right_camera
  right_transform.header.stamp = stamp;
  right_transform.header.frame_id = camera_link_frame_name_;
  right_transform.child_frame_id = right_camera_frame_name_;
  bool computed_right_transform = false;
  if (pipelines_[0].desc.camera_info_loaded && pipelines_[1].desc.camera_info_loaded) {
    try {
      const tf2::Transform left_optical_pose_right_optical =
        computeLeftOpticalPoseRightOptical(
          pipelines_[0].desc.camera_info, pipelines_[1].desc.camera_info);
      const tf2::Transform cam_link_pose_optical(
        kCamLinkROptical, tf2::Vector3(0.0, 0.0, 0.0));
      const tf2::Transform cam_link_pose_right_optical =
        cam_link_pose_optical * left_optical_pose_right_optical;
      const tf2::Transform cam_link_pose_right =
        cam_link_pose_right_optical * cam_link_pose_optical.inverse();

      right_transform.transform.translation.x = cam_link_pose_right.getOrigin().getX();
      right_transform.transform.translation.y = cam_link_pose_right.getOrigin().getY();
      right_transform.transform.translation.z = cam_link_pose_right.getOrigin().getZ();

      const tf2::Quaternion right_q = cam_link_pose_right.getRotation();
      right_transform.transform.rotation.x = right_q.x();
      right_transform.transform.rotation.y = right_q.y();
      right_transform.transform.rotation.z = right_q.z();
      right_transform.transform.rotation.w = right_q.w();
      computed_right_transform = true;
    } catch (const std::exception & e) {
      RCLCPP_WARN(
        get_logger(),
        "Failed to compute right camera transform from calibration: %s", e.what());
    }
  }

  if (!computed_right_transform) {
    // Note: SIPL does not currently provide a calibration data retrieval API.
    // If camera info is not loaded, we're unable to determine the stereo baseline,
    // and will default to 0.
    RCLCPP_WARN(
      get_logger(),
      "Using default transform for right camera assuming 0 stereo baseline");
    right_transform.transform.translation.x = 0.0;
    right_transform.transform.translation.y = 0.0;
    right_transform.transform.translation.z = 0.0;
    right_transform.transform.rotation.x = 0.0;
    right_transform.transform.rotation.y = 0.0;
    right_transform.transform.rotation.z = 0.0;
    right_transform.transform.rotation.w = 1.0;
  }

  // Right optical: right_camera → right_camera_optical
  right_optical.header.stamp = stamp;
  right_optical.header.frame_id = right_camera_frame_name_;
  right_optical.child_frame_id = right_camera_frame_name_ + "_optical";
  right_optical.transform.translation.x = 0.0;
  right_optical.transform.translation.y = 0.0;
  right_optical.transform.translation.z = 0.0;
  right_optical.transform.rotation.x = optical_q.x();
  right_optical.transform.rotation.y = optical_q.y();
  right_optical.transform.rotation.z = optical_q.z();
  right_optical.transform.rotation.w = optical_q.w();

  std::vector<geometry_msgs::msg::TransformStamped> transforms = {
    left_transform, left_optical, right_transform, right_optical
  };

  tf_static_broadcaster_->sendTransform(transforms);

  RCLCPP_INFO(
    get_logger(), "Published stereo static transforms: %s → {%s, %s}",
    camera_link_frame_name_.c_str(),
    left_camera_frame_name_.c_str(),
    right_camera_frame_name_.c_str());
}

}  // namespace sipl
}  // namespace isaac_ros

// Register as component
#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(isaac_ros::sipl::SiplStereoCameraNode)
