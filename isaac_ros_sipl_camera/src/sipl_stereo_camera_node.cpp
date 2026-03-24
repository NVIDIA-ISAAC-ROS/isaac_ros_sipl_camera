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

#include <Eigen/Dense>
#include "tf2/LinearMath/Quaternion.h"
#include "tf2/LinearMath/Transform.h"

namespace isaac_ros
{
namespace sipl
{

namespace
{

// Compute the right-optical pose relative to the left-optical frame using stereo calibration.
// This mirrors the standard stereo derivation: use left/right R and right P to recover
// rectified translation (t = P.col(3) / fx), then compose right_T_left and invert to match ROS.
tf2::Transform computeLeftOpticalPoseRightOptical(
  const sensor_msgs::msg::CameraInfo & left_camera_info,
  const sensor_msgs::msg::CameraInfo & right_camera_info)
{
  Eigen::Map<const Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> right_r_map(
    right_camera_info.r.data());
  Eigen::Matrix3d right_r = right_r_map;

  Eigen::Map<const Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> left_r_map(
    left_camera_info.r.data());
  Eigen::Matrix3d left_r = left_r_map;

  Eigen::Map<const Eigen::Matrix<double, 3, 4, Eigen::RowMajor>> right_p_map(
    right_camera_info.p.data());
  Eigen::Matrix<double, 3, 4> right_p = right_p_map;

  const double f_x = right_camera_info.p[0];
  if (f_x == 0.0) {
    throw std::runtime_error("Right camera P[0] must be non-zero");
  }

  Eigen::Vector3d t = (1.0 / f_x) * right_p.col(3).head<3>();

  auto matrix4x4 = [](const Eigen::Matrix3d * rotation,
    const Eigen::Vector3d * translation) -> Eigen::Matrix4d {
      Eigen::Matrix4d pose_3d = Eigen::Matrix4d::Identity();
      if (rotation != nullptr) {
        pose_3d.block<3, 3>(0, 0) = *rotation;
      }
      if (translation != nullptr) {
        pose_3d.block<3, 1>(0, 3) = *translation;
      }
      return pose_3d;
    };

  Eigen::Matrix4d right_T_rectified_right =
    matrix4x4(&right_r, nullptr).transpose();
  Eigen::Matrix4d rectified_right_T_rectified_left =
    matrix4x4(nullptr, &t);
  Eigen::Matrix4d rectified_left_T_left = matrix4x4(&left_r, nullptr);

  Eigen::Matrix4d right_T_left = right_T_rectified_right *
    rectified_right_T_rectified_left *
    rectified_left_T_left;

  tf2::Matrix3x3 rotation(
    right_T_left(0, 0), right_T_left(0, 1), right_T_left(0, 2),
    right_T_left(1, 0), right_T_left(1, 1), right_T_left(1, 2),
    right_T_left(2, 0), right_T_left(2, 1), right_T_left(2, 2));
  tf2::Vector3 translation(
    right_T_left(0, 3), right_T_left(1, 3), right_T_left(2, 3));

  tf2::Transform transform(rotation, translation);
  return transform.inverse();
}

}  // namespace

SiplStereoCameraNode::SiplStereoCameraNode(const rclcpp::NodeOptions & options)
: SiplCameraNode(options, true)
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
