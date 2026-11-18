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

#ifndef ISAAC_ROS_SIPL_CAMERA__STEREO_EXTRINSICS_HPP_
#define ISAAC_ROS_SIPL_CAMERA__STEREO_EXTRINSICS_HPP_

#include <Eigen/Dense>

#include <stdexcept>

#include "sensor_msgs/msg/camera_info.hpp"
#include "tf2/LinearMath/Matrix3x3.hpp"
#include "tf2/LinearMath/Transform.hpp"
#include "tf2/LinearMath/Vector3.hpp"

namespace isaac_ros
{
namespace sipl
{

/// Compute the right-optical pose relative to the left-optical frame using stereo
/// calibration. This mirrors the standard stereo derivation: use left/right R and
/// right P to recover rectified translation (t = P.col(3) / fx), then compose
/// right_T_left and invert to match ROS.
///
/// Throws if the right camera's P[0] (fx) is zero.
inline tf2::Transform computeLeftOpticalPoseRightOptical(
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

  tf2::Transform right_pose_left(rotation, translation);
  return right_pose_left.inverse();
}

}  // namespace sipl
}  // namespace isaac_ros

#endif  // ISAAC_ROS_SIPL_CAMERA__STEREO_EXTRINSICS_HPP_
