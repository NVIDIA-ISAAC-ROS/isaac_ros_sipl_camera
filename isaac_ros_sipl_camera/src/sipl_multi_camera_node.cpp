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

#include "isaac_ros_sipl_camera/sipl_multi_camera_node.hpp"

#include <algorithm>
#include <cctype>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>
#include <utility>

#include "tf2/LinearMath/Quaternion.hpp"
#include "tf2/LinearMath/Transform.hpp"

#include "isaac_ros_sipl_camera/stereo_extrinsics.hpp"

namespace isaac_ros
{
namespace sipl
{

namespace
{

std::pair<std::string, std::string> parsePair(const std::string & entry)
{
  const size_t comma = entry.find(',');
  if (comma == std::string::npos || entry.find(',', comma + 1U) != std::string::npos) {
    throw std::runtime_error(
      "stereo_pairs entries must be \"<left>,<right>\"; got '" + entry + "'");
  }
  auto trim = [](std::string s) {
      const auto begin = s.find_first_not_of(" \t");
      if (begin == std::string::npos) {return std::string{};}
      const auto end = s.find_last_not_of(" \t");
      return s.substr(begin, end - begin + 1);
    };
  const std::string left = trim(entry.substr(0, comma));
  const std::string right = trim(entry.substr(comma + 1));
  if (left.empty() || right.empty()) {
    throw std::runtime_error(
      "stereo_pairs entries must name two cameras; got '" + entry + "'");
  }
  return {left, right};
}

}  // namespace

SiplCameraNode::PipelineDescriptor::Output SiplMultiCameraNode::parseOutput(
  const std::string & value, const std::string & camera_name)
{
  if (value == "auto") {return PipelineDescriptor::Output::kAuto;}
  if (value == "isp0") {return PipelineDescriptor::Output::kIsp0;}
  if (value == "icp") {return PipelineDescriptor::Output::kIcp;}
  throw std::runtime_error(
    "camera." + camera_name + ".output must be 'auto', 'isp0' or 'icp'; got '" + value + "'");
}

SiplMultiCameraNode::SiplMultiCameraNode(const rclcpp::NodeOptions & options)
: SiplCameraNode(options, "sipl_multi_camera")
{
  std::vector<std::string> camera_names;
  {
    rcl_interfaces::msg::ParameterDescriptor d;
    d.description =
      "Ordered list of camera names. Each name is the published topic prefix "
      "(<name>/image_raw, <name>/camera_info) and selects the "
      "camera.<name>.* parameter block.";
    camera_names = declare_parameter<std::vector<std::string>>(
      "cameras", std::vector<std::string>{}, d);
  }
  if (camera_names.empty()) {
    throw std::runtime_error("SiplMultiCameraNode requires a non-empty 'cameras' list");
  }

  std::set<std::string> seen_names;
  std::set<uint32_t> seen_sensor_ids;
  std::set<std::string> seen_image_topics;
  std::set<std::string> seen_frame_names;
  cameras_.reserve(camera_names.size());

  for (const auto & name : camera_names) {
    if (name.empty()) {
      throw std::runtime_error("'cameras' entries must not be empty");
    }
    const bool valid_name = std::all_of(
      name.begin(), name.end(), [](unsigned char c) {return std::isalnum(c) || c == '_';});
    if (!valid_name) {
      throw std::runtime_error(
        "Camera name '" + name + "' may contain only letters, digits, and underscores");
    }
    if (!seen_names.insert(name).second) {
      throw std::runtime_error("Duplicate camera name in 'cameras': " + name);
    }

    const std::string prefix = "camera." + name + ".";
    CameraSpec spec;
    spec.name = name;

    const int64_t sensor_id = declare_parameter<int64_t>(prefix + "sensor_id", -1);
    if (sensor_id < 0 ||
      sensor_id > static_cast<int64_t>(std::numeric_limits<uint32_t>::max()))
    {
      throw std::runtime_error(
        prefix + "sensor_id is required and must be a valid SIPL sensor id; got " +
        std::to_string(sensor_id));
    }
    spec.sensor_id = static_cast<uint32_t>(sensor_id);
    if (!seen_sensor_ids.insert(spec.sensor_id).second) {
      throw std::runtime_error(
        "Two cameras claim sensor_id " + std::to_string(spec.sensor_id));
    }

    spec.output = parseOutput(
      declare_parameter<std::string>(prefix + "output", "auto"), name);
    spec.raw_output = declare_parameter<bool>(prefix + "raw_output", false);
    spec.topic_namespace = declare_parameter<std::string>(prefix + "topic_namespace", "");
    // Keep topics under the node namespace even when written with a leading '/'.
    spec.topic_namespace.erase(0, spec.topic_namespace.find_first_not_of('/'));
    spec.frame_name = declare_parameter<std::string>(prefix + "frame_name", name);
    if (spec.frame_name.empty()) {
      throw std::runtime_error(prefix + "frame_name must not be empty");
    }
    if (!seen_frame_names.insert(spec.frame_name).second) {
      throw std::runtime_error(
        "Two cameras claim frame_name '" + spec.frame_name + "'");
    }

    const std::string effective_topic_namespace =
      spec.topic_namespace.empty() ? name : spec.topic_namespace;
    const std::string resolved_image_topic =
      get_node_topics_interface()->resolve_topic_name(
      effective_topic_namespace + "/image_raw");
    if (!seen_image_topics.insert(resolved_image_topic).second) {
      throw std::runtime_error(
        "Two cameras resolve to image topic '" + resolved_image_topic +
        "'; use distinct camera names or topic_namespace values");
    }

    spec.encoding = declare_parameter<std::string>(prefix + "encoding", "");
    spec.nito_file = declare_parameter<std::string>(prefix + "nito_file", "");

    const std::string camera_info_url =
      declare_parameter<std::string>(prefix + "camera_info_url", "");
    if (!camera_info_url.empty()) {
      spec.camera_info = loadCameraInfoFromFile(camera_info_url);
      spec.camera_info_loaded = true;
      RCLCPP_INFO(
        get_logger(), "[%s] Loaded camera info from \"%s\"",
        name.c_str(), camera_info_url.c_str());
    }

    cameras_.push_back(std::move(spec));
  }

  {
    rcl_interfaces::msg::ParameterDescriptor d;
    d.description =
      "Camera pairs that form a stereo rig, each written \"<left>,<right>\". "
      "The right camera's pose relative to the rig is derived from the pair's "
      "calibration; without a pair a camera sits at the rig origin. Also selects "
      "which pairs are timestamp-aligned when align_stereo_timestamps is true.";
    const auto pairs = declare_parameter<std::vector<std::string>>(
      "stereo_pairs", std::vector<std::string>{}, d);
    for (const auto & entry : pairs) {
      stereo_pair_names_.push_back(parsePair(entry));
    }
  }

  std::set<std::string> paired_camera_names;
  for (const auto & [left_name, right_name] : stereo_pair_names_) {
    if (left_name == right_name) {
      throw std::runtime_error(
        "stereo_pairs entry pairs camera '" + left_name + "' with itself");
    }
    if (seen_names.count(left_name) == 0U || seen_names.count(right_name) == 0U) {
      throw std::runtime_error(
        "stereo_pairs names unknown camera(s): '" + left_name + "', '" + right_name + "'");
    }
    if (!paired_camera_names.insert(left_name).second ||
      !paired_camera_names.insert(right_name).second)
    {
      throw std::runtime_error(
        "stereo_pairs assigns camera '" + left_name + "' or '" + right_name +
        "' to more than one pair");
    }
  }

  declareTimestampAlignmentParams();
  if (!stereo_pair_names_.empty() && !align_stereo_timestamps_) {
    RCLCPP_INFO(
      get_logger(),
      "stereo_pairs is set but align_stereo_timestamps is false; no alignment will occur");
  }

  initialize(describePipelines());
}

std::vector<SiplCameraNode::PipelineDescriptor>
SiplMultiCameraNode::describePipelines()
{
  std::vector<PipelineDescriptor> descriptors;
  descriptors.reserve(cameras_.size());
  for (const auto & spec : cameras_) {
    PipelineDescriptor desc;
    desc.name = spec.name;
    desc.topic_prefix = spec.topic_namespace;
    desc.sensor_id = spec.sensor_id;
    desc.sensor_id_explicit = true;
    desc.frame_id = spec.frame_name + "_optical";
    desc.camera_info = spec.camera_info;
    desc.camera_info_loaded = spec.camera_info_loaded;
    desc.requested_output = spec.output;
    desc.raw_output = spec.raw_output;
    desc.encoding = spec.encoding;
    desc.nito_file = spec.nito_file;
    descriptors.push_back(std::move(desc));
  }
  return descriptors;
}

void SiplMultiCameraNode::publishStaticTransforms()
{
  const auto stamp = now();
  tf2::Quaternion optical_q;
  kCamLinkROptical.getRotation(optical_q);
  const tf2::Transform cam_link_pose_optical(kCamLinkROptical, tf2::Vector3(0.0, 0.0, 0.0));

  // Derive stereo poses from calibration; the URDF places independent cameras.
  const tf2::Transform identity(tf2::Matrix3x3::getIdentity(), tf2::Vector3(0.0, 0.0, 0.0));
  std::map<std::string, tf2::Transform> body_poses;
  const auto find_spec = [this](const std::string & name) -> const CameraSpec * {
      const auto it = std::find_if(
        cameras_.begin(), cameras_.end(),
        [&](const CameraSpec & s) {return s.name == name;});
      return it == cameras_.end() ? nullptr : &(*it);
    };
  for (const auto & [left_name, right_name] : stereo_pair_names_) {
    const CameraSpec * left = find_spec(left_name);
    const CameraSpec * right = find_spec(right_name);
    if (left == nullptr || right == nullptr) {
      RCLCPP_WARN(
        get_logger(), "stereo_pairs entry (%s, %s) names a camera absent from 'cameras'",
        left_name.c_str(), right_name.c_str());
      continue;
    }

    // Each stereo pair uses its left camera as the rig origin.
    body_poses[left_name] = identity;

    if (!left->camera_info_loaded || !right->camera_info_loaded) {
      RCLCPP_WARN(
        get_logger(),
        "Cannot derive the (%s, %s) baseline without both calibrations; leaving %s "
        "to the URDF",
        left_name.c_str(), right_name.c_str(), right_name.c_str());
      continue;
    }
    try {
      const tf2::Transform left_optical_pose_right_optical =
        computeLeftOpticalPoseRightOptical(left->camera_info, right->camera_info);
      const tf2::Transform cam_link_pose_right_optical =
        cam_link_pose_optical * left_optical_pose_right_optical;
      body_poses[right_name] =
        cam_link_pose_right_optical * cam_link_pose_optical.inverse();
    } catch (const std::exception & e) {
      RCLCPP_WARN(
        get_logger(), "Failed to compute the (%s, %s) baseline from calibration: %s",
        left_name.c_str(), right_name.c_str(), e.what());
    }
  }

  std::vector<geometry_msgs::msg::TransformStamped> transforms;
  transforms.reserve(cameras_.size() * 2U);
  std::string deferred;
  for (const auto & spec : cameras_) {
    const auto it = body_poses.find(spec.name);
    if (it == body_poses.end()) {
      deferred += (deferred.empty() ? "" : ", ") + spec.frame_name;
    } else {
      geometry_msgs::msg::TransformStamped body;
      body.header.stamp = stamp;
      body.header.frame_id = camera_link_frame_name_;
      body.child_frame_id = spec.frame_name;
      const tf2::Transform & pose = it->second;
      body.transform.translation.x = pose.getOrigin().getX();
      body.transform.translation.y = pose.getOrigin().getY();
      body.transform.translation.z = pose.getOrigin().getZ();
      const tf2::Quaternion body_q = pose.getRotation();
      body.transform.rotation.x = body_q.x();
      body.transform.rotation.y = body_q.y();
      body.transform.rotation.z = body_q.z();
      body.transform.rotation.w = body_q.w();
      transforms.push_back(body);
    }

    geometry_msgs::msg::TransformStamped optical;
    optical.header.stamp = stamp;
    optical.header.frame_id = spec.frame_name;
    optical.child_frame_id = spec.frame_name + "_optical";
    optical.transform.translation.x = 0.0;
    optical.transform.translation.y = 0.0;
    optical.transform.translation.z = 0.0;
    optical.transform.rotation.x = optical_q.x();
    optical.transform.rotation.y = optical_q.y();
    optical.transform.rotation.z = optical_q.z();
    optical.transform.rotation.w = optical_q.w();
    transforms.push_back(optical);
  }

  if (!deferred.empty()) {
    RCLCPP_INFO(
      get_logger(),
      "Not placing %s under %s; the URDF owns those frames. Their optical frames "
      "are still published.",
      deferred.c_str(), camera_link_frame_name_.c_str());
  }
  tf_static_broadcaster_->sendTransform(transforms);
  RCLCPP_INFO(
    get_logger(), "Published %zu static transform(s) for %zu camera(s)",
    transforms.size(), cameras_.size());
}

}  // namespace sipl
}  // namespace isaac_ros

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(isaac_ros::sipl::SiplMultiCameraNode)
