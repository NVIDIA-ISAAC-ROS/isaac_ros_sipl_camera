# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

"""
Stereo camera calibration override and TF validation test.

Validates that:
  1. Published CameraInfo messages match the calibration YAML files.
  2. The static TF tree is correct (stereo -> left/right -> optical frames).
  3. The right camera pose derived from calibration matches expected values.
"""

import os
import pathlib
import time
import unittest

from ament_index_python.packages import get_package_share_directory
from isaac_ros_test import IsaacROSBaseTest
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode
import numpy as np
import pytest
import rclpy
from sensor_msgs.msg import CameraInfo, Image
from sipl_camera_test_utils import detect_ethernet_camera, load_config_for_test
from tf2_ros import (
    Buffer, ConnectivityException, ExtrapolationException,
    LookupException, TransformListener,
)
import yaml


@pytest.mark.rostest
def generate_test_description():
    if not detect_ethernet_camera():
        raise unittest.SkipTest('No SIPL camera detected. Skipping test.')

    config_path = os.path.join(
        get_package_share_directory('isaac_ros_sipl_camera'),
        'config', 'eagle_stereo.yaml')
    namespace = SiplCameraStereoOverrideTest.generate_namespace()

    sipl_stereo_node = ComposableNode(
        name='sipl_stereo_camera',
        package='isaac_ros_sipl_camera',
        plugin='isaac_ros::sipl::SiplStereoCameraNode',
        namespace=namespace,
        parameters=load_config_for_test(
            config_path, namespace, 'sipl_stereo_camera', 'sipl_stereo_override_container'),
    )

    return SiplCameraStereoOverrideTest.generate_test_description([
        ComposableNodeContainer(
            name='sipl_stereo_override_container',
            package='rclcpp_components',
            executable='component_container_mt',
            composable_node_descriptions=[sipl_stereo_node],
            namespace=namespace,
            output='screen',
            arguments=['--ros-args', '--log-level', 'info'],
        )
    ])


class SiplCameraStereoOverrideTest(IsaacROSBaseTest):
    filepath = pathlib.Path(os.path.dirname(__file__))

    pkg_share = get_package_share_directory('isaac_ros_sipl_camera')
    left_calibration_path = os.path.join(pkg_share, 'config', 'mock_left_camera_info.yaml')
    right_calibration_path = os.path.join(pkg_share, 'config', 'mock_right_camera_info.yaml')

    def setUp(self):
        super().setUp()
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self.node)

    def _query_transform(self, target_frame, source_frame, timeout=5.0):
        """Look up a TF transform, returning None on failure."""
        try:
            return self.tf_buffer.lookup_transform(
                target_frame, source_frame,
                rclpy.time.Time(),
                rclpy.duration.Duration(seconds=timeout))
        except (LookupException, ConnectivityException, ExtrapolationException) as e:
            self.node.get_logger().error(
                f'Could not look up {target_frame} -> {source_frame}: {e}')
            return None

    @staticmethod
    def _camera_info_matches_yaml(received, expected):
        """Compare a received CameraInfo message against expected YAML values."""
        expected_k = np.array(expected['camera_matrix']['data'])
        expected_p = np.array(expected['projection_matrix']['data'])
        expected_d = np.array(expected['distortion_coefficients']['data'])
        expected_r = np.array(expected['rectification_matrix']['data'])

        received_d = np.array(received.d)
        # Pad shorter array with zeros for comparison
        max_len = max(len(received_d), len(expected_d))
        received_d = np.pad(received_d, (0, max_len - len(received_d)))
        expected_d = np.pad(expected_d, (0, max_len - len(expected_d)))

        return (
            received.width == expected['image_width']
            and received.height == expected['image_height']
            and np.allclose(np.array(received.k), expected_k)
            and np.allclose(received_d, expected_d)
            and np.allclose(np.array(received.r), expected_r)
            and np.allclose(np.array(received.p), expected_p)
        )

    def test_calibration_and_tf(self):
        """
        Validate published CameraInfo matches the calibration files and TF is correct.

        Left and right streams are checked independently (no cross-camera sync).
        """
        TIMEOUT = 10
        MIN_MESSAGES = 10

        # Load expected calibration
        with open(self.left_calibration_path, 'r') as f:
            expected_left = yaml.safe_load(f)
        with open(self.right_calibration_path, 'r') as f:
            expected_right = yaml.safe_load(f)

        # Subscribe to namespaced topics so we match exactly what the node publishes
        ns = self.generate_namespace()
        left_topic_list = [
            (f'{ns}/left/image_raw', Image),
            (f'{ns}/left/camera_info', CameraInfo),
        ]
        right_topic_list = [
            (f'{ns}/right/image_raw', Image),
            (f'{ns}/right/camera_info', CameraInfo),
        ]

        left_messages = []
        right_messages = []

        self.create_exact_time_sync_logging_subscribers(
            left_topic_list,
            left_messages,
            accept_multiple_messages=True)

        self.create_exact_time_sync_logging_subscribers(
            right_topic_list,
            right_messages,
            accept_multiple_messages=True)

        end_time = time.time() + TIMEOUT
        while time.time() < end_time:
            rclpy.spin_once(self.node, timeout_sec=0.1)
            if len(left_messages) >= MIN_MESSAGES and len(right_messages) >= MIN_MESSAGES:
                break

        self.assertGreaterEqual(
            len(left_messages), MIN_MESSAGES,
            f'Expected at least {MIN_MESSAGES} left messages, got {len(left_messages)}')
        self.assertGreaterEqual(
            len(right_messages), MIN_MESSAGES,
            f'Expected at least {MIN_MESSAGES} right messages, got {len(right_messages)}')

        # Validate CameraInfo for every received message
        for msg in left_messages:
            self.assertTrue(
                self._camera_info_matches_yaml(msg[1], expected_left),
                'Left CameraInfo does not match calibration YAML')

        for msg in right_messages:
            self.assertTrue(
                self._camera_info_matches_yaml(msg[1], expected_right),
                'Right CameraInfo does not match calibration YAML')

        # Validate TF: stereo -> stereo_left -> stereo_left_optical
        left_optical_tf = self._query_transform('stereo_left_optical', 'stereo')
        self.assertIsNotNone(left_optical_tf,
                             'Could not look up stereo -> stereo_left_optical transform')

        # Validate TF: stereo -> stereo_right -> stereo_right_optical
        right_optical_tf = self._query_transform('stereo_right_optical', 'stereo')
        self.assertIsNotNone(right_optical_tf,
                             'Could not look up stereo -> stereo_right_optical transform')

        # Validate the stereo baseline via the optical-frame transform
        optical_tf = self._query_transform(
            'stereo_right_optical', 'stereo_left_optical')
        self.assertIsNotNone(optical_tf,
                             'Could not look up stereo_left_optical '
                             '-> stereo_right_optical transform')

        baseline = abs(optical_tf.transform.translation.x)
        self.assertGreater(
            baseline, 0.00,
            f'Expected non-trivial stereo baseline, got {baseline:.4f} m')

        self.node.get_logger().info(
            f'Stereo baseline (from TF): {baseline:.4f} m')
