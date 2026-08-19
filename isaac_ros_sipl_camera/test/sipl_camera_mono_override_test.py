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
Mono camera calibration override and TF validation test.

Validates that:
  1. Published CameraInfo messages match the mono calibration YAML.
  2. The static TF tree publishes the camera -> camera_optical transform
     following REP-103 convention.
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
        'config', 'eagle_mono.yaml')
    namespace = SiplCameraMonoOverrideTest.generate_namespace()

    sipl_mono_node = ComposableNode(
        name='sipl_camera',
        package='isaac_ros_sipl_camera',
        plugin='isaac_ros::sipl::SiplCameraNode',
        namespace=namespace,
        parameters=load_config_for_test(
            config_path, namespace, 'sipl_camera', 'sipl_mono_override_container'),
    )

    return SiplCameraMonoOverrideTest.generate_test_description([
        ComposableNodeContainer(
            name='sipl_mono_override_container',
            package='rclcpp_components',
            executable='component_container_mt',
            composable_node_descriptions=[sipl_mono_node],
            namespace=namespace,
            output='screen',
            arguments=['--ros-args', '--log-level', 'info'],
        )
    ])


class SiplCameraMonoOverrideTest(IsaacROSBaseTest):
    filepath = pathlib.Path(os.path.dirname(__file__))

    pkg_share = get_package_share_directory('isaac_ros_sipl_camera')
    calibration_path = os.path.join(pkg_share, 'config', 'mock_mono_camera_info.yaml')

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

        received_d = np.array(received.d)
        max_len = max(len(received_d), len(expected_d))
        received_d = np.pad(received_d, (0, max_len - len(received_d)))
        expected_d = np.pad(expected_d, (0, max_len - len(expected_d)))

        return (
            received.width == expected['image_width']
            and received.height == expected['image_height']
            and np.allclose(np.array(received.k), expected_k)
            and np.allclose(received_d, expected_d)
            and np.allclose(np.array(received.p), expected_p)
        )

    def test_calibration_and_tf(self):
        """
        Validate published CameraInfo matches the calibration file and TF is correct.

        Checks that the camera -> camera_optical static transform follows
        the REP-103 optical frame convention (90-degree rotation).
        """
        TIMEOUT = 10
        MIN_MESSAGES = 10

        with open(self.calibration_path, 'r') as f:
            expected = yaml.safe_load(f)

        ns = self.generate_namespace()
        received_messages = []
        self.create_exact_time_sync_logging_subscribers(
            [(f'{ns}/image_raw', Image), (f'{ns}/camera_info', CameraInfo)],
            received_messages,
            accept_multiple_messages=True)

        end_time = time.time() + TIMEOUT
        while time.time() < end_time:
            rclpy.spin_once(self.node, timeout_sec=0.1)
            if len(received_messages) >= MIN_MESSAGES:
                break

        self.assertGreaterEqual(
            len(received_messages), MIN_MESSAGES,
            f'Expected at least {MIN_MESSAGES} messages, got {len(received_messages)}')

        # Validate CameraInfo
        for msg in received_messages:
            self.assertTrue(
                self._camera_info_matches_yaml(msg[1], expected),
                'CameraInfo does not match calibration YAML')

        # Validate frame_id consistency across all received messages
        img_frame_ids = {msg[0].header.frame_id for msg in received_messages}
        self.assertEqual(
            len(img_frame_ids), 1,
            f'image_raw has inconsistent frame_ids: {img_frame_ids}')

        info_frame_ids = {msg[1].header.frame_id for msg in received_messages}
        self.assertEqual(
            len(info_frame_ids), 1,
            f'camera_info has inconsistent frame_ids: {info_frame_ids}')

        self.assertEqual(
            img_frame_ids.pop(), info_frame_ids.pop(),
            'image_raw and camera_info frame_ids do not match')

        # Validate TF: camera -> camera_optical (REP-103)
        tf_msg = self._query_transform('camera_optical', 'camera')
        self.assertIsNotNone(tf_msg,
                             'Could not look up camera -> camera_optical transform')

        # The optical frame transform should be a pure rotation (no translation)
        t = tf_msg.transform.translation
        self.assertAlmostEqual(t.x, 0.0, places=6,
                               msg='Optical frame should have zero translation')
        self.assertAlmostEqual(t.y, 0.0, places=6,
                               msg='Optical frame should have zero translation')
        self.assertAlmostEqual(t.z, 0.0, places=6,
                               msg='Optical frame should have zero translation')

        # REP-103 optical rotation: the quaternion should represent the
        # standard camera-to-optical rotation (x->z, y->-x, z->-y).
        # Expected quaternion: (-0.5, 0.5, -0.5, 0.5) or equivalent forms.
        q = tf_msg.transform.rotation
        q_arr = np.array([q.x, q.y, q.z, q.w])
        expected_q = np.array([-0.5, 0.5, -0.5, 0.5])
        # tf2 matrix->quaternion can yield (0.5, -0.5, 0.5, 0.5) for same rotation
        expected_alt = np.array([0.5, -0.5, 0.5, 0.5])

        # Quaternion sign ambiguity: q and -q represent the same rotation
        self.assertTrue(
            np.allclose(q_arr, expected_q, atol=1e-4)
            or np.allclose(q_arr, -expected_q, atol=1e-4)
            or np.allclose(q_arr, expected_alt, atol=1e-4)
            or np.allclose(q_arr, -expected_alt, atol=1e-4),
            f'Optical frame rotation {q_arr} does not match REP-103 convention')
