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
Basic mono capture smoke test for the SIPL camera driver.

Validates that the mono pipeline publishes image_raw and camera_info with
matching timestamps.
"""

import os
import pathlib
import time

from ament_index_python.packages import get_package_share_directory
from isaac_ros_test import IsaacROSBaseTest
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode
import launch_testing
import pytest
import rclpy

from sensor_msgs.msg import CameraInfo, Image

from sipl_camera_test_utils import (
    detect_ethernet_camera, EXPECTED_FPS,
    load_config_for_test, STARTUP_TIME_MAX_DELAY,
)

TIMEOUT = 15
# Require a meaningful number of frames but cap at 3 seconds worth.
MIN_SYNCED_MSGS = min((TIMEOUT - STARTUP_TIME_MAX_DELAY) * EXPECTED_FPS, EXPECTED_FPS * 3)

# Requested SIPL output format on the publishing node.
ENCODING_DESIRED = 'nv12'
# Python subscribers receive CPU-adapted images from NITROS as rgb8.
EXPECTED_SUBSCRIBER_ENCODING = 'rgb8'
RGB8_BYTES_PER_PIXEL = 3.0


@pytest.mark.rostest
def generate_test_description():
    if detect_ethernet_camera():
        SiplCameraMonoTest.skip_test = False

        config_path = os.path.join(
            get_package_share_directory('isaac_ros_sipl_camera'),
            'config', 'eagle_mono.yaml')
        namespace = SiplCameraMonoTest.generate_namespace()

        sipl_mono_node = ComposableNode(
            name='sipl_camera',
            package='isaac_ros_sipl_camera',
            plugin='isaac_ros::sipl::SiplCameraNode',
            namespace=namespace,
            parameters=load_config_for_test(
                config_path, namespace, 'sipl_camera', 'sipl_mono_container')
            + [{'encoding_desired': ENCODING_DESIRED}],
        )

        return SiplCameraMonoTest.generate_test_description([
            ComposableNodeContainer(
                name='sipl_mono_container',
                package='rclcpp_components',
                executable='component_container_mt',
                composable_node_descriptions=[sipl_mono_node],
                namespace=namespace,
                output='screen',
                arguments=['--ros-args', '--log-level', 'info'],
            )
        ])
    else:
        SiplCameraMonoTest.skip_test = True
        return SiplCameraMonoTest.generate_test_description(
            [launch_testing.actions.ReadyToTest()])


class SiplCameraMonoTest(IsaacROSBaseTest):
    filepath = pathlib.Path(os.path.dirname(__file__))
    skip_test = False

    def test_mono_capture(self):
        """
        Verify that the SIPL mono pipeline publishes image and camera_info.

        Asserts that (image_raw, camera_info) pair messages are
        received within the timeout.
        """
        if self.skip_test:
            self.skipTest('No SIPL camera detected. Skipping test.')

        received_messages = []

        self.create_exact_time_sync_logging_subscribers(
            [('image_raw', Image), ('camera_info', CameraInfo)],
            received_messages,
            accept_multiple_messages=True)

        end_time = time.time() + TIMEOUT
        while time.time() < end_time:
            rclpy.spin_once(self.node, timeout_sec=0.1)
            if len(received_messages) >= MIN_SYNCED_MSGS:
                break

        self.assertGreaterEqual(
            len(received_messages), MIN_SYNCED_MSGS,
            f'Insufficient synced image + camera_info messages: '
            f'{len(received_messages)} < {MIN_SYNCED_MSGS}')
        for img, info in received_messages:
            self.assertEqual(
                img.header.stamp, info.header.stamp,
                'Image and camera_info timestamps do not match')
            self.assertEqual(
                img.encoding, EXPECTED_SUBSCRIBER_ENCODING,
                f'Expected encoding {EXPECTED_SUBSCRIBER_ENCODING}, got {img.encoding}')
            min_step = int(img.width * RGB8_BYTES_PER_PIXEL)
            self.assertGreaterEqual(
                img.step, min_step,
                f'Image step ({img.step}) must be >= '
                f'width * bytes_per_pixel ({min_step})')

            rgb8_min_size = int(img.width * img.height * RGB8_BYTES_PER_PIXEL)
            self.assertGreaterEqual(
                len(img.data), rgb8_min_size,
                f'Data size {len(img.data)} is smaller than RGB8 minimum '
                f'({rgb8_min_size}) for {img.width}x{img.height}')
