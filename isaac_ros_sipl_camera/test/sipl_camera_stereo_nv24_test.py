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
Stereo NV24 encoding test for the SIPL camera driver.

Launches the stereo pipeline with encoding_desired='nv24' and validates that
it published images.
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
ENCODING_DESIRED = 'nv24'
# Python subscribers receive CPU-adapted images from NITROS as rgb8.
EXPECTED_SUBSCRIBER_ENCODING = 'rgb8'
RGB8_BYTES_PER_PIXEL = 3.0


@pytest.mark.rostest
def generate_test_description():
    if detect_ethernet_camera():
        SiplCameraStereoNv24Test.skip_test = False

        config_path = os.path.join(
            get_package_share_directory('isaac_ros_sipl_camera'),
            'config', 'eagle_stereo.yaml')
        namespace = SiplCameraStereoNv24Test.generate_namespace()

        sipl_stereo_node = ComposableNode(
            name='sipl_stereo_camera',
            package='isaac_ros_sipl_camera',
            plugin='isaac_ros::sipl::SiplStereoCameraNode',
            namespace=namespace,
            parameters=load_config_for_test(
                config_path, namespace, 'sipl_stereo_camera',
                'sipl_stereo_nv24_container')
            + [{'encoding_desired': ENCODING_DESIRED}],
        )

        return SiplCameraStereoNv24Test.generate_test_description([
            ComposableNodeContainer(
                name='sipl_stereo_nv24_container',
                package='rclcpp_components',
                executable='component_container_mt',
                composable_node_descriptions=[sipl_stereo_node],
                namespace=namespace,
                output='screen',
                arguments=['--ros-args', '--log-level', 'info'],
            )
        ])
    else:
        SiplCameraStereoNv24Test.skip_test = True
        return SiplCameraStereoNv24Test.generate_test_description(
            [launch_testing.actions.ReadyToTest()])


class SiplCameraStereoNv24Test(IsaacROSBaseTest):
    filepath = pathlib.Path(os.path.dirname(__file__))
    skip_test = False

    def _receive_stereo_messages(self):
        """Spin until at least minimum number of synced pairs arrive for each side."""
        left_messages = []
        right_messages = []

        self.create_exact_time_sync_logging_subscribers(
            [('left/image_raw', Image), ('left/camera_info', CameraInfo)],
            left_messages,
            accept_multiple_messages=True)

        self.create_exact_time_sync_logging_subscribers(
            [('right/image_raw', Image), ('right/camera_info', CameraInfo)],
            right_messages,
            accept_multiple_messages=True)

        end_time = time.time() + TIMEOUT
        while time.time() < end_time:
            rclpy.spin_once(self.node, timeout_sec=0.1)
            if (len(left_messages) >= MIN_SYNCED_MSGS
                    and len(right_messages) >= MIN_SYNCED_MSGS):
                break
        return left_messages, right_messages

    def test_rgb8_encoding_and_data_size(self):
        """Verify encoding is RGB8 and data size is consistent (3 bytes/pixel)."""
        if self.skip_test:
            self.skipTest('No SIPL camera detected. Skipping test.')

        left_messages, right_messages = self._receive_stereo_messages()

        self.assertGreaterEqual(
            len(left_messages), MIN_SYNCED_MSGS,
            f'Left camera produced too few synced messages: '
            f'{len(left_messages)} < {MIN_SYNCED_MSGS}')
        self.assertGreaterEqual(
            len(right_messages), MIN_SYNCED_MSGS,
            f'Right camera produced too few synced messages: '
            f'{len(right_messages)} < {MIN_SYNCED_MSGS}')

        for side, messages in [('Left', left_messages), ('Right', right_messages)]:
            for img, _ in messages:
                self.assertEqual(
                    img.encoding, EXPECTED_SUBSCRIBER_ENCODING,
                    f'{side} image encoding: expected {EXPECTED_SUBSCRIBER_ENCODING}, '
                    f'got {img.encoding}')

                self.assertGreater(img.width, 0,
                                   f'{side} image width must be positive')
                self.assertGreater(img.height, 0,
                                   f'{side} image height must be positive')
                min_step = int(img.width * RGB8_BYTES_PER_PIXEL)
                self.assertGreaterEqual(
                    img.step, min_step,
                    f'{side} image step ({img.step}) must be >= '
                    f'width * bytes_per_pixel ({min_step})')

                rgb8_min_size = img.width * img.height * RGB8_BYTES_PER_PIXEL
                self.assertGreaterEqual(
                    len(img.data), rgb8_min_size,
                    f'{side} data size {len(img.data)} is smaller than RGB8 '
                    f'minimum ({rgb8_min_size}) for {img.width}x{img.height}')
