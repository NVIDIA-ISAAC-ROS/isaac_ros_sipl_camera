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
Basic mono capture with format conversion test for the SIPL camera driver.

Validates that the mono pipeline combined with the format converter publishes
image_converted and camera_info with matching timestamps.
"""

import os
import pathlib
import time

from ament_index_python.packages import get_package_share_directory
from flaky import flaky
from isaac_ros_test import IsaacROSBaseTest
import launch
import launch_ros
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

# Requested format conversion output encoding.
ENCODING_DESIRED = 'mono8'


@pytest.mark.rostest
def generate_test_description():
    if detect_ethernet_camera():
        SiplCameraMonoFormatConversionTest.skip_test = False

        config_path = os.path.join(
            get_package_share_directory('isaac_ros_sipl_camera'),
            'config', 'eagle_mono.yaml')
        namespace = SiplCameraMonoFormatConversionTest.generate_namespace()

        # SIPL monocular camera node (produces nv12 by default if not set)
        sipl_mono_node = ComposableNode(
            name='sipl_camera',
            package='isaac_ros_sipl_camera',
            plugin='isaac_ros::sipl::SiplCameraNode',
            namespace=namespace,
            parameters=load_config_for_test(
                config_path, namespace, 'sipl_camera', 'sipl_mono_container')
        )

        image_width = 2560
        image_height = 1984

        format_converter = ComposableNode(
            name='format_converter',
            package='isaac_ros_image_proc',
            plugin='nvidia::isaac_ros::image_proc::ImageFormatConverterNode',
            namespace=namespace,
            parameters=[{
                'encoding_desired': ENCODING_DESIRED,
                'image_width': image_width,
                'image_height': image_height,
                'num_blocks': 40,
            }],
            remappings=[
                ('image', 'image_converted'),
            ],
        )

        container = ComposableNodeContainer(
            name='sipl_mono_container',
            package='rclcpp_components',
            executable='component_container_mt',
            namespace=namespace,
            output='screen',
            arguments=['--ros-args', '--log-level', 'info'],
        )

        load_format_converter = launch_ros.actions.LoadComposableNodes(
            target_container=container,
            composable_node_descriptions=[format_converter],
        )

        load_camera = launch_ros.actions.LoadComposableNodes(
            target_container=container,
            composable_node_descriptions=[sipl_mono_node],
        )

        # Delay before loading the SIPL camera node when downstream NITROS nodes are
        # present to avoid crash from GXF graph slow initialization.
        deferred_camera = launch.actions.TimerAction(
            period=2.0,
            actions=[load_camera],
        )

        return SiplCameraMonoFormatConversionTest.generate_test_description([
            container, load_format_converter, deferred_camera
        ])
    else:
        SiplCameraMonoFormatConversionTest.skip_test = True
        return SiplCameraMonoFormatConversionTest.generate_test_description(
            [launch_testing.actions.ReadyToTest()])


class SiplCameraMonoFormatConversionTest(IsaacROSBaseTest):
    filepath = pathlib.Path(os.path.dirname(__file__))
    skip_test = False

    # Temporarily mitigate flaky SIPL CoE camera initialization issues.
    @flaky(max_runs=3, min_passes=1)
    def test_mono_format_conversion_capture(self):
        """
        Verify that the format conversion node publishes image_converted properly.

        Asserts that time-synced (image_converted, camera_info) pair messages are
        received within the timeout.
        """
        if self.skip_test:
            self.skipTest('No SIPL camera detected. Skipping test.')

        received_messages = []

        self.create_exact_time_sync_logging_subscribers(
            [('image_converted', Image), ('camera_info', CameraInfo)],
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
                img.encoding, ENCODING_DESIRED,
                f'Expected encoding {ENCODING_DESIRED}, got {img.encoding}')
            self.assertGreaterEqual(
                img.step, img.width,
                f'Image step ({img.step}) must be >= width ({img.width})')

            min_size = img.width * img.height
            self.assertGreaterEqual(
                len(img.data), min_size,
                f'Data size {len(img.data)} is smaller than mono8 minimum '
                f'({min_size}) for {img.width}x{img.height}')
