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
Stereo capture tests for the SIPL camera driver.

test_stereo_capture: validates that both left and right pipelines come up and
publish messages. Each side (left image + camera_info) is verified independently.

test_stereo_exact_time_sync: validates cross-camera exact time synchronization
(all four topics share the same timestamp).
"""

import os
import pathlib
import time
import unittest

from ament_index_python.packages import get_package_share_directory
from flaky import flaky
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
        SiplCameraStereoTest.skip_test = False

        config_path = os.path.join(
            get_package_share_directory('isaac_ros_sipl_camera'),
            'config', 'eagle_stereo.yaml')
        namespace = SiplCameraStereoTest.generate_namespace()

        sipl_stereo_node = ComposableNode(
            name='sipl_stereo_camera',
            package='isaac_ros_sipl_camera',
            plugin='isaac_ros::sipl::SiplStereoCameraNode',
            namespace=namespace,
            parameters=load_config_for_test(
                config_path, namespace, 'sipl_stereo_camera', 'sipl_stereo_container')
            + [{'encoding_desired': ENCODING_DESIRED}],
        )

        return SiplCameraStereoTest.generate_test_description([
            ComposableNodeContainer(
                name='sipl_stereo_container',
                package='rclcpp_components',
                executable='component_container_mt',
                composable_node_descriptions=[sipl_stereo_node],
                namespace=namespace,
                output='screen',
                arguments=['--ros-args', '--log-level', 'info'],
            )
        ])
    else:
        SiplCameraStereoTest.skip_test = True
        return SiplCameraStereoTest.generate_test_description(
            [launch_testing.actions.ReadyToTest()])


class SiplCameraStereoTest(IsaacROSBaseTest):
    filepath = pathlib.Path(os.path.dirname(__file__))
    skip_test = False

    # Temporarily mitigate flaky SIPL CoE camera initialization issues.
    @flaky(max_runs=3, min_passes=1)
    def test_stereo_capture(self):
        """
        Verify that both left and right SIPL pipelines publish image and camera_info.

        Each side is checked independently here.
        """
        if self.skip_test:
            self.skipTest('No SIPL camera detected. Skipping test.')

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

        self.assertGreaterEqual(
            len(left_messages), MIN_SYNCED_MSGS,
            f'Left camera produced too few synced image + camera_info pairs: '
            f'{len(left_messages)} < {MIN_SYNCED_MSGS}')
        self.assertGreaterEqual(
            len(right_messages), MIN_SYNCED_MSGS,
            f'Right camera produced too few synced image + camera_info pairs: '
            f'{len(right_messages)} < {MIN_SYNCED_MSGS}')

        # Verify per-side timestamp consistency, encoding, and data size
        for side, messages in [('Left', left_messages), ('Right', right_messages)]:
            for img, info in messages:
                self.assertEqual(
                    img.header.stamp, info.header.stamp,
                    f'{side} image and camera_info timestamps do not match')
                self.assertEqual(
                    img.encoding, EXPECTED_SUBSCRIBER_ENCODING,
                    f'{side} expected encoding {EXPECTED_SUBSCRIBER_ENCODING}, '
                    f'got {img.encoding}')
                min_step = int(img.width * RGB8_BYTES_PER_PIXEL)
                self.assertGreaterEqual(
                    img.step, min_step,
                    f'{side} image step ({img.step}) must be >= '
                    f'width * bytes_per_pixel ({min_step})')

                rgb8_min_size = int(img.width * img.height * RGB8_BYTES_PER_PIXEL)
                self.assertGreaterEqual(
                    len(img.data), rgb8_min_size,
                    f'{side} data size {len(img.data)} is smaller than RGB8 '
                    f'minimum ({rgb8_min_size}) for {img.width}x{img.height}')

        # Verify frame_id consistency within each side
        for side, messages in [('Left', left_messages), ('Right', right_messages)]:
            img_frame_ids = {msg[0].header.frame_id for msg in messages}
            self.assertEqual(
                len(img_frame_ids), 1,
                f'{side} image_raw has inconsistent frame_ids: {img_frame_ids}')

            info_frame_ids = {msg[1].header.frame_id for msg in messages}
            self.assertEqual(
                len(info_frame_ids), 1,
                f'{side} camera_info has inconsistent frame_ids: {info_frame_ids}')

            self.assertEqual(
                img_frame_ids.pop(), info_frame_ids.pop(),
                f'{side} image_raw and camera_info frame_ids do not match')

    @unittest.skip('SIPL driver does not yet guarantee frame-level sync')
    @flaky(max_runs=3, min_passes=1)
    def test_stereo_exact_time_sync(self):
        """
        Verify cross-camera exact time synchronization.

        All four topics (left/image_raw, right/image_raw, left/camera_info,
        right/camera_info) must share the same timestamp.
        """
        if self.skip_test:
            self.skipTest('No SIPL camera detected. Skipping test.')

        received_messages = []

        self.create_exact_time_sync_logging_subscribers(
            [('left/image_raw', Image), ('right/image_raw', Image),
             ('left/camera_info', CameraInfo), ('right/camera_info', CameraInfo)],
            received_messages,
            accept_multiple_messages=True)

        end_time = time.time() + TIMEOUT
        while time.time() < end_time:
            rclpy.spin_once(self.node, timeout_sec=0.1)
            if len(received_messages) > MIN_SYNCED_MSGS:
                break

        self.assertTrue(
            len(received_messages) > MIN_SYNCED_MSGS,
            'No cross-camera time-synced messages received')

        # This should basically be guaranteed by the exact time sync subscriber.
        for msg in received_messages:
            self.assertTrue(
                msg[0].header.stamp == msg[1].header.stamp
                and msg[1].header.stamp == msg[2].header.stamp
                and msg[2].header.stamp == msg[3].header.stamp,
                'Timestamps of all images and camera infos are not equal')
