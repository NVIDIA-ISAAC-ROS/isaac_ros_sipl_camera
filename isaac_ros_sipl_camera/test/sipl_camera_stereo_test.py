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

The test description is parametrized over every entry in STEREO_CONFIGS
(Eagle CoE and Hawk GMSL today). launch_testing runs the whole test class once
per config; each run is gated by that config's hardware detector and skips if the
camera is not detected.

test_stereo_capture: validates that both left and right pipelines come up and
publish messages. Each side (left image + camera_info) is verified independently.

test_stereo_camera_info_receive_latency: validates that both camera timestamps
remain below the SOF-to-CameraInfo callback latency limit for every stereo config.

test_stereo_exact_time_sync: validates cross-camera exact time synchronization
(all four topics share the same timestamp) when enabled by the camera configuration.
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
    camera_present, EXPECTED_FPS, load_config_for_test,
    STARTUP_TIME_MAX_DELAY, STEREO_CONFIGS,
)

TIMEOUT = 15
# Require a meaningful number of frames but cap at 3 seconds worth.
MIN_SYNCED_MSGS = min((TIMEOUT - STARTUP_TIME_MAX_DELAY) * EXPECTED_FPS, EXPECTED_FPS * 3)

# A stereo timing regression appeared as a persistent one-frame timestamp offset.
CAMERA_INFO_LATENCY_WARMUP_FRAMES = EXPECTED_FPS
CAMERA_INFO_LATENCY_SAMPLE_FRAMES = EXPECTED_FPS * 2
CAMERA_INFO_LATENCY_MAX_MS = 50.0

# Requested SIPL output format on the publishing node.
ENCODING_DESIRED = 'nv12'
# Python subscribers receive CPU-adapted images from NITROS as rgb8.
EXPECTED_SUBSCRIBER_ENCODING = 'rgb8'
RGB8_BYTES_PER_PIXEL = 3.0


@pytest.mark.rostest
@launch_testing.parametrize('config_name', STEREO_CONFIGS)
def generate_test_description(config_name):
    # generate_test_description() is invoked once per config, immediately before
    # that config's tests run, so storing state on the class is safe here.
    SiplCameraStereoTest.config_name = config_name

    if not camera_present(config_name):
        raise unittest.SkipTest(
            f'No SIPL camera detected for {config_name}. Skipping test.')

    config_path = os.path.join(
        get_package_share_directory('isaac_ros_sipl_camera'),
        'config', config_name)
    namespace = SiplCameraStereoTest.generate_namespace()
    config_parameters = load_config_for_test(
        config_path, namespace, 'sipl_stereo_camera', 'sipl_stereo_container')
    align_stereo_timestamps = config_parameters[0].get('align_stereo_timestamps', False)
    if not isinstance(align_stereo_timestamps, bool):
        raise ValueError(
            f'align_stereo_timestamps must be true or false in {config_name}')
    SiplCameraStereoTest.supports_stereo_pairing = align_stereo_timestamps

    sipl_stereo_node = ComposableNode(
        name='sipl_stereo_camera',
        package='isaac_ros_sipl_camera',
        plugin='isaac_ros::sipl::SiplStereoCameraNode',
        namespace=namespace,
        parameters=config_parameters + [{
            'encoding_desired': ENCODING_DESIRED,
            # Latency checks measure SOF stamp -> callback; require HW SOF timestamps.
            'use_hw_timestamp': True,
        }],
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


class SiplCameraStereoTest(IsaacROSBaseTest):
    filepath = pathlib.Path(os.path.dirname(__file__))
    # Set in generate_test_description() so messages identify the active config.
    config_name = ''
    supports_stereo_pairing = False

    def test_stereo_camera_info_receive_latency(self):
        """Verify left and right SOF-to-CameraInfo callback latency.

        Requires use_hw_timestamp=True so messages are stamped with the SOF timestamp.
        We subscribe to camera_info messages since it's cheaper than subscribing to image_raw
        but uses the same timestamp.
        """
        topics = ('left/camera_info', 'right/camera_info')
        received_messages = {}
        self.create_logging_subscribers(
            [(topic, CameraInfo) for topic in topics],
            received_messages,
            use_namespace_lookup=False,
            accept_multiple_messages=True,
            add_received_message_timestamps=True)

        required_frames = (
            CAMERA_INFO_LATENCY_WARMUP_FRAMES + CAMERA_INFO_LATENCY_SAMPLE_FRAMES)
        end_time = time.time() + TIMEOUT
        while time.time() < end_time:
            rclpy.spin_once(self.node, timeout_sec=0.1)
            if all(len(received_messages[topic]) >= required_frames for topic in topics):
                break

        for topic in topics:
            self.assertGreaterEqual(
                len(received_messages[topic]), required_frames,
                f'[{self.config_name}] {topic} produced too few CameraInfo messages: '
                f'{len(received_messages[topic])} < {required_frames}')

            latency_ms = []
            sampled_messages = received_messages[topic][
                CAMERA_INFO_LATENCY_WARMUP_FRAMES:required_frames]
            for message, received_time_s in sampled_messages:
                stamped_time_s = (
                    message.header.stamp.sec + message.header.stamp.nanosec / 1e9)
                latency_ms.append((received_time_s - stamped_time_s) * 1e3)

            self.assertGreaterEqual(
                min(latency_ms), 0.0,
                f'[{self.config_name}] {topic} has a timestamp later than callback receipt')
            max_latency_ms = max(latency_ms)
            print(
                f'[{self.config_name}] {topic} SOF-to-receive '
                f'maximum: {max_latency_ms:.3f} ms')
            self.assertLess(
                max_latency_ms, CAMERA_INFO_LATENCY_MAX_MS,
                f'[{self.config_name}] {topic} SOF-to-receive '
                f'maximum latency {max_latency_ms:.3f} ms is not below '
                f'{CAMERA_INFO_LATENCY_MAX_MS:.1f} ms')

    # Temporarily mitigate flaky SIPL CoE camera initialization issues.
    @flaky(max_runs=3, min_passes=1)
    def test_stereo_capture(self):
        """
        Verify that both left and right SIPL pipelines publish image and camera_info.

        Each side is checked independently here.
        """
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
            f'[{self.config_name}] Left camera produced too few synced '
            f'image + camera_info pairs: {len(left_messages)} < {MIN_SYNCED_MSGS}')
        self.assertGreaterEqual(
            len(right_messages), MIN_SYNCED_MSGS,
            f'[{self.config_name}] Right camera produced too few synced '
            f'image + camera_info pairs: {len(right_messages)} < {MIN_SYNCED_MSGS}')

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

    @flaky(max_runs=3, min_passes=1)
    def test_stereo_exact_time_sync(self):
        """
        Verify cross-camera exact time synchronization.

        For configurations that support timestamp alignment, all four topics
        (left/image_raw, right/image_raw, left/camera_info, right/camera_info)
        must share the same timestamp.
        """
        if not self.supports_stereo_pairing:
            self.skipTest(
                f'[{self.config_name}] Stereo timestamp alignment is disabled by configuration')

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
            f'[{self.config_name}] No cross-camera time-synced messages received')

        # This should basically be guaranteed by the exact time sync subscriber.
        for msg in received_messages:
            self.assertTrue(
                msg[0].header.stamp == msg[1].header.stamp
                and msg[1].header.stamp == msg[2].header.stamp
                and msg[2].header.stamp == msg[3].header.stamp,
                'Timestamps of all images and camera infos are not equal')
