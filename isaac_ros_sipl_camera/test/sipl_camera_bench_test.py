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
Stream stability and timing-consistency test for the SIPL stereo camera driver.

Measures FPS and jitter for each camera stream independently (left and right
are treated as separate pipelines). This test runs even when the two sensors
do not share a common capture timestamp.

The test description is parametrized over every entry in STEREO_CONFIGS
(Eagle CoE and Hawk GMSL today). launch_testing runs the benchmark once per
config; each run is gated by that config's hardware detector and skips if the
camera is not detected.

Metrics reported per stream:
  - Frame count
  - Mean frame rate (expected nominal: EXPECTED_FPS)
  - Max / min / mean jitter
  - Percentage of frames exceeding the jitter tolerance

Pass criteria:
  - Frame count >= Minimum acceptable frame count
  - FPS within EXPECTED_FPS +/- FPS_ACCEPTABLE_VARIANCE
  - Jitter-exceed percentage below EXCEED_TOLERANCE_PERCENT_LIMIT
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
import numpy
import pytest
import rclpy

from sensor_msgs.msg import CameraInfo, Image

from sipl_camera_test_utils import (
    camera_present, EXPECTED_FPS, load_config_for_test,
    STARTUP_TIME_MAX_DELAY, STEREO_CONFIGS,
)

# Stream timing thresholds (stereo SIPL cameras nominally at EXPECTED_FPS)
FPS_ACCEPTABLE_VARIANCE = 5
MIN_ACCEPTABLE_FPS = EXPECTED_FPS - FPS_ACCEPTABLE_VARIANCE
MAX_ACCEPTABLE_FPS = EXPECTED_FPS + FPS_ACCEPTABLE_VARIANCE
JITTER_TOLERANCE_MS = 5
EXCEED_TOLERANCE_PERCENT_LIMIT = 5

TIMEOUT = 30
STABILITY_MIN_MSGS = (TIMEOUT - STARTUP_TIME_MAX_DELAY) * MIN_ACCEPTABLE_FPS

SEC_TO_MS = 10**3
NS_TO_MS = 1e-6
MAX_ROW_WIDTH = 70


@pytest.mark.rostest
@launch_testing.parametrize('config_name', STEREO_CONFIGS)
def generate_test_description(config_name):
    # generate_test_description() is invoked once per config, immediately before
    # that config's tests run, so storing state on the class is safe here.
    SiplCameraBenchTest.config_name = config_name

    if not camera_present(config_name):
        raise unittest.SkipTest(
            f'No SIPL camera detected for {config_name}. Skipping test.')

    config_path = os.path.join(
        get_package_share_directory('isaac_ros_sipl_camera'),
        'config', config_name)
    namespace = SiplCameraBenchTest.generate_namespace()

    sipl_stereo_node = ComposableNode(
        name='sipl_stereo_camera',
        package='isaac_ros_sipl_camera',
        plugin='isaac_ros::sipl::SiplStereoCameraNode',
        namespace=namespace,
        parameters=load_config_for_test(
            config_path, namespace, 'sipl_stereo_camera', 'sipl_bench_container'),
    )

    # This time is the closest approximation to the start of nodes being launched.
    SiplCameraBenchTest.launch_time_ns = time.time_ns()

    return SiplCameraBenchTest.generate_test_description([
        ComposableNodeContainer(
            name='sipl_bench_container',
            package='rclcpp_components',
            executable='component_container_mt',
            composable_node_descriptions=[sipl_stereo_node],
            namespace=namespace,
            output='screen',
            arguments=['--ros-args', '--log-level', 'info'],
        )
    ])


class SiplCameraBenchTest(IsaacROSBaseTest):
    filepath = pathlib.Path(os.path.dirname(__file__))
    # Set in generate_test_description() so failure messages can identify
    # which camera was being tested.
    config_name = ''

    def _extract_timestamps_ms(self, messages):
        """Extract header timestamps in milliseconds from a list of messages."""
        timestamps = []
        for msg in messages:
            ts_ms = msg.header.stamp.sec * SEC_TO_MS + msg.header.stamp.nanosec * NS_TO_MS
            timestamps.append(ts_ms)
        return timestamps

    def _compute_stream_timing_stats(self, stream_name, timestamps_ms):
        """
        Compute and log stream timing statistics for a single stream.

        Returns (frame_rate, exceed_tolerance_percent) for assertion.
        """
        if len(timestamps_ms) < 2:
            self.node.get_logger().error(
                f'[{stream_name}] Not enough frames to compute timing statistics '
                f'(received {len(timestamps_ms)})')
            return 0.0, 100.0

        expected_diff = SEC_TO_MS / EXPECTED_FPS
        diffs = numpy.diff(timestamps_ms)
        jitters = numpy.abs(diffs - expected_diff)

        exceed_count = int(numpy.sum(jitters > JITTER_TOLERANCE_MS))
        exceed_percent = exceed_count * 100.0 / len(diffs)

        duration = timestamps_ms[-1] - timestamps_ms[0]
        frame_rate = len(timestamps_ms) / (duration / SEC_TO_MS) if duration > 0 else 0.0

        # Report
        self.node.get_logger().info('+-{}-+'.format('-' * MAX_ROW_WIDTH))
        heading = f'SIPL Stream Stability Test: {stream_name}'
        self.node.get_logger().info(
            '| {:^{w}} |'.format(heading, w=MAX_ROW_WIDTH))
        self.node.get_logger().info('+-{}-+'.format('-' * MAX_ROW_WIDTH))
        self.node.get_logger().info(
            f'  Frames received          : {len(timestamps_ms)}')
        self.node.get_logger().info(
            f'  Duration (ms)            : {float(duration):.1f}')
        self.node.get_logger().info(
            f'  Mean frame rate (fps)    : {frame_rate:.2f}')
        self.node.get_logger().info(
            f'  Max jitter (ms)          : {float(numpy.max(jitters)):.3f}')
        self.node.get_logger().info(
            f'  Min jitter (ms)          : {float(numpy.min(jitters)):.3f}')
        self.node.get_logger().info(
            f'  Mean jitter (ms)         : {float(numpy.mean(jitters)):.3f}')
        self.node.get_logger().info(
            f'  Jitter exceed ({JITTER_TOLERANCE_MS} ms)    : '
            f'{exceed_percent:.1f}%')
        self.node.get_logger().info('+-{}-+'.format('-' * MAX_ROW_WIDTH))

        return frame_rate, exceed_percent

    # Temporarily mitigate flaky SIPL CoE camera initialization issues.
    @flaky(max_runs=3, min_passes=1)
    def test_stereo_stream_stability(self):
        """
        Run a 30-second capture and validate stream timing stability.

        Left and right streams are measured independently. Both streams must
        satisfy the frame-rate and jitter stability thresholds to pass.
        The first frame latency is also asserted against STARTUP_TIME_MAX_DELAY.
        """
        topics = [
            'left/image_raw', 'right/image_raw',
            'left/camera_info', 'right/camera_info',
        ]
        self.generate_namespace_lookup(topics)

        received_messages = {}
        self.create_logging_subscribers(
            [('left/image_raw', Image),
             ('right/image_raw', Image),
             ('left/camera_info', CameraInfo),
             ('right/camera_info', CameraInfo)],
            received_messages,
            accept_multiple_messages=True)

        end_time = time.time() + TIMEOUT
        while time.time() < end_time:
            rclpy.spin_once(self.node, timeout_sec=0.01)

        launch_ms = self.launch_time_ns * NS_TO_MS

        self.assertTrue(len(received_messages) > 0, 'No messages received from stream topics')

        # Verify both streams produced frames
        left_imgs = received_messages.get('left/image_raw', [])
        right_imgs = received_messages.get('right/image_raw', [])
        left_infos = received_messages.get('left/camera_info', [])
        right_infos = received_messages.get('right/camera_info', [])

        for name, msgs in [
            ('Left image', left_imgs),
            ('Right image', right_imgs),
            ('Left camera_info', left_infos),
            ('Right camera_info', right_infos)
        ]:
            self.assertGreater(
                len(msgs), STABILITY_MIN_MSGS,
                f'{name} produced too few messages ({len(msgs)} < '
                f'{STABILITY_MIN_MSGS}) for TIMEOUT={TIMEOUT}s and '
                f'min_fps={MIN_ACCEPTABLE_FPS:.2f}')

            # Compute per-stream timing statistics
            ts = self._extract_timestamps_ms(msgs)

            first_frame_delay_ms = ts[0] - launch_ms
            self.assertLess(
                first_frame_delay_ms, STARTUP_TIME_MAX_DELAY * SEC_TO_MS,
                f'{name} time to first frame ({first_frame_delay_ms:.0f} ms) '
                f'exceeds {STARTUP_TIME_MAX_DELAY}s limit')

            fps, exceed = self._compute_stream_timing_stats(name.upper(), ts)

            # Assert FPS
            self.assertGreater(
                fps, MIN_ACCEPTABLE_FPS,
                f'{name} frame rate {fps:.2f} below threshold')
            self.assertLess(
                fps, MAX_ACCEPTABLE_FPS,
                f'{name} frame rate {fps:.2f} above threshold')

            # Assert jitter
            self.assertLess(
                exceed, EXCEED_TOLERANCE_PERCENT_LIMIT,
                f'{name}: {exceed:.1f}% of frames exceed jitter tolerance')
