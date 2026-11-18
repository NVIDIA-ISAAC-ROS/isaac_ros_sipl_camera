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
Mixed S56C + 2x SHF3L capture test for the SIPL camera driver.

Covers what the parametrized mono/stereo tests cannot reach:
  - four streams out of one node and one process
  - the ISP-bypass path (YUV422 capture output converted to nv12 on the GPU)
  - per-pipeline resolution, with 2560x1984 and 1920x1536 side by side

Requires ISAAC_ROS_SIPL_CAMERAS to name both s56c and shf3l, plus an
SGX_YUV_GMSL2 driver with per-link serializer addressing; skips otherwise.
"""

import collections
import functools
import json
import os
import pathlib
import time
import unittest

from ament_index_python.packages import get_package_share_directory
from flaky import flaky
from isaac_ros_test import IsaacROSBaseTest
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode
import pytest
import rclpy
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image

from sipl_camera_test_utils import (
    assert_consistent_frame_period, cadence_required_frames,
    config_skip_reason, load_config_for_test,
)
import yaml

CONFIG_NAME = 's56c_shf3l_mixed.yaml'
TIMEOUT = 20

# The node publishes its native SIPL output directly.
EXPECTED_SUBSCRIBER_ENCODING = 'nv12'


def _load_camera_expectations(config_path, node_namespace):
    """Derive topics, native resolutions, and aligned stereo pairs from the config."""
    with open(config_path) as f:
        data = yaml.safe_load(f)
    params = data['sipl_multi_camera']['ros__parameters']
    package_prefix = 'package://isaac_ros_sipl_camera/'
    package_share = get_package_share_directory('isaac_ros_sipl_camera')

    platform_config_url = params['platform_config_file']
    if not platform_config_url.startswith(package_prefix):
        raise ValueError(
            f'unsupported platform_config_file {platform_config_url!r}')
    platform_config_path = os.path.join(
        package_share, platform_config_url.removeprefix(package_prefix))
    with open(platform_config_path) as f:
        platform_config = json.load(f)
    fps_by_sensor_id = {
        int(sensor['id']): float(sensor['virtualChannels'][0]['fps'])
        for camera_config in platform_config['cameraConfigs']
        for sensor in camera_config.get('sensorInfo', [])
    }

    expectations = {}
    for camera_name in params['cameras']:
        camera = params['camera'][camera_name]
        topic_namespace = '/' + '/'.join(
            part for part in (node_namespace.strip('/'), camera['topic_namespace'].strip('/'))
            if part)

        camera_info_url = camera['camera_info_url']
        if not camera_info_url.startswith(package_prefix):
            raise ValueError(
                f'{camera_name}: unsupported camera_info_url {camera_info_url!r}')
        camera_info_path = os.path.join(
            package_share, camera_info_url.removeprefix(package_prefix))
        with open(camera_info_path) as f:
            camera_info = yaml.safe_load(f)

        expectations[camera_name] = {
            'topic_namespace': topic_namespace,
            'resolution': (
                int(camera_info['image_width']), int(camera_info['image_height'])),
            'fps': fps_by_sensor_id[int(camera['sensor_id'])],
        }

    stereo_pairs = []
    for pair_spec in params.get('stereo_pairs', []):
        if not isinstance(pair_spec, str):
            raise ValueError(f'stereo pair must be a string, got {pair_spec!r}')
        pair = tuple(name.strip() for name in pair_spec.split(','))
        if len(pair) != 2 or not all(pair):
            raise ValueError(f'invalid stereo pair {pair_spec!r}')
        unknown = [name for name in pair if name not in expectations]
        if unknown:
            raise ValueError(f'stereo pair {pair_spec!r} names unknown cameras: {unknown}')
        stereo_pairs.append(pair)

    align_stereo_timestamps = params.get('align_stereo_timestamps', False)
    if not isinstance(align_stereo_timestamps, bool):
        raise ValueError('align_stereo_timestamps must be true or false')
    return expectations, stereo_pairs if align_stereo_timestamps else []


@pytest.mark.rostest
def generate_test_description():
    skip_reason = config_skip_reason(CONFIG_NAME)
    if skip_reason:
        raise unittest.SkipTest(skip_reason)

    config_path = os.path.join(
        get_package_share_directory('isaac_ros_sipl_camera'), 'config', CONFIG_NAME)
    namespace = SiplCameraMixedTest.generate_namespace()
    (SiplCameraMixedTest.camera_expectations,
     SiplCameraMixedTest.aligned_stereo_pairs) = _load_camera_expectations(
         config_path, namespace)
    config_parameters = load_config_for_test(
        config_path, namespace, 'sipl_multi_camera', 'sipl_mixed_container')

    sipl_multi_node = ComposableNode(
        name='sipl_multi_camera',
        package='isaac_ros_sipl_camera',
        plugin='isaac_ros::sipl::SiplMultiCameraNode',
        namespace=namespace,
        parameters=config_parameters + [{'use_hw_timestamp': True}],
    )

    return SiplCameraMixedTest.generate_test_description([
        ComposableNodeContainer(
            name='sipl_mixed_container',
            package='rclcpp_components',
            executable='component_container_mt',
            composable_node_descriptions=[sipl_multi_node],
            namespace=namespace,
            output='screen',
            arguments=['--ros-args', '--log-level', 'info'],
        )
    ])


class SiplCameraMixedTest(IsaacROSBaseTest):
    filepath = pathlib.Path(os.path.dirname(__file__))
    camera_expectations = {}
    aligned_stereo_pairs = []

    @flaky(max_runs=3, min_passes=1)
    def test_all_streams_publish_at_native_resolution(self):
        """Verify one synchronized Image and CameraInfo pair from every camera."""
        self.assertEqual(len(self.camera_expectations), 4)
        received = {
            name: collections.deque(maxlen=1)
            for name in self.camera_expectations
        }
        subscriptions = []
        for name, expectation in self.camera_expectations.items():
            topic_namespace = expectation['topic_namespace']
            subscriptions.extend(self.create_exact_time_sync_logging_subscribers(
                [(f'{topic_namespace}/image_raw', Image),
                 (f'{topic_namespace}/camera_info', CameraInfo)],
                received[name],
                accept_multiple_messages=True))

        end_time = time.time() + TIMEOUT
        while time.time() < end_time:
            rclpy.spin_once(self.node, timeout_sec=0.1)
            if all(received.values()):
                break

        for name, expectation in self.camera_expectations.items():
            self.assertTrue(
                received[name],
                f'[{name}] did not publish a synchronized image + camera_info pair')
            img, info = received[name][0]
            width, height = expectation['resolution']
            self.assertEqual(
                (img.width, img.height), (width, height),
                f'[{name}] expected {width}x{height}, got {img.width}x{img.height}')
            self.assertEqual(
                img.encoding, EXPECTED_SUBSCRIBER_ENCODING,
                f'[{name}] expected {EXPECTED_SUBSCRIBER_ENCODING}, got {img.encoding}')

            # A structurally valid NV12 frame proves the UYVY -> nv12 conversion
            # produced a well-formed image for the ISP-bypass cameras.
            self.assertGreaterEqual(
                img.step, img.width,
                f'[{name}] step ({img.step}) must be >= width ({img.width})')
            nv12_min_size = img.step * img.height * 3 // 2
            self.assertGreaterEqual(
                len(img.data), nv12_min_size,
                f'[{name}] data size {len(img.data)} < NV12 minimum {nv12_min_size}')
            self.assertEqual(
                img.header.stamp, info.header.stamp,
                f'[{name}] image and camera_info timestamps do not match')
            self.assertEqual(
                (info.width, info.height), expectation['resolution'],
                f'[{name}] camera_info resolution does not match the stream')

    @flaky(max_runs=3, min_passes=1)
    def test_all_camera_info_streams_publish_at_expected_cadence(self):
        """Verify sustained cadence using lightweight CameraInfo messages."""
        self.assertEqual(len(self.camera_expectations), 4)
        counts = {name: 0 for name in self.camera_expectations}
        stamps_ns = {name: [] for name in self.camera_expectations}
        latest_info = {}

        def receive_camera_info(camera_name, info):
            counts[camera_name] += 1
            latest_info[camera_name] = info
            stamp_ns = info.header.stamp.sec * 1_000_000_000 + info.header.stamp.nanosec
            stamps_ns[camera_name].append(stamp_ns)

        subscriptions = []
        for name, expectation in self.camera_expectations.items():
            subscriptions.append(self.node.create_subscription(
                CameraInfo,
                f'{expectation["topic_namespace"]}/camera_info',
                functools.partial(receive_camera_info, name),
                qos_profile_sensor_data))

        end_time = time.time() + TIMEOUT
        while time.time() < end_time:
            rclpy.spin_once(self.node, timeout_sec=0.1)
            if all(
                counts[name] >= cadence_required_frames(expectation['fps'])
                for name, expectation in self.camera_expectations.items()
            ):
                break

        for name, expectation in self.camera_expectations.items():
            expected_fps = expectation['fps']
            assert_consistent_frame_period(
                self, name, stamps_ns[name], expected_fps)
            self.assertEqual(
                (latest_info[name].width, latest_info[name].height), expectation['resolution'],
                f'[{name}] camera_info resolution does not match the stream')

    @flaky(max_runs=3, min_passes=1)
    def test_stereo_pairs_remain_timestamp_aligned(self):
        """Verify exact stereo CameraInfo pairs maintain the expected cadence."""
        if not self.aligned_stereo_pairs:
            self.skipTest('stereo timestamp alignment is disabled by configuration')

        for left_name, right_name in self.aligned_stereo_pairs:
            names = (left_name, right_name)
            received_pairs = []
            self.create_exact_time_sync_logging_subscribers(
                [(f'{self.camera_expectations[name]["topic_namespace"]}/camera_info', CameraInfo)
                 for name in names],
                received_pairs,
                accept_multiple_messages=True)

            fps = min(self.camera_expectations[name]['fps'] for name in names)
            required_frames = cadence_required_frames(fps)
            end_time = time.time() + TIMEOUT
            while time.time() < end_time:
                rclpy.spin_once(self.node, timeout_sec=0.1)
                if len(received_pairs) >= required_frames:
                    break

            stamps_ns = [
                left.header.stamp.sec * 1_000_000_000 + left.header.stamp.nanosec
                for left, _ in received_pairs
            ]
            assert_consistent_frame_period(
                self, f'{left_name},{right_name}', stamps_ns, fps)
