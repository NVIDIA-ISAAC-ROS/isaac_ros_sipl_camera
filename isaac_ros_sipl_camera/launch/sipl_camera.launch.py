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

import os

from ament_index_python.packages import get_package_share_directory
import isaac_ros_launch_utils as lu
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction, TimerAction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import LoadComposableNodes, Node
from launch_ros.descriptions import ComposableNode


# Delay before loading the SIPL camera node when downstream NITROS nodes are
# present. Downstream NITROS nodes initialize their GXF graph asynchronously;
# if the camera publishes before that graph is ready, startup frames can be
# dropped while SIPL ISP buffers are still held. A short delay makes startup
# successful when downstream GXF nodes are to be loaded.
# Testing observed 0.5s as the minimum reliable value; 2s provides margin.
# todo: This might be able to be removed in the near future after GXF-less NITROS.
_DOWNSTREAM_INIT_DELAY_S = 2.0


def launch_setup(context, *args, **kwargs):
    """Set up composable nodes based on launch arguments."""
    encoding_desired = LaunchConfiguration('encoding_desired')
    target_container = LaunchConfiguration('target_container')
    param_file = LaunchConfiguration('param_file')
    enable_rectify = LaunchConfiguration('enable_rectify')
    log_level = LaunchConfiguration('log_level')

    # Determine if format conversion and rectification are needed
    encoding_desired_val = encoding_desired.perform(context)
    rectify_enabled = enable_rectify.perform(context).lower() == 'true'

    # RectifyNode requires a standard image format (e.g. rgb8, bgr8).
    # Force conversion to rgb8 when rectify is enabled and no format is set.
    launch_actions = []
    if rectify_enabled and encoding_desired_val in ('nv12', 'nv24', ''):
        launch_actions.append(lu.log_info(
            'Rectify enabled: forcing encoding_desired to rgb8 '
            'for RectifyNode compatibility.'
        ))
        encoding_desired_val = 'rgb8'

    needs_conversion = encoding_desired_val not in ('nv12', 'nv24', '')

    # Assume VB1940 Eagle camera resolution as this is the only supported
    # resolution and camera by SIPL as of implementation.
    image_width = 2560
    image_height = 1984

    # SIPL monocular camera node
    sipl_camera_node = ComposableNode(
        name='sipl_camera',
        package='isaac_ros_sipl_camera',
        plugin='isaac_ros::sipl::SiplCameraNode',
        parameters=[param_file],
    )

    downstream_nodes = []

    if needs_conversion:
        format_converter = ComposableNode(
            name='format_converter',
            package='isaac_ros_image_proc',
            plugin='nvidia::isaac_ros::image_proc::ImageFormatConverterNode',
            parameters=[{
                'encoding_desired': encoding_desired_val,
                'image_width': image_width,
                'image_height': image_height,
                'num_blocks': 40,
            }],
            remappings=[
                ('image', 'image_converted'),
            ],
        )
        downstream_nodes.append(format_converter)

    if rectify_enabled:
        rectify_remappings = []
        if needs_conversion:
            rectify_remappings = [('image_raw', 'image_converted')]

        rectify_node = ComposableNode(
            name='rectify',
            package='isaac_ros_image_proc',
            plugin='nvidia::isaac_ros::image_proc::RectifyNode',
            parameters=[{
                'output_width': image_width,
                'output_height': image_height,
            }],
            remappings=rectify_remappings,
        )
        downstream_nodes.append(rectify_node)

    # Container
    container_args = []
    log_level_val = log_level.perform(context)
    if log_level_val:
        container_args = ['--ros-args', '--log-level', f'{log_level_val}']

    container = Node(
        name=target_container,
        package='rclcpp_components',
        executable='component_container_mt',
        output='screen',
        arguments=container_args,
    )

    if downstream_nodes:
        # Two-phase loading: bring up downstream nodes first, then load camera
        # after a short delay so their GXF graphs are ready before first frames.
        load_downstream = LoadComposableNodes(
            target_container=target_container,
            composable_node_descriptions=downstream_nodes,
        )
        load_camera = LoadComposableNodes(
            target_container=target_container,
            composable_node_descriptions=[sipl_camera_node],
        )
        deferred_camera = TimerAction(
            period=_DOWNSTREAM_INIT_DELAY_S,
            actions=[load_camera],
        )
        launch_actions.extend([container, load_downstream, deferred_camera])
    else:
        load_nodes = LoadComposableNodes(
            target_container=target_container,
            composable_node_descriptions=[sipl_camera_node],
        )
        launch_actions.extend([container, load_nodes])

    return launch_actions


def generate_launch_description():
    """Generate launch description for SIPL monocular camera."""
    pkg_share = get_package_share_directory('isaac_ros_sipl_camera')
    default_params = os.path.join(pkg_share, 'config', 'eagle_mono.yaml')

    return LaunchDescription([
        DeclareLaunchArgument(
            'param_file',
            default_value=default_params,
            description='Path to parameter file (default: eagle_mono.yaml)'
        ),
        DeclareLaunchArgument(
            'encoding_desired',
            default_value='',
            description='Output image format, for example: nv12/nv24 (native), '
            ' rgb8, mono8, etc. (converted via ImageFormatConverterNode)'
        ),
        DeclareLaunchArgument(
            'target_container',
            default_value='sipl_camera_container',
            description='Target container name for composable nodes'
        ),
        DeclareLaunchArgument(
            'enable_rectify',
            default_value='false',
            description='Enable image rectification via RectifyNode '
            '(true/false, default: false)'
        ),
        DeclareLaunchArgument(
            'log_level',
            default_value='',
            description='ROS log level for the SIPL camera node '
            '(e.g. debug, info, warn). Empty will use ROS default (info).'
        ),

        OpaqueFunction(function=launch_setup)
    ])
