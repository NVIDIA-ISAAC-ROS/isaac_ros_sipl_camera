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

import isaac_ros_launch_utils as lu
from isaac_ros_sipl_camera.launch_utils import load_image_size
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import LoadComposableNodes, Node
from launch_ros.descriptions import ComposableNode


def launch_setup(context, *args, **kwargs):
    """Set up composable nodes based on launch arguments."""
    encoding_desired = LaunchConfiguration('encoding_desired')
    target_container = LaunchConfiguration('target_container')
    param_file = LaunchConfiguration('param_file')
    enable_rectify = LaunchConfiguration('enable_rectify')
    log_level = LaunchConfiguration('log_level')
    camera_namespace = LaunchConfiguration('camera_namespace').perform(context) or None

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

    image_width, image_height = load_image_size(param_file.perform(context))

    # SIPL monocular camera node
    sipl_params = [param_file]
    if encoding_desired_val in ('nv12', 'nv24'):
        sipl_params.append({'encoding_desired': encoding_desired_val})

    sipl_camera_node = ComposableNode(
        name='sipl_camera',
        namespace=camera_namespace,
        package='isaac_ros_sipl_camera',
        plugin='isaac_ros::sipl::SiplCameraNode',
        parameters=sipl_params,
    )

    downstream_nodes = []

    if needs_conversion:
        format_converter = ComposableNode(
            name='format_converter',
            namespace=camera_namespace,
            package='isaac_ros_image_proc',
            plugin='nvidia::isaac_ros::image_proc::ImageFormatConverterNode',
            parameters=[{
                'encoding_desired': encoding_desired_val,
                'image_width': image_width,
                'image_height': image_height,
                # SIPL ISP buffers are allocated with NvSciColorStd_REC709_ER
                # (see sipl_buffer_manager.cpp), so use the BT.709 matrix.
                # CV-CUDA AdvCvtColor does not support full-range (_ER) variants
                # today, so Y contrast will be slightly off vs. a true full-range
                # decode; the chroma matrix is what matters most and is correct here.
                'yuv_color_spec': 'bt709',
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
            namespace=camera_namespace,
            package='isaac_ros_image_proc',
            plugin='nvidia::isaac_ros::image_proc::RectifyNode',
            parameters=[{
                'output_width': image_width,
                'output_height': image_height,
            }],
            remappings=rectify_remappings,
        )
        downstream_nodes.append(rectify_node)

    if LaunchConfiguration('enable_encoding').perform(context).lower() == 'true':
        encoder_node = ComposableNode(
            name='encoder_node',
            namespace=camera_namespace,
            package='isaac_ros_h264_encoder',
            plugin='nvidia::isaac_ros::h264_encoder::EncoderNode',
            parameters=[{
                'input_width': image_width,
                'input_height': image_height,
            }],
            remappings=[
                ('image_raw', 'image_raw'),
                ('image_compressed', 'image_compressed'),
            ],
        )
        downstream_nodes.append(encoder_node)

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

    load_nodes = LoadComposableNodes(
        target_container=target_container,
        composable_node_descriptions=downstream_nodes + [sipl_camera_node],
    )
    launch_actions.extend([container, load_nodes])

    return launch_actions


def generate_launch_description():
    """Generate launch description for SIPL monocular camera."""
    default_params = str(lu.get_path('isaac_ros_sipl_camera', 'config/eagle_mono.yaml'))

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
            'camera_namespace',
            default_value='',
            description='ROS namespace for the camera node and its topics '
            '(default: empty, root namespace)'
        ),
        DeclareLaunchArgument(
            'enable_rectify',
            default_value='false',
            description='Enable image rectification via RectifyNode '
            '(true/false, default: false)'
        ),
        DeclareLaunchArgument(
            'enable_encoding',
            default_value='false',
            description='H.264 accelerated encoding of the image to image_compressed via '
            'isaac_ros_h264_encoder '
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
