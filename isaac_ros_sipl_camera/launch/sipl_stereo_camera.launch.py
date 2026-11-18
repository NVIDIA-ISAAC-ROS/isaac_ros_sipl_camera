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
from isaac_ros_sipl_camera.launch_utils import (
    encoder_input,
    force_rgb8_for_image_proc,
    load_image_size,
    load_native_encoding,
    load_parameters,
    query_config_arg_description,
    query_config_parameters,
    validate_resize_dimensions,
)
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
    encoding_enabled = LaunchConfiguration('enable_encoding').perform(context).lower() == 'true'
    resize_enabled = LaunchConfiguration('enable_resize').perform(context).lower() == 'true'
    resize_width = int(LaunchConfiguration('resize_width').perform(context))
    resize_height = int(LaunchConfiguration('resize_height').perform(context))
    if resize_enabled:
        validate_resize_dimensions(resize_width, resize_height)

    launch_actions = []
    converter_encoding, conversion_logs = force_rgb8_for_image_proc(
        encoding_desired_val,
        rectify_enabled=rectify_enabled,
        resize_enabled=resize_enabled)
    for message in conversion_logs:
        launch_actions.append(lu.log_info(message))

    needs_conversion = converter_encoding not in ('nv12', 'nv24', '')

    param_file_path = param_file.perform(context)
    native_encoding = load_native_encoding(param_file_path, encoding_desired_val)
    if (needs_conversion or encoding_enabled) and native_encoding == 'nv24':
        raise RuntimeError(
            'Image conversion and H.264 encoding do not support native nv24 output; use nv12.')

    if needs_conversion or rectify_enabled or encoding_enabled:
        image_width, image_height = load_image_size(param_file_path)

    # Stereo frame-pair gating is part of the timestamp-alignment feature: the
    # node aligns left/right capture stamps to a single value, then a
    # CameraDropNode ExactTime-syncs the pair and republishes only
    # complete pairs (drop both if one eye is missing).
    # The enforce_stereo_pairs launch arg is the single switch: it is fed to the
    # node's align_stereo_timestamps parameter and toggles whether the gating
    # drop node is inserted.
    enforce_stereo_pairs = \
        LaunchConfiguration('enforce_stereo_pairs').perform(context).lower() == 'true'

    # When gating is on, the camera publishes its raw (ungated) output under
    # *_unsynced names so the drop node can republish the gated pair under the
    # canonical names that downstream nodes consume.
    sipl_remappings = []
    if enforce_stereo_pairs:
        sipl_remappings = [
            ('left/image_raw', 'left/image_raw_unsynced'),
            ('left/camera_info', 'left/camera_info_unsynced'),
            ('right/image_raw', 'right/image_raw_unsynced'),
            ('right/camera_info', 'right/camera_info_unsynced'),
        ]

    # SIPL stereo camera node
    sipl_params = [load_parameters(param_file_path)]
    query_config = LaunchConfiguration('query_config').perform(context)
    if query_config:
        sipl_params.append(query_config_parameters(param_file_path, query_config))
    if encoding_desired_val in ('nv12', 'nv24'):
        sipl_params.append({'encoding_desired': encoding_desired_val})
    sipl_params.append({'align_stereo_timestamps': enforce_stereo_pairs})

    sipl_stereo_node = ComposableNode(
        name='sipl_stereo_camera',
        namespace=camera_namespace,
        package='isaac_ros_sipl_camera',
        plugin='isaac_ros::sipl::SiplStereoCameraNode',
        parameters=sipl_params,
        remappings=sipl_remappings,
    )

    downstream_nodes = []

    if needs_conversion:
        # SIPL ISP buffers are allocated with NvSciColorStd_REC709_ER
        # (see sipl_buffer_manager.cpp), so use the BT.709 matrix.
        # CV-CUDA AdvCvtColor does not support full-range (_ER) variants
        # today, so Y contrast will be slightly off vs. a true full-range
        # decode; the chroma matrix is what matters most and is correct here.
        converter_params = {
            'encoding_desired': converter_encoding,
            'image_width': image_width,
            'image_height': image_height,
            'yuv_color_spec': 'bt709',
        }

        left_converter = ComposableNode(
            name='left_format_converter',
            namespace=camera_namespace,
            package='isaac_ros_image_proc',
            plugin='nvidia::isaac_ros::image_proc::ImageFormatConverterNode',
            parameters=[converter_params],
            remappings=[
                ('image_raw', 'left/image_raw'),
                ('image', 'left/image_converted'),
            ],
        )

        right_converter = ComposableNode(
            name='right_format_converter',
            namespace=camera_namespace,
            package='isaac_ros_image_proc',
            plugin='nvidia::isaac_ros::image_proc::ImageFormatConverterNode',
            parameters=[converter_params],
            remappings=[
                ('image_raw', 'right/image_raw'),
                ('image', 'right/image_converted'),
            ],
        )

        downstream_nodes.extend([left_converter, right_converter])

    left_image_topic = 'left/image_converted' if needs_conversion else 'left/image_raw'
    right_image_topic = 'right/image_converted' if needs_conversion else 'right/image_raw'

    if rectify_enabled:
        left_rectify = ComposableNode(
            name='left_rectify',
            package='isaac_ros_image_proc',
            plugin='nvidia::isaac_ros::image_proc::RectifyNode',
            namespace=camera_namespace,
            parameters=[{
                'output_width': image_width,
                'output_height': image_height,
            }],
            remappings=[
                ('image_raw', left_image_topic),
                ('camera_info', 'left/camera_info'),
                ('image_rect', 'left/image_rect'),
                ('camera_info_rect', 'left/camera_info_rect'),
            ],
        )

        right_rectify = ComposableNode(
            name='right_rectify',
            package='isaac_ros_image_proc',
            plugin='nvidia::isaac_ros::image_proc::RectifyNode',
            namespace=camera_namespace,
            parameters=[{
                'output_width': image_width,
                'output_height': image_height,
            }],
            remappings=[
                ('image_raw', right_image_topic),
                ('camera_info', 'right/camera_info'),
                ('image_rect', 'right/image_rect'),
                ('camera_info_rect', 'right/camera_info_rect'),
            ],
        )

        downstream_nodes.extend([left_rectify, right_rectify])

    if resize_enabled:
        for side in ('left', 'right'):
            downstream_nodes.append(ComposableNode(
                name=f'{side}_resize',
                namespace=camera_namespace,
                package='isaac_ros_image_proc',
                plugin='nvidia::isaac_ros::image_proc::ResizeNode',
                parameters=[{
                    'output_width': resize_width,
                    'output_height': resize_height,
                    'input_queue_size': 4,
                    'output_queue_size': 4,
                }],
                remappings=[
                    ('image', f'{side}/image_converted' if needs_conversion
                     else f'{side}/image_raw'),
                    ('camera_info', f'{side}/camera_info'),
                    ('resize/image', f'{side}/image_resize'),
                    ('resize/camera_info', f'{side}/camera_info_resize'),
                ],
            ))

    if encoding_enabled:
        left_encoder_topic, encoder_size = encoder_input(
            'left/image_raw', 'left/image_resize', (image_width, image_height),
            resize_enabled, resize_width, resize_height)
        right_encoder_topic = (
            'right/image_resize' if resize_enabled else 'right/image_raw')
        left_encoder = ComposableNode(
            name='left_encoder',
            namespace=camera_namespace,
            package='isaac_ros_h264_encoder',
            plugin='nvidia::isaac_ros::h264_encoder::EncoderNode',
            parameters=[{
                'input_width': encoder_size[0],
                'input_height': encoder_size[1],
            }],
            remappings=[
                ('image_raw', left_encoder_topic),
                ('image_compressed', 'left/image_compressed'),
            ],
        )
        right_encoder = ComposableNode(
            name='right_encoder',
            namespace=camera_namespace,
            package='isaac_ros_h264_encoder',
            plugin='nvidia::isaac_ros::h264_encoder::EncoderNode',
            parameters=[{
                'input_width': encoder_size[0],
                'input_height': encoder_size[1],
            }],
            remappings=[
                ('image_raw', right_encoder_topic),
                ('image_compressed', 'right/image_compressed'),
            ],
        )
        downstream_nodes.extend([left_encoder, right_encoder])

    if enforce_stereo_pairs:
        # ExactTime-sync the aligned left/right image + camera_info set and
        # republish only complete pairs under the canonical topic names.
        # X=0 (drop 0 of every Y) with Y=1 disables any rate dropping, so it acts
        # purely as a gate on pair completeness.
        stereo_pair_sync = ComposableNode(
            name='stereo_pair_sync',
            namespace=camera_namespace,
            package='isaac_ros_topic_tools',
            plugin='nvidia::isaac_ros::topic_tools::CameraDropNode',
            parameters=[{
                'mode': 'stereo',
                'X': 0,
                'Y': 1,
                'sync_queue_size': 5,
            }],
            remappings=[
                ('image_1', 'left/image_raw_unsynced'),
                ('camera_info_1', 'left/camera_info_unsynced'),
                ('image_2', 'right/image_raw_unsynced'),
                ('camera_info_2', 'right/camera_info_unsynced'),
                ('image_1_drop', 'left/image_raw'),
                ('camera_info_1_drop', 'left/camera_info'),
                ('image_2_drop', 'right/image_raw'),
                ('camera_info_2_drop', 'right/camera_info'),
            ],
        )
        downstream_nodes.append(stereo_pair_sync)

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
        composable_node_descriptions=downstream_nodes + [sipl_stereo_node],
    )
    launch_actions.extend([container, load_nodes])

    return launch_actions


def generate_launch_description():
    """Generate launch description for SIPL stereo camera."""
    default_params = str(lu.get_path('isaac_ros_sipl_camera', 'config/eagle_stereo.yaml'))

    return LaunchDescription([
        DeclareLaunchArgument(
            'param_file',
            default_value=default_params,
            description='Path to parameter file (default: eagle_stereo.yaml)'
        ),
        DeclareLaunchArgument(
            'query_config',
            default_value='',
            description=query_config_arg_description()
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
            description='Namespace applied to every node in this launch file. '
            '(default: empty, root namespace)'
        ),
        DeclareLaunchArgument(
            'enable_rectify',
            default_value='false',
            choices=['true', 'false'],
            description='Enable image rectification via RectifyNode '
            '(true/false, default: false)'
        ),
        DeclareLaunchArgument(
            'enable_resize',
            default_value='false',
            choices=['true', 'false'],
            description='Add a ResizeNode per eye that publishes image_resize '
            'and camera_info_resize'
        ),
        DeclareLaunchArgument(
            'resize_width',
            default_value='640',
            description='Width of image_resize and camera_info_resize'
        ),
        DeclareLaunchArgument(
            'resize_height',
            default_value='480',
            description='Height of image_resize and camera_info_resize'
        ),
        DeclareLaunchArgument(
            'enable_encoding',
            default_value='false',
            choices=['true', 'false'],
            description='H.264 accelerated encoding of image_resize when resizing is '
            'enabled, or image_raw otherwise'
        ),
        DeclareLaunchArgument(
            'enforce_stereo_pairs',
            default_value='false',
            choices=['true', 'false'],
            description='Align left/right capture timestamps and gate stereo frame pairs so that '
            'only complete stereo pairs are republished on the canonical image_raw / '
            'camera_info topics. A frame whose partner pair is missing is dropped. '
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
