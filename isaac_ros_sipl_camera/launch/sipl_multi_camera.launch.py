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

from pathlib import Path

import isaac_ros_launch_utils as lu
from isaac_ros_sipl_camera.launch_utils import (
    encoder_input,
    force_rgb8_for_image_proc,
    load_camera_field,
    load_camera_names,
    load_camera_sizes,
    load_native_encoding,
    load_parameters,
    query_config_arg_description,
    query_config_parameters,
    validate_resize_dimensions,
)
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.logging import get_logger
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import LoadComposableNodes, Node
from launch_ros.descriptions import ComposableNode


def _yuv_color_spec(param_file_path, camera_name):
    """Read the explicit color matrix required for each converted stream."""
    explicit = load_camera_field(param_file_path, camera_name, 'yuv_color_spec')
    if explicit:
        return explicit
    raise RuntimeError(
        f'camera.{camera_name}.yuv_color_spec must be set to use a converted '
        'encoding: the ISP0 path produces REC709 and the ISP-bypass (YUV422 '
        'capture) path produces REC601, and which one applies is only known to '
        'the node. Use bt709 for RAW sensors and bt601 for YUV422 sensors.')


def _validate_nv12_inputs(param_file_path, camera_names, launch_encoding):
    """Reject NV24 inputs to image conversion or H.264 encoding."""
    base_encoding = load_native_encoding(param_file_path, launch_encoding)
    nv24_cameras = [
        name for name in camera_names
        if (load_camera_field(param_file_path, name, 'encoding', base_encoding) or
            base_encoding) == 'nv24'
    ]
    if nv24_cameras:
        raise RuntimeError(
            'Image conversion and H.264 encoding do not support native nv24 '
            'output for camera(s): '
            f'{", ".join(nv24_cameras)}; use nv12.')


def _clock_warning():
    """Check whether CPU, GPU and EMC clocks are locked for multi-camera capture."""
    try:
        cpu_limits = list(Path('/sys/devices/system/cpu').glob('cpu[0-9]*/cpufreq'))
        limits = [(path / 'scaling_min_freq', path / 'scaling_max_freq')
                  for path in cpu_limits]
        limits.extend((Path('/sys/class/devfreq') / name / 'min_freq',
                       Path('/sys/class/devfreq') / name / 'max_freq')
                      for name in ('gpu-gpc-0', 'gpu-nvd-0', 'bwmgr'))
        if cpu_limits and all(int(minimum.read_text()) == int(maximum.read_text())
                              for minimum, maximum in limits):
            return None
    except (OSError, ValueError):
        pass
    return (
        'Could not confirm locked Jetson clocks. Multi-camera capture may timeout '
        'when under high load. Run "sudo nvpmodel -m 0; sudo jetson_clocks" '
        'before launch to lock maximum performance; ')


def launch_setup(context, *args, **kwargs):
    """Set up composable nodes based on launch arguments."""
    encoding_desired = LaunchConfiguration('encoding_desired')
    target_container = LaunchConfiguration('target_container')
    param_file = LaunchConfiguration('param_file')
    enable_rectify = LaunchConfiguration('enable_rectify')
    log_level = LaunchConfiguration('log_level')
    camera_namespace = LaunchConfiguration('camera_namespace').perform(context) or None

    param_file_path = param_file.perform(context)
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

    camera_names = load_camera_names(param_file_path)
    if not camera_names:
        raise RuntimeError(
            f'{param_file_path} must define a non-empty cameras list; '
            'use sipl_camera.launch.py or sipl_stereo_camera.launch.py instead')
    if needs_conversion or encoding_enabled:
        _validate_nv12_inputs(param_file_path, camera_names, encoding_desired_val)
    camera_sizes = (
        load_camera_sizes(param_file_path)
        if needs_conversion or rectify_enabled or encoding_enabled else {})

    sipl_params = [load_parameters(param_file_path)]
    query_config = LaunchConfiguration('query_config').perform(context)
    if query_config:
        sipl_params.append(query_config_parameters(param_file_path, query_config))
    if encoding_desired_val in ('nv12', 'nv24'):
        sipl_params.append({'encoding_desired': encoding_desired_val})

    if len(camera_names) > 1:
        warning = _clock_warning()
        if warning:
            get_logger('launch.user').warning(warning)

    sipl_multi_camera_node = ComposableNode(
        name='sipl_multi_camera',
        namespace=camera_namespace,
        package='isaac_ros_sipl_camera',
        plugin='isaac_ros::sipl::SiplMultiCameraNode',
        parameters=sipl_params,
    )

    namespaces = {
        name: (load_camera_field(param_file_path, name, 'topic_namespace') or '').lstrip('/')
        or name
        for name in camera_names
    }

    downstream_nodes = []
    for name in camera_names:
        if camera_sizes:
            image_width, image_height = camera_sizes[name]
        ns = namespaces[name]

        if needs_conversion:
            downstream_nodes.append(ComposableNode(
                name=f'{name}_format_converter',
                namespace=camera_namespace,
                package='isaac_ros_image_proc',
                plugin='nvidia::isaac_ros::image_proc::ImageFormatConverterNode',
                parameters=[{
                    'encoding_desired': converter_encoding,
                    'image_width': image_width,
                    'image_height': image_height,
                    'yuv_color_spec': _yuv_color_spec(param_file_path, name),
                }],
                remappings=[
                    ('image_raw', f'{ns}/image_raw'),
                    ('camera_info', f'{ns}/camera_info'),
                    ('image', f'{ns}/image_converted'),
                ],
            ))

        if rectify_enabled:
            rectify_remappings = [
                ('image_raw', f'{ns}/image_converted' if needs_conversion
                 else f'{ns}/image_raw'),
                ('camera_info', f'{ns}/camera_info'),
                ('image_rect', f'{ns}/image_rect'),
                ('camera_info_rect', f'{ns}/camera_info_rect'),
            ]
            downstream_nodes.append(ComposableNode(
                name=f'{name}_rectify',
                namespace=camera_namespace,
                package='isaac_ros_image_proc',
                plugin='nvidia::isaac_ros::image_proc::RectifyNode',
                parameters=[{
                    'output_width': image_width,
                    'output_height': image_height,
                }],
                remappings=rectify_remappings,
            ))

        if resize_enabled:
            downstream_nodes.append(ComposableNode(
                name=f'{name}_resize',
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
                    ('image', f'{ns}/image_converted' if needs_conversion
                     else f'{ns}/image_raw'),
                    ('camera_info', f'{ns}/camera_info'),
                    ('resize/image', f'{ns}/image_resize'),
                    ('resize/camera_info', f'{ns}/camera_info_resize'),
                ],
            ))

        if encoding_enabled:
            encoder_image_topic, encoder_size = encoder_input(
                f'{ns}/image_raw', f'{ns}/image_resize',
                (image_width, image_height), resize_enabled,
                resize_width, resize_height)
            downstream_nodes.append(ComposableNode(
                name=f'{name}_encoder_node',
                namespace=camera_namespace,
                package='isaac_ros_h264_encoder',
                plugin='nvidia::isaac_ros::h264_encoder::EncoderNode',
                parameters=[{
                    'input_width': encoder_size[0],
                    'input_height': encoder_size[1],
                }],
                remappings=[
                    ('image_raw', encoder_image_topic),
                    ('camera_info', (f'{ns}/camera_info_resize' if resize_enabled
                     else f'{ns}/camera_info')),
                    ('image_compressed', f'{ns}/image_compressed'),
                ],
            ))

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
        composable_node_descriptions=downstream_nodes + [sipl_multi_camera_node],
    )
    launch_actions.extend([container, load_nodes])

    return launch_actions


def generate_launch_description():
    """Generate launch description for a multi-camera SIPL rig."""
    default_params = str(
        lu.get_path('isaac_ros_sipl_camera', 'config/s56c_shf3l_mixed.yaml'))

    return LaunchDescription([
        DeclareLaunchArgument(
            'param_file',
            default_value=default_params,
            description='Path to parameter file (default: s56c_shf3l_mixed.yaml)'
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
            ' rgb8, mono8, etc. (converted via ImageFormatConverterNode). '
            'Overrides the node default only; camera.<name>.encoding still wins.'
        ),
        DeclareLaunchArgument(
            'target_container',
            default_value='sipl_multi_camera_container',
            description='Name of the component container to load nodes into'
        ),
        DeclareLaunchArgument(
            'camera_namespace',
            default_value='',
            description='Namespace applied to every node and topic in this launch file '
            '(default: empty, root namespace)'
        ),
        DeclareLaunchArgument(
            'enable_rectify',
            default_value='false',
            choices=['true', 'false'],
            description='Add a RectifyNode per camera'
        ),
        DeclareLaunchArgument(
            'enable_resize',
            default_value='false',
            choices=['true', 'false'],
            description='Add a ResizeNode per camera'
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
            description='Add an H.264 EncoderNode per camera; encode image_resize '
            'when enable_resize is true'
        ),
        DeclareLaunchArgument(
            'log_level',
            default_value='',
            description='Container log level, e.g. debug'
        ),
        OpaqueFunction(function=launch_setup),
    ])
