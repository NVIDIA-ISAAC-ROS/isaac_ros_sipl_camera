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

"""Shared helpers for the SIPL camera launch files (mono, stereo and multi)."""

from urllib.parse import unquote, urlparse

import isaac_ros_launch_utils as lu
import yaml


# Query JSON and mask-vector position for each supported adapter.
_QUERY_CONFIGS = {
    'unitree_h2plus': {
        'platform_config_file': (
            'package://isaac_ros_sipl_camera/config/'
            's56c_shf3l_mixed_h2plus.json'),
        'transport_count': 1,
        'mask_index': 0,
    },
    'trd1_g2a': {
        'platform_config_file': (
            'package://isaac_ros_sipl_camera/config/'
            's56c_shf3l_mixed_trd1_g2a.json'),
        'transport_count': 2,
        'mask_index': 1,
    },
    'max96724': {
        'platform_config_file': (
            'package://isaac_ros_sipl_camera/config/'
            's56c_shf3l_mixed_max96724.json'),
        'transport_count': 1,
        'mask_index': 0,
    },
}
_BUNDLED_QUERY_JSONS = {
    config['platform_config_file'] for config in _QUERY_CONFIGS.values()
}
_BUNDLED_QUERY_JSONS.add(
    'package://isaac_ros_sipl_camera/config/s56c_shf3l_mixed.json')


def query_config_arg_description():
    """Launch-argument help text listing bundled SIPL query JSON names."""
    names = sorted(_QUERY_CONFIGS)
    if len(names) == 2:
        listed = f'{names[0]} or {names[1]}'
    else:
        listed = f'{", ".join(names[:-1])}, or {names[-1]}'
    return f'Optional SIPL query JSON: {listed}'


def resolve_package_url(url):
    """
    Resolve a camera_info URL to a local filesystem path.

    Supports ``package://<pkg>/<relative_path>`` and ``file://<path>`` URLs, as
    accepted by ``camera_info_manager``. Any other value is returned unchanged so
    that bare filesystem paths continue to work.
    """
    if url.startswith('package://'):
        package_name, relative_path = url.removeprefix('package://').split('/', 1)
        return str(lu.get_path(package_name, relative_path))
    if url.startswith('file://'):
        return unquote(urlparse(url).path)
    return url


def load_parameters(param_file_path):
    """Load node parameters, using defaults for optional empty arrays."""
    optional_empty_array_parameters = {'stereo_pairs'}
    try:
        with open(param_file_path) as f:
            params = next(iter(yaml.safe_load(f).values()))['ros__parameters']
        # ROS launch cannot infer an empty array's type.
        return {name: value for name, value in params.items()
                if name not in optional_empty_array_parameters or value != []}
    except Exception as exc:
        raise RuntimeError(
            f'{param_file_path} must be a ROS parameter file with ros__parameters') from exc


def query_config_parameters(param_file_path, query_config):
    """Return platform_config_file and link_mask overrides for a query JSON."""
    try:
        config = _QUERY_CONFIGS[query_config]
    except KeyError as exc:
        raise RuntimeError(
            f'unknown query_config {query_config!r}; expected one of '
            f'{sorted(_QUERY_CONFIGS)}') from exc

    params = load_parameters(param_file_path)
    if params.get('platform_config') != 'S56C_1_SHF3L_2':
        raise RuntimeError(
            f'query_config applies only to an S56C_1_SHF3L_2 parameter file; '
            f'{param_file_path} selects {params.get("platform_config")!r}')

    link_mask = params.get('link_mask')
    masks = link_mask if isinstance(link_mask, list) else [link_mask]
    if not masks or any(isinstance(mask, bool) or not isinstance(mask, int) for mask in masks):
        raise RuntimeError(
            f'{param_file_path} must define link_mask as an integer or integer list')
    enabled_masks = [mask for mask in masks if mask != 0]
    if len(enabled_masks) != 1:
        raise RuntimeError(
            f'{param_file_path} must select links on exactly one deserializer to use '
            f'query_config; got link_mask={link_mask!r}')
    plain_mask = enabled_masks[0]
    if plain_mask < 0 or plain_mask > 0xFFFF:
        raise RuntimeError(
            f'{param_file_path} link_mask must fit in uint16; got {plain_mask}')

    query_masks = [0] * config['transport_count']
    query_masks[config['mask_index']] = plain_mask
    overrides = {'link_mask': query_masks}
    platform_config_file = params.get('platform_config_file')
    if not platform_config_file or platform_config_file in _BUNDLED_QUERY_JSONS:
        overrides['platform_config_file'] = config['platform_config_file']
    return overrides


def _image_size_from_url(camera_info_url):
    camera_info_path = resolve_package_url(camera_info_url)
    with open(camera_info_path) as f:
        camera_info = yaml.safe_load(f)
    try:
        return int(camera_info['image_width']), int(camera_info['image_height'])
    except KeyError as exc:
        raise RuntimeError(
            f'{camera_info_path} must define image_width and image_height') from exc


def load_image_size(param_file_path):
    """Read image_width/image_height from the camera_info referenced by a param file."""
    params = load_parameters(param_file_path)
    try:
        camera_info_url = params['camera_info_url']
    except KeyError as exc:
        raise RuntimeError(
            f'{param_file_path} must define camera_info_url') from exc
    return _image_size_from_url(camera_info_url)


def load_camera_names(param_file_path):
    """Return the ordered camera names of a multi-camera parameter file."""
    return list(load_parameters(param_file_path).get('cameras', []))


def load_parameter(param_file_path, parameter_name, default=None):
    """Read a top-level ROS parameter from a parameter file."""
    return load_parameters(param_file_path).get(parameter_name, default)


def load_native_encoding(param_file_path, launch_encoding):
    """Return the node's native encoding after applying a launch override."""
    if launch_encoding in ('nv12', 'nv24'):
        return launch_encoding
    return load_parameter(param_file_path, 'encoding_desired', 'nv12')


def validate_resize_dimensions(width, height):
    """Reject invalid resize dimensions before loading components."""
    if width <= 0 or height <= 0:
        raise RuntimeError(
            f'resize_width and resize_height must be positive, got '
            f'{width}x{height}')


_NATIVE_SIPL_ENCODINGS = ('nv12', 'nv24', '')


def force_rgb8_for_image_proc(
        encoding_desired, *, rectify_enabled, resize_enabled):
    """
    Force rgb8 when RectifyNode or ResizeNode would see native NV12/NV24.

    Those nodes only accept packed formats, so ImageFormatConverterNode must
    run first. Returns ``(encoding, log_messages)``.
    """
    if encoding_desired not in _NATIVE_SIPL_ENCODINGS:
        return encoding_desired, []
    logs = []
    if rectify_enabled:
        logs.append(
            'Rectify enabled: converting native images to rgb8 '
            'for RectifyNode compatibility.')
    if resize_enabled:
        logs.append(
            'Resize enabled: converting native images to rgb8 '
            'for ResizeNode compatibility.')
    if not logs:
        return encoding_desired, []
    return 'rgb8', logs


def encoder_input(native_topic, resized_topic, native_size, resize_enabled,
                  resize_width, resize_height):
    """Return an encoder image topic and dimensions for optional resizing."""
    if resize_enabled:
        return resized_topic, (resize_width, resize_height)
    return native_topic, native_size


def load_camera_field(param_file_path, camera_name, field, default=None):
    """Read a per-camera field from nested or dotted ROS parameters."""
    params = load_parameters(param_file_path)
    nested = params.get('camera', {})
    if isinstance(nested, dict) and camera_name in nested:
        value = nested[camera_name].get(field)
        if value is not None:
            return value
    return params.get(f'camera.{camera_name}.{field}', default)


def load_camera_sizes(param_file_path):
    """Read calibrated image sizes by camera name; mono/stereo use the empty key."""
    names = load_camera_names(param_file_path)
    if not names:
        return {'': load_image_size(param_file_path)}

    sizes = {}
    for name in names:
        camera_info_url = load_camera_field(param_file_path, name, 'camera_info_url')
        if not camera_info_url:
            raise RuntimeError(
                f'{param_file_path} must define camera.{name}.camera_info_url')
        sizes[name] = _image_size_from_url(camera_info_url)
    return sizes
