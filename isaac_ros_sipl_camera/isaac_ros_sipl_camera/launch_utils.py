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

"""Shared helpers for the SIPL camera launch files (mono and stereo)."""

from urllib.parse import unquote, urlparse

import isaac_ros_launch_utils as lu
import yaml


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


def load_image_size(param_file_path):
    """Read image_width/image_height from the camera_info referenced by a param file."""
    try:
        with open(param_file_path) as f:
            params = next(iter(yaml.safe_load(f).values()))['ros__parameters']
    except Exception as exc:
        raise RuntimeError(
            f'{param_file_path} must be a ROS parameter file with ros__parameters') from exc

    try:
        camera_info_path = resolve_package_url(params['camera_info_url'])
    except KeyError as exc:
        raise RuntimeError(
            f'{param_file_path} must define camera_info_url') from exc

    with open(camera_info_path) as f:
        camera_info = yaml.safe_load(f)
    try:
        return int(camera_info['image_width']), int(camera_info['image_height'])
    except KeyError as exc:
        raise RuntimeError(
            f'{camera_info_path} must define image_width and image_height') from exc
