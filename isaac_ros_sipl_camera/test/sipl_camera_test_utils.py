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

"""Shared utilities for SIPL camera integration tests."""

import os
import subprocess

import yaml

# Camera-over-Ethernet Eagle VB1940 camera defaults
SIPL_NETWORK_INTERFACE = 'mgbe0_0'
SIPL_CAMERA_IP = '192.168.0.2'
STARTUP_TIME_MAX_DELAY = 10
EXPECTED_FPS = 30


def load_config_for_test(config_path, namespace, node_name, container_name=None):
    """
    Load a node config YAML and return a parameters list for use with ComposableNode.

    Config files use keys like 'sipl_camera' or 'sipl_stereo_camera' with ros__parameters
    underneath. For composable nodes loaded in a test namespace, the launch system may not
    apply keyed parameter dicts. This helper returns the ros__parameters content as a
    single dict so it is applied as direct parameter overrides to the node.
    """
    with open(config_path) as f:
        data = yaml.safe_load(f)
    if not data or len(data) != 1:
        keys = list(data.keys()) if data else []
        raise ValueError(
            f'Config {config_path} must have exactly one top-level key (node name), got {keys}')
    node_block = list(data.values())[0]
    # Pass only the inner param dict. launch_ros flattens nested dicts with "." so
    # {"ros__parameters": {"ip_address": "x"}} would become "ros__parameters.ip_address".
    # The node expects "ip_address", so we pass the inner dict directly.
    if isinstance(node_block, dict) and 'ros__parameters' in node_block:
        return [node_block['ros__parameters']]
    return [node_block]


def detect_ethernet_camera(interface=SIPL_NETWORK_INTERFACE, camera_ip=SIPL_CAMERA_IP):
    """Return True if the CoE camera is reachable via the expected network interface."""
    if not os.path.exists(f'/sys/class/net/{interface}'):
        return False

    # Check if a physical link is established
    try:
        with open(f'/sys/class/net/{interface}/carrier', 'r') as f:
            if f.read().strip() != '1':
                return False
    except OSError:
        pass

    try:
        result = subprocess.run(
            ['ping', '-c', '1', '-W', '1', camera_ip],
            capture_output=True, text=True, timeout=5)
        return result.returncode == 0
    except Exception as e:
        raise RuntimeError(f'Failed to ping {camera_ip}: {e}') from e
