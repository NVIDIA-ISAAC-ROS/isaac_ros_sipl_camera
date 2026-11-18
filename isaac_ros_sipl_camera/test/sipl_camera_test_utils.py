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

import collections
import glob
import math
import os
import subprocess

import yaml

# Camera-over-Ethernet Eagle VB1940 camera defaults
SIPL_NETWORK_INTERFACE = 'mgbe0_0'
SIPL_CAMERA_IP = '192.168.0.2'
STARTUP_TIME_MAX_DELAY = 10
EXPECTED_FPS = 30

CADENCE_WARMUP_SECONDS = 1
CADENCE_VALIDATION_SECONDS = 3
FRAME_PERIOD_TOLERANCE = 0.05

# Hawk GMSL defaults (single Hawk on link 0).
HAWK_RESOLUTION = (1920, 1200)
HAWK_LINK_MASK = 0x0001


def cadence_required_frames(expected_fps):
    """Return the frames needed for warmup plus a complete cadence window."""
    if expected_fps <= 0.0:
        raise ValueError(f'expected_fps must be positive, got {expected_fps}')
    warmup_frames = math.ceil(expected_fps * CADENCE_WARMUP_SECONDS)
    validation_intervals = math.ceil(expected_fps * CADENCE_VALIDATION_SECONDS)
    return warmup_frames + validation_intervals + 1


def assert_consistent_frame_period(test_case, stream_name, stamps_ns, expected_fps):
    """Require every post-warmup capture interval to remain within 5% of nominal."""
    required_frames = cadence_required_frames(expected_fps)
    test_case.assertGreaterEqual(
        len(stamps_ns), required_frames,
        f'[{stream_name}] insufficient CameraInfo messages for cadence validation: '
        f'{len(stamps_ns)} < {required_frames}')

    warmup_frames = math.ceil(expected_fps * CADENCE_WARMUP_SECONDS)
    samples = stamps_ns[warmup_frames:required_frames]
    intervals_ns = [
        current - previous
        for previous, current in zip(samples, samples[1:])
    ]
    nominal_period_ns = 1_000_000_000.0 / expected_fps
    minimum_period_ns = nominal_period_ns * (1.0 - FRAME_PERIOD_TOLERANCE)
    maximum_period_ns = nominal_period_ns * (1.0 + FRAME_PERIOD_TOLERANCE)

    for sample, interval_ns in enumerate(intervals_ns):
        test_case.assertGreaterEqual(
            interval_ns, minimum_period_ns,
            f'[{stream_name}] post-warmup interval {sample} is too short: '
            f'{interval_ns / 1e6:.3f} ms; expected '
            f'{nominal_period_ns / 1e6:.3f} ms +/- '
            f'{FRAME_PERIOD_TOLERANCE * 100:.1f}%')
        test_case.assertLessEqual(
            interval_ns, maximum_period_ns,
            f'[{stream_name}] post-warmup interval {sample} is too long: '
            f'{interval_ns / 1e6:.3f} ms; expected '
            f'{nominal_period_ns / 1e6:.3f} ms +/- '
            f'{FRAME_PERIOD_TOLERANCE * 100:.1f}%')


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


def detect_hawk_camera():
    """
    Return True if a Hawk GMSL camera is wired up on this host.

    Detected via the device tree: the kernel only populates
    ``/proc/device-tree/tegra-camera-platform/modules/*/badge`` with ``ar0234``
    when the Hawk DT overlay is applied and the GMSL links lock.
    """
    for badge in glob.glob('/proc/device-tree/tegra-camera-platform/modules/module*/badge'):
        try:
            with open(badge, 'rb') as f:
                if b'ar0234' in f.read().lower():
                    return True
        except OSError:
            continue
    return False


# Sensing cameras require explicit bench selection; UDDF drivers provide no passive probe.
NVSIPL_DRV_DIR = '/usr/lib/nvsipl_drv'
SGX_DRIVER_SO = f'{NVSIPL_DRV_DIR}/libnvuddf_sgx_yuv_gmsl2_cameramodule_library.so'
S56C_DRIVER_SO = f'{NVSIPL_DRV_DIR}/libnvuddf_s56c_cameramodule_library.so'


def detect_sgx_driver_fixed():
    """
    Return True if the SGX_YUV_GMSL2 driver assigns per-link serializer addresses.

    The stale vendor build hardcodes 0x4C for every MAX96717, so two SHF3L
    modules collide on one I2C address and the second link dies with
    NOTIF_ERROR_ICP_CAPTURE_FAILURE. Only the rebuilt driver can run both.
    """
    try:
        with open(SGX_DRIVER_SO, 'rb') as f:
            blob = f.read()
    except OSError:
        return False
    return b'final I2C address' in blob and b'unique 0x4C' not in blob


def _s56c_driver_installed():
    """Report whether the S56C camera module driver is installed."""
    return os.path.exists(S56C_DRIVER_SO)


def _sgx_driver_installed():
    """Report whether the SGX_YUV_GMSL2 (SHF3L) camera module driver is installed."""
    return os.path.exists(SGX_DRIVER_SO)


# Names the cameras wired up on this bench, comma and/or space separated and
# matched case-insensitively, e.g. ISAAC_ROS_SIPL_CAMERAS='eagle, s56c'.
CAMERA_SELECTOR_ENV = 'ISAAC_ROS_SIPL_CAMERAS'

DEFAULT_CAMERAS = ('eagle', 'hawk')

# probe: passive presence check, or None when no reliable one exists.
# prerequisites: (predicate, failure text) pairs saying whether a camera that is
# present could actually be driven here.
_CameraSpec = collections.namedtuple(
    '_CameraSpec', ['probe', 'probe_failure', 'prerequisites'],
    defaults=(None, None, ()))

CAMERAS = {
    'eagle': _CameraSpec(
        probe=detect_ethernet_camera,
        probe_failure=f'nothing answers at {SIPL_CAMERA_IP} over {SIPL_NETWORK_INTERFACE}'),
    'hawk': _CameraSpec(
        probe=detect_hawk_camera,
        probe_failure='no ar0234 module in the device tree'),
    's56c': _CameraSpec(
        prerequisites=(
            (_s56c_driver_installed, f'{S56C_DRIVER_SO} is not installed'),)),
    'shf3l': _CameraSpec(
        prerequisites=(
            (_sgx_driver_installed, f'{SGX_DRIVER_SO} is not installed'),)),
}


def selected_cameras():
    """
    Return the set of camera names the operator declared present on this bench.

    Unset selects DEFAULT_CAMERAS; set is an explicit allowlist, so an empty
    value selects nothing and skips every camera test.
    """
    declared = os.environ.get(CAMERA_SELECTOR_ENV)
    if declared is None:
        return set(DEFAULT_CAMERAS)
    names = {name.lower() for name in declared.replace(',', ' ').split()}
    unknown = sorted(names - set(CAMERAS))
    if unknown:
        raise ValueError(
            f'{CAMERA_SELECTOR_ENV} names unknown camera(s) {unknown}; '
            f'valid names are {sorted(CAMERAS)}')
    return names


def camera_skip_reason(camera):
    """Return why `camera` cannot be exercised on this host, or None if it can."""
    spec = CAMERAS[camera]
    if camera not in selected_cameras():
        if spec.probe is None:
            return (f'{camera} has no passive probe, so name it in '
                    f'{CAMERA_SELECTOR_ENV} to declare it wired up')
        return (f'{camera} is not named in {CAMERA_SELECTOR_ENV}='
                f'{os.environ.get(CAMERA_SELECTOR_ENV, "")}')
    if spec.probe is not None and not spec.probe():
        return f'{camera} is selected but not detected ({spec.probe_failure})'
    for prerequisite_met, failure in spec.prerequisites:
        if not prerequisite_met():
            return f'{camera} is selected but cannot be driven ({failure})'
    return None


# Config file -> cameras it drives, plus any requirement of that rig alone.
_ConfigRequirement = collections.namedtuple(
    '_ConfigRequirement', ['cameras', 'prerequisites'], defaults=((), ()))

CONFIG_REQUIREMENTS = {
    'eagle_mono.yaml': _ConfigRequirement(cameras=('eagle',)),
    'eagle_stereo.yaml': _ConfigRequirement(cameras=('eagle',)),
    'hawk_stereo.yaml': _ConfigRequirement(cameras=('hawk',)),
    's56c_stereo.yaml': _ConfigRequirement(cameras=('s56c',)),
    'shf3l_mono.yaml': _ConfigRequirement(cameras=('shf3l',)),
    'shf3l_mono_cam3.yaml': _ConfigRequirement(cameras=('shf3l',)),
    's56c_shf3l_mixed.yaml': _ConfigRequirement(
        cameras=('s56c', 'shf3l'),
        prerequisites=(
            (detect_sgx_driver_fixed,
             'the installed SGX_YUV_GMSL2 driver gives every serializer the same '
             'I2C address, so it cannot run two SHF3L modules at once'),)),
}

# Stereo configs that parametrized stereo tests iterate over. Each config is
# gated by config_skip_reason() so a host that only has one camera (or none)
# skips the absent configurations cleanly. Eagle is listed first because it is
# the day-zero supported path.
STEREO_CONFIGS = ['eagle_stereo.yaml', 'hawk_stereo.yaml', 's56c_stereo.yaml']

MONO_CONFIGS = ['eagle_mono.yaml', 'shf3l_mono.yaml', 'shf3l_mono_cam3.yaml']


def config_skip_reason(config_name):
    """Return why `config_name` cannot be exercised on this host, or None if it can."""
    requirement = CONFIG_REQUIREMENTS[config_name]
    for camera in requirement.cameras:
        reason = camera_skip_reason(camera)
        if reason is not None:
            return f'{config_name}: {reason}'
    for prerequisite_met, failure in requirement.prerequisites:
        if not prerequisite_met():
            return f'{config_name}: {failure}'
    return None


def load_camera_info_from_yaml(yaml_path: str):
    """Load a ROS camera_info YAML file into a sensor_msgs.msg.CameraInfo."""
    from sensor_msgs.msg import CameraInfo

    with open(yaml_path) as f:
        data = yaml.safe_load(f)

    info = CameraInfo()
    info.width = int(data['image_width'])
    info.height = int(data['image_height'])
    info.distortion_model = data.get('distortion_model', '')
    info.k = [float(v) for v in data['camera_matrix']['data']]
    info.d = [float(v) for v in data['distortion_coefficients']['data']]
    info.r = [float(v) for v in data['rectification_matrix']['data']]
    info.p = [float(v) for v in data['projection_matrix']['data']]
    return info
