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

import importlib.util
from pathlib import Path

from isaac_ros_sipl_camera.launch_utils import (
    encoder_input,
    force_rgb8_for_image_proc,
    validate_resize_dimensions,
)
from launch import LaunchContext
from launch.utilities import perform_substitutions
from launch_ros.actions import LoadComposableNodes
from launch_ros.utilities import evaluate_parameters, to_parameters_list
import pytest


_LAUNCH_FILE = Path(__file__).parents[1] / 'launch' / 'sipl_multi_camera.launch.py'
_SPEC = importlib.util.spec_from_file_location('sipl_multi_camera_launch', _LAUNCH_FILE)
_LAUNCH = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_LAUNCH)


def test_h264_rejects_top_level_nv24(tmp_path):
    """Reject a YAML-native NV24 stream even without per-camera overrides."""
    params = tmp_path / 'cameras.yaml'
    params.write_text(
        'sipl_multi_camera:\n'
        '  ros__parameters:\n'
        '    encoding_desired: nv24\n'
        '    cameras: [left, right]\n',
        encoding='utf-8')

    with pytest.raises(RuntimeError, match='left, right'):
        _LAUNCH._validate_nv12_inputs(str(params), ['left', 'right'], '')


def test_native_launch_override_takes_precedence(tmp_path):
    """Allow an explicit NV12 native override of the YAML base encoding."""
    params = tmp_path / 'cameras.yaml'
    params.write_text(
        'sipl_multi_camera:\n'
        '  ros__parameters:\n'
        '    encoding_desired: nv24\n'
        '    cameras: [left]\n',
        encoding='utf-8')

    _LAUNCH._validate_nv12_inputs(str(params), ['left'], 'nv12')


def test_empty_camera_encoding_inherits_top_level_nv24(tmp_path):
    """Treat an empty per-camera encoding as inheritance from the base encoding."""
    params = tmp_path / 'cameras.yaml'
    params.write_text(
        'sipl_multi_camera:\n'
        '  ros__parameters:\n'
        '    encoding_desired: nv24\n'
        '    cameras: [left]\n'
        '    camera:\n'
        '      left:\n'
        '        encoding: ""\n',
        encoding='utf-8')

    with pytest.raises(RuntimeError, match='left'):
        _LAUNCH._validate_nv12_inputs(str(params), ['left'], '')


@pytest.fixture(params=['sipl_camera', 'sipl_stereo_camera', 'sipl_multi_camera'])
def camera_launch(request):
    path = _LAUNCH_FILE.with_name(f'{request.param}.launch.py')
    spec = importlib.util.spec_from_file_location(request.param, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return request.param, module


def _launch_context(param_file, namespace='', **overrides):
    context = LaunchContext()
    context.launch_configurations.update({
        'param_file': str(param_file),
        'encoding_desired': '',
        'target_container': 'test_container',
        'enable_rectify': 'false',
        'enable_encoding': 'false',
        'enable_resize': 'false',
        'resize_width': '640',
        'resize_height': '480',
        'enforce_stereo_pairs': 'false',
        'log_level': '',
        'camera_namespace': namespace,
        'query_config': '',
    })
    context.launch_configurations.update(overrides)
    return context


def _launch_nodes(module, context, monkeypatch):
    monkeypatch.setattr(
        module, 'LoadComposableNodes',
        lambda **kwargs: kwargs['composable_node_descriptions'])
    return module.launch_setup(context)[-1]


def _node_parameters(context, node):
    name = perform_substitutions(context, node.node_name)
    namespace = perform_substitutions(context, node.node_namespace or [])
    parameters = to_parameters_list(
        context, name, namespace, evaluate_parameters(context, node.parameters))
    return {parameter.name: parameter.value for parameter in parameters}


def _camera_parameters(module, param_file, namespace, monkeypatch):
    context = _launch_context(param_file, namespace)
    return _node_parameters(context, _launch_nodes(module, context, monkeypatch)[-1])


def test_resize_preserves_native_override(camera_launch, tmp_path, monkeypatch):
    """Keep the NV12 override on the camera and RGB8 on the converter."""
    name, module = camera_launch
    calibration = tmp_path / 'camera_info.yaml'
    calibration.write_text('image_width: 1920\nimage_height: 1536\n', encoding='utf-8')
    params = tmp_path / 'camera.yaml'
    params.write_text(
        f'{name}:\n'
        '  ros__parameters:\n'
        '    encoding_desired: nv24\n'
        f'    camera_info_url: {calibration}\n'
        '    cameras: [left]\n'
        '    camera:\n'
        '      left:\n'
        '        yuv_color_spec: bt709\n'
        f'        camera_info_url: {calibration}\n', encoding='utf-8')
    context = _launch_context(
        params, encoding_desired='nv12', enable_resize='true', enable_encoding='true')
    nodes = _launch_nodes(module, context, monkeypatch)
    assert _node_parameters(context, nodes[-1])['encoding_desired'] == 'nv12'
    converters = [
        node for node in nodes
        if perform_substitutions(context, node.node_plugin).endswith('ImageFormatConverterNode')
    ]
    assert len(converters) == (2 if name == 'sipl_stereo_camera' else 1)
    assert all(_node_parameters(context, node)['encoding_desired'] == 'rgb8'
               for node in converters)

    # Without the override, reject NV24 even when H.264 is disabled.
    context = _launch_context(params, enable_resize='true')
    with pytest.raises(RuntimeError, match='nv24'):
        _launch_nodes(module, context, monkeypatch)


def test_resize_rejects_per_camera_nv24(tmp_path, monkeypatch):
    params = tmp_path / 'cameras.yaml'
    params.write_text(
        'sipl_multi_camera:\n'
        '  ros__parameters:\n'
        '    cameras: [left]\n'
        '    camera.left.encoding: nv24\n', encoding='utf-8')
    context = _launch_context(params, encoding_desired='nv12', enable_resize='true')
    with pytest.raises(RuntimeError, match='native nv24.*left'):
        _launch_nodes(_LAUNCH, context, monkeypatch)


def test_namespaced_launch_keeps_camera_parameters(camera_launch, tmp_path, monkeypatch):
    name, module = camera_launch
    calibration = tmp_path / 'camera_info.yaml'
    calibration.write_text('image_width: 1920\nimage_height: 1536\n', encoding='utf-8')
    params = tmp_path / 'camera.yaml'
    params.write_text(
        f'{name}:\n'
        '  ros__parameters:\n'
        '    platform_config: S56C_1_SHF3L_2\n'
        '    link_mask: [0, 13]\n'
        f'    camera_info_url: {calibration}\n'
        '    cameras: [left]\n'
        '    camera:\n'
        '      left:\n'
        '        sensor_id: 2\n'
        f'        camera_info_url: {calibration}\n', encoding='utf-8')

    actual = _camera_parameters(module, params, '/robot', monkeypatch)

    assert actual['platform_config'] == 'S56C_1_SHF3L_2'
    assert list(actual['link_mask']) == [0, 13]
    assert actual['camera.left.sensor_id'] == 2


def test_native_launch_accepts_uncalibrated_camera(camera_launch, tmp_path, monkeypatch):
    name, module = camera_launch
    params = tmp_path / 'camera.yaml'
    params.write_text(
        f'{name}:\n'
        '  ros__parameters:\n'
        '    platform_config: S56C_1_SHF3L_2\n'
        '    cameras: [left]\n'
        '    camera:\n'
        '      left:\n'
        '        sensor_id: 2\n', encoding='utf-8')

    actual = _camera_parameters(module, params, '', monkeypatch)

    assert actual['platform_config'] == 'S56C_1_SHF3L_2'


@pytest.mark.parametrize('width,height', [(0, 480), (640, 0), (-1, 480)])
def test_resize_rejects_nonpositive_dimensions(width, height):
    """Reject invalid dimensions before loading any GPU components."""
    with pytest.raises(RuntimeError, match=f'{width}x{height}'):
        validate_resize_dimensions(width, height)


def test_encoder_uses_resized_image_when_resize_is_enabled():
    """Encode the configured derivative instead of the native camera image."""
    topic, size = encoder_input(
        '/head/left/image_raw', '/head/left/image_resize',
        (2560, 1984), True, 640, 480)

    assert topic == '/head/left/image_resize'
    assert size == (640, 480)


def test_encoder_preserves_native_input_when_resize_is_disabled():
    """Keep the existing native encoding behavior by default."""
    topic, size = encoder_input(
        '/head/left/image_raw', '/head/left/image_resize',
        (2560, 1984), False, 640, 480)

    assert topic == '/head/left/image_raw'
    assert size == (2560, 1984)


def test_force_rgb8_for_resize_and_rectify():
    """Native NV12/NV24 must convert to rgb8 before RectifyNode or ResizeNode."""
    encoding, logs = force_rgb8_for_image_proc(
        '', rectify_enabled=False, resize_enabled=True)
    assert encoding == 'rgb8'
    assert any('Resize' in message for message in logs)

    encoding, logs = force_rgb8_for_image_proc(
        'nv12', rectify_enabled=True, resize_enabled=False)
    assert encoding == 'rgb8'
    assert any('Rectify' in message for message in logs)

    encoding, logs = force_rgb8_for_image_proc(
        'bgr8', rectify_enabled=True, resize_enabled=True)
    assert encoding == 'bgr8'
    assert logs == []


@pytest.mark.parametrize('topic_namespace', ['head/left', '/head/left'])
def test_multi_camera_launch_wires_resized_outputs(tmp_path, topic_namespace):
    """Load each resize node in the camera namespace with calibrated outputs."""
    camera_info = tmp_path / 'camera_info.yaml'
    camera_info.write_text(
        'image_width: 2560\n'
        'image_height: 1984\n',
        encoding='utf-8')
    params = tmp_path / 'cameras.yaml'
    params.write_text(
        'sipl_multi_camera:\n'
        '  ros__parameters:\n'
        '    cameras: [left]\n'
        '    camera:\n'
        '      left:\n'
        f'        topic_namespace: {topic_namespace}\n'
        '        yuv_color_spec: bt709\n'
        f'        camera_info_url: file://{camera_info}\n',
        encoding='utf-8')

    context = LaunchContext()
    context.launch_configurations.update({
        'camera_namespace': 'camera',
        'enable_encoding': 'false',
        'enable_rectify': 'false',
        'enable_resize': 'true',
        'encoding_desired': '',
        'log_level': '',
        'param_file': str(params),
        'resize_height': '480',
        'resize_width': '640',
        'query_config': '',
        'target_container': 'camera_container',
    })

    load_actions = [
        action for action in _LAUNCH.launch_setup(context)
        if isinstance(action, LoadComposableNodes)
    ]
    assert len(load_actions) == 1
    nodes = load_actions[0]._LoadComposableNodes__composable_node_descriptions
    converter_nodes = [
        node for node in nodes
        if perform_substitutions(context, node.node_plugin) ==
        'nvidia::isaac_ros::image_proc::ImageFormatConverterNode'
    ]
    assert len(converter_nodes) == 1
    resize_nodes = [
        node for node in nodes
        if perform_substitutions(context, node.node_plugin) ==
        'nvidia::isaac_ros::image_proc::ResizeNode'
    ]
    assert len(resize_nodes) == 1

    resize_node = resize_nodes[0]
    assert perform_substitutions(context, resize_node.node_name) == 'left_resize'
    assert perform_substitutions(context, resize_node.node_namespace) == 'camera'
    remappings = {
        perform_substitutions(context, source):
        perform_substitutions(context, destination)
        for source, destination in resize_node.remappings
    }
    assert remappings == {
        'image': 'head/left/image_converted',
        'camera_info': 'head/left/camera_info',
        'resize/image': 'head/left/image_resize',
        'resize/camera_info': 'head/left/camera_info_resize',
    }


@pytest.mark.parametrize(
    ('launch_name', 'image_topic', 'camera_info_topic'),
    [
        ('sipl_camera.launch.py', 'image_resize', 'camera_info_resize'),
        ('sipl_stereo_camera.launch.py',
         '{side}/image_resize', '{side}/camera_info_resize'),
        ('sipl_multi_camera.launch.py',
         '{ns}/image_resize', '{ns}/camera_info_resize'),
    ],
)
def test_all_camera_launches_expose_resize_outputs(
        launch_name, image_topic, camera_info_topic):
    """Keep optional resized image and CameraInfo outputs consistent."""
    source = (_LAUNCH_FILE.parent / launch_name).read_text()

    assert "'enable_resize'" in source
    assert "'resize_width'" in source
    assert "'resize_height'" in source
    assert 'ResizeNode' in source
    assert image_topic in source
    assert camera_info_topic in source
    assert 'image_converted' in source
    assert 'force_rgb8_for_image_proc' in source
