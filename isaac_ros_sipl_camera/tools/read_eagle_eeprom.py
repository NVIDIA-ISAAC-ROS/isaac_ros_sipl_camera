#!/usr/bin/env python3
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

r"""
Read Eagle camera EEPROM calibration data and write ROS 2 CameraInfo YAML.

Simple tool to read Eagle camera EEPROM calibration data and write
ROS 2 camera_info_manager-compatible YAML.

Usage:
    python3 read_eagle_eeprom.py --camera left --part 0 \
        --output-dir ./eagle_camera_info

EEPROM layout follows hololink/sensors/vb1940/vb1940.py:
    L_fx  L_fy  L_cx  L_cy  L_k1  L_k2  L_p1  L_p2
    L_k3  L_k4  L_k5  L_k6  R_fx  R_fy  R_cx  R_cy
    R_k1  R_k2  R_p1  R_p2  R_k3  R_k4  R_k5  R_k6
    Rx    Ry    Rz    Tx    Ty    Tz    [pad] sn
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import argparse
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
import sys
import traceback

import cv2
import numpy as np

DEFAULT_IP_ADDR = '192.168.0.2'
DISTORTION_MODEL = 'rational_polynomial'
DEFAULT_OUTPUT_DIR = 'eagle_camera_info'
# The Eagle sensor's native resolution is 2560x1984. The EEPROM factory
# calibration intrinsics (focal length, principal point) are in pixel units
# calibrated at this resolution and are only valid at this size.
EAGLE_NATIVE_SIZE = (2560, 1984)
PART_ORDER = ('RGB', 'IR')


def import_hololink():
    """Import hololink and validate that the native extension is available."""
    try:
        import hololink as hololink_module
    except ImportError as exc:
        print(
            f'FATAL: Failed to import hololink package: {exc}\n'
            'Install the hololink package or run this script inside '
            'the hololink-demo container.',
            file=sys.stderr,
        )
        raise SystemExit(1) from exc
    return hololink_module


def enumerate_camera(hololink_module, ip_addr: str, timeout_s: float) -> dict:
    """Enumerate the Hololink board and verify the camera is reachable."""
    try:
        timeout = hololink_module.Timeout(timeout_s=timeout_s)
        channel_metadata = hololink_module.Enumerator.find_channel(
            channel_ip=ip_addr,
            timeout=timeout,
        )
    except Exception as exc:
        print(
            f'FATAL: Camera not found at {ip_addr} '
            f'(timeout after {timeout_s}s): {exc}. Check what camera is reachable with '
            'running `timeout 3 hololink-enumerate`',
            file=sys.stderr,
        )
        raise SystemExit(1) from exc

    print(f'Camera enumerated at {ip_addr}', file=sys.stderr)
    return channel_metadata


def _format_float(value: float) -> str:
    return f'{value:.12g}'


def _format_list(values: Iterable[float]) -> str:
    return f'[{", ".join(_format_float(value) for value in values)}]'


def _normalize_distortion(coeffs: Iterable[float], length: int = 8) -> list[float]:
    normalized = list(coeffs)[:length]
    if len(normalized) < length:
        normalized.extend([0.0] * (length - len(normalized)))
    return normalized


def _flatten_row_major(matrix: Iterable[Iterable[float]]) -> list[float]:
    return np.asarray(matrix, dtype=float).reshape(-1).tolist()


def _ensure_matrix_3x3(matrix: list[list[float]], name: str) -> None:
    if len(matrix) != 3 or any(len(row) != 3 for row in matrix):
        raise ValueError(f'{name} must be a 3x3 matrix.')


def _extract_intrinsics(matrix: list[list[float]]) -> tuple[float, float, float, float]:
    _ensure_matrix_3x3(matrix, 'intrinsic_matrix')
    fx = matrix[0][0]
    fy = matrix[1][1]
    cx = matrix[0][2]
    cy = matrix[1][2]
    return fx, fy, cx, cy


@dataclass(frozen=True)
class SingleCameraCalibration:
    intrinsic_matrix: list[list[float]]
    distortion_coefficients: list[float]


@dataclass(frozen=True)
class StereoCalibration:
    left: SingleCameraCalibration
    right: SingleCameraCalibration
    rotation_rvec: list[float]
    translation: list[float]
    serial_number: int | None


@dataclass(frozen=True)
class CalibrationBundle:
    raw: dict[str, dict[str, object]]
    parts: dict[str, StereoCalibration]


@dataclass(frozen=True)
class CameraInfoYaml:
    image_width: int
    image_height: int
    camera_name: str
    camera_matrix: list[float]
    distortion_model: str
    distortion_coefficients: list[float]
    rectification_matrix: list[float]
    projection_matrix: list[float]

    def to_yaml(self) -> str:
        return '\n'.join(
            [
                f'image_width: {self.image_width}',
                f'image_height: {self.image_height}',
                f'camera_name: "{self.camera_name}"',
                'camera_matrix:',
                '  rows: 3',
                '  cols: 3',
                f'  data: {_format_list(self.camera_matrix)}',
                f'distortion_model: {self.distortion_model}',
                'distortion_coefficients:',
                '  rows: 1',
                f'  cols: {len(self.distortion_coefficients)}',
                f'  data: {_format_list(self.distortion_coefficients)}',
                'rectification_matrix:',
                '  rows: 3',
                '  cols: 3',
                f'  data: {_format_list(self.rectification_matrix)}',
                'projection_matrix:',
                '  rows: 3',
                '  cols: 4',
                f'  data: {_format_list(self.projection_matrix)}',
            ]
        ) + '\n'


class CalibrationWorkflow(ABC):
    """Abstract pipeline for extracting and exporting calibration data."""

    def run(self) -> None:
        raw = self.read_camera()
        calibration = self.convert_to_calibration_data(raw)
        self.output_to_camera_info_file(calibration)

    @abstractmethod
    def read_camera(self) -> dict[str, dict[str, object]]:
        raise NotImplementedError

    @abstractmethod
    def convert_to_calibration_data(
        self, raw: dict[str, dict[str, object]]
    ) -> CalibrationBundle:
        raise NotImplementedError

    @abstractmethod
    def output_to_camera_info_file(self, calibration: CalibrationBundle) -> None:
        raise NotImplementedError


class EagleCalibrationWorkflow(CalibrationWorkflow):
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args

    def read_camera(self) -> dict[str, dict[str, object]]:
        hololink_module = import_hololink()
        channel_metadata = enumerate_camera(
            hololink_module, self.args.ip_addr, self.args.timeout_s
        )

        hololink_channel = hololink_module.DataChannel(channel_metadata)
        hololink = hololink_channel.hololink()
        hololink.start()

        try:
            print('Connecting to Eagle camera...', file=sys.stderr)
            print(
                '(This may fail with a core dump if camera is not ready)',
                file=sys.stderr,
            )
            camera = hololink_module.sensors.vb1940.Vb1940Cam(hololink_channel)

            part_names = {0: 'RGB', 1: 'IR', 2: 'RGB and IR'}
            print(
                f'Reading {part_names[self.args.part]} calibration data from EEPROM...',
                file=sys.stderr,
            )
            return camera.get_calibration_data(self.args.part)
        finally:
            try:
                hololink.stop()
            except Exception as e:
                print(f'Warning: Failed to stop hololink: {e}', file=sys.stderr)

    def convert_to_calibration_data(
        self, raw: dict[str, dict[str, object]]
    ) -> CalibrationBundle:
        parts: dict[str, StereoCalibration] = {}
        for part_name in PART_ORDER:
            part_data = raw.get(part_name, {})
            if part_data:
                parts[part_name] = self._parse_part(part_name, part_data)

        if not parts:
            raise RuntimeError('No calibration data was returned from EEPROM.')

        return CalibrationBundle(raw=raw, parts=parts)

    def output_to_camera_info_file(self, calibration: CalibrationBundle) -> None:
        for part_name, part_calibration in calibration.parts.items():
            sn = part_calibration.serial_number
            if sn is not None and not (isinstance(sn, float) and np.isnan(sn)):
                print(
                    f'{part_name} serial number: {sn}',
                    file=sys.stderr,
                )

        targets = self._build_targets(calibration)
        camera_infos = {
            target: self._build_camera_info(target, calibration)
            for target in targets
        }

        output_paths = self._resolve_output_paths(camera_infos)
        for target, camera_info in camera_infos.items():
            output_path = output_paths[target]
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(camera_info.to_yaml(), encoding='utf-8')
            print(f'Wrote ROS 2 CameraInfo to {output_path}', file=sys.stderr)

    def _parse_part(
        self, part_name: str, part_data: dict[str, object]
    ) -> StereoCalibration:
        left_intrinsic = part_data.get('left_intrinsic_parameter')
        right_intrinsic = part_data.get('right_intrinsic_parameter')
        left_distortion = part_data.get('left_distortion_parameters')
        right_distortion = part_data.get('right_distortion_parameters')

        if left_intrinsic is None or right_intrinsic is None:
            raise RuntimeError(f'{part_name} intrinsic parameters are missing.')
        if left_distortion is None or right_distortion is None:
            raise RuntimeError(f'{part_name} distortion parameters are missing.')

        left = SingleCameraCalibration(
            intrinsic_matrix=left_intrinsic,
            distortion_coefficients=_normalize_distortion(left_distortion),
        )
        right = SingleCameraCalibration(
            intrinsic_matrix=right_intrinsic,
            distortion_coefficients=_normalize_distortion(right_distortion),
        )
        # Rx,Ry,Rz and Tx,Ty,Tz are supplied by vb1940.py; data[30] is reserved.
        rotation = part_data.get('R', [0.0, 0.0, 0.0])
        translation = part_data.get('T', [0.0, 0.0, 0.0])
        serial = part_data.get('sn')
        return StereoCalibration(
            left=left,
            right=right,
            rotation_rvec=rotation,
            translation=translation,
            serial_number=serial,
        )

    def _ordered_parts(self, calibration: CalibrationBundle) -> list[str]:
        return [part for part in PART_ORDER if part in calibration.parts]

    def _build_targets(self, calibration: CalibrationBundle) -> list[tuple[str, str]]:
        parts = self._ordered_parts(calibration)
        if self.args.camera == 'both':
            sides = ['left', 'right']
        else:
            sides = [self.args.camera]
        return [(part, side) for part in parts for side in sides]

    def _build_camera_info(
        self, target: tuple[str, str], calibration: CalibrationBundle
    ) -> CameraInfoYaml:
        part_name, side = target
        stereo_calibration = calibration.parts[part_name]
        camera_cal = (
            stereo_calibration.left if side == 'left' else stereo_calibration.right
        )
        R1, R2, P1, P2 = self._compute_stereo_rectification(
            stereo_calibration,
        )

        fx, fy, cx, cy = _extract_intrinsics(camera_cal.intrinsic_matrix)
        camera_matrix = [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]
        # Use R1/P1 for left, R2/P2 for right -- directly from
        # cv2.stereoRectify, matching rectify_params_generator.cpp.
        if side == 'left':
            rectification_matrix = _flatten_row_major(R1)
            projection_matrix = _flatten_row_major(P1)
        else:
            rectification_matrix = _flatten_row_major(R2)
            projection_matrix = _flatten_row_major(P2)

        image_width, image_height = self._resolve_image_size()
        camera_name = self._resolve_camera_name(
            target, len(calibration.parts), self.args.camera
        )

        return CameraInfoYaml(
            image_width=image_width,
            image_height=image_height,
            camera_name=camera_name,
            camera_matrix=camera_matrix,
            distortion_model=DISTORTION_MODEL,
            distortion_coefficients=camera_cal.distortion_coefficients,
            rectification_matrix=rectification_matrix,
            projection_matrix=projection_matrix,
        )

    def _resolve_camera_name(
        self, target: tuple[str, str], part_count: int, camera_selection: str
    ) -> str:
        part_name, side = target
        multiple = part_count > 1 or camera_selection == 'both'
        if self.args.camera_name:
            if multiple:
                return f'{self.args.camera_name}_{part_name.lower()}_{side}'
            return self.args.camera_name
        return f'eagle_{part_name.lower()}_{side}'

    def _resolve_image_size(self) -> tuple[int, int]:
        return EAGLE_NATIVE_SIZE

    def _compute_stereo_rectification(
        self, stereo_calibration: StereoCalibration
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        # Uses cv2.stereoRectify, matching the approach in
        # isaac_ros rectify_params_generator.cpp:
        #   cv::stereoRectify(left_K, left_D, right_K, right_D,
        #                     imagesize, R, T, R1, R2, P1, P2, Q,
        #                     cv::CALIB_ZERO_DISPARITY, alpha)
        #
        # EEPROM layout (from vb1940.py):
        #   Rx,Ry,Rz = rotation vector (Rodrigues), converted to
        #              3x3 via cv2.Rodrigues.
        #   Tx,Ty,Tz = translation vector (left-to-right).
        #   Distortion order from EEPROM: k1,k2,p1,p2,k3,k4,k5,k6
        #     which is already OpenCV order.
        #
        # Build intrinsic matrices (3x3) from parsed calibration.
        left_K = np.array(
            stereo_calibration.left.intrinsic_matrix,
            dtype=np.float64,
        )
        right_K = np.array(
            stereo_calibration.right.intrinsic_matrix,
            dtype=np.float64,
        )

        for name, K in [('left', left_K), ('right', right_K)]:
            if K[0, 0] <= 0 or K[1, 1] <= 0:
                raise ValueError(
                    f'Invalid {name} intrinsics: fx={K[0, 0]}, fy={K[1, 1]}'
                )

        # Distortion coefficients -- already in OpenCV order.
        left_D = np.array(
            _normalize_distortion(
                stereo_calibration.left.distortion_coefficients
            ),
            dtype=np.float64,
        )
        right_D = np.array(
            _normalize_distortion(
                stereo_calibration.right.distortion_coefficients
            ),
            dtype=np.float64,
        )

        # Convert Rodrigues rotation vector to 3x3 matrix.
        rvec = np.array(
            stereo_calibration.rotation_rvec,
            dtype=np.float64,
        ).reshape(3, 1)
        R, _ = cv2.Rodrigues(rvec)

        T = np.array(
            stereo_calibration.translation,
            dtype=np.float64,
        ).reshape(3, 1)

        image_size = self._resolve_image_size()

        R1, R2, P1, P2, _, _, _ = cv2.stereoRectify(
            left_K,
            left_D,
            right_K,
            right_D,
            image_size,
            R,
            T,
            flags=cv2.CALIB_ZERO_DISPARITY,
            alpha=0.0,
        )

        return R1, R2, P1, P2

    def _resolve_output_paths(
        self, camera_infos: dict[tuple[str, str], CameraInfoYaml]
    ) -> dict[tuple[str, str], Path]:
        output_dir = (
            Path(self.args.output_dir)
            if self.args.output_dir
            else Path.cwd() / DEFAULT_OUTPUT_DIR
        )
        return {
            target: output_dir / f'{info.camera_name}.yaml'
            for target, info in camera_infos.items()
        }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Read Eagle camera EEPROM calibration data '
                    'and output ROS 2 CameraInfo YAML files.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        '--ip-addr',
        default=DEFAULT_IP_ADDR,
        help=f'IP address of Hololink board (default: {DEFAULT_IP_ADDR})',
    )
    parser.add_argument(
        '--part',
        type=int,
        choices=[0, 1, 2],
        default=2,
        help=(
            'Which calibration data to read: 0=RGB only, 1=IR only, '
            '2=Both (default: 2)'
        ),
    )
    parser.add_argument(
        '--camera',
        choices=['left', 'right', 'both'],
        default='both',
        help='Which camera to export into ROS 2 CameraInfo YAML (default: both)',
    )
    parser.add_argument(
        '--output-dir',
        default=None,
        help=(
            'Directory to write output YAML files '
            f'(default: ./{DEFAULT_OUTPUT_DIR}).'
        ),
    )
    parser.add_argument(
        '--camera-name',
        default=None,
        help='Override camera_name in YAML (optional).',
    )
    parser.add_argument(
        '--timeout-s',
        type=float,
        default=10.0,
        help='Timeout for enumeration and connection (default: 10.0).',
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    try:
        workflow = EagleCalibrationWorkflow(args)
        workflow.run()
    except KeyboardInterrupt:
        print('\nInterrupted by user.', file=sys.stderr)
        sys.exit(130)
    except Exception as exc:
        print(f'Error: {exc}', file=sys.stderr)
        print('\nFull traceback:', file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
