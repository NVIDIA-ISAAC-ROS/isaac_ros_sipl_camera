// SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// SPDX-License-Identifier: Apache-2.0

#ifndef ISAAC_ROS_SIPL_CAMERA__YUV_CONVERT_HPP_
#define ISAAC_ROS_SIPL_CAMERA__YUV_CONVERT_HPP_

#include <cuda_runtime.h>
#include <nppdefs.h>

#include <cstddef>
#include <cstdint>
#include <optional>

#include "isaac_ros_sipl_camera/sipl_definitions.hpp"

namespace isaac_ros
{
namespace sipl
{

/// Byte order within a packed 4:2:2 macro-pixel, as reported by an ICP buffer's
/// NvSciBufImageAttrKey_PlaneColorFormat.
enum class PackedYuvOrder
{
  kUyvy,   ///< NvSciColor_U8Y8V8Y8 — U Y V Y
  kYuyv,   ///< NvSciColor_Y8U8Y8V8 — Y U Y V
};

std::optional<PackedYuvOrder> packedYuvOrderFromColorFormat(NvSciBufAttrValColorFmt format);

const char * toString(PackedYuvOrder order);

/// Convert packed 4:2:2 to contiguous NV12 on the supplied CUDA stream.
/// The destination needs width * height * 3 / 2 bytes, with luma/chroma pitch = width.
/// Chroma uses the even source row; the source color standard is preserved.
/// Requires even width/height and src_pitch >= width * 2. Completion is asynchronous.
NppStatus convertPackedYuv422ToNv12(
  const uint8_t * src, size_t src_pitch,
  uint8_t * dst, uint32_t width, uint32_t height,
  PackedYuvOrder order, cudaStream_t stream);

}  // namespace sipl
}  // namespace isaac_ros

#endif  // ISAAC_ROS_SIPL_CAMERA__YUV_CONVERT_HPP_
