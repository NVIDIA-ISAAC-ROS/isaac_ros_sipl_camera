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

#include "isaac_ros_sipl_camera/yuv_convert.hpp"

#include <nppi_color_conversion.h>

#include <limits>
#include <mutex>

namespace isaac_ros
{
namespace sipl
{

namespace
{

// Cache device properties; each call supplies its own stream and stream flags.
const NppStreamContext * deviceStreamContext()
{
  static NppStreamContext ctx{};
  static bool valid = false;
  static std::once_flag once;

  std::call_once(once, []() {
      if (cudaGetDevice(&ctx.nCudaDeviceId) != cudaSuccess) {
        return;
      }
      cudaDeviceProp props{};
      if (cudaGetDeviceProperties(&props, ctx.nCudaDeviceId) != cudaSuccess) {
        return;
      }
      ctx.nMultiProcessorCount = props.multiProcessorCount;
      ctx.nMaxThreadsPerMultiProcessor = props.maxThreadsPerMultiProcessor;
      ctx.nMaxThreadsPerBlock = props.maxThreadsPerBlock;
      ctx.nSharedMemPerBlock = props.sharedMemPerBlock;
      ctx.nCudaDevAttrComputeCapabilityMajor = props.major;
      ctx.nCudaDevAttrComputeCapabilityMinor = props.minor;
      valid = true;
    });

  return valid ? &ctx : nullptr;
}

}  // namespace

std::optional<PackedYuvOrder> packedYuvOrderFromColorFormat(NvSciBufAttrValColorFmt format)
{
  switch (format) {
    case NvSciColor_U8Y8V8Y8:
      return PackedYuvOrder::kUyvy;
    case NvSciColor_Y8U8Y8V8:
      return PackedYuvOrder::kYuyv;
    default:
      return std::nullopt;
  }
}

const char * toString(PackedYuvOrder order)
{
  switch (order) {
    case PackedYuvOrder::kUyvy:
      return "uyvy";
    case PackedYuvOrder::kYuyv:
      return "yuyv";
  }
  return "unknown";
}

NppStatus convertPackedYuv422ToNv12(
  const uint8_t * src, size_t src_pitch,
  uint8_t * dst, uint32_t width, uint32_t height,
  PackedYuvOrder order, cudaStream_t stream)
{
  if (src == nullptr || dst == nullptr) {
    return NPP_NULL_POINTER_ERROR;
  }
  if (width == 0U || height == 0U || (width % 2U) != 0U || (height % 2U) != 0U) {
    return NPP_SIZE_ERROR;
  }
  if (src_pitch < static_cast<size_t>(width) * 2U ||
    src_pitch > static_cast<size_t>(std::numeric_limits<int>::max()))
  {
    return NPP_STEP_ERROR;
  }

  const NppStreamContext * device_ctx = deviceStreamContext();
  if (device_ctx == nullptr) {
    return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  }
  NppStreamContext ctx = *device_ctx;
  ctx.hStream = stream;
  if (cudaStreamGetFlags(stream, &ctx.nStreamFlags) != cudaSuccess) {
    return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  }

  const NppiSize roi{static_cast<int>(width), static_cast<int>(height)};
  const int src_step = static_cast<int>(src_pitch);
  const int dst_step = static_cast<int>(width);
  Npp8u * dst_y = reinterpret_cast<Npp8u *>(dst);
  Npp8u * dst_cbcr = dst_y + static_cast<size_t>(width) * height;

  switch (order) {
    case PackedYuvOrder::kUyvy:
      return nppiCbYCr422ToYCbCr420_8u_C2P2R_Ctx(
        reinterpret_cast<const Npp8u *>(src), src_step,
        dst_y, dst_step, dst_cbcr, dst_step, roi, ctx);
    case PackedYuvOrder::kYuyv:
      return nppiYCbCr422ToYCbCr420_8u_C2P2R_Ctx(
        reinterpret_cast<const Npp8u *>(src), src_step,
        dst_y, dst_step, dst_cbcr, dst_step, roi, ctx);
  }
  return NPP_NOT_SUPPORTED_MODE_ERROR;
}

}  // namespace sipl
}  // namespace isaac_ros
