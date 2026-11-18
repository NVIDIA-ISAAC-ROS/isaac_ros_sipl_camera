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


#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

#include "isaac_ros_sipl_camera/yuv_convert.hpp"

namespace isaac_ros
{
namespace sipl
{
namespace
{

constexpr uint32_t kWidth = 8U;
constexpr uint32_t kHeight = 4U;
// Deliberately padded so the converter cannot assume pitch == width * 2.
constexpr size_t kSrcPitch = kWidth * 2U + 16U;

bool cudaAvailable()
{
  int count = 0;
  return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}

/// Build a packed 4:2:2 image whose every component is uniquely identifiable.
/// Luma encodes its pixel position; chroma encodes its macro-pixel position.
std::vector<uint8_t> makeSource(PackedYuvOrder order)
{
  std::vector<uint8_t> src(kSrcPitch * kHeight, 0U);
  for (uint32_t y = 0U; y < kHeight; ++y) {
    for (uint32_t gx = 0U; gx < kWidth / 2U; ++gx) {
      const auto y0 = static_cast<uint8_t>(10U + y * kWidth + gx * 2U);
      const auto y1 = static_cast<uint8_t>(y0 + 1U);
      const auto u = static_cast<uint8_t>(100U + y * 4U + gx);
      const auto v = static_cast<uint8_t>(180U + y * 4U + gx);

      uint8_t * px = src.data() + y * kSrcPitch + gx * 4U;
      if (order == PackedYuvOrder::kUyvy) {
        px[0] = u; px[1] = y0; px[2] = v; px[3] = y1;
      } else {
        px[0] = y0; px[1] = u; px[2] = y1; px[3] = v;
      }
    }
  }
  return src;
}

std::vector<uint8_t> convert(const std::vector<uint8_t> & src, PackedYuvOrder order)
{
  const size_t dst_bytes = static_cast<size_t>(kWidth) * kHeight * 3U / 2U;

  uint8_t * d_src = nullptr;
  uint8_t * d_dst = nullptr;
  EXPECT_EQ(cudaMalloc(&d_src, src.size()), cudaSuccess);
  EXPECT_EQ(cudaMalloc(&d_dst, dst_bytes), cudaSuccess);
  EXPECT_EQ(
    cudaMemcpy(d_src, src.data(), src.size(), cudaMemcpyHostToDevice), cudaSuccess);

  cudaStream_t stream = nullptr;
  EXPECT_EQ(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), cudaSuccess);
  const NppStatus status = convertPackedYuv422ToNv12(
    d_src, kSrcPitch, d_dst, kWidth, kHeight, order, stream);
  EXPECT_EQ(status, NPP_SUCCESS);
  EXPECT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
  EXPECT_EQ(cudaStreamDestroy(stream), cudaSuccess);

  std::vector<uint8_t> dst(dst_bytes, 0U);
  EXPECT_EQ(
    cudaMemcpy(dst.data(), d_dst, dst_bytes, cudaMemcpyDeviceToHost), cudaSuccess);
  EXPECT_EQ(cudaFree(d_src), cudaSuccess);
  EXPECT_EQ(cudaFree(d_dst), cudaSuccess);
  return dst;
}

class YuvConvertTest : public ::testing::TestWithParam<PackedYuvOrder>
{
protected:
  void SetUp() override
  {
    if (!cudaAvailable()) {
      GTEST_SKIP() << "No CUDA device available";
    }
  }
};

TEST_P(YuvConvertTest, LumaIsPreservedExactly)
{
  const auto order = GetParam();
  const auto src = makeSource(order);
  const auto dst = convert(src, order);

  const size_t luma_offset = (order == PackedYuvOrder::kUyvy) ? 1U : 0U;
  for (uint32_t y = 0U; y < kHeight; ++y) {
    for (uint32_t x = 0U; x < kWidth; ++x) {
      // Macro-pixel gx holds pixels 2*gx and 2*gx+1 at +0 and +2 from the luma base.
      const uint8_t expected =
        src[y * kSrcPitch + (x / 2U) * 4U + luma_offset + (x % 2U) * 2U];
      EXPECT_EQ(dst[y * kWidth + x], expected)
        << "luma mismatch at (" << x << ", " << y << ") for order " << toString(order);
    }
  }
}

TEST_P(YuvConvertTest, ChromaIsDecimatedVerticallyAndInterleaved)
{
  const auto order = GetParam();
  const auto src = makeSource(order);
  const auto dst = convert(src, order);

  const size_t u_offset = (order == PackedYuvOrder::kUyvy) ? 0U : 1U;
  const size_t v_offset = (order == PackedYuvOrder::kUyvy) ? 2U : 3U;
  const uint8_t * uv = dst.data() + static_cast<size_t>(kWidth) * kHeight;

  // NPP samples chroma from even rows (measured on Thor with CUDA 13).
  for (uint32_t cy = 0U; cy < kHeight / 2U; ++cy) {
    for (uint32_t gx = 0U; gx < kWidth / 2U; ++gx) {
      const size_t top = (cy * 2U) * kSrcPitch + gx * 4U;
      const size_t bot = (cy * 2U + 1U) * kSrcPitch + gx * 4U;
      const int top_u = src[top + u_offset];
      const int bot_u = src[bot + u_offset];
      const int top_v = src[top + v_offset];
      const int bot_v = src[bot + v_offset];
      const int got_u = uv[cy * kWidth + gx * 2U];
      const int got_v = uv[cy * kWidth + gx * 2U + 1U];

      EXPECT_NEAR(got_u, top_u, 1)
        << "U at chroma (" << gx << ", " << cy << "): got " << got_u
        << ", top=" << top_u << " bot=" << bot_u
        << " avg=" << ((top_u + bot_u) / 2);
      EXPECT_NEAR(got_v, top_v, 1)
        << "V at chroma (" << gx << ", " << cy << "): got " << got_v
        << ", top=" << top_v << " bot=" << bot_v
        << " avg=" << ((top_v + bot_v) / 2);
    }
  }
}

INSTANTIATE_TEST_SUITE_P(
  PackedOrders, YuvConvertTest,
  ::testing::Values(PackedYuvOrder::kUyvy, PackedYuvOrder::kYuyv),
  [](const ::testing::TestParamInfo<PackedYuvOrder> & info) {
    return std::string(toString(info.param));
  });

TEST(YuvConvertColorFormatTest, MapsSupportedPackedFormats)
{
  EXPECT_EQ(packedYuvOrderFromColorFormat(NvSciColor_U8Y8V8Y8), PackedYuvOrder::kUyvy);
  EXPECT_EQ(packedYuvOrderFromColorFormat(NvSciColor_Y8U8Y8V8), PackedYuvOrder::kYuyv);
}

TEST(YuvConvertColorFormatTest, RejectsUnsupportedFormats)
{
  EXPECT_FALSE(packedYuvOrderFromColorFormat(NvSciColor_V8Y8U8Y8).has_value());
  EXPECT_FALSE(packedYuvOrderFromColorFormat(NvSciColor_Y8V8Y8U8).has_value());
  EXPECT_FALSE(packedYuvOrderFromColorFormat(NvSciColor_A8Y8U8V8).has_value());
  EXPECT_FALSE(packedYuvOrderFromColorFormat(NvSciColor_A16Y16U16V16).has_value());
  EXPECT_FALSE(packedYuvOrderFromColorFormat(NvSciColor_Y8).has_value());
  EXPECT_FALSE(packedYuvOrderFromColorFormat(NvSciColor_V8U8).has_value());
}

TEST(YuvConvertValidationTest, RejectsBadArguments)
{
  uint8_t dummy = 0U;
  EXPECT_EQ(
    convertPackedYuv422ToNv12(&dummy, 64U, &dummy, 7U, 4U, PackedYuvOrder::kUyvy, 0),
    NPP_SIZE_ERROR);
  EXPECT_EQ(
    convertPackedYuv422ToNv12(&dummy, 64U, &dummy, 8U, 3U, PackedYuvOrder::kUyvy, 0),
    NPP_SIZE_ERROR);
  EXPECT_EQ(
    convertPackedYuv422ToNv12(&dummy, 8U, &dummy, 8U, 4U, PackedYuvOrder::kUyvy, 0),
    NPP_STEP_ERROR);
  EXPECT_EQ(
    convertPackedYuv422ToNv12(nullptr, 64U, &dummy, 8U, 4U, PackedYuvOrder::kUyvy, 0),
    NPP_NULL_POINTER_ERROR);
}

}  // namespace
}  // namespace sipl
}  // namespace isaac_ros
