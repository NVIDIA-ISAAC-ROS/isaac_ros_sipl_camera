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

#ifndef ISAAC_ROS_SIPL_CAMERA__SIPL_BUFFER_MANAGER_HPP_
#define ISAAC_ROS_SIPL_CAMERA__SIPL_BUFFER_MANAGER_HPP_

#include <array>
#include <memory>
#include <unordered_map>
#include <vector>

#include "rclcpp/rclcpp.hpp"
#include "isaac_ros_sipl_camera/sipl_definitions.hpp"

namespace isaac_ros
{
namespace sipl
{

/// Opaque CUDA device pointer (void* required by CUDA/NvSci interop APIs)
using CudaDevicePtr = void *;  // NOSONAR

// Buffer attribute structure for analyzing NvSciBufObj
struct BufferAttributes
{
  static constexpr uint32_t MAX_NUM_SURFACES = 3;

  uint64_t size;
  uint32_t plane_count;
  std::array<uint32_t, MAX_NUM_SURFACES> plane_widths{};
  std::array<uint32_t, MAX_NUM_SURFACES> plane_heights{};
  std::array<uint32_t, MAX_NUM_SURFACES> plane_pitches{};
  std::array<uint32_t, MAX_NUM_SURFACES> plane_bits_per_pixels{};
  std::array<uint64_t, MAX_NUM_SURFACES> plane_offsets{};
  std::array<NvSciBufAttrValColorFmt, MAX_NUM_SURFACES> plane_color_formats{};
};

/**
 * @brief Manages NvSci buffer allocation and NVMM to CUDA memory operations
 *
 * This class handles:
 * - NvSciBuf allocation for SIPL camera registration
 * - Buffer attribute extraction and analysis
 * - GPU-to-GPU memory copies from NVMM (SIPL) to CUDA buffers
 */
class SiplBufferManager
{
public:
  explicit SiplBufferManager(
    std::shared_ptr<NvSciBufModuleRec> sci_buf_module,
    uint32_t num_buffers,
    rclcpp::Logger logger,
    bool enable_debug_logs = false);
  ~SiplBufferManager();

  /**
   * @brief Allocate and register NvSci buffers with SIPL camera
   *
   * @param camera SIPL camera instance
   * @param sensor_id Sensor ID from CameraSystemConfig
   * @param output_type Output type (ISP0, ISP1, ISP2, or ICP)
   * @param enable_cpu_access If true, enables CPU access and ReadWrite permissions (needed for VPI/OpenCV)
   * @param surf_sample_type Surface sample type (e.g., 420 for NV12, 444 for NV24). Ignored for ICP.
   * @return nvsipl::SIPLStatus Success or error status
   */
  nvsipl::SIPLStatus allocateAndRegisterBuffers(
    nvsipl::INvSIPLCamera * camera,
    uint32_t sensor_id,
    nvsipl::INvSIPLClient::ConsumerDesc::OutputType output_type,
    bool enable_cpu_access = false,
    NvSciBufSurfSampleType surf_sample_type = NvSciSurfSampleType_420);

  /**
   * @brief Get buffer attributes from NvSciBufObj
   *
   * @param sci_buf_obj NvSci buffer object
   * @param attrs Output buffer attributes structure
   * @return nvsipl::SIPLStatus Success or error status
   */
  nvsipl::SIPLStatus getBufferAttributes(
    NvSciBufObj sci_buf_obj,
    BufferAttributes & attrs);

  /**
   * @brief Map SIPL NvM buffer to a CUDA device pointer using CUDA-NvSciBuf interop (zero-copy)
   *
   * This imports the underlying NvSciBufObj as CUDA external memory and returns the mapped
   * device pointer. No device-to-device copy is performed.
   *
   * Waits on the EOF fence via NvSciSyncFenceWait() to ensure the ISP hardware has
   * finished all DMA writes before the caller reads pixel data.
   *
   * IMPORTANT: The returned pointer aliases the SIPL-owned buffer. The SIPL buffer must NOT
   * be released back to SIPL until all downstream GPU consumers are done.
   *
   * @param sipl_buffer SIPL NvM buffer wrapper containing NvSciBuf image
   * @param fence EOF fence to wait on
   * @param cpu_wait_context NvSci CPU wait context
   * @param out_cuda_buffer Mapped CUDA device pointer
   * @param out_cuda_buffer_size Size of the NvSciBuf allocation in bytes (best-effort)
   * @param out_attrs Optional output for buffer plane attributes
   * @return nvsipl::SIPLStatus Success or error status
   */
  nvsipl::SIPLStatus mapNvmmToCuda(
    nvsipl::INvSIPLClient::INvSIPLNvMBuffer * sipl_buffer,
    NvSciSyncFence & fence,
    NvSciSyncCpuWaitContext cpu_wait_context,
    CudaDevicePtr * out_cuda_buffer,
    size_t * out_cuda_buffer_size,
    BufferAttributes * out_attrs);

  /**
   * @brief Get the logger
   *
   * @return rclcpp::Logger
   */
  rclcpp::Logger get_logger() const {return logger_;}

private:
  nvsipl::SIPLStatus getOrCreateCudaMapping(
    NvSciBufObj sci_buf_obj,
    size_t mapping_size,
    CudaDevicePtr * out_dev_ptr);

  std::shared_ptr<NvSciBufModuleRec> sci_buf_module_;
  uint32_t num_buffers_;
  rclcpp::Logger logger_;
  std::vector<NvSciBufObj> buffers_;
  bool enable_debug_logs_;

  struct CudaBufferMapping
  {
    cudaExternalMemory_t mem;
    CudaDevicePtr ptr;
  };
  std::unordered_map<NvSciBufObj, CudaBufferMapping> cuda_mappings_;
  std::unordered_map<NvSciBufObj, BufferAttributes> buffer_attr_cache_;

  void logBufferAttributes(NvSciBufObj buf_obj);
};

}  // namespace sipl
}  // namespace isaac_ros

#endif  // ISAAC_ROS_SIPL_CAMERA__SIPL_BUFFER_MANAGER_HPP_
