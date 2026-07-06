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

#include "isaac_ros_sipl_camera/sipl_buffer_manager.hpp"

#include <cuda.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <chrono>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace isaac_ros
{
namespace sipl
{

namespace
{
std::string CuResultToString(CUresult result)
{
  const char * name = "UNKNOWN";
  const char * desc = "unknown error";
  (void)cuGetErrorName(result, &name);
  (void)cuGetErrorString(result, &desc);
  std::ostringstream oss;
  oss << name << " (" << static_cast<int>(result) << "): " << desc;
  return oss.str();
}
}  // namespace

SiplBufferManager::SiplBufferManager(
  std::shared_ptr<NvSciBufModuleRec> sci_buf_module,
  uint32_t num_buffers,
  rclcpp::Logger logger,
  bool enable_debug_logs)
: sci_buf_module_(sci_buf_module), num_buffers_(num_buffers), logger_(logger),
  enable_debug_logs_(enable_debug_logs)
{
  if (sci_buf_module_ == nullptr) {
    throw std::runtime_error("SiplBufferManager constructor: sci_buf_module_ is nullptr");
  }
  buffers_.reserve(num_buffers_);
}

SiplBufferManager::~SiplBufferManager()
{
  // Clean up CUDA mappings
  for (auto & pair : cuda_mappings_) {
    if (pair.second.ptr) {
      cudaFree(pair.second.ptr);
    }
    if (pair.second.mem) {
      cudaDestroyExternalMemory(pair.second.mem);
    }
  }
  cuda_mappings_.clear();
  buffer_attr_cache_.clear();

  // Free all allocated NvSci buffer objects
  for (auto & buf_obj : buffers_) {
    if (buf_obj != nullptr) {
      NvSciBufObjFree(buf_obj);
    }
  }
  buffers_.clear();
}

nvsipl::SIPLStatus SiplBufferManager::allocateAndRegisterBuffers(
  nvsipl::INvSIPLCamera * camera,
  uint32_t sensor_id,
  nvsipl::INvSIPLClient::ConsumerDesc::OutputType output_type,
  NvSciBufSurfSampleType surf_sample_type)
{
  struct AttrListGuard
  {
    NvSciBufAttrList list{nullptr};
    ~AttrListGuard()
    {
      if (list != nullptr) {
        NvSciBufAttrListFree(list);
      }
    }
  };

  // Create buffer attribute list
  AttrListGuard attr_list{};
  NvSciError err = NvSciBufAttrListCreate(sci_buf_module_.get(), &attr_list.list);

  if (err != NvSciError_Success) {
    throw std::runtime_error("Failed to create NvSciBuf attribute list");
  }

  // Set consumer requirements
  constexpr NvSciBufType buf_type = NvSciBufType_Image;
  const NvSciBufAttrValAccessPerm access_perm = NvSciBufAccessPerm_Readonly;

  // Retrieve GPU UUID for CUDA interop
  CUresult res = cuInit(0);
  if (res != CUDA_SUCCESS) {
    std::ostringstream oss;
    oss << "Failed to initialize CUDA Driver API (cuInit): " << CuResultToString(res);
    throw std::runtime_error(oss.str());
  }

  CUdevice device{};
  res = cuDeviceGet(&device, 0);
  if (res != CUDA_SUCCESS) {
    std::ostringstream oss;
    oss << "Failed to get CUDA device 0 (cuDeviceGet): " << CuResultToString(res);
    throw std::runtime_error(oss.str());
  }

  CUuuid uuid;
  res = cuDeviceGetUuid(&uuid, device);
  if (res != CUDA_SUCCESS) {
    std::ostringstream oss;
    oss << "Failed to get CUDA device UUID (cuDeviceGetUuid): " << CuResultToString(res);
    throw std::runtime_error(oss.str());
  }
  NvSciRmGpuId gpu_id;
  std::memcpy(&gpu_id.bytes, uuid.bytes, sizeof(gpu_id.bytes));

  // Define attributes for ISP output buffers.
  // These specific attributes (PitchLinear, NV12, SemiPlanar) are critical for compatibility
  // We explicitly request these attributes to ensure the NvSciBuf buffer is allocated
  // with a format (NV12 Pitch Linear) that is compatible with both the SIPL ISP output
  // and the downstream CUDA/ROS consumers. Without these explicit requests, NvSciBuf
  // might allocate BlockLinear or other formats that are not supported by our CUDA mapping logic.

  // CPU mapping is not requested; buffers are consumed on GPU (CUDA) only.
  constexpr bool is_cpu_access_req = false;
  constexpr bool is_cpu_cache_enabled = false;

  constexpr NvSciBufAttrValImageLayoutType layout = NvSciBufImage_PitchLinearType;
  constexpr NvSciBufSurfType surf_type = NvSciSurfType_YUV;
  const NvSciBufSurfSampleType requested_sample_type = surf_sample_type;
  constexpr NvSciBufSurfBPC surf_bpc = NvSciSurfBPC_8;
  constexpr NvSciBufSurfMemLayout surf_mem_layout = NvSciSurfMemLayout_SemiPlanar;
  constexpr NvSciBufSurfComponentOrder surf_comp_order = NvSciSurfComponentOrder_YUV;
  constexpr NvSciBufAttrValColorStd surf_color_std[] = {NvSciColorStd_REC709_ER};

  NvSciBufAttrKeyValuePair attr_kvp[] = {
    // Common Attributes (Indices 0-2)
    {NvSciBufGeneralAttrKey_Types, &buf_type, sizeof(buf_type)},
    {NvSciBufGeneralAttrKey_RequiredPerm, &access_perm, sizeof(access_perm)},
    {NvSciBufGeneralAttrKey_GpuId, &gpu_id, sizeof(gpu_id)},

    // ISP image attributes (Indices 3-11)
    // Required for ISP output to ensure NV12 format
    {NvSciBufGeneralAttrKey_NeedCpuAccess, &is_cpu_access_req, sizeof(is_cpu_access_req)},
    {NvSciBufGeneralAttrKey_EnableCpuCache, &is_cpu_cache_enabled, sizeof(is_cpu_cache_enabled)},
    {NvSciBufImageAttrKey_Layout, &layout, sizeof(layout)},
    {NvSciBufImageAttrKey_SurfType, &surf_type, sizeof(surf_type)},
    {NvSciBufImageAttrKey_SurfBPC, &surf_bpc, sizeof(surf_bpc)},
    {NvSciBufImageAttrKey_SurfMemLayout, &surf_mem_layout, sizeof(surf_mem_layout)},
    {NvSciBufImageAttrKey_SurfSampleType, &requested_sample_type, sizeof(requested_sample_type)},
    {NvSciBufImageAttrKey_SurfComponentOrder, &surf_comp_order, sizeof(surf_comp_order)},
    {NvSciBufImageAttrKey_SurfColorStd, &surf_color_std, sizeof(surf_color_std)}
  };

  // Determine how many attributes to use based on output type. There's ICP or ISP{0,1,2}.
  // If ICP: Use only the first 3 attributes (Type, Perm, GPU ID). ICP buffers are
  // raw and don't need the specific ISP formats.
  // If ISP: Use all attributes. This ensures the buffer is allocated as NV12 Pitch
  // Linear, which is expected by the ISP and consumer.
  const size_t num_attrs = (output_type == nvsipl::INvSIPLClient::ConsumerDesc::OutputType::ICP) ?
    3U :
    sizeof(attr_kvp) / sizeof(attr_kvp[0]);

  err = NvSciBufAttrListSetAttrs(attr_list.list, attr_kvp, num_attrs);
  if (err != NvSciError_Success) {
    throw std::runtime_error("Failed to set NvSciBuf attributes");
  }

  // Fill with SIPL camera requirements
  nvsipl::SIPLStatus status = camera->GetImageAttributes(
    sensor_id, output_type, attr_list.list);
  if (status != nvsipl::NVSIPL_STATUS_OK) {
    throw std::runtime_error("Failed to get SIPL image attributes");
  }

  // Reconcile attribute lists
  AttrListGuard reconciled_attr_list{};
  AttrListGuard conflict_attr_list{};

  NvSciBufAttrList lists[] = {attr_list.list};
  err = NvSciBufAttrListReconcile(
    lists, 1U,
    &reconciled_attr_list.list,
    &conflict_attr_list.list);
  if (err != NvSciError_Success) {
    throw std::runtime_error("Failed to reconcile NvSciBuf attribute lists");
  }

  // Allocate buffer objects
  for (size_t i = 0U; i < num_buffers_; i++) {
    NvSciBufObj buf_obj{};
    err = NvSciBufObjAlloc(reconciled_attr_list.list, &buf_obj);
    if (err != NvSciError_Success) {
      throw std::runtime_error("Failed to allocate NvSciBuf object");
    }
    if (buf_obj == nullptr) {
      throw std::runtime_error("NvSciBuf object allocation returned null");
    }
    buffers_.push_back(buf_obj);
  }

  // Register buffers with SIPL camera
  status = camera->RegisterImages(sensor_id, output_type, buffers_);

  if (!buffers_.empty() && enable_debug_logs_) {
    logBufferAttributes(buffers_[0]);
  }

  if (status != nvsipl::NVSIPL_STATUS_OK) {
    RCLCPP_FATAL(get_logger(), "Failed to register images with SIPL camera (status: %d, type: %d)",
      static_cast<int>(status), static_cast<int>(output_type));
    throw std::runtime_error("Failed to register images with SIPL camera");
  }

  return nvsipl::NVSIPL_STATUS_OK;
}

nvsipl::SIPLStatus SiplBufferManager::getOrCreateCudaMapping(
  NvSciBufObj sci_buf_obj,
  size_t mapping_size,
  CudaDevicePtr * out_dev_ptr)
{
  if (out_dev_ptr == nullptr) {
    return nvsipl::NVSIPL_STATUS_ERROR;
  }

  // Check if mapping already exists
  auto it = cuda_mappings_.find(sci_buf_obj);
  if (it != cuda_mappings_.end()) {
    *out_dev_ptr = it->second.ptr;
    return nvsipl::NVSIPL_STATUS_OK;
  }

  // Define External Memory Handle Descriptor for NvSciBuf
  cudaExternalMemoryHandleDesc ext_mem_desc{};
  ext_mem_desc.type = cudaExternalMemoryHandleTypeNvSciBuf;
  ext_mem_desc.handle.nvSciBufObject = sci_buf_obj;
  ext_mem_desc.size = mapping_size;

  // Import NvSciBuf as CUDA External Memory
  cudaExternalMemory_t ext_mem{};
  cudaError_t cuda_err = cudaImportExternalMemory(&ext_mem, &ext_mem_desc);
  if (cuda_err != cudaSuccess) {
    return nvsipl::NVSIPL_STATUS_ERROR;
  }

  // Map the external memory to a device pointer
  CudaDevicePtr dev_ptr = nullptr;
  cudaExternalMemoryBufferDesc buf_desc{};
  buf_desc.offset = 0;
  buf_desc.size = mapping_size;
  buf_desc.flags = 0;
  cuda_err = cudaExternalMemoryGetMappedBuffer(&dev_ptr, ext_mem, &buf_desc);
  if (cuda_err != cudaSuccess) {
    cudaDestroyExternalMemory(ext_mem);
    return nvsipl::NVSIPL_STATUS_ERROR;
  }

  // Cache the mapping
  cuda_mappings_[sci_buf_obj] = CudaBufferMapping{ext_mem, dev_ptr};
  *out_dev_ptr = dev_ptr;
  return nvsipl::NVSIPL_STATUS_OK;
}

nvsipl::SIPLStatus SiplBufferManager::getBufferAttributes(
  NvSciBufObj sci_buf_obj,
  BufferAttributes & attrs)
{
  // Check cache first
  auto it = buffer_attr_cache_.find(sci_buf_obj);
  if (it != buffer_attr_cache_.end()) {
    attrs = it->second;
    return nvsipl::NVSIPL_STATUS_OK;
  }

  NvSciBufAttrList buf_attr_list;

  // Get all buffer attributes needed for analysis
  NvSciBufAttrKeyValuePair img_attrs[] = {
    {NvSciBufImageAttrKey_Size, NULL, 0},                   // 0
    {NvSciBufImageAttrKey_PlaneCount, NULL, 0},             // 1
    {NvSciBufImageAttrKey_PlanePitch, NULL, 0},             // 2
    {NvSciBufImageAttrKey_PlaneWidth, NULL, 0},             // 3
    {NvSciBufImageAttrKey_PlaneHeight, NULL, 0},            // 4
    {NvSciBufImageAttrKey_PlaneBitsPerPixel, NULL, 0},      // 5
    {NvSciBufImageAttrKey_PlaneOffset, NULL, 0},            // 6
    {NvSciBufImageAttrKey_PlaneColorFormat, NULL, 0},       // 7
  };

  NvSciError err = NvSciBufObjGetAttrList(sci_buf_obj, &buf_attr_list);
  if (err != NvSciError_Success) {
    return nvsipl::NVSIPL_STATUS_ERROR;
  }

  err = NvSciBufAttrListGetAttrs(
    buf_attr_list, img_attrs,
    sizeof(img_attrs) / sizeof(img_attrs[0]));
  if (err != NvSciError_Success) {
    return nvsipl::NVSIPL_STATUS_ERROR;
  }

  // Extract buffer attributes
  if (img_attrs[0].len != 0) {
    attrs.size = *(static_cast<const uint64_t *>(img_attrs[0].value));
  }

  if (img_attrs[1].len != 0) {
    attrs.plane_count = *(static_cast<const uint32_t *>(img_attrs[1].value));
  }

  uint32_t max_planes = (attrs.plane_count > BufferAttributes::MAX_NUM_SURFACES) ?
    BufferAttributes::MAX_NUM_SURFACES : attrs.plane_count;

  // Plane pitches
  if (img_attrs[2].len != 0) {
    memcpy(
      attrs.plane_pitches.data(),
      static_cast<const uint32_t *>(img_attrs[2].value),
      max_planes * sizeof(attrs.plane_pitches[0]));
  }

  // Plane dimensions
  if (img_attrs[3].len != 0) {
    memcpy(
      attrs.plane_widths.data(),
      static_cast<const uint32_t *>(img_attrs[3].value),
      max_planes * sizeof(attrs.plane_widths[0]));
  }

  if (img_attrs[4].len != 0) {
    memcpy(
      attrs.plane_heights.data(),
      static_cast<const uint32_t *>(img_attrs[4].value),
      max_planes * sizeof(attrs.plane_heights[0]));
  }

  // Bits per pixel
  if (img_attrs[5].len != 0) {
    memcpy(
      attrs.plane_bits_per_pixels.data(),
      static_cast<const uint32_t *>(img_attrs[5].value),
      max_planes * sizeof(attrs.plane_bits_per_pixels[0]));
  }

  // Plane offsets
  if (img_attrs[6].len != 0) {
    memcpy(
      attrs.plane_offsets.data(),
      static_cast<const uint64_t *>(img_attrs[6].value),
      max_planes * sizeof(attrs.plane_offsets[0]));
  }

  // Color formats
  if (img_attrs[7].len != 0) {
    memcpy(
      attrs.plane_color_formats.data(),
      static_cast<const NvSciBufAttrValColorFmt *>(img_attrs[7].value),
      max_planes * sizeof(attrs.plane_color_formats[0]));
  }

  // Cache attributes
  buffer_attr_cache_[sci_buf_obj] = attrs;

  return nvsipl::NVSIPL_STATUS_OK;
}

nvsipl::SIPLStatus SiplBufferManager::queryAllocatedBufferAttributes(BufferAttributes & attrs)
{
  if (buffers_.empty()) {
    return nvsipl::NVSIPL_STATUS_ERROR;
  }
  return getBufferAttributes(buffers_[0], attrs);
}

nvsipl::SIPLStatus SiplBufferManager::mapNvmmToCuda(
  nvsipl::INvSIPLClient::INvSIPLNvMBuffer * sipl_buffer,
  NvSciSyncFence & fence,
  NvSciSyncCpuWaitContext cpu_wait_context,
  CudaDevicePtr * out_cuda_buffer,
  size_t * out_cuda_buffer_size,
  BufferAttributes * out_attrs)
{
  if (sipl_buffer == nullptr || out_cuda_buffer == nullptr || out_cuda_buffer_size == nullptr) {
    return nvsipl::NVSIPL_STATUS_ERROR;
  }
  *out_cuda_buffer = nullptr;
  *out_cuda_buffer_size = 0U;

  // Wait on the EOF fence to ensure the ISP hardware has finished all DMA writes
  // to this buffer. The fence is signaled by the ISP engine after the frame is
  // fully committed to memory. Without this wait, reads would see incomplete data.
  constexpr uint64_t kFenceTimeoutUs = 200000;  // 200ms
  auto fence_wait_start = std::chrono::steady_clock::now();
  NvSciError sci_err = NvSciSyncFenceWait(&fence, cpu_wait_context, kFenceTimeoutUs);
  if (sci_err != NvSciError_Success) {
    NvSciSyncFenceClear(&fence);
    return nvsipl::NVSIPL_STATUS_ERROR;
  }
  NvSciSyncFenceClear(&fence);

  if (enable_debug_logs_) {
    auto fence_wait_end = std::chrono::steady_clock::now();
    auto fence_wait_us = std::chrono::duration_cast<std::chrono::microseconds>(
      fence_wait_end - fence_wait_start).count();
    RCLCPP_DEBUG(get_logger(),
      "[Buffer manager] NvSciSyncFenceWait latency: %ld us", fence_wait_us);
  }

  // Get NvSciBuf object from SIPL buffer
  NvSciBufObj nvmm_obj = sipl_buffer->GetNvSciBufImage();
  if (nvmm_obj == nullptr) {
    return nvsipl::NVSIPL_STATUS_ERROR;
  }

  // Get buffer attributes (cached after first call per buffer object)
  BufferAttributes attrs;
  nvsipl::SIPLStatus status = getBufferAttributes(nvmm_obj, attrs);
  if (status != nvsipl::NVSIPL_STATUS_OK) {
    return nvsipl::NVSIPL_STATUS_ERROR;
  }

  // Copy attributes to output if requested
  if (out_attrs != nullptr) {
    *out_attrs = attrs;
  }

  const size_t src_size = attrs.size;
  if (src_size == 0U) {
    RCLCPP_ERROR(get_logger(), "Buffer attribute reports size is 0");
    return nvsipl::NVSIPL_STATUS_ERROR;
  }
  RCLCPP_DEBUG(get_logger(), "[Buffer Manager] Mapping NVMM to CUDA. Size: %lu", src_size);

  CudaDevicePtr src_dev_ptr = nullptr;
  status = getOrCreateCudaMapping(nvmm_obj, src_size, &src_dev_ptr);
  if (status != nvsipl::NVSIPL_STATUS_OK || src_dev_ptr == nullptr) {
    return nvsipl::NVSIPL_STATUS_ERROR;
  }

  // Zero-copy mapping: return the CUDA pointer that aliases the underlying NvSciBuf allocation.
  *out_cuda_buffer = src_dev_ptr;
  *out_cuda_buffer_size = src_size;

  return nvsipl::NVSIPL_STATUS_OK;
}


void SiplBufferManager::logBufferAttributes(NvSciBufObj buf_obj)
{
  std::stringstream ss;
  ss << "\nBuffer Attributes for first buffer:";
  NvSciBufAttrList buf_attr_list;
  NvSciError sci_err = NvSciBufObjGetAttrList(buf_obj, &buf_attr_list);
  if (sci_err != NvSciError_Success) {
    ss << "\n  Failed to get attr list from obj: " << sci_err;
    RCLCPP_INFO(get_logger(), "%s", ss.str().c_str());
    return;
  }
  BufferAttributes attrs;
  nvsipl::SIPLStatus status = getBufferAttributes(buf_obj, attrs);
  if (status != nvsipl::NVSIPL_STATUS_OK) {
    ss << "\n  Failed to get attributes from obj: " << status;
    RCLCPP_INFO(get_logger(), "%s", ss.str().c_str());
    return;
  }
  ss << "\n  Size: " << attrs.size;
  ss << "\n  Plane Count: " << attrs.plane_count;

  const uint32_t num_planes_to_log = std::min(attrs.plane_count,
    static_cast<uint32_t>(BufferAttributes::MAX_NUM_SURFACES));
  for (uint32_t i = 0; i < num_planes_to_log; ++i) {
    ss << "\n  Plane " << i << ":"
       << " pitch=" << attrs.plane_pitches[i]
       << " width=" << attrs.plane_widths[i]
       << " height=" << attrs.plane_heights[i]
       << " offset=" << attrs.plane_offsets[i]
       << " bpp=" << attrs.plane_bits_per_pixels[i]
       << " color_fmt=" << attrs.plane_color_formats[i];
  }
  RCLCPP_INFO(get_logger(), "%s", ss.str().c_str());
}

}  // namespace sipl
}  // namespace isaac_ros
