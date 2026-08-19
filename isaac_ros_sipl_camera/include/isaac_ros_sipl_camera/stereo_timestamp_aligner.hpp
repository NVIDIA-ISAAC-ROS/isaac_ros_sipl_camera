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

#ifndef ISAAC_ROS_SIPL_CAMERA__STEREO_TIMESTAMP_ALIGNER_HPP_
#define ISAAC_ROS_SIPL_CAMERA__STEREO_TIMESTAMP_ALIGNER_HPP_

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <stdexcept>

namespace isaac_ros
{
namespace sipl
{

/**
 * @brief Non-blocking stereo timestamp aligner; matches partners by capture-time proximity.
 *
 * Hardware-synced stereo sensors share a nearly identical capture instant, but each
 * host-stamped timestamp differs slightly (sub-millisecond) and the two frames arrive on
 * independent threads in any order, possibly skewed by up to ~one frame period when the two
 * pipelines have asymmetric latency. This utility makes a matched pair publish one identical
 * timestamp without either stream waiting for the other.
 *
 * Each arriving frame is matched against a bounded buffer of unmatched first-arrivers from the
 * partner sensor, by capture-time proximity and arrival order, so a partner that arrives a full
 * frame late is still matched to the correct capture:
 *   - diff <= max_diff_ns                  : in-sync pair; the lagging frame adopts the
 *                                            first-arriver's stamp so the pair is bit-identical.
 *   - max_diff_ns < diff <= half_period_ns : same capture but skewed too far to claim identical;
 *                                            the frame keeps its own stamp and is reported through
 *                                            Result::desync.
 *   - no match within half_period_ns       : the frame is itself a first-arriver (its partner has
 *                                            not arrived yet, or was dropped) and is buffered.
 *
 * Working in the converted (ROS-time ns) domain lets the lagging frame adopt the leader's already
 * converted stamp, so the matched pair stays identical even across TSC->ROS recalibration.
 *
 * "Non-blocking" means no eye waits for its partner: a first-arriver publishes immediately with its
 * own stamp. The mutex below only serializes the small bookkeeping buffer for a few nanoseconds; it
 * never waits for a frame.
 *
 * Matching is disambiguated from half_period_ns strictly less than the frame period: two
 * distinct captures differ by more than half_period_ns, so an arriving frame matches at most one
 * pending capture (the closest).
 */
class StereoTimestampAligner
{
public:
  struct Result
  {
    int64_t ns;     // Nanoseconds the caller should stamp (adopted, or its own).
    bool desync;    // True only for a same-capture partner skewed beyond threshold.
  };

  /// @param max_diff_ns    A pair within this window adopts a single stamp and is considered
  ///                       in-sync.
  /// @param half_period_ns Upper bound (ns) of the "same capture but skewed" band
  ///        (should be set to half the frame interval). Diffs above this are assumed to be
  ///        a new capture.
  /// @param capacity       Max unmatched timestamps buffered to sustain one-sided drops and the
  ///         arrival skew (in number of frames) that can still be matched. Must be in
  ///         [1, kMaxCapacity] or will throw.
  StereoTimestampAligner(
    int64_t max_diff_ns, int64_t half_period_ns, std::size_t capacity = kMaxCapacity)
  : max_diff_ns_(max_diff_ns),
    half_period_ns_(half_period_ns),
    capacity_(capacity)
  {
    if (capacity == 0 || capacity > kMaxCapacity) {
      throw std::invalid_argument(
        "StereoTimestampAligner: capacity must be in [1, " + std::to_string(kMaxCapacity) + "]");
    }
  }

  /**
  * @param capture_timestamp Converted-time-domain (ROS-time ns) capture timestamp of the frame.
  *        Returns the timestamp the caller frame should publish. A matched lagging frame
  *        adopts its partner's stamp (identical pair) and an unmatched or too-skewed frame keeps its
  *        own stamp.
  */
  Result align(int64_t capture_timestamp)
  {
    const std::lock_guard<std::mutex> lock(mutex_);

    // Find closest pending previously seen timestamp. At most one can lie within half_period_ns,
    // so the closest is the only possible partner.
    const auto abs_diff_to = [capture_timestamp](int64_t ns) {
        return capture_timestamp > ns ? capture_timestamp - ns : ns - capture_timestamp;
      };
    const auto first = pending_.begin();
    const auto last = first + count_;
    const auto best = std::min_element(
      first, last,
      [&abs_diff_to](int64_t a, int64_t b) {return abs_diff_to(a) < abs_diff_to(b);});

    if (best != last && abs_diff_to(*best) <= half_period_ns_) {
      const int64_t partner_ns = *best;
      erase(static_cast<std::size_t>(best - first));  // partner resolved; remove from buffer
      if (abs_diff_to(partner_ns) <= max_diff_ns_) {
        return {partner_ns, false};      // in-sync: adopt the first-arriver's stamp
      }
      return {capture_timestamp, true};  // same capture, skewed beyond threshold
    }

    push(capture_timestamp);             // first-arriver: buffer it for its partner
    return {capture_timestamp, false};
  }

private:
  static constexpr std::size_t kMaxCapacity = 8;

  // pending_ holds unmatched first-arrivers oldest-first; capacity is tiny so O(n) shifts are fine.
  void erase(std::size_t i)
  {
    for (std::size_t j = i + 1; j < count_; ++j) {
      pending_[j - 1] = pending_[j];
    }
    --count_;
  }

  void push(int64_t ns)
  {
    if (count_ == capacity_) {
      erase(0);                          // drop oldest orphan (its partner never arrived)
    }
    pending_[count_++] = ns;
  }

  const int64_t max_diff_ns_;
  const int64_t half_period_ns_;
  const std::size_t capacity_;

  std::mutex mutex_;
  std::array<int64_t, kMaxCapacity> pending_{};
  std::size_t count_{0};
};

}  // namespace sipl
}  // namespace isaac_ros

#endif  // ISAAC_ROS_SIPL_CAMERA__STEREO_TIMESTAMP_ALIGNER_HPP_
