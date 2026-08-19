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

#include "isaac_ros_sipl_camera/stereo_timestamp_aligner.hpp"

#include <gtest/gtest.h>

#include <cstdint>

using isaac_ros::sipl::StereoTimestampAligner;

namespace
{
// Self-consistent ROS-time nanosecond values: threshold < half_period < frame period.
constexpr int64_t kThresholdNs = 500'000;                    // 500 us in-sync window
constexpr int64_t kHalfPeriodNs = 16'666'667;                // ~half of a 30 fps frame
constexpr int64_t kPeriodNs = 33'333'333;                    // 30 fps frame period
constexpr int64_t kBaseNs = 1'700'000'000'000'000'000LL;     // realistic ROS-time epoch ns
}  // namespace

// Whichever frame is processed first sets the reference; the partner adopts it, so
// both publish the first-arriver's stamp independent of arrival order.
TEST(StereoTimestampAligner, InSyncPairAdoptsFirstArriver)
{
  {
    StereoTimestampAligner s(kThresholdNs, kHalfPeriodNs);
    auto a = s.align(kBaseNs);              // left first
    auto b = s.align(kBaseNs + 10'000);     // right, 10 us later
    EXPECT_EQ(a.ns, kBaseNs);
    EXPECT_EQ(b.ns, kBaseNs);                 // adopted
    EXPECT_FALSE(a.desync);
    EXPECT_FALSE(b.desync);
  }
  {
    StereoTimestampAligner s(kThresholdNs, kHalfPeriodNs);
    auto b = s.align(kBaseNs + 10'000);     // right first
    auto a = s.align(kBaseNs);              // left second
    EXPECT_EQ(b.ns, kBaseNs + 10'000);
    EXPECT_EQ(a.ns, kBaseNs + 10'000);        // adopted
  }
}

// A stale reference from the previous pairing (~one period away) must not match: the
// new pair's first-arriver keeps its own stamp, then its partner adopts that.
TEST(StereoTimestampAligner, StaleReferenceNotMatched)
{
  {
    StereoTimestampAligner s(kThresholdNs, kHalfPeriodNs);
    auto a = s.align(kBaseNs);                               // pair 0
    auto b = s.align(kBaseNs + 10'000);
    EXPECT_EQ(a.ns, b.ns);                                   // adopted
    a = s.align(kBaseNs + kPeriodNs);                        // pair 1, one period later
    b = s.align(kBaseNs + kPeriodNs + 10'000);
    EXPECT_EQ(a.ns, kBaseNs + kPeriodNs);                    // own (stale ref rejected)
    EXPECT_EQ(b.ns, kBaseNs + kPeriodNs);                    // adopts pair-1 first-arriver
    EXPECT_EQ(a.ns, b.ns);                                   // adopted
  }
  {
    StereoTimestampAligner s(kThresholdNs, kHalfPeriodNs);
    auto a = s.align(kBaseNs);
    // No second frame. Dropped edge case.
    a = s.align(kBaseNs + kPeriodNs);                        // pair 1, one period later
    auto b = s.align(kBaseNs + kPeriodNs + 10'000);
    EXPECT_EQ(a.ns, kBaseNs + kPeriodNs);                    // own (stale ref rejected)
    EXPECT_EQ(b.ns, kBaseNs + kPeriodNs);                    // adopts pair-1 first-arriver
    EXPECT_EQ(a.ns, b.ns);                                   // adopted
  }
}

// Exactly at threshold adopts; one ns beyond does not (and is flagged desync).
TEST(StereoTimestampAligner, ThresholdBoundary)
{
  {
    StereoTimestampAligner s(kThresholdNs, kHalfPeriodNs);
    s.align(kBaseNs);
    auto b = s.align(kBaseNs + kThresholdNs);      // diff == threshold
    EXPECT_EQ(b.ns, kBaseNs);                          // adopted
    EXPECT_FALSE(b.desync);
  }
  {
    StereoTimestampAligner s(kThresholdNs, kHalfPeriodNs);
    s.align(kBaseNs);
    auto b = s.align(kBaseNs + kThresholdNs + 1);  // diff == threshold + 1
    EXPECT_EQ(b.ns, kBaseNs + kThresholdNs + 1);      // own
    EXPECT_TRUE(b.desync);                             // within half-period -> flagged
  }
}

// A skew in (threshold, half_period] is a same-capture desync; a larger gap is a
// different capture (stale reference / dropped frame) and is not flagged.
TEST(StereoTimestampAligner, DesyncBandVsCrossCapture)
{
  StereoTimestampAligner s(kThresholdNs, kHalfPeriodNs);
  s.align(kBaseNs);
  auto skewed = s.align(kBaseNs + 5'000'000);    // 5 ms, in (500 us, 16.67 ms]
  EXPECT_EQ(skewed.ns, kBaseNs + 5'000'000);
  EXPECT_TRUE(skewed.desync);

  auto far = s.align(kBaseNs + 100'000'000);     // 100 ms, well beyond half-period
  EXPECT_EQ(far.ns, kBaseNs + 100'000'000);
  EXPECT_FALSE(far.desync);
}

// Out-of-order arrival: one eye lags the other by a full frame (e.g. asymmetric pipeline latency).
// Partners are matched by capture-time proximity, not arrival order, so each capture's pair still
// resolves to one identical stamp.
TEST(StereoTimestampAligner, OutOfOrderArrivalMatchesByCaptureProximity)
{
  StereoTimestampAligner s(kThresholdNs, kHalfPeriodNs);
  const int64_t r0 = kBaseNs;
  const int64_t r1 = kBaseNs + kPeriodNs;
  const int64_t r2 = kBaseNs + 2 * kPeriodNs;
  const int64_t l0 = kBaseNs + 9'000;                  // capture 0, +9 us FSYNC skew
  const int64_t l1 = kBaseNs + kPeriodNs + 9'000;      // capture 1

  // Arrival interleaves one frame ahead on the right: R0, R1, L0, R2, L1.
  auto a_r0 = s.align(r0);   // first-arriver
  auto a_r1 = s.align(r1);   // first-arriver (next capture; r0 still pending)
  auto a_l0 = s.align(l0);   // matches r0 across the one-frame gap, not r1
  auto a_r2 = s.align(r2);   // first-arriver
  auto a_l1 = s.align(l1);   // matches r1

  EXPECT_EQ(a_r0.ns, r0);
  EXPECT_EQ(a_l0.ns, r0);    // capture-0 pair identical despite interleaving
  EXPECT_FALSE(a_l0.desync);
  EXPECT_EQ(a_r1.ns, r1);
  EXPECT_EQ(a_l1.ns, r1);    // capture-1 pair identical
  EXPECT_FALSE(a_l1.desync);
}

// A dropped partner leaves an orphan that must never falsely match a later capture's frame.
TEST(StereoTimestampAligner, DroppedPartnerNeverFalselyMatches)
{
  StereoTimestampAligner s(kThresholdNs, kHalfPeriodNs);
  auto a_r0 = s.align(kBaseNs);                          // R0; its partner L0 is dropped
  auto a_r1 = s.align(kBaseNs + kPeriodNs);              // R1
  auto a_l1 = s.align(kBaseNs + kPeriodNs + 9'000);      // L1 -> matches R1, not stale R0
  EXPECT_EQ(a_r0.ns, kBaseNs);                           // R0 published its own stamp
  EXPECT_EQ(a_r1.ns, kBaseNs + kPeriodNs);
  EXPECT_EQ(a_l1.ns, kBaseNs + kPeriodNs);               // matched R1 (R0 orphan ignored)
  EXPECT_FALSE(a_l1.desync);
}

// The orphan buffer is capacity-bounded: once full, the oldest orphan is evicted and can no longer
// match, while live pairs keep matching.
TEST(StereoTimestampAligner, BoundedCapacityEvictsOldestOrphan)
{
  StereoTimestampAligner s(kThresholdNs, kHalfPeriodNs, /*capacity=*/2);
  s.align(kBaseNs);                                      // orphan A
  s.align(kBaseNs + kPeriodNs);                          // orphan B
  s.align(kBaseNs + 2 * kPeriodNs);                      // orphan C -> evicts A (oldest)

  auto late_a = s.align(kBaseNs + 9'000);                // A's partner, but A was evicted
  EXPECT_EQ(late_a.ns, kBaseNs + 9'000);                 // treated as a fresh first-arriver
  EXPECT_FALSE(late_a.desync);

  auto c_partner = s.align(kBaseNs + 2 * kPeriodNs + 9'000);
  EXPECT_EQ(c_partner.ns, kBaseNs + 2 * kPeriodNs);      // C still matches
}
