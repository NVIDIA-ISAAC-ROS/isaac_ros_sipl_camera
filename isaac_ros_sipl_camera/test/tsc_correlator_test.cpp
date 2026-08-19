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

#include "isaac_ros_sipl_camera/tsc_correlator.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <thread>
#include <vector>

#define PRINT_DEBUG 0

using isaac_ros::sipl::TscCorrelator;

/// Exposes protected members for unit testing.
class TestableTscCorrelator : public TscCorrelator
{
public:
  using TscCorrelator::TscCorrelator;
  using TscCorrelator::ticksToNs;
  using TscCorrelator::isCalibrated;
  using TscCorrelator::tscFrequency;
  using TscCorrelator::offsetNs;
};

class TscCorrelatorTest : public ::testing::Test
{
protected:
  void SetUp() override
  {
    rclcpp::init(0, nullptr);
  }

  void TearDown() override
  {
    rclcpp::shutdown();
  }
};

TEST_F(TscCorrelatorTest, NotCalibratedByDefault)
{
  auto clock = std::make_shared<rclcpp::Clock>(RCL_ROS_TIME);
  TestableTscCorrelator correlator(clock);
  EXPECT_FALSE(correlator.isCalibrated());
  correlator.calibrate();
  EXPECT_TRUE(correlator.isCalibrated());
}

TEST_F(TscCorrelatorTest, FrequencyIsNonZero)
{
  auto clock = std::make_shared<rclcpp::Clock>(RCL_ROS_TIME);
  TestableTscCorrelator correlator(clock);

  uint64_t freq = correlator.tscFrequency();
  EXPECT_GT(freq, 0u);
  std::cout << "  TSC frequency: " << freq << " Hz ("
            << static_cast<double>(freq) / 1e6 << " MHz)\n";
}

TEST_F(TscCorrelatorTest, TicksToNsConsistentWithFrequency)
{
  auto clock = std::make_shared<rclcpp::Clock>(RCL_ROS_TIME);
  TestableTscCorrelator correlator(clock);

  uint64_t freq = correlator.tscFrequency();
  // freq ticks should equal exactly 1 second.
  int64_t one_second_ns = correlator.ticksToNs(freq);
  EXPECT_EQ(one_second_ns, 1'000'000'000LL)
    << "ticksToNs(frequency) conversion should equal exactly 1 second";

  EXPECT_EQ(correlator.ticksToNs(0), 0);
}

TEST_F(TscCorrelatorTest, NanosecondsToTscTicksConsistentWithFrequency)
{
  auto clock = std::make_shared<rclcpp::Clock>(RCL_ROS_TIME);
  TestableTscCorrelator correlator(clock);

  const uint64_t frequency = correlator.tscFrequency();
  EXPECT_EQ(correlator.nanosecondsToTscTicks(1'000'000'000U), frequency);
}

TEST_F(TscCorrelatorTest, TicksToNsPrecisionEdgeCases)
{
  auto clock = std::make_shared<rclcpp::Clock>(RCL_ROS_TIME);
  TestableTscCorrelator correlator(clock);

  uint64_t freq = correlator.tscFrequency();

  // Test a variety of edge cases to ensure the 64-bit fixed-point math
  // does not lose precision and exactly matches pure mathematical division.
  const std::vector<uint64_t> test_ticks = {
    0ULL,
    1ULL,
    freq - 1,
    freq,
    freq + 1,
    1'000'000'000ULL,                       // 1 second of 1Gz ticks
    32'000'000ULL,                          // 1 second of 32Mhz ticks
    1'000'000'000ULL * 60 * 60 * 24,        // 1 day of 1Gz ticks
    (1ULL << 53),                           // 2^53 (double precision limit)
    (UINT64_MAX / 1000),                    // ~2^64 / 1000 (huge uptime)
  };

  for (uint64_t ticks : test_ticks) {
    int64_t exact_math_ns = static_cast<int64_t>(
      (static_cast<__uint128_t>(ticks) * 1'000'000'000ULL) / freq);

    EXPECT_EQ(correlator.ticksToNs(ticks), exact_math_ns)
      << "Fixed-point ticksToNs calculation incorrectly differs from exact high precision 128-bit "
      << "mathematical division at ticks = " << ticks;
  }
}

TEST_F(TscCorrelatorTest, TscToRosReturnsPlausibleTime)
{
  auto clock = std::make_shared<rclcpp::Clock>(RCL_ROS_TIME);
  TestableTscCorrelator correlator(clock);
  correlator.calibrate();

  // Collect the current TSC and ROS timestamps and compare with the converted TSC time.
  // Expect this to be within a small reasonable difference < 50ms to account for potential for
  // context switches and other interrupts and just validate the conversion is in the ballpark
  // without being a flaky test.
  uint64_t cntvct;
  asm volatile ("mrs %0, cntvct_el0" : "=r" (cntvct));
  rclcpp::Time now_ros = clock->now();
  rclcpp::Time converted = correlator.tscToRos(cntvct);

  constexpr int64_t kToleranceNs = 50'000'000;
  int64_t diff_ns = std::llabs(now_ros.nanoseconds() - converted.nanoseconds());
  EXPECT_LT(diff_ns, kToleranceNs)
    << "Converted TSC time unexpectedly differs from ROS now by " << diff_ns << " ns (> 50 ms)";
}

TEST_F(TscCorrelatorTest, OffsetStabilityOverPeriod)
{
  auto clock = std::make_shared<rclcpp::Clock>(RCL_ROS_TIME);
  TestableTscCorrelator correlator(clock);

  constexpr int kNumSamples = 10;
  constexpr auto kSampleInterval = std::chrono::milliseconds{200};

  std::vector<int64_t> offsets;
  offsets.reserve(kNumSamples);

  for (int i = 0; i < kNumSamples; ++i) {
    correlator.calibrate();
    int64_t offset = correlator.offsetNs();
    offsets.push_back(offset);

    if (i > 0) {
      // Expect some difference but not too much. Typically much less than 100ns but in CI there
      // is much higher offset differences so we set a higher but still bounded expectation.
      int64_t delta = std::llabs(offset - offsets[i - 1]);
      EXPECT_LT(delta,
        50'000)  << "Sample " << i << " offset jumped " << delta
                 << " ns from previous (> 50 us)";
    }

    std::this_thread::sleep_for(kSampleInterval);
  }

  EXPECT_TRUE(correlator.isCalibrated());

  int64_t min_off = offsets.front();
  int64_t max_off = offsets.front();
  double sum = 0.0;

  for (int64_t offset : offsets) {
    min_off = std::min(min_off, offset);
    max_off = std::max(max_off, offset);
    sum += static_cast<double>(offset);
  }

  int64_t spread = max_off - min_off;
  EXPECT_LT(spread, 500'000)
    << "Offset spread of " << spread << " ns across " << kNumSamples
    << " samples exceeds 500 us";

  #if PRINT_DEBUG
  // Calculate the mean and standard deviation of the offsets.
  double mean = sum / offsets.size();
  double sq_sum = 0.0;
  for (int64_t offset : offsets) {
    double diff = static_cast<double>(offset) - mean;
    sq_sum += diff * diff;
  }
  double stddev = std::sqrt(sq_sum / offsets.size());

  int64_t total_drift = offsets.back() - offsets.front();
    std::cout << "\n"
              << std::string(76, '=') << "\n"
              << "  TSC-to-ROS Offset Drift Summary\n"
              << std::string(76, '-') << "\n"
              << "    Samples     : " << offsets.size() << "\n"
              << "    Min offset  : " << min_off << " ns\n"
              << "    Max offset  : " << max_off << " ns\n"
              << "    Mean offset : " << std::fixed << std::setprecision(1) << mean << " ns\n"
              << "    Stddev      : " << std::fixed << std::setprecision(1) << stddev << " ns\n"
              << "    Spread      : " << spread << " ns\n"
              << "    Total drift : " << total_drift << " ns (last - first)\n"
              << std::string(76, '=') << "\n\n";
  #endif
}
