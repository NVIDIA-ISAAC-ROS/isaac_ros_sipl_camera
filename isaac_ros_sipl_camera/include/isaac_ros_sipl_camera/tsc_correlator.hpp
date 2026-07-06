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

#ifndef ISAAC_ROS_SIPL_CAMERA__TSC_CORRELATOR_HPP_
#define ISAAC_ROS_SIPL_CAMERA__TSC_CORRELATOR_HPP_

#include <array>
#include <atomic>
#include <chrono>
#include <cinttypes>
#include <cstdint>
#include <stdexcept>
#include <tuple>
#include <utility>

#include "rclcpp/rclcpp.hpp"

namespace isaac_ros
{
namespace sipl
{
/// Correlates ARM64 TSC (Timer System Counter, CNTVCT_EL0) timestamps with
/// the ROS clock. Reads the timer frequency from CNTFRQ_EL0 to convert raw
/// tick counts to nanoseconds, calibrates the TSC-to-ROS offset using multiple
/// samples for optimizing accuracy.
///
/// It's recommended to update the offset regularly to account for gradual clock drift.
/// This can be done by calling calibrate() before tscToRos() or by setting a recalibration
/// interval with setRecalibrationInterval() to automatically calibrate before converting
/// your TSC timestamps to ROS time.
///
/// The Jetson platform specific TSC frequency is read from hardware register and
/// converted to nanoseconds per tick.
///
/// TSC rollover is unlikely as it starts at zero on boot and won't
/// wrap for ~585 years at 1 GHz so we assume monotonic TSC values here without rollover checks.
///
/// Thread safety: tscToRos() and calibrate() may be called concurrently from
/// multiple threads. Shared state uses atomics.
/// Concurrent calibrations are intentionally allowed to run in a race
/// rather than serialized so each stores a fresh offset, avoiding stale reads and
/// the cost of a lock as calibrating TSC to ROS time is cheap and contention is infrequent.
///
/// Usage:
///   TscCorrelator correlator(node->get_clock(), node->get_logger());
///   correlator.calibrate();                             // update regularly to account for drift
///   rclcpp::Time t = correlator.tscToRos(raw_ticks);    // convert raw TSC ticks to ROS time
class TscCorrelator
{
public:
  /// @param clock  ROS clock used as the target time domain.
  /// @param logger ROS logger for printing diagnostics.
  /// @throws std::runtime_error if the TSC frequency register reads zero.
  explicit TscCorrelator(
    rclcpp::Clock::SharedPtr clock,
    rclcpp::Logger logger = rclcpp::get_logger("TscCorrelator"))
  : clock_(std::move(clock)),
    logger_(logger)
  {
    if (archCounterGetCntvct() == 0) {
      throw std::runtime_error("TSC tick (CNTVCT_EL0) is unexpectedly zero.");
    }

    tsc_freq_ = archCounterGetCntfrq();
    if (tsc_freq_ == 0) {
      throw std::runtime_error("TSC frequency (CNTFRQ_EL0) is zero. This is necessary to know the"
                               " scaling factor for TSC ticks to nanoseconds.");
    }
    RCLCPP_INFO(logger_, "TSC frequency: %" PRIu64 " MHz", tsc_freq_ / 1000000);

    // Pre-calculate nanoseconds scaling to avoid division in the conversion hotpath.
    ns_per_tick_int_ = kNanosecondsPerSecond / tsc_freq_;

    // Use integer math with a fixed-point fraction for the division
    // remainder to avoid precision loss.
    // Shift the remainder left by 64 bits before dividing by freq to fill the
    // full 64-bit range with fractional precision. At conversion time after
    // multiplying by ticks, shift back to recover the actual remainder, avoiding
    // expensive 128-bit software division (__udivti3) on the hot path and precision loss.
    const uint64_t remainder = kNanosecondsPerSecond % tsc_freq_;
    ns_per_tick_frac_ = static_cast<uint64_t>(
      (static_cast<__uint128_t>(remainder) << 64) / tsc_freq_
    );
  }

  /// Enable auto-recalibration on tscToRos() calls according to the interval given
  /// a non-zero interval.
  ///
  /// When the TSC ticks elapsed since the last calibration exceed the
  /// interval, tscToRos() will call calibrate() automatically.
  /// An interval of 0 (default) disables auto-recalibration.
  ///
  /// Use this if you always call tscToRos() with TSC timestamps close to the
  /// current TSC time such that the recalibrated offset is accurate for your
  /// timestamp.
  void setRecalibrationInterval(std::chrono::milliseconds interval)
  {
    recalibration_interval_ticks_.store(
      static_cast<uint64_t>(interval.count()) * (tsc_freq_ / 1000),
      std::memory_order_relaxed);
  }

  /// Sample the TSC and ROS clocks using a multi-sample bracketed algorithm
  /// and update the cached offset. Thread-safe: concurrent calls produce
  /// nearly identical offsets; all computation is local and stores are atomic.
  ///
  /// Takes kNumCalibrationSamples bracketed measurements of the form
  /// (tsc_before, ros_time, tsc_after) and picks the sample with the
  /// shortest bracket interval for best accuracy.
  ///
  /// Recalibrate as needed to account for clock drift.
  void calibrate()
  {
    std::array<std::tuple<int64_t, int64_t, int64_t>, kNumCalibrationSamples> measurements;

    for (int i = 0; i < kNumCalibrationSamples; ++i) {
      uint64_t tsc_before = archCounterGetCntvct();
      const int64_t ros_ns = clock_->now().nanoseconds();
      uint64_t tsc_after = archCounterGetCntvct();

      int64_t tsc_ns_before = ticksToNs(tsc_before);
      int64_t tsc_ns_after = ticksToNs(tsc_after);

      measurements[i] = std::make_tuple(tsc_ns_before, ros_ns, tsc_ns_after);
    }

    // Pick the sample with the shortest bracket interval.
    int64_t best_ros_ns = 0;
    int64_t best_tsc_ns = 0;
    int64_t shortest_interval = 0;
    bool initialized = false;

    for (int i = 0; i < kNumCalibrationSamples; ++i) {
      const auto & [t1, ros, t2] = measurements[i];
      int64_t interval = t2 - t1;
      int64_t tsc_midpoint = t1 + interval / 2;

      if (!initialized || interval < shortest_interval) {
        best_ros_ns = ros;
        best_tsc_ns = tsc_midpoint;
        shortest_interval = interval;
        initialized = true;
      }
    }

    // Log a warning if the final interval is large, meaning all samples
    // were likely preempted or delayed as the expected execution time for calibrate() is
    // in nanoseconds.
    constexpr int64_t kIrregularlyHighIntervalNs = 100000;  // 100us
    if (initialized && (shortest_interval < 0 || shortest_interval > kIrregularlyHighIntervalNs)) {
      if (calibrated_.load(std::memory_order_acquire)) {
        RCLCPP_WARN_THROTTLE(
          logger_, *clock_, 1000,
          "TSC calibration measurement took longer than expected (%" PRId64
              " ns, expected <= %" PRId64 " ns). "
          "Timestamp conversion accuracy is currently degraded until next calibration.",
          shortest_interval,
          kIrregularlyHighIntervalNs);
      }
    }

    tsc_to_ros_offset_.store(best_ros_ns - best_tsc_ns, std::memory_order_relaxed);
    last_calibration_ticks_.store(archCounterGetCntvct(), std::memory_order_relaxed);
    calibrated_.store(true, std::memory_order_release);
  }

  /// Convert raw TSC tick counts to ROS time.
  /// If a recalibration interval is set and elapsed time since last calibration
  /// exceeds the interval, calibrate() will be called automatically.
  rclcpp::Time tscToRos(uint64_t tsc_ticks)
  {
    const uint64_t interval = recalibration_interval_ticks_.load(std::memory_order_relaxed);
    const uint64_t last_calib = last_calibration_ticks_.load(std::memory_order_relaxed);
    if (!calibrated_.load(std::memory_order_acquire) ||
      (interval > 0 && (tsc_ticks - last_calib) >= interval))
    {
      calibrate();
    }
    const int64_t tsc_ns = ticksToNs(tsc_ticks);
    const int64_t ros_ns = tsc_ns + tsc_to_ros_offset_.load(std::memory_order_relaxed);
    return rclcpp::Time(ros_ns, clock_->get_clock_type());
  }

protected:
  /// Convert raw TSC tick counts to nanoseconds.
  int64_t ticksToNs(uint64_t ticks) const
  {
    // Exact math using 64-bit fixed-point arithmetic without division or precision loss.
    return static_cast<int64_t>(
      (ticks * ns_per_tick_int_) +
      static_cast<uint64_t>((static_cast<__uint128_t>(ticks) * ns_per_tick_frac_) >> 64)
    );
  }

  /// True after at least one successful calibrate() call.
  bool isCalibrated() const {return calibrated_.load(std::memory_order_relaxed);}

  /// TSC frequency in Hz as reported by CNTFRQ_EL0.
  uint64_t tscFrequency() const {return tsc_freq_;}

  /// Current cached offset from TSC to ROS time in nanoseconds.
  int64_t offsetNs() const {return tsc_to_ros_offset_.load(std::memory_order_relaxed);}

private:
  static uint64_t archCounterGetCntvct()
  {
    uint64_t cnt;
    asm volatile ("mrs %0, cntvct_el0" : "=r" (cnt));
    return cnt;
  }

  static uint64_t archCounterGetCntfrq()
  {
    uint64_t freq;
    asm volatile ("mrs %0, cntfrq_el0" : "=r" (freq));
    return freq;
  }

  static constexpr int kNumCalibrationSamples = 5;
  static constexpr uint64_t kNanosecondsPerSecond = 1000000000ULL;

  rclcpp::Clock::SharedPtr clock_;
  rclcpp::Logger logger_;
  uint64_t tsc_freq_{0};
  uint64_t ns_per_tick_int_{0};
  uint64_t ns_per_tick_frac_{0};
  std::atomic<int64_t> tsc_to_ros_offset_{0};
  std::atomic<uint64_t> last_calibration_ticks_{0};
  std::atomic<uint64_t> recalibration_interval_ticks_{0};
  std::atomic<bool> calibrated_{false};
};

}  // namespace sipl
}  // namespace isaac_ros

#endif  // ISAAC_ROS_SIPL_CAMERA__TSC_CORRELATOR_HPP_
