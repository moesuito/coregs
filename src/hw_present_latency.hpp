#pragma once
// IN-521: keeps the display queue short under vsync. At 60 frames on a 60 Hz display one frame goes in and one comes out per
// refresh, so every refresh the display skips adds a frame to the queue, and it stays there: time from presentation to display
// 37 ms at the start of a lap and 53 ms later, 25 ms without vsync (a stream's virtual display). When that time stays high, one
// presentation is left out, which takes one frame out of the queue; with a frame waiting in it the display does not even repeat
// one. It stops trying at the level where that does not help, the shortest queue this display takes. A display that skips a
// refresh just then looks like that too, so a level clearly over the lowest seen gets a second try.
//
// A run can also land with its presentations just before the flip (the pacer's grid has no relation to the display's), and some
// then miss it by a little: the frame comes a refresh late and the next one with it, 12 times in a lap in one run. The second
// such miss within 15 s moves the grid earlier by kEdgeStepMs, away from that edge, for a few ms more to the display.
//
// The shown times come from VK_KHR_present_wait2; without it nothing here runs.
// Measured and dropped: walking the pacer's grid towards the flip as well reached 35 ms, with 18 frames repeated per lap.

#include <algorithm>
#include <cstdint>
#include <limits>
#include <mutex>

namespace coregs {

class PresentLatencyControl {
public:
  static constexpr double kFrameMs = 1000.0 / 60.0;
  static constexpr int kWindow = 60;  // shown frames per decision
  static constexpr int kSettle = 30;  // shown frames after leaving one out, before judging it
  static constexpr double kStartTargetMs = 33.0;
  static constexpr double kEdgeStepMs = 4.0;
  static constexpr int kEdgeWindow = 900;  // shown frames between two misses of the flip that count as being at its edge
  static constexpr int kMaxEdgeSteps = 6;

  explicit PresentLatencyControl(double targetMs = kStartTargetMs) : targetMs_(targetMs) {}

  // Worker thread: one frame reached the display, `lagMs` after its presentation and `intervalMs` after the frame before it.
  void shown(double lagMs, double intervalMs) {
    const std::lock_guard<std::mutex> lock(mutex_);
    note_flip_miss(intervalMs);
    if (judging_ && ++settled_ <= kSettle) return;
    windowMin_ = std::min(windowMin_, lagMs);
    if (++windowCount_ < kWindow) return;
    const double lag = windowMin_;
    windowMin_ = std::numeric_limits<double>::infinity();
    windowCount_ = 0;
    lastLagMs_ = lag;
    lowestMs_ = std::min(lowestMs_, lag);
    if (judging_) {
      judging_ = false;
      if (lag <= lagBefore_ - kFrameMs / 2.0) return;  // it helped
      if (lagBefore_ > lowestMs_ + kFrameMs / 2.0) {
        lowestMs_ = lag;  // no gain, over a level seen before: one more try, and this is the level from now on
      } else {
        targetMs_ = std::max(targetMs_, std::max(lag, lagBefore_) + 2.0);
        return;
      }
    }
    if (lag > targetMs_) {
      leaveOut_ = true;
      judging_ = true;
      settled_ = 0;
      lagBefore_ = lag;
    }
  }

  // GS thread, before each presentation: true once per decision.
  bool leave_out() {
    const std::lock_guard<std::mutex> lock(mutex_);
    if (!leaveOut_) return false;
    leaveOut_ = false;
    ++leftOut_;
    return true;
  }

  // GS thread, before each presentation: how much later the pacer's grid should run, in microseconds, once.
  std::int32_t take_grid_shift_us() {
    const std::lock_guard<std::mutex> lock(mutex_);
    const std::int32_t shift = gridShiftUs_;
    gridShiftUs_ = 0;
    return shift;
  }

  int edge_steps() {
    const std::lock_guard<std::mutex> lock(mutex_);
    return edgeSteps_;
  }
  std::uint64_t left_out() {
    const std::lock_guard<std::mutex> lock(mutex_);
    return leftOut_;
  }
  double target_ms() {
    const std::lock_guard<std::mutex> lock(mutex_);
    return targetMs_;
  }
  double last_lag_ms() {
    const std::lock_guard<std::mutex> lock(mutex_);
    return lastLagMs_;
  }

private:
  // A frame a refresh late with the next one right behind it missed its flip. A display that skips a refresh has no such pair.
  void note_flip_miss(double intervalMs) {
    const bool miss = previousLong_ && intervalMs < kFrameMs / 2.0;
    previousLong_ = intervalMs > kFrameMs * 1.5;
    ++framesSinceMiss_;
    if (!miss) return;
    if (framesSinceMiss_ > kEdgeWindow || edgeSteps_ >= kMaxEdgeSteps) {
      framesSinceMiss_ = 0;
      return;
    }
    gridShiftUs_ -= static_cast<std::int32_t>(kEdgeStepMs * 1000.0);
    ++edgeSteps_;
    framesSinceMiss_ = kEdgeWindow + 1;  // the misses that follow count anew, from where the grid is now
  }

  std::mutex mutex_;
  double targetMs_;
  double windowMin_ = std::numeric_limits<double>::infinity();
  double lowestMs_ = std::numeric_limits<double>::infinity();
  int windowCount_ = 0;
  bool leaveOut_ = false;
  bool judging_ = false;
  int settled_ = 0;
  double lagBefore_ = 0.0;
  double lastLagMs_ = 0.0;
  std::uint64_t leftOut_ = 0;
  bool previousLong_ = false;
  int framesSinceMiss_ = kEdgeWindow + 1;
  int edgeSteps_ = 0;
  std::int32_t gridShiftUs_ = 0;
};

}  // namespace coregs
