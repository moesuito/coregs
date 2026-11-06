#pragma once
// TASK-082: when to show each picture, so a framerate between the divisors of 60 keeps even intervals. With frames left out, pictures
// arrive at uneven times (24, 24, 33 ms for 38 frames per second), since the game's lists come on its 16.7 ms steps. A picture that
// comes early waits for its place, at about the mean interval from the last one; a picture that comes late goes at once. The small
// delay this adds settles at what the latest arrivals need. Host only, in microseconds, so the fixture can drive it.

#include <algorithm>
#include <cstdint>

namespace coregs {

class PresentSpacing {
public:
  static constexpr std::int64_t kFrameUs = 16667;
  static constexpr std::int64_t kRestartUs = 250000;  // a stop this long (a menu, a hitch) starts again

  // `shareOf256`: how much of the mean interval a picture waits for. Just under all of it lets lost time come back. Where the wait
  // holds up the drawing of the next picture it must leave more room (see the backend), or waiting raises the mean.
  explicit PresentSpacing(std::int64_t shareOf256 = 248) : share_(shareOf256) {}

  // A picture arrived at `arrivalUs`, shown or not: a picture that could not be shown still counts, or missing pictures raise the mean
  // and the mean makes more of them miss, and it locks at a slower rate.
  void note_arrival(std::int64_t arrivalUs) {
    if (!started_ || arrivalUs - lastArrival_ > kRestartUs) {
      started_ = true;
      lastArrival_ = arrivalUs;
      lastShown_ = arrivalUs - meanUs_;
      return;
    }
    const std::int64_t sample = std::clamp(arrivalUs - lastArrival_, kFrameUs / 2, std::max(meanUs_ * 3, kFrameUs));
    meanUs_ += (sample - meanUs_) / 16;
    lastArrival_ = arrivalUs;
  }

  // When to show the picture that arrived at `arrivalUs`, never before it arrived. With another one waiting behind it, sooner: but not
  // at once, or the two go together, the very thing this is here to avoid.
  std::int64_t place(std::int64_t arrivalUs, bool anotherWaiting = false) {
    const std::int64_t share = anotherWaiting ? share_ * 3 / 4 : share_;
    lastShown_ = std::max(arrivalUs, lastShown_ + meanUs_ * share / 256);
    return lastShown_;
  }

  std::int64_t when(std::int64_t arrivalUs) {
    note_arrival(arrivalUs);
    return place(arrivalUs);
  }

  std::int64_t mean_us() const { return meanUs_; }

private:
  std::int64_t share_;
  bool started_ = false;
  std::int64_t meanUs_ = kFrameUs;
  std::int64_t lastArrival_ = 0;
  std::int64_t lastShown_ = 0;
};

}  // namespace coregs
