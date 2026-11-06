// CTest hw_present_spacing: TASK-082, when each picture is shown (hw_present_spacing.hpp), host only.
#include "hw_present_spacing.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

using coregs::PresentSpacing;

namespace {

void check(bool condition, const char *what) {
  if (condition) return;
  std::fprintf(stderr, "hw_present_spacing: %s\n", what);
  std::exit(1);
}

struct Result {
  std::int64_t smallest, largest, mean, longestDelay;
};

// Pictures arrive with these gaps, over and over. Skips the first ones, while the mean settles.
Result run(const std::vector<std::int64_t> &gaps, int pictures) {
  PresentSpacing spacing;
  std::int64_t arrival = 1000000, last = 0;
  Result r{INT64_MAX, 0, 0, 0};
  std::int64_t total = 0, count = 0;
  for (int i = 0; i < pictures; ++i) {
    arrival += gaps[static_cast<std::size_t>(i) % gaps.size()];
    const std::int64_t shown = spacing.when(arrival);
    check(shown >= arrival, "a picture shown before it arrived");
    if (i > 200) {
      const std::int64_t interval = shown - last;
      r.smallest = std::min(r.smallest, interval);
      r.largest = std::max(r.largest, interval);
      r.longestDelay = std::max(r.longestDelay, shown - arrival);
      total += interval;
      ++count;
    }
    last = shown;
  }
  r.mean = total / count;
  return r;
}

}  // namespace

int main() {
  {  // 38 frames per second from 16.7 ms steps, drawing at 24 ms: arrivals 24, 24, 33 become about 27 each.
    const Result r = run({24000, 24000, 33333}, 2000);
    check(r.largest - r.smallest < 4000, "the intervals are not even at 24, 24, 33");
    check(r.mean > 26500 && r.mean < 27500, "the mean moved");
    check(r.longestDelay < 12000, "too long a delay");
  }
  {  // 45 frames per second: steps of 1, 1 and 2.
    const Result r = run({16667, 16667, 33333}, 2000);
    check(r.largest - r.smallest < 4000, "the intervals are not even at 16, 16, 33");
  }
  {  // Even arrivals stay as they are, with no delay to speak of.
    const Result r = run({33333}, 2000);
    check(r.largest - r.smallest < 1500, "even arrivals were disturbed");
    check(r.longestDelay < 1000, "even arrivals were delayed");
  }
  {  // A hitch: the picture goes at once, and the rate comes back.
    PresentSpacing spacing;
    std::int64_t arrival = 0;
    for (int i = 0; i < 300; ++i) spacing.when(arrival += 25000);
    arrival += 100000;
    check(spacing.when(arrival) == arrival, "a late picture waited");
    std::int64_t last = arrival;
    std::int64_t shown = 0;
    for (int i = 0; i < 100; ++i) {
      arrival += 25000;
      shown = spacing.when(arrival);
      last = shown;
    }
    check(spacing.mean_us() < 26000 && last - arrival < 3000, "the rate did not come back after a hitch");
  }
  {  // After a long stop it starts again: no wait for a mean from before.
    PresentSpacing spacing;
    std::int64_t arrival = 0;
    for (int i = 0; i < 100; ++i) spacing.when(arrival += 40000);
    arrival += PresentSpacing::kRestartUs + 1;
    check(spacing.when(arrival) == arrival, "waited after a long stop");
  }
  std::puts("hw_present_spacing: ok (38 and 45 frames per second, even arrivals, a hitch, a long stop)");
  return 0;
}
