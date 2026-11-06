#pragma once
// GT4: the light-flare visibility table (128 qwords, read once per race frame), answered by a deadline.
// The EE posts the read at its place in the GS stream and waits for the worker to finish it until a deadline. In time the
// answer is the exact one. Past the deadline it is the last table the worker finished, as long as that table is from
// the last kVblanksBehind VBlanks; otherwise the EE keeps waiting, so another scene's table is never an answer. The
// worker always runs the read, and its table is there for the next frame.

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <mutex>

namespace coregs::gt4 {

// How long after the VBlank the answer may come. What the EE does after the read changes with the scene (about 4 ms on an easy
// stretch, over 6 ms with a car ahead), so the limit follows it: the answer's time plus what the frame had left at its
// end is the latest answer that would still have kept the frame on time. It comes down fast and goes back up slowly.
// kAllowanceMs on top: a frame this little behind the pacer's grid catches up by itself and keeps its interval, and many frames
// pass the limit by less than that (6x race, late answers 13% to 9%, frames over 17.5 ms 4 against 6 of 3,700).
class FlareDeadline {
public:
  static constexpr double kStartMs = 12.0, kLowestMs = 7.0, kHighestMs = 14.0, kAllowanceMs = 0.7;

  void frame(double answerMs, double slackMs) {
    const double latest = answerMs + slackMs;
    limitMs_ += (latest - limitMs_) * (latest < limitMs_ ? 0.35 : 0.03);
    const double lowest = kLowestMs - kAllowanceMs, highest = kHighestMs - kAllowanceMs;
    limitMs_ = limitMs_ < lowest ? lowest : limitMs_ > highest ? highest : limitMs_;
  }
  double ms() const { return limitMs_ + kAllowanceMs; }

private:
  double limitMs_ = kStartMs - kAllowanceMs;
};

class FlareTableReads {
public:
  static constexpr std::uint32_t kQwords = 128;
  static constexpr std::size_t kBytes = kQwords * 16u;
  static constexpr std::uint64_t kVblanksBehind = 3;
  using Table = std::array<std::uint8_t, kBytes>;
  using Clock = std::chrono::steady_clock;

  // EE: a read of this VBlank is about to enter the stream. `dest`: where the worker leaves the words, for a read the EE
  // does not wait for (IN-522).
  std::uint64_t post(std::uint64_t vblank, std::uint8_t *dest = nullptr) {
    const std::uint64_t ticket = ++posted_;
    postedVblank_[ticket % postedVblank_.size()] = vblank;
    postedDest_[ticket % postedDest_.size()].store(dest, std::memory_order_relaxed);
    return ticket;
  }

  // Worker: where the next read to finish leaves its words, null when the EE takes them itself.
  std::uint8_t *next_dest() {
    const std::lock_guard<std::mutex> lock(mutex_);
    return postedDest_[(done_ + 1u) % postedDest_.size()].load(std::memory_order_relaxed);
  }

  // Any thread (IN-522, the VU1 worker): waits for this read and copies part of its table. A VU1 job that uses a read comes
  // after it in the GS stream, so a later read cannot finish first, and the table is still this read's.
  void wait_copy(std::uint64_t ticket, std::size_t offset, std::size_t bytes, std::uint8_t *out) {
    std::unique_lock<std::mutex> lock(mutex_);
    doneChanged_.wait(lock, [&] { return done_ >= ticket; });
    std::copy_n(table_.begin() + static_cast<std::ptrdiff_t>(offset), bytes, out);
  }

  // EE: copies the answer of `ticket` to `data`; returns whether it is a late one (an earlier read's table).
  bool answer(std::uint64_t ticket, std::uint64_t vblank, Clock::time_point deadline, std::uint8_t *data) {
    std::unique_lock<std::mutex> lock(mutex_);
    const auto finished = [&] { return done_ >= ticket; };
    bool late = !doneChanged_.wait_until(lock, deadline, finished);
    if (late && !usable(ticket, vblank)) {
      doneChanged_.wait(lock, finished);
      late = false;
    }
    std::copy(table_.begin(), table_.end(), data);
    return late;
  }

  // Worker: the next posted read has finished with this table. Returns that read's ticket.
  std::uint64_t finish(const Table &table) {
    std::uint64_t ticket = 0;
    {
      const std::lock_guard<std::mutex> lock(mutex_);
      table_ = table;
      ticket = ++done_;
    }
    doneChanged_.notify_all();
    return ticket;
  }

private:
  bool usable(std::uint64_t ticket, std::uint64_t vblank) const {
    if (done_ == 0 || done_ + postedVblank_.size() <= ticket) return false;  // no table, or its VBlank is gone from the list
    return vblank - postedVblank_[done_ % postedVblank_.size()] <= kVblanksBehind;
  }

  std::uint64_t posted_ = 0;                        // EE only
  std::array<std::uint64_t, 16> postedVblank_{};    // written by the EE, read by the EE
  std::array<std::atomic<std::uint8_t *>, 16> postedDest_{};
  std::mutex mutex_;
  std::condition_variable doneChanged_;
  std::uint64_t done_ = 0;
  Table table_{};
};

}  // namespace coregs::gt4
