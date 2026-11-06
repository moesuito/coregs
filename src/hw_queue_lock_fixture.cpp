// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 João Alano and GT4Recomp contributors.
// CTest hw_queue_lock: TASK-174, one queue shared by the GS thread and the presentation thread, host only (no Vulkan).
//
// 1. QueueLock keeps every call on a fake queue to one thread at a time (Vulkan external synchronization), with several threads
//    calling at once, and charges the time a caller waited for the lock to that caller.
// 2. hand_over_verdict, case by case.
// 3. A model of the presentation of hw_gs_backend.cpp (present_on_shared_queue and present_paced, pacer_worker,
//    show_paced_picture) on fake queues, with a fake GPU and display, with the GS thread's rule of hw_shared_present.hpp: on the
//    shared queue with frames left out on and off, and on IN-516's second queue with frames left out. The GS thread never draws
//    over a picture whose copy is still to be submitted or still to run on the GPU, every drawn picture's semaphore is waited on
//    exactly once (shown or released), every presentation ends as shown, not shown or failed, and the GS thread never waits for the
//    display. The same model with the rule present_paced had before the TASK-174 review must see pictures drawn over before their
//    copy ran.
#include "hw_queue_lock.hpp"
#include "hw_shared_present.hpp"

#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <mutex>
#include <thread>
#include <vector>

using coregs::HandOverVerdict;
using coregs::PresenterLoad;
using coregs::QueueLock;
using Clock = std::chrono::steady_clock;

namespace {

void check(bool condition, const char *what) {
  if (condition) return;
  std::fprintf(stderr, "hw_queue_lock: FAIL: %s\n", what);
  std::exit(1);
}

// A queue that notices two callers inside it at the same time.
struct FakeQueue {
  std::atomic<int> inside{0};
  std::atomic<std::uint64_t> overlaps{0};
  std::atomic<std::uint64_t> calls{0};

  // One queue call that takes about `busy` (spinning, so the scheduler cannot hide an overlap).
  int call(std::chrono::microseconds busy) {
    if (inside.fetch_add(1, std::memory_order_acq_rel) != 0) overlaps.fetch_add(1, std::memory_order_relaxed);
    const auto until = Clock::now() + busy;
    while (Clock::now() < until) {
    }
    inside.fetch_sub(1, std::memory_order_acq_rel);
    calls.fetch_add(1, std::memory_order_relaxed);
    return 0;  // VK_SUCCESS
  }
};

// Sleeps round up to the system tick (15.6 ms by default on Windows), far too coarse for these timings: pauses yield instead.
void pause_us(std::int64_t microseconds) {
  const auto until = Clock::now() + std::chrono::microseconds(microseconds);
  while (Clock::now() < until) std::this_thread::yield();
}

// Small deterministic generator, one per thread.
struct Random {
  std::uint32_t state;
  std::uint32_t next() {
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return state;
  }
  std::uint32_t below(std::uint32_t limit) { return limit == 0 ? 0 : next() % limit; }
};

void test_exclusion() {
  QueueLock lock;
  FakeQueue queue;
  constexpr int kCallsPerThread = 4000;
  const QueueLock::Caller callers[3] = {QueueLock::Caller::gs_thread, QueueLock::Caller::presentation_thread,
                                        QueueLock::Caller::device_idle};
  std::vector<std::thread> threads;
  for (int t = 0; t < 3; ++t) {
    threads.emplace_back([&, t] {
      Random random{0x9E3779B9u + static_cast<std::uint32_t>(t)};
      for (int i = 0; i < kCallsPerThread; ++i) {
        lock.run(callers[t], [&] { return queue.call(std::chrono::microseconds(random.below(20))); });
      }
    });
  }
  for (std::thread &thread : threads) thread.join();
  check(queue.overlaps.load() == 0, "two threads were inside the queue at the same time");
  check(queue.calls.load() == 3u * kCallsPerThread, "a queue call was lost");
  for (const QueueLock::Caller caller : callers) {
    check(lock.totals(caller).calls == kCallsPerThread, "the calls of a caller are not counted");
  }
  std::printf("exclusion: %llu calls from 3 threads, 0 overlaps\n", static_cast<unsigned long long>(queue.calls.load()));
}

// A present that blocks inside the lock (some drivers wait for the display there) makes the GS thread's submissions wait: the
// wait is charged to the GS thread, the time in the call to the presentation thread.
void test_accounting() {
  QueueLock lock;
  FakeQueue queue;
  constexpr int kPresents = 10;
  constexpr auto kPresentHold = std::chrono::milliseconds(3);
  std::atomic<bool> presenting{true};
  std::thread presenter([&] {
    for (int i = 0; i < kPresents; ++i) {
      lock.run(QueueLock::Caller::presentation_thread, [&] { return queue.call(kPresentHold); });
      pause_us(1000);
    }
    presenting = false;
  });
  std::uint64_t gs_calls = 0;
  while (presenting.load()) {
    lock.run(QueueLock::Caller::gs_thread, [&] { return queue.call(std::chrono::microseconds(5)); });
    ++gs_calls;
  }
  presenter.join();
  const QueueLock::Totals gs = lock.totals(QueueLock::Caller::gs_thread);
  const QueueLock::Totals shown = lock.totals(QueueLock::Caller::presentation_thread);
  check(queue.overlaps.load() == 0, "overlap in the accounting test");
  check(gs.calls == gs_calls, "the GS thread's calls are not all counted");
  check(shown.calls == kPresents, "the presents are not all counted");
  check(shown.call_ns >= static_cast<std::uint64_t>(kPresents) * 3000000u, "the time inside the presents is short");
  check(gs.wait_ns > 1000000u, "the GS thread's wait for the blocked presents is not charged to it");
  check(gs.wait_ns <= shown.call_ns + 50000000u, "the GS thread waited far longer than the presents held the queue");
  std::printf("accounting: gs waited %.2f ms over %llu calls while %d presents held the queue %.2f ms\n", gs.wait_ns / 1e6,
              static_cast<unsigned long long>(gs.calls), kPresents, shown.call_ns / 1e6);
}

void test_verdicts() {
  constexpr std::size_t kSpacedQueue = 2, kInFlight = 2;
  const auto verdict = [&](std::size_t queued, bool busy, std::uint32_t held, std::uint32_t slot, bool spaced) {
    return coregs::hand_over_verdict(PresenterLoad{queued, busy, held}, slot, spaced, kSpacedQueue, kInFlight);
  };
  check(verdict(0, false, 0u, 0, false) == HandOverVerdict::draw, "an idle presentation thread takes the picture");
  check(verdict(1, false, 0b010u, 0, false) == HandOverVerdict::draw, "one picture on its way leaves room");
  check(verdict(0, true, 0b010u, 0, false) == HandOverVerdict::draw, "one busy picture leaves room");
  check(verdict(1, true, 0b110u, 0, false) == HandOverVerdict::presenter_behind, "two pictures on their way: not shown");
  check(verdict(2, false, 0b110u, 0, false) == HandOverVerdict::presenter_behind, "two queued: not shown");
  check(verdict(1, true, 0b110u, 0, true) == HandOverVerdict::draw, "spaced: one waiting for its place leaves room");
  check(verdict(2, false, 0b110u, 0, true) == HandOverVerdict::presenter_behind, "spaced: two waiting for their place: not shown");
  check(verdict(0, false, 0b001u, 0, false) == HandOverVerdict::slot_held, "a held slot is never drawn");
  check(verdict(0, true, 0b100u, 2, true) == HandOverVerdict::slot_held, "a held slot is never drawn, spaced either");
  check(verdict(1, true, 0b011u, 2, true) == HandOverVerdict::draw, "the oldest slot is free with two held");
  // hand_over_decision: the fence last, and never on a held slot or when the presentation thread is behind.
  int fence_asked = 0;
  const auto decision = [&](std::size_t queued, bool busy, std::uint32_t held, std::uint32_t slot, bool spaced, bool copy_done) {
    return coregs::hand_over_decision(PresenterLoad{queued, busy, held}, slot, spaced, kSpacedQueue, kInFlight, [&] {
      ++fence_asked;
      return copy_done;
    });
  };
  check(decision(0, false, 0u, 0, true, true) == HandOverVerdict::draw && fence_asked == 1, "a free slot whose copy ran is drawn");
  check(decision(1, true, 0b110u, 0, true, false) == HandOverVerdict::copy_pending && fence_asked == 2,
        "the presentation thread busy with n + 1, n + 2 queued: n + 3 waits for the copy of n");
  check(decision(0, false, 0b001u, 0, true, true) == HandOverVerdict::slot_held && fence_asked == 2,
        "the fence of a held slot is never asked");
  check(decision(2, false, 0b110u, 0, true, true) == HandOverVerdict::presenter_behind && fence_asked == 2,
        "the fence is not asked when the presentation thread is behind");
  std::printf("verdicts: 14 cases\n");
}

// The model. One slot holds a picture, a fence (the copy of the picture into the swapchain image ran on the GPU) and the picture's
// semaphore (signalled by the drawing, waited on by the copy or the release).
constexpr std::uint32_t kSlots = 3;  // frame_slots in the backend
constexpr std::size_t kPacedPictures = 2;
constexpr std::size_t kPresentsInFlight = 2;

struct ModelSlot {
  std::atomic<std::int64_t> copy_done_us{0};  // when the GPU finishes the copy of the last picture of this slot
  std::atomic<std::uint64_t> picture{0};      // which frame the picture holds
  std::atomic<bool> semaphore_signalled{false};
};

// Which path the model runs.
struct ModelSetup {
  const char *name;
  bool one_queue;               // true: the shared queue under the lock (TASK-174); false: IN-516's second queue, no lock
  bool always_spaced;           // every picture shown at its place (present_paced, frames left out); else 150 in every 400
  bool check_copy;              // the rule since the TASK-174 review (hand_over_decision); false: the rule present_paced had before
  std::uint32_t copy_delay_us;  // at most this long after its submission the GPU runs a copy (on the second queue it waits for the
                                // swapchain image the display gives back)
  std::uint64_t frames;
};

struct Model {
  explicit Model(const ModelSetup &chosen) : setup(chosen) {}
  const ModelSetup setup;
  QueueLock lock;
  FakeQueue render_queue;
  FakeQueue second_queue;  // IN-516's presentation queue; unused on one queue
  std::array<ModelSlot, kSlots> slots;
  const Clock::time_point start = Clock::now();
  std::int64_t now_us() const {
    return std::chrono::duration_cast<std::chrono::microseconds>(Clock::now() - start).count();
  }

  // One queue call: on the shared queue under the lock, as submit_to and present_locked; on two queues each has one thread.
  void queue_call(bool presentation, std::uint32_t busy_us) {
    if (!setup.one_queue) {
      (presentation ? second_queue : render_queue).call(std::chrono::microseconds(busy_us));
      return;
    }
    lock.run(presentation ? QueueLock::Caller::presentation_thread : QueueLock::Caller::gs_thread,
             [&] { return render_queue.call(std::chrono::microseconds(busy_us)); });
  }

  // The presentation thread's queue, as in hw_gs_backend.cpp.
  struct Picture {
    std::uint32_t slot;
    std::uint64_t frame;
    bool spaced;
  };
  std::mutex pacer_mutex;
  std::condition_variable pacer_changed;
  std::deque<Picture> pacer_queue;
  bool pacer_busy = false, pacer_stop = false;
  std::uint32_t pacer_slots_held = 0;

  std::atomic<std::uint64_t> shown{0}, not_shown{0}, failed{0}, semaphore_errors{0};
  std::atomic<std::uint64_t> overwritten_before_submit{0};  // drawn over before the presentation thread submitted its copy
  std::uint64_t overwritten_before_copy = 0;                // drawn over while the GPU had still to run the copy (GS thread)
  std::uint64_t copy_pending_drops = 0;                     // frames not shown because of that (GS thread)
  std::uint64_t requests = 0, submissions = 0;
  double gs_longest_call_ms = 0;

  void presenter() {
    Random random{0xC0FFEEu};
    std::uint32_t finished = UINT32_MAX;
    for (;;) {
      Picture picture{};
      {
        std::unique_lock<std::mutex> guard(pacer_mutex);
        pacer_busy = false;
        if (finished != UINT32_MAX) pacer_slots_held &= ~(1u << finished);
        pacer_changed.notify_all();
        pacer_changed.wait(guard, [&] { return pacer_stop || !pacer_queue.empty(); });
        if (pacer_queue.empty()) return;
        picture = pacer_queue.front();
        pacer_queue.pop_front();
        pacer_busy = true;
        finished = picture.slot;
      }
      ModelSlot &slot = slots[picture.slot];
      // A spaced picture waits for its place; every picture then waits for a swapchain image (the display: up to 16 ms, now and
      // then up to 40 ms).
      if (picture.spaced) pause_us(random.below(8000));
      pause_us(random.below(random.below(8) == 0 ? 40000 : 16000));
      const std::uint32_t outcome = random.below(100);  // 3 in 100 no image in time, 1 in 100 out of date
      if (slot.picture.load() != picture.frame) overwritten_before_submit.fetch_add(1);
      if (!slot.semaphore_signalled.exchange(false)) semaphore_errors.fetch_add(1);  // the copy or release waits on it
      if (outcome < 4) {
        queue_call(true, 10);  // the release
        slot.copy_done_us = now_us();
        (outcome < 3 ? not_shown : failed).fetch_add(1);
        continue;
      }
      queue_call(true, 30);  // the copy
      slot.copy_done_us = now_us() + random.below(setup.copy_delay_us);
      queue_call(true, random.below(10) == 0 ? 2000 : 20);  // the present; now and then a driver holds the queue in it
      shown.fetch_add(1);
    }
  }

  // present_on_shared_queue or present_paced, on the GS thread.
  void present(std::uint64_t frame, bool spaced) {
    ++requests;
    const auto call_start = Clock::now();
    const std::uint32_t slot_index = static_cast<std::uint32_t>(submissions % kSlots);
    ModelSlot &slot = slots[slot_index];
    HandOverVerdict verdict = HandOverVerdict::draw;
    {
      const std::lock_guard<std::mutex> guard(pacer_mutex);
      const PresenterLoad load{pacer_queue.size(), pacer_busy, pacer_slots_held};
      const auto copy_done = [&] { return slot.copy_done_us.load() <= now_us(); };  // the fence, with timeout 0
      verdict = setup.check_copy ? coregs::hand_over_decision(load, slot_index, spaced, kPacedPictures, kPresentsInFlight, copy_done)
                                 : coregs::hand_over_verdict(load, slot_index, spaced, kPacedPictures, kPresentsInFlight);
    }
    if (verdict == HandOverVerdict::copy_pending) ++copy_pending_drops;
    if (verdict != HandOverVerdict::draw) {
      not_shown.fetch_add(1);
    } else {
      if (slot.copy_done_us.load() > now_us()) ++overwritten_before_copy;
      slot.picture = frame;  // drawing overwrites the picture
      if (slot.semaphore_signalled.exchange(true)) semaphore_errors.fetch_add(1);  // signalled twice without a wait
      queue_call(false, 15);
      ++submissions;
      {
        const std::lock_guard<std::mutex> guard(pacer_mutex);
        pacer_queue.push_back(Picture{slot_index, frame, spaced});
        pacer_slots_held |= 1u << slot_index;
      }
      pacer_changed.notify_all();
    }
    const double call_ms = std::chrono::duration<double, std::milli>(Clock::now() - call_start).count();
    if (call_ms > gs_longest_call_ms) gs_longest_call_ms = call_ms;
  }

  void drain_and_stop() {
    {
      std::unique_lock<std::mutex> guard(pacer_mutex);
      pacer_changed.wait(guard, [&] { return pacer_queue.empty() && !pacer_busy; });
      pacer_stop = true;
    }
    pacer_changed.notify_all();
  }
};

// Runs one model and prints what it saw.
void run_model(Model &model) {
  std::thread presenter([&] { model.presenter(); });
  Random random{0x51ED270Bu};
  for (std::uint64_t frame = 1; frame <= model.setup.frames; ++frame) {
    const bool spaced = model.setup.always_spaced || frame % 400 < 150;  // frames left out for 150 frames in every 400
    // The renderer's own submissions between presentations, on the rendering queue.
    for (std::uint32_t i = random.below(4); i > 0; --i) model.queue_call(false, 10);
    pause_us(1000 + random.below(9000));  // 1 to 10 ms of drawing
    model.present(frame, spaced);
  }
  model.drain_and_stop();
  presenter.join();
  const QueueLock::Totals gs = model.lock.totals(QueueLock::Caller::gs_thread);
  std::printf("model %s: %llu presentations, shown %llu, not shown %llu (copy pending %llu), failed %llu; drawn over before the "
              "copy %llu; gs lock wait %.2f ms in total, longest presentation call %.3f ms; overlaps %llu\n",
              model.setup.name, static_cast<unsigned long long>(model.requests), static_cast<unsigned long long>(model.shown.load()),
              static_cast<unsigned long long>(model.not_shown.load()), static_cast<unsigned long long>(model.copy_pending_drops),
              static_cast<unsigned long long>(model.failed.load()), static_cast<unsigned long long>(model.overwritten_before_copy),
              gs.wait_ns / 1e6, model.gs_longest_call_ms,
              static_cast<unsigned long long>(model.render_queue.overlaps.load() + model.second_queue.overlaps.load()));
}

// The checks every path with the rule of hand_over_decision must pass.
void check_model(const Model &model) {
  const std::uint64_t ended = model.shown.load() + model.not_shown.load() + model.failed.load();
  check(model.render_queue.overlaps.load() == 0 && model.second_queue.overlaps.load() == 0,
        "model: two threads inside a queue at once");
  check(model.overwritten_before_submit.load() == 0, "model: a picture was drawn over before its copy was submitted");
  check(model.overwritten_before_copy == 0, "model: a picture was drawn over before the GPU ran its copy");
  check(model.semaphore_errors.load() == 0, "model: a picture semaphore was signalled twice or waited on unsignalled");
  check(ended == model.requests, "model: a presentation is not accounted for");
  check(model.shown.load() > model.setup.frames / 5, "model: too few presentations shown");
  check(model.not_shown.load() > 0, "model: the display never fell behind (the test did not test that)");
  // The display waits (up to 40 ms) are the presentation thread's; on one queue the GS thread only meets a present holding the
  // queue (2 ms).
  check(model.gs_longest_call_ms < 15.0, "model: the GS thread waited for the display");
}

void test_models() {
  // TASK-174: the shared queue, frames left out on and off.
  Model shared(ModelSetup{"shared queue", true, false, true, 3000, 1500});
  run_model(shared);
  check_model(shared);
  // IN-516's second queue with frames left out (present_paced), with the rule since the TASK-174 review: the copy waits for its
  // swapchain image on the second queue, up to 12 ms after its submission. The case must have come up and been handled.
  Model paced(ModelSetup{"second queue, frames left out", false, true, true, 12000, 1000});
  run_model(paced);
  check_model(paced);
  check(paced.copy_pending_drops > 0, "model: no frame met a copy still to run (the test did not test that)");
  // The same with the rule present_paced had before: the model must see pictures drawn over before their copy ran, or it could
  // not have caught that.
  Model before(ModelSetup{"second queue, frames left out, rule before the review", false, true, false, 12000, 600});
  run_model(before);
  check(before.overwritten_before_copy > 0, "model: the rule before the review drew over no picture (the model misses the case)");
}

}  // namespace

int main() {
  test_exclusion();
  test_accounting();
  test_verdicts();
  test_models();
  std::printf("hw_queue_lock: all checks passed\n");
  return 0;
}
