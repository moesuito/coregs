// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 João Alano and GT4Recomp contributors.
#pragma once
// TASK-174: one VkQueue used by two threads. On a device whose graphics family has a single queue (RADV on the Steam Deck), the
// presentation runs on its own thread but on the rendering queue, so the GS thread never waits for the display. Vulkan requires
// external synchronization of a queue: no two threads may be inside vkQueueSubmit, vkQueuePresentKHR, vkQueueWaitIdle or
// vkDeviceWaitIdle (which uses every queue of the device) at the same time. So every such call on the shared queue, from any
// thread, runs inside QueueLock::run. The lock also keeps, per caller, how long it waited for the other thread's call and how
// long it spent in its own, for the presentation report (the GS thread's waits are the number the Deck report asks for).
// Host only, without Vulkan types, so the fixture (hw_queue_lock_fixture.cpp) can drive it with a fake queue.
//
// Lock order: this lock first, then the backend's swapchain mutex (vkQueuePresentKHR needs both). Nothing takes them the other
// way round.

#include <atomic>
#include <chrono>
#include <cstdint>
#include <mutex>
#include <type_traits>
#include <utility>

namespace coregs {

class QueueLock {
public:
  // Who makes a queue call, for the time accounting.
  enum class Caller : std::uint32_t {
    gs_thread = 0,            // the renderer's submissions and the drawing of the pictures
    presentation_thread = 1,  // the copy of a picture into the swapchain image, and the present
    device_idle = 2,          // vkDeviceWaitIdle before a swapchain rebuild, a window change or the end
  };
  static constexpr std::uint32_t kCallers = 3;

  struct Totals {
    std::uint64_t calls = 0;
    std::uint64_t wait_ns = 0;  // waiting for the lock: another thread was inside a queue call
    std::uint64_t call_ns = 0;  // inside the queue call itself
    std::uint64_t longest_wait_ns = 0;
    std::uint64_t longest_call_ns = 0;
  };

  // Runs `call` (one queue call, returning its result) with the queue to itself.
  template <typename Call> auto run(Caller caller, Call &&call) -> decltype(call()) {
    static_assert(!std::is_void_v<decltype(call())>, "a queue call returns its VkResult");
    const auto asked = Clock::now();
    const std::lock_guard<std::mutex> lock(mutex_);
    const auto entered = Clock::now();
    auto result = std::forward<Call>(call)();
    const auto left = Clock::now();
    Counters &counters = counters_[static_cast<std::uint32_t>(caller)];
    const std::uint64_t waited = nanoseconds(entered - asked), inside = nanoseconds(left - entered);
    counters.calls.fetch_add(1, std::memory_order_relaxed);
    counters.wait_ns.fetch_add(waited, std::memory_order_relaxed);
    counters.call_ns.fetch_add(inside, std::memory_order_relaxed);
    // Under the lock: no other caller updates these at the same time.
    if (waited > counters.longest_wait_ns.load(std::memory_order_relaxed)) counters.longest_wait_ns.store(waited, std::memory_order_relaxed);
    if (inside > counters.longest_call_ns.load(std::memory_order_relaxed)) counters.longest_call_ns.store(inside, std::memory_order_relaxed);
    return result;
  }

  // Any thread: the session totals of one caller.
  Totals totals(Caller caller) const {
    const Counters &counters = counters_[static_cast<std::uint32_t>(caller)];
    Totals out;
    out.calls = counters.calls.load(std::memory_order_relaxed);
    out.wait_ns = counters.wait_ns.load(std::memory_order_relaxed);
    out.call_ns = counters.call_ns.load(std::memory_order_relaxed);
    out.longest_wait_ns = counters.longest_wait_ns.load(std::memory_order_relaxed);
    out.longest_call_ns = counters.longest_call_ns.load(std::memory_order_relaxed);
    return out;
  }

private:
  using Clock = std::chrono::steady_clock;
  struct Counters {
    std::atomic<std::uint64_t> calls{0};
    std::atomic<std::uint64_t> wait_ns{0};
    std::atomic<std::uint64_t> call_ns{0};
    std::atomic<std::uint64_t> longest_wait_ns{0};
    std::atomic<std::uint64_t> longest_call_ns{0};
  };

  static std::uint64_t nanoseconds(Clock::duration elapsed) {
    return static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(elapsed).count());
  }

  std::mutex mutex_;
  Counters counters_[kCallers];
};

}  // namespace coregs
