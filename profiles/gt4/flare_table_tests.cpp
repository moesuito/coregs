// CTest gt4_flare_table: the deadline of the flare table read (flare_table.hpp), host only.
#include "flare_table.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <thread>

using coregs::gt4::FlareTableReads;
using Clock = FlareTableReads::Clock;
using namespace std::chrono_literals;

namespace {

FlareTableReads::Table table_of(std::uint8_t value) {
  FlareTableReads::Table table;
  table.fill(value);
  return table;
}

void check(bool condition, const char *what) {
  if (condition) return;
  std::fprintf(stderr, "gt4_flare_table: %s\n", what);
  std::exit(1);
}

}  // namespace

int main() {
  FlareTableReads::Table out;
  {  // In time: the exact answer, not late.
    FlareTableReads reads;
    const std::uint64_t ticket = reads.post(10);
    reads.finish(table_of(1));
    check(!reads.answer(ticket, 10, Clock::now() + 1h, out.data()), "a finished read is late");
    check(out[0] == 1 && out[FlareTableReads::kBytes - 1] == 1, "a finished read gave another table");
  }
  {  // Past the deadline with a recent table: that table, late. The worker's read then answers the next frame.
    FlareTableReads recent;
    const std::uint64_t first = recent.post(10);
    recent.finish(table_of(2));
    check(!recent.answer(first, 10, Clock::now() + 1h, out.data()), "the first read is late");
    const std::uint64_t second = recent.post(11);
    check(recent.answer(second, 11, Clock::now() + 1ms, out.data()), "an unfinished read is not late");
    check(out[0] == 2, "a late answer is not the last table");
    recent.finish(table_of(3));
    const std::uint64_t third = recent.post(12);
    check(recent.answer(third, 12, Clock::now() + 1ms, out.data()), "the third read is not late");
    check(out[0] == 3, "the late read's table did not reach the next frame");
  }
  {  // No table yet: it waits past the deadline for the exact one.
    FlareTableReads reads;
    const std::uint64_t ticket = reads.post(10);
    std::thread worker([&] {
      std::this_thread::sleep_for(20ms);
      reads.finish(table_of(4));
    });
    check(!reads.answer(ticket, 10, Clock::now() + 1ms, out.data()), "the first read of all was late");
    check(out[0] == 4, "the first read of all gave another table");
    worker.join();
  }
  {  // A table too old (another scene): it waits past the deadline for the exact one.
    FlareTableReads reads;
    const std::uint64_t first = reads.post(10);
    reads.finish(table_of(5));
    check(!reads.answer(first, 10, Clock::now() + 1h, out.data()), "the first read is late");
    const std::uint64_t later = reads.post(10 + FlareTableReads::kVblanksBehind + 1);
    std::thread worker([&] {
      std::this_thread::sleep_for(20ms);
      reads.finish(table_of(6));
    });
    check(!reads.answer(later, 10 + FlareTableReads::kVblanksBehind + 1, Clock::now() + 1ms, out.data()), "an old table was an answer");
    check(out[0] == 6, "an old table was copied");
    worker.join();
  }
  {  // The newest VBlank a late answer may come from.
    FlareTableReads reads;
    const std::uint64_t first = reads.post(10);
    reads.finish(table_of(7));
    check(!reads.answer(first, 10, Clock::now() + 1h, out.data()), "the first read is late");
    const std::uint64_t later = reads.post(10 + FlareTableReads::kVblanksBehind);
    check(reads.answer(later, 10 + FlareTableReads::kVblanksBehind, Clock::now() + 1ms, out.data()), "a recent table was refused");
    check(out[0] == 7, "a recent table was not copied");
  }
  {  // IN-522: a read the EE does not wait for: the VU1 worker waits for its words, and the worker has their memory.
    FlareTableReads reads;
    std::uint8_t guest[FlareTableReads::kBytes] = {};
    const std::uint64_t ticket = reads.post(10, guest);
    check(reads.next_dest() == guest, "the worker does not know where the words go");
    std::thread worker([&] {
      std::this_thread::sleep_for(20ms);
      reads.finish(table_of(8));
    });
    std::uint8_t part[16] = {};
    reads.wait_copy(ticket, 32, sizeof part, part);
    check(part[0] == 8 && part[15] == 8, "the VU1 worker did not get the read's words");
    worker.join();
    const std::uint64_t waited = reads.post(11);
    check(reads.next_dest() == nullptr, "a read the EE waits for has a destination");
    reads.finish(table_of(9));
    check(!reads.answer(waited, 11, Clock::now() + 1h, out.data()) && out[0] == 9, "the waited read is wrong");
  }
  {  // The limit follows what the frame had left, down fast, up slowly, within its bounds.
    coregs::gt4::FlareDeadline deadline;
    check(deadline.ms() == coregs::gt4::FlareDeadline::kStartMs, "the limit does not start at its start");
    for (int i = 0; i < 20; ++i) deadline.frame(11.0, -1.0);  // answered at 11 ms, 1 ms late: 10 ms was the latest
    check(deadline.ms() > 10.6 && deadline.ms() < 10.8, "the limit did not come down to the latest answer plus the allowance");
    const double low = deadline.ms();
    deadline.frame(9.0, 7.0);
    check(deadline.ms() > low && deadline.ms() < low + 0.3, "the limit went up too fast");
    for (int i = 0; i < 2000; ++i) deadline.frame(16.0, 7.67);
    check(deadline.ms() == coregs::gt4::FlareDeadline::kHighestMs, "the limit passed its highest value");
    for (int i = 0; i < 100; ++i) deadline.frame(9.0, -8.0);
    check(deadline.ms() == coregs::gt4::FlareDeadline::kLowestMs, "the limit passed its lowest value");
  }
  std::puts("gt4_flare_table: ok (in time, late, none yet, too old, oldest allowed, limit)");
  return 0;
}
