// MIT License. Copyright (c) 2026 João Alano and GT4Recomp contributors.

#include "hw_present_mode.hpp"

#include <stdexcept>

namespace coregs {

std::atomic<bool> frame_skip_recent{false};
std::atomic<std::int64_t> grid_shift_us{0};

PresentMode parse_present_mode(const char *value) {
  if (value == nullptr || *value == '\0') return PresentMode::fifo;
  const std::string text = value;
  if (text == "fifo") return PresentMode::fifo;
  if (text == "mailbox") return PresentMode::mailbox;
  if (text == "immediate") return PresentMode::immediate;
  throw std::invalid_argument("unknown value \"" + text + "\" (fifo, mailbox or immediate)");
}

const char *present_mode_name(PresentMode mode) {
  switch (mode) {
  case PresentMode::fifo: return "fifo";
  case PresentMode::mailbox: return "mailbox";
  case PresentMode::immediate: return "immediate";
  }
  return "unknown";
}

PresentMode available_present_mode(PresentMode wanted, bool mailbox_available, bool immediate_available) {
  if (wanted == PresentMode::fifo) return PresentMode::fifo;
  const bool wanted_available = wanted == PresentMode::mailbox ? mailbox_available : immediate_available;
  if (wanted_available) return wanted;
  if (wanted == PresentMode::mailbox && immediate_available) return PresentMode::immediate;
  if (wanted == PresentMode::immediate && mailbox_available) return PresentMode::mailbox;
  return PresentMode::fifo;
}

}  // namespace coregs
