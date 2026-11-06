// MIT License. Copyright (c) 2026 João Alano and GT4Recomp contributors.
//
// Presentation mode selection for the CoreGS front end: FIFO (vsync), mailbox or immediate.
// A host sets it through COREGS_PRESENT_MODE; the surface decides which modes it accepts.
#pragma once

#include <atomic>
#include <cstdint>
#include <string>

namespace coregs {

enum class PresentMode { fifo, mailbox, immediate };

// Parses "fifo", "mailbox" or "immediate" (case-sensitive, like every other CoreGS option).
// Throws std::invalid_argument naming the option and the value on anything else.
PresentMode parse_present_mode(const char *value);

const char *present_mode_name(PresentMode mode);

// The mode to use on a surface that may not have the wanted one. FIFO always exists (Vulkan requires it).
// Mailbox and immediate stand in for each other (both present without waiting for vsync), then FIFO.
PresentMode available_present_mode(PresentMode wanted, bool mailbox_available, bool immediate_available);

// Set by the host when it is leaving frames out (its own pacing): the renderer then presents at most one
// frame per VBlank instead of queueing them.
extern std::atomic<bool> frame_skip_recent;
// The presentation latency control's grid move, in microseconds, for a host clock to apply (0: no move).
extern std::atomic<std::int64_t> grid_shift_us;

}  // namespace coregs
