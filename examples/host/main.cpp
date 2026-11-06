// MIT License. Copyright (c) 2026 João Alano and GT4Recomp contributors.
//
// CoreGS example host: the smallest program that opens a window, feeds a synthetic GS stream and
// presents the picture. It is the worked example of docs/INTEGRATION.md.
//
//   coregs_host [--frames N] [--scale S] [--profile gt4] [--headless]
//
// The stream is generated in memory: a display setup, a depth buffer, a full-screen sprite and a
// triangle. Nothing here needs game data.

#include "coregs/coregs.h"

#if defined(COREGS_WITH_GT4_PROFILE)
#include "gt4_profile.h"
#endif

#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

namespace {

constexpr std::uint32_t kDisplayWidth = 640;
constexpr std::uint32_t kDisplayHeight = 448;

LRESULT CALLBACK window_proc(HWND window, UINT message, WPARAM wparam, LPARAM lparam) {
  if (message == WM_CLOSE) {
    DestroyWindow(window);
    return 0;
  }
  if (message == WM_DESTROY) {
    PostQuitMessage(0);
    return 0;
  }
  return DefWindowProcW(window, message, wparam, lparam);
}

HWND create_window(std::uint32_t width, std::uint32_t height) {
  WNDCLASSW window_class{};
  window_class.lpfnWndProc = window_proc;
  window_class.hInstance = GetModuleHandleW(nullptr);
  window_class.lpszClassName = L"CoreGSExampleHost";
  window_class.hCursor = LoadCursorW(nullptr, MAKEINTRESOURCEW(32512));
  RegisterClassW(&window_class);
  RECT outer{0, 0, static_cast<LONG>(width), static_cast<LONG>(height)};
  AdjustWindowRect(&outer, WS_OVERLAPPEDWINDOW, FALSE);
  HWND window = CreateWindowExW(0, window_class.lpszClassName, L"CoreGS example host", WS_OVERLAPPEDWINDOW,
                                CW_USEDEFAULT, CW_USEDEFAULT, outer.right - outer.left, outer.bottom - outer.top,
                                nullptr, nullptr, window_class.hInstance, nullptr);
  if (window) ShowWindow(window, SW_SHOW);
  return window;
}

// One GIF packet in the packed format, A+D only: every item is one register write (or one vertex kick),
// with the register address in the high half of its 128-bit qword. This is the smallest packet form and
// the one the GS front end handles fastest.
struct RegWord {
  std::uint8_t address;
  std::uint64_t data;
};

void append_packet(std::vector<std::uint8_t> &out, std::initializer_list<RegWord> items) {
  const std::uint32_t nloop = static_cast<std::uint32_t>(items.size());
  std::uint64_t lo = nloop & 0x7FFFu;
  lo |= 1ull << 15;    // EOP
  lo |= 0ull << 58;    // PACKED
  lo |= 1ull << 60;    // NREG = 1
  const std::uint64_t hi = 0xEu;  // descriptor 0xE: A+D, the address is in the data qword
  const auto push = [&out](std::uint64_t value) {
    for (int byte = 0; byte < 8; ++byte) out.push_back(static_cast<std::uint8_t>(value >> (8 * byte)));
  };
  push(lo);
  push(hi);
  for (const RegWord &item : items) {
    push(item.data);
    push(item.address & 0x7Fu);
  }
}

std::uint64_t rgba(std::uint32_t colour) { return static_cast<std::uint64_t>(colour) | (0x3F800000ull << 32); }
std::uint64_t xyz(std::uint32_t x, std::uint32_t y, std::uint32_t z = 0) {
  return static_cast<std::uint64_t>(x * 16u) | (static_cast<std::uint64_t>(y * 16u) << 16) |
         (static_cast<std::uint64_t>(z) << 32);
}

void configure_display(coregs::Renderer &renderer) {
  // PMODE: circuit 1, module output, alpha 255.
  renderer.write_privileged(0x12000000u, 1u | (1u << 5) | (0xFFu << 8));
  // SMODE1: NTSC.
  renderer.write_privileged(0x12000010u, std::uint64_t{2} << 13);
  // DISPFB1: framebuffer page 0, FBW 1, PSMCT32.
  renderer.write_privileged(0x12000070u, std::uint64_t{0} | (std::uint64_t{10} << 9));  // FBW 10 = 640 pixels
  // DISPLAY1: MAGH 3 (640 pixels), DW 639, DH 447 (448 lines).
  renderer.write_privileged(0x12000080u, (std::uint64_t{3} << 23) | (std::uint64_t{639} << 32) |
                                              (std::uint64_t{447} << 44));
}

std::vector<std::uint8_t> frame_stream(std::uint32_t frame) {
  std::vector<std::uint8_t> stream;
  const std::uint32_t tint = (frame * 3u) & 0x7Fu;
  // State: FRAME_1, ZBUF_1, SCISSOR_1, XYOFFSET_1, TEST_1.
  append_packet(stream,
                {{0x4C, std::uint64_t{0} | (std::uint64_t{10} << 9)},   // FRAME_1: FBP 0, FBW 10 (640 px), CT32
                 {0x4E, std::uint64_t{64} | (std::uint64_t{10} << 9)},  // ZBUF_1: ZBP 64, FBW 10
                 {0x40, (static_cast<std::uint64_t>(kDisplayWidth - 1) << 16) |
                            (static_cast<std::uint64_t>(kDisplayHeight - 1) << 48)},  // SCISSOR_1
                 {0x18, 0},                                                          // XYOFFSET_1
                 {0x47, (std::uint64_t{1} << 16) | (std::uint64_t{1} << 17)}});      // TEST_1: ZTE, ZTST ALWAYS
  // A full-screen sprite, the background of the frame.
  append_packet(stream,
                {{0x00, 6},                                                    // PRIM
                 {0x01, rgba(0x00302010u | (tint << 16))},                     // RGBAQ
                 {0x05, xyz(0, 0)},                                            // XYZ2: top-left
                 {0x05, xyz(kDisplayWidth, kDisplayHeight)}});                 // XYZ2: size
  // A triangle that moves down the screen.
  const std::uint32_t top = 80u + frame % 120u;
  append_packet(stream,
                {{0x00, 3}, {0x01, rgba(0xFF0000FFu)},
                 {0x05, xyz(160, top)}, {0x05, xyz(480, top)}, {0x05, xyz(320, 360)}});
  return stream;
}

int usage() {
  std::fprintf(stderr, "usage: coregs_host [--frames N] [--scale S] [--profile gt4] [--headless]\n");
  return 2;
}

}  // namespace

int main(int argc, char **argv) {
  std::uint32_t frames = 600;
  std::uint32_t scale = 1;
  bool headless = false;
  bool gt4_profile = false;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--frames" && i + 1 < argc) {
      frames = static_cast<std::uint32_t>(std::strtoul(argv[++i], nullptr, 10));
    } else if (arg == "--scale" && i + 1 < argc) {
      scale = static_cast<std::uint32_t>(std::strtoul(argv[++i], nullptr, 10));
    } else if (arg == "--profile" && i + 1 < argc) {
      const std::string name = argv[++i];
      if (name != "gt4") return usage();
      gt4_profile = true;
    } else if (arg == "--headless") {
      headless = true;
    } else {
      return usage();
    }
  }

  // 1. Create the renderer and choose the options.
  coregs::Renderer renderer;
  coregs::Options options;
  options.scale = scale;
  options.half_pixel = coregs::HalfPixel::native;
  options.sync_mode = coregs::SyncMode::cache;
  std::string error;
#if defined(COREGS_WITH_GT4_PROFILE)
  if (gt4_profile) {
    coregs::gt4::Settings settings;
    renderer.set_profile(coregs::gt4::MakeProfile(settings));
  }
#else
  (void)gt4_profile;
#endif
  if (!renderer.init(options, error)) {
    std::fprintf(stderr, "coregs_host: %s\n", error.c_str());
    return 1;
  }

  // 2. Attach a window (skip it for a headless run) and hand it the client size.
  HWND window = nullptr;
  if (!headless) {
    window = create_window(1280, 720);
    if (!window) {
      std::fprintf(stderr, "coregs_host: could not create the window\n");
      return 1;
    }
    RECT client{};
    GetClientRect(window, &client);
    if (!renderer.attach_window(window, static_cast<std::uint32_t>(client.right),
                                static_cast<std::uint32_t>(client.bottom), error)) {
      std::fprintf(stderr, "coregs_host: %s\n", error.c_str());
      return 1;
    }
  }

  // 3. Configure the GS display once (the equivalent of the game's register writes).
  configure_display(renderer);

  // 4. Feed frames: GIF packets in, one present per guest VBlank.
  const auto start = std::chrono::steady_clock::now();
  bool running = true;
  for (std::uint32_t frame = 0; frame < frames && running; ++frame) {
    if (window) {
      MSG message{};
      while (PeekMessageW(&message, nullptr, 0, 0, PM_REMOVE)) {
        if (message.message == WM_QUIT) running = false;
        TranslateMessage(&message);
        DispatchMessageW(&message);
      }
      if (!running) break;
    }
    const std::vector<std::uint8_t> stream = frame_stream(frame);
    if (!renderer.gif_transfer(1, stream.data(), stream.size())) {
      std::fprintf(stderr, "coregs_host: the GIF stream was refused\n");
      return 1;
    }
    renderer.flush();  // VBlank: commit the frame's recorded work
    renderer.next_frame();
    renderer.present(coregs::PresentPhase::progressive);

    // A 60 Hz host clock. A real game runtime already has its own VBlank timing; this loop only keeps
    // the window responsive.
    const auto due = start + std::chrono::microseconds(16667 * static_cast<std::int64_t>(frame + 1));
    std::this_thread::sleep_until(due);
  }

  // 5. Shut down: let the GPU finish, then release the window.
  renderer.flush();
  renderer.wait_idle();
  const coregs::Stats stats = renderer.stats();
  const coregs::RenderCounters counters = renderer.render_counters();
  std::printf("batches=%llu gpu_draws=%llu privileged=%llu faults=%llu\n",
              static_cast<unsigned long long>(counters.batches),
              static_cast<unsigned long long>(counters.gpu_draws),
              static_cast<unsigned long long>(stats.privileged_writes),
              static_cast<unsigned long long>(stats.faults));
  std::printf("frames=%llu presents=%llu black_presents=%llu gif_bytes=%llu device=%s\n",
              static_cast<unsigned long long>(stats.frames), static_cast<unsigned long long>(stats.presents),
              static_cast<unsigned long long>(stats.black_presents), static_cast<unsigned long long>(stats.gif_bytes),
              renderer.device_report().device_name.c_str());
  if (window) {
    renderer.detach_window();
    DestroyWindow(window);
  }
  return 0;
}
