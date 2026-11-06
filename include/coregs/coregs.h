// MIT License. Copyright (c) 2026 João Alano and GT4Recomp contributors.
//
// CoreGS: a generic, hardware-faithful PlayStation 2 Graphics Synthesizer renderer on Vulkan.
//
// This is the header a host includes. It is Vulkan-free: every Vulkan handle stays inside the
// library, so a game runtime (a static recompilation, an emulator-style host, a tool) only feeds
// GS traffic and presents a picture.
//
// See docs/INTEGRATION.md for the call order, the readback contract and the threading rules.
#pragma once

#include "coregs/options.h"
#include "coregs/profile.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace coregs {

// Guest-side stream events, delivered inline while a GIF packet is parsed.
enum class StreamEvent { signal, finish, label };

// Returning true asks the caller to flush its own batching.
using StreamEventHandler = bool (*)(void *context, StreamEvent event, std::uint32_t value);

// Display phase of the VBlank being closed.
enum class PresentPhase { field_even, field_odd, progressive };

// scanout(): the output image as host RGBA8 (red in the low byte), at internal resolution.
struct ScanoutRequest {
  PresentPhase phase = PresentPhase::progressive;
  std::uint32_t width = 0;   // filled by the renderer
  std::uint32_t height = 0;  // filled by the renderer
  std::vector<std::uint32_t> pixels;
};

// Session counters. They explain a missing draw or a slow frame and are cheap enough to read
// once per second; nothing here is required for rendering.
struct Stats {
  std::uint64_t frames = 0;             // next_frame() calls
  std::uint64_t gif_bytes = 0;          // bytes accepted by gif_transfer
  std::uint64_t privileged_writes = 0;  // accepted by write_privileged
  std::uint64_t flushes = 0;            // flush() calls
  std::uint64_t presents = 0;           // present() calls (window or headless)
  std::uint64_t black_presents = 0;     // presents that displayed a black frame
  std::uint64_t swapchain_recreations = 0;
  std::uint64_t faults = 0;             // out-of-range reads and writes

  std::uint64_t held_presents = 0;      // re-presented images (no new frame this VBlank)
  std::uint64_t dropped_frames = 0;     // frame ends replaced before they reached the display
  std::uint64_t auxiliary_frames = 0;   // render-only passes that restored guest memory
  double translate_us = 0;              // batch translation, last frame
  double execute_us = 0;                // GPU drawer submission, last frame
  double compose_us = 0;                // display composition, last frame
  double upload_us = 0;                 // composed image upload, last present
  double present_us = 0;                // swapchain acquire, submit and present, last present

  // Destination rectangle of the last presented image inside the swapchain.
  std::uint32_t present_left = 0, present_top = 0, present_width = 0, present_height = 0;
  std::uint32_t present_fill = 0;
  std::uint32_t present_fill_millionths = 0;  // the fill target ratio x 1e6 (0 = the whole window)
  std::uint32_t present_downscale_levels = 0;  // 2x2 box steps before the final blit (upscaled frames)
  std::uint32_t present_source_width = 0, present_source_height = 0;  // the presented frame (scaled size)

  // How frames reach the display and what that costs the GS thread (session totals).
  const char *present_path = "";          // "rendering_queue", "own_queue" or "shared_queue"
  std::uint64_t present_requests = 0;     // presentations asked of the path; each ends as one of the four below
  std::uint64_t presents_shown = 0;       // vkQueuePresentKHR returned VK_SUCCESS or VK_SUBOPTIMAL_KHR
  std::uint64_t presents_not_shown = 0;   // the presentation side was behind: no free slot, picture or swapchain image in time
  std::uint64_t presents_failed = 0;      // an acquire or present error (out of date, surface lost)
  std::uint64_t presents_left_out = 0;    // left out by the display latency control
  double gs_present_step_us = 0;          // GS thread: the whole presentation step
  double gs_present_wait_us = 0;          // GS thread: inside its fence waits, acquire, queue submit and present calls there
  double gs_queue_lock_wait_us = 0;       // GS thread: waiting for the shared queue's lock (all its submissions)
  double gs_queue_lock_longest_wait_us = 0;  // GS thread: its longest single wait for that lock
  std::uint64_t gs_queue_calls = 0;       // GS thread: queue calls under the shared queue's lock
  double presenter_queue_lock_wait_us = 0;  // presentation thread: waiting for the shared queue's lock
  double presenter_queue_call_us = 0;       // presentation thread: inside its submit and present calls on the shared queue
  double presenter_queue_longest_call_us = 0;  // presentation thread: its longest single call there (what the GS thread may wait)
  double presenter_display_wait_us = 0;  // presentation thread: waiting for the display (acquire, previous presentation shown)
};

// Renderer counters, flattened so this header stays free of the renderer's internals.
struct RenderCounters {
  std::uint64_t batches = 0;
  std::uint64_t primitives = 0;
  std::uint64_t primitives_skipped_texture = 0;
  std::uint64_t batches_empty = 0;
  std::uint64_t textures = 0;
  std::uint64_t texels = 0;
  std::uint64_t gpu_draws = 0;
  std::uint64_t writeback_pixels = 0;
  std::uint64_t flat_deindexed = 0;
  std::uint64_t sprites = 0;
  // Output stage.
  std::uint64_t presentation_frames = 0;
  std::uint64_t blank_frames = 0;
  std::uint64_t deinterlaced_frames = 0;
  std::uint64_t interlace_offset_frames = 0;
  std::uint64_t merge_alpha_from_texel = 0;
  std::uint64_t writeback_depth_words = 0;
  std::uint64_t fifo_overruns = 0;
  // Session totals: CPU microseconds and work counts.
  double translate_us = 0.0;
  double execute_us = 0.0;
  double decode_us = 0.0;
  double cache_writeback_us = 0.0;
  double cache_load_us = 0.0;
  double gif_us = 0.0;     // the whole GS stream (front end, translate, execute, transfers)
  double upload_us = 0.0;  // host uploads
  std::uint64_t gpu_busy_ns = 0;            // GPU busy time (first to last timestamp)
  std::uint64_t gpu_ns[8] = {};             // pass, decode, clut, swizzle, upload, download, scaled, refresh
  std::uint64_t gpu_early_submissions = 0;  // mid-frame submissions
  std::uint64_t vblanks_interlaced = 0;     // SMODE2.INT = 1 (480i)
  std::uint64_t vblanks_progressive = 0;    // SMODE2.INT = 0 (480p)
  // Internal scale: draws and passes repeated at scale, scaled-image refreshes, GPU composes and their circuits.
  std::uint32_t scale = 1;
  std::uint64_t scaled_draws = 0;
  std::uint64_t scaled_passes = 0;
  std::uint64_t scaled_refreshes = 0;
  std::uint64_t scaled_refresh_pixels = 0;
  std::uint64_t scaled_composes = 0;
  std::uint64_t scaled_circuits_from_slots = 0;
  std::uint64_t scaled_circuits_from_glm = 0;
  std::uint64_t scaled_circuits_from_shadows = 0;
  std::uint64_t scaled_shadow_saves = 0;
  std::uint64_t tone_runs = 0;
  std::uint64_t tone_carries = 0;
  std::uint64_t tone_strip_pages = 0;      // pages written by one dispatch
  std::uint64_t tone_strip_batches = 0;    // strip batches those dispatches replaced
  std::uint64_t tone_verify_mismatch_bytes = 0;  // bytes that differ from the batches (verification mode)
  std::uint64_t gpu_composed_frames = 0;
  std::uint64_t gpu_compose_skips[4] = {};  // why a VBlank fell back to the 1x CPU presentation
  std::uint64_t gpu_render_passes = 0;
  std::uint64_t gpu_upload_bytes = 0;
  std::uint64_t gpu_readback_bytes = 0;
  std::uint64_t gpu_scaled_image_bytes = 0;
  std::uint64_t textures_cached = 0;
  std::uint64_t cache_writebacks = 0;
  std::uint64_t cache_loads = 0;
  std::uint64_t gpu_chunks = 0;
  std::uint64_t gpu_submissions = 0;
  std::uint64_t gpu_barriers = 0;
  std::uint64_t writeback_reasons[9] = {};  // other, texture, CLUT, transfer, display, eviction, load overlap,
                                            // fallback batch, CPU sprite
};

// Result of the device feature report.
struct DeviceReport {
  std::string device_name;
  std::uint32_t vendor_id = 0;
  bool amd = false;
  std::vector<std::string> missing_required;  // refusal list when non-empty
  std::vector<std::string> optional_found;
  std::vector<std::string> optional_missing;
  std::uint32_t max_image_dimension_2d = 0;  // also the internal scale ceiling (/ 1280)
  std::uint32_t max_compute_workgroup_count_x = 0;
};

class Renderer {
public:
  Renderer();
  ~Renderer();
  Renderer(const Renderer &) = delete;
  Renderer &operator=(const Renderer &) = delete;

  // Creates the instance, the device and the queues. Never presents. On failure `error` names the
  // missing requirement and the function returns false.
  bool init(const Options &options, std::string &error);

  // Compiles the shaders the driver cache lacks (`progress` gets done and total, only when there is work).
  bool prepare_shaders(const std::function<void(std::size_t, std::size_t)> &progress, std::string &error);

  // The game profile: rules that recognise a particular game's content (off by default, see
  // docs/PROFILES.md). Call before init() for the shader precompilation list, or any time for the
  // per-draw rules; the generic profile renders by the hardware rules alone.
  void set_profile(const GameProfile &profile);
  const GameProfile &profile() const;

  void set_stream_event_handler(StreamEventHandler handler, void *context);

  // Feeds raw GIF qwords for path 1, 2 or 3. Transfers may split at any qword boundary.
  bool gif_transfer(int path, const std::uint8_t *bytes, std::size_t byte_count);
  bool gif_busy(int path) const;
  void reset_gif_paths();

  // Stores privileged register writes in 0x12000000..0x12001FFF. Returns false outside that range.
  bool write_privileged(std::uint32_t address, std::uint64_t value);
  std::uint64_t read_privileged(std::uint32_t address) const;

  void reset_context();
  void clear_vram();

  // Copies up to `byte_count` bytes of the pending LOCAL->HOST transfer (a multiple of 16, one GIF
  // qword) and zero-fills the rest of the buffer. Returns the number of real bytes copied, at most
  // `byte_count`. The contract is in docs/INTEGRATION.md (readbacks).
  std::size_t read_transfer_fifo(void *destination, std::size_t byte_count);

  // CPU view of GS memory coherent with the recorded work (4 MiB). Valid until the next call that
  // changes GS memory; nullptr when the current mode keeps no CPU copy.
  const std::uint8_t *map_vram(std::uint32_t offset, std::uint32_t size);

  void flush();       // submits recorded work, never waits
  void next_frame();  // headless frame boundary
  // Field of the VBlank being closed (CSR.FIELD from the VSYNC record). Set before present() and
  // before a scanout dump so the field offset and the deinterlacer use this VBlank's field.
  void set_display_field(bool odd);
  void set_age_targets(bool age);  // off for a VBlank with no drawing on purpose
  bool has_presentation_thread() const;
  void wait_idle();

  // Builds the display image the way present() would. Returns false when there is no image (black).
  bool scanout(ScanoutRequest &request);

  // Swapchain on an existing native window (passed as void* to keep this header platform-free:
  // HWND on Windows). `width`/`height` are the client area.
  bool attach_window(void *native_window, std::uint32_t width, std::uint32_t height, std::string &error);
  void detach_window();
  // Resize and fullscreen only set a flag; recreation happens before the next present.
  void resize_window(std::uint32_t width, std::uint32_t height);
  // Display-aspect fill: with `fill` the next presents put the picture into the largest centred
  // rectangle of `ratio_millionths` / 1e6 (0 = the whole window) instead of the 16:9 rectangle.
  // Any thread; effective from the next present.
  void set_display_aspect(bool fill, std::uint32_t ratio_millionths);
  // One call per guest VBlank: end_frame(phase) then one present_slot(). Headless (no window) only counts.
  void present(PresentPhase phase);
  // The two halves of present(): end_frame composes the guest frame and submits its stamp to the
  // present scheduler (false when nothing can be shown: headless, no swapchain); present_slot is one
  // host present opportunity (new image, held image or nothing). A host clock faster than the guest
  // calls present_slot() more often; it must run on the thread owning the queue.
  bool end_frame(PresentPhase phase);
  void present_slot();

  // The host is leaving frames out with its own pacing: present at most one frame per VBlank.
  void set_frames_left_out(bool left_out);

  // True while a window (swapchain) is attached; headless runs advance frames themselves.
  bool has_window() const;

  const Options &options() const;
  const DeviceReport &device_report() const;
  Stats stats() const;
  RenderCounters render_counters() const;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace coregs
