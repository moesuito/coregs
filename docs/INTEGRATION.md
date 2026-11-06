# Connecting CoreGS to a game runtime

This is the practical guide for a port author. It assumes the game's EE/VU/IOP code already runs
natively and that you can observe the GS traffic the game produces: GIF packets, privileged GS
register writes, VSyncs, and the host↔local transfers the game starts.

CoreGS is not an emulator. It never runs guest code, never reads a disc image and never asks for a
save. It only needs the GS traffic and a window (or not even that, for headless runs).

```
guest runtime ── GIF bytes + register writes ──▶ CoreGS ── picture ──▶ window
      ▲                                             │
      └────────── LOCAL→HOST readback words ◀───────┘
```

## 1. Lifecycle

```cpp
coregs::Renderer gs;
coregs::Options options;
std::string error;

if (!gs.init(options, error)) { /* report error */ }
if (!gs.attach_window(native_window, client_width, client_height, error)) { /* report error */ }

// ... frames ...

gs.wait_idle();
gs.detach_window();
```

`init()` creates the Vulkan instance, picks a device, creates the queues and prepares the renderer.
It never presents. A failure leaves the object unusable and `error` names what is missing (for
example a required Vulkan feature); there is no silent fallback to another renderer.

`attach_window()` takes a native window handle (`HWND` on Windows, passed as `void *`) and the
client area in pixels. The window belongs to the host: CoreGS creates the surface and the swapchain
and never pumps messages. Call `resize_window(width, height)` when the client area changes; the
swapchain is rebuilt before the next present.

`detach_window()` destroys the surface and the swapchain. Call it after `wait_idle()` if the host is
about to destroy the window, so no presentation is in flight.

## 2. The guest frame

Feed the GS traffic in guest order. The GS thread is the only thread that may call the feed
functions (`gif_transfer`, `write_privileged`, `flush`, `read_transfer_fifo`, `reset_*`,
`clear_vram`).

```cpp
for (;;) {
  // The traffic the guest produced since the last call, in order:
  gs.write_privileged(address, value);          // PMODE, DISPFB, DISPLAY, SMODE, CSR, IMR, ...
  gs.gif_transfer(1, packet, bytes);            // GIF path 1
  gs.gif_transfer(2, packet, bytes);            // path 2
  gs.gif_transfer(3, packet, bytes);            // path 3
  if (gs.gif_busy(1)) { /* the guest would wait for the GS FIFO */ }

  // LOCAL→HOST transfer the game started with a TRXDIR write:
  std::uint32_t words[8];
  const std::size_t got = gs.read_transfer_fifo(words, 8);
  // got < 8: the transfer ended; zero-fill the rest.

  gs.flush();                                   // VBlank: commit the frame
  gs.set_display_field(odd_field);              // CSR.FIELD of this VBlank
  gs.next_frame();
  gs.present(odd_field ? coregs::PresentPhase::field_odd : coregs::PresentPhase::field_even);
}
```

### GIF packets

`gif_transfer(path, bytes, byte_count)` accepts the raw GIF stream exactly as the guest DMA writes
it. `bytes` must be 16-byte aligned in size and at least 16 bytes; a transfer may split at any
128-bit qword boundary. Paths 1, 2 and 3 are independent and carry packed, reglist and image
packets. Path 0 and a non-multiple-of-16 size are refused (the call returns false).

The renderer honours the GIF events a packet can request: `SIGNAL` and `FINISH` are reported through
the stream-event handler if one is installed, so a host that models the GS interrupt can flush its
own batching there. Most hosts can ignore them.

### Privileged registers

`write_privileged(address, value)` accepts the GS register block at `0x12000000`–`0x12001FFF`, 8
bytes aligned. The display registers (`PMODE`, `SMODE1/2`, `DISPFB1/2`, `DISPLAY1/2`, `SYNCV`,
`CSR`, `IMR`, `BGCOLOR`) drive composition and presentation; the others are recorded for the
renderer's own state. Out-of-range or misaligned writes return false and count a fault in the stats.

Read a register back with `read_privileged(address)`.

### VSync and field

Call `flush()` once per guest VBlank, then `set_display_field()` with `CSR.FIELD` of that VBlank
(`true` for the odd field) before presenting or dumping. The field decides the interlace offset and
which field the deinterlacer treats as the newest one. Interlaced titles (480i menus, for example)
draw two fields per frame; a progressive title passes `PresentPhase::progressive`.

`set_age_targets(false)` marks a VBlank as intentionally empty (for example a title that presents
without drawing); the renderer then does not count that VBlank as target age.

### Uploads

Host→local uploads are ordinary GIF image packets (`TRXDIR = 0`) and need no separate call. The
renderer writes them into GS memory, keeps page identities for the texture cache and, when the
upload overlaps a drawn target, patches that target on the GPU instead of dropping it.

### Downloads and the readback contract

A game's `LOCAL→HOST` transfer (`TRXDIR = 1`) is answered synchronously:

1. The renderer flushes the pending batch, so every earlier draw is recorded.
2. It reads the source rectangle back from any render target that holds newer pixels than GS memory
   (a GPU reduction followed by a copy), or from GS memory directly when nothing was drawn there.
3. `read_transfer_fifo(destination, qwords)` copies the words out. The call returns the number of
   whole qwords of real data; once the transfer is drained it returns 0 and the caller zero-fills.

The readback is exact and may block the GS thread until the GPU finishes the work that feeds it.
That is the price of fidelity for a game that inspects its own rendering (light flare visibility,
palette lookups, menu effects). Two options change it:

- `COREGS_CORE2_DELAYED_DOWNLOADS=1` answers a repeated download of the same rectangle with the
  previous result, one frame late, and never blocks. Only enable it when the game tolerates a
  one-frame-old answer; it is off by default and is never implied by a profile.
- The GT4 profile's `hide_light_flare_table` rule answers one specific table read with zeros. It is
  a picture change (the effect disappears) and is off unless a host asks for it.

The FIFO never overruns silently: a read past the end of the transfer counts `fifo_overruns` in
`render_counters()`.

### Reset

- `reset_context()` resets the GS state the way the guest's GS reset does: registers, CLUT, every
  GIF path, the pending read and the display state. GS memory is kept.
- `clear_vram()` zeroes GS memory and drops the targets and caches.
- `reset_gif_paths()` discards partially received GIF packets on all three paths.

## 3. Presentation

`present(phase)` is the simple path: it ends the frame and offers one presentation opportunity.
`end_frame(phase)` and `present_slot()` are the two halves for hosts whose display clock differs
from the guest's:

- `end_frame()` composes the frame and hands its stamp to the internal scheduler (returns false
  when nothing can be shown: no window, no swapchain).
- `present_slot()` is one host present opportunity: it shows the new frame, re-presents the held
  one, or does nothing. A 120/144 Hz host calls it more often than `end_frame()`; the scheduler
  keeps the guest's 60 Hz spacing and leaves frames out instead of queuing them.

`present_slot()` must run on the thread that owns the render queue. With `present()` both calls
happen on the calling thread, which is the usual GS thread.

The presented picture goes into the largest centred 16:9 rectangle of the window, or into the
largest rectangle of a requested display ratio:

```cpp
gs.set_display_aspect(true, 21u * 1000000u / 9u);  // a 21:9 picture in any window
gs.set_display_aspect(false, 0);                   // back to the fixed 16:9 rectangle
```

`COREGS_PRESENT_MODE` selects `fifo` (vsync, the default), `mailbox` or `immediate`; a surface that
does not offer the requested mode falls back to the closest one it has, with a log line.

`scanout(request)` builds the same picture into host pixels for a screenshot, a video encoder or a
headless test. `PresentPhase` selects the field.

For headless runs, skip `attach_window()`: `present()` counts presents and `scanout()` still works.

## 4. Options

`coregs::Options` is the stable configuration surface. A host sets the fields directly, or calls
`coregs::parse_options(lookup)` to read the same fields from `COREGS_HW_*` environment variables
(useful for a launcher's settings file). `describe_options()` and `options_json()` report the
effective values.

| Option | Meaning |
| --- | --- |
| `scale` | Internal resolution, 1–12x (clamped to `maxImageDimension2D / 1280`). |
| `half_pixel` | Vertex half-pixel mode: `native`, `normal`, `off`. |
| `blend` | Blend accuracy: `basic`, `medium`, `high` (default), `full`. |
| `accurate_atest` | Exact alpha test at low blend levels. |
| `mipmap` | `off`, `basic`, `full`. |
| `filter` | Texture filtering: `ps2` (default), `nearest`, `forced`, `forced_no_sprite`. |
| `dither` | `native`, `fine`, `off`. |
| `sync_mode` | `cache` (resident targets, default) or `writeback` (reference path). |
| `readback` | `exact` (default), `full`, `none`. |
| `deinterlace` | `automatic`, `off`, `weave`, `bob`, `blend`, `adaptive` (+ `bottom_field_first`). |
| `antiblur` | GS anti-blur for interlaced content. |
| `postfx` | Bit set: shadeboost, FXAA, CAS, scanlines. |
| `present_filter` | `bilinear`, `nearest`, `sharp`. |
| `crop` | Native pixels cropped from left, top, right, bottom. |
| `textures_root` | Texture-replacement pack folder (empty: off). |
| `dump_textures`, `dump_format`, `dump_source` | Texture and frame dumps for debugging. |
| `gpu_timing` | GPU timestamps in the counters (adds a little GPU work). |

Invalid values are refused with `coregs::OptionError`; nothing silently changes meaning.

## 5. Threading rules

- **GS thread:** `gif_transfer`, `write_privileged`, `flush`, `read_transfer_fifo`, `scanout`,
  `reset_context`, `clear_vram`, `reset_gif_paths`, `set_display_field`, `set_age_targets`,
  `next_frame`, `end_frame`, `present`, `map_vram`.
- **Any thread:** `set_display_aspect`, `set_frames_left_out`, `stats`, `render_counters`,
  `options`, `device_report`, `has_window`.
- **Queue-owning thread:** `present_slot` (the GS thread with `present()`).
- **Not concurrent with the GS thread:** `set_profile`, `set_stream_event_handler`,
  `prepare_shaders`, `attach_window`, `detach_window`, `resize_window`, `wait_idle` (the last two
  may be called when the renderer is idle).

The renderer never starts the presentation thread for a host that uses `present()`; the internal
presenter is used when the host asks for the paced path, and it is joined by `wait_idle()` and
`detach_window()`.

## 6. The example host

`examples/host/main.cpp` is the smallest complete host. Read it as the reference:

1. It creates a `coregs::Renderer` and sets `Options`.
2. It creates a Win32 window and calls `attach_window()`.
3. It writes the display registers once (`PMODE`, `SMODE1`, `DISPFB1`, `DISPLAY1`).
4. Per frame it builds a synthetic GIF stream (A+D register writes: frame buffer, depth buffer,
   scissor, test, a full-screen sprite and a triangle), calls `gif_transfer()`, `flush()`,
   `next_frame()` and `present()`.
5. It calls `wait_idle()` and `detach_window()` on exit.

Run it with `--headless` to exercise the same path without a window, `--scale N` to render above 1x,
and `--profile gt4` to put the GT4 profile's rules on top. The packet builder in that file is also
the smallest documented example of the GS packet format CoreGS expects.

## 7. Common pitfalls

- **Nothing is drawn:** a GIF packet whose `NLOOP` does not match the number of data qwords is
  dropped by the front end. Each packed item is one 128-bit qword.
- **The picture is black:** check `SMODE1` and `DISPLAY1`/`DISPFB1`; a display register that
  describes no known video mode leaves the output black, and `Stats::black_presents` counts it.
- **A readback hangs:** a LOCAL→HOST read waits for the GPU work that produces its rectangle. A game
  that starts a read while the GPU is starved will stall; keep the frame's submissions flowing.
- **Tearing or skipped frames:** `mailbox`/`immediate` present without vsync. Use `fifo` for a
  faithful 60 Hz picture, or call `present_slot()` from a faster host clock.
- **A game looks wrong but the picture is stable:** open `docs/FEATURES.md`, check the format or
  effect you are seeing, and report it with the game's name and a capture (see CONTRIBUTING.md).
