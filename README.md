# CoreGS

**A generic PlayStation 2 Graphics Synthesizer renderer for port authors — Vulkan, Apache-2.0, clean-room.**

CoreGS renders the PS2 GS the way the hardware does it: it parses the GIF stream, tracks the
privileged registers, keeps GS local memory coherent, decodes textures and palettes, and draws
through storage images with exact blending, depth, alpha test, DATE and fog. It is not an emulator
and it is not tied to one game: a port feeds it GS traffic and presents a picture.

> **Status:** a generic PS2 GS hardware renderer, proven on Gran Turismo 4. Other games need testing
> and, often, per-game optimisation. The renderer follows the hardware rules by default; a game
> profile may add rules that are only correct for one game (Gran Turismo 4's profile is included as
> the example). See [docs/FEATURES.md](docs/FEATURES.md) for what is supported today.

## Why

Porting a PS2 game means running its EE/VU/IOP code natively and still producing the picture the GS
produced. Reusing an emulator's renderer is often impossible (licence, GPL, another emulator's
internals) or undesirable (a JIT-shaped interface). CoreGS is a standalone library with a small,
stable interface:

- **Self-contained.** Only Vulkan and the C++ standard library. No emulator code, no game data.
- **Hardware-faithful by default.** Blending, depth, alpha test, DATE, CLUT, swizzles, readbacks and
  the display path follow the GS rules; a profile can trade an effect for speed, but never silently.
- **Made for static recompilation and emulator-style hosts.** Feed GIF packets and register writes
  from the GS thread; present from the same or another thread.
- **Apache-2.0.** Link it into a commercial or open-source port; the licence asks only that you
  keep the notice and credit the project.

## Features at a glance

- GIF paths 1, 2 and 3, packed/reglist/image packets, PATH1 and PATH3 streaming.
- All colour and depth formats (PSMCT32/24/16/16S, PSMZ32/24/16/16S), indexed formats (T8, T4, T8H,
  T4HL, T4HH), mipmaps and the LOD formula.
- Exact blend modes (the cases a GPU cannot express natively are evaluated in the pixel shader),
  alpha test, DATE, fog, dithering, TFX, channel write masks, SCANMSK.
- CLUT: CSM1/CSM2, TEXCLUT, palettes drawn by the game into targets and read back on the GPU.
- Texture cache, mip levels, texture replacement packs (PNG/DDS keys, dumps) compatible with the packs players already have.
- Readbacks with an explicit synchronisation contract (LOCAL→HOST transfers, FIFO depth).
- Render targets with dirty tracking, readbacks at 1x, and a GPU memory mirror (GLM) for textures.
- Internal resolution 1x–12x (device ceiling), FSR1/Lanczos presentation upscale, FXAA, shadeboost,
  scanlines, CAS sharpening, deinterlacing (auto/weave/bob/blend/adaptive).
- Presentation to a host window: FIFO/mailbox/immediate, 16:9 or display-aspect fill, paced presenter.
- A game profile mechanism for content-aware rules, with Gran Turismo 4 as the worked example.

### Texture packs

CoreGS reads the de-facto PS2 texture-pack layout, so the packs players already use work unchanged.
Point `COREGS_HW_TEXTURES` at the `Textures` folder (GT4Recomp uses
`Documents\Gran Turismo 4\Textures`) and drop a pack into it or into any subfolder: every loose
`.png` or `.dds` with a replacement key in its name is picked up. `COREGS_HW_DUMP_TEXTURES=1` writes
textures with that same naming, ready to be edited and dropped back in. The exact names are in
[docs/FEATURES.md](docs/FEATURES.md#texture-packs).

Detailed support matrix and GPU requirements: [docs/FEATURES.md](docs/FEATURES.md).

## Quick start

Requirements: CMake 3.24+, a C++20 compiler, the Vulkan SDK (headers, loader and
`glslangValidator`). Windows is the primary platform; Linux builds the renderer and the fixtures
(the windowed front end is Windows-only for now). **Linux CI: added, first run pending.**

```sh
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build --output-on-failure
```

On Windows with the Visual Studio Build Tools and clang-cl:

```sh
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER="C:/Program Files/LLVM/bin/clang-cl.exe"
cmake --build build
```

Then run the example host: it opens a window, draws a synthetic GS stream and presents it.

```sh
build/examples/host/coregs_example_host.exe --frames 600
build/examples/host/coregs_example_host.exe --frames 30 --headless
build/examples/host/coregs_example_host.exe --profile gt4     # the GT4 rules on top
```

## Integrating in five steps

```cpp
#include "coregs/coregs.h"

coregs::Renderer gs;

// 1. Create the renderer and choose the options.
coregs::Options options;
options.scale = 2;                       // internal resolution: 1x .. 12x
std::string error;
if (!gs.init(options, error)) return fail(error);

// 2. Attach a window (or skip it for headless).
if (!gs.attach_window(hwnd, client_width, client_height, error)) return fail(error);

// 3. Feed the GS traffic of each guest frame, then flush at VBlank.
gs.write_privileged(0x12000000u, pmode);        // PMODE, DISPFB, DISPLAY, SMODE, ...
gs.gif_transfer(1, packet_bytes, byte_count);   // GIF paths 1, 2, 3
gs.flush();                                     // end of the guest frame
gs.next_frame();
gs.present(coregs::PresentPhase::progressive);  // one present per guest VBlank

// 4. Readbacks: the game's LOCAL->HOST transfers come out of read_transfer_fifo().
std::uint32_t words[8];                       // 32 bytes, two GIF qwords
const std::size_t got = gs.read_transfer_fifo(words, sizeof words);  // bytes of real data

// 5. Shut down: let the GPU finish, then detach the window.
gs.wait_idle();
gs.detach_window();
```

Readbacks are synchronous: `read_transfer_fifo()` returns the words of the pending transfer and the
renderer has already made every earlier draw visible (it records a readback of the touched targets).
See [docs/INTEGRATION.md](docs/INTEGRATION.md) for the exact contract, threading rules and timing.

## Documentation

| Document | What it covers |
| --- | --- |
| [docs/INTEGRATION.md](docs/INTEGRATION.md) | Connecting CoreGS to a game runtime: call order, threading, readbacks, present, options, the example host |
| [docs/FEATURES.md](docs/FEATURES.md) | Everything the renderer supports today, marked Supported / Partial / Not yet, and the GPU requirements |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | How the renderer is built: front end, batches, translation, targets, GPU drawer, display |
| [docs/PROFILES.md](docs/PROFILES.md) | Per-game profiles: the rules, the GT4 example, how to add one |
| [CONTRIBUTING.md](CONTRIBUTING.md) | Clean-room rule, adding a profile, reporting a game that renders wrong |

## Repository layout

```
include/coregs/     the public headers: coregs.h, options.h, profile.h
src/hw_gs_backend.cpp  device, swapchain, present scheduler (Windows front end)
src/core2/          the CoreGS 2.0 renderer: targets, readbacks, texture cache, GPU drawer
src/frontend/       GIF parser, registers, transfers, batching
src/memory/         GS local memory, swizzles, page/block identities
src/clut/           the CLUT unit (CSM1/CSM2, TEXCLUT)
src/pixel/          the per-pixel reference (blend, test, fog, dither, depth)
src/display/        display registers, merge, deinterlace, post effects
src/cache/          surface-cache bookkeeping (reference path)
src/decide/         per-draw decisions (clears, barriers, shuffles) for the reference path
src/lod/            mipmaps and the LOD formula
src/replace/        texture-replacement packs
src/writeback/      batch translation and the GPU drawer shared by both paths
src/shaders/        GLSL kernels, compiled to embedded SPIR-V at build time
profiles/gt4/       the Gran Turismo 4 profile (the reference example)
examples/host/      the example host
tests/              self-contained fixtures (synthetic GS streams, no game data)
tools/              the replay tool
```

## Building details

- **Shaders** are compiled at build time by `glslangValidator`. Set `COREGS_GLSLANG` (default:
  `$VULKAN_SDK/Bin/glslangValidator`) and keep the pinned SHA-256, or set `COREGS_GLSLANG_PIN=OFF`
  deliberately on a machine whose package differs (the Linux CI job does this and records the version;
  the Windows job installs the pinned Vulkan SDK and checks the tool against the pin).
- **Vulkan** headers and loader come from the SDK on Windows (`COREGS_VULKAN_SDK`, default
  `$VULKAN_SDK`) and from `find_package(Vulkan)` elsewhere.
- **Options** are fields of `coregs::Options`; the same fields exist as `COREGS_HW_*`
  environment variables for hosts that prefer a settings file. `describe_options()` prints them.
- **Diagnostics** such as `COREGS_CORE2_*` switches are documented next to the code; they are for
  A/B testing and never change the default behaviour.

## Licence and credits

Apache License 2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).

CoreGS was written for [GT4Recomp](https://github.com/moesuito/gt4-staticrecomp), the static
recompilation of Gran Turismo 4, from that project's own clean-room specification of the PS2 GS.
Thank you to the GT4Recomp contributors and to the PS2 homebrew and preservation community whose
public documentation of the GS made the clean-room work possible.

The licence requires the notice to travel with the code: if you distribute a product built with
CoreGS, reproduce [NOTICE](NOTICE) in your documentation or materials. You are also asked to credit
CoreGS visibly — in your product's About screen, credits or documentation — together with
[GT4Recomp](https://github.com/moesuito/gt4-staticrecomp), where users can see it.

Third-party: [stb_image](https://github.com/nothings/stb) (MIT/public domain) and
[xxHash](https://github.com/Cyan4973/xxHash) (BSD-2-Clause), used by texture replacement; the
presentation resampling shader follows AMD FidelityFX Super Resolution 1.0 (MIT). Their notices are
reproduced in [NOTICE](NOTICE).
