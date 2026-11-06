# CoreGS architecture

CoreGS is a renderer for one hardware unit: the PlayStation 2 Graphics Synthesizer. It consumes the
GS stream a game produces and turns it into pictures. Everything between those two points is
organised so that each stage can be tested on its own and compared against the GS rules.

```
GIF packet bytes ──▶ front end ──▶ batches ──▶ translation ──▶ GPU drawer ──▶ targets (at scale)
register writes ────▶             (registers,   (vertices,       (storage          │
                                   CLUT,        pixel state,     images,           │
                                   vertices)    textures)        compute)          │
                                                                                   ▼
window ◀── present ◀── upload ◀── display image ◀── compose / deinterlace / post ◀── display registers
   ▲
   └── readbacks: targets → 1x words → GS memory → FIFO
```

## 1. The public surface

`include/coregs/coregs.h` declares `coregs::Renderer` and the data it exchanges with a host:
`Options`, `Stats`, `RenderCounters`, `DeviceReport`, `PresentPhase` and `ScanoutRequest`.
`include/coregs/profile.h` declares `GameProfile`, the optional content rules. Nothing in the public
headers exposes Vulkan: a host does not need the Vulkan SDK to include or link against them
(although the library does need the loader at run time).

The implementation of that surface is `src/hw_gs_backend.cpp` (device, swapchain, presenter) plus
the renderer cores. The backend owns the Vulkan instance and device; the renderer cores receive
`gpu::VulkanHandles` (instance, physical device, device, queue, queue family) and never create their
own device.

## 2. Front end: packets to batches

`src/frontend/` is pure C++ (no Vulkan, no platform headers) and is shared by both cores and by the
CPU reference.

- **`hw_gif_parser`** decodes GIF tags and consumes packed, reglist and image packets, per path.
  Packed descriptors cover PRIM, RGBAQ, STQ, UV, XYZF2/XYZ2, TEX0/CLAMP, FOG, the A+D form (a full
  128-bit register write) and the non-drawing XYZF3/XYZ3 kicks. Malformed streams are refused.
- **`hw_gs_registers`** keeps the privileged register set and the two context register sets
  (`TEX0/1`, `CLAMP/1`, `FRAME/1`, `ALPHA/1`, `TEST/1`, `ZBUF/1`, `XYOFFSET/1`, `PRMODECONT`).
  Writing PRIM switches context as the hardware does.
- **`hw_transfers`** implements the transfer engine: host→local (image packets), local→local moves
  and local→host reads (`TRXDIR`), with the block/page geometry and the overlap rules.
- **`hw_front_end`** turns register writes and vertex kicks into `Batch` objects: one primitive run
  with the state snapshot, the vertex list, the texture unit state and the palette segments. It
  applies the batching rules (state changes, scissor changes, transfers, auto-flush heuristics) and
  reports each batch to a `BatchSink`. It also models the CLUT unit (`src/clut/`) and calls the
  renderer's hooks when a CLUT load could change a pending batch.

A `Batch` is the unit every later stage works on. It carries `BatchState` (context registers,
effective PRIM, scissor, frame/z views) and the vertices; it does not carry GPU data.

## 3. GS memory

`src/memory/` models the 4 MiB of GS local memory as the hardware sees it:

- **Pages** (8 KiB) and **blocks** (256 bytes) with the per-format swizzle tables
  (`PSMCT32/24/16/16S`, `PSMZ32/24/16/16S`, `T8/T4/T8H/T4HL/T4HH`).
- **Page and block identities**: every write bumps a version, which is how the texture cache knows a
  decode is stale without hashing 4 MiB.
- **Buffer views** (base pointer, buffer width, format) and the rectangle-to-address mapping used by
  transfers, readbacks and texture decoding.
- **Local→host FIFO** state: the pending read, its rectangle, its position, and the overrun counter.

The memory model is deliberately CPU-side and exact: the GPU never writes GS memory behind its back
without telling it. CoreGS 2.0 keeps a 4x-sized *GPU* mirror (the GLM) for textures, but GS memory
itself remains the authority for what the guest reads back.

## 4. CoreGS 2.0 (`src/core2/`)

The production renderer.

### Targets

Every render target is **one image at the internal scale** (`s`), colour or depth, 32-bit words of
GS data (`R32_UINT`). A target is keyed by its `FRAME`/`ZBUF` view (base pointer, buffer width,
format) and holds:

- the image and its native size;
- the rectangles it has drawn and the rectangles that are valid (the valid area starts as the
  uploads the guest made and grows with draws);
- the page identities of its memory, so a texture or a readback can tell whether the target is
  newer than GS memory;
- dirty rectangles: writes the guest made through uploads or moves that the target has not patched
  yet.

A target grows when the game changes the size (a new image, content copied across at scale), is
replaced when the width changes or the whole content is dirtied, and is aged out after a number of
frames without drawing.

### Translation and drawing

`writeback::TranslateBatchHooked` (`src/writeback/hw_gs_writeback.cpp`, shared with the reference
core) turns a `Batch` into vertices, draw ranges, a pixel state and texture data. CoreGS 2.0 then:

- resolves the texture from a target when the target holds newer pixels (GPU copy), from the GLM
  when the texture overlaps newer target data (GPU decode), or from the CPU texture cache;
- patches any pending upload rectangles into the target before the draw that needs them (compute
  `core2_patch` / `core2_patch_glm`);
- records the draw into the GPU drawer, which turns draws that share no pixel into one render pass
  and keeps the rest in hardware order;
- runs the optional compute paths: constant fills (`core2_fill`), CLUT expansion
  (`core2_clut_target`), target materialisation into the GLM (`core2_materialize`) and the
  profile's tone pass (`core2_tone`).

Draws use an "uber" fragment shader (`src/shaders/hw_pixel_uber.frag`) that evaluates blending,
alpha test, DATE, fog, dithering, TFX and the frame-buffer mask against the storage images. Variants
exist for a plain (non-interlocked) path, a colour-attachment path, and a specialised path per draw
state; a device without `VK_EXT_fragment_shader_interlock` orders overlapping fragments with waits
instead.

### Readbacks

When the guest reads a rectangle (`TRXDIR = 1`), the renderer:

1. flushes the pending batch;
2. finds every target that overlaps the rectangle and holds newer pixels;
3. records a reduction of the native rectangle to 1x stored words (`core2_reduce`) and a copy into a
   readback buffer, then waits for that submission;
4. writes the words into GS memory and lets `read_transfer_fifo` copy them out.

Optional batching reduces several readbacks in one submission. The delayed-download mode keeps a
per-rectangle result and answers the next identical read from it, one frame late.

### Textures

The texture cache is keyed by `(TBP0, TBW, PSM, extent, TEXA, palette hash, mip hash)`. A cached
decode is valid while the page identities it read are unchanged. Indexed textures are expanded:

- on the CPU into a 256-entry palette image (palette rows), when the palette is a CPU one;
- on the GPU through the CLUT buffer (`hw_glm_clut`) when the palette was written by a GPU load;
- from a drawn target, through `core2_clut_target`.

Mip levels decode next to level 0 and are sampled by the shader through `TEX1`/`MIPTBP`.

### Display

`ComposeFrameGpu` builds the display image from the two display circuits (`DISPFB1/2`,
`DISPLAY1/2`, `PMODE`, `SMODE1/2`) with `core2_compose`: merge the two circuits (MMOD, ALP), apply
the field offset, the background colour and the 24/16-bit conversions. The result is:

- passed through FXAA (`hw_postfx_fxaa`) when asked;
- resampled to the present rectangle (`core2_present_scale`: Lanczos-2 or FSR1 EASU, then RCAS)
  when the window is not the native size;
- deinterlaced in place (`hw_deinterlace_scaled`) for interlaced VBlanks, with a four-entry history;
- read back for `scanout`/dumps, or presented from the GPU image.

## 5. The reference write-back core

`src/writeback/` contains the first, conformance-oriented core: it draws a batch by uploading the
touched frame and depth words, drawing, and writing the changed words back to GS memory. It is
slower and simpler, and it is the oracle the fixtures compare CoreGS 2.0 against. It is selected
only by `COREGS_HW_CORE=1`; the public `Renderer` always selects CoreGS 2.0. Keeping it in the tree
is deliberate: the pixel, target and readback rules are validated on both paths on every test run.

## 6. CPU references

Two CPU implementations keep the GPU honest:

- `src/pixel/` evaluates one pixel exactly: TFX, TEXA, fog, alpha test, blending, colour clamp,
  dither, frame-buffer mask, DATE, depth. The GPU shader must agree with it on the fixtures.
- `src/cpuref/` rasterises whole batches on the CPU (coverage, edges, sprites, textures, CLUT) and
  produces a reference image and GS memory. The GPU equivalence fixtures compare against it.

Neither is used on the normal rendering path.

## 7. Display and output (`src/display/`)

The display unit is pure CPU: video-mode decoding (`SMODE1/2`, `SYNCV`, `DISPLAY`, `DISPFB`,
`PMODE`), the two-circuit merge plan, the interlace field offset, deinterlace algorithms, shadeboost,
scanlines and the presentation rectangle arithmetic. CoreGS 2.0 uses the same plan for the GPU
compose and for the 1x reference compose; the reference core uses it for its CPU compose.

## 8. Backend and presentation (`src/hw_gs_backend.cpp`)

The backend owns:

- the Vulkan instance and device, chosen from the required features (Vulkan 1.2, swapchain, dual
  source blend, fragment stores and atomics, the storage formats) and the optional ones reported in
  `DeviceReport`;
- the swapchain, its format (8-bit UNORM, display-referred) and the present mode;
- the presentation scheduler: it keeps one image per guest VBlank, holds it when the host ticks
  faster than the guest, and leaves frames out when the host is behind, without growing a queue;
- the upload path for CPU-composed frames and the blit into the swapchain (with the pre-transform
  correction and the box-halving rule for upscaled frames);
- the Win32 window surface (`attach_window`).

The renderer never assumes it owns the device's queue for its whole lifetime: a host may pass a
shared queue (for a presentation thread); the renderer serialises its submissions with the queue
lock and reports the wait times in `Stats`.

## 9. Determinism and validation

- Every fixture is self-contained: a synthetic GS stream is replayed and the GPU result is compared
  with the CPU reference pixel by pixel, plus the GS memory after every frame.
- GPU tests exit 77 when no Vulkan 1.2 device exists; CTest reports them as skipped, never as
  passed.
- The reference write-back core and CoreGS 2.0 run the same streams; a difference is a regression.
- Undefined host behaviour is never used to stand in for guest behaviour: fixed-width arithmetic,
  explicit wrapping, defined shifts. Addresses and offsets are recorded in the code and the tests.

## 10. Source map

| Path | Contents |
| --- | --- |
| `src/frontend/` | GIF parser, registers, transfers, batching, `Batch` types |
| `src/memory/` | 4 MiB memory model, swizzles, pages, blocks, FIFO |
| `src/clut/` | CLUT unit: CSM1/CSM2, TEXCLUT, palette copies |
| `src/core2/` | CoreGS 2.0: targets, translation loop, GPU drawer, uploads, depth aliasing |
| `src/writeback/` | shared batch translation, GPU drawer wrapper, reference write-back core |
| `src/pixel/` | per-pixel reference |
| `src/decide/` | per-draw decisions for the reference path |
| `src/display/` | display registers, merge plan, deinterlace, post effects |
| `src/lod/` | mipmap layout and the LOD formula |
| `src/replace/` | texture-replacement packs, keys, PNG loading |
| `src/cpuref/` | CPU rasteriser and capture replay |
| `src/shaders/` | GLSL kernels and the build-time SPIR-V embedder |
| `profiles/gt4/` | the Gran Turismo 4 profile and its pattern classifier |
| `tests/` | CTest fixtures |
| `tools/` | replay tool |
| `examples/host/` | the example host |
