# What CoreGS supports

This page is the honest status of the renderer. The default path is the **CoreGS 2.0** core
(`src/core2/`), which renders every target at the internal scale and keeps a 1x copy of GS memory.
The older **reference write-back** core (`src/writeback/`) is used by the conformance fixtures and
by the replay tool; it is not the production path and is not described here except where noted.

Legend: **Supported** = implemented and covered by a fixture; **Partial** = implemented with a
documented limit; **Not yet** = not implemented, with the workaround a host can use.

## GS front end

| Feature | Status | Notes |
| --- | --- | --- |
| GIF paths 1, 2, 3 | Supported | Packed, reglist and image packets; transfers may split at any qword. |
| PATH1 / PATH3 streaming | Supported | The packet parser keeps per-path state. |
| SIGNAL / FINISH / label events | Supported | Delivered through `set_stream_event_handler`; never invented. |
| PATH2 | Supported (functional) | Accepted and parsed; path 2 has no independent FIFO depth model. |
| Malformed packets | Supported | Refused and counted; never silently guessed. |
| Interlaced and progressive VSync | Supported | `set_display_field` and `PresentPhase` drive the field logic. |

## Primitives and pixel pipeline

| Feature | Status | Notes |
| --- | --- | --- |
| Points, lines, line strips, triangles, strips, fans, sprites | Supported | Sprite corner rules, provoking vertex, flat and Gouraud shading. |
| 16-bit and 24-bit vertices | Supported | Including the packed XYZF/XYZ2 kick forms. |
| Alpha blending (all ALPHA register modes) | Supported | The modes a GPU cannot express natively are evaluated in the pixel shader; `Blend` picks the accuracy level. |
| Alpha test (all comparisons and fail actions) | Supported | Including the depth-only fail action. |
| Destination alpha test (DATE) | Supported | Interlock path where available; ordered fallback otherwise. |
| Fog (linear and exponential, all modes) | Supported | FOGCOL and the per-vertex fog value. |
| Dithering (DTHE, matrix and fine modes) | Supported | Off, native or fine. |
| TFX (modulate, decal, highlight, highlight2) | Supported | All four. |
| Colour clamping (COLCLAMP) and channel write masks (FBMSK) | Supported | Per-channel. |
| SCANMSK (interlace row masking) | Supported | Even/odd row masking. |
| Test and ZBUF registers, ZTST/ZMSK | Supported | 32/24/16-bit depth, GEQUAL/LESS/ALWAYS/NEVER and the rest. |
| Fill and copy primitives | Supported | Including whole-target clears and constant fills. |
| SPRITE size limits | Supported | The hardware's 1×1 to 256×256 range; larger sprites are split as the hardware does. |

## Textures, CLUT and formats

| Feature | Status | Notes |
| --- | --- | --- |
| PSMCT32, PSMCT24, PSMCT16, PSMCT16S | Supported | As render targets and as textures. |
| PSMZ32, PSMZ24, PSMZ16, PSMZ16S | Supported | As depth targets; Z16S/Z16 depth reads through the GLM. |
| T8, T4, T8H, T4HL, T4HH | Supported | Indexed, through CPU palettes or the GPU CLUT. |
| Texture wrapping (REPEAT, CLAMP, REGION_CLAMP, REGION_REPEAT) | Supported | All modes, including the region-repeat fix bits. |
| Texture filtering | Supported | Nearest/bilinear as the game asks; forced modes for modern looks. |
| Mipmaps and LOD | Supported | MIPTBP0/1, MMIN/MMAG, LCM, trilinear; the LOD formula is in `src/lod/`. |
| CLUT: CSM1, CSM2, TEXCLUT | Supported | Palette loads, copies, CSM2 offsets. |
| Palettes drawn by the game into targets | Supported | Loaded on the GPU from the drawn target, no CPU readback. |
| Palette segments (one T8 batch, several palettes) | Partial | A profile rule (GT4 example); experimental and off by default. |
| Texture cache | Supported | Keyed by address, format, extent, TEXA and palette contents. |
| Texture replacement packs | Supported | PNG images keyed by the game's texture key, pack folders, texture dumps. |

## Render targets, readbacks and memory

| Feature | Status | Notes |
| --- | --- | --- |
| One image per target at the internal scale | Supported | Colour and depth; targets grow when the game changes the size. |
| Dirty rectangles and page identities | Supported | Keeps texture decodes and readbacks exact. |
| Full target clears and constant fills | Supported | Compute clears when they preserve every byte; otherwise drawn. |
| LOCAL→HOST reads (TRXDIR 1) | Supported | Synchronous and exact; the contract is in [INTEGRATION.md](INTEGRATION.md). |
| Deferred readbacks | Supported (opt-in) | `COREGS_CORE2_DELAYED_DOWNLOADS=1` answers a repeated download one frame late. |
| Host→local uploads while a target is resident | Supported | Patched on the GPU, no CPU round trip. |
| GPU memory mirror (GLM) for textures | Supported | Textures over newer target data decode on the GPU. |
| Whole-memory view (`map_vram`) | Supported | Coherent with recorded work; valid until the next memory-changing call. |
| Reset and clear VRAM | Supported | See `reset_context()` and `clear_vram()`. |

## Scaling, output and presentation

| Feature | Status | Notes |
| --- | --- | --- |
| Internal resolution 1x–12x | Supported | Clamped to `maxImageDimension2D / 1280`; integer scaling, half-pixel modes. |
| Presentation upscale to the window | Supported | Lanczos-2 per axis, or FSR1 EASU when both axes grow; RCAS sharpening. |
| 16:9 rectangle and display-aspect fill | Supported | Ultrawide and super-ultrawide windows; 21:9, 32:9, arbitrary ratios. |
| Frame-rate independent presentation | Supported | `end_frame` + `present_slot`; held frames, left-out frames. |
| FIFO / mailbox / immediate present | Supported | Fallback to the closest mode the surface offers. |
| Deinterlacing | Supported | Auto, off, weave, bob, blend, adaptive; bottom-field-first. |
| FXAA | Supported | Edge-only, presentation stage. |
| Shadeboost, scanlines, CAS | Supported | Presentation post effects. |
| Frame dumps (PPM) and scanout | Supported | `vram`, `scanout` and `upscaled` sources. |
| HUD / performance overlay | Not included | It belongs to the host, not the renderer; GT4Recomp keeps its own. |
| Windowed front end on Linux | Not yet | The renderer and fixtures build on Linux; the window surface path is Windows-only today. |
| Android | Not yet | The core is platform-neutral Vulkan; the surface/present path is not written. |

## GPU requirements

- **Vulkan 1.2** with a graphics queue and `VK_KHR_swapchain`.
- **Dual-source blending** (`dualSrcBlend`) and **fragment stores and atomics**
  (`fragmentStoresAndAtomics`).
- Storage images for `R32_UINT` (targets), `R8G8B8A8` (decoded textures, replacement images) and the
  depth formats the game uses; compute and graphics pipelines share them.

Optional, with fallbacks:

| Item | Used for | Without it |
| --- | --- | --- |
| `VK_EXT_fragment_shader_interlock` | exact ordering of overlapping fragments inside one draw | primitives that may share a pixel are ordered by separate waits/draws |
| `largePoints` | points wider than 1 px at scale | each point is drawn as two triangles |
| `shaderStorageImageArrayDynamicIndexing` | deinterlace history arrays | a constant-index variant of the kernel |
| `VK_EXT_memory_budget` | ring and image budgets from the device's heaps | conservative fixed budgets |
| `VK_EXT_line_rasterization`, `VK_EXT_provoking_vertex` | exact line and provoking-vertex rules | the Vulkan core rules; reported in `DeviceReport` |

The renderer never silently falls back to a different *picture*: an optional feature only changes
how a result is produced, never what the hardware rules say. A device missing a required item is
refused with the list in `DeviceReport::missing_required`.

### Mobile notes

The core is written against desktop Vulkan 1.2. A mobile GPU needs the same required features and a
Vulkan 1.2 driver; the presentation path and the Android surface are not written yet. The image and
ring budgets already adapt to an integrated GPU (the device profile reports it), so the renderer
does not assume a discrete card's memory.

## Known limits

- **Other games are untested.** CoreGS was developed and validated against Gran Turismo 4. Its
  hardware rules are generic, but every game finds its own edge cases. Expect to test and, often,
  to add a profile rule or a fixture for a game that renders wrong.
- **The tone-run HLE is a GT4 rule.** A channel-shuffle run is drawn normally by the generic core.
  The GT4 profile substitutes it; another game with a different use of the same technique must not
  inherit that rule.
- **The palette-segment path is experimental.** It is off by default and preserved as a profile rule
  because a measured GT4 scene did not gain from it.
- **The reference write-back core is not feature-complete** (it is the conformance oracle, not a
  renderer to ship). CoreGS 2.0 is the renderer.
