// MIT License. Copyright (c) 2026 João Alano and GT4Recomp contributors.
// Written from docs/hw-renderer-spec (TASK-027 M4, chapters 03 and 15), clean room (decision 0069).
//
// Vulkan-free interface of the HWR reference GPU drawer (write-back mode). One Execute() call draws
// one chunk of a guest batch: frame and depth words are uploaded from the shadow local memory, the
// primitives are drawn with the uber shader, and the frame and depth words are read back so the
// caller can write them to the SLM (chapter 00 reference mode). Only the implementation file
// includes Vulkan headers.
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "../hw_device_profile.hpp"

namespace coregs {
class QueueLock;  // hw_queue_lock.hpp
}

namespace coregs::gpu {

// Vulkan handles owned by the caller (the HWR backend). Passed as void* to keep this header Vulkan-free.
struct VulkanHandles {
    void *instance = nullptr;
    void *physical = nullptr;
    void *device = nullptr;
    void *queue = nullptr;
    std::uint32_t queueFamily = 0;
    // The device was created with VK_EXT_fragment_shader_interlock and fragmentShaderPixelInterlock (IN-345).
    bool pixelInterlock = false;
    // The device was created with independentBlend (IN-481: per-attachment write masks of the ROP path).
    bool independentBlend = false;
    bool dualSourceBlend = false;  // IN-482: dualSrcBlend
    // TASK-174: set when the presentation thread shares `queue` (a device with one queue in its graphics family). Every
    // vkQueueSubmit and vkDeviceWaitIdle on it then runs inside this lock (Vulkan external synchronization); null otherwise.
    coregs::QueueLock *queueLock = nullptr;
    // TASK-179: the capability probe of the device (vulkan_loader::probe_device_profile), with what it was created with
    // (largePoints, shaderStorageImageArrayDynamicIndexing). Not probed (`profile.probed` false): the renderers take the
    // paths that need neither feature.
    coregs::DeviceProfile profile;
};

// Vertex record of chapter 03.3 (32 bytes). X and Y are 12.4 two's complement values.
struct GpuVertex {
    float s;
    float t;
    std::uint8_t rgba[4];
    float q;
    std::uint16_t x;
    std::uint16_t y;
    std::uint32_t z;
    std::uint16_t u;
    std::uint16_t v;
    std::uint8_t fog;
    std::uint8_t pad;
    // TASK-168 (core 2 only, not read by the old core's pipelines): the palette row of this vertex's primitive for an
    // indexed memory texture, plus one; 0 = the batch's row (DrawBatch::paletteRow). A batch with several palette
    // segments gives the vertices of its later segments their own rows.
    std::uint16_t paletteRow;
};
static_assert(sizeof(GpuVertex) == 32, "GpuVertex must match the 32-byte record of chapter 03.3");
static_assert(offsetof(GpuVertex, fog) == 28 && offsetof(GpuVertex, paletteRow) == 30,
              "the vertex input of the pipelines reads F at byte 28 and the palette row at byte 30");

enum class GpuTopology : std::uint8_t { Points, Lines, Triangles };

// One primitive, drawn with its own scissor (its coverage rectangle clipped to the scissor).
struct GpuDraw {
    GpuTopology topology = GpuTopology::Triangles;
    std::uint32_t first = 0;     // first vertex in GpuChunk::vertices
    std::uint32_t count = 0;     // vertices of this primitive
    std::int32_t x0 = 0, y0 = 0; // inclusive scissor rectangle in GS pixels
    std::int32_t x1 = 0, y1 = 0;
    // IN-397: the scissor of the scaled pass in native pixels (inclusive), the 1x one widened by the IN-391 slack and
    // clipped to the GS scissor. Empty (the default) = the 1x rectangle.
    std::int32_t scaledX0 = 0, scaledY0 = 0;
    std::int32_t scaledX1 = -1, scaledY1 = -1;
};

// Pixel-stage state, the uniform block of hw_pixel_uber.frag (std140, all 4-byte fields).
// Field order must match PixelStateBlock in shaders/hw_pixel_uber.frag.
struct GpuPixelState {
    std::uint32_t fields[48] = {};
};

// Field indices of GpuPixelState (names follow the shader block).
namespace gpufield {
enum : std::uint32_t {
    Format, Fbmsk, Fba, Dither, DimxLo, DimxHi, Ate, Atst, Afail, Aref, DateOn, Datm, Ztest, Ztst, Zmsk,
    Zformat, BlendOn, Pabe, BlendA, BlendB, BlendC, BlendD, Fix, Colclamp, TexOn, Tfx, Tcc, TexelKind,
    Ta0, Ta1, Aem, FogOn, FogR, FogG, FogB, TexFst, TexLinear, WrapS, WrapT, WrapMinS, WrapMaxS,
    WrapMinT, WrapMaxT, TexLog2W, TexLog2H, TexOriginX, TexOriginY,
    KeepTopByte,  // CORE2: a CT24 write keeps the stored top byte (shared 32/24 target)
    Count
};
static_assert(Count <= 48, "GpuPixelState has 48 fields");
} // namespace gpufield

// One chunk of work. The frame and depth rectangles are inclusive and equal; words are the raw
// stored values of the 32-bit view (SLM readPixel), row-major with width = rect width.
// Target slots (IN-344): GPU images that hold resident colour and depth targets. The last slot of each kind is
// the write-back mode's scratch pair, so a write-back batch never disturbs a resident target.
constexpr std::uint32_t kGpuColourSlots = 16;  // IN-400: the game uses many small targets per frame (was 6)
constexpr std::uint32_t kGpuDepthSlots = 8;     // IN-400 (was 3)
constexpr std::uint32_t kGpuScratchColourSlot = kGpuColourSlots - 1;
constexpr std::uint32_t kGpuScratchDepthSlot = kGpuDepthSlots - 1;

// The four transfer flags select what Execute moves between the words and the frame/depth images over the
// rectangle: the write-back mode sets all four (upload, draw, read back); the resident-target cache (M7,
// IN-340) uploads only when a target is loaded, draws with no transfer, and reads back only when the shadow
// memory must become coherent (a chunk with no draws is a pure transfer).
struct GpuChunk {
    std::int32_t rectX0 = 0, rectY0 = 0, rectX1 = -1, rectY1 = -1;
    std::uint32_t colourSlot = 0;  // kGpuColourSlots images; the write-back mode uses the scratch slots
    std::uint32_t depthSlot = 0;
    bool uploadFrame = true;
    bool uploadDepth = true;
    bool readbackFrame = true;
    bool readbackDepth = true;
    std::vector<std::uint32_t> frameWords;   // uploaded and/or read back per the flags
    std::vector<std::uint32_t> depthWords;
    bool hasTexture = false;
    // IN-346: when >= 0, the texels are copied on the GPU from this colour slot's image (rectangle 0,0 to
    // texWidth x texHeight) instead of being uploaded from `texels`.
    std::int32_t textureSlot = -1;
    // IN-384: the target's native row width (its FBW * 64), or 0 for a chunk that does not name a resident target.
    // The scaled images of this chunk's slot pair grow to cover it and the rectangle's rows before it is recorded.
    std::uint32_t targetWidth = 0;
    std::uint32_t texWidth = 0;              // texels (power of two up to 1024 or region extent)
    std::uint32_t texHeight = 0;
    // IN-384: the sampled footprint of a texture taken from a resident slot (IN-380), 0 = the whole extent. Only the
    // scaled copy uses it: the scaled texel image and the source slot's image grow to it, not to the 2^TW x 2^TH view.
    std::uint32_t texCopyWidth = 0;
    std::uint32_t texCopyHeight = 0;
    std::vector<std::uint32_t> texels;       // RGBA8 packed as R | G<<8 | B<<16 | A<<24
    // IN-347: when set, the texels are decoded on the GPU from the GLM (texture view below; indexed formats
    // through `palette`) instead of being uploaded or copied. The caller made the GLM current first.
    bool glmTexture = false;
    std::uint32_t glmBp = 0, glmBw = 0, glmPsm = 0;
    bool glmIndexed = false;
    std::vector<std::uint32_t> palette;      // 256 RGBA8 entries when glmIndexed (CPU CLUT)
    // IN-349: indexed texels expanded from the GPU CLUT buffer instead of `palette` (Decode's CPSM, CSA, TEXA).
    bool glmGpuClut = false;
    std::uint32_t glmCpsm = 0, glmCsa = 0, glmTa0 = 0, glmTa1 = 0, glmAem = 0;
    GpuPixelState state;
    std::vector<GpuVertex> vertices;
    std::vector<GpuDraw> draws;
    // IN-365: when set, the chunk's vertices (and texels) are read from these borrowed arrays instead of the vectors
    // above (no per-chunk copy); they must stay valid until Execute returns.
    const GpuVertex *vertexSpan = nullptr;
    std::size_t vertexSpanCount = 0;
    const std::uint32_t *texelSpan = nullptr;
    std::size_t texelSpanCount = 0;
    const GpuVertex *VertexData() const { return vertexSpan ? vertexSpan : vertices.data(); }
    std::size_t VertexCount() const { return vertexSpan ? vertexSpanCount : vertices.size(); }
    const std::uint32_t *TexelData() const { return texelSpan ? texelSpan : texels.data(); }
    std::size_t TexelCount() const { return texelSpan ? texelSpanCount : texels.size(); }
};

// M12 (chapter 06, IN-371): the scaled images of the target slots cover the native rectangle (0, 0) to
// (kGpuScaledExtent - 1, kGpuScaledExtent - 1) of each slot; draws and loads outside it reach only the 1x words.
constexpr std::int32_t kGpuScaledExtent = 1024;
constexpr std::uint32_t kGpuMaxScale = 12;  // COREGS_HW_SCALE range (06.1; owner 2026-10-27: up to 12x, 8K)
// Owner test 6 (IN-377): scaled copies of finished display buffers whose target slot was dropped (the game reuses the
// memory of the buffer it shows next as a read-only depth buffer, which drops the colour slot one frame before the
// buffer is displayed). Each shadow keeps the native rectangle (0, 0) to (kGpuScaledExtent - 1,
// kGpuShadowRows - 1) at scale with its 1x words; the scaled compose reads it as source kGpuColourSlots + shadow
// wherever the GLM still holds the saved word (per pixel), and the GLM at 1x elsewhere.
constexpr std::uint32_t kGpuDisplayShadows = 2;
// IN-381: the deinterlace images of the scaled presentation hold up to kGpuDeinterlaceWidth x kGpuDeinterlaceRows
// native pixels (NTSC and PAL 480i/576i frames); a larger interlaced frame keeps the 1x CPU presentation.
constexpr std::uint32_t kGpuDeinterlaceWidth = 640;
constexpr std::uint32_t kGpuDeinterlaceRows = 512;
constexpr std::uint32_t kGpuDeinterlaceHistory = 4;
constexpr std::int32_t kGpuShadowRows = 512;

// Limits of one Execute() call (the caller splits larger batches).
constexpr std::uint32_t kGpuMaxVertices = 1u << 20;   // 32 MiB of vertex data
constexpr std::int32_t kGpuSurface = 2048;             // GS window size: frame and depth images
constexpr std::uint32_t kGpuMaxTextureSide = 1024;     // largest texture side (region modes may use less)
constexpr std::uint32_t kGpuTexelAtlasSide = 4096;      // texel atlas: one region per textured chunk of a run (IN-352)

struct GpuStats {
    std::uint64_t executions = 0;
    std::uint64_t draws = 0;
    std::uint64_t vertices = 0;
    std::uint64_t uploadBytes = 0;
    std::uint64_t readbackBytes = 0;
    std::uint64_t barriers = 0;
    std::uint64_t submissions = 0;  // queue submissions (IN-342: several chunks per submission)
    std::uint64_t runs = 0;         // render passes holding batched chunks (IN-352)
    std::uint64_t runChunks = 0;    // chunks drawn inside runs
    std::uint64_t earlySubmissions = 0;  // submissions in mid-frame (IN-358)
    std::uint64_t barriersElided = 0;    // full barriers skipped right after another (IN-368)
    std::uint64_t clutDispatches = 0;        // CLUT loads recorded on the GPU (after IN-370 collapsing)
    std::uint64_t clutMaterialisations = 0;  // times deferred CLUT loads were recorded (IN-370)
    std::uint64_t decodesGlmPalette = 0;     // indexed decodes that read the palette from the GLM (IN-370)
    std::uint64_t decodesReused = 0;     // GLM texture decodes skipped: same texture earlier in the run (IN-369)
    std::uint64_t renderPasses = 0;      // render pass instances (runs and per-chunk passes; IN-367 efficiency)
    // M12 upscaling (chapter 06, IN-371): draws repeated into the scaled images, scaled render passes, and
    // refreshes of a scaled image from its 1x words after a load (uploads and GLM loads).
    std::uint64_t scaledDraws = 0;
    std::uint64_t scaledPasses = 0;
    std::uint64_t scaledRefreshes = 0;
    std::uint64_t scaledRefreshCarries = 0;  // IN-398: refreshes that kept another view's scaled blocks
    std::uint64_t scaledRefreshPixels = 0;  // native pixels refreshed
    std::uint64_t scaledTextureCopies = 0;  // IN-374: textures copied from a slot's scaled image
    std::uint64_t fxaaFrames = 0;           // IN-376: presentation frames passed through FXAA
    std::uint64_t scaledShadowSaves = 0;    // IN-377: display shadows saved from a dropped slot's scaled image
    std::uint64_t toneCarries = 0;          // IN-379: scaled images rebuilt after a tone-curve sequence
    std::uint64_t scaledDeinterlaces = 0;   // IN-381: scaled presentation frames deinterlaced on the GPU
    std::uint64_t scaledDrawsPaused = 0;    // IN-382: scaled draws not recorded while the scaled work is paused
    std::uint64_t scaledRefreshesPaused = 0;
    std::uint64_t toneStripPages = 0;       // IN-383: tone-sequence pages written by one strip dispatch
    std::uint64_t toneStripDispatches = 0;  // IN-383: strip dispatches (one per page)
    std::uint64_t toneStripBatches = 0;     // IN-383: strip batches folded into those dispatches
    // IN-384 (parity plan Part A): the scaled images only cover what their slot uses, and are grown on demand.
    std::uint64_t scaledImageGrows = 0;
    std::uint64_t scaledDrawsClipped = 0;   // scaled draws whose scissor left a pair image (0 expected: must stay 0)
    std::uint64_t scaledImageBytes = 0;     // device memory of every scaled image (slots, shadows, tone)
    // COREGS_HW_GPU_TIMING (12.8, roadmap requirement 4): GPU time in ns from timestamps written after every
    // recorded operation; the interval since the previous timestamp is charged to the operation's category
    // (so a barrier wait before an operation counts with it). busy = first to last timestamp per submission.
    // pass (1x draws), decode, clut, swizzle (both directions), upload, download (readbacks and the presentation
    // compute: compose, tone carry, deinterlace, FXAA), scaled (the M12 scaled passes), refresh (scaled-image refreshes
    // from 1x words).
    std::uint64_t gpuNs[8] = {};
    std::uint64_t gpuBusyNs = 0;
};

class GpuDrawer {
public:
    GpuDrawer();
    ~GpuDrawer();
    GpuDrawer(const GpuDrawer &) = delete;
    GpuDrawer &operator=(const GpuDrawer &) = delete;

    // M12 (chapter 06, IN-371): internal resolution scale s (1..kGpuMaxScale), set before Init. With s > 1 every
    // target slot also gets an image of kGpuScaledExtent * s pixels per side; every draw into a slot is rasterised a
    // second time into it (vertex positions times s, 06.3 half-pixel shift `halfPixelShift` in scaled pixels times 2,
    // so 1.0 * s is the native mode), and every load of 1x words into a slot refreshes it. The 1x words stay the
    // only authority: memory, the GLM, readbacks and every 1x gate are unchanged at any scale.
    void SetScale(std::uint32_t scale, float scaleOffset);
    std::uint32_t Scale() const;

    // Creates the shader modules, pipelines, images and buffers. Fails with a readable message.
    bool Init(const VulkanHandles &handles, std::string &error);
    bool Ready() const;

    // Records one chunk (uploads, draws, readbacks as its flags say). A chunk with a readback submits and waits,
    // so its words are filled on return; other chunks are submitted later (IN-342), when a ring is full, a
    // readback needs the results, or Flush() is called.
    bool Execute(GpuChunk &chunk, std::string &error);

    // Submits the recorded chunks; with `wait`, also waits for them.
    bool Flush(bool wait, std::string &error);

    // ---- GLM (M10, chapter 04.9, IN-347): a 4 MiB GPU mirror of GS local memory ----
    // Records copies of the given byte ranges of `localMemory` (the 4 MiB shadow memory) into the GLM.
    bool UploadGlm(const std::uint8_t *localMemory, const std::vector<std::pair<std::uint32_t, std::uint32_t>> &ranges,
                   std::string &error);
    // Records the swizzle of a resident target's rectangle (inclusive) into the GLM at its own (BP, BW, PSM).
    bool SwizzleToGlm(bool depthSlot, std::uint32_t slot, std::uint32_t bp, std::uint32_t bw, std::uint32_t psm,
                      std::int32_t x0, std::int32_t y0, std::int32_t x1, std::int32_t y1, std::string &error);
    // Records the inverse (IN-350): the slot's rectangle receives the stored words of (BP, BW, PSM) from the GLM.
    bool LoadSlotFromGlm(bool depthSlot, std::uint32_t slot, std::uint32_t bp, std::uint32_t bw, std::uint32_t psm,
                         std::int32_t x0, std::int32_t y0, std::int32_t x1, std::int32_t y1, std::string &error);
    // Records a merge of words into the GLM (IN-350): `triples` holds (GLM word index, value, bit mask) per word;
    // the masked bits of each word take the value. Word indices must be unique.
    bool MergeIntoGlm(const std::vector<std::uint32_t> &triples, std::string &error);

    // Records an EE->GS upload into the GLM (IN-351): `pixels` texels of `bpp` bits from `data` (host order) are
    // written at (x0 + i % width, y0 + i / width) of (BP, BW, PSM), each with only its own bits. The rectangle
    // must not wrap onto itself inside the buffer (distinct addresses).
    bool UploadHostToGlm(std::uint32_t bp, std::uint32_t bw, std::uint32_t psm, std::uint32_t x0, std::uint32_t y0,
                         std::uint32_t width, std::uint32_t pixels, std::uint32_t bpp, const std::uint8_t *data,
                         std::size_t byteCount, std::string &error);
    // IN-355 asynchronous downloads: the copy of the given GLM byte ranges is recorded into the current
    // recording (it runs after everything recorded before it) into one of two host buffers; Complete waits only
    // for the submission that holds it (submitting it first if it is still being recorded) and copies the bytes
    // into `localMemory` at the same offsets. A ticket's buffer is reused two downloads later, so a ticket must
    // be completed (or dropped) before the second next one is recorded.
    struct DownloadTicket {
        bool valid = false;
        std::uint32_t buffer = 0;
        std::uint64_t serial = 0;  // the submission that holds the copy
    };
    DownloadTicket RecordGlmDownload(const std::vector<std::pair<std::uint32_t, std::uint32_t>> &ranges,
                                     std::string &error);
    bool CompleteGlmDownload(const DownloadTicket &ticket, std::uint8_t *localMemory, std::string &error);
    // IN-364: records the display compose kernel (13.3, progressive frames) into the display image: `parameters`
    // is the block hw_display_compose.comp reads (background, merge alpha, the two circuits' sources and
    // placements), out_width x out_height pixels. The caller made the GLM current for the circuits.
    bool ComposeDisplay(const std::vector<std::uint32_t> &parameters, std::uint32_t outWidth, std::uint32_t outHeight,
                        std::string &error);
    // M12 (IN-372): the same compose at s x s output pixels per native pixel into the display image (sized for it).
    // The block holds 13 words per circuit (RC1 at 5, RC2 at 18): the 12 of ComposeDisplay plus the colour slot
    // whose scaled image the circuit reads, plus one (0: read the GLM at 1x). out_* are scaled pixels.
    bool ComposeDisplayScaled(const std::vector<std::uint32_t> &parameters, std::uint32_t outWidth,
                              std::uint32_t outHeight, std::string &error);
    // IN-377: records a copy of colour slot `slot`'s scaled image over the native rectangle (x0, y0) - (x1, y1)
    // (inclusive, clipped to the shadow) into display shadow `shadow`, after every draw recorded before it. The
    // scaled compose reads that shadow when its circuit word is kGpuColourSlots + shadow + 1.
    bool SaveScaledShadow(std::uint32_t slot, std::uint32_t shadow, std::int32_t x0, std::int32_t y0, std::int32_t x1,
                          std::int32_t y1, std::string &error);
    // IN-379 (09.4 variant A at scale): SaveToneSource copies colour slot `slot`'s scaled image and 1x words over the
    // native rectangle (clipped to kGpuScaledExtent x kGpuShadowRows) before the game's tone-curve sequence reads it at 1x;
    // CarryTone then rebuilds colour slot `slot`'s scaled image over the rectangle from that copy through the per-
    // channel table that maps the saved 1x words to the slot's current 1x words (hw_scale_tone_carry.comp). Only the
    // scaled image changes.
    bool SaveToneSource(std::uint32_t slot, std::int32_t x0, std::int32_t y0, std::int32_t x1, std::int32_t y1,
                        std::string &error);
    bool CarryTone(std::uint32_t slot, std::int32_t x0, std::int32_t y0, std::int32_t x1, std::int32_t y1,
                   std::string &error);
    // IN-381 (13.4 at scale): deinterlaces the top-left width x height pixels of the display image in place, with
    // the persistent history the caller tracks (hw_deinterlace_scaled.comp). `mode`: 2 weave, 3 blend, 4 bob,
    // 5 adaptive. The display image is first copied into history slot `current`. History indices are
    // 0..kGpuDeinterlaceHistory-1, or kGpuDeinterlaceHistory for none.
    struct DeinterlaceStep {
        std::uint32_t mode = 5;
        std::uint32_t width = 0, height = 0;
        std::uint32_t rowParity = 0;
        std::uint32_t field = 0;
        std::uint32_t current = 0;
        std::uint32_t previousSame = kGpuDeinterlaceHistory;
        std::uint32_t previousOther = kGpuDeinterlaceHistory;
        std::uint32_t olderOther = kGpuDeinterlaceHistory;  // IN-396: the opposite-parity field three back
        bool haveOutput = false;
        bool initWoven = false;
        std::uint32_t thresholdPermille = 80;
        // IN-386: output rows per algorithm row. 1 = one output row, the reference's behaviour (a one-line field
        // mismatch becomes a sub-native comb); scale = one native row, the IN-381 interpretation (a full native-row
        // ghost at scale).
        std::uint32_t rowStep = 1;
    };
    bool DeinterlaceScaled(const DeinterlaceStep &step, std::string &error);
    // IN-382: while paused, draws are not repeated into the scaled images and loads do not refresh them (the caller
    // rebuilds or refreshes every scaled image the paused work touched). RefreshScaledSlot refreshes a slot's
    // scaled image over a native rectangle (inclusive) from its 1x words.
    void PauseScaled(bool paused);
    // IN-398: until ClearRefreshCarry, a refresh of a slot of this kind keeps the scaled block of `slot` wherever its
    // 1x word equals the loaded one in `mask` (the same memory under another view: CT32/CT24, Z32/Z24), inside
    // its valid native rectangle x0..x1, y0..y1. Presentation only: the 1x words are loaded as before.
    void SetRefreshCarry(bool depthSlot, std::uint32_t slot, std::uint32_t mask, std::int32_t x0, std::int32_t y0,
                         std::int32_t x1, std::int32_t y1);
    void ClearRefreshCarry();
    // Diagnostics: COREGS_HW_DRAW_TRACE / COREGS_HW_SCALED_TRACE write only while armed (the renderer arms them from the
    // VBlank named by COREGS_HW_TRACE_FROM_VBLANK; armed from the start without it).
    void ArmTrace(bool armed);
    bool RefreshScaledSlot(bool depthSlot, std::uint32_t slot, std::int32_t x0, std::int32_t y0, std::int32_t x1,
                           std::int32_t y1, std::string &error);
    // IN-383 (parity plan item 2, chapter 09.4 variant A): the eight decal T8 strip sprites that write one frame
    // buffer page back as CT24, in one dispatch. Every strip is eight wide and 32 tall with one texel per pixel, so
    // its exact effect is dest(x, y).rgb = palette(index(x % 8, y)).rgb with the alpha byte untouched; the palette is
    // the 16x16 quadrant of the temporary page its CBP names (07.4 CSM1) and the indices come from the strip's T8
    // texture. Both are read from the GLM (the caller made it current and owns the destination pages afterwards).
    struct ToneStripPage {
        std::uint32_t destBp = 0, destBw = 0;   // the destination page through its CT24 view
        std::int32_t destX = 0, destY = 0;      // page origin (64- and 32-aligned pixels)
        std::uint32_t sourceBp = 0, sourceBw = 0;  // the temporary page (CT32, 64 x 32)
        std::uint32_t indexBp = 0, indexBw = 0;    // the strip's T8 texture (8 x 32 texels)
        std::uint32_t stripCount = 0;              // 1..8
        std::uint32_t quadrants[8] = {};           // per strip: qx + 4 * qy of the palette quadrant
    };
    bool ToneStrips(const ToneStripPage &page, std::string &error);
    // IN-384 (parity plan Part A): the native extent of a slot's scaled image (0 x 0 when it has none yet) and
    // whether that image covers a native rectangle, which is what the scaled presentation compose needs.
    void ScaledExtent(bool depthSlot, std::uint32_t slot, std::uint32_t &width, std::uint32_t &height) const;
    bool ScaledCovers(bool depthSlot, std::uint32_t slot, std::int32_t x1, std::int32_t y1) const;
    std::uint64_t ScaledImageBytes() const;  // device memory of every scaled image (IN-384 reporting)
    // The display image (R32_UINT, RGBA8 packed R | G << 8 | B << 16 | A << 24, max(kGpuSurface,
    // kGpuScaledExtent * scale) square, GENERAL layout).
    // As an opaque handle (the VkImage value): this header carries no Vulkan types.
    std::uint64_t DisplayImage() const;
    // Fixtures: the top-left width x height words of the display image, or of the FXAA image (submits and waits).
    // Diagnostics (COREGS_HW_TONE_DUMP, COREGS_HW_DISPLAY_DUMP): the top-left width x height words of an image: 0 colour
    // slot `slot`'s scaled image, 1 the saved tone source (scaled), 2 colour slot `slot`'s 1x image, 3 display shadow
    // `slot`, 4 the composed display image (submits and waits).
    bool ReadImageDebug(std::uint32_t which, std::uint32_t slot, std::uint32_t width, std::uint32_t height,
                        std::vector<std::uint32_t> &words, std::string &error);
    bool ReadDisplay(std::uint32_t width, std::uint32_t height, std::vector<std::uint32_t> &words, std::string &error,
                     bool fromFxaa = false);
    // 13.5 FXAA (IN-376): records hw_postfx_fxaa.comp over the top-left width x height pixels of the display image
    // into the FXAA image (created on first use, same size and packing as the display image).
    bool ApplyFxaa(std::uint32_t width, std::uint32_t height, std::string &error);
    std::uint64_t FxaaImage() const;
    // Copies the given byte ranges of the GLM into `localMemory` at the same offsets (IN-350: pages whose newest
    // content lives only in the GLM, before the CPU reads or writes them). Submits and waits.
    bool DownloadGlm(const std::vector<std::pair<std::uint32_t, std::uint32_t>> &ranges, std::uint8_t *localMemory,
                     std::string &error);
    // Records a CLUT load from the GLM into the GPU CLUT buffer (chapter 07.7, IN-349): the same as
    // clut::ClutUnit::Copy with these TEX0/TEXCLUT fields (cbp in blocks; psm decides 16 or 256 entries).
    bool LoadClutFromGlm(std::uint32_t cbp, std::uint32_t cpsm, std::uint32_t csm, std::uint32_t csa, std::uint32_t psm,
                         std::uint32_t cbw, std::uint32_t cou, std::uint32_t cov, std::string &error);
    // Records an upload of the CPU CLUT buffer (512 halfwords) into the GPU CLUT buffer.
    bool UploadClut(const std::uint16_t *halfwords, std::string &error);
    // Fixtures: read the GPU CLUT buffer (512 halfwords; submits and waits).
    bool ReadClut(std::vector<std::uint16_t> &halfwords, std::string &error);
    // Fixtures: read the whole GLM, or the texel image's top-left width x height words (submits and waits).
    bool ReadGlm(std::vector<std::uint32_t> &words, std::string &error);
    bool ReadTexels(std::uint32_t width, std::uint32_t height, std::vector<std::uint32_t> &words, std::string &error);

    // IN-352 A/B switch: chunks of triangles drawn with the pixel interlock are batched into runs (one render
    // pass, texture preparation recorded before it, no barrier between the chunks). Off: one pass per chunk.
    void SetRuns(bool enabled);
    // IN-354 A/B switch: submissions alternate between two sets of rings and command buffers (CPU records the
    // next frame while the GPU executes the previous one). Off: every recording waits for the last submission.
    void SetDoubleBuffering(bool enabled);
    // COREGS_HW_GPU_TIMING: per-category GPU timestamps (GpuStats::gpuNs). Off by default.
    void SetGpuTiming(bool enabled);
    void SetDecodeReuse(bool enabled);  // IN-369
    void SetClutDefer(bool enabled);    // IN-370
    // IN-358 A/B: submit the recording every `chunks` chunks (0: only at VBlank, a full ring or a readback).
    void SetChunksPerSubmission(std::uint32_t chunks);

    const GpuStats &Stats() const;

private:
    bool read_image(std::uint64_t image, std::uint32_t width, std::uint32_t height, std::vector<std::uint32_t> &words,
                    std::string &error);
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace coregs::gpu
