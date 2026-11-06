// MIT License. Copyright (c) 2026 João Alano and GT4Recomp contributors.
// Written from docs/hw-renderer-spec (TASK-027 M4, chapters 00 reference mode, 03.1 to 03.4, 03.7,
// 03.10, 12.3), clean room (decision 0069).
//
// The GPU side of the write-back reference mode. Each chunk is drawn synchronously: upload the
// frame and depth words and the texture, draw each primitive with its own scissor and a barrier
// between draws (storage image ordering, chapter 08.8 barrier policy), read the frame and depth
// words back. Vulkan entry points are called through pointers (see hw_gs_vulkan_loader.hpp).

// Vulkan create-info structs are initialised with sType only (pNext defaults to null by the spec).
#pragma clang diagnostic ignored "-Wmissing-field-initializers"

#include "hw_gs_writeback_gpu.hpp"

#include "hw_gs_vulkan_device_table.hpp"
#include "../hw_queue_lock.hpp"
#include "../memory/hw_memory.hpp"
#include "../shaders/hw_shader_blobs.hpp"

#include <algorithm>
#include <array>
#include <cstring>

namespace {
std::uint32_t FloatToBits(float value) {
  std::uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}
} // namespace

namespace coregs::gpu {
namespace {

constexpr std::uint32_t kSurfaceSize = static_cast<std::uint32_t>(kGpuSurface);  // 2048
constexpr std::uint32_t kTextureMax = kGpuMaxTextureSide;                       // texels per axis
constexpr VkDeviceSize kFrameBytes = static_cast<VkDeviceSize>(kSurfaceSize) * kSurfaceSize * 4;
// Staging, vertex and uniform memory are rings (IN-342): chunks are recorded into one command buffer and
// submitted when a readback needs the result, a ring is full, or the caller flushes (each VBlank). The staging
// ring holds at least one chunk of the largest size (two full surfaces and a full texture).
constexpr VkDeviceSize kStagingBytes = 2 * kFrameBytes + static_cast<VkDeviceSize>(kTextureMax) * kTextureMax * 4 +
                                       (static_cast<VkDeviceSize>(32) << 20);
constexpr VkDeviceSize kReadbackBytes = 2 * kFrameBytes;
constexpr VkDeviceSize kVertexBytes = static_cast<VkDeviceSize>(kGpuMaxVertices) * sizeof(GpuVertex);
constexpr VkDeviceSize kGlmBytes = static_cast<VkDeviceSize>(4) << 20;           // GS local memory (IN-347)
constexpr VkDeviceSize kPaletteRingBytes = static_cast<VkDeviceSize>(4) << 20;   // 4,096 palettes of 256 RGBA8
constexpr VkDeviceSize kClutBytes = 512 * sizeof(std::uint32_t);                  // GPU CLUT buffer (IN-349)
constexpr VkDeviceSize kUniformBytes = 256;                   // one pixel-state block
constexpr VkDeviceSize kUniformRingBytes = static_cast<VkDeviceSize>(4) << 20;  // 16,384 blocks at 256 B
constexpr std::uint32_t kPushBytes = 32;  // hw_vertex.vert PushConstants (IN-374: + tex_scale)
// IN-371: the pixel shader reads the scale from the same block (native dither cells, 06.5).
constexpr VkShaderStageFlags kGraphicsPushStages = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;

// IN-397: the union of the scaled-pass scissors of some draws (native pixels, clamped to the surface); a draw without
// its own scaled rectangle contributes its 1x one.
struct ScaledBounds {
  std::int32_t x0 = kGpuSurface, y0 = kGpuSurface, x1 = -1, y1 = -1;
  void add(const GpuDraw &draw) {
    const bool own = draw.scaledX1 >= draw.scaledX0 && draw.scaledY1 >= draw.scaledY0;
    x0 = std::min(x0, std::max<std::int32_t>(0, own ? draw.scaledX0 : draw.x0));
    y0 = std::min(y0, std::max<std::int32_t>(0, own ? draw.scaledY0 : draw.y0));
    x1 = std::max(x1, std::min<std::int32_t>(kGpuSurface - 1, own ? draw.scaledX1 : draw.x1));
    y1 = std::max(y1, std::min<std::int32_t>(kGpuSurface - 1, own ? draw.scaledY1 : draw.y1));
  }
  VkRect2D rect() const {
    if (x1 < x0 || y1 < y0) return VkRect2D{{0, 0}, {0, 0}};
    return VkRect2D{{x0, y0}, {static_cast<std::uint32_t>(x1 - x0 + 1), static_cast<std::uint32_t>(y1 - y0 + 1)}};
  }
};

const char *result_text(VkResult result) {
  switch (result) {
  case VK_SUCCESS: return "VK_SUCCESS";
  case VK_ERROR_OUT_OF_HOST_MEMORY: return "VK_ERROR_OUT_OF_HOST_MEMORY";
  case VK_ERROR_OUT_OF_DEVICE_MEMORY: return "VK_ERROR_OUT_OF_DEVICE_MEMORY";
  case VK_ERROR_DEVICE_LOST: return "VK_ERROR_DEVICE_LOST";
  default: return "VkResult error";
  }
}

bool check(VkResult result, const char *what, std::string &error) {
  if (result == VK_SUCCESS) return true;
  error = std::string(what) + " failed: " + result_text(result);
  return false;
}

} // namespace

struct GpuDrawer::Impl {
  VkInstance instance = VK_NULL_HANDLE;
  VkPhysicalDevice physical = VK_NULL_HANDLE;
  VkDevice device = VK_NULL_HANDLE;
  VkQueue queue = VK_NULL_HANDLE;
  coregs::QueueLock *queueLock = nullptr;  // TASK-174: the presentation thread shares `queue` (VulkanHandles::queueLock)
  VkPhysicalDeviceMemoryProperties memory{};

  // Every queue call of the drawer goes through these two.
  VkResult submit_to_queue(const VkSubmitInfo &submit, VkFence signalled) {
    const auto call = [&] { return coregs_hw_vkQueueSubmit(queue, 1, &submit, signalled); };
    return queueLock ? queueLock->run(coregs::QueueLock::Caller::gs_thread, call) : call();
  }
  void device_wait_idle() {
    const auto call = [&] { return coregs_hw_vkDeviceWaitIdle(device); };
    if (queueLock) {
      queueLock->run(coregs::QueueLock::Caller::device_idle, call);
    } else {
      call();
    }
  }

  VkCommandPool pool = VK_NULL_HANDLE;
  VkCommandBuffer cmd = VK_NULL_HANDLE;
  VkFence fence = VK_NULL_HANDLE;

  VkShaderModule vertexModule = VK_NULL_HANDLE;
  VkShaderModule fragmentModule = VK_NULL_HANDLE;
  VkDescriptorSetLayout setLayout = VK_NULL_HANDLE;
  VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
  VkDescriptorPool descriptorPool = VK_NULL_HANDLE;
  // One descriptor set per (colour slot, depth slot) pair (IN-344).
  VkDescriptorSet descriptorSets[kGpuColourSlots * kGpuDepthSlots] = {};
  VkRenderPass renderPass = VK_NULL_HANDLE;
  VkFramebuffer framebuffer = VK_NULL_HANDLE;
  VkPipeline pipelines[3] = {VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE};
  // IN-345: triangle pipeline whose pixel shader orders overlapping fragments with pixel interlock, so a
  // chunk of triangles is one draw without barriers. Null when the device lacks the feature.
  bool interlock = false;
  VkShaderModule interlockModule = VK_NULL_HANDLE;
  VkPipeline interlockTriangles = VK_NULL_HANDLE;

  struct Resource {
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    void *mapped = nullptr;
    VkDeviceSize size = 0;
  };
  struct Image {
    VkImage image = VK_NULL_HANDLE;
    VkImageView view = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
  };

  Resource staging;
  Resource readback;
  Resource vertices;
  Resource uniforms;
  Image frames[kGpuColourSlots];  // colour target slots (IN-344); the last one is the write-back mode's
  Image depths[kGpuDepthSlots];   // depth target slots; the last one is the write-back mode's
  Image texels;
  Image display;  // IN-364: GPU-composed presentation frame (IN-384: grows to the composed frame)

  // ---- M12 upscaling (chapter 06, IN-371/IN-372, Part A IN-384) ----
  // At scale s > 1 every slot has a second image holding the same words at s x s sub-pixels per native pixel. Draws
  // are recorded twice (1x words, then the scaled image); loads into a slot refresh the scaled image from the 1x
  // words (hw_scale_upsample.comp). Only the presentation reads it. IN-384: an image covers only the native
  // rectangle its slot really uses (its FBW * 64 width and the rows loaded or drawn, in pages), not a 1024 x 1024
  // square: at 6x a race frame buffer is 3840 x 2688 (41 MB) instead of 6144 x 6144 (151 MB). It is created on first
  // use and grows on demand, copying the old content.
  struct ScaledExtent {
    std::uint32_t width = 0;   // native pixels (0 x 0: no image)
    std::uint32_t height = 0;
  };
  std::uint32_t scale = 1;
  float scaleOffset = 1.0f;       // hw_vertex.vert scale_offset at scale (06.3: s in native mode)
  std::uint32_t scaledSide = 0;   // kGpuScaledExtent * scale: the largest scaled image (0 at 1x, no scaled images)
  std::uint32_t displayWidth = kSurfaceSize;   // the display image's own extent (grows on demand, IN-384)
  std::uint32_t displayHeight = kSurfaceSize;
  ScaledExtent scaledFrameExtent[kGpuColourSlots];
  ScaledExtent scaledDepthExtent[kGpuDepthSlots];
  ScaledExtent scaledTexelExtent;                      // IN-374 scaled textures
  ScaledExtent scaledShadowExtent[kGpuDisplayShadows];  // IN-377 scaled display shadows
  ScaledExtent shadowNativeExtent[kGpuDisplayShadows];  // their 1x words
  ScaledExtent toneScaledExtent;                        // IN-379 the saved scaled tone source
  std::uint64_t scaledImageBytes = 0;                   // device memory of all of the above (+ display/fxaa)
  bool scaledDescriptorsReady = false;                  // the scaled descriptor sets exist (set by create_scaled_resources)
  Image scaledFrames[kGpuColourSlots];
  Image scaledDepths[kGpuDepthSlots];
  Image scaledShadows[kGpuDisplayShadows];  // IN-377: kGpuScaledExtent * s wide, kGpuShadowRows * s high
  Image nativeShadows[kGpuDisplayShadows];  // IN-377: the same rectangle's 1x words (kGpuScaledExtent x kGpuShadowRows)
  Image toneScaled;  // IN-379: the tone-curve source's scaled image (as a display shadow)
  Image toneNative;  // IN-379: its 1x words
  Resource toneTable;  // IN-379: 3 x 256 table entries
  VkDescriptorSetLayout toneSetLayout = VK_NULL_HANDLE;
  VkPipelineLayout toneLayout = VK_NULL_HANDLE;
  VkShaderModule toneModule = VK_NULL_HANDLE;
  VkPipeline tonePipeline = VK_NULL_HANDLE;
  VkDescriptorSet toneSets[kGpuColourSlots] = {};
  Image scaledTexels;  // IN-374: a texture copied from a slot's scaled image (graphics binding 4 of scaled sets)

  // ---- 13.5 FXAA (IN-376), created on first use ----
  Image fxaaImage;
  std::uint32_t fxaaWidth = 0, fxaaHeight = 0;  // IN-384: the FXAA image's extent
  VkDescriptorSetLayout fxaaSetLayout = VK_NULL_HANDLE;
  VkPipelineLayout fxaaLayout = VK_NULL_HANDLE;
  VkDescriptorPool fxaaPool = VK_NULL_HANDLE;
  VkDescriptorSet fxaaSet = VK_NULL_HANDLE;
  VkShaderModule fxaaModule = VK_NULL_HANDLE;
  VkPipeline fxaaPipeline = VK_NULL_HANDLE;
  bool create_fxaa(std::string &error);
  // IN-381: scaled deinterlace (created on first use).
  bool create_deinterlace(std::string &error);
  Image deinterlaceHistory[kGpuDeinterlaceHistory];
  std::uint32_t lastDeinterlaceCurrent = 0;  // IN-387: the merge image of the last scaled deinterlace (diagnostics)
  Image deinterlaceWoven;
  Image deinterlaceOutput;
  VkDescriptorSetLayout deinterlaceSetLayout = VK_NULL_HANDLE;
  VkPipelineLayout deinterlaceLayout = VK_NULL_HANDLE;
  VkShaderModule deinterlaceModule = VK_NULL_HANDLE;
  // TASK-179: the device was created with shaderStorageImageArrayDynamicIndexing (hw_deinterlace_scaled.comp indexes its
  // history images with a push constant); without it the constant-index build of the kernel runs.
  bool storageImageArrayDynamicIndexing = false;
  VkPipeline deinterlacePipeline = VK_NULL_HANDLE;
  VkDescriptorPool deinterlacePool = VK_NULL_HANDLE;
  VkDescriptorSet deinterlaceSet = VK_NULL_HANDLE;
  VkFramebuffer scaledFramebuffer = VK_NULL_HANDLE;
  VkDescriptorSet scaledDescriptorSets[kGpuColourSlots * kGpuDepthSlots] = {};  // current frame set
  VkDescriptorPool scalePool = VK_NULL_HANDLE;
  VkDescriptorSetLayout upsampleSetLayout = VK_NULL_HANDLE;
  VkPipelineLayout upsampleLayout = VK_NULL_HANDLE;
  VkShaderModule upsampleModule = VK_NULL_HANDLE;
  VkPipeline upsamplePipeline = VK_NULL_HANDLE;
  VkDescriptorSet upsampleSets[kGpuColourSlots + kGpuDepthSlots] = {};
  VkDescriptorSetLayout composeScaledSetLayout = VK_NULL_HANDLE;
  VkPipelineLayout composeScaledLayout = VK_NULL_HANDLE;
  VkShaderModule composeScaledModule = VK_NULL_HANDLE;
  VkPipeline composeScaledPipeline = VK_NULL_HANDLE;
  VkDescriptorSet composeScaledSets[2] = {};  // per frame set (binding 2 is that set's ring)
  bool scaled() const { return scaledSide != 0; }

  // One draw of a scaled pass: the 1x draw's set, uniform, push words and scissor, its absolute first vertex in the
  // vertex ring, and its pipeline.
  struct ScaledDraw {
    std::uint32_t descriptorSet = 0;
    std::uint32_t uniformOffset = 0;
    const std::uint32_t *push = nullptr;
    VkRect2D scissor{};
    std::uint32_t firstVertex = 0;
    std::uint32_t vertexCount = 0;
    VkPipeline pipeline = VK_NULL_HANDLE;
    bool scaledTexture = false;  // IN-374: sample the scaled texel image (a texture copied from a slot at scale)
    bool textured = false;       // IN-389: the chunk samples a texture at all (diagnostics)
  };
  std::vector<ScaledDraw> scaledDraws;  // reused

  // Records `scaledDraws` into the scaled images (IN-371): one render pass over the scaled framebuffer, the same
  // descriptor set index (scaled images, same texel atlas and uniform), positions times s, the 06.3 shift and the
  // scissor times s. Draws into the write-back mode's scratch colour slot are not repeated (nothing presents it).
  // `barrierBetween`: storage-image barriers between draws (the per-primitive path without pixel interlock).
  bool scaledPaused = false;  // IN-382
  bool traceArmed = true;     // diagnostics only (ArmTrace)
  // IN-384 follow-up (owner test 11): COREGS_HW_SCALED_FILTER=linear samples the 1x texel atlas bilinearly in the scaled
  // pass, which smooths the source's 6-pixel steps the way the reference renderer's upscaling filter does.
  bool scaledLinearFilter = false;
  // The reference dithers per OUTPUT pixel by default at scale; the
  // per-native-pixel cell (06.5 native mode) stays available with COREGS_HW_DITHER=native.
  bool ditherFine = true;
  // IN-384 follow-up: bilinear refresh of the colour slots (the reference's default for a 1x upload at scale).
  // Default NEAREST: the 06.7 gate (box downsample against 1x, 25 dB on the synthetic capture) prefers the exact
  // block; COREGS_HW_SCALED_REFRESH=linear turns the smoothing on for the owner's A/B.
  bool scaledRefreshLinear = false;
  // IN-388: native alignment of the scaled pass (the reference's the game half-pixel mode). A primitive edge at a GS integer
  // coordinate lands at the CENTRE of a native pixel's s x s block with the 06.3 shift, so a quad that covers one native
  // pixel at 1x covers half of two blocks at scale and paints a half-native-pixel sliver at every edge (the owner's thin
  // white lines at 6x). Snapping the vertex to the block grid makes each native pixel all-or-nothing, as at 1x.
  bool scaledSnap = false;
  // IN-392: at scale the fragment's Z comes from the rasteriser's position depth (Z * 2^-32 in gl_Position.z), floored
  // to an integer, as the reference does; the float varying it replaces rounds differently at the non-centred sample
  // points of a scaled pixel, so a second pass over the same vertices failed its GEQUAL tie at scattered sub-pixels
  // (the dark notches in white rings and on wheels). COREGS_HW_SCALED_DEPTH=varying
  // restores the varying for an A/B. 1x draws always use the varying (their words are the exact-memory authority).
  bool scaledDepthFromPosition = true;
  // Diagnostics only (COREGS_HW_SCALED_DIAG): 0x08000000 skips a KEEP alpha test, 0x04000000 replaces the texel by a
  // neutral grey, both in the scaled pass only.
  std::uint32_t scaledDiagBits = 0;
  bool scaledScissorShifted = false;  // IN-393: COREGS_HW_SCALED_SCISSOR=shifted
  void record_scaled_draws(bool barrierBetween) {
    if (!scaled() || scaledDraws.empty()) return;
    if (scaledPaused) {
      stats.scaledDrawsPaused += scaledDraws.size();
      scaledDraws.clear();
      return;
    }
    // IN-389: COREGS_HW_SCALED_TRACE=<file> writes one line per scaled pass with the per-colour-slot draw histogram and
    // the number of slot-copy-textured draws, so a layer that never reaches the sampler at scale can be identified.
    if (const char *tracePath = traceArmed ? std::getenv("COREGS_HW_SCALED_TRACE") : nullptr) {
      std::uint32_t perSlot[kGpuColourSlots] = {};
      std::uint32_t texSlot[kGpuColourSlots] = {};
      std::uint32_t copyTextured = 0;
      for (const ScaledDraw &draw : scaledDraws) {
        const std::uint32_t colour = draw.descriptorSet / kGpuDepthSlots;
        if (colour < kGpuColourSlots) ++perSlot[colour];
        if (colour < kGpuColourSlots && draw.textured) ++texSlot[colour];
        if (draw.scaledTexture) ++copyTextured;
      }
      if (std::FILE *file = std::fopen(tracePath, "ab")) {
        std::fprintf(file,
                     "pass draws=%zu copyTex=%u slot0=%u/%u slot1=%u/%u slot2=%u/%u slot3=%u/%u slot4=%u/%u "
                     "slot5=%u/%u\n",
                     scaledDraws.size(), copyTextured, perSlot[0], texSlot[0], perSlot[1], texSlot[1], perSlot[2],
                     texSlot[2], perSlot[3], texSlot[3], perSlot[4], texSlot[4], perSlot[5], texSlot[5]);
        std::fclose(file);
      }
    }
    VkRenderPassBeginInfo pass{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
    pass.renderPass = renderPass;
    pass.framebuffer = scaledFramebuffer;
    pass.renderArea = {{0, 0}, {scaledSide, scaledSide}};
    recorded_work();
    coregs_hwd_vkCmdBeginRenderPass(cmd, &pass, VK_SUBPASS_CONTENTS_INLINE);
    ++stats.renderPasses;
    ++stats.scaledPasses;
    const VkDeviceSize ringStart = 0;
    coregs_hwd_vkCmdBindVertexBuffers(cmd, 0, 1, &vertices.buffer, &ringStart);
    VkPipeline boundPipeline = VK_NULL_HANDLE;
    std::uint32_t lastPush[8] = {};
    bool pushed = false;
    VkRect2D lastScissor{{-1, -1}, {0, 0}};
    std::uint32_t viewportWidth = 0, viewportHeight = 0;
    bool anyDraw = false;
    const std::int32_t s = static_cast<std::int32_t>(scale);
    for (const ScaledDraw &draw : scaledDraws) {
      const std::uint32_t colour = draw.descriptorSet / kGpuDepthSlots;
      const std::uint32_t depth = draw.descriptorSet % kGpuDepthSlots;
      if (colour == kGpuScratchColourSlot) continue;
      // IN-384: the scaled image of this slot pair, not a 1024 * s square. Every collection of scaled draws has
      // grown the images it writes before recording them.
      const std::int32_t imageWidth = static_cast<std::int32_t>(scaledFrameExtent[colour].width) * s;
      const std::int32_t imageHeight = static_cast<std::int32_t>(scaledFrameExtent[colour].height) * s;
      const std::int32_t depthWidth = static_cast<std::int32_t>(scaledDepthExtent[depth].width) * s;
      const std::int32_t depthHeight = static_cast<std::int32_t>(scaledDepthExtent[depth].height) * s;
      if (imageWidth <= 0 || imageHeight <= 0) continue;
      // The GS scissor applies to the native sample points. In the 06.3 native mode native pixel N's sample point
      // (native position N) lands at gpu s*N + s/2, the centre of sub-pixel block [s*N, s*N + s), and sub-pixel j
      // samples native (j + 0.5 - s/2) / s, whose nearest native sample point is floor((j + 0.5) / s): block N owns
      // native pixel N, so the scaled scissor is the native one times s (the reference's rule).
      // IN-393: the IN-372 scissor moved it by h = s/2, half a native pixel right and down;
      // COREGS_HW_SCALED_SCISSOR=shifted restores that for an A/B.
      const std::int32_t h = scaledScissorShifted ? s / 2 : 0;
      const std::int32_t x0 = draw.scissor.offset.x * s + h;
      const std::int32_t y0 = draw.scissor.offset.y * s + h;
      if (x0 >= imageWidth || y0 >= imageHeight) continue;  // outside the scaled image: 1x words only
      std::int32_t x1 =
          std::min<std::int32_t>(imageWidth, x0 + static_cast<std::int32_t>(draw.scissor.extent.width) * s);
      std::int32_t y1 = std::min<std::int32_t>(imageHeight, y0 + static_cast<std::int32_t>(draw.scissor.extent.height) * s);
      // The depth image of this pair covers every draw of its chunks too (Execute grows it before recording them);
      // this clip is the guard that no draw ever samples outside it.
      const std::int32_t clippedX1 = std::min(x1, depthWidth);
      const std::int32_t clippedY1 = std::min(y1, depthHeight);
      if (clippedX1 != x1 || clippedY1 != y1) {
        x1 = clippedX1;
        y1 = clippedY1;
        ++stats.scaledDrawsClipped;
      }
      if (x1 <= x0 || y1 <= y0) continue;
      const VkRect2D scissor{{x0, y0}, {static_cast<std::uint32_t>(x1 - x0), static_cast<std::uint32_t>(y1 - y0)}};
      if (barrierBetween && anyDraw) {
        VkMemoryBarrier between{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        between.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        between.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        coregs_hw_vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0,
                                    1, &between, 0, nullptr, 0, nullptr);
        ++stats.barriers;
      }
      if (draw.pipeline != boundPipeline) {
        coregs_hwd_vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, draw.pipeline);
        boundPipeline = draw.pipeline;
      }
      coregs_hwd_vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0, 1,
                                      &scaledDescriptorSets[draw.descriptorSet], 1, &draw.uniformOffset);
      if (viewportWidth != static_cast<std::uint32_t>(imageWidth) ||
          viewportHeight != static_cast<std::uint32_t>(imageHeight)) {
        VkViewport viewport{0.0f, 0.0f, static_cast<float>(imageWidth), static_cast<float>(imageHeight), 0.0f, 1.0f};
        coregs_hwd_vkCmdSetViewport(cmd, 0, 1, &viewport);
        viewportWidth = static_cast<std::uint32_t>(imageWidth);
        viewportHeight = static_cast<std::uint32_t>(imageHeight);
      }
      std::uint32_t push[8];
      std::memcpy(push, draw.push, sizeof(push));
      const float sizes[2] = {static_cast<float>(imageWidth), static_cast<float>(imageHeight)};
      std::memcpy(push, sizes, sizeof(sizes));
      push[2] = FloatToBits(static_cast<float>(scale));
      push[3] = FloatToBits(scaleOffset);
      push[7] = (draw.scaledTexture ? scale : 1u) | (scaledLinearFilter ? 0x80000000u : 0u) |
                (ditherFine ? 0x40000000u : 0u) |
                (scaledSnap ? 0x20000000u : 0u) |  // IN-374/384/388: tex_scale, forced-linear, dither-fine, snap
                (scaledDepthFromPosition ? 0x10000000u : 0u) |  // IN-392: Z from the position depth
                scaledDiagBits;  // diagnostics: COREGS_HW_SCALED_DIAG=noalpha / notex
      if (!pushed || std::memcmp(lastPush, push, sizeof(push)) != 0) {
        coregs_hwd_vkCmdPushConstants(cmd, pipelineLayout, kGraphicsPushStages, 0, kPushBytes, push);
        std::memcpy(lastPush, push, sizeof(push));
        pushed = true;
      }
      if (std::memcmp(&lastScissor, &scissor, sizeof(VkRect2D)) != 0) {
        coregs_hwd_vkCmdSetScissor(cmd, 0, 1, &scissor);
        lastScissor = scissor;
      }
      recorded_work();
      coregs_hwd_vkCmdDraw(cmd, draw.vertexCount, 1, draw.firstVertex, 0);
      ++stats.scaledDraws;
      anyDraw = true;
    }
    recorded_work();
    coregs_hwd_vkCmdEndRenderPass(cmd);
    scaledDraws.clear();
  }

  // Refreshes the scaled image of a slot over a native rectangle (inclusive) from its 1x words (IN-371). Records
  // the barriers on both sides itself.
  void refresh_scaled(bool depthSlot, std::uint32_t slot, std::int32_t x0, std::int32_t y0, std::int32_t x1,
                      std::int32_t y1) {
    if (!scaled()) return;
    if (scaledPaused) {
      ++stats.scaledRefreshesPaused;
      return;
    }
    // IN-384: the slot's own image, not a 1024 x 1024 square.
    const ScaledExtent &extent = depthSlot ? scaledDepthExtent[slot] : scaledFrameExtent[slot];
    x0 = std::max(x0, 0);
    y0 = std::max(y0, 0);
    x1 = std::min(x1, static_cast<std::int32_t>(extent.width) - 1);
    y1 = std::min(y1, static_cast<std::int32_t>(extent.height) - 1);
    if (x1 < x0 || y1 < y0) return;
    // IN-389: with COREGS_HW_SCALED_TRACE the refresh is logged with the filter it actually pushes, so a 1x -> 6x
    // magnification that ignores COREGS_HW_SCALED_REFRESH can be told from one that never runs.
    if (const char *tracePath = traceArmed ? std::getenv("COREGS_HW_SCALED_TRACE") : nullptr) {
      if (std::FILE *file = std::fopen(tracePath, "ab")) {
        std::fprintf(file, "refresh depth=%u slot=%u linear=%u rect=%d,%d,%d,%d\n", depthSlot ? 1u : 0u, slot,
                     depthSlot ? 0u : (scaledRefreshLinear ? 1u : 0u), x0, y0, x1, y1);
        std::fclose(file);
      }
    }
    mark(kTimeUpload);  // the load before the refresh (12.8 timing: each interval goes to the mark after it)
    full_barrier();
    const std::uint32_t set = depthSlot ? kGpuColourSlots + slot : slot;
    // IN-398: the carry slot (set 1) when the renderer named one for this kind of slot, else the target's own set.
    const bool carry = refreshCarry.active && refreshCarry.depth == depthSlot && refreshCarry.slot != slot;
    const std::uint32_t carrySet = carry ? (depthSlot ? kGpuColourSlots + refreshCarry.slot : refreshCarry.slot) : set;
    const VkDescriptorSet sets[2] = {upsampleSets[set], upsampleSets[carrySet]};
    coregs_hwd_vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, upsamplePipeline);
    coregs_hwd_vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, upsampleLayout, 0, 2, sets, 0, nullptr);
    if (carry) ++stats.scaledRefreshCarries;
    boundComputePipeline = upsamplePipeline;  // bind_compute rebinds its own pipeline and set afterwards
    boundComputeSet = upsampleSets[set];
    const std::uint32_t width = static_cast<std::uint32_t>(x1 - x0 + 1);
    const std::uint32_t height = static_cast<std::uint32_t>(y1 - y0 + 1);
    // IN-384 follow-up: bilinear for the colour slots (the reference's default for a 1x upload into a scaled target),
    // nearest for the depth slots (their words are compared, not shown).
    const std::uint32_t push[12] = {static_cast<std::uint32_t>(x0),
                                    static_cast<std::uint32_t>(y0),
                                    width,
                                    height,
                                    scale,
                                    depthSlot ? 0u : (scaledRefreshLinear ? 1u : 0u),
                                    carry ? 1u : 0u,
                                    refreshCarry.mask,
                                    static_cast<std::uint32_t>(std::max(refreshCarry.x0, 0)),
                                    static_cast<std::uint32_t>(std::max(refreshCarry.y0, 0)),
                                    static_cast<std::uint32_t>(std::max(refreshCarry.x1, 0)),
                                    static_cast<std::uint32_t>(std::max(refreshCarry.y1, 0))};
    coregs_hwd_vkCmdPushConstants(cmd, upsampleLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), push);
    recorded_work();
    coregs_hwd_vkCmdDispatch(cmd, (width * scale + 7) / 8, (height * scale + 7) / 8, 1);
    full_barrier();
    mark(kTimeRefresh);
    ++stats.scaledRefreshes;
    stats.scaledRefreshPixels += static_cast<std::uint64_t>(width) * height;
  }

  bool create_scaled_resources(std::string &error);
  // IN-398: the slot whose images the next refreshes of this kind may keep (GpuDrawer::SetRefreshCarry).
  struct RefreshCarry {
    bool active = false;
    bool depth = false;
    std::uint32_t slot = 0;
    std::uint32_t mask = 0xFFFFFFFFu;
    std::int32_t x0 = 0, y0 = 0, x1 = -1, y1 = -1;  // its valid native rectangle (inclusive)
  } refreshCarry;
  VkDescriptorSet displaySets[2] = {};  // compute set 0 with binding 4 = the display image, per frame set
  VkShaderModule displayModule = VK_NULL_HANDLE;
  VkPipeline displayPipeline = VK_NULL_HANDLE;

  bool ready = false;
  GpuStats stats;

  // Recording state of the rings (IN-342).
  bool recording = false;        // the command buffer is open
  bool inFlight = false;         // a submission may still read the rings
  VkDeviceSize stagingUsed = 0;  // bytes
  VkDeviceSize vertexUsed = 0;   // vertices
  VkDeviceSize uniformUsed = 0;  // bytes
  VkDeviceSize uniformStride = kUniformBytes;

  // GLM (IN-347): device-local mirror of GS memory, address tables, palette ring, compute pipelines. One
  // descriptor set per target slot (binding 4 = that slot's image; colour slots first, then depth slots).
  Resource glm;
  Resource glmTables;
  Resource palettes;
  Resource clut;                 // GPU CLUT buffer: 512 halfwords, one per uint (IN-349)
  VkShaderModule clutModule = VK_NULL_HANDLE;
  VkPipeline clutPipeline = VK_NULL_HANDLE;
  VkDeviceSize paletteUsed = 0;  // bytes of the palette ring used by the open recording
  VkDescriptorSetLayout computeSetLayout = VK_NULL_HANDLE;
  VkPipelineLayout computeLayout = VK_NULL_HANDLE;
  VkDescriptorPool computePool = VK_NULL_HANDLE;
  VkDescriptorSet computeSets[kGpuColourSlots + kGpuDepthSlots] = {};
  VkShaderModule decodeModule = VK_NULL_HANDLE;
  VkShaderModule swizzleModule = VK_NULL_HANDLE;
  VkPipeline decodePipeline = VK_NULL_HANDLE;
  VkPipeline swizzlePipeline = VK_NULL_HANDLE;
  VkShaderModule unswizzleModule = VK_NULL_HANDLE;
  VkPipeline unswizzlePipeline = VK_NULL_HANDLE;
  VkShaderModule mergeModule = VK_NULL_HANDLE;
  VkPipeline mergePipeline = VK_NULL_HANDLE;
  VkShaderModule hostUploadModule = VK_NULL_HANDLE;
  VkPipeline hostUploadPipeline = VK_NULL_HANDLE;
  VkShaderModule toneStripsModule = VK_NULL_HANDLE;  // IN-383
  VkPipeline toneStripsPipeline = VK_NULL_HANDLE;

  // ---- Runs (M9, IN-352) ----
  // Chunks of triangles drawn with the ordered pixel interlock are batched: their texture preparation (GLM
  // decode, staging copy) is recorded at once into the chunk's own region of the texel atlas, their draws are kept
  // here and recorded in one render pass when the run closes. The interlock orders overlapping fragments across
  // the draws of a pass in submission order, so no barrier is needed between the chunks. A run closes before
  // anything that touches a target image outside the pass (slot loads, swizzles, transfers, readbacks, the
  // per-chunk path) and before a submission.
  struct PendingDraw {
    std::uint32_t descriptorSet = 0;
    std::uint32_t uniformOffset = 0;
    std::uint32_t push[8] = {};
    VkDeviceSize vertexOffsetBytes = 0;
    VkRect2D scissor{};
    VkRect2D scaledScissor{};  // IN-397: native pixels, widened by the IN-391 slack
    std::uint32_t firstVertex = 0;
    std::uint32_t vertexCount = 0;
  };
  bool runsEnabled = true;
  bool runOpen = false;
  std::vector<PendingDraw> runDraws;
  std::uint32_t atlasX = 0, atlasY = 0, atlasShelf = 0;  // shelf allocator of the texel atlas

  bool atlas_alloc(std::uint32_t width, std::uint32_t height, std::uint32_t &x, std::uint32_t &y) {
    if (atlasX + width > kGpuTexelAtlasSide) {
      atlasY += atlasShelf;
      atlasX = 0;
      atlasShelf = 0;
    }
    if (atlasY + height > kGpuTexelAtlasSide) return false;
    x = atlasX;
    y = atlasY;
    atlasX += width;
    atlasShelf = std::max(atlasShelf, height);
    return true;
  }

  // Starts a run: earlier work (a previous pass reading the atlas, transfers into target images) completes first.
  void open_run() {
    if (runOpen) return;
    full_barrier_with_fragment();
    runOpen = true;
    atlasX = atlasY = atlasShelf = 0;
    runDecodes.clear();
  }

  // IN-368: true while the last command recorded is a full barrier. Every recorded work command (copy, dispatch,
  // draw, render pass) calls recorded_work() first; a full barrier right after another one is skipped, since both
  // carry the same stages and access masks. A new command buffer starts with false.
  bool barrierIsLast = false;
  void recorded_work() {
    barrierIsLast = false;
    ++workGeneration;
  }

  // IN-369: GLM texture decodes of the open run. A chunk whose decode parameters equal an earlier decode of the run
  // reuses that atlas region when no other GPU work (a GLM write, a CLUT load, a copy, a submission) was recorded
  // since: `workGeneration` changes on every recorded command except the run's own decodes (recorded_decode).
  struct RunDecode {
    std::array<std::uint32_t, 12> key{};  // the decode push constants without the atlas origin
    std::uint32_t x = 0, y = 0;
    std::uint64_t generation = 0;
  };
  bool decodeReuse = true;
  std::uint64_t workGeneration = 0;
  std::vector<RunDecode> runDecodes;
  void recorded_decode() { barrierIsLast = false; }

  // IN-370: deferred GPU CLUT loads. A CLUT load (07.7) reads the GLM when the TEX0 write happens; recording it is
  // put off while the GLM cannot change: every GpuDrawer call that writes the GLM, or reads or replaces the CLUT
  // buffer, first records the pending loads (materialise_clut). Meanwhile an indexed decode whose palette entries
  // were all written by the last pending load (same CPSM and CSA, at least as many entries) reads them straight
  // from the GLM with that load's addressing (decode mode 3) and needs no CLUT buffer. When the loads are
  // recorded, a load whose halfwords a later pending load all rewrites is dropped.
  struct ClutLoadRecord {
    std::uint32_t cbp = 0, cpsm = 0, csm = 0, csa = 0, eightBit = 0, cbw = 0, cou = 0, cov = 0;
    std::array<std::uint64_t, 8> halfwords{};  // the 512 CLUT buffer halfwords this load writes
  };
  bool clutDefer = true;
  std::vector<ClutLoadRecord> pendingClut;
  static constexpr std::size_t kMaxPendingClut = 64;
  static std::array<std::uint64_t, 8> clut_halfwords_written(std::uint32_t cpsm, std::uint32_t csa,
                                                             std::uint32_t eightBit) {
    // The destinations of hw_glm_clut.comp / clut::ClutUnit::Copy.
    std::array<std::uint64_t, 8> mask{};
    auto set = [&mask](std::uint32_t halfword) { mask[halfword >> 6] |= std::uint64_t{1} << (halfword & 63u); };
    const std::uint32_t count = eightBit != 0 ? 256u : 16u;
    for (std::uint32_t e = 0; e < count; ++e) {
      if (cpsm == 0u) {
        const std::uint32_t group = (e >> 4) + (csa & 15u);
        if (group > 15u) continue;
        set(16u * group + (e & 15u));
        set(16u * group + (e & 15u) + 256u);
      } else {
        set((16u * csa + e) & 511u);
      }
    }
    return mask;
  }
  // The last pending load wrote every entry an indexed texture with this CPSM, CSA and index width looks up.
  const ClutLoadRecord *direct_palette_load(std::uint32_t cpsm, std::uint32_t csa, std::uint32_t eightBit) const {
    if (!clutDefer || pendingClut.empty()) return nullptr;
    const ClutLoadRecord &last = pendingClut.back();
    if (last.cpsm != cpsm || last.csa != csa || last.eightBit < eightBit) return nullptr;
    return &last;
  }
  bool materialise_clut(std::string &error) {
    if (pendingClut.empty()) return true;
    if (!begin_recording(error)) return false;
    std::vector<ClutLoadRecord> loads;
    loads.swap(pendingClut);
    // Keep only the loads that still write a halfword no later load rewrites.
    std::vector<bool> keep(loads.size(), false);
    std::array<std::uint64_t, 8> later{};
    for (std::size_t i = loads.size(); i-- > 0;) {
      for (std::size_t w = 0; w < later.size(); ++w) {
        if (loads[i].halfwords[w] & ~later[w]) keep[i] = true;
        later[w] |= loads[i].halfwords[w];
      }
    }
    ++stats.clutMaterialisations;
    full_barrier();  // earlier GLM writes and CLUT buffer readers finish first
    bind_compute(clutPipeline, computeSets[0]);
    std::array<std::uint64_t, 8> sinceBarrier{};
    for (std::size_t i = 0; i < loads.size(); ++i) {
      if (!keep[i]) continue;
      const ClutLoadRecord &load = loads[i];
      bool overlap = false;
      for (std::size_t w = 0; w < sinceBarrier.size(); ++w) overlap = overlap || (load.halfwords[w] & sinceBarrier[w]);
      if (overlap) {  // a later load rewrites halfwords of an earlier one: keep their order
        full_barrier();
        bind_compute(clutPipeline, computeSets[0]);
        sinceBarrier = {};
      }
      for (std::size_t w = 0; w < sinceBarrier.size(); ++w) sinceBarrier[w] |= load.halfwords[w];
      const std::uint32_t push[8] = {load.cbp, load.cpsm, load.csm, load.csa, load.eightBit, load.cbw, load.cou, load.cov};
      coregs_hwd_vkCmdPushConstants(cmd, computeLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), push);
      recorded_work();
      coregs_hwd_vkCmdDispatch(cmd, load.eightBit != 0 ? 4u : 1u, 1, 1);
      ++stats.clutDispatches;
    }
    mark(kTimeClut);
    full_barrier();
    return true;
  }
  const RunDecode *find_run_decode(const std::array<std::uint32_t, 12> &key) const {
    for (auto it = runDecodes.rbegin(); it != runDecodes.rend(); ++it) {
      if (it->generation != workGeneration) break;  // older entries are older still
      if (it->key == key) return &*it;
    }
    return nullptr;
  }

  void full_barrier_with_fragment() {
    if (barrierIsLast) {  // IN-368: a second barrier with no work in between adds no dependency
      ++stats.barriersElided;
      return;
    }
    barrierIsLast = true;
    VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT |
                            VK_ACCESS_TRANSFER_WRITE_BIT;
    const VkPipelineStageFlags stages = VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT |
                                        VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
    coregs_hw_vkCmdPipelineBarrier(cmd, stages, stages, 0, 1, &barrier, 0, nullptr, 0, nullptr);
    ++stats.barriers;
  }

  bool flush_run(std::string &error);

  // IN-362: compute bindings persist in a command buffer; rebinding the same pipeline or set is skipped.
  VkPipeline boundComputePipeline = VK_NULL_HANDLE;
  VkDescriptorSet boundComputeSet = VK_NULL_HANDLE;
  void bind_compute(VkPipeline pipeline, VkDescriptorSet set) {
    if (pipeline != boundComputePipeline) {
      coregs_hwd_vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
      boundComputePipeline = pipeline;
    }
    if (set != boundComputeSet) {
      coregs_hwd_vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, computeLayout, 0, 1, &set, 0, nullptr);
      boundComputeSet = set;
    }
  }

  // ---- Double-buffered submission (IN-354) ----
  // Two sets of the per-submission resources (command buffer, fence, staging, vertex, uniform and palette rings
  // and the descriptor sets that point at the rings). A submission switches to the other set, so the CPU records
  // the next frame while the GPU executes the previous one; a recording waits only for the submission that last
  // used the same set. The fields cmd, fence, inFlight, staging, vertices, uniforms, palettes, descriptorSets and
  // computeSets always describe the current set.
  struct FrameResources {
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    VkFence fence = VK_NULL_HANDLE;
    bool inFlight = false;
    std::uint64_t serial = 0;
    Resource staging;
    Resource vertices;
    Resource uniforms;
    Resource palettes;
    VkDescriptorSet descriptorSets[kGpuColourSlots * kGpuDepthSlots] = {};
    VkDescriptorSet scaledDescriptorSets[kGpuColourSlots * kGpuDepthSlots] = {};  // IN-371
    VkDescriptorSet computeSets[kGpuColourSlots + kGpuDepthSlots] = {};
  };
  FrameResources frameSets[2];
  std::uint32_t currentFrame = 0;
  bool secondFrameReady = false;  // frame set 1 exists
  bool doubleBuffered = false;    // submissions alternate between the two sets

  void save_frame(std::uint32_t index) {
    FrameResources &f = frameSets[index];
    f.cmd = cmd;
    f.fence = fence;
    f.inFlight = inFlight;
    f.serial = serial;
    f.staging = staging;
    f.vertices = vertices;
    f.uniforms = uniforms;
    f.palettes = palettes;
    std::copy(std::begin(descriptorSets), std::end(descriptorSets), std::begin(f.descriptorSets));
    std::copy(std::begin(scaledDescriptorSets), std::end(scaledDescriptorSets), std::begin(f.scaledDescriptorSets));
    std::copy(std::begin(computeSets), std::end(computeSets), std::begin(f.computeSets));
  }
  void load_frame(std::uint32_t index) {
    const FrameResources &f = frameSets[index];
    cmd = f.cmd;
    fence = f.fence;
    inFlight = f.inFlight;
    serial = f.serial;
    staging = f.staging;
    vertices = f.vertices;
    uniforms = f.uniforms;
    palettes = f.palettes;
    std::copy(std::begin(f.descriptorSets), std::end(f.descriptorSets), std::begin(descriptorSets));
    std::copy(std::begin(f.scaledDescriptorSets), std::end(f.scaledDescriptorSets), std::begin(scaledDescriptorSets));
    std::copy(std::begin(f.computeSets), std::end(f.computeSets), std::begin(computeSets));
  }
  bool wait_all(std::string &error);
  bool create_second_frame(std::string &error);

  // IN-372: the scaled compose set of frame set `index` (its ring is `ring`).
  bool make_compose_scaled_set(std::uint32_t index, const Resource &ring, std::string &error) {
    if (!composeScaledSets[index]) {
      VkDescriptorSetAllocateInfo allocate{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
      allocate.descriptorPool = scalePool;
      allocate.descriptorSetCount = 1;
      allocate.pSetLayouts = &composeScaledSetLayout;
      if (!check(coregs_hwd_vkAllocateDescriptorSets(device, &allocate, &composeScaledSets[index]),
                 "vkAllocateDescriptorSets", error)) {
        return false;
      }
    }
    VkDescriptorBufferInfo buffers[3] = {{glm.buffer, 0, kGlmBytes}, {glmTables.buffer, 0, glmTables.size},
                                         {ring.buffer, 0, kPaletteRingBytes}};
    VkDescriptorImageInfo out{VK_NULL_HANDLE, display.view, VK_IMAGE_LAYOUT_GENERAL};
    // Sources 0..kGpuColourSlots-1 are the slots' scaled images, then the IN-377 display shadows.
    VkDescriptorImageInfo frameViews[kGpuColourSlots + kGpuDisplayShadows];
    for (std::uint32_t i = 0; i < kGpuColourSlots; ++i) frameViews[i] = {VK_NULL_HANDLE, scaledFrames[i].view,
                                                                         VK_IMAGE_LAYOUT_GENERAL};
    for (std::uint32_t i = 0; i < kGpuDisplayShadows; ++i) {
      frameViews[kGpuColourSlots + i] = {VK_NULL_HANDLE, scaledShadows[i].view, VK_IMAGE_LAYOUT_GENERAL};
    }
    VkDescriptorImageInfo nativeViews[kGpuDisplayShadows];
    for (std::uint32_t i = 0; i < kGpuDisplayShadows; ++i) {
      nativeViews[i] = {VK_NULL_HANDLE, nativeShadows[i].view, VK_IMAGE_LAYOUT_GENERAL};
    }
    VkWriteDescriptorSet writes[6]{};
    for (std::uint32_t i = 0; i < 6; ++i) {
      writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
      writes[i].dstSet = composeScaledSets[index];
      writes[i].descriptorCount = 1;
    }
    for (std::uint32_t i = 0; i < 3; ++i) {
      writes[i].dstBinding = i;
      writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
      writes[i].pBufferInfo = &buffers[i];
    }
    writes[3].dstBinding = 4;
    writes[3].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    writes[3].pImageInfo = &out;
    writes[4].dstBinding = 6;
    writes[4].descriptorCount = kGpuColourSlots + kGpuDisplayShadows;
    writes[4].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    writes[4].pImageInfo = frameViews;
    writes[5].dstBinding = 7;
    writes[5].descriptorCount = kGpuDisplayShadows;
    writes[5].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    writes[5].pImageInfo = nativeViews;
    coregs_hwd_vkUpdateDescriptorSets(device, 6, writes, 0, nullptr);
    return true;
  }

  // IN-364: the display set of frame set `index`: a copy of that set's compute set 0 with binding 4 = display.
  bool make_display_set(std::uint32_t index, VkDescriptorSet source, std::string &error) {
    VkDescriptorSetAllocateInfo allocate{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    allocate.descriptorPool = computePool;
    allocate.descriptorSetCount = 1;
    allocate.pSetLayouts = &computeSetLayout;
    if (!check(coregs_hwd_vkAllocateDescriptorSets(device, &allocate, &displaySets[index]), "vkAllocateDescriptorSets",
               error)) {
      return false;
    }
    VkCopyDescriptorSet copies[5]{};
    const std::uint32_t copied[5] = {0, 1, 2, 3, 5};
    for (std::uint32_t c = 0; c < 5; ++c) {
      copies[c].sType = VK_STRUCTURE_TYPE_COPY_DESCRIPTOR_SET;
      copies[c].srcSet = source;
      copies[c].srcBinding = copied[c];
      copies[c].dstSet = displaySets[index];
      copies[c].dstBinding = copied[c];
      copies[c].descriptorCount = 1;
    }
    VkDescriptorImageInfo image{VK_NULL_HANDLE, display.view, VK_IMAGE_LAYOUT_GENERAL};
    VkWriteDescriptorSet write{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    write.dstSet = displaySets[index];
    write.dstBinding = 4;
    write.descriptorCount = 1;
    write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    write.pImageInfo = &image;
    coregs_hwd_vkUpdateDescriptorSets(device, 1, &write, 5, copies);
    return true;
  }

  // IN-358: the recording is submitted every kChunksPerSubmission chunks (with double buffering), so the GPU
  // draws the frame while the CPU records the rest of it. Without this nothing reached the GPU before the VBlank
  // or a readback, and a readback in mid-frame (the game's lens flare) waited for the GPU to draw the whole scene so far.
  static constexpr std::uint32_t kChunksPerSubmission = 64;
  std::uint32_t chunksSinceSubmission = 0;
  std::uint32_t chunksPerSubmission = kChunksPerSubmission;  // 0 = only at VBlank, ring full and readbacks
  bool note_chunk(std::string &error) {
    ++chunksSinceSubmission;
    if (doubleBuffered && chunksPerSubmission != 0 && chunksSinceSubmission >= chunksPerSubmission) {
      ++stats.earlySubmissions;
      return submit(false, error);
    }
    return true;
  }

  // ---- GPU timing (COREGS_HW_GPU_TIMING) ----
  enum TimeCategory : std::uint8_t { kTimePass, kTimeDecode, kTimeClut, kTimeSwizzle, kTimeUpload, kTimeDownload,
                                     kTimeScaled, kTimeRefresh, kTimeStart };
  static constexpr std::uint32_t kQueries = 32768;
  struct QueryState {
    VkQueryPool pool = VK_NULL_HANDLE;
    std::uint32_t count = 0;
    std::vector<std::uint8_t> categories;
  };
  bool gpuTiming = false;
  double timestampPeriodNs = 1.0;
  QueryState queries[2];
  std::uint32_t recordingSet = 0;  // the frame set the open recording belongs to

  void mark(TimeCategory category) {
    if (!gpuTiming || !recording) return;
    QueryState &q = queries[recordingSet];
    if (!q.pool || q.count >= kQueries) return;
    coregs_hwd_vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, q.pool, q.count);
    q.categories[q.count] = category;
    ++q.count;
  }
  // The submission of `set` has completed: its timestamps are charged to their categories.
  void collect_timing(std::uint32_t set) {
    QueryState &q = queries[set];
    if (!q.pool || q.count < 2) {
      q.count = 0;
      return;
    }
    std::vector<std::uint64_t> stamps(q.count);
    if (coregs_hwd_vkGetQueryPoolResults(device, q.pool, 0, q.count, stamps.size() * sizeof(std::uint64_t), stamps.data(),
                                      sizeof(std::uint64_t), VK_QUERY_RESULT_64_BIT) == VK_SUCCESS) {
      for (std::uint32_t i = 1; i < q.count; ++i) {
        if (stamps[i] < stamps[i - 1] || q.categories[i] >= kTimeStart) continue;
        stats.gpuNs[q.categories[i]] += static_cast<std::uint64_t>((stamps[i] - stamps[i - 1]) * timestampPeriodNs);
      }
      if (stamps[q.count - 1] > stamps[0]) {
        stats.gpuBusyNs += static_cast<std::uint64_t>((stamps[q.count - 1] - stamps[0]) * timestampPeriodNs);
      }
    }
    q.count = 0;
  }

  // Submission serials (IN-355): serial of the current set's last submission (mirror; FrameResources::serial
  // for the other set) and the highest serial known complete.
  std::uint64_t serial = 0;
  std::uint64_t completedSerial = 0;
  // Asynchronous download buffers (IN-355): two host buffers, with the copies recorded into each.
  Resource downloads[2];
  std::vector<std::pair<std::uint32_t, std::uint32_t>> downloadRanges[2];
  std::uint32_t nextDownload = 0;

  VkPipeline make_compute(VkShaderModule module, std::string &error) {
    VkComputePipelineCreateInfo info{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    info.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    info.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    info.stage.module = module;
    info.stage.pName = "main";
    info.layout = computeLayout;
    VkPipeline pipeline = VK_NULL_HANDLE;
    check(coregs_hwd_vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &info, nullptr, &pipeline),
          "vkCreateComputePipelines", error);
    return pipeline;
  }

  // Every earlier GPU write (copies, compute and fragment shaders) is visible to every later read or write.
  void full_barrier() {
    if (barrierIsLast) {  // IN-368: a second barrier with no work in between adds no dependency
      ++stats.barriersElided;
      return;
    }
    barrierIsLast = true;
    VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT |
                            VK_ACCESS_TRANSFER_WRITE_BIT;
    const VkPipelineStageFlags stages = VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT |
                                        VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
    coregs_hw_vkCmdPipelineBarrier(cmd, stages, stages, 0, 1, &barrier, 0, nullptr, 0, nullptr);
    ++stats.barriers;
  }

  bool submit(bool wait, std::string &error);
  bool wait_idle(std::string &error);
  bool begin_recording(std::string &error);

  // Chooses a memory type with all `wanted` flags; `preferred` flags are tried first.
  bool memory_type(std::uint32_t typeBits, VkMemoryPropertyFlags wanted, VkMemoryPropertyFlags preferred,
                   std::uint32_t &chosen) const {
    for (int pass = 0; pass < 2; ++pass) {
      const VkMemoryPropertyFlags required = pass == 0 ? (wanted | preferred) : wanted;
      for (std::uint32_t i = 0; i < memory.memoryTypeCount; ++i) {
        if (!(typeBits & (1u << i))) continue;
        if ((memory.memoryTypes[i].propertyFlags & required) == required) {
          chosen = i;
          return true;
        }
      }
      if (preferred == 0) break;
    }
    return false;
  }

  bool make_buffer(Resource &resource, VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags wanted,
                   VkMemoryPropertyFlags preferred, bool map, std::string &error) {
    VkBufferCreateInfo info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    info.size = size;
    info.usage = usage;
    info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if (!check(coregs_hwd_vkCreateBuffer(device, &info, nullptr, &resource.buffer), "vkCreateBuffer", error)) return false;
    VkMemoryRequirements requirements{};
    coregs_hwd_vkGetBufferMemoryRequirements(device, resource.buffer, &requirements);
    std::uint32_t type = 0;
    if (!memory_type(requirements.memoryTypeBits, wanted, preferred, type)) {
      error = "no host-visible memory type for a HWR buffer";
      return false;
    }
    VkMemoryAllocateInfo allocate{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    allocate.allocationSize = requirements.size;
    allocate.memoryTypeIndex = type;
    if (!check(coregs_hwd_vkAllocateMemory(device, &allocate, nullptr, &resource.memory), "vkAllocateMemory", error)) return false;
    if (!check(coregs_hwd_vkBindBufferMemory(device, resource.buffer, resource.memory, 0), "vkBindBufferMemory", error)) return false;
    resource.size = size;
    if (map) {
      if (!check(coregs_hwd_vkMapMemory(device, resource.memory, 0, size, 0, &resource.mapped), "vkMapMemory", error)) return false;
    }
    return true;
  }

  bool make_image(Image &image, std::uint32_t width, std::uint32_t height, std::string &error) {
    VkImageCreateInfo info{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    info.imageType = VK_IMAGE_TYPE_2D;
    info.format = VK_FORMAT_R32_UINT;
    info.extent = {width, height, 1};
    info.mipLevels = 1;
    info.arrayLayers = 1;
    info.samples = VK_SAMPLE_COUNT_1_BIT;
    info.tiling = VK_IMAGE_TILING_OPTIMAL;
    info.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    if (!check(coregs_hwd_vkCreateImage(device, &info, nullptr, &image.image), "vkCreateImage", error)) return false;
    VkMemoryRequirements requirements{};
    coregs_hwd_vkGetImageMemoryRequirements(device, image.image, &requirements);
    std::uint32_t type = 0;
    if (!memory_type(requirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 0, type)) {
      error = "no device-local memory type for a HWR image";
      return false;
    }
    VkMemoryAllocateInfo allocate{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    allocate.allocationSize = requirements.size;
    allocate.memoryTypeIndex = type;
    if (!check(coregs_hwd_vkAllocateMemory(device, &allocate, nullptr, &image.memory), "vkAllocateMemory", error)) return false;
    if (!check(coregs_hwd_vkBindImageMemory(device, image.image, image.memory, 0), "vkBindImageMemory", error)) return false;
    VkImageViewCreateInfo view{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    view.image = image.image;
    view.viewType = VK_IMAGE_VIEW_TYPE_2D;
    view.format = VK_FORMAT_R32_UINT;
    view.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    return check(coregs_hwd_vkCreateImageView(device, &view, nullptr, &image.view), "vkCreateImageView", error);
  }

  // IN-384: destroys one image, view and memory (used when a scaled image grows).
  void destroy_image(Image &image) {
    if (image.view) coregs_hwd_vkDestroyImageView(device, image.view, nullptr);
    if (image.image) coregs_hwd_vkDestroyImage(device, image.image, nullptr);
    if (image.memory) coregs_hwd_vkFreeMemory(device, image.memory, nullptr);
    image = Image{};
  }

  static std::uint32_t round_up(std::uint32_t value, std::uint32_t multiple) {
    return (value + multiple - 1u) / multiple * multiple;
  }

  // Device memory of every image that only exists for the scaled presentation (reported, IN-384).
  std::uint64_t scaled_bytes() const {
    std::uint64_t bytes = 0;
    const auto add = [&bytes](const Image &image, std::uint32_t width, std::uint32_t height) {
      if (image.image) bytes += static_cast<std::uint64_t>(width) * height * 4u;
    };
    for (std::uint32_t i = 0; i < kGpuColourSlots; ++i) {
      add(scaledFrames[i], scaledFrameExtent[i].width * scale, scaledFrameExtent[i].height * scale);
    }
    for (std::uint32_t i = 0; i < kGpuDepthSlots; ++i) {
      add(scaledDepths[i], scaledDepthExtent[i].width * scale, scaledDepthExtent[i].height * scale);
    }
    add(scaledTexels, scaledTexelExtent.width * scale, scaledTexelExtent.height * scale);
    for (std::uint32_t i = 0; i < kGpuDisplayShadows; ++i) {
      add(scaledShadows[i], scaledShadowExtent[i].width * scale, scaledShadowExtent[i].height * scale);
      add(nativeShadows[i], shadowNativeExtent[i].width, shadowNativeExtent[i].height);
    }
    add(toneScaled, toneScaledExtent.width * scale, toneScaledExtent.height * scale);
    add(toneNative, kGpuScaledExtent, kGpuShadowRows);
    return bytes;
  }

  // Keeps the reported number (stats.scaledImageBytes) in step with the images that exist.
  void note_scaled_bytes() {
    scaledImageBytes = scaled_bytes();
    stats.scaledImageBytes = scaledImageBytes;
  }

  // Rewrites every descriptor set that names a scaled image. Growth is rare (a slot's first use or a bigger target),
  // and ensure_scaled_image has waited for the GPU, so no submission is using these sets. Every frame set has its own
  // copies of the per-pair sets, so all of them are rewritten (a stale copy would name a destroyed view).
  void update_scaled_descriptors() {
    if (!scaledDescriptorsReady) return;  // create_scaled_resources writes them from the current views
    const auto write_pairs = [this](VkDescriptorSet *sets) {
      for (std::uint32_t colour = 0; colour < kGpuColourSlots; ++colour) {
        for (std::uint32_t depth = 0; depth < kGpuDepthSlots; ++depth) {
          VkDescriptorSet set = sets[colour * kGpuDepthSlots + depth];
          if (!set) continue;
          VkDescriptorImageInfo images[3] = {{VK_NULL_HANDLE, scaledFrames[colour].view, VK_IMAGE_LAYOUT_GENERAL},
                                             {VK_NULL_HANDLE, scaledDepths[depth].view, VK_IMAGE_LAYOUT_GENERAL},
                                             {VK_NULL_HANDLE, scaledTexels.view, VK_IMAGE_LAYOUT_GENERAL}};
          const std::uint32_t bindings[3] = {1, 2, 4};
          VkWriteDescriptorSet writes[3]{};
          for (std::uint32_t i = 0; i < 3; ++i) {
            writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[i].dstSet = set;
            writes[i].dstBinding = bindings[i];
            writes[i].descriptorCount = 1;
            writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            writes[i].pImageInfo = &images[i];
          }
          coregs_hwd_vkUpdateDescriptorSets(device, 3, writes, 0, nullptr);
        }
      }
    };
    write_pairs(scaledDescriptorSets);
    write_pairs(frameSets[0].scaledDescriptorSets);
    write_pairs(frameSets[1].scaledDescriptorSets);
    for (std::uint32_t set = 0; set < kGpuColourSlots + kGpuDepthSlots; ++set) {
      const bool depthSet = set >= kGpuColourSlots;
      const std::uint32_t slot = depthSet ? set - kGpuColourSlots : set;
      VkDescriptorImageInfo image{VK_NULL_HANDLE, depthSet ? scaledDepths[slot].view : scaledFrames[slot].view,
                                  VK_IMAGE_LAYOUT_GENERAL};
      VkWriteDescriptorSet write{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
      write.dstSet = upsampleSets[set];
      write.dstBinding = 1;
      write.descriptorCount = 1;
      write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
      write.pImageInfo = &image;
      coregs_hwd_vkUpdateDescriptorSets(device, 1, &write, 0, nullptr);
    }
    for (std::uint32_t slot = 0; slot < kGpuColourSlots; ++slot) {
      if (!toneSets[slot]) continue;
      VkDescriptorImageInfo images[2] = {{VK_NULL_HANDLE, toneScaled.view, VK_IMAGE_LAYOUT_GENERAL},
                                         {VK_NULL_HANDLE, scaledFrames[slot].view, VK_IMAGE_LAYOUT_GENERAL}};
      VkWriteDescriptorSet writes[2]{};
      for (std::uint32_t i = 0; i < 2; ++i) {
        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet = toneSets[slot];
        writes[i].dstBinding = i == 0 ? 1u : 3u;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        writes[i].pImageInfo = &images[i];
      }
      coregs_hwd_vkUpdateDescriptorSets(device, 2, writes, 0, nullptr);
    }
    for (std::uint32_t frame = 0; frame < 2; ++frame) {
      if (!composeScaledSets[frame]) continue;
      // Each frame's compose set names that frame's palette ring (IN-372); never another frame's.
      std::string ignored;
      make_compose_scaled_set(frame, secondFrameReady ? frameSets[frame].palettes : palettes, ignored);
    }
    note_scaled_bytes();
  }

  // Creates or grows an image so it covers width x height native pixels (width * pixelScale x height * pixelScale
  // pixels), copying the old content. The width is rounded up to a page (64) and the height doubled per growth, so a
  // target that grows row by row is rebuilt a few times at most. Nothing may reference the old image while it is
  // replaced: the recording is submitted and waited for before and after the copy (that is also what makes
  // destroying the old image and rewriting the descriptor sets safe). `pixelScale` is the internal scale for the
  // presentation images and 1 for the few that are read at 1x (the IN-377 shadow words).
  bool ensure_scaled_image(Image &image, ScaledExtent &extent, std::uint32_t width, std::uint32_t height,
                           std::string &error, std::uint32_t pixelScale = 0) {
    if (pixelScale == 0) pixelScale = scale;
    width = std::max<std::uint32_t>(64u, round_up(width, 64u));
    height = std::max<std::uint32_t>(32u, round_up(height, 32u));
    if (extent.width >= width && extent.height >= height) return true;
    const std::uint32_t newWidth = std::max(width, extent.width);
    const std::uint32_t newHeight = std::max(height, extent.height * 2u);
    if (image.image && !submit(true, error)) return false;
    Image fresh;
    if (!make_image(fresh, newWidth * pixelScale, newHeight * pixelScale, error)) return false;
    if (!begin_recording(error)) return false;
    VkImageMemoryBarrier toGeneral{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    toGeneral.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    toGeneral.newLayout = VK_IMAGE_LAYOUT_GENERAL;
    toGeneral.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    toGeneral.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    toGeneral.image = fresh.image;
    toGeneral.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    toGeneral.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    coregs_hw_vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0,
                                nullptr, 0, nullptr, 1, &toGeneral);
    if (image.image) {
      VkImageCopy region{};
      region.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
      region.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
      region.extent = {extent.width * pixelScale, extent.height * pixelScale, 1};
      recorded_work();
      coregs_hwd_vkCmdCopyImage(cmd, image.image, VK_IMAGE_LAYOUT_GENERAL, fresh.image, VK_IMAGE_LAYOUT_GENERAL, 1,
                             &region);
      full_barrier();
    }
    if (!submit(true, error)) return false;
    destroy_image(image);  // the GPU finished with it, and the next line updates every set that named it
    image = fresh;
    extent.width = newWidth;
    extent.height = newHeight;
    update_scaled_descriptors();
    ++stats.scaledImageGrows;
    static const bool trace = std::getenv("COREGS_HW_SCALE_TRACE") != nullptr;  // diagnostic only
    if (trace) {
      std::fprintf(stderr, "[scale] grow to %ux%u native (%ux%u px)\n", newWidth, newHeight, newWidth * pixelScale,
                   newHeight * pixelScale);
    }
    return true;
  }

  // IN-384: the display image (what the presentation reads) grows to the composed frame, not to a 1024 * s square.
  // Its old content is never needed (a new size implies a new compose), so no copy.
  bool ensure_display_image(std::uint32_t width, std::uint32_t height, std::string &error) {
    if (width <= displayWidth && height <= displayHeight) return true;
    const std::uint32_t newWidth = std::max(width, displayWidth);
    const std::uint32_t newHeight = std::max(height, displayHeight);
    if (display.image && !submit(true, error)) return false;
    Image fresh;
    if (!make_image(fresh, newWidth, newHeight, error)) return false;
    if (!begin_recording(error)) return false;
    VkImageMemoryBarrier toGeneral{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    toGeneral.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    toGeneral.newLayout = VK_IMAGE_LAYOUT_GENERAL;
    toGeneral.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    toGeneral.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    toGeneral.image = fresh.image;
    toGeneral.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    toGeneral.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT;
    coregs_hw_vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0,
                                nullptr, 0, nullptr, 1, &toGeneral);
    recorded_work();
    if (!submit(true, error)) return false;
    destroy_image(display);
    display = fresh;
    displayWidth = newWidth;
    displayHeight = newHeight;
    // The compose kernels and the FXAA pass name the display image.
    for (std::uint32_t frame = 0; frame < 2; ++frame) {
      if (!displaySets[frame]) continue;
      VkDescriptorImageInfo image{VK_NULL_HANDLE, display.view, VK_IMAGE_LAYOUT_GENERAL};
      VkWriteDescriptorSet write{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
      write.dstSet = displaySets[frame];
      write.dstBinding = 4;
      write.descriptorCount = 1;
      write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
      write.pImageInfo = &image;
      coregs_hwd_vkUpdateDescriptorSets(device, 1, &write, 0, nullptr);
    }
    for (std::uint32_t frame = 0; frame < 2; ++frame) {
      if (!composeScaledSets[frame]) continue;
      std::string ignored;
      make_compose_scaled_set(frame, secondFrameReady ? frameSets[frame].palettes : palettes, ignored);
    }
    if (fxaaSet) {
      VkDescriptorImageInfo image{VK_NULL_HANDLE, display.view, VK_IMAGE_LAYOUT_GENERAL};
      VkWriteDescriptorSet write{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
      write.dstSet = fxaaSet;
      write.dstBinding = 0;
      write.descriptorCount = 1;
      write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
      write.pImageInfo = &image;
      coregs_hwd_vkUpdateDescriptorSets(device, 1, &write, 0, nullptr);
    }
    if (deinterlaceSet) {
      // IN-381: binding 6 of the scaled deinterlace kernel is the display image it reads the composed frame from.
      VkDescriptorImageInfo image{VK_NULL_HANDLE, display.view, VK_IMAGE_LAYOUT_GENERAL};
      VkWriteDescriptorSet write{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
      write.dstSet = deinterlaceSet;
      write.dstBinding = 6;
      write.descriptorCount = 1;
      write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
      write.pImageInfo = &image;
      coregs_hwd_vkUpdateDescriptorSets(device, 1, &write, 0, nullptr);
    }
    note_scaled_bytes();
    return true;
  }

  VkShaderModule make_module(const std::uint32_t *words, std::size_t count, std::string &error) {
    VkShaderModuleCreateInfo info{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    info.codeSize = count * sizeof(std::uint32_t);
    info.pCode = words;
    VkShaderModule module = VK_NULL_HANDLE;
    if (!check(coregs_hwd_vkCreateShaderModule(device, &info, nullptr, &module), "vkCreateShaderModule", error)) {
      return VK_NULL_HANDLE;
    }
    return module;
  }

  VkPipeline make_pipeline(VkPrimitiveTopology topology, std::string &error, VkShaderModule fragment = VK_NULL_HANDLE) {
    VkPipelineShaderStageCreateInfo stages[2]{};
    stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
    stages[0].module = vertexModule;
    stages[0].pName = "main";
    stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    stages[1].module = fragment ? fragment : fragmentModule;
    stages[1].pName = "main";

    VkVertexInputBindingDescription binding{};
    binding.binding = 0;
    binding.stride = sizeof(GpuVertex);
    binding.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
    VkVertexInputAttributeDescription attributes[7]{};
    attributes[0] = {0, 0, VK_FORMAT_R32G32_SFLOAT, 0};      // S, T
    attributes[1] = {1, 0, VK_FORMAT_R8G8B8A8_UINT, 8};      // R, G, B, A
    attributes[2] = {2, 0, VK_FORMAT_R32_SFLOAT, 12};        // Q
    attributes[3] = {3, 0, VK_FORMAT_R16G16_UINT, 16};       // X, Y
    attributes[4] = {4, 0, VK_FORMAT_R32_UINT, 20};          // Z
    attributes[5] = {5, 0, VK_FORMAT_R16G16_UINT, 24};       // U, V
    attributes[6] = {6, 0, VK_FORMAT_R8_UNORM, 28};          // F
    VkPipelineVertexInputStateCreateInfo vertexInput{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
    vertexInput.vertexBindingDescriptionCount = 1;
    vertexInput.pVertexBindingDescriptions = &binding;
    vertexInput.vertexAttributeDescriptionCount = 7;
    vertexInput.pVertexAttributeDescriptions = attributes;

    VkPipelineInputAssemblyStateCreateInfo assembly{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
    assembly.topology = topology;

    VkPipelineViewportStateCreateInfo viewport{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
    viewport.viewportCount = 1;
    viewport.scissorCount = 1;

    VkPipelineRasterizationStateCreateInfo raster{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
    raster.polygonMode = VK_POLYGON_MODE_FILL;
    raster.cullMode = VK_CULL_MODE_NONE;
    raster.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    raster.lineWidth = 1.0f;

    VkPipelineMultisampleStateCreateInfo multisample{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
    multisample.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    VkPipelineColorBlendStateCreateInfo blend{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
    blend.attachmentCount = 0;  // no colour attachment: the fragment stage writes storage images

    VkDynamicState dynamicStates[2] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo dynamic{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
    dynamic.dynamicStateCount = 2;
    dynamic.pDynamicStates = dynamicStates;

    VkGraphicsPipelineCreateInfo info{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
    info.stageCount = 2;
    info.pStages = stages;
    info.pVertexInputState = &vertexInput;
    info.pInputAssemblyState = &assembly;
    info.pViewportState = &viewport;
    info.pRasterizationState = &raster;
    info.pMultisampleState = &multisample;
    info.pColorBlendState = &blend;
    info.pDynamicState = &dynamic;
    info.layout = pipelineLayout;
    info.renderPass = renderPass;
    info.subpass = 0;
    VkPipeline pipeline = VK_NULL_HANDLE;
    check(coregs_hwd_vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &info, nullptr, &pipeline),
          "vkCreateGraphicsPipelines", error);
    return pipeline;
  }

  // Records a whole-image layout change UNDEFINED -> GENERAL for the three images, once at start.
  bool make_images_general(std::string &error) {
    VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    if (!check(coregs_hw_vkBeginCommandBuffer(cmd, &begin), "vkBeginCommandBuffer", error)) return false;
    constexpr int kMaxImages = static_cast<int>(2 * (kGpuColourSlots + kGpuDepthSlots) + 3 + 2 * kGpuDisplayShadows + 2);
    VkImageMemoryBarrier barriers[kMaxImages]{};
    VkImage images[kMaxImages] = {};
    int kImages = 0;
    for (std::uint32_t i = 0; i < kGpuColourSlots; ++i) images[kImages++] = frames[i].image;
    for (std::uint32_t i = 0; i < kGpuDepthSlots; ++i) images[kImages++] = depths[i].image;
    images[kImages++] = texels.image;
    images[kImages++] = display.image;
    if (scaled()) {  // IN-371
      for (std::uint32_t i = 0; i < kGpuColourSlots; ++i) images[kImages++] = scaledFrames[i].image;
      for (std::uint32_t i = 0; i < kGpuDepthSlots; ++i) images[kImages++] = scaledDepths[i].image;
      images[kImages++] = scaledTexels.image;
      for (std::uint32_t i = 0; i < kGpuDisplayShadows; ++i) images[kImages++] = scaledShadows[i].image;
      for (std::uint32_t i = 0; i < kGpuDisplayShadows; ++i) images[kImages++] = nativeShadows[i].image;
      images[kImages++] = toneScaled.image;
      images[kImages++] = toneNative.image;
    }
    for (int i = 0; i < kImages; ++i) {
      barriers[i].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
      barriers[i].oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
      barriers[i].newLayout = VK_IMAGE_LAYOUT_GENERAL;
      barriers[i].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
      barriers[i].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
      barriers[i].image = images[i];
      barriers[i].subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
      barriers[i].dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT |
                                  VK_ACCESS_TRANSFER_WRITE_BIT;
    }
    coregs_hw_vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0,
                                nullptr, 0, nullptr, static_cast<std::uint32_t>(kImages), barriers);
    if (scaled()) {
      // The scaled images start as zero words, so a sub-pixel never shows undefined memory (IN-371).
      VkClearColorValue zero{};
      const VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
      for (Image &image : scaledFrames) {
        coregs_hwd_vkCmdClearColorImage(cmd, image.image, VK_IMAGE_LAYOUT_GENERAL, &zero, 1, &range);
      }
      for (Image &image : scaledDepths) {
        coregs_hwd_vkCmdClearColorImage(cmd, image.image, VK_IMAGE_LAYOUT_GENERAL, &zero, 1, &range);
      }
      for (Image &image : scaledShadows) {
        coregs_hwd_vkCmdClearColorImage(cmd, image.image, VK_IMAGE_LAYOUT_GENERAL, &zero, 1, &range);
      }
      for (Image &image : nativeShadows) {
        coregs_hwd_vkCmdClearColorImage(cmd, image.image, VK_IMAGE_LAYOUT_GENERAL, &zero, 1, &range);
      }
    }
    if (!check(coregs_hw_vkEndCommandBuffer(cmd), "vkEndCommandBuffer", error)) return false;
    VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &cmd;
    if (!check(submit_to_queue(submit, fence), "vkQueueSubmit", error)) return false;
    coregs_hw_vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX);
    coregs_hw_vkResetFences(device, 1, &fence);
    return true;
  }

  void destroy() {
    if (recording && device) {
      std::string ignored;
      submit(true, ignored);  // recorded work completes before the resources go
    }
    if (device) device_wait_idle();
    if (device && secondFrameReady) {
      // Both frame sets' rings and fences (the current-set fields are copies; cleared so they are not freed twice).
      save_frame(currentFrame);
      for (FrameResources &f : frameSets) {
        for (Resource *resource : {&f.staging, &f.vertices, &f.uniforms, &f.palettes}) {
          if (resource->mapped) coregs_hwd_vkUnmapMemory(device, resource->memory);
          if (resource->buffer) coregs_hwd_vkDestroyBuffer(device, resource->buffer, nullptr);
          if (resource->memory) coregs_hwd_vkFreeMemory(device, resource->memory, nullptr);
        }
        if (f.fence) coregs_hw_vkDestroyFence(device, f.fence, nullptr);
        f = FrameResources{};
      }
      for (QueryState &q : queries) {
        if (q.pool) coregs_hwd_vkDestroyQueryPool(device, q.pool, nullptr);
        q = QueryState{};
      }
      for (Resource &buffer : downloads) {
        if (buffer.mapped) coregs_hwd_vkUnmapMemory(device, buffer.memory);
        if (buffer.buffer) coregs_hwd_vkDestroyBuffer(device, buffer.buffer, nullptr);
        if (buffer.memory) coregs_hwd_vkFreeMemory(device, buffer.memory, nullptr);
        buffer = Resource{};
      }
      staging = vertices = uniforms = palettes = Resource{};
      fence = VK_NULL_HANDLE;
      secondFrameReady = doubleBuffered = false;
    }
    if (device) {
      for (VkPipeline &pipeline : pipelines) {
        if (pipeline) coregs_hwd_vkDestroyPipeline(device, pipeline, nullptr);
        pipeline = VK_NULL_HANDLE;
      }
      if (interlockTriangles) coregs_hwd_vkDestroyPipeline(device, interlockTriangles, nullptr);
      interlockTriangles = VK_NULL_HANDLE;
      if (interlockModule) coregs_hwd_vkDestroyShaderModule(device, interlockModule, nullptr);
      interlockModule = VK_NULL_HANDLE;
      if (framebuffer) coregs_hwd_vkDestroyFramebuffer(device, framebuffer, nullptr);
      if (renderPass) coregs_hwd_vkDestroyRenderPass(device, renderPass, nullptr);
      if (descriptorPool) coregs_hwd_vkDestroyDescriptorPool(device, descriptorPool, nullptr);
      if (pipelineLayout) coregs_hwd_vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
      if (setLayout) coregs_hwd_vkDestroyDescriptorSetLayout(device, setLayout, nullptr);
      if (vertexModule) coregs_hwd_vkDestroyShaderModule(device, vertexModule, nullptr);
      if (fragmentModule) coregs_hwd_vkDestroyShaderModule(device, fragmentModule, nullptr);
      auto destroyImage = [&](Image *image) {
        if (image->view) coregs_hwd_vkDestroyImageView(device, image->view, nullptr);
        if (image->image) coregs_hwd_vkDestroyImage(device, image->image, nullptr);
        if (image->memory) coregs_hwd_vkFreeMemory(device, image->memory, nullptr);
      };
      for (Image &image : frames) destroyImage(&image);
      for (Image &image : depths) destroyImage(&image);
      destroyImage(&texels);
      destroyImage(&display);
      for (Image &image : scaledFrames) destroyImage(&image);
      for (Image &image : scaledDepths) destroyImage(&image);
      for (Image &image : scaledShadows) destroyImage(&image);
      for (Image &image : nativeShadows) destroyImage(&image);
      destroyImage(&toneScaled);
      for (Image &image : deinterlaceHistory) destroyImage(&image);
      destroyImage(&deinterlaceWoven);
      destroyImage(&deinterlaceOutput);
      if (deinterlacePipeline) coregs_hwd_vkDestroyPipeline(device, deinterlacePipeline, nullptr);
      if (deinterlaceModule) coregs_hwd_vkDestroyShaderModule(device, deinterlaceModule, nullptr);
      if (deinterlaceLayout) coregs_hwd_vkDestroyPipelineLayout(device, deinterlaceLayout, nullptr);
      if (deinterlaceSetLayout) coregs_hwd_vkDestroyDescriptorSetLayout(device, deinterlaceSetLayout, nullptr);
      if (deinterlacePool) coregs_hwd_vkDestroyDescriptorPool(device, deinterlacePool, nullptr);
      deinterlacePipeline = VK_NULL_HANDLE;
      deinterlaceModule = VK_NULL_HANDLE;
      deinterlaceLayout = VK_NULL_HANDLE;
      deinterlaceSetLayout = VK_NULL_HANDLE;
      deinterlacePool = VK_NULL_HANDLE;
      deinterlaceSet = VK_NULL_HANDLE;
      destroyImage(&toneNative);
      if (toneTable.buffer) coregs_hwd_vkDestroyBuffer(device, toneTable.buffer, nullptr);
      if (toneTable.memory) coregs_hwd_vkFreeMemory(device, toneTable.memory, nullptr);
      toneTable = Resource{};
      if (tonePipeline) coregs_hwd_vkDestroyPipeline(device, tonePipeline, nullptr);
      if (toneModule) coregs_hwd_vkDestroyShaderModule(device, toneModule, nullptr);
      if (toneLayout) coregs_hwd_vkDestroyPipelineLayout(device, toneLayout, nullptr);
      if (toneSetLayout) coregs_hwd_vkDestroyDescriptorSetLayout(device, toneSetLayout, nullptr);
      tonePipeline = VK_NULL_HANDLE;
      toneModule = VK_NULL_HANDLE;
      toneLayout = VK_NULL_HANDLE;
      toneSetLayout = VK_NULL_HANDLE;
      destroyImage(&scaledTexels);
      destroyImage(&fxaaImage);
      if (fxaaPipeline) coregs_hwd_vkDestroyPipeline(device, fxaaPipeline, nullptr);
      if (fxaaModule) coregs_hwd_vkDestroyShaderModule(device, fxaaModule, nullptr);
      if (fxaaLayout) coregs_hwd_vkDestroyPipelineLayout(device, fxaaLayout, nullptr);
      if (fxaaSetLayout) coregs_hwd_vkDestroyDescriptorSetLayout(device, fxaaSetLayout, nullptr);
      if (fxaaPool) coregs_hwd_vkDestroyDescriptorPool(device, fxaaPool, nullptr);
      fxaaPipeline = VK_NULL_HANDLE;
      fxaaModule = VK_NULL_HANDLE;
      fxaaLayout = VK_NULL_HANDLE;
      fxaaSetLayout = VK_NULL_HANDLE;
      fxaaPool = VK_NULL_HANDLE;
      fxaaSet = VK_NULL_HANDLE;
      if (scaledFramebuffer) coregs_hwd_vkDestroyFramebuffer(device, scaledFramebuffer, nullptr);
      scaledFramebuffer = VK_NULL_HANDLE;
      for (VkPipeline *pipeline : {&upsamplePipeline, &composeScaledPipeline}) {
        if (*pipeline) coregs_hwd_vkDestroyPipeline(device, *pipeline, nullptr);
        *pipeline = VK_NULL_HANDLE;
      }
      for (VkShaderModule *module : {&upsampleModule, &composeScaledModule}) {
        if (*module) coregs_hwd_vkDestroyShaderModule(device, *module, nullptr);
        *module = VK_NULL_HANDLE;
      }
      for (VkPipelineLayout *layout : {&upsampleLayout, &composeScaledLayout}) {
        if (*layout) coregs_hwd_vkDestroyPipelineLayout(device, *layout, nullptr);
        *layout = VK_NULL_HANDLE;
      }
      for (VkDescriptorSetLayout *layout : {&upsampleSetLayout, &composeScaledSetLayout}) {
        if (*layout) coregs_hwd_vkDestroyDescriptorSetLayout(device, *layout, nullptr);
        *layout = VK_NULL_HANDLE;
      }
      if (scalePool) coregs_hwd_vkDestroyDescriptorPool(device, scalePool, nullptr);
      scalePool = VK_NULL_HANDLE;
      if (clutPipeline) coregs_hwd_vkDestroyPipeline(device, clutPipeline, nullptr);
      if (clutModule) coregs_hwd_vkDestroyShaderModule(device, clutModule, nullptr);
      clutPipeline = VK_NULL_HANDLE;
      clutModule = VK_NULL_HANDLE;
      if (unswizzlePipeline) coregs_hwd_vkDestroyPipeline(device, unswizzlePipeline, nullptr);
      if (unswizzleModule) coregs_hwd_vkDestroyShaderModule(device, unswizzleModule, nullptr);
      unswizzlePipeline = VK_NULL_HANDLE;
      unswizzleModule = VK_NULL_HANDLE;
      if (mergePipeline) coregs_hwd_vkDestroyPipeline(device, mergePipeline, nullptr);
      if (mergeModule) coregs_hwd_vkDestroyShaderModule(device, mergeModule, nullptr);
      mergePipeline = VK_NULL_HANDLE;
      mergeModule = VK_NULL_HANDLE;
      if (displayPipeline) coregs_hwd_vkDestroyPipeline(device, displayPipeline, nullptr);
      if (displayModule) coregs_hwd_vkDestroyShaderModule(device, displayModule, nullptr);
      displayPipeline = VK_NULL_HANDLE;
      displayModule = VK_NULL_HANDLE;
      if (hostUploadPipeline) coregs_hwd_vkDestroyPipeline(device, hostUploadPipeline, nullptr);
      if (hostUploadModule) coregs_hwd_vkDestroyShaderModule(device, hostUploadModule, nullptr);
      hostUploadPipeline = VK_NULL_HANDLE;
      hostUploadModule = VK_NULL_HANDLE;
      if (toneStripsPipeline) coregs_hwd_vkDestroyPipeline(device, toneStripsPipeline, nullptr);
      if (toneStripsModule) coregs_hwd_vkDestroyShaderModule(device, toneStripsModule, nullptr);
      toneStripsPipeline = VK_NULL_HANDLE;
      toneStripsModule = VK_NULL_HANDLE;
      if (decodePipeline) coregs_hwd_vkDestroyPipeline(device, decodePipeline, nullptr);
      if (swizzlePipeline) coregs_hwd_vkDestroyPipeline(device, swizzlePipeline, nullptr);
      if (decodeModule) coregs_hwd_vkDestroyShaderModule(device, decodeModule, nullptr);
      if (swizzleModule) coregs_hwd_vkDestroyShaderModule(device, swizzleModule, nullptr);
      if (computePool) coregs_hwd_vkDestroyDescriptorPool(device, computePool, nullptr);
      if (computeLayout) coregs_hwd_vkDestroyPipelineLayout(device, computeLayout, nullptr);
      if (computeSetLayout) coregs_hwd_vkDestroyDescriptorSetLayout(device, computeSetLayout, nullptr);
      decodePipeline = swizzlePipeline = VK_NULL_HANDLE;
      decodeModule = swizzleModule = VK_NULL_HANDLE;
      computePool = VK_NULL_HANDLE;
      computeLayout = VK_NULL_HANDLE;
      computeSetLayout = VK_NULL_HANDLE;
      for (Resource *resource : {&staging, &readback, &vertices, &uniforms, &glm, &glmTables, &palettes, &clut}) {
        if (resource->mapped) coregs_hwd_vkUnmapMemory(device, resource->memory);
        if (resource->buffer) coregs_hwd_vkDestroyBuffer(device, resource->buffer, nullptr);
        if (resource->memory) coregs_hwd_vkFreeMemory(device, resource->memory, nullptr);
      }
      if (fence) coregs_hw_vkDestroyFence(device, fence, nullptr);
      if (pool) coregs_hw_vkDestroyCommandPool(device, pool, nullptr);
    }
    pipelineLayout = VK_NULL_HANDLE;
    setLayout = VK_NULL_HANDLE;
    descriptorPool = VK_NULL_HANDLE;
    renderPass = VK_NULL_HANDLE;
    framebuffer = VK_NULL_HANDLE;
    vertexModule = VK_NULL_HANDLE;
    fragmentModule = VK_NULL_HANDLE;
    fence = VK_NULL_HANDLE;
    pool = VK_NULL_HANDLE;
    for (Image &image : frames) image = Image{};
    for (Image &image : depths) image = Image{};
    texels = Image{};
    display = Image{};
    for (Image &image : scaledFrames) image = Image{};
    for (Image &image : scaledDepths) image = Image{};
    for (Image &image : scaledShadows) image = Image{};
    for (Image &image : nativeShadows) image = Image{};
    toneScaled = Image{};
    for (Image &image : deinterlaceHistory) image = Image{};
    deinterlaceWoven = Image{};
    deinterlaceOutput = Image{};
    toneNative = Image{};
    scaledTexels = Image{};
    fxaaImage = Image{};
    staging = Resource{};
    readback = Resource{};
    vertices = Resource{};
    uniforms = Resource{};
    glm = Resource{};
    glmTables = Resource{};
    palettes = Resource{};
    clut = Resource{};
    ready = false;
  }
};

GpuDrawer::GpuDrawer() : impl_(std::make_unique<Impl>()) {}

GpuDrawer::~GpuDrawer() {
  if (impl_) impl_->destroy();
}

bool GpuDrawer::Ready() const { return impl_->ready; }

const GpuStats &GpuDrawer::Stats() const { return impl_->stats; }

bool GpuDrawer::Init(const VulkanHandles &handles, std::string &error) {
  Impl &s = *impl_;
  if (s.ready) return true;
  s.instance = static_cast<VkInstance>(handles.instance);
  s.physical = static_cast<VkPhysicalDevice>(handles.physical);
  s.device = static_cast<VkDevice>(handles.device);
  s.queue = static_cast<VkQueue>(handles.queue);
  s.queueLock = handles.queueLock;
  s.storageImageArrayDynamicIndexing = handles.profile.storage_image_array_dynamic_indexing_enabled;  // TASK-179
  if (!s.instance || !s.physical || !s.device || !s.queue) {
    error = "GpuDrawer::Init needs instance, physical device, device and queue";
    return false;
  }
  std::vector<std::string> missing;
  if (!coregs::vulkan_device::load_device_entry_points(s.instance, missing)) {
    error = "missing Vulkan entry points:";
    for (const std::string &name : missing) error += " " + name;
    return false;
  }
  coregs_hwd_vkGetPhysicalDeviceMemoryProperties(s.physical, &s.memory);

  VkCommandPoolCreateInfo poolInfo{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
  poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
  poolInfo.queueFamilyIndex = handles.queueFamily;
  if (!check(coregs_hw_vkCreateCommandPool(s.device, &poolInfo, nullptr, &s.pool), "vkCreateCommandPool", error)) return false;
  VkCommandBufferAllocateInfo allocate{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
  allocate.commandPool = s.pool;
  allocate.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  allocate.commandBufferCount = 1;
  if (!check(coregs_hw_vkAllocateCommandBuffers(s.device, &allocate, &s.cmd), "vkAllocateCommandBuffers", error)) return false;
  VkFenceCreateInfo fenceInfo{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
  if (!check(coregs_hw_vkCreateFence(s.device, &fenceInfo, nullptr, &s.fence), "vkCreateFence", error)) return false;

  s.vertexModule = s.make_module(coregs::shaders::hw_vertex_smooth_words, coregs::shaders::hw_vertex_smooth_word_count, error);
  s.fragmentModule = s.make_module(coregs::shaders::hw_pixel_uber_smooth_words,
                                   coregs::shaders::hw_pixel_uber_smooth_word_count, error);
  if (!s.vertexModule || !s.fragmentModule) return false;

  VkDescriptorSetLayoutBinding bindings[5]{};
  bindings[0] = {0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr};
  bindings[1] = {1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr};
  bindings[2] = {2, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr};
  bindings[3] = {3, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr};
  // IN-374: binding 4 = the scaled texel image (scaled sets); 1x sets point it at the texel atlas (never read).
  bindings[4] = {4, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr};
  VkDescriptorSetLayoutCreateInfo setInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
  setInfo.bindingCount = 5;
  setInfo.pBindings = bindings;
  if (!check(coregs_hwd_vkCreateDescriptorSetLayout(s.device, &setInfo, nullptr, &s.setLayout), "vkCreateDescriptorSetLayout", error)) return false;

  VkPushConstantRange push{};
  push.stageFlags = kGraphicsPushStages;
  push.size = kPushBytes;
  VkPipelineLayoutCreateInfo layoutInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
  layoutInfo.setLayoutCount = 1;
  layoutInfo.pSetLayouts = &s.setLayout;
  layoutInfo.pushConstantRangeCount = 1;
  layoutInfo.pPushConstantRanges = &push;
  if (!check(coregs_hwd_vkCreatePipelineLayout(s.device, &layoutInfo, nullptr, &s.pipelineLayout), "vkCreatePipelineLayout", error)) return false;

  constexpr std::uint32_t kSets = kGpuColourSlots * kGpuDepthSlots;
  VkDescriptorPoolSize sizes[2] = {{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 4 * kSets},
                                   {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 4 * 4 * kSets}};
  VkDescriptorPoolCreateInfo poolSetInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
  poolSetInfo.maxSets = 4 * kSets;  // two frame sets (IN-354), each with 1x and scaled sets (IN-371)
  poolSetInfo.poolSizeCount = 2;
  poolSetInfo.pPoolSizes = sizes;
  if (!check(coregs_hwd_vkCreateDescriptorPool(s.device, &poolSetInfo, nullptr, &s.descriptorPool), "vkCreateDescriptorPool", error)) return false;
  VkDescriptorSetAllocateInfo setAllocate{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
  setAllocate.descriptorPool = s.descriptorPool;
  VkDescriptorSetLayout layouts[kSets];
  for (VkDescriptorSetLayout &layout : layouts) layout = s.setLayout;
  setAllocate.descriptorSetCount = kSets;
  setAllocate.pSetLayouts = layouts;
  if (!check(coregs_hwd_vkAllocateDescriptorSets(s.device, &setAllocate, s.descriptorSets), "vkAllocateDescriptorSets", error)) return false;

  if (!s.make_buffer(s.staging, kStagingBytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, 0, true, error)) return false;
  if (!s.make_buffer(s.readback, kReadbackBytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                     VK_MEMORY_PROPERTY_HOST_CACHED_BIT, true, error)) return false;
  if (!s.make_buffer(s.vertices, kVertexBytes, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, 0, true, error)) return false;
  {
    VkPhysicalDeviceProperties properties{};
    coregs_hw_vkGetPhysicalDeviceProperties(s.physical, &properties);
    const VkDeviceSize align = std::max<VkDeviceSize>(properties.limits.minUniformBufferOffsetAlignment, 1);
    s.uniformStride = (kUniformBytes + align - 1) / align * align;
  }
  if (!s.make_buffer(s.uniforms, kUniformRingBytes, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, 0, true, error)) return false;

  for (Impl::Image &image : s.frames) {
    if (!s.make_image(image, kSurfaceSize, kSurfaceSize, error)) return false;
  }
  for (Impl::Image &image : s.depths) {
    if (!s.make_image(image, kSurfaceSize, kSurfaceSize, error)) return false;
  }
  if (!s.make_image(s.texels, kGpuTexelAtlasSide, kGpuTexelAtlasSide, error)) return false;
  s.scaledSide = s.scale > 1 ? static_cast<std::uint32_t>(kGpuScaledExtent) * s.scale : 0u;
  if (!s.make_image(s.display, s.displayWidth, s.displayHeight, error)) return false;
  if (s.scaled()) {
    // IN-384: a small image per slot to give every descriptor a view; a slot grows its own on first use. The
    // 64 x 32 native minimum is one page (384 x 192 at 6x, 281 KB), so nothing is allocated for a target that is
    // never used at scale.
    for (std::uint32_t i = 0; i < kGpuColourSlots; ++i) {
      if (!s.ensure_scaled_image(s.scaledFrames[i], s.scaledFrameExtent[i], 64, 32, error)) return false;
    }
    for (std::uint32_t i = 0; i < kGpuDepthSlots; ++i) {
      if (!s.ensure_scaled_image(s.scaledDepths[i], s.scaledDepthExtent[i], 64, 32, error)) return false;
    }
    if (!s.ensure_scaled_image(s.scaledTexels, s.scaledTexelExtent, 64, 32, error)) return false;
    for (std::uint32_t i = 0; i < kGpuDisplayShadows; ++i) {
      if (!s.ensure_scaled_image(s.scaledShadows[i], s.scaledShadowExtent[i], 64, 32, error)) return false;
      // The shadow's 1x words are read at 1x by the compose kernel: this image is not scaled.
      if (!s.ensure_scaled_image(s.nativeShadows[i], s.shadowNativeExtent[i], 64, 32, error, 1)) return false;
    }
    if (!s.ensure_scaled_image(s.toneScaled, s.toneScaledExtent, 64, 32, error)) return false;
    if (!s.make_image(s.toneNative, static_cast<std::uint32_t>(kGpuScaledExtent), static_cast<std::uint32_t>(kGpuShadowRows),
                      error)) {
      return false;
    }
    if (!s.make_buffer(s.toneTable, 768 * sizeof(std::uint32_t), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                       VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 0, false, error)) {
      return false;
    }
  }

  VkDescriptorBufferInfo uniformInfo{s.uniforms.buffer, 0, kUniformBytes};
  for (std::uint32_t colour = 0; colour < kGpuColourSlots; ++colour) {
    for (std::uint32_t depth = 0; depth < kGpuDepthSlots; ++depth) {
      const VkDescriptorSet set = s.descriptorSets[colour * kGpuDepthSlots + depth];
      VkDescriptorImageInfo imageInfos[4] = {
          {VK_NULL_HANDLE, s.frames[colour].view, VK_IMAGE_LAYOUT_GENERAL},
          {VK_NULL_HANDLE, s.depths[depth].view, VK_IMAGE_LAYOUT_GENERAL},
          {VK_NULL_HANDLE, s.texels.view, VK_IMAGE_LAYOUT_GENERAL},
          {VK_NULL_HANDLE, s.texels.view, VK_IMAGE_LAYOUT_GENERAL}};
      VkWriteDescriptorSet writes[5]{};
      writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
      writes[0].dstSet = set;
      writes[0].dstBinding = 0;
      writes[0].descriptorCount = 1;
      writes[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC;
      writes[0].pBufferInfo = &uniformInfo;
      for (int i = 0; i < 4; ++i) {
        writes[i + 1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i + 1].dstSet = set;
        writes[i + 1].dstBinding = static_cast<std::uint32_t>(i + 1);
        writes[i + 1].descriptorCount = 1;
        writes[i + 1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        writes[i + 1].pImageInfo = &imageInfos[i];
      }
      coregs_hwd_vkUpdateDescriptorSets(s.device, 5, writes, 0, nullptr);
    }
  }

  VkSubpassDescription subpass{};
  subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
  VkRenderPassCreateInfo passInfo{VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO};
  passInfo.subpassCount = 1;
  passInfo.pSubpasses = &subpass;
  if (!check(coregs_hwd_vkCreateRenderPass(s.device, &passInfo, nullptr, &s.renderPass), "vkCreateRenderPass", error)) return false;
  VkFramebufferCreateInfo fbInfo{VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
  fbInfo.renderPass = s.renderPass;
  fbInfo.attachmentCount = 0;
  fbInfo.width = kSurfaceSize;
  fbInfo.height = kSurfaceSize;
  fbInfo.layers = 1;
  if (!check(coregs_hwd_vkCreateFramebuffer(s.device, &fbInfo, nullptr, &s.framebuffer), "vkCreateFramebuffer", error)) return false;

  s.pipelines[0] = s.make_pipeline(VK_PRIMITIVE_TOPOLOGY_POINT_LIST, error);
  s.pipelines[1] = s.make_pipeline(VK_PRIMITIVE_TOPOLOGY_LINE_LIST, error);
  s.pipelines[2] = s.make_pipeline(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST, error);
  if (!s.pipelines[0] || !s.pipelines[1] || !s.pipelines[2]) return false;
  if (handles.pixelInterlock) {
    s.interlockModule = s.make_module(coregs::shaders::hw_pixel_uber_smooth_interlock_words,
                                      coregs::shaders::hw_pixel_uber_smooth_interlock_word_count, error);
    if (!s.interlockModule) return false;
    s.interlockTriangles = s.make_pipeline(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST, error, s.interlockModule);
    if (!s.interlockTriangles) return false;
    s.interlock = true;
  }

  if (!s.make_images_general(error)) return false;

  // ---- GLM (IN-347) ----
  {
    const memory::GpuAddressTables tables = memory::gpuAddressTables();
    std::vector<std::uint32_t> words(tables.directory);
    words.insert(words.end(), tables.entries.begin(), tables.entries.end());
    const VkDeviceSize tableBytes = words.size() * sizeof(std::uint32_t);
    if (!s.make_buffer(s.glm, kGlmBytes,
                       VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                       VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 0, false, error)) return false;
    if (!s.make_buffer(s.glmTables, tableBytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                       VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, 0, true, error)) return false;
    std::memcpy(s.glmTables.mapped, words.data(), tableBytes);
    if (!s.make_buffer(s.palettes, kPaletteRingBytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                       VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, 0, true, error)) return false;
    if (!s.make_buffer(s.clut, kClutBytes,
                       VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                       VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 0, false, error)) return false;

    VkDescriptorSetLayoutBinding computeBindings[6]{};
    computeBindings[0] = {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    computeBindings[1] = {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    computeBindings[2] = {2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    computeBindings[3] = {3, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    computeBindings[4] = {4, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    computeBindings[5] = {5, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    VkDescriptorSetLayoutCreateInfo computeSetInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    computeSetInfo.bindingCount = 6;
    computeSetInfo.pBindings = computeBindings;
    if (!check(coregs_hwd_vkCreateDescriptorSetLayout(s.device, &computeSetInfo, nullptr, &s.computeSetLayout),
               "vkCreateDescriptorSetLayout", error)) return false;
    VkPushConstantRange computePush{VK_SHADER_STAGE_COMPUTE_BIT, 0, 32 * sizeof(std::uint32_t)};  // 128 bytes
    VkPipelineLayoutCreateInfo computeLayoutInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    computeLayoutInfo.setLayoutCount = 1;
    computeLayoutInfo.pSetLayouts = &s.computeSetLayout;
    computeLayoutInfo.pushConstantRangeCount = 1;
    computeLayoutInfo.pPushConstantRanges = &computePush;
    if (!check(coregs_hwd_vkCreatePipelineLayout(s.device, &computeLayoutInfo, nullptr, &s.computeLayout),
               "vkCreatePipelineLayout", error)) return false;
    constexpr std::uint32_t kComputeSets = kGpuColourSlots + kGpuDepthSlots;
    VkDescriptorPoolSize computeSizes[2] = {{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2 * 4 * (kComputeSets + 1)},
                                            {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 2 * 2 * (kComputeSets + 1)}};
    VkDescriptorPoolCreateInfo computePoolInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    computePoolInfo.maxSets = 2 * (kComputeSets + 1);  // two frame sets (IN-354), each with a display set (IN-364)
    computePoolInfo.poolSizeCount = 2;
    computePoolInfo.pPoolSizes = computeSizes;
    if (!check(coregs_hwd_vkCreateDescriptorPool(s.device, &computePoolInfo, nullptr, &s.computePool),
               "vkCreateDescriptorPool", error)) return false;
    VkDescriptorSetLayout computeLayouts[kComputeSets];
    for (VkDescriptorSetLayout &layout : computeLayouts) layout = s.computeSetLayout;
    VkDescriptorSetAllocateInfo computeAllocate{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    computeAllocate.descriptorPool = s.computePool;
    computeAllocate.descriptorSetCount = kComputeSets;
    computeAllocate.pSetLayouts = computeLayouts;
    if (!check(coregs_hwd_vkAllocateDescriptorSets(s.device, &computeAllocate, s.computeSets), "vkAllocateDescriptorSets",
               error)) return false;
    VkDescriptorBufferInfo buffers[3] = {{s.glm.buffer, 0, kGlmBytes},
                                         {s.glmTables.buffer, 0, tableBytes},
                                         {s.palettes.buffer, 0, kPaletteRingBytes}};
    VkDescriptorBufferInfo clutInfo{s.clut.buffer, 0, kClutBytes};
    for (std::uint32_t set = 0; set < kComputeSets; ++set) {
      const VkImageView source = set < kGpuColourSlots ? s.frames[set].view : s.depths[set - kGpuColourSlots].view;
      VkDescriptorImageInfo images[2] = {{VK_NULL_HANDLE, s.texels.view, VK_IMAGE_LAYOUT_GENERAL},
                                         {VK_NULL_HANDLE, source, VK_IMAGE_LAYOUT_GENERAL}};
      VkWriteDescriptorSet writes[6]{};
      for (std::uint32_t i = 0; i < 6; ++i) {
        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet = s.computeSets[set];
        writes[i].dstBinding = i;
        writes[i].descriptorCount = 1;
        if (i < 3 || i == 5) {
          writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
          writes[i].pBufferInfo = i == 5 ? &clutInfo : &buffers[i];
        } else {
          writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
          writes[i].pImageInfo = &images[i - 3];
        }
      }
      coregs_hwd_vkUpdateDescriptorSets(s.device, 6, writes, 0, nullptr);
    }
    s.decodeModule = s.make_module(coregs::shaders::hw_glm_decode_words, coregs::shaders::hw_glm_decode_word_count, error);
    s.swizzleModule = s.make_module(coregs::shaders::hw_glm_swizzle_words, coregs::shaders::hw_glm_swizzle_word_count, error);
    if (!s.decodeModule || !s.swizzleModule) return false;
    s.decodePipeline = s.make_compute(s.decodeModule, error);
    s.swizzlePipeline = s.make_compute(s.swizzleModule, error);
    if (!s.decodePipeline || !s.swizzlePipeline) return false;
    s.clutModule = s.make_module(coregs::shaders::hw_glm_clut_words, coregs::shaders::hw_glm_clut_word_count, error);
    if (!s.clutModule) return false;
    s.clutPipeline = s.make_compute(s.clutModule, error);
    if (!s.clutPipeline) return false;
    s.unswizzleModule =
        s.make_module(coregs::shaders::hw_glm_unswizzle_words, coregs::shaders::hw_glm_unswizzle_word_count, error);
    if (!s.unswizzleModule) return false;
    s.unswizzlePipeline = s.make_compute(s.unswizzleModule, error);
    if (!s.unswizzlePipeline) return false;
    s.mergeModule = s.make_module(coregs::shaders::hw_glm_merge_words, coregs::shaders::hw_glm_merge_word_count, error);
    if (!s.mergeModule) return false;
    s.mergePipeline = s.make_compute(s.mergeModule, error);
    if (!s.mergePipeline) return false;
    s.hostUploadModule =
        s.make_module(coregs::shaders::hw_glm_host_upload_words, coregs::shaders::hw_glm_host_upload_word_count, error);
    if (!s.hostUploadModule) return false;
    s.hostUploadPipeline = s.make_compute(s.hostUploadModule, error);
    if (!s.hostUploadPipeline) return false;
    s.toneStripsModule =
        s.make_module(coregs::shaders::hw_tone_strips_words, coregs::shaders::hw_tone_strips_word_count, error);
    if (!s.toneStripsModule) return false;
    s.toneStripsPipeline = s.make_compute(s.toneStripsModule, error);
    if (!s.toneStripsPipeline) return false;
    s.displayModule =
        s.make_module(coregs::shaders::hw_display_compose_words, coregs::shaders::hw_display_compose_word_count, error);
    if (!s.displayModule) return false;
    s.displayPipeline = s.make_compute(s.displayModule, error);
    if (!s.displayPipeline) return false;
    if (!s.make_display_set(0, s.computeSets[0], error)) return false;
  }
  if (s.scaled() && !s.create_scaled_resources(error)) return false;
  if (!s.create_second_frame(error)) return false;
  s.ready = true;
  return true;
}

// IN-376: the FXAA image (display-sized, GENERAL layout), its descriptor set and kernel. Records the layout change
// into the current recording.
bool GpuDrawer::Impl::create_deinterlace(std::string &error) {
  if (deinterlacePipeline) return true;
  const std::uint32_t width = kGpuDeinterlaceWidth * scale;
  const std::uint32_t height = kGpuDeinterlaceRows * scale;
  Image *images[kGpuDeinterlaceHistory + 2];
  for (std::uint32_t i = 0; i < kGpuDeinterlaceHistory; ++i) images[i] = &deinterlaceHistory[i];
  images[kGpuDeinterlaceHistory] = &deinterlaceWoven;
  images[kGpuDeinterlaceHistory + 1] = &deinterlaceOutput;
  for (Image *image : images) {
    if (!make_image(*image, width, height, error)) return false;
    VkImageMemoryBarrier toGeneral{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    toGeneral.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    toGeneral.newLayout = VK_IMAGE_LAYOUT_GENERAL;
    toGeneral.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    toGeneral.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    toGeneral.image = image->image;
    toGeneral.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    toGeneral.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT |
                              VK_ACCESS_TRANSFER_WRITE_BIT;
    coregs_hw_vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0,
                                nullptr, 0, nullptr, 1, &toGeneral);
  }
  VkDescriptorSetLayoutBinding bindings[4]{};
  bindings[0] = {0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, kGpuDeinterlaceHistory, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
  bindings[1] = {4, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
  bindings[2] = {5, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
  bindings[3] = {6, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
  VkDescriptorSetLayoutCreateInfo setInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
  setInfo.bindingCount = 4;
  setInfo.pBindings = bindings;
  if (!check(coregs_hwd_vkCreateDescriptorSetLayout(device, &setInfo, nullptr, &deinterlaceSetLayout),
             "vkCreateDescriptorSetLayout", error)) {
    return false;
  }
  VkPushConstantRange push{VK_SHADER_STAGE_COMPUTE_BIT, 0, 14 * sizeof(std::uint32_t)};
  VkPipelineLayoutCreateInfo layoutInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
  layoutInfo.setLayoutCount = 1;
  layoutInfo.pSetLayouts = &deinterlaceSetLayout;
  layoutInfo.pushConstantRangeCount = 1;
  layoutInfo.pPushConstantRanges = &push;
  if (!check(coregs_hwd_vkCreatePipelineLayout(device, &layoutInfo, nullptr, &deinterlaceLayout),
             "vkCreatePipelineLayout", error)) {
    return false;
  }
  VkDescriptorPoolSize size{VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, kGpuDeinterlaceHistory + 3};
  VkDescriptorPoolCreateInfo poolInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
  poolInfo.maxSets = 1;
  poolInfo.poolSizeCount = 1;
  poolInfo.pPoolSizes = &size;
  if (!check(coregs_hwd_vkCreateDescriptorPool(device, &poolInfo, nullptr, &deinterlacePool), "vkCreateDescriptorPool",
             error)) {
    return false;
  }
  VkDescriptorSetAllocateInfo allocate{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
  allocate.descriptorPool = deinterlacePool;
  allocate.descriptorSetCount = 1;
  allocate.pSetLayouts = &deinterlaceSetLayout;
  if (!check(coregs_hwd_vkAllocateDescriptorSets(device, &allocate, &deinterlaceSet), "vkAllocateDescriptorSets", error)) {
    return false;
  }
  VkDescriptorImageInfo history[kGpuDeinterlaceHistory];
  for (std::uint32_t i = 0; i < kGpuDeinterlaceHistory; ++i) {
    history[i] = {VK_NULL_HANDLE, deinterlaceHistory[i].view, VK_IMAGE_LAYOUT_GENERAL};
  }
  VkDescriptorImageInfo others[3] = {{VK_NULL_HANDLE, deinterlaceWoven.view, VK_IMAGE_LAYOUT_GENERAL},
                                     {VK_NULL_HANDLE, deinterlaceOutput.view, VK_IMAGE_LAYOUT_GENERAL},
                                     {VK_NULL_HANDLE, display.view, VK_IMAGE_LAYOUT_GENERAL}};
  VkWriteDescriptorSet writes[4]{};
  for (std::uint32_t i = 0; i < 4; ++i) {
    writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[i].dstSet = deinterlaceSet;
    writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
  }
  writes[0].dstBinding = 0;
  writes[0].descriptorCount = kGpuDeinterlaceHistory;
  writes[0].pImageInfo = history;
  for (std::uint32_t i = 0; i < 3; ++i) {
    writes[i + 1].dstBinding = 4 + i;
    writes[i + 1].descriptorCount = 1;
    writes[i + 1].pImageInfo = &others[i];
  }
  coregs_hwd_vkUpdateDescriptorSets(device, 4, writes, 0, nullptr);
  deinterlaceModule = storageImageArrayDynamicIndexing
                          ? make_module(coregs::shaders::hw_deinterlace_scaled_words,
                                        coregs::shaders::hw_deinterlace_scaled_word_count, error)
                          : make_module(coregs::shaders::hw_deinterlace_scaled_constant_words,
                                        coregs::shaders::hw_deinterlace_scaled_constant_word_count, error);
  if (!deinterlaceModule) return false;
  VkComputePipelineCreateInfo info{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
  info.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  info.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
  info.stage.module = deinterlaceModule;
  info.stage.pName = "main";
  info.layout = deinterlaceLayout;
  return check(coregs_hwd_vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &info, nullptr, &deinterlacePipeline),
               "vkCreateComputePipelines", error);
}

bool GpuDrawer::Impl::create_fxaa(std::string &error) {
  if (fxaaPipeline) return true;
  fxaaWidth = displayWidth;
  fxaaHeight = displayHeight;
  if (!make_image(fxaaImage, fxaaWidth, fxaaHeight, error)) return false;
  VkImageMemoryBarrier toGeneral{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
  toGeneral.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  toGeneral.newLayout = VK_IMAGE_LAYOUT_GENERAL;
  toGeneral.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  toGeneral.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  toGeneral.image = fxaaImage.image;
  toGeneral.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
  toGeneral.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT;
  coregs_hw_vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr,
                              0, nullptr, 1, &toGeneral);
  VkDescriptorSetLayoutBinding bindings[2]{};
  bindings[0] = {0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
  bindings[1] = {1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
  VkDescriptorSetLayoutCreateInfo setInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
  setInfo.bindingCount = 2;
  setInfo.pBindings = bindings;
  if (!check(coregs_hwd_vkCreateDescriptorSetLayout(device, &setInfo, nullptr, &fxaaSetLayout),
             "vkCreateDescriptorSetLayout", error)) {
    return false;
  }
  VkPushConstantRange push{VK_SHADER_STAGE_COMPUTE_BIT, 0, 2 * sizeof(std::uint32_t)};
  VkPipelineLayoutCreateInfo layoutInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
  layoutInfo.setLayoutCount = 1;
  layoutInfo.pSetLayouts = &fxaaSetLayout;
  layoutInfo.pushConstantRangeCount = 1;
  layoutInfo.pPushConstantRanges = &push;
  if (!check(coregs_hwd_vkCreatePipelineLayout(device, &layoutInfo, nullptr, &fxaaLayout), "vkCreatePipelineLayout",
             error)) {
    return false;
  }
  VkDescriptorPoolSize size{VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 2};
  VkDescriptorPoolCreateInfo poolInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
  poolInfo.maxSets = 1;
  poolInfo.poolSizeCount = 1;
  poolInfo.pPoolSizes = &size;
  if (!check(coregs_hwd_vkCreateDescriptorPool(device, &poolInfo, nullptr, &fxaaPool), "vkCreateDescriptorPool", error)) {
    return false;
  }
  VkDescriptorSetAllocateInfo allocate{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
  allocate.descriptorPool = fxaaPool;
  allocate.descriptorSetCount = 1;
  allocate.pSetLayouts = &fxaaSetLayout;
  if (!check(coregs_hwd_vkAllocateDescriptorSets(device, &allocate, &fxaaSet), "vkAllocateDescriptorSets", error)) {
    return false;
  }
  VkDescriptorImageInfo images[2] = {{VK_NULL_HANDLE, display.view, VK_IMAGE_LAYOUT_GENERAL},
                                     {VK_NULL_HANDLE, fxaaImage.view, VK_IMAGE_LAYOUT_GENERAL}};
  VkWriteDescriptorSet writes[2]{};
  for (std::uint32_t i = 0; i < 2; ++i) {
    writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[i].dstSet = fxaaSet;
    writes[i].dstBinding = i;
    writes[i].descriptorCount = 1;
    writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    writes[i].pImageInfo = &images[i];
  }
  coregs_hwd_vkUpdateDescriptorSets(device, 2, writes, 0, nullptr);
  fxaaModule = make_module(coregs::shaders::hw_postfx_fxaa_words, coregs::shaders::hw_postfx_fxaa_word_count, error);
  if (!fxaaModule) return false;
  VkComputePipelineCreateInfo info{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
  info.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  info.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
  info.stage.module = fxaaModule;
  info.stage.pName = "main";
  info.layout = fxaaLayout;
  return check(coregs_hwd_vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &info, nullptr, &fxaaPipeline),
               "vkCreateComputePipelines", error);
}

// IN-371/IN-372: framebuffer, descriptor sets and kernels of the scaled images (frame set 0; create_second_frame
// adds frame set 1's).
bool GpuDrawer::Impl::create_scaled_resources(std::string &error) {
  VkFramebufferCreateInfo fbInfo{VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
  fbInfo.renderPass = renderPass;
  fbInfo.attachmentCount = 0;
  fbInfo.width = scaledSide;
  fbInfo.height = scaledSide;
  fbInfo.layers = 1;
  if (!check(coregs_hwd_vkCreateFramebuffer(device, &fbInfo, nullptr, &scaledFramebuffer), "vkCreateFramebuffer", error)) {
    return false;
  }
  // Graphics sets: binding 0 the frame set's uniform ring, 1 and 2 the scaled images, 3 the texel atlas.
  constexpr std::uint32_t kSets = kGpuColourSlots * kGpuDepthSlots;
  VkDescriptorSetLayout layouts[kSets];
  for (VkDescriptorSetLayout &layout : layouts) layout = setLayout;
  VkDescriptorSetAllocateInfo setAllocate{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
  setAllocate.descriptorPool = descriptorPool;
  setAllocate.descriptorSetCount = kSets;
  setAllocate.pSetLayouts = layouts;
  if (!check(coregs_hwd_vkAllocateDescriptorSets(device, &setAllocate, scaledDescriptorSets), "vkAllocateDescriptorSets",
             error)) {
    return false;
  }
  VkDescriptorBufferInfo uniformInfo{uniforms.buffer, 0, kUniformBytes};
  for (std::uint32_t colour = 0; colour < kGpuColourSlots; ++colour) {
    for (std::uint32_t depth = 0; depth < kGpuDepthSlots; ++depth) {
      const VkDescriptorSet set = scaledDescriptorSets[colour * kGpuDepthSlots + depth];
      VkDescriptorImageInfo imageInfos[4] = {{VK_NULL_HANDLE, scaledFrames[colour].view, VK_IMAGE_LAYOUT_GENERAL},
                                             {VK_NULL_HANDLE, scaledDepths[depth].view, VK_IMAGE_LAYOUT_GENERAL},
                                             {VK_NULL_HANDLE, texels.view, VK_IMAGE_LAYOUT_GENERAL},
                                             {VK_NULL_HANDLE, scaledTexels.view, VK_IMAGE_LAYOUT_GENERAL}};
      VkWriteDescriptorSet writes[5]{};
      writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
      writes[0].dstSet = set;
      writes[0].dstBinding = 0;
      writes[0].descriptorCount = 1;
      writes[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC;
      writes[0].pBufferInfo = &uniformInfo;
      for (int i = 0; i < 4; ++i) {
        writes[i + 1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i + 1].dstSet = set;
        writes[i + 1].dstBinding = static_cast<std::uint32_t>(i + 1);
        writes[i + 1].descriptorCount = 1;
        writes[i + 1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        writes[i + 1].pImageInfo = &imageInfos[i];
      }
      coregs_hwd_vkUpdateDescriptorSets(device, 5, writes, 0, nullptr);
    }
  }

  // Kernel pool: 9 upsample sets (2 images each) and 2 compose sets (GLM, tables, ring, display, 6 scaled frames and
  // the IN-377 display shadows).
  constexpr std::uint32_t kUpsampleSets = kGpuColourSlots + kGpuDepthSlots;
  // IN-379: plus one tone-carry set per colour slot (4 images and the table each).
  VkDescriptorPoolSize poolSizes[2] = {{VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                                        2 * kUpsampleSets + 2 * (1 + kGpuColourSlots + 2 * kGpuDisplayShadows) +
                                            4 * kGpuColourSlots},
                                       {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2 * 3 + kGpuColourSlots}};
  VkDescriptorPoolCreateInfo poolInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
  poolInfo.maxSets = kUpsampleSets + 2 + kGpuColourSlots;
  poolInfo.poolSizeCount = 2;
  poolInfo.pPoolSizes = poolSizes;
  if (!check(coregs_hwd_vkCreateDescriptorPool(device, &poolInfo, nullptr, &scalePool), "vkCreateDescriptorPool", error)) {
    return false;
  }

  // Upsample kernel (hw_scale_upsample.comp): binding 0 the 1x image, 1 the scaled image.
  VkDescriptorSetLayoutBinding upsampleBindings[2]{};
  upsampleBindings[0] = {0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
  upsampleBindings[1] = {1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
  VkDescriptorSetLayoutCreateInfo upsampleSetInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
  upsampleSetInfo.bindingCount = 2;
  upsampleSetInfo.pBindings = upsampleBindings;
  if (!check(coregs_hwd_vkCreateDescriptorSetLayout(device, &upsampleSetInfo, nullptr, &upsampleSetLayout),
             "vkCreateDescriptorSetLayout", error)) {
    return false;
  }
  // IN-398: twelve push words (the range was five while six were pushed, so the IN-384 `linear` word never reached
  // the kernel) and a second set: the carry slot's images, in the same layout.
  VkPushConstantRange upsamplePush{VK_SHADER_STAGE_COMPUTE_BIT, 0, 12 * sizeof(std::uint32_t)};
  const VkDescriptorSetLayout upsamplePipelineSets[2] = {upsampleSetLayout, upsampleSetLayout};
  VkPipelineLayoutCreateInfo upsampleLayoutInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
  upsampleLayoutInfo.setLayoutCount = 2;
  upsampleLayoutInfo.pSetLayouts = upsamplePipelineSets;
  upsampleLayoutInfo.pushConstantRangeCount = 1;
  upsampleLayoutInfo.pPushConstantRanges = &upsamplePush;
  if (!check(coregs_hwd_vkCreatePipelineLayout(device, &upsampleLayoutInfo, nullptr, &upsampleLayout),
             "vkCreatePipelineLayout", error)) {
    return false;
  }
  VkDescriptorSetLayout upsampleLayouts[kUpsampleSets];
  for (VkDescriptorSetLayout &layout : upsampleLayouts) layout = upsampleSetLayout;
  VkDescriptorSetAllocateInfo upsampleAllocate{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
  upsampleAllocate.descriptorPool = scalePool;
  upsampleAllocate.descriptorSetCount = kUpsampleSets;
  upsampleAllocate.pSetLayouts = upsampleLayouts;
  if (!check(coregs_hwd_vkAllocateDescriptorSets(device, &upsampleAllocate, upsampleSets), "vkAllocateDescriptorSets",
             error)) {
    return false;
  }
  for (std::uint32_t set = 0; set < kUpsampleSets; ++set) {
    const bool depthSet = set >= kGpuColourSlots;
    const std::uint32_t slot = depthSet ? set - kGpuColourSlots : set;
    VkDescriptorImageInfo images[2] = {
        {VK_NULL_HANDLE, depthSet ? depths[slot].view : frames[slot].view, VK_IMAGE_LAYOUT_GENERAL},
        {VK_NULL_HANDLE, depthSet ? scaledDepths[slot].view : scaledFrames[slot].view, VK_IMAGE_LAYOUT_GENERAL}};
    VkWriteDescriptorSet writes[2]{};
    for (std::uint32_t i = 0; i < 2; ++i) {
      writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
      writes[i].dstSet = upsampleSets[set];
      writes[i].dstBinding = i;
      writes[i].descriptorCount = 1;
      writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
      writes[i].pImageInfo = &images[i];
    }
    coregs_hwd_vkUpdateDescriptorSets(device, 2, writes, 0, nullptr);
  }
  upsampleModule = make_module(coregs::shaders::hw_scale_upsample_words, coregs::shaders::hw_scale_upsample_word_count, error);
  if (!upsampleModule) return false;
  {
    VkComputePipelineCreateInfo info{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    info.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    info.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    info.stage.module = upsampleModule;
    info.stage.pName = "main";
    info.layout = upsampleLayout;
    if (!check(coregs_hwd_vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &info, nullptr, &upsamplePipeline),
               "vkCreateComputePipelines", error)) {
      return false;
    }
  }

  // IN-379 tone-carry kernel (hw_scale_tone_carry.comp): 0 saved 1x words, 1 saved scaled image, 2 the slot's 1x
  // image, 3 the slot's scaled image, 4 the table; one set per colour slot.
  {
    VkDescriptorSetLayoutBinding bindings[5]{};
    for (std::uint32_t i = 0; i < 4; ++i) {
      bindings[i] = {i, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    }
    bindings[4] = {4, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    VkDescriptorSetLayoutCreateInfo setInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    setInfo.bindingCount = 5;
    setInfo.pBindings = bindings;
    if (!check(coregs_hwd_vkCreateDescriptorSetLayout(device, &setInfo, nullptr, &toneSetLayout),
               "vkCreateDescriptorSetLayout", error)) {
      return false;
    }
    VkPushConstantRange push{VK_SHADER_STAGE_COMPUTE_BIT, 0, 6 * sizeof(std::uint32_t)};
    VkPipelineLayoutCreateInfo layoutInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    layoutInfo.setLayoutCount = 1;
    layoutInfo.pSetLayouts = &toneSetLayout;
    layoutInfo.pushConstantRangeCount = 1;
    layoutInfo.pPushConstantRanges = &push;
    if (!check(coregs_hwd_vkCreatePipelineLayout(device, &layoutInfo, nullptr, &toneLayout), "vkCreatePipelineLayout",
               error)) {
      return false;
    }
    VkDescriptorSetLayout layouts[kGpuColourSlots];
    for (VkDescriptorSetLayout &layout : layouts) layout = toneSetLayout;
    VkDescriptorSetAllocateInfo allocate{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    allocate.descriptorPool = scalePool;
    allocate.descriptorSetCount = kGpuColourSlots;
    allocate.pSetLayouts = layouts;
    if (!check(coregs_hwd_vkAllocateDescriptorSets(device, &allocate, toneSets), "vkAllocateDescriptorSets", error)) {
      return false;
    }
    for (std::uint32_t slot = 0; slot < kGpuColourSlots; ++slot) {
      VkDescriptorImageInfo images[4] = {{VK_NULL_HANDLE, toneNative.view, VK_IMAGE_LAYOUT_GENERAL},
                                         {VK_NULL_HANDLE, toneScaled.view, VK_IMAGE_LAYOUT_GENERAL},
                                         {VK_NULL_HANDLE, frames[slot].view, VK_IMAGE_LAYOUT_GENERAL},
                                         {VK_NULL_HANDLE, scaledFrames[slot].view, VK_IMAGE_LAYOUT_GENERAL}};
      VkDescriptorBufferInfo table{toneTable.buffer, 0, 768 * sizeof(std::uint32_t)};
      VkWriteDescriptorSet writes[5]{};
      for (std::uint32_t i = 0; i < 5; ++i) {
        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet = toneSets[slot];
        writes[i].dstBinding = i;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType = i < 4 ? VK_DESCRIPTOR_TYPE_STORAGE_IMAGE : VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        if (i < 4) {
          writes[i].pImageInfo = &images[i];
        } else {
          writes[i].pBufferInfo = &table;
        }
      }
      coregs_hwd_vkUpdateDescriptorSets(device, 5, writes, 0, nullptr);
    }
    toneModule = make_module(coregs::shaders::hw_scale_tone_carry_words, coregs::shaders::hw_scale_tone_carry_word_count,
                             error);
    if (!toneModule) return false;
    VkComputePipelineCreateInfo info{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    info.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    info.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    info.stage.module = toneModule;
    info.stage.pName = "main";
    info.layout = toneLayout;
    if (!check(coregs_hwd_vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &info, nullptr, &tonePipeline),
               "vkCreateComputePipelines", error)) {
      return false;
    }
  }

  // Scaled compose kernel (hw_display_compose_scaled.comp): 0 GLM, 1 tables, 2 ring, 4 display, 6 scaled frames.
  VkDescriptorSetLayoutBinding composeBindings[6]{};
  composeBindings[0] = {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
  composeBindings[1] = {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
  composeBindings[2] = {2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
  composeBindings[3] = {4, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
  composeBindings[4] = {6, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, kGpuColourSlots + kGpuDisplayShadows,
                        VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
  composeBindings[5] = {7, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, kGpuDisplayShadows, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
  VkDescriptorSetLayoutCreateInfo composeSetInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
  composeSetInfo.bindingCount = 6;
  composeSetInfo.pBindings = composeBindings;
  if (!check(coregs_hwd_vkCreateDescriptorSetLayout(device, &composeSetInfo, nullptr, &composeScaledSetLayout),
             "vkCreateDescriptorSetLayout", error)) {
    return false;
  }
  VkPushConstantRange composePush{VK_SHADER_STAGE_COMPUTE_BIT, 0, 4 * sizeof(std::uint32_t)};
  VkPipelineLayoutCreateInfo composeLayoutInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
  composeLayoutInfo.setLayoutCount = 1;
  composeLayoutInfo.pSetLayouts = &composeScaledSetLayout;
  composeLayoutInfo.pushConstantRangeCount = 1;
  composeLayoutInfo.pPushConstantRanges = &composePush;
  if (!check(coregs_hwd_vkCreatePipelineLayout(device, &composeLayoutInfo, nullptr, &composeScaledLayout),
             "vkCreatePipelineLayout", error)) {
    return false;
  }
  composeScaledModule = make_module(coregs::shaders::hw_display_compose_scaled_words,
                                    coregs::shaders::hw_display_compose_scaled_word_count, error);
  if (!composeScaledModule) return false;
  {
    VkComputePipelineCreateInfo info{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    info.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    info.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    info.stage.module = composeScaledModule;
    info.stage.pName = "main";
    info.layout = composeScaledLayout;
    if (!check(coregs_hwd_vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &info, nullptr, &composeScaledPipeline),
               "vkCreateComputePipelines", error)) {
      return false;
    }
  }
  if (!make_compose_scaled_set(0, palettes, error)) return false;
  // IN-384: from here on a grown scaled image can rewrite the descriptor sets.
  scaledDescriptorsReady = true;
  note_scaled_bytes();
  return true;
}


// Submits the recorded commands. With `wait`, blocks until the GPU finished them and the rings are free again;
// without, the fence is waited for before the next recording reuses the rings (the GPU works while the CPU
// prepares the next frame).
bool GpuDrawer::Impl::flush_run(std::string &error) {
  (void)error;
  if (!runOpen) return true;
  runOpen = false;
  if (runDraws.empty()) return true;
  full_barrier_with_fragment();  // texture preparation of the run -> its draws
  VkRenderPassBeginInfo pass{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
  pass.renderPass = renderPass;
  pass.framebuffer = framebuffer;
  pass.renderArea = {{0, 0}, {kSurfaceSize, kSurfaceSize}};
  recorded_work();
  coregs_hwd_vkCmdBeginRenderPass(cmd, &pass, VK_SUBPASS_CONTENTS_INLINE);
  ++stats.renderPasses;
  coregs_hwd_vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, interlockTriangles);
  VkViewport viewport{0.0f, 0.0f, static_cast<float>(kSurfaceSize), static_cast<float>(kSurfaceSize), 0.0f, 1.0f};
  coregs_hwd_vkCmdSetViewport(cmd, 0, 1, &viewport);
  // IN-362: the vertex ring is bound once (draws use absolute first vertices); push constants and the scissor are
  // set only when they change. The descriptor set is bound per draw (its dynamic uniform offset is the chunk's).
  const VkDeviceSize ringStart = 0;
  coregs_hwd_vkCmdBindVertexBuffers(cmd, 0, 1, &vertices.buffer, &ringStart);
  const std::uint32_t *lastPush = nullptr;
  VkRect2D lastScissor{{-1, -1}, {0, 0}};
  for (const PendingDraw &draw : runDraws) {
    coregs_hwd_vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0, 1,
                                    &descriptorSets[draw.descriptorSet], 1, &draw.uniformOffset);
    if (!lastPush || std::memcmp(lastPush, draw.push, sizeof(draw.push)) != 0) {
      coregs_hwd_vkCmdPushConstants(cmd, pipelineLayout, kGraphicsPushStages, 0, kPushBytes, draw.push);
      lastPush = draw.push;
    }
    if (std::memcmp(&lastScissor, &draw.scissor, sizeof(VkRect2D)) != 0) {
      coregs_hwd_vkCmdSetScissor(cmd, 0, 1, &draw.scissor);
      lastScissor = draw.scissor;
    }
    const std::uint32_t ringFirst =
        static_cast<std::uint32_t>(draw.vertexOffsetBytes / sizeof(GpuVertex)) + draw.firstVertex;
    recorded_work();
    coregs_hwd_vkCmdDraw(cmd, draw.vertexCount, 1, ringFirst, 0);
    ++stats.draws;
  }
  recorded_work();
  coregs_hwd_vkCmdEndRenderPass(cmd);
  if (scaled() && !scaledPaused) {
    mark(kTimePass);  // the 1x run (12.8: the scaled pass is timed on its own)
    // IN-371: the same draws into the scaled images (other images: no barrier against the 1x pass).
    scaledDraws.clear();
    for (const PendingDraw &draw : runDraws) {
      ScaledDraw scaledDraw;
      scaledDraw.descriptorSet = draw.descriptorSet;
      scaledDraw.uniformOffset = draw.uniformOffset;
      scaledDraw.push = draw.push;
      scaledDraw.scissor = draw.scaledScissor;
      scaledDraw.firstVertex = static_cast<std::uint32_t>(draw.vertexOffsetBytes / sizeof(GpuVertex)) + draw.firstVertex;
      scaledDraw.vertexCount = draw.vertexCount;
      scaledDraw.pipeline = interlockTriangles;
      scaledDraw.textured = true;  // IN-389: the interlock run path batches textured triangles only
      scaledDraws.push_back(scaledDraw);
    }
    record_scaled_draws(false);
    mark(kTimeScaled);
  }
  mark(kTimePass);
  full_barrier_with_fragment();  // the run's draws -> anything after (target reads, the next run's atlas writes)
  ++stats.runs;
  stats.runChunks += runDraws.size();
  runDraws.clear();
  return true;
}

bool GpuDrawer::Impl::submit(bool wait, std::string &error) {
  if (recording && !flush_run(error)) return false;
  if (recording) {
    if (!check(coregs_hw_vkEndCommandBuffer(cmd), "vkEndCommandBuffer", error)) return false;
    VkSubmitInfo submitInfo{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &cmd;
    coregs_hw_vkResetFences(device, 1, &fence);
    if (!check(submit_to_queue(submitInfo, fence), "vkQueueSubmit", error)) return false;
    recording = false;
    inFlight = true;
    ++stats.submissions;
    chunksSinceSubmission = 0;
    serial = stats.submissions;
    if (doubleBuffered) {
      save_frame(currentFrame);
      currentFrame ^= 1u;
      load_frame(currentFrame);
      stagingUsed = 0;  // the new set's rings; begin_recording waits for their last submission
      vertexUsed = 0;
      uniformUsed = 0;
      paletteUsed = 0;
    }
  }
  if (wait) return wait_all(error);
  return true;
}

bool GpuDrawer::Impl::wait_all(std::string &error) {
  if (secondFrameReady) {
    FrameResources &other = frameSets[currentFrame ^ 1u];
    if (other.inFlight) {
      if (!check(coregs_hw_vkWaitForFences(device, 1, &other.fence, VK_TRUE, UINT64_MAX), "vkWaitForFences", error)) {
        return false;
      }
      other.inFlight = false;
      completedSerial = std::max(completedSerial, other.serial);
      collect_timing(currentFrame ^ 1u);
    }
  }
  return wait_idle(error);
}

bool GpuDrawer::Impl::create_second_frame(std::string &error) {
  save_frame(0);
  VkCommandBufferAllocateInfo allocate{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
  allocate.commandPool = pool;
  allocate.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  allocate.commandBufferCount = 1;
  FrameResources &f = frameSets[1];
  if (!check(coregs_hw_vkAllocateCommandBuffers(device, &allocate, &f.cmd), "vkAllocateCommandBuffers", error)) return false;
  VkFenceCreateInfo fenceInfo{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
  if (!check(coregs_hw_vkCreateFence(device, &fenceInfo, nullptr, &f.fence), "vkCreateFence", error)) return false;
  const VkMemoryPropertyFlags hostCoherent = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
  if (!make_buffer(f.staging, kStagingBytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                   hostCoherent, 0, true, error) ||
      !make_buffer(f.vertices, kVertexBytes, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, hostCoherent, 0, true, error) ||
      !make_buffer(f.uniforms, kUniformRingBytes, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, hostCoherent, 0, true, error) ||
      !make_buffer(f.palettes, kPaletteRingBytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, hostCoherent, 0, true, error)) {
    return false;
  }
  constexpr std::uint32_t kSets = kGpuColourSlots * kGpuDepthSlots;
  constexpr std::uint32_t kComputeSets = kGpuColourSlots + kGpuDepthSlots;
  VkDescriptorSetLayout layouts[kSets];
  for (VkDescriptorSetLayout &layout : layouts) layout = setLayout;
  VkDescriptorSetAllocateInfo setAllocate{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
  setAllocate.descriptorPool = descriptorPool;
  setAllocate.descriptorSetCount = kSets;
  setAllocate.pSetLayouts = layouts;
  if (!check(coregs_hwd_vkAllocateDescriptorSets(device, &setAllocate, f.descriptorSets), "vkAllocateDescriptorSets", error)) {
    return false;
  }
  VkDescriptorSetLayout computeLayouts[kComputeSets];
  for (VkDescriptorSetLayout &layout : computeLayouts) layout = computeSetLayout;
  VkDescriptorSetAllocateInfo computeAllocate{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
  computeAllocate.descriptorPool = computePool;
  computeAllocate.descriptorSetCount = kComputeSets;
  computeAllocate.pSetLayouts = computeLayouts;
  if (!check(coregs_hwd_vkAllocateDescriptorSets(device, &computeAllocate, f.computeSets), "vkAllocateDescriptorSets",
             error)) {
    return false;
  }
  // Same images and shared buffers as set 0 (copied), own rings (written).
  const FrameResources &first = frameSets[0];
  VkDescriptorBufferInfo uniformInfo{f.uniforms.buffer, 0, kUniformBytes};
  for (std::uint32_t i = 0; i < kSets; ++i) {
    VkCopyDescriptorSet copies[4]{};
    for (std::uint32_t b = 0; b < 4; ++b) {
      copies[b].sType = VK_STRUCTURE_TYPE_COPY_DESCRIPTOR_SET;
      copies[b].srcSet = first.descriptorSets[i];
      copies[b].srcBinding = b + 1;
      copies[b].dstSet = f.descriptorSets[i];
      copies[b].dstBinding = b + 1;
      copies[b].descriptorCount = 1;
    }
    VkWriteDescriptorSet write{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    write.dstSet = f.descriptorSets[i];
    write.dstBinding = 0;
    write.descriptorCount = 1;
    write.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC;
    write.pBufferInfo = &uniformInfo;
    coregs_hwd_vkUpdateDescriptorSets(device, 1, &write, 4, copies);
  }
  VkDescriptorBufferInfo paletteInfo{f.palettes.buffer, 0, kPaletteRingBytes};
  for (std::uint32_t i = 0; i < kComputeSets; ++i) {
    VkCopyDescriptorSet copies[5]{};
    const std::uint32_t copied[5] = {0, 1, 3, 4, 5};
    for (std::uint32_t c = 0; c < 5; ++c) {
      copies[c].sType = VK_STRUCTURE_TYPE_COPY_DESCRIPTOR_SET;
      copies[c].srcSet = first.computeSets[i];
      copies[c].srcBinding = copied[c];
      copies[c].dstSet = f.computeSets[i];
      copies[c].dstBinding = copied[c];
      copies[c].descriptorCount = 1;
    }
    VkWriteDescriptorSet write{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    write.dstSet = f.computeSets[i];
    write.dstBinding = 2;
    write.descriptorCount = 1;
    write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    write.pBufferInfo = &paletteInfo;
    coregs_hwd_vkUpdateDescriptorSets(device, 1, &write, 5, copies);
  }
  for (Resource &buffer : downloads) {
    if (!make_buffer(buffer, kGlmBytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT, hostCoherent, VK_MEMORY_PROPERTY_HOST_CACHED_BIT,
                     true, error)) {
      return false;
    }
  }
  {
    VkPhysicalDeviceProperties properties{};
    coregs_hw_vkGetPhysicalDeviceProperties(physical, &properties);
    timestampPeriodNs = properties.limits.timestampPeriod > 0.0f ? properties.limits.timestampPeriod : 1.0;
    VkQueryPoolCreateInfo queryInfo{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
    queryInfo.queryType = VK_QUERY_TYPE_TIMESTAMP;
    queryInfo.queryCount = kQueries;
    for (QueryState &q : queries) {
      if (!check(coregs_hwd_vkCreateQueryPool(device, &queryInfo, nullptr, &q.pool), "vkCreateQueryPool", error)) {
        return false;
      }
      q.categories.assign(kQueries, kTimeStart);
    }
  }
  if (!make_display_set(1, f.computeSets[0], error)) return false;
  if (scaled()) {
    // IN-371: frame set 1's scaled graphics sets (images copied from set 0's, own uniform ring) and compose set.
    VkDescriptorSetAllocateInfo scaledAllocate{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    scaledAllocate.descriptorPool = descriptorPool;
    scaledAllocate.descriptorSetCount = kSets;
    scaledAllocate.pSetLayouts = layouts;
    if (!check(coregs_hwd_vkAllocateDescriptorSets(device, &scaledAllocate, f.scaledDescriptorSets),
               "vkAllocateDescriptorSets", error)) {
      return false;
    }
    for (std::uint32_t i = 0; i < kSets; ++i) {
      VkCopyDescriptorSet copies[4]{};
      for (std::uint32_t b = 0; b < 4; ++b) {
        copies[b].sType = VK_STRUCTURE_TYPE_COPY_DESCRIPTOR_SET;
        copies[b].srcSet = first.scaledDescriptorSets[i];
        copies[b].srcBinding = b + 1;
        copies[b].dstSet = f.scaledDescriptorSets[i];
        copies[b].dstBinding = b + 1;
        copies[b].descriptorCount = 1;
      }
      VkWriteDescriptorSet write{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
      write.dstSet = f.scaledDescriptorSets[i];
      write.dstBinding = 0;
      write.descriptorCount = 1;
      write.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC;
      write.pBufferInfo = &uniformInfo;
      coregs_hwd_vkUpdateDescriptorSets(device, 1, &write, 4, copies);
    }
    if (!make_compose_scaled_set(1, f.palettes, error)) return false;
  }
  secondFrameReady = true;
  doubleBuffered = true;
  load_frame(0);
  return true;
}

bool GpuDrawer::Impl::wait_idle(std::string &error) {
  if (inFlight) {
    if (!check(coregs_hw_vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX), "vkWaitForFences", error)) return false;
    inFlight = false;
    completedSerial = std::max(completedSerial, serial);
    collect_timing(currentFrame);
  }
  stagingUsed = 0;
  vertexUsed = 0;
  uniformUsed = 0;
  paletteUsed = 0;
  return true;
}

bool GpuDrawer::Impl::begin_recording(std::string &error) {
  if (recording) return true;
  if (!wait_idle(error)) return false;  // the rings may still be read by the previous submission
  VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
  begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  if (!check(coregs_hw_vkBeginCommandBuffer(cmd, &begin), "vkBeginCommandBuffer", error)) return false;
  recording = true;
  recordingSet = currentFrame;
  barrierIsLast = false;
  ++workGeneration;
  boundComputePipeline = VK_NULL_HANDLE;  // a new command buffer has no bindings
  boundComputeSet = VK_NULL_HANDLE;
  if (gpuTiming && queries[recordingSet].pool) {
    coregs_hwd_vkCmdResetQueryPool(cmd, queries[recordingSet].pool, 0, kQueries);
    queries[recordingSet].count = 0;
    mark(kTimeStart);
  }
  return true;
}

bool GpuDrawer::Flush(bool wait, std::string &error) { return impl_->submit(wait, error); }

void GpuDrawer::SetRuns(bool enabled) { impl_->runsEnabled = enabled; }
void GpuDrawer::SetDecodeReuse(bool enabled) { impl_->decodeReuse = enabled; }
void GpuDrawer::SetClutDefer(bool enabled) { impl_->clutDefer = enabled; }

void GpuDrawer::SetChunksPerSubmission(std::uint32_t chunks) { impl_->chunksPerSubmission = chunks; }

void GpuDrawer::SetGpuTiming(bool enabled) {
  Impl &s = *impl_;
  std::string error;
  s.submit(true, error);  // timing starts at the next recording
  s.gpuTiming = enabled;
}

void GpuDrawer::SetDoubleBuffering(bool enabled) {
  Impl &s = *impl_;
  if (!s.secondFrameReady) return;
  std::string error;
  s.submit(true, error);  // both sets idle; the current set stays current
  s.doubleBuffered = enabled;
}

bool GpuDrawer::Execute(GpuChunk &chunk, std::string &error) {
  Impl &s = *impl_;
  if (!s.ready) {
    error = "GpuDrawer::Execute before Init";
    return false;
  }
  const bool anyTransfer = chunk.uploadFrame || chunk.uploadDepth || chunk.readbackFrame || chunk.readbackDepth ||
                           (chunk.hasTexture && chunk.glmTexture);
  if (chunk.draws.empty() && !anyTransfer) return true;
  if (chunk.colourSlot >= kGpuColourSlots || chunk.depthSlot >= kGpuDepthSlots) {
    error = "chunk names a target slot that does not exist";
    return false;
  }
  const VkImage frameImage = s.frames[chunk.colourSlot].image;
  const VkImage depthImage = s.depths[chunk.depthSlot].image;
  if (chunk.VertexCount() > kGpuMaxVertices) {
    error = "chunk exceeds the vertex buffer (the caller must split it)";
    return false;
  }
  const std::int32_t width = chunk.rectX1 - chunk.rectX0 + 1;
  const std::int32_t height = chunk.rectY1 - chunk.rectY0 + 1;
  if (width <= 0 || height <= 0 || chunk.rectX0 < 0 || chunk.rectY0 < 0 || chunk.rectX1 >= kGpuSurface ||
      chunk.rectY1 >= kGpuSurface) {
    error = "chunk rectangle is outside the 2048 x 2048 surface";
    return false;
  }
  const std::size_t words = static_cast<std::size_t>(width) * static_cast<std::size_t>(height);
  if (((chunk.uploadFrame || chunk.readbackFrame) && chunk.frameWords.size() != words) ||
      ((chunk.uploadDepth || chunk.readbackDepth) && chunk.depthWords.size() != words)) {
    error = "frame or depth word count does not match the rectangle";
    return false;
  }
  const bool textureFromGlm = chunk.hasTexture && chunk.glmTexture;
  const bool textureFromSlot = chunk.hasTexture && !textureFromGlm && chunk.textureSlot >= 0;
  if (textureFromSlot && chunk.textureSlot >= static_cast<std::int32_t>(kGpuColourSlots)) {
    error = "texture source slot does not exist";
    return false;
  }
  if (chunk.hasTexture && (chunk.texWidth == 0 || chunk.texHeight == 0 || chunk.texWidth > kTextureMax ||
                           chunk.texHeight > kTextureMax ||
                           (!textureFromSlot && !textureFromGlm &&
                            chunk.TexelCount() != static_cast<std::size_t>(chunk.texWidth) * chunk.texHeight))) {
    error = "texture size is outside the texel image";
    return false;
  }

  const bool drawingChunk = !chunk.draws.empty();
  bool allTriangles = s.interlock;
  for (const GpuDraw &draw : chunk.draws) allTriangles = allTriangles && draw.topology == GpuTopology::Triangles;
  const bool runnable = s.runsEnabled && allTriangles && drawingChunk && !chunk.uploadFrame && !chunk.uploadDepth &&
                        !chunk.readbackFrame && !chunk.readbackDepth && !textureFromSlot;
  // IN-384: the scaled images of this chunk's slot pair cover its target's width and the rows it reaches; a target
  // that grows past its image grows it here (copying the old content) before anything of this chunk is recorded.
  if (s.scaled() && chunk.targetWidth != 0) {
    const std::uint32_t targetWidth = chunk.targetWidth;
    const std::uint32_t targetHeight = static_cast<std::uint32_t>(chunk.rectY1 + 1);
    if (chunk.colourSlot != kGpuScratchColourSlot &&
        !s.ensure_scaled_image(s.scaledFrames[chunk.colourSlot], s.scaledFrameExtent[chunk.colourSlot], targetWidth,
                               targetHeight, error)) {
      return false;
    }
    if (!s.ensure_scaled_image(s.scaledDepths[chunk.depthSlot], s.scaledDepthExtent[chunk.depthSlot], targetWidth,
                               targetHeight, error)) {
      return false;
    }
  }
  // IN-374/IN-384: a texture copied from a resident slot copies the texels the batch can sample (IN-380 footprint),
  // so the slot's scaled image has to cover that footprint; growing it here (never a clipped copy) keeps the scaled
  // pass sampling the slot's own image.
  const std::uint32_t copyWidth =
      chunk.texCopyWidth != 0 ? std::min(chunk.texCopyWidth, chunk.texWidth) : chunk.texWidth;
  const std::uint32_t copyHeight =
      chunk.texCopyHeight != 0 ? std::min(chunk.texCopyHeight, chunk.texHeight) : chunk.texHeight;
  bool scaledTextureCopy = s.scaled() && textureFromSlot &&
                           chunk.textureSlot != static_cast<std::int32_t>(kGpuScratchColourSlot) &&
                           copyWidth <= static_cast<std::uint32_t>(kGpuScaledExtent) &&
                           copyHeight <= static_cast<std::uint32_t>(kGpuScaledExtent);
  if (scaledTextureCopy) {
    scaledTextureCopy =
        s.ensure_scaled_image(s.scaledFrames[chunk.textureSlot], s.scaledFrameExtent[chunk.textureSlot], copyWidth,
                              copyHeight, error) &&
        s.ensure_scaled_image(s.scaledTexels, s.scaledTexelExtent, copyWidth, copyHeight, error);
    if (!scaledTextureCopy) return false;
  }
  if (!runnable && s.recording && !s.flush_run(error)) return false;

  // Diagnostics (COREGS_HW_DRAW_TRACE=<file>, COREGS_HW_DRAW_TRACE_RECT=x0,y0,x1,y1 in native pixels): one line per chunk
  // that touches the probe rectangle, with the texture path and the filter state (IN-387: which layer carries a
  // per-native-pixel pattern, and why the forced-linear switch does not reach it).
  if (const char *tracePath = s.traceArmed ? std::getenv("COREGS_HW_DRAW_TRACE") : nullptr) {
    static const char *rectText = std::getenv("COREGS_HW_DRAW_TRACE_RECT");
    static const int limit = [] {
      const char *text = std::getenv("COREGS_HW_DRAW_TRACE_LIMIT");
      return text ? std::atoi(text) : 400;
    }();
    static int lines = 0;
    std::int32_t px0 = 0, py0 = 0, px1 = 639, py1 = 447;
    if (rectText) std::sscanf(rectText, "%d,%d,%d,%d", &px0, &py0, &px1, &py1);
    const bool touches = chunk.rectX1 >= px0 && chunk.rectX0 <= px1 && chunk.rectY1 >= py0 && chunk.rectY0 <= py1;
    if (touches && lines < limit) {
      ++lines;
      std::FILE *file = std::fopen(tracePath, "ab");
      if (file) {
        std::fprintf(file,
                     "chunk target=%u depth=%u rect=%d,%d,%d,%d draws=%zu tex=%d slot=%d glm=%d scaled=%d "
                     "tw=%u th=%u cw=%u ch=%u texlin=%u texfst=%u texon=%u psm=%u tpsm=%u tlog=%u,%u "
                     "zte=%u ztst=%u zmsk=%u zfmt=%u fbmsk=%08x ate=%u atst=%u afail=%u tbp=%u\n",
                     chunk.colourSlot, chunk.depthSlot, chunk.rectX0, chunk.rectY0, chunk.rectX1, chunk.rectY1,
                     chunk.draws.size(), chunk.hasTexture ? 1 : 0, chunk.textureSlot, chunk.glmTexture ? 1 : 0,
                     scaledTextureCopy ? 1 : 0, chunk.texWidth, chunk.texHeight, copyWidth, copyHeight,
                     chunk.state.fields[gpufield::TexLinear], chunk.state.fields[gpufield::TexFst],
                     chunk.state.fields[gpufield::TexOn], chunk.state.fields[gpufield::Format], chunk.glmPsm,
                     chunk.state.fields[gpufield::TexLog2W], chunk.state.fields[gpufield::TexLog2H],
                     chunk.state.fields[gpufield::Ztest], chunk.state.fields[gpufield::Ztst],
                     chunk.state.fields[gpufield::Zmsk], chunk.state.fields[gpufield::Zformat],
                     chunk.state.fields[gpufield::Fbmsk], chunk.state.fields[gpufield::Ate],
                     chunk.state.fields[gpufield::Atst], chunk.state.fields[gpufield::Afail], chunk.glmBp);
        // COREGS_HW_DRAW_TRACE_VERTS=N: the vertices (12.4 X/Y, Z, colour) of every traced chunk with at least N draws.
        static const char *vertsText = std::getenv("COREGS_HW_DRAW_TRACE_VERTS");
        if (vertsText && chunk.draws.size() >= static_cast<std::size_t>(std::atoi(vertsText))) {
          const GpuVertex *vertex = chunk.VertexData();
          for (std::size_t i = 0; i < chunk.VertexCount(); ++i) {
            std::fprintf(file, "v %zu x=%u y=%u z=%u rgba=%u,%u,%u,%u\n", i, vertex[i].x, vertex[i].y, vertex[i].z,
                         vertex[i].rgba[0], vertex[i].rgba[1], vertex[i].rgba[2], vertex[i].rgba[3]);
          }
        }
        std::fclose(file);
      }
    }
  }

  // Ring space this chunk needs (IN-342): the staging bytes of its uploads, its vertices and one uniform slot.
  const VkDeviceSize frameBytes = static_cast<VkDeviceSize>(words) * 4;
  const VkDeviceSize textureBytes = (chunk.hasTexture && !textureFromSlot && !textureFromGlm)
                                         ? static_cast<VkDeviceSize>(chunk.TexelCount()) * 4
                                         : 0;
  const VkDeviceSize stagingNeeded =
      (chunk.uploadFrame ? frameBytes : 0) + (chunk.uploadDepth ? frameBytes : 0) + textureBytes;
  const VkDeviceSize vertexNeeded = chunk.VertexCount();
  const bool drawing = !chunk.draws.empty();
  const VkDeviceSize paletteNeeded =
      (textureFromGlm && chunk.glmIndexed && !chunk.glmGpuClut) ? 256 * sizeof(std::uint32_t) : 0;
  if (s.stagingUsed + stagingNeeded > kStagingBytes || s.vertexUsed + vertexNeeded > kGpuMaxVertices ||
      s.uniformUsed + s.uniformStride > kUniformRingBytes || s.paletteUsed + paletteNeeded > kPaletteRingBytes) {
    if (!s.submit(false, error)) return false;  // the next recording waits for the rings it reuses
  }
  if (!s.begin_recording(error)) return false;
  std::uint32_t originX = 0;
  std::uint32_t originY = 0;
  // IN-369: the decode parameters (push constants of hw_glm_decode without the palette base and the atlas origin).
  // A CPU palette (indexed mode 1) lives in the palette ring and is never reused.
  std::uint32_t indexedMode = !chunk.glmIndexed ? 0u : chunk.glmGpuClut ? 2u : 1u;
  // IN-370: mode 3 reads the palette from the GLM with the last pending CLUT load's addressing; otherwise a GPU
  // CLUT decode needs the pending loads recorded first.
  std::uint32_t paletteLoad[6] = {};  // CBP, CSM, entries (1: 256), CBW, COU, COV of that load
  if (textureFromGlm && indexedMode == 2u) {
    const std::uint32_t textureEightBit = (chunk.glmPsm & 7u) == 3u ? 1u : 0u;
    if (const Impl::ClutLoadRecord *load = s.direct_palette_load(chunk.glmCpsm, chunk.glmCsa, textureEightBit)) {
      indexedMode = 3u;
      const std::uint32_t fields[6] = {load->cbp, load->csm, load->eightBit, load->cbw, load->cou, load->cov};
      std::memcpy(paletteLoad, fields, sizeof(fields));
      ++s.stats.decodesGlmPalette;
    } else if (!s.materialise_clut(error)) {
      return false;
    }
  }
  const std::array<std::uint32_t, 12> decodeKey = {chunk.glmBp,   chunk.glmBw,   chunk.glmPsm, chunk.texWidth,
                                                   chunk.texHeight, indexedMode, chunk.glmCpsm, chunk.glmCsa,
                                                   chunk.glmTa0,  chunk.glmTa1,  chunk.glmAem, 0u};
  bool decodeReused = false;
  if (runnable) {
    s.open_run();
    if (textureFromGlm && s.decodeReuse && indexedMode != 1u) {
      if (const Impl::RunDecode *earlier = s.find_run_decode(decodeKey)) {
        originX = earlier->x;
        originY = earlier->y;
        decodeReused = true;
      }
    }
    if (chunk.hasTexture && !decodeReused && !s.atlas_alloc(chunk.texWidth, chunk.texHeight, originX, originY)) {
      if (!s.flush_run(error)) return false;  // atlas full: close the run, start a new one
      s.open_run();
      if (!s.atlas_alloc(chunk.texWidth, chunk.texHeight, originX, originY)) {
        error = "texture larger than the texel atlas";
        return false;
      }
    }
  }

  auto *stagingBase = static_cast<std::uint8_t *>(s.staging.mapped);
  const VkDeviceSize frameOffset = s.stagingUsed;
  if (chunk.uploadFrame) {
    std::memcpy(stagingBase + s.stagingUsed, chunk.frameWords.data(), frameBytes);
    s.stagingUsed += frameBytes;
  }
  const VkDeviceSize depthOffset = s.stagingUsed;
  if (chunk.uploadDepth) {
    std::memcpy(stagingBase + s.stagingUsed, chunk.depthWords.data(), frameBytes);
    s.stagingUsed += frameBytes;
  }
  const VkDeviceSize textureOffset = s.stagingUsed;
  const std::uint32_t paletteBase = static_cast<std::uint32_t>(s.paletteUsed / sizeof(std::uint32_t));
  if (paletteNeeded != 0) {
    if (chunk.palette.size() != 256) {
      error = "a GLM-decoded indexed texture needs a 256-entry palette";
      return false;
    }
    std::memcpy(static_cast<std::uint8_t *>(s.palettes.mapped) + s.paletteUsed, chunk.palette.data(), paletteNeeded);
    s.paletteUsed += paletteNeeded;
  }
  if (chunk.hasTexture && !textureFromSlot && !textureFromGlm) {
    std::memcpy(stagingBase + s.stagingUsed, chunk.TexelData(), textureBytes);
    s.stagingUsed += textureBytes;
  }
  const VkDeviceSize vertexOffsetBytes = s.vertexUsed * sizeof(GpuVertex);
  if (chunk.VertexCount() != 0) {
    std::memcpy(static_cast<std::uint8_t *>(s.vertices.mapped) + vertexOffsetBytes, chunk.VertexData(),
                chunk.VertexCount() * sizeof(GpuVertex));
    s.vertexUsed += vertexNeeded;
  }
  const std::uint32_t uniformOffset = static_cast<std::uint32_t>(s.uniformUsed);
  if (drawing) {
    std::uint32_t uniform[64] = {};
    static_assert(sizeof(chunk.state.fields) <= sizeof(uniform), "uniform copy");
    std::memcpy(uniform, chunk.state.fields, sizeof(chunk.state.fields));
    uniform[gpufield::TexOriginX] = originX;
    uniform[gpufield::TexOriginY] = originY;
    std::memcpy(static_cast<std::uint8_t *>(s.uniforms.mapped) + uniformOffset, uniform, sizeof(uniform));
    s.uniformUsed += s.uniformStride;
  }

  const std::uint32_t zFormat = chunk.state.fields[gpufield::Zformat];
  const std::uint32_t zMax = zFormat == 1 ? 0xFFFFFFu : zFormat == 2 ? 0xFFFFu : 0xFFFFFFFFu;
  // PushConstants of hw_vertex.vert: viewport (vec2), scale, scale_offset (1.0 at 1x), offset X and
  // Y (0: the batch vertices already carry XYOFFSET), z_max.
  const float viewportSize[2] = {static_cast<float>(kSurfaceSize), static_cast<float>(kSurfaceSize)};
  std::uint32_t pushWords[8];
  std::memcpy(pushWords, viewportSize, sizeof(viewportSize));
  pushWords[2] = FloatToBits(1.0f);
  pushWords[3] = FloatToBits(1.0f);
  pushWords[4] = 0;
  pushWords[5] = 0;
  pushWords[6] = zMax;
  pushWords[7] = 1;  // tex_scale: the texel atlas holds 1x texels

  if (runnable) {
    // IN-352: texture preparation now, into the chunk's atlas region; the draw is recorded when the run closes.
    if (decodeReused) {
      ++s.stats.decodesReused;
    } else if (textureFromGlm) {
      s.bind_compute(s.decodePipeline, s.computeSets[0]);
      const std::uint32_t push[20] = {chunk.glmBp,    chunk.glmBw,    chunk.glmPsm,   chunk.texWidth, chunk.texHeight,
                                      indexedMode,    paletteBase,    chunk.glmCpsm,  chunk.glmCsa,   chunk.glmTa0,
                                      chunk.glmTa1,   chunk.glmAem,   originX,        originY,        paletteLoad[0],
                                      paletteLoad[1], paletteLoad[2], paletteLoad[3], paletteLoad[4], paletteLoad[5]};
      coregs_hwd_vkCmdPushConstants(s.cmd, s.computeLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), push);
      s.recorded_decode();
      coregs_hwd_vkCmdDispatch(s.cmd, (chunk.texWidth + 7) / 8, (chunk.texHeight + 7) / 8, 1);
      if (indexedMode != 1u) s.runDecodes.push_back({decodeKey, originX, originY, s.workGeneration});
    } else if (chunk.hasTexture) {
      VkBufferImageCopy texCopy{};
      texCopy.bufferOffset = textureOffset;
      texCopy.bufferRowLength = chunk.texWidth;
      texCopy.bufferImageHeight = chunk.texHeight;
      texCopy.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
      texCopy.imageOffset = {static_cast<std::int32_t>(originX), static_cast<std::int32_t>(originY), 0};
      texCopy.imageExtent = {chunk.texWidth, chunk.texHeight, 1};
      s.recorded_work();
      coregs_hwd_vkCmdCopyBufferToImage(s.cmd, s.staging.buffer, s.texels.image, VK_IMAGE_LAYOUT_GENERAL, 1, &texCopy);
    }
    if (chunk.hasTexture && !decodeReused) s.mark(textureFromGlm ? Impl::kTimeDecode : Impl::kTimeUpload);
    std::int32_t x0 = kGpuSurface, y0 = kGpuSurface, x1 = -1, y1 = -1;
    ScaledBounds scaledBounds;
    std::uint32_t firstVertex = UINT32_MAX, endVertex = 0;
    for (const GpuDraw &draw : chunk.draws) {
      if (draw.count == 0 || draw.first + draw.count > chunk.VertexCount()) continue;
      x0 = std::min(x0, std::max<std::int32_t>(0, draw.x0));
      y0 = std::min(y0, std::max<std::int32_t>(0, draw.y0));
      x1 = std::max(x1, std::min<std::int32_t>(kGpuSurface - 1, draw.x1));
      y1 = std::max(y1, std::min<std::int32_t>(kGpuSurface - 1, draw.y1));
      scaledBounds.add(draw);
      firstVertex = std::min(firstVertex, draw.first);
      endVertex = std::max(endVertex, draw.first + draw.count);
    }
    if (x1 >= x0 && y1 >= y0 && endVertex > firstVertex) {
      Impl::PendingDraw pending;
      pending.scaledScissor = scaledBounds.rect();
      pending.descriptorSet = chunk.colourSlot * kGpuDepthSlots + chunk.depthSlot;
      pending.uniformOffset = uniformOffset;
      std::memcpy(pending.push, pushWords, sizeof(pushWords));
      pending.vertexOffsetBytes = vertexOffsetBytes;
      pending.scissor = {{x0, y0}, {static_cast<std::uint32_t>(x1 - x0 + 1), static_cast<std::uint32_t>(y1 - y0 + 1)}};
      pending.firstVertex = firstVertex;
      pending.vertexCount = endVertex - firstVertex;
      s.runDraws.push_back(pending);
    }
    s.stats.executions++;
    s.stats.vertices += chunk.VertexCount();
    s.stats.uploadBytes += stagingNeeded + chunk.VertexCount() * sizeof(GpuVertex);
    return s.note_chunk(error);
  }

  // Earlier chunks of this or a previous submission wrote the frame, depth and texel images (copies and shader
  // writes): make those writes visible and finish their reads before this chunk copies or draws.
  VkMemoryBarrier previousWork{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
  previousWork.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
  previousWork.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT |
                               VK_ACCESS_TRANSFER_WRITE_BIT;
  coregs_hw_vkCmdPipelineBarrier(s.cmd, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
                              VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1,
                              &previousWork, 0, nullptr, 0, nullptr);
  s.stats.barriers++;

  VkBufferImageCopy frameCopy{};
  frameCopy.bufferOffset = frameOffset;
  frameCopy.bufferRowLength = static_cast<std::uint32_t>(width);
  frameCopy.bufferImageHeight = static_cast<std::uint32_t>(height);
  frameCopy.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
  frameCopy.imageOffset = {chunk.rectX0, chunk.rectY0, 0};
  frameCopy.imageExtent = {static_cast<std::uint32_t>(width), static_cast<std::uint32_t>(height), 1};
  VkBufferImageCopy depthCopy = frameCopy;
  depthCopy.bufferOffset = depthOffset;
  const bool anyUpload = chunk.uploadFrame || chunk.uploadDepth || chunk.hasTexture;
  if (chunk.uploadFrame) {
    s.recorded_work();
    coregs_hwd_vkCmdCopyBufferToImage(s.cmd, s.staging.buffer, frameImage, VK_IMAGE_LAYOUT_GENERAL, 1, &frameCopy);
  }
  if (chunk.uploadDepth) {
    s.recorded_work();
    coregs_hwd_vkCmdCopyBufferToImage(s.cmd, s.staging.buffer, depthImage, VK_IMAGE_LAYOUT_GENERAL, 1, &depthCopy);
  }
  if (textureFromGlm) {
    // IN-347: decode the texels from the GLM with the texture's addressing (the caller made the GLM current).
    s.full_barrier();
    s.bind_compute(s.decodePipeline, s.computeSets[0]);
    const std::uint32_t push[20] = {chunk.glmBp,    chunk.glmBw,    chunk.glmPsm,   chunk.texWidth, chunk.texHeight,
                                    indexedMode,    paletteBase,    chunk.glmCpsm,  chunk.glmCsa,   chunk.glmTa0,
                                    chunk.glmTa1,   chunk.glmAem,   0u,             0u,             paletteLoad[0],
                                    paletteLoad[1], paletteLoad[2], paletteLoad[3], paletteLoad[4], paletteLoad[5]};
    coregs_hwd_vkCmdPushConstants(s.cmd, s.computeLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), push);
    s.recorded_work();
    coregs_hwd_vkCmdDispatch(s.cmd, (chunk.texWidth + 7) / 8, (chunk.texHeight + 7) / 8, 1);
    s.full_barrier();
  } else if (textureFromSlot) {
    // IN-346: the texture is (part of) a resident colour target: copy its words on the GPU.
    VkImageCopy region{};
    region.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    region.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    region.extent = {chunk.texWidth, chunk.texHeight, 1};
    s.recorded_work();
    coregs_hwd_vkCmdCopyImage(s.cmd, s.frames[chunk.textureSlot].image, VK_IMAGE_LAYOUT_GENERAL, s.texels.image,
                           VK_IMAGE_LAYOUT_GENERAL, 1, &region);
    if (scaledTextureCopy) {
      // IN-374 (06.8 "sampling a target directly: the source inherits the target's scale"): the same region of the
      // slot's scaled image, for the scaled pass of this chunk, over the sampled footprint (IN-384).
      VkImageCopy scaledRegion = region;
      scaledRegion.extent = {copyWidth * s.scale, copyHeight * s.scale, 1};
      s.recorded_work();
      coregs_hwd_vkCmdCopyImage(s.cmd, s.scaledFrames[chunk.textureSlot].image, VK_IMAGE_LAYOUT_GENERAL,
                             s.scaledTexels.image, VK_IMAGE_LAYOUT_GENERAL, 1, &scaledRegion);
      ++s.stats.scaledTextureCopies;
    }
  } else if (chunk.hasTexture) {
    VkBufferImageCopy texCopy{};
    texCopy.bufferOffset = textureOffset;
    texCopy.bufferRowLength = chunk.texWidth;
    texCopy.bufferImageHeight = chunk.texHeight;
    texCopy.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    texCopy.imageExtent = {chunk.texWidth, chunk.texHeight, 1};
    s.recorded_work();
    coregs_hwd_vkCmdCopyBufferToImage(s.cmd, s.staging.buffer, s.texels.image, VK_IMAGE_LAYOUT_GENERAL, 1, &texCopy);
  }
  if (anyUpload) {
    VkMemoryBarrier copyBarrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    copyBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    copyBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT;
    coregs_hw_vkCmdPipelineBarrier(s.cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                                VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1,
                                &copyBarrier, 0, nullptr, 0, nullptr);
    s.stats.barriers++;
  }
  // IN-371: a load of 1x words into a resident slot refreshes its scaled image (the scratch pair has none in use).
  if (chunk.uploadFrame && chunk.colourSlot != kGpuScratchColourSlot) {
    s.refresh_scaled(false, chunk.colourSlot, chunk.rectX0, chunk.rectY0, chunk.rectX1, chunk.rectY1);
  }
  if (chunk.uploadDepth && chunk.depthSlot != kGpuScratchDepthSlot) {
    s.refresh_scaled(true, chunk.depthSlot, chunk.rectX0, chunk.rectY0, chunk.rectX1, chunk.rectY1);
  }

  if (drawing) {
    VkRenderPassBeginInfo pass{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
    pass.renderPass = s.renderPass;
    pass.framebuffer = s.framebuffer;
    pass.renderArea = {{0, 0}, {kSurfaceSize, kSurfaceSize}};
    s.recorded_work();
    coregs_hwd_vkCmdBeginRenderPass(s.cmd, &pass, VK_SUBPASS_CONTENTS_INLINE);
    ++s.stats.renderPasses;
    coregs_hwd_vkCmdBindDescriptorSets(s.cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, s.pipelineLayout, 0, 1,
                                    &s.descriptorSets[chunk.colourSlot * kGpuDepthSlots + chunk.depthSlot], 1,
                                    &uniformOffset);
    coregs_hwd_vkCmdPushConstants(s.cmd, s.pipelineLayout, kGraphicsPushStages, 0, kPushBytes, pushWords);
    VkViewport viewport{0.0f, 0.0f, static_cast<float>(kSurfaceSize), static_cast<float>(kSurfaceSize), 0.0f, 1.0f};
    coregs_hwd_vkCmdSetViewport(s.cmd, 0, 1, &viewport);
    VkDeviceSize vertexBufferOffset = vertexOffsetBytes;
    coregs_hwd_vkCmdBindVertexBuffers(s.cmd, 0, 1, &s.vertices.buffer, &vertexBufferOffset);

    bool allTriangles = s.interlock;
    for (const GpuDraw &draw : chunk.draws) allTriangles = allTriangles && draw.topology == GpuTopology::Triangles;
    if (allTriangles) {
      // IN-345: one draw for the whole chunk. Pixel interlock orders overlapping fragments in primitive order,
      // as the per-primitive barriers did. A triangle never covers a pixel outside its own coverage rectangle,
      // so the union of the per-primitive scissors clips exactly what they clipped.
      std::int32_t x0 = kGpuSurface, y0 = kGpuSurface, x1 = -1, y1 = -1;
      ScaledBounds scaledBounds;
      std::uint32_t firstVertex = UINT32_MAX, endVertex = 0;
      for (const GpuDraw &draw : chunk.draws) {
        if (draw.count == 0 || draw.first + draw.count > chunk.VertexCount()) continue;
        x0 = std::min(x0, std::max<std::int32_t>(0, draw.x0));
        y0 = std::min(y0, std::max<std::int32_t>(0, draw.y0));
        x1 = std::max(x1, std::min<std::int32_t>(kGpuSurface - 1, draw.x1));
        y1 = std::max(y1, std::min<std::int32_t>(kGpuSurface - 1, draw.y1));
        scaledBounds.add(draw);
        firstVertex = std::min(firstVertex, draw.first);
        endVertex = std::max(endVertex, draw.first + draw.count);
      }
      if (x1 >= x0 && y1 >= y0 && endVertex > firstVertex) {
        VkRect2D scissor{{x0, y0}, {static_cast<std::uint32_t>(x1 - x0 + 1), static_cast<std::uint32_t>(y1 - y0 + 1)}};
        coregs_hwd_vkCmdSetScissor(s.cmd, 0, 1, &scissor);
        coregs_hwd_vkCmdBindPipeline(s.cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, s.interlockTriangles);
        s.recorded_work();
        coregs_hwd_vkCmdDraw(s.cmd, endVertex - firstVertex, 1, firstVertex, 0);
        s.stats.draws++;
        if (s.scaled()) {
          Impl::ScaledDraw scaledDraw;
          scaledDraw.descriptorSet = chunk.colourSlot * kGpuDepthSlots + chunk.depthSlot;
          scaledDraw.uniformOffset = uniformOffset;
          scaledDraw.push = pushWords;
          scaledDraw.scissor = scaledBounds.rect();
          scaledDraw.firstVertex = static_cast<std::uint32_t>(vertexOffsetBytes / sizeof(GpuVertex)) + firstVertex;
          scaledDraw.vertexCount = endVertex - firstVertex;
          scaledDraw.pipeline = s.interlockTriangles;
          scaledDraw.scaledTexture = scaledTextureCopy;
          scaledDraw.textured = chunk.hasTexture;  // IN-389: diagnostics
          s.scaledDraws.push_back(scaledDraw);
        }
      }
    }
    bool anyDraw = false;
    for (std::size_t i = 0; !allTriangles && i < chunk.draws.size(); ++i) {
      const GpuDraw &draw = chunk.draws[i];
      if (draw.count == 0 || draw.first + draw.count > chunk.VertexCount()) continue;
      const std::int32_t x0 = std::max<std::int32_t>(0, draw.x0);
      const std::int32_t y0 = std::max<std::int32_t>(0, draw.y0);
      const std::int32_t x1 = std::min<std::int32_t>(kGpuSurface - 1, draw.x1);
      const std::int32_t y1 = std::min<std::int32_t>(kGpuSurface - 1, draw.y1);
      if (x1 < x0 || y1 < y0) continue;
      if (anyDraw) {
        // Storage images: the next draw reads what this draw wrote (no implicit ordering in Vulkan).
        VkMemoryBarrier between{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        between.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        between.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        coregs_hw_vkCmdPipelineBarrier(s.cmd, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                                    0, 1, &between, 0, nullptr, 0, nullptr);
        s.stats.barriers++;
      }
      VkRect2D scissor{{x0, y0}, {static_cast<std::uint32_t>(x1 - x0 + 1), static_cast<std::uint32_t>(y1 - y0 + 1)}};
      coregs_hwd_vkCmdSetScissor(s.cmd, 0, 1, &scissor);
      const std::size_t pipelineIndex = draw.topology == GpuTopology::Points  ? 0
                                        : draw.topology == GpuTopology::Lines ? 1
                                                                              : 2;
      coregs_hwd_vkCmdBindPipeline(s.cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, s.pipelines[pipelineIndex]);
      s.recorded_work();
      coregs_hwd_vkCmdDraw(s.cmd, draw.count, 1, draw.first, 0);
      anyDraw = true;
      s.stats.draws++;
      if (s.scaled()) {
        Impl::ScaledDraw scaledDraw;
        scaledDraw.descriptorSet = chunk.colourSlot * kGpuDepthSlots + chunk.depthSlot;
        scaledDraw.uniformOffset = uniformOffset;
        scaledDraw.push = pushWords;
        ScaledBounds one;
        one.add(draw);
        scaledDraw.scissor = one.rect();
        scaledDraw.firstVertex = static_cast<std::uint32_t>(vertexOffsetBytes / sizeof(GpuVertex)) + draw.first;
        scaledDraw.vertexCount = draw.count;
        scaledDraw.pipeline = s.pipelines[pipelineIndex];
        scaledDraw.scaledTexture = scaledTextureCopy;
        scaledDraw.textured = chunk.hasTexture;  // IN-389: diagnostics
        s.scaledDraws.push_back(scaledDraw);
      }
    }
    s.recorded_work();
    coregs_hwd_vkCmdEndRenderPass(s.cmd);
    if (s.scaled() && !s.scaledDraws.empty()) {
      s.mark(Impl::kTimePass);  // the 1x draws (12.8: the scaled pass is timed on its own)
      s.record_scaled_draws(!allTriangles);  // IN-371
      s.mark(Impl::kTimeScaled);
    }
  }
  s.mark(drawing ? Impl::kTimePass : Impl::kTimeUpload);

  s.stats.executions++;
  s.stats.vertices += chunk.VertexCount();
  s.stats.uploadBytes += stagingNeeded + chunk.VertexCount() * sizeof(GpuVertex);

  if (!(chunk.readbackFrame || chunk.readbackDepth)) return s.note_chunk(error);  // submitted later (IN-342)

  // Readback: make the shader and copy writes visible to the copies, copy, then submit and wait.
  VkMemoryBarrier toReadback{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
  toReadback.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
  toReadback.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
  coregs_hw_vkCmdPipelineBarrier(s.cmd, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
                              VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &toReadback, 0, nullptr, 0, nullptr);
  VkBufferImageCopy frameReadback = frameCopy;
  frameReadback.bufferOffset = 0;
  VkBufferImageCopy depthReadback = frameCopy;
  depthReadback.bufferOffset = frameBytes;
  if (chunk.readbackFrame) {
    s.recorded_work();
    coregs_hwd_vkCmdCopyImageToBuffer(s.cmd, frameImage, VK_IMAGE_LAYOUT_GENERAL, s.readback.buffer, 1, &frameReadback);
  }
  if (chunk.readbackDepth) {
    s.recorded_work();
    coregs_hwd_vkCmdCopyImageToBuffer(s.cmd, depthImage, VK_IMAGE_LAYOUT_GENERAL, s.readback.buffer, 1,
                                   &depthReadback);
  }
  VkMemoryBarrier readbackDone{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
  readbackDone.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
  readbackDone.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
  coregs_hw_vkCmdPipelineBarrier(s.cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &readbackDone,
                              0, nullptr, 0, nullptr);
  s.stats.barriers += 2;
  if (!s.submit(true, error)) return false;

  const auto *readback = static_cast<const std::uint8_t *>(s.readback.mapped);
  if (chunk.readbackFrame) std::memcpy(chunk.frameWords.data(), readback, frameBytes);
  if (chunk.readbackDepth) std::memcpy(chunk.depthWords.data(), readback + frameBytes, frameBytes);
  s.stats.readbackBytes += frameBytes * ((chunk.readbackFrame ? 1 : 0) + (chunk.readbackDepth ? 1 : 0));
  return true;
}

bool GpuDrawer::UploadGlm(const std::uint8_t *localMemory,
                          const std::vector<std::pair<std::uint32_t, std::uint32_t>> &ranges, std::string &error) {
  Impl &s = *impl_;
  VkDeviceSize total = 0;
  for (const auto &range : ranges) total += range.second;
  if (total == 0) return true;
  if (total > kStagingBytes) {
    error = "GLM upload larger than the staging ring";
    return false;
  }
  if (s.stagingUsed + total > kStagingBytes && !s.submit(false, error)) return false;
  if (!s.begin_recording(error) || !s.materialise_clut(error)) return false;  // IN-370
  std::vector<VkBufferCopy> copies;
  copies.reserve(ranges.size());
  auto *staging = static_cast<std::uint8_t *>(s.staging.mapped);
  for (const auto &range : ranges) {
    if (range.second == 0) continue;
    if (static_cast<VkDeviceSize>(range.first) + range.second > kGlmBytes) {
      error = "GLM upload range outside the 4 MiB memory";
      return false;
    }
    std::memcpy(staging + s.stagingUsed, localMemory + range.first, range.second);
    copies.push_back(VkBufferCopy{s.stagingUsed, range.first, range.second});
    s.stagingUsed += range.second;
  }
  s.full_barrier();  // earlier GLM readers finish before the bytes change
  s.recorded_work();
  coregs_hwd_vkCmdCopyBuffer(s.cmd, s.staging.buffer, s.glm.buffer, static_cast<std::uint32_t>(copies.size()),
                          copies.data());
  s.mark(Impl::kTimeUpload);
  s.full_barrier();
  s.stats.uploadBytes += total;
  return true;
}

bool GpuDrawer::SwizzleToGlm(bool depthSlot, std::uint32_t slot, std::uint32_t bp, std::uint32_t bw, std::uint32_t psm,
                             std::int32_t x0, std::int32_t y0, std::int32_t x1, std::int32_t y1, std::string &error) {
  Impl &s = *impl_;
  if ((depthSlot && slot >= kGpuDepthSlots) || (!depthSlot && slot >= kGpuColourSlots) || x1 < x0 || y1 < y0 || x0 < 0 ||
      y0 < 0 || x1 >= kGpuSurface || y1 >= kGpuSurface) {
    error = "GLM swizzle outside a slot";
    return false;
  }
  if (!s.begin_recording(error) || !s.materialise_clut(error)) return false;  // IN-370
  if (!s.flush_run(error)) return false;  // the swizzle reads the target image the run draws into
  const std::uint32_t set = depthSlot ? kGpuColourSlots + slot : slot;
  s.full_barrier();  // the target's draws and the GLM's earlier writes are complete
  s.bind_compute(s.swizzlePipeline, s.computeSets[set]);
  const std::uint32_t width = static_cast<std::uint32_t>(x1 - x0 + 1);
  const std::uint32_t height = static_cast<std::uint32_t>(y1 - y0 + 1);
  const std::uint32_t push[8] = {bp, bw, psm, static_cast<std::uint32_t>(x0), static_cast<std::uint32_t>(y0), width, height,
                                 0u};
  coregs_hwd_vkCmdPushConstants(s.cmd, s.computeLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), push);
  s.recorded_work();
  coregs_hwd_vkCmdDispatch(s.cmd, (width + 7) / 8, (height + 7) / 8, 1);
  s.mark(Impl::kTimeSwizzle);
  s.full_barrier();
  return true;
}

bool GpuDrawer::LoadSlotFromGlm(bool depthSlot, std::uint32_t slot, std::uint32_t bp, std::uint32_t bw,
                                std::uint32_t psm, std::int32_t x0, std::int32_t y0, std::int32_t x1, std::int32_t y1,
                                std::string &error) {
  Impl &s = *impl_;
  if ((depthSlot && slot >= kGpuDepthSlots) || (!depthSlot && slot >= kGpuColourSlots) || x1 < x0 || y1 < y0 || x0 < 0 ||
      y0 < 0 || x1 >= kGpuSurface || y1 >= kGpuSurface) {
    error = "GLM load outside a slot";
    return false;
  }
  // IN-384: the load grows the slot's scaled image first (copying the old content), so the refresh below covers
  // every loaded row instead of being clipped.
  if (s.scaled() &&
      !s.ensure_scaled_image(depthSlot ? s.scaledDepths[slot] : s.scaledFrames[slot],
                             depthSlot ? s.scaledDepthExtent[slot] : s.scaledFrameExtent[slot],
                             std::max<std::uint32_t>(bw, 1u) * 64u, static_cast<std::uint32_t>(y1 + 1), error)) {
    return false;
  }
  if (!s.begin_recording(error)) return false;
  if (!s.flush_run(error)) return false;  // the load writes the target image the run may draw into
  const std::uint32_t set = depthSlot ? kGpuColourSlots + slot : slot;
  s.full_barrier();  // earlier draws into the slot and the GLM's writes are complete
  s.bind_compute(s.unswizzlePipeline, s.computeSets[set]);
  const std::uint32_t width = static_cast<std::uint32_t>(x1 - x0 + 1);
  const std::uint32_t height = static_cast<std::uint32_t>(y1 - y0 + 1);
  const std::uint32_t push[8] = {bp, bw, psm, static_cast<std::uint32_t>(x0), static_cast<std::uint32_t>(y0), width, height,
                                 0u};
  coregs_hwd_vkCmdPushConstants(s.cmd, s.computeLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), push);
  s.recorded_work();
  coregs_hwd_vkCmdDispatch(s.cmd, (width + 7) / 8, (height + 7) / 8, 1);
  s.mark(Impl::kTimeSwizzle);
  s.full_barrier();
  s.refresh_scaled(depthSlot, slot, x0, y0, x1, y1);  // IN-371
  return true;
}

bool GpuDrawer::MergeIntoGlm(const std::vector<std::uint32_t> &triples, std::string &error) {
  Impl &s = *impl_;
  if (triples.empty()) return true;
  if (triples.size() % 3 != 0) {
    error = "GLM merge list is not made of triples";
    return false;
  }
  const VkDeviceSize bytes = triples.size() * sizeof(std::uint32_t);
  if (bytes > kPaletteRingBytes) {
    error = "GLM merge list larger than the ring";
    return false;
  }
  if (s.paletteUsed + bytes > kPaletteRingBytes && !s.submit(false, error)) return false;
  if (!s.begin_recording(error) || !s.materialise_clut(error)) return false;  // IN-370
  const std::uint32_t base = static_cast<std::uint32_t>(s.paletteUsed / sizeof(std::uint32_t));
  std::memcpy(static_cast<std::uint8_t *>(s.palettes.mapped) + s.paletteUsed, triples.data(), bytes);
  s.paletteUsed += bytes;
  s.full_barrier();  // earlier GLM readers and writers finish first
  s.bind_compute(s.mergePipeline, s.computeSets[0]);
  const std::uint32_t count = static_cast<std::uint32_t>(triples.size() / 3);
  const std::uint32_t push[2] = {base, count};
  coregs_hwd_vkCmdPushConstants(s.cmd, s.computeLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), push);
  s.recorded_work();
  coregs_hwd_vkCmdDispatch(s.cmd, (count + 63) / 64, 1, 1);
  s.mark(Impl::kTimeUpload);
  s.full_barrier();
  s.stats.uploadBytes += bytes;
  return true;
}

bool GpuDrawer::UploadHostToGlm(std::uint32_t bp, std::uint32_t bw, std::uint32_t psm, std::uint32_t x0,
                                std::uint32_t y0, std::uint32_t width, std::uint32_t pixels, std::uint32_t bpp,
                                const std::uint8_t *data, std::size_t byteCount, std::string &error) {
  Impl &s = *impl_;
  if (pixels == 0) return true;
  if (width == 0 || (bpp != 32 && bpp != 24 && bpp != 16 && bpp != 8 && bpp != 4) ||
      static_cast<std::uint64_t>(pixels) * bpp > static_cast<std::uint64_t>(byteCount) * 8u) {
    error = "GLM host upload with inconsistent size";
    return false;
  }
  // Whole words plus one spare word (a 24-bit texel at the end reads the next word).
  const VkDeviceSize words = (byteCount + 3) / 4 + 1;
  const VkDeviceSize bytes = words * sizeof(std::uint32_t);
  if (bytes > kPaletteRingBytes) {
    error = "GLM host upload larger than the ring";
    return false;
  }
  if (s.paletteUsed + bytes > kPaletteRingBytes && !s.submit(false, error)) return false;
  if (!s.begin_recording(error) || !s.materialise_clut(error)) return false;  // IN-370
  const std::uint32_t base = static_cast<std::uint32_t>(s.paletteUsed / sizeof(std::uint32_t));
  auto *ring = static_cast<std::uint8_t *>(s.palettes.mapped) + s.paletteUsed;
  std::memcpy(ring, data, byteCount);
  std::memset(ring + byteCount, 0, static_cast<std::size_t>(bytes) - byteCount);
  s.paletteUsed += bytes;
  s.full_barrier();  // earlier GLM readers and writers finish first
  s.bind_compute(s.hostUploadPipeline, s.computeSets[0]);
  const std::uint32_t push[9] = {bp, bw, psm, x0, y0, width, pixels, bpp, base};
  coregs_hwd_vkCmdPushConstants(s.cmd, s.computeLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), push);
  s.recorded_work();
  coregs_hwd_vkCmdDispatch(s.cmd, (pixels + 63) / 64, 1, 1);
  s.mark(Impl::kTimeUpload);
  s.full_barrier();
  s.stats.uploadBytes += byteCount;
  return true;
}

GpuDrawer::DownloadTicket GpuDrawer::RecordGlmDownload(
    const std::vector<std::pair<std::uint32_t, std::uint32_t>> &ranges, std::string &error) {
  Impl &s = *impl_;
  DownloadTicket ticket;
  if (!s.secondFrameReady) {
    error = "asynchronous downloads need the second frame set";
    return ticket;
  }
  std::vector<VkBufferCopy> copies;
  for (const auto &range : ranges) {
    if (range.second == 0) continue;
    if (static_cast<VkDeviceSize>(range.first) + range.second > kGlmBytes) {
      error = "GLM download range outside the 4 MiB memory";
      return ticket;
    }
    copies.push_back(VkBufferCopy{range.first, range.first, range.second});  // same offsets in the host buffer
  }
  if (!s.begin_recording(error)) return ticket;
  if (!s.flush_run(error)) return ticket;  // pending draws come first in command order
  const std::uint32_t buffer = s.nextDownload;
  s.nextDownload ^= 1u;
  if (!copies.empty()) {
    s.full_barrier();
    s.recorded_work();
    coregs_hwd_vkCmdCopyBuffer(s.cmd, s.glm.buffer, s.downloads[buffer].buffer, static_cast<std::uint32_t>(copies.size()),
                            copies.data());
  s.mark(Impl::kTimeDownload);
    VkMemoryBarrier done{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    done.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    done.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
    coregs_hw_vkCmdPipelineBarrier(s.cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &done, 0,
                                nullptr, 0, nullptr);
    s.stats.barriers += 2;
  }
  s.downloadRanges[buffer] = ranges;
  ticket.valid = true;
  ticket.buffer = buffer;
  ticket.serial = s.stats.submissions + 1;  // the submission that will end this recording
  return ticket;
}

bool GpuDrawer::CompleteGlmDownload(const DownloadTicket &ticket, std::uint8_t *localMemory, std::string &error) {
  Impl &s = *impl_;
  if (!ticket.valid) {
    error = "invalid download ticket";
    return false;
  }
  if (ticket.serial > s.stats.submissions && !s.submit(false, error)) return false;  // still being recorded
  if (s.completedSerial < ticket.serial) {
    // The set that holds the submission (or a later one on the same queue, which completes after it).
    Impl::FrameResources &other = s.frameSets[s.currentFrame ^ 1u];
    if (s.serial >= ticket.serial && s.inFlight) {
      if (!check(coregs_hw_vkWaitForFences(s.device, 1, &s.fence, VK_TRUE, UINT64_MAX), "vkWaitForFences", error)) {
        return false;
      }
      s.inFlight = false;
      s.completedSerial = std::max(s.completedSerial, s.serial);
      s.collect_timing(s.currentFrame);
    }
    if (s.completedSerial < ticket.serial && other.serial >= ticket.serial && other.inFlight) {
      if (!check(coregs_hw_vkWaitForFences(s.device, 1, &other.fence, VK_TRUE, UINT64_MAX), "vkWaitForFences", error)) {
        return false;
      }
      other.inFlight = false;
      s.completedSerial = std::max(s.completedSerial, other.serial);
      s.collect_timing(s.currentFrame ^ 1u);
    }
    if (s.completedSerial < ticket.serial) s.completedSerial = ticket.serial;  // neither set in flight: complete
  }
  const auto *bytes = static_cast<const std::uint8_t *>(s.downloads[ticket.buffer].mapped);
  for (const auto &range : s.downloadRanges[ticket.buffer]) {
    std::memcpy(localMemory + range.first, bytes + range.first, range.second);
  }
  return true;
}

bool GpuDrawer::ComposeDisplay(const std::vector<std::uint32_t> &parameters, std::uint32_t outWidth,
                               std::uint32_t outHeight, std::string &error) {
  Impl &s = *impl_;
  if (outWidth == 0 || outHeight == 0) {
    error = "display compose with no size";
    return false;
  }
  // IN-384: the display image grows to the composed frame (it is only allocated for the size really used).
  if (s.display.image && !s.ensure_display_image(outWidth, outHeight, error)) return false;
  const VkDeviceSize bytes = parameters.size() * sizeof(std::uint32_t);
  if (s.paletteUsed + bytes > kPaletteRingBytes && !s.submit(false, error)) return false;
  if (!s.begin_recording(error)) return false;
  if (!s.flush_run(error)) return false;  // the frame's pending draws come first
  const std::uint32_t base = static_cast<std::uint32_t>(s.paletteUsed / sizeof(std::uint32_t));
  std::memcpy(static_cast<std::uint8_t *>(s.palettes.mapped) + s.paletteUsed, parameters.data(), bytes);
  s.paletteUsed += bytes;
  s.full_barrier_with_fragment();  // GLM swizzles and uploads, and an earlier present's copy, finish first
  s.bind_compute(s.displayPipeline, s.displaySets[s.currentFrame]);
  const std::uint32_t push[3] = {base, outWidth, outHeight};
  coregs_hwd_vkCmdPushConstants(s.cmd, s.computeLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), push);
  s.recorded_work();
  coregs_hwd_vkCmdDispatch(s.cmd, (outWidth + 7) / 8, (outHeight + 7) / 8, 1);
  s.mark(Impl::kTimeDownload);
  s.full_barrier();
  return true;
}

bool GpuDrawer::ComposeDisplayScaled(const std::vector<std::uint32_t> &parameters, std::uint32_t outWidth,
                                     std::uint32_t outHeight, std::string &error) {
  Impl &s = *impl_;
  if (!s.scaled()) {
    error = "scaled display compose at 1x";
    return false;
  }
  if (outWidth == 0 || outHeight == 0) {
    error = "scaled display compose with no size";
    return false;
  }
  if (s.display.image && !s.ensure_display_image(outWidth, outHeight, error)) return false;
  const VkDeviceSize bytes = parameters.size() * sizeof(std::uint32_t);
  if (s.paletteUsed + bytes > kPaletteRingBytes && !s.submit(false, error)) return false;
  if (!s.begin_recording(error)) return false;
  if (!s.flush_run(error)) return false;  // the frame's pending draws come first
  const std::uint32_t base = static_cast<std::uint32_t>(s.paletteUsed / sizeof(std::uint32_t));
  std::memcpy(static_cast<std::uint8_t *>(s.palettes.mapped) + s.paletteUsed, parameters.data(), bytes);
  s.paletteUsed += bytes;
  s.full_barrier_with_fragment();  // scaled draws, GLM swizzles and uploads, and an earlier present's copy finish first
  coregs_hwd_vkCmdBindPipeline(s.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, s.composeScaledPipeline);
  coregs_hwd_vkCmdBindDescriptorSets(s.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, s.composeScaledLayout, 0, 1,
                                  &s.composeScaledSets[s.currentFrame], 0, nullptr);
  s.boundComputePipeline = s.composeScaledPipeline;
  s.boundComputeSet = s.composeScaledSets[s.currentFrame];
  const std::uint32_t push[4] = {base, outWidth, outHeight, s.scale};
  coregs_hwd_vkCmdPushConstants(s.cmd, s.composeScaledLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), push);
  s.recorded_work();
  coregs_hwd_vkCmdDispatch(s.cmd, (outWidth + 7) / 8, (outHeight + 7) / 8, 1);
  s.mark(Impl::kTimeDownload);
  s.full_barrier();
  return true;
}

bool GpuDrawer::SaveScaledShadow(std::uint32_t slot, std::uint32_t shadow, std::int32_t x0, std::int32_t y0,
                                 std::int32_t x1, std::int32_t y1, std::string &error) {
  Impl &s = *impl_;
  if (!s.scaled() || slot >= kGpuColourSlots || shadow >= kGpuDisplayShadows) {
    error = "display shadow save outside the scaled slots";
    return false;
  }
  x0 = std::max(x0, 0);
  y0 = std::max(y0, 0);
  // IN-384: the slot's own image, and the shadow grows to this rectangle.
  x1 = std::min(x1, static_cast<std::int32_t>(s.scaledFrameExtent[slot].width) - 1);
  y1 = std::min(y1, static_cast<std::int32_t>(s.scaledFrameExtent[slot].height) - 1);
  if (x1 < x0 || y1 < y0) return true;  // nothing of the slot lies inside the shadow
  if (!s.ensure_scaled_image(s.scaledShadows[shadow], s.scaledShadowExtent[shadow],
                             static_cast<std::uint32_t>(x1 + 1), static_cast<std::uint32_t>(y1 + 1), error) ||
      !s.ensure_scaled_image(s.nativeShadows[shadow], s.shadowNativeExtent[shadow],
                             static_cast<std::uint32_t>(x1 + 1), static_cast<std::uint32_t>(y1 + 1), error)) {
    return false;
  }
  if (!s.begin_recording(error)) return false;
  if (!s.flush_run(error)) return false;  // the slot's pending draws come first
  s.full_barrier_with_fragment();         // scaled draws and refreshes of the slot finish first
  VkImageCopy region{};
  region.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
  region.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
  region.srcOffset = {x0 * static_cast<std::int32_t>(s.scale), y0 * static_cast<std::int32_t>(s.scale), 0};
  region.dstOffset = region.srcOffset;
  region.extent = {static_cast<std::uint32_t>(x1 - x0 + 1) * s.scale, static_cast<std::uint32_t>(y1 - y0 + 1) * s.scale,
                   1};
  s.recorded_work();
  coregs_hwd_vkCmdCopyImage(s.cmd, s.scaledFrames[slot].image, VK_IMAGE_LAYOUT_GENERAL, s.scaledShadows[shadow].image,
                         VK_IMAGE_LAYOUT_GENERAL, 1, &region);
  VkImageCopy native = region;  // the 1x words the compose compares with the GLM, pixel by pixel
  native.srcOffset = {x0, y0, 0};
  native.dstOffset = native.srcOffset;
  native.extent = {static_cast<std::uint32_t>(x1 - x0 + 1), static_cast<std::uint32_t>(y1 - y0 + 1), 1};
  s.recorded_work();
  coregs_hwd_vkCmdCopyImage(s.cmd, s.frames[slot].image, VK_IMAGE_LAYOUT_GENERAL, s.nativeShadows[shadow].image,
                         VK_IMAGE_LAYOUT_GENERAL, 1, &native);
  s.mark(Impl::kTimeDownload);
  s.full_barrier();  // a later reuse of the slot (load, draw) and the compose see the copy finished
  ++s.stats.scaledShadowSaves;
  return true;
}

bool GpuDrawer::SaveToneSource(std::uint32_t slot, std::int32_t x0, std::int32_t y0, std::int32_t x1, std::int32_t y1,
                               std::string &error) {
  Impl &s = *impl_;
  if (!s.scaled() || slot >= kGpuColourSlots) {
    error = "tone source save outside the scaled slots";
    return false;
  }
  x0 = std::max(x0, 0);
  y0 = std::max(y0, 0);
  // IN-384: the slot's own image, and the saved tone source grows to this rectangle.
  x1 = std::min(x1, static_cast<std::int32_t>(s.scaledFrameExtent[slot].width) - 1);
  y1 = std::min(y1, static_cast<std::int32_t>(s.scaledFrameExtent[slot].height) - 1);
  if (x1 < x0 || y1 < y0) return true;
  if (!s.ensure_scaled_image(s.toneScaled, s.toneScaledExtent, static_cast<std::uint32_t>(x1 + 1),
                             static_cast<std::uint32_t>(y1 + 1), error)) {
    return false;
  }
  if (!s.begin_recording(error)) return false;
  if (!s.flush_run(error)) return false;
  s.full_barrier_with_fragment();
  VkImageCopy region{};
  region.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
  region.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
  region.srcOffset = {x0 * static_cast<std::int32_t>(s.scale), y0 * static_cast<std::int32_t>(s.scale), 0};
  region.dstOffset = region.srcOffset;
  region.extent = {static_cast<std::uint32_t>(x1 - x0 + 1) * s.scale, static_cast<std::uint32_t>(y1 - y0 + 1) * s.scale,
                   1};
  s.recorded_work();
  coregs_hwd_vkCmdCopyImage(s.cmd, s.scaledFrames[slot].image, VK_IMAGE_LAYOUT_GENERAL, s.toneScaled.image,
                         VK_IMAGE_LAYOUT_GENERAL, 1, &region);
  VkImageCopy native = region;
  native.srcOffset = {x0, y0, 0};
  native.dstOffset = native.srcOffset;
  native.extent = {static_cast<std::uint32_t>(x1 - x0 + 1), static_cast<std::uint32_t>(y1 - y0 + 1), 1};
  s.recorded_work();
  coregs_hwd_vkCmdCopyImage(s.cmd, s.frames[slot].image, VK_IMAGE_LAYOUT_GENERAL, s.toneNative.image,
                         VK_IMAGE_LAYOUT_GENERAL, 1, &native);
  s.mark(Impl::kTimeDownload);
  s.full_barrier();
  return true;
}

bool GpuDrawer::CarryTone(std::uint32_t slot, std::int32_t x0, std::int32_t y0, std::int32_t x1, std::int32_t y1,
                          std::string &error) {
  Impl &s = *impl_;
  if (!s.scaled() || slot >= kGpuColourSlots) {
    error = "tone carry outside the scaled slots";
    return false;
  }
  x0 = std::max(x0, 0);
  y0 = std::max(y0, 0);
  x1 = std::min(x1, std::min(static_cast<std::int32_t>(s.scaledFrameExtent[slot].width) - 1,
                             static_cast<std::int32_t>(s.toneScaledExtent.width) - 1));
  y1 = std::min(y1, std::min(static_cast<std::int32_t>(s.scaledFrameExtent[slot].height) - 1,
                             static_cast<std::int32_t>(s.toneScaledExtent.height) - 1));
  if (x1 < x0 || y1 < y0) return true;
  if (!s.begin_recording(error)) return false;
  if (!s.flush_run(error)) return false;  // the slot's pending draws come first
  s.full_barrier_with_fragment();
  coregs_hwd_vkCmdBindPipeline(s.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, s.tonePipeline);
  coregs_hwd_vkCmdBindDescriptorSets(s.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, s.toneLayout, 0, 1, &s.toneSets[slot], 0,
                                  nullptr);
  s.boundComputePipeline = s.tonePipeline;  // bind_compute rebinds its own pipeline and set afterwards
  s.boundComputeSet = s.toneSets[slot];
  const std::uint32_t width = static_cast<std::uint32_t>(x1 - x0 + 1);
  const std::uint32_t height = static_cast<std::uint32_t>(y1 - y0 + 1);
  const std::uint32_t groups[4][2] = {{8, 2},
                                      {(width + 7) / 8, (height + 7) / 8},
                                      {1, 1},
                                      {(width * s.scale + 7) / 8, (height * s.scale + 7) / 8}};
  for (std::uint32_t mode = 0; mode < 4; ++mode) {
    const std::uint32_t push[6] = {mode, static_cast<std::uint32_t>(x0), static_cast<std::uint32_t>(y0), width, height,
                                   s.scale};
    coregs_hwd_vkCmdPushConstants(s.cmd, s.toneLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), push);
    s.recorded_work();
    coregs_hwd_vkCmdDispatch(s.cmd, groups[mode][0], groups[mode][1], 1);
    s.full_barrier();
  }
  s.mark(Impl::kTimeDownload);
  ++s.stats.toneCarries;
  return true;
}

void GpuDrawer::PauseScaled(bool paused) { impl_->scaledPaused = paused; }

void GpuDrawer::SetRefreshCarry(bool depthSlot, std::uint32_t slot, std::uint32_t mask, std::int32_t x0,
                                std::int32_t y0, std::int32_t x1, std::int32_t y1) {
  Impl &s = *impl_;
  s.refreshCarry = Impl::RefreshCarry{};
  if (!s.scaled() || slot >= (depthSlot ? kGpuDepthSlots : kGpuColourSlots)) return;
  // The carry slot's scaled image must hold the rectangle (it may be smaller than its 1x words, IN-384).
  const Impl::ScaledExtent &extent = depthSlot ? s.scaledDepthExtent[slot] : s.scaledFrameExtent[slot];
  x1 = std::min(x1, static_cast<std::int32_t>(extent.width) - 1);
  y1 = std::min(y1, static_cast<std::int32_t>(extent.height) - 1);
  if (x1 < x0 || y1 < y0) return;
  s.refreshCarry = Impl::RefreshCarry{true, depthSlot, slot, mask, x0, y0, x1, y1};
}

void GpuDrawer::ClearRefreshCarry() { impl_->refreshCarry = Impl::RefreshCarry{}; }

void GpuDrawer::ArmTrace(bool armed) { impl_->traceArmed = armed; }

void GpuDrawer::ScaledExtent(bool depthSlot, std::uint32_t slot, std::uint32_t &width, std::uint32_t &height) const {
  const Impl &s = *impl_;
  width = 0;
  height = 0;
  if (!s.scaled() || slot >= (depthSlot ? kGpuDepthSlots : kGpuColourSlots)) return;
  const Impl::ScaledExtent &extent = depthSlot ? s.scaledDepthExtent[slot] : s.scaledFrameExtent[slot];
  width = extent.width;
  height = extent.height;
}

bool GpuDrawer::ScaledCovers(bool depthSlot, std::uint32_t slot, std::int32_t x1, std::int32_t y1) const {
  std::uint32_t width = 0, height = 0;
  ScaledExtent(depthSlot, slot, width, height);
  return width != 0 && height != 0 && x1 < static_cast<std::int32_t>(width) && y1 < static_cast<std::int32_t>(height);
}

std::uint64_t GpuDrawer::ScaledImageBytes() const { return impl_->scaledImageBytes; }

bool GpuDrawer::RefreshScaledSlot(bool depthSlot, std::uint32_t slot, std::int32_t x0, std::int32_t y0,
                                  std::int32_t x1, std::int32_t y1, std::string &error) {
  Impl &s = *impl_;
  if (!s.scaled() || slot >= (depthSlot ? kGpuDepthSlots : kGpuColourSlots)) {
    error = "scaled refresh outside the slots";
    return false;
  }
  if (!s.begin_recording(error)) return false;
  if (!s.flush_run(error)) return false;
  s.full_barrier_with_fragment();
  s.refresh_scaled(depthSlot, slot, x0, y0, x1, y1);
  return true;
}

bool GpuDrawer::ToneStrips(const ToneStripPage &page, std::string &error) {
  Impl &s = *impl_;
  if (page.stripCount == 0 || page.stripCount > 8 || page.destX < 0 || page.destY < 0 ||
      page.destX + static_cast<std::int32_t>(page.stripCount) * 8 > kGpuSurface || page.destY + 32 > kGpuSurface) {
    error = "tone strip page outside the surface";
    return false;
  }
  if (!s.begin_recording(error) || !s.materialise_clut(error)) return false;  // IN-370: the dispatch writes the GLM
  if (!s.flush_run(error)) return false;  // the temporary page's draws come first
  s.full_barrier();                       // the caller's GLM writes and the run's draws are complete
  s.bind_compute(s.toneStripsPipeline, s.computeSets[0]);
  const std::uint32_t width = page.stripCount * 8u;
  const std::uint32_t constPart[11] = {page.destBp,      page.destBw,
                                       static_cast<std::uint32_t>(page.destX), static_cast<std::uint32_t>(page.destY),
                                       width,           32u,
                                       page.sourceBp,   page.sourceBw,
                                       page.indexBp,    page.indexBw,
                                       page.stripCount};
  std::uint32_t push[19] = {};
  std::memcpy(push, constPart, sizeof constPart);
  for (std::uint32_t strip = 0; strip < 8; ++strip) push[11 + strip] = page.quadrants[strip];
  coregs_hwd_vkCmdPushConstants(s.cmd, s.computeLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                             static_cast<std::uint32_t>(sizeof push), push);
  s.recorded_work();
  coregs_hwd_vkCmdDispatch(s.cmd, (width * 32u + 63u) / 64u, 1, 1);
  s.mark(Impl::kTimePass);  // 1x raster work (IN-383: the strips are drawn by one dispatch instead of eight draws)
  s.full_barrier();
  ++s.stats.toneStripPages;
  ++s.stats.toneStripDispatches;
  s.stats.toneStripBatches += page.stripCount;
  return true;
}

bool GpuDrawer::DeinterlaceScaled(const DeinterlaceStep &step, std::string &error) {
  Impl &s = *impl_;
  if (!s.scaled() || step.width == 0 || step.height == 0 || step.width > kGpuDeinterlaceWidth * s.scale ||
      step.height > kGpuDeinterlaceRows * s.scale || step.current >= kGpuDeinterlaceHistory || step.mode < 2 ||
      step.mode > 5) {
    error = "scaled deinterlace outside its images";
    return false;
  }
  if (!s.begin_recording(error)) return false;
  if (!s.flush_run(error)) return false;
  if (!s.create_deinterlace(error)) return false;
  s.full_barrier();  // the compose wrote the display image
  VkImageCopy region{};
  region.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
  region.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
  region.extent = {step.width, step.height, 1};
  s.recorded_work();
  coregs_hwd_vkCmdCopyImage(s.cmd, s.display.image, VK_IMAGE_LAYOUT_GENERAL, s.deinterlaceHistory[step.current].image,
                         VK_IMAGE_LAYOUT_GENERAL, 1, &region);
  s.full_barrier();
  coregs_hwd_vkCmdBindPipeline(s.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, s.deinterlacePipeline);
  coregs_hwd_vkCmdBindDescriptorSets(s.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, s.deinterlaceLayout, 0, 1, &s.deinterlaceSet,
                                  0, nullptr);
  s.boundComputePipeline = s.deinterlacePipeline;
  s.boundComputeSet = s.deinterlaceSet;
  auto dispatch = [&](std::uint32_t mode) {
    const std::uint32_t push[14] = {mode, step.width, step.height, s.scale, step.rowParity, step.field, step.current,
                                    step.previousSame, step.previousOther, step.haveOutput ? 1u : 0u,
                                    step.thresholdPermille, step.initWoven ? 1u : 0u, step.rowStep,
                                    step.olderOther};
    coregs_hwd_vkCmdPushConstants(s.cmd, s.deinterlaceLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), push);
    s.recorded_work();
    coregs_hwd_vkCmdDispatch(s.cmd, (step.width + 7) / 8, (step.height + 7) / 8, 1);
    s.full_barrier();
  };
  if (step.mode == 2 || step.mode == 3) {
    dispatch(1);  // weave update
    dispatch(step.mode);
  } else {
    dispatch(step.mode);
  }
  if (step.mode == 5) {
    s.recorded_work();
    coregs_hwd_vkCmdCopyImage(s.cmd, s.display.image, VK_IMAGE_LAYOUT_GENERAL, s.deinterlaceOutput.image,
                           VK_IMAGE_LAYOUT_GENERAL, 1, &region);
    s.full_barrier();
  }
  s.lastDeinterlaceCurrent = step.current;  // IN-387: which merge image a later diagnostic read returns
  s.mark(Impl::kTimeDownload);
  ++s.stats.scaledDeinterlaces;
  return true;
}

bool GpuDrawer::ApplyFxaa(std::uint32_t width, std::uint32_t height, std::string &error) {
  Impl &s = *impl_;
  if (width == 0 || height == 0) {
    error = "FXAA with no size";
    return false;
  }
  if (!s.begin_recording(error)) return false;
  if (!s.flush_run(error)) return false;
  if (!s.create_fxaa(error)) return false;
  if (width > s.fxaaWidth || height > s.fxaaHeight) {
    // IN-384: the FXAA image follows the display image's growth.
    if (!s.submit(true, error)) return false;
    s.destroy_image(s.fxaaImage);
    s.fxaaWidth = std::max(width, s.displayWidth);
    s.fxaaHeight = std::max(height, s.displayHeight);
    if (!s.make_image(s.fxaaImage, s.fxaaWidth, s.fxaaHeight, error)) return false;
    if (!s.begin_recording(error)) return false;
    VkImageMemoryBarrier toGeneral{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    toGeneral.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    toGeneral.newLayout = VK_IMAGE_LAYOUT_GENERAL;
    toGeneral.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    toGeneral.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    toGeneral.image = s.fxaaImage.image;
    toGeneral.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    toGeneral.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT;
    coregs_hw_vkCmdPipelineBarrier(s.cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0,
                                nullptr, 0, nullptr, 1, &toGeneral);
    VkDescriptorImageInfo image{VK_NULL_HANDLE, s.fxaaImage.view, VK_IMAGE_LAYOUT_GENERAL};
    VkWriteDescriptorSet write{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    write.dstSet = s.fxaaSet;
    write.dstBinding = 1;
    write.descriptorCount = 1;
    write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    write.pImageInfo = &image;
    coregs_hwd_vkUpdateDescriptorSets(s.device, 1, &write, 0, nullptr);
    s.scaledImageBytes = s.scaled_bytes();
  }
  s.full_barrier();  // the compose wrote the display image; an earlier present may still copy the FXAA image
  coregs_hwd_vkCmdBindPipeline(s.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, s.fxaaPipeline);
  coregs_hwd_vkCmdBindDescriptorSets(s.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, s.fxaaLayout, 0, 1, &s.fxaaSet, 0, nullptr);
  s.boundComputePipeline = s.fxaaPipeline;
  s.boundComputeSet = s.fxaaSet;
  const std::uint32_t push[2] = {width, height};
  coregs_hwd_vkCmdPushConstants(s.cmd, s.fxaaLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), push);
  s.recorded_work();
  coregs_hwd_vkCmdDispatch(s.cmd, (width + 7) / 8, (height + 7) / 8, 1);
  s.mark(Impl::kTimeDownload);
  s.full_barrier();
  ++s.stats.fxaaFrames;
  return true;
}

std::uint64_t GpuDrawer::FxaaImage() const { return reinterpret_cast<std::uint64_t>(impl_->fxaaImage.image); }

void GpuDrawer::SetScale(std::uint32_t scale, float scaleOffset) {
  impl_->scale = std::clamp<std::uint32_t>(scale, 1u, kGpuMaxScale);
  impl_->scaleOffset = scaleOffset;
  const char *filter = std::getenv("COREGS_HW_SCALED_FILTER");
  impl_->scaledLinearFilter = filter != nullptr && std::strcmp(filter, "linear") == 0;
  const char *dither = std::getenv("COREGS_HW_DITHER");
  impl_->ditherFine = dither == nullptr || std::strcmp(dither, "native") != 0;
  const char *refresh = std::getenv("COREGS_HW_SCALED_REFRESH");
  impl_->scaledRefreshLinear = refresh != nullptr && std::strcmp(refresh, "linear") == 0;
  const char *snap = std::getenv("COREGS_HW_SCALED_SNAP");
  // IN-388: default OFF. Measured on a menu dialog at 6x against the 1x presentation of the same VBlank: the snap
  // halves the strongest sliver (23.5 -> 11.9 levels) but splits it into two, shifts the whole picture by half a native
  // pixel and costs 1.5 dB of box PSNR (23.06 against 24.52), so it is not the fix; it stays as a diagnostic switch.
  impl_->scaledSnap = snap != nullptr && std::strcmp(snap, "native") == 0;
  const char *depth = std::getenv("COREGS_HW_SCALED_DEPTH");
  impl_->scaledDepthFromPosition = depth == nullptr || std::strcmp(depth, "varying") != 0;  // IN-392
  const char *scissor = std::getenv("COREGS_HW_SCALED_SCISSOR");
  impl_->scaledScissorShifted = scissor != nullptr && std::strcmp(scissor, "shifted") == 0;  // IN-393
  const char *diag = std::getenv("COREGS_HW_SCALED_DIAG");
  impl_->scaledDiagBits = (diag && std::strstr(diag, "noalpha") ? 0x08000000u : 0u) |
                          (diag && std::strstr(diag, "notex") ? 0x04000000u : 0u) |
                          (diag && std::strstr(diag, "noz") ? 0x02000000u : 0u);
  // IN-387: one line so a diagnostic run can prove which switches reached the drawer (the three raw switches are not
  // part of the option table and therefore never appear in the option dump).
  std::fprintf(stderr, "[scale] scale=%u filter=%s refresh=%s dither=%s snap=%s depth=%s\n", impl_->scale,
               impl_->scaledLinearFilter ? "linear" : "nearest",
               impl_->scaledRefreshLinear ? "linear" : "nearest", impl_->ditherFine ? "fine" : "native",
               impl_->scaledSnap ? "native" : "off", impl_->scaledDepthFromPosition ? "position" : "varying");
}

std::uint32_t GpuDrawer::Scale() const { return impl_->scale; }

std::uint64_t GpuDrawer::DisplayImage() const { return reinterpret_cast<std::uint64_t>(impl_->display.image); }

bool GpuDrawer::ReadImageDebug(std::uint32_t which, std::uint32_t slot, std::uint32_t width, std::uint32_t height,
                               std::vector<std::uint32_t> &words, std::string &error) {
  Impl &s = *impl_;
  // IN-387: 3 = a display shadow (IN-377), 4 = the composed display image and 5 = the last scaled deinterlace's
  // merge image (before the algorithm), so one VBlank can be split into the slot's scaled image, the shadow copy,
  // the merge and the composed frame (the owner's flat-menu artifact).
  // 6 = a depth slot's scaled image and 7 = its 1x words (the notched 3D rings at scale).
  if (!s.scaled() || which > 7 || (which != 3 && which < 6 && slot >= kGpuColourSlots) ||
      (which == 3 && slot >= kGpuDisplayShadows) || (which >= 6 && slot >= kGpuDepthSlots)) {
    error = "debug read outside the scaled images";
    return false;
  }
  const VkImage source = which == 0   ? s.scaledFrames[slot].image
                         : which == 1 ? s.toneScaled.image
                         : which == 2 ? s.frames[slot].image
                         : which == 3 ? s.scaledShadows[slot].image
                         : which == 4 ? s.display.image
                         : which == 6 ? s.scaledDepths[slot].image
                         : which == 7 ? s.depths[slot].image
                                      : s.deinterlaceHistory[s.lastDeinterlaceCurrent].image;
  return read_image(reinterpret_cast<std::uint64_t>(source), width, height, words, error);
}

bool GpuDrawer::ReadDisplay(std::uint32_t width, std::uint32_t height, std::vector<std::uint32_t> &words,
                            std::string &error, bool fromFxaa) {
  Impl &s = *impl_;
  if (fromFxaa && !s.fxaaImage.image) {
    error = "FXAA image read before any FXAA pass";
    return false;
  }
  const VkImage source = fromFxaa ? s.fxaaImage.image : s.display.image;
  if (width == 0 || height == 0 || width > s.displayWidth || height > s.displayHeight) {
    error = "display read outside the display image";
    return false;
  }
  return read_image(reinterpret_cast<std::uint64_t>(source), width, height, words, error);
}

bool GpuDrawer::read_image(std::uint64_t image, std::uint32_t width, std::uint32_t height,
                          std::vector<std::uint32_t> &words, std::string &error) {
  Impl &s = *impl_;
  const VkImage source = reinterpret_cast<VkImage>(image);
  words.resize(static_cast<std::size_t>(width) * height);
  // IN-372: a scaled frame can exceed the readback buffer; it is read in bands of whole rows.
  const std::uint32_t bandRows =
      static_cast<std::uint32_t>(std::min<VkDeviceSize>(height, kReadbackBytes / (static_cast<VkDeviceSize>(width) * 4)));
  for (std::uint32_t row = 0; row < height; row += bandRows) {
    const std::uint32_t rows = std::min(bandRows, height - row);
    if (!s.begin_recording(error)) return false;
    if (!s.flush_run(error)) return false;
    s.full_barrier();
    VkBufferImageCopy copy{};
    copy.bufferRowLength = width;
    copy.bufferImageHeight = rows;
    copy.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    copy.imageOffset = {0, static_cast<std::int32_t>(row), 0};
    copy.imageExtent = {width, rows, 1};
    s.recorded_work();
    coregs_hwd_vkCmdCopyImageToBuffer(s.cmd, source, VK_IMAGE_LAYOUT_GENERAL, s.readback.buffer, 1, &copy);
    VkMemoryBarrier done{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    done.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    done.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
    coregs_hw_vkCmdPipelineBarrier(s.cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &done, 0,
                                nullptr, 0, nullptr);
    if (!s.submit(true, error)) return false;
    std::memcpy(words.data() + static_cast<std::size_t>(row) * width, s.readback.mapped,
                static_cast<std::size_t>(rows) * width * sizeof(std::uint32_t));
  }
  return true;
}

bool GpuDrawer::DownloadGlm(const std::vector<std::pair<std::uint32_t, std::uint32_t>> &ranges,
                            std::uint8_t *localMemory, std::string &error) {
  Impl &s = *impl_;
  std::vector<VkBufferCopy> copies;
  VkDeviceSize packed = 0;
  for (const auto &range : ranges) {
    if (range.second == 0) continue;
    if (static_cast<VkDeviceSize>(range.first) + range.second > kGlmBytes) {
      error = "GLM download range outside the 4 MiB memory";
      return false;
    }
    copies.push_back(VkBufferCopy{range.first, packed, range.second});
    packed += range.second;
  }
  if (copies.empty()) return true;
  if (packed > kReadbackBytes) {
    error = "GLM download larger than the readback buffer";
    return false;
  }
  if (!s.begin_recording(error) || !s.materialise_clut(error)) return false;  // IN-370
  s.full_barrier();
  s.recorded_work();
  coregs_hwd_vkCmdCopyBuffer(s.cmd, s.glm.buffer, s.readback.buffer, static_cast<std::uint32_t>(copies.size()),
                          copies.data());
  s.mark(Impl::kTimeDownload);
  VkMemoryBarrier done{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
  done.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
  done.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
  coregs_hw_vkCmdPipelineBarrier(s.cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &done, 0, nullptr,
                              0, nullptr);
  if (!s.submit(true, error)) return false;
  const auto *bytes = static_cast<const std::uint8_t *>(s.readback.mapped);
  for (const VkBufferCopy &copy : copies) {
    std::memcpy(localMemory + copy.srcOffset, bytes + copy.dstOffset, static_cast<std::size_t>(copy.size));
  }
  s.stats.readbackBytes += packed;
  return true;
}

bool GpuDrawer::LoadClutFromGlm(std::uint32_t cbp, std::uint32_t cpsm, std::uint32_t csm, std::uint32_t csa,
                                std::uint32_t psm, std::uint32_t cbw, std::uint32_t cou, std::uint32_t cov,
                                std::string &error) {
  Impl &s = *impl_;
  Impl::ClutLoadRecord load;
  load.cbp = cbp;
  load.cpsm = cpsm;
  load.csm = csm;
  load.csa = csa;
  load.eightBit = (psm & 7u) == 3u ? 1u : 0u;
  load.cbw = cbw;
  load.cou = cou;
  load.cov = cov;
  load.halfwords = Impl::clut_halfwords_written(cpsm, csa, load.eightBit);
  s.pendingClut.push_back(load);
  ++s.workGeneration;  // IN-369: a decode after this load must not reuse one from before it
  if (!s.clutDefer || s.pendingClut.size() >= Impl::kMaxPendingClut) return s.materialise_clut(error);
  return true;
}

bool GpuDrawer::UploadClut(const std::uint16_t *halfwords, std::string &error) {
  Impl &s = *impl_;
  if (s.stagingUsed + kClutBytes > kStagingBytes && !s.submit(false, error)) return false;
  if (!s.begin_recording(error) || !s.materialise_clut(error)) return false;  // IN-370
  auto *staging = reinterpret_cast<std::uint32_t *>(static_cast<std::uint8_t *>(s.staging.mapped) + s.stagingUsed);
  for (std::uint32_t i = 0; i < 512; ++i) staging[i] = halfwords[i];
  s.full_barrier();
  const VkBufferCopy copy{s.stagingUsed, 0, kClutBytes};
  s.recorded_work();
  coregs_hwd_vkCmdCopyBuffer(s.cmd, s.staging.buffer, s.clut.buffer, 1, &copy);
  s.stagingUsed += kClutBytes;
  s.mark(Impl::kTimeClut);
  s.full_barrier();
  return true;
}

bool GpuDrawer::ReadClut(std::vector<std::uint16_t> &halfwords, std::string &error) {
  Impl &s = *impl_;
  if (!s.begin_recording(error) || !s.materialise_clut(error)) return false;  // IN-370
  s.full_barrier();
  const VkBufferCopy copy{0, 0, kClutBytes};
  s.recorded_work();
  coregs_hwd_vkCmdCopyBuffer(s.cmd, s.clut.buffer, s.readback.buffer, 1, &copy);
  VkMemoryBarrier done{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
  done.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
  done.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
  coregs_hw_vkCmdPipelineBarrier(s.cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &done, 0, nullptr,
                              0, nullptr);
  if (!s.submit(true, error)) return false;
  const auto *words = static_cast<const std::uint32_t *>(s.readback.mapped);
  halfwords.resize(512);
  for (std::uint32_t i = 0; i < 512; ++i) halfwords[i] = static_cast<std::uint16_t>(words[i]);
  return true;
}

bool GpuDrawer::ReadGlm(std::vector<std::uint32_t> &words, std::string &error) {
  Impl &s = *impl_;
  if (!s.begin_recording(error) || !s.materialise_clut(error)) return false;  // IN-370
  s.full_barrier();
  const VkBufferCopy copy{0, 0, kGlmBytes};
  s.recorded_work();
  coregs_hwd_vkCmdCopyBuffer(s.cmd, s.glm.buffer, s.readback.buffer, 1, &copy);
  VkMemoryBarrier done{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
  done.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
  done.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
  coregs_hw_vkCmdPipelineBarrier(s.cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &done, 0, nullptr,
                              0, nullptr);
  if (!s.submit(true, error)) return false;
  words.resize(kGlmBytes / 4);
  std::memcpy(words.data(), s.readback.mapped, kGlmBytes);
  return true;
}

bool GpuDrawer::ReadTexels(std::uint32_t width, std::uint32_t height, std::vector<std::uint32_t> &words,
                           std::string &error) {
  Impl &s = *impl_;
  if (width == 0 || height == 0 || width > kTextureMax || height > kTextureMax) {
    error = "texel read outside the texel image";
    return false;
  }
  if (!s.begin_recording(error)) return false;
  s.full_barrier();
  VkBufferImageCopy copy{};
  copy.bufferRowLength = width;
  copy.bufferImageHeight = height;
  copy.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
  copy.imageExtent = {width, height, 1};
  s.recorded_work();
  coregs_hwd_vkCmdCopyImageToBuffer(s.cmd, s.texels.image, VK_IMAGE_LAYOUT_GENERAL, s.readback.buffer, 1, &copy);
  VkMemoryBarrier done{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
  done.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
  done.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
  coregs_hw_vkCmdPipelineBarrier(s.cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &done, 0, nullptr,
                              0, nullptr);
  if (!s.submit(true, error)) return false;
  words.resize(static_cast<std::size_t>(width) * height);
  std::memcpy(words.data(), s.readback.mapped, words.size() * sizeof(std::uint32_t));
  return true;
}

} // namespace coregs::gpu
