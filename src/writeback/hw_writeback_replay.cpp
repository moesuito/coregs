// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Joao Alano and GT4Recomp contributors.
// Written from the TASK-026 specification (docs/hw-renderer-spec), clean room (decision 0069).
//
// HWR replay tool (M6): replays a recorded GS stream through the GPU write-back renderer and writes one
// PPM per frame, so the GPU path can be compared with the CPU reference on exactly the same input.
// Chapters 14.3 (replay) and 12.9 (frame dumps); the record formats are the ones cpuref_replay reads.
//
// Usage: hw_writeback_replay <capture> <out-dir> [--every N] [--slm-frame N] [--trace-frame N]
//        [--trace-rect x0,y0,x1,y1] [--slm-hashes <file>] [--compose reference|presentation]
//        [--deinterlace auto|off|weave|bob|blend|adaptive[_bff]] [--sync-mode cache|writeback] [--no-interlock]
//        [--no-target-textures] [--no-glm] [--no-cpu-sprites] [--no-gpu-clut] [--no-glm-park] [--no-gpu-uploads]
//        [--no-runs] [--single-buffer] [--async-present] [--no-batched-writebacks] [--gpu-timing]
//        [--gpu-compose-check]
//        [--chunks-per-submission N]
// --sync-mode selects COREGS_HW_SYNC_MODE (default cache, as in the game; writeback is the reference mode).
// --compose presentation writes the image the window would show (13.3 field offset, 13.4 deinterlace with
// the default `auto`) instead of the reference image (12.9 `vram`, the CPU reference's image).
// --slm-hashes writes "<frame> <16 hex digits>" per frame (the FNV-1a hash of the 4 MiB local memory),
// the same format as hw_cpuref_replay, so the two paths can be compared in memory on every frame.
// --trace-frame prints every batch of that frame (state, rectangle, vertices) whose rectangle overlaps
// --trace-rect (default: everything), so a pixel difference can be tied to the batch that drew it.
// Exit 77 means "no Vulkan device with graphics and compute" (the caller reports it as skipped).

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "platform/hw_platform.hpp"

#include "hw_gs_vulkan_loader.hpp"
#include "hw_gs_vulkan_device_table.hpp"
#include "writeback/hw_gs_writeback.hpp"
#include "../cpuref/cpuref_replay.hpp"
#include "writeback/hw_writeback_replay_target.hpp"

namespace {

using namespace coregs;

struct HeadlessDevice {
    VkInstance instance = VK_NULL_HANDLE;
    VkPhysicalDevice physical = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    VkQueue queue = VK_NULL_HANDLE;
    std::uint32_t family = 0;
    bool wantInterlock = true;
    bool interlock = false;
    bool independentBlend = false;
    bool dualSourceBlend = false;
    coregs::DeviceProfile profile;  // TASK-179: the capability probe, with the features enabled below

    bool Create(std::string& error) {
        if (!coregs::vulkan_loader::load_loader(error)) return false;
        VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
        app.pApplicationName = "hw_writeback_replay";
        app.apiVersion = VK_API_VERSION_1_2;
        VkInstanceCreateInfo info{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
        info.pApplicationInfo = &app;
        if (coregs_hw_vkCreateInstance(&info, nullptr, &instance) != VK_SUCCESS) {
            error = "vkCreateInstance failed";
            return false;
        }
        std::vector<std::string> missing;
        if (!coregs::vulkan_loader::load_headless_entry_points(instance, missing)) {
            error = "instance entry points missing:";
            for (const std::string& name : missing) error += " " + name;
            return false;
        }
        std::uint32_t count = 0;
        coregs_hw_vkEnumeratePhysicalDevices(instance, &count, nullptr);
        std::vector<VkPhysicalDevice> devices(count);
        coregs_hw_vkEnumeratePhysicalDevices(instance, &count, devices.data());
        for (VkPhysicalDevice candidate : devices) {
            VkPhysicalDeviceProperties props{};
            coregs_hw_vkGetPhysicalDeviceProperties(candidate, &props);
            if (props.apiVersion < VK_API_VERSION_1_2) continue;
            std::uint32_t families = 0;
            coregs_hw_vkGetPhysicalDeviceQueueFamilyProperties(candidate, &families, nullptr);
            std::vector<VkQueueFamilyProperties> queueProps(families);
            coregs_hw_vkGetPhysicalDeviceQueueFamilyProperties(candidate, &families, queueProps.data());
            for (std::uint32_t f = 0; f < families; ++f) {
                const VkQueueFlags wanted = VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT;
                if ((queueProps[f].queueFlags & wanted) != wanted) continue;
                physical = candidate;
                family = f;
                break;
            }
            if (physical) break;
        }
        if (!physical) {
            error = "no Vulkan 1.2 device with graphics and compute";
            return false;
        }
        const float priority = 1.0f;
        VkDeviceQueueCreateInfo queueInfo{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
        queueInfo.queueFamilyIndex = family;
        queueInfo.queueCount = 1;
        queueInfo.pQueuePriorities = &priority;
        // IN-345: enable pixel interlock when the device has it (the batch-per-draw path), unless the
        // caller asked for the per-primitive path (A/B: --no-interlock / COREGS_HW_TEST_NO_INTERLOCK=1).
        interlock = wantInterlock && coregs::vulkan_loader::pixel_interlock_supported(physical);
        VkPhysicalDeviceFragmentShaderInterlockFeaturesEXT interlockFeatures{
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FRAGMENT_SHADER_INTERLOCK_FEATURES_EXT};
        interlockFeatures.fragmentShaderPixelInterlock = VK_TRUE;
        const char* interlockExtension = VK_EXT_FRAGMENT_SHADER_INTERLOCK_EXTENSION_NAME;
        VkDeviceCreateInfo deviceInfo{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
        deviceInfo.queueCreateInfoCount = 1;
        deviceInfo.pQueueCreateInfos = &queueInfo;
        VkPhysicalDeviceFeatures2 supportedFeatures{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
        VkPhysicalDeviceFeatures enabledFeatures{};
        coregs_hw_vkGetPhysicalDeviceFeatures2(physical, &supportedFeatures);
        enabledFeatures.independentBlend = supportedFeatures.features.independentBlend;
        enabledFeatures.dualSrcBlend = supportedFeatures.features.dualSrcBlend;
        // TASK-179: largePoints and shaderStorageImageArrayDynamicIndexing where the device has them, as the game's device.
        profile = coregs::vulkan_loader::probe_device_profile(instance, physical);
        coregs::vulkan_loader::enable_profile_features(profile, enabledFeatures);
        dualSourceBlend = supportedFeatures.features.dualSrcBlend == VK_TRUE;
        independentBlend = supportedFeatures.features.independentBlend == VK_TRUE;
        deviceInfo.pEnabledFeatures = &enabledFeatures;
        if (interlock) {
            deviceInfo.pNext = &interlockFeatures;
            deviceInfo.enabledExtensionCount = 1;
            deviceInfo.ppEnabledExtensionNames = &interlockExtension;
        }
        if (coregs_hw_vkCreateDevice(physical, &deviceInfo, nullptr, &device) != VK_SUCCESS) {
            error = "vkCreateDevice failed";
            return false;
        }
        coregs_hw_vkGetDeviceQueue(device, family, 0, &queue);
        return coregs::vulkan_device::load_device_entry_points(instance, missing);
    }
};

bool WritePpm(const std::string& path, const cpuref::FrameImage& frame) {
    std::FILE* file = std::fopen(path.c_str(), "wb");
    if (!file) return false;
    std::fprintf(file, "P6\n%u %u\n255\n", frame.image.width, frame.image.height);
    for (std::uint32_t y = 0; y < frame.image.height; ++y) {
        for (std::uint32_t x = 0; x < frame.image.width; ++x) {
            const display::Rgba8& pixel = frame.image.at(x, y);
            const unsigned char rgb[3] = {pixel.r, pixel.g, pixel.b};
            std::fwrite(rgb, 1, 3, file);
        }
    }
    std::fclose(file);
    return true;
}

// IN-372: an RGBA8-packed GPU frame (red in the low byte) as a P6 PPM.
bool WritePackedPpm(const std::string& path, std::uint32_t width, std::uint32_t height,
                    const std::vector<std::uint32_t>& pixels) {
    std::FILE* file = std::fopen(path.c_str(), "wb");
    if (!file) return false;
    std::fprintf(file, "P6\n%u %u\n255\n", width, height);
    std::vector<unsigned char> row(static_cast<std::size_t>(width) * 3);
    for (std::uint32_t y = 0; y < height; ++y) {
        for (std::uint32_t x = 0; x < width; ++x) {
            const std::uint32_t pixel = pixels[static_cast<std::size_t>(y) * width + x];
            row[x * 3 + 0] = static_cast<unsigned char>(pixel & 0xFFu);
            row[x * 3 + 1] = static_cast<unsigned char>((pixel >> 8) & 0xFFu);
            row[x * 3 + 2] = static_cast<unsigned char>((pixel >> 16) & 0xFFu);
        }
        std::fwrite(row.data(), 1, row.size(), file);
    }
    std::fclose(file);
    return true;
}

bool ReadFile(const char* path, std::vector<std::uint8_t>& out) {
    std::FILE* file = std::fopen(path, "rb");
    if (!file) return false;
    // 64-bit offsets: race captures are several GB (long is 32 bits on Windows).
    platform::SeekFile(file, 0, SEEK_END);
    const long long size = static_cast<long long>(platform::TellFile(file));
    platform::SeekFile(file, 0, SEEK_SET);
    if (size > 0) {
        out.resize(static_cast<std::size_t>(size));
        const std::size_t read = std::fread(out.data(), 1, out.size(), file);
        out.resize(read);
    }
    std::fclose(file);
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: hw_writeback_replay <capture> <out-dir> [--every N]\n");
        return 2;
    }
    const std::string capturePath = argv[1];
    const std::string outDir = argv[2];
    std::uint64_t every = 1;
    std::uint64_t slmFrame = UINT64_MAX;  // the frame whose raw shadow local memory is dumped
    std::uint64_t traceFrame = UINT64_MAX;  // the frame whose batches are printed
    std::string slmHashPath;                // per-frame local memory hashes
    writeback::ComposeKind composeKind = writeback::ComposeKind::Reference;
    writeback::OutputSettings outputSettings;  // --deinterlace (13.4 names, `_bff` suffix allowed)
    bool cacheMode = true;                     // --sync-mode
    PixelRect traceRect{0, 0, 2047, 2047};
    std::uint32_t scale = 1;     // --scale N (M12, IN-371): internal resolution of the scaled images
    // IN-384 follow-up: --half-pixel native|normal|off sets the 06.3 vertex shift exactly as COREGS_HW_HALF_PIXEL does in
    // The game, so the presentation can be compared between the three conventions.
    float halfPixelShift = 0.0f;  // 0: the native mode (shift = scale), as the game defaults to
    std::string upscaledDir;     // --upscaled-dir DIR: the GPU presentation compose at that scale (IN-372)
    // --check-presented (owner test 6, IN-377): every frame's GPU presentation compose (the image the window shows) is
    // read back and its s x s blocks inspected. A frame sourced from 1x memory has every block uniform (each native
    // texel repeated s x s times); a really upscaled frame has edge blocks with several values. Owner test 7 (IN-379,
    // IN-380): a frame can also be a 1x picture with a few scaled overlays (the HUD). IN-381: and a flat 2D menu has
    // few edges at all. So detail is measured against the frame's own 1x picture: every native pixel that differs
    // from its right or lower neighbour is an edge, and an upscaled frame has a mixed block along almost every edge
    // (06.3 moves edges inside the blocks; real sub-pixel detail does the rest). A frame with edges on at least
    // kMinEdgeShare of its native pixels is low-detail when its mixed blocks are fewer than kMinDetailRatio x its
    // edges; frames with fewer edges are flat (black screens, fades) and only counted.
    bool checkPresented = false;
    for (int i = 3; i < argc; ++i) {
        if (std::strcmp(argv[i], "--check-presented") == 0) checkPresented = true;
    }
    for (int i = 3; i + 1 < argc; i += 2) {
        if (std::strcmp(argv[i], "--no-interlock") == 0 || std::strcmp(argv[i], "--no-target-textures") == 0 ||
            std::strcmp(argv[i], "--no-glm") == 0 || std::strcmp(argv[i], "--no-cpu-sprites") == 0 ||
            std::strcmp(argv[i], "--no-gpu-clut") == 0 || std::strcmp(argv[i], "--no-glm-park") == 0 ||
            std::strcmp(argv[i], "--no-gpu-uploads") == 0 || std::strcmp(argv[i], "--no-runs") == 0 ||
            std::strcmp(argv[i], "--single-buffer") == 0 || std::strcmp(argv[i], "--async-present") == 0 ||
            std::strcmp(argv[i], "--no-batched-writebacks") == 0 ||
            std::strcmp(argv[i], "--gpu-timing") == 0 || std::strcmp(argv[i], "--gpu-compose-check") == 0 ||
            std::strcmp(argv[i], "--fxaa") == 0 || std::strcmp(argv[i], "--check-presented") == 0 ||
            std::strcmp(argv[i], "--no-gpu-compose") == 0 || std::strcmp(argv[i], "--no-tone-hle") == 0 ||
            std::strcmp(argv[i], "--tone-verify") == 0) {
            --i;  // a flag without a value
            continue;
        }
        if (std::strcmp(argv[i], "--chunks-per-submission") == 0) continue;  // applied after Init
        if (std::strcmp(argv[i], "--every") == 0) {
            const long value = std::strtol(argv[i + 1], nullptr, 10);
            if (value > 0) every = static_cast<std::uint64_t>(value);
        } else if (std::strcmp(argv[i], "--slm-frame") == 0) {
            slmFrame = static_cast<std::uint64_t>(std::strtoull(argv[i + 1], nullptr, 10));
        } else if (std::strcmp(argv[i], "--compose") == 0) {
            if (std::strcmp(argv[i + 1], "presentation") == 0) {
                composeKind = writeback::ComposeKind::Presentation;
            } else if (std::strcmp(argv[i + 1], "reference") != 0) {
                std::fprintf(stderr, "--compose takes reference or presentation\n");
                return 2;
            }
        } else if (std::strcmp(argv[i], "--sync-mode") == 0) {
            if (std::strcmp(argv[i + 1], "writeback") == 0) {
                cacheMode = false;
            } else if (std::strcmp(argv[i + 1], "cache") != 0) {
                std::fprintf(stderr, "--sync-mode takes cache or writeback\n");
                return 2;
            }
        } else if (std::strcmp(argv[i], "--deinterlace") == 0) {
            const auto setting = display::parse_deinterlace_option(argv[i + 1]);
            if (!setting) {
                std::fprintf(stderr, "--deinterlace takes auto|off|weave|bob|blend|adaptive with optional _bff\n");
                return 2;
            }
            outputSettings.deinterlace = setting->mode;
            outputSettings.bottomFieldFirst = setting->bottom_first;
        } else if (std::strcmp(argv[i], "--scale") == 0) {
            const long value = std::strtol(argv[i + 1], nullptr, 10);
            if (value < 1 || value > 8) {
                std::fprintf(stderr, "--scale takes 1..8\n");
                return 2;
            }
            scale = static_cast<std::uint32_t>(value);
        } else if (std::strcmp(argv[i], "--upscaled-dir") == 0) {
            upscaledDir = argv[i + 1];
        } else if (std::strcmp(argv[i], "--slm-hashes") == 0) {
            slmHashPath = argv[i + 1];
        } else if (std::strcmp(argv[i], "--half-pixel") == 0) {
            const char *mode = argv[i + 1];
            if (std::strcmp(mode, "off") == 0) {
                halfPixelShift = 1.0f;
            } else if (std::strcmp(mode, "normal") == 0) {
                // The game's COREGS_HW_HALF_PIXEL=normal (backend scale_offset_normal).
                halfPixelShift = scale % 2u == 0u ? static_cast<float>(scale) + 0.2f
                                 : (scale & 2u) == 0u   ? static_cast<float>(scale) + 0.3f
                                                        : static_cast<float>(scale) + 0.1f;
            } else {
                halfPixelShift = 0.0f;  // native (the default): shift = scale
            }
        } else if (std::strcmp(argv[i], "--trace-frame") == 0) {
            traceFrame = static_cast<std::uint64_t>(std::strtoull(argv[i + 1], nullptr, 10));
        } else if (std::strcmp(argv[i], "--trace-rect") == 0) {
            int x0 = 0, y0 = 0, x1 = 0, y1 = 0;
            if (std::sscanf(argv[i + 1], "%d,%d,%d,%d", &x0, &y0, &x1, &y1) == 4) traceRect = PixelRect{x0, y0, x1, y1};
        }
    }

    std::vector<std::uint8_t> stream;
    if (!ReadFile(capturePath.c_str(), stream) || stream.empty()) {
        std::fprintf(stderr, "cannot read capture: %s\n", capturePath.c_str());
        return 2;
    }

    std::printf("capture=%s bytes=%llu magic=%.8s v2=%d\n", capturePath.c_str(),
                static_cast<unsigned long long>(stream.size()),
                stream.size() >= 8 ? reinterpret_cast<const char*>(stream.data()) : "",
                stream.size() >= 8 && std::memcmp(stream.data(), "GSSTREAM", 8) == 0 ? 1 : 0);
    HeadlessDevice headless;
    for (int i = 3; i < argc; ++i) {
        if (std::strcmp(argv[i], "--no-interlock") == 0) headless.wantInterlock = false;  // IN-345 A/B
    }
    std::string error;
    if (!headless.Create(error)) {
        std::fprintf(stderr, "skipped: %s\n", error.c_str());
        return 77;
    }
    writeback::Renderer renderer;
    gpu::VulkanHandles handles;
    handles.instance = headless.instance;
    handles.physical = headless.physical;
    handles.device = headless.device;
    handles.queue = headless.queue;
    handles.queueFamily = headless.family;
    handles.pixelInterlock = headless.interlock;
    handles.independentBlend = headless.independentBlend;
    handles.dualSourceBlend = headless.dualSourceBlend;
    handles.profile = headless.profile;  // TASK-179
    std::printf("pixel_interlock=%d scale=%u\n", headless.interlock ? 1 : 0, scale);
    renderer.SetScale(scale, halfPixelShift > 0.0f ? halfPixelShift : static_cast<float>(scale));  // 06.3 half-pixel
    if (!renderer.Init(handles, error)) {
        std::fprintf(stderr, "GPU drawer init failed: %s\n", error.c_str());
        return 1;
    }
    renderer.SetOutputSettings(outputSettings);
    renderer.SetSyncMode(cacheMode);
    for (int i = 3; i < argc; ++i) {
        if (std::strcmp(argv[i], "--no-target-textures") == 0) renderer.SetTextureFromTargets(false);  // IN-346 A/B
        if (std::strcmp(argv[i], "--no-glm") == 0) renderer.SetGlm(false);                             // IN-347 A/B
        if (std::strcmp(argv[i], "--no-cpu-sprites") == 0) renderer.SetCpuSmallSprites(false);          // IN-348 A/B
        if (std::strcmp(argv[i], "--no-gpu-clut") == 0) renderer.SetGpuClut(false);                     // IN-349 A/B
        if (std::strcmp(argv[i], "--no-glm-park") == 0) renderer.SetGlmParking(false);                  // IN-350 A/B
        if (std::strcmp(argv[i], "--no-gpu-uploads") == 0) renderer.SetGpuUploads(false);              // IN-351 A/B
        if (std::strcmp(argv[i], "--no-runs") == 0) renderer.SetRuns(false);                            // IN-352 A/B
        if (std::strcmp(argv[i], "--single-buffer") == 0) renderer.SetDoubleBuffering(false);           // IN-354 A/B
        if (std::strcmp(argv[i], "--async-present") == 0) renderer.SetAsyncPresentation(true);          // IN-355
        if (std::strcmp(argv[i], "--no-batched-writebacks") == 0) renderer.SetBatchedWritebacks(false); // IN-357
        if (std::strcmp(argv[i], "--no-decode-reuse") == 0) renderer.SetDecodeReuse(false);              // IN-369 A/B
        if (std::strcmp(argv[i], "--no-clut-defer") == 0) renderer.SetClutDefer(false);                  // IN-370 A/B
        if (std::strcmp(argv[i], "--no-tone-hle") == 0) renderer.SetToneHle(false);                      // IN-383 A/B
        if (std::strcmp(argv[i], "--tone-verify") == 0) renderer.SetVerifyShuffle(true);                 // IN-383 (09.4)
        if (std::strcmp(argv[i], "--gpu-timing") == 0) renderer.SetGpuTiming(true);                     // 12.8
        if (std::strcmp(argv[i], "--fxaa") == 0) renderer.SetFxaa(true);                                // IN-376
        // A/B: every presentation from the CPU compose at 1x (what candidate 271 did for every 480i frame, IN-381).
        if (std::strcmp(argv[i], "--no-gpu-compose") == 0) renderer.SetGpuCompose(false);
        if (std::strcmp(argv[i], "--chunks-per-submission") == 0 && i + 1 < argc) {                   // IN-358 A/B
            renderer.SetChunksPerSubmission(static_cast<std::uint32_t>(std::strtoul(argv[i + 1], nullptr, 10)));
        }
    }

    // Frame counter for the trace: EndFrame of frame N is the N-th call (0-based), as in onFrame.
    std::uint64_t currentFrame = 0;
    std::uint64_t traceBatch = 0;
    if (traceFrame != UINT64_MAX) {
        renderer.SetBatchObserver([&](const Batch& batch, const writeback::BatchTranslation& t) {
            if (currentFrame != traceFrame) return;
            const std::uint64_t index = traceBatch++;
            if (!RectsOverlap(batch.drawRect, traceRect)) return;
            const BatchState& st = batch.state;
            std::printf("trace frame=%llu batch=%llu prims=%zu verts=%zu prim=0x%llx frame=0x%llx zbuf=0x%llx "
                        "tex0=0x%llx tex1=0x%llx clamp=0x%llx alpha=0x%llx test=0x%llx scissor=0x%llx "
                        "rect=%d,%d,%d,%d draws=%zu skipped=%d\n",
                        static_cast<unsigned long long>(currentFrame), static_cast<unsigned long long>(index),
                        batch.prims.size(), batch.vertices.size(), static_cast<unsigned long long>(st.effectivePrim),
                        static_cast<unsigned long long>(st.ctx.frame), static_cast<unsigned long long>(st.ctx.zbuf),
                        static_cast<unsigned long long>(st.ctx.tex0), static_cast<unsigned long long>(st.ctx.tex1),
                        static_cast<unsigned long long>(st.ctx.clamp), static_cast<unsigned long long>(st.ctx.alpha),
                        static_cast<unsigned long long>(st.ctx.test), static_cast<unsigned long long>(st.ctx.scissor),
                        batch.drawRect.x0, batch.drawRect.y0, batch.drawRect.x1, batch.drawRect.y1, t.draws.size(),
                        t.skippedTexture ? 1 : 0);
            const std::size_t shown = batch.vertices.size() < 8 ? batch.vertices.size() : 8;
            for (std::size_t v = 0; v < shown; ++v) {
                const Vertex& x = batch.vertices[v];
                std::printf("  v%zu xy4=%d,%d z=%u rgba=%u,%u,%u,%u fog=%u stq=%.9g,%.9g,%.9g uv=%u,%u\n", v, x.x4, x.y4,
                            x.z, x.r, x.g, x.b, x.a, x.fog, x.s, x.t, x.q, x.u, x.v);
            }
        });
    }

    std::FILE* slmHashes = nullptr;
    if (!slmHashPath.empty()) {
        slmHashes = std::fopen(slmHashPath.c_str(), "w");
        if (!slmHashes) {
            std::fprintf(stderr, "cannot create %s\n", slmHashPath.c_str());
            return 2;
        }
    }

    writeback::WritebackReplayTarget target(renderer, composeKind);
    for (int i = 3; i < argc; ++i) {
        if (std::strcmp(argv[i], "--gpu-compose-check") == 0) target.SetGpuComposeCheck(true);  // IN-364
    }
    cpuref::ReplaySummary summary;
    std::uint64_t written = 0;
    std::uint64_t presentedFrames = 0, presented1xFrames = 0, presentedMinMixedBlocks = UINT64_MAX;
    std::uint64_t presentedMixedBlocks = 0, presentedBlocks = 0, presentedLowDetailFrames = 0;
    constexpr double kMinEdgeShare = 0.005;
    constexpr double kMinDetailRatio = 0.25;
    std::uint64_t presentedFlatFrames = 0, presentedEdges = 0;
    // Edge pixels of the frame's 1x picture (the replay target's compose).
    const auto nativeEdges = [](const cpuref::FrameImage& image) {
        std::uint64_t edges = 0;
        const display::Image& img = image.image;
        for (std::uint32_t y = 0; y < img.height; ++y) {
            for (std::uint32_t x = 0; x < img.width; ++x) {
                const display::Rgba8& p = img.at(x, y);
                const auto differs = [&](const display::Rgba8& q) { return p.r != q.r || p.g != q.g || p.b != q.b; };
                if ((x + 1 < img.width && differs(img.at(x + 1, y))) || (y + 1 < img.height && differs(img.at(x, y + 1)))) {
                    ++edges;
                }
            }
        }
        return edges;
    };
    std::string presented1xList;
    const auto onFrame = [&](std::uint64_t frame, const cpuref::FrameImage& image) {
        currentFrame = frame + 1;
        traceBatch = 0;
        if (slmHashes) {
            std::fprintf(slmHashes, "%llu %016llx\n", static_cast<unsigned long long>(frame),
                         static_cast<unsigned long long>(cpuref::HashLocalMemory(renderer.Slm(), 4u * 1024u * 1024u)));
        }
        if (!image.valid) return;
        if (checkPresented && scale > 1) {
            writeback::GpuFrame gpuFrame;
            std::vector<std::uint32_t> pixels;
            if (renderer.ComposeFrameGpu(gpuFrame) && renderer.ReadGpuFrame(gpuFrame, pixels) &&
                gpuFrame.width % scale == 0 && gpuFrame.height % scale == 0) {
                std::uint64_t mixed = 0;
                for (std::uint32_t by = 0; by < gpuFrame.height / scale; ++by) {
                    for (std::uint32_t bx = 0; bx < gpuFrame.width / scale; ++bx) {
                        const std::uint32_t first =
                            pixels[static_cast<std::size_t>(by * scale) * gpuFrame.width + bx * scale] & 0xFFFFFFu;
                        bool uniform = true;
                        for (std::uint32_t dy = 0; dy < scale && uniform; ++dy) {
                            for (std::uint32_t dx = 0; dx < scale && uniform; ++dx) {
                                uniform = (pixels[static_cast<std::size_t>(by * scale + dy) * gpuFrame.width +
                                                  bx * scale + dx] & 0xFFFFFFu) == first;
                            }
                        }
                        if (!uniform) ++mixed;
                    }
                }
                ++presentedFrames;
                presentedMixedBlocks += mixed;
                const std::uint64_t blocks =
                    static_cast<std::uint64_t>(gpuFrame.width / scale) * (gpuFrame.height / scale);
                presentedBlocks += blocks;
                const std::uint64_t edges = nativeEdges(image);
                const std::uint64_t nativePixels =
                    static_cast<std::uint64_t>(image.image.width) * image.image.height;
                if (static_cast<double>(edges) < kMinEdgeShare * static_cast<double>(nativePixels)) {
                    ++presentedFlatFrames;
                } else {
                    presentedEdges += edges;
                    static const bool detailTrace = std::getenv("COREGS_REPLAY_DETAIL_TRACE") != nullptr;  // per frame
                    if (detailTrace) {
                        std::printf("detail frame=%llu mixed_per_edge=%.3f\n", static_cast<unsigned long long>(frame),
                                    static_cast<double>(mixed) / static_cast<double>(edges));
                    }
                    if (static_cast<double>(mixed) < kMinDetailRatio * static_cast<double>(edges)) ++presentedLowDetailFrames;
                }
                if (mixed < presentedMinMixedBlocks) presentedMinMixedBlocks = mixed;
                if (mixed == 0) {
                    ++presented1xFrames;
                    if (presented1xList.size() < 200) presented1xList += " " + std::to_string(frame);
                }
            } else if (!gpuFrame.valid) {
                // IN-381: no GPU compose this VBlank (before it: every 480i menu frame): the window shows the CPU
                // presentation compose at 1x, blown up by the present. It counts as a 1x and low-detail frame.
                ++presentedFrames;
                ++presented1xFrames;
                const std::uint64_t edges = nativeEdges(image);
                if (static_cast<double>(edges) <
                    kMinEdgeShare * static_cast<double>(image.image.width) * image.image.height) {
                    ++presentedFlatFrames;
                } else {
                    ++presentedLowDetailFrames;
                }
                presentedMinMixedBlocks = 0;
                if (presented1xList.size() < 200) presented1xList += " " + std::to_string(frame) + "(cpu)";
            }
        }
        if (every > 1 && frame % every != 0) return;
        if (!upscaledDir.empty()) {
            writeback::GpuFrame gpuFrame;
            std::vector<std::uint32_t> pixels;
            if (renderer.ComposeFrameGpu(gpuFrame) && renderer.ReadGpuFrame(gpuFrame, pixels)) {
                char upscaledName[64];
                std::snprintf(upscaledName, sizeof(upscaledName), "frame_%05llu.ppm", static_cast<unsigned long long>(frame));
                WritePackedPpm(upscaledDir + "\\" + upscaledName, gpuFrame.width, gpuFrame.height, pixels);
            }
        }
        char name[64];
        std::snprintf(name, sizeof(name), "frame_%05llu.ppm", static_cast<unsigned long long>(frame));
        if (WritePpm(outDir + "\\" + name, image)) ++written;
        if (frame == slmFrame) {
            char slmName[64];
            std::snprintf(slmName, sizeof(slmName), "slm_%05llu.bin", static_cast<unsigned long long>(frame));
            std::FILE* slm = std::fopen((outDir + "\\" + slmName).c_str(), "wb");
            if (slm) {
                std::fwrite(renderer.Slm(), 1, 4u * 1024u * 1024u, slm);
                std::fclose(slm);
            }
        }
    };
    const bool ok = cpuref::ReplayCapture(stream, target, summary, onFrame, error);

    if (slmHashes) std::fclose(slmHashes);
    std::printf("records=%llu gif_packets=%llu gif_bytes=%llu privileged=%llu fifo_reads=%llu frames=%llu "
                "images=%llu rejected_gif=%llu resets=%llu\n",
                static_cast<unsigned long long>(summary.records),
                static_cast<unsigned long long>(summary.gifPackets),
                static_cast<unsigned long long>(summary.gifBytes),
                static_cast<unsigned long long>(summary.privileged),
                static_cast<unsigned long long>(summary.fifoReads),
                static_cast<unsigned long long>(summary.frames),
                static_cast<unsigned long long>(summary.framesWithImage),
                static_cast<unsigned long long>(summary.rejectedGifPackets),
                static_cast<unsigned long long>(summary.resets));
    const writeback::Counters& counters = renderer.GetCounters();
    std::printf("batches=%llu primitives=%llu skipped_texture=%llu empty=%llu textures=%llu gpu_draws=%llu "
                "writeback_pixels=%llu writeback_depth_words=%llu fifo_overruns=%llu failed=%d\n",
                static_cast<unsigned long long>(counters.batches),
                static_cast<unsigned long long>(counters.primitives),
                static_cast<unsigned long long>(counters.primitivesSkippedTexture),
                static_cast<unsigned long long>(counters.batchesEmpty),
                static_cast<unsigned long long>(counters.texturesDecoded),
                static_cast<unsigned long long>(counters.gpuDraws),
                static_cast<unsigned long long>(counters.writebackPixels),
                static_cast<unsigned long long>(counters.writebackDepthWords),
                static_cast<unsigned long long>(counters.fifoOverruns), renderer.Failed() ? 1 : 0);
    std::printf("writeback_reasons other=%llu texture=%llu clut=%llu transfer=%llu display=%llu eviction=%llu "
                "load_overlap=%llu fallback=%llu cpu_sprite=%llu\n",
                static_cast<unsigned long long>(counters.cacheWritebackByReason[0]),
                static_cast<unsigned long long>(counters.cacheWritebackByReason[1]),
                static_cast<unsigned long long>(counters.cacheWritebackByReason[2]),
                static_cast<unsigned long long>(counters.cacheWritebackByReason[3]),
                static_cast<unsigned long long>(counters.cacheWritebackByReason[4]),
                static_cast<unsigned long long>(counters.cacheWritebackByReason[5]),
                static_cast<unsigned long long>(counters.cacheWritebackByReason[6]),
                static_cast<unsigned long long>(counters.cacheWritebackByReason[7]),
                static_cast<unsigned long long>(counters.cacheWritebackByReason[8]));
    {
        const gpu::GpuStats& gpuStats = renderer.GpuCounters();
        std::printf("gpu executions=%llu vk_draws=%llu barriers=%llu barriers_elided=%llu render_passes=%llu decodes_reused=%llu clut_dispatches=%llu clut_materialisations=%llu decodes_glm_palette=%llu submissions=%llu runs=%llu run_chunks=%llu\n",
                    static_cast<unsigned long long>(gpuStats.executions),
                    static_cast<unsigned long long>(gpuStats.draws),
                    static_cast<unsigned long long>(gpuStats.barriers),
                    static_cast<unsigned long long>(gpuStats.barriersElided),
                    static_cast<unsigned long long>(gpuStats.renderPasses),
                    static_cast<unsigned long long>(gpuStats.decodesReused),
                    static_cast<unsigned long long>(gpuStats.clutDispatches),
                    static_cast<unsigned long long>(gpuStats.clutMaterialisations),
                    static_cast<unsigned long long>(gpuStats.decodesGlmPalette),
                    static_cast<unsigned long long>(gpuStats.submissions),
                    static_cast<unsigned long long>(gpuStats.runs),
                    static_cast<unsigned long long>(gpuStats.runChunks));
        const std::uint64_t frameCount = renderer.TimedFrames();
        if (gpuStats.gpuBusyNs != 0 && frameCount != 0) {
            const double n = static_cast<double>(frameCount) * 1e6;
            std::printf("gpu_ms_per_frame busy=%.2f pass=%.2f decode=%.2f clut=%.2f swizzle=%.2f upload=%.2f download=%.2f "
                        "scaled=%.2f refresh=%.2f\n",
                        gpuStats.gpuBusyNs / n, gpuStats.gpuNs[0] / n, gpuStats.gpuNs[1] / n, gpuStats.gpuNs[2] / n,
                        gpuStats.gpuNs[3] / n, gpuStats.gpuNs[4] / n, gpuStats.gpuNs[5] / n, gpuStats.gpuNs[6] / n,
                        gpuStats.gpuNs[7] / n);
        }
    }
    std::printf("textures_cached=%llu textures_from_targets=%llu textures_from_glm=%llu glm_uploads=%llu glm_swizzles=%llu "
                "cpu_sprite_batches=%llu gpu_clut_loads=%llu glm_parks=%llu glm_slot_loads=%llu glm_downloads=%llu "
                "glm_pages_downloaded=%llu glm_upload_merges=%llu glm_host_uploads=%llu ",
                static_cast<unsigned long long>(counters.texturesCached),
                static_cast<unsigned long long>(counters.texturesFromTargets),
                static_cast<unsigned long long>(counters.texturesFromGlm),
                static_cast<unsigned long long>(counters.glmPagesUploaded),
                static_cast<unsigned long long>(counters.glmSwizzles),
                static_cast<unsigned long long>(counters.cpuSpriteBatches),
                static_cast<unsigned long long>(counters.gpuClutLoads),
                static_cast<unsigned long long>(counters.glmParks),
                static_cast<unsigned long long>(counters.glmSlotLoads),
                static_cast<unsigned long long>(counters.glmDownloads),
                static_cast<unsigned long long>(counters.glmPagesDownloaded),
                static_cast<unsigned long long>(counters.glmUploadMerges),
                static_cast<unsigned long long>(counters.glmHostUploads));
    std::printf("tone_strip_pages=%llu tone_strip_batches=%llu tone_strip_deferred=%llu tone_verify_pages=%llu "
                "tone_verify_mismatch_bytes=%llu tone_clut_held=%llu tone_clut_dropped=%llu\n",
                static_cast<unsigned long long>(counters.toneStripPages),
                static_cast<unsigned long long>(counters.toneStripBatches),
                static_cast<unsigned long long>(counters.toneStripBatchesDeferred),
                static_cast<unsigned long long>(counters.toneVerifyPages),
                static_cast<unsigned long long>(counters.toneVerifyMismatchBytes),
                static_cast<unsigned long long>(counters.toneClutLoadsHeld),
                static_cast<unsigned long long>(counters.toneClutLoadsDropped));
    std::printf("sync_mode=%s resident_batches=%llu fallback_batches=%llu loads=%llu load_words=%llu writebacks=%llu "
                "writeback_words=%llu\n",
                cacheMode ? "cache" : "writeback", static_cast<unsigned long long>(counters.cacheResidentBatches),
                static_cast<unsigned long long>(counters.cacheFallbackBatches),
                static_cast<unsigned long long>(counters.cacheLoads),
                static_cast<unsigned long long>(counters.cacheLoadWords),
                static_cast<unsigned long long>(counters.cacheWritebacks),
                static_cast<unsigned long long>(counters.cacheWritebackWords));
    {
        const writeback::PassTimings& totals = renderer.TimingTotals();
        const double frames = renderer.TimedFrames() ? static_cast<double>(renderer.TimedFrames()) : 1.0;
        std::printf("per_frame_ms clut=%.2f compose=%.2f gif=%.2f batch=%.2f translate=%.2f execute=%.2f upload=%.2f writeback=%.2f load=%.2f "
                    "decode=%.2f\n",
                    counters.clutUs / frames / 1000.0, counters.composeUs / frames / 1000.0,
                    counters.gifUs / frames / 1000.0,
                    counters.batchUs / frames / 1000.0,
                    totals.translateUs / frames / 1000.0,
                    totals.executeUs / frames / 1000.0, counters.uploadUs / frames / 1000.0,
                    counters.cacheWritebackUs / frames / 1000.0, counters.cacheLoadUs / frames / 1000.0,
                    counters.textureDecodeUs / frames / 1000.0);
    }
    std::printf("images_written=%llu\n", static_cast<unsigned long long>(written));
    {
        const gpu::GpuStats& gpuStats = renderer.GpuCounters();
        std::printf("scale=%u scaled_draws=%llu scaled_passes=%llu scaled_refreshes=%llu scaled_refresh_pixels=%llu "
                    "scaled_composes=%llu circuits_from_slots=%llu circuits_from_glm=%llu circuits_from_shadows=%llu "
                    "shadow_saves=%llu tone_runs=%llu tone_carries=%llu scaled_image_mb=%.2f scaled_image_grows=%llu "
                    "scaled_draws_clipped=%llu\n",
                    renderer.Scale(), static_cast<unsigned long long>(gpuStats.scaledDraws),
                    static_cast<unsigned long long>(gpuStats.scaledPasses),
                    static_cast<unsigned long long>(gpuStats.scaledRefreshes),
                    static_cast<unsigned long long>(gpuStats.scaledRefreshPixels),
                    static_cast<unsigned long long>(counters.scaledComposedFrames),
                    static_cast<unsigned long long>(counters.scaledCircuitsFromSlots),
                    static_cast<unsigned long long>(counters.scaledCircuitsFromGlm),
                    static_cast<unsigned long long>(counters.scaledCircuitsFromShadows),
                    static_cast<unsigned long long>(counters.scaledShadowSaves),
                    static_cast<unsigned long long>(counters.toneRuns),
                    static_cast<unsigned long long>(counters.toneCarries),
                    static_cast<double>(gpuStats.scaledImageBytes) / 1e6,
                    static_cast<unsigned long long>(gpuStats.scaledImageGrows),
                    static_cast<unsigned long long>(gpuStats.scaledDrawsClipped));
    }
    if (checkPresented && scale > 1) {
        std::printf("presented_frames=%llu presented_1x_frames=%llu min_mixed_blocks=%llu mean_mixed_blocks=%.1f\n",
                    static_cast<unsigned long long>(presentedFrames),
                    static_cast<unsigned long long>(presented1xFrames),
                    static_cast<unsigned long long>(presentedFrames ? presentedMinMixedBlocks : 0),
                    presentedFrames ? static_cast<double>(presentedMixedBlocks) / presentedFrames : 0.0);
        if (presented1xFrames) std::printf("presented_1x_list%s\n", presented1xList.c_str());
        std::printf("presented_low_detail_frames=%llu flat_frames=%llu mean_mixed_share=%.3f mixed_per_edge=%.3f "
                    "(low detail: mixed blocks below %.2f x the 1x edge pixels)\n",
                    static_cast<unsigned long long>(presentedLowDetailFrames),
                    static_cast<unsigned long long>(presentedFlatFrames),
                    presentedBlocks ? static_cast<double>(presentedMixedBlocks) / presentedBlocks : 0.0,
                    presentedEdges ? static_cast<double>(presentedMixedBlocks) / presentedEdges : 0.0, kMinDetailRatio);
    }
    std::printf("gpu_compose_equal=%llu gpu_compose_different=%llu\n",
                static_cast<unsigned long long>(target.GpuComposeEqual()),
                static_cast<unsigned long long>(target.GpuComposeDifferent()));
    std::printf("fifo_reads=%llu fifo_hash=%016llx\n", static_cast<unsigned long long>(target.FifoReads()),
                static_cast<unsigned long long>(target.FifoHash()));
    if (renderer.Failed()) {
        std::fprintf(stderr, "GPU write-back renderer failed: %s\n", renderer.LastError().c_str());
        return 1;
    }
    if (!ok) {
        std::fprintf(stderr, "replay failed: %s\n", error.c_str());
        return 1;
    }
    return 0;
}
