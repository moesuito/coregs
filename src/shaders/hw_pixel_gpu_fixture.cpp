// SPDX-License-Identifier: Apache-2.0
//
// Bit-exactness fixture for the HWR GPU pixel path (TASK-027, M4). Copyright (c) 2026 GT4Recomp contributors.
//
// Runs the embedded SPIR-V compute kernels (hw_blend_exact.comp, hw_pixel_exact.comp) on a headless
// Vulkan compute queue and compares every result bit for bit with the CPU reference in
// src/pixel. No window, no swapchain. Written from the CoreGS specification (chapters 01.5,
// 03.4, 08), clean room (decision 0069).
//
// Sweeps: the 81-mode blend sweep of the pixel fixture (42,467,328 cases: mode x COLCLAMP x Cs x Cd x
// As x Ad), TFX and TEXA sweeps, fog, the alpha-test outcome sweep, and a seeded random sweep over
// every pixel state field (DATE, depth, FBMSK, dither, PABE, FBA, all three frame formats).
//
// Device selection mirrors the M0 backend (hw_gs_backend.cpp, chapter 12.6): a Vulkan 1.2 device with a
// compute queue, discrete GPUs preferred. Missing device: exit 77 (CTest reports Skipped).
//
// Usage: hw_pixel_gpu_fixture [test-name ...]. Names: hw_gpu_blend_sweep, hw_gpu_pixel_directed,
// hw_gpu_pixel_random. With no arguments every test runs.

#include "hw_gs_pixel.hpp"
#include "hw_shader_blobs.hpp"

#include <vulkan/vulkan.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <string>
#include <vector>

using namespace coregs::pixel;

namespace {

constexpr int kSkipCode = 77;
constexpr std::uint32_t kWorkgroupSize = 64;
constexpr std::uint64_t kBlendChunk = 1ull << 22;  // cases per blend dispatch
constexpr std::uint64_t kPixelChunk = 1ull << 16;  // records per pixel dispatch
constexpr int kReportLimit = 8;

// ---- Case record: mirrors HwCase in hw_pixel_exact.comp (50 uint32 words, this order) ---------------

struct GpuPixelCase {
    std::uint32_t format, fbmsk, fba, dither, dimx_lo, dimx_hi;
    std::uint32_t ate, atst, afail, aref;
    std::uint32_t date_on, datm;
    std::uint32_t ztest, ztst, zmsk, zformat;
    std::uint32_t blend_on, pabe, blend_a, blend_b, blend_c, blend_d, fix, colclamp;
    std::uint32_t tex_on, tfx, tcc, texel_kind, texel_raw, ta0, ta1, aem;
    std::uint32_t fog_on, fog, fog_r, fog_g, fog_b;
    std::uint32_t x, y, fr, fg, fb, fa, fz;
    std::uint32_t frame_old, depth_old;
    std::uint32_t frame_written, frame_stored, depth_written, depth_stored;
};
static_assert(sizeof(GpuPixelCase) == 50 * 4, "record layout must match HwCase in hw_pixel_exact.comp");

// ---- Deterministic generator (splitmix64), one stream per case index ---------------------------------

std::uint64_t mix64(std::uint64_t x) {
    x += 0x9E3779B97F4A7C15ull;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
    return x ^ (x >> 31);
}

struct Rng {
    std::uint64_t state;
    explicit Rng(std::uint64_t seed) : state(mix64(seed)) {}
    std::uint32_t next32() {
        state = mix64(state);
        return static_cast<std::uint32_t>(state);
    }
    std::uint32_t below(std::uint32_t n) { return next32() % n; }
    std::uint32_t byte() { return next32() & 0xFFu; }
    // Channel values: about a quarter are the edge values where clamp and wrap decisions change.
    std::uint32_t channel() {
        static const std::uint32_t edges[] = {0, 1, 127, 128, 254, 255};
        return below(4) == 0 ? edges[below(6)] : byte();
    }
};

// ---- Vulkan bootstrap ----------------------------------------------------------------------------------

struct Vulkan {
    VkInstance instance = VK_NULL_HANDLE;
    VkPhysicalDevice physical = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    VkQueue queue = VK_NULL_HANDLE;
    std::uint32_t queue_family = 0;
    VkCommandPool pool = VK_NULL_HANDLE;
    VkCommandBuffer command = VK_NULL_HANDLE;
    VkFence fence = VK_NULL_HANDLE;
    std::string device_name;

    ~Vulkan() { destroy(); }

    void destroy() {
        if (device) {
            vkDeviceWaitIdle(device);
            if (fence) vkDestroyFence(device, fence, nullptr);
            if (pool) vkDestroyCommandPool(device, pool, nullptr);
            vkDestroyDevice(device, nullptr);
        }
        if (instance) vkDestroyInstance(instance, nullptr);
        instance = VK_NULL_HANDLE;
        device = VK_NULL_HANDLE;
        pool = VK_NULL_HANDLE;
        fence = VK_NULL_HANDLE;
    }

    // Returns 0 on success, kSkipCode when no compute-capable Vulkan 1.2 device exists, 2 on error.
    int create(std::string& error) {
        VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
        app.pApplicationName = "hw_pixel_gpu_fixture";
        app.apiVersion = VK_API_VERSION_1_2;
        VkInstanceCreateInfo instance_info{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
        instance_info.pApplicationInfo = &app;
        const VkResult instance_result = vkCreateInstance(&instance_info, nullptr, &instance);
        if (instance_result != VK_SUCCESS) {
            // A machine with no Vulkan driver installed fails here, before any device can be enumerated (a CI
            // runner without a GPU, for example). That is the same "no device" the fixtures skip on, and the
            // reason is printed; every other result stays an error.
            if (instance_result == VK_ERROR_INCOMPATIBLE_DRIVER) {
                error = "no Vulkan driver installed (vkCreateInstance returned VK_ERROR_INCOMPATIBLE_DRIVER)";
                return kSkipCode;
            }
            error = "vkCreateInstance failed: VkResult " + std::to_string(static_cast<int>(instance_result));
            return 2;
        }

        std::uint32_t count = 0;
        vkEnumeratePhysicalDevices(instance, &count, nullptr);
        std::vector<VkPhysicalDevice> devices(count);
        if (count > 0) vkEnumeratePhysicalDevices(instance, &count, devices.data());

        int best_score = -1;
        for (VkPhysicalDevice candidate : devices) {
            VkPhysicalDeviceProperties props{};
            vkGetPhysicalDeviceProperties(candidate, &props);
            if (props.apiVersion < VK_API_VERSION_1_2) continue;
            std::uint32_t family_count = 0;
            vkGetPhysicalDeviceQueueFamilyProperties(candidate, &family_count, nullptr);
            std::vector<VkQueueFamilyProperties> families(family_count);
            vkGetPhysicalDeviceQueueFamilyProperties(candidate, &family_count, families.data());
            for (std::uint32_t i = 0; i < family_count; ++i) {
                if ((families[i].queueFlags & VK_QUEUE_COMPUTE_BIT) == 0) continue;
                const int score = props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU ? 2 : 1;
                if (score > best_score) {
                    best_score = score;
                    physical = candidate;
                    queue_family = i;
                    device_name = props.deviceName;
                }
                break;
            }
        }
        if (physical == VK_NULL_HANDLE) {
            error = "no Vulkan 1.2 device with a compute queue";
            return kSkipCode;
        }

        const float priority = 1.0f;
        VkDeviceQueueCreateInfo queue_info{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
        queue_info.queueFamilyIndex = queue_family;
        queue_info.queueCount = 1;
        queue_info.pQueuePriorities = &priority;
        VkDeviceCreateInfo device_info{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
        device_info.queueCreateInfoCount = 1;
        device_info.pQueueCreateInfos = &queue_info;
        if (vkCreateDevice(physical, &device_info, nullptr, &device) != VK_SUCCESS) {
            error = "vkCreateDevice failed";
            return 2;
        }
        vkGetDeviceQueue(device, queue_family, 0, &queue);

        VkCommandPoolCreateInfo pool_info{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
        pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        pool_info.queueFamilyIndex = queue_family;
        vkCreateCommandPool(device, &pool_info, nullptr, &pool);
        VkCommandBufferAllocateInfo alloc{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        alloc.commandPool = pool;
        alloc.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        alloc.commandBufferCount = 1;
        vkAllocateCommandBuffers(device, &alloc, &command);
        VkFenceCreateInfo fence_info{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
        vkCreateFence(device, &fence_info, nullptr, &fence);
        return 0;
    }

    std::int32_t memory_type(std::uint32_t type_bits, VkMemoryPropertyFlags wanted) const {
        VkPhysicalDeviceMemoryProperties memory{};
        vkGetPhysicalDeviceMemoryProperties(physical, &memory);
        for (std::uint32_t i = 0; i < memory.memoryTypeCount; ++i) {
            const bool allowed = (type_bits & (1u << i)) != 0;
            if (allowed && (memory.memoryTypes[i].propertyFlags & wanted) == wanted) {
                return static_cast<std::int32_t>(i);
            }
        }
        return -1;
    }
};

// One compute pipeline with a single storage buffer at set 0 binding 0 and a push-constant block.
struct Kernel {
    Vulkan* vk = nullptr;
    VkShaderModule module = VK_NULL_HANDLE;
    VkDescriptorSetLayout set_layout = VK_NULL_HANDLE;
    VkPipelineLayout layout = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
    VkDescriptorPool descriptor_pool = VK_NULL_HANDLE;
    VkDescriptorSet set = VK_NULL_HANDLE;
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    void* mapped = nullptr;
    VkDeviceSize bytes = 0;
    std::uint32_t push_bytes = 0;

    ~Kernel() { destroy(); }

    bool create(Vulkan& device, const std::uint32_t* words, std::size_t word_count, VkDeviceSize buffer_bytes,
                std::uint32_t push_size, std::string& error) {
        vk = &device;
        push_bytes = push_size;
        bytes = buffer_bytes;
        VkShaderModuleCreateInfo module_info{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
        module_info.codeSize = word_count * sizeof(std::uint32_t);
        module_info.pCode = words;
        if (vkCreateShaderModule(device.device, &module_info, nullptr, &module) != VK_SUCCESS) {
            error = "vkCreateShaderModule failed";
            return false;
        }

        VkDescriptorSetLayoutBinding binding{};
        binding.binding = 0;
        binding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        binding.descriptorCount = 1;
        binding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        VkDescriptorSetLayoutCreateInfo set_info{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        set_info.bindingCount = 1;
        set_info.pBindings = &binding;
        vkCreateDescriptorSetLayout(device.device, &set_info, nullptr, &set_layout);

        VkPushConstantRange range{};
        range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        range.size = push_size;
        VkPipelineLayoutCreateInfo layout_info{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        layout_info.setLayoutCount = 1;
        layout_info.pSetLayouts = &set_layout;
        layout_info.pushConstantRangeCount = 1;
        layout_info.pPushConstantRanges = &range;
        vkCreatePipelineLayout(device.device, &layout_info, nullptr, &layout);

        VkPipelineShaderStageCreateInfo stage{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
        stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        stage.module = module;
        stage.pName = "main";
        VkComputePipelineCreateInfo pipeline_info{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        pipeline_info.stage = stage;
        pipeline_info.layout = layout;
        if (vkCreateComputePipelines(device.device, VK_NULL_HANDLE, 1, &pipeline_info, nullptr, &pipeline) !=
            VK_SUCCESS) {
            error = "vkCreateComputePipelines failed";
            return false;
        }

        VkBufferCreateInfo buffer_info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        buffer_info.size = buffer_bytes;
        buffer_info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
        buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        if (vkCreateBuffer(device.device, &buffer_info, nullptr, &buffer) != VK_SUCCESS) {
            error = "vkCreateBuffer failed";
            return false;
        }
        VkMemoryRequirements requirements{};
        vkGetBufferMemoryRequirements(device.device, buffer, &requirements);
        const std::int32_t type =
            device.memory_type(requirements.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        if (type < 0) {
            error = "no host-visible coherent memory type";
            return false;
        }
        VkMemoryAllocateInfo alloc{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        alloc.allocationSize = requirements.size;
        alloc.memoryTypeIndex = static_cast<std::uint32_t>(type);
        if (vkAllocateMemory(device.device, &alloc, nullptr, &memory) != VK_SUCCESS ||
            vkBindBufferMemory(device.device, buffer, memory, 0) != VK_SUCCESS ||
            vkMapMemory(device.device, memory, 0, VK_WHOLE_SIZE, 0, &mapped) != VK_SUCCESS) {
            error = "buffer allocation or mapping failed";
            return false;
        }

        VkDescriptorPoolSize pool_size{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1};
        VkDescriptorPoolCreateInfo pool_info{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        pool_info.maxSets = 1;
        pool_info.poolSizeCount = 1;
        pool_info.pPoolSizes = &pool_size;
        vkCreateDescriptorPool(device.device, &pool_info, nullptr, &descriptor_pool);
        VkDescriptorSetAllocateInfo set_alloc{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        set_alloc.descriptorPool = descriptor_pool;
        set_alloc.descriptorSetCount = 1;
        set_alloc.pSetLayouts = &set_layout;
        vkAllocateDescriptorSets(device.device, &set_alloc, &set);
        VkDescriptorBufferInfo buffer_binding{buffer, 0, VK_WHOLE_SIZE};
        VkWriteDescriptorSet write{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        write.dstSet = set;
        write.dstBinding = 0;
        write.descriptorCount = 1;
        write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        write.pBufferInfo = &buffer_binding;
        vkUpdateDescriptorSets(device.device, 1, &write, 0, nullptr);
        return true;
    }

    // Records and runs one dispatch, waits for it, and makes its writes visible to the host.
    bool run(std::uint32_t item_count, const void* push, std::string& error) {
        VkDevice device = vk->device;
        VkCommandBuffer cmd = vk->command;
        vkResetCommandBuffer(cmd, 0);
        VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkBeginCommandBuffer(cmd, &begin);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, layout, 0, 1, &set, 0, nullptr);
        vkCmdPushConstants(cmd, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, push_bytes, push);
        vkCmdDispatch(cmd, (item_count + kWorkgroupSize - 1) / kWorkgroupSize, 1, 1);
        VkBufferMemoryBarrier barrier{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
        barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.buffer = buffer;
        barrier.size = VK_WHOLE_SIZE;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 0, nullptr, 1,
                             &barrier, 0, nullptr);
        if (vkEndCommandBuffer(cmd) != VK_SUCCESS) {
            error = "vkEndCommandBuffer failed";
            return false;
        }
        VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        submit.commandBufferCount = 1;
        submit.pCommandBuffers = &cmd;
        vkResetFences(device, 1, &vk->fence);
        if (vkQueueSubmit(vk->queue, 1, &submit, vk->fence) != VK_SUCCESS ||
            vkWaitForFences(device, 1, &vk->fence, VK_TRUE, UINT64_MAX) != VK_SUCCESS) {
            error = "dispatch failed (device lost?)";
            return false;
        }
        return true;
    }

    void destroy() {
        if (!vk || !vk->device) return;
        VkDevice device = vk->device;
        if (mapped) vkUnmapMemory(device, memory);
        if (buffer) vkDestroyBuffer(device, buffer, nullptr);
        if (memory) vkFreeMemory(device, memory, nullptr);
        if (descriptor_pool) vkDestroyDescriptorPool(device, descriptor_pool, nullptr);
        if (pipeline) vkDestroyPipeline(device, pipeline, nullptr);
        if (layout) vkDestroyPipelineLayout(device, layout, nullptr);
        if (set_layout) vkDestroyDescriptorSetLayout(device, set_layout, nullptr);
        if (module) vkDestroyShaderModule(device, module, nullptr);
        mapped = nullptr;
        buffer = VK_NULL_HANDLE;
        memory = VK_NULL_HANDLE;
        descriptor_pool = VK_NULL_HANDLE;
        pipeline = VK_NULL_HANDLE;
        layout = VK_NULL_HANDLE;
        set_layout = VK_NULL_HANDLE;
        module = VK_NULL_HANDLE;
    }
};

// ---- Blend sweep (42,467,328 cases, the 81-mode sweep of the pixel fixture) -----------------------------

// Case index layout: ((((mode * 2 + clamp) * 256 + cs) * 256 + cd) * 2 + ai) * 2 + di.
constexpr std::uint64_t kBlendCases = 81ull * 2 * 256 * 256 * 2 * 2;

int test_blend_sweep(Vulkan& vk) {
    Kernel kernel;
    std::string error;
    if (!kernel.create(vk, coregs::shaders::hw_blend_exact_words, coregs::shaders::hw_blend_exact_word_count,
                       kBlendChunk * sizeof(std::uint32_t), 8, error)) {
        std::fprintf(stderr, "hw_blend_exact: %s\n", error.c_str());
        return 1;
    }
    std::uint64_t mismatches = 0;
    int reported = 0;
    for (std::uint64_t base = 0; base < kBlendCases; base += kBlendChunk) {
        const std::uint64_t count = std::min(kBlendChunk, kBlendCases - base);
        const std::uint32_t push[2] = {static_cast<std::uint32_t>(base), static_cast<std::uint32_t>(count)};
        if (!kernel.run(static_cast<std::uint32_t>(count), push, error)) {
            std::fprintf(stderr, "hw_blend_exact: %s\n", error.c_str());
            return 1;
        }
        const std::uint32_t* got = static_cast<const std::uint32_t*>(kernel.mapped);
        for (std::uint64_t i = 0; i < count; ++i) {
            const std::uint64_t index = base + i;
            const int di = static_cast<int>(index & 1);
            const int ai = static_cast<int>((index >> 1) & 1);
            const int cd = static_cast<int>((index >> 2) & 255);
            const int cs = static_cast<int>((index >> 10) & 255);
            const bool clamp = ((index >> 18) & 1) != 0;
            const int mode = static_cast<int>(index >> 19);

            BlendState state;
            state.enabled = true;
            state.a = static_cast<BlendSource>(mode % 3);
            state.b = static_cast<BlendSource>((mode / 3) % 3);
            state.c = static_cast<BlendFactor>((mode / 9) % 3);
            state.d = static_cast<BlendSource>(mode / 27);
            state.fix = 0x81;
            state.colour_clamp = clamp;
            const std::int32_t as = ai ? 0xFF : 0x40;
            const std::int32_t ad = di ? 0x80 : 0x00;
            const Colour blended = blend_unclamped(state, Colour{cs, cs, cs, as}, Colour{cd, cd, cd, ad}, as, ad);
            const std::uint32_t expected = static_cast<std::uint32_t>(colour_clamp_or_wrap(blended.r, clamp));
            if (got[i] != expected) {
                ++mismatches;
                if (reported < kReportLimit) {
                    ++reported;
                    std::fprintf(stderr, "blend index %llu mode %d clamp %d cs %d cd %d as %d ad %d: gpu %u cpu %u\n",
                                 static_cast<unsigned long long>(index), mode, clamp ? 1 : 0, cs, cd, as, ad,
                                 got[i], expected);
                }
            }
        }
    }
    std::printf("%-24s %llu cases, %llu mismatches (%s)\n", "hw_gpu_blend_sweep",
                static_cast<unsigned long long>(kBlendCases), static_cast<unsigned long long>(mismatches),
                vk.device_name.c_str());
    return mismatches == 0 ? 0 : 1;
}

// ---- Pixel records --------------------------------------------------------------------------------------

// The CPU composition of the GPU path: texel, TFX, fog, then process_pixel. Mirrors hw_shade_fragment and
// hw_process_pixel in hw_pixel_core.glsl.
PixelOutcome expected_outcome(const GpuPixelCase& c) {
    Colour frag{static_cast<std::int32_t>(c.fr), static_cast<std::int32_t>(c.fg), static_cast<std::int32_t>(c.fb),
                static_cast<std::int32_t>(c.fa)};
    Colour shaded = frag;
    if (c.tex_on != 0) {
        Colour texel;
        const std::uint32_t raw = c.texel_raw;
        if (c.texel_kind == 1) {
            texel = expand_texel16(static_cast<std::uint16_t>(raw), static_cast<std::uint8_t>(c.ta0),
                                   static_cast<std::uint8_t>(c.ta1), c.aem != 0);
        } else if (c.texel_kind == 2) {
            texel = expand_texel24(raw, static_cast<std::uint8_t>(c.ta0), c.aem != 0);
        } else {
            texel = Colour{static_cast<std::int32_t>(raw & 0xFF), static_cast<std::int32_t>((raw >> 8) & 0xFF),
                           static_cast<std::int32_t>((raw >> 16) & 0xFF), static_cast<std::int32_t>(raw >> 24)};
        }
        shaded = texture_function(static_cast<Tfx>(c.tfx), c.tcc != 0, texel, frag);
    }
    if (c.fog_on != 0) {
        shaded = apply_fog(shaded, static_cast<std::int32_t>(c.fog),
                           Colour{static_cast<std::int32_t>(c.fog_r), static_cast<std::int32_t>(c.fog_g),
                                  static_cast<std::int32_t>(c.fog_b), 0});
    }

    PixelState state;
    state.frame.format = static_cast<FrameFormat>(c.format);
    state.frame.fbmsk = c.fbmsk;
    state.frame.fba = c.fba != 0;
    state.frame.dither = c.dither != 0;
    state.frame.dimx = (static_cast<std::uint64_t>(c.dimx_hi) << 32) | c.dimx_lo;
    state.alpha_test.enabled = c.ate != 0;
    state.alpha_test.compare = static_cast<AlphaCompare>(c.atst);
    state.alpha_test.fail = static_cast<AlphaFail>(c.afail);
    state.alpha_test.reference = static_cast<std::uint8_t>(c.aref);
    state.date.enabled = c.date_on != 0;
    state.date.expected_msb = c.datm != 0;
    state.depth.test_enabled = c.ztest != 0;
    state.depth.compare = static_cast<DepthCompare>(c.ztst);
    state.depth.write_masked = c.zmsk != 0;
    state.z_format = static_cast<ZFormat>(c.zformat);
    state.blend.enabled = c.blend_on != 0;
    state.blend.pabe = c.pabe != 0;
    state.blend.a = static_cast<BlendSource>(c.blend_a);
    state.blend.b = static_cast<BlendSource>(c.blend_b);
    state.blend.d = static_cast<BlendSource>(c.blend_d);
    state.blend.c = static_cast<BlendFactor>(c.blend_c);
    state.blend.fix = static_cast<std::uint8_t>(c.fix);
    state.blend.colour_clamp = c.colclamp != 0;

    Fragment fragment;
    fragment.x = static_cast<std::int32_t>(c.x);
    fragment.y = static_cast<std::int32_t>(c.y);
    fragment.colour = shaded;
    fragment.z = c.fz;
    return process_pixel(state, fragment, c.frame_old, c.depth_old);
}

// Every field at the value that makes the path a plain CT32 write with everything disabled.
GpuPixelCase plain_case() {
    GpuPixelCase c{};
    c.format = 0;  // CT32
    c.frame_old = 0x0A0B0C0Du;
    c.depth_old = 0x00000001u;
    c.x = 3;
    c.y = 5;
    c.fr = c.fg = c.fb = c.fa = 0x80;
    c.fz = 0x12345678u;
    return c;
}

using CaseMaker = std::function<GpuPixelCase(std::uint64_t)>;

// Runs `total` records through the pixel kernel in chunks and compares each outcome with the CPU reference.
std::uint64_t run_pixel_suite(Kernel& kernel, const char* label, std::uint64_t total, const CaseMaker& make) {
    std::vector<GpuPixelCase> records(static_cast<std::size_t>(kPixelChunk));
    std::uint64_t mismatches = 0;
    std::uint64_t frame_writes = 0;
    std::uint64_t depth_writes = 0;
    int reported = 0;
    std::string error;
    for (std::uint64_t base = 0; base < total; base += kPixelChunk) {
        const std::uint64_t count = std::min(kPixelChunk, total - base);
        for (std::uint64_t i = 0; i < count; ++i) {
            records[static_cast<std::size_t>(i)] = make(base + i);
        }
        std::memcpy(kernel.mapped, records.data(), static_cast<std::size_t>(count) * sizeof(GpuPixelCase));
        const std::uint32_t push = static_cast<std::uint32_t>(count);
        if (!kernel.run(push, &push, error)) {
            std::fprintf(stderr, "%s: %s\n", label, error.c_str());
            return total;
        }
        const auto* got = static_cast<const GpuPixelCase*>(kernel.mapped);
        for (std::uint64_t i = 0; i < count; ++i) {
            const GpuPixelCase& input = records[static_cast<std::size_t>(i)];
            const GpuPixelCase& output = got[i];
            const PixelOutcome expected = expected_outcome(input);
            frame_writes += expected.frame_written ? 1 : 0;
            depth_writes += expected.depth_written ? 1 : 0;
            const bool same = (output.frame_written != 0) == expected.frame_written &&
                              output.frame_stored == expected.frame_stored &&
                              (output.depth_written != 0) == expected.depth_written &&
                              output.depth_stored == expected.depth_stored;
            if (!same) {
                ++mismatches;
                if (reported < kReportLimit) {
                    ++reported;
                    std::fprintf(stderr,
                                 "%s case %llu: gpu frame %u/%08X depth %u/%08X, cpu frame %d/%08X depth %d/%08X "
                                 "(format %u blend %u/%u/%u/%u/%u fix %u clamp %u tex %u tfx %u ate %u atst %u afail %u)\n",
                                 label, static_cast<unsigned long long>(base + i), output.frame_written,
                                 output.frame_stored, output.depth_written, output.depth_stored,
                                 expected.frame_written ? 1 : 0, expected.frame_stored, expected.depth_written ? 1 : 0,
                                 expected.depth_stored, input.format, input.blend_on, input.blend_a, input.blend_b,
                                 input.blend_c, input.blend_d, input.fix, input.colclamp, input.tex_on, input.tfx,
                                 input.ate, input.atst, input.afail);
                }
            }
        }
    }
    std::printf("%-24s %llu cases, %llu mismatches (CPU frame writes %llu, depth writes %llu)\n", label,
                static_cast<unsigned long long>(total), static_cast<unsigned long long>(mismatches),
                static_cast<unsigned long long>(frame_writes), static_cast<unsigned long long>(depth_writes));
    return mismatches;
}

// Directed sweeps: TFX (4 functions x TCC x texel x fragment), TEXA (16-bit texel expansion), fog, and the
// alpha-test outcome of every compare, fail mode, reference and fragment alpha.
std::uint64_t run_directed(Vulkan& vk) {
    Kernel kernel;
    std::string error;
    if (!kernel.create(vk, coregs::shaders::hw_pixel_exact_words, coregs::shaders::hw_pixel_exact_word_count,
                       kPixelChunk * sizeof(GpuPixelCase), 4, error)) {
        std::fprintf(stderr, "hw_pixel_exact: %s\n", error.c_str());
        return 1;
    }
    std::uint64_t failures = 0;

    failures += run_pixel_suite(kernel, "hw_gpu_tfx_sweep", 4ull * 2 * 256 * 256, [](std::uint64_t i) {
        GpuPixelCase c = plain_case();
        const std::uint32_t tfx = static_cast<std::uint32_t>(i / (2 * 256 * 256));
        const std::uint32_t tcc = static_cast<std::uint32_t>((i / (256 * 256)) % 2);
        const std::uint32_t t = static_cast<std::uint32_t>((i / 256) % 256);
        const std::uint32_t v = static_cast<std::uint32_t>(i % 256);
        c.tex_on = 1;
        c.tfx = tfx;
        c.tcc = tcc;
        c.texel_kind = 0;
        c.texel_raw = t | (t << 8) | (t << 16) | (t << 24);
        c.fr = c.fg = c.fb = c.fa = v;
        return c;
    });

    failures += run_pixel_suite(kernel, "hw_gpu_texa_sweep", 65536ull * 4 * 2 * 2, [](std::uint64_t i) {
        static const std::uint32_t ta0_values[4] = {0x00, 0x40, 0x80, 0xFF};
        static const std::uint32_t ta1_values[2] = {0x20, 0x80};
        GpuPixelCase c = plain_case();
        c.tex_on = 1;
        c.tfx = 1;  // DECAL: the output is the expanded texel
        c.tcc = 1;
        c.texel_kind = 1;
        c.texel_raw = static_cast<std::uint32_t>(i % 65536);
        c.ta0 = ta0_values[(i / 65536) % 4];
        c.ta1 = ta1_values[(i / (65536 * 4)) % 2];
        c.aem = static_cast<std::uint32_t>((i / (65536 * 8)) % 2);
        return c;
    });

    failures += run_pixel_suite(kernel, "hw_gpu_fog_sweep", 256ull * 256 * 2, [](std::uint64_t i) {
        GpuPixelCase c = plain_case();
        const std::uint32_t fog = static_cast<std::uint32_t>(i % 256);
        const std::uint32_t fr = static_cast<std::uint32_t>((i / 256) % 256);
        const std::uint32_t fc = ((i / 65536) % 2) ? 255u : 0u;
        c.fog_on = 1;
        c.fog = fog;
        c.fog_r = fc;
        c.fog_g = fc ^ 0x5Au;
        c.fog_b = fc == 255u ? 0x80u : 0x10u;
        c.fr = fr;
        c.fg = 255u - fr;
        c.fb = fr / 2;
        c.fa = 0x80;
        return c;
    });

    failures += run_pixel_suite(kernel, "hw_gpu_atest_sweep", 8ull * 4 * 256 * 256, [](std::uint64_t i) {
        GpuPixelCase c = plain_case();
        c.ate = 1;
        c.atst = static_cast<std::uint32_t>(i % 8);
        c.afail = static_cast<std::uint32_t>((i / 8) % 4);
        c.aref = static_cast<std::uint32_t>((i / 32) % 256);
        c.fa = static_cast<std::uint32_t>((i / 8192) % 256);
        c.fr = 0x55;
        c.fg = 0x66;
        c.fb = 0x77;
        c.frame_old = 0x11223344u;
        c.depth_old = 0x0000ABCDu;
        c.ztest = 1;
        c.ztst = 1;  // ALWAYS, so depth writes come only from the alpha-test outcome
        return c;
    });

    kernel.destroy();
    return failures;
}

// Seeded random sweep over every pixel state field. Edge values of colour and Z are over-represented.
GpuPixelCase random_case(std::uint64_t index) {
    Rng rng(index * 0x2545F4914F6CDD1Dull + 0x51A7);
    GpuPixelCase c{};
    c.format = rng.below(3);
    switch (rng.below(3)) {
        case 0: c.fbmsk = 0; break;
        case 1: c.fbmsk = rng.next32(); break;
        default: c.fbmsk = rng.below(2) ? 0xFFFFFFFFu : 0x000000FFu; break;
    }
    c.fba = rng.below(2);
    c.dither = rng.below(2);
    c.dimx_lo = rng.next32();
    c.dimx_hi = rng.next32();
    c.ate = rng.below(2);
    c.atst = rng.below(8);
    c.afail = rng.below(4);
    c.aref = rng.byte();
    c.date_on = rng.below(2);
    c.datm = rng.below(2);
    c.ztest = rng.below(2);
    c.ztst = rng.below(4);
    c.zmsk = rng.below(2);
    c.zformat = rng.below(3);
    c.blend_on = rng.below(2);
    c.pabe = rng.below(2);
    c.blend_a = rng.below(3);
    c.blend_b = rng.below(3);
    c.blend_c = rng.below(3);
    c.blend_d = rng.below(3);
    c.fix = rng.byte();
    c.colclamp = rng.below(2);
    c.tex_on = rng.below(2);
    c.tfx = rng.below(4);
    c.tcc = rng.below(2);
    c.texel_kind = rng.below(3);
    c.texel_raw = rng.next32();
    c.ta0 = rng.byte();
    c.ta1 = rng.byte();
    c.aem = rng.below(2);
    c.fog_on = rng.below(2);
    c.fog = rng.byte();
    c.fog_r = rng.byte();
    c.fog_g = rng.byte();
    c.fog_b = rng.byte();
    c.x = rng.below(1024);
    c.y = rng.below(1024);
    c.fr = rng.channel();
    c.fg = rng.channel();
    c.fb = rng.channel();
    c.fa = rng.channel();
    switch (rng.below(3)) {
        case 0: c.fz = rng.next32(); break;
        case 1: c.fz = rng.below(2) ? 0xFFFFFFFFu : 0x00FFFFFFu; break;
        default: c.fz = rng.below(2) ? 0xFFFFu : rng.next32() >> 8; break;
    }
    c.frame_old = rng.next32();
    c.depth_old = rng.next32();
    return c;
}

std::uint64_t run_random(Vulkan& vk) {
    Kernel kernel;
    std::string error;
    if (!kernel.create(vk, coregs::shaders::hw_pixel_exact_words, coregs::shaders::hw_pixel_exact_word_count,
                       kPixelChunk * sizeof(GpuPixelCase), 4, error)) {
        std::fprintf(stderr, "hw_pixel_exact: %s\n", error.c_str());
        return 1;
    }
    const std::uint64_t failures = run_pixel_suite(kernel, "hw_gpu_pixel_random", 1000000ull, random_case);
    kernel.destroy();
    return failures;
}

struct TestEntry {
    const char* name;
    std::uint64_t (*run)(Vulkan&);
};

std::uint64_t run_blend_entry(Vulkan& vk) {
    return static_cast<std::uint64_t>(test_blend_sweep(vk) != 0);
}

}  // namespace

int main(int argc, char** argv) {
    Vulkan vk;
    std::string error;
    const int init = vk.create(error);
    if (init == kSkipCode) {
        std::printf("hw_pixel_gpu_fixture: skipped: %s\n", error.c_str());
        return kSkipCode;
    }
    if (init != 0) {
        std::fprintf(stderr, "hw_pixel_gpu_fixture: %s\n", error.c_str());
        return 2;
    }
    std::printf("device: %s (headless compute, no window)\n", vk.device_name.c_str());

    const TestEntry tests[] = {
        {"hw_gpu_blend_sweep", run_blend_entry},
        {"hw_gpu_pixel_directed", run_directed},
        {"hw_gpu_pixel_random", run_random},
    };
    int ran = 0;
    std::uint64_t failed_tests = 0;
    for (const TestEntry& entry : tests) {
        bool selected = argc <= 1;
        for (int i = 1; i < argc; ++i) {
            if (std::strcmp(argv[i], entry.name) == 0) selected = true;
        }
        if (!selected) continue;
        ++ran;
        failed_tests += entry.run(vk) != 0;
    }
    if (ran == 0) {
        std::fprintf(stderr, "no test selected\n");
        return 2;
    }
    std::printf("%s\n", failed_tests == 0 ? "all selected GPU tests passed" : "GPU mismatches found");
    return failed_tests == 0 ? 0 : 1;
}
