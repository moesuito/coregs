// MIT License. Copyright (c) 2026 João Alano and GT4Recomp contributors.
// TASK-179 (Android plan stage A1): the capability probe of CoreGS 2.0. Written from the Vulkan specification, clean room
// (decision 0070).
//
// One small record of what the chosen GPU is and has, filled once at start (vulkan_loader::probe_device_profile), logged
// in one line and handed to the renderer with the device (gpu::VulkanHandles::profile), so that later stages can key
// settings off the vendor and device class instead of the single "integrated GPU" test of today. Only the structure is
// new: `integrated_defaults` is today's rule (VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU) and nothing reads `device_class`
// yet. Also the sizing of core 2's host-visible rings from the heaps (plan_core2_rings).
//
// Pure arithmetic and strings, no Vulkan: the numbers that come from Vulkan are copied in as plain integers (the
// physical device type has the VkPhysicalDeviceType values, the transform bits the VkSurfaceTransformFlagBitsKHR ones).
#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <string>

namespace coregs {

// PCI vendor ids (and the Khronos ids of vendors without one) that Vulkan reports in VkPhysicalDeviceProperties::vendorID.
constexpr std::uint32_t kVendorAmd = 0x1002;
constexpr std::uint32_t kVendorNvidia = 0x10DE;
constexpr std::uint32_t kVendorIntel = 0x8086;
constexpr std::uint32_t kVendorArm = 0x13B5;        // Mali
constexpr std::uint32_t kVendorQualcomm = 0x5143;   // Adreno
constexpr std::uint32_t kVendorImagination = 0x1010;  // PowerVR
constexpr std::uint32_t kVendorSamsung = 0x144D;    // Xclipse
constexpr std::uint32_t kVendorApple = 0x106B;
constexpr std::uint32_t kVendorMesa = 0x10005;      // VK_VENDOR_ID_MESA (llvmpipe, lavapipe)

// VkPhysicalDeviceType values.
constexpr std::uint32_t kDeviceTypeOther = 0;
constexpr std::uint32_t kDeviceTypeIntegrated = 1;
constexpr std::uint32_t kDeviceTypeDiscrete = 2;
constexpr std::uint32_t kDeviceTypeVirtual = 3;
constexpr std::uint32_t kDeviceTypeCpu = 4;

inline const char *vendor_name(std::uint32_t vendor_id) {
  switch (vendor_id) {
  case kVendorAmd: return "AMD";
  case kVendorNvidia: return "NVIDIA";
  case kVendorIntel: return "Intel";
  case kVendorArm: return "ARM";
  case kVendorQualcomm: return "Qualcomm";
  case kVendorImagination: return "Imagination";
  case kVendorSamsung: return "Samsung";
  case kVendorApple: return "Apple";
  case kVendorMesa: return "Mesa";
  default: return "unknown";
  }
}

// What kind of GPU it is, for settings that depend on it.
//   discrete: its own memory (VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU).
//   integrated: an immediate-mode GPU sharing the system memory, from a desktop vendor (AMD, Intel, NVIDIA): the class
//     today's "integrated GPU" defaults were tuned on (an Intel HD 620).
//   tiler_or_unknown: everything else: the tile-based mobile GPUs (Adreno, Mali, PowerVR, Apple, Xclipse until measured)
//     and software, virtual or unknown devices. Nothing is tuned for this class yet (Android plan stages A4, A5).
enum class DeviceClass : std::uint8_t { discrete, integrated, tiler_or_unknown };

inline const char *device_class_name(DeviceClass device_class) {
  switch (device_class) {
  case DeviceClass::discrete: return "discrete";
  case DeviceClass::integrated: return "integrated";
  case DeviceClass::tiler_or_unknown: return "tiler-or-unknown";
  }
  return "tiler-or-unknown";
}

inline bool desktop_vendor(std::uint32_t vendor_id) {
  return vendor_id == kVendorAmd || vendor_id == kVendorIntel || vendor_id == kVendorNvidia;
}

inline DeviceClass classify_device(std::uint32_t vendor_id, std::uint32_t device_type) {
  if (device_type == kDeviceTypeDiscrete) return DeviceClass::discrete;
  if (device_type == kDeviceTypeIntegrated && desktop_vendor(vendor_id)) return DeviceClass::integrated;
  return DeviceClass::tiler_or_unknown;
}

// Today's rule for core 2's integrated-GPU defaults (IN-431, IN-476, IN-481, IN-494, IN-496, IN-517 and TASK-134): the
// device type alone. Kept exactly; a phone GPU that reports itself integrated still gets them until stage A4 keys them
// off the class instead.
inline bool integrated_defaults(std::uint32_t device_type) { return device_type == kDeviceTypeIntegrated; }

struct HeapInfo {
  std::uint64_t size = 0;    // VkMemoryHeap::size
  std::uint64_t budget = 0;  // VK_EXT_memory_budget heapBudget at the probe; 0 without the extension
  std::uint64_t usage = 0;   // VK_EXT_memory_budget heapUsage at the probe
  bool device_local = false;
};

struct DeviceProfile {
  bool probed = false;  // false: the device's creator did not run the probe (every field below is a default)
  std::string name;
  std::uint32_t vendor_id = 0, device_id = 0, device_type = kDeviceTypeOther;
  std::uint32_t api_version = 0, driver_version = 0;
  DeviceClass device_class = DeviceClass::tiler_or_unknown;
  bool integrated_defaults = false;

  // Features that drawing at a scale above 1x relies on. `*_supported` is what the device has, `*_enabled` what the
  // device was created with (the creator sets it); the renderer keys its paths off `*_enabled`.
  bool large_points_supported = false, large_points_enabled = false;  // gl_PointSize = scale (hw_vertex.vert)
  float point_size_max = 1.0f, point_size_granularity = 0.0f;
  bool wide_lines_supported = false;  // reported only: every pipeline draws its lines 1.0 wide, so it is never needed
  float line_width_max = 1.0f;
  bool storage_image_array_dynamic_indexing_supported = false;  // hw_deinterlace_scaled.comp: history[index]
  bool storage_image_array_dynamic_indexing_enabled = false;

  // Memory: every heap, with its budget where VK_EXT_memory_budget exists (enabled on the device when it does).
  bool memory_budget = false;
  std::uint32_t heap_count = 0;
  std::array<HeapInfo, 16> heaps{};

  // True when points of `size` pixels can be drawn as points (else core 2 draws them as two triangles).
  bool points_of_size(float size) const { return size <= 1.0f || (large_points_enabled && size <= point_size_max); }
};

// The largest heap with (device_local) or without device-local memory; -1 when there is none.
inline int largest_heap(const DeviceProfile &profile, bool device_local) {
  int best = -1;
  for (std::uint32_t i = 0; i < profile.heap_count && i < profile.heaps.size(); ++i) {
    if (profile.heaps[i].device_local != device_local) continue;
    if (best < 0 || profile.heaps[i].size > profile.heaps[static_cast<std::size_t>(best)].size) best = static_cast<int>(i);
  }
  return best;
}

inline std::string mib_text(std::uint64_t bytes) { return std::to_string(bytes >> 20) + " MiB"; }

// The one log line of the probe.
inline std::string profile_line(const DeviceProfile &p) {
  char text[512];
  auto yes = [](bool value) { return value ? "yes" : "no"; };
  std::snprintf(text, sizeof text,
                "%s, vendor %s (0x%04X) device 0x%04X, class %s, integrated defaults %s, Vulkan %u.%u.%u; "
                "large points %s (max %.1f, %s), wide lines %s (max %.1f, not used), "
                "storage image array dynamic indexing %s (%s); memory budget %s",
                p.name.c_str(), vendor_name(p.vendor_id), p.vendor_id, p.device_id, device_class_name(p.device_class),
                p.integrated_defaults ? "on" : "off", p.api_version >> 22, (p.api_version >> 12) & 0x3FFu,
                p.api_version & 0xFFFu, yes(p.large_points_supported), static_cast<double>(p.point_size_max),
                p.large_points_enabled ? "enabled" : "points drawn as quads above 1x", yes(p.wide_lines_supported),
                static_cast<double>(p.line_width_max), yes(p.storage_image_array_dynamic_indexing_supported),
                p.storage_image_array_dynamic_indexing_enabled ? "enabled" : "constant-index shader",
                p.memory_budget ? "yes" : "no");
  std::string line = text;
  for (std::uint32_t i = 0; i < p.heap_count && i < p.heaps.size(); ++i) {
    const HeapInfo &heap = p.heaps[i];
    line += (i == 0 ? ", heaps " : ", ") + std::to_string(i) + (heap.device_local ? ":local " : ":host ") + mib_text(heap.size);
    if (p.memory_budget) line += " (budget " + mib_text(heap.budget) + ", used " + mib_text(heap.usage) + ")";
  }
  return line;
}

// ---- Core 2's host-visible rings, sized from the device (TASK-179) -------------------------------------------------
//
// Core 2 maps four rings: staging, vertex and uniform with one share per recording in flight, and the readback buffer.
// Today's sizes come from constants: per share 32 MiB staging, 8 MiB vertex and 2 MiB uniform, 8 or 16 shares (8 on an
// integrated GPU, 16 on a discrete one), and 64 MiB of readback: 400 or 736 MiB. Those stay the sizes, and the cap,
// wherever they fit in a quarter of the heap they live in (its VK_EXT_memory_budget budget when the device reports one,
// else its size). Where they do not, the number of shares goes down first (a share keeps its size, so no request that
// fits today stops fitting; fewer recordings are in flight), to two at least; then the readback buffer, to 16 MiB at
// least (a whole 2048 x 2048 1x reduction; larger display reads go in bands). A phone's low-memory killer acts fast.

constexpr std::uint64_t kRingMiB = std::uint64_t{1} << 20;
constexpr std::uint32_t kRingMinimumShares = 2;
constexpr std::uint64_t kReadbackMinimumBytes = 16 * kRingMiB;

struct RingRequest {
  std::uint64_t heap_size = 0;    // the heap of the ring memory type
  std::uint64_t heap_budget = 0;  // its budget (VK_EXT_memory_budget), 0 when unknown
  std::uint64_t forced_budget = 0;  // COREGS_CORE2_RINGS=<MiB>: a budget to plan for instead (tests on a large desktop)
  bool fixed = false;               // COREGS_CORE2_RINGS=fixed: today's sizes whatever the heap
  std::uint32_t shares = 16;        // today's share count
  std::uint64_t staging_share = 32 * kRingMiB, vertex_share = 8 * kRingMiB, uniform_share = 2 * kRingMiB;
  std::uint64_t readback = 64 * kRingMiB;
  std::uint64_t other = 0;  // other mapped buffers in the same heap (counted, never resized)
};

struct RingPlan {
  std::uint32_t shares = 0;
  std::uint64_t staging = 0, vertex = 0, uniform = 0, readback = 0;
  std::uint64_t allowance = 0;  // a quarter of the budget planned for (0 with `fixed`)
  bool reduced = false;         // below today's sizes
  const char *basis = "";       // "fixed", "forced budget", "heap budget" or "heap size"

  std::uint64_t total() const { return staging + vertex + uniform + readback; }
};

inline RingPlan plan_core2_rings(const RingRequest &r) {
  RingPlan plan;
  const std::uint64_t per_share = r.staging_share + r.vertex_share + r.uniform_share;
  auto fill = [&](std::uint32_t shares, std::uint64_t readback) {
    plan.shares = shares;
    plan.staging = r.staging_share * shares;
    plan.vertex = r.vertex_share * shares;
    plan.uniform = r.uniform_share * shares;
    plan.readback = readback;
  };
  fill(r.shares, r.readback);
  if (r.fixed) {
    plan.basis = "fixed";
    return plan;
  }
  std::uint64_t base = r.heap_size;
  plan.basis = "heap size";
  if (r.forced_budget != 0) {
    base = r.forced_budget;
    plan.basis = "forced budget";
  } else if (r.heap_budget != 0) {
    base = r.heap_size != 0 ? std::min(r.heap_budget, r.heap_size) : r.heap_budget;
    plan.basis = "heap budget";
  }
  if (base == 0) {  // nothing known about the heap: today's sizes
    plan.basis = "unknown heap";
    return plan;
  }
  plan.allowance = base / 4;
  const std::uint64_t room = plan.allowance > r.other ? plan.allowance - r.other : 0;
  if (plan.total() <= room) return plan;
  plan.reduced = true;
  const std::uint64_t for_shares = room > r.readback ? room - r.readback : 0;
  const std::uint64_t fitting = per_share != 0 ? for_shares / per_share : r.shares;
  const std::uint32_t shares = static_cast<std::uint32_t>(
      std::clamp<std::uint64_t>(fitting, kRingMinimumShares, std::max<std::uint64_t>(r.shares, kRingMinimumShares)));
  std::uint64_t readback = r.readback;
  const std::uint64_t share_bytes = per_share * shares;
  if (share_bytes + readback > room) {
    const std::uint64_t left = room > share_bytes ? room - share_bytes : 0;
    readback = std::clamp<std::uint64_t>(left / kRingMiB * kRingMiB, std::min(kReadbackMinimumBytes, r.readback), r.readback);
  }
  fill(shares, readback);
  return plan;
}

}  // namespace coregs
