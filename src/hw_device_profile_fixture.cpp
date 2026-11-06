// CTest hw_device_profile: TASK-179, the capability profile's class rule and log line, and the sizing of core 2's
// host-visible rings from the heap (hw_device_profile.hpp), host only.
#include "hw_device_profile.hpp"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>

using namespace coregs;

namespace {

void check(bool condition, const char *what) {
  if (condition) return;
  std::fprintf(stderr, "hw_device_profile: %s\n", what);
  std::exit(1);
}

constexpr std::uint64_t MiB = std::uint64_t{1} << 20;
constexpr std::uint64_t GiB = std::uint64_t{1} << 30;

RingRequest request(std::uint32_t shares, std::uint64_t heap, std::uint64_t budget) {
  RingRequest r;
  r.shares = shares;
  r.heap_size = heap;
  r.heap_budget = budget;
  r.other = 256 * 1024;  // core 2's delayed-reduction buffer
  return r;
}

bool today(const RingPlan &plan, std::uint32_t shares) {
  return plan.shares == shares && plan.staging == 32 * MiB * shares && plan.vertex == 8 * MiB * shares &&
         plan.uniform == 2 * MiB * shares && plan.readback == 64 * MiB && !plan.reduced;
}

}  // namespace

int main() {
  // ---- The class rule, and today's integrated-GPU rule kept as it is ----
  check(classify_device(kVendorAmd, kDeviceTypeDiscrete) == DeviceClass::discrete, "AMD discrete");
  check(classify_device(kVendorNvidia, kDeviceTypeDiscrete) == DeviceClass::discrete, "NVIDIA discrete");
  check(classify_device(kVendorIntel, kDeviceTypeIntegrated) == DeviceClass::integrated, "Intel integrated");
  check(classify_device(kVendorAmd, kDeviceTypeIntegrated) == DeviceClass::integrated, "AMD APU");
  check(classify_device(kVendorQualcomm, kDeviceTypeIntegrated) == DeviceClass::tiler_or_unknown, "Adreno is a tiler");
  check(classify_device(kVendorArm, kDeviceTypeIntegrated) == DeviceClass::tiler_or_unknown, "Mali is a tiler");
  check(classify_device(kVendorImagination, kDeviceTypeIntegrated) == DeviceClass::tiler_or_unknown, "PowerVR is a tiler");
  check(classify_device(kVendorMesa, kDeviceTypeCpu) == DeviceClass::tiler_or_unknown, "a software device is unknown");
  check(classify_device(kVendorNvidia, kDeviceTypeVirtual) == DeviceClass::tiler_or_unknown, "a virtual device is unknown");
  check(integrated_defaults(kDeviceTypeIntegrated), "the integrated defaults follow the device type");
  check(!integrated_defaults(kDeviceTypeDiscrete) && !integrated_defaults(kDeviceTypeOther), "only the integrated type");
  check(std::string(vendor_name(kVendorQualcomm)) == "Qualcomm" && std::string(vendor_name(0x1234)) == "unknown", "vendor names");

  // ---- Points: one pixel always, larger only with largePoints and within the device's largest size ----
  DeviceProfile profile;
  check(profile.points_of_size(1.0f) && !profile.points_of_size(2.0f), "no largePoints: one-pixel points only");
  profile.large_points_supported = true;
  check(!profile.points_of_size(3.0f), "supported but not enabled is not enough");
  profile.large_points_enabled = true;
  profile.point_size_max = 64.0f;
  check(profile.points_of_size(12.0f) && !profile.points_of_size(65.0f), "enabled, up to the largest size");

  // ---- The log line ----
  profile.probed = true;
  profile.name = "Test GPU";
  profile.vendor_id = kVendorAmd;
  profile.device_type = kDeviceTypeDiscrete;
  profile.device_class = DeviceClass::discrete;
  profile.api_version = (1u << 22) | (4u << 12) | 335u;
  profile.memory_budget = true;
  profile.heap_count = 2;
  profile.heaps[0] = HeapInfo{16 * GiB, 15 * GiB, 1 * GiB, true};
  profile.heaps[1] = HeapInfo{32 * GiB, 30 * GiB, 2 * GiB, false};
  const std::string line = profile_line(profile);
  check(line.find("Test GPU, vendor AMD (0x1002)") == 0, "the line starts with the name and vendor");
  check(line.find("class discrete") != std::string::npos, "the line names the class");
  check(line.find("Vulkan 1.4.335") != std::string::npos, "the line names the Vulkan version");
  check(line.find("large points yes (max 64.0, enabled)") != std::string::npos, "the line names largePoints");
  check(line.find("heaps 0:local 16384 MiB (budget 15360 MiB, used 1024 MiB), 1:host 32768 MiB") != std::string::npos,
        "the line lists the heaps");
  check(largest_heap(profile, true) == 0 && largest_heap(profile, false) == 1, "largest heaps");

  // ---- Rings: today's sizes wherever they fit in a quarter of the heap's budget ----
  check(today(plan_core2_rings(request(16, 16 * GiB, 15 * GiB)), 16), "a desktop keeps today's 736 MiB");
  check(plan_core2_rings(request(16, 16 * GiB, 15 * GiB)).total() == 736 * MiB, "736 MiB on a discrete GPU");
  check(today(plan_core2_rings(request(8, 4 * GiB, 0)), 8), "an integrated GPU without a budget keeps today's 400 MiB");
  check(plan_core2_rings(request(8, 4 * GiB, 0)).total() == 400 * MiB, "400 MiB on an integrated GPU");
  check(std::string(plan_core2_rings(request(8, 4 * GiB, 0)).basis) == "heap size", "the heap size without a budget");
  check(std::string(plan_core2_rings(request(8, 4 * GiB, 3 * GiB)).basis) == "heap budget", "the budget when there is one");
  check(plan_core2_rings(request(8, 4 * GiB, 3 * GiB)).allowance == 768 * MiB, "a quarter of the budget");
  check(plan_core2_rings(request(8, 2 * GiB, 3 * GiB)).allowance == 512 * MiB, "the budget is never above the heap");
  check(today(plan_core2_rings(request(16, 0, 0)), 16), "an unknown heap keeps today's sizes");
  {
    RingRequest fixed = request(16, 256 * MiB, 0);
    fixed.fixed = true;
    check(today(plan_core2_rings(fixed), 16), "COREGS_CORE2_RINGS=fixed keeps today's sizes on any heap");
  }
  {  // 2 GiB of budget: 512 MiB allowed; 64 MiB of readback leaves 10 shares of 42 MiB.
    const RingPlan plan = plan_core2_rings(request(16, 6 * GiB, 2 * GiB));
    check(plan.reduced && plan.shares == 10 && plan.readback == 64 * MiB, "2 GiB: 10 shares");
    check(plan.staging == 320 * MiB && plan.vertex == 80 * MiB && plan.uniform == 20 * MiB, "2 GiB: ring sizes");
    check(plan.total() <= plan.allowance, "2 GiB: within the allowance");
  }
  {  // A forced budget of 256 MiB: 64 MiB allowed, below the floor of two shares and 16 MiB of readback.
    RingRequest forced = request(16, 16 * GiB, 15 * GiB);
    forced.forced_budget = 256 * MiB;
    const RingPlan plan = plan_core2_rings(forced);
    check(std::string(plan.basis) == "forced budget", "the forced budget wins");
    check(plan.shares == kRingMinimumShares && plan.readback == kReadbackMinimumBytes, "the floor: two shares, 16 MiB");
  }
  {  // 600 MiB of allowance: 13 shares would leave too little readback; 12 shares and 64 MiB fit.
    const RingPlan plan = plan_core2_rings(request(16, 8 * GiB, 2400 * MiB));
    check(plan.shares == 12 && plan.readback == 64 * MiB && plan.total() <= plan.allowance, "2400 MiB: 12 shares");
  }
  {  // Two shares fit but not the whole readback: it shrinks, in whole MiB, not below 16.
    RingRequest small = request(16, 1 * GiB, 0);
    small.forced_budget = 512 * MiB;  // 128 MiB allowed: two shares are 84 MiB, 43 MiB left (minus the 256 KiB)
    const RingPlan plan = plan_core2_rings(small);
    check(plan.shares == 2 && plan.readback == 43 * MiB && plan.total() <= plan.allowance, "128 MiB: two shares, 43 MiB");
  }
  // Every budget: never above today's sizes, never below the floor, within the allowance whenever the floor allows.
  for (std::uint64_t budget = 16 * MiB; budget <= 24 * GiB; budget += 48 * MiB) {
    for (std::uint32_t shares : {8u, 16u, 24u}) {
      const RingPlan plan = plan_core2_rings(request(shares, 32 * GiB, budget));
      check(plan.shares >= kRingMinimumShares && plan.shares <= shares, "shares between the floor and today's");
      check(plan.readback >= kReadbackMinimumBytes && plan.readback <= 64 * MiB, "readback between 16 and 64 MiB");
      check(plan.staging == 32 * MiB * plan.shares, "a share keeps its size");
      const bool atFloor = plan.shares == kRingMinimumShares && plan.readback == kReadbackMinimumBytes;
      check(atFloor || plan.total() + 256 * 1024 <= plan.allowance, "within the allowance above the floor");
      check(plan.reduced == !today(plan, shares), "reduced exactly when below today's sizes");
    }
  }
  std::printf("hw_device_profile: ok\n");
  return 0;
}
