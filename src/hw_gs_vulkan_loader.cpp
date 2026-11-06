// MIT License. Copyright (c) 2026 João Alano and GT4Recomp contributors.
// Written from the Vulkan specification. See hw_gs_vulkan_loader.hpp for why the names are renamed.

#include "hw_gs_vulkan_loader.hpp"

// Vulkan create-info and query structs are initialised with sType only (pNext defaults to null by the spec).
#pragma clang diagnostic ignored "-Wmissing-field-initializers"

#include <cstdio>
#include <cstdlib>
#include <cstring>

#if !defined(_WIN32)
#include <dlfcn.h>
#endif

namespace coregs::vulkan_loader {
namespace {

PFN_vkGetInstanceProcAddr get_instance_proc = nullptr;

// Resolves one entry point through vkGetInstanceProcAddr; a missing name is reported, not called.
template <typename Pointer>
void resolve(Pointer &slot, VkInstance instance, const char *name, std::vector<std::string> &missing) {
  slot = reinterpret_cast<Pointer>(get_instance_proc(instance, name));
  if (!slot) missing.push_back(name);
}

} // namespace

PFN_vkVoidFunction instance_proc(VkInstance instance, const char *name) { return get_instance_proc(instance, name); }

bool load_loader(std::string &error) {
  if (get_instance_proc) return true;
#if defined(_WIN32)
  HMODULE library = LoadLibraryExW(L"vulkan-1.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
  const char *const name = "vulkan-1.dll";
  void *symbol = library ? reinterpret_cast<void *>(GetProcAddress(library, "vkGetInstanceProcAddr")) : nullptr;
#else
  void *library = dlopen("libvulkan.so.1", RTLD_NOW | RTLD_LOCAL);
  const char *const name = "libvulkan.so.1";
  void *symbol = library ? dlsym(library, "vkGetInstanceProcAddr") : nullptr;
#endif
  if (!library) {
    error = std::string(name) + " could not be loaded";
    return false;
  }
  get_instance_proc = reinterpret_cast<PFN_vkGetInstanceProcAddr>(symbol);
  if (!get_instance_proc) {
    error = std::string(name) + " lacks vkGetInstanceProcAddr";
    return false;
  }
  coregs_hw_vkCreateInstance = reinterpret_cast<PFN_vkCreateInstance>(get_instance_proc(nullptr, "vkCreateInstance"));
  if (!coregs_hw_vkCreateInstance) {
    error = std::string(name) + " lacks vkCreateInstance";
    return false;
  }
  return true;
}

namespace {

// Entry points a windowless device needs: no surface, no swapchain (M4 fixtures and headless runs).
void resolve_headless(VkInstance instance, std::vector<std::string> &missing) {
  resolve(coregs_hw_vkAllocateCommandBuffers, instance, "vkAllocateCommandBuffers", missing);
  resolve(coregs_hw_vkBeginCommandBuffer, instance, "vkBeginCommandBuffer", missing);
  resolve(coregs_hw_vkCmdClearColorImage, instance, "vkCmdClearColorImage", missing);
  resolve(coregs_hw_vkCmdPipelineBarrier, instance, "vkCmdPipelineBarrier", missing);
  resolve(coregs_hw_vkCreateCommandPool, instance, "vkCreateCommandPool", missing);
  resolve(coregs_hw_vkCreateDevice, instance, "vkCreateDevice", missing);
  resolve(coregs_hw_vkCreateFence, instance, "vkCreateFence", missing);
  resolve(coregs_hw_vkCreateSemaphore, instance, "vkCreateSemaphore", missing);
  resolve(coregs_hw_vkDestroyCommandPool, instance, "vkDestroyCommandPool", missing);
  resolve(coregs_hw_vkDestroyDevice, instance, "vkDestroyDevice", missing);
  resolve(coregs_hw_vkDestroyFence, instance, "vkDestroyFence", missing);
  resolve(coregs_hw_vkDestroyInstance, instance, "vkDestroyInstance", missing);
  resolve(coregs_hw_vkDestroySemaphore, instance, "vkDestroySemaphore", missing);
  resolve(coregs_hw_vkDeviceWaitIdle, instance, "vkDeviceWaitIdle", missing);
  resolve(coregs_hw_vkEndCommandBuffer, instance, "vkEndCommandBuffer", missing);
  resolve(coregs_hw_vkEnumerateDeviceExtensionProperties, instance, "vkEnumerateDeviceExtensionProperties", missing);
  resolve(coregs_hw_vkEnumeratePhysicalDevices, instance, "vkEnumeratePhysicalDevices", missing);
  resolve(coregs_hw_vkGetDeviceQueue, instance, "vkGetDeviceQueue", missing);
  resolve(coregs_hw_vkGetPhysicalDeviceFeatures2, instance, "vkGetPhysicalDeviceFeatures2", missing);
  resolve(coregs_hw_vkGetPhysicalDeviceFormatProperties, instance, "vkGetPhysicalDeviceFormatProperties", missing);
  resolve(coregs_hw_vkGetPhysicalDeviceProperties, instance, "vkGetPhysicalDeviceProperties", missing);
  resolve(coregs_hw_vkGetPhysicalDeviceQueueFamilyProperties, instance, "vkGetPhysicalDeviceQueueFamilyProperties", missing);
  resolve(coregs_hw_vkQueueSubmit, instance, "vkQueueSubmit", missing);
  resolve(coregs_hw_vkResetCommandBuffer, instance, "vkResetCommandBuffer", missing);
  resolve(coregs_hw_vkResetFences, instance, "vkResetFences", missing);
  resolve(coregs_hw_vkWaitForFences, instance, "vkWaitForFences", missing);
}

// Surface and presentation entry points: only the windowed path needs them.
void resolve_surface(VkInstance instance, std::vector<std::string> &missing) {
  resolve(coregs_hw_vkAcquireNextImageKHR, instance, "vkAcquireNextImageKHR", missing);
  resolve(coregs_hw_vkCreateSwapchainKHR, instance, "vkCreateSwapchainKHR", missing);
#if defined(VK_USE_PLATFORM_WIN32_KHR)
  resolve(coregs_hw_vkCreateWin32SurfaceKHR, instance, "vkCreateWin32SurfaceKHR", missing);
#endif
  resolve(coregs_hw_vkDestroySurfaceKHR, instance, "vkDestroySurfaceKHR", missing);
  resolve(coregs_hw_vkDestroySwapchainKHR, instance, "vkDestroySwapchainKHR", missing);
  resolve(coregs_hw_vkGetPhysicalDeviceSurfaceCapabilitiesKHR, instance, "vkGetPhysicalDeviceSurfaceCapabilitiesKHR", missing);
  resolve(coregs_hw_vkGetPhysicalDeviceSurfaceFormatsKHR, instance, "vkGetPhysicalDeviceSurfaceFormatsKHR", missing);
  resolve(coregs_hw_vkGetPhysicalDeviceSurfacePresentModesKHR, instance, "vkGetPhysicalDeviceSurfacePresentModesKHR", missing);
  resolve(coregs_hw_vkGetPhysicalDeviceSurfaceSupportKHR, instance, "vkGetPhysicalDeviceSurfaceSupportKHR", missing);
  resolve(coregs_hw_vkGetSwapchainImagesKHR, instance, "vkGetSwapchainImagesKHR", missing);
  resolve(coregs_hw_vkQueuePresentKHR, instance, "vkQueuePresentKHR", missing);
}

} // namespace

bool load_instance_entry_points(VkInstance instance, std::vector<std::string> &missing) {
  resolve_headless(instance, missing);
  resolve_surface(instance, missing);
  return missing.empty();
}

bool pixel_interlock_supported(VkPhysicalDevice physical) {
  if (!coregs_hw_vkEnumerateDeviceExtensionProperties || !coregs_hw_vkGetPhysicalDeviceFeatures2) return false;
  std::uint32_t count = 0;
  coregs_hw_vkEnumerateDeviceExtensionProperties(physical, nullptr, &count, nullptr);
  std::vector<VkExtensionProperties> extensions(count);
  coregs_hw_vkEnumerateDeviceExtensionProperties(physical, nullptr, &count, extensions.data());
  bool present = false;
  for (const VkExtensionProperties &extension : extensions) {
    if (std::strcmp(extension.extensionName, VK_EXT_FRAGMENT_SHADER_INTERLOCK_EXTENSION_NAME) == 0) present = true;
  }
  if (!present) return false;
  VkPhysicalDeviceFragmentShaderInterlockFeaturesEXT interlock{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FRAGMENT_SHADER_INTERLOCK_FEATURES_EXT};
  VkPhysicalDeviceFeatures2 features{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
  features.pNext = &interlock;
  coregs_hw_vkGetPhysicalDeviceFeatures2(physical, &features);
  return interlock.fragmentShaderPixelInterlock == VK_TRUE;
}

bool load_headless_entry_points(VkInstance instance, std::vector<std::string> &missing) {
  resolve_headless(instance, missing);
  return missing.empty();
}

coregs::DeviceProfile probe_device_profile(VkInstance instance, VkPhysicalDevice physical) {
  coregs::DeviceProfile profile;
  if (!coregs_hw_vkGetPhysicalDeviceProperties || !coregs_hw_vkGetPhysicalDeviceFeatures2) return profile;
  VkPhysicalDeviceProperties properties{};
  coregs_hw_vkGetPhysicalDeviceProperties(physical, &properties);
  VkPhysicalDeviceFeatures2 features{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
  coregs_hw_vkGetPhysicalDeviceFeatures2(physical, &features);
  profile.probed = true;
  profile.name = properties.deviceName;
  profile.vendor_id = properties.vendorID;
  profile.device_id = properties.deviceID;
  profile.device_type = static_cast<std::uint32_t>(properties.deviceType);
  profile.api_version = properties.apiVersion;
  profile.driver_version = properties.driverVersion;
  profile.device_class = coregs::classify_device(profile.vendor_id, profile.device_type);
  profile.integrated_defaults = coregs::integrated_defaults(profile.device_type);
  profile.large_points_supported = features.features.largePoints == VK_TRUE;
  profile.point_size_max = properties.limits.pointSizeRange[1];
  profile.point_size_granularity = properties.limits.pointSizeGranularity;
  profile.wide_lines_supported = features.features.wideLines == VK_TRUE;
  profile.line_width_max = properties.limits.lineWidthRange[1];
  profile.storage_image_array_dynamic_indexing_supported = features.features.shaderStorageImageArrayDynamicIndexing == VK_TRUE;

  if (const char *pretend = std::getenv("COREGS_HW_PRETEND_MISSING")) {
    if (std::strstr(pretend, "large_points")) profile.large_points_supported = false;
    if (std::strstr(pretend, "dynamic_indexing")) profile.storage_image_array_dynamic_indexing_supported = false;
    std::fprintf(stderr, "[hw] COREGS_HW_PRETEND_MISSING=%s: those features are treated as missing (diagnostics)\n", pretend);
  }

  if (coregs_hw_vkEnumerateDeviceExtensionProperties) {
    std::uint32_t count = 0;
    coregs_hw_vkEnumerateDeviceExtensionProperties(physical, nullptr, &count, nullptr);
    std::vector<VkExtensionProperties> extensions(count);
    coregs_hw_vkEnumerateDeviceExtensionProperties(physical, nullptr, &count, extensions.data());
    for (const VkExtensionProperties &extension : extensions) {
      if (std::strcmp(extension.extensionName, VK_EXT_MEMORY_BUDGET_EXTENSION_NAME) == 0) profile.memory_budget = true;
    }
  }
  const auto properties2 = reinterpret_cast<PFN_vkGetPhysicalDeviceMemoryProperties2>(
      instance_proc(instance, "vkGetPhysicalDeviceMemoryProperties2"));
  const auto properties1 = reinterpret_cast<PFN_vkGetPhysicalDeviceMemoryProperties>(
      instance_proc(instance, "vkGetPhysicalDeviceMemoryProperties"));
  VkPhysicalDeviceMemoryBudgetPropertiesEXT budget{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT};
  VkPhysicalDeviceMemoryProperties2 memory{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2};
  if (profile.memory_budget && properties2) {
    memory.pNext = &budget;
    properties2(physical, &memory);
  } else if (properties1) {
    profile.memory_budget = false;
    properties1(physical, &memory.memoryProperties);
  } else {
    profile.memory_budget = false;
  }
  profile.heap_count = std::min<std::uint32_t>(memory.memoryProperties.memoryHeapCount,
                                               static_cast<std::uint32_t>(profile.heaps.size()));
  for (std::uint32_t i = 0; i < profile.heap_count; ++i) {
    coregs::HeapInfo &heap = profile.heaps[i];
    heap.size = memory.memoryProperties.memoryHeaps[i].size;
    heap.device_local = (memory.memoryProperties.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) != 0;
    if (profile.memory_budget) {
      heap.budget = budget.heapBudget[i];
      heap.usage = budget.heapUsage[i];
    }
  }
  return profile;
}

void enable_profile_features(coregs::DeviceProfile &profile, VkPhysicalDeviceFeatures &features) {
  // Points are drawn gl_PointSize = scale wide (hw_vertex.vert): above 1 they need largePoints (core 2 draws them as two
  // triangles without it). The deinterlace kernel indexes its history images with a push constant (hw_deinterlace_scaled.comp),
  // which needs shaderStorageImageArrayDynamicIndexing (without it the constant-index build of the kernel runs).
  profile.large_points_enabled = profile.large_points_supported;
  profile.storage_image_array_dynamic_indexing_enabled = profile.storage_image_array_dynamic_indexing_supported;
  if (profile.large_points_enabled) features.largePoints = VK_TRUE;
  if (profile.storage_image_array_dynamic_indexing_enabled) features.shaderStorageImageArrayDynamicIndexing = VK_TRUE;
}

const char *memory_budget_extension(const coregs::DeviceProfile &profile) {
  return profile.memory_budget ? VK_EXT_MEMORY_BUDGET_EXTENSION_NAME : nullptr;
}

} // namespace coregs::vulkan_loader
