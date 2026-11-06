// MIT License. Copyright (c) 2026 João Alano and GT4Recomp contributors.
// Written from the Vulkan specification.
//
// Vulkan entry points of the CoreGS front end, owned by this translation unit.
//
// Every call goes through a pointer resolved at run time from the system Vulkan loader. That keeps
// CoreGS free of a link-time dependency on a particular loader, and of symbol clashes with a host
// that already exports the Vulkan names.
//
// Include this header only from translation units that implement the CoreGS backend.
#pragma once

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#define VK_USE_PLATFORM_WIN32_KHR
#endif
#define VK_NO_PROTOTYPES
#include <vulkan/vulkan.h>

#include <string>
#include <vector>

#include "hw_device_profile.hpp"

// Renames: the backend source keeps the Vulkan names; the preprocessor points them at the pointers below.
#define vkAcquireNextImageKHR coregs_hw_vkAcquireNextImageKHR
#define vkAllocateCommandBuffers coregs_hw_vkAllocateCommandBuffers
#define vkBeginCommandBuffer coregs_hw_vkBeginCommandBuffer
#define vkCmdClearColorImage coregs_hw_vkCmdClearColorImage
#define vkCmdPipelineBarrier coregs_hw_vkCmdPipelineBarrier
#define vkCreateCommandPool coregs_hw_vkCreateCommandPool
#define vkCreateDevice coregs_hw_vkCreateDevice
#define vkCreateFence coregs_hw_vkCreateFence
#define vkCreateInstance coregs_hw_vkCreateInstance
#define vkCreateSemaphore coregs_hw_vkCreateSemaphore
#define vkCreateSwapchainKHR coregs_hw_vkCreateSwapchainKHR
#define vkCreateWin32SurfaceKHR coregs_hw_vkCreateWin32SurfaceKHR
#define vkDestroyCommandPool coregs_hw_vkDestroyCommandPool
#define vkDestroyDevice coregs_hw_vkDestroyDevice
#define vkDestroyFence coregs_hw_vkDestroyFence
#define vkDestroyInstance coregs_hw_vkDestroyInstance
#define vkDestroySemaphore coregs_hw_vkDestroySemaphore
#define vkDestroySurfaceKHR coregs_hw_vkDestroySurfaceKHR
#define vkDestroySwapchainKHR coregs_hw_vkDestroySwapchainKHR
#define vkDeviceWaitIdle coregs_hw_vkDeviceWaitIdle
#define vkEndCommandBuffer coregs_hw_vkEndCommandBuffer
#define vkEnumerateDeviceExtensionProperties coregs_hw_vkEnumerateDeviceExtensionProperties
#define vkEnumeratePhysicalDevices coregs_hw_vkEnumeratePhysicalDevices
#define vkGetDeviceQueue coregs_hw_vkGetDeviceQueue
#define vkGetPhysicalDeviceFeatures2 coregs_hw_vkGetPhysicalDeviceFeatures2
#define vkGetPhysicalDeviceFormatProperties coregs_hw_vkGetPhysicalDeviceFormatProperties
#define vkGetPhysicalDeviceProperties coregs_hw_vkGetPhysicalDeviceProperties
#define vkGetPhysicalDeviceQueueFamilyProperties coregs_hw_vkGetPhysicalDeviceQueueFamilyProperties
#define vkGetPhysicalDeviceSurfaceCapabilitiesKHR coregs_hw_vkGetPhysicalDeviceSurfaceCapabilitiesKHR
#define vkGetPhysicalDeviceSurfaceFormatsKHR coregs_hw_vkGetPhysicalDeviceSurfaceFormatsKHR
#define vkGetPhysicalDeviceSurfacePresentModesKHR coregs_hw_vkGetPhysicalDeviceSurfacePresentModesKHR
#define vkGetPhysicalDeviceSurfaceSupportKHR coregs_hw_vkGetPhysicalDeviceSurfaceSupportKHR
#define vkGetSwapchainImagesKHR coregs_hw_vkGetSwapchainImagesKHR
#define vkQueuePresentKHR coregs_hw_vkQueuePresentKHR
#define vkQueueSubmit coregs_hw_vkQueueSubmit
#define vkResetCommandBuffer coregs_hw_vkResetCommandBuffer
#define vkResetFences coregs_hw_vkResetFences
#define vkWaitForFences coregs_hw_vkWaitForFences

inline PFN_vkAcquireNextImageKHR coregs_hw_vkAcquireNextImageKHR = nullptr;
inline PFN_vkAllocateCommandBuffers coregs_hw_vkAllocateCommandBuffers = nullptr;
inline PFN_vkBeginCommandBuffer coregs_hw_vkBeginCommandBuffer = nullptr;
inline PFN_vkCmdClearColorImage coregs_hw_vkCmdClearColorImage = nullptr;
inline PFN_vkCmdPipelineBarrier coregs_hw_vkCmdPipelineBarrier = nullptr;
inline PFN_vkCreateCommandPool coregs_hw_vkCreateCommandPool = nullptr;
inline PFN_vkCreateDevice coregs_hw_vkCreateDevice = nullptr;
inline PFN_vkCreateFence coregs_hw_vkCreateFence = nullptr;
inline PFN_vkCreateInstance coregs_hw_vkCreateInstance = nullptr;
inline PFN_vkCreateSemaphore coregs_hw_vkCreateSemaphore = nullptr;
inline PFN_vkCreateSwapchainKHR coregs_hw_vkCreateSwapchainKHR = nullptr;
inline PFN_vkCreateWin32SurfaceKHR coregs_hw_vkCreateWin32SurfaceKHR = nullptr;
inline PFN_vkDestroyCommandPool coregs_hw_vkDestroyCommandPool = nullptr;
inline PFN_vkDestroyDevice coregs_hw_vkDestroyDevice = nullptr;
inline PFN_vkDestroyFence coregs_hw_vkDestroyFence = nullptr;
inline PFN_vkDestroyInstance coregs_hw_vkDestroyInstance = nullptr;
inline PFN_vkDestroySemaphore coregs_hw_vkDestroySemaphore = nullptr;
inline PFN_vkDestroySurfaceKHR coregs_hw_vkDestroySurfaceKHR = nullptr;
inline PFN_vkDestroySwapchainKHR coregs_hw_vkDestroySwapchainKHR = nullptr;
inline PFN_vkDeviceWaitIdle coregs_hw_vkDeviceWaitIdle = nullptr;
inline PFN_vkEndCommandBuffer coregs_hw_vkEndCommandBuffer = nullptr;
inline PFN_vkEnumerateDeviceExtensionProperties coregs_hw_vkEnumerateDeviceExtensionProperties = nullptr;
inline PFN_vkEnumeratePhysicalDevices coregs_hw_vkEnumeratePhysicalDevices = nullptr;
inline PFN_vkGetDeviceQueue coregs_hw_vkGetDeviceQueue = nullptr;
inline PFN_vkGetPhysicalDeviceFeatures2 coregs_hw_vkGetPhysicalDeviceFeatures2 = nullptr;
inline PFN_vkGetPhysicalDeviceFormatProperties coregs_hw_vkGetPhysicalDeviceFormatProperties = nullptr;
inline PFN_vkGetPhysicalDeviceProperties coregs_hw_vkGetPhysicalDeviceProperties = nullptr;
inline PFN_vkGetPhysicalDeviceQueueFamilyProperties coregs_hw_vkGetPhysicalDeviceQueueFamilyProperties = nullptr;
inline PFN_vkGetPhysicalDeviceSurfaceCapabilitiesKHR coregs_hw_vkGetPhysicalDeviceSurfaceCapabilitiesKHR = nullptr;
inline PFN_vkGetPhysicalDeviceSurfaceFormatsKHR coregs_hw_vkGetPhysicalDeviceSurfaceFormatsKHR = nullptr;
inline PFN_vkGetPhysicalDeviceSurfacePresentModesKHR coregs_hw_vkGetPhysicalDeviceSurfacePresentModesKHR = nullptr;
inline PFN_vkGetPhysicalDeviceSurfaceSupportKHR coregs_hw_vkGetPhysicalDeviceSurfaceSupportKHR = nullptr;
inline PFN_vkGetSwapchainImagesKHR coregs_hw_vkGetSwapchainImagesKHR = nullptr;
inline PFN_vkQueuePresentKHR coregs_hw_vkQueuePresentKHR = nullptr;
inline PFN_vkQueueSubmit coregs_hw_vkQueueSubmit = nullptr;
inline PFN_vkResetCommandBuffer coregs_hw_vkResetCommandBuffer = nullptr;
inline PFN_vkResetFences coregs_hw_vkResetFences = nullptr;
inline PFN_vkWaitForFences coregs_hw_vkWaitForFences = nullptr;

namespace coregs::vulkan_loader {

// Loads the system Vulkan loader and vkGetInstanceProcAddr, then vkCreateInstance. False with `error` on failure.
bool load_loader(std::string &error);

// Fills every instance-level pointer for `instance`. Names the entry points that did not resolve.
bool load_instance_entry_points(VkInstance instance, std::vector<std::string> &missing);

// Fills only the entry points a windowless device needs (no surface, no swapchain). False with the
// missing names when one did not resolve; used by the headless fixtures and tools.
bool load_headless_entry_points(VkInstance instance, std::vector<std::string> &missing);

// VK_EXT_fragment_shader_interlock with fragmentShaderPixelInterlock on `physical`. A device that
// has it is created with the extension and the feature so the pixel shader can
// order the read-modify-write of overlapping fragments in primitive order inside one draw.
bool pixel_interlock_supported(VkPhysicalDevice physical);

// An entry point outside the tables above (diagnostics), null when it does not resolve.
PFN_vkVoidFunction instance_proc(VkInstance instance, const char *name);

// The capability probe (hw_device_profile.hpp): identity, class, the features drawing at scale relies on and every
// memory heap with its VK_EXT_memory_budget budget. Fills the `*_supported` fields; the `*_enabled` ones stay false
// until enable_profile_features. COREGS_HW_PRETEND_MISSING=large_points,dynamic_indexing (diagnostics) reports the
// named features as missing, so their fallbacks run on a device that has them.
coregs::DeviceProfile probe_device_profile(VkInstance instance, VkPhysicalDevice physical);

// Turns on in `features` what the profile says the device has and the renderer uses (largePoints,
// shaderStorageImageArrayDynamicIndexing) and records that in the profile's `*_enabled` fields.
void enable_profile_features(coregs::DeviceProfile &profile, VkPhysicalDeviceFeatures &features);

// VK_EXT_memory_budget, to be enabled on the device when the profile found it (null otherwise).
const char *memory_budget_extension(const coregs::DeviceProfile &profile);

} // namespace coregs::vulkan_loader
