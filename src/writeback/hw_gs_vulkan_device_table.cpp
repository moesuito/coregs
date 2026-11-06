// MIT License. Copyright (c) 2026 João Alano and GT4Recomp contributors.
// Written from the Vulkan specification, clean room (decision 0069). See the header for the naming rule.
#include "hw_gs_vulkan_device_table.hpp"

namespace coregs::vulkan_device {
namespace {

template <typename Pointer>
void resolve(Pointer &slot, PFN_vkGetInstanceProcAddr proc, VkInstance instance, const char *name,
             std::vector<std::string> &missing) {
  slot = reinterpret_cast<Pointer>(proc(instance, name));
  if (!slot) missing.push_back(name);
}

} // namespace

bool load_device_entry_points(VkInstance instance, std::vector<std::string> &missing) {
  HMODULE library = GetModuleHandleW(L"vulkan-1.dll");
  if (!library) {
    missing.push_back("vulkan-1.dll (not loaded)");
    return false;
  }
  auto proc = reinterpret_cast<PFN_vkGetInstanceProcAddr>(GetProcAddress(library, "vkGetInstanceProcAddr"));
  if (!proc) {
    missing.push_back("vkGetInstanceProcAddr");
    return false;
  }
  resolve(coregs_hwd_vkAllocateDescriptorSets, proc, instance, "vkAllocateDescriptorSets", missing);
  resolve(coregs_hwd_vkAllocateMemory, proc, instance, "vkAllocateMemory", missing);
  resolve(coregs_hwd_vkBindBufferMemory, proc, instance, "vkBindBufferMemory", missing);
  resolve(coregs_hwd_vkBindImageMemory, proc, instance, "vkBindImageMemory", missing);
  resolve(coregs_hwd_vkCmdBeginRenderPass, proc, instance, "vkCmdBeginRenderPass", missing);
  resolve(coregs_hwd_vkCmdBindDescriptorSets, proc, instance, "vkCmdBindDescriptorSets", missing);
  resolve(coregs_hwd_vkCmdBindPipeline, proc, instance, "vkCmdBindPipeline", missing);
  resolve(coregs_hwd_vkCmdBindVertexBuffers, proc, instance, "vkCmdBindVertexBuffers", missing);
  resolve(coregs_hwd_vkCmdBlitImage, proc, instance, "vkCmdBlitImage", missing);
  resolve(coregs_hwd_vkCmdClearColorImage, proc, instance, "vkCmdClearColorImage", missing);
  resolve(coregs_hwd_vkCmdCopyBufferToImage, proc, instance, "vkCmdCopyBufferToImage", missing);
  resolve(coregs_hwd_vkCmdCopyImage, proc, instance, "vkCmdCopyImage", missing);
  resolve(coregs_hwd_vkCreateComputePipelines, proc, instance, "vkCreateComputePipelines", missing);
  resolve(coregs_hwd_vkCreatePipelineCache, proc, instance, "vkCreatePipelineCache", missing);
  resolve(coregs_hwd_vkDestroyPipelineCache, proc, instance, "vkDestroyPipelineCache", missing);
  resolve(coregs_hwd_vkGetPipelineCacheData, proc, instance, "vkGetPipelineCacheData", missing);
  resolve(coregs_hwd_vkCmdDispatch, proc, instance, "vkCmdDispatch", missing);
  resolve(coregs_hwd_vkCmdCopyBuffer, proc, instance, "vkCmdCopyBuffer", missing);
  resolve(coregs_hwd_vkCreateQueryPool, proc, instance, "vkCreateQueryPool", missing);
  resolve(coregs_hwd_vkDestroyQueryPool, proc, instance, "vkDestroyQueryPool", missing);
  resolve(coregs_hwd_vkCmdResetQueryPool, proc, instance, "vkCmdResetQueryPool", missing);
  resolve(coregs_hwd_vkCmdWriteTimestamp, proc, instance, "vkCmdWriteTimestamp", missing);
  resolve(coregs_hwd_vkGetQueryPoolResults, proc, instance, "vkGetQueryPoolResults", missing);
  resolve(coregs_hwd_vkCmdBeginQuery, proc, instance, "vkCmdBeginQuery", missing);
  resolve(coregs_hwd_vkCmdEndQuery, proc, instance, "vkCmdEndQuery", missing);
  resolve(coregs_hwd_vkCmdCopyImageToBuffer, proc, instance, "vkCmdCopyImageToBuffer", missing);
  resolve(coregs_hwd_vkCmdDraw, proc, instance, "vkCmdDraw", missing);
  resolve(coregs_hwd_vkCmdEndRenderPass, proc, instance, "vkCmdEndRenderPass", missing);
  resolve(coregs_hwd_vkCmdPushConstants, proc, instance, "vkCmdPushConstants", missing);
  resolve(coregs_hwd_vkCmdSetScissor, proc, instance, "vkCmdSetScissor", missing);
  resolve(coregs_hwd_vkCmdSetViewport, proc, instance, "vkCmdSetViewport", missing);
  resolve(coregs_hwd_vkCreateBuffer, proc, instance, "vkCreateBuffer", missing);
  resolve(coregs_hwd_vkCreateDescriptorPool, proc, instance, "vkCreateDescriptorPool", missing);
  resolve(coregs_hwd_vkCreateDescriptorSetLayout, proc, instance, "vkCreateDescriptorSetLayout", missing);
  resolve(coregs_hwd_vkCreateFramebuffer, proc, instance, "vkCreateFramebuffer", missing);
  resolve(coregs_hwd_vkCreateGraphicsPipelines, proc, instance, "vkCreateGraphicsPipelines", missing);
  resolve(coregs_hwd_vkCreateImage, proc, instance, "vkCreateImage", missing);
  resolve(coregs_hwd_vkCreateImageView, proc, instance, "vkCreateImageView", missing);
  resolve(coregs_hwd_vkCreatePipelineLayout, proc, instance, "vkCreatePipelineLayout", missing);
  resolve(coregs_hwd_vkCreateRenderPass, proc, instance, "vkCreateRenderPass", missing);
  resolve(coregs_hwd_vkCreateSampler, proc, instance, "vkCreateSampler", missing);
  resolve(coregs_hwd_vkDestroySampler, proc, instance, "vkDestroySampler", missing);
  resolve(coregs_hwd_vkCreateShaderModule, proc, instance, "vkCreateShaderModule", missing);
  resolve(coregs_hwd_vkDestroyBuffer, proc, instance, "vkDestroyBuffer", missing);
  resolve(coregs_hwd_vkDestroyDescriptorPool, proc, instance, "vkDestroyDescriptorPool", missing);
  resolve(coregs_hwd_vkDestroyDescriptorSetLayout, proc, instance, "vkDestroyDescriptorSetLayout", missing);
  resolve(coregs_hwd_vkDestroyFramebuffer, proc, instance, "vkDestroyFramebuffer", missing);
  resolve(coregs_hwd_vkDestroyImage, proc, instance, "vkDestroyImage", missing);
  resolve(coregs_hwd_vkDestroyImageView, proc, instance, "vkDestroyImageView", missing);
  resolve(coregs_hwd_vkDestroyPipeline, proc, instance, "vkDestroyPipeline", missing);
  resolve(coregs_hwd_vkDestroyPipelineLayout, proc, instance, "vkDestroyPipelineLayout", missing);
  resolve(coregs_hwd_vkDestroyRenderPass, proc, instance, "vkDestroyRenderPass", missing);
  resolve(coregs_hwd_vkDestroyShaderModule, proc, instance, "vkDestroyShaderModule", missing);
  resolve(coregs_hwd_vkFreeMemory, proc, instance, "vkFreeMemory", missing);
  resolve(coregs_hwd_vkGetBufferMemoryRequirements, proc, instance, "vkGetBufferMemoryRequirements", missing);
  resolve(coregs_hwd_vkGetImageMemoryRequirements, proc, instance, "vkGetImageMemoryRequirements", missing);
  resolve(coregs_hwd_vkGetPhysicalDeviceMemoryProperties, proc, instance, "vkGetPhysicalDeviceMemoryProperties",
          missing);
  resolve(coregs_hwd_vkMapMemory, proc, instance, "vkMapMemory", missing);
  resolve(coregs_hwd_vkUnmapMemory, proc, instance, "vkUnmapMemory", missing);
  resolve(coregs_hwd_vkUpdateDescriptorSets, proc, instance, "vkUpdateDescriptorSets", missing);
  resolve(coregs_hwd_vkResetDescriptorPool, proc, instance, "vkResetDescriptorPool", missing);
  return missing.empty();
}

} // namespace coregs::vulkan_device
