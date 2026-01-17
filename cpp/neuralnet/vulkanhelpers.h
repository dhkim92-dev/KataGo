#ifdef USE_VULKAN_BACKEND
#pragma once;

#include <string>
#include <vulkan/vulkan.h>
#include <vector>
#include <vma/vk_mem_alloc.h>
#include "../core/global.h"
#include "../core/logger.h"


struct VulkanDeviceInfo {
  uint32_t deviceId;
  VkPhysicalDevice physicalDevice;
  std::string vendor;
  VkPhysicalDeviceType deviceType;
  std::string deviceName;
  VkPhysicalDeviceFeatures features;
  VkPhysicalDeviceShaderFloat16Int8Features shaderFloat16Int8Features;
  VkPhysicalDeviceProperties properties;
  VkPhysicalDeviceMemoryProperties memoryProperties;
  VkPhysicalDeviceCooperativeMatrixFeaturesKHR cooperativeMatrixFeatures;
};

struct VulkanDevice {
  VulkanDeviceInfo info;
  VkDevice device;
  VkQueue queue;
};

struct VulkanContext {
  VkInstance instance = VK_NULL_HANDLE;
  uint32_t defaultGpuIdx;
  std::vector<VulkanDevice*> devicesToUse;
  std::vector<std::string> uniqueDeviceNamesToUse;
  VkDebugUtilsMessengerEXT debugMessenger = VK_NULL_HANDLE;

  VulkanContext(
    VkInstance instance, 
    const std::vector<VulkanDevice *>& devicesToUse, 
    Logger* logger
  );
  ~VulkanContext();
  VulkanContext() = delete;
  VulkanContext(const VulkanContext&) = delete;
  VulkanContext& operator=(const VulkanContext&) = delete;

  const VulkanDevice* findGpuExn(int gpuIdx) const;
  std::vector<const VulkanDevice*> findDevicesToUseWithName(const std::string& name) const;
  std::vector<VkDevice> findDeviceIdsToUseWithName(const std::string& name) const;
};

struct VulkanBuffer {
  VkDevice device;
  VkBuffer buffer;
  VmaAllocation allocation;
  VmaAllocationInfo allocationInfo;
  size_t size;

  VulkanBuffer(
      VkDevice device,
      VkDeviceSize size,
      VmaMemoryUsage memoryUsage
  );
  ~VulkanBuffer();
  VulkanBuffer() = delete;
  VulkanBuffer(const VulkanBuffer&) = delete;
  VulkanBuffer& operator=(const VulkanBuffer&) = delete;
};

/**
 * @brief Vulkan Utility functions for instance and error handling
 */
namespace VkHelpers {

  VkInstance createVulkanInstance();
  std::vector<VulkanDeviceInfo> enumerateVulkanDevices(VkInstance instance,Logger* logger);
  std::string vkErrorToString(VkResult res);
  std::string vkPhysicalDeviceTypeToString(VkPhysicalDeviceType type);
  VulkanDevice* createVulkanDevice(
    VulkanDeviceInfo deviceInfo,
    std::vector<const char *> requiredExtensions,
    Logger* logger
  );

  VkShaderModule createShaderModuleFromSPIRVBytes(
    VkDevice device,
    const std::vector<unsigned char>& spirvBytes,
    VkResult *res
  );

  VkPipelineCache createPipelineCache(
    VkDevice device,
    VkResult *res
  );

  VkPipelineLayout createPipelineLayout(
    VkDevice device,
    const std::vector<VkDescriptorSetLayout>& descriptorSetLayouts,
    const std::vector<VkPushConstantRange>& pushConstantRanges,
    VkResult *res
  );

  VkPipeline createComputePipeline(
    VkDevice device,
    VkPipelineLayout pipelineLayout,
    VkPipelineCache pipelineCache,
    VkShaderModule computeShaderModule,
    VkResult* res,
    const char* entryPointName = "main"
  );

  VkDescriptorSetLayout createDescriptorSetLayout(
    VkDevice device,
    const std::vector<VkDescriptorSetLayoutBinding>& bindings,
    VkResult *res
  );

  inline VkDescriptorSetLayoutBinding descriptorSetLayoutBinding(
    uint32_t binding,
    VkDescriptorType descriptorType,
    VkShaderStageFlags stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
    uint32_t descriptorCount = 1
  ) {
    VkDescriptorSetLayoutBinding layoutBinding = {};
    layoutBinding.binding = binding;
    layoutBinding.descriptorType = descriptorType;
    layoutBinding.descriptorCount = descriptorCount;
    layoutBinding.stageFlags = stageFlags;
    layoutBinding.pImmutableSamplers = nullptr;
    return layoutBinding;
  }
};

namespace VkDebug {
  static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(
    VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
    VkDebugUtilsMessageTypeFlagsEXT messageType,
    const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
    void* pUserData
  );
}

#define CHECK_VK(res) { \
  if(res != VK_SUCCESS) { \
    std::cerr << "[Vulkan error] " << VkHelpers::vkErrorToString(res) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
    exit(EXIT_FAILURE); \
  } \
}


#endif
