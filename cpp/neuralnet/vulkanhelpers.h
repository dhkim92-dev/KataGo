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

  std::string vkErrorToString(VkResult res);
  std::string vkPhysicalDeviceTypeToString(VkPhysicalDeviceType type);
  VkInstance createVulkanInstance();
  std::vector<VulkanDeviceInfo> enumerateVulkanDevices(VkInstance instance,Logger* logger);
  VulkanDevice* createVulkanDevice(
    VulkanDeviceInfo deviceInfo,
    std::vector<const char *> requiredExtensions,
    Logger* logger
  );

  VkShaderModule createShaderModuleFromSPIRVBytes(
    VkDevice device,
    const std::vector<unsigned char>& spirvBytes,
    VkResult *result
  );

  VkPipelineCache createPipelineCache(
    VkDevice device,
    VkResult *result
  );

  VkPipelineLayout createPipelineLayout(
    VkDevice device,
    const std::vector<VkDescriptorSetLayout>& descriptorSetLayouts,
    const std::vector<VkPushConstantRange>& pushConstantRanges,
    VkResult *result
  );

  VkPipeline createComputePipeline(
    VkDevice device,
    VkPipelineLayout pipelineLayout,
    VkPipelineCache pipelineCache,
    VkShaderModule computeShaderModule,
    VkResult* result,
    VkSpecializationInfo* specializationInfo = nullptr,
    std::string entryPointName = "main"
  );

  VkDescriptorSetLayout createDescriptorSetLayout(
    VkDevice device,
    const std::vector<VkDescriptorSetLayoutBinding>& bindings,
    VkResult *result
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

#define CHECK_VK_MSG(msg, res) { \
  if(res != VK_SUCCESS) { \
    std::cerr << "[Vulkan error] " << VkHelpers::vkErrorToString(res) << " at " << __FILE__ << ":" << __LINE__ << " - " << msg << std::endl; \
    exit(EXIT_FAILURE); \
  } \
}


#endif
