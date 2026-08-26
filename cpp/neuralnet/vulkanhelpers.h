/**
 * @file vulkanhelpers.h
 * @author Dohoon Kim(https://github.com/dhkim92-dev)
 * @brief Vulkan Helper headers
 */
#ifdef USE_VULKAN_BACKEND
#pragma once

#include <string>
#include <vulkan/vulkan.h>
#include <vector>
#include "../external/vulkan/vk_mem_alloc.h"
#include "../core/global.h"
#include "../core/logger.h"

struct VulkanDeviceInfo {
  uint32_t deviceId;
  VkPhysicalDevice physicalDevice;
  std::string vendor;
  VkPhysicalDeviceType deviceType;
  std::string deviceName;
  VkPhysicalDeviceFeatures features;
  VkPhysicalDeviceProperties properties;
  VkPhysicalDeviceMemoryProperties memoryProperties;
  VkPhysicalDeviceShaderFloat16Int8Features shaderFloat16Int8Features;
  VkPhysicalDeviceCooperativeMatrixFeaturesKHR cooperativeMatrixFeatures;
  VkPhysicalDeviceMaintenance4FeaturesKHR maintenance4Features;
};

struct VulkanDevice {
  VulkanDeviceInfo info;
  VkDevice device;
  VkQueue queue;
  VkDescriptorPool descriptorPool;
  VkCommandPool commandPool;
  VmaAllocator allocator;
  bool skipWaitOnDestruction = false;
  ~VulkanDevice();
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
  const VulkanDevice* device;
  VkBuffer buffer;
  VmaAllocation allocation;
  VmaAllocationInfo allocationInfo;
  VkDeviceSize requestedSize;
  VulkanBuffer() = default;
  ~VulkanBuffer() = default;
  VulkanBuffer(const VulkanBuffer&) = delete;
  VulkanBuffer& operator=(const VulkanBuffer&) = delete;
};

struct WriteDescriptorSet {
  VkDescriptorBufferInfo bufferInfo;
  VkWriteDescriptorSet vkWriteDescriptorSet;
};


struct Pipeline {
  VkPipelineLayout layout = VK_NULL_HANDLE;
  VkPipeline pipeline = VK_NULL_HANDLE;
  VkDescriptorSetLayout descriptorSetLayout = VK_NULL_HANDLE;
};

/**
 * @brief Vulkan Utility functions for instance and error handling
 */
namespace vk_helper {

  size_t roundUpToMultiple(size_t size, size_t ofThis);

  int roundUpToMultipleInt(int size, int ofThis);

  size_t powerOf2ify(size_t size);

  std::string vkErrorToString(VkResult res);

  std::string vkPhysicalDeviceTypeToString(VkPhysicalDeviceType type);

  VkInstance createVulkanInstance();

  std::vector<VulkanDeviceInfo> enumerateVulkanDevices(VkInstance instance,Logger* logger);

  VulkanDevice* createVulkanDevice(
    VkInstance instance,
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

  inline WriteDescriptorSet writeDescriptorSetBuffer(
    VkDescriptorSet dstSet,
    uint32_t dstBinding,
    const VulkanBuffer *buffer,
    VkDescriptorType descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
    uint32_t descriptorCount = 1
  ) {
    WriteDescriptorSet result;
    result.bufferInfo.buffer = buffer->buffer;
    result.bufferInfo.offset = 0;
    result.bufferInfo.range = buffer->requestedSize;
    result.vkWriteDescriptorSet.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    result.vkWriteDescriptorSet.dstSet = dstSet;
    result.vkWriteDescriptorSet.dstBinding = dstBinding;
    result.vkWriteDescriptorSet.dstArrayElement = 0;
    result.vkWriteDescriptorSet.descriptorType = descriptorType;
    result.vkWriteDescriptorSet.descriptorCount = descriptorCount;
    result.vkWriteDescriptorSet.pBufferInfo = &result.bufferInfo;
    result.vkWriteDescriptorSet.pImageInfo = nullptr;
    result.vkWriteDescriptorSet.pTexelBufferView = nullptr;
    result.vkWriteDescriptorSet.pNext = nullptr;
    return result;
  }

  VkDescriptorPool createDescriptorPool(
    const VulkanDevice *device,
    const std::vector<VkDescriptorPoolSize>& poolSizes,
    uint32_t maxSets,
    VkResult *result
  );

  VkDescriptorSet allocateDescriptorSet(
    const VulkanDevice *device,
    VkDescriptorSetLayout descriptorSetLayout,
    VkResult *result
  );

  VkResult updateDescriptorSets(
    const VulkanDevice *device,
    const std::vector<WriteDescriptorSet>& writeDescriptorSets
  );

  VkCommandPool createCommandPool(
    const VulkanDevice *device,
    VkResult *result
  );

  VkCommandBuffer allocateCommandBuffer(
    const VulkanDevice *device,
    VkResult* result = nullptr
  );

  VkResult beginCommandBuffer(
    VkCommandBuffer commandBuffer
  );

  VkResult endCommandBuffer(
    VkCommandBuffer commandBuffer
  );

  VkResult submitCommandBuffers(
    const VulkanDevice *device,
    const std::vector<VkCommandBuffer>& commandBuffers,
    VkFence fence = VK_NULL_HANDLE
  );

  VkCommandBuffer beginSingleTimeCommandBuffer(
    const VulkanDevice *device
  );

  void submitSingleTimeCommandBufferAndWaitIdle(
    const VulkanDevice *device,
    VkCommandBuffer commandBuffer
  );

  VulkanBuffer* createDeviceBuffer(
    const VulkanDevice* device,
    size_t size,
    bool readOnly,
    VkResult *result
  );

  VulkanBuffer* createDeviceBufferWithData(
    const VulkanDevice* device,
    size_t size,
    const void* data,
    bool readOnly,
    VkResult *result
  );

  VulkanBuffer* createStagingBuffer(
    const VulkanDevice* device,
    size_t size,
    VkResult *result
  );

  VulkanBuffer* createReadbackBuffer(
    const VulkanDevice* device,
    size_t size,
    VkResult *result
  );

  void copyDeviceBufferToHost(
    const VulkanDevice* device,
    VulkanBuffer* deviceBuffer,
    VkDeviceSize copySize,
    void* hostPtr,
    bool waitForIdle,
    VkResult *result
  );

  void copyHostToDeviceBuffer(
    const VulkanDevice* device,
    const void* hostPtr,
    VulkanBuffer* deviceBuffer,
    VkDeviceSize copySize,
    bool waitForIdle,
    VkResult *result
  );

  void releaseVulkanBuffer(
    const VulkanDevice* device,
    VulkanBuffer* buffer
  );

  VkFence createFence(const VulkanDevice* device, VkResult* result);

  VkResult resetFence(const VulkanDevice* device, VkFence fence);

  void destroyFence(const VulkanDevice* device, VkFence fence);

  void barrierCommandBuffer(
    VkCommandBuffer commandBuffer,
    VkPipelineStageFlags srcStageMask = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 
    VkAccessFlags srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
    VkPipelineStageFlags dstStageMask = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
    VkAccessFlags dstAccessMask = VK_ACCESS_SHADER_READ_BIT
  );

  void barrierCommandBufferForBuffer(
    VkCommandBuffer commandBuffer,
    VulkanBuffer* buffer,
    VkPipelineStageFlags srcStageMask = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 
    VkAccessFlags srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
    VkPipelineStageFlags dstStageMask = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
    VkAccessFlags dstAccessMask = VK_ACCESS_SHADER_READ_BIT
  );

  std::vector<int32_t> createSpecData(void* data, size_t dataSize);

  std::vector<VkSpecializationMapEntry> createSpecMapEntries(size_t dataCount);

  VkSpecializationInfo createSpecializationInfo(
    const std::vector<int32_t>& specData,
    const std::vector<VkSpecializationMapEntry>& specMapEntries
  );

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
    std::cerr << "[Vulkan error] " << vk_helper::vkErrorToString(res) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
    exit(EXIT_FAILURE); \
  } \
}

#define CHECK_VK_MSG(msg, res) { \
  if(res != VK_SUCCESS) { \
    std::cerr << "[Vulkan error] " << vk_helper::vkErrorToString(res) << " at " << __FILE__ << ":" << __LINE__ << " - " << msg << std::endl; \
    exit(EXIT_FAILURE); \
  } \
}


#endif
