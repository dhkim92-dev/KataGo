#ifdef USE_VULKAN_BACKEND
#pragma once;

#include <string>
#include <vulkan/vulkan.h>
#include <vector>

#ifndef VMA_IMPLEMENTATION
#define VMA_IMPLEMENTATION
#include "../external/vulkan/vk_mem_alloc.h"
#endif
#include "../core/global.h"
#include "../core/logger.h"
#include "../neuralnet/vulkanhelpers.h"

VulkanDevice::~VulkanDevice() {
  if (this->device != VK_NULL_HANDLE) {
    vkQueueWaitIdle(this->queue);
    vkDeviceWaitIdle(this->device);
    vmaDestroyAllocator(this->allocator);

    if (this->descriptorPool != VK_NULL_HANDLE) {
      vkResetDescriptorPool(this->device, this->descriptorPool, 0);
      vkDestroyDescriptorPool(this->device, this->descriptorPool, nullptr);
      this->descriptorPool = VK_NULL_HANDLE;
    }

    if (this->commandPool != VK_NULL_HANDLE) {
      vkResetCommandPool(this->device, this->commandPool, 0);
      vkDestroyCommandPool(this->device, this->commandPool, nullptr);
      this->commandPool = VK_NULL_HANDLE;
    }
    vkDestroyDevice(this->device, nullptr);
    this->device = VK_NULL_HANDLE;
  }
}

VulkanContext::VulkanContext(
  VkInstance instance,
  const std::vector<VulkanDevice *>& devicesToUse,
  Logger* logger
) {
  this->instance = instance;
  this->devicesToUse = devicesToUse;
  this->defaultGpuIdx = 0;
  for ( const VulkanDevice* device : devicesToUse ) {
    if ( std::find(this->uniqueDeviceNamesToUse.begin(), this->uniqueDeviceNamesToUse.end(), device->info.deviceName) == this->uniqueDeviceNamesToUse.end() ) {
      this->uniqueDeviceNamesToUse.push_back(device->info.deviceName);
    }
  }

  #ifdef VULKAN_API_DEBUG
  // Setup debug messenger
  VkDebugUtilsMessengerCreateInfoEXT debugCreateInfo = {};
  debugCreateInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
  debugCreateInfo.messageSeverity = 
    VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT | 
    VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | 
    VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
  debugCreateInfo.messageType = 
    VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
    VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
    VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
  debugCreateInfo.pfnUserCallback = nullptr; // You can set a callback function here if needed
  debugCreateInfo.pUserData = nullptr; // Optional user data
  auto func = (PFN_vkCreateDebugUtilsMessengerEXT)vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT");
  if (func != nullptr) {
    VkResult res = func(instance, &debugCreateInfo, nullptr, &this->debugMessenger);
    CHECK_VK_MSG("CreateDebugUtilsMessengerEXT", res);
  } else {
    logger->write("Warning: Could not set up Vulkan debug messenger.");
  }
  #endif
}

VulkanContext::~VulkanContext() {
  for ( VulkanDevice* device : this->devicesToUse ) {
    delete device;
    device = nullptr;
  }
  this->devicesToUse.clear();
  if (this->instance != VK_NULL_HANDLE) {
    vkDestroyInstance(this->instance, nullptr);
    this->instance = VK_NULL_HANDLE;
  }
}

const VulkanDevice* VulkanContext::findGpuExn(int gpuIdx) const {
  for ( const VulkanDevice* device : this->devicesToUse ) {
    if ( device->info.deviceId == static_cast<uint32_t>(gpuIdx) ) {
      return device;
    }
  }
  throw StringError("Could not find Vulkan device with GPU index " + std::to_string(gpuIdx));
}

std::vector<const VulkanDevice*> VulkanContext::findDevicesToUseWithName(const std::string& name) const {
  std::vector<const VulkanDevice*> devices;
  for ( const VulkanDevice* device : this->devicesToUse ) {
    if ( device->info.deviceName == name ) {
      devices.push_back(device);
    }
  }
  return devices;
}

std::vector<VkDevice> VulkanContext::findDeviceIdsToUseWithName(const std::string& name) const {
  std::vector<VkDevice> deviceIds;
  for ( const VulkanDevice* device : this->devicesToUse ) {
    if ( device->info.deviceName == name ) {
      deviceIds.push_back(device->device);
    }
  }
  return deviceIds;
}

size_t VkHelpers::roundUpToMultiple(size_t size, size_t ofThis) {
  return (size + ofThis - 1) / ofThis * ofThis;
}

size_t VkHelpers::powerOf2ify(size_t size) {
  if(size <= 2)
    return size;
  if(size <= 4)
    return 4;
  size_t s = 1;
  while(s * 4 < size)
    s *= 2;

  if(s >= size)
    return s;
  if(s * 2 >= size)
    return s * 2;
  if(s * 3 >= size)
    return s * 3;
  assert(s * 4 >= size);
  return s * 4;
}

std::string VkHelpers::vkErrorToString(VkResult res) {
  switch(res) {
    case VK_SUCCESS: return "VK_SUCCESS";
    case VK_NOT_READY: return "VK_NOT_READY";
    case VK_TIMEOUT: return "VK_TIMEOUT";
    case VK_EVENT_SET: return "VK_EVENT_SET";
    case VK_EVENT_RESET: return "VK_EVENT_RESET";
    case VK_INCOMPLETE: return "VK_INCOMPLETE";
    case VK_ERROR_OUT_OF_HOST_MEMORY: return "VK_ERROR_OUT_OF_HOST_MEMORY";
    case VK_ERROR_OUT_OF_DEVICE_MEMORY: return "VK_ERROR_OUT_OF_DEVICE_MEMORY";
    case VK_ERROR_INITIALIZATION_FAILED: return "VK_ERROR_INITIALIZATION_FAILED";
    case VK_ERROR_DEVICE_LOST: return "VK_ERROR_DEVICE_LOST";
    case VK_ERROR_MEMORY_MAP_FAILED: return "VK_ERROR_MEMORY_MAP_FAILED";
    case VK_ERROR_LAYER_NOT_PRESENT: return "VK_ERROR_LAYER_NOT_PRESENT";
    case VK_ERROR_EXTENSION_NOT_PRESENT: return "VK_ERROR_EXTENSION_NOT_PRESENT";
    case VK_ERROR_FEATURE_NOT_PRESENT: return "VK_ERROR_FEATURE_NOT_PRESENT";
    case VK_ERROR_INCOMPATIBLE_DRIVER: return "VK_ERROR_INCOMPATIBLE_DRIVER";
    case VK_ERROR_TOO_MANY_OBJECTS: return "VK_ERROR_TOO_MANY_OBJECTS";
    case VK_ERROR_FORMAT_NOT_SUPPORTED: return "VK_ERROR_FORMAT_NOT_SUPPORTED";
    case VK_ERROR_FRAGMENTED_POOL: return "VK_ERROR_FRAGMENTED_POOL";
    // Add more cases as needed
    default: return "Unknown VkResult error code";
  }
}

std::string VkHelpers::vkPhysicalDeviceTypeToString(VkPhysicalDeviceType type) {
  switch(type) {
    case VK_PHYSICAL_DEVICE_TYPE_OTHER: return "Other";
    case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU: return "Integrated GPU";
    case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU: return "Discrete GPU";
    case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU: return "Virtual GPU";
    case VK_PHYSICAL_DEVICE_TYPE_CPU: return "CPU";
    default: return "Unknown Device Type";
  }
}

VkInstance VkHelpers::createVulkanInstance() {
  VkInstance instance = VK_NULL_HANDLE;
  std::vector<const char*> requiredLayers = {
#ifdef VULKAN_API_DEBUG
    "VK_LAYER_KHRONOS_validation",
#endif
  };

  std::vector<const char*> requiredExtensions = {
#ifdef VULKAN_API_DEBUG
    VK_EXT_DEBUG_UTILS_EXTENSION_NAME,
#endif
#ifdef __MACOS__
    "VK_KHR_portability_enumeration",
    "VK_KHR_get_physical_device_properties2",
#endif
  };

  VkApplicationInfo appInfo = {};
  appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
  appInfo.pApplicationName = "KataGo";
  appInfo.applicationVersion = VK_MAKE_VERSION(1, 6, 4);
  appInfo.pEngineName = "Katago Vulkan Backend";
  appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
  appInfo.apiVersion = VK_API_VERSION_1_2;
  VkInstanceCreateInfo instanceCI = {};
  instanceCI.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
  instanceCI.pApplicationInfo = &appInfo;
  instanceCI.pNext = nullptr;
  instanceCI.enabledLayerCount = static_cast<uint32_t>(requiredLayers.size());
  instanceCI.ppEnabledLayerNames = requiredLayers.data();
  instanceCI.enabledExtensionCount = static_cast<uint32_t>(requiredExtensions.size());
  instanceCI.ppEnabledExtensionNames = requiredExtensions.data();
#ifdef __MACOS__
  instanceCI.flags = VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;
#else
  instanceCI.flags = 0;
#endif
  VkResult res = vkCreateInstance(&instanceCI, nullptr, &instance);
  CHECK_VK(res);
  return instance;
}

std::vector<VulkanDeviceInfo> VkHelpers::enumerateVulkanDevices(VkInstance instance, Logger* logger) {
  uint32_t numDevices = 0;
  VkResult res = vkEnumeratePhysicalDevices(instance, &numDevices, nullptr);
  CHECK_VK(res);
  if(numDevices == 0) {
    throw StringError("No Vulkan-compatible GPU found");
  }
  std::vector<VkPhysicalDevice> physicalDevices(numDevices);
  res = vkEnumeratePhysicalDevices(instance, &numDevices, physicalDevices.data());
  CHECK_VK(res);

  std::vector<VulkanDeviceInfo> deviceInfos;
  for(uint32_t i = 0; i < numDevices; i++) {
    logger->write("Found Vulkan Physical Device[" + Global::uint32ToString(i) + "]");
    VkPhysicalDevice physicalDevice = physicalDevices[i];
    VulkanDeviceInfo deviceInfo;
    deviceInfo.deviceId = i;
    deviceInfo.physicalDevice = physicalDevice;

    // Get device properties
    VkPhysicalDeviceProperties2 properties2 = {};
    properties2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
    vkGetPhysicalDeviceProperties2(physicalDevice, &properties2);
    deviceInfo.properties = properties2.properties;
    deviceInfo.deviceType = deviceInfo.properties.deviceType;

    // Get device name
    deviceInfo.deviceName = std::string(deviceInfo.properties.deviceName);
    logger->write("  Name: " + deviceInfo.deviceName);
    logger->write("  Type: " + VkHelpers::vkPhysicalDeviceTypeToString(deviceInfo.deviceType));
    logger->write("  API Version: " +
      std::to_string(VK_VERSION_MAJOR(deviceInfo.properties.apiVersion)) + "." +
      std::to_string(VK_VERSION_MINOR(deviceInfo.properties.apiVersion)) + "." +
      std::to_string(VK_VERSION_PATCH(deviceInfo.properties.apiVersion))
    );
    // Get device features
    VkPhysicalDeviceFeatures2 features2 = {};
    // Get shader float16/int8 features
    VkPhysicalDeviceShaderFloat16Int8Features shaderFloat16Int8Features = {};
    shaderFloat16Int8Features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES;
    features2.pNext = &shaderFloat16Int8Features;
    vkGetPhysicalDeviceFeatures2(physicalDevice, &features2);
    deviceInfo.shaderFloat16Int8Features = shaderFloat16Int8Features;
    logger->write("  Shader Float16 Support: " + std::string(deviceInfo.shaderFloat16Int8Features.shaderFloat16 == VK_TRUE ? "Yes" : "No")); 

    // Get memory properties2
    VkPhysicalDeviceMemoryProperties2 memoryProperties2 = {};
    memoryProperties2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2;
    // Cooperative Matrix features
    VkPhysicalDeviceCooperativeMatrixFeaturesKHR cooperativeMatrixFeatures = {};
    cooperativeMatrixFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR;
    memoryProperties2.pNext = &cooperativeMatrixFeatures;
    vkGetPhysicalDeviceMemoryProperties2(physicalDevice, &memoryProperties2);
    deviceInfo.memoryProperties = memoryProperties2.memoryProperties;
    deviceInfo.cooperativeMatrixFeatures = cooperativeMatrixFeatures;
    logger->write("  Cooperative Matrix Support: " + std::string(deviceInfo.cooperativeMatrixFeatures.cooperativeMatrix == VK_TRUE ? "Yes" : "No"));

    deviceInfos.push_back(deviceInfo);
  }

  return deviceInfos;
}

VulkanDevice* VkHelpers::createVulkanDevice(
  VkInstance instance,
  VulkanDeviceInfo deviceInfo,
  std::vector<const char *> requiredExtensions,
  Logger* logger
) {
  VkPhysicalDevice physicalDevice = deviceInfo.physicalDevice;

  // Create logical device
  
  std::vector<VkExtensionProperties> availableExtensions;
  uint32_t extensionCount = 0;
  vkEnumerateDeviceExtensionProperties(physicalDevice, nullptr, &extensionCount, nullptr);
  if(extensionCount > 0) {
    availableExtensions.resize(extensionCount);
    vkEnumerateDeviceExtensionProperties(physicalDevice, nullptr, &extensionCount, availableExtensions.data());
  } else {
    logger->write("No device extensions available");
  }

  for (const char* requiredExt : requiredExtensions) {
    bool found = false;
    for (const auto& availExt : availableExtensions) {
      if (std::string(requiredExt) == std::string(availExt.extensionName)) {
        found = true;
        break;
      }
    }
    if (!found) {
      throw StringError("Required device extension " + std::string(requiredExt) + " is not available on device " + deviceInfo.deviceName);
    }
  }

  float queuePriority = 1.0f;
  VkDeviceQueueCreateInfo queueCI = {};
  queueCI.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
  queueCI.queueFamilyIndex = 0; // TODO: select proper queue family index, this code assumes index 0 supports compute
  queueCI.queueCount = 1;
  queueCI.pQueuePriorities = &queuePriority;

  VkDeviceCreateInfo deviceCI = {};
  deviceCI.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
  deviceCI.pNext = &deviceInfo.shaderFloat16Int8Features;
  deviceCI.queueCreateInfoCount = 1;
  deviceCI.pQueueCreateInfos = &queueCI;
  deviceCI.enabledExtensionCount = static_cast<uint32_t>(requiredExtensions.size());
  deviceCI.ppEnabledExtensionNames = requiredExtensions.data();

  VkDevice device;
  VkResult res = vkCreateDevice(physicalDevice, &deviceCI, nullptr, &device);
  CHECK_VK(res);
  logger->write("Created Vulkan Logical Device for " + deviceInfo.deviceName);

  // Get compute queue
  VkQueue queue;
  vkGetDeviceQueue(device, queueCI.queueFamilyIndex, 0, &queue);

  VulkanDevice* vulkanDevice = new VulkanDevice();
  vulkanDevice->info = deviceInfo;
  vulkanDevice->device = device;
  vulkanDevice->queue = queue;

  VmaAllocator allocator = VK_NULL_HANDLE;

  VmaAllocatorCreateInfo allocatorCI = {};
  allocatorCI.physicalDevice = physicalDevice;
  allocatorCI.device = device;
  allocatorCI.instance = instance;

#ifndef VK_API_VERSION_1_3
  allocatorCI.vulkanApiVersion = VK_API_VERSION_1_2;
  allocatorCI.flags = VMA_ALLOCATOR_CREATE_EXTERNALLY_SYNCHRONIZED_BIT;= 
#else
  allocatorCI.vulkanApiVersion = VK_API_VERSION_1_3;
#endif

  res = vmaCreateAllocator(&allocatorCI, &allocator);
  CHECK_VK_MSG("VMA create for device : " + deviceInfo.deviceName, res);
  vulkanDevice->allocator = allocator;

  vulkanDevice->commandPool = VkHelpers::createCommandPool(
    vulkanDevice,
    &res
  );
  CHECK_VK_MSG("CreateCommandPool for device : " + deviceInfo.deviceName, res);

  vulkanDevice->descriptorPool = VkHelpers::createDescriptorPool(
    vulkanDevice,
    {
      {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, vulkanDevice->info.properties.limits.maxDescriptorSetStorageBuffers},
    },
    vulkanDevice->info.properties.limits.maxBoundDescriptorSets,
    &res
  );
  CHECK_VK_MSG("CreateDescriptorPool for device : " + deviceInfo.deviceName, res);

  return vulkanDevice;
}

VkShaderModule VkHelpers::createShaderModuleFromSPIRVBytes(
    VkDevice device,
    const std::vector<unsigned char>& spirvBytes,
    VkResult* result
) {
  VkShaderModuleCreateInfo shaderModuleCI = {};
  shaderModuleCI.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
  shaderModuleCI.codeSize = spirvBytes.size();
  shaderModuleCI.pCode = reinterpret_cast<const uint32_t*>(spirvBytes.data());
  VkShaderModule shaderModule = VK_NULL_HANDLE;
  *result = vkCreateShaderModule(device, &shaderModuleCI, nullptr, &shaderModule);
  return shaderModule;
}

VkPipelineCache VkHelpers::createPipelineCache(
  VkDevice device,
  VkResult *result
) {
  VkPipelineCacheCreateInfo pipelineCacheCI = {};
  pipelineCacheCI.sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO;
  VkPipelineCache pipelineCache = VK_NULL_HANDLE;
  *result = vkCreatePipelineCache(device, &pipelineCacheCI, nullptr, &pipelineCache);
  return pipelineCache;
}

VkPipelineLayout VkHelpers::createPipelineLayout(
    VkDevice device,
    const std::vector<VkDescriptorSetLayout>& descriptorSetLayouts,
    const std::vector<VkPushConstantRange>& pushConstantRanges,
    VkResult *result
) {
  VkPipelineLayoutCreateInfo pipelineLayoutCI = {};
  pipelineLayoutCI.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  pipelineLayoutCI.setLayoutCount = static_cast<uint32_t>(descriptorSetLayouts.size());
  pipelineLayoutCI.pSetLayouts = descriptorSetLayouts.data();
  pipelineLayoutCI.pushConstantRangeCount = static_cast<uint32_t>(pushConstantRanges.size());
  pipelineLayoutCI.pPushConstantRanges = pushConstantRanges.data();
  VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
  *result = vkCreatePipelineLayout(device, &pipelineLayoutCI, nullptr, &pipelineLayout);
  return pipelineLayout;
}

VkPipeline VkHelpers::createComputePipeline(
    VkDevice device,
    VkPipelineLayout pipelineLayout,
    VkPipelineCache pipelineCache,
    VkShaderModule computeShaderModule,
    VkResult *result,
    VkSpecializationInfo* specializationInfo,
    std::string entryPointName
) {
  VkComputePipelineCreateInfo pipelineCI = {};
  pipelineCI.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
  pipelineCI.layout = pipelineLayout;
  pipelineCI.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  pipelineCI.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
  pipelineCI.stage.module = computeShaderModule;
  pipelineCI.stage.pName = entryPointName.data();
  pipelineCI.stage.pSpecializationInfo = specializationInfo;
  VkPipeline pipeline = VK_NULL_HANDLE;
  *result = vkCreateComputePipelines(device, pipelineCache, 1, &pipelineCI, nullptr, &pipeline);
  return pipeline;
}

VkDescriptorSetLayout VkHelpers::createDescriptorSetLayout(
    VkDevice device,
    const std::vector<VkDescriptorSetLayoutBinding>& bindings,
    VkResult *result
) {
  VkDescriptorSetLayoutCreateInfo layoutCI = {};
  layoutCI.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
  layoutCI.bindingCount = static_cast<uint32_t>(bindings.size());
  layoutCI.pBindings = bindings.data();
  VkDescriptorSetLayout descriptorSetLayout = VK_NULL_HANDLE;
  *result = vkCreateDescriptorSetLayout(device, &layoutCI, nullptr, &descriptorSetLayout);
  return descriptorSetLayout;
}

VkDescriptorPool VkHelpers::createDescriptorPool(
  const VulkanDevice *device,
  const std::vector<VkDescriptorPoolSize>& poolSizes,
  uint32_t maxSets,
  VkResult *result
) {
  VkDescriptorPoolCreateInfo poolCI = {};
  poolCI.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
  poolCI.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
  poolCI.pPoolSizes = poolSizes.data();
  poolCI.maxSets = maxSets;
  VkDescriptorPool descriptorPool = VK_NULL_HANDLE;
  *result = vkCreateDescriptorPool(
    device->device,
    &poolCI,
    nullptr,
    &descriptorPool
  );
  return descriptorPool;
}

VkDescriptorSet VkHelpers::allocateDescriptorSet(
  const VulkanDevice *device,
  VkDescriptorSetLayout descriptorSetLayout,
  VkResult *result
) {
  VkDescriptorSetAllocateInfo allocInfo = {};
  allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
  allocInfo.descriptorPool = device->descriptorPool;
  allocInfo.descriptorSetCount = 1;
  allocInfo.pSetLayouts = &descriptorSetLayout;

  VkDescriptorSet descriptorSet = VK_NULL_HANDLE;
  *result = vkAllocateDescriptorSets(
    device->device,
    &allocInfo,
    &descriptorSet
  );
  return descriptorSet;
}

VkResult VkHelpers::updateDescriptorSets(
  const VulkanDevice *device,
  const std::vector<VkWriteDescriptorSet>& writeDescriptorSets
) {
  vkUpdateDescriptorSets(
    device->device,
    static_cast<uint32_t>(writeDescriptorSets.size()),
    writeDescriptorSets.data(),
    0,
    nullptr
  );
  return VK_SUCCESS;
}

VkCommandPool VkHelpers::createCommandPool(
  const VulkanDevice *device,
  VkResult *result
) {
  VkCommandPoolCreateInfo poolCI = {};
  poolCI.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
  poolCI.queueFamilyIndex = 0; // TODO: select proper queue family index, this code assumes index 0 supports compute
  poolCI.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
  VkCommandPool commandPool = VK_NULL_HANDLE;
  *result = vkCreateCommandPool(
    device->device,
    &poolCI,
    nullptr,
    &commandPool
  );
  return commandPool;
}

VkCommandBuffer VkHelpers::allocateCommandBuffer(
  const VulkanDevice *device
) {
  VkCommandBufferAllocateInfo allocInfo = {};
  allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  allocInfo.commandPool = device->commandPool;
  allocInfo.commandBufferCount = 1;

  VkCommandBuffer commandBuffer;
  vkAllocateCommandBuffers(device->device, &allocInfo, &commandBuffer);
  return commandBuffer;
}

VkResult VkHelpers::beginCommandBuffer(
  VkCommandBuffer commandBuffer
) {
  VkCommandBufferBeginInfo beginInfo = {};
  beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  beginInfo.flags = 0; // Optional
  beginInfo.pInheritanceInfo = nullptr; // Not required.
  beginInfo.flags = VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT;

  return vkBeginCommandBuffer(commandBuffer, &beginInfo);
}

VkResult VkHelpers::endCommandBuffer(
  VkCommandBuffer commandBuffer
) {
  return vkEndCommandBuffer(commandBuffer);
}

VkResult VkHelpers::submitCommandBuffers(
  const VulkanDevice *device,
  const std::vector<VkCommandBuffer>& commandBuffers,
  VkFence fence
) {
  VkSubmitInfo submitInfo = {};
  submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  submitInfo.commandBufferCount = static_cast<uint32_t>(commandBuffers.size());
  submitInfo.pCommandBuffers = commandBuffers.data();
  return vkQueueSubmit(device->queue, 1, &submitInfo, fence);
}

VkCommandBuffer VkHelpers::beginSingleTimeCommandBuffer(
  const VulkanDevice *device
) {
  VkCommandBufferAllocateInfo allocInfo = {};
  allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  allocInfo.commandPool = device->commandPool;
  allocInfo.commandBufferCount = 1;

  VkCommandBuffer commandBuffer;
  vkAllocateCommandBuffers(device->device, &allocInfo, &commandBuffer);

  VkCommandBufferBeginInfo beginInfo = {};
  beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

  vkBeginCommandBuffer(commandBuffer, &beginInfo);
  return commandBuffer;
}

void VkHelpers::submitSingleTimeCommandBufferAndWaitIdle(
  const VulkanDevice *device,
  VkCommandBuffer commandBuffer
) {
  vkEndCommandBuffer(commandBuffer);
  VkSubmitInfo submitInfo = {};
  submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  submitInfo.commandBufferCount = 1;
  submitInfo.pCommandBuffers = &commandBuffer;
  vkQueueSubmit(device->queue, 1, &submitInfo, VK_NULL_HANDLE);
  vkQueueWaitIdle(device->queue);
  vkFreeCommandBuffers(device->device, device->commandPool, 1, &commandBuffer);
}


VulkanBuffer* VkHelpers::createDeviceBuffer(
  const VulkanDevice *device,
  size_t size,
  bool readOnly,
  VkResult *result
) {
  VulkanBuffer *buffer = new VulkanBuffer();
  buffer->device = device;
  buffer->buffer = VK_NULL_HANDLE;
  VkBufferCreateInfo bufferCI = {};
  bufferCI.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  bufferCI.size = static_cast<VkDeviceSize>(size);
  if ( readOnly) {
    bufferCI.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
  } else {
    bufferCI.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
  }
  bufferCI.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  VmaAllocationCreateInfo allocCI = {};
  allocCI.usage = VMA_MEMORY_USAGE_GPU_ONLY;
  VkResult res = vmaCreateBuffer(
    device->allocator,
    &bufferCI,
    &allocCI,
    &buffer->buffer,
    &buffer->allocation,
    &buffer->allocationInfo
  );
  *result = res;
  return buffer;
}

VulkanBuffer* VkHelpers::createDeviceBufferWithData(
  const VulkanDevice *device,
  size_t size,
  const void* data,
  bool readOnly,
  VkResult *result
) {
  VulkanBuffer* deviceBuffer = createDeviceBuffer(
    device,
    size,
    readOnly,
    result
  );

  if(*result != VK_SUCCESS) {
    return nullptr;
  }

  VulkanBuffer* stagingBuffer = createStagingBuffer(
    device,
    size,
    result
  );

  if(*result != VK_SUCCESS) {
    releaseVulkanBuffer(device, deviceBuffer);
    return nullptr;
  }

  // Copy data to staging buffer
  memcpy(stagingBuffer->allocationInfo.pMappedData, data, size);
  // Copy staging buffer to device buffer
  VkCommandBuffer commandBuffer = VkHelpers::beginSingleTimeCommandBuffer(device);
  VkBufferCopy copyRegion = {};
  copyRegion.srcOffset = 0;
  copyRegion.dstOffset = 0;
   copyRegion.size = size;
  vkCmdCopyBuffer(
    commandBuffer,
    stagingBuffer->buffer,
    deviceBuffer->buffer,
    1,
    &copyRegion
  );
  VkHelpers::submitSingleTimeCommandBufferAndWaitIdle(
    device,
    commandBuffer
  );
  // Release staging buffer
  VkHelpers::releaseVulkanBuffer(
    device,
    stagingBuffer
  );
  return deviceBuffer;
}

VulkanBuffer* VkHelpers::createStagingBuffer(
  const VulkanDevice *device,
  size_t size,
  VkResult *result
) {
  VulkanBuffer *buffer = new VulkanBuffer();
  buffer->device = device;
  VkBufferCreateInfo bufferCI = {};
  bufferCI.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  bufferCI.size = size;
  bufferCI.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
  bufferCI.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  VmaAllocationCreateInfo allocCI = {};
  allocCI.usage = VMA_MEMORY_USAGE_CPU_TO_GPU;
  VkResult res = vmaCreateBuffer(
    device->allocator,
    &bufferCI,
    &allocCI,
    &buffer->buffer,
    &buffer->allocation,
    &buffer->allocationInfo
  );
  *result = res;
  return buffer;
}

VulkanBuffer* VkHelpers::createReadbackBuffer(
  const VulkanDevice *device,
  size_t size,
  VkResult *result
) {
  VulkanBuffer *buffer = new VulkanBuffer();
  buffer->device = device;
  VkBufferCreateInfo bufferCI = {};
  bufferCI.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  bufferCI.size = size;
  bufferCI.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
  bufferCI.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  VmaAllocationCreateInfo allocCI = {};
  allocCI.usage = VMA_MEMORY_USAGE_GPU_TO_CPU;
  VkResult res = vmaCreateBuffer(
    device->allocator,
    &bufferCI,
    &allocCI,
    &buffer->buffer,
    &buffer->allocation,
    &buffer->allocationInfo
  );
  *result = res;
  return buffer;
}

void VkHelpers::releaseVulkanBuffer(
  const VulkanDevice *device,
  VulkanBuffer *buffer
) {
  if(buffer != nullptr) {
    vmaDestroyBuffer(
      device->allocator,
      buffer->buffer,
      buffer->allocation
    );
    delete buffer;
    buffer = nullptr;
  }
}

VkFence VkHelpers::createFence(
  const VulkanDevice *device,
  VkResult *result
) {
  VkFenceCreateInfo fenceCI = {};
  fenceCI.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
  fenceCI.flags = VK_FENCE_CREATE_SIGNALED_BIT;
  VkFence fence = VK_NULL_HANDLE;
  *result = vkCreateFence(
    device->device,
    &fenceCI,
    nullptr,
    &fence
  );
  return fence;
}

void VkHelpers::destroyFence(
  const VulkanDevice *device,
  VkFence fence
) {
  vkDestroyFence(
    device->device,
    fence,
    nullptr
  );
}

VkResult VkHelpers::resetFence(
  const VulkanDevice *device,
  VkFence fence
) {
  return vkResetFences(device->device, 1 , &fence);
}

#endif
