#ifdef USE_VULKAN_BACKEND
#pragma once;

#include <string>
#include <vulkan/vulkan.h>
#include <vector>
#include <vma/vk_mem_alloc.h>
#include "../core/global.h"
#include "../core/logger.h"
#include "../neuralnet/vulkanhelpers.h"

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
  appInfo.pEngineName = "KatagoEngine";
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



#endif
