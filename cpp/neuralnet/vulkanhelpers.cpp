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
    // std::cout << "[VulkanDevice::~VulkanDevice] Destroying Vulkan device " << this->info.deviceName << std::endl;
  if (this->device != VK_NULL_HANDLE) {
    if(!skipWaitOnDestruction) {
      vkQueueWaitIdle(this->queue);
      vkDeviceWaitIdle(this->device);
      vmaDestroyAllocator(this->allocator);
      // std::cout << "[VulkanDevice::~VulkanDevice] Destroyed allocator for device " << this->info.deviceName << std::endl;

      if (this->descriptorPool != VK_NULL_HANDLE) {
        vkResetDescriptorPool(this->device, this->descriptorPool, 0);
        vkDestroyDescriptorPool(this->device, this->descriptorPool, nullptr);
        // std::cout << "[VulkanDevice::~VulkanDevice] Destroyed descriptor pool for device " << this->info.deviceName << std::endl;
        this->descriptorPool = VK_NULL_HANDLE;
      }

      if (this->commandPool != VK_NULL_HANDLE) {
        vkResetCommandPool(this->device, this->commandPool, 0);
        vkDestroyCommandPool(this->device, this->commandPool, nullptr);
        // std::cout << "[VulkanDevice::~VulkanDevice] Destroyed command pool for device " << this->info.deviceName << std::endl;
        this->commandPool = VK_NULL_HANDLE;
      }
    }
    vkDestroyDevice(this->device, nullptr);
    // std::cout << "[VulkanDevice::~VulkanDevice] Destroyed device " << this->info.deviceName << std::endl;
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
  debugCreateInfo.pfnUserCallback = VkDebug::debugCallback;
  debugCreateInfo.pUserData = nullptr; // Optional user data
  auto func = (PFN_vkCreateDebugUtilsMessengerEXT)vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT");
  if (func != nullptr) {
    VkResult res = func(instance, &debugCreateInfo, nullptr, &this->debugMessenger);
    CHECK_VK_MSG("CreateDebugUtilsMessengerEXT", res);
  } else {
    if ( logger ) {
      logger->write("Warning: Could not set up Vulkan debug messenger.");
    }
  }
  #endif
  if ( logger ) {
    logger->write("VulkanContext created with " + std::to_string(devicesToUse.size()) + " devices.");
  }
}

VulkanContext::~VulkanContext() {
  for ( VulkanDevice* device : this->devicesToUse ) {
    delete device;
    device = nullptr;
  }
  this->devicesToUse.clear();

  #ifdef VULKAN_API_DEBUG
    if ( this->debugMessenger != VK_NULL_HANDLE ) {
      auto func = (PFN_vkDestroyDebugUtilsMessengerEXT)vkGetInstanceProcAddr(this->instance, "vkDestroyDebugUtilsMessengerEXT");
      if (func != nullptr) {
        func(this->instance, this->debugMessenger, nullptr);
        this->debugMessenger = VK_NULL_HANDLE;
      }
    }
  #endif

  if (this->instance != VK_NULL_HANDLE) {
    vkDestroyInstance(this->instance, nullptr);
    this->instance = VK_NULL_HANDLE;
  }
}

const VulkanDevice* VulkanContext::findGpuExn(int gpuIdx) const {
  if ( gpuIdx < 0 ) {
    gpuIdx = 0;
  }

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

int VkHelpers::roundUpToMultipleInt(int size, int ofThis) {
  // Use 64-bit arithmetic to avoid overflow when multiplying large sizes,
  // then clamp to INT_MAX to keep return value in int range.
  if(ofThis <= 0) return size;
  int64_t s = static_cast<int64_t>(size);
  int64_t o = static_cast<int64_t>(ofThis);
  int64_t ret = ((s + o - 1) / o) * o;
  if(ret > static_cast<int64_t>(std::numeric_limits<int>::max()))
    return std::numeric_limits<int>::max();
  return static_cast<int>(ret);
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
#ifdef __APPLE__
    "VK_KHR_portability_enumeration",
#endif
    "VK_KHR_get_physical_device_properties2",
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
#ifdef __APPLE__
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
    if ( logger != nullptr ) {
      logger->write("Found Vulkan Physical Device[" + Global::uint32ToString(i) + "]");
    }
    VkPhysicalDevice physicalDevice = physicalDevices[i];
    VulkanDeviceInfo deviceInfo = {};
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
    if ( logger != nullptr ) {
      logger->write("  Name: " + deviceInfo.deviceName);
      logger->write("  Type: " + VkHelpers::vkPhysicalDeviceTypeToString(deviceInfo.deviceType));
      logger->write("  API Version: " +
        std::to_string(VK_VERSION_MAJOR(deviceInfo.properties.apiVersion)) + "." +
        std::to_string(VK_VERSION_MINOR(deviceInfo.properties.apiVersion)) + "." +
        std::to_string(VK_VERSION_PATCH(deviceInfo.properties.apiVersion))
      );
    }
    // Get device features
    VkPhysicalDeviceFeatures2 features2 = {};
    features2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    // Get shader float16/int8 features
    VkPhysicalDeviceShaderFloat16Int8Features shaderFloat16Int8Features = {};
    VkPhysicalDeviceCooperativeMatrixFeaturesKHR cooperativeMatrixFeatures = {};
    VkPhysicalDeviceMaintenance4FeaturesKHR maintenance4Features = {};
    shaderFloat16Int8Features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES;
    features2.pNext = &shaderFloat16Int8Features;
    cooperativeMatrixFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR;
    shaderFloat16Int8Features.pNext = &cooperativeMatrixFeatures;
    maintenance4Features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_4_FEATURES_KHR;
    cooperativeMatrixFeatures.pNext = &maintenance4Features;
    features2.pNext = &shaderFloat16Int8Features;
    vkGetPhysicalDeviceFeatures2(physicalDevice, &features2);
    deviceInfo.features = features2.features;
    deviceInfo.shaderFloat16Int8Features = shaderFloat16Int8Features;
    if ( logger != nullptr ) {
      logger->write("  Shader Float16 Support: " + std::string(deviceInfo.shaderFloat16Int8Features.shaderFloat16 == VK_TRUE ? "Yes" : "No"));
    } 

    // VkPhysicalDeviceCooperativeMatrixFeaturesKHR cooperativeMatrixFeatures = {};
    // cooperativeMatrixFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR;
    // features2.pNext = &cooperativeMatrixFeatures;
    // vkGetPhysicalDeviceFeatures2(physicalDevice, &features2);
    deviceInfo.cooperativeMatrixFeatures = cooperativeMatrixFeatures;
    deviceInfo.maintenance4Features = maintenance4Features;
    deviceInfo.shaderFloat16Int8Features = shaderFloat16Int8Features;

    if ( logger != nullptr ) {
      logger->write("  Cooperative Matrix Support: " + std::string(deviceInfo.cooperativeMatrixFeatures.cooperativeMatrix == VK_TRUE ? "Yes" : "No"));
    }


    // Get memory properties2
    VkPhysicalDeviceMemoryProperties2 memoryProperties2 = {};
    memoryProperties2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2;
    // Cooperative Matrix features
    memoryProperties2.pNext = nullptr;
    vkGetPhysicalDeviceMemoryProperties2(physicalDevice, &memoryProperties2);
    deviceInfo.memoryProperties = memoryProperties2.memoryProperties;
    
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
  if ( logger ) {
    logger->write("Creating Vulkan Logical Device for " + deviceInfo.deviceName);
  }
  VkPhysicalDevice physicalDevice = deviceInfo.physicalDevice;
  // Create logical device
#ifdef __APPLE__
  requiredExtensions.push_back("VK_KHR_portability_subset");
#endif
  
  std::vector<VkExtensionProperties> availableExtensions;
  uint32_t extensionCount = 0;
  vkEnumerateDeviceExtensionProperties(physicalDevice, nullptr, &extensionCount, nullptr);
  if(extensionCount > 0) {
    availableExtensions.resize(extensionCount);
    vkEnumerateDeviceExtensionProperties(physicalDevice, nullptr, &extensionCount, availableExtensions.data());
  } else {
    if ( logger ) {
      logger->write("No device extensions available");
    }
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

  VkPhysicalDeviceFeatures2 requestedFeatures = {};
  requestedFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
  requestedFeatures.features = deviceInfo.features; // start with all available features, we'll disable unsupported

  VkPhysicalDeviceShaderFloat16Int8Features f16Feat = {};
  f16Feat.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES;

  if ( deviceInfo.shaderFloat16Int8Features.shaderFloat16 == VK_FALSE ) {
    std::cout << "Device " << deviceInfo.deviceName << " does not support shaderFloat16, disabling it." << std::endl;
    f16Feat.shaderFloat16 = VK_FALSE;
  } else {
    f16Feat.shaderFloat16 = VK_TRUE;
  }

  VkPhysicalDeviceCooperativeMatrixFeaturesKHR cmFeatures = {};
  cmFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR;
  requestedFeatures.pNext = &cmFeatures;
  if ( deviceInfo.cooperativeMatrixFeatures.cooperativeMatrix == VK_FALSE ) {
      std::cout << "Device " << deviceInfo.deviceName << " does not support cooperativeMatrix, disabling it." << std::endl;
      cmFeatures.cooperativeMatrix = VK_FALSE;
  } else {
    cmFeatures.cooperativeMatrix = VK_TRUE;
  }

  VkPhysicalDeviceMaintenance4FeaturesKHR m4Features = {};
  m4Features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_4_FEATURES_KHR;
  if ( deviceInfo.maintenance4Features.maintenance4 == VK_FALSE ) {
    std::cout << "Device " << deviceInfo.deviceName << " does not support maintenance4, disabling it." << std::endl;
    m4Features.maintenance4 = VK_FALSE;
  } else {
    m4Features.maintenance4 = VK_TRUE;
  }
  
  cmFeatures.pNext = &m4Features;
  f16Feat.pNext = &cmFeatures;
  requestedFeatures.pNext = &f16Feat;

  VkDeviceCreateInfo deviceCI = {};
  deviceCI.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
  deviceCI.pNext = &requestedFeatures;
  deviceCI.queueCreateInfoCount = 1;
  deviceCI.pQueueCreateInfos = &queueCI;
  deviceCI.enabledExtensionCount = static_cast<uint32_t>(requiredExtensions.size());
  deviceCI.ppEnabledExtensionNames = requiredExtensions.data();

  VkDevice device;
  VkResult res = vkCreateDevice(physicalDevice, &deviceCI, nullptr, &device);
  CHECK_VK_MSG("Create Vulkan Logical Device for " + deviceInfo.deviceName, res);
  if ( logger ) {
    logger->write("Created Vulkan Logical Device for " + deviceInfo.deviceName);
  }

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
      {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 4096},
    },
    1024,      
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
  const std::vector<WriteDescriptorSet>& writeDescriptorSets
) {
  std::vector<VkWriteDescriptorSet> vkWriteDescriptorSets;
  vkWriteDescriptorSets.reserve(writeDescriptorSets.size());
  for ( size_t i = 0; i < writeDescriptorSets.size(); i++ ) {
    VkWriteDescriptorSet writeSet = writeDescriptorSets[i].vkWriteDescriptorSet;
    // Fix dangling pointer: pBufferInfo must point to the bufferInfo in the vector, not a stale address
    writeSet.pBufferInfo = &writeDescriptorSets[i].bufferInfo;
    vkWriteDescriptorSets.push_back(writeSet);
  }

  vkUpdateDescriptorSets(
    device->device,
    static_cast<uint32_t>(vkWriteDescriptorSets.size()),
    vkWriteDescriptorSets.data(),
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
  const VulkanDevice *device,
  VkResult* result
) {
  VkCommandBufferAllocateInfo allocInfo = {};
  allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  allocInfo.commandPool = device->commandPool;
  allocInfo.commandBufferCount = 1;

  VkCommandBuffer commandBuffer = VK_NULL_HANDLE;
  VkResult allocationResult = vkAllocateCommandBuffers(device->device, &allocInfo, &commandBuffer);
  if(result != nullptr)
    *result = allocationResult;
  return commandBuffer;
}

VkResult VkHelpers::beginCommandBuffer(
  VkCommandBuffer commandBuffer
) {
  VkCommandBufferBeginInfo beginInfo = {};
  beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  beginInfo.pInheritanceInfo = nullptr; // Not required.
  beginInfo.flags = 0;//VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

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
  if ( !result ) {
    throw StringError("VkHelpers::createDeviceBuffer: result pointer is null");
  }
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
  buffer->requestedSize = static_cast<VkDeviceSize>(size);
  if(res != VK_SUCCESS) {
    delete buffer;
    return nullptr;
  }
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

  memcpy(stagingBuffer->allocationInfo.pMappedData, data, size);
  // Copy staging buffer to device buffer
  vmaFlushAllocation(
    device->allocator,
    stagingBuffer->allocation,
    0,
    size
  );
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
  allocCI.flags = VMA_ALLOCATION_CREATE_MAPPED_BIT;
  VkResult res = vmaCreateBuffer(
    device->allocator,
    &bufferCI,
    &allocCI,
    &buffer->buffer,
    &buffer->allocation,
    &buffer->allocationInfo
  );
  *result = res;
  buffer->requestedSize = static_cast<VkDeviceSize>(size);
  if(res != VK_SUCCESS) {
    delete buffer;
    return nullptr;
  }
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
  buffer->requestedSize = static_cast<VkDeviceSize>(size);
  if(res != VK_SUCCESS) {
    delete buffer;
    return nullptr;
  }
  return buffer;
}

/**
 * Copies data from a device-local Vulkan buffer to a host pointer.
 * @param device The Vulkan device.
 * @param deviceBuffer The source Vulkan buffer located on the device.
 * @param copySize The size of data to copy in bytes.
 * @param hostPtr The destination host pointer where data will be copied.
 * @param waitForIdle If true, waits for the device to be idle after the copy operation.
 * @param result Pointer to a VkResult variable to capture the result of the operation.
 */
void VkHelpers::copyDeviceBufferToHost(
  const VulkanDevice* device,
  VulkanBuffer* deviceBuffer,
  VkDeviceSize copySize,
  void* hostPtr,
  bool waitForIdle,
  VkResult *result
) {
  // Create readback buffer
  VulkanBuffer* readbackBuffer = VkHelpers::createReadbackBuffer(
    device,
    static_cast<size_t>(copySize),
    result
  );

  if(*result != VK_SUCCESS) {
    return;
  }

  // Copy device buffer to readback buffer
  VkCommandBuffer commandBuffer = VkHelpers::beginSingleTimeCommandBuffer(device);
  VkBufferCopy copyRegion = {};
  copyRegion.srcOffset = 0;
  copyRegion.dstOffset = 0;
  copyRegion.size = copySize;
  vkCmdCopyBuffer(
    commandBuffer,
    deviceBuffer->buffer,
    readbackBuffer->buffer,
    1,
    &copyRegion
  );
  VkHelpers::submitSingleTimeCommandBufferAndWaitIdle(
    device,
    commandBuffer
  );

  // Map readback buffer and copy data to hostPtr
  void* mappedData = nullptr;
  *result = vmaMapMemory(
    device->allocator,
    readbackBuffer->allocation,
    &mappedData
  );

  if ( *result != VK_SUCCESS ) {
    VkHelpers::releaseVulkanBuffer(
      device,
      readbackBuffer
    );
    return;
  }
  memcpy(hostPtr, mappedData, static_cast<size_t>(copySize));
  vmaUnmapMemory(
    device->allocator,
    readbackBuffer->allocation
  );

  // Release readback buffer
  VkHelpers::releaseVulkanBuffer(
    device,
    readbackBuffer
  );
}

void VkHelpers::copyHostToDeviceBuffer(
  const VulkanDevice* device,
  const void* hostPtr,
  VulkanBuffer* deviceBuffer,
  VkDeviceSize copySize,
  bool waitForIdle,
  VkResult *result
) {
  // Create staging buffer
  VulkanBuffer* stagingBuffer = VkHelpers::createStagingBuffer(
    device,
    static_cast<size_t>(copySize),
    result
  );

  if(*result != VK_SUCCESS) {
    return;
  }

  // Copy data from hostPtr to staging buffer
  memcpy(
    stagingBuffer->allocationInfo.pMappedData,
    hostPtr,
    static_cast<size_t>(copySize)
  );
  vmaFlushAllocation(
    device->allocator,
    stagingBuffer->allocation,
    0,
    copySize
  );

  // Copy staging buffer to device buffer
  VkCommandBuffer commandBuffer = VkHelpers::beginSingleTimeCommandBuffer(device);
  VkBufferCopy copyRegion = {};
  copyRegion.srcOffset = 0;
  copyRegion.dstOffset = 0;
  copyRegion.size = copySize;
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
  
  if(waitForIdle) {
    vkDeviceWaitIdle(device->device);
  }
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

void VkHelpers::barrierCommandBuffer(
  VkCommandBuffer commandBuffer,
  VkPipelineStageFlags srcStageMask,
  VkAccessFlags srcAccessMask,
  VkPipelineStageFlags dstStageMask,
  VkAccessFlags dstAccessMask
) {
  VkMemoryBarrier memoryBarrier = {};
  memoryBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
  memoryBarrier.srcAccessMask = srcAccessMask;
  memoryBarrier.dstAccessMask = dstAccessMask;
  vkCmdPipelineBarrier(commandBuffer, srcStageMask, dstStageMask, 0, 1, &memoryBarrier, 0, nullptr, 0, nullptr);
}

void VkHelpers::barrierCommandBufferForBuffer(
  VkCommandBuffer commandBuffer,
  VulkanBuffer* buffer,
  VkPipelineStageFlags srcStageMask,
  VkAccessFlags srcAccessMask,
  VkPipelineStageFlags dstStageMask,
  VkAccessFlags dstAccessMask
) {
  VkBufferMemoryBarrier bufferBarrier = {};
  bufferBarrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
  bufferBarrier.srcAccessMask = srcAccessMask;
  bufferBarrier.dstAccessMask = dstAccessMask;
  bufferBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  bufferBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  bufferBarrier.buffer = buffer->buffer;
  bufferBarrier.offset = 0;
  bufferBarrier.size = VK_WHOLE_SIZE;
  vkCmdPipelineBarrier(commandBuffer, srcStageMask, dstStageMask, 0, 0, nullptr, 1, &bufferBarrier, 0, nullptr);
}

std::vector<int32_t> VkHelpers::createSpecData(void* data, size_t dataSize) {
  size_t cnt = dataSize / sizeof(int32_t);
  int32_t* intData = reinterpret_cast<int32_t*>(data);
  std::vector<int32_t> specData(intData, intData + cnt);
  return specData;
}

std::vector<VkSpecializationMapEntry> VkHelpers::createSpecMapEntries(size_t dataCount) {
  std::vector<VkSpecializationMapEntry> mapEntries(dataCount);
  for ( size_t i = 0; i < dataCount; i++ ) {
    mapEntries[i].constantID = static_cast<uint32_t>(i);
    mapEntries[i].offset = static_cast<uint32_t>(i * sizeof(int32_t));
    mapEntries[i].size = sizeof(int32_t);
  }
  return mapEntries;
}

VkSpecializationInfo VkHelpers::createSpecializationInfo(const std::vector<int32_t>& specData, const std::vector<VkSpecializationMapEntry>& mapEntries) {
  VkSpecializationInfo specInfo = {};
  specInfo.mapEntryCount = static_cast<uint32_t>(mapEntries.size());
  specInfo.pMapEntries = mapEntries.data();
  specInfo.dataSize = specData.size() * sizeof(int32_t);
  specInfo.pData = specData.data();
  return specInfo;
}

namespace VkDebug {

VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(
    VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
    VkDebugUtilsMessageTypeFlagsEXT messageType,
    const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
    void* pUserData
) {
    const char* severity = "INFO";
    if (messageSeverity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT)
        severity = "VERBOSE";
    else if (messageSeverity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT)
        severity = "INFO";
    else if (messageSeverity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT)
        severity = "WARNING";
    else if (messageSeverity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT)
        severity = "ERROR";

    const char* type = "UNKNOWN";
    if (messageType & VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT)
        type = "GENERAL";
    else if (messageType & VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT)
        type = "VALIDATION";
    else if (messageType & VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT)
        type = "PERFORMANCE";

    std::fprintf(
        stderr,
        "[Vulkan][%s][%s] %s\n",
        severity,
        type,
        pCallbackData->pMessage
    );

    return VK_FALSE;
}

} // namespace VkDebug
#endif
