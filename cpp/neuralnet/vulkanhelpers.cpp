/**
 * @file vulkanhelpers.cpp
 * @author Dohoon Kim(https://github.com/dhkim92-dev)
 * @brief Vulkan Helper
 */
#ifdef USE_VULKAN_BACKEND
#pragma once;

#include <string>
#include <vulkan/vulkan.h>
#include <vector>
#include <algorithm>
#include <memory>

#ifndef VMA_IMPLEMENTATION
#define VMA_IMPLEMENTATION
#include "../external/vulkan/vk_mem_alloc.h"
#endif
#include "../external/half-2.2.0/include/half.hpp"
#include "../core/global.h"
#include "../core/logger.h"
#include "../neuralnet/vulkanhelpers.h"

using half_t = half_float::half;

namespace {
  uint32_t getVulkanInstanceVersion() {
    uint32_t version = VK_API_VERSION_1_0;
    auto enumerateVersion = reinterpret_cast<PFN_vkEnumerateInstanceVersion>(
      vkGetInstanceProcAddr(VK_NULL_HANDLE, "vkEnumerateInstanceVersion")
    );
    if(enumerateVersion != nullptr) {
      VkResult res = enumerateVersion(&version);
      CHECK_VK(res);
    }
#if defined(VK_VERSION_1_3)
    return std::min(version, uint32_t(VK_API_VERSION_1_3));
#else
    return std::min(version, uint32_t(VK_API_VERSION_1_2));
#endif
  }

  template<typename Property, typename Enumerate>
  std::vector<Property> enumerateVulkanProperties(Enumerate enumerate) {
    std::vector<Property> properties;
    VkResult res;
    do {
      uint32_t count = 0;
      res = enumerate(&count, nullptr);
      CHECK_VK(res);
      properties.resize(count);
      if(count == 0)
        return properties;
      res = enumerate(&count, properties.data());
      if(res != VK_INCOMPLETE) {
        CHECK_VK(res);
        properties.resize(count);
      }
    } while(res == VK_INCOMPLETE);
    return properties;
  }

  bool hasDeviceExtension(const VulkanDeviceInfo& info, const char* name) {
    return std::any_of(info.extensions.begin(), info.extensions.end(), [name](const VkExtensionProperties& extension) {
      return std::string(extension.extensionName) == name;
    });
  }
}

VulkanDevice::~VulkanDevice() {
    // std::cout << "[VulkanDevice::~VulkanDevice] Destroying Vulkan device " << this->info.deviceName << std::endl;
  if (this->device != VK_NULL_HANDLE) {
    if(!skipWaitOnDestruction) {
      vkQueueWaitIdle(this->queue);
      vkDeviceWaitIdle(this->device);
      vmaDestroyAllocator(this->allocator);
      // std::cout << "[VulkanDevice::~VulkanDevice] Destroyed allocator for device " << this->info.deviceName << std::endl;

    }
    vkDestroyDevice(this->device, nullptr);
    // std::cout << "[VulkanDevice::~VulkanDevice] Destroyed device " << this->info.deviceName << std::endl;
    this->device = VK_NULL_HANDLE;
  }
}

VkDeviceSize vk_helper::getTotalAllocatedGpuMemory(const VulkanDevice* device) {
  if(device == nullptr || device->allocator == VK_NULL_HANDLE)
    return 0;

  VmaTotalStatistics statistics = {};
  vmaCalculateStatistics(device->allocator, &statistics);
  return statistics.total.statistics.blockBytes;
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
  const auto extensions = enumerateVulkanProperties<VkExtensionProperties>([](uint32_t* count, VkExtensionProperties* data) {
    return vkEnumerateInstanceExtensionProperties(nullptr, count, data);
  });
  const bool debugUtilsAvailable = std::any_of(extensions.begin(), extensions.end(), [](const VkExtensionProperties& extension) {
    return std::string(extension.extensionName) == VK_EXT_DEBUG_UTILS_EXTENSION_NAME;
  });
  auto func = debugUtilsAvailable ?
    (PFN_vkCreateDebugUtilsMessengerEXT)vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT") : nullptr;
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

size_t vk_helper::roundUpToMultiple(size_t size, size_t ofThis) {
  return (size + ofThis - 1) / ofThis * ofThis;
}

int vk_helper::roundUpToMultipleInt(int size, int ofThis) {
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

size_t vk_helper::powerOf2ify(size_t size) {
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

std::string vk_helper::vkErrorToString(VkResult res) {
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

std::string vk_helper::vkPhysicalDeviceTypeToString(VkPhysicalDeviceType type) {
  switch(type) {
    case VK_PHYSICAL_DEVICE_TYPE_OTHER: return "Other";
    case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU: return "Integrated GPU";
    case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU: return "Discrete GPU";
    case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU: return "Virtual GPU";
    case VK_PHYSICAL_DEVICE_TYPE_CPU: return "CPU";
    default: return "Unknown Device Type";
  }
}

VkInstance vk_helper::createVulkanInstance() {
  VkInstance instance = VK_NULL_HANDLE;
  const uint32_t apiVersion = getVulkanInstanceVersion();
  if(apiVersion < VK_API_VERSION_1_1)
    throw StringError("The Vulkan backend requires Vulkan 1.1 or newer");
  const auto layers = enumerateVulkanProperties<VkLayerProperties>([](uint32_t* count, VkLayerProperties* data) {
    return vkEnumerateInstanceLayerProperties(count, data);
  });
  const auto extensions = enumerateVulkanProperties<VkExtensionProperties>([](uint32_t* count, VkExtensionProperties* data) {
    return vkEnumerateInstanceExtensionProperties(nullptr, count, data);
  });
  std::vector<const char*> requiredLayers;
  std::vector<const char*> requiredExtensions;
#ifdef VULKAN_API_DEBUG
  const bool validationAvailable = std::any_of(layers.begin(), layers.end(), [](const VkLayerProperties& layer) {
    return std::string(layer.layerName) == "VK_LAYER_KHRONOS_validation";
  });
  if(validationAvailable)
    requiredLayers.push_back("VK_LAYER_KHRONOS_validation");
  else
    std::cerr << "Vulkan validation layer is unavailable, continuing without it." << std::endl;
  const bool debugUtilsAvailable = std::any_of(extensions.begin(), extensions.end(), [](const VkExtensionProperties& extension) {
    return std::string(extension.extensionName) == VK_EXT_DEBUG_UTILS_EXTENSION_NAME;
  });
  if(debugUtilsAvailable)
    requiredExtensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
#endif
#ifdef __APPLE__
  const bool portabilityAvailable = std::any_of(extensions.begin(), extensions.end(), [](const VkExtensionProperties& extension) {
    return std::string(extension.extensionName) == VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME;
  });
  if(portabilityAvailable)
    requiredExtensions.push_back(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);
#endif

  VkApplicationInfo appInfo = {};
  appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
  appInfo.pApplicationName = "KataGo";
  appInfo.applicationVersion = VK_MAKE_VERSION(1, 6, 4);
  appInfo.pEngineName = "Katago Vulkan Backend";
  appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
  appInfo.apiVersion = apiVersion;
  VkInstanceCreateInfo instanceCI = {};
  instanceCI.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
  instanceCI.pApplicationInfo = &appInfo;
  instanceCI.pNext = nullptr;
  instanceCI.enabledLayerCount = static_cast<uint32_t>(requiredLayers.size());
  instanceCI.ppEnabledLayerNames = requiredLayers.data();
  instanceCI.enabledExtensionCount = static_cast<uint32_t>(requiredExtensions.size());
  instanceCI.ppEnabledExtensionNames = requiredExtensions.data();
#ifdef __APPLE__
  instanceCI.flags = portabilityAvailable ? VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR : 0;
#else
  instanceCI.flags = 0;
#endif
  VkResult res = vkCreateInstance(&instanceCI, nullptr, &instance);
  CHECK_VK(res);
  return instance;
}

std::vector<VulkanDeviceInfo> vk_helper::enumerateVulkanDevices(VkInstance instance, Logger* logger) {
  const auto physicalDevices = enumerateVulkanProperties<VkPhysicalDevice>([instance](uint32_t* count, VkPhysicalDevice* data) {
    return vkEnumeratePhysicalDevices(instance, count, data);
  });
  if(physicalDevices.empty())
    throw StringError("No Vulkan-compatible GPU found");

  std::vector<VulkanDeviceInfo> deviceInfos;
  for(uint32_t i = 0; i < physicalDevices.size(); i++) {
    if ( logger != nullptr ) {
      logger->write("Found Vulkan Physical Device[" + Global::uint32ToString(i) + "]");
    }
    VkPhysicalDevice physicalDevice = physicalDevices[i];
    VulkanDeviceInfo deviceInfo = {};
    deviceInfo.deviceId = i;
    deviceInfo.physicalDevice = physicalDevice;
    vkGetPhysicalDeviceProperties(physicalDevice, &deviceInfo.properties);
    deviceInfo.apiVersion = std::min(getVulkanInstanceVersion(), deviceInfo.properties.apiVersion);
    deviceInfo.extensions = enumerateVulkanProperties<VkExtensionProperties>([physicalDevice](uint32_t* count, VkExtensionProperties* data) {
      return vkEnumerateDeviceExtensionProperties(physicalDevice, nullptr, count, data);
    });
    const bool core12 = deviceInfo.apiVersion >= VK_API_VERSION_1_2;
    const bool core13 = deviceInfo.apiVersion >= VK_MAKE_VERSION(1, 3, 0);
#if defined(VK_KHR_cooperative_matrix)
    const bool hasCooperativeMatrix = hasDeviceExtension(deviceInfo, VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME);
#endif
#if defined(VK_KHR_cooperative_matrix) && defined(VK_NV_cooperative_matrix2)
    const bool hasMatrix2 = core13 && hasCooperativeMatrix &&
      hasDeviceExtension(deviceInfo, VK_NV_COOPERATIVE_MATRIX_2_EXTENSION_NAME);
#endif
#if defined(VK_EXT_subgroup_size_control)
    const bool hasSubgroupSizeControl = core13 || hasDeviceExtension(deviceInfo, VK_EXT_SUBGROUP_SIZE_CONTROL_EXTENSION_NAME);
#endif
#if defined(VK_KHR_maintenance4)
    const bool hasMaintenance4 = core13 || hasDeviceExtension(deviceInfo, VK_KHR_MAINTENANCE_4_EXTENSION_NAME);
#endif
#if defined(VK_KHR_cooperative_matrix)
    deviceInfo.cooperativeMatrixPropertiesFn = hasCooperativeMatrix ?
      reinterpret_cast<PFN_vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR>(
        vkGetInstanceProcAddr(instance, "vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR")
      ) : nullptr;
#endif

    // Get device properties
    VkPhysicalDeviceProperties2 properties2 = {};
    properties2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
    VkPhysicalDeviceSubgroupProperties subgroupProperties = {};
    subgroupProperties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES;
#if defined(VK_EXT_subgroup_size_control)
    VkPhysicalDeviceSubgroupSizeControlPropertiesEXT subgroupSizeProperties = {};
    subgroupSizeProperties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_PROPERTIES_EXT;
#endif
#if defined(VK_KHR_cooperative_matrix)
    VkPhysicalDeviceCooperativeMatrixPropertiesKHR cooperativeMatrixProperties = {};
    cooperativeMatrixProperties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_PROPERTIES_KHR;
#endif
#if defined(VK_KHR_cooperative_matrix) && defined(VK_NV_cooperative_matrix2)
    VkPhysicalDeviceCooperativeMatrix2PropertiesNV matrix2Properties = {};
    matrix2Properties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_2_PROPERTIES_NV;
#endif
    auto addProperty = [&properties2](auto& property) {
      property.pNext = properties2.pNext;
      properties2.pNext = &property;
    };
    if(deviceInfo.apiVersion >= VK_API_VERSION_1_1)
      addProperty(subgroupProperties);
#if defined(VK_EXT_subgroup_size_control)
    if(hasSubgroupSizeControl)
      addProperty(subgroupSizeProperties);
#endif
#if defined(VK_KHR_cooperative_matrix)
    if(hasCooperativeMatrix)
      addProperty(cooperativeMatrixProperties);
#endif
#if defined(VK_KHR_cooperative_matrix) && defined(VK_NV_cooperative_matrix2)
    if(hasMatrix2)
      addProperty(matrix2Properties);
#endif
    vkGetPhysicalDeviceProperties2(physicalDevice, &properties2);
    deviceInfo.properties = properties2.properties;
    deviceInfo.subgroupProperties = subgroupProperties;
#if defined(VK_EXT_subgroup_size_control)
    deviceInfo.subgroupSizeProperties = subgroupSizeProperties;
#endif
#if defined(VK_KHR_cooperative_matrix)
    deviceInfo.cooopertavieMatrixProperties = cooperativeMatrixProperties;
#endif
#if defined(VK_KHR_cooperative_matrix) && defined(VK_NV_cooperative_matrix2)
    deviceInfo.cooperativeMatrix2Properties = matrix2Properties;
    auto flexibleProperties = reinterpret_cast<PFN_vkGetPhysicalDeviceCooperativeMatrixFlexibleDimensionsPropertiesNV>(
      vkGetInstanceProcAddr(instance, "vkGetPhysicalDeviceCooperativeMatrixFlexibleDimensionsPropertiesNV")
    );
    if(hasMatrix2 && flexibleProperties != nullptr) {
      uint32_t count = 0;
      if(flexibleProperties(physicalDevice, &count, nullptr) == VK_SUCCESS) {
        deviceInfo.flexibleMatrixProperties.resize(count);
        for(auto& property: deviceInfo.flexibleMatrixProperties)
          property.sType = VK_STRUCTURE_TYPE_COOPERATIVE_MATRIX_FLEXIBLE_DIMENSIONS_PROPERTIES_NV;
        if(flexibleProperties(physicalDevice, &count, deviceInfo.flexibleMatrixProperties.data()) != VK_SUCCESS)
          deviceInfo.flexibleMatrixProperties.clear();
        else
          deviceInfo.flexibleMatrixProperties.resize(count);
      }
    }
#endif
    deviceInfo.deviceType = deviceInfo.properties.deviceType;

    // Get device name
    deviceInfo.deviceName = std::string(deviceInfo.properties.deviceName);
    if ( logger != nullptr ) {
      logger->write("  Name: " + deviceInfo.deviceName);
      logger->write("  Type: " + vk_helper::vkPhysicalDeviceTypeToString(deviceInfo.deviceType));
      logger->write("  API Version: " +
        std::to_string(VK_VERSION_MAJOR(deviceInfo.properties.apiVersion)) + "." +
        std::to_string(VK_VERSION_MINOR(deviceInfo.properties.apiVersion)) + "." +
        std::to_string(VK_VERSION_PATCH(deviceInfo.properties.apiVersion))
      );
    }
    // Get device features
    VkPhysicalDeviceFeatures2 features2 = {};
    features2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    VkPhysicalDevice16BitStorageFeatures storage16BitFeatures = {};
    VkPhysicalDeviceShaderFloat16Int8Features shaderFloat16Int8Features = {};
    VkPhysicalDeviceVulkanMemoryModelFeatures vulkanMemoryModelFeatures = {};
#if defined(VK_KHR_cooperative_matrix)
    VkPhysicalDeviceCooperativeMatrixFeaturesKHR cooperativeMatrixFeatures = {};
#endif
#if defined(VK_KHR_cooperative_matrix) && defined(VK_NV_cooperative_matrix2)
    VkPhysicalDeviceCooperativeMatrix2FeaturesNV cooperativeMatrix2Features = {};
#endif
#if defined(VK_KHR_maintenance4)
    VkPhysicalDeviceMaintenance4FeaturesKHR maintenance4Features = {};
#endif
#if defined(VK_EXT_subgroup_size_control)
    VkPhysicalDeviceSubgroupSizeControlFeaturesEXT subgroupSizeControlFeatures = {};
#endif
    storage16BitFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES;
    shaderFloat16Int8Features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES;
    vulkanMemoryModelFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_MEMORY_MODEL_FEATURES;
#if defined(VK_KHR_cooperative_matrix)
    cooperativeMatrixFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR;
#endif
#if defined(VK_KHR_maintenance4)
    maintenance4Features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_4_FEATURES_KHR;
#endif
#if defined(VK_EXT_subgroup_size_control)
    subgroupSizeControlFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_FEATURES_EXT;
#endif
#if defined(VK_KHR_cooperative_matrix) && defined(VK_NV_cooperative_matrix2)
    cooperativeMatrix2Features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_2_FEATURES_NV;
#endif
    auto addFeature = [&features2](auto& feature) {
      feature.pNext = features2.pNext;
      features2.pNext = &feature;
    };
    if(deviceInfo.apiVersion >= VK_API_VERSION_1_1 || hasDeviceExtension(deviceInfo, VK_KHR_16BIT_STORAGE_EXTENSION_NAME))
      addFeature(storage16BitFeatures);
    if(core12 || hasDeviceExtension(deviceInfo, VK_KHR_SHADER_FLOAT16_INT8_EXTENSION_NAME))
      addFeature(shaderFloat16Int8Features);
    if(core12 || hasDeviceExtension(deviceInfo, VK_KHR_VULKAN_MEMORY_MODEL_EXTENSION_NAME))
      addFeature(vulkanMemoryModelFeatures);
#if defined(VK_KHR_cooperative_matrix)
    if(hasCooperativeMatrix)
      addFeature(cooperativeMatrixFeatures);
#endif
#if defined(VK_KHR_maintenance4)
    if(hasMaintenance4)
      addFeature(maintenance4Features);
#endif
#if defined(VK_EXT_subgroup_size_control)
    if(hasSubgroupSizeControl)
      addFeature(subgroupSizeControlFeatures);
#endif
#if defined(VK_KHR_cooperative_matrix) && defined(VK_NV_cooperative_matrix2)
    if(hasMatrix2)
      addFeature(cooperativeMatrix2Features);
#endif
    vkGetPhysicalDeviceFeatures2(physicalDevice, &features2);
    deviceInfo.features = features2.features;
    deviceInfo.storage16BitFeatures = storage16BitFeatures;
    deviceInfo.shaderFloat16Int8Features = shaderFloat16Int8Features;
    deviceInfo.vulkanMemoryModelFeatures = vulkanMemoryModelFeatures;
#if defined(VK_KHR_cooperative_matrix)
    deviceInfo.cooperativeMatrixFeatures = cooperativeMatrixFeatures;
#endif
#if defined(VK_KHR_cooperative_matrix) && defined(VK_NV_cooperative_matrix2) && defined(VK_KHR_maintenance4)
    deviceInfo.cooperativeMatrix2Features = cooperativeMatrix2Features;
    deviceInfo.canUseRegisterAttentionNHWC =
      VULKAN_HAS_NV2_SHADER &&
      hasMatrix2 && cooperativeMatrixFeatures.cooperativeMatrix &&
      maintenance4Features.maintenance4 && vulkanMemoryModelFeatures.vulkanMemoryModel &&
      cooperativeMatrix2Features.cooperativeMatrixWorkgroupScope &&
      cooperativeMatrix2Features.cooperativeMatrixFlexibleDimensions &&
      cooperativeMatrix2Features.cooperativeMatrixReductions &&
      cooperativeMatrix2Features.cooperativeMatrixConversions &&
      cooperativeMatrix2Features.cooperativeMatrixPerElementOperations;
#endif
#if defined(VK_KHR_maintenance4)
    deviceInfo.maintenance4Features = maintenance4Features;
#endif
#if defined(VK_EXT_subgroup_size_control)
    deviceInfo.subgroupSizeControlFeatures = subgroupSizeControlFeatures;
#endif

    auto writeFeatureSupport = [logger](const std::string& message) {
      if(logger != nullptr)
        logger->write(message);
      if(logger == nullptr || (!logger->isLoggingToStdout() && !logger->isLoggingToStderr()))
        std::cerr << message << std::endl;
    };
    writeFeatureSupport("Vulkan device features: " + deviceInfo.deviceName);
    writeFeatureSupport("  Shader Float16 Support: " + std::string(deviceInfo.shaderFloat16Int8Features.shaderFloat16 == VK_TRUE ? "Yes" : "No"));
    writeFeatureSupport("  Shader Int8 Support: " + std::string(deviceInfo.shaderFloat16Int8Features.shaderInt8 == VK_TRUE ? "Yes" : "No"));
    writeFeatureSupport("  Storage Buffer 16Bit Support: " + std::string(deviceInfo.storage16BitFeatures.storageBuffer16BitAccess == VK_TRUE ? "Yes" : "No"));
    writeFeatureSupport("  Vulkan Memory Model Support: " + std::string(deviceInfo.vulkanMemoryModelFeatures.vulkanMemoryModel == VK_TRUE ? "Yes" : "No"));
#if defined(VK_KHR_cooperative_matrix)
    writeFeatureSupport("  Cooperative Matrix Support: " + std::string(deviceInfo.cooperativeMatrixFeatures.cooperativeMatrix == VK_TRUE ? "Yes" : "No"));
#endif
#if defined(VK_KHR_cooperative_matrix)
    writeFeatureSupport("  Cooperative Matrix Robust Buffer Access Support: " + std::string(deviceInfo.cooperativeMatrixFeatures.cooperativeMatrixRobustBufferAccess == VK_TRUE ? "Yes" : "No"));
#endif
#if defined(VK_KHR_maintenance4)
    writeFeatureSupport("  Maintenance4 Support: " + std::string(deviceInfo.maintenance4Features.maintenance4 == VK_TRUE ? "Yes" : "No"));
#endif
#if defined(VK_EXT_subgroup_size_control)
    writeFeatureSupport("  Subgroup Size Control Support: " + std::string(deviceInfo.subgroupSizeControlFeatures.subgroupSizeControl == VK_TRUE ? "Yes" : "No"));
#endif
#if defined(VK_EXT_subgroup_size_control)
    writeFeatureSupport("  Compute Full Subgroups Support: " + std::string(deviceInfo.subgroupSizeControlFeatures.computeFullSubgroups == VK_TRUE ? "Yes" : "No"));
#endif
#if defined(VK_KHR_cooperative_matrix)
    if(deviceInfo.cooperativeMatrixFeatures.cooperativeMatrix == VK_TRUE) {
      const auto componentTypeName = [](VkComponentTypeKHR type) {
        switch(type) {
        case VK_COMPONENT_TYPE_FLOAT16_KHR: return std::string("FLOAT16");
        case VK_COMPONENT_TYPE_FLOAT32_KHR: return std::string("FLOAT32");
        case VK_COMPONENT_TYPE_FLOAT64_KHR: return std::string("FLOAT64");
        case VK_COMPONENT_TYPE_SINT8_KHR: return std::string("SINT8");
        case VK_COMPONENT_TYPE_SINT16_KHR: return std::string("SINT16");
        case VK_COMPONENT_TYPE_SINT32_KHR: return std::string("SINT32");
        case VK_COMPONENT_TYPE_SINT64_KHR: return std::string("SINT64");
        case VK_COMPONENT_TYPE_UINT8_KHR: return std::string("UINT8");
        case VK_COMPONENT_TYPE_UINT16_KHR: return std::string("UINT16");
        case VK_COMPONENT_TYPE_UINT32_KHR: return std::string("UINT32");
        case VK_COMPONENT_TYPE_UINT64_KHR: return std::string("UINT64");
        default: return "UNKNOWN(" + std::to_string(static_cast<int>(type)) + ")";
        }
      };
      std::vector<VkCooperativeMatrixPropertiesKHR> cooperativeProperties;
      while(deviceInfo.cooperativeMatrixPropertiesFn != nullptr) {
        uint32_t propertyCount = 0;
        VkResult propertyResult = deviceInfo.cooperativeMatrixPropertiesFn(
          physicalDevice, &propertyCount, nullptr
        );
        if((propertyResult != VK_SUCCESS && propertyResult != VK_INCOMPLETE) || propertyCount == 0)
          break;
        cooperativeProperties.assign(propertyCount, {});
        for(VkCooperativeMatrixPropertiesKHR& property: cooperativeProperties) {
          property.sType = VK_STRUCTURE_TYPE_COOPERATIVE_MATRIX_PROPERTIES_KHR;
          property.pNext = nullptr;
        }
        propertyResult = deviceInfo.cooperativeMatrixPropertiesFn(
          physicalDevice, &propertyCount, cooperativeProperties.data()
        );
        if(propertyResult == VK_INCOMPLETE)
          continue;
        if(propertyResult != VK_SUCCESS)
          cooperativeProperties.clear();
        else
          cooperativeProperties.resize(propertyCount);
        break;
      }
      for(size_t propertyIndex = 0; propertyIndex < cooperativeProperties.size(); propertyIndex++) {
        const VkCooperativeMatrixPropertiesKHR& property = cooperativeProperties[propertyIndex];
        writeFeatureSupport(
          "  Cooperative Matrix Property[" + std::to_string(propertyIndex) + "]: " +
          "M=" + std::to_string(property.MSize) +
          " N=" + std::to_string(property.NSize) +
          " K=" + std::to_string(property.KSize) +
          " AType=" + componentTypeName(property.AType) +
          " BType=" + componentTypeName(property.BType) +
          " CType=" + componentTypeName(property.CType) +
          " ResultType=" + componentTypeName(property.ResultType)
        );
      }
    }
#endif

    // Get memory properties2
    VkPhysicalDeviceMemoryProperties2 memoryProperties2 = {};
    memoryProperties2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2;
    // Cooperative Matrix features
    memoryProperties2.pNext = nullptr;
    vkGetPhysicalDeviceMemoryProperties2(physicalDevice, &memoryProperties2);
    deviceInfo.memoryProperties = memoryProperties2.memoryProperties;
    
    // Query chains point to stack storage and must not escape enumeration.
    deviceInfo.subgroupProperties.pNext = nullptr;
#if defined(VK_EXT_subgroup_size_control)
    deviceInfo.subgroupSizeProperties.pNext = nullptr;
#endif
#if defined(VK_KHR_cooperative_matrix)
    deviceInfo.cooopertavieMatrixProperties.pNext = nullptr;
#endif
#if defined(VK_KHR_cooperative_matrix) && defined(VK_NV_cooperative_matrix2)
    deviceInfo.cooperativeMatrix2Properties.pNext = nullptr;
#endif
    deviceInfo.storage16BitFeatures.pNext = nullptr;
    deviceInfo.shaderFloat16Int8Features.pNext = nullptr;
    deviceInfo.vulkanMemoryModelFeatures.pNext = nullptr;
#if defined(VK_KHR_cooperative_matrix)
    deviceInfo.cooperativeMatrixFeatures.pNext = nullptr;
#endif
#if defined(VK_KHR_cooperative_matrix) && defined(VK_NV_cooperative_matrix2)
    deviceInfo.cooperativeMatrix2Features.pNext = nullptr;
#endif
#if defined(VK_KHR_maintenance4)
    deviceInfo.maintenance4Features.pNext = nullptr;
#endif
#if defined(VK_EXT_subgroup_size_control)
    deviceInfo.subgroupSizeControlFeatures.pNext = nullptr;
#endif

    deviceInfos.push_back(deviceInfo);
  }

  return deviceInfos;
}

VulkanDevice* vk_helper::createVulkanDevice(
  VkInstance instance,
  VulkanDeviceInfo deviceInfo,
  std::vector<const char *> requiredExtensions,
  Logger* logger,
  float dummyQueuePriority
) {
  if ( logger ) {
    logger->write("Creating Vulkan Logical Device for " + deviceInfo.deviceName);
  }
  VkPhysicalDevice physicalDevice = deviceInfo.physicalDevice;
  if(deviceInfo.apiVersion < VK_API_VERSION_1_1)
    throw StringError("Device " + deviceInfo.deviceName + " does not support the Vulkan 1.1 shader target");
  const bool core12 = deviceInfo.apiVersion >= VK_API_VERSION_1_2;
  const bool core13 = deviceInfo.apiVersion >= VK_MAKE_VERSION(1, 3, 0);
  // Existing callers may request names for features promoted into core Vulkan.
  requiredExtensions.erase(std::remove_if(requiredExtensions.begin(), requiredExtensions.end(), [core12, core13](const char* extension) {
    const std::string name(extension);
    return name == VK_KHR_16BIT_STORAGE_EXTENSION_NAME ||
      (core12 && (name == VK_KHR_SHADER_FLOAT16_INT8_EXTENSION_NAME || name == VK_KHR_VULKAN_MEMORY_MODEL_EXTENSION_NAME)) ||
      (core13 && (name == "VK_EXT_subgroup_size_control" || name == "VK_KHR_maintenance4"));
  }), requiredExtensions.end());
  auto enableExtension = [&requiredExtensions, &deviceInfo](const char* name) {
    if(!hasDeviceExtension(deviceInfo, name))
      throw StringError("Required device extension " + std::string(name) + " is not available on device " + deviceInfo.deviceName);
    if(std::none_of(requiredExtensions.begin(), requiredExtensions.end(), [name](const char* extension) {
      return std::string(extension) == name;
    }))
      requiredExtensions.push_back(name);
  };
  for(const char* extension: requiredExtensions) {
    if(!hasDeviceExtension(deviceInfo, extension))
      throw StringError("Required device extension " + std::string(extension) + " is not available on device " + deviceInfo.deviceName);
  }
  if(hasDeviceExtension(deviceInfo, "VK_KHR_portability_subset"))
    enableExtension("VK_KHR_portability_subset");
  const bool shaderFloat16Available = core12 ||
    hasDeviceExtension(deviceInfo, VK_KHR_SHADER_FLOAT16_INT8_EXTENSION_NAME);
  const bool enableShaderFloat16 = shaderFloat16Available &&
    deviceInfo.shaderFloat16Int8Features.shaderFloat16 == VK_TRUE;
  if(enableShaderFloat16 && !core12)
    enableExtension(VK_KHR_SHADER_FLOAT16_INT8_EXTENSION_NAME);
  const bool vulkanMemoryModelAvailable = core12 ||
    hasDeviceExtension(deviceInfo, VK_KHR_VULKAN_MEMORY_MODEL_EXTENSION_NAME);
  const bool enableVulkanMemoryModel = vulkanMemoryModelAvailable &&
    deviceInfo.vulkanMemoryModelFeatures.vulkanMemoryModel == VK_TRUE;
  if(enableVulkanMemoryModel && !core12)
    enableExtension(VK_KHR_VULKAN_MEMORY_MODEL_EXTENSION_NAME);
#if defined(VK_KHR_cooperative_matrix)
  const bool enableCooperativeMatrix =
    hasDeviceExtension(deviceInfo, VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME) &&
    deviceInfo.cooperativeMatrixFeatures.cooperativeMatrix == VK_TRUE &&
    enableVulkanMemoryModel;
  if(enableCooperativeMatrix)
    enableExtension(VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME);
#endif
#if defined(VK_KHR_cooperative_matrix) && defined(VK_NV_cooperative_matrix2) && defined(VK_KHR_maintenance4)
  const bool enableRegisterAttentionNHWC = enableCooperativeMatrix && core13 &&
    deviceInfo.canUseRegisterAttentionNHWC && deviceInfo.maintenance4Features.maintenance4 == VK_TRUE &&
    hasDeviceExtension(deviceInfo, VK_NV_COOPERATIVE_MATRIX_2_EXTENSION_NAME);
  if(enableRegisterAttentionNHWC)
    enableExtension(VK_NV_COOPERATIVE_MATRIX_2_EXTENSION_NAME);
#else
  const bool enableRegisterAttentionNHWC = false;
#endif
#if defined(VK_EXT_subgroup_size_control)
  const bool enableComputeFullSubgroups =
    deviceInfo.subgroupSizeControlFeatures.computeFullSubgroups == VK_TRUE &&
    (core13 || hasDeviceExtension(deviceInfo, VK_EXT_SUBGROUP_SIZE_CONTROL_EXTENSION_NAME));
  if(enableComputeFullSubgroups && !core13)
    enableExtension(VK_EXT_SUBGROUP_SIZE_CONTROL_EXTENSION_NAME);
#endif
#if defined(VK_KHR_maintenance4)
  const bool enableMaintenance4 = deviceInfo.maintenance4Features.maintenance4 == VK_TRUE &&
    (core13 || hasDeviceExtension(deviceInfo, VK_KHR_MAINTENANCE_4_EXTENSION_NAME));
  if(enableMaintenance4 && !core13)
    enableExtension(VK_KHR_MAINTENANCE_4_EXTENSION_NAME);
#endif

  const float queuePriorities[2] = {1.0f, dummyQueuePriority};
  uint32_t queueFamilyPropertyCount = 0;
  vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyPropertyCount, nullptr);
  std::vector<VkQueueFamilyProperties> queueFamilyProperties(queueFamilyPropertyCount);
  vkGetPhysicalDeviceQueueFamilyProperties(
    physicalDevice, &queueFamilyPropertyCount, queueFamilyProperties.data()
  );
  const auto computeFamily = std::find_if(queueFamilyProperties.begin(), queueFamilyProperties.end(), [](const VkQueueFamilyProperties& family) {
    return family.queueCount > 0 && (family.queueFlags & VK_QUEUE_COMPUTE_BIT) != 0;
  });
  if(computeFamily == queueFamilyProperties.end())
    throw StringError("No compute queue family available on device " + deviceInfo.deviceName);
  const uint32_t queueFamilyIndex = static_cast<uint32_t>(computeFamily - queueFamilyProperties.begin());
  const uint32_t queueCount = computeFamily->queueCount >= 2 ? 2 : 1;
  VkDeviceQueueCreateInfo queueCI = {};
  queueCI.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
  queueCI.queueFamilyIndex = queueFamilyIndex;
  queueCI.queueCount = queueCount;
  queueCI.pQueuePriorities = queuePriorities;

  VkPhysicalDeviceFeatures2 requestedFeatures = {};
  requestedFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
  requestedFeatures.features = deviceInfo.features; // start with all available features, we'll disable unsupported

  VkPhysicalDevice16BitStorageFeatures storage16BitFeatures = {};
  storage16BitFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES;
  storage16BitFeatures.storageBuffer16BitAccess = deviceInfo.storage16BitFeatures.storageBuffer16BitAccess;
  storage16BitFeatures.uniformAndStorageBuffer16BitAccess = deviceInfo.storage16BitFeatures.uniformAndStorageBuffer16BitAccess;
  VkPhysicalDeviceShaderFloat16Int8Features f16Feat = {};
  f16Feat.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES;
  f16Feat.shaderFloat16 = enableShaderFloat16 ? VK_TRUE : VK_FALSE;
  VkPhysicalDeviceVulkanMemoryModelFeatures vulkanMemoryModelFeatures = {};
  vulkanMemoryModelFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_MEMORY_MODEL_FEATURES;
  vulkanMemoryModelFeatures.vulkanMemoryModel = enableVulkanMemoryModel ? VK_TRUE : VK_FALSE;
#if defined(VK_KHR_cooperative_matrix)
  VkPhysicalDeviceCooperativeMatrixFeaturesKHR cmFeatures = {};
  cmFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR;
  cmFeatures.cooperativeMatrix = enableCooperativeMatrix ? VK_TRUE : VK_FALSE;
#endif
#if defined(VK_KHR_maintenance4)
  VkPhysicalDeviceMaintenance4FeaturesKHR m4Features = {};
  m4Features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_4_FEATURES_KHR;
  m4Features.maintenance4 = enableMaintenance4 ? VK_TRUE : VK_FALSE;
#endif
#if defined(VK_EXT_subgroup_size_control)
  VkPhysicalDeviceSubgroupSizeControlFeaturesEXT subgroupSizeControlFeatures = {};
  subgroupSizeControlFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_FEATURES_EXT;
  if(enableComputeFullSubgroups) {
    subgroupSizeControlFeatures.subgroupSizeControl = deviceInfo.subgroupSizeControlFeatures.subgroupSizeControl;
    subgroupSizeControlFeatures.computeFullSubgroups = VK_TRUE;
  }
#endif
#if defined(VK_KHR_cooperative_matrix) && defined(VK_NV_cooperative_matrix2)
  VkPhysicalDeviceCooperativeMatrix2FeaturesNV cm2Features = {};
  cm2Features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_2_FEATURES_NV;
  if(enableRegisterAttentionNHWC) {
    cm2Features.cooperativeMatrixWorkgroupScope = VK_TRUE;
    cm2Features.cooperativeMatrixFlexibleDimensions = VK_TRUE;
    cm2Features.cooperativeMatrixReductions = VK_TRUE;
    cm2Features.cooperativeMatrixConversions = VK_TRUE;
    cm2Features.cooperativeMatrixPerElementOperations = VK_TRUE;
  }
#endif
  auto addFeature = [&requestedFeatures](auto& feature) {
    feature.pNext = requestedFeatures.pNext;
    requestedFeatures.pNext = &feature;
  };
  addFeature(storage16BitFeatures);
  if(core12 || enableShaderFloat16)
    addFeature(f16Feat);
  if(core12 || enableVulkanMemoryModel)
    addFeature(vulkanMemoryModelFeatures);
#if defined(VK_KHR_cooperative_matrix)
  if(enableCooperativeMatrix)
    addFeature(cmFeatures);
#endif
#if defined(VK_KHR_maintenance4)
  if(enableMaintenance4)
    addFeature(m4Features);
#endif
#if defined(VK_EXT_subgroup_size_control)
  if(enableComputeFullSubgroups)
    addFeature(subgroupSizeControlFeatures);
#endif
#if defined(VK_KHR_cooperative_matrix) && defined(VK_NV_cooperative_matrix2)
  if(enableRegisterAttentionNHWC)
    addFeature(cm2Features);
#endif

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

  std::unique_ptr<VulkanDevice> vulkanDevice(new VulkanDevice());
  vulkanDevice->info = deviceInfo;
  vulkanDevice->queueFamilyIndex = queueFamilyIndex;
  vulkanDevice->info.canUseRegisterAttentionNHWC = enableRegisterAttentionNHWC;
  vulkanDevice->info.storage16BitFeatures = storage16BitFeatures;
  vulkanDevice->info.storage16BitFeatures.pNext = nullptr;
  vulkanDevice->info.shaderFloat16Int8Features = f16Feat;
  vulkanDevice->info.shaderFloat16Int8Features.pNext = nullptr;
  vulkanDevice->info.vulkanMemoryModelFeatures = vulkanMemoryModelFeatures;
  vulkanDevice->info.vulkanMemoryModelFeatures.pNext = nullptr;
#if defined(VK_KHR_cooperative_matrix)
  vulkanDevice->info.cooperativeMatrixFeatures = cmFeatures;
  vulkanDevice->info.cooperativeMatrixFeatures.pNext = nullptr;
#endif
#if defined(VK_KHR_maintenance4)
  vulkanDevice->info.maintenance4Features = m4Features;
  vulkanDevice->info.maintenance4Features.pNext = nullptr;
#endif
#if defined(VK_EXT_subgroup_size_control)
  vulkanDevice->info.subgroupSizeControlFeatures = subgroupSizeControlFeatures;
  vulkanDevice->info.subgroupSizeControlFeatures.pNext = nullptr;
#endif
#if defined(VK_KHR_cooperative_matrix) && defined(VK_NV_cooperative_matrix2)
  vulkanDevice->info.cooperativeMatrix2Features = cm2Features;
  vulkanDevice->info.cooperativeMatrix2Features.pNext = nullptr;
#endif
  vulkanDevice->device = device;
  vulkanDevice->queue = queue;
  if(queueCount >= 2)
    vkGetDeviceQueue(device, queueCI.queueFamilyIndex, 1, &vulkanDevice->dummyQueue);

  VmaAllocator allocator = VK_NULL_HANDLE;

  VmaAllocatorCreateInfo allocatorCI = {};
  allocatorCI.physicalDevice = physicalDevice;
  allocatorCI.device = device;
  allocatorCI.instance = instance;

  allocatorCI.vulkanApiVersion = deviceInfo.apiVersion;

  res = vmaCreateAllocator(&allocatorCI, &allocator);
  CHECK_VK_MSG("VMA create for device : " + deviceInfo.deviceName, res);
  vulkanDevice->allocator = allocator;

  return vulkanDevice.release();
}

VkShaderModule vk_helper::createShaderModuleFromSPIRVBytes(
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

VkPipelineCache vk_helper::createPipelineCache(
  VkDevice device,
  VkResult *result
) {
  VkPipelineCacheCreateInfo pipelineCacheCI = {};
  pipelineCacheCI.sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO;
  VkPipelineCache pipelineCache = VK_NULL_HANDLE;
  *result = vkCreatePipelineCache(device, &pipelineCacheCI, nullptr, &pipelineCache);
  return pipelineCache;
}

VkPipelineLayout vk_helper::createPipelineLayout(
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

VkPipeline vk_helper::createComputePipeline(
    VkDevice device,
    VkPipelineLayout pipelineLayout,
    VkPipelineCache pipelineCache,
    VkShaderModule computeShaderModule,
    VkResult *result,
    VkSpecializationInfo* specializationInfo,
    std::string entryPointName,
    VkPipelineShaderStageCreateFlags stageFlags
) {
  VkComputePipelineCreateInfo pipelineCI = {};
  pipelineCI.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
  pipelineCI.layout = pipelineLayout;
  pipelineCI.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  pipelineCI.stage.flags = stageFlags;
  pipelineCI.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
  pipelineCI.stage.module = computeShaderModule;
  pipelineCI.stage.pName = entryPointName.data();
  pipelineCI.stage.pSpecializationInfo = specializationInfo;
  VkPipeline pipeline = VK_NULL_HANDLE;
  *result = vkCreateComputePipelines(device, pipelineCache, 1, &pipelineCI, nullptr, &pipeline);
  return pipeline;
}

VkDescriptorSetLayout vk_helper::createDescriptorSetLayout(
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

VkDescriptorPool vk_helper::createDescriptorPool(
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

VkDescriptorSet vk_helper::allocateDescriptorSet(
  const VulkanDevice *device,
  VkDescriptorSetLayout descriptorSetLayout,
  VkResult *result,
  VkDescriptorPool descriptorPool
) {
  assert(descriptorPool != VK_NULL_HANDLE);
  VkDescriptorSetAllocateInfo allocInfo = {};
  allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
  allocInfo.descriptorPool = descriptorPool;
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

VkResult vk_helper::updateDescriptorSets(
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

VkCommandPool vk_helper::createCommandPool(
  const VulkanDevice *device,
  VkResult *result
) {
  VkCommandPoolCreateInfo poolCI = {};
  poolCI.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
  poolCI.queueFamilyIndex = device->queueFamilyIndex;
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

VkCommandBuffer vk_helper::allocateCommandBuffer(
  const VulkanDevice *device,
  VkResult* result,
  VkCommandPool commandPool
) {
  assert(commandPool != VK_NULL_HANDLE);
  VkCommandBufferAllocateInfo allocInfo = {};
  allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  allocInfo.commandPool = commandPool;
  allocInfo.commandBufferCount = 1;

  VkCommandBuffer commandBuffer = VK_NULL_HANDLE;
  VkResult allocationResult = vkAllocateCommandBuffers(device->device, &allocInfo, &commandBuffer);
  if(result != nullptr)
    *result = allocationResult;
  return commandBuffer;
}

VkResult vk_helper::beginCommandBuffer(
  VkCommandBuffer commandBuffer
) {
  VkCommandBufferBeginInfo beginInfo = {};
  beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  beginInfo.pInheritanceInfo = nullptr; // Not required.
  beginInfo.flags = 0;//VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

  return vkBeginCommandBuffer(commandBuffer, &beginInfo);
}

VkResult vk_helper::endCommandBuffer(
  VkCommandBuffer commandBuffer
) {
  return vkEndCommandBuffer(commandBuffer);
}

VkResult vk_helper::submitCommandBuffers(
  const VulkanDevice *device,
  const std::vector<VkCommandBuffer>& commandBuffers,
  VkFence fence
) {
  std::lock_guard<std::mutex> lock(device->queueSubmitMutex);
  VkSubmitInfo submitInfo = {};
  submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  submitInfo.commandBufferCount = static_cast<uint32_t>(commandBuffers.size());
  submitInfo.pCommandBuffers = commandBuffers.data();
  return vkQueueSubmit(device->queue, 1, &submitInfo, fence);
}

VkCommandBuffer vk_helper::beginSingleTimeCommandBuffer(
  const VulkanDevice *device,
  VkCommandPool commandPool
) {
  assert(commandPool != VK_NULL_HANDLE);
  VkCommandBufferAllocateInfo allocInfo = {};
  allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  allocInfo.commandPool = commandPool;
  allocInfo.commandBufferCount = 1;

  VkCommandBuffer commandBuffer;
  vkAllocateCommandBuffers(device->device, &allocInfo, &commandBuffer);

  VkCommandBufferBeginInfo beginInfo = {};
  beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

  vkBeginCommandBuffer(commandBuffer, &beginInfo);
  return commandBuffer;
}

void vk_helper::submitSingleTimeCommandBufferAndWaitIdle(
  const VulkanDevice *device,
  VkCommandBuffer commandBuffer,
  VkCommandPool commandPool
) {
  assert(commandPool != VK_NULL_HANDLE);
  vkEndCommandBuffer(commandBuffer);
  VkSubmitInfo submitInfo = {};
  submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  submitInfo.commandBufferCount = 1;
  submitInfo.pCommandBuffers = &commandBuffer;
  {
    std::lock_guard<std::mutex> lock(device->queueSubmitMutex);
    CHECK_VK_MSG("vk_helper::submitSingleTimeCommandBufferAndWaitIdle",vkQueueSubmit(device->queue, 1, &submitInfo, VK_NULL_HANDLE));
    CHECK_VK_MSG("vk_helper::waitIdle",vkQueueWaitIdle(device->queue));
  }
  vkFreeCommandBuffers(device->device, commandPool, 1, &commandBuffer);
}


VulkanBuffer* vk_helper::createReadOnlyBuffer(
  const VulkanDevice* device,
  const std::vector<float>& data,
  const bool useFP16,
  VkResult* result,
  VkCommandPool commandPool
) {
  if ( useFP16 ) {
    std::vector<half_t> dataHalf(data.size());
    for ( size_t i = 0 ; i < data.size() ; ++i ) {
      dataHalf[i] = half_float::half_cast<half_t>(data[i]);
    }
    return createDeviceBufferWithData(
        device,
        sizeof(half_t) * dataHalf.size(),
        dataHalf.data(),
        true,
        result,
        commandPool
      );
  }
  return createDeviceBufferWithData(
    device,
    sizeof(float) * data.size(),
    data.data(),
    true,
    result,
    commandPool
  );
}


VulkanBuffer* vk_helper::createReadWriteBuffer(
  const VulkanDevice* device,
  const std::vector<float>& data,
  const bool useFP16,
  VkResult* result,
  VkCommandPool commandPool
) {
  if ( useFP16 ) {
    std::vector<half_t> halfData(data.size());
    for(size_t i = 0 ; i < data.size() ; ++i ) {
      halfData[i] = half_float::half_cast<half_t>(data[i]);
    }
    return createDeviceBufferWithData(
      device,
      sizeof(half_t) * halfData.size(),
      halfData.data(),
      false,
      result,
      commandPool
    );
  }
  return createDeviceBufferWithData(
    device,
    sizeof(float) * data.size(),
    data.data(),
    false,
    result,
    commandPool
  );
}

VulkanBuffer* vk_helper::createDeviceBuffer(
  const VulkanDevice *device,
  size_t size,
  bool readOnly,
  VkResult *result
) {
  if ( !result ) {
    throw StringError("vk_helper::createDeviceBuffer: result pointer is null");
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

VulkanBuffer* vk_helper::createDeviceBufferWithData(
  const VulkanDevice *device,
  size_t size,
  const void* data,
  bool readOnly,
  VkResult *result,
  VkCommandPool commandPool
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
  VkCommandBuffer commandBuffer = vk_helper::beginSingleTimeCommandBuffer(device, commandPool);
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
  vk_helper::barrierCommandBufferForBuffer(
    commandBuffer, deviceBuffer,
    VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT
  );
  vk_helper::submitSingleTimeCommandBufferAndWaitIdle(
    device,
    commandBuffer,
    commandPool
  );
  // Release staging buffer
  vk_helper::releaseVulkanBuffer(
    device,
    stagingBuffer
  );
  return deviceBuffer;
}

VulkanBuffer* vk_helper::createStagingBuffer(
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

VulkanBuffer* vk_helper::createReadbackBuffer(
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

void vk_helper::copyHostToStagingBuffer(
  const VulkanDevice* device,
  const void* hostPtr,
  VulkanBuffer* stagingBuffer,
  VkDeviceSize stagingOffset,
  VkDeviceSize copySize,
  VkResult* result
) {
  if(stagingOffset + copySize > stagingBuffer->requestedSize || stagingBuffer->allocationInfo.pMappedData == nullptr) {
    *result = VK_ERROR_MEMORY_MAP_FAILED;
    return;
  }
  memcpy(
    static_cast<char*>(stagingBuffer->allocationInfo.pMappedData) + stagingOffset,
    hostPtr,
    static_cast<size_t>(copySize)
  );
  *result = vmaFlushAllocation(device->allocator, stagingBuffer->allocation, stagingOffset, copySize);
}

void vk_helper::recordBufferCopy(
  VkCommandBuffer commandBuffer,
  VulkanBuffer* sourceBuffer,
  VulkanBuffer* destinationBuffer,
  VkDeviceSize sourceOffset,
  VkDeviceSize destinationOffset,
  VkDeviceSize copySize
) {
  VkBufferCopy copyRegion = {};
  copyRegion.srcOffset = sourceOffset;
  copyRegion.dstOffset = destinationOffset;
  copyRegion.size = copySize;
  vkCmdCopyBuffer(commandBuffer, sourceBuffer->buffer, destinationBuffer->buffer, 1, &copyRegion);
}

void vk_helper::copyReadbackBufferToHost(
  const VulkanDevice* device,
  VulkanBuffer* readbackBuffer,
  VkDeviceSize readbackOffset,
  VkDeviceSize copySize,
  void* hostPtr,
  VkResult* result
) {
  if(readbackOffset + copySize > readbackBuffer->requestedSize || readbackBuffer->allocationInfo.pMappedData == nullptr) {
    *result = VK_ERROR_MEMORY_MAP_FAILED;
    return;
  }
  *result = vmaInvalidateAllocation(device->allocator, readbackBuffer->allocation, readbackOffset, copySize);
  if(*result != VK_SUCCESS)
    return;
  memcpy(
    hostPtr,
    static_cast<const char*>(readbackBuffer->allocationInfo.pMappedData) + readbackOffset,
    static_cast<size_t>(copySize)
  );
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
void vk_helper::copyDeviceBufferToHost(
  const VulkanDevice* device,
  VulkanBuffer* deviceBuffer,
  VkDeviceSize copySize,
  void* hostPtr,
  bool waitForIdle,
  VkResult *result,
  VkCommandPool commandPool
) {
  (void)waitForIdle;
  // Create readback buffer
  VulkanBuffer* readbackBuffer = vk_helper::createReadbackBuffer(
    device,
    static_cast<size_t>(copySize),
    result
  );

  if(*result != VK_SUCCESS) {
    return;
  }

  // Copy device buffer to readback buffer
  VkCommandBuffer commandBuffer = vk_helper::beginSingleTimeCommandBuffer(device, commandPool);
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
  vk_helper::barrierCommandBufferForBuffer(
    commandBuffer, readbackBuffer,
    VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
    VK_PIPELINE_STAGE_HOST_BIT, VK_ACCESS_HOST_READ_BIT
  );
  vk_helper::submitSingleTimeCommandBufferAndWaitIdle(
    device,
    commandBuffer,
    commandPool
  );

  copyReadbackBufferToHost(device, readbackBuffer, 0, copySize, hostPtr, result);

  // Release readback buffer
  vk_helper::releaseVulkanBuffer(
    device,
    readbackBuffer
  );
}

void vk_helper::copyHostToDeviceBuffer(
  const VulkanDevice* device,
  const void* hostPtr,
  VulkanBuffer* deviceBuffer,
  VkDeviceSize copySize,
  bool waitForIdle,
  VkResult *result,
  VkCommandPool commandPool
) {
  // Create staging buffer
  VulkanBuffer* stagingBuffer = vk_helper::createStagingBuffer(
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
  VkCommandBuffer commandBuffer = vk_helper::beginSingleTimeCommandBuffer(device, commandPool);
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
  vk_helper::barrierCommandBufferForBuffer(
    commandBuffer, deviceBuffer,
    VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT
  );
  vk_helper::submitSingleTimeCommandBufferAndWaitIdle(
    device,
    commandBuffer,
    commandPool
  );

  // Release staging buffer
  vk_helper::releaseVulkanBuffer(
    device,
    stagingBuffer
  );
  
  if(waitForIdle) {
    std::lock_guard<std::mutex> lock(device->queueSubmitMutex);
    vkDeviceWaitIdle(device->device);
  }
}

void vk_helper::releaseVulkanBuffer(
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

VkFence vk_helper::createFence(
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

void vk_helper::destroyFence(
  const VulkanDevice *device,
  VkFence fence
) {
  vkDestroyFence(
    device->device,
    fence,
    nullptr
  );
}

VkResult vk_helper::resetFence(
  const VulkanDevice *device,
  VkFence fence
) {
  return vkResetFences(device->device, 1 , &fence);
}

void vk_helper::barrierCommandBuffer(
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

void vk_helper::barrierCommandBufferForBuffer(
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

std::vector<int32_t> vk_helper::createSpecData(void* data, size_t dataSize) {
  size_t cnt = dataSize / sizeof(int32_t);
  int32_t* intData = reinterpret_cast<int32_t*>(data);
  std::vector<int32_t> specData(intData, intData + cnt);
  return specData;
}

std::vector<VkSpecializationMapEntry> vk_helper::createSpecMapEntries(size_t dataCount) {
  std::vector<VkSpecializationMapEntry> mapEntries(dataCount);
  for ( size_t i = 0; i < dataCount; i++ ) {
    mapEntries[i].constantID = static_cast<uint32_t>(i);
    mapEntries[i].offset = static_cast<uint32_t>(i * sizeof(int32_t));
    mapEntries[i].size = sizeof(int32_t);
  }
  return mapEntries;
}

VkSpecializationInfo vk_helper::createSpecializationInfo(const std::vector<int32_t>& specData, const std::vector<VkSpecializationMapEntry>& mapEntries) {
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
