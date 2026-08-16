// Best-effort query of a GPU's real execution-core count, used by
// recordReduce (vkzg.cpp) to pick a dispatch shape.
#ifndef VKZG_GPU_TOPOLOGY_H
#define VKZG_GPU_TOPOLOGY_H

#include <cstdint>
#include <vulkan/vulkan.h>

namespace vkzg {

// Total shader-core count, or 0 if it can't be determined -- callers must
// treat 0 as "unknown" and fall back to a topology-independent default.
uint32_t queryGpuTotalCores(VkPhysicalDevice physDev);

} // namespace vkzg

#endif // VKZG_GPU_TOPOLOGY_H
