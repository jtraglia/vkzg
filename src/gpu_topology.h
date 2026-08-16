// Best-effort query of a GPU's real execution-core count, used to size
// dispatch shapes (see recordReduce in vkzg.cpp) from actual
// hardware topology instead of a hardcoded constant or a device-name guess.
#ifndef VKZG_GPU_TOPOLOGY_H
#define VKZG_GPU_TOPOLOGY_H

#include <cstdint>
#include <vulkan/vulkan.h>

namespace vkzg {

// Returns the GPU's total shader-core count (clusters * cores-per-cluster
// for multi-cluster designs), or 0 if it can't be determined -- e.g. on a
// non-Linux platform, a driver without VK_EXT_physical_device_drm, or a
// kernel GPU driver this doesn't know how to query. Callers must treat 0 as
// "unknown" and fall back to a topology-independent default; this is never
// required for correctness, only used to pick a better-tuned dispatch shape.
uint32_t queryGpuTotalCores(VkPhysicalDevice physDev);

} // namespace vkzg

#endif // VKZG_GPU_TOPOLOGY_H
