#include "gpu_topology.h"

#include <cstring>
#include <vector>

// Vulkan has no core-count query, so this finds the DRM render node via
// VK_EXT_physical_device_drm and asks the kernel driver directly. Only one
// backend (Asahi) exists right now; add others as separate branches.
#if defined(__linux__) && __has_include(<drm/asahi_drm.h>)
#define VKZG_HAVE_ASAHI_TOPOLOGY 1
#include <drm/asahi_drm.h>
#include <dirent.h>
#include <fcntl.h>
#include <string>
#include <sys/ioctl.h>
#include <sys/stat.h>
#include <sys/sysmacros.h>
#include <unistd.h>
#endif

namespace vkzg {

#if VKZG_HAVE_ASAHI_TOPOLOGY

namespace {

// Vulkan hands back a (major, minor) device number, not a path; recover the
// actual /dev/dri node by scanning for the character device it names.
std::string findDrmNode(int64_t wantMajor, int64_t wantMinor) {
    DIR *dir = opendir("/dev/dri");
    if (!dir) return {};
    std::string result;
    for (struct dirent *entry; (entry = readdir(dir)) != nullptr;) {
        const std::string path = std::string("/dev/dri/") + entry->d_name;
        struct stat st{};
        if (stat(path.c_str(), &st) != 0 || !S_ISCHR(st.st_mode)) continue;
        if ((int64_t)major(st.st_rdev) == wantMajor && (int64_t)minor(st.st_rdev) == wantMinor) {
            result = path;
            break;
        }
    }
    closedir(dir);
    return result;
}

uint32_t queryAsahiTotalCores(const std::string &path) {
    const int fd = open(path.c_str(), O_RDWR);
    if (fd < 0) return 0;

    drm_asahi_params_global params{};
    drm_asahi_get_params req{};
    req.param_group = 0;
    req.pointer = reinterpret_cast<uint64_t>(&params);
    req.size = sizeof(params);
    const int rc = ioctl(fd, DRM_IOCTL_ASAHI_GET_PARAMS, &req);
    close(fd);
    if (rc != 0 || params.num_clusters_total == 0 || params.num_cores_per_cluster == 0) return 0;
    return params.num_clusters_total * params.num_cores_per_cluster;
}

} // namespace

uint32_t queryGpuTotalCores(VkPhysicalDevice physDev) {
    uint32_t extCount = 0;
    vkEnumerateDeviceExtensionProperties(physDev, nullptr, &extCount, nullptr);
    std::vector<VkExtensionProperties> exts(extCount);
    vkEnumerateDeviceExtensionProperties(physDev, nullptr, &extCount, exts.data());
    bool hasDrmExt = false;
    for (const auto &e : exts) {
        if (strcmp(e.extensionName, VK_EXT_PHYSICAL_DEVICE_DRM_EXTENSION_NAME) == 0) {
            hasDrmExt = true;
            break;
        }
    }
    if (!hasDrmExt) return 0;

    VkPhysicalDeviceDrmPropertiesEXT drmProps{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRM_PROPERTIES_EXT};
    VkPhysicalDeviceProperties2 props2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
    props2.pNext = &drmProps;
    vkGetPhysicalDeviceProperties2(physDev, &props2);
    if (!drmProps.hasRender) return 0;

    const std::string node = findDrmNode(drmProps.renderMajor, drmProps.renderMinor);
    if (node.empty()) return 0;
    return queryAsahiTotalCores(node);
}

#else

uint32_t queryGpuTotalCores(VkPhysicalDevice) { return 0; }

#endif

} // namespace vkzg
