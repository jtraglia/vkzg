// Vulkan host layer and public API implementation.
//
// Buffers are passed to kernels as raw GPU addresses (VK_KHR_buffer_device_
// address) via push constants rather than descriptor sets: every buffer is
// allocated HOST_VISIBLE | HOST_COHERENT (the direct analogue of Metal's
// MTLResourceStorageModeShared, and the natural choice on Apple Silicon's
// unified memory) and persistently mapped, and its VkDeviceAddress is fetched
// once at allocation time. This keeps the dispatch code below close in shape
// to the Metal original, which passed buffers the same way.
#include "vulkan_prover.h"
#include "cpu/bls12_381.h"
#include "cpu/setup.h"
#include "internal.h"
#include "setup_data.h"
#include "profile.h"
#include "shaders/shader_source.h"

#include <vulkan/vulkan.h>

#include <algorithm>
#include <cstring>
#include <ctime>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

using namespace vkp;

namespace {

double nowMs() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1e3 + (double)ts.tv_nsec / 1e6;
}

uint32_t ilog2(uint32_t n) {
    uint32_t r = 0;
    while ((1u << r) < n) r++;
    return r;
}

// Mirrors the push-constant block in k_ntt_pass.comp.
struct NttParams {
    uint64_t outAddr, inAddr, rootsAddr;
    uint32_t n;
    uint32_t log_n;
    uint32_t in_stride_t;
    uint32_t in_stride_i;
    uint32_t out_stride_t;
    uint32_t out_stride_i;
    uint32_t root_stride;
    uint32_t twiddle_stride;
    uint32_t full_n;
    uint32_t in_batch;
    uint32_t out_batch;
    uint32_t scale;
    uint32_t scale_val[kFrLimbs];
};

} // namespace

// --------------------------------------------------------------- GPU buffer

namespace {

struct VkBuf {
    VkBuffer buf = VK_NULL_HANDLE;
    VkDeviceMemory mem = VK_NULL_HANDLE;
    void *mapped = nullptr;
    VkDeviceAddress addr = 0;
    size_t size = 0;
};

} // namespace

struct vkp_prover {
    VkInstance instance = VK_NULL_HANDLE;
    VkPhysicalDevice physDev = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    VkQueue queue = VK_NULL_HANDLE;
    uint32_t queueFamily = 0;
    VkCommandPool cmdPool = VK_NULL_HANDLE;
    VkCommandBuffer cmdBuf = VK_NULL_HANDLE;
    VkFence fence = VK_NULL_HANDLE;

    VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
    VkPipeline psoBlobToFr = VK_NULL_HANDLE;
    VkPipeline psoNtt = VK_NULL_HANDLE;
    VkPipeline psoBuildCirculant = VK_NULL_HANDLE;
    VkPipeline psoPhaseASort = VK_NULL_HANDLE;
    VkPipeline psoPhaseA = VK_NULL_HANDLE;
    VkPipeline psoPhaseB = VK_NULL_HANDLE;
    VkPipeline psoLadder = VK_NULL_HANDLE;
    VkPipeline psoNormalize = VK_NULL_HANDLE;
    VkPipeline psoCompress = VK_NULL_HANDLE;
    VkPipeline psoReduce = VK_NULL_HANDLE;

    // Setup-derived, immutable.
    VkBuf bufTable, bufRootsFwd, bufRootsInv, bufKernelItems, bufKernelOffsets, bufKernelPerm;

    // Per-batch working set.
    VkBuf bufBlob, bufLagrange, bufWorkA, bufPolyExt, bufCoeffs, bufBuckets,
        bufItems, bufStarts, bufPerm, bufPoints, bufProofs, bufLadderJac, bufLadderAff,
        bufProofsAff, bufProofBytes, bufNormScratch, bufErr;

    uint32_t maxBatch = 0;
    std::string deviceName;
    std::mutex mutex;

    uint32_t invBlob[kFrLimbs] = {0};

    StageTimes lastStage; // filled by computeBatch, read by profile_batch

    ~vkp_prover();
};

// --------------------------------------------------------------------- utils

const char *vkp_error_string(vkp_result r) {
    switch (r) {
        case VKP_OK: return "ok";
        case VKP_ERR_BADARGS: return "invalid argument";
        case VKP_ERR_MALLOC: return "allocation failed";
        case VKP_ERR_IO: return "i/o error";
        case VKP_ERR_SETUP: return "malformed trusted setup";
        case VKP_ERR_GPU: return "gpu error";
        case VKP_ERR_INVALID_BLOB: return "blob contains a non-canonical field element";
    }
    return "unknown error";
}

void vkp_options_default(vkp_options *opts) {
    if (!opts) return;
    opts->table_cache_path = nullptr;
    opts->validate_setup = 0;
    opts->max_batch_size = 0;
}

const char *vkp_prover_device_name(const vkp_prover *p) {
    return p ? p->deviceName.c_str() : "";
}

// ---------------------------------------------------------------- buffers

namespace {

uint32_t findMemoryType(VkPhysicalDevice physDev, uint32_t typeBits, VkMemoryPropertyFlags want) {
    VkPhysicalDeviceMemoryProperties props;
    vkGetPhysicalDeviceMemoryProperties(physDev, &props);
    for (uint32_t i = 0; i < props.memoryTypeCount; i++) {
        if ((typeBits & (1u << i)) && (props.memoryTypes[i].propertyFlags & want) == want) {
            return i;
        }
    }
    return UINT32_MAX;
}

bool createBuffer(VkDevice device, VkPhysicalDevice physDev, size_t bytes, VkBuf &out) {
    if (bytes == 0) bytes = 16;
    VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bi.size = bytes;
    bi.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
    bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if (vkCreateBuffer(device, &bi, nullptr, &out.buf) != VK_SUCCESS) return false;

    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(device, out.buf, &req);
    const uint32_t typeIdx = findMemoryType(
        physDev, req.memoryTypeBits,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    if (typeIdx == UINT32_MAX) return false;

    VkMemoryAllocateFlagsInfo flagsInfo{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO};
    flagsInfo.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
    VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    ai.pNext = &flagsInfo;
    ai.allocationSize = req.size;
    ai.memoryTypeIndex = typeIdx;
    if (vkAllocateMemory(device, &ai, nullptr, &out.mem) != VK_SUCCESS) return false;
    if (vkBindBufferMemory(device, out.buf, out.mem, 0) != VK_SUCCESS) return false;
    if (vkMapMemory(device, out.mem, 0, bytes, 0, &out.mapped) != VK_SUCCESS) return false;

    VkBufferDeviceAddressInfo addrInfo{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO};
    addrInfo.buffer = out.buf;
    out.addr = vkGetBufferDeviceAddress(device, &addrInfo);
    out.size = bytes;
    return true;
}

bool createBufferFrom(VkDevice device, VkPhysicalDevice physDev, const void *src, size_t bytes,
                       VkBuf &out) {
    if (!createBuffer(device, physDev, bytes, out)) return false;
    if (bytes && src) memcpy(out.mapped, src, bytes);
    return true;
}

void destroyBuffer(VkDevice device, VkBuf &b) {
    if (b.mapped) vkUnmapMemory(device, b.mem);
    if (b.buf) vkDestroyBuffer(device, b.buf, nullptr);
    if (b.mem) vkFreeMemory(device, b.mem, nullptr);
    b = VkBuf{};
}

bool allocateWorkingSet(vkp_prover *p, uint32_t batch) {
    const size_t B = batch;
    bool ok = true;
    ok &= createBuffer(p->device, p->physDev, B * VKP_BYTES_PER_BLOB, p->bufBlob);
    ok &= createBuffer(p->device, p->physDev, B * kFieldElementsPerBlob * kFrLimbs * 4, p->bufLagrange);
    ok &= createBuffer(p->device, p->physDev, B * kFieldElementsPerExtBlob * kFrLimbs * 4, p->bufWorkA);
    ok &= createBuffer(p->device, p->physDev, B * kFieldElementsPerExtBlob * kFrLimbs * 4, p->bufPolyExt);
    ok &= createBuffer(p->device, p->physDev, B * kCirculantSize * kPhaseATerms * kFrLimbs * 4, p->bufCoeffs);
    ok &= createBuffer(p->device, p->physDev, B * kCirculantSize * kNumBuckets * kJacobianWords * 4, p->bufBuckets);
    ok &= createBuffer(p->device, p->physDev, B * kCirculantSize * kPhaseAItems * 2, p->bufItems);
    ok &= createBuffer(p->device, p->physDev, B * kCirculantSize * (kNumBuckets + 1) * 4, p->bufStarts);
    ok &= createBuffer(p->device, p->physDev, B * kCirculantSize * kNumBuckets * 4, p->bufPerm);
    ok &= createBuffer(p->device, p->physDev, B * kCirculantSize * kJacobianWords * 4, p->bufPoints);
    ok &= createBuffer(p->device, p->physDev, B * kCirculantSize * kJacobianWords * 4, p->bufProofs);
    ok &= createBuffer(p->device, p->physDev,
                       B * kCirculantSize * kLadderPositions * kJacobianWords * 4, p->bufLadderJac);
    ok &= createBuffer(p->device, p->physDev,
                       B * kCirculantSize * kLadderPositions * kAffineWords * 4, p->bufLadderAff);
    ok &= createBuffer(p->device, p->physDev, B * kCirculantSize * kAffineWords * 4, p->bufProofsAff);
    ok &= createBuffer(p->device, p->physDev, B * kCirculantSize * VKP_BYTES_PER_PROOF, p->bufProofBytes);
    // Prefix products for the batched inversion; sized for the larger of the
    // two normalisation passes.
    ok &= createBuffer(p->device, p->physDev,
                       B * kCirculantSize * kLadderPositions * kFpLimbs * 4, p->bufNormScratch);
    ok &= createBuffer(p->device, p->physDev, 4, p->bufErr);
    if (!ok) return false;

    p->maxBatch = batch;
    return true;
}

void freeWorkingSet(vkp_prover *p) {
    destroyBuffer(p->device, p->bufBlob);
    destroyBuffer(p->device, p->bufLagrange);
    destroyBuffer(p->device, p->bufWorkA);
    destroyBuffer(p->device, p->bufPolyExt);
    destroyBuffer(p->device, p->bufCoeffs);
    destroyBuffer(p->device, p->bufBuckets);
    destroyBuffer(p->device, p->bufItems);
    destroyBuffer(p->device, p->bufStarts);
    destroyBuffer(p->device, p->bufPerm);
    destroyBuffer(p->device, p->bufPoints);
    destroyBuffer(p->device, p->bufProofs);
    destroyBuffer(p->device, p->bufLadderJac);
    destroyBuffer(p->device, p->bufLadderAff);
    destroyBuffer(p->device, p->bufProofsAff);
    destroyBuffer(p->device, p->bufProofBytes);
    destroyBuffer(p->device, p->bufNormScratch);
    destroyBuffer(p->device, p->bufErr);
}

} // namespace

vkp_prover::~vkp_prover() {
    if (device == VK_NULL_HANDLE) return;
    vkDeviceWaitIdle(device);
    freeWorkingSet(this);
    destroyBuffer(device, bufTable);
    destroyBuffer(device, bufRootsFwd);
    destroyBuffer(device, bufRootsInv);
    destroyBuffer(device, bufKernelItems);
    destroyBuffer(device, bufKernelOffsets);
    destroyBuffer(device, bufKernelPerm);
    for (VkPipeline pso : {psoBlobToFr, psoNtt, psoBuildCirculant, psoPhaseASort,
                           psoPhaseA, psoPhaseB, psoLadder, psoNormalize, psoCompress, psoReduce}) {
        if (pso) vkDestroyPipeline(device, pso, nullptr);
    }
    if (pipelineLayout) vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
    if (fence) vkDestroyFence(device, fence, nullptr);
    if (cmdPool) vkDestroyCommandPool(device, cmdPool, nullptr);
    vkDestroyDevice(device, nullptr);
    if (instance) vkDestroyInstance(instance, nullptr);
}

// ------------------------------------------------------------ device setup

namespace {

// Required so every kernel's buffer_reference pointers, 8/16-bit storage
// buffers and 64-bit scalars work; verified against this project's actual
// target (Mesa's Honeykrisp Vulkan driver on Apple Silicon, Vulkan 1.4
// conformant) but written against the portable Vulkan 1.2 core feature bits.
bool physicalDeviceSuitable(VkPhysicalDevice dev, uint32_t &queueFamilyOut) {
    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(dev, &props);
    if (props.apiVersion < VK_API_VERSION_1_2) return false;

    VkPhysicalDeviceVulkan12Features f12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
    VkPhysicalDeviceVulkan11Features f11{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES};
    f11.pNext = &f12;
    VkPhysicalDeviceFeatures2 f2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
    f2.pNext = &f11;
    vkGetPhysicalDeviceFeatures2(dev, &f2);

    if (!f2.features.shaderInt64 || !f2.features.shaderInt16) return false;
    if (!f11.storageBuffer16BitAccess) return false;
    if (!f12.bufferDeviceAddress || !f12.storageBuffer8BitAccess || !f12.shaderInt8) return false;

    VkPhysicalDeviceSubgroupProperties sub{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES};
    VkPhysicalDeviceProperties2 p2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
    p2.pNext = &sub;
    vkGetPhysicalDeviceProperties2(dev, &p2);
    const auto need = VK_SUBGROUP_FEATURE_SHUFFLE_BIT | VK_SUBGROUP_FEATURE_SHUFFLE_RELATIVE_BIT;
    if ((sub.supportedOperations & need) != need) return false;
    if (!(sub.supportedStages & VK_SHADER_STAGE_COMPUTE_BIT)) return false;

    uint32_t count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(dev, &count, nullptr);
    std::vector<VkQueueFamilyProperties> families(count);
    vkGetPhysicalDeviceQueueFamilyProperties(dev, &count, families.data());
    for (uint32_t i = 0; i < count; i++) {
        if (families[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
            queueFamilyOut = i;
            return true;
        }
    }
    return false;
}

int deviceTypeScore(VkPhysicalDeviceType t) {
    switch (t) {
        case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU: return 3;
        case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU: return 2;
        case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU: return 1;
        default: return 0;
    }
}

VkPhysicalDevice pickPhysicalDevice(VkInstance instance, uint32_t &queueFamilyOut) {
    uint32_t count = 0;
    vkEnumeratePhysicalDevices(instance, &count, nullptr);
    if (count == 0) return VK_NULL_HANDLE;
    std::vector<VkPhysicalDevice> devices(count);
    vkEnumeratePhysicalDevices(instance, &count, devices.data());

    VkPhysicalDevice best = VK_NULL_HANDLE;
    int bestScore = -1;
    uint32_t bestQueueFamily = 0;
    for (VkPhysicalDevice dev : devices) {
        uint32_t qf = 0;
        if (!physicalDeviceSuitable(dev, qf)) continue;
        VkPhysicalDeviceProperties props;
        vkGetPhysicalDeviceProperties(dev, &props);
        const int score = deviceTypeScore(props.deviceType);
        if (score > bestScore) {
            bestScore = score;
            best = dev;
            bestQueueFamily = qf;
        }
    }
    queueFamilyOut = bestQueueFamily;
    return best;
}

vkp_result buildPipelines(vkp_prover *p) {
    struct {
        const char *name;
        VkPipeline *slot;
    } kernels[] = {
        {"k_blob_to_fr", &p->psoBlobToFr},
        {"k_ntt_pass", &p->psoNtt},
        {"k_build_circulant", &p->psoBuildCirculant},
        {"k_phase_a_sort", &p->psoPhaseASort},
        {"k_phase_a", &p->psoPhaseA},
        {"k_phase_b", &p->psoPhaseB},
        {"k_ladder", &p->psoLadder},
        {"k_normalize", &p->psoNormalize},
        {"k_compress_proofs", &p->psoCompress},
        {"k_bucket_reduce", &p->psoReduce},
    };

    VkPushConstantRange pcRange{};
    pcRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pcRange.offset = 0;
    pcRange.size = 128; // covers every kernel's push-constant block (largest is 104 bytes)

    VkPipelineLayoutCreateInfo layoutInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    layoutInfo.pushConstantRangeCount = 1;
    layoutInfo.pPushConstantRanges = &pcRange;
    if (vkCreatePipelineLayout(p->device, &layoutInfo, nullptr, &p->pipelineLayout) != VK_SUCCESS) {
        return VKP_ERR_GPU;
    }

    for (auto &k : kernels) {
        const ShaderSpv *spv = nullptr;
        for (size_t i = 0; i < kShaderCount; i++) {
            if (strcmp(kShaders[i].name, k.name) == 0) {
                spv = &kShaders[i];
                break;
            }
        }
        if (!spv) return VKP_ERR_GPU;

        VkShaderModuleCreateInfo modInfo{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
        modInfo.codeSize = spv->words * 4;
        modInfo.pCode = spv->code;
        VkShaderModule module = VK_NULL_HANDLE;
        if (vkCreateShaderModule(p->device, &modInfo, nullptr, &module) != VK_SUCCESS) {
            return VKP_ERR_GPU;
        }

        VkPipelineShaderStageCreateInfo stage{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
        stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        stage.module = module;
        stage.pName = "main";

        VkComputePipelineCreateInfo pipeInfo{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        pipeInfo.stage = stage;
        pipeInfo.layout = p->pipelineLayout;
        const VkResult rc =
            vkCreateComputePipelines(p->device, VK_NULL_HANDLE, 1, &pipeInfo, nullptr, k.slot);
        vkDestroyShaderModule(p->device, module, nullptr);
        if (rc != VK_SUCCESS) return VKP_ERR_GPU;
    }
    return VKP_OK;
}

vkp_result createProver(vkp_prover **out, SetupTables &tables, const vkp_options *opts) {
    auto *p = new vkp_prover();

    VkApplicationInfo appInfo{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    appInfo.pApplicationName = "vulkan-prover";
    appInfo.apiVersion = VK_API_VERSION_1_2;
    VkInstanceCreateInfo instInfo{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    instInfo.pApplicationInfo = &appInfo;
    if (vkCreateInstance(&instInfo, nullptr, &p->instance) != VK_SUCCESS) {
        delete p;
        return VKP_ERR_GPU;
    }

    p->physDev = pickPhysicalDevice(p->instance, p->queueFamily);
    if (p->physDev == VK_NULL_HANDLE) {
        delete p;
        return VKP_ERR_GPU;
    }
    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(p->physDev, &props);
    p->deviceName = props.deviceName;

    const float queuePriority = 1.0f;
    VkDeviceQueueCreateInfo queueInfo{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    queueInfo.queueFamilyIndex = p->queueFamily;
    queueInfo.queueCount = 1;
    queueInfo.pQueuePriorities = &queuePriority;

    VkPhysicalDeviceVulkan12Features f12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
    f12.bufferDeviceAddress = VK_TRUE;
    f12.storageBuffer8BitAccess = VK_TRUE;
    f12.shaderInt8 = VK_TRUE;
    VkPhysicalDeviceVulkan11Features f11{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES};
    f11.storageBuffer16BitAccess = VK_TRUE;
    f11.pNext = &f12;
    VkPhysicalDeviceFeatures2 f2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
    f2.pNext = &f11;
    f2.features.shaderInt64 = VK_TRUE;
    f2.features.shaderInt16 = VK_TRUE;

    VkDeviceCreateInfo devInfo{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    devInfo.pNext = &f2;
    devInfo.queueCreateInfoCount = 1;
    devInfo.pQueueCreateInfos = &queueInfo;
    if (vkCreateDevice(p->physDev, &devInfo, nullptr, &p->device) != VK_SUCCESS) {
        delete p;
        return VKP_ERR_GPU;
    }
    vkGetDeviceQueue(p->device, p->queueFamily, 0, &p->queue);

    VkCommandPoolCreateInfo poolInfo{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    poolInfo.queueFamilyIndex = p->queueFamily;
    if (vkCreateCommandPool(p->device, &poolInfo, nullptr, &p->cmdPool) != VK_SUCCESS) {
        delete p;
        return VKP_ERR_GPU;
    }
    VkCommandBufferAllocateInfo cmdAlloc{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cmdAlloc.commandPool = p->cmdPool;
    cmdAlloc.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cmdAlloc.commandBufferCount = 1;
    if (vkAllocateCommandBuffers(p->device, &cmdAlloc, &p->cmdBuf) != VK_SUCCESS) {
        delete p;
        return VKP_ERR_GPU;
    }

    VkFenceCreateInfo fenceInfo{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    if (vkCreateFence(p->device, &fenceInfo, nullptr, &p->fence) != VK_SUCCESS) {
        delete p;
        return VKP_ERR_GPU;
    }

    vkp_result rc = buildPipelines(p);
    if (rc != VKP_OK) {
        delete p;
        return rc;
    }

    bool ok = true;
    ok &= createBufferFrom(p->device, p->physDev, tables.position_table.data(),
                           tables.position_table.size() * 4, p->bufTable);
    ok &= createBufferFrom(p->device, p->physDev, tables.roots_fwd.data(), tables.roots_fwd.size() * 4,
                           p->bufRootsFwd);
    ok &= createBufferFrom(p->device, p->physDev, tables.roots_inv.data(), tables.roots_inv.size() * 4,
                           p->bufRootsInv);
    ok &= createBufferFrom(p->device, p->physDev, tables.kernel_items.data(),
                           tables.kernel_items.size() * 4, p->bufKernelItems);
    ok &= createBufferFrom(p->device, p->physDev, tables.kernel_offsets.data(),
                           tables.kernel_offsets.size() * 4, p->bufKernelOffsets);
    ok &= createBufferFrom(p->device, p->physDev, tables.kernel_perm.data(),
                           tables.kernel_perm.size() * 4, p->bufKernelPerm);
    if (!ok) {
        delete p;
        return VKP_ERR_MALLOC;
    }
    memcpy(p->invBlob, tables.inv_blob, sizeof(p->invBlob));

    uint32_t batch = opts && opts->max_batch_size ? opts->max_batch_size : 4;
    if (!allocateWorkingSet(p, batch)) {
        delete p;
        return VKP_ERR_MALLOC;
    }
    *out = p;
    return VKP_OK;
}

} // namespace

// --------------------------------------------------------------- constructors

vkp_result vkp_prover_new(vkp_prover **out, const uint8_t *g1_monomial_bytes,
                              size_t g1_monomial_len, const vkp_options *opts) {
    if (!out || !g1_monomial_bytes) return VKP_ERR_BADARGS;
    *out = nullptr;

    SetupTables tables;
    const bool validate = opts && opts->validate_setup;
    const char *cache = opts ? opts->table_cache_path : nullptr;

    // The cache records the digest of the setup it was derived from; hashing
    // the input is far cheaper than building the tables just to learn it.
    bool have = false;
    if (cache) {
        const uint64_t digest = compute_setup_digest(g1_monomial_bytes, g1_monomial_len);
        have = load_table_cache(cache, digest, tables) == VKP_OK;
    }
    if (!have) {
        vkp_result rc = build_setup_tables(g1_monomial_bytes, g1_monomial_len, validate, tables);
        if (rc != VKP_OK) return rc;
        if (cache) save_table_cache(cache, tables);
    }

    return createProver(out, tables, opts);
}

vkp_result vkp_prover_new_default(vkp_prover **out, const vkp_options *opts) {
    return vkp_prover_new(out, kEmbeddedSetupG1Monomial, kEmbeddedSetupSize, opts);
}

void vkp_prover_free(vkp_prover *p) { delete p; }

// -------------------------------------------------------------------- compute

namespace {

void barrier(VkCommandBuffer cmd) {
    VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    mb.srcAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                        0, 1, &mb, 0, nullptr, 0, nullptr);
}

template <typename PC>
void dispatch(VkCommandBuffer cmd, VkPipeline pso, VkPipelineLayout layout, const PC &pc,
             uint32_t groupsX, uint32_t groupsY, uint32_t groupsZ = 1) {
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pso);
    vkCmdPushConstants(cmd, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PC), &pc);
    vkCmdDispatch(cmd, groupsX, groupsY, groupsZ);
    barrier(cmd);
}

void recordNtt(VkCommandBuffer cmd, vkp_prover *p, const VkBuf &out, const VkBuf &in,
              const VkBuf &roots, const NttParams &params, uint32_t count, uint32_t batch) {
    NttParams pc = params;
    pc.outAddr = out.addr;
    pc.inAddr = in.addr;
    pc.rootsAddr = roots.addr;
    dispatch(cmd, p->psoNtt, p->pipelineLayout, pc, count, batch);
}

// Configures one pass of an N = N1 * N2 four-step transform.
NttParams nttPass1(uint32_t N, uint32_t N1, uint32_t N2, uint32_t inBatch, uint32_t outBatch) {
    NttParams q{};
    q.n = N1;
    q.log_n = ilog2(N1);
    q.in_stride_t = 1;
    q.in_stride_i = N2;
    q.out_stride_t = 1;
    q.out_stride_i = N2;
    q.root_stride = kFieldElementsPerExtBlob / N1;
    q.twiddle_stride = kFieldElementsPerExtBlob / N;
    q.full_n = N;
    q.in_batch = inBatch;
    q.out_batch = outBatch;
    q.scale = 0;
    return q;
}

NttParams nttPass2(uint32_t N, uint32_t N1, uint32_t N2, uint32_t inBatch, uint32_t outBatch) {
    NttParams q{};
    q.n = N2;
    q.log_n = ilog2(N2);
    q.in_stride_t = N2;
    q.in_stride_i = 1;
    q.out_stride_t = 1;
    q.out_stride_i = N1;
    q.root_stride = kFieldElementsPerExtBlob / N2;
    q.twiddle_stride = 0;
    q.full_n = N;
    q.in_batch = inBatch;
    q.out_batch = outBatch;
    q.scale = 0;
    return q;
}

// Points per thread in the batched inversion.
//
// Each chunk pays one field inversion, which is ~570 multiplies deep and so
// costs a fixed latency no matter how few points it covers. Bigger chunks
// amortise that better; smaller chunks give more threads. Rather than pin a
// constant tuned to one GPU, size the chunk so the dispatch keeps a few
// thousand threads busy, with a floor that keeps the inversion below ~18
// multiplies per point.
uint32_t inversionChunk(uint32_t count, uint32_t floorChunk, uint32_t ceilChunk,
                        uint32_t targetThreads) {
    uint32_t chunk = count / targetThreads;
    if (chunk < floorChunk) chunk = floorChunk;
    if (chunk > ceilChunk) chunk = ceilChunk;
    return chunk;
}

struct ReducePC {
    uint64_t outAddr, bucketsAddr;
    uint32_t count;
};

void recordReduce(VkCommandBuffer cmd, vkp_prover *p, const VkBuf &out, uint32_t count,
                  uint32_t batch) {
    ReducePC pc{out.addr, p->bufBuckets.addr, count};
    const uint32_t groups = (count + L_REDUCE_OUTPUTS_PER_TG - 1) / L_REDUCE_OUTPUTS_PER_TG;
    dispatch(cmd, p->psoReduce, p->pipelineLayout, pc, groups, batch);
}

struct NormalizePC {
    uint64_t outAffineAddr, inJacobianAddr, scratchAddr;
    uint32_t count;
    uint32_t chunk;
};

void recordNormalize(VkCommandBuffer cmd, vkp_prover *p, const VkBuf &outAffine,
                     const VkBuf &inJacobian, uint32_t count, uint32_t chunk) {
    NormalizePC pc{outAffine.addr, inJacobian.addr, p->bufNormScratch.addr, count, chunk};
    const uint32_t threads = (count + chunk - 1) / chunk;
    const uint32_t groups = (threads + 63) / 64;
    dispatch(cmd, p->psoNormalize, p->pipelineLayout, pc, groups, 1);
}

// Ends, submits and waits on `cmd`, then reopens it for further recording.
// Used only when profiling: it turns one command buffer into N synchronous
// round-trips, so it costs real submission overhead (~0.1ms/flush) and must
// never run on the production path.
//
// This replaced an earlier attempt at profiling via vkCmdWriteTimestamp +
// VkQueryPool, which looked reasonable but produced numbers that didn't add
// up: the "normalize ladder" bucket consistently read ~1.4s regardless of
// how much (or how little) work that dispatch actually had to do, while
// varying wildly between otherwise-identical runs which *other* bucket
// absorbed the real cost. Forcing a CPU-side drain between every dispatch
// (this function) and comparing gave a stable, reproducible breakdown where
// the two MSM phases dominate, as expected — so the timestamp path was
// mismeasuring on this driver, not the kernels misbehaving. Given that, this
// slower-but-trustworthy approach is what `profile_stages` uses.
double flushAndTime(vkp_prover *p, VkCommandBuffer &cmd, double &prev) {
    vkEndCommandBuffer(cmd);
    vkResetFences(p->device, 1, &p->fence);
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cmd;
    vkQueueSubmit(p->queue, 1, &si, p->fence);
    vkWaitForFences(p->device, 1, &p->fence, VK_TRUE, UINT64_MAX);
    const double now = nowMs();
    const double delta = now - prev;
    prev = now;

    vkResetCommandBuffer(cmd, 0);
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd, &bi);
    return delta;
}

vkp_result computeBatch(vkp_prover *p, uint8_t *proofs, const uint8_t *blobs, uint32_t batch,
                        bool profile) {
    StageTimes &st = p->lastStage;
    st = StageTimes{};
    const double tStart = nowMs();
    memcpy(p->bufBlob.mapped, blobs, (size_t)batch * VKP_BYTES_PER_BLOB);
    *(uint32_t *)p->bufErr.mapped = 0;

    vkResetCommandBuffer(p->cmdBuf, 0);
    VkCommandBufferBeginInfo beginInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(p->cmdBuf, &beginInfo);
    VkCommandBuffer cmd = p->cmdBuf;

    double prev = nowMs();
    auto flush = [&](double *slot) {
        if (!profile) return;
        const double d = flushAndTime(p, cmd, prev);
        if (slot) *slot = d;
    };

    // 1. blob bytes -> bit-reversed Lagrange values
    {
        struct { uint64_t outAddr, blobAddr, errAddr; } pc{p->bufLagrange.addr, p->bufBlob.addr,
                                                            p->bufErr.addr};
        dispatch(cmd, p->psoBlobToFr, p->pipelineLayout, pc, kFieldElementsPerBlob / 256, batch);
    }

    // 2. inverse transform of size 4096 -> monomial coefficients
    {
        NttParams q = nttPass1(4096, 64, 64, kFieldElementsPerBlob, kFieldElementsPerExtBlob);
        recordNtt(cmd, p, p->bufWorkA, p->bufLagrange, p->bufRootsInv, q, 64, batch);
        NttParams r = nttPass2(4096, 64, 64, kFieldElementsPerExtBlob, kFieldElementsPerExtBlob);
        r.scale = 1;
        memcpy(r.scale_val, p->invBlob, sizeof(r.scale_val));
        recordNtt(cmd, p, p->bufPolyExt, p->bufWorkA, p->bufRootsInv, r, 64, batch);
    }
    flush(&st.scalar_stage);

    // 3. circulant columns and their size-128 transforms
    {
        struct { uint64_t coeffsAddr, polyAddr, rootsAddr; } pc{
            p->bufCoeffs.addr, p->bufPolyExt.addr, p->bufRootsFwd.addr};
        dispatch(cmd, p->psoBuildCirculant, p->pipelineLayout, pc, kPhaseATerms, batch);
    }

    // 4a. phase A scalar pass: recode, histogram, load-order, sort
    {
        struct { uint64_t itemsAddr, startsAddr, permAddr, coeffsAddr; } pc{
            p->bufItems.addr, p->bufStarts.addr, p->bufPerm.addr, p->bufCoeffs.addr};
        dispatch(cmd, p->psoPhaseASort, p->pipelineLayout, pc, kCirculantSize, batch);
    }

    // 4b. phase A curve pass: fixed-base bucket MSM
    {
        struct { uint64_t bucketsAddr, itemsAddr, startsAddr, permAddr, tableAddr; } pc{
            p->bufBuckets.addr, p->bufItems.addr, p->bufStarts.addr, p->bufPerm.addr,
            p->bufTable.addr};
        dispatch(cmd, p->psoPhaseA, p->pipelineLayout, pc, kCirculantSize, batch);
    }
    flush(&st.phase_a);

    recordReduce(cmd, p, p->bufPoints, kCirculantSize, batch);
    flush(&st.reduce_a);

    // 5. doubling ladder over u[j], then to affine for the mixed adds
    {
        struct { uint64_t ladderAddr, uAddr; } pc{p->bufLadderJac.addr, p->bufPoints.addr};
        dispatch(cmd, p->psoLadder, p->pipelineLayout, pc, kCirculantSize / 128, batch);
    }
    flush(&st.ladder);

    const uint32_t ladderPoints = (uint32_t)batch * kCirculantSize * kLadderPositions;
    recordNormalize(cmd, p, p->bufLadderAff, p->bufLadderJac, ladderPoints,
                    inversionChunk(ladderPoints, 32, 128, 8192));
    flush(&st.normalize_ladder);

    // 6. phase B: the fused circulant map
    {
        struct { uint64_t outAddr, ladderAddr, itemsAddr, offsetsAddr, permAddr; } pc{
            p->bufBuckets.addr, p->bufLadderAff.addr, p->bufKernelItems.addr,
            p->bufKernelOffsets.addr, p->bufKernelPerm.addr};
        dispatch(cmd, p->psoPhaseB, p->pipelineLayout, pc, kCirculantSize, batch);
    }
    flush(&st.phase_b);

    recordReduce(cmd, p, p->bufProofs, kCirculantSize, batch);
    flush(&st.reduce_b);

    // 7. proofs -> affine -> compressed bytes
    const uint32_t proofPoints = (uint32_t)batch * kCirculantSize;
    recordNormalize(cmd, p, p->bufProofsAff, p->bufProofs, proofPoints,
                    inversionChunk(proofPoints, 4, 32, 2048));
    flush(&st.normalize_proofs);

    {
        struct { uint64_t outAddr, affineAddr; } pc{p->bufProofBytes.addr, p->bufProofsAff.addr};
        dispatch(cmd, p->psoCompress, p->pipelineLayout, pc, kCirculantSize / 64, batch);
    }
    flush(&st.compress);

    vkEndCommandBuffer(cmd);

    vkResetFences(p->device, 1, &p->fence);
    VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &cmd;
    if (vkQueueSubmit(p->queue, 1, &submit, p->fence) != VK_SUCCESS) return VKP_ERR_GPU;
    if (vkWaitForFences(p->device, 1, &p->fence, VK_TRUE, UINT64_MAX) != VK_SUCCESS) {
        return VKP_ERR_GPU;
    }

    if (*(uint32_t *)p->bufErr.mapped != 0) return VKP_ERR_INVALID_BLOB;

    memcpy(proofs, p->bufProofBytes.mapped, (size_t)batch * kCirculantSize * VKP_BYTES_PER_PROOF);
    st.total = nowMs() - tStart;
    return VKP_OK;
}

} // namespace

vkp_result vkp_compute_proofs(vkp_prover *p, uint8_t *proofs, const uint8_t *blob) {
    return vkp_compute_proofs_batch(p, proofs, blob, 1);
}

vkp_result vkp_compute_proofs_batch(vkp_prover *p, uint8_t *proofs, const uint8_t *blobs,
                                        size_t num_blobs) {
    if (!p || !blobs || !proofs) return VKP_ERR_BADARGS;
    if (num_blobs == 0) return VKP_OK;

    std::lock_guard<std::mutex> lock(p->mutex);
    const size_t proofBytes = (size_t)kCirculantSize * VKP_BYTES_PER_PROOF;

    for (size_t done = 0; done < num_blobs;) {
        const uint32_t batch = (uint32_t)std::min<size_t>(p->maxBatch, num_blobs - done);
        vkp_result rc = computeBatch(p, proofs + done * proofBytes, blobs + done * VKP_BYTES_PER_BLOB,
                                       batch, /*profile=*/false);
        if (rc != VKP_OK) return rc;
        done += batch;
    }
    return VKP_OK;
}

// ------------------------------------------------------------------ profiling
//
// Development helper: computeBatch records what it measured into the prover,
// and this just runs it and hands the numbers back.  Reporting the real path
// matters here rather than a separate re-implementation of the dispatch order.
#include "profile.h"

namespace vkp {

vkp_result profile_batch(vkp_prover *p, unsigned char *proofs, const unsigned char *blobs,
                           unsigned batch, StageTimes &out) {
    if (!p || !blobs) return VKP_ERR_BADARGS;
    std::lock_guard<std::mutex> lock(p->mutex);
    const vkp_result rc = computeBatch(p, proofs, blobs, batch, /*profile=*/true);
    out = p->lastStage;
    return rc;
}

} // namespace vkp
