// Vulkan host layer and public API implementation.
//
// Buffers are passed to kernels as raw GPU addresses (VK_KHR_buffer_device_
// address) via push constants rather than descriptor sets. Every buffer is
// HOST_VISIBLE | HOST_COHERENT and persistently mapped.
#include "vkzg.h"
#include "gpu_topology.h"
#include "internal.h"
#include "precomputed_tables.h"
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

#include <fcntl.h>
#include <unistd.h>

using namespace vkzg;

// include/vkzg.h's public size macros and src/layout_defs.h's internal ones
// are independently maintained but must describe the same protocol; catch
// any future drift between them at compile time.
static_assert(VKZG_FIELD_ELEMENTS_PER_BLOB == kFieldElementsPerBlob, "public/internal size mismatch");
static_assert(VKZG_BYTES_PER_BLOB == L_BYTES_PER_BLOB, "public/internal size mismatch");
static_assert(VKZG_BYTES_PER_PROOF == kBytesPerProof, "public/internal size mismatch");

namespace {

double nowMs() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1e3 + (double)ts.tv_nsec / 1e6;
}

// Mesa logs harmless "Permission denied" errors to stderr while probing
// DRM nodes during device enumeration; there's no Vulkan API to quiet a
// driver's own logging, so suppress fd 2 around just that call.
struct ScopedStderrSuppress {
    int saved = -1;
    ScopedStderrSuppress() {
        fflush(stderr);
        const int devnull = open("/dev/null", O_WRONLY);
        if (devnull < 0) return;
        saved = dup(STDERR_FILENO);
        if (saved >= 0) dup2(devnull, STDERR_FILENO);
        close(devnull);
    }
    ~ScopedStderrSuppress() {
        if (saved < 0) return;
        fflush(stderr);
        dup2(saved, STDERR_FILENO);
        close(saved);
    }
};

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

struct vkzg_prover {
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
    VkPipeline psoLadder = VK_NULL_HANDLE;
    VkPipeline psoFoldLadder = VK_NULL_HANDLE;
    VkPipeline psoPhaseBSplit = VK_NULL_HANDLE;
    VkPipeline psoCombineSplit = VK_NULL_HANDLE;
    VkPipeline psoNormalize = VK_NULL_HANDLE;
    VkPipeline psoCompress = VK_NULL_HANDLE;
    VkPipeline psoReduceThroughput = VK_NULL_HANDLE; // L_REDUCE_LANES lanes, see layout_defs.h
    VkPipeline psoReduceLatency = VK_NULL_HANDLE;    // 2x L_REDUCE_LANES lanes

    // Cell recovery.
    VkPipeline psoEwMul = VK_NULL_HANDLE;
    VkPipeline psoEwDiv = VK_NULL_HANDLE;
    VkPipeline psoPowSeq = VK_NULL_HANDLE;
    VkPipeline psoVanishingPoly = VK_NULL_HANDLE;
    VkPipeline psoCellsToFr = VK_NULL_HANDLE;
    VkPipeline psoFrToCells = VK_NULL_HANDLE;

    // Setup-derived, immutable.
    VkBuf bufTable, bufRootsFwd, bufRootsInv;
    VkBuf bufKernelItemsPlus, bufKernelOffsetsPlus, bufKernelPermPlus;
    VkBuf bufKernelItemsMinus, bufKernelOffsetsMinus, bufKernelPermMinus;
    // Powers of the coset shift factor (7) and its inverse, populated once by
    // k_pow_seq.comp at prover creation -- shared by every recover call. Same
    // for bufInvExtBlob (a single Fr slot: 1/FIELD_ELEMENTS_PER_EXT_BLOB).
    VkBuf bufCosetShiftFwd, bufCosetShiftInv, bufInvExtBlob;

    // Per-batch working set.
    VkBuf bufBlob, bufLagrange, bufWorkA, bufPolyExt, bufCoeffs, bufBuckets,
        bufItems, bufStarts, bufPerm, bufPoints, bufProofs, bufLadderJac, bufLadderJacFolded,
        bufLadderAff, bufProofsAff, bufProofBytes, bufNormScratch, bufErr;

    // Cell recovery working set (see recoverBatch for the data-flow order).
    VkBuf bufRecMissing, bufRecScratch, bufRecBrp, bufRecZCoeff, bufRecZEval, bufRecEZEval,
        bufRecEZCoeff, bufRecShiftedA, bufRecCosetA, bufRecCosetB, bufRecPCoset, bufRecPCoeffU,
        bufRecPCoeff, bufRecPFull, bufRecOut;

    uint32_t maxBatch = 0;
    std::string deviceName;
    uint32_t gpuTotalCores = 0; // 0 = unknown topology; see gpu_topology.h
    std::mutex mutex;

    uint32_t invBlob[kFrLimbs] = {0};

    StageTimes lastStage; // filled by computeBatch, read by profile_batch

    ~vkzg_prover();
};

// --------------------------------------------------------------------- utils

const char *vkzg_error_string(vkzg_result r) {
    switch (r) {
        case VKZG_OK: return "ok";
        case VKZG_ERR_BADARGS: return "invalid argument";
        case VKZG_ERR_MALLOC: return "allocation failed";
        case VKZG_ERR_IO: return "i/o error";
        case VKZG_ERR_SETUP: return "malformed trusted setup";
        case VKZG_ERR_GPU: return "gpu error";
        case VKZG_ERR_INVALID_BLOB: return "blob contains a non-canonical field element";
    }
    return "unknown error";
}

void vkzg_options_default(vkzg_options *opts) {
    if (!opts) return;
    opts->max_batch_size = 0;
}

const char *vkzg_prover_device_name(const vkzg_prover *p) {
    return p ? p->deviceName.c_str() : "";
}

uint32_t vkzg_prover_gpu_core_count(const vkzg_prover *p) {
    return p ? p->gpuTotalCores : 0;
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

bool allocateWorkingSet(vkzg_prover *p, uint32_t batch) {
    const size_t B = batch;
    bool ok = true;
    ok &= createBuffer(p->device, p->physDev, B * VKZG_BYTES_PER_BLOB, p->bufBlob);
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
                       B * kCirculantSize * kLadderPositions * kJacobianWords * 4,
                       p->bufLadderJacFolded);
    ok &= createBuffer(p->device, p->physDev,
                       B * kCirculantSize * kLadderPositions * kAffineWords * 4, p->bufLadderAff);
    ok &= createBuffer(p->device, p->physDev, B * kCirculantSize * kAffineWords * 4, p->bufProofsAff);
    ok &= createBuffer(p->device, p->physDev, B * kCirculantSize * VKZG_BYTES_PER_PROOF, p->bufProofBytes);
    // Prefix products for the batched inversion; sized for the larger of the
    // two normalisation passes.
    ok &= createBuffer(p->device, p->physDev,
                       B * kCirculantSize * kLadderPositions * kFpLimbs * 4, p->bufNormScratch);
    ok &= createBuffer(p->device, p->physDev, 4, p->bufErr);

    // Cell recovery: one full-extended-domain Fr array per named intermediate
    // (see recoverBatch), plus a small per-blob missing-cell-index buffer.
    const size_t recBufBytes = B * kFieldElementsPerExtBlob * kFrLimbs * 4;
    ok &= createBuffer(p->device, p->physDev, B * 65 * 4, p->bufRecMissing);
    ok &= createBuffer(p->device, p->physDev, recBufBytes, p->bufRecScratch);
    ok &= createBuffer(p->device, p->physDev, recBufBytes, p->bufRecBrp);
    ok &= createBuffer(p->device, p->physDev, recBufBytes, p->bufRecZCoeff);
    ok &= createBuffer(p->device, p->physDev, recBufBytes, p->bufRecZEval);
    ok &= createBuffer(p->device, p->physDev, recBufBytes, p->bufRecEZEval);
    ok &= createBuffer(p->device, p->physDev, recBufBytes, p->bufRecEZCoeff);
    ok &= createBuffer(p->device, p->physDev, recBufBytes, p->bufRecShiftedA);
    ok &= createBuffer(p->device, p->physDev, recBufBytes, p->bufRecCosetA);
    ok &= createBuffer(p->device, p->physDev, recBufBytes, p->bufRecCosetB);
    ok &= createBuffer(p->device, p->physDev, recBufBytes, p->bufRecPCoset);
    ok &= createBuffer(p->device, p->physDev, recBufBytes, p->bufRecPCoeffU);
    ok &= createBuffer(p->device, p->physDev, recBufBytes, p->bufRecPCoeff);
    ok &= createBuffer(p->device, p->physDev, recBufBytes, p->bufRecPFull);
    ok &= createBuffer(p->device, p->physDev, recBufBytes, p->bufRecOut);
    if (!ok) return false;

    p->maxBatch = batch;
    return true;
}

void freeWorkingSet(vkzg_prover *p) {
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
    destroyBuffer(p->device, p->bufLadderJacFolded);
    destroyBuffer(p->device, p->bufLadderAff);
    destroyBuffer(p->device, p->bufProofsAff);
    destroyBuffer(p->device, p->bufProofBytes);
    destroyBuffer(p->device, p->bufNormScratch);
    destroyBuffer(p->device, p->bufErr);

    destroyBuffer(p->device, p->bufRecMissing);
    destroyBuffer(p->device, p->bufRecScratch);
    destroyBuffer(p->device, p->bufRecBrp);
    destroyBuffer(p->device, p->bufRecZCoeff);
    destroyBuffer(p->device, p->bufRecZEval);
    destroyBuffer(p->device, p->bufRecEZEval);
    destroyBuffer(p->device, p->bufRecEZCoeff);
    destroyBuffer(p->device, p->bufRecShiftedA);
    destroyBuffer(p->device, p->bufRecCosetA);
    destroyBuffer(p->device, p->bufRecCosetB);
    destroyBuffer(p->device, p->bufRecPCoset);
    destroyBuffer(p->device, p->bufRecPCoeffU);
    destroyBuffer(p->device, p->bufRecPCoeff);
    destroyBuffer(p->device, p->bufRecPFull);
    destroyBuffer(p->device, p->bufRecOut);
}

} // namespace

vkzg_prover::~vkzg_prover() {
    if (device == VK_NULL_HANDLE) return;
    vkDeviceWaitIdle(device);
    freeWorkingSet(this);
    destroyBuffer(device, bufTable);
    destroyBuffer(device, bufRootsFwd);
    destroyBuffer(device, bufRootsInv);
    destroyBuffer(device, bufKernelItemsPlus);
    destroyBuffer(device, bufKernelOffsetsPlus);
    destroyBuffer(device, bufKernelPermPlus);
    destroyBuffer(device, bufKernelItemsMinus);
    destroyBuffer(device, bufKernelOffsetsMinus);
    destroyBuffer(device, bufKernelPermMinus);
    destroyBuffer(device, bufCosetShiftFwd);
    destroyBuffer(device, bufCosetShiftInv);
    destroyBuffer(device, bufInvExtBlob);
    for (VkPipeline pso : {psoBlobToFr, psoNtt, psoBuildCirculant, psoPhaseASort, psoPhaseA,
                           psoLadder, psoFoldLadder, psoPhaseBSplit, psoCombineSplit, psoNormalize, psoCompress,
                           psoReduceThroughput, psoReduceLatency, psoEwMul, psoEwDiv,
                           psoPowSeq, psoVanishingPoly, psoCellsToFr, psoFrToCells}) {
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

// Required for every kernel's buffer_reference pointers, 8/16-bit storage
// buffers and 64-bit scalars.
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
    {
        ScopedStderrSuppress suppress;
        vkEnumeratePhysicalDevices(instance, &count, nullptr);
    }
    if (count == 0) return VK_NULL_HANDLE;
    std::vector<VkPhysicalDevice> devices(count);
    {
        ScopedStderrSuppress suppress;
        vkEnumeratePhysicalDevices(instance, &count, devices.data());
    }

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

vkzg_result buildPipelines(vkzg_prover *p) {
    struct {
        const char *name;
        VkPipeline *slot;
    } kernels[] = {
        {"k_blob_to_fr", &p->psoBlobToFr},
        {"k_ntt_pass", &p->psoNtt},
        {"k_build_circulant", &p->psoBuildCirculant},
        {"k_phase_a_sort", &p->psoPhaseASort},
        {"k_phase_a", &p->psoPhaseA},
        {"k_ladder", &p->psoLadder},
        {"k_fold_ladder", &p->psoFoldLadder},
        {"k_phase_b_split", &p->psoPhaseBSplit},
        {"k_combine_split", &p->psoCombineSplit},
        {"k_normalize", &p->psoNormalize},
        {"k_compress_proofs", &p->psoCompress},
        {"k_ew_mul", &p->psoEwMul},
        {"k_ew_div", &p->psoEwDiv},
        {"k_pow_seq", &p->psoPowSeq},
        {"k_vanishing_poly", &p->psoVanishingPoly},
        {"k_cells_to_fr", &p->psoCellsToFr},
        {"k_fr_to_cells", &p->psoFrToCells},
    };

    VkPushConstantRange pcRange{};
    pcRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pcRange.offset = 0;
    pcRange.size = 128; // covers every kernel's push-constant block (largest is 104 bytes)

    VkPipelineLayoutCreateInfo layoutInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    layoutInfo.pushConstantRangeCount = 1;
    layoutInfo.pPushConstantRanges = &pcRange;
    if (vkCreatePipelineLayout(p->device, &layoutInfo, nullptr, &p->pipelineLayout) != VK_SUCCESS) {
        return VKZG_ERR_GPU;
    }

    for (auto &k : kernels) {
        const ShaderSpv *spv = nullptr;
        for (size_t i = 0; i < kShaderCount; i++) {
            if (strcmp(kShaders[i].name, k.name) == 0) {
                spv = &kShaders[i];
                break;
            }
        }
        if (!spv) return VKZG_ERR_GPU;

        VkShaderModuleCreateInfo modInfo{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
        modInfo.codeSize = spv->words * 4;
        modInfo.pCode = spv->code;
        VkShaderModule module = VK_NULL_HANDLE;
        if (vkCreateShaderModule(p->device, &modInfo, nullptr, &module) != VK_SUCCESS) {
            return VKZG_ERR_GPU;
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
        if (rc != VK_SUCCESS) return VKZG_ERR_GPU;
    }

    // k_bucket_reduce is special: one SPIR-V module, two VkPipelines that
    // differ only in the L_REDUCE_LANES specialization constant (id 0) --
    // recordReduce picks between them per dispatch based on batch size and
    // queried GPU core count. See the comment above L_REDUCE_LANES in
    // layout_defs.h for why.
    {
        const ShaderSpv *spv = nullptr;
        for (size_t i = 0; i < kShaderCount; i++) {
            if (strcmp(kShaders[i].name, "k_bucket_reduce") == 0) {
                spv = &kShaders[i];
                break;
            }
        }
        if (!spv) return VKZG_ERR_GPU;

        VkShaderModuleCreateInfo modInfo{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
        modInfo.codeSize = spv->words * 4;
        modInfo.pCode = spv->code;
        VkShaderModule module = VK_NULL_HANDLE;
        if (vkCreateShaderModule(p->device, &modInfo, nullptr, &module) != VK_SUCCESS) {
            return VKZG_ERR_GPU;
        }

        const uint32_t lanesThroughput = L_REDUCE_LANES;
        const uint32_t lanesLatency = L_REDUCE_LANES * 2;
        VkSpecializationMapEntry mapEntry{0, 0, sizeof(uint32_t)};
        VkSpecializationInfo specThroughput{1, &mapEntry, sizeof(uint32_t), &lanesThroughput};
        VkSpecializationInfo specLatency{1, &mapEntry, sizeof(uint32_t), &lanesLatency};

        VkPipelineShaderStageCreateInfo stageThroughput{
            VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
        stageThroughput.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        stageThroughput.module = module;
        stageThroughput.pName = "main";
        stageThroughput.pSpecializationInfo = &specThroughput;

        VkPipelineShaderStageCreateInfo stageLatency = stageThroughput;
        stageLatency.pSpecializationInfo = &specLatency;

        VkComputePipelineCreateInfo pipeInfos[2]{};
        pipeInfos[0] = {VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        pipeInfos[0].stage = stageThroughput;
        pipeInfos[0].layout = p->pipelineLayout;
        pipeInfos[1] = {VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        pipeInfos[1].stage = stageLatency;
        pipeInfos[1].layout = p->pipelineLayout;

        VkPipeline pipelines[2] = {VK_NULL_HANDLE, VK_NULL_HANDLE};
        const VkResult rc = vkCreateComputePipelines(p->device, VK_NULL_HANDLE, 2, pipeInfos,
                                                      nullptr, pipelines);
        vkDestroyShaderModule(p->device, module, nullptr);
        if (rc != VK_SUCCESS) return VKZG_ERR_GPU;
        p->psoReduceThroughput = pipelines[0];
        p->psoReduceLatency = pipelines[1];
    }
    return VKZG_OK;
}

bool populateCosetShiftTables(vkzg_prover *p);

vkzg_result createProver(vkzg_prover **out, PrecomputedTables &tables, const vkzg_options *opts) {
    auto *p = new vkzg_prover();

    VkApplicationInfo appInfo{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    appInfo.pApplicationName = "vkzg";
    appInfo.apiVersion = VK_API_VERSION_1_2;
    VkInstanceCreateInfo instInfo{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    instInfo.pApplicationInfo = &appInfo;
    if (vkCreateInstance(&instInfo, nullptr, &p->instance) != VK_SUCCESS) {
        delete p;
        return VKZG_ERR_GPU;
    }

    p->physDev = pickPhysicalDevice(p->instance, p->queueFamily);
    if (p->physDev == VK_NULL_HANDLE) {
        delete p;
        return VKZG_ERR_GPU;
    }
    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(p->physDev, &props);
    p->deviceName = props.deviceName;
    p->gpuTotalCores = queryGpuTotalCores(p->physDev);

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
        return VKZG_ERR_GPU;
    }
    vkGetDeviceQueue(p->device, p->queueFamily, 0, &p->queue);

    VkCommandPoolCreateInfo poolInfo{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    poolInfo.queueFamilyIndex = p->queueFamily;
    if (vkCreateCommandPool(p->device, &poolInfo, nullptr, &p->cmdPool) != VK_SUCCESS) {
        delete p;
        return VKZG_ERR_GPU;
    }
    VkCommandBufferAllocateInfo cmdAlloc{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cmdAlloc.commandPool = p->cmdPool;
    cmdAlloc.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cmdAlloc.commandBufferCount = 1;
    if (vkAllocateCommandBuffers(p->device, &cmdAlloc, &p->cmdBuf) != VK_SUCCESS) {
        delete p;
        return VKZG_ERR_GPU;
    }

    VkFenceCreateInfo fenceInfo{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    if (vkCreateFence(p->device, &fenceInfo, nullptr, &p->fence) != VK_SUCCESS) {
        delete p;
        return VKZG_ERR_GPU;
    }

    vkzg_result rc = buildPipelines(p);
    if (rc != VKZG_OK) {
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
    ok &= createBufferFrom(p->device, p->physDev, tables.kernel_items_plus.data(),
                           tables.kernel_items_plus.size() * 4, p->bufKernelItemsPlus);
    ok &= createBufferFrom(p->device, p->physDev, tables.kernel_offsets_plus.data(),
                           tables.kernel_offsets_plus.size() * 4, p->bufKernelOffsetsPlus);
    ok &= createBufferFrom(p->device, p->physDev, tables.kernel_perm_plus.data(),
                           tables.kernel_perm_plus.size() * 4, p->bufKernelPermPlus);
    ok &= createBufferFrom(p->device, p->physDev, tables.kernel_items_minus.data(),
                           tables.kernel_items_minus.size() * 4, p->bufKernelItemsMinus);
    ok &= createBufferFrom(p->device, p->physDev, tables.kernel_offsets_minus.data(),
                           tables.kernel_offsets_minus.size() * 4, p->bufKernelOffsetsMinus);
    ok &= createBufferFrom(p->device, p->physDev, tables.kernel_perm_minus.data(),
                           tables.kernel_perm_minus.size() * 4, p->bufKernelPermMinus);
    ok &= createBuffer(p->device, p->physDev, (size_t)kFieldElementsPerExtBlob * kFrLimbs * 4,
                       p->bufCosetShiftFwd);
    ok &= createBuffer(p->device, p->physDev, (size_t)kFieldElementsPerExtBlob * kFrLimbs * 4,
                       p->bufCosetShiftInv);
    ok &= createBuffer(p->device, p->physDev, (size_t)kFrLimbs * 4, p->bufInvExtBlob);
    if (!ok) {
        delete p;
        return VKZG_ERR_MALLOC;
    }
    memcpy(p->invBlob, tables.inv_blob, sizeof(p->invBlob));

    uint32_t batch = opts && opts->max_batch_size ? opts->max_batch_size : 4;
    if (!allocateWorkingSet(p, batch)) {
        delete p;
        return VKZG_ERR_MALLOC;
    }

    // One-time setup: populate the coset shift-power tables cell recovery
    // needs (see k_pow_seq.comp) -- shared by every blob and every recover
    // call, so this never needs to run again.
    if (!populateCosetShiftTables(p)) {
        delete p;
        return VKZG_ERR_GPU;
    }

    *out = p;
    return VKZG_OK;
}

} // namespace

// --------------------------------------------------------------- constructors

vkzg_result vkzg_prover_new(vkzg_prover **out, const vkzg_options *opts) {
    if (!out) return VKZG_ERR_BADARGS;
    *out = nullptr;

    PrecomputedTables tables;
    vkzg_result rc = load_precomputed_tables(tables);
    if (rc != VKZG_OK) return rc;

    return createProver(out, tables, opts);
}

void vkzg_prover_free(vkzg_prover *p) { delete p; }

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

// See the forward declaration in createProver: populates bufCosetShiftFwd/Inv
// via k_pow_seq.comp, once, at prover creation.
bool populateCosetShiftTables(vkzg_prover *p) {
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(p->cmdBuf, &bi);
    struct { uint64_t outAddr, invExtBlobAddr; uint32_t mode; } pcFwd{
        p->bufCosetShiftFwd.addr, p->bufInvExtBlob.addr, 0u};
    dispatch(p->cmdBuf, p->psoPowSeq, p->pipelineLayout, pcFwd, 1, 1);
    struct { uint64_t outAddr, invExtBlobAddr; uint32_t mode; } pcInv{
        p->bufCosetShiftInv.addr, p->bufInvExtBlob.addr, 1u};
    dispatch(p->cmdBuf, p->psoPowSeq, p->pipelineLayout, pcInv, 1, 1);
    vkEndCommandBuffer(p->cmdBuf);
    vkResetFences(p->device, 1, &p->fence);
    VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &p->cmdBuf;
    if (vkQueueSubmit(p->queue, 1, &submit, p->fence) != VK_SUCCESS) return false;
    return vkWaitForFences(p->device, 1, &p->fence, VK_TRUE, UINT64_MAX) == VK_SUCCESS;
}

void recordNtt(VkCommandBuffer cmd, vkzg_prover *p, const VkBuf &out, const VkBuf &in,
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

// Picks the reduce dispatch's lane count (see layout_defs.h) from batch size
// and GPU core count. The threshold is fit to measurements on two GPUs, not
// derived -- revisit if it disagrees with a differently-shaped GPU.
bool useLatencyReduce(uint32_t count, uint32_t batch, uint32_t gpuTotalCores) {
    if (gpuTotalCores == 0) return false; // unknown topology: keep the safe default
    constexpr uint32_t kWorkgroupsPerCoreThreshold = 6;
    const uint64_t workgroupsAtDefault =
        (uint64_t)L_REDUCE_LANES * count / 32 * batch;
    return workgroupsAtDefault < (uint64_t)kWorkgroupsPerCoreThreshold * gpuTotalCores;
}

void recordReduce(VkCommandBuffer cmd, vkzg_prover *p, const VkBuf &out, uint32_t count,
                  uint32_t batch) {
    const bool latency = useLatencyReduce(count, batch, p->gpuTotalCores);
    const uint32_t lanes = latency ? (L_REDUCE_LANES * 2) : L_REDUCE_LANES;
    const uint32_t outputsPerTg = 32 / lanes;
    ReducePC pc{out.addr, p->bufBuckets.addr, count};
    const uint32_t groups = (count + outputsPerTg - 1) / outputsPerTg;
    dispatch(cmd, latency ? p->psoReduceLatency : p->psoReduceThroughput, p->pipelineLayout, pc,
             groups, batch);
}

struct NormalizePC {
    uint64_t outAffineAddr, inJacobianAddr, scratchAddr;
    uint32_t count;
    uint32_t chunk;
};

void recordNormalize(VkCommandBuffer cmd, vkzg_prover *p, const VkBuf &outAffine,
                     const VkBuf &inJacobian, uint32_t count, uint32_t chunk) {
    NormalizePC pc{outAffine.addr, inJacobian.addr, p->bufNormScratch.addr, count, chunk};
    const uint32_t threads = (count + chunk - 1) / chunk;
    const uint32_t groups = (threads + 63) / 64;
    dispatch(cmd, p->psoNormalize, p->pipelineLayout, pc, groups, 1);
}

// Ends, submits and waits on `cmd`, then reopens it for further recording.
// Used only when profiling (real submission overhead per call) -- never on
// the production path.
double flushAndTime(vkzg_prover *p, VkCommandBuffer &cmd, double &prev) {
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

vkzg_result computeBatch(vkzg_prover *p, uint8_t *proofs, const uint8_t *blobs, uint32_t batch,
                        bool profile) {
    StageTimes &st = p->lastStage;
    st = StageTimes{};
    const double tStart = nowMs();
    memcpy(p->bufBlob.mapped, blobs, (size_t)batch * VKZG_BYTES_PER_BLOB);
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

    // 5. doubling ladder over u[j], then fold into L+/L- (see layout_defs.h).
    {
        struct { uint64_t ladderAddr, uAddr; } pc{p->bufLadderJac.addr, p->bufPoints.addr};
        dispatch(cmd, p->psoLadder, p->pipelineLayout, pc, kCirculantSize / 32, batch);
    }
    flush(&st.ladder);

    {
        struct { uint64_t outAddr, ladderAddr; } pc{p->bufLadderJacFolded.addr, p->bufLadderJac.addr};
        const uint32_t groups = (kCirculantHalf * kLadderPositions) / 128;
        dispatch(cmd, p->psoFoldLadder, p->pipelineLayout, pc, groups, batch);
    }
    flush(&st.fold_ladder);

    const uint32_t ladderPoints = (uint32_t)batch * kCirculantSize * kLadderPositions;
    recordNormalize(cmd, p, p->bufLadderAff, p->bufLadderJacFolded, ladderPoints,
                    inversionChunk(ladderPoints, 32, 128, 8192));
    flush(&st.normalize_ladder);

    // 6. phase B, split form (see layout_defs.h). bufPoints is reused as
    // scratch for the reduced C+/C- pair; nothing else needs it by now.
    {
        struct {
            uint64_t outAddr, ladderAddr;
            uint64_t itemsPlusAddr, offsetsPlusAddr, permPlusAddr;
            uint64_t itemsMinusAddr, offsetsMinusAddr, permMinusAddr;
        } pc{p->bufBuckets.addr,
             p->bufLadderAff.addr,
             p->bufKernelItemsPlus.addr,
             p->bufKernelOffsetsPlus.addr,
             p->bufKernelPermPlus.addr,
             p->bufKernelItemsMinus.addr,
             p->bufKernelOffsetsMinus.addr,
             p->bufKernelPermMinus.addr};
        dispatch(cmd, p->psoPhaseBSplit, p->pipelineLayout, pc, kCirculantSize, batch);
    }
    flush(&st.phase_b);
    recordReduce(cmd, p, p->bufPoints, kCirculantSize, batch);
    flush(&st.reduce_b);

    // 6b. recombine: out[a] = C+[a]+C-[a], out[a+64] = C+[a]-C-[a].
    {
        struct { uint64_t outAddr, reducedAddr; } pc{p->bufProofs.addr, p->bufPoints.addr};
        dispatch(cmd, p->psoCombineSplit, p->pipelineLayout, pc, 1, batch);
    }
    flush(&st.combine);

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
    if (vkQueueSubmit(p->queue, 1, &submit, p->fence) != VK_SUCCESS) return VKZG_ERR_GPU;
    if (vkWaitForFences(p->device, 1, &p->fence, VK_TRUE, UINT64_MAX) != VK_SUCCESS) {
        return VKZG_ERR_GPU;
    }

    if (*(uint32_t *)p->bufErr.mapped != 0) return VKZG_ERR_INVALID_BLOB;

    memcpy(proofs, p->bufProofBytes.mapped, (size_t)batch * kCirculantSize * VKZG_BYTES_PER_PROOF);
    st.total = nowMs() - tStart;
    return VKZG_OK;
}

// -------------------------------------------------------------- cell recovery

// 7-bit reversal of a cell index (0..127), matching c-kzg-4844's
// reverse_bits_limited(CELLS_PER_EXT_BLOB, i): the missing-cell roots are
// indexed into the natural-order root table by the *reversed* wire index.
uint32_t brevCell(uint32_t i) {
    uint32_t r = 0;
    for (int b = 0; b < 7; b++) r |= ((i >> b) & 1u) << (6 - b);
    return r;
}

// One full unscaled 8192-point transform: two k_ntt_pass dispatches (four-
// step, N1=128 x N2=64), via bufRecScratch as the pass-1/pass-2 handoff.
// `roots` selects direction (bufRootsFwd = forward, bufRootsInv = inverse,
// unscaled -- callers needing the 1/N inverse-transform scale apply it
// afterwards with a k_ew_mul against bufInvExtBlob).
void recoverNtt(VkCommandBuffer cmd, vkzg_prover *p, const VkBuf &out, const VkBuf &in,
               const VkBuf &roots, uint32_t batch) {
    NttParams q = nttPass1(kFieldElementsPerExtBlob, 128, 64, kFieldElementsPerExtBlob,
                           kFieldElementsPerExtBlob);
    recordNtt(cmd, p, p->bufRecScratch, in, roots, q, 64, batch);
    NttParams r = nttPass2(kFieldElementsPerExtBlob, 128, 64, kFieldElementsPerExtBlob,
                           kFieldElementsPerExtBlob);
    recordNtt(cmd, p, out, p->bufRecScratch, roots, r, 128, batch);
}

// Mirrors k_ew_mul.comp's B_MODE_* constants.
enum class EwMulBMode : uint32_t { kBatched = 0, kSharedTable = 1, kBroadcast = 2 };

void recoverEwMul(VkCommandBuffer cmd, vkzg_prover *p, const VkBuf &out, const VkBuf &a,
                  const VkBuf &b, EwMulBMode bMode, uint32_t batch) {
    struct { uint64_t outAddr, aAddr, bAddr; uint32_t bMode; } pc{
        out.addr, a.addr, b.addr, (uint32_t)bMode};
    dispatch(cmd, p->psoEwMul, p->pipelineLayout, pc,
            kFieldElementsPerExtBlob / 256, batch);
}

void recoverEwDiv(VkCommandBuffer cmd, vkzg_prover *p, const VkBuf &out, const VkBuf &a,
                  const VkBuf &b, uint32_t batch) {
    struct { uint64_t outAddr, aAddr, bAddr; } pc{out.addr, a.addr, b.addr};
    dispatch(cmd, p->psoEwDiv, p->pipelineLayout, pc, kFieldElementsPerExtBlob / 256, batch);
}

const size_t kCellBytes = (size_t)kFieldElementsPerCell * VKZG_BYTES_PER_FIELD_ELEMENT;

vkzg_result recoverBatch(vkzg_prover *p, uint8_t *cells_out, const uint8_t *cells,
                         const uint8_t *cell_present, uint32_t batch) {
    const double tStart = nowMs();
    memcpy(p->bufRecScratch.mapped, cells, (size_t)batch * kCirculantSize * kCellBytes);

    // Host-side: which cells are missing per blob, bit-reversed per the
    // reference algorithm (see vanishing_polynomial_for_missing_cells).
    std::vector<uint32_t> missing((size_t)batch * 65, 0);
    for (uint32_t b = 0; b < batch; b++) {
        uint32_t count = 0;
        for (uint32_t i = 0; i < (uint32_t)kCirculantSize; i++) {
            if (!cell_present[(size_t)b * kCirculantSize + i]) {
                if (count >= (uint32_t)kFieldElementsPerCell) return VKZG_ERR_BADARGS;
                missing[(size_t)b * 65 + 1 + count] = brevCell(i);
                count++;
            }
        }
        missing[(size_t)b * 65] = count;
    }
    memcpy(p->bufRecMissing.mapped, missing.data(), missing.size() * 4);
    *(uint32_t *)p->bufErr.mapped = 0;

    vkResetCommandBuffer(p->cmdBuf, 0);
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(p->cmdBuf, &bi);
    VkCommandBuffer cmd = p->cmdBuf;

    // 1. cell bytes (wire order) -> Montgomery Fr (natural FFT domain order).
    {
        struct { uint64_t outAddr, cellsAddr; } pc{p->bufRecBrp.addr, p->bufRecScratch.addr};
        dispatch(cmd, p->psoCellsToFr, p->pipelineLayout, pc, kFieldElementsPerExtBlob / 256, batch);
    }

    // 2. Z(x), monomial form, vanishing on every missing cell.
    {
        struct { uint64_t outAddr, rootsAddr, missingAddr; } pc{
            p->bufRecZCoeff.addr, p->bufRootsFwd.addr, p->bufRecMissing.addr};
        dispatch(cmd, p->psoVanishingPoly, p->pipelineLayout, pc, 1, batch);
    }

    // 3. Z(x) -> evaluation form.
    recoverNtt(cmd, p, p->bufRecZEval, p->bufRecZCoeff, p->bufRootsFwd, batch);

    // 4. (E*Z)(x) in evaluation form.
    recoverEwMul(cmd, p, p->bufRecEZEval, p->bufRecBrp, p->bufRecZEval, EwMulBMode::kBatched, batch);

    // 5. (E*Z)(x) -> monomial form (unscaled transform, then the 1/N scale).
    recoverNtt(cmd, p, p->bufRecEZCoeff, p->bufRecEZEval, p->bufRootsInv, batch);
    recoverEwMul(cmd, p, p->bufRecEZCoeff, p->bufRecEZCoeff, p->bufInvExtBlob,
                EwMulBMode::kBroadcast, batch);

    // 6. (E*Z)(x) and Z(x) -> evaluation form over a coset (avoids dividing
    // by zero: Z is zero only on the original domain's missing points).
    recoverEwMul(cmd, p, p->bufRecShiftedA, p->bufRecEZCoeff, p->bufCosetShiftFwd,
                EwMulBMode::kSharedTable, batch);
    recoverNtt(cmd, p, p->bufRecCosetA, p->bufRecShiftedA, p->bufRootsFwd, batch);
    recoverEwMul(cmd, p, p->bufRecShiftedA, p->bufRecZCoeff, p->bufCosetShiftFwd,
                EwMulBMode::kSharedTable, batch);
    recoverNtt(cmd, p, p->bufRecCosetB, p->bufRecShiftedA, p->bufRootsFwd, batch);

    // 7. P(x) = (E*Z)(x) / Z(x), in evaluation form over the coset.
    recoverEwDiv(cmd, p, p->bufRecPCoset, p->bufRecCosetA, p->bufRecCosetB, batch);

    // 8. P(x) -> monomial form (coset_ifft: unscaled ifft, 1/N scale, un-shift).
    recoverNtt(cmd, p, p->bufRecPCoeffU, p->bufRecPCoset, p->bufRootsInv, batch);
    recoverEwMul(cmd, p, p->bufRecPCoeffU, p->bufRecPCoeffU, p->bufInvExtBlob,
                EwMulBMode::kBroadcast, batch);
    recoverEwMul(cmd, p, p->bufRecPCoeff, p->bufRecPCoeffU, p->bufCosetShiftInv,
                EwMulBMode::kSharedTable, batch);

    // 9. Evaluate P(x) at every point of the extended domain.
    recoverNtt(cmd, p, p->bufRecPFull, p->bufRecPCoeff, p->bufRootsFwd, batch);

    // 10. Montgomery Fr (natural FFT domain order) -> cell bytes (wire order).
    {
        struct { uint64_t outAddr, inAddr; } pc{p->bufRecOut.addr, p->bufRecPFull.addr};
        dispatch(cmd, p->psoFrToCells, p->pipelineLayout, pc, kFieldElementsPerExtBlob / 256, batch);
    }

    vkEndCommandBuffer(cmd);
    vkResetFences(p->device, 1, &p->fence);
    VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &cmd;
    if (vkQueueSubmit(p->queue, 1, &submit, p->fence) != VK_SUCCESS) return VKZG_ERR_GPU;
    if (vkWaitForFences(p->device, 1, &p->fence, VK_TRUE, UINT64_MAX) != VK_SUCCESS) {
        return VKZG_ERR_GPU;
    }

    memcpy(cells_out, p->bufRecOut.mapped, (size_t)batch * kCirculantSize * kCellBytes);
    (void)tStart;
    return VKZG_OK;
}

} // namespace

vkzg_result vkzg_compute_proofs_batch(vkzg_prover *p, uint8_t *proofs, const uint8_t *blobs,
                                       size_t num_blobs) {
    if (!p || !blobs || !proofs) return VKZG_ERR_BADARGS;
    if (num_blobs == 0) return VKZG_OK;

    std::lock_guard<std::mutex> lock(p->mutex);
    const size_t proofBytes = (size_t)kCirculantSize * VKZG_BYTES_PER_PROOF;

    for (size_t done = 0; done < num_blobs;) {
        const uint32_t batch = (uint32_t)std::min<size_t>(p->maxBatch, num_blobs - done);
        vkzg_result rc = computeBatch(p, proofs + done * proofBytes, blobs + done * VKZG_BYTES_PER_BLOB,
                                       batch, /*profile=*/false);
        if (rc != VKZG_OK) return rc;
        done += batch;
    }
    return VKZG_OK;
}

vkzg_result vkzg_recover_cells_batch(vkzg_prover *p, uint8_t *cells_out, const uint8_t *cells,
                                      const uint8_t *cell_present, size_t num_blobs) {
    if (!p || !cells_out || !cells || !cell_present) return VKZG_ERR_BADARGS;
    if (num_blobs == 0) return VKZG_OK;

    std::lock_guard<std::mutex> lock(p->mutex);
    const size_t cellsBytes = (size_t)kCirculantSize * kCellBytes;
    const size_t presentBytes = (size_t)kCirculantSize;

    for (size_t done = 0; done < num_blobs;) {
        const uint32_t batch = (uint32_t)std::min<size_t>(p->maxBatch, num_blobs - done);
        vkzg_result rc = recoverBatch(p, cells_out + done * cellsBytes, cells + done * cellsBytes,
                                      cell_present + done * presentBytes, batch);
        if (rc != VKZG_OK) return rc;
        done += batch;
    }
    return VKZG_OK;
}

// ------------------------------------------------------------------ profiling

namespace vkzg {

vkzg_result profile_batch(vkzg_prover *p, unsigned char *proofs, const unsigned char *blobs,
                           unsigned batch, StageTimes &out) {
    if (!p || !blobs) return VKZG_ERR_BADARGS;
    std::lock_guard<std::mutex> lock(p->mutex);
    const vkzg_result rc = computeBatch(p, proofs, blobs, batch, /*profile=*/true);
    out = p->lastStage;
    return rc;
}

} // namespace vkzg
