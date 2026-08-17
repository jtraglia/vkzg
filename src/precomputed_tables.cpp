// precomputed_tables.bin holds the FK20 position table, root-of-unity
// tables and phase-B kernel item lists derived from the mainnet trusted
// setup (see src/setup_data.cpp) -- fixed for the protocol's lifetime, so
// they're generated once rather than rebuilt on every prover creation.
#include "precomputed_tables.h"
#include "internal.h"
#include "install_paths.h"

#include <cstdio>
#include <cstdlib>

namespace vkzg {

namespace {

constexpr uint32_t kMagic = 0x475A4B56u; // 'VKZG'
constexpr uint32_t kVersion = 1u;

struct Header {
    uint32_t magic;
    uint32_t version;
    uint32_t items_plus_words;
    uint32_t items_minus_words;
};

bool read_exact(FILE *f, void *p, size_t n) { return fread(p, 1, n, f) == n; }

template <typename T>
bool read_vec(FILE *f, std::vector<T> &v, size_t count) {
    v.resize(count);
    return read_exact(f, v.data(), count * sizeof(T));
}

vkzg_result load_from(const char *path, PrecomputedTables &out) {
    FILE *f = fopen(path, "rb");
    if (!f) return VKZG_ERR_IO;

    Header h{};
    vkzg_result rc = VKZG_ERR_SETUP;
    if (!read_exact(f, &h, sizeof(h)) || h.magic != kMagic || h.version != kVersion) goto done;

    if (!read_vec(f, out.position_table, (size_t)kPositionTableWords) ||
        !read_vec(f, out.roots_fwd, (size_t)kFieldElementsPerExtBlob * kFrLimbs) ||
        !read_vec(f, out.roots_inv, (size_t)kFieldElementsPerExtBlob * kFrLimbs) ||
        !read_vec(f, out.kernel_items_plus, h.items_plus_words) ||
        !read_vec(f, out.kernel_items_minus, h.items_minus_words) ||
        !read_vec(f, out.kernel_offsets_plus, (size_t)kNumBuckets + 1) ||
        !read_vec(f, out.kernel_offsets_minus, (size_t)kNumBuckets + 1) ||
        !read_vec(f, out.kernel_perm_plus, (size_t)kNumBuckets) ||
        !read_vec(f, out.kernel_perm_minus, (size_t)kNumBuckets) ||
        !read_exact(f, out.inv_blob, sizeof(out.inv_blob))) {
        goto done;
    }
    rc = VKZG_OK;
done:
    fclose(f);
    return rc;
}

} // namespace

// Tries $VKZG_TABLES_PATH first (for callers that bundle the file
// themselves rather than relying on a build-tree or install-tree layout,
// e.g. the Java bindings), then the build-tree copy (so tests/bench/example
// work straight out of `cmake --build` with no install step), then the
// installed location.
vkzg_result load_precomputed_tables(PrecomputedTables &out) {
    if (const char *env = std::getenv("VKZG_TABLES_PATH")) {
        if (load_from(env, out) == VKZG_OK) return VKZG_OK;
    }
    if (load_from(VKZG_TABLES_PATH_BUILD, out) == VKZG_OK) return VKZG_OK;
    return load_from(VKZG_TABLES_PATH_INSTALL, out);
}

} // namespace vkzg
