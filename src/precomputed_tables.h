// Loads the mainnet FK20/position tables, precomputed once from the trusted
// setup and shipped as a binary file rather than derived at runtime.
#pragma once

#include "../include/vkzg.h"

#include <cstdint>
#include <vector>

namespace vkzg {

struct PrecomputedTables {
    std::vector<uint32_t> position_table;
    std::vector<uint32_t> roots_fwd, roots_inv;
    std::vector<uint32_t> kernel_items_plus, kernel_items_minus;
    std::vector<uint32_t> kernel_offsets_plus, kernel_offsets_minus;
    std::vector<uint32_t> kernel_perm_plus, kernel_perm_minus;
    uint32_t inv_blob[8];
};

vkzg_result load_precomputed_tables(PrecomputedTables &out);

} // namespace vkzg
