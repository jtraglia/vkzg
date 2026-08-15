/*
 * Minimal end-to-end example: build a prover and compute the 128 cell
 * proofs for one blob.  The mainnet trusted setup is compiled into the
 * library, so there is no file to load and no path to configure.
 *
 * Build:  cmake -B build && cmake --build build
 * Run:    ./build/example
 */
#include "vulkan_prover.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(void) {
    vkp_options opts;
    vkp_options_default(&opts);
    /* Deriving the FK20 tables from the setup takes about a second; caching
     * them brings subsequent starts down to ~60ms. */
    opts.table_cache_path = "/tmp/vkp_prover_tables.cache";
    opts.max_batch_size = 16; /* batch to keep the GPU busy; see the README */

    vkp_prover *prover = NULL;
    vkp_result rc = vkp_prover_new_default(&prover, &opts);
    if (rc != VKP_OK) {
        fprintf(stderr, "failed to create prover: %s\n", vkp_error_string(rc));
        return 1;
    }
    printf("prover ready on %s\n", vkp_prover_device_name(prover));

    /* A blob is 4096 big-endian field elements, each below the BLS12-381
     * scalar field modulus.  Zeroing the top byte guarantees that. */
    uint8_t *blob = malloc(VKP_BYTES_PER_BLOB);
    for (int i = 0; i < VKP_FIELD_ELEMENTS_PER_BLOB; i++) {
        uint8_t *fe = blob + (size_t)i * VKP_BYTES_PER_FIELD_ELEMENT;
        for (int j = 0; j < VKP_BYTES_PER_FIELD_ELEMENT; j++) {
            fe[j] = (uint8_t)((i * 31 + j * 7 + 11) & 0xff);
        }
        fe[0] = 0;
    }

    uint8_t *proofs = malloc((size_t)VKP_NUM_CELL_PROOFS * VKP_BYTES_PER_PROOF);

    rc = vkp_compute_proofs(prover, proofs, blob);
    if (rc != VKP_OK) {
        fprintf(stderr, "compute failed: %s\n", vkp_error_string(rc));
        return 1;
    }

    printf("produced %d cell proofs\n", VKP_NUM_CELL_PROOFS);
    printf("proof[0] = ");
    for (int i = 0; i < VKP_BYTES_PER_PROOF; i++) printf("%02x", proofs[i]);
    printf("\n");

    free(blob);
    free(proofs);
    vkp_prover_free(prover);
    return 0;
}
