/*
 * Minimal end-to-end example: build a prover and compute the cells and cell
 * proofs for one blob.  The mainnet trusted setup is compiled into the
 * library, so there is no file to load and no path to configure.
 *
 * Build:  cmake -B build && cmake --build build
 * Run:    ./build/kzgpu_example
 */
#include "kzgpu.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(void) {
    kzgpu_options opts;
    kzgpu_options_default(&opts);
    /* Deriving the FK20 tables from the setup takes about a second; caching
     * them brings subsequent starts down to ~60ms. */
    opts.table_cache_path = "/tmp/kzgpu_tables.cache";
    opts.max_batch_size = 16; /* batch to keep the GPU busy; see the README */

    kzgpu_prover *prover = NULL;
    kzgpu_result rc = kzgpu_prover_new_default(&prover, &opts);
    if (rc != KZGPU_OK) {
        fprintf(stderr, "failed to create prover: %s\n", kzgpu_error_string(rc));
        return 1;
    }
    printf("prover ready on %s\n", kzgpu_device_name(prover));

    /* A blob is 4096 big-endian field elements, each below the BLS12-381
     * scalar field modulus.  Zeroing the top byte guarantees that. */
    uint8_t *blob = malloc(KZGPU_BYTES_PER_BLOB);
    for (int i = 0; i < KZGPU_FIELD_ELEMENTS_PER_BLOB; i++) {
        uint8_t *fe = blob + (size_t)i * KZGPU_BYTES_PER_FIELD_ELEMENT;
        for (int j = 0; j < KZGPU_BYTES_PER_FIELD_ELEMENT; j++) {
            fe[j] = (uint8_t)((i * 31 + j * 7 + 11) & 0xff);
        }
        fe[0] = 0;
    }

    uint8_t *cells = malloc((size_t)KZGPU_CELLS_PER_EXT_BLOB * KZGPU_BYTES_PER_CELL);
    uint8_t *proofs = malloc((size_t)KZGPU_CELLS_PER_EXT_BLOB * KZGPU_BYTES_PER_PROOF);

    rc = kzgpu_compute_cells_and_proofs(prover, cells, proofs, blob);
    if (rc != KZGPU_OK) {
        fprintf(stderr, "compute failed: %s\n", kzgpu_error_string(rc));
        return 1;
    }

    printf("produced %d cells and %d proofs\n", KZGPU_CELLS_PER_EXT_BLOB,
           KZGPU_CELLS_PER_EXT_BLOB);
    printf("proof[0] = ");
    for (int i = 0; i < KZGPU_BYTES_PER_PROOF; i++) printf("%02x", proofs[i]);
    printf("\n");

    free(blob);
    free(cells);
    free(proofs);
    kzgpu_prover_free(prover);
    return 0;
}
