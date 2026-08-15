/*
 * Minimal end-to-end example: build a prover and compute the cells and cell
 * proofs for one blob.  The mainnet trusted setup is compiled into the
 * library, so there is no file to load and no path to configure.
 *
 * Build:  cmake -B build && cmake --build build
 * Run:    ./build/mp_prover_example
 */
#include "metal_prover.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(void) {
    mp_options opts;
    mp_options_default(&opts);
    /* Deriving the FK20 tables from the setup takes about a second; caching
     * them brings subsequent starts down to ~60ms. */
    opts.table_cache_path = "/tmp/mp_prover_tables.cache";
    opts.max_batch_size = 16; /* batch to keep the GPU busy; see the README */

    mp_prover *prover = NULL;
    mp_result rc = mp_prover_new_default(&prover, &opts);
    if (rc != MP_OK) {
        fprintf(stderr, "failed to create prover: %s\n", mp_error_string(rc));
        return 1;
    }
    printf("prover ready on %s\n", mp_prover_device_name(prover));

    /* A blob is 4096 big-endian field elements, each below the BLS12-381
     * scalar field modulus.  Zeroing the top byte guarantees that. */
    uint8_t *blob = malloc(MP_BYTES_PER_BLOB);
    for (int i = 0; i < MP_FIELD_ELEMENTS_PER_BLOB; i++) {
        uint8_t *fe = blob + (size_t)i * MP_BYTES_PER_FIELD_ELEMENT;
        for (int j = 0; j < MP_BYTES_PER_FIELD_ELEMENT; j++) {
            fe[j] = (uint8_t)((i * 31 + j * 7 + 11) & 0xff);
        }
        fe[0] = 0;
    }

    uint8_t *cells = malloc((size_t)MP_CELLS_PER_EXT_BLOB * MP_BYTES_PER_CELL);
    uint8_t *proofs = malloc((size_t)MP_CELLS_PER_EXT_BLOB * MP_BYTES_PER_PROOF);

    rc = mp_compute_cells_and_proofs(prover, cells, proofs, blob);
    if (rc != MP_OK) {
        fprintf(stderr, "compute failed: %s\n", mp_error_string(rc));
        return 1;
    }

    printf("produced %d cells and %d proofs\n", MP_CELLS_PER_EXT_BLOB,
           MP_CELLS_PER_EXT_BLOB);
    printf("proof[0] = ");
    for (int i = 0; i < MP_BYTES_PER_PROOF; i++) printf("%02x", proofs[i]);
    printf("\n");

    free(blob);
    free(cells);
    free(proofs);
    mp_prover_free(prover);
    return 0;
}
