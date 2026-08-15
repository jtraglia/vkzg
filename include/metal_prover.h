/*
 * metal-prover -- EIP-7594 cell KZG proof generation on Apple GPUs.
 *
 * The whole pipeline runs on the GPU: one command buffer per call, with the
 * host doing nothing but copying blobs in and cells/proofs out.  That is
 * deliberate -- on a headless node the GPU is idle while the CPU has real work
 * to do, so spending CPU cycles here would be taking them from something else.
 *
 * The API is intentionally plain C so that Rust, Go, Java (JNI/Panama) and
 * others can bind to it without a C++ shim.  All functions are thread safe
 * unless stated otherwise; a single `mp_prover` may be shared between
 * threads, and concurrent calls are serialised internally per GPU queue.
 *
 * This library only *produces* cells and proofs.  Verification is deliberately
 * out of scope.
 */
#ifndef METAL_PROVER_H
#define METAL_PROVER_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ------------------------------------------------------------------ sizes */

#define MP_FIELD_ELEMENTS_PER_BLOB 4096
#define MP_FIELD_ELEMENTS_PER_EXT_BLOB 8192
#define MP_FIELD_ELEMENTS_PER_CELL 64
#define MP_CELLS_PER_EXT_BLOB 128
#define MP_BYTES_PER_FIELD_ELEMENT 32
#define MP_BYTES_PER_BLOB (MP_FIELD_ELEMENTS_PER_BLOB * MP_BYTES_PER_FIELD_ELEMENT)
#define MP_BYTES_PER_CELL (MP_FIELD_ELEMENTS_PER_CELL * MP_BYTES_PER_FIELD_ELEMENT)
#define MP_BYTES_PER_PROOF 48
/* Number of G1 points in the monomial-form trusted setup we consume. */
#define MP_NUM_SETUP_G1_POINTS MP_FIELD_ELEMENTS_PER_BLOB
#define MP_BYTES_PER_G1 48

/* ----------------------------------------------------------------- status */

typedef enum {
    MP_OK = 0,
    MP_ERR_BADARGS = 1,     /* caller passed a null/invalid argument */
    MP_ERR_MALLOC = 2,      /* host allocation failed */
    MP_ERR_IO = 3,          /* trusted setup or cache file could not be read */
    MP_ERR_SETUP = 4,       /* trusted setup was malformed or off-curve */
    MP_ERR_GPU = 5,         /* no Metal device, or a shader failed to build */
    MP_ERR_INVALID_BLOB = 6 /* a field element in the blob was not canonical */
} mp_result;

/* Human readable form of a status code. Never returns NULL. */
const char *mp_error_string(mp_result r);

/* ---------------------------------------------------------------- options */

typedef struct {
    /*
     * Path used to cache the derived FK20 tables.  Building them takes on the
     * order of a second; loading the cache takes milliseconds.  If NULL, no
     * cache is read or written.  The file is self-validating: it stores a
     * digest of the trusted setup and the table layout version, and is
     * silently rebuilt on mismatch.
     */
    const char *table_cache_path;

    /*
     * Verify that every trusted setup point is on the curve and in the correct
     * subgroup.  Costs roughly a second.  Recommended for untrusted input;
     * safe to skip for the canonical mainnet setup shipped with a client.
     */
    int validate_setup;


    /*
     * Number of blobs the prover should be able to have in flight.  Larger
     * values raise steady-state throughput at the cost of memory (roughly
     * 2 MiB per blob).  0 selects a sensible default.
     */
    uint32_t max_batch_size;
} mp_options;

/* Fills `opts` with the recommended defaults. */
void mp_options_default(mp_options *opts);

/* ----------------------------------------------------------------- prover */

typedef struct mp_prover mp_prover;

/*
 * Build a prover using the Ethereum mainnet trusted setup, which is compiled
 * into the library.  This is what production callers want: the ceremony values
 * are fixed for the lifetime of the protocol, so there is no file to ship,
 * locate or validate at runtime.
 */
mp_result mp_prover_new_default(mp_prover **out, const mp_options *opts);

/*
 * Build a prover from a caller-supplied monomial-form G1 trusted setup, for
 * testnets or a future ceremony.
 *
 * `g1_monomial_bytes` is MP_NUM_SETUP_G1_POINTS compressed points
 * (48 bytes each), in the same order and encoding c-kzg-4844 uses.
 */
mp_result mp_prover_new(mp_prover **out, const uint8_t *g1_monomial_bytes,
                              size_t g1_monomial_len, const mp_options *opts);


void mp_prover_free(mp_prover *p);

/* Name of the Metal device in use, e.g. "Apple M1". Valid for the prover's lifetime. */
const char *mp_prover_device_name(const mp_prover *p);

/* ---------------------------------------------------------------- compute */

/*
 * Compute all 128 cells and all 128 cell proofs for one blob.
 *
 * `blob`   is MP_BYTES_PER_BLOB bytes: 4096 big-endian canonical field
 *          elements.
 * `cells`  receives MP_CELLS_PER_EXT_BLOB * MP_BYTES_PER_CELL bytes, or
 *          NULL if the caller only wants proofs.
 * `proofs` receives MP_CELLS_PER_EXT_BLOB * MP_BYTES_PER_PROOF bytes, or
 *          NULL if the caller only wants cells.
 *
 * At least one of `cells` and `proofs` must be non-NULL.
 */
mp_result mp_compute_cells_and_proofs(mp_prover *p, uint8_t *cells, uint8_t *proofs,
                                            const uint8_t *blob);

/*
 * Batched form.  `blobs` is `num_blobs` consecutive blobs; `cells` and
 * `proofs` are the corresponding consecutive output arrays.  Batching keeps
 * the GPU saturated and is markedly more efficient per blob than repeated
 * single calls -- this is the entry point supernodes should use.
 */
mp_result mp_compute_cells_and_proofs_batch(mp_prover *p, uint8_t *cells, uint8_t *proofs,
                                                  const uint8_t *blobs, size_t num_blobs);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* METAL_PROVER_H */
