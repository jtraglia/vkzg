/*
 * kzgpu -- EIP-7594 cell KZG proof generation on Apple GPUs.
 *
 * The API is intentionally plain C so that Rust, Go, Java (JNI/Panama) and
 * others can bind to it without a C++ shim.  All functions are thread safe
 * unless stated otherwise; a single `kzgpu_prover` may be shared between
 * threads, and concurrent calls are serialised internally per GPU queue.
 *
 * This library only *produces* cells and proofs.  Verification is deliberately
 * out of scope.
 */
#ifndef KZGPU_H
#define KZGPU_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ------------------------------------------------------------------ sizes */

#define KZGPU_FIELD_ELEMENTS_PER_BLOB 4096
#define KZGPU_FIELD_ELEMENTS_PER_EXT_BLOB 8192
#define KZGPU_FIELD_ELEMENTS_PER_CELL 64
#define KZGPU_CELLS_PER_EXT_BLOB 128
#define KZGPU_BYTES_PER_FIELD_ELEMENT 32
#define KZGPU_BYTES_PER_BLOB (KZGPU_FIELD_ELEMENTS_PER_BLOB * KZGPU_BYTES_PER_FIELD_ELEMENT)
#define KZGPU_BYTES_PER_CELL (KZGPU_FIELD_ELEMENTS_PER_CELL * KZGPU_BYTES_PER_FIELD_ELEMENT)
#define KZGPU_BYTES_PER_PROOF 48
/* Number of G1 points in the monomial-form trusted setup we consume. */
#define KZGPU_NUM_SETUP_G1_POINTS KZGPU_FIELD_ELEMENTS_PER_BLOB
#define KZGPU_BYTES_PER_G1 48

/* ----------------------------------------------------------------- status */

typedef enum {
    KZGPU_OK = 0,
    KZGPU_ERR_BADARGS = 1,     /* caller passed a null/invalid argument */
    KZGPU_ERR_MALLOC = 2,      /* host allocation failed */
    KZGPU_ERR_IO = 3,          /* trusted setup or cache file could not be read */
    KZGPU_ERR_SETUP = 4,       /* trusted setup was malformed or off-curve */
    KZGPU_ERR_GPU = 5,         /* no Metal device, or a shader failed to build */
    KZGPU_ERR_INVALID_BLOB = 6 /* a field element in the blob was not canonical */
} kzgpu_result;

/* Human readable form of a status code. Never returns NULL. */
const char *kzgpu_error_string(kzgpu_result r);

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
} kzgpu_options;

/* Fills `opts` with the recommended defaults. */
void kzgpu_options_default(kzgpu_options *opts);

/* ----------------------------------------------------------------- prover */

typedef struct kzgpu_prover kzgpu_prover;

/*
 * Build a prover from the monomial-form G1 trusted setup.
 *
 * `g1_monomial_bytes` is KZGPU_NUM_SETUP_G1_POINTS compressed points
 * (48 bytes each), in the same order and encoding c-kzg-4844 uses.
 */
kzgpu_result kzgpu_prover_new(kzgpu_prover **out, const uint8_t *g1_monomial_bytes,
                              size_t g1_monomial_len, const kzgpu_options *opts);

/*
 * Convenience loader for the standard `trusted_setup.txt` format:
 *     <n1> <n2> <n1 lagrange g1> <n2 g2> <n1 monomial g1>
 * Only the monomial G1 section is used; the rest is parsed and discarded.
 */
kzgpu_result kzgpu_prover_new_from_file(kzgpu_prover **out, const char *trusted_setup_path,
                                        const kzgpu_options *opts);

void kzgpu_prover_free(kzgpu_prover *p);

/* Name of the Metal device in use, e.g. "Apple M1". Valid for the prover's lifetime. */
const char *kzgpu_device_name(const kzgpu_prover *p);

/* ---------------------------------------------------------------- compute */

/*
 * Compute all 128 cells and all 128 cell proofs for one blob.
 *
 * `blob`   is KZGPU_BYTES_PER_BLOB bytes: 4096 big-endian canonical field
 *          elements.
 * `cells`  receives KZGPU_CELLS_PER_EXT_BLOB * KZGPU_BYTES_PER_CELL bytes, or
 *          NULL if the caller only wants proofs.
 * `proofs` receives KZGPU_CELLS_PER_EXT_BLOB * KZGPU_BYTES_PER_PROOF bytes, or
 *          NULL if the caller only wants cells.
 *
 * At least one of `cells` and `proofs` must be non-NULL.
 */
kzgpu_result kzgpu_compute_cells_and_proofs(kzgpu_prover *p, uint8_t *cells, uint8_t *proofs,
                                            const uint8_t *blob);

/*
 * Batched form.  `blobs` is `num_blobs` consecutive blobs; `cells` and
 * `proofs` are the corresponding consecutive output arrays.  Batching keeps
 * the GPU saturated and is markedly more efficient per blob than repeated
 * single calls -- this is the entry point supernodes should use.
 */
kzgpu_result kzgpu_compute_cells_and_proofs_batch(kzgpu_prover *p, uint8_t *cells, uint8_t *proofs,
                                                  const uint8_t *blobs, size_t num_blobs);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* KZGPU_H */
