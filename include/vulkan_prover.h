/*
 * vulkan-prover -- EIP-7594 cell KZG proof generation on the GPU via Vulkan.
 *
 * The whole pipeline runs on the GPU: one command buffer per call, with the
 * host doing nothing but copying blobs in and cells/proofs out.  That is
 * deliberate -- on a headless node the GPU is idle while the CPU has real work
 * to do, so spending CPU cycles here would be taking them from something else.
 *
 * The API is intentionally plain C so that Rust, Go, Java (JNI/Panama) and
 * others can bind to it without a C++ shim.  All functions are thread safe
 * unless stated otherwise; a single `vkp_prover` may be shared between
 * threads, and concurrent calls are serialised internally per GPU queue.
 *
 * This library only *produces* cells and proofs.  Verification is deliberately
 * out of scope.
 */
#ifndef VULKAN_PROVER_H
#define VULKAN_PROVER_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ------------------------------------------------------------------ sizes */

#define VKP_FIELD_ELEMENTS_PER_BLOB 4096
#define VKP_FIELD_ELEMENTS_PER_EXT_BLOB 8192
#define VKP_FIELD_ELEMENTS_PER_CELL 64
#define VKP_CELLS_PER_EXT_BLOB 128
#define VKP_BYTES_PER_FIELD_ELEMENT 32
#define VKP_BYTES_PER_BLOB (VKP_FIELD_ELEMENTS_PER_BLOB * VKP_BYTES_PER_FIELD_ELEMENT)
#define VKP_BYTES_PER_CELL (VKP_FIELD_ELEMENTS_PER_CELL * VKP_BYTES_PER_FIELD_ELEMENT)
#define VKP_BYTES_PER_PROOF 48
/* Number of G1 points in the monomial-form trusted setup we consume. */
#define VKP_NUM_SETUP_G1_POINTS VKP_FIELD_ELEMENTS_PER_BLOB
#define VKP_BYTES_PER_G1 48

/* ----------------------------------------------------------------- status */

typedef enum {
    VKP_OK = 0,
    VKP_ERR_BADARGS = 1,     /* caller passed a null/invalid argument */
    VKP_ERR_MALLOC = 2,      /* host allocation failed */
    VKP_ERR_IO = 3,          /* trusted setup or cache file could not be read */
    VKP_ERR_SETUP = 4,       /* trusted setup was malformed or off-curve */
    VKP_ERR_GPU = 5,         /* no Vulkan device, or a shader failed to build */
    VKP_ERR_INVALID_BLOB = 6 /* a field element in the blob was not canonical */
} vkp_result;

/* Human readable form of a status code. Never returns NULL. */
const char *vkp_error_string(vkp_result r);

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
} vkp_options;

/* Fills `opts` with the recommended defaults. */
void vkp_options_default(vkp_options *opts);

/* ----------------------------------------------------------------- prover */

typedef struct vkp_prover vkp_prover;

/*
 * Build a prover using the Ethereum mainnet trusted setup, which is compiled
 * into the library.  This is what production callers want: the ceremony values
 * are fixed for the lifetime of the protocol, so there is no file to ship,
 * locate or validate at runtime.
 */
vkp_result vkp_prover_new_default(vkp_prover **out, const vkp_options *opts);

/*
 * Build a prover from a caller-supplied monomial-form G1 trusted setup, for
 * testnets or a future ceremony.
 *
 * `g1_monomial_bytes` is VKP_NUM_SETUP_G1_POINTS compressed points
 * (48 bytes each), in the same order and encoding c-kzg-4844 uses.
 */
vkp_result vkp_prover_new(vkp_prover **out, const uint8_t *g1_monomial_bytes,
                              size_t g1_monomial_len, const vkp_options *opts);


void vkp_prover_free(vkp_prover *p);

/* Name of the Vulkan device in use, e.g. "Apple M1 (G13G B1)". Valid for the prover's lifetime. */
const char *vkp_prover_device_name(const vkp_prover *p);

/* ---------------------------------------------------------------- compute */

/*
 * Compute all 128 cells and all 128 cell proofs for one blob.
 *
 * `blob`   is VKP_BYTES_PER_BLOB bytes: 4096 big-endian canonical field
 *          elements.
 * `cells`  receives VKP_CELLS_PER_EXT_BLOB * VKP_BYTES_PER_CELL bytes, or
 *          NULL if the caller only wants proofs.
 * `proofs` receives VKP_CELLS_PER_EXT_BLOB * VKP_BYTES_PER_PROOF bytes, or
 *          NULL if the caller only wants cells.
 *
 * At least one of `cells` and `proofs` must be non-NULL.
 */
vkp_result vkp_compute_cells_and_proofs(vkp_prover *p, uint8_t *cells, uint8_t *proofs,
                                            const uint8_t *blob);

/*
 * Batched form.  `blobs` is `num_blobs` consecutive blobs; `cells` and
 * `proofs` are the corresponding consecutive output arrays.  Batching keeps
 * the GPU saturated and is markedly more efficient per blob than repeated
 * single calls -- this is the entry point supernodes should use.
 */
vkp_result vkp_compute_cells_and_proofs_batch(vkp_prover *p, uint8_t *cells, uint8_t *proofs,
                                                  const uint8_t *blobs, size_t num_blobs);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* VULKAN_PROVER_H */
