/*
 * vkzg -- EIP-7594 cell KZG proof generation on the GPU via Vulkan.
 *
 * The whole pipeline runs on the GPU: one command buffer per call, with the
 * host doing nothing but copying blobs in and proofs out.  That is
 * deliberate -- on a headless node the GPU is idle while the CPU has real work
 * to do, so spending CPU cycles here would be taking them from something else.
 *
 * The API is intentionally plain C so that Rust, Go, Java (JNI/Panama) and
 * others can bind to it without a C++ shim.  All functions are thread safe
 * unless stated otherwise; a single `vkzg_prover` may be shared between
 * threads, and concurrent calls are serialised internally per GPU queue.
 *
 * This library only *produces* cell proofs -- not the cells themselves
 * (computing cells from a blob is cheap on a CPU and out of scope here) and
 * not verification.
 */
#ifndef VKZG_H
#define VKZG_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ------------------------------------------------------------------ sizes */

#define VKZG_FIELD_ELEMENTS_PER_BLOB 4096
#define VKZG_NUM_CELL_PROOFS 128
#define VKZG_BYTES_PER_FIELD_ELEMENT 32
#define VKZG_BYTES_PER_BLOB (VKZG_FIELD_ELEMENTS_PER_BLOB * VKZG_BYTES_PER_FIELD_ELEMENT)
#define VKZG_BYTES_PER_PROOF 48
/* Number of G1 points in the monomial-form trusted setup we consume. */
#define VKZG_NUM_SETUP_G1_POINTS VKZG_FIELD_ELEMENTS_PER_BLOB
#define VKZG_BYTES_PER_G1 48

/* ----------------------------------------------------------------- status */

typedef enum {
    VKZG_OK = 0,
    VKZG_ERR_BADARGS = 1,     /* caller passed a null/invalid argument */
    VKZG_ERR_MALLOC = 2,      /* host allocation failed */
    VKZG_ERR_IO = 3,          /* trusted setup or cache file could not be read */
    VKZG_ERR_SETUP = 4,       /* trusted setup was malformed or off-curve */
    VKZG_ERR_GPU = 5,         /* no Vulkan device, or a shader failed to build */
    VKZG_ERR_INVALID_BLOB = 6 /* a field element in the blob was not canonical */
} vkzg_result;

/* Human readable form of a status code. Never returns NULL. */
const char *vkzg_error_string(vkzg_result r);

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
} vkzg_options;

/* Fills `opts` with the recommended defaults. */
void vkzg_options_default(vkzg_options *opts);

/* ----------------------------------------------------------------- prover */

typedef struct vkzg_prover vkzg_prover;

/*
 * Build a prover using the Ethereum mainnet trusted setup, which is compiled
 * into the library.  This is what production callers want: the ceremony values
 * are fixed for the lifetime of the protocol, so there is no file to ship,
 * locate or validate at runtime.
 */
vkzg_result vkzg_prover_new_default(vkzg_prover **out, const vkzg_options *opts);

/*
 * Build a prover from a caller-supplied monomial-form G1 trusted setup, for
 * testnets or a future ceremony.
 *
 * `g1_monomial_bytes` is VKZG_NUM_SETUP_G1_POINTS compressed points
 * (48 bytes each), in the same order and encoding c-kzg-4844 uses.
 */
vkzg_result vkzg_prover_new(vkzg_prover **out, const uint8_t *g1_monomial_bytes,
                              size_t g1_monomial_len, const vkzg_options *opts);

void vkzg_prover_free(vkzg_prover *p);

/* Name of the Vulkan device in use, e.g. "Apple M1 (G13G B1)". Valid for the prover's lifetime. */
const char *vkzg_prover_device_name(const vkzg_prover *p);

/* ---------------------------------------------------------------- compute */

/*
 * Compute all 128 cell proofs for one blob.
 *
 * `blob`   is VKZG_BYTES_PER_BLOB bytes: 4096 big-endian canonical field
 *          elements.
 * `proofs` receives VKZG_NUM_CELL_PROOFS * VKZG_BYTES_PER_PROOF bytes.
 */
vkzg_result vkzg_compute_proofs(vkzg_prover *p, uint8_t *proofs, const uint8_t *blob);

/*
 * Batched form.  `blobs` is `num_blobs` consecutive blobs; `proofs` is the
 * corresponding consecutive output array.  Batching keeps the GPU saturated
 * and is markedly more efficient per blob than repeated single calls -- this
 * is the entry point supernodes should use.
 */
vkzg_result vkzg_compute_proofs_batch(vkzg_prover *p, uint8_t *proofs, const uint8_t *blobs,
                                        size_t num_blobs);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* VKZG_H */
