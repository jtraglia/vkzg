// Compute kernels for the EIP-7594 cell proof pipeline.
//
// Dispatch order for a batch of B blobs:
//
//   k_blob_to_fr        blob bytes -> bit-reversed Lagrange values
//   k_ntt_pass x2       inverse transform of size 4096 -> monomial coefficients
//   k_ntt_pass x2       forward transform of size 8192 -> extended evaluations
//   k_serialize_cells   evaluations -> cell bytes
//   k_build_circulant   64 circulant columns, each transformed to size 128
//   k_phase_a_accum     fixed-base bucket MSM against the precomputed setup
//   k_bucket_reduce     buckets -> u[j]
//   k_ladder            u[j] -> 2^(8d) * u[j]
//   (host normalises the ladder to affine)
//   k_phase_b_accum     circulant bucket MSM over the ladder
//   k_bucket_reduce     buckets -> proofs
//
// Every kernel takes the blob index in `.y` of the grid, so batching is just a
// wider dispatch.

static inline uint brev(uint x, uint bits) { return reverse_bits(x) >> (32u - bits); }

// ------------------------------------------------------------ deserialisation

kernel void k_blob_to_fr(device uint *out [[buffer(0)]], device const uchar *blob [[buffer(1)]],
                         device atomic_uint *err [[buffer(2)]],
                         uint2 gid [[thread_position_in_grid]]) {
    const uint i = gid.x;
    const uint b = gid.y;
    device const uchar *src = blob + (ulong)b * L_BYTES_PER_BLOB + (ulong)i * 32u;

    // 32 big-endian bytes -> 8 little-endian 32-bit limbs.
    Fr v;
#pragma clang loop unroll(full)
    for (int limb = 0; limb < FR_NLIMBS; limb++) {
        const uint off = (uint)(7 - limb) * 4u;
        v.v[limb] = ((uint)src[off] << 24) | ((uint)src[off + 1] << 16) |
                    ((uint)src[off + 2] << 8) | (uint)src[off + 3];
    }
    if (!fr_is_canonical(v)) {
        atomic_fetch_or_explicit(err, 1u, memory_order_relaxed);
    }
    const Fr m = fr_from_canonical(v);
    // The blob holds evaluations in bit-reversed order; undo that here so the
    // inverse transform sees natural order.
    const uint dst = brev(i, L_LOG_BLOB);
    fr_store(out + ((ulong)b * L_FIELD_ELEMENTS_PER_BLOB + dst) * FR_NLIMBS, m);
}

kernel void k_serialize_cells(device uchar *out [[buffer(0)]], device const uint *evals [[buffer(1)]],
                              uint2 gid [[thread_position_in_grid]]) {
    const uint i = gid.x;
    const uint b = gid.y;
    Fr v = fr_load(evals + ((ulong)b * L_FIELD_ELEMENTS_PER_EXT_BLOB + i) * FR_NLIMBS);
    v = fr_to_canonical(v);
    const uint dst = brev(i, L_LOG_EXT_BLOB);
    device uchar *o = out + (ulong)b * (L_FIELD_ELEMENTS_PER_EXT_BLOB * 32u) + (ulong)dst * 32u;
#pragma clang loop unroll(full)
    for (int limb = 0; limb < FR_NLIMBS; limb++) {
        const uint w = v.v[limb];
        const uint off = (uint)(7 - limb) * 4u;
        o[off + 0] = (uchar)(w >> 24);
        o[off + 1] = (uchar)(w >> 16);
        o[off + 2] = (uchar)(w >> 8);
        o[off + 3] = (uchar)w;
    }
}

// ------------------------------------------------------------------ transforms

// One pass of a four-step (N = N1 * N2) transform.  Each threadgroup owns one
// sub-transform, held in threadgroup memory, so a 4096 or 8192 point transform
// costs two dispatches instead of one per radix-2 stage.
struct NttParams {
    uint n;              // sub-transform size (<= 128)
    uint log_n;
    uint in_stride_t;    // element stride between transforms, input
    uint in_stride_i;    // element stride within a transform, input
    uint out_stride_t;
    uint out_stride_i;
    uint root_stride;    // 8192 / n, to index the shared root table
    uint twiddle_stride; // 8192 / N for the inter-pass twiddle, or 0 for none
    uint full_n;         // N
    uint in_batch;       // elements per blob, input
    uint out_batch;      // elements per blob, output
    uint scale;          // non-zero to multiply the result by scale_val
    uint scale_val[FR_NLIMBS];
};

kernel void k_ntt_pass(device uint *out [[buffer(0)]], device const uint *in [[buffer(1)]],
                       device const uint *roots [[buffer(2)]], constant NttParams &P [[buffer(3)]],
                       uint2 tgid [[threadgroup_position_in_grid]],
                       uint2 tid2 [[thread_position_in_threadgroup]]) {
    const uint tid = tid2.x;
    threadgroup uint tile[L_CIRCULANT_SIZE * FR_NLIMBS];

    const uint t = tgid.x;
    const uint b = tgid.y;
    device const uint *src = in + (ulong)b * P.in_batch * FR_NLIMBS;
    device uint *dst = out + (ulong)b * P.out_batch * FR_NLIMBS;

    if (tid < P.n) {
        const Fr v = fr_load(src + (ulong)(t * P.in_stride_t + tid * P.in_stride_i) * FR_NLIMBS);
        const uint j = brev(tid, P.log_n);
#pragma clang loop unroll(full)
        for (int k = 0; k < FR_NLIMBS; k++) tile[j * FR_NLIMBS + k] = v.v[k];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint len = 2u; len <= P.n; len <<= 1) {
        const uint hlen = len >> 1;
        const uint step = P.n / len;
        if (tid < (P.n >> 1)) {
            const uint blk = tid / hlen;
            const uint j = tid - blk * hlen;
            const uint base = blk * len;
            const Fr w = fr_load(roots + (ulong)(j * step * P.root_stride) * FR_NLIMBS);
            Fr a, c;
#pragma clang loop unroll(full)
            for (int k = 0; k < FR_NLIMBS; k++) a.v[k] = tile[(base + j) * FR_NLIMBS + k];
#pragma clang loop unroll(full)
            for (int k = 0; k < FR_NLIMBS; k++) c.v[k] = tile[(base + j + hlen) * FR_NLIMBS + k];
            c = fr_mul(c, w);
            const Fr s = fr_add(a, c);
            const Fr d = fr_sub(a, c);
#pragma clang loop unroll(full)
            for (int k = 0; k < FR_NLIMBS; k++) tile[(base + j) * FR_NLIMBS + k] = s.v[k];
#pragma clang loop unroll(full)
            for (int k = 0; k < FR_NLIMBS; k++) tile[(base + j + hlen) * FR_NLIMBS + k] = d.v[k];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid < P.n) {
        Fr v;
#pragma clang loop unroll(full)
        for (int k = 0; k < FR_NLIMBS; k++) v.v[k] = tile[tid * FR_NLIMBS + k];
        if (P.twiddle_stride != 0u) {
            const uint e = (t * tid) % P.full_n;
            v = fr_mul(v, fr_load(roots + (ulong)(e * P.twiddle_stride) * FR_NLIMBS));
        }
        if (P.scale != 0u) {
            Fr s;
#pragma clang loop unroll(full)
            for (int k = 0; k < FR_NLIMBS; k++) s.v[k] = P.scale_val[k];
            v = fr_mul(v, s);
        }
        fr_store(dst + (ulong)(t * P.out_stride_t + tid * P.out_stride_i) * FR_NLIMBS, v);
    }
}

// Builds the 64 circulant first-columns from the polynomial, transforms each to
// size 128, and writes the result transposed so that phase A reads one
// contiguous scalar vector per output.
kernel void k_build_circulant(device uint *coeffs [[buffer(0)]], device const uint *poly [[buffer(1)]],
                              device const uint *roots [[buffer(2)]],
                              uint2 tgid [[threadgroup_position_in_grid]],
                              uint2 tid2 [[thread_position_in_threadgroup]]) {
    const uint tid = tid2.x;
    threadgroup uint tile[L_CIRCULANT_SIZE * FR_NLIMBS];

    const uint i = tgid.x; // circulant offset, 0..63
    const uint b = tgid.y;
    // The polynomial lives in the zero-padded 8192-element buffer that feeds the
    // forward transform, so the per-blob stride is the extended one.
    device const uint *p = poly + (ulong)b * L_FIELD_ELEMENTS_PER_EXT_BLOB * FR_NLIMBS;

    // c_i[0] = poly[d - i]; c_i[128 - j] = poly[d - i - 64j] for j in 1..62.
    const uint d = L_FIELD_ELEMENTS_PER_BLOB - 1u;
    Fr v = fr_zero();
    if (tid == 0u) {
        v = fr_load(p + (ulong)(d - i) * FR_NLIMBS);
    } else if (tid >= 66u) {
        const uint j = L_CIRCULANT_SIZE - tid; // 1..62
        v = fr_load(p + (ulong)(d - i - j * L_FIELD_ELEMENTS_PER_CELL) * FR_NLIMBS);
    }
    {
        const uint br = brev(tid, L_LOG_CIRCULANT);
#pragma clang loop unroll(full)
        for (int k = 0; k < FR_NLIMBS; k++) tile[br * FR_NLIMBS + k] = v.v[k];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint root_stride = L_FIELD_ELEMENTS_PER_EXT_BLOB / L_CIRCULANT_SIZE;
    for (uint len = 2u; len <= L_CIRCULANT_SIZE; len <<= 1) {
        const uint hlen = len >> 1;
        const uint step = L_CIRCULANT_SIZE / len;
        if (tid < (L_CIRCULANT_SIZE / 2u)) {
            const uint blk = tid / hlen;
            const uint jj = tid - blk * hlen;
            const uint base = blk * len;
            const Fr w = fr_load(roots + (ulong)(jj * step * root_stride) * FR_NLIMBS);
            Fr a, c;
#pragma clang loop unroll(full)
            for (int k = 0; k < FR_NLIMBS; k++) a.v[k] = tile[(base + jj) * FR_NLIMBS + k];
#pragma clang loop unroll(full)
            for (int k = 0; k < FR_NLIMBS; k++) c.v[k] = tile[(base + jj + hlen) * FR_NLIMBS + k];
            c = fr_mul(c, w);
            const Fr s = fr_add(a, c);
            const Fr q = fr_sub(a, c);
#pragma clang loop unroll(full)
            for (int k = 0; k < FR_NLIMBS; k++) tile[(base + jj) * FR_NLIMBS + k] = s.v[k];
#pragma clang loop unroll(full)
            for (int k = 0; k < FR_NLIMBS; k++) tile[(base + jj + hlen) * FR_NLIMBS + k] = q.v[k];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    {
        Fr r;
#pragma clang loop unroll(full)
        for (int k = 0; k < FR_NLIMBS; k++) r.v[k] = tile[tid * FR_NLIMBS + k];
        // coeffs[j][i], j == tid
        fr_store(coeffs + ((ulong)b * L_CIRCULANT_SIZE * L_PHASE_A_TERMS +
                           (ulong)tid * L_PHASE_A_TERMS + i) *
                              FR_NLIMBS,
                 r);
    }
}

// ------------------------------------------------------------------- phase A
//
// Split into a scalar pass and a curve pass on purpose.  The recoding, the
// histogram and the bucket sort together are 0.6% of phase A's runtime, but
// keeping them in the same kernel as the accumulation forces one register
// allocation and one threadgroup-memory footprint for both, and occupancy is
// exactly what the accumulation is short of.  Separated, the curve kernel needs
// no threadgroup memory at all.

// Signed-digit recoding, histogram, load ordering and bucket sort for one
// output.  64 threads, one per scalar.
kernel void k_phase_a_sort(device ushort *items [[buffer(0)]], device uint *starts [[buffer(1)]],
                           device uint *perm [[buffer(2)]], device const uint *coeffs [[buffer(3)]],
                           uint2 tgid [[threadgroup_position_in_grid]],
                           uint2 tid2 [[thread_position_in_threadgroup]]) {
    const uint tid = tid2.x;
    threadgroup atomic_uint tg_count[L_NUM_BUCKETS];
    threadgroup uint tg_start[L_NUM_BUCKETS + 1];
    threadgroup uint tg_hist[L_LOAD_CLASSES];

    const uint j = tgid.x;
    const uint b = tgid.y;
    const ulong out_index = (ulong)b * L_CIRCULANT_SIZE + j;
    device ushort *my_items = items + out_index * L_PHASE_A_ITEMS;
    device uint *my_starts = starts + out_index * (L_NUM_BUCKETS + 1);
    device uint *my_perm = perm + out_index * L_NUM_BUCKETS;

    if (tid < L_NUM_BUCKETS / 2u) {
        atomic_store_explicit(&tg_count[tid], 0u, memory_order_relaxed);
        atomic_store_explicit(&tg_count[tid + L_NUM_BUCKETS / 2u], 0u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // digit = byte + carry, borrowing into the next digit past the half window.
    // Every scalar is below r < 2^255, so the carry chain always terminates
    // inside 32 digits and no 33rd digit is ever needed.
    Fr scalar = fr_load(coeffs + ((ulong)b * L_CIRCULANT_SIZE * L_PHASE_A_TERMS +
                                  (ulong)j * L_PHASE_A_TERMS + tid) *
                                     FR_NLIMBS);
    scalar = fr_to_canonical(scalar);
    int carry = 0;
#pragma clang loop unroll(full)
    for (uint dd = 0u; dd < L_NUM_DIGITS; dd++) {
        const uint byte = (scalar.v[dd >> 2] >> ((dd & 3u) * 8u)) & 0xffu;
        int v = (int)byte + carry;
        carry = 0;
        if (v > L_NUM_BUCKETS) {
            v -= 2 * L_NUM_BUCKETS;
            carry = 1;
        }
        if (v != 0) {
            atomic_fetch_add_explicit(&tg_count[(uint)abs(v) - 1u], 1u, memory_order_relaxed);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0u) {
        uint acc = 0u;
        for (uint k = 0u; k < L_NUM_BUCKETS; k++) {
            tg_start[k] = acc;
            acc += atomic_load_explicit(&tg_count[k], memory_order_relaxed);
            atomic_store_explicit(&tg_count[k], tg_start[k], memory_order_relaxed);
        }
        tg_start[L_NUM_BUCKETS] = acc;

        // Order the buckets by descending load before handing them to lanes.
        // A SIMD group runs until its slowest lane finishes, and loads are
        // Poisson(16): the max within a 32-lane group measures 24.8 against a
        // mean of 15.9, so a random assignment makes all four groups pay the
        // tail.  Counting sort into L_LOAD_CLASSES bins, heaviest first.
        for (uint c = 0u; c < L_LOAD_CLASSES; c++) tg_hist[c] = 0u;
        for (uint k = 0u; k < L_NUM_BUCKETS; k++) {
            tg_hist[min(tg_start[k + 1u] - tg_start[k], (uint)L_LOAD_CLASSES - 1u)]++;
        }
        uint pos = 0u;
        for (uint c = L_LOAD_CLASSES; c-- > 0u;) {
            const uint n = tg_hist[c];
            tg_hist[c] = pos;
            pos += n;
            if (c == 0u) break;
        }
        for (uint k = 0u; k < L_NUM_BUCKETS; k++) {
            const uint cls = min(tg_start[k + 1u] - tg_start[k], (uint)L_LOAD_CLASSES - 1u);
            my_perm[tg_hist[cls]++] = k;
        }
        for (uint k = 0u; k <= L_NUM_BUCKETS; k++) my_starts[k] = tg_start[k];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Scatter into bucket order; tg_count is now a per-bucket write cursor.
    carry = 0;
#pragma clang loop unroll(full)
    for (uint dd = 0u; dd < L_NUM_DIGITS; dd++) {
        const uint byte = (scalar.v[dd >> 2] >> ((dd & 3u) * 8u)) & 0xffu;
        int v = (int)byte + carry;
        carry = 0;
        if (v > L_NUM_BUCKETS) {
            v -= 2 * L_NUM_BUCKETS;
            carry = 1;
        }
        if (v != 0) {
            const uint neg = v < 0 ? 1u : 0u;
            const uint slot = atomic_fetch_add_explicit(&tg_count[(uint)abs(v) - 1u], 1u,
                                                        memory_order_relaxed);
            my_items[slot] = (ushort)((tid << 6) | (dd << 1) | neg);
        }
    }
}

// Bucket accumulation: one thread per bucket, no threadgroup memory.
kernel void k_phase_a(device uint *buckets [[buffer(0)]], device const ushort *items [[buffer(1)]],
                      device const uint *starts [[buffer(2)]], device const uint *perm [[buffer(3)]],
                      device const uint *table [[buffer(4)]],
                      uint2 tgid [[threadgroup_position_in_grid]],
                      uint2 tid2 [[thread_position_in_threadgroup]]) {
    const uint tid = tid2.x;
    const uint j = tgid.x;
    const uint b = tgid.y;
    const ulong out_index = (ulong)b * L_CIRCULANT_SIZE + j;

    const uint bucket = perm[out_index * L_NUM_BUCKETS + tid];
    device const uint *my_starts = starts + out_index * (L_NUM_BUCKETS + 1);
    device const ushort *my_items = items + out_index * L_PHASE_A_ITEMS;
    const uint lo = my_starts[bucket];
    const uint hi = my_starts[bucket + 1u];

    device const uint *tbase = table + (ulong)j * L_PHASE_A_TERMS * L_NUM_DIGITS * L_AFFINE_WORDS;
    G1 acc = g1_identity();
    for (uint it = lo; it < hi; it++) {
        const uint packed = my_items[it];
        G1A pt = g1a_load(tbase + (ulong)((packed >> 6) * L_NUM_DIGITS + ((packed >> 1) & 31u)) *
                                      L_AFFINE_WORDS);
        if ((packed & 1u) != 0u) pt.y = fp_neg(pt.y);
        acc = g1_madd(acc, pt);
    }
    g1_store(buckets +
                 ((ulong)b * L_CIRCULANT_SIZE * L_NUM_BUCKETS + (ulong)j * L_NUM_BUCKETS + bucket) *
                     L_JACOBIAN_WORDS,
             acc);
}

// ------------------------------------------------------------------- phase B

// out[a] = sum_e kappa[e] * u[a - e], the fused IFFT/truncate/FFT map.
//
// The taps and their signed digits are identical for every output -- only the
// point index rotates -- so the bucket-sorted item list comes precomputed from
// the host and no sorting happens here.
kernel void k_phase_b(device uint *out [[buffer(0)]], device const uint *ladder [[buffer(1)]],
                      device const uint *items [[buffer(2)]],
                      device const uint *offsets [[buffer(3)]],
                      device const uint *perm [[buffer(4)]],
                      uint2 tgid [[threadgroup_position_in_grid]],
                      uint2 tid2 [[thread_position_in_threadgroup]]) {
    const uint tid = tid2.x;
    const uint a = tgid.x;
    const uint b = tgid.y;
    // Same load-ordering trick as phase A, except the taps are fixed so the
    // permutation is computed once on the host.
    const uint bucket = perm[tid];
    const uint lo = offsets[bucket];
    const uint hi = offsets[bucket + 1u];
    device const uint *lbase =
        ladder + (ulong)b * L_CIRCULANT_SIZE * L_LADDER_POSITIONS * L_AFFINE_WORDS;

    G1 acc = g1_identity();
    for (uint it = lo; it < hi; it++) {
        const uint packed = items[it];
        const uint e = packed & 0xffu;
        const uint d = (packed >> 8) & 0xffu;
        const uint sign = (packed >> 16) & 1u;
        const uint src = (a - e) & (L_CIRCULANT_SIZE - 1u);
        G1A pt = g1a_load(lbase + (ulong)(src * L_LADDER_POSITIONS + d) * L_AFFINE_WORDS);
        if (sign != 0u) pt.y = fp_neg(pt.y);
        acc = g1_madd(acc, pt);
    }

    g1_store(out +
                 ((ulong)b * L_CIRCULANT_SIZE * L_NUM_BUCKETS + (ulong)a * L_NUM_BUCKETS + bucket) *
                     L_JACOBIAN_WORDS,
             acc);
}

// ---------------------------------------------------------- bucket reduction

// One SIMD group serves L_REDUCE_LANES-sized teams; a 128-thread threadgroup
// therefore reduces 16 outputs at once with no threadgroup memory at all.
kernel void k_bucket_reduce(device uint *out [[buffer(0)]], device const uint *buckets [[buffer(1)]],
                            constant uint &count [[buffer(2)]],
                            uint2 tgid [[threadgroup_position_in_grid]],
                            uint2 tid2 [[thread_position_in_threadgroup]]) {
    const uint tid = tid2.x;
    const uint team = tid / L_REDUCE_LANES;
    const uint pos = tid % L_REDUCE_LANES;
    // `count` need not be a multiple of the 16 outputs a threadgroup covers.
    // Teams past the end are clamped onto the last valid output rather than
    // returning early: reduce_buckets exchanges points with SIMD shuffles, and
    // those require every lane in the group to participate.
    const uint raw = tgid.x * L_REDUCE_OUTPUTS_PER_TG + team;
    const uint j = min(raw, count - 1u);
    const uint b = tgid.y;

    device const uint *base =
        buckets + ((ulong)b * L_CIRCULANT_SIZE * L_NUM_BUCKETS + (ulong)j * L_NUM_BUCKETS) *
                      L_JACOBIAN_WORDS;
    const G1 total = reduce_buckets(base, pos);
    if (pos == 0u && raw < count) {
        g1_store(out + ((ulong)b * L_CIRCULANT_SIZE + j) * L_JACOBIAN_WORDS, total);
    }
}

// ---------------------------------------------------------------- ladder

// L[j][d] = 2^(8d) * u[j], the shared base set for phase B.
//
// This is 248 sequential doublings over 128 points per blob, so at small batch
// sizes it is latency-bound rather than throughput-bound; batching is what
// makes it cheap, since the parallelism is 128 * blobs.
kernel void k_ladder(device uint *ladder [[buffer(0)]], device const uint *u [[buffer(1)]],
                     uint2 gid [[thread_position_in_grid]]) {
    const uint j = gid.x;
    const uint b = gid.y;
    G1 acc = g1_load(u + ((ulong)b * L_CIRCULANT_SIZE + j) * L_JACOBIAN_WORDS);
    for (uint d = 0u; d < L_LADDER_POSITIONS; d++) {
        g1_store(ladder + (((ulong)b * L_CIRCULANT_SIZE + j) * L_LADDER_POSITIONS + d) *
                              L_JACOBIAN_WORDS,
                 acc);
#pragma clang loop unroll(full)
        for (uint s = 0u; s < L_WINDOW_BITS; s++) acc = g1_dbl(acc);
    }
}

// ------------------------------------------------------- affine conversion

struct NormalizeParams {
    uint count; // total points
    uint chunk; // points per thread; one field inversion each
};

// Jacobian -> affine for `count` points, Montgomery's trick per chunk.
// Points at infinity become (0, 0), which the accumulation kernels treat as the
// identity.
kernel void k_normalize(device uint *out_affine [[buffer(0)]],
                        device const uint *in_jacobian [[buffer(1)]],
                        device uint *scratch [[buffer(2)]],
                        constant NormalizeParams &P [[buffer(3)]],
                        uint gid [[thread_position_in_grid]]) {
    const uint lo = gid * P.chunk;
    if (lo >= P.count) return;
    const uint hi = min(lo + P.chunk, P.count);

    Fp acc = fp_one();
    for (uint i = lo; i < hi; i++) {
        const Fp z = fp_load(in_jacobian + (ulong)i * L_JACOBIAN_WORDS + 2 * FP_NLIMBS);
        fp_store(scratch + (ulong)i * FP_NLIMBS, acc);
        if (!fp_is_zero(z)) acc = fp_mul(acc, z);
    }
    Fp inv = fp_inv(acc);

    for (uint i = hi; i-- > lo;) {
        device uint *dst = out_affine + (ulong)i * L_AFFINE_WORDS;
        const Fp z = fp_load(in_jacobian + (ulong)i * L_JACOBIAN_WORDS + 2 * FP_NLIMBS);
        if (fp_is_zero(z)) {
#pragma clang loop unroll(full)
            for (int k = 0; k < L_AFFINE_WORDS; k++) dst[k] = 0u;
        } else {
            const Fp pre = fp_load(scratch + (ulong)i * FP_NLIMBS);
            const Fp zinv = fp_mul(pre, inv);
            inv = fp_mul(inv, z);
            const Fp z2 = fp_sqr(zinv);
            const Fp z3 = fp_mul(z2, zinv);
            const Fp x = fp_load(in_jacobian + (ulong)i * L_JACOBIAN_WORDS);
            const Fp y = fp_load(in_jacobian + (ulong)i * L_JACOBIAN_WORDS + FP_NLIMBS);
            const Fp xa = fp_mul(x, z2);
            const Fp ya = fp_mul(y, z3);
            fp_store(dst, xa);
            fp_store(dst + FP_NLIMBS, ya);
        }
        if (i == lo) break;
    }
}

// ------------------------------------------------------ proof serialisation

// Affine proofs -> 48-byte compressed encoding, applying the bit-reversal
// permutation on the way out (FK20 emits natural order; cells are indexed
// bit-reversed).
kernel void k_compress_proofs(device uchar *out [[buffer(0)]],
                              device const uint *affine [[buffer(1)]],
                              uint2 gid [[thread_position_in_grid]]) {
    const uint i = gid.x;
    const uint b = gid.y;
    const G1A p = g1a_load(affine + ((ulong)b * L_CIRCULANT_SIZE + i) * L_AFFINE_WORDS);
    const uint dst = brev(i, L_LOG_CIRCULANT);
    device uchar *o = out + ((ulong)b * L_CIRCULANT_SIZE + dst) * L_BYTES_PER_PROOF;

    if (g1a_is_identity(p)) {
#pragma clang loop unroll(full)
        for (int k = 0; k < L_BYTES_PER_PROOF; k++) o[k] = 0;
        o[0] = 0xc0; // compressed | infinity
        return;
    }

    Fp x = p.x;
    Fp y = p.y;
    x = fp_to_canonical(x);
    y = fp_to_canonical(y);
    // 12 little-endian 32-bit limbs -> 48 big-endian bytes.
#pragma clang loop unroll(full)
    for (int limb = 0; limb < FP_NLIMBS; limb++) {
        const uint w = x.v[limb];
        const uint off = (uint)(FP_NLIMBS - 1 - limb) * 4u;
        o[off + 0] = (uchar)(w >> 24);
        o[off + 1] = (uchar)(w >> 16);
        o[off + 2] = (uchar)(w >> 8);
        o[off + 3] = (uchar)w;
    }
    uchar flags = 0x80; // compressed
    if (fp_is_lex_largest(y)) flags |= 0x20;
    o[0] |= flags;
}
