# kzgpu

EIP-7594 cell KZG proof generation for Ethereum blobs, on Apple GPUs.

Given a blob, `kzgpu` produces all 128 cells and all 128 cell proofs. It is
built for supernodes, which generate proofs constantly and care about
throughput. Proof *verification* is deliberately out of scope — this library
only produces.

Verified against every `compute_cells_and_kzg_proofs` vector in the consensus
spec test suite, cells and proofs byte-for-byte.

## Results

Apple M1 (8 GPU cores, 4P+4E CPU cores), macOS 27, passively-cooled MacBook
Air. c-kzg and kzgpu measured back to back in the same thermal state.

**Single blob (latency).** c-kzg cannot parallelise one blob, so this is its
single-thread time:

| | per blob | speedup |
|---|---|---|
| c-kzg-4844, `precompute=8`, on a P-core | 171 ms | 1.0× |
| kzgpu, GPU only | 70 ms | 2.4× |
| **kzgpu, GPU + CPU** | **42.7 ms** | **4.0×** |

**Throughput.** c-kzg is single-threaded per call but a supernode can run one
per core, which is the comparison that actually matters:

| | blobs/s | ms/blob |
|---|---|---|
| c-kzg-4844, 1 thread | 4.9 | 204 |
| c-kzg-4844, 4 threads | 19.0 | 53 |
| c-kzg-4844, 8 threads | 26.1 | 38 |
| **c-kzg-4844, 10-12 threads (saturated)** | **26.9** | **37.2** |
| kzgpu, GPU only, batch 64 | 21.0 | 47.7 |
| **kzgpu, GPU + CPU, batch 64** | **33-35** | **28.6-30.7** |

So against c-kzg using every core, this library is **~1.3× faster** in
throughput and **4× faster** on a single blob.

Note the GPU-only row: on this machine it is *slower* than 8-thread c-kzg. That
is not a defect in the port — it is what the M1's GPU is, see
[the integer multiplier](#1-the-gpus-integer-multiplier-is-the-scarce-resource)
below. The M1 has the smallest GPU Apple ships (8 cores); this library's GPU
path scales with GPU core count while c-kzg scales with CPU core count, so the
gap widens in kzgpu's favour on every larger part.

Cells alone (no proofs) are 0.47 ms against c-kzg's 2.5 ms.

### Batching

Efficiency plateaus quickly and then holds:

| batch | 1 | 2 | 4 | 8 | 16 | 32 | 64 | 128 |
|---|---|---|---|---|---|---|---|---|
| ms/blob (GPU+CPU) | 42.6 | 33.9 | 30.8 | 30.5 | 29.2 | 28.7 | **28.6** | 29.9 |
| ms/blob (GPU only) | 70.0 | 57.6 | 55.3 | 51.5 | 48.7 | 48.8 | 48.0 | 47.5 |

(Run-to-run spread on a fanless Air is a few percent; 28.6 was the best batch-64
figure observed, 30.7 the worst.)

Most of the win is in by 8; 64 is the optimum; 128 gives back ~4% to memory
pressure. Working set is **4.8 MiB per blob** plus a fixed ~28 MiB (the 24 MiB
FK20 table and the root tables), so a batch of 64 costs 334 MiB and 128 costs
640 MiB.

A full block is at most a handful of blobs today, so batch 8–16 covers block
production; use 32–64 when reconstructing in bulk.

### Other Apple GPUs

Extrapolated, not measured — I only have an 8-core M1. The GPU part is
compute-bound on integer multiplies, so it should scale with GPU core count;
the ladder and scalar stages (~2.5 ms/blob) do not.

| | GPU cores | est. ms/blob, GPU only | est. blobs/s |
|---|---|---|---|
| M1 (measured) | 8 | 48.0 | 20.8 |
| M1 Pro | 16 | ~25 | ~40 |
| M1 Max | 32 | ~14 | ~72 |
| M1 Ultra | 64 | ~8 | ~120 |
| M4 Max | 40 | ~11.5 | ~85 |

Caveats worth taking seriously:

- These assume per-core integer-multiply throughput matches the M1's measured
  8.9 cycles per 32×32→64 multiply. Apple has reworked the GPU since (dynamic
  caching in M3, further changes in M5) and I cannot measure any of it. If
  integer multiply got faster per core, every row above is pessimistic.
- I would not put a number on an M5 Max without knowing its GPU core count. If
  it lands near the M4 Max's 40 cores, the M4 Max row is the right ballpark.
  Note that M5's headline GPU feature is a Neural Accelerator per core — matrix
  units for ML, which do nothing for 381-bit integer arithmetic.
- The Ultra parts are two dies over an interconnect. This workload is
  compute-bound rather than bandwidth-bound, so it should scale better than
  memory-heavy kernels, but I have not verified it.
- On those larger parts the CPU assist matters much less, because the GPU stops
  being the smaller half of the machine. On an M1 Ultra the GPU-only estimate
  (~120 blobs/s) already beats what 20 CPU cores could do with c-kzg.

## Quick start

```sh
cmake -B build && cmake --build build -j
ctest --test-dir build --output-on-failure
./build/kzgpu_example data/trusted_setup.txt
```

```c
#include "kzgpu.h"

kzgpu_options opts;
kzgpu_options_default(&opts);
opts.table_cache_path = "/var/cache/kzgpu/tables.bin";  /* ~1s to build, ms to load */
opts.max_batch_size = 8;

kzgpu_prover *prover;
kzgpu_prover_new_from_file(&prover, "trusted_setup.txt", &opts);

/* one blob */
kzgpu_compute_cells_and_proofs(prover, cells, proofs, blob);

/* or a batch, which is markedly more efficient per blob */
kzgpu_compute_cells_and_proofs_batch(prover, cells, proofs, blobs, n);
```

The whole API is in [`include/kzgpu.h`](include/kzgpu.h) and is plain C, so
Rust / Go / Java bindings need no C++ shim. A `kzgpu_prover` is safe to share
between threads; calls serialise internally on the GPU queue.

## How it works, and why it looks like this

The shape of this library is the result of two measurements taken before any
of it was written.

### 1. The GPU's integer multiplier is the scarce resource

BLS12-381 arithmetic is 381-bit modular multiplication, which on a GPU means
32×32→64 integer multiplies. On Apple7 (M1) one of those issues about every
**8.9 cycles per lane** — Apple's GPUs are built for float throughput, not
integer multiplies. Measured ceilings:

| | rate |
|---|---|
| `(ulong)a * b` (widening 32×32) | 147 G/s |
| F_p multiply (12 limbs, CIOS) | 392 M/s |
| (for scale: one CPU core does 33 M/s) | |
| F_p add / sub | 4.0 G/s |
| G1 mixed addition | 22.2 M/s |

FK20 for one blob is ~528k mixed additions, so **~24 ms is the floor** for this
algorithm on this GPU even at perfect efficiency. That framed everything below,
and it is why the target is 5× rather than 10×.

### 2. Single-thread latency has a hard floor

One F_p multiply takes **3.43 µs** for one thread and never gets faster, no
matter how idle the GPU is — throughput saturates at ~2048 resident threads,
below which you are simply paying latency.

This is fatal for the textbook FK20. c-kzg computes the proofs as
`IFFT → truncate → FFT` over G1, and each butterfly of those size-128
transforms multiplies a G1 point by a root of unity: a ~255-doubling dependency
chain, ≈2000 F_p multiplies deep, ≈7 ms of pure latency *each*. Seven levels of
that, twice, is ~170 ms of latency alone. On this M1 those two transforms are
48.6 ms of c-kzg's 168 ms — the single worst-mapping part of the algorithm.

**So the two G1 transforms are fused away.** Composing `IFFT ∘ truncate ∘ FFT`
gives a cyclic convolution whose kernel collapses because ω⁶⁴ = −1:

```
out[a] = Σ_e κ[e] · u[a-e],   κ[0] = 1/2,  κ[e] = −1/(64(ω^e − 1)) for odd e,  else 0
```

65 non-zero taps instead of two transforms, and no G1 point is ever multiplied
by a full scalar in a dependency chain. It becomes one more bucket MSM over a
shared doubling ladder. `tests/test_reference` proves the fused form agrees
with the two-transform form on all 128 outputs.

### The pipeline

```
blob → F_r, bit-reversed                  GPU
inverse NTT 4096 → monomial coefficients  GPU   (four-step, 2 dispatches)
forward NTT 8192 → cells                  GPU
64 circulant columns, each NTT-128        GPU
phase A: fixed-base bucket MSM            GPU ∥ CPU
doubling ladder 2^(8d)·u[j] → affine      CPU
phase B: the fused circulant map          GPU ∥ CPU
proofs → affine → compressed              CPU
```

Both MSM phases use a signed-digit window of 8 bits: 32 digits, 128 buckets.
Phase A's bases are fixed, so the setup precomputes `2^(8d)·P` for every base
and position — a 24 MiB table, cached on disk.

Three things about the MSM turned out to matter more than expected:

- **Bucket loads are Poisson(16), and a SIMD group waits for its slowest lane.**
  Randomly assigned, all four groups pay the global tail (max 24.8 vs mean
  15.9). Sorting buckets by descending load first, so only the first group sees
  the tail, was worth **1.24×**.
- **Where the bucket reduction lives.** As a standalone kernel it needed 13.8 KB
  of threadgroup memory for partial points, capping residency at 512 threads and
  costing as much as the MSM itself. Fused into the MSM kernel it was *worse*
  (2.7×), because register allocation is per-kernel. A pure shuffle tree was
  worse still, because every lane executes every level. The shape that works is
  8 lanes per output, each collapsing 16 buckets serially, then a short shuffle
  tree.
- **The ladder belongs on the CPU.** 248 sequential doublings over 128 points is
  almost pure latency: 19.6 ms on the GPU, well under 1 ms across idle cores.

### 3. Why a GPU library uses the CPU

Because on this machine they are the same size. The GPU does ~392M F_p
multiplies/s; one CPU core does ~23M, and there are eight. Using only the GPU
leaves half the machine idle.

Both phases have 128 independent outputs, so the GPU takes `[0, split)` and a
persistent thread pool takes the rest, running while the command buffer is in
flight. The split is re-balanced from measured per-output cost. That is the
49 ms → 33 ms in the table.

This is configurable — `cpu_assist_threads = -1` disables it entirely — because
a node process may well want its cores back. The GPU-only path is a supported
configuration, not a fallback.

## Layout

```
include/kzgpu.h        public C API
src/layout_defs.h      sizes shared by host and shaders (single source of truth)
src/shaders/           Metal: field.metal (F_p/F_r/G1), kernels.metal (pipeline)
src/cpu/               host BLS12-381, trusted setup + FK20 tables, CPU MSM workers
src/cpu/reference.cpp  CPU implementation of the exact same pipeline, for tests
src/kzgpu_metal.mm     Metal host layer, scheduling, CPU/GPU split
tools/                 constant and shader generators
tests/                 unit tests, spec vectors, GPU end-to-end
bench/                 latency/throughput benchmark, per-stage profiler
```

Shaders are compiled from embedded source at load (~80 ms once): the offline
`metal` compiler ships with Xcode, not the Command Line Tools, and embedding
keeps the built library self-contained.

Field constants for the host (64-bit limbs) and the device (32-bit limbs) are
emitted from one script, so the two representations cannot drift.

## Limits and what would move next

- **~24 ms is the algorithmic floor** on an M1 for the current formulation. Real
  gains from here need fewer curve operations, not better scheduling.
- **Phase B's tap count** can drop from 65 to ~43 by recursively splitting the
  cyclic convolution (`X¹²⁸−1 = (X⁶⁴−1)(X⁶⁴+1)`), which costs only additions in
  the ladder domain. Worth ~15% of total; not implemented.
- **GLV** would halve the ladder's depth (248 → 120 doublings) and its memory.
  The host-side pieces (`glv_split`, the endomorphism) are implemented and
  tested; the phase B kernel does not use them yet.
- **A faster host F_p multiply.** Ours is 29.9 ns of portable C++ against blst's
  ~20 ns of hand-scheduled assembly. Since the CPU carries roughly half the
  work, closing the remaining gap is worth ~10% overall.
- **`recover_cells_and_kzg_proofs`** is the natural next entry point for
  supernodes. The expensive half of it is exactly the proof computation this
  library already does; what is missing is the erasure-recovery step.

## License

The trusted setup and test vectors under `data/` and `tests/vectors/` are from
[c-kzg-4844](https://github.com/ethereum/c-kzg-4844) (Apache-2.0).
