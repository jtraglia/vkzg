# kzgpu

EIP-7594 cell KZG proof generation for Ethereum blobs, on Apple GPUs.

Given a blob, `kzgpu` produces all 128 cells and all 128 cell proofs. It is
built for supernodes, which generate proofs constantly and care about
throughput. Proof *verification* is deliberately out of scope — this library
only produces.

Verified against every `compute_cells_and_kzg_proofs` vector in the consensus
spec test suite, cells and proofs byte-for-byte.

## Results

Everything runs on the GPU. One command buffer per call; the host only copies
blobs in and cells/proofs out. No worker threads, no CPU field arithmetic in
the compute path — on a headless node the GPU is the idle resource and the CPU
has real work to do.

Apple M1, 8 GPU cores, macOS 27, fanless MacBook Air:

| | ms/blob | blobs/s |
|---|---|---|
| batch 1 | 87.2 | 11.5 |
| batch 8 | 49.8 | 20.1 |
| batch 32 | 47.3 | 21.2 |
| **batch 64** | **46.8** | **21.4** |
| batch 128 | 47.9 | 20.9 |

Cells alone (no proofs) are 0.47 ms/blob.

For reference, c-kzg-4844 on the same machine: 171 ms/blob single-threaded, and
26.9 blobs/s when saturating all 8 CPU cores.

**Read that comparison carefully.** On *this* machine — the smallest GPU Apple
ships, 8 cores — the GPU alone does not beat eight CPU cores at raw throughput
(21.4 vs 26.9). What it does is deliver 21.4 blobs/s while using **no CPU at
all**, so the relevant question for a node is not "GPU versus eight free cores"
but "GPU versus however many cores you can actually spare". Against a single
core it is 4.4× the throughput and 2× the single-blob latency.

The reason is measured, not incidental: Apple's GPUs have a weak integer
multiplier (see [below](#1-the-gpus-integer-multiplier-is-the-scarce-resource)),
so an 8-core GPU lands in the same range as 8 CPU cores for 381-bit modular
arithmetic. Everything in this pipeline scales with GPU core count, so the
picture changes completely on larger parts.

### Batching

Batch at least 16. Two stages need it: the doubling ladder is 128 independent
chains per blob (128 × batch threads), and the batched inversions amortise a
~2 ms field inversion across their chunk. Below batch 8 both are latency-bound
and you pay for it.

| batch | 1 | 2 | 4 | 8 | 16 | 32 | 64 | 128 |
|---|---|---|---|---|---|---|---|---|
| ms/blob | 87.2 | 63.6 | 56.7 | 49.8 | 49.8 | 47.3 | **46.8** | 47.9 |

Working set is **5.6 MiB per blob** plus a fixed ~28 MiB (the 24 MiB FK20 table
and the root tables): 384 MiB at batch 64, 739 MiB at batch 128. Run-to-run
spread on a fanless Air is a few percent.

### Larger Apple GPUs

Extrapolated, not measured — I only have an 8-core M1. Since the whole pipeline
is now GPU-resident and compute-bound on integer multiplies, it should track
GPU core count fairly directly:

| | GPU cores | est. ms/blob | est. blobs/s |
|---|---|---|---|
| M1 *(measured, batch 64)* | 8 | 46.8 | 21.4 |
| M1 Pro | 16 | ~24 | ~42 |
| M1 Max | 32 | ~13 | ~78 |
| **M1 Ultra** | 64 | **~7-9** | **~110-170** |
| M4 Max | 40 | ~10-12 | ~85-100 |

On an M1 Ultra that would be roughly 4-6× what all 20 of its CPU cores could do
with c-kzg, while leaving those cores free.

Use larger batches on larger GPUs — batch 64-128 on an Ultra, since saturation
needs roughly 8× the threads of an 8-core part. Memory is the only limit
(5.6 MiB/blob).

Caveats worth taking seriously:

- These assume per-core integer-multiply throughput matches the M1's measured
  8.9 cycles per 32×32→64 multiply. Apple reworked the GPU in M3 (dynamic
  caching) and again in M5, and I cannot measure any of it. If integer multiply
  improved per core, every row is pessimistic.
- The Ultra parts are two dies over an interconnect. This workload is
  compute-bound rather than bandwidth-bound so it should scale better than
  memory-heavy kernels, but I have not verified it.
- I would not put a number on an M5 Max without knowing its GPU core count. If
  it lands near the M4 Max's 40, that row is the ballpark. Note that M5's
  headline GPU feature is a Neural Accelerator per core — matrix units for ML,
  which do nothing for 381-bit integer arithmetic.
- A few stages (the ladder, the inversions) are latency-bound rather than
  throughput-bound, so they will not shrink with core count the way the MSMs
  do. That is why the Ultra estimate is a range rather than 46.8/8 = 5.9.

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
| (for scale: one M1 CPU core does 33 M/s) | |
| F_p add / sub | 4.0 G/s |
| G1 mixed addition | 22.2 M/s |

FK20 for one blob is ~528k mixed additions, so **~24 ms is the floor** for this
algorithm on an 8-core GPU even at perfect efficiency. That framed everything
below — and it scales: the same floor on a 64-core part is ~3 ms.

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
phase A: fixed-base bucket MSM            GPU
doubling ladder 2^(8d)·u[j]               GPU
ladder → affine (batched inversion)       GPU
phase B: the fused circulant map          GPU
proofs → affine → compressed              GPU
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

### 3. Everything else that used to want a CPU

Three stages resisted the GPU at first and were briefly run on the host. All
three are back on the device:

- **The doubling ladder** (`2^(8d)·u[j]`, the shared base set for phase B) is
  248 sequential doublings over 128 points. At batch 1 that is 128 threads and
  it runs at latency, not throughput — 19 ms. Batched it is 128 × blobs threads
  and costs ~2.6 ms/blob at batch 8, less beyond. This is the main reason to
  batch.
- **Jacobian → affine** needs a field inversion, which is ~570 multiplies deep
  by Fermat: a fixed ~2 ms of latency however many points you have. Montgomery's
  trick amortises one inversion over a chunk of points, and the chunk size is
  chosen from the point count so the dispatch stays busy on a bigger GPU
  instead of inheriting this machine's shape.
- **Proof compression** (bit-reversal, canonical form, 48-byte encoding) is
  trivial once the points are affine, and writes straight into the output
  buffer.

The result is one command buffer with no host synchronisation inside it.

## Checking the tuning on your own GPU

Everything above was tuned on an 8-core M1, so on a larger part it is worth
confirming rather than trusting. Two tools:

```sh
./build/bench_kzgpu data/trusted_setup.txt 10 128   # batch sweep, blobs/s
./build/profile_stages data/trusted_setup.txt 64 6  # per-stage GPU breakdown
```

`profile_stages` re-runs the real dispatch sequence with the command buffer
flushed at stage boundaries. What to look for on a bigger GPU:

- **`phase A` and `phase B` should dominate** (they were ~70% here). If they do,
  the machine is being used well and the remaining levers are algorithmic.
- **`ladder` growing as a share** means the batch is too small: it is 128
  independent chains per blob, so it needs `batch × 128` to reach saturation.
  Raise the batch before anything else.
- **`normalize ladder` growing** means the inversion chunking is off for that
  core count; `inversionChunk`'s target thread count in `src/kzgpu_metal.mm` is
  the knob.
- **`reduce A`/`reduce B` growing** points at `L_REDUCE_LANES` in
  `src/layout_defs.h` (8 here): more lanes give more threads at the cost of
  wasted SIMD width in the tree.

The MSM window (`L_WINDOW_BITS`, 8) sets the whole shape — digits, buckets and
the 24 MiB table — and was optimal by a clear margin on this part; it is
unlikely to want changing.

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

- **~24 ms is the algorithmic floor** on an 8-core M1 for the current
  formulation (proportionally less on larger GPUs). Real gains from here need
  fewer curve operations, not better scheduling.
- **Phase B's tap count** can drop from 65 to ~43 by recursively splitting the
  cyclic convolution (`X¹²⁸−1 = (X⁶⁴−1)(X⁶⁴+1)`), which costs only additions in
  the ladder domain. Worth ~15% of total; not implemented.
- **GLV** would halve the ladder's depth (248 → 120 doublings) and its memory,
  which matters most at small batch where the ladder is latency-bound. The
  host-side pieces (`glv_split`, the endomorphism) are implemented and tested;
  the phase B kernel does not use them yet.
- **`recover_cells_and_kzg_proofs`** is the natural next entry point for
  supernodes. The expensive half of it is exactly the proof computation this
  library already does; what is missing is the erasure-recovery step.

## License

The trusted setup and test vectors under `data/` and `tests/vectors/` are from
[c-kzg-4844](https://github.com/ethereum/c-kzg-4844) (Apache-2.0).
