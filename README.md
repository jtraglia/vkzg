# kzgpu

EIP-7594 cell KZG proof generation for Ethereum blobs, on Apple GPUs.

Given a blob, `kzgpu` produces all 128 cells and all 128 cell proofs. It is
built for supernodes, which generate proofs constantly and care about
throughput. Proof *verification* is deliberately out of scope — this library
only produces.

Verified against every `compute_cells_and_kzg_proofs` vector in the consensus
spec test suite, cells and proofs byte-for-byte.

## Results

Apple M1 (8 GPU cores, 4P+4E CPU cores), macOS 27, one blob:

| | per blob | vs baseline |
|---|---|---|
| c-kzg-4844, single thread, no precompute | 251.9 ms | — |
| c-kzg-4844, single thread, `precompute=8` | 168.5 ms | 1.0× |
| **kzgpu, GPU only, single blob** | **70.0 ms** | 2.4× |
| **kzgpu, GPU only, batch of 8** | **51.2 ms** | 3.3× |
| **kzgpu, GPU + CPU, single blob** | **49.1 ms** | 3.4× |
| **kzgpu, GPU + CPU, batch of 8** | **33.6 ms** | **5.0×** |

Cells alone (no proofs) are 0.47 ms against c-kzg's 2.5 ms.

Batching is what supernodes should use: it keeps the GPU saturated and lets the
CPU assist overlap cleanly. Throughput flattens at about 30 blobs/s from a
batch of 8 onwards.

All numbers are best-of-N on an idle passively-cooled MacBook Air; expect a
Pro/Max to scale close to linearly with GPU core count.

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
- **A faster host F_p multiply.** Ours is 43.6 ns of portable C++ against blst's
  ~20 ns of assembly. Since the CPU now carries roughly half the work, closing
  that gap is worth ~20% overall.
- **`recover_cells_and_kzg_proofs`** is the natural next entry point for
  supernodes. The expensive half of it is exactly the proof computation this
  library already does; what is missing is the erasure-recovery step.

## License

The trusted setup and test vectors under `data/` and `tests/vectors/` are from
[c-kzg-4844](https://github.com/ethereum/c-kzg-4844) (Apache-2.0).
