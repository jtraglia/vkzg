# vulkan-prover

EIP-7594 cell KZG proof generation for Ethereum blobs, on the GPU via Vulkan
compute.

Given a blob, `vulkan-prover` produces all 128 cells and all 128 cell proofs.
It is built for supernodes, which generate proofs constantly and care about
throughput. Proof *verification* is deliberately out of scope — this library
only produces.

Verified against every `compute_cells_and_kzg_proofs` vector in the consensus
spec test suite, cells and proofs byte-for-byte.

This project started as `metal-prover`, targeting Apple's Metal API on
macOS. It has since been rewritten for Linux: the whole GPU pipeline (NTTs,
both MSM phases, the doubling ladder, batched inversion, proof compression)
now runs as Vulkan 1.2 compute shaders, with buffers passed to kernels as raw
GPU addresses (`VK_KHR_buffer_device_address`) via push constants rather than
descriptor sets — the closest Vulkan analogue of Metal's `device T *`
arguments, chosen so the kernels could stay close in shape to the originals.
Nothing here is Apple-specific: any Vulkan 1.2 device exposing
`bufferDeviceAddress`, `shaderInt64`/`shaderInt16`/`shaderInt8`, 8/16-bit
storage buffers and subgroup shuffle should work. It has only been tested on
one: an Apple M1 running Asahi Linux, through Mesa's "Honeykrisp" driver
(Vulkan 1.4 conformant).

## Results

Everything runs on the GPU. One command buffer per call, with a full
pipeline barrier between dispatches; the host only copies blobs in and
cells/proofs out.

Apple M1, 8 GPU cores, Fedora Asahi Remix, Mesa 26.1.6 (Honeykrisp):

| | ms/blob | blobs/s |
|---|---|---|
| batch 1 | 111.2 | 9.0 |
| batch 8 | 63.7 | 15.7 |
| batch 16 | 62.1 | 16.1 |
| batch 32 | 60.8 | 16.5 |
| **batch 64** | **60.1** | **16.6** |

Cells alone (no proofs) are 0.83 ms/blob.

**This is not the Metal number for the same silicon**, and the two aren't
directly comparable: the old `metal-prover` measured 46.8 ms/blob (21.4
blobs/s) at batch 64 on the same M1 through Metal. Some of the gap is a
Vulkan tax that's inherent to this design (a `vkCmdPipelineBarrier` after
every dispatch, where Metal's single-encoder hazard tracking inserted
narrower barriers automatically), and some of it is very likely tuning that
carried over unexamined from the Metal version rather than being re-derived
for this driver — see [Checking the tuning](#checking-the-tuning-on-your-own-gpu)
below. Nobody has gone through that exercise for Vulkan/Honeykrisp yet; this
is the correctness-first port, not the tuned one.

### Batching

Batch at least 8. Two stages need it: the doubling ladder is 128 independent
chains per blob (128 × batch threads), and the batched inversions amortise a
multi-millisecond field inversion across their chunk. Below batch 8 both are
latency-bound and you pay for it — batch 1 costs nearly 2× batch 64's
per-blob time.

Working set is **5.6 MiB per blob** plus a fixed ~28 MiB (the 24 MiB FK20
table and the root tables): 384 MiB at batch 64.

## Quick start

```sh
cmake -B build && cmake --build build -j
ctest --test-dir build --output-on-failure
./build/example
```

Building needs a Vulkan 1.2+ loader and headers, and `glslangValidator` to
compile the GLSL kernels to SPIR-V at build time (e.g. on Fedora:
`dnf install vulkan-loader-devel vulkan-headers glslang`).

Nothing to download or configure at runtime: the trusted setup is compiled
in, and so is the compiled SPIR-V.

```c
#include "vulkan_prover.h"

vkp_options opts;
vkp_options_default(&opts);
opts.table_cache_path = "/var/cache/vulkan-prover/tables.bin";  /* seconds to build, ~60ms to load */
opts.max_batch_size = 16;

vkp_prover *prover;
vkp_prover_new_default(&prover, &opts);   /* mainnet setup is compiled in */

/* one blob */
vkp_compute_cells_and_proofs(prover, cells, proofs, blob);

/* or a batch, which is markedly more efficient per blob */
vkp_compute_cells_and_proofs_batch(prover, cells, proofs, blobs, n);
```

The whole API is in [`include/vulkan_prover.h`](include/vulkan_prover.h) and is plain C, so
Rust / Go / Java bindings need no C++ shim. A `vkp_prover` is safe to share
between threads; calls serialise internally on the GPU queue.

### The trusted setup

The mainnet ceremony's monomial G1 points are **compiled into the library** —
4096 compressed points, 192 KiB, in `src/setup_data.cpp`. The Lagrange G1
points are for commitments and the G2 points for verification, neither of which
this library does, so neither is carried. There is no file to ship, locate or
validate at runtime, and `vkp_prover_new_default()` is all a caller needs.

Provenance is checkable: the generated header records the sha256 of the bytes
(`08797579f6cfd578…`), and `tools/embed_setup.py` regenerates them from a
canonical `trusted_setup.txt` so anyone can confirm the blob matches the
ceremony output. The whole test suite runs against the embedded setup, so the
spec vectors passing is itself evidence the points are right.

`vkp_prover_new()` takes raw setup bytes if you ever need to target a
different ceremony.

Deriving the FK20 tables from the setup takes a couple of seconds on this
machine; `table_cache_path` brings that down to well under 100 ms on
subsequent starts. The cache stores a digest of the setup and is rebuilt
silently if it does not match. (This step is pure host-side CPU work,
unchanged from the Metal version — it isn't part of the GPU port.)

## How it works, and why it looks like this

The shape of this library's *algorithm* is unchanged from the Metal original
and is the result of two measurements taken before any of it was written, on
that Metal/M1 combination. The reasoning is almost certainly still directionally
right on Apple's GPU architecture generally (it's about the hardware, not the
API), but the specific rates below were measured under Metal and have not
been independently re-measured under Vulkan.

### 1. The GPU's integer multiplier is the scarce resource

BLS12-381 arithmetic is 381-bit modular multiplication, which on a GPU means
32×32→64 integer multiplies. On Apple7 (M1), Metal measured one of those
issuing about every **8.9 cycles per lane** — Apple's GPUs are built for
float throughput, not integer multiplies. Measured ceilings (Metal):

| | rate |
|---|---|
| `(ulong)a * b` (widening 32×32) | 147 G/s |
| F_p multiply (12 limbs, CIOS) | 392 M/s |
| (for scale: one M1 CPU core does 33 M/s) | |
| F_p add / sub | 4.0 G/s |
| G1 mixed addition | 22.2 M/s |

FK20 for one blob is ~528k mixed additions, so **~24 ms was the Metal floor**
for this algorithm on an 8-core GPU even at perfect efficiency.

### 2. Single-thread latency has a hard floor

One F_p multiply took **3.43 µs** for one thread under Metal and never got
faster, no matter how idle the GPU was — throughput saturates at ~2048
resident threads, below which you are simply paying latency.

This is fatal for the textbook FK20. c-kzg computes the proofs as
`IFFT → truncate → FFT` over G1, and each butterfly of those size-128
transforms multiplies a G1 point by a root of unity: a ~255-doubling
dependency chain, ≈2000 F_p multiplies deep, several milliseconds of pure
latency *each*. Seven levels of that, twice, dominates c-kzg's runtime — the
single worst-mapping part of the algorithm.

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

Three things about the MSM turned out to matter more than expected on Metal,
and the code still reflects them (unverified under Vulkan so far):

- **Bucket loads are Poisson(16), and a SIMD group waits for its slowest lane.**
  Randomly assigned, all four groups pay the global tail. Sorting buckets by
  descending load first, so only the first group sees the tail, was worth
  **1.24×** on Metal.
- **Where the bucket reduction lives.** As a standalone kernel it needed too
  much threadgroup memory for partial points to keep occupancy up; fused into
  the MSM kernel it was *worse*, because register allocation is per-kernel. A
  pure shuffle tree was worse still, because every lane executes every level.
  The shape that works: `L_REDUCE_LANES` lanes (4, in `src/layout_defs.h`)
  cooperate on one output, each first collapsing 32 buckets serially with a
  running sum, then a short subgroup-shuffle tree across just those 4 lanes.
  This is implemented with Vulkan subgroup ops (`subgroupShuffleDown`) in
  `src/shaders/field.glsl.inc`'s `reduce_buckets`, matching Metal's
  `simd_shuffle_down` one for one — both rely on a 32-wide hardware SIMD/
  subgroup, which Apple's GPU exposes identically under either API
  (`subgroupSize = 32`, confirmed via `vulkaninfo` on this driver).
- **The ladder belongs off the fast path.** 248 sequential doublings over 128
  points is almost pure latency, which is why batching (not more GPU cores)
  is what fixes it — see Batching above.

### 3. Everything runs on the device

- **The doubling ladder** (`2^(8d)·u[j]`, the shared base set for phase B) is
  248 sequential doublings over 128 points. At batch 1 that is 128 threads and
  it runs at latency, not throughput. Batched, it is 128 × blobs threads. This
  is the main reason to batch.
- **Jacobian → affine** needs a field inversion, which is ~570 multiplies deep
  by Fermat: a fixed multi-millisecond latency however many points there are.
  Montgomery's trick amortises one inversion over a chunk of points, and the
  chunk size (`inversionChunk` in `src/vulkan_prover.cpp`) is chosen from the
  point count so the dispatch stays busy on a bigger GPU instead of
  inheriting this machine's shape.
- **Proof compression** (bit-reversal, canonical form, 48-byte encoding) is
  trivial once the points are affine, and writes straight into the output
  buffer.

The result is one command buffer with no host synchronisation inside it —
just GPU-side pipeline barriers between dispatches, since Vulkan (unlike
Metal) does not track buffer hazards automatically.

## Checking the tuning on your own GPU

The Metal version's tuning notes above have not been re-validated for
Vulkan/Honeykrisp, and this is the only GPU it's been tested on at all. Two
tools:

```sh
./build/bench 10 128          # batch sweep, blobs/s
./build/profile_stages 64 6   # per-stage GPU breakdown, via timestamp queries
```

On this M1 via Vulkan, the per-stage shape at batch 64 looks different from
what the Metal build reported (phase A + phase B ~70% there): here, phase A
is ~43% and `normalize ladder` is ~37%, with `reduce A` also a larger share
than expected. That's a strong hint that at least one of the knobs below
wants re-tuning for this backend rather than inheriting the Metal-tuned
values — this hasn't been investigated yet:

- **`normalize ladder` (or `normalize proofs`) growing** means the inversion
  chunking is off for this core count; `inversionChunk`'s target thread count
  in `src/vulkan_prover.cpp` is the knob. Unlike Metal, `k_normalize.comp`'s
  workgroup size is fixed at compile time (64) rather than chosen per
  dispatch, since Vulkan doesn't allow a pipeline's local size to vary at
  dispatch time — that fixed-size tradeoff is a candidate explanation worth
  checking before assuming it's purely the chunk-size math.
- **`ladder` growing as a share** means the batch is too small: it is 128
  independent chains per blob, so it needs `batch × 128` to reach saturation.
  Raise the batch before anything else.
- **`reduce A`/`reduce B` growing** points at `L_REDUCE_LANES` in
  `src/layout_defs.h` (4 here): more lanes give more threads at the cost of
  wasted subgroup width in the tree.
- **The pipeline barrier after every dispatch** (`barrier()` in
  `src/vulkan_prover.cpp`) is deliberately conservative — a full
  read/write memory barrier on every stage boundary, rather than the minimal
  set Metal's automatic hazard tracking would have inferred. Narrowing these
  to only the buffers each stage actually depends on is unexplored and could
  recover some of the gap to the Metal numbers.

The MSM window (`L_WINDOW_BITS`, 8) sets the whole shape — digits, buckets and
the 24 MiB table — and was optimal by a clear margin on Metal; it is a
reasonable starting assumption here too, but untested.

## Layout

```
include/vulkan_prover.h public C API (vkp_* / VKP_*)
src/layout_defs.h       sizes shared by host and shaders (single source of truth)
src/setup_data.{h,cpp}  the mainnet trusted setup, generated by tools/embed_setup.py
src/shaders/            GLSL compute: field.glsl.inc (F_p/F_r/G1), k_*.comp (one file per kernel)
src/cpu/                host BLS12-381, trusted setup and FK20 table derivation
src/cpu/reference.cpp   CPU implementation of the exact same pipeline, for tests
src/vulkan_prover.cpp   Vulkan host layer and dispatch scheduling
tools/                  constant, shader and trusted-setup generators
tests/                  unit tests, spec vectors, GPU end-to-end
bench/                  latency/throughput benchmark, per-stage profiler
```

Kernels are compiled from GLSL to SPIR-V by `glslangValidator` at *build*
time (`tools/embed_shaders.py`) and the resulting words are embedded in a
generated header, loaded into `VkShaderModule`s at prover-creation time — no
`.spv` files to locate at runtime, and (unlike the Metal version, which
compiled MSL source at load time) no shader compilation happens at runtime
at all.

Buffers are never bound through descriptor sets: every kernel receives its
buffers as `VkDeviceAddress` values via push constants, and dereferences them
in GLSL through `buffer_reference` pointer types
(`GL_EXT_buffer_reference2`) declared once in `field.glsl.inc`. This needs
`VK_KHR_buffer_device_address`, which is core in Vulkan 1.2 and was chosen
specifically because it let the kernels keep the same buffer-pointer style
as the Metal originals instead of being restructured around bound array
bindings.

Field constants for the host (64-bit limbs) and the device (32-bit limbs) are
emitted from one script (`tools/gen_constants.py`), so the two representations
cannot drift.

## Limits and what would move next

- **Vulkan-specific tuning is unstarted** — see
  [Checking the tuning](#checking-the-tuning-on-your-own-gpu). The gap between
  the 60.1 ms/blob measured here and the Metal build's 46.8 ms/blob on the
  same silicon is the most immediately actionable thing in this repository.
- **Phase B's tap count** can drop from 65 to ~43 by recursively splitting the
  cyclic convolution (`X¹²⁸−1 = (X⁶⁴−1)(X⁶⁴+1)`), which costs only additions in
  the ladder domain. Not implemented.
- **GLV** would halve the ladder's depth (248 → 120 doublings) and its memory,
  which matters most at small batch where the ladder is latency-bound. The
  host-side pieces (`glv_split`, the endomorphism) are implemented and tested;
  the phase B kernel does not use them yet.
- **`recover_cells_and_kzg_proofs`** is the natural next entry point for
  supernodes. The expensive half of it is exactly the proof computation this
  library already does; what is missing is the erasure-recovery step.

## License

The embedded trusted setup (`src/setup_data.cpp`) and the test vectors under
`tests/vectors/` come from
[c-kzg-4844](https://github.com/ethereum/c-kzg-4844) (Apache-2.0).
