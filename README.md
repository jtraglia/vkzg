# vulkan-prover

EIP-7594 cell KZG proof generation for Ethereum blobs, on the GPU via Vulkan
compute.

Given a blob, `vulkan-prover` produces all 128 cell proofs. It is built for
supernodes, which generate proofs constantly and care about throughput.
Computing the cells themselves (as opposed to their proofs) is cheap on a CPU
and deliberately out of scope here, as is proof *verification* — this
library only produces proofs.

Verified against every `compute_cells_and_kzg_proofs` vector in the consensus
spec test suite, proofs byte-for-byte (the same vectors also carry expected
cells, which this library doesn't compute and so doesn't check).

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
proofs out.

Apple M1, 8 GPU cores, Fedora Asahi Remix, Mesa 26.1.6 (Honeykrisp):

| | ms/blob | blobs/s |
|---|---|---|
| batch 1 | 125.4 | 8.0 |
| batch 8 | 50.1 | 20.0 |
| batch 16 | 47.1 | 21.2 |
| batch 32 | 45.9 | 21.8 |
| batch 64 | 45.1 | 22.2 |
| **batch 128** | **44.5** | **22.5** |

**This now beats the original `metal-prover`'s Metal numbers on the same
8-core M1** (46.8 ms/blob, 21.4 blobs/s at batch 64) — not by inherently
being faster than Metal, but because phase B no longer does the amount of
work the Metal version's algorithm did: see
[the split-convolution rewrite](#4-the-128-point-convolution-splits-into-two-64-point-ones)
below. Batch 1 is the one regression (110.6 → 125.4 ms/blob unbatched, since
the split form pays for a handful of extra small dispatches that don't pay
for themselves until there's enough batch to amortise them) — irrelevant for
the high-throughput batched use this library targets, but worth knowing if
you were relying on low-batch latency specifically.

An Apple M1 Ultra (64 GPU cores), same driver stack, on the same real
hardware — not extrapolated, but measured *before* everything in
[How it works](#how-it-works-and-why-it-looks-like-this) below, so treat it
as a shape (how it scales with core count) rather than a number that's still
current:

| | ms/blob | blobs/s |
|---|---|---|
| batch 1 | 119.2 | 8.4 |
| batch 8 | 21.6 | 46.2 |
| batch 16 | 15.0 | 66.7 |
| batch 32 | 12.3 | 81.1 |
| **batch 64** | **11.9** | **83.9** |

8× the GPU cores buys ~5× the throughput at batch 64 (45→12 ms/blob), not
8×: some stages (the ladder, the batched inversions) are latency-bound
rather than throughput-bound, so they don't scale with core count the way
the two MSM phases do. It also means small batches leave far more of a
64-core part idle: batch 1 is 10× slower per blob than batch 64 on the
Ultra, versus under 1.5× on the 8-core M1.

### Batching

Batch at least 8. Two stages need it: the doubling ladder is 128 independent
chains per blob (128 × batch threads), and the batched inversions amortise a
multi-millisecond field inversion across their chunk. Below batch 8 both are
latency-bound and you pay for it — batch 1 costs almost 3× batch 128's
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

/* one blob -> 128 cell proofs */
vkp_compute_proofs(prover, proofs, blob);

/* or a batch, which is markedly more efficient per blob */
vkp_compute_proofs_batch(prover, proofs, blobs, n);
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
64 circulant columns, each NTT-128        GPU
phase A: fixed-base bucket MSM            GPU
doubling ladder 2^(8d)·u[j]               GPU
fold ladder into L+/L- (see below)        GPU
ladder → affine (batched inversion)       GPU
phase B+: cyclic half of the circulant    GPU
phase B-: negacyclic half of the circulant GPU
combine: out[a] = C+[a] ± C-[a]           GPU
proofs → affine → compressed              GPU
```

(An earlier version of this pipeline also computed the 128 cells themselves
— a forward NTT of size 8192 plus a serialisation kernel. That's gone: this
library only produces proofs, and cell computation is cheap enough on a CPU
that it didn't belong in a GPU-focused library.)

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

### 4. The 128-point convolution splits into two 64-point ones

Phase B's fused kernel (section 2 above) is a 128-point cyclic convolution
with 65 non-zero taps. `X¹²⁸−1` factors as `(X⁶⁴−1)(X⁶⁴+1)`, and since gcd of
those two factors is 2 (invertible mod the scalar field's prime), CRT splits
the whole convolution into two independent 64-point ones: an ordinary cyclic
convolution and a *negacyclic* one (`mod X⁶⁴+1`, meaning a wrap-around also
flips sign).

```
u+[j] = u[j] + u[j+64]                     (fold, j in [0,64))
u-[j] = u[j] - u[j+64]
C+[a] = Σ_e κ+[e]·u+[a-e]                  (ordinary cyclic, mod X^64-1)
C-[a] = Σ_e κ-[e]·u-[a-e]·sign(e,a)        (negacyclic, mod X^64+1)
out[a]    = C+[a] + C-[a]                  (for a in [0,64))
out[a+64] = C+[a] - C-[a]
```

where `κ±[i] = (κ[i] ± κ[i+64]) / 2` (the `/2` folded in so the final combine
is pure addition) and `sign(e,a) = −1` exactly when tap `e` wraps past output
index `a` (`e > a`), `+1` otherwise.

Two things made this worth doing rather than a purely theoretical curiosity:

- **The tap density barely changes, but the outputs-per-kernel halves.**
  Each half kernel (`κ+`, `κ-`) turns out to have the *same* sparsity
  pattern as the original — non-zero at `e=0` and every odd `e` — so 33
  taps each (65 → 33+33), almost a wash on tap count. The saving is that
  each half only has to cover 64 outputs instead of 128: total *tap ×
  output* work (what the bucket-MSM actually pays for) drops from
  `65 × 128 = 8320` to `33 × 64 × 2 = 4224`, a ~49% cut. Verified exactly —
  with real G1 arithmetic, not just the scalar identity — in
  `tests/test_reference.cpp` before any GPU code was touched.
- **The negacyclic half's extra sign flip stays a shared, rotation-only
  table.** The obvious way to handle `mod X⁶⁴+1` (twisting by a root of
  unity into an ordinary cyclic problem) would require *scalar-multiplying*
  every folded ladder point by a twist factor — exactly the expensive
  per-point scalar multiplication this whole design exists to avoid. Instead,
  the wrap sign is just `e > a`, a comparison the kernel can make at
  dispatch time from values it already has (the tap `e` from the
  precomputed item, the output index `a` from its own workgroup ID) — so
  the negacyclic half keeps the exact same "one item list shared by every
  output, only a rotation differs" structure the flat form and the cyclic
  half both have. No per-output tables, no extra host precomputation size.

The fold itself (`u±[j] = u[j] ± u[j+64]`) runs on the already-computed
doubling ladder (`k_fold_ladder.comp`) — cheap point additions, not new
scalar multiplications, matching the "costs only additions" framing. Both
halves reuse the *same* bucket-MSM kernel (`k_phase_b_split.comp`), just
with a push-constant flag selecting cyclic vs. negacyclic and which half of
the folded ladder to read; a final tiny kernel (`k_combine_split.comp`)
does the `C+ ± C-` reconstruction.

Measured on the 8-core M1 (batch 64): phase B's total cost (both bucket-MSM
dispatches, both reductions, the combine step) dropped from ~1716ms to
~1042ms of a batch-64 call — about 39%, short of the ~49% theoretical cut
because the split form pays for three extra dispatches the flat form didn't
need (the fold, a second bucket-MSM, a second reduce). Net effect on total
throughput: ~18% faster at batch ≥ 8 — see [Results](#results).

## Checking the tuning on your own GPU

The Metal version's tuning notes above have not been fully re-validated for
Vulkan/Honeykrisp, and this is the only GPU this has been tested on at all.
Two tools:

```sh
./build/bench 10 128          # batch sweep, blobs/s
./build/profile_stages 64 6   # per-stage breakdown, CPU-flushed between stages
```

**A word of warning if you touch `profile_stages`'s internals: don't switch
it back to GPU timestamp queries (`vkCmdWriteTimestamp` + `VkQueryPool`)
without independently cross-checking the numbers.** That was the first
implementation here, and it looked entirely plausible — consistent-seeming
numbers that added up to the measured wall time — while being wrong. It
reported `normalize ladder` at ~37% of a batch-64 call regardless of how
much work that dispatch actually had (changing its thread count 4× moved
nothing), and which *other* stage's name absorbed the real cost changed
between otherwise-identical runs. Forcing an actual CPU-side drain
(`vkQueueSubmit` + `vkWaitForFences`) between every dispatch and comparing
gave a stable, reproducible breakdown instead — that's what `profile_stages`
does now, at the cost of real submission overhead (~0.1ms/stage) it's
honest about paying. The lesson: on a driver this young, a timestamp-based
profiler is worth distrusting until you've checked it against a slower but
unambiguous method at least once.

With that fixed, the real breakdown at batch 64 on this M1 was (before the
phase B split below existed) phase A ~39%, phase B ~40%, `reduce A`/`reduce
B` ~16% combined, the ladder ~3%, and everything else under 2% — phase A and
phase B (the two bucket MSMs) dominating, as the Metal design expects. Two
concrete things came out of chasing that shape (a third, much bigger one —
splitting phase B's convolution in two — is its own section, see
[How it works](#4-the-128-point-convolution-splits-into-two-64-point-ones)):

- **The ladder's workgroups were needlessly small.** `k_ladder.comp` used a
  workgroup of 8 threads, matching the Metal original — reasonable there, but
  on this driver it left 3/4 of every 32-wide subgroup idle on every
  dispatch. Each thread's doubling chain is fully independent (no shared
  memory, no barriers), so nothing stops widening the workgroup; going to 64
  or 128 (same result either way) cut the ladder's share of a batch-64 call
  from ~369ms to ~98ms — a real ~7% win on total throughput, not a rounding
  error. Worth checking on any kernel whose workgroup size was chosen for a
  32-wide Metal simdgroup without re-examining whether it still makes sense here.
- **`L_REDUCE_LANES` inverts, but isn't a clean win.** Metal's sweep found 4
  lanes optimal, with subgroup-shuffle cooperation still profitable at 2 and
  only breaking even at 1. On this driver, cooperation *never* profits at
  large batch (1 lane beats 4 at batch 64, and the trend is monotonic all the
  way down) — plausible if `subgroupShuffleDown` is simply more expensive
  relative to plain ALU work on this younger shader compiler than on Metal's.
  But fewer lanes also means fewer, fatter threads per reduce dispatch
  (`batch * 128 / L_REDUCE_LANES` total), and at small batch that's not
  enough threads to hide latency: 1 lane is ~45% *slower* than 4 at batch 1.
  `L_REDUCE_LANES` stays at 4 because it's the only setting that doesn't
  regress the batch ≥ 8 range this library recommends; revisit if that
  recommendation changes, or on a different Vulkan implementation. Full sweep
  in the comment above `L_REDUCE_LANES` in `src/layout_defs.h`.

Still open, in roughly the order worth checking next:

- **`inversionChunk`'s target thread count** (in `src/vulkan_prover.cpp`) was
  chosen to match Metal's occupancy shape and hasn't been independently
  re-swept on this driver, even though `normalize_ladder`/`normalize_proofs`
  turned out to be cheap here (~1% combined) so it's a low-priority check.
- **The pipeline barrier after every dispatch** (`barrier()` in
  `src/vulkan_prover.cpp`) is deliberately conservative — a full read/write
  memory barrier on every stage boundary, rather than the minimal set
  Metal's automatic hazard tracking would have inferred. Narrowing these to
  only the buffers each stage actually depends on is unexplored.
- **Other kernels' workgroup sizes** (`k_phase_a_sort.comp` at 64,
  `k_build_circulant.comp`/`k_phase_a.comp`/`k_phase_b_split.comp`/
  `k_bucket_reduce.comp` at 128) inherited Metal's threadgroup sizes the same
  way the ladder did; only the ladder has been checked for the same
  small-workgroup effect so far, and it's the one with by far the smallest
  original size (8) so the others may simply not have room to improve the
  same way.

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

- **Vulkan-specific tuning is partially done** — see
  [Checking the tuning](#checking-the-tuning-on-your-own-gpu) for what's been
  checked (the profiler itself, the ladder's workgroup size, `L_REDUCE_LANES`)
  and what's still open (`inversionChunk`, the per-dispatch barriers, other
  kernels' workgroup sizes, and now also the split form's own three extra
  dispatches — see [How it works](#4-the-128-point-convolution-splits-into-two-64-point-ones)).
  This library now runs *faster* than the original Metal build on the same
  8-core M1, so this is no longer about closing a gap, just further headroom.
- **Recursing the split further** (`X⁶⁴−1 = (X³²−1)(X³²+1)`, and so on) keeps
  halving tap density in the same self-similar way; a back-of-envelope check
  during development (plain field arithmetic, no GPU) found the *tap ×
  output* cost keeps dropping at every level (8320 → 4224 → 2176 → 1152 →
  ...), but each level also adds fixed per-dispatch overhead (another fold,
  another bucket-MSM pair, another combine), so there's a real
  diminishing-returns point past the one level implemented here. Not
  measured where that point is on this driver.
- **GLV** would halve the ladder's depth (248 → 120 doublings) and its memory,
  which matters most at small batch where the ladder is latency-bound. The
  host-side pieces (`glv_split`, the endomorphism) are implemented and tested;
  the phase B kernel does not use them yet.

This library deliberately does not compute cells or do erasure recovery —
both are cheap on a CPU and out of scope for a GPU-focused proof generator.

## License

The embedded trusted setup (`src/setup_data.cpp`) and the test vectors under
`tests/vectors/` come from
[c-kzg-4844](https://github.com/ethereum/c-kzg-4844) (Apache-2.0).
