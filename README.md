# vkzg

`vkzg` generates EIP-7594 cell KZG proofs for Ethereum blobs entirely on the
GPU, using Vulkan compute. Given a blob, it produces all 128 cell proofs; it
does not compute the cells themselves (cheap on a CPU) or verify proofs. This
matters for supernodes, which need to prove blobs continuously and are
throughput-bound rather than latency-bound — batching many blobs per call
keeps the GPU saturated and gets far more proofs per second than one blob at
a time. It's tested on Apple Silicon under Asahi Linux (Mesa's Honeykrisp
Vulkan driver) but targets any Vulkan 1.2 device.

## Quick start

```sh
cmake -B build && cmake --build build -j
ctest --test-dir build --output-on-failure
./build/example
```

Needs a Vulkan 1.2+ loader and `glslangValidator` (e.g. on Fedora:
`dnf install vulkan-loader-devel vulkan-headers glslang`).

```c
#include "vkzg.h"

vkzg_options opts;
vkzg_options_default(&opts);
opts.max_batch_size = 16;

vkzg_prover *prover;
vkzg_prover_new_default(&prover, &opts);   /* mainnet trusted setup is compiled in */

vkzg_compute_proofs(prover, proofs, blob);              /* one blob */
vkzg_compute_proofs_batch(prover, proofs, blobs, n);    /* batched, much cheaper per blob */
```

The whole API is in [`include/vkzg.h`](include/vkzg.h), plain C, thread-safe.

## Benchmarks

Time to prove N blobs, batched in one call:

| blobs | Apple M1 (8 GPU cores) | Apple M1 Ultra (64 GPU cores) |
|---|---|---|
| 1 | 98 ms | 129 ms |
| 64 | 2.88 s (45 ms/blob) | 0.63 s (9.8 ms/blob) |

The Ultra is slower at batch 1 (not enough parallel work to spread across
its many cores) but ~4.6x faster at batch 64, so batch what you can.

## License

The embedded trusted setup (`src/setup_data.cpp`) and the test vectors under
`tests/vectors/` come from
[c-kzg-4844](https://github.com/ethereum/c-kzg-4844) (Apache-2.0).
