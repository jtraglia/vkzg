# Vulkan KZG (vkzg)

The `vkzg` library uses your GPU (via Vulkan) to provide really fast functions
for recovering cells and computing proofs for Ethereum blobs. These are time
sensitive operations performed by supernodes in the network. Currently, it only
supports Apple M-series systems running Asahi Linux. In theory, it could
support a wide variety of different GPUs.

## Functions

* `recoverCellsBatch` - Recover cells for multiple blobs in batch.
* `computeProofsBatch` - Compute proofs for multiple blobs in batch.

## Prerequisites

Vulkan 1.2+ loader and `glslangValidator`:

```
dnf install vulkan-loader-devel vulkan-headers glslang
```

## Build

```sh
cmake -B build
cmake --build build -j
./build/bench
```

## Bindings

* [Java](bindings/java)

## Benchmarks

```
device: Apple M1 (G13G B1) (8 GPU cores)

Compute cell KZG proofs for N blobs:

  blobs:   1   best:     98.83 ms   avg:    100.38 ms   per blob:     98.83 ms
  blobs:   2   best:    136.99 ms   avg:    138.02 ms   per blob:     68.49 ms
  blobs:   4   best:    219.07 ms   avg:    220.04 ms   per blob:     54.77 ms
  blobs:   8   best:    400.59 ms   avg:    401.50 ms   per blob:     50.07 ms
  blobs:  16   best:    740.77 ms   avg:    750.56 ms   per blob:     46.30 ms
  blobs:  32   best:   1465.60 ms   avg:   1468.64 ms   per blob:     45.80 ms
  blobs:  64   best:   2881.35 ms   avg:   2891.69 ms   per blob:     45.02 ms
  blobs: 128   best:   5705.95 ms   avg:   5714.26 ms   per blob:     44.58 ms
  blobs: 256   best:  11301.47 ms   avg:  11346.52 ms   per blob:     44.15 ms

Recover cells for N blobs given 50% of cells:

  blobs:   1   best:     18.21 ms   avg:     18.38 ms   per blob:     18.21 ms
  blobs:   2   best:     28.15 ms   avg:     28.34 ms   per blob:     14.07 ms
  blobs:   4   best:     47.86 ms   avg:     48.10 ms   per blob:     11.96 ms
  blobs:   8   best:     86.46 ms   avg:     86.78 ms   per blob:     10.81 ms
  blobs:  16   best:    164.04 ms   avg:    165.52 ms   per blob:     10.25 ms
  blobs:  32   best:    320.33 ms   avg:    321.25 ms   per blob:     10.01 ms
  blobs:  64   best:    634.92 ms   avg:    638.30 ms   per blob:      9.92 ms
  blobs: 128   best:   1264.71 ms   avg:   1271.02 ms   per blob:      9.88 ms
  blobs: 256   best:   2535.67 ms   avg:   2543.28 ms   per blob:      9.90 ms
```

```
device: Apple M1 Ultra (G13D C0) (64 GPU cores)

Compute cell KZG proofs for N blobs:

  blobs:   1   best:    127.45 ms   avg:    128.00 ms   per blob:    127.45 ms
  blobs:   2   best:    129.63 ms   avg:    130.32 ms   per blob:     64.82 ms
  blobs:   4   best:    146.45 ms   avg:    146.66 ms   per blob:     36.61 ms
  blobs:   8   best:    168.96 ms   avg:    169.76 ms   per blob:     21.12 ms
  blobs:  16   best:    225.05 ms   avg:    225.62 ms   per blob:     14.07 ms
  blobs:  32   best:    351.38 ms   avg:    353.07 ms   per blob:     10.98 ms
  blobs:  64   best:    629.04 ms   avg:    630.65 ms   per blob:      9.83 ms
  blobs: 128   best:   1187.15 ms   avg:   1190.95 ms   per blob:      9.27 ms
  blobs: 256   best:   2287.76 ms   avg:   2290.32 ms   per blob:      8.94 ms

Recover cells for N blobs given 50% of cells:

  blobs:   1   best:     10.87 ms   avg:     11.12 ms   per blob:     10.87 ms
  blobs:   2   best:     11.01 ms   avg:     11.31 ms   per blob:      5.51 ms
  blobs:   4   best:     13.59 ms   avg:     13.96 ms   per blob:      3.40 ms
  blobs:   8   best:     18.63 ms   avg:     18.78 ms   per blob:      2.33 ms
  blobs:  16   best:     30.61 ms   avg:     31.40 ms   per blob:      1.91 ms
  blobs:  32   best:     48.11 ms   avg:     48.59 ms   per blob:      1.50 ms
  blobs:  64   best:     87.06 ms   avg:     87.99 ms   per blob:      1.36 ms
  blobs: 128   best:    165.33 ms   avg:    166.94 ms   per blob:      1.29 ms
  blobs: 256   best:    323.84 ms   avg:    329.30 ms   per blob:      1.26 ms
```
