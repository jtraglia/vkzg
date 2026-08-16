# Vulkan KZG (vkzg)

The `vkzg` library is uses the GPU (via Vulkan) to compute KZG cell proofs for
Ethereum blobs. Currently, it only supports Apple M-series systems running
Asahi Linux. In theory, it could support a wide variety of different GPUs.

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

## Benchmarks

```
device: Apple M1 (G13G B1) (8 GPU cores)   setup: 139 ms

  1 blob    best   98.50 ms   avg   98.92 ms   per blob  98.50 ms
  2 blobs   best  136.74 ms   avg  137.47 ms   per blob  68.37 ms
  4 blobs   best  219.66 ms   avg  220.94 ms   per blob  54.92 ms
  8 blobs   best  400.88 ms   avg  403.60 ms   per blob  50.11 ms
  16 blobs  best  742.50 ms   avg  753.26 ms   per blob  46.41 ms
  32 blobs  best 1463.41 ms   avg 1467.84 ms   per blob  45.73 ms
  64 blobs  best 2875.85 ms   avg 2887.41 ms   per blob  44.94 ms
  128 blobs  best 5690.19 ms   avg 5708.42 ms   per blob  44.45 ms
  256 blobs  best 11298.18 ms   avg 11332.78 ms   per blob  44.13 ms
```

```
device: Apple M1 Ultra (G13D C0) (64 GPU cores)   setup: 455 ms

  1 blob    best  127.04 ms   avg  127.65 ms   per blob 127.04 ms
  2 blobs   best  129.79 ms   avg  130.09 ms   per blob  64.89 ms
  4 blobs   best  145.10 ms   avg  145.79 ms   per blob  36.28 ms
  8 blobs   best  168.29 ms   avg  169.08 ms   per blob  21.04 ms
  16 blobs  best  224.69 ms   avg  225.17 ms   per blob  14.04 ms
  32 blobs  best  351.31 ms   avg  352.70 ms   per blob  10.98 ms
  64 blobs  best  625.22 ms   avg  626.99 ms   per blob   9.77 ms
  128 blobs  best 1181.36 ms   avg 1184.31 ms   per blob   9.23 ms
  256 blobs  best 2271.91 ms   avg 2278.91 ms   per blob   8.87 ms
```
