# Vulkan KZG (vkzg)

The `vkzg` library uses the GPU (via Vulkan) to compute KZG cell proofs for
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

## Bindings

* [Java](bindings/java)

## Benchmarks

```
device: Apple M1 (G13G B1) (8 GPU cores)

  blobs:   1   best:     98.32 ms   avg:     98.72 ms   per blob:     98.32 ms
  blobs:   2   best:    136.48 ms   avg:    137.47 ms   per blob:     68.24 ms
  blobs:   4   best:    219.73 ms   avg:    221.26 ms   per blob:     54.93 ms
  blobs:   8   best:    400.06 ms   avg:    403.77 ms   per blob:     50.01 ms
  blobs:  16   best:    741.31 ms   avg:    749.57 ms   per blob:     46.33 ms
  blobs:  32   best:   1461.88 ms   avg:   1466.01 ms   per blob:     45.68 ms
  blobs:  64   best:   2875.55 ms   avg:   2884.90 ms   per blob:     44.93 ms
  blobs: 128   best:   5685.34 ms   avg:   5702.31 ms   per blob:     44.42 ms
  blobs: 256   best:  11289.23 ms   avg:  11322.79 ms   per blob:     44.10 ms
```

```
device: Apple M1 Ultra (G13D C0) (64 GPU cores)

  blobs:   1   best:    126.85 ms   avg:    127.59 ms   per blob:    126.85 ms
  blobs:   2   best:    129.48 ms   avg:    129.89 ms   per blob:     64.74 ms
  blobs:   4   best:    145.53 ms   avg:    146.12 ms   per blob:     36.38 ms
  blobs:   8   best:    168.02 ms   avg:    169.17 ms   per blob:     21.00 ms
  blobs:  16   best:    224.56 ms   avg:    225.46 ms   per blob:     14.04 ms
  blobs:  32   best:    350.98 ms   avg:    352.63 ms   per blob:     10.97 ms
  blobs:  64   best:    624.05 ms   avg:    626.99 ms   per blob:      9.75 ms
  blobs: 128   best:   1180.80 ms   avg:   1184.38 ms   per blob:      9.23 ms
  blobs: 256   best:   2272.53 ms   avg:   2280.45 ms   per blob:      8.88 ms
```
