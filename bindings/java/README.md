# vkzg Java bindings

Java bindings via the JDK's Foreign Function & Memory API (no JNI glue code,
no build tool beyond `javac`). Requires JDK 22+. Shaped like c-kzg-4844's
Java bindings ([jc-kzg-4844](https://github.com/ConsenSys/jc-kzg-4844)): a
global trusted setup loaded once, `byte[]` in and out.

## Build

The native library must be shared, not static:

```sh
cmake -B build -DBUILD_SHARED_LIBS=ON
cmake --build build -j
```

Then compile the bindings:

```sh
cd bindings/java
javac -d out vkzg/Vkzg.java Example.java
```

## Run

```sh
java --enable-native-access=ALL-UNNAMED -Djava.library.path=../../build -cp out Example
```

`-Djava.library.path` (or the `LD_LIBRARY_PATH` environment variable) must
point at the directory containing `libvkzg.so`. Alternatively, pass the
`.so` path directly: `-Dvkzg.library.path=/path/to/libvkzg.so`.

## Usage

```java
import vkzg.Vkzg;

Vkzg.loadTrustedSetup();
byte[] proofs = Vkzg.computeCellKzgProofs(blob);       // one blob
byte[] proofs = Vkzg.computeCellKzgProofs(blobs, n);   // batched, much cheaper per blob
Vkzg.freeTrustedSetup();
```
