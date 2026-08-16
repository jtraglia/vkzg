# vkzg Java bindings

Java bindings via the JDK's Foreign Function & Memory API. Requires JDK 22+.

## Install

```sh
cmake -B build -DBUILD_SHARED_LIBS=ON && cmake --build build -j
javac -d bindings/java/out bindings/java/vkzg/Vkzg.java
```

## Example

```java
import vkzg.Vkzg;

Vkzg.init();
byte[] proofs = Vkzg.computeCellKzgProofs(blob);       // one blob
byte[] proofs = Vkzg.computeCellKzgProofs(blobs, n);   // batched, much cheaper per blob
Vkzg.deinit();
```

```sh
java --enable-native-access=ALL-UNNAMED -Djava.library.path=build -cp bindings/java/out YourProgram
```
