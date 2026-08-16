# vkzg Java bindings

Java bindings via the JDK's Foreign Function & Memory API. Requires JDK 22+.

## Install

```xml
<repository>
    <id>jitpack.io</id>
    <url>https://jitpack.io</url>
</repository>

<dependency>
    <groupId>com.github.jtraglia</groupId>
    <artifactId>vkzg</artifactId>
    <version>v0.1.0</version>
</dependency>
```

```groovy
repositories { maven { url 'https://jitpack.io' } }
dependencies { implementation 'com.github.jtraglia:vkzg:v0.1.0' }
```

The native library isn't published with the jar; build it separately and
point the JVM at it:

```sh
cmake -B build -DBUILD_SHARED_LIBS=ON && cmake --build build -j
```

## Example

```java
import vkzg.Vkzg;

Vkzg.init();
byte[] proofs = Vkzg.computeCellKzgProofs(blobs, blob_count);
Vkzg.deinit();
```

```sh
java --enable-native-access=ALL-UNNAMED -Djava.library.path=build -cp vkzg.jar YourProgram
```
