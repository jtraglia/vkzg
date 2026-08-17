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
    <version>v0.1.9</version>
</dependency>
```

```groovy
repositories { maven { url 'https://jitpack.io' } }
dependencies { implementation 'com.github.jtraglia:vkzg:v0.1.9' }
```

## Example

```java
import vkzg.Vkzg;

Vkzg.init();
byte[] proofs = Vkzg.computeCellKzgProofsBatch(blobs, blob_count);
Vkzg.deinit();
```

```sh
java --enable-native-access=ALL-UNNAMED -cp vkzg.jar YourProgram
```
