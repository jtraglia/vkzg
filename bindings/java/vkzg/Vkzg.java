package vkzg;

import java.lang.foreign.Arena;
import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.Linker;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.SymbolLookup;
import java.lang.foreign.ValueLayout;
import java.lang.invoke.MethodHandle;

/**
 * Java bindings for vkzg (EIP-7594 cell KZG proof generation on the GPU),
 * via the JDK's Foreign Function &amp; Memory API -- no JNI glue code needed.
 *
 * Shaped like c-kzg-4844's Java bindings (jc-kzg-4844): a global GPU prover
 * opened once with {@link #init()} and closed with {@link #deinit()}, plain
 * {@code byte[]} in and out. vkzg only produces cell proofs (not cells, not
 * verification), and its whole point is GPU throughput, so
 * {@link #computeCellKzgProofs(byte[], long)} takes a batch of blobs in one
 * call rather than one blob at a time.
 *
 * Needs a shared library: build the C library with -DBUILD_SHARED_LIBS=ON
 * and either put it on java.library.path or point -Dvkzg.library.path at
 * the .so file directly.
 */
public final class Vkzg {
    private Vkzg() {}

    public static final int FIELD_ELEMENTS_PER_BLOB = 4096;
    public static final int BYTES_PER_FIELD_ELEMENT = 32;
    public static final int BYTES_PER_BLOB = FIELD_ELEMENTS_PER_BLOB * BYTES_PER_FIELD_ELEMENT;
    public static final int CELL_PROOFS_PER_BLOB = 128;
    public static final int BYTES_PER_PROOF = 48;

    public static final class VkzgException extends RuntimeException {
        public final int code;

        VkzgException(int code, String message) {
            super(message);
            this.code = code;
        }
    }

    private static final Linker LINKER = Linker.nativeLinker();
    private static final SymbolLookup LOOKUP = loadLibrary();

    private static SymbolLookup loadLibrary() {
        String explicit = System.getProperty("vkzg.library.path");
        if (explicit != null) {
            return SymbolLookup.libraryLookup(explicit, Arena.global());
        }
        // System.loadLibrary honors java.library.path (unlike
        // SymbolLookup.libraryLookup, which only sees LD_LIBRARY_PATH /
        // absolute paths); loaderLookup then finds symbols in it.
        System.loadLibrary("vkzg");
        return SymbolLookup.loaderLookup();
    }

    private static MethodHandle handle(String name, FunctionDescriptor fd) {
        MemorySegment sym = LOOKUP.find(name).orElseThrow(
                () -> new UnsatisfiedLinkError("symbol not found: " + name));
        return LINKER.downcallHandle(sym, fd);
    }

    private static final MethodHandle ERROR_STRING = handle("vkzg_error_string",
            FunctionDescriptor.of(ValueLayout.ADDRESS, ValueLayout.JAVA_INT));
    private static final MethodHandle OPTIONS_DEFAULT = handle("vkzg_options_default",
            FunctionDescriptor.ofVoid(ValueLayout.ADDRESS));
    private static final MethodHandle PROVER_NEW = handle("vkzg_prover_new",
            FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS, ValueLayout.ADDRESS));
    private static final MethodHandle PROVER_FREE = handle("vkzg_prover_free",
            FunctionDescriptor.ofVoid(ValueLayout.ADDRESS));
    private static final MethodHandle PROVER_DEVICE_NAME = handle("vkzg_prover_device_name",
            FunctionDescriptor.of(ValueLayout.ADDRESS, ValueLayout.ADDRESS));
    private static final MethodHandle PROVER_GPU_CORE_COUNT = handle("vkzg_prover_gpu_core_count",
            FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS));
    private static final MethodHandle COMPUTE_PROOFS = handle("vkzg_compute_proofs",
            FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS, ValueLayout.ADDRESS,
                    ValueLayout.ADDRESS, ValueLayout.JAVA_LONG));

    // sizeof(vkzg_options): one uint32_t (max_batch_size).
    private static final long OPTIONS_SIZE = 4;

    private static String errorString(int code) {
        try {
            MemorySegment s = (MemorySegment) ERROR_STRING.invokeExact(code);
            return s.reinterpret(Long.MAX_VALUE).getString(0);
        } catch (Throwable t) {
            throw new AssertionError(t);
        }
    }

    private static MemorySegment proverHandle;

    /** Opens the GPU prover with a sensible default batch capacity. */
    public static synchronized void init() {
        init(0);
    }

    /**
     * Opens the GPU prover: picks a Vulkan device, compiles the shader
     * kernels, and allocates its GPU buffers. {@code maxBatchSize} bounds
     * how many blobs a single {@link #computeCellKzgProofs(byte[], long)}
     * call keeps in flight on the GPU at once (larger is faster per blob,
     * at ~5.6 MiB of GPU memory each); 0 selects a sensible default. Larger
     * batches passed to computeCellKzgProofs are chunked transparently.
     */
    public static synchronized void init(long maxBatchSize) {
        if (proverHandle != null) {
            throw new IllegalStateException("already initialized");
        }
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment opts = arena.allocate(OPTIONS_SIZE);
            OPTIONS_DEFAULT.invokeExact(opts);
            if (maxBatchSize != 0) {
                opts.set(ValueLayout.JAVA_INT, 0, (int) maxBatchSize);
            }
            MemorySegment outPtr = arena.allocate(ValueLayout.ADDRESS);
            int rc = (int) PROVER_NEW.invokeExact(outPtr, opts);
            if (rc != 0) {
                throw new VkzgException(rc, errorString(rc));
            }
            proverHandle = outPtr.get(ValueLayout.ADDRESS, 0);
        } catch (Throwable t) {
            if (t instanceof VkzgException e) throw e;
            throw new AssertionError(t);
        }
    }

    /** Closes the GPU prover, freeing its GPU memory. Safe to {@link #init()} again afterwards. */
    public static synchronized void deinit() {
        checkLoaded();
        try {
            PROVER_FREE.invokeExact(proverHandle);
        } catch (Throwable t) {
            throw new AssertionError(t);
        }
        proverHandle = null;
    }

    /** Computes all 128 cell proofs for one blob. */
    public static byte[] computeCellKzgProofs(byte[] blob) {
        return computeCellKzgProofs(blob, 1);
    }

    /**
     * Computes all 128 cell proofs for each of {@code blobCount} consecutive
     * blobs packed into {@code blobs}. Returns the proofs, concatenated in
     * the same order (128 * 48 bytes per blob). Batching is markedly more
     * efficient per blob than repeated single-blob calls -- pass as many
     * blobs as you have.
     */
    public static synchronized byte[] computeCellKzgProofs(byte[] blobs, long blobCount) {
        checkLoaded();
        if (blobs.length != blobCount * BYTES_PER_BLOB) {
            throw new IllegalArgumentException(
                    "blobs.length must be blobCount * " + BYTES_PER_BLOB);
        }
        final long proofsLen = blobCount * CELL_PROOFS_PER_BLOB * BYTES_PER_PROOF;
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment blobsSeg = arena.allocate(blobs.length);
            MemorySegment.copy(blobs, 0, blobsSeg, ValueLayout.JAVA_BYTE, 0, blobs.length);
            MemorySegment proofsSeg = arena.allocate(proofsLen);

            int rc = (int) COMPUTE_PROOFS.invokeExact(proverHandle, proofsSeg, blobsSeg, blobCount);
            if (rc != 0) {
                throw new VkzgException(rc, errorString(rc));
            }

            byte[] proofs = new byte[(int) proofsLen];
            MemorySegment.copy(proofsSeg, ValueLayout.JAVA_BYTE, 0, proofs, 0, proofs.length);
            return proofs;
        } catch (Throwable t) {
            if (t instanceof VkzgException e) throw e;
            throw new AssertionError(t);
        }
    }

    /** Name of the Vulkan device in use, e.g. "Apple M1 (G13G B1)". */
    public static synchronized String deviceName() {
        checkLoaded();
        try {
            MemorySegment s = (MemorySegment) PROVER_DEVICE_NAME.invokeExact(proverHandle);
            return s.reinterpret(Long.MAX_VALUE).getString(0);
        } catch (Throwable t) {
            throw new AssertionError(t);
        }
    }

    /** GPU core count, or 0 if it can't be determined on this driver. */
    public static synchronized long gpuCoreCount() {
        checkLoaded();
        try {
            return Integer.toUnsignedLong((int) PROVER_GPU_CORE_COUNT.invokeExact(proverHandle));
        } catch (Throwable t) {
            throw new AssertionError(t);
        }
    }

    private static void checkLoaded() {
        if (proverHandle == null) {
            throw new IllegalStateException("not initialized; call init() first");
        }
    }
}
