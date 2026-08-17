package vkzg;

import java.io.IOException;
import java.io.InputStream;
import java.io.UncheckedIOException;
import java.lang.foreign.Arena;
import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.Linker;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.SymbolLookup;
import java.lang.foreign.ValueLayout;
import java.lang.invoke.MethodHandle;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;

/**
 * Java bindings for vkzg (EIP-7594 cell KZG proof generation on the GPU),
 * via the JDK's Foreign Function &amp; Memory API -- no JNI glue code needed.
 *
 * A global GPU prover is opened once with {@link #init()} and closed with
 * {@link #deinit()}; {@link #computeCellKzgProofsBatch(byte[], long)} takes a
 * batch of blobs in one call, since batching for GPU throughput is the
 * whole point.
 *
 * The native library is bundled in the jar and loaded automatically; no
 * separate build step is needed. Only Apple Silicon under Linux (Vulkan
 * via the Mesa Honeykrisp driver) is supported.
 */
public final class Vkzg {
    private Vkzg() {}

    public static final int FIELD_ELEMENTS_PER_BLOB = 4096;
    public static final int BYTES_PER_FIELD_ELEMENT = 32;
    public static final int BYTES_PER_BLOB = FIELD_ELEMENTS_PER_BLOB * BYTES_PER_FIELD_ELEMENT;
    public static final int CELL_PROOFS_PER_BLOB = 128;
    public static final int BYTES_PER_PROOF = 48;
    public static final int FIELD_ELEMENTS_PER_CELL = 64;
    public static final int BYTES_PER_CELL = FIELD_ELEMENTS_PER_CELL * BYTES_PER_FIELD_ELEMENT;

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
        try (InputStream in = Vkzg.class.getResourceAsStream("/native/libvkzg.so")) {
            if (in == null) {
                throw new UnsatisfiedLinkError("bundled native/libvkzg.so resource not found in jar");
            }
            Path tmp = Files.createTempFile("libvkzg", ".so");
            tmp.toFile().deleteOnExit();
            Files.copy(in, tmp, StandardCopyOption.REPLACE_EXISTING);
            System.load(tmp.toAbsolutePath().toString());
        } catch (IOException e) {
            throw new UncheckedIOException("failed to extract bundled libvkzg.so", e);
        }
        extractPrecomputedTables();
        return SymbolLookup.loaderLookup();
    }

    /**
     * The native library reads its precomputed FK20/position tables from a
     * path baked in at compile time (a build-tree or install-tree location);
     * neither exists for a jar downloaded onto an arbitrary machine, so
     * extract the bundled copy and point the library at it via
     * $VKZG_TABLES_PATH, which it checks first.
     */
    private static void extractPrecomputedTables() {
        try (InputStream in = Vkzg.class.getResourceAsStream("/native/precomputed_tables.bin")) {
            if (in == null) {
                throw new UnsatisfiedLinkError(
                        "bundled native/precomputed_tables.bin resource not found in jar");
            }
            Path tmp = Files.createTempFile("vkzg_precomputed_tables", ".bin");
            tmp.toFile().deleteOnExit();
            Files.copy(in, tmp, StandardCopyOption.REPLACE_EXISTING);
            setenv("VKZG_TABLES_PATH", tmp.toAbsolutePath().toString());
        } catch (IOException e) {
            throw new UncheckedIOException("failed to extract bundled precomputed_tables.bin", e);
        }
    }

    private static void setenv(String name, String value) {
        MemorySegment sym = LINKER.defaultLookup().find("setenv").orElseThrow(
                () -> new UnsatisfiedLinkError("symbol not found: setenv"));
        MethodHandle setenv = LINKER.downcallHandle(sym,
                FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS, ValueLayout.ADDRESS,
                        ValueLayout.JAVA_INT));
        try (Arena arena = Arena.ofConfined()) {
            int rc = (int) setenv.invokeExact(arena.allocateFrom(name), arena.allocateFrom(value), 1);
            if (rc != 0) {
                throw new UnsatisfiedLinkError("setenv(" + name + ") failed");
            }
        } catch (Throwable t) {
            if (t instanceof UnsatisfiedLinkError e) throw e;
            throw new AssertionError(t);
        }
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
    private static final MethodHandle COMPUTE_PROOFS_BATCH = handle("vkzg_compute_proofs_batch",
            FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS, ValueLayout.ADDRESS,
                    ValueLayout.ADDRESS, ValueLayout.JAVA_LONG));
    private static final MethodHandle RECOVER_CELLS_BATCH = handle("vkzg_recover_cells_batch",
            FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS, ValueLayout.ADDRESS,
                    ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.JAVA_LONG));

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
     * how many blobs a single {@link #computeCellKzgProofsBatch(byte[], long)}
     * call keeps in flight on the GPU at once (larger is faster per blob,
     * at ~5.6 MiB of GPU memory each); 0 selects a sensible default. Larger
     * batches passed to computeCellKzgProofsBatch are chunked transparently.
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

    /**
     * Computes all 128 cell proofs for each of {@code blobCount} consecutive
     * blobs packed into {@code blobs} (pass 1 for a single blob). Returns
     * the proofs, concatenated in the same order (128 * 48 bytes per blob).
     * Batching is markedly more efficient per blob than repeated
     * single-blob calls -- pass as many blobs as you have.
     */
    public static synchronized byte[] computeCellKzgProofsBatch(byte[] blobs, long blobCount) {
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

            int rc = (int) COMPUTE_PROOFS_BATCH.invokeExact(proverHandle, proofsSeg, blobsSeg,
                    blobCount);
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

    /**
     * Recovers all 128 cells for each of {@code blobCount} consecutive blobs,
     * given at least half of each blob's cells.
     *
     * @param cells       {@code blobCount} consecutive {@code CELL_PROOFS_PER_BLOB
     *                    * BYTES_PER_CELL}-byte arrays: a blob's full 128-cell
     *                    extended array, in column order. A missing cell's
     *                    bytes are ignored.
     * @param cellPresent {@code blobCount} consecutive {@code CELL_PROOFS_PER_BLOB}
     *                    -byte arrays, one byte per cell (nonzero = present).
     *                    At least 64 of 128 must be present per blob.
     * @return every cell, recovered: {@code blobCount} consecutive {@code
     *     CELL_PROOFS_PER_BLOB * BYTES_PER_CELL} byte arrays.
     */
    public static synchronized byte[] recoverCellsBatch(byte[] cells, byte[] cellPresent,
            long blobCount) {
        checkLoaded();
        final long cellsPerBlobBytes = (long) CELL_PROOFS_PER_BLOB * BYTES_PER_CELL;
        if (cells.length != blobCount * cellsPerBlobBytes) {
            throw new IllegalArgumentException(
                    "cells.length must be blobCount * " + cellsPerBlobBytes);
        }
        if (cellPresent.length != blobCount * CELL_PROOFS_PER_BLOB) {
            throw new IllegalArgumentException(
                    "cellPresent.length must be blobCount * " + CELL_PROOFS_PER_BLOB);
        }
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment cellsSeg = arena.allocate(cells.length);
            MemorySegment.copy(cells, 0, cellsSeg, ValueLayout.JAVA_BYTE, 0, cells.length);
            MemorySegment presentSeg = arena.allocate(cellPresent.length);
            MemorySegment.copy(cellPresent, 0, presentSeg, ValueLayout.JAVA_BYTE, 0, cellPresent.length);
            MemorySegment outSeg = arena.allocate(cells.length);

            int rc = (int) RECOVER_CELLS_BATCH.invokeExact(proverHandle, outSeg, cellsSeg,
                    presentSeg, blobCount);
            if (rc != 0) {
                throw new VkzgException(rc, errorString(rc));
            }

            byte[] out = new byte[cells.length];
            MemorySegment.copy(outSeg, ValueLayout.JAVA_BYTE, 0, out, 0, out.length);
            return out;
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
