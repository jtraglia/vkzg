package vkzg;

// Latency and throughput benchmark for the GPU prover -- the Java analog of
// bench/bench.cpp. Run with:
//   mvn test-compile
//   java --enable-native-access=ALL-UNNAMED -cp target/classes:target/test-classes vkzg.Bench
public class Bench {
    private static final int REPS = 10;
    private static final int MAX_BATCH = 256;

    public static void main(String[] args) {
        // Set to the highest blob count in the sweep, so every measurement below
        // exercises a single, real dispatch rather than being silently chunked.
        Vkzg.init(MAX_BATCH);
        System.out.printf("device: %s (%d GPU cores)%n%n", Vkzg.deviceName(), Vkzg.gpuCoreCount());

        System.out.println("Compute all cell KZG proofs for N blobs:\n");
        byte[] blobs = new byte[MAX_BATCH * Vkzg.BYTES_PER_BLOB];
        for (int i = 0; i < MAX_BATCH; i++) {
            fillBlob(blobs, i * Vkzg.BYTES_PER_BLOB, i + 1);
        }
        for (int n = 1; n <= MAX_BATCH; n *= 2) {
            runComputeProofs(blobs, n);
        }

        System.out.println("\nGiven 50% of cells for N blobs, recover the other half:\n");
        final int cellsPerBlobBytes = Vkzg.CELL_PROOFS_PER_BLOB * Vkzg.BYTES_PER_CELL;
        byte[] cells = new byte[MAX_BATCH * cellsPerBlobBytes];
        byte[] present = new byte[MAX_BATCH * Vkzg.CELL_PROOFS_PER_BLOB];
        for (int i = 0; i < MAX_BATCH; i++) {
            fillCells(cells, i * cellsPerBlobBytes, i + 1);
            // First 64 of 128 cells present, the rest missing -- a genuine
            // 50%-missing scenario (not the easier "every other cell" case).
            int base = i * Vkzg.CELL_PROOFS_PER_BLOB;
            for (int c = 0; c < 64; c++) {
                present[base + c] = 1;
            }
        }
        for (int n = 1; n <= MAX_BATCH; n *= 2) {
            runRecoverCells(cells, present, n);
        }

        Vkzg.deinit();
    }

    private static void runComputeProofs(byte[] blobs, int n) {
        byte[] slice = java.util.Arrays.copyOf(blobs, n * Vkzg.BYTES_PER_BLOB);
        Vkzg.computeCellKzgProofsBatch(slice, n); // warm up
        double best = Double.MAX_VALUE, total = 0;
        for (int r = 0; r < REPS; r++) {
            long start = System.nanoTime();
            Vkzg.computeCellKzgProofsBatch(slice, n);
            double ms = (System.nanoTime() - start) / 1_000_000.0;
            best = Math.min(best, ms);
            total += ms;
        }
        printRow(n, best, total / REPS);
    }

    private static void runRecoverCells(byte[] cells, byte[] present, int n) {
        final int cellsPerBlobBytes = Vkzg.CELL_PROOFS_PER_BLOB * Vkzg.BYTES_PER_CELL;
        byte[] cellsSlice = java.util.Arrays.copyOf(cells, n * cellsPerBlobBytes);
        byte[] presentSlice = java.util.Arrays.copyOf(present, n * Vkzg.CELL_PROOFS_PER_BLOB);
        Vkzg.recoverCellsBatch(cellsSlice, presentSlice, n); // warm up
        double best = Double.MAX_VALUE, total = 0;
        for (int r = 0; r < REPS; r++) {
            long start = System.nanoTime();
            Vkzg.recoverCellsBatch(cellsSlice, presentSlice, n);
            double ms = (System.nanoTime() - start) / 1_000_000.0;
            best = Math.min(best, ms);
            total += ms;
        }
        printRow(n, best, total / REPS);
    }

    private static void printRow(int n, double best, double avg) {
        System.out.printf(
                "  blobs: %3d   best: %9.2f ms   avg: %9.2f ms   per blob: %9.2f ms%n",
                n, best, avg, best / n);
    }

    // Deterministic pseudo-random canonical blob.
    private static void fillBlob(byte[] out, int offset, long seed) {
        long s = seed * 0x9E3779B97F4A7C15L + 1;
        for (int i = 0; i < Vkzg.FIELD_ELEMENTS_PER_BLOB; i++) {
            int feOff = offset + i * Vkzg.BYTES_PER_FIELD_ELEMENT;
            for (int j = 0; j < Vkzg.BYTES_PER_FIELD_ELEMENT; j++) {
                s = xorshift(s);
                out[feOff + j] = (byte) (s >>> 24);
            }
            // The modulus r's top byte is 0x73 (0b0111_0011); clearing just
            // its top two bits (rather than the whole byte) is enough to
            // guarantee every element is below r regardless of the
            // remaining bytes, while keeping 6 bits of real entropy.
            out[feOff] &= 0x3F;
        }
    }

    // Deterministic pseudo-random canonical full 128-cell extended array
    // (the same shape recoverCellsBatch expects: cell recovery never sees
    // an actual blob, only cells). Real, non-zero data matters here -- an
    // all-zero array would make every scalar multiply in the pipeline
    // degenerate and give misleadingly fast timings.
    private static void fillCells(byte[] out, int offset, long seed) {
        long s = seed * 0x9E3779B97F4A7C15L + 1;
        final int totalElements = Vkzg.CELL_PROOFS_PER_BLOB * Vkzg.FIELD_ELEMENTS_PER_CELL;
        for (int i = 0; i < totalElements; i++) {
            int feOff = offset + i * Vkzg.BYTES_PER_FIELD_ELEMENT;
            for (int j = 0; j < Vkzg.BYTES_PER_FIELD_ELEMENT; j++) {
                s = xorshift(s);
                out[feOff + j] = (byte) (s >>> 24);
            }
            out[feOff] &= 0x3F;
        }
    }

    private static long xorshift(long s) {
        s ^= s << 13;
        s ^= s >>> 7;
        s ^= s << 17;
        return s;
    }
}
