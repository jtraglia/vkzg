import vkzg.Vkzg;

/**
 * Minimal end-to-end example, mirroring examples/example.c's C version.
 *
 * Build:  javac -d out vkzg/Vkzg.java Example.java
 * Run:    java --enable-native-access=ALL-UNNAMED -Djava.library.path=../../build -cp out Example
 */
public class Example {
    public static void main(String[] args) {
        Vkzg.init();
        try {
            System.out.println("prover ready on " + Vkzg.deviceName());

            // A blob is 4096 big-endian field elements, each below the
            // BLS12-381 scalar field modulus. Zeroing the top byte
            // guarantees that, matching examples/example.c's fixture.
            byte[] blob = new byte[Vkzg.BYTES_PER_BLOB];
            for (int i = 0; i < Vkzg.FIELD_ELEMENTS_PER_BLOB; i++) {
                int off = i * Vkzg.BYTES_PER_FIELD_ELEMENT;
                for (int j = 0; j < Vkzg.BYTES_PER_FIELD_ELEMENT; j++) {
                    blob[off + j] = (byte) ((i * 31 + j * 7 + 11) & 0xff);
                }
                blob[off] = 0;
            }

            byte[] proofs = Vkzg.computeCellKzgProofs(blob);
            System.out.println("produced " + Vkzg.CELL_PROOFS_PER_BLOB + " cell proofs");
            StringBuilder hex = new StringBuilder();
            for (int i = 0; i < Vkzg.BYTES_PER_PROOF; i++) {
                hex.append(String.format("%02x", proofs[i]));
            }
            System.out.println("proof[0] = " + hex);
        } finally {
            Vkzg.deinit();
        }
    }
}
