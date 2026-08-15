// The trusted setup is compiled into the library, so nothing at runtime can
// catch a transcription error.  This re-parses the canonical trusted_setup.txt
// and compares it byte for byte against the embedded array.
#include "../src/cpu/setup.h"
#include "../src/setup_data.h"

#include <cstdio>
#include <cstring>
#include <vector>

int main(int argc, char **argv) {
    const char *path = argc > 1 ? argv[1] : "data/trusted_setup.txt";

    std::vector<uint8_t> from_file;
    if (kzgpu::read_trusted_setup_file(path, from_file) != KZGPU_OK) {
        printf("FAIL: cannot read %s\n", path);
        return 1;
    }

    if (from_file.size() != kzgpu::kEmbeddedSetupSize) {
        printf("FAIL: embedded setup is %zu bytes, file has %zu\n", kzgpu::kEmbeddedSetupSize,
               from_file.size());
        return 1;
    }
    if (memcmp(from_file.data(), kzgpu::kEmbeddedSetupG1Monomial, from_file.size()) != 0) {
        for (size_t i = 0; i < from_file.size(); i++) {
            if (from_file[i] != kzgpu::kEmbeddedSetupG1Monomial[i]) {
                printf("FAIL: first difference at byte %zu (point %zu): file %02x, embedded %02x\n",
                       i, i / 48, from_file[i], kzgpu::kEmbeddedSetupG1Monomial[i]);
                break;
            }
        }
        return 1;
    }

    // And that every embedded point actually decompresses onto the curve, in
    // the right subgroup -- the property the whole scheme rests on.
    kzgpu::SetupTables probe;
    int bad = 0;
    for (int i = 0; i < KZGPU_NUM_SETUP_G1_POINTS; i++) {
        kzgpu::G1Affine a;
        const uint8_t *p = kzgpu::kEmbeddedSetupG1Monomial + (size_t)i * KZGPU_BYTES_PER_G1;
        if (!kzgpu::g1_decompress(a, p) || !kzgpu::g1_affine_is_on_curve(a) ||
            !kzgpu::g1_affine_in_subgroup(a)) {
            if (bad == 0) printf("FAIL: point %d is not a valid G1 element\n", i);
            bad++;
        }
    }
    if (bad) {
        printf("FAIL: %d invalid points\n", bad);
        return 1;
    }

    printf("ok: embedded setup matches %s (%zu bytes, %d points on curve and in subgroup)\n", path,
           kzgpu::kEmbeddedSetupSize, KZGPU_NUM_SETUP_G1_POINTS);
    return 0;
}
