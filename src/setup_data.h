#pragma once

#include <cstddef>
#include <cstdint>

namespace vkzg {

/*
 * Monomial-form G1 trusted setup: 4096 compressed points, 48 bytes each.
 *
 * These are the Ethereum mainnet KZG ceremony values and are fixed for the
 * lifetime of the protocol, so they are compiled in rather than loaded.
 * sha256 of the bytes below: 08797579f6cfd5788eddc1a215d64dcfabd04acbcaf2953fb2c1afb830f43315
 */
constexpr size_t kEmbeddedSetupSize = 196608;
extern const uint8_t kEmbeddedSetupG1Monomial[kEmbeddedSetupSize];

} // namespace vkzg
