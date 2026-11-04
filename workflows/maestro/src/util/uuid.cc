#include "util/uuid.hh"

#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <random>
#include <string>

namespace LWM {
  std::string uuid_v7() {
    const auto now { std::chrono::system_clock::now().time_since_epoch() };

    using namespace std::literals;
    const std::uint64_t ms { static_cast<std::uint64_t>(now / 1ms) };

    // Gives 32-bits entropy per call (Linux at least.)
    std::random_device rd;
    const std::uint64_t r_hi { (std::uint64_t(rd()) << 32) | rd() };
    const std::uint32_t r_lo { rd() };

    // Construct a UUIDv7 identifier from timestamp and 2 random numbers
    // 48-bit timestamp (big endian)
    // v7 + 4 random bits (random #1)
    // 8 bits random #1
    // random #2

    std::array<std::uint8_t, 16> bytes {};
    for (int i = 0; i < 6; ++i) {
      bytes[i] = std::uint8_t(ms >> (40 - 8 * i));
    }

    bytes[6] = std::uint8_t(0x70 | ((r_lo >> 8) & 0x0F));
    bytes[7] = std::uint8_t(r_lo & 0xFF);
    for (int i = 0; i < 8; ++i) {
      bytes[8 + i] = std::uint8_t(r_hi >> (56 - 8 * i));
    }
    bytes[8] = std::uint8_t(0x80 | (bytes[8] & 0x3F));

    char out[37];
    std::snprintf(out,
                  sizeof(out),
                  "%02x%02x%02x%02x-%02x%02x-%02x%02x-%02x%02x-%02x%02x%02x%02x%02x%02x",
                  bytes[0],
                  bytes[1],
                  bytes[2],
                  bytes[3],
                  bytes[4],
                  bytes[5],
                  bytes[6],
                  bytes[7],
                  bytes[8],
                  bytes[9],
                  bytes[10],
                  bytes[11],
                  bytes[12],
                  bytes[13],
                  bytes[14],
                  bytes[15]);

    return out;
  }
} // namespace LWM
