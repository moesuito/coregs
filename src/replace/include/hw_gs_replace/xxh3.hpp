// XXH3-64 with seed 0 (chapter 11.2 / 11.3): one-shot and streaming. Streaming equals one-shot
// over the concatenated bytes (chapter 11.3). Written from the TASK-026 specification.
// Copyright (c) 2026 GT4Recomp contributors, MIT. Hash implementation: xxHash 0.8.3 (BSD-2).
#pragma once

#include <cstddef>
#include <cstdint>
#include <span>

namespace hw_gs_replace {

uint64_t Xxh3Hash64(std::span<const uint8_t> bytes);

class Xxh3Stream {
 public:
  Xxh3Stream();
  ~Xxh3Stream();
  Xxh3Stream(const Xxh3Stream&) = delete;
  Xxh3Stream& operator=(const Xxh3Stream&) = delete;

  void Update(std::span<const uint8_t> bytes);
  // Digest of everything fed so far. Does not change the stream; more data may follow.
  uint64_t Digest() const;

 private:
  void* state_;
};

}  // namespace hw_gs_replace
