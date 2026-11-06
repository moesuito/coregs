// Written from the TASK-026 specification (docs/hw-renderer-spec), chapters 11.2, 11.3.
// Clean room (decision 0069). Copyright (c) 2026 GT4Recomp contributors, MIT.
#include "hw_gs_replace/xxh3.hpp"

#include <stdexcept>

#include "xxhash.h"

namespace hw_gs_replace {

uint64_t Xxh3Hash64(std::span<const uint8_t> bytes) {
  return static_cast<uint64_t>(XXH3_64bits(bytes.data(), bytes.size()));
}

Xxh3Stream::Xxh3Stream() : state_(XXH3_createState()) {
  if (state_ == nullptr) {
    throw std::runtime_error("XXH3_createState failed");
  }
  if (XXH3_64bits_reset(static_cast<XXH3_state_t*>(state_)) == XXH_ERROR) {
    XXH3_freeState(static_cast<XXH3_state_t*>(state_));
    throw std::runtime_error("XXH3_64bits_reset failed");
  }
}

Xxh3Stream::~Xxh3Stream() { XXH3_freeState(static_cast<XXH3_state_t*>(state_)); }

void Xxh3Stream::Update(std::span<const uint8_t> bytes) {
  if (bytes.empty()) {
    return;
  }
  if (XXH3_64bits_update(static_cast<XXH3_state_t*>(state_), bytes.data(), bytes.size()) ==
      XXH_ERROR) {
    throw std::runtime_error("XXH3_64bits_update failed");
  }
}

uint64_t Xxh3Stream::Digest() const {
  return static_cast<uint64_t>(XXH3_64bits_digest(static_cast<const XXH3_state_t*>(state_)));
}

}  // namespace hw_gs_replace
