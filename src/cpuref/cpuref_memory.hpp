// SPDX-License-Identifier: Apache-2.0
//
// HWR CPU reference: local memory sink. Implements the front end's TransferSink on top of the
// M2 shadow local memory (memory::LocalMemory), so transfers land in the same SLM the
// rasterizer reads and writes.
// Copyright 2026 GT4Recomp contributors.
// Written from the TASK-026 specification (docs/hw-renderer-spec): chapters 02.7, 10.2.
#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

#include "hw_memory.hpp"
#include "hw_transfers.hpp"

namespace coregs::cpuref {

class LocalMemorySink final : public TransferSink {
public:
    LocalMemorySink();

    memory::LocalMemory& Local() { return local_; }
    const memory::LocalMemory& Local() const { return local_; }

    // TransferSink
    void InvalidateDestination(const UploadCommit& commit) override;
    void CommitUpload(const UploadCommit& commit, const uint8_t* bytes, size_t byteCount) override;
    void LocalToLocalMove(const LocalMoveCommit& move) override;
    void BeginLocalToHost(const LocalToHostSetup& setup) override;
    void CancelLocalToHost() override { ResetRead(); }

    // Performs the pending LOCAL->HOST read and returns its host bytes (empty when none is
    // pending or the format is illegal). The pending read is consumed.
    std::vector<uint8_t> TakeLocalToHostData();
    std::vector<uint8_t> ReadLocalToHost(size_t bytes, bool& overrun);
    void ResetRead();
    bool ReadDrained() const { return !pending_ && readCursor_ == readData_.size(); }
    bool LocalToHostPending() const { return pending_.has_value(); }

    // Counters for the replay summary.
    uint64_t uploadCount = 0;
    uint64_t moveCount = 0;
    uint64_t readCount = 0;

private:
    memory::LocalMemory local_;
    std::optional<LocalToHostSetup> pending_;
    std::vector<uint8_t> readData_;
    size_t readCursor_ = 0;
};

}  // namespace coregs::cpuref
