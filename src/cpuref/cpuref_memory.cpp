// SPDX-License-Identifier: MIT
// Copyright 2026 GT4Recomp contributors.
// Written from the TASK-026 specification (docs/hw-renderer-spec): chapters 02.7, 10.2.
#include "cpuref_memory.hpp"

#include <algorithm>

namespace coregs::cpuref {

namespace {

// TRXPOS.DIR (two bits) to the memory model's direction enum.
memory::TransferDirection DirectionOf(uint32_t dir) {
    return static_cast<memory::TransferDirection>(dir & 3u);
}

}  // namespace

LocalMemorySink::LocalMemorySink() = default;

void LocalMemorySink::InvalidateDestination(const UploadCommit&) {
    // The CPU reference has no caches to drop: the SLM is always the truth.
}

void LocalMemorySink::CommitUpload(const UploadCommit& commit, const uint8_t* bytes, size_t byteCount) {
    const auto psm = memory::decodePsm(commit.dpsm);
    if (!psm) {
        return;  // invalid PSM: the transfer is dropped (counted by the front end's rules)
    }
    const memory::BufferView destination{commit.dbp, commit.dbw, *psm};
    const memory::TransferRect rect{commit.dsax, commit.dsay, commit.rrw, commit.rrh};
    memory::hostToLocal(local_, destination, rect, std::span<const uint8_t>(bytes, byteCount));
    ++uploadCount;
}

void LocalMemorySink::LocalToLocalMove(const LocalMoveCommit& move) {
    const auto sourcePsm = memory::decodePsm(move.spsm);
    const auto destinationPsm = memory::decodePsm(move.dpsm);
    if (!sourcePsm || !destinationPsm) {
        return;
    }
    memory::localToLocal(local_,
                         memory::BufferView{move.sbp, move.sbw, *sourcePsm}, move.ssax, move.ssay,
                         memory::BufferView{move.dbp, move.dbw, *destinationPsm}, move.dsax, move.dsay,
                         move.rrw, move.rrh, DirectionOf(move.dir));
    ++moveCount;
}

void LocalMemorySink::BeginLocalToHost(const LocalToHostSetup& setup) {
    ResetRead();
    pending_ = setup;
    ++readCount;
}

void LocalMemorySink::ResetRead() {
    pending_.reset();
    readData_.clear();
    readCursor_ = 0;
}

std::vector<uint8_t> LocalMemorySink::ReadLocalToHost(size_t bytes, bool& overrun) {
    if (pending_) {
        readData_ = TakeLocalToHostData();
        readCursor_ = 0;
    }
    const size_t available = readData_.size() - readCursor_;
    const size_t copied = std::min(bytes, available);
    overrun = copied < bytes;
    std::vector<uint8_t> result(bytes, 0);
    std::copy_n(readData_.begin() + static_cast<std::ptrdiff_t>(readCursor_), copied, result.begin());
    readCursor_ += copied;
    return result;
}

std::vector<uint8_t> LocalMemorySink::TakeLocalToHostData() {
    std::vector<uint8_t> data;
    if (!pending_) {
        return data;
    }
    const LocalToHostSetup setup = *pending_;
    pending_.reset();
    const auto psm = memory::decodePsm(setup.spsm);
    if (!psm) {
        return data;
    }
    const memory::BufferView source{setup.sbp, setup.sbw, *psm};
    const memory::TransferRect rect{setup.ssax, setup.ssay, setup.rrw, setup.rrh};
    memory::localToHost(local_, source, rect, data);  // 4-bit sources are refused (data stays empty)
    return data;
}

}  // namespace coregs::cpuref
