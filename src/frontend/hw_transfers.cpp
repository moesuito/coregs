// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 GT4Recomp contributors.
// Written from the TASK-026 specification (docs/hw-renderer-spec) (clean room).
#include "hw_transfers.hpp"

#include <algorithm>
#include <utility>

#include "hw_gs_types.hpp"

namespace coregs {

size_t HostToLocalByteCount(uint32_t rrw, uint32_t rrh, uint32_t bitsPerPixel) {
    const uint64_t bits = static_cast<uint64_t>(rrw) * rrh * bitsPerPixel;
    const uint64_t bytes = (bits + 7u) / 8u;
    return static_cast<size_t>(std::min<uint64_t>(bytes, kTransferCapBytes));
}

TransferUnit::TransferUnit(TransferSink& sink, TransferHooks hooks, CounterSet& counters)
    : sink_(sink), hooks_(std::move(hooks)), counters_(counters) {}

void TransferUnit::WriteBitbltbuf(uint64_t value) {
    if (value != regs_.bitbltbuf) {
        CommitPending(TransferReason::RegisterChange);
    }
    regs_.bitbltbuf = value;
}

void TransferUnit::WriteTrxpos(uint64_t value) {
    if (value != regs_.trxpos) {
        CommitPending(TransferReason::RegisterChange);
    }
    regs_.trxpos = value;
}

void TransferUnit::WriteTrxreg(uint64_t value) {
    if (value != regs_.trxreg) {
        CommitPending(TransferReason::RegisterChange);
    }
    regs_.trxreg = value;
}

void TransferUnit::WriteTrxdir(uint64_t value) {
    CommitPending(TransferReason::TrxdirWrite);
    sink_.CancelLocalToHost();
    active_ = false;
    buffer_.clear();
    regs_.xdir = static_cast<uint32_t>(value & 3u);

    const uint32_t dbp = static_cast<uint32_t>(BitRange(regs_.bitbltbuf, 32, 45));
    const uint32_t dbw = static_cast<uint32_t>(BitRange(regs_.bitbltbuf, 48, 53));
    const uint32_t dpsm = static_cast<uint32_t>(BitRange(regs_.bitbltbuf, 56, 61));
    const uint32_t sbp = static_cast<uint32_t>(BitRange(regs_.bitbltbuf, 0, 13));
    const uint32_t sbw = static_cast<uint32_t>(BitRange(regs_.bitbltbuf, 16, 21));
    const uint32_t spsm = static_cast<uint32_t>(BitRange(regs_.bitbltbuf, 24, 29));
    const uint32_t ssax = static_cast<uint32_t>(BitRange(regs_.trxpos, 0, 10));
    const uint32_t ssay = static_cast<uint32_t>(BitRange(regs_.trxpos, 16, 26));
    const uint32_t dsax = static_cast<uint32_t>(BitRange(regs_.trxpos, 32, 42));
    const uint32_t dsay = static_cast<uint32_t>(BitRange(regs_.trxpos, 48, 58));
    const uint32_t dir = static_cast<uint32_t>(BitRange(regs_.trxpos, 59, 60));
    const uint32_t rrw = static_cast<uint32_t>(BitRange(regs_.trxreg, 0, 11));
    const uint32_t rrh = static_cast<uint32_t>(BitRange(regs_.trxreg, 32, 43));

    switch (regs_.xdir) {
        case 0: {  // host to local: arm and wait for data
            const uint32_t bpp = TransferBitsPerPixel(dpsm);
            if (bpp == 0u) {
                counters_.Add(Counter::InvalidPsm);
                return;
            }
            pending_ = UploadCommit{};
            pending_.dbp = dbp;
            pending_.dbw = dbw;
            pending_.dpsm = dpsm;
            pending_.dsax = dsax;
            pending_.dsay = dsay;
            pending_.rrw = rrw;
            pending_.rrh = rrh;
            pending_.bitsPerPixel = bpp;
            totalBytes_ = HostToLocalByteCount(rrw, rrh, bpp);
            active_ = totalBytes_ > 0u;
            buffer_.reserve(totalBytes_);
            break;
        }
        case 1: {  // local to host: the read path (chapter 10) takes over
            LocalToHostSetup setup;
            setup.sbp = sbp;
            setup.sbw = sbw;
            setup.spsm = spsm;
            setup.ssax = ssax;
            setup.ssay = ssay;
            setup.rrw = rrw;
            setup.rrh = rrh;
            setup.bitsPerPixel = TransferBitsPerPixel(spsm);
            if (hooks_.beforeRead) {
                hooks_.beforeRead(setup);
            }
            sink_.BeginLocalToHost(setup);
            break;
        }
        case 2: {  // local to local: overlap check, then the move
            LocalMoveCommit move;
            move.sbp = sbp;
            move.sbw = sbw;
            move.spsm = spsm;
            move.ssax = ssax;
            move.ssay = ssay;
            move.dbp = dbp;
            move.dbw = dbw;
            move.dpsm = dpsm;
            move.dsax = dsax;
            move.dsay = dsay;
            move.rrw = rrw;
            move.rrh = rrh;
            move.dir = dir;
            if (hooks_.beforeMove) {
                hooks_.beforeMove(move);
            }
            sink_.LocalToLocalMove(move);
            break;
        }
        default:
            break;
    }
}

void TransferUnit::AcceptData(const uint8_t* bytes, size_t count) {
    if (!active_ || regs_.xdir != 0u) {
        counters_.Add(Counter::ImageDiscarded);
        return;
    }
    const size_t room = totalBytes_ - buffer_.size();
    const size_t take = std::min(room, count);
    buffer_.insert(buffer_.end(), bytes, bytes + take);
    if (buffer_.size() == totalBytes_) {
        Commit(true, TransferReason::Complete);
    }
}

size_t TransferUnit::AcceptImageQwords(const uint8_t* bytes, size_t qwords) {
    if (!active_ || regs_.xdir != 0u) {
        counters_.Add(Counter::ImageDiscarded);
        return 1;
    }
    const size_t room = totalBytes_ - buffer_.size();
    const size_t take = std::min(room, qwords * 16);
    if (buffer_.empty() && take == totalBytes_) {
        // IN-365: the whole transfer is here: commit it from the packet bytes (no copy into buffer_).
        CommitFrom(bytes, take, true);
        return std::min(std::max<size_t>(1, (take + 15) / 16), qwords);
    }
    buffer_.insert(buffer_.end(), bytes, bytes + take);
    const size_t used = std::max<size_t>(1, (take + 15) / 16);
    if (buffer_.size() == totalBytes_) {
        Commit(true, TransferReason::Complete);
    }
    return std::min(used, qwords);
}

void TransferUnit::CommitPending(TransferReason reason) {
    if (active_ && !buffer_.empty()) {
        Commit(false, reason);
    }
}

void TransferUnit::Commit(bool complete, TransferReason reason) {
    (void)reason;
    CommitFrom(buffer_.data(), buffer_.size(), complete);
}

void TransferUnit::CommitFrom(const uint8_t* data, size_t byteCount, bool complete) {
    UploadCommit commit = pending_;

    if (complete) {
        commit.rrh = pending_.rrh;
        commit.partial = false;
    } else {
        // Partial commit: rows = ceil(pixels / RRW), at most the rows left.
        const uint64_t pixels = (static_cast<uint64_t>(byteCount) * 8u) / pending_.bitsPerPixel;
        uint64_t rows = pixels;
        if (pending_.rrw > 0u) {
            rows = (pixels + pending_.rrw - 1u) / pending_.rrw;
        }
        commit.rrh = static_cast<uint32_t>(std::min<uint64_t>(rows, pending_.rrh));
        commit.partial = true;
    }

    if (hooks_.beforeUpload) {
        hooks_.beforeUpload(commit);
    }
    sink_.InvalidateDestination(commit);
    sink_.CommitUpload(commit, data, byteCount);
    if (hooks_.afterUpload) {
        hooks_.afterUpload(commit);
    }
    counters_.Add(Counter::TransferCommits);

    buffer_.clear();
    if (complete) {
        active_ = false;
        totalBytes_ = 0;
        return;
    }
    pending_.dsay += commit.rrh;
    pending_.rrh -= commit.rrh;
    totalBytes_ = HostToLocalByteCount(pending_.rrw, pending_.rrh, pending_.bitsPerPixel);
    if (pending_.rrh == 0u || totalBytes_ == 0u) {
        active_ = false;
        totalBytes_ = 0;
    }
}

void TransferUnit::Reset() {
    sink_.CancelLocalToHost();
    active_ = false;
    totalBytes_ = 0;
    buffer_.clear();
    regs_ = TransferRegisters{};
}

void TransferUnit::ForceTrxdirIdle() {
    sink_.CancelLocalToHost();
    active_ = false;
    totalBytes_ = 0;
    buffer_.clear();
    regs_.xdir = 3;
}

}  // namespace coregs
