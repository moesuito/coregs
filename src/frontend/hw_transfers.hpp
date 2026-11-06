// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 GT4Recomp contributors.
// HWR transfer bookkeeping (chapter 02.7) and the sink interface the memory
// model (M2, written in parallel) implements.
// Written from the TASK-026 specification (docs/hw-renderer-spec) (clean room).
#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <vector>

#include "hw_gs_counters.hpp"

namespace coregs {

// Destination rectangle of a host->local upload, in buffer pixels.
struct UploadCommit {
    uint32_t dbp = 0;
    uint32_t dbw = 0;
    uint32_t dpsm = 0;
    uint32_t dsax = 0;
    uint32_t dsay = 0;
    uint32_t rrw = 0;
    uint32_t rrh = 0;            // rows covered by this commit
    uint32_t bitsPerPixel = 0;
    bool partial = false;        // true when the transfer was not complete
};

struct LocalMoveCommit {
    uint32_t sbp = 0, sbw = 0, spsm = 0, ssax = 0, ssay = 0;
    uint32_t dbp = 0, dbw = 0, dpsm = 0, dsax = 0, dsay = 0;
    uint32_t rrw = 0, rrh = 0;
    uint32_t dir = 0;            // TRXPOS.DIR, chapter 01.1
};

struct LocalToHostSetup {
    uint32_t sbp = 0, sbw = 0, spsm = 0, ssax = 0, ssay = 0;
    uint32_t rrw = 0, rrh = 0;
    uint32_t bitsPerPixel = 0;
};

// Interface the memory model implements (M2). The front end calls it in the
// specified commit order; M2 owns the SLM, the surface cache and the transfer log.
class TransferSink {
public:
    virtual ~TransferSink() = default;
    // Chapter 02.7 step 2: drop sources and dirty targets over the rectangle.
    virtual void InvalidateDestination(const UploadCommit& commit) = 0;
    // Chapter 02.7 step 3: swizzle the received bytes into the SLM.
    virtual void CommitUpload(const UploadCommit& commit, const uint8_t* bytes, size_t byteCount) = 0;
    // Chapter 02.7: local to local move (after the overlap check).
    virtual void LocalToLocalMove(const LocalMoveCommit& move) = 0;
    // Chapter 10: local to host read begins.
    virtual void BeginLocalToHost(const LocalToHostSetup& setup) = 0;
    // Discards an unfinished FIFO read when TRXDIR changes or the GS/GIF resets (10.2, 02.1).
    virtual void CancelLocalToHost() {}
};

// Hooks the front end supplies so the bookkeeping can run the overlap checks
// (chapter 02.7) and keep the CLUT dirty tracker informed (step 4).
struct TransferHooks {
    std::function<void(const UploadCommit&)> beforeUpload;   // step 1: flush batch if overlapping
    std::function<void(const UploadCommit&)> afterUpload;    // step 4: transfer log, CLUT tracker
    std::function<void(const LocalMoveCommit&)> beforeMove;  // overlap check for TRXDIR = 2
    std::function<void(const LocalToHostSetup&)> beforeRead; // overlap check for TRXDIR = 1
};

// Destination transfer registers (chapter 01.1).
struct TransferRegisters {
    uint64_t bitbltbuf = 0;
    uint64_t trxpos = 0;
    uint64_t trxreg = 0;
    uint32_t xdir = 3;           // 0 host->local, 1 local->host, 2 local->local, 3 none
};

// Why buffered transfer data is committed (chapter 02.7).
enum class TransferReason : uint8_t { Complete, RegisterChange, TrxdirWrite, ClutLoad, BatchFlush, VBlank, Reset };

class TransferUnit {
public:
    TransferUnit(TransferSink& sink, TransferHooks hooks, CounterSet& counters);

    // BITBLTBUF, TRXPOS, TRXREG: a changed value first commits buffered data.
    void WriteBitbltbuf(uint64_t value);
    void WriteTrxpos(uint64_t value);
    void WriteTrxreg(uint64_t value);
    // TRXDIR: commits pending data first, then arms or runs the transfer.
    void WriteTrxdir(uint64_t value);

    // IMAGE qwords and HWREG writes for host->local (XDIR must be 0).
    void AcceptData(const uint8_t* bytes, size_t count);
    // IN-353: up to `qwords` image qwords (16 bytes each) at once. Takes what the transfer still needs, commits it
    // when complete, and returns how many qwords were used (at least 1; an inactive or local->host transfer
    // discards one qword and counts it), exactly as AcceptData on each qword in turn.
    size_t AcceptImageQwords(const uint8_t* bytes, size_t qwords);

    // Commits buffered data (batch flush, CLUT load, VBlank, reset).
    void CommitPending(TransferReason reason);

    // Discards the transfer state (GS reset). Sets XDIR to 3.
    void Reset();

    // GIF reset: sets TRXDIR to 3 and drops an armed host to local transfer
    // without committing it.
    void ForceTrxdirIdle();

    bool HostToLocalActive() const { return active_; }
    uint32_t Xdir() const { return regs_.xdir; }
    const TransferRegisters& Registers() const { return regs_; }

    // Maximum buffered bytes for one host to local transfer (chapter 02.7).
    static constexpr size_t kCapBytes = 4u * 1024u * 1024u;

private:
    void Commit(bool complete, TransferReason reason);
    // Commit from `data` (byteCount bytes) instead of buffer_: IN-365, a whole transfer that arrived in one run of
    // image qwords is committed straight from the packet. buffer_ is empty then.
    void CommitFrom(const uint8_t* data, size_t byteCount, bool complete);

    TransferSink& sink_;
    TransferHooks hooks_;
    CounterSet& counters_;
    TransferRegisters regs_;

    bool active_ = false;
    UploadCommit pending_;       // destination, bits per pixel, rows (rrh is the remaining rows)
    size_t totalBytes_ = 0;
    std::vector<uint8_t> buffer_;
};

// Size in bytes of a host to local transfer, capped at 4 MiB (chapter 02.7).
size_t HostToLocalByteCount(uint32_t rrw, uint32_t rrh, uint32_t bitsPerPixel);

}  // namespace coregs
