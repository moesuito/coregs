// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 GT4Recomp contributors.
// HWR M3 fixtures: GIF split invariance, flush rules, auto-flush, vertex trace,
// ST rounding, CLUT loads and snapshots, transfer commits and counters.
// Written from the TASK-026 specification (docs/hw-renderer-spec) (clean room).
//
// Usage: hw_gs_m3_tests <test-name>   (names are listed in CMakeLists.txt)
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "hw_batch.hpp"
#include "hw_clut.hpp"
#include "hw_front_end.hpp"
#include "hw_gif_parser.hpp"
#include "hw_gs_registers.hpp"
#include "hw_gs_types.hpp"

using namespace coregs;

namespace {

int g_failures = 0;

#define CHECK(cond)                                                                          \
    do {                                                                                     \
        if (!(cond)) {                                                                       \
            std::fprintf(stderr, "%s:%d: CHECK failed: %s\n", __FILE__, __LINE__, #cond);   \
            ++g_failures;                                                                    \
        }                                                                                    \
    } while (0)

#define CHECK_EQ(actual, expected)                                                           \
    do {                                                                                     \
        const unsigned long long a_ = static_cast<unsigned long long>(actual);              \
        const unsigned long long e_ = static_cast<unsigned long long>(expected);            \
        if (a_ != e_) {                                                                      \
            std::fprintf(stderr, "%s:%d: %s == %llu, expected %llu\n", __FILE__, __LINE__,  \
                         #actual, a_, e_);                                                   \
            ++g_failures;                                                                    \
        }                                                                                    \
    } while (0)

void AppendNumber(std::string& out, unsigned long long value) {
    char buffer[32];
    std::snprintf(buffer, sizeof(buffer), "%llx,", value);
    out += buffer;
}

std::string SerializeBatchBody(const Batch& batch);

std::string SerializeBatch(const Batch& batch) {
    std::string out = "B";
    AppendNumber(out, static_cast<unsigned long long>(batch.reason));
    return out + SerializeBatchBody(batch);
}

// Everything of a batch but its flush reason (TASK-168 compares a segment with the batch a CLUT-load flush made).
std::string SerializeBatchBody(const Batch& batch) {
    std::string out;
    AppendNumber(out, batch.state.effectivePrim);
    AppendNumber(out, batch.state.contextIndex);
    AppendNumber(out, batch.state.ctx.tex0);
    AppendNumber(out, batch.state.clutGeneration);
    AppendNumber(out, static_cast<unsigned long long>(batch.cls));
    AppendNumber(out, batch.prims.size());
    for (const PrimRecord& record : batch.prims) {
        AppendNumber(out, static_cast<unsigned long long>(record.type));
        AppendNumber(out, record.first);
        AppendNumber(out, record.count);
        AppendNumber(out, static_cast<unsigned long long>(record.rect.x0 + 100000));
        AppendNumber(out, static_cast<unsigned long long>(record.rect.y0 + 100000));
        AppendNumber(out, static_cast<unsigned long long>(record.rect.x1 + 100000));
        AppendNumber(out, static_cast<unsigned long long>(record.rect.y1 + 100000));
    }
    for (const Vertex& v : batch.vertices) {
        AppendNumber(out, static_cast<unsigned long long>(v.x4 + 100000));
        AppendNumber(out, static_cast<unsigned long long>(v.y4 + 100000));
        AppendNumber(out, v.z);
        AppendNumber(out, v.r | (v.g << 8) | (v.b << 16) | (static_cast<unsigned long long>(v.a) << 24));
        AppendNumber(out, v.fog);
        AppendNumber(out, FloatToBits(v.q));
        AppendNumber(out, FloatToBits(v.s));
        AppendNumber(out, FloatToBits(v.t));
        AppendNumber(out, v.u);
        AppendNumber(out, v.v);
    }
    // TASK-168: palette segments (none for a batch with one palette).
    AppendNumber(out, batch.paletteSegments.size());
    for (const PaletteSegment& segment : batch.paletteSegments) {
        AppendNumber(out, segment.firstPrim);
        AppendNumber(out, segment.firstVertex);
        AppendNumber(out, segment.clutGeneration);
        unsigned long long hash = 1469598103934665603ull;
        for (uint16_t halfword : segment.clut) {
            hash = (hash ^ halfword) * 1099511628211ull;
        }
        AppendNumber(out, hash);
    }
    out += "\n";
    return out;
}

// Recording sinks. Each appends to the world's event log in stream order.
struct Batches : BatchSink {
    std::string* log = nullptr;
    std::vector<Batch> emitted;
    void OnBatch(Batch&& batch) override {
        *log += SerializeBatch(batch);
        emitted.push_back(std::move(batch));
    }
};

struct Sync : SyncSink {
    std::string* log = nullptr;
    bool flushOnSync = false;
    bool OnSync(SyncKind kind, uint32_t id, uint32_t idMask) override {
        AppendNumber(*log, 0x5000u + static_cast<unsigned>(kind));
        AppendNumber(*log, id);
        AppendNumber(*log, idMask);
        *log += "\n";
        return flushOnSync;
    }
};

struct Memory : TransferSink {
    std::string* log = nullptr;
    std::vector<UploadCommit> uploads;
    std::vector<uint8_t> lastBytes;
    void InvalidateDestination(const UploadCommit& commit) override {
        *log += "I";
        AppendNumber(*log, commit.dbp);
        AppendNumber(*log, commit.dsay);
        AppendNumber(*log, commit.rrh);
        *log += "\n";
    }
    void CommitUpload(const UploadCommit& commit, const uint8_t* bytes, size_t byteCount) override {
        *log += "U";
        AppendNumber(*log, commit.dbp);
        AppendNumber(*log, commit.dsay);
        AppendNumber(*log, commit.rrh);
        AppendNumber(*log, byteCount);
        AppendNumber(*log, commit.partial ? 1u : 0u);
        unsigned long long hash = 1469598103934665603ull;
        for (size_t i = 0; i < byteCount; ++i) {
            hash = (hash ^ bytes[i]) * 1099511628211ull;
        }
        AppendNumber(*log, hash);
        *log += "\n";
        uploads.push_back(commit);
        lastBytes.assign(bytes, bytes + byteCount);
    }
    void LocalToLocalMove(const LocalMoveCommit& move) override {
        *log += "M";
        AppendNumber(*log, move.sbp);
        AppendNumber(*log, move.dbp);
        AppendNumber(*log, move.rrw);
        AppendNumber(*log, move.rrh);
        *log += "\n";
    }
    void BeginLocalToHost(const LocalToHostSetup& setup) override {
        *log += "R";
        AppendNumber(*log, setup.sbp);
        AppendNumber(*log, setup.rrh);
        *log += "\n";
    }
};

// A front end with recording sinks and one shared event log.
struct World {
    explicit World(FrontEndOptions options = FrontEndOptions{})
        : fe(options, batches, sync, memory) {
        batches.log = &log;
        sync.log = &log;
        memory.log = &log;
    }

    std::string log;
    Batches batches;
    Sync sync;
    Memory memory;
    FrontEnd fe;
};

// ---- GIF packet construction -------------------------------------------------

constexpr uint32_t kPacked = 0;
constexpr uint32_t kRegList = 1;
constexpr uint32_t kImage = 2;

Qword Tag(uint32_t nloop, bool eop, bool pre, uint32_t prim, uint32_t flg, uint32_t nreg,
          const std::vector<uint8_t>& regs) {
    Qword q;
    q.lo = (static_cast<uint64_t>(nloop) & 0x7FFFu) | (eop ? (uint64_t{1} << 15) : 0u) |
           (pre ? (uint64_t{1} << 46) : 0u) | (static_cast<uint64_t>(prim & 0x7FFu) << 47) |
           (static_cast<uint64_t>(flg & 3u) << 58) | (static_cast<uint64_t>(nreg & 0xFu) << 60);
    for (size_t i = 0; i < regs.size() && i < 16; ++i) {
        q.hi |= static_cast<uint64_t>(regs[i] & 0xFu) << (4 * i);
    }
    return q;
}

// Writes registers through A+D in PACKED mode: one tag, one qword per write.
void AppendAd(std::vector<Qword>& stream, const std::vector<std::pair<uint8_t, uint64_t>>& writes) {
    stream.push_back(Tag(static_cast<uint32_t>(writes.size()), true, false, 0, kPacked, 1, {0xE}));
    for (const auto& write : writes) {
        Qword q;
        q.lo = write.second;
        q.hi = write.first;
        stream.push_back(q);
    }
}

// Scissor over the whole window, FRAME 10 pages wide at block 0, texture at block 0x10
// (CT32, 16 x 16), Z buffer at page 0x80 with writes disabled, PRIM triangle.
void AppendBaseline(std::vector<Qword>& stream) {
    const uint64_t scissor = (uint64_t{639} << 16) | (uint64_t{447} << 48);
    const uint64_t frame = uint64_t{10} << 16;
    const uint64_t zbuf = 0x80u | (uint64_t{1} << 32);
    const uint64_t tex0 = 0x10u | (uint64_t{10} << 14) | (uint64_t{4} << 26) | (uint64_t{4} << 30);
    AppendAd(stream, {{reg::kScissorOne, scissor}, {reg::kFrameOne, frame}, {reg::kZbufOne, zbuf},
                      {reg::kTex0One, tex0}, {reg::kXyoffsetOne, 0}, {reg::kPrim, 3}});
}

// One triangle (RGBAQ + XYZ2 per vertex) at window coordinates (x, y), size 160.
// drawing = false sets ADC on every vertex.
void AppendTriangle(std::vector<Qword>& stream, uint32_t x, uint32_t y, bool drawing = true) {
    const uint32_t xs[3] = {x, x + 160, x};
    const uint32_t ys[3] = {y, y, y + 160};
    stream.push_back(Tag(3, true, false, 0, kPacked, 2, {0x1, 0x5}));
    for (int v = 0; v < 3; ++v) {
        Qword rgba;
        rgba.lo = 0x40u | (uint64_t{0x80} << 8) | (uint64_t{0xC0} << 16) | (uint64_t{0xFF} << 24);
        stream.push_back(rgba);
        Qword xyz;
        xyz.lo = static_cast<uint64_t>(xs[v] * 16u) | (static_cast<uint64_t>(ys[v] * 16u) << 32);
        xyz.hi = 0x1000u | (drawing ? 0u : (uint64_t{1} << 47));
        stream.push_back(xyz);
    }
}

void Feed(World& world, const std::vector<Qword>& stream) {
    world.fe.GifTransfer(1, stream.data(), stream.size());
}

// Random GIF stream exercising every packet format.
std::vector<Qword> BuildFuzzStream(uint32_t seed) {
    std::mt19937_64 rng(seed);
    const uint8_t addresses[] = {0x40, 0x42, 0x47, 0x4C, 0x4E, 0x18, 0x06, 0x14, 0x50, 0x51, 0x52, 0x53,
                                 0x3B, 0x3D, 0x44, 0x45, 0x46, 0x49, 0x4A, 0x1A, 0x1B, 0x1C, 0x22, 0x34,
                                 0x36, 0x08, 0x3F, 0x60, 0x61, 0x62, 0x00, 0x0C, 0x16, 0x17, 0x01, 0x02};
    const uint8_t regPool[] = {0x0, 0x1, 0x2, 0x3, 0x4, 0x5, 0xA, 0xC, 0xD, 0xE, 0xF, 0x6, 0x7, 0x14};
    std::vector<Qword> stream;

    for (int packet = 0; packet < 80; ++packet) {
        const uint32_t kind = static_cast<uint32_t>(rng() % 5u);
        if (kind == 0u) {  // A+D writes
            const uint32_t count = 1u + static_cast<uint32_t>(rng() % 6u);
            stream.push_back(Tag(count, (rng() & 1u) != 0u, false, 0, kPacked, 1, {0xE}));
            for (uint32_t i = 0; i < count; ++i) {
                Qword q;
                q.lo = rng();
                q.hi = addresses[rng() % sizeof(addresses)] | ((rng() & 0xFFu) << 8);  // upper bits ignored
                stream.push_back(q);
            }
        } else if (kind == 1u || kind == 2u) {  // PACKED with random descriptors
            const uint32_t nreg = 1u + static_cast<uint32_t>(rng() % 16u);
            const uint32_t nloop = 1u + static_cast<uint32_t>(rng() % 4u);
            std::vector<uint8_t> regs;
            for (uint32_t i = 0; i < nreg; ++i) {
                regs.push_back(regPool[rng() % sizeof(regPool)]);
            }
            const bool pre = (rng() % 4u) == 0u;
            stream.push_back(Tag(nloop, (rng() & 1u) != 0u, pre, static_cast<uint32_t>(rng() & 0x7FFu),
                                 kPacked, nreg, regs));
            for (uint32_t i = 0; i < nloop * nreg; ++i) {
                Qword q;
                q.lo = rng();
                q.hi = rng();
                stream.push_back(q);
            }
        } else if (kind == 3u) {  // REGLIST: 64-bit words, two per qword
            const uint32_t nreg = 1u + static_cast<uint32_t>(rng() % 4u);
            const uint32_t nloop = 1u + static_cast<uint32_t>(rng() % 4u);
            std::vector<uint8_t> regs;
            for (uint32_t i = 0; i < nreg; ++i) {
                regs.push_back(regPool[rng() % sizeof(regPool)]);
            }
            stream.push_back(Tag(nloop, false, (rng() % 3u) == 0u, 0, kRegList, nreg, regs));
            const uint32_t words = nloop * nreg;
            for (uint32_t w = 0; w < words; w += 2u) {
                Qword q;
                q.lo = rng();
                q.hi = rng();
                stream.push_back(q);
            }
        } else {  // IMAGE
            const uint32_t nloop = 1u + static_cast<uint32_t>(rng() % 3u);
            stream.push_back(Tag(nloop, false, false, 0, kImage, 0, {}));
            for (uint32_t i = 0; i < nloop; ++i) {
                Qword q;
                q.lo = rng();
                q.hi = rng();
                stream.push_back(q);
            }
        }
    }
    return stream;
}

// Runs a stream fed in chunks of the given qword counts (cycled); returns the event log.
std::string RunChunked(const std::vector<Qword>& stream, const std::vector<size_t>& chunkSizes) {
    World world;
    size_t offset = 0;
    size_t index = 0;
    while (offset < stream.size()) {
        const size_t take = std::min(chunkSizes[index % chunkSizes.size()], stream.size() - offset);
        world.fe.GifTransfer(1, stream.data() + offset, take);
        offset += take;
        ++index;
    }
    world.fe.FlushBatch(FlushReason::Sync);
    return world.log;
}

// ---- GIF tests --------------------------------------------------------------

void TestGifSplit() {
    const std::vector<Qword> stream = BuildFuzzStream(0xC0FFEEu);
    const std::string whole = RunChunked(stream, {stream.size()});
    CHECK(!whole.empty());

    // Fixed chunk sizes from the specification: 1, 2, 5 and 7 qwords.
    for (size_t size : {size_t{1}, size_t{2}, size_t{5}, size_t{7}}) {
        CHECK(RunChunked(stream, {size}) == whole);
    }
    // Randomised split points on qword boundaries.
    std::mt19937 rng(12345u);
    for (int trial = 0; trial < 24; ++trial) {
        std::vector<size_t> sizes;
        for (int i = 0; i < 32; ++i) {
            sizes.push_back(1u + rng() % 9u);
        }
        CHECK(RunChunked(stream, sizes) == whole);
    }
    // Other seeds, single-qword feeding.
    for (uint32_t seed : {1u, 2u, 77u}) {
        const std::vector<Qword> other = BuildFuzzStream(seed);
        CHECK(RunChunked(other, {1}) == RunChunked(other, {other.size()}));
    }
}

struct GifRecorder : GifSink {
    std::vector<std::pair<uint8_t, uint64_t>> writes;
    std::vector<GifKick> kicks;
    void WriteRegister(uint8_t address, uint64_t data) override { writes.emplace_back(address, data); }
    void Kick(const GifKick& kick) override { kicks.push_back(kick); }
    void ImageQword(uint32_t, const Qword&) override {}
};

void TestGifParserVectors() {
    CounterSet counters;
    GifParser parser(counters);
    GifRecorder sink;

    // PRE with PACKED writes PRIM; RGBA takes the Q latch (1.0 after a tag with NLOOP > 0).
    std::vector<Qword> packet;
    packet.push_back(Tag(2, true, true, 0x7FFu, kPacked, 1, {0x1}));
    Qword rgba;
    rgba.lo = 0x11u | (uint64_t{0x22} << 32);
    rgba.hi = 0x33u | (uint64_t{0x44} << 32);
    packet.push_back(rgba);
    packet.push_back(rgba);
    parser.Transfer(1, packet.data(), packet.size(), sink);
    CHECK(sink.writes.size() == 3u);
    CHECK_EQ(sink.writes[0].first, reg::kPrim);
    CHECK_EQ(sink.writes[0].second, 0x7FFu);
    CHECK_EQ(sink.writes[1].first, reg::kRgbaq);
    CHECK_EQ(sink.writes[1].second >> 32, FloatToBits(1.0f));
    CHECK_EQ(sink.writes[1].second & 0xFFu, 0x11u);
    CHECK_EQ((sink.writes[1].second >> 24) & 0xFFu, 0x44u);
    CHECK(!parser.Busy(1));

    // Q sanitisation: STQ with Q = +0 gives the smallest positive normal.
    sink.writes.clear();
    // One packet: STQ (Q = +0 into the latch), then RGBA which takes the latch.
    // A new tag with NLOOP > 0 would reset the latch to 1.0 (chapter 02.1).
    std::vector<Qword> stq;
    stq.push_back(Tag(1, true, false, 0, kPacked, 2, {0x2, 0x1}));
    stq.push_back(Qword{});  // S = T = Q = +0
    stq.push_back(rgba);
    parser.Transfer(2, stq.data(), stq.size(), sink);
    CHECK_EQ(counters.Get(Counter::QSanitised), 1u);
    CHECK_EQ(FloatToBits(parser.QLatch()), 0x00800000u);
    CHECK_EQ(sink.writes.back().second >> 32, 0x00800000u);

    // A+D RGBAQ with Q = +0 becomes 1.0; NaN becomes the largest finite float.
    sink.writes.clear();
    std::vector<Qword> ad;
    ad.push_back(Tag(2, true, false, 0, kPacked, 1, {0xE}));
    Qword zeroQ;
    zeroQ.hi = reg::kRgbaq;
    ad.push_back(zeroQ);
    Qword nanQ;
    nanQ.lo = uint64_t{0x7FC00000u} << 32;
    nanQ.hi = reg::kRgbaq;
    ad.push_back(nanQ);
    parser.Transfer(3, ad.data(), ad.size(), sink);
    CHECK(sink.writes.size() == 2u);
    CHECK_EQ(sink.writes[0].second >> 32, 0x3F800000u);
    CHECK_EQ(sink.writes[1].second >> 32, 0x7F7FFFFFu);
    CHECK_EQ(counters.Get(Counter::QSanitised), 3u);

    // REGLIST: NREG 2, NLOOP 1 gives two words in one qword. XYZ2 in REGLIST uses the
    // register layout (Y at bit 16, Z at bit 32) and is a drawing kick. Descriptor E
    // is a no-op in REGLIST (unlike the A+D descriptor E in PACKED mode).
    sink.kicks.clear();
    sink.writes.clear();
    std::vector<Qword> reglist;
    reglist.push_back(Tag(1, true, false, 0, kRegList, 2, {0x5, 0x0E}));
    Qword words;
    words.lo = uint64_t{5} | (uint64_t{6} << 16) | (uint64_t{7} << 32);
    words.hi = 0x12345678u;
    reglist.push_back(words);
    parser.Transfer(1, reglist.data(), reglist.size(), sink);
    CHECK(sink.kicks.size() == 1u);
    if (sink.kicks.size() == 1u) {
        CHECK_EQ(sink.kicks[0].x, 5u);
        CHECK_EQ(sink.kicks[0].y, 6u);
        CHECK_EQ(sink.kicks[0].z, 7u);
        CHECK(sink.kicks[0].drawing);
    }
    CHECK(sink.writes.empty());

    // A path-1 packet without EOP is forced to zero by the XGKICK end.
    std::vector<Qword> image;
    image.push_back(Tag(4, false, false, 0, kImage, 0, {}));
    image.push_back(Qword{});
    parser.Transfer(1, image.data(), image.size(), sink);
    CHECK(parser.Busy(1));
    parser.EndPathOnePacket();
    CHECK(!parser.Busy(1));
    CHECK_EQ(counters.Get(Counter::GifPathOneNoEop), 1u);
}

// IN-326: general-register XYZ writes kick in A+D as well as REGLIST (01.1, 02.1-02.3).
// Expected field values are independent of either parser path, so equivalence alone cannot
// hide an omitted kick or accidentally decode A+D as the dedicated PACKED XYZ descriptor.
void CheckKickFields(const GifKick& actual, const GifKick& expected) {
    CHECK_EQ(actual.x, expected.x);
    CHECK_EQ(actual.y, expected.y);
    CHECK_EQ(actual.z, expected.z);
    CHECK_EQ(actual.hasFog, expected.hasFog);
    CHECK_EQ(actual.fog, expected.fog);
    CHECK_EQ(actual.drawing, expected.drawing);
}

void TestAdXyzParser() {
    const std::vector<uint8_t> addresses = {reg::kXyz2, reg::kXyzf2, reg::kXyz3, reg::kXyzf3};
    const std::vector<uint64_t> words = {
        uint64_t{0x1234} | (uint64_t{0x5678} << 16) | (uint64_t{0x89ABCDEF} << 32),
        uint64_t{0x4321} | (uint64_t{0x8765} << 16) | (uint64_t{0xABCDEF} << 32) | (uint64_t{0xA5} << 56),
        uint64_t{0x2222} | (uint64_t{0x3333} << 16) | (uint64_t{0xFEDCBA98} << 32),
        uint64_t{0x4444} | (uint64_t{0x5555} << 16) | (uint64_t{0x654321} << 32) | (uint64_t{0x5A} << 56)};
    const GifKick expected[] = {
        {0x1234, 0x5678, 0x89ABCDEF, false, 0, true},
        {0x4321, 0x8765, 0xABCDEF, true, 0xA5, true},
        {0x2222, 0x3333, 0xFEDCBA98, false, 0, false},
        {0x4444, 0x5555, 0x654321, true, 0x5A, false}};
    std::vector<Qword> ad;
    ad.push_back(Tag(4, true, false, 0, kPacked, 1, {0xE}));
    for (size_t i = 0; i < words.size(); ++i) {
        // Address bit 7 is masked; bit 111 and all other non-address bits are ignored.
        ad.push_back(Qword{words[i], uint64_t{addresses[i]} | 0x80u | (uint64_t{1} << 47)});
    }
    std::vector<Qword> reglist{Tag(1, true, false, 0, kRegList, 4, addresses),
                               Qword{words[0], words[1]}, Qword{words[2], words[3]}};
    for (uint32_t path : {1u, 2u, 3u}) {
        for (const auto* stream : {&ad, &reglist}) {
            for (size_t chunk : {size_t{1}, size_t{2}, size_t{5}, size_t{7}, stream->size()}) {
                CounterSet counters;
                GifParser parser(counters);
                GifRecorder sink;
                for (size_t at = 0; at < stream->size(); at += chunk) {
                    parser.Transfer(path, stream->data() + at, std::min(chunk, stream->size() - at), sink);
                }
                CHECK(!parser.Busy(path));
                CHECK(sink.writes.empty());
                CHECK_EQ(sink.kicks.size(), 4);
                for (size_t i = 0; i < std::min(sink.kicks.size(), size_t{4}); ++i) {
                    CheckKickFields(sink.kicks[i], expected[i]);
                }
            }
        }
    }
    // PACKED descriptor 5 still uses Y at bit 32 and honours ADC at bit 111.
    CounterSet counters;
    GifParser parser(counters);
    GifRecorder sink;
    const std::vector<Qword> packed{Tag(2, true, false, 0, kPacked, 1, {5}),
        Qword{uint64_t{0x1234} | (uint64_t{0x5678} << 32), 0x89ABCDEF},
        Qword{uint64_t{0x1234} | (uint64_t{0x5678} << 32), uint64_t{0x89ABCDEF} | (uint64_t{1} << 47)}};
    parser.Transfer(1, packed.data(), packed.size(), sink);
    CHECK_EQ(sink.kicks.size(), 2);
    if (sink.kicks.size() == 2) {
        CheckKickFields(sink.kicks[0], expected[0]);
        GifKick adc = expected[0];
        adc.drawing = false;
        CheckKickFields(sink.kicks[1], adc);
    }
}

void TestAdXyzFrontEnd() {
    const std::vector<uint8_t> addresses = {reg::kXyzf3, reg::kXyz2, reg::kXyz3, reg::kXyzf2};
    const std::vector<uint64_t> words = {
        uint64_t{16} | (uint64_t{16} << 16) | (uint64_t{0x123} << 32) | (uint64_t{0xA5} << 56),
        uint64_t{32} | (uint64_t{16} << 16) | (uint64_t{0x12345678} << 32),
        uint64_t{48} | (uint64_t{16} << 16) | (uint64_t{0x23456789} << 32),
        uint64_t{64} | (uint64_t{16} << 16) | (uint64_t{0x345678} << 32) | (uint64_t{0x5A} << 56)};
    std::vector<Qword> baseline;
    AppendBaseline(baseline);
    AppendAd(baseline, {{reg::kPrim, 0}, {reg::kRgbaq, uint64_t{0x80402010} | (uint64_t{0x3F800000} << 32)}});
    auto ad = baseline;
    std::vector<std::pair<uint8_t, uint64_t>> writes;
    for (size_t i = 0; i < words.size(); ++i) writes.emplace_back(addresses[i], words[i]);
    AppendAd(ad, writes);
    auto reglist = baseline;
    reglist.push_back(Tag(1, true, false, 0, kRegList, 4, addresses));
    reglist.push_back(Qword{words[0], words[1]});
    reglist.push_back(Qword{words[2], words[3]});

    World reference;
    Feed(reference, reglist);
    reference.fe.FlushBatch(FlushReason::Sync);
    CHECK_EQ(reference.fe.Counters().Get(Counter::Vertices), 4);
    CHECK_EQ(reference.fe.Counters().Get(Counter::PrimsSkipped), 2);
    CHECK_EQ(reference.batches.emitted.size(), 1);
    if (reference.batches.emitted.size() == 1) {
        const auto& batch = reference.batches.emitted[0];
        CHECK_EQ(batch.prims.size(), 2);
        CHECK_EQ(batch.vertices.size(), 2);
        if (batch.vertices.size() == 2) {
            CHECK_EQ(batch.vertices[0].x4, 32);
            CHECK_EQ(batch.vertices[0].y4, 16);
            CHECK_EQ(batch.vertices[0].z, 0x12345678);
            CHECK_EQ(batch.vertices[0].fog, 0xA5); // XYZF3 updated the latch despite no draw.
            CHECK_EQ(batch.vertices[1].x4, 64);
            CHECK_EQ(batch.vertices[1].z, 0x345678);
            CHECK_EQ(batch.vertices[1].fog, 0x5A);
        }
    }
    for (const auto* stream : {&ad, &reglist}) {
        for (size_t chunk : {size_t{1}, size_t{2}, size_t{5}, size_t{7}, stream->size()}) {
            CHECK(RunChunked(*stream, {chunk}) == reference.log);
        }
    }
    World actual;
    Feed(actual, ad);
    actual.fe.FlushBatch(FlushReason::Sync);
    CHECK_EQ(actual.fe.Counters().Get(Counter::Vertices), 4);
    CHECK_EQ(actual.fe.Counters().Get(Counter::PrimsSkipped), 2);
    CHECK_EQ(actual.batches.emitted.size(), 1);
}

// ---- flush rules ------------------------------------------------------------

// Draws one triangle so a batch exists, applies the change, draws a second triangle
// and reports the batches emitted so far.
struct FlushCase {
    size_t batches = 0;
    FlushReason firstReason = FlushReason::Count;
    uint64_t counted = 0;
};

FlushCase RunFlushCase(const std::vector<Qword>& prelude, const std::vector<Qword>& change,
                       FlushReason reason) {
    World world;
    std::vector<Qword> stream = prelude;
    AppendTriangle(stream, 0, 0);
    Feed(world, stream);
    std::vector<Qword> second = change;
    AppendTriangle(second, 200, 200);
    Feed(world, second);
    FlushCase result;
    result.batches = world.batches.emitted.size();
    if (!world.batches.emitted.empty()) {
        result.firstReason = world.batches.emitted.front().reason;
    }
    result.counted = world.fe.FlushReasonCount(reason);
    return result;
}

void TestFlushRules() {
    std::vector<Qword> base;
    AppendBaseline(base);

    // Rule 1: PRIM class change (triangle to sprite).
    {
        World world;
        std::vector<Qword> s = base;
        AppendTriangle(s, 0, 0);
        AppendAd(s, {{reg::kPrim, 6}});
        AppendTriangle(s, 200, 200);
        Feed(world, s);
        CHECK(!world.batches.emitted.empty());
        CHECK_EQ(world.fe.FlushReasonCount(FlushReason::PrimChanged), 1u);
        if (!world.batches.emitted.empty()) {
            CHECK(world.batches.emitted.front().reason == FlushReason::PrimChanged);
        }
    }

    // Rule 2: TEST changed (ZTE on, ZTST GEQUAL).
    {
        std::vector<Qword> change;
        AppendAd(change, {{reg::kTestOne, (uint64_t{1} << 16) | (uint64_t{2} << 17)}});
        const FlushCase r = RunFlushCase(base, change, FlushReason::TestOrScissor);
        CHECK_EQ(r.counted, 1u);
        CHECK_EQ(r.batches, 1u);
    }

    // Rule 3: DIMX dirty while dithering.
    {
        std::vector<Qword> prelude = base;
        AppendAd(prelude, {{reg::kDthe, 1}});
        std::vector<Qword> change;
        AppendAd(change, {{reg::kDimx, 0x1234u}});
        CHECK_EQ(RunFlushCase(prelude, change, FlushReason::DitherDimx).counted, 1u);
    }

    // Rule 4: ALPHA dirty while blending (ABE in the snapshot). Without ABE nothing happens.
    {
        std::vector<Qword> prelude = base;
        AppendAd(prelude, {{reg::kPrim, 3 | (1u << 6)}});
        std::vector<Qword> change;
        AppendAd(change, {{reg::kAlphaOne, 0x1u}});
        CHECK_EQ(RunFlushCase(prelude, change, FlushReason::AlphaBlend).counted, 1u);
        CHECK_EQ(RunFlushCase(base, change, FlushReason::AlphaBlend).batches, 0u);
    }

    // Rule 5: FOGCOL dirty with FGE in the snapshot.
    {
        std::vector<Qword> prelude = base;
        AppendAd(prelude, {{reg::kPrim, 3 | (1u << 5)}});
        std::vector<Qword> change;
        AppendAd(change, {{reg::kFogcol, 0x123456u}});
        CHECK_EQ(RunFlushCase(prelude, change, FlushReason::FogColour).counted, 1u);
    }

    // Rule 6: FRAME changed while the snapshot writes colour.
    {
        std::vector<Qword> change;
        AppendAd(change, {{reg::kFrameOne, (uint64_t{10} << 16) | 0x20u}});
        CHECK_EQ(RunFlushCase(base, change, FlushReason::ColourWrite).counted, 1u);
    }

    // Rule 7: ZBUF changed while the snapshot can write Z (ATE off).
    {
        std::vector<Qword> change;
        AppendAd(change, {{reg::kZbufOne, 0x90u}});
        CHECK_EQ(RunFlushCase(base, change, FlushReason::DepthWrite).counted, 1u);
    }

    // Rule 8: TEX0 TBP changed for a textured batch.
    {
        std::vector<Qword> prelude = base;
        AppendAd(prelude, {{reg::kPrim, 3 | (1u << 4)}});
        std::vector<Qword> change;
        AppendAd(change, {{reg::kTex0One, 0x20u | (uint64_t{10} << 14) | (uint64_t{4} << 26) | (uint64_t{4} << 30)}});
        CHECK_EQ(RunFlushCase(prelude, change, FlushReason::Texture).counted, 1u);
    }

    // No flush: a TEX0 CBP change for a non-indexed format is not in rule 8's mask.
    {
        std::vector<Qword> prelude = base;
        AppendAd(prelude, {{reg::kPrim, 3 | (1u << 4)}});
        std::vector<Qword> change;
        AppendAd(change, {{reg::kTex0One, 0x10u | (uint64_t{10} << 14) | (uint64_t{4} << 26) |
                                              (uint64_t{4} << 30) | (uint64_t{0x123} << 37)}});
        CHECK_EQ(RunFlushCase(prelude, change, FlushReason::Texture).batches, 0u);
    }

    // No flush: writing the snapshot value back clears the dirty state.
    {
        std::vector<Qword> change;
        AppendAd(change, {{reg::kFrameOne, (uint64_t{10} << 16) | 0x20u}, {reg::kFrameOne, uint64_t{10} << 16}});
        CHECK_EQ(RunFlushCase(base, change, FlushReason::ColourWrite).batches, 0u);
    }

    // No flush: a non-drawing kick of the same class keeps the batch.
    {
        World world;
        std::vector<Qword> s = base;
        AppendTriangle(s, 0, 0);
        AppendTriangle(s, 200, 200, false);
        Feed(world, s);
        CHECK_EQ(world.batches.emitted.size(), 0u);
    }

    // No flush: TEXFLUSH sets the flag only.
    {
        World world;
        std::vector<Qword> s = base;
        AppendTriangle(s, 0, 0);
        AppendAd(s, {{reg::kTexflush, 0}});
        AppendTriangle(s, 200, 200);
        Feed(world, s);
        CHECK_EQ(world.batches.emitted.size(), 0u);
        CHECK(world.fe.TexflushPending());
        CHECK_EQ(world.fe.Counters().Get(Counter::TexflushSet), 1u);
    }

    // Continuation: a triangle strip split by a TEST flush keeps its queue.
    {
        World world;
        std::vector<Qword> s = base;
        AppendAd(s, {{reg::kPrim, 4}});  // triangle strip
        const uint32_t xs[4] = {0, 160, 0, 160};
        const uint32_t ys[4] = {0, 0, 160, 160};
        for (int v = 0; v < 4; ++v) {
            if (v == 3) {
                AppendAd(s, {{reg::kTestOne, (uint64_t{1} << 16) | (uint64_t{2} << 17)}});
            }
            s.push_back(Tag(1, true, false, 0, kPacked, 2, {0x1, 0x5}));
            Qword rgba;
            rgba.lo = 0xFFu;
            s.push_back(rgba);
            Qword xyz;
            xyz.lo = static_cast<uint64_t>(xs[v] * 16u) | (static_cast<uint64_t>(ys[v] * 16u) << 32);
            xyz.hi = 0x1000u;
            s.push_back(xyz);
        }
        Feed(world, s);
        world.fe.FlushBatch(FlushReason::Sync);
        CHECK_EQ(world.batches.emitted.size(), 2u);
        if (world.batches.emitted.size() == 2u) {
            CHECK_EQ(world.batches.emitted[0].prims.size(), 1u);
            CHECK_EQ(world.batches.emitted[1].prims.size(), 1u);
            const Batch& second = world.batches.emitted[1];
            CHECK_EQ(second.vertices[0].x4, 160 * 16);  // v1 of the strip
            CHECK_EQ(second.vertices[2].y4, 160 * 16);  // v3 of the strip
        }
    }
}

void TestSignalFlush() {
    World world;
    world.sync.flushOnSync = true;
    std::vector<Qword> s;
    AppendBaseline(s);
    AppendTriangle(s, 0, 0);
    AppendAd(s, {{reg::kSignal, 1}});
    Feed(world, s);
    CHECK_EQ(world.batches.emitted.size(), 1u);
    if (!world.batches.emitted.empty()) {
        CHECK(world.batches.emitted[0].reason == FlushReason::Sync);
    }
}

void TestVBlankAndReset() {
    World world;
    std::vector<Qword> s;
    AppendBaseline(s);
    AppendTriangle(s, 0, 0);
    Feed(world, s);
    // A display range away from the frame keeps the batch (chapter 02.8).
    const BlockRange far{4096, 64};
    world.fe.VBlank(&far, 1);
    CHECK_EQ(world.batches.emitted.size(), 0u);
    // An overlapping display flushes.
    const BlockRange near{0, 64};
    world.fe.VBlank(&near, 1);
    CHECK_EQ(world.batches.emitted.size(), 1u);
    if (!world.batches.emitted.empty()) {
        CHECK(world.batches.emitted[0].reason == FlushReason::VBlank);
    }

    // GS reset drops the pending batch without emitting it.
    std::vector<Qword> again;
    AppendBaseline(again);
    AppendTriangle(again, 0, 0);
    Feed(world, again);
    world.fe.ResetGs();
    world.fe.FlushBatch(FlushReason::Sync);
    CHECK_EQ(world.batches.emitted.size(), 1u);
}

// ---- auto-flush -------------------------------------------------------------

// Two sprites on a textured frame at block 0 (texture base equal to the frame base).
// The second sprite samples texels the first one wrote.
void DrawSpritePair(World& world, bool texflush, uint32_t secondTexelU) {
    std::vector<Qword> s;
    AppendBaseline(s);
    // Texture at block 0 (equal to the FRAME base), CT32, 16 x 16.
    AppendAd(s, {{reg::kTex0One, (uint64_t{10} << 14) | (uint64_t{4} << 26) | (uint64_t{4} << 30)},
                 {reg::kPrim, 6 | (1u << 4) | (1u << 8)}});  // sprite, TME, UV
    if (texflush) {
        AppendAd(s, {{reg::kTexflush, 0}});
    }
    auto vertex = [&s](uint32_t u, uint32_t v, uint32_t x, uint32_t y) {
        s.push_back(Tag(1, true, false, 0, kPacked, 2, {0x3, 0x5}));
        Qword uv;
        uv.lo = static_cast<uint64_t>(u * 16u) | (static_cast<uint64_t>(v * 16u) << 32);
        s.push_back(uv);
        Qword xyz;
        xyz.lo = static_cast<uint64_t>(x * 16u) | (static_cast<uint64_t>(y * 16u) << 32);
        xyz.hi = 0x1000u;
        s.push_back(xyz);
    };
    // Sprite 1: screen (0,0)-(16,16), texels (0,0)-(16,16).
    vertex(0, 0, 0, 0);
    vertex(16, 16, 16, 16);
    // Sprite 2: screen (32,0)-(48,16), texels from secondTexelU.
    vertex(secondTexelU, 0, 32, 0);
    vertex(secondTexelU + 16, 16, 48, 16);
    world.fe.GifTransfer(1, s.data(), s.size());
    world.fe.FlushBatch(FlushReason::Sync);
}

void TestAutoflush() {
    // TEXFLUSH pending and sprite 2 samples sprite 1's screen area: the batch splits.
    {
        World world;
        DrawSpritePair(world, true, 0);
        CHECK_EQ(world.fe.Counters().Get(Counter::AutoflushSplits), 1u);
        CHECK_EQ(world.batches.emitted.size(), 2u);
        if (world.batches.emitted.size() == 2u) {
            CHECK(world.batches.emitted[0].reason == FlushReason::Autoflush);
            CHECK_EQ(world.batches.emitted[0].prims.size(), 1u);
        }
    }
    // No TEXFLUSH: sprite 2 starts on the same texture page as sprite 1's last
    // texel, so the batch does not split.
    {
        World world;
        DrawSpritePair(world, false, 0);
        CHECK_EQ(world.fe.Counters().Get(Counter::AutoflushSplits), 0u);
        CHECK_EQ(world.batches.emitted.size(), 1u);
    }
    // Autoflush off: no split even with TEXFLUSH.
    {
        FrontEndOptions options;
        options.autoflush = AutoflushMode::Off;
        World world(options);
        DrawSpritePair(world, true, 0);
        CHECK_EQ(world.fe.Counters().Get(Counter::AutoflushSplits), 0u);
        CHECK_EQ(world.batches.emitted.size(), 1u);
    }
}

// ---- vertex trace and overlap -------------------------------------------------

Batch MakeTriangleBatch(uint64_t effectivePrim) {
    Batch batch;
    batch.state.effectivePrim = effectivePrim;
    batch.state.contextIndex = 0;
    batch.cls = PrimClass::Triangle;
    PrimRecord record;
    record.type = PrimType::Triangle;
    record.first = 0;
    record.count = 3;
    batch.prims.push_back(record);
    Vertex a;
    a.x4 = 16 * 10;
    a.y4 = 16 * 20;
    a.z = 100;
    a.r = 10;
    a.g = 200;
    a.b = 30;
    a.a = 128;
    a.q = 1.0f;
    a.s = 0.0f;
    a.t = 0.0f;
    a.fog = 7;
    Vertex b = a;
    b.x4 = 16 * 50;
    b.z = 300;
    b.r = 90;
    b.a = 200;
    Vertex c = a;
    c.y4 = 16 * 80;
    c.z = 200;
    c.g = 150;
    c.a = 60;
    batch.vertices = {a, b, c};
    return batch;
}

void TestVertexTrace() {
    // Gouraud triangle without texture (IIP bit 3).
    const uint64_t gouraud = 3u | (1u << primbit::kIip);
    const Batch batch = MakeTriangleBatch(gouraud);
    const VertexTrace trace = ComputeVertexTrace(batch, nullptr);
    CHECK_EQ(trace.xMin, 10);
    CHECK_EQ(trace.xMax, 50);
    CHECK_EQ(trace.yMin, 20);
    CHECK_EQ(trace.yMax, 80);
    CHECK_EQ(trace.zMin, 100);
    CHECK_EQ(trace.zMax, 300);
    CHECK_EQ(trace.fogMin, 7);
    CHECK_EQ(trace.fogMax, 7);
    CHECK_EQ(trace.rgbaMin[0], 10);
    CHECK_EQ(trace.rgbaMax[0], 90);
    CHECK_EQ(trace.rgbaMin[1], 150);
    CHECK_EQ(trace.rgbaMax[1], 200);
    CHECK_EQ(trace.vertexAlphaMin, 60);
    CHECK_EQ(trace.vertexAlphaMax, 200);
    CHECK((trace.constantMask & (1u << 7)) != 0u);   // fog constant
    CHECK((trace.constantMask & (1u << 10)) != 0u);  // Q constant
    CHECK((trace.constantMask & (1u << 6)) == 0u);   // Z varies
    CHECK((trace.constantMask & (1u << 0)) == 0u);   // R varies
    CHECK((trace.constantMask & (1u << 4)) == 0u);   // x varies

    // Flat shading: only the provoking vertex (c) counts for colour and alpha.
    const Batch flat = MakeTriangleBatch(3u);
    const VertexTrace flatTrace = ComputeVertexTrace(flat, nullptr);
    CHECK_EQ(flatTrace.rgbaMin[0], 10);
    CHECK_EQ(flatTrace.rgbaMax[0], 10);
    CHECK_EQ(flatTrace.vertexAlphaMin, 60);
    CHECK_EQ(flatTrace.vertexAlphaMax, 60);

    // AA1 without ABE forces alpha 128 on triangles.
    const Batch aa = MakeTriangleBatch(3u | (1u << primbit::kAa1));
    CHECK_EQ(ComputeVertexTrace(aa, nullptr).vertexAlphaMin, 128);
    CHECK_EQ(ComputeVertexTrace(aa, nullptr).vertexAlphaMax, 128);

    // Filter: MXL = 0 follows MMAG.
    Batch filterBatch = MakeTriangleBatch(gouraud);
    CHECK(ComputeVertexTrace(filterBatch, nullptr).filter == VertexTrace::Filter::Nearest);
    filterBatch.state.ctx.tex1 = uint64_t{1} << 5;  // MMAG
    CHECK(ComputeVertexTrace(filterBatch, nullptr).filter == VertexTrace::Filter::Linear);

    // Filter with mipmaps: K = 32 gives LOD 2 (> 0), so MMIN decides.
    Batch lod = MakeTriangleBatch(gouraud | (1u << primbit::kTme) | (1u << primbit::kFst));
    lod.state.ctx.tex1 = (uint64_t{3} << 2) | (uint64_t{32} << 32) | (uint64_t{1} << 6) | 1u;  // MMIN linear, LCM 1
    CHECK(ComputeVertexTrace(lod, nullptr).filter == VertexTrace::Filter::Linear);
    lod.state.ctx.tex1 = (uint64_t{3} << 2) | (uint64_t{32} << 32) | 1u;  // MMIN nearest
    CHECK(ComputeVertexTrace(lod, nullptr).filter == VertexTrace::Filter::Nearest);

    // STQ texel from 0/0 is NaN: excluded and flagged.
    Batch nanBatch = MakeTriangleBatch(gouraud | (1u << primbit::kTme));
    nanBatch.vertices[0].q = 0.0f;
    nanBatch.vertices[0].s = 0.0f;
    CHECK(ComputeVertexTrace(nanBatch, nullptr).texelNaN);

    // Overlap: two triangles forming an axis-aligned quad with a shared diagonal: none.
    Batch quad;
    quad.cls = PrimClass::Triangle;
    quad.state.effectivePrim = 3;
    auto vtx = [](int32_t x, int32_t y) {
        Vertex v;
        v.x4 = x * 16;
        v.y4 = y * 16;
        return v;
    };
    quad.vertices = {vtx(0, 0), vtx(10, 0), vtx(10, 10), vtx(0, 0), vtx(10, 10), vtx(0, 10)};
    quad.prims = {PrimRecord{PrimType::Triangle, 0, 3, PixelRect{}},
                  PrimRecord{PrimType::Triangle, 3, 3, PixelRect{}}};
    CHECK(ClassifyOverlap(quad) == OverlapKind::None);

    // Sprite draw list: sprite 2 overlaps sprite 1 (new group); sprite 3 joins group 2.
    Batch sprites;
    sprites.cls = PrimClass::Sprite;
    sprites.state.effectivePrim = 6;
    sprites.prims = {PrimRecord{PrimType::Sprite, 0, 2, PixelRect{0, 0, 15, 15}},
                     PrimRecord{PrimType::Sprite, 2, 2, PixelRect{8, 8, 20, 20}},
                     PrimRecord{PrimType::Sprite, 4, 2, PixelRect{100, 100, 110, 110}}};
    sprites.vertices.resize(6);
    const std::vector<DrawGroup> groups = BuildSpriteDrawList(sprites);
    CHECK(groups.size() == 2u);
    if (groups.size() == 2u) {
        CHECK_EQ(groups[0].primitiveCount, 1u);
        CHECK_EQ(groups[0].bounds.x0, -1);
        CHECK_EQ(groups[1].primitiveCount, 2u);
    }
    CHECK(ClassifyOverlap(sprites) == OverlapKind::Possible);
}

// ---- ST rounding ------------------------------------------------------------

void TestStRounding() {
    // Chapter 02.4 vector: S = 0.5 (exponent -1), Q = 1.0 (exponent 0) gives m = 10.
    CHECK_EQ(StMantissaBits(UnbiasedExponent(0.5f), UnbiasedExponent(1.0f)), 10u);
    CHECK_EQ(StMantissaBits(UnbiasedExponent(1.0f), UnbiasedExponent(1.0f)), 9u);
    CHECK_EQ(StMantissaBits(UnbiasedExponent(std::ldexp(1.0f, -14)), UnbiasedExponent(1.0f)), 23u);

    // 2^-20 relative to 0.5 is mantissa bit 3: cleared by the low 10 bits.
    const float s = 0.5f * (1.0f + std::ldexp(1.0f, -20));
    CHECK_EQ(FloatToBits(ClearMantissaLowBits(s, 10)), FloatToBits(0.5f));
    const float q = 1.0f + std::ldexp(1.0f, -20);
    CHECK_EQ(FloatToBits(ClearMantissaLowBits(q, 8)), FloatToBits(1.0f));

    // STQ sprite batch: S and T rounded on both vertices, Q only on the second.
    Batch batch;
    batch.cls = PrimClass::Sprite;
    batch.state.effectivePrim = 6 | (1u << primbit::kTme);
    batch.prims.push_back(PrimRecord{PrimType::Sprite, 0, 2, PixelRect{}});
    Vertex a;
    a.s = s;
    a.t = s;
    a.q = q;
    batch.vertices = {a, a};
    CHECK(ApplyStRounding(batch, false));
    CHECK_EQ(FloatToBits(batch.vertices[0].s), FloatToBits(0.5f));
    CHECK_EQ(FloatToBits(batch.vertices[0].q), FloatToBits(q));
    CHECK_EQ(FloatToBits(batch.vertices[1].q), FloatToBits(1.0f));

    // Not applied for UV addressing; applied to a non-sprite only with constant Z.
    Batch uv = batch;
    uv.state.effectivePrim = 6 | (1u << primbit::kTme) | (1u << primbit::kFst);
    uv.vertices = {a, a};
    CHECK(!ApplyStRounding(uv, true));
    Batch tri = batch;
    tri.cls = PrimClass::Triangle;
    tri.prims[0].type = PrimType::Triangle;
    tri.prims[0].count = 3;
    tri.vertices = {a, a, a};
    CHECK(!ApplyStRounding(tri, false));
    CHECK(ApplyStRounding(tri, true));
}

// ---- CLUT -------------------------------------------------------------------

// Writes a 256-entry CT32 palette at CBP in the CSM1 layout.
void WritePaletteCsm1(std::vector<uint8_t>& memory, uint32_t cbp, uint32_t (*value)(uint32_t)) {
    for (uint32_t i = 0; i < 256; ++i) {
        uint32_t px = 0;
        uint32_t py = 0;
        clut::CsmEntryPosition(i, true, px, py);
        clut::PixelLocation location;
        clut::LocatePixel(psm::kCt32, cbp, 1, px, py, location);
        const uint32_t entry = value(i);
        for (uint32_t k = 0; k < 4; ++k) {
            memory[location.PhysicalByte() + k] = static_cast<uint8_t>(entry >> (8 * k));
        }
    }
}

uint32_t PaletteA(uint32_t i) { return 0xFF000000u | (i * 0x010101u); }
uint32_t PaletteB(uint32_t i) { return 0x00FF0000u | i; }

void TestClutLoad() {
    // CSM1 position vectors (chapter 07.4).
    uint32_t x = 0;
    uint32_t y = 0;
    clut::CsmEntryPosition(0, true, x, y);
    CHECK(x == 0 && y == 0);
    clut::CsmEntryPosition(7, true, x, y);
    CHECK(x == 7 && y == 0);
    clut::CsmEntryPosition(8, true, x, y);
    CHECK(x == 0 && y == 1);
    clut::CsmEntryPosition(16, true, x, y);
    CHECK(x == 8 && y == 0);
    clut::CsmEntryPosition(24, true, x, y);
    CHECK(x == 8 && y == 1);
    clut::CsmEntryPosition(32, true, x, y);
    CHECK(x == 0 && y == 2);
    clut::CsmEntryPosition(255, true, x, y);
    CHECK(x == 15 && y == 15);
    clut::CsmEntryPosition(9, false, x, y);
    CHECK(x == 1 && y == 1);

    // TEXA expansion (chapter 01.5 and 07.5 vectors).
    clut::TexaState texa;
    texa.ta0 = 0x40;
    texa.ta1 = 0x80;
    texa.aem = false;
    CHECK_EQ(clut::ExpandRgb16(0x8000, texa, false), 0x80000000u);
    CHECK_EQ(clut::ExpandRgb16(0x7FFF, texa, false), 0x40F8F8F8u);
    texa.aem = true;
    CHECK_EQ(clut::ExpandRgb16(0x0000, texa, true), 0u);
    // IN-07 (resolved): the A bit is checked first, so 0x8000 keeps TA1 under AEM whichever
    // flag the caller passes, and both forms agree for every value.
    CHECK_EQ(clut::ExpandRgb16(0x8000, texa, false) >> 24, 0x80u);
    CHECK_EQ(clut::ExpandRgb16(0x8000, texa, true) >> 24, 0x80u);
    CHECK_EQ(clut::ExpandRgb16(0x0000, texa, false) >> 24, 0u);

    // CLD sequence (chapter 07.2 table). Loads report whether the buffer was rewritten.
    CounterSet counters;
    clut::ClutUnit unit(counters);
    std::vector<uint8_t> memory(kLocalMemoryBytes, 0);
    auto write = [](uint32_t cbp, uint32_t cld, uint32_t format = 0x13u) {
        clut::Tex0Write w;
        w.psm = format;
        w.cbp = cbp;
        w.cpsm = 0;
        w.csm = 0;
        w.csa = 0;
        w.cld = cld;
        return w;
    };
    auto process = [&](const clut::Tex0Write& w) { return unit.ProcessWrite(w, memory.data(), nullptr); };

    CHECK(process(write(0x100, 4)));          // CBP0 0 -> 0x100: load
    CHECK_EQ(unit.Cbp0(), 0x100u);
    CHECK(!process(write(0x100, 4)));         // same CBP0: no load
    CHECK(process(write(0x200, 4)));          // different: load, CBP0 := 0x200
    CHECK_EQ(unit.Cbp0(), 0x200u);
    CHECK(!process(write(0x300, 2, psm::kCt32)));  // not indexed: nothing
    CHECK_EQ(unit.Cbp0(), 0x200u);
    CHECK(process(write(0x400, 3)));          // CBP1 := 0x400
    CHECK_EQ(unit.Cbp1(), 0x400u);
    CHECK(!process(write(0x400, 5)));         // same CBP1
    CHECK(process(write(0x500, 2)));          // CLD 2 always loads, CBP0 := 0x500
    CHECK_EQ(unit.Cbp0(), 0x500u);
    CHECK(!process(write(0x500, 0)));         // CLD 0: nothing
    CHECK(!process(write(0x600, 6)));         // reserved
    CHECK(!process(write(0x600, 7)));         // reserved
    CHECK_EQ(unit.Cbp0(), 0x500u);
    CHECK_EQ(unit.Cbp1(), 0x400u);

    // CSM1 round trip with CSA = 0: the palette comes back unchanged.
    const uint32_t cbp = 0x100;
    WritePaletteCsm1(memory, cbp, PaletteA);
    clut::ClutUnit palette(counters);
    clut::Tex0Write load = write(cbp, 1);
    CHECK(palette.ProcessWrite(load, memory.data(), nullptr));
    const clut::Palette straight = palette.Decode(0x13, psm::kCt32, 0, 0);
    CHECK_EQ(straight.count, 256u);
    bool allMatch = true;
    for (uint32_t i = 0; i < 256; ++i) {
        allMatch = allMatch && straight.entries[i] == PaletteA(i);
    }
    CHECK(allMatch);

    // CSA clamp: with CSA = 8, palette groups 0..7 load to buffer groups 8..15 and
    // groups 8..15 decode buffer group 15 (palette group 7, entries 112..127).
    clut::ClutUnit offset(counters);
    clut::Tex0Write loadOffset = write(cbp, 1);
    loadOffset.csa = 8;
    CHECK(offset.ProcessWrite(loadOffset, memory.data(), nullptr));
    const clut::Palette shifted = offset.Decode(0x13, psm::kCt32, 8, 0);
    CHECK_EQ(shifted.entries[0], straight.entries[0]);
    CHECK_EQ(shifted.entries[127], straight.entries[127]);
    CHECK_EQ(shifted.entries[128], straight.entries[112]);
    CHECK_EQ(shifted.entries[255], straight.entries[127]);

    // Redundancy filter: an unchanged source with the same key is skipped; a write
    // into the source blocks re-enables the load.
    clut::ClutUnit filter(counters);
    CHECK(filter.ProcessWrite(load, memory.data(), nullptr));
    CHECK(!filter.ProcessWrite(load, memory.data(), nullptr));
    filter.NoteBlocksWritten(cbp, 1);
    CHECK(filter.ProcessWrite(load, memory.data(), nullptr));

    // Bad CPSM writes nothing and is counted.
    clut::Tex0Write bad = write(cbp, 1);
    bad.cpsm = 1;
    clut::ClutUnit badUnit(counters);
    const uint64_t badBefore = counters.Get(Counter::ClutBadCpsm);
    badUnit.ProcessWrite(bad, memory.data(), nullptr);
    CHECK_EQ(counters.Get(Counter::ClutBadCpsm), badBefore + 1u);

    // IN-457: the CSM1 offset tables load the same halfwords as LocatePixel at CBP, for every CPSM, both index widths
    // and base pointers that wrap past the end of memory.
    uint64_t seed = 0x457u;
    for (uint8_t& byte : memory) {
        seed = seed * 6364136223846793005ull + 1442695040888963407ull;
        byte = static_cast<uint8_t>(seed >> 56);
    }
    for (const uint32_t cpsm : {psm::kCt32, psm::kCt16, psm::kCt16S}) {
        for (const bool eightBit : {false, true}) {
            for (const uint32_t tableCbp : {0u, 0x100u, 0x2A7u, kBlocksTotal - 5u, kBlocksTotal - 1u}) {
                clut::ClutUnit tableUnit(counters);
                clut::Tex0Write tableLoad = write(tableCbp, 1, eightBit ? 0x13u : 0x14u);
                tableLoad.cpsm = cpsm;
                tableUnit.ProcessWrite(tableLoad, memory.data(), nullptr);
                uint16_t expected[512] = {};
                for (uint32_t e = 0; e < (eightBit ? 256u : 16u); ++e) {
                    uint32_t ex = 0;
                    uint32_t ey = 0;
                    clut::CsmEntryPosition(e, eightBit, ex, ey);
                    clut::PixelLocation location;
                    CHECK(clut::LocatePixel(cpsm, tableCbp, 1, ex, ey, location));
                    const uint8_t* source = memory.data() + location.PhysicalByte();
                    expected[e] = static_cast<uint16_t>(source[0] | (source[1] << 8));
                    if (cpsm == psm::kCt32) expected[e + 256u] = static_cast<uint16_t>(source[2] | (source[3] << 8));
                }
                CHECK(std::memcmp(tableUnit.Halfwords(), expected, sizeof expected) == 0);
            }
        }
    }
}

void TestClutSnapshot() {
    // A palette loaded by TEX0 is the palette draws see until the next load, even when
    // the palette memory changes. The always-load switch is on, because this test
    // changes SLM without a transfer notification (the filter would otherwise be right
    // to skip the copy in the real pipeline only after notifications).
    std::vector<uint8_t> slm(kLocalMemoryBytes, 0);
    WritePaletteCsm1(slm, 0x100, PaletteA);

    FrontEndOptions options;
    options.clutAlwaysLoad = true;
    World world(options);
    world.fe.SetLocalMemory(slm.data());

    std::vector<Qword> s;
    AppendBaseline(s);
    // TEX0: T8 at CBP 0x100 (CT32, CSM 0), CLD 1 loads now; PRIM textured.
    const uint64_t tex0 = 0x10u | (uint64_t{1} << 14) | (uint64_t{0x13} << 20) | (uint64_t{4} << 26) |
                          (uint64_t{4} << 30) | (uint64_t{0x100} << 37) | (uint64_t{1} << 61);
    AppendAd(s, {{reg::kTex0One, tex0}, {reg::kPrim, 3 | (1u << primbit::kTme)}});
    AppendTriangle(s, 0, 0);
    Feed(world, s);

    const uint32_t generation = world.fe.Clut().Generation();
    CHECK_EQ(world.fe.Clut().Decode(0x13, 0, 0, 0).entries[5], PaletteA(5));

    // Overwrite the palette memory with B; a draw does not reload it.
    WritePaletteCsm1(slm, 0x100, PaletteB);
    std::vector<Qword> draw;
    AppendTriangle(draw, 200, 200);
    Feed(world, draw);
    CHECK_EQ(world.fe.Clut().Generation(), generation);
    CHECK_EQ(world.fe.Clut().Decode(0x13, 0, 0, 0).entries[5], PaletteA(5));

    // The textured batch carries the generation of its snapshot.
    world.fe.FlushBatch(FlushReason::Sync);
    CHECK(!world.batches.emitted.empty());
    if (!world.batches.emitted.empty()) {
        CHECK_EQ(world.batches.emitted[0].state.clutGeneration, generation);
    }

    // A new TEX0 load picks up palette B.
    std::vector<Qword> reload;
    AppendAd(reload, {{reg::kTex0One, tex0}});
    Feed(world, reload);
    CHECK_EQ(world.fe.Clut().Decode(0x13, 0, 0, 0).entries[5], PaletteB(5));
    CHECK(world.fe.Clut().Generation() != generation);
}

// ---- transfers --------------------------------------------------------------

void TestTransfers() {
    World world;
    // BITBLTBUF: DBP 0x40, DBW 4, DPSM CT32. TRXPOS: DSAX 0, DSAY 0. TRXREG: 16 x 4.
    const uint64_t bitbltbuf = (uint64_t{0x40} << 32) | (uint64_t{4} << 48);
    const uint64_t trxreg = uint64_t{16} | (uint64_t{4} << 32);
    std::vector<Qword> s;
    AppendAd(s, {{reg::kBitbltbuf, bitbltbuf}, {reg::kTrxpos, 0}, {reg::kTrxreg, trxreg}, {reg::kTrxdir, 0}});
    // 16 x 4 x 32 bits = 256 bytes = 16 qwords; send 8 qwords (128 bytes) first.
    s.push_back(Tag(8, false, false, 0, kImage, 0, {}));
    for (int i = 0; i < 8; ++i) {
        Qword q;
        q.lo = 0x0101010101010101ull * static_cast<uint64_t>(i + 1);
        q.hi = 0x0202020202020202ull * static_cast<uint64_t>(i + 1);
        s.push_back(q);
    }
    Feed(world, s);
    CHECK(world.memory.uploads.empty());

    // A changed BITBLTBUF commits the rows the received 128 bytes cover: 2 rows.
    std::vector<Qword> change;
    AppendAd(change, {{reg::kBitbltbuf, bitbltbuf | 1u}});
    Feed(world, change);
    CHECK_EQ(world.memory.uploads.size(), 1u);
    if (world.memory.uploads.size() == 1u) {
        CHECK(world.memory.uploads[0].partial);
        CHECK_EQ(world.memory.uploads[0].rrh, 2u);
        CHECK_EQ(world.memory.lastBytes.size(), 128u);
    }

    // The second half: a new transfer for the remaining 2 rows starting at DSAY 2.
    std::vector<Qword> rest;
    AppendAd(rest, {{reg::kBitbltbuf, bitbltbuf},
                    {reg::kTrxpos, uint64_t{2} << 48},
                    {reg::kTrxreg, uint64_t{16} | (uint64_t{2} << 32)},
                    {reg::kTrxdir, 0}});
    rest.push_back(Tag(8, true, false, 0, kImage, 0, {}));
    for (int i = 0; i < 8; ++i) {
        Qword q;
        q.lo = 7;
        q.hi = 9;
        rest.push_back(q);
    }
    Feed(world, rest);
    CHECK_EQ(world.memory.uploads.size(), 2u);
    if (world.memory.uploads.size() == 2u) {
        CHECK(!world.memory.uploads[1].partial);
        CHECK_EQ(world.memory.uploads[1].rrh, 2u);
        CHECK_EQ(world.memory.uploads[1].dsay, 2u);
    }
    // Invalidation runs before the swizzle commit (chapter 02.7 order).
    CHECK(world.log.find('I') < world.log.find('U'));
    CHECK_EQ(HostToLocalByteCount(16, 4, 32), 256u);
    CHECK_EQ(HostToLocalByteCount(4096, 4096, 32), kTransferCapBytes);
}

// TASK-149: an upload over a mip level (1..MXL) of the pending batch's texture draws the batch first, as one over level 0
// does; an upload elsewhere, or over the same blocks while the draw does not use mip levels, does not.
uint32_t MipUploadFlushes(uint64_t tex1, uint32_t uploadBlock) {
    World world;
    std::vector<Qword> s;
    AppendBaseline(s);
    const uint64_t miptbp1 = uint64_t{0x2000} | (uint64_t{2} << 14) | (uint64_t{0x2400} << 20) | (uint64_t{2} << 34);
    AppendAd(s, {{reg::kTex1One, tex1}, {reg::kMiptbp1One, miptbp1}, {reg::kPrim, 3 | (1u << primbit::kTme)}});
    AppendTriangle(s, 0, 0);
    const uint64_t bitbltbuf = (uint64_t{uploadBlock} << 32) | (uint64_t{1} << 48);
    AppendAd(s, {{reg::kBitbltbuf, bitbltbuf}, {reg::kTrxpos, 0}, {reg::kTrxreg, uint64_t{16} | (uint64_t{4} << 32)},
                 {reg::kTrxdir, 0}});
    s.push_back(Tag(16, true, false, 0, kImage, 0, {}));
    for (int i = 0; i < 16; ++i) s.push_back(Qword{});
    Feed(world, s);
    return world.fe.FlushReasonCount(FlushReason::Transfer);
}

void TestMipLevelTransferFlush() {
    const uint64_t mipmapped = (uint64_t{2} << 2) | (uint64_t{2} << 6);  // MXL 2, MMIN 2 (nearest mip level)
    CHECK_EQ(MipUploadFlushes(mipmapped, 0x2400), 1u);                   // level 2
    CHECK_EQ(MipUploadFlushes(mipmapped, 0x2000), 1u);                   // level 1
    CHECK_EQ(MipUploadFlushes(mipmapped, 0x3000), 0u);                   // no level there
    CHECK_EQ(MipUploadFlushes(uint64_t{1} << 6, 0x2400), 0u);            // MMIN 1: level 0 only
    CHECK_EQ(MipUploadFlushes(uint64_t{2} << 6, 0x2400), 0u);            // MXL 0
}

// ---- counters and register normalisation -----------------------------------

void TestCounters() {
    World world;
    std::vector<Qword> s;
    // V-01-2: FRAME with PSM T4HL becomes CT32 with the forced FBMSK 0xF0FFFFFF.
    AppendAd(s, {{reg::kFrameOne, uint64_t{0x24} << 24}});
    // V-01-1: TEXCLUT bits 22..23 are counted and not stored.
    AppendAd(s, {{reg::kTexclut, (uint64_t{3} << 22) | 0x123u}});
    // V-02-2: PRMODECONT.AC = 0.
    AppendAd(s, {{reg::kPrmodecont, 0}});
    // TEX0 CPSM is masked with 0b1010: CPSM 1 becomes 0.
    AppendAd(s, {{reg::kTex0One, uint64_t{1} << 51}});
    // TEX2 replaces only PSM, CBP, CPSM, CSM, CSA and CLD: TBP0 stays.
    AppendAd(s, {{reg::kTex0One, 0x77u | (uint64_t{1} << 14)},
                 {reg::kTex2One, (uint64_t{0x13} << 20) | (uint64_t{0x55} << 37)}});
    // FRAME width above 32 is limited to 32 (context 2).
    AppendAd(s, {{reg::kFrameTwo, uint64_t{40} << 16}});
    AppendAd(s, {{reg::kSignal, 0x1122u | (uint64_t{0xFF} << 32)}, {reg::kFinish, 0}});
    Feed(world, s);

    const RegisterFile& regs = world.fe.Registers();
    CHECK_EQ(world.fe.Counters().Get(Counter::FramePsm4h), 1u);
    CHECK_EQ(frame::Psm(regs.Context(0).frame), 0u);
    CHECK_EQ(frame::Fbmsk(regs.Context(0).frame), 0xF0FFFFFFu);
    CHECK_EQ(world.fe.Counters().Get(Counter::TexclutHighBits), 1u);
    CHECK_EQ(regs.Env().texclut, 0x123u);
    CHECK_EQ(world.fe.Counters().Get(Counter::PrmodecontAcZero), 1u);
    CHECK_EQ(tex0::Cpsm(regs.Context(0).tex0), 0u);
    CHECK_EQ(tex0::Tbp0(regs.Context(0).tex0), 0x77u);
    CHECK_EQ(tex0::Psm(regs.Context(0).tex0), 0x13u);
    CHECK_EQ(tex0::Cbp(regs.Context(0).tex0), 0x55u);
    CHECK_EQ(frame::Fbw(regs.Context(1).frame), 32u);
    CHECK(world.log.find('\n') != std::string::npos);  // sync events were delivered

    // Pre-in-REGLIST counter (V-02-1).
    std::vector<Qword> reglist;
    reglist.push_back(Tag(1, true, true, 0, kRegList, 1, {0x1}));
    reglist.push_back(Qword{});
    Feed(world, reglist);
    CHECK_EQ(world.fe.Counters().Get(Counter::PreInRegList), 1u);
}

}  // namespace

// IN-353: the CLUT unit's written-range early-out and its copy-free mode change no decision. Three units see the
// same random sequence of TEX0 writes and memory writes: A notes each write as one range, R notes it block by
// block (the reference: a single block cannot be mis-classified by a range test), B notes ranges and does not
// copy entries. Every load decision, every counter and A's entries must equal R's; B's decisions must equal A's.
// IN-360: the constant-time BlockRangesOverlap equals the block-by-block definition, including ranges that wrap at
// the top of memory, cover all of it, touch end to end, and zero counts.
void TestBlockRangesOverlap() {
    uint64_t state = 0xB10Cu;
    auto next = [&] {
        state = state * 6364136223846793005ull + 1442695040888963407ull;
        return static_cast<uint32_t>(state >> 32);
    };
    const uint32_t counts[] = {0u, 1u, 2u, 31u, 32u, 4480u, 16383u, 16384u, 16390u, 40000u};
    int checked = 0;
    for (int i = 0; i < 200000; ++i) {
        const uint32_t b0 = (i % 7 == 0) ? 16384u - 1u - next() % 64u : next() % 40000u;
        const uint32_t b1 = (i % 5 == 0) ? (b0 + counts[next() % 10]) : next() % 40000u;  // end-to-end cases
        const uint32_t c0 = (i % 3 == 0) ? counts[next() % 10] : next() % 9000u;
        const uint32_t c1 = (i % 4 == 0) ? counts[next() % 10] : next() % 9000u;
        CHECK_EQ(BlockRangesOverlap(b0, c0, b1, c1), BlockRangesOverlapByWalk(b0, c0, b1, c1));
        ++checked;
    }
    std::printf("hw_block_ranges_overlap: %d random cases checked\n", checked);
}

void TestClutCopyFree() {
    std::vector<uint8_t> memory(kLocalMemoryBytes);
    uint64_t state = 0xC1A7u;
    auto next = [&] {
        state = state * 6364136223846793005ull + 1442695040888963407ull;
        return static_cast<uint32_t>(state >> 32);
    };
    for (uint8_t& byte : memory) byte = static_cast<uint8_t>(next());
    CounterSet countersA, countersR, countersB;
    clut::ClutUnit unitA(countersA), unitR(countersR), unitB(countersB);
    unitB.SetCopiesData(false);
    const uint32_t bases[] = {0u, 32u, 4096u, 14528u, 14560u, 16352u};
    int loadsPerformed = 0;
    for (int step = 0; step < 4000; ++step) {
        if (next() % 3u == 0u) {
            uint32_t begin = 0;
            uint32_t count = 1;
            switch (next() % 4u) {
                case 0: begin = bases[next() % 6u] + next() % 64u; count = 1u + next() % 64u; break;
                case 1: begin = 16384u - 1u - next() % 40u; count = 1u + next() % 200u; break;  // wraps
                case 2: begin = next() % 16384u; count = 4480u; break;                           // a frame buffer
                default: begin = next() % 16384u; count = 16384u + next() % 10u; break;           // everything
            }
            unitA.NoteBlocksWritten(begin, count);
            unitB.NoteBlocksWritten(begin, count);
            for (uint32_t i = 0; i < count; ++i) unitR.NoteBlocksWritten((begin + i) % 16384u, 1);
            continue;
        }
        // Half the loads repeat one of four fixed keys (CLD 1), so redundant loads and stale sources both occur.
        clut::Tex0Write w;
        const uint32_t preset = next() % 8u;
        const bool fixed = preset < 4u;
        w.psm = fixed ? (preset & 1u ? 0x13u : 0x14u) : ((next() & 1u) ? 0x13u : 0x14u);
        w.cpsm = fixed ? 0x00u : ((next() & 1u) ? 0x00u : 0x02u);
        w.cbp = fixed ? bases[2 + preset] : bases[next() % 6u] + (next() & 1u) * 4u;
        w.csm = fixed ? 0u : (next() % 8u == 0u ? 1u : 0u);
        w.csa = fixed ? preset : (next() & 31u);
        w.cld = fixed ? 1u : 1u + next() % 5u;
        w.texclutCbw = 1u + next() % 4u;
        w.texclutCou = next() % 4u;
        w.texclutCov = next() % 64u;
        for (int repeat = 0; repeat < (fixed ? 2 : 1); ++repeat) {  // a fixed key twice: the second is redundant
            const bool loadA = unitA.ProcessWrite(w, memory.data(), {});
            const bool loadR = unitR.ProcessWrite(w, memory.data(), {});
            const bool loadB = unitB.ProcessWrite(w, memory.data(), {});
            CHECK_EQ(loadA, loadR);
            CHECK_EQ(loadA, loadB);
            loadsPerformed += loadA ? 1 : 0;
        }
    }
    for (size_t c = 0; c < CounterSet::kCount; ++c) {
        CHECK_EQ(countersA.Get(static_cast<Counter>(c)), countersR.Get(static_cast<Counter>(c)));
        CHECK_EQ(countersA.Get(static_cast<Counter>(c)), countersB.Get(static_cast<Counter>(c)));
    }
    for (uint32_t i = 0; i < 512; ++i) CHECK_EQ(unitA.Halfwords()[i], unitR.Halfwords()[i]);
    CHECK(loadsPerformed > 100);
    // Deterministic sweep around one palette's source blocks (a 16 x 16 CT32 palette at block 4096, also the
    // wrap at the top of memory): every range that starts or ends at a boundary block.
    int staleSeen = 0;
    for (const uint32_t cbp : {4096u, 16352u}) {
        for (uint32_t offset = 0; offset < 48; ++offset) {
            for (const uint32_t count : {1u, 2u, 3u, 8u, 40u}) {
                CounterSet ca, cr;
                clut::ClutUnit a(ca), r(cr);
                clut::Tex0Write w;
                w.psm = 0x13u;
                w.cpsm = 0x00u;
                w.cbp = cbp;
                w.cld = 1u;
                a.ProcessWrite(w, memory.data(), {});
                r.ProcessWrite(w, memory.data(), {});
                const uint32_t begin = (cbp + 16384u - 8u + offset) % 16384u;
                a.NoteBlocksWritten(begin, count);
                for (uint32_t i = 0; i < count; ++i) r.NoteBlocksWritten((begin + i) % 16384u, 1);
                const bool reloadA = a.ProcessWrite(w, memory.data(), {});
                const bool reloadR = r.ProcessWrite(w, memory.data(), {});
                CHECK_EQ(reloadA, reloadR);
                staleSeen += reloadR ? 1 : 0;
            }
        }
    }
    CHECK(staleSeen > 0);
    CHECK(countersA.Get(Counter::ClutLoadsSkipped) > 0);
    // IN-485: remembered palettes against a read on every load, with memory really changing under the writes: the same
    // buffer after every load, for every source format, CSA and layout, and many of the loads answered from memory.
    {
        CounterSet countersM, countersN;
        clut::ClutUnit remembering(countersM), reading(countersN);
        reading.SetDataMemo(false);
        int peekedLoads = 0;
        const auto changeBlocks = [&](uint32_t begin, uint32_t count) {
            for (uint32_t i = 0; i < count; ++i) {
                uint8_t *block = memory.data() + static_cast<size_t>((begin + i) % 16384u) * 256u;
                for (uint32_t b = 0; b < 256u; ++b) block[b] = static_cast<uint8_t>(next());
            }
        };
        for (int step = 0; step < 6000; ++step) {
            const uint32_t action = next() % 16u;
            if (action <= 1u) {
                const uint32_t begin = bases[next() % 6u] + next() % 64u, count = 1u + next() % 40u;
                changeBlocks(begin, count);
                remembering.NoteBlocksWritten(begin, count);
                reading.NoteBlocksWritten(begin, count);
                continue;
            }
            if (action == 2u) {  // GPU pixels stored into memory: no GS write, nobody is told
                changeBlocks(bases[next() % 6u] + next() % 64u, 1u + next() % 8u);
                continue;
            }
            clut::Tex0Write w;
            w.psm = (next() & 1u) ? 0x13u : 0x14u;
            w.cpsm = (next() & 1u) ? 0x00u : ((next() & 1u) ? 0x02u : 0x0Au);
            w.cbp = bases[next() % 6u];
            w.csm = next() % 8u == 0u ? 1u : 0u;
            w.csa = next() & 31u;
            w.cld = 1u;
            w.texclutCbw = 1u + next() % 4u;
            w.texclutCou = next() % 4u;
            w.texclutCov = next() % 8u;
            // IN-534: PeekLoad reads the buffer a load from memory gives, without loading.
            uint16_t peeked[512];
            reading.PeekLoad(w, memory.data(), peeked);
            const bool loadM = remembering.ProcessWrite(w, memory.data(), {});
            const bool loadN = reading.ProcessWrite(w, memory.data(), {});
            CHECK_EQ(loadM, loadN);
            for (uint32_t i = 0; i < 512; ++i) CHECK_EQ(remembering.Halfwords()[i], reading.Halfwords()[i]);
            if (loadN) {
                for (uint32_t i = 0; i < 512; ++i) CHECK_EQ(peeked[i], reading.Halfwords()[i]);
                ++peekedLoads;
            }
        }
        CHECK(peekedLoads > 1000);
        CHECK(remembering.DataMemoHits() > 500u);
        CHECK(remembering.DataMemoReads() > 100u);
    }
    std::printf("hw_clut_copy_free: %d loads performed, %llu skipped as redundant\n", loadsPerformed,
                static_cast<unsigned long long>(countersA.Get(Counter::ClutLoadsSkipped)));
}

// IN-366: the fast paths (strip runs decoded at once, no candidate copy for non-drawing kicks, the flush decision
// reused while nothing it reads changed) give exactly the generic path's batches, sync events, transfers and
// counters. Streams: the game-like strip tags (PACKED STQ, RGBA, XYZF2 with PRE triangle strip, fan or list; random ADC,
// NLOOP 1-60) between A+D material changes that often repeat the current value, XYOFFSET, PRIM and scissor changes,
// plus the generic fuzz packets; every stream is fed in random transfer splits (strip tags cut across transfers
// take the generic path).
void TestFastPaths() {
    int streams = 0;
    uint64_t fastVertices = 0;
    for (uint32_t seed = 1; seed <= 40; ++seed) {
        std::mt19937_64 rng(seed * 7919u);
        std::vector<Qword> stream;
        AppendBaseline(stream);
        const uint64_t tex0s[2] = {0x10u | (uint64_t{10} << 14) | (uint64_t{4} << 26) | (uint64_t{4} << 30),
                                   0x40u | (uint64_t{10} << 14) | (uint64_t{4} << 26) | (uint64_t{4} << 30)};
        uint64_t currentTex0 = tex0s[0];
        for (int packet = 0; packet < 120; ++packet) {
            const uint32_t kind = static_cast<uint32_t>(rng() % 10u);
            if (kind < 6u) {  // strip-shaped tag
                const uint32_t nloop = 1u + static_cast<uint32_t>(rng() % 60u);
                const uint32_t primType = (rng() % 6u == 0u) ? 5u : (rng() % 8u == 0u ? 3u : 4u);  // fan / list / strip
                const uint32_t prim = primType | (1u << 3) | (1u << 4) | ((rng() & 1u) << 5) | ((rng() & 1u) << 6);
                stream.push_back(Tag(nloop, (rng() & 1u) != 0u, (rng() % 5u) != 0u, prim, kPacked, 3, {0x2, 0x1, 0x4}));
                const uint32_t adcPercent = static_cast<uint32_t>(rng() % 100u);
                for (uint32_t v = 0; v < nloop; ++v) {
                    Qword stq;
                    stq.lo = (uint64_t{0x3F000000u + static_cast<uint32_t>(rng() % 0x800000u)}) |
                             (uint64_t{0x3E800000u + static_cast<uint32_t>(rng() % 0x800000u)} << 32);
                    stq.hi = (rng() % 16u == 0u) ? 0u : (0x3F800000u + static_cast<uint32_t>(rng() % 0x100000u));
                    Qword rgba;
                    rgba.lo = (rng() & 0xFFu) | ((rng() & 0xFFu) << 32);
                    rgba.hi = (rng() & 0xFFu) | ((rng() & 0xFFu) << 32);
                    Qword xyz;
                    const uint32_t x = static_cast<uint32_t>(rng() % 12000u);
                    const uint32_t y = static_cast<uint32_t>(rng() % 8000u);
                    xyz.lo = static_cast<uint64_t>(x) | (static_cast<uint64_t>(y) << 32);
                    xyz.hi = (static_cast<uint64_t>(rng() & 0xFFFFFFu) << 4) | ((rng() & 0xFFu) << 36) |
                             ((static_cast<uint32_t>(rng() % 100u) < adcPercent) ? (uint64_t{1} << 47) : 0u);
                    stream.push_back(stq);
                    stream.push_back(rgba);
                    stream.push_back(xyz);
                }
            } else if (kind < 9u) {  // material change: often the same TEX0 again, CLAMP/TEX1, sometimes XYOFFSET
                std::vector<std::pair<uint8_t, uint64_t>> writes;
                if (rng() % 3u != 0u) currentTex0 = tex0s[rng() % 2u];
                writes.push_back({reg::kTex0One, currentTex0});
                writes.push_back({reg::kClampOne, rng() % 2u == 0u ? 0u : 5u});
                writes.push_back({0x14, rng() % 2u == 0u ? 0u : 0x60u});  // TEX1_1
                if (rng() % 10u == 0u) writes.push_back({reg::kXyoffsetOne, (rng() % 4u) * 16u});
                if (rng() % 12u == 0u) writes.push_back({reg::kPrim, 4u | (1u << 4)});
                if (rng() % 15u == 0u) {
                    writes.push_back({reg::kScissorOne, (uint64_t{(rng() % 640u)} << 16) | (uint64_t{447} << 48)});
                }
                AppendAd(stream, writes);
            } else {  // generic fuzz packets
                const std::vector<Qword> fuzz = BuildFuzzStream(static_cast<uint32_t>(rng()));
                stream.insert(stream.end(), fuzz.begin(), fuzz.end());  // whole packets
            }
        }
        FrontEndOptions fastOptions;
        FrontEndOptions slowOptions;
        slowOptions.fastPaths = false;
        World fast(fastOptions);
        World slow(slowOptions);
        size_t offset = 0;
        while (offset < stream.size()) {
            const size_t take = std::min<size_t>(stream.size() - offset, 1u + static_cast<size_t>(rng() % 3000u));
            fast.fe.GifTransfer(1, stream.data() + offset, take);
            slow.fe.GifTransfer(1, stream.data() + offset, take);
            offset += take;
        }
        fast.fe.VBlank(nullptr, 0);
        slow.fe.VBlank(nullptr, 0);
        CHECK(fast.log == slow.log);
        CHECK_EQ(fast.batches.emitted.size(), slow.batches.emitted.size());
        for (size_t c = 0; c < CounterSet::kCount; ++c) {
            CHECK_EQ(fast.fe.Counters().Get(static_cast<Counter>(c)), slow.fe.Counters().Get(static_cast<Counter>(c)));
        }
        fastVertices += fast.fe.FastStripVertices();
        ++streams;
    }
    CHECK(fastVertices > 20000);
    std::printf("hw_front_end_fast_paths: %d streams, %llu strip vertices on the fast path\n", streams,
                static_cast<unsigned long long>(fastVertices));
}

// ---- TASK-168: palette segments ------------------------------------------------------------------------------

// The pieces of the segment fixtures: a T8 16 x 16 texture at block 0x2000 and CT32 CSM1 palettes from block 0x3000, both
// away from the frame (FBP 0, FBW 10) and the depth buffer (page 0x80, never used: ZMSK = 1, no depth test).
constexpr uint32_t kSegmentTexture = 0x2000;
constexpr uint32_t kSegmentPalettes[3] = {0x3000, 0x3020, 0x3040};

uint64_t SegmentTex0(uint32_t cbp, uint32_t cld, uint32_t tbp = kSegmentTexture, uint32_t csm = 0) {
    return tbp | (uint64_t{1} << 14) | (uint64_t{0x13} << 20) | (uint64_t{4} << 26) | (uint64_t{4} << 30) |
           (uint64_t{cbp} << 37) | (uint64_t{csm} << 55) | (uint64_t{cld} << 61);
}

uint32_t PaletteC(uint32_t i) { return 0x80000000u | (i << 8) | (255u - i); }

void AppendSegmentBaseline(std::vector<Qword>& stream, uint64_t prim) {
    const uint64_t scissor = (uint64_t{639} << 16) | (uint64_t{447} << 48);
    const uint64_t frame = uint64_t{10} << 16;
    const uint64_t zbuf = 0x80u | (uint64_t{1} << 32);
    AppendAd(stream, {{reg::kScissorOne, scissor}, {reg::kFrameOne, frame}, {reg::kZbufOne, zbuf},
                      {reg::kTex0One, SegmentTex0(kSegmentPalettes[0], 0)}, {reg::kXyoffsetOne, 0},
                      {reg::kPrim, prim}});
}

constexpr uint64_t kSegmentSpritePrim = 6u | (1u << primbit::kTme) | (1u << primbit::kFst);
constexpr uint64_t kSegmentTrianglePrim = 3u | (1u << primbit::kTme);

// One FST sprite (RGBAQ, UV, XYZ2 per vertex) 8 pixels wide at (x, y).
void AppendSegmentSprite(std::vector<Qword>& stream, uint32_t x, uint32_t y) {
    stream.push_back(Tag(2, true, false, 0, kPacked, 3, {0x1, 0x3, 0x5}));
    for (uint32_t v = 0; v < 2; ++v) {
        Qword rgba;
        rgba.lo = 0x80u | (uint64_t{0x80} << 32);
        rgba.hi = 0x80u | (uint64_t{0x80} << 32);
        Qword uv;
        uv.lo = static_cast<uint64_t>(v * 16u * 16u) | (static_cast<uint64_t>(v * 16u * 16u) << 32);
        Qword xyz;
        xyz.lo = static_cast<uint64_t>((x + 8u * v) * 16u) | (static_cast<uint64_t>((y + 8u * v) * 16u) << 32);
        xyz.hi = uint64_t{0x100} << 4;
        stream.push_back(rgba);
        stream.push_back(uv);
        stream.push_back(xyz);
    }
}

// One STQ triangle (ST, RGBAQ, XYZ2 per vertex) at (x, y) with depth z; S and T have low mantissa bits for 02.4 to clear.
void AppendSegmentTriangle(std::vector<Qword>& stream, uint32_t x, uint32_t y, uint32_t z) {
    const uint32_t xs[3] = {x, x + 20, x};
    const uint32_t ys[3] = {y, y, y + 20};
    stream.push_back(Tag(3, true, false, 0, kPacked, 3, {0x2, 0x1, 0x5}));
    for (uint32_t v = 0; v < 3; ++v) {
        Qword st;
        st.lo = uint64_t{FloatToBits(0.1234567f + 0.01f * static_cast<float>(v))} |
                (uint64_t{FloatToBits(0.7654321f - 0.01f * static_cast<float>(v))} << 32);
        st.hi = FloatToBits(1.0f + 0.03125f * static_cast<float>(v));
        Qword rgba;
        rgba.lo = 0x80u | (uint64_t{0x80} << 32);
        rgba.hi = 0x80u | (uint64_t{0x80} << 32);
        Qword xyz;
        xyz.lo = static_cast<uint64_t>(xs[v] * 16u) | (static_cast<uint64_t>(ys[v] * 16u) << 32);
        xyz.hi = static_cast<uint64_t>(z) << 4;
        stream.push_back(st);
        stream.push_back(rgba);
        stream.push_back(xyz);
    }
}

// A front end over a memory image holding the three palettes, whose sink also keeps the palette the CLUT unit decodes for
// each batch when it arrives (what a renderer reading the live CLUT unit would draw with).
struct SegmentWorld {
    explicit SegmentWorld(bool segments, FrontEndOptions options = FrontEndOptions{})
        : slm(kLocalMemoryBytes, 0), world(options) {
        WritePaletteCsm1(slm, kSegmentPalettes[0], PaletteA);
        WritePaletteCsm1(slm, kSegmentPalettes[1], PaletteB);
        WritePaletteCsm1(slm, kSegmentPalettes[2], PaletteC);
        world.fe.SetLocalMemory(slm.data());
        if (segments) {
            world.fe.SetPaletteSegmentHook([this](const clut::Tex0Write&, const Batch&) {
                ++hookQuestions;
                return hookAnswer;
            });
        }
    }
    std::vector<uint8_t> slm;
    World world;
    bool hookAnswer = true;
    uint32_t hookQuestions = 0;
};

// The batches of a run without segments next to one with them: each segment, as PaletteSegmentBatch gives it, must be
// the batch the CLUT-load flush made, with the palette the CLUT unit held for that batch. All is compared but the flush
// reason and the TEX0 fields that only steer CLUT loads (CBP, CSM, CLD): the merged batch keeps its first primitive's
// TEX0 snapshot, as any batch does over a TEX0 write that changes only those (chapter 02.5 rule 8 masks).
// `mergedPalette` is the palette the CLUT unit held when the merged batch arrived (for a batch without segments).
void CheckSegmentsMatchFlushes(const std::vector<Batch>& flushed, const std::vector<clut::Palette>& flushedPalettes,
                               const Batch& merged, const clut::Palette& mergedPalette, uint32_t firstFlushed = 0) {
    constexpr uint64_t kClutLoadFields = ((uint64_t{1} << 14) - 1u) << 37 | uint64_t{1} << 55 | uint64_t{7} << 61;
    CounterSet counters;
    clut::ClutUnit unused(counters);
    const uint32_t segments = PaletteSegmentCount(merged);
    CHECK(firstFlushed + segments <= flushed.size());
    for (uint32_t s = 0; s < segments && firstFlushed + s < flushed.size(); ++s) {
        Batch actual = PaletteSegmentBatch(merged, s);
        Batch expected = flushed[firstFlushed + s];
        CHECK_EQ(actual.state.clutGeneration, expected.state.clutGeneration);
        for (Batch* batch : {&actual, &expected}) {
            batch->paletteSegments.clear();
            batch->state.ctx.tex0 &= ~kClutLoadFields;
        }
        const std::string a = SerializeBatchBody(actual);
        const std::string e = SerializeBatchBody(expected);
        CHECK(a == e);
        if (a != e) std::fprintf(stderr, "  segment %u: %s  expected %s", s, a.substr(0, 160).c_str(), e.substr(0, 160).c_str());
        const clut::Palette palette = merged.paletteSegments.empty() ? mergedPalette : DecodeSegmentPalette(merged, s, unused);
        CHECK(std::memcmp(palette.entries, flushedPalettes[firstFlushed + s].entries, sizeof palette.entries) == 0);
    }
}

// A sink that decodes the live CLUT unit's palette for every batch it receives.
struct PaletteRecorder : BatchSink {
    const FrontEnd* fe = nullptr;
    std::vector<Batch> batches;
    std::vector<clut::Palette> palettes;
    void OnBatch(Batch&& batch) override {
        const uint64_t tex0 = batch.state.ctx.tex0;
        palettes.push_back(fe->Clut().Decode(tex0::Psm(tex0), tex0::Cpsm(tex0), tex0::Csa(tex0), batch.state.env.texa));
        batches.push_back(std::move(batch));
    }
};

struct NoSync : SyncSink {
    bool OnSync(SyncKind, uint32_t, uint32_t) override { return false; }
};

struct NoMemory : TransferSink {
    void InvalidateDestination(const UploadCommit&) override {}
    void CommitUpload(const UploadCommit&, const uint8_t*, size_t) override {}
    void LocalToLocalMove(const LocalMoveCommit&) override {}
    void BeginLocalToHost(const LocalToHostSetup&) override {}
};

// Runs `stream` with and without palette segments; returns the batches and live palettes of each.
struct SegmentRun {
    std::vector<Batch> batches;
    std::vector<clut::Palette> palettes;
    uint64_t kept = 0;
    uint64_t clutFlushes = 0;
    uint32_t hookQuestions = 0;
    uint32_t filterQuestions = 0;
    std::vector<uint8_t> clutAtEnd;
};

// `filterAnswer`: -1 sets no batch filter (TASK-168b), else the filter's answer.
SegmentRun RunSegments(const std::vector<Qword>& stream, bool segments, bool hookAnswer = true,
                       FrontEndOptions options = FrontEndOptions{}, int filterAnswer = -1) {
    std::vector<uint8_t> slm(kLocalMemoryBytes, 0);
    WritePaletteCsm1(slm, kSegmentPalettes[0], PaletteA);
    WritePaletteCsm1(slm, kSegmentPalettes[1], PaletteB);
    WritePaletteCsm1(slm, kSegmentPalettes[2], PaletteC);
    PaletteRecorder recorder;
    NoSync sync;
    NoMemory memory;
    FrontEnd fe(options, recorder, sync, memory);
    recorder.fe = &fe;
    fe.SetLocalMemory(slm.data());
    SegmentRun run;
    if (segments) {
        fe.SetPaletteSegmentHook([&](const clut::Tex0Write&, const Batch&) {
            ++run.hookQuestions;
            return hookAnswer;
        });
        if (filterAnswer >= 0) {
            fe.SetPaletteSegmentBatchFilter([&](const Batch&) {
                ++run.filterQuestions;
                return filterAnswer != 0;
            });
        }
    }
    fe.GifTransfer(1, stream.data(), stream.size());
    fe.FlushBatch(FlushReason::Sync);
    run.batches = std::move(recorder.batches);
    run.palettes = std::move(recorder.palettes);
    run.kept = fe.ClutLoadsKept();
    run.clutFlushes = fe.FlushReasonCount(FlushReason::ClutLoad);
    run.clutAtEnd.assign(reinterpret_cast<const uint8_t*>(fe.Clut().Halfwords()),
                         reinterpret_cast<const uint8_t*>(fe.Clut().Halfwords() + clut::kClutHalfwords));
    return run;
}

// The segment basics: loads of palettes A, B, then C and A back to back keep one batch with three segments (the second
// load in a row replaces the palette of the segment it opened); each segment equals the batch its flush made.
void TestPaletteSegments() {
    std::vector<Qword> s;
    AppendSegmentBaseline(s, kSegmentSpritePrim);
    AppendAd(s, {{reg::kTex0One, SegmentTex0(kSegmentPalettes[0], 1)}});
    AppendSegmentSprite(s, 0, 0);
    AppendSegmentSprite(s, 16, 0);
    AppendAd(s, {{reg::kTex0One, SegmentTex0(kSegmentPalettes[1], 1)}});
    AppendSegmentSprite(s, 32, 0);
    AppendAd(s, {{reg::kTex0One, SegmentTex0(kSegmentPalettes[2], 1)}});
    AppendAd(s, {{reg::kTex0One, SegmentTex0(kSegmentPalettes[0], 1)}});
    AppendSegmentSprite(s, 48, 0);
    AppendSegmentSprite(s, 64, 0);

    const SegmentRun flushed = RunSegments(s, false);
    const SegmentRun merged = RunSegments(s, true);
    CHECK_EQ(flushed.batches.size(), 3u);
    CHECK_EQ(flushed.clutFlushes, 2u);
    CHECK_EQ(flushed.kept, 0u);
    CHECK_EQ(merged.batches.size(), 1u);
    CHECK_EQ(merged.clutFlushes, 0u);
    CHECK_EQ(merged.kept, 3u);
    CHECK(merged.clutAtEnd == flushed.clutAtEnd);
    if (merged.batches.size() != 1u || flushed.batches.size() != 3u) return;
    const Batch& batch = merged.batches[0];
    CHECK_EQ(batch.paletteSegments.size(), 3u);
    CHECK_EQ(batch.prims.size(), 5u);
    if (batch.paletteSegments.size() == 3u) {
        CHECK_EQ(batch.paletteSegments[0].firstPrim, 0u);
        CHECK_EQ(batch.paletteSegments[1].firstPrim, 2u);
        CHECK_EQ(batch.paletteSegments[1].firstVertex, 4u);
        CHECK_EQ(batch.paletteSegments[2].firstPrim, 3u);
        CHECK_EQ(batch.paletteSegments[2].firstVertex, 6u);
        CHECK_EQ(batch.state.clutGeneration, batch.paletteSegments[0].clutGeneration);
    }
    // The palettes A, B and A, decoded from the copies (the CLUT unit given is not read).
    CounterSet counters;
    clut::ClutUnit unused(counters);
    CHECK_EQ(DecodeSegmentPalette(batch, 0, unused).entries[7], PaletteA(7));
    CHECK_EQ(DecodeSegmentPalette(batch, 1, unused).entries[7], PaletteB(7));
    CHECK_EQ(DecodeSegmentPalette(batch, 2, unused).entries[200], PaletteA(200));
    CHECK_EQ(PaletteSegmentEndPrim(batch, 0), 2u);
    CHECK_EQ(PaletteSegmentEndPrim(batch, 2), 5u);
    CheckSegmentsMatchFlushes(flushed.batches, flushed.palettes, batch, merged.palettes[0]);
}

// Chapter 02.4 per segment: triangles of constant Z in each segment, a different Z per segment. The batches of the
// flushes round S, T and Q (constant Z); so must the merged batch, segment by segment.
void TestPaletteSegmentRounding() {
    std::vector<Qword> s;
    AppendSegmentBaseline(s, kSegmentTrianglePrim);
    AppendAd(s, {{reg::kTex0One, SegmentTex0(kSegmentPalettes[0], 1)}});
    AppendSegmentTriangle(s, 0, 0, 0x100);
    AppendSegmentTriangle(s, 40, 0, 0x100);
    AppendAd(s, {{reg::kTex0One, SegmentTex0(kSegmentPalettes[1], 1)}});
    AppendSegmentTriangle(s, 80, 0, 0x200);
    AppendSegmentTriangle(s, 120, 0, 0x200);
    AppendAd(s, {{reg::kTex0One, SegmentTex0(kSegmentPalettes[2], 1)}});
    AppendSegmentTriangle(s, 160, 0, 0x300);
    AppendSegmentTriangle(s, 200, 0, 0x301);  // not constant: no rounding in this segment
    const SegmentRun flushed = RunSegments(s, false);
    const SegmentRun merged = RunSegments(s, true);
    CHECK_EQ(flushed.batches.size(), 3u);
    CHECK_EQ(merged.batches.size(), 1u);
    if (merged.batches.size() != 1u || flushed.batches.size() != 3u) return;
    CHECK_EQ(merged.batches[0].paletteSegments.size(), 3u);
    CheckSegmentsMatchFlushes(flushed.batches, flushed.palettes, merged.batches[0], merged.palettes[0]);
    // The rounding really ran in the first segment (constant Z) and not in the last (Z 0x300 and 0x301), as the
    // per-batch rule of the flushes decided: the whole merged batch has no constant Z.
    CHECK(merged.batches[0].vertices[0].s != 0.1234567f);
    CHECK(merged.batches[0].vertices[12].s == 0.1234567f);
    CHECK(flushed.batches[2].vertices[0].s == 0.1234567f);
}

// The loads that must still end the batch: a texture change, CSM2, a palette in the frame's memory, a texture inside
// the frame, the other context, the renderer's refusal (the hook, or TASK-168b's batch filter, asked before the hook),
// and the segment limit.
void TestPaletteSegmentsStillFlush() {
    auto twoSprites = [](uint64_t secondTex0, uint8_t tex0Register = reg::kTex0One, uint32_t firstTbp = kSegmentTexture) {
        std::vector<Qword> s;
        AppendSegmentBaseline(s, kSegmentSpritePrim);
        AppendAd(s, {{reg::kTex0One, SegmentTex0(kSegmentPalettes[0], 1, firstTbp)}});
        AppendSegmentSprite(s, 0, 0);
        AppendAd(s, {{tex0Register, secondTex0}});
        AppendSegmentSprite(s, 16, 0);
        return s;
    };
    struct Case {
        const char* name;
        std::vector<Qword> stream;
        bool hookAnswer;
        uint64_t batches;
        int filterAnswer = -1;         // TASK-168b: -1 no batch filter
        uint32_t hookQuestions = ~0u;  // checked when not ~0u
    };
    const Case cases[] = {
        {"same texture (kept)", twoSprites(SegmentTex0(kSegmentPalettes[1], 1)), true, 1},
        {"texture base changes", twoSprites(SegmentTex0(kSegmentPalettes[1], 1, kSegmentTexture + 0x20)), true, 2},
        {"CSM2", twoSprites(SegmentTex0(kSegmentPalettes[1], 1, kSegmentTexture, 1)), true, 2},
        {"palette in the frame", twoSprites(SegmentTex0(0x40, 1)), true, 2},
        {"texture in the frame", twoSprites(SegmentTex0(kSegmentPalettes[1], 1, 0x40), reg::kTex0One, 0x40), true, 2},
        {"other context", twoSprites(SegmentTex0(kSegmentPalettes[1], 1), reg::kTex0Two), true, 2},
        {"renderer refuses", twoSprites(SegmentTex0(kSegmentPalettes[1], 1)), false, 2},
        {"batch filter refuses, hook not asked", twoSprites(SegmentTex0(kSegmentPalettes[1], 1)), true, 2, 0, 0},
        {"batch filter allows", twoSprites(SegmentTex0(kSegmentPalettes[1], 1)), true, 1, 1, 1},
    };
    for (const Case& c : cases) {
        const SegmentRun run = RunSegments(c.stream, true, c.hookAnswer, FrontEndOptions{}, c.filterAnswer);
        if (c.hookQuestions != ~0u) CHECK_EQ(run.hookQuestions, c.hookQuestions);
        if (c.filterAnswer >= 0) CHECK_EQ(run.filterQuestions, 1u);
        if (run.batches.size() != c.batches) std::fprintf(stderr, "  case '%s'\n", c.name);
        CHECK_EQ(run.batches.size(), c.batches);
        // Whatever happened, the batches equal those of the run without segments, segment by segment.
        const SegmentRun reference = RunSegments(c.stream, false);
        uint32_t next = 0;
        for (size_t b = 0; b < run.batches.size(); ++b) {
            CheckSegmentsMatchFlushes(reference.batches, reference.palettes, run.batches[b], run.palettes[b], next);
            next += PaletteSegmentCount(run.batches[b]);
        }
        CHECK_EQ(next, reference.batches.size());
    }

    // The limit: a load and a sprite, again and again. A batch holds kMaxPaletteSegments segments; the next load ends it.
    std::vector<Qword> s;
    AppendSegmentBaseline(s, kSegmentSpritePrim);
    const uint32_t loads = kMaxPaletteSegments + 10u;
    for (uint32_t i = 0; i < loads; ++i) {
        AppendAd(s, {{reg::kTex0One, SegmentTex0(kSegmentPalettes[i % 3u], 1)}});
        AppendSegmentSprite(s, (i * 8u) % 600u, ((i * 8u) / 600u) * 8u);
    }
    const SegmentRun limited = RunSegments(s, true);
    const SegmentRun reference = RunSegments(s, false);
    CHECK_EQ(reference.batches.size(), loads);
    CHECK_EQ(limited.batches.size(), 2u);
    if (limited.batches.size() == 2u) {
        CHECK_EQ(limited.batches[0].paletteSegments.size(), kMaxPaletteSegments);
        CHECK_EQ(limited.batches[1].paletteSegments.size(), 10u);
        CHECK(limited.batches[0].reason == FlushReason::ClutLoad);
    }
    uint32_t next = 0;
    for (size_t b = 0; b < limited.batches.size(); ++b) {
        CheckSegmentsMatchFlushes(reference.batches, reference.palettes, limited.batches[b], limited.palettes[b], next);
        next += PaletteSegmentCount(limited.batches[b]);
    }
    CHECK_EQ(next, loads);
}

// The capacity limit counts from the last segment's start, where the flush would have started a batch: 10 sprites, a
// load, then 16400 sprites. The flushes give batches of 10, 16383 (32766 vertices, the sprite limit) and 17 sprites; the
// merged run gives the same three batches.
void TestPaletteSegmentCapacity() {
    std::vector<Qword> s;
    AppendSegmentBaseline(s, kSegmentSpritePrim);
    AppendAd(s, {{reg::kTex0One, SegmentTex0(kSegmentPalettes[0], 1)}});
    for (uint32_t i = 0; i < 10u; ++i) AppendSegmentSprite(s, 8u * i, 0);
    AppendAd(s, {{reg::kTex0One, SegmentTex0(kSegmentPalettes[1], 1)}});
    for (uint32_t i = 0; i < 16400u; ++i) AppendSegmentSprite(s, (8u * i) % 600u, 16u + ((8u * i) / 600u) % 400u);
    const SegmentRun flushed = RunSegments(s, false);
    const SegmentRun merged = RunSegments(s, true);
    CHECK_EQ(flushed.batches.size(), 3u);
    CHECK_EQ(merged.batches.size(), 3u);
    if (flushed.batches.size() != 3u || merged.batches.size() != 3u) return;
    CHECK_EQ(flushed.batches[1].prims.size(), 16383u);
    for (uint32_t i = 0; i < 3u; ++i) {
        CHECK_EQ(PaletteSegmentCount(merged.batches[i]), 1u);
        CheckSegmentsMatchFlushes(flushed.batches, flushed.palettes, merged.batches[i], merged.palettes[i], i);
    }
    CHECK(merged.batches[0].reason == FlushReason::Capacity);
    CHECK(merged.batches[1].reason == FlushReason::Capacity);
}

// A segment's copy is the palette its load produced, whatever the palette memory holds later (07.3 snapshot, IN-353).
void TestPaletteSegmentSnapshot() {
    SegmentWorld world(true);
    std::vector<Qword> s;
    AppendSegmentBaseline(s, kSegmentSpritePrim);
    AppendAd(s, {{reg::kTex0One, SegmentTex0(kSegmentPalettes[0], 1)}});
    AppendSegmentSprite(s, 0, 0);
    Feed(world.world, s);
    const uint32_t generationA = world.world.fe.Clut().Generation();
    std::vector<Qword> second;
    AppendAd(second, {{reg::kTex0One, SegmentTex0(kSegmentPalettes[1], 1)}});
    AppendSegmentSprite(second, 16, 0);
    Feed(world.world, second);
    const uint32_t generationB = world.world.fe.Clut().Generation();
    CHECK(generationB != generationA);
    // The palette memory changes under the pending batch; the copies do not.
    WritePaletteCsm1(world.slm, kSegmentPalettes[0], PaletteC);
    WritePaletteCsm1(world.slm, kSegmentPalettes[1], PaletteC);
    world.world.fe.FlushBatch(FlushReason::Sync);
    CHECK_EQ(world.world.batches.emitted.size(), 1u);
    CHECK(world.hookQuestions == 1u);
    if (world.world.batches.emitted.size() != 1u) return;
    const Batch& batch = world.world.batches.emitted[0];
    CHECK_EQ(batch.paletteSegments.size(), 2u);
    if (batch.paletteSegments.size() != 2u) return;
    CHECK_EQ(batch.paletteSegments[0].clutGeneration, generationA);
    CHECK_EQ(batch.paletteSegments[1].clutGeneration, generationB);
    CHECK_EQ(DecodeSegmentPalette(batch, 0, world.world.fe.Clut()).entries[9], PaletteA(9));
    CHECK_EQ(DecodeSegmentPalette(batch, 1, world.world.fe.Clut()).entries[9], PaletteB(9));
    // A batch without segments reads the CLUT unit's buffer.
    Batch plain = PaletteSegmentBatch(batch, 1);
    plain.paletteSegments.clear();
    CHECK_EQ(DecodeSegmentPalette(plain, 0, world.world.fe.Clut()).entries[9], PaletteB(9));
}

// The IN-366 fast paths with palette segments: random strips, sprites and palette loads (the same texture or another one)
// through both front ends with the hook on give byte-identical batches, segments included.
void TestFastPathsPaletteSegments() {
    int segmented = 0;
    for (uint32_t seed = 1; seed <= 20; ++seed) {
        std::mt19937_64 rng(seed * 104729u);
        std::vector<Qword> stream;
        AppendSegmentBaseline(stream, kSegmentTrianglePrim | (1u << 3));
        for (int packet = 0; packet < 160; ++packet) {
            const uint32_t kind = static_cast<uint32_t>(rng() % 10u);
            if (kind < 4u) {
                const uint32_t nloop = 1u + static_cast<uint32_t>(rng() % 40u);
                const uint32_t primType = rng() % 3u == 0u ? 3u : 4u;
                stream.push_back(Tag(nloop, (rng() & 1u) != 0u, true, primType | (1u << 3) | (1u << primbit::kTme),
                                     kPacked, 3, {0x2, 0x1, 0x4}));
                for (uint32_t v = 0; v < nloop; ++v) {
                    Qword stq;
                    stq.lo = (uint64_t{0x3F000000u + static_cast<uint32_t>(rng() % 0x800000u)}) |
                             (uint64_t{0x3E800000u + static_cast<uint32_t>(rng() % 0x800000u)} << 32);
                    stq.hi = 0x3F800000u + static_cast<uint32_t>(rng() % 0x100000u);
                    Qword rgba;
                    rgba.lo = (rng() & 0xFFu) | ((rng() & 0xFFu) << 32);
                    rgba.hi = (rng() & 0xFFu) | ((rng() & 0xFFu) << 32);
                    Qword xyz;
                    xyz.lo = static_cast<uint64_t>(rng() % 9000u) | (static_cast<uint64_t>(rng() % 6000u) << 32);
                    xyz.hi = (static_cast<uint64_t>(rng() % 4u) << 4) | ((rng() % 4u == 0u) ? (uint64_t{1} << 47) : 0u);
                    stream.push_back(stq);
                    stream.push_back(rgba);
                    stream.push_back(xyz);
                }
            } else if (kind < 8u) {  // a palette load, mostly for the same texture
                const uint32_t tbp = rng() % 6u == 0u ? kSegmentTexture + 0x20u : kSegmentTexture;
                AppendAd(stream, {{reg::kTex0One, SegmentTex0(kSegmentPalettes[rng() % 3u], 1, tbp)}});
            } else {  // sprites
                AppendAd(stream, {{reg::kPrim, kSegmentSpritePrim}});
                for (uint32_t i = 0; i < 1u + rng() % 5u; ++i) {
                    AppendSegmentSprite(stream, static_cast<uint32_t>(rng() % 600u), static_cast<uint32_t>(rng() % 400u));
                }
                AppendAd(stream, {{reg::kPrim, kSegmentTrianglePrim | (1u << 3)}});
            }
        }
        FrontEndOptions slowOptions;
        slowOptions.fastPaths = false;
        SegmentWorld fast(true);
        SegmentWorld slow(true, slowOptions);
        size_t offset = 0;
        while (offset < stream.size()) {
            const size_t take = std::min<size_t>(stream.size() - offset, 1u + static_cast<size_t>(rng() % 2000u));
            fast.world.fe.GifTransfer(1, stream.data() + offset, take);
            slow.world.fe.GifTransfer(1, stream.data() + offset, take);
            offset += take;
        }
        fast.world.fe.VBlank(nullptr, 0);
        slow.world.fe.VBlank(nullptr, 0);
        fast.world.fe.FlushBatch(FlushReason::Sync);
        slow.world.fe.FlushBatch(FlushReason::Sync);
        CHECK(fast.world.log == slow.world.log);
        CHECK_EQ(fast.world.fe.ClutLoadsKept(), slow.world.fe.ClutLoadsKept());
        for (const Batch& batch : fast.world.batches.emitted) segmented += batch.paletteSegments.empty() ? 0 : 1;
    }
    CHECK(segmented > 20);
    std::printf("hw_front_end_fast_paths: %d batches with palette segments\n", segmented);
}

// ---- TASK-168b: batches dropped on arrival ---------------------------------------------------------------------

// A sink that drops every batch on arrival, and one that receives them all.
struct DropEverything : BatchSink {
    uint64_t dropped = 0;
    uint64_t received = 0;
    bool DropOnArrival(const Batch&) override {
        ++dropped;
        return true;
    }
    void OnBatch(Batch&&) override { ++received; }
};
struct ReceiveEverything : BatchSink {
    uint64_t received = 0;
    void OnBatch(Batch&&) override { ++received; }
};

// What a run leaves in the front end: the flush counts, the batch and CLUT counters and the CLUT buffer.
struct FrontEndOutcome {
    std::vector<uint64_t> flushes;
    std::vector<uint64_t> counters;
    std::vector<uint16_t> clut;
    uint64_t dropped = 0;
};

FrontEndOutcome RunWithSink(const std::vector<Qword>& stream, BatchSink& sink) {
    std::vector<uint8_t> slm(kLocalMemoryBytes, 0);
    WritePaletteCsm1(slm, kSegmentPalettes[0], PaletteA);
    WritePaletteCsm1(slm, kSegmentPalettes[1], PaletteB);
    WritePaletteCsm1(slm, 0x40, PaletteC);  // inside the frame's first page row
    NoSync sync;
    NoMemory memory;
    FrontEnd fe(FrontEndOptions{}, sink, sync, memory);
    fe.SetLocalMemory(slm.data());
    fe.GifTransfer(1, stream.data(), stream.size());
    fe.FlushBatch(FlushReason::Sync);
    FrontEndOutcome out;
    for (size_t r = 0; r < static_cast<size_t>(FlushReason::Count); ++r) {
        out.flushes.push_back(fe.FlushReasonCount(static_cast<FlushReason>(r)));
    }
    for (Counter c : {Counter::BatchesEmitted, Counter::BatchesDiscarded, Counter::ClutLoadsRequested,
                      Counter::ClutLoadsPerformed, Counter::ClutLoadsSkipped}) {
        out.counters.push_back(fe.Counters().Get(c));
    }
    out.clut.assign(fe.Clut().Halfwords(), fe.Clut().Halfwords() + clut::kClutHalfwords);
    out.dropped = fe.BatchesDroppedOnArrival();
    return out;
}

// A sink that drops every batch on arrival leaves the front end exactly as one that receives them: the same flushes by
// reason, the same CLUT loads performed and skipped (the dropped batches still make the palette in their frame rows
// stale, chapter 07.3), the same CLUT buffer. Two streams: the segment basics (loads between sprites), and a palette
// inside the frame's rows that a dropped batch makes stale.
void TestDropOnArrival() {
    std::vector<std::vector<Qword>> streams(2);
    AppendSegmentBaseline(streams[0], kSegmentSpritePrim);
    AppendAd(streams[0], {{reg::kTex0One, SegmentTex0(kSegmentPalettes[0], 1)}});
    AppendSegmentSprite(streams[0], 0, 0);
    AppendSegmentSprite(streams[0], 16, 0);
    AppendAd(streams[0], {{reg::kTex0One, SegmentTex0(kSegmentPalettes[1], 1)}});
    AppendSegmentSprite(streams[0], 32, 0);
    AppendAd(streams[0], {{reg::kTex0One, SegmentTex0(kSegmentPalettes[0], 1)}});
    AppendSegmentSprite(streams[0], 48, 0);
    // The palette at block 0x40 lies in the frame's first page row. Its first load is performed; a sprite draws over those
    // rows; another texture ends that batch (its frame rows make the palette stale); the same load is then performed
    // again, and once more right after (no flush in between: redundant).
    AppendSegmentBaseline(streams[1], kSegmentSpritePrim);
    AppendAd(streams[1], {{reg::kTex0One, SegmentTex0(0x40, 1)}});
    AppendSegmentSprite(streams[1], 0, 0);
    AppendAd(streams[1], {{reg::kTex0One, SegmentTex0(0x40, 0, kSegmentTexture + 0x20)}});
    AppendSegmentSprite(streams[1], 16, 0);
    AppendAd(streams[1], {{reg::kTex0One, SegmentTex0(0x40, 1)}});
    AppendSegmentSprite(streams[1], 32, 0);
    AppendAd(streams[1], {{reg::kTex0One, SegmentTex0(0x40, 1)}});
    for (const std::vector<Qword>& stream : streams) {
        ReceiveEverything receiver;
        DropEverything dropper;
        const FrontEndOutcome received = RunWithSink(stream, receiver);
        const FrontEndOutcome dropped = RunWithSink(stream, dropper);
        CHECK(receiver.received > 0);
        CHECK_EQ(dropper.received, 0u);
        CHECK_EQ(dropper.dropped, receiver.received);
        CHECK_EQ(dropped.dropped, receiver.received);
        CHECK_EQ(received.dropped, 0u);
        CHECK(dropped.flushes == received.flushes);
        CHECK(dropped.counters == received.counters);
        CHECK(dropped.clut == received.clut);
        if (&stream == &streams[1]) {
            CHECK_EQ(received.counters[3], 2u);  // the stale palette was loaded again
            CHECK(received.counters[4] >= 1u);   // the last load was redundant
        }
        std::printf("  %llu batches, CLUT loads performed %llu / skipped %llu\n",
                    static_cast<unsigned long long>(receiver.received), static_cast<unsigned long long>(received.counters[3]),
                    static_cast<unsigned long long>(received.counters[4]));
    }
}

int main(int argc, char** argv) {
    const std::string name = argc > 1 ? argv[1] : "";
    struct Entry {
        const char* name;
        void (*fn)();
    };
    const Entry entries[] = {
        {"hw_gif_split", [] { TestGifSplit(); TestGifParserVectors(); }},
        {"hw_gif_ad_xyz", [] { TestAdXyzParser(); TestAdXyzFrontEnd(); }},
        {"hw_flush_rules", [] { TestFlushRules(); TestSignalFlush(); TestVBlankAndReset(); }},
        {"hw_autoflush", [] { TestAutoflush(); }},
        {"hw_vertex_trace", [] { TestVertexTrace(); }},
        {"hw_st_rounding", [] { TestStRounding(); }},
        {"hw_clut_load", [] { TestClutLoad(); }},
        {"hw_clut_snapshot", [] { TestClutSnapshot(); TestPaletteSegmentSnapshot(); }},
        {"hw_clut_copy_free", [] { TestClutCopyFree(); }},
        {"hw_front_end_fast_paths", [] { TestFastPaths(); TestFastPathsPaletteSegments(); }},
        // TASK-168.
        {"hw_palette_segments", [] {
             TestPaletteSegments();
             TestPaletteSegmentRounding();
             TestPaletteSegmentsStillFlush();
             TestPaletteSegmentCapacity();
         }},
        {"hw_front_end_drop_on_arrival", [] { TestDropOnArrival(); }},
        {"hw_block_ranges_overlap", [] { TestBlockRangesOverlap(); }},
        {"hw_transfers", [] { TestTransfers(); TestMipLevelTransferFlush(); }},
        {"hw_counters", [] { TestCounters(); }},
    };
    bool ran = false;
    for (const Entry& entry : entries) {
        if (name == entry.name) {
            entry.fn();
            ran = true;
        }
    }
    if (!ran) {
        std::fprintf(stderr, "unknown test '%s'\n", name.c_str());
        return 2;
    }
    if (g_failures != 0) {
        std::fprintf(stderr, "%s: %d failure(s)\n", name.c_str(), g_failures);
        return 1;
    }
    std::printf("%s: OK\n", name.c_str());
    return 0;
}
