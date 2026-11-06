// HWR register file: environment registers, two drawing contexts and write
// normalisation (chapter 01.1 and chapter 02.2).
// Copyright 2026 GT4Recomp contributors. Licensed under the MIT License (see LICENSE).
// Written from the TASK-026 specification (docs/hw-renderer-spec) (clean room).
#pragma once

#include <cstdint>

#include "hw_gs_counters.hpp"
#include "hw_gs_types.hpp"

namespace coregs {

// Registers shared by both contexts (chapter 02.2 list).
struct EnvironmentRegs {
    uint64_t prim = 0;          // 11 bits
    uint64_t prmode = 0;        // bits 3..10 only
    uint64_t prmodecont = 0;    // AC bit 0
    uint64_t texclut = 0;       // CBW, COU, COV (bits 0..21)
    uint64_t scanmsk = 0;       // bits 0..1
    uint64_t texa = 0;          // TA0, AEM, TA1
    uint64_t fogcol = 0;        // bits 0..23
    uint64_t dimx = 0;          // 64 bits
    uint64_t dthe = 0;          // bit 0
    uint64_t colclamp = 0;      // bit 0
    uint64_t pabe = 0;          // bit 0
    uint64_t bitbltbuf = 0;
    uint64_t trxpos = 0;
    uint64_t trxreg = 0;
    uint64_t trxdir = 0;        // XDIR bits 0..1
};

// Registers held per drawing context (chapter 01.1, _1 and _2 pairs).
struct ContextRegs {
    uint64_t xyoffset = 0;
    uint64_t tex0 = 0;
    uint64_t tex1 = 0;
    uint64_t clamp = 0;
    uint64_t miptbp1 = 0;
    uint64_t miptbp2 = 0;
    uint64_t scissor = 0;
    uint64_t alpha = 0;
    uint64_t test = 0;
    uint64_t fba = 0;
    uint64_t frame = 0;
    uint64_t zbuf = 0;
    // Chapter 01.1 coupling: ZBUF is treated with the colour layout when FRAME
    // uses a Z format (and with the Z layout otherwise). Recomputed on writes.
    bool zbufUsesColourLayout = false;
};

// Field accessors shared by the front end, the CLUT unit and the analysis code.
namespace tex0 {
inline uint32_t Tbp0(uint64_t v) { return static_cast<uint32_t>(BitRange(v, 0, 13)); }
inline uint32_t Tbw(uint64_t v) { return static_cast<uint32_t>(BitRange(v, 14, 19)); }
inline uint32_t Psm(uint64_t v) { return static_cast<uint32_t>(BitRange(v, 20, 25)); }
inline uint32_t Tw(uint64_t v) { return static_cast<uint32_t>(BitRange(v, 26, 29)); }
inline uint32_t Th(uint64_t v) { return static_cast<uint32_t>(BitRange(v, 30, 33)); }
inline uint32_t Tcc(uint64_t v) { return static_cast<uint32_t>(BitRange(v, 34, 34)); }
inline uint32_t Tfx(uint64_t v) { return static_cast<uint32_t>(BitRange(v, 35, 36)); }
inline uint32_t Cbp(uint64_t v) { return static_cast<uint32_t>(BitRange(v, 37, 50)); }
inline uint32_t Cpsm(uint64_t v) { return static_cast<uint32_t>(BitRange(v, 51, 54)); }
inline uint32_t Csm(uint64_t v) { return static_cast<uint32_t>(BitRange(v, 55, 55)); }
inline uint32_t Csa(uint64_t v) { return static_cast<uint32_t>(BitRange(v, 56, 60)); }
inline uint32_t Cld(uint64_t v) { return static_cast<uint32_t>(BitRange(v, 61, 63)); }
}  // namespace tex0

namespace tex1 {
inline uint32_t Lcm(uint64_t v) { return static_cast<uint32_t>(BitRange(v, 0, 0)); }
inline uint32_t Mxl(uint64_t v) { return static_cast<uint32_t>(BitRange(v, 2, 4)); }
inline uint32_t Mmag(uint64_t v) { return static_cast<uint32_t>(BitRange(v, 5, 5)); }
inline uint32_t Mmin(uint64_t v) { return static_cast<uint32_t>(BitRange(v, 6, 8)); }
inline uint32_t Mtba(uint64_t v) { return static_cast<uint32_t>(BitRange(v, 9, 9)); }
inline uint32_t L(uint64_t v) { return static_cast<uint32_t>(BitRange(v, 19, 20)); }
// K is signed 12-bit with 4 fraction bits.
inline int32_t K(uint64_t v) {
    const uint32_t raw = static_cast<uint32_t>(BitRange(v, 32, 43));
    return (raw & 0x800u) ? static_cast<int32_t>(raw | 0xFFFFF000u) : static_cast<int32_t>(raw);
}
}  // namespace tex1

namespace clampreg {
inline uint32_t Wms(uint64_t v) { return static_cast<uint32_t>(BitRange(v, 0, 1)); }
inline uint32_t Wmt(uint64_t v) { return static_cast<uint32_t>(BitRange(v, 2, 3)); }
inline uint32_t Minu(uint64_t v) { return static_cast<uint32_t>(BitRange(v, 4, 13)); }
inline uint32_t Maxu(uint64_t v) { return static_cast<uint32_t>(BitRange(v, 14, 23)); }
inline uint32_t Minv(uint64_t v) { return static_cast<uint32_t>(BitRange(v, 24, 33)); }
inline uint32_t Maxv(uint64_t v) { return static_cast<uint32_t>(BitRange(v, 34, 43)); }
}  // namespace clampreg

namespace frame {
inline uint32_t Fbp(uint64_t v) { return static_cast<uint32_t>(BitRange(v, 0, 8)); }
inline uint32_t Fbw(uint64_t v) { return static_cast<uint32_t>(BitRange(v, 16, 21)); }
inline uint32_t Psm(uint64_t v) { return static_cast<uint32_t>(BitRange(v, 24, 29)); }
inline uint32_t Fbmsk(uint64_t v) { return static_cast<uint32_t>(BitRange(v, 32, 63)); }
}  // namespace frame

namespace zbuf {
inline uint32_t Zbp(uint64_t v) { return static_cast<uint32_t>(BitRange(v, 0, 8)); }
inline uint32_t Psm(uint64_t v) { return static_cast<uint32_t>(BitRange(v, 24, 27)); }
inline uint32_t Zmsk(uint64_t v) { return static_cast<uint32_t>(BitRange(v, 32, 32)); }
}  // namespace zbuf

namespace test {
inline uint32_t Ate(uint64_t v) { return static_cast<uint32_t>(BitRange(v, 0, 0)); }
inline uint32_t Atst(uint64_t v) { return static_cast<uint32_t>(BitRange(v, 1, 3)); }
inline uint32_t Aref(uint64_t v) { return static_cast<uint32_t>(BitRange(v, 4, 11)); }
inline uint32_t Afail(uint64_t v) { return static_cast<uint32_t>(BitRange(v, 12, 13)); }
inline uint32_t Date(uint64_t v) { return static_cast<uint32_t>(BitRange(v, 14, 14)); }
inline uint32_t Datm(uint64_t v) { return static_cast<uint32_t>(BitRange(v, 15, 15)); }
inline uint32_t Zte(uint64_t v) { return static_cast<uint32_t>(BitRange(v, 16, 16)); }
inline uint32_t Ztst(uint64_t v) { return static_cast<uint32_t>(BitRange(v, 17, 18)); }
}  // namespace test

// ATST and ZTST enumerations (chapter 01.1).
constexpr uint32_t kAtstNever = 0;
constexpr uint32_t kZtstNever = 0;
constexpr uint32_t kZtstGequal = 2;
constexpr uint32_t kZtstGreater = 3;
constexpr uint32_t kAfailZbOnly = 2;

// Result of one register write, for the front end's semantic handling.
struct RegisterWriteResult {
    bool recognised = false;
    bool contextScoped = false;
    int context = -1;            // 0 or 1 when contextScoped
    bool tex0Written = false;    // TEX0 or TEX2 (both change TEX0 fields)
    bool tex2Written = false;
};

class RegisterFile {
public:
    RegisterFile();

    // Applies the canonical (chapter 01.1 layout) write for every register the
    // file owns. Returns recognised = false for addresses it does not own
    // (vertex, RGBAQ, ST, UV, FOG, XYZ, A+D, HWREG, SIGNAL/FINISH/LABEL, TEXFLUSH).
    RegisterWriteResult Write(uint8_t address, uint64_t data, CounterSet& counters);

    const EnvironmentRegs& Env() const { return env_; }
    EnvironmentRegs& MutableEnv() { return env_; }
    const ContextRegs& Context(int index) const { return context_[index & 1]; }

    // Effective PRIM attribute bits: type bits from PRIM; attributes from PRIM
    // when PRMODECONT.AC = 1, from PRMODE otherwise (chapter 02.2).
    uint64_t EffectivePrim() const;

    // Returns the value the ZBUF layout depends on (chapter 01.1 coupling).
    bool ZbufUsesColourLayout(int context) const;

    void Reset();

private:
    EnvironmentRegs env_;
    ContextRegs context_[2];
};

}  // namespace coregs
