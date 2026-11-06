// Copyright 2026 GT4Recomp contributors. Licensed under the MIT License (see LICENSE).
// Written from the TASK-026 specification (docs/hw-renderer-spec) (clean room).
#include "hw_gs_registers.hpp"

namespace coregs {
namespace {

// Keeps only the chapter 01.1 fields for a context register write.
uint64_t NormaliseAlpha(uint64_t data) {
    // Selector value 3 behaves as 2 (zero for A, B, D; FIX for C).
    for (unsigned shift : {0u, 2u, 4u, 6u}) {
        if (((data >> shift) & 3u) == 3u) {
            data = (data & ~(uint64_t{3} << shift)) | (uint64_t{2} << shift);
        }
    }
    return data & MaskBits(0, 39);
}

// Replaces the TEX0 fields TEX2 may write (PSM, CBP, CPSM, CSM, CSA, CLD).
uint64_t MergeTex2(uint64_t currentTex0, uint64_t data) {
    const uint64_t replaced = MaskBits(20, 25) | MaskBits(37, 63);
    return (currentTex0 & ~replaced) | (data & replaced);
}

// CPSM is masked with 0b1010 (chapter 01.1 normalisation).
uint64_t NormaliseTex0(uint64_t data) {
    const uint64_t cpsmMask = MaskBits(51, 54);
    const uint64_t cpsmValue = (data >> 51) & 0xFu;
    const uint64_t masked = cpsmValue & 0xAu;
    return (data & ~cpsmMask) | (masked << 51);
}

}  // namespace

RegisterFile::RegisterFile() { Reset(); }

void RegisterFile::Reset() {
    env_ = EnvironmentRegs{};
    // PRMODECONT.AC resets to 1: primitive attributes come from PRIM.
    env_.prmodecont = 1;
    context_[0] = ContextRegs{};
    context_[1] = ContextRegs{};
}

uint64_t RegisterFile::EffectivePrim() const {
    // AC = 1: every attribute comes from PRIM. AC = 0: attributes come from
    // PRMODE, except CTXT (bit 9), which always comes from PRIM (drawing context
    // selection does not depend on AC; the specification's PRMODE layout note).
    uint64_t attributes = 0;
    if (env_.prmodecont & 1u) {
        attributes = env_.prim;
    } else {
        attributes = (env_.prmode & ~MaskBits(9, 9)) | (env_.prim & MaskBits(9, 9));
    }
    return (env_.prim & 7u) | (attributes & MaskBits(3, 10));
}

bool RegisterFile::ZbufUsesColourLayout(int context) const {
    return context_[context & 1].zbufUsesColourLayout;
}

RegisterWriteResult RegisterFile::Write(uint8_t address, uint64_t data, CounterSet& counters) {
    RegisterWriteResult result;
    result.recognised = true;

    // Environment registers.
    switch (address) {
        case reg::kPrim:
            env_.prim = data & 0x7FFu;
            return result;
        case reg::kPrmodecont:
            env_.prmodecont = data & 1u;
            if ((data & 1u) == 0u) {
                counters.Add(Counter::PrmodecontAcZero);
            }
            return result;
        case reg::kPrmode:
            env_.prmode = data & MaskBits(3, 10);
            return result;
        case reg::kTexclut: {
            if (((data >> 22) & 3u) != 0u) {
                counters.Add(Counter::TexclutHighBits);
            }
            // COV is masked to 10 bits (V-01-1); CBW and COU are unchanged.
            env_.texclut = data & MaskBits(0, 21);
            return result;
        }
        case reg::kScanmsk:
            env_.scanmsk = data & 3u;
            return result;
        case reg::kTexa:
            env_.texa = data & (MaskBits(0, 7) | MaskBits(15, 15) | MaskBits(32, 39));
            return result;
        case reg::kFogcol:
            env_.fogcol = data & MaskBits(0, 23);
            return result;
        case reg::kDimx:
            env_.dimx = data;
            return result;
        case reg::kDthe:
            env_.dthe = data & 1u;
            return result;
        case reg::kColclamp:
            env_.colclamp = data & 1u;
            return result;
        case reg::kPabe:
            env_.pabe = data & 1u;
            return result;
        case reg::kBitbltbuf:
            env_.bitbltbuf = data;
            return result;
        case reg::kTrxpos:
            env_.trxpos = data;
            return result;
        case reg::kTrxreg:
            env_.trxreg = data;
            return result;
        case reg::kTrxdir:
            env_.trxdir = data & 3u;
            return result;
        default:
            break;
    }

    // Context registers: even addresses are context 1 (_1), odd are context 2 (_2).
    int which = -1;
    int context = -1;
    switch (address) {
        case reg::kTex0One: case reg::kTex0Two: which = 0; break;
        case reg::kClampOne: case reg::kClampTwo: which = 1; break;
        case reg::kTex1One: case reg::kTex1Two: which = 2; break;
        case reg::kTex2One: case reg::kTex2Two: which = 3; break;  // TEX2 path
        case reg::kXyoffsetOne: case reg::kXyoffsetTwo: which = 4; break;
        case reg::kMiptbp1One: case reg::kMiptbp1Two: which = 5; break;
        case reg::kMiptbp2One: case reg::kMiptbp2Two: which = 6; break;
        case reg::kScissorOne: case reg::kScissorTwo: which = 7; break;
        case reg::kAlphaOne: case reg::kAlphaTwo: which = 8; break;
        case reg::kTestOne: case reg::kTestTwo: which = 9; break;
        case reg::kFbaOne: case reg::kFbaTwo: which = 10; break;
        case reg::kFrameOne: case reg::kFrameTwo: which = 11; break;
        case reg::kZbufOne: case reg::kZbufTwo: which = 12; break;
        default:
            result.recognised = false;
            return result;
    }
    // Context pairs are even/odd (_1 even, _2 odd) except TEST, which is 0x47 (_1)
    // and 0x48 (_2) (chapter 01.1 table).
    if (address == reg::kTestOne) {
        context = 0;
    } else if (address == reg::kTestTwo) {
        context = 1;
    } else {
        context = (address & 1u) == 0u ? 0 : 1;
    }
    ContextRegs& ctx = context_[context];
    result.contextScoped = true;
    result.context = context;

    switch (which) {
        case 0: {  // TEX0
            ctx.tex0 = NormaliseTex0(data);
            result.tex0Written = true;
            break;
        }
        case 3: {  // TEX2: only PSM, CBP, CPSM, CSM, CSA, CLD
            ctx.tex0 = NormaliseTex0(MergeTex2(ctx.tex0, data));
            result.tex0Written = true;
            result.tex2Written = true;
            break;
        }
        case 1: ctx.clamp = data; break;
        case 2: ctx.tex1 = data; break;
        case 4: ctx.xyoffset = data & (MaskBits(0, 15) | MaskBits(32, 47)); break;
        case 5: ctx.miptbp1 = data; break;
        case 6: ctx.miptbp2 = data; break;
        case 7: ctx.scissor = data & (MaskBits(0, 10) | MaskBits(16, 26) | MaskBits(32, 42) | MaskBits(48, 58)); break;
        case 8: ctx.alpha = NormaliseAlpha(data); break;
        case 9: ctx.test = data & MaskBits(0, 18); break;
        case 10: ctx.fba = data & 1u; break;
        case 11: {  // FRAME
            uint64_t value = data & (MaskBits(0, 8) | MaskBits(16, 21) | MaskBits(24, 29) | MaskBits(32, 63));
            if (frame::Fbw(value) > 32u) {
                value = (value & ~MaskBits(16, 21)) | (uint64_t{32} << 16);
            }
            const uint32_t format = frame::Psm(value);
            if (format == psm::kT8H || format == psm::kT4Hl || format == psm::kT4Hh) {
                counters.Add(Counter::FramePsm4h);
                uint64_t forced = 0;
                if (format == psm::kT8H) {
                    forced = 0x00FFFFFFu;
                } else if (format == psm::kT4Hl) {
                    forced = 0xF0FFFFFFu;
                } else {
                    forced = 0x0FFFFFFFu;
                }
                // Stored as CT32 (PSM 0) with the forced write mask.
                value = (value & ~(MaskBits(24, 29) | MaskBits(32, 63))) | (forced << 32);
            }
            ctx.frame = value;
            break;
        }
        case 12: ctx.zbuf = data & (MaskBits(0, 8) | MaskBits(24, 27) | MaskBits(32, 32)); break;
        default: break;
    }

    // The ZBUF layout coupling (chapter 01.1) follows the FRAME format of the
    // same context, whichever of the two is written.
    ctx.zbufUsesColourLayout = !IsZPsm(frame::Psm(ctx.frame));
    return result;
}

}  // namespace coregs
