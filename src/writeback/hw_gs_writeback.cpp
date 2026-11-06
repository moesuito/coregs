// MIT License. Copyright (c) 2026 João Alano and GT4Recomp contributors.
// Written from docs/hw-renderer-spec (TASK-027 M4, chapters 00, 01.5, 02.8, 03.2 to 03.5, 03.7, 03.8, 10.4,
// 12.9, 13.1 to 13.3, 15), clean room (decision 0069).
//
// Translation of one batch into GPU work follows the CPU reference (cpuref/cpuref_draw.cpp) rule for
// rule where it decides WHAT is drawn: scissor and coverage rectangles, texture skip, flat colour
// (last vertex), sprite corners, point cell. Rasterisation itself is done by the GPU (chapter 03.3 vertex
// mapping), so coverage can differ at triangle edges; see IMPLEMENTATION-NOTES IN-500.

#include "hw_gs_writeback.hpp"

#include "../core2/core2_renderer.hpp"

#include <algorithm>
#include <bit>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iterator>
#include <optional>
#include <unordered_map>

#include "hw_batch.hpp"
#include "hw_clut.hpp"
#include "hw_front_end.hpp"
#include "hw_gs_display.hpp"
#include "hw_gs_pixel.hpp"
#include "hw_gs_registers.hpp"
#include "hw_gs_types.hpp"
#include "hw_memory.hpp"
#include "../cpuref/cpuref_draw.hpp"
#include "../cpuref/cpuref_memory.hpp"

namespace coregs::writeback {
namespace {

using pixel::Colour;

constexpr BlockRange kEverything{0, kBlocksTotal};

// Privileged register addresses (chapter 01.7), as the CPU reference defines them.
namespace priv {
constexpr uint32_t kPage = 0x12000000u;
constexpr uint32_t kPmode = kPage + 0x00;
constexpr uint32_t kSmode1 = kPage + 0x10;
constexpr uint32_t kSmode2 = kPage + 0x20;
constexpr uint32_t kSyncv = kPage + 0x60;
constexpr uint32_t kDispfb1 = kPage + 0x70;
constexpr uint32_t kDisplay1 = kPage + 0x80;
constexpr uint32_t kDispfb2 = kPage + 0x90;
constexpr uint32_t kDisplay2 = kPage + 0xA0;
constexpr uint32_t kBgcolor = kPage + 0xE0;
constexpr uint32_t kCsr = kPage + 0x1000;
} // namespace priv

struct Clip {
    int32_t x0 = 0, x1 = -1, y0 = 0, y1 = -1;
};

Clip Intersect(const Clip& a, const Clip& b) {
    return Clip{std::max(a.x0, b.x0), std::min(a.x1, b.x1), std::max(a.y0, b.y0), std::min(a.y1, b.y1)};
}

bool Empty(const Clip& c) { return c.x0 > c.x1 || c.y0 > c.y1; }

Clip ToClip(const PixelRect& r) { return Clip{r.x0, r.x1, r.y0, r.y1}; }

pixel::FrameFormat FrameFormatOf(uint32_t psmCode) {
    switch (psmCode) {
        case 0x01:
        case 0x31:
            return pixel::FrameFormat::Ct24;
        case 0x02:
        case 0x0A:
        case 0x32:
        case 0x3A:
            return pixel::FrameFormat::Ct16;
        default:
            return pixel::FrameFormat::Ct32;
    }
}

pixel::ZFormat ZFormatOf(uint32_t psmCode) {
    switch (psmCode) {
        case 0x31:
            return pixel::ZFormat::Z24;
        case 0x32:
        case 0x3A:
            return pixel::ZFormat::Z16;
        default:
            return pixel::ZFormat::Z32;
    }
}

enum class TexClass : uint8_t { Ct32, Ct24, Ct16, Indexed };

std::optional<TexClass> TextureClassOf(uint32_t psmCode) {
    switch (psmCode) {
        case 0x00:
        case 0x30:
            return TexClass::Ct32;
        case 0x01:
        case 0x31:
            return TexClass::Ct24;
        case 0x02:
        case 0x0A:
        case 0x32:
        case 0x3A:
            return TexClass::Ct16;
        case 0x13:
        case 0x14:
        case 0x1B:
        case 0x24:
        case 0x2C:
            return TexClass::Indexed;
        default:
            return std::nullopt;
    }
}

pixel::DepthCompare DepthCompareOf(uint32_t ztst) {
    switch (ztst) {
        case kZtstNever:
            return pixel::DepthCompare::Never;
        case 1:
            return pixel::DepthCompare::Always;
        case kZtstGequal:
            return pixel::DepthCompare::GreaterEqual;
        default:
            return pixel::DepthCompare::Greater;
    }
}

pixel::AlphaCompare AlphaCompareOf(uint32_t atst) { return static_cast<pixel::AlphaCompare>(atst & 7u); }
pixel::AlphaFail AlphaFailOf(uint32_t afail) { return static_cast<pixel::AlphaFail>(afail & 3u); }

uint32_t PackRgba(const Colour& c) {
    return static_cast<uint32_t>(c.r & 0xFF) | (static_cast<uint32_t>(c.g & 0xFF) << 8) |
           (static_cast<uint32_t>(c.b & 0xFF) << 16) | (static_cast<uint32_t>(c.a & 0xFF) << 24);
}

// Texture unit as the CPU reference builds it (cpuref_draw.cpp BuildTexture).
struct TextureUnit {
    memory::BufferView view{0, 0, memory::Psm::Ct32};
    TexClass cls = TexClass::Ct32;
    uint32_t twLog2 = 0;
    uint32_t thLog2 = 0;
    pixel::WrapAxis wrapS;
    pixel::WrapAxis wrapT;
    bool linear = false;
    bool fst = false;
    bool tcc = false;
    pixel::Tfx tfx = pixel::Tfx::Modulate;
    clut::TexaState texa;
    clut::Palette palette;
};

bool BuildTexture(const BatchState& state, const clut::ClutUnit& clut, TextureUnit& out) {
    const uint64_t tex0Register = state.ctx.tex0;
    const uint64_t tex1Register = state.ctx.tex1;
    const uint64_t clampRegister = state.ctx.clamp;
    const uint32_t psmCode = tex0::Psm(tex0Register);
    const std::optional<TexClass> cls = TextureClassOf(psmCode);
    const auto psm = memory::decodePsm(psmCode);
    if (!cls || !psm) return false;
    out.cls = *cls;
    out.view = memory::BufferView{tex0::Tbp0(tex0Register), tex0::Tbw(tex0Register), *psm};
    out.twLog2 = tex0::Tw(tex0Register);
    out.thLog2 = tex0::Th(tex0Register);
    out.wrapS = pixel::WrapAxis{static_cast<pixel::WrapMode>(clampreg::Wms(clampRegister)),
                                static_cast<int32_t>(clampreg::Minu(clampRegister)),
                                static_cast<int32_t>(clampreg::Maxu(clampRegister))};
    out.wrapT = pixel::WrapAxis{static_cast<pixel::WrapMode>(clampreg::Wmt(clampRegister)),
                                static_cast<int32_t>(clampreg::Minv(clampRegister)),
                                static_cast<int32_t>(clampreg::Maxv(clampRegister))};
    if (!pixel::texture_axis_usable(static_cast<int32_t>(out.twLog2), out.wrapS.mode) ||
        !pixel::texture_axis_usable(static_cast<int32_t>(out.thLog2), out.wrapT.mode)) {
        return false;
    }
    out.linear = tex1::Mmag(tex1Register) != 0;
    out.tcc = tex0::Tcc(tex0Register) != 0;
    out.tfx = static_cast<pixel::Tfx>(tex0::Tfx(tex0Register) & 3u);
    out.texa = clut::TexaState::FromRegister(state.env.texa);
    return true;
}

// The CPU palette of an indexed texture (CLUT buffer through CPSM, CSA and TEXA). Only the CPU decode and the
// palette-ring GLM decode need it; with the GPU CLUT (IN-349) the GPU expands the indices (IN-353: not decoded).
// TASK-168: the CLUT buffer of the batch's palette segment `segment` (the CLUT unit's buffer without segments).
void DecodePalette(const Batch& batch, uint32_t segment, const clut::ClutUnit& clut, TextureUnit& texture) {
    texture.palette = DecodeSegmentPalette(batch, segment, clut);
}

// Colour of one stored texel value after TEXA and palette expansion, packed RGBA8 (the GPU's texel format).
uint32_t ExpandTexel(const TextureUnit& t, uint32_t value) {
    switch (t.cls) {
        case TexClass::Ct32:
            return value;
        case TexClass::Ct24:
            return PackRgba(pixel::expand_texel24(value, static_cast<uint8_t>(t.texa.ta0), t.texa.aem));
        case TexClass::Ct16:
            return PackRgba(pixel::expand_texel16(static_cast<uint16_t>(value), static_cast<uint8_t>(t.texa.ta0),
                                                  static_cast<uint8_t>(t.texa.ta1), t.texa.aem));
        case TexClass::Indexed:
            return t.palette.entries[value & 255u];
    }
    return 0;
}

gpu::GpuVertex ToGpuVertex(const Vertex& v) {
    gpu::GpuVertex g{};
    g.s = v.s;
    g.t = v.t;
    g.rgba[0] = v.r;
    g.rgba[1] = v.g;
    g.rgba[2] = v.b;
    g.rgba[3] = v.a;
    g.q = v.q;
    g.x = static_cast<uint16_t>(static_cast<uint32_t>(v.x4) & 0xFFFFu);
    g.y = static_cast<uint16_t>(static_cast<uint32_t>(v.y4) & 0xFFFFu);
    g.z = v.z;
    g.u = static_cast<uint16_t>(v.u & 0xFFFFu);
    g.v = static_cast<uint16_t>(v.v & 0xFFFFu);
    g.fog = v.fog;
    return g;
}

using Translation = BatchTranslation;

// IN-380: the texels a batch can sample, inclusive: the texture coordinates of its vertices (texture space; the
// interpolated coordinates of a primitive stay between its vertices', perspective included when every Q has the
// same sign) widened by one texel for rounding and bilinear neighbours, then put through the wrap mode: CLAMP
// clamps to the texture, REGION_CLAMP to its region; REPEAT that leaves the texture and REGION_REPEAT give the whole
// texture. A texture taken from a resident target only needs this rectangle loaded, not the whole power-of-two
// texture (GT4's full-screen passes sample a 640 x 448 frame through a 1024 x 512 texture).
PixelRect TextureFootprint(const Batch& batch, const TextureUnit& t, uint32_t width, uint32_t height) {
    const PixelRect whole{0, 0, static_cast<int32_t>(width) - 1, static_cast<int32_t>(height) - 1};
    if (batch.vertices.empty()) return whole;
    double uMin = 0, uMax = 0, vMin = 0, vMax = 0;
    bool first = true;
    bool positiveQ = false, negativeQ = false;
    for (const Vertex& vertex : batch.vertices) {
        double u = 0, v = 0;
        if (t.fst) {
            u = vertex.u / 16.0;
            v = vertex.v / 16.0;
        } else {
            if (!(vertex.q == vertex.q) || vertex.q == 0.0f) return whole;  // NaN or a zero Q: unknown
            positiveQ = positiveQ || vertex.q > 0.0f;
            negativeQ = negativeQ || vertex.q < 0.0f;
            u = static_cast<double>(vertex.s) / vertex.q * width;
            v = static_cast<double>(vertex.t) / vertex.q * height;
            if (!(u == u) || !(v == v) || std::abs(u) > 1e6 || std::abs(v) > 1e6) return whole;
        }
        uMin = first ? u : std::min(uMin, u);
        uMax = first ? u : std::max(uMax, u);
        vMin = first ? v : std::min(vMin, v);
        vMax = first ? v : std::max(vMax, v);
        first = false;
    }
    if (positiveQ && negativeQ) return whole;
    auto axis = [](double low, double high, int32_t size, const pixel::WrapAxis& wrap, int32_t& out0, int32_t& out1) {
        int32_t a = static_cast<int32_t>(std::floor(low)) - 1;
        int32_t b = static_cast<int32_t>(std::ceil(high)) + 1;
        if (wrap.mode == pixel::WrapMode::Clamp) {
            a = std::clamp(a, 0, size - 1);
            b = std::clamp(b, 0, size - 1);
        } else if (wrap.mode == pixel::WrapMode::RegionClamp && wrap.minimum >= 0 && wrap.minimum <= wrap.maximum &&
                   wrap.maximum < size) {
            a = std::clamp(a, wrap.minimum, wrap.maximum);  // REGION_CLAMP: every sample lies in [MIN, MAX]
            b = std::clamp(b, wrap.minimum, wrap.maximum);
        } else if (wrap.mode != pixel::WrapMode::Repeat || a < 0 || b > size - 1) {
            a = 0;  // region modes, or REPEAT that wraps: the whole axis
            b = size - 1;
        }
        out0 = a;
        out1 = b;
    };
    PixelRect footprint;
    axis(uMin, uMax, static_cast<int32_t>(width), t.wrapS, footprint.x0, footprint.x1);
    axis(vMin, vMax, static_cast<int32_t>(height), t.wrapT, footprint.y0, footprint.y1);
    return footprint;
}

// Texel rectangle needed by the region wrap modes may exceed 2^TW; the decoded image covers it.
bool TextureExtent(const TextureUnit& t, uint32_t& width, uint32_t& height) {
    width = 1u << t.twLog2;
    height = 1u << t.thLog2;
    if (t.wrapS.mode == pixel::WrapMode::RegionClamp || t.wrapS.mode == pixel::WrapMode::RegionRepeat) {
        width = std::max<uint32_t>(width, static_cast<uint32_t>(std::max(t.wrapS.maximum, 0)) + 1u);
    }
    if (t.wrapT.mode == pixel::WrapMode::RegionClamp || t.wrapT.mode == pixel::WrapMode::RegionRepeat) {
        height = std::max<uint32_t>(height, static_cast<uint32_t>(std::max(t.wrapT.maximum, 0)) + 1u);
    }
    return width <= gpu::kGpuMaxTextureSide && height <= gpu::kGpuMaxTextureSide;
}

// Wall-clock stage timer of one pass (roadmap requirement 4). CPU only: the GPU timestamp pool is a
// later step, and PassTimings::executeGpuUs stays 0 so a reader never mistakes it for a measurement.
class StageTimer {
public:
    explicit StageTimer(double& accumulator) : accumulator_(accumulator), start_(Clock::now()) {}
    ~StageTimer() {
        const std::chrono::duration<double, std::micro> elapsed = Clock::now() - start_;
        accumulator_ += elapsed.count();
    }
    StageTimer(const StageTimer&) = delete;
    StageTimer& operator=(const StageTimer&) = delete;

private:
    using Clock = std::chrono::steady_clock;
    double& accumulator_;
    Clock::time_point start_;
};




uint64_t HashPalette(const clut::Palette& palette) {
    uint64_t hash = 1469598103934665603ull;
    for (uint32_t i = 0; i < 256; ++i) {
        hash ^= palette.entries[i];
        hash *= 1099511628211ull;
    }
    return hash;
}

void Translate(const Batch& batch, const memory::LocalMemory& local, const clut::ClutUnit& clut, Translation& out,
               Counters& counters, const TextureReadHook& beforeTextureRead = {},
               TextureCacheView* textureCache = nullptr, const TextureFromTargetHook& textureFromTarget = {},
               const TextureFromGlmHook& textureFromGlm = {}) {
    const BatchState& state = batch.state;
    const EnvironmentRegs& env = state.env;
    const ContextRegs& ctx = state.ctx;
    const uint64_t prim = state.effectivePrim;

    const bool flat = ((prim >> primbit::kIip) & 1u) == 0;
    const bool fogEnabled = ((prim >> primbit::kFge) & 1u) != 0;
    const bool blendEnabled = ((prim >> primbit::kAbe) & 1u) != 0;
    const Colour fogColour{static_cast<int32_t>(Field(env.fogcol, 0, 8)), static_cast<int32_t>(Field(env.fogcol, 8, 8)),
                           static_cast<int32_t>(Field(env.fogcol, 16, 8)), 0};
    const Clip scissor{static_cast<int32_t>(Field(ctx.scissor, 0, 11)), static_cast<int32_t>(Field(ctx.scissor, 16, 11)),
                       static_cast<int32_t>(Field(ctx.scissor, 32, 11)), static_cast<int32_t>(Field(ctx.scissor, 48, 11))};
    const Clip surface{0, gpu::kGpuSurface - 1, 0, gpu::kGpuSurface - 1};
    {
        const Clip visible = Intersect(scissor, surface);
        out.scissor = Empty(visible) ? PixelRect{} : PixelRect{visible.x0, visible.y0, visible.x1, visible.y1};
    }

    const uint32_t framePsmCode = frame::Psm(ctx.frame);
    const auto framePsm = memory::decodePsm(framePsmCode);
    out.frameView = memory::BufferView{memory::blockOfPage(frame::Fbp(ctx.frame)), frame::Fbw(ctx.frame),
                                       framePsm ? *framePsm : memory::Psm::Ct32};
    const uint32_t zbufPsmCode = 0x30u | zbuf::Psm(ctx.zbuf);
    const auto zPsm = memory::decodePsm(zbufPsmCode);
    out.zView = memory::BufferView{memory::blockOfPage(zbuf::Zbp(ctx.zbuf)), frame::Fbw(ctx.frame),
                                   zPsm ? *zPsm : memory::Psm::Z32};

    // Pixel state, computed exactly as the CPU reference sets it (01.5, 08).
    pixel::PixelState ps;
    ps.frame.format = FrameFormatOf(framePsmCode);
    ps.frame.fbmsk = frame::Fbmsk(ctx.frame);
    ps.frame.fba = (ctx.fba & 1u) != 0;
    ps.frame.dither = (env.dthe & 1u) != 0;
    ps.frame.dimx = env.dimx;
    ps.alpha_test.enabled = test::Ate(ctx.test) != 0;
    ps.alpha_test.compare = AlphaCompareOf(test::Atst(ctx.test));
    ps.alpha_test.reference = static_cast<uint8_t>(test::Aref(ctx.test));
    ps.alpha_test.fail = AlphaFailOf(test::Afail(ctx.test));
    ps.date.enabled = test::Date(ctx.test) != 0;
    ps.date.expected_msb = test::Datm(ctx.test) != 0;
    ps.depth.test_enabled = test::Zte(ctx.test) != 0;
    ps.depth.compare = DepthCompareOf(test::Ztst(ctx.test));
    ps.depth.write_masked = zbuf::Zmsk(ctx.zbuf) != 0;
    ps.z_format = ZFormatOf(zbufPsmCode);
    pixel::decode_alpha_register(ctx.alpha, ps.blend);
    ps.blend.enabled = blendEnabled;
    ps.blend.pabe = (env.pabe & 1u) != 0;
    ps.blend.colour_clamp = (env.colclamp & 1u) != 0;

    TextureUnit texture;
    const bool tme = ((prim >> primbit::kTme) & 1u) != 0;
    if (tme) {
        texture.fst = ((prim >> primbit::kFst) & 1u) != 0;
        if (!BuildTexture(state, clut, texture)) {
            out.skippedTexture = true;
            return;
        }
        uint32_t width = 0;
        uint32_t height = 0;
        if (!TextureExtent(texture, width, height)) {
            out.skippedTexture = true;
            return;
        }
        out.hasTexture = true;
        out.texWidth = width;
        out.texHeight = height;
        const bool rawKind = texture.cls == TexClass::Ct32 || texture.cls == TexClass::Ct24 || texture.cls == TexClass::Ct16;
        if (rawKind && textureFromTarget) {
            const PixelRect footprint = TextureFootprint(batch, texture, width, height);
            out.textureSlot = textureFromTarget(texture.view, width, height, footprint);
            // IN-384: the scaled copy of a texture taken from a slot only needs the texels the batch can sample (IN-380),
            // so a 1024 x 512 texture view of a 640 x 448 frame does not grow the slot's scaled image.
            if (out.textureSlot >= 0 && !footprint.Empty()) {
                out.texCopyWidth = static_cast<uint32_t>(footprint.x1 + 1);
                out.texCopyHeight = static_cast<uint32_t>(footprint.y1 + 1);
            }
        }
        bool gpuClut = false;
        if (out.textureSlot < 0 && textureFromGlm &&
            textureFromGlm(texture.view, width, height, texture.cls == TexClass::Indexed, gpuClut)) {
            out.glmTexture = true;
            out.glmGpuClut = gpuClut && texture.cls == TexClass::Indexed;
            out.glmCpsm = tex0::Cpsm(state.ctx.tex0);
            out.glmCsa = tex0::Csa(state.ctx.tex0);
            out.glmTa0 = texture.texa.ta0;
            out.glmTa1 = texture.texa.ta1;
            out.glmAem = texture.texa.aem ? 1u : 0u;
            out.glmBp = texture.view.basePointer;
            out.glmBw = texture.view.bufferWidth;
            out.glmPsm = static_cast<uint32_t>(texture.view.psm);
            out.glmIndexed = texture.cls == TexClass::Indexed;
            if (out.glmIndexed && !out.glmGpuClut) {
                DecodePalette(batch, 0, clut, texture);
                out.palette.assign(texture.palette.entries, texture.palette.entries + 256);
                // TASK-168: every segment's palette, for a palette row each.
                const uint32_t segments = PaletteSegmentCount(batch);
                if (segments > 1u) {
                    out.segmentPalettes.reserve(256u * segments);
                    for (uint32_t segment = 0; segment < segments; ++segment) {
                        const clut::Palette palette = DecodeSegmentPalette(batch, segment, clut);
                        out.segmentPalettes.insert(out.segmentPalettes.end(), palette.entries, palette.entries + 256);
                    }
                }
            }
            counters.texturesFromGlm++;
        }
        // TASK-168: a texture expanded with one palette (the CPU decode below, or the GPU CLUT) cannot draw a batch with
        // several palettes.
        if (texture.cls == TexClass::Indexed && PaletteSegmentCount(batch) > 1u && (!out.glmTexture || out.glmGpuClut)) {
            out.paletteSegmentsUnsupported = true;
            out.skippedTexture = true;
            return;
        }
        if (out.textureSlot >= 0 || out.glmTexture) {
            // Raw stored words on the GPU; the pixel shader expands CT16/CT24 with TEXA (IN-346, IN-347).
            if (out.textureSlot >= 0) counters.texturesFromTargets++;
        } else {
        if (beforeTextureRead) beforeTextureRead(texture.view, width, height);
        if (texture.cls == TexClass::Indexed) DecodePalette(batch, 0, clut, texture);
        TextureKey key;
        key.tbp = texture.view.basePointer;
        key.tbw = texture.view.bufferWidth;
        key.psm = static_cast<uint32_t>(texture.view.psm);
        key.width = width;
        key.height = height;
        if (texture.cls == TexClass::Ct16 || texture.cls == TexClass::Ct24) {
            key.ta0 = texture.texa.ta0;
            key.ta1 = texture.texa.ta1;
            key.aem = texture.texa.aem ? 1u : 0u;
        }
        if (texture.cls == TexClass::Indexed) key.paletteHash = HashPalette(texture.palette);
        const std::vector<uint32_t>* cached = textureCache ? textureCache->find(key, texture.view) : nullptr;
        if (cached) {
            out.texels = *cached;
            counters.texturesCached++;
        } else {
            StageTimer decodeTimer(counters.textureDecodeUs);
            out.texels.resize(static_cast<size_t>(width) * height);
            local.readRect(texture.view, 0, 0, width, height, out.texels.data());  // stored values (IN-343)
            for (uint32_t& texel : out.texels) texel = ExpandTexel(texture, texel);
            counters.texturesDecoded++;
            counters.texelsDecoded += static_cast<uint64_t>(width) * height;
            if (textureCache) textureCache->store(key, texture.view, out.texels);
        }
        }
    }

    // GPU pixel-state block (hw_pixel_uber.frag). Field values equal the CPU reference's PixelState.
    gpu::GpuPixelState& st = out.state;
    st.fields[gpu::gpufield::Format] = static_cast<uint32_t>(ps.frame.format);
    st.fields[gpu::gpufield::Fbmsk] = ps.frame.fbmsk;
    st.fields[gpu::gpufield::Fba] = ps.frame.fba ? 1u : 0u;
    st.fields[gpu::gpufield::Dither] = ps.frame.dither ? 1u : 0u;
    st.fields[gpu::gpufield::DimxLo] = static_cast<uint32_t>(ps.frame.dimx);
    st.fields[gpu::gpufield::DimxHi] = static_cast<uint32_t>(ps.frame.dimx >> 32);
    st.fields[gpu::gpufield::Ate] = ps.alpha_test.enabled ? 1u : 0u;
    st.fields[gpu::gpufield::Atst] = static_cast<uint32_t>(ps.alpha_test.compare);
    st.fields[gpu::gpufield::Afail] = static_cast<uint32_t>(ps.alpha_test.fail);
    st.fields[gpu::gpufield::Aref] = ps.alpha_test.reference;
    st.fields[gpu::gpufield::DateOn] = ps.date.enabled ? 1u : 0u;
    st.fields[gpu::gpufield::Datm] = ps.date.expected_msb ? 1u : 0u;
    st.fields[gpu::gpufield::Ztest] = ps.depth.test_enabled ? 1u : 0u;
    st.fields[gpu::gpufield::Ztst] = static_cast<uint32_t>(ps.depth.compare);
    st.fields[gpu::gpufield::Zmsk] = ps.depth.write_masked ? 1u : 0u;
    st.fields[gpu::gpufield::Zformat] = static_cast<uint32_t>(ps.z_format);
    st.fields[gpu::gpufield::BlendOn] = ps.blend.enabled ? 1u : 0u;
    st.fields[gpu::gpufield::Pabe] = ps.blend.pabe ? 1u : 0u;
    st.fields[gpu::gpufield::BlendA] = static_cast<uint32_t>(ps.blend.a);
    st.fields[gpu::gpufield::BlendB] = static_cast<uint32_t>(ps.blend.b);
    st.fields[gpu::gpufield::BlendC] = static_cast<uint32_t>(ps.blend.c);
    st.fields[gpu::gpufield::BlendD] = static_cast<uint32_t>(ps.blend.d);
    st.fields[gpu::gpufield::Fix] = ps.blend.fix;
    st.fields[gpu::gpufield::Colclamp] = ps.blend.colour_clamp ? 1u : 0u;
    st.fields[gpu::gpufield::TexOn] = tme ? 1u : 0u;
    st.fields[gpu::gpufield::Tfx] = static_cast<uint32_t>(texture.tfx);
    st.fields[gpu::gpufield::Tcc] = texture.tcc ? 1u : 0u;
    // RGBA8 words expanded on the CPU, or raw words copied from a resident target and expanded in the shader
    // with the TEXA fields (IN-346; HW_TEXEL_* of hw_pixel_core.glsl: 0 RGBA8, 1 CT16, 2 CT24).
    st.fields[gpu::gpufield::TexelKind] = 0;
    if (out.textureSlot >= 0 || (out.glmTexture && !out.glmIndexed)) {
        st.fields[gpu::gpufield::TexelKind] = texture.cls == TexClass::Ct16 ? 1u : texture.cls == TexClass::Ct24 ? 2u : 0u;
    }
    st.fields[gpu::gpufield::Ta0] = texture.texa.ta0;
    st.fields[gpu::gpufield::Ta1] = texture.texa.ta1;
    st.fields[gpu::gpufield::Aem] = texture.texa.aem ? 1u : 0u;
    st.fields[gpu::gpufield::FogOn] = fogEnabled ? 1u : 0u;
    st.fields[gpu::gpufield::FogR] = static_cast<uint32_t>(fogColour.r);
    st.fields[gpu::gpufield::FogG] = static_cast<uint32_t>(fogColour.g);
    st.fields[gpu::gpufield::FogB] = static_cast<uint32_t>(fogColour.b);
    st.fields[gpu::gpufield::TexFst] = texture.fst ? 1u : 0u;
    st.fields[gpu::gpufield::TexLinear] = texture.linear ? 1u : 0u;
    st.fields[gpu::gpufield::WrapS] = static_cast<uint32_t>(texture.wrapS.mode);
    st.fields[gpu::gpufield::WrapT] = static_cast<uint32_t>(texture.wrapT.mode);
    st.fields[gpu::gpufield::WrapMinS] = static_cast<uint32_t>(texture.wrapS.minimum);
    st.fields[gpu::gpufield::WrapMaxS] = static_cast<uint32_t>(texture.wrapS.maximum);
    st.fields[gpu::gpufield::WrapMinT] = static_cast<uint32_t>(texture.wrapT.minimum);
    st.fields[gpu::gpufield::WrapMaxT] = static_cast<uint32_t>(texture.wrapT.maximum);
    st.fields[gpu::gpufield::TexLog2W] = texture.twLog2;
    st.fields[gpu::gpufield::TexLog2H] = texture.thLog2;
    out.blended = blendEnabled;

    // Primitives, in stream order (chapter 03.2). Each keeps its own scissor (the CPU clip).
    bool haveRect = false;
    // TASK-168: the palette segment of each draw, when the batch has several palettes and they reach the GPU per draw.
    const bool segmentPerDraw = !out.segmentPalettes.empty();
    uint32_t segment = 0;
    uint32_t segmentEnd = PaletteSegmentEndPrim(batch, 0);
    for (uint32_t primIndex = 0; primIndex < batch.prims.size(); ++primIndex) {
        const PrimRecord& record = batch.prims[primIndex];
        while (primIndex >= segmentEnd && segment + 1u < PaletteSegmentCount(batch)) {
            ++segment;
            segmentEnd = PaletteSegmentEndPrim(batch, segment);
        }
        counters.primitives++;
        if (record.rect.Empty()) continue;
        // IN-391: never clip a primitive with its own draw rectangle. The reference computes one rectangle per DRAW and
        // uses it (expanded by one native pixel) only for target sizing and dirty-rect work; it never scissors geometry
        // with it. Ours used the per-primitive rectangle of chapter 03.1 step 9 as the
        // scissor, and that rectangle's exclusive end rounds DOWN (floor(max + 0.6)), so it is one pixel tighter than
        // the rasteriser's coverage: the last pixel row/column of every primitive is cut away. Invisible at 1x, a
        // one-native-pixel notch at scale - the owner's bitten wheel rims, the dialog border notches and the same cut on
        // the car in a race. One native pixel of slack per side restores the reference's rule.
        // IN-397: the slack belongs to the SCALED pass only. The front end's rectangle is the exact 1x coverage (the
        // first and last pixel inside [min, max)), so the 1x draw and the drawn/dirty rectangle keep it; at scale the
        // 06.3 shift puts up to half a native pixel of real sub-pixel coverage past it, which the scaled scissor must
        // not cut. IN-391 widened the drawn rectangle too: GT4's per-frame colour filter draws exact 32-row pages, the
        // extra row reached the next page row, the temporary page overlapped the frame and every race frame lost its
        // scaled image (the owner's 1x races from candidate 276). COREGS_HW_PRIM_SLACK=<n> sets the scaled slack (A/B).
        static const int32_t slack =
            std::getenv("COREGS_HW_PRIM_SLACK") != nullptr ? std::atoi(std::getenv("COREGS_HW_PRIM_SLACK")) : 1;
        const Clip clip = Intersect(Intersect(ToClip(record.rect), scissor), surface);
        if (Empty(clip)) continue;
        PixelRect widened = record.rect;
        widened.x0 -= slack;
        widened.y0 -= slack;
        widened.x1 += slack;
        widened.y1 += slack;
        const Clip scaledClip = Intersect(Intersect(ToClip(widened), scissor), surface);
        const Vertex* v = &batch.vertices[record.first];
        const uint32_t firstVertex = static_cast<uint32_t>(out.vertices.size());
        gpu::GpuDraw draw;
        draw.x0 = clip.x0;
        draw.y0 = clip.y0;
        draw.x1 = clip.x1;
        draw.y1 = clip.y1;
        draw.scaledX0 = scaledClip.x0;
        draw.scaledY0 = scaledClip.y0;
        draw.scaledX1 = scaledClip.x1;
        draw.scaledY1 = scaledClip.y1;
        draw.first = firstVertex;
        switch (record.type) {
            case PrimType::Point: {
                // Window pixel cell: the vertex is moved to the pixel corner (IN-300).
                Vertex p = v[0];
                p.x4 = (p.x4 >> 4) << 4;
                p.y4 = (p.y4 >> 4) << 4;
                out.vertices.push_back(ToGpuVertex(p));
                draw.topology = gpu::GpuTopology::Points;
                draw.count = 1;
                break;
            }
            case PrimType::Line:
            case PrimType::LineStrip: {
                Vertex a = v[0];
                Vertex b = v[1];
                if (flat) {
                    a.r = b.r; a.g = b.g; a.b = b.b; a.a = b.a;  // last vertex (de-index, IN-212)
                    counters.flatDeindexedPrimitives++;
                }
                const gpu::GpuVertex line[2] = {ToGpuVertex(a), ToGpuVertex(b)};
                out.vertices.insert(out.vertices.end(), std::begin(line), std::end(line));
                draw.topology = gpu::GpuTopology::Lines;
                draw.count = 2;
                break;
            }
            case PrimType::Triangle:
            case PrimType::TriangleStrip:
            case PrimType::TriangleFan: {
                Vertex a = v[0];
                Vertex b = v[1];
                Vertex c = v[2];
                if (flat) {
                    a.r = c.r; a.g = c.g; a.b = c.b; a.a = c.a;  // last vertex (de-index, IN-212)
                    b.r = c.r; b.g = c.g; b.b = c.b; b.a = c.a;
                    counters.flatDeindexedPrimitives++;
                }
                // TASK-134: the primitive's vertices go in with one insert (one capacity check and one copy), not one
                // push_back each.
                const gpu::GpuVertex triangle[3] = {ToGpuVertex(a), ToGpuVertex(b), ToGpuVertex(c)};
                out.vertices.insert(out.vertices.end(), std::begin(triangle), std::end(triangle));
                draw.topology = gpu::GpuTopology::Triangles;
                draw.count = 3;
                break;
            }
            case PrimType::Sprite: {
                // Chapter 03.2: corner c takes X and S/U from (c & 1 ? second : first), Y and T/V from
                // (c & 2 ? second : first); colour, Z, fog and Q always from the second vertex.
                const Vertex& first = v[0];
                const Vertex& second = v[1];
                Vertex corner[4];
                for (int c = 0; c < 4; ++c) {
                    const Vertex& horizontal = (c & 1) ? second : first;
                    const Vertex& vertical = (c & 2) ? second : first;
                    Vertex q = second;  // colour, Z, fog, Q
                    q.x4 = horizontal.x4;
                    q.s = horizontal.s;
                    q.u = horizontal.u;
                    q.y4 = vertical.y4;
                    q.t = vertical.t;
                    q.v = vertical.v;
                    corner[c] = q;
                }
                const gpu::GpuVertex corners[4] = {ToGpuVertex(corner[0]), ToGpuVertex(corner[1]), ToGpuVertex(corner[2]),
                                                   ToGpuVertex(corner[3])};
                const gpu::GpuVertex sprite[6] = {corners[0], corners[1], corners[2], corners[1], corners[2], corners[3]};
                out.vertices.insert(out.vertices.end(), std::begin(sprite), std::end(sprite));
                draw.topology = gpu::GpuTopology::Triangles;
                draw.count = 6;
                counters.spritesExpanded++;
                break;
            }
            case PrimType::Reserved:
                continue;
        }
        out.draws.push_back(draw);
        if (segmentPerDraw) out.drawSegments.push_back(segment);
        if (!haveRect) {
            out.rect = PixelRect{clip.x0, clip.y0, clip.x1, clip.y1};
            haveRect = true;
        } else {
            out.rect = UnionRects(out.rect, PixelRect{clip.x0, clip.y0, clip.x1, clip.y1});
        }
    }
}

// ---- Block spans (M7 coherence, IN-340) ------------------------------------------------------------
// A conservative range of GS blocks (modulo 16384) that a rectangle of a buffer can touch: whole page rows
// from the row of y0 to the row of y1, plus one page when the base is not page aligned and one page row per
// buffer width the rectangle extends past (the row wraps). count >= 16384 means "all memory".
struct BlockSpan {
    uint32_t begin = 0;
    uint32_t count = 0;
};

constexpr uint32_t kAllBlocks = kBlocksTotal;

// IN-383: block offsets of the eight 16x16 quadrants of a CT32 page (01.3 block table, 07.4), quadrant q = qx + 4 * qy.
constexpr uint32_t kToneQuadrantBlock[8] = {0, 4, 16, 20, 8, 12, 24, 28};

BlockSpan SpanOfRect(const memory::BufferView& view, int32_t x0, int32_t y0, int32_t x1, int32_t y1) {
    if (x1 < x0 || y1 < y0) return BlockSpan{};
    if (x0 < 0 || y0 < 0 || x1 >= 2048 || y1 >= 2048) return BlockSpan{0, kAllBlocks};
    const memory::PsmInfo& info = memory::psmInfo(view.psm);
    const uint32_t bufferWidth = std::max<uint32_t>(view.bufferWidth, 1u) * 64u;
    const uint32_t pagesPerRow = std::max<uint32_t>(1u, (bufferWidth + info.pageWidth - 1u) / info.pageWidth);
    const uint32_t firstRow = static_cast<uint32_t>(y0) / info.pageHeight;
    uint32_t rows = static_cast<uint32_t>(y1) / info.pageHeight - firstRow + 1u;
    if (static_cast<uint32_t>(x1) >= bufferWidth) rows += 1u + static_cast<uint32_t>(x1) / bufferWidth;
    // One extra page only when the base is not page aligned (the rectangle then spills into the next page).
    const uint64_t spill = (view.basePointer % kBlocksPerPage) != 0 ? kBlocksPerPage : 0;
    const uint64_t count = static_cast<uint64_t>(rows) * pagesPerRow * kBlocksPerPage + spill;
    if (count >= kAllBlocks) return BlockSpan{0, kAllBlocks};
    const uint32_t begin = (view.basePointer + firstRow * pagesPerRow * kBlocksPerPage) % kAllBlocks;
    return BlockSpan{begin, static_cast<uint32_t>(count)};
}

BlockSpan SpanOfRect(const memory::BufferView& view, const PixelRect& r) { return SpanOfRect(view, r.x0, r.y0, r.x1, r.y1); }

bool SpansOverlap(const BlockSpan& a, const BlockSpan& b) {
    if (a.count == 0 || b.count == 0) return false;
    if (a.count >= kAllBlocks || b.count >= kAllBlocks) return true;
    const uint32_t ab = (b.begin + kAllBlocks - a.begin) % kAllBlocks;
    const uint32_t ba = (a.begin + kAllBlocks - b.begin) % kAllBlocks;
    return ab < a.count || ba < b.count;
}

// Span of a transfer rectangle (02.7): coordinates wrap at 2048, which the conservative span reports as all
// memory.
BlockSpan SpanOfTransfer(uint32_t bp, uint32_t bw, uint32_t psmCode, uint32_t x, uint32_t y, uint32_t w, uint32_t h) {
    const auto psm = memory::decodePsm(psmCode);
    if (!psm) return BlockSpan{0, kAllBlocks};
    if (w == 0 || h == 0) return BlockSpan{};
    return SpanOfRect(memory::BufferView{bp, bw, *psm}, static_cast<int32_t>(x), static_cast<int32_t>(y),
                      static_cast<int32_t>(x + w - 1u), static_cast<int32_t>(y + h - 1u));
}

bool SameView(const memory::BufferView& a, const memory::BufferView& b) {
    return a.basePointer == b.basePointer && a.bufferWidth == b.bufferWidth && a.psm == b.psm;
}

bool Contains(const PixelRect& outer, const PixelRect& inner) {
    return !outer.Empty() && outer.x0 <= inner.x0 && outer.y0 <= inner.y0 && outer.x1 >= inner.x1 && outer.y1 >= inner.y1;
}

// The batch reads or writes depth (01.5): a depth test other than ALWAYS, or depth writes not masked. A
// batch that does neither leaves the depth image untouched, so no depth target is needed for it.
bool DepthUsed(const gpu::GpuPixelState& st) {
    const bool testReads = st.fields[gpu::gpufield::Ztest] != 0 &&
                           st.fields[gpu::gpufield::Ztst] != static_cast<uint32_t>(pixel::DepthCompare::Always);
    return testReads || st.fields[gpu::gpufield::Zmsk] == 0;
}


} // namespace

bool ExecuteBatch(gpu::GpuDrawer &gpu, const BatchTranslation &t, memory::LocalMemory &local, Counters &counters,
                  std::string &error) {
    if (t.draws.empty()) return true;
    const std::int32_t rectX0 = t.rect.x0, rectY0 = t.rect.y0, rectX1 = t.rect.x1, rectY1 = t.rect.y1;
    const std::size_t words =
        static_cast<std::size_t>(rectX1 - rectX0 + 1) * static_cast<std::size_t>(rectY1 - rectY0 + 1);
    // Chunks of whole draws, each within the vertex buffer limit. Every chunk reads and writes the same
    // rectangle, so a later chunk sees the earlier chunk's pixels through the SLM.
    std::size_t begin = 0;
    while (begin < t.draws.size()) {
        std::size_t end = begin;
        std::uint32_t vertexCount = 0;
        while (end < t.draws.size() && vertexCount + t.draws[end].count <= gpu::kGpuMaxVertices) {
            vertexCount += t.draws[end].count;
            ++end;
        }
        if (end == begin) {
            error = "a primitive exceeds the vertex buffer";
            return false;
        }
        const std::uint32_t baseVertex = t.draws[begin].first;
        gpu::GpuChunk chunk;
        chunk.rectX0 = rectX0;
        chunk.rectY0 = rectY0;
        chunk.rectX1 = rectX1;
        chunk.rectY1 = rectY1;
        chunk.colourSlot = gpu::kGpuScratchColourSlot;  // the write-back mode never touches resident targets
        chunk.depthSlot = gpu::kGpuScratchDepthSlot;
        chunk.state = t.state;
        chunk.hasTexture = t.hasTexture;
        chunk.textureSlot = t.textureSlot;
        chunk.glmTexture = t.glmTexture;
        chunk.glmBp = t.glmBp;
        chunk.glmBw = t.glmBw;
        chunk.glmPsm = t.glmPsm;
        chunk.glmIndexed = t.glmIndexed;
        chunk.glmGpuClut = t.glmGpuClut;
        chunk.glmCpsm = t.glmCpsm;
        chunk.glmCsa = t.glmCsa;
        chunk.glmTa0 = t.glmTa0;
        chunk.glmTa1 = t.glmTa1;
        chunk.glmAem = t.glmAem;
        chunk.palette = t.palette;
        chunk.texWidth = t.texWidth;
        chunk.texHeight = t.texHeight;
        chunk.texCopyWidth = t.texCopyWidth;
        chunk.texCopyHeight = t.texCopyHeight;
        chunk.texels = t.texels;
        chunk.vertices.assign(t.vertices.begin() + baseVertex, t.vertices.begin() + baseVertex + vertexCount);
        for (std::size_t i = begin; i < end; ++i) {
            gpu::GpuDraw draw = t.draws[i];
            draw.first -= baseVertex;
            chunk.draws.push_back(draw);
        }
        chunk.frameWords.resize(words);
        chunk.depthWords.resize(words);
        const uint32_t rectWidth = static_cast<uint32_t>(rectX1 - rectX0 + 1);
        const uint32_t rectHeight = static_cast<uint32_t>(rectY1 - rectY0 + 1);
        local.readRect(t.frameView, static_cast<uint32_t>(rectX0), static_cast<uint32_t>(rectY0), rectWidth, rectHeight,
                       chunk.frameWords.data());
        local.readRect(t.zView, static_cast<uint32_t>(rectX0), static_cast<uint32_t>(rectY0), rectWidth, rectHeight,
                       chunk.depthWords.data());
        std::size_t index = 0;
        const std::vector<uint32_t> frameBefore = chunk.frameWords;
        const std::vector<uint32_t> depthBefore = chunk.depthWords;
        if (!gpu.Execute(chunk, error)) return false;
        counters.gpuChunks++;
        counters.gpuDraws += chunk.draws.size();
        // Write back only the words the draws changed (IN-335). The frame and depth buffers may share
        // local memory (a 2D pass with an unused ZBUF pointing at the other display buffer, or at the
        // frame itself): writing the untouched pre-draw depth words over the rectangle would put the
        // old pixels back over what this chunk just drew. Per pixel the frame word is stored before the
        // depth word, the order the CPU reference uses (cpuref_draw.cpp), so a changed depth word that
        // aliases a changed frame word wins as it does there.
        uint64_t frameChanged = 0;
        uint64_t depthChanged = 0;
        index = 0;
        for (int32_t y = rectY0; y <= rectY1; ++y) {
            for (int32_t x = rectX0; x <= rectX1; ++x, ++index) {
                if (chunk.frameWords[index] != frameBefore[index]) {
                    local.writePixel(t.frameView, static_cast<uint32_t>(x), static_cast<uint32_t>(y),
                                     chunk.frameWords[index]);
                    ++frameChanged;
                }
                if (chunk.depthWords[index] != depthBefore[index]) {
                    local.writePixel(t.zView, static_cast<uint32_t>(x), static_cast<uint32_t>(y),
                                     chunk.depthWords[index]);
                    ++depthChanged;
                }
            }
        }
        counters.writebackPixels += frameChanged;
        counters.writebackDepthWords += depthChanged;
        begin = end;
    }
    return true;
}

void TranslateBatch(const Batch &batch, const memory::LocalMemory &local, const clut::ClutUnit &clut,
                    BatchTranslation &out, Counters &counters) {
    Translate(batch, local, clut, out, counters);
}

void TranslateBatchHooked(const Batch &batch, const memory::LocalMemory &local, const clut::ClutUnit &clut,
                          BatchTranslation &out, Counters &counters, const TextureReadHook &beforeTextureRead,
                          TextureCacheView *textureCache, const TextureFromTargetHook &textureFromTarget,
                          const TextureFromGlmHook &textureFromGlm) {
    Translate(batch, local, clut, out, counters, beforeTextureRead, textureCache, textureFromTarget, textureFromGlm);
}

// ---- Renderer -----------------------------------------------------------------------------------

// Why resident targets are written back (counters only; the M7 tuning needs to know).
// Why a resident target was written back (profile counters; Other = host reads of the whole memory).
enum class WritebackReason : uint8_t {
    Other, Texture, Clut, Transfer, Display, Eviction, LoadOverlap, Fallback, CpuSprite, Count
};

struct ReasonScope {
    WritebackReason& slot;
    WritebackReason saved;
    ReasonScope(WritebackReason& target, WritebackReason reason) : slot(target), saved(target) { slot = reason; }
    ~ReasonScope() { slot = saved; }
};

class Renderer::Impl final : public BatchSink, public SyncSink, public TransferSink, public TextureCacheView {
public:
    // ---- Shadow-memory write tracking and the decoded-texture cache (IN-341) ----------------------------
    // Every write into the shadow memory bumps the sequence number of the pages it may touch (conservative
    // spans). A cached decode is reused only when none of its pages changed after it was made.
    void MarkWritten(const BlockSpan& span) {
        if (span.count == 0) return;
        BumpMemory(span, "reject-shadow-write");
        ++writeSequence_;
        if (span.count >= kAllBlocks) {
            pageWrites_.fill(writeSequence_);
            return;
        }
        const uint32_t firstPage = span.begin / kBlocksPerPage;
        const uint32_t lastBlock = span.begin + span.count - 1u;
        for (uint32_t page = firstPage; page <= lastBlock / kBlocksPerPage; ++page) {
            pageWrites_[page % memory::kPagesTotal] = writeSequence_;
        }
    }

    // IN-398 follow-up: a version per page bumped by EVERY write into GS memory, the shadow memory (MarkWritten) or
    // the GLM (slot swizzles, EE uploads into GLM-owned pages, tone strips). A dropped slot is a carry source only
    // while its pages still hold the version they had when it was dropped: an older copy of the same memory never
    // supplies scaled blocks (the owner's flickering shadow shapes: a target cleared to the same value kept an older
    // slot's scaled remnants).
    void BumpMemory(const BlockSpan& span, const char* writer = "shadow") {
        if (span.count == 0) return;
        ++memoryClock_;
        if (span.count >= kAllBlocks) {
            memoryVersion_.fill(memoryClock_);
            memoryWriter_.fill(writer);
            return;
        }
        const uint32_t firstPage = span.begin / kBlocksPerPage;
        const uint32_t lastBlock = span.begin + span.count - 1u;
        for (uint32_t page = firstPage; page <= lastBlock / kBlocksPerPage; ++page) {
            memoryVersion_[page % memory::kPagesTotal] = memoryClock_;
            memoryWriter_[page % memory::kPagesTotal] = writer;
        }
    }

    // Diagnostics: the writer of the newest page of `span`.
    const char* NewestWriter(const BlockSpan& span) const {
        if (span.count == 0 || span.count >= kAllBlocks) return "reject-all";
        uint64_t newest = 0;
        const char* writer = "reject-none";
        const uint32_t firstPage = span.begin / kBlocksPerPage;
        const uint32_t lastBlock = span.begin + span.count - 1u;
        for (uint32_t page = firstPage; page <= lastBlock / kBlocksPerPage; ++page) {
            if (memoryVersion_[page % memory::kPagesTotal] >= newest) {
                newest = memoryVersion_[page % memory::kPagesTotal];
                writer = memoryWriter_[page % memory::kPagesTotal];
            }
        }
        return writer;
    }

    uint64_t MemoryVersion(const BlockSpan& span) const {
        uint64_t newest = 0;
        if (span.count == 0) return 0;
        if (span.count >= kAllBlocks) {
            for (uint64_t value : memoryVersion_) newest = std::max(newest, value);
            return newest;
        }
        const uint32_t firstPage = span.begin / kBlocksPerPage;
        const uint32_t lastBlock = span.begin + span.count - 1u;
        for (uint32_t page = firstPage; page <= lastBlock / kBlocksPerPage; ++page) {
            newest = std::max(newest, memoryVersion_[page % memory::kPagesTotal]);
        }
        return newest;
    }

    bool UnchangedSince(const BlockSpan& span, uint64_t sequence) const {
        if (span.count >= kAllBlocks) {
            for (uint64_t value : pageWrites_) {
                if (value > sequence) return false;
            }
            return true;
        }
        const uint32_t firstPage = span.begin / kBlocksPerPage;
        const uint32_t lastBlock = span.begin + span.count - 1u;
        for (uint32_t page = firstPage; page <= lastBlock / kBlocksPerPage; ++page) {
            if (pageWrites_[page % memory::kPagesTotal] > sequence) return false;
        }
        return true;
    }

    struct CachedTexture {
        TextureKey key;
        BlockSpan span;
        uint64_t sequence = 0;
        uint64_t lastUse = 0;
        std::vector<uint32_t> texels;
    };

    static uint64_t KeyHash(const TextureKey& k) {
        const uint64_t words[] = {k.tbp, k.tbw, k.psm, k.width, k.height, k.ta0, k.ta1, k.aem, k.paletteHash};
        uint64_t hash = 1469598103934665603ull;
        for (uint64_t w : words) {
            hash ^= w;
            hash *= 1099511628211ull;
        }
        return hash;
    }

    const std::vector<uint32_t>* find(const TextureKey& key, const memory::BufferView&) override {
        if (!textureCacheEnabled_) return nullptr;
        auto it = textures_.find(KeyHash(key));
        if (it == textures_.end() || !(it->second.key == key)) return nullptr;
        if (!UnchangedSince(it->second.span, it->second.sequence)) {
            textures_.erase(it);
            return nullptr;
        }
        it->second.lastUse = ++textureUses_;
        return &it->second.texels;
    }

    void store(const TextureKey& key, const memory::BufferView& view, const std::vector<uint32_t>& texels) override {
        if (!textureCacheEnabled_) return;
        if (textures_.size() >= kTextureCacheEntries) {
            // Least recently used entry goes (the cache is small: GT4 menus reuse a few dozen textures).
            auto oldest = textures_.begin();
            for (auto it = textures_.begin(); it != textures_.end(); ++it) {
                if (it->second.lastUse < oldest->second.lastUse) oldest = it;
            }
            textures_.erase(oldest);
        }
        CachedTexture entry;
        entry.key = key;
        entry.span = SpanOfRect(view, 0, 0, static_cast<int32_t>(key.width) - 1, static_cast<int32_t>(key.height) - 1);
        entry.sequence = writeSequence_;
        entry.lastUse = ++textureUses_;
        entry.texels = texels;
        textures_[KeyHash(key)] = std::move(entry);
    }

    static FrontEndOptions RendererFrontEndOptions() {
        FrontEndOptions options;
        options.vertexTrace = false;  // no consumer yet (IN-353)
        return options;
    }

    // IN-353: the CPU CLUT copies entries only while the GPU CLUT is not authoritative. Leaving GPU CLUT mode
    // reads the GPU CLUT back into the CPU unit so it is exact again.
    void UpdateClutCopies() {
        clut::ClutUnit& unit = frontEnd_.MutableClut();
        const bool copies = !GpuClutActive();
        if (copies == unit.CopiesData()) return;
        if (copies && gpuClutSynced_ && gpu_.Ready()) {
            std::vector<uint16_t> halfwords;
            std::string error;
            if (!gpu_.ReadClut(halfwords, error)) {
                Fail(error);
                return;
            }
            unit.SetHalfwords(halfwords.data());
        }
        unit.SetCopiesData(copies);
    }

    Impl() : frontEnd_(RendererFrontEndOptions(), *this, *this, *this) {
        frontEnd_.SetLocalMemory(memory_.Local().bytes().data());
        InitSlots();
        frontEnd_.SetClutLoaded([this](const clut::Tex0Write& write) { OnClutLoaded(write); });
        frontEnd_.SetBeforeLocalRead([this](const BlockRange& range) {
            if (GpuClutActive()) return;  // IN-349: the GPU repeats the load from the GLM; no write-back
            {
                ReasonScope scope(writebackReason_, WritebackReason::Clut);
                CoherentForRead(BlockSpan{range.begin, range.count});
            }
        });
    }

    // ---- Transfers (02.7, 10.2): the resident targets are made coherent, then the shadow memory works ----
    void InvalidateDestination(const UploadCommit& c) override {
        FlushToneStrips();  // IN-383: the collected strips write local memory before the transfer does
        ReasonScope scope(writebackReason_, WritebackReason::Transfer);
        // With GLM parking the slots park and CommitUpload merges into GLM-owned pages (IN-350).
        CoherentForWrite(SpanOfTransfer(c.dbp, c.dbw, c.dpsm, c.dsax, c.dsay, c.rrw, c.rrh), nullptr,
                         GlmParkingActive());
        memory_.InvalidateDestination(c);
    }
    void CommitUpload(const UploadCommit& c, const uint8_t* bytes, size_t byteCount) override {
        FlushToneStrips();  // IN-383: the collected strips write local memory before the upload does

        const BlockSpan span = SpanOfTransfer(c.dbp, c.dbw, c.dpsm, c.dsax, c.dsay, c.rrw, c.rrh);
        StageTimer timer(counters.uploadUs);
        if (GlmParkingActive()) {
            {
                ReasonScope scope(writebackReason_, WritebackReason::Transfer);
                CoherentForWrite(span, nullptr, true);  // resident slots park (no CPU wait)
            }
            if (gpuUploads_ && UploadOnGpu(c, bytes, byteCount, span)) return;
            if (AnyGlmOwned(span)) {
                MergeUploadIntoGlm(c, bytes, byteCount, span);
                return;
            }
            memory_.CommitUpload(c, bytes, byteCount);
            MarkWritten(span);
            return;
        }
        {
            ReasonScope scope(writebackReason_, WritebackReason::Transfer);
            CoherentForWrite(span);
        }
        memory_.CommitUpload(c, bytes, byteCount);
        MarkWritten(span);
    }

    // IN-351: the upload is written into the GLM by a kernel instead of into the shadow memory pixel by pixel on
    // the CPU (GT4 streams megabytes of textures per race frame). The stale pages of the destination are uploaded
    // to the GLM first, then the kernel writes the transfer's pixels, and the pages become GLM-owned: GPU users
    // read them there; a CPU user downloads them first. Only rectangles whose pixels have distinct addresses go
    // this way (no row wrap inside the buffer width, no coordinate wrap); others take the CPU path.
    bool UploadOnGpu(const UploadCommit& c, const uint8_t* bytes, size_t byteCount, const BlockSpan& span) {
        const auto psm = memory::decodePsm(c.dpsm);
        if (!psm || c.rrw == 0 || c.rrh == 0 || span.count >= kAllBlocks) return false;
        const memory::PsmInfo& info = memory::psmInfo(*psm);
        const uint32_t perRow = std::max<uint32_t>(1u, (c.dbw * 64u) / info.pageWidth);
        const uint32_t rowLimit = std::min<uint32_t>(perRow * info.pageWidth, memory::kTransferCoordinateWrap);
        if (c.dsax + c.rrw > rowLimit || c.dsay + c.rrh > memory::kTransferCoordinateWrap) return false;
        const uint64_t available = static_cast<uint64_t>(byteCount) * 8u / info.transferBpp;
        const uint32_t pixels =
            static_cast<uint32_t>(std::min<uint64_t>(available, static_cast<uint64_t>(c.rrw) * c.rrh));
        if (!UploadStalePages(span)) return true;
        std::string error;
        if (!gpu_.UploadHostToGlm(c.dbp, c.dbw, c.dpsm, c.dsax, c.dsay, c.rrw, pixels, info.transferBpp, bytes,
                                  byteCount, error)) {
            Fail(error);
            return true;
        }
        BumpMemory(span, "reject-host-upload");
        ForEachPage(span, [&](uint32_t page) {
            if (!glmOwned_[page]) {
                glmOwned_[page] = true;
                ++glmOwnedPages_;
            }
        });
        counters.glmHostUploads++;
        return true;
    }

    // IN-350: an upload whose destination has GLM-owned pages. The exact bytes it writes there are found by
    // writing it twice into those pages filled with 0x00 and with 0xFF (a bit the upload writes is equal in both
    // passes); they are merged into the GLM, so the pages stay GLM-owned and need no download. The shadow memory
    // gets the upload too (the other pages are then exact; GLM-owned pages are replaced on their next download).
    void MergeUploadIntoGlm(const UploadCommit& c, const uint8_t* bytes, size_t byteCount, const BlockSpan& span) {
        const auto psm = memory::decodePsm(c.dpsm);
        if (!psm) return;  // invalid PSM: the transfer is dropped (front-end rule), as in the shadow memory sink
        const memory::BufferView destination{c.dbp, c.dbw, *psm};
        const memory::TransferRect rect{c.dsax, c.dsay, c.rrw, c.rrh};
        std::vector<uint32_t> owned;
        ForEachPage(span, [&](uint32_t page) {
            if (glmOwned_[page]) owned.push_back(page);
        });
        uint8_t* local = memory_.Local().bytes().data();
        constexpr size_t kPageBytes = 8192;
        std::vector<uint8_t> saved(owned.size() * kPageBytes);
        std::vector<uint8_t> zeros(owned.size() * kPageBytes);
        for (size_t i = 0; i < owned.size(); ++i) {
            std::memcpy(saved.data() + i * kPageBytes, local + owned[i] * kPageBytes, kPageBytes);
            std::memset(local + owned[i] * kPageBytes, 0x00, kPageBytes);
        }
        memory::hostToLocal(memory_.Local(), destination, rect, std::span<const uint8_t>(bytes, byteCount));
        for (size_t i = 0; i < owned.size(); ++i) {
            std::memcpy(zeros.data() + i * kPageBytes, local + owned[i] * kPageBytes, kPageBytes);
            std::memset(local + owned[i] * kPageBytes, 0xFF, kPageBytes);
        }
        memory::hostToLocal(memory_.Local(), destination, rect, std::span<const uint8_t>(bytes, byteCount));
        std::vector<uint32_t> triples;
        for (size_t i = 0; i < owned.size(); ++i) {
            const uint8_t* ones = local + owned[i] * kPageBytes;
            const uint8_t* zero = zeros.data() + i * kPageBytes;
            for (uint32_t word = 0; word < kPageBytes / 4; ++word) {
                // Bit granularity: a T4 upload writes nibbles, so a written bit is one equal in both passes.
                uint32_t fromZeros = 0;
                uint32_t fromOnes = 0;
                std::memcpy(&fromZeros, zero + word * 4, 4);
                std::memcpy(&fromOnes, ones + word * 4, 4);
                const uint32_t mask = ~(fromZeros ^ fromOnes);
                const uint32_t value = fromZeros & mask;
                if (mask != 0) {
                    triples.push_back(owned[i] * static_cast<uint32_t>(kPageBytes / 4) + word);
                    triples.push_back(value);
                    triples.push_back(mask);
                }
            }
            std::memcpy(local + owned[i] * kPageBytes, saved.data() + i * kPageBytes, kPageBytes);
        }
        memory_.CommitUpload(c, bytes, byteCount);  // the real write (exact for the pages the GLM does not own)
        std::string error;
        if (!gpu_.MergeIntoGlm(triples, error)) {
            Fail(error);
            return;
        }
        MarkWritten(span);  // decoded-texture cache entries expire; pages the GLM does not own become stale there
        for (uint32_t page : owned) glmPageSequence_[page] = pageWrites_[page] + 1;  // the GLM holds the write
        counters.glmUploadMerges++;
        counters.glmMergedWords += triples.size() / 3;
    }
    void LocalToLocalMove(const LocalMoveCommit& m) override {
        FlushToneStrips();  // IN-383: the collected strips write local memory before the move reads it

        {
            ReasonScope scope(writebackReason_, WritebackReason::Transfer);
            CoherentForRead(SpanOfTransfer(m.sbp, m.sbw, m.spsm, m.ssax, m.ssay, m.rrw, m.rrh));
        }
        const BlockSpan destination = SpanOfTransfer(m.dbp, m.dbw, m.dpsm, m.dsax, m.dsay, m.rrw, m.rrh);
        CoherentForWrite(destination);
        memory_.LocalToLocalMove(m);
        MarkWritten(destination);
    }
    void BeginLocalToHost(const LocalToHostSetup& r) override {
        FlushToneStrips();  // IN-383: a readback observes the strips
        pendingReadSpan_ = SpanOfTransfer(r.sbp, r.sbw, r.spsm, r.ssax, r.ssay, r.rrw, r.rrh);
        {
            ReasonScope scope(writebackReason_, WritebackReason::Transfer);
            CoherentForRead(SpanOfTransfer(r.sbp, r.sbw, r.spsm, r.ssax, r.ssay, r.rrw, r.rrh));
        }
        memory_.BeginLocalToHost(r);
    }
    void CancelLocalToHost() override { memory_.CancelLocalToHost(); }

    // ---- Resident targets (M7, IN-340, IN-344) ----------------------------------------------------------
    // Colour and depth targets stay in GPU target slots between batches (kGpuColourSlots - 1 colour slots and
    // kGpuDepthSlots - 1 depth slots; the last of each is the write-back mode's scratch pair). A target is an
    // exact (BP, BW, PSM) at 1x; `loaded` is the rectangle its image holds (equal to the shadow memory when it
    // was loaded), `drawn` the part the GPU changed since (newer than the shadow memory). Any shadow-memory read
    // of a drawn range writes it back first; any write over a loaded range writes back and drops the target; no
    // two slots ever hold overlapping memory (loading one writes back and drops the others it overlaps). A batch
    // whose colour and depth buffers share memory takes the write-back path, so the cache mode stores exactly
    // what the write-back mode stores (spec 15.2 M7 gate). The least recently used slot is evicted.
    struct ResidentSlot {
        bool depth = false;
        uint32_t gpuIndex = 0;     // GPU image of this slot
        bool hasIdentity = false;
        memory::BufferView view{0, 0, memory::Psm::Ct32};
        PixelRect loaded;  // empty = nothing loaded
        PixelRect drawn;   // empty = nothing newer than the shadow memory
        uint64_t lastUse = 0;
        // IN-350: the GLM already holds this slot's drawn pixels when no draw and no GLM page upload under them
        // happened since its last swizzle (GT4 draws palettes into a target and loads CLUTs from it many times).
        uint64_t drawSequence = 0;        // bumped by every draw into the slot
        uint64_t glmSwizzledDraw = ~0ull; // drawSequence at the last swizzle (~0: none)
        uint64_t glmSwizzledClock = 0;    // glmUploadClock_ at the last swizzle
        // IN-398: the scaled image lags the 1x words (drawn while a tone run paused the scaled pass and not
        // refreshed after it); never a carry source.
        bool scaledStale = false;
        // IN-398: what the slot held when it was last dropped. Its images stay as they were until the slot takes a new
        // identity, so a later load of the same memory (under any interchangeable view) can keep its scaled blocks.
        bool droppedIntact = false;
        memory::BufferView droppedView{0, 0, memory::Psm::Ct32};
        PixelRect droppedLoaded;
        uint64_t droppedVersion = 0;  // MemoryVersion of droppedLoaded's pages at the drop
    };

    static constexpr uint32_t kResidentColourSlots = gpu::kGpuColourSlots - 1;
    static constexpr uint32_t kResidentDepthSlots = gpu::kGpuDepthSlots - 1;

    void InitSlots() {
        for (uint32_t i = 0; i < kResidentColourSlots; ++i) slots_.push_back(ResidentSlot{false, i});
        for (uint32_t i = 0; i < kResidentDepthSlots; ++i) slots_.push_back(ResidentSlot{true, i});
    }

    BlockSpan LoadedSpan(const ResidentSlot& slot) const {
        return slot.loaded.Empty() ? BlockSpan{} : SpanOfRect(slot.view, slot.loaded);
    }

    void WriteBack(ResidentSlot& slot) {
        if (slot.drawn.Empty() || failed_) return;
        // The words around the drawn rectangle in GLM-owned pages are newer in the GLM (IN-350).
        Materialise(SpanOfRect(slot.view, slot.drawn));
        counters.cacheWritebackByReason[static_cast<size_t>(writebackReason_)]++;
        StageTimer timer(counters.cacheWritebackUs);
        const PixelRect r = slot.drawn;
        gpu::GpuChunk chunk;
        chunk.rectX0 = r.x0;
        chunk.rectY0 = r.y0;
        chunk.rectX1 = r.x1;
        chunk.rectY1 = r.y1;
        chunk.colourSlot = slot.depth ? gpu::kGpuScratchColourSlot : slot.gpuIndex;
        chunk.depthSlot = slot.depth ? slot.gpuIndex : gpu::kGpuScratchDepthSlot;
        chunk.uploadFrame = false;
        chunk.uploadDepth = false;
        chunk.readbackFrame = !slot.depth;
        chunk.readbackDepth = slot.depth;
        const size_t words = static_cast<size_t>(r.x1 - r.x0 + 1) * static_cast<size_t>(r.y1 - r.y0 + 1);
        std::vector<uint32_t>& data = slot.depth ? chunk.depthWords : chunk.frameWords;
        data.resize(words);
        std::string error;
        if (!gpu_.Execute(chunk, error)) {
            Fail(error);
            return;
        }
        memory_.Local().writeRect(slot.view, static_cast<uint32_t>(r.x0), static_cast<uint32_t>(r.y0),
                                  static_cast<uint32_t>(r.x1 - r.x0 + 1), static_cast<uint32_t>(r.y1 - r.y0 + 1),
                                  data.data());
        counters.cacheWritebacks++;
        counters.cacheWritebackWords += words;
        MarkWritten(SpanOfRect(slot.view, r));
        slot.drawn = PixelRect{};
    }

    // IN-377: views the display circuits showed in the last few VBlanks. GT4 double-buffers its race frame: the
    // buffer it shows next is finished one frame before, and the game then reuses part of that memory as a
    // read-only depth buffer, which drops its colour slot (and its scaled image) before the VBlank shows it.
    void NoteDisplayedView(const memory::BufferView& view) {
        for (const memory::BufferView& known : recentDisplayViews_) {
            if (SameView(known, view)) return;
        }
        recentDisplayViews_[recentDisplayNext_] = view;
        recentDisplayNext_ = (recentDisplayNext_ + 1u) % recentDisplayViews_.size();
    }
    bool RecentlyDisplayed(const memory::BufferView& view) const {
        for (const memory::BufferView& known : recentDisplayViews_) {
            if (known.bufferWidth != 0 && SameView(known, view)) return true;
        }
        return DisplayedView(view);
    }

    // IN-377: before a colour slot of a (recently) displayed buffer gives up its image at an internal scale > 1, its
    // scaled image and its 1x words are copied into a display shadow. The scaled compose shows a shadow sub-pixel
    // only where the native pixel under it still holds the word saved with it (compared with the GLM per pixel);
    // any pixel the game changed since is shown from the GLM at 1x. The 1x words stay the only authority (nothing
    // here reaches memory).
    void SaveDisplayShadow(const ResidentSlot& slot) {
        if (failed_ || gpu_.Scale() <= 1 || slot.depth || !slot.hasIdentity || slot.loaded.Empty()) return;
        if (!RecentlyDisplayed(slot.view)) return;
        // IN-384: the shadow is copied out of the slot's own scaled image, so its rectangle is that image's extent.
        uint32_t scaledWidth = 0, scaledHeight = 0;
        gpu_.ScaledExtent(false, slot.gpuIndex, scaledWidth, scaledHeight);
        const PixelRect saved{std::max(slot.loaded.x0, 0), std::max(slot.loaded.y0, 0),
                              std::min(slot.loaded.x1, static_cast<int32_t>(scaledWidth) - 1),
                              std::min(slot.loaded.y1, static_cast<int32_t>(scaledHeight) - 1)};
        if (saved.Empty()) return;
        size_t chosen = shadows_.size();
        for (size_t i = 0; i < shadows_.size() && chosen == shadows_.size(); ++i) {
            if (shadows_[i].valid && SameView(shadows_[i].view, slot.view)) chosen = i;
        }
        for (size_t i = 0; i < shadows_.size() && chosen == shadows_.size(); ++i) {
            if (!shadows_[i].valid) chosen = i;
        }
        if (chosen == shadows_.size()) {
            chosen = 0;
            for (size_t i = 1; i < shadows_.size(); ++i) {
                if (shadows_[i].savedAt < shadows_[chosen].savedAt) chosen = i;
            }
        }
        std::string error;
        if (!gpu_.SaveScaledShadow(slot.gpuIndex, static_cast<uint32_t>(chosen), saved.x0, saved.y0, saved.x1,
                                   saved.y1, error)) {
            Fail(error);
            return;
        }
        DisplayShadow& shadow = shadows_[chosen];
        shadow.valid = true;
        shadow.view = slot.view;
        shadow.saved = saved;
        shadow.savedAt = ++shadowSaves_;
        counters.scaledShadowSaves++;
    }

    // IN-377: the display shadow holding the whole `read` rectangle of `view`, or -1 (the compose kernel checks every
    // pixel against the GLM).
    int32_t DisplayShadowFor(const memory::BufferView& view, const PixelRect& read) const {
        for (size_t i = 0; i < shadows_.size(); ++i) {
            const DisplayShadow& shadow = shadows_[i];
            if (shadow.valid && SameView(shadow.view, view) && Contains(shadow.saved, read)) {
                return static_cast<int32_t>(i);
            }
        }
        return -1;
    }

    // S1 diagnostics (SCALE-ONLY-PLAN): COREGS_HW_SLOT_TRACE=<file> logs slot drops, loads and refreshes from
    // COREGS_HW_TRACE_FROM_VBLANK on, so the reloads that rebuild a scaled image from 1x words can be attributed.
    void SlotTrace(const char* event, const ResidentSlot& slot, const PixelRect& rect) {
        static const char* path = std::getenv("COREGS_HW_SLOT_TRACE");
        static const long from = std::getenv("COREGS_HW_TRACE_FROM_VBLANK") ? std::atol(std::getenv("COREGS_HW_TRACE_FROM_VBLANK")) : 0L;
        if (!path || static_cast<long>(vblanks_) < from) return;
        if (std::FILE* file = std::fopen(path, "ab")) {
            std::fprintf(file, "v%llu %s gpu=%u depth=%u bp=%u bw=%u psm=%u rect=%d,%d,%d,%d reason=%u shown=%u\n",
                         static_cast<unsigned long long>(vblanks_), event, slot.gpuIndex, slot.depth ? 1u : 0u,
                         slot.view.basePointer, slot.view.bufferWidth, static_cast<unsigned>(slot.view.psm), rect.x0,
                         rect.y0, rect.x1, rect.y1, static_cast<unsigned>(writebackReason_),
                         (!slot.depth && DisplayedView(slot.view)) ? 1u : 0u);
            std::fclose(file);
        }
    }

    void Drop(ResidentSlot& slot, bool keepShadow = true) {
        if (slot.hasIdentity) SlotTrace("drop", slot, slot.loaded);
        if (keepShadow) SaveDisplayShadow(slot);
        static const bool trace = std::getenv("COREGS_HW_SCALE_TRACE") != nullptr;  // diagnostic only
        if (trace && slot.hasIdentity && !slot.depth && DisplayedView(slot.view)) {
            std::fprintf(stderr, "[scale] drop displayed slot gpu=%u bp=%u bw=%u psm=%u reason=%u\n", slot.gpuIndex,
                         slot.view.basePointer, slot.view.bufferWidth, static_cast<unsigned>(slot.view.psm),
                         static_cast<unsigned>(writebackReason_));
        }
        slot.droppedIntact = slot.hasIdentity && !slot.loaded.Empty() && !slot.scaledStale;
        slot.droppedView = slot.view;
        slot.droppedLoaded = slot.loaded;
        slot.droppedVersion = slot.droppedIntact ? MemoryVersion(SpanOfRect(slot.view, slot.loaded)) : 0;
        slot.loaded = PixelRect{};
        slot.drawn = PixelRect{};
        slot.hasIdentity = false;
        slot.scaledStale = false;
        slot.glmSwizzledDraw = ~0ull;
    }

    // The shadow memory of `span` becomes current for a CPU reader: drawn slots are written back, GLM-owned
    // pages are downloaded.
    void CoherentForRead(const BlockSpan& span) {
        if (GlmParkingActive() && glmBatchedWritebacks_) {
            std::vector<ResidentSlot*> slots;
            for (ResidentSlot& slot : slots_) {
                if (!slot.drawn.Empty() && SpansOverlap(SpanOfRect(slot.view, slot.drawn), span)) {
                    slots.push_back(&slot);
                }
            }
            WriteBackViaGlm(slots, span);
            return;
        }
        for (ResidentSlot& slot : slots_) {
            if (!slot.drawn.Empty() && SpansOverlap(SpanOfRect(slot.view, slot.drawn), span)) WriteBack(slot);
        }
        Materialise(span);
    }

    // IN-357: the drawn rectangles of `slots` and the GLM-owned pages of `span` reach the shadow memory through
    // ONE GLM download (one submission and one wait) instead of one readback per slot: the stale pages are
    // uploaded, the slots swizzled into the GLM (skipped when it already holds them), then every page that may
    // differ from the shadow memory (the slots' drawn pages and the GLM-owned pages of `span`) is downloaded. The
    // slots stay resident with nothing GPU-newer, as after WriteBack. A slot whose drawn rectangle is wider than
    // its buffer (aliasing addresses) takes the per-slot WriteBack.
    void WriteBackViaGlm(const std::vector<ResidentSlot*>& slots, const BlockSpan& span) {
        if (failed_) return;
        std::vector<ResidentSlot*> batched;
        for (ResidentSlot* slot : slots) {
            if (slot->drawn.Empty()) continue;
            const int32_t bufferWidth = static_cast<int32_t>(std::max<uint32_t>(slot->view.bufferWidth, 1u) * 64u);
            if (slot->drawn.x1 >= bufferWidth) {
                WriteBack(*slot);
                continue;
            }
            batched.push_back(slot);
        }
        if (batched.empty() && !AnyGlmOwned(span)) return;
        StageTimer timer(counters.cacheWritebackUs);
        for (ResidentSlot* slot : batched) {
            if (!SwizzleSlotIntoGlm(*slot)) return;  // uploads the stale pages under it first
        }
        std::vector<bool> wanted(memory::kPagesTotal, false);
        for (ResidentSlot* slot : batched) {
            ForEachPage(SpanOfRect(slot->view, slot->drawn), [&](uint32_t page) { wanted[page] = true; });
        }
        ForEachPage(span, [&](uint32_t page) {
            if (glmOwned_[page]) wanted[page] = true;
        });
        // Every wanted page must be current in the GLM before it is downloaded (pages under a slot's drawn
        // rectangle were uploaded by its swizzle; GLM-owned pages are current by definition).
        std::vector<std::pair<uint32_t, uint32_t>> ranges;
        std::vector<uint32_t> pages;
        for (uint32_t page = 0; page < memory::kPagesTotal; ++page) {
            if (!wanted[page]) continue;
            if (!glmOwned_[page] && (glmPageSequence_[page] == 0 || glmPageSequence_[page] < pageWrites_[page] + 1)) {
                Fail("GLM page " + std::to_string(page) + " is stale at a batched write-back");
                return;
            }
            const uint32_t offset = page * 8192u;
            if (!ranges.empty() && ranges.back().first + ranges.back().second == offset) {
                ranges.back().second += 8192u;
            } else {
                ranges.push_back({offset, 8192u});
            }
            pages.push_back(page);
        }
        std::string error;
        if (!gpu_.DownloadGlm(ranges, memory_.Local().bytes().data(), error)) {
            Fail(error);
            return;
        }
        for (uint32_t page : pages) {
            if (glmOwned_[page]) {
                glmOwned_[page] = false;
                --glmOwnedPages_;
            }
            MarkWritten(BlockSpan{page * kBlocksPerPage, kBlocksPerPage});  // decoded-texture cache entries expire
            glmPageSequence_[page] = pageWrites_[page] + 1;                  // the GLM equals the shadow memory
        }
        for (ResidentSlot* slot : batched) {
            counters.cacheWritebackByReason[static_cast<size_t>(writebackReason_)]++;
            counters.cacheWritebacks++;
            slot->drawn = PixelRect{};
        }
        counters.glmBatchedWritebacks++;
        counters.glmPagesDownloaded += pages.size();
    }

    // Every slot that holds memory in `span` (except `keep`, may be null) gives it up. For a CPU writer
    // (`forGpu` false) the slots are written back and GLM-owned pages downloaded, so the shadow memory is
    // current before it changes. For a GPU consumer (a slot about to load `span`) the slots are parked in the
    // GLM instead (IN-350): no CPU stall; the consumer then loads those pages from the GLM.
    void CoherentForWrite(const BlockSpan& span, const ResidentSlot* keep = nullptr, bool forGpu = false) {
        const bool park = forGpu && GlmParkingActive();
        for (ResidentSlot& slot : slots_) {
            if (&slot == keep) continue;
            if (SpansOverlap(LoadedSpan(slot), span)) {
                static const bool trace = std::getenv("COREGS_HW_SCALE_TRACE") != nullptr;  // diagnostic only
                if (trace && keep && slot.hasIdentity && !slot.depth && DisplayedView(slot.view)) {
                    std::fprintf(stderr, "[scale] overlap by %s slot bp=%u bw=%u psm=%u span=%u+%u\n",
                                 keep->depth ? "depth" : "colour", keep->view.basePointer, keep->view.bufferWidth,
                                 static_cast<unsigned>(keep->view.psm), span.begin, span.count);
                }
                if (park) {
                    Park(slot);
                } else {
                    WriteBack(slot);
                }
                Drop(slot);
            }
        }
        if (!forGpu) Materialise(span);
    }

    void WriteBackAll() {
        for (ResidentSlot& slot : slots_) WriteBack(slot);
        Materialise(BlockSpan{0, kAllBlocks});
    }

    // ---- GLM-owned pages (IN-350, chapters 04.9 and 05.3) ----------------------------------------------
    // A page is GLM-owned when its newest content lives only in the GLM: a slot's drawn rectangle was parked
    // there (swizzled into the GLM at its own layout) instead of being written back. The GLM copy of the rest of
    // such a page is current (its stale pages were uploaded before the swizzle), so the GLM holds the whole page.
    // GPU users (slot loads, GLM texture decodes, GPU CLUT loads) read it from the GLM directly; any CPU reader or
    // writer of the shadow memory downloads it first (Materialise). This is GT4's buffer reuse: the same memory
    // drawn under other widths and formats (a 640-wide depth buffer, then a 64-wide one at the same base; a
    // 128-wide target inside the frame buffer) moves between slots on the GPU.
    bool GlmParkingActive() const { return cacheMode_ && glmEnabled_ && glmParking_; }

    template <typename Visit>
    static void ForEachPage(const BlockSpan& span, Visit visit) {
        if (span.count == 0) return;
        if (span.count >= kAllBlocks) {
            for (uint32_t page = 0; page < memory::kPagesTotal; ++page) visit(page);
            return;
        }
        const uint32_t first = span.begin / kBlocksPerPage;
        const uint32_t last = (span.begin + span.count - 1u) / kBlocksPerPage;
        for (uint32_t page = first; page <= last; ++page) visit(page % memory::kPagesTotal);
    }

    bool AnyGlmOwned(const BlockSpan& span) const {
        if (glmOwnedPages_ == 0) return false;
        bool any = false;
        ForEachPage(span, [&](uint32_t page) { any = any || glmOwned_[page]; });
        return any;
    }

    // Makes the GLM hold the slot's drawn pixels: uploads the stale pages under them, then swizzles, unless the
    // GLM already holds them (no draw into the slot and no upload of those pages since its last swizzle).
    bool SwizzleSlotIntoGlm(ResidentSlot& slot) {
        const BlockSpan span = SpanOfRect(slot.view, slot.drawn);
        if (!UploadStalePages(span)) return false;
        bool current = slot.glmSwizzledDraw == slot.drawSequence;
        if (current) {
            ForEachPage(span, [&](uint32_t page) {
                current = current && glmPageUploadClock_[page] <= slot.glmSwizzledClock;
            });
        }
        if (current) {
            counters.glmSwizzlesSkipped++;
            return true;
        }
        std::string error;
        if (!gpu_.SwizzleToGlm(slot.depth, slot.gpuIndex, slot.view.basePointer, slot.view.bufferWidth,
                               static_cast<uint32_t>(slot.view.psm), slot.drawn.x0, slot.drawn.y0, slot.drawn.x1,
                               slot.drawn.y1, error)) {
            Fail(error);
            return false;
        }
        slot.glmSwizzledDraw = slot.drawSequence;
        slot.glmSwizzledClock = glmUploadClock_;
        BumpMemory(SpanOfRect(slot.view, slot.drawn), "reject-swizzle");
        counters.glmSwizzles++;
        return true;
    }

    // Swizzles the slot's drawn rectangle into the GLM and marks those pages GLM-owned (no CPU wait).
    void Park(ResidentSlot& slot) {
        if (slot.drawn.Empty() || failed_) return;
        const int32_t bufferWidth = static_cast<int32_t>(std::max<uint32_t>(slot.view.bufferWidth, 1u) * 64u);
        if (slot.drawn.x1 >= bufferWidth) {
            WriteBack(slot);  // wraps into the next page row: no defined last writer for a parallel swizzle
            return;
        }
        const BlockSpan span = SpanOfRect(slot.view, slot.drawn);
        if (!SwizzleSlotIntoGlm(slot)) return;
        ForEachPage(span, [&](uint32_t page) {
            if (!glmOwned_[page]) {
                glmOwned_[page] = true;
                ++glmOwnedPages_;
            }
        });
        counters.glmParks++;
        slot.drawn = PixelRect{};
    }

    // Downloads the GLM-owned pages of `span` into the shadow memory (submits and waits).
    void Materialise(const BlockSpan& span) {
        if (glmOwnedPages_ == 0 || failed_) return;
        std::vector<std::pair<uint32_t, uint32_t>> ranges;
        std::vector<uint32_t> pages;
        ForEachPage(span, [&](uint32_t page) {
            if (!glmOwned_[page]) return;
            const uint32_t offset = page * 8192u;
            if (!ranges.empty() && ranges.back().first + ranges.back().second == offset) {
                ranges.back().second += 8192u;
            } else {
                ranges.push_back({offset, 8192u});
            }
            pages.push_back(page);
        });
        if (pages.empty()) return;
        StageTimer timer(counters.cacheWritebackUs);
        std::string error;
        if (!gpu_.DownloadGlm(ranges, memory_.Local().bytes().data(), error)) {
            Fail(error);
            return;
        }
        for (uint32_t page : pages) {
            glmOwned_[page] = false;
            --glmOwnedPages_;
            MarkWritten(BlockSpan{page * kBlocksPerPage, kBlocksPerPage});  // decoded-texture cache entries expire
            glmPageSequence_[page] = pageWrites_[page] + 1;                  // the GLM equals the shadow memory
        }
        counters.glmDownloads++;
        counters.glmPagesDownloaded += pages.size();
    }

    // The memory a display circuit reads (13.2 read rectangle; a wrapping or unknown read reports all memory).
    static BlockSpan DisplayCircuitSpan(const display::CircuitRect& circuit) {
        const auto psm = memory::decodePsm(circuit.psm);
        if (!psm) return BlockSpan{0, kAllBlocks};
        return SpanOfRect(memory::BufferView{memory::blockOfPage(circuit.fbp), circuit.fbw, *psm}, circuit.dbx,
                          circuit.dby, circuit.dbx + static_cast<int32_t>(circuit.read_width) - 1,
                          circuit.dby + static_cast<int32_t>(circuit.read_height) - 1);
    }

    // This VBlank's display layout (13.2), computed once per VBlank (it advances the DBY history).
    void EnsureLayout() {
        if (layoutVblank_ != vblanks_ || !layoutValid_) {
            layout_ = display::compute_layout(displayRegs_, previousDby_);
            layoutVblank_ = vblanks_;
            layoutValid_ = true;
        }
    }

    // IN-355: records the download of the displayed memory into the current recording (before the frame's
    // submission). The GLM is made current for it first: stale pages uploaded, drawn targets swizzled.
    void CaptureDisplay() {
        captureShown_ = std::move(captureNext_);
        captureNext_ = DisplayCapture{};
        EnsureLayout();
        {
            display::MergePlan plan;
            std::array<bool, 2> available{};
            if (GpuComposeApplies(layout_, plan, available)) return;  // composed on the GPU at present (IN-364)
        }
        DisplayCapture capture;
        capture.layout = layout_;
        capture.regs = displayRegs_;
        capture.background = background_;
        capture.oddField = oddField_;
        std::vector<std::pair<uint32_t, uint32_t>> ranges;
        if (layout_.valid) {
            for (const display::CircuitRect& circuit : layout_.circuits) {
                if (!circuit.enabled || circuit.read_width == 0 || circuit.read_height == 0) continue;
                const BlockSpan span = DisplayCircuitSpan(circuit);
                if (!UploadStalePages(span)) return;
                for (ResidentSlot& slot : slots_) {
                    if (slot.drawn.Empty() || !SpansOverlap(SpanOfRect(slot.view, slot.drawn), span)) continue;
                    const int32_t bufferWidth =
                        static_cast<int32_t>(std::max<uint32_t>(slot.view.bufferWidth, 1u) * 64u);
                    if (slot.drawn.x1 >= bufferWidth) {
                        ReasonScope scope(writebackReason_, WritebackReason::Display);
                        WriteBack(slot);
                        if (!UploadStalePages(span)) return;
                        continue;
                    }
                    if (!SwizzleSlotIntoGlm(slot)) return;
                }
                ForEachPage(span, [&](uint32_t page) {
                    const uint32_t offset = page * 8192u;
                    for (const auto& range : ranges) {
                        if (offset >= range.first && offset < range.first + range.second) return;
                    }
                    if (!ranges.empty() && ranges.back().first + ranges.back().second == offset) {
                        ranges.back().second += 8192u;
                    } else {
                        ranges.push_back({offset, 8192u});
                    }
                });
            }
        }
        std::string error;
        capture.ticket = gpu_.RecordGlmDownload(ranges, error);
        if (!capture.ticket.valid) {
            Fail(error);
            return;
        }
        capture.valid = true;
        captureNext_ = std::move(capture);
        counters.displayCaptures++;
    }

    // The shadow memory replaced wholesale (auxiliary frame end, VRAM clear): GLM-owned content is discarded.
    void DiscardGlmOwned() {
        glmOwned_.fill(false);
        glmOwnedPages_ = 0;
    }

    void DropAll() {
        FlushToneStrips();      // IN-383: the collected strips write the pages before the targets go away
        toneRun_.active = false;  // IN-379: a reset or auxiliary frame ends a tone sequence without a carry
        gpu_.PauseScaled(false);
        for (ResidentSlot& slot : slots_) Drop(slot, false);  // resets and auxiliary frames keep no shadow
        for (DisplayShadow& shadow : shadows_) shadow.valid = false;
    }

    // The slot holding `view`, or the least recently used slot of that kind (written back and dropped).
    // A view one of the display circuits reads now (DISPFB1/2: FBP, FBW, PSM).
    bool DisplayedView(const memory::BufferView& view) const {
        for (const display::CircuitRegisters& registers : displayRegs_.circuits) {
            const display::CircuitFields fields = display::decode_circuit(registers);
            const auto psm = memory::decodePsm(fields.psm);
            if (psm && SameView(view, memory::BufferView{memory::blockOfPage(fields.fbp), fields.fbw, *psm})) return true;
        }
        return false;
    }

    // IN-398: the bits two views of the same memory share when both lay out one 32-bit word per pixel (CT32/CT24,
    // Z32/Z24): all 32 for the same format, the low 24 when either is a 24-bit view. 0 = not interchangeable.
    static uint32_t SharedViewMask(const memory::BufferView& a, const memory::BufferView& b) {
        if (a.basePointer != b.basePointer || a.bufferWidth != b.bufferWidth) return 0u;
        const auto family = [](memory::Psm psm) {
            switch (psm) {
                case memory::Psm::Ct32: case memory::Psm::Ct24: return 1;
                case memory::Psm::Z32: case memory::Psm::Z24: return 2;
                default: return 0;
            }
        };
        const int fa = family(a.psm), fb = family(b.psm);
        if (fa == 0 || fa != fb) return 0u;
        const bool narrow = a.psm == memory::Psm::Ct24 || b.psm == memory::Psm::Ct24 || a.psm == memory::Psm::Z24 ||
                            b.psm == memory::Psm::Z24;
        return narrow ? 0x00FFFFFFu : 0xFFFFFFFFu;
    }

    ResidentSlot& Acquire(bool depth, const memory::BufferView& view) {
        ResidentSlot* oldest = nullptr;
        for (ResidentSlot& slot : slots_) {
            if (slot.depth != depth) continue;
            if (slot.hasIdentity && SameView(slot.view, view)) {
                slot.lastUse = ++slotUses_;
                return slot;
            }
        }
        // IN-373: at an internal scale > 1 a colour target the display circuits show is evicted last, so the
        // presentation finds its scaled image (GT4 keeps the shown buffer of its double buffer untouched while it
        // draws the other one and many small targets). At 1x the order is the plain LRU of IN-344.
        const bool keepDisplayed = !depth && gpu_.Scale() > 1;
        for (int pass = 0; pass < 2 && !oldest; ++pass) {
            for (ResidentSlot& slot : slots_) {
                if (slot.depth != depth || &slot == protectedSlot_) continue;  // never evict this batch's texture
                if (pass == 0 && keepDisplayed && slot.hasIdentity && DisplayedView(slot.view)) continue;
                // IN-398: at scale, the slot holding this memory under another view is not the victim: its images
                // stay intact until the load below, which keeps its scaled blocks (the load's own overlap pass
                // still writes it back and drops it).
                if (pass == 0 && gpu_.Scale() > 1 &&
                    ((slot.hasIdentity && SharedViewMask(slot.view, view) != 0u) ||
                     (!slot.hasIdentity && slot.droppedIntact && SharedViewMask(slot.droppedView, view) != 0u))) {
                    continue;
                }
                if (!oldest || (!slot.hasIdentity && oldest->hasIdentity) ||
                    (slot.hasIdentity == oldest->hasIdentity && slot.lastUse < oldest->lastUse)) {
                    oldest = &slot;
                }
            }
        }
        if (oldest->hasIdentity) counters.cacheEvictions++;
        {
            ReasonScope scope(writebackReason_, WritebackReason::Eviction);
            if (GlmParkingActive()) {
                Park(*oldest);
            } else {
                WriteBack(*oldest);
            }
        }
        Drop(*oldest);
        oldest->droppedIntact = false;  // IN-398: its images are about to hold the new view
        oldest->hasIdentity = true;
        oldest->view = view;
        oldest->lastUse = ++slotUses_;
        return *oldest;
    }

    // Makes `slot` hold at least `rect` loaded from the shadow memory.
    // Fills `rect` of the slot's image from memory: from the GLM when a page under it is GLM-owned (IN-350),
    // otherwise from the shadow memory.
    void UploadIntoSlot(ResidentSlot& slot, const PixelRect& rect) {
        SlotTrace("load", slot, rect);
        const BlockSpan span = SpanOfRect(slot.view, rect);
        if (GlmParkingActive() && (glmLoadsAlways_ || AnyGlmOwned(span))) {
            if (!UploadStalePages(span)) return;
            std::string error;
            if (!gpu_.LoadSlotFromGlm(slot.depth, slot.gpuIndex, slot.view.basePointer, slot.view.bufferWidth,
                                      static_cast<uint32_t>(slot.view.psm), rect.x0, rect.y0, rect.x1, rect.y1, error)) {
                Fail(error);
                return;
            }
            counters.glmSlotLoads++;
            return;
        }
        gpu::GpuChunk chunk;
        chunk.rectX0 = rect.x0;
        chunk.rectY0 = rect.y0;
        chunk.rectX1 = rect.x1;
        chunk.rectY1 = rect.y1;
        chunk.colourSlot = slot.depth ? gpu::kGpuScratchColourSlot : slot.gpuIndex;
        chunk.depthSlot = slot.depth ? slot.gpuIndex : gpu::kGpuScratchDepthSlot;
        // IN-384: the load grows the slot's scaled image to its row width and to these rows before it is recorded.
        chunk.targetWidth = std::max<uint32_t>(slot.view.bufferWidth, 1u) * 64u;
        chunk.uploadFrame = !slot.depth;
        chunk.uploadDepth = slot.depth;
        chunk.readbackFrame = false;
        chunk.readbackDepth = false;
        const size_t words = static_cast<size_t>(rect.x1 - rect.x0 + 1) * static_cast<size_t>(rect.y1 - rect.y0 + 1);
        std::vector<uint32_t>& data = slot.depth ? chunk.depthWords : chunk.frameWords;
        data.resize(words);
        memory_.Local().readRect(slot.view, static_cast<uint32_t>(rect.x0), static_cast<uint32_t>(rect.y0),
                                 static_cast<uint32_t>(rect.x1 - rect.x0 + 1), static_cast<uint32_t>(rect.y1 - rect.y0 + 1),
                                 data.data());
        std::string error;
        if (!gpu_.Execute(chunk, error)) {
            Fail(error);
            return;
        }
        counters.cacheLoadWords += words;
    }

    // Makes `slot` hold at least `rect` loaded from the shadow memory. The loaded rectangle grows to the union;
    // only the part outside the old rectangle is uploaded (IN-350: up to four strips), so the slot's own GPU-newer
    // pixels stay where they are and no write-back is needed. Every other slot that holds part of the new area is
    // written back and dropped first (two images never hold the same bytes).
    void Load(ResidentSlot& slot, const PixelRect& rect) {
        if (Contains(slot.loaded, rect)) return;
        StageTimer timer(counters.cacheLoadUs);
        const PixelRect old = slot.loaded;
        const PixelRect wanted = old.Empty() ? rect : UnionRects(old, rect);
        // IN-398 (SCALE-ONLY-PLAN S1): another slot holding this memory under an interchangeable view (GT4 reads its
        // race frame as CT32 and CT24 in turn). The overlap pass below writes it back and drops it, but its images
        // stay until reused, so the refreshes of this load keep its scaled blocks wherever the words still match.
        static const bool carryEnabled = std::getenv("COREGS_HW_SCALED_CARRY") == nullptr ||
                                         std::strcmp(std::getenv("COREGS_HW_SCALED_CARRY"), "0") != 0;
        bool carrying = false;
        if (carryEnabled && gpu_.Scale() > 1) {
            // A resident slot of the same memory first, else one dropped with its images intact.
            for (int pass = 0; pass < 2 && !carrying; ++pass) {
                for (const ResidentSlot& other : slots_) {
                    if (&other == &slot || other.depth != slot.depth) continue;
                    const bool resident = other.hasIdentity && !other.loaded.Empty() && !other.scaledStale;
                    // A colour source must still be the newest copy of its memory (a stale scaled remnant would be
                    // shown); a depth source is only compared, never shown, so its per-pixel 1x match is enough (the
                    // depth refreshed in 1x blocks drew the dark stripes on the track: the road's layers fail the
                    // depth test at scale in rows).
                    const bool dropped = !other.hasIdentity && other.droppedIntact &&
                                         (other.depth ||
                                          MemoryVersion(SpanOfRect(other.droppedView, other.droppedLoaded)) ==
                                              other.droppedVersion);
                    if (pass == 1 && !other.hasIdentity && other.droppedIntact && !dropped &&
                        SharedViewMask(other.droppedView, slot.view) != 0u) {
                        SlotTrace(NewestWriter(SpanOfRect(other.droppedView, other.droppedLoaded)), other, other.droppedLoaded);  // diagnostics: the newer write
                    }
                    if (pass == 0 ? !resident : !dropped) continue;
                    const memory::BufferView& otherView = pass == 0 ? other.view : other.droppedView;
                    const PixelRect& otherRect = pass == 0 ? other.loaded : other.droppedLoaded;
                    const uint32_t mask = SharedViewMask(otherView, slot.view);
                    if (mask == 0u) continue;
                    gpu_.SetRefreshCarry(slot.depth, other.gpuIndex, mask, otherRect.x0, otherRect.y0, otherRect.x1,
                                         otherRect.y1);
                    SlotTrace(pass == 0 ? "carry-resident" : "carry-dropped", other, otherRect);
                    carrying = true;
                    break;
                }
            }
        }
        {
            ReasonScope scope(writebackReason_, WritebackReason::LoadOverlap);
            CoherentForWrite(SpanOfRect(slot.view, wanted), &slot, true);
        }
        if (old.Empty()) {
            UploadIntoSlot(slot, wanted);
        } else {
            // wanted minus old: the rows above and below old over the full width of wanted, then the columns
            // left and right of old inside old's rows.
            if (wanted.y0 < old.y0) UploadIntoSlot(slot, PixelRect{wanted.x0, wanted.y0, wanted.x1, old.y0 - 1});
            if (wanted.y1 > old.y1) UploadIntoSlot(slot, PixelRect{wanted.x0, old.y1 + 1, wanted.x1, wanted.y1});
            if (wanted.x0 < old.x0) UploadIntoSlot(slot, PixelRect{wanted.x0, old.y0, old.x0 - 1, old.y1});
            if (wanted.x1 > old.x1) UploadIntoSlot(slot, PixelRect{old.x1 + 1, old.y0, wanted.x1, old.y1});
        }
        if (carrying) gpu_.ClearRefreshCarry();
        slot.loaded = wanted;
        counters.cacheLoads++;
    }

    // The rectangle to load for a batch: its own rectangle, widened to the context's scissor when that is a
    // plausible buffer (inside the buffer width, at most 512 lines) and the widened area holds no other resident
    // target (IN-350: widening into a neighbour, such as GT4's depth buffer right after its colour buffer, would
    // drop that neighbour on every switch). Later batches of the same target then rarely need to grow it.
    PixelRect LoadRectFor(const Translation& t, const memory::BufferView& view, const ResidentSlot& slot) const {
        const int32_t bufferWidth = static_cast<int32_t>(std::max<uint32_t>(view.bufferWidth, 1u) * 64u);
        const PixelRect& sc = t.scissor;
        if (sc.Empty() || sc.x1 >= bufferWidth || sc.y1 - sc.y0 >= 512 || sc.y1 >= gpu::kGpuSurface) return t.rect;
        const PixelRect widened = UnionRects(t.rect, sc);
        const BlockSpan span = SpanOfRect(view, widened);
        for (const ResidentSlot& other : slots_) {
            if (&other != &slot && SpansOverlap(LoadedSpan(other), span)) return t.rect;
        }
        return widened;
    }

    // IN-346: the resident colour slot holding a texture (exact view, the sampled texels loaded), or -1. Only in
    // cache mode. The slot is protected from eviction for this batch. IN-380: the batch's footprint must be loaded,
    // not the whole power-of-two extent (texels outside the footprint are copied but never sampled).
    int32_t TextureSlotFor(const memory::BufferView& view, uint32_t width, uint32_t height, const PixelRect& footprint) {
        if (!cacheMode_ || !textureFromTargets_) return -1;
        const PixelRect extent{0, 0, static_cast<int32_t>(width) - 1, static_cast<int32_t>(height) - 1};
        const PixelRect needed = footprint.Empty() ? extent : footprint;
        for (ResidentSlot& slot : slots_) {
            if (slot.depth || !slot.hasIdentity || !SameView(slot.view, view) || !Contains(slot.loaded, needed)) continue;
            protectedSlot_ = &slot;
            slot.lastUse = ++slotUses_;
            return static_cast<int32_t>(slot.gpuIndex);
        }
        return -1;
    }

    // ---- GLM (M10, chapter 04.9, IN-347) ------------------------------------------------------------------
    // The GPU mirror of local memory. glmPageSequence_[page] is the shadow-memory write sequence the GLM copy of
    // that page reflects (0: never uploaded). A GLM read of span R: upload every page of R (and of the drawn
    // rectangles of the targets that overlap R) whose shadow-memory content changed, then swizzle those
    // targets' drawn rectangles into the GLM; the GLM then holds what the shadow memory would hold after a
    // write-back, without one.
    bool UploadStalePages(const BlockSpan& span) {
        std::vector<std::pair<uint32_t, uint32_t>>& ranges = staleRanges_;  // reused (IN-365; ~2,000 calls per frame)
        ranges.clear();
        auto stale = [&](uint32_t page) {
            return glmPageSequence_[page] == 0 || glmPageSequence_[page] < pageWrites_[page] + 1;
        };
        auto add = [&](uint32_t page) {
            if (glmOwned_[page]) {
                // The shadow memory changed under a GLM-owned page without a download first: a coherence bug.
                Fail("GLM-owned page " + std::to_string(page) + " changed in the shadow memory");
                return;
            }
            const uint32_t offset = page * 8192u;
            if (!ranges.empty() && ranges.back().first + ranges.back().second == offset) {
                ranges.back().second += 8192u;
            } else {
                ranges.push_back({offset, 8192u});
            }
            glmPageSequence_[page] = pageWrites_[page] + 1;  // +1: 0 means "never uploaded"
            glmPageUploadClock_[page] = ++glmUploadClock_;
        };
        if (span.count >= kAllBlocks) {
            for (uint32_t page = 0; page < memory::kPagesTotal; ++page) {
                if (stale(page)) add(page);
            }
        } else if (span.count != 0) {
            const uint32_t first = span.begin / kBlocksPerPage;
            const uint32_t last = (span.begin + span.count - 1u) / kBlocksPerPage;
            for (uint32_t page = first; page <= last; ++page) {
                const uint32_t wrapped = page % memory::kPagesTotal;
                if (stale(wrapped)) add(wrapped);
            }
        }
        if (failed_) return false;
        if (ranges.empty()) return true;
        std::string error;
        if (!gpu_.UploadGlm(memory_.Local().bytes().data(), ranges, error)) {
            Fail(error);
            return false;
        }
        counters.glmPagesUploaded += ranges.size();
        return true;
    }

    // `force`: the texture must be decoded on the GPU even without GPU-newer data under it (indexed textures
    // while the GPU CLUT is authoritative, IN-349).
    bool PrepareGlm(const memory::BufferView& view, uint32_t width, uint32_t height, bool force = false) {
        if (!cacheMode_ || !glmEnabled_) return false;
        const BlockSpan span = SpanOfRect(view, 0, 0, static_cast<int32_t>(width) - 1, static_cast<int32_t>(height) - 1);
        std::vector<ResidentSlot*> sources;
        for (ResidentSlot& slot : slots_) {
            if (slot.drawn.Empty() || !SpansOverlap(SpanOfRect(slot.view, slot.drawn), span)) continue;
            // A drawn rectangle wider than its buffer wraps into the next page row: two pixels share an address
            // and a parallel swizzle has no defined last writer. Use the write-back path then.
            const int32_t bufferWidth = static_cast<int32_t>(std::max<uint32_t>(slot.view.bufferWidth, 1u) * 64u);
            if (slot.drawn.x1 >= bufferWidth) {
                // Write it back instead (the shadow memory then holds it; the GLM upload below picks it up).
                ReasonScope scope(writebackReason_, WritebackReason::Texture);
                WriteBack(slot);
                if (!force) return false;
                continue;
            }
            sources.push_back(&slot);
        }
        if (sources.empty() && !force && !AnyGlmOwned(span)) {
            return false;  // nothing GPU-newer: the shadow memory decode is exact
        }
        if (!UploadStalePages(span)) return false;
        for (ResidentSlot* slot : sources) {
            if (!SwizzleSlotIntoGlm(*slot)) return false;
        }
        return true;
    }

    // IN-348 candidates: small untextured sprite batches whose memory no resident target holds (IN-350: a small
    // sprite on a resident target, such as a HUD box on the frame buffer, stays on the GPU; drawing it on the CPU
    // would write back and reload the whole target).
    bool SmallUntexturedSprites(const Batch& batch, const Translation& t) const {
        if (t.hasTexture || t.rect.Empty()) return false;
        const int64_t area = static_cast<int64_t>(t.rect.x1 - t.rect.x0 + 1) * (t.rect.y1 - t.rect.y0 + 1);
        if (area > 64 * 64) return false;
        for (const PrimRecord& record : batch.prims) {
            if (record.type != PrimType::Sprite) return false;
        }
        const BlockSpan frameSpan = SpanOfRect(t.frameView, t.rect);
        const BlockSpan depthSpan = SpanOfRect(t.zView, t.rect);
        if (AnyGlmOwned(frameSpan) || AnyGlmOwned(depthSpan)) return false;  // would need a download (IN-350)
        for (const ResidentSlot& slot : slots_) {
            const BlockSpan loaded = LoadedSpan(slot);
            if (SpansOverlap(loaded, frameSpan) || SpansOverlap(loaded, depthSpan)) return false;
        }
        return true;
    }

    // ---- GPU CLUT (chapter 07.7, IN-349) --------------------------------------------------------------
    // In cache mode with the GLM, every performed CLUT load is repeated on the GPU from the GLM, the GPU CLUT is
    // authoritative and indexed textures are expanded from it on the GPU; the CPU CLUT unit still decides when
    // a load happens. So a palette the game drew on the GPU needs no write-back.
    bool GpuClutActive() const { return cacheMode_ && glmEnabled_ && gpuClutEnabled_; }

    void OnClutLoaded(const clut::Tex0Write& write, bool deferrable = true) {
        // IN-383: the strips of GT4's tone sequence are collected with the palette load that precedes each of them, so
        // a page that has to run its batches afterwards can repeat the load instead of using a later palette.
        toneClut_ = write;
        toneClutValid_ = true;
        if (!GpuClutActive() || failed_) return;
        // IN-383: a collected strip's palette is not needed by its page's dispatch (the dispatch reads the palette
        // itself from the temporary page), and the eight loads of a page all write different groups over the same
        // 256 entries, so only the LAST one can be observed: the load of the next strip is held and the previous one
        // is dropped. The held load is recorded before anything else reads the CLUT (another load, the page's flush,
        // a reset or a mode change), which is the same palette the normal path leaves behind.
        if (deferrable && toneHle_ && tonePage_.active && tonePage_.strips < 8 &&
            write.cbp == tonePage_.absoluteCbp[0] + kToneQuadrantBlock[tonePage_.strips] &&
            write.psm == static_cast<uint32_t>(memory::Psm::T8) && write.cpsm == 0 && write.csm == 0 && write.csa == 0) {
            if (tonePendingClutValid_) ++counters.toneClutLoadsDropped;  // superseded by this strip's own group
            tonePendingClut_ = write;
            tonePendingClutValid_ = true;
            ++counters.toneClutLoadsHeld;
            return;
        }
        RecordTonePendingClut();
        RecordGpuClutLoad(write);
    }

    // The held strip palette load, recorded on the GPU now (IN-383). The GPU CLUT ends as the normal path leaves it.
    void RecordTonePendingClut() {
        if (!tonePendingClutValid_) return;
        const clut::Tex0Write write = tonePendingClut_;
        tonePendingClutValid_ = false;
        RecordGpuClutLoad(write);
    }

    void RecordGpuClutLoad(const clut::Tex0Write& write) {
        if (!GpuClutActive() || failed_) return;
        StageTimer timer(counters.clutUs);
        if (!gpuClutSynced_) SyncGpuClut();
        const BlockSpan span = write.csm == 0 ? BlockSpan{write.cbp & ~31u, 64u} : BlockSpan{0, kAllBlocks};
        // Make the GLM current for the palette (stale pages, GPU-newer targets), as PrepareGlm does for textures.
        if (!UploadStalePages(span)) return;
        for (ResidentSlot& slot : slots_) {
            if (slot.drawn.Empty() || !SpansOverlap(SpanOfRect(slot.view, slot.drawn), span)) continue;
            const int32_t bufferWidth = static_cast<int32_t>(std::max<uint32_t>(slot.view.bufferWidth, 1u) * 64u);
            if (slot.drawn.x1 >= bufferWidth) {
                ReasonScope scope(writebackReason_, WritebackReason::Clut);
                WriteBack(slot);
                if (!UploadStalePages(SpanOfRect(slot.view, PixelRect{0, 0, bufferWidth - 1, 2047}))) return;
                continue;
            }
            if (!SwizzleSlotIntoGlm(slot)) return;
        }
        std::string error;
        if (!gpu_.LoadClutFromGlm(write.cbp, write.cpsm, write.csm, write.csa, write.psm, write.texclutCbw,
                                  write.texclutCou, write.texclutCov, error)) {
            Fail(error);
            return;
        }
        counters.gpuClutLoads++;
    }

    // The GPU CLUT starts as a copy of the CPU CLUT buffer (start-up, GS reset).
    void SyncGpuClut() {
        std::string error;
        if (!gpu_.UploadClut(frontEnd_.Clut().Halfwords(), error)) {
            Fail(error);
            return;
        }
        gpuClutSynced_ = true;
    }

    // One translated batch through the resident targets. Returns false (nothing done) when the batch must
    // take the write-back path instead.
    bool ExecuteResident(const Translation& t, std::string& error) {
        const bool depthUsed = DepthUsed(t.state);
        const BlockSpan frameSpan = SpanOfRect(t.frameView, t.rect);
        const BlockSpan depthSpan = depthUsed ? SpanOfRect(t.zView, t.rect) : BlockSpan{};
        if (depthUsed && SpansOverlap(frameSpan, depthSpan)) return false;  // colour and depth share memory
        ResidentSlot& colour = Acquire(false, t.frameView);
        ResidentSlot* depth = depthUsed ? &Acquire(true, t.zView) : nullptr;
        Load(colour, LoadRectFor(t, t.frameView, colour));
        if (depth) Load(*depth, LoadRectFor(t, t.zView, *depth));
        if (failed_) return true;
        // The colour slot's load may have dropped the depth slot (or the reverse) when they overlap; then the
        // batch takes the write-back path with both stored first.
        if (depth && (!colour.hasIdentity || !depth->hasIdentity || colour.loaded.Empty() || depth->loaded.Empty() ||
                      SpansOverlap(LoadedSpan(colour), LoadedSpan(*depth)))) {
            ReasonScope scope(writebackReason_, WritebackReason::Fallback);
            WriteBack(colour);
            Drop(colour);
            WriteBack(*depth);
            Drop(*depth);
            return false;
        }
        if (!Contains(colour.loaded, t.rect) || (depth && !Contains(depth->loaded, t.rect))) {
            error = "resident target does not cover the batch rectangle";
            return true;
        }
        size_t begin = 0;
        while (begin < t.draws.size()) {
            size_t end = begin;
            uint32_t vertexCount = 0;
            while (end < t.draws.size() && vertexCount + t.draws[end].count <= gpu::kGpuMaxVertices) {
                vertexCount += t.draws[end].count;
                ++end;
            }
            if (end == begin) {
                error = "a primitive exceeds the vertex buffer";
                return true;
            }
            const uint32_t baseVertex = t.draws[begin].first;
            gpu::GpuChunk chunk;
            chunk.rectX0 = t.rect.x0;
            chunk.rectY0 = t.rect.y0;
            chunk.rectX1 = t.rect.x1;
            chunk.rectY1 = t.rect.y1;
            chunk.colourSlot = colour.gpuIndex;
            chunk.depthSlot = depth ? depth->gpuIndex : gpu::kGpuScratchDepthSlot;
            // IN-384: the scaled images of this pair cover the target's row width and the rows this chunk reaches.
            chunk.targetWidth = std::max<uint32_t>(t.frameView.bufferWidth, 1u) * 64u;
            chunk.uploadFrame = chunk.uploadDepth = chunk.readbackFrame = chunk.readbackDepth = false;
            chunk.state = t.state;
            chunk.hasTexture = t.hasTexture;
            chunk.textureSlot = t.textureSlot;
            chunk.glmTexture = t.glmTexture;
            chunk.glmBp = t.glmBp;
            chunk.glmBw = t.glmBw;
            chunk.glmPsm = t.glmPsm;
            chunk.glmIndexed = t.glmIndexed;
            chunk.glmGpuClut = t.glmGpuClut;
            chunk.glmCpsm = t.glmCpsm;
            chunk.glmCsa = t.glmCsa;
            chunk.glmTa0 = t.glmTa0;
            chunk.glmTa1 = t.glmTa1;
            chunk.glmAem = t.glmAem;
            chunk.palette = t.palette;
            chunk.texWidth = t.texWidth;
            chunk.texHeight = t.texHeight;
            chunk.texCopyWidth = t.texCopyWidth;
            chunk.texCopyHeight = t.texCopyHeight;
            chunk.texelSpan = t.texels.data();  // IN-365: borrowed, no copy per chunk
            chunk.texelSpanCount = t.texels.size();
            chunk.vertexSpan = t.vertices.data() + baseVertex;
            chunk.vertexSpanCount = vertexCount;
            chunk.draws.reserve(end - begin);
            for (size_t i = begin; i < end; ++i) {
                gpu::GpuDraw draw = t.draws[i];
                draw.first -= baseVertex;
                chunk.draws.push_back(draw);
            }
            if (!gpu_.Execute(chunk, error)) return true;
            counters.gpuChunks++;
            counters.gpuDraws += chunk.draws.size();
            begin = end;
        }
        colour.drawn = colour.drawn.Empty() ? t.rect : UnionRects(colour.drawn, t.rect);
        colour.drawSequence++;
        if (depth) depth->drawSequence++;
        if (depth) depth->drawn = depth->drawn.Empty() ? t.rect : UnionRects(depth->drawn, t.rect);
        counters.cacheResidentBatches++;
        return true;
    }

    // ---- IN-379: GT4's RGB tone curve at scale (chapter 09.4 variant A, policy 2) -------------------------
    // Every race frame, GT4 reads its 32-bit frame buffer page by page through 8-bit (T8) views and a ramp CLUT into
    // a temporary page and writes the result back, through 8-bit views again, into the same memory as a CT24
    // target. All of that runs at 1x through the GLM (exact), which left the frame's scaled image as the 1x result
    // blown up (owner test 7). The sequence starts with a T8 batch that reads a resident 32-bit colour target: that
    // target's scaled image and 1x words are saved. It ends at the first batch that is not T8-textured (or a VBlank
    // or a compose): the saved view is made resident again (its 1x words are the sequence's exact result) and its
    // scaled image is rebuilt from the saved one through the per-channel table the 1x words imply
    // (hw_scale_tone_carry.comp; any pixel the table does not explain stays 1x). Memory is never touched by this.
    struct ToneRun {
        bool active = false;
        memory::BufferView source{};
        PixelRect saved{};
        uint64_t firstUse = 0;  // slotUses_ when the sequence started (IN-382)
    };
    ToneRun toneRun_;

    void TrackToneRun(const Batch& batch) {
        if (!cacheMode_ || gpu_.Scale() <= 1 || failed_ || auxiliary_) return;
        const ContextRegs& ctx = batch.state.ctx;
        const bool textured = ((batch.state.effectivePrim >> primbit::kTme) & 1u) != 0;
        const bool indexed8 = textured && tex0::Psm(ctx.tex0) == static_cast<uint32_t>(memory::Psm::T8);
        if (toneRun_.active) {
            if (indexed8) return;  // still inside the sequence
            FinishToneRun();
        }
        if (!indexed8) return;
        const memory::BufferView texture{tex0::Tbp0(ctx.tex0), tex0::Tbw(ctx.tex0), memory::Psm::T8};
        const int32_t width = 1 << std::min<uint32_t>(tex0::Tw(ctx.tex0), 10u);
        const int32_t height = 1 << std::min<uint32_t>(tex0::Th(ctx.tex0), 10u);
        const BlockSpan read = SpanOfRect(texture, 0, 0, width - 1, height - 1);
        for (const ResidentSlot& slot : slots_) {
            if (slot.depth || !slot.hasIdentity || slot.loaded.Empty()) continue;
            if (slot.view.psm != memory::Psm::Ct32 && slot.view.psm != memory::Psm::Ct24) continue;
            if (!SpansOverlap(LoadedSpan(slot), read)) continue;
            // IN-384: the tone source is copied out of the slot's own scaled image, so its rectangle is that extent.
            uint32_t scaledWidth = 0, scaledHeight = 0;
            gpu_.ScaledExtent(false, slot.gpuIndex, scaledWidth, scaledHeight);
            const PixelRect saved{std::max(slot.loaded.x0, 0), std::max(slot.loaded.y0, 0),
                                  std::min(slot.loaded.x1, static_cast<int32_t>(scaledWidth) - 1),
                                  std::min(slot.loaded.y1, static_cast<int32_t>(scaledHeight) - 1)};
            if (saved.Empty()) return;
            std::string error;
            if (!gpu_.SaveToneSource(slot.gpuIndex, saved.x0, saved.y0, saved.x1, saved.y1, error)) {
                Fail(error);
                return;
            }
            toneRun_.active = true;
            toneRun_.source = slot.view;
            toneRun_.saved = saved;
            toneRun_.firstUse = slotUses_;
            counters.toneRuns++;
            // IN-382 (owner performance requirement): the sequence's draws are not repeated at scale. About 280 of
            // GT4's ~320 runs per race frame are this sequence (140 pages x temporary page and write-back), and its
            // scaled result is rebuilt by the carry anyway; slots it touched are refreshed at its end.
            gpu_.PauseScaled(true);
            return;
        }
    }

    // IN-381: one 13.4 deinterlace of the scaled frame in the display image, with the GPU history (the same rules as
    // display::Deinterlacer, in native rows).
    bool DeinterlaceGpu(display::DeinterlaceMode mode, int vblankField, uint32_t width, uint32_t height,
                        std::string& error) {
        GpuDeinterlaceHistory& h = gpuDeinterlace_;
        if (h.width != width || h.height != height) {
            h.Clear();
            h.width = width;
            h.height = height;
        }
        const int rowParity = (vblankField ^ (settings_.bottomFieldFirst ? 1 : 0)) & 1;
        gpu::GpuDrawer::DeinterlaceStep step;
        step.width = width;
        step.height = height;
        step.rowParity = static_cast<uint32_t>(rowParity);
        step.field = static_cast<uint32_t>(vblankField & 1);
        // IN-386: the reference deinterlaces the merged image with a "row" of ONE output row (a one-line field
        // difference becomes a sub-native comb), while IN-381 used one native row (a full native-row ghost at scale,
        // the owner's doubled menu lines). Default output rows; COREGS_HW_DEINTERLACE_ROWS=native restores IN-381.
        {
            static const bool nativeRows = std::getenv("COREGS_HW_DEINTERLACE_ROWS") != nullptr &&
                                           std::strcmp(std::getenv("COREGS_HW_DEINTERLACE_ROWS"), "native") == 0;
            step.rowStep = nativeRows ? gpu_.Scale() : 1u;
        }
        // The slot for this field: a free one, else the oldest (the CPU history keeps the last four fields).
        uint32_t current = gpu::kGpuDeinterlaceHistory;
        for (uint32_t i = 0; i < gpu::kGpuDeinterlaceHistory && current == gpu::kGpuDeinterlaceHistory; ++i) {
            if (!h.valid[i]) current = i;
        }
        if (current == gpu::kGpuDeinterlaceHistory) {
            current = 0;
            for (uint32_t i = 1; i < gpu::kGpuDeinterlaceHistory; ++i) {
                if (h.age[i] < h.age[current]) current = i;
            }
        }
        // Previous same- and opposite-parity fields: the most recent older entries (IN-39, IN-337), and IN-396 the
        // older opposite-parity field (the reference's motion test compares the woven field with it).
        uint64_t sameAge = 0, otherAge = 0, olderAge = 0;
        for (uint32_t i = 0; i < gpu::kGpuDeinterlaceHistory; ++i) {
            if (i == current || !h.valid[i]) continue;
            if (h.parity[i] == rowParity) {
                if (step.previousSame == gpu::kGpuDeinterlaceHistory || h.age[i] > sameAge) {
                    step.previousSame = i;
                    sameAge = h.age[i];
                }
            } else if (step.previousOther == gpu::kGpuDeinterlaceHistory || h.age[i] > otherAge) {
                step.olderOther = step.previousOther;
                olderAge = otherAge;
                step.previousOther = i;
                otherAge = h.age[i];
            } else if (step.olderOther == gpu::kGpuDeinterlaceHistory || h.age[i] > olderAge) {
                step.olderOther = i;
                olderAge = h.age[i];
            }
        }
        step.current = current;
        step.haveOutput = h.output;
        step.initWoven = !h.woven;
        switch (mode) {
            case display::DeinterlaceMode::Weave: step.mode = 2; break;
            case display::DeinterlaceMode::Blend: step.mode = 3; break;
            case display::DeinterlaceMode::Bob: step.mode = 4; break;
            default: step.mode = 5; break;
        }
        if (!gpu_.DeinterlaceScaled(step, error)) return false;
        if (step.mode == 5) {
            h.valid[current] = true;  // only the adaptive mode keeps a field history (as the CPU deinterlacer)
            h.parity[current] = rowParity;
            h.age[current] = ++h.clock;
            h.output = true;
        }
        if (step.mode == 2 || step.mode == 3) h.woven = true;
        counters.deinterlacedFrames++;
        return true;
    }

    void FinishToneRun() {
        if (!toneRun_.active) return;
        toneRun_.active = false;
        gpu_.PauseScaled(false);  // IN-382
        if (failed_) return;
        // IN-382: every other slot the paused sequence used gets its scaled image back from its 1x words (GT4: the
        // temporary page), so no scaled image is left stale.
        for (const ResidentSlot& used : slots_) {
            if (!used.hasIdentity || used.loaded.Empty() || used.lastUse <= toneRun_.firstUse) continue;
            // The source memory (any view of it) is reloaded and rebuilt by the carry below.
            if (SpansOverlap(LoadedSpan(used), SpanOfRect(toneRun_.source, toneRun_.saved))) {
                const_cast<ResidentSlot&>(used).scaledStale = true;  // IN-398: rebuilt by the carry, not refreshed
                continue;
            }
            std::string error;
            if (!gpu_.RefreshScaledSlot(used.depth, used.gpuIndex, used.loaded.x0, used.loaded.y0, used.loaded.x1,
                                        used.loaded.y1, error)) {
                Fail(error);
                return;
            }
        }
        // The result is in the same memory, maybe under another 32-bit view (GT4 writes it back as CT24); the next
        // reader (GT4: the CT32 texture of the CT16 display conversion) finds the source view resident at scale.
        ResidentSlot& slot = Acquire(false, toneRun_.source);
        Load(slot, toneRun_.saved);
        if (failed_ || !Contains(slot.loaded, toneRun_.saved)) return;
        // Diagnostic switch (IN-384 follow-up): with COREGS_HW_TONE_CARRY=0 the sequence's scaled result is left at the
        // plain 1x refresh, which isolates a scaled-image artifact that comes from the carry (the picture then shows
        // the race at 1x, so this is a bisection tool, not a mode).
        static const bool carryEnabled = std::getenv("COREGS_HW_TONE_CARRY") == nullptr ||
                                         std::strcmp(std::getenv("COREGS_HW_TONE_CARRY"), "0") != 0;
        if (!carryEnabled) return;
        std::string error;
        if (!gpu_.CarryTone(slot.gpuIndex, toneRun_.saved.x0, toneRun_.saved.y0, toneRun_.saved.x1, toneRun_.saved.y1,
                            error)) {
            Fail(error);
            return;
        }
        counters.toneCarries++;
        // IN-384 follow-up (owner test 11, sporadic 1x frames): the tone sequence rewrites the frame buffer every
        // race frame, so a display shadow saved before it no longer matches the GLM and its per-pixel check falls back
        // to 1x for the WHOLE frame. The slot is resident again with the carried image, so re-save the shadow from it:
        // the shadow then holds the carried 6x picture and the 1x words the sequence produced, and its check passes.
        SaveDisplayShadow(slot);
        DumpToneRun(slot.gpuIndex);
    }

    // Diagnostics (COREGS_HW_DISPLAY_DUMP=<folder>): the first COREGS_HW_DISPLAY_DUMP_LIMIT composes (default 1) write the
    // displayed circuits' source images and the composed frame as PPM files, so one VBlank can be split into the
    // slot's scaled image, the shadow copy and the composed frame (IN-387, the owner's flat-menu artifact).
    void DumpDisplayImages(const std::array<uint32_t, 2>& sources, uint32_t outWidth, uint32_t outHeight,
                           const display::DisplayLayout& layout) {
        static const char* folder = std::getenv("COREGS_HW_DISPLAY_DUMP");
        if (!folder || !*folder) return;
        static const int limit = [] {
            const char* text = std::getenv("COREGS_HW_DISPLAY_DUMP_LIMIT");
            return text ? std::atoi(text) : 1;
        }();
        static int written = 0;
        if (written >= limit) return;
        ++written;
        const uint32_t scale = gpu_.Scale();
        auto writePpm = [](const char* path, uint32_t width, uint32_t height, const std::vector<uint32_t>& words) {
            std::FILE* file = std::fopen(path, "wb");
            if (!file) return;
            std::fprintf(file, "P6\n%u %u\n255\n", width, height);
            for (uint32_t word : words) {
                const unsigned char rgb[3] = {static_cast<unsigned char>(word), static_cast<unsigned char>(word >> 8),
                                              static_cast<unsigned char>(word >> 16)};
                std::fwrite(rgb, 1, 3, file);
            }
            std::fclose(file);
        };
        for (std::size_t i = 0; i < 2; ++i) {
            if (sources[i] == 0) continue;
            const display::CircuitRect& circuit = layout.circuits[i];
            const uint32_t width = circuit.read_width * scale;
            const uint32_t height = circuit.read_height * scale;
            std::vector<uint32_t> words;
            std::string error;
            const uint32_t which = sources[i] <= gpu::kGpuColourSlots ? 0u : 3u;
            const uint32_t index = sources[i] <= gpu::kGpuColourSlots ? sources[i] - 1u : sources[i] - gpu::kGpuColourSlots - 1u;
            char path[512];
            std::snprintf(path, sizeof path, "%s\\circuit%zu-%s%u.ppm", folder, i,
                          which == 0 ? "slot" : "shadow", index);
            if (gpu_.ReadImageDebug(which, index, width, height, words, error)) writePpm(path, width, height, words);
        }
        std::vector<uint32_t> words;
        std::string error;
        if (gpu_.ReadImageDebug(4, 0, outWidth, outHeight, words, error)) {
            char path[512];
            std::snprintf(path, sizeof path, "%s\\composed.ppm", folder);
            writePpm(path, outWidth, outHeight, words);
        }
        words.clear();
        if (gpu_.ReadImageDebug(5, 0, outWidth, outHeight, words, error)) {  // IN-387: before the deinterlace
            char path[512];
            std::snprintf(path, sizeof path, "%s\\merge.ppm", folder);
            writePpm(path, outWidth, outHeight, words);
        }
        // IN-387: the 1x words of the same slot, to tell a 6x draw from a magnification of 1x content.
        for (std::size_t i = 0; i < 2; ++i) {
            if (sources[i] == 0 || sources[i] > gpu::kGpuColourSlots) continue;
            const display::CircuitRect& circuit = layout.circuits[i];
            words.clear();
            if (!gpu_.ReadImageDebug(2, sources[i] - 1u, circuit.read_width, circuit.read_height, words, error)) continue;
            char path[512];
            std::snprintf(path, sizeof path, "%s\\circuit%zu-1x.ppm", folder, i);
            writePpm(path, circuit.read_width, circuit.read_height, words);
        }
        // IN-387: every colour slot's scaled image and 1x words, so the slot that holds a layer can be found.
        for (std::uint32_t slot = 0; slot < gpu::kGpuColourSlots; ++slot) {
            std::uint32_t width = 0, height = 0;
            gpu_.ScaledExtent(false, slot, width, height);
            if (width == 0 || height == 0) continue;
            words.clear();
            if (gpu_.ReadImageDebug(0, slot, width * scale, height * scale, words, error)) {
                char path[512];
                std::snprintf(path, sizeof path, "%s\\slot%u-scaled.ppm", folder, slot);
                writePpm(path, width * scale, height * scale, words);
            }
            words.clear();
            if (gpu_.ReadImageDebug(2, slot, width, height, words, error)) {
                char path[512];
                std::snprintf(path, sizeof path, "%s\\slot%u-1x.ppm", folder, slot);
                writePpm(path, width, height, words);
            }
        }
        // Every depth slot's scaled image and 1x words as raw little-endian 32-bit words (width x height in the
        // file name), so a depth test at scale can be compared with the 1x one.
        auto writeRaw = [](const char* path, const std::vector<uint32_t>& words) {
            if (std::FILE* file = std::fopen(path, "wb")) {
                std::fwrite(words.data(), sizeof(uint32_t), words.size(), file);
                std::fclose(file);
            }
        };
        for (std::uint32_t slot = 0; slot < gpu::kGpuDepthSlots; ++slot) {
            std::uint32_t width = 0, height = 0;
            gpu_.ScaledExtent(true, slot, width, height);
            if (width == 0 || height == 0) continue;
            char path[512];
            words.clear();
            if (gpu_.ReadImageDebug(6, slot, width * scale, height * scale, words, error)) {
                std::snprintf(path, sizeof path, "%s\\depth%u-scaled-%ux%u.bin", folder, slot, width * scale,
                              height * scale);
                writeRaw(path, words);
            }
            words.clear();
            if (gpu_.ReadImageDebug(7, slot, width, height, words, error)) {
                std::snprintf(path, sizeof path, "%s\\depth%u-1x-%ux%u.bin", folder, slot, width, height);
                writeRaw(path, words);
            }
        }
    }

    // Diagnostics (COREGS_HW_TONE_DUMP=<folder>): the first three tone sequences write the saved scaled source, the
    // carried scaled image and the 1x result as PPM files.
    void DumpToneRun(uint32_t gpuSlot) {
        static const char* folder = std::getenv("COREGS_HW_TONE_DUMP");
        if (!folder || !*folder || counters.toneCarries > 3) return;
        const uint32_t scale = gpu_.Scale();
        const uint32_t sw = static_cast<uint32_t>(toneRun_.saved.x1 + 1) * scale;
        const uint32_t sh = static_cast<uint32_t>(toneRun_.saved.y1 + 1) * scale;
        const struct {
            uint32_t which, width, height;
            const char* name;
        } images[3] = {{1, sw, sh, "source-scaled"}, {0, sw, sh, "carried-scaled"},
                       {2, sw / scale, sh / scale, "result-1x"}};
        for (const auto& image : images) {
            std::vector<uint32_t> words;
            std::string error;
            if (!gpu_.ReadImageDebug(image.which, gpuSlot, image.width, image.height, words, error)) continue;
            char path[512];
            std::snprintf(path, sizeof path, "%s\\tone-%llu-%s.ppm", folder,
                          static_cast<unsigned long long>(counters.toneCarries), image.name);
            std::FILE* file = std::fopen(path, "wb");
            if (!file) continue;
            std::fprintf(file, "P6\n%u %u\n255\n", image.width, image.height);
            for (uint32_t word : words) {
                const unsigned char rgb[3] = {static_cast<unsigned char>(word), static_cast<unsigned char>(word >> 8),
                                              static_cast<unsigned char>(word >> 16)};
                std::fwrite(rgb, 1, 3, file);
            }
            std::fclose(file);
        }
    }

    // ---- IN-383: the tone sequence's strip batches (09.4 variant A; parity plan item 2) --------------------
    // The eight decal T8 strips that write one frame-buffer page back as CT24 are the sequence's bulk: about 1,120 of
    // the 1,540 tone batches per 480p race frame. Each strip is eight wide and 32 tall with one texel per pixel, so
    // the strips of one page are one GPU dispatch over the page (hw_tone_strips.comp): it reads the temporary page
    // and the strip's T8 index texture from the GLM and writes the page through its CT24 view. The page becomes
    // GLM-owned, as a parked target leaves it, so GPU readers see it without a download and CPU readers materialise
    // it. Matching is a refusal: anything that is not exactly the shape of 09.4's strip takes the normal path, and a
    // page whose palette bases do not decode as the temporary page's quadrants is replayed batch by batch (exact).
    struct ToneStripRecord {
        BatchState state;
        Vertex vertices[2];
        PrimRecord prim;
        PixelRect drawRect;
        FlushReason reason = FlushReason::Count;
        clut::Tex0Write clut;         // the palette load performed before this strip
        bool clutValid = false;
    };
    struct ToneStripPage {
        bool active = false;
        memory::BufferView destination{};
        int32_t x0 = 0, y0 = 0;
        uint32_t strips = 0;
        uint32_t absoluteCbp[8] = {};
        uint32_t indexBp = 0, indexBw = 0;  // the strip's T8 index texture (same for every strip of the page)
        bool templateOk = true;
        std::vector<ToneStripRecord> records;  // the deferred batches, replayed when the page does not decode
    };
    ToneStripPage tonePage_;
    clut::Tex0Write toneClut_{};  // the last palette load (IN-383: repeated when a collected strip runs normally)
    bool toneClutValid_ = false;
    clut::Tex0Write tonePendingClut_{};  // IN-383: a collected strip's palette load, held until it can be observed
    bool tonePendingClutValid_ = false;
    bool toneHle_ = false;        // reference-path option: the tone-strip HLE, set by its fixture
    bool verifyShuffle_ = false;  // COREGS_HW_VERIFY_SHUFFLE: run both and compare the page (09.4)

    // The per-batch half of the template: registers, blend, tests and texture format of one strip (09.4 variant A).
    bool MatchesToneStripTemplate(const Batch& batch) const {
        if (!cacheMode_ || !glmEnabled_ || failed_ || auxiliary_ || batch.Empty()) return false;
        if (batch.prims.size() != 1 || batch.vertices.size() != 2) return false;
        const PrimRecord& primitive = batch.prims[0];
        if (primitive.type != PrimType::Sprite || primitive.count != 2) return false;
        const BatchState& state = batch.state;
        const uint64_t attributes = state.effectivePrim;
        if ((attributes & 7u) != 6u) return false;                    // sprite
        if (((attributes >> primbit::kTme) & 1u) == 0) return false;  // textured
        if (((attributes >> primbit::kAbe) & 1u) != 0) return false;  // opaque (decal only)
        if (((attributes >> primbit::kFge) & 1u) != 0) return false;
        if (state.env.dthe != 0 || state.env.pabe != 0) return false;  // dither would change the RGB
        if (test::Ate(state.ctx.test) != 0 || test::Date(state.ctx.test) != 0) return false;
        if (test::Zte(state.ctx.test) != 0 && test::Ztst(state.ctx.test) != 1u) return false;  // 1 = ALWAYS
        if (zbuf::Zmsk(state.ctx.zbuf) == 0) return false;  // the strips never write depth
        if (frame::Fbmsk(state.ctx.frame) != 0xFF000000u) return false;
        if (frame::Psm(state.ctx.frame) != static_cast<uint32_t>(memory::Psm::Ct24)) return false;
        if (tex0::Psm(state.ctx.tex0) != static_cast<uint32_t>(memory::Psm::T8)) return false;
        if (tex0::Cpsm(state.ctx.tex0) != 0 || tex0::Csm(state.ctx.tex0) != 0 || tex0::Csa(state.ctx.tex0) != 0) {
            return false;  // CT32 palettes in the CSM1 layout, from the palette's first group (07.4)
        }
        if (tex0::Tfx(state.ctx.tex0) != 1 || tex0::Tcc(state.ctx.tex0) != 1) return false;  // decal
        if (tex0::Tw(state.ctx.tex0) != 3 || tex0::Th(state.ctx.tex0) != 5 || tex0::Tbw(state.ctx.tex0) != 1) {
            return false;  // an 8 x 32 index texture, one texel per pixel
        }
        return true;
    }

    // IN-383: appends the batch to the page being collected. False when it is not one of the template's strips.
    bool TryDeferToneStrip(const Batch& batch) {
        if (!toneHle_ || !MatchesToneStripTemplate(batch)) return false;
        const PixelRect& rect = batch.drawRect;
        if (rect.Empty() || rect.x0 < 0 || rect.y0 < 0 || rect.x1 - rect.x0 != 7 || rect.y1 - rect.y0 != 31) {
            return false;
        }
        const memory::BufferView destination{memory::blockOfPage(frame::Fbp(batch.state.ctx.frame)),
                                             frame::Fbw(batch.state.ctx.frame), memory::Psm::Ct24};
        const uint32_t cbp = tex0::Cbp(batch.state.ctx.tex0);
        const uint32_t indexBp = tex0::Tbp0(batch.state.ctx.tex0);
        const uint32_t indexBw = tex0::Tbw(batch.state.ctx.tex0);
        if (tonePage_.active) {
            const bool continues = SameView(tonePage_.destination, destination) &&
                                   rect.x0 == tonePage_.x0 + static_cast<int32_t>(8u * tonePage_.strips) &&
                                   rect.y0 == tonePage_.y0 && tonePage_.strips < 8 &&
                                   indexBp == tonePage_.indexBp && indexBw == tonePage_.indexBw;
            if (continues) {
                // The palette base of this strip must be exactly its quadrant of the temporary page. Anything else
                // is not 09.4's sequence: the batch then takes the normal path, after the collected prefix is
                // written (its palette is put back for it there).
                if (cbp != tonePage_.absoluteCbp[0] + kToneQuadrantBlock[tonePage_.strips]) return false;
            } else {
                FlushToneStrips();  // the previous page's strips come first (exact order)
            }
        }
        if (!tonePage_.active) {
            // One dispatch covers a page: its origin must be a page origin (its palette base is the page's first
            // quadrant, checked by construction here and by the block table at the flush).
            if ((rect.x0 & 63) != 0 || (rect.y0 & 31) != 0 || (cbp & (kBlocksPerPage - 1u)) != 0) return false;
            tonePage_.records.clear();
            tonePage_.destination = destination;
            tonePage_.x0 = rect.x0;
            tonePage_.y0 = rect.y0;
            tonePage_.indexBp = indexBp;
            tonePage_.indexBw = indexBw;
            tonePage_.strips = 0;
            tonePage_.templateOk = true;
            tonePage_.active = true;
        }
        ToneStripRecord record;
        record.state = batch.state;
        record.vertices[0] = batch.vertices[0];
        record.vertices[1] = batch.vertices[1];
        record.prim = batch.prims[0];
        record.drawRect = batch.drawRect;
        record.reason = batch.reason;
        record.clut = toneClut_;
        record.clutValid = toneClutValid_;
        tonePage_.records.push_back(record);
        tonePage_.absoluteCbp[tonePage_.strips] = cbp;
        ++tonePage_.strips;
        counters.toneStripBatchesDeferred++;
        return true;
    }

    // The batches of the collected page through the normal path, in order (exact: from `first` on the page did not
    // decode). A strip's palette is only correct in the state its own TEX0 write left, so deferred batches are never
    // replayed after the fact: a page that stops decoding runs its remaining strips here, in order.
    void ReplayToneStrips(uint32_t first = 0) {
        for (uint32_t index = first; index < tonePage_.records.size(); ++index) {
            const ToneStripRecord& record = tonePage_.records[index];
            // The palette this strip must use: the load performed before it. Later strips of the page have already
            // loaded their own palettes into the GPU CLUT, so it is put back (a no-op for the first one).
            if (record.clutValid) {
                OnClutLoaded(record.clut);
                if (failed_) break;
            }
            Batch batch;
            batch.state = record.state;
            batch.vertices.assign(record.vertices, record.vertices + 2);
            batch.prims.push_back(record.prim);
            batch.drawRect = record.drawRect;
            batch.reason = record.reason;
            batch.trace = ComputeVertexTrace(batch, nullptr);
            TrackToneRun(batch);
            RunBatch(batch);
            if (failed_) break;
        }
        tonePage_.records.clear();
    }

    // Writes the collected page: one dispatch, or the batches themselves when the page does not decode as 09.4's
    // strip sequence. Every source is made current in the GLM first; the page becomes GLM-owned afterwards.
    void FlushToneStrips() {
        if (!tonePage_.active) {
            RecordTonePendingClut();  // IN-383: a held strip palette is put back before anything else reads the CLUT
            return;
        }
        tonePage_.active = false;
        RecordTonePendingClut();  // IN-383
        const uint32_t strips = tonePage_.strips;
        if (strips == 0) return;
        if (failed_) {
            tonePage_.records.clear();
            return;
        }
        // Quadrant of every strip from its CBP: the temporary page's 16x16 quadrants start at blocks
        // {0, 4, 16, 20, 8, 12, 24, 28} (01.3 block table, 07.4), quadrant id = qx + 4 * qy.
        const uint32_t base = tonePage_.absoluteCbp[0];
        uint32_t quadrants[8] = {};
        // Leading strips whose palette base decodes. TryDeferToneStrip only collects a strip whose CBP is its own
        // quadrant of the base, so the whole collected prefix decodes; the check stays as the refusal for a base
        // that is not a page base.
        uint32_t drawable = 0;
        if (base % kBlocksPerPage == 0) {
            for (uint32_t strip = 0; strip < strips && strip < 8; ++strip) {
                const uint32_t delta = tonePage_.absoluteCbp[strip] - base;
                int quadrant = -1;
                for (uint32_t candidate = 0; candidate < 8; ++candidate) {
                    if (delta == kToneQuadrantBlock[candidate]) {
                        quadrant = static_cast<int>(candidate);
                        break;
                    }
                }
                if (quadrant < 0) break;
                quadrants[strip] = static_cast<uint32_t>(quadrant);
                ++drawable;
            }
        }
        const uint32_t rest = strips - drawable;
        const bool decodes = drawable != 0;
        const int32_t width = static_cast<int32_t>(8u * drawable);
        const memory::BufferView source{base, 1, memory::Psm::Ct32};  // the temporary page's first page holds every quadrant
        const memory::BufferView index{tonePage_.indexBp, tonePage_.indexBw, memory::Psm::T8};
        // The destination rectangle is one memory page (a page-aligned 64 x 32 rectangle). The spans of SpanOfRect are
        // the coherence granularity (whole page rows), so the page is named directly here.
        const memory::PsmInfo& destinationInfo = memory::psmInfo(memory::Psm::Ct24);
        const uint32_t bufferWidth =
            std::max<uint32_t>(tonePage_.destination.bufferWidth, 1u) * 64u;
        const uint32_t pagesPerRow =
            std::max<uint32_t>(1u, (bufferWidth + destinationInfo.pageWidth - 1u) / destinationInfo.pageWidth);
        const uint32_t pageIndex = (static_cast<uint32_t>(tonePage_.y0) / destinationInfo.pageHeight) * pagesPerRow +
                                   (static_cast<uint32_t>(tonePage_.x0) / destinationInfo.pageWidth);
        const BlockSpan span{(tonePage_.destination.basePointer + pageIndex * kBlocksPerPage) % kBlocksTotal,
                             kBlocksPerPage};
        const BlockSpan sourceSpan{base % kBlocksTotal, kBlocksPerPage};
        static const bool trace = std::getenv("COREGS_HW_TONE_TRACE") != nullptr;  // diagnostic only
        if (trace) {
            std::fprintf(stderr, "[tone] flush strips=%u base=%u decodes=%d span=%u+%u source=%u+%u\n", strips, base,
                         decodes ? 1 : 0, span.begin, span.count, sourceSpan.begin, sourceSpan.count);
        }
        if (!decodes || SpansOverlap(span, sourceSpan)) {
            ReplayToneStrips();
            return;
        }
        // Both sources must be current in the GLM: stale pages uploaded, drawn targets swizzled.
        if (!PrepareGlm(source, 64, 32, true) || failed_) {
            ReplayToneStrips();
            return;
        }
        if (!PrepareGlm(index, 8, 32, true) || failed_) {
            ReplayToneStrips();
            return;
        }
        if (verifyShuffle_) {
            if (rest == 0) VerifyToneStrips(span, quadrants, base, drawable);
            else ReplayToneStrips();
            return;
        }
        // The page's own slots park in the GLM (the dispatch writes it there) and the page is uploaded.
        CoherentForWrite(span, nullptr, true);
        if (!UploadStalePages(span)) return;
        if (failed_) return;
        gpu::GpuDrawer::ToneStripPage page;
        page.destBp = tonePage_.destination.basePointer;
        page.destBw = tonePage_.destination.bufferWidth;
        page.destX = tonePage_.x0;
        page.destY = tonePage_.y0;
        page.sourceBp = base;
        page.sourceBw = 1;
        page.indexBp = tonePage_.indexBp;
        page.indexBw = tonePage_.indexBw;
        page.stripCount = drawable;
        std::memcpy(page.quadrants, quadrants, sizeof quadrants);
        std::string error;
        if (!gpu_.ToneStrips(page, error)) {
            Fail(error);
            return;
        }
        // IN-398: the race colour filter's strips rewrite the frame's pages, but its scaled result is rebuilt by
        // the tone carry at the end of the run (IN-379), so they do not retire a carry source.
        ForEachPage(span, [&](uint32_t pageIndex) {
            if (!glmOwned_[pageIndex]) {
                glmOwned_[pageIndex] = true;
                ++glmOwnedPages_;
            }
        });
        counters.toneStripPages++;
        counters.toneStripBatches += drawable;
        // The strips the template did not explain run afterwards, in order, through the normal path.
        if (rest != 0) ReplayToneStrips(drawable);
        else tonePage_.records.clear();
    }

    // COREGS_HW_VERIFY_SHUFFLE (09.4): writes the page with the strip dispatch, keeps the result, then writes it again
    // with the batches themselves and counts the bytes that differ. The page's memory and resident state afterwards
    // are the normal path's, so a verification run stays exact.
    void VerifyToneStrips(const BlockSpan& span, const uint32_t* quadrants, uint32_t base, uint32_t strips) {
        const uint32_t page = span.begin / kBlocksPerPage;
        const uint32_t offset = page * 8192u;
        Materialise(span);
        CoherentForWrite(span);
        if (failed_) return;
        uint8_t* const local = memory_.Local().bytes().data();
        const std::vector<uint8_t> before(local + offset, local + offset + 8192u);
        gpu::GpuDrawer::ToneStripPage pageInfo;
        pageInfo.destBp = tonePage_.destination.basePointer;
        pageInfo.destBw = tonePage_.destination.bufferWidth;
        pageInfo.destX = tonePage_.x0;
        pageInfo.destY = tonePage_.y0;
        pageInfo.sourceBp = base;
        pageInfo.sourceBw = 1;
        pageInfo.indexBp = tonePage_.indexBp;
        pageInfo.indexBw = tonePage_.indexBw;
        pageInfo.stripCount = strips;
        std::memcpy(pageInfo.quadrants, quadrants, sizeof pageInfo.quadrants);
        std::string error;
        if (!UploadStalePages(span) || !gpu_.ToneStrips(pageInfo, error)) {
            Fail(error.empty() ? std::string("tone strip verification failed") : error);
            return;
        }
        // IN-398: as above (the colour filter, rebuilt by the tone carry).
        if (!glmOwned_[page]) {
            glmOwned_[page] = true;
            ++glmOwnedPages_;
        }
        std::vector<uint8_t> hle(8192u);
        if (!gpu_.DownloadGlm({{offset, 8192u}}, local, error)) {
            Fail(error);
            return;
        }
        std::memcpy(hle.data(), local + offset, 8192u);
        // The normal path again, from the same starting page.
        std::memcpy(local + offset, before.data(), 8192u);
        MarkWritten(span);
        glmOwned_[page] = false;
        --glmOwnedPages_;
        glmPageSequence_[page] = 0;
        ReplayToneStrips();
        if (failed_) return;
        CoherentForWrite(span);  // a resident target of the normal path writes the page back
        uint64_t differing = 0;
        const uint8_t* const now = memory_.Local().bytes().data() + offset;
        for (uint32_t byte = 0; byte < 8192u; ++byte) differing += now[byte] != hle[byte] ? 1u : 0u;
        counters.toneVerifyPages++;
        counters.toneVerifyMismatchBytes += differing;
        tonePage_.records.clear();
    }

    void RunBatch(Batch& batch) {
        TrackToneRun(batch);  // IN-379 (scale > 1 only)
        Translation t;
        // IN-360: reuse the previous batch's vector storage (cleared, capacity kept).
        t.vertices.swap(spareVertices_);
        t.draws.swap(spareDraws_);
        t.texels.swap(spareTexels_);
        t.vertices.clear();
        t.draws.clear();
        t.texels.clear();
        struct ReturnStorage {
            Translation& t;
            Impl& s;
            ~ReturnStorage() {
                s.spareVertices_.swap(t.vertices);
                s.spareDraws_.swap(t.draws);
                s.spareTexels_.swap(t.texels);
            }
        } returnStorage{t, *this};
        {
            StageTimer timer(frameTimings_.translateUs);
            Translate(batch, memory_.Local(), frontEnd_.Clut(), t, counters,
                      [this](const memory::BufferView& view, uint32_t width, uint32_t height) {
                          {
                              ReasonScope scope(writebackReason_, WritebackReason::Texture);
                              CoherentForRead(SpanOfRect(view, 0, 0, static_cast<int32_t>(width) - 1,
                                                     static_cast<int32_t>(height) - 1));
                          }
                      },
                      this,
                      [this](const memory::BufferView& view, uint32_t width, uint32_t height,
                             const PixelRect& footprint) -> int32_t {
                          return TextureSlotFor(view, width, height, footprint);
                      },
                      [this](const memory::BufferView& view, uint32_t width, uint32_t height, bool indexed,
                             bool& gpuClut) {
                          gpuClut = GpuClutActive();
                          return PrepareGlm(view, width, height, indexed && gpuClut);
                      });
        }
        if (observer_) observer_(batch, t);
        struct ClearProtection {
            const ResidentSlot*& slot;
            ~ClearProtection() { slot = nullptr; }
        } clearProtection{protectedSlot_};
        if (t.skippedTexture) {
            counters.primitivesSkippedTexture += batch.prims.size();
            counters.batchesSkippedTexture++;
            return;
        }
        if (t.blended) counters.batchesBlended++;
        if (t.draws.empty()) {
            counters.batchesEmpty++;
            return;
        }
        std::string error;
        {
            StageTimer timer(frameTimings_.executeUs);
            bool done = false;
            if (cacheMode_ && cpuSmallSprites_ && SmallUntexturedSprites(batch, t)) {
                // IN-348: small flat sprite batches (GT4 draws its palettes this way, chapter 07.8) are drawn
                // by the CPU reference straight into the shadow memory, which is exact for untextured sprites
                // (the GPU path draws the same words). A CLUT load from them then needs no GPU write-back.
                ReasonScope scope(writebackReason_, WritebackReason::CpuSprite);
                CoherentForWrite(SpanOfRect(t.frameView, t.rect));
                CoherentForWrite(SpanOfRect(t.zView, t.rect));
                cpuref::DrawStats stats;
                cpuref::DrawBatch(batch, memory_.Local(), frontEnd_.Clut(), stats);
                MarkWritten(SpanOfRect(t.frameView, t.rect));
                MarkWritten(SpanOfRect(t.zView, t.rect));
                counters.cpuSpriteBatches++;
                done = true;
            }
            if (!done && cacheMode_) {
                done = ExecuteResident(t, error);
                if (done && !error.empty()) Fail(error);
            }
            if (!done) {
                if (cacheMode_) {
                    // Write-back path for this batch (scratch GPU slots): no resident target may hold its memory.
                    ReasonScope scope(writebackReason_, WritebackReason::Fallback);
                    CoherentForWrite(SpanOfRect(t.frameView, t.rect));
                    CoherentForWrite(SpanOfRect(t.zView, t.rect));
                    counters.cacheFallbackBatches++;
                }
                if (!ExecuteBatch(gpu_, t, memory_.Local(), counters, error)) Fail(error);
                MarkWritten(SpanOfRect(t.frameView, t.rect));
                MarkWritten(SpanOfRect(t.zView, t.rect));
            }
        }
    }

    void OnBatch(Batch&& batch) override {
        StageTimer batchTimer(counters.batchUs);
        counters.batches++;
        if (!gpu_.Ready()) {
            Fail("GPU drawer is not initialised");
            return;
        }
        // IN-383: a strip of GT4's tone sequence is collected; every other batch writes the collected page first.
        if (TryDeferToneStrip(batch)) {
            if (observer_) observer_(batch, BatchTranslation{});  // the trace still lists every batch
            if (tonePage_.strips == 8) FlushToneStrips();
            return;
        }
        FlushToneStrips();
        RunBatch(batch);
    }


    bool OnSync(SyncKind, uint32_t, uint32_t) override { return false; }

    void Fail(const std::string& message) {
        failed_ = true;
        if (lastError_.empty()) lastError_ = message;
    }

    cpuref::LocalMemorySink memory_;
    FrontEnd frontEnd_;
    gpu::GpuDrawer gpu_;
    Counters counters;
    PassTimings timings_;        // last completed frame
    PassTimings frameTimings_;   // being accumulated
    PassTimings timingTotals_;   // session sums
    std::uint64_t timedFrames_ = 0;
    std::vector<std::uint8_t> auxSnapshot_;  // guest SLM at BeginAuxiliaryFrame()
    bool auxiliary_ = false;
    display::DisplayRegisters displayRegs_;
    std::array<uint32_t, 2> previousDby_{};
    std::array<uint8_t, 3> background_{};
    std::optional<display::Image> lastImage_;      // last reference image (KeepPrevious, 13.3)
    std::optional<display::Image> lastPresented_;  // last presentation image (KeepPrevious, 13.3)
    display::Deinterlacer deinterlacer_;           // 13.4 history (presentation only)
    OutputSettings settings_;
    bool oddField_ = false;                        // field of the VBlank being closed (12.4)
    std::uint64_t vblanks_ = 0;                    // Flush() count: one per VBlank
    std::uint64_t layoutVblank_ = 0;               // VBlank whose layout is cached in layout_
    bool layoutValid_ = false;
    display::DisplayLayout layout_;

    // ---- Asynchronous presentation source (IN-355) ----
    // With a window, each Flush records a download of the displayed memory (made current in the GLM) into the
    // frame's own submission; the presentation compose of the next VBlank composes that capture once the GPU has
    // finished the frame (one frame of presentation latency), so the CPU never waits for the frame it just
    // submitted. Reference composes (dumps) stay synchronous and exact for their own VBlank.
    struct DisplayCapture {
        bool valid = false;
        display::DisplayLayout layout;
        display::DisplayRegisters regs;
        std::array<uint8_t, 3> background{};
        bool oddField = false;
        gpu::GpuDrawer::DownloadTicket ticket;
        bool completed = false;
    };
    bool asyncPresentation_ = false;
    bool gpuCompose_ = true;  // IN-364 (off for A/B runs)

    // IN-364: the GPU compose applies to this VBlank's layout (progressive, a composite plan, GLM mode).
    bool GpuComposeApplies(const display::DisplayLayout& layout, display::MergePlan& plan,
                           std::array<bool, 2>& available) const {
        if (!gpuCompose_ || !cacheMode_ || !glmEnabled_ || auxiliary_ || !layout.valid) return false;
        // IN-381: an interlaced frame is composed (and deinterlaced) on the GPU only at an internal scale above 1;
        // at 1x the CPU presentation compose stays the reference path.
        if (layout.really_interlaced && gpu_.Scale() <= 1) return false;
        for (std::size_t i = 0; i < 2; ++i) {
            const display::CircuitRect& circuit = layout.circuits[i];
            available[i] = circuit.enabled && circuit.read_width != 0 && circuit.read_height != 0;
        }
        const display::PmodeFields pmode = display::decode_pmode(displayRegs_.pmode);
        plan = display::plan_merge(pmode, layout.circuits[0], layout.circuits[1], available[0], available[1]);
        return plan.outcome == display::MergeOutcome::Composite;
    }
    // IN-381: history of the scaled GPU deinterlace (the GPU form of display::Deinterlacer's state).
    struct GpuDeinterlaceHistory {
        uint32_t width = 0, height = 0;
        std::array<bool, gpu::kGpuDeinterlaceHistory> valid{};
        std::array<int, gpu::kGpuDeinterlaceHistory> parity{};
        std::array<uint64_t, gpu::kGpuDeinterlaceHistory> age{};
        uint64_t clock = 0;
        bool woven = false;
        bool output = false;
        void Clear() {
            valid.fill(false);
            woven = false;
            output = false;
        }
    };
    GpuDeinterlaceHistory gpuDeinterlace_;
    // A second GPU compose of a deinterlaced frame in the same VBlank (a dump next to the present) returns the first
    // one's frame: the deinterlace history must advance once per VBlank.
    uint64_t gpuComposeVblank_ = UINT64_MAX;
    GpuFrame lastGpuFrame_{};
    bool lastGpuFrameDeinterlaced_ = false;
    DisplayCapture captureNext_;   // recorded at the last Flush
    DisplayCapture captureShown_;  // recorded at the Flush before; composed by this VBlank's presentation
    memory::LocalMemory displayMemory_;
    bool failed_ = false;
    std::string lastError_;
    Renderer::BatchObserver observer_;
    bool cacheMode_ = false;       // COREGS_HW_SYNC_MODE: false = write-back reference mode
    bool fxaa_ = false;            // 13.5 FXAA on GPU-composed presentation frames (IN-376)
    // Decoded-texture cache (IN-341). On in both sync modes: it only skips re-decoding unchanged memory.
    static constexpr size_t kTextureCacheEntries = 256;
    bool textureCacheEnabled_ = true;
    std::array<uint64_t, memory::kPagesTotal> pageWrites_{};
    std::array<uint64_t, memory::kPagesTotal> memoryVersion_{};  // IN-398: every GS memory write (BumpMemory)
    uint64_t memoryClock_ = 0;
    std::array<const char*, memory::kPagesTotal> memoryWriter_{};  // diagnostics: the last writer of each page
    // IN-377 display shadows.
    struct DisplayShadow {
        bool valid = false;
        memory::BufferView view{};
        PixelRect saved{};
        uint64_t savedAt = 0;  // save order (the oldest is replaced)
    };
    std::array<DisplayShadow, gpu::kGpuDisplayShadows> shadows_{};
    uint64_t shadowSaves_ = 0;
    std::array<memory::BufferView, 4> recentDisplayViews_{};
    uint32_t recentDisplayNext_ = 0;
    uint64_t writeSequence_ = 0;
    uint64_t textureUses_ = 0;
    std::unordered_map<uint64_t, CachedTexture> textures_;
    std::vector<ResidentSlot> slots_;  // resident colour and depth targets (IN-344)
    WritebackReason writebackReason_ = WritebackReason::Other;  // why the next write-backs happen (counters)
    const ResidentSlot* protectedSlot_ = nullptr;  // texture source of the current batch (IN-346)
    bool textureFromTargets_ = true;               // IN-346 (off for A/B runs)
    bool cpuSmallSprites_ = true;                  // IN-348 (off for A/B runs)
    bool glmEnabled_ = true;                       // IN-347 COREGS_HW_GLM (off for A/B runs: write-back path)
    bool gpuClutEnabled_ = true;                   // IN-349 (off for A/B runs)
    bool glmParking_ = true;                       // IN-350 (off for A/B runs: evictions and overlaps write back)
    bool glmLoadsAlways_ = false;                  // IN-350 fixtures: every slot load goes through the GLM
    BlockSpan pendingReadSpan_{0, kAllBlocks};    // source of the last LOCAL->HOST transfer (10.2)
    bool glmBatchedWritebacks_ = true;
    std::vector<std::pair<uint32_t, uint32_t>> staleRanges_;
    std::vector<Qword> gifScratch_;              // IN-360: unaligned GIF packets copied here (reused)
    std::vector<gpu::GpuVertex> spareVertices_;  // IN-360: storage reused by the next batch's translation
    std::vector<gpu::GpuDraw> spareDraws_;
    std::vector<uint32_t> spareTexels_;             // IN-357 (off for A/B runs: one readback per slot)
    bool gpuUploads_ = true;                       // IN-351 (off for A/B runs: uploads written on the CPU)
    std::array<bool, memory::kPagesTotal> glmOwned_{};  // newest content only in the GLM (IN-350)
    uint32_t glmOwnedPages_ = 0;
    std::array<uint64_t, memory::kPagesTotal> glmPageUploadClock_{};  // glmUploadClock_ at the page's last upload
    uint64_t glmUploadClock_ = 0;
    bool gpuClutSynced_ = false;                   // the GPU CLUT was initialised from the CPU CLUT
    std::array<uint64_t, memory::kPagesTotal> glmPageSequence_{};
    uint64_t slotUses_ = 0;
};

Renderer::Renderer(bool prefer_core2) : impl_(std::make_unique<Impl>()) {
    // COREGS_HW_CORE=1 selects the reference write-back core, =2 the CoreGS 2.0 core; unset uses the
    // caller's choice (the public renderer asks for CoreGS 2.0).
    const char* core = std::getenv("COREGS_HW_CORE");
    const bool use_core2 = core ? std::strcmp(core, "2") == 0 : prefer_core2;
    if (use_core2) {
        core2_ = std::make_unique<core2::Renderer>();
        std::fprintf(stderr, "[hw] CoreGS 2.0 core (one image per target at scale)\n");
    }
}

Renderer::~Renderer() = default;

void Renderer::SetProfile(const coregs::GameProfile &profile) {
  if (core2_) core2_->SetProfile(profile);
}

void Renderer::SetScale(std::uint32_t scale, float scaleOffset) {
    if (core2_) return core2_->SetScale(scale, scaleOffset);
    impl_->gpu_.SetScale(scale, scaleOffset);
    impl_->frontEnd_.SetCullUncovered(scale <= 1);  // IN-403
}

std::uint32_t Renderer::Scale() const { return core2_ ? core2_->Scale() : impl_->gpu_.Scale(); }

bool Renderer::Init(const gpu::VulkanHandles& handles, std::string& error) {
    if (core2_) return core2_->Init(handles, error);
    if (!impl_->gpu_.Init(handles, error)) return false;
    impl_->UpdateClutCopies();
    return true;
}

bool Renderer::GifTransfer(std::uint32_t path, const std::uint8_t* bytes, std::size_t byteCount) {
    if (core2_) return core2_->GifTransfer(path, bytes, byteCount);
    if (path == 0 || path > 3) return false;  // IN-312
    if (byteCount % 16 != 0) return false;
    StageTimer timer(impl_->counters.gifUs);
    // IN-360: on a little-endian host the stream bytes are Qwords already; they are used in place when aligned,
    // otherwise copied once into a reused buffer (no allocation per packet).
    const std::size_t count = byteCount / 16;
    if constexpr (std::endian::native == std::endian::little && sizeof(Qword) == 16) {
        if (reinterpret_cast<std::uintptr_t>(bytes) % alignof(Qword) == 0) {
            impl_->frontEnd_.GifTransfer(path, reinterpret_cast<const Qword*>(bytes), count);
            return true;
        }
        impl_->gifScratch_.resize(count);
        std::memcpy(impl_->gifScratch_.data(), bytes, byteCount);
    } else {
        impl_->gifScratch_.resize(count);
        for (std::size_t i = 0; i < count; ++i) {
            std::memcpy(&impl_->gifScratch_[i].lo, bytes + 16 * i, 8);  // as before IN-360
            std::memcpy(&impl_->gifScratch_[i].hi, bytes + 16 * i + 8, 8);
        }
    }
    impl_->frontEnd_.GifTransfer(path, impl_->gifScratch_.data(), count);
    return true;
}

bool Renderer::GifBusy(std::uint32_t path) const {
    return core2_ ? core2_->GifBusy(path) : impl_->frontEnd_.GifBusy(path);
}

void Renderer::GifReset(std::uint32_t pathMask) {
    if (core2_) return core2_->GifReset(pathMask);
    impl_->frontEnd_.GifReset(pathMask);
    impl_->memory_.ResetRead();  // a GIF reset cancels the pending LOCAL->HOST read (10.2)
}

bool Renderer::WritePrivileged(std::uint32_t address, std::uint64_t value) {
    if (core2_) return core2_->WritePrivileged(address, value);
    using namespace priv;
    auto& regs = impl_->displayRegs_;
    switch (address) {
        case kPmode: regs.pmode = value; return true;
        case kSmode1: regs.smode1 = value; return true;
        case kSmode2: regs.smode2 = value; return true;
        case kSyncv: regs.syncv = value; return true;
        case kDispfb1: regs.circuits[0].dispfb = value; return true;
        case kDisplay1: regs.circuits[0].display = value; return true;
        case kDispfb2: regs.circuits[1].dispfb = value; return true;
        case kDisplay2: regs.circuits[1].display = value; return true;
        case kBgcolor:
            impl_->background_[0] = static_cast<uint8_t>(Field(value, 0, 8));
            impl_->background_[1] = static_cast<uint8_t>(Field(value, 8, 8));
            impl_->background_[2] = static_cast<uint8_t>(Field(value, 16, 8));
            return true;
        case kCsr:
            // CSR.RESET (bit 9) resets the GS as in the CPU reference; CSR.FIELD is read-only, the
            // VSYNC record supplies the display field.
            if (Field(value, 9, 1) != 0) ResetGs();
            return true;
        default:
            return false;
    }
}

void Renderer::ResetGs() {
    if (core2_) return core2_->ResetGs();
    Impl& s = *impl_;
    s.WriteBackAll();  // a GS reset keeps local memory, so resident draws are stored first
    s.DropAll();
    s.frontEnd_.ResetGs();
    s.gpuClutSynced_ = false;  // the GPU CLUT is re-initialised from the CPU CLUT on its next use (IN-349)
    s.frontEnd_.GifReset(7u);
    s.memory_.ResetRead();
    s.displayRegs_ = {};
    s.displayRegs_.anti_blur = s.settings_.antiBlur;
    s.previousDby_ = {};
    s.background_ = {};
    s.lastImage_.reset();
    s.lastPresented_.reset();
    s.deinterlacer_.clear_history();
    s.layoutValid_ = false;
}

bool Renderer::BeginAuxiliaryFrame() {
    if (core2_) return false;  // CORE2: no auxiliary frames yet
    Impl& s = *impl_;
    if (s.auxiliary_) return false;
    s.WriteBackAll();  // the snapshot is the guest's memory including resident draws
    const auto bytes = s.memory_.Local().bytes();
    s.auxSnapshot_.assign(bytes.begin(), bytes.end());
    s.auxiliary_ = true;
    return true;
}

void Renderer::EndAuxiliaryFrame() {
    Impl& s = *impl_;
    if (!s.auxiliary_) return;
    s.DropAll();  // resident draws of the auxiliary frame are discarded with it
    s.DiscardGlmOwned();
    const auto bytes = s.memory_.Local().bytes();
    std::copy(s.auxSnapshot_.begin(), s.auxSnapshot_.end(), bytes.begin());
    s.MarkWritten(BlockSpan{0, kAllBlocks});
    s.auxSnapshot_.clear();
    s.auxSnapshot_.shrink_to_fit();
    s.memory_.ResetRead();  // no partial LOCAL->HOST state survives the auxiliary frame
    s.auxiliary_ = false;
    s.counters.auxiliaryFrames++;
}

bool Renderer::AuxiliaryActive() const { return impl_->auxiliary_; }

const PassTimings& Renderer::Timings() const { return impl_->timings_; }

const PassTimings& Renderer::TimingTotals() const { return impl_->timingTotals_; }

std::uint64_t Renderer::TimedFrames() const { return impl_->timedFrames_; }

void Renderer::ClearVram() {
    if (core2_) return core2_->ClearVram();
    impl_->DropAll();
    impl_->DiscardGlmOwned();
    auto bytes = impl_->memory_.Local().bytes();
    std::fill(bytes.begin(), bytes.end(), std::uint8_t{0});
    impl_->MarkWritten(BlockSpan{0, kAllBlocks});
}

std::size_t Renderer::ReadFifo(void* destination, std::size_t byteCount) {
    if (core2_) return core2_->ReadFifo(destination, byteCount);
    Impl& s = *impl_;
    if (s.auxiliary_) {
        // A guest readback is never answered from an auxiliary frame (roadmap requirement 2).
        s.counters.auxiliaryReadRefused++;
        return 0;
    }
    if (byteCount == 0) return 0;
    // The first FIFO access observes every earlier draw and upload (10.2).
    s.frontEnd_.FlushBatch(FlushReason::Transfer);
    if (s.memory_.LocalToHostPending()) {
        // IN-357: the first FIFO access takes the transfer's data from the shadow memory; only its source
        // rectangle must be current there (draws since TRXDIR included). It used to write back every resident
        // target on every FIFO read (GT4's lens flare: about 7 per frame, each a submission and a wait).
        ReasonScope scope(s.writebackReason_, WritebackReason::Transfer);
        s.CoherentForRead(s.pendingReadSpan_);
    }
    bool overrun = false;
    const std::vector<uint8_t> data = s.memory_.ReadLocalToHost(byteCount, overrun);
    if (overrun) s.counters.fifoOverruns++;
    if (s.memory_.ReadDrained()) s.frontEnd_.WriteRegister(reg::kTrxdir, 3);
    std::memcpy(destination, data.data(), data.size());
    return data.size();
}

void Renderer::Flush() {
    // Diagnostics (CORE2 bring-up): COREGS_HW_SLM_DUMP=<vblank>:<file> writes GS memory, with every drawn target written
    // back, at that VBlank, so the two cores' memory can be compared in the game.
    {
        static long long dumpAt = -2;
        static std::string dumpPath;
        if (dumpAt == -2) {
            dumpAt = -1;
            if (const char* text = std::getenv("COREGS_HW_SLM_DUMP")) {
                const char* colon = std::strchr(text, ':');
                if (colon) {
                    dumpAt = std::atoll(text);
                    dumpPath = colon + 1;
                }
            }
        }
        static long long flushes = 0;
        if (dumpAt >= 0 && flushes++ == dumpAt) {
            const std::uint8_t* bytes = Slm();
            if (std::FILE* file = std::fopen(dumpPath.c_str(), "wb")) {
                std::fwrite(bytes, 1, memory::kLocalMemoryBytes, file);
                std::fclose(file);
                std::fprintf(stderr, "[hw] GS memory written to %s at VBlank %lld\n", dumpPath.c_str(), dumpAt);
            }
        }
    }
    if (core2_) return core2_->Flush();
    Impl& s = *impl_;
    s.frontEnd_.VBlank(&kEverything, 1);
    s.frontEnd_.FlushBatch(FlushReason::VBlank);
    s.FlushToneStrips();  // IN-383: a VBlank ends a collected tone page
    s.FinishToneRun();  // IN-379: a VBlank ends a tone sequence
    // Frame end: publish the CPU cost of the passes that just ran (roadmap requirement 4), so a run
    // without a window still reports translate and execute.
    s.timings_ = s.frameTimings_;
    s.timingTotals_.translateUs += s.frameTimings_.translateUs;
    s.timingTotals_.executeUs += s.frameTimings_.executeUs;
    s.frameTimings_ = PassTimings{};
    ++s.timedFrames_;
    ++s.vblanks_;  // the next compose computes this VBlank's display layout (13.2)
    // Diagnostics: COREGS_HW_TRACE_FROM_VBLANK=N keeps the draw and scaled traces silent before VBlank N, so a trace of a
    // late scene (the GT Mode map at 9600) does not slow the whole run down.
    static const long traceFrom = [] {
        const char* text = std::getenv("COREGS_HW_TRACE_FROM_VBLANK");
        return text ? std::atol(text) : 0L;
    }();
    if (traceFrom > 0) s.gpu_.ArmTrace(static_cast<long>(s.vblanks_) >= traceFrom);
    if (s.displayRegs_.smode2 & 1u) {  // SMODE2.INT: which video mode the guest ran this VBlank
        ++s.counters.vblanksInterlaced;
    } else {
        ++s.counters.vblanksProgressive;
    }
    if (s.asyncPresentation_ && s.cacheMode_ && s.glmEnabled_ && s.gpu_.Ready() && !s.failed_ && !s.auxiliary_) {
        s.CaptureDisplay();  // IN-355: into this frame's submission, composed at the next VBlank
    }
    // Submit the frame's recorded chunks without waiting (IN-342): the GPU draws while the CPU composes and
    // starts the next frame; the next recording waits for it before reusing the rings.
    std::string error;
    if (s.gpu_.Ready() && !s.gpu_.Flush(false, error)) s.Fail(error);
}

FrameOutput Renderer::Compose(ComposeKind kind) {
    FrameOutput out;
    ComposeFrame(out, kind);
    return out;
}

void Renderer::SetAgeTargets(bool age) {
    if (core2_) core2_->SetAgeTargets(age);
}

void Renderer::SetDisplayField(bool odd) {
    if (core2_) core2_->SetDisplayField(odd);
    impl_->oddField_ = odd;
}

bool Renderer::DisplayField() const { return impl_->oddField_; }

void Renderer::SetOutputSettings(const OutputSettings& settings) {
    if (core2_) core2_->SetOutputSettings(settings);
    impl_->settings_ = settings;
    impl_->displayRegs_.anti_blur = settings.antiBlur;
}

const OutputSettings& Renderer::GetOutputSettings() const { return impl_->settings_; }

namespace {

// Row field of 13.3/13.4 (0 = the field on even output rows, unshifted) for a display field as the VSYNC
// record carries it (CSR.FIELD, 1 = odd). IN-339 (Confirmed on GT4 menus by measurement, against the IN-35
// reading): CSR.FIELD = 1 is the upper (even-row) field, CSR.FIELD = 0 the lower one.
int RowFieldOfDisplayField(bool csrFieldOdd) { return csrFieldOdd ? 0 : 1; }

void CopyImage(const display::Image& image, FrameOutput& out) {
    out.valid = image.width > 0 && image.height > 0;
    out.width = image.width;
    out.height = image.height;
    out.pixels.resize(image.pixels.size());
    uint32_t* dst = out.pixels.data();
    for (const display::Rgba8& p : image.pixels) {
        *dst++ = static_cast<uint32_t>(p.r) | (static_cast<uint32_t>(p.g) << 8) | (static_cast<uint32_t>(p.b) << 16) |
                 (static_cast<uint32_t>(p.a) << 24);
    }
}

// One circuit's source rectangle read from the SLM (13.2, 13.3). CT24 alpha is 0x80 (IN-313).
display::Image ReadCircuit(const memory::LocalMemory& local, const display::CircuitRect& circuit) {
    display::Image image(circuit.read_width, circuit.read_height);
    const auto psm = memory::decodePsm(circuit.psm);
    if (!psm) return image;
    const memory::BufferView view{memory::blockOfPage(circuit.fbp), circuit.fbw, *psm};
    const pixel::FrameFormat format = [&] {
        switch (circuit.psm) {
            case 0x01: return pixel::FrameFormat::Ct24;
            case 0x02:
            case 0x0A: return pixel::FrameFormat::Ct16;
            default: return pixel::FrameFormat::Ct32;
        }
    }();
    // Stored words of the read rectangle; readRect when it does not wrap at 2048 (IN-24), else per pixel.
    std::vector<uint32_t> words(static_cast<size_t>(circuit.read_width) * circuit.read_height);
    const bool wraps = circuit.dbx < 0 || circuit.dby < 0 ||
                       static_cast<uint32_t>(circuit.dbx) + circuit.read_width > 2048u ||
                       static_cast<uint32_t>(circuit.dby) + circuit.read_height > 2048u;
    if (!wraps) {
        local.readRect(view, static_cast<uint32_t>(circuit.dbx), static_cast<uint32_t>(circuit.dby), circuit.read_width,
                       circuit.read_height, words.data());
    }
    display::Rgba8* pixels = image.pixels.data();
    for (uint32_t y = 0; y < circuit.read_height; ++y) {
        for (uint32_t x = 0; x < circuit.read_width; ++x) {
            const uint32_t sx = static_cast<uint32_t>(circuit.dbx + static_cast<int32_t>(x)) & 2047u;
            const uint32_t sy = static_cast<uint32_t>(circuit.dby + static_cast<int32_t>(y)) & 2047u;
            const size_t index = static_cast<size_t>(y) * circuit.read_width + x;
            const uint32_t stored = wraps ? local.readPixel(view, sx, sy) : words[index];
            const pixel::Colour colour = pixel::frame_word_to_colour(pixel::unpack_frame_word(format, stored));
            display::Rgba8& dst = pixels[index];
            dst.r = static_cast<uint8_t>(colour.r);
            dst.g = static_cast<uint8_t>(colour.g);
            dst.b = static_cast<uint8_t>(colour.b);
            dst.a = format == pixel::FrameFormat::Ct24 ? 0x80 : static_cast<uint8_t>(colour.a);
        }
    }
    return image;
}

}  // namespace

void Renderer::ComposeFrame(FrameOutput& out, ComposeKind kind) {
    if (core2_) {
        (void)kind;
        return core2_->ComposeFrame(out);
    }
    Impl& s = *impl_;
    out.valid = false;
    out.width = 0;
    out.height = 0;
    double composeUs = 0.0;
    {
        StageTimer timer(composeUs);
        // 13.2 once per VBlank: compute_layout advances the DBY history, so a second compose of the
        // same VBlank (a dump next to a present) must reuse the layout instead of recomputing it.
        s.EnsureLayout();
        const bool presentation = kind == ComposeKind::Presentation;
        // IN-355: a presentation compose with asynchronous presentation composes the previous VBlank's capture
        // (its download is complete once the GPU finished that frame; nothing waits for the frame just submitted).
        Impl::DisplayCapture& capture = s.captureShown_;
        bool fromCapture = presentation && s.asyncPresentation_ && capture.valid;
        if (fromCapture && !capture.completed) {
            std::string error;
            if (!s.gpu_.CompleteGlmDownload(capture.ticket, s.displayMemory_.bytes().data(), error)) {
                s.Fail(error);
                fromCapture = false;
            } else {
                capture.completed = true;
            }
        }
        if (presentation && s.asyncPresentation_ && !fromCapture && s.captureNext_.valid) {
            // The first capture is recorded but not composable yet (the first VBlank of asynchronous
            // presentation): nothing to show this VBlank. The deinterlacer and the presentation history are not
            // touched, so the composed frames are the synchronous sequence one VBlank later. Without any capture
            // (presents without a Flush between them) the compose stays synchronous.
            s.timings_.composeUs = 0.0;
            return;
        }
        const display::DisplayLayout& layout = fromCapture ? capture.layout : s.layout_;
        const display::DisplayRegisters& regs = fromCapture ? capture.regs : s.displayRegs_;
        const std::array<uint8_t, 3>& background = fromCapture ? capture.background : s.background_;
        const bool displayField = fromCapture ? capture.oddField : s.oddField_;
        const memory::LocalMemory& source = fromCapture ? s.displayMemory_ : s.memory_.Local();
        if (fromCapture) s.counters.presentationsFromCapture++;
        const auto blank = [&] {
            if (presentation) {
                s.counters.blankFrames++;
                s.deinterlacer_.clear_history();  // 13.4: no source, no stale history next time
                s.gpuDeinterlace_.Clear();        // IN-381: the GPU form too
            }
        };
        const auto body = [&] {
            if (!layout.valid) {
                blank();
                return;
            }
            const display::PmodeFields pmode = display::decode_pmode(regs.pmode);
            std::array<display::Image, 2> sources{};
            std::array<bool, 2> available{false, false};
            for (std::size_t i = 0; i < 2; ++i) {
                const display::CircuitRect& circuit = layout.circuits[i];
                if (!circuit.enabled || circuit.read_width == 0 || circuit.read_height == 0) continue;
                if (!fromCapture) {
                    // The display reads this circuit's rectangle from the shadow memory: GPU-newer targets there
                    // are written back first (only those; a wrapping read rectangle reports all memory).
                    ReasonScope scope(s.writebackReason_, WritebackReason::Display);
                    s.CoherentForRead(Impl::DisplayCircuitSpan(circuit));
                }
                sources[i] = ReadCircuit(source, circuit);
                available[i] = true;
            }
            const display::MergePlan plan =
                display::plan_merge(pmode, layout.circuits[0], layout.circuits[1], available[0], available[1]);
            if (plan.outcome == display::MergeOutcome::KeepPrevious) {
                // 13.3: a circuit is enabled without a source; nothing new this VBlank, previous image kept.
                const std::optional<display::Image>& kept = presentation ? s.lastPresented_ : s.lastImage_;
                if (kept) CopyImage(*kept, out);
                blank();
                return;
            }
            // Effective deinterlace (13.4). The reference image never deinterlaces (it is the 1x decode).
            display::DeinterlaceMode effective = display::DeinterlaceMode::Off;
            if (presentation) {
                effective = s.settings_.deinterlace == display::DeinterlaceMode::Auto
                                ? display::resolve_auto_deinterlace(layout, layout.ffmd, regs.scanmsk_active)
                                : s.settings_.deinterlace;
            }
            // IN-339: the display field of a frame (VSYNC field = CSR.FIELD, 1 = odd) maps to the LOWER row
            // phase when it is 0 and to the upper one when it is 1: GT4 draws its CSR.FIELD = 1 frames half a
            // line higher, so with the IN-35 reading (field 1 shifted down) every woven menu line landed two
            // rows from its partner. `_bff` still inverts the result.
            const int vblankField = RowFieldOfDisplayField(displayField);
            const int fieldPhase = vblankField ^ (s.settings_.bottomFieldFirst ? 1 : 0);
            const int32_t offsetRows =
                presentation ? display::interlace_offset_rows(layout.really_interlaced, layout.ffmd, effective, 1,
                                                              fieldPhase)
                             : 0;
            const display::OutputSize size =
                display::merge_output_size(layout, effective == display::DeinterlaceMode::Off, 1);
            const display::SourcePlacement rc1{available[0] ? &sources[0] : nullptr, layout.circuits[0].screen_x,
                                               layout.circuits[0].screen_y,
                                               static_cast<int32_t>(layout.circuits[0].display_width),
                                               static_cast<int32_t>(layout.circuits[0].display_height)};
            const display::SourcePlacement rc2{available[1] ? &sources[1] : nullptr, layout.circuits[1].screen_x,
                                               layout.circuits[1].screen_y,
                                               static_cast<int32_t>(layout.circuits[1].display_width),
                                               static_cast<int32_t>(layout.circuits[1].display_height)};
            display::Image merged = display::compose_merge(plan, pmode, background[0], background[1],
                                                           background[2], size.width, size.height, 1, offsetRows,
                                                           rc1, rc2);
            if (presentation) {
                if (plan.use_rc1 && plan.use_rc2 && !pmode.mmod) s.counters.mergeAlphaFromTexel++;  // V-01-12
                if (offsetRows != 0) s.counters.interlaceOffsetFrames++;
                const bool anySource = available[0] || available[1];
                const bool runs = anySource && layout.really_interlaced && effective != display::DeinterlaceMode::Off;
                if (runs) {
                    merged = s.deinterlacer_.process(merged, vblankField, effective,
                                                     s.settings_.bottomFieldFirst, 1);
                    s.counters.deinterlacedFrames++;
                } else if (!anySource) {
                    s.deinterlacer_.clear_history();
                }
                s.counters.presentationFrames++;
            }
            CopyImage(merged, out);
            if (!s.auxiliary_) {
                // An auxiliary frame never replaces the guest's display history.
                if (presentation) {
                    s.lastPresented_ = std::move(merged);
                } else {
                    s.lastImage_ = std::move(merged);
                }
            }
        };
        body();
    }
    s.timings_.composeUs = composeUs;
    s.counters.composeUs += composeUs;
    s.timings_.executeGpuUs = 0.0;  // reserved for GPU timestamps (not measured yet)
}

void Renderer::SetBatchObserver(BatchObserver observer) { impl_->observer_ = std::move(observer); }

const std::uint8_t* Renderer::Slm() const {
    if (core2_) return core2_->Slm();
    impl_->WriteBackAll();  // a caller reading the shadow memory sees every draw (M7 coherence)
    return impl_->memory_.Local().bytes().data();
}

void Renderer::SetSyncMode(bool cache) {
    if (!cache) {
        impl_->WriteBackAll();
        impl_->DropAll();
    }
    impl_->cacheMode_ = cache;
    impl_->UpdateClutCopies();
}

bool Renderer::CacheMode() const { return impl_->cacheMode_; }

void Renderer::SetTextureFromTargets(bool enabled) { impl_->textureFromTargets_ = enabled; }

void Renderer::SetGlm(bool enabled) {
    impl_->glmEnabled_ = enabled;
    impl_->UpdateClutCopies();
}

void Renderer::SetGpuClut(bool enabled) {
    impl_->FlushToneStrips();  // IN-383: a held strip palette is recorded while the GPU CLUT is still authoritative
    impl_->gpuClutEnabled_ = enabled;
    impl_->UpdateClutCopies();
}

void Renderer::SetGlmParking(bool enabled) {
    if (!enabled) impl_->WriteBackAll();  // GLM-owned pages go back to the shadow memory
    impl_->glmParking_ = enabled;
}

void Renderer::SetGlmLoadsAlways(bool enabled) { impl_->glmLoadsAlways_ = enabled; }

void Renderer::SetGpuUploads(bool enabled) { impl_->gpuUploads_ = enabled; }

void Renderer::SetRuns(bool enabled) { impl_->gpu_.SetRuns(enabled); }

void Renderer::SetBatchedWritebacks(bool enabled) { impl_->glmBatchedWritebacks_ = enabled; }

void Renderer::SetGpuTiming(bool enabled) { impl_->gpu_.SetGpuTiming(enabled); }
void Renderer::SetDecodeReuse(bool enabled) { impl_->gpu_.SetDecodeReuse(enabled); }
void Renderer::SetClutDefer(bool enabled) { impl_->gpu_.SetClutDefer(enabled); }

void Renderer::SetToneHle(bool enabled) {
    // A pending page is written before the path changes, so the switch is exact at any point.
    impl_->FlushToneStrips();
    impl_->toneHle_ = enabled;
}

void Renderer::SetVerifyShuffle(bool enabled) {
    impl_->FlushToneStrips();
    impl_->verifyShuffle_ = enabled;
}

void Renderer::SetChunksPerSubmission(std::uint32_t chunks) { impl_->gpu_.SetChunksPerSubmission(chunks); }

void Renderer::SetDoubleBuffering(bool enabled) { impl_->gpu_.SetDoubleBuffering(enabled); }

bool Renderer::ComposeFrameGpu(GpuFrame& out) {
    if (core2_) return core2_->ComposeFrameGpu(out);
    Impl& s = *impl_;
    s.FlushToneStrips();  // IN-383: the compose reads local memory
    s.FinishToneRun();  // IN-379: the presentation reads the carried scaled image
    out = GpuFrame{};
    if (s.gpuComposeVblank_ == s.vblanks_ && s.lastGpuFrame_.valid && s.lastGpuFrameDeinterlaced_) {
        out = s.lastGpuFrame_;  // IN-381: already composed and deinterlaced this VBlank (the history moves once)
        return true;
    }
    double composeUs = 0.0;
    {
        StageTimer timer(composeUs);
        s.EnsureLayout();
        const display::DisplayLayout& layout = s.layout_;
        display::MergePlan plan;
        std::array<bool, 2> available{};
        if (!s.GpuComposeApplies(layout, plan, available)) {
            ++s.counters.gpuComposeSkips[0];
            return false;
        }
        const display::PmodeFields pmode = display::decode_pmode(s.displayRegs_.pmode);
        // The same choices as the CPU presentation compose (13.3, 13.4). A progressive frame has no field offset
        // and no deinterlace; an interlaced one (IN-381, scale > 1 only) gets the 13.3 step 3 offset of one native
        // row in the field's phase and the GPU deinterlace below, both counted in native rows.
        const display::DeinterlaceMode effective =
            s.settings_.deinterlace == display::DeinterlaceMode::Auto
                ? display::resolve_auto_deinterlace(layout, layout.ffmd, s.displayRegs_.scanmsk_active)
                : s.settings_.deinterlace;
        const display::OutputSize size =
            display::merge_output_size(layout, effective == display::DeinterlaceMode::Off, 1);
        if (size.width == 0 || size.height == 0) {
            ++s.counters.gpuComposeSkips[1];
            return false;
        }
        const bool interlaced = layout.really_interlaced;
        const int vblankField = RowFieldOfDisplayField(s.oddField_);
        const int fieldPhase = vblankField ^ (s.settings_.bottomFieldFirst ? 1 : 0);
        const int32_t offsetNativeRows =
            interlaced ? display::interlace_offset_rows(true, layout.ffmd, effective, 1, fieldPhase) : 0;
        const bool deinterlace = interlaced && effective != display::DeinterlaceMode::Off;
        {
            // IN-396 diagnostics: one line whenever the GPU compose's deinterlace decision or its inputs change.
            static int lastKey = -1;
            const int key = static_cast<int>(effective) | (layout.ffmd ? 16 : 0) | (interlaced ? 32 : 0) |
                            (layout.one_circuit_dby_changed ? 64 : 0) | (s.displayRegs_.scanmsk_active ? 128 : 0);
            if (key != lastKey) {
                lastKey = key;
                std::fprintf(stderr, "[hw] deinterlace vblank=%llu effective=%d ffmd=%d interlaced=%d dby_xor=%d scanmsk=%d\n",
                             static_cast<unsigned long long>(s.vblanks_), static_cast<int>(effective),
                             layout.ffmd ? 1 : 0, interlaced ? 1 : 0, layout.one_circuit_dby_changed ? 1 : 0,
                             s.displayRegs_.scanmsk_active ? 1 : 0);
            }
        }
        if (interlaced && (size.width > gpu::kGpuDeinterlaceWidth || size.height > gpu::kGpuDeinterlaceRows)) {
            ++s.counters.gpuComposeSkips[2];
            return false;  // larger than the deinterlace images: the CPU presentation at 1x
        }
        // The GLM is made current for the circuits (stale pages uploaded, drawn targets swizzled).
        for (std::size_t i = 0; i < 2; ++i) {
            if (!available[i]) continue;
            const BlockSpan span = Impl::DisplayCircuitSpan(layout.circuits[i]);
            if (!s.UploadStalePages(span)) return false;
            for (auto& slot : s.slots_) {
                if (slot.drawn.Empty() || !SpansOverlap(SpanOfRect(slot.view, slot.drawn), span)) continue;
                const int32_t bufferWidth = static_cast<int32_t>(std::max<uint32_t>(slot.view.bufferWidth, 1u) * 64u);
                if (slot.drawn.x1 >= bufferWidth) {
                    ReasonScope scope(s.writebackReason_, WritebackReason::Display);
                    s.WriteBack(slot);
                    if (!s.UploadStalePages(span)) return false;
                    continue;
                }
                if (!s.SwizzleSlotIntoGlm(slot)) return false;
            }
        }
        std::vector<uint32_t> parameters(29, 0u);
        parameters[0] = uint32_t{s.background_[0]} | (uint32_t{s.background_[1]} << 8) |
                        (uint32_t{s.background_[2]} << 16) | (std::min<uint32_t>(pmode.alp, 255u) << 24);
        parameters[1] = std::min<uint32_t>(pmode.alp, 255u);
        parameters[2] = pmode.mmod ? 1u : 0u;
        parameters[3] = plan.use_rc1 ? 1u : 0u;
        parameters[4] = plan.use_rc2 ? 1u : 0u;
        for (std::size_t i = 0; i < 2; ++i) {
            const display::CircuitRect& circuit = layout.circuits[i];
            const std::size_t at = i == 0 ? 5u : 17u;
            const auto psm = memory::decodePsm(circuit.psm);
            if (!available[i] || !psm) continue;
            parameters[at + 0] = memory::blockOfPage(circuit.fbp);
            parameters[at + 1] = circuit.fbw;
            parameters[at + 2] = circuit.psm;
            parameters[at + 3] = circuit.psm == 0x01u ? 1u : (circuit.psm == 0x02u || circuit.psm == 0x0Au) ? 2u : 0u;
            parameters[at + 4] = static_cast<uint32_t>(circuit.dbx);
            parameters[at + 5] = static_cast<uint32_t>(circuit.dby);
            parameters[at + 6] = circuit.read_width;
            parameters[at + 7] = circuit.read_height;
            parameters[at + 8] = static_cast<uint32_t>(circuit.screen_x);
            parameters[at + 9] = static_cast<uint32_t>(circuit.screen_y);
            parameters[at + 10] = circuit.display_width;
            parameters[at + 11] = circuit.display_height;
        }
        std::string error;
        const uint32_t scale = s.gpu_.Scale();
        uint32_t outWidth = size.width;
        uint32_t outHeight = size.height;
        std::array<uint32_t, 2> sources{};  // IN-387: the source word of each displayed circuit (diagnostics only)
        bool composed = false;
        if (scale > 1) {
            // IN-372: the same block with a 13th word per circuit: the resident colour slot whose scaled image holds
            // the circuit's whole read rectangle (+1), or 0 to read the GLM at 1x.
            std::vector<uint32_t> scaled(31, 0u);
            std::copy(parameters.begin(), parameters.begin() + 5, scaled.begin());
            for (std::size_t i = 0; i < 2; ++i) {
                const std::size_t from = i == 0 ? 5u : 17u;
                const std::size_t at = i == 0 ? 5u : 18u;
                std::copy(parameters.begin() + from, parameters.begin() + from + 12, scaled.begin() + at);
                // 13.3 step 3 at scale: the destination moves down by one native row in the odd phase.
                scaled[at + 9] = static_cast<uint32_t>(static_cast<int32_t>(scaled[at + 9]) + offsetNativeRows);
                const display::CircuitRect& circuit = layout.circuits[i];
                const auto psm = memory::decodePsm(circuit.psm);
                if (!available[i] || !psm) continue;
                const memory::BufferView view{memory::blockOfPage(circuit.fbp), circuit.fbw, *psm};
                s.NoteDisplayedView(view);
                const PixelRect read{circuit.dbx, circuit.dby, circuit.dbx + static_cast<int32_t>(circuit.read_width) - 1,
                                     circuit.dby + static_cast<int32_t>(circuit.read_height) - 1};
                uint32_t slotWord = 0;
                // IN-384: the slot's scaled image must cover the whole read rectangle (it only holds what the slot
                // loaded), not a fixed 1024 x 1024 square.
                if (read.x0 >= 0 && read.y0 >= 0) {
                    for (const auto& slot : s.slots_) {
                        if (slot.depth || !slot.hasIdentity || !SameView(slot.view, view) || !Contains(slot.loaded, read)) {
                            continue;
                        }
                        if (!s.gpu_.ScaledCovers(false, slot.gpuIndex, read.x1, read.y1)) continue;
                        slotWord = slot.gpuIndex + 1u;
                        break;
                    }
                }
                bool fromShadow = false;
                if (slotWord == 0) {
                    // IN-377: no resident slot holds the buffer; a display shadow may (checked per pixel on the GPU).
                    const int32_t shadow = s.DisplayShadowFor(view, read);
                    if (shadow >= 0) {
                        slotWord = gpu::kGpuColourSlots + static_cast<uint32_t>(shadow) + 1u;
                        fromShadow = true;
                    }
                }
                scaled[at + 12] = slotWord;
                sources[i] = slotWord;
                static const bool trace = std::getenv("COREGS_HW_SCALE_TRACE") != nullptr;  // diagnostic only
                if (trace) {
                    std::fprintf(stderr, "[scale] circuit %zu bp=%u bw=%u psm=%u read=%d,%d,%d,%d slot=%u\n", i,
                                 view.basePointer, view.bufferWidth, static_cast<unsigned>(view.psm), read.x0, read.y0,
                                 read.x1, read.y1, slotWord);
                    for (const auto& slot : s.slots_) {
                        std::fprintf(stderr, "[scale]   slot depth=%d gpu=%u id=%d bp=%u bw=%u psm=%u loaded=%d,%d,%d,%d "
                                     "drawn=%d,%d,%d,%d\n", slot.depth ? 1 : 0, slot.gpuIndex, slot.hasIdentity ? 1 : 0,
                                     slot.view.basePointer, slot.view.bufferWidth, static_cast<unsigned>(slot.view.psm),
                                     slot.loaded.x0, slot.loaded.y0, slot.loaded.x1, slot.loaded.y1, slot.drawn.x0,
                                     slot.drawn.y0, slot.drawn.x1, slot.drawn.y1);
                    }
                }
                if (fromShadow) {
                    s.counters.scaledCircuitsFromShadows++;
                } else if (slotWord != 0) {
                    s.counters.scaledCircuitsFromSlots++;
                } else {
                    s.counters.scaledCircuitsFromGlm++;
                }
            }
            outWidth = size.width * scale;
            outHeight = size.height * scale;
            composed = s.gpu_.ComposeDisplayScaled(scaled, outWidth, outHeight, error);
            if (composed) s.counters.scaledComposedFrames++;
        } else {
            composed = s.gpu_.ComposeDisplay(parameters, size.width, size.height, error);
        }
        if (composed && interlaced) {
            if (offsetNativeRows != 0) s.counters.interlaceOffsetFrames++;
            if (deinterlace) composed = s.DeinterlaceGpu(effective, vblankField, outWidth, outHeight, error);
        }
        const bool fxaa = composed && s.fxaa_;
        if (fxaa) composed = s.gpu_.ApplyFxaa(outWidth, outHeight, error);  // IN-376
        if (!composed || !s.gpu_.Flush(false, error)) {
            ++s.counters.gpuComposeSkips[3];
            s.Fail(error);
            return false;
        }
        if (plan.use_rc1 && plan.use_rc2 && !pmode.mmod) s.counters.mergeAlphaFromTexel++;  // V-01-12
        s.counters.presentationFrames++;
        s.counters.gpuComposedFrames++;
        if (scale > 1) s.DumpDisplayImages(sources, outWidth, outHeight, layout);  // IN-387, diagnostic only
        s.lastPresented_.reset();  // a later KeepPrevious falls back to the CPU history (none: blank)
        out.valid = true;
        out.width = outWidth;
        out.height = outHeight;
        out.fxaa = fxaa;
        out.image = fxaa ? s.gpu_.FxaaImage() : s.gpu_.DisplayImage();
        s.gpuComposeVblank_ = s.vblanks_;
        s.lastGpuFrame_ = out;
        s.lastGpuFrameDeinterlaced_ = deinterlace;
    }
    s.timings_.composeUs = composeUs;
    s.counters.composeUs += composeUs;
    return true;
}

void Renderer::SetGpuCompose(bool enabled) { impl_->gpuCompose_ = enabled; }

void Renderer::SetFxaa(bool enabled) {
    if (core2_) core2_->SetFxaa(enabled);
    impl_->fxaa_ = enabled;
}

void Renderer::SetPresentSize(std::uint32_t width, std::uint32_t height) {
    if (core2_) core2_->SetPresentSize(width, height);
}

void Renderer::SetForcedBilinear(std::uint32_t mode) {
    if (core2_) core2_->SetForcedBilinear(mode);
}

bool Renderer::ReadGpuFrame(const GpuFrame& frame, std::vector<std::uint32_t>& pixels) {
    if (core2_) return core2_->ReadGpuFrame(frame, pixels);
    std::string error;
    if (!frame.valid || !impl_->gpu_.ReadDisplay(frame.width, frame.height, pixels, error, frame.fxaa)) return false;
    return true;
}

void Renderer::SetAsyncPresentation(bool enabled) {
    impl_->asyncPresentation_ = enabled;
    impl_->captureNext_ = Impl::DisplayCapture{};
    impl_->captureShown_ = Impl::DisplayCapture{};
}

void Renderer::SetCpuSmallSprites(bool enabled) { impl_->cpuSmallSprites_ = enabled; }

const core2::Core2Counters* Renderer::Core2Stats() const { return core2_ ? &core2_->Core2Stats() : nullptr; }

bool Renderer::PrepareCore2Shaders(const std::function<void(std::size_t, std::size_t)>& progress, std::string& error) {
    if (!core2_) {
        error = "the shader cache belongs to CORE2 (COREGS_HW_CORE=2)";
        return false;
    }
    return core2_->PrepareShaders(progress, error);
}

const Counters& Renderer::GetCounters() const { return core2_ ? core2_->GetCounters() : impl_->counters; }

const gpu::GpuStats& Renderer::GpuCounters() const { return impl_->gpu_.Stats(); }

const std::string& Renderer::LastError() const { return impl_->lastError_; }

bool Renderer::Failed() const { return core2_ ? core2_->Failed() : impl_->failed_; }

} // namespace coregs::writeback
