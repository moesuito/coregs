// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 GT4Recomp contributors.
// Written from docs/hw-renderer-spec (clean room, decision 0069): chapters 01.4, 01.5, 02.9,
// 03.3 to 03.5 and 08. See IMPLEMENTATION-NOTES.md, entries IN-300 to IN-312.
#include "cpuref_draw.hpp"

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdlib>
#include <optional>

#include "hw_gs_pixel.hpp"
#include "hw_gs_registers.hpp"
#include "hw_gs_types.hpp"

namespace coregs::cpuref {

namespace {

using pixel::Colour;

// Exact 64-bit edge function in 12.4 units. Positive on the interior side once the triangle
// is normalised to a positive area.
int64_t EdgeFunction(int64_t ax, int64_t ay, int64_t bx, int64_t by, int64_t px, int64_t py) {
    return (bx - ax) * (py - ay) - (by - ay) * (px - ax);
}

// ceil(value / 16) for a 12.4 value (arithmetic shift rounds towards minus infinity).
int64_t CeilPixel(int64_t value) { return (value + 15) >> 4; }

// Inclusive window pixel rectangle.
struct Clip {
    int32_t x0 = 0;
    int32_t x1 = -1;
    int32_t y0 = 0;
    int32_t y1 = -1;
};

Clip Intersect(const Clip& a, const Clip& b) {
    return Clip{std::max(a.x0, b.x0), std::min(a.x1, b.x1), std::max(a.y0, b.y0), std::min(a.y1, b.y1)};
}

// Attributes of one fragment before the texture stage. Colour is in 0..255 units.
struct Sample {
    double r = 0, g = 0, b = 0, a = 0;
    double fog = 0;
    double z = 0;
    double s = 0, t = 0, q = 1;
    double u = 0, v = 0;  // 10.4 texels
};

enum class TexClass : uint8_t { Ct32, Ct24, Ct16, Indexed };

struct TextureUnit {
    bool enabled = false;
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

struct DrawContext {
    memory::LocalMemory* local = nullptr;
    pixel::PixelState pixel;
    memory::BufferView frameView{0, 0, memory::Psm::Ct32};
    memory::BufferView zView{0, 0, memory::Psm::Z32};
    TextureUnit texture;
    Colour fogColour;
    bool fogEnabled = false;
    bool flat = false;  // IIP = 0
    bool blendEnabled = false;
    Clip scissor;
    DrawStats* stats = nullptr;
};

// ---- Register decoding helpers --------------------------------------------------------

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
            return pixel::FrameFormat::Ct32;  // CT32, Z32 and other codes (V-01-2 counts these)
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
            return pixel::DepthCompare::Greater;  // kZtstGreater
    }
}

pixel::AlphaCompare AlphaCompareOf(uint32_t atst) {
    // GS order: NEVER, ALWAYS, LESS, LEQUAL, EQUAL, GEQUAL, GREATER, NOTEQUAL.
    return static_cast<pixel::AlphaCompare>(atst & 7u);
}

pixel::AlphaFail AlphaFailOf(uint32_t afail) {
    return static_cast<pixel::AlphaFail>(afail & 3u);  // KEEP, FB_ONLY, ZB_ONLY, RGB_ONLY
}

Colour PackedColour(uint32_t packed) {
    Colour c;
    c.r = static_cast<int32_t>(packed & 0xFFu);
    c.g = static_cast<int32_t>((packed >> 8) & 0xFFu);
    c.b = static_cast<int32_t>((packed >> 16) & 0xFFu);
    c.a = static_cast<int32_t>((packed >> 24) & 0xFFu);
    return c;
}

// Truncates a colour or fog value to 0..255 (IN-304).
int32_t ToByte(double value) {
    if (!(value > 0.0)) {
        return 0;
    }
    if (value >= 255.0) {
        return 255;
    }
    return static_cast<int32_t>(value);
}

// Texel coordinate (texels) to 10.4 fixed point, saturated far outside the 1024 range so
// that wrapping stays defined.
int32_t ToTexel104(double texels) {
    const double scaled = std::floor(texels * 16.0);
    if (scaled > 1.0e9) {
        return 1000000000;
    }
    if (scaled < -1.0e9) {
        return -1000000000;
    }
    return static_cast<int32_t>(scaled);
}

// ---- Texture stage -------------------------------------------------------------------

Colour FetchTexel(const DrawContext& d, const TextureUnit& t, int32_t x, int32_t y) {
    const uint32_t value = d.local->readPixel(t.view, static_cast<uint32_t>(x), static_cast<uint32_t>(y));
    switch (t.cls) {
        case TexClass::Ct32:
            return PackedColour(value);
        case TexClass::Ct24:
            return pixel::expand_texel24(value, static_cast<uint8_t>(t.texa.ta0), t.texa.aem);
        case TexClass::Ct16:
            return pixel::expand_texel16(static_cast<uint16_t>(value), static_cast<uint8_t>(t.texa.ta0),
                                         static_cast<uint8_t>(t.texa.ta1), t.texa.aem);
        case TexClass::Indexed:
            return PackedColour(t.palette.entries[value & 255u]);
    }
    return Colour{};
}

Colour SampleTexture(const DrawContext& d, const TextureUnit& t, int32_t uvs104, int32_t uvt104) {
    const int32_t tw = static_cast<int32_t>(t.twLog2);
    const int32_t th = static_cast<int32_t>(t.thLog2);
    if (!t.linear) {
        const int32_t x = pixel::wrap_texel(pixel::nearest_texel(uvs104), tw, t.wrapS);
        const int32_t y = pixel::wrap_texel(pixel::nearest_texel(uvt104), th, t.wrapT);
        return FetchTexel(d, t, x, y);
    }
    // Bilinear with 4-bit weights per axis; the sum is shifted by 8 (IN-305).
    const pixel::BilinearSample bs = pixel::bilinear_base(uvs104);
    const pixel::BilinearSample bt = pixel::bilinear_base(uvt104);
    const int32_t x0 = pixel::wrap_texel(bs.base, tw, t.wrapS);
    const int32_t x1 = pixel::wrap_texel(bs.base + 1, tw, t.wrapS);
    const int32_t y0 = pixel::wrap_texel(bt.base, th, t.wrapT);
    const int32_t y1 = pixel::wrap_texel(bt.base + 1, th, t.wrapT);
    const int32_t fx = bs.fraction;
    const int32_t fy = bt.fraction;
    const Colour c00 = FetchTexel(d, t, x0, y0);
    const Colour c10 = FetchTexel(d, t, x1, y0);
    const Colour c01 = FetchTexel(d, t, x0, y1);
    const Colour c11 = FetchTexel(d, t, x1, y1);
    const int32_t w00 = (16 - fx) * (16 - fy);
    const int32_t w10 = fx * (16 - fy);
    const int32_t w01 = (16 - fx) * fy;
    const int32_t w11 = fx * fy;
    auto mix = [&](int32_t a, int32_t b, int32_t c, int32_t e) {
        return (a * w00 + b * w10 + c * w01 + e * w11) >> 8;
    };
    Colour out;
    out.r = mix(c00.r, c10.r, c01.r, c11.r);
    out.g = mix(c00.g, c10.g, c01.g, c11.g);
    out.b = mix(c00.b, c10.b, c01.b, c11.b);
    out.a = mix(c00.a, c10.a, c01.a, c11.a);
    return out;
}

// ---- Per-pixel pipeline (chapter 01.5 order) ------------------------------------------

void ShadePixel(DrawContext& d, int32_t x, int32_t y, const Sample& sample) {
    Colour fragment;
    fragment.r = ToByte(sample.r);
    fragment.g = ToByte(sample.g);
    fragment.b = ToByte(sample.b);
    fragment.a = ToByte(sample.a);

    Colour colour = fragment;
    if (d.texture.enabled) {
        int32_t uvs = 0;
        int32_t uvt = 0;
        if (d.texture.fst) {
            uvs = ToTexel104(sample.u / 16.0);
            uvt = ToTexel104(sample.v / 16.0);
        } else {
            // Per-pixel divide by Q (03.5). A zero Q is treated as 1 (IN-310).
            const double q = sample.q == 0.0 ? 1.0 : sample.q;
            const double scaleS = static_cast<double>(1u << d.texture.twLog2);
            const double scaleT = static_cast<double>(1u << d.texture.thLog2);
            uvs = ToTexel104((sample.s / q) * scaleS);
            uvt = ToTexel104((sample.t / q) * scaleT);
        }
        const Colour texel = SampleTexture(d, d.texture, uvs, uvt);
        colour = pixel::texture_function(d.texture.tfx, d.texture.tcc, texel, fragment);
    }
    if (d.fogEnabled) {
        colour = pixel::apply_fog(colour, ToByte(sample.fog), d.fogColour);
    }

    const uint32_t ux = static_cast<uint32_t>(x);
    const uint32_t uy = static_cast<uint32_t>(y);
    const uint32_t z = pixel::interpolated_z(sample.z, d.pixel.z_format);
    const uint32_t frameOld = d.local->readPixel(d.frameView, ux, uy);
    const uint32_t depthOld = d.local->readPixel(d.zView, ux, uy);

    pixel::Fragment frag;
    frag.x = x;
    frag.y = y;
    frag.colour = colour;
    frag.z = z;
    const pixel::PixelOutcome outcome = pixel::process_pixel(d.pixel, frag, frameOld, depthOld);
    d.stats->coveredPixels++;
    if (d.blendEnabled) {
        d.stats->blendedPixels++;
    }
    if (outcome.frame_written) {
        d.local->writePixel(d.frameView, ux, uy, outcome.frame_stored);
        d.stats->framePixelsWritten++;
    }
    if (outcome.depth_written) {
        d.local->writePixel(d.zView, ux, uy, outcome.depth_stored);
        d.stats->depthPixelsWritten++;
    }
}

// Pixel bounds of 12.4 vertices, as window pixels.
Clip BoundsOf(const Vertex* vertices, size_t count) {
    int64_t minX = vertices[0].x4, maxX = vertices[0].x4;
    int64_t minY = vertices[0].y4, maxY = vertices[0].y4;
    for (size_t i = 1; i < count; ++i) {
        minX = std::min<int64_t>(minX, vertices[i].x4);
        maxX = std::max<int64_t>(maxX, vertices[i].x4);
        minY = std::min<int64_t>(minY, vertices[i].y4);
        maxY = std::max<int64_t>(maxY, vertices[i].y4);
    }
    return Clip{static_cast<int32_t>(minX >> 4), static_cast<int32_t>(maxX >> 4),
                static_cast<int32_t>(minY >> 4), static_cast<int32_t>(maxY >> 4)};
}

// Barycentric interpolation (flat colour from the last vertex, IN-304).
// One interpolated attribute. A constant attribute returns its exact vertex value: the barycentric
// weights are doubles that do not always sum to 1, so the general expression turns a constant fog of
// 90 into 89.999.. and the 0..255 truncation of 03.7 would lose one unit (IN-328). The rule holds on
// hardware too, where a constant attribute interpolates to itself.
double Lerp3(double a, double b, double c, double w0, double w1, double w2) {
    if (a == b && b == c) {
        return a;
    }
    return w0 * a + w1 * b + w2 * c;
}

Sample Blend(const Vertex& v0, const Vertex& v1, const Vertex& v2, double w0, double w1, double w2, bool flat) {
    Sample s;
    if (flat) {
        s.r = v2.r;
        s.g = v2.g;
        s.b = v2.b;
        s.a = v2.a;
    } else {
        s.r = Lerp3(v0.r, v1.r, v2.r, w0, w1, w2);
        s.g = Lerp3(v0.g, v1.g, v2.g, w0, w1, w2);
        s.b = Lerp3(v0.b, v1.b, v2.b, w0, w1, w2);
        s.a = Lerp3(v0.a, v1.a, v2.a, w0, w1, w2);
    }
    s.fog = Lerp3(v0.fog, v1.fog, v2.fog, w0, w1, w2);
    s.z = Lerp3(static_cast<double>(v0.z), static_cast<double>(v1.z), static_cast<double>(v2.z), w0, w1, w2);
    s.s = Lerp3(v0.s, v1.s, v2.s, w0, w1, w2);
    s.t = Lerp3(v0.t, v1.t, v2.t, w0, w1, w2);
    s.q = Lerp3(v0.q, v1.q, v2.q, w0, w1, w2);
    s.u = Lerp3(static_cast<double>(v0.u), static_cast<double>(v1.u), static_cast<double>(v2.u), w0, w1, w2);
    s.v = Lerp3(static_cast<double>(v0.v), static_cast<double>(v1.v), static_cast<double>(v2.v), w0, w1, w2);
    return s;
}

// Barycentric weights of pixel (px, py) when its centre is covered (IN-302).
bool TriangleWeights(const Vertex& a, const Vertex& b, const Vertex& c, int32_t px, int32_t py,
                     double& w0, double& w1, double& w2) {
    const int64_t ax = a.x4, ay = a.y4, bx = b.x4, by = b.y4, cx = c.x4, cy = c.y4;
    const int64_t area = EdgeFunction(ax, ay, bx, by, cx, cy);
    if (area == 0) {
        return false;
    }
    const int64_t sign = area > 0 ? 1 : -1;
    const int64_t centreX = static_cast<int64_t>(px) * 16;
    const int64_t centreY = static_cast<int64_t>(py) * 16;
    // Edge opposite each vertex: 0 = b to c, 1 = c to a, 2 = a to b.
    const int64_t e[3] = {
        EdgeFunction(bx, by, cx, cy, centreX, centreY),
        EdgeFunction(cx, cy, ax, ay, centreX, centreY),
        EdgeFunction(ax, ay, bx, by, centreX, centreY),
    };
    // Edge directions after normalising the orientation to positive area.
    const int64_t dirX[3] = {(cx - bx) * sign, (ax - cx) * sign, (bx - ax) * sign};
    const int64_t dirY[3] = {(cy - by) * sign, (ay - cy) * sign, (by - ay) * sign};
    for (int i = 0; i < 3; ++i) {
        const int64_t value = e[i] * sign;  // positive inside
        if (value < 0) {
            return false;
        }
        if (value == 0) {
            // Top-left rule (01.4): boundary pixels belong to top edges (horizontal, going
            // right) and left edges (going up) in the normalised orientation.
            const bool topLeft = dirY[i] < 0 || (dirY[i] == 0 && dirX[i] > 0);
            if (!topLeft) {
                return false;
            }
        }
    }
    const double denominator = static_cast<double>(area);
    w0 = static_cast<double>(e[0]) / denominator;
    w1 = static_cast<double>(e[1]) / denominator;
    w2 = static_cast<double>(e[2]) / denominator;
    return true;
}

void DrawTriangle(DrawContext& d, const Vertex& v0, const Vertex& v1, const Vertex& v2, const PixelRect& rect) {
    const Vertex verts[3] = {v0, v1, v2};
    Clip box = BoundsOf(verts, 3);
    box = Intersect(box, Clip{rect.x0, rect.x1, rect.y0, rect.y1});
    box = Intersect(box, d.scissor);
    for (int32_t py = box.y0; py <= box.y1; ++py) {
        for (int32_t px = box.x0; px <= box.x1; ++px) {
            double w0 = 0, w1 = 0, w2 = 0;
            if (!TriangleWeights(v0, v1, v2, px, py, w0, w1, w2)) {
                continue;
            }
            ShadePixel(d, px, py, Blend(v0, v1, v2, w0, w1, w2, d.flat));
        }
    }
}

void DrawSprite(DrawContext& d, const Vertex& v0, const Vertex& v1, const PixelRect& rect) {
    // Pixel n is covered when min <= n < max on each axis (IN-303).
    const int64_t minX = std::min<int64_t>(v0.x4, v1.x4), maxX = std::max<int64_t>(v0.x4, v1.x4);
    const int64_t minY = std::min<int64_t>(v0.y4, v1.y4), maxY = std::max<int64_t>(v0.y4, v1.y4);
    Clip box{static_cast<int32_t>(CeilPixel(minX)), static_cast<int32_t>(CeilPixel(maxX) - 1),
             static_cast<int32_t>(CeilPixel(minY)), static_cast<int32_t>(CeilPixel(maxY) - 1)};
    box = Intersect(box, Clip{rect.x0, rect.x1, rect.y0, rect.y1});
    box = Intersect(box, d.scissor);
    const double spanX = (v1.x4 - v0.x4) / 16.0;
    const double spanY = (v1.y4 - v0.y4) / 16.0;
    for (int32_t py = box.y0; py <= box.y1; ++py) {
        for (int32_t px = box.x0; px <= box.x1; ++px) {
            if (!SpriteCoversPixel(v0.x4, v1.x4, px) || !SpriteCoversPixel(v0.y4, v1.y4, py)) {
                continue;
            }
            const double fx = spanX == 0.0 ? 0.0 : (px - v0.x4 / 16.0) / spanX;
            const double fy = spanY == 0.0 ? 0.0 : (py - v0.y4 / 16.0) / spanY;
            // Colour, Z, fog and Q come from the second vertex; texture coordinates interpolate
            // across the rectangle (01.4, 03.2).
            Sample sample;
            sample.r = v1.r;
            sample.g = v1.g;
            sample.b = v1.b;
            sample.a = v1.a;
            sample.fog = v1.fog;
            sample.z = static_cast<double>(v1.z);
            sample.q = v1.q;
            sample.s = v0.s + (v1.s - v0.s) * fx;
            sample.t = v0.t + (v1.t - v0.t) * fy;
            sample.u = static_cast<double>(v0.u) + (static_cast<double>(v1.u) - v0.u) * fx;
            sample.v = static_cast<double>(v0.v) + (static_cast<double>(v1.v) - v0.v) * fy;
            ShadePixel(d, px, py, sample);
        }
    }
}

void DrawLine(DrawContext& d, const Vertex& v0, const Vertex& v1, const PixelRect& rect) {
    const int32_t x0 = WindowPixel(v0.x4), y0 = WindowPixel(v0.y4);
    const int32_t x1 = WindowPixel(v1.x4), y1 = WindowPixel(v1.y4);
    const int64_t dX = static_cast<int64_t>(x1) - x0;
    const int64_t dY = static_cast<int64_t>(y1) - y0;
    const int64_t len2 = dX * dX + dY * dY;
    const Clip clip = Intersect(Intersect(d.scissor, Clip{rect.x0, rect.x1, rect.y0, rect.y1}),
                                Clip{0, 2047, 0, 2047});
    for (const PixelCoord& p : LinePixels(v0.x4, v0.y4, v1.x4, v1.y4)) {
        if (p.x < clip.x0 || p.x > clip.x1 || p.y < clip.y0 || p.y > clip.y1) {
            continue;
        }
        const double t = len2 == 0 ? 0.0
                                   : static_cast<double>((p.x - x0) * dX + (p.y - y0) * dY) / static_cast<double>(len2);
        const double w1 = t;
        const double w0 = 1.0 - t;
        Sample sample;
        sample.r = d.flat ? v1.r : w0 * v0.r + w1 * v1.r;
        sample.g = d.flat ? v1.g : w0 * v0.g + w1 * v1.g;
        sample.b = d.flat ? v1.b : w0 * v0.b + w1 * v1.b;
        sample.a = d.flat ? v1.a : w0 * v0.a + w1 * v1.a;
        sample.fog = w0 * v0.fog + w1 * v1.fog;
        sample.z = w0 * static_cast<double>(v0.z) + w1 * static_cast<double>(v1.z);
        sample.s = w0 * v0.s + w1 * v1.s;
        sample.t = w0 * v0.t + w1 * v1.t;
        sample.q = w0 * v0.q + w1 * v1.q;
        sample.u = w0 * static_cast<double>(v0.u) + w1 * static_cast<double>(v1.u);
        sample.v = w0 * static_cast<double>(v0.v) + w1 * static_cast<double>(v1.v);
        ShadePixel(d, p.x, p.y, sample);
    }
}

void DrawPoint(DrawContext& d, const Vertex& v, const PixelRect& rect) {
    const int32_t x = WindowPixel(v.x4);
    const int32_t y = WindowPixel(v.y4);
    const Clip clip = Intersect(d.scissor, Clip{rect.x0, rect.x1, rect.y0, rect.y1});
    if (x < clip.x0 || x > clip.x1 || y < clip.y0 || y > clip.y1) {
        return;
    }
    Sample sample;
    sample.r = v.r;
    sample.g = v.g;
    sample.b = v.b;
    sample.a = v.a;
    sample.fog = v.fog;
    sample.z = static_cast<double>(v.z);
    sample.s = v.s;
    sample.t = v.t;
    sample.q = v.q;
    sample.u = v.u;
    sample.v = v.v;
    ShadePixel(d, x, y, sample);
}

// Builds the texture unit from TEX0, TEX1, CLAMP, TEXA and the palette (01.1, 07.5). Returns
// false when the texture cannot be used: invalid format or an axis above 1024 texels without
// a region mode (01.5, IN-307).
bool BuildTexture(const Batch& batch, const clut::ClutUnit& clut, TextureUnit& out) {
    const BatchState& state = batch.state;
    const uint64_t tex0Register = state.ctx.tex0;
    const uint64_t tex1Register = state.ctx.tex1;
    const uint64_t clampRegister = state.ctx.clamp;
    const uint32_t psmCode = tex0::Psm(tex0Register);
    const std::optional<TexClass> cls = TextureClassOf(psmCode);
    const auto psm = memory::decodePsm(psmCode);
    if (!cls || !psm) {
        return false;
    }
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
    out.linear = tex1::Mmag(tex1Register) != 0;  // LOD <= 0 uses MMAG; mip levels are M13 (IN-306)
    out.tcc = tex0::Tcc(tex0Register) != 0;
    out.tfx = static_cast<pixel::Tfx>(tex0::Tfx(tex0Register) & 3u);
    out.texa = clut::TexaState::FromRegister(state.env.texa);
    if (out.cls == TexClass::Indexed) {
        // TASK-168: the first palette segment's (the CLUT unit's buffer for a batch without segments).
        out.palette = DecodeSegmentPalette(batch, 0, clut);
    }
    return true;
}

}  // namespace

// ---- Public coverage helpers ------------------------------------------------------------

int32_t WindowPixel(int32_t coordinate4) { return coordinate4 >> 4; }

bool TriangleCoversPixel(const Vertex& a, const Vertex& b, const Vertex& c, int32_t px, int32_t py) {
    double w0 = 0, w1 = 0, w2 = 0;
    return TriangleWeights(a, b, c, px, py, w0, w1, w2);
}

bool SpriteCoversPixel(int32_t firstCoordinate4, int32_t secondCoordinate4, int32_t n) {
    const int64_t lo = std::min(firstCoordinate4, secondCoordinate4);
    const int64_t hi = std::max(firstCoordinate4, secondCoordinate4);
    const int64_t centre = static_cast<int64_t>(n) * 16;
    return centre >= lo && centre < hi;
}

std::vector<PixelCoord> LinePixels(int32_t x0_4, int32_t y0_4, int32_t x1_4, int32_t y1_4) {
    std::vector<PixelCoord> out;
    int32_t x = WindowPixel(x0_4);
    int32_t y = WindowPixel(y0_4);
    const int32_t xEnd = WindowPixel(x1_4);
    const int32_t yEnd = WindowPixel(y1_4);
    const int32_t dx = std::abs(xEnd - x);
    const int32_t dy = -std::abs(yEnd - y);
    const int32_t sx = x < xEnd ? 1 : -1;
    const int32_t sy = y < yEnd ? 1 : -1;
    int32_t error = dx + dy;
    for (;;) {
        out.push_back(PixelCoord{x, y});
        if (x == xEnd && y == yEnd) {
            break;
        }
        const int32_t doubled = 2 * error;
        if (doubled >= dy) {
            error += dy;
            x += sx;
        }
        if (doubled <= dx) {
            error += dx;
            y += sy;
        }
    }
    return out;
}

// ---- Batch drawing --------------------------------------------------------------------

void DrawBatch(const Batch& batch, memory::LocalMemory& local, const clut::ClutUnit& clut, DrawStats& stats) {
    stats.batches++;
    const BatchState& state = batch.state;
    const EnvironmentRegs& env = state.env;
    const ContextRegs& ctx = state.ctx;
    const uint64_t prim = state.effectivePrim;

    DrawContext d;
    d.local = &local;
    d.stats = &stats;
    d.flat = ((prim >> primbit::kIip) & 1u) == 0;
    d.fogEnabled = ((prim >> primbit::kFge) & 1u) != 0;
    d.blendEnabled = ((prim >> primbit::kAbe) & 1u) != 0;
    d.fogColour = Colour{static_cast<int32_t>(Field(env.fogcol, 0, 8)),
                         static_cast<int32_t>(Field(env.fogcol, 8, 8)),
                         static_cast<int32_t>(Field(env.fogcol, 16, 8)), 0};
    d.scissor = Clip{static_cast<int32_t>(Field(ctx.scissor, 0, 11)),
                     static_cast<int32_t>(Field(ctx.scissor, 16, 11)),
                     static_cast<int32_t>(Field(ctx.scissor, 32, 11)),
                     static_cast<int32_t>(Field(ctx.scissor, 48, 11))};

    // Frame and depth views. FBP and ZBP are page addresses (01.1), converted to blocks.
    const uint32_t framePsmCode = frame::Psm(ctx.frame);
    const auto framePsm = memory::decodePsm(framePsmCode);
    d.frameView = memory::BufferView{memory::blockOfPage(frame::Fbp(ctx.frame)), frame::Fbw(ctx.frame),
                                     framePsm ? *framePsm : memory::Psm::Ct32};
    // ZBUF.PSM is 4 bits; the high bits of the Z format codes are implied (01.1): 0x30 | field.
    const uint32_t zbufPsmCode = 0x30u | zbuf::Psm(ctx.zbuf);
    const auto zPsm = memory::decodePsm(zbufPsmCode);
    d.zView = memory::BufferView{memory::blockOfPage(zbuf::Zbp(ctx.zbuf)), frame::Fbw(ctx.frame),
                                 zPsm ? *zPsm : memory::Psm::Z32};

    // Per-pixel state (01.5, 08).
    pixel::PixelState& ps = d.pixel;
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
    ps.blend.enabled = d.blendEnabled;
    ps.blend.pabe = (env.pabe & 1u) != 0;
    ps.blend.colour_clamp = (env.colclamp & 1u) != 0;

    // Texture unit (only when TME is set).
    const bool tme = ((prim >> primbit::kTme) & 1u) != 0;
    d.texture.enabled = tme;
    if (tme) {
        d.texture.fst = ((prim >> primbit::kFst) & 1u) != 0;
        if (!BuildTexture(batch, clut, d.texture)) {
            stats.primitivesSkippedTexture += batch.prims.size();
            return;
        }
    }

    // TASK-168: each palette segment's primitives draw with that segment's palette.
    const bool indexed = tme && d.texture.cls == TexClass::Indexed;
    uint32_t segment = 0;
    uint32_t segmentEnd = PaletteSegmentEndPrim(batch, 0);
    for (uint32_t primIndex = 0; primIndex < batch.prims.size(); ++primIndex) {
        const PrimRecord& record = batch.prims[primIndex];
        if (primIndex >= segmentEnd && segment + 1u < PaletteSegmentCount(batch)) {
            while (primIndex >= segmentEnd && segment + 1u < PaletteSegmentCount(batch)) {
                ++segment;
                segmentEnd = PaletteSegmentEndPrim(batch, segment);
            }
            if (indexed) d.texture.palette = DecodeSegmentPalette(batch, segment, clut);
        }
        stats.primitives++;
        if (record.rect.Empty()) {
            continue;
        }
        const Vertex* v = &batch.vertices[record.first];
        switch (record.type) {
            case PrimType::Point:
                DrawPoint(d, v[0], record.rect);
                break;
            case PrimType::Line:
            case PrimType::LineStrip:
                DrawLine(d, v[0], v[1], record.rect);
                break;
            case PrimType::Triangle:
            case PrimType::TriangleStrip:
            case PrimType::TriangleFan:
                DrawTriangle(d, v[0], v[1], v[2], record.rect);
                break;
            case PrimType::Sprite:
                DrawSprite(d, v[0], v[1], record.rect);
                break;
            case PrimType::Reserved:
                break;
        }
        if (tme) {
            stats.primitivesTextured++;
        }
    }
}

}  // namespace coregs::cpuref
