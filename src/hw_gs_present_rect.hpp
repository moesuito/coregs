// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 João Alano and GT4Recomp contributors.
// Written from the TASK-026 specification (docs/hw-renderer-spec), clean room per decision 0069.
// Implements chapter 13.6 (fixed 16:9 presentation rectangle) and the TASK-080 display-aspect fill of the race view.
// Pure arithmetic, no Vulkan.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace coregs {

struct PresentRect {
  std::uint32_t left, top, width, height;
};

// Largest 16:9 rectangle centred in width x height. Zero extents give an empty rectangle;
// the arithmetic uses 64-bit intermediates so UINT32_MAX inputs cannot overflow.
inline PresentRect present_rect_widescreen(std::uint32_t width, std::uint32_t height) {
  if (!width || !height) return {0, 0, 0, 0};
  const std::uint64_t numerator = 16;
  const std::uint64_t denominator = 9;
  const auto fitted_width = static_cast<std::uint32_t>(std::max<std::uint64_t>(
      1, std::min<std::uint64_t>(width, height * numerator / denominator)));
  const auto fitted_height = static_cast<std::uint32_t>(std::max<std::uint64_t>(
      1, std::min<std::uint64_t>(height, width * denominator / numerator)));
  return {(width - fitted_width) / 2, (height - fitted_height) / 2, fitted_width, fitted_height};
}

// TASK-080 display-aspect fill (GS-STREAM-PROTOCOL 6.7 CONTROL display_aspect semantics; the same rectangle as the
// default renderer's display-aspect rectangle): the largest rectangle of `ratio` (width / height) centred
// in width x height, rounded to the nearest pixel, so a ratio wider than the window is letterboxed and never stretched
// to the whole window. A ratio of 0 (or not positive) is the protocol's "whole window".
inline PresentRect present_rect_ratio(std::uint32_t width, std::uint32_t height, double ratio) {
  if (!width || !height) return {0, 0, 0, 0};
  if (!(ratio > 0.0)) return {0, 0, width, height};
  const double fitted_by_height = static_cast<double>(height) * ratio;
  std::uint32_t fitted_width = width, fitted_height = height;
  if (fitted_by_height <= static_cast<double>(width)) {
    fitted_width = static_cast<std::uint32_t>(std::max(1.0, std::floor(fitted_by_height + 0.5)));
  } else {
    fitted_height = static_cast<std::uint32_t>(std::max(1.0, std::floor(static_cast<double>(width) / ratio + 0.5)));
  }
  fitted_width = std::min(fitted_width, width);
  fitted_height = std::min(fitted_height, height);
  return {(width - fitted_width) / 2, (height - fitted_height) / 2, fitted_width, fitted_height};
}

// The rectangle of one present: the race view filling the screen at `fill_ratio` when the game asks for it (a display
// ratio other than 16:9), otherwise the fixed 16:9 rectangle of chapter 13.6.
inline PresentRect present_rect_for(std::uint32_t width, std::uint32_t height, bool fill, double fill_ratio) {
  return fill ? present_rect_ratio(width, height, fill_ratio) : present_rect_widescreen(width, height);
}

// IN-406: the number of 2x2 box halvings before the final bilinear blit of an upscaled frame into a dst_w x dst_h
// rectangle. A level is halved only while the halved level still covers the destination on both axes, so the final
// blit always reduces (a ratio in (0.5, 1] on the tighter axis) and never magnifies a halved level back up. That is the
// reference's default "smooth" present for a reduction under 2x: one bilinear sample of the whole image
// instead of halving to half the destination: a 3x frame in a 1080p window
// (1920 x 1344 into 1920 x 1080) was halved to 960 x 672 and stretched back: half the detail was thrown away (owner
// test 13). max_levels is the image's mip level count (level 0 included).
inline std::uint32_t present_box_halvings(std::uint32_t src_w, std::uint32_t src_h, std::uint32_t dst_w,
                                          std::uint32_t dst_h, std::uint32_t max_levels) {
  std::uint32_t levels = 0;
  while (levels + 1 < max_levels && (src_w >> 1) >= dst_w && (src_h >> 1) >= dst_h && dst_w && dst_h) {
    src_w >>= 1;
    src_h >>= 1;
    ++levels;
  }
  return levels;
}

// The fill request packed into one 64-bit word so it can cross threads atomically: bit 0 = fill, bits 32..63 = the
// target ratio x 1e6 (0 = the whole window), as the stream protocol carries it.
inline std::uint64_t pack_display_aspect(bool fill, std::uint32_t ratio_millionths) {
  return (fill ? 1u : 0u) | (static_cast<std::uint64_t>(fill ? ratio_millionths : 0u) << 32);
}
inline bool display_aspect_fill(std::uint64_t packed) { return (packed & 1u) != 0; }
inline double display_aspect_ratio(std::uint64_t packed) { return static_cast<double>(packed >> 32) / 1e6; }

} // namespace coregs
