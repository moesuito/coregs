// MIT License. Copyright (c) 2026 João Alano and GT4Recomp contributors.
// TASK-179 (Android plan stage A1): the swapchain pre-transform and where the picture goes for it. Written from the Vulkan
// specification (VkSwapchainCreateInfoKHR::preTransform, VkSurfaceTransformFlagBitsKHR, vkCmdBlitImage), clean room
// (decision 0070). Pure arithmetic, no Vulkan: the transform bits are the VkSurfaceTransformFlagBitsKHR values.
//
// preTransform tells the presentation engine which transform the application has ALREADY applied to the image content.
// The backend used to pass the surface's currentTransform and draw the picture unrotated, which is right only when that
// transform is the identity (Windows, Linux desktops). An Android surface usually reports the display's rotation there,
// so the picture would have been shown rotated or squashed. Now:
//   1. IDENTITY whenever the surface supports it (every desktop surface; Android's too): the picture is drawn as it is
//      and the platform's compositor rotates it where the display needs it. Nothing changes on Windows.
//   2. Else INHERIT when supported (the platform decides; the picture is drawn as it is).
//   3. Else a transform the final blit can express: ROTATE_180 (both axes reversed), HORIZONTAL_MIRROR (x reversed),
//      HORIZONTAL_MIRROR_ROTATE_180 (y reversed). The blit into the swapchain image writes the picture with those axes
//      reversed (vkCmdBlitImage mirrors along an axis whose destination offsets are reversed), so the presentation
//      engine's view of it is upright.
//   4. Else (only 90 or 270 degree transforms, with or without the mirror): not handled. A blit cannot turn an image by
//      a quarter turn, and no known surface lacks IDENTITY; the swapchain is refused with a clear error instead of
//      showing a turned picture. A device that needs it gets a rotation pass in Android stage A6.
#pragma once

#include <cstdint>

#include "hw_gs_present_rect.hpp"

namespace coregs {

constexpr std::uint32_t kTransformIdentity = 0x001;
constexpr std::uint32_t kTransformRotate90 = 0x002;
constexpr std::uint32_t kTransformRotate180 = 0x004;
constexpr std::uint32_t kTransformRotate270 = 0x008;
constexpr std::uint32_t kTransformMirror = 0x010;
constexpr std::uint32_t kTransformMirrorRotate90 = 0x020;
constexpr std::uint32_t kTransformMirrorRotate180 = 0x040;
constexpr std::uint32_t kTransformMirrorRotate270 = 0x080;
constexpr std::uint32_t kTransformInherit = 0x100;

struct PresentTransform {
  std::uint32_t pre_transform = kTransformIdentity;  // the one bit for VkSwapchainCreateInfoKHR::preTransform
  bool handled = true;     // false: no supported transform this backend can draw (case 4 above)
  bool reverse_x = false;  // the blit writes the picture with this axis reversed
  bool reverse_y = false;
  bool upright() const { return !reverse_x && !reverse_y; }
};

inline const char *transform_name(std::uint32_t bit) {
  switch (bit) {
  case kTransformIdentity: return "identity";
  case kTransformRotate90: return "rotate 90";
  case kTransformRotate180: return "rotate 180";
  case kTransformRotate270: return "rotate 270";
  case kTransformMirror: return "mirror";
  case kTransformMirrorRotate90: return "mirror rotate 90";
  case kTransformMirrorRotate180: return "mirror rotate 180";
  case kTransformMirrorRotate270: return "mirror rotate 270";
  case kTransformInherit: return "inherit";
  default: return "none";
  }
}

// `supported` and `current` as the surface reports them (VkSurfaceCapabilitiesKHR).
inline PresentTransform choose_present_transform(std::uint32_t supported, std::uint32_t current) {
  PresentTransform chosen;
  if (supported & kTransformIdentity) return chosen;
  if (supported & kTransformInherit) {
    chosen.pre_transform = kTransformInherit;
    return chosen;
  }
  // A transform the blit can draw, the surface's own first.
  struct Option {
    std::uint32_t bit;
    bool reverse_x, reverse_y;
  };
  constexpr Option options[] = {{kTransformRotate180, true, true},
                                {kTransformMirror, true, false},
                                {kTransformMirrorRotate180, false, true}};
  for (int pass = 0; pass < 2; ++pass) {
    for (const Option &option : options) {
      const bool wanted = pass == 0 ? option.bit == current : true;
      if (wanted && (supported & option.bit)) {
        chosen.pre_transform = option.bit;
        chosen.reverse_x = option.reverse_x;
        chosen.reverse_y = option.reverse_y;
        return chosen;
      }
    }
  }
  chosen.pre_transform = current;
  chosen.handled = false;
  return chosen;
}

// The destination offsets of the blit that puts the picture into `rect` of a width x height swapchain image under
// `transform`: (x0, y0) receives the picture's top-left corner and (x1, y1) its bottom-right one, so a reversed axis
// has x0 > x1 (or y0 > y1). The rectangle itself moves to its mirror position, so a centred rectangle stays centred.
struct BlitDestination {
  std::int32_t x0, y0, x1, y1;
};

inline BlitDestination present_blit_destination(const PresentRect &rect, std::uint32_t width, std::uint32_t height,
                                                const PresentTransform &transform) {
  const std::int64_t left = rect.left, top = rect.top, right = left + rect.width, bottom = top + rect.height;
  const std::int64_t w = width, h = height;
  BlitDestination d{};
  d.x0 = static_cast<std::int32_t>(transform.reverse_x ? w - left : left);
  d.x1 = static_cast<std::int32_t>(transform.reverse_x ? w - right : right);
  d.y0 = static_cast<std::int32_t>(transform.reverse_y ? h - top : top);
  d.y1 = static_cast<std::int32_t>(transform.reverse_y ? h - bottom : bottom);
  return d;
}

// Where the picture's pixel (x, y) of a width x height picture placed at `rect` lands in the swapchain image (pixel
// centres; the fixture checks the blit offsets against it).
struct PixelPosition {
  std::int64_t x, y;
};
inline PixelPosition present_pixel_position(const PresentRect &rect, std::uint32_t width, std::uint32_t height,
                                            const PresentTransform &transform, std::int64_t x, std::int64_t y) {
  const std::int64_t px = rect.left + x, py = rect.top + y;
  return {transform.reverse_x ? static_cast<std::int64_t>(width) - 1 - px : px,
          transform.reverse_y ? static_cast<std::int64_t>(height) - 1 - py : py};
}

}  // namespace coregs
