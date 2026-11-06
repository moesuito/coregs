// CTest hw_present_transform: TASK-179, the swapchain pre-transform chosen from a surface's transforms and the blit that
// draws the picture for it (hw_present_transform.hpp), host only. No device here rotates its surface, so the arithmetic
// is checked on its own: every transform the surface may report, and every pixel of a picture through the blit's rule.
#include "hw_present_transform.hpp"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <set>
#include <utility>

using namespace coregs;

namespace {

void check(bool condition, const char *what) {
  if (condition) return;
  std::fprintf(stderr, "hw_present_transform: %s\n", what);
  std::exit(1);
}

constexpr std::uint32_t kAll = 0x1FF;

// vkCmdBlitImage: destination pixel i of an axis runs from offset 0 towards offset 1; reversed offsets mirror the axis.
std::int64_t blit_pixel(std::int32_t from, std::int32_t to, std::int64_t i) { return from <= to ? from + i : from - 1 - i; }

// Every pixel of a w x h picture through the blit lands where present_pixel_position says, inside the image, once.
void check_blit(const PresentRect &rect, std::uint32_t width, std::uint32_t height, const PresentTransform &transform) {
  const BlitDestination d = present_blit_destination(rect, width, height, transform);
  check((d.x0 > d.x1) == transform.reverse_x && (d.y0 > d.y1) == transform.reverse_y, "reversed offsets on the reversed axes");
  check(static_cast<std::uint32_t>(d.x0 > d.x1 ? d.x0 - d.x1 : d.x1 - d.x0) == rect.width, "the blit keeps the width");
  check(static_cast<std::uint32_t>(d.y0 > d.y1 ? d.y0 - d.y1 : d.y1 - d.y0) == rect.height, "the blit keeps the height");
  std::set<std::pair<std::int64_t, std::int64_t>> seen;
  for (std::int64_t y = 0; y < rect.height; ++y) {
    for (std::int64_t x = 0; x < rect.width; ++x) {
      const PixelPosition expected = present_pixel_position(rect, width, height, transform, x, y);
      check(blit_pixel(d.x0, d.x1, x) == expected.x && blit_pixel(d.y0, d.y1, y) == expected.y, "blit pixel position");
      check(expected.x >= 0 && expected.y >= 0 && expected.x < width && expected.y < height, "inside the image");
      seen.insert({expected.x, expected.y});
    }
  }
  check(seen.size() == static_cast<std::size_t>(rect.width) * rect.height, "every picture pixel lands once");
  // Read back through the presentation engine's transform (each one undoes itself), the picture is upright at `rect`.
  for (std::int64_t y = 0; y < rect.height; y += rect.height - 1) {
    for (std::int64_t x = 0; x < rect.width; x += rect.width - 1) {
      const PixelPosition p = present_pixel_position(rect, width, height, transform, x, y);
      const std::int64_t shownX = transform.reverse_x ? width - 1 - p.x : p.x;
      const std::int64_t shownY = transform.reverse_y ? height - 1 - p.y : p.y;
      check(shownX == rect.left + x && shownY == rect.top + y, "shown upright after the engine's transform");
    }
  }
}

}  // namespace

int main() {
  // Windows and desktop Linux: the identity, as before (the picture's rectangle unchanged).
  {
    const PresentTransform t = choose_present_transform(kTransformIdentity, kTransformIdentity);
    check(t.handled && t.pre_transform == kTransformIdentity && t.upright(), "a desktop surface: identity");
    const PresentRect rect = present_rect_widescreen(1720, 720);
    const BlitDestination d = present_blit_destination(rect, 1720, 720, t);
    check(d.x0 == static_cast<std::int32_t>(rect.left) && d.y0 == static_cast<std::int32_t>(rect.top) &&
              d.x1 == static_cast<std::int32_t>(rect.left + rect.width) && d.y1 == static_cast<std::int32_t>(rect.top + rect.height),
          "identity: the blit of before");
  }
  // An Android surface on a turned display: every transform supported, the current one a quarter turn. The identity is
  // chosen (the compositor turns the picture), where before the quarter turn was passed with an unturned picture.
  for (std::uint32_t current : {kTransformRotate90, kTransformRotate180, kTransformRotate270, kTransformMirrorRotate90}) {
    const PresentTransform t = choose_present_transform(kAll, current);
    check(t.handled && t.pre_transform == kTransformIdentity && t.upright(), "identity whenever supported");
  }
  // No identity: inherit, then the transforms a blit can draw (the surface's current one first), else refused.
  check(choose_present_transform(kTransformInherit | kTransformRotate90, kTransformRotate90).pre_transform == kTransformInherit,
        "inherit before a quarter turn");
  {
    const PresentTransform t = choose_present_transform(kTransformRotate180 | kTransformRotate90, kTransformRotate90);
    check(t.handled && t.pre_transform == kTransformRotate180 && t.reverse_x && t.reverse_y, "a half turn: both axes");
  }
  {
    const PresentTransform t = choose_present_transform(kTransformMirror, kTransformMirror);
    check(t.handled && t.pre_transform == kTransformMirror && t.reverse_x && !t.reverse_y, "the mirror: x");
  }
  {
    const PresentTransform t = choose_present_transform(kTransformMirrorRotate180, kTransformMirrorRotate180);
    check(t.handled && t.pre_transform == kTransformMirrorRotate180 && !t.reverse_x && t.reverse_y, "mirror and half turn: y");
  }
  check(choose_present_transform(kTransformRotate180 | kTransformMirror, kTransformMirror).pre_transform == kTransformMirror,
        "the current transform first");
  for (std::uint32_t quarter : {kTransformRotate90, kTransformRotate270, kTransformMirrorRotate90, kTransformMirrorRotate270}) {
    const PresentTransform t = choose_present_transform(kTransformRotate90 | kTransformRotate270 | kTransformMirrorRotate90 |
                                                            kTransformMirrorRotate270,
                                                        quarter);
    check(!t.handled && t.pre_transform == quarter, "quarter turns only: refused");
  }

  // Every handled transform, on centred and off-centre rectangles of odd and even sizes.
  const PresentTransform handled[] = {choose_present_transform(kTransformIdentity, kTransformIdentity),
                                      choose_present_transform(kTransformRotate180, kTransformRotate180),
                                      choose_present_transform(kTransformMirror, kTransformMirror),
                                      choose_present_transform(kTransformMirrorRotate180, kTransformMirrorRotate180)};
  for (const PresentTransform &t : handled) {
    check_blit(present_rect_widescreen(64, 48), 64, 48, t);
    check_blit(present_rect_widescreen(37, 29), 37, 29, t);
    check_blit(PresentRect{3, 5, 10, 7}, 21, 17, t);
    check_blit(PresentRect{0, 0, 16, 9}, 16, 9, t);
    // A centred rectangle stays centred: the mirrored span is the same span.
    const PresentRect rect = present_rect_widescreen(1920, 1200);
    const BlitDestination d = present_blit_destination(rect, 1920, 1200, t);
    const std::int32_t low_x = d.x0 < d.x1 ? d.x0 : d.x1, low_y = d.y0 < d.y1 ? d.y0 : d.y1;
    check(low_x == static_cast<std::int32_t>(rect.left) && low_y == static_cast<std::int32_t>(rect.top), "centred stays centred");
  }
  std::printf("hw_present_transform: ok\n");
  return 0;
}
