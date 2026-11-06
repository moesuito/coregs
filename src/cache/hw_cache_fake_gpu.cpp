// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 GT4Recomp contributors
//
// Written from the TASK-026 specification (docs/hw-renderer-spec), chapters 00, 04, 05 and 10. Clean room.

#include "hw_cache_fake_gpu.hpp"

#include <algorithm>

namespace coregs::cache::fake {

std::uint32_t drawPattern(std::uint32_t drawId, std::uint32_t x, std::uint32_t y) {
    std::uint32_t value = drawId * 0x9E3779B1u ^ (x * 0x85EBCA77u) ^ (y * 0xC2B2AE3Du);
    value ^= value >> 15;
    value *= 0x2C1B3C6Du;
    value ^= value >> 12;
    return value;
}

std::uint32_t FakeGpu::garbage(std::uint32_t imageId, std::uint64_t index) {
    return static_cast<std::uint32_t>(0x9E3779B9u * (imageId + 1u) + static_cast<std::uint32_t>(index) * 0x85EBCA6Bu);
}

FakeGpu::Image FakeGpu::makeImage(std::uint32_t imageId, std::uint32_t width, std::uint32_t height,
                                  const memory::BufferView& view) const {
    Image image;
    image.width = width;
    image.height = height;
    image.view = view;
    image.words.resize(static_cast<std::size_t>(width) * height);
    for (std::size_t index = 0; index < image.words.size(); ++index) {
        image.words[index] = garbage(imageId, index);
    }
    return image;
}

void FakeGpu::execute(const GpuOp& op, memory::LocalMemory& slm) {
    switch (op.kind) {
        case OpKind::CreateImage:
            images_[op.imageId] = makeImage(op.imageId, op.width, op.height, op.view);
            return;

        case OpKind::ResizeImage: {
            Image grown = makeImage(op.imageId, op.width, op.height, op.view);
            const auto found = images_.find(op.imageId);
            if (found != images_.end()) {
                const Image& old = found->second;
                for (std::uint32_t y = 0; y < old.height && y < op.height; ++y) {
                    for (std::uint32_t x = 0; x < old.width && x < op.width; ++x) {
                        grown.words[static_cast<std::size_t>(y) * op.width + x] =
                            old.words[static_cast<std::size_t>(y) * old.width + x];
                    }
                }
            }
            images_[op.imageId] = std::move(grown);
            return;
        }

        case OpKind::DeleteImage:
            images_.erase(op.imageId);
            return;

        case OpKind::UploadRect: {
            Image& image = images_.at(op.imageId);
            const std::uint32_t mask = channelWordMask(op.view.psm, op.channels);
            for (std::int32_t y = op.rect.y0; y < op.rect.y1; ++y) {
                for (std::int32_t x = op.rect.x0; x < op.rect.x1; ++x) {
                    if (static_cast<std::uint32_t>(x) >= image.width || static_cast<std::uint32_t>(y) >= image.height) {
                        ++outOfBounds_;
                        continue;
                    }
                    const std::uint32_t slmWord = slm.readPixel(op.view, static_cast<std::uint32_t>(x),
                                                                static_cast<std::uint32_t>(y));
                    std::uint32_t& word = image.words[static_cast<std::size_t>(y) * image.width + x];
                    word = (word & ~mask) | (slmWord & mask);
                }
            }
            return;
        }

        case OpKind::DownloadRect: {
            const Image& image = images_.at(op.imageId);
            const std::uint32_t mask = channelWordMask(op.view.psm, op.channels);
            for (std::int32_t y = op.rect.y0; y < op.rect.y1; ++y) {
                for (std::int32_t x = op.rect.x0; x < op.rect.x1; ++x) {
                    if (static_cast<std::uint32_t>(x) >= image.width || static_cast<std::uint32_t>(y) >= image.height) {
                        ++outOfBounds_;
                        continue;
                    }
                    const std::uint32_t word = image.words[static_cast<std::size_t>(y) * image.width + x];
                    const std::uint32_t slmWord = slm.readPixel(op.view, static_cast<std::uint32_t>(x),
                                                                static_cast<std::uint32_t>(y));
                    slm.writePixel(op.view, static_cast<std::uint32_t>(x), static_cast<std::uint32_t>(y),
                                   (slmWord & ~mask) | (word & mask));
                }
            }
            return;
        }

        case OpKind::Draw: {
            Image& image = images_.at(op.imageId);
            const std::uint32_t mask = channelWordMask(op.view.psm, op.channels);
            for (std::int32_t y = op.rect.y0; y < op.rect.y1; ++y) {
                for (std::int32_t x = op.rect.x0; x < op.rect.x1; ++x) {
                    if (static_cast<std::uint32_t>(x) >= image.width || static_cast<std::uint32_t>(y) >= image.height) {
                        ++outOfBounds_;
                        continue;
                    }
                    const std::uint32_t pattern = drawPattern(op.drawId, static_cast<std::uint32_t>(x),
                                                              static_cast<std::uint32_t>(y));
                    std::uint32_t& word = image.words[static_cast<std::size_t>(y) * image.width + x];
                    word = (word & ~mask) | (pattern & mask);
                }
            }
            return;
        }

        case OpKind::DecodeMemorySource: {
            const std::uint32_t width = static_cast<std::uint32_t>(op.rect.width());
            const std::uint32_t height = static_cast<std::uint32_t>(op.rect.height());
            Image image = makeImage(op.imageId, width, height, op.view);
            for (std::uint32_t y = 0; y < height; ++y) {
                for (std::uint32_t x = 0; x < width; ++x) {
                    image.words[static_cast<std::size_t>(y) * width + x] =
                        slm.readPixel(op.view, static_cast<std::uint32_t>(op.rect.x0) + x,
                                      static_cast<std::uint32_t>(op.rect.y0) + y);
                }
            }
            images_[op.imageId] = std::move(image);
            return;
        }
    }
}

std::uint32_t FakeGpu::texel(std::uint32_t imageId, std::uint32_t x, std::uint32_t y) const {
    const Image& image = images_.at(imageId);
    if (x >= image.width || y >= image.height) return 0;
    return image.words[static_cast<std::size_t>(y) * image.width + x];
}

// ---------------------------------------------------------------------------
// Reference
// ---------------------------------------------------------------------------

void ReferenceMemory::hostUpload(const memory::BufferView& destination, const memory::TransferRect& rect,
                                 std::span<const std::uint8_t> data) {
    memory::hostToLocal(slm, destination, rect, data);
}

void ReferenceMemory::localMove(const memory::BufferView& source, std::uint32_t sourceX, std::uint32_t sourceY,
                                const memory::BufferView& destination, std::uint32_t destinationX,
                                std::uint32_t destinationY, std::uint32_t width, std::uint32_t height,
                                memory::TransferDirection direction) {
    memory::localToLocal(slm, source, sourceX, sourceY, destination, destinationX, destinationY, width, height,
                         direction);
}

void ReferenceMemory::draw(const DrawCommand& command) {
    const std::uint32_t mask = channelWordMask(command.frame.psm, static_cast<std::uint8_t>(
                                                                      command.writeChannels &
                                                                      supportedChannels(command.frame.psm)));
    // Same frame-width clip as the cache (IN-08).
    const std::int32_t widthLimit = static_cast<std::int32_t>(std::max<std::uint32_t>(command.frame.bufferWidth, 1u) * 64u);
    const Rect rect = command.rect.intersect(Rect{0, 0, widthLimit, 0x7FFFFFFF});
    for (std::int32_t y = rect.y0; y < rect.y1; ++y) {
        for (std::int32_t x = rect.x0; x < rect.x1; ++x) {
            const std::uint32_t px = static_cast<std::uint32_t>(x);
            const std::uint32_t py = static_cast<std::uint32_t>(y);
            const std::uint32_t old = slm.readPixel(command.frame, px, py);
            const std::uint32_t pattern = drawPattern(command.drawId, px, py);
            slm.writePixel(command.frame, px, py, (old & ~mask) | (pattern & mask));
        }
    }
}

}  // namespace coregs::cache::fake
