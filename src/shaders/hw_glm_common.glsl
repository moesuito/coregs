// SPDX-License-Identifier: MIT
//
// GS local memory addressing on the GPU (TASK-027 M10 GLM, IN-347). Copyright (c) 2026 GT4Recomp contributors.
// Written from docs/hw-renderer-spec chapters 01.3 and 04.9, clean room (decision 0070).
//
// The GLM is a 4 MiB storage buffer holding GS local memory at 1x. The address tables come from
// memory::gpuAddressTables(): a 64-entry directory (4 uint each: table offset, page width, page height, Z
// block XOR flag) followed by one packed entry per pixel of a page (bits 0-4 block in the page, 5-12 byte in
// the block, 13-17 bit shift, 18-23 bit width). The arithmetic is the one of memory::pixelSlot (AfterAdd Z
// rule), so a GLM read equals LocalMemory::readPixel on the same bytes.

layout(std430, set = 0, binding = 0) coherent buffer GlmWords {
    uint glm_words[];
};
layout(std430, set = 0, binding = 1) readonly buffer AddressTables {
    uint glm_tables[];  // [0, 256): directory; [256, ...): entries
};

struct GlmSlot {
    uint word_index;  // 32-bit word of the GLM that holds the pixel
    uint shift;       // bit position of the pixel inside that word
    uint mask;        // mask of the pixel's bits (before the shift)
};

// The slot with an explicit number of pages per row (the CLUT unit uses CBW itself, 0 allowed).
GlmSlot glm_slot_rows(uint bp, uint per_row, uint psm, uint x, uint y) {
    uint dir = (psm & 63u) * 4u;
    uint table_offset = glm_tables[dir];
    uint page_width = glm_tables[dir + 1u];
    uint page_height = glm_tables[dir + 2u];
    uint z_xor = glm_tables[dir + 3u];
    uint page = (y / page_height) * per_row + (x / page_width);
    uint entry = glm_tables[256u + table_offset + (y % page_height) * page_width + (x % page_width)];
    uint table_block = entry & 31u;
    uint byte_in_block = (entry >> 5) & 255u;
    uint bit_shift = (entry >> 13) & 31u;
    uint bit_width = (entry >> 18) & 63u;
    uint block = bp + 32u * page + table_block;
    if (z_xor != 0u) {
        block ^= 24u;
    }
    block &= 16383u;
    uint byte_offset = (block * 256u + byte_in_block) & 0x3FFFFFu;
    GlmSlot slot;
    slot.word_index = byte_offset >> 2;
    slot.shift = (byte_offset & 3u) * 8u + bit_shift;
    slot.mask = bit_width >= 32u ? 0xFFFFFFFFu : ((1u << bit_width) - 1u);
    return slot;
}

GlmSlot glm_slot(uint bp, uint bw, uint psm, uint x, uint y) {
    uint page_width = glm_tables[(psm & 63u) * 4u + 1u];
    return glm_slot_rows(bp, max(1u, (bw * 64u) / page_width), psm, x, y);
}

uint glm_read(uint bp, uint bw, uint psm, uint x, uint y) {
    GlmSlot slot = glm_slot(bp, bw, psm, x, y);
    return (glm_words[slot.word_index] >> slot.shift) & slot.mask;
}

// Writes only the pixel's bits; other pixels sharing the word may be written concurrently (atomics).
void glm_write(uint bp, uint bw, uint psm, uint x, uint y, uint value) {
    GlmSlot slot = glm_slot(bp, bw, psm, x, y);
    if (slot.mask == 0xFFFFFFFFu) {
        glm_words[slot.word_index] = value;
        return;
    }
    atomicAnd(glm_words[slot.word_index], ~(slot.mask << slot.shift));
    atomicOr(glm_words[slot.word_index], (value & slot.mask) << slot.shift);
}
