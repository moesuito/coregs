// MIT License. Copyright (c) 2026 Joao Alano and GT4Recomp contributors.
// Written from the TASK-026 specification (docs/hw-renderer-spec), clean room (decision 0069).
//
// Present scheduler (M6 interface): a finished image and a decision about when to show it. Chapter 12.5
// assumes one present per guest VBlank; the owner's variable-frame-rate readiness requirements
// (docs/COREGS2-ROADMAP.md) need the decision separated from the VBlank, so that
//   - a VBlank without a new image can re-present or hold the last one without stalling,
//   - the host may present more often than the guest updates (VRR, 120/144 Hz),
//   - a late input hook can run just before the decision.
// Nothing here touches guest state: the scheduler only owns stamps and counters. Pure C++, no Vulkan.
#pragma once

#include <cstdint>

namespace coregs::present {

// Display phase a composed image belongs to (chapter 12.4: field parity travels with the image).
enum class Phase : std::uint8_t { Progressive, FieldEven, FieldOdd };

// One frame end handed to the scheduler. `hasImage` reports whether the guest display produced a
// picture; without one the presenter shows black (chapter 12.4), so the stamp is still submitted. The
// scheduler never invents frame ids, and a held image is always identifiable in a log.
struct FrameStamp {
    std::uint64_t guestFrame = 0;
    Phase phase = Phase::Progressive;
    bool hasImage = false;
};

enum class Action : std::uint8_t {
    none,        // nothing to show (no image yet, or holding disabled and no new frame)
    presentNew,  // the image submitted since the last tick
    presentHeld  // the previous image shown again
};

// Called immediately before a present decision, so input sampling can happen as late as possible
// (roadmap requirement 5). The hook must not block.
using LateInputHook = void (*)(void *context);

struct Policy {
    // Re-present the last image when no new one arrived. False = hold the display (no present call).
    bool hold_last_image = true;
};

class Scheduler {
public:
    void SetPolicy(const Policy &policy);
    const Policy &GetPolicy() const { return policy_; }

    void SetLateInputHook(LateInputHook hook, void *context);
    // Runs the hook if set; the scheduler calls it from Tick() before deciding.
    void RunLateInput();

    // A frame end (with or without a picture). Submitting again before a tick replaces the pending
    // stamp and counts a drop, so a slow host never builds a queue. Once presented, the stamp describes
    // what the display holds, which is what a hold re-presents.
    void SubmitFrame(const FrameStamp &stamp);

    // One present opportunity. Presents at most `presents_per_tick` images, newest first.
    Action Tick();

    // Diagnostics.
    std::uint64_t presentedNew() const { return presentedNew_; }
    std::uint64_t presentedHeld() const { return presentedHeld_; }
    std::uint64_t dropped() const { return dropped_; }
    std::uint64_t ticks() const { return ticks_; }
    bool hasImage() const { return shown_.hasImage; }
    const FrameStamp &lastShown() const { return shown_; }
    const FrameStamp &pending() const { return pending_; }
    bool pendingFrame() const { return pendingFrame_; }

    // Shutdown/resize: forget the pending frame but keep the counters.
    void Reset();

private:
    Policy policy_{};
    LateInputHook hook_ = nullptr;
    void *hookContext_ = nullptr;
    FrameStamp pending_{};
    bool pendingFrame_ = false;
    FrameStamp shown_{};  // what the display currently holds
    std::uint64_t presentedNew_ = 0;
    std::uint64_t presentedHeld_ = 0;
    std::uint64_t dropped_ = 0;
    std::uint64_t ticks_ = 0;
};

}  // namespace coregs::present
