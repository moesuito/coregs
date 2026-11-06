// MIT License. Copyright (c) 2026 Joao Alano and GT4Recomp contributors.
// Written from the TASK-026 specification (docs/hw-renderer-spec), clean room (decision 0069).
//
// Present scheduler, see hw_gs_present_scheduler.hpp. Chapter 12.5 plus the owner's
// variable-frame-rate readiness requirements (docs/COREGS2-ROADMAP.md).
#include "hw_gs_present_scheduler.hpp"

namespace coregs::present {

void Scheduler::SetPolicy(const Policy &policy) { policy_ = policy; }

void Scheduler::SetLateInputHook(LateInputHook hook, void *context) {
    hook_ = hook;
    hookContext_ = context;
}

void Scheduler::RunLateInput() {
    if (hook_) {
        hook_(hookContext_);
    }
}

void Scheduler::SubmitFrame(const FrameStamp &stamp) {
    if (pendingFrame_) {
        // A frame that never reached the display: the newest one wins.
        ++dropped_;
    }
    pending_ = stamp;
    pendingFrame_ = true;
}

Action Scheduler::Tick() {
    ++ticks_;
    RunLateInput();
    if (pendingFrame_) {
        shown_ = pending_;
        pending_ = FrameStamp{};
        pendingFrame_ = false;
        ++presentedNew_;
        return Action::presentNew;
    }
    if (policy_.hold_last_image && shown_.hasImage) {
        ++presentedHeld_;
        return Action::presentHeld;
    }
    return Action::none;
}

void Scheduler::Reset() {
    pending_ = FrameStamp{};
    pendingFrame_ = false;
    shown_ = FrameStamp{};
}

}  // namespace coregs::present
