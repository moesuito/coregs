// MIT License. Copyright (c) 2026 João Alano and GT4Recomp contributors.
// TASK-135: the performance HUD's panel, premultiplied pixels drawn on the CPU (hud/hw_hud.cpp).
#version 450

layout(set = 0, binding = 0) uniform sampler2D panel;

layout(location = 0) in vec2 panel_uv;
layout(location = 0) out vec4 colour;

void main() {
  colour = texture(panel, panel_uv);
}
