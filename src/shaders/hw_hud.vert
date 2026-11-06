// MIT License. Copyright (c) 2026 João Alano and GT4Recomp contributors.
// TASK-135: the performance HUD's panel, a quad placed by push constants (normalised device coordinates).
#version 450

layout(push_constant) uniform Quad {
  vec2 origin;
  vec2 size;
} quad;

layout(location = 0) out vec2 panel_uv;

void main() {
  const vec2 corner = vec2(float(gl_VertexIndex & 1), float(gl_VertexIndex >> 1));
  panel_uv = corner;
  gl_Position = vec4(quad.origin + corner * quad.size, 0.0, 1.0);
}
