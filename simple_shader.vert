#version 450

precision mediump float;
layout(location = 0) in vec3 pos;

layout(binding = 0) uniform UBO {
    mat4 model;
    mat4 view;
    mat4 proj;
    vec3 l0;
    vec3 l1;
} ubo;

void main() {
    gl_Position = ubo.proj * ubo.view * ubo.model * vec4(pos, 1.0);
}