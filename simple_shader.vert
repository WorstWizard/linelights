#version 450

layout(location = 0) in vec3 inPos;
layout(location = 0) out vec3 outPos;

layout(binding = 0) uniform UBO {
    mat4 model;
    mat4 view;
    mat4 proj;
    vec4 l0;
    vec4 l1;
} ubo;


void main() {
    outPos = inPos;
    gl_Position = ubo.proj * ubo.view * ubo.model * vec4(inPos, 1.0);
}