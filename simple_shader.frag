#version 450

precision mediump float;

layout(binding = 0) uniform UBO {
    mat4 model;
    mat4 view;
    mat4 proj;
    vec3 l0;
    vec3 l1;
} ubo;

layout(location = 0) out vec4 outColor;

void main() {
    outColor = vec4(1.0,1.0,1.0,1.0);
}