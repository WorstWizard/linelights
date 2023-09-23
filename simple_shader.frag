#version 450

layout(location = 0) in vec3 pos;
layout(location = 0) out vec4 outColor;

layout(binding = 0) uniform UBO {
    mat4 model;
    mat4 view;
    mat4 proj;
    vec4 l0;
    vec4 l1;
} ubo;

layout(std430, binding = 1) readonly buffer vertexBuffer {
    vec3 pos[];
} verts;

void main() {

    float I = 10.0;
    // float irr_0 = I/pow( distance(pos, ubo.l0.xyz), 2.0);
    // float irr_0 = 0.05;
    // float irr_1 = I/pow( distance(pos, ubo.l1.xyz), 2.0);

    float irr_0 = I/pow( distance(pos, verts.pos[0]), 2.0);
    float irr_1 = I/pow( distance(pos, verts.pos[3]), 2.0);

    float irr = irr_0 + irr_1;

    vec3 color = vec3(1.0,1.0,1.0) * irr;

    outColor = vec4(color,1.0);
}