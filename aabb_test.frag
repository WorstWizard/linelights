#version 450
#extension GL_EXT_scalar_block_layout : enable

layout(location = 0) in vec3 inPos;
layout(location = 1) in vec3 inNormal;
layout(location = 2) flat in int vertIndex;
layout(location = 0) out vec4 outColor;

layout(binding = 0) uniform UBO {
    mat4 model;
    mat4 view;
    mat4 proj;
    vec4 l0_ubo;
    vec4 l1_ubo;
};

struct Vertex {
    vec3 pos;
    vec3 normal;
};
layout(scalar, binding = 2) readonly buffer vertexBuffer {
    Vertex verts[];
};
layout(binding = 3) readonly buffer indexBuffer {
    uint indices[];
};

bool test() {
    return false;
}

uint rng_state;
float rand_pcg() {
    uint state = rng_state;
    rng_state = rng_state * 747796405u + 2891336453u;
    uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    uint n = (word >> 22u) ^ word;
    return float(n) * (1.0/float(0xffffffffU));
}

void main() {
    rng_state = vertIndex;
    float k = (rand_pcg() - 0.5) * 0.2;
    vec3 color = test() ? vec3(1.0,0.2,0.2) + vec3(k) : vec3(0.3) + vec3(k);
    outColor = vec4(color,1.0);
}