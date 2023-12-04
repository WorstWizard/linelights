#version 450

layout(location = 0) in vec3 inPos;
layout(location = 1) in vec3 inNormal;

layout(location = 0) out vec3 outPos;
layout(location = 1) out vec3 outNormal;
layout(location = 2) out int vertIndex;

layout(binding = 0) uniform UBO {
    mat4 model;
    mat4 view;
    mat4 proj;
    vec4 l0;
    vec4 l1;
};

void main() {
    outPos = inPos;
    outNormal = inNormal;
    vertIndex = gl_VertexIndex;
    gl_Position = proj * view * model * vec4(inPos, 1.0);
}