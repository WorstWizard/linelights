#version 450
#extension GL_EXT_scalar_block_layout : enable

layout(location = 0) in vec3 inPos;
layout(location = 0) out vec4 outColor;

layout(binding = 0) uniform UBO {
    mat4 model;
    mat4 view;
    mat4 proj;
    vec4 l0;
    vec4 l1;
};

layout(scalar, binding = 2) readonly buffer vertexBuffer {
    vec3 verts[];
};
layout(binding = 3) readonly buffer indexBuffer {
    uint indices[];
};

void main() {
    // vec3 pos = to_world(inPos);
    // vec3 l0 = to_world(l0);
    // vec3 l1 = to_world(l1);

    vec3 color = vec3(1.0,0.0,0.0);

    outColor = vec4(color,1.0);
}