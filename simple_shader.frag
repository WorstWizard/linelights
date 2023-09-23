#version 450
#extension GL_EXT_scalar_block_layout : enable

layout(location = 0) in vec3 pos;
layout(location = 0) out vec4 outColor;

layout(binding = 0) uniform UBO {
    mat4 model;
    mat4 view;
    mat4 proj;
    vec4 l0;
    vec4 l1;
} ubo;

layout(scalar, binding = 2) readonly buffer vertexBuffer {
    vec3 verts[];
};
layout(binding = 3) readonly buffer indexBuffer {
    uint indices[];
};

struct Ray {
    vec3 origin;
    vec3 direction;
};

// Møller-Trumbore intersection
bool ray_triangle_intersect(Ray r, vec3 v0, vec3 v1, vec3 v2) {
    const float EPS = 1e-4;

    vec3 edge1 = v1 - v0;
    vec3 edge2 = v2 - v0;
    vec3 h = cross(r.direction, edge2);
    float a = dot(edge1, h);

    if (abs(a) < EPS) return false;

    float f = 1.0 / a;
    vec3 s = r.origin - v0;
    float u = f * dot(s, h);

    if (u < 0.0 || u > 1.0) return false;

    vec3 q = cross(s, edge1);
    float v = f * dot(r.direction, q);

    if (v < 0.0 || (u+v) > 1.0) return false;

    float t = f * dot(edge2, q);
    
    if (t > EPS) return true;

    return false;
}

void main() {
    float I = 10.0;

    // float irr_0 = 0.5;
    // float irr_1 = 0.5;

    float irr_0 = I/pow( distance(pos, ubo.l0.xyz), 2.0);
    float irr_1 = I/pow( distance(pos, ubo.l1.xyz), 2.0);

    Ray ray0; ray0.origin = pos; ray0.direction = normalize(ubo.l0.xyz - pos);
    Ray ray1; ray1.origin = pos; ray1.direction = normalize(ubo.l1.xyz - pos);

    for (int i = 0; i < indices.length(); i += 3) {
        vec3 v0 = verts[indices[i]];
        vec3 v1 = verts[indices[i+1]];
        vec3 v2 = verts[indices[i+2]];

        if ( ray_triangle_intersect(ray0, v0, v1, v2) ) irr_0 = 0.0;
        if ( ray_triangle_intersect(ray1, v0, v1, v2) ) irr_1 = 0.0;
    }
    float irr = irr_0 + irr_1;
    vec3 color = vec3(1.0,1.0,1.0) * irr + vec3(0.1);

    // vec3 color = ray_triangle_intersect(ray0, verts[4], verts[5], verts[6]);

    outColor = vec4(color,1.0);
}