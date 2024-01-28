#version 450
#extension GL_EXT_scalar_block_layout : enable

layout(location = 0) in vec3 inPos;
layout(location = 1) in vec3 inNormal;
layout(location = 0) out vec4 outColor;

layout(scalar, binding = 0) uniform UBO {
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
    uint acceleration_indices[];
};

vec3 to_world(vec3 v) {
    return (model*vec4(v,1.0)).xyz;
}
vec3 to_world(vec4 v) {
    return (model*v).xyz;
}

const int GRID_SIZE = 4;
const int BBOX_COUNT = GRID_SIZE*GRID_SIZE*GRID_SIZE;

struct BufferView {
    int offset;
    int size;
};
struct BLAS {
    // uvec2 mask;
    BufferView buffer_views[BBOX_COUNT];
};
struct TLAS {
    vec3 size;
    vec3 origin;
    BLAS subgrids[BBOX_COUNT];
};

layout(scalar, binding = 1) uniform accelerationStructure {
    TLAS accel_struct;
};


// Simple heatmap from 0.0 to +1.0
vec3 heatmap(float t) {
    vec3 dark = vec3(0.0,0.0,50.0/255.0);
    vec3 mid = vec3(1.0,80.0/255.0,0.0);
    vec3 bright = vec3(1.0);
    if (t > 0.5)
        return mix(mid,bright,2.0*t - 1.0);
    return mix(dark,mid,2.0*t);
}

// 2D hash with good performance, as per https://www.shadertoy.com/view/4tXyWN, recommended/tested in [Jarzynski 2020]
float noise() {
    uvec2 x = uvec2(uint(gl_FragCoord.x),uint(gl_FragCoord.y));
    uvec2 q = 1103515245U * ( (x>>1U) ^ (x.yx   ) );
    uint  n = 1103515245U * ( (q.x  ) ^ (q.y>>3U) );
    return float(n) * (1.0/float(0xffffffffU));
}