#version 450
#extension GL_EXT_scalar_block_layout : enable

layout(location = 0) in vec3 inPos;
layout(location = 1) in vec3 inNormal;
layout(location = 2) flat in int vertIndex;
layout(location = 0) out vec4 outColor;

const int BBOX_COUNT = 4;
struct AccelStruct {
    vec3 bbox_size;
    vec3 bbox_origins[BBOX_COUNT];
    uint sizes[BBOX_COUNT];
};

layout(scalar, binding = 0) uniform UBO {
    mat4 model;
    mat4 view;
    mat4 proj;
    vec4 l0_ubo;
    vec4 l1_ubo;
    AccelStruct accel_struct;
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

uint rng_state;
float rand_pcg() {
    uint state = rng_state;
    rng_state = rng_state * 747796405u + 2891336453u;
    uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    uint n = (word >> 22u) ^ word;
    return float(n) * (1.0/float(0xffffffffU));
}


struct ProjNormalVals {
    vec2 n_xy;
    vec2 n_yz;
    vec2 n_zx;
    float d_xy;
    float d_yz;
    float d_zx;
};
float _step(float x) {
    return 2.0*step(0.0, x) - 1.0;
}
ProjNormalVals projected_normal_vals(vec3 edge, vec3 vert, vec3 n, vec3 d_p) {
    ProjNormalVals pn;
    pn.n_xy = vec2(-edge.y, edge.x) * _step(n.z);
    pn.n_yz = vec2(-edge.z, edge.y) * _step(n.x);
    pn.n_zx = vec2(-edge.x, edge.z) * _step(n.y);
    pn.d_xy = -dot(pn.n_xy, vert.xy) + max(0.0, d_p.x * pn.n_xy.x) + max(0.0, d_p.y * pn.n_xy.y);
    pn.d_yz = -dot(pn.n_yz, vert.yz) + max(0.0, d_p.y * pn.n_yz.x) + max(0.0, d_p.z * pn.n_yz.y);
    pn.d_zx = -dot(pn.n_zx, vert.zx) + max(0.0, d_p.z * pn.n_zx.x) + max(0.0, d_p.x * pn.n_zx.y);
    return pn;
}
struct PrecomputeVals {
    float d1;
    float d2;
    vec3 d_p;
    vec3 n;
    vec3 tri_bbox_min;
    vec3 tri_bbox_max;
    ProjNormalVals pn_0;
    ProjNormalVals pn_1;
    ProjNormalVals pn_2;
};
PrecomputeVals tri_aabb_precompute(vec3 v0, vec3 v1, vec3 v2, vec3 d_p) {
    PrecomputeVals pc;
    pc.d_p = d_p;
    pc.tri_bbox_min = vec3(
        min(v0.x, min(v1.x, v2.x)),
        min(v0.y, min(v1.y, v2.y)),
        min(v0.z, min(v1.z, v2.z))
    );
    pc.tri_bbox_max = vec3(
        max(v0.x, max(v1.x, v2.x)),
        max(v0.y, max(v1.y, v2.y)),
        max(v0.z, max(v1.z, v2.z))
    );
    vec3 e0 = v1 - v0;
    vec3 e1 = v2 - v1;
    vec3 e2 = v0 - v2;
    pc.n = cross(e0,e1);
    vec3 c = vec3(d_p.x * step(0.0, pc.n.x), d_p.y * step(0.0, pc.n.y), d_p.z * step(0.0, pc.n.z));
    pc.d1 = dot(pc.n, c - v0);
    pc.d2 = dot(pc.n, (d_p - c) - v0);
    pc.pn_0 = projected_normal_vals(e0, v0, pc.n, d_p);
    pc.pn_1 = projected_normal_vals(e1, v1, pc.n, d_p);
    pc.pn_2 = projected_normal_vals(e2, v2, pc.n, d_p);
    return pc;
}

bool projected_normal_check(ProjNormalVals pn, vec3 pos) {
    if (dot(pn.n_xy, pos.xy) + pn.d_xy < 0.0) return false;
    if (dot(pn.n_yz, pos.yz) + pn.d_yz < 0.0) return false;
    if (dot(pn.n_zx, pos.zx) + pn.d_zx < 0.0) return false;
    return true;
}
bool interval_overlaps(float x1, float x2, float y1, float y2) {
    return !(x1 >= y2 || y1 >= x2);
}

bool tri_aabb_intersect(PrecomputeVals pc, vec3 bbox_pos) {
    bool bbox_intersects =
        interval_overlaps(pc.tri_bbox_min.x, pc.tri_bbox_max.x, bbox_pos.x, bbox_pos.x + pc.d_p.x) &&
        interval_overlaps(pc.tri_bbox_min.y, pc.tri_bbox_max.y, bbox_pos.y, bbox_pos.y + pc.d_p.y) &&
        interval_overlaps(pc.tri_bbox_min.z, pc.tri_bbox_max.z, bbox_pos.z, bbox_pos.z + pc.d_p.z);
    if (!bbox_intersects) return false;
    if ((dot(pc.n, bbox_pos) + pc.d1) * (dot(pc.n, bbox_pos) + pc.d2) > 0.0) return false;
    return projected_normal_check(pc.pn_0, bbox_pos) &&
           projected_normal_check(pc.pn_1, bbox_pos) &&
           projected_normal_check(pc.pn_2, bbox_pos);
}


void main() {
    // rng_state = vertIndex;
    // float k = (rand_pcg() - 0.5) * 0.2;
    // vec3 color = test() ? vec3(1.0,0.2,0.2) + vec3(k) : vec3(0.3) + vec3(k);

    vec3 color = vec3(0.1,0.1,0.1);
    vec3[BBOX_COUNT] colors = {
        vec3(1.0,0.0,0.0),
        vec3(1.0,1.0,0.0),
        vec3(0.0,1.0,0.0),
        vec3(0.0,0.0,1.0),
    };

    vec3 l0 = vec3(-0.1,20.0,0.0);
    vec3 l1 = vec3(0.1,20.0,0.0);
    vec3 pos = inPos;

    PrecomputeVals precompute = tri_aabb_precompute(l0, l1, pos, accel_struct.bbox_size);
    for (int i=0; i<BBOX_COUNT; i++) {
        if (tri_aabb_intersect(precompute, accel_struct.bbox_origins[i])) {
            color = colors[i];
        }
    }
    outColor = vec4(color,1.0);
}