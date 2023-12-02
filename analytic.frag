#version 450
#extension GL_EXT_scalar_block_layout : enable

layout(location = 0) in vec3 inPos;
layout(location = 1) in vec3 inNormal;
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
    uint acceleration_indices[];
};

vec3 to_world(vec3 v) {
    return (model*vec4(v,1.0)).xyz;
}
vec3 to_world(vec4 v) {
    return (model*v).xyz;
}

float sample_line_light_analytic(vec3 pos, vec3 n, vec3 l0, vec3 l1, float I) {
    float A, B, C, D, E;
    vec3 ld = l1 - l0;
    l0 = l0 - pos;

    A = dot(l0, n);
    B = dot(ld, n);

    C = dot(l0, l0);
    D = dot(ld, l0)*2.0;
    E = dot(ld, ld);

    float sqr_C = sqrt(C);
    float sqr_CDE = sqrt(C+D+E);

    float t1, t2, t3, t4, t5;
    t1 = 2.0*D*sqr_C*(A-B);
    t2 = 4.0*A*E*sqr_C;
    t3 = 4.0*B*C*(sqr_CDE - sqr_C);
    t4 = 2.0*A*D*sqr_CDE;
    t5 = sqr_CDE*(4.0*pow(C,1.5)*E - D*D*sqr_C);

    return I * (t1 + t2 + t3 - t4) / t5;
}

void sort(inout float a, inout float b) {
    if (a > b) {
        float c = a;
        a = b;
        b = c;
    }
}
float projected_sqr_dist_to_line(vec3 l0, vec3 l1, vec3 p) {
    float x0, x1, x2, z0, z1, z2;
    x0 = p.x; x1 = l0.x; x2 = l1.x;
    z0 = p.z; z1 = l0.z; z2 = l1.z;
    return abs( (x2-x1)*(z1-z0) - (z2-z1)*(x1-x0) );
}
bool linesegments_intersect(vec2 p1, vec2 p2, vec2 p3, vec2 p4) {
    float a = p3.x-p4.x;
    float b = p1.x-p3.x;
    float c = p3.y-p4.y;
    float d = p1.y-p3.y;
    float e = p1.x-p2.x;
    float f = p1.y-p2.y;
    float t = (b*c - d*a)/(e*c - f*a);
    float u = (b*f - d*e)/(e*c - f*a);
    if (t >= 0.0 && t <= 1.0 && u >= 0.0 && u <= 1.0) return true;
    return false;
}
// Computes t-value of intersection along line-segment p1-->p2
float line_line_intersect_2d(vec2 p1, vec2 p2, vec2 p3, vec2 p4) {
    float dx = p3.x-p4.x;
    float dy = p3.y-p4.y;
    float top = (p1.x-p3.x)*dy - (p1.y-p3.y)*dx;
    float bot = (p1.x-p2.x)*dy - (p1.y-p2.y)*dx;
    return top/bot;
}
// Plane given by normal n and point p0, line by direction l and point l0
void line_plane_intersect(vec3 n, vec3 p0, vec3 l, vec3 l0, out vec3 isect) {
    float d = dot(p0 - l0, n) / dot(l, n);
    isect = l0 + d*l;
}
vec2 compute_intervals_custom(
    vec3 l0,
    vec3 l1,
    vec3 pos,
    vec3 v0,
    vec3 v1,
    vec3 v2,
    vec3 n, // Normal vector for plane defined by l0,l1,pos
    float dd1, // Product of signed distances of v0 and v1 to triangle l0,l1,pos
    float dd2 // Product of signed distances of v0 and v2 to triangle l0,l1,pos
) {
    // Compute intersection points between triangle v0-v1-v2 and plane defined by dot(p - pos, n) = 0
    vec3 isect0, isect1;
    if (dd1 < 0.0) { // Line v1-v0 crosses plane
        line_plane_intersect(n, pos, v1 - v0, v0, isect0);
        if (dd2 < 0.0) { // Line v2-v0 crosses plane
            line_plane_intersect(n, pos, v2 - v0, v0, isect1);
        } else {
            line_plane_intersect(n, pos, v2 - v1, v1, isect1);
        }
    } else { // Lines v1-v0 does not cross plane, the others do
        line_plane_intersect(n, pos, v2 - v0, v0, isect0);
        line_plane_intersect(n, pos, v2 - v1, v1, isect1);
    }

    // Project intersections onto line t*(l1-l0) + l0 by computation of t-values
    float t0, t1, tmp1, tmp2;

    // It may occur that the intersection points are further away from the light than
    // the sampled point, in which case there is no occlusion
    float dp = projected_sqr_dist_to_line(l0, l1, pos);
    float di0 = projected_sqr_dist_to_line(l0, l1, isect0);
    float di1 = projected_sqr_dist_to_line(l0, l1, isect1);
    if (di0 > dp && di1 > dp) return vec2(2.0, 2.0); // arbitrary non-occluding interval

    t0 = line_line_intersect_2d(l0.xz,l1.xz,isect0.xz,pos.xz);
    t1 = line_line_intersect_2d(l0.xz,l1.xz,isect1.xz,pos.xz);

    if (di0 < dp && di1 < dp) { // Most common case, t-values are already good
        sort(t0, t1);
        return vec2(t0, t1);
    }

    // If one intersection is further away from the line than the sampled point,
    // its corresponding t-value should be at infinity
    const float INF = 1e10;
    // Let t1 correspond to the point closer than pos, t0 the more distant point
    // Ergo, t0 will be put at +/- infinity, while t1 is kept
    if (di1 >= dp) t1 = t0;
    
    bool intersects_left = linesegments_intersect(l0.xz,pos.xz,isect0.xz,isect1.xz);
    bool intersects_right = linesegments_intersect(l1.xz,pos.xz,isect0.xz,isect1.xz);
    if (intersects_left) {
        if (intersects_right) { // Both
            return vec2(-1.0, 2.0);
        } else { // Only left
            return vec2(-INF, t1);
        }
    } else if (intersects_right) { // Only right
        return vec2(t1, INF);
    } else {
        return vec2(2.0, 2.0);
    }
}


bool tri_tri_intersect_custom(
    vec3 l0,
    vec3 l1,
    vec3 pos,
    vec3 v0,
    vec3 v1,
    vec3 v2,
    out vec2 interval
) {
    // Plane equation for occluding triangle: dot(n, x) + d = 0
    vec3 e0 = v1 - v0;
    vec3 e1 = v2 - v0;
    vec3 n = cross(e0, e1);
    float d = -dot(n, v0);

    // Put light triangle into plane equation
    float d_l0 = dot(n, l0) + d;
    float d_l1 = dot(n, l1) + d;
    float d_pos = dot(n, pos) + d;

    // Same sign on all means they're on same side of plane
    if (d_l0*d_l1 > 0.0 && d_l0*d_pos > 0.0) return false;

    // Plane equation for light triangle: dot(n, x) + d = 0
    vec3 L = l1 - l0;
    e1 = pos - l0;
    n = cross(L, e1);
    d = -dot(n, l0);

    // Put triangle 1 into plane equation 2
    float dv0 = dot(n, v0) + d;
    float dv1 = dot(n, v1) + d;
    float dv2 = dot(n, v2) + d;

    float ddv1 = dv0*dv1;
    float ddv2 = dv0*dv2;

    if (ddv1 > 0.0 && ddv2 > 0.0) return false;

    interval = compute_intervals_custom(l0, l1, pos, v0, v1, v2, n, ddv1, ddv2);
    if (interval[0] > 1.0 || interval[1] < 0.0) {
        return false;
    }
    return true;
}

// Records info on which parts of a linelight is visible as an array of intervals (t-values in [0,1])
const int ARR_MAX = 32;
struct IntervalArray {
    int size;
    vec2[ARR_MAX] data;
};
// No bounds checking for speed, just don't make mistakes ;)
void remove_interval(inout IntervalArray int_arr, int i) {
    vec2 last_interval = int_arr.data[int_arr.size - 1];
    int_arr.data[i] = last_interval;
    int_arr.size--;
}
void add_interval(inout IntervalArray int_arr, vec2 new_interval) {
    if (int_arr.size < ARR_MAX) { // Avoid overflow
        int_arr.data[int_arr.size] = new_interval;
        int_arr.size++;
    }
}
// Given an interval of occlusion, update the array to reflect the new visible intervals
void occlude_intervals(inout IntervalArray int_arr, vec2 occ_int) {
    for (int i=0; i < int_arr.size; i++) {
        vec2 interval = int_arr.data[i];
        
        if (occ_int.x <= interval.x && occ_int.y >= interval.y) {
            // Interval is fully occluded, remove it but do not increment `i`
            // as the swapped-in element needs to be checked too
            remove_interval(int_arr, i);
            i--;
        } else if (occ_int.x > interval.x && occ_int.y < interval.y) {
            // Middle is occluded, shrink existing to the left and add new interval to the right
            add_interval(int_arr, vec2(occ_int.y, interval.y));
            int_arr.data[i].y = occ_int.x;
        } else if (occ_int.x > interval.x && occ_int.x < interval.y) {
            // Right side is occluded, shrink to fit
            int_arr.data[i].y = occ_int.x;
        } else if (occ_int.y > interval.x && occ_int.y < interval.y) {
            // Left side is occluded, shrink to fit
            int_arr.data[i].x = occ_int.y;
        }
    }
}

// 2D hash with good performance, as per https://www.shadertoy.com/view/4tXyWN, recommended/tested in [Jarzynski 2020]
float noise() {
    uvec2 x = uvec2(uint(gl_FragCoord.x),uint(gl_FragCoord.y));
    uvec2 q = 1103515245U * ( (x>>1U) ^ (x.yx   ) );
    uint  n = 1103515245U * ( (q.x  ) ^ (q.y>>3U) );
    return float(n) * (1.0/float(0xffffffffU));
}

// Simple heatmap from -1.0 to +1.0
vec3 heatmap(float t) {
    vec3 blue = vec3(0.0,0.0,1.0);
    vec3 red = vec3(1.0,0.0,0.0);
    vec3 white = vec3(1.0);
    if (t > 0.0)
        return mix(red,white,t);
    return mix(red,blue,-t);
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
    float I = 5.0;
    vec3 ambient = vec3(0.1);

    vec3 pos = to_world(inPos);
    vec3 n = normalize(to_world(inNormal));

    vec3 l0 = to_world(l0_ubo);
    vec3 l1 = to_world(l1_ubo);
    vec3 L = l1 - l0;

    // Precompute AABB grid intermediate values for fast intersection
    PrecomputeVals precompute = tri_aabb_precompute(l0_ubo.xyz, l1_ubo.xyz, inPos, accel_struct.bbox_size);

    // Initialize interval array
    IntervalArray int_arr;
    int_arr.size = 0;
    add_interval(int_arr, vec2(0.0,1.0));

    // For each bounding box, test first if it intersects the light-triangle at all
    int buffer_offset = 0;
    bool early_out = false;
    for (int bbox_i=0; bbox_i<BBOX_COUNT; bbox_i++) {
        // If it does, iterate over triangles inside the bounding box
        uint num_bbox_indices = accel_struct.sizes[bbox_i];
        // if (true) {
        if (tri_aabb_intersect(precompute, accel_struct.bbox_origins[bbox_i])) {
            if (early_out) break;


            // For each triangle, compute whether it could occlude the linelight, if so, update intervals
            for (int i = buffer_offset; i < buffer_offset+num_bbox_indices; i += 3) {
                if (int_arr.size == 0) early_out = true; // Early stop
                if (early_out) break;

                vec3 v0 = to_world(verts[acceleration_indices[i]].pos);
                vec3 v1 = to_world(verts[acceleration_indices[i+1]].pos);
                vec3 v2 = to_world(verts[acceleration_indices[i+2]].pos);

                vec2 interval;
                if (tri_tri_intersect_custom(l0,l1,pos+0.001*n, v0,v1,v2, interval)) {
                    occlude_intervals(int_arr, interval);
                }
            }


        }
        buffer_offset += int(num_bbox_indices);
    }
    


    float irr = 0.0;
    for (int i = 0; i < int_arr.size; i++) {
        vec2 interval = int_arr.data[i];
        vec3 p0 = L * interval.x + l0;
        vec3 p1 = L * interval.y + l0;
        float fraction_of_light = I * (interval.y - interval.x);
        irr += sample_line_light_analytic(pos, n, p0, p1, fraction_of_light);
    }
    // vec3 color = heatmap(int_arr.size / 4.0 - 1.0);
    vec3 color = 1.0 - exp(-irr * vec3(1.0) - ambient);

    // Fix color banding by adding noise: https://pixelmager.github.io/linelight/banding.html
    color += noise() / 255.0;
    outColor = vec4(color,1.0);
}