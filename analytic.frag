#version 450
#extension GL_EXT_scalar_block_layout : enable

layout(location = 0) in vec3 inPos;
layout(location = 1) in vec3 inNormal;

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
float sqr_dist_to_line(vec3 l, vec3 p0, vec3 p) {
    vec3 k = p - p0;
    vec3 d = cross(k, l);
    return dot(d,d);
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
    vec3 L = l1 - l0;
    vec3 P = pos - l0;

    int i = 0;
    int j = 1;

    tmp1 = isect0[i]*P[j] - isect0[j]*P[i] + pos[i]*l0[j] - pos[j]*l0[i];
    tmp2 = isect0[i]*L[j] - isect0[j]*L[i] + pos[j]*L[i]  - pos[i]*L[j];
    t0 = tmp1/tmp2;

    tmp1 = isect1[i]*P[j] - isect1[j]*P[i] + pos[i]*l0[j] - pos[j]*l0[i];
    tmp2 = isect1[i]*L[j] - isect1[j]*L[i] + pos[j]*L[i]  - pos[i]*L[j];
    t1 = tmp1/tmp2;

    // If an intersection is further away from the line than the sampled point, the t-value will cross infinity
    // If both are further away, we will already have ruled out an intersection before calling this function, so this is not a concern
    float dp = sqr_dist_to_line(L, l0, pos);
    float di0 = sqr_dist_to_line(L, l0, isect0);
    float di1 = sqr_dist_to_line(L, l0, isect1);
    
    const float INF = 1e10;
    if (di0 > dp) t0 = -sign(t0) * INF;
    if (di1 > dp) t1 = -sign(t1) * INF;

    sort(t0, t1);
    return vec2(t0, t1);
}


bool tri_tri_intersect_custom(
    vec3 l0,
    vec3 l1,
    vec3 pos,
    vec3 v0,
    vec3 v1,
    vec3 v2,
    out vec2 interval,
    out vec3 out0,
    out vec3 out1
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

    out0 = L * max(interval.x, 0.0) + l0;
    out1 = L * min(interval.y, 1.0) + l0;

    return true;
}

void main() {
    float I = 3.0;
    vec3 ambient = vec3(0.1);

    vec3 pos = to_world(inPos);
    vec3 l0 = to_world(l0_ubo);
    vec3 l1 = to_world(l1_ubo);

    vec3 v0 = to_world(verts[4].pos);
    vec3 v1 = to_world(verts[5].pos);
    vec3 v2 = to_world(verts[6].pos);

    vec3 n = normalize(to_world(inNormal));

    float irr;
    vec2 interval;
    vec3 is0, is1;
    if (tri_tri_intersect_custom(l0, l1, pos + 0.001*n, v0, v1, v2, interval, is0, is1)) {
        if (interval.x < 0.0) { // Lower edge occluded
            I *= distance(is1,l1)/distance(l0,l1);
            irr = sample_line_light_analytic(pos, n, is1, l1, I);
        } else if (interval.y > 1.0) { // Upper edge occluded
            I *= distance(l0,is0)/distance(l0,l1);
            irr = sample_line_light_analytic(pos, n, l0, is0, I);
        } else { // Middle partially occluded
            float I0 = I*distance(l0,is0)/distance(l0,l1);
            float I1 = I*distance(is1,l1)/distance(l0,l1);
            irr =  sample_line_light_analytic(pos, n, l0, is0, I0);
            irr += sample_line_light_analytic(pos, n, is1, l1, I1);
        }
    } else {
        irr = sample_line_light_analytic(pos, n, l0, l1, I);
    }

    vec3 color = irr * vec3(1.0) + ambient;

    outColor = vec4(color,1.0);
}