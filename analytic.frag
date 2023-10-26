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

// Completely analytical methods for computing diffuse light from line
// light exist [K.P. Picott 92], but the integral is complicated, will implement later
// For now, the shading itself is computed in a stratified manner,
// while the visibility function will be computed analytically.
// Would like to implement LTC for shading later
const int LINE_SAMPLES = 16;
float sample_line_light(vec3 pos, vec3 n, vec3 l0, vec3 l1, float I) {
    float irr = 0.0;
    float t = 0.0;
    float h = 1.0/float(LINE_SAMPLES-1);

    vec3 l_dir = l1 - l0;

    for (int i = 0; i < LINE_SAMPLES; i++) {
        vec3 l = l0 + t*l_dir - pos; // Vector from pos to sample position on line light
        float sqr_dist = dot(l,l); // ||l||^2
        float clamped_cos = clamp(dot(n,l/sqrt(sqr_dist)), 0.0, 1.0);

        irr += I * clamped_cos / sqr_dist / float(LINE_SAMPLES);
        // irr += I / sqr_dist / float(LINE_SAMPLES);
        t += h;
    }
    return irr;
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

struct Line {
    vec3 l0;
    vec3 l1;
};

struct SegmentArray {
    Line[32] segments;
};


SegmentArray visible_line_segments(vec3 pos, Line light) {
    Line[32] line_arr;
    return SegmentArray( line_arr );
}


void isect2(
    vec3 v0,
    vec3 v1,
    vec3 v2,
    float vp0,
    float vp1,
    float vp2,
    float d0,
    float d1,
    float d2,
    out float isect0,
    out float isect1,
    out vec3 out0,
    out vec3 out1
) {
    float tmp = d0/(d0-d1);
    vec3 diff = v1 - v0;
    diff = diff * tmp;
    out0 = diff + v0;
    isect0 = vp0 + (vp1 - vp0) * tmp;

    tmp = d0/(d0-d2);
    diff = v2 - v0;
    diff = diff * tmp;
    out1 = diff + v0;
    isect1 = vp0 + (vp2-vp0) * tmp;
}
bool compute_intervals_isectline(
    vec3 v0,
    vec3 v1,
    vec3 v2,
    float vp0,
    float vp1,
    float vp2,
    float d0,
    float d1,
    float d2,
    float d0d1,
    float d0d2,
    out float isect0,
    out float isect1,
    out vec3 out0,
    out vec3 out1
) {
    if (d0d1 > 0.0) {
        isect2(v2, v0, v1, vp2, vp0, vp1, d2, d0, d1, isect0, isect1, out0, out1);
    } else if (d0d2 > 0.0) {
        isect2(v1, v0, v2, vp1, vp0, vp2, d1, d0, d2, isect0, isect1, out0, out1);
    } else if (d1*d2 > 0.0 || d0 != 0.0) {
        isect2(v0, v1, v2, vp0, vp1, vp2, d0, d1, d2, isect0, isect1, out0, out1);
    } else if (d1 != 0.0) {
        isect2(v1, v0, v2, vp1, vp0, vp2, d1, d0, d2, isect0, isect1, out0, out1);
    } else if (d2 != 0.0) {
        isect2(v2, v0, v1, vp2, vp0, vp1, d2, d0, d1, isect0, isect1, out0, out1);
    } else {
        //Triangles are coplanar, not a case I want to handle
        return false;
    }
    return true;
}

void sort(inout float a, inout float b) {
    if (a > b) {
        float c = a;
        a = b;
        b = c;
    }
}

bool sort2(inout float a, inout float b) {
    if (a > b) {
        float c = a;
        a = b;
        b = c;
        return false;
    }
    return true;
}

// Triangle-triangle intersection with intersection line-segment computed
// By Thomas Möller 1997
// https://fileadmin.cs.lth.se/cs/Personal/Tomas_Akenine-Moller/code/tritri_isectline.txt
//
// This version returns true if an intersection exists, false if not, or if triangles are coplanar
bool tri_tri_intersect(
    vec3 v0, vec3 v1, vec3 v2,
    vec3 u0, vec3 u1, vec3 u2,
    out vec3 out0, out vec3 out1
) {
    // Plane equation 1: dot(n1, x) + d1 = 0
    vec3 e1 = v1 - v0;
    vec3 e2 = v2 - v0;
    vec3 n1 = cross(e1, e2);
    float d1 = -dot(n1, v0);

    // Put triangle 2 into plane equation 1
    float du0 = dot(n1, u0) + d1;
    float du1 = dot(n1, u1) + d1;
    float du2 = dot(n1, u2) + d1;

    float du0du1 = du0*du1;
    float du0du2 = du0*du2;

    // Same sign on all means they're on same side of plane
    if (du0du1 > 0.0 && du0du2 > 0.0) return false;

    // Plane equation 2: dot(n2, x) + d2 = 0
    e1 = u1 - u0;
    e2 = u2 - u0;
    vec3 n2 = cross(e1,e2);
    float d2 = -dot(n2, u0);

    // Put triangle 1 into plane equation 2
    float dv0 = dot(n2, v0) + d2;
    float dv1 = dot(n2, v1) + d2;
    float dv2 = dot(n2, v2) + d2;

    float dv0dv1 = dv0*dv1;
    float dv0dv2 = dv0*dv2;

    if (dv0dv1 > 0.0 && dv0dv2 > 0.0) return false;

    // Compute intersection line direction
    // vec3 dir = cross(n1,n2);
    
    // Test: Using direction of |l1-l0| instead, which are currently args v2 and v1
    vec3 dir = normalize(v2 - v1);

    // Simplified projection onto line
    // Pick out largest component of d
    // float max_comp = abs(dir.x);
    // int i = 0;
    // if (abs(dir.y) > max_comp) { max_comp = abs(dir.y); i = 1; }
    // if (abs(dir.z) > max_comp) { max_comp = abs(dir.z); i = 2; }

    // float vp0 = v0[i];
    // float vp1 = v1[i];
    // float vp2 = v2[i];

    // float up0 = u0[i];
    // float up1 = u1[i];
    // float up2 = u2[i];

    // Test: Projection onto |l1-l0| instead
    // float vp0 = dot(dir, v0);
    // float vp1 = dot(dir, v1);
    // float vp2 = dot(dir, v2);

    // float up0 = dot(dir, u0);
    // float up1 = dot(dir, u1);
    // float up2 = dot(dir, u2);

    // // // Compute intervals
    // float isect0_0, isect0_1, isect1_0, isect1_1;
    // vec3 isect_pt_A0, isect_pt_A1, isect_pt_B0, isect_pt_B1;
    // compute_intervals_isectline(
    //     v0, v1, v2, vp0, vp1, vp2,
    //     dv0, dv1, dv2, dv0dv1, dv0dv2,
    //     isect0_0, isect0_1, isect_pt_A0, isect_pt_A1
    // );
    // compute_intervals_isectline(
    //     u0, u1, u2, up0, up1, up2,
    //     du0, du1, du2, du0du1, du0du2,
    //     isect1_0, isect1_1, isect_pt_B0, isect_pt_B1
    // );

    // float isect0_0, isect0_1, isect1_0, isect1_1;
    // // compute_intervals_custom();

    // bool smallest0 = sort2(isect0_0, isect0_1);
    // bool smallest1 = sort2(isect1_0, isect1_1);

    // if (isect0_1 < isect1_0 || isect1_1 < isect0_0) return false;

    // if (isect1_0 < isect0_0) {
    //     if (smallest0) out0 = isect_pt_A0; else out0 = isect_pt_A1;
    //     if (isect1_1 < isect0_1) {
    //         if (smallest1) out1 = isect_pt_B1; else out1 = isect_pt_B0;
    //     } else {
    //         if (smallest0) out1 = isect_pt_A1; else out1 = isect_pt_A0;
    //     }
    // } else {
    //     if (smallest1) out0 = isect_pt_B0; else out0 = isect_pt_B1;
    //     if (isect1_1 > isect0_1) {
    //         if (smallest0) out1 = isect_pt_A1; else out1 = isect_pt_A0;
    //     } else {
    //         if (smallest1) out1 = isect_pt_B1; else out1 = isect_pt_B0;
    //     }
    // }

    return true;
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

    // // Project intersections onto line t*(l1-l0) + l0 by computation of t-values
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
    float I = 5.0;
    vec3 ambient = vec3(0.0);

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
    if (tri_tri_intersect_custom(l0, l1, pos, v0, v1, v2, interval, is0, is1)) {
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
        // irr = abs(interval.y);
    } else {
        irr = sample_line_light_analytic(pos, n, l0, l1, I);
    }

    vec3 color = irr * vec3(1.0) + ambient;

    outColor = vec4(color,1.0);
}