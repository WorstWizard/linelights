#version 450
#extension GL_EXT_scalar_block_layout : enable

layout(location = 0) in vec3 inPos;
layout(location = 0) out vec4 outColor;

layout(binding = 0) uniform UBO {
    mat4 model;
    mat4 view;
    mat4 proj;
    vec4 l0_ubo;
    vec4 l1_ubo;
};

layout(scalar, binding = 2) readonly buffer vertexBuffer {
    vec3 verts[];
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

    // TODO: C, D and E can be computed on a per-light basis, rather than per-fragment
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
    vec3 dir = cross(n1,n2);

    // Simplified projection onto line
    // Pick out largest component of d
    float max_comp = abs(dir.x);
    int i = 0;
    if (abs(dir.y) > max_comp) { max_comp = abs(dir.y); i = 1; }
    if (abs(dir.z) > max_comp) { max_comp = abs(dir.z); i = 2; }

    float vp0 = v0[i];
    float vp1 = v1[i];
    float vp2 = v2[i];

    float up0 = u0[i];
    float up1 = u1[i];
    float up2 = u2[i];

    // // Compute intervals
    float isect0_0, isect0_1, isect1_0, isect1_1;
    vec3 isect_pt_A0, isect_pt_A1, isect_pt_B0, isect_pt_B1;
    compute_intervals_isectline(
        v0, v1, v2, vp0, vp1, vp2,
        dv0, dv1, dv2, dv0dv1, dv0dv2,
        isect0_0, isect0_1, isect_pt_A0, isect_pt_A1
    );
    compute_intervals_isectline(
        u0, u1, u2, up0, up1, up2,
        du0, du1, du2, du0du1, du0du2,
        isect1_0, isect1_1, isect_pt_B0, isect_pt_B1
    );

    bool smallest0 = sort2(isect0_0, isect0_1);
    bool smallest1 = sort2(isect1_0, isect1_1);

    if (isect0_1 < isect1_0 || isect1_1 < isect0_0) return false;

    if (isect1_0 < isect0_0) {
        if (smallest0) out0 = isect_pt_A0; else out0 = isect_pt_A1;
        if (isect1_1 < isect0_1) {
            if (smallest1) out1 = isect_pt_B1; else out1 = isect_pt_B0;
        } else {
            if (smallest0) out1 = isect_pt_A1; else out1 = isect_pt_A0;
        }
    } else {
        if (smallest1) out0 = isect_pt_B0; else out1 = isect_pt_B1;
        if (isect1_1 > isect0_1) {
            if (smallest0) out1 = isect_pt_A1; else out1 = isect_pt_A0;
        } else {
            if (smallest1) out1 = isect_pt_B1; else out1 = isect_pt_B0;
        }
    }

    return true;
}






void main() {
    float I = 1.0;
    vec3 ambient = vec3(0.1);

    vec3 pos = to_world(inPos);
    vec3 l0 = to_world(l0_ubo);
    vec3 l1 = to_world(l1_ubo);

    // float irr = sample_line_light(pos, vec3(0.0,-1.0,0.0), l0, l1, I);
    // float irr = sample_line_light_analytic(pos, vec3(0.0,-1.0,0.0), l0, l1, I);

    vec3 v0 = to_world(verts[4]);
    vec3 v1 = to_world(verts[5]);
    vec3 v2 = to_world(verts[6]);

    vec3 is0, is1;
    float irr = 1.0;
    vec3 color;
    if (tri_tri_intersect(pos + vec3(-0.01, -0.01, 0.0), l0, l1, v0, v1, v2, is0, is1)) {
        // irr = 1.0 - clamp(length(is1 - is0), 0.0, 1.0);
        color = vec3(0.5,0.0,0.0);
        // irr = -1.0;
    } else {
        irr = sample_line_light_analytic(pos, vec3(0.0,-1.0,0.0), l0, l1, I);
        color = irr * vec3(1.0);
        // irr *= 10.0;
        // irr = irr - fract(irr);
        // irr /= 10.0;
    }

    // vec3 color = irr * vec3(1.0,1.0,1.0) + ambient;

    outColor = vec4(color,1.0);
}