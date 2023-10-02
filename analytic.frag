#version 450
#extension GL_EXT_scalar_block_layout : enable

layout(location = 0) in vec3 inPos;
layout(location = 0) out vec4 outColor;

struct Line {
    vec4 l0;
    vec4 l1;
};

layout(binding = 0) uniform UBO {
    mat4 model;
    mat4 view;
    mat4 proj;
    Line light;
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

void main() {
    float I = 2.0;
    vec3 ambient = vec3(0.1);

    vec3 pos = to_world(inPos);
    vec3 l0 = to_world(light.l0);
    vec3 l1 = to_world(light.l1);

    // for (int i = 0; i < indices.length(); i += 3) {
    //     vec3 v0 = to_world(verts[indices[i]]);
    //     vec3 v1 = to_world(verts[indices[i+1]]);
    //     vec3 v2 = to_world(verts[indices[i+2]]);
    // }

    // float irr = sample_line_light(pos, vec3(0.0,-1.0,0.0), l0, l1, I);
    float irr = sample_line_light_analytic(pos, vec3(0.0,-1.0,0.0), l0, l1, I);
    vec3 color = irr * vec3(1.0,1.0,1.0) + ambient;

    outColor = vec4(color,1.0);
}