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

struct Ray {
    vec3 origin;
    vec3 direction;
    float t_max;
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
    
    if (t > EPS && t < r.t_max) return true;

    return false;
}

vec3 to_world(vec3 v) {
    return (model*vec4(v,1.0)).xyz;
}
vec3 to_world(vec4 v) {
    return (model*v).xyz;
}





// Triangle-triangle intersection per Möller 97
// https://fileadmin.cs.lth.se/cs/Personal/Tomas_Akenine-Moller/code/opttritri.txt
bool compute_intervals(
    float vp0,
    float vp1,
    float vp2,
    float d0,
    float d1,
    float d2,
    float d0d1,
    float d0d2,
    out float a,
    out float b,
    out float c,
    out float x0,
    out float x1
) {
    if (d0d1 > 0.0) {
        a = vp2; b = (vp0-vp2)*d2; c = (vp1-vp2)*d2; x0 = d2-d0; x1 = d2-d1;
    } else if (d0d2 > 0.0) {
        a = vp1; b = (vp0-vp1)*d1; c = (vp2-vp1)*d1; x0 = d1-d0; x1 = d1-d2;
    } else if (d1*d2 > 0.0 || d0 != 0.0) {
        a = vp0; b = (vp1-vp0)*d0; c = (vp2-vp0)*d0; x0 = d0-d1; x1 = d0-d2;
    } else if (d1 != 0.0) {
        a = vp1; b = (vp0-vp1)*d1; c = (vp2-vp1)*d1; x0 = d1-d0; x1 = d1-d2;
    } else if (d2 != 0.0) {
        a = vp2; b = (vp0-vp2)*d2; c = (vp1-vp2)*d2; x0 = d2-d0; x1 = d2-d1;
    } else {
        //Triangles are coplanar, not a case I want to handle, do nothing in this case
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

bool tri_tri_intersect(
    vec3 v0, vec3 v1, vec3 v2,
    vec3 u0, vec3 u1, vec3 u2
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

    // // Compute interval for triangle 1
    float a, b, c, d, e, f, x0, x1, y0, y1;
    compute_intervals(vp0, vp1, vp2, dv0, dv1, dv2, dv0dv1, dv0dv2, a, b, c, x0, x1);
    compute_intervals(up0, up1, up2, du0, du1, du2, du0du1, du0du2, d, e, f, y0, y1);

    float xx, yy, xxyy, tmp;
    xx = x0*x1;
    yy = y0*y1;
    xxyy = xx*yy;

    tmp = a*xxyy;
    float isect1_0 = tmp + b*x1*yy;
    float isect1_1 = tmp + c*x0*yy;

    tmp = d*xxyy;
    float isect2_0 = tmp + e*xx*y1;
    float isect2_1 = tmp + f*xx*y0;

    sort(isect1_0, isect1_1);
    sort(isect2_0, isect2_1);

    if (isect1_1 < isect2_0 || isect2_1 < isect1_0) return false;

    return true;
}









void main() {
    vec3 pos = to_world(inPos);
    vec3 l0 = to_world(l0);
    vec3 l1 = to_world(l1);

    // float d0 = distance(pos, l0_pos);
    // float d1 = distance(pos, l1_pos);

    // float irr_0 = 0.5;
    // float irr_1 = 0.5;

    // Ray ray0; ray0.origin = pos; ray0.direction = normalize(l0_pos - pos); ray0.t_max = d0;
    // Ray ray1; ray1.origin = pos; ray1.direction = normalize(l1_pos - pos); ray1.t_max = d1;

    float irr = 1.0;

    for (int i = 0; i < indices.length(); i += 3) {
        vec3 v0 = to_world(verts[indices[i]]);
        vec3 v1 = to_world(verts[indices[i+1]]);
        vec3 v2 = to_world(verts[indices[i+2]]);

        // if ( ray_triangle_intersect(ray0, v0, v1, v2) ) irr_0 = 0.0;
        // if ( ray_triangle_intersect(ray1, v0, v1, v2) ) irr_1 = 0.0;
        // Light triangle: pos_l0_l1 
        if (tri_tri_intersect(pos + vec3(-0.01, -0.01, 0.0), l0, l1, v0, v1, v2)) irr = 0.2;
    }
    // float irr = irr_0 + irr_1;

    // Upright triangle
    // vec3 v0 = to_world(verts[4]);
    // vec3 v1 = to_world(verts[5]);
    // vec3 v2 = to_world(verts[6]);

    // vec3 color = tri_tri_intersect(pos + vec3(-0.01, -0.01, 0.0), l0, l1, v0, v1, v2)/2.0 + vec3(0.5);
    vec3 color = vec3(1.0,0.0,0.0) * irr + vec3(0.0);

    outColor = vec4(color,1.0);
}