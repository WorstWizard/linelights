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
float dist_to_line_2d_unnormalized(vec2 p, vec2 v0, vec2 v1) {
    return (v1.x - v0.x)*(v0.y - p.y) - (v1.y - v0.y)*(v0.x - p.x);
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
    const float INF = 1e10;

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
    float dp = dist_to_line_2d_unnormalized(l0.xz, l1.xz, pos.xz);
    float sign_of_dp = sign(dp);
    dp = abs(dp);
    float di0 = sign_of_dp * dist_to_line_2d_unnormalized(l0.xz, l1.xz, isect0.xz);
    float di1 = sign_of_dp * dist_to_line_2d_unnormalized(l0.xz, l1.xz, isect1.xz);
    if (di0 < 0.0 || di1 < 0.0 || (di0 > dp && di1 > dp)) return vec2(INF, INF); // arbitrary non-occluding interval

    t0 = line_line_intersect_2d(l0.xz,l1.xz,isect0.xz,pos.xz);
    t1 = line_line_intersect_2d(l0.xz,l1.xz,isect1.xz,pos.xz);

    if (di0 < dp && di1 < dp) { // Most common case, t-values are already good
        sort(t0, t1);
        return vec2(t0, t1);
    }

    // If one intersection is further away from the line than the sampled point,
    // its corresponding t-value should be at infinity
    // Let t1 correspond to the point closer than pos, t0 the more distant point
    // Ergo, t0 will be put at +/- infinity, while t1 is kept
    if (di1 >= dp) t1 = t0;
    
    bool intersects_left = linesegments_intersect(l0.xz,pos.xz,isect0.xz,isect1.xz);
    bool intersects_right = linesegments_intersect(l1.xz,pos.xz,isect0.xz,isect1.xz);
    if (intersects_left) {
        if (intersects_right) { // Both
            return vec2(-INF, INF);
        } else { // Only left
            return vec2(-INF, t1);
        }
    } else if (intersects_right) { // Only right
        return vec2(t1, INF);
    } else {
        return vec2(INF, INF);
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






IntervalArray intersect_scene_top_bottom(vec3 pos, vec3 n, vec3 l0, vec3 l1) {
    // Precompute AABB grid intermediate values for fast intersection
    vec3 blas_size = accel_struct.size / float(GRID_SIZE);
    vec3 bbox_size = blas_size / float(GRID_SIZE);
    PrecomputeVals blas_precompute = tri_aabb_precompute(l0_ubo.xyz, l1_ubo.xyz, inPos, blas_size);
    PrecomputeVals bbox_precompute = tri_aabb_precompute(l0_ubo.xyz, l1_ubo.xyz, inPos, bbox_size);

    // Initialize interval array
    IntervalArray int_arr;
    int_arr.size = 0;
    add_interval(int_arr, vec2(0.0,1.0));

    vec3 l0_dir = normalize(l0 - pos);
    vec3 l1_dir = normalize(l1 - pos);
    float dot_0 = dot(n, l0_dir);
    float dot_1 = dot(n, l1_dir);
    if (dot_0 < 0.0 && dot_1 < 0.0) {
        remove_interval(int_arr, 0);
        return int_arr;
    }
    float alpha = max(0.0, min(dot_0, dot_1));

    // For each bounding box, test first if it intersects the light-triangle at all
    // bool early_out = false;
    int blas_index = 0;
    for (int blas_i=0; blas_i<GRID_SIZE; blas_i++) {
    for (int blas_j=0; blas_j<GRID_SIZE; blas_j++) {
    for (int blas_k=0; blas_k<GRID_SIZE; blas_k++) {
        vec3 ijk = vec3(blas_i, blas_j, blas_k);
        vec3 blas_origin = accel_struct.origin + ijk * blas_size;

        if (tri_aabb_intersect(blas_precompute, blas_origin)) {
            // if (early_out) break;

            int bbox_index = 0;
            for (int bbox_i=0; bbox_i<GRID_SIZE; bbox_i++) {
            for (int bbox_j=0; bbox_j<GRID_SIZE; bbox_j++) {
            for (int bbox_k=0; bbox_k<GRID_SIZE; bbox_k++) {
                vec3 ijk = vec3(bbox_i, bbox_j, bbox_k);
                vec3 bbox_origin = blas_origin + ijk * bbox_size;

                if (tri_aabb_intersect(bbox_precompute, bbox_origin)) {

                    BufferView buffer_view = accel_struct.subgrids[blas_index].buffer_views[bbox_index];
                    
                    // For each triangle, compute whether it could occlude the linelight, if so, update intervals
                    for (int i = buffer_view.offset; i < buffer_view.offset+buffer_view.size; i += 3) {
                        // if (int_arr.size == 0) early_out = true; // Early stop
                        // if (early_out) break;

                        vec3 v0 = to_world(verts[acceleration_indices[i]].pos);
                        vec3 v1 = to_world(verts[acceleration_indices[i+1]].pos);
                        vec3 v2 = to_world(verts[acceleration_indices[i+2]].pos);

                        // vec3 tri_normal_0 = to_world(verts[acceleration_indices[i]].normal);
                        // vec3 tri_normal_1 = to_world(verts[acceleration_indices[i+1]].normal);
                        // vec3 tri_normal_2 = to_world(verts[acceleration_indices[i+2]].normal);
                        // vec3 mean_vertex_norm = tri_normal_0 + tri_normal_1 + tri_normal_2;
                        // vec3 face_normal = normalize(cross(v1-v0, v2-v0));


                        vec3 face_normal = verts[acceleration_indices[i]].normal;
                        float beta = 1.0 + dot(n, face_normal);

                        // if (dot(mean_vertex_norm, face_normal) < 0.0) face_normal = -face_normal;
                        if (alpha > beta) continue;

                        // float r_0 = dot(n, v0 - pos);
                        // float r_1 = dot(n, v1 - pos);
                        // float r_2 = dot(n, v2 - pos);

                        // if (r_0 < 0.0 && r_1 < 0.0 && r_2 < 0.0) {
                        //     continue;
                        // }

                        vec2 interval;
                        if (tri_tri_intersect_custom(l0,l1,pos+0.001*n, v0,v1,v2, interval)) {
                            occlude_intervals(int_arr, interval);
                        }
                    }
                }

                bbox_index++;
            }}}
        }
        blas_index++;
    }}}

    return int_arr;
}

IntervalArray intersect_scene_bottom_only(vec3 pos, vec3 n, vec3 l0, vec3 l1) {
    // Precompute AABB grid intermediate values for fast intersection
    vec3 blas_size = accel_struct.size / float(GRID_SIZE);
    vec3 bbox_size = blas_size / float(GRID_SIZE);
    PrecomputeVals blas_precompute = tri_aabb_precompute(l0_ubo.xyz, l1_ubo.xyz, inPos, blas_size);
    PrecomputeVals bbox_precompute = tri_aabb_precompute(l0_ubo.xyz, l1_ubo.xyz, inPos, bbox_size);

    // Initialize interval array
    IntervalArray int_arr;
    int_arr.size = 0;
    add_interval(int_arr, vec2(0.0,1.0));

    // For each bounding box, test first if it intersects the light-triangle at all
    // bool early_out = false;
    int blas_index = 0;
    for (int blas_i=0; blas_i<GRID_SIZE; blas_i++) {
    for (int blas_j=0; blas_j<GRID_SIZE; blas_j++) {
    for (int blas_k=0; blas_k<GRID_SIZE; blas_k++) {
        vec3 ijk = vec3(blas_i, blas_j, blas_k);
        vec3 blas_origin = accel_struct.origin + ijk * blas_size;

        if (true) {
            // if (early_out) break;

            int bbox_index = 0;
            for (int bbox_i=0; bbox_i<GRID_SIZE; bbox_i++) {
            for (int bbox_j=0; bbox_j<GRID_SIZE; bbox_j++) {
            for (int bbox_k=0; bbox_k<GRID_SIZE; bbox_k++) {
                vec3 ijk = vec3(bbox_i, bbox_j, bbox_k);
                vec3 bbox_origin = blas_origin + ijk * bbox_size;

                if (tri_aabb_intersect(bbox_precompute, bbox_origin)) {

                    BufferView buffer_view = accel_struct.subgrids[blas_index].buffer_views[bbox_index];
                    
                    // For each triangle, compute whether it could occlude the linelight, if so, update intervals
                    for (int i = buffer_view.offset; i < buffer_view.offset+buffer_view.size; i += 3) {
                        // if (int_arr.size == 0) early_out = true; // Early stop
                        // if (early_out) break;

                        vec3 v0 = to_world(verts[acceleration_indices[i]].pos);
                        vec3 v1 = to_world(verts[acceleration_indices[i+1]].pos);
                        vec3 v2 = to_world(verts[acceleration_indices[i+2]].pos);

                        vec2 interval;
                        if (tri_tri_intersect_custom(l0,l1,pos+0.001*n, v0,v1,v2, interval)) {
                            occlude_intervals(int_arr, interval);
                        }
                    }
                }

                bbox_index++;
            }}}
        }
        blas_index++;
    }}}

    return int_arr;
}

IntervalArray intersect_scene_brute(vec3 pos, vec3 n, vec3 l0, vec3 l1) {
    // Initialize interval array
    IntervalArray int_arr;
    int_arr.size = 0;
    add_interval(int_arr, vec2(0.0,1.0));

    // For each bounding box, test first if it intersects the light-triangle at all
    // bool early_out = false;
    for (int blas_i=0; blas_i<GRID_SIZE*GRID_SIZE*GRID_SIZE; blas_i++) {
    for (int bbox_i=0; bbox_i<GRID_SIZE*GRID_SIZE*GRID_SIZE; bbox_i++) {
        BufferView buffer_view = accel_struct.subgrids[blas_i].buffer_views[bbox_i];
        for (int i = buffer_view.offset; i < buffer_view.offset+buffer_view.size; i += 3) {
            vec3 v0 = to_world(verts[acceleration_indices[i]].pos);
            vec3 v1 = to_world(verts[acceleration_indices[i+1]].pos);
            vec3 v2 = to_world(verts[acceleration_indices[i+2]].pos);

            vec2 interval;
            if (tri_tri_intersect_custom(l0,l1,pos+0.001*n, v0,v1,v2, interval)) {
                occlude_intervals(int_arr, interval);
            }
        }
    }
    }

    return int_arr;
}





void main() {
    float I = 10.0;
    vec3 ambient = vec3(0.1);

    vec3 pos = to_world(inPos);
    vec3 n = normalize(to_world(inNormal));

    vec3 l0 = to_world(l0_ubo);
    vec3 l1 = to_world(l1_ubo);
    vec3 L = l1 - l0;


    // A possible optimization, but didn't seem to confer any measurable benefit
    // // If line-light intersects plane of the triangle, clamp it
    // float clamp_t = dot(pos - l0, n) / dot(L, n);
    // if (clamp_t >= 0.0 && clamp_t <= 1.0) {
    //     if (dot(l0 - pos, n) > 0.0) {
    //         I = clamp_t * I;
    //         l1 = clamp_t * L + l0;
    //     } else {
    //         I = (1.0 - clamp_t) * I;
    //         l0 = clamp_t * L + l0;
    //     }
    //     L = l1 - l0;
    // }

    // IntervalArray int_arr = intersect_scene_brute(pos, n, l0, l1);
    IntervalArray int_arr = intersect_scene_top_bottom(pos, n, l0, l1);

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