struct Ray {
    vec3 origin;
    vec3 direction;
    float t_max;
    bool hit;
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

// https://iquilezles.org/articles/intersectors/
bool ray_aabb_intersect(Ray r, vec3 bbox_origin, vec3 bbox_size) {
    vec3 bbox_center = bbox_origin + 0.5*bbox_size;
    vec3 m = 1.0/r.direction;
    vec3 n = m*(r.origin - bbox_center);
    vec3 k = vec3(abs(m.x),abs(m.y),abs(m.z))*bbox_size*0.5;
    vec3 t1 = -n - k;
    vec3 t2 = -n + k;
    float tN = max( max( t1.x, t1.y ), t1.z );
    float tF = min( min( t2.x, t2.y ), t2.z );
    return !(tN>tF || tF<0.0);
}

// 2D hash with good performance, as per https://www.shadertoy.com/view/4tXyWN, recommended/tested in [Jarzynski 2020]
uint hash(uvec2 x) {
    uvec2 q = 1103515245U * ( (x>>1U) ^ (x.yx   ) );
    uint  n = 1103515245U * ( (q.x  ) ^ (q.y>>3U) );
    return n;
}
// PCG PRNG as per https://www.reedbeta.com/blog/hash-functions-for-gpu-rendering/
uint rng_state;
float rand_pcg() {
    uint state = rng_state;
    rng_state = rng_state * 747796405u + 2891336453u;
    uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    uint n = (word >> 22u) ^ word;
    return float(n) * (1.0/float(0xffffffffU));
}

const int NUM_SAMPLES = 4;
float sample_line_light_stochastic(vec3 pos, vec3 n, vec3 l0, vec3 l1, float I) {
    vec3 l = l1 - l0;

    // Precompute AABB grid intermediate values for fast intersection
    vec3 blas_size = accel_struct.size / float(GRID_SIZE);
    vec3 bbox_size = blas_size / float(GRID_SIZE);

    float irr = 0.0;
    for (int i = 0; i < NUM_SAMPLES; i++) {
        float t = rand_pcg();
        vec3 target = l0 + t*l;
        vec3 origin = pos + 0.001*n;
        vec3 dir = target - origin;
        float d = length(dir);

        Ray r;
        r.origin = origin;
        r.direction = dir/d;
        r.t_max = d;
        r.hit = false;

        int blas_index = 0;
        for (int blas_i=0; blas_i<GRID_SIZE; blas_i++) {
        for (int blas_j=0; blas_j<GRID_SIZE; blas_j++) {
        for (int blas_k=0; blas_k<GRID_SIZE; blas_k++) {
            if (r.hit) break;
            vec3 ijk = vec3(blas_i, blas_j, blas_k);
            vec3 blas_origin = accel_struct.origin + ijk * blas_size;

            // if (true) {
            if (ray_aabb_intersect(r, blas_origin, blas_size)) {
                // if (early_out) break;

                int bbox_index = 0;
                for (int bbox_i=0; bbox_i<GRID_SIZE; bbox_i++) {
                for (int bbox_j=0; bbox_j<GRID_SIZE; bbox_j++) {
                for (int bbox_k=0; bbox_k<GRID_SIZE; bbox_k++) {
                    vec3 ijk = vec3(bbox_i, bbox_j, bbox_k);
                    vec3 bbox_origin = blas_origin + ijk * bbox_size;

                    // if (ray_aabb_intersect(r, bbox_origin, bbox_size)) {
                    if (true) {
                        BufferView buffer_view = accel_struct.subgrids[blas_index].buffer_views[bbox_index];
                        
                        // For each triangle, compute whether it could occlude the linelight, if so, update intervals
                        for (int i = buffer_view.offset; i < buffer_view.offset+buffer_view.size; i += 3) {
                            // if (int_arr.size == 0) early_out = true; // Early stop
                            // if (early_out) break;

                            vec3 v0 = to_world(verts[acceleration_indices[i]].pos);
                            vec3 v1 = to_world(verts[acceleration_indices[i+1]].pos);
                            vec3 v2 = to_world(verts[acceleration_indices[i+2]].pos);

                            if ( ray_triangle_intersect(r, v0, v1, v2) ) {
                                r.hit = true;
                                break;
                            }
                        }
                    }
                    bbox_index++;
                }}}
            }
            blas_index++;
        }}}

        if ( !r.hit ) {
            float clamped_cos = clamp(dot(n,r.direction), 0.0, 1.0);
            irr += I * clamped_cos / (r.t_max * r.t_max) / float(NUM_SAMPLES);
        }
    }

    return irr;
}

void main() {
    rng_state = hash(uvec2(uint(gl_FragCoord.x),uint(gl_FragCoord.y)));

    float I = 5.0;
    vec3 ambient = vec3(0.1);

    vec3 pos = to_world(inPos);
    vec3 l0 = to_world(l0_ubo);
    vec3 l1 = to_world(l1_ubo);

    vec3 n = normalize(to_world(inNormal));

    float irr = sample_line_light_stochastic(pos, n, l0, l1, I);

    vec3 color = 1.0 - exp(-irr * vec3(1.0) - ambient);
    outColor = vec4(color,1.0);
}