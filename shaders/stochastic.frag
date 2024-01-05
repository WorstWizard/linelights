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

    // Generate rays
    Ray[NUM_SAMPLES] rays;
    for (int i=0; i<NUM_SAMPLES; i++) {
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

        rays[i] = r;
    }

    for (int i = 0; i < acceleration_indices.length(); i += 3) {
        vec3 v0 = to_world(verts[acceleration_indices[i]].pos);
        vec3 v1 = to_world(verts[acceleration_indices[i+1]].pos);
        vec3 v2 = to_world(verts[acceleration_indices[i+2]].pos);

        for (int i=0; i<NUM_SAMPLES; i++) {
            Ray r = rays[i];
            if ( ray_triangle_intersect(r, v0, v1, v2) ) {
                rays[i].hit = true;
            }
        }
    }

    float irr = 0.0;
    for (int i=0; i<NUM_SAMPLES; i++) {
        Ray r = rays[i];
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