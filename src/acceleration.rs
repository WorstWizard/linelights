use glam::{Vec3, vec3, vec2, Vec3Swizzles, Vec2};

use crate::scene_loading::Scene;

// Schwarz 2010
pub fn tri_aabb_intersect(v0: Vec3, v1: Vec3, v2: Vec3, p: Vec3, d_p: Vec3) -> bool {
    let precompute = tri_aabb_precompute(v0, v1, v2, d_p);
    tri_precomputed_aabb_intersect(&precompute, p)
}

struct ProjNormalVals {
    n_xy: Vec2,
    n_yz: Vec2,
    n_zx: Vec2,
    d_xy: f32,
    d_yz: f32,
    d_zx: f32
}
struct PrecomputedVals {
    n: Vec3,
    d_p: Vec3,
    d1: f32,
    d2: f32,
    tri_bbox: (Vec3, Vec3),
    pn_0: ProjNormalVals,
    pn_1: ProjNormalVals,
    pn_2: ProjNormalVals
}

fn tri_aabb_precompute(v0: Vec3, v1: Vec3, v2: Vec3, d_p: Vec3) -> PrecomputedVals {
    fn step(x: f32) -> f32 {
        if x > 0.0 {
            1.0
        } else {
            0.0
        }
    }

    let tri_bbox = (
        vec3(
            v0.x.min(v1.x.min(v2.x)),
            v0.y.min(v1.y.min(v2.y)),
            v0.z.min(v1.z.min(v2.z)),
        ),
        vec3(
            v0.x.max(v1.x.max(v2.x)),
            v0.y.max(v1.y.max(v2.y)),
            v0.z.max(v1.z.max(v2.z)),
        ),
    );

    let e0 = v1 - v0;
    let e1 = v2 - v1;
    let e2 = v0 - v2;
    let n = e0.cross(e1);
    let c = vec3(d_p.x * step(n.x), d_p.y * step(n.y), d_p.z * step(n.z));
    let d1 = n.dot(c - v0);
    let d2 = n.dot((d_p - c) - v0);

    fn projected_normal_vals(edge: Vec3, vert: Vec3, n: Vec3, d_p: Vec3) -> ProjNormalVals {
        let n_xy = vec2(-edge.y, edge.x) * if n.z >= 0.0 { 1.0 } else { -1.0 };
        let d_xy = -n_xy.dot(vert.xy())
            + f32::max(0.0, d_p.x * n_xy.x)
            + f32::max(0.0, d_p.y * n_xy.y);
        let n_yz = vec2(-edge.z, edge.y) * if n.x >= 0.0 { 1.0 } else { -1.0 };
        let d_yz = -n_yz.dot(vert.yz())
            + f32::max(0.0, d_p.y * n_yz.x)
            + f32::max(0.0, d_p.z * n_yz.y);
        let n_zx = vec2(-edge.x, edge.z) * if n.y >= 0.0 { 1.0 } else { -1.0 };
        let d_zx = -n_zx.dot(vert.zx())
            + f32::max(0.0, d_p.z * n_zx.x)
            + f32::max(0.0, d_p.x * n_zx.y);
        ProjNormalVals { n_xy, n_yz, n_zx, d_xy, d_yz, d_zx }
    }

    let pn_0 = projected_normal_vals(e0, v0, n, d_p);
    let pn_1 = projected_normal_vals(e1, v1, n, d_p);
    let pn_2 = projected_normal_vals(e2, v2, n, d_p);

    PrecomputedVals { n, d_p, d1, d2, tri_bbox, pn_0, pn_1, pn_2 }
}

fn tri_precomputed_aabb_intersect(precompute: &PrecomputedVals, p: Vec3) -> bool {
    fn projected_normal_check(pn: &ProjNormalVals, p: Vec3) -> bool {
        if pn.n_xy.dot(p.xy()) + pn.d_xy < 0.0 { return false }
        if pn.n_yz.dot(p.yz()) + pn.d_yz < 0.0 { return false }
        if pn.n_zx.dot(p.zx()) + pn.d_zx < 0.0 { return false }
        true
    }
    fn interval_overlaps(x1: f32, x2: f32, y1: f32, y2: f32) -> bool {
        !(x1 >= y2 || y1 >= x2)
    }

    match &precompute { &PrecomputedVals {n, d_p, d1, d2, tri_bbox, pn_0, pn_1, pn_2 } => {
        // Do the bounding boxes intersect
        if !(interval_overlaps(tri_bbox.0.x, tri_bbox.1.x, p.x, (p + *d_p).x)
        && interval_overlaps(tri_bbox.0.y, tri_bbox.1.y, p.y, (p + *d_p).y)
        && interval_overlaps(tri_bbox.0.z, tri_bbox.1.z, p.z, (p + *d_p).z))
        {
            return false;
        }

        // Does the triangle plane intersect the box
        if (n.dot(p) + d1) * (n.dot(p) + d2) > 0.0 {
            return false;
        }

        projected_normal_check(&pn_0, p) &&
        projected_normal_check(&pn_1, p) &&
        projected_normal_check(&pn_2, p)
    }}
}

pub struct AccelStruct {
    pub bbox_size: Vec3,
    pub bbox_origins: Vec<Vec3>
}

pub fn build_acceleration_structure(scene: &Scene) -> AccelStruct {
    let bbox_size = vec3(4.0, 4.0, 4.0);
    let bbox_origins = vec![
        vec3(0.0, 0.0, 0.0),
        vec3(4.0, 0.0, 0.0),
        vec3(-4.0, 0.0, 0.0),
    ];

    // for tri in scene.indices.chunks_exact(3) {

    // }

    AccelStruct { bbox_size, bbox_origins }
}