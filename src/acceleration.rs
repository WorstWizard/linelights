// Schwarz 2010
fn tri_aabb_intersect(v0: Vec3, v1: Vec3, v2: Vec3, p: Vec3, d_p: Vec3) -> bool {
    fn step(x: f32) -> f32 {
        if x > 0.0 {
            1.0
        } else {
            0.0
        }
    }
    fn interval_overlaps(x1: f32, x2: f32, y1: f32, y2: f32) -> bool {
        !(x1 >= y2 || y1 >= x2)
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

    if !(interval_overlaps(tri_bbox.0.x, tri_bbox.1.x, p.x, (p+d_p).x)
        && interval_overlaps(tri_bbox.0.y, tri_bbox.1.y, p.y, (p+d_p).y)
        && interval_overlaps(tri_bbox.0.z, tri_bbox.1.z, p.z, (p+d_p).z))
    {
        return false;
    }

    let e0 = v1 - v0;
    let e1 = v2 - v1;
    let e2 = v0 - v2;
    let n = e0.cross(e1);
    let c = vec3(d_p.x * step(n.x), d_p.y * step(n.y), d_p.z * step(n.z));
    let d1 = n.dot(c - v0);
    let d2 = n.dot((d_p - c) - v0);
    // Does the triangle plane intersect the box
    if (n.dot(p) + d1) * (n.dot(p) + d2) > 0.0 {
        return false;
    }

    let edges = [e0, e1, e2];
    let verts = [v0, v1, v2];
    for i in 0..3 {
        let n_xy = vec2(-edges[i].y, edges[i].x) * if n.z >= 0.0 { 1.0 } else { -1.0 };
        let d = -n_xy.dot(verts[i].xy())
            + f32::max(0.0, d_p.x * n_xy.x)
            + f32::max(0.0, d_p.y * n_xy.y);
        if n_xy.dot(p.xy()) + d < 0.0 {
            return false;
        }

        let n_xz = vec2(edges[i].z, -edges[i].x) * if n.y >= 0.0 { 1.0 } else { -1.0 };
        let d = -n_xz.dot(verts[i].xz())
            + f32::max(0.0, d_p.x * n_xz.x)
            + f32::max(0.0, d_p.z * n_xz.y);
        if n_xz.dot(p.xz()) + d < 0.0 {
            return false;
        }

        let n_yz = vec2(-edges[i].z, edges[i].y) * if n.x >= 0.0 { 1.0 } else { -1.0 };
        let d = -n_yz.dot(verts[i].yz())
            + f32::max(0.0, d_p.y * n_yz.x)
            + f32::max(0.0, d_p.z * n_yz.y);
        if n_yz.dot(p.yz()) + d < 0.0 {
            return false;
        }
    }
    true
}