use glam::{vec2, vec3, Vec2, Vec3, Vec3Swizzles};

use crate::scene_loading::Scene;

// Schwarz 2010
pub fn tri_aabb_intersect(v0: Vec3, v1: Vec3, v2: Vec3, p: Vec3, d_p: Vec3) -> bool {
    let precompute = tri_aabb_precompute(v0, v1, v2, d_p);
    precomputed_tri_aabb_intersect(&precompute, p)
}

struct ProjNormalVals {
    n_xy: Vec2,
    n_yz: Vec2,
    n_zx: Vec2,
    d_xy: f32,
    d_yz: f32,
    d_zx: f32,
}
pub struct PrecomputedVals {
    n: Vec3,
    d_p: Vec3,
    d1: f32,
    d2: f32,
    tri_bbox: (Vec3, Vec3),
    pn_0: ProjNormalVals,
    pn_1: ProjNormalVals,
    pn_2: ProjNormalVals,
}

pub fn tri_aabb_precompute(v0: Vec3, v1: Vec3, v2: Vec3, d_p: Vec3) -> PrecomputedVals {
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
        let d_xy =
            -n_xy.dot(vert.xy()) + f32::max(0.0, d_p.x * n_xy.x) + f32::max(0.0, d_p.y * n_xy.y);
        let n_yz = vec2(-edge.z, edge.y) * if n.x >= 0.0 { 1.0 } else { -1.0 };
        let d_yz =
            -n_yz.dot(vert.yz()) + f32::max(0.0, d_p.y * n_yz.x) + f32::max(0.0, d_p.z * n_yz.y);
        let n_zx = vec2(-edge.x, edge.z) * if n.y >= 0.0 { 1.0 } else { -1.0 };
        let d_zx =
            -n_zx.dot(vert.zx()) + f32::max(0.0, d_p.z * n_zx.x) + f32::max(0.0, d_p.x * n_zx.y);
        ProjNormalVals {
            n_xy,
            n_yz,
            n_zx,
            d_xy,
            d_yz,
            d_zx,
        }
    }

    let pn_0 = projected_normal_vals(e0, v0, n, d_p);
    let pn_1 = projected_normal_vals(e1, v1, n, d_p);
    let pn_2 = projected_normal_vals(e2, v2, n, d_p);

    PrecomputedVals {
        n,
        d_p,
        d1,
        d2,
        tri_bbox,
        pn_0,
        pn_1,
        pn_2,
    }
}

pub fn precomputed_tri_aabb_intersect(precompute: &PrecomputedVals, p: Vec3) -> bool {
    fn projected_normal_check(pn: &ProjNormalVals, p: Vec3) -> bool {
        if pn.n_xy.dot(p.xy()) + pn.d_xy < 0.0 {
            return false;
        }
        if pn.n_yz.dot(p.yz()) + pn.d_yz < 0.0 {
            return false;
        }
        if pn.n_zx.dot(p.zx()) + pn.d_zx < 0.0 {
            return false;
        }
        true
    }
    fn interval_overlaps(x1: f32, x2: f32, y1: f32, y2: f32) -> bool {
        !(x1 >= y2 || y1 >= x2)
    }

    match &precompute {
        &PrecomputedVals {
            n,
            d_p,
            d1,
            d2,
            tri_bbox,
            pn_0,
            pn_1,
            pn_2,
        } => {
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

            projected_normal_check(&pn_0, p)
                && projected_normal_check(&pn_1, p)
                && projected_normal_check(&pn_2, p)
        }
    }
}

pub type AccelStruct = TLAS;
pub const GRID_SIZE: usize = 4;
pub const BBOX_COUNT: usize = GRID_SIZE*GRID_SIZE*GRID_SIZE;
#[repr(C)]
#[derive(Clone, Copy, Default, Debug)]
pub struct BufferView {
    pub offset: i32,
    pub size: i32,
}
#[derive(Clone, Debug)]
#[repr(C)]
pub struct BLAS {
    buffer_views: [BufferView; BBOX_COUNT],
}
#[repr(C)]
pub struct TLAS {
    pub size: Vec3,
    pub origin: Vec3,
    subgrids: [BLAS; BBOX_COUNT]
}

pub fn build_acceleration_structure(scene: &Scene) -> (TLAS, Vec<u32>) {
    use std::{time::Instant, io::{stdout, Write}};
    print!("Building acceleration structure...");
    stdout().flush().unwrap();

    let scene_aabb = {
        let mut min = scene.vertices[0].position;
        let mut max = min;
        for pos in scene.vertices.iter().map(|v| v.position) {
            if pos.x < min.x {
                min.x = pos.x
            } else if pos.x > max.x {
                max.x = pos.x
            }
            if pos.y < min.y {
                min.y = pos.y
            } else if pos.y > max.y {
                max.y = pos.y
            }
            if pos.z < min.z {
                min.z = pos.z
            } else if pos.z > max.z {
                max.z = pos.z
            }
        }
        (min - Vec3::splat(0.1), max + Vec3::splat(0.1))
    };

    let size = scene_aabb.1 - scene_aabb.0;
    let origin = scene_aabb.0;
    let mut subgrids: Vec<BLAS> = Vec::with_capacity(BBOX_COUNT);
    let mut index_buffer = Vec::new();
    
    // Construct grid of BLAS's
    let mut timer = Instant::now();
    let blas_size = size/(GRID_SIZE as f32);
    for i in 0..GRID_SIZE {
        for j in 0..GRID_SIZE {
            for k in 0..GRID_SIZE {
                if timer.elapsed().as_secs_f32() > 1.0 {
                    print!(" {}/{}...", subgrids.len(), BBOX_COUNT);
                    stdout().flush().unwrap();
                    timer = Instant::now()
                }
                let ijk = vec3(i as f32, j as f32, k as f32);
                let blas_origin = origin + ijk * blas_size;
                subgrids.push(build_blas(&scene, &mut index_buffer, blas_origin, blas_size));
            }
        }
    }
    println!();

    (TLAS { size, origin, subgrids: subgrids.try_into().unwrap() }, index_buffer)
}

fn build_blas(scene: &Scene, index_buffer: &mut Vec<u32>, origin: Vec3, size: Vec3) -> BLAS {
    let bbox_size = size/(GRID_SIZE as f32);
    let mut buffer_views = Vec::with_capacity(BBOX_COUNT);
    for i in 0..GRID_SIZE {
        for j in 0..GRID_SIZE {
            for k in 0..GRID_SIZE {
                let ijk = vec3(i as f32, j as f32, k as f32);
                let bbox_origin = origin + ijk * bbox_size;
                let length_before = index_buffer.len();
                
                for tri_indices in scene.indices.chunks_exact(3) {
                    let v0 = scene.vertices[tri_indices[0] as usize].position;
                    let v1 = scene.vertices[tri_indices[1] as usize].position;
                    let v2 = scene.vertices[tri_indices[2] as usize].position;
            
                    if tri_aabb_intersect(v0, v1, v2, bbox_origin, bbox_size) {
                        index_buffer.extend(tri_indices);
                    }
                }
                let length_after = index_buffer.len();
                buffer_views.push(
                    BufferView { offset: length_before as i32, size: (length_after-length_before) as i32 }
                )
            }
        }
    }
    BLAS { buffer_views: buffer_views.try_into().unwrap() }
}


// pub fn build_acceleration_structure(scene: &Scene) -> (AccelStruct, Vec<u32>, (Vec3, Vec3)) {
//     println!("# bboxes {BBOX_COUNT}");
//     let scene_aabb = {
//         let mut min = scene.vertices[0].position;
//         let mut max = min;
//         for pos in scene.vertices.iter().map(|v| v.position) {
//             if pos.x < min.x {
//                 min.x = pos.x
//             } else if pos.x > max.x {
//                 max.x = pos.x
//             }
//             if pos.y < min.y {
//                 min.y = pos.y
//             } else if pos.y > max.y {
//                 max.y = pos.y
//             }
//             if pos.z < min.z {
//                 min.z = pos.z
//             } else if pos.z > max.z {
//                 max.z = pos.z
//             }
//         }
//         (min - Vec3::splat(0.1), max + Vec3::splat(0.1))
//     };

//     let bbox_size = (scene_aabb.1 - scene_aabb.0)/(GRID_SIZE as f32);
//     let origin = scene_aabb.0;
//     let mut bbox_origins = Vec::with_capacity(BBOX_COUNT);
//     for i in 0..GRID_SIZE {
//         for j in 0..GRID_SIZE {
//             for k in 0..GRID_SIZE {
//                 let ijk = vec3(i as f32, j as f32, k as f32);
//                 let bbox_origin = origin + ijk * bbox_size;
//                 bbox_origins.push(bbox_origin);
//             }
//         }
//     }

//     // For each triangle in the scene, check which bounding boxes contains the triangle (may be multiple)
//     // and fill their corresponding index-lists with the triangle indices
//     let mut index_arrays = vec![Vec::new(); BBOX_COUNT];
//     for tri_indices in scene.indices.chunks_exact(3) {
//         let v0 = scene.vertices[tri_indices[0] as usize].position;
//         let v1 = scene.vertices[tri_indices[1] as usize].position;
//         let v2 = scene.vertices[tri_indices[2] as usize].position;

//         let precompute = tri_aabb_precompute(v0, v1, v2, bbox_size);
//         for (bbox_i, bbox_pos) in bbox_origins.iter().enumerate() {
//             if precomputed_tri_aabb_intersect(&precompute, *bbox_pos) {
//                 index_arrays[bbox_i].extend_from_slice(tri_indices);
//             }
//         }
//     }
//     let sizes = index_arrays
//         .iter()
//         .map(|list| list.len() as u32)
//         .collect::<Vec<u32>>()
//         .try_into()
//         .unwrap();

//     (
//         AccelStruct {
//             bbox_size,
//             origin,
//             sizes,
//         },
//         index_arrays.into_iter().flatten().collect(),
//         scene_aabb,
//     )
// }