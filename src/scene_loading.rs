use glam::{Vec3, vec3};
use crate::datatypes::*;

pub struct Scene {
    pub vertices: Vec<Vertex>,
    pub indices: Vec<u32>,
    pub light: LineSegment,
}
impl Scene {
    pub fn test_scene_one() -> Self {
        let obj = tobj::load_obj(
            "test.obj",
            &tobj::LoadOptions {
                ignore_lines: false, // Want the line-light
                ..tobj::GPU_LOAD_OPTIONS
            },
        ).expect("Failed to load model");

        let mut positions = Vec::new();
        let mut normals = Vec::new();
        let mut light = LineSegment(Vec3::ZERO,Vec3::ZERO);
        let mut indices = Vec::new();
        for model in obj.0 {
            if model.name == "Line" {
                light.0 = unflatten_vec3(&model.mesh.positions[0..3]);
                light.1 = unflatten_vec3(&model.mesh.positions[3..6]);
                continue
            }
            let pos_len = positions.len() as u32;
            for i in model.mesh.indices {
                indices.push(i + pos_len)
            }
            if model.name == "Triangle" {
                normals.push(
                    vec![-unflatten_vec3(&model.mesh.normals[0..3]); model.mesh.positions.len() / 3] // Negative sign to make the triangle face the right direction
                );                
            } else {
                normals.push(
                    vec![unflatten_vec3(&model.mesh.normals[0..3]); model.mesh.positions.len() / 3]
                );
            }
            positions.extend(unflatten_positions(model.mesh.positions));
        }
        let normals = normals.concat();

        let vertices = positions.into_iter()
            .zip(normals)
            .map(|(position, normal)| Vertex { position, normal })
            .collect();

        Scene { vertices, indices, light }
    }
}

fn unflatten_vec3(chunk: &[f32]) -> Vec3 {
    vec3(chunk[0], chunk[1], chunk[2])
}
fn unflatten_positions(positions: Vec<f32>) -> Vec<Vec3> {
    positions
        .chunks_exact(3)
        .map(|chunk| unflatten_vec3(chunk))
        .collect()
}