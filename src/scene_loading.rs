#![allow(dead_code)]

use crate::datatypes::*;
use glam::{vec3, Mat4, Vec3};

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
        )
        .expect("Failed to load model");

        let mut positions = Vec::new();
        let mut normals = Vec::new();
        let mut light = LineSegment(Vec3::ZERO, Vec3::ZERO);
        let mut indices = Vec::new();
        for model in obj.0 {
            if model.name == "Line" {
                light.0 = unflatten_vec3(&model.mesh.positions[0..3]);
                light.1 = unflatten_vec3(&model.mesh.positions[3..6]);
                continue;
            }
            let pos_len = positions.len() as u32;
            for i in model.mesh.indices {
                indices.push(i + pos_len)
            }
            if model.name == "Triangle" {
                normals.push(
                    vec![
                        -unflatten_vec3(&model.mesh.normals[0..3]);
                        model.mesh.positions.len() / 3
                    ], // Negative sign to make the triangle face the right direction
                );
            } else {
                normals.push(vec![
                    unflatten_vec3(&model.mesh.normals[0..3]);
                    model.mesh.positions.len() / 3
                ]);
            }
            positions.extend(unflatten_positions(model.mesh.positions));
        }
        let normals = normals.concat();

        let vertices = positions
            .into_iter()
            .zip(normals)
            .map(|(position, normal)| Vertex { position, normal })
            .collect();

        Scene {
            vertices,
            indices,
            light,
        }
    }

    pub fn test_scene_two() -> Self {
        let (vertices, indices, light) = Scene::load_gltf_mesh("better_test_scene.glb");
        Scene {
            vertices,
            indices,
            light: light.unwrap()
        }
    }

    pub fn sponza() -> Self {
        let (vertices, indices, light) = Scene::load_gltf_mesh("sponza.glb");
        Scene {
            vertices,
            indices,
            light: light.expect("Not using proper version of Sponza scene: Needs a 'Linelight' object.")
        }
    }

    fn load_gltf_mesh(path: &str) -> (Vec<Vertex>, Vec<u32>, Option<LineSegment>) {
        let (doc, buffers, _) = gltf::import(path).unwrap();

        let mut positions = Vec::new();
        let mut normals = Vec::new();
        let mut indices = Vec::new();
        let mut light = None;

        for node in doc.nodes() {
            // println!("node: {:?}", node.name());
            let transform_mat = Mat4::from_cols_array_2d(&node.transform().matrix());
            let mesh = if let Some(mesh) = node.mesh() { mesh } else { continue };

            let first_primitive = mesh.primitives().next().unwrap();
            let reader = first_primitive
                .reader(|buf| buffers.get(buf.index()).map(|data| data.0.as_slice()));

            if let Some("Linelight") = mesh.name() {
                let light_pos: Vec<Vec3> = reader
                    .read_positions()
                    .unwrap()
                    .map(|chunk| transform_mat.transform_point3(unflatten_vec3(&chunk)))
                    .collect();
                light = Some(LineSegment(light_pos[0], light_pos[1]));
                continue;
            }

            let pos_len = positions.len() as u32;
            indices.extend(
                reader
                    .read_indices()
                    .unwrap()
                    .into_u32()
                    .map(|i| i + pos_len),
            );
            positions.extend(
                reader
                    .read_positions()
                    .unwrap()
                    .map(|chunk| transform_mat.transform_point3(unflatten_vec3(&chunk))),
            );
            normals.extend(
                reader
                    .read_normals()
                    .unwrap()
                    .map(|chunk| transform_mat.transform_vector3(unflatten_vec3(&chunk))),
            );
        }
        let vertices = positions
            .into_iter()
            .zip(normals)
            .map(|(position, normal)| Vertex { position, normal })
            .collect();

        (vertices, indices, light)
    }
}

fn unflatten_vec3(chunk: &[f32]) -> Vec3 {
    vec3(chunk[0], chunk[1], chunk[2])
}
fn unflatten_positions(positions: Vec<f32>) -> Vec<Vec3> {
    positions.chunks_exact(3).map(unflatten_vec3).collect()
}
