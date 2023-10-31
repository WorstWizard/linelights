use std::f32::consts::PI;

use glam::{Vec3, Quat};

pub struct Scene {
    pub vertices: Vec<Vec3>,
    pub indices: Vec<u32>,
    pub light: LineSegment,
}
pub struct Camera {
    pub eye: Vec3,
    direction: Vec3,
    up: Vec3,
    angle_phi: f32, // Rotation around `up` vector
    angle_theta: f32, // Rotation around `eye` cross `up`
}
const FORWARD: Vec3 = Vec3::NEG_Z; // This should be wrong, not sure what's up
impl Camera {
    pub fn new() -> Self {
        Camera { eye: Vec3::ZERO, direction: FORWARD, up: Vec3::Y, angle_phi: 0.0, angle_theta: 0.0 }
    }
    pub fn rotate(&mut self, delta_phi: f32, delta_theta: f32) {
        self.angle_phi = (self.angle_phi + delta_phi) % (2.0*PI);
        self.angle_theta = (self.angle_theta + delta_theta).clamp(-PI/2.0 * 9.0/10.0, PI/2.0 * 9.0/10.0);
        self.direction = Quat::from_rotation_y(self.angle_phi).mul_vec3(FORWARD);
        self.direction = Quat::from_axis_angle(self.direction.cross(self.up), self.angle_theta).mul_vec3(self.direction);
    }
    pub fn direction(&self) -> Vec3 {
        self.direction
    }
    pub fn up(&self) -> Vec3 {
        self.up
    }
}

#[repr(C)]
pub struct DebugOverlay {
    pub light: LineSegment,
    pub tri_e0: LineSegment,
    pub tri_e1: LineSegment,
    pub isect0: LineSegment,
    pub isect1: LineSegment,
}
impl DebugOverlay {
    pub fn num_verts() -> u32 {
        (std::mem::size_of::<DebugOverlay>() / std::mem::size_of::<Vec3>()) as u32
    }
}
#[repr(C)]
#[derive(Clone, Copy)]
pub struct LineSegment(pub Vec3, pub Vec3);