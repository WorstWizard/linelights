use glam::{Quat, Vec3};
use std::f32::consts::PI;

#[repr(C)]
pub struct LineLightUniform {
    pub mvp: vk_engine::MVP,
    pub l0: glam::Vec4,
    pub l1: glam::Vec4,
    pub accel_struct: crate::acceleration::TLAS,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct Vertex {
    pub position: Vec3,
    pub normal: Vec3,
}

pub struct Camera {
    pub eye: Vec3,
    direction: Vec3,
    up: Vec3,
    angle_theta: f32, // Rotation around `eye` cross `up`
    angle_phi: f32,   // Rotation around `up` vector
}
const FORWARD: Vec3 = Vec3::NEG_Z; // This should be wrong, not sure what's up
impl Camera {
    pub fn new() -> Self {
        Camera {
            eye: Vec3::ZERO,
            direction: FORWARD,
            up: Vec3::Y,
            angle_phi: 0.0,
            angle_theta: 0.0,
        }
    }
    pub fn rotate(&mut self, delta_theta: f32, delta_phi: f32) {
        self.angle_phi = (self.angle_phi + delta_phi) % (2.0 * PI);
        self.angle_theta = (self.angle_theta + delta_theta).clamp(-PI / 2.0001, PI / 2.0001);
        self.direction = Quat::from_rotation_y(self.angle_phi).mul_vec3(FORWARD);
        self.direction = Quat::from_axis_angle(self.direction.cross(self.up), self.angle_theta)
            .mul_vec3(self.direction);
    }
    pub fn polar_angles(&self) -> (f32, f32) {
        (self.angle_theta, self.angle_phi)
    }
    pub fn direction(&self) -> Vec3 {
        self.direction
    }
    pub fn up(&self) -> Vec3 {
        self.up
    }
}

pub const MAX_DEBUG_BOXES: usize = 32;
#[repr(C)]
#[derive(Default)]
pub struct DebugOverlay {
    pub light_triangle: [LineSegment; 3],
    pub boxes: [WireframeBox; MAX_DEBUG_BOXES],
}
impl DebugOverlay {
    pub fn num_verts() -> u32 {
        (std::mem::size_of::<DebugOverlay>() / std::mem::size_of::<Vec3>()) as u32
    }
}
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct LineSegment(pub Vec3, pub Vec3);

#[derive(Clone, Copy, Default)]
pub struct WireframeBox {
    _lines: [LineSegment; 12]
}
impl WireframeBox {
    pub fn aabb(a: Vec3, b: Vec3) -> Self {
        let d = b - a;
        let mut dx = d;
        dx.y = 0.0;
        dx.z = 0.0;
        let mut dy = d;
        dy.x = 0.0;
        dy.z = 0.0;
        let mut dz = d;
        dz.x = 0.0;
        dz.y = 0.0;
        Self {
            _lines: [
                LineSegment(a, a + dx),
                LineSegment(a, a + dy),
                LineSegment(a, a + dz),
                LineSegment(a + dx, a + dx + dy),
                LineSegment(a + dx, a + dx + dz),
                LineSegment(a + dy, a + dy + dx),
                LineSegment(a + dy, a + dy + dz),
                LineSegment(a + dz, a + dz + dx),
                LineSegment(a + dz, a + dz + dy),
                LineSegment(b, b - dx),
                LineSegment(b, b - dy),
                LineSegment(b, b - dz),
            ],
        }
    }
}