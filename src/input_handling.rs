use glam::Vec2;
use winit::{
    event::{ElementState, MouseButton, VirtualKeyCode, WindowEvent},
    event_loop::ControlFlow,
};

#[derive(Default)]
pub struct Inputs {
    pub cursor_pos: Vec2,
    last_cursor_pos: Vec2,
    pub left_click: bool,
    pub right_click: bool,

    pub screenshot: bool,
    pub info: bool,

    pub move_forward: bool,
    pub move_backward: bool,
    pub move_left: bool,
    pub move_right: bool,
    pub move_up: bool,
    pub move_down: bool,
}
impl Inputs {
    pub fn do_input(&mut self, event: WindowEvent, control_flow: &mut ControlFlow) {
        match event {
            // Mouse input
            WindowEvent::CursorMoved { position, .. } => {
                self.cursor_pos.x = position.x as f32;
                self.cursor_pos.y = position.y as f32;
            }
            WindowEvent::MouseInput { state, button, .. } => {
                let pressed = match state {
                    ElementState::Pressed => true,
                    ElementState::Released => false,
                };
                match button {
                    MouseButton::Left => self.left_click = pressed,
                    MouseButton::Right => self.right_click = pressed,
                    _ => (),
                }
            }
            // Keyboard input
            WindowEvent::KeyboardInput { input, .. } => {
                if let Some(keycode) = input.virtual_keycode {
                    let pressed = match input.state {
                        ElementState::Pressed => true,
                        ElementState::Released => false,
                    };
                    match keycode {
                        VirtualKeyCode::Escape => *control_flow = ControlFlow::Exit,

                        VirtualKeyCode::P => self.screenshot = pressed,
                        VirtualKeyCode::I => self.info = pressed,

                        // WASD + Space/Ctrl Flying camera movement
                        VirtualKeyCode::W => self.move_forward = pressed,
                        VirtualKeyCode::S => self.move_backward = pressed,
                        VirtualKeyCode::A => self.move_left = pressed,
                        VirtualKeyCode::D => self.move_right = pressed,
                        VirtualKeyCode::Space => self.move_up = pressed,
                        VirtualKeyCode::LControl => self.move_down = pressed,

                        _ => (),
                    }
                }
            }
            _ => panic!("Unexpected event type"),
        }
    }

    // Returns delta since last time the function was run, so when run every frame, provides a frame-accurate delta
    pub fn cursor_delta(&mut self) -> Vec2 {
        let delta = self.cursor_pos - self.last_cursor_pos;
        self.last_cursor_pos = self.cursor_pos;
        delta
    }
}
