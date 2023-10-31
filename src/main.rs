use std::ops::Sub;

use ash::vk;
use glam::{vec3, Mat4, Vec4Swizzles, vec2, vec4, Vec4, Vec3};
use vk_engine::engine_core::{write_struct_to_buffer, write_vec_to_buffer};
use winit::event::{Event, VirtualKeyCode, WindowEvent, ElementState};
use winit::event_loop::ControlFlow;

mod linelight_vk;

fn main() {
    let shaders = linelight_vk::make_shaders("simple_shader.vert", "analytic.frag");
    let debug_shaders = linelight_vk::make_shaders("debugger.vert", "debugger.frag");
    let ubo_bindings = linelight_vk::make_ubo_bindings();

    let (mut app, event_loop, vid, num_indices, (l0, l1), scene_verts, scene_indices) =
        linelight_vk::make_custom_app(&shaders, &debug_shaders, &ubo_bindings);

    let mut current_frame = 0;
    let mut timer = std::time::Instant::now();
    let mut theta = 0.0;

    const ROT_P_SEC: f32 = -0.01;
    const TWO_PI: f32 = 2.0 * 3.1415926535;

    // Stuff for debugging overlay
    let mut debug_verts = vec![
        l0.xyz(),
        l1.xyz(),
    ];
    unsafe {
        write_vec_to_buffer(
            app.debug_buffer
                .memory_ptr
                .expect("Uniform buffer not mapped!"),
            &debug_verts,
        );
    }
    let mut mouse_clicked_this_frame = true;
    let mut mouse_position = vec2(0.0, 0.0);

    let mut model = Mat4::IDENTITY;
    let mut view = Mat4::IDENTITY;
    let mut projection = Mat4::IDENTITY;

    event_loop.run(move |event, _, control_flow| {
        *control_flow = ControlFlow::Poll;
        match event {
            Event::WindowEvent { event, .. } => match event {
                WindowEvent::CloseRequested => *control_flow = ControlFlow::Exit,
                //WindowEvent::Resized(new_size) => app.recreate_swapchain(&shaders, &vid, Some(ubo_bindings.clone())),
                WindowEvent::KeyboardInput { input, .. } => match input.virtual_keycode {
                    Some(VirtualKeyCode::Escape) => *control_flow = ControlFlow::Exit,
                    _ => (),
                },
                WindowEvent::CursorMoved { position, .. } => {
                    mouse_position.x = position.x as f32;
                    mouse_position.y = position.y as f32;
                },
                WindowEvent::MouseInput { state, button: winit::event::MouseButton::Left, .. } => match state {
                    ElementState::Pressed => {
                        if mouse_clicked_this_frame {
                            let normalized_window_coord = 2.0 * mouse_position * vec2(1.0/app.swapchain_extent.width as f32, 1.0/app.swapchain_extent.height as f32) - vec2(1.0, 1.0);
                            // println!("Window coords: {}", normalized_window_coord);
                            let inverse_mat = (projection.mul_mat4(&view.mul_mat4(&model))).inverse();
                            let mut point_in_object_space_1 = inverse_mat.mul_vec4(vec4(normalized_window_coord.x, normalized_window_coord.y, 0.0, 1.0));
                            point_in_object_space_1 *= Vec4::splat(1.0/point_in_object_space_1.w);
                            let mut point_in_object_space_2 = inverse_mat.mul_vec4(vec4(normalized_window_coord.x, normalized_window_coord.y, 0.5, 1.0));
                            point_in_object_space_2 *= Vec4::splat(1.0/point_in_object_space_2.w);
                            let dir = point_in_object_space_2.sub(point_in_object_space_1).xyz().normalize();

                            let collision = ray_scene_intersect(point_in_object_space_1.xyz(), dir, &scene_verts, &scene_indices);
                            if let Some(point) = collision {
                                debug_verts.push(point_in_object_space_1.xyz());
                                debug_verts.push(point);
                                unsafe {
                                    write_vec_to_buffer(
                                        app.debug_buffer
                                            .memory_ptr
                                            .expect("Uniform buffer not mapped!"),
                                        &debug_verts,
                                    );
                                }    
                            }

                            // println!("{}", point_in_object_space_1.xyz());
                            // println!("{}", second_point);
                        }
                        mouse_clicked_this_frame = false;
                    }
                    ElementState::Released => {
                        mouse_clicked_this_frame = true;
                    }
                }
                _ => (),
            },
            Event::MainEventsCleared => {
                app.wait_for_in_flight_fence(current_frame);

                let img_index = match app.acquire_next_image(current_frame) {
                    Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                        app.recreate_swapchain(&shaders, &debug_shaders, &vid, ubo_bindings.clone());
                        return;
                    }
                    Ok((i, _)) => i, // Swapchain may be suboptimal, but it's easier to just proceed
                    _ => panic!("Could not acquire image from swapchain"),
                };

                app.reset_in_flight_fence(current_frame);

                let eye = vec3(0.0, -4.0, 5.0);
                let model_pos = vec3(0.0, 0.0, 0.0);
                let up = vec3(0.0, -1.0, 0.0);
                let aspect_ratio =
                    app.swapchain_extent.width as f32 / app.swapchain_extent.height as f32;
                let model_scale = 0.5;

                theta = (theta + (ROT_P_SEC * TWO_PI) * timer.elapsed().as_secs_f32()) % TWO_PI;

                model = Mat4::from_scale_rotation_translation(
                    vec3(model_scale, -model_scale, model_scale),
                    glam::Quat::from_rotation_y(theta),
                    model_pos,
                );
                view = Mat4::look_at_rh(eye, vec3(0.0, 0.0, 0.0), -up);
                projection =
                    Mat4::perspective_infinite_rh(f32::to_radians(90.0), aspect_ratio, 0.01);
                // let mut correction_mat = Mat4::IDENTITY;
                // correction_mat.y_axis = glam::vec4(0.0, -1.0, 0.0, 0.0);

                let mvp = vk_engine::MVP {
                    model,
                    view,
                    projection,
                };

                let ubo = linelight_vk::LineLightUniform { l0, l1, mvp };

                unsafe {
                    write_struct_to_buffer(
                        app.uniform_buffers[current_frame]
                            .memory_ptr
                            .expect("Uniform buffer not mapped!"),
                        &ubo as *const linelight_vk::LineLightUniform,
                    );


                    app.record_command_buffer(current_frame, |app| {
                        drawing_commands(app, current_frame, img_index, num_indices, debug_verts.len() as u32);
                    })
                }

                app.submit_drawing_command_buffer(current_frame);

                match app.present_image(img_index, app.sync.render_finished[current_frame]) {
                    Ok(true) | Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                        app.recreate_swapchain(&shaders, &debug_shaders, &vid, ubo_bindings.clone())
                    }
                    Ok(false) => (),
                    _ => panic!("Could not present image!"),
                }

                timer = std::time::Instant::now();
                current_frame = (current_frame + 1) % vk_engine::engine_core::MAX_FRAMES_IN_FLIGHT;
            }
            _ => (),
        }
    });
}

pub fn drawing_commands(
    app: &mut linelight_vk::LineLightApp,
    buffer_index: usize,
    swapchain_image_index: u32,
    num_indices: u32,
    num_debug_verts: u32,
) {
    //Start render pass
    let render_area = vk::Rect2D::builder()
        .offset(vk::Offset2D { x: 0, y: 0 })
        .extent(app.swapchain_extent);
    let mut clear_values = [vk::ClearValue::default(); 2];
    clear_values[0].color.float32 = [0.0, 0.0, 0.0, 1.0];
    clear_values[1].depth_stencil = vk::ClearDepthStencilValue {
        depth: 1.0,
        stencil: 0,
    };
    let renderpass_begin_info = vk::RenderPassBeginInfo::builder()
        .render_pass(app.render_pass)
        .framebuffer(app.framebuffers[swapchain_image_index as usize])
        .render_area(*render_area)
        .clear_values(&clear_values);
    unsafe {
        app.logical_device.cmd_begin_render_pass(
            app.command_buffers[buffer_index],
            &renderpass_begin_info,
            vk::SubpassContents::INLINE,
        );
        app.logical_device.cmd_bind_pipeline(
            app.command_buffers[buffer_index],
            vk::PipelineBindPoint::GRAPHICS,
            app.graphics_pipeline,
        );
        let vertex_buffers = [app.vertex_buffer.buffer];
        let offsets = [0];
        app.logical_device.cmd_bind_vertex_buffers(
            app.command_buffers[buffer_index],
            0,
            &vertex_buffers,
            &offsets,
        );
        app.logical_device.cmd_bind_index_buffer(
            app.command_buffers[buffer_index],
            app.index_buffer.buffer,
            0,
            vk::IndexType::UINT32,
        );
        app.logical_device.cmd_bind_descriptor_sets(
            app.command_buffers[buffer_index],
            vk::PipelineBindPoint::GRAPHICS,
            app.graphics_pipeline_layout,
            0,
            &[app.descriptor_sets[buffer_index]],
            &[],
        );

        // Drawing commands begin
        app.logical_device.cmd_draw_indexed(
            app.command_buffers[buffer_index],
            num_indices,
            1,
            0,
            0,
            0,
        );
        // Debug drawing subpass
        app.logical_device.cmd_next_subpass(app.command_buffers[buffer_index], vk::SubpassContents::INLINE);
        app.logical_device.cmd_bind_vertex_buffers(
            app.command_buffers[buffer_index],
            0,
            &[app.debug_buffer.buffer],
            &[0],
        );
        app.logical_device.cmd_bind_pipeline(
            app.command_buffers[buffer_index],
            vk::PipelineBindPoint::GRAPHICS,
            app.debug_pipeline,
        );
        app.logical_device.cmd_draw(
            app.command_buffers[buffer_index],
            num_debug_verts,
            1,
            0,
            0,
        );
        // Drawing commands end

        //End the render pass
        app.logical_device
            .cmd_end_render_pass(app.command_buffers[buffer_index]);
    }
}

fn ray_triangle_intersect(origin: Vec3, direction: Vec3, t_max: f32, v0: Vec3, v1: Vec3, v2: Vec3) -> Option<f32> {
    const EPS: f32 = 1e-5;
    let e1 = v1 - v0;
    let e2 = v2 - v0;
    let h = direction.cross(e2);
    let a = e1.dot(h);
    if a.abs() < EPS { return None }

    let f = 1.0/a;
    let s = origin - v0;
    let u = f * s.dot(h);
    if u < 0.0 || u > 1.0 { return None }

    let q = s.cross(e1);
    let v = f * direction.dot(q);
    if v < 0.0 || (u+v) > 1.0 { return None }

    let t = f * e2.dot(q);
    if t > EPS && t < t_max { return Some(t) }
    None
}
fn ray_scene_intersect(origin: Vec3, direction: Vec3, vertices: &Vec<Vec3>, indices: &Vec<u32>) -> Option<Vec3> {
    let closest_t = indices.chunks_exact(3).filter_map(|tri| {
        let v0 = vertices[tri[0] as usize];
        let v1 = vertices[tri[1] as usize];
        let v2 = vertices[tri[2] as usize];

        ray_triangle_intersect(origin, direction, f32::MAX, v0, v1, v2)
    }).reduce(f32::min);

    if let Some(t) = closest_t {
        return Some( t * direction + origin )
    }
    None
}