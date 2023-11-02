use std::ops::Sub;

use ash::vk;
use glam::{vec2, vec3, vec4, Mat4, Vec2, Vec3, Vec4, Vec4Swizzles};
use scene_loading::Scene;
use vk_engine::engine_core::write_struct_to_buffer;
use winit::event::{Event, WindowEvent};
use winit::event_loop::ControlFlow;

mod datatypes;
mod input_handling;
mod linelight_vk;
mod scene_loading;

use datatypes::*;
use input_handling::*;

fn main() {
    let shaders = linelight_vk::make_shaders("simple_shader.vert", "analytic.frag");
    let debug_shaders = linelight_vk::make_shaders("debugger.vert", "debugger.frag");
    let ubo_bindings = linelight_vk::make_ubo_bindings();

    let (mut app, event_loop, vid, scene) =
        linelight_vk::make_custom_app(&shaders, &debug_shaders, &ubo_bindings);

    let mut current_frame = 0;
    let mut timer = std::time::Instant::now();

    // const ROT_P_SEC: f32 = -0.00;
    // const TWO_PI: f32 = 2.0 * 3.1415926535;
    const SPEED: f32 = 100.0;
    const ENABLE_DEBUG: bool = true;

    // Stuff for debugging overlay
    let mut debug_overlay = DebugOverlay {
        light: scene.light,
        tri_e0: scene.light,
        tri_e1: scene.light,
        isect0: scene.light,
        isect1: scene.light,
    };
    unsafe {
        write_struct_to_buffer(
            app.debug_buffer
                .memory_ptr
                .expect("Uniform buffer not mapped!"),
            &debug_overlay as *const DebugOverlay,
        );
    }

    let mut inputs = Inputs::default();
    // Facing wrong way? Everything regarding view/projection is scuffed, gotta fix it at some point
    let mut camera = Camera::new();
    camera.eye = vec3(0.0, -4.0, 5.0);

    let model_pos = vec3(0.0, 0.0, 0.0);
    let model_scale = 0.5;
    let mut mvp = vk_engine::MVP {
        model: Mat4::from_scale_rotation_translation(
            vec3(model_scale, -model_scale, model_scale),
            glam::Quat::IDENTITY,
            model_pos,
        ),
        view: Mat4::IDENTITY,
        projection: Mat4::IDENTITY,
    };
    event_loop.run(move |event, _, control_flow| {
        *control_flow = ControlFlow::Poll;
        match event {
            Event::WindowEvent { event, .. } => match event {
                WindowEvent::CloseRequested => *control_flow = ControlFlow::Exit,
                WindowEvent::KeyboardInput { .. }
                | WindowEvent::CursorMoved { .. }
                | WindowEvent::MouseInput { .. } => inputs.do_input(event, control_flow),
                _ => (),
            },
            Event::MainEventsCleared => {
                app.wait_for_in_flight_fence(current_frame);

                let img_index = match app.acquire_next_image(current_frame) {
                    Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                        app.recreate_swapchain(
                            &shaders,
                            &debug_shaders,
                            &vid,
                            &ubo_bindings,
                        );
                        return;
                    }
                    Ok((i, _)) => i, // Swapchain may be suboptimal, but it's easier to just proceed
                    _ => panic!("Could not acquire image from swapchain"),
                };

                app.reset_in_flight_fence(current_frame);

                // Do debug overlay
                if ENABLE_DEBUG && inputs.left_click {
                    let cursor_pos = inputs.cursor_pos;
                    let window_size = vec2(
                        app.swapchain_extent.width as f32,
                        app.swapchain_extent.height as f32,
                    );
                    update_debug_overlay(
                        cursor_pos,
                        &mut app,
                        &mvp,
                        window_size,
                        &scene,
                        &mut debug_overlay,
                    )
                }

                // Do camera movement
                let delta_time = timer.elapsed().as_secs_f32();
                if inputs.move_forward {
                    camera.eye += camera.direction() * delta_time * SPEED * delta_time
                }
                if inputs.move_backward {
                    camera.eye -= camera.direction() * delta_time * SPEED * delta_time
                }
                if inputs.move_right {
                    camera.eye +=
                        camera.direction().cross(camera.up()) * delta_time * SPEED * delta_time
                }
                if inputs.move_left {
                    camera.eye -=
                        camera.direction().cross(camera.up()) * delta_time * SPEED * delta_time
                }
                if inputs.move_up {
                    camera.eye -= camera.up() * delta_time * SPEED * delta_time
                }
                if inputs.move_down {
                    camera.eye += camera.up() * delta_time * SPEED * delta_time
                }

                let cursor_delta = inputs.cursor_delta();
                if inputs.right_click {
                    camera.rotate(-cursor_delta.x * 0.01, cursor_delta.y * 0.01);
                }

                mvp.view = Mat4::look_to_rh(camera.eye, camera.direction(), camera.up());

                let aspect_ratio =
                    app.swapchain_extent.width as f32 / app.swapchain_extent.height as f32;
                mvp.projection =
                    Mat4::perspective_infinite_rh(f32::to_radians(90.0), aspect_ratio, 0.01);
                let ubo = LineLightUniform {
                    l0: Vec4::from((scene.light.0, 1.0)),
                    l1: Vec4::from((scene.light.1, 1.0)),
                    mvp: mvp.clone(),
                };

                unsafe {
                    write_struct_to_buffer(
                        app.uniform_buffers[current_frame]
                            .memory_ptr
                            .expect("Uniform buffer not mapped!"),
                        &ubo as *const LineLightUniform,
                    );

                    if ENABLE_DEBUG {
                        app.record_command_buffer(current_frame, |app| {
                            drawing_commands(
                                app,
                                current_frame,
                                img_index,
                                scene.indices.len() as u32,
                                DebugOverlay::num_verts(),
                            );
                        })
                    } else {
                        app.record_command_buffer(current_frame, |app| {
                            drawing_commands(
                                app,
                                current_frame,
                                img_index,
                                scene.indices.len() as u32,
                                0,
                            );
                        })
                    }
                }

                app.submit_drawing_command_buffer(current_frame);

                match app.present_image(img_index, app.sync.render_finished[current_frame]) {
                    Ok(true) | Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                        app.recreate_swapchain(&shaders, &debug_shaders, &vid, &ubo_bindings)
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

fn update_debug_overlay(
    clicked_pos: Vec2,
    app: &mut linelight_vk::LineLightApp,
    mvp: &vk_engine::MVP,
    window_size: Vec2,
    scene: &Scene,
    debug_overlay: &mut DebugOverlay,
) {
    let normalized_window_coord =
        2.0 * clicked_pos * vec2(1.0 / window_size.x, 1.0 / window_size.y) - vec2(1.0, 1.0);
    let inverse_mat = (mvp.projection.mul_mat4(&mvp.view.mul_mat4(&mvp.model))).inverse();
    let mut point_in_scene_space_0 = inverse_mat.mul_vec4(vec4(
        normalized_window_coord.x,
        normalized_window_coord.y,
        0.0,
        1.0,
    ));
    point_in_scene_space_0 *= Vec4::splat(1.0 / point_in_scene_space_0.w);
    let mut point_in_scene_space_1 = inverse_mat.mul_vec4(vec4(
        normalized_window_coord.x,
        normalized_window_coord.y,
        0.5,
        1.0,
    ));
    point_in_scene_space_1 *= Vec4::splat(1.0 / point_in_scene_space_1.w);
    let dir = point_in_scene_space_1
        .sub(point_in_scene_space_0)
        .xyz()
        .normalize();

    let collision = ray_scene_intersect(point_in_scene_space_0.xyz(), dir, scene);
    if let Some(point) = collision {
        let l0 = scene.light.0;
        let l1 = scene.light.1;
        debug_overlay.tri_e0 = LineSegment(point, l0);
        debug_overlay.tri_e1 = LineSegment(point, l1);

        let intersection = tri_tri_intersect(
            l0,
            l1,
            point,
            scene.vertices[4].position,
            scene.vertices[5].position,
            scene.vertices[6].position,
        );
        if let Some((_interval, isect0, isect1)) = intersection {
            debug_overlay.isect0 = LineSegment(point, isect0);
            debug_overlay.isect1 = LineSegment(point, isect1);
        } else {
            debug_overlay.isect0 = LineSegment(l0, l1);
            debug_overlay.isect1 = LineSegment(l0, l1);
        }

        unsafe {
            write_struct_to_buffer(
                app.debug_buffer
                    .memory_ptr
                    .expect("Uniform buffer not mapped!"),
                debug_overlay as *const DebugOverlay,
            );
        }
    }
}

fn drawing_commands(
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
        app.logical_device.cmd_next_subpass(
            app.command_buffers[buffer_index],
            vk::SubpassContents::INLINE,
        );
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
        app.logical_device
            .cmd_draw(app.command_buffers[buffer_index], num_debug_verts, 1, 0, 0);
        // Drawing commands end

        //End the render pass
        app.logical_device
            .cmd_end_render_pass(app.command_buffers[buffer_index]);
    }
}

fn ray_triangle_intersect(
    origin: Vec3,
    direction: Vec3,
    t_max: f32,
    v0: Vec3,
    v1: Vec3,
    v2: Vec3,
) -> Option<f32> {
    const EPS: f32 = 1e-5;
    let e1 = v1 - v0;
    let e2 = v2 - v0;
    let h = direction.cross(e2);
    let a = e1.dot(h);
    if a.abs() < EPS {
        return None;
    }

    let f = 1.0 / a;
    let s = origin - v0;
    let u = f * s.dot(h);
    if !(0.0..=1.0).contains(&u) {
        return None;
    }

    let q = s.cross(e1);
    let v = f * direction.dot(q);
    if v < 0.0 || (u + v) > 1.0 {
        return None;
    }

    let t = f * e2.dot(q);
    if t > EPS && t < t_max {
        return Some(t);
    }
    None
}
fn ray_scene_intersect(origin: Vec3, direction: Vec3, scene: &Scene) -> Option<Vec3> {
    let closest_t = scene
        .indices
        .chunks_exact(3)
        .filter_map(|tri| {
            let v0 = scene.vertices[tri[0] as usize].position;
            let v1 = scene.vertices[tri[1] as usize].position;
            let v2 = scene.vertices[tri[2] as usize].position;

            ray_triangle_intersect(origin, direction, f32::MAX, v0, v1, v2)
        })
        .reduce(f32::min);

    if let Some(t) = closest_t {
        return Some(t * direction + origin);
    }
    None
}
fn sort(a: &mut f32, b: &mut f32) {
    if a > b {
        // println!("swapped {}", a);
        std::mem::swap(&mut (*a), &mut (*b));
    }
}

fn sqr_dist_to_line(l: Vec3, p0: Vec3, p: Vec3) -> f32 {
    let k = p - p0;
    let d = k.cross(l);
    d.dot(d)
}
fn line_plane_intersect(n: Vec3, p0: Vec3, l: Vec3, l0: Vec3) -> Vec3 {
    let d = (p0 - l0).dot(n) / l.dot(n);
    l0 + d * l
}
fn compute_intervals_custom(
    l0: Vec3,
    l1: Vec3,
    pos: Vec3,
    v0: Vec3,
    v1: Vec3,
    v2: Vec3,
    n: Vec3,  // Normal vector for plane defined by l0,l1,pos
    dd1: f32, // Product of signed distances of v0 and v1 to triangle l0,l1,pos
    dd2: f32, // Product of signed distances of v0 and v2 to triangle l0,l1,pos
) -> Vec2 {
    // Compute intersection points between triangle v0-v1-v2 and plane defined by dot(p - pos, n) = 0
    let isect0;
    let isect1;
    if dd1 < 0.0 {
        // Line v1-v0 crosses plane
        isect0 = line_plane_intersect(n, pos, v1 - v0, v0);
        if dd2 < 0.0 {
            // Line v2-v0 crosses plane
            isect1 = line_plane_intersect(n, pos, v2 - v0, v0);
        } else {
            isect1 = line_plane_intersect(n, pos, v2 - v1, v1);
        }
    } else {
        // Lines v1-v0 does not cross plane, the others do
        isect0 = line_plane_intersect(n, pos, v2 - v0, v0);
        isect1 = line_plane_intersect(n, pos, v2 - v1, v1);
    }

    // // If distance from light to point is between distances from light to intersections, bad zone.
    // let bad_zone = {
    //     fn dist(n: Vec3, a: Vec3, p: Vec3) -> f32 {
    //         let k = p-a;
    //         let d = k.cross(n);
    //         d.dot(d)
    //     }

    //     let mut d0 = dist(l, l0, isect0);
    //     let mut d1 = dist(l, l0, isect1);
    //     let dp = dist(l, l0, pos);
    //     sort(&mut d0, &mut d1);
    //     if d0 < dp && dp < d1 {
    //         true
    //     } else {
    //         false
    //     }
    // };

    // Project intersections onto line t*(l1-l0) + l0 by computation of t-values
    let l = l1 - l0;
    let p = pos - l0;

    let i = 0;
    let j = 2;

    let mut tmp1 = isect0[i] * p[j] - isect0[j] * p[i] + pos[i] * l0[j] - pos[j] * l0[i];
    let mut tmp2 = isect0[i] * l[j] - isect0[j] * l[i] + pos[j] * l[i] - pos[i] * l[j];
    let mut t0 = tmp1 / tmp2;

    // println!("tmp1 {}, tmp2 {}, t0 {}", tmp1, tmp2, t0);

    tmp1 = isect1[i] * p[j] - isect1[j] * p[i] + pos[i] * l0[j] - pos[j] * l0[i];
    tmp2 = isect1[i] * l[j] - isect1[j] * l[i] + pos[j] * l[i] - pos[i] * l[j];
    let mut t1 = tmp1 / tmp2;

    let dp = sqr_dist_to_line(l, l0, pos);
    let di0 = sqr_dist_to_line(l, l0, isect0);
    let di1 = sqr_dist_to_line(l, l0, isect1);

    const INF: f32 = 1e10;
    if di0 > dp {
        t0 = -t0.signum() * INF
    };
    if di1 > dp {
        t1 = -t1.signum() * INF
    };

    sort(&mut t0, &mut t1);
    vec2(t0, t1)
}
fn tri_tri_intersect(
    l0: Vec3,
    l1: Vec3,
    pos: Vec3,
    v0: Vec3,
    v1: Vec3,
    v2: Vec3,
) -> Option<(Vec2, Vec3, Vec3)> {
    // Plane equation for occluding triangle: dot(n, x) + d = 0
    let e0 = v1 - v0;
    let mut e1 = v2 - v0;
    let mut n = e0.cross(e1);
    let mut d = -n.dot(v0);

    // Put light triangle into plane equation
    let d_l0 = n.dot(l0) + d;
    let d_l1 = n.dot(l1) + d;
    let d_pos = n.dot(pos) + d;

    // Same sign on all means they're on same side of plane
    if d_l0 * d_l1 > 0.0 && d_l0 * d_pos > 0.0 {
        return None;
    }

    // Plane equation for light triangle: dot(n, x) + d = 0
    let l = l1 - l0;
    e1 = pos - l0;
    n = l.cross(e1);
    d = -n.dot(l0);

    // Put triangle 1 into plane equation 2
    let dv0 = n.dot(v0) + d;
    let dv1 = n.dot(v1) + d;
    let dv2 = n.dot(v2) + d;

    let ddv1 = dv0 * dv1;
    let ddv2 = dv0 * dv2;

    if ddv1 > 0.0 && ddv2 > 0.0 {
        return None;
    }

    let interval = compute_intervals_custom(l0, l1, pos, v0, v1, v2, n, ddv1, ddv2);
    if interval[0] > 1.0 || interval[1] < 0.0 {
        return None;
    }

    let out0 = l * interval.x + l0;
    let out1 = l * interval.y + l0;

    Some((interval, out0, out1))
}
