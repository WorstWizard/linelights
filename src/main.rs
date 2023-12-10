use acceleration::{precomputed_tri_aabb_intersect, tri_aabb_precompute, AccelStruct};
use ash::vk;
use glam::{vec2, vec3, vec4, Mat4, Vec2, Vec3, Vec4, Vec4Swizzles};
use scene_loading::Scene;
use vk_engine::engine_core::{self, write_struct_to_buffer};
use winit::event::{Event, WindowEvent};
use winit::event_loop::ControlFlow;

use tracy_client::{self, frame_mark, span};

mod acceleration;
mod datatypes;
mod input_handling;
mod linelight_vk;
mod scene_loading;

use datatypes::*;
use input_handling::*;

// Some config options
const SPEED: f32 = 1.0;
const ENABLE_DEBUG: bool = true;


fn main() {
    // Connect to tracy for performance statistics
    let _client = tracy_client::Client::start();
    let _span = span!("init");

    let analytic_shader = linelight_vk::make_shaders("shaders/simple_shader.vert", "shaders/analytic.frag");
    let stochastic_shader = linelight_vk::make_shaders("shaders/simple_shader.vert", "shaders/stochastic.frag");
    let debug_shader = linelight_vk::make_shaders("shaders/simple_shader.vert", "shaders/debug.frag");

    let shaders = vec![analytic_shader, stochastic_shader, debug_shader];
    let debug_shaders = linelight_vk::make_shaders("shaders/debugger.vert", "shaders/debugger.frag");
    let ubo_bindings = linelight_vk::make_ubo_bindings();
    

    println!("Loading model...");
    let scene = Scene::dragon_small_light(32);
    // let scene = Scene::sponza(32);
    let (accel_struct, accel_indices, _) =
        acceleration::build_acceleration_structure(&scene);

    let mut debug_overlay = DebugOverlay::default();
    debug_overlay.light_triangle[0] = scene.light;

    let (mut app, event_loop, vid, did) = linelight_vk::make_custom_app(
        &shaders,
        &debug_shaders,
        &ubo_bindings,
        &scene,
        &accel_struct,
        &accel_indices,
    );

    let mut current_frame = 0;
    let mut timer = std::time::Instant::now();

    unsafe {
        write_struct_to_buffer(
            app.debug_buffer
                .memory_ptr
                .expect("Uniform buffer not mapped!"),
            &debug_overlay as *const DebugOverlay,
        );
    }

    let mut inputs = Inputs::default();
    let mut just_took_screenshot = false; // Helper variable to ensure only one is taken per keypress
    let mut just_printed_info = false;
    // Facing wrong way? Everything regarding view/projection is scuffed, gotta fix it at some point
    let mut camera = Camera::new();
    // camera.eye = vec3(0.0, -4.0, 5.0);
    // camera.eye = vec3(0.0, -6.0, 0.0);
    // camera.rotate(std::f32::consts::FRAC_PI_2, 0.0);
    camera.eye = vec3(-1.3054297, -2.1971848, -4.514163);
    camera.rotate(0.3527179, -2.8560042);

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
    
    drop(_span);

    #[cfg(feature="gpu_trace")]
    let _gpu_ctx = {
        app.reset_timestamps(app.command_buffers[0]);
        let period = app.get_timestamp_period();
        let timestamp = app.get_timestamp_immediately();
        _client.new_gpu_context(
            Some("GPU Context"),
            tracy_client::GpuContextType::Vulkan,
            timestamp,
            period,
        )
        .unwrap()
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
                        app.recreate_swapchain(&shaders, &debug_shaders, &vid, &did, &ubo_bindings);
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
                        &accel_struct,
                        &mut debug_overlay,
                    )
                }

                // Do camera movement
                let delta_time = timer.elapsed().as_secs_f32();
                // println!("{}",delta_time*1000.0);
                timer = std::time::Instant::now();

                // println!("delta time {delta_time}");
                if inputs.move_forward {
                    camera.eye += camera.direction() * delta_time * SPEED
                }
                if inputs.move_backward {
                    camera.eye -= camera.direction() * delta_time * SPEED
                }
                if inputs.move_right {
                    camera.eye += camera.direction().cross(camera.up()).normalize_or_zero()
                        * delta_time
                        * SPEED
                }
                if inputs.move_left {
                    camera.eye -= camera.direction().cross(camera.up()).normalize_or_zero()
                        * delta_time
                        * SPEED
                }
                if inputs.move_up {
                    camera.eye -= camera.up() * delta_time * SPEED
                }
                if inputs.move_down {
                    camera.eye += camera.up() * delta_time * SPEED
                }

                let cursor_delta = inputs.cursor_delta();
                if inputs.right_click {
                    camera.rotate(cursor_delta.y * 0.01, -cursor_delta.x * 0.01);
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
                if inputs.info && !just_printed_info {
                    just_printed_info = true;
                    println!(
                        "Camera:\n\t Position: {}, Direction: {}, Angles: {:?}",
                        camera.eye,
                        camera.direction(),
                        camera.polar_angles()
                    );
                } else if !inputs.info {
                    just_printed_info = false;
                }

                let selected_shader = if inputs.selected_shader < app.graphics_pipelines.len() {
                    inputs.selected_shader
                } else {
                    0
                };

                #[cfg(feature="gpu_trace")]
                let mut t_stamp = (0, 0);
                #[cfg(feature="gpu_trace")]
                let mut _span;
                unsafe {
                    write_struct_to_buffer(
                        app.uniform_buffers[current_frame]
                            .memory_ptr
                            .expect("Uniform buffer not mapped!"),
                        &ubo as *const LineLightUniform,
                    );

                    #[cfg(feature="gpu_trace")]
                    {
                        // Record first timestamp immediately before GPU work
                        app.reset_timestamps(app.command_buffers[current_frame]);
                        t_stamp.0 = app.get_timestamp_immediately();
                        // app.record_immediate_timestamp(app.command_buffers[current_frame], true);
                        _span = _gpu_ctx
                            .span_alloc("Drawing", "event_loop", "main.rs", 184)
                            .unwrap();
                    }

                    if ENABLE_DEBUG {
                        app.record_command_buffer(current_frame, |app| {
                            drawing_commands(
                                app,
                                current_frame,
                                img_index,
                                selected_shader,
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
                                selected_shader,
                                scene.indices.len() as u32,
                                0,
                            );
                        })
                    }
                }

                // Submit commands, begin work
                app.submit_drawing_command_buffer(current_frame);

                // Record second timestamp immediately after GPU work
                #[cfg(feature="gpu_trace")]
                {
                    t_stamp.1 = app.get_timestamp_immediately();
                    _span.end_zone();
                    _span.upload_timestamp(t_stamp.0, t_stamp.1);
                }

                // After drawing, grab a screenshot if requested
                if inputs.screenshot && !just_took_screenshot {
                    just_took_screenshot = true;
                    take_screenshot(&app, img_index);
                } else if !inputs.screenshot {
                    just_took_screenshot = false;
                }

                match app.present_image(img_index, app.sync.render_finished[current_frame], current_frame) {
                    Ok(true) | Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                        app.recreate_swapchain(&shaders, &debug_shaders, &vid, &did, &ubo_bindings)
                    }
                    Ok(false) => (),
                    _ => panic!("Could not present image!"),
                }

                current_frame = (current_frame + 1) % vk_engine::engine_core::MAX_FRAMES_IN_FLIGHT;
                frame_mark();
            }
            _ => (),
        }
    });
}

fn take_screenshot(app: &linelight_vk::LineLightApp, img_index: u32) {
    let (width, height) = (app.swapchain_extent.width, app.swapchain_extent.height);

    let _screenshot_span = span!("Take screenshot");
    let _buffer_span = span!("Buffer creation");
    let (screenshot_buffer, memory_size) = app.make_screenshot_buffer();
    drop(_buffer_span);
    let _transfer_span = span!("Data transfer");
    let buf_copy = vk::BufferImageCopy::builder()
        .image_extent(vk::Extent3D {
            width,
            height,
            depth: 1,
        })
        .image_subresource(
            *vk::ImageSubresourceLayers::builder()
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .layer_count(1),
        );

    unsafe {
        engine_core::immediate_commands(
            &app.logical_device,
            app.command_pool,
            app.graphics_queue,
            |buf| {
                app.logical_device.cmd_copy_image_to_buffer(
                    buf,
                    app.swapchain_images[img_index as usize],
                    vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                    screenshot_buffer.buffer,
                    &[*buf_copy],
                );
            },
        )
    };
    drop(_transfer_span);
    let _copy_span = span!("Copying buffer to CPU");
    let samples = unsafe {
        std::slice::from_raw_parts(
            screenshot_buffer.memory_ptr.unwrap() as *const u8,
            memory_size as usize,
        )
    };
    let mut img = image::RgbaImage::from_raw(width, height, samples.into()).unwrap();
    drop(_copy_span);

    // Save file in detached thread: This takes a little while and doesn't need to block
    std::thread::spawn(move || {
        let _thread_span = span!("Saving screenshot to file");
        for pix in img.pixels_mut() {
            let b = pix.0[0];
            pix.0[0] = pix.0[2];
            pix.0[2] = b;
        }
        match img.save_with_format("screenshot.png", image::ImageFormat::Png) {
            Ok(_) => println!("Saved screenshot to 'screenshot.png'"),
            Err(_) => println!("Failed to save screenshot"),
        }
    });
}

fn drawing_commands(
    app: &mut linelight_vk::LineLightApp,
    buffer_index: usize,
    swapchain_image_index: u32,
    pipeline_index: usize,
    num_indices: u32,
    num_debug_verts: u32,
    // gpu_ctx: Option<&tracy_client::GpuContext>,
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
        app.logical_device.cmd_reset_query_pool(
            app.command_buffers[buffer_index],
            app.query_pool,
            0,
            2,
        );

        app.logical_device.cmd_begin_render_pass(
            app.command_buffers[buffer_index],
            &renderpass_begin_info,
            vk::SubpassContents::INLINE,
        );
        app.logical_device.cmd_bind_pipeline(
            app.command_buffers[buffer_index],
            vk::PipelineBindPoint::GRAPHICS,
            app.graphics_pipelines[pipeline_index],
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
        if ENABLE_DEBUG {
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
        }
        // Drawing commands end

        //End the render pass
        app.logical_device
            .cmd_end_render_pass(app.command_buffers[buffer_index]);
    }
}

fn update_debug_overlay(
    clicked_pos: Vec2,
    app: &mut linelight_vk::LineLightApp,
    mvp: &vk_engine::MVP,
    window_size: Vec2,
    scene: &Scene,
    accel_struct: &AccelStruct,
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
    let dir = (point_in_scene_space_1 - point_in_scene_space_0)
        .xyz()
        .normalize();

    if let Some((point, normal)) = ray_scene_intersect(point_in_scene_space_0.xyz(), dir, scene) {
        // debug_overlay.occluding_tris = [scene.light; 3 * 7];
        // debug_overlay.intersections = [scene.light; 2 * ARR_MAX];

        let point = point + 0.005 * normal.normalize();
        let l0 = scene.light.0;
        let l1 = scene.light.1;

        // Draw normal of clicked point
        // debug_overlay.normal = LineSegment(point, point + normal.normalize());

        // Draw lines from point to linelight end-points, blocked by geometry
        // if let Some((isect, _)) = ray_scene_intersect(point, (l0 - point).normalize(), scene) {
        //     debug_overlay.tri_e0 = LineSegment(point, isect);
        // } else {
        //     debug_overlay.tri_e0 = LineSegment(point, l0);
        // }
        // if let Some((isect, _)) = ray_scene_intersect(point, (l1 - point).normalize(), scene) {
        //     debug_overlay.tri_e1 = LineSegment(point, isect);
        // } else {
        //     debug_overlay.tri_e1 = LineSegment(point, l1);
        // }
        use acceleration::GRID_SIZE;

        debug_overlay.light_triangle[1] = LineSegment(l0, point);
        debug_overlay.light_triangle[2] = LineSegment(l1, point);
        debug_overlay.boxes = [WireframeBox::default(); MAX_DEBUG_BOXES];

        let mut hit_boxes = 0;
        let pc = tri_aabb_precompute(l0, l1, point, accel_struct.bbox_size);
        for i in 0..GRID_SIZE {
            for j in 0..GRID_SIZE {
                for k in 0..GRID_SIZE {
                    let ijk = vec3(i as f32, j as f32, k as f32);
                    let bbox_origin = accel_struct.origin + ijk * accel_struct.bbox_size;
                    if hit_boxes < MAX_DEBUG_BOXES && precomputed_tri_aabb_intersect(&pc, bbox_origin) {
                        debug_overlay.boxes[hit_boxes] = WireframeBox::aabb(bbox_origin, bbox_origin + accel_struct.bbox_size);
                        hit_boxes += 1;
                    }
                }
            }
        }


        // let mut int_arr = IntervalArray {
        //     size: 0,
        //     data: [Vec2::ZERO; ARR_MAX],
        // };
        // add_interval(&mut int_arr, vec2(0.0, 1.0));

        // // let mut occluders = 0;
        // for tri in scene.indices.chunks_exact(3) {
        //     if int_arr.size == 0 {
        //         break;
        //     }

        //     let v0 = scene.vertices[tri[0] as usize].position;
        //     let v1 = scene.vertices[tri[1] as usize].position;
        //     let v2 = scene.vertices[tri[2] as usize].position;

        //     if let Some((interval, _)) = tri_tri_intersect(l0, l1, point, v0, v1, v2) {
        //         // println!("occluding interval: {}", interval);
        //         occlude_intervals(&mut int_arr, interval);
        //         // if occluders < debug_overlay.occluding_tris.len()/3 {
        //         //     debug_overlay.occluding_tris[occluders*3] = LineSegment(v0,v1);
        //         //     debug_overlay.occluding_tris[occluders*3+1] = LineSegment(v1,v2);
        //         //     debug_overlay.occluding_tris[occluders*3+2] = LineSegment(v2,v0);
        //         //     debug_overlay.intersections[2*occluders] = line;
        //         // }
        //         // occluders += 1;
        //     }
        // }

        // println!("number of intervals: {}", int_arr.size);
        // println!("intervals: {:?}", int_arr.data.get(0..int_arr.size));

        // for (i, interval) in int_arr.data.iter().enumerate() {
        //     let i0 = l * interval.x + l0;
        //     let i1 = l * interval.y + l0;
        //     debug_overlay.intersections[2 * i] = LineSegment(point, i0);
        //     debug_overlay.intersections[2 * i + 1] = LineSegment(point, i1);
        // }

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

fn ray_triangle_intersect(
    origin: Vec3,
    direction: Vec3,
    t_max: f32,
    v0: Vec3,
    v1: Vec3,
    v2: Vec3,
) -> Option<(f32, Vec3)> {
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
        return Some((t, e1.cross(e2)));
    }
    None
}
fn ray_scene_intersect(origin: Vec3, direction: Vec3, scene: &Scene) -> Option<(Vec3, Vec3)> {
    let closest_hit = scene
        .indices
        .chunks_exact(3)
        .filter_map(|tri| {
            let v0 = scene.vertices[tri[0] as usize].position;
            let v1 = scene.vertices[tri[1] as usize].position;
            let v2 = scene.vertices[tri[2] as usize].position;

            ray_triangle_intersect(origin, direction, f32::MAX, v0, v1, v2)
        })
        .reduce(|acc, e| if e.0 < acc.0 { e } else { acc });

    if let Some(hit) = closest_hit {
        return Some((hit.0 * direction + origin, hit.1));
    }
    None
}

/*
fn sort(a: &mut f32, b: &mut f32) {
    if a > b {
        // println!("swapped {}", a);
        std::mem::swap(&mut (*a), &mut (*b));
    }
}

fn projected_sqr_dist_to_line(l0: Vec3, l1: Vec3, p: Vec3) -> f32 {
    let x0 = p.x;
    let x1 = l0.x;
    let x2 = l1.x;
    let z0 = p.z;
    let z1 = l0.z;
    let z2 = l1.z;
    ((x2 - x1) * (z1 - z0) - (z2 - z1) * (x1 - x0)).abs()
}
fn line_line_intersect_2d(p1: Vec2, p2: Vec2, p3: Vec2, p4: Vec2) -> f32 {
    let dx = p3.x - p4.x;
    let dy = p3.y - p4.y;
    let top = (p1.x - p3.x) * dy - (p1.y - p3.y) * dx;
    let bot = (p1.x - p2.x) * dy - (p1.y - p2.y) * dx;
    top / bot
}
fn linesegments_intersect(p1: Vec2, p2: Vec2, p3: Vec2, p4: Vec2) -> bool {
    let a = p3.x - p4.x;
    let b = p1.x - p3.x;
    let c = p3.y - p4.y;
    let d = p1.y - p3.y;
    let e = p1.x - p2.x;
    let f = p1.y - p2.y;
    let t = (b * c - d * a) / (e * c - f * a);
    let u = (b * f - d * e) / (e * c - f * a);
    if t >= 0.0 && t <= 1.0 && u >= 0.0 && u <= 1.0 {
        true
    } else {
        false
    }
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
) -> (Vec2, LineSegment) {
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

    // It may occur that the intersection points are further away from the light than
    // the sampled point, in which case there is no occlusion
    let dp = projected_sqr_dist_to_line(l0, l1, pos);
    let di0 = projected_sqr_dist_to_line(l0, l1, isect0);
    let di1 = projected_sqr_dist_to_line(l0, l1, isect1);
    if di0 > dp && di1 > dp {
        return (vec2(2.0, 2.0), LineSegment(isect0, isect1));
    };

    // If one intersection is further away from the line than the sampled point,
    // its corresponding t-value should be at infinity
    const INF: f32 = 1e10;
    let mut t0 = line_line_intersect_2d(l0.xz(), l1.xz(), isect0.xz(), pos.xz());
    let mut t1 = line_line_intersect_2d(l0.xz(), l1.xz(), isect1.xz(), pos.xz());
    if di0 < dp && di1 < dp {
        // Best and most common case, t-values are already good
        sort(&mut t0, &mut t1);
        return (vec2(t0, t1), LineSegment(isect0, isect1));
    }

    // Let t0 correspond to the point closer than pos, t1 the more distant
    if di1 >= dp {
        t1 = t0;
    }
    let intersects_left = linesegments_intersect(l0.xz(), pos.xz(), isect0.xz(), isect1.xz());
    let intersects_right = linesegments_intersect(l1.xz(), pos.xz(), isect0.xz(), isect1.xz());
    if intersects_left {
        if intersects_right {
            // Both
            t0 = -t1.signum() * INF;
        } else {
            // Only left
            t0 = -INF;
        }
    } else if intersects_right {
        // Only right
        t0 = INF;
    } else {
        // No intersection!
        return (vec2(2.0, 2.0), LineSegment(isect0, isect1));
    }

    sort(&mut t0, &mut t1);
    (vec2(t0, t1), LineSegment(isect0, isect1))
}
fn tri_tri_intersect(
    l0: Vec3,
    l1: Vec3,
    pos: Vec3,
    v0: Vec3,
    v1: Vec3,
    v2: Vec3,
) -> Option<(Vec2, LineSegment)> {
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

    let (interval, line) = compute_intervals_custom(l0, l1, pos, v0, v1, v2, n, ddv1, ddv2);
    if interval[0] > 1.0 || interval[1] < 0.0 {
        return None;
    }

    Some((interval, line))
}

// Records info on which parts of a linelight is visible as an array of intervals (t-values in [0,1])
const ARR_MAX: usize = 8;
struct IntervalArray {
    pub size: usize,
    pub data: [Vec2; ARR_MAX],
}
// No bounds checking for speed, just don't make mistakes ;)
fn remove_interval(int_arr: &mut IntervalArray, i: usize) {
    int_arr.data[i] = int_arr.data[int_arr.size - 1];
    int_arr.data[int_arr.size - 1] = Vec2::ZERO; // just for debug overlay to look proper
    int_arr.size -= 1;
}
fn add_interval(int_arr: &mut IntervalArray, new_interval: Vec2) {
    if int_arr.size < ARR_MAX {
        // Avoid overflow
        int_arr.data[int_arr.size] = new_interval;
        int_arr.size += 1;
    }
    // println!("new intervals {:?}", int_arr.data)
}
// Given an interval of occlusion, update the array to reflect the new visible intervals
fn occlude_intervals(int_arr: &mut IntervalArray, occ_int: Vec2) {
    let mut i = 0;
    while i < int_arr.size {
        // println!("index: {}", i);
        let interval = int_arr.data[i];
        // println!("existing interval {}", interval);

        if occ_int.x <= interval.x && occ_int.y >= interval.y {
            // Interval is fully occluded, remove it but do not increment `i`
            // as the swapped-in element needs to be checked too
            // println!("removing");
            remove_interval(int_arr, i);
            continue;
        } else if occ_int.x > interval.x && occ_int.y < interval.y {
            // Middle is occluded, shrink existing to the left and add new interval to the right
            // println!("add new");
            add_interval(int_arr, vec2(occ_int.y, interval.y));
            int_arr.data[i].y = occ_int.x;
        } else if occ_int.x > interval.x && occ_int.x < interval.y {
            // Right side is occluded, shrink to fit
            // println!("shrink left");
            int_arr.data[i].y = occ_int.x;
        } else if occ_int.y > interval.x && occ_int.y < interval.y {
            // Left side is occluded, shrink to fit
            // println!("shrink right");
            int_arr.data[i].x = occ_int.y;
        }
        i += 1;
    }
}
*/