use ash::vk;
use glam::{vec3, Mat4};
use vk_engine::engine_core::write_struct_to_buffer;
use winit::event::{Event, VirtualKeyCode, WindowEvent};
use winit::event_loop::ControlFlow;

mod linelight_vk;

fn main() {
    let shaders = linelight_vk::make_shaders("simple_shader.vert", "analytic.frag");
    let ubo_bindings = linelight_vk::make_ubo_bindings();

    let (mut app, event_loop, vid, num_indices, (l0, l1)) =
        linelight_vk::make_custom_app(&shaders, &ubo_bindings);

    let mut current_frame = 0;
    let mut timer = std::time::Instant::now();
    let mut theta = 0.0;

    const ROT_P_SEC: f32 = -0.01;
    const TWO_PI: f32 = 2.0 * 3.1415926535;

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
                _ => (),
            },
            Event::MainEventsCleared => {
                app.wait_for_in_flight_fence(current_frame);

                let img_index = match app.acquire_next_image(current_frame) {
                    Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                        app.recreate_swapchain(&shaders, &vid, ubo_bindings.clone());
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

                let model = Mat4::from_scale_rotation_translation(
                    vec3(model_scale, -model_scale, model_scale),
                    glam::Quat::from_rotation_y(theta),
                    model_pos,
                );
                let view = Mat4::look_at_rh(eye, vec3(0.0, 0.0, 0.0), -up);
                let projection =
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
                        drawing_commands(app, current_frame, img_index)
                    })
                }

                app.submit_drawing_command_buffer(current_frame);

                match app.present_image(img_index, app.sync.render_finished[current_frame]) {
                    Ok(true) | Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                        app.recreate_swapchain(&shaders, &vid, ubo_bindings.clone())
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

        // Drawing commands end

        //End the render pass
        app.logical_device
            .cmd_end_render_pass(app.command_buffers[buffer_index]);
    }
}
