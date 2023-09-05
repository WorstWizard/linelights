use ash::vk;
use glam::{vec3, Mat4, Vec3};
use vk_engine::engine_core::write_struct_to_buffer;
use vk_engine::*;
use winit::event::{Event, VirtualKeyCode, WindowEvent};
use winit::event_loop::ControlFlow;

static APP_NAME: &str = "Linelight Experiments";

fn main() {
    println!("Compiling shaders...");
    let shaders = vec![
        shaders::compile_shader("test.vert", None, shaders::ShaderType::Vertex)
            .expect("Could not compile vertex shader"),
        shaders::compile_shader("test.frag", None, shaders::ShaderType::Fragment)
            .expect("Could not compile fragment shader"),
    ];

    println!("Loading model...");
    // Vertices of a cube
    let verts = vec![
        vec3(-0.5, -0.5, -0.5),
        vec3(0.5, -0.5, -0.5),
        vec3(-0.5, 0.5, -0.5),
        vec3(0.5, 0.5, -0.5),
        vec3(-0.5, -0.5, 0.5),
        vec3(0.5, -0.5, 0.5),
        vec3(-0.5, 0.5, 0.5),
        vec3(0.5, 0.5, 0.5),
    ];
    let indices: Vec<u16> = vec![
        0, 1, 2, 1, 3, 2, //front
        5, 4, 6, 5, 6, 7, //back
        4, 0, 6, 0, 2, 6, //left
        1, 5, 3, 5, 7, 3, //right
        4, 5, 0, 5, 1, 0, //top
        2, 3, 6, 3, 7, 6, //bottom
    ];
    let num_indices = indices.len() as u32;
    let vid = VertexInputDescriptors {
        attributes: vec![*vk::VertexInputAttributeDescription::builder()
            .binding(0)
            .location(0)
            .format(vk::Format::R32G32B32_SFLOAT)
            .offset(0)],
        bindings: vec![*vk::VertexInputBindingDescription::builder()
            .binding(0)
            .input_rate(vk::VertexInputRate::VERTEX)
            .stride(std::mem::size_of::<Vec3>() as u32)],
    };
    let ubo_vec = vec![vk_engine::MVP {
        model: Mat4::IDENTITY,
        view: Mat4::IDENTITY,
        projection: Mat4::IDENTITY,
    }];
    let ubo_bindings = uniform_buffer_descriptor_set_layout_bindings(ubo_vec.len());

    println!("Setting up window...");
    let (window, event_loop) = init_window(APP_NAME, 800, 600);

    println!("Initializing application...");
    let mut app = BaseApp::new(
        window,
        APP_NAME,
        &shaders,
        verts,
        indices,
        &vid,
        Some(ubo_vec),
        Some(ubo_bindings.clone()),
    );

    let mut current_frame = 0;
    let mut timer = std::time::Instant::now();
    let mut theta = 0.0;

    const ROT_P_SEC: f32 = 0.25;
    const TWO_PI: f32 = 2.0*3.1415926535;

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
                        app.recreate_swapchain(&shaders, &vid, Some(ubo_bindings.clone()));
                        return;
                    }
                    Ok((i, _)) => i, // Swapchain may be suboptimal, but it's easier to just proceed
                    _ => panic!("Could not acquire image from swapchain"),
                };

                app.reset_in_flight_fence(current_frame);

                let eye = vec3(0.0, -1.0, 0.0);
                let model_pos = vec3(0.0, 0.0, 2.0);
                let up = vec3(0.0, -1.0, 0.0);
                let aspect_ratio =
                    app.swapchain_extent.width as f32 / app.swapchain_extent.height as f32;

                theta = (theta + (ROT_P_SEC * TWO_PI) * timer.elapsed().as_secs_f32()) % TWO_PI;

                let model = Mat4::from_rotation_translation(glam::Quat::from_rotation_y(theta), model_pos);
                let view = Mat4::look_at_lh(eye, model_pos, up);
                let projection =
                    Mat4::perspective_infinite_lh(f32::to_radians(90.0), aspect_ratio, 0.01);
                let mut correction_mat = Mat4::IDENTITY;
                correction_mat.y_axis = glam::vec4(0.0, -1.0, 0.0, 0.0);

                let ubo = vk_engine::MVP {
                    model,
                    view: correction_mat.mul_mat4(&view),
                    projection,
                };

                unsafe {
                    write_struct_to_buffer(
                        app.uniform_buffers[current_frame]
                            .memory_ptr
                            .expect("Uniform buffer not mapped!"),
                        &ubo as *const vk_engine::MVP,
                    );

                    app.record_command_buffer(current_frame, |app| {
                        vk_engine::drawing_commands(
                            app,
                            current_frame,
                            img_index,
                            |app| {
                                app.logical_device.cmd_draw_indexed(
                                    app.command_buffers[current_frame],
                                    num_indices,
                                    1,
                                    0,
                                    0,
                                    0,
                                );
                            },
                            &[0.0],
                        )
                    })
                }

                app.submit_drawing_command_buffer(current_frame);

                match app.present_image(img_index, app.sync.render_finished[current_frame]) {
                    Ok(true) | Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => app.recreate_swapchain(&shaders, &vid, Some(ubo_bindings.clone())),
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
