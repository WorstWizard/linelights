use std::mem::MaybeUninit;

use ash::vk;
use glam::{vec3, Mat4, Vec3};
use vk_engine::engine_core::write_struct_to_buffer;
use vk_engine::*;
use winit::event::{Event, VirtualKeyCode, WindowEvent};
use winit::event_loop::ControlFlow;

use tobj::{self, Model};

static APP_NAME: &str = "Linelight Experiments";

fn unflatten_positions(positions: Vec<f32>) -> Vec<Vec3> {
    positions.chunks_exact(3).map(|chunk| vec3(chunk[0],chunk[1],chunk[2])).collect()
}
fn indices_to_u16(indices: Vec<u32>) -> Vec<u16> {
    indices.into_iter().map(|num| num as u16).collect()
}

fn load_bunny() -> (Vec<Vec3>, Vec<u16>) {
    let bunny = tobj::load_obj("bunny.obj", &tobj::GPU_LOAD_OPTIONS);
    let (mut models, _) = bunny.expect("Failed to load model!");
    let bunny_model = models.pop().unwrap();

    let vertices = unflatten_positions(bunny_model.mesh.positions);
    let indices = indices_to_u16(bunny_model.mesh.indices);

    (vertices, indices)
}

fn load_test_scene() -> (Model, Model, Model) {
    let obj = tobj::load_obj("test.obj", &tobj::LoadOptions{
        ignore_lines: false, // Want the line-light
        ..tobj::GPU_LOAD_OPTIONS
    });
    let (mut models, _) = obj.expect("Failed to load test scene");
    let mut plane = MaybeUninit::<Model>::uninit();
    let mut triangle = MaybeUninit::<Model>::uninit();
    let mut line = MaybeUninit::<Model>::uninit();
    for m in models.drain(..) {
        match m.name.as_str() {
            "Plane" => {plane.write(m);},
            "Triangle" => {triangle.write(m);},
            "Line" => {line.write(m);},
            _ => panic!("Wrong model in .obj")
        }
    }

    unsafe{ (plane.assume_init(), triangle.assume_init(), line.assume_init()) }
}

fn main() {

    
    println!("Compiling shaders...");
    let shaders = vec![
        shaders::compile_shader("test.vert", None, shaders::ShaderType::Vertex)
            .expect("Could not compile vertex shader"),
        shaders::compile_shader("test.frag", None, shaders::ShaderType::Fragment)
            .expect("Could not compile fragment shader"),
    ];
            
    println!("Loading model...");
    
    let (plane, triangle, line) = load_test_scene();

    let verts = [unflatten_positions(plane.mesh.positions), unflatten_positions(triangle.mesh.positions)].concat();
    let plane_indices = indices_to_u16(plane.mesh.indices);
    let triangle_indices: Vec<u16> = indices_to_u16(triangle.mesh.indices)
        .into_iter()
        .map(|i| i + 4) // Four verts in plane
        .collect();
    let indices = [plane_indices, triangle_indices].concat();

    // let (verts, indices) = load_bunny();
    // println!("verts {}, indices {}", plane.mesh.positions.len()/3, plane.mesh.indices.len());
    // let verts = unflatten_positions(plane.mesh.positions);
    // let indices = indices_to_u16(plane.mesh.indices);

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

    const ROT_P_SEC: f32 = 0.05;
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

                let eye = vec3(0.0, -3.0, 5.0);
                let model_pos = vec3(0.0, 0.0, 0.0);
                let up = vec3(0.0, -1.0, 0.0);
                let aspect_ratio =
                    app.swapchain_extent.width as f32 / app.swapchain_extent.height as f32;
                let model_scale = 1.0;

                theta = (theta + (ROT_P_SEC * TWO_PI) * timer.elapsed().as_secs_f32()) % TWO_PI;

                let model = Mat4::from_scale_rotation_translation(vec3(model_scale, -model_scale, model_scale), glam::Quat::from_rotation_y(theta), model_pos);
                let view = Mat4::look_at_rh(eye, vec3(0.0, 0.0, 0.0), -up);
                let projection =
                    Mat4::perspective_infinite_rh(f32::to_radians(90.0), aspect_ratio, 0.01);
                // let mut correction_mat = Mat4::IDENTITY;
                // correction_mat.y_axis = glam::vec4(0.0, -1.0, 0.0, 0.0);

                let ubo = vk_engine::MVP {
                    model,
                    view,
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
