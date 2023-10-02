use std::mem::MaybeUninit;
use std::rc::Rc;

use ash::vk;
use glam::{vec3, Mat4, Vec3};
use vk_engine::engine_core::write_struct_to_buffer;
use vk_engine::*;
use winit::event::{Event, VirtualKeyCode, WindowEvent};
use winit::event_loop::ControlFlow;

use tobj::{self, Model};

static APP_NAME: &str = "Linelight Experiments";

fn unflatten_positions(positions: Vec<f32>) -> Vec<Vec3> {
    positions
        .chunks_exact(3)
        .map(|chunk| vec3(chunk[0], chunk[1], chunk[2]))
        .collect()
}
// fn indices_to_u16(indices: Vec<u32>) -> Vec<u16> {
//     indices.into_iter().map(|num| num as u16).collect()
// }

// fn load_bunny() -> (Vec<Vec3>, Vec<u16>) {
//     let bunny = tobj::load_obj("bunny.obj", &tobj::GPU_LOAD_OPTIONS);
//     let (mut models, _) = bunny.expect("Failed to load model!");
//     let bunny_model = models.pop().unwrap();

//     let vertices = unflatten_positions(bunny_model.mesh.positions);
//     let indices = indices_to_u16(bunny_model.mesh.indices);

//     (vertices, indices)
// }

#[repr(C)]
struct LineLightUniform {
    mvp: MVP,
    l0: glam::Vec4,
    l1: glam::Vec4,
}
#[repr(C)]
struct Vertex {
    pos: Vec3,
    normal: Vec3,
}

fn load_test_scene() -> (Model, Model, Model) {
    let obj = tobj::load_obj(
        "test.obj",
        &tobj::LoadOptions {
            ignore_lines: false, // Want the line-light
            ..tobj::GPU_LOAD_OPTIONS
        },
    );
    let (mut models, _) = obj.expect("Failed to load test scene");
    let mut plane = MaybeUninit::<Model>::uninit();
    let mut triangle = MaybeUninit::<Model>::uninit();
    let mut line = MaybeUninit::<Model>::uninit();
    for m in models.drain(..) {
        match m.name.as_str() {
            "Plane" => {
                plane.write(m);
            }
            "Triangle" => {
                triangle.write(m);
            }
            "Line" => {
                line.write(m);
            }
            _ => panic!("Wrong model in .obj"),
        }
    }

    unsafe {
        (
            plane.assume_init(),
            triangle.assume_init(),
            line.assume_init(),
        )
    }
}

// fn make_vertices(model: Model) -> Vec<Vertex> {
//     // TODO TOMORROW
//     let positions: Vec<Vec3> = model
//         .mesh
//         .positions
//         .chunks_exact(3)
//         .map(|c| vec3(c[0], c[1], c[2]))
//         .collect();
//     let mut normals = Vec::with_capacity(model.mesh.normal_indices.len());
//     for idx in model.mesh.normal_indices {
//         normals.push(model.mesh.normals[idx as usize])
//     }

//     positions.into_iter().zip(normals.into_iter()).map(|pos, normal| Vertex {pos, normal})
// }

fn main() {
    println!("Compiling shaders...");
    let shaders = vec![
        shaders::compile_shader("simple_shader.vert", None, shaders::ShaderType::Vertex)
            .expect("Could not compile vertex shader"),
        shaders::compile_shader("analytic.frag", None, shaders::ShaderType::Fragment)
            .expect("Could not compile fragment shader"),
    ];

    println!("Loading model...");

    let (plane, triangle, line) = load_test_scene();

    let verts = [
        unflatten_positions(plane.mesh.positions),
        unflatten_positions(triangle.mesh.positions),
    ]
    .concat();
    let num_verts = verts.len();
    // let plane_indices = indices_to_u16(plane.mesh.indices);
    let plane_indices = plane.mesh.indices;
    // let triangle_indices: Vec<u16> = indices_to_u16(triangle.mesh.indices)
    let triangle_indices: Vec<u32> = triangle
        .mesh
        .indices
        .into_iter()
        .map(|i| i + 4) // Four verts in plane
        .collect();
    let indices = [plane_indices, triangle_indices].concat();

    let line_verts = unflatten_positions(line.mesh.positions);
    let (l0, l1) = (line_verts[0], line_verts[1]);

    let l0 = glam::Vec4::from((l0, 1.0));
    let l1 = glam::Vec4::from((l1, 1.0));

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

    let ubo_bindings = {
        let mut binding_vec = Vec::with_capacity(3);
        binding_vec.push(
            *vk::DescriptorSetLayoutBinding::builder()
                .binding(0)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT),
        );
        binding_vec.push(
            *vk::DescriptorSetLayoutBinding::builder()
                .binding(1)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::FRAGMENT),
        );
        binding_vec.push(
            *vk::DescriptorSetLayoutBinding::builder()
                .binding(2)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::FRAGMENT),
        );
        binding_vec.push(
            *vk::DescriptorSetLayoutBinding::builder()
                .binding(3)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::FRAGMENT),
        );
        binding_vec
    };

    println!("Setting up window...");
    let (window, event_loop) = init_window(APP_NAME, 800, 600);

    println!("Initializing application...");
    let mut app = BaseApp::new::<Vec3, u32, LineLightUniform>(
        window,
        APP_NAME,
        &shaders,
        verts.clone(),
        indices.clone(),
        &vid,
        ubo_bindings.clone(),
    );

    // Change vertex and index buffer to support usage in as ssbo in fragment shader

    fn remake_buffer<T: Sized>(
        app: &mut BaseApp,
        buffer_data: &Vec<T>,
        usage_flags: vk::BufferUsageFlags,
    ) -> engine_core::ManagedBuffer {
        let new_buffer = {
            let buffer = engine_core::buffer::create_buffer(
                &app.logical_device,
                (std::mem::size_of::<T>() * buffer_data.len()) as u64,
                usage_flags | vk::BufferUsageFlags::TRANSFER_DST,
            );
            let buffer_memory = engine_core::buffer::allocate_and_bind_buffer(
                &app.instance,
                &app.physical_device,
                &app.logical_device,
                buffer,
                vk::MemoryPropertyFlags::DEVICE_LOCAL,
            );

            engine_core::ManagedBuffer {
                logical_device: Rc::clone(&app.logical_device),
                // memory_size,
                buffer,
                buffer_memory: Some(buffer_memory),
                memory_ptr: None,
            }
        };

        let mut staging_buffer = engine_core::create_staging_buffer(
            &app.instance,
            &app.physical_device,
            &app.logical_device,
            (std::mem::size_of::<T>() * buffer_data.len()) as u64,
        );
        staging_buffer.map_buffer_memory();

        unsafe {
            engine_core::write_vec_to_buffer(staging_buffer.memory_ptr.unwrap(), buffer_data)
        };
        engine_core::copy_buffer(
            &app.logical_device,
            app.command_pool,
            app.graphics_queue,
            *staging_buffer,
            *new_buffer,
            (std::mem::size_of::<T>() * buffer_data.len()) as u64,
        );

        new_buffer
    }

    *app.vertex_buffer = remake_buffer(
        &mut app,
        &verts,
        vk::BufferUsageFlags::VERTEX_BUFFER | vk::BufferUsageFlags::STORAGE_BUFFER,
    );
    *app.index_buffer = remake_buffer(
        &mut app,
        &indices,
        vk::BufferUsageFlags::INDEX_BUFFER | vk::BufferUsageFlags::STORAGE_BUFFER,
    );

    // Update descriptors
    app.update_descriptor_sets::<LineLightUniform, Vec3, u32>(num_verts as u64, num_indices as u64);

    let mut current_frame = 0;
    let mut timer = std::time::Instant::now();
    let mut theta = 0.0;

    const ROT_P_SEC: f32 = 0.0;
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

                let ubo = LineLightUniform {
                    l0,
                    l1,
                    mvp,
                    // linelight: (l0, l1)
                };

                unsafe {
                    write_struct_to_buffer(
                        app.uniform_buffers[current_frame]
                            .memory_ptr
                            .expect("Uniform buffer not mapped!"),
                        &ubo as *const LineLightUniform,
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
                            vk::IndexType::UINT32,
                        )
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
