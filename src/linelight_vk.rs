use ash::vk;
use glam::{vec3, Vec3, Vec4};
use std::mem::MaybeUninit;
use std::rc::Rc;
use tobj::{self, Model};
use vk_engine::{engine_core, shaders, BaseApp};

static APP_NAME: &str = "Linelight Experiments";

fn unflatten_positions(positions: Vec<f32>) -> Vec<Vec3> {
    positions
        .chunks_exact(3)
        .map(|chunk| vec3(chunk[0], chunk[1], chunk[2]))
        .collect()
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

#[repr(C)]
pub struct LineLightUniform {
    pub mvp: vk_engine::MVP,
    pub l0: glam::Vec4,
    pub l1: glam::Vec4,
}

pub fn make_shaders(vert_path: &str, frag_path: &str) -> Vec<vk_engine::shaders::Shader> {
    println!("Compiling shaders...");
    let shaders = vec![
        shaders::compile_shader(vert_path, None, shaders::ShaderType::Vertex)
            .expect("Could not compile vertex shader"),
        shaders::compile_shader(frag_path, None, shaders::ShaderType::Fragment)
            .expect("Could not compile fragment shader"),
    ];
    shaders
}

pub fn make_ubo_bindings() -> Vec<vk::DescriptorSetLayoutBinding> {
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
}

pub fn make_custom_app(
    shaders: &Vec<vk_engine::shaders::Shader>,
    ubo_bindings: &Vec<vk::DescriptorSetLayoutBinding>,
) -> (vk_engine::BaseApp, winit::event_loop::EventLoop<()>, vk_engine::VertexInputDescriptors, u32, (Vec4, Vec4)) {
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
    let vid = vk_engine::VertexInputDescriptors {
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

    println!("Setting up window...");
    let (window, event_loop) = vk_engine::init_window(APP_NAME, 800, 600);

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


    // // Change texture to a noise texture
    // let new_texture = engine_core::load_image_immediate(&app.instance, &app.physical_device, &app.logical_device, app.command_pool, app.graphics_queue, "noise.png");
    // *app.texture = new_texture;

    // Update descriptors
    app.update_descriptor_sets::<LineLightUniform, Vec3, u32>(num_verts as u64, num_indices as u64);

    (app, event_loop, vid, num_indices, (l0, l1))
}
