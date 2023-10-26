use ash::extensions::ext::DebugUtils;
use ash::extensions::khr::{Surface, Swapchain};
use ash::vk;
use glam::{vec3, Vec3, Vec4};
use std::mem::ManuallyDrop;
use std::{mem::MaybeUninit, ffi::CString};
use std::rc::Rc;
use tobj::{self, Model};
use vk_engine::{engine_core, shaders};

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
#[derive(Clone, Copy)]
pub struct Vertex {
    position: Vec3,
    normal: Vec3
}
impl Vertex {
    fn input_descriptors() -> vk_engine::VertexInputDescriptors {
        vk_engine::VertexInputDescriptors {
            attributes: vec![*vk::VertexInputAttributeDescription::builder()
                .binding(0)
                .location(0)
                .format(vk::Format::R32G32B32_SFLOAT)
                .offset(0),
            *vk::VertexInputAttributeDescription::builder()
                .binding(0)
                .location(1)
                .format(vk::Format::R32G32B32_SFLOAT)
                .offset(std::mem::size_of::<Vec3>() as u32)],
            bindings: vec![*vk::VertexInputBindingDescription::builder()
                .binding(0)
                .input_rate(vk::VertexInputRate::VERTEX)
                .stride(std::mem::size_of::<Self>() as u32)],
        }
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
) -> (LineLightApp, winit::event_loop::EventLoop<()>, vk_engine::VertexInputDescriptors, u32, (Vec4, Vec4)) {
    println!("Loading model...");
    let (plane, triangle, line) = load_test_scene();

    let normals = [
        vec![unflatten_positions(plane.mesh.normals)[0]; plane.mesh.positions.len() / 3],
        vec![-unflatten_positions(triangle.mesh.normals)[0]; triangle.mesh.positions.len() / 3], // Negated because triangle faces wrong direction
    ].concat();
    let positions = [
        unflatten_positions(plane.mesh.positions),
        unflatten_positions(triangle.mesh.positions),
    ]
    .concat();

    let verts: Vec<Vertex> = positions.into_iter().zip(normals).map(|(p, n)| Vertex {position: p, normal: n}).collect();

    let num_verts = verts.len();

    let plane_indices = plane.mesh.indices;
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
    let vid = Vertex::input_descriptors();

    println!("Setting up window...");
    let (window, event_loop) = vk_engine::init_window(APP_NAME, 800, 600);

    println!("Initializing application...");
    let mut app = LineLightApp::new(
        window,
        &shaders,
        &vid,
        ubo_bindings.clone(),
        &verts,
        &indices,
    );

    (app, event_loop, vid, num_indices, (l0, l1))
}


use raw_window_handle::{HasRawDisplayHandle, HasRawWindowHandle};
pub struct LineLightApp {
    swapchain_loader: ash::extensions::khr::Swapchain,
    swapchain: vk::SwapchainKHR,
    pub swapchain_extent: vk::Extent2D,
    image_views: Vec<vk::ImageView>,
    pub render_pass: vk::RenderPass,
    pub graphics_pipeline: vk::Pipeline,
    pub graphics_pipeline_layout: vk::PipelineLayout,
    pub vertex_buffer: ManuallyDrop<engine_core::ManagedBuffer>,
    pub index_buffer: ManuallyDrop<engine_core::ManagedBuffer>,
    depth_image: ManuallyDrop<engine_core::ManagedImage>,
    pub descriptor_sets: Vec<vk::DescriptorSet>,
    descriptor_set_layout: vk::DescriptorSetLayout,
    descriptor_pool: vk::DescriptorPool,
    pub framebuffers: Vec<vk::Framebuffer>,
    present_queue: vk::Queue,
    graphics_queue: vk::Queue,
    pub command_buffers: Vec<vk::CommandBuffer>,
    command_pool: vk::CommandPool,
    pub uniform_buffers: Vec<vk_engine::engine_core::ManagedBuffer>,
    pub sync: engine_core::SyncPrims,
    surface_loader: ash::extensions::khr::Surface,
    surface: vk::SurfaceKHR,
    window: winit::window::Window,
    pub logical_device: Rc<ash::Device>,
    instance: Box<ash::Instance>,
}
impl LineLightApp {
    pub fn new(
        window: winit::window::Window,
        shaders: &Vec<vk_engine::shaders::Shader>,
        vertex_input_descriptors: &vk_engine::VertexInputDescriptors,
        descriptor_set_bindings: Vec<vk::DescriptorSetLayoutBinding>,
        vertices: &Vec<Vertex>,
        indices: &Vec<u32>
    ) -> Self {
        
        type VertexType = Vertex;
        type IndexType = u32;
        type UBOType = LineLightUniform;

        let entry = Box::new(unsafe { ash::Entry::load() }.unwrap());
        if engine_core::VALIDATION_ENABLED && !engine_core::check_validation_layer_support(&entry) {
            panic!("Validation layer requested but not available!");
        }

        //// Application info
        let app_name = CString::new(APP_NAME).unwrap();
        let engine_name = CString::new("Vulkan Engine").unwrap();

        let app_info = vk::ApplicationInfo::builder()
            .application_name(&app_name)
            .application_version(vk::make_api_version(0, 1, 0, 0))
            .engine_name(&engine_name)
            .engine_version(vk::API_VERSION_1_0)
            .api_version(vk::API_VERSION_1_0);

        let mut instance_extensions =
            ash_window::enumerate_required_extensions(window.raw_display_handle())
                .unwrap()
                .to_vec();
        if engine_core::VALIDATION_ENABLED {
            instance_extensions.push(DebugUtils::name().as_ptr());
        }

        //// Instance & debug messenger
        let mut messenger_info = engine_core::init_debug_messenger_info();
        let mut instance_info = vk::InstanceCreateInfo::builder()
            .application_info(&app_info)
            .enabled_extension_names(&instance_extensions);
        if engine_core::VALIDATION_ENABLED {
            instance_info = instance_info
                .enabled_layer_names(&engine_core::VALIDATION_LAYERS)
                .push_next(&mut messenger_info);
        }
        let instance = Box::new(
            unsafe { entry.create_instance(&instance_info, None) }
                .expect("Failed to create Vulkan instance!"),
        );
        let (_debug_loader, _messenger) = if engine_core::VALIDATION_ENABLED {
            //Messenger attached
            let debug_loader = DebugUtils::new(&entry, &instance);
            let messenger =
                unsafe { &debug_loader.create_debug_utils_messenger(&messenger_info, None) }
                    .unwrap();
            (debug_loader, messenger)
        } else {
            (
                DebugUtils::new(&entry, &instance),
                vk::DebugUtilsMessengerEXT::default(),
            )
        };

        //// Window surface creation
        let surface_loader = Surface::new(&entry, &instance);
        let surface = unsafe {
            ash_window::create_surface(
                &entry,
                &instance,
                window.raw_display_handle(),
                window.raw_window_handle(),
                None,
            )
        }
        .unwrap();

        //// Physical device and queues
        let (physical_device, queue_family_indices) =
        engine_core::find_physical_device(&instance, &surface_loader, &surface);

        //// Logical device
        let logical_device =
            engine_core::create_logical_device(&instance, &physical_device, queue_family_indices);
        let (graphics_queue, present_queue) =
            engine_core::get_queue_handles(&logical_device, queue_family_indices);

        //// Swapchain
        let swapchain_loader = Swapchain::new(&instance, &logical_device);
        let (swapchain, image_format, swapchain_extent, swapchain_images) =
            engine_core::create_swapchain(
                &window,
                &surface_loader,
                &surface,
                &physical_device,
                &swapchain_loader,
                queue_family_indices,
            );

        //// Image views
        let image_views = engine_core::create_swapchain_image_views(
            &logical_device,
            &swapchain_images,
            image_format,
        );

        //// Push constants
        let push_constants = [1.0];

        //// Graphics pipeline
        let (graphics_pipeline, graphics_pipeline_layout, descriptor_set_layout, render_pass) =
            engine_core::create_graphics_pipeline(
                &logical_device,
                swapchain_extent,
                image_format,
                &shaders,
                vertex_input_descriptors,
                descriptor_set_bindings,
                push_constants,
            );

        //// Depth image
        // Could check for supported formats for depth, but for now just going with D32_SFLOAT
        // https://vulkan-tutorial.com/en/Depth_buffering
        let depth_image = engine_core::create_image(
            &instance,
            &physical_device,
            &logical_device,
            vk::Format::D32_SFLOAT,
            vk::ImageTiling::OPTIMAL,
            vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
            vk::ImageAspectFlags::DEPTH,
            (swapchain_extent.width, swapchain_extent.height),
        );

        //// Framebuffers
        let framebuffers = engine_core::create_framebuffers(
            &logical_device,
            render_pass,
            swapchain_extent,
            &image_views,
            depth_image.image_view,
        );

        //// Command pool and buffers
        let command_pool_info = vk::CommandPoolCreateInfo::builder()
            .queue_family_index(queue_family_indices.graphics_queue)
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
        let command_pool = unsafe { logical_device.create_command_pool(&command_pool_info, None) }
            .expect("Could not create command pool!");


        // Create vertex and index buffers with ssbo usage in fragment stage
        let vertex_buffer = {
            let buffer = engine_core::buffer::create_buffer(
                &logical_device,
                (std::mem::size_of::<VertexType>() * vertices.len()) as u64,
                vk::BufferUsageFlags::VERTEX_BUFFER |vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::STORAGE_BUFFER
            );
            let buffer_mem = engine_core::buffer::allocate_and_bind_buffer(
                &instance,
                &physical_device,
                &logical_device,
                buffer,
                vk::MemoryPropertyFlags::DEVICE_LOCAL
            );
            engine_core::ManagedBuffer {
                logical_device: Rc::clone(&logical_device),
                buffer,
                buffer_memory: Some(buffer_mem),
                memory_ptr: None,
            }
        };
        let index_buffer = {
            let buffer = engine_core::buffer::create_buffer(
                &logical_device,
                (std::mem::size_of::<IndexType>() * indices.len()) as u64,
                vk::BufferUsageFlags::INDEX_BUFFER |vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::STORAGE_BUFFER
            );
            let buffer_mem = engine_core::buffer::allocate_and_bind_buffer(
                &instance,
                &physical_device,
                &logical_device,
                buffer,
                vk::MemoryPropertyFlags::DEVICE_LOCAL
            );
            engine_core::ManagedBuffer {
                logical_device: Rc::clone(&logical_device),
                buffer,
                buffer_memory: Some(buffer_mem),
                memory_ptr: None,
            }
        };

        // Copy over data via staging buffer
        {
            let mut staging_buffer = engine_core::create_staging_buffer(
                &instance,
                &physical_device,
                &logical_device,
                u64::max(
                    (std::mem::size_of::<VertexType>() * vertices.len()) as u64,
                    (std::mem::size_of::<IndexType>() * indices.len()) as u64
                )
            );
            staging_buffer.map_buffer_memory();

            unsafe {
                engine_core::write_vec_to_buffer(staging_buffer.memory_ptr.unwrap(), vertices)
            };
            engine_core::copy_buffer(
                &logical_device,
                command_pool,
                graphics_queue,
                *staging_buffer,
                *vertex_buffer,
                (std::mem::size_of::<VertexType>() * vertices.len()) as u64,
            );
            unsafe {
                engine_core::write_vec_to_buffer(staging_buffer.memory_ptr.unwrap(), indices)
            };
            engine_core::copy_buffer(
                &logical_device,
                command_pool,
                graphics_queue,
                *staging_buffer,
                *index_buffer,
                (std::mem::size_of::<IndexType>() * indices.len()) as u64,
            );
        }

        //// Uniform buffers
        let uniform_buffers = engine_core::create_uniform_buffers(
            &instance,
            &physical_device,
            &logical_device,
            std::mem::size_of::<UBOType>() as u64,
            engine_core::MAX_FRAMES_IN_FLIGHT,
        );

        //// Command buffers
        let command_buffers = engine_core::allocate_command_buffers(
            &logical_device,
            command_pool,
            image_views.len() as u32,
        );

        //// Descriptor pool
        let descriptor_pool = {
            let pool_sizes = [
                *vk::DescriptorPoolSize::builder()
                    .ty(vk::DescriptorType::UNIFORM_BUFFER)
                    .descriptor_count(engine_core::MAX_FRAMES_IN_FLIGHT as u32)
            ];
            let pool_info = vk::DescriptorPoolCreateInfo::builder()
                .pool_sizes(&pool_sizes)
                .max_sets(engine_core::MAX_FRAMES_IN_FLIGHT as u32);
            unsafe { logical_device.create_descriptor_pool(&pool_info, None) }
                .expect("Failed to create descriptor pool")
        };

        //// Descriptor sets
        let descriptor_sets = {
            let layouts = vec![descriptor_set_layout; engine_core::MAX_FRAMES_IN_FLIGHT];
            let alloc_info = vk::DescriptorSetAllocateInfo::builder()
                .descriptor_pool(descriptor_pool)
                .set_layouts(layouts.as_slice());
            unsafe { logical_device.allocate_descriptor_sets(&alloc_info) }
                .expect("Failed to allocate descriptor sets")
        };
        let descriptor_writes = {
            let mut v = Vec::with_capacity(descriptor_sets.len());
            for (i, set) in descriptor_sets.iter().enumerate() {
                let descriptor_buffer_info = [*vk::DescriptorBufferInfo::builder()
                    .buffer(*uniform_buffers[i])
                    .offset(0)
                    .range(std::mem::size_of::<UBOType>() as u64)];
                v.push(
                    *vk::WriteDescriptorSet::builder()
                        .dst_set(*set)
                        .dst_binding(0)
                        .dst_array_element(0)
                        .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                        .buffer_info(&descriptor_buffer_info),
                );
            }
            v
        };
        unsafe { logical_device.update_descriptor_sets(&descriptor_writes, &[]) }

        //// Create semaphores for in-render-pass synchronization
        let sync = engine_core::create_sync_primitives(&logical_device);

        LineLightApp {
            command_buffers,
            command_pool,
            uniform_buffers,
            graphics_queue,
            logical_device,
            present_queue,
            swapchain,
            swapchain_loader,
            sync,
            depth_image: ManuallyDrop::new(depth_image),
            index_buffer: ManuallyDrop::new(index_buffer),
            vertex_buffer: ManuallyDrop::new(vertex_buffer),
            descriptor_set_layout,
            descriptor_sets,
            descriptor_pool,
            framebuffers,
            graphics_pipeline,
            graphics_pipeline_layout,
            image_views,
            instance,
            render_pass,
            surface,
            surface_loader,
            swapchain_extent,
            window
        }
    }

    pub fn wait_for_in_flight_fence(&self, fence_index: usize) {
        let wait_fences = [self.sync.in_flight[fence_index]];
        unsafe {
            self.logical_device
                .wait_for_fences(&wait_fences, true, u64::MAX)
        }
        .unwrap();
    }
    pub fn acquire_next_image(
        &mut self,
        framebuffer_index: usize,
    ) -> Result<(u32, bool), vk::Result> {
        unsafe {
            self.swapchain_loader.acquire_next_image(
                self.swapchain,
                u64::MAX,
                self.sync.image_available[framebuffer_index],
                vk::Fence::null(),
            )
        }
    }
    pub fn reset_in_flight_fence(&self, fence_index: usize) {
        let wait_fences = [self.sync.in_flight[fence_index]];
        unsafe { self.logical_device.reset_fences(&wait_fences) }.unwrap();
    }
    pub unsafe fn record_command_buffer<F>(&mut self, buffer_index: usize, commands: F)
    where
        F: Fn(&mut LineLightApp),
    {
        //Begin recording command buffer
        let command_buffer_begin_info = vk::CommandBufferBeginInfo::builder();
        self.logical_device
            .begin_command_buffer(
                self.command_buffers[buffer_index],
                &command_buffer_begin_info,
            )
            .expect("Could not begin command buffer recording!");

        commands(self);

        self.logical_device
            .end_command_buffer(self.command_buffers[buffer_index])
            .expect("Failed recording command buffer!");
    }
    pub fn submit_drawing_command_buffer(&self, buffer_index: usize) {
        let wait_sems = [self.sync.image_available[buffer_index]];
        let wait_stages = [vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
        let signal_sems = [self.sync.render_finished[buffer_index]];
        let cmd_buffers = [self.command_buffers[buffer_index]];
        let submits = [*vk::SubmitInfo::builder()
            .wait_semaphores(&wait_sems)
            .wait_dst_stage_mask(&wait_stages)
            .command_buffers(&cmd_buffers)
            .signal_semaphores(&signal_sems)];
        unsafe {
            self.logical_device
                .queue_submit(
                    self.graphics_queue,
                    &submits,
                    self.sync.in_flight[buffer_index],
                )
                .expect("Queue submission failed!");
        }
    }
    pub fn present_image(
        &self,
        image_index: u32,
        wait_semaphore: vk::Semaphore,
    ) -> Result<bool, vk::Result> {
        let swapchain_arr = [self.swapchain];
        let image_index_arr = [image_index];
        let wait_semaphore_arr = [wait_semaphore];
        let present_info = vk::PresentInfoKHR::builder()
            .wait_semaphores(&wait_semaphore_arr)
            .swapchains(&swapchain_arr)
            .image_indices(&image_index_arr);
        unsafe {
            self.swapchain_loader
                .queue_present(self.present_queue, &present_info)
        }
    }

    pub fn recreate_swapchain(
        &mut self,
        shaders: &Vec<shaders::Shader>,
        vertex_input_descriptors: &vk_engine::VertexInputDescriptors,
        descriptor_set_bindings: Vec<vk::DescriptorSetLayoutBinding>,
    ) {
        unsafe {
            self.logical_device.device_wait_idle().unwrap();
            self.clean_swapchain_and_dependants();
        }

        let (physical_device, queue_family_indices) =
            engine_core::find_physical_device(&self.instance, &self.surface_loader, &self.surface);
        let (swapchain, image_format, swapchain_extent, swapchain_images) =
            engine_core::create_swapchain(
                &self.window,
                &self.surface_loader,
                &self.surface,
                &physical_device,
                &self.swapchain_loader,
                queue_family_indices,
            );
        let image_views = engine_core::create_swapchain_image_views(
            &self.logical_device,
            &swapchain_images,
            image_format,
        );
        let (graphics_pipeline, graphics_pipeline_layout, descriptor_set_layout, render_pass) =
            engine_core::create_graphics_pipeline(
                &self.logical_device,
                swapchain_extent,
                image_format,
                shaders,
                vertex_input_descriptors,
                descriptor_set_bindings,
                [0.0],
            );
        let depth_image = engine_core::create_image(
            &self.instance,
            &physical_device,
            &self.logical_device,
            vk::Format::D32_SFLOAT,
            vk::ImageTiling::OPTIMAL,
            vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
            vk::ImageAspectFlags::DEPTH,
            (swapchain_extent.width, swapchain_extent.height),
        );
        let framebuffers = engine_core::create_framebuffers(
            &self.logical_device,
            render_pass,
            swapchain_extent,
            &image_views,
            depth_image.image_view,
        );

        unsafe { ManuallyDrop::drop(&mut self.depth_image) };
        self.depth_image = ManuallyDrop::new(depth_image);

        self.swapchain = swapchain;
        self.swapchain_extent = swapchain_extent;
        self.image_views = image_views;
        self.render_pass = render_pass;
        self.graphics_pipeline = graphics_pipeline;
        self.graphics_pipeline_layout = graphics_pipeline_layout;
        self.descriptor_set_layout = descriptor_set_layout;
        self.framebuffers = framebuffers;
    }
    unsafe fn clean_swapchain_and_dependants(&mut self) {
        for buffer in self.framebuffers.drain(..) {
            self.logical_device.destroy_framebuffer(buffer, None);
        }
        self.logical_device
            .destroy_pipeline(self.graphics_pipeline, None);
        self.logical_device
            .destroy_pipeline_layout(self.graphics_pipeline_layout, None);
        self.logical_device
            .destroy_descriptor_set_layout(self.descriptor_set_layout, None);
        self.logical_device
            .destroy_render_pass(self.render_pass, None);
        for view in self.image_views.drain(..) {
            self.logical_device.destroy_image_view(view, None);
        }
        self.swapchain_loader
            .destroy_swapchain(self.swapchain, None);
    }
}