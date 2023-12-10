use ash::extensions::ext::DebugUtils;
use ash::extensions::khr::{Surface, Swapchain};
use ash::vk;
use cstr::cstr;
use glam::Vec3;
use raw_window_handle::{HasRawDisplayHandle, HasRawWindowHandle};
use std::ffi::CString;
use std::ffi::{c_char, CStr};
use std::mem::ManuallyDrop;
use std::rc::Rc;
use vk_engine::engine_core::ManagedBuffer;
use vk_engine::{engine_core, shaders};

use tracy_client::span;

use crate::datatypes::*;
use crate::scene_loading::*;
use crate::acceleration::AccelStruct;

static APP_NAME: &str = "Linelight Experiments";

impl Vertex {
    fn input_descriptors() -> vk_engine::VertexInputDescriptors {
        vk_engine::VertexInputDescriptors {
            attributes: vec![
                *vk::VertexInputAttributeDescription::builder()
                    .binding(0)
                    .location(0)
                    .format(vk::Format::R32G32B32_SFLOAT)
                    .offset(0),
                *vk::VertexInputAttributeDescription::builder()
                    .binding(0)
                    .location(1)
                    .format(vk::Format::R32G32B32_SFLOAT)
                    .offset(std::mem::size_of::<Vec3>() as u32),
            ],
            bindings: vec![*vk::VertexInputBindingDescription::builder()
                .binding(0)
                .input_rate(vk::VertexInputRate::VERTEX)
                .stride(std::mem::size_of::<Self>() as u32)],
        }
    }
}

pub fn make_shaders(vert_path: &str, frag_path: &str) -> Vec<vk_engine::shaders::Shader> {
    let _span = span!("make_shaders");
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
    vec![
        *vk::DescriptorSetLayoutBinding::builder()
            .binding(0)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT),
        *vk::DescriptorSetLayoutBinding::builder()
            .binding(1)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::FRAGMENT),
        *vk::DescriptorSetLayoutBinding::builder()
            .binding(2)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::FRAGMENT),
        *vk::DescriptorSetLayoutBinding::builder()
            .binding(3)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::FRAGMENT),
    ]
}

pub fn make_custom_app(
    shaders: &Vec<Vec<shaders::Shader>>,
    debug_shaders: &[vk_engine::shaders::Shader],
    ubo_bindings: &[vk::DescriptorSetLayoutBinding],
    scene: &Scene,
    acceleration_struct: &AccelStruct,
    acceleration_indices: &Vec<u32>,
) -> (
    LineLightApp,
    winit::event_loop::EventLoop<()>,
    vk_engine::VertexInputDescriptors,
    vk_engine::VertexInputDescriptors,
) {
    let _span = span!("make_app");

    let vid = Vertex::input_descriptors();
    let did = vk_engine::VertexInputDescriptors {
        bindings: vec![*vk::VertexInputBindingDescription::builder()
            .binding(0)
            .input_rate(vk::VertexInputRate::VERTEX)
            .stride(std::mem::size_of::<Vec3>() as u32)],
        attributes: vec![*vk::VertexInputAttributeDescription::builder()
            .binding(0)
            .location(0)
            .format(vk::Format::R32G32B32_SFLOAT)
            .offset(0)],
    };

    println!("Setting up window...");
    let (window, event_loop) = vk_engine::init_window(APP_NAME, 800, 600);

    println!("Initializing application...");
    let app = LineLightApp::new(
        window,
        shaders,
        debug_shaders,
        &vid,
        &did,
        ubo_bindings.clone(),
        &scene.vertices,
        &scene.indices,
        acceleration_struct,
        acceleration_indices,
    );

    app.update_descriptor_sets(
        scene.vertices.len() as u64,
        acceleration_indices.len() as u64,
    );
    (app, event_loop, vid, did)
}

pub struct LineLightApp {
    swapchain_loader: ash::extensions::khr::Swapchain,
    swapchain: vk::SwapchainKHR,
    pub swapchain_images: Vec<vk::Image>,
    pub swapchain_extent: vk::Extent2D,
    image_views: Vec<vk::ImageView>,
    pub framebuffers: Vec<vk::Framebuffer>,
    pub command_buffers: Vec<vk::CommandBuffer>,
    pub render_pass: vk::RenderPass,
    pub graphics_pipelines: Vec<vk::Pipeline>,
    pub graphics_pipeline_layout: vk::PipelineLayout,
    pub debug_pipeline: vk::Pipeline,
    pub debug_pipeline_layout: vk::PipelineLayout,
    pub debug_buffer: ManuallyDrop<engine_core::ManagedBuffer>,
    pub vertex_buffer: ManuallyDrop<engine_core::ManagedBuffer>,
    pub index_buffer: ManuallyDrop<engine_core::ManagedBuffer>,
    pub uniform_buffers: ManuallyDrop<Vec<vk_engine::engine_core::ManagedBuffer>>,
    pub acceleration_buffer: ManuallyDrop<engine_core::ManagedBuffer>,
    pub acceleration_index_buffer: ManuallyDrop<engine_core::ManagedBuffer>,
    depth_image: ManuallyDrop<engine_core::ManagedImage>,
    pub descriptor_sets: Vec<vk::DescriptorSet>,
    descriptor_set_layout: vk::DescriptorSetLayout,
    _descriptor_pool: vk::DescriptorPool,
    present_queue: vk::Queue,
    pub graphics_queue: vk::Queue,
    pub command_pool: vk::CommandPool,
    pub sync: engine_core::SyncPrims,
    surface_loader: ash::extensions::khr::Surface,
    surface: vk::SurfaceKHR,
    window: winit::window::Window,
    debug_loader: DebugUtils,
    debug_messenger: vk::DebugUtilsMessengerEXT,
    pub query_pool: vk::QueryPool,
    pub logical_device: Rc<ash::Device>,
    pub instance: Box<ash::Instance>,
}
impl Drop for LineLightApp {
    fn drop(&mut self) {
        unsafe {
            let _span = span!("cleanup");
            self.logical_device.device_wait_idle().unwrap();

            for i in 0..engine_core::MAX_FRAMES_IN_FLIGHT {
                self.logical_device
                    .destroy_semaphore(self.sync.image_available[i], None);
                self.logical_device
                    .destroy_semaphore(self.sync.render_finished[i], None);
                self.logical_device
                    .destroy_fence(self.sync.in_flight[i], None);
            }
            self.logical_device
                .destroy_descriptor_pool(self._descriptor_pool, None);
            self.logical_device
                .destroy_command_pool(self.command_pool, None);
            self.logical_device
                .destroy_query_pool(self.query_pool, None);

            ManuallyDrop::drop(&mut self.debug_buffer);
            ManuallyDrop::drop(&mut self.vertex_buffer);
            ManuallyDrop::drop(&mut self.index_buffer);
            ManuallyDrop::drop(&mut self.uniform_buffers);
            ManuallyDrop::drop(&mut self.acceleration_buffer);
            ManuallyDrop::drop(&mut self.acceleration_index_buffer);
            ManuallyDrop::drop(&mut self.depth_image);

            self.clean_swapchain_and_dependants();
            
            self.logical_device.destroy_device(None);
            self.surface_loader.destroy_surface(self.surface, None);

            if engine_core::VALIDATION_ENABLED {
                self.debug_loader
                    .destroy_debug_utils_messenger(self.debug_messenger, None);
            }

            self.instance.destroy_instance(None);
        }
    }
}

type VertexType = Vertex;
type IndexType = u32;
type UBOType = LineLightUniform;
impl LineLightApp {
    pub fn new(
        window: winit::window::Window,
        shaders: &Vec<Vec<shaders::Shader>>,
        debug_shaders: &[vk_engine::shaders::Shader],
        vertex_input_descriptors: &vk_engine::VertexInputDescriptors,
        debug_input_descriptors: &vk_engine::VertexInputDescriptors,
        descriptor_set_bindings: &[vk::DescriptorSetLayoutBinding],
        vertices: &Vec<Vertex>,
        indices: &Vec<u32>,
        acceleration_struct: &AccelStruct,
        acceleration_indices: &Vec<u32>,
    ) -> Self {
        let _span = span!("LineLightApp::new()");

        let entry = Box::new(unsafe { ash::Entry::load() }.unwrap());
        if engine_core::VALIDATION_ENABLED && !engine_core::check_validation_layer_support(&entry) {
            panic!("Validation layer requested but not available!");
        }

        //// Application info
        let app_name = CString::new(APP_NAME).unwrap();
        let engine_name = CString::new("Vulkan Engine").unwrap();

        let app_info = vk::ApplicationInfo::builder()
            .application_name(&app_name)
            .application_version(vk::make_api_version(0, 1, 1, 0))
            .engine_name(&engine_name)
            .engine_version(vk::API_VERSION_1_2)
            .api_version(vk::API_VERSION_1_2);

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
        let (debug_loader, debug_messenger) = if engine_core::VALIDATION_ENABLED {
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
                vk::PresentModeKHR::IMMEDIATE,
                queue_family_indices,
            );

        //// Image views
        let image_views = engine_core::create_swapchain_image_views(
            &logical_device,
            &swapchain_images,
            image_format,
        );

        //// Push constants
        let _push_constants = [1.0];

        //// Graphics pipeline
        let render_pass = render_pass(&logical_device, image_format);
        let mut graphics_pipelines = Vec::with_capacity(shaders.len());
        let (graphics_pipeline_layout, descriptor_set_layout) =
            pipeline_layouts(&logical_device, descriptor_set_bindings, [0.0]);
        for shader_group in shaders {
            let pipeline = main_pipeline(
                &logical_device,
                render_pass,
                swapchain_extent,
                shader_group,
                vertex_input_descriptors,
                graphics_pipeline_layout
            );
            graphics_pipelines.push(pipeline);
        }

        let (debug_pipeline, debug_pipeline_layout) = debug_pipeline(
            &logical_device,
            render_pass,
            swapchain_extent,
            debug_shaders,
            debug_input_descriptors,
            descriptor_set_layout,
            [0.0],
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
                vk::BufferUsageFlags::VERTEX_BUFFER
                    | vk::BufferUsageFlags::TRANSFER_DST
                    | vk::BufferUsageFlags::STORAGE_BUFFER,
            );
            let buffer_mem = engine_core::buffer::allocate_and_bind_buffer(
                &instance,
                &physical_device,
                &logical_device,
                buffer,
                vk::MemoryPropertyFlags::DEVICE_LOCAL,
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
                vk::BufferUsageFlags::INDEX_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
            );
            let buffer_mem = engine_core::buffer::allocate_and_bind_buffer(
                &instance,
                &physical_device,
                &logical_device,
                buffer,
                vk::MemoryPropertyFlags::DEVICE_LOCAL,
            );
            engine_core::ManagedBuffer {
                logical_device: Rc::clone(&logical_device),
                buffer,
                buffer_memory: Some(buffer_mem),
                memory_ptr: None,
            }
        };

        let acceleration_index_buffer = {
            let buffer = engine_core::buffer::create_buffer(
                &logical_device,
                (std::mem::size_of::<IndexType>() * acceleration_indices.len()) as u64,
                vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
            );
            let buffer_mem = engine_core::buffer::allocate_and_bind_buffer(
                &instance,
                &physical_device,
                &logical_device,
                buffer,
                vk::MemoryPropertyFlags::DEVICE_LOCAL,
            );
            engine_core::ManagedBuffer {
                logical_device: Rc::clone(&logical_device),
                buffer,
                buffer_memory: Some(buffer_mem),
                memory_ptr: None,
            }
        };
        let acceleration_buffer = {
            let buffer = engine_core::buffer::create_buffer(
                &logical_device,
                std::mem::size_of::<AccelStruct>() as u64,
                vk::BufferUsageFlags::UNIFORM_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
            );
            let buffer_mem = engine_core::buffer::allocate_and_bind_buffer(
                &instance,
                &physical_device,
                &logical_device,
                buffer,
                vk::MemoryPropertyFlags::DEVICE_LOCAL,
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
                [
                    std::mem::size_of::<VertexType>() * vertices.len(),
                    std::mem::size_of::<IndexType>() * indices.len(),
                    std::mem::size_of::<IndexType>() * acceleration_indices.len(),
                    std::mem::size_of::<AccelStruct>(),
                ].into_iter().max().unwrap() as u64
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
            unsafe {
                engine_core::write_vec_to_buffer(
                    staging_buffer.memory_ptr.unwrap(),
                    acceleration_indices,
                )
            };
            engine_core::copy_buffer(
                &logical_device,
                command_pool,
                graphics_queue,
                *staging_buffer,
                *acceleration_index_buffer,
                (std::mem::size_of::<IndexType>() * acceleration_indices.len()) as u64,
            );
            unsafe {
                engine_core::write_struct_to_buffer(
                    staging_buffer.memory_ptr.unwrap(),
                    acceleration_struct,
                )
            };
            engine_core::copy_buffer(
                &logical_device,
                command_pool,
                graphics_queue,
                *staging_buffer,
                *acceleration_buffer,
                std::mem::size_of::<AccelStruct>() as u64,
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

        // Debug vertex buffers
        const DEBUG_BUFFER_SIZE: u64 = 1<<22; // Should be large enough to contain *anything* I want to draw
        let mut debug_buffer = {
            let buffer = engine_core::buffer::create_buffer(
                &logical_device,
                DEBUG_BUFFER_SIZE,
                vk::BufferUsageFlags::VERTEX_BUFFER | vk::BufferUsageFlags::STORAGE_BUFFER,
            );
            let buffer_mem = engine_core::buffer::allocate_and_bind_buffer(
                &instance,
                &physical_device,
                &logical_device,
                buffer,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            );
            engine_core::ManagedBuffer {
                logical_device: Rc::clone(&logical_device),
                buffer,
                buffer_memory: Some(buffer_mem),
                memory_ptr: None,
            }
        };
        debug_buffer.map_buffer_memory();

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
                    .descriptor_count(engine_core::MAX_FRAMES_IN_FLIGHT as u32),
                *vk::DescriptorPoolSize::builder()
                    .ty(vk::DescriptorType::UNIFORM_BUFFER)
                    .descriptor_count(engine_core::MAX_FRAMES_IN_FLIGHT as u32),
                *vk::DescriptorPoolSize::builder()
                    .ty(vk::DescriptorType::STORAGE_BUFFER)
                    .descriptor_count(engine_core::MAX_FRAMES_IN_FLIGHT as u32),
                *vk::DescriptorPoolSize::builder()
                    .ty(vk::DescriptorType::STORAGE_BUFFER)
                    .descriptor_count(engine_core::MAX_FRAMES_IN_FLIGHT as u32),
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

        let query_pool = {
            let timestamp_pool_create_info = vk::QueryPoolCreateInfo::builder()
                .query_count(2)
                .query_type(vk::QueryType::TIMESTAMP)
                .pipeline_statistics(vk::QueryPipelineStatisticFlags::empty());
            unsafe { logical_device.create_query_pool(&timestamp_pool_create_info, None) }.unwrap()
        };

        LineLightApp {
            command_buffers,
            command_pool,
            graphics_queue,
            logical_device,
            present_queue,
            swapchain,
            swapchain_images,
            swapchain_loader,
            sync,
            uniform_buffers: ManuallyDrop::new(uniform_buffers),
            depth_image: ManuallyDrop::new(depth_image),
            index_buffer: ManuallyDrop::new(index_buffer),
            vertex_buffer: ManuallyDrop::new(vertex_buffer),
            debug_buffer: ManuallyDrop::new(debug_buffer),
            acceleration_buffer: ManuallyDrop::new(acceleration_buffer),
            acceleration_index_buffer: ManuallyDrop::new(acceleration_index_buffer),
            descriptor_set_layout,
            descriptor_sets,
            _descriptor_pool: descriptor_pool,
            framebuffers,
            graphics_pipelines,
            graphics_pipeline_layout,
            debug_pipeline,
            debug_pipeline_layout,
            query_pool,
            image_views,
            render_pass,
            surface,
            surface_loader,
            swapchain_extent,
            window,
            debug_loader,
            debug_messenger,
            instance,
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
        &mut self,
        image_index: u32,
        wait_semaphore: vk::Semaphore,
        // buffer_index: usize
    ) -> Result<bool, vk::Result> {
        // let barrier = vk::ImageMemoryBarrier::builder()
        //     .old_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
        //     .new_layout(vk::ImageLayout::PRESENT_SRC_KHR)
        //     .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        //     .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        //     .image(self.swapchain_images[image_index as usize])
        //     .subresource_range(
        //         *vk::ImageSubresourceRange::builder()
        //             .aspect_mask(vk::ImageAspectFlags::COLOR)
        //             .base_mip_level(0)
        //             .level_count(1)
        //             .base_array_layer(0)
        //             .layer_count(1),
        //     )
        //     .src_access_mask(vk::AccessFlags::empty())
        //     .dst_access_mask(vk::AccessFlags::empty());

        // let src_stage = vk::PipelineStageFlags::TOP_OF_PIPE;
        // let dst_stage = vk::PipelineStageFlags::TRANSFER;

        // let cmd_buffer = [self.command_buffers[buffer_index]];

        // unsafe {
        //     self.logical_device.queue_wait_idle(self.graphics_queue).unwrap();
        //     self.record_command_buffer(buffer_index, |app| {
        //         app.logical_device.cmd_pipeline_barrier(
        //             cmd_buffer[0],
        //             src_stage,
        //             dst_stage,
        //             vk::DependencyFlags::empty(),
        //             &[],
        //             &[],
        //             &[*barrier],
        //         );
        //     });
        //     let submit_info = vk::SubmitInfo::builder().command_buffers(&cmd_buffer);
        //     self.logical_device.queue_submit(self.graphics_queue, &[*submit_info], vk::Fence::null()).unwrap();
        //     self.logical_device.queue_wait_idle(self.graphics_queue).unwrap();
        // }

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
    pub fn update_descriptor_sets(&self, num_verts: u64, num_acceleration_indices: u64) {
        let descriptor_writes = {
            let mut v = Vec::with_capacity(self.descriptor_sets.len());
            for (i, set) in self.descriptor_sets.iter().enumerate() {
                let descriptor_buffer_info = [*vk::DescriptorBufferInfo::builder()
                    .buffer(*self.uniform_buffers[i])
                    .offset(0)
                    .range(std::mem::size_of::<UBOType>() as u64)];
                let descriptor_reused_vert_buffer_info = [*vk::DescriptorBufferInfo::builder()
                    .buffer(**self.vertex_buffer)
                    .offset(0)
                    .range(std::mem::size_of::<VertexType>() as u64 * num_verts)];
                let descriptor_acceleration_buffer_info =
                    [*vk::DescriptorBufferInfo::builder()
                        .buffer(**self.acceleration_buffer)
                        .offset(0)
                        .range(std::mem::size_of::<AccelStruct>() as u64)];
                let descriptor_acceleration_index_buffer_info =
                    [*vk::DescriptorBufferInfo::builder()
                        .buffer(**self.acceleration_index_buffer)
                        .offset(0)
                        .range(std::mem::size_of::<IndexType>() as u64 * num_acceleration_indices)];
                v.push(
                    *vk::WriteDescriptorSet::builder()
                        .dst_set(*set)
                        .dst_binding(0)
                        .dst_array_element(0)
                        .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                        .buffer_info(&descriptor_buffer_info),
                );
                v.push(
                    *vk::WriteDescriptorSet::builder()
                        .dst_set(*set)
                        .dst_binding(1)
                        .dst_array_element(0)
                        .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                        .buffer_info(&descriptor_acceleration_buffer_info),
                );
                v.push(
                    *vk::WriteDescriptorSet::builder()
                        .dst_set(*set)
                        .dst_binding(2)
                        .dst_array_element(0)
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .buffer_info(&descriptor_reused_vert_buffer_info),
                );
                v.push(
                    *vk::WriteDescriptorSet::builder()
                        .dst_set(*set)
                        .dst_binding(3)
                        .dst_array_element(0)
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .buffer_info(&descriptor_acceleration_index_buffer_info),
                );
            }
            v
        };
        unsafe {
            self.logical_device
                .update_descriptor_sets(&descriptor_writes, &[])
        }
    }

    pub fn make_screenshot_buffer(&self) -> (ManagedBuffer, u64) {
        let (physical_device, _) =
            engine_core::find_physical_device(&self.instance, &self.surface_loader, &self.surface);
        let memory_size =
            4 * self.swapchain_extent.width as u64 * self.swapchain_extent.height as u64;
        let buffer = engine_core::buffer::create_buffer(
            &self.logical_device,
            memory_size,
            vk::BufferUsageFlags::TRANSFER_DST,
        );
        let memory = engine_core::buffer::allocate_and_bind_buffer(
            &self.instance,
            &physical_device,
            &self.logical_device,
            buffer,
            vk::MemoryPropertyFlags::HOST_VISIBLE,
        );
        let mut screenshot_buffer = engine_core::ManagedBuffer {
            buffer,
            logical_device: Rc::clone(&self.logical_device),
            buffer_memory: Some(memory),
            memory_ptr: None,
        };
        screenshot_buffer.map_buffer_memory();
        (screenshot_buffer, memory_size)
    }

    pub fn recreate_swapchain(
        &mut self,
        shaders: &Vec<Vec<shaders::Shader>>,
        debug_shaders: &[shaders::Shader],
        vertex_input_descriptors: &vk_engine::VertexInputDescriptors,
        debug_input_descriptors: &vk_engine::VertexInputDescriptors,
        descriptor_set_bindings: &[vk::DescriptorSetLayoutBinding],
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
                vk::PresentModeKHR::IMMEDIATE,
                queue_family_indices,
            );
        let image_views = engine_core::create_swapchain_image_views(
            &self.logical_device,
            &swapchain_images,
            image_format,
        );
        let render_pass = render_pass(&self.logical_device, image_format);
        let mut graphics_pipelines = Vec::with_capacity(shaders.len());
        let (graphics_pipeline_layout, descriptor_set_layout) =
            pipeline_layouts(&self.logical_device, descriptor_set_bindings, [0.0]);
        for shader_group in shaders {
            let pipeline = main_pipeline(
                &self.logical_device,
                render_pass,
                swapchain_extent,
                shader_group,
                vertex_input_descriptors,
                graphics_pipeline_layout
            );
            graphics_pipelines.push(pipeline);
        }
        let (debug_pipeline, debug_pipeline_layout) = debug_pipeline(
            &self.logical_device,
            render_pass,
            swapchain_extent,
            debug_shaders,
            debug_input_descriptors,
            descriptor_set_layout,
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
        self.graphics_pipelines = graphics_pipelines;
        self.graphics_pipeline_layout = graphics_pipeline_layout;
        self.debug_pipeline = debug_pipeline;
        self.debug_pipeline_layout = debug_pipeline_layout;
        self.descriptor_set_layout = descriptor_set_layout;
        self.framebuffers = framebuffers;
    }
    unsafe fn clean_swapchain_and_dependants(&mut self) {
        for pipeline in self.graphics_pipelines.iter() {
            self.logical_device.destroy_pipeline(*pipeline, None);
        }
        self.logical_device
            .destroy_pipeline_layout(self.graphics_pipeline_layout, None);
        self.logical_device
            .destroy_pipeline(self.debug_pipeline, None);
        self.logical_device
            .destroy_pipeline_layout(self.debug_pipeline_layout, None);
        self.logical_device
            .destroy_descriptor_set_layout(self.descriptor_set_layout, None);
        self.logical_device
            .destroy_render_pass(self.render_pass, None);
        for buffer in self.framebuffers.drain(..) {
            self.logical_device.destroy_framebuffer(buffer, None);
        }
        for view in self.image_views.drain(..) {
            self.logical_device.destroy_image_view(view, None);
        }
        self.swapchain_loader
            .destroy_swapchain(self.swapchain, None);
    }

    #[cfg(feature="gpu_trace")]
    pub fn reset_timestamps(&self, command_buffer: vk::CommandBuffer) {
        let recording_info = vk::CommandBufferBeginInfo::builder()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        let submit_info = *vk::SubmitInfo::builder().command_buffers(&[command_buffer]);
        unsafe {
            self.logical_device
                .begin_command_buffer(command_buffer, &recording_info)
                .unwrap();
            self.logical_device
                .cmd_reset_query_pool(command_buffer, self.query_pool, 0, 2);
            self.logical_device
                .end_command_buffer(command_buffer)
                .unwrap();
            self.logical_device
                .queue_submit(self.graphics_queue, &[submit_info], vk::Fence::null())
                .unwrap();
            self.logical_device
                .queue_wait_idle(self.graphics_queue)
                .unwrap();
        }
    }
    #[cfg(feature="gpu_trace")]
    pub fn get_timestamp_immediately(&self) -> i64 {
        let mut timestamp: [i64; 1] = [0];
        unsafe {
            engine_core::immediate_commands(
                &self.logical_device,
                self.command_pool,
                self.graphics_queue,
                |buffer| {
                    // self.logical_device.cmd_reset_query_pool(buffer, self.query_pool, 0, 2);
                    self.logical_device.cmd_write_timestamp(
                        buffer,
                        vk::PipelineStageFlags::TOP_OF_PIPE,
                        self.query_pool,
                        0,
                    );
                },
            );
            self.logical_device
                .get_query_pool_results(
                    self.query_pool,
                    0,
                    1,
                    &mut timestamp,
                    vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WAIT,
                )
                .expect("Couldn't get timestamp")
        }

        timestamp[0]
    }
    #[cfg(feature="gpu_trace")]
    pub fn get_timestamp_period(&self) -> f32 {
        let (physical_device, _) =
            engine_core::find_physical_device(&self.instance, &self.surface_loader, &self.surface);
        let props = unsafe {
            self.instance
                .get_physical_device_properties(physical_device)
        };
        props.limits.timestamp_period
    }
}

fn render_pass(logical_device: &ash::Device, image_format: vk::Format) -> vk::RenderPass {
    let color_attachments = [*vk::AttachmentDescription::builder()
        .format(image_format)
        .samples(vk::SampleCountFlags::TYPE_1)
        .load_op(vk::AttachmentLoadOp::CLEAR)
        .store_op(vk::AttachmentStoreOp::STORE)
        .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
        .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
        .initial_layout(vk::ImageLayout::UNDEFINED)
        .final_layout(vk::ImageLayout::PRESENT_SRC_KHR)];
    let depth_attachments = [*vk::AttachmentDescription::builder()
        .format(vk::Format::D32_SFLOAT)
        .samples(vk::SampleCountFlags::TYPE_1)
        .load_op(vk::AttachmentLoadOp::CLEAR)
        .store_op(vk::AttachmentStoreOp::DONT_CARE)
        .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
        .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
        .initial_layout(vk::ImageLayout::UNDEFINED)
        .final_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)];
    let attachments = [color_attachments, depth_attachments].concat();

    // Subpass
    let main_subpass = vk::SubpassDependency::builder()
        .src_subpass(vk::SUBPASS_EXTERNAL)
        .dst_subpass(0)
        .src_stage_mask(
            vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT
                | vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS,
        )
        .dst_stage_mask(
            vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT
                | vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS,
        )
        .src_access_mask(vk::AccessFlags::empty())
        .dst_access_mask(
            vk::AccessFlags::COLOR_ATTACHMENT_WRITE
                | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
        );
    let debug_subpass = vk::SubpassDependency::builder()
        .src_subpass(0)
        .dst_subpass(1)
        .src_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
        .dst_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
        .src_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_READ)
        .dst_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)
        .dependency_flags(vk::DependencyFlags::BY_REGION);
    let dependencies = [*main_subpass, *debug_subpass];
    let color_attachment_refs = [*vk::AttachmentReference::builder()
        .attachment(0) //First attachment in array -> color_attachment
        .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)];
    let depth_attachment_ref = vk::AttachmentReference::builder()
        .attachment(1)
        .layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL);
    let subpasses = [
        *vk::SubpassDescription::builder()
            .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
            .color_attachments(&color_attachment_refs)
            .depth_stencil_attachment(&depth_attachment_ref),
        *vk::SubpassDescription::builder()
            .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
            .color_attachments(&color_attachment_refs),
    ];

    let renderpass_info = vk::RenderPassCreateInfo::builder()
        .attachments(&attachments)
        .subpasses(&subpasses)
        .dependencies(&dependencies);

    unsafe { logical_device.create_render_pass(&renderpass_info, None) }
        .expect("Failed to create renderpass!")
}

fn main_pipeline(
    logical_device: &ash::Device,
    render_pass: vk::RenderPass,
    swapchain_extent: vk::Extent2D,
    shaders: &[shaders::Shader],
    vertex_input_descriptors: &vk_engine::VertexInputDescriptors,
    pipeline_layout: vk::PipelineLayout
) -> vk::Pipeline {
    // Vertex input settings
    let binding_descriptions = &vertex_input_descriptors.bindings;
    let attribute_descriptions = &vertex_input_descriptors.attributes;
    let pipeline_vertex_input_state_info = vk::PipelineVertexInputStateCreateInfo::builder()
        .vertex_binding_descriptions(binding_descriptions.as_slice())
        .vertex_attribute_descriptions(attribute_descriptions.as_slice());
    // Input assembly settings
    let pipeline_input_assembly_state_info = vk::PipelineInputAssemblyStateCreateInfo::builder()
        .topology(vk::PrimitiveTopology::TRIANGLE_LIST)
        .primitive_restart_enable(false);
    // Viewport settings
    let viewports = [*vk::Viewport::builder()
        .x(0.0)
        .y(0.0)
        .width(swapchain_extent.width as f32)
        .height(swapchain_extent.height as f32)
        .min_depth(0.0)
        .max_depth(1.0)];
    let scissor_rects = [*vk::Rect2D::builder()
        .offset(vk::Offset2D { x: 0, y: 0 })
        .extent(swapchain_extent)];
    let pipeline_viewport_state_info = vk::PipelineViewportStateCreateInfo::builder()
        .viewports(&viewports)
        .scissors(&scissor_rects);
    // Rasterizer settings
    let pipeline_rasterization_state_info = vk::PipelineRasterizationStateCreateInfo::builder()
        .depth_clamp_enable(false)
        .rasterizer_discard_enable(false)
        .polygon_mode(vk::PolygonMode::FILL)
        .line_width(1.0)
        .cull_mode(vk::CullModeFlags::BACK)
        .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
        .depth_bias_enable(false);
    // Multisampling settings
    let pipeline_multisample_state_info = vk::PipelineMultisampleStateCreateInfo::builder()
        .sample_shading_enable(false)
        .rasterization_samples(vk::SampleCountFlags::TYPE_1);
    // Color blending settings
    let pipeline_color_blend_attachment_states =
        [*vk::PipelineColorBlendAttachmentState::builder()
            .color_write_mask(
                vk::ColorComponentFlags::R
                    | vk::ColorComponentFlags::G
                    | vk::ColorComponentFlags::B
                    | vk::ColorComponentFlags::A,
            )
            .blend_enable(false)];
    let pipeline_color_blend_state_info = vk::PipelineColorBlendStateCreateInfo::builder()
        .logic_op_enable(false)
        .attachments(&pipeline_color_blend_attachment_states);

    let shader_module_vec = shaders
        .iter()
        .map(|shader| create_shader_module(logical_device, shader))
        .collect::<Vec<(vk::ShaderModule, vk::PipelineShaderStageCreateInfo)>>();
    let shader_modules = shader_module_vec.as_slice();
    let shader_stages: Vec<vk::PipelineShaderStageCreateInfo> =
        shader_modules.iter().map(|pair| pair.1).collect();

    let depth_stencil_info = vk::PipelineDepthStencilStateCreateInfo::builder()
        .depth_test_enable(true)
        .depth_write_enable(true)
        .depth_compare_op(vk::CompareOp::LESS)
        .depth_bounds_test_enable(false)
        .min_depth_bounds(0.0)
        .max_depth_bounds(1.0)
        .stencil_test_enable(false);

    let graphics_pipeline_infos = [*vk::GraphicsPipelineCreateInfo::builder()
        .stages(&shader_stages)
        .vertex_input_state(&pipeline_vertex_input_state_info)
        .input_assembly_state(&pipeline_input_assembly_state_info)
        .viewport_state(&pipeline_viewport_state_info)
        .rasterization_state(&pipeline_rasterization_state_info)
        .multisample_state(&pipeline_multisample_state_info)
        .color_blend_state(&pipeline_color_blend_state_info)
        .depth_stencil_state(&depth_stencil_info)
        .layout(pipeline_layout)
        .render_pass(render_pass)
        .subpass(0)];
    let graphics_pipeline = unsafe {
        logical_device.create_graphics_pipelines(
            vk::PipelineCache::null(),
            &graphics_pipeline_infos,
            None,
        )
    }
    .unwrap()[0];

    //Once the graphics pipeline has been created, the SPIR-V bytecode is compiled into the pipeline itself
    //The shader modules can therefore already be destroyed
    unsafe {
        for module in shader_modules {
            logical_device.destroy_shader_module(module.0, None)
        }
    }
    graphics_pipeline
}

fn pipeline_layouts(
    logical_device: &ash::Device,
    descriptor_set_bindings: &[vk::DescriptorSetLayoutBinding],
    push_constants: [f32; 1],
) -> (vk::PipelineLayout, vk::DescriptorSetLayout) {
    // Descriptor set layout
    let descriptor_set_layout = {
        let descriptor_set_layout_info =
            vk::DescriptorSetLayoutCreateInfo::builder().bindings(descriptor_set_bindings);

        unsafe { logical_device.create_descriptor_set_layout(&descriptor_set_layout_info, None) }
            .unwrap()
    };
    // Pipeline layout
    let push_constant_ranges = [*vk::PushConstantRange::builder()
        .stage_flags(vk::ShaderStageFlags::VERTEX)
        .offset(0)
        .size((push_constants.len() * std::mem::size_of::<f32>()) as u32)];
    let mut pipeline_layout_info =
        vk::PipelineLayoutCreateInfo::builder().push_constant_ranges(&push_constant_ranges);
    let pipeline_layout = {
        let layout = [descriptor_set_layout];
        pipeline_layout_info = pipeline_layout_info.set_layouts(&layout);
        unsafe { logical_device.create_pipeline_layout(&pipeline_layout_info, None) }.unwrap()
    };

    (pipeline_layout, descriptor_set_layout)
}

const DEBUG_SUBPASS_IDX: u32 = 1;
fn debug_pipeline(
    logical_device: &ash::Device,
    render_pass: vk::RenderPass,
    swapchain_extent: vk::Extent2D,
    shaders: &[shaders::Shader],
    vertex_input_descriptors: &vk_engine::VertexInputDescriptors,
    descriptor_set_layout: vk::DescriptorSetLayout,
    push_constants: [f32; 1],
) -> (vk::Pipeline, vk::PipelineLayout) {
    // Vertex input settings
    let binding_descriptions = &vertex_input_descriptors.bindings;
    let attribute_descriptions = &vertex_input_descriptors.attributes;
    let pipeline_vertex_input_state_info = vk::PipelineVertexInputStateCreateInfo::builder()
        .vertex_binding_descriptions(binding_descriptions.as_slice())
        .vertex_attribute_descriptions(attribute_descriptions.as_slice());
    // Input assembly settings
    let pipeline_input_assembly_state_info = vk::PipelineInputAssemblyStateCreateInfo::builder()
        .topology(vk::PrimitiveTopology::LINE_LIST)
        .primitive_restart_enable(false);
    // Viewport settings
    let viewports = [*vk::Viewport::builder()
        .x(0.0)
        .y(0.0)
        .width(swapchain_extent.width as f32)
        .height(swapchain_extent.height as f32)
        .min_depth(0.0)
        .max_depth(1.0)];
    let scissor_rects = [*vk::Rect2D::builder()
        .offset(vk::Offset2D { x: 0, y: 0 })
        .extent(swapchain_extent)];
    let pipeline_viewport_state_info = vk::PipelineViewportStateCreateInfo::builder()
        .viewports(&viewports)
        .scissors(&scissor_rects);
    // Rasterizer settings
    let pipeline_rasterization_state_info = vk::PipelineRasterizationStateCreateInfo::builder()
        .depth_clamp_enable(false)
        .rasterizer_discard_enable(false)
        .polygon_mode(vk::PolygonMode::FILL)
        .line_width(1.0)
        .cull_mode(vk::CullModeFlags::BACK)
        .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
        .depth_bias_enable(false);
    // Multisampling settings
    let pipeline_multisample_state_info = vk::PipelineMultisampleStateCreateInfo::builder()
        .sample_shading_enable(false)
        .rasterization_samples(vk::SampleCountFlags::TYPE_1);
    // Color blending settings
    let pipeline_color_blend_attachment_states =
        [*vk::PipelineColorBlendAttachmentState::builder()
            .color_write_mask(
                vk::ColorComponentFlags::R
                    | vk::ColorComponentFlags::G
                    | vk::ColorComponentFlags::B
                    | vk::ColorComponentFlags::A,
            )
            .blend_enable(false)];
    let pipeline_color_blend_state_info = vk::PipelineColorBlendStateCreateInfo::builder()
        .logic_op_enable(false)
        .attachments(&pipeline_color_blend_attachment_states);

    // Pipeline layout
    let push_constant_ranges = [*vk::PushConstantRange::builder()
        .stage_flags(vk::ShaderStageFlags::VERTEX)
        .offset(0)
        .size((push_constants.len() * std::mem::size_of::<f32>()) as u32)];

    let mut pipeline_layout_info =
        vk::PipelineLayoutCreateInfo::builder().push_constant_ranges(&push_constant_ranges);
    let pipeline_layout = {
        let layout = [descriptor_set_layout];
        pipeline_layout_info = pipeline_layout_info.set_layouts(&layout);
        unsafe { logical_device.create_pipeline_layout(&pipeline_layout_info, None) }.unwrap()
    };

    let shader_module_vec = shaders
        .iter()
        .map(|shader| create_shader_module(logical_device, shader))
        .collect::<Vec<(vk::ShaderModule, vk::PipelineShaderStageCreateInfo)>>();
    let shader_modules = shader_module_vec.as_slice();

    let shader_stages: Vec<vk::PipelineShaderStageCreateInfo> =
        shader_modules.iter().map(|pair| pair.1).collect();

    // let depth_stencil_info = vk::PipelineDepthStencilStateCreateInfo::builder()
    //     .depth_test_enable(true)
    //     .depth_write_enable(true)
    //     .depth_compare_op(vk::CompareOp::LESS)
    //     .depth_bounds_test_enable(false)
    //     .min_depth_bounds(0.0)
    //     .max_depth_bounds(1.0)
    //     .stencil_test_enable(false);

    let debug_pipeline_infos = [*vk::GraphicsPipelineCreateInfo::builder()
        .stages(&shader_stages)
        .vertex_input_state(&pipeline_vertex_input_state_info)
        .input_assembly_state(&pipeline_input_assembly_state_info)
        .viewport_state(&pipeline_viewport_state_info)
        .rasterization_state(&pipeline_rasterization_state_info)
        .multisample_state(&pipeline_multisample_state_info)
        .color_blend_state(&pipeline_color_blend_state_info)
        // .depth_stencil_state(&depth_stencil_info)
        .layout(pipeline_layout)
        .render_pass(render_pass)
        .subpass(DEBUG_SUBPASS_IDX)];
    let debug_pipeline = unsafe {
        logical_device.create_graphics_pipelines(
            vk::PipelineCache::null(),
            &debug_pipeline_infos,
            None,
        )
    }
    .unwrap()[0];

    //Once the graphics pipeline has been created, the SPIR-V bytecode is compiled into the pipeline itself
    //The shader modules can therefore already be destroyed
    unsafe {
        for module in shader_modules {
            logical_device.destroy_shader_module(module.0, None)
        }
    }

    (debug_pipeline, pipeline_layout)
}

const DEFAULT_ENTRY: *const c_char = cstr!("main").as_ptr();
fn create_shader_module(
    logical_device: &ash::Device,
    shader: &shaders::Shader,
) -> (vk::ShaderModule, vk::PipelineShaderStageCreateInfo) {
    let entry_point = unsafe { CStr::from_ptr(DEFAULT_ENTRY) };
    let shader_stage_flag = match shader.shader_type {
        shaders::ShaderType::Vertex => vk::ShaderStageFlags::VERTEX,
        shaders::ShaderType::Fragment => vk::ShaderStageFlags::FRAGMENT,
    };

    let decoded = &shader.data;
    let shader_module_info = vk::ShaderModuleCreateInfo::builder().code(decoded);
    let shader_module =
        unsafe { logical_device.create_shader_module(&shader_module_info, None) }.unwrap();
    let stage_info = vk::PipelineShaderStageCreateInfo::builder()
        .stage(shader_stage_flag)
        .module(shader_module)
        .name(entry_point);

    (shader_module, *stage_info)
}
