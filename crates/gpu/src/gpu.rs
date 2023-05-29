use std::sync::Arc;

use ambient_std::asset_cache::SyncAssetKey;
use bytemuck::{Pod, Zeroable};
use glam::{uvec2, UVec2, UVec3, UVec4, Vec2, Vec3, Vec4};
use wgpu::{InstanceDescriptor, PresentMode, TextureFormat};
use winit::window::Window;

use crate::settings::Settings;

// #[cfg(debug_assertions)]
pub const DEFAULT_SAMPLE_COUNT: u32 = 1;
// #[cfg(not(debug_assertions))]
// pub const DEFAULT_SAMPLE_COUNT: u32 = 4;

#[derive(Debug)]
pub struct GpuKey;
impl SyncAssetKey<Arc<Gpu>> for GpuKey {}

#[derive(Debug)]
pub struct Gpu {
    pub surface: Option<wgpu::Surface>,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub swapchain_format: Option<TextureFormat>,
    pub swapchain_mode: Option<PresentMode>,
    pub adapter: wgpu::Adapter,
    /// If this is true, we don't need to use blocking device.polls, since they are assumed to be polled elsewhere
    pub will_be_polled: bool,
}
impl Gpu {
    pub async fn new(window: Option<&Window>) -> Self {
        Self::with_config(window, false, &Settings::default()).await
    }
    #[tracing::instrument(level = "info")]
    pub async fn with_config(
        window: Option<&Window>,
        will_be_polled: bool,
        settings: &Settings,
    ) -> Self {
        // From: https://github.com/KhronosGroup/Vulkan-Loader/issues/552
        #[cfg(not(target_os = "unknown"))]
        {
            std::env::set_var("DISABLE_LAYER_AMD_SWITCHABLE_GRAPHICS_1", "1");
            std::env::set_var("DISABLE_LAYER_NV_OPTIMUS_1", "1");
        }

        #[cfg(target_os = "windows")]
        let backend = wgpu::Backends::VULKAN;

        #[cfg(all(not(target_os = "windows"), not(target_os = "unknown")))]
        let backend = wgpu::Backends::PRIMARY;

        #[cfg(target_os = "unknown")]
        let backend = wgpu::Backends::all();

        let instance = wgpu::Instance::new(InstanceDescriptor {
            backends: backend,
            // TODO upgrade to Dxc ?
            // https://docs.rs/wgpu/latest/wgpu/enum.Dx12Compiler.html
            dx12_shader_compiler: wgpu::Dx12Compiler::Fxc,
        });
        let surface = window.map(|window| unsafe { instance.create_surface(window).unwrap() });
        #[cfg(not(target_os = "unknown"))]
        {
            tracing::debug!("Available adapters:");
            for adapter in instance.enumerate_adapters(wgpu::Backends::PRIMARY) {
                tracing::debug!("Adapter: {:?}", adapter.get_info());
            }
        }

        tracing::debug!("Requesting adapter");
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: surface.as_ref(),
                force_fallback_adapter: false,
            })
            .await
            .expect("Failed to find an appropriate adapter");

        tracing::info!("Using gpu adapter: {:?}", adapter.get_info());
        tracing::debug!("Adapter features:\n{:#?}", adapter.features());
        let adapter_limits = adapter.limits();
        tracing::debug!("Adapter limits:\n{:#?}", adapter_limits);

        #[cfg(target_os = "macos")]
        let features = wgpu::Features::empty();
        #[cfg(not(target_os = "macos"))]
        let features =
            wgpu::Features::MULTI_DRAW_INDIRECT | wgpu::Features::MULTI_DRAW_INDIRECT_COUNT;

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: None,
                    features: wgpu::Features::default()
                        | wgpu::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES
                        // | wgpu::Features::POLYGON_MODE_LINE
                        | features,
                    limits: wgpu::Limits {
                        max_bind_groups: 8,
                        max_storage_buffer_binding_size: adapter_limits
                            .max_storage_buffer_binding_size,
                        ..Default::default()
                    },
                },
                None,
            )
            .await
            .expect("Failed to create device");

        tracing::info!("Device limits:\n{:#?}", device.limits());

        let swapchain_format = surface
            .as_ref()
            .map(|surface| surface.get_capabilities(&adapter).formats[0]);
        tracing::debug!("Swapchain format: {swapchain_format:?}");
        let swapchain_mode = if surface.is_some() {
            if settings.vsync() {
                // From wgpu docs:
                // "Chooses FifoRelaxed -> Fifo based on availability."
                Some(PresentMode::AutoVsync)
            } else {
                // From wgpu docs:
                // "Chooses Immediate -> Mailbox -> Fifo (on web) based on availability."
                Some(PresentMode::AutoNoVsync)
            }
        } else {
            None
        };
        tracing::debug!("Swapchain present mode: {swapchain_mode:?}");

        if let (Some(window), Some(surface), Some(mode), Some(format)) =
            (window, &surface, swapchain_mode, swapchain_format)
        {
            let size = window.inner_size();
            surface.configure(
                &device,
                &Self::create_sc_desc(format, mode, uvec2(size.width, size.height)),
            );
        }
        tracing::debug!("Created gpu");

        Self {
            device,
            surface,
            queue,
            swapchain_format,
            swapchain_mode,
            adapter,
            will_be_polled,
        }
    }

    pub fn resize(&self, size: winit::dpi::PhysicalSize<u32>) {
        if let Some(surface) = &self.surface {
            if size.width > 0 && size.height > 0 {
                tracing::info!("Resizing to {size:?}");
                surface.configure(&self.device, &self.sc_desc(uvec2(size.width, size.height)));
            }
        }
    }
    pub fn swapchain_format(&self) -> TextureFormat {
        self.swapchain_format
            .unwrap_or(TextureFormat::Rgba8UnormSrgb)
    }
    pub fn swapchain_mode(&self) -> PresentMode {
        self.swapchain_mode.unwrap_or(PresentMode::Immediate)
    }
    pub fn sc_desc(&self, size: UVec2) -> wgpu::SurfaceConfiguration {
        Self::create_sc_desc(self.swapchain_format(), self.swapchain_mode(), size)
    }
    fn create_sc_desc(
        format: TextureFormat,
        present_mode: PresentMode,
        size: UVec2,
    ) -> wgpu::SurfaceConfiguration {
        wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width: size.x,
            height: size.y,
            present_mode,
            alpha_mode: wgpu::CompositeAlphaMode::Auto,
            view_formats: vec![],
        }
    }
}

pub trait WgslType: Zeroable + Pod + 'static {
    fn wgsl_type() -> &'static str;
}
impl WgslType for f32 {
    fn wgsl_type() -> &'static str {
        "f32"
    }
}
impl WgslType for i32 {
    fn wgsl_type() -> &'static str {
        "i32"
    }
}
impl WgslType for u32 {
    fn wgsl_type() -> &'static str {
        "u32"
    }
}

impl WgslType for Vec2 {
    fn wgsl_type() -> &'static str {
        "vec2<f32>"
    }
}

impl WgslType for Vec3 {
    fn wgsl_type() -> &'static str {
        "vec3<f32>"
    }
}

impl WgslType for Vec4 {
    fn wgsl_type() -> &'static str {
        "vec4<f32>"
    }
}

impl WgslType for UVec2 {
    fn wgsl_type() -> &'static str {
        "vec2<u32>"
    }
}

impl WgslType for UVec3 {
    fn wgsl_type() -> &'static str {
        "vec3<u32>"
    }
}

impl WgslType for UVec4 {
    fn wgsl_type() -> &'static str {
        "vec4<u32>"
    }
}
