use crate::{AmbientLight, DirectionalLight, MeshMeta, MeshTransform, PbrPipeline, PointLight};
use bevy_math::{Mat4, Vec3, Vec4};
use bevy_ecs::system::lifetimeless::*;
use bevy_ecs::{prelude::*, system::SystemState};
use crevice::std140::AsStd140;
use bevy_render2::{
	render_resource::*,
    renderer::{RenderContext, RenderDevice, RenderQueue},
	shader::Shader,
};

pub const MAX_NUM_LODS: u8 = 8;


pub struct ExtractedGiVolume {
    size: f32,
    num_lods: u8,
    resolution: u8,
    transform: Mat4,
}

// this contains everything needed for one entire volume, including lods
#[repr(C)]
#[derive(Copy, Clone, AsStd140, Default, Debug)]
pub struct GpuGiVolume {
	size: f32,
	num_lods: u32,
	resolution: u32,
	view_projection: Mat4,
}

// this helps indicate which lod we are on
#[repr(C)]
#[derive(Copy, Clone, AsStd140, Default, Debug)]
pub struct GpuMipMap {
	level: u32,
}

pub struct VoxelizePipeline {
	shader_module: ShaderModule,
	voxelize_pipeline: ComputePipeline,
	mipmap_pipeline: ComputePipeline,
	voxelize_layout: BindGroupLayout,
	mipmap_layout: BindGroupLayout,
	volume_texture_sampler: Sampler,
}

// TODO: fromworld to get the voxelization shader + buffers and descriptors ready
// extract step to get all volumes from the world (although there's only one)
// prepare step to put everything into the gpu version of the structs
// queue step to write everything to the gpu buffers
// and a node that runs the compute shader (IS THIS REALLY NEEDED? CAN THIS BE DONE IN QUEUE?)
impl FromWorld for VoxelizePipeline {

	fn from_world(world: &mut World) -> Self {
		let render_device = world.get_resource::<RenderDevice>().unwrap();
		let pbr_shaders = world.get_resource::<PbrPipeline>().unwrap();

		let shader = Shader::from_wgsl(include_str!("gi.wgsl"))
			.process(&[])
			.unwrap();

		let shader_module = render_device.create_shader_module(&shader);

		// for voxelizing the scene into the texture
		let voxelize_layout = render_device.create_bind_group_layout(&BindGroupLayoutDescriptor {
			entries: &[
				// this one stores the volume settings
				BindGroupLayoutEntry {
					binding: 0,
					visibility: ShaderStage::COMPUTE,
					ty: BindingType::Buffer {
						ty: BufferBindingType::Uniform,
						has_dynamic_offset: true,
						min_binding_size: BufferSize::new(GpuGiVolume::std140_size_static() as u64),
					},
					count: None, 
				},
				// and we also want our volume texture to write to
				BindGroupLayoutEntry {
					binding: 1,
					visibility: ShaderStage::COMPUTE,
					ty: BindingType::StorageTexture {
						format: TextureFormat::Rgba32Float, // TODO: see if 16 bit floats work?
						access: StorageTextureAccess::ReadWrite,
						view_dimension: TextureViewDimension::D3,
					},
					count: None,
				},

			],
			label: None,
		});

		// and for generating mipmaps
		let mipmap_layout = render_device.create_bind_group_layout(&BindGroupLayoutDescriptor {
			entries: &[
				// for indicating what level we are at
				BindGroupLayoutEntry {
					binding: 0,
					visibility: ShaderStage::COMPUTE,
					ty: BindingType::Buffer {
						ty: BufferBindingType::Uniform,
						has_dynamic_offset: true,
						min_binding_size: BufferSize::new(GpuMipMap::std140_size_static() as u64),
					},
					count: None, 
				},
				// and we also want our volume texture to write to
				BindGroupLayoutEntry {
					binding: 1,
					visibility: ShaderStage::COMPUTE,
					ty: BindingType::StorageTexture {
						format: TextureFormat::Rgba32Float, // TODO: see if 16 bit floats work?
						access: StorageTextureAccess::ReadWrite,
						view_dimension: TextureViewDimension::D3,
					},
					count: None,
				},

			],
			label: None,
		});

		// next up, make the pipelines

		let mipmap_pipeline_layout = render_device.create_pipeline_layout(&PipelineLayoutDescriptor {
			label: None,
			push_constant_ranges: &[],
			bind_group_layouts: &[&mipmap_layout],
		});

		let voxelize_pipeline_layout = render_device.create_pipeline_layout(&PipelineLayoutDescriptor {
			label: None,
			push_constant_ranges: &[],
			bind_group_layouts: &[&voxelize_layout, &pbr_shaders.mesh_layout], // TODO: also needs the light layout + actual mesh info
		});

		let mipmap_pipeline = render_device.create_compute_pipeline(&ComputePipelineDescriptor {
			label: None,
			layout: Some(&mipmap_pipeline_layout),
			module: &shader_module,
			entry_point: "mipmap",
		});

		let voxelize_pipeline = render_device.create_compute_pipeline(&ComputePipelineDescriptor {
			label: None,
			layout: Some(&voxelize_pipeline_layout),
			module: &shader_module,
			entry_point: "voxelize",
		});

		VoxelizePipeline {
			shader_module,
			voxelize_pipeline,
			mipmap_pipeline,
			voxelize_layout,
			mipmap_layout,
			volume_texture_sampler: render_device.create_sampler(&SamplerDescriptor {
				address_mode_u: AddressMode::ClampToEdge,
				address_mode_v: AddressMode::ClampToEdge,
				address_mode_w: AddressMode::ClampToEdge,
				mag_filter: FilterMode::Nearest,
				min_filter: FilterMode::Nearest,
				mipmap_filter: FilterMode::Linear,
				..Default::default()
			}),
		}
	}
}


