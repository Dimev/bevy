use crate::{AmbientLight, DirectionalLight, MeshMeta, MeshTransform, PbrPipeline, PointLight, GiVolume};
use bevy_math::{Mat4, Vec3, Vec4};
use bevy_ecs::system::lifetimeless::*;
use bevy_ecs::{prelude::*, system::SystemState};
use crevice::std140::AsStd140;
use bevy_render2::{
	render_resource::*,
    renderer::{RenderContext, RenderDevice, RenderQueue},
	shader::Shader,
	render_graph::{Node, NodeRunError, RenderGraphContext, SlotInfo, SlotType},
	texture::*,
	render_phase::{
        Draw, DrawFunctionId, DrawFunctions, PhaseItem, RenderPhase, TrackedRenderPass,
    },
	view::{ExtractedView, ViewMeta, ViewUniformOffset},
};

use bevy_core_pipeline::Transparent3d;
use bevy_transform::components::GlobalTransform;

/// maximum number of cascades/lods the Gi volumes can have
pub const MAX_NUM_LODS: u8 = 8;

/// the format to use for the volume texture
pub const VOLUME_TEXTURE_FORMAT: TextureFormat = TextureFormat::Rgba32Float; // TODO: see if 16 bit floats work?

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

pub struct VoxelizeShaders {
	voxelize_pipeline: ComputePipeline,
	mipmap_pipeline: ComputePipeline,
	volume_layout: BindGroupLayout,
	voxelize_layout: BindGroupLayout,
	mipmap_layout: BindGroupLayout,
	volume_texture_sampler: Sampler,
}

// TODO: fromworld to get the voxelization shader + buffers and descriptors ready
// extract step to get all volumes from the world (although there's only one)
// prepare step to put everything into the gpu version of the structs
// queue step to write everything to the gpu buffers
// and a node that runs the compute shader (IS THIS REALLY NEEDED? CAN THIS BE DONE IN QUEUE?)
impl FromWorld for VoxelizeShaders {

	fn from_world(world: &mut World) -> Self {
		let render_device = world.get_resource::<RenderDevice>().unwrap();
		let pbr_shaders = world.get_resource::<PbrPipeline>().unwrap();

		let shader = Shader::from_wgsl(include_str!("gi.wgsl"))
			.process(&[])
			.unwrap();

		let shader_module = render_device.create_shader_module(&shader);

		// volume layout, for storing the volume texture
		let volume_layout = render_device.create_bind_group_layout(&BindGroupLayoutDescriptor {
			entries: &[
				// we also want our volume texture to write to
				BindGroupLayoutEntry {
					binding: 0,
					visibility: ShaderStage::COMPUTE,
					ty: BindingType::StorageTexture {
						format: VOLUME_TEXTURE_FORMAT,
						access: StorageTextureAccess::ReadWrite,
						view_dimension: TextureViewDimension::D3,
					},
					count: None,
				},

			],
			label: None,
		});

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

			],
			label: None,
		});

		// next up, make the pipelines
		let mipmap_pipeline_layout = render_device.create_pipeline_layout(&PipelineLayoutDescriptor {
			label: None,
			push_constant_ranges: &[],
			bind_group_layouts: &[&volume_layout, &mipmap_layout],
		});

		let voxelize_pipeline_layout = render_device.create_pipeline_layout(&PipelineLayoutDescriptor {
			label: None,
			push_constant_ranges: &[],
			bind_group_layouts: &[&volume_layout, &voxelize_layout, &pbr_shaders.mesh_layout], // TODO: also needs the light layout + actual mesh info
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

		VoxelizeShaders {
			volume_layout,
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



// get the volumes from the world ecs
pub fn extract_volumes(
	mut commands: Commands,
	volumes: Query<(Entity, &GiVolume, &GlobalTransform)>
) {

	// we're only allowing one volume at a time
	if let Ok((_, volume, transform)) = volumes.single() {

		// and insert it
		commands.insert_resource(ExtractedGiVolume {
			num_lods: volume.num_lods.min(MAX_NUM_LODS),
			size: volume.size,
			resolution: volume.resolution,
			transform: transform.compute_matrix(),
		});
	}
}

//pub struct ViewGiVolumes {
//	pub volume_texture: Texture,
//	pub volume_texture_view: TextureView,
//	pub volumes: Vec<Entity>, // stores all the render passes: TODO: NEEDED? 
//	pub gpu_volume_binding_index: u32,
//}

#[derive(Default)]
pub struct GiMeta {
	pub gi_texture_bind: Option<BindGroup>,
	pub gi_volume_bind: Option<BindGroup>,
	pub gi_volume_buffer: Option<Buffer>,
	pub gi_mipmaps_bind: Vec<BindGroup>,
}

// prepares a volume for each view
// each view is basically a render from each camera
// here we get the volume for each view, + the texture associated with the volume
pub fn prepare_volumes(
	mut commands: Commands,
	mut texture_cache: ResMut<TextureCache>,
	render_device: Res<RenderDevice>,
	render_queue: Res<RenderQueue>,
	mut gi_meta: ResMut<GiMeta>,
	volume: Res<ExtractedGiVolume>,
	shaders: Res<VoxelizeShaders>
) {

	// get the volume texture for this view
	let volume_texture = texture_cache.get(
		&render_device,
		TextureDescriptor {
			size: Extent3d { width: volume.resolution as u32, height: volume.resolution as u32, depth_or_array_layers: volume.resolution as u32 * volume.num_lods as u32 },
			mip_level_count: volume.num_lods.next_power_of_two().trailing_zeros() + 1, // same as log2
			sample_count: 1,
			dimension: TextureDimension::D3,
			format: VOLUME_TEXTURE_FORMAT,
			usage: TextureUsage::SAMPLED | TextureUsage::STORAGE,
			label: None,
		}
	);

	let volume_texture_view = volume_texture.texture.create_view(&TextureViewDescriptor {
		label: None,
		format: None,
		dimension: Some(TextureViewDimension::D3),
		aspect: TextureAspect::All,
		base_mip_level: 0,
		mip_level_count: None,
		base_array_layer: 0,
		array_layer_count: None,
	});

	// set the volume texture bind group
	gi_meta.gi_texture_bind = Some(render_device.create_bind_group(&BindGroupDescriptor {
		label: Some("volume texture"),
		layout: &shaders.volume_layout,
		entries: &[BindGroupEntry {
			binding: 0,
			resource: BindingResource::TextureView(&volume_texture_view)
		}],
	}));

	let gpu_volume = GpuGiVolume {
		size: volume.size,
		num_lods: volume.num_lods as u32,
		resolution: volume.resolution as u32,
		view_projection: volume.transform,
	};

	// TODO: only write if volume changed
	let gpu_volume_buffer = render_device.create_buffer_with_data(&BufferInitDescriptor {
		label: None,
		contents: gpu_volume.as_std140(),
		usage: BufferUsage::UNIFORM,
	});

	gi_meta.gi_volume_buffer = Some(gpu_volume_buffer);

	// and set the gpu volume as a bind group
	gi_meta.gi_volume_bind = Some(render_device.create_bind_group(&BindGroupDescriptor {
		label: None,
		layout: &shaders.voxelize_layout,
		entries: &[BindGroupEntry {
			binding: 0,
			resource: BindingResource::Buffer(BufferBinding {
				buffer: gpu_volume_buffer,
				offset: 0,
				size: None,
			})
		}],
	}));

	// TODO: store the mipmap gen inside the view mipmpas, and for each view, also store the amount of mipmaps it needs
	// OR do mip gen in the shader manually
	//commands.entity(entity).insert(ViewGiVolumes {
	//	volume_texture: volume_texture.texture,
	//	volume_texture_view,
	//	volumes: vec![],
	//	gpu_volume_binding_index: gi_meta.view_gi_volumes.push(gpu_volume)
	//});
	// TODO: add the other stuff to this view, idk how yet

}

// TODO: find a way to make the render pass work

// node that runs the voxelization pass
pub struct VoxelizePassNode {
	main_query: QueryState<&'static GiMeta>,
}

impl VoxelizePassNode {
	pub const IN_VIEW: &'static str = "view";

	pub fn new(world: &mut World) -> Self {
		Self { main_query: QueryState::new(world) }
	}
}

impl Node for VoxelizePassNode {
	fn input(&self) -> Vec<SlotInfo> {
		vec![SlotInfo::new(VoxelizePassNode::IN_VIEW, SlotType::Entity)]
	}

	fn update(&mut self, world: &mut World) {
		self.main_query.update_archetypes(world);
	}

	fn run(&self, graph: &mut RenderGraphContext, render_context: &mut RenderContext, world: &World) -> Result<(), NodeRunError> {
		let view_entity = graph.get_input_entity(Self::IN_VIEW)?;
		let shaders = world.get_resource::<VoxelizeShaders>().unwrap();
		let meta = world.get_resource::<GiMeta>().unwrap();

		// there's only one volume, so only one thing to do
		// step 1: clear the texture
		// TODO: wait for 10.0
		// render_context.command_encoder.clear_texture(&view_volume.volume_texture, ImageSubresourceRange)

		// next up, we want to voxelize all meshes, so start a new pipeline to do that
		let mut compute_pass = render_context.command_encoder.begin_compute_pass(&ComputePassDescriptor {
			label: None,
		});

		// we want to voxelize things
		compute_pass.set_pipeline(&shaders.voxelize_pipeline);

		// set the volume texture as our output
		compute_pass.set_bind_group(0, meta.gi_texture_bind.unwrap(), &[]);

		// TODO: figure out where we need to create the bind group for the volume texture

		// go over all meshes
		// TODO
		// and run the shader!
		compute_pass.dispatch(16, 16, 16);

		// next up, we want to make mipmaps

		// and drop it because we are done
		drop(compute_pass);



		

		Ok(())
	}
}
