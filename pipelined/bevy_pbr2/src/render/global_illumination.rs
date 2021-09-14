use crate::{AmbientLight, DirectionalLight, MeshMeta, MeshTransform, PbrPipeline, PointLight, GiVolume};
use bevy_core::bytes_of;
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

pub struct VoxelizePipeline {
	voxelize_pipeline: ComputePipeline,
	clear_pipeline: ComputePipeline,
	volume_layout: BindGroupLayout,
	voxelize_layout: BindGroupLayout,
}

// TODO: fromworld to get the voxelization shader + buffers and descriptors ready
// extract step to get all volumes from the world (although there's only one)
// prepare step to put everything into the gpu version of the structs
// queue step to write everything to the gpu buffers
// and a node that runs the compute shader (IS THIS REALLY NEEDED? CAN THIS BE DONE IN QUEUE?)
impl FromWorld for VoxelizePipeline {

	fn from_world(world: &mut World) -> Self {
		let render_device = world.get_resource::<RenderDevice>().unwrap();
		//let pbr_shaders = world.get_resource::<PbrPipeline>().unwrap();

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
					ty: BindingType::Buffer {
						ty: BufferBindingType::Storage { read_only: false },
						min_binding_size: None,
						has_dynamic_offset: false,
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

		// next up, make the pipelines
		let clear_pipeline_layout = render_device.create_pipeline_layout(&PipelineLayoutDescriptor {
			label: None,
			push_constant_ranges: &[],
			bind_group_layouts: &[&volume_layout, /*&mipmap_layout*/],
		});

		let voxelize_pipeline_layout = render_device.create_pipeline_layout(&PipelineLayoutDescriptor {
			label: None,
			push_constant_ranges: &[],
			bind_group_layouts: &[&volume_layout, /*&voxelize_layout, &pbr_shaders.mesh_layout*/], // TODO: also needs the light layout + actual mesh info
		});

		let clear_pipeline = render_device.create_compute_pipeline(&ComputePipelineDescriptor {
			label: None,
			layout: Some(&clear_pipeline_layout),
			module: &shader_module,
			entry_point: "clear",
		});

		let voxelize_pipeline = render_device.create_compute_pipeline(&ComputePipelineDescriptor {
			label: None,
			layout: Some(&voxelize_pipeline_layout),
			module: &shader_module,
			entry_point: "voxelize",
		});

		VoxelizePipeline {
			volume_layout,
			voxelize_pipeline,
			clear_pipeline,
			voxelize_layout,
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

pub struct ViewGiVolume {
	pub volume_buffer: Buffer,
	pub volume_buffer_bind_group: BindGroup,
	pub gpu_binding_index: u32,
}

// TODO: move this to have per-view textures, per view mipmaps too
#[derive(Default)]
pub struct GiMeta {
	pub view_gi_volumes: DynamicUniformVec<GpuGiVolume>,
}

// prepares a volume for each view
// each view is basically a render from each camera
// here we get the volume for each view, + the texture associated with the volume
pub fn prepare_volumes(
	mut commands: Commands,
	mut buffer_cache: ResMut<BufferCache>,
	render_device: Res<RenderDevice>,
	render_queue: Res<RenderQueue>,
	mut gi_meta: ResMut<GiMeta>,
	views: Query<Entity, With<RenderPhase<Transparent3d>>>,
	volume: Res<ExtractedGiVolume>,
	shaders: Res<VoxelizePipeline>
) {

	// reserve the needed stuff
	gi_meta.view_gi_volumes.reserve_and_clear(views.iter().count(), &render_device);

	// go over all views
	for entity in views.iter() {

		// get the buffer
		// TODO: CACHE
		let volume_buffer = buffer_cache.get(
			&render_device,
			BufferDescriptor {
			label: Some("Volume buffer"),
			mapped_at_creation: false,
			usage: BufferUsage::STORAGE,
			// TODO: Proper data layout
			size: volume.resolution as u64 * volume.resolution as u64 * volume.resolution as u64 * 16, // 16 here due to using a vec4<f32> TODO CORRECT SIZE
		});

		// set the volume texture bind group
		let volume_buffer_bind = Some(render_device.create_bind_group(&BindGroupDescriptor {
			label: None,
			layout: &shaders.volume_layout,
			entries: &[BindGroupEntry {
				binding: 0,
				resource: volume_buffer.buffer.as_entire_binding()
			}],
		}));

		// get the volume settings
		let gpu_volume = GpuGiVolume {
			size: volume.size,
			num_lods: volume.num_lods as u32,
			resolution: volume.resolution as u32,
			view_projection: volume.transform,
		};

		// next up, add the gpu volume to the view
		commands.entity(entity).insert(ViewGiVolume {
			volume_buffer: volume_buffer.buffer,
			// volume_texture_view,
			volume_buffer_bind_group: volume_buffer_bind.unwrap(),
			gpu_binding_index: gi_meta.view_gi_volumes.push(gpu_volume),
		});
	}
}

// node that runs the voxelization pass
pub struct VoxelizePassNode {
	main_query: QueryState<&'static ViewGiVolume>,
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
		let shaders = world.get_resource::<VoxelizePipeline>().unwrap();
		let meta = world.get_resource::<GiMeta>().unwrap();

		// go over all views (happens outside this function, so we just need to get the relevant view)
		if let Ok(view_volume) = self.main_query.get_manual(world, view_entity) {

			// next up, we want to voxelize all meshes, so start a new pipeline to do that
			let mut compute_pass = render_context.command_encoder.begin_compute_pass(&ComputePassDescriptor {
				label: None,
			});

			// first, we want to clear the buffer from potential previous content
			compute_pass.set_pipeline(&shaders.clear_pipeline);

			// set the volume texture as our output
			compute_pass.set_bind_group(0, &view_volume.volume_buffer_bind_group.value(), &[]);

			// and run!
			compute_pass.dispatch(32, 1, 1);

			println!("dispatched");

			// we want to voxelize things
			compute_pass.set_pipeline(&shaders.voxelize_pipeline);

			// set the volume settings

			// go over all meshes
			// TODO


			// and run the shader!
			compute_pass.dispatch(32, 32, 32);

			

			// and drop it because we are done
			drop(compute_pass);

		}
		
		Ok(())
	}
}
