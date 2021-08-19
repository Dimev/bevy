use bevy_math::{Mat4, Vec3, Vec4};
use crevice::std140::AsStd140;
use bevy_render2::{
	render_resource::*,
};

pub const MAX_NUM_LODS: u8 = 8;


pub struct ExtractedGiVolume {
    size: f32,
    num_lods: u8,
    resolution: u8,
    transform: Mat4,
}

#[repr(C)]
#[derive(Copy, Clone, AsStd140, Default, Debug)]
pub struct GpuGiVolume {
	size: f32,
	num_lods: u32,
	resolution: u32,
	view_projection: Mat4,
}

pub struct VoxelizePipeline {
	pipeline: ComputePipeline,
	view_layout: BindGroupLayout,
	volume_texture_sampler: Sampler,
}

// TODO: fromworld to get the voxelization shader + buffers and descriptors ready
// extract step to get all volumes from the world (although there's only one)
// prepare step to put everything into the gpu version of the structs
// queue step to write everything to the gpu buffers
// and a node that runs the compute shader (IS THIS REALLY NEEDED? CAN THIS BE DONE IN QUEUE?)


