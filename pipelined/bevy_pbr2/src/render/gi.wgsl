// what to store inside the volume buffer
struct Voxel {
	color_and_opacity: vec4<f32>;
};

// entire volume
[[block]]
struct Volume {
	voxels: array<Voxel>;
};

// the volume buffer to write to
[[group(0), binding(0)]]
var<storage> volume_buffer: [[access(read_write)]] Volume;

[[stage(compute), workgroup_size(1)]]
fn voxelize(
	[[builtin(global_invocation_id)]] global_id: vec3<u32>,
) {
}

[[stage(compute), workgroup_size(32)]]
fn clear(
	[[builtin(global_invocation_id)]] global_id: vec3<u32>,
) {
	volume_buffer.voxels[global_id.x] = Voxel (vec4<f32>(0.0, 1.0, 2.0, 3.0));
}