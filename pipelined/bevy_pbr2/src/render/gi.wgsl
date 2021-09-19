// what to store inside the volume buffer
struct Voxel {
	color_and_opacity: vec4<f32>;
};

// entire volume
[[block]]
struct Volume {
	voxels: array<Voxel>;
};

// what a vertex stores
struct Vertex {
    position: vec3<f32>;
    normal: vec3<f32>;
    uv: vec2<f32>;
};

// the volume buffer to write to
[[group(0), binding(0)]]
var<storage> volume_buffer: [[access(read_write)]] Volume;

[[stage(compute), workgroup_size(64)]]
fn voxelize(
	[[builtin(global_invocation_id)]] global_id: vec3<u32>,
) {
}

[[stage(compute), workgroup_size(64)]]
fn clear(
	[[builtin(global_invocation_id)]] global_id: vec3<u32>,
) {

	// get the length of the buffer
	let array_len = arrayLength(&volume_buffer.voxels);

	// and only write if it's allowed
	if (global_id.x < array_len) {
		volume_buffer.voxels[global_id.x] = Voxel (vec4<f32>(0.0));
	}
}