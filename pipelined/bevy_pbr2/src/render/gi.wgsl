// the volume texture to write to
[[group(0), binding(0)]]
var volume_texture: [[access(write)]] texture_storage_3d<rgba32float>;

[[stage(compute), workgroup_size(1)]]
fn voxelize(
	[[builtin(global_invocation_id)]] global_id: vec3<u32>,
) {
	textureStore(volume_texture, vec3<i32>(global_id), vec4<f32>(1.0, 0.0, 0.0, 0.0));
}

[[stage(compute), workgroup_size(1)]]
fn mipmap(
	[[builtin(global_invocation_id)]] global_id: vec3<u32>,
) {
	textureStore(volume_texture, vec3<i32>(global_id), vec4<f32>(0.0, 1.0, 0.0, 0.0));
}