#[derive(Debug, Clone, Copy)]
pub struct GiVolume {
    /// size of the first volume
    pub size: f32,

    /// how many lods there are for the volume, each one is scaled up by 2
    /// There is a maximum of 8 lods allowed
    pub num_lods: u8,

    /// resolution of each lod volume, aka how many voxels there are per volume
    /// more looks better, but will perform worse
    pub resolution: u8,
}

impl Default for GiVolume {
    fn default() -> Self {
        Self {
            size: 16.0,
            num_lods: 4,
            resolution: 64,
        }
    }
}
