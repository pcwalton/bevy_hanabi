#import bevy_hanabi::vfx_common::{BatchDescriptor, BatchEffectIndices, EffectMetadata}

/// Key-value pair for sorting, with optional second sort key.
struct KeyValuePair {
    /// Sorting key.
    key: u32,
    /// Secondary sorting key. Sorts value with the same primary key.
    key2: u32,
    /// Value associated with the sort key(s), generally an index to some other data.
    /// Copied as is and otherwise ignored by the sorting algorithm.
    value: u32,
}

/// Particle buffer as an array of u32. This prevents having to specialize this shader
/// for every single particle layout.
struct RawParticleBuffer {
    data: array<u32>,
}

// NB: Keep in sync with `EffectSortMetadata` in `vfx_common`.
struct EffectSortMetadataAtomic {
    first_sort_buffer_index: u32,
    last_sort_buffer_index: atomic<u32>,
    // Index of the `IndirectDispatch` array in `dispatch_indirect_buffer`.
    indirect_command_index: u32,
    pad: u32,
}

@group(0) @binding(0) var<storage, read_write> sort_buffer : array<KeyValuePair>;
@group(0) @binding(1) var<storage, read> particle_buffer : RawParticleBuffer;
@group(0) @binding(2) var<storage, read> indirect_index_buffer : array<u32>;
// Technically read-only, but the type contains atomic<> fields and wasm is strict about it
@group(0) @binding(3) var<storage, read_write> effect_metadata : array<EffectMetadata>;
@group(0) @binding(4) var<storage, read> batch_descriptor : BatchDescriptor;
@group(0) @binding(5) var<storage, read> batch_effect_indices : array<BatchEffectIndices>;
@group(0) @binding(6) var<storage, read_write> effect_sort_metadata :
    array<EffectSortMetadataAtomic>;

/// Fill the sorting key-value pair buffer with data to prepare for actual sorting.
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_invocation_id: vec3<u32>) {
    let thread_index = global_invocation_id.x;

    var effect_metadata_index = 0u;
    var effect_sort_metadata_index = 0u;
    var effect_index_offset = batch_descriptor.first_batch_effect_index_offset;
    var instance_index = thread_index;
    while (effect_index_offset < batch_descriptor.last_batch_effect_index_offset) {
        effect_metadata_index = batch_effect_indices[effect_index_offset].effect_metadata_index;
        effect_sort_metadata_index =
            batch_effect_indices[effect_index_offset].effect_sort_metadata_index;
        // FIXME: This shouldn't be atomic.
        let effect_instance_count =
            atomicLoad(&effect_metadata[effect_metadata_index].instance_count);
        if (instance_index < effect_instance_count) {
            break;
        }
        instance_index -= effect_instance_count;
        effect_index_offset += 1u;
    }
    if (effect_index_offset == batch_descriptor.last_batch_effect_index_offset) {
        return;
    }

    let read_index = effect_metadata[effect_metadata_index].ping;
    let particle_index = indirect_index_buffer[
        (instance_index + effect_metadata[effect_metadata_index].base_instance) * 3u + read_index
    ];

    let particle_offset = particle_index * effect_metadata[effect_metadata_index].particle_stride;
    let key_offset = particle_offset + effect_metadata[effect_metadata_index].sort_key_offset;
    let key2_offset = particle_offset + effect_metadata[effect_metadata_index].sort_key2_offset;

    let pair_index =
        atomicAdd(&effect_sort_metadata[effect_sort_metadata_index].last_sort_buffer_index, 1u);

    sort_buffer[pair_index].key = particle_buffer.data[key_offset];
    sort_buffer[pair_index].key2 = particle_buffer.data[key2_offset];
    sort_buffer[pair_index].value = particle_index;
}
