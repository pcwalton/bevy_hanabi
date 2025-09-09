#import bevy_hanabi::vfx_common::{
    BatchDescriptor, BatchEffectIndices, EffectMetadata, EffectSortMetadata
}

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

struct IndirectIndexBuffer {
    data: array<u32>,
}

@group(0) @binding(0) var<storage, read_write> indirect_index_buffer : IndirectIndexBuffer;
@group(0) @binding(1) var<storage, read> sort_buffer : array<KeyValuePair>;
// Technically read-only, but the type contains atomic<> fields and wasm is strict about it
@group(0) @binding(2) var<storage, read_write> effect_metadata : array<EffectMetadata>;
@group(0) @binding(3) var<storage, read> batch_descriptor : BatchDescriptor;
@group(0) @binding(4) var<storage, read> batch_effect_indices : array<BatchEffectIndices>;
@group(0) @binding(5) var<storage, read> effect_sort_metadata : array<EffectSortMetadata>;

/// Copy the sorted particle indices back into the effect index buffer.
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_invocation_id: vec3<u32>) {
    let thread_index = global_invocation_id.x;

    var effect_metadata_index = 0u;
    var effect_index_offset = batch_descriptor.first_batch_effect_index_offset;
    var row_index = thread_index;
    while (effect_index_offset < batch_descriptor.last_batch_effect_index_offset) {
        effect_metadata_index = batch_effect_indices[effect_index_offset].effect_metadata_index;
        // FIXME: This shouldn't be atomic.
        let effect_instance_count =
            atomicLoad(&effect_metadata[effect_metadata_index].instance_count);
        if (row_index < effect_instance_count) {
            break;
        }
        row_index -= effect_instance_count;
        effect_index_offset += 1u;
    }
    if (effect_index_offset == batch_descriptor.last_batch_effect_index_offset) {
        return;
    }

    let effect_sort_metadata_index = effect_metadata[effect_metadata_index].sort_metadata_index;
    let first_sort_buffer_index =
        effect_sort_metadata[effect_sort_metadata_index].first_sort_buffer_index;
    
    // Always write into ping, read from pong
    let write_index = effect_metadata[effect_metadata_index].ping;

    let particle_index = sort_buffer[first_sort_buffer_index + row_index].value;
    indirect_index_buffer.data[
        (row_index + effect_metadata[effect_metadata_index].base_instance) * 3u + write_index
    ] = particle_index;
}
