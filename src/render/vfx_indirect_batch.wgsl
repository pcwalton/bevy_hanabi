#import bevy_hanabi::vfx_common::{
    BatchDescriptor, BatchEffectIndices, BatchMetadata, EffectMetadata, IndirectDispatch
}

@group(0) @binding(0) var<uniform> batch_metadata : BatchMetadata;
@group(0) @binding(1) var<storage, read> batch_descriptors : array<BatchDescriptor>;
@group(0) @binding(2) var<storage, read> batch_effect_indices : array<BatchEffectIndices>;
@group(0) @binding(3) var<storage, read_write> effect_metadata : array<EffectMetadata>;
@group(0) @binding(4) var<storage, read_write> dispatch_indirect_buffer : array<IndirectDispatch>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_invocation_id: vec3<u32>) {
    let thread_index = global_invocation_id.x;
    if (thread_index >= arrayLength(&effect_metadata)) {
        return;
    }
    if (thread_index >= batch_metadata.total_batch_count) {
        return;
    }

    let first_batch_effect_index_offset =
        batch_descriptors[thread_index].first_batch_effect_index_offset;
    let last_batch_effect_index_offset =
        batch_descriptors[thread_index].last_batch_effect_index_offset;

    var total_alive_count = 0u;
    for (var batch_effect_index_offset = first_batch_effect_index_offset;
            batch_effect_index_offset < last_batch_effect_index_offset;
            batch_effect_index_offset += 1u) {
        let batch_effect_index =
            batch_effect_indices[batch_effect_index_offset].effect_metadata_index;
        total_alive_count += effect_metadata[batch_effect_index].max_update;
    }

    dispatch_indirect_buffer[thread_index].x = (total_alive_count + 63u) >> 6u;
    dispatch_indirect_buffer[thread_index].y = 1;
    dispatch_indirect_buffer[thread_index].z = 1;
}