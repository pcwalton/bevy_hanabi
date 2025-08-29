#import bevy_hanabi::vfx_common::{
    BatchDescriptor, BatchMetadata, ChildInfoBuffer, EffectMetadata, IndirectDispatch
}

@group(0) @binding(0) var<uniform> batch_metadata : BatchMetadata;
@group(0) @binding(1) var<storage, read> batch_descriptors_with_events : array<u32>;
@group(0) @binding(2) var<storage, read> batch_descriptors : array<BatchDescriptor>;
@group(0) @binding(3) var<storage, read> batch_effect_indices : array<u32>;
@group(0) @binding(4) var<storage, read_write> effect_metadata : array<EffectMetadata>;
@group(0) @binding(5) var<storage, read> child_info : ChildInfoBuffer;
@group(0) @binding(6) var<storage, read_write> init_indirect_dispatch : array<IndirectDispatch>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_invocation_id: vec3<u32>) {
    let thread_index = global_invocation_id.x;
    if (thread_index >= batch_metadata.total_render_batches_with_events_count) {
        return;
    }

    let batch_descriptor_index = batch_descriptors_with_events[thread_index];
    let first_batch_effect_index_offset =
        batch_descriptors[batch_descriptor_index].first_batch_effect_index_offset;
    let last_batch_effect_index_offset =
        batch_descriptors[batch_descriptor_index].last_batch_effect_index_offset;

    var effect_metadata_index = 0u;
    var child_info_index = 0u;
    var total_event_count = 0u;
    for (var batch_effect_index_offset = first_batch_effect_index_offset;
            batch_effect_index_offset < last_batch_effect_index_offset;
            batch_effect_index_offset += 1u) {
        effect_metadata_index = batch_effect_indices[batch_effect_index_offset];
        child_info_index = effect_metadata[effect_metadata_index].global_child_index;
        total_event_count += child_info[child_info_index].event_count;
    }

    let init_indirect_dispatch_index = child_info[child_info_index].init_indirect_dispatch_index;

    init_indirect_dispatch[init_indirect_dispatch_index].x = (total_event_count + 63u) >> 6u;
    init_indirect_dispatch[init_indirect_dispatch_index].y = 1;
    init_indirect_dispatch[init_indirect_dispatch_index].z = 1;
}
