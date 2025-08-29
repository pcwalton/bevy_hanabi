#import bevy_hanabi::vfx_common::{
    BatchDescriptor, EffectMetadata, EffectSortMetadata, IndirectDispatch
}

@group(0) @binding(0) var<storage, read> batch_descriptors_requiring_sorting : array<u32>;
@group(0) @binding(1) var<storage, read> batch_descriptors : array<BatchDescriptor>;
@group(0) @binding(2) var<storage, read> batch_effect_indices : array<u32>;
@group(0) @binding(3) var<storage, read_write> effect_metadata : array<EffectMetadata>;
@group(0) @binding(4) var<storage, read> effect_sort_metadata : array<EffectSortMetadata>;
@group(0) @binding(5) var<storage, read_write> dispatch_indirect_buffer : array<IndirectDispatch>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_invocation_id: vec3<u32>) {
    let thread_index = global_invocation_id.x;
    if (thread_index >= arrayLength(&batch_descriptors_requiring_sorting)) {
        return;
    }

    let batch_descriptor_index = batch_descriptors_requiring_sorting[thread_index];
    let first_batch_effect_index_offset =
        batch_descriptors[batch_descriptor_index].first_batch_effect_index_offset;
    let last_batch_effect_index_offset =
        batch_descriptors[batch_descriptor_index].last_batch_effect_index_offset;

    var effect_metadata_index = 0u;
    var total_instance_count = 0u;
    for (var batch_effect_index_offset = first_batch_effect_index_offset;
            batch_effect_index_offset < last_batch_effect_index_offset;
            batch_effect_index_offset += 1u) {
        effect_metadata_index = batch_effect_indices[batch_effect_index_offset];
        total_instance_count += effect_metadata[effect_metadata_index].instance_count;
    }

    let effect_sort_metadata_index = effect_metadata[effect_metadata_index].sort_metadata_index;
    let indirect_command_index =
        effect_sort_metadata[effect_sort_metadata_index].indirect_command_index;

    dispatch_indirect_buffer[indirect_command_index].x = (total_instance_count + 63u) >> 6u;
    dispatch_indirect_buffer[indirect_command_index].y = 1;
    dispatch_indirect_buffer[indirect_command_index].z = 1;
}
