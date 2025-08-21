#import bevy_hanabi::vfx_common::{
    BatchDescriptor, BatchMetadata, EffectMetadata, IndexedIndirectDrawCommand,
    NonIndexedIndirectDrawCommand
}

@group(0) @binding(0) var<uniform> batch_metadata : BatchMetadata;
@group(0) @binding(1) var<storage, read> batch_descriptors : array<BatchDescriptor>;
@group(0) @binding(2) var<storage, read> batch_effect_indices : array<u32>;
@group(0) @binding(3) var<storage, read_write> effect_metadata : array<EffectMetadata>;
@group(0) @binding(4) var<storage, read_write> indexed_indirect_draw_commands :
    array<IndexedIndirectDrawCommand>;
@group(0) @binding(5) var<storage, read_write> non_indexed_indirect_draw_commands :
    array<NonIndexedIndirectDrawCommand>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_invocation_id: vec3<u32>) {
    let thread_index = global_invocation_id.x;
    if (thread_index >= batch_metadata.total_batch_count) {
        return;
    }

    let first_batch_effect_index_offset =
        batch_descriptors[thread_index].first_batch_effect_index_offset;
    let last_batch_effect_index_offset =
        batch_descriptors[thread_index].last_batch_effect_index_offset;
    let indirect_draw_command_offset =
        batch_descriptors[thread_index].indirect_draw_command_offset;
    let mesh_is_indexed = batch_descriptors[thread_index].mesh_is_indexed != 0u;

    // Shouldn't happen but let's be safe.
    if (first_batch_effect_index_offset == last_batch_effect_index_offset) {
        return;
    }

    var total_instance_count = 0u;
    for (var batch_effect_index_offset = first_batch_effect_index_offset;
            batch_effect_index_offset < last_batch_effect_index_offset;
            batch_effect_index_offset += 1u) {
        let batch_effect_index = batch_effect_indices[batch_effect_index_offset];
        total_instance_count += effect_metadata[batch_effect_index].instance_count;
    }

    let first_batch_effect_index = batch_effect_indices[first_batch_effect_index_offset];

    let index_or_vertex_count = effect_metadata[first_batch_effect_index].vertex_count;
    let first_index_or_vertex_offset =
        effect_metadata[first_batch_effect_index].first_index_or_vertex_offset;
    let vertex_offset_or_base_instance =
        u32(effect_metadata[first_batch_effect_index].vertex_offset_or_base_instance);

    let effect_count = last_batch_effect_index_offset - first_batch_effect_index_offset;
    for (var effect_index = 0u; effect_index < effect_count; effect_index += 1u) {
        let base_instance = effect_metadata[first_batch_effect_index + effect_index].base_instance;
        let indirect_draw_command_index = indirect_draw_command_offset + effect_index;
        if (mesh_is_indexed) {
            indexed_indirect_draw_commands[indirect_draw_command_index].index_count =
                index_or_vertex_count;
            indexed_indirect_draw_commands[indirect_draw_command_index].instance_count =
                total_instance_count;
            indexed_indirect_draw_commands[indirect_draw_command_index].first_index =
                first_index_or_vertex_offset;
            indexed_indirect_draw_commands[indirect_draw_command_index].vertex_offset =
                vertex_offset_or_base_instance;
            indexed_indirect_draw_commands[indirect_draw_command_index].base_instance = base_instance;
        } else {
            non_indexed_indirect_draw_commands[indirect_draw_command_index].vertex_count =
                index_or_vertex_count;
            non_indexed_indirect_draw_commands[indirect_draw_command_index].instance_count =
                total_instance_count;
            non_indexed_indirect_draw_commands[indirect_draw_command_index].vertex_offset =
                first_index_or_vertex_offset;
            non_indexed_indirect_draw_commands[indirect_draw_command_index].base_instance =
                base_instance;
        }
    }
}
