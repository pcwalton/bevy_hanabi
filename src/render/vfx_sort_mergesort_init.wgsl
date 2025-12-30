// Naive single-threaded prefix sum.

#import bevy_hanabi::vfx_common::{BatchMetadata, EffectSortMetadata, IndirectDispatch}

@group(0) @binding(0) var<storage, read_write> effect_sort_metadata : array<EffectSortMetadata>;
@group(0) @binding(1) var<storage, read> sort_metadata_indices : array<u32>;
@group(0) @binding(2) var<storage, read_write> dispatch_indirect_buffer : array<IndirectDispatch>;
@group(0) @binding(3) var<uniform> batch_metadata : BatchMetadata;

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) global_invocation_id: vec3<u32>) {
    var total_particle_count = 0u;
    for (var metadata_index_index = 0u;
            metadata_index_index < arrayLength(&sort_metadata_indices);
            metadata_index_index += 1u) {
        let metadata_index = sort_metadata_indices[metadata_index_index];
        effect_sort_metadata[metadata_index].first_global_particle_index = total_particle_count;
        total_particle_count += effect_sort_metadata[metadata_index].last_sort_buffer_index -
            effect_sort_metadata[metadata_index].first_sort_buffer_index;
    }

    let workgroup_count = (total_particle_count + 255u) / 256u;

    var pass_count = 0u;
    if (total_particle_count > 0u) {
        pass_count = 32u - countLeadingZeros(total_particle_count - 1u);
    }

    // FIXME(pcwalton): Could be a separate pass.
    // FIXME(pcwalton): Be smarter and set the dispatch count to 0 if we're past
    // the total pass count.
    // FIXME(pcwalton): Use `total_particles_potentially_requiring_sorting` for
    // the actual length of this buffer.
    for (var dispatch_index = 0u;
            dispatch_index < arrayLength(&dispatch_indirect_buffer);
            dispatch_index += 1u) {
        dispatch_indirect_buffer[dispatch_index].x = workgroup_count;
        dispatch_indirect_buffer[dispatch_index].y = 1u;
        dispatch_indirect_buffer[dispatch_index].z = 1u;
    }
}
