// Pass.

#import bevy_hanabi::vfx_common::{BatchMetadata, EffectSortMetadata}

struct KeyValuePair {
    key: u32,
    key2: u32,
    value: u32,
}

@group(0) @binding(0) var<storage, read_write> sort_buffer_a : array<KeyValuePair>;
@group(0) @binding(1) var<storage, read_write> sort_buffer_b : array<KeyValuePair>;
@group(0) @binding(2) var<storage, read> effect_sort_metadata : array<EffectSortMetadata>;
@group(0) @binding(3) var<storage, read> sort_metadata_indices : array<u32>;
@group(0) @binding(4) var<uniform> batch_metadata : BatchMetadata;

var<push_constant> pass_index: u32;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_invocation_id: vec3<u32>) {
    let global_particle_index = global_invocation_id.x;
    if (global_particle_index >= batch_metadata.total_particles_potentially_requiring_sorting_count) {
        return;
    }

    // Binary search to find the metadata index index.
    var metadata_index_index_low = 0u;
    var metadata_index_index_mid = 0u;
    var metadata_index_index_high = arrayLength(&sort_metadata_indices);
    while (metadata_index_index_low < metadata_index_index_high) {
        metadata_index_index_mid = metadata_index_index_low +
            (metadata_index_index_high - metadata_index_index_low) / 2u;
        let metadata_index_mid = sort_metadata_indices[metadata_index_index_mid];
        if (global_particle_index <
                effect_sort_metadata[metadata_index_mid].first_global_particle_index) {
            metadata_index_index_high = metadata_index_index_mid;
        } else if (global_particle_index >=
                effect_sort_metadata[metadata_index_mid].last_sort_buffer_index -
                effect_sort_metadata[metadata_index_mid].first_sort_buffer_index +
                effect_sort_metadata[metadata_index_mid].first_global_particle_index) {
            metadata_index_index_low = metadata_index_index_mid + 1u;
        } else {
            break;
        }
    }
    let metadata_index = sort_metadata_indices[metadata_index_index_mid];

    let sort_buffer_index = effect_sort_metadata[metadata_index].first_sort_buffer_index +
        global_particle_index - effect_sort_metadata[metadata_index].first_global_particle_index;

    // TODO(pcwalton): Make this actually sort.
    if ((pass_index & 1u) == 0u) {
        sort_buffer_b[sort_buffer_index] = sort_buffer_a[sort_buffer_index];
    } else {
        sort_buffer_a[sort_buffer_index] = sort_buffer_b[sort_buffer_index];
    }

    // TODO(pcwalton): Add a special thing here that copies back to buffer A
    // from buffer B after the final pass if we need to.
}
