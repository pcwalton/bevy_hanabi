// A single pass of mergesort.
//
// There are always an even number of passes, so that we ping-pong back to the
// right buffer.

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

// `a_index` should be the index of the a element and `b_index` should be the
// index of the b element.
// Return a number < 0 if a < b and a number > 0 if a > b.
// a and b should never be equal, as this is a requirement of parallel
// mergesort.
fn compare_elements(
    a: ptr<function, KeyValuePair>,
    a_index: u32,
    b: ptr<function, KeyValuePair>,
    b_index: u32
) -> i32 {
    if (a.key < b.key) {
        return -1;
    }
    if (a.key > b.key) {
        return 1;
    }
    if (a.key2 < b.key2) {
        return -1;
    }
    if (a.key2 > b.key2) {
        return 1;
    }
    if (a_index < b_index) {
        return -1;
    }
    return 1;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_invocation_id: vec3<u32>) {
    let global_particle_index = global_invocation_id.x;

    if (arrayLength(&sort_metadata_indices) == 0u) {
        return;
    }
    let last_sort_metadata_index = sort_metadata_indices[arrayLength(&sort_metadata_indices) - 1u];
    let last_global_particle_index =
        effect_sort_metadata[last_sort_metadata_index].first_global_particle_index + 
        effect_sort_metadata[last_sort_metadata_index].last_sort_buffer_index - 
        effect_sort_metadata[last_sort_metadata_index].first_sort_buffer_index;
    if (global_particle_index >= last_global_particle_index) {
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
                effect_sort_metadata[metadata_index_mid].first_global_particle_index +
                effect_sort_metadata[metadata_index_mid].last_sort_buffer_index -
                effect_sort_metadata[metadata_index_mid].first_sort_buffer_index) {
            metadata_index_index_low = metadata_index_index_mid + 1u;
        } else {
            break;
        }
    }
    let metadata_index = sort_metadata_indices[metadata_index_index_mid];

    let effect_first_sort_buffer_index =
        effect_sort_metadata[metadata_index].first_sort_buffer_index;
    let effect_first_global_particle_index =
        effect_sort_metadata[metadata_index].first_global_particle_index;
    let effect_sort_buffer_len = effect_sort_metadata[metadata_index].last_sort_buffer_index -
        effect_first_sort_buffer_index;

    let this_index = global_particle_index - effect_first_global_particle_index;
    let this_sort_buffer_index = effect_first_sort_buffer_index + this_index;

    var this_element: KeyValuePair;
    if ((pass_index & 1u) == 0u) {
        this_element = sort_buffer_a[this_sort_buffer_index];
    } else {
        this_element = sort_buffer_b[this_sort_buffer_index];
    }

    // TODO: Fast path when we're past the last pass!

    let merge_slice_size = 1u << (pass_index + 1u);
    // Which slice are we?
    let merge_slice_index = this_index / merge_slice_size;

    // Calculate the slice boundaries.
    let merge_slice_start = merge_slice_index * merge_slice_size;
    let merge_slice_end = (merge_slice_index + 1) * merge_slice_size;
    let left_slice_start = merge_slice_start;
    let left_slice_end = merge_slice_start + (merge_slice_end - merge_slice_start) / 2;
    let right_slice_start = left_slice_end;
    let right_slice_end = merge_slice_end;

    let in_right_slice = this_index >= right_slice_start;

    // Determine which is our slice and which is the other slice.
    // We already know our sorted index in our slice, since it's sorted.
    // So we only need to binary search the other slice to determine the final
    // index.
    let this_slice_start = select(left_slice_start, right_slice_start, in_right_slice);
    let this_slice_end = select(left_slice_end, right_slice_end, in_right_slice);
    let that_slice_start = select(right_slice_start, left_slice_start, in_right_slice);
    let that_slice_end = select(right_slice_end, left_slice_end, in_right_slice);

    // Initialize the dest index with the relative index in our slice.
    // We will then add the index that binary search gave us and add the left
    // slice start to produce the final index.
    var dest_index = this_index - this_slice_start;

    // Binary search to find the spot in the other list.
    var search_index_low = min(that_slice_start, effect_sort_buffer_len);
    var search_index_high = min(that_slice_end, effect_sort_buffer_len);
    if (search_index_low != search_index_high) {
        while (search_index_low < search_index_high) {
            let search_index_mid = search_index_low + (search_index_high - search_index_low) / 2;
            let that_sort_buffer_index = effect_first_sort_buffer_index + search_index_mid;

            // Fetch the element.
            var that_element: KeyValuePair;
            if ((pass_index & 1u) == 0u) {
                that_element = sort_buffer_a[that_sort_buffer_index];
            } else {
                that_element = sort_buffer_b[that_sort_buffer_index];
            }

            let comparison = compare_elements(
                &this_element,
                this_index,
                &that_element,
                search_index_mid
            );
            if (comparison < 0) {
                search_index_high = search_index_mid;
            } else {
                // `comparison` can't be 0.
                search_index_low = search_index_mid + 1u;
            }
        }

        dest_index += search_index_low - that_slice_start;
    }

    // Compute the final destination index and the sort buffer index.
    let dest_sort_buffer_index = dest_index + merge_slice_start + effect_first_sort_buffer_index;

    // Write it out.
    if ((pass_index & 1u) == 0u) {
        sort_buffer_b[dest_sort_buffer_index] = this_element;
    } else {
        sort_buffer_a[dest_sort_buffer_index] = this_element;
    }
}
