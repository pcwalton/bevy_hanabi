#import bevy_hanabi::vfx_common::EffectSortMetadata;

struct KeyValuePair {
    /// Sorting key.
    key: u32,
    /// Secondary sorting key. Sorts value with the same primary key.
    key2: u32,
    /// Value associated with the sort key(s), generally an index to some other data.
    /// Copied as is and otherwise ignored by the sorting algorithm.
    value: u32,
}

/// Check whether kv1 > kv2, comparing the key(s) of each pair.
fn compare_greater(kv1: KeyValuePair, kv2: KeyValuePair) -> bool {
    if (kv1.key > kv2.key) {
        return true;
    }
#ifdef HAS_DUAL_KEY
    if (kv1.key == kv2.key) {
        return kv1.key2 > kv2.key2;
    }
#endif
    return false;
}

@group(0) @binding(0) var<storage, read_write> sort_buffer : array<KeyValuePair>;
@group(0) @binding(1) var<storage, read> effect_sort_metadata : EffectSortMetadata;

/// Naive insertion sort. TODO: replace with something faster.
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_invocation_id: vec3<u32>) {
    // Naive single-threaded sort
    if (global_invocation_id.x != 0) {
        return;
    }

    // TODO: Don't use a dynamic offset here; instead, index an effect sort
    // metadata array.
    let first_sort_buffer_index = i32(effect_sort_metadata.first_sort_buffer_index);
    let last_sort_buffer_index = i32(effect_sort_metadata.last_sort_buffer_index);

    // Insertion sort
    let num_items = last_sort_buffer_index - first_sort_buffer_index;
    for (var i: i32 = 1; i < num_items; i += 1) {
        var kv = sort_buffer[first_sort_buffer_index + i];
        var j = i;
        while (j > 0 && compare_greater(sort_buffer[first_sort_buffer_index + j - 1], kv)) {
            sort_buffer[first_sort_buffer_index + j] = sort_buffer[first_sort_buffer_index + j - 1];
            j -= 1;
        }
        sort_buffer[first_sort_buffer_index + j] = kv;
    }
}
