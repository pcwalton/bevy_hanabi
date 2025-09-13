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
    if (kv1.key == kv2.key) {
        return kv1.key2 > kv2.key2;
    }
    return false;
}

@group(0) @binding(0) var<storage, read_write> sort_buffer : array<KeyValuePair>;
@group(0) @binding(1) var<storage, read> effect_sort_metadata : array<EffectSortMetadata>;
@group(0) @binding(2) var<storage, read> sort_metadata_indices : array<u32>;

fn swap(i: u32, j: u32) {
    let tmp = sort_buffer[i];
    sort_buffer[i] = sort_buffer[j];
    sort_buffer[j] = tmp;
}

fn i_left_child(i: u32) -> u32 {
    return 2u * i + 1u;
}

/// Naive heapsort. TODO: replace with something faster.
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_invocation_id: vec3<u32>) {
    // Naive single-threaded sort
    let metadata_index_index = u32(global_invocation_id.x);
    if (metadata_index_index >= arrayLength(&sort_metadata_indices)) {
        return;
    }
    let metadata_index = sort_metadata_indices[metadata_index_index];

    let first_sort_buffer_index = effect_sort_metadata[metadata_index].first_sort_buffer_index;
    let last_sort_buffer_index = effect_sort_metadata[metadata_index].last_sort_buffer_index;

    // Heapsort:
    // https://en.wikipedia.org/wiki/Heapsort#Standard_implementation
    let count = last_sort_buffer_index - first_sort_buffer_index;
    var start = count / 2u;
    var end = count;
    while (end > 1u) {
        if (start > 0u) {
            start -= 1u;
        } else {
            end -= 1u;
            swap(first_sort_buffer_index + end, first_sort_buffer_index + 0u);
        }

        // siftDown(a, start, end)
        var root = start;
        while (i_left_child(root) < end) {
            var child = i_left_child(root);
            // If there is a right child and that child is greater:
            if (child + 1u < end && compare_greater(
                sort_buffer[first_sort_buffer_index + child + 1u],
                sort_buffer[first_sort_buffer_index + child]
            )) {
                child += 1u;
            }

            if (compare_greater(
                sort_buffer[first_sort_buffer_index + child],
                sort_buffer[first_sort_buffer_index + root]
            )) {
                swap(first_sort_buffer_index + root, first_sort_buffer_index + child);
                // Repeat to continue sifting down the child now.
                root = child;
            } else {
                // Return to outer loop.
                break;
            }
        }
    }
}
