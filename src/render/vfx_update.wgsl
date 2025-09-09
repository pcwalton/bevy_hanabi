#import bevy_hanabi::vfx_common::{
    BatchDescriptor, BatchEffectIndices, BatchMetadata, ChildInfo, ChildInfoBuffer, EventBuffer,
    IndirectDispatch, IndirectBuffer, EffectMetadata, RenderGroupIndirect,
    SimParams, Spawner, seed, tau, pcg_hash, to_float01, frand, frand2, frand3,
    frand4, rand_uniform_f, rand_uniform_vec2, rand_uniform_vec3,
    rand_uniform_vec4, rand_normal_f, rand_normal_vec2, rand_normal_vec3,
    rand_normal_vec4, proj
}

struct Particle {
{{ATTRIBUTES}}
}

struct ParticleBuffer {
    particles: array<Particle>,
}

#ifdef READ_PARENT_PARTICLE

struct ParentParticle {
    {{PARENT_ATTRIBUTES}}
}

struct ParentParticleBuffer {
    particles: array<ParentParticle>,
}

#endif

{{PROPERTIES}}

@group(0) @binding(0) var<uniform> sim_params : SimParams;

// "particle" group @1
@group(1) @binding(0) var<storage, read_write> particle_buffer : ParticleBuffer;
@group(1) @binding(1) var<storage, read_write> indirect_buffer : IndirectBuffer;
#ifdef READ_PARENT_PARTICLE
@group(1) @binding(2) var<storage, read> parent_particle_buffer : ParentParticleBuffer;
#endif

// "spawner" group @2
@group(2) @binding(0) var<storage, read> spawners : array<Spawner>;
{{PROPERTIES_BINDING}}

// "metadata" group @3
@group(3) @binding(0) var<storage, read> batch_descriptor : BatchDescriptor;
@group(3) @binding(1) var<storage, read> batch_effect_indices : array<BatchEffectIndices>;
@group(3) @binding(2) var<storage, read_write> effect_metadata : array<EffectMetadata>;
#ifdef EMITS_GPU_SPAWN_EVENTS
{{EMIT_EVENT_BUFFER_BINDINGS}}
#endif

{{UPDATE_EXTRA}}

#ifdef EMITS_GPU_SPAWN_EVENTS
{{EMIT_EVENT_BUFFER_APPEND_FUNCS}}
#endif

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_invocation_id: vec3<u32>) {
    let thread_index = global_invocation_id.x;

    // Step through render batch descriptors. Cap to `max_update`.
    // TODO: This should be a prefix sum and binary search or something instead
    // of linear search.
    var effect_metadata_index = 0u;
    var spawner_index = 0u;
    var effect_index_offset = batch_descriptor.first_batch_effect_index_offset;
    var indirect_particle_index = thread_index;
    while (effect_index_offset < batch_descriptor.last_batch_effect_index_offset) {
        effect_metadata_index = batch_effect_indices[effect_index_offset].effect_metadata_index;
        spawner_index = batch_effect_indices[effect_index_offset].spawner_index;
        let this_max_update = u32(effect_metadata[effect_metadata_index].max_update);
        if (indirect_particle_index < this_max_update) {
            break;
        }
        indirect_particle_index -= this_max_update;
        effect_index_offset += 1u;
    }
    if (effect_index_offset == batch_descriptor.last_batch_effect_index_offset) {
        return;
    }

    // Always write into ping, read from pong
    let write_index = effect_metadata[effect_metadata_index].ping;
    let read_index = 1u - write_index;

    let particle_index = indirect_buffer.indices[
        3u * (indirect_particle_index + effect_metadata[effect_metadata_index].base_instance) + read_index
    ];

    // Initialize the PRNG seed
    seed = pcg_hash(particle_index ^ spawners[spawner_index].seed);

    var particle: Particle = particle_buffer.particles[particle_index];
    {{AGE_CODE}}
    {{REAP_CODE}}
    {{UPDATE_CODE}}

    {{WRITEBACK_CODE}}

    // Check if alive
    if (!is_alive) {
        // Save dead index
        let dead_index = atomicAdd(&effect_metadata[effect_metadata_index].dead_count, 1u) +
            effect_metadata[effect_metadata_index].base_instance;
        indirect_buffer.indices[3u * dead_index + 2u] = particle_index;

        // Also increment copy of dead count, which was updated in dispatch indirect
        // pass just before, and need to remain correct after this pass
        atomicAdd(&effect_metadata[effect_metadata_index].max_spawn, 1u);
        atomicSub(&effect_metadata[effect_metadata_index].alive_count, 1u);
    } else {
        // Increment alive particle count and write indirection index for later rendering
        let indirect_index = atomicAdd(&effect_metadata[effect_metadata_index].instance_count, 1u) +
            effect_metadata[effect_metadata_index].base_instance;
        indirect_buffer.indices[3u * indirect_index + write_index] = particle_index;
    }
}
