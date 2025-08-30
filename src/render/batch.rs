use std::{fmt::Debug, num::NonZeroU32, ops::Range};

use bevy::{
    ecs::entity::EntityHashMap,
    prelude::*,
    render::{render_resource::CachedComputePipelineId, sync_world::MainEntity},
};
use fixedbitset::FixedBitSet;
use indexmap::IndexMap;

use super::{
    effect_cache::{DispatchBufferIndices, EffectSlice},
    event::{CachedChildInfo, CachedEffectEvents},
    BufferBindingSource, CachedMesh, LayoutFlags, PropertyBindGroupKey,
};
use crate::{
    render::CachedMeshLocation, AlphaMode, EffectAsset, EffectShader, ParticleLayout, TextureLayout,
};

#[derive(Debug, Clone, Copy)]
pub(crate) enum BatchSpawnInfo {
    /// Spawn a number of particles uploaded from CPU each frame.
    CpuSpawner {
        /// Total number of particles to spawn for the batch. This is only used
        /// to calculate the number of compute workgroups to dispatch.
        total_spawn_count: u32,
    },

    /// Spawn a number of particles calculated on GPU from "spawn events", which
    /// generally emitted by another effect.
    GpuSpawner {
        /// Index of the [`EventBuffer`] where the GPU spawn events consumed by
        /// this batch are stored.
        ///
        /// [`EventBuffer`]: super::event::EventBuffer
        #[allow(dead_code)]
        event_buffer_index: u32,
    },
}

/// Internal information about a single instance of an effect.
#[derive(Debug, Clone)]
pub(crate) struct EffectInstance {
    /// Handle of the underlying effect asset describing the effect.
    pub handle: Handle<EffectAsset>,
    /// Index of the [`EffectBuffer`].
    ///
    /// [`EffectBuffer`]: super::effect_cache::EffectBuffer
    pub buffer_index: u32,
    /// Slice of particles in the GPU effect buffer referenced by
    /// [`EffectBatch::buffer_index`].
    pub slice: Range<u32>,
    /// Spawn info for this batch
    pub spawn_info: BatchSpawnInfo,
    /// Specialized init and update compute pipelines.
    pub init_and_update_pipeline_ids: InitAndUpdatePipelineIds,
    /// Configured shader used for the particle rendering of this group.
    /// Note that we don't need to keep the init/update shaders alive because
    /// their pipeline specialization is doing it via the specialization key.
    pub render_shader: Handle<Shader>,
    pub parent_min_binding_size: Option<NonZeroU32>,
    pub parent_binding_source: Option<BufferBindingSource>,
    /// Event buffers of child effects, if any.
    pub child_event_buffers: Vec<(Entity, BufferBindingSource)>,
    /// Index of the property buffer, if any.
    pub property_key: Option<PropertyBindGroupKey>,
    /// Index of the first [`GpuSpawnerParams`] entry of the effects in the
    /// batch. Subsequent batched effects have their entries following linearly
    /// after that one.
    ///
    /// [`GpuSpawnerParams`]: super::GpuSpawnerParams
    pub spawner_base: u32,
    /// The indices within the various indirect dispatch buffers.
    pub dispatch_buffer_indices: DispatchBufferIndices,
    /// Particle layout shared by all batched effects and groups.
    pub particle_layout: ParticleLayout,
    /// Flags describing the render layout.
    pub layout_flags: LayoutFlags,
    /// Asset ID of the effect mesh to draw.
    pub mesh: AssetId<Mesh>,
    /// Texture layout.
    pub texture_layout: TextureLayout,
    /// Textures.
    pub textures: Vec<Handle<Image>>,
    /// Alpha mode.
    pub alpha_mode: AlphaMode,
    /// Entities holding the source [`ParticleEffect`] instances which were
    /// batched into this single batch. Used to determine visibility per view.
    ///
    /// [`ParticleEffect`]: crate::ParticleEffect
    pub entities: Vec<u32>,
    pub cached_effect_events: Option<CachedEffectEvents>,
    pub cached_mesh_location: Option<CachedMeshLocation>,
    pub position: Vec3,
    pub main_entity: MainEntity,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct EffectInstanceIndex(pub u32);

#[derive(Debug, Default, Resource)]
pub(crate) struct SortedEffects {
    /// Effect instances in the order they were inserted by [`push()`], indexed
    /// by the returned [`EffectBatchIndex`].
    ///
    /// [`push()`]: Self::push
    pub(super) instances: Vec<EffectInstance>,
    /// Index of the dispatch queue used for indirect fill dispatch and
    /// submitted to [`GpuBufferOperations`].
    pub(super) dispatch_queue_index: Option<u32>,
    /// Effect batches in the order they were inserted.
    pub(super) batches: IndexMap<EffectBatchKey, EffectBatch>,
}

/// Identifies effect batches.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) struct EffectBatchKey {
    asset_id: AssetId<EffectAsset>,
    buffer_index: u32,
}

/// Information about a single batched set of effects.
#[derive(Clone, Debug, Default)]
pub(crate) struct EffectBatch {
    /// Indices of the effects in this batch.
    pub(super) effect_instance_indices: Vec<EffectInstanceIndex>,
    /// Index of the GPU batch descriptor.
    pub(super) batch_descriptor_index: u32,
    /// The index of the [`GpuDispatchIndirect`] row in the GPU buffer
    /// init dispatch indirect buffer, if this effect batch has a GPU spawner.
    pub(super) init_dispatch_indirect_buffer_row_index: Option<u32>,
    /// The index of the [`GpuDispatchIndirect`] row in the GPU buffer
    /// [`EffectsMeta::update_dispatch_indirect_buffer`].
    ///
    /// [`EffectsMeta::update_dispatch_indirect_buffer`]: super::EffectsMeta::update_dispatch_indirect_buffer
    pub(crate) update_dispatch_indirect_buffer_row_index: u32,
    /// The index of the [`GpuDispatchIndirect`] row in the GPU buffer
    /// [`EffectsMeta::sort_dispatch_indirect_buffer`].
    ///
    /// [`EffectsMeta::sort_dispatch_indirect_buffer`]: super::EffectsMeta::sort_dispatch_indirect_buffer
    pub(crate) sort_dispatch_indirect_buffer_row_index: u32,
}

impl SortedEffects {
    pub fn clear(&mut self) {
        self.instances.clear();
        self.dispatch_queue_index = None;
        self.batches.clear();
    }

    pub fn push(&mut self, effect_instance: EffectInstance) -> EffectInstanceIndex {
        let index = self.instances.len() as u32;
        self.instances.push(effect_instance);
        EffectInstanceIndex(index)
    }

    #[allow(dead_code)]
    pub fn len(&self) -> usize {
        self.instances.len()
    }

    pub fn is_empty(&self) -> bool {
        self.instances.is_empty()
    }

    pub fn get(&self, index: EffectInstanceIndex) -> Option<&EffectInstance> {
        if index.0 < self.instances.len() as u32 {
            Some(&self.instances[index.0 as usize])
        } else {
            None
        }
    }
}

impl EffectBatchKey {
    pub(crate) fn new(asset_id: AssetId<EffectAsset>, buffer_index: u32) -> EffectBatchKey {
        EffectBatchKey {
            asset_id,
            buffer_index,
        }
    }
}

/// Sorts effects into the proper order for batching.
///
/// This places parents before children and also tries to place effects in the
/// same buffer together.
pub(crate) struct EffectSorter {
    /// Information that we keep about each effect.
    pub(crate) effects: Vec<EffectToBeSorted>,
    /// A mapping from a child to its parent, if it has one.
    pub(crate) child_to_parent: EntityHashMap<Entity>,
}

/// Information that the [`EffectSorter`] maintains in order to sort each
/// effect into the proper order.
pub(crate) struct EffectToBeSorted {
    /// The render-world entity of the effect.
    pub(crate) entity: Entity,
    /// The index of the buffer that the indirect indices for this effect are
    /// stored in.
    pub(crate) buffer_index: u32,
    /// The offset within the buffer described above at which the indirect
    /// indices for this effect start.
    ///
    /// This is in elements, not bytes.
    pub(crate) base_instance: u32,
}

/// The key that we sort effects by for optimum batching.
#[derive(Clone, Copy, PartialEq, PartialOrd, Eq, Ord)]
struct EffectSortKey {
    /// The level in the dependency graph.
    ///
    /// Parents always have lower levels than their children.
    level: u32,
    /// The index of the buffer that the indirect indices for this effect are
    /// stored in.
    buffer_index: u32,
    /// The offset within the buffer described above at which the indirect
    /// indices for this effect start.
    base_instance: u32,
}

impl EffectSorter {
    /// Creates a new [`EffectSorter`].
    pub(crate) fn new() -> EffectSorter {
        EffectSorter {
            effects: vec![],
            child_to_parent: EntityHashMap::default(),
        }
    }

    /// Sorts all the effects into the optimal order for batching.
    pub(crate) fn sort(&mut self) {
        // First, create a map of entity to index.
        let mut entity_to_index = EntityHashMap::default();
        for (index, effect) in self.effects.iter().enumerate() {
            entity_to_index.insert(effect.entity, index);
        }

        // Next, create a map of children to their parents.
        let mut children_to_parent: Vec<_> = (0..self.effects.len()).map(|_| vec![]).collect();
        for (kid, parent) in self.child_to_parent.iter() {
            let (parent_index, kid_index) = (entity_to_index[parent], entity_to_index[kid]);
            children_to_parent[kid_index].push(parent_index);
        }

        // Now topologically sort the graph. Create an ordering that places
        // children before parents.
        // https://en.wikipedia.org/wiki/Topological_sorting#Depth-first_search
        let mut ordering = vec![0; self.effects.len()];
        let mut visiting = FixedBitSet::with_capacity(self.effects.len());
        let mut visited = FixedBitSet::with_capacity(self.effects.len());
        while let Some(effect_index) = visited.zeroes().next() {
            visit(
                &mut ordering,
                &mut visiting,
                &mut visited,
                &children_to_parent,
                effect_index,
            );
        }

        // Compute levels.
        let mut levels = vec![0; self.effects.len()];
        for effect_index in ordering.into_iter().rev() {
            let level = levels[effect_index];
            for &parent in &children_to_parent[effect_index] {
                levels[parent] = levels[parent].max(level + 1);
            }
        }

        // Now sort the result.
        self.effects.sort_unstable_by_key(|effect| EffectSortKey {
            level: levels[entity_to_index[&effect.entity]],
            buffer_index: effect.buffer_index,
            base_instance: effect.base_instance,
        });

        // A helper function for topologically sorting the effect dependency
        // graph.
        fn visit(
            ordering: &mut Vec<usize>,
            visiting: &mut FixedBitSet,
            visited: &mut FixedBitSet,
            children_to_parent: &[Vec<usize>],
            effect_index: usize,
        ) {
            if visited.contains(effect_index) {
                return;
            }
            debug_assert!(
                !visiting.contains(effect_index),
                "Parent-child effect relation contains a cycle"
            );

            visiting.insert(effect_index);

            for &parent in &children_to_parent[effect_index] {
                visit(ordering, visiting, visited, children_to_parent, parent);
            }

            visited.insert(effect_index);
            ordering.push(effect_index);
        }
    }
}

/// Single effect batch to drive rendering.
///
/// This component is spawned into the render world during the prepare phase
/// ([`prepare_effects()`]), once per effect batch per group. In turns it
/// references an [`EffectBatch`] component containing all the shared data for
/// all the groups of the effect.
#[derive(Debug, Component)]
pub(crate) struct EffectDrawBatch {
    /// Indices of the indirect draw commands in the indirect draw command
    /// buffer.
    pub indirect_draw_command_range: Range<u32>,
    /// The first effect instance in the batch.
    pub representative_effect_instance_index: EffectInstanceIndex,
    /// Position of the emitter so we can compute distance to camera.
    pub representative_translation: Vec3,
    /// The main-world entity that contains this effect.
    #[allow(dead_code)]
    pub representative_main_entity: MainEntity,
    pub render_batch_descriptor_index: u32,
}

impl EffectInstance {
    /// Create a new batch from a single input.
    pub fn from_input(
        cached_mesh: &CachedMesh,
        cached_effect_events: Option<&CachedEffectEvents>,
        cached_child_info: Option<&CachedChildInfo>,
        cached_mesh_location: Option<&CachedMeshLocation>,
        input: &mut InstanceInput,
        dispatch_buffer_indices: DispatchBufferIndices,
        property_key: Option<PropertyBindGroupKey>,
        main_entity: MainEntity,
    ) -> EffectInstance {
        let spawn_info = if let Some(event_buffer_index) = input.event_buffer_index {
            BatchSpawnInfo::GpuSpawner { event_buffer_index }
        } else {
            BatchSpawnInfo::CpuSpawner {
                total_spawn_count: input.spawn_count,
            }
        };

        EffectInstance {
            handle: input.handle.clone(),
            buffer_index: input.effect_slice.buffer_index,
            slice: input.effect_slice.slice.clone(),
            spawn_info,
            init_and_update_pipeline_ids: input.init_and_update_pipeline_ids,
            render_shader: input.shaders.render.clone(),
            parent_min_binding_size: cached_child_info
                .map(|cci| cci.parent_particle_layout.min_binding_size32()),
            parent_binding_source: cached_child_info
                .map(|cci| cci.parent_buffer_binding_source.clone()),
            child_event_buffers: input.child_effects.clone(),
            property_key,
            spawner_base: input.spawner_index,
            particle_layout: input.effect_slice.particle_layout.clone(),
            dispatch_buffer_indices,
            layout_flags: input.layout_flags,
            mesh: cached_mesh.mesh,
            texture_layout: input.texture_layout.clone(),
            textures: input.textures.clone(),
            alpha_mode: input.alpha_mode,
            entities: vec![input.main_entity.id().index()],
            cached_effect_events: cached_effect_events.cloned(),
            cached_mesh_location: cached_mesh_location.cloned(),
            position: input.position,
            main_entity,
        }
    }
}

/// Effect batching input, obtained from extracted effects.
#[derive(Debug, Component)]
pub(crate) struct InstanceInput {
    /// Handle of the underlying effect asset describing the effect.
    pub handle: Handle<EffectAsset>,
    /// Main entity of the [`ParticleEffect`], used for visibility.
    pub main_entity: MainEntity,
    /// Render entity of the [`CachedEffect`].
    #[allow(dead_code)]
    pub entity: Entity,
    /// Effect slices.
    pub effect_slice: EffectSlice,
    /// Compute pipeline IDs of the specialized and cached pipelines.
    pub init_and_update_pipeline_ids: InitAndUpdatePipelineIds,
    /// Index of the event buffer, if this effect consumes GPU spawn events.
    pub event_buffer_index: Option<u32>,
    /// Child effects, if any.
    pub child_effects: Vec<(Entity, BufferBindingSource)>,
    /// Various flags related to the effect.
    pub layout_flags: LayoutFlags,
    /// Texture layout.
    pub texture_layout: TextureLayout,
    /// Textures.
    pub textures: Vec<Handle<Image>>,
    /// Alpha mode.
    pub alpha_mode: AlphaMode,
    #[allow(dead_code)]
    pub particle_layout: ParticleLayout,
    /// Effect shaders.
    pub shaders: EffectShader,
    /// Index of the [`GpuSpawnerParams`] in the
    /// [`EffectsCache::spawner_buffer`].
    pub spawner_index: u32,
    /// Number of particles to spawn for this effect.
    pub spawn_count: u32,
    /// Emitter position.
    pub position: Vec3,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct InitAndUpdatePipelineIds {
    pub init: CachedComputePipelineId,
    pub update: CachedComputePipelineId,
}

#[cfg(test)]
mod tests {
    use bevy::ecs::entity::Entity;

    use super::*;

    fn insert_entry(
        sorter: &mut EffectSorter,
        entity: Entity,
        buffer_index: u32,
        base_instance: u32,
        parent: Option<Entity>,
    ) {
        sorter.effects.push(EffectToBeSorted {
            entity,
            base_instance,
            buffer_index,
        });
        if let Some(parent) = parent {
            sorter.child_to_parent.insert(entity, parent);
        }
    }

    #[test]
    fn toposort_batches() {
        let mut sorter = EffectSorter::new();

        // Some "parent" effect
        let e1 = Entity::from_raw(1);
        insert_entry(&mut sorter, e1, 42, 0, None);
        assert_eq!(sorter.effects.len(), 1);
        assert_eq!(sorter.effects[0].entity, e1);
        assert!(sorter.child_to_parent.is_empty());

        // Some "child" effect in a different buffer
        let e2 = Entity::from_raw(2);
        insert_entry(&mut sorter, e2, 5, 30, Some(e1));
        assert_eq!(sorter.effects.len(), 2);
        assert_eq!(sorter.effects[0].entity, e1);
        assert_eq!(sorter.effects[1].entity, e2);
        assert_eq!(sorter.child_to_parent.len(), 1);
        assert_eq!(sorter.child_to_parent[&e2], e1);

        sorter.sort();
        assert_eq!(sorter.effects.len(), 2);
        assert_eq!(sorter.effects[0].entity, e2); // child first
        assert_eq!(sorter.effects[1].entity, e1); // parent after
        assert_eq!(sorter.child_to_parent.len(), 1); // unchanged
        assert_eq!(sorter.child_to_parent[&e2], e1); // unchanged

        // Some "child" effect in the same buffer as its parent
        let e3 = Entity::from_raw(3);
        insert_entry(&mut sorter, e3, 42, 20, Some(e1));
        assert_eq!(sorter.effects.len(), 3);
        assert_eq!(sorter.effects[0].entity, e2); // from previous sort
        assert_eq!(sorter.effects[1].entity, e1); // from previous sort
        assert_eq!(sorter.effects[2].entity, e3); // simply appended
        assert_eq!(sorter.child_to_parent.len(), 2);
        assert_eq!(sorter.child_to_parent[&e2], e1);
        assert_eq!(sorter.child_to_parent[&e3], e1);

        sorter.sort();
        assert_eq!(sorter.effects.len(), 3);
        assert_eq!(sorter.effects[0].entity, e2); // child first
        assert_eq!(sorter.effects[1].entity, e3); // other child next (in same buffer as parent)
        assert_eq!(sorter.effects[2].entity, e1); // finally, parent
        assert_eq!(sorter.child_to_parent.len(), 2);
        assert_eq!(sorter.child_to_parent[&e2], e1);
        assert_eq!(sorter.child_to_parent[&e3], e1);
    }
}
