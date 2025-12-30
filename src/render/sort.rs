use std::{
    num::{NonZeroU32, NonZeroU64},
    ops::Range,
};

use bevy::{
    asset::Handle,
    ecs::{resource::Resource, world::World},
    log::info,
    platform::collections::{hash_map::Entry, HashMap},
    prelude::Shader,
    render::{
        render_resource::{
            BindGroup, BindGroupLayout, Buffer, BufferId, CachedComputePipelineId,
            ComputePipelineDescriptor, PipelineCache, RawBufferVec, ShaderType,
        },
        renderer::{RenderDevice, RenderQueue},
    },
    utils::default,
};
use bytemuck::{Pod, Zeroable};
use wgpu::{
    BindGroupEntry, BindGroupLayoutEntry, BindingResource, BindingType, BufferBinding,
    BufferBindingType, BufferUsages, PushConstantRange, ShaderStages,
};

use super::{GpuEffectMetadata, StorageType};
use crate::{
    render::{GpuEffectSortMetadata, GpuRenderBatchDescriptor, GpuRenderBatchMetadata},
    Attribute, ParticleLayout,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct SortFillBindGroupLayoutKey {
    particle_min_binding_size: NonZeroU32,
    particle_ribbon_id_offset: u32,
    particle_age_offset: u32,
}

impl SortFillBindGroupLayoutKey {
    pub fn from_particle_layout(particle_layout: &ParticleLayout) -> Result<Self, ()> {
        let particle_ribbon_id_offset = particle_layout.offset(Attribute::RIBBON_ID).ok_or(())?;
        let particle_age_offset = particle_layout.offset(Attribute::AGE).ok_or(())?;
        let key = SortFillBindGroupLayoutKey {
            particle_min_binding_size: particle_layout.min_binding_size32(),
            particle_ribbon_id_offset,
            particle_age_offset,
        };
        Ok(key)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct SortIndirectBatchBindGroupKey {
    effect_metadata: BufferId,
    effect_sort_metadata: BufferId,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct SortFillBindGroupKey {
    particle: BufferId,
    indirect_index: BufferId,
    effect_metadata: BufferId,
    effect_sort_metadata: BufferId,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct SortMergesortInitBindGroupKey {
    effect_sort_metadata: BufferId,
    sort_metadata_indices: BufferId,
    mergesort_dispatch_indirect: BufferId,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct SortMergesortPassBindGroupKey {
    effect_sort_metadata: BufferId,
    sort_metadata_indices: BufferId,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct SortCopyBindGroupKey {
    indirect_index: BufferId,
    effect_metadata: BufferId,
    effect_sort_metadata: BufferId,
}

#[repr(C)]
#[derive(Debug, Copy, Clone, Default, Pod, Zeroable, ShaderType)]
struct GpuSortKeyValuePair {
    key: u32,
    key2: u32,
    value: u32,
}

#[derive(Resource)]
pub struct SortBindGroups {
    /// Render device.
    render_device: RenderDevice,
    /// Sort-fill pass compute shader.
    sort_fill_shader: Handle<Shader>,
    /// GPU buffer of key-value pairs to sort.
    sort_buffer: RawBufferVec<GpuSortKeyValuePair>,
    sort_temp_buffer: RawBufferVec<GpuSortKeyValuePair>,
    sort_indirect_batch_bind_groups: HashMap<SortIndirectBatchBindGroupKey, BindGroup>,
    /// Bind group layouts for group #0 of the sort-fill compute pass.
    sort_fill_bind_group_layouts:
        HashMap<SortFillBindGroupLayoutKey, (BindGroupLayout, CachedComputePipelineId)>,
    /// Bind groups for group #0 of the sort-fill compute pass.
    sort_fill_bind_groups: Option<SortFillBindGroups>,
    /// Bind groups for group #0 of the mergesort init compute pass.
    mergesort_init_bind_groups: Option<SortMergesortInitBindGroups>,
    /// Bind groups for group #0 of the mergesort sort compute pass.
    mergesort_pass_bind_groups: Option<SortMergesortPassBindGroups>,
    /// Bind groups for group #0 of the sort compute pass.
    sort_bind_group: Option<CachedSortBindGroup>,
    sort_indirect_batch_bind_group_layout: BindGroupLayout,
    sort_copy_bind_group_layout: BindGroupLayout,
    sort_bind_group_layout: BindGroupLayout,
    mergesort_init_bind_group_layout: BindGroupLayout,
    mergesort_pass_bind_group_layout: BindGroupLayout,
    sort_indirect_batch_pipeline_id: CachedComputePipelineId,
    /// Pipeline for sort pass.
    sort_pipeline_id: CachedComputePipelineId,
    /// Pipeline for mergesort init pass.
    mergesort_init_pipeline_id: CachedComputePipelineId,
    /// Pipeline for mergesort pass pass.
    mergesort_pass_pipeline_id: CachedComputePipelineId,
    /// Pipeline for sort-copy pass.
    sort_copy_pipeline_id: CachedComputePipelineId,
    /// Bind groups for group #0 of the sort-copy compute pass.
    sort_copy_bind_groups: Option<SortCopyBindGroups>,
}

/// Bind groups for group #0 of the sort-fill compute pass.
struct SortFillBindGroups {
    // This is not part of the bind group key, because we don't want to cache
    // sort buffer bind groups. Otherwise they'll leak when we go to resize the
    // sort buffer.
    sort_buffer_id: BufferId,
    bind_groups: HashMap<SortFillBindGroupKey, BindGroup>,
}

/// Bind groups for group #0 of the mergesort-init compute pass.
struct SortMergesortInitBindGroups {
    bind_groups: HashMap<SortMergesortInitBindGroupKey, BindGroup>,
}

/// Bind groups for group #0 of the mergesort-pass compute pass.
struct SortMergesortPassBindGroups {
    bind_groups: HashMap<SortMergesortPassBindGroupKey, BindGroup>,
}

/// Bind groups for group #0 of the sort-copy compute pass.
struct SortCopyBindGroups {
    // This is not part of the bind group key, because we don't want to cache
    // sort buffer bind groups. Otherwise they'll leak when we go to resize the
    // sort buffer.
    sort_buffer_id: BufferId,
    bind_groups: HashMap<SortCopyBindGroupKey, BindGroup>,
}

struct CachedSortBindGroup {
    sort_buffer_id: BufferId,
    sort_metadata_buffer_id: BufferId,
    sort_metadata_indices_buffer_id: BufferId,
    sort_metadata_indices_count: usize,
    bind_group: BindGroup,
}

impl SortBindGroups {
    pub fn new(
        world: &mut World,
        sort_indirect_batch_shader: Handle<Shader>,
        sort_fill_shader: Handle<Shader>,
        sort_shader: Handle<Shader>,
        sort_copy_shader: Handle<Shader>,
        mergesort_init_shader: Handle<Shader>,
        mergesort_pass_shader: Handle<Shader>,
    ) -> Self {
        let render_device = world.resource::<RenderDevice>();
        let pipeline_cache = world.resource::<PipelineCache>();

        let storage_alignment = render_device.limits().min_storage_buffer_offset_alignment;
        let sort_metadata_size = GpuEffectSortMetadata::aligned_size(storage_alignment);

        let mut sort_buffer = RawBufferVec::new(BufferUsages::STORAGE);
        sort_buffer.set_label(Some("hanabi:buffer:sort"));
        let mut sort_temp_buffer = RawBufferVec::new(BufferUsages::STORAGE);
        sort_temp_buffer.set_label(Some("hanabi:buffer:sort_temp"));

        let sort_bind_group_layout = render_device.create_bind_group_layout(
            "hanabi:bind_group_layout:sort",
            &[
                // @group(0) @binding(0) var<storage, read_write> sort_buffer :
                // array<KeyValuePair>;
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: Some(NonZeroU64::new(12).unwrap()), // dual kv pair
                    },
                    count: None,
                },
                // @group(0) @binding(1) var<storage, read> effect_sort_metadata
                // : array<EffectSortMetadata>;
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: Some(sort_metadata_size),
                    },
                    count: None,
                },
                // @group(0) @binding(2) var<storage, read>
                // sort_metadata_indices : array<u32>;
                BindGroupLayoutEntry {
                    binding: 2,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: Some(u32::min_size()),
                    },
                    count: None,
                },
            ],
        );

        let sort_pipeline_id = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some("hanabi:pipeline:sort".into()),
            layout: vec![sort_bind_group_layout.clone()],
            shader: sort_shader,
            // TODO: Do we need to put some common shader defs in here?
            shader_defs: vec![],
            entry_point: Some("main".into()),
            push_constant_ranges: vec![],
            zero_initialize_workgroup_memory: false,
        });

        let min_storage_buffer_offset_alignment =
            render_device.limits().min_storage_buffer_offset_alignment;
        let effect_metadata_min_binding_size =
            GpuEffectMetadata::aligned_size(min_storage_buffer_offset_alignment);
        let batch_descriptor_size =
            GpuRenderBatchDescriptor::aligned_size(min_storage_buffer_offset_alignment);

        let sort_indirect_batch_bind_group_layout = render_device.create_bind_group_layout(
            "hanabi:bind_group_layout:sort_indirect_batch",
            &[
                // @group(0) @binding(0) var<uniform> batch_metadata :
                // BatchMetadata;
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: Some(GpuRenderBatchMetadata::min_size()),
                    },
                    count: None,
                },
                // @group(0) @binding(1) var<storage, read>
                // batch_descriptors_requiring_sorting : array<u32>;
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // @group(0) @binding(2) var<storage, read> batch_descriptors :
                // array<BatchDescriptor>;
                BindGroupLayoutEntry {
                    binding: 2,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // @group(0) @binding(3) var<storage, read> batch_effect_indices
                // : array<u32>;
                BindGroupLayoutEntry {
                    binding: 3,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // @group(0) @binding(4) var<storage, read_write>
                // effect_metadata : array<EffectMetadata>;
                BindGroupLayoutEntry {
                    binding: 4,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // @group(0) @binding(5) var<storage, read> effect_sort_metadata
                // : array<EffectSortMetadata>;
                BindGroupLayoutEntry {
                    binding: 5,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // @group(0) @binding(6) var<storage, read_write>
                // dispatch_indirect_buffer : array<IndirectDispatch>;
                BindGroupLayoutEntry {
                    binding: 6,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        );

        let sort_indirect_batch_pipeline_id =
            pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
                label: Some("hanabi:pipeline:sort_indirect_batch".into()),
                layout: vec![sort_indirect_batch_bind_group_layout.clone()],
                shader: sort_indirect_batch_shader,
                // TODO: Do we need to put some common shader defs in here?
                shader_defs: vec![],
                entry_point: Some("main".into()),
                push_constant_ranges: vec![],
                zero_initialize_workgroup_memory: false,
            });

        let mergesort_init_bind_group_layout = render_device.create_bind_group_layout(
            "hanabi:bind_group_layout:sort_mergesort_init",
            &[
                // @group(0) @binding(0) var<storage, read_write>
                // effect_sort_metadata : array<EffectSortMetadata>;
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // @group(0) @binding(1) var<storage, read>
                // sort_metadata_indices : array<u32>;
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: Some(u32::min_size()),
                    },
                    count: None,
                },
                // @group(0) @binding(2) var<storage, read_write>
                // dispatch_indirect_buffer : array<IndirectDispatch>;
                BindGroupLayoutEntry {
                    binding: 2,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // @group(0) @binding(3) var<uniform> batch_metadata :
                // BatchMetadata;
                BindGroupLayoutEntry {
                    binding: 3,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: Some(GpuRenderBatchMetadata::min_size()),
                    },
                    count: None,
                },
            ],
        );

        let mergesort_pass_bind_group_layout = render_device.create_bind_group_layout(
            "hanabi:bind_group_layout:sort_mergesort_pass",
            &[
                // @group(0) @binding(0) var<storage, read_write> sort_buffer_a
                // : array<KeyValuePair>;
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: Some(NonZeroU64::new(12).unwrap()), // dual K-V pair
                    },
                    count: None,
                },
                // @group(0) @binding(1) var<storage, read_write> sort_buffer_b
                // : array<KeyValuePair>;
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: Some(NonZeroU64::new(12).unwrap()), // dual K-V pair
                    },
                    count: None,
                },
                // @group(0) @binding(2) var<storage, read> effect_sort_metadata
                // : array<EffectSortMetadata>;
                BindGroupLayoutEntry {
                    binding: 2,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // @group(0) @binding(3) var<storage, read>
                // sort_metadata_indices : array<u32>;
                BindGroupLayoutEntry {
                    binding: 3,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // @group(0) @binding(4) var<uniform> batch_metadata :
                // BatchMetadata;
                BindGroupLayoutEntry {
                    binding: 4,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: Some(GpuRenderBatchMetadata::min_size()),
                    },
                    count: None,
                },
            ],
        );

        let sort_copy_bind_group_layout = render_device.create_bind_group_layout(
            "hanabi:bind_group_layout:sort_copy",
            &[
                // @group(0) @binding(0) var<storage, read_write> indirect_index_buffer :
                // IndirectIndexBuffer;
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: Some(NonZeroU64::new(12).unwrap()), // ping/pong+dead
                    },
                    count: None,
                },
                // @group(0) @binding(1) var<storage, read> sort_buffer : array<KeyValuePair>;
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: Some(NonZeroU64::new(12).unwrap()), // dual kv pair
                    },
                    count: None,
                },
                // @group(0) @binding(2) var<storage, read_write>
                // effect_metadata : array<EffectMetadata>;
                BindGroupLayoutEntry {
                    binding: 2,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: Some(effect_metadata_min_binding_size),
                    },
                    count: None,
                },
                // @group(0) @binding(3) var<storage, read> batch_descriptor :
                // BatchDescriptor;
                BindGroupLayoutEntry {
                    binding: 3,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: true,
                        min_binding_size: Some(batch_descriptor_size),
                    },
                    count: None,
                },
                // @group(0) @binding(4) var<storage, read> batch_effect_indices
                // : array<u32>;
                BindGroupLayoutEntry {
                    binding: 4,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // @group(0) @binding(5) var<storage, read> effect_sort_metadata
                // : array<EffectSortMetadata>;
                BindGroupLayoutEntry {
                    binding: 5,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        );

        let mergesort_init_pipeline_id =
            pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
                label: Some("hanabi:pipeline:sort_mergesort_init".into()),
                layout: vec![mergesort_init_bind_group_layout.clone()],
                push_constant_ranges: vec![],
                shader: mergesort_init_shader,
                shader_defs: vec![],
                entry_point: Some("main".into()),
                zero_initialize_workgroup_memory: false,
            });

        let mergesort_pass_pipeline_id =
            pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
                label: Some("hanabi:pipeline:sort_mergesort_pass".into()),
                layout: vec![mergesort_pass_bind_group_layout.clone()],
                push_constant_ranges: vec![PushConstantRange {
                    stages: ShaderStages::COMPUTE,
                    range: 0..4,
                }],
                shader: mergesort_pass_shader,
                shader_defs: vec![],
                entry_point: Some("main".into()),
                zero_initialize_workgroup_memory: false,
            });

        let sort_copy_pipeline_id =
            pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
                label: Some("hanabi:pipeline:sort_copy".into()),
                layout: vec![sort_copy_bind_group_layout.clone()],
                shader: sort_copy_shader,
                shader_defs: vec![],
                entry_point: Some("main".into()),
                push_constant_ranges: vec![],
                zero_initialize_workgroup_memory: false,
            });

        Self {
            render_device: render_device.clone(),
            sort_fill_shader,
            sort_buffer,
            sort_temp_buffer,
            sort_indirect_batch_bind_groups: default(),
            sort_fill_bind_group_layouts: default(),
            sort_fill_bind_groups: default(),
            mergesort_init_bind_groups: default(),
            mergesort_pass_bind_groups: default(),
            sort_bind_group: default(),
            sort_indirect_batch_bind_group_layout,
            sort_copy_bind_group_layout,
            sort_bind_group_layout,
            mergesort_init_bind_group_layout,
            mergesort_pass_bind_group_layout,
            sort_indirect_batch_pipeline_id,
            sort_pipeline_id,
            mergesort_init_pipeline_id,
            mergesort_pass_pipeline_id,
            sort_copy_pipeline_id,
            sort_copy_bind_groups: default(),
        }
    }

    #[inline]
    #[allow(dead_code)]
    pub fn sort_buffer(&self) -> Option<&Buffer> {
        self.sort_buffer.buffer()
    }

    #[inline]
    pub fn sort_bind_group(&self) -> Option<&BindGroup> {
        self.sort_bind_group
            .as_ref()
            .map(|sort_bind_group| &sort_bind_group.bind_group)
    }

    #[inline]
    pub fn sort_indirect_batch_pipeline_id(&self) -> CachedComputePipelineId {
        self.sort_indirect_batch_pipeline_id
    }

    #[inline]
    pub fn sort_pipeline_id(&self) -> CachedComputePipelineId {
        self.sort_pipeline_id
    }

    pub fn write_sort_buffer(&mut self, render_device: &RenderDevice, render_queue: &RenderQueue) {
        for mut buffer in [&mut self.sort_buffer, &mut self.sort_temp_buffer] {
            if buffer.is_empty() {
                buffer.push(default());
            }
            buffer.write_buffer(render_device, render_queue);
        }
    }

    pub fn ensure_sort_fill_bind_group_layout(
        &mut self,
        pipeline_cache: &PipelineCache,
        particle_layout: &ParticleLayout,
    ) -> Result<&BindGroupLayout, ()> {
        let key = SortFillBindGroupLayoutKey::from_particle_layout(particle_layout)?;
        let (layout, _) = self
            .sort_fill_bind_group_layouts
            .entry(key)
            .or_insert_with(|| {
                let storage_alignment = self
                    .render_device
                    .limits()
                    .min_storage_buffer_offset_alignment;
                let batch_descriptor_size =
                    GpuRenderBatchDescriptor::aligned_size(storage_alignment);

                let bind_group_layout = self.render_device.create_bind_group_layout(
                    "hanabi:bind_group_layout:sort_fill",
                    &[
                        // @group(0) @binding(0) var<storage, read_write> pairs: array<KeyValuePair>;
                        BindGroupLayoutEntry {
                            binding: 0,
                            visibility: ShaderStages::COMPUTE,
                            ty: BindingType::Buffer {
                                ty: BufferBindingType::Storage { read_only: false },
                                has_dynamic_offset: false,
                                min_binding_size: Some(NonZeroU64::new(16).unwrap()), // count + dual kv pair
                            },
                            count: None,
                        },
                        // @group(0) @binding(1) var<storage, read> particle_buffer: ParticleBuffer;
                        BindGroupLayoutEntry {
                            binding: 1,
                            visibility: ShaderStages::COMPUTE,
                            ty: BindingType::Buffer {
                                ty: BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: Some(key.particle_min_binding_size.into()),
                            },
                            count: None,
                        },
                        // @group(0) @binding(2) var<storage, read> indirect_index_buffer : array<u32>;
                        BindGroupLayoutEntry {
                            binding: 2,
                            visibility: ShaderStages::COMPUTE,
                            ty: BindingType::Buffer {
                                ty: BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: Some(NonZeroU64::new(12).unwrap()), // ping/pong+dead
                            },
                            count: None,
                        },
                        // @group(0) @binding(3) var<storage, read_write>
                        // effect_metadata : array<EffectMetadata>;
                        BindGroupLayoutEntry {
                            binding: 3,
                            visibility: ShaderStages::COMPUTE,
                            ty: BindingType::Buffer {
                                ty: BufferBindingType::Storage { read_only: false },
                                has_dynamic_offset: false,
                                min_binding_size: Some(GpuEffectMetadata::aligned_size(
                                    storage_alignment,
                                )),
                            },
                            count: None,
                        },
                        // @group(0) @binding(4) var<storage, read>
                        // batch_descriptor : BatchDescriptor;
                        BindGroupLayoutEntry {
                            binding: 4,
                            visibility: ShaderStages::COMPUTE,
                            ty: BindingType::Buffer {
                                ty: BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: true,
                                min_binding_size: Some(batch_descriptor_size),
                            },
                            count: None,
                        },
                        // @group(0) @binding(5) var<storage, read>
                        // batch_effect_indices : array<BatchEffectIndices>;
                        BindGroupLayoutEntry {
                            binding: 5,
                            visibility: ShaderStages::COMPUTE,
                            ty: BindingType::Buffer {
                                ty: BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        // @group(0) @binding(6) var<storage, read_write>
                        // effect_sort_metadata : array<EffectSortMetadataAtomic>;
                        BindGroupLayoutEntry {
                            binding: 6,
                            visibility: ShaderStages::COMPUTE,
                            ty: BindingType::Buffer {
                                ty: BufferBindingType::Storage { read_only: false },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                    ],
                );
                let pipeline_id =
                    pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
                        label: Some("hanabi:pipeline:sort_fill".into()),
                        layout: vec![bind_group_layout.clone()],
                        shader: self.sort_fill_shader.clone(),
                        shader_defs: vec![],
                        entry_point: Some("main".into()),
                        push_constant_ranges: vec![],
                        zero_initialize_workgroup_memory: false,
                    });
                (bind_group_layout, pipeline_id)
            });
        Ok(layout)
    }

    // We currently only use the bind group layout internally in
    // ensure_sort_fill_bind_group()
    #[allow(dead_code)]
    pub fn get_sort_fill_bind_group_layout(
        &self,
        particle_layout: &ParticleLayout,
    ) -> Option<&BindGroupLayout> {
        let key = SortFillBindGroupLayoutKey::from_particle_layout(particle_layout).ok()?;
        self.sort_fill_bind_group_layouts
            .get(&key)
            .map(|(layout, _)| layout)
    }

    pub fn get_sort_fill_pipeline_id(
        &self,
        particle_layout: &ParticleLayout,
    ) -> Option<CachedComputePipelineId> {
        let key = SortFillBindGroupLayoutKey::from_particle_layout(particle_layout).ok()?;
        self.sort_fill_bind_group_layouts
            .get(&key)
            .map(|(_, pipeline_id)| *pipeline_id)
    }

    pub fn get_mergesort_init_pipeline_id(&self) -> CachedComputePipelineId {
        self.mergesort_init_pipeline_id
    }

    pub fn get_mergesort_pass_pipeline_id(&self) -> CachedComputePipelineId {
        self.mergesort_pass_pipeline_id
    }

    pub fn get_sort_copy_pipeline_id(&self) -> CachedComputePipelineId {
        self.sort_copy_pipeline_id
    }

    pub fn ensure_sort_indirect_batch_bind_group(
        &mut self,
        effect_metadata_buffer: &Buffer,
        effect_sort_metadata_buffer: &Buffer,
        render_batch_descriptor_buffer: &Buffer,
        batch_effect_indices_buffer: &Buffer,
        sort_dispatch_indirect_buffer: &Buffer,
        render_batch_descriptors_requiring_sorting_buffer: &Buffer,
        batch_metadata_buffer: &Buffer,
    ) -> Result<&BindGroup, ()> {
        let key = SortIndirectBatchBindGroupKey {
            effect_metadata: effect_metadata_buffer.id(),
            effect_sort_metadata: effect_sort_metadata_buffer.id(),
        };
        let entry = self.sort_indirect_batch_bind_groups.entry(key);
        let bind_group = match entry {
            Entry::Occupied(entry) => entry.into_mut(),
            Entry::Vacant(entry) => {
                entry.insert(self.render_device.create_bind_group(
                    "hanabi:bind_group:sort_indirect_batch",
                    &self.sort_indirect_batch_bind_group_layout,
                    &[
                        // @group(0) @binding(0) var<uniform> batch_metadata :
                        // BatchMetadata;
                        BindGroupEntry {
                            binding: 0,
                            resource: BindingResource::Buffer(BufferBinding {
                                buffer: batch_metadata_buffer,
                                offset: 0,
                                size: None,
                            }),
                        },
                        // @group(0) @binding(1) var<storage, read>
                        // batch_descriptors_requiring_sorting : array<u32>;
                        BindGroupEntry {
                            binding: 1,
                            resource: BindingResource::Buffer(BufferBinding {
                                buffer: render_batch_descriptors_requiring_sorting_buffer,
                                offset: 0,
                                size: None,
                            }),
                        },
                        // @group(0) @binding(2) var<storage, read>
                        // batch_descriptors : array<BatchDescriptor>;
                        BindGroupEntry {
                            binding: 2,
                            resource: BindingResource::Buffer(BufferBinding {
                                buffer: render_batch_descriptor_buffer,
                                offset: 0,
                                size: None,
                            }),
                        },
                        // @group(0) @binding(3) var<storage, read>
                        // batch_effect_indices : array<BatchEffectIndices>;
                        BindGroupEntry {
                            binding: 3,
                            resource: BindingResource::Buffer(BufferBinding {
                                buffer: batch_effect_indices_buffer,
                                offset: 0,
                                size: None,
                            }),
                        },
                        // @group(0) @binding(4) var<storage, read_write>
                        // effect_metadata : array<EffectMetadata>;
                        BindGroupEntry {
                            binding: 4,
                            resource: BindingResource::Buffer(BufferBinding {
                                buffer: effect_metadata_buffer,
                                offset: 0,
                                size: None,
                            }),
                        },
                        // @group(0) @binding(5) var<storage, read>
                        // effect_sort_metadata : array<EffectSortMetadata>;
                        BindGroupEntry {
                            binding: 5,
                            resource: BindingResource::Buffer(BufferBinding {
                                buffer: effect_sort_metadata_buffer,
                                offset: 0,
                                size: None,
                            }),
                        },
                        // @group(0) @binding(6) var<storage, read_write>
                        // dispatch_indirect_buffer :
                        // array<IndirectDispatch>;
                        BindGroupEntry {
                            binding: 6,
                            resource: BindingResource::Buffer(BufferBinding {
                                buffer: sort_dispatch_indirect_buffer,
                                offset: 0,
                                size: None,
                            }),
                        },
                    ],
                ))
            }
        };
        Ok(bind_group)
    }

    pub fn sort_indirect_batch_bind_group(
        &self,
        effect_metadata: BufferId,
        effect_sort_metadata: BufferId,
    ) -> Option<&BindGroup> {
        let key = SortIndirectBatchBindGroupKey {
            effect_metadata,
            effect_sort_metadata,
        };
        self.sort_indirect_batch_bind_groups.get(&key)
    }

    pub fn ensure_sort_fill_bind_group(
        &mut self,
        particle_layout: &ParticleLayout,
        particle: &Buffer,
        indirect_index: &Buffer,
        effect_metadata: &Buffer,
        effect_sort_metadata: &Buffer,
        render_batch_descriptor_buffer: &Buffer,
        batch_effect_indices_buffer: &Buffer,
    ) -> Result<&BindGroup, ()> {
        let sort_buffer = self
            .sort_buffer
            .buffer()
            .expect("Sort buffer must be present");
        let sort_buffer_id = sort_buffer.id();
        if self
            .sort_fill_bind_groups
            .as_ref()
            .is_some_and(|sort_fill_bind_groups| {
                sort_fill_bind_groups.sort_buffer_id != sort_buffer_id
            })
        {
            info!("Sort buffer resized; clearing old sort fill bind groups.");
            self.sort_fill_bind_groups = None;
        }

        let sort_fill_bind_groups =
            self.sort_fill_bind_groups
                .get_or_insert_with(|| SortFillBindGroups {
                    sort_buffer_id,
                    bind_groups: HashMap::default(),
                });

        let key = SortFillBindGroupKey {
            particle: particle.id(),
            indirect_index: indirect_index.id(),
            effect_metadata: effect_metadata.id(),
            effect_sort_metadata: effect_sort_metadata.id(),
        };
        let entry = sort_fill_bind_groups.bind_groups.entry(key);
        let bind_group = match entry {
            Entry::Occupied(entry) => entry.into_mut(),
            Entry::Vacant(entry) => {
                let storage_alignment = self
                    .render_device
                    .limits()
                    .min_storage_buffer_offset_alignment;
                let render_batch_descriptor_size =
                    GpuRenderBatchDescriptor::aligned_size(storage_alignment);

                // Note: can't use get_bind_group_layout() because the function call mixes the
                // lifetimes of the two hash maps and complains the bind group one is already
                // borrowed. Doing a manual access to the layout one instead makes the compiler
                // happy.
                let key = SortFillBindGroupLayoutKey::from_particle_layout(particle_layout)?;
                let layout = &self.sort_fill_bind_group_layouts.get(&key).ok_or(())?.0;
                entry.insert(self.render_device.create_bind_group(
                    "hanabi:bind_group:sort_fill",
                    layout,
                    &[
                        // @group(0) @binding(0) var<storage, read_write> pairs:
                        // array<KeyValuePair>;
                        BindGroupEntry {
                            binding: 0,
                            resource: BindingResource::Buffer(BufferBinding {
                                buffer: sort_buffer,
                                offset: 0,
                                size: None,
                            }),
                        },
                        // @group(0) @binding(1) var<storage, read> particle_buffer:
                        // ParticleBuffer;
                        BindGroupEntry {
                            binding: 1,
                            resource: BindingResource::Buffer(BufferBinding {
                                buffer: particle,
                                offset: 0,
                                size: None,
                            }),
                        },
                        // @group(0) @binding(2) var<storage, read> indirect_index_buffer :
                        // array<u32>;
                        BindGroupEntry {
                            binding: 2,
                            resource: BindingResource::Buffer(BufferBinding {
                                buffer: indirect_index,
                                offset: 0,
                                size: None,
                            }),
                        },
                        // @group(0) @binding(3) var<storage, read> effect_metadata :
                        // array<EffectMetadata>;
                        BindGroupEntry {
                            binding: 3,
                            resource: BindingResource::Buffer(BufferBinding {
                                buffer: effect_metadata,
                                offset: 0,
                                size: None,
                            }),
                        },
                        // @group(0) @binding(4) var<storage, read>
                        // batch_descriptor : BatchDescriptor;
                        BindGroupEntry {
                            binding: 4,
                            resource: BindingResource::Buffer(BufferBinding {
                                buffer: render_batch_descriptor_buffer,
                                offset: 0,
                                size: Some(render_batch_descriptor_size),
                            }),
                        },
                        // @group(0) @binding(5) var<storage, read>
                        // batch_effect_indices : array<BatchEffectIndices>;
                        BindGroupEntry {
                            binding: 5,
                            resource: BindingResource::Buffer(BufferBinding {
                                buffer: batch_effect_indices_buffer,
                                offset: 0,
                                size: None,
                            }),
                        },
                        // @group(0) @binding(6) var<storage, read_write>
                        // effect_sort_metadata : array<EffectSortMetadataAtomic>;
                        BindGroupEntry {
                            binding: 6,
                            resource: BindingResource::Buffer(BufferBinding {
                                buffer: effect_sort_metadata,
                                offset: 0,
                                size: None,
                            }),
                        },
                    ],
                ))
            }
        };
        Ok(bind_group)
    }

    pub fn sort_fill_bind_group(
        &self,
        particle: BufferId,
        indirect_index: BufferId,
        effect_metadata: BufferId,
        effect_sort_metadata: BufferId,
    ) -> Option<&BindGroup> {
        let sort_buffer_id = self
            .sort_buffer
            .buffer()
            .expect("Sort buffer must be present")
            .id();
        self.sort_fill_bind_groups
            .as_ref()
            .and_then(|sort_fill_bind_groups| {
                if sort_buffer_id == sort_fill_bind_groups.sort_buffer_id {
                    let key = SortFillBindGroupKey {
                        particle,
                        indirect_index,
                        effect_metadata,
                        effect_sort_metadata,
                    };
                    sort_fill_bind_groups.bind_groups.get(&key)
                } else {
                    None
                }
            })
    }

    pub fn ensure_sort_mergesort_init_bind_group(
        &mut self,
        effect_sort_metadata: &Buffer,
        sort_metadata_indices_buffer: &Buffer,
        sort_metadata_indices_count: usize,
        mergesort_dispatch_indirect_buffer: &Buffer,
        batch_metadata_buffer: &Buffer,
    ) -> Result<&BindGroup, ()> {
        let mergesort_init_bind_groups =
            self.mergesort_init_bind_groups
                .get_or_insert_with(|| SortMergesortInitBindGroups {
                    bind_groups: HashMap::default(),
                });

        let key = SortMergesortInitBindGroupKey {
            effect_sort_metadata: effect_sort_metadata.id(),
            sort_metadata_indices: sort_metadata_indices_buffer.id(),
            mergesort_dispatch_indirect: mergesort_dispatch_indirect_buffer.id(),
        };

        let entry = mergesort_init_bind_groups.bind_groups.entry(key);
        let bind_group = match entry {
            Entry::Occupied(entry) => entry.into_mut(),
            Entry::Vacant(entry) => {
                entry.insert(
                    self.render_device.create_bind_group(
                        "hanabi:bind_group:sort_mergesort_init",
                        &self.mergesort_init_bind_group_layout,
                        &[
                            // @group(0) @binding(0) var<storage, read_write>
                            // effect_sort_metadata : array<EffectSortMetadata>;
                            BindGroupEntry {
                                binding: 0,
                                resource: BindingResource::Buffer(BufferBinding {
                                    buffer: effect_sort_metadata,
                                    offset: 0,
                                    size: None,
                                }),
                            },
                            // @group(0) @binding(1) var<storage, read>
                            // sort_metadata_indices : array<u32>;
                            BindGroupEntry {
                                binding: 1,
                                resource: BindingResource::Buffer(BufferBinding {
                                    buffer: sort_metadata_indices_buffer,
                                    offset: 0,
                                    size: Some(
                                        NonZeroU64::try_from(
                                            (sort_metadata_indices_count as u64).max(1)
                                                * u64::from(u32::min_size()),
                                        )
                                        .unwrap(),
                                    ),
                                }),
                            },
                            // @group(0) @binding(2) var<storage, read_write>
                            // dispatch_indirect_buffer :
                            // array<IndirectDispatch>;
                            BindGroupEntry {
                                binding: 2,
                                resource: BindingResource::Buffer(BufferBinding {
                                    buffer: mergesort_dispatch_indirect_buffer,
                                    offset: 0,
                                    size: None,
                                }),
                            },
                            // @group(0) @binding(3) var<uniform> batch_metadata
                            // : BatchMetadata;
                            BindGroupEntry {
                                binding: 3,
                                resource: BindingResource::Buffer(BufferBinding {
                                    buffer: batch_metadata_buffer,
                                    offset: 0,
                                    size: None,
                                }),
                            },
                        ],
                    ),
                )
            }
        };

        Ok(bind_group)
    }

    pub fn sort_mergesort_init_bind_group(
        &self,
        effect_sort_metadata: BufferId,
        sort_metadata_indices: BufferId,
        mergesort_dispatch_indirect: BufferId,
    ) -> Option<&BindGroup> {
        self.mergesort_init_bind_groups
            .as_ref()
            .and_then(|mergesort_init_bind_groups| {
                let key = SortMergesortInitBindGroupKey {
                    effect_sort_metadata,
                    sort_metadata_indices,
                    mergesort_dispatch_indirect,
                };
                mergesort_init_bind_groups.bind_groups.get(&key)
            })
    }

    pub fn ensure_sort_mergesort_pass_bind_group(
        &mut self,
        effect_sort_metadata_buffer: &Buffer,
        sort_metadata_indices_buffer: &Buffer,
        sort_metadata_indices_count: usize,
        batch_metadata_buffer: &Buffer,
    ) -> Result<&BindGroup, ()> {
        let (&Some(ref sort_buffer), &Some(ref sort_temp_buffer)) =
            (&self.sort_buffer.buffer(), &self.sort_temp_buffer.buffer())
        else {
            return Err(());
        };

        let mergesort_pass_bind_groups =
            self.mergesort_pass_bind_groups
                .get_or_insert_with(|| SortMergesortPassBindGroups {
                    bind_groups: HashMap::default(),
                });

        let key = SortMergesortPassBindGroupKey {
            effect_sort_metadata: effect_sort_metadata_buffer.id(),
            sort_metadata_indices: sort_metadata_indices_buffer.id(),
        };

        let entry = mergesort_pass_bind_groups.bind_groups.entry(key);
        let bind_group = match entry {
            Entry::Occupied(entry) => entry.into_mut(),
            Entry::Vacant(entry) => {
                entry.insert(
                    self.render_device.create_bind_group(
                        "hanabi:bind_group:sort_mergesort_pass",
                        &self.mergesort_pass_bind_group_layout,
                        &[
                            // @group(0) @binding(0) var<storage, read_write>
                            // sort_buffer_a : array<KeyValuePair>;
                            BindGroupEntry {
                                binding: 0,
                                resource: BindingResource::Buffer(BufferBinding {
                                    buffer: sort_buffer,
                                    offset: 0,
                                    size: None,
                                }),
                            },
                            // @group(0) @binding(1) var<storage, read_write>
                            // sort_buffer_b : array<KeyValuePair>;
                            BindGroupEntry {
                                binding: 1,
                                resource: BindingResource::Buffer(BufferBinding {
                                    buffer: sort_temp_buffer,
                                    offset: 0,
                                    size: None,
                                }),
                            },
                            // @group(0) @binding(2) var<storage, read>
                            // effect_sort_metadata : array<EffectSortMetadata>;
                            BindGroupEntry {
                                binding: 2,
                                resource: BindingResource::Buffer(BufferBinding {
                                    buffer: effect_sort_metadata_buffer,
                                    offset: 0,
                                    size: None,
                                }),
                            },
                            // @group(0) @binding(3) var<storage, read>
                            // sort_metadata_indices : array<u32>;
                            BindGroupEntry {
                                binding: 3,
                                resource: BindingResource::Buffer(BufferBinding {
                                    buffer: sort_metadata_indices_buffer,
                                    offset: 0,
                                    size: Some(
                                        NonZeroU64::try_from(
                                            (sort_metadata_indices_count as u64).max(1)
                                                * u64::from(u32::min_size()),
                                        )
                                        .unwrap(),
                                    ),
                                }),
                            },
                            // @group(0) @binding(4) var<uniform> batch_metadata
                            // : BatchMetadata;
                            BindGroupEntry {
                                binding: 4,
                                resource: BindingResource::Buffer(BufferBinding {
                                    buffer: batch_metadata_buffer,
                                    offset: 0,
                                    size: None,
                                }),
                            },
                        ],
                    ),
                )
            }
        };

        Ok(bind_group)
    }

    pub fn sort_mergesort_pass_bind_group(
        &self,
        effect_sort_metadata: BufferId,
        sort_metadata_indices: BufferId,
    ) -> Option<&BindGroup> {
        self.mergesort_pass_bind_groups
            .as_ref()
            .and_then(|mergesort_pass_bind_groups| {
                let key = SortMergesortPassBindGroupKey {
                    effect_sort_metadata,
                    sort_metadata_indices,
                };
                mergesort_pass_bind_groups.bind_groups.get(&key)
            })
    }

    pub fn ensure_sort_copy_bind_group(
        &mut self,
        indirect_index_buffer: &Buffer,
        effect_metadata_buffer: &Buffer,
        effect_sort_metadata_buffer: &Buffer,
        render_batch_descriptor_buffer: &Buffer,
        batch_effect_indices_buffer: &Buffer,
    ) -> Result<&BindGroup, ()> {
        let sort_buffer = self
            .sort_buffer
            .buffer()
            .expect("Sort buffer must be present");
        let sort_buffer_id = sort_buffer.id();
        if self
            .sort_copy_bind_groups
            .as_ref()
            .is_some_and(|sort_copy_bind_groups| {
                sort_copy_bind_groups.sort_buffer_id != sort_buffer_id
            })
        {
            info!("Sort buffer resized; clearing old sort copy bind groups.");
            self.sort_copy_bind_groups = None;
        }

        let sort_copy_bind_groups =
            self.sort_copy_bind_groups
                .get_or_insert_with(|| SortCopyBindGroups {
                    sort_buffer_id,
                    bind_groups: HashMap::default(),
                });

        let key = SortCopyBindGroupKey {
            indirect_index: indirect_index_buffer.id(),
            effect_metadata: effect_metadata_buffer.id(),
            effect_sort_metadata: effect_sort_metadata_buffer.id(),
        };
        let entry = sort_copy_bind_groups.bind_groups.entry(key);
        let bind_group = match entry {
            Entry::Occupied(entry) => entry.into_mut(),
            Entry::Vacant(entry) => {
                let storage_alignment = self
                    .render_device
                    .limits()
                    .min_storage_buffer_offset_alignment;
                let render_batch_descriptor_size =
                    GpuRenderBatchDescriptor::aligned_size(storage_alignment);

                entry.insert(self.render_device.create_bind_group(
                    "hanabi:bind_group:sort_copy",
                    &self.sort_copy_bind_group_layout,
                    &[
                        // @group(0) @binding(0) var<storage, read_write> indirect_index_buffer
                        // : IndirectIndexBuffer;
                        BindGroupEntry {
                            binding: 0,
                            resource: BindingResource::Buffer(BufferBinding {
                                buffer: indirect_index_buffer,
                                offset: 0,
                                size: None,
                            }),
                        },
                        // @group(0) @binding(1) var<storage, read> sort_buffer : SortBuffer;
                        BindGroupEntry {
                            binding: 1,
                            resource: BindingResource::Buffer(BufferBinding {
                                buffer: sort_buffer,
                                offset: 0,
                                size: None,
                            }),
                        },
                        // @group(0) @binding(2) var<storage, read> effect_metadata :
                        // EffectMetadata;
                        BindGroupEntry {
                            binding: 2,
                            resource: BindingResource::Buffer(BufferBinding {
                                buffer: effect_metadata_buffer,
                                offset: 0,
                                size: None,
                            }),
                        },
                        // @group(0) @binding(3) var<storage, read>
                        // batch_descriptor : BatchDescriptor;
                        BindGroupEntry {
                            binding: 3,
                            resource: BindingResource::Buffer(BufferBinding {
                                buffer: render_batch_descriptor_buffer,
                                offset: 0,
                                size: Some(render_batch_descriptor_size),
                            }),
                        },
                        // @group(0) @binding(4) var<storage, read>
                        // batch_effect_indices : array<BatchEffectIndices>;
                        BindGroupEntry {
                            binding: 4,
                            resource: BindingResource::Buffer(BufferBinding {
                                buffer: batch_effect_indices_buffer,
                                offset: 0,
                                size: None,
                            }),
                        },
                        // @group(0) @binding(5) var<storage, read>
                        // effect_sort_metadata : EffectSortMetadata;
                        BindGroupEntry {
                            binding: 5,
                            resource: BindingResource::Buffer(BufferBinding {
                                buffer: effect_sort_metadata_buffer,
                                offset: 0,
                                size: None,
                            }),
                        },
                    ],
                ))
            }
        };
        Ok(bind_group)
    }

    pub fn sort_copy_bind_group(
        &self,
        indirect_index: BufferId,
        effect_metadata: BufferId,
        effect_sort_metadata: BufferId,
    ) -> Option<&BindGroup> {
        let sort_buffer_id = self
            .sort_buffer
            .buffer()
            .expect("Sort buffer must be present")
            .id();
        self.sort_copy_bind_groups
            .as_ref()
            .and_then(|sort_copy_bind_groups| {
                if sort_buffer_id == sort_copy_bind_groups.sort_buffer_id {
                    let key = SortCopyBindGroupKey {
                        indirect_index,
                        effect_metadata,
                        effect_sort_metadata,
                    };
                    sort_copy_bind_groups.bind_groups.get(&key)
                } else {
                    None
                }
            })
    }

    pub(crate) fn ensure_sort_bind_group(
        &mut self,
        effect_sort_metadata: &Buffer,
        sort_metadata_indices_buffer: &Buffer,
        sort_metadata_indices_count: usize,
    ) -> Result<&BindGroup, ()> {
        let sort_buffer = self
            .sort_buffer
            .buffer()
            .expect("Sort buffer must be present");

        if self.sort_bind_group.as_ref().is_none_or(|sort_bind_group| {
            sort_bind_group.sort_metadata_buffer_id != effect_sort_metadata.id()
                || sort_bind_group.sort_buffer_id != sort_buffer.id()
                || sort_bind_group.sort_metadata_indices_buffer_id
                    != sort_metadata_indices_buffer.id()
                || sort_bind_group.sort_metadata_indices_count != sort_metadata_indices_count
        }) {
            let sort_bind_group = self.render_device.create_bind_group(
                "hanabi:bind_group:sort",
                &self.sort_bind_group_layout,
                &[
                    // @group(0) @binding(0) var<storage, read_write> pairs : array<KeyValuePair>;
                    BindGroupEntry {
                        binding: 0,
                        resource: BindingResource::Buffer(BufferBinding {
                            buffer: sort_buffer,
                            offset: 0,
                            size: None,
                        }),
                    },
                    // @group(0) @binding(1) var<storage, read> effect_sort_metadata
                    // : array<EffectSortMetadata>;
                    BindGroupEntry {
                        binding: 1,
                        resource: BindingResource::Buffer(BufferBinding {
                            buffer: effect_sort_metadata,
                            offset: 0,
                            size: None,
                        }),
                    },
                    // @group(0) @binding(2) var<storage, read>
                    // sort_metadata_indices : array<u32>;
                    BindGroupEntry {
                        binding: 2,
                        resource: BindingResource::Buffer(BufferBinding {
                            buffer: sort_metadata_indices_buffer,
                            offset: 0,
                            size: Some(
                                NonZeroU64::try_from(
                                    (sort_metadata_indices_count as u64).max(1)
                                        * u64::from(u32::min_size()),
                                )
                                .unwrap(),
                            ),
                        }),
                    },
                ],
            );

            self.sort_bind_group = Some(CachedSortBindGroup {
                sort_buffer_id: sort_buffer.id(),
                sort_metadata_buffer_id: effect_sort_metadata.id(),
                sort_metadata_indices_buffer_id: sort_metadata_indices_buffer.id(),
                sort_metadata_indices_count,
                bind_group: sort_bind_group,
            });
        }

        self.sort_bind_group
            .as_ref()
            .map(|cached_sort_bind_group| &cached_sort_bind_group.bind_group)
            .ok_or(())
    }

    pub(crate) fn clear_sort_buffer(&mut self) {
        self.sort_buffer.clear();
        self.sort_temp_buffer.clear();
    }

    pub(crate) fn allocate_sort_buffer_slots(&mut self, len: u32) -> Range<u32> {
        // FIXME: This is really inefficient. We don't need the CPU side buffer
        // at all!
        let start = self.sort_buffer.len() as u32;

        for buffer in [&mut self.sort_buffer, &mut self.sort_temp_buffer] {
            buffer.reserve_internal(len as usize);
            for _ in 0..len {
                buffer.push(default());
            }
        }

        start..(start + len)
    }
}

pub fn compute_mergesort_dispatch_count(particle_count: u32) -> u32 {
    if particle_count == 0 {
        0
    } else {
        //32 - (particle_count - 1).leading_zeros()
        // FIXME
        20
    }
}
