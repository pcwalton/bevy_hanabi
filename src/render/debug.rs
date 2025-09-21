//! Debugging utilities.

use std::{
    fs::{self, OpenOptions},
    io::{BufWriter, Seek, Write},
    iter, mem,
    path::PathBuf,
    sync::{Arc, Mutex},
};

use bevy::{
    asset::AssetId,
    ecs::system::{Query, Res, ResMut},
    log::{error, info, warn},
    platform::collections::HashMap,
    render::{
        render_resource::{Buffer, ShaderSize},
        renderer::{RenderDevice, RenderQueue},
        sync_world::{MainEntity, MainEntityHashMap},
    },
    time::Time,
};
use wgpu::{BufferDescriptor, BufferUsages, CommandEncoderDescriptor, MapMode};

use crate::{
    render::{
        batch::InstanceInput,
        effect_cache::{CachedEffect, DispatchBufferIndices},
        EffectCache, EffectsMeta, GpuEffectMetadata, RenderDebugSettings,
    },
    ParticleLayout, ScalarType, ValueType,
};

// This needs to be run right after `render_system`.
pub(crate) fn debug_dump_buffers(
    q_cached_effects: Query<(
        &MainEntity,
        &CachedEffect,
        &DispatchBufferIndices,
        &InstanceInput,
    )>,
    render_device: ResMut<RenderDevice>,
    render_queue: ResMut<RenderQueue>,
    debug_settings: Res<RenderDebugSettings>,
    effects_meta: Res<EffectsMeta>,
    effect_cache: Res<EffectCache>,
    time: Res<Time>,
) {
    // TODO: Use this!
    let Some(ref output_directory_path) = debug_settings.dump_particles_to_csv else {
        return;
    };

    if let Err(err) = fs::create_dir_all(output_directory_path) {
        error!(
            "Failed to create `{}`: {:?}",
            output_directory_path.display(),
            err
        );
        return;
    };
    info!(
        "Dumping particle effects to `{}`",
        output_directory_path.display()
    );

    // Read back effect metadata buffer.
    let Some(effect_metadata_buffer) = effects_meta.effect_metadata_buffer.buffer() else {
        return;
    };

    let mut needed_effect_buffers: HashMap<u32, Vec<MainEntity>> = HashMap::new();
    let mut needed_readback_effect_info: MainEntityHashMap<ReadbackEffectInfo> =
        MainEntityHashMap::default();
    for (effect_entity, cached_effect, dispatch_buffer_indices, instance_input) in &q_cached_effects
    {
        needed_readback_effect_info.insert(
            *effect_entity,
            ReadbackEffectInfo {
                name: match instance_input.handle.id() {
                    AssetId::Index { index, marker: _ } => format!("{:x}", index.to_bits()),
                    AssetId::Uuid { uuid } => uuid.to_string(),
                },
                dispatch_buffer_indices: *dispatch_buffer_indices,
                particle_layout: instance_input.particle_layout.clone(),
                effect_buffer_index: cached_effect.buffer_index,
            },
        );

        if effect_cache
            .get_buffer(cached_effect.buffer_index)
            .is_none()
        {
            warn!(
                "Skipping effect {:?} because its effect buffer doesn't exist.",
                effect_entity
            );
            continue;
        }
        needed_effect_buffers
            .entry(cached_effect.buffer_index)
            .or_default()
            .push(*effect_entity);
    }

    let mut needed_source_buffers = HashMap::from([(
        ReadbackBufferKey::EffectMetadata,
        effect_metadata_buffer.clone(),
    )]);

    // Accumulate effect buffers.

    for &needed_effect_buffer in needed_effect_buffers.keys() {
        let effect_buffer = effect_cache
            .get_buffer(needed_effect_buffer)
            .expect("Should have rejected nonexistent buffers above");
        needed_source_buffers.insert(
            ReadbackBufferKey::Particle {
                effect_buffer_index: needed_effect_buffer,
            },
            effect_buffer.particle_buffer().clone(),
        );
        needed_source_buffers.insert(
            ReadbackBufferKey::IndirectIndex {
                effect_buffer_index: needed_effect_buffer,
            },
            effect_buffer.indirect_index_buffer().clone(),
        );
    }

    let mut needed_staging_buffers = HashMap::new();
    for (&buffer_key, buffer) in &needed_source_buffers {
        let staging_buffer = render_device.create_buffer(&BufferDescriptor {
            label: Some(&*format!("hanabi:buffer:staging@{:?}", buffer_key)),
            size: buffer.size(),
            usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        needed_staging_buffers.insert(buffer_key, staging_buffer);
    }

    let mut command_encoder = render_device.create_command_encoder(&CommandEncoderDescriptor {
        label: Some("hanabi:command_encoder:debug_dump_buffers"),
    });
    for (buffer_key, source_buffer) in &needed_source_buffers {
        let dest_buffer = &needed_staging_buffers[buffer_key];
        command_encoder.copy_buffer_to_buffer(
            source_buffer,
            0,
            dest_buffer,
            0,
            source_buffer.size(),
        );
    }
    let command_buffer = command_encoder.finish();
    render_queue.submit(iter::once(command_buffer));

    // Callback counter pattern.

    let dumper = Arc::new(Mutex::new(DebugBufferDumper {
        mapped_buffer_count: 0,
        readback_effect_info: needed_readback_effect_info,
        staging_buffers: needed_staging_buffers.clone(),
        output_directory_path: (*output_directory_path).clone(),
        time: time.elapsed_secs(),
    }));

    let needed_source_buffer_count = needed_staging_buffers.len();

    for (buffer_key, staging_buffer) in needed_staging_buffers {
        let dumper = dumper.clone();
        render_device.map_buffer(&staging_buffer.slice(..), MapMode::Read, move |result| {
            if let Err(err) = result {
                error!("Failed to map buffer {:?}: {:?}", buffer_key, err);
                return;
            }

            let mut dumper = dumper.lock().unwrap();
            dumper.mapped_buffer_count += 1;
            if dumper.mapped_buffer_count as usize == needed_source_buffer_count {
                dumper.dump_and_unmap();
            }
        });
    }
}

struct DebugBufferDumper {
    mapped_buffer_count: u32,
    readback_effect_info: MainEntityHashMap<ReadbackEffectInfo>,
    staging_buffers: HashMap<ReadbackBufferKey, Buffer>,
    output_directory_path: PathBuf,
    time: f32,
}

impl DebugBufferDumper {
    fn dump_and_unmap(&mut self) {
        {
            let effects_metadata = self.staging_buffers[&ReadbackBufferKey::EffectMetadata]
                .slice(..)
                .get_mapped_range();
            let effects_metadata_aligned_size =
                <GpuEffectMetadata as ShaderSize>::SHADER_SIZE.get() as usize;

            for (main_entity, readback_effect_info) in &self.readback_effect_info {
                let csv_file_path = self.output_directory_path.join(format!(
                    "{}-{}-{}.csv",
                    main_entity.index(),
                    main_entity.generation(),
                    readback_effect_info.name
                ));
                let mut csv_file = match OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(&csv_file_path)
                {
                    Ok(csv_file) => BufWriter::new(csv_file),
                    Err(err) => {
                        error!("Failed to open `{}`: {:?}", csv_file_path.display(), err);
                        continue;
                    }
                };

                let effect_metadata_start_pos = effects_metadata_aligned_size
                    * (readback_effect_info
                        .dispatch_buffer_indices
                        .effect_metadata_buffer_table_id
                        .0 as usize);
                let effect_metadata: &[GpuEffectMetadata] = bytemuck::cast_slice(
                    &effects_metadata[effect_metadata_start_pos
                        ..(effect_metadata_start_pos + mem::size_of::<GpuEffectMetadata>())],
                );
                let effect_metadata = &effect_metadata[0];

                let indirect_buffer = &self.staging_buffers[&ReadbackBufferKey::IndirectIndex {
                    effect_buffer_index: readback_effect_info.effect_buffer_index,
                }];
                let indirect_indices = indirect_buffer.slice(..).get_mapped_range();
                let indirect_indices: &[u32] = bytemuck::cast_slice(&indirect_indices);

                let particle_buffer = &self.staging_buffers[&ReadbackBufferKey::Particle {
                    effect_buffer_index: readback_effect_info.effect_buffer_index,
                }];
                let particle_data = &*particle_buffer.slice(..).get_mapped_range();
                let particle_size = readback_effect_info.particle_layout.size() as usize;

                if csv_file.stream_position().is_ok_and(|pos| pos == 0) {
                    let _ = write!(csv_file, "time,particle_index");
                    for attribute_layout in readback_effect_info.particle_layout.attributes() {
                        let _ = write!(csv_file, ",{}", attribute_layout.attribute.name());
                    }
                    let _ = writeln!(csv_file);
                }

                for indirect_particle_index in 0..effect_metadata.alive_count {
                    // Always write into ping, read from pong
                    let update_write_index = effect_metadata.ping;
                    let particle_index = indirect_indices[3
                        * (indirect_particle_index as usize
                            + effect_metadata.base_instance as usize)
                        + update_write_index as usize];

                    write_row(
                        &mut csv_file,
                        self.time,
                        particle_index,
                        &particle_data[(particle_index as usize * particle_size)
                            ..((particle_index as usize + 1) * particle_size)],
                        readback_effect_info,
                    );
                    let _ = writeln!(csv_file);
                }
            }
        }

        for staging_buffer in self.staging_buffers.values_mut() {
            staging_buffer.unmap();
        }
    }
}

fn write_row(
    writer: &mut impl Write,
    time: f32,
    particle_index: u32,
    particle_data: &[u8],
    readback_effect_info: &ReadbackEffectInfo,
) {
    let _ = write!(writer, "{},{}", time, particle_index);

    for attribute_layout in readback_effect_info.particle_layout.attributes() {
        let _ = write!(writer, ",");

        let attribute_data = &particle_data[(attribute_layout.offset as usize)
            ..(attribute_layout.offset as usize + attribute_layout.attribute.size())];
        match attribute_layout.attribute.value_type() {
            ValueType::Scalar(scalar_type) => write_scalar(writer, scalar_type, attribute_data),
            ValueType::Vector(vector_type) => {
                let _ = write!(writer, "\"");
                for element in 0..vector_type.count() {
                    if element > 0 {
                        let _ = write!(writer, ",");
                    }
                    let elem_type = vector_type.elem_type();
                    write_scalar(
                        writer,
                        elem_type,
                        &attribute_data
                            [(elem_type.size() * element)..(elem_type.size() * (element + 1))],
                    );
                }
                let _ = write!(writer, "\"");
            }
            ValueType::Matrix(_) => {
                let _ = write!(writer, "TodoMatrixOutput");
            }
        }
    }
}

fn write_scalar(writer: &mut impl Write, scalar_type: ScalarType, data: &[u8]) {
    match scalar_type {
        ScalarType::Bool => {
            let _ = write!(writer, "{}", if data[0] == 0 { "false" } else { "true" });
        }
        ScalarType::Float => {
            let data: &[f32] = bytemuck::cast_slice(data);
            let _ = write!(writer, "{}", data[0]);
        }
        ScalarType::Int => {
            let data: &[i32] = bytemuck::cast_slice(data);
            let _ = write!(writer, "{}", data[0]);
        }
        ScalarType::Uint => {
            let data: &[u32] = bytemuck::cast_slice(data);
            let _ = write!(writer, "{}", data[0]);
        }
    }
}

struct ReadbackEffectInfo {
    name: String,
    dispatch_buffer_indices: DispatchBufferIndices,
    particle_layout: ParticleLayout,
    effect_buffer_index: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum ReadbackBufferKey {
    Particle { effect_buffer_index: u32 },
    IndirectIndex { effect_buffer_index: u32 },
    EffectMetadata,
}
