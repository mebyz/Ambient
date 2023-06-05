use std::{
    collections::{HashMap, HashSet},
    sync::Arc,
};

use ambient_ecs::{query, ArchetypeFilter, EntityId, FramedEventsReader, QueryState, World};
use ambient_gpu::{
    gpu::Gpu,
    multi_buffer::{MultiBufferSizeStrategy, SubBufferId, TypedMultiBuffer},
    shader_module::{GraphicsPipeline, GraphicsPipelineInfo},
};
use ambient_std::asset_cache::AssetCache;
use glam::{uvec2, UVec2, UVec4};
use itertools::Itertools;
use wgpu::DepthBiasState;

use super::{
    double_sided, lod::cpu_lod_visible, primitives, CollectPrimitive, DrawIndexedIndirect, FSMain,
    PrimitiveIndex, RendererCollectState, RendererResources, RendererShader, SharedMaterial,
};
use crate::{bind_groups::BindGroups, is_transparent, scissors, PostSubmitFunc, RendererConfig};

pub struct TreeRendererConfig {
    pub renderer_config: RendererConfig,
    pub filter: ArchetypeFilter,
    pub targets: Vec<Option<wgpu::ColorTargetState>>,
    pub renderer_resources: RendererResources,
    pub fs_main: FSMain,
    pub opaque_only: bool,
    pub depth_stencil: bool,
    pub cull_mode: Option<wgpu::Face>,
    pub depth_bias: DepthBiasState,
}

pub struct TreeRenderer {
    config: Arc<TreeRendererConfig>,
    tree: HashMap<String, ShaderNode>,
    entity_primitive_count: HashMap<EntityId, usize>,
    primitives_lookup: HashMap<(EntityId, PrimitiveIndex), (String, String, usize)>,
    loc_changed_reader: FramedEventsReader<EntityId>,

    primitives: TypedMultiBuffer<CollectPrimitive>,
    primitives_bind_group: Option<wgpu::BindGroup>,
    spawn_qs: QueryState,
    despawn_qs: QueryState,
    material_indices: MaterialIndices,
}
impl TreeRenderer {
    pub fn new(gpu: &Gpu, config: TreeRendererConfig) -> Self {
        Self {
            tree: HashMap::new(),
            entity_primitive_count: HashMap::new(),
            primitives_lookup: HashMap::new(),
            loc_changed_reader: FramedEventsReader::new(),

            primitives_bind_group: None,
            primitives: TypedMultiBuffer::new(
                gpu,
                "TreeRenderer.primitives",
                wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::INDIRECT,
                MultiBufferSizeStrategy::Pow2,
            ),

            config: Arc::new(config),
            spawn_qs: QueryState::new(),
            despawn_qs: QueryState::new(),
            material_indices: MaterialIndices::new(),
        }
    }
    fn create_primitives_bind_group(
        gpu: &Gpu,
        layout: &wgpu::BindGroupLayout,
        buffer: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: buffer.as_entire_binding(),
            }],
            label: Some("TreeRenderer.primitives"),
        })
    }
    #[ambient_profiling::function]
    pub fn update(&mut self, gpu: &Gpu, assets: &AssetCache, world: &mut World) {
        tracing::debug!("Updating TreeRenderer");
        let mut to_update = HashSet::new();
        let mut spawn_qs = std::mem::replace(&mut self.spawn_qs, QueryState::new());
        let mut despawn_qs = std::mem::replace(&mut self.despawn_qs, QueryState::new());

        for (id, (primitives,)) in query((primitives().changed(),))
            .optional_changed(cpu_lod_visible())
            .filter(&self.config.filter)
            .iter(world, Some(&mut spawn_qs))
        {
            if let Some(primitive_count) = self.entity_primitive_count.get(&id) {
                for primitive_index in 0..*primitive_count {
                    if let Some(update) = self.remove_primitive(id, primitive_index) {
                        to_update.insert(update);
                    }
                }
            }
            for (primitive_index, primitive) in primitives.iter().enumerate() {
                let primitive_shader = (primitive.shader)(assets, &self.config.renderer_config);
                if let Some(update) = self.insert(
                    gpu,
                    world,
                    id,
                    primitive_index,
                    &primitive_shader,
                    &primitive.material,
                ) {
                    to_update.insert(update);
                }
            }
            self.entity_primitive_count.insert(id, primitives.len());
        }

        for (id, _) in query(())
            .incl(primitives())
            .filter(&self.config.filter)
            .despawned()
            .iter(world, Some(&mut despawn_qs))
        {
            if let Some(primitive_count) = self.entity_primitive_count.get(&id) {
                for primitive_index in 0..*primitive_count {
                    if let Some(update) = self.remove_primitive(id, primitive_index) {
                        to_update.insert(update);
                    }
                }
            }
            self.entity_primitive_count.remove(&id);
        }

        self.spawn_qs = spawn_qs;
        self.despawn_qs = despawn_qs;
        self.clean_empty(gpu);
        for (_, id) in self.loc_changed_reader.iter(world.loc_changed()) {
            if let Ok(primitives) = world.get_ref(*id, primitives()) {
                for primivite_index in 0..primitives.len() {
                    if let Some((shader_id, material_id, _)) =
                        self.primitives_lookup.get(&(*id, primivite_index))
                    {
                        to_update.insert((shader_id.clone(), material_id.clone()));
                    }
                }
            }
        }

        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("TreeRenderer.update"),
            });
        let mut primitives_to_write = Vec::new();
        for (shader_id, material_id) in to_update.into_iter() {
            if let Some(shader) = self.tree.get(&shader_id) {
                if let Some(mat) = shader.tree.get(&material_id) {
                    let primitives = mat
                        .primitives
                        .iter()
                        .map(|(id, primitive_index)| {
                            CollectPrimitive::from_primitive(
                                world,
                                *id,
                                *primitive_index,
                                mat.material_index,
                            )
                        })
                        .collect_vec();
                    self.primitives
                        .resize_buffer_with_encoder(
                            gpu,
                            &mut encoder,
                            mat.primitives_subbuffer,
                            primitives.len() as u64,
                        )
                        .unwrap();
                    primitives_to_write.push((mat.primitives_subbuffer, primitives));
                }
            }
        }

        gpu.queue.submit(Some(encoder.finish()));
        for (subbuffer, primitives) in primitives_to_write.into_iter() {
            self.primitives
                .write(gpu, subbuffer, 0, &primitives)
                .unwrap();
        }

        for node in self.tree.values_mut() {
            for mat in node.tree.values_mut() {
                // TODO: Materials can be shared between many renderers, so this should be moved
                // somewhere where it's done just once for all of them
                mat.material.update(gpu, world);
            }
        }

        self.primitives_bind_group = if self.primitives.total_len() > 0 {
            Some(Self::create_primitives_bind_group(
                gpu,
                &self.config.renderer_resources.primitives_layout,
                self.primitives.buffer(),
            ))
        } else {
            None
        };
    }
    pub fn run_collect(
        &self,
        gpu: &Gpu,
        assets: &AssetCache,
        encoder: &mut wgpu::CommandEncoder,
        post_submit: &mut Vec<PostSubmitFunc>,
        resources_bind_group: &wgpu::BindGroup,
        entities_bind_group: &wgpu::BindGroup,
        collect_state: &mut RendererCollectState,
    ) {
        tracing::debug!(
            "Collecting tree renderer {:?} {:?}",
            self.primitives.total_len(),
            self.tree.keys().collect_vec()
        );
        let mut material_layouts = vec![UVec2::ZERO; self.material_indices.counter as usize];
        for node in self.tree.values() {
            for mat in node.tree.values() {
                let offset = self
                    .primitives
                    .buffer_offset(mat.primitives_subbuffer)
                    .unwrap();
                material_layouts[mat.material_index as usize] =
                    uvec2(offset as u32, mat.primitives.len() as u32);
            }
        }

        self.config.renderer_resources.collect.run(
            gpu,
            assets,
            encoder,
            post_submit,
            resources_bind_group,
            entities_bind_group,
            &self.primitives,
            collect_state,
            self.primitives.total_len() as u32,
            material_layouts,
        );
    }

    fn insert(
        &mut self,
        gpu: &Gpu,
        world: &World,
        id: EntityId,
        primitive_index: usize,
        shader: &Arc<RendererShader>,
        material: &SharedMaterial,
    ) -> Option<(String, String)> {
        let transparent = is_transparent(world, id, material, shader);
        if (!transparent || !self.config.opaque_only)
            && world.get(id, cpu_lod_visible()).unwrap_or(true)
        {
            let config = &self.config;
            let double_sided = world
                .get(id, double_sided())
                .unwrap_or(material.double_sided().unwrap_or(shader.double_sided));
            let scissors = world.get(id, scissors()).ok();
            let shader_id = format!("{}-{}", shader.id, double_sided);
            let material_id = format!("{}-{:?}", material.id(), scissors);
            let node = self
                .tree
                .entry(shader_id.clone())
                .or_insert_with(|| ShaderNode::new(gpu, config, shader.clone(), double_sided));

            let mat = node
                .tree
                .entry(material_id.clone())
                .or_insert_with(|| MaterialNode {
                    material_index: self.material_indices.acquire_index(),
                    primitives_subbuffer: self.primitives.create_buffer(gpu, None),
                    material: material.clone(),
                    primitives: Vec::new(),
                    scissors,
                });
            self.primitives_lookup.insert(
                (id, primitive_index),
                (shader_id.clone(), material_id.clone(), mat.primitives.len()),
            );
            mat.primitives.push((id, primitive_index));
            Some((shader_id, material_id))
        } else {
            None
        }
    }

    fn remove_primitive(
        &mut self,
        id: EntityId,
        primitive_index: usize,
    ) -> Option<(String, String)> {
        if let Some((shader_id, material_id, index)) =
            self.primitives_lookup.remove(&(id, primitive_index))
        {
            let shader = self.tree.get_mut(&shader_id).unwrap();
            let material = shader.tree.get_mut(&material_id).unwrap();
            let is_last = material.primitives.len() == index + 1;
            if !is_last {
                if let Some(last_id) = material.primitives.last() {
                    self.primitives_lookup.get_mut(last_id).unwrap().2 = index;
                }
            }
            material.primitives.swap_remove(index);
            Some((shader_id, material_id))
        } else {
            None
        }
    }
    fn clean_empty(&mut self, gpu: &Gpu) {
        for node in self.tree.values_mut() {
            node.tree.retain(|_, mat| {
                let to_remove = mat.primitives.is_empty();
                if to_remove {
                    self.primitives
                        .remove_buffer(gpu, mat.primitives_subbuffer)
                        .unwrap();
                    self.material_indices.release_index(mat.material_index);
                }
                !to_remove
            });
        }
        self.tree.retain(|_, v| !v.is_empty());
    }
    #[ambient_profiling::function]
    pub fn render<'a>(
        &'a self,
        render_pass: &mut wgpu::RenderPass<'a>,
        collect_state: &'a RendererCollectState,
        bind_groups: &BindGroups<'a>,
        render_target_size: wgpu::Extent3d,
    ) {
        let primitives_bind_group = if let Some(primitives_bind_group) = &self.primitives_bind_group
        {
            primitives_bind_group
        } else {
            return; // Nothing to render
        };

        #[cfg(target_os = "macos")]
        let counts = collect_state.counts_cpu.lock().clone();

        let mut is_bound = false;

        for node in self.tree.values() {
            tracing::debug!(?node, "Drawing node");
            render_pass.set_pipeline(node.pipeline.pipeline());
            // Bind on first invocation
            let bind_groups = [
                bind_groups.globals,
                bind_groups.entities,
                primitives_bind_group,
            ];
            if !is_bound {
                for (i, bind_group) in bind_groups.iter().enumerate() {
                    render_pass.set_bind_group(i as _, bind_group, &[]);
                    is_bound = true
                }
            }

            for mat in node.tree.values() {
                let material = &mat.material;

                render_pass.set_bind_group(bind_groups.len() as _, material.bind_group(), &[]);
                if let Some(scissors) = mat.scissors {
                    render_pass.set_scissor_rect(scissors.x, scissors.y, scissors.z, scissors.w);
                } else {
                    render_pass.set_scissor_rect(
                        0,
                        0,
                        render_target_size.width,
                        render_target_size.height,
                    );
                }

                let offset = self
                    .primitives
                    .buffer_offset(mat.primitives_subbuffer)
                    .unwrap();
                #[cfg(not(target_os = "macos"))]
                {
                    render_pass.multi_draw_indexed_indirect_count(
                        collect_state.commands.buffer(),
                        offset * std::mem::size_of::<DrawIndexedIndirect>() as u64,
                        collect_state.counts.buffer(),
                        mat.material_index as u64 * std::mem::size_of::<u32>() as u64,
                        mat.primitives.len() as u32,
                    );
                }
                #[cfg(target_os = "macos")]
                {
                    if let Some(count) = counts.get(mat.material_index as usize) {
                        for i in 0..*count {
                            render_pass.draw_indexed_indirect(
                                collect_state.commands.buffer(),
                                (offset + i as u64)
                                    * std::mem::size_of::<DrawIndexedIndirect>() as u64,
                            );
                        }
                    }
                }
            }
        }
    }
    pub fn n_entities(&self) -> usize {
        self.tree.values().fold(0, |p, n| p + n.n_entities())
    }
    pub fn n_nodes(&self) -> usize {
        self.tree.values().fold(0, |p, n| p + n.n_nodes())
    }
    pub fn dump(&self, f: &mut dyn std::io::Write) {
        for (key, node) in self.tree.iter() {
            writeln!(f, "    shader {key:?}").unwrap();
            node.dump(f);
        }
    }
}
struct ShaderNode {
    pipeline: GraphicsPipeline,
    tree: HashMap<String, MaterialNode>,
}
impl ShaderNode {
    pub fn new(
        gpu: &Gpu,
        config: &TreeRendererConfig,
        shader: Arc<RendererShader>,
        double_sided: bool,
    ) -> Self {
        let mut pipeline_info = GraphicsPipelineInfo {
            vs_main: &shader.vs_main,
            fs_main: shader.get_fs_main_name(config.fs_main),
            targets: &config.targets,
            cull_mode: config
                .cull_mode
                .and_then(|f| if double_sided { None } else { Some(f) }),
            ..Default::default()
        };
        if config.depth_stencil {
            pipeline_info = pipeline_info
                .with_depth()
                .with_depth_bias(config.depth_bias);
        }

        let pipeline = shader.shader.to_pipeline(gpu, pipeline_info);

        Self {
            pipeline,
            tree: HashMap::new(),
        }
    }
    fn is_empty(&self) -> bool {
        self.tree.is_empty()
    }
    pub fn n_entities(&self) -> usize {
        self.tree.values().fold(0, |p, n| p + n.primitives.len())
    }
    pub fn n_nodes(&self) -> usize {
        self.tree.len() + 1
    }
    pub fn dump(&self, f: &mut dyn std::io::Write) {
        for (_key, node) in self.tree.iter() {
            writeln!(
                f,
                "      material {:?}: {} entities",
                node.material.name(),
                node.primitives.len()
            )
            .unwrap();
        }
    }
}

impl std::fmt::Debug for MaterialNode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ShaderNode")
            .field("material", &self.material.name())
            .field("primitive_count", &self.primitives.len())
            .finish()
    }
}

impl std::fmt::Debug for ShaderNode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ShaderNode")
            .field("pipeline", &self.pipeline.name())
            .field("tree", &self.tree)
            .finish()
    }
}

struct MaterialNode {
    material_index: u32,
    primitives_subbuffer: SubBufferId,
    material: SharedMaterial,
    primitives: Vec<(EntityId, PrimitiveIndex)>,
    scissors: Option<UVec4>,
}

struct MaterialIndices {
    free_indices: Vec<u32>,
    counter: u32,
}
impl MaterialIndices {
    fn new() -> Self {
        Self {
            free_indices: Vec::new(),
            counter: 0,
        }
    }
    fn acquire_index(&mut self) -> u32 {
        if let Some(index) = self.free_indices.pop() {
            index
        } else {
            self.counter += 1;
            self.counter - 1
        }
    }
    fn release_index(&mut self, index: u32) {
        self.free_indices.push(index);
    }
}
