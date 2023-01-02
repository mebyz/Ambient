use std::{collections::HashMap, io::Cursor, path::PathBuf, sync::Arc};

use anyhow::Context;
use elements_animation::{animation_bind_id_from_name, AnimationClip};
use elements_core::{
    bounding::local_bounding_aabb, hierarchy::{children}, name, transform::{local_to_parent, local_to_world, mesh_to_local, TransformSystem}
};
use elements_ecs::{query, query_mut, EntityData, EntityId, FrameEvent, System, World};
use elements_model::{
    animation_bind_id, model_def, model_skin_ix, model_skins, pbr_renderer_primitives_from_url, Model, ModelDef, PbrRenderPrimitiveFromUrl
};
use elements_physics::{
    collider::{character_controller_height, character_controller_radius, collider, ColliderDef, ColliderFromUrls}, mesh::PhysxGeometryFromUrl, physx::PhysicsKey
};
use elements_renderer::{
    double_sided, lod::{gpu_lod, lod_cutoffs}, materials::pbr_material::PbrMaterialFromUrl
};
use elements_std::{
    asset_cache::{AssetCache, SyncAssetKeyExt}, asset_url::AssetUrl, download_asset::{AssetsCacheDir, ContentLoc}, mesh::Mesh, shapes::AABB
};
use futures::{FutureExt};
use glam::{Mat4, Vec3};

use image::{ImageOutputFormat, RgbaImage};
use itertools::Itertools;
use ordered_float::Float;
use physxx::{PxConvexFlag, PxConvexMeshDesc, PxDefaultMemoryOutputStream, PxMeshFlag, PxTriangleMeshDesc};

use crate::{MaterialFilter, TextureResolver};

#[derive(Debug, Clone)]
pub struct AssetLoc {
    pub id: String,
    pub url: String,
    pub path: String,
}

pub struct AssetMapLoc {
    base_url: String,
    store: String,
    extension: String,
}
impl AssetMapLoc {
    pub fn path(&self, id: impl Into<String>) -> String {
        format!("{}/{}.{}", self.store, id.into(), self.extension)
    }
    pub fn url(&self, id: impl Into<String>) -> String {
        format!("{}{}", self.base_url, self.path(id))
    }
    pub fn id_from_path<'a>(&self, path: &'a str) -> Option<&'a str> {
        if let Some((store, file)) = path.split_once('/') {
            if store != self.store {
                return None;
            }
            if let Some((id, extension)) = file.rsplit_once('.') {
                if extension != self.extension {
                    return None;
                }
                return Some(id);
            }
        }
        None
    }
    pub fn id_from_url<'a>(&self, url: &'a str) -> Option<&'a str> {
        if let Some(path) = url.strip_prefix(&self.base_url) {
            self.id_from_path(path)
        } else {
            None
        }
    }
}

pub struct AssetMap<T> {
    pub loc: AssetMapLoc,
    pub content: HashMap<String, T>,
    pub serialize: fn(&T) -> Vec<u8>,
}
impl<T: Send + 'static> AssetMap<T> {
    fn new(base_url: impl Into<String>, store: &str, extension: &str, serialize: fn(&T) -> Vec<u8>) -> Self {
        Self {
            loc: AssetMapLoc { base_url: base_url.into(), store: store.to_string(), extension: extension.into() },
            content: Default::default(),
            serialize,
        }
    }

    pub fn get_by_url(&self, url: &str) -> Option<&T> {
        self.content.get(self.loc.id_from_url(url)?)
    }
    pub fn get_by_path(&self, path: &str) -> Option<&T> {
        self.content.get(self.loc.id_from_path(path)?)
    }
    pub fn insert(&mut self, id: impl Into<String>, content: T) -> AssetLoc {
        let id: String = id.into();
        self.content.insert(id.clone(), content);
        AssetLoc { url: self.loc.url(&id), path: self.loc.path(&id), id }
    }
    pub fn to_items(&self) -> Vec<AssetItem> {
        self.content
            .iter()
            .map(|(id, content)| AssetItem { url: self.loc.url(id), path: self.loc.path(id), data: Arc::new((self.serialize)(content)) })
            .collect()
    }
    pub fn iter_urls(&self) -> impl Iterator<Item = String> + '_ {
        self.content.keys().map(|id| self.loc.url(id))
    }
}

pub struct ModelCrate {
    pub base_url: String,

    pub models: AssetMap<Model>,
    pub objects: AssetMap<World>,
    pub meshes: AssetMap<Mesh>,
    pub animations: AssetMap<AnimationClip>,
    pub images: AssetMap<image::RgbaImage>,
    pub materials: AssetMap<PbrMaterialFromUrl>,
    pub px_triangle_meshes: AssetMap<Vec<u8>>,
    pub px_convex_meshes: AssetMap<Vec<u8>>,
    pub colliders: AssetMap<ColliderFromUrls>,
}
impl ModelCrate {
    pub fn new(base_url: impl Into<String>) -> Self {
        let base_url: String = format!("{}/", base_url.into());
        Self {
            base_url: base_url.clone(),
            models: AssetMap::new(&base_url, "models", "json", |v| serde_json::to_vec(v).unwrap()),
            objects: AssetMap::new(&base_url, "objects", "json", |v| serde_json::to_vec(v).unwrap()),
            meshes: AssetMap::new(&base_url, "meshes", "mesh", |v| bincode::serialize(v).unwrap()),
            animations: AssetMap::new(&base_url, "animations", "anim", |v| bincode::serialize(v).unwrap()),
            images: AssetMap::new(&base_url, "images", "png", |v| {
                let mut data = Cursor::new(Vec::new());
                v.write_to(&mut data, ImageOutputFormat::Png).unwrap();
                data.into_inner()
            }),
            materials: AssetMap::new(&base_url, "materials", "json", |v| serde_json::to_vec(v).unwrap()),
            px_triangle_meshes: AssetMap::new(&base_url, "px_triangle_meshes", "pxtm", |v| v.clone()),
            px_convex_meshes: AssetMap::new(&base_url, "px_convex_meshes", "pxcm", |v| v.clone()),
            colliders: AssetMap::new(&base_url, "colliders", "json", |v| serde_json::to_vec(v).unwrap()),
        }
    }
    pub async fn local_import(assets: &AssetCache, url: &str, normalize: bool, force_assimp: bool) -> anyhow::Result<Self> {
        let source_url = ContentLoc::parse(url)?;
        let cache_path = AssetsCacheDir.get(assets).join("pipelines").join(&source_url.cache_path_string());
        let mut model = Self::new(cache_path.to_str().unwrap());
        model
            .import(
                assets,
                url,
                normalize,
                force_assimp,
                Arc::new(|path| {
                    async move {
                        let path: PathBuf = path.into();
                        let filename = path.file_name().unwrap().to_str().unwrap().to_string();
                        println!("XXX {:?}", filename);
                        None
                    }
                    .boxed()
                }),
            )
            .await?;
        model.update_node_primitive_aabbs_from_cpu_meshes();
        model.model_mut().update_model_aabb();
        Ok(model)
    }
    pub const IN_MEMORY: &str = "in-memory://tmp/";
    pub async fn write_to_fs(&self) {
        let base_url: PathBuf = self.base_url.clone().into();
        for item in self.to_items() {
            let path = base_url.join(item.path);
            std::fs::create_dir_all(path.parent().unwrap()).context(format!("Failed to create dir: {:?}", path.parent().unwrap())).unwrap();
            tokio::fs::write(&path, &*item.data).await.context(format!("Failed to write file: {:?}", path)).unwrap();
        }
    }
    pub fn to_items(&self) -> Vec<AssetItem> {
        [
            self.models.to_items().into_iter(),
            self.objects.to_items().into_iter(),
            self.meshes.to_items().into_iter(),
            self.animations.to_items().into_iter(),
            self.images.to_items().into_iter(),
            self.materials.to_items().into_iter(),
            self.px_triangle_meshes.to_items().into_iter(),
            self.px_convex_meshes.to_items().into_iter(),
            self.colliders.to_items().into_iter(),
        ]
        .into_iter()
        .flatten()
        .collect_vec()
    }
    pub const MAIN: &str = "main";
    pub fn model(&self) -> &Model {
        self.models.content.get(Self::MAIN).unwrap()
    }
    pub fn model_mut(&mut self) -> &mut Model {
        self.models.content.get_mut(Self::MAIN).unwrap()
    }
    pub fn model_world(&self) -> &World {
        self.models.content.get(Self::MAIN).map(|x| &x.0).unwrap()
    }
    pub fn model_world_mut(&mut self) -> &mut World {
        self.models.content.get_mut(Self::MAIN).map(|x| &mut x.0).unwrap()
    }
    pub fn object_world(&self) -> &World {
        self.objects.content.get(Self::MAIN).unwrap()
    }
    pub fn object_world_mut(&mut self) -> &mut World {
        self.objects.content.get_mut(Self::MAIN).unwrap()
    }

    pub async fn produce_local_model_url(&self, assets: &AssetCache) -> anyhow::Result<String> {
        let url = self.models.iter_urls().next().unwrap();
        self.write_to_fs().await;
        Ok(url)
    }
    pub async fn produce_local_model(&self, assets: &AssetCache) -> anyhow::Result<Model> {
        let url = self.produce_local_model_url(assets).await?;
        let mut model = Model::from_file(&url).await?;
        model.load(assets).await?;
        Ok(model)
    }

    pub async fn import(
        &mut self,
        assets: &AssetCache,
        url: &str,
        normalize: bool,
        force_assimp: bool,
        resolve_texture: TextureResolver,
    ) -> anyhow::Result<()> {
        let is_fbx = url.to_lowercase().contains(".fbx");
        let is_glb = url.to_lowercase().contains(".glb");
        if force_assimp {
            crate::assimp::import_url(assets, url, self, resolve_texture).await?;
        } else if is_fbx {
            if let Err(err) = crate::fbx::import_url(assets, url, self).await {
                match err.downcast::<fbxcel::tree::any::Error>() {
                    Ok(err) => {
                        if let fbxcel::tree::any::Error::ParserCreation(fbxcel::pull_parser::any::Error::Header(
                            fbxcel::low::HeaderError::MagicNotDetected,
                        )) = &err
                        {
                            crate::assimp::import_url(assets, url, self, resolve_texture).await?;
                        } else {
                            return Err(err.into());
                        }
                    }
                    Err(err) => return Err(err.into()),
                }
            }
        } else if is_glb {
            crate::gltf::import_url(assets, url, self).await?;
        } else {
            crate::assimp::import_url(assets, url, self, resolve_texture).await?;
        }
        if normalize {
            self.model_mut().rotate_yup_to_zup();
            if is_fbx {
                self.model_mut().transform(Mat4::from_scale(Vec3::ONE / 100.));
            }
        }
        Ok(())
    }
    pub fn merge_mesh_lods(&mut self, cutoffs: Option<Vec<f32>>, lods: Vec<ModelNodeRef>) {
        let default_min_screen_size = 0.04; // i.e. 4%
        let lod_step = (1. / default_min_screen_size).powf(1. / (lods.len() - 1) as f32);
        let mut cutoffs = cutoffs.unwrap_or_else(|| (0..lods.len()).map(|i| 1. / lod_step.powi(i as i32)).collect_vec());
        cutoffs.resize(20, 0.);
        let cutoffs: [f32; 20] = cutoffs.try_into().unwrap();

        let lod_0_node = lods[0].get_node_id();
        let lod_0_world = lods[0].world();

        let mut world = World::new("model mesh lods");
        world.add_resource(name(), format!("{}_merged_lods", lod_0_world.resource_opt(name()).map(|x| x as &str).unwrap_or("unknown")));
        if let Some(aabb) = lod_0_world.resource_opt(local_bounding_aabb()) {
            world.add_resource(local_bounding_aabb(), *aabb);
        }
        if let Some(ltp) = lod_0_world.resource_opt(local_to_parent()) {
            world.add_resource(local_to_parent(), *ltp);
        }

        let mut root = EntityData::new()
            .set(name(), "root".to_string())
            .set(lod_cutoffs(), cutoffs)
            .set_default(gpu_lod())
            .set(
                mesh_to_local(),
                lod_0_world.get(lod_0_node, local_to_world()).unwrap_or_default()
                    * lod_0_world.get(lod_0_node, mesh_to_local()).unwrap_or(Mat4::IDENTITY),
            )
            .set(double_sided(), lod_0_world.get(lod_0_node, double_sided()).unwrap_or_default())
            .set(local_bounding_aabb(), lod_0_world.get(lod_0_node, local_bounding_aabb()).unwrap_or_default())
            .set_default(local_to_world())
            .set_default(pbr_renderer_primitives_from_url());

        for (i, lod) in lods.iter().enumerate() {
            let lod_world = lod.world();
            let lod_id = lod.get_node_id();
            for primitive in lod_world.get_ref(lod_id, pbr_renderer_primitives_from_url()).cloned().unwrap_or_default() {
                let mesh_id = format!("{}_{}", i, lod.model.meshes.loc.id_from_url(&primitive.mesh).unwrap());
                let mesh_url = self.meshes.insert(mesh_id.clone(), lod.model.meshes.get_by_url(&primitive.mesh).unwrap().clone()).url;
                let material = primitive.material.as_ref().and_then(|mat_url| {
                    let mat_id = lod.model.materials.loc.id_from_url(mat_url)?;
                    let lod_mat = lod.model.materials.content.get(mat_id)?;
                    Some(self.materials.insert(i.to_string(), lod_mat.clone()).url)
                });
                root.get_mut(pbr_renderer_primitives_from_url()).unwrap().push(PbrRenderPrimitiveFromUrl {
                    mesh: mesh_url,
                    material,
                    lod: i,
                });
            }
        }
        let root = root.spawn(&mut world);
        world.add_resource(children(), vec![root]);
        self.models.insert(ModelCrate::MAIN, Model(world));
    }
    pub fn merge_unity_style_mesh_lods(&mut self, source: &ModelCrate, cutoffs: Option<Vec<f32>>) {
        let mut lods = source.model_world().resource(children()).clone();
        lods.sort_by_key(|id| {
            let name = source.model_world().get_ref(*id, name()).unwrap();
            &name[(name.len() - 2)..]
        });
        self.merge_mesh_lods(cutoffs, lods.into_iter().map(|id| ModelNodeRef { model: source, root: Some(id) }).collect())
    }
    pub fn set_all_material(&mut self, material: PbrMaterialFromUrl) {
        self.materials.content.clear();
        let mat_url = self.materials.insert("main".to_string(), material).url;
        for (_, primitives, _) in query_mut(pbr_renderer_primitives_from_url(), ()).iter(self.model_world_mut(), None) {
            for primitive in primitives.iter_mut() {
                primitive.material = Some(mat_url.clone());
            }
        }
    }
    pub fn update_node_primitive_aabbs_from_cpu_meshes(&mut self) {
        let world = &mut self.models.content.get_mut(ModelCrate::MAIN).unwrap().0;
        let joint_matrices = world.resource_opt(model_skins()).map(|skins| {
            skins
                .iter()
                .map(|skin| {
                    skin.joints
                        .iter()
                        .zip(skin.inverse_bind_matrices.iter())
                        .map(|(joint, inv_bind_mat)| world.get(*joint, local_to_world()).unwrap_or_default() * *inv_bind_mat)
                        .collect_vec()
                })
                .collect::<Vec<_>>()
        });
        for (node, primitives) in query(pbr_renderer_primitives_from_url()).collect_cloned(world, None) {
            let aabbs = primitives
                .iter()
                .filter_map(|p| {
                    if let Some(mesh) = self.meshes.get_by_url(&p.mesh) {
                        if let Ok(skin_id) = world.get(node, model_skin_ix()) {
                            if let Some(joint_matrices) = joint_matrices.as_ref().unwrap().get(skin_id) {
                                let mut mesh = mesh.clone();
                                let joint_matrices = joint_matrices
                                    .iter()
                                    .map(|mat| {
                                        (world.get(node, local_to_world()).unwrap_or_default()
                                            * world.get(node, mesh_to_local()).unwrap_or_default())
                                        .inverse()
                                            * *mat
                                    })
                                    .collect_vec();
                                mesh.apply_skin(&joint_matrices);
                                return mesh.aabb();
                            }
                        } else {
                            return mesh.aabb();
                        }
                    }
                    None
                })
                .collect_vec();
            if let Some(aabb) = AABB::unions(&aabbs) {
                world.add_component(node, local_bounding_aabb(), aabb).unwrap();
            }
        }
    }
    pub fn make_new_root(&mut self, node_id: EntityId) {
        let world = self.model_world_mut();
        *world.resource_mut(children()) = vec![node_id];
        // TODO(fred): Do the below; clear out unused materials etc.
        // let mut model = Self {
        //     name: format!("{}#{}", self.name, node_id),
        //     source_url: self.source_url.clone(),
        //     roots: vec![node_id.to_string()],
        //     ..Default::default()
        // };
        // self.add_nodes_to_model_recursive(&mut model, node_id);
        // model.update_local_to_models();
        // model.update_model_aabb();
        // model
    }
    // fn add_nodes_to_model_recursive(&self, target: &mut Self, id: &str) {
    //     if let Some(node) = self.nodes.get(id) {
    //         target.nodes.insert(id.to_string(), node.clone());
    //         for primitive in &node.primitives {
    //             if !target.cpu_meshes.contains_key(&primitive.mesh) {
    //                 if let Some(mesh) = self.cpu_meshes.get(&primitive.mesh) {
    //                     target.cpu_meshes.insert(primitive.mesh.clone(), mesh.clone());
    //                 }
    //             }
    //             if !target.gpu_meshes.contains_key(&primitive.mesh) {
    //                 if let Some(mesh) = self.gpu_meshes.get(&primitive.mesh) {
    //                     target.gpu_meshes.insert(primitive.mesh.clone(), mesh.clone());
    //                 }
    //             }
    //             if let Some(material_id) = &primitive.material {
    //                 if !target.gpu_materials.contains_key(material_id) {
    //                     if let Some(material) = self.gpu_materials.get(material_id) {
    //                         target.gpu_materials.insert(material_id.clone(), material.clone());
    //                     }
    //                 }
    //             }
    //         }
    //         if let Some(skin_id) = &node.skin {
    //             if !target.skins.contains_key(skin_id) {
    //                 if let Some(skin) = self.skins.get(skin_id) {
    //                     target.skins.insert(skin_id.clone(), skin.clone());
    //                 }
    //             }
    //         }
    //         for c in &node.children {
    //             self.add_nodes_to_model_recursive(target, c);
    //         }
    //     }
    // }
    pub fn override_material(&mut self, filter: &MaterialFilter, material: PbrMaterialFromUrl) {
        if filter.is_all() {
            self.set_all_material(material);
        } else {
            for old_mat in self.materials.content.values_mut() {
                if filter.matches(&*old_mat) {
                    *old_mat = material.clone();
                }
            }
        }
    }
    pub fn cap_texture_sizes(&mut self, max_size: u32) {
        for image in self.images.content.values_mut() {
            cap_texture_size(image, max_size);
        }
    }
    pub fn update_transforms(&mut self) {
        TransformSystem::new().run(self.model_world_mut(), &FrameEvent);
    }
    pub fn create_animation_bind_ids(&mut self) {
        let world = self.model_world_mut();
        for (id, name) in query(name()).collect_cloned(world, None) {
            world.add_component(id, animation_bind_id(), animation_bind_id_from_name(&name)).unwrap();
        }
    }
    pub fn finalize_model(&mut self) {
        self.update_transforms();
        self.update_node_primitive_aabbs_from_cpu_meshes();
        self.model_mut().update_model_aabb();
        self.create_animation_bind_ids();
        self.model_mut().remove_non_storage_matrices();
    }

    pub fn create_object(&mut self) {
        let mut object = World::new("object_asset");
        let o = EntityData::new().set(model_def(), ModelDef(self.models.loc.url(ModelCrate::MAIN))).spawn(&mut object);
        object.add_resource(children(), vec![o]);

        let object_url = self.objects.insert(ModelCrate::MAIN, object).url;
    }
    pub fn create_character_collider(&mut self, radius: Option<f32>, height: Option<f32>) {
        let world = self.object_world_mut();
        let object = world.resource(children())[0];
        world.add_component(object, character_controller_radius(), radius.unwrap_or(0.5)).unwrap();
        world.add_component(object, character_controller_height(), height.unwrap_or(2.0)).unwrap();
    }
    pub fn create_collider_from_model(&mut self, assets: &AssetCache) -> anyhow::Result<()> {
        self.update_transforms();
        let physics = PhysicsKey.get(assets);
        let create_triangle_mesh = |asset_crate: &mut ModelCrate, id: &str| -> bool {
            if asset_crate.px_triangle_meshes.content.contains_key(id) {
                return true;
            }
            let mesh = asset_crate.meshes.content.get(id).unwrap();
            if let Some(desc) = physx_triangle_mesh_desc_from_mesh(mesh, false) {
                let stream = PxDefaultMemoryOutputStream::new();
                let mut res = physxx::PxTriangleMeshCookingResult::Success;
                if !physics.cooking.cook_triangle_mesh(&desc, &stream, &mut res) {
                    log::error!("Failed to cook triangle mesh: {:?}", res);
                    return false;
                }
                asset_crate.px_triangle_meshes.content.insert(id.to_string(), stream.get_data());
                true
            } else {
                false
            }
        };
        let create_convex_mesh = |asset_crate: &mut ModelCrate, id: &str, scale_signum: Vec3| -> Option<String> {
            // Physx doesn't support negative scaling on Convex meshes, so we need to generate a mesh with the right
            // scale signum first, and then scale that with the absolute scale
            let to_sign = |v| if v >= 0. { "p" } else { "n" }.to_string();
            let full_id = format!("{id}_{}{}{}", to_sign(scale_signum.x), to_sign(scale_signum.y), to_sign(scale_signum.z));
            if asset_crate.px_convex_meshes.content.contains_key(&full_id) {
                return Some(asset_crate.px_convex_meshes.loc.url(&full_id));
            }
            let mesh = asset_crate.meshes.content.get(id).unwrap();

            let desc = PxConvexMeshDesc {
                // Apply the correct mirroring according to the base scale
                points: mesh.positions.as_ref().unwrap().iter().map(|&p| p * scale_signum).collect_vec(),
                indices: mesh.indices.clone(),
                vertex_limit: None,
                flags: Some(PxConvexFlag::COMPUTE_CONVEX),
            };
            let stream = PxDefaultMemoryOutputStream::new();
            let mut res = physxx::PxConvexMeshCookingResult::Success;
            if !physics.cooking.cook_convex_mesh(&desc, &stream, &mut res) {
                log::error!("Failed to cook convex mesh: {:?}", res);
                return None;
            }
            Some(asset_crate.px_convex_meshes.insert(full_id, stream.get_data()).url)
        };
        let mut convex = Vec::new();
        let mut triangle = Vec::new();
        let world_transform = self.model().get_transform().unwrap_or_default();
        let entities = {
            let world = self.model_world();
            query(pbr_renderer_primitives_from_url()).collect_cloned(world, None)
        };
        for (id, prims) in entities {
            let ltw = self.model_world().get(id, local_to_world()).unwrap_or_default();
            let max_lod = prims.iter().map(|x| x.lod).max().unwrap();
            let mtl = self.model_world().get(id, mesh_to_local()).unwrap_or_default();
            // Only use the "max" lod for colliders
            for primitive in prims.into_iter().filter(|x| x.lod == max_lod) {
                let transform = world_transform * ltw * mtl;
                let (scale, rot, pos) = transform.to_scale_rotation_translation();
                let mesh_id = self.meshes.loc.id_from_url(&primitive.mesh).unwrap();
                if create_triangle_mesh(self, mesh_id) {
                    if let Some(convex_url) = create_convex_mesh(self, mesh_id, scale.signum()) {
                        convex.push((Mat4::from_scale_rotation_translation(scale.abs(), rot, pos), PhysxGeometryFromUrl(convex_url)));
                        triangle.push((transform, PhysxGeometryFromUrl(self.px_triangle_meshes.loc.url(mesh_id))));
                    }
                }
            }
        }
        let obj_collider = self.colliders.insert(ModelCrate::MAIN.to_string(), ColliderFromUrls { convex, concave: triangle });
        let object = self.object_world_mut();
        object
            .add_component(
                object.resource(children())[0],
                collider(),
                ColliderDef::Asset { collider: AssetUrl::new(obj_collider.url, "collider") },
            )
            .unwrap();
        Ok(())
    }
}
pub struct AssetItem {
    pub url: String,
    pub path: String,
    pub data: Arc<Vec<u8>>,
}

pub fn cap_texture_size(image: &mut RgbaImage, max_size: u32) {
    if image.width() > max_size || image.height() > max_size {
        let (width, height) = if image.width() >= image.height() {
            (max_size, (max_size as f32 * image.height() as f32 / image.width() as f32) as u32)
        } else {
            ((max_size as f32 * image.width() as f32 / image.height() as f32) as u32, max_size)
        };
        *image = image::imageops::resize(&*image as &image::RgbaImage, width, height, image::imageops::FilterType::CatmullRom);
    }
}

pub struct ModelNodeRef<'a> {
    pub model: &'a ModelCrate,
    pub root: Option<EntityId>,
}
impl<'a> ModelNodeRef<'a> {
    fn world(&self) -> &World {
        self.model.model_world()
    }
    fn get_node_id(&self) -> EntityId {
        self.root.unwrap_or(self.world().resource(children())[0])
    }
}

pub fn physx_triangle_mesh_desc_from_mesh(mesh: &Mesh, flip_normals: bool) -> Option<PxTriangleMeshDesc> {
    let mut desc = PxTriangleMeshDesc {
        points: mesh.positions.clone()?,
        indices: mesh.indices.clone()?,
        flags: if flip_normals { Some(PxMeshFlag::FLIPNORMALS) } else { None },
    };
    if desc.points.is_empty() || desc.indices.is_empty() {
        return None;
    }
    // Seems like Physx expect indicies in another order than what we use. https://github.com/PlayDims/Elements/issues/197
    for i in 0..(desc.indices.len() / 3) {
        desc.indices.swap(i * 3 + 1, i * 3 + 2);
    }
    Some(desc)
}