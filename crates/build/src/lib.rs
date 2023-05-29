use std::{
    path::{Path, PathBuf},
    sync::Arc,
};

use ambient_asset_cache::{AssetCache, SyncAssetKeyExt};
use ambient_physics::physx::{Physics, PhysicsKey};
use ambient_project::Manifest as ProjectManifest;
use ambient_std::{asset_url::AbsAssetUrl, path::path_to_unix_string};
use anyhow::Context;
use futures::FutureExt;
use itertools::Itertools;
use pipelines::{FileCollection, ProcessCtx, ProcessCtxKey};
use walkdir::WalkDir;

pub mod pipelines;

#[derive(Clone, Debug, serde::Deserialize, serde::Serialize)]
pub struct Metadata {
    client_component_paths: Vec<String>,
    server_component_paths: Vec<String>,
}

impl Metadata {
    pub fn component_paths(&self, target: &str) -> &[String] {
        match target {
            "client" => &self.client_component_paths,
            "server" => &self.server_component_paths,
            _ => panic!("Unknown target `{}`", target),
        }
    }

    pub fn parse(contents: &str) -> anyhow::Result<Self> {
        toml::from_str(contents).context("failed to parse build metadata")
    }
}

/// This takes the path to an Ambient project and builds it. An Ambient project is expected to
/// have the following structure:
///
/// assets/**  Here assets such as .glb files are stored. Any files found in this directory will be processed
/// src/**  This is where you store Rust source files
/// build  This is the output directory, and is created when building
/// ambient.toml  This is a metadata file to describe the project
pub async fn build(
    physics: Physics,
    _assets: &AssetCache,
    path: PathBuf,
    manifest: &ProjectManifest,
    optimize: bool,
) -> Metadata {
    log::info!(
        "Building project `{}` ({})",
        manifest.project.id,
        manifest
            .project
            .name
            .as_deref()
            .unwrap_or_else(|| manifest.project.id.as_ref())
    );

    ambient_ecs::ComponentRegistry::get_mut()
        .add_external(ambient_project_native::all_defined_components(manifest, false).unwrap());

    let build_path = path.join("build");
    let assets_path = path.join("assets");

    std::fs::create_dir_all(&build_path).unwrap();
    build_assets(physics, &assets_path, &build_path).await;
    build_rust_if_available(&path, manifest, &build_path, optimize)
        .await
        .unwrap();
    store_manifest(manifest, &build_path).await.unwrap();
    store_metadata(&build_path).await.unwrap()
}

async fn build_assets(physics: Physics, assets_path: &Path, build_path: &Path) {
    let files = WalkDir::new(assets_path)
        .into_iter()
        .filter_map(|e| e.ok())
        .filter(|e| e.metadata().map(|x| x.is_file()).unwrap_or(false))
        .map(|x| AbsAssetUrl::from_file_path(x.into_path()))
        .collect_vec();
    let assets = AssetCache::new_with_config(tokio::runtime::Handle::current(), None);
    PhysicsKey.insert(&assets, physics);
    let ctx = ProcessCtx {
        assets: assets.clone(),
        files: FileCollection(Arc::new(files)),
        in_root: AbsAssetUrl::from_directory_path(assets_path),
        out_root: AbsAssetUrl::from_directory_path(build_path.join("assets")),
        input_file_filter: None,
        package_name: "".to_string(),
        write_file: Arc::new({
            let build_path = build_path.to_owned();
            move |path, contents| {
                let path = build_path.join("assets").join(path);
                async move {
                    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
                    tokio::fs::write(&path, contents).await.unwrap();
                    AbsAssetUrl::from_file_path(path)
                }
                .boxed()
            }
        }),
        on_status: Arc::new(|msg| {
            log::info!("{}", msg);
            async {}.boxed()
        }),
        on_error: Arc::new(|err| {
            log::error!("{:?}", err);
            async {}.boxed()
        }),
    };
    ProcessCtxKey.insert(&ctx.assets, ctx.clone());
    pipelines::process_pipelines(&ctx).await;
}

async fn build_rust_if_available(
    project_path: &Path,
    manifest: &ProjectManifest,
    build_path: &Path,
    optimize: bool,
) -> anyhow::Result<()> {
    let cargo_toml_path = project_path.join("Cargo.toml");
    if !cargo_toml_path.exists() {
        return Ok(());
    }

    let toml = cargo_toml::Manifest::from_str(&tokio::fs::read_to_string(&cargo_toml_path).await?)?;
    match toml.package {
        Some(package) if package.name == manifest.project.id.as_ref() => {}
        Some(package) => {
            anyhow::bail!(
                "The name of the package in the Cargo.toml ({}) does not match the project's ID ({})",
                package.name,
                manifest.project.id
            );
        }
        None => anyhow::bail!(
            "No [package] present in Cargo.toml for project {}",
            manifest.project.id.as_ref()
        ),
    }

    let rustc = ambient_rustc::Rust::get_system_installation().await?;

    for feature in &manifest.build.rust.feature_multibuild {
        for (path, bytecode) in rustc.build(
            project_path,
            manifest.project.id.as_ref(),
            optimize,
            &[feature],
        )? {
            let component_bytecode = ambient_wasm::shared::build::componentize(&bytecode)?;

            let output_path = build_path.join(feature);
            std::fs::create_dir_all(&output_path)?;

            let filename = path.file_name().context("no filename")?;
            tokio::fs::write(output_path.join(filename), component_bytecode).await?;
        }
    }

    Ok(())
}

fn get_component_paths(target: &str, build_path: &Path) -> Vec<String> {
    std::fs::read_dir(build_path.join(target))
        .ok()
        .map(|rd| {
            rd.filter_map(Result::ok)
                .map(|p| p.path())
                .filter(|p| p.extension().unwrap_or_default() == "wasm")
                .map(|p| path_to_unix_string(p.strip_prefix(build_path).unwrap()))
                .collect()
        })
        .unwrap_or_default()
}

async fn store_manifest(manifest: &ProjectManifest, build_path: &Path) -> anyhow::Result<()> {
    let manifest_path = build_path.join("ambient.toml");
    tokio::fs::write(&manifest_path, toml::to_string(&manifest)?).await?;
    Ok(())
}

async fn store_metadata(build_path: &Path) -> anyhow::Result<Metadata> {
    let metadata = Metadata {
        client_component_paths: get_component_paths("client", build_path),
        server_component_paths: get_component_paths("server", build_path),
    };
    let metadata_path = build_path.join("metadata.toml");
    tokio::fs::write(&metadata_path, toml::to_string(&metadata)?).await?;
    Ok(metadata)
}
