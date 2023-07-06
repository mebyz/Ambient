pub mod deploy_proto {
    include!("../proto/ambient.run.deploy.rs");
}

use std::{
    collections::HashMap,
    path::{Path, PathBuf},
    str::FromStr,
};

use ambient_project::Manifest;
use md5::Digest;
use tokio_stream::StreamExt;
use tonic::{
    codegen::{CompressionEncoding, InterceptedService},
    metadata::MetadataValue,
    service::Interceptor,
    transport::{Certificate, Channel, ClientTlsConfig, Uri},
    Request,
};
use walkdir::WalkDir;

use deploy_proto::{
    asset_content::ContentDescription, deploy_asset_response::Message,
    deployer_client::DeployerClient, AssetContent, DeployAssetRequest, Deployment,
};

const CHUNK_SIZE: usize = 1024 * 1024 * 3; // 3MB

async fn asset_requests_from_file_path(
    ember_id: impl AsRef<str>,
    asset_path: impl AsRef<str>,
    file_path: impl AsRef<Path>,
) -> anyhow::Result<Vec<DeployAssetRequest>> {
    let content = ambient_sys::fs::read(file_path.as_ref()).await?;
    let total_size = content.len() as u64;

    // handle empty file
    if content.is_empty() {
        return Ok(vec![DeployAssetRequest {
            ember_id: ember_id.as_ref().into(),
            contents: vec![AssetContent {
                path: asset_path.as_ref().into(),
                total_size,
                content_description: Some(ContentDescription::Chunk(content)),
            }],
        }]);
    }

    // using single AssetContent per chunk because gRPC message has to be <4MB
    Ok(content
        .chunks(CHUNK_SIZE)
        .map(|chunk| DeployAssetRequest {
            ember_id: ember_id.as_ref().into(),
            contents: vec![AssetContent {
                path: asset_path.as_ref().into(),
                total_size,
                content_description: Some(ContentDescription::Chunk(chunk.to_vec())),
            }],
        })
        .collect())
}

/// This takes the path to an Ambient ember and deploys it. An Ambient ember is expected to
/// be already built.
pub async fn deploy(
    runtime: &tokio::runtime::Handle,
    api_server: &str,
    auth_token: &str,
    path: impl AsRef<Path>,
    manifest: &Manifest,
    force_upload: bool,
) -> anyhow::Result<String> {
    let ember_id = manifest.ember.id.to_string();
    log::info!(
        "Deploying ember `{}` ({})",
        ember_id,
        manifest
            .ember
            .name
            .as_deref()
            .unwrap_or_else(|| manifest.ember.id.as_ref())
    );
    let base_path = path.as_ref().to_owned();

    // create a client and open channel to the server
    let mut client = create_client(api_server, auth_token).await?;

    // collect all files to deploy (everything in the build directory)
    let asset_path_to_file_path: Option<HashMap<String, PathBuf>> =
        WalkDir::new(path.as_ref().join("build"))
            .into_iter()
            .filter_map(|e| e.ok())
            .filter(|e| e.metadata().map(|x| x.is_file()).unwrap_or(false))
            .map(|x| {
                let file_path = x.into_path();
                let path = file_path
                    .strip_prefix(&base_path)
                    .expect("file path should be in base path")
                    .to_str();
                if let Some(path) = path {
                    if path.chars().any(|c| c == '\n' || c == '\r') {
                        log::error!(
                            "Path contains Line Feed or Carriage Return character: {:?}",
                            file_path
                        );
                        None
                    } else {
                        Some((path.into(), file_path))
                    }
                } else {
                    log::error!("Non-UTF-8 path: {:?}", file_path);
                    None
                }
            })
            .collect();
    let Some(asset_path_to_file_path) = asset_path_to_file_path else {
        anyhow::bail!("Can only deploy files with UTF-8 paths that don't have newline characters");
    };
    log::debug!("Found {} files to deploy", asset_path_to_file_path.len());

    // create a separate task for reading files
    let (file_request_tx, file_request_rx) = flume::unbounded::<FileRequest>();
    let (deploy_asset_request_tx, deploy_asset_request_rx) =
        flume::bounded::<DeployAssetRequest>(16);
    let handle = runtime.spawn(process_file_requests(
        file_request_rx,
        deploy_asset_request_tx,
        ember_id.clone(),
        asset_path_to_file_path.clone(),
    ));

    // notify FileSender to send all files to the server (either by MD5 or by contents)
    for (asset_path, _) in asset_path_to_file_path.iter() {
        let asset_path = asset_path.clone();
        // note: this send shouldn't block as we have an unbounded channel
        file_request_tx.send(if force_upload {
            FileRequest::SendContents(asset_path)
        } else {
            FileRequest::SendMD5(asset_path)
        })?;
    }

    // process responses from the server
    let response = client
        .deploy_assets(deploy_asset_request_rx.into_stream())
        .await?;
    let mut response_stream = response.into_inner();
    let mut deployment = None;
    while let Some(resp) = response_stream.next().await {
        match resp {
            Ok(resp) => {
                log::trace!("Deployed asset response: {:?}", resp);
                match resp.message {
                    Some(Message::Finished(Deployment { id })) => {
                        if deployment.is_some() {
                            log::warn!("Received multiple deployment finished messages");
                        }
                        deployment = Some(id);
                    }
                    Some(Message::Error(err)) => {
                        // error from the server -> just log it and abort as we can't continue
                        log::error!("Received error message: {:?}", err);
                        handle.abort();
                    }
                    Some(Message::AcceptedPath(path)) => {
                        // uploaded file has been accepted (either after MD5 or contents)
                        file_request_tx
                            .send_async(FileRequest::Accepted(path))
                            .await?;
                    }
                    Some(Message::MissingPath(path)) => {
                        // we've sent MD5 for a file but the server doesn't have it
                        file_request_tx
                            .send_async(FileRequest::SendContents(path))
                            .await?;
                    }
                    None => {
                        log::warn!("Received empty message");
                    }
                }
            }
            Err(err) => {
                log::error!("Failed to deploy asset: {:?}", err);
            }
        }
    }

    // wait for the file reading task to finish to handle any errors
    handle.await??;

    // this should have arrived in Finished message from the server
    deployment.ok_or_else(|| anyhow::anyhow!("No deployment id returned from deploy"))
}

/// Created a client for the deploy API server.
async fn create_client(
    api_server: &str,
    auth_token: &str,
) -> anyhow::Result<DeployerClient<InterceptedService<Channel, impl Interceptor>>> {
    // set up TLS config if needed
    let tls = if api_server.starts_with("https://") {
        let domain_name = Uri::from_str(api_server)?
            .host()
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "Could not parse domain name from API server URI: {}",
                    api_server
                )
            })?
            .to_string();

        let mut tls = ClientTlsConfig::new().domain_name(domain_name);

        // set test CA cert if provided
        if let Ok(test_ca_cert) = std::env::var("AMBIENT_DEPLOY_TEST_CA_CERT") {
            let pem = std::fs::read_to_string(test_ca_cert)?;
            let ca = Certificate::from_pem(pem);
            tls = tls.ca_certificate(ca);
        }
        Some(tls)
    } else {
        None
    };

    // set up the endpoint and connect
    let channel = {
        let mut endpoint = Channel::from_shared(api_server.to_owned())?;
        if let Some(tls) = tls {
            endpoint = endpoint.tls_config(tls)?
        }
        endpoint.connect().await?
    };

    // set up client with auth token and compression
    let token: MetadataValue<_> = format!("Bearer {}", auth_token).parse()?;
    let client = DeployerClient::new(InterceptedService::new(channel, {
        move |mut req: Request<()>| {
            req.metadata_mut().insert("authorization", token.clone());
            Ok(req)
        }
    }))
    .send_compressed(CompressionEncoding::Gzip)
    .accept_compressed(CompressionEncoding::Gzip);
    Ok(client)
}

enum FileRequest {
    SendMD5(String),
    SendContents(String),
    Accepted(String),
}

impl FileRequest {
    fn path(&self) -> &str {
        match self {
            FileRequest::SendMD5(path) => path,
            FileRequest::SendContents(path) => path,
            FileRequest::Accepted(path) => path,
        }
    }
}

async fn process_file_requests(
    rx: flume::Receiver<FileRequest>,
    tx: flume::Sender<DeployAssetRequest>,
    ember_id: String,
    mut asset_path_to_file_path: HashMap<String, PathBuf>,
) -> anyhow::Result<()> {
    while let Ok(request) = rx.recv_async().await {
        // get file path
        let path = request.path();
        let Some(file_path) = asset_path_to_file_path.get(path) else {
            // we do a check here to prevent server from asking for files that it shouldn't ask for
            log::warn!("Unknown asset path: {:?}", path);
            continue;
        };

        match request {
            FileRequest::SendMD5(path) => {
                // send MD5 hash
                let content = ambient_sys::fs::read(file_path).await?;
                let md5_digest = md5::Md5::default()
                    .chain_update(&content)
                    .finalize()
                    .to_vec();
                log::debug!("Sending MD5 for {:?} = {}", path, hex(&md5_digest));
                tx.send_async(DeployAssetRequest {
                    ember_id: ember_id.clone(),
                    contents: vec![AssetContent {
                        path,
                        total_size: content.len() as u64,
                        content_description: Some(ContentDescription::Md5(md5_digest)),
                    }],
                })
                .await?;
            }
            FileRequest::SendContents(path) => {
                // send file contents
                log::debug!("Sending contents for {:?}", path);
                let requests = asset_requests_from_file_path(&ember_id, &path, file_path).await?;
                let count = requests.len();
                for (idx, request) in requests.into_iter().enumerate() {
                    let Some(content) = request.contents.first() else { unreachable!() };
                    let Some(ContentDescription::Chunk(chunk)) = content.content_description.as_ref() else { unreachable!() };
                    log::debug!(
                        "Deploying asset chunk {}/{} {} {}B/{}B",
                        idx + 1,
                        count,
                        path,
                        chunk.len(),
                        content.total_size,
                    );
                    tx.send_async(request).await?;
                }
            }
            FileRequest::Accepted(path) => {
                // clear deployed file
                let removed = asset_path_to_file_path.remove(&path);
                if removed.is_none() {
                    log::warn!(
                        "Received accepted path for unknown (or previously accepted) asset: {:?}",
                        path
                    );
                }
                if asset_path_to_file_path.is_empty() {
                    log::debug!("All assets deployed");
                    break;
                }
            }
        }
    }
    // note: leaving this function drops the tx which will cause the deployer to finish
    anyhow::Ok(())
}

fn hex(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        s.push_str(&format!("{:02x}", byte));
    }
    s
}
