use ambient_core::project_name;
use ambient_ecs::{ComponentRegistry, ExternalComponentDesc};
use ambient_std::asset_url::AbsAssetUrl;

pub mod client;
pub mod server;

#[derive(Debug, serde::Serialize, serde::Deserialize)]
/// Request sent by the client to the server
pub enum ClientRequest {
    /// Connect to the server with the specified user id
    Connect(String),
    /// Client wants to disconnect
    Disconnect,
}

#[derive(Debug, serde::Serialize, serde::Deserialize)]
/// Frame used by the server to send information to the client
pub enum ServerPush {
    ServerInfo(ServerInfo),
    /// Graceful disconnect
    Disconnect,
}

pub(crate) const VERSION: &str = env!("CARGO_PKG_VERSION");

pub fn get_version_with_revision() -> String {
    format!(
        "{}-{}",
        VERSION,
        ambient_std::parse_git_revision(git_version::git_version!())
            .expect("Failed to find git revision. Please open an issue with `git describe`")
    )
}

/// Miscellaneous information about the server that needs to be sent to the client during the handshake.
#[derive(serde::Serialize, serde::Deserialize, Debug, Clone)]
pub struct ServerInfo {
    /// The name of the project. Used by the client to figure out what to title its window. Defaults to "Ambient".
    pub project_name: String,

    // Base url of the content server.
    pub content_base_url: AbsAssetUrl,

    /// The version of the server. Used by the client to determine whether or not to keep connecting.
    /// Defaults to the version of the crate.
    /// TODO: use semver
    pub version: String,
    pub external_components: Vec<ExternalComponentDesc>,
}

impl ServerInfo {
    pub fn new(state: &mut crate::server::ServerState, content_base_url: AbsAssetUrl) -> Self {
        let instance = state
            .instances
            .get(crate::server::MAIN_INSTANCE_ID)
            .unwrap();
        let world = &instance.world;
        let external_components = ComponentRegistry::get()
            .all_external()
            .map(|x| x.0)
            .collect();

        Self {
            project_name: world.resource(project_name()).clone(),
            content_base_url,
            version: get_version_with_revision(),
            external_components,
        }
    }
}
