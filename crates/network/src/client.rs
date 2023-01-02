use std::{
    fmt::{Debug, Display}, net::SocketAddr, sync::Arc, time::Duration
};

use anyhow::Context;
use elements_core::{asset_cache, gpu, runtime};
use elements_ecs::{components, query, EntityData, EntityId, SystemGroup, World, WorldDiff};
use elements_element::{Element, ElementComponent, ElementComponentExt, Hooks};
use elements_renderer::RenderTarget;
use elements_rpc::RpcRegistry;
use elements_std::{fps_counter::FpsSample, log_result, to_byte_unit, CallbackFn, Cb};
use elements_ui::{height, width, Button, Centered, FlowColumn, FlowRow, Image, Text, Throbber, WindowSized};
use futures::{io::BufReader, AsyncBufReadExt, AsyncReadExt, Future, StreamExt};
use glam::UVec2;
use parking_lot::Mutex;
use quinn::{Connection, NewConnection};
use serde::{de::DeserializeOwned, Serialize};

use crate::{
    client_game_state::ClientGameState, create_client_endpoint_random_port, events::event_registry, is_remote_entity, log_network_result, player, protocol::{ClientInfo, ClientProtocol}, rpc_request, server::SharedServerState, user_id, NetworkError
};

components!("network", {
    game_client: Option<GameClient>,
});

pub fn get_player_entity(world: &World, target_user_id: &str) -> Option<EntityId> {
    query((user_id(), player())).iter(world, None).find(|(_, (uid, _))| uid.as_str() == target_user_id).map(|kv| kv.0)
}

#[derive(Clone)]
pub struct GameRpcArgs {
    pub state: SharedServerState,
    pub user_id: String,
}
impl GameRpcArgs {
    pub fn get_player(&self, world: &World) -> Option<EntityId> {
        get_player_entity(world, &self.user_id)
    }
}

#[derive(Debug, Clone)]
/// Manages the client side connection to the server.
pub struct GameClient {
    pub connection: Connection,
    pub rpc_registry: Arc<RpcRegistry<GameRpcArgs>>,
    pub user_id: String,
    pub game_state: Arc<Mutex<ClientGameState>>,
    pub render_target: Arc<RenderTarget>,
}

impl GameClient {
    pub fn new(
        connection: Connection,
        rpc_registry: Arc<RpcRegistry<GameRpcArgs>>,
        game_state: Arc<Mutex<ClientGameState>>,
        render_target: Arc<RenderTarget>,
        user_id: String,
    ) -> Self {
        Self { connection, rpc_registry, user_id, game_state, render_target }
    }

    const SIZE_LIMIT: usize = 100_000_000;

    pub async fn rpc<
        Req: Serialize + DeserializeOwned + Send + 'static,
        Resp: Serialize + DeserializeOwned + Send,
        F: Fn(GameRpcArgs, Req) -> L + Send + Sync + Copy + 'static,
        L: Future<Output = Resp> + Send,
    >(
        &self,
        func: F,
        req: Req,
    ) -> Result<Resp, NetworkError> {
        rpc_request(&self.connection, self.rpc_registry.clone(), func, req, Self::SIZE_LIMIT).await
    }

    pub fn make_standalone_rpc_wrapper<
        Req: Serialize + DeserializeOwned + Send + 'static,
        Resp: Serialize + DeserializeOwned + Send,
        F: Fn(GameRpcArgs, Req) -> L + Send + Sync + Copy + 'static,
        L: Future<Output = Resp> + Send,
    >(
        &self,
        runtime: &tokio::runtime::Handle,
        func: F,
    ) -> Cb<impl Fn(Req)> {
        let runtime = runtime.clone();
        let (connection, rpc_registry) = (self.connection.clone(), self.rpc_registry.clone());
        Cb::new(move |req| {
            let (connection, rpc_registry) = (connection.clone(), rpc_registry.clone());
            runtime.spawn(async move {
                log_network_result!(rpc_request(&connection, rpc_registry, func, req, Self::SIZE_LIMIT).await);
            });
        })
    }

    pub fn with_physics_world<R>(&self, f: impl Fn(&mut World) -> R) -> R {
        f(&mut self.game_state.lock().world)
    }
}

#[derive(Debug)]
pub struct UseOnce<T> {
    val: Mutex<Option<T>>,
}

impl<T> UseOnce<T> {
    pub fn new(val: T) -> Self {
        Self { val: Mutex::new(Some(val)) }
    }

    pub fn take(&self) -> Option<T> {
        self.val.lock().take()
    }
}

pub type InitCallback = Box<dyn FnOnce(&mut World, Arc<RenderTarget>) + Send + Sync>;

#[allow(clippy::type_complexity)]
#[derive(Debug)]
pub struct GameClientView<T: Debug + Send + 'static> {
    pub server_addr: SocketAddr,
    pub user_id: String,
    pub size: UVec2,
    pub resolution: UVec2,
    pub systems_and_resources: Cb<dyn Fn() -> (SystemGroup, EntityData) + Sync + Send>,
    pub init_world: Cb<UseOnce<InitCallback>>,
    pub error_view: Cb<dyn Fn(String) -> Element + Sync + Send>,
    pub on_loaded: Cb<dyn Fn(Arc<Mutex<ClientGameState>>, GameClient) -> anyhow::Result<T> + Sync + Send>,
    pub on_in_entities: Option<Cb<dyn Fn(&WorldDiff) + Sync + Send>>,
    pub on_disconnect: Cb<dyn Fn(Option<T>) + Sync + Send + 'static>,
    pub create_rpc_registry: Cb<dyn Fn() -> RpcRegistry<GameRpcArgs> + Sync + Send>,
    pub ui: Element,
}

impl<T: Debug + Send + 'static> Clone for GameClientView<T> {
    fn clone(&self) -> Self {
        Self {
            server_addr: self.server_addr,
            user_id: self.user_id.clone(),
            size: self.size,
            resolution: self.resolution,
            systems_and_resources: self.systems_and_resources.clone(),
            init_world: self.init_world.clone(),
            error_view: self.error_view.clone(),
            on_loaded: self.on_loaded.clone(),
            on_in_entities: self.on_in_entities.clone(),
            on_disconnect: self.on_disconnect.clone(),
            create_rpc_registry: self.create_rpc_registry.clone(),
            ui: self.ui.clone(),
        }
    }
}

impl<T> ElementComponent for GameClientView<T>
where
    T: Debug + Send + 'static,
{
    fn render(self: Box<Self>, world: &mut World, hooks: &mut Hooks) -> Element {
        let Self {
            server_addr,
            user_id,
            size,
            resolution,
            init_world,
            error_view,
            systems_and_resources,
            create_rpc_registry,
            on_loaded,
            on_in_entities,
            ui,
            on_disconnect,
        } = *self;

        let (_, client_stats_ctx) = hooks.consume_context::<GameClientNetworkStats>().unwrap();
        let (_, server_stats_ctx) = hooks.consume_context::<GameClientServerStats>().unwrap();

        let gpu = world.resource(gpu()).clone();

        let render_target = hooks.use_memo_with(resolution, || Arc::new(RenderTarget::new(gpu.clone(), resolution, None)));

        let (connection_status, set_connection_status) = hooks.use_state("Connecting".to_string());

        let assets = world.resource(asset_cache()).clone();
        let game_state = hooks.use_ref_with(|| {
            let (systems, resources) = systems_and_resources();
            let mut state = ClientGameState::new(world, assets.clone(), user_id.clone(), render_target.clone(), systems, resources);

            (init_world.take().expect("Init called twice"))(&mut state.world, render_target.clone());

            state
        });

        // The game client will be set once a connection establishes
        let (game_client, set_game_client) = hooks.use_state(None as Option<GameClient>);

        {
            let game_state = game_state.clone();
            let render_target = render_target.clone();
            hooks.use_frame(move |_| {
                let mut game_state = game_state.lock();
                game_state.on_frame(&render_target);
            });
        }

        let (error, set_error) = hooks.use_state(None);

        let reg = game_state.lock().world.resource(event_registry()).clone();

        let task = {
            let render_target = render_target.clone();
            let runtime = world.resource(runtime()).clone();

            hooks.use_memo_with((), move || {
                let task = runtime.spawn(async move {
                    // These are the callbacks for everything that can happen
                    let mut on_event = {
                        let game_state = game_state.clone();
                        let reg = reg.clone();
                        move |event_name: String, event_data| {
                            let event_name = event_name.trim();
                            let res = reg.handle_event(&game_state, event_name, event_data);
                            log_result!(res);
                        }
                    };

                    let mut on_init = {
                        let game_state = game_state.clone();
                        move |conn, info: ClientInfo| {
                            let game_client = GameClient::new(
                                conn,
                                Arc::new(create_rpc_registry()),
                                game_state.clone(),
                                render_target.clone(),
                                info.user_id,
                            );

                            game_state.lock().world.add_resource(self::game_client(), Some(game_client.clone()));

                            // Update parent client
                            set_game_client(Some(game_client.clone()));
                            on_loaded(game_state.clone(), game_client).context("Failed to initialize game client view")
                        }
                    };

                    let mut on_diff = |diff| {
                        if let Some(on_in_entities) = &on_in_entities {
                            on_in_entities(&diff);
                        }
                        let mut gs = game_state.lock();
                        diff.apply(&mut gs.world, EntityData::new().set(is_remote_entity(), ()), false);
                    };

                    let mut on_server_stats = |stats| {
                        server_stats_ctx(stats);
                    };

                    let mut on_client_stats = |stats| {
                        client_stats_ctx(stats);
                    };

                    let client_loop = ClientInstance::<T> {
                        set_connection_status,
                        server_addr,
                        user_id,
                        on_init: &mut on_init,
                        on_diff: &mut on_diff,
                        on_server_stats: &mut on_server_stats,
                        on_client_stats: &mut on_client_stats,
                        on_event: &mut on_event,
                        on_disconnect,
                        data: None,
                    };

                    match client_loop.run().await {
                        Err(err) => {
                            if let Some(err) = err.downcast_ref::<NetworkError>() {
                                if let NetworkError::ConnectionClosed = err {
                                    log::info!("Connection closed by peer");
                                } else {
                                    log::error!("Network error: {:?}", err);
                                }
                            } else {
                                log::error!("Game failed: {:?}", err);
                            }
                            set_error(Some(format!("{:?}", err)));
                        }
                        Ok(()) => {
                            log::info!("Client disconnected");
                        }
                    };
                });
                Arc::new(task)
            })
        };

        // When the GameClientView is despawned, stop the task.
        {
            let task = task.clone();
            hooks.use_spawn(move |_| Box::new(move |_| task.abort()));
        }

        if let Some(err) = error {
            return error_view(err);
        }

        if let Some(game_client) = game_client {
            // Provide the context
            hooks.provide_context(|| game_client.clone());
            world.add_resource(self::game_client(), Some(game_client.clone()));

            Element::new().children(vec![
                Image { texture: Some(Arc::new(render_target.screen_buffer.create_view(&Default::default()))) }
                    .el()
                    .set(width(), size.x as _)
                    .set(height(), size.y as _),
                ui,
            ])
        } else {
            WindowSized(vec![Centered(vec![FlowColumn::el([
                FlowRow::el([Text::el(connection_status), Throbber.el()]),
                Button::new("Cancel", move |_| task.abort()).el(),
            ])])
            .el()])
            .el()
        }
    }
}

struct ClientInstance<'a, T> {
    set_connection_status: CallbackFn<String>,
    server_addr: SocketAddr,
    user_id: String,

    /// Called when the client connected and received the world.
    on_init: &'a mut (dyn FnMut(Connection, ClientInfo) -> anyhow::Result<T> + Send + Sync),
    on_diff: &'a mut (dyn FnMut(WorldDiff) + Send + Sync),

    on_server_stats: &'a mut (dyn FnMut(GameClientServerStats) + Send + Sync),
    on_client_stats: &'a mut (dyn FnMut(GameClientNetworkStats) + Send + Sync),
    on_event: &'a mut (dyn FnMut(String, Box<[u8]>) + Send + Sync),
    on_disconnect: Cb<dyn Fn(Option<T>) + Sync + Send + 'static>,
    data: Option<T>,
}

impl<'a, T> Drop for ClientInstance<'a, T> {
    fn drop(&mut self) {
        (self.on_disconnect)(self.data.take())
    }
}

impl<'a, T> ClientInstance<'a, T> {
    #[tracing::instrument(skip(self))]
    async fn run(mut self) -> anyhow::Result<()> {
        tracing::info!("Connecting to server at: {}", self.server_addr);
        (self.set_connection_status)(format!("Connecting to {}", self.server_addr));
        let conn = open_connection(self.server_addr).await?;

        (self.set_connection_status)("Waiting for server to respond".to_string());

        // Set up the protocol.
        let mut protocol = ClientProtocol::new(conn, self.user_id.clone()).await?;

        let stats_interval = 5;
        let mut stats_timer = tokio::time::interval(Duration::from_secs_f32(stats_interval as f32));
        let mut prev_stats = protocol.connection().stats();

        // The first WorldDiff initializes the world, so wait for that until we say things are "ready"
        (self.set_connection_status)("Receiving world".to_string());

        let msg = protocol.diff_stream.next().await?;
        (self.on_diff)(msg);
        self.data = Some((self.on_init)(protocol.connection(), protocol.client_info().clone()).context("Client initialization failed")?);

        // The server
        loop {
            tokio::select! {
                msg = protocol.diff_stream.next() => {
                    profiling::scope!("game_in_entities");
                    let msg: WorldDiff  = msg?;
                    (self.on_diff)(msg);
                }
                _ = stats_timer.tick() => {
                    let stats = protocol.connection().stats();

                    (self.on_client_stats)(GameClientNetworkStats {
                        latency_ms: protocol.connection().rtt().as_millis() as u64,
                        bytes_sent: (stats.udp_tx.bytes - prev_stats.udp_tx.bytes) / stats_interval,
                        bytes_received: (stats.udp_rx.bytes - prev_stats.udp_rx.bytes) / stats_interval,
                    });

                    prev_stats = stats;
                }
                Ok(stats) = protocol.stat_stream.next() => {
                    (self.on_server_stats)(GameClientServerStats(stats));
                }
                Some(Ok(msg)) = protocol.conn.uni_streams.next() => {
                    let mut reader = BufReader::new(msg);

                    let mut event_name = String::new();
                    reader.read_line(&mut event_name).await.context("Event did not contain valid UTF-8")?;

                    let mut event_data = Vec::new();

                    reader.read_to_end(&mut event_data).await?;

                    (self.on_event)(event_name, event_data.into_boxed_slice());
                }
            }
        }
    }
}

/// Set up and manage a connection to the server
#[derive(Debug, Clone, Default)]
pub struct GameClientNetworkStats {
    pub latency_ms: u64,
    pub bytes_sent: u64,
    pub bytes_received: u64,
}

impl Display for GameClientNetworkStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{:?}ms, {}/s out, {}/s in",
            self.latency_ms,
            to_byte_unit(self.bytes_sent as usize),
            to_byte_unit(self.bytes_received as usize)
        )
    }
}

#[derive(Debug, Clone, Default)]
pub struct GameClientServerStats(pub FpsSample);

/// Connnect to the server endpoint.
/// Does not handle a protocol.
#[tracing::instrument]
pub async fn open_connection(server_addr: SocketAddr) -> anyhow::Result<NewConnection> {
    tracing::info!("Connecting to world instance: {:?}", server_addr);

    let endpoint = create_client_endpoint_random_port().context("Failed to create client endpoint")?;

    tracing::info!("Got endpoint");
    let conn = endpoint.connect(server_addr, "localhost")?.await?;

    tracing::info!("Got connection");
    Ok(conn)
}