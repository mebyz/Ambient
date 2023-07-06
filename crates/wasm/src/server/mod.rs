use crate::shared::{self, message::RuntimeMessageExt};
use ambient_ecs::{query, EntityId, FnSystem, SystemGroup, World};
use ambient_network::server::{ForkingEvent, ShutdownEvent};
use std::sync::Arc;

mod implementation;
mod network;

pub fn initialize(
    world: &mut World,
    messenger: Arc<dyn Fn(&World, EntityId, shared::MessageType, &str) + Send + Sync>,
) -> anyhow::Result<()> {
    shared::initialize(world, messenger, |id| Bindings {
        base: Default::default(),
        world_ref: Default::default(),
        id,
    })?;

    network::initialize(world);

    Ok(())
}

pub fn systems() -> SystemGroup {
    SystemGroup::new(
        "core/wasm/server",
        vec![
            Box::new(FnSystem::new(move |world, _| {
                ambient_profiling::scope!("WASM module collision event");
                // trigger collision event
                let collisions = match world.resource_opt(ambient_physics::collisions()) {
                    Some(collisions) => collisions.lock().clone(),
                    None => return,
                };
                for (a, b) in collisions.into_iter() {
                    ambient_ecs::generated::messages::Collision::new(vec![a, b])
                        .run(world, None)
                        .unwrap();
                }
            })),
            Box::new(FnSystem::new(move |world, _| {
                ambient_profiling::scope!("WASM module collider loads");
                // trigger collider loads
                let collider_loads = match world.resource_opt(ambient_physics::collider_loads()) {
                    Some(collider_loads) => collider_loads.clone(),
                    None => return,
                };

                if collider_loads.is_empty() {
                    return;
                }

                ambient_ecs::generated::messages::ColliderLoads::new(collider_loads)
                    .run(world, None)
                    .unwrap();
            })),
            Box::new(shared::systems()),
        ],
    )
}

pub fn on_forking_systems() -> SystemGroup<ForkingEvent> {
    SystemGroup::new(
        "core/wasm/server/on_forking_systems",
        vec![Box::new(FnSystem::new(move |world, _| {
            // Reset the states of all the modules when we fork.
            shared::reload_all(world);
        }))],
    )
}

pub fn on_shutdown_systems() -> SystemGroup<ShutdownEvent> {
    SystemGroup::new(
        "core/wasm/server/on_shutdown_systems",
        vec![Box::new(FnSystem::new(move |world, _| {
            let modules = query(()).incl(shared::module()).collect_ids(world, None);
            for module_id in modules {
                shared::unload(world, module_id, "shutting down");
            }
        }))],
    )
}

#[derive(Clone)]
struct Bindings {
    base: shared::bindings::BindingsBase,
    world_ref: shared::bindings::WorldRef,
    id: EntityId,
}
impl Bindings {
    pub fn world(&self) -> &World {
        unsafe { self.world_ref.world() }
    }
    pub fn world_mut(&mut self) -> &mut World {
        unsafe { self.world_ref.world_mut() }
    }
}

impl shared::bindings::BindingsBound for Bindings {
    fn base(&self) -> &shared::bindings::BindingsBase {
        &self.base
    }

    fn base_mut(&mut self) -> &mut shared::bindings::BindingsBase {
        &mut self.base
    }
    fn set_world(&mut self, world: &mut World) {
        unsafe {
            self.world_ref.set_world(world);
        }
    }
    fn clear_world(&mut self) {
        unsafe {
            self.world_ref.clear_world();
        }
    }
}
