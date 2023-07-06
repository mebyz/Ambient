use ambient_api::{
    components::core::{
        app::main_scene,
        camera::aspect_ratio_from_window,
        ecs::{children, parent},
        physics::{
            character_controller_height, character_controller_radius, physics_controlled,
            sphere_collider,
        },
        player::{player, user_id},
        primitives::{cube},
        rendering::{color, sky, water, fog_density, transparency_group},
        transform::{local_to_parent, rotation, scale, translation},
    },
    concepts::{make_perspective_infinite_reverse_camera, make_transformable, make_sphere,},
    prelude::*,
};

use components::{
    player_head_ref, player_movement_direction, player_pitch, player_yaw, track_audio_url,
};
use std::f32::consts::{PI, TAU};

mod tooling;

fn make_vegetation(vegetation_type: &str) {
    let (seed, num_vegetation) = match vegetation_type {
        "trees" => (123456, 1),
        // "trees2" => (123460, 30),
        // "rocks" => (123457, 60),
        _ => panic!("Invalid vegetation type"),
    };

    for i in 0..num_vegetation {
        let x = tooling::gen_rn(seed + i, 0.0, 10.0) * 2.0;
        let y = tooling::gen_rn(seed + i + 1, 0.0, 10.0) * 2.0;
        let position = vec3(x, y, tooling::get_height(x, y) * 2.0 + 1.0);

        Entity::new()
        .with_merge(make_transformable())
        .with_merge(make_sphere())
        .with(scale(), Vec3::ONE * 0.1)
        .with(color(), vec4(0.0, 0.0, 0.0, 0.0))
        .with(transparency_group(), 0)
        .with(sphere_collider(), 5.0)
        .with(translation(), position)
        .spawn();
    }
}

#[main]
pub fn main() {
    let bgm_url = asset::url("assets/forest.ogg").unwrap();

    entity::add_component(entity::synchronized_resources(), track_audio_url(), bgm_url);

    Entity::new()
        .with_merge(make_transformable())
        .with_default(water())
        .with(scale(), Vec3::ONE * 100.)
        .spawn();

    Entity::new()
        .with_merge(make_transformable())
        .with_default(sky())
        .with(fog_density(), 10.0)
        .spawn();


        make_vegetation("trees");
        //make_vegetation("trees2");
        //make_vegetation("rocks");


    spawn_query((player(), user_id())).bind(move |players| {
        for (id, (_, uid)) in players {
            let head = Entity::new()
                .with_merge(make_perspective_infinite_reverse_camera())
                .with(aspect_ratio_from_window(), EntityId::resources())
                .with_default(main_scene())
                .with(user_id(), uid)
                .with(translation(), Vec3::Z)
                //.with(translation(), vec3(0.0, 0.0, 4.0))
                //.with(lookat_target(), vec3(5.0, 5.0, 3.0))
                .with(parent(), id)
                .with_default(local_to_parent())
                .with(rotation(), Quat::from_rotation_x(PI / 2.))
                .spawn();

            entity::add_components(
                id,
                Entity::new()
                    .with_merge(make_transformable())
                    .with_default(cube())
                    .with(scale(), Vec3::ONE)
                    .with(character_controller_height(), 2.)
                    .with(character_controller_radius(), 0.5)
                    .with_default(physics_controlled())
                    .with(player_head_ref(), head)
                    .with(children(), vec![head])
                    .with(player_pitch(), 0.0)
                    .with(player_yaw(), 0.0),
            );
        }
    });

    messages::Input::subscribe(move |source, msg| {
        let Some(player_id) = source.client_entity_id() else { return; };

        entity::add_component(player_id, player_movement_direction(), msg.direction);

        let yaw = entity::mutate_component(player_id, player_yaw(), |yaw| {
            *yaw = (*yaw + msg.mouse_delta.x * 0.01) % TAU;
        })
        .unwrap_or_default();
        let pitch = entity::mutate_component(player_id, player_pitch(), |pitch| {
            *pitch = (*pitch + msg.mouse_delta.y * 0.01).clamp(-PI / 3., PI / 3.);
        })
        .unwrap_or_default();

        entity::set_component(player_id, rotation(), Quat::from_rotation_z(yaw));
        if let Some(head_id) = entity::get_component(player_id, player_head_ref()) {
            entity::set_component(head_id, rotation(), Quat::from_rotation_x(PI / 2. + pitch));
        }
    });

    query((player(), player_movement_direction(), rotation())).each_frame(move |players| {
        for (player_id, (_, direction, rot)) in players {
            let speed = 0.02;

            let mut displace = rot * (direction.normalize_or_zero() * speed);

            let mut pos = entity::get_component(player_id, translation()).unwrap_or_default();
            let h = tooling::get_height(pos.x, pos.y) * 2.0;

            let displace_z = h - pos.z;

            if displace != Vec3::ZERO {
                // println!(
                //     "x:{} y:{} z:{} h:{} d:{}",
                //     pos.x, pos.y, pos.z, h, displace_z
                // );
                displace = Vec3::new(displace.x, displace.y, displace_z);
            }

            //entity::set_component(player_id, translation(), pos);
            physics::move_character(player_id, displace, 0.01, delta_time());
        }
    });
}