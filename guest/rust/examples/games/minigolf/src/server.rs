use crate::{
    components::{
        ball, next_player_hue, origin, player_ball, player_camera_state, player_color,
        player_indicator, player_indicator_arrow, player_restore_point, player_stroke_count,
        player_text, player_text_container,
    },
    concepts::{make_player_camera_state, make_player_state},
};
use ambient_api::{
    components::core::{
        app::main_scene,
        camera::{active_camera, aspect_ratio_from_window},
        ecs::children,
        model::model_from_url,
        physics::{
            angular_velocity, collider_from_url, dynamic, kinematic, linear_velocity,
            physics_controlled, sphere_collider,
        },
        player::{player, user_id},
        prefab::prefab_from_url,
        rendering::{color, fog_density, light_diffuse, sky, sun, water},
        text::{font_size, text},
        transform::{
            inv_local_to_world, local_to_parent, local_to_world, mesh_to_local, mesh_to_world,
            rotation, scale, spherical_billboard, translation,
        },
    },
    concepts::{make_perspective_infinite_reverse_camera, make_transformable},
    entity::resources,
    prelude::*,
};
use components::player_shoot_requested;
use utils::CameraState;

mod utils;

const BALL_RADIUS: f32 = 0.34;

fn create_environment() {
    make_transformable()
        .with_default(water())
        .with(scale(), Vec3::ONE * 2000.)
        .spawn();

    make_transformable()
        .with_default(sun())
        .with(rotation(), Quat::from_rotation_y(-45_f32.to_radians()))
        .with(light_diffuse(), Vec3::ONE)
        .with(fog_density(), 0.)
        .with_default(main_scene())
        .spawn();

    make_transformable().with_default(sky()).spawn();

    make_transformable()
        .with(prefab_from_url(), asset::url("assets/level.glb").unwrap())
        .with(translation(), Vec3::Z * -0.25)
        .spawn();

    make_transformable()
        .with(model_from_url(), asset::url("assets/fan.glb").unwrap())
        .with(collider_from_url(), asset::url("assets/fan.glb").unwrap())
        .with(kinematic(), ())
        .with(dynamic(), true)
        .with(angular_velocity(), vec3(0., 90_f32.to_radians(), 0.))
        .with(translation(), vec3(-35., 161., 8.4331))
        .with(rotation(), Quat::from_rotation_z(180_f32.to_radians()))
        .spawn();
}

fn make_golf_ball() -> Entity {
    make_transformable()
        .with_default(ball())
        .with_default(physics_controlled())
        .with(dynamic(), true)
        .with(sphere_collider(), BALL_RADIUS)
        .with(model_from_url(), asset::url("assets/ball.glb").unwrap())
}

fn make_text() -> Entity {
    Entity::new()
        .with(
            local_to_parent(),
            Mat4::from_scale(Vec3::ONE * 0.02) * Mat4::from_rotation_x(-180_f32.to_radians()),
        )
        .with(color(), vec4(1., 0., 0., 1.))
        .with(font_size(), 36.)
        .with_default(main_scene())
        .with_default(local_to_world())
        .with_default(mesh_to_local())
        .with_default(mesh_to_world())
}

#[main]
pub fn main() {
    create_environment();

    // Set the initial next player hue.
    entity::add_component(resources(), next_player_hue(), 0.);

    // When a player spawns, create their player state.
    spawn_query(user_id())
        .requires(player())
        .bind(move |players| {
            for (player, player_user_id) in players {
                let next_color = utils::hsv_to_rgb(&[
                    entity::get_component(resources(), next_player_hue()).unwrap_or_default(),
                    0.7,
                    1.0,
                ])
                .extend(1.);
                // 80 + 22.5; pseudo random color, with 16 being unique
                entity::mutate_component(resources(), next_player_hue(), |h| *h += 102.5);

                entity::add_components(player, make_player_state());

                let camera_state = make_player_camera_state().spawn();
                entity::add_component(player, player_camera_state(), camera_state);

                make_perspective_infinite_reverse_camera()
                    .with(aspect_ratio_from_window(), EntityId::resources())
                    .with(user_id(), player_user_id.clone())
                    .with(player_camera_state(), camera_state)
                    .with_default(main_scene())
                    .with_default(local_to_world())
                    .with_default(inv_local_to_world())
                    .with_default(translation())
                    .with_default(rotation())
                    .spawn();

                // TODO: This is a bit... odd
                entity::add_component(player, player_color(), next_color * 2.2);

                let text = make_text()
                    .with(color(), next_color)
                    .with(user_id(), player_user_id.clone())
                    .with(text(), player_user_id.clone())
                    .spawn();
                entity::add_component(player, player_text(), text);

                entity::add_component(
                    player,
                    player_text_container(),
                    make_transformable()
                        .with_default(main_scene())
                        .with_default(local_to_world())
                        .with_default(spherical_billboard())
                        .with(translation(), vec3(-5., 0., 5.))
                        .with(children(), vec![text])
                        .spawn(),
                );

                entity::add_component(
                    player,
                    player_ball(),
                    make_golf_ball()
                        .with(color(), next_color)
                        .with(user_id(), player_user_id.clone())
                        .with(translation(), vec3(-5., 0., 10.))
                        .spawn(),
                );

                entity::add_component(
                    player,
                    player_indicator(),
                    make_transformable()
                        .with(color(), next_color)
                        .with(user_id(), player_user_id.clone())
                        .with(
                            model_from_url(),
                            asset::url("assets/indicator.glb").unwrap(),
                        )
                        .spawn(),
                );

                entity::add_component(
                    player,
                    player_indicator_arrow(),
                    make_transformable()
                        .with(color(), next_color)
                        .with(user_id(), player_user_id.clone())
                        .with(
                            model_from_url(),
                            asset::url("assets/indicator_arrow.glb").unwrap(),
                        )
                        .spawn(),
                );

                entity::add_component(player, player_shoot_requested(), false);
            }
        });

    let flag = make_transformable()
        .with(model_from_url(), asset::url("assets/flag.glb").unwrap())
        .with(collider_from_url(), asset::url("assets/flag.glb").unwrap())
        .with(dynamic(), true)
        .with(kinematic(), ())
        .with(origin(), vec3(-35., 205., 0.3166))
        .spawn();

    // Update the flag every frame.
    query(translation())
        .requires(ball())
        .each_frame(move |balls| {
            let flag_origin = entity::get_component(flag, origin()).unwrap_or_default();
            let mut min_distance = std::f32::MAX;
            for (_, ball_position) in &balls {
                let distance = ball_position.distance(flag_origin);
                if distance < min_distance {
                    min_distance = distance;
                }
            }
            if min_distance < 5. {
                entity::set_component(
                    flag,
                    translation(),
                    flag_origin + Vec3::Z * (5. - min_distance),
                );
            } else {
                entity::set_component(flag, translation(), flag_origin);
            }
        });

    // Update player cameras every frame.
    query(player_camera_state())
        .requires(active_camera())
        .each_frame(move |cameras| {
            for (id, camera_state) in cameras {
                let camera_state = CameraState(camera_state);
                let (camera_translation, camera_rotation) = camera_state.get_transform();
                entity::set_component(id, translation(), camera_translation);
                entity::set_component(id, rotation(), camera_rotation * Quat::from_rotation_x(90.));
            }
        });

    // When a player despawns, clean up their objects.
    let player_objects_query = query(user_id()).build();
    despawn_query(user_id()).requires(player()).bind({
        move |players| {
            let player_objects = player_objects_query.evaluate();
            for (_, player_user_id) in &players {
                for (id, _) in player_objects
                    .iter()
                    .filter(|(_, object_user_id)| *player_user_id == *object_user_id)
                {
                    entity::despawn(*id);
                }
            }
        }
    });

    messages::Input::subscribe(|source, msg| {
        let Some(user_id) = source.client_entity_id() else { return; };

        if let Some(player_camera_state) = entity::get_component(user_id, player_camera_state()) {
            let player_camera_state = CameraState(player_camera_state);

            player_camera_state.zoom(msg.camera_zoom / 25.);
            if msg.camera_rotation.length_squared() > 0. {
                player_camera_state.rotate(msg.camera_rotation / 250.);
            }

            if msg.shoot {
                entity::set_component(user_id, player_shoot_requested(), true);
            }
        }
    });

    ambient_api::messages::Collision::subscribe(move |msg| {
        // TODO: change msg.ids[0] to the bouncing ball
        messages::Bonk::new(msg.ids[0]).send_client_broadcast_reliable();
    });

    let start_time = time();

    // Update player ball each frame.
    query((
        player_ball(),
        player_text(),
        player_text_container(),
        player_indicator(),
        player_indicator_arrow(),
        player_camera_state(),
        player_shoot_requested(),
    ))
    .requires(player())
    .each_frame(move |players| {
        for (
            player,
            (
                player_ball,
                player_text,
                player_text_container,
                player_indicator,
                player_indicator_arrow,
                player_camera_state,
                player_shoot_requested,
            ),
        ) in players
        {
            let player_camera_state = CameraState(player_camera_state);

            let ball_position =
                entity::get_component(player_ball, translation()).unwrap_or_default();

            player_camera_state.set_position(ball_position);

            let can_shoot = {
                let lv = entity::get_component(player_ball, linear_velocity()).unwrap_or_default();
                lv.xy().length_squared() < 1.0 && !is_vertically_moving(lv)
            };

            let force_multiplier = {
                let mut mul = (time() - start_time).as_secs_f32() % 2.0;
                if mul > 1.0 {
                    mul = 1.0 - (mul - 1.0);
                }
                mul
            };

            entity::set_component(
                player_text_container,
                translation(),
                ball_position + Vec3::Z * 2.,
            );

            // TODO: This can be removed after #114 is resolved.
            let player_color = entity::get_component(player, player_color()).unwrap_or_default();
            entity::set_component(player_ball, color(), player_color);
            entity::set_component(player_indicator, color(), player_color);
            entity::set_component(player_indicator_arrow, color(), player_color);

            let camera_rotation = Quat::from_rotation_z(player_camera_state.get_yaw());
            let camera_direction = camera_rotation * -Vec3::Y;

            entity::set_component(player_indicator, translation(), ball_position);
            entity::set_component(player_indicator, rotation(), camera_rotation);

            if can_shoot {
                entity::set_component(player_indicator, scale(), vec3(1.0, force_multiplier, 1.0));

                let arrow_position = ball_position + camera_direction * force_multiplier * 10.;
                entity::set_components(
                    player_indicator_arrow,
                    Entity::new()
                        .with(translation(), arrow_position)
                        .with(rotation(), camera_rotation)
                        .with(scale(), Vec3::ONE),
                );
            } else {
                entity::set_component(player_indicator, scale(), Vec3::ZERO);
                entity::set_component(player_indicator_arrow, scale(), Vec3::ZERO);
            }

            if ball_position.z < 0.25 {
                entity::set_component(player_ball, linear_velocity(), Vec3::ZERO);
                entity::set_component(player_ball, angular_velocity(), Vec3::ZERO);
                entity::set_component(
                    player_ball,
                    translation(),
                    entity::get_component(player, player_restore_point()).unwrap_or_default(),
                );
            }

            if player_shoot_requested {
                if can_shoot {
                    entity::set_component(player, player_restore_point(), ball_position);
                    entity::set_component(
                        player_ball,
                        linear_velocity(),
                        camera_direction * 50. * force_multiplier,
                    );
                    messages::Hit::new(player_ball).send_client_broadcast_reliable();
                    let stroke_count = entity::get_component(player, player_stroke_count())
                        .unwrap_or_default()
                        + 1;
                    entity::set_component(player_text, text(), stroke_count.to_string());
                    entity::set_component(player, player_stroke_count(), stroke_count);
                }
                entity::set_component(player, components::player_shoot_requested(), false);
            }

            // HACK: Artificially slow down ball until https://github.com/AmbientRun/Ambient/issues/182 is available
            physics::add_force(player_ball, {
                let lv = entity::get_component(player_ball, linear_velocity()).unwrap_or_default();
                let lvl = lv.length();
                if lvl > 0.0 && !is_vertically_moving(lv) {
                    -65.0 * frametime() * lv.xy().extend(0.0) * (1.0 / lvl)
                } else {
                    Vec3::ZERO
                }
            });
        }
    });
}

fn is_vertically_moving(linear_velocity: Vec3) -> bool {
    linear_velocity.z.abs() > 0.1
}
