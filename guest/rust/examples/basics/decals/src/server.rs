use ambient_api::{
    components::core::{
        app::main_scene,
        camera::aspect_ratio_from_window,
        primitives::{cube, quad},
        rendering::{color, decal_from_url, transparency_group},
        transform::{lookat_target, rotation, scale, translation},
    },
    concepts::make_perspective_infinite_reverse_camera,
    prelude::*,
};

use core::f32::consts::PI;

// from discussion at
// https://discord.com/channels/894505972289134632/1078283561540530216/1096581219925377195

#[main]
pub fn main() {
    // Camera.
    Entity::new()
        .with_merge(make_perspective_infinite_reverse_camera())
        .with(aspect_ratio_from_window(), EntityId::resources())
        .with(translation(), vec3(1.0, 1.0, 2.0))
        .with(lookat_target(), vec3(0.0, 0.0, 0.0))
        .with_default(main_scene())
        .spawn();

    // Scene geometry.
    Entity::new()
        .with_default(cube())
        .with(translation(), vec3(-0.5, -0.5, 0.0))
        .with(scale(), vec3(0.9, 0.9, 0.9))
        .with(color(), vec4(0.5, 0.5, 0.5, 1.0))
        .spawn();
    Entity::new()
        .with_default(quad())
        .with(scale(), 3.0 * Vec3::ONE)
        .spawn();

    // Decal projection volume.
    let decal_scale = vec3(1.0, 1.0, 1.0);
    let decal_rotation = Quat::from_rotation_z(PI / 4.0);
    let decal_url = asset::url("assets/pipeline.toml/0/mat.json").unwrap();
    Entity::new()
        .with(rotation(), decal_rotation)
        .with(scale(), decal_scale)
        .with(decal_from_url(), decal_url)
        .spawn();

    // Decal projection volume visualization.
    Entity::new()
        .with_default(cube())
        .with(rotation(), decal_rotation)
        .with(scale(), decal_scale)
        .with(color(), vec4(0.0, 1.0, 1.0, 0.5))
        .with(transparency_group(), 0)
        .spawn();
}
