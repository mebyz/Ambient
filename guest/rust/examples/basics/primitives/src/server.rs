use ambient_api::{
    components::core::{
        app::main_scene,
        camera::aspect_ratio_from_window,
        primitives::{
            star_inner_radius, star_outer_radius, star_width, star_spikes_number,
            torus_inner_radius, torus_loops, torus_outer_radius, torus_slices,
            capsule_half_height, capsule_latitudes, capsule_longitudes, capsule_radius,
            capsule_rings, cube, quad, sphere_radius, sphere_sectors, sphere_stacks,
        },
        rendering::color,
        transform::{lookat_target, scale, translation},
    },
    concepts::{
        make_star, make_torus, make_capsule, make_perspective_infinite_reverse_camera, make_sphere, make_transformable,
    },
    prelude::*,
};

#[main]
pub fn main() {
    Entity::new()
        .with_merge(make_perspective_infinite_reverse_camera())
        .with(aspect_ratio_from_window(), EntityId::resources())
        .with_default(main_scene())
        .with(translation(), vec3(5., 5., 6.))
        .with(lookat_target(), vec3(0., 0., 2.))
        .spawn();

    Entity::new()
        .with_merge(make_transformable())
        .with_default(quad())
        .with(scale(), Vec3::ONE * 10.)
        .with(color(), vec4(1., 0., 0., 1.))
        .spawn();

    Entity::new()
        .with_merge(make_transformable())
        .with_default(cube())
        .with(translation(), vec3(2., 0., 0.5))
        .with(scale(), Vec3::ONE)
        .with(color(), vec4(0., 1., 0., 1.))
        .spawn();

    Entity::new()
        .with_merge(make_transformable())
        .with_merge(make_sphere())
        .with(sphere_radius(), 1.)
        .with(sphere_sectors(), 12)
        .with(sphere_stacks(), 6)
        .with(translation(), vec3(0., 2., 0.5))
        .with(color(), vec4(0., 0., 1., 1.))
        .spawn();

    Entity::new()
        .with_merge(make_transformable())
        .with_merge(make_sphere())
        .with(translation(), vec3(2., 2., 0.25))
        .with(color(), vec4(1., 1., 0., 1.))
        .spawn();

    Entity::new()
        .with_merge(make_transformable())
        .with_merge(make_capsule())
        .with(translation(), vec3(-2.0, 2.0, 1.0))
        .with(color(), vec4(1.0, 0.25, 0.0, 1.0))
        .spawn();

    Entity::new()
        .with_merge(make_transformable())
        .with_merge(make_capsule())
        .with(capsule_radius(), 0.25)
        .with(capsule_half_height(), 0.25)
        .with(capsule_rings(), 0)
        .with(capsule_latitudes(), 16)
        .with(capsule_longitudes(), 32)
        .with(translation(), vec3(-2.0, 0.0, 0.5))
        .with(color(), vec4(1.0, 0.0, 0.25, 1.0))
        .spawn();

    Entity::new()
        .with_merge(make_transformable())
        .with_merge(make_torus())
        .with(torus_inner_radius(), 0.25)
        .with(torus_outer_radius(), 0.5)
        .with(torus_slices(), 32)
        .with(torus_loops(), 16)
        .with(translation(), vec3(0.0, -2.0, 0.5))
        .with(color(), vec4(0.0, 1.0, 0.25, 1.0))
        .spawn();

    Entity::new()
        .with_merge(make_transformable())
        .with_merge(make_star())
        .with(star_inner_radius(), 0.25)
        .with(star_outer_radius(), 0.5)
        .with(star_spikes_number(), 10)
        .with(star_width(), 0.1)
        .with(translation(), vec3(-2.0, -2.0, 0.5))
        .with(color(), vec4(0.5, 0.5, 0.5, 1.0))
        .spawn();

    Entity::new()
        .with_merge(make_transformable())
        .with_merge(make_star())
        .with(star_inner_radius(), 0.1)
        .with(star_outer_radius(), 0.4)
        .with(star_spikes_number(), 5)
        .with(star_width(), 0.2)
        .with(translation(), vec3(2.0, -2.0, 0.5))
        .with(color(), vec4(0.2, 0.4, 0.8, 1.0))
        .spawn();

    Entity::new()
        .with_merge(make_transformable())
        .with_merge(make_star())
        .with(star_inner_radius(), 0.2)
        .with(star_outer_radius(), 0.6)
        .with(star_spikes_number(), 6)
        .with(star_width(), 0.3)
        .with(translation(), vec3(-2.0, 0.0, 0.5))
        .with(color(), vec4(0.8, 0.6, 0.2, 1.0))
        .spawn();

    Entity::new()
        .with_merge(make_transformable())
        .with_merge(make_star())
        .with(star_inner_radius(), 0.3)
        .with(star_outer_radius(), 0.8)
        .with(star_spikes_number(), 7)
        .with(star_width(), 0.4)
        .with(translation(), vec3(0.0, 2.0, 0.5))
        .with(color(), vec4(0.4, 0.8, 0.2, 1.0))
        .spawn();
}
