use ambient_api::{
    client::{material, mesh, sampler, texture},
    components::core::{
        camera::aspect_ratio_from_window,
        procedurals::{procedural_material, procedural_mesh},
    },
    concepts::{make_perspective_infinite_reverse_camera, make_transformable},
    prelude::*,
};

use components::rotating_sun;
use palette::IntoColor;

mod grid;
mod tooling;
mod tree;

const RESOLUTION_X: u32 = 32;
const RESOLUTION_Y: u32 = 32;
const TEXTURE_RESOLUTION_X: u32 = 4 * RESOLUTION_X;
const TEXTURE_RESOLUTION_Y: u32 = 4 * RESOLUTION_Y;
const SIZE_X: f32 = RESOLUTION_X as f32 / RESOLUTION_Y as f32;
const SIZE_Y: f32 = 1.0;

const TAU: f32 = std::f32::consts::TAU;
const WAVE_AMPLITUDE: f32 = 0.25;
const WAVE_FREQUENCY: f32 = 0.5 * TAU;

fn make_camera() {
    Entity::new()
        .with_merge(make_perspective_infinite_reverse_camera())
        .with(aspect_ratio_from_window(), EntityId::resources())
        .with_default(main_scene())
        .with(translation(), vec3(0.0, 0.0, 6.0))
        .with(lookat_target(), vec3(5.0, 5.0, 3.0))
        .spawn();
}

#[element_component]
fn App(_hooks: &mut Hooks, sun_id: EntityId) -> Element {
    FocusRoot::el([FlowColumn::el([FlowRow::el([Button::new(
        "Toggle sun rotation",
        move |_| {
            entity::mutate_component(sun_id, rotating_sun(), |rotating_sun| {
                *rotating_sun = !*rotating_sun;
            });
        },
    )
    .el()])])
    .with_padding_even(10.0)])
}

fn make_lighting() {
    let sun_id = Entity::new()
        .with_merge(make_transformable())
        .with_default(sun())
        .with(
            rotation(),
            Quat::from_rotation_y(-90_f32.to_radians())
                * Quat::from_rotation_z(-90_f32.to_radians()),
        )
        .with(light_diffuse(), Vec3::ONE * 10.0)
        .with_default(main_scene())
        .with(rotating_sun(), false)
        .spawn();
    App::el(sun_id).spawn_interactive();
    query((rotation(), (rotating_sun())))
        .requires(sun())
        .each_frame(move |suns| {
            for (sun_id, (sun_rotation, rotating_sun)) in suns {
                if !rotating_sun {
                    continue;
                }
                entity::set_component(
                    sun_id,
                    rotation(),
                    Quat::from_rotation_z(frametime() / 10.0) * sun_rotation,
                );
            }
        });
}

fn make_texture<PixelFn>(mut pixel_fn: PixelFn) -> ProceduralTextureHandle
where
    PixelFn: FnMut(f32, f32) -> [u8; 4],
{
    let mut pixels = vec![0_u8; (4 * TEXTURE_RESOLUTION_X * TEXTURE_RESOLUTION_Y) as usize];
    for y in 0..TEXTURE_RESOLUTION_Y {
        for x in 0..TEXTURE_RESOLUTION_X {
            let dst = (4 * (x + y * TEXTURE_RESOLUTION_X)) as usize;
            let dst = &mut pixels[dst..(dst + 4)];
            let px = (x as f32 + 0.5) / (TEXTURE_RESOLUTION_X as f32);
            let py = (y as f32 + 0.5) / (TEXTURE_RESOLUTION_Y as f32);
            dst.copy_from_slice(&pixel_fn(px, py));
        }
    }
    texture::create_2d(&texture::Descriptor2D {
        width: TEXTURE_RESOLUTION_X,
        height: TEXTURE_RESOLUTION_Y,
        format: texture::Format::Rgba8Unorm,
        data: &pixels,
    })
}

fn default_nearest_sampler() -> ProceduralSamplerHandle {
    sampler::create(&sampler::Descriptor {
        address_mode_u: sampler::AddressMode::ClampToEdge,
        address_mode_v: sampler::AddressMode::ClampToEdge,
        address_mode_w: sampler::AddressMode::ClampToEdge,
        mag_filter: sampler::FilterMode::Nearest,
        min_filter: sampler::FilterMode::Nearest,
        mipmap_filter: sampler::FilterMode::Nearest,
    })
}

fn register_augmentors() {
    let _rng = rand_pcg::Pcg64::seed_from_u64(0);
    let _dist_zero_to_255 = rand::distributions::Uniform::new_inclusive(0_u8, 255_u8);

    let base_color_map = make_texture(|x, _| {
        let hsl = palette::Hsl::new(360.0 * x, 1.0, 0.5).into_format::<f32>();
        let rgb: palette::LinSrgb = hsl.into_color();
        let r = 50; //(255.0 * rgb.red/2.0) as u8;
        let g = (255.0 * rgb.green) as u8;
        let b = 50; //(255.0 * rgb.blue/2.0) as u8;
        let a = 255;
        [r, g, b, a]
    });

    let base_color_map2 = make_texture(|x, y| {
        let mx = x * 10.0;
        let my = y * 10.0;
        let mut h = tooling::get_height(mx, my);
        h = h * 255.0 / 4.0;
        let r = h as u8 + 100;
        let g = h as u8 + 50;
        let b = h as u8;
        let a = 255 as u8;
        [r, g, b, a]
    });

    let metallic_roughness_map2 = make_texture(|x, y| {
        let mx = x * 10.0;
        let my = y * 10.0;
        let mut h = tooling::get_height(mx, my);
        h = h * 255.0 / 10.0;
        let r = h as u8;
        let g = h as u8;
        let b = h as u8;
        let a = 255 as u8;
        [r, g, b, a]
    });

    let normal_map = make_texture(|_, _| [128, 128, 255, 0]);
    let _normal_map2 = make_texture(|_, _| [255, 128, 255, 0]);
    let metallic_roughness_map = make_texture(|_, _| [255, 255, 0, 0]);
    let sampler = sampler::create(&sampler::Descriptor {
        address_mode_u: sampler::AddressMode::ClampToEdge,
        address_mode_v: sampler::AddressMode::ClampToEdge,
        address_mode_w: sampler::AddressMode::ClampToEdge,
        mag_filter: sampler::FilterMode::Nearest,
        min_filter: sampler::FilterMode::Nearest,
        mipmap_filter: sampler::FilterMode::Nearest,
    });

    let material2 = material::create(&material::Descriptor {
        base_color_map: base_color_map2,
        normal_map: base_color_map2,
        metallic_roughness_map: metallic_roughness_map2,
        sampler,
        transparent: false,
    });
    let _material = material::create(&material::Descriptor {
        base_color_map: base_color_map,
        normal_map,
        metallic_roughness_map,
        sampler,
        transparent: false,
    });

    spawn_query((
        components::tree_seed(),
        components::tree_foliage_density(),
        components::tree_foliage_radius(),
        components::tree_foliage_segments(),
        components::tree_branch_length(),
        components::tree_branch_angle(),
        components::tree_trunk_height(),
        components::tree_trunk_radius(),
        components::tree_trunk_segments(),
    ))
    .bind(move |trees| {
        for (
            id,
            (
                seed,
                foliage_density,
                foliage_radius,
                foliage_segments,
                branch_length,
                branch_angle,
                trunk_height,
                trunk_radius,
                trunk_segments,
            ),
        ) in trees
        {
            let tree = tree::create_tree(tree::TreeMesh {
                seed,
                trunk_radius,
                trunk_height,
                trunk_segments,
                branch_length,
                branch_angle,
                branch_segments: 8,
                foliage_radius,
                foliage_density,
                foliage_segments,
            });
            let mesh = mesh::create(&mesh::Descriptor {
                vertices: &tree.vertices,
                indices: &tree.indices,
            });

            entity::add_components(
                id,
                Entity::new().with(procedural_mesh(), mesh), //.with_default(cast_shadows()),
            );
        }
    });

    spawn_query((
        components::tile_seed(),
        components::tile_size(),
        components::tile_x(),
        components::tile_y(),
    ))
    .bind(move |tiles| {
        for (id, (_seed, size, tile_x, tile_y)) in tiles {
            let tile = grid::create_tile(grid::GridMesh {
                top_left: Vec2 {
                    x: tile_x as f32 * size,
                    y: tile_y as f32 * size,
                },
                size: Vec2 { x: size, y: size },
                n_vertices_width: 10,
                n_vertices_height: 10,
                uv_min: Vec2 {
                    x: (tile_x as f32) * size / 5.0,
                    y: (tile_y as f32) * size / 5.0,
                },
                uv_max: Vec2 {
                    x: (tile_x as f32 + 1.0) * size / 5.0,
                    y: (tile_y as f32 + 1.0) * size / 5.0,
                },
                normal: Vec3 {
                    x: 0.0,
                    y: 0.0,
                    z: 1.0,
                },
            });
            let mesh = mesh::create(&mesh::Descriptor {
                vertices: &tile.vertices,
                indices: &tile.indices,
            });

            entity::add_components(
                id,
                Entity::new()
                    .with(procedural_mesh(), mesh)
                    .with(scale(), 2.0 * Vec3::ONE)
                    .with(procedural_material(), material2),
            );
        }
    });
}
fn make_vegetation(vegetation_type: &str) {
    let (seed, num_vegetation) = match vegetation_type {
        "trees" => (123456, 10),
        "trees2" => (123460, 10),
        _ => panic!("Invalid vegetation type"),
    };

    for i in 0..num_vegetation {
        let (
            trunk_radius,
            trunk_height,
            trunk_segments,
            branch_length,
            branch_angle,
            foliage_density,
            foliage_radius,
            foliage_segments,
        ) = match vegetation_type {
            "trees" => (
                tooling::gen_rn(seed + i, 10.0, 15.0),
                tooling::gen_rn(seed + i, 15.0, 20.0),
                tooling::gen_rn(seed + i, 12.0, 20.0) as u32,
                tooling::gen_rn(seed + i, 0.1, 0.3),
                tooling::gen_rn(seed + i, 10.0, 12.0),
                5,
                2.0,
                5,
            ),
            "trees2" => (
                tooling::gen_rn(seed + i, 2.5, 4.0),
                tooling::gen_rn(seed + i, 1.0, 3.0),
                tooling::gen_rn(seed + i, 6.0, 12.0) as u32,
                tooling::gen_rn(seed + i, 0.3, 0.4),
                tooling::gen_rn(seed + i, 60.0, 90.0),
                1,
                1.0,
                1,
            ),
            _ => panic!("Invalid vegetation type"),
        };

        let x = tooling::gen_rn(seed + i, 0.0, 5.0) * 2.0;
        let y = tooling::gen_rn(seed + seed + i, 0.0, 5.0) * 2.0;
        let position = vec3(x, y, tooling::get_height(x, y) * 2.0 - 0.1);

        let _id = Entity::new()
            .with_merge(concepts::make_tree())
            .with_merge(make_transformable())
            .with(
                scale(),
                Vec3::ONE
                    * tooling::gen_rn(
                        i,
                        0.05,
                        0.1
                    ),
            )
            .with(translation(), position)
            .with(components::tree_seed(), seed + i)
            .with(components::tree_trunk_radius(), trunk_radius)
            .with(components::tree_trunk_height(), trunk_height)
            .with(components::tree_trunk_segments(), trunk_segments)
            .with(components::tree_branch_length(), branch_length)
            .with(components::tree_branch_angle(), branch_angle)
            .with(components::tree_foliage_density(), foliage_density)
            .with(components::tree_foliage_radius(), foliage_radius)
            .with(components::tree_foliage_segments(), foliage_segments)
            .with(
                pbr_material_from_url(),
                if vegetation_type == "trees2" {
                    asset::url("assets/pipeline.json/1/mat.json").unwrap()
                } else {
                    asset::url("assets/pipeline.json/0/mat.json").unwrap()
                },
            )
            .spawn();
    }
}

fn make_tiles() {
    let num_tiles_x = 5;
    let num_tiles_y = 5;
    let size = 1.0;
    let seed = 123456;

    for num_tile_x in 0..num_tiles_x {
        for num_tile_y in 0..num_tiles_y {
            let _id = Entity::new()
                .with_merge(concepts::make_tile())
                .with_merge(make_transformable())
                .with(
                    components::tile_seed(),
                    seed + num_tile_x + num_tile_y * num_tiles_x,
                )
                .with(components::tile_size(), size)
                .with(components::tile_x(), num_tile_x)
                .with(components::tile_y(), num_tile_y)
                .spawn();
        }
    }
}

#[main]
pub async fn main() {
    let last_player_position = vec3(0.0, 0.0, 0.0);

    make_lighting();

    register_augmentors();
    make_tiles();
    make_vegetation("trees");
    make_vegetation("trees2");

    let mut cursor_lock = input::CursorLockGuard::new(true);
    ambient_api::messages::Frame::subscribe(move |_| {
        let input = input::get();
        if !cursor_lock.auto_unlock_on_escape(&input) {
            return;
        }

        let mut displace = Vec3::ZERO;
        if input.keys.contains(&KeyCode::W) {
            displace.y -= 1.0;
        }
        if input.keys.contains(&KeyCode::S) {
            displace.y += 1.0;
        }
        if input.keys.contains(&KeyCode::A) {
            displace.x -= 1.0;
        }
        if input.keys.contains(&KeyCode::D) {
            displace.x += 1.0;
        }

        messages::Input::new(displace, input.mouse_delta).send_server_unreliable();
    });
}
