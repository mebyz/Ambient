use ambient_api::{
    client::{material, mesh, sampler, texture},
    components::core::{
        camera::aspect_ratio_from_window,
        primitives::quad,
        prefab::prefab_from_url,
        procedurals::{procedural_material, procedural_mesh},
    },
    concepts::{make_perspective_infinite_reverse_camera, make_transformable},
    mesh::Vertex,
    prelude::*,
};
use noise::{utils::*, Fbm, Perlin};
use components::rotating_sun;
use palette::IntoColor;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

const RESOLUTION_X: u32 = 32;
const RESOLUTION_Y: u32 = 8;
const TEXTURE_RESOLUTION_X: u32 = 4 * RESOLUTION_X;
const TEXTURE_RESOLUTION_Y: u32 = 4 * RESOLUTION_Y;
const SIZE_X: f32 = RESOLUTION_X as f32 / RESOLUTION_Y as f32;
const SIZE_Y: f32 = 1.0;
use image::{DynamicImage,GenericImageView};


const TAU: f32 = std::f32::consts::TAU;
const WAVE_AMPLITUDE: f32 = 0.25;
const WAVE_FREQUENCY: f32 = 0.5 * TAU;

#[derive(Clone)]
pub struct TreeMesh {
    pub seed: i32,
    pub trunk_radius: f32,
    pub trunk_height: f32,
    pub trunk_segments: u32,
    pub branch_length: f32,
    pub branch_angle: f32,
    pub branch_segments: u32,
    pub foliage_radius: f32,
    pub foliage_density: u32,
    pub foliage_segments: u32,
}

pub struct MeshDescriptor {
    vertices: Vec<Vertex>,
    indices: Vec<u32>,
}

pub fn create_tree(tree: TreeMesh) -> MeshDescriptor {
    // Create the trunk
    let (mut vertices1, top_vertices1, mut normals1, mut uvs1, trunk_direction) =
        build_trunk(&tree);

    let sectors = 12;
    let trunk_segments = tree.trunk_segments;
    let mut indices = Vec::new();

    // Connect trunk segments
    for i in 0..(trunk_segments) {
        for j in 0..sectors {
            let k1 = i * (sectors + 1) + j;
            let k2 = (i + 1) * (sectors + 1) + j;

            indices.push(k1);
            indices.push(k1 + 1);
            indices.push(k2);

            indices.push(k1 + 1);
            indices.push(k2 + 1);
            indices.push(k2);
        }
    }

    // Generate branches
    let branch_count = tree.branch_segments;
    let branch_radius_variance = 0.02;
    let branch_position_variance = vec3(0.8, 0.8, 0.7);

    let mut rng = ChaCha8Rng::seed_from_u64(tree.seed as u64);

    for i in 0..branch_count {
        let branch_radius = 0.3;
        //* (1.0 - rng.gen_range(0.0..1.0) * branch_radius_variance);
        let mut branch_position = top_vertices1[rng.gen_range(0..top_vertices1.len())]
            + vec3(
                rng.gen_range(0.0..1.0) * branch_position_variance.x,
                rng.gen_range(0.0..1.0) * branch_position_variance.y,
                rng.gen_range(0.0..1.0) * branch_position_variance.z - 1.0,
            );

        let segments = tree.branch_segments;
        let sector_step = 2. * std::f32::consts::PI / segments as f32;

        let mut branch_vertices = Vec::new();
        let mut branch_normals = Vec::new();
        let mut branch_uvs = Vec::new();

        let mut direction = vec3(0.0, 0.0, 1.0);
        let direction_variance = 0.05;

        // Get a random vertex from the top vertices of the trunk
        let random_vertex_index = rng.gen_range(0..top_vertices1.len());
        let random_vertex = normals1[random_vertex_index];

        // Calculate the initial direction of the branch from the chosen vertex
        direction = (random_vertex - branch_position).normalize()
            + vec3(
                gen_rn(tree.seed + i as i32 + 4, -1.0, 1.0),
                gen_rn(tree.seed + i as i32 + 5, -1.0, 1.0),
                0.0,
            );

        for i in 0..=segments {
            let random_direction = vec3(
                gen_rn(tree.seed + i as i32 + 1, 0.0, 1.0) - 0.5,
                gen_rn(tree.seed + i as i32 + 2, 0.0, 1.0) - 0.5,
                gen_rn(tree.seed + i as i32 + 3, 0.0, 1.0) - 0.5,
            )
            .normalize()
                * direction_variance;
            direction = (direction + random_direction).normalize();

            let theta = (i as f32 / segments as f32) * tree.branch_angle;
            let height = branch_position.z + (tree.branch_length * theta.cos()) * direction.z;
            let segment_radius = branch_radius * theta.sin();

            for j in 0..=sectors {
                let phi = j as f32 * sector_step;
                let x = branch_position.x + segment_radius * phi.cos();
                let y = branch_position.y + segment_radius * phi.sin();
                let z = height;

                branch_vertices.push(vec3(x, y, z));
                branch_normals.push(
                    vec3(
                        x - branch_position.x,
                        y - branch_position.y,
                        z - branch_position.z,
                    )
                    .normalize(),
                );
                branch_uvs.push(vec2(j as f32 / sectors as f32, i as f32 / segments as f32));
            }
            branch_position = branch_position
                + vec3(
                    rng.gen_range(-1.0..1.0) * branch_position_variance.x,
                    rng.gen_range(-1.0..1.0) * branch_position_variance.y,
                    rng.gen_range(0.0..1.0) * branch_position_variance.z,
                );
        }

        let branch_indices = generate_branch_indices(segments as usize, sectors as usize);

        vertices1.extend(branch_vertices.clone());
        normals1.extend(branch_normals);
        uvs1.extend(branch_uvs);

        let offset = vertices1.len() - branch_vertices.len();
        indices.extend(branch_indices.iter().map(|i| *i + offset as u32));
    }

    // Generate foliage
    let foliage_count = tree.foliage_density + tree.foliage_segments;
    let foliage_radius_variance = 0.05;
    let foliage_position_variance = vec3(3.0, 3.0, 3.0);

    for i in 0..foliage_count {
        let foliage_radius = tree.foliage_radius
            * (1.0 - gen_rn(tree.seed + i as i32, 0.0, 1.0) * foliage_radius_variance);
        let foliage_position = top_vertices1
            [gen_rn(tree.seed, 0.0, top_vertices1.len() as f32) as usize]
            + vec3(
                gen_rn(tree.seed + i as i32, -1.0, 1.0) * foliage_position_variance.x,
                gen_rn(tree.seed + i as i32 + 1, -1.0, 1.0) * foliage_position_variance.y,
                gen_rn(tree.seed + i as i32 + 2, 0.0, 1.0) * foliage_position_variance.z + 2.0,
            );

        let segments = tree.foliage_segments;
        let density = tree.foliage_density;
        let sector_step = 2. * std::f32::consts::PI / density as f32;

        let mut sphere_vertices = Vec::new();
        let mut sphere_normals = Vec::new();
        let mut sphere_uvs = Vec::new();

        for i in 0..=segments {
            let theta = (i as f32 / segments as f32) * std::f32::consts::PI;
            let height = foliage_position.z + (foliage_radius * theta.cos());
            let segment_radius = foliage_radius * theta.sin();

            for j in 0..=density {
                let phi = j as f32 * sector_step;
                let x = foliage_position.x + segment_radius * phi.cos();
                let y = foliage_position.y + segment_radius * phi.sin();
                let z = height;

                sphere_vertices.push(vec3(x, y, z));
                sphere_normals.push(
                    vec3(
                        x - foliage_position.x,
                        y - foliage_position.y,
                        z - foliage_position.z,
                    )
                    .normalize(),
                );
                sphere_uvs.push(vec2(j as f32 / density as f32, i as f32 / segments as f32));
            }
        }

        let sphere_indices = generate_sphere_indices(segments as usize, density as usize);

        vertices1.extend(sphere_vertices.clone());
        normals1.extend(sphere_normals);
        uvs1.extend(sphere_uvs);

        let offset = vertices1.len() - sphere_vertices.len();
        indices.extend(sphere_indices.iter().map(|i| *i + offset as u32));
    }

    // Function to generate indices for a sphere based on segments and density
    fn generate_sphere_indices(segments: usize, density: usize) -> Vec<u32> {
        let mut indices = Vec::with_capacity(segments * density * 6);

        for i in 0..segments {
            for j in 0..density {
                let index1 = i * (density + 1) + j;
                let index2 = index1 + 1;
                let index3 = (i + 1) * (density + 1) + j;
                let index4 = index3 + 1;

                indices.push(index1 as u32);
                indices.push(index2 as u32);
                indices.push(index3 as u32);

                indices.push(index2 as u32);
                indices.push(index4 as u32);
                indices.push(index3 as u32);
            }
        }
        indices
    }

    // Function to generate indices for a branch based on segments and sectors
    fn generate_branch_indices(segments: usize, sectors: usize) -> Vec<u32> {
        let mut indices = Vec::with_capacity(segments * sectors * 6);

        for i in 0..segments {
            for j in 0..sectors {
                let index1 = i * (sectors + 1) + j;
                let index2 = index1 + 1;
                let index3 = (i + 1) * (sectors + 1) + j;
                let index4 = index3 + 1;

                indices.push(index1 as u32);
                indices.push(index2 as u32);
                indices.push(index3 as u32);

                indices.push(index2 as u32);
                indices.push(index4 as u32);
                indices.push(index3 as u32);
            }
        }
        indices
    }

    let mut vertices: Vec<Vertex> = Vec::with_capacity(vertices1.len());

    for i in 0..vertices1.len() {
        let px = vertices1[i].x;
        let py = vertices1[i].y;
        let pz = vertices1[i].z;
        let u = uvs1[i].x;
        let v = uvs1[i].y;
        let nx = normals1[i].x;
        let ny = normals1[i].y;
        let nz = normals1[i].z;

        let v = mesh::Vertex {
            position: vec3(px, py, pz) + vec3(-0.5 * SIZE_X, -0.5 * SIZE_Y, 0.0),
            normal: vec3(nx, ny, nz),
            tangent: vec3(1.0, 0.0, 0.0),
            texcoord0: vec2(u, v),
        };
        vertices.push(v);
    }

    MeshDescriptor { vertices, indices }
}

fn build_trunk(tree: &TreeMesh) -> (Vec<Vec3>, Vec<Vec3>, Vec<Vec3>, Vec<Vec2>, Vec3) {
    let sectors = 12;
    let sector_step = 2. * std::f32::consts::PI / sectors as f32;

    let mut vertices = Vec::new();
    let mut top_vertices = Vec::new();
    let mut normals = Vec::new();
    let mut uvs = Vec::new();

    let mut trunk_direction = vec3(0.0, 0.0, 1.0);
    let direction_variance = 0.08;

    let radius_variance = 0.02;

    for i in 0..=tree.trunk_segments {
        let variance = gen_rn(tree.seed + i as i32, 0.0, 1.0) * radius_variance;
        let z = tree.trunk_height * (i as f32 / tree.trunk_segments as f32);
        let s = tree.trunk_radius;
        let radius = s * (1.0 - i as f32 / tree.trunk_segments as f32) * (1.0 - variance);

        let top_segment_radius = tree.trunk_radius * 0.1;
        let radius = if i == tree.trunk_segments && radius < top_segment_radius {
            top_segment_radius
        } else {
            radius
        };

        let random_direction = vec3(
            gen_rn(tree.seed + i as i32 + 1, 0.0, 1.0) - 0.5,
            gen_rn(tree.seed + i as i32 + 2, 0.0, 1.0) - 0.5,
            gen_rn(tree.seed + i as i32 + 3, 0.0, 1.0) - 0.5,
        )
        .normalize()
            * direction_variance;
        trunk_direction = (trunk_direction + random_direction).normalize();

        let top_position = trunk_direction * z;

        let gravity_factor = (1.0 - (i as f32 / tree.trunk_segments as f32)).powf(2.0);
        let gravity_offset = trunk_direction * gravity_factor * 2.0 * i as f32;

        for j in 0..=sectors {
            let sector_angle = j as f32 * sector_step;
            let x = radius * sector_angle.cos();
            let y = radius * sector_angle.sin();

            vertices.push(top_position + vec3(x, y, 0.0) - gravity_offset);
            normals.push(vec3(x, y, 0.0).normalize());
            uvs.push(vec2(
                j as f32 / sectors as f32,
                i as f32 / tree.trunk_segments as f32,
            ));
        }

        if i == tree.trunk_segments {
            let top_vertex_start = vertices.len() - sectors - 1;
            let top_vertex_end = vertices.len();
            top_vertices.extend(vertices[top_vertex_start..top_vertex_end].iter().cloned());

            // Add faces to connect the last ring of vertices
            for j in 0..sectors {
                let v1 = top_vertex_start + j;
                let v2 = top_vertex_start + j + 1;
                let v3 = top_vertex_end - 1;
                let v4 = top_vertex_start;

                // First triangle
                vertices.push(vertices[v1] - gravity_offset);
                vertices.push(vertices[v2] - gravity_offset);
                vertices.push(vertices[v3] - gravity_offset);

                normals.push(normals[v1]);
                normals.push(normals[v2]);
                normals.push(normals[v3]);

                uvs.push(uvs[v1]);
                uvs.push(uvs[v2]);
                uvs.push(uvs[v3]);

                // Second triangle
                vertices.push(vertices[v1] - gravity_offset);
                vertices.push(vertices[v3] - gravity_offset);
                vertices.push(vertices[v4] - gravity_offset);

                normals.push(normals[v1]);
                normals.push(normals[v3]);
                normals.push(normals[v4]);

                uvs.push(uvs[v1]);
                uvs.push(uvs[v3]);
                uvs.push(uvs[v4]);
            }
        }
    }

    (vertices, top_vertices, normals, uvs, trunk_direction)
}

pub fn gen_rn(seed: i32, min: f32, max: f32) -> f32 {
    let mut rng = ChaCha8Rng::seed_from_u64(seed as u64);
    rng.gen_range(min..max)
}

fn make_camera() {
    Entity::new()
        .with_merge(make_perspective_infinite_reverse_camera())
        .with(aspect_ratio_from_window(), EntityId::resources())
        .with_default(main_scene())
        .with(translation(), vec3(-10.0, -10.0, 5.0))
        .with(lookat_target(), vec3(70.0, 70.0, 0.0))
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
            Quat::from_rotation_y(-45_f32.to_radians())
                * Quat::from_rotation_z(-45_f32.to_radians()),
        )
        .with(light_diffuse(), Vec3::ONE * 4.0)
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
                    Quat::from_rotation_z(frametime()) * sun_rotation,
                );
            }
        });
}

fn make_ground() {
    Entity::new()
        .with_merge(make_transformable())
        .with_default(quad())
        .with(color(), vec4(0.25, 1.0, 0.25, 1.0))
        .with(translation(), vec3(0.0, 0.0, -0.5))
        .with(scale(), 32.0 * Vec3::ONE)
        .spawn();
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

struct SimplexNoise {
    grad3: [[i32; 3]; 12],
    perm: Vec<i32>,
    sqrt3: f32,
}

impl SimplexNoise {
    fn new() -> SimplexNoise {
        let grad3: [[i32; 3]; 12] = [
            [1, 1, 0],
            [-1, 1, 0],
            [1, -1, 0],
            [-1, -1, 0],
            [1, 0, 1],
            [-1, 0, 1],
            [1, 0, -1],
            [-1, 0, -1],
            [0, 1, 1],
            [0, -1, 1],
            [0, 1, -1],
            [0, -1, -1],
        ];

        let p: Vec<i32> = vec![
            151, 160, 137, 91, 90, 15, 131, 13, 201, 95, 96, 53, 194, 233, 7, 225, 140, 36, 103, 30,
            69, 142, 8, 99, 37, 240, 21, 10, 23, 190, 6, 148, 247, 120, 234, 75, 0, 26, 197, 62, 94,
            252, 219, 203, 117, 35, 11, 32, 57, 177, 33, 88, 237, 149, 56, 87, 174, 20, 125, 136, 171,
            168, 68, 175, 74, 165, 71, 134, 139, 48, 27, 166, 77, 146, 158, 231, 83, 111, 229, 122, 60,
            211, 133, 230, 220, 105, 92, 41, 55, 46, 245, 40, 244, 102, 143, 54, 65, 25, 63, 161, 1,
            216, 80, 73, 209, 76, 132, 187, 208, 89, 18, 169, 200, 196, 135, 130, 116, 188, 159, 86,
            164, 100, 109, 198, 173, 186, 3, 64, 52, 217, 226, 250, 124, 123, 5, 202, 38, 147, 118,
            126, 255, 82, 85, 212, 207, 206, 59, 227, 47, 16, 58, 17, 182, 189, 28, 42, 223, 183, 170,
            213, 119, 248, 152, 2, 44, 154, 163, 70, 221, 153, 101, 155, 167, 43, 172, 9, 129, 22, 39,
            253, 19, 98, 108, 110, 79, 113, 224, 232, 178, 185, 112, 104, 218, 246, 97, 228, 251, 34,
            242, 193, 238, 210, 144, 12, 191, 179, 162, 241, 81, 51, 145, 235, 249, 14, 239, 107, 49,
            192, 214, 31, 181, 199, 106, 157, 184, 84, 204, 176, 115, 121, 50, 45, 127, 4, 150, 254,
            138, 236, 205, 93, 222, 114, 67, 29, 24, 72, 243, 141, 128, 195, 78, 66, 215, 61, 156, 180,
        ];

        let mut perm = vec![0; 512];
        for i in 0..512 {
            perm[i] = p[i & 255];
        }

        SimplexNoise {
            grad3,
            perm,
            sqrt3: f32::sqrt(3.0),
        }
    }

    fn floor(&self, n: f32) -> i32 {
        if n > 0.0 {
            n as i32
        } else {
            (n as i32) - 1
        }
    }

    fn dot(&self, g: [i32; 3], x: f32, y: f32) -> f32 {
        (g[0] as f32) * x + (g[1] as f32) * y
    }

    fn noise(&self, x: f32, y: f32) -> f32 {
        let n0: f32;
        let n1: f32;
        let n2: f32;
        let f2: f32 = 0.5 * (self.sqrt3 - 1.0);
        let s: f32 = (x + y) * f2;
        let i: i32 = self.floor(x + s);
        let j: i32 = self.floor(y + s);
        let g2: f32 = (3.0 - self.sqrt3) / 6.0;
        let t: f32 = (i + j) as f32 * g2;
        let x0: f32 = x - (i as f32) + t;
        let y0: f32 = y - (j as f32) + t;
        let (i1, j1): (i32, i32);
        if x0 > y0 {
            i1 = 1;
            j1 = 0;
        } else {
            i1 = 0;
            j1 = 1;
        }
        let x1: f32 = x0 - (i1 as f32) + g2;
        let y1: f32 = y0 - (j1 as f32) + g2;
        let x2: f32 = x0 - 1.0 + 2.0 * g2;
        let y2: f32 = y0 - 1.0 + 2.0 * g2;
        let ii: i32 = i & 255;
        let jj: i32 = j & 255;
        let gi0: i32 = self.perm[(ii + self.perm[jj as usize]) as usize] % 12;
        let gi1: i32 = self.perm[(ii + i1 + self.perm[(jj + j1) as usize]) as usize] % 12;
        let gi2: i32 = self.perm[(ii + 1 + self.perm[(jj + 1) as usize]) as usize] % 12;
        let t0: f32 = 0.5 - x0 * x0 - y0 * y0;
        if t0 < 0.0 {
            n0 = 0.0;
        } else {
            let t02: f32 = t0 * t0;
            n0 = t02 * t02 * self.dot(self.grad3[gi0 as usize], x0, y0);
        }
        let t1: f32 = 0.5 - x1 * x1 - y1 * y1;
        if t1 < 0.0 {
            n1 = 0.0;
        } else {
            let t12: f32 = t1 * t1;
            n1 = t12 * t12 * self.dot(self.grad3[gi1 as usize], x1, y1);
        }
        let t2: f32 = 0.5 - x2 * x2 - y2 * y2;
        if t2 < 0.0 {
            n2 = 0.0;
        } else {
            let t22: f32 = t2 * t2;
            n2 = t22 * t22 * self.dot(self.grad3[gi2 as usize], x2, y2);
        }
        70.0 * (n0 + n1 + n2)
    }

    fn harmonic_noise_2d(
        &self,
        x: f32,
        y: f32,
        harmonics: i32,
        freq_x: f32,
        freq_y: f32,
        smoothness: f32,
    ) -> f32 {
        let mut h: f32 = 1.0;
        let mut sum: f32 = 0.0;
        for _ in 0..harmonics {
            sum += self.noise(x * h * freq_x, y * h * freq_y) / smoothness;
            h *= 2.0;
        }
        sum
    }
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
    let mut rng = rand_pcg::Pcg64::seed_from_u64(0);
    let dist_zero_to_255 = rand::distributions::Uniform::new_inclusive(0_u8, 255_u8);

    let base_color_map = make_texture(|x, _| {
        let hsl = palette::Hsl::new(360.0 * x, 1.0, 0.5).into_format::<f32>();
        let rgb: palette::LinSrgb = hsl.into_color();
        let r = (255.0 * rgb.red) as u8;
        let g = (255.0 * rgb.green) as u8;
        let b = (255.0 * rgb.blue) as u8;
        let a = 255;
        [r, g, b, a]
    });

let fbm = Fbm::<Perlin>::default();
let bounds = 2.0;
let noise_map = PlaneMapBuilder::<_, 2>::new(fbm)
    .set_size(TEXTURE_RESOLUTION_X as _, TEXTURE_RESOLUTION_Y as _)
    .set_y_bounds(-bounds * f64::from(SIZE_Y), bounds * f64::from(SIZE_Y))
    .set_x_bounds(-bounds * f64::from(SIZE_X), bounds * f64::from(SIZE_X))
    .build();
let mut noise_iter = noise_map.iter();


    let base_color_map2 = make_texture(|x, y| {
            let hsl = palette::Hsl::new(360.0 * (WAVE_AMPLITUDE * f32::sin(WAVE_FREQUENCY * y)), 1.0, 0.5).into_format::<f32>();
            let rgb: palette::LinSrgb = hsl.into_color();
            let r = (255.0 * rgb.red) as u8;
            let g = (255.0 * rgb.green) as u8;
            let b = (255.0 * rgb.blue) as u8;
            let a = 255;
            [r, g, b, a]

        // let r = dist_zero_to_255.sample(&mut rng);
        // let g = 255;
        // let b = 255;
        // let a = dist_zero_to_255.sample(&mut rng);
        // [r, g, b, a]
        ///// let height = (x+y)/1000.0;//get_height2((x*500.0)as i32, (y*500.0) as i32) * 100.0;
        ///// let gray_u8 = (height) as u8;
        ///// [255-gray_u8, 255-gray_u8, 255, 255]
        // let noise = *noise_iter.next().unwrap();
        // let noise = (255.0 * 0.5 * (noise + 1.0)) as u8;
        // [noise, noise, noise, 255]
    }/*|x, y| {
        let height: f32 = 255.0 - (x+y) as f32;// get_height(x as i32, y as i32) as f32 *10.0; //get_height(x as i32, y as i32) as f32 * 100.0; // 255.0;
        let gray_u8 = height.clamp(0.0, 255.0) as u8;
        [gray_u8, gray_u8, gray_u8, 255]
    }*/);



    let normal_map = make_texture(|_, _| [128, 128, 255, 0]);
    let normal_map2 = make_texture(|_, _| [255, 128, 255, 0]);
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
        normal_map : normal_map2,
        metallic_roughness_map : metallic_roughness_map,
        sampler : default_nearest_sampler(),
        transparent: false,
    });
    let material = material::create(&material::Descriptor {
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
            let tree = create_tree(TreeMesh {
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
                Entity::new()
                    .with(procedural_mesh(), mesh)
                    .with(procedural_material(), material)
                    .with_default(cast_shadows()),
            );
        }
    });

    spawn_query(
        (components::tile_seed(),
        components::tile_size(),
        components::tile_x(),
        components::tile_y(),
    )).bind(move |tiles| {
        for (id, (seed, size, tile_x, tile_y)) in tiles {
            let tile = create_tile(GridMesh {
                top_left: Vec2 { x: tile_x as f32 * size, y: tile_y as f32 * size},
                size: Vec2 { x: size, y: size },
                n_vertices_width: 10,
                n_vertices_height: 10,
                uv_min: Vec2 { x: (tile_x as f32)/5.0, y: (tile_y as f32)/5.0 },
                uv_max: Vec2 { x: (tile_x as f32 + 1.0)/5.0, y: (tile_y as f32 + 1.0)/5.0},
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
                    //.with(color(), vec4(0.25, 1.0, 0.25, 1.0))
                    .with(procedural_material(), material2)
                    // .with(
                    //     pbr_material_from_url(),
                    //     asset::url("assets/pipeline.json/0/mat.json").unwrap(),
                    // )
                    .with_default(cast_shadows()),
            );
        }
    });
}

fn make_trees() {
    let seed = 123456;
    let num_trees = 100;

    // lets plant some trees :)
    for i in 0..num_trees {
        let trunk_radius = gen_rn(seed + i, 2.0, 3.0);
        let trunk_height = gen_rn(seed + i, 15.0, 20.0);
        let trunk_segments = gen_rn(seed + i, 6.0, 12.0) as u32;
        let branch_length = gen_rn(seed + i, 0.1, 0.3);
        let branch_angle = gen_rn(seed + i, 10., 12.);

        let position = vec3(
            gen_rn(seed + i, 0.0, 50.0),
            gen_rn(seed + seed + i, 0.0, 50.0),
            -1.0,
        );

        let id = Entity::new()
            .with_merge(concepts::make_tree())
            .with_merge(make_transformable())
            .with(scale(), Vec3::ONE * gen_rn(i, 0.2, 0.4))
            .with(translation(), position)
            .with(components::tree_seed(), seed + i)
            .with(components::tree_trunk_radius(), trunk_radius)
            .with(components::tree_trunk_height(), trunk_height)
            .with(components::tree_trunk_segments(), trunk_segments)
            .with(components::tree_branch_length(), branch_length)
            .with(components::tree_branch_angle(), branch_angle)
            .spawn();
    }
}

fn make_tiles() {
    let num_tiles_x = 5;
    let num_tiles_y = 5;
    let size = 5.0;
    let seed = 123;

    for num_tile_x in 0..num_tiles_x {
        for num_tile_y in 0..num_tiles_y {

            let id = Entity::new()
                .with_merge(concepts::make_tile())
                .with_merge(make_transformable())
                .with(components::tile_seed(), seed + num_tile_x + num_tile_y * num_tiles_x)
                .with(components::tile_size(), size)
                .with(components::tile_x(), num_tile_x)
                .with(components::tile_y(), num_tile_y)
                .spawn();
        }
    }
}

#[derive(Debug, Clone)]
pub struct GridMesh {
    pub top_left: glam::Vec2,
    pub size: glam::Vec2,
    pub n_vertices_width: usize,
    pub n_vertices_height: usize,
    pub uv_min: glam::Vec2,
    pub uv_max: glam::Vec2,
    pub normal: glam::Vec3,
}

impl Default for GridMesh {
    fn default() -> GridMesh {
        GridMesh {
            top_left: glam::Vec2::ZERO,
            size: glam::Vec2::ONE,
            n_vertices_width: 2,
            n_vertices_height: 2,
            uv_min: glam::Vec2::ZERO,
            uv_max: glam::Vec2::ONE,
            normal: glam::Vec3::Z,
        }
    }
}

pub fn create_tile(grid: GridMesh) -> MeshDescriptor {
    // Create the tile
    let (mut vertices1, mut uvs1, mut normals1, mut indices) = build_tile(&grid);

    let mut vertices: Vec<Vertex> = Vec::with_capacity(vertices1.len());

    for i in 0..vertices1.len() {
        let px = vertices1[i].x;
        let py = vertices1[i].y;
        let pz = vertices1[i].z;
        let u = uvs1[i].x;
        let v = uvs1[i].y;
        let nx = normals1[i].x;
        let ny = normals1[i].y;
        let nz = normals1[i].z;

        let v = mesh::Vertex {
            position: vec3(px, py, pz) + vec3(-0.5 * SIZE_X, -0.5 * SIZE_Y, 0.0),
            normal: vec3(nx, ny, nz),
            tangent: vec3(1.0, 0.0, 0.0),
            texcoord0: vec2(u, v),
        };
        vertices.push(v);
    }

    MeshDescriptor { vertices, indices }
}

pub fn build_tile(grid: &GridMesh) -> (Vec<Vec3>, Vec<Vec2>, Vec<Vec3>, Vec<u32>) {
    let mut positions = Vec::new();
    let mut texcoords = Vec::new();
    let mut normals = Vec::new();
    let mut indices = Vec::new();
    for y in 0..grid.n_vertices_height {
        for x in 0..grid.n_vertices_width {
            let p = glam::Vec2::new(
                x as f32 / (grid.n_vertices_width as f32 - 1.0),
                y as f32 / (grid.n_vertices_height as f32 - 1.0),
            );
            positions.push(vec3(
                grid.top_left.x + grid.size.x * p.x,
                grid.top_left.y + grid.size.y * p.y,
                get_height2((grid.top_left.x + grid.size.x * p.x) as i32, (grid.top_left.y + grid.size.y * p.y) as i32),
            ));
            texcoords.push(vec2(
                grid.uv_min.x + (grid.uv_max.x - grid.uv_min.x) * p.x,
                grid.uv_min.y + (grid.uv_max.y - grid.uv_min.y) * p.y,
            ));
            let normal = grid.normal;
            normals.push(vec3(normal.x, normal.y, normal.z));
            if y < grid.n_vertices_height - 1 && x < grid.n_vertices_width - 1 {
                let vert_index = x + y * grid.n_vertices_width;
                indices.push((vert_index) as u32);
                indices.push((vert_index + 1) as u32);
                indices.push((vert_index + grid.n_vertices_width) as u32);

                indices.push((vert_index + 1) as u32);
                indices.push((vert_index + grid.n_vertices_width + 1) as u32);
                indices.push((vert_index + grid.n_vertices_width) as u32);
            }
        }
    }

    (positions, texcoords, normals, indices)
}

fn get_height(x:i32, y:i32) -> f32 {
    let x = x as f32;
    let y = y as f32;
    // perlin noise without crate
    let noise = (x.sin() + y.cos()) * 0.5;
    let height = noise * 0.5 + 0.5;
    height
}
fn get_height2(x: i32, y: i32) -> f32 {
    let simplex = SimplexNoise::new();
    let mut height: f32 = 0.0;
    let mut level: f32 = 8.0;
    height += (simplex.noise(x as f32 / level, y as f32 / level) / 2.0 + 0.5) * 5.25;
    // level *= 3.0;
    // height += (simplex.noise(x as f32 / level, y as f32 / level) / 2.0 + 0.5) * 0.7;
    // level *= 2.0;
    // height += (simplex.noise(x as f32 / level, y as f32 / level) / 2.0 + 0.5) * 1.0;
    // level *= 2.0;
    // height -= (f32::cos((x / 2 + 50) as f32 / 40.0) * 2.0)
    //     + (f32::sin((y / 2 + 110) as f32 / 40.0) * 2.0)
    //     + 6.0;
    // height += (simplex.noise(x as f32 / level, y as f32 / level) / 2.0 + 0.5) * 1.8;
    // height /= 1.0 + 0.5 + 0.25 + 0.125;
    //height *= 3.6;
    height// * 20000.0 + 50.0;
    //255.0
}

#[main]
pub async fn main() {

    make_camera();
    make_lighting();
    //make_ground();

    register_augmentors();
    make_trees();
    make_tiles();


    // let unit_id = Entity::new()
    //     .with_merge(make_transformable())
    //     .with(
    //         prefab_from_url(),
    //         asset::url("assets/scene.glb").unwrap(),
    //     )
    //     .with(color(), vec4(1.0, 0.0, 0.0, 1.0))
    //     /*.with(
    //         pbr_material_from_url(),
    //         asset::url("assets/pipeline.json/0/mat.json").unwrap(),
    //     )*/
    //     .with(name(), "974".to_string())
    //     .spawn();
}
