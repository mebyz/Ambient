
struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) texcoord: vec2<f32>,
    @location(1) world_position: vec4<f32>,
    @location(2) @interpolate(flat) instance_index: u32,
    @location(3) inv_local_to_world_0: vec4<f32>,
    @location(4) inv_local_to_world_1: vec4<f32>,
    @location(5) inv_local_to_world_2: vec4<f32>,
    @location(6) inv_local_to_world_3: vec4<f32>,
};

fn get_entity_primitive_mesh(loc: vec2<u32>, index: u32) -> u32 {
    let i = index >> 2u;
    let j = index & 3u;

    var meshes = get_entity_gpu_primitives_mesh(loc);
    return bitcast<u32>(meshes[i][j]);
}

@vertex
fn vs_main(@builtin(instance_index) instance_index: u32, @builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    var out: VertexOutput;

    let primitive = primitives.data[instance_index];
    let entity_loc = primitive.xy;
    var mesh_index = get_entity_primitive_mesh(entity_loc, primitive.z);

    let mesh = get_mesh_base(mesh_index, vertex_index);

    out.instance_index = instance_index;
    out.texcoord = mesh.texcoord0;

    let local_to_world = get_entity_mesh_to_world(entity_loc);

    out.world_position = local_to_world * vec4<f32>(mesh.position, 1.);
    let vertex_transformed = global_params.projection_view * out.world_position;

    out.position = vertex_transformed;

    let inv_local_to_world = inverse(local_to_world);
    out.inv_local_to_world_0 = inv_local_to_world[0];
    out.inv_local_to_world_1 = inv_local_to_world[1];
    out.inv_local_to_world_2 = inv_local_to_world[2];
    out.inv_local_to_world_3 = inv_local_to_world[3];

    return out;
}

struct Decal {
    depth: f32,
    material_in: MaterialInput
}

fn get_decal(in: VertexOutput) -> Decal {
    let screen_size = vec2<f32>(textureDimensions(solids_screen_depth));
    let screen_tc = screen_pixel_to_uv(in.position.xy, screen_size);
    let screen_ndc = screen_uv_to_ndc(screen_tc);
    let screen_depth = get_solids_screen_depth(screen_ndc);
    var res: Decal;
    res.depth = screen_depth;
    let world_position = project_point(global_params.inv_projection_view, vec3<f32>(screen_ndc.x, screen_ndc.y, screen_depth));

    let inv_local_to_world = mat4x4<f32>(
        in.inv_local_to_world_0,
        in.inv_local_to_world_1,
        in.inv_local_to_world_2,
        in.inv_local_to_world_3,
    );
    let local_pos = project_point(inv_local_to_world, world_position);
    if local_pos.x < -0.5 || local_pos.x > 0.5 || local_pos.y < -0.5 || local_pos.y > 0.5 {
        discard;
    }

    var material_in: MaterialInput;
    material_in.position = in.position;
    // Note: Decals assume we're using a unit cube
    material_in.texcoord.y = (0.5 - local_pos.x);
    material_in.texcoord.x = (local_pos.y + 0.5);
    material_in.world_position = world_position;
    let screen_normal_mat = mat3_from_quat(get_solids_screen_normal_quat(screen_ndc));
    material_in.normal = screen_normal_mat * vec3<f32>(0., 0., 1.);
    material_in.normal_matrix = screen_normal_mat;
    material_in.instance_index = in.instance_index;
    res.material_in = material_in;

    return res;
}

@fragment
fn fs_shadow_main(in: VertexOutput, @builtin(front_facing) is_front: bool) -> @builtin(frag_depth) f32 {
    let decal = get_decal(in);
    var material = get_material(decal.material_in);

    if material.opacity < material.alpha_cutoff {
        discard;
    }
    return decal.depth;
}

fn get_outline(instance_index: u32) -> vec4<f32> {
    let entity_loc = primitives.data[instance_index].xy;
    return get_entity_outline_or(entity_loc, vec4<f32>(0., 0., 0., 0.));
}

struct FsOutputs {
  @builtin(frag_depth) depth: f32,
  @location(0) color: vec4<f32>
}

@fragment
fn fs_forward_lit_main(in: VertexOutput, @builtin(front_facing) is_front: bool) -> FsOutputs {
    let decal = get_decal(in);
    var material = get_material(decal.material_in);

    if material.opacity < material.alpha_cutoff {
        discard;
    }
    var res: FsOutputs;
    res.color = shading(material, vec4<f32>(decal.material_in.world_position, 1.));
    res.depth = decal.depth + 0.0001;
    return res;
}

@fragment
fn fs_forward_unlit_main(in: VertexOutput, @builtin(front_facing) is_front: bool) -> FsOutputs {
    let decal = get_decal(in);
    var material = get_material(decal.material_in);

    if material.opacity < material.alpha_cutoff {
        discard;
    }
    var res: FsOutputs;
    res.color = vec4<f32>(material.base_color, material.opacity);
    res.depth = decal.depth + 0.0001;
    return res;
}

@fragment
fn fs_outlines_main(in: VertexOutput, @builtin(front_facing) is_front: bool) -> @location(0) vec4<f32> {
    let decal = get_decal(in);
    var material = get_material(decal.material_in);

    if material.opacity < material.alpha_cutoff {
        discard;
    }
    return get_outline(in.instance_index);
}
