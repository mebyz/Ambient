use ambient_std::mesh::{generate_tangents, Mesh, MeshBuilder};
use glam::*;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct StarMesh {
    pub spikes_number: u32,
    pub inner_radius: f32,
    pub outer_radius: f32,
    pub width: f32,
}

impl Default for StarMesh {
    fn default() -> Self {
        Self {
            spikes_number: 5,
            inner_radius: 0.2,
            outer_radius: 1.0,
            width: 0.5,
        }
    }
}

impl From<StarMesh> for Mesh {
    fn from(star: StarMesh) -> Self {
        let mut vertices = Vec::new();
        let mut normals = Vec::new();
        let mut uvs = Vec::new();
        let mut indices = Vec::new();

        let spikes = star.spikes_number;
        let inner_radius = star.inner_radius;
        let outer_radius = star.outer_radius;
        let width = star.width;

        let sector_angle = 2.0 * std::f32::consts::PI / spikes as f32;

        // Generate vertices and normals for each spike
        for i in 0..spikes {
            let angle = i as f32 * sector_angle;

            // Inner spike vertex
            let inner_x = inner_radius * angle.cos();
            let inner_y = inner_radius * angle.sin();
            vertices.push(Vec3::new(inner_x, inner_y, 0.0));
            normals.push(Vec3::new(0.0, 0.0, 1.0));
            uvs.push(Vec2::new(inner_x, inner_y));

            // Outer spike vertex
            let outer_x = outer_radius * angle.cos();
            let outer_y = outer_radius * angle.sin();
            vertices.push(Vec3::new(outer_x, outer_y, 0.0));
            normals.push(Vec3::new(0.0, 0.0, 1.0));
            uvs.push(Vec2::new(outer_x, outer_y));

            // Indices for the current spike
            let index_base = i as u32 * 2;
            indices.push(index_base);
            indices.push(index_base + 1);
            indices.push(index_base + 2);
            indices.push(index_base + 1);
            indices.push(index_base + 3);
            indices.push(index_base + 2);

            // Generate vertices and normals for the width of the star
            if width > 0.0 {
                let width_offset = Vec3::new(width * angle.sin(), -width * angle.cos(), 0.0);

                // Inner width vertex
                vertices.push(Vec3::new(inner_x, inner_y, 0.0) + width_offset);
                normals.push(Vec3::new(0.0, 0.0, 1.0));
                uvs.push(Vec2::new(inner_x, inner_y));

                // Outer width vertex
                vertices.push(Vec3::new(outer_x, outer_y, 0.0) + width_offset);
                normals.push(Vec3::new(0.0, 0.0, 1.0));
                uvs.push(Vec2::new(outer_x, outer_y));

                // Indices for the width
                let width_index_base = i as u32 * 4 + 4;
                indices.push(index_base + 1);
                indices.push(width_index_base);
                indices.push(index_base + 3);
                indices.push(width_index_base);
                indices.push(width_index_base + 1);
                indices.push(index_base + 3);
            }
        }

        let tangents = generate_tangents(&vertices, &uvs, &indices);

        MeshBuilder {
            positions: vertices,
            normals,
            tangents,
            texcoords: vec![uvs],
            indices,
            ..MeshBuilder::default()
        }
        .build()
        .expect("Invalid star mesh")
    }
}