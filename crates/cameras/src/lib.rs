use elements_core::{camera::*, get_window_scaled_resolution, transform::*, ui_scene};
use elements_ecs::{components, query_mut, SystemGroup, World};
use elements_element::{element_component, Element, Hooks};
use elements_std::shapes::BoundingBox;
use glam::{Mat4, Quat, Vec3};
use winit::event::Event;

use crate::{free::free_camera_system, spherical::spherical_camera_system};

pub mod free;
pub mod spherical;

components!("camera", {
    camera_movement_speed: f32,
    ui_camera: (),
});

pub fn init_all_components() {
    free::init_components();
    init_components();
    spherical::init_components();
}

pub fn assets_camera_systems() -> SystemGroup<Event<'static, ()>> {
    SystemGroup::new(
        "assets_camera_systems",
        vec![Box::new(free_camera_system()), Box::new(spherical_camera_system()), Box::new(ui_camera_system())],
    )
}

pub fn ui_camera_system() -> SystemGroup<Event<'static, ()>> {
    SystemGroup::new(
        "ui_camera_system",
        vec![query_mut((orthographic(), local_to_world()), (ui_camera(),)).to_system(|q, world, qs, _| {
            let window_size = get_window_scaled_resolution(world).as_vec2();
            for (_, (orth, ltw), (_,)) in q.iter(world, qs) {
                *ltw = Mat4::from_translation((window_size / 2.).extend(0.));
                orth.left = -window_size.x / 2.;
                orth.right = window_size.x / 2.;
                orth.top = -window_size.y / 2.;
                orth.bottom = window_size.y / 2.;
            }
        })],
    )
}

#[element_component]
pub fn UICamera(_: &mut World, _: &mut Hooks) -> Element {
    Element::new()
        .init_default(local_to_world())
        .init_default(inv_local_to_world())
        .init(near(), -1.)
        .init(far(), 1.0)
        .init(orthographic(), OrthographicRect { left: 0.0, right: 100., top: 0., bottom: 100. })
        .init_default(projection())
        .init_default(projection_view())
        .init_default(translation())
        .init_default(rotation())
        .init_default(ui_camera())
        .init_default(ui_scene())
}

#[element_component]
pub fn LookatCamera(_: &mut World, _: &mut Hooks, eye: Vec3, lookat: Vec3, up: Vec3) -> Element {
    Element::new()
        .init_default(local_to_world())
        .init_default(inv_local_to_world())
        .init(near(), 0.1)
        .init(fovy(), 1.0)
        .init(perspective_infinite_reverse(), ())
        .init(aspect_ratio(), 1.)
        .init(aspect_ratio_from_window(), ())
        .init_default(projection())
        .init_default(projection_view())
        .set(translation(), eye)
        .set(lookat_center(), lookat)
        .set(lookat_up(), up)
}

#[element_component]
pub fn FreeCamera(_: &mut World, _: &mut Hooks, position: Vec3, rotation: Quat) -> Element {
    Element::new()
        .init_default(local_to_world())
        .init_default(inv_local_to_world())
        .init(near(), 0.1)
        .init(fovy(), 1.0)
        .init(perspective_infinite_reverse(), ())
        .init(aspect_ratio(), 1.)
        .init(aspect_ratio_from_window(), ())
        .init_default(projection())
        .init_default(projection_view())
        .set(elements_core::transform::translation(), position)
        .set(elements_core::transform::rotation(), rotation)
}

#[element_component]
pub fn FittedOrthographicCamera(_: &mut World, _: &mut Hooks, eye: Vec3, lookat: Vec3, up: Vec3, fit: BoundingBox, aspect: f32) -> Element {
    Element::new().extend(Camera::fitted_ortographic(eye, lookat, up, fit, aspect).to_entity_data())
}