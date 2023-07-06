use ambient_api::{components::core::layout::space_between_items, prelude::*};

#[element_component]
fn App(hooks: &mut Hooks) -> Element {
    let (f32_value, set_f32_value) = hooks.use_state(0.);
    let (f32_exp_value, set_f32_exp_value) = hooks.use_state(0.1);
    let (i32_value, set_i32_value) = hooks.use_state(0);

    FocusRoot::el([FlowColumn::el([
        Slider {
            value: f32_value,
            on_change: Some(set_f32_value),
            min: 0.,
            max: 100.,
            width: 100.,
            logarithmic: false,
            round: Some(2),
            suffix: Some("%"),
        }
        .el(),
        Slider {
            value: f32_exp_value,
            on_change: Some(set_f32_exp_value),
            min: 0.1,
            max: 1000.,
            width: 100.,
            logarithmic: true,
            round: Some(2),
            suffix: None,
        }
        .el(),
        IntegerSlider {
            value: i32_value,
            on_change: Some(set_i32_value),
            min: 0,
            max: 100,
            width: 100.,
            logarithmic: false,
            suffix: None,
        }
        .el(),
    ])])
    .with(space_between_items(), STREET)
    .with_padding_even(STREET)
}

#[main]
pub fn main() {
    App.el().spawn_interactive();
}
