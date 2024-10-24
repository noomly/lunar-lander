use avian2d::prelude::LinearVelocity;
use bevy::{color::palettes::css, prelude::*};
use bevy_egui::{
    egui::{self, Color32},
    EguiContexts,
};
use rand::prelude::*;
use rand_chacha::ChaCha8Rng;

use crate::{
    utils::map_range, DebugSettings, Focused, Fuel, LandingTimeLimit, Life, Planet, Spaceship,
};

pub fn draw_velocity(
    mut gizmos: Gizmos,
    window_query: Query<&Window>,
    camera_query: Query<(&Camera, &GlobalTransform)>,
    ship_query: Query<Option<&LinearVelocity>, With<Spaceship>>,
) {
    let window = window_query.get_single().unwrap();
    let (camera, camera_t) = camera_query.get_single().unwrap();
    let lv = ship_query.get_single().unwrap();

    let window_half_extents = Vec2::new(window.width() / 2., window.height() / 2.);

    let compass_world_pos = camera
        .viewport_to_world_2d(camera_t, window_half_extents)
        .unwrap();

    if let Some(lv) = lv {
        let speed = lv.xy().length();
        let color = if speed > 150. {
            Some(css::RED)
        } else if speed > 30. {
            Some(css::BLUE)
        } else {
            None
        };

        if let Some(color) = color {
            gizmos.arrow_2d(
                compass_world_pos,
                compass_world_pos + lv.xy().normalize() * 100.,
                color,
            );
        }
    }
}

pub fn draw_compass(
    mut gizmos: Gizmos,
    window_query: Query<&Window>,
    camera_query: Query<(&Camera, &GlobalTransform)>,
    ship_query: Query<&Transform, With<Spaceship>>,
    planets_query: Query<(Entity, &Transform, Option<&Focused>), With<Planet>>,
) {
    let window = window_query.get_single().unwrap();
    let (camera, camera_t) = camera_query.get_single().unwrap();
    let ship_t = ship_query.get_single().unwrap();

    let camera_angle = {
        let (_, rotation, _) = camera_t.to_scale_rotation_translation();
        rotation.to_euler(EulerRot::XYZ).2
    };
    let window_half_extents = Vec2::new(window.width() / 2., window.height() / 2.);

    let compass_world_pos = camera
        .viewport_to_world_2d(camera_t, window_half_extents)
        .unwrap();

    let compass_radius = 100.;

    gizmos.circle_2d(compass_world_pos, compass_radius, css::LIGHT_GRAY);

    //vec![planets_query.iter().next().unwrap()]
    planets_query
        .iter()
        .for_each(|(planet_entity, planet_t, focused)| {
            let direction = (planet_t.translation.xy() - ship_t.translation.xy()).normalize();
            let angle = direction.y.atan2(direction.x) - camera_angle - std::f32::consts::PI * 2.;
            let angle = -angle;
            let planet_compass_pos = Vec2::new(angle.cos(), angle.sin()) * 65.;
            let planet_vp_pos = window_half_extents + planet_compass_pos;
            let planet_world_pos = camera
                .viewport_to_world_2d(camera_t, planet_vp_pos)
                .unwrap();

            let mut rng = ChaCha8Rng::seed_from_u64(planet_entity.index() as u64);
            let planet_color = Color::hsl(rng.gen_range(0f32..=360.), 0.5, 0.7);

            //let radius = {
            //    let (dist_min, dist_max) = (1500., 4500.);
            //
            //    let distance = planet_t
            //        .translation
            //        .xy()
            //        .distance(ship_t.translation.xy())
            //        .max(dist_min)
            //        .min(dist_max);
            //
            //    let (radius_min, radius_max) = (15., 6.);
            //
            //    let radius = (distance - dist_min) / (dist_max - dist_min) * (radius_max - radius_min)
            //        + radius_min;
            //
            //    radius
            //};

            let (dist_min, dist_max) = (1500., 4500.);
            let distance = planet_t
                .translation
                .xy()
                .distance(ship_t.translation.xy())
                .max(dist_min)
                .min(dist_max);
            let radius = map_range(dist_min, dist_max, 15., 6., distance);

            gizmos.circle_2d(planet_world_pos, radius, planet_color);

            if let Some(_) = focused {
                gizmos.rect_2d(
                    planet_world_pos,
                    std::f32::consts::FRAC_PI_4 + camera_angle,
                    Vec2::new(30., 30.),
                    css::RED,
                );
            }
        });
}

pub fn game_ui(
    mut contexts: EguiContexts,
    mut debug_settings: ResMut<DebugSettings>,
    mut ship_query: Query<(&Life, &mut Fuel, &Transform, Option<&LinearVelocity>)>,
    planet_query: Query<(Entity, &Transform), (With<Planet>, With<Focused>)>,
    landing_query: Query<&LandingTimeLimit>,
) {
    let (ship_life, mut ship_fuel, ship_t, ship_vel) = ship_query.get_single_mut().unwrap();
    let planet = planet_query.get_single();
    let landing_timer = landing_query.get_single();

    egui::Window::new("Lunar Lander")
        .max_size(egui::vec2(0., 0.))
        .show(contexts.ctx_mut(), |ui| {
            egui::Grid::new("info")
                .num_columns(2)
                .striped(true)
                .show(ui, |ui| {
                    ui.label(egui::RichText::new("Game Info").strong());
                    ui.end_row();

                    ui.label("life:");
                    ui.add(egui::ProgressBar::new(ship_life.0 / 5.).show_percentage());
                    ui.end_row();

                    ui.label("fuel:");
                    ui.add(egui::ProgressBar::new(ship_fuel.0 / 100.).show_percentage());
                    ui.end_row();

                    let landing_time = landing_timer
                        .ok()
                        .map_or("none".to_string(), |landing_timer| {
                            format!("{}s", landing_timer.remaining_secs().trunc())
                        });

                    ui.label("landing time left:");
                    ui.colored_label(Color32::WHITE, landing_time);
                    ui.end_row();

                    let (index, distance, speed) = planet.ok().map_or(
                        ("none".to_string(), 0., 0.),
                        |(planet_entity, planet_t)| {
                            (
                                planet_entity.index().to_string(),
                                ship_t.translation.distance(planet_t.translation),
                                ship_vel.map_or(0., |vel| vel.xy().length()),
                            )
                        },
                    );

                    ui.label("focused planet:");
                    ui.colored_label(Color32::WHITE, index);
                    ui.end_row();

                    ui.label("distance from planet:");
                    ui.colored_label(Color32::WHITE, format!("{}m", distance.trunc()));
                    ui.end_row();

                    ui.label("speed:");
                    ui.colored_label(Color32::WHITE, format!("{}m/s", speed.trunc()));
                    ui.end_row();
                });

            ui.add_space(10.);
            ui.separator();
            ui.add_space(10.);

            egui::Grid::new("debug")
                .num_columns(2)
                .striped(true)
                .show(ui, |ui| {
                    ui.label(egui::RichText::new("Debug Settings").strong());
                    ui.end_row();

                    ui.label("refill fuel:");
                    if ui.button("refill").clicked() {
                        ship_fuel.0 = 100.;
                    }
                    ui.end_row();

                    ui.label("don't consume fuel:");
                    ui.checkbox(&mut debug_settings.dont_consume_fuel, "");
                    ui.end_row();

                    ui.label("invincible:");
                    ui.checkbox(&mut debug_settings.invincible, "");
                    ui.end_row();
                });
        });
}
