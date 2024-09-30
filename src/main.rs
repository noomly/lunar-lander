use std::f32::consts::PI;

use avian2d::prelude::*;
use bevy::{
    color::palettes::css,
    ecs::schedule::{LogLevel, ScheduleBuildSettings},
    prelude::*,
    sprite::{MaterialMesh2dBundle, Mesh2dHandle},
};
use bevy_egui::{egui, EguiContexts, EguiPlugin, EguiSettings};
use rand::{prelude::*, rngs::mock::StepRng};
use rand_chacha::ChaCha8Rng;
use utils::map_range;

mod terrain;
mod utils;

#[derive(Component)]
struct Spaceship;

#[derive(Component, PartialEq)]
enum Propulsor {
    Top,
    Bottom,
    TopLeft,
    TopRight,
    Left,
    Right,
}

#[derive(Component)]
struct Force(pub Vec2);

#[derive(Component)]
struct Planet;

#[derive(Component, Debug)]
struct Fuel(f32);

#[derive(Component, Clone)]
struct TransitionCameraToPlanet {
    planet: Entity,
    /// In seconds
    elapsed_time: f32,
    /// In seconds
    duration: f32,
    /// In radians
    starting_angle: f32,
}

#[derive(States, Debug, Clone, PartialEq, Eq, Hash)]
enum GameState {
    InGame,
    GameOver,
}

#[derive(SystemSet, Debug, Clone, PartialEq, Eq, Hash)]
struct GameplaySet;

#[derive(Resource)]
struct DebugSettings {
    invincible: bool,
    dont_consume_fuel: bool,
}

impl Default for DebugSettings {
    fn default() -> Self {
        Self {
            invincible: true,
            dont_consume_fuel: true,
        }
    }
}

fn gameover_ui(mut next_state: ResMut<NextState<GameState>>, mut contexts: EguiContexts) {
    egui::Window::new("Game Over").show(contexts.ctx_mut(), |ui| {
        if ui.button("restart").clicked() {
            next_state.set(GameState::InGame);
        }
    });
}

fn cleanup_spaceship(
    mut commands: Commands,
    mut collision_events: ResMut<Events<Collision>>,
    query: Query<Entity, With<Spaceship>>,
) {
    let entity = query.get_single().unwrap();

    collision_events.clear();
    commands.entity(entity).despawn_recursive();
}

fn main() {
    let mut app = App::new();

    // Enable ambiguity warnings
    app.edit_schedule(Update, |schedule| {
        schedule.set_build_settings(ScheduleBuildSettings {
            ambiguity_detection: LogLevel::Warn,
            ..default()
        });
    });
    app.edit_schedule(FixedUpdate, |schedule| {
        schedule.set_build_settings(ScheduleBuildSettings {
            ambiguity_detection: LogLevel::Warn,
            ..default()
        });
    });

    app.add_plugins(DefaultPlugins)
        .add_plugins((PhysicsPlugins::default(), PhysicsDebugPlugin::default()))
        .add_plugins(EguiPlugin)
        .insert_state(GameState::InGame)
        .init_resource::<DebugSettings>()
        .add_systems(Startup, (setup_graphics, setup_planets));

    app.configure_sets(Update, GameplaySet.run_if(in_state(GameState::InGame)));
    app.configure_sets(FixedUpdate, GameplaySet.run_if(in_state(GameState::InGame)));
    app.configure_sets(PostUpdate, GameplaySet.run_if(in_state(GameState::InGame)));

    app.add_systems(OnEnter(GameState::InGame), setup_spaceship);
    app.add_systems(
        FixedUpdate,
        (
            activate_propulsors,
            update_gravity2,
            spaceship_collision,
            spawn_fuel_orbs,
        )
            .chain()
            .in_set(GameplaySet),
    );
    app.add_systems(Update, (game_ui).in_set(GameplaySet));
    app.add_systems(
        PostUpdate,
        (
            update_camera
                .chain()
                .after(PhysicsSet::Sync)
                .before(TransformSystem::TransformPropagate),
            draw_compass.after(TransformSystem::TransformPropagate),
        )
            .in_set(GameplaySet),
    );

    app.add_systems(
        Update,
        gameover_ui
            .after(GameplaySet)
            .run_if(in_state(GameState::GameOver)),
    );
    app.add_systems(
        OnExit(GameState::GameOver),
        cleanup_spaceship
            .before(PhysicsSet::Sync)
            .before(PhysicsSet::StepSimulation),
    );

    app.run();
}

#[derive(Component)]
struct Compass;

fn draw_compass(
    mut gizmos: Gizmos,
    window_query: Query<&Window>,
    camera_query: Query<(&Camera, &GlobalTransform)>,
    ship_query: Query<&Transform, With<Spaceship>>,
    planets_query: Query<(Entity, &Transform), With<Planet>>,
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
    planets_query.iter().for_each(|(planet_entity, planet_t)| {
        let direction = (planet_t.translation.xy() - ship_t.translation.xy()).normalize();
        let angle = direction.y.atan2(direction.x) - camera_angle - std::f32::consts::PI * 2.;
        let angle = -angle;
        let planet_compass_pos = Vec2::new(angle.cos(), angle.sin()) * 66.;
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
    });
}

fn setup_graphics(mut commands: Commands, mut egui_settings: ResMut<EguiSettings>) {
    let mut camera = Camera2dBundle::default();
    camera.projection.scale *= 1.5;
    commands.spawn(camera);

    egui_settings.scale_factor *= 0.7;
}

#[derive(Component)]
struct FuelOrb;

fn spawn_fuel_orbs(
    mut commands: Commands,
    spatial_query: SpatialQuery,
    planet_query: Query<(Entity, &Transform), With<Planet>>,
    fuel_orbs_query: Query<(), With<FuelOrb>>,
) {
    let orbs_count = fuel_orbs_query.iter().count();
    if orbs_count > 10 {
        return;
    }

    for (entity, transform) in planet_query.iter() {
        let mut rng = ChaCha8Rng::seed_from_u64(entity.index() as u64);
        let planet_pos = transform.translation.xy();

        for _ in 0..10 {
            let angle = rng.gen_range(-PI..=PI);
            let direction = Vec2::new(angle.cos(), angle.sin());

            let hit = spatial_query
                .cast_ray(
                    planet_pos,
                    Dir2::new(direction).unwrap(),
                    500000.0,
                    false,
                    SpatialQueryFilter::default(),
                )
                .unwrap();

            let orb_radius = 20.;
            let hit = planet_pos + direction * (hit.time_of_impact + orb_radius * 3.);

            commands.spawn((
                FuelOrb,
                TransformBundle::from(Transform::from_xyz(hit.x, hit.y, 0.)),
                Collider::circle(orb_radius),
            ));

            //commands
            //    .get_entity(entity)
            //    .unwrap()
            //    .with_children(|parent| {
            //        parent.spawn((
            //            FuelOrb,
            //            TransformBundle::from(Transform::from_xyz(hit.x, hit.y, 0.)),
            //            Collider::circle(orb_radius),
            //        ));
            //    });
        }
    }
}

fn setup_planets(mut commands: Commands) {
    {
        let points = terrain::generate_points(0);
        let simplified = terrain::simplify_terrain(points);
        let collider = terrain::make_compound(simplified);

        commands.spawn((
            Planet,
            collider,
            RigidBody::Kinematic,
            //AngularVelocity(std::f32::consts::PI * 2. / (60. * 5.)),
            //AngularVelocity(2f32.to_radians()),
            ColliderDensity(0.),
            Restitution::new(0.),
            TransformBundle::from(Transform::from_xyz(0.0, 0.0, 0.0)),
        ));
    }

    {
        let points = terrain::generate_points(1);
        let simplified = terrain::simplify_terrain(points);
        let collider = terrain::make_compound(simplified);

        commands.spawn((
            Planet,
            collider,
            RigidBody::Kinematic,
            //AngularVelocity(std::f32::consts::PI * 2. / (60. * 5.)),
            //AngularVelocity(2f32.to_radians()),
            ColliderDensity(0.),
            Restitution::new(0.),
            TransformBundle::from(Transform::from_xyz(0.0, 3800.0, 0.0)),
        ));
    }
}

fn setup_spaceship(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<ColorMaterial>>,
) {
    let ship_size = Vec2::new(60., 35.);

    commands
        .spawn(Spaceship)
        .insert((
            RigidBody::Dynamic,
            Friction::new(10.).with_dynamic_coefficient(1.),
            Collider::rectangle(ship_size.x, ship_size.y),
            TransformBundle::from(
                Transform::from_xyz(0.0, 1800.0, 0.0)
                    .with_rotation(Quat::from_rotation_z(std::f32::consts::FRAC_PI_2)),
            ),
            Fuel(100.),
        ))
        .with_children(|parent| {
            let intensity = 200_000.;

            let propulsor_fire_size = 20.;
            let propulsor_fire_mesh = meshes.add(Triangle2d::new(
                Vec2::new(-1., -1.) * propulsor_fire_size / 2.,
                Vec2::new(0., 1.) * propulsor_fire_size / 2.,
                Vec2::new(1., -1.) * propulsor_fire_size / 2.,
            ));

            let pos = Vec2::new(ship_size.x / 2., 0.);
            parent.spawn(Propulsor::Top).insert((
                Position(pos.clone()),
                Force(-Vec2::X * intensity),
                MaterialMesh2dBundle {
                    mesh: Mesh2dHandle(propulsor_fire_mesh.clone()),
                    material: materials.add(Color::srgb(0.8, 0.2, 0.2)),
                    transform: Transform::from_xyz(pos.x + propulsor_fire_size / 2., pos.y, 0.)
                        .with_rotation(Quat::from_rotation_z(-std::f32::consts::FRAC_PI_2)),
                    visibility: Visibility::Hidden,
                    ..default()
                },
            ));

            let pos = Vec2::new(-ship_size.x / 2., 0.);
            parent.spawn(Propulsor::Bottom).insert((
                Position(pos.clone()),
                Force(Vec2::X * intensity),
                MaterialMesh2dBundle {
                    mesh: Mesh2dHandle(propulsor_fire_mesh.clone()),
                    material: materials.add(Color::srgb(0.8, 0.2, 0.2)),
                    transform: Transform::from_xyz(pos.x - propulsor_fire_size / 2., pos.y, 0.)
                        .with_rotation(Quat::from_rotation_z(PI / 2.)),
                    visibility: Visibility::Hidden,
                    ..default()
                },
            ));

            let pos = Vec2::new(0., -ship_size.y / 2.);
            parent.spawn(Propulsor::Left).insert((
                Position(pos.clone()),
                Force(Vec2::new(0., 1. * (intensity - intensity * 0.1))),
                MaterialMesh2dBundle {
                    mesh: Mesh2dHandle(propulsor_fire_mesh.clone()),
                    material: materials.add(Color::srgb(0.8, 0.2, 0.2)),
                    transform: Transform::from_xyz(pos.x, pos.y - propulsor_fire_size / 2., 0.)
                        .with_rotation(Quat::from_rotation_z(PI)),
                    visibility: Visibility::Hidden,
                    ..default()
                },
            ));

            let pos = Vec2::new(0., ship_size.y / 2.);
            parent.spawn(Propulsor::Right).insert((
                Position(pos.clone()),
                Force(Vec2::new(0., -1. * (intensity - intensity * 0.1))),
                MaterialMesh2dBundle {
                    mesh: Mesh2dHandle(propulsor_fire_mesh.clone()),
                    material: materials.add(Color::srgb(0.8, 0.2, 0.2)),
                    transform: Transform::from_xyz(pos.x, pos.y + propulsor_fire_size / 2., 0.)
                        .with_rotation(Quat::from_rotation_z(PI * 2.)),
                    visibility: Visibility::Hidden,
                    ..default()
                },
            ));

            let pos = Vec2::new(ship_size.x / 3., -ship_size.y / 2.);
            parent.spawn(Propulsor::TopLeft).insert((
                Position(pos.clone()),
                Force(Vec2::new(0., 1. * (intensity - intensity * 0.1))),
                MaterialMesh2dBundle {
                    mesh: Mesh2dHandle(propulsor_fire_mesh.clone()),
                    material: materials.add(Color::srgb(0.8, 0.2, 0.2)),
                    transform: Transform::from_xyz(pos.x, pos.y - propulsor_fire_size / 2., 0.)
                        .with_rotation(Quat::from_rotation_z(PI)),
                    visibility: Visibility::Hidden,
                    ..default()
                },
            ));

            let pos = Vec2::new(ship_size.x / 3., ship_size.y / 2.);
            parent.spawn(Propulsor::TopRight).insert((
                Position(pos.clone()),
                Force(Vec2::new(0., -1. * (intensity - intensity * 0.1))),
                MaterialMesh2dBundle {
                    mesh: Mesh2dHandle(propulsor_fire_mesh.clone()),
                    material: materials.add(Color::srgb(0.8, 0.2, 0.2)),
                    transform: Transform::from_xyz(pos.x, pos.y + propulsor_fire_size / 2., 0.)
                        .with_rotation(Quat::from_rotation_z(PI * 2.)),
                    visibility: Visibility::Hidden,
                    ..default()
                },
            ));
        });
}

fn apply_quat(vector: Vec2, quat: Quat) -> Vec2 {
    (quat * Vec3::new(vector.x, vector.y, 0.)).xy()
}

fn game_ui(
    mut contexts: EguiContexts,
    mut debug_settings: ResMut<DebugSettings>,
    mut query: Query<&mut Fuel>,
) {
    let mut fuel = query.get_single_mut().unwrap();

    egui::Window::new("Lunar Lander")
        .max_size(egui::vec2(0., 0.))
        .show(contexts.ctx_mut(), |ui| {
            egui::Grid::new("info")
                .num_columns(2)
                .striped(true)
                .show(ui, |ui| {
                    ui.label(egui::RichText::new("Game Info").strong());
                    ui.end_row();

                    ui.label("fuel:");
                    ui.add(egui::ProgressBar::new(fuel.0 / 100.).show_percentage());
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
                        fuel.0 = 100.;
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

fn update_planet_movement(
    time: Res<Time>,
    mut query: Query<(&mut Transform, &AngularVelocity), With<Planet>>,
    ship_query: Query<
        (&Transform, &LinearVelocity, &AngularVelocity),
        (With<Spaceship>, Without<Planet>),
    >,
) {
    let (planet_t, planet_v) = query.get_single_mut().unwrap();

    let (ship_t, ship_lv, ship_av) = ship_query.get_single().unwrap();

    //println!("{:>10?} {:>10?}", ship_t.translation.xy(), ship_lv.xy());
    //println!("{:?}", ship_t.translation.xy());

    //println!("{:?<10} {:?<10}", planet_v.0, ship_v.0);

    //let direction = ship_t.translation.xy() - planet_t.translation.xy();
    //let rotated_lv = Mat2::from_angle(direction.y.atan2(direction.x)) * ship_lv.0;
    //println!("{:?}", ship_lv.0);
    //println!("{:?}", rotated_lv);

    //t.rotate_local_z((5. * time.delta_seconds()).to_radians());
}

fn update_camera(
    mut commands: Commands,
    time: Res<Time<Fixed>>,
    mut cam_query: Query<
        (
            Entity,
            &mut Transform,
            Option<&mut TransitionCameraToPlanet>,
        ),
        (With<Camera>, Without<Spaceship>, Without<Planet>),
    >,
    ship_query: Query<&Transform, (With<Spaceship>, Without<Camera>, Without<Planet>)>,
    planets_query: Query<(Entity, &Transform), (With<Planet>, Without<Camera>, Without<Spaceship>)>,
) {
    let (camera_entity, mut cam_t, transition) = cam_query.get_single_mut().unwrap();
    let ship_t = ship_query.get_single().unwrap();
    let ship_pos = ship_t.translation.xy();
    let (planet_closest_entity, planet_t) = planets_query
        .into_iter()
        .sort_by::<&Transform>(|planet1_t, planet2_t| {
            planet1_t
                .translation
                .xy()
                .distance(ship_pos)
                .total_cmp(&planet2_t.translation.xy().distance(ship_pos))
        })
        .next()
        .unwrap();

    cam_t.translation.x = ship_t.translation.x;
    cam_t.translation.y = ship_t.translation.y;

    let planet_pos = planet_t.translation.xy();
    let direction = (planet_pos - ship_t.translation.xy()).normalize();
    let optimal_angle = direction.y.atan2(direction.x) + std::f32::consts::FRAC_PI_2;
    // Normalize optimal_angle to be between -PI and +PI
    let optimal_angle = (optimal_angle + PI).rem_euclid(2. * PI) - PI;

    let current_angle = cam_t.rotation.to_euler(EulerRot::XYZ).2;

    let current_angle_diff = optimal_angle - current_angle;
    let linear_angle_step = PI / 8.;

    let transition = match transition {
        Some(mut transition) => {
            if transition.planet != planet_closest_entity {
                (*transition).planet = planet_closest_entity;
                (*transition).elapsed_time = 0.;
                (*transition).starting_angle = current_angle;
            } else {
                (*transition).elapsed_time += time.delta_seconds();
            }

            Some(transition.clone())
        }
        None => {
            if current_angle_diff.abs() > linear_angle_step {
                let transition = TransitionCameraToPlanet {
                    planet: planet_closest_entity,
                    elapsed_time: 0.,
                    duration: 1.5,
                    starting_angle: current_angle,
                };

                commands.entity(camera_entity).insert(transition.clone());

                Some(transition)
            } else {
                None
            }
        }
    };

    match transition {
        Some(transition) => {
            let dt = (transition.elapsed_time / transition.duration).clamp(0., 1.);

            let angle_diff = optimal_angle - transition.starting_angle;
            let angle_diff_norm = (angle_diff + PI).rem_euclid(PI * 2.) - PI;

            let new_angle = transition.starting_angle + angle_diff_norm * ease_in_out_sine(dt);

            cam_t.rotation = Quat::from_rotation_z(new_angle);

            if transition.elapsed_time >= transition.duration {
                commands
                    .entity(camera_entity)
                    .remove::<TransitionCameraToPlanet>();
            }
        }
        None => {
            let angle_step_dt = linear_angle_step * time.delta_seconds();

            let new_angle = if current_angle_diff.abs() > angle_step_dt {
                let angle_diff_norm = (current_angle_diff + PI).rem_euclid(PI * 2.) - PI;
                let optimal_direction = angle_diff_norm.signum();
                let new_angle = current_angle + angle_step_dt * optimal_direction;

                new_angle
            } else {
                optimal_angle
            };

            cam_t.rotation = Quat::from_rotation_z(new_angle);
        }
    }
}

fn ease_in_out_sine(t: f32) -> f32 {
    -(0.5 * ((t * std::f32::consts::PI).cos() - 1.0))
}

fn update_gravity2(
    mut gravity: ResMut<Gravity>,
    planets_query: Query<&Transform, (With<Planet>, Without<Spaceship>)>,
    ship_query: Query<&Transform, (With<Spaceship>, Without<Planet>)>,
) {
    let ship_t = ship_query.get_single().unwrap();
    let ship_pos = ship_t.translation.xy();

    let new_gravity = planets_query
        .iter()
        .fold(Vec2::ZERO, |total_gravity, planet_t| {
            let planet_pos = planet_t.translation.xy();
            let direction = (planet_pos - ship_pos).normalize();
            let distance = planet_pos.distance(ship_pos);
            let dist_min = 1650.;
            let dist_max = 2200.;

            let base_gravity = 60. * direction;
            let gravity = base_gravity * map_range(dist_min, dist_max, 1., 0., distance);

            total_gravity + gravity
        });

    gravity.0 = new_gravity;
}

fn update_gravity(
    mut gravity: ResMut<Gravity>,
    planets_query: Query<&Transform, (With<Planet>, Without<Spaceship>)>,
    ship_query: Query<&Transform, (With<Spaceship>, Without<Planet>)>,
) {
    let ship_t = ship_query.get_single().unwrap();
    let ship_pos = ship_t.translation.xy();
    let planet_t = planets_query
        .into_iter()
        .sort_by::<&Transform>(|planet1_t, planet2_t| {
            planet1_t
                .translation
                .xy()
                .distance(ship_pos)
                .total_cmp(&planet2_t.translation.xy().distance(ship_pos))
        })
        .next()
        .unwrap();

    let ship_pos = ship_t.translation.xy();
    let planet_pos = planet_t.translation.xy();

    let base_gravity = Vec2::Y * 60.;
    let direction = (planet_pos - ship_pos).normalize();
    let angle = direction.y.atan2(direction.x);
    let rotated_gravity = apply_quat(
        base_gravity,
        Quat::from_rotation_z(angle - std::f32::consts::FRAC_PI_2),
    );

    gravity.0 = rotated_gravity;
}

fn spaceship_collision(
    mut commands: Commands,
    time: Res<Time<Substeps>>,
    debug_settings: Res<DebugSettings>,
    mut next_state: ResMut<NextState<GameState>>,
    mut collision_event_reader: EventReader<Collision>,
    mut bodies_query: Query<(Entity, Option<&mut Fuel>), Or<(With<FuelOrb>, With<Spaceship>)>>,
) {
    for Collision(contacts) in collision_event_reader.read() {
        let dt = time.delta_seconds();

        let bodies = bodies_query.get_many_mut([contacts.entity1, contacts.entity2]);
        if let Ok([(entity1, fuel1), (entity2, fuel2)]) = bodies {
            println!("{:?}", [(entity1, &fuel1), (entity2, &fuel2)]);
            // Check for Fuel Orbs collision
            let orb = fuel1.as_ref().map_or(entity1, |_| entity2);
            let mut ship_fuel = fuel1.or(fuel2).unwrap();

            commands.entity(orb).despawn_recursive();
            ship_fuel.0 = (ship_fuel.0 + 40.).min(100.);
        } else {
            // Check for physical collision
            let total_impact_force =
                contacts.total_normal_force(dt) + contacts.total_tangent_force(dt);

            if total_impact_force > 15_000_000. && !debug_settings.invincible {
                next_state.set(GameState::GameOver);
            }
        }
    }
}

fn activate_propulsors(
    mut commands: Commands,
    debug_settings: Res<DebugSettings>,
    keyboard_input: Res<ButtonInput<KeyCode>>,
    time: Res<Time>,
    mut ship_query: Query<
        (
            Entity,
            &Transform,
            Option<&mut AngularVelocity>,
            Option<&mut LinearVelocity>,
            &mut Fuel,
        ),
        With<Spaceship>,
    >,
    mut propulsors_query: Query<
        (&Propulsor, &Transform, &Force, &mut Visibility),
        Without<Spaceship>,
    >,
) {
    let (entity, t, mut av, mut lv, mut fuel) = ship_query.get_single_mut().unwrap();
    //let (entity, t, mut fuel) = ship_query.get_single_mut().unwrap();
    let mut propulsors = propulsors_query.iter_mut().collect::<Vec<_>>();

    let get_force = |local_point: Vec2, local_force: Vec2| {
        let rotated_point = apply_quat(local_point, t.rotation);
        let rotated_force = apply_quat(local_force, t.rotation);

        return (rotated_force, rotated_point);
    };

    let mut active_propulsors = Vec::new();

    //if keyboard_input.pressed(KeyCode::KeyA) {
    //av.0 = 0.;
    //lv.0 = Vec2::ZERO;
    //}

    if keyboard_input.pressed(KeyCode::KeyK) {
        active_propulsors.push(Propulsor::Top);
    }

    if keyboard_input.pressed(KeyCode::KeyI) {
        active_propulsors.push(Propulsor::Bottom);
    }

    if keyboard_input.pressed(KeyCode::KeyJ) {
        active_propulsors.push(Propulsor::Left);
    }

    if keyboard_input.pressed(KeyCode::KeyL) {
        active_propulsors.push(Propulsor::Right);
    }

    if keyboard_input.pressed(KeyCode::KeyU) {
        active_propulsors.push(Propulsor::TopLeft);
    }

    if keyboard_input.pressed(KeyCode::KeyO) {
        active_propulsors.push(Propulsor::TopRight);
    }

    let mut forces = Vec::new();

    for (propulsor, transform, force, visibility) in propulsors.iter_mut() {
        if fuel.0 > 0. && active_propulsors.contains(propulsor) {
            **visibility = Visibility::Visible;
            forces.push(get_force(transform.translation.xy(), force.0));
        } else {
            **visibility = Visibility::Hidden;
        }
    }

    if forces.len() > 0 {
        let mut external_force = ExternalForce::default().with_persistence(false);
        forces.into_iter().for_each(|(force, point)| {
            external_force.apply_force_at_point(force, point, Vec2::ZERO);
        });
        if !debug_settings.dont_consume_fuel {
            fuel.0 = fuel.0 - (100. / 20.) * time.delta_seconds();
        }
        //println!("force: {}", external_force.length());
        commands.entity(entity).insert(external_force);
    }
}
