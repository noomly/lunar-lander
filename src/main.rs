use std::{f32::consts::PI, time::Duration};

use avian2d::prelude::*;
use bevy::{
    color::palettes::css,
    ecs::schedule::{LogLevel, ScheduleBuildSettings},
    input::common_conditions::input_just_pressed,
    prelude::*,
    sprite::{MaterialMesh2dBundle, Mesh2dHandle},
};
use bevy_egui::{egui, EguiContexts, EguiPlugin, EguiSettings};
use rand::{prelude::*, rngs::mock::StepRng};
use rand_chacha::ChaCha8Rng;
use utils::map_range;

mod terrain;
mod ui;
mod utils;

#[derive(Component, Debug)]
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

#[derive(Component, Deref, DerefMut)]
struct Terrain(Vec<Vec2>);

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

fn cleanup<T: Component>(mut commands: Commands, query: Query<Entity, With<T>>) {
    query.iter().for_each(|entity| {
        commands.entity(entity).despawn_recursive();
    })
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

#[derive(Component)]
struct DebugPermanentGizmo;

fn main() {
    let mut app = App::new();

    app.add_event::<LandedEvent>();
    app.add_event::<FireAtEvent>();

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
        .init_resource::<DebugSettings>();

    app.add_systems(Startup, setup_graphics);

    app.configure_sets(Update, GameplaySet.run_if(in_state(GameState::InGame)));
    app.configure_sets(FixedUpdate, GameplaySet.run_if(in_state(GameState::InGame)));
    app.configure_sets(PostUpdate, GameplaySet.run_if(in_state(GameState::InGame)));

    app.add_systems(
        OnEnter(GameState::InGame),
        (setup_spaceship, setup_planets, setup_spots).chain(),
    );

    app.add_systems(
        FixedUpdate,
        (cleanup::<LandingSpot>, setup_spots)
            .chain()
            .run_if(on_event::<LandedEvent>()),
    );

    app.add_systems(
        FixedUpdate,
        (
            spawn_fuel_orbs,
            activate_propulsors,
            update_gravity2,
            update_focused_planet,
            update_landing_time_limit,
            update_landing,
        )
            .chain()
            .in_set(GameplaySet),
    );

    app.add_systems(
        Update,
        |mut commands: Commands,
         mut collision_event_reader: EventReader<Collision>,
         entity_query: Query<(Entity, &Transform)>| {
            for Collision(contacts) in collision_event_reader.read() {
                let (e1_id, e1_t) = entity_query.get(contacts.entity1).unwrap();
                let (e2_id, e2_t) = entity_query.get(contacts.entity2).unwrap();

                let e1_id = e1_id.index();
                let e2_id = e2_id.index();
                if e1_id == 8 && e2_id == 15 || e1_id == 32 && e2_id == 37 {
                    continue;
                }

                contacts.manifolds.iter().for_each(|manifold| {
                    manifold.contacts.iter().for_each(|contacts| {
                        let point1 = contacts
                            .global_point1(&e1_t.translation.xy().into(), &e1_t.rotation.into());

                        commands.spawn((
                            DebugPermanentGizmo,
                            TransformBundle::from(Transform::from_xyz(point1.x, point1.y, 0.)),
                        ));
                    });
                });

                println!(
                    "Entities {:?} and {:?} are colliding",
                    contacts.entity1, contacts.entity2,
                );
            }
        },
    );
    app.add_systems(
        Update,
        |mut commands: Commands,
         mut debug_query: Query<&Transform, With<DebugPermanentGizmo>>,
         mut gizmos: Gizmos,
         window_query: Query<&Window>,
         camera_query: Query<(&Camera, &GlobalTransform)>| {
            let window = window_query.get_single().unwrap();
            let (camera, camera_t) = camera_query.get_single().unwrap();

            debug_query.iter().for_each(|t| {
                let camera_angle = {
                    let (_, rotation, _) = camera_t.to_scale_rotation_translation();
                    rotation.to_euler(EulerRot::XYZ).2
                };
                let window_half_extents = Vec2::new(window.width() / 2., window.height() / 2.);

                //let compass_world_pos = camera
                //    .viewport_to_world_2d(camera_t, window_half_extents)
                //    .unwrap();

                let compass_radius = 10.;

                gizmos.circle_2d(t.translation.xy(), compass_radius, css::RED);
            });
        },
    );

    app.add_systems(
        Update,
        (
            ui::game_ui,
            focus_planet,
            spaceship_collisions,
            projectile_collisions,
            unalive,
            //spawn_turrets,
            fire_spaceship.run_if(input_just_pressed(KeyCode::Space)),
            fire_projectile.run_if(on_event::<FireAtEvent>()),
            fire_turrets,
        )
            .chain()
            .in_set(GameplaySet),
    );

    app.add_systems(
        PostUpdate,
        (
            update_camera
                .chain()
                .after(PhysicsSet::Sync)
                .before(TransformSystem::TransformPropagate),
            (ui::draw_compass, ui::draw_velocity)
                .chain()
                .after(TransformSystem::TransformPropagate),
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
        (
            cleanup_spaceship
                .before(PhysicsSet::Sync)
                .before(PhysicsSet::StepSimulation),
            cleanup::<Planet>,
            cleanup::<FuelOrb>,
            cleanup::<LandingSpot>,
            cleanup::<Turret>,
        )
            .chain(),
    );

    app.run();
}

fn setup_graphics(mut commands: Commands, mut egui_settings: ResMut<EguiSettings>) {
    let mut camera = Camera2dBundle::default();
    camera.projection.scale *= 1.5;
    commands.spawn(camera);

    egui_settings.scale_factor *= 0.7;
}

fn unalive(
    mut commands: Commands,
    mut next_state: ResMut<NextState<GameState>>,
    alive_query: Query<(Entity, &Life)>,
    ship_query: Query<Entity, With<Spaceship>>,
) {
    alive_query.iter().for_each(|(alive_id, alive_life)| {
        if alive_life.0 <= 0. {
            let is_ship = ship_query.get(alive_id).is_ok();
            if is_ship {
                next_state.set(GameState::GameOver);
            } else {
                commands.get_entity(alive_id).unwrap().despawn_recursive();
            }
        }
    });
}

#[derive(Component, Reflect, Debug)]
struct Turret;

#[derive(Component, Deref, DerefMut, Reflect, Debug)]
struct TurretCooldown(Timer);

fn spawn_turrets(
    mut commands: Commands,
    spatial_query: SpatialQuery,
    planet_query: Query<(Entity, &Transform), Added<Planet>>,
) {
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

            let turret_size = 75.;
            let hit = planet_pos + direction * (hit.time_of_impact + turret_size);

            commands.spawn((
                Turret,
                TurretCooldown(Timer::from_seconds(5., TimerMode::Repeating)),
                Life(1.),
                TransformBundle::from(Transform::from_xyz(hit.x, hit.y, 0.)),
                Collider::rectangle(turret_size, turret_size),
            ));
        }
    }
}

#[derive(Event, Reflect, Debug)]
struct FireAtEvent {
    shooter: Entity,
    velocity: Vec2,
}

fn fire_projectile(
    mut commands: Commands,
    mut fire_at_er: EventReader<FireAtEvent>,
    shooter_query: Query<(Entity, &Transform, Option<&LinearVelocity>)>,
) {
    fire_at_er.read().for_each(|fire_at| {
        if let Ok((_shooter_id, shooter_t, shooter_lv)) = shooter_query.get(fire_at.shooter) {
            let shooter_pos = shooter_t.translation.xy() + fire_at.velocity.normalize() * 60.;
            let projectile_velocity =
                shooter_lv.unwrap_or(&LinearVelocity::default()).xy() + fire_at.velocity;

            commands.spawn((
                Projectile(1.),
                RigidBody::Dynamic,
                Collider::rectangle(35., 1.),
                TransformBundle::from(
                    Transform::from_xyz(shooter_pos.x, shooter_pos.y, 0.)
                        .with_rotation(Quat::from_rotation_z(fire_at.velocity.to_angle())),
                ),
                LinearVelocity(projectile_velocity),
            ));
        }
    });
}

fn fire_turrets(
    timer: Res<Time<Fixed>>,
    mut fire_at_ew: EventWriter<FireAtEvent>,
    spatial_query: SpatialQuery,
    ship_query: Query<(Entity, &Transform), With<Spaceship>>,
    mut turrets_query: Query<(Entity, &Transform, &mut TurretCooldown), With<Turret>>,
) {
    let (_ship_id, ship_t) = ship_query.get_single().unwrap();
    let ship_pos = ship_t.translation;

    spatial_query
        .shape_intersections(
            &Collider::circle(700.),
            ship_pos.xy(),
            0.,
            SpatialQueryFilter::default(),
        )
        .into_iter()
        .for_each(|entity| {
            let (turret_id, turret_t, mut turret_timer) =
                if let Ok(turret) = turrets_query.get_mut(entity) {
                    turret
                } else {
                    return;
                };

            if !turret_timer.tick(timer.delta()).finished() {
                return;
            }

            fire_at_ew.send(FireAtEvent {
                shooter: turret_id,
                velocity: (ship_pos.xy() - turret_t.translation.xy()).normalize() * 1000.,
            });
        });
}

#[derive(Component, Debug)]
struct FuelOrb;

fn spawn_fuel_orbs(
    mut commands: Commands,
    spatial_query: SpatialQuery,
    planet_query: Query<(Entity, &Transform), With<Planet>>,
    fuel_orbs_query: Query<(), With<FuelOrb>>,
) {
    let orbs_count = fuel_orbs_query.iter().count();
    if orbs_count >= 10 {
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

#[derive(Component)]
struct LandingSpot;

#[derive(Component, Deref, DerefMut)]
struct LandingTimeLimit(Timer);

#[derive(Component)]
struct LandingShipPosition {
    last_ship_pos: Transform,
    secured_for: Timer,
}

#[derive(Component, Deref, DerefMut)]
struct LandingTimer(Timer);

#[derive(Event)]
struct LandedEvent;

fn update_landing_time_limit(
    time: Res<Time>,
    mut next_state: ResMut<NextState<GameState>>,
    mut query: Query<&mut LandingTimeLimit>,
) {
    let mut timer = query.get_single_mut().unwrap();

    if timer.finished() {
        next_state.set(GameState::GameOver);
    } else {
        timer.tick(time.delta());
    }
}

fn update_landing(
    mut commands: Commands,
    time: Res<Time>,
    mut landed_event_writer: EventWriter<LandedEvent>,
    ship_query: Query<(Entity, &AngularVelocity), With<Spaceship>>,
    mut landing_spot_query: Query<
        (
            Entity,
            Option<&CollidingEntities>,
            Option<&mut LandingTimer>,
        ),
        With<LandingSpot>,
    >,
) {
    let (spot_id, colliding, landing_timer) = landing_spot_query.get_single_mut().unwrap();
    let colliding = if let Some(colliding) = colliding {
        colliding
    } else {
        return;
    };

    let mut spot_entity = commands.get_entity(spot_id).unwrap();

    let (ship_id, ship_av) = ship_query.get_single().unwrap();

    let colliding_with_spaceship = colliding
        .iter()
        .find(|entity| **entity == ship_id)
        .is_some();

    let max_av = 0.1;

    match (colliding_with_spaceship, landing_timer) {
        (true, None) if ship_av.0 < max_av => {
            spot_entity.insert(LandingTimer(Timer::from_seconds(3., TimerMode::Once)));
        }
        (true, Some(mut timer)) if ship_av.0 < max_av => {
            if timer.finished() {
                //spot_entity.despawn_recursive();
                landed_event_writer.send(LandedEvent);
                println!("landed");
            } else {
                timer.tick(time.delta());
            }
        }
        (true, Some(_)) if ship_av.0 >= max_av => {
            spot_entity.remove::<LandingTimer>();
        }
        (false, Some(_)) => {
            spot_entity.remove::<LandingTimer>();
        }
        _ => {}
    }
}

fn setup_spots(mut commands: Commands, query: Query<(Entity, &Terrain)>) {
    //let (entity, terrain) = query.iter().choose(&mut thread_rng()).unwrap();
    let (entity, terrain) = query.get_single().unwrap();

    let landing_spot = terrain::get_landing_spots(&terrain)
        .into_iter()
        .choose(&mut thread_rng())
        .unwrap();
    let landing_spot_mid = landing_spot.0.midpoint(landing_spot.1);
    let landing_spot_angle = (landing_spot.1.y - landing_spot.0.y)
        .atan2(landing_spot.1.x - landing_spot.0.x)
        - std::f32::consts::FRAC_PI_2;

    //dbg!(landing_spot_angle.to_degrees().abs());
    //println!(
    //    "DEBUG_ANGLE: {}",
    //    (landing_spot.1.y - landing_spot.0.y)
    //        .atan2(landing_spot.1.x - landing_spot.0.x)
    //        .to_degrees()
    //        .abs()
    //);

    let landing_spot_quat = Quat::from_rotation_z(landing_spot_angle);

    commands
        .get_entity(entity)
        .unwrap()
        .with_children(|parent| {
            parent.spawn((
                LandingSpot,
                LandingTimeLimit(Timer::new(Duration::from_secs_f32(1000.), TimerMode::Once)),
                Sensor,
                Collider::rectangle(10., landing_spot.0.distance(landing_spot.1)),
                TransformBundle::from(
                    Transform::from_xyz(landing_spot_mid.x, landing_spot_mid.y, 0.)
                        .with_rotation(landing_spot_quat),
                ),
            ));
        });
}

fn setup_planets(mut commands: Commands) {
    {
        let points = terrain::generate_points(0);
        let simplified = terrain::simplify_terrain(points);
        let collider = terrain::make_compound(&simplified);

        commands.spawn((
            Planet,
            Terrain(simplified),
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
        let collider = terrain::make_compound(&simplified);

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
        .spawn((
            Spaceship,
            Fuel(100.),
            Life(5.),
            RigidBody::Dynamic,
            Friction::new(10.).with_dynamic_coefficient(1.),
            Collider::rectangle(ship_size.x, ship_size.y),
            TransformBundle::from(
                Transform::from_xyz(133.2238, 1484.1008, 0.0)
                    .with_rotation(Quat::from_rotation_z(2.8365)),
            ),
            //TransformBundle::from(
            //    Transform::from_xyz(0.0, 1800.0, 0.0)
            //        .with_rotation(Quat::from_rotation_z(std::f32::consts::FRAC_PI_2)),
            //),
        ))
        .with_children(|parent| {
            //let foot1 = parent
            //    .spawn((
            //        RigidBody::Dynamic,
            //        Collider::rectangle(10., 10.),
            //        Friction::new(1.),
            //        Restitution::new(1.).with_combine_rule(CoefficientCombine::Max),
            //        TransformBundle::from(Transform::from_xyz(-20., (ship_size.x + 40.) / 2., 0.)),
            //    ))
            //    .id();
            //let foot2 = parent
            //    .spawn((
            //        RigidBody::Dynamic,
            //        Collider::rectangle(10., 10.),
            //        Friction::new(1.),
            //        Restitution::new(1.).with_combine_rule(CoefficientCombine::Max),
            //        TransformBundle::from(Transform::from_xyz(20., (ship_size.x + 40.) / 2., 0.)),
            //    ))
            //    .id();

            //parent.spawn(
            //    FixedJoint::new(parent.parent_entity(), foot1)
            //        .with_local_anchor_1(Vec2::new(-(ship_size.x / 2. + 15.), -20.))
            //        .with_local_anchor_2(Vec2::ZERO),
            //    //.with_angular_velocity_damping(-100.0),
            //    //.with_linear_velocity_damping(1.)
            //    //.with_compliance(0.00000001),
            //    //.with_compliance(0.000005),
            //);
            //parent.spawn(
            //    DistanceJoint::new(parent.parent_entity(), foot2)
            //        .with_local_anchor_1(Vec2::new(-(ship_size.x / 2. + 15.), 20.))
            //        .with_local_anchor_2(Vec2::ZERO),
            //    //.with_angular_velocity_damping(-100.0),
            //    //.with_linear_velocity_damping(1.)
            //    //.with_compliance(0.000001),
            //    //.with_compliance(0.000005),
            //);

            //parent.spawn(
            //    DistanceJoint::new(parent.parent_entity(), foot1)
            //        .with_local_anchor_1(Vec2::new(-(ship_size.x / 2. + 15.), -20.))
            //        .with_local_anchor_2(Vec2::ZERO)
            //        .with_rest_length(0.0)
            //        .with_linear_velocity_damping(1.)
            //        .with_angular_velocity_damping(10000000.0)
            //        .with_compliance(0.00000001),
            //    //.with_compliance(0.000005),
            //);
            //parent.spawn(
            //    DistanceJoint::new(parent.parent_entity(), foot2)
            //        .with_local_anchor_1(Vec2::new(-(ship_size.x / 2. + 15.), 20.))
            //        .with_local_anchor_2(Vec2::ZERO)
            //        .with_rest_length(0.0)
            //        .with_linear_velocity_damping(1.)
            //        .with_angular_velocity_damping(10000000.0)
            //        .with_compliance(0.000001),
            //    //.with_compliance(0.000005),
            //);

            //parent.spawn(
            //    PrismaticJoint::new(parent.parent_entity(), foot1)
            //        .with_local_anchor_1(Vec2::new(-(ship_size.x / 2. + 20.), -20.))
            //        //.with_local_anchor_2(Vec2::X * 10. / 2.)
            //        .with_local_anchor_2(Vec2::ZERO)
            //        .with_free_axis(Vec2::new(-(ship_size.x / 2. + 20.), -20.).normalize())
            //        .with_limits(-5.0, 5.0)
            //        .with_linear_velocity_damping(1.0)
            //        .with_angular_velocity_damping(10.0)
            //        .with_compliance(0.00000001),
            //);
            //parent.spawn(
            //    PrismaticJoint::new(parent.parent_entity(), foot2)
            //        .with_local_anchor_1(Vec2::new(-(ship_size.x / 2. + 20.), 20.))
            //        //.with_local_anchor_2(Vec2::X * 10. / 2.)
            //        .with_local_anchor_2(Vec2::ZERO)
            //        .with_free_axis(Vec2::new(-(ship_size.x / 2. + 25.), 20.).normalize())
            //        .with_limits(-5., 5.)
            //        .with_linear_velocity_damping(1.)
            //        .with_angular_velocity_damping(10.0)
            //        .with_compliance(0.00000001),
            //);

            let intensity = 240_000.;

            let propulsor_fire_size = 20.;
            let propulsor_fire_mesh = meshes.add(Triangle2d::new(
                Vec2::new(-1., -1.) * propulsor_fire_size / 2.,
                Vec2::new(0., 1.) * propulsor_fire_size / 2.,
                Vec2::new(1., -1.) * propulsor_fire_size / 2.,
            ));

            let pos = Vec2::new(ship_size.x / 2., 0.);
            parent.spawn((
                Propulsor::Top,
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
            parent.spawn((
                Propulsor::Bottom,
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
            parent.spawn((
                Propulsor::Left,
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
            parent.spawn((
                Propulsor::Right,
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
            parent.spawn((
                Propulsor::TopLeft,
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
            parent.spawn((
                Propulsor::TopRight,
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
        With<Camera>,
    >,
    ship_query: Query<&Transform, (With<Spaceship>, Without<Camera>)>,
    planets_query: Query<(Entity, &Transform), (With<Planet>, Without<Camera>)>,
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

fn spaceship_collisions(
    mut commands: Commands,
    time: Res<Time<Substeps>>,
    debug_settings: Res<DebugSettings>,
    mut next_state: ResMut<NextState<GameState>>,
    mut collision_event_reader: EventReader<Collision>,
    mut ship_query: Query<(Entity, &Spaceship, &mut Fuel)>,
    orb_query: Query<(Entity, &FuelOrb)>,
) {
    let (ship_entity, _, mut ship_fuel) = if let Ok(ship) = ship_query.get_single_mut() {
        ship
    } else {
        warn!("no ship in spaceship_collisions");
        return;
    };

    for Collision(contacts) in collision_event_reader.read() {
        if contacts.during_current_frame {
            let dt = time.delta_seconds();

            let other_entity = if ship_entity == contacts.entity2 {
                contacts.entity1
            } else if ship_entity == contacts.entity1 {
                contacts.entity2
            } else {
                continue;
            };

            if let Ok((orb_entity, _)) = orb_query.get(other_entity) {
                // Check for Fuel Orbs collision
                commands.entity(orb_entity).despawn_recursive();
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
}

#[derive(Component, Default)]
struct Focused {
    distance_sample: Option<f32>,
    relative_speed: Option<f32>,
    /// How many seconds ago was the last distance sampled.
    sample_age: f32,
}

fn update_focused_planet(
    time: Res<Time>,
    ship_query: Query<&Transform, With<Spaceship>>,
    mut focused_planet_query: Query<(&Transform, &mut Focused), With<Planet>>,
) {
    let ship_t = ship_query.get_single().unwrap();

    if let Ok((focused_t, mut focused)) = focused_planet_query.get_single_mut() {
        let new_distance = focused_t.translation.distance(ship_t.translation);
        let max_sample_age = 0.05; // seconds
        let new_sample_age = focused.sample_age + time.delta_seconds();

        match focused.distance_sample {
            None => {
                focused.distance_sample = Some(new_distance);
                focused.sample_age = 0.;
            }
            Some(old_distance) if new_sample_age >= max_sample_age => {
                focused.relative_speed =
                    Some((old_distance - new_distance).abs() * (1. / new_sample_age));
                focused.distance_sample = Some(new_distance);
                focused.sample_age = 0.;
            }
            _ => focused.sample_age = new_sample_age,
        }
    }
}

fn focus_planet(
    mut commands: Commands,
    keyboard_input: Res<ButtonInput<KeyCode>>,
    planets_query: Query<(Entity, &Planet, Option<&Focused>)>,
) {
    if keyboard_input.just_pressed(KeyCode::Tab) {
        let planets = planets_query
            .into_iter()
            .sort_by::<Entity>(|entity1, entity2| entity1.index().cmp(&entity2.index()))
            .collect::<Vec<_>>();
        let focused = planets.iter().find(|(_, _, focused)| focused.is_some());

        if let Some((old_focused_entity, _, _)) = focused {
            commands.entity(*old_focused_entity).remove::<Focused>();

            let index = old_focused_entity.index();
            let new_focused = planets
                .iter()
                .find(|(entity, _, focused)| focused.is_none() && entity.index() > index)
                .or_else(|| planets.iter().find(|(_, _, focused)| focused.is_none()));

            if let Some((new_focused_entity, _, _)) = new_focused {
                commands
                    .entity(*new_focused_entity)
                    .insert(Focused::default());
            }
        } else if let Some((new_focused_entity, _, _)) = planets.iter().next() {
            commands
                .entity(*new_focused_entity)
                .insert(Focused::default());
        }
    }
}

#[derive(Component, Reflect, Deref, DerefMut)]
/// Damage in `Life` a colliding living entity would take.
struct Projectile(f32);

#[derive(Component, Reflect, Deref, DerefMut)]
struct Life(f32);

fn projectile_collisions(
    mut commands: Commands,
    projectile_query: Query<(Entity, &Projectile, &CollidingEntities)>,
    mut alive_query: Query<(Entity, &mut Life)>,
) {
    projectile_query
        .into_iter()
        .for_each(|(projectile_id, projectile, projectile_colliding)| {
            let colliding_entity = projectile_colliding.iter().next();

            if let Some(colliding_entity) = colliding_entity {
                //commands.entity(*colliding_entity).log_components();

                commands
                    .get_entity(projectile_id)
                    .unwrap()
                    .despawn_recursive();

                let (_colliding_id, mut colliding_life) =
                    if let Ok(alive) = alive_query.get_mut(*colliding_entity) {
                        alive
                    } else {
                        return;
                    };

                colliding_life.0 -= projectile.0;
            }
        });
}

fn fire_spaceship(
    mut fire_at_ew: EventWriter<FireAtEvent>,
    ship_query: Query<(Entity, &Transform), With<Spaceship>>,
) {
    let (ship_id, ship_t) = ship_query.get_single().unwrap();

    let speed = 600.;
    let velocity = apply_quat(Vec2::new(speed, 0.), ship_t.rotation);

    fire_at_ew.send(FireAtEvent {
        shooter: ship_id,
        velocity,
    });
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
    let (entity, t, av, lv, mut fuel) = ship_query.get_single_mut().unwrap();
    let mut propulsors = propulsors_query.iter_mut().collect::<Vec<_>>();

    let get_force = |local_point: Vec2, local_force: Vec2| {
        let rotated_point = apply_quat(local_point, t.rotation);
        let rotated_force = apply_quat(local_force, t.rotation);

        return (rotated_force, rotated_point);
    };

    let mut active_propulsors = Vec::new();

    if keyboard_input.pressed(KeyCode::KeyA) {
        if let (Some(mut av), Some(mut lv)) = (av, lv) {
            av.0 = 0.;
            lv.0 = Vec2::ZERO;
        }
    }

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
