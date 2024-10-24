use std::f32::consts::PI;

use avian2d::{math::Vector, prelude::*};
use bevy::{
    prelude::*,
    sprite::{MaterialMesh2dBundle, Mesh2dHandle},
    window::PrimaryWindow,
};
use noise::{Fbm, MultiFractal, NoiseFn, OpenSimplex, Perlin, SuperSimplex};
use rand::prelude::*;

pub fn generate_points(seed: u32) -> Vec<Vec2> {
    let noise = SuperSimplex::new(0);
    let fbm = Fbm::<SuperSimplex>::new(seed)
        .set_octaves(4)
        .set_lacunarity(1.8);

    let mut points = Vec::new();

    let precision = 100000.;
    let amplitude = 300.;
    let radius = 1500.;

    let mut max = f64::MIN;
    let mut min = f64::MAX;

    for i in 0..(precision + 1.) as i32 {
        let i = i as f64;

        let x = (i / precision) * std::f64::consts::PI * 2.;

        //let point = noise.get([x * 2., 0.]) * noise.get([x * 5., 0.]) * 250.;
        //let point = noise.get([x.cos() * 2., x.sin() * 2.]) * amplitude;
        let point = fbm.get([x.cos() * 2.5, x.sin() * 2.5]) * amplitude;

        points.push(Vec2::new(
            ((radius + point) * x.cos()) as f32,
            ((radius + point) * x.sin()) as f32,
        ));

        if point > max {
            max = point;
        }

        if point < min {
            min = point;
        }
    }

    points
}

pub fn simplify_terrain(points: Vec<Vec2>) -> Vec<Vec2> {
    let points: Vec<simplify_polyline::Point<2, f32>> = points
        .iter()
        .map(|p| simplify_polyline::Point { vec: [p.x, p.y] })
        .collect();

    let simplified = simplify_polyline::simplify(&points, 2., true);

    let mut simplified = simplified
        .iter()
        .map(|p| Vec2::new(p.vec[0], p.vec[1]))
        .collect::<Vec<_>>();

    simplified.push(simplified[0]);

    simplified
}

pub fn get_landing_spots(points: &Vec<Vec2>) -> Vec<(Vec2, Vec2)> {
    points
        .chunks(2)
        .filter_map(|points| {
            let (point1, point2) = {
                if points.len() != 2 {
                    panic!("there is an odd number of points for terrain");
                } else {
                    (points[0], points[1])
                }
            };

            let mid = point1.midpoint(point2);
            let sphere_angle = mid.y.atan2(mid.x);

            let angle = (point2.y - point1.y).atan2(point2.x - point1.x) - sphere_angle
                + std::f32::consts::FRAC_PI_2;
            let angle = (angle + PI).rem_euclid(2. * PI) - PI;
            let angle = angle.to_degrees().abs();

            let length = point1.distance(point2);

            let max_angle = 26.;
            let min_length = 50.;

            if (angle >= 180. - max_angle || angle <= max_angle) && length >= min_length {
                Some((point1, point2))
            } else {
                None
            }
        })
        .collect()
}

pub fn make_compound(points: &Vec<Vec2>) -> Collider {
    let mut lines = Vec::new();

    let mut last_point = points[0];

    for i in 1..points.len() - 1 {
        let point = points[i];

        let line_length = point.distance(last_point);
        //let line = Collider::rectangle(line_length, 2.);
        let line = Collider::capsule(2., line_length);
        let position = Position::new(last_point.midpoint(point));
        let rotation = avian2d::position::Rotation::radians(
            (point.y - last_point.y).atan2(point.x - last_point.x) + std::f32::consts::FRAC_PI_2,
        );

        last_point = point;
        lines.push((position, rotation, line));
    }

    Collider::compound(lines)
}
