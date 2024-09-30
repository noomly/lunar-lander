use avian2d::math::Scalar;

pub fn map_range(
    min: Scalar,
    max: Scalar,
    new_min: Scalar,
    new_max: Scalar,
    value: Scalar,
) -> Scalar {
    let value = value.clamp(min, max);
    (value - min) / (max - min) * (new_max - new_min) + new_min
}
