//! Parametric helical curves in three-dimensional space.

use super::{NurbsCurve3, Vec3};

/// Direction in which the curve winds when viewed along its positive axis.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HelixDirection {
    Clockwise,
    CounterClockwise,
}

/// Parameters needed to build and measure a circular or conical helix.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HelixCurve {
    pub base_center: [f64; 3],
    pub axis_direction: [f64; 3],
    pub start_direction: [f64; 3],
    pub base_radius: f64,
    pub top_radius: f64,
    pub height: f64,
    pub turns: f64,
    pub direction: HelixDirection,
}

impl HelixCurve {
    const SEGMENTS_PER_TURN: f64 = 6.0;

    fn frame(&self) -> Option<(Vec3, Vec3, Vec3, Vec3)> {
        let center = Vec3::from(self.base_center);
        let axis = Vec3::from(self.axis_direction).normalize()?;
        let radial = Vec3::from(self.start_direction);
        let x_axis = (radial - axis * radial.dot(axis)).normalize()?;
        let y_axis = axis.cross(x_axis).normalize()?;
        Some((center, x_axis, y_axis, axis))
    }

    fn is_valid(&self) -> bool {
        self.base_center.iter().all(|value| value.is_finite())
            && self.axis_direction.iter().all(|value| value.is_finite())
            && self.start_direction.iter().all(|value| value.is_finite())
            && self.base_radius.is_finite()
            && self.top_radius.is_finite()
            && self.height.is_finite()
            && self.turns.is_finite()
            && self.base_radius > 0.0
            && self.top_radius >= 0.0
            && self.turns > 0.0
    }

    fn point_and_derivative(
        &self,
        theta: f64,
        total_angle: f64,
        frame: (Vec3, Vec3, Vec3, Vec3),
    ) -> (Vec3, Vec3) {
        let (center, x_axis, y_axis, axis) = frame;
        let fraction = theta / total_angle;
        let radius = self.base_radius + (self.top_radius - self.base_radius) * fraction;
        let radial_rate = (self.top_radius - self.base_radius) / total_angle;
        let height_rate = self.height / total_angle;
        let winding = match self.direction {
            HelixDirection::Clockwise => -1.0,
            HelixDirection::CounterClockwise => 1.0,
        };
        let cosine = theta.cos();
        let sine = theta.sin();
        let point = center
            + x_axis * (radius * cosine)
            + y_axis * (winding * radius * sine)
            + axis * (self.height * fraction);
        let derivative = x_axis * (radial_rate * cosine - radius * sine)
            + y_axis * (winding * (radial_rate * sine + radius * cosine))
            + axis * height_rate;
        (point, derivative)
    }

    /// Builds a smooth cubic NURBS representation of this helix.
    ///
    /// Each turn is split into six cubic spans. The spans use the analytic
    /// position and tangent of the helix, so adjacent spans meet smoothly and
    /// the resulting curve stays parametric instead of becoming a polyline.
    pub fn nurbs(&self) -> Option<NurbsCurve3> {
        if !self.is_valid() {
            return None;
        }
        let frame = self.frame()?;
        let total_angle = self.turns * std::f64::consts::TAU;
        let segments = (self.turns * Self::SEGMENTS_PER_TURN).ceil().max(1.0) as usize;
        let angle_step = total_angle / segments as f64;
        let mut controls = Vec::with_capacity(segments * 3 + 1);

        for segment in 0..segments {
            let from = segment as f64 * angle_step;
            let to = (segment + 1) as f64 * angle_step;
            let (start, start_tangent) = self.point_and_derivative(from, total_angle, frame);
            let (end, end_tangent) = self.point_and_derivative(to, total_angle, frame);
            if segment == 0 {
                controls.push(start.to_array());
            }
            controls.push((start + start_tangent * (angle_step / 3.0)).to_array());
            controls.push((end - end_tangent * (angle_step / 3.0)).to_array());
            controls.push(end.to_array());
        }

        let mut knots = Vec::with_capacity(controls.len() + 4);
        knots.extend(std::iter::repeat_n(0.0, 4));
        for span in 1..segments {
            knots.extend(std::iter::repeat_n(span as f64, 3));
        }
        knots.extend(std::iter::repeat_n(segments as f64, 4));
        NurbsCurve3::new_strict(3, controls.clone(), knots, vec![1.0; controls.len()])
    }

    /// Exact length of the mathematical circular or conical helix.
    pub fn length(&self) -> Option<f64> {
        if !self.is_valid() || self.frame().is_none() {
            return None;
        }
        let total_angle = self.turns * std::f64::consts::TAU;
        let radial_rate = (self.top_radius - self.base_radius) / total_angle;
        let height_rate = self.height / total_angle;
        let constant = radial_rate.hypot(height_rate);
        if radial_rate.abs() <= 1.0e-15 {
            return Some(total_angle * self.base_radius.hypot(constant));
        }
        let primitive = |radius: f64| {
            if constant <= 1.0e-15 {
                0.5 * radius * radius.abs()
            } else {
                0.5
                    * (radius * radius.hypot(constant)
                        + constant * constant * (radius / constant).asinh())
            }
        };
        Some(((primitive(self.top_radius) - primitive(self.base_radius)) / radial_rate).abs())
    }

    /// Taper angle between the helix envelope and its axis.
    pub fn turn_slope(&self) -> Option<f64> {
        self.is_valid()
            .then(|| (self.top_radius - self.base_radius).atan2(self.height.abs()))
    }
}
