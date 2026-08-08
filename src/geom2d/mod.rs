//! 2D curve algebra.
//!
//! This layer answers the questions the rest of the kernel asks constantly:
//! where do two curves meet, what does this loop offset to, is this point
//! inside, which fragments chain into a closed boundary. Editing commands
//! sit directly on it, and the B-rep layer reaches down into it whenever it
//! has to work in a face's parameter space.
//!
//! Everything here obeys the coordinate policy in [`frame`].

pub mod frame;

pub use frame::Frame;

/// Distance below which two positions are treated as one.
///
/// Carried explicitly rather than read from a global, because a drawing in
/// millimetres and a drawing in survey feet do not want the same value, and
/// because a boolean needs to widen it locally without disturbing anything
/// else.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Tolerance {
    linear: f64,
}

impl Tolerance {
    /// A tolerance of `linear` world units.
    ///
    /// # Panics
    ///
    /// If `linear` is not finite and positive. A non-positive tolerance makes
    /// every comparison in the kernel meaningless, so it is worth refusing at
    /// the boundary rather than producing degenerate topology later.
    pub fn new(linear: f64) -> Self {
        assert!(
            linear.is_finite() && linear > 0.0,
            "tolerance must be finite and positive, got {linear}"
        );
        Self { linear }
    }

    /// The linear tolerance in world units.
    pub const fn linear(&self) -> f64 {
        self.linear
    }

    /// Whether `a` and `b` are within tolerance of each other.
    pub fn are_equal(&self, a: f64, b: f64) -> bool {
        (a - b).abs() <= self.linear
    }
}

impl Default for Tolerance {
    /// 1e-7, matching the point-equality tolerance ACIS records in its
    /// header, so geometry surviving a round trip is judged by the same
    /// yardstick at both ends.
    fn default() -> Self {
        Self::new(1e-7)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn equality_respects_the_configured_tolerance() {
        let tol = Tolerance::new(1e-3);
        assert!(tol.are_equal(1.0, 1.0005));
        assert!(!tol.are_equal(1.0, 1.002));
    }

    #[test]
    #[should_panic(expected = "finite and positive")]
    fn zero_tolerance_is_refused() {
        Tolerance::new(0.0);
    }
}
