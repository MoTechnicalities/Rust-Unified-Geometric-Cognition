/// Resonance modes for semantic propagation.
/// Defines how meaning resonates through geometric structures.

use crate::geom::field::SemanticField;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ResonanceMode {
    Amplify,
    Dampen,
    Balance,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ArithmeticMode {
    Exact,
    BoundedApproximate { max_error: i64 },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResonanceTransform {
    pub mode: ResonanceMode,
    pub magnitude: i64,
    pub arithmetic: ArithmeticMode,
}

impl ResonanceTransform {
    pub fn new(mode: ResonanceMode, magnitude: i64, arithmetic: ArithmeticMode) -> Self {
        Self {
            mode,
            magnitude,
            arithmetic,
        }
    }

    pub fn apply(&self, field: &mut SemanticField) {
        let delta = match self.mode {
            ResonanceMode::Amplify => self.magnitude,
            ResonanceMode::Dampen => -self.magnitude,
            ResonanceMode::Balance => 0,
        };

        match self.arithmetic {
            ArithmeticMode::Exact => field.apply_uniform_delta(delta),
            ArithmeticMode::BoundedApproximate { max_error } => {
                let quantum = (max_error.max(0)) + 1;
                field.map_intensity(|intensity| {
                    let exact = intensity + delta;
                    if quantum == 1 {
                        exact
                    } else {
                        (exact / quantum) * quantum
                    }
                });
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geom::field::FieldPoint;
    use crate::geom::space::Coordinate3;

    #[test]
    fn amplify_increases_intensity() {
        let mut field = SemanticField::new();
        field.upsert_concept(
            "light",
            FieldPoint {
                position: Coordinate3::new(0, 0, 0),
                intensity: 5,
            },
        );

        ResonanceTransform::new(ResonanceMode::Amplify, 3, ArithmeticMode::Exact).apply(&mut field);

        assert_eq!(field.concept("light").map(|p| p.intensity), Some(8));
    }

    #[test]
    fn bounded_approximate_respects_error_bound() {
        let mut field = SemanticField::new();
        field.upsert_concept(
            "signal",
            FieldPoint {
                position: Coordinate3::new(0, 0, 0),
                intensity: 5,
            },
        );

        let exact_value = 12;
        ResonanceTransform::new(
            ResonanceMode::Amplify,
            7,
            ArithmeticMode::BoundedApproximate { max_error: 2 },
        )
        .apply(&mut field);

        let approx_value = field.concept("signal").map(|p| p.intensity).unwrap_or_default();
        assert!((exact_value - approx_value).abs() <= 2);
    }
}
