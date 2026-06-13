/// Semantic field definitions.
/// Represents the field structure that encodes meaning and relationships.

use crate::geom::space::Coordinate3;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FieldPoint {
    pub position: Coordinate3,
    pub intensity: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticField {
    concept_map: BTreeMap<String, FieldPoint>,
}

impl SemanticField {
    pub fn new() -> Self {
        Self {
            concept_map: BTreeMap::new(),
        }
    }

    pub fn upsert_concept(&mut self, concept: impl Into<String>, point: FieldPoint) {
        self.concept_map.insert(concept.into(), point);
    }

    pub fn concept(&self, concept: &str) -> Option<&FieldPoint> {
        self.concept_map.get(concept)
    }

    pub fn concept_count(&self) -> usize {
        self.concept_map.len()
    }

    pub fn ordered_concepts(&self) -> impl Iterator<Item = (&String, &FieldPoint)> {
        self.concept_map.iter()
    }

    pub fn apply_uniform_delta(&mut self, delta: i64) {
        for point in self.concept_map.values_mut() {
            point.intensity += delta;
        }
    }

    pub fn map_intensity<F>(&mut self, mut mapper: F)
    where
        F: FnMut(i64) -> i64,
    {
        for point in self.concept_map.values_mut() {
            point.intensity = mapper(point.intensity);
        }
    }
}

impl Default for SemanticField {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn field_has_deterministic_order() {
        let mut field = SemanticField::new();
        field.upsert_concept(
            "zeta",
            FieldPoint {
                position: Coordinate3::new(0, 0, 0),
                intensity: 1,
            },
        );
        field.upsert_concept(
            "alpha",
            FieldPoint {
                position: Coordinate3::new(1, 0, 0),
                intensity: 2,
            },
        );

        let concepts: Vec<&str> = field.ordered_concepts().map(|(k, _)| k.as_str()).collect();
        assert_eq!(concepts, vec!["alpha", "zeta"]);
    }
}
