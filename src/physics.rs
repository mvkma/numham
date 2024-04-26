use ndarray::Array1;
use std::ops::{Add, Mul};

#[derive(Clone)]
pub struct PhaseSpaceVector {
    pub dim: usize,
    pub p: Array1<f32>,
    pub q: Array1<f32>,
}

impl PhaseSpaceVector {
    pub fn new(p: Array1<f32>, q: Array1<f32>) -> PhaseSpaceVector {
        PhaseSpaceVector { dim: p.len(), p, q }
    }
}

impl Add for PhaseSpaceVector {
    type Output = PhaseSpaceVector;

    fn add(self, rhs: Self) -> Self::Output {
        Self::Output {
            dim: self.dim,
            p: self.p + rhs.p,
            q: self.q + rhs.q,
        }
    }
}

impl Add<&PhaseSpaceVector> for PhaseSpaceVector {
    type Output = PhaseSpaceVector;

    fn add(self, rhs: &PhaseSpaceVector) -> Self::Output {
        Self::Output {
            dim: self.dim,
            p: self.p + &rhs.p,
            q: self.q + &rhs.q,
        }
    }
}

impl Add<PhaseSpaceVector> for &PhaseSpaceVector {
    type Output = PhaseSpaceVector;

    fn add(self, rhs: PhaseSpaceVector) -> Self::Output {
        Self::Output {
            dim: self.dim,
            p: &self.p + rhs.p,
            q: &self.q + rhs.q,
        }
    }
}

impl Mul<f32> for PhaseSpaceVector {
    type Output = PhaseSpaceVector;

    fn mul(self, rhs: f32) -> Self::Output {
        Self::Output {
            dim: self.dim,
            p: self.p * rhs,
            q: self.q * rhs,
        }
    }
}

impl Mul<&PhaseSpaceVector> for f32 {
    type Output = PhaseSpaceVector;

    fn mul(self, rhs: &PhaseSpaceVector) -> Self::Output {
        Self::Output {
            dim: rhs.dim,
            p: self * &rhs.p,
            q: self * &rhs.q,
        }
    }
}

impl Mul<PhaseSpaceVector> for f32 {
    type Output = PhaseSpaceVector;

    fn mul(self, rhs: PhaseSpaceVector) -> Self::Output {
        Self::Output {
            dim: rhs.dim,
            p: self * rhs.p,
            q: self * rhs.q,
        }
    }
}
