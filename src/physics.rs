use ndarray::{s, Array1, ArrayView1};
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

pub trait Hamiltonian {
    fn eom(&self, t: f32, pq: &PhaseSpaceVector, pqdot: &mut PhaseSpaceVector);
    fn positions<'a>(&'a self, pq: &'a PhaseSpaceVector) -> Vec<ArrayView1<f32>>;
    fn momenta<'a>(&'a self, pq: &'a PhaseSpaceVector) -> Vec<ArrayView1<f32>>;
}

pub struct TwoBodyHamiltonian {
    m: f32,
    k: f32,
}

impl TwoBodyHamiltonian {
    pub fn new(m: f32, k: f32) -> Self {
        TwoBodyHamiltonian { m, k }
    }
}

impl Hamiltonian for TwoBodyHamiltonian {
    fn eom(&self, _t: f32, pq: &PhaseSpaceVector, pqdot: &mut PhaseSpaceVector) {
        let pos = self.positions(pq);
        let r10 = &pos[1] - &pos[0];

        let r = (r10[0] * r10[0] + r10[1] * r10[1]).sqrt();
        let r3 = f32::powi(r, 3);

        pqdot.p = -self.k * &pq.q / r3;
        pqdot.q = &pq.p / self.m;
    }

    fn positions<'a>(&'a self, pq: &'a PhaseSpaceVector) -> Vec<ArrayView1<f32>> {
        vec![pq.q.slice(s![0..2]), pq.q.slice(s![2..4])]
    }

    fn momenta<'a>(&'a self, pq: &'a PhaseSpaceVector) -> Vec<ArrayView1<f32>> {
        vec![pq.p.slice(s![0..2]), pq.p.slice(s![2..4])]
    }
}
