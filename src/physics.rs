use ndarray::{array, s, Array1, ArrayView1};
use std::ops::{Add, Mul};

#[derive(Clone)]
pub struct PhaseSpaceVector {
    pub dim: usize,
    pub p: Array1<f64>,
    pub q: Array1<f64>,
}

impl PhaseSpaceVector {
    pub fn new(p: Array1<f64>, q: Array1<f64>) -> PhaseSpaceVector {
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

impl Mul<f64> for PhaseSpaceVector {
    type Output = PhaseSpaceVector;

    fn mul(self, rhs: f64) -> Self::Output {
        Self::Output {
            dim: self.dim,
            p: self.p * rhs,
            q: self.q * rhs,
        }
    }
}

impl Mul<&PhaseSpaceVector> for f64 {
    type Output = PhaseSpaceVector;

    fn mul(self, rhs: &PhaseSpaceVector) -> Self::Output {
        Self::Output {
            dim: rhs.dim,
            p: self * &rhs.p,
            q: self * &rhs.q,
        }
    }
}

impl Mul<PhaseSpaceVector> for f64 {
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
    fn eom(&self, t: f64, pq: &PhaseSpaceVector, pqdot: &mut PhaseSpaceVector);
    fn positions<'a>(&'a self, pq: &'a PhaseSpaceVector) -> Vec<ArrayView1<f64>>;
    fn momenta<'a>(&'a self, pq: &'a PhaseSpaceVector) -> Vec<ArrayView1<f64>>;
}

pub struct TwoBodyHamiltonian {
    m: f64,
    k: f64,
}

impl TwoBodyHamiltonian {
    pub fn new(m: f64, k: f64) -> Self {
        TwoBodyHamiltonian { m, k }
    }
}

impl Hamiltonian for TwoBodyHamiltonian {
    fn eom(&self, _t: f64, pq: &PhaseSpaceVector, pqdot: &mut PhaseSpaceVector) {
        let pos = self.positions(pq);
        let r10 = &pos[1] - &pos[0];

        let r = (r10[0] * r10[0] + r10[1] * r10[1]).sqrt();
        let r3 = f64::powi(r, 3);

        pqdot.p = -self.k * &pq.q / r3;
        pqdot.q = &pq.p / self.m;
    }

    fn positions<'a>(&'a self, pq: &'a PhaseSpaceVector) -> Vec<ArrayView1<f64>> {
        vec![pq.q.slice(s![0..2]), pq.q.slice(s![2..4])]
    }

    fn momenta<'a>(&'a self, pq: &'a PhaseSpaceVector) -> Vec<ArrayView1<f64>> {
        vec![pq.p.slice(s![0..2]), pq.p.slice(s![2..4])]
    }
}

pub struct ThreeBodyHamiltonian {
    m: f64,
    k: f64,
}

impl ThreeBodyHamiltonian {
    pub fn new(m: f64, k: f64) -> Self {
        Self { m, k }
    }
}

impl Hamiltonian for ThreeBodyHamiltonian {
    fn eom(&self, t: f64, pq: &PhaseSpaceVector, pqdot: &mut PhaseSpaceVector) {
        let pos = self.positions(pq);
        let r01 = &pos[0] - &pos[1];
        let r02 = &pos[0] - &pos[2];
        let r12 = &pos[1] - &pos[2];

        let r01_3 = f64::powi((r01[0] * r01[0] + r01[1] * r01[1]).sqrt(), 3);
        let r02_3 = f64::powi((r02[0] * r02[0] + r02[1] * r02[1]).sqrt(), 3);
        let r12_3 = f64::powi((r12[0] * r12[0] + r12[1] * r12[1]).sqrt(), 3);

        let f0 = -self.k * (&r01 / r01_3 + &r02 / r02_3);
        let f1 = -self.k * (-&r01 / r01_3 + &r12 / r12_3);
        let f2 = -self.k * (-&r02 / r02_3 - &r12 / r12_3);

        pqdot.p = array![f0[0], f0[1], f1[0], f1[1], f2[0], f2[1]];

        pqdot.q = &pq.p / self.m;
    }

    fn positions<'a>(&'a self, pq: &'a PhaseSpaceVector) -> Vec<ArrayView1<f64>> {
        vec![
            pq.q.slice(s![0..2]),
            pq.q.slice(s![2..4]),
            pq.q.slice(s![4..6]),
        ]
    }

    fn momenta<'a>(&'a self, pq: &'a PhaseSpaceVector) -> Vec<ArrayView1<f64>> {
        vec![
            pq.p.slice(s![0..2]),
            pq.p.slice(s![2..4]),
            pq.p.slice(s![4..6]),
        ]
    }
}
