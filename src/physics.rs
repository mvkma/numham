use ndarray::{concatenate, s, ArrayView2, ArrayViewMut2, Axis};
use std::ops::{Div, Neg};

pub trait Hamiltonian {
    fn eom(&self, t: f64, pq: ArrayView2<f64>, pqdot: ArrayViewMut2<f64>);
    fn num_particles(&self) -> usize;
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
    fn eom(&self, _t: f64, pq: ArrayView2<f64>, mut pqdot: ArrayViewMut2<f64>) {
        let p = pq.index_axis(Axis(0), 0);
        let q = pq.index_axis(Axis(0), 1);

        let r3 = (&q.slice(s![0..2]) - &q.slice(s![2..4]))
            .map(|x| x * x)
            .sum()
            .sqrt()
            .powi(3);

        let pdot = &q * self.k.neg().div(r3);
        let qdot = &p / self.m;

        pqdot.index_axis_mut(Axis(0), 0).assign(&pdot);
        pqdot.index_axis_mut(Axis(0), 1).assign(&qdot);
    }

    fn num_particles(&self) -> usize {
        2
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
    fn eom(&self, _t: f64, pq: ArrayView2<f64>, mut pqdot: ArrayViewMut2<f64>) {
        let q = pq.index_axis(Axis(0), 1);

        let mut r01 = &q.slice(s![0..2]) - &q.slice(s![2..4]);
        let mut r02 = &q.slice(s![0..2]) - &q.slice(s![4..6]);
        let mut r12 = &q.slice(s![2..4]) - &q.slice(s![4..6]);

        let r01_3 = r01.map(|x| x.powi(2)).sum().sqrt().powi(3);
        let r02_3 = r02.map(|x| x.powi(2)).sum().sqrt().powi(3);
        let r12_3 = r12.map(|x| x.powi(2)).sum().sqrt().powi(3);

        r01 /= r01_3 * self.k;
        r02 /= r02_3 * self.k;
        r12 /= r12_3 * self.k;

        let pdot = concatenate![Axis(0), -(&r01 + &r02), (&r01 - &r12), (&r02 + &r12)];
        let qdot = &pq.index_axis(Axis(0), 0) / self.m;

        pqdot.index_axis_mut(Axis(0), 0).assign(&pdot);
        pqdot.index_axis_mut(Axis(0), 1).assign(&qdot);
    }

    fn num_particles(&self) -> usize {
        3
    }
}
