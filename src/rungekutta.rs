use std::ops::Div;

use crate::physics::Hamiltonian;
use ndarray::{stack, Array, Array2, Array3, ArrayView2, Axis};

pub struct IntegrationParams {
    pub step_size: f64,
    pub t0: f64,
    pub pq0: Array2<f64>,
    pub tmax: Option<f64>,
}

pub struct RungeKuttaIntegrator {
    pub params: IntegrationParams,
    pub ham: Box<dyn Hamiltonian>,
    t: f64,
    pq: Array2<f64>,
    pqdot: Array2<f64>,
    stages: Array3<f64>,
}

impl RungeKuttaIntegrator {
    pub fn new(params: IntegrationParams, ham: Box<dyn Hamiltonian>) -> Self {
        let t = params.t0;
        let pq = params.pq0.to_owned();
        let dim = pq.raw_dim();
        let pqdot = Array::zeros(dim);
        let stages = stack![Axis(0), pqdot, pqdot, pqdot, pqdot, pqdot];

        RungeKuttaIntegrator {
            params,
            ham,
            t,
            pq,
            pqdot,
            stages,
        }
    }

    pub fn state(&self) -> (f64, ArrayView2<f64>) {
        (self.t, self.pq.view())
    }

    pub fn next(&mut self) -> Option<(f64, ArrayView2<f64>)> {
        if self.params.tmax.is_some_and(|tmax| tmax >= self.t) {
            None
        } else {
            let h = self.params.step_size;
            let h2 = h * 0.5;
            let t = self.t;
            let pq = self.pq.view();

            self.stages.index_axis_mut(Axis(0), 0).assign(&self.pqdot);

            self.ham.eom(t, pq, self.stages.index_axis_mut(Axis(0), 1));

            let dpq = &self.stages.index_axis(Axis(0), 1) * h2;
            self.ham.eom(
                h.mul_add(0.5, t),
                (dpq + &pq).view(),
                self.stages.index_axis_mut(Axis(0), 2),
            );

            let dpq = &self.stages.index_axis(Axis(0), 2) * h2;
            self.ham.eom(
                h.mul_add(0.5, t),
                (dpq + &pq).view(),
                self.stages.index_axis_mut(Axis(0), 3),
            );

            let dpq = &self.stages.index_axis(Axis(0), 3) * h;
            self.ham.eom(
                t + h,
                (dpq + &pq).view(),
                self.stages.index_axis_mut(Axis(0), 4),
            );

            self.t += h;

            let dpq = &self.stages.index_axis(Axis(0), 1)
                + &self.stages.index_axis(Axis(0), 2) * 2.0
                + &self.stages.index_axis(Axis(0), 3) * 2.0
                + &self.stages.index_axis(Axis(0), 4);

            self.pq = h.div(6.0) * dpq + pq;
            Some(self.state())
        }
    }
}
