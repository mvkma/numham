use std::ops::Div;

use crate::physics::Hamiltonian;
use ndarray::{array, s, stack, Array, Array1, Array2, ArrayView1, Axis};

pub struct IntegrationParams {
    pub step_size: f64,
    pub t0: f64,
    pub pq0: Array1<f64>,
    pub tmax: Option<f64>,
}

pub struct RungeKuttaIntegrator {
    pub params: IntegrationParams,
    pub ham: Box<dyn Hamiltonian>,
    pub t: f64,
    pq: Array1<f64>,
    pqdot: Array1<f64>,
    stages: Array2<f64>,
    // Butcher tableau
    a: Array2<f64>,
    b: Array1<f64>,
    c: Array1<f64>,
    d: Array1<f64>,
}

impl RungeKuttaIntegrator {
    pub fn new(params: IntegrationParams, ham: Box<dyn Hamiltonian>) -> Self {
        let t = params.t0;
        let pq = params.pq0.to_owned();
        let dim = pq.raw_dim();
        let pqdot = Array::zeros(dim);
        // nstages + 1
        let stages = stack![Axis(0), pqdot, pqdot, pqdot, pqdot, pqdot, pqdot, pqdot];

        RungeKuttaIntegrator {
            params,
            ham,
            t,
            pq,
            pqdot,
            stages,
            // nstages - 1
            a: stack![
                Axis(0),
                // array![0.5, 0.0, 0.0],
                // array![0.0, 0.5, 0.0],
                // array![0.0, 0.0, 1.0],
                array![1.0 / 4.0, 0.0, 0.0, 0.0, 0.0],
                array![3.0 / 32.0, 9.0 / 32.0, 0.0, 0.0, 0.0],
                array![1932.0 / 2197.0, -7200.0 / 2197.0, 7296.0 / 2197.0, 0.0, 0.0],
                array![439.0 / 216.0, -8.0, 3680.0 / 513.0, -845.0 / 4104.0, 0.0],
                array![
                    -8.0 / 27.0,
                    2.0,
                    -3544.0 / 2565.0,
                    1859.0 / 4104.0,
                    -11.0 / 40.0
                ],
            ],
            // nstages
            // b: array![1.0 / 6.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 6.0],
            b: array![
                16.0 / 135.0,
                0.0,
                6656.0 / 12825.0,
                28561.0 / 56430.0,
                -9.0 / 50.0,
                2.0 / 55.0
            ],
            // nstages - 1
            // c: array![0.5, 0.5, 1.0],
            c: array![1.0 / 4.0, 3.0 / 8.0, 12.0 / 13.0, 1.0, 0.5],
            // nstages + 1
            // d: array![0.0, 0.0, 0.0, 0.0, 0.0],
            d: array![
                25.0 / 216.0,
                0.0,
                1408.0 / 2565.0,
                2197.0 / 4104.0,
                -1.0 / 5.0,
                0.0,
            ],
        }
    }

    pub fn state(&self) -> (f64, ArrayView1<f64>) {
        (self.t, self.pq.view())
    }

    pub fn next(&mut self) -> Option<(f64, ArrayView1<f64>)> {
        if self.params.tmax.is_some_and(|tmax| tmax >= self.t) {
            None
        } else {
            let h = self.params.step_size;
            let t = self.t;

            self.stages.slice_mut(s![0, ..]).assign(&self.pqdot);

            for (i, (a, c)) in self.a.axis_iter(Axis(0)).zip(&self.c).enumerate() {
                let dpq = self
                    .stages
                    .slice(s![..i + 1, ..])
                    .t()
                    .dot(&a.slice(s![..i + 1]))
                    * h;
                self.ham.eom(
                    c.mul_add(h, t),
                    (dpq + &self.pq).view(),
                    self.stages.slice_mut(s![i + 1, ..]),
                );
            }

            self.pq = h * self.stages.slice(s![..-1, ..]).t().dot(&self.b) + &self.pq;

            self.ham.eom(
                self.t + h,
                self.pq.view(),
                self.stages.slice_mut(s![-1, ..]),
            );
            self.pqdot.assign(&self.stages.slice(s![-1, ..]));

            let error = self
                .stages
                .slice(s![..-1, ..])
                .t()
                .dot(&(&self.b - &self.d));

            let err = error.dot(&error).sqrt();
            let hnew = (0.00000001 / err).powf(1.0 / 5.0) * h * 0.9;

            if hnew < 0.001 {
                self.params.step_size = hnew;
            } else {
                self.params.step_size = 0.001;
            }

            if hnew.is_nan() {
                println!("err = {}", err);
                panic!();
            }

            self.t += h;

            Some(self.state())
        }
    }
}
