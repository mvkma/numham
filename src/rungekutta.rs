use crate::{physics::Hamiltonian, PhaseSpaceVector};
use ndarray::Array;

pub struct IntegrationParams {
    pub step_size: f64,
    pub t0: f64,
    pub pq0: PhaseSpaceVector,
    pub tmax: Option<f64>,
}

pub struct RungeKuttaIntegrator {
    pub params: IntegrationParams,
    pub ham: Box<dyn Hamiltonian>,
    t: f64,
    pq: PhaseSpaceVector,
    k1: PhaseSpaceVector,
    k2: PhaseSpaceVector,
    k3: PhaseSpaceVector,
    k4: PhaseSpaceVector,
}

impl RungeKuttaIntegrator {
    pub fn new(params: IntegrationParams, ham: Box<dyn Hamiltonian>) -> RungeKuttaIntegrator {
        let t = params.t0;
        let pq = params.pq0.clone();
        let dim = pq.p.raw_dim();

        RungeKuttaIntegrator {
            params,
            ham,
            t,
            pq,
            k1: PhaseSpaceVector::new(Array::zeros(dim), Array::zeros(dim)),
            k2: PhaseSpaceVector::new(Array::zeros(dim), Array::zeros(dim)),
            k3: PhaseSpaceVector::new(Array::zeros(dim), Array::zeros(dim)),
            k4: PhaseSpaceVector::new(Array::zeros(dim), Array::zeros(dim)),
        }
    }

    pub fn state(&self) -> (f64, PhaseSpaceVector) {
        (self.t, self.pq.clone())
    }
}

impl Iterator for RungeKuttaIntegrator {
    type Item = (f64, PhaseSpaceVector);

    fn next(&mut self) -> Option<(f64, PhaseSpaceVector)> {
        if self.params.tmax.is_some_and(|tmax| tmax >= self.t) {
            None
        } else {
            let h = self.params.step_size;
            let h2 = h * 0.5;
            let t = self.t;
            let pq = &self.pq;

            self.ham.eom(t, pq, &mut self.k1);
            self.ham.eom(t + h2, &(pq + h2 * &self.k1), &mut self.k2);
            self.ham.eom(t + h2, &(pq + h2 * &self.k2), &mut self.k3);
            self.ham.eom(t + h, &(pq + h * &self.k3), &mut self.k4);

            self.t += h;

            self.pq = pq + h / 6.0 * (&self.k1 + 2.0 * &self.k2 + 2.0 * &self.k3 + &self.k4);
            Some(self.state())
        }
    }
}
