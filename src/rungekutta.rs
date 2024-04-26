use crate::PhaseSpaceVector;
use ndarray::Array;

pub struct IntegrationParams {
    pub step_size: f32,
    pub t0: f32,
    pub pq0: PhaseSpaceVector,
    pub tmax: Option<f32>,
}

pub struct RungeKuttaIntegrator {
    params: IntegrationParams,
    func: fn(f32, &PhaseSpaceVector, &mut PhaseSpaceVector),
    t: f32,
    pq: PhaseSpaceVector,
    k1: PhaseSpaceVector,
    k2: PhaseSpaceVector,
    k3: PhaseSpaceVector,
    k4: PhaseSpaceVector,
}

impl RungeKuttaIntegrator {
    pub fn new(
        params: IntegrationParams,
        func: fn(f32, &PhaseSpaceVector, &mut PhaseSpaceVector),
    ) -> RungeKuttaIntegrator {
        let t = params.t0;
        let pq = params.pq0.clone();
        let dim = pq.p.raw_dim();

        RungeKuttaIntegrator {
            params,
            func,
            t,
            pq,
            k1: PhaseSpaceVector::new(Array::zeros(dim), Array::zeros(dim)),
            k2: PhaseSpaceVector::new(Array::zeros(dim), Array::zeros(dim)),
            k3: PhaseSpaceVector::new(Array::zeros(dim), Array::zeros(dim)),
            k4: PhaseSpaceVector::new(Array::zeros(dim), Array::zeros(dim)),
        }
    }

    pub fn state(&self) -> (f32, PhaseSpaceVector) {
        (self.t, self.pq.clone())
    }
}

impl Iterator for RungeKuttaIntegrator {
    type Item = (f32, PhaseSpaceVector);

    fn next(&mut self) -> Option<(f32, PhaseSpaceVector)> {
        if self.params.tmax.is_some_and(|tmax| tmax >= self.t) {
            None
        } else {
            let h = self.params.step_size;
            let h2 = h * 0.5;
            let f = self.func;
            let t = self.t;
            let pq = &self.pq;

            f(t, pq, &mut self.k1);
            f(t + h2, &(pq + h2 * &self.k1), &mut self.k2);
            f(t + h2, &(pq + h2 * &self.k2), &mut self.k3);
            f(t + h, &(pq + h * &self.k3), &mut self.k4);

            self.t += h;

            self.pq = pq + h / 6.0 * (&self.k1 + 2.0 * &self.k2 + 2.0 * &self.k3 + &self.k4);
            Some(self.state())
        }
    }
}
