use ndarray::{Array, Array1};

pub struct IntegrationParams {
    pub step_size: f32,
    pub t0: f32,
    pub pq0: Array1<f32>,
    pub tmax: f32,
}

pub struct RungeKuttaIntegrator {
    params: IntegrationParams,
    func: fn(f32, &Array1<f32>, &mut Array1<f32>),
    t: f32,
    pq: Array1<f32>,
    k1: Array1<f32>,
    k2: Array1<f32>,
    k3: Array1<f32>,
    k4: Array1<f32>,
}

impl RungeKuttaIntegrator {
    pub fn new(
        params: IntegrationParams,
        func: fn(f32, &Array1<f32>, &mut Array1<f32>),
    ) -> RungeKuttaIntegrator {
        let t = params.t0;
        let pq = params.pq0.clone();
        let dim = pq.raw_dim();

        RungeKuttaIntegrator {
            params,
            func,
            t,
            pq,
            k1: Array::zeros(dim),
            k2: Array::zeros(dim),
            k3: Array::zeros(dim),
            k4: Array::zeros(dim),
        }
    }

    pub fn state(&self) -> (f32, Array1<f32>) {
        (self.t, self.pq.clone())
    }
}

impl Iterator for RungeKuttaIntegrator {
    type Item = (f32, Array1<f32>);

    fn next(&mut self) -> Option<(f32, Array1<f32>)> {
        if self.t >= self.params.tmax {
            None
        } else {
            let h = self.params.step_size;
            let h2 = h * 0.5;
            let f = self.func;
            f(self.t, &self.pq, &mut self.k1);
            f(self.t + h2, &(&self.pq + h2 * &self.k1), &mut self.k2);
            f(self.t + h2, &(&self.pq + h2 * &self.k2), &mut self.k3);
            f(self.t + h, &(&self.pq + h * &self.k3), &mut self.k4);

            self.t += h;

            self.pq = &self.pq + h / 6.0 * (&self.k1 + 2.0 * &self.k2 + 2.0 * &self.k3 + &self.k4);
            Some(self.state())
        }
    }
}
