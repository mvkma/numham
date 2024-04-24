use ndarray::{array, Array, Array1};

struct IntegrationParams {
    step_size: f32,
    t0: f32,
    pq0: Array1<f32>,
    tmax: f32,
}

struct RungeKuttaIntegrator {
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
    fn new(
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

#[allow(dead_code)]
fn ham_eom_1d_harmonic_oscillator(_t: f32, pq: &Array1<f32>, dest: &mut Array1<f32>) {
    // H = p^2 / (2 * m) + k * q^2 / 2
    // \partial_q H = k * q
    // \partial_p H = p / m
    let m: f32 = 1.0;
    let k: f32 = 0.5;

    // array![-k * pq[1], pq[0] / m]
    dest[0] = -k * pq[1];
    dest[1] = pq[0] / m;
}

#[allow(dead_code)]
fn ham_eom_3d_harmonic_oscillator(_t: f32, pq: &Array1<f32>, dest: &mut Array1<f32>) {
    let m: f32 = 1.0;
    let k: f32 = 0.5;

    dest[0] = -k * pq[3];
    dest[1] = -k * pq[4];
    dest[2] = 0.0;
    dest[3] = pq[0] / m;
    dest[4] = pq[1] / m;
    dest[5] = pq[2] / m;
}

fn main() {
    // let params = IntegrationParams {
    //     step_size: 0.01,
    //     t0: 0.0,
    //     pq0: array![0.0, 1.0],
    //     tmax: 10.0,
    // };

    // let rk4 = RungeKuttaIntegrator::new(params, ham_eom_1d_harmonic_oscillator);

    let params = IntegrationParams {
        step_size: 0.01,
        t0: 0.0,
        pq0: array![0.0, 0.5, 0.8, -0.3, 0.4, 0.0],
        tmax: 10.0,
    };

    let rk4 = RungeKuttaIntegrator::new(params, ham_eom_3d_harmonic_oscillator);

    for state in rk4 {
        println!("{} {}", state.0, state.1);
    }
}
