use ndarray::{array, Array, Array1};

struct IntegrationParams {
    step_size: f32,
    t0: f32,
    tmax: f32,
}

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

fn runge_kutta(params: IntegrationParams, pq0: Array1<f32>) {
    let h = params.step_size;
    let tmax = params.tmax;
    let mut pq = pq0;
    let mut t = params.t0;

    let mut k1 = Array::zeros(pq.raw_dim());
    let mut k2 = Array::zeros(pq.raw_dim());
    let mut k3 = Array::zeros(pq.raw_dim());
    let mut k4 = Array::zeros(pq.raw_dim());

    let h2 = h * 0.5;

    while tmax >= t {
        println!("{} {} {}", t, pq[0], pq[1]);
        ham_eom_1d_harmonic_oscillator(t, &pq, &mut k1);
        ham_eom_1d_harmonic_oscillator(t + h2, &(&pq + h2 * &k1), &mut k2);
        ham_eom_1d_harmonic_oscillator(t + h2, &(&pq + h2 * &k2), &mut k3);
        ham_eom_1d_harmonic_oscillator(t + h, &(&pq + h * &k3), &mut k4);

        pq = &pq + h / 6.0 * (&k1 + 2.0 * &k2 + 2.0 * &k3 + &k4);

        t += h;
    }
    println!("{} {} {}", t, pq[0], pq[1]);
}

fn main() {
    let params = IntegrationParams {
        step_size: 0.01,
        t0: 0.0,
        tmax: 10.0,
    };

    runge_kutta(params, array![0.0, 1.0]);
}
