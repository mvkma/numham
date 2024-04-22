use ndarray::{arr1, array, Array1};

fn ham_eom_1d_harmonic_oscillator(_t: f32, pq: &Array1<f32>) -> Array1<f32> {
    // H = p^2 / (2 * m) + k * q^2 / 2
    // \partial_q H = k * q
    // \partial_p H = p / m
    let m: f32 = 1.0;
    let k: f32 = 0.5;

    array![-k * pq[1], pq[0] / m]
}

fn runge_kutta(h: f32, t0: f32, pq0: Array1<f32>, tmax: f32) {
    let mut pq = pq0;
    let mut t = t0;

    let mut k1: Array1<f32>;
    let mut k2: Array1<f32>;
    let mut k3: Array1<f32>;
    let mut k4: Array1<f32>;

    let h2 = h * 0.5;

    while tmax >= t {
        println!("{} {} {}", t, pq[0], pq[1]);
        k1 = ham_eom_1d_harmonic_oscillator(t, &pq);
        k2 = ham_eom_1d_harmonic_oscillator(t + h2, &(&pq + h2 * &k1));
        k3 = ham_eom_1d_harmonic_oscillator(t + h2, &(&pq + h2 * &k2));
        k4 = ham_eom_1d_harmonic_oscillator(t + h, &(&pq + h * &k3));

        pq = &pq + h / 6.0 * (&k1 + 2.0 * &k2 + 2.0 * &k3 + &k4);

        t += h;
    }
    println!("{} {} {}", t, pq[0], pq[1]);
}

fn main() {
    runge_kutta(0.01, 0.0, array![0.0, 1.0], 10.0);
}
