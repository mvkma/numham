use crate::physics::PhaseSpaceVector;
use crate::rungekutta::*;
use crate::visual::Stage;
use miniquad::conf;
use ndarray::{array, Array1};
use physics::TwoBodyHamiltonian;

mod physics;
mod rungekutta;
mod visual;

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

#[allow(dead_code)]
fn ham_eom_isoceles_three_body(_t: f32, pq: &Array1<f32>, dest: &mut Array1<f32>) {
    let m: f32 = 1.0;
    let m3: f32 = 1.5;

    let alpha = m / m3;
    let x1x2 = f32::powi(pq[3], 2) + f32::powi(pq[4], 2);

    dest[0] = -alpha * pq[3] / f32::powf(x1x2, 3.0 / 2.0)
        - 4.0 * pq[3] / f32::powf(x1x2 + (1.0 + 2.0 * alpha) * f32::powi(pq[5], 2), 3.0 / 2.0);
    dest[1] = -alpha * pq[4] / f32::powf(x1x2, 3.0 / 2.0)
        - 4.0 * pq[4] / f32::powf(x1x2 + (1.0 + 2.0 * alpha) * f32::powi(pq[5], 2), 3.0 / 2.0);
    dest[2] = -4.0 * (1.0 + 2.0 * alpha) * pq[4]
        / f32::powf(x1x2 + (1.0 + 2.0 * alpha) * f32::powi(pq[5], 2), 3.0 / 2.0);
    dest[3] = pq[0];
    dest[4] = pq[1];
    dest[4] = pq[2];
}

fn main() {
    let twobodyham = TwoBodyHamiltonian::new(1.0, 1.5);

    let params = IntegrationParams {
        step_size: 0.01,
        t0: 0.0,
        pq0: PhaseSpaceVector::new(array![0.0, 0.2, 0.0, -0.5], array![0.3, 0.0, -0.4, 0.0]),
        tmax: None,
    };

    let rk4 = RungeKuttaIntegrator::new(params, Box::new(twobodyham));

    let conf = conf::Conf::default();

    miniquad::start(conf, move || Box::new(Stage::new(rk4)));
}
