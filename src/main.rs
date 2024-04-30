#![feature(test)]

extern crate test;

use crate::physics::PhaseSpaceVector;
use crate::rungekutta::*;
use crate::visual::Stage;
use miniquad::conf;
use ndarray::{array, Array1};
use physics::{ThreeBodyHamiltonian, TwoBodyHamiltonian};
use visual::StageConf;

mod physics;
mod rungekutta;
mod visual;

#[allow(dead_code)]
fn ham_eom_1d_harmonic_oscillator(_t: f64, pq: &Array1<f64>, dest: &mut Array1<f64>) {
    // H = p^2 / (2 * m) + k * q^2 / 2
    // \partial_q H = k * q
    // \partial_p H = p / m
    let m: f64 = 1.0;
    let k: f64 = 0.5;

    // array![-k * pq[1], pq[0] / m]
    dest[0] = -k * pq[1];
    dest[1] = pq[0] / m;
}

#[allow(dead_code)]
fn ham_eom_3d_harmonic_oscillator(_t: f64, pq: &Array1<f64>, dest: &mut Array1<f64>) {
    let m: f64 = 1.0;
    let k: f64 = 0.5;

    dest[0] = -k * pq[3];
    dest[1] = -k * pq[4];
    dest[2] = 0.0;
    dest[3] = pq[0] / m;
    dest[4] = pq[1] / m;
    dest[5] = pq[2] / m;
}

#[allow(dead_code)]
fn ham_eom_isoceles_three_body(_t: f64, pq: &Array1<f64>, dest: &mut Array1<f64>) {
    let m: f64 = 1.0;
    let m3: f64 = 1.5;

    let alpha = m / m3;
    let x1x2 = f64::powi(pq[3], 2) + f64::powi(pq[4], 2);

    dest[0] = -alpha * pq[3] / f64::powf(x1x2, 3.0 / 2.0)
        - 4.0 * pq[3] / f64::powf(x1x2 + (1.0 + 2.0 * alpha) * f64::powi(pq[5], 2), 3.0 / 2.0);
    dest[1] = -alpha * pq[4] / f64::powf(x1x2, 3.0 / 2.0)
        - 4.0 * pq[4] / f64::powf(x1x2 + (1.0 + 2.0 * alpha) * f64::powi(pq[5], 2), 3.0 / 2.0);
    dest[2] = -4.0 * (1.0 + 2.0 * alpha) * pq[4]
        / f64::powf(x1x2 + (1.0 + 2.0 * alpha) * f64::powi(pq[5], 2), 3.0 / 2.0);
    dest[3] = pq[0];
    dest[4] = pq[1];
    dest[4] = pq[2];
}

fn main() {
    // let twobodyham = TwoBodyHamiltonian::new(1.0, 1.5);

    // let params = IntegrationParams {
    //     step_size: 0.01,
    //     t0: 0.0,
    //     pq0: PhaseSpaceVector::new(array![0.0, 0.2, 0.0, -0.5], array![0.3, 0.0, -0.4, 0.0]),
    //     tmax: None,
    // };

    // let rk4 = RungeKuttaIntegrator::new(params, Box::new(twobodyham));

    let threebodyham = ThreeBodyHamiltonian::new(1.0, 1.0);

    // let v1 = 0.3471168881;
    // let v2 = 0.5327249454;
    // let v1 = 0.3068934205;
    // let v2 = 0.1255065670;
    // let v1 = 0.6150407229;
    // let v2 = 0.5226158545;
    let v1 = 0.5379557207;
    let v2 = 0.3414578545;
    // let v1 = 0.4112926910;
    // let v2 = 0.2607551013;
    let params = IntegrationParams {
        step_size: 0.001,
        t0: 0.0,
        pq0: PhaseSpaceVector::new(
            array![v1, v2, v1, v2, -2.0 * v1, -2.0 * v2],
            array![-1.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        ),
        tmax: None,
    };

    let rk4 = RungeKuttaIntegrator::new(params, Box::new(threebodyham));

    let stage_conf = StageConf {
        scale: 2.0,
        steps_per_frame: 20,
        trail_length: 5000,
    };

    let conf = conf::Conf {
        window_height: 900,
        window_width: 900,
        ..Default::default()
    };

    miniquad::start(conf, move || Box::new(Stage::new(stage_conf, rk4)));
}

#[cfg(test)]
mod tests {
    use super::*;
    use test::Bencher;

    #[bench]
    fn bench_rk4_next(b: &mut Bencher) {
        let v1 = 0.5379557207;
        let v2 = 0.3414578545;

        let params = IntegrationParams {
            step_size: 0.001,
            t0: 0.0,
            pq0: PhaseSpaceVector::new(
                array![v1, v2, v1, v2, -2.0 * v1, -2.0 * v2],
                array![-1.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            ),
            tmax: None,
        };

        let threebodyham = ThreeBodyHamiltonian::new(1.0, 1.0);
        let mut rk4 = RungeKuttaIntegrator::new(params, Box::new(threebodyham));

        b.iter(|| {
            rk4.next();
        })
    }
}
