use custos::{Matrix, cpu::CPU, AsDev, number::Float};
use custos_math::FnsOps;

pub fn roughly_equals<T: Float>(lhs: &[T], rhs: &[T], diff: T) {
    for (a, b) in lhs.iter().zip(rhs) {
        let abs = (*a - *b).abs();
        if abs > diff {
            panic!("\n left: '{:?}',\n right: '{:?}', \n left elem.: {} != right elem. {}", lhs, rhs, a, b)
        }
    }
}

#[test]
fn test_fns_cpu() {
    let device = CPU::new().select();

    let x = Matrix::from((&device, (1, 4), [1.5, 2., 6., -3., ]));
    
    let res = device.neg(&x);
    assert_eq!(res.read(), vec![-1.5, -2., -6., 3.]);

    let res = device.exp(&x);
    roughly_equals(&res.read(), &[4.4816, 7.3891, 403.4288, 0.04979], 0.001);

    let res = device.ln(&x);
    roughly_equals(&res.read(), &[0.405, 0.6931, 1.7917, f64::NAN], 0.001);
}