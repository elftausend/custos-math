use custos::{number::Float, CPU};
use custos_math::{FnsOps, Matrix};

pub fn roughly_equals<T: Float>(lhs: &[T], rhs: &[T], diff: T) {
    for (a, b) in lhs.iter().zip(rhs) {
        let abs = (*a - *b).abs();
        if abs > diff {
            panic!(
                "\n left: '{:?}',\n right: '{:?}', \n left elem.: {} != right elem. {}",
                lhs, rhs, a, b
            )
        }
    }
}

#[cfg(feature = "cpu")]
#[test]
fn test_fns_cpu() {
    let device = CPU::new();

    let x = Matrix::from((&device, (1, 4), [1.5, 2., 6., -3.]));

    let res = device.neg(&x);
    assert_eq!(res.read(), vec![-1.5, -2., -6., 3.]);

    let res = device.exp(&x);
    roughly_equals(&res.read(), &[4.4816, 7.3891, 403.4288, 0.04979], 0.001);

    let res = device.ln(&x);
    roughly_equals(&res.read(), &[0.405, 0.6931, 1.7917, f64::NAN], 0.001);

    let res = device.powf(&x, 2.);
    roughly_equals(&res.read(), &[2.25, 4., 36., 9.], 0.001);
}

#[cfg(feature = "opencl")]
#[test]
fn test_fns_cl() -> Result<(), custos::Error> {
    let device = custos::OpenCL::new(0)?;

    let x = Matrix::from((&device, (1, 4), [1.5, 2., 6., -3.]));

    let res = device.neg(&x);
    assert_eq!(res.read(), vec![-1.5, -2., -6., 3.]);

    let res = device.exp(&x);
    roughly_equals(&res.read(), &[4.4816, 7.3891, 403.4288, 0.04979], 0.001);

    let res = device.ln(&x);
    roughly_equals(&res.read(), &[0.405, 0.6931, 1.7917, f32::NAN], 0.001);

    let res = device.powf(&x, 2.);
    roughly_equals(&res.read(), &[2.25, 4., 36., 9.], 0.001);

    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn test_fns_cuda() -> Result<(), custos::Error> {
    let device = custos::CUDA::new(0)?;

    let x = Matrix::from((&device, (1, 4), [1.5, 2., 6., -3.]));

    let res = device.neg(&x);
    assert_eq!(res.read(), vec![-1.5, -2., -6., 3.]);

    let res = device.exp(&x);
    roughly_equals(&res.read(), &[4.4816, 7.3891, 403.4288, 0.04979], 0.001);

    let res = device.ln(&x);
    roughly_equals(&res.read(), &[0.405, 0.6931, 1.7917, f32::NAN], 0.001);

    let res = device.powf(&x, 2.);
    roughly_equals(&res.read(), &[2.25, 4., 36., 9.], 0.001);

    Ok(())
}
