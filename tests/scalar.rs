use custos::{cpu::CPU, number::Float, AsDev, Matrix};
use custos_math::scalar_apply;

#[cfg(feature="cuda")]
use custos::VecRead;

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

#[cfg(feature="opencl")]
#[test]
fn test_scalar() {
    use custos_math::AdditionalOps;
    let device = CPU::new().select();
    let x = Matrix::from((&device, (1, 5), [-1.31, 2.12, 1., 5., 4.]));

    let res = device.adds(&x, 2.0);
    assert_eq!(res.read(), vec![0.69, 4.12, 3., 7., 6.]);

    let device = custos::CLDevice::new(0).unwrap().select();
    let x = Matrix::from((&device, (1, 5), [-1.31f32, 2.12, 1., 5., 4.]));

    let res = device.adds(&x, 2.0);
    roughly_equals(&res.read(), &[0.69, 4.12, 3., 7., 6.], 1E-5);
}

#[test]
fn test_scalar_apply() {
    let device = CPU::new().select();
    let x = Matrix::from((&device, (1, 5), [-1.31, 2.12, 1., 5., 4.]));

    let res = scalar_apply(&device, &x, 0., |c, a, _| *c = a.abs() + 1.);
    assert_eq!(res.read(), vec![2.31, 3.12, 2., 6., 5.,]);
}

#[cfg(feature="cuda")]
#[test]
fn test_scalar_op_cuda() -> custos::Result<()> {
    use custos_math::cu_scalar_op;
    let device = custos::CudaDevice::new(0)?.select();
    let x = Matrix::from((&device, (1, 5), [-1.31, 2.12, 1., 5., 4.]));

    let res = cu_scalar_op(&device, &x, 0.5, "+")?;
    assert_eq!(device.read(&res), vec![-0.81, 2.62, 1.5, 5.5, 4.5]);
    Ok(())
}