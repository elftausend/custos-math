use custos::cpu::CPU;
use custos_math::{ColOp, Matrix};

#[test]
fn test_col_op() {
    let device = CPU::new();

    let a = Matrix::from((&device, (3, 3), [1., 2., 3., 4., 5., 6., 7., 8., 9.]));
    let b = Matrix::from((&device, (3, 1), [1., 2., 3.]));

    let c = device.add_col(&a, &b);
    assert_eq!(c.read(), vec![2., 3., 4., 6., 7., 8., 10., 11., 12.]);
}

#[cfg(feature = "opencl")]
#[test]
fn test_col_op_cl() {
    let device = custos::OpenCL::new(0).unwrap();

    let a = Matrix::from((&device, (3, 3), [1., 2., 3., 4., 5., 6., 7., 8., 9.]));
    let b = Matrix::from((&device, (3, 1), [1., 2., 3.]));

    let c = device.add_col(&a, &b);
    assert_eq!(c.read(), vec![2., 3., 4., 6., 7., 8., 10., 11., 12.]);
}
