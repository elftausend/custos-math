use custos::{cpu::CPU, opencl::CLDevice, AsDev, Matrix};
use custos_math::ColOp;

#[test]
fn test_col_op() {
    let device = CPU::new().select();

    let a = Matrix::from((&device, (3, 3), [1., 2., 3., 4., 5., 6., 7., 8., 9.]));
    let b = Matrix::from((&device, (3, 1), [1., 2., 3.]));

    let c = device.add_col(&a, &b);
    assert_eq!(c.read(), vec![2., 3., 4., 6., 7., 8., 10., 11., 12.]);

    let device = CLDevice::new(0).unwrap().select();

    let a = Matrix::from((&device, (3, 3), [1., 2., 3., 4., 5., 6., 7., 8., 9.]));
    let b = Matrix::from((&device, (3, 1), [1., 2., 3.]));

    let c = device.add_col(&a, &b);
    assert_eq!(c.read(), vec![2., 3., 4., 6., 7., 8., 10., 11., 12.]);
}
