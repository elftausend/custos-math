use custos::{cpu::CPU, Matrix, AsDev, opencl::CLDevice};
use custos_math::AdditionalOps;

#[test]
fn test_scalar() {
    let device = CPU::new().select();
    let x = Matrix::from((&device, (1, 5), [-1.31, 2.12, 1., 5., 4.,]));

    let res = device.adds(x, 2.0);
    assert_eq!(res.read(), vec![0.69, 4.12, 3., 7., 6.]);


    let device = CLDevice::get(0).unwrap().select();
    let x = Matrix::from((&device, (1, 5), [-1.31, 2.12, 1., 5., 4.,]));

    let res = device.adds(x, 2.0);
    assert_eq!(res.read(), vec![0.69, 4.12, 3., 7., 6.]);
}