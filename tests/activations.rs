use custos::{libs::cpu::CPU, AsDev, Matrix};
use custos_math::ActivationOps;



#[test]
fn test_relu() {
    let device = CPU::new().select();

    let x = Matrix::from((&device, (1, 5), [-1.31, 2.12, -0.68, 5., 4.,]));
    let res = device.relu(x);
    assert_eq!(res.read(), [0., 2.12, 0., 5., 4.]);

    let res = device.relu_grad(x);
    assert_eq!(res.read(), [0., 1., 0., 1., 1.]);
}