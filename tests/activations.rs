#[cfg(feature = "opencl")]
#[test]
fn test_relu() {
    use custos_math::{nn::ActivationOps, Matrix};

    let device = custos::CPU::new();

    let x = Matrix::from((&device, (1, 5), [-1.31, 2.12, -0.68, 5., 4.]));
    let res = device.relu(&x);
    assert_eq!(res.read(), [0., 2.12, 0., 5., 4.]);

    let res = device.relu_grad(&x);
    assert_eq!(res.read(), [0., 1., 0., 1., 1.]);

    let device = custos::CLDevice::new(0).unwrap();

    let x = Matrix::from((&device, (1, 5), [-1.31f32, 2.12, -0.68, 5., 4.]));
    let res = device.relu(&x);
    assert_eq!(res.read(), [0., 2.12, 0., 5., 4.]);

    let res = device.relu_grad(&x);
    assert_eq!(res.read(), [0., 1., 0., 1., 1.]);
}

#[cfg(feature = "cuda")]
#[test]
fn test_relu_cuda() {
    use custos_math::{nn::ActivationOps, Matrix};

    let device = custos::CudaDevice::new(0).unwrap();

    let x = Matrix::from((&device, (1, 5), [-1.31f32, 2.12, -0.68, 5., 4.]));
    let res = device.relu(&x);
    assert_eq!(res.read(), [0., 2.12, 0., 5., 4.]);

    let res = device.relu_grad(&x);
    assert_eq!(res.read(), [0., 1., 0., 1., 1.]);
}
