use custos_math::Matrix;

#[cfg(feature = "cpu")]
#[test]
fn test_relu_mut() {
    let device = custos::CPU::new();

    let mut mat = Matrix::from((&device, 2, 2, [-1., -2., 3., 0.3]));
    mat.relu_mut();

    assert_eq!(mat.read(), [0., 0., 3., 0.3]);
}

#[cfg(feature = "opencl")]
#[test]
fn test_relu_mut_cl() -> custos::Result<()> {
    let device = custos::OpenCL::new(0)?;

    let mut mat = Matrix::from((&device, 2, 2, [-1., -2., 3., 0.3]));
    mat.relu_mut();

    assert_eq!(mat.read(), [0., 0., 3., 0.3]);

    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn test_relu_mut_cu() -> custos::Result<()> {
    let device = custos::CUDA::new(0)?;

    let mut mat = Matrix::from((&device, 2, 2, [-1., -2., 3., 0.3]));
    // may not work
    mat.relu_mut();

    assert_eq!(mat.read(), [0., 0., 3., 0.3]);

    Ok(())
}

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

    let device = custos::OpenCL::new(0).unwrap();

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

    let device = custos::CUDA::new(0).unwrap();

    let x = Matrix::from((&device, (1, 5), [-1.31f32, 2.12, -0.68, 5., 4.]));
    let res = device.relu(&x);
    assert_eq!(res.read(), [0., 2.12, 0., 5., 4.]);

    let res = device.relu_grad(&x);
    assert_eq!(res.read(), [0., 1., 0., 1., 1.]);
}


#[cfg(feature = "cuda")]
#[test]
fn test_tanh_cuda() {
    use custos_math::{nn::ActivationOps, Matrix};

    let device = custos::CUDA::new(0).unwrap();

    let x = Matrix::from((&device, (1, 5), [-1.31f32, 2.12, -0.68, 5., 4.]));
    let _res = device.tanh(&x);

    let _res = device.tanh_grad(&x);

}