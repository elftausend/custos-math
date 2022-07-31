use custos::{range, CPU};
use custos_math::{
    nn::{cce_grad, SoftmaxOps},
    Matrix,
};

#[test]
fn test_softmax_cpu() {
    let device = CPU::new();

    let targets = Matrix::<f32>::from((&device, (2, 3), [0., 0., 1., 1., 0., 0.]));
    let activated = Matrix::from((&device, (2, 3), [0.1, 0.1, 0.8, 0.9, 0.05, 0.05]));

    let grads = cce_grad(&device, &activated, &targets);

    println!("hi");
    for _ in range(1000) {
        device.softmax_grad(&activated, &grads);
    }
}

#[cfg(feature = "opencl")]
#[test]
fn test_softmax_cl() -> custos::Result<()> {
    let device = custos::CLDevice::new(0)?;

    let targets = Matrix::<f32>::from((&device, (2, 3), [0., 0., 1., 1., 0., 0.]));
    let activated = Matrix::from((&device, (2, 3), [0.1, 0.1, 0.8, 0.9, 0.05, 0.05]));

    let grads = cce_grad(&device, &activated, &targets);

    for _ in range(1000) {
        device.softmax_grad(&activated, &grads);
    }
    Ok(())
}

#[cfg(feature = "opencl")]
#[test]
fn test_softmax_kernel_cl() -> custos::Result<()> {
    let device = custos::CLDevice::new(0)?;

    let targets = Matrix::<f32>::from((&device, (2, 3), [0., 0., 1., 1., 0., 0.]));
    let activated = Matrix::from((&device, (2, 3), [0.1, 0.1, 0.8, 0.9, 0.05, 0.05]));

    let grads = cce_grad(&device, &activated, &targets);

    let _out = device.softmax_grad(&activated, &grads);
    //println!("out: {out:?}");

    //println!("grads: {:?}", grads.dims());
    let _out = custos_math::nn::cl_softmax(&device, activated, &grads)?;
    //println!("out: {out:?}");
    Ok(())
}
