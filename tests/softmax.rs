use custos::{range, AsDev, CLDevice, Matrix, CPU};
use custos_math::nn::{cce_grad, SoftmaxOps, cl_softmax};

#[test]
fn test_softmax_cpu() {
    let device = CPU::new().select();

    let targets = Matrix::<f32>::from((&device, (2, 3), [0., 0., 1., 1., 0., 0.]));
    let activated = Matrix::from((&device, (2, 3), [0.1, 0.1, 0.8, 0.9, 0.05, 0.05]));

    let grads = cce_grad(&device, &activated, &targets);

    println!("hi");
    for _ in range(1000) {
        device.softmax_grad(&activated, &grads);
    }
}

#[test]
fn test_softmax_cl() {
    let device = CLDevice::new(0).unwrap().select();

    let targets = Matrix::<f32>::from((&device, (2, 3), [0., 0., 1., 1., 0., 0.]));
    let activated = Matrix::from((&device, (2, 3), [0.1, 0.1, 0.8, 0.9, 0.05, 0.05]));

    let grads = cce_grad(&device, &activated, &targets);

    for _ in range(1000) {
        device.softmax_grad(&activated, &grads);
    }
}

#[test]
fn test_softmax_kernel_cl() -> custos::Result<()> {
    let device = CLDevice::new(0)?.select();

    let targets = Matrix::<f32>::from((&device, (2, 3), [0., 0., 1., 1., 0., 0.]));
    let activated = Matrix::from((&device, (2, 3), [0.1, 0.1, 0.8, 0.9, 0.05, 0.05]));

    let grads = cce_grad(&device, &activated, &targets);

    let _out = device.softmax_grad(&activated, &grads);
    //println!("out: {out:?}");

    //println!("grads: {:?}", grads.dims());
    let _out = cl_softmax(&device, activated, &grads)?;
    //println!("out: {out:?}");
    Ok(())
}

