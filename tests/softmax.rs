use custos::{CPU, AsDev, Matrix, range, CLDevice};
use custos_math::nn::{cce_grad, Softmax};

#[test]
fn test_softmax_cpu() {
    let device = CPU::new().select();

    let targets = Matrix::<f32>::from((&device, (2, 3), 
        [0., 0., 1.,
        1., 0., 0.])
    );

    let activated = Matrix::from((&device, (2, 3), 
        [0.1, 0.1, 0.8,
         0.9, 0.05, 0.05])
    );

    let grads = cce_grad(&device, activated, targets);

    for _ in range(1000) {
        device.softmax_grad(activated, grads);
    }
}

#[test]
fn test_softmax_cl() {
    let device = CLDevice::get(0).unwrap().select();


    let targets = Matrix::<f32>::from((&device, (2, 3), 
        [0., 0., 1.,
        1., 0., 0.])
    );

    let activated = Matrix::from((&device, (2, 3), 
        [0.1, 0.1, 0.8,
         0.9, 0.05, 0.05])
    );

    let grads = cce_grad(&device, activated, targets);

    for _ in range(1000) {
        device.softmax_grad(activated, grads);
    }
}