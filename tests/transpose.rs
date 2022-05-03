use custos::{cpu::CPU, opencl::CLDevice, AsDev, Matrix};
use custos_math::Transpose;

#[test]
fn test_transpose_cpu() {
    let device = CPU::new().select();

    let a = Matrix::from((&device, (2, 3), [1., 2., 3., 4., 5., 6.]));

    let res = a.T();
    assert_eq!(vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0], res.read());
}

#[test]
fn test_transpose_cl() {
    let device = CLDevice::get(0).unwrap().select();

    let a = Matrix::from((&device, (2, 3), [6f32, 5., 4., 3., 2., 1.]));
    let res = a.T();

    assert_eq!(vec![6.0, 3.0, 5.0, 2.0, 4.0, 1.0], res.read());
}

#[cfg(not(target_os = "macos"))]
#[test]
fn test_transpose_cl_f64() {
    let device = CLDevice::get(0).unwrap().select();

    let a = Matrix::from((&device, (2, 3), [6f64, 5., 4., 3., 2., 1.]));
    let res = a.T();

    assert_eq!(vec![6.0, 3.0, 5.0, 2.0, 4.0, 1.0], res.read());
}
