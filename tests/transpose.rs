use custos::{cpu::CPU, AsDev};
use custos_math::{Matrix};

#[cfg(feature="cuda")]
use custos_math::TransposeOp;

#[test]
fn test_transpose_cpu() {
    let device = CPU::new().select();

    let a = Matrix::from((&device, (2, 3), [1., 2., 3., 4., 5., 6.]));

    let res = a.T();
    assert_eq!(vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0], res.read());
}

#[cfg(feature="opencl")]
#[test]
fn test_transpose_cl() {
    let device = custos::CLDevice::new(0).unwrap().select();

    let a = Matrix::from((&device, (2, 3), [6f32, 5., 4., 3., 2., 1.]));

    let res = a.T();
    assert_eq!(vec![6.0, 3.0, 5.0, 2.0, 4.0, 1.0], res.read());
}

#[cfg(feature="opencl")]
#[cfg(not(target_os = "macos"))]
#[test]
fn test_transpose_cl_f64() {
    let device = custos::CLDevice::new(0).unwrap().select();

    let a = Matrix::from((&device, (2, 3), [6f64, 5., 4., 3., 2., 1.]));
    let res = a.T();

    assert_eq!(vec![6.0, 3.0, 5.0, 2.0, 4.0, 1.0], res.read());
}

#[cfg(feature="cuda")]
#[test]
fn test_transpose_cuda() -> custos::Result<()> {
    let device = custos::CudaDevice::new(0)?.select();

    let a = Matrix::from((&device, 2, 3, [1f32, 2., 3.,
                                                     4., 5., 6.,]));
    let out = device.transpose(&a);
    assert_eq!(vec![1., 4., 2., 5., 3., 6.,], out.read());

    Ok(())
}

#[cfg(feature="cuda")]
#[test]
fn test_transpose_selected_cuda() -> custos::Result<()> {
    let device = custos::CudaDevice::new(0)?.select();

    let a = Matrix::from((&device, (2, 3), [6f32, 5., 4., 3., 2., 1.]));
    let res = a.T();

    assert_eq!(vec![6.0, 3.0, 5.0, 2.0, 4.0, 1.0], res.read());
    Ok(())
}