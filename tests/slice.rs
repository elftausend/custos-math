use custos::cpu::CPU;
use custos_math::Matrix;

#[cfg(feature = "cuda")]
use custos_math::SliceOps;

#[test]
fn test_slice_cpu() {
    let device = CPU::new();

    let source = Matrix::from((&device, (3, 2), [1., 2., 3., 4., 5., 6.]));
    let slice = device.slice(&source, .., 0..1);
    assert_eq!(vec![1., 3., 5.], slice.read());
}

#[cfg(feature = "opencl")]
#[test]
fn test_slice_cl() {
    let device = custos::OpenCL::new(0).unwrap();

    let source = Matrix::from((&device, (3, 2), [1., 2., 3., 4., 5., 6.]));
    let slice = device.slice(&source, .., 0..1);
    assert_eq!(vec![1., 3., 5.], slice.read());
}

#[cfg(feature = "cuda")]
#[test]
fn test_slice_cuda() -> custos::Result<()> {
    let device = custos::CUDA::new(0)?;

    let source = Matrix::from((&device, (3, 2), [1., 2., 3., 4., 5., 6.]));
    let slice = device.slice(&source, .., 0..1);
    assert_eq!(vec![1., 3., 5.], slice.read());
    Ok(())
}
