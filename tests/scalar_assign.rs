use custos::CPU;
use custos_math::Matrix;

#[test]
fn test_scalar_assign() {
    let device = CPU::new();

    let mut buf = Matrix::from((&device, 2, 2, [1., 2., 3., 4.]));
    buf += 1.;
    assert_eq!(buf.as_slice(), &[2., 3., 4., 5.,]);
}

#[cfg(feature = "opencl")]
#[test]
fn test_scalar_assign_cl() -> custos::Result<()> {
    use custos::OpenCL;

    let device = OpenCL::new(0)?;

    let mut buf = Matrix::from((&device, 2, 2, [1., 2., 3., 4.]));
    buf += 1.;
    assert_eq!(buf.read(), vec![2., 3., 4., 5.,]);
    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn test_scalar_assign_cu() -> custos::Result<()> {
    use custos::CudaDevice;

    let device = CudaDevice::new(0)?;

    let mut buf = Matrix::from((&device, 2, 2, [1., 2., 3., 4.]));
    buf += 1.;
    assert_eq!(buf.read(), vec![2., 3., 4., 5.,]);
    Ok(())
}
