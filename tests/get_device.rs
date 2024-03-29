/*
#[cfg(feature = "opencl")]
use custos::devices::opencl::OpenCL;
use custos::{devices::cpu::CPU, get_device, AsDev, Error, VecRead};
use custos_math::{BaseOps, Matrix};

#[test]
fn test_matrix_read_cpu() -> Result<(), Error> {
    let device = CPU::new();

    let read = get_device!(device.dev(), VecRead<f32>);

    let matrix = Matrix::from((&device, (2, 3), [1.51, 6.123, 7., 5.21, 8.62, 4.765]));
    let read = read.read(matrix.as_buf());
    assert_eq!(&read, &[1.51, 6.123, 7., 5.21, 8.62, 4.765]);
    Ok(())
}

#[cfg(feature = "opencl")]
#[test]
fn test_matrix_read_cl() -> Result<(), Error> {
    let device = OpenCL::new(0)?;

    let read = get_device!(device.dev(), VecRead<f32>);

    let matrix = Matrix::from((&device, (2, 3), [1.51, 6.123, 7., 5.21, 8.62, 4.765]));
    let read = read.read(matrix.as_buf());
    assert_eq!(&read, &[1.51, 6.123, 7., 5.21, 8.62, 4.765]);

    Ok(())
}

#[test]
fn test_baseops() -> Result<(), Error> {
    let device = CPU::new();

    let matrix = Matrix::from((&device, (2, 3), [1.51, 6.123, 7., 5.21, 8.62, 4.765]));
    let b = Matrix::from((&device, (2, 3), [1., 1., 1., 1., 1., 1.]));

    let base_ops = get_device!(device.dev(), BaseOps<f32>);
    let out = base_ops.add(&matrix, &b);

    assert_eq!(out.read(), vec![2.51, 7.123, 8., 6.21, 9.62, 5.765]);
    Ok(())
}
*/
