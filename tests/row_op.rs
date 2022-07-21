use custos::{cpu::CPU, AsDev};
use custos_math::{row_op, Matrix, RowOp};

#[test]
fn test_row_op() {
    let device = CPU::new().select();

    let a = Matrix::from((&device, (3, 3), [1., 2., 3., 4., 5., 6., 7., 8., 9.]));
    let b = Matrix::from((&device, (1, 3), [1., 2., 3.]));

    let c = row_op(&device, &a, &b, |c, a, b| *c = a + b);
    assert_eq!(c.read(), vec![2., 4., 6., 5., 7., 9., 8., 10., 12.]);
}

#[cfg(feature = "opencl")]
#[test]
fn test_row_op_cl() -> custos::Result<()> {
    use custos_math::RowOp;
    let device = custos::CLDevice::new(0)?.select();

    let a = Matrix::from((&device, (3, 3), [1., 2., 3., 4., 5., 6., 7., 8., 9.]));
    let b = Matrix::from((&device, (1, 3), [1., 2., 3.]));

    let c = device.add_row(&a, &b);
    assert_eq!(c.read(), vec![2., 4., 6., 5., 7., 9., 8., 10., 12.]);
    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn test_row_op_cu() -> custos::Result<()> {
    use custos_math::RowOp;
    let device = custos::CudaDevice::new(0)?.select();

    let a = Matrix::from((&device, (3, 3), [1., 2., 3., 4., 5., 6., 7., 8., 9.]));
    let b = Matrix::from((&device, (1, 3), [1., 2., 3.]));

    let c = device.add_row(&a, &b);
    assert_eq!(c.read(), vec![2., 4., 6., 5., 7., 9., 8., 10., 12.]);
    Ok(())
}

#[test]
fn test_row_op_mut() {
    let device = CPU::new().select();

    let mut a = Matrix::from((&device, (3, 3), [1., 2., 3., 4., 5., 6., 7., 8., 9.]));
    let b = Matrix::from((&device, (1, 3), [1., 2., 3.]));

    device.add_row_mut(&mut a, &b);
    assert_eq!(a.as_slice(), &[2., 4., 6., 5., 7., 9., 8., 10., 12.]);
}

#[cfg(feature="opencl")]
#[test]
fn test_row_op_mut_cl() -> custos::Result<()> {
    let device = custos::CLDevice::new(0)?.select();

    let mut a = Matrix::from((&device, (3, 3), [1., 2., 3., 4., 5., 6., 7., 8., 9.]));
    let b = Matrix::from((&device, (1, 3), [1., 2., 3.]));

    device.add_row_mut(&mut a, &b);
    assert_eq!(a.read(), vec![2., 4., 6., 5., 7., 9., 8., 10., 12.]);
    Ok(())
}

#[cfg(feature="cuda")]
#[test]
fn test_row_op_mut_cu() -> custos::Result<()> {
    let device = custos::CudaDevice::new(0)?.select();

    let mut a = Matrix::from((&device, (3, 3), [1., 2., 3., 4., 5., 6., 7., 8., 9.]));
    let b = Matrix::from((&device, (1, 3), [1., 2., 3.]));

    device.add_row_mut(&mut a, &b);
    assert_eq!(a.read(), vec![2., 4., 6., 5., 7., 9., 8., 10., 12.]);
    Ok(())
}