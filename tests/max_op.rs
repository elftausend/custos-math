use custos::CPU;
use custos_math::{Matrix, MaxOps};

#[test]
fn test_max_ops() {
    let device = CPU::new();
    let a = Matrix::from((
        &device,
        (3, 3),
        [-10., -2., -3., -4., -5., -6., -7., -8., -9.],
    ));

    let res = device.max(&a);
    assert!(res == -2.);

    let res = device.max_cols(&a);
    assert_eq!(res.read(), vec![-2., -4., -7.]);

    let res = device.max_rows(&a);
    assert_eq!(res.read(), vec![-4., -2., -3.]);
}

#[cfg(feature = "opencl")]
#[test]
fn test_max_cl() -> custos::Result<()> {
    let device = custos::OpenCL::new(0).unwrap();

    let a = Matrix::from((
        &device,
        (3, 3),
        [-10f32, -2., -3., -4., -5., -6., -7., -8., -9.],
    ));

    let res = device.max(&a);
    assert!(res == -2.);

    let res = device.max_cols(&a);
    assert_eq!(res.read(), vec![-2., -4., -7.]);

    let res = device.max_rows(&a);
    assert_eq!(res.read(), vec![-4., -2., -3.]);
    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn test_max_cuda() -> custos::Result<()> {
    let device = custos::CUDA::new(0)?;
    let a = Matrix::from((&device, (3, 3), [1f32, 2., -3., 4., 5., 6., 7., -8., 9.]));
    let max = device.max(&a);
    assert_eq!(max, 9.);

    let max_cols = device.max_cols(&a);
    assert_eq!(max_cols.read(), vec![2., 6., 9.]);

    let max_rows = device.max_rows(&a);
    assert_eq!(max_rows.read(), vec![7., 5., 9.,]);
    Ok(())
}
