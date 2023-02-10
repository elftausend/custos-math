use custos::CPU;
use custos_math::{Matrix, SumOps, SumOverOps};

#[test]
fn test_sum_ops() {
    let device = CPU::new();
    let a = Matrix::from((
        &device,
        (3, 3),
        [-10., -2., -3., -4., -5., -6., -7., -8., -9.],
    ));

    let res = device.sum(&a);
    assert!(res == -54.);

    let res = device.mean(&a);
    assert!(res == -54. / a.size() as f32);

    let res: Matrix = device.sum_cols(&a);
    assert_eq!(res.read(), vec![-15., -15., -24.]);

    let res: Matrix = device.sum_rows(&a);
    assert_eq!(res.read(), vec![-21., -15., -18.]);
}

#[cfg(feature = "opencl")]
#[test]
fn test_sum_ops_cl() -> custos::Result<()> {
    let device = custos::OpenCL::new(0)?;

    let a = Matrix::from((
        &device,
        (3, 3),
        [-10f32, -2., -3., -4., -5., -6., -7., -8., -9.],
    ));

    let res = device.sum(&a);
    assert!(res == -54.);

    let res = device.mean(&a);
    assert!(res == -54. / a.size() as f32);

    let res = device.sum_cols(&a);
    assert_eq!(res.read(), vec![-15., -15., -24.]);

    let res = device.sum_rows(&a);
    assert_eq!(res.read(), vec![-21., -15., -18.]);
    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn test_sum_ops_cu() -> custos::Result<()> {
    let device = custos::CUDA::new(0)?;

    let a = Matrix::from((
        &device,
        (3, 3),
        [-10f32, -2., -3., -4., -5., -6., -7., -8., -9.],
    ));

    let res = device.sum(&a);
    assert!(res == -54.);

    let res = device.mean(&a);
    assert!(res == -54. / a.size() as f32);

    let res = device.sum_cols(&a);
    assert_eq!(res.read(), vec![-15., -15., -24.]);

    let res = device.sum_rows(&a);
    assert_eq!(res.read(), vec![-21., -15., -18.]);
    Ok(())
}
