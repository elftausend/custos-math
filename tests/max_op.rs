use custos::{CPU, Matrix, AsDev, CLDevice};
use custos_math::MaxOps;

#[test]
fn test_max_ops() {
    let device = CPU::new().select();
    let a = Matrix::from((&device, (3, 3),
    [-10., -2., -3., 
    -4., -5., -6., 
    -7., -8., -9.,]));

    let res = device.max(&a);
    assert!(res == -2.);

    let res = device.max_cols(&a);
    assert_eq!(res.read(), vec![-2., -4., -7.]);

    let res = device.max_rows(&a);
    assert_eq!(res.read(), vec![-4., -2., -3.]);

    
    let device = CLDevice::get(0).unwrap().select();

    let a = Matrix::from((&device, (3, 3),
    [-10f32, -2., -3., 
    -4., -5., -6., 
    -7., -8., -9.,]));

    let res = device.max(&a);
    assert!(res == -2.);

    let res = device.max_cols(&a);
    assert_eq!(res.read(), vec![-2., -4., -7.]);

    let res = device.max_rows(&a);
    assert_eq!(res.read(), vec![-4., -2., -3.])
}