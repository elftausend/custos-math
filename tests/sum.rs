use custos::{CPU, Matrix, AsDev, CLDevice};
use custos_math::SumOp;

#[test]
fn test_sum_ops() {
    let device = CPU::new().select();
    let a = Matrix::from((&device, (3, 3),
    [-10., -2., -3., 
    -4., -5., -6., 
    -7., -8., -9.,]));

    let res = device.sum(a);
    assert!(res == -54.);

    let res = device.mean(a);
    assert!(res == -54. / a.size() as f32);

    let res = device.sum_cols(a);
    assert_eq!(res.read(), vec![-15., -15., -24.]);

    let res = device.sum_rows(a);
    assert_eq!(res.read(), vec![-21., -15., -18.]);

    
    let device = CLDevice::get(0).unwrap().select();

    let a = Matrix::from((&device, (3, 3),
    [-10f32, -2., -3., 
    -4., -5., -6., 
    -7., -8., -9.,]));

    let res = device.sum(a);
    assert!(res == -54.);

    let res = device.mean(a);
    assert!(res == -54. / a.size() as f32);

    let res = device.sum_cols(a);
    assert_eq!(res.read(), vec![-15., -15., -24.]);

    let res = device.sum_rows(a);
    assert_eq!(res.read(), vec![-21., -15., -18.])
}