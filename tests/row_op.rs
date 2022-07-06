use custos::{cpu::CPU, AsDev, Matrix};
use custos_math::{row_op, RowOp};

#[test]
fn test_row_op() {
    let device = CPU::new().select();

    let a = Matrix::from((&device, (3, 3), [1., 2., 3., 4., 5., 6., 7., 8., 9.]));
    let b = Matrix::from((&device, (1, 3), [1., 2., 3.]));

    let c = row_op(&device, &a, &b, |c, a, b| *c = a + b);
    assert_eq!(c.read(), vec![2., 4., 6., 5., 7., 9., 8., 10., 12.]);
}

#[test]
fn test_row_op_cl() {
    let device = custos::CLDevice::new(0).unwrap().select();

    let a = Matrix::from((&device, (3, 3), [1., 2., 3., 4., 5., 6., 7., 8., 9.]));
    let b = Matrix::from((&device, (1, 3), [1., 2., 3.]));

    let c = device.add_row(&a, &b);
    assert_eq!(c.read(), vec![2., 4., 6., 5., 7., 9., 8., 10., 12.])
}
