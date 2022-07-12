use custos::{AsDev, CPU};
use custos_math::{nn::cce, Matrix};

#[test]
fn test_cce() {
    let device = CPU::new().select();

    let x = Matrix::from((&device, (2, 3), [0.1, 0.2, 0.7, 0.1, 0.8, 0.1]));

    let y = Matrix::from((&device, (2, 3), [0., 0., 1., 0., 1., 0.]));
    let res = cce(&device, &x, &y);
    println!("res: {:?}", res);
}
