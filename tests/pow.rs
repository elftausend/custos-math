use custos::{CPU, AsDev, Matrix};
use custos_math::Fns;


#[test]
fn test_pow() {

    let device = CPU::new().select();
    
    let a = Matrix::from((&device, 2, 2, [3., 1., 6., 4.]));
    let b = Matrix::from((&device, 2, 2, [2.3, 1.5, 4.31, 6.53]));

    let x = (b * a.ln()).exp();
    println!("x: {x:?}");
}