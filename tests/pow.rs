use custos::CPU;
use custos_math::Matrix;

#[cfg(feature="cpu")]
#[test]
fn test_pow() {
    let device = CPU::new();

    let a = Matrix::from((&device, 2, 2, [3., 1., 6., 4.]));
    let b = Matrix::from((&device, 2, 2, [2.3, 1.5, 4.31, 6.53]));

    //let x = a.ln().exp();
    let _x = (b * a.ln()).exp();
    //println!("x: {x:?}");
}
