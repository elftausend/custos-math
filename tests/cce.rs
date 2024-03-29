use custos::CPU;
use custos_math::Matrix;

#[cfg(feature = "cpu")]
#[test]
fn test_cce() {
    let device = CPU::new();

    let x = Matrix::from((&device, (2, 3), [0.1, 0.2, 0.7, 0.1, 0.8, 0.1]));

    let y = Matrix::from((&device, (2, 3), [0., 0., 1., 0., 1., 0.]));
    let res = x.cce_loss(&y);

    println!("res: {:?}", res);
}
