use custos::{Buffer, CPU};
#[cfg(feature = "cuda")]
use custos::{Read, CUDA};

#[cfg(feature = "fastrand")]
use custos_math::RandOp;

#[cfg(feature = "cuda")]
#[test]
fn test_rand_cuda() -> custos::Result<()> {
    use custos_math::Matrix;

    let device = CUDA::new(0)?;

    let mut a: Matrix<_, _> = (Buffer::new(&device, 10), 2, 5).into();
    device.rand(&mut a, -1., 1.);
    println!("{:?}", device.read(&a));
    Ok(())
}

#[cfg(feature = "fastrand")]
#[cfg(feature = "cpu")]
#[test]
fn test_rand() -> custos::Result<()> {
    use custos_math::Matrix;

    let device = CPU::new();

    let mut a: Matrix<f32> = (Buffer::<f32>::new(&device, 10), 2, 5).into();
    device.rand(&mut a, -5., 6.);
    println!("{:?}", a);
    Ok(())
}
