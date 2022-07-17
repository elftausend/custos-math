use custos::{AsDev, Buffer, CPU};
#[cfg(feature = "cuda")]
use custos::{CudaDevice, VecRead};
use custos_math::RandOp;

#[cfg(feature = "cuda")]
#[test]
fn test_rand_cuda() -> custos::Result<()> {
    use custos_math::Matrix;

    let device = CudaDevice::new(0)?;

    let mut a: Matrix<f32> = (Buffer::<f32>::new(&device, 10), 2, 5).into();
    device.rand(&mut a, -1., 1.);
    println!("{:?}", device.read(&a));
    Ok(())
}

#[test]
fn test_rand() -> custos::Result<()> {
    use custos_math::Matrix;

    let device = CPU::new().select();

    let mut a: Matrix<f32> = (Buffer::<f32>::new(&device, 10), 2, 5).into();
    device.rand(&mut a, -5., 6.);
    println!("{:?}", a);
    Ok(())
}
