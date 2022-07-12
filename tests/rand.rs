#[cfg(feature="cuda")]
use custos_math::RandOp;
#[cfg(feature="cuda")]
use custos::{Buffer, VecRead, CudaDevice};

#[cfg(feature="cuda")]
#[test]
fn test_rand_cuda() -> custos::Result<()> {
    use custos_math::Matrix;

    let device = CudaDevice::new(0)?;
    
    let mut a: Matrix<f32> = (Buffer::<f32>::new(&device, 10), 2, 5).into();
    device.rand(&mut a, -1., 1.);
    println!("{:?}", device.read(&a));
    Ok(())
}