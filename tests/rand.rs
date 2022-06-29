use custos_math::RandOp;
use custos::{Buffer, Matrix, VecRead};
//#[cfg(feature="cuda")]
use custos::CudaDevice;


#[test]
fn test_rand_cuda() -> custos::Result<()> {
    let device = CudaDevice::new(0)?;
    
    let mut a: Matrix<f32> = (Buffer::<f32>::new(&device, 10), 2, 5).into();
    device.rand(&mut a, -1., 1.);
    println!("{:?}", device.read(&a));
    Ok(())
}