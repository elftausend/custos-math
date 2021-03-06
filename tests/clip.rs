use custos::{cpu::CPU, AsDev};
use custos_math::{ClipOp, Matrix};

#[cfg(feature = "opencl")]
use custos::opencl::CLDevice;

#[test]
fn test_clip_cpu() {
    let device = CPU::new().select();

    let x = Matrix::<i32>::from((&device, (1, 5), [100, 10, 2000, -500, -5]));

    let res = device.clip(&x, -99, 99);
    assert_eq!(vec![99, 10, 99, -99, -5], res.read());
}

#[cfg(feature = "opencl")]
#[test]
fn test_clip_cl() {
    let device = CLDevice::new(0).unwrap().select();

    let x = Matrix::<i32>::from((&device, (1, 5), [100, 10, 2000, -500, -5]));

    let res = device.clip(&x, -99, 99);
    assert_eq!(vec![99, 10, 99, -99, -5], res.read());
}

#[cfg(feature = "cuda")]
#[test]
fn test_clip_cuda() {
    let device = custos::CudaDevice::new(0).unwrap().select();

    let x = Matrix::<i32>::from((&device, (1, 5), [100, 10, 2000, -500, -5]));

    let res = device.clip(&x, -99, 99);
    assert_eq!(vec![99, 10, 99, -99, -5], res.read());
}
