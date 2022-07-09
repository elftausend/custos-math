use custos::Matrix;
use custos_math::FnsOps;

#[cfg(feature="opencl")]
use custos::{CLDevice, opencl::cpu_exec, AsDev, range};
#[cfg(feature="opencl")]
use custos_math::{nn::SoftmaxOps, cl_to_cpu_s};

#[cfg(feature="cuda")]
use custos::{CudaDevice, VecRead, BaseOps};
#[cfg(feature="cuda")]
use custos_math::{cu_to_cpu_scalar, cu_to_cpu_s, cu_to_cpu_lr, SumOps};

#[cfg(feature="opencl")]
#[test]
fn test_unified_mem_device_switch() -> custos::Result<()> {
    let device = CLDevice::new(0)?.select();

    let a = Matrix::from((&device, 2, 3, [1., 2., 3., 4., 5., 6.,]));

    let start = std::time::Instant::now();
    for _ in range(10000) {
        let _m = cpu_exec(&device, &a, |cpu, m| cpu.ln(&m))?;
    }

    println!("duration: {:?}", start.elapsed());

    let m = cpu_exec(&device, &a, |cpu, m| cpu.ln(&m))?;
    println!("m: {m:?}");
    Ok(())
}

#[cfg(feature="opencl")]
#[test]
fn test_unified_mem_device_switch_softmax() -> custos::Result<()> {
    let device = CLDevice::new(0)?.select();

    let a = Matrix::from((&device, 2, 3, [1., 2., 3., 4., 5., 6.,]));

    let start = std::time::Instant::now();
    for _ in range(10000) {
        //let x: Matrix<f32> = (cached::<f32>(6), 2, 3).into();
        cl_to_cpu_s(&device, &a, |cpu, m| cpu.softmax(&m));
        //cpu_exec(&device, &a, |cpu, m| cpu.softmax(&m))?;
    }

    println!("duration: {:?}", start.elapsed());

    let m = cpu_exec(&device, &a, |cpu, m| cpu.ln(&m))?;
    println!("m: {m:?}");
    Ok(())
}

#[cfg(feature="cuda")]
#[test]
fn test_scalar_switch_cuda() -> custos::Result<()> {
    let device = CudaDevice::new(0)?;
    let a = Matrix::from((&device, 3, 2, [1, 2, 3, 4, 5, 6,]));
    let sum = cu_to_cpu_scalar(&device, &a, |cpu, x| cpu.sum(&x));
    
    assert_eq!(sum, 21);

    Ok(())
}

#[cfg(feature="cuda")]
#[test]
fn test_single_switch_cuda() -> custos::Result<()> {
    let device = CudaDevice::new(0)?;
    let a = Matrix::from((&device, 3, 2, [1., 2., 3., 4., 5., 6.]));
    let res = cu_to_cpu_s(&device, &a, |cpu, x| cpu.neg(&x));
    assert_eq!(device.read(&res), vec![-1., -2., -3., -4., -5., -6.]);
    Ok(())
}

#[cfg(feature="cuda")]
#[test]
fn test_lr_switch_cuda() -> custos::Result<()> {
    let device = CudaDevice::new(0)?;
    let lhs = Matrix::from((&device, 3, 2, [1, 2, 3, 4, 5, 6,]));
    let rhs = Matrix::from((&device, 3, 2, [2, 2, 3, 4, 5, 7,]));
    
    let out = cu_to_cpu_lr(&device, &lhs, &rhs, |cpu, l, r| cpu.add(l, r));
    assert_eq!(device.read(&out), vec![3, 4, 6, 8, 10, 13]);
    Ok(())
}