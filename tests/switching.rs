use custos::{CLDevice, Matrix, AsDev, range, opencl::cpu_exec, CudaDevice};
use custos_math::{FnsOps, nn::SoftmaxOps, cl_to_cpu_s, cu_to_cpu_scalar, SumOps};


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


#[test]
fn test_basic_switch_cuda() -> custos::Result<()> {
    let device = CudaDevice::new(0)?;
    let a = Matrix::from((&device, 3, 2, [1, 2, 3, 4, 5, 6,]));
    let sum = cu_to_cpu_scalar(&device, &a, |cpu, x| cpu.sum(&x));
    println!("sum: {sum}");
    Ok(())
}