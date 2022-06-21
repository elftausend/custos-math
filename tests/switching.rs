use custos::{CLDevice, Matrix, AsDev, range, opencl::cpu_exec};
use custos_math::{FnsOps, nn::SoftmaxOps, cl_to_cpu_s};


#[test]
fn test_unified_mem_device_switch() -> custos::Result<()> {
    let device = CLDevice::new(0)?.select();

    let a = Matrix::from((&device, 2, 3, [1., 2., 3., 4., 5., 6.,]));

    let start = std::time::Instant::now();
    for _ in range(100000) {
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



