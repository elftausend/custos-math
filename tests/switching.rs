use custos::{range, CLDevice};
use custos_math::{cl_to_cpu_s, nn::SoftmaxOps};
use custos_math::{cpu_exec_lhs_rhs_mut, FnsOps, Matrix, RowOp};

#[cfg(feature = "cuda")]
use custos::{CudaDevice, VecRead};
#[cfg(feature = "cuda")]
use custos_math::{cu_to_cpu_lr, cu_to_cpu_s, cu_to_cpu_scalar, SumOps};

#[test]
fn test_swtich_mut_cl() -> custos::Result<()> {
    let device = CLDevice::new(0)?;
    let unified = device.unified_mem();
    device.set_unified_mem(false);

    let test = || {
        let mut matrix = Matrix::from((&device, 2, 3, [1., 2., 3., 4., 5., 6.]));
        let rhs = Matrix::from((&device, 1, 3, [1., 2., 3.]));
        cpu_exec_lhs_rhs_mut(&device, &mut matrix, &rhs, |cpu, matrix, rhs| {
            cpu.add_row_mut(matrix, rhs)
        })?;
        custos::Result::Ok(matrix.read())
    };

    assert_eq!(test()?, vec![2.0, 4.0, 6.0, 5.0, 7.0, 9.0]);
    if !unified {
        return Ok(());
    }
    device.set_unified_mem(true);
    assert_eq!(test()?, vec![2.0, 4.0, 6.0, 5.0, 7.0, 9.0]);
    Ok(())
}

#[test]
fn test_unified_mem_device_switch_exact() -> custos::Result<()> {
    use custos_math::{cpu_exec, Matrix};

    let device = CLDevice::new(0)?;

    let a = Matrix::from((&device, 2, 3, [1., 2., 3., 4., 5., 6.]));

    let start = std::time::Instant::now();
    for _ in range(10000) {
        let _m = cpu_exec(&device, &a, |cpu, m| cpu.neg(&m))?;
    }

    println!("duration: {:?}", start.elapsed());

    let m = cpu_exec(&device, &a, |cpu, m| cpu.neg(&m))?;
    println!("return m: {m:?}");
    Ok(())
}

#[test]
fn test_unified_mem_device_switch_softmax() -> custos::Result<()> {
    use custos_math::{cpu_exec, Matrix};

    let device = CLDevice::new(0)?;

    let a = Matrix::from((&device, 2, 3, [1., 2., 3., 4., 5., 6.]));

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

#[cfg(feature = "cuda")]
#[test]
fn test_switch_mut_cu() -> custos::Result<()> {
    use custos_math::cu_to_cpu_lr_mut;

    let device = custos::CudaDevice::new(0)?;

    let mut matrix = Matrix::from((&device, 2, 3, [1., 2., 3., 4., 5., 6.]));
    let rhs = Matrix::from((&device, 1, 3, [1., 2., 3.]));
    cu_to_cpu_lr_mut(&device, &mut matrix, &rhs, |cpu, matrix, rhs| {
        cpu.add_row_mut(matrix, rhs)
    });

    assert_eq!(matrix.read(), vec![2.0, 4.0, 6.0, 5.0, 7.0, 9.0]);

    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn test_scalar_switch_cuda() -> custos::Result<()> {
    use custos_math::Matrix;

    let device = CudaDevice::new(0)?;
    let a = Matrix::from((&device, 3, 2, [1, 2, 3, 4, 5, 6]));
    let sum = cu_to_cpu_scalar(&device, &a, |cpu, x| cpu.sum(&x));

    assert_eq!(sum, 21);

    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn test_single_switch_cuda() -> custos::Result<()> {
    use custos_math::{FnsOps, Matrix};

    let device = CudaDevice::new(0)?;
    let a = Matrix::from((&device, 3, 2, [1., 2., 3., 4., 5., 6.]));
    let res = cu_to_cpu_s(&device, &a, |cpu, x| cpu.neg(&x));
    assert_eq!(device.read(&res), vec![-1., -2., -3., -4., -5., -6.]);
    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn test_lr_switch_cuda() -> custos::Result<()> {
    use custos_math::{BaseOps, Matrix};

    let device = CudaDevice::new(0)?;
    let lhs = Matrix::from((&device, 3, 2, [1, 2, 3, 4, 5, 6]));
    let rhs = Matrix::from((&device, 3, 2, [2, 2, 3, 4, 5, 7]));

    let out = cu_to_cpu_lr(&device, &lhs, &rhs, |cpu, l, r| cpu.add(l, r));
    assert_eq!(device.read(&out), vec![3, 4, 6, 8, 10, 13]);
    Ok(())
}
