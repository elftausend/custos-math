use crate::Matrix;
use custos::{
    cache::Cache,
    cuda::launch_kernel1d,
    prelude::{CUBuffer, Number},
    CDatatype, Device, Read, WriteBuf, CPU, CUDA,
};

pub fn cu_scalar_op<'a, T: CDatatype + Number>(
    device: &'a CUDA,
    lhs: &CUBuffer<T>,
    rhs: T,
    op: &str,
) -> custos::Result<CUBuffer<'a, T>> {
    let src = format!(
        r#"extern "C" __global__ void scalar_op({datatype}* lhs, {datatype} rhs, {datatype}* out, int numElements)
            {{
                int idx = blockDim.x * blockIdx.x + threadIdx.x;
                if (idx < numElements) {{
                    out[idx] = lhs[idx] {op} rhs;
                }}
              
            }}
    "#,
        datatype = T::as_c_type_str()
    );

    let out = device.retrieve::<T, ()>(lhs.len(), lhs);
    launch_kernel1d(
        lhs.len(),
        device,
        &src,
        "scalar_op",
        &[&lhs, &rhs, &out, &lhs.len()],
    )?;
    Ok(out)
}

pub fn cu_str_op<'a, T: CDatatype>(
    device: &'a CUDA,
    lhs: &CUBuffer<T>,
    op: &str,
) -> custos::Result<CUBuffer<'a, T>> {
    let src = format!(
        r#"extern "C" __global__ void str_op({datatype}* lhs, {datatype}* out, int numElements)
            {{
                int idx = blockDim.x * blockIdx.x + threadIdx.x;
                if (idx < numElements) {{
                    {datatype} x = lhs[idx];
                    out[idx] = {op};
                }}
            }}
    "#,
        datatype = T::as_c_type_str()
    );

    let out = device.retrieve::<T, ()>(lhs.len(), lhs);
    launch_kernel1d(lhs.len(), device, &src, "str_op", &[&lhs, &out, &lhs.len()])?;
    Ok(out)
}

pub fn cu_str_op_mut<'a, T: CDatatype>(
    device: &'a CUDA,
    lhs: &CUBuffer<T>,
    op: &str,
) -> custos::Result<()> {
    let src = format!(
        r#"extern "C" __global__ void str_op_mut({datatype}* lhs, int numElements)
            {{
                int idx = blockDim.x * blockIdx.x + threadIdx.x;
                if (idx < numElements) {{
                    {datatype} x = lhs[idx];
                    lhs[idx] = {op};
                }}
            }}
    "#,
        datatype = T::as_c_type_str()
    );

    launch_kernel1d(lhs.len(), device, &src, "str_op_mut", &[&lhs, &lhs.len()])?;
    Ok(())
}

pub fn cu_to_cpu_lr<'o, T, F>(
    device: &'o CUDA,
    lhs: &Matrix<T, CUDA>,
    rhs: &Matrix<T, CUDA>,
    f: F,
) -> Matrix<'o, T, CUDA>
where
    T: Copy + Default,
    F: for<'b> Fn(&'b CPU, &Matrix<T>, &Matrix<T>) -> Matrix<'b, T>,
{
    let cpu = custos::CPU::new();
    let lhs = Matrix::from((&cpu, lhs.dims(), device.read(lhs)));
    let rhs = Matrix::from((&cpu, rhs.dims(), device.read(rhs)));

    let result = f(&cpu, &lhs, &rhs);
    Matrix::from((device, result))
}

pub fn cu_to_cpu_lr_mut<T: Copy + Default, F: Fn(&CPU, &mut Matrix<T>, &Matrix<T>)>(
    device: &CUDA,
    lhs: &mut Matrix<T, CUDA>,
    rhs: &Matrix<T, CUDA>,
    f: F,
) {
    let cpu = custos::CPU::new();
    let mut cpu_lhs = Matrix::from((&cpu, lhs.dims(), device.read(lhs)));
    let cpu_rhs = Matrix::from((&cpu, rhs.dims(), device.read(rhs)));

    f(&cpu, &mut cpu_lhs, &cpu_rhs);
    device.write(lhs, &cpu_lhs);
}

pub fn cu_to_cpu_s<'o, T, F>(device: &'o CUDA, x: &Matrix<T, CUDA>, f: F) -> Matrix<'o, T, CUDA>
where
    T: Copy + Default,
    F: for<'b> Fn(&'b CPU, &Matrix<T>) -> Matrix<'b, T>,
{
    let cpu = custos::CPU::new();
    let x = Matrix::from((&cpu, x.dims(), x.read()));

    let result = f(&cpu, &x);
    Matrix::from((device, result))
}

pub fn cu_to_cpu_s_mut<T: Copy + Default, F: Fn(&CPU, &mut Matrix<T>)>(
    x: &mut Matrix<T, CUDA>,
    f: F,
) {
    let cpu = custos::CPU::new();
    let mut cpux = Matrix::from((&cpu, x.dims(), x.read()));

    f(&cpu, &mut cpux);
    x.write(&cpux);
}

pub fn cu_to_cpu_scalar<T: Copy + Default, F: Fn(&CPU, Matrix<T>) -> T>(
    x: &Matrix<T, CUDA>,
    f: F,
) -> T {
    let cpu = custos::CPU::new();
    let x = Matrix::from((&cpu, x.dims(), x.read()));
    f(&cpu, x)
}
