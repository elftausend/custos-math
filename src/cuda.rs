use crate::Matrix;
use custos::{
    cache::Cache, cuda::launch_kernel1d, Buffer, CDatatype, CudaDevice, VecRead, WriteBuf, CPU,
};

pub fn cu_scalar_op<'a, T: CDatatype>(
    device: &'a CudaDevice,
    lhs: &Buffer<T>,
    rhs: T,
    op: &str,
) -> custos::Result<Buffer<'a, T>> {
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

    let out = Cache::get::<T, _>(device, lhs.len, lhs.node.idx);
    launch_kernel1d(
        lhs.len,
        device,
        &src,
        "scalar_op",
        &[&lhs, &rhs, &out, &lhs.len],
    )?;
    Ok(out)
}

pub fn cu_str_op<'a, T: CDatatype>(
    device: &'a CudaDevice,
    lhs: &Buffer<T>,
    op: &str,
) -> custos::Result<Buffer<'a, T>> {
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

    let out = Cache::get::<T, _>(device, lhs.len, lhs.node.idx);
    launch_kernel1d(lhs.len, device, &src, "str_op", &[&lhs, &out, &lhs.len])?;
    Ok(out)
}

pub fn cu_to_cpu_lr<'o, T, F>(
    device: &'o CudaDevice,
    lhs: &Matrix<T>,
    rhs: &Matrix<T>,
    f: F,
) -> Matrix<'o, T>
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
    device: &CudaDevice,
    lhs: &mut Matrix<T>,
    rhs: &Matrix<T>,
    f: F,
) {
    let cpu = custos::CPU::new();
    let mut cpu_lhs = Matrix::from((&cpu, lhs.dims(), device.read(lhs)));
    let cpu_rhs = Matrix::from((&cpu, rhs.dims(), device.read(rhs)));

    f(&cpu, &mut cpu_lhs, &cpu_rhs);
    device.write(lhs, &cpu_lhs);
}

pub fn cu_to_cpu_s<'o, T, F>(device: &'o CudaDevice, x: &Matrix<T>, f: F) -> Matrix<'o, T>
where
    T: Copy + Default,
    F: for<'b> Fn(&'b CPU, &Matrix<T>) -> Matrix<'b, T>,
{
    let cpu = custos::CPU::new();
    let x = Matrix::from((&cpu, x.dims(), device.read(x)));

    let result = f(&cpu, &x);
    Matrix::from((device, result))
}

pub fn cu_to_cpu_s_mut<T: Copy + Default, F: Fn(&CPU, &mut Matrix<T>)>(
    device: &CudaDevice,
    x: &mut Matrix<T>,
    f: F,
) {
    let cpu = custos::CPU::new();
    let mut cpux = Matrix::from((&cpu, x.dims(), device.read(x)));

    f(&cpu, &mut cpux);
    device.write(x, &cpux)
}

pub fn cu_to_cpu_scalar<T: Copy + Default, F: Fn(&CPU, Matrix<T>) -> T>(
    device: &CudaDevice,
    x: &Matrix<T>,
    f: F,
) -> T {
    let cpu = custos::CPU::new();
    let x = Matrix::from((&cpu, x.dims(), device.read(x)));
    f(&cpu, x)
}
