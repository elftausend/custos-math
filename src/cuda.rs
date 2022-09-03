mod ew;
mod switching;

pub use ew::*;
pub use switching::*;

use custos::{cache::Cache, cuda::launch_kernel1d, Buffer, CDatatype, CudaDevice};

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

    let out = Cache::get::<T, _>(device, lhs.len);
    launch_kernel1d(
        lhs.len,
        device,
        &src,
        "scalar_op",
        vec![&lhs, &rhs, &out, &lhs.len],
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

    let out = Cache::get::<T, _>(device, lhs.len);
    launch_kernel1d(lhs.len, device, &src, "str_op", vec![&lhs, &out, &lhs.len])?;
    Ok(out)
}
