use custos::{CudaDevice, Buffer, CDatatype, cuda::{CudaCache, launch_kernel1d}};

pub fn cu_scalar_op<T: CDatatype>(device: &CudaDevice, lhs: &Buffer<T>, rhs: T, op: &str) -> custos::Result<Buffer<T>> {
    let src = format!(
        r#"extern "C" __global__ void scalar_op({datatype}* lhs, {datatype} rhs, {datatype}* out, int numElements)
            {{
                int idx = blockDim.x * blockIdx.x + threadIdx.x;
                if (idx < numElements) {{
                    out[idx] = lhs[idx] {op} rhs;
                }}
              
            }}
    "#, datatype=T::as_c_type_str());

    let out = CudaCache::get::<T>(&device, lhs.len);
    launch_kernel1d(
        lhs.len, &device, 
        &src, "scalar_op", 
        vec![&lhs, &rhs, &out, &lhs.len],
    )?;
    Ok(out)
}