use custos::{CDatatype, CudaDevice, Buffer, cuda::launch_kernel1d};

pub fn cu_assign_scalar<'a, T: CDatatype>(
    device: &'a CudaDevice,
    lhs: &Buffer<T>,
    rhs: T,
    op: &str,
) -> custos::Result<()> {
    let src = format!(
        r#"extern "C" __global__ void scalar_assign({datatype}* lhs, {datatype} rhs, int numElements)
            {{
                int idx = blockDim.x * blockIdx.x + threadIdx.x;
                if (idx < numElements) {{
                    lhs[idx] {op}= rhs;
                }}
              
            }}
    "#,
        datatype = T::as_c_type_str()
    );

    launch_kernel1d(
        lhs.len,
        device,
        &src,
        "scalar_assign",
        &[&lhs, &rhs, &lhs.len],
    )?;
    Ok(())
}