use custos::{OpenCL, Buffer, CDatatype, opencl::enqueue_kernel};

pub fn cl_assign_scalar<'a, T>(
    device: &'a OpenCL,
    x: &Buffer<T>,
    scalar: T,
    op: &str,
) -> custos::Result<()>
where
    T: CDatatype,
{
    let src = format!("
    __kernel void scalar_assign(__global {datatype}* x, const {datatype} scalar) {{
        size_t id = get_global_id(0);
        
        x[id] {op}= scalar;
    }}
    ", datatype=T::as_c_type_str());

    enqueue_kernel(device, &src, [x.len, 0, 0], None, &[x, &scalar])?;
    Ok(())
}
