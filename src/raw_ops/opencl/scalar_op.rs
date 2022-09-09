use custos::{opencl::enqueue_kernel, Buffer, CDatatype, CLDevice, Cache};

pub fn cl_scalar_op<'a, T>(
    device: &'a CLDevice,
    x: &Buffer<T>,
    scalar: T,
    op: &str,
) -> custos::Result<Buffer<'a, T>>
where
    T: CDatatype,
{
    let src = format!("
    __kernel void scalar_r_op(__global const {datatype}* x, const {datatype} scalar, __global {datatype}* out) {{
        size_t id = get_global_id(0);
        
        out[id] = x[id]{op}scalar;
    }}
    ", datatype=T::as_c_type_str());

    let out = Cache::get::<T, _>(device, x.len, x.node.idx);
    enqueue_kernel(device, &src, [x.len, 0, 0], None, &[x, &scalar, &out])?;
    Ok(out)
}
