use custos::{opencl::enqueue_kernel, prelude::CLBuffer, CDatatype, Device, OpenCL};

pub fn cl_str_op<'a, T>(
    device: &'a OpenCL,
    x: &CLBuffer<T>,
    op: &str,
) -> custos::Result<CLBuffer<'a, T>>
where
    T: CDatatype,
{
    let src = format!(
        "
        __kernel void str_op(__global const {datatype}* lhs, __global {datatype}* out) {{
            size_t id = get_global_id(0);
            {datatype} x = lhs[id];
            out[id] = {op};
        }}
    ",
        datatype = T::as_c_type_str()
    );

    let out: CLBuffer<T> = device.retrieve(x.len, x.node.idx);
    enqueue_kernel(device, &src, [x.len, 0, 0], None, &[x, &out])?;
    Ok(out)
}
