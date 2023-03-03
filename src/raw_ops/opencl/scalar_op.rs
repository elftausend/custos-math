use custos::{
    number::Number, opencl::enqueue_kernel, prelude::CLBuffer, CDatatype, Device, OpenCL,
};

pub fn cl_scalar_op<'a, T>(
    device: &'a OpenCL,
    x: &CLBuffer<T>,
    scalar: T,
    op: &str,
) -> custos::Result<CLBuffer<'a, T>>
where
    T: CDatatype + Number,
{
    let src = format!("
    __kernel void scalar_r_op(__global const {datatype}* x, const {datatype} scalar, __global {datatype}* out) {{
        size_t id = get_global_id(0);
        
        out[id] = x[id]{op}scalar;
    }}
    ", datatype=T::as_c_type_str());

    //let out = Cache::get::<T, _>(device, x.len, x.node.idx);
    let out: CLBuffer<T> = device.retrieve(x.len(), x.node.idx);
    enqueue_kernel(device, &src, [x.len(), 0, 0], None, &[x, &scalar, &out])?;
    Ok(out)
}
