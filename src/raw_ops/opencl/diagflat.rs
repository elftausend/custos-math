use custos::prelude::*;

pub fn cl_diagflat<'a, T: CDatatype>(
    device: &'a OpenCL,
    x: &CLBuffer<T>,
    batch_size: usize,
    len: usize,
) -> custos::Result<CLBuffer<'a, T>> {

    let src = format!(
        r#"__kernel void diagflat(__global const {datatype}* input, const int cols, __global {datatype}* output) {{
            size_t x = get_global_id(0);
            size_t y = get_global_id(1);
        
            output[x * cols + x + y * cols * cols] = input[x + y*cols];
            
        }}"#,
        datatype = T::as_c_type_str()
    );

    
    let out: CLBuffer<T> = device.retrieve(len * len * batch_size, x.node.idx);
    enqueue_kernel(
        device,
        &src,
        [len, batch_size, 0],
        None,
        &[x, &(len as i32), &out],
    )?;
    Ok(out)
}
