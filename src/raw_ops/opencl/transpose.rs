use custos::{CLDevice, CDatatype, Buffer, Cache};

pub fn cl_transpose<'a, T: CDatatype>(
    device: &'a CLDevice,
    x: &Buffer<T>,
    rows: usize,
    cols: usize,
) -> custos::Result<Buffer<'a, T>> {
    use custos::opencl::enqueue_kernel;

    let src = format!(
        "
        #define MODULO(x,N) (x % N)
        #define I0 {rows}
        #define I1 {cols}
        #define I_idx(i0,i1) ((size_t)(i0))*I1+(i1)
        #define I_idx_mod(i0,i1) MODULO( ((size_t)(i0)) ,I0)*I1+MODULO( (i1),I1)

        #define MODULO(x,N) (x % N)
        #define O0 {cols}
        #define O1 {rows}
        #define O_idx(o0,o1) ((size_t)(o0))*O1+(o1)
        #define O_idx_mod(o0,o1) MODULO( ((size_t)(o0)) ,O0)*O1+MODULO( (o1),O1)
        __kernel void transpose(__global const {datatype}* I, __global {datatype}* O) {{
            size_t gid = get_global_id(0);
            size_t gid_original = gid;size_t i1 = gid % I1;size_t i0 = gid / I1;gid = gid_original;
        
            O[O_idx(i1,i0)] = I[gid];
        }}
    
   ",
        datatype = T::as_c_type_str()
    );

    let gws = [x.len, 0, 0];
    let out = Cache::get::<T, _>(device, x.len, x.node.idx);
    enqueue_kernel(device, &src, gws, None, &[x, &out])?;
    Ok(out)
}
