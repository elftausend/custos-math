use custos::{libs::opencl::{cl_device::InternCLDevice, KernelOptions}, Matrix, Error, GenericOCL,};

pub fn str_op<T: GenericOCL>(device: InternCLDevice, x: Matrix<T>, op: &str) -> Result<Matrix<T>, Error> {
    let src = format!("
        __kernel void str_op(__global const {datatype}* x, __global {datatype}* out) {{
            size_t id = get_global_id(0);
            {datatype} I = x[id];
            out[id] = {op};
        }}
    ", datatype=T::as_ocl_type_str());

    KernelOptions::new(&device, &x, [x.size(), 0, 0], &src)
        .with_output(x.dims())
        .run()
}

pub fn scalar_op<T: GenericOCL>(device: InternCLDevice, x: Matrix<T>, scalar: T, op: &str) -> Result<Matrix<T>, Error> {
    let src = format!("
        __kernel void scalar_r_op(__global const {datatype}* x, const {datatype} scalar, __global {datatype}* out) {{
            size_t id = get_global_id(0);
            
            out[id] = x[id]{op}scalar;
        }}
    ", datatype=T::as_ocl_type_str());

    KernelOptions::new(&device, &x, [x.size(), 0, 0], &src)
        .add_arg(&scalar)
        .with_output(x.dims())
        .run()
}

