mod gemm;
mod switching;
mod tew;

pub use gemm::cl_gemm;
pub use switching::*;
pub use tew::*;

use custos::{
    libs::opencl::{cl_device::CLDevice, KernelOptions},
    opencl::{
        api::{enqueue_write_buffer, wait_for_event},
        KernelArg,
    },
    Buffer, CDatatype, Error,
};

use crate::Matrix;

pub fn cl_str_op<T: CDatatype>(
    device: &CLDevice,
    x: &Matrix<T>,
    op: &str,
) -> Result<Matrix<T>, Error> {
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

    let buf = KernelOptions::new(device, x.as_buf(), [x.size(), 0, 0], &src)?
        .with_output(x.size())
        .run()?
        .unwrap();
    Ok((buf, x.dims()).into())
}

pub fn cl_scalar_op<T: CDatatype>(
    device: &CLDevice,
    x: &Matrix<T>,
    scalar: T,
    op: &str,
) -> Result<Matrix<T>, Error> {
    let src = format!("
        __kernel void scalar_r_op(__global const {datatype}* x, const {datatype} scalar, __global {datatype}* out) {{
            size_t id = get_global_id(0);
            
            out[id] = x[id]{op}scalar;
        }}
    ", datatype=T::as_c_type_str());

    let buf = KernelOptions::new(device, x.as_buf(), [x.size(), 0, 0], &src)?
        .add_arg(&scalar)
        .with_output(x.size())
        .run();
    // TODO: unwrap, Ok()?
    buf.map(|buf| (buf.unwrap(), x.dims()).into())
}

pub fn cl_write<T>(device: &CLDevice, x: &mut Buffer<T>, data: &[T]) {
    let event = unsafe { enqueue_write_buffer(&device.queue(), x.ptr.1, data, true).unwrap() };
    wait_for_event(event).unwrap();
}

impl<'a, T: Copy> KernelArg<'a, T> for Matrix<T> {
    fn some_buf(&'a self) -> Option<&'a Buffer<T>> {
        Some(self.as_buf())
    }
}

impl<'a, T: Copy> KernelArg<'a, T> for &'a Matrix<T> {
    fn some_buf(&self) -> Option<&'a Buffer<T>> {
        Some(self.as_buf())
    }
}
