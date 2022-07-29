mod gemm;
mod switching;
mod tew;

pub use gemm::cl_gemm;
pub use switching::*;
pub use tew::*;

use custos::{
    libs::opencl::{cl_device::CLDevice},
    opencl::{
        api::{enqueue_write_buffer, wait_for_event}, CLCache, enqueue_kernel, AsClCvoidPtr,
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

    let out = CLCache::get::<T>(&device, x.size());
    enqueue_kernel(device, &src, [x.size(), 0, 0], None, &[x, &out])?;
    Ok((out, x.dims()).into())
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

    let out = CLCache::get::<T>(device, x.size());
    enqueue_kernel(device, &src, [x.size(), 0, 0], None, &[x, &scalar, &out])?;
    
    Ok((out, x.dims()).into())
}

pub fn cl_write<T>(device: &CLDevice, x: &mut Buffer<T>, data: &[T]) {
    let event = unsafe { enqueue_write_buffer(&device.queue(), x.ptr.1, data, true).unwrap() };
    wait_for_event(event).unwrap();
}

impl<T> AsClCvoidPtr for Matrix<T> {
    fn as_cvoid_ptr(&self) -> *const std::ffi::c_void {
        self.ptr.1
    }
}

impl<T> AsClCvoidPtr for &Matrix<T> {
    fn as_cvoid_ptr(&self) -> *const std::ffi::c_void {
        self.ptr.1
    }
}