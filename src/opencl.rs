mod switching;

pub use switching::*;

use custos::{
    devices::opencl::cl_device::CLDevice,
    opencl::{
        api::{enqueue_write_buffer, wait_for_event},
        AsClCvoidPtr,
    },
    Buffer, CDatatype, Error,
};

use crate::{cl_scalar_op, cl_str_op, Matrix};

#[inline]
pub fn cl_str_op_mat<'a, T: CDatatype>(
    device: &'a CLDevice,
    x: &Matrix<T>,
    op: &str,
) -> Result<Matrix<'a, T>, Error> {
    let out = cl_str_op(device, x, op)?;
    Ok((out, x.dims()).into())
}

pub fn cl_scalar_op_mat<'a, T: CDatatype>(
    device: &'a CLDevice,
    x: &Matrix<T>,
    scalar: T,
    op: &str,
) -> Result<Matrix<'a, T>, Error> {
    let out = cl_scalar_op(device, x, scalar, op)?;
    Ok((out, x.dims()).into())
}

pub fn cl_write<T>(device: &CLDevice, x: &mut Buffer<T>, data: &[T]) {
    let event = unsafe { enqueue_write_buffer(&device.queue(), x.ptr.1, data, true).unwrap() };
    wait_for_event(event).unwrap();
}

impl<'a, T> AsClCvoidPtr for Matrix<'a, T> {
    fn as_cvoid_ptr(&self) -> *const std::ffi::c_void {
        self.ptr.1
    }
}

impl<'a, T> AsClCvoidPtr for &Matrix<'a, T> {
    fn as_cvoid_ptr(&self) -> *const std::ffi::c_void {
        self.ptr.1
    }
}
