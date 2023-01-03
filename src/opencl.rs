//mod switching;
//pub use switching::*;

use custos::{
    devices::opencl::cl_device::OpenCL,
    opencl::{
        api::{enqueue_write_buffer, wait_for_event},
        AsClCvoidPtr,
    },
    prelude::CLBuffer,
    CDatatype, Error, GraphReturn, WriteBuf, CPU,
};
use std::fmt::Debug;

use crate::{cl_scalar_op, cl_str_op, Matrix};

#[inline]
pub fn cl_str_op_mat<'a, T: CDatatype>(
    device: &'a OpenCL,
    x: &Matrix<T, OpenCL>,
    op: &str,
) -> Result<Matrix<'a, T, OpenCL>, Error> {
    let out = cl_str_op(device, x, op)?;
    Ok((out, x.dims()).into())
}

pub fn cl_scalar_op_mat<'a, T: CDatatype>(
    device: &'a OpenCL,
    x: &Matrix<T, OpenCL>,
    scalar: T,
    op: &str,
) -> Result<Matrix<'a, T, OpenCL>, Error> {
    let out = cl_scalar_op(device, x, scalar, op)?;
    Ok((out, x.dims()).into())
}

pub fn cl_write<T>(device: &OpenCL, x: &mut CLBuffer<T>, data: &[T]) {
    let event = unsafe { enqueue_write_buffer(&device.queue(), x.ptr.ptr, data, true).unwrap() };
    wait_for_event(event).unwrap();
}

impl<'a, T> AsClCvoidPtr for Matrix<'a, T, OpenCL> {
    fn as_cvoid_ptr(&self) -> *const std::ffi::c_void {
        self.ptr.ptr
    }
}

impl<'a, T> AsClCvoidPtr for &Matrix<'a, T, OpenCL> {
    fn as_cvoid_ptr(&self) -> *const std::ffi::c_void {
        self.ptr.ptr
    }
}

/// Compute operations on the CPU even though the matrix was created with an OpenCL device.
/// There were some optimizations implemented regarding unified memory architectures.
///
/// # Example
/// ```
/// use custos::{OpenCL, Read};
/// use custos_math::{Matrix, opencl::cpu_exec, FnsOps};
///
/// fn main() -> custos::Result<()> {
///     let device = OpenCL::new(0)?;
///     let a = Matrix::from((&device, 2, 2, [1f32, 2., 3., 4.]));
///     let res = cpu_exec(&device, &a, |cpu, x| cpu.neg(x))?;
///     assert_eq!(res.read(), vec![-1., -2., -3., -4.]);
///     Ok(())
/// }
/// ```
pub fn cpu_exec<'c, 'a, 'o, T, F>(
    device: &'o OpenCL,
    matrix: &Matrix<'a, T, OpenCL>,
    f: F,
) -> custos::Result<Matrix<'o, T, OpenCL>>
where
    F: for<'b> Fn(&'b CPU, &Matrix<T>) -> Matrix<'b, T>,
    T: Copy + Default + Debug,
{
    
    
    /*
    // TODO: use compile time unified_cl flag -> get from custos?
    #[cfg(not(feature = "realloc"))]
    if device.unified_mem() {
        // Using a CPU stored in a OpenCL in order to get a (correct) cache entry.
        // Due to the (new) caching architecture, using a new CPU isn't possible,
        // as the cache would be newly created every iteration.
        // host ptr matrix
        let no_drop = f(&device.cpu, matrix);

        let dims = no_drop.dims();
        // convert host ptr / CPU matrix into a host ptr + OpenCL ptr matrix
        return unsafe {
            custos::opencl::construct_buffer(device, no_drop.to_buf(), matrix.node.idx)
        }
        .map(|buf| (buf, dims).into());
    }
    */
    let cpu = CPU::new();

    // TODO: fix
    /*#[cfg(feature = "realloc")]
    if device.unified_mem() {
        return Ok(Matrix::from((device, f(&cpu, matrix))));
    }*/

    // convert an OpenCL buffer to a cpu buffer
    let cpu_buf: Matrix<T> = Matrix::from((&cpu, matrix.dims(), matrix.read()));
    let mat: Matrix<T> = f(&cpu, &cpu_buf);
    let mut convert = Matrix::from((device, mat));
    convert.node = device.graph().add(convert.len(), matrix.node.idx);
    Ok(convert)
}

pub fn cpu_exec_mut<T, F>(
    device: &OpenCL,
    matrix: &mut Matrix<T, OpenCL>,
    f: F,
) -> custos::Result<()>
where
    F: Fn(&CPU, &mut Matrix<T>),
    T: Copy + Default,
{
    let cpu = CPU::new();

    /*
    // uses same memory as CPU
    if device.unified_mem() {
        return Ok(f(&cpu, matrix));
    }
    */

    //convert an OpenCL buffer to a cpu matrix
    let mut cpu_matrix = Matrix::from((&cpu, matrix.dims(), matrix.read()));
    f(&cpu, &mut cpu_matrix);
    // write result as slice back to OpenCL Matrix
    device.write(matrix, &cpu_matrix);
    Ok(())
}

pub fn cpu_exec_lhs_rhs<'a, 'o, T, F>(
    device: &'o OpenCL,
    lhs: &Matrix<'a, T, OpenCL>,
    rhs: &Matrix<'a, T, OpenCL>,
    f: F,
) -> custos::Result<Matrix<'o, T, OpenCL>>
where
    F: for<'b> Fn(&'b CPU, &Matrix<T>, &Matrix<T>) -> Matrix<'b, T>,
    T: Copy + Default + Debug,
{
    let cpu = CPU::new();

    /*
    #[cfg(not(feature = "realloc"))]
    if device.unified_mem() {
        let no_drop = f(&device.cpu, lhs, rhs);

        let no_drop_dims = no_drop.dims();
        // convert host ptr / CPU matrix into a host ptr + OpenCL ptr matrix
        return unsafe {
            custos::opencl::construct_buffer(device, no_drop.to_buf(), (lhs.node.idx, rhs.node.idx))
        }
        .map(|buf| (buf, no_drop_dims).into());
    }
    */

    /*
    // TODO: fix
    
    #[cfg(feature = "realloc")]
    if device.unified_mem() {
        return Ok(Matrix::from((device, f(&cpu, lhs, rhs))));
    }*/

    // convert an OpenCL buffer to a cpu buffer
    let lhs = Matrix::from((&cpu, lhs.dims(), lhs.read()));
    let rhs = Matrix::from((&cpu, rhs.dims(), rhs.read()));

    let mut convert = Matrix::from((device, f(&cpu, &lhs, &rhs)));
    convert.node = device
        .graph()
        .add(convert.len(), (lhs.node.idx, rhs.node.idx));

    Ok(convert)
}

pub fn cpu_exec_lhs_rhs_mut<T, F>(
    device: &OpenCL,
    lhs: &mut Matrix<T, OpenCL>,
    rhs: &Matrix<T, OpenCL>,
    f: F,
) -> custos::Result<()>
where
    F: Fn(&CPU, &mut Matrix<T>, &Matrix<T>),
    T: Copy + Default,
{
    let cpu = CPU::new();

    /*
    // uses same memory as CPU
    if device.unified_mem() {
        return Ok(f(&cpu, lhs, rhs));
    }
    */

    //convert OpenCL matrix to cpu matrix
    let mut cpu_lhs = Matrix::from((&cpu, lhs.dims(), lhs.read()));
    let cpu_rhs = Matrix::from((&cpu, rhs.dims(), rhs.read()));
    f(&cpu, &mut cpu_lhs, &cpu_rhs);

    // write result as slice back to OpenCL Matrix
    device.write(lhs, &cpu_lhs);
    Ok(())
}

pub fn cpu_exec_scalar<T, F>(matrix: &Matrix<T, OpenCL>, f: F) -> T
where
    F: Fn(&CPU, &Matrix<T>) -> T,
    T: Copy + Default,
{
    let cpu = CPU::new();

    /*
    if device.unified_mem() {
        return f(&cpu, matrix);
    }
    */

    // convert an OpenCL buffer to a cpu buffer
    let cpu_buf = Matrix::from((&cpu, matrix.dims(), matrix.read()));

    f(&cpu, &cpu_buf)
}
