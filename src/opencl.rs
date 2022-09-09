//mod switching;
//pub use switching::*;

use std::fmt::Debug;
use custos::{
    devices::opencl::cl_device::CLDevice,
    opencl::{
        api::{enqueue_write_buffer, wait_for_event},
        AsClCvoidPtr, construct_buffer,
    },
    Buffer, CDatatype, Error, CPU, GraphReturn, VecRead, WriteBuf,
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


/// Compute operations on the CPU even though the matrix was created with an OpenCL device.
/// There were some optimizations implemented regarding unified memory architectures.
///
/// # Example
/// ```
/// use custos::{CLDevice, VecRead};
/// use custos_math::{Matrix, opencl::cpu_exec, FnsOps};
///
/// fn main() -> custos::Result<()> {
///     let device = CLDevice::new(0)?;
///     let a = Matrix::<f32>::from((&device, 2, 2, [1., 2., 3., 4.]));
///     let res = cpu_exec(&device, &a, |cpu, x| cpu.neg(x))?;
///     assert_eq!(device.read(&res), vec![-1., -2., -3., -4.]);
///     Ok(())
/// }
/// ```
pub fn cpu_exec<'c, 'a, 'o, T, F>(
    device: &'o CLDevice,
    matrix: &Matrix<'a, T>,
    f: F,
) -> custos::Result<Matrix<'o, T>>
where
    F: for<'b> Fn(&'b CPU, &Matrix<T>) -> Matrix<'b, T>,
    T: Copy + Default + Debug,
{
    #[cfg(not(feature = "realloc"))]
    if device.unified_mem() {
        // Using a CPU stored in a CLDevice in order to get a (correct) cache entry.
        // Due to the (new) caching architecture, using a new CPU isn't possible,
        // as the cache would be newly created every iteration.
        // host ptr matrix
        let no_drop = f(&device.cpu, matrix);

        let dims = no_drop.dims();
        // convert host ptr / CPU matrix into a host ptr + OpenCL ptr matrix
        return unsafe { construct_buffer(device, no_drop.to_buf(), matrix.node.idx) }
            .map(|buf| (buf, dims).into());
    }

    let cpu = CPU::new();

    #[cfg(feature = "realloc")]
    if device.unified_mem() {
        return Ok(Matrix::from((device, f(&cpu, matrix))));
    }

    // convert an OpenCL buffer to a cpu buffer
    let cpu_buf: Matrix<T> = Matrix::from((&cpu, matrix.dims(), device.read(matrix)));
    let mat: Matrix<T> = f(&cpu, &cpu_buf);
    let mut convert = Matrix::from((device, mat));
    convert.node = device.graph().add(convert.len, matrix.node.idx);
    Ok(convert)
}

pub fn cpu_exec_mut<T, F>(device: &CLDevice, matrix: &mut Matrix<T>, f: F) -> custos::Result<()>
where
    F: Fn(&CPU, &mut Matrix<T>),
    T: Copy + Default,
{
    let cpu = CPU::new();

    // uses same memory as CPU
    if device.unified_mem() {
        return Ok(f(&cpu, matrix));
    }

    //convert an OpenCL buffer to a cpu matrix
    let mut cpu_matrix = Matrix::from((&cpu, matrix.dims(), device.read(matrix)));
    f(&cpu, &mut cpu_matrix);
    // write result as slice back to OpenCL Matrix
    device.write(matrix, &cpu_matrix);
    Ok(())
}

pub fn cpu_exec_lhs_rhs<'a, 'o, T, F>(
    device: &'o CLDevice,
    lhs: &Matrix<'a, T>,
    rhs: &Matrix<'a, T>,
    f: F,
) -> custos::Result<Matrix<'o, T>>
where
    F: for<'b> Fn(&'b CPU, &Matrix<T>, &Matrix<T>) -> Matrix<'b, T>,
    T: Copy + Default + Debug,
{
    let cpu = CPU::new();

    #[cfg(not(feature = "realloc"))]
    if device.unified_mem() {
        let no_drop = f(&device.cpu, lhs, rhs);

        let no_drop_dims = no_drop.dims();
        // convert host ptr / CPU matrix into a host ptr + OpenCL ptr matrix
        return unsafe { construct_buffer(device, no_drop.to_buf(), (lhs.node.idx, rhs.node.idx)) }
            .map(|buf| (buf, no_drop_dims).into());
    }

    #[cfg(feature = "realloc")]
    if device.unified_mem() {
        return Ok(Matrix::from((device, f(&cpu, lhs, rhs))));
    }

    // convert an OpenCL buffer to a cpu buffer
    let lhs = Matrix::from((&cpu, lhs.dims(), device.read(lhs)));
    let rhs = Matrix::from((&cpu, rhs.dims(), device.read(rhs)));

    let mut convert = Matrix::from((device, f(&cpu, &lhs, &rhs)));
    convert.node = device
        .graph()
        .add(convert.len, (lhs.node.idx, rhs.node.idx));

    Ok(convert)
}

pub fn cpu_exec_lhs_rhs_mut<T, F>(
    device: &CLDevice,
    lhs: &mut Matrix<T>,
    rhs: &Matrix<T>,
    f: F,
) -> custos::Result<()>
where
    F: Fn(&CPU, &mut Matrix<T>, &Matrix<T>),
    T: Copy + Default,
{
    let cpu = CPU::new();

    // uses same memory as CPU
    if device.unified_mem() {
        return Ok(f(&cpu, lhs, rhs));
    }

    //convert OpenCL matrix to cpu matrix
    let mut cpu_lhs = Matrix::from((&cpu, lhs.dims(), device.read(lhs)));
    let cpu_rhs = Matrix::from((&cpu, rhs.dims(), device.read(rhs)));
    f(&cpu, &mut cpu_lhs, &cpu_rhs);

    // write result as slice back to OpenCL Matrix
    device.write(lhs, &cpu_lhs);
    Ok(())
}

pub fn cpu_exec_scalar<T, F>(device: &CLDevice, matrix: &Matrix<T>, f: F) -> T
where
    F: Fn(&CPU, &Matrix<T>) -> T,
    T: Copy + Default,
{
    let cpu = CPU::new();
    if device.unified_mem() {
        return f(&cpu, matrix);
    }

    // convert an OpenCL buffer to a cpu buffer
    let cpu_buf = Matrix::from((&cpu, matrix.dims(), device.read(matrix)));

    f(&cpu, &cpu_buf)
}
