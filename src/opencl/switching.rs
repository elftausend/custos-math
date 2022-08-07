use std::fmt::Debug;
use custos::{CLDevice, VecRead, WriteBuf, CPU};
use crate::Matrix;

#[cfg(not(feature="realloc"))]
use custos::opencl::construct_buffer;

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
    #[cfg(not(feature="realloc"))]
    if device.unified_mem() {
        // Using the global cpu in order to get a (correct) cache entry.
        // Due to the (new) caching architecture, using a local cache isn't possible, 
        // as the cache is newly created every iteration.
        return custos::GLOBAL_CPU.with(|cpu| {
            // host ptr matrix
            let no_drop = f(cpu, matrix);

            let dims = no_drop.dims();
            // convert host ptr / CPU matrix into a host ptr + OpenCL ptr matrix
            unsafe { construct_buffer(device, no_drop.to_buf())}.map(|buf| (buf, dims).into())
        });
    }

    let cpu = CPU::new();

    #[cfg(feature="realloc")]
    if device.unified_mem() {
        return Ok(Matrix::from((device, f(&cpu, matrix))))
    }
    

    // convert an OpenCL buffer to a cpu buffer
    let cpu_buf: Matrix<T> = Matrix::from((&cpu, matrix.dims(), device.read(matrix)));
    let mat: Matrix<T> = f(&cpu, &cpu_buf);
    let convert = Matrix::from((device, mat));
    
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

    #[cfg(not(feature="realloc"))]
    if device.unified_mem() {
        return custos::GLOBAL_CPU.with(|cpu| {
            let no_drop = f(cpu, lhs, rhs);

            let no_drop_dims = no_drop.dims();
            // convert host ptr / CPU matrix into a host ptr + OpenCL ptr matrix
            unsafe { construct_buffer(device, no_drop.to_buf()) }.map(|buf| (buf, no_drop_dims).into())
        }); 
    }

    #[cfg(feature="realloc")]
    if device.unified_mem() {
        return Ok(Matrix::from((device, f(&cpu, lhs, rhs))))
    }
    
    // convert an OpenCL buffer to a cpu buffer
    let lhs = Matrix::from((&cpu, lhs.dims(), device.read(lhs)));
    let rhs = Matrix::from((&cpu, rhs.dims(), device.read(rhs)));

    Ok(Matrix::from((device, f(&cpu, &lhs, &rhs))))
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
