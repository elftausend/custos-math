pub mod nn;

mod arithmetic;
mod assign;
mod clip;
mod col_op;
mod diagflat;
mod fns;
mod gemm;
mod max;
mod row_op;
mod scalar;
mod scalar_assign;
mod sum;
mod transpose;

#[cfg(feature = "fastrand")]
mod random;

pub use arithmetic::*;
pub use assign::*;
pub use clip::*;
pub use col_op::*;
pub use diagflat::*;
pub use fns::*;
pub use gemm::*;
pub use max::*;
pub use row_op::*;
pub use scalar::*;
pub use scalar_assign::*;
pub use sum::*;
pub use transpose::*;

#[cfg(feature = "fastrand")]
pub use random::*;

#[cfg(feature = "opencl")]
use crate::Matrix;
#[cfg(feature = "opencl")]
use custos::cpu::CPU;

#[cfg(feature = "opencl")]
use custos::OpenCL;

#[cfg(feature = "opencl")]
///OpenCL
pub fn cl_to_cpu_lr<'a, 'o, T, F>(
    device: &'a OpenCL,
    lhs: &Matrix<T, OpenCL>,
    rhs: &Matrix<T, OpenCL>,
    f: F,
) -> Matrix<'a, T, OpenCL>
where
    T: Copy + Default + std::fmt::Debug,
    F: for<'b> Fn(&'b CPU, &Matrix<T>, &Matrix<T>) -> Matrix<'b, T>,
{
    use crate::opencl::cpu_exec_lhs_rhs;
    cpu_exec_lhs_rhs(device, lhs, rhs, f).unwrap()
}

#[cfg(feature = "opencl")]
///OpenCL
pub fn cl_to_cpu_s<'a, 'o, T, F>(
    device: &'o OpenCL,
    x: &Matrix<'a, T, OpenCL>,
    f: F,
) -> Matrix<'o, T, OpenCL>
where
    T: Copy + Default + std::fmt::Debug,
    F: for<'b> Fn(&'b CPU, &Matrix<'_, T>) -> Matrix<'b, T>,
{
    use crate::opencl::cpu_exec;
    cpu_exec(device, x, &f).unwrap()
}

#[cfg(feature = "opencl")]
///OpenCL
fn cl_to_cpu_scalar<T: Default + Copy, F: Fn(&CPU, &Matrix<T>) -> T>(
    device: &OpenCL,
    x: &Matrix<T, OpenCL>,
    f: F,
) -> T {
    use crate::opencl::cpu_exec_scalar;
    cpu_exec_scalar(device, x, f)
}
