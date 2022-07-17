pub mod nn;

mod arithmetic;
mod assign;
mod clip;
mod col_op;
mod diagflat;
mod fns;
mod gemm;
mod max;
mod random;
mod row_op;
mod scalar;
mod sum;
mod transpose;

pub use arithmetic::*;
pub use assign::*;
pub use clip::*;
pub use col_op::*;
pub use diagflat::*;
pub use fns::*;
pub use gemm::*;
pub use max::*;
pub use random::*;
pub use row_op::*;
pub use scalar::*;
pub use sum::*;
pub use transpose::*;

#[cfg(any(feature = "opencl", feature = "cuda"))]
use crate::Matrix;
#[cfg(any(feature = "opencl", feature = "cuda"))]
use custos::cpu::CPU;

#[cfg(feature = "opencl")]
use custos::CLDevice;

#[cfg(feature = "opencl")]
///OpenCL
pub fn cl_to_cpu_lr<T: Copy + Default, F: Fn(&CPU, &Matrix<T>, &Matrix<T>) -> Matrix<T>>(
    device: &CLDevice,
    lhs: &Matrix<T>,
    rhs: &Matrix<T>,
    f: F,
) -> Matrix<T> {
    use crate::opencl::cpu_exec_lhs_rhs;
    cpu_exec_lhs_rhs(device, lhs, rhs, f).unwrap()
}

#[cfg(feature = "opencl")]
///OpenCL
pub fn cl_to_cpu_s<T: Copy + Default, F: Fn(&CPU, Matrix<T>) -> Matrix<T>>(
    device: &CLDevice,
    x: &Matrix<T>,
    f: F,
) -> Matrix<T> {
    use crate::opencl::cpu_exec;
    cpu_exec(device, x, f).unwrap()
}

#[cfg(feature = "opencl")]
///OpenCL
fn cl_to_cpu_scalar<T: Default + Copy, F: Fn(&CPU, Matrix<T>) -> T>(
    device: &CLDevice,
    x: &Matrix<T>,
    f: F,
) -> T {
    use crate::opencl::cpu_exec_scalar;
    cpu_exec_scalar(device, x, f)
}

#[cfg(feature = "cuda")]
use custos::CudaDevice;

#[cfg(feature = "cuda")]
pub fn cu_to_cpu_lr<T: Copy + Default, F: Fn(&CPU, &Matrix<T>, &Matrix<T>) -> Matrix<T>>(
    device: &CudaDevice,
    lhs: &Matrix<T>,
    rhs: &Matrix<T>,
    f: F,
) -> Matrix<T> {
    let cpu = custos::CPU::new();
    let lhs = Matrix::from((&cpu, lhs.dims(), custos::VecRead::read(device, lhs)));
    let rhs = Matrix::from((&cpu, rhs.dims(), custos::VecRead::read(device, rhs)));

    let result = f(&cpu, &lhs, &rhs);
    Matrix::from((device, result))
}

#[cfg(feature = "cuda")]
pub fn cu_to_cpu_s<T: Copy + Default, F: Fn(&CPU, Matrix<T>) -> Matrix<T>>(
    device: &CudaDevice,
    x: &Matrix<T>,
    f: F,
) -> Matrix<T> {
    let cpu = custos::CPU::new();
    let x = Matrix::from((&cpu, x.dims(), custos::VecRead::read(device, x)));

    let result = f(&cpu, x);
    Matrix::from((device, result))
}

#[cfg(feature = "cuda")]
pub fn cu_to_cpu_scalar<T: Copy + Default, F: Fn(&CPU, Matrix<T>) -> T>(
    device: &CudaDevice,
    x: &Matrix<T>,
    f: F,
) -> T {
    let cpu = custos::CPU::new();
    let x = Matrix::from((&cpu, x.dims(), custos::VecRead::read(device, x)));
    f(&cpu, x)
}
