pub mod nn;

mod clip;
mod col_op;
mod diagflat;
mod fns;
mod max;
mod row_op;
mod scalar;
mod sum;
mod transpose;

pub use clip::*;
pub use col_op::*;
use custos::{
    cpu::{InternCPU, CPU},
    number::Number,
    opencl::InternCLDevice,
    GenericOCL, Matrix, VecRead,
};
pub use diagflat::*;
pub use fns::*;
pub use max::*;
pub use row_op::*;
pub use scalar::*;
pub use sum::*;
pub use transpose::*;

///OpenCL
pub fn switch_to_cpu_help_lr<
    T: GenericOCL,
    F: Fn(&InternCPU, &Matrix<T>, &Matrix<T>) -> Matrix<T>,
>(
    device: &InternCLDevice,
    lhs: &Matrix<T>,
    rhs: &Matrix<T>,
    f: F,
) -> Matrix<T> {
    let cpu = CPU::new();
    let lhs = Matrix::from((&cpu, lhs.dims(), device.read(lhs.data())));
    let rhs = Matrix::from((&cpu, rhs.dims(), device.read(rhs.data())));

    let result = f(&cpu, &lhs, &rhs);
    Matrix::from((device, result))
}

///OpenCL
pub fn switch_to_cpu_help_s<T: GenericOCL, F: Fn(&InternCPU, Matrix<T>) -> Matrix<T>>(
    device: &InternCLDevice,
    x: &Matrix<T>,
    f: F,
) -> Matrix<T> {
    let cpu = CPU::new();
    let x = Matrix::from((&cpu, x.dims(), device.read(x.data())));

    let result = f(&cpu, x);
    Matrix::from((device, result))
}

///OpenCL
fn switch_to_cpu_help_scalar<T: Number, F: Fn(&InternCPU, Matrix<T>) -> T>(
    device: &InternCLDevice,
    x: &Matrix<T>,
    f: F,
) -> T {
    let cpu = CPU::new();
    let x = Matrix::from((&cpu, x.dims(), device.read(x.data())));
    f(&cpu, x)
}
