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
mod random;

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
pub use random::*;

///OpenCL
pub fn switch_to_cpu_help_lr<
    T: GenericOCL,
    F: Fn(&InternCPU, &Matrix<T>, &Matrix<T>) -> Matrix<T>,
>(device: &InternCLDevice, lhs: &Matrix<T>, rhs: &Matrix<T>, f: F) -> Matrix<T> 
{
    use custos::opencl::cpu_exec_lhs_rhs;
    cpu_exec_lhs_rhs(device, lhs, rhs, f).unwrap()
    /*
    let cpu = CPU::new();
    let lhs = Matrix::from((&cpu, lhs.dims(), device.read(lhs.as_buf())));
    let rhs = Matrix::from((&cpu, rhs.dims(), device.read(rhs.as_buf())));

    let result = f(&cpu, &lhs, &rhs);
    Matrix::from((device, result))
    */
}

///OpenCL
pub fn switch_to_cpu_help_s<
    T: GenericOCL, F: 
    Fn(&InternCPU, Matrix<T>) -> Matrix<T>
>(device: &InternCLDevice, x: &Matrix<T>, f: F) -> Matrix<T> 
{
    use custos::opencl::cpu_exec;
    cpu_exec(device, x, f).unwrap()
    /*
    let cpu = CPU::new();
    let x = Matrix::from((&cpu, x.dims(), device.read(x.as_buf())));

    let result = f(&cpu, x);
    Matrix::from((device, result))*/
}

///OpenCL
fn switch_to_cpu_help_scalar<T: Number, F: Fn(&InternCPU, Matrix<T>) -> T>(
    device: &InternCLDevice,
    x: &Matrix<T>,
    f: F,
) -> T {
    use custos::opencl::cpu_exec_scalar;
    cpu_exec_scalar(device, x, f)
    /*
    let cpu = CPU::new();
    let x = Matrix::from((&cpu, x.dims(), device.read(x.as_buf())));
    f(&cpu, x)
    */
}

