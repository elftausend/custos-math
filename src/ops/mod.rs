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
    cpu::InternCPU,
    number::Number,
    opencl::InternCLDevice,
    CDatatype, Matrix,
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
pub fn cl_to_cpu_lr<
    T: CDatatype,
    F: Fn(&InternCPU, &Matrix<T>, &Matrix<T>) -> Matrix<T>,
>(device: &InternCLDevice, lhs: &Matrix<T>, rhs: &Matrix<T>, f: F) -> Matrix<T> 
{
    use custos::opencl::cpu_exec_lhs_rhs;
    cpu_exec_lhs_rhs(device, lhs, rhs, f).unwrap()
    /* 
    let cpu = custos::CPU::new();
    let lhs = Matrix::from((&cpu, lhs.dims(), custos::VecRead::read(device, lhs)));
    let rhs = Matrix::from((&cpu, rhs.dims(), custos::VecRead::read(device, rhs)));

    let result = f(&cpu, &lhs, &rhs);
    Matrix::from((device, result))*/
    
}

///OpenCL
pub fn cl_to_cpu_s<
    T: CDatatype, F: 
    Fn(&InternCPU, Matrix<T>) -> Matrix<T>
>(device: &InternCLDevice, x: &Matrix<T>, f: F) -> Matrix<T> 
{
    use custos::opencl::cpu_exec;
    cpu_exec(device, x, f).unwrap()
    
    /*let cpu = custos::CPU::new();
    let x = Matrix::from((&cpu, x.dims(), custos::VecRead::read(device, x)));

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
    
    /*let cpu = custos::CPU::new();
    let x = Matrix::from((&cpu, x.dims(), custos::VecRead::read(device, x)));
    f(&cpu, x)*/
    
}

