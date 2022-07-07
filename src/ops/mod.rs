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
    cpu::CPU,
    number::Number,
    opencl::CLDevice,
    Matrix,
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
    T: Copy+Default,
    F: Fn(&CPU, &Matrix<T>, &Matrix<T>) -> Matrix<T>,
>(device: &CLDevice, lhs: &Matrix<T>, rhs: &Matrix<T>, f: F) -> Matrix<T> 
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
    T: Copy+Default, F: 
    Fn(&CPU, Matrix<T>) -> Matrix<T>
>(device: &CLDevice, x: &Matrix<T>, f: F) -> Matrix<T> 
{
    use custos::opencl::cpu_exec;
    cpu_exec(device, x, f).unwrap()
    
    /*let cpu = custos::CPU::new();
    let x = Matrix::from((&cpu, x.dims(), custos::VecRead::read(device, x)));

    let result = f(&cpu, x);
    Matrix::from((device, result))*/
}

///OpenCL
fn cl_to_cpu_scalar<T: Number, F: Fn(&CPU, Matrix<T>) -> T>(
    device: &CLDevice,
    x: &Matrix<T>,
    f: F,
) -> T {
    use custos::opencl::cpu_exec_scalar;
    cpu_exec_scalar(device, x, f)
    
    /*let cpu = custos::CPU::new();
    let x = Matrix::from((&cpu, x.dims(), custos::VecRead::read(device, x)));
    f(&cpu, x)*/
}

#[cfg(feature="cuda")]
use custos::CudaDevice;

#[cfg(feature="cuda")]
pub fn cu_to_cpu_lr<
    T: Copy+Default,
    F: Fn(&CPU, &Matrix<T>, &Matrix<T>) -> Matrix<T>,
>(device: &CudaDevice, lhs: &Matrix<T>, rhs: &Matrix<T>, f: F) -> Matrix<T> 
{
    let cpu = custos::CPU::new();
    let lhs = Matrix::from((&cpu, lhs.dims(), custos::VecRead::read(device, lhs)));
    let rhs = Matrix::from((&cpu, rhs.dims(), custos::VecRead::read(device, rhs)));

    let result = f(&cpu, &lhs, &rhs);
    Matrix::from((device, result))
}

#[cfg(feature="cuda")]
pub fn cu_to_cpu_s<
    T: Copy+Default, F: 
    Fn(&CPU, Matrix<T>) -> Matrix<T>
>(device: &CudaDevice, x: &Matrix<T>, f: F) -> Matrix<T> 
{    
    let cpu = custos::CPU::new();
    let x = Matrix::from((&cpu, x.dims(), custos::VecRead::read(device, x)));

    let result = f(&cpu, x);
    Matrix::from((device, result))
}

#[cfg(feature="cuda")]
pub fn cu_to_cpu_scalar<T: Copy+Default, F: Fn(&CPU, Matrix<T>) -> T>(
    device: &CudaDevice,
    x: &Matrix<T>,
    f: F,
) -> T {
    let cpu = custos::CPU::new();
    let x = Matrix::from((&cpu, x.dims(), custos::VecRead::read(device, x)));
    f(&cpu, x)
}
