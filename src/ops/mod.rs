mod activations;
mod scalar;
mod row_op;
mod diagflat;
mod transpose;
mod clip;

pub use activations::*;
use custos::{opencl::{GenericOCL, InternCLDevice}, cpu::{InternCPU, CPU}, Matrix, VecRead};
pub use scalar::*;
pub use row_op::*;
pub use diagflat::*;
pub use transpose::*;
pub use clip::*;

///OpenCL
fn switch_to_cpu_help_lr<T: GenericOCL, F: Fn(&InternCPU, Matrix<T>, Matrix<T>) -> Matrix<T>>(device: &InternCLDevice, lhs: Matrix<T>, rhs: Matrix<T>, f: F) -> Matrix<T> {
    let cpu = CPU::new();
    let lhs = Matrix::from((&cpu, lhs.dims(), device.read(lhs.data())));
    let rhs = Matrix::from((&cpu, rhs.dims(), device.read(rhs.data())));

    let result = f(&cpu, lhs, rhs);
    Matrix::from( (device, result) )
}

///OpenCL
fn switch_to_cpu_help_s<T: GenericOCL, F: Fn(&InternCPU, Matrix<T>) -> Matrix<T>>(device: &InternCLDevice, x: Matrix<T>, f: F) -> Matrix<T> {
    let cpu = CPU::new();
    let x = Matrix::from((&cpu, x.dims(), device.read(x.data())));
    
    let result = f(&cpu, x);
    Matrix::from( (device, result) )
}