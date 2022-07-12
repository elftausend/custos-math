use custos::{number::Number, CPU};
use crate::{cpu::col_op, Matrix};

#[cfg(feature="opencl")]
use super::cl_to_cpu_lr;
#[cfg(feature="opencl")]
use custos::CLDevice;

pub trait ColOp<T> {
    fn add_col(&self, lhs: &Matrix<T>, rhs: &Matrix<T>) -> Matrix<T>;
    fn sub_col(&self, lhs: &Matrix<T>, rhs: &Matrix<T>) -> Matrix<T>;
    fn div_col(&self, lhs: &Matrix<T>, rhs: &Matrix<T>) -> Matrix<T>;
}

impl<T: Number> ColOp<T> for CPU {
    fn add_col(&self, lhs: &Matrix<T>, rhs: &Matrix<T>) -> Matrix<T> {
        col_op(self, lhs, rhs, |c, a, b| *c = a + b)
    }

    fn sub_col(&self, lhs: &Matrix<T>, rhs: &Matrix<T>) -> Matrix<T> {
        col_op(self, lhs, rhs, |c, a, b| *c = a - b)
    }

    fn div_col(&self, lhs: &Matrix<T>, rhs: &Matrix<T>) -> Matrix<T> {
        col_op(self, lhs, rhs, |c, a, b| *c = a / b)
    }
}

#[cfg(feature="opencl")]
impl<T: custos::CDatatype> ColOp<T> for CLDevice {
    fn add_col(&self, lhs: &Matrix<T>, rhs: &Matrix<T>) -> Matrix<T> {
        cl_to_cpu_lr(self, lhs, rhs, |device, lhs, rhs| device.add_col(lhs, rhs))
    }

    fn sub_col(&self, lhs: &Matrix<T>, rhs: &Matrix<T>) -> Matrix<T> {
        cl_to_cpu_lr(self, lhs, rhs, |device, lhs, rhs| device.sub_col(lhs, rhs))
    }

    fn div_col(&self, lhs: &Matrix<T>, rhs: &Matrix<T>) -> Matrix<T> {
        cl_to_cpu_lr(self, lhs, rhs, |device, lhs, rhs| device.div_col(lhs, rhs))
    }
}
#[cfg(feature="cuda")]
use custos::CudaDevice;
#[cfg(feature="cuda")]
use super::cu_to_cpu_lr;

#[cfg(feature="cuda")]
impl<T: custos::CDatatype> ColOp<T> for CudaDevice {
    fn add_col(&self, lhs: &Matrix<T>, rhs: &Matrix<T>) -> Matrix<T> {
        cu_to_cpu_lr(self, lhs, rhs, |device, lhs, rhs| device.add_col(lhs, rhs))
    }

    fn sub_col(&self, lhs: &Matrix<T>, rhs: &Matrix<T>) -> Matrix<T> {
        cu_to_cpu_lr(self, lhs, rhs, |device, lhs, rhs| device.sub_col(lhs, rhs))
    }

    fn div_col(&self, lhs: &Matrix<T>, rhs: &Matrix<T>) -> Matrix<T> {
        cu_to_cpu_lr(self, lhs, rhs, |device, lhs, rhs| device.div_col(lhs, rhs))
    }
}