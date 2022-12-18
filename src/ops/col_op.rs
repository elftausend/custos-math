use crate::{cpu::col_op, Matrix};
use custos::{number::Number, Device, MainMemory, CPU};

#[cfg(feature = "opencl")]
use super::cl_to_cpu_lr;
#[cfg(feature = "opencl")]
use custos::OpenCL;

pub trait ColOp<T, D: Device = Self>: Device {
    fn add_col(&self, lhs: &Matrix<T, D>, rhs: &Matrix<T, D>) -> Matrix<T, Self>;
    fn sub_col(&self, lhs: &Matrix<T, D>, rhs: &Matrix<T, D>) -> Matrix<T, Self>;
    fn div_col(&self, lhs: &Matrix<T, D>, rhs: &Matrix<T, D>) -> Matrix<T, Self>;
}

impl<T: Number, D: MainMemory> ColOp<T, D> for CPU {
    fn add_col(&self, lhs: &Matrix<T, D>, rhs: &Matrix<T, D>) -> Matrix<T> {
        col_op(self, lhs, rhs, |c, a, b| *c = a + b)
    }

    fn sub_col(&self, lhs: &Matrix<T, D>, rhs: &Matrix<T, D>) -> Matrix<T> {
        col_op(self, lhs, rhs, |c, a, b| *c = a - b)
    }

    fn div_col(&self, lhs: &Matrix<T, D>, rhs: &Matrix<T, D>) -> Matrix<T> {
        col_op(self, lhs, rhs, |c, a, b| *c = a / b)
    }
}

#[cfg(feature = "opencl")]
impl<T: custos::CDatatype> ColOp<T> for OpenCL {
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
#[cfg(feature = "cuda")]
use crate::cu_to_cpu_lr;
#[cfg(feature = "cuda")]
use custos::CudaDevice;

#[cfg(feature = "cuda")]
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
