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

#[cfg(feature = "cpu")]
impl<T: Number, D: MainMemory> ColOp<T, D> for CPU {
    #[inline]
    fn add_col(&self, lhs: &Matrix<T, D>, rhs: &Matrix<T, D>) -> Matrix<T> {
        col_op(self, lhs, rhs, |c, a, b| *c = a + b)
    }

    #[inline]
    fn sub_col(&self, lhs: &Matrix<T, D>, rhs: &Matrix<T, D>) -> Matrix<T> {
        col_op(self, lhs, rhs, |c, a, b| *c = a - b)
    }

    #[inline]
    fn div_col(&self, lhs: &Matrix<T, D>, rhs: &Matrix<T, D>) -> Matrix<T> {
        col_op(self, lhs, rhs, |c, a, b| *c = a / b)
    }
}

#[cfg(feature = "opencl")]
impl<T: custos::CDatatype + Number> ColOp<T> for OpenCL {
    #[inline]
    fn add_col(&self, lhs: &Matrix<T, Self>, rhs: &Matrix<T, Self>) -> Matrix<T, Self> {
        cl_to_cpu_lr(self, lhs, rhs, |device, lhs, rhs| device.add_col(lhs, rhs))
    }

    #[inline]
    fn sub_col(&self, lhs: &Matrix<T, Self>, rhs: &Matrix<T, Self>) -> Matrix<T, Self> {
        cl_to_cpu_lr(self, lhs, rhs, |device, lhs, rhs| device.sub_col(lhs, rhs))
    }

    #[inline]
    fn div_col(&self, lhs: &Matrix<T, Self>, rhs: &Matrix<T, Self>) -> Matrix<T, Self> {
        cl_to_cpu_lr(self, lhs, rhs, |device, lhs, rhs| device.div_col(lhs, rhs))
    }
}
#[cfg(feature = "cuda")]
use crate::cu_to_cpu_lr;
#[cfg(feature = "cuda")]
use custos::CUDA;

#[cfg(feature = "cuda")]
impl<T: custos::CDatatype + Number> ColOp<T> for CUDA {
    #[inline]
    fn add_col(&self, lhs: &Matrix<T, Self>, rhs: &Matrix<T, Self>) -> Matrix<T, Self> {
        cu_to_cpu_lr(self, lhs, rhs, |device, lhs, rhs| device.add_col(lhs, rhs))
    }

    #[inline]
    fn sub_col(&self, lhs: &Matrix<T, Self>, rhs: &Matrix<T, Self>) -> Matrix<T, Self> {
        cu_to_cpu_lr(self, lhs, rhs, |device, lhs, rhs| device.sub_col(lhs, rhs))
    }

    #[inline]
    fn div_col(&self, lhs: &Matrix<T, Self>, rhs: &Matrix<T, Self>) -> Matrix<T, Self> {
        cu_to_cpu_lr(self, lhs, rhs, |device, lhs, rhs| device.div_col(lhs, rhs))
    }
}
