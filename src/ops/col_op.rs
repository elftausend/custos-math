use custos::{number::Number, CDatatype, InternCLDevice, InternCPU, Matrix};

use crate::cpu::col_op;

use super::cl_to_cpu_lr;

pub trait ColOp<T> {
    fn add_col(&self, lhs: &Matrix<T>, rhs: &Matrix<T>) -> Matrix<T>;
    fn sub_col(&self, lhs: &Matrix<T>, rhs: &Matrix<T>) -> Matrix<T>;
    fn div_col(&self, lhs: &Matrix<T>, rhs: &Matrix<T>) -> Matrix<T>;
}

impl<T: Number> ColOp<T> for InternCPU {
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

impl<T: CDatatype> ColOp<T> for InternCLDevice {
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
