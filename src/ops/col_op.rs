use custos::{Matrix, InternCPU, number::Number, opencl::GenericOCL, InternCLDevice};

use crate::cpu::col_op;

use super::switch_to_cpu_help_lr;

pub trait ColOp<T> {
    fn add_col(&self, lhs: Matrix<T>, rhs: Matrix<T>) -> Matrix<T>;
}

impl <T: Number>ColOp<T> for InternCPU {
    fn add_col(&self, lhs: Matrix<T>, rhs: Matrix<T>) -> Matrix<T> {
        col_op(self, lhs, rhs, |c, a, b| *c = a + b)
    }
}

impl <T: GenericOCL>ColOp<T> for InternCLDevice {
    fn add_col(&self, lhs: Matrix<T>, rhs: Matrix<T>) -> Matrix<T> {
        switch_to_cpu_help_lr(self, lhs, rhs, |device, lhs, rhs| device.add_col(lhs, rhs))
        
    }
}
