use custos::{Matrix, cpu::InternCPU, number::Number, opencl::{InternCLDevice, GenericOCL}, get_device};
use crate::cpu::row_op;

use super::switch_to_cpu_help_lr;

pub trait Row<T> {
    fn add_row(self, rhs: Matrix<T>) -> Matrix<T>;
}

impl <T: GenericOCL>Row<T> for Matrix<T> {
    fn add_row(self, rhs: Matrix<T>) -> Matrix<T> {
        let device = get_device!(RowOp, T).unwrap();
        device.add_row(self, rhs)
    }
}

pub trait RowOp<T> {
    fn add_row(&self, lhs: Matrix<T>, rhs: Matrix<T>) -> Matrix<T>;
}

impl <T: Number>RowOp<T> for InternCPU {
    fn add_row(&self, lhs: Matrix<T>, rhs: Matrix<T>) -> Matrix<T> {
        row_op(self, lhs, rhs, |c, a, b| *c = a + b)
    }
}

impl <T: GenericOCL>RowOp<T> for InternCLDevice {
    fn add_row(&self, lhs: Matrix<T>, rhs: Matrix<T>) -> Matrix<T> {
        switch_to_cpu_help_lr(self, lhs, rhs, |device, lhs, rhs| device.add_row(lhs, rhs))
    }
}