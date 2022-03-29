use custos::{Matrix, cpu::{InternCPU, CPU}, number::Number, opencl::{InternCLDevice, GenericOCL}, VecRead};

use crate::cpu::row_op;


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
        let device = CPU::new();
        let lhs = Matrix::from((&device, lhs.dims(), self.read(lhs.data())));
        let rhs = Matrix::from((&device, rhs.dims(), self.read(rhs.data())));
        
        let added = device.add_row(lhs, rhs);
        Matrix::from( (self, added) )
    }
}