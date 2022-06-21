use crate::{cpu::row_op, Mat};
use custos::{
    cpu::CPU, get_device, number::Number, opencl::CLDevice, CDatatype, Matrix,
};

use super::cl_to_cpu_lr;

pub trait Row<T, R: Mat<T>> {
    fn add_row(self, rhs: R) -> Matrix<T>;
}

impl<T: CDatatype, L: Mat<T>, R: Mat<T>> Row<T, R> for L {
    fn add_row(self, rhs: R) -> Matrix<T> {
        let device = get_device!(RowOp, T).unwrap();
        device.add_row(self.as_mat(), rhs.as_mat())
    }
}

pub trait RowOp<T> {
    fn add_row(&self, lhs: &Matrix<T>, rhs: &Matrix<T>) -> Matrix<T>;
}

impl<T: Number> RowOp<T> for CPU {
    fn add_row(&self, lhs: &Matrix<T>, rhs: &Matrix<T>) -> Matrix<T> {
        row_op(self, lhs, rhs, |c, a, b| *c = a + b)
    }
}

impl<T: CDatatype> RowOp<T> for CLDevice {
    fn add_row(&self, lhs: &Matrix<T>, rhs: &Matrix<T>) -> Matrix<T> {
        cl_to_cpu_lr(self, lhs, rhs, |device, lhs, rhs| device.add_row(lhs, rhs))
    }
}
