use custos::{Matrix, opencl::{InternCLDevice, GenericOCL}, cpu::{InternCPU, CPUCache}};

use crate::{opencl::scalar_r_op, cpu::scalar_apply};

pub trait AdditionalOps<T> {
    fn adds(&self, lhs: Matrix<T>, rhs: T) -> Matrix<T>;
}


impl <T: GenericOCL>AdditionalOps<T> for InternCLDevice {
    fn adds(&self, lhs: Matrix<T>, rhs: T) -> Matrix<T> {
        scalar_r_op(self.clone(), lhs, rhs, "+").unwrap()
    }
}

impl <T: GenericOCL>AdditionalOps<T> for InternCPU {
    fn adds(&self, lhs: Matrix<T>, rhs: T) -> Matrix<T> {
        scalar_apply(self, lhs, rhs, |c, a, b| *c = a+b)
    }
}