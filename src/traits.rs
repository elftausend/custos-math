use custos::{Matrix, libs::{opencl::{GenericOCL, cl_device::InternCLDevice}, cpu::{each_op, InternCPU}}, number::Float};

use crate::opencl::str_op;

pub trait ActivationOps<T> {
    fn sigmoid(&self, x: Matrix<T>) -> Matrix<T>;   
}

impl <T: GenericOCL+Float>ActivationOps<T> for InternCLDevice {
    fn sigmoid(&self, x: Matrix<T>) -> Matrix<T> {
        str_op(self.clone(), x, "1.0 / (1.0 + exp(-I))").unwrap()
    }
}

impl <T: Float>ActivationOps<T> for InternCPU {
    fn sigmoid(&self, x: Matrix<T>) -> Matrix<T> {
        each_op(self, x, |x| T::one() / (T::one() + x.negate().exp()))
    }
}