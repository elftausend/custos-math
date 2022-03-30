use custos::{Matrix, cpu::{InternCPU, each_op}, number::Float, opencl::{GenericOCL, InternCLDevice}};

use crate::opencl::str_op;

pub trait FnsOps<T> {
    fn exp(&self, x: Matrix<T>) -> Matrix<T>;
    fn ln(&self, x: Matrix<T>) -> Matrix<T>;
    fn neg(&self, x: Matrix<T>) -> Matrix<T>;
}

impl <T: Float>FnsOps<T> for InternCPU {
    fn exp(&self, x: Matrix<T>) -> Matrix<T> {
        each_op(self, x, |x| x.exp())
    }

    fn ln(&self, x: Matrix<T>) -> Matrix<T> {
        each_op(self, x, |x| x.ln())
    }

    fn neg(&self, x: Matrix<T>) -> Matrix<T> {
        each_op(self, x, |x| x.negate())
    }
}

impl <T: GenericOCL>FnsOps<T> for InternCLDevice {
    fn exp(&self, x: Matrix<T>) -> Matrix<T> {
        str_op(self.clone(), x, "exp(I)").unwrap()
    }

    fn ln(&self, x: Matrix<T>) -> Matrix<T> {
        str_op(self.clone(), x, "ln(I)").unwrap()
    }

    fn neg(&self, x: Matrix<T>) -> Matrix<T> {
        str_op(self.clone(), x, "-I").unwrap()
    }
}